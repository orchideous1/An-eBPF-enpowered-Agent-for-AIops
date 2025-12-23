import asyncio
import json
from typing import Any, List, Optional, Union, AsyncGenerator, Dict

from pydantic import Field

from SREgent.agents.react import ReActAgent
from SREgent.logger import logger
from SREgent.config import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from SREgent.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from SREgent.tool import CreateChatCompletion, Terminate, ToolCollection, AskUser


TOOL_CALL_REQUIRED = "Tool calls required but none provided"



class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate(), AskUser()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    self.results = None
    # _current_base64_image: Optional[str] = None

    # max_steps: int = 30

    @staticmethod
    def _clean_args(raw_args: Optional[str]) -> str:
        """Clean Markdown code blocks from arguments string"""
        if not raw_args:
            return "{}"
        raw_args = raw_args.strip()
        if raw_args.startswith("```"):
            # Remove starting ```json or ```
            parts = raw_args.split("\n", 1)
            if len(parts) > 1:
                raw_args = parts[1]
            # Remove ending ```
            if raw_args.strip().endswith("```"):
                raw_args = raw_args.strip().rsplit("\n", 1)[0]
        return raw_args

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            # if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
            #     token_limit_error = e.__cause__Token
            #     logger.error(
            #         f"🚨 Token limit error (from RetryError): {token_limit_error}"
            #     )
            #     self.memory.add_message(
            #         Message.assistant_message(
            #             f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
            #         )
            #     )
            #     self.state = AgentState.FINISHED
            #     return False
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        if len(tool_calls) > 1:
            self.memory.add_message(
                Message.assistant_message(
                    "请你一次只调用一个工具"
                )
            )
            return False
        content = response.content if response and response.content else ""
        self.results = {'reasoning' : content}
        self.results.update({'tool_use' : [call.function.name for call in tool_calls]})

        # Log response info
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            # Parse arguments
            raw_args = tool_calls[0].function.arguments or "{}"
            
            # Use generic cleaning method
            raw_args = self._clean_args(raw_args)
            self.tool_calls[0].function.arguments = raw_args
            logger.info(f"🔧 Tool arguments: {raw_args}")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> Dict:
        """Execute tool calls and handle their results"""

        for command in self.tool_calls:
            # Reset base64_image for each tool call
            # self._current_base64_image = None

            result = await self.execute_tool(command)
            self.results.update({'result' : result})


            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                # base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)

        return self.results

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name

        # Clean arguments before parsing
        cleaned_args = self._clean_args(command.function.arguments)

        if name == "ask_user":
            try:
                args = json.loads(cleaned_args)
                question = args.get("question", "")
                logger.info(f"❓ Agent is asking user: {question}")
                self.state = AgentState.AWAITING_INPUT
                return f"ASK_USER_WAITING: {question}"
            except Exception as e:
                logger.error(f"Error parsing ask_user arguments: {e}")
                return "Error: Invalid arguments for ask_user"

        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            args = json.loads(cleaned_args)

            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # Handle special tools
            await self._handle_special_tool(name=name, result=result)

            # # Check if result is a ToolResult with base64_image
            # if hasattr(result, "base64_image") and result.base64_image:
            #     # Store the base64_image for later use in tool_message
            #     self._current_base64_image = result.base64_image

            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """Clean up resources used by the agent's tools."""
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None) -> AsyncGenerator[str, str]:
        """Run the agent with cleanup when done."""
        try:
            async for output in super().run(request):
                yield output
        finally:
            await self.cleanup()
