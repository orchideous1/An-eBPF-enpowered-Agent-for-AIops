from typing import List

from pydantic import Field

from SREgent.agents.toolcall import ToolCallAgent
from SREgent.config import SystemPrompts
from SREgent.tool import Bash, Terminate, ToolCollection, AskUser# ,  StrReplaceEditor,


class SWEAgent(ToolCallAgent):
    """An agent that implements the SWEAgent paradigm for executing code and natural conversations."""

    name: str = "swe"
    description: str = "an autonomous AI programmer that interacts directly with the computer to solve tasks."

    system_prompt: str = SystemPrompts["Ops expert"]
    next_step_prompt: str = ""

    available_tools: ToolCollection = ToolCollection(
        Bash(), Terminate(), AskUser()#  StrReplaceEditor()
    )
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    max_steps: int = 10
