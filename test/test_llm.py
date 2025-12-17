import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加上级目录到模块搜索路径
from dotenv import load_dotenv
from llm import LLM
from schema import Message

# Load environment variables
load_dotenv()

async def test_ask():
    print("=== Testing ask() ===")
    try:
        llm = LLM()
        messages = [
            Message.user_message("Hello! Please reply with 'pong'.")
        ]
        
        # 注意：根据当前的 llm.py 定义，ask 需要 max_tokens 和 temperature 参数
        response = await llm.ask(
            messages=messages
        )
        print(f"Response: {response}")
        assert len(response) > 0
        print("✅ ask() test passed")
    except Exception as e:
        print(f"❌ ask() test failed: {e}")

async def test_ask_tool():
    print("\n=== Testing ask_tool() ===")
    try:
        llm = LLM()
        
        # 定义一个简单的工具
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        
        messages = [
            Message.user_message("What's the weather like in Shanghai?")
        ]
        
        response = await llm.ask_tool(
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        if response.tool_calls:
            print(f"Tool calls detected: {len(response.tool_calls)}")
            tool_call = response.tool_calls[0]
            print(f"Function Name: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")
            assert tool_call.function.name == "get_current_weather"
            print("✅ ask_tool() test passed")
        else:
            print(f"Response content: {response.content}")
            print("⚠️ No tool calls generated (this might be expected depending on the model)")
            
    except Exception as e:
        print(f"❌ ask_tool() test failed: {e}")

async def main():
    await test_ask()
    await test_ask_tool()

if __name__ == "__main__":
    asyncio.run(main())