import argparse
from agents import BaseAgent
from tool.tool import register_default_tools
from dotenv import load_dotenv
load_dotenv()

def run_base(stream: bool, use_tools: bool, use_rag: bool, max_tool_steps: int, verbose: bool):
    agent = BaseAgent()
    register_default_tools(agent, include_categories=["file", "command"])
    print("Interactive BaseAgent. Type 'exit' to quit.")
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user.lower() in ("exit", "quit"):
            print("Bye.")
            break

        # 流式目前不执行工具调用；若需要工具调用请关闭流式
        if stream and not use_tools:
            print("Assistant> ", end="", flush=True)
            for chunk in agent.get_streaming_response(user, use_rag=use_rag):
                print(chunk, end="", flush=True)
            print()
        else:
            resp = agent.get_response(user, use_tools=use_tools, use_rag=use_rag)
            print(f"Assistant> {resp}")


def main():
    parser = argparse.ArgumentParser(description="SREgent interactive chat")
    parser.add_argument("--stream", action="store_true", help="Stream tokens (BaseAgent only, no tool calls)")
    parser.add_argument("--use-tools", action="store_true", default=True, help="Disable tool calls (BaseAgent)")
    parser.add_argument("--use-rag", action="store_true", default=False, help="Disable RAG")
    parser.add_argument("--max-tool_steps", type=int, default=10, help="ReActAgent max iterations")
    parser.add_argument("--verbose", action="store_true", help="Verbose ReAct loop")
    args = parser.parse_args()

    run_base(stream=args.stream, use_tools=args.use_tools, use_rag=args.use_rag, max_tool_steps=args.max_tool_steps, verbose=args.verbose)

if __name__ == "__main__":
    # 可选：提前设置 sudo 密码用于命令工具
    # os.environ.setdefault("SUDO_PASSWORD", "your_password")
    main()
