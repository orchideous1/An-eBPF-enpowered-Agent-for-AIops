import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent


PROJECT_ROOT = get_project_root()

LLM_DEFAULT_CONFIG = {
    "DEFAULT_MODEL": os.getenv("OPENAI_MODEL", "qwen3-coder-plus"), # "qwen2.5-coder-3b-instruct"
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL")
}

LLM_STEP_CONFIG = {
    "TEMPERATURE": 0.2,
    "MAX_TOKENS": 4096,
}

SystemPrompts = {
    "default": "You are a helpful assistant.",
    "code_assistant": "You are a coding assistant specialized in Python.",
    "debug_assistant": "You are an expert in debugging code.",
    "Ops expert": """
        你是一个性能分析专家，熟悉计算机系统基本原理，能够熟练调用命令行工具；

        对于用户的问题，你可以根据需要调用注册的工具/函数来获取信息或执行操作，直到你有足够的信息来回答用户的问题为止。
        在此期间，如果你需要进一步与用户进行交互，你需要执行"ask_user"工具来与用户进行交互，以获取详细信息或向用户确认需求
        你会根据上下文和工具调用的结果，逐步推理并形成最终答案。

        这里还有几条规则你需要遵守：
        - 请不要模拟工具执行结果，必须通过工具与真实场景交互。
        - 在涉及到工具调用时，你的思维链遵循是否需要调用工具，调用什么工具，如何调用工具三步，确保格式满足工具定义。
        - 工具中还包含辅助工具，如工具说明工具，终止对话工具等，请你判断何时需要调用。
        """,
}

SYSTEM_PROMPT = "You are an agent that can execute tool calls"

NEXT_STEP_PROMPT = (
    "是否需要与用户交互？目前是否需要调用工具，调用什么工具，如何调用工具？"
)
