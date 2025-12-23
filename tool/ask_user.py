from SREgent.tool.base import BaseTool


class AskUser(BaseTool):
    name: str = "ask_user"
    description: str = "Ask the user a question or request confirmation."
    parameters: dict = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user."
            }
        },
        "required": ["question"]
    }

    async def execute(self, question: str) -> str:
        return f"User was asked: {question}"

