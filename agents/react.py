from abc import ABC, abstractmethod
from typing import Optional, Dict

from pydantic import Field

from SREgent.agents.base import BaseAgent
from SREgent.llm import LLM
from SREgent.schema import AgentState, Memory


class ReActAgent(BaseAgent, ABC):
    name: str
    description: Optional[str] = None

    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None

    llm: Optional[LLM] = Field(default_factory=LLM)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE
    results: Dict = Field(default_factory=dict)

    max_steps: int = 10
    current_step: int = 0

    @abstractmethod
    async def think(self) -> bool:
        """Process current state and decide next action"""

    @abstractmethod
    async def act(self) -> Dict:
        """Execute decided actions"""

    async def step(self) -> str:
        """Execute a single step: think and act."""
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        self.results = await self.act()
        return await self.answer()

    @abstractmethod
    async def answer(self) -> str:
        """ Formulate format answer to user"""
