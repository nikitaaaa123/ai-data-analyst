"""One user session = workspace + agent + turn history."""
from __future__ import annotations

from .agent import DataAnalystAgent
from .config import Settings
from .demo_llm import DemoLLM
from .llm import AnthropicLLM, LLMClient
from .models import AgentTurn
from .workspace import Workspace


class AnalystSession:
    def __init__(self, settings: Settings | None = None, api_key: str | None = None):
        self.settings = settings or Settings.from_env()
        self.workspace = Workspace(self.settings)
        self.turns: list[AgentTurn] = []
        self.agent = DataAnalystAgent(self._make_llm(api_key or self.settings.anthropic_api_key),
                                      self.workspace, self.settings)

    def _make_llm(self, api_key: str | None) -> LLMClient:
        if api_key:
            return AnthropicLLM(api_key, self.settings.model, self.settings.max_tokens)
        return DemoLLM(self.workspace)

    @property
    def mode(self) -> str:
        return self.agent.llm.name

    def set_api_key(self, api_key: str | None) -> None:
        """Switch between live model and offline demo; conversation memory is reset."""
        self.agent.llm = self._make_llm(api_key or None)
        self.agent.reset()

    def ask(self, question: str) -> AgentTurn:
        turn = self.agent.ask(question)
        self.turns.append(turn)
        return turn

    def reset_conversation(self) -> None:
        self.agent.reset()
        self.turns.clear()
