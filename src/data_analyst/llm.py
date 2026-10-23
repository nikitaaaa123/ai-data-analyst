"""LLM provider abstraction. The agent only depends on :class:`LLMClient`."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from .errors import LLMError

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: list[dict]            # Anthropic-style blocks: text / tool_use
    stop_reason: str               # "end_turn" | "tool_use" | ...
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def text(self) -> str:
        return "\n".join(b["text"] for b in self.content if b.get("type") == "text").strip()

    @property
    def tool_uses(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]


class LLMClient(Protocol):
    name: str

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


@dataclass
class AnthropicLLM:
    api_key: str
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 2048
    name: str = field(init=False, default="anthropic")

    def __post_init__(self) -> None:
        import anthropic  # local import keeps the offline demo usable without the SDK configured

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=self.api_key, max_retries=3, timeout=60.0)
        self.name = f"anthropic:{self.model}"

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        a = self._anthropic
        try:
            resp = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system,
                messages=messages, tools=tools,
            )
        except a.AuthenticationError as exc:
            raise LLMError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.") from exc
        except a.RateLimitError as exc:
            raise LLMError("Rate limit reached. Please wait a moment and try again.") from exc
        except a.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API. Check your network connection.") from exc
        except a.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc

        blocks: list[dict] = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        usage = getattr(resp, "usage", None)
        return LLMResponse(blocks, resp.stop_reason or "end_turn",
                           getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
