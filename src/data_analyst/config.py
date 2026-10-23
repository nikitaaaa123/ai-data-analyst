"""Application settings, loaded from environment variables."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = None
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 2048
    max_tool_iterations: int = 8
    max_upload_mb: int = 50
    max_result_rows: int = 1000      # rows kept for the UI
    llm_preview_rows: int = 15       # rows sent back to the LLM
    max_history_messages: int = 40
    query_timeout_s: float = 20.0
    app_password: str | None = None
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, default))
            except ValueError:
                return default

        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            model=os.getenv("ANALYST_MODEL", cls.model),
            max_tokens=_int("ANALYST_MAX_TOKENS", cls.max_tokens),
            max_tool_iterations=_int("ANALYST_MAX_TOOL_ITERATIONS", cls.max_tool_iterations),
            max_upload_mb=_int("ANALYST_MAX_UPLOAD_MB", cls.max_upload_mb),
            app_password=os.getenv("APP_PASSWORD") or None,
            log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
        )


def setup_logging(level: str = "INFO") -> None:
    """Idempotent structured-ish logging setup."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
