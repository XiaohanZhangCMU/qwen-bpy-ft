"""
LLM generator: wraps an OpenAI-compatible API to produce bpy code turns.

Holds no conversation state — that lives in conversation.py.
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from shared.logging_utils import get_logger

logger = get_logger(__name__)


class Generator:
    """
    Thin wrapper around an OpenAI-compatible chat completion endpoint.

    Args:
        model_id:     Model name (e.g. "gpt-4o", "Qwen/Qwen2.5-Coder-7B-Instruct").
        api_base:     Base URL of the API server.
        api_key:      API key.  Falls back to OPENAI_API_KEY env var.
        temperature:  Sampling temperature.
        max_tokens:   Max tokens per completion.
    """

    def __init__(
        self,
        model_id: str,
        api_base: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=api_base,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def complete(self, messages: list[dict]) -> str:
        """
        Send *messages* (OpenAI chat format) and return the assistant reply text.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            The assistant's reply as a plain string.
        """
        logger.debug("Calling LLM", extra={"model": self.model_id, "n_messages": len(messages)})
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.debug("LLM response received", extra={"chars": len(content)})
        return content
