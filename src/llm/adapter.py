"""
LLM Adapter — unified interface for all LLM calls.
Backend is selected via LLM_BACKEND env var: "groq" (default) or "ollama".
All pipeline components MUST call the LLM only through this adapter.
"""

from __future__ import annotations

import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import httpx
from openai import OpenAI
from dotenv import load_dotenv

# Explicitly load .env from project root
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")
load_dotenv()


@dataclass

class LLMResponse:
    """Structured response from any LLM backend."""
    content: str
    model: str
    usage: dict = field(default_factory=dict)  # {prompt_tokens, completion_tokens, total_tokens}
    latency_ms: float = 0.0
    raw_response: Any = None


class BaseLLMAdapter(ABC):
    """Abstract base class — every backend implements this."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        ...

    def load_prompt_template(self, template_path: str) -> str:
        """Load a prompt template from a file. All prompts live in prompts/*.txt."""
        # Resolve relative to project root
        if not os.path.isabs(template_path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            template_path = os.path.join(project_root, template_path)
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()


class GroqAdapter(BaseLLMAdapter):
    """Groq Cloud — OpenAI-compatible chat completions."""

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
                api_key = "test_key_for_testing"
            else:
                raise ValueError("GROQ_API_KEY environment variable is required for GroqAdapter")
        self.client = OpenAI(

            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=response,
        )


class OllamaAdapter(BaseLLMAdapter):
    """Local Ollama backend."""

    def __init__(self):
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "gemma4:26b")
        self.host = self.host.rstrip("/")

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": options,
            "stream": False,
            # Gemma 4 enables thinking by default. Keep its reasoning out of
            # the answer channel and avoid spending the response budget on it.
            "think": False,
        }
        if json_mode:
            payload["format"] = "json"

        start = time.perf_counter()
        response = httpx.post(
            f"{self.host}/api/chat", json=payload, timeout=300.0
        )
        response.raise_for_status()
        response_data = response.json()
        latency_ms = (time.perf_counter() - start) * 1000

        prompt_tokens = response_data.get("prompt_eval_count", 0) or 0
        completion_tokens = response_data.get("eval_count", 0) or 0
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        return LLMResponse(
            content=response_data.get("message", {}).get("content", "") or "",
            model=self.model,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=response_data,
        )


# ── Singleton factory ────────────────────────────────────────────────────────

_adapter_instance: Optional[BaseLLMAdapter] = None


def get_adapter() -> BaseLLMAdapter:
    """Return the configured LLM adapter (singleton). Selected via LLM_BACKEND env var."""
    global _adapter_instance
    if _adapter_instance is None:
        backend = os.environ.get("LLM_BACKEND", "groq").lower()
        if backend == "ollama":
            _adapter_instance = OllamaAdapter()
        else:
            _adapter_instance = GroqAdapter()
    return _adapter_instance


def reset_adapter() -> None:
    """Reset the singleton — useful for tests."""
    global _adapter_instance
    _adapter_instance = None


# Convenience alias
LLMAdapter = BaseLLMAdapter
