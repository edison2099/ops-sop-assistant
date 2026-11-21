"""LLM providers. All return (text, input_tokens, output_tokens)."""

import re
from typing import Protocol

from app.config import Settings


class LLM(Protocol):
    def complete(self, system: str, user: str) -> tuple[str, int, int]: ...


class AnthropicLLM:
    def __init__(self, s: Settings):
        self.s = s
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.s.anthropic_api_key)
        return self._client

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        msg = self.client.messages.create(
            model=self.s.llm_model,
            max_tokens=self.s.llm_max_tokens,
            # No temperature: the Claude API no longer accepts sampling parameters.
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return text, msg.usage.input_tokens, msg.usage.output_tokens


class OpenAILLM:
    def __init__(self, s: Settings):
        self.s = s
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.s.openai_api_key)
        return self._client

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        resp = self.client.chat.completions.create(
            model=self.s.llm_model,
            max_tokens=self.s.llm_max_tokens,
            temperature=self.s.llm_temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = resp.usage
        return resp.choices[0].message.content or "", usage.prompt_tokens, usage.completion_tokens


class FakeLLM:
    """Offline stand-in, no API key needed: returns the sentence from source [1] that shares the
    most words with the question. For tests and demos only; it does not really understand text."""

    _stop = set(
        "a an the of to and or in on for is are be by with at as it this that "
        "what when who how do i we can must".split()
    )

    def _words(self, text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in self._stop}

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        m = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\[2\]|\nQuestion:)", user, re.S)
        q = re.search(r"Question:\s*(.+)$", user, re.S)
        if not m or not q:
            return "I can't find this in the documents.", len(user) // 4, 10
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", m.group(1).replace("\n", " ")) if s.strip()]
        qwords = self._words(q.group(1))
        best = max(sentences, key=lambda s: len(qwords & self._words(s)))
        return f"{best} [1]", len(user) // 4, len(best) // 4


def get_llm(s: Settings) -> LLM:
    if s.llm_provider == "anthropic":
        return AnthropicLLM(s)
    if s.llm_provider == "openai":
        return OpenAILLM(s)
    if s.llm_provider == "fake":
        return FakeLLM()
    raise ValueError(f"Unknown llm_provider: {s.llm_provider}")
