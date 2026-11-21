"""Embedding providers. Use the same provider for indexing and for questions."""

import hashlib
import math
import re
from typing import Protocol

from app.config import Settings


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: str | None):
        self.model = model
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:  # created on first use, so the app starts without a key
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):  # stay under request size limits
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + 100])
            out.extend(d.embedding for d in resp.data)
        return out


_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to and or in on for is are be by with at as it this that from was were".split())


class HashEmbedder:
    """Offline keyword embedder (hashed bag of words). No API key needed.
    Good enough for tests and a demo; use OpenAI embeddings for real meaning-based search."""

    def __init__(self, dims: int = 512):
        self.dims = dims

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in _WORD.findall(text.lower()):
            if word in _STOP:
                continue
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings.embedding_model, settings.openai_api_key)
    if settings.embedding_provider == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedding_provider: {settings.embedding_provider}")
