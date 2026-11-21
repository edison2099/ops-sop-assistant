"""Split page text into overlapping chunks without breaking sentences."""

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+|\n{2,}")
_HEADING = re.compile(r"^\s*(\d+(\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 ,/&()\-]{2,80}$")


@dataclass
class Chunk:
    text: str
    page: int
    section_heading: str
    index: int


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(text)]
    return [p for p in parts if p]


def detect_heading(line: str) -> bool:
    """A short line that looks like a title, e.g. '4.2 Bird Strike Procedure'."""
    line = line.strip()
    return bool(line) and len(line.split()) <= 12 and not line.endswith(".") and bool(_HEADING.match(line))


def chunk_pages(pages: list[tuple[int, str]], chunk_words: int = 450, overlap_words: int = 75) -> list[Chunk]:
    """pages: list of (page_number, text). Chunks never span pages, so every chunk has one page citation."""
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    chunks: list[Chunk] = []
    heading = ""
    for page_no, text in pages:
        for line in text.splitlines():
            if detect_heading(line):
                heading = line.strip()
                break

        buffer: list[str] = []
        count = 0
        for sentence in split_sentences(text):
            words = len(sentence.split())
            if buffer and count + words > chunk_words:
                chunks.append(Chunk(" ".join(buffer), page_no, heading, len(chunks)))
                # carry the last sentences forward as overlap
                carry: list[str] = []
                carried = 0
                for s in reversed(buffer):
                    n = len(s.split())
                    if carried + n > overlap_words:
                        break
                    carry.insert(0, s)
                    carried += n
                buffer, count = carry, carried
            buffer.append(sentence)
            count += words
        if buffer:
            chunks.append(Chunk(" ".join(buffer), page_no, heading, len(chunks)))
    return chunks
