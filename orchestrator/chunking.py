"""Message windowing: pure function, messages in, chunks out (D4, ARCH §6).

`max_messages_per_chunk` is the primary bound; `max_words_per_chunk` is a
secondary cut inside that window. A single message over the word limit is split
at a sentence boundary, and its parts keep the parent chunk_id plus an ordinal so
provenance survives the split.
"""

import re
from dataclasses import dataclass
from typing import Any

from orchestrator.suite import Chunking

Message = dict[str, Any]

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """One unit of ingestion: a chunk_id, its messages, and where they came from.

    `source_indices` are positions in the context's message list. Gold evidence is
    annotated in dataset units, so this is what lets the plan map `m3` onto a chunk
    id — including across a split, where two chunks share one source index.
    """

    chunk_id: str
    messages: tuple[Message, ...]
    source_indices: tuple[int, ...] = ()


def _words(text: str) -> int:
    return len(text.split())


def _split_oversized(message: Message, max_words: int) -> list[list[str]]:
    """Group a message's sentences into runs of at most `max_words` words each."""
    sentences = [s for s in _SENTENCE.split(message["content"].strip()) if s]
    groups: list[list[str]] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        size = _words(sentence)
        if current and count + size > max_words:
            groups.append(current)
            current, count = [], 0
        current.append(sentence)
        count += size
    if current:
        groups.append(current)
    return groups


def chunk_messages(messages: list[Message], config: Chunking) -> list[Chunk]:
    """Return the chunks for `messages` under the suite's chunking pins."""
    chunks: list[Chunk] = []
    window: list[Message] = []
    sources: list[int] = []
    window_words = 0
    index = 0

    def flush() -> None:
        nonlocal window, sources, window_words, index
        if window:
            chunks.append(
                Chunk(
                    chunk_id=f"c{index}",
                    messages=tuple(window),
                    source_indices=tuple(sources),
                )
            )
            index += 1
            window, sources, window_words = [], [], 0

    for position, message in enumerate(messages):
        size = _words(message["content"])

        if size > config.max_words_per_chunk:
            flush()
            groups = _split_oversized(message, config.max_words_per_chunk)
            for ordinal, group in enumerate(groups):
                part = {**message, "content": " ".join(group)}
                chunks.append(
                    Chunk(
                        chunk_id=f"c{index}.{ordinal}",
                        messages=(part,),
                        source_indices=(position,),
                    )
                )
            index += 1
            continue

        if window and (
            len(window) >= config.max_messages_per_chunk
            or window_words + size > config.max_words_per_chunk
        ):
            flush()

        window.append(message)
        sources.append(position)
        window_words += size

    flush()
    return chunks
