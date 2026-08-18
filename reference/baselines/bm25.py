"""`bm25` — lexical retrieval, the bar a memory system has to clear (ARCH §10).

Answers "does this system beat lexical search?". Okapi BM25 over per-user chunk
documents, stdlib only. `Bm25Adapter` in `adapters.py` is the interface face of it.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens."""
    return _WORD.findall(text.lower())


@dataclass
class _Doc:
    mem_id: str
    content: str
    chunk_id: str | None
    counts: Counter[str]
    length: int


@dataclass(frozen=True)
class Hit:
    """One scored document. `chunk_id` is what puts the row on the exact recall track."""

    mem_id: str
    content: str
    score: float
    chunk_id: str | None


@dataclass
class Bm25System:
    """Per-user BM25 index. `user_id` is the only isolation boundary."""

    _docs: dict[str, list[_Doc]] = field(default_factory=dict)
    _counter: int = 0

    def add(self, user_id: str, contents: list[str], chunk_id: str | None) -> None:
        """Index one document per message under `user_id`."""
        for content in contents:
            self._counter += 1
            tokens = tokenize(content)
            self._docs.setdefault(user_id, []).append(
                _Doc(
                    mem_id=f"mem_{self._counter}",
                    content=content,
                    chunk_id=chunk_id,
                    counts=Counter(tokens),
                    length=len(tokens),
                )
            )

    def search(self, user_id: str, query: str, top_k: int) -> list[Hit]:
        """Return the top_k best-scoring documents for `user_id`, best first."""
        docs = self._docs.get(user_id, [])
        if not docs:
            return []

        total = len(docs)
        avg_length = sum(d.length for d in docs) / total
        frequencies = Counter(term for doc in docs for term in doc.counts)

        scored: list[tuple[float, _Doc]] = []
        for doc in docs:
            score = 0.0
            for term in tokenize(query):
                occurrences = doc.counts.get(term, 0)
                if not occurrences:
                    continue
                containing = frequencies[term]
                idf = math.log(1 + (total - containing + 0.5) / (containing + 0.5))
                norm = K1 * (1 - B + B * doc.length / avg_length) if avg_length else K1
                score += idf * occurrences * (K1 + 1) / (occurrences + norm)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda pair: (-pair[0], pair[1].mem_id))
        return [
            Hit(
                mem_id=doc.mem_id,
                content=doc.content,
                score=round(score, 6),
                chunk_id=doc.chunk_id,
            )
            for score, doc in scored[:top_k]
        ]
