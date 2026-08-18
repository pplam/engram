"""Reference baselines (ARCH §10).

`no_memory` and `bm25` are ordinary `MemoryAdapter` implementations and pass conformance
unchanged — the cheapest possible check that the interface is implementable.
`oracle_gold` and `long_context` cannot be: they need gold evidence and the whole corpus,
which the interface deliberately does not carry, so they run in-process and emit
`retrieve.jsonl` directly.
"""

from .adapters import (
    BASELINE_ADAPTERS,
    BaselineAdapter,
    Bm25Adapter,
    NoMemoryAdapter,
    build_baseline,
)
from .bm25 import Bm25System
from .in_process import inject_retrieval

__all__ = [
    "BASELINE_ADAPTERS",
    "BaselineAdapter",
    "Bm25Adapter",
    "Bm25System",
    "NoMemoryAdapter",
    "build_baseline",
    "inject_retrieval",
]
