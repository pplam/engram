"""Gold evidence handles → chunk ids (ARCH §7.2).

Datasets annotate evidence in their own unit: LoCoMo names messages (`m3`),
LongMemEval names sessions (`sess0`). Ingestion works in chunks, and recall compares
against the `x_source_ids` a system echoes, which are chunk ids. This is the one
place that expansion happens, so the unit label on the report stays a property of
the dataset rather than something a scorer infers.

A coarse unit expands to *every* chunk derived from it; ARCH §7.2 notes that makes
recall an easier target, which is why the unit travels with the number.
"""

from collections.abc import Sequence

from orchestrator.chunking import Chunk
from orchestrator.datasets import Granularity


def expand_evidence(
    handles: Sequence[str], chunks: Sequence[Chunk], unit: Granularity
) -> tuple[str, ...]:
    """Return the chunk ids covering `handles`, in chunk order, with no duplicates."""
    if not handles:
        return ()

    wanted = set(handles)
    matched: list[str] = []

    for chunk in chunks:
        if unit == "message":
            # A dataset may name messages by position (`m3`) or by its own id (LoCoMo's
            # `D1:3`). Both are message-granular, so both resolve here rather than forcing
            # every adapter to renumber its evidence into our positions.
            positions = {f"m{index}" for index in chunk.source_indices}
            ids = {
                str(message["dia_id"])
                for message in chunk.messages
                if message.get("dia_id") is not None
            }
            hit = bool((positions | ids) & wanted)
        else:
            sessions = {
                str(message["session_id"])
                for message in chunk.messages
                if message.get("session_id") is not None
            }
            hit = bool(sessions & wanted)
        if hit:
            matched.append(chunk.chunk_id)

    return tuple(matched)
