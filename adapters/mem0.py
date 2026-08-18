"""The self-hosted mem0 server → the two methods (ARCH §10).

mem0 ships a FastAPI server (`server/` in its repo: FastAPI + Postgres/pgvector). You
start it; this adapter is a client. Nothing mem0-shaped runs inside the harness.

**Not `HttpAdapter`.** That one speaks the harness's retired `/add`+`/search` wire
contract, and mem0's server shares only the path name `/search` with it. Every field
differs, so reusing it would send mem0 a payload it rejects:

| interface     | mem0 server                                            |
| ------------- | ------------------------------------------------------ |
| `add`         | `POST /memories`                                       |
| `user_id`     | `user_id=` on add, `filters={"user_id": ...}` on search |
| `messages`    | `messages=[{role, content}]` — no `timestamp` field     |
| `chunk_id`    | `metadata={"x_chunk_id": ...}`                          |
| `top_k`       | `top_k=`                                                |
| `content`     | `memory`                                                |
| `source_ids`  | `metadata["x_chunk_id"]`, echoed back as a list         |

Carrying `chunk_id` through is what puts mem0 on the **exact** recall track. Without it
the row would still score quality, but retrieval would read `unavailable` — the system
could not say which chunk a memory came from, so recall would not be measurable.

Status classification matches `adapters/http.py`, because the split is the contract's
and not this adapter's invention: 5xx and 429 are an unwell upstream the suite may
retry; 4xx is us being refused, and retrying a rejected request just spends money to be
rejected again.
"""

import os
from collections.abc import Sequence
from typing import Any

import httpx

from contract.adapter import AdapterError, ContractViolation, Memory, Message

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_CHUNK_KEY = "x_chunk_id"


class Mem0Adapter:
    """Benchmarks a separately-run mem0 server through the two methods."""

    def __init__(
        self,
        base_url: str,
        api_key_env: str | None = None,
        timeout: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # Resolved at construction so a missing credential fails `bench doctor` rather
        # than surfacing as a 401 after ingest has been paid for. The value is never
        # logged and never appears in the registry, which names the variable only.
        headers: dict[str, str] = {}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise ContractViolation(f"env var {api_key_env} is unset or empty")
            headers["X-API-Key"] = key

        # The timeout is long by default: `add` blocks on mem0's fact extraction, which
        # is an LLM call, so the harness's own per-stage timeout is the real bound.
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        """Store one chunk, tagging it so retrieval can name its source."""
        # `timestamp` is dropped: the server's Message model is {role, content} and
        # rejects extra fields, so sending it would 422 every chunk.
        await self._post(
            "/memories",
            {
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "user_id": user_id,
                "metadata": {_CHUNK_KEY: chunk_id},
            },
        )
        # No success assertion. mem0 legitimately extracts no fact from a chunk and
        # returns an empty result list; that is its algorithm, not a failed write.

    async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
        """Return memories for `user_id`, preserving mem0's rank order."""
        # The query goes through verbatim: rewriting it would measure our rewriting.
        # `filters` rather than a top-level user_id — the latter is deprecated on this
        # endpoint and logs a warning per call.
        payload = await self._post(
            "/search",
            {"query": query, "filters": {"user_id": user_id}, "top_k": top_k},
        )

        memories: list[Memory] = []
        for item in self._results(payload):
            content = item.get("memory")
            if not content:
                # Empty content would score as a wrong answer, which reads as a memory
                # result rather than as the broken integration it is.
                raise ContractViolation("/search returned a memory with no content")
            metadata = item.get("metadata")
            chunk_id = metadata.get(_CHUNK_KEY) if isinstance(metadata, dict) else None
            score = item.get("score")
            memories.append(
                Memory(
                    # mem0 calls it `memory`; the interface calls it `content`.
                    content=str(content),
                    score=float(score) if isinstance(score, int | float) else None,
                    source_ids=(str(chunk_id),) if chunk_id is not None else (),
                )
            )
        return memories

    async def use_model_key(self, value: str) -> None:
        """Present `value` on mem0's own model calls, so its cost is attributable (D6(a)).

        `POST /configure` rather than the container's environment. The key is named for a
        run that does not exist when the container starts, so delivering it through
        `.env` would mean a restart per run; mem0 reads `config.api_key or
        os.getenv("OPENAI_API_KEY")`, so a config value wins over the static env key
        without one. Re-configuring rebuilds mem0's `Memory` instance but leaves the
        pgvector collection alone, so memories from an earlier run survive.

        Both halves get it: extraction and embedding are separately configured and each
        reads its own `api_key`. The payload is deliberately partial — mem0 deep-merges
        it, so the model, temperature, and completion budget that `scripts/setup_mem0.py`
        pinned stay as they are, and naming a provider here would re-run mem0's bundled
        provider validation for no gain.

        Note mem0 persists config overrides to its Postgres `Settings` table, so the value
        outlives the run there. Harmless for a local benchmark — the key is scoped to a
        finished run and the gateway can revoke it — but it is not as short-lived as its
        name suggests. Never logged, here or anywhere.
        """
        await self._post(
            "/configure",
            {
                "llm": {"config": {"api_key": value}},
                "embedder": {"config": {"api_key": value}},
            },
        )

    async def close(self) -> None:
        """Release the underlying connection pool."""
        await self._client.aclose()

    @staticmethod
    def _results(payload: Any) -> list[dict[str, Any]]:
        """Return the memory list from either reply shape.

        v1.1 returns `{"results": [...]}`; older builds returned a bare list. Accept
        both rather than pinning a mem0 version we do not control.
        """
        found = payload.get("results") if isinstance(payload, dict) else payload
        if not isinstance(found, list):
            raise ContractViolation("/search response carries no results list")
        for item in found:
            if not isinstance(item, dict):
                raise ContractViolation("/search results contain a non-object item")
        return found

    async def _post(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            response = await self._client.post(path, json=payload)
        except httpx.HTTPError as err:
            raise AdapterError(f"transport error calling {path}: {err}") from err

        status = response.status_code
        if status in _RETRYABLE_STATUS:
            raise AdapterError(f"{path} returned HTTP {status}")
        if status // 100 != 2:
            raise ContractViolation(f"{path} returned HTTP {status}, which is not retryable")

        try:
            return response.json()
        except ValueError as err:
            raise ContractViolation(f"{path} response body is not JSON: {err}") from err


__all__ = ["Mem0Adapter"]
