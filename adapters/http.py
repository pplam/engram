"""`HttpAdapter` — for a system that is a container or a hosted service (ARCH §10).

The escape hatch from the import-an-adapter design: a system you cannot import, because
it is a container or somebody else's deployment, is still benchmarkable by URL. It ships
with the harness so nobody has to write it twice.

It is also the only place a status code is still classified, and that classification is
the whole reason the rest of the harness does not have to care about HTTP:

- **5xx and 429** → `AdapterError`. Upstream is unwell; the suite's policy may retry.
- **4xx** → `ContractViolation`. Our request is wrong or we are being refused; retrying
  a rejected request just spends money to be rejected again.
- **2xx with a body that does not fit** → `ContractViolation`.
"""

import os
from collections.abc import Sequence
from typing import Any

import httpx

from contract.adapter import AdapterError, ContractViolation, Memory, Message

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_SCHEMES = {"bearer": "Bearer", "token": "Token"}


class HttpAdapter:
    """Speaks `/add` and `/search` over HTTP against one system's base URL."""

    def __init__(
        self,
        base_url: str,
        auth: dict[str, str] | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = self._headers(auth)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    @staticmethod
    def _headers(auth: dict[str, str] | None) -> dict[str, str]:
        """Return auth headers, reading the credential from the env var it names.

        Resolved at construction so a missing credential fails `bench doctor` rather
        than surfacing as a 401 after ingest has been paid for. The value is never
        logged and never appears in the registry.
        """
        if not auth:
            return {}
        key_env = auth.get("key_env")
        if not key_env:
            raise ContractViolation("auth is configured without a key_env naming the credential")
        key = os.environ.get(key_env)
        if not key:
            raise ContractViolation(f"env var {key_env} is unset or empty")
        scheme = auth.get("scheme", "bearer")
        if scheme in _SCHEMES:
            return {"Authorization": f"{_SCHEMES[scheme]} {key}"}
        if scheme == "api_key":
            return {"X-Api-Key": key}
        raise ContractViolation(f"unknown auth scheme {scheme!r}")

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        """POST `/add`; returns once the system reports the chunk searchable."""
        payload = await self._post(
            "/add",
            {
                "user_id": user_id,
                "x_chunk_id": chunk_id,
                "messages": [
                    {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                    for m in messages
                ],
            },
        )
        if payload.get("success") is not True:
            raise ContractViolation(
                f"/add did not report success: true (got {payload.get('success')!r})"
            )

    async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
        """POST `/search` and map the response onto `Memory`, order preserved."""
        payload = await self._post("/search", {"user_id": user_id, "query": query, "top_k": top_k})
        data = payload.get("data")
        if not isinstance(data, list):
            raise ContractViolation("/search response has no data list")

        memories: list[Memory] = []
        for item in data:
            if not isinstance(item, dict):
                raise ContractViolation("/search data contains a non-object item")
            if "content" not in item:
                raise ContractViolation("/search data item is missing content")
            memories.append(
                Memory(
                    content=str(item["content"]),
                    score=self._score(item.get("score")),
                    source_ids=self._source_ids(item.get("x_source_ids")),
                )
            )
        return memories

    async def close(self) -> None:
        """Release the underlying connection pool."""
        await self._client.aclose()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
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
            body: Any = response.json()
        except ValueError as err:
            raise ContractViolation(f"{path} response body is not JSON: {err}") from err
        if not isinstance(body, dict):
            raise ContractViolation(
                f"{path} response body is JSON {type(body).__name__}, expected an object"
            )
        return body

    @staticmethod
    def _score(raw: Any) -> float | None:
        return float(raw) if isinstance(raw, int | float) else None

    @staticmethod
    def _source_ids(raw: Any) -> tuple[str, ...]:
        return tuple(str(value) for value in raw) if isinstance(raw, list) else ()


__all__ = ["HttpAdapter"]
