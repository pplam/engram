"""The model seam (§2.8).

One narrow protocol. `OpenAiCompatibleChat` is the direct-provider implementation
used in Phase 2; Phase 3 points the same class at the gateway's base URL and adds
the virtual key plus `x-bf-dim-*` headers, so nothing downstream changes.

Never log prompts, messages, or completions here — IDs, statuses, and counts only.
"""

from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol, Self

import httpx

from orchestrator.gateway import Gateway

Message = dict[str, Any]


class ModelError(Exception):
    """The model call failed, or the response was not usable."""


@dataclass(frozen=True)
class Completion:
    """One completion plus what it cost."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class ChatClient(Protocol):
    """The only model surface any stage may use."""

    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        seed: int | None = None,
    ) -> Completion:
        """Return the completion for `messages` under the pinned model settings."""
        ...


@dataclass(frozen=True)
class StubCall:
    """One recorded call, for tests to assert against."""

    messages: list[Message]
    model: str
    temperature: float
    seed: int | None


@dataclass
class StubChat:
    """An in-memory ChatClient. Tests use this; no test may make a real model call."""

    replies: list[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: list[StubCall] = field(default_factory=list)

    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        seed: int | None = None,
    ) -> Completion:
        """Return the next canned reply, cycling when exhausted."""
        self.calls.append(StubCall(list(messages), model, temperature, seed))
        text = self.replies[(len(self.calls) - 1) % len(self.replies)]
        return Completion(
            text=text,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


class OpenAiCompatibleChat:
    """Chat completions over any OpenAI-compatible endpoint, including the gateway."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        path: str = "/v1/chat/completions",
    ) -> None:
        self._client = client
        self._path = path
        self._headers = dict(headers or {})
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    async def use_key(self, value: str, run_id: str, phase: str) -> None:
        """Present `value` on subsequent calls, labelled for `run_id` and `phase`.

        Set after construction rather than passed in, because the key is provisioned at
        run start and the client is built before the run knows its suite. An unkeyed
        client is a working unmetered run, so this stays optional: the harness calls it
        when a gateway provisioned keys, and skips it otherwise.

        Async only to match how the caller reaches every other optional member; nothing
        here awaits.
        """
        self._headers.update(Gateway.headers_for(value, run_id, phase))

    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        seed: int | None = None,
    ) -> Completion:
        """POST one chat completion and return its text and token counts."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed

        try:
            response = await self._client.post(self._path, json=payload, headers=self._headers)
        except httpx.HTTPError as err:
            raise ModelError(f"model call failed: {err}") from err

        if response.status_code // 100 != 2:
            detail = ""
            try:
                body = response.json()
            except ValueError:
                body = {}
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    detail = f" {error.get('type') or error.get('message') or ''}".rstrip()
                elif isinstance(error, str):
                    detail = f" {error}"
            raise ModelError(
                f"model call returned HTTP {response.status_code}:{detail or ' error'}"
            )

        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices:
            raise ModelError("model response carried no choices")
        usage = body.get("usage") or {}
        return Completion(
            text=choices[0].get("message", {}).get("content") or "",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )
