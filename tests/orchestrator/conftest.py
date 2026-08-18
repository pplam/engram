"""Keeping CLI-level tests off the network.

`bench run` always goes to the gateway — there is no stub flag to pass and no endpoint to
redirect — so any test that drives a whole run has to supply a transport. This patches the
two seams the CLI exposes for it: `model_transport` for model calls, with a local
`httpx.MockTransport`, and `gateway_client` for key provisioning and metering.

Autouse, because the failure it prevents is a test quietly reaching a live endpoint.
Opting in per test would mean every new run-level test is one forgotten fixture away from
doing so. A test that wants specific replies, or wants to be metered, patches the seam
itself; doing so overrides this fixture for that test.

`gateway_client` is disabled rather than mocked, so runs here are unmetered — which is what
these tests already assert (a row's cost reads `unavailable` "because these runs had no
gateway"). Metering has its own file, `test_cli_metering.py`, where the seam is patched
with a transport that provisions keys and serves counters.
"""

import httpx
import pytest

# What the judge prompt asks for. An answer stage reads `choices[0].message.content` the
# same way, so one reply serves both stages: the answer becomes this JSON string, and the
# judge parses it into CORRECT.
CANNED_REPLY = '{"label": "CORRECT"}'


@pytest.fixture(autouse=True)
def offline_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every model call at a local transport, and leave runs unmetered."""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": CANNED_REPLY}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            },
        )

    monkeypatch.setattr("orchestrator.cli.model_transport", lambda: httpx.MockTransport(handle))
    monkeypatch.setattr("orchestrator.cli.gateway_client", lambda: None)
