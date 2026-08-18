"""Target selection for the conformance suite.

The default target is the in-repo conformant fake — no socket, no server, so the suite
runs in CI for free. Point it at a registered system with
`pytest contract/conformance --system=mysystem`.

There is no `--base-url`: a URL is one kind of system, reachable through
`adapters.http:HttpAdapter` like any other, which means it needs a registry entry rather
than a special case here.
"""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from contract.adapter import MemoryAdapter
from orchestrator.registry import DEFAULT_REGISTRY, RegistryError, load_adapter, load_system
from reference.fakes import FakeMemory


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--system",
        action="store",
        default=None,
        help="Registry entry to check; defaults to the in-repo conformant fake.",
    )
    parser.addoption(
        "--registry",
        action="store",
        default=str(DEFAULT_REGISTRY),
        help=f"Registry directory (default: {DEFAULT_REGISTRY}).",
    )


@pytest.fixture(scope="session")
def conformance_run_id() -> str:
    """A fresh namespace per suite run, so repeats never collide."""
    return f"conf-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def system(request: pytest.FixtureRequest) -> AsyncIterator[MemoryAdapter]:
    name = request.config.getoption("--system")
    if name is None:
        yield FakeMemory()
        return

    registry = Path(str(request.config.getoption("--registry")))
    try:
        adapter = load_adapter(load_system(str(name), registry))
    except RegistryError as err:
        pytest.fail(f"registry error: {err}")

    try:
        yield adapter
    finally:
        closer = getattr(adapter, "close", None)
        if closer is not None:
            await closer()
