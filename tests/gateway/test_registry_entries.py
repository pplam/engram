"""Committed registry entries must not carry a credential, and must load.

These checks run over the real files in `registry/`, not synthetic ones: a literal key
in a committed file is a leak, and it is the committed copy that leaks. `load_system`
alone would catch a malformed entry, so the credential scan is the part worth having.

There is no model-pinning check here any more. Pinning was checkable when a registry
entry configured the system's own LLM and embedder — the in-process `mem0-proxy` entry
did, so its ids could be compared against the suite's. The self-hosted mem0 server reads
that config from its own `.env` and `POST /configure` instead, which is outside this
repo and outside anything a test here can assert. `registry/mem0.yaml` says so at
the point where someone would otherwise look for the pin.
"""

from pathlib import Path

import pytest
import yaml

from orchestrator.registry import load_system

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "registry"

# Anything that looks like a key rather than a reference to one. `env:` and `key_env`
# name a variable; a literal starts with a vendor prefix.
_KEY_PREFIXES = ("sk-", "sk_", "m0sk_", "Bearer ")

ENTRIES = sorted(p.name for p in REGISTRY.glob("*.yaml"))


def test_the_registry_is_not_empty() -> None:
    """A glob over an empty directory would make every check below vacuously pass."""
    assert ENTRIES


@pytest.mark.parametrize("name", ENTRIES)
def test_no_credential_is_written_into_a_registry_entry(name: str) -> None:
    """Credentials are env var names. A literal key in a committed file is a leak."""
    raw = (REGISTRY / name).read_text()
    for prefix in _KEY_PREFIXES:
        assert prefix not in raw, f"{name} looks like it carries a literal credential"


@pytest.mark.parametrize("name", ENTRIES)
def test_every_committed_entry_loads(name: str) -> None:
    """A committed entry that cannot be read is broken for everyone, not just its author."""
    spec = load_system(Path(name).stem, REGISTRY)
    assert spec.adapter, f"{name} names no adapter"


@pytest.mark.parametrize("name", ENTRIES)
def test_an_auth_credential_is_named_by_variable_not_value(name: str) -> None:
    """`api_key_env` / `key_env` hold the *name* of a variable, resolved at construction."""
    config = yaml.safe_load((REGISTRY / name).read_text()).get("config") or {}
    for field in ("api_key_env", "key_env"):
        if field in config:
            assert not config[field].startswith(_KEY_PREFIXES)
    auth = config.get("auth")
    if isinstance(auth, dict) and "key_env" in auth:
        assert not auth["key_env"].startswith(_KEY_PREFIXES)
