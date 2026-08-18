"""Registry loading: the whole integration for a system is ~3 lines of YAML (§1.3).

`name`, `adapter`, and an optional `config` passed to `__init__` verbatim. There is no
`auth` block: credentials are the adapter's own business, read from its own environment
exactly as its SDK would. Nothing here names a credential, so nothing here can leak one.

Every failure mode gets its own message, because "could not load mysystem" sends you
looking in the wrong place.
"""

from pathlib import Path
from typing import Any

import pytest

from contract.adapter import MemoryAdapter
from orchestrator.registry import RegistryError, load_adapter, load_system
from reference.fakes import FakeMemory

GOOD = """
name: mysystem
adapter: reference.fakes:FakeMemory
config: {}
"""


def write(tmp_path: Path, name: str, body: str) -> Path:
    root = tmp_path / "registry"
    root.mkdir(exist_ok=True)
    (root / f"{name}.yaml").write_text(body)
    return root


def test_loads_a_well_formed_registry_entry(tmp_path: Path) -> None:
    spec = load_system("mysystem", write(tmp_path, "mysystem", GOOD))
    assert spec.name == "mysystem"
    assert spec.adapter == "reference.fakes:FakeMemory"
    assert spec.config == {}


def test_config_is_optional(tmp_path: Path) -> None:
    body = "name: mysystem\nadapter: reference.fakes:FakeMemory\n"
    assert load_system("mysystem", write(tmp_path, "mysystem", body)).config == {}


def test_config_reaches_the_constructor_unchanged(tmp_path: Path) -> None:
    body = """
name: flaky
adapter: reference.fakes:FakeMemory
config: {index_late: true}
"""
    adapter = load_adapter(load_system("flaky", write(tmp_path, "flaky", body)))
    # `load_adapter` returns the protocol, not the concrete class, so narrow before
    # reading a field only the fake has.
    assert isinstance(adapter, FakeMemory)
    assert adapter.index_late is True


def test_a_loaded_adapter_satisfies_the_protocol(tmp_path: Path) -> None:
    adapter = load_adapter(load_system("mysystem", write(tmp_path, "mysystem", GOOD)))
    assert isinstance(adapter, MemoryAdapter)


def test_missing_adapter_field_fails_naming_the_field(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="adapter"):
        load_system("mysystem", write(tmp_path, "mysystem", "name: mysystem\n"))


def test_unknown_system_fails_naming_the_file(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="absent.yaml"):
        load_system("absent", write(tmp_path, "mysystem", GOOD))


def test_name_mismatch_between_file_and_field_fails(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="name"):
        load_system("other", write(tmp_path, "other", GOOD))


def test_an_auth_block_is_rejected_rather_than_silently_ignored(tmp_path: Path) -> None:
    """It used to be meaningful. Ignoring it would leave a system unauthenticated."""
    body = GOOD + "auth: {scheme: bearer, key_env: MYSYSTEM_TOKEN}\n"
    with pytest.raises(RegistryError, match="auth"):
        load_system("mysystem", write(tmp_path, "mysystem", body))


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("no_colon_at_all", "module:Class"),
        ("reference.fakes:", "module:Class"),
        (":FakeMemory", "module:Class"),
    ],
)
def test_a_malformed_target_fails_showing_the_expected_form(
    tmp_path: Path, target: str, expected: str
) -> None:
    # Quoted: a bare `module:` is not even valid YAML, and that is a different error.
    body = f'name: mysystem\nadapter: "{target}"\n'
    with pytest.raises(RegistryError, match=expected):
        load_adapter(load_system("mysystem", write(tmp_path, "mysystem", body)))


def test_an_unimportable_module_fails_naming_the_target(tmp_path: Path) -> None:
    body = "name: mysystem\nadapter: nonexistent.module:Thing\n"
    spec = load_system("mysystem", write(tmp_path, "mysystem", body))
    with pytest.raises(RegistryError, match="nonexistent.module"):
        load_adapter(spec)


def test_a_missing_attribute_fails_naming_the_attribute(tmp_path: Path) -> None:
    body = "name: mysystem\nadapter: reference.fakes:NoSuchClass\n"
    spec = load_system("mysystem", write(tmp_path, "mysystem", body))
    with pytest.raises(RegistryError, match="NoSuchClass"):
        load_adapter(spec)


def test_a_target_that_is_not_a_class_fails(tmp_path: Path) -> None:
    body = "name: mysystem\nadapter: reference.fakes:_OVERSHOOT_BY\n"
    spec = load_system("mysystem", write(tmp_path, "mysystem", body))
    with pytest.raises(RegistryError, match="not a class"):
        load_adapter(spec)


def test_a_class_that_does_not_satisfy_the_protocol_fails(tmp_path: Path) -> None:
    body = "name: mysystem\nadapter: pathlib:Path\n"
    spec = load_system("mysystem", write(tmp_path, "mysystem", body))
    with pytest.raises(RegistryError, match="add.*search|search.*add"):
        load_adapter(spec)


def test_a_constructor_that_raises_fails_naming_the_system(tmp_path: Path) -> None:
    """The common case is a missing SDK, which surfaces from `__init__` as ImportError."""
    body = "name: mysystem\nadapter: tests.orchestrator.test_registry:Exploding\n"
    spec = load_system("mysystem", write(tmp_path, "mysystem", body))
    with pytest.raises(RegistryError, match="mysystem"):
        load_adapter(spec)


def test_the_import_happens_lazily_not_at_load_time(tmp_path: Path) -> None:
    """One broken SDK must not take down `bench ps` or `bench rescore` (ARCH §14)."""
    body = "name: broken\nadapter: nonexistent.module:Thing\n"
    spec = load_system("broken", write(tmp_path, "broken", body))
    assert spec.adapter == "nonexistent.module:Thing"


def test_a_registry_entry_names_no_credential(tmp_path: Path) -> None:
    """The structural guarantee: there is no field in which a key could be written."""
    spec = load_system("mysystem", write(tmp_path, "mysystem", GOOD))
    assert not hasattr(spec, "auth_key_env")


class Exploding:
    """An adapter whose constructor fails, standing in for a missing SDK."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        raise ImportError("no module named 'mysystem_sdk'")

    async def add(self, user_id: str, messages: Any, chunk_id: str) -> None:
        """Never reached."""

    async def search(self, user_id: str, query: str, top_k: int) -> Any:
        """Never reached."""
