"""Registry loading: one YAML file per system under test (ARCH §11).

Three fields — `name`, `adapter`, and an optional `config` handed to `__init__`
verbatim. That is the entire integration.

There is no `auth` block. Credentials are the adapter's own business now, read from its
own environment exactly as its SDK would. Nothing here names a credential, so nothing
here can leak one; an `auth` block from the previous scheme is rejected rather than
ignored, because silently dropping it would leave a system unauthenticated.

Importing is **lazy**: `load_system` reads YAML and nothing else, so one broken SDK
cannot stop `bench ps` or `bench rescore` from working (ARCH §14). The import happens in
`load_adapter`, and every way it can fail gets its own message — "could not load
mysystem" sends you looking in the wrong place.
"""

import importlib
import inspect
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from contract.adapter import MemoryAdapter

DEFAULT_REGISTRY = Path("registry")


class RegistryError(Exception):
    """A registry entry is missing, malformed, or its adapter cannot be constructed."""


class SystemSpec(BaseModel):
    """A system under test, as the orchestrator sees it. No credential, by construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    adapter: str
    config: dict[str, Any] = {}


def load_system(name: str, root: Path = DEFAULT_REGISTRY) -> SystemSpec:
    """Return the SystemSpec for `name` from `root/<name>.yaml`, importing nothing."""
    path = root / f"{name}.yaml"
    if not path.is_file():
        raise RegistryError(f"no registry entry at {path.name} (looked in {root})")

    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise RegistryError(f"{path.name} must contain a YAML mapping")

    if "auth" in raw:
        raise RegistryError(
            f"{path.name} declares an `auth` block, which no longer exists: an adapter "
            "reads its own credentials from its own environment. Remove it and read the "
            "key inside the adapter, or use adapters.http:HttpAdapter with an `auth` "
            "entry in `config` for a hosted system."
        )

    try:
        spec = SystemSpec.model_validate(raw)
    except ValidationError as err:
        fields = ", ".join(str(e["loc"][0]) for e in err.errors()) or "entry"
        raise RegistryError(f"{path.name} is invalid ({fields}): {err}") from err

    if spec.name != name:
        raise RegistryError(f"{path.name} declares name {spec.name!r}, expected {name!r}")
    return spec


def load_adapter(spec: SystemSpec) -> MemoryAdapter:
    """Import, construct, and return the adapter `spec` names."""
    module_name, _, attribute = spec.adapter.partition(":")
    if not module_name or not attribute:
        raise RegistryError(
            f"system {spec.name!r} has adapter {spec.adapter!r}; expected the form "
            "module:Class, for example adapters.mysystem:MySystemAdapter"
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        raise RegistryError(
            f"system {spec.name!r} names adapter {spec.adapter!r}, but module "
            f"{module_name!r} could not be imported: {err}"
        ) from err

    try:
        target = getattr(module, attribute)
    except AttributeError as err:
        raise RegistryError(
            f"system {spec.name!r} names adapter {spec.adapter!r}, but {module_name!r} "
            f"has no attribute {attribute!r}"
        ) from err

    if not inspect.isclass(target):
        raise RegistryError(
            f"system {spec.name!r} names adapter {spec.adapter!r}, which is a "
            f"{type(target).__name__} and not a class"
        )

    missing = [method for method in ("add", "search") if not hasattr(target, method)]
    if missing:
        raise RegistryError(
            f"system {spec.name!r} names adapter {spec.adapter!r}, which does not "
            f"implement MemoryAdapter: missing {', '.join(missing)}"
        )

    try:
        adapter = target(**spec.config) if spec.config else target()
    except Exception as err:
        # Typically a missing SDK surfacing from `__init__`. Broad on purpose: a
        # third-party constructor may raise anything, and one system failing to
        # construct must read as that system's problem, not as a harness crash.
        raise RegistryError(
            f"system {spec.name!r} could not construct {spec.adapter!r}: "
            f"{type(err).__name__}: {err}"
        ) from err

    if not isinstance(adapter, MemoryAdapter):
        raise RegistryError(
            f"system {spec.name!r} constructed {spec.adapter!r}, but it does not "
            "satisfy MemoryAdapter (needs async add and search)"
        )
    return adapter


__all__ = ["DEFAULT_REGISTRY", "RegistryError", "SystemSpec", "load_adapter", "load_system"]
