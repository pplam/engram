"""Populate `datasets/*/data/` from the urls and hashes pinned in each `dataset.yaml`.

The corpora are fetched, not vendored. Two reasons: LongMemEval-S Cleaned is 265 MB, and
LoCoMo-Refined is CC BY-NC 4.0, which we should not redistribute. A manifest pins both a
`url` and a `sha256`, so a fetched file is verifiable — that pin is the integrity
guarantee, and a mismatch is refused rather than installed.

Reads the manifests, so adding a dataset never means editing this file.

    uv run python -m scripts.fetch_datasets
"""

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

CHUNK = 1 << 20


class FetchError(Exception):
    """A download failed, or arrived with a hash that does not match the manifest."""


@dataclass(frozen=True)
class Target:
    """One corpus to fetch: where from, where to, and what it must hash to."""

    name: str
    url: str
    sha256: str
    destination: Path


def targets(root: Path) -> list[Target]:
    """Return a fetchable target for every dataset under `root` that pins a url."""
    found: list[Target] = []
    for manifest in sorted(root.glob("*/dataset.yaml")):
        raw = yaml.safe_load(manifest.read_text())
        if not isinstance(raw, dict):
            continue
        source = raw.get("source") or {}
        url = source.get("url")
        if not url:
            # No url means the corpus ships with the repo, as the fixtures do.
            continue
        found.append(
            Target(
                name=str(raw.get("name", manifest.parent.name)),
                url=str(url),
                sha256=str(source.get("sha256", "")),
                destination=manifest.parent / str(source.get("path", "")),
            )
        )
    return found


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def fetch(target: Target, client: httpx.Client) -> bool:
    """Download `target` unless it is already correct; return True if it was written."""
    if target.destination.is_file() and _digest(target.destination) == target.sha256:
        return False

    target.destination.parent.mkdir(parents=True, exist_ok=True)
    temp = target.destination.with_suffix(target.destination.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with client.stream("GET", target.url, follow_redirects=True, timeout=1800) as response:
            response.raise_for_status()
            with temp.open("wb") as handle:
                for block in response.iter_bytes(CHUNK):
                    digest.update(block)
                    handle.write(block)
    except httpx.HTTPError as err:
        temp.unlink(missing_ok=True)
        raise FetchError(f"{target.name}: download failed: {err}") from err

    actual = digest.hexdigest()
    if actual != target.sha256:
        # Never install a file the manifest does not vouch for.
        temp.unlink(missing_ok=True)
        raise FetchError(
            f"{target.name}: sha256 mismatch — manifest pins {target.sha256[:12]}..., "
            f"download is {actual[:12]}..."
        )

    temp.replace(target.destination)
    return True


def main(root: Path = Path("datasets")) -> int:
    """Fetch every pinned corpus that is missing or stale; return an exit code."""
    found = targets(root)
    if not found:
        print(f"no fetchable datasets under {root}", file=sys.stderr)
        return 1

    failures = 0
    for target in found:
        try:
            written = fetch(target, httpx.Client())
        except FetchError as err:
            print(f"{err}", file=sys.stderr)
            failures += 1
            continue
        state = "fetched" if written else "already present"
        print(f"{target.name}: {state} ({target.destination})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
