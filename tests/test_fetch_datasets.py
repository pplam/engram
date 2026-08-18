"""`scripts/fetch_datasets.py` populates `datasets/*/data/` from pinned manifests.

The corpora are not vendored: a manifest pins a `url` and a `sha256`, so the file is
verifiable without living in git (and LoCoMo-Refined is CC BY-NC 4.0, which should not be
redistributed). This script is the one way to get them, and it must refuse to install a
file whose hash does not match the pin.
"""

import hashlib
from pathlib import Path

import httpx
import pytest

from scripts.fetch_datasets import FetchError, Target, fetch, targets

REPO = Path(__file__).resolve().parents[1]


def transport(body: bytes) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(200, content=body))


def test_targets_are_read_from_the_manifests() -> None:
    """A new dataset must not require editing this script (ARCH §9)."""
    found = {t.name for t in targets(REPO / "datasets")}
    assert {"locomo-refined", "longmemeval-cleaned"} <= found


def test_a_target_carries_its_pinned_hash_and_url() -> None:
    locomo = next(t for t in targets(REPO / "datasets") if t.name == "locomo-refined")
    assert locomo.url and locomo.url.startswith("https://")
    assert len(locomo.sha256) == 64


def test_fetch_writes_the_file_when_the_hash_matches(tmp_path: Path) -> None:
    body = b'[{"sample_id": "x"}]'
    target = Target(
        name="d",
        url="https://example.invalid/d.json",
        sha256=hashlib.sha256(body).hexdigest(),
        destination=tmp_path / "data" / "d.json",
    )
    with httpx.Client(transport=transport(body)) as client:
        assert fetch(target, client) is True
    assert target.destination.read_bytes() == body


def test_fetch_refuses_a_file_whose_hash_does_not_match(tmp_path: Path) -> None:
    """The pin is the integrity guarantee; installing anyway would defeat it."""
    target = Target(
        name="d",
        url="https://example.invalid/d.json",
        sha256="0" * 64,
        destination=tmp_path / "data" / "d.json",
    )
    with (
        httpx.Client(transport=transport(b"tampered")) as client,
        pytest.raises(FetchError, match="sha256"),
    ):
        fetch(target, client)
    assert not target.destination.exists(), "a mismatched download must not be left behind"


def test_fetch_skips_a_file_that_is_already_correct(tmp_path: Path) -> None:
    """Re-running must not re-download 265 MB."""
    body = b"already here"
    destination = tmp_path / "data" / "d.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(body)

    target = Target(
        name="d",
        url="https://example.invalid/d.json",
        sha256=hashlib.sha256(body).hexdigest(),
        destination=destination,
    )

    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError("an already-correct file must not be re-downloaded")

    with httpx.Client(transport=httpx.MockTransport(explode)) as client:
        assert fetch(target, client) is False


def test_a_manifest_without_a_url_is_skipped() -> None:
    """The fixtures pin no url; they ship with the repo and need no fetching."""
    fixtures = REPO / "tests" / "datasets" / "fixtures"
    assert targets(fixtures) == []
