"""One test per verifiable contract row, against whatever `system` points at (§1.5)."""

import pytest

from contract.adapter import MemoryAdapter
from contract.conformance.checks import CHECKS, Check, CheckFailed


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.name)
async def test_contract_check(check: Check, system: MemoryAdapter, conformance_run_id: str) -> None:
    try:
        await check.run(system, conformance_run_id)
    except CheckFailed as err:
        pytest.fail(f"{check.name}: {err}")
