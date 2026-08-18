"""Bounded fan-out for the two stages that touch the system under test.

`AdapterClient` already bounds concurrency with the suite's worker semaphore. This
adds the one thing a TaskGroup gets wrong for us: a failing task raises inside an
`ExceptionGroup`, which would hide a `ContractViolation` from the caller. Stages
must fail loudly *as themselves*, so the first error is re-raised unwrapped.
"""

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def _first_leaf(group: BaseExceptionGroup[BaseException]) -> BaseException:
    for error in group.exceptions:
        if isinstance(error, BaseExceptionGroup):
            return _first_leaf(error)
        return error
    return group


async def run_each(items: Sequence[T], work: Callable[[T], Coroutine[Any, Any, None]]) -> None:
    """Run `work` over every item concurrently, re-raising the first failure unwrapped."""
    try:
        async with asyncio.TaskGroup() as group:
            for item in items:
                group.create_task(work(item))
    except BaseExceptionGroup as group_error:
        raise _first_leaf(group_error) from None
