import asyncio
from collections.abc import Callable
from typing import Any, Coroutine

import pytest

from asynchedging.hedge import AllTasksFailedError, NoDelaysProvidedError, hedge

pytestmark = pytest.mark.asyncio


async def _ok(val: str, delay: float = 0.0) -> str:
    if delay:
        await asyncio.sleep(delay)
    return val


async def _fail(delay: float = 0.0) -> str:
    if delay:
        await asyncio.sleep(delay)
    raise RuntimeError("boom")


def _accept_non_empty(s: str) -> bool:
    return bool(s.strip())


# Hedge: all fail, should raise AllTasksFailedError
async def test_hedge_all_fail_raises_with_errors() -> None:
    async def factory() -> str:
        return await _fail(0.01)

    with pytest.raises(AllTasksFailedError) as exc:
        await hedge(factory, delays_s=[0.0, 0.01])
    assert len(exc.value.exceptions) == 3


# Hedge: no delays provided, should raise NoDelaysProvidedError
async def test_hedge_no_delays_raises() -> None:
    async def factory() -> str:
        return await _ok("A")

    with pytest.raises(NoDelaysProvidedError):
        await hedge(factory, delays_s=[])


# Hedge: fastest wins, all start before winner completes
async def test_hedge_multiple_instances_with_stagger() -> None:
    results = ["primary", "backup1", "backup2"]
    delays = [0.4, 0.4]  # backup1 and backup2 start after 0.4s
    times = [1.0, 0.4, 0.8]  # primary=1.0s, backup1=0.4s, backup2=0.8s

    def factory_gen() -> Callable[[], Coroutine[Any, Any, str]]:
        i = 0

        async def factory() -> str:
            nonlocal i
            idx = i
            i += 1
            return await _ok(results[idx], times[idx])

        return factory

    factory = factory_gen()
    result, info = await hedge(factory, delays_s=delays)
    assert info.started_count == 3
    assert result == "backup1"
    assert info.index == 1


# Hedge: accept predicate, first returns empty, second returns non-empty
async def test_hedge_respects_accept_predicate() -> None:
    vals = ["", "ok"]
    times = [0.05, 0.06]

    def factory_gen() -> Callable[[], Coroutine[Any, Any, str]]:
        i = 0

        async def factory() -> str:
            nonlocal i
            idx = i
            i += 1
            return await _ok(vals[idx], times[idx])

        return factory

    factory = factory_gen()
    result, info = await hedge(factory, delays_s=[0.01], accept=_accept_non_empty)
    assert result == "ok"
    assert info.index == 1


async def test_hedge_parent_cancellation_prevents_delayed_branch_start() -> None:
    primary_started = asyncio.Event()
    delayed_started = asyncio.Event()
    block = asyncio.Event()

    def factory_gen() -> Callable[[], Coroutine[Any, Any, str]]:
        started = 0

        async def factory() -> str:
            nonlocal started
            started += 1
            if started == 1:
                primary_started.set()
            else:
                delayed_started.set()

            await block.wait()
            return "done"

        return factory

    task = asyncio.create_task(hedge(factory_gen(), delays_s=[0.05]))

    await asyncio.wait_for(primary_started.wait(), timeout=0.1)
    await asyncio.sleep(0.01)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(delayed_started.wait(), timeout=0.08)
