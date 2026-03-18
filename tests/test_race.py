import asyncio

import pytest

from asynchedging.hedge import AllTasksFailedError, race

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


# All tasks fail which should raise AllTasksFailedError
async def test_race_all_fail_raises_with_errors() -> None:
    with pytest.raises(AllTasksFailedError) as exc:
        await race([(_fail(0.05), 0.0), (_fail(0.01), 0.0)])

    assert len(exc.value.exceptions) == 2, "Expected two exceptions in AllTasksFailedError"


# Total timeout occurs before any task can complete
# Total timeout is set to 0.1, but both tasks will take at least 1 second to complete
async def test_race_total_timeout() -> None:
    with pytest.raises(TimeoutError):
        await race([(_ok("A", 1.0), 0.0), (_ok("B", 1.2), 0.0)], total_timeout_s=0.1)


# Race with multiple factories and explicit delays
# The fastest task should win
async def test_race_multiple_factories_with_stagger() -> None:
    # backup1 is the fastest despite starting later
    # all three tasks start before the first completes
    coros_with_delays = [
        (_ok("primary", 1.0), 0.0),
        (_ok("backup1", 0.4), 0.4),
        (_ok("backup2", 0.8), 0.4),
    ]
    result, info = await race(coros_with_delays)
    assert info.started_count == 3, "Expected three tasks to start"
    assert result == "backup1", "Expected backup1 to win"
    assert info.index == 1, "Expected index of winning task to be 1"

    # backup1 is the fastest despite starting later
    # only primary and backup1 start before the first completes
    coros_with_delays = [
        (_ok("primary", 1.0), 0.0),
        (_ok("backup1", 0.4), 0.4),
        (_ok("backup2", 0.8), 1.0),
    ]
    result, info = await race(coros_with_delays)
    assert info.started_count == 2, "Expected two tasks to start"
    assert result == "backup1", "Expected backup1 to win"
    assert info.index == 1, "Expected index of winning task to be 1"


# Only accepts non-empty results
# First task is fast but returns empty
# Second task is slightly slower but returns non-empty
# We expect the second task to be the winner
async def test_race_respects_accept_predicate() -> None:
    # First returns empty, second returns non-empty
    coros_with_delays = [(_ok("", 0.05), 0.0), (_ok("ok", 0.06), 0.0)]
    result, info = await race(coros_with_delays, accept=_accept_non_empty)
    assert result == "ok", "Expected 'ok' to be the winning result"
    assert info.index == 1, "Expected index of winning task to be 1"


async def test_race_parent_cancellation_cancels_all_started_children() -> None:
    all_started = asyncio.Event()
    all_cancelled = asyncio.Event()
    started: list[str] = []
    cancelled: list[str] = []
    block = asyncio.Event()

    async def worker(name: str) -> str:
        started.append(name)
        if len(started) == 3:
            all_started.set()

        try:
            await block.wait()
        except asyncio.CancelledError:
            cancelled.append(name)
            if len(cancelled) == 3:
                all_cancelled.set()
            raise

        return name

    task = asyncio.create_task(race([worker("a"), worker("b"), worker("c")]))

    await asyncio.wait_for(all_started.wait(), timeout=0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(all_cancelled.wait(), timeout=0.1)
    assert sorted(cancelled) == ["a", "b", "c"]


async def test_race_success_keeps_pending_tasks_when_cancel_pending_disabled() -> None:
    loser_started = asyncio.Event()
    loser_completed = asyncio.Event()
    loser_cancelled = asyncio.Event()
    allow_loser_to_finish = asyncio.Event()

    async def winner() -> str:
        await asyncio.sleep(0)
        return "winner"

    async def loser() -> str:
        loser_started.set()
        try:
            await allow_loser_to_finish.wait()
        except asyncio.CancelledError:
            loser_cancelled.set()
            raise

        loser_completed.set()
        return "loser"

    result, info = await race([winner(), loser()], cancel_pending=False)

    assert result == "winner"
    assert info.index == 0
    assert loser_started.is_set()
    assert not loser_cancelled.is_set()

    allow_loser_to_finish.set()
    await asyncio.wait_for(loser_completed.wait(), timeout=0.1)


async def test_race_timeout_cancels_pending_children() -> None:
    cancelled = 0
    all_cancelled = asyncio.Event()
    block = asyncio.Event()

    async def worker() -> str:
        nonlocal cancelled
        try:
            await block.wait()
        except asyncio.CancelledError:
            cancelled += 1
            if cancelled == 2:
                all_cancelled.set()
            raise

        return "done"

    with pytest.raises(TimeoutError):
        await race([worker(), worker()], total_timeout_s=0.01)

    await asyncio.wait_for(all_cancelled.wait(), timeout=0.1)
