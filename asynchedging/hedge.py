"""Asynchronous hedging and racing utilities."""

import asyncio
import contextlib
from collections.abc import Callable, Coroutine, Iterable, Sequence
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    TypeVar,
    overload,
)

T = TypeVar("T")


@dataclass(slots=True)
class WinnerInfo:
    """Information about the winning task in a race or hedge."""

    index: Annotated[int, "Index of the winning factory in the input sequence."]
    started_count: Annotated[
        int,
        (
            "Number of tasks that had started when the winner completed, including the winner. "
            "This will exclude any tasks that were delayed and had not yet started."
        ),
    ]
    elapsed_s: Annotated[float, "Elapsed seconds from the first task start to winner completion."]


class AllTasksFailedError(ExceptionGroup):
    """Raised when every raced task completes with an exception."""


class TaskResultNotAcceptedError[T](Exception):
    """Raised when a task produces a result that is not accepted by the accept function."""

    def __init__(self, result: T) -> None:
        super().__init__("Result not accepted")
        self.result = result


class NoCoroutinesProvidedError(ValueError):
    """Raised when no coroutines are provided to race()."""


class NoDelaysProvidedError(ValueError):
    """Raised when no delays are provided to hedge()."""


@overload
async def race(
    coros: Sequence[Coroutine[Any, Any, T]],
    *,
    accept: Callable[[T], bool] | None = None,
    cancel_pending: bool = True,
    total_timeout_s: float | None = None,
) -> tuple[T, WinnerInfo]: ...


@overload
async def race(
    coros: Sequence[tuple[Coroutine[Any, Any, T], float]],
    *,
    accept: Callable[[T], bool] | None = None,
    cancel_pending: bool = True,
    total_timeout_s: float | None = None,
) -> tuple[T, WinnerInfo]: ...


async def race[T](
    coros: Sequence[tuple[Coroutine[Any, Any, T], float]] | Sequence[Coroutine[Any, Any, T]],
    *,
    accept: Callable[[T], bool] | None = None,
    cancel_pending: bool = True,
    total_timeout_s: float | None = None,
) -> tuple[T, WinnerInfo]:
    """
    Race multiple coroutines concurrently.

    Run multiple coroutines concurrently, returning the result of the first one that completes
    successfully and is accepted by the `accept` function.

    Args:
    coros: A sequence of coroutines to race. Each item can be either a coroutine or a tuple of
             (coroutine, delay), where delay is the number of seconds to wait before starting that
             coroutine.
    accept: An optional function that takes a result and returns True if it is acceptable.
              If None, all results are considered acceptable.
    cancel_pending: If True, cancels all other pending tasks once a winner is found.
    total_timeout_s: An optional total timeout in seconds for the entire race.

    Raises:
    - NoCoroutinesProvidedError: If no coroutines are provided.
    - AllTasksFailedError: If all tasks fail or produce unaccepted results.
    - TimeoutError: If the total timeout is reached before any task completes successfully.

    """
    if not coros:
        raise NoCoroutinesProvidedError

    # This will track when each task started: [index, start_time]
    # where an item is added only when a task starts after a possible delay
    started_at: dict[int, float] = {}

    racing_tasks = [
        asyncio.create_task(_run_race_entry(coro=coro, index=i, delay=delay, accept=accept, started_at=started_at))
        for i, (coro, delay) in enumerate(map(_ensure_coro_has_delay, coros))
    ]

    start_time = asyncio.get_running_loop().time()
    errors: list[Exception] = []

    try:
        async for done in asyncio.as_completed(racing_tasks, timeout=total_timeout_s):
            try:
                index, res = await done
                elapsed = asyncio.get_running_loop().time() - start_time
                if cancel_pending:
                    await _cancel_tasks(racing_tasks, reason="Race won by another task")
                return res, WinnerInfo(index=index, started_count=len(started_at), elapsed_s=elapsed)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
    except TimeoutError:
        await _cancel_tasks(racing_tasks, reason="Total timeout reached")
        raise

    msg = "All tasks failed or were rejected"
    raise AllTasksFailedError(msg, errors)


def _ensure_coro_has_delay[T](
    coros: tuple[Coroutine[Any, Any, T], float] | Coroutine[Any, Any, T],
) -> tuple[Coroutine[Any, Any, T], float]:
    match coros:
        case (coro, delay):
            return coro, delay
        case _:
            return coros, 0.0


async def _run_race_entry(
    *,
    coro: Coroutine[Any, Any, T],
    index: int,
    delay: float,
    accept: Callable[[T], bool] | None,
    started_at: dict[int, float],
) -> tuple[int, T]:
    # Delay, run, and check acceptance
    if delay > 0:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            # If cancelled during delay, close the coro to avoid warnings
            coro.close()
            raise
    started_at[index] = asyncio.get_running_loop().time()
    result = await coro

    def _check_acceptance(res: T) -> None:
        if accept and not accept(res):
            raise TaskResultNotAcceptedError(res)

    try:
        _check_acceptance(result)
    except TaskResultNotAcceptedError:
        raise
    except Exception as e:
        msg = f"Accept function raised: {e}"
        raise TaskResultNotAcceptedError(msg) from e
    return index, result


async def _cancel_tasks(tasks: Iterable[asyncio.Task[tuple[int, Any]]], reason: str) -> None:
    for task in tasks:
        if not task.done():
            task.cancel(reason)
    with contextlib.suppress(Exception):
        await asyncio.gather(*tasks, return_exceptions=True)


# hedge is the same as race except it accpets only one coroutine and it will duplicate it n times
async def hedge(
    factory: Callable[[], Coroutine[Any, Any, T]],
    *,
    delays_s: Sequence[float],
    accept: Callable[[T], bool] | None = None,
    cancel_pending: bool = True,
    total_timeout_s: float | None = None,
) -> tuple[T, WinnerInfo]:
    """
    Hedge multiple instances of a coroutine factory.

    Run multiple instances of a coroutine factory concurrently, returning the result of the first one that completes
    successfully and is accepted by the `accept` function.

    Args:
    factory: A callable that returns a coroutine to be raced.
    delays_s: A sequence of delays in seconds before starting each coroutine instance.
    accept: An optional function that takes a result and returns True if it is acceptable.
              If None, all results are considered acceptable.
    cancel_pending: If True, cancels all other pending tasks once a winner is found.
    total_timeout_s: An optional total timeout in seconds for the entire hedge.

    Raises:
    NoDelaysProvidedError: If no delays are provided.
    AllTasksFailedError: If all tasks fail or produce unaccepted results.
    TimeoutError: If the total timeout is reached before any task completes successfully.

    """
    if not delays_s:
        raise NoDelaysProvidedError

    return await race(
        coros=[(factory(), 0.0)] + [(factory(), delay) for delay in delays_s],
        accept=accept,
        cancel_pending=cancel_pending,
        total_timeout_s=total_timeout_s,
    )
