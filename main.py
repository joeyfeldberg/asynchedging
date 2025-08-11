"""Demo of asynchedging's hedge and race functions."""

import asyncio
from typing import Coroutine, Tuple

from asynchedging import WinnerInfo, hedge, race


async def simulate(name: str, delay_s: float, *, fail: bool = False) -> str:
    await asyncio.sleep(delay_s)
    if fail:
        raise RuntimeError(f"{name} failed")
    return f"{name} finished in {delay_s:.2f}s"


async def demo_hedge() -> Tuple[str, WinnerInfo]:
    async def factory() -> str:
        return await simulate("A", 0.6)

    return await hedge(factory, delays_s=(0.25, 0.75), total_timeout_s=5.0)


async def demo_race() -> Tuple[str, WinnerInfo]:
    coros = [simulate("A", 0.6), simulate("B", 0.3), simulate("C", 0.9)]
    return await race(coros, total_timeout_s=5.0)


async def demo_race_with_staggered_starts() -> Tuple[str, WinnerInfo]:
    # Multiple different providers, staggered starts, backup1 should win
    def provider(name: str, delay: float) -> Coroutine[object, object, str]:
        async def call() -> str:
            print(f"starting {name}")
            return await simulate(name, delay)

        return call()

    coros_with_delays = [
        (provider("primary", 0.8), 0.0),  # starts at t=0.00, finishes ~0.80
        (provider("backup1", 0.4), 0.25),  # starts at t=0.25, finishes ~0.65 (winner)
        (provider("backup2", 0.5), 0.50),  # starts at t=0.50, finishes ~1.00
    ]
    return await race(coros_with_delays, total_timeout_s=5.0)


async def run() -> None:
    print("-- hedge demo --")
    r, info = await demo_hedge()
    print(f"winner: {r}")
    print(f"meta: index={info.index} started={info.started_count} elapsed={info.elapsed_s:.3f}s")

    print("-- race demo --")
    r, info = await demo_race()
    print(f"winner: {r}")
    print(f"meta: index={info.index} started={info.started_count} elapsed={info.elapsed_s:.3f}s")

    print("\n-- race with staggered starts demo --")
    r2, info2 = await demo_race_with_staggered_starts()
    print(f"winner: {r2}")
    print(f"meta: index={info2.index} started={info2.started_count} elapsed={info2.elapsed_s:.3f}s")


if __name__ == "__main__":
    asyncio.run(run())
