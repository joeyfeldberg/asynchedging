# Asynchedging: Async Racing And Hedging

This package provides tiny `asyncio` utilities for:

- Race multiple calls, optionally with per-call start delays, and return the first acceptable result, cancelling the rest
- Hedge a single call by starting backups immediately or after delays, returning the first winner

## Install

```bash
pip install asynchedging
```

## Demo

Run the included demo (sleep-based simulations, no network required):

```bash
uv run python main.py
```

## Example of using asynchedging with OpenAI

```python
import asyncio
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI
from asynchedging import WinnerInfo, hedge, race


client = AsyncOpenAI()

async def call_openai(prompt: str, model: str) -> Awaitable[Any]:
    return await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )


async def main() -> None:
    # Race two OpenAI models. race(...) can accept a list of (awaitable, delay) pairs
    result, winner_info = await race([
        (call_openai("Why is the sky blue?", model="gpt-4o"), 0.0),  # starts immediately but takes ~0.80s to finish
        (call_openai("Why is the sky blue?", model="gpt-4o-mini"), 0.25),  # starts at 0.25s but takes ~0.65s to finish (likely winner)
    ])
    print(result)  # OpenAI response object
    print(
        f"meta: index={winner_info.index} started={winner_info.started_count} elapsed={winner_info.elapsed_s:.3f}s"
    )

    # Race two OpenAI models without delays, first to finish wins and the rest are cancelled
    result, winner_info = await race([
        call_openai("Why is the sky blue?", model="gpt-4o"),
        call_openai("Why is the sky blue?", model="gpt-4o-mini"),
    ])

    # Hedge one OpenAI call (start backups if slow) — delays are required (can be zero)
    # Each delay will become the start time for the corresponding backup call
    factory = call_openai("Another question", model="gpt-4o-mini")
    result, winner_info = await hedge(factory, delays_s=(0.25, 0.75))


asyncio.run(main())
```

Notes:

- Use `accept=` to validate answers (e.g., non-empty content) and keep waiting if unacceptable
- Use `total_timeout_s=` to set a hard cap for the entire race/hedge
- Pending tasks are cancelled when a winner is chosen
- `race(...)` accepts either awaitables or (awaitable, delay) pairs, `hedge(...)` requires `delays_s` for replicas (delays can be zero)
