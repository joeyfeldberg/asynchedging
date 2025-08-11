"""Public API for the asynchedging package."""

from .hedge import (
    AllTasksFailedError,
    WinnerInfo,
    hedge,
    race,
)

__all__ = [
    "AllTasksFailedError",
    "WinnerInfo",
    "hedge",
    "race",
]

__version__ = "1.0.0"
