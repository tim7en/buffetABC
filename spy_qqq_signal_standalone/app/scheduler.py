"""APScheduler daily signal refresh — runs inside uvicorn's asyncio event loop.

Schedule: every trading day at 17:30 ET (22:30 UTC) — 30 minutes after the
NYSE close, giving FRED enough time to publish same-day data.

Using AsyncIOScheduler avoids spawning a dedicated OS thread for the scheduler
itself, which keeps this compatible with resource-constrained environments
(e.g. Docker Toolbox on VirtualBox).
"""

from __future__ import annotations

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .pipeline import run_full_refresh
from .state import save_state

log = logging.getLogger(__name__)

_is_refreshing = False


def _is_refreshing_now() -> bool:
    return _is_refreshing


def _run_refresh_sync(triggered_by: str) -> None:
    """Synchronous refresh worker (called from executor or directly)."""
    global _is_refreshing
    if _is_refreshing:
        log.info("Refresh already in progress — skipping duplicate trigger.")
        return
    _is_refreshing = True
    try:
        log.info("Starting refresh (triggered by %s) …", triggered_by)
        state_dict = run_full_refresh()
        save_state(state_dict)
        signal = state_dict.get("policy_signal", "unknown")
        as_of = state_dict.get("as_of_date", "?")
        log.info("Refresh complete: signal=%s as_of=%s", signal, as_of)
    except Exception:
        log.exception("Scheduled refresh failed.")
    finally:
        _is_refreshing = False


async def _do_refresh_async(triggered_by: str = "scheduler") -> None:
    """Async entry-point: runs the sync refresh via executor, with fallback."""
    loop = asyncio.get_event_loop()
    try:
        # Prefer non-blocking thread pool execution
        await loop.run_in_executor(None, _run_refresh_sync, triggered_by)
    except RuntimeError as exc:
        if "can't start new thread" in str(exc):
            # Thread pool unavailable (Docker Toolbox / resource-constrained VM)
            # Run synchronously — event loop blocks during refresh (~2-15 min)
            log.warning(
                "Thread pool unavailable; running refresh synchronously "
                "(API will be unresponsive during refresh)."
            )
            _run_refresh_sync(triggered_by)
        else:
            raise


def trigger_refresh_background(reason: str = "manual") -> None:
    """Schedule an async refresh task on the running event loop."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_do_refresh_async(reason))
    except RuntimeError:
        log.warning("No running event loop for background refresh trigger")


def create_scheduler() -> AsyncIOScheduler:
    """Build and return the AsyncIOScheduler instance (not yet started)."""
    scheduler = AsyncIOScheduler(timezone="UTC")
    # 17:30 ET = 22:30 UTC year-round (conservative — data always available).
    scheduler.add_job(
        func=_do_refresh_async,
        args=["scheduler"],
        trigger=CronTrigger(hour=22, minute=30, timezone="UTC"),
        id="daily_signal_refresh",
        name="Daily SPY-gate × QQQ signal refresh",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    return scheduler
