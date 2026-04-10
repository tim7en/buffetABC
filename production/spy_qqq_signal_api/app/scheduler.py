"""APScheduler background job — daily signal refresh.

Schedule: every trading day at 17:30 ET (22:30 UTC) — 30 minutes after the
NYSE close, giving FRED enough time to publish same-day data.

On weekends / US market holidays the job still runs but the price data will
be unchanged, so the signal is stable until Monday.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .pipeline import run_full_refresh
from .state import save_state

log = logging.getLogger(__name__)

# Shared flag so the API can report whether a refresh is in progress
_refresh_lock = threading.Lock()
_is_refreshing = False


def _is_refreshing_now() -> bool:
    return _is_refreshing


def _do_refresh(triggered_by: str = "scheduler") -> None:
    global _is_refreshing
    with _refresh_lock:
        if _is_refreshing:
            log.info("Refresh already in progress — skipping duplicate trigger.")
            return
        _is_refreshing = True

    try:
        log.info("Starting scheduled refresh (triggered by %s) …", triggered_by)
        state_dict = run_full_refresh()
        save_state(state_dict)
        signal = state_dict.get("policy_signal", "unknown")
        as_of = state_dict.get("as_of_date", "?")
        log.info("Refresh complete: signal=%s as_of=%s", signal, as_of)
    except Exception:
        log.exception("Scheduled refresh failed.")
    finally:
        _is_refreshing = False


def trigger_refresh_background(reason: str = "manual") -> None:
    """Fire a refresh in a daemon thread (returns immediately)."""
    t = threading.Thread(target=_do_refresh, kwargs={"triggered_by": reason}, daemon=True)
    t.start()


def create_scheduler() -> BackgroundScheduler:
    """Build and return the APScheduler instance (not yet started)."""
    scheduler = BackgroundScheduler(timezone="UTC")
    # 17:30 ET = 22:30 UTC (21:30 UTC during US Daylight Saving Time)
    # Using 22:30 UTC year-round is conservative — data is always available by then.
    scheduler.add_job(
        func=lambda: _do_refresh("scheduler"),
        trigger=CronTrigger(hour=22, minute=30, timezone="UTC"),
        id="daily_signal_refresh",
        name="Daily SPY-gate × QQQ signal refresh",
        replace_existing=True,
        misfire_grace_time=3600,    # run even if the server was briefly down
    )
    return scheduler
