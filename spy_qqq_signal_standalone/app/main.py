"""FastAPI application — SPY-gate × QQQ ensemble DCA signal API.

Endpoints:
  GET  /              → redirect to /docs
  GET  /health        → system status + last refresh time
  GET  /signal        → today's policy signal with all probabilities
  GET  /signal/history?n=30  → last N daily signals (default 30, max 90)
  POST /refresh       → manually trigger a full data + signal refresh
                        (requires Authorization: Bearer <REFRESH_TOKEN>)

Environment variables:
  DATA_DIR        path to persistent volume (default /data)
  REFRESH_TOKEN   secret used to authenticate POST /refresh (default: none,
                  meaning the endpoint is open — set this in production!)
  MAX_STATE_AGE_H how many hours before the signal is considered stale (default 25)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .scheduler import create_scheduler, trigger_refresh_background
from .state import SignalState, load_state, save_state

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
REFRESH_TOKEN: Optional[str] = os.environ.get("REFRESH_TOKEN")
MAX_STATE_AGE_H: float = float(os.environ.get("MAX_STATE_AGE_H", "25"))

# --------------------------------------------------------------------------- #
# Lifespan: start scheduler + warm-up on startup
# --------------------------------------------------------------------------- #
_scheduler = create_scheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _scheduler.start()
    log.info("APScheduler started.")

    # If no fresh state exists, kick off a background refresh so the first
    # API call isn't blocked.  The initial run takes ~10–15 minutes.
    state = load_state()
    if state is None or not state.is_fresh(MAX_STATE_AGE_H):
        log.info("State is missing or stale — triggering background refresh on startup …")
        trigger_refresh_background("startup")
    else:
        log.info("State is fresh (as_of=%s) — skipping startup refresh.", state.as_of_date)

    yield

    _scheduler.shutdown(wait=False)
    log.info("Scheduler shut down.")


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="SPY-gate × QQQ Ensemble DCA Signal API",
    description=(
        "Daily DCA signal (risk_on | neutral | risk_off) derived from a "
        "walk-forward SPY-gated QQQ ensemble blend of logistic regression and "
        "random forest models trained on 40+ macro features. "
        "Backtest: $1 in 2014 grew to ~$8.50 by 2024 vs $6.20 for buy-and-hold QQQ "
        "at 2× leverage.\n\n"
        "**Signal meaning:**\n"
        "- `risk_on` → deploy at 2× leverage (e.g. QLD or 2× QQQ)\n"
        "- `neutral` → hold 1× unlevered QQQ\n"
        "- `risk_off` → park new contributions in cash; reduce existing 2× QQQ exposure gradually\n\n"
        "Signals are updated daily at ~17:30 ET after market close."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_bearer = HTTPBearer(auto_error=False)


def _require_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> None:
    """Validate the Bearer token for write endpoints.  If REFRESH_TOKEN is not
    set the endpoint is open (useful for local dev)."""
    if not REFRESH_TOKEN:
        return
    if credentials is None or credentials.credentials != REFRESH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token.",
        )


def _get_state_or_404() -> SignalState:
    state = load_state()
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Signal not yet available — the pipeline is still running for "
                "the first time.  Try again in ~15 minutes."
            ),
        )
    return state


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    summary="System health",
    tags=["meta"],
)
def health() -> dict[str, Any]:
    """Returns the current system status and when the signal was last computed."""
    from .scheduler import _is_refreshing_now  # local import avoids circular
    state = load_state()
    if state is None:
        return {
            "status": "initializing",
            "signal_ready": False,
            "refresh_in_progress": _is_refreshing_now(),
            "last_computed_utc": None,
            "as_of_date": None,
            "state_age_hours": None,
            "is_stale": True,
        }

    try:
        computed = datetime.fromisoformat(state.computed_at_utc)
        if computed.tzinfo is None:
            computed = computed.replace(tzinfo=timezone.utc)
        age_h = round((datetime.now(timezone.utc) - computed).total_seconds() / 3600, 2)
    except Exception:
        age_h = None

    return {
        "status": "ok" if state.error is None else "error",
        "signal_ready": True,
        "refresh_in_progress": _is_refreshing_now(),
        "last_computed_utc": state.computed_at_utc,
        "as_of_date": state.as_of_date,
        "state_age_hours": age_h,
        "is_stale": not state.is_fresh(MAX_STATE_AGE_H),
        "last_error": state.error,
    }


@app.get(
    "/signal",
    summary="Today's DCA signal",
    tags=["signal"],
)
def get_signal() -> dict[str, Any]:
    """Returns the current DCA signal with full model details.

    - **policy_signal**: the final gated signal to act on
    - **spy_signal**: SPY regime (the broad-market gate)
    - **qqq_signal**: QQQ ensemble regime (the expression vehicle)
    - **vix_accel_trigger**: when True, double your DCA contribution this month
    - **probabilities**: raw model outputs for transparency
    """
    state = _get_state_or_404()
    stale_warning = (
        "Signal is stale — data refresh may have failed." if not state.is_fresh(MAX_STATE_AGE_H) else None
    )

    return {
        "as_of_date": state.as_of_date,
        "policy_signal": state.policy_signal,
        "action": state.action_label() if state.error is None else "ERROR — see error field",
        "spy_signal": state.spy_signal,
        "qqq_signal": state.qqq_signal,
        "vix_accel_trigger": state.vix_accel_trigger,
        "vix_level": state.vix_level,
        "probabilities": {
            "spy_risk_off": state.spy_risk_off_prob,
            "spy_jump_in": state.spy_jump_in_prob,
            "qqq_risk_off": state.qqq_risk_off_prob,
            "qqq_jump_in": state.qqq_jump_in_prob,
        },
        "model_info": {
            "spy_trained_through": state.spy_model_trained_through,
            "qqq_trained_through": state.qqq_model_trained_through,
            "risk_off_threshold": 0.45,
            "jump_in_threshold": 0.55,
            "leverage_on_risk_on": "2×",
        },
        "computed_at_utc": state.computed_at_utc,
        "stale_warning": stale_warning,
        "error": state.error,
    }


@app.get(
    "/signal/history",
    summary="Recent daily signals",
    tags=["signal"],
)
def get_signal_history(n: int = 30) -> dict[str, Any]:
    """Returns the last *n* daily signals (max 90, sorted newest-first)."""
    n = max(1, min(n, 90))
    state = _get_state_or_404()
    history = list(reversed(state.history[-n:]))
    return {
        "as_of_date": state.as_of_date,
        "count": len(history),
        "history": history,
    }


@app.get(
    "/notice",
    summary="Single ENTER / HOLD / EXIT trade notice",
    tags=["signal"],
)
def get_notice() -> dict[str, Any]:
    """Returns a single, plain-language trade notice for the SPY-gated QQQ 2× DCA strategy.

    **notice** is always one of:
    - ``ENTER`` — SPY gate is open and QQQ regime is positive: deploy 2× leverage (DCA full size)
    - ``ENTER — DOUBLE DCA`` — same as ENTER but VIX-acceleration trigger is active: double the contribution
    - ``HOLD`` — SPY gate is open but QQQ regime is neutral: stay 1× unlevered, no new 2× leverage
    - ``REDUCE`` — SPY circuit-breaker fired: park new DCA contributions and reduce (do NOT fully liquidate) existing 2× leveraged QQQ positions

    Signals are recomputed daily at ~17:30 ET. Poll ``GET /health`` to confirm freshness.
    """
    state = _get_state_or_404()

    # Map the three-state policy to a binary ENTER / HOLD / EXIT direction
    if state.policy_signal == "risk_on":
        if state.vix_accel_trigger:
            notice = "ENTER — DOUBLE DCA"
            detail = (
                "VIX elevated and rising while QQQ dips → double your DCA contribution. "
                "Deploy at 2× leverage (QLD or 2× QQQ)."
            )
        else:
            notice = "ENTER"
            detail = "SPY gate open, QQQ regime positive. Deploy at 2× leverage (DCA full size)."
    elif state.policy_signal == "risk_off":
        notice = "REDUCE"
        detail = (
            "SPY circuit-breaker active. Park new DCA contributions and gradually reduce "
            "existing 2× leveraged QQQ exposure. Do NOT liquidate the full portfolio — "
            "hold remaining positions and resume deploying when SPY regime recovers."
        )
    else:  # neutral
        notice = "HOLD"
        detail = (
            "SPY gate is open but QQQ regime is neutral. "
            "Maintain 1× unlevered QQQ; do not add new 2× leverage."
        )

    stale_warning = (
        "Signal is stale — data refresh may have failed."
        if not state.is_fresh(MAX_STATE_AGE_H)
        else None
    )

    return {
        "as_of_date": state.as_of_date,
        "notice": notice,
        "detail": detail,
        "spy_signal": state.spy_signal,
        "qqq_signal": state.qqq_signal,
        "vix_accel_trigger": state.vix_accel_trigger,
        "vix_level": state.vix_level,
        "computed_at_utc": state.computed_at_utc,
        "is_stale": stale_warning is not None,
        "stale_warning": stale_warning,
    }


@app.post(
    "/refresh",
    summary="Trigger a full data + signal refresh",
    tags=["admin"],
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_refresh(_: None = Depends(_require_token)) -> dict[str, str]:
    """Kick off a background refresh (download latest data, retrain, recompute signal).

    The job runs asynchronously — poll ``GET /health`` to see when it finishes.
    Requires ``Authorization: Bearer <REFRESH_TOKEN>`` if ``REFRESH_TOKEN`` is set.
    """
    from .scheduler import _is_refreshing_now
    if _is_refreshing_now():
        return {"status": "already_running", "message": "A refresh is already in progress."}
    trigger_refresh_background("api")
    return {"status": "accepted", "message": "Refresh started. Poll /health for status."}
