"""Signal state — persists the latest computed signals across restarts."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

STATE_FILE = Path("/data/signal_state.json")


@dataclass
class SignalState:
    # When the pipeline last ran
    computed_at_utc: str
    # Last trading date for which a signal exists
    as_of_date: str
    # SPY-gate x QQQ ensemble policy
    policy_signal: str          # "risk_on" | "neutral" | "risk_off"
    spy_signal: str
    qqq_signal: str
    # Raw ensemble probabilities from the latest month-end model run
    spy_risk_off_prob: Optional[float]
    spy_jump_in_prob: Optional[float]
    qqq_risk_off_prob: Optional[float]
    qqq_jump_in_prob: Optional[float]
    # VIX-acceleration DCA trigger (double contribution when True in risk_on)
    vix_accel_trigger: bool
    vix_level: Optional[float]
    # Walk-forward training coverage
    spy_model_trained_through: Optional[str]
    qqq_model_trained_through: Optional[str]
    # Rolling 90-day signal history
    history: list[dict[str, Any]]
    # Non-None if the last pipeline run failed
    error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def is_fresh(self, max_age_hours: float = 25.0) -> bool:
        """Return True if the state was computed within *max_age_hours*."""
        try:
            computed = datetime.fromisoformat(self.computed_at_utc)
            if computed.tzinfo is None:
                computed = computed.replace(tzinfo=timezone.utc)
            age_h = (datetime.now(timezone.utc) - computed).total_seconds() / 3600.0
            return age_h <= max_age_hours
        except Exception:
            return False

    def action_label(self) -> str:
        """Human-readable DCA action for the current signal."""
        if self.policy_signal == "risk_on":
            return "DEPLOY 2x — invest at 2× leverage"
        if self.policy_signal == "risk_off":
            return "HOLD CASH — park new contributions, reduce exposure"
        return "HOLD 1x — invest unlevered, no new leverage"


def load_state() -> Optional[SignalState]:
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return SignalState(**data)
    except Exception as exc:
        log.warning("Could not load state file: %s", exc)
        return None


def save_state(data: dict[str, Any]) -> SignalState:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return SignalState(**data)
