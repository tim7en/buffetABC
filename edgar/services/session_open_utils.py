from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


SESSION_OPEN_UTC: dict[str, tuple[int, int]] = {
    "tokyo_open": (0, 0),
    "asia_open": (0, 0),
    "hong_kong_open": (1, 30),
}
_NY_TZ = ZoneInfo("America/New_York")


@dataclass
class SessionBar:
    anchor: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def bars_per_day_24x7(interval: str) -> int:
    text = (interval or "").strip().lower()
    if text.endswith("m") and text[:-1].isdigit():
        minutes = max(int(text[:-1]), 1)
    elif text.endswith("h") and text[:-1].isdigit():
        minutes = max(int(text[:-1]) * 60, 1)
    else:
        minutes = 60
    return max(int((24 * 60) / minutes), 1)


def session_anchor_for_ts(ts: datetime, session_open: str) -> datetime:
    if session_open == "new_york_equity_open":
        aware_utc = ts.replace(tzinfo=timezone.utc)
        local = aware_utc.astimezone(_NY_TZ)
        anchor_local = local.replace(hour=9, minute=30, second=0, microsecond=0)
        if (local.hour, local.minute) < (9, 30):
            anchor_local -= timedelta(days=1)
        return anchor_local.astimezone(timezone.utc).replace(tzinfo=None)
    if session_open not in SESSION_OPEN_UTC:
        raise ValueError(f"session_open must be one of {sorted(list(SESSION_OPEN_UTC) + ['new_york_equity_open'])}")
    open_hour, open_minute = SESSION_OPEN_UTC[session_open]
    anchor = ts.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0)
    if (ts.hour, ts.minute) < (open_hour, open_minute):
        anchor -= timedelta(days=1)
    return anchor


def minutes_since_session_open(ts: datetime, session_open: str) -> int:
    anchor = session_anchor_for_ts(ts, session_open)
    return int((ts - anchor).total_seconds() / 60)


def aggregate_session_bars(bars: list[dict], session_open: str) -> tuple[list[SessionBar], list[int]]:
    sessions: list[SessionBar] = []
    bar_to_session_idx: list[int] = []
    current_anchor: datetime | None = None
    current: SessionBar | None = None

    for bar in bars:
        anchor = session_anchor_for_ts(bar["timestamp"], session_open)
        if anchor != current_anchor:
            if current is not None:
                sessions.append(current)
            current_anchor = anchor
            current = SessionBar(
                anchor=anchor,
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                volume=float(bar.get("volume", 0.0) or 0.0),
            )
        else:
            current.high = max(float(current.high), float(bar["high"]))
            current.low = min(float(current.low), float(bar["low"]))
            current.close = float(bar["close"])
            current.volume = float(current.volume) + float(bar.get("volume", 0.0) or 0.0)
        bar_to_session_idx.append(len(sessions))

    if current is not None:
        sessions.append(current)
    return sessions, bar_to_session_idx
