"""The F1 calendar, as the tool sees it: what is on now, what is next, what just ended.

Sessions come from the OpenF1 index (UTC start/end for every session of the
year, including sprints and qualifying), cached under `data/raw/` and refreshed
in the background.  Each session is mapped to the weekend it belongs to
(`src.config.EVENTS`) so the rest of the tool knows which model to load.

A session counts as **live** from `LIVE_LEAD_MIN` minutes before its scheduled
start until `LIVE_TAIL_MIN` minutes after its scheduled end — the feed carries
data on both sides of the flag, and a red-flag delay pushes the end out, which
the daemon detects from the session status rather than the clock.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from src.config import DATA_RAW, EVENTS, OPENF1_SESSIONS_FILE, Event

log = logging.getLogger("degless.schedule")

LIVE_LEAD_MIN = 15
LIVE_TAIL_MIN = 25
REFRESH_S = 6 * 3600


@dataclass
class Session:
    key: int
    meeting_key: int
    name: str            # "Practice 1" | "Qualifying" | "Sprint" | "Race" ...
    type: str            # "Practice" | "Qualifying" | "Race"
    start: datetime
    end: datetime
    circuit: str
    country: str
    event: Event | None  # the weekend in the registry, if known

    @property
    def event_key(self) -> str | None:
        return self.event.key if self.event else None

    @property
    def label(self) -> str:
        where = self.event.name if self.event else f"{self.circuit} {self.start.year}"
        return f"{where} · {self.name}"

    @property
    def is_race(self) -> bool:
        return self.name in ("Race", "Sprint")

    def state(self, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        if now < self.start - timedelta(minutes=LIVE_LEAD_MIN):
            return "upcoming"
        if now <= self.end + timedelta(minutes=LIVE_TAIL_MIN):
            return "live"
        return "finished"

    def as_dict(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        return {
            "key": self.key, "meeting_key": self.meeting_key, "name": self.name, "type": self.type,
            "start": self.start.isoformat(), "end": self.end.isoformat(),
            "circuit": self.circuit, "country": self.country, "event_key": self.event_key,
            "label": self.label, "state": self.state(now), "is_race": self.is_race,
            "minutes_to_start": (self.start - now).total_seconds() / 60,
            "minutes_to_end": (self.end - now).total_seconds() / 60,
        }


def _parse(dt: str) -> datetime:
    d = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def refresh(year: int | None = None, *, timeout: int = 20) -> bool:
    """Re-download the OpenF1 session index.  Returns True on success."""
    year = year or datetime.now(timezone.utc).year
    try:
        r = requests.get("https://api.openf1.org/v1/sessions", params={"year": year}, timeout=timeout)
        r.raise_for_status()
        rows = r.json()
    except Exception as exc:
        log.warning("could not refresh the session index: %s", exc)
        return False
    if not rows:
        return False
    Path(OPENF1_SESSIONS_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(OPENF1_SESSIONS_FILE).write_text(json.dumps(rows))
    return True


def load(*, refresh_if_older_s: float = REFRESH_S) -> list:
    p = Path(OPENF1_SESSIONS_FILE)
    if not p.exists() or time.time() - p.stat().st_mtime > refresh_if_older_s:
        refresh()
    if not p.exists():
        return []
    rows = json.loads(p.read_text())
    by_meeting = {e.meeting_key: e for e in EVENTS.values()}
    out = []
    for r in rows:
        try:
            out.append(Session(
                key=int(r["session_key"]), meeting_key=int(r["meeting_key"]),
                name=str(r["session_name"]), type=str(r.get("session_type") or ""),
                start=_parse(r["date_start"]), end=_parse(r["date_end"]),
                circuit=str(r.get("circuit_short_name") or ""), country=str(r.get("country_name") or ""),
                event=by_meeting.get(int(r["meeting_key"])),
            ))
        except Exception:
            continue
    out.sort(key=lambda s: s.start)
    return out


def status(now: datetime | None = None, sessions: list | None = None) -> dict:
    """What is on: the live session (if any), the next one, the last one, and
    the sessions of the current or next weekend."""
    now = now or datetime.now(timezone.utc)
    sessions = sessions if sessions is not None else load()
    live = next((s for s in sessions if s.state(now) == "live"), None)
    upcoming = [s for s in sessions if s.state(now) == "upcoming"]
    finished = [s for s in sessions if s.state(now) == "finished"]
    nxt = upcoming[0] if upcoming else None
    last = finished[-1] if finished else None
    focus = live or nxt or last
    weekend = [s for s in sessions if focus and s.meeting_key == focus.meeting_key]
    return {
        "now": now.isoformat(),
        "live": live.as_dict(now) if live else None,
        "next": nxt.as_dict(now) if nxt else None,
        "last": last.as_dict(now) if last else None,
        "event_key": focus.event_key if focus else None,
        "weekend": [s.as_dict(now) for s in weekend],
    }


def describe(st: dict) -> str:
    """One line for a status strip."""
    if st.get("live"):
        s = st["live"]
        left = s["minutes_to_end"]
        if s["minutes_to_start"] > 0:
            return f"LIVE SOON · {s['label']} starts in {s['minutes_to_start']:.0f} min"
        if left > 0:
            return f"LIVE · {s['label']} · {left:.0f} min to the scheduled end"
        return f"LIVE · {s['label']} · past the scheduled end (running late or finishing)"
    if st.get("next"):
        s = st["next"]
        m = s["minutes_to_start"]
        if m < 90:
            return f"Next: {s['label']} in {m:.0f} min"
        if m < 48 * 60:
            return f"Next: {s['label']} in {m/60:.1f} h"
        return f"Next: {s['label']} in {m/1440:.1f} days"
    return "No session scheduled"
