"""Committed plans: what the wall has decided, written down where the live view can see it.

The outlook and the Strategy desk produce candidates; a strategist commits
one - per driver, or one default for the team - and from then on the live
race view measures the car against it: is the planned stop still inside its
window, has the live degradation multiplier crossed the trigger at which the
alternative becomes faster, is the undercut exposure the card warned about
now real.  Plans live under `data/live/plans/<event>.json`, one file per
weekend, so they survive an app restart and a feed restart alike.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.live.store import LIVE_DIR, atomic_write

PLANS_DIR = LIVE_DIR / "plans"
COMPOUND_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H"}


def _path(event_key: str) -> Path:
    return PLANS_DIR / f"{event_key}.json"


def load_plans(event_key: str) -> list:
    p = _path(event_key)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text()).get("plans", [])
    except Exception:
        return []


def save_plans(event_key: str, plans: list) -> None:
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(_path(event_key), json.dumps({"event": event_key, "plans": plans}, indent=1, default=str))


def short_label(compounds, pit_laps) -> str:
    seq = "-".join(COMPOUND_LETTER.get(str(c).upper(), str(c)[0]) for c in compounds)
    pits = ",".join(str(int(p)) for p in pit_laps)
    return f"{len(pit_laps)}-stop {seq}" + (f" @ {pits}" if pits else "")


def commit_plan(event_key: str, *, compounds: list, pit_laps: list, push: float,
                driver: str | None = None, label: str | None = None, note: str = "",
                windows: list | None = None, triggers: dict | None = None,
                alternative: dict | None = None, source: str = "", n_race_laps: int | None = None) -> dict:
    """Commit (or replace) the plan for a driver; `driver=None` is the team default."""
    plans = [p for p in load_plans(event_key) if (p.get("driver") or None) != (driver or None)]
    lens = []
    if n_race_laps:
        bounds = [0, *[int(x) for x in pit_laps], int(n_race_laps)]
        lens = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
    plan = {
        "id": uuid.uuid4().hex[:8], "driver": driver, "label": label or short_label(compounds, pit_laps),
        "compounds": [str(c).upper() for c in compounds], "pit_laps": [int(x) for x in pit_laps],
        "stint_lens": lens, "push": float(push), "note": note, "windows": windows or [],
        "triggers": triggers or {}, "alternative": alternative or {}, "source": source,
        "committed_utc": datetime.now(timezone.utc).isoformat(),
    }
    plans.append(plan)
    save_plans(event_key, plans)
    return plan


def remove_plan(event_key: str, plan_id: str) -> None:
    save_plans(event_key, [p for p in load_plans(event_key) if p.get("id") != plan_id])


def plan_for_driver(plans: list, driver: str | None) -> dict | None:
    """The driver's own committed plan, else the team default, else None."""
    if driver:
        for p in plans:
            if (p.get("driver") or "").upper() == str(driver).upper():
                return p
    for p in plans:
        if not p.get("driver"):
            return p
    return None


def next_stop(plan: dict, lap_now: int) -> tuple:
    """`(index, lap)` of the first planned stop not yet reached, or `(None, None)`."""
    for i, p in enumerate(plan.get("pit_laps", [])):
        if int(p) >= int(lap_now):
            return i, int(p)
    return None, None


def as_markdown(plan: dict, event_name: str, n_race_laps: int) -> str:
    """The decision card as text, for printing or pasting into the team chat."""
    comps = plan.get("compounds", [])
    pits = plan.get("pit_laps", [])
    bounds = [0, *pits, n_race_laps]
    lines = [f"# {event_name} — decision card" + (f" · {plan['driver']}" if plan.get("driver") else " · team default"),
             "",
             f"**Plan A: {plan.get('label', '')}** · push {plan.get('push', 1.0):.2f}"
             + (f" · committed {plan['committed_utc'][:16].replace('T', ' ')} UTC" if plan.get("committed_utc") else ""),
             ""]
    for k, c in enumerate(comps):
        a, b = bounds[k] + 1, bounds[k + 1]
        lines.append(f"- Stint {k + 1}: **{c}** laps {a}–{b} ({b - a + 1} laps)")
    wins = {w["stop"]: w for w in plan.get("windows", [])}
    if pits:
        lines += ["", "## Stops"]
        for i, p in enumerate(pits):
            w = wins.get(i + 1)
            lines.append(f"- Stop {i + 1}: lap **{p}**"
                         + (f", window {w['lo']}–{w['hi']} (within 1 s of optimal)" if w else "")
                         + f" → {comps[i + 1]}")
    tr = plan.get("triggers") or {}
    if tr:
        lines += ["", "## Triggers"]
        for k, v in tr.items():
            lines.append(f"- {v}" if isinstance(v, str) else f"- {k}: {v}")
    alt = plan.get("alternative") or {}
    if alt:
        lines += ["", f"## Plan B: {alt.get('label', '')}"]
        if alt.get("delta_s") is not None:
            lines.append(f"- {alt['delta_s']:+.1f} s against Plan A in expectation"
                         + (f"; becomes the faster plan if live degradation exceeds ×{alt['switch_mult']:.2f}"
                            if alt.get("switch_mult") else ""))
        if alt.get("when"):
            lines.append(f"- {alt['when']}")
    if plan.get("note"):
        lines += ["", f"_{plan['note']}_"]
    return "\n".join(lines)
