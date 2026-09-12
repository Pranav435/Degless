"""Calibrated constants: what the scored weekends say the plan-deciding numbers are.

`src.config` carries the *defaults* with their derivations, most of them
calibrated by hand on Barcelona and Hungary 2026 when those were the only two
scored weekends.  Seven scored weekends now exist, and the numbers that decide
a plan - the grip budget, the management trade-off, the grid-start penalty,
the dirty-air cost, the undercut-exposure weight and the plan-shape prior -
should be set from all of them and checked out of sample.

`scripts/80_recalibrate.py` does that: for every scored weekend it re-derives
each constant from the *other* weekends' races (leave-one-out), and also from
all of them together (the value a new, unscored weekend gets).  The result is
written to `data/processed/calibration.json`:

    {"written_utc": ..., "weekends": [...],
     "global": {<constants>},                 # for a weekend with no race yet
     "loo":    {<event_key>: {<constants>}},  # for a scored weekend: its own race never used
     "per_weekend": {<event_key>: {<the measurements that fed the pooling>}}}

This module is the reader.  `get_calibration(event)` returns the constants a
weekend should run with - its leave-one-out set if it has one, the global set
otherwise, the config defaults if nothing has been calibrated yet - so the
firewall between a weekend's race and its own recommendation is kept by
construction and a reader can always see which numbers were used and why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from src.config import (
    DATA_PROCESSED,
    DIRTY_AIR_S_PER_LAP,
    GRID_START_PENALTY_S,
    GRIP_BUDGET_S,
    MANAGE_COST_S,
    MANAGE_WEAR_FLOOR,
    PLAN_PRIOR_TAU_S,
    SIGMA_RACE_LAP_S,
    UNDERCUT_EXPOSURE_LAMBDA,
    VALID_COMPOUNDS,
    Event,
)

CALIBRATION_PATH = DATA_PROCESSED / "calibration.json"


@dataclass
class Calibration:
    """The constants a weekend runs with, and where they came from."""

    grip_budget_s: float = GRIP_BUDGET_S
    grip_budget_by_compound: dict = field(default_factory=dict)
    manage_cost_s: float = MANAGE_COST_S
    manage_wear_floor: float = MANAGE_WEAR_FLOOR
    grid_start_penalty_s: float = GRID_START_PENALTY_S
    dirty_air_s_per_lap: float = DIRTY_AIR_S_PER_LAP
    undercut_lambda: float = UNDERCUT_EXPOSURE_LAMBDA
    plan_prior_tau_s: float = PLAN_PRIOR_TAU_S
    sigma_race_lap_s: float = SIGMA_RACE_LAP_S
    driver_factors: dict = field(default_factory=dict)   # driver -> multiplicative rate factor
    source: str = "config defaults (no calibration on disk)"
    detail: dict = field(default_factory=dict)

    @property
    def budgets(self) -> dict:
        """Per-compound grip budget, the pooled value where a compound has none."""
        return {c: float(self.grip_budget_by_compound.get(c, self.grip_budget_s)) for c in VALID_COMPOUNDS}

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


_KEYS = ("grip_budget_s", "grip_budget_by_compound", "manage_cost_s", "manage_wear_floor",
         "grid_start_penalty_s", "dirty_air_s_per_lap", "undercut_lambda", "plan_prior_tau_s",
         "sigma_race_lap_s", "driver_factors")


def load_calibration_file(path: Path | None = None) -> dict:
    p = Path(path) if path else CALIBRATION_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def get_calibration(event: Event | str | None = None, *, loo: bool = True,
                    path: Path | None = None) -> Calibration:
    """The constants for `event`.

    `loo=True` (the default) hands a scored weekend the set derived without its
    own race.  `loo=False` asks for the global set regardless, which is what a
    reader wants when comparing two weekends on the same footing.
    """
    key = event if isinstance(event, str) or event is None else event.key
    d = load_calibration_file(path)
    if not d:
        return Calibration()
    src, block = "global", d.get("global") or {}
    if loo and key and key in (d.get("loo") or {}):
        src, block = f"leave-one-out ({key} held out)", d["loo"][key]
    if not block:
        return Calibration()
    kw = {k: block[k] for k in _KEYS if k in block}
    cal = Calibration(**kw)
    cal.source = f"{src}, calibrated {d.get('written_utc', '?')[:10]} on {len(d.get('weekends', []))} weekends"
    cal.detail = {"weekends": d.get("weekends", []), "block": src}
    return cal
