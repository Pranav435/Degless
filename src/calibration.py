"""Calibrated constants: what the scored weekends say the plan-deciding numbers are.

`src.config` carries the *defaults* with their derivations, most of them
calibrated by hand on Barcelona and Hungary 2026 when those were the only two
scored weekends.  Seven scored weekends now exist, and the numbers that decide
a plan - the grip budget, the management trade-off, the grid-start penalty,
the dirty-air cost, the undercut-exposure weight, the plan-shape prior and the
first-stop prior's weight `first_stop_kappa_s` - should be set from all of them
and checked out of sample.  Two of them are no longer single numbers: dirty air
is per circuit (`dirty_air_for`) and the grip budget is per compound with the
censored estimate behind it in `grip_budget_detail`.

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
    FIRST_STOP_KAPPA_S,
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
    dirty_air_by_circuit: dict = field(default_factory=dict)   # circuit -> s/lap, from its own races
    undercut_lambda: float = UNDERCUT_EXPOSURE_LAMBDA
    plan_prior_tau_s: float = PLAN_PRIOR_TAU_S
    first_stop_kappa_s: float = FIRST_STOP_KAPPA_S
    sigma_race_lap_s: float = SIGMA_RACE_LAP_S
    driver_factors: dict = field(default_factory=dict)   # driver -> multiplicative rate factor
    driver_factor_ln_sd: dict = field(default_factory=dict)    # ...and how precisely it was measured
    team_factors: dict = field(default_factory=dict)           # team -> pooled rate factor
    grip_budget_detail: dict = field(default_factory=dict)     # per compound: the censored estimate
    percar_mode: str = "team_pooled"                           # "team_pooled" (V3) | "hist" (V2) | "none"
    # V4.  `family_temper_s` is the rival field's family logit temperature
    # (`racestate.RivalFieldConfig`): the seconds of tyre-plus-pit cost that
    # halve a plan family's share of the rivals, fitted by maximum likelihood on
    # the donors' start-compound and stop-count shares in
    # `scripts/80_recalibrate.py`.  The default 3.0 s is WP-A's own default -
    # roughly the spread between adjacent plan families' costs, so a field that
    # nothing has calibrated is neither one-family nor uniform.
    family_temper_s: float = 3.0
    # WP-B's tyre-life extrapolation width (log-sd of the wear rate at twice the
    # practice support).  0.0 is V3 bit for bit; the measured value comes from
    # `bench/bench_extrapolation.py` through `src.tyre.EXTRAP_LN_SD_MEASURED`.
    extrap_ln_sd: float = 0.0
    # which objective the sweeps were run against: "v4" is the race-state
    # objective (kappa fixed at 0, lambda on the later stops).  A file written by
    # the V3 script carries no such key and the reader stamps it "v3".
    objective_version: str = "v4"
    source: str = "config defaults (no calibration on disk)"
    detail: dict = field(default_factory=dict)

    @property
    def budgets(self) -> dict:
        """Per-compound grip budget, the pooled value where a compound has none."""
        return {c: float(self.grip_budget_by_compound.get(c, self.grip_budget_s)) for c in VALID_COMPOUNDS}

    def dirty_air_for(self, circuit: str | None = None) -> float:
        """Dirty air at `circuit`, in s/lap - its own measurement where there is one.

        The cost of running within 3 s of the car ahead is a property of the
        circuit, not of the season: Hungary measures +0.43 s/lap and Monza -0.20
        (the tow), and pooling them to one number prices an extra stop at Monza
        as if it cost the Hungarian penalty.  `dirty_air_by_circuit` is built in
        `scripts/80_recalibrate.py` from each circuit's *historical* races, which
        is what lets a weekend use its own circuit's value without its own 2026
        race entering its calibration; the pooled 2026 median is the fallback for
        a circuit with no history.
        """
        if circuit:
            v = self.dirty_air_by_circuit.get(str(circuit))
            if v is None:
                v = self.dirty_air_by_circuit.get(str(circuit).lower())
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return float(self.dirty_air_s_per_lap)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


_KEYS = ("grip_budget_s", "grip_budget_by_compound", "manage_cost_s", "manage_wear_floor",
         "grid_start_penalty_s", "dirty_air_s_per_lap", "dirty_air_by_circuit", "undercut_lambda",
         "plan_prior_tau_s", "first_stop_kappa_s", "sigma_race_lap_s", "driver_factors",
         "driver_factor_ln_sd", "team_factors", "grip_budget_detail", "percar_mode",
         "family_temper_s", "extrap_ln_sd", "objective_version")


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
    # A V2 file carries none of the V3 keys and a future one may carry a null
    # where a measurement failed; both must land on the dataclass default.
    kw = {k: block[k] for k in _KEYS if block.get(k) is not None}
    # a block with no `objective_version` was written by the V3 script: the
    # sweeps behind its lambda, tau and kappa were run on the V3 objective
    kw.setdefault("objective_version", "v3")
    cal = Calibration(**kw)
    cal.source = f"{src}, calibrated {d.get('written_utc', '?')[:10]} on {len(d.get('weekends', []))} weekends"
    cal.detail = {"weekends": d.get("weekends", []), "block": src}
    return cal
