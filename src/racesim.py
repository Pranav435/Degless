"""The whole Grand Prix, lap by lap, with the live strategy engine in the loop.

A race that has not happened yet cannot be replayed, so it is simulated - and
simulated *through the same code path the pit wall runs on race day*: every
lap of the synthetic race is turned into the timing messages the official feed
would send (`TimingData`, `TimingAppData`, `LapCount`, `TrackStatus`), applied
to a `LiveState`, and the `RaceEngine` ticks on it exactly as `scripts/50_live.py`
does.  The two Haas cars then *do what the engine says*: PIT NOW boxes them at
the end of the lap, BOX BY LAP x boxes them on lap x, STAY OUT and WAIT keep
them out.  The other twenty cars run the plan the weekend fit's per-driver
search gave each of them, with small seeded jitter and two reactions a real
strategist makes (cover a car that undercuts, box under a safety car).

The truth the engine has to discover is one posterior draw of the sealed tyre
model (rate, compound offsets, the regime multiplier it is paired with, the
extrapolation widening beyond the practice support), scaled by a scenario's
wear multiplier and a small per-car factor.  So the engine is tested against a
race that is *consistent with the model's own uncertainty*, not against the
model's mean - which is the honest version of "does the live update work".

Three runs share one truth, one lap-noise matrix and one field, and differ
only in what the two Haas cars are told:

* ``engine``  - the live race-execution engine drives both cars;
* ``plan``    - the sealed pre-race plan, followed to the lap;
* ``mirror``  - no model: cover the car ahead when it stops, else stop late.

The difference between them is what the system is worth, in seconds and in
places, on a race whose truth is known.  Everything the engine says lap by lap
(the call, its confidence, the window, the wear and drop-off estimates, the
rivals, the rejoin projection, the reasons) is kept for the two Haas cars, and
every tick is timed, so the Race sim tab can show the system working at the
speed it works.

Physics of the synthetic race, in one place (`SimConfig`): base pace from the
qualifying gap to pole (compressed, as race gaps are), the fuel effect from the
2026 physics prior, tyre loss from the truth draw, a linear track evolution,
close-following losses and a pass threshold, the standing start, in-lap and
out-lap pit-loss split, and a safety car that bunches the field.  None of it is
tuned to flatter the engine: the pass threshold and dirty air are the measured
2026 values the objective already uses.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    OUT_LAP_PENALTY_S,
    SC_PIT_LOSS_FRACTION,
    VALID_COMPOUNDS,
    Event,
    get_event,
)
from src.fuel import get_prior
from src.live.engine import RaceEngine, WeekendModel
from src.live.state import LiveState
from src.live.store import _clean
from src.live.streams import Message
from src.tyre import grip_loss, load_profile

HAAS = ("OCO", "BEA")
TEAM = "Haas F1 Team"
MODES = ("engine", "plan", "mirror")
RACE_START_UTC = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)   # the epoch of the synthetic feed


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class SimConfig:
    event: str = "spain-2026"
    seed: int = 0
    deg_mult: float = 1.0          # the truth's wear against the model's draw (scenario)
    sc_lap: int | None = None      # safety car deployed on this lap ...
    sc_laps: int = 4               # ... for this many laps
    haas_mode: str = "engine"      # engine | plan | mirror
    n_draws: int = 300             # engine posterior draws
    truth: str = "median"          # which draw is the truth: "median" (the model's central wear) | "random"
    # -- race physics --------------------------------------------------------
    race_offset_s: float = 1.2     # fresh-tyre race pace above the qualifying lap, fuel term aside
    quali_compress: float = 0.7    # race gaps are this fraction of qualifying gaps
    no_time_penalty_s: float = 0.35  # a car with no qualifying time: team-mate plus this
    lap_noise_s: float = 0.35      # per-lap noise of a clean racing lap
    car_mult_ln_sd: float = 0.08   # per-car wear multiplier spread (log)
    evo_s_per_lap: float = -0.03   # the track getting faster over the race
    pit_loss_sd_s: float = 0.8     # the pit lane's own noise around the prior
    pit_in_share: float = 0.45     # share of the pit loss on the in-lap; the rest on the out-lap
    start_lap_extra_s: float = 5.0  # the standing start
    start_accordion_s: float = 0.22  # ...plus this per grid slot
    dirty_air_close_s: float = 0.25  # within 1.5 s of the car ahead
    dirty_air_far_s: float = 0.10    # within 3.0 s
    pass_mid_s: float = 0.25         # pace advantage at which a pass succeeds half the time (DRS included)
    pass_width_s: float = 0.15       # ...and how quickly that chance rises with the advantage
    hold_gap_s: float = 0.8          # a held car's gap to the car ahead
    sc_lap_extra_s: float = 28.0     # a lap behind the safety car
    sc_bunch_gap_s: float = 1.2      # the interval the pack closes to
    field_jitter_laps: int = 2       # the field's stops move up to this many laps from plan
    cover_gap_s: float = 1.5         # a car this close behind that stops gets covered next lap
    mirror_latest_lap: int = 8       # `mirror` mode stops at the latest this many laps before the flag

    def scenario_id(self) -> str:
        return f"deg{self.deg_mult:.2f}_sc{self.sc_lap or 0}_seed{self.seed}_{self.truth}"


# --------------------------------------------------------------------------
# The grid: from qualifying, cached on disk so the app never touches the network
# --------------------------------------------------------------------------


def grid_path(key: str) -> Path:
    return DATA_PROCESSED / f"grid_{key}.json"


def load_grid(event: Event | str, *, refresh: bool = False) -> dict:
    """`{"cars": [...], "source": ...}`: number, code, team, colour, grid slot and
    best qualifying lap per car.  Cars without a time start from the back, in
    the order the results list them.  Falls back to the practice lap table
    (fastest lap order) when the qualifying archive is not available."""
    ev = get_event(event) if isinstance(event, str) else event
    p = grid_path(ev.key)
    if p.exists() and not refresh:
        return json.loads(p.read_text())
    cars, source = [], ""
    try:
        from src.ingest import load_session

        s = load_session(ev, "Qualifying")
        r = s.results
        timed, untimed = [], []
        for _, x in r.iterrows():
            qs = [t for t in (x.get("Q1"), x.get("Q2"), x.get("Q3")) if pd.notna(t)]
            best = min(qs).total_seconds() if qs else None
            row = {"num": str(x["DriverNumber"]), "code": str(x["Abbreviation"]), "team": str(x["TeamName"]),
                   "colour": (str(x.get("TeamColor")) if pd.notna(x.get("TeamColor")) else None),
                   "quali_best_s": best,
                   "quali_position": (int(x["Position"]) if pd.notna(x.get("Position")) else None)}
            (timed if best is not None else untimed).append(row)
        timed.sort(key=lambda c: (c["quali_position"] is None, c["quali_position"] or 99))
        cars = timed + untimed
        source = f"FastF1 qualifying results ({len(timed)} timed, {len(untimed)} without a time, at the back)"
    except Exception as exc:  # pragma: no cover - network / archive dependent
        source = f"practice order (qualifying not available: {str(exc)[:80]})"
        lp = DATA_PROCESSED / f"laps_{ev.key}_practice.parquet"
        if lp.exists():
            lt = pd.read_parquet(lp)
            best = lt.groupby("driver")["lap_time_s"].min().sort_values()
            teams = lt.drop_duplicates("driver").set_index("driver")["team"].to_dict()
            nums = (lt.drop_duplicates("driver").set_index("driver")["driver_number"].astype(str).to_dict()
                    if "driver_number" in lt else {})
            cars = [{"num": nums.get(d, str(i + 1)), "code": d, "team": teams.get(d), "colour": None,
                     "quali_best_s": float(t), "quali_position": i + 1} for i, (d, t) in enumerate(best.items())]
    for i, c in enumerate(cars):
        c["grid"] = i + 1
    out = {"event": ev.key, "source": source, "cars": cars,
           "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if cars:
        p.write_text(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------
# The field's plans (the weekend fit's per-driver search)
# --------------------------------------------------------------------------


def _plan_of(row: dict) -> dict | None:
    comps = row.get("compounds")
    comps = [c for c in str(comps).split("-") if c] if not isinstance(comps, list) else [str(c) for c in comps]
    pits = row.get("pit_laps")
    if not isinstance(pits, list):
        try:
            pits = json.loads(str(pits))
        except Exception:
            return None
    if not comps or len(comps) != len(pits) + 1:
        return None
    return {"compounds": [c.upper() for c in comps], "pit_laps": [int(x) for x in pits]}


def field_plans(meta: dict) -> dict:
    """`{code: {compounds, pit_laps}}` plus `'*'` for the field's best plan."""
    out = {}
    for r in meta.get("per_driver") or []:
        pl = _plan_of(r)
        if pl:
            out[str(r.get("driver"))] = pl
    bp = (meta.get("strategy") or {}).get("best_plan") or {}
    if bp.get("compounds") and bp.get("pit_laps"):
        out["*"] = {"compounds": [str(c) for c in bp["compounds"]], "pit_laps": [int(x) for x in bp["pit_laps"]]}
    if "*" not in out and out:
        out["*"] = next(iter(out.values()))
    return out


# --------------------------------------------------------------------------
# The truth
# --------------------------------------------------------------------------


@dataclass
class Truth:
    draw: int
    rate: dict                  # compound -> wear per lap at reference load (multiplier in)
    pace_offset: dict           # compound -> s/lap
    budget: dict                # compound -> s
    regime_mult: float
    z: float                    # the draw's extrapolation normal
    car_mult: dict              # code -> multiplier on the rate
    pit_loss_s: float
    evo_s_per_lap: float
    fuel_s_per_lap: float
    support: dict
    extrap_ln_sd: float

    def as_dict(self) -> dict:
        return asdict(self)


def make_truth(wm: WeekendModel, cfg: SimConfig, codes: list, rng: np.random.Generator) -> Truth:
    m = wm.model
    if cfg.truth == "random":
        d = int(rng.integers(m.n_draws))
    else:
        # the draw whose race-regime MEDIUM wear is the median of the model's:
        # the truth is then the model's own central forecast, and the scenario
        # multiplier moves it from there
        ref = "MEDIUM" if "MEDIUM" in m.compounds else m.compounds[0]
        r = m.wear_rate[ref] * wm.m_prior
        d = int(np.argsort(r)[len(r) // 2])
    z = float(m.extrap_z()[d]) if m.extrap_ln_sd else 0.0
    car_mult = {c: float(np.exp(rng.normal(0.0, cfg.car_mult_ln_sd))) for c in codes}
    return Truth(
        draw=d,
        rate={c: float(m.wear_rate[c][d] * wm.m_prior[d] * cfg.deg_mult) for c in m.compounds},
        pace_offset={c: float(m.pace_offset[c][d]) for c in m.compounds},
        budget={c: float(m.budget_of(c)) for c in m.compounds},
        regime_mult=float(wm.m_prior[d]), z=z, car_mult=car_mult,
        pit_loss_s=float(wm.pit_loss_s + rng.normal(0.0, cfg.pit_loss_sd_s)),
        evo_s_per_lap=float(cfg.evo_s_per_lap), fuel_s_per_lap=float(get_prior(wm.event, "2026").s_per_lap),
        support=dict(m.support), extrap_ln_sd=float(m.extrap_ln_sd))


# --------------------------------------------------------------------------
# One car
# --------------------------------------------------------------------------


@dataclass
class Car:
    num: str
    code: str
    team: str
    colour: str | None
    grid: int
    base_s: float
    plan: dict                       # {compounds, pit_laps}
    mode: str                        # engine | plan | mirror | field
    compound: str = ""
    stint: int = 0
    stint_first_lap: int = 1
    wear: float = 0.0
    stops: list = field(default_factory=list)    # [{lap, from, to}]
    used: list = field(default_factory=list)     # compounds run
    cum_s: float = 0.0
    lap_times: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    intervals: list = field(default_factory=list)
    compounds: list = field(default_factory=list)
    ages: list = field(default_factory=list)
    wears: list = field(default_factory=list)
    pit_in_laps: set = field(default_factory=set)
    pit_out_laps: set = field(default_factory=set)
    next_pit: tuple | None = None   # (lap, compound) the strategy has decided on
    pending_stops: list = field(default_factory=list)   # the field's planned stops, jittered
    forced: bool = False

    @property
    def age(self) -> int:
        return len(self.ages) - self.stint_first_lap + 1 if self.ages else 0


def _fmt_lap(t: float) -> str:
    m = int(t // 60)
    return f"{m}:{t - 60 * m:06.3f}"


def _fmt_gap(g: float | None, lap_time: float) -> str:
    if g is None:
        return ""
    if g >= lap_time:
        return f"+{int(g // lap_time)} LAP" + ("S" if g >= 2 * lap_time else "")
    return f"+{g:.3f}"


# --------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------


class RaceSim:
    """One synthetic race.  `run()` returns the result dict (JSON-safe)."""

    def __init__(self, cfg: SimConfig, *, wm: WeekendModel | None = None, meta: dict | None = None,
                 grid: dict | None = None, engine: bool | None = None):
        self.cfg = cfg
        self.ev = get_event(cfg.event)
        self.n = int(self.ev.n_race_laps)
        self.wm = wm or WeekendModel.load(self.ev, n_draws=cfg.n_draws)
        mp = DATA_PROCESSED / f"weekend_{self.ev.key}.json"
        self.meta = meta if meta is not None else (json.loads(mp.read_text()) if mp.exists() else {})
        self.grid = grid or load_grid(self.ev)
        self.plans = field_plans(self.meta)
        self.use_engine = (cfg.haas_mode == "engine") if engine is None else bool(engine)
        self.load = load_profile(self.ev)
        self.fp = get_prior(self.ev, "2026")
        self.rng = np.random.default_rng(cfg.seed)
        codes = [c["code"] for c in self.grid["cars"]]
        self.truth = make_truth(self.wm, cfg, codes, self.rng)
        # every random number the race needs, drawn once, so that the three
        # modes see the same race
        self.noise = self.rng.normal(0.0, cfg.lap_noise_s, size=(len(codes), self.n + 1))
        self.jitter = {c: int(self.rng.integers(-cfg.field_jitter_laps, cfg.field_jitter_laps + 1)) for c in codes}
        self.start_shuffle = self.rng.random(len(codes))
        self.pass_u = self.rng.random(size=(len(codes), self.n + 1))     # the dice every overtaking attempt rolls
        self.cars = self._build_cars()
        self.state: LiveState | None = None
        self.engine: RaceEngine | None = None
        self.records: dict = {c.code: [] for c in self.cars}
        self.ticks: list = []
        self.alerts: list = []
        self.track_status = "1"
        self.sc_laps: set = (set(range(cfg.sc_lap, cfg.sc_lap + cfg.sc_laps)) if cfg.sc_lap else set())
        self._t0 = RACE_START_UTC

    # -- setup ---------------------------------------------------------------

    def _base_pace(self, cars: list) -> dict:
        cfg = self.cfg
        timed = {c["code"]: c["quali_best_s"] for c in cars if c.get("quali_best_s")}
        pole = min(timed.values()) if timed else float(self.ev.t_lap_ref_s)
        base = {}
        for c in cars:
            q = c.get("quali_best_s")
            if q:
                base[c["code"]] = pole + cfg.quali_compress * (q - pole) + cfg.race_offset_s
        teams: dict = {}
        for c in cars:
            teams.setdefault(c["team"], []).append(c["code"])
        med = float(np.median(list(base.values()))) if base else pole + cfg.race_offset_s
        for c in cars:
            if c["code"] in base:
                continue
            mates = [base[m] for m in teams.get(c["team"], []) if m in base]
            base[c["code"]] = (mates[0] if mates else med + 0.5) + cfg.no_time_penalty_s
        return base

    def _build_cars(self) -> list:
        base = self._base_pace(self.grid["cars"])
        out = []
        for c in self.grid["cars"]:
            code = c["code"]
            plan = self.plans.get(code) or self.plans.get("*") or {"compounds": ["MEDIUM", "HARD"],
                                                                     "pit_laps": [self.n // 2]}
            mode = self.cfg.haas_mode if code in HAAS else "field"
            car = Car(num=str(c["num"]), code=code, team=c.get("team") or "", colour=c.get("colour"),
                      grid=int(c["grid"]), base_s=float(base[code]), plan=plan, mode=mode)
            car.compound = plan["compounds"][0]
            car.used = [car.compound]
            if mode in ("field", "plan"):
                pits = plan["pit_laps"]
                jit = self.jitter[code] if mode == "field" else 0
                car.pending_stops = [(max(6, min(self.n - 6, p + jit)), plan["compounds"][i + 1])
                                     for i, p in enumerate(pits)]
            out.append(car)
        return out

    # -- the feed ------------------------------------------------------------

    def _msg(self, topic: str, payload, t: float) -> Message:
        return Message(topic=topic, payload=payload, utc=self._t0 + timedelta(seconds=float(t)),
                       t_session=float(t))

    def _start_messages(self) -> list:
        ev = self.ev
        msgs = [self._msg("SessionInfo", {"Name": "Race", "Type": "Race", "Key": 0, "StartDate": self._t0.isoformat(),
                                          "Path": f"sim/{ev.key}/", "Meeting": {"Name": ev.name},
                                          "Simulated": True}, 0.0),
                self._msg("DriverList", {c.num: {"Tla": c.code, "TeamName": c.team, "TeamColour": c.colour or "",
                                                 "FullName": c.code, "Line": c.grid} for c in self.cars}, 0.1),
                # the timing line first, as the real feed does: a car's lap 1 has
                # to exist before its opening set is declared, or the set has no
                # first lap and no tyre age
                self._msg("TimingData", {"Lines": {c.num: {"Position": c.grid, "GapToLeader": "",
                                                           "IntervalToPositionAhead": {"Value": ""}, "InPit": False}
                                                   for c in self.cars}}, 0.2),
                self._msg("TimingAppData", {"Lines": {c.num: {"Stints": {"0": {"Compound": c.compound, "New": "true",
                                                                                "StartLaps": 0, "TotalLaps": 0}}}
                                                      for c in self.cars}}, 0.3),
                self._msg("LapCount", {"CurrentLap": 1, "TotalLaps": self.n}, 0.4),
                self._msg("TrackStatus", {"Status": "1", "Message": "AllClear"}, 0.5),
                self._msg("WeatherData", {"TrackTemp": "53.0", "AirTemp": "31.0", "Rainfall": "0"}, 0.6),
                self._msg("SessionStatus", {"Status": "Started"}, 1.0)]
        return msgs

    # -- physics -------------------------------------------------------------

    def _tyre_loss(self, car: Car, lap: int) -> tuple:
        """(loss_s, wear_after): one lap on the set, truth draw."""
        tr = self.truth
        c = car.compound
        age = lap - car.stint_first_lap + 1
        s = tr.extrap_ln_sd * max(0.0, age / tr.support[c] - 1.0) if (tr.extrap_ln_sd and tr.support.get(c)) else 0.0
        inc = tr.rate[c] * tr.car_mult[car.code] * self.load[lap - 1] * math.exp(tr.z * s)
        w_new = car.wear + inc
        loss = float(grip_loss(np.array([w_new - 0.5 * inc]), budget=tr.budget[c])[0]) + tr.pace_offset[c]
        if age == 1:
            loss += OUT_LAP_PENALTY_S
        return loss, w_new

    # -- strategy of the cars that are not driven by the engine ---------------

    def _field_decisions(self, lap: int, order: list, stopped_last_lap: set) -> None:
        """Decide who pits on `lap` (the in-lap) among the plan-driven cars."""
        cfg = self.cfg
        sc_now = lap in self.sc_laps
        by_code = {c.code: c for c in self.cars}
        pos_of = {c.code: i for i, c in enumerate(order)}
        for car in self.cars:
            if car.mode == "engine" or car.next_pit is not None:
                continue
            if car.mode == "mirror":
                if len(car.used) >= 2:
                    continue
                ahead = order[pos_of[car.code] - 1] if pos_of[car.code] > 0 else None
                if lap >= self.n - cfg.mirror_latest_lap:
                    car.next_pit = (lap, "HARD" if car.compound != "HARD" else "MEDIUM")
                elif ahead is not None and ahead.code in stopped_last_lap and lap >= 8:
                    car.next_pit = (lap, ahead.compound if ahead.compound != car.compound else "HARD")
                elif sc_now and lap >= 10:
                    car.next_pit = (lap, "HARD" if car.compound != "HARD" else "MEDIUM")
                continue
            if not car.pending_stops:
                continue
            p_lap, comp = car.pending_stops[0]
            take = lap >= p_lap
            # a strategist's two reflexes: cover the car that just undercut, box
            # under the safety car when the stop is near anyway.  The `plan`
            # baseline has neither: it is the sealed plan followed to the lap.
            if not take and lap >= 8 and car.mode == "field":
                behind = order[pos_of[car.code] + 1] if pos_of[car.code] + 1 < len(order) else None
                if behind is not None and behind.code in stopped_last_lap and (p_lap - lap) <= 4 \
                        and behind.intervals and behind.intervals[-1] is not None \
                        and behind.intervals[-1] <= cfg.cover_gap_s:
                    take = True
                if sc_now and (p_lap - lap) <= 10:
                    take = True
            if take:
                car.next_pit = (lap, comp if comp != car.compound or len(car.used) >= 2 else "HARD")
                car.pending_stops.pop(0)

    def _engine_decisions(self, lap: int, field_rows: dict) -> None:
        """The two Haas cars do what the engine said on its last tick (for lap `lap`)."""
        for car in self.cars:
            if car.mode != "engine" or car.next_pit is not None:
                continue
            row = field_rows.get(car.num) or {}
            plan = row.get("plan") or {}
            dec = plan.get("decision") or {}
            stop_lap, comp = None, plan.get("next_compound")
            if dec and dec.get("action_kind") in ("PIT_NOW", "WAIT", "BOX_BY") and dec.get("lap") is not None:
                stop_lap = int(dec["lap"])
            elif not dec and plan.get("next_stop") is not None:
                stop_lap = int(plan["next_stop"])
            if stop_lap is not None and stop_lap <= lap and plan.get("best_kind", 1) > 0:
                car.next_pit = (lap, comp or ("HARD" if car.compound != "HARD" else "MEDIUM"))
            # a second compound is mandatory: the engine never leaves this to
            # chance (stay-out is illegal on one compound) but the sim will not
            # finish a race on it either
            if car.next_pit is None and len(car.used) < 2 and lap >= self.n - 5:
                car.next_pit = (lap, "HARD" if car.compound != "HARD" else "MEDIUM")
                car.forced = True

    # -- one lap -------------------------------------------------------------

    def _run_lap(self, lap: int, order: list) -> list:
        """Simulate lap `lap` for every car; returns the new order."""
        cfg, tr = self.cfg, self.truth
        sc_now = lap in self.sc_laps
        idx = {c.code: i for i, c in enumerate(self.cars)}
        raw, pit_in_now = {}, set()
        for c in order:
            fuel = tr.fuel_s_per_lap * (self.n - lap)
            loss, w_new = self._tyre_loss(c, lap)
            t = c.base_s + fuel + loss + tr.evo_s_per_lap * (lap - 1) + self.noise[idx[c.code], lap]
            if lap == 1:
                t += cfg.start_lap_extra_s + cfg.start_accordion_s * (c.grid - 1)
            if c.next_pit is not None and c.next_pit[0] == lap:
                pit_in_now.add(c.code)
                t += tr.pit_loss_s * cfg.pit_in_share * (SC_PIT_LOSS_FRACTION if sc_now else 1.0)
            if lap in c.pit_out_laps:
                t += tr.pit_loss_s * (1.0 - cfg.pit_in_share) * (SC_PIT_LOSS_FRACTION if (lap - 1) in self.sc_laps else 1.0)
            if sc_now:
                t += cfg.sc_lap_extra_s
                w_new = c.wear + 0.3 * (w_new - c.wear)      # cruising behind the safety car
            raw[c.code] = (t, w_new)
        # close following and passing, in the order the lap started
        new_cum: dict = {}
        finished: list = []
        for i, c in enumerate(order):
            t, w_new = raw[c.code]
            ahead = finished[-1] if finished else None
            if ahead is not None and lap > 1 and not sc_now:
                gap0 = c.cum_s - ahead.cum_s
                if gap0 < 1.5:
                    t += cfg.dirty_air_close_s
                elif gap0 < 3.0:
                    t += cfg.dirty_air_far_s
            cum = c.cum_s + t
            if ahead is not None:
                cum_a = new_cum[ahead.code]
                if cum < cum_a:
                    pitting = (ahead.code in pit_in_now) or (lap in ahead.pit_out_laps) \
                        or (c.code in pit_in_now) or (lap in c.pit_out_laps)
                    # a pass is a chance, not a threshold: half the time at
                    # `pass_mid_s` of pace advantage, nearly always at twice it
                    adv = raw[ahead.code][0] - raw[c.code][0]
                    p_pass = 1.0 / (1.0 + math.exp(-(adv - cfg.pass_mid_s) / cfg.pass_width_s))
                    faster = self.pass_u[idx[c.code], lap] < p_pass
                    if sc_now or not (pitting or faster or lap == 1 and self.start_shuffle[idx[c.code]] < 0.15):
                        cum = cum_a + cfg.hold_gap_s
                        t = cum - c.cum_s
                    else:
                        # at most one pass a lap: never ahead of the car two places up
                        if len(finished) >= 2:
                            cum2 = new_cum[finished[-2].code]
                            if cum < cum2 + cfg.hold_gap_s:
                                cum = cum2 + cfg.hold_gap_s
                                t = cum - c.cum_s
            new_cum[c.code] = cum
            # keep `finished` sorted by cum for the next car's comparison
            finished.append(c)
            finished.sort(key=lambda x: new_cum[x.code])
            c.lap_times.append(float(t))
            c.wear = float(w_new)
        # safety car: the pack closes up behind the leader
        if sc_now:
            srt = sorted(self.cars, key=lambda x: new_cum[x.code])
            for a, b in zip(srt[:-1], srt[1:]):
                gap = new_cum[b.code] - new_cum[a.code]
                new_cum[b.code] = new_cum[a.code] + min(gap, cfg.sc_bunch_gap_s + 0.3 * self.noise[idx[b.code], lap] ** 2)
        for c in self.cars:
            c.cum_s = new_cum[c.code]
        new_order = sorted(self.cars, key=lambda x: x.cum_s)
        leader = new_order[0]
        lap_ref = float(np.median([c.lap_times[-1] for c in self.cars]))
        for p, c in enumerate(new_order, start=1):
            gap = c.cum_s - leader.cum_s
            c.positions.append(p)
            c.gaps.append(float(gap))
            c.intervals.append(None if p == 1 else float(c.cum_s - new_order[p - 2].cum_s))
            c.compounds.append(c.compound)
            c.ages.append(lap - c.stint_first_lap + 1)
            c.wears.append(float(c.wear))
        # the pit stops: the in-lap is `lap`, the out-lap `lap + 1`
        for c in self.cars:
            if c.code in pit_in_now:
                _, comp = c.next_pit
                c.stops.append({"lap": lap, "from": c.compound, "to": comp, "position_in": c.positions[-1],
                                "gap_in": c.gaps[-1]})
                c.pit_in_laps.add(lap)
                c.pit_out_laps.add(lap + 1)
                c.compound, c.stint, c.stint_first_lap, c.wear = comp, c.stint + 1, lap + 1, 0.0
                c.used.append(comp)
                c.next_pit = None
        self._lap_ref = lap_ref
        return new_order

    def _lap_messages(self, lap: int, order: list) -> list:
        """The feed for lap `lap`: every car's boundary, pit flags and new stints, sorted by time."""
        msgs = []
        base_t = 0.0
        lap_ref = self._lap_ref
        pitted = {c.code for c in self.cars if lap in c.pit_in_laps}
        for c in order:
            t = c.cum_s
            lt = round(c.lap_times[-1], 3)
            s1 = round(0.31 * lt, 3)
            s2 = round(0.40 * lt, 3)
            s3 = round(lt - s1 - s2, 3)
            if c.code in pitted:
                msgs.append(self._msg("TimingData", {"Lines": {c.num: {"InPit": True}}}, t - 12.0))
            line = {"NumberOfLaps": lap, "LastLapTime": {"Value": _fmt_lap(lt)},
                    "Sectors": {"0": {"Value": f"{s1:.3f}"}, "1": {"Value": f"{s2:.3f}"}, "2": {"Value": f"{s3:.3f}"}},
                    "Position": c.positions[-1],
                    "GapToLeader": ("" if c.positions[-1] == 1 else _fmt_gap(c.gaps[-1], lap_ref)),
                    "IntervalToPositionAhead": {"Value": ("" if c.positions[-1] == 1 else _fmt_gap(c.intervals[-1], lap_ref))}}
            if c.code in pitted:
                line["InPit"] = False        # the exit belongs to the lap about to start (the out-lap)
            msgs.append(self._msg("TimingData", {"Lines": {c.num: line}}, t))
            if c.code in pitted:
                stint = c.stint
                msgs.append(self._msg("TimingAppData", {"Lines": {c.num: {"Stints": {str(stint): {
                    "Compound": c.compound, "New": "true", "StartLaps": 0, "TotalLaps": 0}}}}}, t + 0.5))
            if c.positions[-1] == 1:
                base_t = t
                msgs.append(self._msg("LapCount", {"CurrentLap": min(lap + 1, self.n)}, t + 0.01))
        # track status changes are announced at the start of the lap they apply to
        nxt = lap + 1
        if nxt in self.sc_laps and lap not in self.sc_laps:
            msgs.append(self._msg("TrackStatus", {"Status": "4", "Message": "SCDeployed"}, base_t + 0.02))
        if lap in self.sc_laps and nxt not in self.sc_laps:
            msgs.append(self._msg("TrackStatus", {"Status": "1", "Message": "AllClear"}, base_t + 0.02))
        msgs.sort(key=lambda m: m.t_session)
        return msgs

    # -- the engine's record -------------------------------------------------

    def _record(self, lap_done: int, snap: dict) -> dict:
        rows = {r["driver_number"]: r for r in snap.get("field", [])}
        now_lap = lap_done + 1
        for c in self.cars:
            r = rows.get(c.num) or {}
            p = r.get("plan") or {}
            dec = p.get("decision") or {}
            rec = {"lap": now_lap, "position": r.get("position"), "gap_leader_s": r.get("gap_leader_s"),
                   "interval_s": r.get("interval_s"), "compound": r.get("compound"), "tyre_age": r.get("tyre_age"),
                   "wear": r.get("wear"), "p_past_cliff": r.get("p_past_cliff"),
                   "laps_to_cliff_p10": r.get("laps_to_cliff_p10"), "laps_to_cliff_p50": r.get("laps_to_cliff_p50"),
                   "laps_to_cliff_p90": r.get("laps_to_cliff_p90"), "deg_now_s_per_lap": r.get("deg_now_s_per_lap"),
                   "m_mean": r.get("m_mean"), "m_lo": r.get("m_lo"), "m_hi": r.get("m_hi"), "n_clean": r.get("n_clean"),
                   "m_eff": r.get("m_eff"), "m_eff_lo": r.get("m_eff_lo"), "m_eff_hi": r.get("m_eff_hi"),
                   "cliff_alarm": r.get("cliff_alarm"), "collapse_kind": r.get("collapse_kind"),
                   "best": p.get("best"), "best_kind": p.get("best_kind"), "next_stop": p.get("next_stop"),
                   "next_compound": p.get("next_compound"), "window_lo": p.get("window_lo"),
                   "window_hi": p.get("window_hi"), "delta_box_now_s": p.get("delta_box_now_s"),
                   "delta_stay_out_s": p.get("delta_stay_out_s"), "win_prob": p.get("win_prob"),
                   "action": dec.get("action"), "action_kind": dec.get("action_kind"), "dec_lap": dec.get("lap"),
                   "confidence": dec.get("confidence"), "projected_position": dec.get("projected_position"),
                   "projected_position_if_now": dec.get("projected_position_if_now"),
                   "delta_vs_alternative_s": dec.get("delta_vs_alternative_s"),
                   "held": dec.get("held_by_hysteresis"), "changed": dec.get("changed"),
                   "true_wear": (c.wears[-1] if c.wears else None), "true_age": (c.ages[-1] if c.ages else None)}
            if c.code in HAAS:
                uc = r.get("undercut") or {}
                rec.update({
                    "headline": dec.get("headline"), "principal": dec.get("principal"), "reasons": dec.get("reasons") or [],
                    "rivals": [{k: d.get(k) for k in ("driver", "gap_s", "virtual_gap_s", "why", "compound", "tyre_age",
                                                      "stops", "p_cover", "stop_lap_median", "on_track", "position")}
                               for d in (dec.get("rivals") or [])],
                    "actions": [{k: a.get(k) for k in ("action", "lap", "cost_s", "p_best", "delta_s")}
                                for a in (dec.get("actions") or [])],
                    "rejoin_if_box_now": r.get("rejoin_if_box_now"),
                    "rejoin": dec.get("rejoin"),
                    "proj": r.get("proj") or [],
                    "level_s": r.get("level_s"),
                    "options": [{k: o.get(k) for k in ("label", "delta_s", "win_prob")} for o in (p.get("options") or [])][:4],
                    "undercut_threat": ({k: (uc["threat"] or {}).get(k) for k in ("driver", "gap_s", "p_undercut_1lap", "p_undercut_3lap", "laps_needed", "new_compound")}
                                        if uc.get("threat") else None),
                    "undercut_opportunity": ({k: (uc["opportunity"] or {}).get(k) for k in ("driver", "gap_s", "p_undercut_1lap", "p_undercut_3lap", "laps_needed", "new_compound")}
                                             if uc.get("opportunity") else None),
                    "window": [(x["lap"], round(float(x["loss_s"]), 2)) for x in (p.get("window") or [])[:40]],
                })
            self.records[c.code].append(rec)
        return rows

    # -- run --------------------------------------------------------------------

    def run(self) -> dict:
        cfg = self.cfg
        t_start = time.perf_counter()
        order = sorted(self.cars, key=lambda c: c.grid)
        rows: dict = {}
        if self.use_engine:
            self.state = LiveState()
            self.engine = RaceEngine(self.wm, race_state=True)
            for m in self._start_messages():
                self.state.apply(m)
            self.state.drain_events()
            # the engine's first look: a full field on lap 1, nothing to learn from
            t1 = time.perf_counter()
            snap = self.engine.tick(self.state)
            self.ticks.append({"lap": 0, "engine_ms": 1000 * (time.perf_counter() - t1), "apply_ms": 0.0, "n_msgs": 0})
            rows = self._record(0, snap)
        stopped_last: set = set()
        for lap in range(1, self.n + 1):
            self._field_decisions(lap, order, stopped_last)
            if self.use_engine:
                self._engine_decisions(lap, rows)
            elif cfg.haas_mode == "plan":
                pass          # the Haas cars are `plan`-mode field cars
            order = self._run_lap(lap, order)
            stopped_last = {c.code for c in self.cars if lap in c.pit_in_laps}
            if self.use_engine:
                msgs = self._lap_messages(lap, order)
                ta = time.perf_counter()
                for m in msgs:
                    self.state.apply(m)
                self.state.drain_events()
                apply_ms = 1000 * (time.perf_counter() - ta)
                if lap < self.n:
                    t1 = time.perf_counter()
                    snap = self.engine.tick(self.state)
                    eng_ms = 1000 * (time.perf_counter() - t1)
                    self.ticks.append({"lap": lap, "engine_ms": eng_ms, "apply_ms": apply_ms, "n_msgs": len(msgs)})
                    rows = self._record(lap, snap)
                    self.alerts = list(self.engine.alerts)
        wall = time.perf_counter() - t_start
        return self._result(order, wall)

    # -- the result ----------------------------------------------------------------

    def _result(self, order: list, wall_s: float) -> dict:
        cfg = self.cfg
        cars = []
        for p, c in enumerate(order, start=1):
            cars.append({"num": c.num, "code": c.code, "team": c.team, "colour": c.colour, "grid": c.grid,
                         "haas": c.code in HAAS, "mode": c.mode, "base_pace_s": round(c.base_s, 3),
                         "finish": p, "race_time_s": round(c.cum_s, 3), "gap_to_winner_s": round(c.cum_s - order[0].cum_s, 3),
                         "stops": c.stops, "compounds": list(c.used), "plan": c.plan,
                         "car_mult": round(self.truth.car_mult.get(c.code, 1.0), 4),
                         "forced_stop": c.forced,
                         "laps": {"lap_time_s": [round(x, 3) for x in c.lap_times], "position": c.positions,
                                  "gap_leader_s": [round(x, 3) for x in c.gaps],
                                  "interval_s": [None if x is None else round(x, 3) for x in c.intervals],
                                  "compound": c.compounds, "tyre_age": c.ages, "wear": [round(x, 4) for x in c.wears],
                                  "pit_in": sorted(c.pit_in_laps), "pit_out": sorted(c.pit_out_laps)}})
        ticks = pd.DataFrame(self.ticks) if self.ticks else pd.DataFrame(columns=["lap", "engine_ms", "apply_ms", "n_msgs"])
        steady = ticks[ticks["lap"] >= 1] if len(ticks) else ticks
        out = {
            "event": self.ev.key, "event_name": self.ev.name, "n_laps": self.n, "config": asdict(cfg),
            "scenario_id": cfg.scenario_id(), "mode": cfg.haas_mode, "engine_in_loop": bool(self.use_engine),
            "truth": self.truth.as_dict(), "grid_source": self.grid.get("source", ""),
            "model_source": self.wm.source, "sealed_file": self.wm.sealed_file,
            "sc_laps": sorted(self.sc_laps), "cars": cars,
            "engine": {c.code: self.records[c.code] for c in self.cars if c.code in HAAS} if self.use_engine else {},
            "field_calls": ({c.code: [{k: r.get(k) for k in ("lap", "position", "best", "next_stop", "window_lo",
                                                              "window_hi", "action", "confidence")}
                                       for r in self.records[c.code]] for c in self.cars if c.code not in HAAS}
                            if self.use_engine else {}),
            "alerts": [a for a in self.alerts if a.get("driver") in HAAS][-80:],
            "ticks": ticks.round(2).to_dict("records"),
            "tick_ms": ({"first": round(float(ticks["engine_ms"].iloc[0]), 1),
                         "median": round(float(steady["engine_ms"].median()), 1),
                         "mean": round(float(steady["engine_ms"].mean()), 1),
                         "p95": round(float(steady["engine_ms"].quantile(0.95)), 1),
                         "max": round(float(steady["engine_ms"].max()), 1),
                         "apply_median": round(float(steady["apply_ms"].median()), 2),
                         "n_ticks": int(len(steady)), "n_msgs": int(steady["n_msgs"].sum()),
                         "lap_time_s": round(float(np.median([x for c in self.cars for x in c.lap_times])), 2)}
                        if len(steady) else {}),
            "wall_s": round(wall_s, 2),
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return _clean(out)


# --------------------------------------------------------------------------
# Scenarios: the engine run and its two baselines, and what the system is worth
# --------------------------------------------------------------------------


def _haas_summary(res: dict) -> dict:
    out = {}
    for c in res["cars"]:
        if c["haas"]:
            out[c["code"]] = {"finish": c["finish"], "race_time_s": c["race_time_s"], "gap_to_winner_s": c["gap_to_winner_s"],
                              "stops": [(s["lap"], s["to"]) for s in c["stops"]], "compounds": c["compounds"],
                              "grid": c["grid"], "forced_stop": c.get("forced_stop", False)}
    return out


def _engine_validation(res: dict) -> dict:
    """How well the engine read the race it was driving: wear and drop-off
    estimates against the truth, the rejoin projection against what happened,
    and how early the call settled on the lap the car stopped."""
    out = {}
    cars = {c["code"]: c for c in res["cars"]}
    for code, recs in (res.get("engine") or {}).items():
        car = cars.get(code)
        if not car or not recs:
            continue
        stops = car["stops"]
        first = stops[0]["lap"] if stops else None
        # wear: estimate vs truth on the laps both exist (the estimate is for the lap about to start)
        pairs = [(r["wear"], r["true_wear"]) for r in recs if r.get("wear") is not None and r.get("true_wear") is not None]
        wear_mae = float(np.mean([abs(a - b) for a, b in pairs])) if pairs else None
        # the call: first lap from which the called stop lap stayed within one lap of the actual stop
        settled = None
        if first is not None:
            called = [(r["lap"], r["dec_lap"]) for r in recs if r["lap"] <= first and r.get("dec_lap") is not None]
            for i, (lap, dl) in enumerate(called):
                if all(abs(int(d) - first) <= 1 for _, d in called[i:]):
                    settled = lap
                    break
        changes = sum(1 for r in recs if r.get("changed"))
        # rejoin: the projection at the in-lap's call vs the position after the out-lap
        rejoin = None
        if first is not None:
            rec_in = next((r for r in recs if r["lap"] == first), None)
            pos_after = car["laps"]["position"][first] if first < len(car["laps"]["position"]) else None
            proj = (rec_in or {}).get("projected_position") or (rec_in or {}).get("projected_position_if_now")
            if proj is not None and pos_after is not None:
                rejoin = {"projected": int(proj), "actual": int(pos_after), "error": int(pos_after) - int(proj)}
        # drop-off: what the engine said was left on the set at the in-lap vs the truth
        life = None
        if first is not None:
            rec_in = next((r for r in recs if r["lap"] == first), None)
            if rec_in and rec_in.get("laps_to_cliff_p50") is not None and rec_in.get("true_wear") is not None:
                tw = float(rec_in["true_wear"])
                tr = res["truth"]
                comp = rec_in.get("compound")
                rate = tr["rate"].get(comp, 0.0) * tr["car_mult"].get(code, 1.0) if comp else 0.0
                true_left = ((1.0 - tw) / rate) if rate > 0 else None
                life = {"p10": rec_in.get("laps_to_cliff_p10"), "p50": rec_in.get("laps_to_cliff_p50"),
                        "p90": rec_in.get("laps_to_cliff_p90"), "true_laps_left": (round(true_left, 1) if true_left is not None else None),
                        "wear_est": rec_in.get("wear"), "wear_true": tw}
        rec_stop = next((r for r in recs if first is not None and r["lap"] == first), None) or {}
        m_at_stop = rec_stop.get("m_eff") if rec_stop.get("m_eff") is not None else rec_stop.get("m_mean")
        out[code] = {"wear_mae": wear_mae, "call_settled_lap": settled, "first_stop": first,
                     "n_call_changes": changes, "rejoin": rejoin, "life_at_stop": life,
                     "regime_est_at_stop": m_at_stop, "regime_true": res["truth"]["regime_mult"] * res["config"]["deg_mult"],
                     "n_alerts": sum(1 for a in res.get("alerts", []) if a.get("driver") == code)}
    return out


def run_scenario(cfg: SimConfig, *, wm: WeekendModel | None = None, meta: dict | None = None,
                 grid: dict | None = None, baselines: bool = True) -> dict:
    """The engine-driven race plus the two baselines on the same truth."""
    wm = wm or WeekendModel.load(cfg.event, n_draws=cfg.n_draws)
    main = RaceSim(SimConfig(**{**asdict(cfg), "haas_mode": "engine"}), wm=wm, meta=meta, grid=grid).run()
    out = {"engine": main, "baselines": {}, "impact": {}, "validation": _engine_validation(main)}
    if baselines:
        for mode in ("plan", "mirror"):
            r = RaceSim(SimConfig(**{**asdict(cfg), "haas_mode": mode}), wm=wm, meta=meta, grid=grid).run()
            out["baselines"][mode] = {"haas": _haas_summary(r), "wall_s": r["wall_s"],
                                      "cars": [{k: c[k] for k in ("code", "finish", "race_time_s", "stops", "compounds")}
                                               for c in r["cars"]]}
        eng = _haas_summary(main)
        for code, e in eng.items():
            row = {"engine": e}
            for mode, b in out["baselines"].items():
                bb = b["haas"].get(code)
                if bb:
                    row[mode] = bb
                    row[f"places_vs_{mode}"] = int(bb["finish"] - e["finish"])
                    row[f"time_vs_{mode}_s"] = round(float(bb["race_time_s"] - e["race_time_s"]), 2)
            out["impact"][code] = row
    return out


DEFAULT_SCENARIOS = [
    {"id": "base", "label": "As forecast", "deg_mult": 1.0, "sc_lap": None, "seed": 0},
    {"id": "sc", "label": "Safety car on lap 22", "deg_mult": 1.0, "sc_lap": 22, "seed": 0},
    {"id": "hot", "label": "Tyres wearing twice the forecast", "deg_mult": 2.0, "sc_lap": None, "seed": 0},
    {"id": "cliff", "label": "Tyres falling off (five times the forecast)", "deg_mult": 5.0, "sc_lap": None, "seed": 0},
    {"id": "cool", "label": "Tyres wearing 40% slower", "deg_mult": 0.6, "sc_lap": None, "seed": 0},
    {"id": "other1", "label": "Another tyre the model thinks possible", "deg_mult": 1.0, "sc_lap": None, "seed": 1, "truth": "random"},
    {"id": "other2", "label": "Another race on the forecast tyre", "deg_mult": 1.0, "sc_lap": None, "seed": 2},
]


def result_path(key: str) -> Path:
    return DATA_PROCESSED / f"racesim_{key}.json"


def build(event: Event | str, scenarios: list | None = None, *, n_draws: int = 300, write: bool = True,
          baselines: bool = True, log=print) -> dict:
    """Run every scenario for a weekend and write `racesim_<key>.json`."""
    ev = get_event(event) if isinstance(event, str) else event
    wm = WeekendModel.load(ev, n_draws=n_draws)
    mp = DATA_PROCESSED / f"weekend_{ev.key}.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    grid = load_grid(ev)
    out = {"event": ev.key, "event_name": ev.name, "n_laps": ev.n_race_laps, "model_source": wm.source,
           "sealed_file": wm.sealed_file, "grid_source": grid.get("source", ""),
           "grid": grid["cars"], "scenarios": {}, "order": [],
           "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for sc in (scenarios or DEFAULT_SCENARIOS):
        cfg = SimConfig(event=ev.key, seed=int(sc.get("seed", 0)), deg_mult=float(sc.get("deg_mult", 1.0)),
                        sc_lap=sc.get("sc_lap"), n_draws=n_draws, truth=str(sc.get("truth", "median")))
        t0 = time.perf_counter()
        r = run_scenario(cfg, wm=wm, meta=meta, grid=grid, baselines=baselines)
        r["id"], r["label"] = sc["id"], sc.get("label", sc["id"])
        out["scenarios"][sc["id"]] = r
        out["order"].append(sc["id"])
        e = r["engine"]
        imp = r.get("impact", {})
        log(f"  {sc['id']:8s} {sc.get('label', ''):28s} "
            + " · ".join(f"{code} P{e_['finish']} (plan P{imp.get(code, {}).get('plan', {}).get('finish', '?')}, "
                         f"mirror P{imp.get(code, {}).get('mirror', {}).get('finish', '?')}) "
                         f"stops {e_['stops']}" for code, e_ in _haas_summary(e).items())
            + f" · tick {e['tick_ms'].get('median', '?')} ms · {time.perf_counter() - t0:.1f} s")
    if write:
        result_path(ev.key).write_text(json.dumps(out, separators=(",", ":"), default=float))
    return out


def load_result(key: str) -> dict | None:
    p = result_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None
