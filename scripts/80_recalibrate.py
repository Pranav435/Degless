"""Leave-one-out recalibration of the plan-deciding constants.

Six numbers decide the plan the tool recommends - the grip budget, the pace
step (through the measured net), the management cost and wear floor, the
grid-start penalty and the dirty-air cost - and two more now shape it: the
undercut-exposure weight and the plan-shape prior.  They were calibrated by
hand on Barcelona and Hungary 2026.  This script re-derives every one of
them from all the scored weekends, with each weekend held out in turn, so
that a weekend's own race never touches the constants it is judged with,
and writes the result to `data/processed/calibration.json`:

    global        every scored weekend                (a new, unscored weekend)
    loo[key]      every scored weekend but `key`      (the scored weekends)
    per_weekend   the measurements that fed the pooling, per weekend
    sweeps        the objective along each constant's grid, for the report

What is measured directly (no search):

* **grip budget per compound** - `src.cliff`'s censored estimate over every
  donor race stint: a stint that ended *on the cliff* (the detector finds the
  break and the box within three laps of it) observes the budget, every other
  stint gives a lower bound, and the log-normal likelihood of that
  right-censored sample is maximised.  V2's estimator - race rate times the
  longest stint, taken as a bound - is kept beside it as `raw.budget_v2` for
  the report, because on six of seven weekends it returned the prior;
* **dirty air, per circuit** - lap time lost within 3 s of the car ahead.  The
  cost is a property of the circuit (Hungary +0.43 s/lap, Monza -0.20: the
  tow), so the shipped value is each circuit's own, measured on its **2023-25
  races** through `history.CircuitPrior.dirty_air`, which is what lets a
  weekend use its circuit's number without its own race informing it.  The
  donors' 2026 median is the fallback for a circuit with no history;
* **driver rate factors** - each driver's degradation relative to the field,
  pooled across the weekends they raced and shrunk toward 1, now with the
  log-scale standard error (`driver_factor_ln_sd`, so the model can shrink what
  was measured loosely) and the same pooling at team level (`team_factors`).

What is calibrated against the field's revealed behaviour, one sweep at a
time on each donor weekend's *own* posterior:

* **management cost and wear floor** - so the push the optimiser chooses
  implies the practice->race factor the donor races measured;
* **undercut lambda** - so the recommended **second** stop lands on the field's
  median green second stop among the finishers who made the same number of
  stops.  Under V4 the *first* stop is timed by the race state, not by lambda,
  so the first-stop objective that calibrated lambda in V3 no longer identifies
  it: what lambda still prices is the exposure of the stops after the first,
  and that is what it is now graded on.  Where the objective is flat the
  smallest value wins and the file says lambda is unidentified;
* **first-stop kappa is not swept**: it is structurally 0 (`KAPPA_V4`).  The
  circuit's first-stop history stays in the plan-family prior, in the stint
  caps and in the *rivals*' plausible stop laps, never as a term on our own
  lap.  V3's swept value is kept under `raw.v3_first_stop_kappa_s`;
* **the rival field's family temperature** (`family_temper_s`, WP-A) - by
  maximum likelihood of the donors' revealed start-compound and stop-count
  shares under the family logit `q(g) ~ exp(-C_g / tau_f) p_hist(g)^w`, with the
  family costs read off each donor's own final search.  No extra searches;
* **plan-prior tau, grid-start penalty** - so the recommended sequence is one
  the field ran, at the largest share.

Every search in every sweep carries the **race state** (`src.objective`), and
inside a leave-one-out block it is measured without *two* races: the weekend the
block holds out and the donor being searched.  The whole objective - race state,
lambda, tau, the grid penalty, the dirty air - is assembled once per search as a
`V4Objective`, which is the same object the pipeline, the outlook and the desk
price with.

    .venv/bin/python scripts/80_recalibrate.py                # every scored weekend
    .venv/bin/python scripts/80_recalibrate.py --quick        # coarser sweeps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import cliff, firststop, objective, strategy as strat  # noqa: E402
from src.calibration import CALIBRATION_PATH, Calibration, load_calibration_file  # noqa: E402
from src.objective import V4Objective  # noqa: E402
from src.compounds import measure_dirty_air  # noqa: E402
from src.config import (  # noqa: E402
    DATA_PROCESSED, DIRTY_AIR_S_PER_LAP, EVENTS, GRID_START_PENALTY_S, GRIP_BUDGET_S,
    MANAGE_COST_S, MANAGE_WEAR_FLOOR, PLAN_PRIOR_TAU_S, SIGMA_RACE_LAP_S, UNDERCUT_EXPOSURE_LAMBDA,
    VALID_COMPOUNDS, get_event,
)
from src.history import circuit_prior, race_deg_slopes, race_driver_factors  # noqa: E402
from src.model_bayes import BayesFit  # noqa: E402
from src.regime import RegimeFactor  # noqa: E402
from src.tyre import TyreModel  # noqa: E402

N_DRAWS = 150
SEARCH_STEP = 2
SHORTLIST = 600
# The stop-lap sweep (lambda: 8 values against 4-6 for the others) is graded on
# one integer, the winner's second stop lap, which the phase-1 mean cost already
# all but decides.  So it runs on a shorter shortlist: phase 2 is the
# draw-by-draw half and the best plan is always first in it either way.
# Everything that is *reported* - the manage sweep, tau, the grid penalty and
# the final scores - runs at the full setting.  Measured on three weekends:
# 0.38 s -> 0.27 s per search, which is what keeps the seven-weekend run inside
# twelve minutes (V4 adds the pack equilibria: +0.02 s per search at these
# settings, measured on hungary-2026).
SWEEP_SHORTLIST = 300
DRIVER_PRIOR_LN_SD = 0.15       # shrinkage of a driver's rate factor toward the field
DRIVER_SE_FLOOR = 0.10          # no single race pins a driver tighter than this (log scale)
GRIDS = {
    "lambda": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6],
    "tau": [0.0, 1.0, 2.5, 4.0, 6.0, 8.0],
    "grid": [0.0, 0.5, 1.0, 1.5],
    "floor": [0.35, 0.45, 0.55],
    "cost": [0.6, 0.9, 1.2],
    # the rival field's family logit temperature, in seconds of family cost
    # (WP-A's `RivalFieldConfig.family_temper_s`); scored by likelihood on the
    # donors' revealed shares, not by a search, so the grid costs arithmetic
    # 1-8 s was the plan's grid; the likelihood was monotone up to its edge on
    # every fold (the field's start/stop variety is wider than the model's family
    # costs imply), so the grid was widened before the final calibration to let
    # the maximum sit inside it - 200 s is a family distribution that is the
    # history prior alone, in effect.  A widening decided on the fit's own
    # diagnostic, never on a benchmark outcome.
    "family_temper": [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0, 200.0],
}
GRIDS_QUICK = {"lambda": [0.0, 0.1, 0.2, 0.3], "tau": [0.0, 2.5, 5.0],
               "grid": [0.0, 1.0], "floor": [0.45], "cost": [0.9],
               "family_temper": [1.0, 3.0, 8.0]}

# V4: the first-stop history prior is structurally off - the race state times
# the first stop - so kappa is fixed at 0 and never swept.  The V3 value is
# carried into the report under `raw.v3_first_stop_kappa_s`.
KAPPA_V4 = 0.0
FAMILY_PRIOR_WEIGHT_K0 = 10.0    # WP-A's `family_prior_weight_k0`: n / (n + k0) on the history
FAMILY_LL_IDENTIFIED_NATS = 1.0  # a flatter likelihood than this does not identify the temperature
FIRST_STOP_LL_FLOOR = 0.05       # `src.firststop.UNIFORM_FLOOR`: mass spread over the race under any predicted density


def scored_weekends() -> list:
    out = []
    for k, ev in EVENTS.items():
        if (DATA_PROCESSED / f"posterior_{k}.npz").exists() and (DATA_PROCESSED / f"laps_{k}_race.parquet").exists() \
                and ((DATA_PROCESSED / f"fitstage_{k}.json").exists() or (DATA_PROCESSED / f"meta_{k}.json").exists()):
            out.append(k)
    return out


def _inputs(key: str) -> dict:
    p = DATA_PROCESSED / f"fitstage_{key}.json"
    if not p.exists():
        p = DATA_PROCESSED / f"meta_{key}.json"
    return json.loads(p.read_text())


class Donor:
    """Everything a donor weekend contributes, loaded once."""

    def __init__(self, key: str):
        self.key = key
        self.ev = get_event(key)
        self.fs = _inputs(key)
        self.race = pd.read_parquet(DATA_PROCESSED / f"laps_{key}_race.parquet")
        self.clean = pd.read_parquet(DATA_PROCESSED / f"clean_{key}_practice.parquet")
        self.fit = BayesFit.load(DATA_PROCESSED / f"posterior_{key}.npz")
        rg = self.fs["regime"]
        self.regime = RegimeFactor(ratio=float(rg["ratio"]), ln_sd=float(rg["ln_sd"]))
        self.self_regime = ((self.fs.get("regime") or {}).get("self_measured") or {}).get("ratio")
        if self.self_regime is None:
            from src.regime import measure_regime
            m = measure_regime(self.ev, race=self.race, practice=self.clean)
            self.self_regime = float(m.ratio) if np.isfinite(m.ratio) else None
        self.caps = {c: int(v) for c, v in ((self.fs.get("circuit_history") or {}).get("stint_cap") or {}).items()} or None
        self.alloc = (self.fs.get("allocation") or {}).get("caps") or {c: 2 for c in VALID_COMPOUNDS}
        self.plan_prior = self.fs.get("plan_prior") or {}
        ns = self.fs.get("net_step") or {}
        self.net = ns.get("measured")
        self.net_se = ns.get("se") or 0.0
        self.support = {k: float(v) for k, v in (self.fs.get("age_support_by_compound") or {}).items()}
        self.pit_loss = float(strat.measure_pit_loss(self.race).seconds)
        if not np.isfinite(self.pit_loss):
            self.pit_loss = 22.0
        # -- the field --------------------------------------------------------
        stops = strat.race_stops(self.race, self.ev)
        cls = {d: v for d, v in stops.items() if v["classified"]}
        self.n_cls = len(cls)
        self.seq_counts = pd.Series(["-".join(v["compounds"]) for v in cls.values()]).value_counts()
        self.stop_counts = pd.Series([len(v["in_laps"]) for v in cls.values()]).value_counts()
        self.start_counts = pd.Series([v["compounds"][0] for v in cls.values()]).value_counts()
        firsts = [(v["in_laps"][0], v["sc"][0]) for v in cls.values() if v["in_laps"]]
        green = [p for p, s in firsts if not s]
        self.green_first = [int(p) for p in green]     # the field's green first stops, for the rival-field check
        self.first_median = float(np.median(green)) if green else None
        self.sc_set = bool(firsts and np.mean([s for _, s in firsts]) > 0.4)
        # V4: the *second* stop is what lambda now prices (the race state times
        # the first).  The field's median green second stop, among the finishers
        # who made the same number of stops as the plan being judged - a
        # two-stopper's second stop and a three-stopper's are different
        # decisions and pooling them compares a plan with a plan nobody ran.
        self.second_by_stops: dict = {}
        for v in cls.values():
            n = len(v["in_laps"])
            if n >= 2 and not v["sc"][1]:
                self.second_by_stops.setdefault(n, []).append(int(v["in_laps"][1]))
        self.second_median_by_stops = {k: float(np.median(x)) for k, x in self.second_by_stops.items()}
        self.second_n_by_stops = {k: len(x) for k, x in self.second_by_stops.items()}
        self.field_stop_mode = int(self.stop_counts.idxmax()) if len(self.stop_counts) else None
        # -- race-measured quantities ---------------------------------------------
        rc = self.race[self.race["is_accurate"] & ~self.race["pit_in"] & ~self.race["pit_out"]
                       & (self.race["track_status"].astype(str) == "1")]
        self.race_rates = race_deg_slopes(rc, self.ev.fuel_effect_s_per_lap)
        st = (self.race.groupby(["driver", "stint"]).agg(compound=("compound", "first"), n=("lap_number", "size"),
                                                          end=("lap_number", "max")).reset_index())
        st = st[(st["n"] >= 3) & st["compound"].isin(VALID_COMPOUNDS) & st["driver"].isin(list(cls))]
        self.longest = st.groupby("compound")["n"].max().to_dict()
        self.p90 = st.groupby("compound")["n"].quantile(0.9).to_dict()
        self.dirty = measure_dirty_air(self.race, self.ev)
        self.driver_factors = race_driver_factors(rc)
        self.teams = (pd.concat([self.clean, self.race])
                      .drop_duplicates("driver").set_index("driver")["team"].to_dict()
                      if "team" in self.race else {})
        # -- within-stint collapse: what the budget is actually read off ------
        self.cliff_rows = cliff.race_collapses(self.race, event=self.ev)
        self.collapse_counts = ({str(k): int(v) for k, v in self.cliff_rows["kind"].value_counts().items()}
                                if not self.cliff_rows.empty else {})
        # -- the circuit's own history: dirty air and the first-stop density --
        # Both come from the circuit's 2023-25 races and never from a 2026 one,
        # so they are the same object in every leave-one-out block and a weekend
        # may use its own circuit's value without its own race informing it.
        # The fit stage writes them into `circuit_history`; a V2 fitstage has
        # neither, so the same `circuit_prior` the fit stage used is rebuilt here
        # (it reads only the cached historical summaries: ~0.3 s, then free).
        ch = self.fs.get("circuit_history") or {}
        self.dirty_hist = dict(ch.get("dirty_air") or {})
        self.first_stop_green = dict(ch.get("first_stop_green") or {})
        if not self.dirty_hist or not self.first_stop_green:
            try:
                cp = circuit_prior(self.ev, probe_practice_temp=False)
            except Exception as exc:
                log_line = f"  {key}: no circuit history ({str(exc)[:70]})"
                print(log_line)
                cp = None
            if cp is not None and cp.available:
                self.dirty_hist = self.dirty_hist or dict(cp.dirty_air or {})
                self.first_stop_green = self.first_stop_green or dict(cp.first_stop_green or {})
        dv = self.dirty_hist.get("s_per_lap")
        self.dirty_circuit = float(dv) if dv is not None and np.isfinite(float(dv)) else None
        self.first_stop_table = firststop.first_stop_penalty_table(
            self.first_stop_green or None, self.ev.n_race_laps, VALID_COMPOUNDS)
        # The summary is reported at the *history's* modal start compound and stop
        # count (from the fitstage's plan prior, which reads only past races), not
        # at this weekend's own, so a per-weekend block never quotes a number
        # derived from the race being held out.
        _pp_starts = (self.plan_prior or {}).get("starts") or {}
        _pp_stops = (self.plan_prior or {}).get("stops") or {}
        modal_start = (max(_pp_starts, key=_pp_starts.get) if _pp_starts
                       else (self.start_counts.index[0] if len(self.start_counts) else None))
        modal_stops = int(max(_pp_stops, key=_pp_stops.get)) if _pp_stops else None
        self.first_stop_prior_summary = firststop.first_stop_summary(
            self.first_stop_green or None, self.ev.n_race_laps,
            start_compound=modal_start, n_stops=modal_stops)
        self._model_cache: dict = {}
        self._rs_cache: dict = {}

    def race_state(self, held_out: str | None = None):
        """The race-state constants for a search on this donor, inside a block
        that holds `held_out` out.

        Both races are excluded: this donor's own (it is the weekend being
        priced) and the block's held-out weekend (whose race may not inform a
        constant the block will hand it).  The global block excludes only the
        donor itself."""
        keys = frozenset({self.key} | ({held_out} if held_out else set()))
        if keys not in self._rs_cache:
            self._rs_cache[keys] = objective.measure_constants_excluding(keys)
        return self._rs_cache[keys]

    def dirty_for(self, cal: dict) -> float:
        """This circuit's dirty-air cost, the pooled 2026 value where it has none."""
        return float(self.dirty_circuit if self.dirty_circuit is not None else cal["dirty"])

    def implied_budget(self) -> dict:
        out = {}
        for c, d in self.race_rates.items():
            if d["slope"] > 0.005 and c in self.longest:
                out[c] = float(d["slope"] * self.longest[c])
        return out

    def model(self, budgets: dict, floor: float, cost: float) -> TyreModel:
        key = (tuple(sorted(budgets.items())), floor, cost)
        if key not in self._model_cache:
            total = self.fit.posterior["lin"].shape[0]
            rng = np.random.default_rng(0)
            idx = rng.choice(total, size=min(N_DRAWS, total), replace=False)
            self._model_cache[key] = TyreModel.from_fit(self.fit, draws=idx, budget=budgets, manage_floor=floor,
                                                        manage_cost_s=cost, carry_drivers=False)
        return self._model_cache[key]

    def objective(self, cal: dict, *, lam: float, tau: float, grid: float) -> V4Objective:
        """The V4 objective this weekend is priced with: the race state (both
        races excluded), the rival field at the block's family temperature, its
        own circuit's dirty air, its own circuit's first-stop density for the
        *rivals*' stop laps, and the weights under test.

        `kappa` is not a parameter: under V4 the first-stop history prior is
        zero everywhere, which is why lambda is swept against the second stop."""
        return V4Objective(race_state=self.race_state(cal.get("held_out")),
                           rival_field=objective.rival_field_default(
                               None, family_temper_s=float(cal.get("family_temper",
                                                                   objective.FAMILY_TEMPER_S_DEFAULT)),
                               mode=str(cal.get("rival_mode", "hetero")),
                               use_history_prior=bool(cal.get("rival_history", True))),
                           undercut_lambda=float(lam), plan_prior=self.plan_prior,
                           plan_prior_tau_s=float(tau), first_stop_kappa_s=KAPPA_V4,
                           traffic_s_per_lap=self.dirty_for(cal), grid_penalty_s=float(grid),
                           extrap_ln_sd=float(cal.get("extrap_ln_sd", 0.0) or 0.0),
                           first_stop_prior=self.first_stop_table,
                           n_race_laps=int(self.ev.n_race_laps), event_key=self.key)

    def _sim_kw(self, cal: dict, *, lam: float, tau: float, grid: float, kappa: float = KAPPA_V4,
                shortlist: int = SHORTLIST) -> dict:
        assert float(kappa or 0.0) == 0.0, "V4 fixes the first-stop history prior at kappa = 0"
        return dict(regime=self.regime, step=SEARCH_STEP, shortlist=shortlist, support=self.support,
                    max_per_compound=self.alloc, max_stint=self.caps,
                    **self.objective(cal, lam=lam, tau=tau, grid=grid).sim_kwargs())

    def search(self, cal: dict, *, lam: float, tau: float, grid: float, kappa: float = KAPPA_V4,
               calibrated_model=None, shortlist: int = SHORTLIST):
        m = calibrated_model if calibrated_model is not None else self.model(cal["budgets"], cal["floor"], cal["cost"])
        return strat.simulate_model(m, self.ev, self.pit_loss,
                                    **self._sim_kw(cal, lam=lam, tau=tau, grid=grid, kappa=kappa,
                                                   shortlist=shortlist))

    def calibrated_model(self, cal: dict, *, lam: float, tau: float, grid: float,
                         kappa: float = KAPPA_V4) -> TyreModel:
        m = self.model(cal["budgets"], cal["floor"], cal["cost"])
        if self.net is None or not np.isfinite(self.net):
            return m
        m2, _, _ = strat.search_with_pace_calibration(m, self.ev, self.pit_loss, net_step_s=float(self.net),
                                                      net_step_se_s=float(self.net_se),
                                                      **self._sim_kw(cal, lam=lam, tau=tau, grid=grid, kappa=kappa))
        return m2

    # -- the objectives -------------------------------------------------------

    def decision_scores(self, res) -> dict:
        if res.table.empty:
            return {"first_err": np.nan, "second_err": np.nan, "seq_share": 0.0, "start_ok": 0.0, "stops_ok": 0.0,
                    "implied": np.nan, "first_lap": None, "second_lap": None, "first_stop_s": np.nan,
                    "race_state_s": np.nan, "family_costs": {}}
        seq = "-".join(res.best["compounds"])
        share = float(self.seq_counts.get(seq, 0) / max(self.n_cls, 1))
        start_ok = float(self.start_counts.index[0] == res.best["compounds"][0]) if len(self.start_counts) else 0.0
        stops_ok = float(int(self.stop_counts.idxmax()) == int(res.best["n_stops"])) if len(self.stop_counts) else 0.0
        first_err = ((float(res.best["pit_laps"][0]) - self.first_median)
                     if (self.first_median is not None and res.best["pit_laps"] and not self.sc_set) else np.nan)
        # V4: lambda's objective is the *second* stop.  Only where both the
        # recommendation and the field's mode are two stops or more is there a
        # second stop to compare, and the field median is taken among the
        # finishers who made the same number of stops as the recommendation.
        n_rec = int(res.best["n_stops"])
        second_lap = int(res.best["pit_laps"][1]) if len(res.best["pit_laps"]) >= 2 else None
        med2 = self.second_median_by_stops.get(n_rec)
        second_err = (abs(float(second_lap) - med2)
                      if (second_lap is not None and med2 is not None and n_rec >= 2
                          and (self.field_stop_mode or 0) >= 2) else np.nan)
        return {"first_err": first_err, "second_err": second_err, "seq_share": share, "start_ok": start_ok,
                "stops_ok": stops_ok, "implied": float(res.implied_regime), "best": res.best_label,
                "first_lap": (int(res.best["pit_laps"][0]) if res.best["pit_laps"] else None),
                "second_lap": second_lap, "n_stops": n_rec,
                "first_stop_s": float(res.best.get("first_stop_s", 0.0)),
                "race_state_s": float(res.best.get("race_state_s", 0.0)),
                "family_costs": family_costs(res)}


# --------------------------------------------------------------------------
# The rival field's family temperature, by maximum likelihood
# --------------------------------------------------------------------------


def family_costs(res) -> dict:
    """`{group label: {cost_s, compounds, start, n_stops}}` - the best *tyre*
    cost in each plan family (start compound, second compound, stop count).

    The family cost `C_g` the rival field's logit reads: the plan the group's
    own tyres would run, with no prior and no position term on it.  Taken from
    WP-A's own table where the merged code exposes it (`res.race_state
    ["rival_field"]["families"]`), and otherwise from the scored table, which is
    the same minimum over the shortlist the search ranked."""
    rf = ((getattr(res, "race_state", None) or {}).get("rival_field") or {})
    fams = (rf.get("families") or rf.get("types")) if isinstance(rf, dict) else None
    rows = list(fams if isinstance(fams, list) else (fams or {}).values()) if fams else []
    rows = [f for f in rows if isinstance(f, dict) and ("cost_s" in f or "tyre_s" in f)]
    if rows:
        out = {}
        for f in rows:
            lab, cost = f.get("label") or f.get("group"), f.get("cost_s", f.get("tyre_s"))
            comps = list(f.get("compounds") or [])
            if lab and cost is not None and len(comps) >= 2:
                out[str(lab)] = {"cost_s": float(cost), "compounds": comps, "start": str(comps[0]),
                                 "n_stops": int(f.get("n_stops", len(comps) - 1))}
        if out:
            return out
    if res.table is None or res.table.empty:
        return {}
    groups: dict = {}
    for r in res.table.itertuples():
        seq = str(r.compounds).split("-")
        if len(seq) < 2:
            continue
        g = (seq[0], seq[1], int(r.n_stops))
        cur = groups.get(g)
        if cur is None or float(r.tyre_s) < cur["cost_s"]:
            groups[g] = {"cost_s": float(r.tyre_s), "compounds": seq, "start": seq[0], "n_stops": int(r.n_stops)}
    return {strat._group_label(g): v for g, v in groups.items()}


def family_logit(costs: dict, plan_prior: dict | None, temper: float,
                 k0: float = FAMILY_PRIOR_WEIGHT_K0) -> dict:
    """The rivals' plan-family distribution:

        q(g)  proportional to  exp(-C_g / temper) * p_hist(g) ** w,
        w = n / (n + k0)

    `p_hist` is the circuit's smoothed plan-family frequency, which
    `strategy.plan_prior_penalty(seq, prior, 1.0)` gives as `-log(p / p_max)` -
    so the history enters as `exp(-w * penalty)` and the constant `p_max`
    divides out in the normalisation.  `temper -> 0` is every rival on the
    cheapest family; `temper -> inf` is a uniform field."""
    if not costs:
        return {}
    n = float((plan_prior or {}).get("n", 0) or 0)
    w = n / (n + float(k0)) if n > 0 else 0.0
    c0 = min(v["cost_s"] for v in costs.values())
    z = {}
    for lab, v in costs.items():
        pen = strat.plan_prior_penalty(v["compounds"], plan_prior, 1.0) if w > 0 else 0.0
        z[lab] = float(np.exp(-(v["cost_s"] - c0) / max(float(temper), 1e-6) - w * pen))
    tot = sum(z.values())
    if tot <= 0:
        return {lab: 1.0 / len(costs) for lab in costs}
    return {lab: v / tot for lab, v in z.items()}


def family_loglik(q: dict, costs: dict, start_counts, stop_counts, *, eps: float = 1e-6) -> float:
    """Log-likelihood of one donor's revealed shares under the family logit.

    The field's classified finishers are the sample: each one's *start
    compound* and *stop count* is a draw from the logit's marginals (the
    sequence itself is not - a rival's later compounds are a choice it makes
    during the race, and scoring them would grade the temperature on something
    the field re-decided).  A start or stop count no family in the table
    produces is floored at `eps`, which is the same constant at every
    temperature and so does not move the argmax."""
    p_start: dict = {}
    p_stops: dict = {}
    for lab, prob in q.items():
        v = costs.get(lab) or {}
        p_start[str(v.get("start"))] = p_start.get(str(v.get("start")), 0.0) + prob
        p_stops[int(v.get("n_stops", 0))] = p_stops.get(int(v.get("n_stops", 0)), 0.0) + prob
    ll = 0.0
    for c, k in dict(start_counts).items():
        ll += float(k) * float(np.log(max(p_start.get(str(c), 0.0), eps)))
    for st, k in dict(stop_counts).items():
        ll += float(k) * float(np.log(max(p_stops.get(int(st), 0.0), eps)))
    return float(ll)


def field_q_of(res) -> tuple | None:
    """The first-stop distribution a search's rival model implies for the field.

    The heterogeneous field carries it as the type mixture
    (`res.race_state["rival_field"]["field_stop_distribution"]`); the symmetric
    pack's is the recommended group's own choice distribution, because in that
    model the field *is* four copies of us."""
    rs = getattr(res, "race_state", None) or {}
    d = (rs.get("rival_field") or {}).get("field_stop_distribution")
    if d and d.get("laps"):
        return np.asarray(d["laps"], dtype=int), np.asarray(d["q"], dtype=float)
    best = rs.get("best") or {}
    if best.get("laps"):
        return np.asarray(best["laps"], dtype=int), np.asarray(best["q"], dtype=float)
    return None


def stop_loglik(laps: np.ndarray, q: np.ndarray, stops: list, n_race_laps: int,
                floor: float = FIRST_STOP_LL_FLOOR) -> tuple:
    """Log-likelihood of the field's actual green first stops under a predicted
    distribution on the lap grid, with a uniform floor over the race so that a
    stop the model gave no mass to is expensive, not impossible (the same
    floor `src.firststop` puts under its own density).  Returns `(sum, n)`."""
    q = np.asarray(q, dtype=float)
    q = q / q.sum() if q.sum() > 0 else np.full(len(laps), 1.0 / max(len(laps), 1))
    p = dict(zip((int(l) for l in laps), (1.0 - floor) * q))
    base = floor / float(n_race_laps)
    ll = float(sum(np.log(p.get(int(l), 0.0) + base) for l in stops))
    return ll, int(len(stops))


def sweep_family_temper(donors: list, cal: dict, grid: list, models: dict) -> tuple:
    """The rival field, validated on what the field actually did.

    A rival model exists to say when the cars around us will box, so it is
    scored on exactly that: the log-likelihood, per stop, of every donor's
    green first stops under the first-stop distribution the model implies
    (`field_q_of`), leave-one-out like every other constant here.  Three
    things come out of it:

    * the **family temperature**: the grid value with the highest pooled
      likelihood under the heterogeneous field (flat within one nat over the
      grid = unidentified, and the default stands);
    * the **rival mode**: the heterogeneous field at that temperature is kept
      only if it predicts the field's stops at least as well as Task 1's
      symmetric pack; otherwise the block records `"symmetric"` and the search
      runs Task 1's pack.  That is the rule `docs/v4_plan.md` set for "if
      heterogeneous modelling does not generalise, keep the simpler model" -
      decided on the rivals' behaviour, never on our own plan's benchmark;
    * a diagnostic row for the field with no historical stop prior.

    One search per donor per grid value at the sweep shortlist; the family
    costs that the temperature reweights do not depend on the temperature, so
    the searches differ only in the field equilibrium."""
    def pooled(c: dict) -> tuple:
        tot, n, per = 0.0, 0, {}
        for d in donors:
            if not d.green_first:
                continue
            res = d.search(c, lam=c["lambda"], tau=c["tau"], grid=c["grid"],
                           calibrated_model=models.get(d.key), shortlist=SWEEP_SHORTLIST)
            fq = field_q_of(res)
            if fq is None:
                continue
            ll, k = stop_loglik(fq[0], fq[1], d.green_first, d.ev.n_race_laps)
            cq = np.cumsum(fq[1] / max(fq[1].sum(), 1e-12))
            per[d.key] = {"ll": round(ll, 3), "n": k, "ll_per_stop": round(ll / max(k, 1), 4),
                          "predicted_median": int(fq[0][min(int(np.searchsorted(cq, 0.5)), len(fq[0]) - 1)]),
                          "field_median": float(np.median(d.green_first))}
            tot += ll
            n += k
        return (tot / n if n else float("nan")), n, per

    rows = []
    for t in grid:
        c = dict(cal); c["family_temper"] = float(t); c["rival_mode"] = "hetero"; c["rival_history"] = True
        ll, n, per = pooled(c)
        rows.append({"family_temper": float(t), "ll_per_stop": float(ll), "n_stops": n, "by_donor": per})
    if not rows or all(not np.isfinite(r["ll_per_stop"]) for r in rows):
        return float(objective.FAMILY_TEMPER_S_DEFAULT), rows, False, {}
    best = max(rows, key=lambda r: (r["ll_per_stop"] if np.isfinite(r["ll_per_stop"]) else -np.inf))
    n_stops = max(r["n_stops"] for r in rows)
    spread = (best["ll_per_stop"] - min(r["ll_per_stop"] for r in rows if np.isfinite(r["ll_per_stop"]))) * n_stops
    identified = bool(spread >= FAMILY_LL_IDENTIFIED_NATS)
    # on the plan's original 1-8 s grid the likelihood was monotone to the edge
    # on every fold; the grid now runs to 200 s (see GRIDS) and the flag says
    # whether the maximum still sits on an edge, which the report must state.
    for r in rows:
        r["at_grid_edge"] = bool(r is best and (best["family_temper"] == max(grid)
                                                or best["family_temper"] == min(grid)))
    chosen = best if identified else next((r for r in rows
                                           if abs(r["family_temper"] - objective.FAMILY_TEMPER_S_DEFAULT) < 1e-9),
                                          best)
    for r in rows:
        r["chosen"] = bool(r is chosen)
    # the mode decision: the symmetric pack and the no-history field at the chosen temperature
    c_sym = dict(cal); c_sym["family_temper"] = float(chosen["family_temper"]); c_sym["rival_mode"] = "symmetric"
    ll_sym, n_sym, per_sym = pooled(c_sym)
    c_noh = dict(cal); c_noh["family_temper"] = float(chosen["family_temper"]); c_noh["rival_mode"] = "hetero"
    c_noh["rival_history"] = False
    ll_noh, n_noh, per_noh = pooled(c_noh)
    ll_het = float(chosen["ll_per_stop"])
    # the yardstick every rival model has to beat: a stop lap drawn uniformly
    # over the race (what "no rival model" predicts), on the same donors
    n_unif = sum(len(d.green_first) for d in donors)
    ll_unif = (sum(len(d.green_first) * np.log(1.0 / d.ev.n_race_laps) for d in donors) / n_unif
               if n_unif else float("nan"))
    mode = "hetero" if (np.isfinite(ll_het) and (not np.isfinite(ll_sym) or ll_het >= ll_sym)) else "symmetric"
    block = {"mode": mode, "family_temper_s": float(chosen["family_temper"]),
             "ll_per_stop": {"hetero": ll_het, "symmetric": float(ll_sym), "hetero_no_history": float(ll_noh),
                             "uniform_over_race": float(ll_unif)},
             "n_stops": int(n_stops), "by_donor": {"hetero": chosen["by_donor"], "symmetric": per_sym,
                                                    "hetero_no_history": per_noh},
             "rule": "hetero is kept when its leave-one-out log-likelihood per green first stop is at least "
                     "the symmetric pack's; the field's own stops decide, never our plan's benchmark"}
    return float(chosen["family_temper"]), rows, identified, block

BUDGET_INFORMATIVE_S = 3.0      # below this, no stint that weekend came near the cliff: the product is only a bound
BUDGET_BAND_S = (3.0, 4.5)


def pool_budgets(donors: list) -> tuple:
    """V2's grip budget per compound, kept for the report's `raw.budget_v2`.

    Superseded by `pool_cliff_budgets`, which measures the cliff instead of
    bounding it; the two are reported side by side because on six of seven
    weekends this one returned the prior and that is the thing V3 set out to fix.

    Race rate times the longest stint a classified finisher ran is the budget
    only when that stint reached the cliff; at a low-degradation circuit the
    longest stint ends for strategic reasons long before it (Australia's
    MEDIUM: 0.025 s/lap x 28 laps = 0.7 s), so every weekend's product is a
    *lower bound*.  The estimate is therefore the largest bound any weekend
    showed - with each rate taken at its lower 1-sigma value so the maximum is
    not inflated by rate noise - and where no weekend ran a compound within
    `BUDGET_INFORMATIVE_S` of a cliff the data say nothing and the prior
    (`GRIP_BUDGET_S`) stays.  The raw products are returned for the report.
    """
    raw: dict = {}
    lower: dict = {}
    for d in donors:
        for c, dd in d.race_rates.items():
            if dd["slope"] > 0.005 and c in d.longest:
                raw.setdefault(c, []).append(float(dd["slope"] * d.longest[c]))
                lo = max(dd["slope"] - dd.get("se", 0.0), 0.5 * dd["slope"])
                lower.setdefault(c, []).append(float(lo * d.longest[c]))
    per = {}
    for c, v in lower.items():
        mx = max(v)
        per[c] = float(np.clip(mx, *BUDGET_BAND_S)) if mx >= BUDGET_INFORMATIVE_S else float(GRIP_BUDGET_S)
    pooled = float(np.mean(list(per.values()))) if per else float(GRIP_BUDGET_S)
    return pooled, per, {c: {"raw": [round(x, 2) for x in raw[c]], "lower": [round(x, 2) for x in v],
                             "informative": bool(max(v) >= BUDGET_INFORMATIVE_S)} for c, v in lower.items()}


def pool_cliff_budgets(donors: list) -> tuple:
    """Grip budget per compound from the censored stint-collapse sample.

    Every donor's race stints are pooled and `src.cliff` maximises the
    right-censored log-normal likelihood per compound: collapse stints observe
    the budget, the rest bound it from below.  The pooled scalar (for a compound
    no donor ran) is the mean of the per-compound estimates, as V2's was.
    """
    rows = [d.cliff_rows for d in donors if d.cliff_rows is not None and not d.cliff_rows.empty]
    if not rows:
        return float(GRIP_BUDGET_S), {}, {}
    est = cliff.grip_budgets_by_compound(pd.concat(rows, ignore_index=True))
    per = {c: float(v["budget_s"]) for c, v in est.items()}
    pooled = float(np.mean(list(per.values()))) if per else float(GRIP_BUDGET_S)
    return pooled, per, est


def pool_dirty(donors: list) -> tuple:
    vals = [d.dirty["s_per_lap"] for d in donors if d.dirty]
    if not vals:
        return DIRTY_AIR_S_PER_LAP, []
    return float(np.clip(np.median(vals), 0.05, 0.8)), [round(v, 3) for v in vals]


def dirty_by_circuit(donors: list) -> dict:
    """`{circuit: s/lap}` from each circuit's own 2023-25 races.

    Not clipped positive: Monza's tow really does make following cheaper, and
    pretending otherwise is what priced an extra stop there as if it were
    Hungary.  Built over *every* loaded weekend rather than a block's donors, so
    a leave-one-out block still has the held-out circuit's value - legitimately,
    because it was measured on races three years older than the weekend.
    """
    out = {}
    for d in donors:
        if d.dirty_circuit is not None:
            out[str(d.ev.circuit)] = round(float(d.dirty_circuit), 4)
    return out


def pool_drivers(donors: list) -> dict:
    """Precision-weighted, shrunk log rate factors per driver across the donors."""
    acc: dict = {}
    for d in donors:
        for drv, v in d.driver_factors.items():
            f = float(v["factor"])
            if not (0.2 < f < 5.0):
                continue
            lf = float(np.log(f))
            se = max(float(v["factor_se"]) / max(f, 0.2), DRIVER_SE_FLOOR)
            acc.setdefault(drv, []).append((lf, se, d.key))
    out = {}
    for drv, rows in acc.items():
        w = np.array([1.0 / se ** 2 for _, se, _ in rows])
        x = np.array([lf for lf, _, _ in rows])
        prec = w.sum() + 1.0 / DRIVER_PRIOR_LN_SD ** 2
        mean = float((w * x).sum() / prec)
        out[drv] = {"factor": float(np.exp(mean)), "ln_sd": float(np.sqrt(1.0 / prec)), "n_races": len(rows),
                    "raw": [round(float(np.exp(lf)), 2) for lf, _, _ in rows], "events": [k for _, _, k in rows]}
    return out


def pool_teams(donors: list, drivers: dict) -> dict:
    """The same factor at team level: same car, same tyre engineers.

    A team is the strongest grouping there is for tyre behaviour, and it is what
    a driver with one noisy race should be pulled toward rather than the field
    (`src.percar` does the pulling; this only measures the target).  Inverse-
    variance weighted on the log scale over the team's drivers.
    """
    teams: dict = {}
    for d in donors:
        for drv, t in (d.teams or {}).items():
            if drv in drivers and t:
                teams.setdefault(str(t), set()).add(drv)
    out = {}
    for t, drvs in teams.items():
        lf = np.array([np.log(max(drivers[d]["factor"], 1e-6)) for d in sorted(drvs)])
        w = np.array([1.0 / max(float(drivers[d]["ln_sd"]), DRIVER_SE_FLOOR) ** 2 for d in sorted(drvs)])
        mean = float((w * lf).sum() / w.sum())
        out[t] = {"factor": float(np.exp(mean)), "ln_sd": float(np.sqrt(1.0 / w.sum())),
                  "drivers": sorted(drvs)}
    return out


def sweep(donors: list, cal: dict, name: str, values: list, objective, *, models=None,
          shortlist: int = SHORTLIST, tol: float = 0.0) -> tuple:
    """Evaluate `objective(scores_by_donor)` along a grid for one constant.

    The chosen value is the **smallest** grid value whose objective lies
    within `tol` of the best, not the argmin.  These objectives are flat or
    monotone over most of their grids (V2 reported tau saturating and lambda
    flat below 0.3; kappa's plateau runs from 1 to 6), so the argmin is decided
    by noise at the level of one weekend flipping one flag, and a grid that is
    widened by one point can walk a constant to its new edge on a 0.02 gain -
    which is what handed the Australia-held-out block tau = 8 and, with it, a
    two-stop on a weekend the field one-stopped.  `tol` is passed by the caller
    as one weekend's worth of the objective; the parsimonious end of a plateau
    is the honest reading of a constant the data do not identify.
    """
    rows = []
    for v in values:
        c = dict(cal); c[name] = v
        scores = []
        for d in donors:
            m = models.get(d.key) if models else None
            res = d.search(c, lam=c["lambda"], tau=c["tau"], grid=c["grid"], kappa=c["kappa"],
                           calibrated_model=m, shortlist=shortlist)
            scores.append(d.decision_scores(res))
        obj = objective(scores)
        _fin2 = [s["second_err"] for s in scores if np.isfinite(s.get("second_err", np.nan))]
        rows.append({name: v, "objective": obj,
                     "mean_first_err": float(np.nanmean([s["first_err"] for s in scores])) if any(np.isfinite(s["first_err"]) for s in scores) else None,
                     "mean_abs_first_err": float(np.nanmean([abs(s["first_err"]) for s in scores])) if any(np.isfinite(s["first_err"]) for s in scores) else None,
                     "mean_abs_second_err": (float(np.mean(_fin2)) if _fin2 else None),
                     "n_second_donors": len(_fin2),
                     "second_laps": {d.key: s["second_lap"] for d, s in zip(donors, scores)},
                     "mean_race_state_s": float(np.nanmean([s.get("race_state_s", np.nan) for s in scores]))
                     if any(np.isfinite(s.get("race_state_s", np.nan)) for s in scores) else None,
                     "mean_seq_share": float(np.mean([s["seq_share"] for s in scores])),
                     "start_ok": float(np.mean([s["start_ok"] for s in scores])),
                     "stops_ok": float(np.mean([s["stops_ok"] for s in scores])),
                     "first_laps": {d.key: s["first_lap"] for d, s in zip(donors, scores)},
                     "first_stop_penalty_s": float(np.mean([s["first_stop_s"] for s in scores])),
                     "implied_vs_measured": float(np.nanmean([abs(np.log(max(s["implied"], 1e-3)) - np.log(max(d.self_regime or np.nan, 1e-3)))
                                                             for s, d in zip(scores, donors) if d.self_regime]))
                     if any(d.self_regime for d in donors) else None})
    best_obj = min(r["objective"] for r in rows)
    within = [r for r in sorted(rows, key=lambda r: float(r[name])) if r["objective"] <= best_obj + float(tol)]
    chosen = within[0]
    for r in rows:
        r["chosen"] = bool(r is chosen)
        r["within_tol"] = bool(r["objective"] <= best_obj + float(tol))
        r["tol"] = float(tol)
    return chosen[name], rows


def calibrate(donors: list, grids: dict, label: str, *, dirty_circuits: dict | None = None,
              held_out: str | None = None, v3_kappa: float | None = None,
              rival_mode: str = "auto") -> dict:
    t0 = time.time()
    pooled_b, per_b, est_b = pool_cliff_budgets(donors)
    _, per_v2, raw_v2 = pool_budgets(donors)
    dirty, raw_d = pool_dirty(donors)
    drivers = pool_drivers(donors)
    teams = pool_teams(donors, drivers)
    # V4: every search in the sweeps carries the race state (measured without
    # this block's held-out weekend *and* without the donor being searched) and
    # kappa is fixed at 0 - the race state times the first stop, so the
    # first-stop objective no longer identifies lambda and lambda is swept
    # against the second stop instead.
    cal = {"budgets": {**{c: pooled_b for c in VALID_COMPOUNDS}, **per_b}, "floor": MANAGE_WEAR_FLOOR, "cost": MANAGE_COST_S,
           "dirty": dirty, "lambda": UNDERCUT_EXPOSURE_LAMBDA, "tau": PLAN_PRIOR_TAU_S, "grid": GRID_START_PENALTY_S,
           "kappa": KAPPA_V4, "held_out": held_out,
           "family_temper": objective.FAMILY_TEMPER_S_DEFAULT,
           "extrap_ln_sd": objective._extrap_ln_sd(None)}
    sweeps = {}

    def obj_regime(scores):
        errs = [abs(np.log(max(s["implied"], 1e-3)) - np.log(max(d.self_regime, 1e-3)))
                for s, d in zip(scores, donors) if d.self_regime and np.isfinite(s["implied"])]
        return float(np.mean(errs)) if errs else 0.0

    def obj_first(scores):
        errs = [abs(s["first_err"]) for s in scores if np.isfinite(s["first_err"])]
        return float(np.mean(errs)) if errs else 0.0

    def obj_second(scores):
        """V4's lambda objective: the *later* stops.

        Mean |recommended second stop - the field's median green second stop
        among the finishers who made the same number of stops|, over the donors
        whose recommendation and field mode are both two stops or more.  The
        first stop is the race state's, and charging lambda for it twice is what
        Task 1 left behind."""
        errs = [s["second_err"] for s in scores if np.isfinite(s.get("second_err", np.nan))]
        return float(np.mean(errs)) if errs else 0.0

    def obj_shape(scores):
        return float(-np.mean([s["seq_share"] for s in scores]) - 0.25 * np.mean([s["start_ok"] for s in scores])
                     - 0.25 * np.mean([s["stops_ok"] for s in scores]))

    # 1. management trade-off: floor and cost, on the uncalibrated pace (they set the push, not the compound)
    best_pair, best_obj, rows_m = None, np.inf, []
    for fl in grids["floor"]:
        for co in grids["cost"]:
            c = dict(cal); c["floor"] = fl; c["cost"] = co
            scores = [d.decision_scores(d.search(c, lam=c["lambda"], tau=c["tau"], grid=c["grid"], kappa=c["kappa"]))
                      for d in donors]
            o = obj_regime(scores)
            rows_m.append({"floor": fl, "cost": co, "objective": o,
                           "implied": [round(s["implied"], 2) for s in scores]})
            if o < best_obj:
                best_obj, best_pair = o, (fl, co)
    cal["floor"], cal["cost"] = best_pair
    sweeps["manage"] = rows_m
    # the calibrated-pace models at the chosen trade-off, reused by the remaining sweeps
    models = {d.key: d.calibrated_model(cal, lam=cal["lambda"], tau=cal["tau"], grid=cal["grid"], kappa=cal["kappa"])
              for d in donors}
    # 2. lambda against the field's *second* stops, then the two shape weights
    #    against the sequences it ran.  Twice round: lambda moves the cost
    #    surface tau is then priced against, and the reverse.
    # One weekend's worth of each objective is the tolerance of the parsimony
    # rule in `sweep`: for the stop-lap objective (a mean of |laps| over the
    # donors that have a second stop to compare) one lap on one donor; for the
    # shape objective (a mean field share plus a quarter each for the start and
    # stop flags) one donor flipping one flag.
    n_first = max(sum(1 for d in donors if d.first_median is not None and not d.sc_set), 1)
    tol_first = 1.0 / n_first
    # ...and one lap on one donor for the second-stop objective, over the donors
    # that have a second stop to compare at all
    n_second = max(sum(1 for d in donors
                       if (d.field_stop_mode or 0) >= 2 and any(k >= 2 for k in d.second_median_by_stops)), 1)
    tol_second = 1.0 / n_second
    tol_shape = 0.25 / max(len(donors), 1)
    for _ in range(2):
        cal["lambda"], sweeps["lambda"] = sweep(donors, cal, "lambda", grids["lambda"], obj_second,
                                                models=models, shortlist=SWEEP_SHORTLIST, tol=tol_second)
        cal["tau"], sweeps["tau"] = sweep(donors, cal, "tau", grids["tau"], obj_shape, models=models, tol=tol_shape)
        cal["grid"], sweeps["grid"] = sweep(donors, cal, "grid", grids["grid"], obj_shape, models=models, tol=tol_shape)
    # final scores at the chosen constants
    final = {d.key: d.decision_scores(d.search(cal, lam=cal["lambda"], tau=cal["tau"], grid=cal["grid"],
                                               kappa=cal["kappa"], calibrated_model=models[d.key])) for d in donors}
    # the rival field's family temperature, by maximum likelihood on the shares
    # the donors' fields revealed - no further searches (the family costs come
    # from the final tables above)
    cal["family_temper"], sweeps["family_temper"], ft_identified, rival_block = sweep_family_temper(
        donors, cal, grids.get("family_temper", GRIDS["family_temper"]), models)
    cal["rival_mode"] = rival_block.get("mode", "hetero") if rival_mode == "auto" else str(rival_mode)
    rival_block["mode_forced"] = (None if rival_mode == "auto" else str(rival_mode))
    # the final scores are re-taken with the rival field as chosen (mode and temperature)
    final = {d.key: d.decision_scores(d.search(cal, lam=cal["lambda"], tau=cal["tau"], grid=cal["grid"],
                                               kappa=cal["kappa"], calibrated_model=models[d.key])) for d in donors}
    lam_rows = sweeps.get("lambda") or []
    lam_objs = [r["objective"] for r in lam_rows]
    lam_identified = bool(lam_objs and (max(lam_objs) - min(lam_objs)) > tol_second)
    n_second_scored = max((r.get("n_second_donors") or 0) for r in lam_rows) if lam_rows else 0
    out = {
        "grip_budget_s": pooled_b, "grip_budget_by_compound": per_b,
        "grip_budget_detail": {c: {k: v for k, v in e.items()} for c, e in est_b.items()},
        "manage_cost_s": cal["cost"], "manage_wear_floor": cal["floor"],
        "grid_start_penalty_s": cal["grid"], "dirty_air_s_per_lap": cal["dirty"],
        "dirty_air_by_circuit": dict(dirty_circuits or dirty_by_circuit(donors)),
        "undercut_lambda": cal["lambda"], "plan_prior_tau_s": cal["tau"],
        # V4: structurally zero.  The race state times the first stop; the
        # circuit's first-stop history is kept for the rivals' stop laps and the
        # plan-family prior, never as a term on our own lap.
        "first_stop_kappa_s": KAPPA_V4,
        "family_temper_s": cal["family_temper"],
        "rival_field_mode": cal["rival_mode"],
        "rival_field_validation": rival_block,
        "extrap_ln_sd": cal["extrap_ln_sd"],
        "objective_version": "v4",
        "objective_label": donors[0].objective(cal, lam=cal["lambda"], tau=cal["tau"],
                                               grid=cal["grid"]).label if donors else "",
        "objective_notes": {
            "kappa": "fixed at 0 under V4: the race state times the first stop, so the kappa sweep is "
                     "not run.  V3's swept value is kept under raw.v3_first_stop_kappa_s.",
            "lambda": ("swept against the later stops: mean |recommended second stop - field median green "
                       "second stop among finishers with the same stop count| over the donors whose "
                       f"recommendation and field mode are >= 2 stops ({n_second_scored} donors scored)"
                       + ("" if lam_identified else "; the objective is flat over the grid, so lambda is "
                          "unidentified and the smallest value wins")),
            "family_temper_s": ("maximum leave-one-out likelihood of the donors' green first stops under the "
                                "first-stop distribution the heterogeneous rival field implies"
                                + ("" if ft_identified else "; the likelihood is flat over the grid "
                                   "(< 1 nat), so the temperature is unidentified and WP-A's default stands")),
            "rival_field_mode": rival_block.get("rule", ""),
            "extrap_ln_sd": ("from src.tyre.EXTRAP_LN_SD_MEASURED (WP-B's measurement)"
                             if cal["extrap_ln_sd"] else "0.0: no measured extrapolation width on this "
                             "checkout (WP-B's src.tyre.EXTRAP_LN_SD_MEASURED is absent or zero)"),
        },
        "undercut_lambda_identified": lam_identified,
        "family_temper_identified": ft_identified,
        "family_temper_at_grid_edge": bool(any(r.get("chosen") and r.get("at_grid_edge")
                                               for r in sweeps["family_temper"])),
        "race_state": {d.key: d.race_state(held_out).as_dict() for d in donors},
        "held_out": held_out,
        "sigma_race_lap_s": SIGMA_RACE_LAP_S,
        "driver_factors": {k: v["factor"] for k, v in drivers.items()},
        "driver_factor_ln_sd": {k: v["ln_sd"] for k, v in drivers.items()},
        "driver_factor_detail": drivers,
        "team_factors": {k: v["factor"] for k, v in teams.items()},
        "team_factor_detail": teams,
        "percar_mode": "team_pooled",
        "donors": [d.key for d in donors],
        "raw": {"implied_budgets": raw_v2, "budget_v2": per_v2, "dirty_air": raw_d,
                "collapse_counts": {d.key: d.collapse_counts for d in donors},
                # V3's swept kappa, for the report only: under V4 nothing charges it
                "v3_first_stop_kappa_s": (float(v3_kappa) if v3_kappa is not None else None),
                "v3_first_stop_kappa_note": "the value V3's kappa sweep chose for this block, kept for the "
                                            "report; V4 fixes kappa at 0 and never sweeps it",
                "second_stop_field_median_by_stops": {d.key: d.second_median_by_stops for d in donors},
                "second_stop_n_by_stops": {d.key: d.second_n_by_stops for d in donors}},
        "sweeps": sweeps, "final_scores": final,
        "seconds": round(time.time() - t0, 1),
    }
    n_obs = sum(int(e.get("n_obs", 0)) for e in est_b.values())
    print(f"  [{label}] budgets {({c: round(v, 2) for c, v in per_b.items()})} (pooled {pooled_b:.2f}, "
          f"{n_obs} collapse observations; V2 would say {({c: round(v, 2) for c, v in per_v2.items()})}); "
          f"floor {cal['floor']} cost {cal['cost']}; dirty {dirty:.3f} pooled + {len(out['dirty_air_by_circuit'])} circuits; "
          f"lambda {cal['lambda']}{'' if lam_identified else ' (unidentified)'} on the second stop; "
          f"kappa {KAPPA_V4} (fixed; V3 said {v3_kappa}); tau {cal['tau']}; grid {cal['grid']}; "
          f"family temper {cal['family_temper']}{'' if ft_identified else ' (unidentified)'} "
          f"rival mode {cal['rival_mode']} (ll/stop hetero {rival_block.get('ll_per_stop', {}).get('hetero', float('nan')):.3f} "
          f"vs symmetric {rival_block.get('ll_per_stop', {}).get('symmetric', float('nan')):.3f}); "
          f"extrap ln sd {cal['extrap_ln_sd']}; "
          f"{len(drivers)} drivers / {len(teams)} teams; {out['seconds']}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rival-mode", default="auto", choices=["auto", "hetero", "symmetric"],
                    help="the rival model the searches run: 'auto' lets each block's stop-likelihood "
                         "validation choose; a fixed mode is what docs/v4_plan.md §3 (E1) decides after "
                         "the generalisation experiment, and the validation block is still written")
    ap.add_argument("--events", nargs="*", default=None)
    ap.add_argument("--out", default=str(CALIBRATION_PATH))
    args = ap.parse_args()
    grids = GRIDS_QUICK if args.quick else GRIDS
    keys = args.events or scored_weekends()
    print(f"scored weekends: {keys}")
    t0 = time.time()
    donors = {k: Donor(k) for k in keys}
    print(f"loaded {len(donors)} weekends in {time.time()-t0:.0f}s")
    dirty_circuits = dirty_by_circuit(list(donors.values()))
    for k, d in donors.items():
        print(f"  {k}: implied budgets {d.implied_budget()}; dirty air 2026 {d.dirty.get('s_per_lap', float('nan')):+.3f} "
              f"/ {d.ev.circuit} history {('%+.3f' % d.dirty_circuit) if d.dirty_circuit is not None else 'none'}; "
              f"first stop median (green) {d.first_median} sc-set {d.sc_set}; net {d.net}; self regime {d.self_regime}")
        print(f"      stint collapse: {d.collapse_counts or 'no stint scored'}; first-stop prior "
              + (f"mode {d.first_stop_prior_summary['mode']} median {d.first_stop_prior_summary['median']:.0f} "
                 f"({d.first_stop_prior_summary['p25']:.0f}-{d.first_stop_prior_summary['p75']:.0f}, "
                 f"n={d.first_stop_prior_summary['n']})" if d.first_stop_prior_summary else "none: no circuit history"))
    # V3's swept kappa, per block, read off the calibration file being replaced -
    # kept in the report so a reader can see what V4 switched off.
    prev = load_calibration_file()
    v3_kappa = {k: ((prev.get("loo") or {}).get(k) or {}).get("first_stop_kappa_s") for k in keys}
    v3_kappa["_global"] = (prev.get("global") or {}).get("first_stop_kappa_s")
    result = {"written_utc": datetime.now(timezone.utc).isoformat(), "weekends": keys,
              "objective_version": "v4",
              "objective": "V4: race state on the first stop (constants leave-two-out inside a block), "
                           "lambda on the later stops, tau on the plan family, kappa fixed at 0",
              "dirty_air_by_circuit": dirty_circuits,
              "per_weekend": {k: {"implied_budgets": d.implied_budget(), "longest_stint": d.longest,
                                  "second_stop_median_green_by_stops": d.second_median_by_stops,
                                  "second_stop_n_by_stops": d.second_n_by_stops,
                                  "dirty_air": d.dirty, "dirty_air_circuit_history": d.dirty_hist,
                                  "dirty_air_used": d.dirty_circuit,
                                  "first_stop_median_green": d.first_median, "sc_set_first_stops": d.sc_set,
                                  "first_stop_prior": d.first_stop_prior_summary,
                                  "collapse_counts": d.collapse_counts,
                                  "collapse_rows": (d.cliff_rows.to_dict("records") if not d.cliff_rows.empty else []),
                                  "field_modal_seq": (d.seq_counts.index[0] if len(d.seq_counts) else None),
                                  "field_stop_mode": (int(d.stop_counts.idxmax()) if len(d.stop_counts) else None),
                                  "net_step": d.net, "self_regime": d.self_regime, "pit_loss": d.pit_loss}
                             for k, d in donors.items()},
              "global": {}, "loo": {}}
    print("\n== global (every scored weekend) ==")
    # the global block holds nothing out, so each donor's race state excludes
    # only its own race
    result["global"] = calibrate(list(donors.values()), grids, "global", dirty_circuits=dirty_circuits,
                                 held_out=None, v3_kappa=v3_kappa.get("_global"), rival_mode=args.rival_mode)
    for k in keys:
        print(f"\n== leave-one-out: {k} held out ==")
        result["loo"][k] = calibrate([d for kk, d in donors.items() if kk != k], grids, f"loo {k}",
                                     dirty_circuits=dirty_circuits, held_out=k, v3_kappa=v3_kappa.get(k),
                                     rival_mode=args.rival_mode)
    Path(args.out).write_text(json.dumps(result, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else
                                          int(o) if isinstance(o, np.integer) else str(o)))
    print(f"\nwrote {args.out} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
