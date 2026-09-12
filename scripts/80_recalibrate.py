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

* **grip budget per compound** - race-measured rate (driver + lap fixed
  effects) times the longest stint each compound was run to by a classified
  finisher; a stint that ends before the cliff is a lower bound, so the upper
  quartile across weekends is taken;
* **dirty air** - lap time lost within 3 s of the car ahead, with driver and
  lap fixed effects, median across weekends, floored at 0.05 s/lap (Monza's
  tow reads negative);
* **driver rate factors** - each driver's degradation relative to the field,
  pooled across the weekends they raced and shrunk toward 1.

What is calibrated against the field's revealed behaviour, one sweep at a
time on each donor weekend's *own* posterior:

* **management cost and wear floor** - so the push the optimiser chooses
  implies the practice->race factor the donor races measured;
* **undercut lambda** - so the recommended first stop lands on the field's
  median green-flag first stop (safety-car-set weekends excluded);
* **plan-prior tau, grid-start penalty** - so the recommended sequence is one
  the field ran, at the largest share.

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

from src import strategy as strat  # noqa: E402
from src.calibration import CALIBRATION_PATH, Calibration  # noqa: E402
from src.compounds import measure_dirty_air  # noqa: E402
from src.config import (  # noqa: E402
    DATA_PROCESSED, DIRTY_AIR_S_PER_LAP, EVENTS, GRID_START_PENALTY_S, GRIP_BUDGET_S, MANAGE_COST_S,
    MANAGE_WEAR_FLOOR, PLAN_PRIOR_TAU_S, SIGMA_RACE_LAP_S, UNDERCUT_EXPOSURE_LAMBDA, VALID_COMPOUNDS, get_event,
)
from src.history import race_deg_slopes, race_driver_factors  # noqa: E402
from src.model_bayes import BayesFit  # noqa: E402
from src.regime import RegimeFactor  # noqa: E402
from src.tyre import TyreModel  # noqa: E402

N_DRAWS = 150
SEARCH_STEP = 2
SHORTLIST = 600
DRIVER_PRIOR_LN_SD = 0.15       # shrinkage of a driver's rate factor toward the field
DRIVER_SE_FLOOR = 0.10          # no single race pins a driver tighter than this (log scale)
GRIDS = {
    "lambda": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45],
    "tau": [0.0, 1.0, 2.5, 4.0, 6.0],
    "grid": [0.0, 0.5, 1.0, 1.5],
    "floor": [0.35, 0.45, 0.55],
    "cost": [0.6, 0.9, 1.2],
}
GRIDS_QUICK = {"lambda": [0.0, 0.1, 0.2, 0.3], "tau": [0.0, 2.5, 5.0], "grid": [0.0, 1.0],
               "floor": [0.45], "cost": [0.9]}


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
        self.first_median = float(np.median(green)) if green else None
        self.sc_set = bool(firsts and np.mean([s for _, s in firsts]) > 0.4)
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
        self._model_cache: dict = {}

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

    def search(self, cal: dict, *, lam: float, tau: float, grid: float, calibrated_model=None):
        m = calibrated_model if calibrated_model is not None else self.model(cal["budgets"], cal["floor"], cal["cost"])
        kw = dict(regime=self.regime, step=SEARCH_STEP, shortlist=SHORTLIST, support=self.support,
                  max_per_compound=self.alloc, max_stint=self.caps, undercut_lambda=lam,
                  plan_prior=self.plan_prior, plan_prior_tau_s=tau, traffic_s_per_lap=cal["dirty"],
                  grid_penalty_s=grid)
        return strat.simulate_model(m, self.ev, self.pit_loss, **kw)

    def calibrated_model(self, cal: dict, *, lam: float, tau: float, grid: float) -> TyreModel:
        m = self.model(cal["budgets"], cal["floor"], cal["cost"])
        if self.net is None or not np.isfinite(self.net):
            return m
        kw = dict(regime=self.regime, step=SEARCH_STEP, shortlist=SHORTLIST, support=self.support,
                  max_per_compound=self.alloc, max_stint=self.caps, undercut_lambda=lam,
                  plan_prior=self.plan_prior, plan_prior_tau_s=tau, traffic_s_per_lap=cal["dirty"],
                  grid_penalty_s=grid)
        m2, _, _ = strat.search_with_pace_calibration(m, self.ev, self.pit_loss, net_step_s=float(self.net),
                                                      net_step_se_s=float(self.net_se), **kw)
        return m2

    # -- the objectives -------------------------------------------------------

    def decision_scores(self, res) -> dict:
        if res.table.empty:
            return {"first_err": np.nan, "seq_share": 0.0, "start_ok": 0.0, "stops_ok": 0.0, "implied": np.nan}
        seq = "-".join(res.best["compounds"])
        share = float(self.seq_counts.get(seq, 0) / max(self.n_cls, 1))
        start_ok = float(self.start_counts.index[0] == res.best["compounds"][0]) if len(self.start_counts) else 0.0
        stops_ok = float(int(self.stop_counts.idxmax()) == int(res.best["n_stops"])) if len(self.stop_counts) else 0.0
        first_err = ((float(res.best["pit_laps"][0]) - self.first_median)
                     if (self.first_median is not None and res.best["pit_laps"] and not self.sc_set) else np.nan)
        return {"first_err": first_err, "seq_share": share, "start_ok": start_ok, "stops_ok": stops_ok,
                "implied": float(res.implied_regime), "best": res.best_label}


BUDGET_INFORMATIVE_S = 3.0      # below this, no stint that weekend came near the cliff: the product is only a bound
BUDGET_BAND_S = (3.0, 4.5)


def pool_budgets(donors: list) -> tuple:
    """Censoring-aware grip budget per compound.

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


def pool_dirty(donors: list) -> tuple:
    vals = [d.dirty["s_per_lap"] for d in donors if d.dirty]
    if not vals:
        return DIRTY_AIR_S_PER_LAP, []
    return float(np.clip(np.median(vals), 0.05, 0.8)), [round(v, 3) for v in vals]


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


def sweep(donors: list, cal: dict, name: str, values: list, objective, *, models=None) -> tuple:
    """Evaluate `objective(scores_by_donor)` along a grid for one constant."""
    rows = []
    for v in values:
        c = dict(cal); c[name] = v
        scores = []
        for d in donors:
            m = models.get(d.key) if models else None
            res = d.search(c, lam=c["lambda"], tau=c["tau"], grid=c["grid"], calibrated_model=m)
            scores.append(d.decision_scores(res))
        obj = objective(scores)
        rows.append({name: v, "objective": obj,
                     "mean_first_err": float(np.nanmean([s["first_err"] for s in scores])) if any(np.isfinite(s["first_err"]) for s in scores) else None,
                     "mean_abs_first_err": float(np.nanmean([abs(s["first_err"]) for s in scores])) if any(np.isfinite(s["first_err"]) for s in scores) else None,
                     "mean_seq_share": float(np.mean([s["seq_share"] for s in scores])),
                     "start_ok": float(np.mean([s["start_ok"] for s in scores])),
                     "stops_ok": float(np.mean([s["stops_ok"] for s in scores])),
                     "implied_vs_measured": float(np.nanmean([abs(np.log(max(s["implied"], 1e-3)) - np.log(max(d.self_regime or np.nan, 1e-3)))
                                                             for s, d in zip(scores, donors) if d.self_regime]))
                     if any(d.self_regime for d in donors) else None})
    best = min(rows, key=lambda r: r["objective"])
    return best[name], rows


def calibrate(donors: list, grids: dict, label: str) -> dict:
    t0 = time.time()
    pooled_b, per_b, raw_b = pool_budgets(donors)
    dirty, raw_d = pool_dirty(donors)
    drivers = pool_drivers(donors)
    cal = {"budgets": {**{c: pooled_b for c in VALID_COMPOUNDS}, **per_b}, "floor": MANAGE_WEAR_FLOOR, "cost": MANAGE_COST_S,
           "dirty": dirty, "lambda": UNDERCUT_EXPOSURE_LAMBDA, "tau": PLAN_PRIOR_TAU_S, "grid": GRID_START_PENALTY_S}
    sweeps = {}

    def obj_regime(scores):
        errs = [abs(np.log(max(s["implied"], 1e-3)) - np.log(max(d.self_regime, 1e-3)))
                for s, d in zip(scores, donors) if d.self_regime and np.isfinite(s["implied"])]
        return float(np.mean(errs)) if errs else 0.0

    def obj_first(scores):
        errs = [abs(s["first_err"]) for s in scores if np.isfinite(s["first_err"])]
        return float(np.mean(errs)) if errs else 0.0

    def obj_shape(scores):
        return float(-np.mean([s["seq_share"] for s in scores]) - 0.25 * np.mean([s["start_ok"] for s in scores])
                     - 0.25 * np.mean([s["stops_ok"] for s in scores]))

    # 1. management trade-off: floor and cost, on the uncalibrated pace (they set the push, not the compound)
    best_pair, best_obj, rows_m = None, np.inf, []
    for fl in grids["floor"]:
        for co in grids["cost"]:
            c = dict(cal); c["floor"] = fl; c["cost"] = co
            scores = [d.decision_scores(d.search(c, lam=c["lambda"], tau=c["tau"], grid=c["grid"])) for d in donors]
            o = obj_regime(scores)
            rows_m.append({"floor": fl, "cost": co, "objective": o,
                           "implied": [round(s["implied"], 2) for s in scores]})
            if o < best_obj:
                best_obj, best_pair = o, (fl, co)
    cal["floor"], cal["cost"] = best_pair
    sweeps["manage"] = rows_m
    # the calibrated-pace models at the chosen trade-off, reused by the remaining sweeps
    models = {d.key: d.calibrated_model(cal, lam=cal["lambda"], tau=cal["tau"], grid=cal["grid"]) for d in donors}
    # 2. undercut lambda against the field's first stops
    for _ in range(2):
        cal["lambda"], sweeps["lambda"] = sweep(donors, cal, "lambda", grids["lambda"], obj_first, models=models)
        cal["tau"], sweeps["tau"] = sweep(donors, cal, "tau", grids["tau"], obj_shape, models=models)
        cal["grid"], sweeps["grid"] = sweep(donors, cal, "grid", grids["grid"], obj_shape, models=models)
    # final scores at the chosen constants
    final = {d.key: d.decision_scores(d.search(cal, lam=cal["lambda"], tau=cal["tau"], grid=cal["grid"],
                                               calibrated_model=models[d.key])) for d in donors}
    out = {
        "grip_budget_s": pooled_b, "grip_budget_by_compound": per_b,
        "manage_cost_s": cal["cost"], "manage_wear_floor": cal["floor"],
        "grid_start_penalty_s": cal["grid"], "dirty_air_s_per_lap": cal["dirty"],
        "undercut_lambda": cal["lambda"], "plan_prior_tau_s": cal["tau"],
        "sigma_race_lap_s": SIGMA_RACE_LAP_S,
        "driver_factors": {k: v["factor"] for k, v in drivers.items()},
        "driver_factor_detail": drivers,
        "donors": [d.key for d in donors],
        "raw": {"implied_budgets": raw_b, "dirty_air": raw_d},
        "sweeps": sweeps, "final_scores": final,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  [{label}] budgets {per_b} (pooled {pooled_b:.2f}); floor {cal['floor']} cost {cal['cost']}; dirty {dirty:.3f}; "
          f"lambda {cal['lambda']}; tau {cal['tau']}; grid {cal['grid']}; {len(drivers)} drivers; {out['seconds']}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--events", nargs="*", default=None)
    ap.add_argument("--out", default=str(CALIBRATION_PATH))
    args = ap.parse_args()
    grids = GRIDS_QUICK if args.quick else GRIDS
    keys = args.events or scored_weekends()
    print(f"scored weekends: {keys}")
    t0 = time.time()
    donors = {k: Donor(k) for k in keys}
    print(f"loaded {len(donors)} weekends in {time.time()-t0:.0f}s")
    for k, d in donors.items():
        print(f"  {k}: implied budgets {d.implied_budget()}; dirty air {d.dirty.get('s_per_lap', float('nan')):+.3f}; "
              f"first stop median (green) {d.first_median} sc-set {d.sc_set}; net {d.net}; self regime {d.self_regime}")
    result = {"written_utc": datetime.now(timezone.utc).isoformat(), "weekends": keys,
              "per_weekend": {k: {"implied_budgets": d.implied_budget(), "longest_stint": d.longest,
                                  "dirty_air": d.dirty, "first_stop_median_green": d.first_median, "sc_set_first_stops": d.sc_set,
                                  "field_modal_seq": (d.seq_counts.index[0] if len(d.seq_counts) else None),
                                  "field_stop_mode": (int(d.stop_counts.idxmax()) if len(d.stop_counts) else None),
                                  "net_step": d.net, "self_regime": d.self_regime, "pit_loss": d.pit_loss}
                             for k, d in donors.items()},
              "global": {}, "loo": {}}
    print("\n== global (every scored weekend) ==")
    result["global"] = calibrate(list(donors.values()), grids, "global")
    for k in keys:
        print(f"\n== leave-one-out: {k} held out ==")
        result["loo"][k] = calibrate([d for kk, d in donors.items() if kk != k], grids, f"loo {k}")
    Path(args.out).write_text(json.dumps(result, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else
                                          int(o) if isinstance(o, np.integer) else str(o)))
    print(f"\nwrote {args.out} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
