"""Stability benchmark: how much does the answer move with (a) the sampler
seed and (b) how much practice has run?

Weekend-setting fits (2 chains x 800 warmup x 800 draws, lap-time channel,
linear, soft per-circuit ladder) on Barcelona 2026 and Hungary 2026:

  sessions  FP1 | FP1+FP2 | FP1+FP2+FP3   (seed 0)
  seeds     0 | 1 | 2                       (all sessions)

For each fit: per-compound slope, the strategy recommendation (the same
search as the pipeline - calibrated constants, pace calibration, position
term, first-stop prior at the calibrated kappa, this circuit's dirty air -
300 draws, 1-lap grid), stint-rate MAE vs the race, wall time.
Circuit history is folded in exactly as the weekend script does.

Usage: python bench/bench_stability.py [--events barcelona-2026 hungary-2026]
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import (EVENTS, arg_events, dirty_air_of, dump, first_stop_tables, fs_kwargs, kappa_of,  # noqa: E402
                    memoise_regime, meta, offline, race_table)
from src import strategy as strat
from src.calibration import get_calibration
from src.compounds import pace_step_prior
from src.config import WEEKEND_NUTS_CHAINS, WEEKEND_NUTS_DRAWS, WEEKEND_NUTS_WARMUP, get_event
from src.evolution import add_evolution_correction, fit_evolution_auto
from src.fuel import add_fuel_correction
from src.history import apply_circuit_prior, circuit_prior, plan_prior_for, stint_caps_for
from src.ingest import load_for_fitting
from src.laps import build_lap_table, clean_laps
from src.model_bayes import fit_bayes
from src.regime import RegimeFactor
from src.tyre import EXTRAP_LN_SD_MEASURED, TyreModel
from src.validate import _curve_block, score_race

AGES = np.arange(0, 41, dtype=float)
DEFAULT_EVENTS = ["barcelona-2026", "hungary-2026"]


def one(key: str, sessions: list, seed: int, cp, regime, m, race_clean, cal) -> dict:
    ev = get_event(key)
    t0 = time.perf_counter()
    raw = load_for_fitting(ev, sessions)
    laps = build_lap_table(raw, ev)
    clean = clean_laps(laps)
    clean = add_fuel_correction(clean, ev, "2026")
    evo = fit_evolution_auto(clean, laps_all=laps, event=ev)
    clean = add_evolution_correction(clean, evo)
    t_prep = time.perf_counter() - t0
    pstep = pace_step_prior(ev, circuit=cp)
    ladder = cp.ladder if cp.available else None
    t1 = time.perf_counter()
    f = fit_bayes(clean, ev, prior="2026", compound_prior="soft", circuit_ladder=ladder,
                  chains=WEEKEND_NUTS_CHAINS, warmup=WEEKEND_NUTS_WARMUP, draws=WEEKEND_NUTS_DRAWS, seed=seed)
    t_fit = time.perf_counter() - t1
    slopes_prac = {r["compound"]: round(r["slope_s_per_lap"], 4) for _, r in f.slope_table().iterrows()}
    if cp.available and cp.rate_prior:
        f, _ = apply_circuit_prior(f, ev, regime, cp)
    slopes = {r["compound"]: round(r["slope_s_per_lap"], 4) for _, r in f.slope_table().iterrows()}
    t2 = time.perf_counter()
    caps = stint_caps_for(ev, cp) if cp.available else None
    total = f.posterior["lin"].shape[0]
    idx = np.random.default_rng(0).choice(total, size=min(300, total), replace=False)
    model = TyreModel.from_fit(f, draws=idx, budget=cal.budgets, manage_floor=cal.manage_wear_floor, manage_cost_s=cal.manage_cost_s,
                               support=clean.groupby("compound")["tyre_age"].max().to_dict(),
                               extrap_ln_sd=EXTRAP_LN_SD_MEASURED)
    # the shipped objective: V4's race-state first stop (constants from the
    # other races), which replaces the first-stop prior
    from src import racestate
    kw = dict(regime=regime, support=clean.groupby("compound")["tyre_age"].max().to_dict(),
              max_per_compound=m["allocation"]["caps"], max_stint=caps, undercut_lambda=cal.undercut_lambda,
              plan_prior=plan_prior_for(cp), plan_prior_tau_s=cal.plan_prior_tau_s,
              traffic_s_per_lap=dirty_air_of(cal, ev.circuit), grid_penalty_s=cal.grid_start_penalty_s,
              race_state=racestate.measure_constants(exclude=ev.key))
    _, res, _ = strat.search_with_pace_calibration(model, ev, float(m["pit_loss_s"]),
                                                   net_step_s=float(pstep.get("net_stint_step_measured", np.nan)),
                                                   net_step_se_s=float(pstep.get("net_stint_step_se") or 0.0), **kw)
    t_strat = time.perf_counter() - t2
    rm = regime.draws(f.posterior["lin"].shape[0], seed=5)
    sealed = {"ages": AGES.tolist(), "curves": {},
              "race_curves": {c: _curve_block(f.deg_loss(c, AGES) * rm[:, None]) for c in f.compounds},
              "sigma_obs": float(f.posterior["sigma_obs"].mean()), "sigma_race": cal.sigma_race_lap_s, "knee": {}}
    sc = score_race(sealed, race_clean, ev)
    return {"sessions": sessions, "seed": seed, "n_clean": int(len(clean)), "n_apex": int(f.n_apex),
            "rhat": round(float(f.max_rhat), 4), "div": int(f.n_divergences),
            "slopes_practice": slopes_prac, "slopes_final": slopes,
            "best": res.best_label, "tyre_optimal": res.tyre_optimal_label, "push": res.best.get("push"),
            "first_stop": (int(res.best["pit_laps"][0]) if res.best["pit_laps"] else None),
            "first_stop_s": res.best.get("first_stop_s"),
            "by_stops": {int(r["n_stops"]): (r["compounds"], round(float(r["delta_s"]), 1), round(float(r["win_prob_any"]), 2))
                         for _, r in res.by_stops.iterrows()},
            "rate_mae": round(float(sc.mae), 4), "rate_bias": round(float(sc.bias), 4),
            "cov90": round(float(sc.coverage.get(0.9, np.nan)), 3), "rate_cov90": round(float(sc.rate_coverage90), 3),
            "t_prep_s": round(t_prep, 1), "t_fit_s": round(t_fit, 1), "t_strategy_s": round(t_strat, 1)}


def main() -> None:
    args = arg_events(__doc__)
    offline()
    memoise_regime()
    # the stability run is two weekends by design (it refits them five times
    # each); `--events` narrows or widens that, it does not default to all seven
    keys = args.events if args.events != list(EVENTS) else DEFAULT_EVENTS
    out = {}
    for key in keys:
        ev = get_event(key)
        m = meta(key)
        cal = get_calibration(ev)
        regime = RegimeFactor(ratio=float(m["regime"]["ratio"]), ln_sd=float(m["regime"]["ln_sd"]),
                              measured=True, label="from meta")
        cp = circuit_prior(ev, probe_practice_temp=False)
        race_clean = clean_laps(race_table(key))
        sess = list(ev.practice_sessions)
        runs = []
        subsets = [sess[:i] for i in range(1, len(sess) + 1)]
        for s in subsets:
            r = one(key, s, 0, cp, regime, m, race_clean, cal)
            runs.append(r)
            print(f"{key} {s} seed0: clean {r['n_clean']} fit {r['t_fit_s']}s rhat {r['rhat']} slopes {r['slopes_final']} "
                  f"best {r['best']} MAE {r['rate_mae']} bias {r['rate_bias']:+.3f}", flush=True)
        for seed in (1, 2):
            r = one(key, sess, seed, cp, regime, m, race_clean, cal)
            runs.append(r)
            print(f"{key} all seed{seed}: fit {r['t_fit_s']}s rhat {r['rhat']} slopes {r['slopes_final']} best {r['best']} "
                  f"MAE {r['rate_mae']}", flush=True)
        out[key] = runs
        dump("stability.json", out)
    print(pd.DataFrame([{k: v for k, v in r.items() if k not in ("by_stops",)} for k in out for r in out[k]]).to_string())


if __name__ == "__main__":
    main()
