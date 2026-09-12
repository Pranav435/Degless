"""Speed benchmark: wall time of every stage of the offline pipeline on one
weekend, at the production settings and at the weekend (quick) settings, plus
the strategy-desk calls, the live tick and the app cold start.

Usage: python bench/bench_speed.py [barcelona-2026]
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from common import EVENTS, ROOT, Timer, dump, meta, offline, race_table  # noqa: E402


def main() -> None:
    offline()
    key = sys.argv[1] if len(sys.argv) > 1 else "barcelona-2026"
    T = Timer()
    t_import = time.perf_counter()
    import jax  # noqa: F401
    import numpyro  # noqa: F401
    T.rows.append({"stage": "import jax+numpyro", "seconds": round(time.perf_counter() - t_import, 3)})

    from src import strategy as strat
    from src.calibration import get_calibration
    from src.compounds import pace_step_prior
    from src.config import DATA_PROCESSED, WEEKEND_NUTS_CHAINS, WEEKEND_NUTS_DRAWS, WEEKEND_NUTS_WARMUP, get_event
    from src.evolution import add_evolution_correction, fit_evolution_auto
    from src.fuel import add_fuel_correction
    from src.history import apply_circuit_prior, circuit_prior, plan_prior_for, stint_caps_for
    from src.ingest import load_for_fitting, load_race
    from src.laps import build_lap_table, cascade_counts, clean_laps
    from src.model_bayes import fit_bayes
    from src.model_fallback import fit_mixedlm
    from src.regime import RegimeFactor, regime_prior
    from src.replay import build_replay
    from src.telemetry import load_apex, select_corners
    from src.tyre import TyreModel
    from src.validate import score_race, seal_predictions

    ev = get_event(key)
    m = meta(key)
    cal = get_calibration(ev)
    qk = dict(chains=WEEKEND_NUTS_CHAINS, warmup=WEEKEND_NUTS_WARMUP, draws=WEEKEND_NUTS_DRAWS)
    with T("load practice laps (FastF1 cache)"):
        raw = load_for_fitting(ev)
    with T("lap table + clean cascade"):
        laps = build_lap_table(raw, ev)
        cascade_counts(laps)
        clean = clean_laps(laps)
    with T("fuel + track evolution"):
        clean = add_fuel_correction(clean, ev, "2026")
        evo = fit_evolution_auto(clean, laps_all=laps, event=ev)
        clean = add_evolution_correction(clean, evo)
    with T("circuit history prior (cached) + ladder + plan prior"):
        cp = circuit_prior(ev, probe_practice_temp=True)
        pp = plan_prior_for(cp)
    with T("compound pace-step prior (donor races + circuit)"):
        pstep = pace_step_prior(ev, circuit=cp)
    with T("regime prior (donor races, temperature model)"):
        regime = regime_prior(ev, clean=clean)
    with T("MixedLM, no bootstrap"):
        fit_mixedlm(clean, n_boot=0)
    with T("MixedLM, 50 stint bootstraps"):
        fit_mixedlm(clean, n_boot=50)
    ladder = cp.ladder if cp.available else None
    with T("NUTS 4x1500x1500 lap-time only, linear, soft ladder (production)"):
        f = fit_bayes(clean, ev, prior="2026", compound_prior="soft", circuit_ladder=ladder)
    with T("NUTS 2x800x800 lap-time only (weekend refit)"):
        fit_bayes(clean, ev, prior="2026", compound_prior="soft", circuit_ladder=ladder, **qk)
    with T("NUTS 2x800x800 lap-time only, hinge (diagnostic)"):
        fit_bayes(clean, ev, prior="2026", compound_prior="soft", circuit_ladder=ladder, use_hinge=True, **qk)
    with T("apex speeds (cached) + corner selection"):
        apex = load_apex(ev)
        sel = select_corners(apex)
        apex_use = apex[apex["corner"].isin(sel.corners)]
    with T("NUTS 4x1500x1500 joint lap-time + apex (diagnostic)"):
        fit_bayes(clean, ev, prior="2026", compound_prior="soft", circuit_ladder=ladder, apex=apex_use)
    with T("circuit history combination"):
        if cp.available and cp.rate_prior:
            f, _ = apply_circuit_prior(f, ev, regime, cp)
    with T("seal predictions"):
        seal_predictions(f, ev, regime=regime, cliff=cp.cliff(), note="benchmark run; discard")
    with T("load race laps + lap table"):
        race = build_lap_table(load_race(ev), ev)
        race_clean = clean_laps(race)
    with T("score race"):
        from common import sealed_for
        score_race(sealed_for(key), race_clean, ev)
    pit_loss = float(m["pit_loss_s"])
    caps = stint_caps_for(ev, cp) if cp.available else None
    support = clean.groupby("compound")["tyre_age"].max().to_dict()
    total = f.posterior["lin"].shape[0]
    idx = np.random.default_rng(0).choice(total, size=500, replace=False)
    model = TyreModel.from_fit(f, draws=idx, budget=cal.budgets, manage_floor=cal.manage_wear_floor, manage_cost_s=cal.manage_cost_s)
    kw = dict(regime=regime, support=support, max_per_compound=m["allocation"]["caps"], max_stint=caps,
              undercut_lambda=cal.undercut_lambda, plan_prior=pp, plan_prior_tau_s=cal.plan_prior_tau_s,
              traffic_s_per_lap=cal.dirty_air_s_per_lap, grid_penalty_s=cal.grid_start_penalty_s)
    with T("strategy search 500 draws, 1-lap grid, full objective (production)"):
        res = strat.simulate_model(model, ev, pit_loss, step=1, **kw)
    with T("strategy search with pace calibration (2-3 searches)"):
        model, res, _ = strat.search_with_pace_calibration(model, ev, pit_loss, net_step_s=float(pstep.get("net_stint_step_measured", np.nan)),
                                                           net_step_se_s=float(pstep.get("net_stint_step_se") or 0.0), step=1, **kw)
    with T("strategy search 500 draws, 2-lap grid"):
        strat.simulate_model(model, ev, pit_loss, step=2, **kw)
    with T("strategy search 200 draws, 1-lap grid"):
        strat.simulate_model(model.subsample(np.arange(200)), ev, pit_loss, step=1, **kw)
    with T("pit window sweep (position term)"):
        strat.pit_window_model(model, ev, res.best, pit_loss, max_stint=res.max_stint, push=res.best["push"],
                               undercut_lambda=cal.undercut_lambda)
    with T("undercut window"):
        strat.undercut_window_model(model, "MEDIUM", "SOFT", max_age=30, push=res.best["push"])
    with T("counterfactual (all drivers, vectorised, SC-aware, per-car)"):
        strat.counterfactual(model, ev, race, pit_loss, max_stint=res.max_stint, push=res.best["push"],
                             race_factors=cal.driver_factors, undercut_lambda=cal.undercut_lambda)
    with T("per-car plans (20 drivers, 2-lap grid)"):
        strat.per_driver_plans(model, ev, pit_loss, sorted(race["driver"].unique()), race_factors=cal.driver_factors, **kw)
    with T("replay precompute"):
        build_replay(f, ev, race)
    plans = [{"compounds": res.best["compounds"], "pit_laps": res.best["pit_laps"]},
             {"compounds": ["MEDIUM", "HARD"], "pit_laps": [ev.n_race_laps // 2]},
             {"compounds": ["SOFT", "HARD", "HARD"], "pit_laps": [15, 40]}]
    with T("desk: evaluate 3 plans"):
        strat.evaluate_plans(model, ev, plans, pit_loss, undercut_lambda=cal.undercut_lambda, plan_prior=pp, plan_prior_tau_s=cal.plan_prior_tau_s)
    with T("desk: degradation crossover (21 multipliers)"):
        strat.deg_crossover(model, ev, plans[0], plans[1], pit_loss)
    with T("desk: safety-car playbook"):
        strat.sc_playbook(model, ev, {**plans[0], "push": res.best["push"]}, pit_loss)
    with T("desk: value of information"):
        strat.value_of_information(res)
    with T("desk: undercut duel"):
        strat.undercut_duel(model, my_compound="MEDIUM", my_age=12, their_compound="MEDIUM", their_age=14,
                            gap_s=2.0, new_compound="HARD")
    with T("outlook build (300 draws, scenarios)"):
        from src.outlook import build
        build(ev, write=False)
    with T("app cold start (streamlit AppTest)"):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120)
        at.run()
        app_exc = len(at.exception)
    tbl = T.table()
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    recorded = {k: {"total": meta(k).get("runtime_s"), "decide": meta(k).get("runtime_decide_s"),
                    "timings": meta(k).get("timings")} for k in EVENTS}
    weekend = {}
    for p in DATA_PROCESSED.glob("weekend_*.json"):
        w = json.loads(p.read_text())
        weekend[w["event"]] = {"runtime_s": w.get("runtime_s"), "sessions": w.get("sessions_used"), "timings": w.get("timings")}
    imp = subprocess.run([sys.executable, "-c", "import time;t=time.time();import jax,numpyro;print(time.time()-t)"],
                         capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = {"event": key, "n_clean_laps": int(len(clean)), "n_apex": int(len(apex_use)),
           "n_strategies": int(res.n_strategies), "stages": tbl.to_dict("records"),
           "peak_rss_mb": round(peak_mb, 0), "app_exceptions": app_exc, "import_retimed_s": imp,
           "pipeline_runtime_recorded_s": recorded, "weekend_runtime_recorded": weekend}
    dump("speed.json", out)
    print(tbl.to_string(index=False))
    print(f"peak RSS {peak_mb:.0f} MB; app exceptions {app_exc}; import re-timed {imp}s")
    print("recorded full-pipeline runtimes:", {k: v["total"] for k, v in recorded.items()})
    print("recorded weekend-script runtimes:", {k: v["runtime_s"] for k, v in weekend.items()})


if __name__ == "__main__":
    main()
