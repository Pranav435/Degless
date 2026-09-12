"""Speed benchmark: wall time of every stage of the offline pipeline on one
weekend, at the production settings and at the weekend (quick) settings, plus
the strategy-desk calls, the live tick and the app cold start.

The V3 rows run the V3 objective — the first-stop penalty tables at the
calibrated kappa and this circuit's own dirty-air cost — under their original
names, so earlier runs compare row for row.  The V3 stages added to the table
are the within-stint collapse detector (`cliff.race_collapses`), the
stint-fixed-effects gate (`model_fallback.stint_fe_baseline`) and the pit-window
sweep with the first-stop term in it.  V4 adds the shipped race-state rows: the
constants, the search with the race-state first stop, its window sweep and the
per-car plans under it.

Peak memory is `resource.getrusage` where it exists and the process's peak
working set on Windows.

The one call in bench/ that writes outside bench/ is `seal_predictions`, timed
here and then deleted again (`common.discard_sealed`), so `predictions/sealed/`
is left exactly as `make history` wrote it.

Usage: python bench/bench_speed.py [--events barcelona-2026]  (the first event is timed)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

try:
    import resource

    def _peak_rss_mb() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
except ImportError:            # Windows has no `resource`: the process's peak working set
    import ctypes
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    def _peak_rss_mb() -> float:
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE          # a 64-bit pseudo-handle, not a C int
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return float("nan")
        return pmc.PeakWorkingSetSize / (1024 ** 2)

from common import (EVENTS, ROOT, Timer, arg_events, cp_for, dirty_air_of, discard_sealed,  # noqa: E402
                    dump, first_stop_tables, fs_kwargs, kappa_of, memoise_regime, meta, offline,
                    race_table)


def main() -> None:
    args = arg_events(__doc__)
    offline()
    memoise_regime()
    key = args.events[0]
    T = Timer()
    t_import = time.perf_counter()
    import jax  # noqa: F401
    import numpyro  # noqa: F401
    T.rows.append({"stage": "import jax+numpyro", "seconds": round(time.perf_counter() - t_import, 3)})

    from src import strategy as strat
    from src.calibration import get_calibration
    from src.compounds import pace_step_prior
    from src.config import (DATA_PROCESSED, VALID_COMPOUNDS, WEEKEND_NUTS_CHAINS, WEEKEND_NUTS_DRAWS,
                            WEEKEND_NUTS_WARMUP, get_event)
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

    try:
        from src.cliff import race_collapses
    except ImportError:
        race_collapses = None
    try:
        from src.model_fallback import stint_fe_baseline
    except ImportError:
        stint_fe_baseline = None
    try:
        from src.firststop import first_stop_penalty_table
    except ImportError:
        first_stop_penalty_table = None

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
    if stint_fe_baseline is not None:
        with T("stint-FE baseline, 50 stint bootstraps (the gate)"):
            fe_base = stint_fe_baseline(clean, n_boot=50)
    if first_stop_penalty_table is not None and (cp.first_stop_green if cp.available else None):
        with T("first-stop prior: KDE penalty table per start compound"):
            first_stop_penalty_table(cp.first_stop_green, ev.n_race_laps, list(VALID_COMPOUNDS))
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
        sealed_path, _ = seal_predictions(f, ev, regime=regime, cliff=cp.cliff(), note="benchmark run; discard")
    discard_sealed(sealed_path)      # bench/ must leave predictions/sealed/ untouched
    with T("load race laps + lap table"):
        race = build_lap_table(load_race(ev), ev)
        race_clean = clean_laps(race)
    with T("score race"):
        from common import sealed_for
        score_race(sealed_for(key), race_clean, ev)
    if race_collapses is not None:
        with T("within-stint collapse detector, whole race"):
            collapses = race_collapses(race, event=ev)
    pit_loss = float(m["pit_loss_s"])
    caps = stint_caps_for(ev, cp) if cp.available else None
    support = clean.groupby("compound")["tyre_age"].max().to_dict()
    total = f.posterior["lin"].shape[0]
    idx = np.random.default_rng(0).choice(total, size=500, replace=False)
    model = TyreModel.from_fit(f, draws=idx, budget=cal.budgets, manage_floor=cal.manage_wear_floor, manage_cost_s=cal.manage_cost_s)
    fs_tables = first_stop_tables(cp, ev, list(model.compounds))
    kappa = kappa_of(cal)
    kw = dict(regime=regime, support=support, max_per_compound=m["allocation"]["caps"], max_stint=caps,
              undercut_lambda=cal.undercut_lambda, plan_prior=pp, plan_prior_tau_s=cal.plan_prior_tau_s,
              traffic_s_per_lap=dirty_air_of(cal, ev.circuit), grid_penalty_s=cal.grid_start_penalty_s,
              **fs_kwargs(strat.simulate_model, fs_tables, kappa))
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
    with T("pit window sweep (position term + first-stop prior)"):
        strat.pit_window_model(model, ev, res.best, pit_loss, max_stint=res.max_stint, push=res.best["push"],
                               undercut_lambda=cal.undercut_lambda,
                               **fs_kwargs(strat.pit_window_model, fs_tables, kappa))
    # V4: the shipped objective times the first stop with the race state (its
    # constants measured on the other races, the first-stop prior off)
    from src import racestate
    with T("race-state constants (measured on the donor races)"):
        rs_c = racestate.measure_constants(exclude=key)
    kw4 = {**{k: v for k, v in kw.items() if k not in ("first_stop_prior", "first_stop_kappa_s")}, "race_state": rs_c}
    with T("strategy search 500 draws, 1-lap grid, race-state objective (V4 production)"):
        res4 = strat.simulate_model(model, ev, pit_loss, step=1, **kw4)
    with T("pit window sweep (race-state term)"):
        strat.pit_window_model(model, ev, res4.best, pit_loss, max_stint=res4.max_stint, push=res4.best["push"],
                               undercut_lambda=cal.undercut_lambda,
                               race_state_term=racestate.term_by_lap(res4.race_state.get("best"), ev.n_race_laps))
    from src.objective import terms_by_group

    rs_terms = terms_by_group(res4, ev.n_race_laps)      # V4: every plan group's first-stop term
    with T("per-car plans (20 drivers, 2-lap grid, race-state objective)"):
        strat.per_driver_plans(model, ev, pit_loss, sorted(race["driver"].unique()), race_factors=cal.driver_factors,
                               **kw4)
    with T("undercut window"):
        strat.undercut_window_model(model, "MEDIUM", "SOFT", max_age=30, push=res.best["push"])
    with T("counterfactual (all drivers, vectorised, SC-aware, per-car, V4 objective)"):
        strat.counterfactual(model, ev, race, pit_loss, max_stint=res.max_stint, push=res.best["push"],
                             race_factors=cal.driver_factors, undercut_lambda=cal.undercut_lambda,
                             traffic_s_per_lap=dirty_air_of(cal, ev.circuit), race_state_terms=rs_terms)
    with T("counterfactual (V3 objective baseline)"):
        strat.counterfactual(model, ev, race, pit_loss, max_stint=res.max_stint, push=res.best["push"],
                             race_factors=cal.driver_factors, undercut_lambda=cal.undercut_lambda,
                             **fs_kwargs(strat.counterfactual, fs_tables, kappa))
    with T("per-car plans (20 drivers, 2-lap grid)"):
        strat.per_driver_plans(model, ev, pit_loss, sorted(race["driver"].unique()), race_factors=cal.driver_factors, **kw)
    with T("replay precompute"):
        build_replay(f, ev, race)
    plans = [{"compounds": res.best["compounds"], "pit_laps": res.best["pit_laps"]},
             {"compounds": ["MEDIUM", "HARD"], "pit_laps": [ev.n_race_laps // 2]},
             {"compounds": ["SOFT", "HARD", "HARD"], "pit_laps": [15, 40]}]
    # The desk prices a hand-built plan on the shipped V4 objective: the plan
    # group's race-state term on the first stop (from the race-state search
    # above), lambda on the later stops, tau on the family.  The V3 row is kept
    # beside it so the report can say what the terms cost.
    with T("desk: evaluate 3 plans (V4 objective, race-state terms)"):
        strat.evaluate_plans(model, ev, plans, pit_loss, undercut_lambda=cal.undercut_lambda, plan_prior=pp,
                             plan_prior_tau_s=cal.plan_prior_tau_s,
                             traffic_s_per_lap=dirty_air_of(cal, ev.circuit),
                             grid_penalty_s=cal.grid_start_penalty_s, race_state_terms=rs_terms)
    with T("desk: evaluate 3 plans (V3 objective baseline)"):
        strat.evaluate_plans(model, ev, plans, pit_loss, undercut_lambda=cal.undercut_lambda, plan_prior=pp,
                             plan_prior_tau_s=cal.plan_prior_tau_s,
                             **fs_kwargs(strat.evaluate_plans, fs_tables, kappa))
    with T("desk: degradation crossover (21 multipliers, V4 objective)"):
        strat.deg_crossover(model, ev, plans[0], plans[1], pit_loss, race_state_terms=rs_terms)
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
    peak_mb = _peak_rss_mb()
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
           "pipeline_runtime_recorded_s": recorded, "weekend_runtime_recorded": weekend,
           "objective": {"first_stop_prior": bool(fs_tables), "first_stop_kappa_s": kappa,
                         "dirty_air_s_per_lap": dirty_air_of(cal, ev.circuit),
                         "dirty_air_pooled": float(cal.dirty_air_s_per_lap),
                         "percar_mode": getattr(cal, "percar_mode", None)},
           "v3_stages": {"n_collapse_rows": (int(len(collapses)) if race_collapses is not None else None),
                         "n_collapses": (int(collapses["collapse"].fillna(False).astype(bool).sum())
                                         if race_collapses is not None and len(collapses) else None),
                         "stint_fe_pooled_slope": ((fe_base or {}).get("pooled_slope")
                                                   if stint_fe_baseline is not None else None)}}
    dump("speed.json", out)
    print(tbl.to_string(index=False))
    print(f"peak RSS {peak_mb:.0f} MB; app exceptions {app_exc}; import re-timed {imp}s")
    print("recorded full-pipeline runtimes:", {k: v["total"] for k, v in recorded.items()})
    print("recorded weekend-script runtimes:", {k: v["runtime_s"] for k, v in weekend.items()})


if __name__ == "__main__":
    main()
