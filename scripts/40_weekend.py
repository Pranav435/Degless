"""Weekend model: fit on whatever practice has run so far, seal, and write the
pre-race plan.  No race data is needed, or read.

This is the script to run on Friday night and again after FP3 - the
supervisor does it by itself after every practice session.  One fit, at the
quick sampler settings, on the lap-time channel alone; the circuit's history
folded into the rate; the calibrated constants; the ladder gate enforced.
Under 30 s on a quiet disk.

    .venv/bin/python scripts/40_weekend.py --event italy-2026
    .venv/bin/python scripts/40_weekend.py --event italy-2026 --sessions "Practice 1" "Practice 2"
    .venv/bin/python scripts/40_weekend.py --event italy-2026 --full     # 4 chains x 1500 + 1500
    .venv/bin/python scripts/40_weekend.py --event italy-2026 --apex     # add the apex-speed channel (diagnostic)

Outputs (all under data/processed/):
    posterior_<key>.npz/.json   posterior draws for the live engine
    weekend_<key>.json          everything the app's pre-race view and the engine need
    curves_/life_/plan_/bystops_/pitwindow_/undercut_/strategy_/perdriver_<key>.parquet
    clean_<key>_practice.parquet, cascade_<key>.parquet, laps_<key>_practice.parquet
predictions/sealed/<key>_<utc>.json (+ sha256)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import firststop, haascar, objective, percar, strategy as strat  # noqa: E402
from src.objective import V4Objective  # noqa: E402
from src.calibration import get_calibration  # noqa: E402
from src.compounds import allocation_prior, model_net_step_draws, pace_step_prior, summary_table as compound_table  # noqa: E402
from src.config import (  # noqa: E402
    DATA_PROCESSED, MC_DRAWS, RHAT_GATE, TYRE_LOAD_EXPONENT, VALID_COMPOUNDS, WEEKEND_NUTS_CHAINS,
    WEEKEND_NUTS_DRAWS, WEEKEND_NUTS_WARMUP, get_event,
)
from src.evolution import add_evolution_correction, fit_evolution_auto  # noqa: E402
from src.fuel import add_fuel_correction, get_prior, summary_table  # noqa: E402
from src.ingest import FirewallError, load_for_fitting  # noqa: E402
from src.laps import build_lap_table, cascade_counts, clean_laps, compound_summary  # noqa: E402
from src.history import apply_circuit_prior, circuit_prior, plan_prior_for, stint_caps_for  # noqa: E402
from src.live.engine import pit_loss_prior  # noqa: E402
from src.model_bayes import fit_bayes  # noqa: E402
from src.model_fallback import fit_mixedlm  # noqa: E402
from src.regime import regime_prior  # noqa: E402
from src.telemetry import extract_apex_speeds, load_apex, save_apex, select_corners  # noqa: E402
from src.tyre import EXTRAP_LN_SD_MEASURED, TyreModel  # noqa: E402
from src.validate import load_sealed, seal_predictions  # noqa: E402

log = logging.getLogger("degless.weekend")
GATES: list = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    GATES.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def step(msg: str) -> float:
    print(f"\n=== {msg} ===", flush=True)
    return time.time()


def _jd(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="practice sessions to fit on (default: every one that has run)")
    ap.add_argument("--apex", action="store_true", help="add the apex-speed channel (diagnostic; 3x the fit time)")
    ap.add_argument("--full", action="store_true", help="4 chains x 1500 + 1500 instead of 2 x 800 + 800")
    ap.add_argument("--quick", action="store_true", help="(kept for the supervisor; the quick settings are the default)")
    ap.add_argument("--mc-draws", type=int, default=MC_DRAWS)
    ap.add_argument("--boot", type=int, default=50)
    ap.add_argument("--pit-loss", type=float, default=None, help="override the pit-loss prior (s)")
    ap.add_argument("--race-temp", type=float, default=None, help="race-day track temperature forecast (degC)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ev = get_event(args.event)
    key = ev.key
    t_all = time.time()
    timings = {}

    # -- 0. firewall ---------------------------------------------------------
    step(f"0. {ev.name}: practice-only firewall")
    try:
        load_for_fitting(ev, "Race")
        gate("firewall blocks race data", False, "did NOT raise")
    except FirewallError as exc:
        gate("firewall blocks race data", True, str(exc)[:80])

    # -- 1. practice laps ----------------------------------------------------
    t = step("1. practice laps + clean-lap cascade")
    sessions = args.sessions or list(ev.practice_sessions)
    frames, used = [], []
    for s in sessions:
        try:
            df = load_for_fitting(ev, s)
        except Exception as exc:
            print(f"  {s}: not available ({str(exc)[:70]})")
            continue
        if not df.empty:
            frames.append(df)
            used.append(s)
    if not frames:
        print("no practice data yet"); return 1
    raw = pd.concat(frames, ignore_index=True)
    laps = build_lap_table(raw, ev)
    casc = cascade_counts(laps)
    clean = clean_laps(laps)
    comp_sum = compound_summary(laps)
    print(casc.to_string(index=False)); print(comp_sum.to_string(index=False))
    timings["load_s"] = round(time.time() - t, 1)
    gate("enough clean long-run laps (>= 60)", len(clean) >= 60,
         f"{len(clean)} clean laps from {len(laps)} raw over {used} ({time.time()-t:.0f}s)")
    gate("no non-green lap survives", bool((clean["track_status"].astype(str) == "1").all()), "")
    if len(clean) < 30:
        return 1

    # -- 2. physics + priors ------------------------------------------------
    t = step("2. fuel physics, circuit history, compound ladder, regime, allocation, pit loss, calibration")
    print(summary_table(ev).to_string(index=False))
    cp = circuit_prior(ev)
    if cp.available:
        print(f"  circuit history: {ev.circuit} {cp.years}; stops {cp.stops}; plans {cp.plans}; starts {cp.starts}")
        for c in ("SOFT", "MEDIUM", "HARD"):
            if c in cp.stint_typical:
                t_ = cp.stint_typical[c]
                print(f"    {c:7s} stints p50 {t_['p50']:.0f} p90 {t_['p90']:.0f} longest {cp.stint_longest.get(c, float('nan')):.0f} "
                      f"cap {cp.stint_cap.get(c)} laps  race deg {cp.rate_prior.get(c, {}).get('raw_mean_s_per_lap', float('nan')):.3f} s/lap"
                      f" x season {cp.season.get('factor', 1):.2f}")
        print(f"    ladder from the circuit's races: {cp.ladder}")
        th = cp.thermal
        if th.get("track_temp_now") is not None and th.get("track_temp_hist") is not None:
            print(f"    track {th['track_temp_now']:.0f}°C this weekend vs {th['track_temp_hist']:.0f}°C in those races: "
                  f"degradation x{th['multiplier']:.2f} ({th['beta_per_c']:+.3f}/°C)")
        if not cp.soft_race_tyre:
            print("    the SOFT has not been a race tyre here")
    else:
        print(f"  circuit history: none for {ev.circuit}")
    pstep = pace_step_prior(ev, circuit=cp if cp.available else None)
    print(f"  compound pace step prior: {pstep['step_s']:.3f} s [{pstep['label']}]; measured net "
          f"{pstep.get('net_stint_step_measured', float('nan')):+.3f} +/- {pstep.get('net_stint_step_se', float('nan')):.3f} s")
    clean = add_fuel_correction(clean, ev, "2026")
    evo = fit_evolution_auto(clean, laps_all=laps, event=ev)
    clean = add_evolution_correction(clean, evo)
    evo_rng = evo.iterations[-1]["evo_range_s"] if evo.iterations else 0.0
    _it = evo.iterations[-1] if evo.iterations else {}
    gate("track evolution identified & plausible (0-5 s)", 0.0 < evo_rng < 5.0,
         f"push-lap range {evo_rng:.2f}s per session {_it.get('per_session', {})}; long-run backfit "
         f"would have given {_it.get('backfit_range_s', float('nan')):.2f}s; sessions on the backfit: {evo.skipped}")
    # `--race-temp` is the *forecast* path and the only way a temperature term
    # enters: with it the mode is "forecast", without it "none" (the donors'
    # raw ratios, pooled with the median, plus the circuit's own history).
    regime = regime_prior(ev, clean=clean, race_temp_c=args.race_temp)
    print(f"  practice->race factor: {regime.ratio:.3f}x [{regime.p05:.2f}-{regime.p95:.2f}] ({regime.label}; "
          f"temperature mode: {(regime.temperature or {}).get('mode', 'none')})")
    print(f"    {regime.derivation}")
    alloc = allocation_prior(ev)
    print(f"  allocation: {alloc['caps']} ({'measured' if alloc['measured'] else 'default'})")
    pit_loss, pit_src = pit_loss_prior(ev)
    if cp.available and cp.pit_loss_s:
        pit_loss, pit_src = float(cp.pit_loss_s), f"this pit lane, {cp.years}"
    if args.pit_loss:
        pit_loss, pit_src = float(args.pit_loss), "override"
    print(f"  pit loss prior: {pit_loss:.1f} s ({pit_src})")
    cal = get_calibration(ev)
    dirty_air = cal.dirty_air_for(ev.circuit)
    print(f"  calibration: {cal.source}; budgets {cal.budgets}; lambda {cal.undercut_lambda:.3f}; "
          f"tau {cal.plan_prior_tau_s:.2f}; kappa {cal.first_stop_kappa_s:.2f}; dirty air {dirty_air:.2f} s/lap "
          f"({'this circuit' if ev.circuit in cal.dirty_air_by_circuit else 'pooled'})")
    plan_prior = plan_prior_for(cp if cp.available else None)
    if plan_prior:
        print(f"  plan-shape prior ({plan_prior.get('source')}): "
              + ", ".join(f"{k} {v}" for k, v in list(plan_prior["sequences"].items())[:5]))
    # The circuit's green-flag first stops as a density, from its past races only,
    # conditioned on the start compound and the plan's stop count
    fs_tables = firststop.first_stop_penalty_table(
        cp.first_stop_green if cp.available else None, ev.n_race_laps, list(VALID_COMPOUNDS))
    _starts = plan_prior.get("starts") or {}
    _stops_marginal = plan_prior.get("stops") or {}
    fs_summary = firststop.first_stop_summary(
        cp.first_stop_green if cp.available else None, ev.n_race_laps,
        start_compound=(max(_starts, key=_starts.get) if _starts else None),
        n_stops=(int(max(_stops_marginal, key=_stops_marginal.get)) if _stops_marginal else None))
    print(f"  first-stop prior: " + (f"mode lap {fs_summary['mode']}, median {fs_summary['median']:.0f} "
                                     f"({fs_summary['p25']:.0f}-{fs_summary['p75']:.0f}), n={fs_summary['n']}"
                                     + (f"; on a {fs_summary['by_stops']['n_stops']}-stop plan mode lap "
                                        f"{fs_summary['by_stops']['mode']}" if fs_summary.get("by_stops") else "")
                                     if fs_summary else "none for this circuit"))
    timings["priors_s"] = round(time.time() - t, 1)

    # -- 3. baseline -----------------------------------------------------------
    t = step("3. MixedLM baseline")
    mlm = fit_mixedlm(clean, n_boot=args.boot)
    print(mlm.table().round(4).to_string(index=False))
    timings["mixedlm_s"] = round(time.time() - t, 1)

    # -- 4. apex (diagnostic) --------------------------------------------------
    apex_use, apex_sel = None, None
    if args.apex:
        t = step("4. apex speeds (diagnostic channel)")
        apex = load_apex(ev)
        need = set(zip(clean["driver"], clean["lap_number"]))
        have = set(zip(apex["driver"], apex["lap_number"])) if len(apex) else set()
        if not need.issubset(have):
            try:
                apex = extract_apex_speeds(clean, ev)
                save_apex(apex, ev)
            except Exception as exc:
                print(f"  apex extraction failed: {exc}")
        if len(apex):
            apex_sel = select_corners(apex)
            apex_use = apex[apex["corner"].isin(apex_sel.corners)]
            print(f"  {len(apex_use)} apex rows on corners {apex_sel.corners} ({time.time()-t:.0f}s)")

    # -- 5. Bayes ------------------------------------------------------------
    t = step("5. hierarchical fit (lap-time channel, linear, soft per-circuit ladder)")
    ladder = cp.ladder if cp.available else None
    kw = dict(compound_prior="soft", circuit_ladder=ladder,
              pace_step_s=(pstep["step_s"] if not (ladder or {}).get("pace_step_usable") else None))
    if not args.full:
        kw.update(chains=WEEKEND_NUTS_CHAINS, warmup=WEEKEND_NUTS_WARMUP, draws=WEEKEND_NUTS_DRAWS)
    f = fit_bayes(clean, ev, prior="2026", apex=apex_use, **kw)
    timings["fit_s"] = round(time.time() - t, 1)
    print(f.slope_table().round(4).to_string(index=False))
    gate("convergence: r_hat < 1.01 and zero divergences",
         f.max_rhat < RHAT_GATE and f.n_divergences == 0,
         f"max r_hat {f.max_rhat:.4f}, {f.n_divergences} divergences ({timings['fit_s']}s)")
    f_practice = f
    f_practice.save(DATA_PROCESSED / f"posterior_{key}_practice.npz")
    moved = pd.DataFrame()
    if cp.available and cp.rate_prior:
        f, moved = apply_circuit_prior(f_practice, ev, regime, cp)
        print("  practice posterior combined with circuit history (rate only, capped 3x, floored; practice-regime s/lap):")
        print(moved.round(4).to_string(index=False))
    post_path = f.save(DATA_PROCESSED / f"posterior_{key}.npz")
    print(f"  posterior saved: {post_path.name}")

    # -- 6. seal -------------------------------------------------------------
    step("6. seal")
    sealed_path, sha = seal_predictions(f, ev, regime=regime, cliff=(cp.cliff() if cp.available else {}),
                                        note=f"weekend model on {used}; lap-time channel, linear, soft ladder; no race data read")
    sealed = load_sealed(sealed_path)
    print(f"  {sealed_path.name} sha256 {sha[:24]}...")

    # -- 7. pre-race plan ----------------------------------------------------
    t = step("7. pre-race strategy (calibrated constants, ladder gate enforced, position term)")
    per_comp_support = clean.groupby("compound")["tyre_age"].max().to_dict()
    caps = stint_caps_for(ev, cp) if cp.available else None
    if caps:
        print(f"  stint caps from this circuit's history: {caps}")
    total = f.posterior["lin"].shape[0]
    rng = np.random.default_rng(0)
    draws = rng.choice(total, size=min(args.mc_draws, total), replace=False)
    # V4/WP-B: the practice age support and the log-sd of the wear rate at twice
    # it, so a stint planned past the evidence is priced as an extrapolation
    # (`src.tyre.EXTRAP_LN_SD_MEASURED`, measured by bench/bench_extrapolation.py)
    model = TyreModel.from_fit(f, draws=draws, budget=cal.budgets, manage_floor=cal.manage_wear_floor,
                               manage_cost_s=cal.manage_cost_s,
                               support={k: float(v) for k, v in per_comp_support.items()},
                               extrap_ln_sd=EXTRAP_LN_SD_MEASURED)
    # V4: the race state times the first stop (constants from every 2026 race
    # but this weekend's); the circuit's first-stop history is then only a
    # plausibility prior through the plan family, so its lap term is off.  One
    # objective, assembled once (`src.objective`) and used for the search, the
    # pit window and the per-car plans.
    rs_const = objective.measure_constants_excluding({key})
    obj = V4Objective.for_event(ev, cal, plan_prior=plan_prior, first_stop_tables=fs_tables,
                                dirty_air=dirty_air, race_state=rs_const)
    kappa_used = float(obj.first_stop_kappa_s)
    print(f"  objective: {obj.label} (lambda {obj.undercut_lambda:.3f} on "
          f"{obj.as_dict()['undercut_applies_to']}, tau {obj.plan_prior_tau_s:.2f} s, "
          f"first-stop history kappa {kappa_used:.2f} s/nat, dirty air {obj.traffic_s_per_lap:.2f} s/lap, "
          f"grid {obj.grid_penalty_s:.2f} s)")
    sim_kw = dict(regime=regime, support=per_comp_support, max_per_compound=alloc["caps"], max_stint=caps,
                  **obj.sim_kwargs())
    net = pstep.get("net_stint_step_measured", float("nan"))
    model, res, pace_cal = strat.search_with_pace_calibration(
        model, ev, pit_loss, net_step_s=float(net if net is not None else np.nan),
        net_step_se_s=float(pstep.get("net_stint_step_se") or 0.0), **sim_kw)
    pw, uc, pdp = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    hcm, haas_terms = None, {}
    if res.table.empty:
        print("  no legal plan")
    else:
        print(f"  pace calibration: {pace_cal}")
        print(res.head(6).drop(columns=["pit_laps"]).round(2).to_string(index=False))
        print(res.by_stops.drop(columns=["pit_laps"]).round(2).to_string(index=False))
        print(res.life.round(2).to_string(index=False))
        print(f"  tyre-optimal plan: {res.tyre_optimal_label} ({res.tyre_optimal.get('delta_s', 0):+.1f} s under the full objective); "
              f"position term {res.best.get('position_s', 0):.1f} s, plan-prior handicap {res.best.get('prior_s', 0):.1f} s, "
              f"first-stop prior {res.best.get('first_stop_s', 0):.1f} s")
        pw = strat.pit_window_model(model, ev, res.best, pit_loss, max_stint=res.max_stint, push=res.best["push"],
                                    **obj.window_kwargs(res))
        if res.race_state.get("best"):
            _b = res.race_state["best"]
            print(f"  race state ({res.race_state.get('best_group')}): the tyre alone would stop on lap "
                  f"{_b.get('tyre_best_lap')}, against the pack lap {_b.get('best_lap')} "
                  f"(a place worth {rs_const.place_value_s:.2f} s)")
        if cp.available:
            ok = all(L <= cp.stint_cap.get(c, 10 ** 6) for c, L in zip(res.best["compounds"], res.best["stint_lens"]))
            gate("every recommended stint is within what this circuit has supported", ok,
                 "; ".join(f"{c} {L} laps (cap {cp.stint_cap.get(c, '—')})" for c, L in zip(res.best["compounds"], res.best["stint_lens"])))
            modal = max(cp.stops, key=cp.stops.get) if cp.stops else None
            gate("recommended stop count matches the circuit's usual", modal is None or res.best["n_stops"] == modal,
                 f"{res.best['n_stops']} stops; history {cp.stops}")
        _L = float(np.median(res.best["stint_lens"]))
        _model_net = float(model_net_step_draws(model, _L, float(res.best["push"]), ev).mean())
        if net is not None and np.isfinite(net):
            gate("model reproduces the measured net stint-level compound step (enforced)", abs(_model_net - net) < 0.12,
                 f"model {_model_net:+.3f} vs measured {net:+.3f} s/lap over a {_L:.0f}-lap stint")
        if "MEDIUM" in model.compounds and "SOFT" in model.compounds:
            uc_age = int(min(res.max_stint.get("MEDIUM", 40), res.max_stint.get("SOFT", 40)))
            uc = strat.undercut_window_model(model, "MEDIUM", "SOFT", max_age=max(uc_age, 8), push=res.best["push"])
        t0 = time.time()
        # Team-pooled practice deviation: a car with no long run on a compound
        # inherits its team-mate's behaviour, not the field's.  Only practice has
        # run, so the team map comes from the practice laps.
        teams = (clean.drop_duplicates("driver").set_index("driver")["team"].to_dict()
                 if "team" in clean else {})
        pooled_dev = percar.team_pooled_dev(model.driver_dev, teams,
                                            n_laps_by_driver=clean.groupby("driver").size().to_dict())
        try:
            _prac_raw = DATA_PROCESSED / f"laps_{key}_practice.parquet"
            hcm = haascar.HaasCarModel.from_weekend(
                ev, model, clean, calibration=cal,
                practice_laps=(pd.read_parquet(_prac_raw) if _prac_raw.exists() else None))
            haas_terms = {d: haascar.car_terms(st) for d, st in hcm.states.items()}
        except Exception as exc:
            print(f"  haas car model unavailable: {exc}")
        pdp = strat.per_driver_plans(model, ev, pit_loss, sorted(clean["driver"].unique()),
                                     race_factors=cal.driver_factors, dev_by_driver=pooled_dev,
                                     warmup_by_driver={d: t["warmup_s"] for d, t in haas_terms.items()} or None,
                                     traffic_mult_by_driver={d: t["traffic_mult"] for d, t in haas_terms.items()} or None,
                                     factor_ln_sd=cal.driver_factor_ln_sd, factor_shrink=None, **sim_kw)
        timings["per_driver_s"] = round(time.time() - t0, 1)
        if not pdp.empty:
            print(f"  per-car plans ({cal.percar_mode}, {len(pooled_dev)} cars pooled over "
                  f"{len(set(teams.values()))} teams): {int(pdp['same_shape_as_field'].sum())}/{len(pdp)} "
                  f"share the field plan's shape")
    timings["strategy_s"] = round(time.time() - t, 1)
    print(f"  ({time.time()-t:.0f}s)")

    # -- 8. artifacts --------------------------------------------------------
    step("8. write artifacts")
    ages = np.arange(0, 41, dtype=float)
    c = f.curve_table(ages); c["variant"] = "2026"
    rm = regime.draws(f.posterior["lin"].shape[0], seed=5)
    rc = []
    for comp in f.compounds:
        d = f.deg_loss(comp, ages) * rm[:, None]
        rc.append(pd.DataFrame({"compound": comp, "tyre_age": ages, "mean": d.mean(0),
                                "lo": np.quantile(d, 0.05, axis=0), "hi": np.quantile(d, 0.95, axis=0),
                                "variant": "2026_race"}))
    pd.concat([c] + rc, ignore_index=True).to_parquet(DATA_PROCESSED / f"curves_{key}.parquet", index=False)
    laps.to_parquet(DATA_PROCESSED / f"laps_{key}_practice.parquet", index=False)
    clean.to_parquet(DATA_PROCESSED / f"clean_{key}_practice.parquet", index=False)
    casc.to_parquet(DATA_PROCESSED / f"cascade_{key}.parquet", index=False)
    if not res.table.empty:
        res.table.head(400).assign(pit_laps=res.table.head(400)["pit_laps"].astype(str),
                                   stint_lens=res.table.head(400)["stint_lens"].astype(str)
                                   ).to_parquet(DATA_PROCESSED / f"strategy_{key}.parquet", index=False)
        res.by_stops.assign(pit_laps=res.by_stops["pit_laps"].astype(str),
                            stint_lens=res.by_stops["stint_lens"].astype(str)
                            ).to_parquet(DATA_PROCESSED / f"bystops_{key}.parquet", index=False)
        res.life.to_parquet(DATA_PROCESSED / f"life_{key}.parquet", index=False)
        rows, lap0 = [], 0
        for i, (comp, L) in enumerate(zip(res.best["compounds"], res.best["stint_lens"])):
            rows.append({"stint": i + 1, "compound": comp, "start_lap": lap0 + 1, "end_lap": lap0 + L, "laps": int(L)})
            lap0 += L
        pd.DataFrame(rows).to_parquet(DATA_PROCESSED / f"plan_{key}.parquet", index=False)
        if not pw.empty:
            pw.to_parquet(DATA_PROCESSED / f"pitwindow_{key}.parquet", index=False)
        if not uc.empty:
            uc.to_parquet(DATA_PROCESSED / f"undercut_{key}.parquet", index=False)
        if not pdp.empty:
            pdp.assign(pit_laps=pdp["pit_laps"].astype(str), practice_dev_s_per_lap=pdp["practice_dev_s_per_lap"].astype(str),
                       eff_rate=pdp["eff_rate"].astype(str)).to_parquet(DATA_PROCESSED / f"perdriver_{key}.parquet", index=False)
    ec = [pd.DataFrame({"session": s, "lap_start_s": g["lap_start_s"].to_numpy(), "evo_s": g["evo_s"].to_numpy()})
          .sort_values("lap_start_s") for s, g in clean.groupby("session")]
    pd.concat(ec, ignore_index=True).to_parquet(DATA_PROCESSED / f"evolution_{key}.parquet", index=False)

    fp26 = get_prior(ev, "2026")
    meta = {
        "event": key, "event_name": ev.name, "n_race_laps": ev.n_race_laps,
        "sessions_used": used, "sealed_file": sealed_path.name, "sealed_sha256": sha,
        "physics": {"burn_kg_per_lap": fp26.burn_kg_per_lap, "k_track_s_per_kg": fp26.k_track_s_per_kg,
                    "fuel_effect_s_per_lap": fp26.s_per_lap, "derivation": fp26.derivation},
        "compound_ladder": {"pace_step_s": float(pstep["step_s"]), "label": pstep["label"],
                            "derivation": pstep.get("derivation", ""), "circuit_ladder": (cp.ladder if cp.available else {}),
                            "table": compound_table(ev, f.compounds, pace_step_s=pstep["step_s"], circuit_ladder=ladder).to_dict("records"),
                            "fitted_offsets": f.comp_offset, "prior_label": f.compound_prior_label},
        "net_step": {"measured": pstep.get("net_stint_step_measured"), "se": pstep.get("net_stint_step_se"),
                     "derivation": pstep.get("derivation", ""), "detail": pstep["detail"]},
        "pace_calibration": pace_cal,
        "regime": regime.as_dict(),
        "circuit_history": (cp.as_dict() if cp.available else {}),
        "cliff_history": (cp.cliff() if cp.available else {}),
        "history_combination": moved.to_dict("records") if not moved.empty else [],
        "plan_prior": plan_prior,
        "first_stop_prior": fs_summary,
        "allocation": alloc,
        "calibration": {**{k: v for k, v in cal.as_dict().items() if k != "driver_factors"}, "source": cal.source,
                        "dirty_air_used": float(dirty_air)},
        "pit_loss_s": float(pit_loss), "pit_loss_source": pit_src, "pit_stops_measured": 0,
        "load_effect": {"exponent": float(TYRE_LOAD_EXPONENT)},
        "n_raw_laps": int(len(laps)), "n_clean_laps": int(len(clean)),
        "compound_counts": comp_sum.to_dict("records"),
        "mixedlm": {"slopes": mlm.slopes, "table": mlm.table().to_dict("records"), "n_stints": mlm.n_stints},
        "bayes": {"max_rhat": float(f.max_rhat), "n_divergences": int(f.n_divergences),
                  "n_laps": int(f.n_laps), "n_apex": int(f.n_apex), "use_hinge": bool(f.use_hinge),
                  "corners": [int(x) for x in (apex_sel.corners if apex_sel else [])],
                  "slopes": f.slope_table().to_dict("records"),
                  "k_track_mean": float(f.k_track.mean()), "k_track_sd": float(f.k_track.std()),
                  "k_track_rel_sd": float(f.k_track.std() / f.k_track.mean()),
                  "comp_offset": f.comp_offset, "drivers": list(f.drivers),
                  "chains": (4 if args.full else WEEKEND_NUTS_CHAINS), "draws": (1500 if args.full else WEEKEND_NUTS_DRAWS)},
        "evolution": {"range_s": float(evo_rng), "iterations": evo.iterations, "skipped": evo.skipped},
        "age_support_by_compound": {k: float(v) for k, v in per_comp_support.items()},
        "max_stint_laps": (res.max_stint if not res.table.empty else {}),
        "strategy": ({
            "n_strategies": int(res.n_strategies), "n_scored": int(res.n_scored), "n_draws": int(res.n_draws),
            "best": res.best_label,
            "best_plan": {k: (list(v) if isinstance(v, list) else v) for k, v in res.best.items()},
            "tyre_optimal": {**res.tyre_optimal, "label": res.tyre_optimal_label},
            "push": float(res.best.get("push", float("nan"))), "implied_regime": float(res.implied_regime),
            "grip_budget_s": float(model.budget), "grip_budgets": dict(model.budgets),
            "undercut_lambda": float(res.undercut_lambda), "plan_prior_tau_s": float(res.plan_prior_tau_s),
            "first_stop_kappa_s": float(res.first_stop_kappa_s),
            "position_s": float(res.best.get("position_s", 0.0)), "prior_s": float(res.best.get("prior_s", 0.0)),
            "first_stop_s": float(res.best.get("first_stop_s", 0.0)),
            "race_state_s": float(res.best.get("race_state_s", 0.0)),
            "race_state_terms": obj.terms_json(res),
            "by_stops": res.by_stops.assign(pit_laps=res.by_stops["pit_laps"].astype(str),
                                            stint_lens=res.by_stops["stint_lens"].astype(str)).to_dict("records"),
            "life": res.life.to_dict("records"),
            "pit_windows": ([{"stop": int(k), "recommended": int(res.best["pit_laps"][int(k) - 1]),
                              "lo": int(g[g["in_window"]]["lap"].min()), "hi": int(g[g["in_window"]]["lap"].max())}
                             for k, g in pw.groupby("stop")] if not pw.empty else []),
        } if not res.table.empty else {}),
        "race_state": (objective.race_state_block(res, rs_const, pw, ev.n_race_laps) if not res.table.empty
                       else {"enabled": rs_const is not None,
                             "constants": (rs_const.as_dict() if rs_const is not None else None)}),
        "objective": obj.as_dict(),
        "haas": (haascar.haas_block(hcm, pdp, None) if hcm is not None else {}),
        "per_driver": (pdp.assign(pit_laps=pdp["pit_laps"].astype(str)).to_dict("records") if not pdp.empty else []),
        "gates": GATES, "timings": timings, "runtime_s": round(time.time() - t_all, 1),
        "written_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (DATA_PROCESSED / f"weekend_{key}.json").write_text(json.dumps(meta, indent=2, default=_jd))
    n_fail = sum(1 for g in GATES if not g["pass"])
    print(f"\n=== {len(GATES) - n_fail}/{len(GATES)} gates passed in {time.time() - t_all:.0f}s "
          f"(fit {timings.get('fit_s')}s) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
