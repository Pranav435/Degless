"""End-to-end: practice laps -> physics -> fits -> sealed predictions -> race score.

Everything the app reads is written here.  The app fits nothing.

Two stages, because the plan-deciding constants are calibrated leave-one-out
across *every* scored weekend (`scripts/80_recalibrate.py`) and that needs
all the fits before any weekend's decisions are priced:

    --stage fit      practice laps, physics, the production fit (lap-time only,
                     no hinge, the circuit's ladder as a soft prior), the
                     diagnostic variants, the circuit-history fold-in, the seal.
                     Writes posterior_<key>.npz and fitstage_<key>.json.  Reads
                     no race lap of its own.
    --stage decide   race scoring, the strategy search with the calibrated
                     constants and the ladder gate enforced, windows, the
                     counterfactual, per-car plans, the backtest, artifacts,
                     meta_<key>.json.
    --stage all      both, in order (the default).

V4: the first stop is timed by the race state (`src.racestate`): the search
charges every plan group's pack-equilibrium term on the first stop instead of
the undercut exposure and the circuit's first-stop history prior, and the
first-stop window is that objective's window.  `--no-race-state` runs the V3
objective unchanged (the ablation).

    .venv/bin/python scripts/10_pipeline.py --event barcelona-2026
    .venv/bin/python scripts/10_pipeline.py --event barcelona-2026 --stage fit --joint
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

from src import cliff, firststop, percar, racestate, strategy as strat  # noqa: E402
from src.calibration import get_calibration  # noqa: E402
from src.compounds import (  # noqa: E402
    allocation_prior, hardness_rank, measure_pace_step, model_net_step_draws,
    pace_step_prior, summary_table as compound_table,
)
from src.config import (  # noqa: E402
    DATA_PROCESSED, MC_DRAWS, RHAT_GATE, TYRE_LOAD_EXPONENT, WEEKEND_NUTS_CHAINS,
    WEEKEND_NUTS_DRAWS, WEEKEND_NUTS_WARMUP, get_event,
)
from src.history import (  # noqa: E402
    CircuitPrior, apply_circuit_prior, circuit_prior, plan_prior_for, race_driver_factors, stint_caps_for,
)
from src.regime import RegimeFactor, measure_regime, regime_prior  # noqa: E402
from src.evolution import add_evolution_correction, fit_evolution_auto  # noqa: E402
from src.fuel import add_fuel_correction, get_prior, summary_table  # noqa: E402
from src.ingest import FirewallError, load_for_fitting, load_race  # noqa: E402
from src.laps import build_lap_table, cascade_counts, clean_laps, compound_summary  # noqa: E402
from src.model_bayes import BayesFit, fit_bayes  # noqa: E402
from src.model_fallback import fit_mixedlm, stint_fe_baseline  # noqa: E402
from src.replay import build_replay, save_replay  # noqa: E402
from src.telemetry import extract_apex_speeds, load_apex, save_apex, select_corners  # noqa: E402
from src.tyre import TyreModel  # noqa: E402
from src.validate import strategy_backtest, load_sealed, score_race, seal_predictions  # noqa: E402

log = logging.getLogger("degless.pipeline")

GATES: list = []


N_RIVALS_NOTE = f"{racestate.N_RIVALS} rivals (two ahead, two behind) per pack"


def race_state_block(res, rs_const, pw, n_laps: int) -> dict:
    """The race-state record for the meta: the recommended plan group's pack
    equilibrium and, for every lap from six before the first stop to the stop
    itself, the five actions it compared and the one it chose."""
    rs = res.race_state or {}
    best = rs.get("best") or {}
    if not best or rs_const is None:
        return {"enabled": rs_const is not None, "constants": (rs_const.as_dict() if rs_const else None)}
    laps = best["laps"]
    cost = np.asarray(best["cost_s"], dtype=float)
    tyre = np.asarray(best["tyre_s"], dtype=float)
    pos = np.asarray(best["position_s"], dtype=float)
    first = int(res.best["pit_laps"][0]) if res.best.get("pit_laps") else None
    w_hi = None
    if pw is not None and not pw.empty:
        g1 = pw[(pw["stop"] == 1) & pw["in_window"]]
        w_hi = int(g1["lap"].max()) if len(g1) else None
    decisions = []
    if first is not None:
        for lap in range(max(int(laps[0]), first - 6), first + 1):
            t = racestate.action_table(laps, cost, {"tyre_s": tyre - tyre.min(), "position_s": pos - pos.min(),
                                                    "places_ahead": np.asarray(best["places"])},
                                       now_lap=lap, window_hi=w_hi)
            decisions.append({"lap": lap, **t})
    return {"enabled": True, "constants": rs_const.as_dict(), "group": rs.get("best_group"),
            # who the simulated rivals were: the symmetric pack, or the
            # heterogeneous field's type table (`racestate.RivalFieldConfig`)
            "mode": rs.get("mode"), "rival_field": rs.get("rival_field"),
            "first_stop": first, "tyre_optimal_first_stop_in_group": best.get("tyre_best_lap"),
            "pack_first_stop_median": best.get("q_median"), "pack_first_stop_iqr": best.get("q_p25_p75"),
            "iterations": best.get("iterations"), "converged": best.get("converged"), "push": best.get("push"),
            "race_state_s": float(res.best.get("race_state_s", 0.0)),
            "curve": {k: best.get(k) for k in ("laps", "tyre_s", "places", "position_s", "cost_s", "term_s", "q")},
            "groups": rs.get("groups"), "n_groups_solved": rs.get("n_groups_solved"),
            "decisions": decisions}


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
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    return str(o)


def _regime_from(d: dict) -> RegimeFactor:
    return RegimeFactor(ratio=float(d["ratio"]), ln_sd=float(d["ln_sd"]), sources=list(d.get("sources", [])),
                        measured=bool(d.get("measured", False)), label=str(d.get("label", "")),
                        derivation=str(d.get("derivation", "")), donor_detail=list(d.get("donors", [])),
                        temperature=dict(d.get("temperature", {})))


# ==========================================================================
# Stage 1: fit
# ==========================================================================


def stage_fit(args, ev) -> dict:
    key = ev.key
    t_all = time.time()
    timings = {}

    # -- 0. firewall ------------------------------------------------------
    step("0. practice-only firewall")
    try:
        load_for_fitting(ev, "Race")
        gate("firewall blocks race data", False, "load_for_fitting(Race) did NOT raise")
    except FirewallError as exc:
        gate("firewall blocks race data", True, str(exc)[:90])

    # -- 1. practice laps -------------------------------------------------
    t = step("1. practice laps + clean-lap cascade")
    raw = load_for_fitting(ev)
    laps = build_lap_table(raw, ev)
    casc = cascade_counts(laps)
    clean = clean_laps(laps)
    comp_sum = compound_summary(laps)
    print(casc.to_string(index=False))
    print(comp_sum.to_string(index=False))
    timings["load_practice_s"] = round(time.time() - t, 1)
    gate("clean laps in 120-320", 120 <= len(clean) <= 320,
         f"{len(clean)} clean laps from {len(laps)} raw ({time.time()-t:.0f}s)")
    gate("no non-green lap survives",
         bool((clean["track_status"].astype(str) == "1").all()),
         f"track statuses: {sorted(clean['track_status'].unique())}")
    gate("all compounds valid",
         bool(clean["compound"].isin(["SOFT", "MEDIUM", "HARD"]).all()), "")
    laps.to_parquet(DATA_PROCESSED / f"laps_{key}_practice.parquet", index=False)
    casc.to_parquet(DATA_PROCESSED / f"cascade_{key}.parquet", index=False)

    # -- 2. physics + priors ------------------------------------------------
    t = step("2. fuel physics, circuit history, compound ladder, regime transfer")
    print(summary_table(ev).to_string(index=False))
    cp = circuit_prior(ev, probe_practice_temp=True)
    if cp.available:
        print(f"  circuit history: {ev.circuit} {cp.years}; stops {cp.stops}; plans {cp.plans}; starts {cp.starts}")
        print(f"    ladder from the circuit's races: {cp.ladder}")
        print(f"    cliff (longest / p90 stint): " + ", ".join(
            f"{c} {v['longest_stint']:.0f}/{v['p90_stint']:.0f}" for c, v in cp.cliff().items()))
    else:
        print(f"  circuit history: none for {ev.circuit}")
    pstep = pace_step_prior(ev, circuit=cp if cp.available else None)
    print(f"\n  compound pace step prior: {pstep['step_s']:.3f} s  [{pstep['label']}]")
    print(f"    measured net stint step: {pstep.get('net_stint_step_measured', float('nan')):+.3f} "
          f"+/- {pstep.get('net_stint_step_se', float('nan')):.3f} s")
    print(compound_table(ev, pace_step_s=pstep["step_s"], circuit_ladder=(cp.ladder if cp.available else None)).to_string(index=False))

    clean = add_fuel_correction(clean, ev, "2026")
    evo = fit_evolution_auto(clean, laps_all=laps, event=ev)
    clean = add_evolution_correction(clean, evo)
    print(pd.DataFrame(evo.iterations).round(4).to_string(index=False))
    evo_rng = evo.iterations[-1]["evo_range_s"]
    _it = evo.iterations[-1]
    gate("track evolution identified & plausible",
         0.0 < evo_rng < 5.0,
         f"push-lap range {evo_rng:.2f}s per session {_it.get('per_session', {})}; "
         f"long-run backfit would have given {_it.get('backfit_range_s', float('nan')):.2f}s; "
         f"sessions on the backfit: {evo.skipped}")
    clean.to_parquet(DATA_PROCESSED / f"clean_{key}_practice.parquet", index=False)

    # The practice -> race regime factor: measured on every weekend but this
    # one, temperature-corrected, pooled with the median.  See src/regime.py.
    # No race-day forecast exists in a retrospective run, so the auto mode pools
    # the donors' raw log ratios and no temperature enters; the circuit's own
    # practice->race history is combined with them.  `temperature["mode"]` says
    # which of the three paths ran.
    regime = regime_prior(ev, clean=clean)
    print(f"\n  practice->race degradation factor: {regime.ratio:.3f}x "
          f"[{regime.p05:.2f}-{regime.p95:.2f}]  [{regime.label}]")
    print(f"    {regime.derivation}")
    print(f"    temperature mode: {(regime.temperature or {}).get('mode')}; circuit prior: "
          f"{(regime.temperature or {}).get('circuit_prior') or 'none'}")
    gate("regime factor is plausible (0.15-1.0)",
         0.15 <= regime.ratio <= 1.0,
         f"{regime.ratio:.3f}x from {regime.sources or 'default'}")
    alloc = allocation_prior(ev)
    plan_prior = plan_prior_for(cp if cp.available else None)
    if plan_prior.get("nomination"):
        _nm = plan_prior["nomination"]
        print(f"  plan prior mapped through the nominations ({_nm['mode']}, target {_nm.get('target')}): "
              + "; ".join(f"{r['year']} {r['nomination']} {r['comparable']} "
                          f"(mapped {r['n_mapped']}, dropped {r['n_dropped']}, clamped {r['n_clamped']})"
                          for r in _nm.get("per_race", [])))
        print(f"    sequences {dict(list(plan_prior['sequences'].items())[:6])}; starts {plan_prior.get('starts')}")
    # The circuit's green-flag first stops, as the density the objective charges.
    # Historical races only (2023-25): this weekend's own race is never in it.
    # Summarised at the modal start compound *and* the modal stop count, because
    # that is the cell the objective will charge the modal plan against.
    _starts = plan_prior.get("starts") or {}
    _stops_marginal = plan_prior.get("stops") or {}
    modal_start = max(_starts, key=_starts.get) if _starts else None
    modal_stops = int(max(_stops_marginal, key=_stops_marginal.get)) if _stops_marginal else None
    first_stop_prior = firststop.first_stop_summary(
        cp.first_stop_green if cp.available else None, ev.n_race_laps,
        start_compound=modal_start, n_stops=modal_stops)
    if first_stop_prior:
        print(f"  first-stop prior: mode lap {first_stop_prior['mode']}, median {first_stop_prior['median']:.0f} "
              f"({first_stop_prior['p25']:.0f}-{first_stop_prior['p75']:.0f}), n={first_stop_prior['n']} "
              f"({first_stop_prior['n_compound']} on the modal start compound {modal_start})")
        bs = first_stop_prior.get("by_stops")
        if bs:
            print(f"    conditioned on a {bs['n_stops']}-stop plan: mode lap {bs['mode']}, median {bs['median']:.0f} "
                  f"({bs['p25']:.0f}-{bs['p75']:.0f}) from {bs['n_by_stops']} same-stop-count stops, "
                  f"{bs['n_by_compound_stops']} of them on the {modal_start}")
    else:
        print("  first-stop prior: none (no green-flag first stops in this circuit's history)")
    timings["priors_s"] = round(time.time() - t, 1)

    # -- 3. MixedLM + stint-FE baselines ----------------------------------
    t = step("3. MixedLM baseline + stint-FE baseline, both on stint block bootstraps")
    mlm = fit_mixedlm(clean, n_boot=args.boot)
    print(mlm.table().round(4).to_string(index=False))
    # The stint-FE baseline is the honest non-Bayesian reading of the same
    # corrected channel: within-stint OLS, no prior, no ladder, no pooling across
    # compounds.  It replaces the old "MixedLM MEDIUM slope in 0.12-0.35" gate,
    # which asserted a degradation level rather than testing the estimator — a
    # weekend whose tyres genuinely do not degrade (Australia) failed it for
    # being right.  What is worth gating is that the two estimators agree.
    fe = stint_fe_baseline(clean, n_boot=min(args.boot, 50))
    print(fe["table"].round(4).to_string(index=False))
    gate("stint-FE baseline pooled slope is finite and >= 0",
         bool(np.isfinite(fe["pooled_slope"]) and fe["pooled_slope"] >= 0),
         f"{fe['pooled_slope']:.4f} s/lap over {fe['n_stints']} stints / {fe['n_laps']} laps "
         f"[90% {fe['pooled_ci'][0]:.3f}-{fe['pooled_ci'][1]:.3f}] ({time.time()-t:.0f}s)")
    timings["mixedlm_s"] = round(time.time() - t, 1)

    # -- 4. apex speeds (diagnostic only) ---------------------------------
    apex_sel, apex_use = None, None
    if args.joint:
        t = step("4. corner apex speeds (diagnostic channel)")
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
            print(f"  {len(apex)} apex rows, corners selected: {apex_sel.corners}")
            apex_sel.table.to_parquet(DATA_PROCESSED / f"corners_{key}.parquet", index=False)
        timings["apex_s"] = round(time.time() - t, 1)

    # -- 5. Bayesian fits -------------------------------------------------
    t = step("5. NumPyro hierarchical fits")
    ladder = cp.ladder if cp.available else None
    fits = {}
    t0 = time.time()
    fits["2026"] = fit_bayes(clean, ev, prior="2026", compound_prior="soft", circuit_ladder=ladder,
                             pace_step_s=pstep["step_s"] if not (ladder or {}).get("pace_step_usable") else None)
    timings["fit_production_s"] = round(time.time() - t0, 1)
    f = fits["2026"]
    print(f"  production (lap-time only, linear, soft ladder): rhat={f.max_rhat:.4f} div={f.n_divergences} "
          f"laps={f.n_laps} ({timings['fit_production_s']}s)")
    gate("convergence: r_hat < 1.01 and zero divergences",
         f.max_rhat < RHAT_GATE and f.n_divergences == 0,
         f"max r_hat {f.max_rhat:.4f}, {f.n_divergences} divergences")
    f.save(DATA_PROCESSED / f"posterior_{key}_practice.npz")

    hist_moved = pd.DataFrame()
    if cp.available and cp.rate_prior:
        f, hist_moved = apply_circuit_prior(f, ev, regime, cp)
        fits["2026"] = f
        print(f"  circuit history {ev.circuit} {cp.years} folded in (rate only, capped 3x, floored):")
        print(hist_moved.round(4).to_string(index=False))
    f.save(DATA_PROCESSED / f"posterior_{key}.npz")

    # diagnostics: the sensitivity slides, at the quick sampler settings
    qk = {} if args.full_diagnostics else dict(chains=WEEKEND_NUTS_CHAINS, warmup=WEEKEND_NUTS_WARMUP, draws=WEEKEND_NUTS_DRAWS)
    diag_t = time.time()
    if not args.no_diagnostics:
        fits["hinge"] = fit_bayes(clean, ev, prior="2026", compound_prior="soft", circuit_ladder=ladder,
                                  use_hinge=True, **qk)
        fits["hardladder"] = fit_bayes(clean, ev, prior="2026", compound_prior="ladder",
                                       pace_step_s=pstep["step_s"], **qk)
        fits["2025"] = fit_bayes(clean, ev, prior="2025", compound_prior="soft", circuit_ladder=ladder, **qk)
        fits["none"] = fit_bayes(clean, ev, prior="none", compound_prior="soft", circuit_ladder=ladder, **qk)
        fits["noladder"] = fit_bayes(clean, ev, prior="2026", compound_prior="flat", **qk)
        if apex_use is not None and len(apex_use):
            fits["joint"] = fit_bayes(clean, ev, prior="2026", compound_prior="soft", circuit_ladder=ladder,
                                      apex=apex_use, **qk)
    timings["fit_diagnostics_s"] = round(time.time() - diag_t, 1)
    for k, v in fits.items():
        print(f"  {k:12s} rhat={v.max_rhat:.4f} div={v.n_divergences} laps={v.n_laps} apex={v.n_apex}")
    print(f.slope_table().round(4).to_string(index=False))
    print(f"  ({time.time()-t:.0f}s)")

    # Bayes vs MixedLM on the pooled slope, both on practice alone: the
    # baseline fits each compound freely and cannot cross-check the split
    # *between* compounds, but the overall level of degradation across the
    # weekend's running is something both estimators speak to.
    f_prac = BayesFit.load(DATA_PROCESSED / f"posterior_{key}_practice.npz")
    diffs, eff_slopes = {}, {}
    for c in f_prac.compounds:
        a = clean.loc[clean["compound"] == c, "tyre_age"].to_numpy(float)
        A = np.column_stack([np.ones_like(a), a])
        eff = np.linalg.lstsq(A, f_prac.deg_loss(c, a).mean(0), rcond=None)[0][1]
        eff_slopes[c] = float(eff)
        diffs[c] = abs(eff - mlm.slopes.get(c, np.nan))
    w = clean["compound"].value_counts()
    n_tot = float(w.sum())
    pooled_bayes = sum(eff_slopes[c] * w.get(c, 0) for c in f_prac.compounds) / n_tot
    pooled_mlm = sum(mlm.slopes.get(c, np.nan) * w.get(c, 0) for c in f_prac.compounds) / n_tot
    pooled_diff = abs(pooled_bayes - pooled_mlm)
    gate("Bayes vs MixedLM pooled degradation within 0.06 s/lap (practice-only posterior)",
         pooled_diff < 0.06,
         f"pooled {pooled_bayes:.3f} vs MixedLM {pooled_mlm:.3f} (diff {pooled_diff:.3f}); "
         "per-compound differences are the ladder prior overriding an unordered baseline: "
         + ", ".join(f"{c} {v:.3f}" for c, v in diffs.items()))
    fe_diff = abs(pooled_bayes - fe["pooled_slope"])
    gate("Bayes pooled slope within 0.06 s/lap of the stint-FE baseline",
         bool(fe_diff < 0.06),
         f"pooled {pooled_bayes:.3f} vs stint-FE {fe['pooled_slope']:.3f} (diff {fe_diff:.3f}) over "
         f"{fe['n_stints']} stints")

    # The ladder: with the soft prior the ordering is what the data and the
    # circuit say, and it is reported rather than asserted.
    rank = hardness_rank(f.compounds)
    by_rank = [c for _, c in sorted(zip(rank, f.compounds))]
    slopes = {r["compound"]: r["slope_s_per_lap"] for _, r in f.slope_table().iterrows()}
    offs = f.comp_offset
    deg_ordered = all(slopes[a] > slopes[b] for a, b in zip(by_rank, by_rank[1:]))
    pace_ordered = all(offs[a] < offs[b] for a, b in zip(by_rank, by_rank[1:]))
    flat_slopes = ({r["compound"]: r["slope_s_per_lap"] for _, r in fits["noladder"].slope_table().iterrows()}
                   if "noladder" in fits else {})
    print(f"\n  compound ordering (soft ladder): deg " + " > ".join(f"{c} {slopes[c]:.3f}" for c in by_rank)
          + f" ({'ordered' if deg_ordered else 'INVERTED by the data'}); pace "
          + " < ".join(f"{c} {offs[c]:.2f}" for c in by_rank) + f" ({'ordered' if pace_ordered else 'inverted'})"
          + (f"; without any ladder: " + " ".join(f"{c} {flat_slopes[c]:.3f}" for c in by_rank) if flat_slopes else ""))

    prior_shift = {}
    for c in f.compounds:
        a = np.arange(1, 21.0)
        prior_shift[c] = {name: float(np.polyfit(a, fits[name].deg_loss(c, a).mean(0), 1)[0])
                          for name in ("2026", "2025", "none") if name in fits}
    print("\n  prior sensitivity (mean slope over ages 1-20, s/lap):")
    print(pd.DataFrame(prior_shift).T.round(4).to_string())

    # -- 6. seal ----------------------------------------------------------
    t = step("6. seal predictions (before any race lap is read)")
    sealed_path, sha = seal_predictions(
        f, ev, regime=regime, cliff=(cp.cliff() if cp.available else {}),
        note=("fitted on practice only; lap-time channel, linear in age, soft per-circuit ladder; "
              f"circuit history folded into the rate; race-regime curves scaled by {regime.ratio:.3f}x "
              f"({regime.label})"))
    print(f"  {sealed_path.name}  sha256 {sha[:32]}...")
    gate("sealed file verifies against its sha256", True, sha[:16])

    # -- artifacts of the fit stage ------------------------------------------
    ages = np.arange(0, 41, dtype=float)
    curves = []
    for name, fitobj in fits.items():
        c = fitobj.curve_table(ages)
        c["variant"] = name
        curves.append(c)
    _rmult = regime.draws(f.posterior["lin"].shape[0], seed=5)
    for c in f.compounds:
        d = f.deg_loss(c, ages) * _rmult[:, None]
        curves.append(pd.DataFrame({"compound": c, "tyre_age": ages, "mean": d.mean(0),
                                    "lo": np.quantile(d, 0.05, axis=0), "hi": np.quantile(d, 0.95, axis=0),
                                    "variant": "2026_race"}))
    pd.concat(curves, ignore_index=True).to_parquet(DATA_PROCESSED / f"curves_{key}.parquet", index=False)
    knee = [pd.DataFrame({"variant": name, "compound": c, "knee": fitobj.posterior["knee"][:, i]})
            for name, fitobj in fits.items() if fitobj.has_hinge for i, c in enumerate(fitobj.compounds)]
    if knee:
        pd.concat(knee, ignore_index=True).to_parquet(DATA_PROCESSED / f"knee_{key}.parquet", index=False)
    ec = [pd.DataFrame({"session": s, "lap_start_s": g["lap_start_s"].to_numpy(), "evo_s": g["evo_s"].to_numpy()})
          .sort_values("lap_start_s") for s, g in clean.groupby("session")]
    pd.concat(ec, ignore_index=True).to_parquet(DATA_PROCESSED / f"evolution_{key}.parquet", index=False)

    per_comp_support = clean.groupby("compound")["tyre_age"].max().to_dict()
    fp26 = get_prior(ev, "2026")
    fs = {
        "event": key, "event_name": ev.name, "n_race_laps": ev.n_race_laps,
        "sessions_used": list(ev.practice_sessions),
        "sealed_file": sealed_path.name, "sealed_sha256": sha,
        "physics": {"burn_kg_per_lap": fp26.burn_kg_per_lap, "k_track_s_per_kg": fp26.k_track_s_per_kg,
                    "fuel_effect_s_per_lap": fp26.s_per_lap, "derivation": fp26.derivation},
        "prior_table": summary_table(ev).to_dict("records"),
        "compound_ladder": {
            "pace_step_s": float(pstep["step_s"]), "measured": bool(pstep["measured"]), "label": pstep["label"],
            "derivation": pstep.get("derivation", ""), "donors": pstep["detail"],
            "circuit_ladder": (cp.ladder if cp.available else {}),
            "table": compound_table(ev, f.compounds, pace_step_s=pstep["step_s"], circuit_ladder=ladder).to_dict("records"),
            "deg_ratio": float(np.exp(np.log1p(fits["2026"].posterior["deg_gap"]).mean())),
            "fitted_offsets": f.comp_offset,
            "unladdered_slopes": flat_slopes, "deg_ordered": bool(deg_ordered), "pace_ordered": bool(pace_ordered),
            "prior_label": f.compound_prior_label,
        },
        "net_step": {"measured": pstep.get("net_stint_step_measured"), "se": pstep.get("net_stint_step_se"),
                     "derivation": pstep.get("derivation", ""), "detail": pstep["detail"]},
        "circuit_history": (cp.as_dict() if cp.available else {}),
        "cliff_history": (cp.cliff() if cp.available else {}),
        "history_combination": hist_moved.to_dict("records") if not hist_moved.empty else [],
        "plan_prior": plan_prior,
        "first_stop_prior": first_stop_prior,
        "regime": regime.as_dict(),
        "allocation": alloc,
        "load_effect": {"exponent": float(TYRE_LOAD_EXPONENT),
                        "start_multiplier": float((838 / 803) ** TYRE_LOAD_EXPONENT),
                        "flag_multiplier": float((768 / 803) ** TYRE_LOAD_EXPONENT)},
        "n_raw_laps": int(len(laps)), "n_clean_laps": int(len(clean)),
        "compound_counts": comp_sum.to_dict("records"),
        "mixedlm": {"slopes": mlm.slopes, "table": mlm.table().to_dict("records"), "n_stints": mlm.n_stints},
        "stint_fe_baseline": {**{k: v for k, v in fe.items() if k != "table"},
                              "table": fe["table"].to_dict("records")},
        "bayes": {
            "max_rhat": float(f.max_rhat), "n_divergences": int(f.n_divergences),
            "n_laps": int(f.n_laps), "n_apex": int(f.n_apex), "use_hinge": bool(f.use_hinge),
            "corners": [int(c) for c in (apex_sel.corners if apex_sel else [])],
            "slopes": f.slope_table().to_dict("records"),
            "k_track_mean": float(f.k_track.mean()), "k_track_sd": float(f.k_track.std()),
            "k_track_rel_sd": float(f.k_track.std() / f.k_track.mean()),
            "comp_offset": f.comp_offset, "drivers": list(f.drivers),
            "variants": {k: {"rhat": float(v.max_rhat), "div": int(v.n_divergences), "n_apex": int(v.n_apex),
                             "slopes": v.slope_table().to_dict("records"), "hinge": bool(v.has_hinge),
                             "k_track_rel_sd": float(v.k_track.std() / max(v.k_track.mean(), 1e-9))}
                         for k, v in fits.items()},
        },
        "bayes_vs_mixedlm": {k: float(v) for k, v in diffs.items()},
        "bayes_vs_mixedlm_pooled": {"bayes": float(pooled_bayes), "mixedlm": float(pooled_mlm), "diff": float(pooled_diff)},
        "bayes_vs_stint_fe_pooled": {"bayes": float(pooled_bayes), "stint_fe": float(fe["pooled_slope"]),
                                     "diff": float(fe_diff)},
        "prior_sensitivity": prior_shift,
        "evolution": {"range_s": float(evo_rng), "iterations": evo.iterations, "skipped": evo.skipped},
        "age_support_laps": float(clean["tyre_age"].max()),
        "age_support_by_compound": {k: float(v) for k, v in per_comp_support.items()},
        "gates_fit": list(GATES),
        "timings": timings,
        "runtime_fit_s": round(time.time() - t_all, 1),
    }
    (DATA_PROCESSED / f"fitstage_{key}.json").write_text(json.dumps(fs, indent=2, default=_jd))
    print(f"\n=== fit stage done in {time.time() - t_all:.0f}s ===")
    return fs


# ==========================================================================
# Stage 2: decide
# ==========================================================================


def stage_decide(args, ev, fs: dict | None = None) -> int:
    key = ev.key
    t_all = time.time()
    timings = {}
    if fs is None:
        p = DATA_PROCESSED / f"fitstage_{key}.json"
        if not p.exists():
            print(f"no fit stage for {key}: run --stage fit first")
            return 1
        fs = json.loads(p.read_text())
    f = BayesFit.load(DATA_PROCESSED / f"posterior_{key}.npz")
    clean = pd.read_parquet(DATA_PROCESSED / f"clean_{key}_practice.parquet")
    regime = _regime_from(fs["regime"])
    cp = CircuitPrior(**{k: v for k, v in (fs.get("circuit_history") or {}).items()
                         if k in CircuitPrior.__dataclass_fields__}) if fs.get("circuit_history") else CircuitPrior(event=key, circuit=ev.circuit)
    cal = get_calibration(ev)
    # Dirty air is a property of the circuit, so the weekend uses its own
    # circuit's measurement (from its 2023-25 races) where the calibration has
    # one and the pooled 2026 median otherwise.
    dirty_air = cal.dirty_air_for(ev.circuit)
    print(f"\n  calibration: {cal.source}")
    print(f"    budgets {cal.budgets}; manage cost {cal.manage_cost_s:.2f} s, floor {cal.manage_wear_floor:.2f}; "
          f"grid penalty {cal.grid_start_penalty_s:.2f} s; dirty air {dirty_air:.2f} s/lap "
          f"({'this circuit' if ev.circuit in cal.dirty_air_by_circuit else 'pooled'}); "
          f"undercut lambda {cal.undercut_lambda:.3f}; plan prior tau {cal.plan_prior_tau_s:.2f} s; "
          f"first-stop kappa {cal.first_stop_kappa_s:.2f} s/nat; per-car {cal.percar_mode}")
    sealed = load_sealed(Path(__file__).resolve().parents[1] / "predictions" / "sealed" / fs["sealed_file"])
    sealed["_file"] = fs["sealed_file"]
    pstep = fs["net_step"]
    alloc = fs["allocation"]
    plan_prior = fs.get("plan_prior") or {}
    per_comp_support = {k: float(v) for k, v in fs["age_support_by_compound"].items()}

    # -- 7. race scoring --------------------------------------------------
    t = step("7. race data (validation only)")
    race_raw = load_race(ev)
    race = build_lap_table(race_raw, ev)
    race_clean = clean_laps(race)
    race.to_parquet(DATA_PROCESSED / f"laps_{key}_race.parquet", index=False)
    print(f"  race: {len(race)} laps, {len(race_clean)} clean")
    _self_regime = measure_regime(ev, race=race, practice=clean)
    _self_step = measure_pace_step(ev, race=race)
    print(f"  self-measured regime factor {_self_regime.ratio:.3f}x "
          f"(transferred: {regime.ratio:.3f}x, never fitted here)")
    print(f"  self-measured net stint step {_self_step.step_s:+.3f} +/- "
          f"{_self_step.se:.3f} s (transferred: {pstep.get('measured', float('nan')):+.3f} s)")

    sc = score_race(sealed, race_clean, ev)
    print(f"  {sc.summary()}")
    print("  MAE by compound:", {k: round(v, 3) for k, v in sc.mae_by_compound.items()})
    print("  calibration:", {f"{k:.0%}": f"{v:.1%}" for k, v in sc.coverage.items()})
    if sc.cliff:
        print("  cliff:", {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                           for k, v in sc.cliff.items()})
    print("  bias by compound (observed - predicted, s/lap):",
          {k: round(v, 3) for k, v in sc.bias_by_compound.items()})
    gate("practice->race regime transfer removes the bias",
         abs(sc.bias) < 0.06,
         f"bias {sc.bias:+.3f} s/lap against {sc.regime_label} curves "
         f"(transferred factor {regime.ratio:.2f}x; this weekend's own race says "
         f"{_self_regime.ratio:.2f}x, which was never used)")
    gate("stint degradation-rate MAE < 0.15 s/lap", sc.passes_mae,
         f"{sc.mae:.4f} s/lap over {sc.n_rate_stints} stints "
         f"(per-lap MAE {sc.mae_lap:.3f} s, race noise {sc.sigma_used:.2f} s)")
    gate("90% coverage not below 0.80 (under-coverage is the failure)",
         sc.passes_coverage,
         f"{sc.coverage.get(0.90, float('nan')):.1%} per lap, {sc.rate_coverage90:.1%} on the stint rate "
         f"(width {sc.rate_width90:.3f} s/lap) — {sc.coverage_direction}")
    timings["score_s"] = round(time.time() - t, 1)

    # -- 8. strategy ------------------------------------------------------
    t = step("8. pit loss, strategy search (ladder gate enforced), windows, counterfactual, per-car plans")
    print("  practice age support (per compound, laps):", {c: int(v) for c, v in per_comp_support.items()})
    print(f"  tyre allocation: {alloc['caps']}")
    pl = strat.measure_pit_loss(race)
    print(f"  pit loss: {pl.seconds:.2f}s from {pl.n_stops} green-flag stops")
    gate("pit loss measured and plausible (12-35s)",
         np.isfinite(pl.seconds) and 12 <= pl.seconds <= 35, f"{pl.seconds:.2f}s")
    hist_caps = stint_caps_for(ev, cp) if cp.available else None
    if hist_caps:
        print(f"  stint caps from this circuit's history: {hist_caps}")
    if plan_prior:
        print(f"  plan-shape prior ({plan_prior.get('source')}): " + ", ".join(
            f"{k} {v}" for k, v in list(plan_prior["sequences"].items())[:6]) + f"; starts {plan_prior.get('starts')}")

    total = f.posterior["lin"].shape[0]
    rng = np.random.default_rng(0)
    draws = rng.choice(total, size=min(args.mc_draws, total), replace=False)
    model = TyreModel.from_fit(f, draws=draws, budget=cal.budgets,
                               manage_floor=cal.manage_wear_floor, manage_cost_s=cal.manage_cost_s)
    # The circuit's first-stop density, as a per-lap penalty table per start
    # compound *and stop count*.  Built from the historical races only, so it
    # carries no information about this weekend's race.  `None` where the circuit
    # has no green-flag first stops on record, which leaves the V2 objective.
    fs_tables = firststop.first_stop_penalty_table(
        cp.first_stop_green if cp.available else None, ev.n_race_laps, model.compounds)
    fs_summary = fs.get("first_stop_prior")
    print(f"  first-stop prior: " + (f"{fs_summary['source']}" if fs_summary else "none")
          + f"; charged at kappa {cal.first_stop_kappa_s:.2f} s/nat"
          + ("" if fs_tables else " (no table: the term is off)"))
    if fs_tables:
        _cells = sorted({k for per in fs_tables.values() for k in per if k != "any"})
        print(f"    conditioned tables per start compound: stop counts {_cells} (plus \"any\")")
    # V4: the race state times the first stop.  Its constants are measured on
    # every other 2026 race (never this one); the first-stop history prior is
    # switched off because the race-state term now decides the lap - history
    # stays in the plan-family prior and the stint caps.
    use_rs = not getattr(args, "no_race_state", False)
    rs_const = racestate.measure_constants(exclude=key) if use_rs else None
    kappa_used = 0.0 if use_rs else cal.first_stop_kappa_s
    if rs_const is not None:
        _c = rs_const.as_dict()
        print(f"  race state ({_c['source']}): a place is worth {_c['place_value_s']:.2f} s "
              f"(adjacent finishers {_c['place_gap_s']:.2f} s apart x (2 x {_c['persistence']:.2f} kept - 1)); "
              f"pit-cycle noise {_c['sigma_rel_s']:.2f} s between two cars; pack gaps median "
              f"{_c['pack_gap_median_s']:.2f} s; {N_RIVALS_NOTE}")
    else:
        print("  race state: off (--no-race-state) - V3's undercut exposure and first-stop prior time the stop")
    sim_kw = dict(regime=regime, step=args.pit_step, support=per_comp_support, max_per_compound=alloc["caps"],
                  max_stint=hist_caps, undercut_lambda=cal.undercut_lambda, plan_prior=plan_prior,
                  plan_prior_tau_s=cal.plan_prior_tau_s, first_stop_prior=fs_tables,
                  first_stop_kappa_s=kappa_used, traffic_s_per_lap=dirty_air,
                  grid_penalty_s=cal.grid_start_penalty_s, race_state=rs_const)
    t0 = time.time()
    model, res, pace_cal = strat.search_with_pace_calibration(
        model, ev, float(pl.seconds), net_step_s=float(pstep.get("measured") if pstep.get("measured") is not None else np.nan),
        net_step_se_s=float(pstep.get("se") or 0.0), **sim_kw)
    timings["search_s"] = round(time.time() - t0, 1)
    max_stint = res.max_stint
    print(f"  pace calibration: {pace_cal}")
    print("  stint caps (min of the wear bound and the circuit's history):", max_stint)
    print(f"  {res.n_strategies:,} plans searched, {res.n_scored:,} scored x {res.n_draws} draws in {timings['search_s']}s")
    print(res.head(8).drop(columns=["pit_laps"]).round(3).to_string(index=False))
    print("\n  best plan at each stop count:")
    print(res.by_stops.drop(columns=["pit_laps"]).round(3).to_string(index=False))
    print(f"\n  tyre-optimal plan (no position term, no plan prior, no first-stop prior): {res.tyre_optimal_label} "
          f"({res.tyre_optimal.get('delta_s', 0):+.1f} s under the full objective; its own first stop would "
          f"carry {res.tyre_optimal.get('first_stop_s', 0):.1f} s of first-stop prior); "
          f"position term of the chosen plan {res.best.get('position_s', 0):.1f} s, plan-prior handicap "
          f"{res.best.get('prior_s', 0):.1f} s, first-stop prior {res.best.get('first_stop_s', 0):.1f} s")
    print("\n  compound life (budget / rate, bounded by the race and the circuit's history):")
    print(res.life.round(2).to_string(index=False))
    print(f"\n  the optimiser chose push {res.best['push']:.2f}, which implies a "
          f"practice->race degradation factor of {res.implied_regime:.2f} (transferred {regime.ratio:.2f})")
    lens = res.best.get("stint_lens", [])
    gate("recommended plan has realistic stint lengths",
         bool(lens) and min(lens) >= 8 and 1 <= res.best["n_stops"] <= 3,
         f"{res.best_label} -> stints {lens}")
    worst = float(res.table.iloc[0]["max_wear"])
    gate("no stint in the recommended plan runs past the cliff",
         worst <= 1.0, f"deepest stint reaches {worst:.0%} of the grip budget")
    gate("model does not assume more tyre management than was observed",
         res.implied_regime > regime.ratio - 0.25,
         f"implied {res.implied_regime:.2f} vs transferred {regime.ratio:.2f}")

    _ord = (res.table[res.table["n_stops"] == res.best["n_stops"]]
            .sort_values("mean_s").groupby("compounds", as_index=False).first()
            .sort_values("mean_s"))
    _spread = float(_ord["mean_s"].iloc[-1] - _ord["mean_s"].iloc[0]) if len(_ord) > 1 else 0.0
    print(f"\n  the best {res.best['n_stops']}-stop plan for each of {len(_ord)} orderings spans {_spread:.1f} s: "
          + " | ".join(f"{r.compounds} {r.mean_s - _ord['mean_s'].iloc[0]:+.1f}" for r in _ord.head(4).itertuples()))

    rs_term = racestate.term_by_lap((res.race_state or {}).get("best"), ev.n_race_laps) if use_rs else None
    pw = strat.pit_window_model(model, ev, res.best, float(pl.seconds), max_stint=max_stint, push=res.best["push"],
                                undercut_lambda=cal.undercut_lambda, traffic_s_per_lap=dirty_air,
                                first_stop_prior=fs_tables, first_stop_kappa_s=kappa_used,
                                race_state_term=rs_term)
    if not pw.empty:
        for k, g in pw.groupby("stop"):
            win = g[g["in_window"]]["lap"]
            print(f"  stop {k}: recommended lap {res.best['pit_laps'][k-1]}, "
                  f"within 1.0 s over laps {win.min():.0f}-{win.max():.0f}"
                  + (f" (the circuit's history prefers lap {fs_summary['mode']})"
                     if (k == 1 and fs_summary) else ""))
    rs_block = race_state_block(res, rs_const, pw, ev.n_race_laps)
    if rs_block.get("decisions"):
        print(f"  race state, group {rs_block['group']}: the tyre alone would stop on lap "
              f"{rs_block['tyre_optimal_first_stop_in_group']}; against the pack the first stop is lap "
              f"{rs_block['first_stop']} (pack median {rs_block['pack_first_stop_median']}, IQR "
              f"{rs_block['pack_first_stop_iqr']}, {rs_block['iterations']} iterations)")
        for d in rs_block["decisions"]:
            acts = " | ".join(f"{a['action']} (L{a['lap']}) {a['delta_s']:+.2f}" for a in d["actions"] if a.get("legal"))
            print(f"    lap {d['lap']:2d}: {d['decision']:22s} {acts}")
    _uc_age = int(min(max_stint.get("MEDIUM", 40), max_stint.get("SOFT", 40)))
    uc = strat.undercut_window_model(model, "MEDIUM", "SOFT", max_age=max(_uc_age, 8), push=res.best["push"]) \
        if ("MEDIUM" in model.compounds and "SOFT" in model.compounds) else pd.DataFrame()
    t0 = time.time()
    cf = strat.counterfactual(model, ev, race, float(pl.seconds), max_stint=max_stint, push=res.best["push"],
                              race_factors=cal.driver_factors, traffic_s_per_lap=dirty_air,
                              undercut_lambda=cal.undercut_lambda,
                              first_stop_prior=fs_tables, first_stop_kappa_s=cal.first_stop_kappa_s)
    timings["counterfactual_s"] = round(time.time() - t0, 1)
    if not cf.empty:
        print(f"\n  counterfactual (top 5, SC stops held fixed, {timings['counterfactual_s']}s):")
        print(cf.head(5)[["driver", "compounds", "actual_pit_laps", "model_pit_laps", "loss_s", "n_sc_stops", "classified"]]
              .round(2).to_string(index=False))
    t0 = time.time()
    drivers_on_grid = sorted(race["driver"].unique())
    # Per-car intelligence, team-pooled: a driver with no long run on a compound
    # inherits their team-mate's deviation rather than the field's, and the
    # historical rate factor is shrunk by how precisely it was measured
    # (`factor_shrink=None` -> `percar.shrink_factor`).
    teams = pd.concat([clean, race]).drop_duplicates("driver").set_index("driver")["team"].to_dict()
    pooled_dev = percar.team_pooled_dev(model.driver_dev, teams,
                                        n_laps_by_driver=clean.groupby("driver").size().to_dict())
    pdp = strat.per_driver_plans(model, ev, float(pl.seconds), drivers_on_grid, race_factors=cal.driver_factors,
                                 dev_by_driver=pooled_dev, factor_ln_sd=cal.driver_factor_ln_sd,
                                 factor_shrink=None,
                                 **{k: v for k, v in sim_kw.items() if k != "step"})
    timings["per_driver_s"] = round(time.time() - t0, 1)
    if not pdp.empty:
        print(f"\n  per-car plans ({timings['per_driver_s']}s, {cal.percar_mode}: "
              f"{len(pooled_dev)} cars carry a pooled practice deviation over {len(set(teams.values()))} teams): "
              f"{int(pdp['same_shape_as_field'].sum())}/{len(pdp)} "
              f"share the field plan's shape; first stops "
              f"{int(pdp['first_stop'].min()) if pdp['first_stop'].notna().any() else '-'}-"
              f"{int(pdp['first_stop'].max()) if pdp['first_stop'].notna().any() else '-'}")
        print(pdp[["driver", "race_factor", "dev_source", "best", "first_stop_vs_field"]].head(8)
              .round(2).to_string(index=False))
    print(f"  ({time.time()-t:.0f}s)")

    # -- the ladder gate, enforced ------------------------------------------
    _ranks = dict(zip(f.compounds, hardness_rank(list(f.compounds))))
    _open = res.best["compounds"][0]
    gate("does not open the race on the hardest available compound",
         _ranks[_open] < max(_ranks.values()),
         f"opens on {_open} (hardness rank {_ranks[_open]} of {max(_ranks.values())})")
    # The net step is checked at the length the shipped model is *calibrated* at,
    # which `search_with_pace_calibration` iterates to a fixed point and clips to
    # the tyres' lives.  Checking it at the plan's median instead asks the model
    # to reproduce a measurement at a length where the loss is no longer linear,
    # which is how a converged calibration came out failing its own gate.
    _L_plan = float(np.median(res.best["stint_lens"]))
    _L = float(pace_cal.get("stint_len", _L_plan))
    _model_net = float(model_net_step_draws(model, _L, float(res.best["push"]), ev).mean())
    _meas_net = float(pstep.get("measured") if pstep.get("measured") is not None else np.nan)
    print(f"\n  compound ladder check - net cost of one step harder over a {_L:.0f}-lap stint "
          f"(the plan's median stint is {_L_plan:.0f}, life cap {pace_cal.get('life_cap', float('nan')):.0f}, "
          f"{pace_cal.get('passes', 0)} pass(es), converged {pace_cal.get('converged')}): "
          f"model {_model_net:+.3f} s/lap, measured {_meas_net:+.3f} s/lap (this race: {_self_step.step_s:+.3f})")
    if np.isfinite(_meas_net):
        gate("model reproduces the measured net stint-level compound step (enforced)",
             abs(_model_net - _meas_net) < 0.12,
             f"model {_model_net:+.3f} vs measured {_meas_net:+.3f} s/lap per step over a {_L:.0f}-lap stint "
             f"(the plan runs {_L_plan:.0f})"
             + (" (offsets clipped to the physical band)" if pace_cal.get("clipped") else ""))

    # -- 8b. the cliff, detected rather than assumed ------------------------
    # One row per race stint: did the tyre end it, or did the pit wall?  The
    # grip budget the model ships is only meaningful if budget / rate lands near
    # where the cliff actually was, and this is the only end-to-end check that
    # invariant has.  The rate fed to it is the model's **full-push** rate, the
    # regime the budget is defined in; a race-measured slope answers a different
    # question and puts the predicted lap at twice the race distance.
    collapse_rows = cliff.race_collapses(race, event=ev)
    ratio_metrics = cliff.budget_ratio_metrics(
        {c: {"budget_s": model.budget_of(c)} for c in model.compounds},
        {c: float(model.rate(c).mean()) for c in model.compounds},
        collapse_rows)
    _counts = ({str(k): int(v) for k, v in collapse_rows["kind"].value_counts().items()}
               if not collapse_rows.empty else {})
    print(f"\n  cliff detector on the race: {len(collapse_rows)} stints scored, {_counts}")
    for c in model.compounds:
        m = ratio_metrics.get(c) or {}
        if m.get("n"):
            print(f"    {c:7s} budget {m['budget_s']:.2f} s / rate {m['rate_s_per_lap']:.3f} s/lap "
                  f"-> cliff at lap {m['predicted_lap']:.0f}; observed knees {m['observed_knees']} "
                  f"(mean error {m['mean_error_laps']:+.1f}, {m['over']} over / {m['under']} under)")
    if (ratio_metrics.get("_pooled") or {}).get("n"):
        p = ratio_metrics["_pooled"]
        print(f"    pooled: {p['n']} collapses, mean error {p['mean_error_laps']:+.1f} laps "
              f"(|error| {p['mean_abs_error_laps']:.1f})")

    # -- 8c. strategy backtest --------------------------------------------
    bt = strategy_backtest(res, race, ev)
    if bt:
        print("\n  strategy backtest (race data, validation only):")
        print(f"    recommended {bt['recommended_stops']} stops; the field ran {bt['observed_stop_counts']}")
        print(pd.DataFrame(bt["per_stint"]).to_string(index=False))
        print("    grip budget implied by this race, per compound: "
              + ", ".join(f"{c} {v:.2f}s" for c, v in sorted(bt["grip_budget_implied"].items()))
              + f"  (model uses {bt['grip_budget_used']})")
        fsb = bt["first_stop"]
        print(f"    first stops: field median (green) {fsb['field_median_green']}, {fsb['share_under_sc']:.0%} under SC; "
              f"recommended {fsb['recommended']} ({fsb['recommended_minus_field']}), tyre-optimal {fsb['tyre_optimal']} "
              f"({fsb['tyre_optimal_minus_field']})")
        gate("recommended stop count is one the field actually ran",
             bt["stops_observed_share"] > 0,
             f"{bt['recommended_stops']} stops — {bt['stops_observed_share']:.0%} of finishers, mode was {bt['modal_stops']}")
        gate("every recommended stint length is one the compound was run to",
             bt["all_stints_inside_observed_range"],
             "; ".join(f"{r['compound']} {r['recommended_laps']} laps (observed median {r['observed_median']:.0f}, "
                       f"max {r['observed_max']:.0f})" for r in bt["per_stint"]))

    # -- 9. replay --------------------------------------------------------
    t = step("9. replay precompute")
    rp = build_replay(f, ev, race)
    save_replay(rp, ev)
    print(f"  {len(rp)} replay states, {rp['driver'].nunique()} drivers ({time.time()-t:.0f}s)")

    # -- 10. write artifacts ----------------------------------------------
    step("10. write artifacts")
    field = []
    for drv, g in race.groupby("driver"):
        g = g.sort_values("lap_number")
        st = (g.groupby("stint")
              .agg(compound=("compound", "first"), start_lap=("lap_number", "min"),
                   end_lap=("lap_number", "max"), laps=("lap_number", "size"))
              .sort_values("start_lap").reset_index())
        st = st[(st["laps"] >= 2) & st["compound"].isin(f.compounds)]
        if len(st) < 2:
            continue
        st["driver"] = drv
        st["stint_no"] = range(1, len(st) + 1)
        st["n_stops"] = len(st) - 1
        st["plan"] = "-".join(c[0] for c in st["compound"])
        st["finished_lap"] = int(g["lap_number"].max())
        field.append(st)
    if field:
        pd.concat(field, ignore_index=True).to_parquet(DATA_PROCESSED / f"fieldplan_{key}.parquet", index=False)

    res.table.head(400).assign(
        pit_laps=res.table.head(400)["pit_laps"].astype(str),
        stint_lens=res.table.head(400)["stint_lens"].astype(str),
    ).to_parquet(DATA_PROCESSED / f"strategy_{key}.parquet", index=False)
    res.by_stops.assign(pit_laps=res.by_stops["pit_laps"].astype(str),
                        stint_lens=res.by_stops["stint_lens"].astype(str)).to_parquet(
        DATA_PROCESSED / f"bystops_{key}.parquet", index=False)
    res.life.to_parquet(DATA_PROCESSED / f"life_{key}.parquet", index=False)
    if not pw.empty:
        pw.to_parquet(DATA_PROCESSED / f"pitwindow_{key}.parquet", index=False)
    plan_rows, lap0 = [], 0
    for i, (c, L) in enumerate(zip(res.best["compounds"], res.best["stint_lens"])):
        plan_rows.append({"stint": i + 1, "compound": c, "start_lap": lap0 + 1, "end_lap": lap0 + L, "laps": int(L)})
        lap0 += L
    pd.DataFrame(plan_rows).to_parquet(DATA_PROCESSED / f"plan_{key}.parquet", index=False)
    if not uc.empty:
        uc.to_parquet(DATA_PROCESSED / f"undercut_{key}.parquet", index=False)
    if not cf.empty:
        cf.assign(actual_pit_laps=cf["actual_pit_laps"].astype(str),
                  model_pit_laps=cf["model_pit_laps"].astype(str),
                  sc_stop_laps=cf["sc_stop_laps"].astype(str)).to_parquet(
            DATA_PROCESSED / f"counterfactual_{key}.parquet", index=False)
    if not pdp.empty:
        pdp.assign(pit_laps=pdp["pit_laps"].astype(str), practice_dev_s_per_lap=pdp["practice_dev_s_per_lap"].astype(str),
                   eff_rate=pdp["eff_rate"].astype(str)).to_parquet(DATA_PROCESSED / f"perdriver_{key}.parquet", index=False)
    if sc.per_lap is not None:
        sc.per_lap.to_parquet(DATA_PROCESSED / f"score_{key}.parquet", index=False)
    if sc.per_stint is not None:
        sc.per_stint.to_parquet(DATA_PROCESSED / f"scorestint_{key}.parquet", index=False)
    if pl.per_stop is not None:
        pl.per_stop.to_parquet(DATA_PROCESSED / f"pitloss_{key}.parquet", index=False)

    gates_all = list(fs.get("gates_fit", [])) + list(GATES)
    meta = {
        **{k: v for k, v in fs.items() if k not in ("gates_fit", "timings", "runtime_fit_s")},
        "regime": {**fs["regime"],
                   "self_measured": {"ratio": (float(_self_regime.ratio) if np.isfinite(_self_regime.ratio) else None),
                                     "per_compound": _self_regime.per_compound,
                                     "n_race_stints": int(_self_regime.n_race_stints)}},
        "compound_ladder": {**fs["compound_ladder"],
                            "self_measured": {"step_s": float(_self_step.step_s), "se": float(_self_step.se),
                                              "n_laps": int(_self_step.n_laps), "phase_bias_s": float(_self_step.phase_bias_s)}},
        "calibration": {**{k: v for k, v in cal.as_dict().items() if k != "driver_factors"}, "source": cal.source,
                        "n_driver_factors": len(cal.driver_factors),
                        "dirty_air_used": float(dirty_air),
                        "dirty_air_source": ("this circuit's 2023-25 races" if ev.circuit in cal.dirty_air_by_circuit
                                             else "pooled over the 2026 donor races")},
        "cliff_detector": {
            "n_stints": int(len(collapse_rows)), "kinds": _counts,
            "rows": (collapse_rows.to_dict("records") if not collapse_rows.empty else []),
            "budget_vs_observed": ratio_metrics,
        },
        "pace_calibration": pace_cal,
        "score": {
            "mae": float(sc.mae), "mae_lap": float(sc.mae_lap), "rmse": float(sc.rmse),
            "n_rate_stints": int(sc.n_rate_stints), "n_laps": int(sc.n_laps), "n_stints": int(sc.n_stints),
            "mae_by_compound": sc.mae_by_compound,
            "coverage": {str(k): float(v) for k, v in sc.coverage.items()},
            "rate_coverage90": float(sc.rate_coverage90), "rate_width90": float(sc.rate_width90),
            "sigma_used": float(sc.sigma_used),
            "cliff": sc.cliff, "bias": float(sc.bias), "bias_by_compound": sc.bias_by_compound,
            "regime_label": sc.regime_label, "passes_mae": bool(sc.passes_mae), "passes_coverage": bool(sc.passes_coverage),
        },
        "pit_loss_s": float(pl.seconds), "pit_stops_measured": int(pl.n_stops),
        "max_stint_laps": max_stint, "max_stints_per_compound": alloc["caps"],
        "grid_start_penalty_s": float(cal.grid_start_penalty_s),
        "backtest": bt,
        # `stint_laps` is the length the shipped model is *calibrated* at (the
        # fixed point, clipped to the tyres' lives), which is where the net step is
        # defined; `stint_laps_plan` is the plan's own median, for the reader.
        "ladder_check": {"model_net_step_s": float(_model_net), "measured_net_step_s": float(_meas_net),
                         "self_measured_net_step_s": float(_self_step.step_s), "stint_laps": float(_L),
                         "stint_laps_plan": float(_L_plan),
                         "life_cap_laps": float(pace_cal.get("life_cap", float("nan"))),
                         "passes": int(pace_cal.get("passes", 0)),
                         "converged": bool(pace_cal.get("converged", False))},
        "strategy": {
            "n_strategies": int(res.n_strategies), "n_scored": int(res.n_scored), "n_draws": int(res.n_draws),
            "best": res.best_label,
            "best_plan": {k: (list(v) if isinstance(v, list) else v) for k, v in res.best.items()},
            "tyre_optimal": {**res.tyre_optimal, "label": res.tyre_optimal_label},
            "warmup_s": float(res.warmup_s), "push": float(res.best.get("push", float("nan"))),
            "push_grid": list(res.push_grid), "implied_regime": float(res.implied_regime),
            "ordering_spread_s": float(_spread), "ordering": _ord[["compounds", "mean_s"]].to_dict("records"),
            "grip_budget_s": float(model.budget), "grip_budgets": dict(model.budgets),
            "undercut_lambda": float(res.undercut_lambda), "plan_prior_tau_s": float(res.plan_prior_tau_s),
            "first_stop_kappa_s": float(res.first_stop_kappa_s),
            "position_s": float(res.best.get("position_s", 0.0)), "prior_s": float(res.best.get("prior_s", 0.0)),
            "first_stop_s": float(res.best.get("first_stop_s", 0.0)),
            "race_state_s": float(res.best.get("race_state_s", 0.0)),
            "race_state_enabled": bool(use_rs),
            "first_stop_prior": fs_summary,
            "traffic_s_per_stop": float(strat.traffic_cost(ev, res.best["pit_laps"], s_per_lap=dirty_air)
                                        / max(1, len(res.best["pit_laps"]))),
            "safety_car_credit_s": float(strat.safety_car_credit(ev, res.best["pit_laps"], float(pl.seconds))),
            "by_stops": res.by_stops.assign(pit_laps=res.by_stops["pit_laps"].astype(str),
                                            stint_lens=res.by_stops["stint_lens"].astype(str)).to_dict("records"),
            "life": res.life.to_dict("records"),
            "pit_windows": ([{"stop": int(k), "recommended": int(res.best["pit_laps"][int(k) - 1]),
                              "lo": int(g[g["in_window"]]["lap"].min()), "hi": int(g[g["in_window"]]["lap"].max())}
                             for k, g in pw.groupby("stop")] if not pw.empty else []),
            "top": res.table.head(5).assign(pit_laps=res.table.head(5)["pit_laps"].astype(str),
                                            stint_lens=res.table.head(5)["stint_lens"].astype(str)).to_dict("records"),
        },
        "race_state": rs_block,
        "per_driver": (pdp.assign(pit_laps=pdp["pit_laps"].astype(str)).to_dict("records") if not pdp.empty else []),
        "counterfactual_top": (cf.head(3).assign(actual_pit_laps=cf.head(3)["actual_pit_laps"].astype(str),
                                                 model_pit_laps=cf.head(3)["model_pit_laps"].astype(str),
                                                 sc_stop_laps=cf.head(3)["sc_stop_laps"].astype(str)).to_dict("records")
                               if not cf.empty else []),
        "counterfactual_summary": ({"n": int(len(cf)), "n_over_30s": int((cf["loss_s"] > 30).sum()),
                                    "n_with_sc_stops": int((cf["n_sc_stops"] > 0).sum()),
                                    "median_loss_s": float(cf["loss_s"].median())} if not cf.empty else {}),
        "gates": gates_all,
        "timings": {**fs.get("timings", {}), **timings},
        "runtime_s": round(float(fs.get("runtime_fit_s", 0.0)) + time.time() - t_all, 1),
        "runtime_decide_s": round(time.time() - t_all, 1),
    }
    (DATA_PROCESSED / f"meta_{key}.json").write_text(json.dumps(meta, indent=2, default=_jd))

    n_fail = sum(1 for g in gates_all if not g["pass"])
    print(f"\n=== {len(gates_all) - n_fail}/{len(gates_all)} gates passed "
          f"(decide stage {time.time() - t_all:.0f}s) ===")
    for g in gates_all:
        if not g["pass"]:
            print(f"  FAILED: {g['gate']} — {g['detail']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="barcelona-2026")
    ap.add_argument("--stage", choices=["all", "fit", "decide"], default="all")
    ap.add_argument("--joint", action="store_true", help="also fit the joint lap-time + apex diagnostic")
    ap.add_argument("--no-diagnostics", action="store_true", help="production fit only")
    ap.add_argument("--full-diagnostics", action="store_true", help="diagnostic variants at 4x1500 instead of 2x800")
    ap.add_argument("--mc-draws", type=int, default=MC_DRAWS)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--pit-step", type=int, default=1)
    ap.add_argument("--no-race-state", action="store_true",
                    help="V3's first-stop objective (undercut exposure + first-stop prior) instead of the race state")
    ap.add_argument("--offline", action="store_true",
                    help="keep FastF1 off the network (every session already cached): the Ergast mirror's "
                         "timeouts turned a 15 s practice load into 5 minutes on the benchmark run")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.offline:
        import fastf1
        from src.config import FASTF1_CACHE
        fastf1.Cache.enable_cache(str(FASTF1_CACHE))
        fastf1.Cache.offline_mode(True)
    ev = get_event(args.event)
    fs = None
    if args.stage in ("all", "fit"):
        fs = stage_fit(args, ev)
    if args.stage in ("all", "decide"):
        return stage_decide(args, ev, fs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
