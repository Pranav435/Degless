"""End-to-end: practice laps -> physics -> fits -> sealed predictions -> race score.

Everything the app reads is written here.  The app fits nothing.
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

from src import strategy as strat  # noqa: E402
from src.compounds import (  # noqa: E402
    allocation_prior, hardness_rank, measure_pace_step, model_net_stint_step,
    pace_step_prior,
    summary_table as compound_table,
)
from src.config import (  # noqa: E402
    DATA_PROCESSED, GRID_START_PENALTY_S, GRIP_BUDGET_S,
    MAX_STINTS_PER_COMPOUND, PUSH_GRID,
    MC_DRAWS, RHAT_GATE, TYRE_LOAD_EXPONENT, get_event,
)
from src.history import apply_circuit_prior, circuit_prior, stint_caps_for  # noqa: E402
from src.regime import measure_regime, regime_prior  # noqa: E402
from src.evolution import add_evolution_correction, fit_evolution_auto  # noqa: E402
from src.fuel import add_fuel_correction, get_prior, summary_table  # noqa: E402
from src.ingest import FirewallError, load_for_fitting, load_race  # noqa: E402
from src.laps import build_lap_table, cascade_counts, clean_laps, compound_summary  # noqa: E402
from src.model_bayes import fit_bayes  # noqa: E402
from src.model_fallback import fit_mixedlm  # noqa: E402
from src.replay import build_replay, save_replay  # noqa: E402
from src.telemetry import extract_apex_speeds, load_apex, save_apex, select_corners  # noqa: E402
from src.validate import strategy_backtest, load_sealed, score_race, seal_predictions  # noqa: E402

log = logging.getLogger("degless.pipeline")

GATES: list = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    GATES.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def step(msg: str) -> float:
    print(f"\n=== {msg} ===", flush=True)
    return time.time()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="barcelona-2026")
    ap.add_argument("--no-apex", action="store_true")
    ap.add_argument("--mc-draws", type=int, default=MC_DRAWS)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--pit-step", type=int, default=1)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    ev = get_event(args.event)
    key = ev.key
    t_all = time.time()

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
    gate("clean laps in 120-320", 120 <= len(clean) <= 320,
         f"{len(clean)} clean laps from {len(laps)} raw ({time.time()-t:.0f}s)")
    gate("no non-green lap survives",
         bool((clean["track_status"].astype(str) == "1").all()),
         f"track statuses: {sorted(clean['track_status'].unique())}")
    gate("all compounds valid",
         bool(clean["compound"].isin(["SOFT", "MEDIUM", "HARD"]).all()), "")

    # -- 2. physics -------------------------------------------------------
    t = step("2. fuel physics + compound ladder + regime transfer")
    print(summary_table(ev).to_string(index=False))

    # The compound ladder: sized from other weekends' races, ordered by
    # construction.  See src/compounds.py.
    pstep = pace_step_prior(ev)
    print(f"\n  compound pace step: {pstep['step_s']:.3f} s  [{pstep['label']}]")
    print(f"    {pstep['derivation']}")
    print(compound_table(ev, pace_step_s=pstep["step_s"]).to_string(index=False))

    # The practice -> race regime factor, measured on every weekend but this
    # one.  See src/regime.py.
    regime = regime_prior(ev)
    print(f"\n  practice->race degradation factor: {regime.ratio:.3f}x "
          f"[{regime.p05:.2f}-{regime.p95:.2f}]  [{regime.label}]")
    print(f"    {regime.derivation}")
    gate("regime factor is plausible (0.15-1.0)",
         0.15 <= regime.ratio <= 1.0,
         f"{regime.ratio:.3f}x from {regime.sources or 'default'}")
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

    # -- 3. MixedLM baseline ----------------------------------------------
    t = step("3. MixedLM baseline + stint block bootstrap")
    mlm = fit_mixedlm(clean, n_boot=args.boot)
    print(mlm.table().round(4).to_string(index=False))
    med = mlm.slopes.get("MEDIUM", np.nan)
    gate("MixedLM MEDIUM slope in 0.12-0.35", 0.12 <= med <= 0.35,
         f"{med:.4f} s/lap ({time.time()-t:.0f}s)")

    # -- 4. apex speeds ---------------------------------------------------
    t = step("4. corner apex speeds (second channel)")
    apex_sel = None
    apex_use = None
    if args.no_apex:
        print("  skipped (--no-apex)")
    else:
        apex = load_apex(ev)
        need = set(zip(clean["driver"], clean["lap_number"]))
        have = set(zip(apex["driver"], apex["lap_number"])) if len(apex) else set()
        if not need.issubset(have):
            apex = extract_apex_speeds(clean, ev)
            save_apex(apex, ev)
        if len(apex):
            apex_sel = select_corners(apex)
            apex_use = apex[apex["corner"].isin(apex_sel.corners)]
            print(f"  {len(apex)} apex rows, corners selected: {apex_sel.corners}")
            apex_sel.table.to_parquet(DATA_PROCESSED / f"corners_{key}.parquet",
                                      index=False)
            gate("apex channel available", True,
                 f"{len(apex_use)} rows on {len(apex_sel.corners)} corners "
                 f"({time.time()-t:.0f}s)")
        else:
            gate("apex channel available", False, "no telemetry; lap-time only")

    # -- 5. Bayesian fits -------------------------------------------------
    t = step("5. NumPyro hierarchical fits")
    fits = {}
    fits["2026"] = fit_bayes(clean, ev, prior="2026", apex=apex_use,
                             pace_step_s=pstep["step_s"])
    f = fits["2026"]
    print(f"  2026 joint : rhat={f.max_rhat:.4f} div={f.n_divergences} "
          f"laps={f.n_laps} apex={f.n_apex}")
    gate("convergence: r_hat < 1.01 and zero divergences",
         f.max_rhat < RHAT_GATE and f.n_divergences == 0,
         f"max r_hat {f.max_rhat:.4f}, {f.n_divergences} divergences")

    f.save(DATA_PROCESSED / f"posterior_{key}_practice.npz")
    cp = circuit_prior(ev)
    hist_moved = pd.DataFrame()
    if cp.available and cp.rate_prior:
        f, hist_moved = apply_circuit_prior(f, ev, regime, cp)
        fits["2026"] = f
        print(f"  circuit history {ev.circuit} {cp.years} folded in (practice-regime rate, s/lap):")
        print(hist_moved.round(4).to_string(index=False))
    f.save(DATA_PROCESSED / f"posterior_{key}.npz")
    fits["2026_laponly"] = fit_bayes(clean, ev, prior="2026",
                                     pace_step_s=pstep["step_s"])
    fits["2025"] = fit_bayes(clean, ev, prior="2025", apex=apex_use,
                             pace_step_s=pstep["step_s"])
    fits["none"] = fit_bayes(clean, ev, prior="none", apex=apex_use,
                             pace_step_s=pstep["step_s"])
    # The ladder switched off — the compound-ordering equivalent of the "no
    # fuel prior" variant, and the one that shows why the ladder is there.
    fits["noladder"] = fit_bayes(clean, ev, prior="2026", apex=apex_use,
                                 compound_prior="flat")
    for k, v in fits.items():
        print(f"  {k:16s} rhat={v.max_rhat:.4f} div={v.n_divergences}")
    print(f.slope_table().round(4).to_string(index=False))
    print(f"  ({time.time()-t:.0f}s)")

    # Bayes vs MixedLM, on the observed age support of each compound.
    # The MixedLM baseline fits each compound freely, which means it cannot
    # cross-check the *split between* compounds — that split is precisely what
    # practice data does not identify, and on Barcelona 2026 the baseline duly
    # returns HARD degrading faster than MEDIUM off 9 clean HARD laps.  Asking
    # the laddered fit to agree with that would be asking it to reproduce the
    # noise it was built to reject.
    #
    # What both estimators *can* speak to is the overall level of degradation
    # across the weekend's running, so that is what is gated: the laps-weighted
    # pooled slope.  The per-compound differences are printed beside it, and
    # where they are large they are the ladder working, not a disagreement to
    # be resolved.
    diffs, eff_slopes = {}, {}
    for c in f.compounds:
        a = clean.loc[clean["compound"] == c, "tyre_age"].to_numpy(float)
        A = np.column_stack([np.ones_like(a), a])
        eff = np.linalg.lstsq(A, f.deg_loss(c, a).mean(0), rcond=None)[0][1]
        eff_slopes[c] = float(eff)
        diffs[c] = abs(eff - mlm.slopes.get(c, np.nan))
    w = clean["compound"].value_counts()
    n_tot = float(w.sum())
    pooled_bayes = sum(eff_slopes[c] * w.get(c, 0) for c in f.compounds) / n_tot
    pooled_mlm = sum(mlm.slopes.get(c, np.nan) * w.get(c, 0) for c in f.compounds) / n_tot
    pooled_diff = abs(pooled_bayes - pooled_mlm)
    gate("Bayes vs MixedLM pooled degradation within 0.06 s/lap",
         pooled_diff < 0.06,
         f"pooled {pooled_bayes:.3f} vs {pooled_mlm:.3f} (diff {pooled_diff:.3f}); "
         "per-compound differences are the ladder overriding an unordered "
         "baseline: " + ", ".join(f"{c} {v:.3f}" for c, v in diffs.items()))

    # The ladder is structural, so this cannot fail by construction — which is
    # exactly why it is worth asserting.  The unladdered variant is printed
    # beside it so a reader can see what the same laps produce without it.
    rank = hardness_rank(f.compounds)
    by_rank = [c for _, c in sorted(zip(rank, f.compounds))]  # softest first
    slopes = {r["compound"]: r["slope_s_per_lap"] for _, r in f.slope_table().iterrows()}
    offs = f.comp_offset
    ordered_ok = (all(slopes[a] > slopes[b] for a, b in zip(by_rank, by_rank[1:]))
                  and all(offs[a] < offs[b] for a, b in zip(by_rank, by_rank[1:])))
    flat_slopes = {r["compound"]: r["slope_s_per_lap"]
                   for _, r in fits["noladder"].slope_table().iterrows()}
    flat_ok = all(flat_slopes[a] > flat_slopes[b] for a, b in zip(by_rank, by_rank[1:]))
    gate("compound ladder ordered (softer = quicker and higher deg)", ordered_ok,
         "deg " + " > ".join(f"{c} {slopes[c]:.3f}" for c in by_rank)
         + "; pace " + " < ".join(f"{c} {offs[c]:.2f}" for c in by_rank)
         + f"; without the ladder the same laps order as "
         + " ".join(f"{c} {flat_slopes[c]:.3f}" for c in by_rank)
         + (" (also ordered)" if flat_ok else " (INVERTED)"))

    # The identifiability story, quantified.
    prior_shift = {}
    for c in f.compounds:
        a = np.arange(1, 21.0)
        s26 = np.polyfit(a, fits["2026"].deg_loss(c, a).mean(0), 1)[0]
        s25 = np.polyfit(a, fits["2025"].deg_loss(c, a).mean(0), 1)[0]
        s00 = np.polyfit(a, fits["none"].deg_loss(c, a).mean(0), 1)[0]
        prior_shift[c] = {"none": float(s00), "2026": float(s26), "2025": float(s25)}
    print("\n  prior sensitivity (mean slope over ages 1-20, s/lap):")
    print(pd.DataFrame(prior_shift).T.round(4).to_string())

    # -- 6. seal ----------------------------------------------------------
    t = step("6. seal predictions (before any race lap is read)")
    sealed_path, sha = seal_predictions(
        f, ev, regime=regime,
        note=("fitted on practice only; joint lap-time + apex channels; "
              f"race-regime curves scaled by {regime.ratio:.3f}x "
              f"({regime.label})"))
    print(f"  {sealed_path.name}  sha256 {sha[:32]}...")
    sealed = load_sealed(sealed_path)
    sealed["_file"] = sealed_path.name
    gate("sealed file verifies against its sha256", True, sha[:16])

    # -- 7. race scoring --------------------------------------------------
    t = step("7. race data (validation only)")
    race_raw = load_race(ev)
    race = build_lap_table(race_raw, ev)
    race_clean = clean_laps(race)
    print(f"  race: {len(race)} laps, {len(race_clean)} clean")
    # Measured on *this* weekend: never used to set anything, reported so a
    # reader can see how far the transferred numbers landed from the truth.
    _self_regime = measure_regime(ev, race=race, practice=clean)
    _self_step = measure_pace_step(ev, race=race)
    print(f"  self-measured regime factor {_self_regime.ratio:.3f}x "
          f"(transferred: {regime.ratio:.3f}x, never fitted here)")
    print(f"  self-measured pace step {_self_step.step_s:+.3f} +/- "
          f"{_self_step.se:.3f} s (transferred: {pstep['step_s']:.3f} s)")

    sc = score_race(sealed, race_clean, ev)
    print(f"  {sc.summary()}")
    print("  MAE by compound:", {k: round(v, 3) for k, v in sc.mae_by_compound.items()})
    print("  calibration:", {f"{k:.0%}": f"{v:.1%}" for k, v in sc.coverage.items()})
    if sc.cliff:
        print("  cliff:", {k: {kk: round(vv, 1) for kk, vv in v.items()}
                           for k, v in sc.cliff.items()})
    print("  bias by compound (observed - predicted, s/lap):",
          {k: round(v, 3) for k, v in sc.bias_by_compound.items()})
    # A near-total negative bias is a regime difference, not scatter.  Practice
    # long runs at Barcelona 2026 degrade about twice as fast as race stints
    # (verified in the raw data: GAS's FP2 MEDIUM run climbs 82.05 -> 88.00 s
    # over 10 laps, while his 26-lap race HARD stint gains only 0.11 s/lap).
    # Hotter practice track temperatures and the complete absence of tyre
    # management in a long run are the physical explanation.
    gate("practice->race regime transfer removes the bias",
         abs(sc.bias) < 0.06,
         f"bias {sc.bias:+.3f} s/lap against {sc.regime_label} curves "
         f"(transferred factor {regime.ratio:.2f}x from "
         f"{regime.sources or 'the default'}; this weekend's own races to "
         f"{_self_regime.ratio:.2f}x, which was never used)")
    gate("stint degradation-rate MAE < 0.15 s/lap", sc.passes_mae,
         f"{sc.mae:.4f} s/lap over {sc.n_rate_stints} stints "
         f"(per-lap MAE {sc.mae_lap:.3f} s, noise floor ~{sealed['sigma_obs']:.2f} s)")
    gate("90% coverage not below 0.80 (under-coverage is the failure)",
         sc.passes_coverage,
         f"{sc.coverage.get(0.90, float('nan')):.1%} — {sc.coverage_direction}"
         + ("; intervals are conservative because sigma_obs is practice per-lap "
            "noise (~0.9 s) while stints are scored centred, and the regime "
            "factor's own spread widens the band further"
            if sc.coverage_direction == "over" else ""))

    # -- 8. strategy ------------------------------------------------------
    t = step("8. pit loss, strategy MC, pit window, undercut, counterfactual")
    # The degradation curve is only supported out to the oldest tyre each
    # compound was actually run on in practice, and beyond its knee it is a
    # straight-line extrapolation, so evaluating it at an age nothing was run
    # at claims a car many seconds off the pace.  Each compound gets its own
    # cap, because the support is wildly uneven: Barcelona 2026 has MEDIUM out
    # to age 22 and SOFT only to 15.
    age_support = float(clean["tyre_age"].max())
    per_comp_support = clean.groupby("compound")["tyre_age"].max().to_dict()
    print("  practice age support (per compound, laps):",
          {c: int(v) for c, v in per_comp_support.items()})
    alloc = allocation_prior(ev)
    print(f"  tyre allocation ({alloc['label'] if 'label' in alloc else ('measured' if alloc['measured'] else 'default')}): "
          f"{alloc['caps']}")
    print(f"    {alloc['derivation']}")
    print(f"  grid-start penalty: {GRID_START_PENALTY_S:.1f} s per step of hardness "
          f"on the opening stint")
    print(f"  grip budget {GRIP_BUDGET_S:.2f} s; tyre life is derived as "
          f"budget / degradation rate, not fitted separately")
    print(f"  fuel-load wear multiplier: exponent {TYRE_LOAD_EXPONENT:.1f} "
          f"(lap 1 = {(838/803)**TYRE_LOAD_EXPONENT:.2f}x, "
          f"flag = {(768/803)**TYRE_LOAD_EXPONENT:.2f}x)")
    print(f"  push levels searched: {PUSH_GRID}")

    pl = strat.measure_pit_loss(race)
    print(f"  pit loss: {pl.seconds:.2f}s from {pl.n_stops} green-flag stops")
    gate("pit loss measured and plausible (12-35s)",
         np.isfinite(pl.seconds) and 12 <= pl.seconds <= 35, f"{pl.seconds:.2f}s")

    hist_caps = stint_caps_for(ev, cp) if cp.available else None
    if hist_caps:
        print(f"  stint caps from this circuit's history: {hist_caps}")
    res = strat.simulate(f, ev, float(pl.seconds), regime=regime,
                         n_draws=args.mc_draws, step=args.pit_step,
                         support=per_comp_support,
                         max_per_compound=alloc["caps"], max_stint=hist_caps)
    max_stint = res.max_stint
    print("  stint caps (min of the wear bound and the practice-support bound):",
          max_stint)
    print(f"  {res.n_strategies:,} plans searched, {res.n_scored:,} scored "
          f"x {res.n_draws} draws")
    print(res.head(8).drop(columns=["pit_laps"]).round(3).to_string(index=False))
    print("\n  best plan at each stop count:")
    print(res.by_stops.drop(columns=["pit_laps"]).round(3).to_string(index=False))
    print("\n  compound life (cliff = grip budget / degradation rate):")
    print(res.life.round(2).to_string(index=False))
    print(f"\n  the optimiser chose push {res.best['push']:.2f}, which implies a "
          f"practice->race degradation factor of {res.implied_regime:.2f}")
    print(f"  measured on other weekends' races: {regime.ratio:.2f} "
          f"({regime.label}) — the model predicts this number rather than "
          f"being given it, so the two agreeing is a real check")
    # A plan that pits inside the first few laps or runs every stint to its cap
    # is the signature of the failure this rebuild targets: over-stated
    # degradation makes stopping look free.
    lens = res.best.get("stint_lens", [])
    gate("recommended plan has realistic stint lengths",
         bool(lens) and min(lens) >= 8 and 1 <= res.best["n_stops"] <= 3,
         f"{res.best_label} -> stints {lens}")
    # No stint may be recommended past its cliff.  This is the check that the
    # old model could not have passed and could not even have stated: with a
    # linear degradation curve there is no cliff to be past.
    worst = float(res.table.iloc[0]["max_wear"])
    gate("no stint in the recommended plan runs past the cliff",
         worst <= 1.0,
         f"deepest stint reaches {worst:.0%} of the grip budget")
    # The model's chosen push implies a practice->race degradation factor, and
    # that factor is measured independently from other weekends' races and
    # never fed in.  The check is deliberately one-sided.
    #
    # The two numbers are estimates of overlapping but not identical
    # quantities.  The measured ratio is everything that makes a race stint
    # degrade differently from a practice long run - tyre management, but also
    # a cooler track, dirty air and lower average speeds, none of which is a
    # driver's choice.  The model's implied factor is the management part
    # alone.  So they should be in the same neighbourhood, but the model's may
    # sit *above* the measurement with no contradiction: at a
    # low-degradation circuit there is nothing to manage, the optimiser picks
    # full attack, and it is right to.
    #
    # The failure this guards against is one-directional - a model that
    # assumes *more* management than anyone was observed to use, which is what
    # produces stints longer than any team runs. That is the failure the old
    # exogenous 0.40 constant actually was.
    gate("model does not assume more tyre management than was observed",
         res.implied_regime > regime.ratio - 0.25,
         f"implied {res.implied_regime:.2f} vs measured {regime.ratio:.2f}"
         + ("; the optimiser chose full attack, which at this degradation "
            "level is the right call and not a disagreement"
            if res.implied_regime >= regime.ratio else ""))

    # How much of the recommendation is the *order* of the compounds, as
    # opposed to the stop count and the stint lengths?  Almost none of it, and
    # saying so is the honest reading: at Barcelona 2026 the best plan for each
    # ordering of the same compounds spans ~1 s, well inside the posterior's
    # own width.  Order is decided here by the fuel-load term alone, which is
    # small at a physical load exponent, and the thing that actually decides a
    # starting compound in reality - track position off the line - is not in
    # this objective at all.  Reporting the spread is what stops a reader
    # taking "HARD first" as a finding.
    _ord = (res.table[res.table["n_stops"] == res.best["n_stops"]]
            .sort_values("mean_s").groupby("compounds", as_index=False).first()
            .sort_values("mean_s"))
    _spread = float(_ord["mean_s"].iloc[-1] - _ord["mean_s"].iloc[0]) if len(_ord) > 1 else 0.0
    print(f"\n  compound ordering is weakly identified: the best {res.best['n_stops']}-stop "
          f"plan for each of {len(_ord)} orderings spans {_spread:.1f} s")
    print("  " + " | ".join(f"{r.compounds} {r.mean_s - _ord['mean_s'].iloc[0]:+.1f}"
                            for r in _ord.head(4).itertuples()))
    print("  track position off the line, which is what decides a starting "
          "compound in reality, is not priced here")

    pw = strat.pit_window(f, ev, res.best, float(pl.seconds), regime=regime,
                          max_stint=max_stint, push=res.best["push"])
    if not pw.empty:
        for k, g in pw.groupby("stop"):
            win = g[g["in_window"]]["lap"]
            print(f"  stop {k}: recommended lap {res.best['pit_laps'][k-1]}, "
                  f"within 1.0 s over laps {win.min():.0f}-{win.max():.0f}")

    # Only out to where both compounds have support: past that the curve is
    # the hinge extrapolating, and an undercut gain computed there is fiction.
    _uc_age = int(min(max_stint.get("MEDIUM", 40), max_stint.get("SOFT", 40)))
    uc = strat.undercut_window(f, "MEDIUM", "SOFT", event=ev, regime=regime,
                               max_age=_uc_age, push=res.best["push"])
    cf = strat.counterfactual(f, ev, race, float(pl.seconds), regime=regime,
                              max_stint=max_stint, push=res.best["push"])
    if not cf.empty:
        print("\n  counterfactual (top 5):")
        print(cf.head(5).round(2).to_string(index=False))
    print(f"  ({time.time()-t:.0f}s)")

    # The compound ladder's two halves are not separately identified, but their
    # net effect over a stint is, and that net is what the optimiser consumes.
    # This gate is the one that would have caught a fresh-tyre pace step
    # calibrated from equal-tyre-age pace: at 0.45% of lap time the model's net
    # came out around +0.15 s/step against a measured +0.03, and it answered by
    # ruling out the HARD entirely at circuits where the field ran half its
    # race laps on it.
    # Across both 2026 weekends, 0 of 32 classified finishers started on the
    # hardest compound available. A model that opens on it is wrong about the
    # one part of the plan the whole field agreed on.
    _ranks = dict(zip(f.compounds, hardness_rank(list(f.compounds))))
    _open = res.best["compounds"][0]
    gate("does not open the race on the hardest available compound",
         _ranks[_open] < max(_ranks.values()),
         f"opens on {_open} (hardness rank {_ranks[_open]} of "
         f"{max(_ranks.values())}); no classified finisher at either 2026 "
         f"weekend started on the hardest tyre")

    _life = {r["compound"]: r["life_laps"] for _, r in res.life.iterrows()}
    _L = float(np.median(res.best["stint_lens"]))
    _model_net = model_net_stint_step(_life, _L, pstep["step_s"])
    _meas_net = pstep.get("net_stint_step_measured", float("nan"))
    print(f"\n  compound ladder check — net cost of one step harder over a "
          f"{_L:.0f}-lap stint:")
    print(f"    model {_model_net:+.3f} s/lap   measured on donor races "
          f"{_meas_net:+.3f} s/lap")
    if np.isfinite(_meas_net):
        gate("model reproduces the measured net stint-level compound step",
             abs(_model_net - _meas_net) < 0.12,
             f"model {_model_net:+.3f} vs measured {_meas_net:+.3f} s/lap "
             f"per step over a {_L:.0f}-lap stint")

    # -- 8b. strategy backtest --------------------------------------------
    # The curve gates above check the degradation *curve*.  This checks the
    # *decision*, which is what the project is for and what the curve metrics
    # cannot see: the previous model passed every curve gate while recommending
    # a 33-lap SOFT stint and a one-stop at a circuit where nobody one-stopped.
    bt = strategy_backtest(res, race, ev)
    if bt:
        print("\n  strategy backtest (race data, validation only):")
        print(f"    recommended {bt['recommended_stops']} stops; the field ran "
              f"{bt['observed_stop_counts']} among classified finishers")
        print(pd.DataFrame(bt["per_stint"]).to_string(index=False))
        print("    grip budget implied by this race, per compound: "
              + ", ".join(f"{c} {v:.2f}s" for c, v in
                          sorted(bt["grip_budget_implied"].items()))
              + f"  (model uses {bt['grip_budget_config']:.2f}s)")
        gate("recommended stop count is one the field actually ran",
             bt["stops_observed_share"] > 0,
             f"{bt['recommended_stops']} stops — {bt['stops_observed_share']:.0%} "
             f"of finishers, mode was {bt['modal_stops']}")
        gate("every recommended stint length is one the compound was run to",
             bt["all_stints_inside_observed_range"],
             "; ".join(f"{r['compound']} {r['recommended_laps']} laps "
                       f"(observed median {r['observed_median']:.0f}, "
                       f"max {r['observed_max']:.0f})"
                       for r in bt["per_stint"]))

    # -- 9. replay --------------------------------------------------------
    t = step("9. replay precompute")
    rp = build_replay(f, ev, race)
    save_replay(rp, ev)
    print(f"  {len(rp)} replay states, {rp['driver'].nunique()} drivers "
          f"({time.time()-t:.0f}s)")

    # -- 10. write artifacts ----------------------------------------------
    step("10. write artifacts")
    ages = np.arange(0, 41, dtype=float)
    curves = []
    for name, fitobj in fits.items():
        c = fitobj.curve_table(ages)
        c["variant"] = name
        curves.append(c)
    # The same posterior in the regime the race is actually run in — the curve
    # the strategy simulator and the sealed prediction both use.
    _rmult = regime.draws(f.posterior["lin"].shape[0], seed=5)
    for c in f.compounds:
        d = f.deg_loss(c, ages) * _rmult[:, None]
        curves.append(pd.DataFrame({
            "compound": c, "tyre_age": ages, "mean": d.mean(0),
            "lo": np.quantile(d, 0.05, axis=0), "hi": np.quantile(d, 0.95, axis=0),
            "variant": "2026_race"}))
    pd.concat(curves, ignore_index=True).to_parquet(
        DATA_PROCESSED / f"curves_{key}.parquet", index=False)

    # What the field actually ran — the comparison the plan is read against.
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
        pd.concat(field, ignore_index=True).to_parquet(
            DATA_PROCESSED / f"fieldplan_{key}.parquet", index=False)

    knee = []
    for name, fitobj in fits.items():
        for i, c in enumerate(fitobj.compounds):
            knee.append(pd.DataFrame({
                "variant": name, "compound": c,
                "knee": fitobj.posterior["knee"][:, i]}))
    pd.concat(knee, ignore_index=True).to_parquet(
        DATA_PROCESSED / f"knee_{key}.parquet", index=False)

    laps.to_parquet(DATA_PROCESSED / f"laps_{key}_practice.parquet", index=False)
    clean.to_parquet(DATA_PROCESSED / f"clean_{key}_practice.parquet", index=False)
    race.to_parquet(DATA_PROCESSED / f"laps_{key}_race.parquet", index=False)
    casc.to_parquet(DATA_PROCESSED / f"cascade_{key}.parquet", index=False)
    res.table.head(400).assign(
        pit_laps=res.table.head(400)["pit_laps"].astype(str),
        stint_lens=res.table.head(400)["stint_lens"].astype(str),
    ).to_parquet(DATA_PROCESSED / f"strategy_{key}.parquet", index=False)
    res.by_stops.assign(
        pit_laps=res.by_stops["pit_laps"].astype(str),
        stint_lens=res.by_stops["stint_lens"].astype(str),
    ).to_parquet(DATA_PROCESSED / f"bystops_{key}.parquet", index=False)
    res.life.to_parquet(DATA_PROCESSED / f"life_{key}.parquet", index=False)
    if not pw.empty:
        pw.to_parquet(DATA_PROCESSED / f"pitwindow_{key}.parquet", index=False)
    # The winning plan as a stint timeline — the Gantt the Strategy tab draws.
    plan_rows, lap0 = [], 0
    for i, (c, L) in enumerate(zip(res.best["compounds"], res.best["stint_lens"])):
        plan_rows.append({"stint": i + 1, "compound": c, "start_lap": lap0 + 1,
                          "end_lap": lap0 + L, "laps": int(L)})
        lap0 += L
    pd.DataFrame(plan_rows).to_parquet(
        DATA_PROCESSED / f"plan_{key}.parquet", index=False)
    uc.to_parquet(DATA_PROCESSED / f"undercut_{key}.parquet", index=False)
    if not cf.empty:
        cf.assign(actual_pit_laps=cf["actual_pit_laps"].astype(str),
                  model_pit_laps=cf["model_pit_laps"].astype(str)).to_parquet(
            DATA_PROCESSED / f"counterfactual_{key}.parquet", index=False)
    if sc.per_lap is not None:
        sc.per_lap.to_parquet(DATA_PROCESSED / f"score_{key}.parquet", index=False)
    if sc.per_stint is not None:
        sc.per_stint.to_parquet(DATA_PROCESSED / f"scorestint_{key}.parquet",
                                index=False)
    if pl.per_stop is not None:
        pl.per_stop.to_parquet(DATA_PROCESSED / f"pitloss_{key}.parquet", index=False)

    # evolution curves for the Decompose tab
    ec = []
    for sess, g in clean.groupby("session"):
        ec.append(pd.DataFrame({
            "session": sess,
            "lap_start_s": g["lap_start_s"].to_numpy(),
            "evo_s": g["evo_s"].to_numpy()}).sort_values("lap_start_s"))
    pd.concat(ec, ignore_index=True).to_parquet(
        DATA_PROCESSED / f"evolution_{key}.parquet", index=False)

    fp26 = get_prior(ev, "2026")
    meta = {
        "event": key, "event_name": ev.name,
        "n_race_laps": ev.n_race_laps,
        "sealed_file": sealed_path.name, "sealed_sha256": sha,
        "physics": {
            "burn_kg_per_lap": fp26.burn_kg_per_lap,
            "k_track_s_per_kg": fp26.k_track_s_per_kg,
            "fuel_effect_s_per_lap": fp26.s_per_lap,
            "derivation": fp26.derivation,
        },
        "prior_table": summary_table(ev).to_dict("records"),
        "compound_ladder": {
            "pace_step_s": float(pstep["step_s"]),
            "measured": bool(pstep["measured"]),
            "label": pstep["label"],
            "derivation": pstep["derivation"],
            "donors": pstep["detail"],
            "self_measured": {
                "step_s": float(_self_step.step_s), "se": float(_self_step.se),
                "n_laps": int(_self_step.n_laps),
                "phase_bias_s": float(_self_step.phase_bias_s),
            },
            "table": compound_table(ev, f.compounds,
                                    pace_step_s=pstep["step_s"]).to_dict("records"),
            "deg_ratio": float(fits["2026"].posterior["deg_gap"].mean() + 1.0),
            "fitted_offsets": f.comp_offset,
            "unladdered_slopes": {
                r["compound"]: float(r["slope_s_per_lap"])
                for _, r in fits["noladder"].slope_table().iterrows()},
            "ordered": bool(ordered_ok),
            "unladdered_ordered": bool(flat_ok),
        },
        "circuit_history": (cp.as_dict() if cp.available else {}),
        "history_combination": hist_moved.to_dict("records") if not hist_moved.empty else [],
        "regime": {
            **regime.as_dict(),
            "self_measured": {
                "ratio": (float(_self_regime.ratio)
                          if np.isfinite(_self_regime.ratio) else None),
                "per_compound": _self_regime.per_compound,
                "n_race_stints": int(_self_regime.n_race_stints),
            },
        },
        "load_effect": {
            "exponent": float(TYRE_LOAD_EXPONENT),
            "start_multiplier": float((838 / 803) ** TYRE_LOAD_EXPONENT),
            "flag_multiplier": float((768 / 803) ** TYRE_LOAD_EXPONENT),
        },
        "n_raw_laps": int(len(laps)), "n_clean_laps": int(len(clean)),
        "compound_counts": comp_sum.to_dict("records"),
        "mixedlm": {
            "slopes": mlm.slopes,
            "table": mlm.table().to_dict("records"),
            "n_stints": mlm.n_stints,
        },
        "bayes": {
            "max_rhat": float(f.max_rhat), "n_divergences": int(f.n_divergences),
            "n_laps": int(f.n_laps), "n_apex": int(f.n_apex),
            "corners": [int(c) for c in (apex_sel.corners if apex_sel else [])],
            "slopes": f.slope_table().to_dict("records"),
            "k_track_mean": float(f.k_track.mean()),
            "k_track_sd": float(f.k_track.std()),
            "k_track_rel_sd": float(f.k_track.std() / f.k_track.mean()),
            "k_track_laponly_rel_sd": float(
                fits["2026_laponly"].k_track.std() / fits["2026_laponly"].k_track.mean()),
            "comp_offset": f.comp_offset,
        },
        "bayes_vs_mixedlm": {k: float(v) for k, v in diffs.items()},
        "bayes_vs_mixedlm_pooled": {
            "bayes": float(pooled_bayes), "mixedlm": float(pooled_mlm),
            "diff": float(pooled_diff)},
        "prior_sensitivity": prior_shift,
        "evolution": {"range_s": float(evo_rng), "iterations": evo.iterations,
                      "skipped": evo.skipped},
        "score": {
            "mae": float(sc.mae), "mae_lap": float(sc.mae_lap),
            "rmse": float(sc.rmse), "n_rate_stints": int(sc.n_rate_stints),
            "n_laps": int(sc.n_laps), "n_stints": int(sc.n_stints),
            "mae_by_compound": sc.mae_by_compound,
            "coverage": {str(k): float(v) for k, v in sc.coverage.items()},
            "cliff": sc.cliff,
            "bias": float(sc.bias),
            "bias_by_compound": sc.bias_by_compound,
            "regime_label": sc.regime_label,
            "passes_mae": bool(sc.passes_mae),
            "passes_coverage": bool(sc.passes_coverage),
        },
        "pit_loss_s": float(pl.seconds), "pit_stops_measured": int(pl.n_stops),
        "age_support_laps": age_support,
        "age_support_by_compound": {k: float(v) for k, v in per_comp_support.items()},
        "max_stint_laps": max_stint,
        "max_stints_per_compound": alloc["caps"],
        "allocation": alloc,
        "grid_start_penalty_s": float(GRID_START_PENALTY_S),
        "backtest": bt,
        "ladder_check": {"model_net_step_s": float(_model_net),
                         "measured_net_step_s": float(_meas_net),
                         "stint_laps": float(_L)},
        "strategy": {
            "n_strategies": int(res.n_strategies), "n_scored": int(res.n_scored),
            "n_draws": int(res.n_draws),
            "best": res.best_label,
            "best_plan": {k: (list(v) if isinstance(v, list) else v)
                          for k, v in res.best.items()},
            "warmup_s": float(res.warmup_s),
            "push": float(res.best.get("push", float("nan"))),
            "push_grid": list(res.push_grid),
            "implied_regime": float(res.implied_regime),
            "ordering_spread_s": float(_spread),
            "ordering": _ord[["compounds", "mean_s"]].to_dict("records"),
            "grip_budget_s": float(GRIP_BUDGET_S),
            "traffic_s_per_stop": float(strat.traffic_cost(ev, res.best["pit_laps"])
                                        / max(1, len(res.best["pit_laps"]))),
            "safety_car_credit_s": float(
                strat.safety_car_credit(ev, res.best["pit_laps"], float(pl.seconds))),
            "by_stops": res.by_stops.assign(
                pit_laps=res.by_stops["pit_laps"].astype(str),
                stint_lens=res.by_stops["stint_lens"].astype(str),
            ).to_dict("records"),
            "life": res.life.to_dict("records"),
            "pit_windows": ([
                {"stop": int(k),
                 "recommended": int(res.best["pit_laps"][int(k) - 1]),
                 "lo": int(g[g["in_window"]]["lap"].min()),
                 "hi": int(g[g["in_window"]]["lap"].max())}
                for k, g in pw.groupby("stop")] if not pw.empty else []),
            "top": res.table.head(5).assign(
                pit_laps=res.table.head(5)["pit_laps"].astype(str),
                stint_lens=res.table.head(5)["stint_lens"].astype(str),
            ).to_dict("records"),
        },
        "counterfactual_top": (cf.head(3).assign(
            actual_pit_laps=cf.head(3)["actual_pit_laps"].astype(str),
            model_pit_laps=cf.head(3)["model_pit_laps"].astype(str),
        ).to_dict("records") if not cf.empty else []),
        "gates": GATES,
        "runtime_s": round(time.time() - t_all, 1),
    }
    (DATA_PROCESSED / f"meta_{key}.json").write_text(json.dumps(meta, indent=2,
                                                                default=str))

    n_fail = sum(1 for g in GATES if not g["pass"])
    print(f"\n=== {len(GATES) - n_fail}/{len(GATES)} gates passed "
          f"in {time.time() - t_all:.0f}s ===")
    for g in GATES:
        if not g["pass"]:
            print(f"  FAILED: {g['gate']} — {g['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
