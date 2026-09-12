"""Weekend model: fit on whatever practice has run so far, seal, and write the
pre-race plan.  No race data is needed, or read.

This is the script to run on Friday night and again after FP3.  It is the
practice half of `10_pipeline.py` — same cascade, same physics, same fits —
without the race-scoring half, plus the posterior draws persisted so the live
engine can load the sealed model during the race without refitting.

    .venv/bin/python scripts/40_weekend.py --event italy-2026
    .venv/bin/python scripts/40_weekend.py --event italy-2026 --sessions "Practice 1" "Practice 2"
    .venv/bin/python scripts/40_weekend.py --event italy-2026 --quick     # lap-time channel only, fewer draws

Outputs (all under data/processed/):
    posterior_<key>.npz/.json   posterior draws for the live engine
    weekend_<key>.json          everything the app's pre-race view and the engine need
    curves_/knee_/life_/plan_/bystops_/pitwindow_/undercut_/strategy_<key>.parquet
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

from src import strategy as strat  # noqa: E402
from src.compounds import allocation_prior, pace_step_prior, summary_table as compound_table  # noqa: E402
from src.config import (  # noqa: E402
    DATA_PROCESSED, GRIP_BUDGET_S, MC_DRAWS, RHAT_GATE, TYRE_LOAD_EXPONENT,
    PRACTICE_SESSIONS, get_event,
)
from src.evolution import add_evolution_correction, fit_evolution_auto  # noqa: E402
from src.fuel import add_fuel_correction, get_prior, summary_table  # noqa: E402
from src.ingest import FirewallError, load_for_fitting  # noqa: E402
from src.laps import build_lap_table, cascade_counts, clean_laps, compound_summary  # noqa: E402
from src.history import apply_circuit_prior, circuit_prior, stint_caps_for  # noqa: E402
from src.live.engine import pit_loss_prior  # noqa: E402
from src.model_bayes import fit_bayes  # noqa: E402
from src.model_fallback import fit_mixedlm  # noqa: E402
from src.regime import regime_prior  # noqa: E402
from src.telemetry import extract_apex_speeds, load_apex, save_apex, select_corners  # noqa: E402
from src.validate import load_sealed, seal_predictions  # noqa: E402

log = logging.getLogger("degless.weekend")
GATES: list = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    GATES.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def step(msg: str) -> float:
    print(f"\n=== {msg} ===", flush=True)
    return time.time()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="practice sessions to fit on (default: every one that has run)")
    ap.add_argument("--no-apex", action="store_true")
    ap.add_argument("--quick", action="store_true", help="lap-time channel only, 2 chains x 800 draws")
    ap.add_argument("--mc-draws", type=int, default=MC_DRAWS)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--pit-loss", type=float, default=None, help="override the pit-loss prior (s)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ev = get_event(args.event)
    key = ev.key
    t_all = time.time()

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
    gate("enough clean long-run laps (>= 60)", len(clean) >= 60,
         f"{len(clean)} clean laps from {len(laps)} raw over {used} ({time.time()-t:.0f}s)")
    gate("no non-green lap survives", bool((clean["track_status"].astype(str) == "1").all()), "")
    if len(clean) < 30:
        return 1

    # -- 2. physics + priors ------------------------------------------------
    step("2. fuel physics, compound ladder, regime, allocation, pit loss")
    print(summary_table(ev).to_string(index=False))
    pstep = pace_step_prior(ev)
    print(f"  compound pace step: {pstep['step_s']:.3f} s [{pstep['label']}]")
    regime = regime_prior(ev)
    print(f"  practice->race factor: {regime.ratio:.3f}x [{regime.p05:.2f}-{regime.p95:.2f}] ({regime.label})")
    alloc = allocation_prior(ev)
    print(f"  allocation: {alloc['caps']} ({'measured' if alloc['measured'] else 'default'})")
    # What this circuit has done before: the prior that practice cannot supply.
    cp = circuit_prior(ev)
    if cp.available:
        print(f"  circuit history: {ev.circuit} {cp.years}; stops {cp.stops}; plans {cp.plans}")
        for c in ("SOFT", "MEDIUM", "HARD"):
            if c in cp.stint_typical:
                t_ = cp.stint_typical[c]
                print(f"    {c:7s} stints p50 {t_['p50']:.0f} p90 {t_['p90']:.0f} cap {cp.stint_cap.get(c)} laps"
                      f"  race deg {cp.rate_prior.get(c, {}).get('raw_mean_s_per_lap', float('nan')):.3f} s/lap"
                      f" x season {cp.season.get('factor', 1):.2f} ({cp.season.get('n_circuits', 0)} shared circuits)")
        th = cp.thermal
        if th.get("track_temp_now") is not None and th.get("track_temp_hist") is not None:
            print(f"    track {th['track_temp_now']:.0f}°C this weekend vs {th['track_temp_hist']:.0f}°C in those races: "
                  f"degradation x{th['multiplier']:.2f} ({th['beta_per_c']:+.3f}/°C)")
        if not cp.soft_race_tyre:
            print("    the SOFT has not been a race tyre here")
    else:
        print(f"  circuit history: none for {ev.circuit}")
    pit_loss, pit_src = pit_loss_prior(ev)
    if cp.available and cp.pit_loss_s:
        pit_loss, pit_src = float(cp.pit_loss_s), f"this pit lane, {cp.years}"
    if args.pit_loss:
        pit_loss, pit_src = float(args.pit_loss), "override"
    print(f"  pit loss prior: {pit_loss:.1f} s ({pit_src})")
    clean = add_fuel_correction(clean, ev, "2026")
    evo = fit_evolution_auto(clean, laps_all=laps, event=ev)
    clean = add_evolution_correction(clean, evo)
    evo_rng = evo.iterations[-1]["evo_range_s"] if evo.iterations else 0.0
    _it = evo.iterations[-1] if evo.iterations else {}
    gate("track evolution identified & plausible (0-5 s)", 0.0 < evo_rng < 5.0,
         f"push-lap range {evo_rng:.2f}s per session {_it.get('per_session', {})}; long-run backfit "
         f"would have given {_it.get('backfit_range_s', float('nan')):.2f}s; sessions on the backfit: {evo.skipped}")

    # -- 3. baseline -----------------------------------------------------------
    t = step("3. MixedLM baseline")
    mlm = fit_mixedlm(clean, n_boot=args.boot)
    print(mlm.table().round(4).to_string(index=False))

    # -- 4. apex ---------------------------------------------------------------
    apex_use, apex_sel = None, None
    if not args.no_apex and not args.quick:
        t = step("4. apex speeds")
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
    t = step("5. hierarchical fit")
    kw = dict(pace_step_s=pstep["step_s"])
    if args.quick:
        kw.update(chains=2, warmup=800, draws=800)
    f = fit_bayes(clean, ev, prior="2026", apex=apex_use, **kw)
    print(f.slope_table().round(4).to_string(index=False))
    gate("convergence: r_hat < 1.01 and zero divergences",
         f.max_rhat < RHAT_GATE and f.n_divergences == 0,
         f"max r_hat {f.max_rhat:.4f}, {f.n_divergences} divergences ({time.time()-t:.0f}s)")
    f_practice = f
    f_practice.save(DATA_PROCESSED / f"posterior_{key}_practice.npz")
    moved = pd.DataFrame()
    if cp.available and cp.rate_prior:
        f, moved = apply_circuit_prior(f_practice, ev, regime, cp)
        print("  practice posterior combined with circuit history (practice-regime rate, s/lap):")
        print(moved.round(4).to_string(index=False))
    post_path = f.save(DATA_PROCESSED / f"posterior_{key}.npz")
    print(f"  posterior saved: {post_path.name}")

    # -- 6. seal -------------------------------------------------------------
    step("6. seal")
    sealed_path, sha = seal_predictions(f, ev, regime=regime,
                                        note=f"weekend model on {used}; no race data read")
    sealed = load_sealed(sealed_path)
    print(f"  {sealed_path.name} sha256 {sha[:24]}...")

    # -- 7. pre-race plan ----------------------------------------------------
    t = step("7. pre-race strategy")
    per_comp_support = clean.groupby("compound")["tyre_age"].max().to_dict()
    caps = stint_caps_for(ev, cp) if cp.available else None
    if caps:
        print(f"  stint caps from this circuit's history: {caps}")
    res = strat.simulate(f, ev, pit_loss, regime=regime, n_draws=args.mc_draws,
                         support=per_comp_support, max_per_compound=alloc["caps"],
                         max_stint=caps)
    if res.table.empty:
        print("  no legal plan"); pw = pd.DataFrame(); uc = pd.DataFrame()
    else:
        print(res.head(6).drop(columns=["pit_laps"]).round(2).to_string(index=False))
        print(res.by_stops.drop(columns=["pit_laps"]).round(2).to_string(index=False))
        print(res.life.round(2).to_string(index=False))
        pw = strat.pit_window(f, ev, res.best, pit_loss, regime=regime, max_stint=res.max_stint,
                              push=res.best["push"])
        if cp.available:
            lens = dict(zip(res.best["compounds"], res.best["stint_lens"]))
            ok = all(L <= cp.stint_cap.get(c, 10 ** 6) for c, L in zip(res.best["compounds"], res.best["stint_lens"]))
            gate("every recommended stint is within what this circuit has supported", ok,
                 "; ".join(f"{c} {L} laps (cap {cp.stint_cap.get(c, '—')})" for c, L in zip(res.best["compounds"], res.best["stint_lens"])))
            modal = max(cp.stops, key=cp.stops.get) if cp.stops else None
            gate("recommended stop count matches the circuit's usual", modal is None or res.best["n_stops"] == modal,
                 f"{res.best['n_stops']} stops; history {cp.stops}")
        uc_age = int(min(res.max_stint.get("MEDIUM", 40), res.max_stint.get("SOFT", 40)))
        uc = strat.undercut_window(f, "MEDIUM", "SOFT", event=ev, regime=regime, max_age=uc_age,
                                   push=res.best["push"])
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
    pd.concat([pd.DataFrame({"variant": "2026", "compound": comp, "knee": f.posterior["knee"][:, i]})
               for i, comp in enumerate(f.compounds)]).to_parquet(DATA_PROCESSED / f"knee_{key}.parquet", index=False)
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
                            "derivation": pstep.get("derivation", ""),
                            "table": compound_table(ev, f.compounds, pace_step_s=pstep["step_s"]).to_dict("records"),
                            "fitted_offsets": f.comp_offset},
        "regime": regime.as_dict(),
        "circuit_history": (cp.as_dict() if cp.available else {}),
        "history_combination": moved.to_dict("records") if not moved.empty else [],
        "allocation": alloc,
        "pit_loss_s": float(pit_loss), "pit_loss_source": pit_src, "pit_stops_measured": 0,
        "load_effect": {"exponent": float(TYRE_LOAD_EXPONENT)},
        "n_raw_laps": int(len(laps)), "n_clean_laps": int(len(clean)),
        "compound_counts": comp_sum.to_dict("records"),
        "mixedlm": {"slopes": mlm.slopes, "table": mlm.table().to_dict("records"), "n_stints": mlm.n_stints},
        "bayes": {"max_rhat": float(f.max_rhat), "n_divergences": int(f.n_divergences),
                  "n_laps": int(f.n_laps), "n_apex": int(f.n_apex),
                  "corners": [int(x) for x in (apex_sel.corners if apex_sel else [])],
                  "slopes": f.slope_table().to_dict("records"),
                  "k_track_mean": float(f.k_track.mean()), "k_track_sd": float(f.k_track.std()),
                  "k_track_rel_sd": float(f.k_track.std() / f.k_track.mean()),
                  "comp_offset": f.comp_offset},
        "evolution": {"range_s": float(evo_rng), "iterations": evo.iterations, "skipped": evo.skipped},
        "age_support_by_compound": {k: float(v) for k, v in per_comp_support.items()},
        "max_stint_laps": (res.max_stint if not res.table.empty else {}),
        "strategy": ({
            "n_strategies": int(res.n_strategies), "n_scored": int(res.n_scored), "n_draws": int(res.n_draws),
            "best": res.best_label,
            "best_plan": {k: (list(v) if isinstance(v, list) else v) for k, v in res.best.items()},
            "push": float(res.best.get("push", float("nan"))), "implied_regime": float(res.implied_regime),
            "grip_budget_s": float(GRIP_BUDGET_S),
            "by_stops": res.by_stops.assign(pit_laps=res.by_stops["pit_laps"].astype(str),
                                            stint_lens=res.by_stops["stint_lens"].astype(str)).to_dict("records"),
            "life": res.life.to_dict("records"),
            "pit_windows": ([{"stop": int(k), "recommended": int(res.best["pit_laps"][int(k) - 1]),
                              "lo": int(g[g["in_window"]]["lap"].min()), "hi": int(g[g["in_window"]]["lap"].max())}
                             for k, g in pw.groupby("stop")] if not pw.empty else []),
        } if not res.table.empty else {}),
        "gates": GATES, "runtime_s": round(time.time() - t_all, 1),
        "written_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (DATA_PROCESSED / f"weekend_{key}.json").write_text(json.dumps(meta, indent=2, default=str))
    n_fail = sum(1 for g in GATES if not g["pass"])
    print(f"\n=== {len(GATES) - n_fail}/{len(GATES)} gates passed in {time.time() - t_all:.0f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
