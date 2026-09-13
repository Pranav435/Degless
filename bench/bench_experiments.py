"""The V4 experiment grid, E0-E8 (docs/v4_plan.md §3), on the pre-race decision.

Every pre-race variant is the same search `bench_ablation.search_variant` runs
(300 draws, 1-lap grid, the ladder gate enforced, the race state measured on
the other 2026 races), followed by the same three scorings `bench_strategy`
applies to the shipped plan:

  * the first stop against the field's green-flag median (the three
    non-safety-car weekends), signed and absolute;
  * the pit window (the sweep at the variant's own race-state term) and the
    share of the field's green first stops inside it;
  * the pure-time oracle regret and the position-aware regret `R_pos`
    (`docs/v4_methodology.md`), on the race-measured oracle tyre model;
  * the sequence / start / stop-count matches.

Nothing is tuned here: the variants are the plan's, fixed before the run, and
every constant they use is the leave-one-out calibration's.

  E0   Task 1 as frozen (`bench/v4_task1/out/strategy.json`; not re-run)
  E1   heterogeneous rivals, otherwise Task 1 (its estimator, no tyre-life widening)
  E1b  heterogeneous rivals with no historical strategy prior at all
  E1c  the symmetric pack on the final code
  E2   the place-value estimators: task1 / lead_lap / regularized (the shipped one)
  E3   the reduced feature set: no family logit (the rivals' plan mix from history
       alone), one degradation level
  E5   the tyre-life widening at 0 / half / twice the measured width
  E8   the full V4 objective as the pipeline ships it

E4 (the Haas model), E6 (the objective unification) and E7 (the live engine)
are scored by their own scripts (`bench_accuracy`, the pipeline's counterfactual
and `bench_live`/`bench_wpd_live`); `bench_v4_compare` collects them.

Usage: python bench/bench_experiments.py [--events ...] [--variants E1 E2 ...]
Writes bench/out/experiments.json (rows per variant per weekend, plus pooled).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from common import (NON_SC_EVENTS, OUT, ROOT, arg_events, cp_for, dirty_air_of, driver_plans, dump,  # noqa: E402
                    first_stop_tables, memoise_regime, meta, offline, race_results, race_table)
import bench_ablation as BA  # noqa: E402
import bench_strategy as BS  # noqa: E402
from src import objective, racestate, strategy as strat
from src.calibration import get_calibration
from src.config import DATA_PROCESSED, get_event
from src.model_bayes import BayesFit
from src.tyre import EXTRAP_LN_SD_MEASURED

T1 = ROOT / "bench" / "v4_task1" / "out" / "strategy.json"


def _rf(cal, **over):
    return objective.rival_field_default(cal, **over)


T1_CAL_PATH = ROOT / "bench" / "v4_task1" / "processed" / "calibration.json"
T1_CAL = {k: v for k, v in ((json.loads(T1_CAL_PATH.read_text()).get("loo") or {}).items())} if T1_CAL_PATH.exists() else {}


def variants(cal, cal_key: str | None = None) -> dict:
    """Name -> the keywords of `bench_ablation.search_variant` that define it.

    `cal_key` is the weekend whose leave-one-out block `cal` is (for E6b)."""
    sym = racestate.RivalFieldConfig(mode="symmetric")
    return {
        # E1: the rival field alone, everything else as Task 1 ran it
        "E1_hetero_rivals": dict(rival_field=_rf(cal, mode="hetero"), estimator="task1", extrap=False),
        "E1b_hetero_no_history": dict(rival_field=_rf(cal, mode="hetero", use_history_prior=False,
                                                       family_prior_weight_k0=float("inf")),
                                      estimator="task1", extrap=False),
        "E1c_symmetric_final": dict(rival_field=sym),
        # E2: the estimators, on the shipped rival field
        "E2_place_value_task1": dict(estimator="task1"),
        "E2_place_value_lead_lap": dict(estimator="lead_lap"),
        "E2_place_value_regularized": dict(estimator="regularized"),
        # E3: demoted features
        "E3_no_family_logit": dict(rival_field=_rf(cal, mode="hetero", family_temper_s=200.0)),
        "E3_rate_levels_1": dict(rival_field=_rf(cal, rate_levels=1)),
        # E5: the tyre-life widening
        "E5_extrap_0": dict(extrap=False),
        "E5_extrap_half": dict(extrap_ln_sd=0.5 * EXTRAP_LN_SD_MEASURED),
        "E5_extrap_double": dict(extrap_ln_sd=2.0 * EXTRAP_LN_SD_MEASURED),
        # E8: the shipped objective
        "E8_full": dict(),
        # E6b (diagnostic): the V4 objective priced with Task 1's per-fold
        # lambda and tau (bench/v4_task1/processed/calibration.json, V3's sweeps).
        # Not a shipped variant: it shows what the V4 recalibration's own
        # constants cost where the donors do not identify them.
        "E6b_task1_constants": dict(lam=T1_CAL.get(cal_key, {}).get("undercut_lambda"),
                                    tau=T1_CAL.get(cal_key, {}).get("plan_prior_tau_s")),
    }


def score(key: str, ev, m: dict, res, model, kw: dict, cal, plans: pd.DataFrame, orc, oracle_opt: dict,
          winner, modal_seq, med_pits, green_first: list) -> dict:
    """The strategy benchmark's scorings, applied to one variant's result."""
    best = res.best
    n_cls = len(plans)
    stop_counts = plans["n_stops"].value_counts().to_dict()
    mode_stops = int(plans["n_stops"].mode().iloc[0]) if n_cls else None
    seq_counts = plans["seq"].value_counts()
    start_counts = plans["seq"].str.split("-").str[0].value_counts()
    rec_seq = "-".join(best["compounds"])
    fmed = float(np.median(green_first)) if green_first else None
    first = int(best["pit_laps"][0]) if best["pit_laps"] else None
    sc_set = bool(plans["first_sc"].mean() > 0.4) if n_cls else True
    # the pit window at the variant's own race-state term
    rs_term = racestate.term_by_lap((res.race_state or {}).get("best"), ev.n_race_laps)
    pw = strat.pit_window_model(model, ev, best, float(m["pit_loss_s"]), max_stint=res.max_stint,
                                push=float(best["push"]), undercut_lambda=float(kw.get("undercut_lambda", 0.0) or 0.0),
                                traffic_s_per_lap=float(kw.get("traffic_s_per_lap", cal.dirty_air_for(ev.circuit))),
                                race_state_term=rs_term)
    windows = ([{"stop": int(k), "recommended": int(best["pit_laps"][int(k) - 1]),
                 "lo": int(g[g["in_window"]]["lap"].min()), "hi": int(g[g["in_window"]]["lap"].max())}
                for k, g in pw.groupby("stop")] if not pw.empty else [])
    same = plans[plans["n_stops"] == best["n_stops"]]
    win = BS.window_rows(windows, same, best)
    # the oracle: pure time and position-aware, on the same candidate list as bench_strategy
    alloc = m["allocation"]["caps"]
    caps = {c: int(v) for c, v in (m.get("circuit_history") or {}).get("stint_cap", {}).items()} or None
    pit = float(m["pit_loss_s"])
    cand = [{"label": "tool", "compounds": best["compounds"], "pit_laps": best["pit_laps"], "push": 1.0}]
    if res.tyre_optimal.get("pit_laps"):
        cand.append({"label": "tyre_optimal", "compounds": res.tyre_optimal["compounds"],
                     "pit_laps": res.tyre_optimal["pit_laps"], "push": 1.0})
    if modal_seq is not None:
        cand.append({"label": "field_modal", "compounds": modal_seq.split("-"), "pit_laps": med_pits, "push": 1.0})
    if winner is not None:
        cand.append({"label": "winner", "compounds": winner["seq"].split("-"), "pit_laps": list(winner["pit_laps"]),
                     "push": 1.0})
    if oracle_opt:
        cand.append({"label": "oracle_opt", **oracle_opt, "push": 1.0})
    cand = [c for c in cand if all(x in orc.compounds for x in c["compounds"])]
    _, det = strat.evaluate_plans(orc, ev, cand, pit, push=1.0, allocation=alloc, stint_cap=caps or {})
    costs = {d["label"]: float(d["times"].mean()) for d in det if d.get("valid")}
    ref = costs.get("oracle_opt", min(costs.values())) if costs else float("nan")
    regret = {k: float(v - ref) for k, v in costs.items()}
    pos, _ = BS.position_block(key, ev, orc, cand, plans, pit, alloc, caps, green_first)
    return {"plan": res.best_label, "n_stops": int(best["n_stops"]), "seq": rec_seq, "start": best["compounds"][0],
            "first": first, "field_median_green": fmed, "sc_set": sc_set,
            "first_minus_field": ((first - fmed) if (first is not None and fmed is not None and not sc_set) else None),
            "window": (windows[0] if windows else None),
            "window_share": (win[0]["share_inside"] if win else None),
            "regret_tool_s": regret.get("tool"), "regret_tyre_optimal_s": regret.get("tyre_optimal"),
            "rpos_tool_s": (pos.get("R_pos") or {}).get("tool"),
            "rpos_tyre_optimal_s": (pos.get("R_pos") or {}).get("tyre_optimal"),
            "L_tool": (pos.get("L") or {}).get("tool"), "P_retain_tool": pos.get("P_retain_tool"),
            "seq_run_by_anyone": bool(seq_counts.get(rec_seq, 0) > 0),
            "seq_share": float(seq_counts.get(rec_seq, 0) / max(n_cls, 1)),
            "start_matches_majority": bool(len(start_counts) and start_counts.index[0] == best["compounds"][0]),
            "stops_match_mode": bool(best["n_stops"] == mode_stops),
            "race_state_mode": (res.race_state or {}).get("mode"),
            "race_state_s": float(best.get("race_state_s", 0.0)),
            "extrap_ln_sd": float(getattr(model, "extrap_ln_sd", 0.0))}


def task1_rows(events: list) -> dict:
    """E0: the frozen Task 1 scorings, read, not re-run."""
    if not T1.exists():
        return {}
    s = json.loads(T1.read_text(encoding="utf-8"))
    out = {}
    for k in events:
        r = s.get(k) or {}
        fs, win, orc = r.get("first_stop") or {}, (r.get("pit_windows") or [{}])[0] or {}, r.get("oracle") or {}
        out[k] = {"plan": r.get("recommended"), "n_stops": r.get("rec_stops"), "seq": r.get("rec_seq"),
                  "start": r.get("start_compound"), "first": fs.get("recommended"),
                  "field_median_green": fs.get("field_median_green"), "sc_set": fs.get("sc_set"),
                  "first_minus_field": (fs.get("rec_minus_field") if not fs.get("sc_set") else None),
                  "window": {"lo": win.get("lo"), "hi": win.get("hi")}, "window_share": win.get("share_inside"),
                  "regret_tool_s": (orc.get("regret_s") or {}).get("tool"),
                  "regret_tyre_optimal_s": (orc.get("regret_s") or {}).get("tyre_optimal"),
                  "rpos_tool_s": ((orc.get("position_aware") or {}).get("R_pos") or {}).get("tool"),
                  "seq_run_by_anyone": r.get("rec_seq_run_by_anyone"), "seq_share": r.get("rec_seq_share"),
                  "start_matches_majority": r.get("start_matches_majority"), "stops_match_mode": r.get("stops_match_mode"),
                  "race_state_mode": "symmetric", "source": "bench/v4_task1/out/strategy.json (frozen)"}
    return out


def pooled(rows: dict) -> dict:
    ks = list(rows)
    fs = [rows[k]["first_minus_field"] for k in ks if k in NON_SC_EVENTS and rows[k].get("first_minus_field") is not None]
    def mean(vals):
        v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
        return float(np.mean(v)) if v else None
    return {"n_events": len(ks),
            "first_stop_error_laps": mean([abs(x) for x in fs]), "first_stop_signed_laps": mean(fs),
            "first_stop_by_event": {k: rows[k].get("first_minus_field") for k in ks if k in NON_SC_EVENTS},
            "window_share": mean([rows[k].get("window_share") for k in ks]),
            "oracle_regret_s": mean([rows[k].get("regret_tool_s") for k in ks]),
            "rpos_tool_s": mean([rows[k].get("rpos_tool_s") for k in ks]),
            "L_tool": mean([rows[k].get("L_tool") for k in ks]),
            "seq_run_by_anyone": int(sum(bool(rows[k].get("seq_run_by_anyone")) for k in ks)),
            "start_matches_majority": int(sum(bool(rows[k].get("start_matches_majority")) for k in ks)),
            "stops_match_mode": int(sum(bool(rows[k].get("stops_match_mode")) for k in ks)),
            "plans": {k: rows[k].get("plan") for k in ks}}


def main() -> None:
    args = arg_events(__doc__, extra=[(("--variants",), {"nargs": "*", "default": None,
                                                         "help": "a subset of the variant names (prefix match)"})])
    offline()
    memoise_regime()
    out: dict = {"E0_task1": task1_rows(args.events)}
    t_all = time.time()
    per_key: dict = {}
    for key in args.events:
        ev = get_event(key)
        m = meta(key)
        cal = get_calibration(ev)
        race = race_table(key)
        plans = driver_plans(race, ev.n_race_laps)
        res_tab = race_results(key)
        if res_tab is not None:
            plans = plans.merge(res_tab, on="driver", how="left").sort_values("position")
        winner = plans.iloc[0] if res_tab is not None and len(plans) else None
        seq_counts = plans["seq"].value_counts()
        modal_seq = seq_counts.index[0] if len(seq_counts) else None
        med_pits = None
        if modal_seq is not None:
            ms = plans[plans["seq"] == modal_seq]
            k_ = int(ms["n_stops"].iloc[0])
            med_pits = [int(np.median([p[i] for p in ms["pit_laps"]])) for i in range(k_)]
        green_first = [p[0] for p, s in zip(plans["pit_laps"], plans["first_sc"]) if p and not s]
        f_ship = BayesFit.load(DATA_PROCESSED / f"posterior_{key}.npz")
        cp = cp_for(key, m)
        fs_tables = first_stop_tables(cp, ev, list(f_ship.compounds))
        orc, _ = BS.oracle_model(key, m)
        alloc = m["allocation"]["caps"]
        caps = {c: int(v) for c, v in (m.get("circuit_history") or {}).get("stint_cap", {}).items()} or None
        sim = strat.simulate_model(orc, ev, float(m["pit_loss_s"]), push_grid=(1.0,), max_per_compound=alloc,
                                   max_stint=caps, shortlist=2000)
        oracle_opt = ({"compounds": sim.best["compounds"], "pit_laps": sim.best["pit_laps"]}
                      if not sim.table.empty else None)
        per_key[key] = dict(ev=ev, m=m, cal=cal, plans=plans, winner=winner, modal_seq=modal_seq, med_pits=med_pits,
                            green_first=green_first, fit=f_ship, fs_tables=fs_tables, orc=orc, oracle_opt=oracle_opt)
    names = list(variants(get_calibration(get_event(args.events[0]))))
    if args.variants:
        names = [n for n in names if any(n.startswith(v) for v in args.variants)]
    for name in names:
        rows = {}
        t0 = time.time()
        for key in args.events:
            d = per_key[key]
            spec = variants(d["cal"], key)[name]
            try:
                res, model, kw = BA.search_variant(key, d["m"], d["fit"], d["cal"], fs_tables=d["fs_tables"], **spec)
                rows[key] = score(key, d["ev"], d["m"], res, model, kw, d["cal"], d["plans"], d["orc"],
                                  d["oracle_opt"], d["winner"], d["modal_seq"], d["med_pits"], d["green_first"])
            except Exception as exc:   # one weekend must not take the grid down
                rows[key] = {"error": str(exc)[:200]}
                print(f"  {name} {key}: {exc}", flush=True)
        pool = pooled({k: v for k, v in rows.items() if "error" not in v})
        # the pooled scalars sit at the top level too: `bench_v4_compare`
        # carries a variant's scalar fields and nothing nested
        out[name] = {**{k: v for k, v in pool.items() if not isinstance(v, dict)},
                     "by_event": rows, "pooled": pool,
                     "seconds": round(time.time() - t0, 1), "spec": {k: (v.as_dict() if hasattr(v, "as_dict") else v)
                                                                      for k, v in variants(get_calibration(get_event(args.events[0])))[name].items()}}
        p = out[name]["pooled"]
        print(f"{name:28s} first err {p['first_stop_error_laps']} (signed {p['first_stop_signed_laps']}) "
              f"window {p['window_share']} regret {p['oracle_regret_s']} rpos {p['rpos_tool_s']} "
              f"seq/start/stops {p['seq_run_by_anyone']}/{p['start_matches_majority']}/{p['stops_match_mode']} "
              f"[{out[name]['seconds']}s]", flush=True)
        dump("experiments.json", out)
    if out.get("E0_task1"):
        pool0 = pooled(out["E0_task1"])
        out["E0_task1"] = {**{k: v for k, v in pool0.items() if not isinstance(v, dict)},
                           "by_event": out["E0_task1"], "pooled": pool0}
    out["_note"] = ("pre-race variants at 300 draws on the same search bench_ablation runs; E0 is the frozen Task 1 "
                    "run at the pipeline's 500 draws; first-stop errors on the non-safety-car weekends "
                    f"{NON_SC_EVENTS}; R_pos per docs/v4_methodology.md")
    out["_seconds"] = round(time.time() - t_all, 1)
    dump("experiments.json", out)
    tbl = pd.DataFrame({k: v["pooled"] for k, v in out.items() if isinstance(v, dict) and "pooled" in v}).T
    print("\n" + tbl.drop(columns=[c for c in ("plans", "first_stop_by_event") if c in tbl]).to_string())


if __name__ == "__main__":
    main()
