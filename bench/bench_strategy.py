"""Strategy-intelligence benchmark: was the recommendation the right decision?

Per weekend, against the classified finishers of the real race:

  * stop count: recommended vs the field's mode and vs the winner
  * compound sequence: is the recommended sequence one anybody ran?  the
    winner's?  what share of the field ran it?  what did the top 5 run?
  * starting compound: matches the winner / the majority?
  * first stop, safety-car aware: the recommended lap, the tyre-optimal lap
    and V2's recommendation against the field's median *green-flag* first
    stop; the seconds of first-stop prior the chosen plan carries; share of
    the field's in-laps inside the model's window
  * the cliff detector: where the shipped grip budget says each compound
    collapses against where `cliff.race_collapses` says it did, and the
    collapse/strategic/undetermined split per compound
  * tyre life: predicted life per compound (bounded by the race and the
    circuit's history, as the tool now states it) and the model's own
    uncapped quotient, vs the longest and p90 stint actually run
  * oracle regret: every candidate plan re-priced on a tyre model whose
    degradation rates are the ones *measured on this race*, with the model's
    own grip budgets and pace offsets
  * per-car plans: how many drivers' own plans share the field plan's shape,
    and whether the per-driver first-stop spread tracks what those drivers did
  * counterfactual sanity: implausible "seconds lost" figures, and how many
    drivers' stops were held fixed for a safety car
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (OUT, arg_events, baseline_meta, baseline_processed, driver_plans, dump,  # noqa: E402
                    memoise_regime, meta, offline, race_results, race_table, v2_meta, v2_processed)
from src import strategy as strat
from src.calibration import get_calibration
from src.config import DATA_PROCESSED, get_event
from src.history import race_deg_slopes
from src.model_bayes import BayesFit
from src.tyre import TyreModel


def accuracy_percar(key: str) -> dict:
    """The three per-car Spearman variants, from `accuracy.json` if it has run.

    Stint ranking is an accuracy question measured on 28-43 stints a weekend;
    re-deriving it here would be the same arithmetic on the same curves, so the
    strategy report quotes it rather than recomputing it, and says so when the
    accuracy stage has not run yet.
    """
    p = OUT / "accuracy.json"
    if not p.exists():
        return {"source": "accuracy.json has not been written yet"}
    try:
        v = (json.loads(p.read_text()).get("per_event") or {}).get(key) or {}
    except Exception as exc:
        return {"source": f"accuracy.json unreadable: {exc}"}
    var = v.get("variants") or {}
    out = {"source": "bench/out/accuracy.json"}
    for name in ("sealed", "sealed_driver_hist", "sealed_driver_practice_dev", "sealed_driver_team_pooled"):
        if name in var:
            out[name] = {"spearman": var[name].get("spearman"), "rate_mae": var[name].get("rate_mae")}
    out["scales"] = v.get("percar_scales")
    return out


def cliff_rows(m: dict, key: str, ev) -> dict:
    """The cliff detector's race rows and the budget's predicted collapse lap.

    Read from `meta["cliff_detector"]` where the V3 decide stage wrote it; a V2
    meta has no such block, and `cliff.race_collapses` is then run here on the
    race lap table so the metric exists for both builds (and so the numbers can
    be compared with the ones the pipeline recorded).
    """
    block = dict(m.get("cliff_detector") or {})
    rows = block.get("rows")
    source = "meta[cliff_detector]"
    if not rows:
        try:
            from src.cliff import race_collapses
            rows = race_collapses(race_table(key), event=ev).to_dict("records")
            source = "bench: cliff.race_collapses on the race lap table"
        except Exception as exc:
            return {"source": f"unavailable: {exc}"}
    df = pd.DataFrame(rows or [])
    if df.empty:
        return {"source": source, "n": 0}
    by_comp = {}
    for c, g in df.groupby("compound"):
        kinds = g["kind"].value_counts().to_dict()
        coll = g[g["collapse"].fillna(False).astype(bool)] if "collapse" in g else g.iloc[0:0]
        by_comp[str(c)] = {"n_stints": int(len(g)),
                           "kinds": {str(k): int(v) for k, v in kinds.items()},
                           "n_collapse": int(len(coll)),
                           "observed_knee_mean": (float(pd.to_numeric(coll["knee_age"], errors="coerce").mean())
                                                  if len(coll) else None),
                           "observed_knees": [float(x) for x in pd.to_numeric(coll.get("knee_age"), errors="coerce").dropna()]
                           if len(coll) else [],
                           "cum_loss_at_knee_mean_s": (float(pd.to_numeric(coll["cum_loss_at_knee_s"], errors="coerce").mean())
                                                       if len(coll) else None)}
    out = {"source": source, "n": int(len(df)),
           "kinds": {str(k): int(v) for k, v in df["kind"].value_counts().items()},
           "n_collapse": int(df["collapse"].fillna(False).astype(bool).sum()) if "collapse" in df else 0,
           "by_compound": by_comp}
    # predicted collapse lap = budget / the model's full-push rate, per compound.
    # The decide stage writes it as `budget_vs_observed`; `budget_ratio_metrics`
    # is the function's own name and is accepted too.
    ratios = block.get("budget_vs_observed") or block.get("budget_ratio_metrics")
    if not ratios:
        try:
            from src.cliff import budget_ratio_metrics
            budgets = (m["strategy"].get("grip_budgets") or {})
            life = {x["compound"]: x for x in m["strategy"]["life"]}
            rates = {c: life.get(c, {}).get("deg_s_per_lap") for c in budgets}
            rates = {c: v for c, v in rates.items() if v}
            if budgets and rates:
                ratios = budget_ratio_metrics(budgets, rates, df)
                out["budget_ratio_source"] = "bench: cliff.budget_ratio_metrics on the shipped budgets and life rates"
        except Exception as exc:
            out["budget_ratio_source"] = f"unavailable: {exc}"
    else:
        out["budget_ratio_source"] = "meta[cliff_detector][budget_vs_observed]"
    out["budget_ratio_metrics"] = ratios
    if ratios:
        pooled = ratios.get("_pooled") or {}
        out["over"] = pooled.get("over")
        out["under"] = pooled.get("under")
        out["mean_abs_error_laps"] = pooled.get("mean_abs_error_laps")
    return out


def oracle_model(key: str, m: dict, n: int = 200, seed: int = 0) -> tuple:
    ev = get_event(key)
    r = race_table(key)
    r = r[r["is_accurate"] & ~r["pit_in"] & ~r["pit_out"] & (r["track_status"].astype(str) == "1")]
    d = race_deg_slopes(r, ev.fuel_effect_s_per_lap)
    rng = np.random.default_rng(seed)
    budgets = (m["strategy"].get("grip_budgets") or {})
    pace = (m.get("pace_calibration") or {}).get("offsets_after") or m["bayes"]["comp_offset"]
    comps = [c for c in m["bayes"]["comp_offset"] if c in d]
    wear, po = {}, {}
    for c in comps:
        s = max(d[c]["slope"], 0.005)
        draws = np.exp(rng.normal(np.log(s), max(d[c]["se"], 0.005) / s, size=n))
        wear[c] = draws / float(budgets.get(c, m["strategy"]["grip_budget_s"]))
        po[c] = np.full(n, float(pace.get(c, 0.0)))
    return TyreModel(compounds=comps, wear_rate=wear, pace_offset=po, budget=float(m["strategy"]["grip_budget_s"]),
                     budgets={c: float(budgets.get(c, m["strategy"]["grip_budget_s"])) for c in comps},
                     n_draws=n, source="race-measured (oracle)"), d


def window_rows(pw: list, same: pd.DataFrame, best: dict) -> list:
    rows = []
    for w in pw:
        k = w["stop"] - 1
        cand = same[[len(p) > k for p in same["pit_laps"]]]
        # green-flag stops only for the first stop: a safety-car stop was not a timing decision
        if k == 0:
            cand = cand[~cand["first_sc"]]
        actual = [p[k] for p in cand["pit_laps"]]
        if not actual:
            continue
        inside = np.mean([(w["lo"] <= a <= w["hi"]) for a in actual])
        rows.append({"stop": w["stop"], "recommended": w["recommended"], "lo": w["lo"], "hi": w["hi"],
                     "n_actual": len(actual), "share_inside": float(inside),
                     "median_abs_err": float(np.median([abs(a - w["recommended"]) for a in actual])),
                     "median_err": float(np.median([w["recommended"] - a for a in actual])),
                     "actual_median": float(np.median(actual))})
    return rows


def _first(plan: dict | None) -> int | None:
    p = (plan or {}).get("pit_laps") or []
    return int(p[0]) if p else None


def main() -> None:
    args = arg_events(__doc__)
    offline()
    memoise_regime()
    out = {}
    rows = []
    for key in args.events:
        ev = get_event(key)
        m = meta(key)
        bm = baseline_meta(key)          # V1
        v2m = v2_meta(key)               # V2
        cal = get_calibration(ev)
        race = race_table(key)
        plans = driver_plans(race, ev.n_race_laps)
        res = race_results(key)
        if res is not None:
            plans = plans.merge(res, on="driver", how="left").sort_values("position")
        st = m["strategy"]
        best = st["best_plan"]
        tyre_opt = st.get("tyre_optimal") or {}
        rec_seq = "-".join(best["compounds"])
        n_cls = len(plans)
        winner = plans.iloc[0] if res is not None and len(plans) else None
        top5 = plans.head(5) if res is not None else plans.iloc[0:0]

        stop_counts = plans["n_stops"].value_counts().to_dict()
        mode_stops = int(plans["n_stops"].mode().iloc[0]) if n_cls else None
        seq_counts = plans["seq"].value_counts()
        modal_seq = seq_counts.index[0] if len(seq_counts) else None
        start_counts = plans["seq"].str.split("-").str[0].value_counts()
        green_first = [p[0] for p, s in zip(plans["pit_laps"], plans["first_sc"]) if p and not s]
        sc_share_first = float(plans["first_sc"].mean()) if n_cls else None
        fmed = float(np.median(green_first)) if green_first else None

        r = {
            "recommended": st["best"], "rec_stops": int(best["n_stops"]), "rec_seq": rec_seq,
            "rec_push": float(best.get("push", np.nan)),
            "tyre_optimal": tyre_opt.get("label"), "tyre_optimal_delta_s": tyre_opt.get("delta_s"),
            "position_s": st.get("position_s"), "prior_s": st.get("prior_s"),
            "n_classified": n_cls, "field_stop_counts": {int(k): int(v) for k, v in stop_counts.items()},
            "mode_stops": mode_stops, "stops_match_mode": bool(best["n_stops"] == mode_stops),
            "stops_share": float(stop_counts.get(best["n_stops"], 0) / max(n_cls, 1)),
            "field_modal_seq": modal_seq, "field_seq_counts": {k: int(v) for k, v in seq_counts.head(6).items()},
            "rec_seq_share": float(seq_counts.get(rec_seq, 0) / max(n_cls, 1)),
            "rec_seq_run_by_anyone": bool(seq_counts.get(rec_seq, 0) > 0),
            "start_compound": best["compounds"][0],
            "field_start_counts": {k: int(v) for k, v in start_counts.items()},
            "start_matches_majority": bool(start_counts.index[0] == best["compounds"][0]) if len(start_counts) else None,
            "winner": (winner["driver"] if winner is not None else None),
            "winner_seq": (winner["seq"] if winner is not None else None),
            "winner_stops": (int(winner["n_stops"]) if winner is not None else None),
            "winner_pits": (list(winner["pit_laps"]) if winner is not None else None),
            "stops_match_winner": (bool(best["n_stops"] == winner["n_stops"]) if winner is not None else None),
            "seq_match_winner": (bool(rec_seq == winner["seq"]) if winner is not None else None),
            "top5_seqs": (top5["seq"].tolist() if len(top5) else []),
            "seq_in_top5": bool(rec_seq in set(top5["seq"])) if len(top5) else None,
            "first_stop": {"field_median_green": fmed, "share_under_sc": sc_share_first,
                           "sc_set": bool(sc_share_first is not None and sc_share_first > 0.4),
                           "recommended": _first(best),
                           "tyre_optimal": _first(tyre_opt),
                           "rec_minus_field": ((_first(best) - fmed) if (fmed is not None and best["pit_laps"]) else None),
                           "tyre_minus_field": ((_first(tyre_opt) - fmed) if (fmed is not None and tyre_opt.get("pit_laps")) else None),
                           # V1
                           "baseline_rec": (_first(bm["strategy"]["best_plan"]) if bm else None),
                           "baseline_minus_field": ((_first(bm["strategy"]["best_plan"]) - fmed)
                                                    if (bm and fmed is not None and bm["strategy"]["best_plan"]["pit_laps"]) else None),
                           # V2
                           "v2_rec": (_first(v2m["strategy"]["best_plan"]) if v2m else None),
                           "v2_minus_field": ((_first(v2m["strategy"]["best_plan"]) - fmed)
                                              if (v2m and fmed is not None and v2m["strategy"]["best_plan"]["pit_laps"]) else None),
                           # V3's new term: the seconds of first-stop prior the chosen plan pays,
                           # and the prior's own shape.  Absent on a V2 meta (kappa = 0 there).
                           "first_stop_s": (best.get("first_stop_s") if best.get("first_stop_s") is not None
                                            else st.get("first_stop_s")),
                           "first_stop_kappa_s": (st.get("first_stop_kappa_s")
                                                  if st.get("first_stop_kappa_s") is not None
                                                  else getattr(cal, "first_stop_kappa_s", None)),
                           "prior": m.get("first_stop_prior"),
                           "prior_from_history": ((m.get("circuit_history") or {}).get("first_stop_green") or {}).get("median_lap")},
            "baseline": ({"recommended": bm["strategy"]["best"], "seq_share": float(seq_counts.get("-".join(bm["strategy"]["best_plan"]["compounds"]), 0) / max(n_cls, 1)),
                          "start_matches_majority": bool(start_counts.index[0] == bm["strategy"]["best_plan"]["compounds"][0]) if len(start_counts) else None,
                          "stops_match_mode": bool(bm["strategy"]["best_plan"]["n_stops"] == mode_stops)} if bm else None),
            "v2": ({"recommended": v2m["strategy"]["best"],
                    "seq_share": float(seq_counts.get("-".join(v2m["strategy"]["best_plan"]["compounds"]), 0) / max(n_cls, 1)),
                    "seq_run_by_anyone": bool(seq_counts.get("-".join(v2m["strategy"]["best_plan"]["compounds"]), 0) > 0),
                    "start_matches_majority": bool(start_counts.index[0] == v2m["strategy"]["best_plan"]["compounds"][0]) if len(start_counts) else None,
                    "stops_match_mode": bool(v2m["strategy"]["best_plan"]["n_stops"] == mode_stops)} if v2m else None),
        }

        # -- pit windows vs actual in-laps of drivers on the same stop count --
        same = plans[plans["n_stops"] == best["n_stops"]]
        r["pit_windows"] = window_rows(st.get("pit_windows", []), same, best)

        # -- tyre life vs the longest stint actually run -----------------------
        life = {x["compound"]: x for x in st["life"]}
        lens = []
        for _, p in plans.iterrows():
            for c, L in zip(p["seq"].split("-"), p["stint_lens"]):
                lens.append({"compound": c, "L": int(L)})
        lens = pd.DataFrame(lens)
        life_rows = []
        for c, g in lens.groupby("compound"):
            lf = life.get(c, {})
            blf = ({x["compound"]: x for x in bm["strategy"]["life"]}.get(c, {}) if bm else {})
            vlf = ({x["compound"]: x for x in v2m["strategy"]["life"]}.get(c, {}) if v2m else {})
            life_rows.append({"compound": c, "pred_life_at_push": lf.get("life_laps"),
                              "pred_life_uncapped": lf.get("life_model_uncapped"), "bound_by": lf.get("bound_by"),
                              "longer_than_race": lf.get("longer_than_race"),
                              "pred_knee_full_push": lf.get("knee_lap"),
                              "grip_budget_s": lf.get("grip_budget_s"), "deg_s_per_lap": lf.get("deg_s_per_lap"),
                              "baseline_life": blf.get("life_laps"), "v2_life": vlf.get("life_laps"),
                              "v2_bound_by": vlf.get("bound_by"), "v2_grip_budget_s": vlf.get("grip_budget_s"),
                              "obs_max": int(g["L"].max()), "obs_p90": float(g["L"].quantile(0.9)),
                              "obs_median": float(g["L"].median()), "n_stints": int(len(g)),
                              "ratio_to_max": (float(lf["life_laps"] / g["L"].max()) if lf.get("life_laps") else None),
                              "baseline_ratio_to_max": (float(blf["life_laps"] / g["L"].max()) if blf.get("life_laps") else None),
                              "v2_ratio_to_max": (float(vlf["life_laps"] / g["L"].max()) if vlf.get("life_laps") else None)})
        r["life"] = life_rows

        # -- oracle regret --------------------------------------------------------
        orc, rates = oracle_model(key, m)
        alloc = m["allocation"]["caps"]
        caps = {c: int(v) for c, v in (m.get("circuit_history") or {}).get("stint_cap", {}).items()} or None
        pit_loss = float(m["pit_loss_s"])
        sim = strat.simulate_model(orc, ev, pit_loss, push_grid=(1.0,), max_per_compound=alloc,
                                   max_stint=caps, shortlist=2000)
        cand = [{"label": "tool", "compounds": best["compounds"], "pit_laps": best["pit_laps"], "push": 1.0}]
        if tyre_opt.get("pit_laps"):
            cand.append({"label": "tyre_optimal", "compounds": tyre_opt["compounds"], "pit_laps": tyre_opt["pit_laps"], "push": 1.0})
        if bm:
            cand.append({"label": "baseline_tool", "compounds": bm["strategy"]["best_plan"]["compounds"],
                         "pit_laps": bm["strategy"]["best_plan"]["pit_laps"], "push": 1.0})
        if v2m:
            cand.append({"label": "v2_tool", "compounds": v2m["strategy"]["best_plan"]["compounds"],
                         "pit_laps": v2m["strategy"]["best_plan"]["pit_laps"], "push": 1.0})
        if modal_seq is not None:
            ms = plans[plans["seq"] == modal_seq]
            k = int(ms["n_stops"].iloc[0])
            med_pits = [int(np.median([p[i] for p in ms["pit_laps"]])) for i in range(k)]
            cand.append({"label": "field_modal", "compounds": modal_seq.split("-"), "pit_laps": med_pits, "push": 1.0})
        if winner is not None:
            cand.append({"label": "winner", "compounds": winner["seq"].split("-"), "pit_laps": list(winner["pit_laps"]), "push": 1.0})
        if not sim.table.empty:
            cand.append({"label": "oracle_opt", "compounds": sim.best["compounds"], "pit_laps": sim.best["pit_laps"], "push": 1.0})
        cand = [c for c in cand if all(x in orc.compounds for x in c["compounds"])]
        tbl, det = strat.evaluate_plans(orc, ev, cand, pit_loss, push=1.0, allocation=alloc, stint_cap=caps or {})
        costs = {d["label"]: float(d["times"].mean()) for d in det if d.get("valid")}
        ref = costs.get("oracle_opt", min(costs.values()))
        r["oracle"] = {"rates": {c: round(v["slope"], 4) for c, v in rates.items()},
                       "oracle_best": sim.best_label if not sim.table.empty else None,
                       "costs_s": costs, "regret_s": {k: round(v - ref, 2) for k, v in costs.items()},
                       "tool_beats_field_modal": (costs.get("tool", np.inf) < costs.get("field_modal", np.inf)),
                       "tool_beats_winner": (costs.get("tool", np.inf) < costs.get("winner", np.inf))}
        # -- per-car plans ----------------------------------------------------
        pd_rows = m.get("per_driver") or []
        if pd_rows:
            pdf = pd.DataFrame(pd_rows).merge(plans[["driver", "n_stops", "seq", "pit_laps", "first_sc"]].rename(
                columns={"n_stops": "actual_stops", "seq": "actual_seq", "pit_laps": "actual_pits"}), on="driver", how="inner")
            pdf["first_actual"] = [p[0] if p else np.nan for p in pdf["actual_pits"]]
            same_shape = float(pdf["same_shape_as_field"].mean()) if len(pdf) else None
            stops_ok = float((pdf["n_stops"] == pdf["actual_stops"]).mean()) if len(pdf) else None
            seq_ok = float((pdf["compounds"] == pdf["actual_seq"]).mean()) if len(pdf) else None
            g = pdf[(~pdf["first_sc"]) & pdf["first_stop"].notna() & pdf["first_actual"].notna()]
            from scipy.stats import spearmanr
            rho = float(spearmanr(g["first_stop"], g["first_actual"]).correlation) if len(g) >= 5 and g["first_stop"].std() > 0 else None
            r["per_driver"] = {"n": int(len(pdf)), "share_same_shape_as_field": same_shape, "stops_match_actual": stops_ok,
                               "seq_match_actual": seq_ok, "first_stop_spearman_vs_actual": rho,
                               "first_stop_spread": (float(pdf["first_stop"].max() - pdf["first_stop"].min()) if pdf["first_stop"].notna().any() else None),
                               "race_factor_range": [float(pdf["race_factor"].min()), float(pdf["race_factor"].max())],
                               "mode": getattr(cal, "percar_mode", None)}
        else:
            r["per_driver"] = {}
        # the three per-car rate-scale variants, ranked against the race's own stint rates
        r["per_driver_variants"] = accuracy_percar(key)
        # -- the cliff detector ------------------------------------------------
        r["cliff_detector"] = cliff_rows(m, key, ev)
        # -- counterfactual sanity ------------------------------------------
        cf = pd.read_parquet(DATA_PROCESSED / f"counterfactual_{key}.parquet")

        def _over30(p):
            return int((pd.read_parquet(p)["loss_s"] > 30).sum()) if p.exists() else None

        r["counterfactual"] = {"n": int(len(cf)), "n_over_30s": int((cf["loss_s"] > 30).sum()),
                               "n_with_sc_stops": int((cf["n_sc_stops"] > 0).sum()) if "n_sc_stops" in cf else None,
                               "n_over_30s_classified": int(((cf["loss_s"] > 30) & cf["classified"]).sum()) if "classified" in cf else None,
                               "median_loss_s": float(cf["loss_s"].median()),
                               "top": cf.head(3)[["driver", "compounds", "loss_s"]].round(1).to_dict("records"),
                               "baseline_n_over_30s": _over30(baseline_processed(f"counterfactual_{key}.parquet")),
                               "v2_n_over_30s": _over30(v2_processed(f"counterfactual_{key}.parquet"))}
        r["cliff"] = m["score"]["cliff"]
        r["ladder_check"] = m.get("ladder_check")
        r["pace_calibration"] = {k: v for k, v in (m.get("pace_calibration") or {}).items() if k in ("applied", "measured_net_s", "model_net_before", "model_net_after", "clipped")}
        r["gates_failed"] = [g["gate"] for g in m["gates"] if not g["pass"]]
        r["calibration"] = {k: v for k, v in (m.get("calibration") or {}).items()
                            if k in ("grip_budget_by_compound", "undercut_lambda", "plan_prior_tau_s",
                                     "grid_start_penalty_s", "dirty_air_s_per_lap", "manage_cost_s",
                                     "manage_wear_floor",
                                     # V3: the first-stop weight, the circuit's own dirty air, the
                                     # censored budget estimate and which per-car term was applied
                                     "first_stop_kappa_s", "dirty_air_used", "dirty_air_source",
                                     "grip_budget_detail", "percar_mode", "source")}
        r["stint_fe_baseline"] = {k: v for k, v in (m.get("stint_fe_baseline") or {}).items() if k != "table"}
        r["bayes_vs_stint_fe_pooled"] = m.get("bayes_vs_stint_fe_pooled")
        out[key] = r
        win = r["pit_windows"]
        rows.append({"event": key, "recommended": st["best"], "tyre_optimal": tyre_opt.get("label"),
                     "baseline": (bm["strategy"]["best"] if bm else None),
                     "v2": (v2m["strategy"]["best"] if v2m else None),
                     "winner": r["winner_seq"], "winner_pits": r["winner_pits"],
                     "field_modal": modal_seq, "stops_match_mode": r["stops_match_mode"], "stops_share": round(r["stops_share"], 2),
                     "seq_share": round(r["rec_seq_share"], 2), "seq_in_top5": r["seq_in_top5"],
                     "start_ok": r["start_matches_majority"],
                     "first_rec_minus_field": r["first_stop"]["rec_minus_field"],
                     "first_tyre_minus_field": r["first_stop"]["tyre_minus_field"],
                     "first_baseline_minus_field": r["first_stop"]["baseline_minus_field"],
                     "first_v2_minus_field": r["first_stop"]["v2_minus_field"],
                     "first_stop_s": r["first_stop"]["first_stop_s"],
                     "sc_set_first": r["first_stop"]["sc_set"],
                     "regret_tool": r["oracle"]["regret_s"].get("tool"), "regret_tyre_opt": r["oracle"]["regret_s"].get("tyre_optimal"),
                     "regret_baseline": r["oracle"]["regret_s"].get("baseline_tool"),
                     "regret_v2": r["oracle"]["regret_s"].get("v2_tool"),
                     "regret_field": r["oracle"]["regret_s"].get("field_modal"), "regret_winner": r["oracle"]["regret_s"].get("winner"),
                     "win1_inside": (win[0]["share_inside"] if win else None), "win1_err": (win[0]["median_abs_err"] if win else None),
                     "collapse_n": (r["cliff_detector"] or {}).get("n_collapse"),
                     "collapse_abs_err_laps": (r["cliff_detector"] or {}).get("mean_abs_error_laps")})
        print(f"\n== {key}: tool {st['best']} | tyre-optimal {tyre_opt.get('label')} | V2 {v2m['strategy']['best'] if v2m else '-'} "
              f"| V1 {bm['strategy']['best'] if bm else '-'} "
              f"| winner {r['winner']} {r['winner_seq']} @ {r['winner_pits']} | modal {modal_seq}")
        print(f"   stops: rec {best['n_stops']} mode {mode_stops} ({r['stops_share']:.0%} of field); seq share {r['rec_seq_share']:.0%}; "
              f"in top5 {r['seq_in_top5']}; start {best['compounds'][0]} vs field {dict(start_counts)}")
        print(f"   first stop: field median (green) {fmed}, {sc_share_first:.0%} under SC; rec {r['first_stop']['recommended']} "
              f"tyre-opt {r['first_stop']['tyre_optimal']} V2 {r['first_stop']['v2_rec']} V1 {r['first_stop']['baseline_rec']}; "
              f"prior penalty paid {r['first_stop']['first_stop_s']} s (kappa {r['first_stop']['first_stop_kappa_s']}), "
              f"prior {(r['first_stop']['prior'] or {}).get('mode') if isinstance(r['first_stop']['prior'], dict) else None}")
        print(f"   oracle rates {r['oracle']['rates']}  regret (s): {r['oracle']['regret_s']}")
        print(f"   windows: {win}")
        print(f"   life: {[(x['compound'], x['pred_life_at_push'], x['bound_by'], x['obs_p90'], x['obs_max']) for x in life_rows]}")
        print(f"   per-driver: {r['per_driver']}")
        print(f"   per-driver variants: { {k: v for k, v in (r['per_driver_variants'] or {}).items() if k != 'scales'} }")
        print(f"   cliff detector: n {(r['cliff_detector'] or {}).get('n')} kinds {(r['cliff_detector'] or {}).get('kinds')} "
              f"collapses {(r['cliff_detector'] or {}).get('n_collapse')} "
              f"predicted-vs-observed |err| {(r['cliff_detector'] or {}).get('mean_abs_error_laps')} laps "
              f"(over {(r['cliff_detector'] or {}).get('over')} / under {(r['cliff_detector'] or {}).get('under')})")
        print(f"   counterfactual >30s: {r['counterfactual']['n_over_30s']}/{r['counterfactual']['n']} "
              f"(V2 {r['counterfactual']['v2_n_over_30s']}, V1 {r['counterfactual']['baseline_n_over_30s']}); "
              f"SC-held drivers {r['counterfactual']['n_with_sc_stops']}")
    dump("strategy.json", out)
    t = pd.DataFrame(rows)
    t.to_csv(dump("strategy_table.json", []).with_suffix(".csv"), index=False)
    print("\n" + t.to_string(index=False))


if __name__ == "__main__":
    main()
