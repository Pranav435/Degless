"""Strategy-intelligence benchmark: was the recommendation the right decision?

Per weekend, against the classified finishers of the real race:

  * stop count: recommended vs the field's mode and vs the winner
  * compound sequence: is the recommended sequence one anybody ran?  the
    winner's?  what share of the field ran it?  what did the top 5 run?
  * starting compound: matches the winner / the majority?
  * first stop, safety-car aware: the recommended lap and the tyre-optimal
    lap against the field's median *green-flag* first stop; share of the
    field's in-laps inside the model's window
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

from common import EVENTS, baseline_meta, driver_plans, dump, meta, offline, race_results, race_table  # noqa: E402
from src import strategy as strat
from src.calibration import get_calibration
from src.config import DATA_PROCESSED, get_event
from src.history import race_deg_slopes
from src.model_bayes import BayesFit
from src.tyre import TyreModel


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


def main() -> None:
    offline()
    out = {}
    rows = []
    for key in EVENTS:
        ev = get_event(key)
        m = meta(key)
        bm = baseline_meta(key)
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
                           "recommended": (int(best["pit_laps"][0]) if best["pit_laps"] else None),
                           "tyre_optimal": (int(tyre_opt["pit_laps"][0]) if tyre_opt.get("pit_laps") else None),
                           "rec_minus_field": ((int(best["pit_laps"][0]) - fmed) if (fmed is not None and best["pit_laps"]) else None),
                           "tyre_minus_field": ((int(tyre_opt["pit_laps"][0]) - fmed) if (fmed is not None and tyre_opt.get("pit_laps")) else None),
                           "baseline_rec": (int(bm["strategy"]["best_plan"]["pit_laps"][0]) if bm and bm["strategy"]["best_plan"]["pit_laps"] else None),
                           "baseline_minus_field": ((int(bm["strategy"]["best_plan"]["pit_laps"][0]) - fmed)
                                                    if (bm and fmed is not None and bm["strategy"]["best_plan"]["pit_laps"]) else None)},
            "baseline": ({"recommended": bm["strategy"]["best"], "seq_share": float(seq_counts.get("-".join(bm["strategy"]["best_plan"]["compounds"]), 0) / max(n_cls, 1)),
                          "start_matches_majority": bool(start_counts.index[0] == bm["strategy"]["best_plan"]["compounds"][0]) if len(start_counts) else None,
                          "stops_match_mode": bool(bm["strategy"]["best_plan"]["n_stops"] == mode_stops)} if bm else None),
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
            life_rows.append({"compound": c, "pred_life_at_push": lf.get("life_laps"),
                              "pred_life_uncapped": lf.get("life_model_uncapped"), "bound_by": lf.get("bound_by"),
                              "longer_than_race": lf.get("longer_than_race"),
                              "pred_knee_full_push": lf.get("knee_lap"), "baseline_life": blf.get("life_laps"),
                              "obs_max": int(g["L"].max()), "obs_p90": float(g["L"].quantile(0.9)),
                              "obs_median": float(g["L"].median()), "n_stints": int(len(g)),
                              "ratio_to_max": (float(lf["life_laps"] / g["L"].max()) if lf.get("life_laps") else None),
                              "baseline_ratio_to_max": (float(blf["life_laps"] / g["L"].max()) if blf.get("life_laps") else None)})
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
                               "race_factor_range": [float(pdf["race_factor"].min()), float(pdf["race_factor"].max())]}
        else:
            r["per_driver"] = {}
        # -- counterfactual sanity ------------------------------------------
        cf = pd.read_parquet(DATA_PROCESSED / f"counterfactual_{key}.parquet")
        r["counterfactual"] = {"n": int(len(cf)), "n_over_30s": int((cf["loss_s"] > 30).sum()),
                               "n_with_sc_stops": int((cf["n_sc_stops"] > 0).sum()) if "n_sc_stops" in cf else None,
                               "n_over_30s_classified": int(((cf["loss_s"] > 30) & cf["classified"]).sum()) if "classified" in cf else None,
                               "median_loss_s": float(cf["loss_s"].median()),
                               "top": cf.head(3)[["driver", "compounds", "loss_s"]].round(1).to_dict("records"),
                               "baseline_n_over_30s": (int((pd.read_parquet(DATA_PROCESSED.parent.parent / "bench" / "baseline" / "processed" / f"counterfactual_{key}.parquet")["loss_s"] > 30).sum())
                                                       if (DATA_PROCESSED.parent.parent / "bench" / "baseline" / "processed" / f"counterfactual_{key}.parquet").exists() else None)}
        r["cliff"] = m["score"]["cliff"]
        r["ladder_check"] = m.get("ladder_check")
        r["pace_calibration"] = {k: v for k, v in (m.get("pace_calibration") or {}).items() if k in ("applied", "measured_net_s", "model_net_before", "model_net_after", "clipped")}
        r["gates_failed"] = [g["gate"] for g in m["gates"] if not g["pass"]]
        r["calibration"] = {k: v for k, v in (m.get("calibration") or {}).items() if k in ("grip_budget_by_compound", "undercut_lambda", "plan_prior_tau_s", "grid_start_penalty_s", "dirty_air_s_per_lap", "manage_cost_s", "manage_wear_floor")}
        out[key] = r
        win = r["pit_windows"]
        rows.append({"event": key, "recommended": st["best"], "tyre_optimal": tyre_opt.get("label"),
                     "baseline": (bm["strategy"]["best"] if bm else None),
                     "winner": r["winner_seq"], "winner_pits": r["winner_pits"],
                     "field_modal": modal_seq, "stops_match_mode": r["stops_match_mode"], "stops_share": round(r["stops_share"], 2),
                     "seq_share": round(r["rec_seq_share"], 2), "seq_in_top5": r["seq_in_top5"],
                     "start_ok": r["start_matches_majority"],
                     "first_rec_minus_field": r["first_stop"]["rec_minus_field"],
                     "first_tyre_minus_field": r["first_stop"]["tyre_minus_field"],
                     "first_baseline_minus_field": r["first_stop"]["baseline_minus_field"],
                     "sc_set_first": r["first_stop"]["sc_set"],
                     "regret_tool": r["oracle"]["regret_s"].get("tool"), "regret_tyre_opt": r["oracle"]["regret_s"].get("tyre_optimal"),
                     "regret_baseline": r["oracle"]["regret_s"].get("baseline_tool"),
                     "regret_field": r["oracle"]["regret_s"].get("field_modal"), "regret_winner": r["oracle"]["regret_s"].get("winner"),
                     "win1_inside": (win[0]["share_inside"] if win else None), "win1_err": (win[0]["median_abs_err"] if win else None)})
        print(f"\n== {key}: tool {st['best']} | tyre-optimal {tyre_opt.get('label')} | baseline {bm['strategy']['best'] if bm else '-'} "
              f"| winner {r['winner']} {r['winner_seq']} @ {r['winner_pits']} | modal {modal_seq}")
        print(f"   stops: rec {best['n_stops']} mode {mode_stops} ({r['stops_share']:.0%} of field); seq share {r['rec_seq_share']:.0%}; "
              f"in top5 {r['seq_in_top5']}; start {best['compounds'][0]} vs field {dict(start_counts)}")
        print(f"   first stop: field median (green) {fmed}, {sc_share_first:.0%} under SC; rec {r['first_stop']['recommended']} "
              f"tyre-opt {r['first_stop']['tyre_optimal']} baseline {r['first_stop']['baseline_rec']}")
        print(f"   oracle rates {r['oracle']['rates']}  regret (s): {r['oracle']['regret_s']}")
        print(f"   windows: {win}")
        print(f"   life: {[(x['compound'], x['pred_life_at_push'], x['bound_by'], x['obs_p90'], x['obs_max']) for x in life_rows]}")
        print(f"   per-driver: {r['per_driver']}")
        print(f"   counterfactual >30s: {r['counterfactual']['n_over_30s']}/{r['counterfactual']['n']} "
              f"(baseline {r['counterfactual']['baseline_n_over_30s']}); SC-held drivers {r['counterfactual']['n_with_sc_stops']}")
    dump("strategy.json", out)
    t = pd.DataFrame(rows)
    t.to_csv(dump("strategy_table.json", []).with_suffix(".csv"), index=False)
    print("\n" + t.to_string(index=False))


if __name__ == "__main__":
    main()
