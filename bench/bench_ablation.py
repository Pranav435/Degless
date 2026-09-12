"""Ablation: which of the changes moved the decision, and by how much.

For every weekend the same search is run on the shipped posterior with one
term switched off at a time, and the answer compared with the field:

  full                the shipped objective (calibrated constants, pace
                      calibration, undercut exposure, plan prior)
  no_position         undercut lambda = 0
  no_plan_prior       plan-prior tau = 0
  no_pace_cal         pace offsets as fitted (the ladder gate not enforced)
  config_constants    the hand-calibrated constants of the previous build
  practice_only       the practice posterior without the circuit history
  hard_ladder         the strictly ordered ladder variant (from the fit stage)
  old_budget          grip budget 3.8 s for every compound

plus the ladder-vs-race check: per-compound race degradation measured two
ways against the model's ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import EVENTS, driver_plans, dump, meta, offline, race_table  # noqa: E402
from src import strategy as strat
from src.calibration import Calibration, get_calibration
from src.config import DATA_PROCESSED, GRIP_BUDGET_S, get_event
from src.history import race_deg_slopes
from src.model_bayes import BayesFit
from src.regime import RegimeFactor, measure_regime
from src.tyre import TyreModel

ORDER = ["SOFT", "MEDIUM", "HARD"]


def ordered(rates: dict) -> bool | None:
    cs = [c for c in ORDER if c in rates]
    if len(cs) < 2:
        return None
    return all(rates[a] > rates[b] for a, b in zip(cs, cs[1:]))


def run(key: str, m: dict, fit: BayesFit, cal: Calibration, *, lam=None, tau=None, pace_cal=True,
        budgets=None, grid=None, dirty=None, n_draws: int = 300) -> dict:
    ev = get_event(key)
    regime = RegimeFactor(ratio=float(m["regime"]["ratio"]), ln_sd=float(m["regime"]["ln_sd"]))
    caps = {c: int(v) for c, v in (m.get("circuit_history") or {}).get("stint_cap", {}).items()} or None
    support = {k: float(v) for k, v in m["age_support_by_compound"].items()}
    total = fit.posterior["lin"].shape[0]
    idx = np.random.default_rng(0).choice(total, size=min(n_draws, total), replace=False)
    model = TyreModel.from_fit(fit, draws=idx, budget=(budgets if budgets is not None else cal.budgets),
                               manage_floor=cal.manage_wear_floor, manage_cost_s=cal.manage_cost_s)
    kw = dict(regime=regime, support=support, max_per_compound=m["allocation"]["caps"], max_stint=caps,
              undercut_lambda=(cal.undercut_lambda if lam is None else lam),
              plan_prior=m.get("plan_prior") or {}, plan_prior_tau_s=(cal.plan_prior_tau_s if tau is None else tau),
              traffic_s_per_lap=(cal.dirty_air_s_per_lap if dirty is None else dirty),
              grid_penalty_s=(cal.grid_start_penalty_s if grid is None else grid), step=1)
    ns = m.get("net_step") or {}
    if pace_cal and ns.get("measured") is not None:
        _, res, _ = strat.search_with_pace_calibration(model, ev, float(m["pit_loss_s"]), net_step_s=float(ns["measured"]),
                                                       net_step_se_s=float(ns.get("se") or 0.0), **kw)
    else:
        res = strat.simulate_model(model, ev, float(m["pit_loss_s"]), **kw)
    return {"best": res.best_label, "p_stops": res.p_stops, "push": res.best["push"],
            "n_stops": res.best["n_stops"], "seq": "-".join(res.best["compounds"]),
            "first": (int(res.best["pit_laps"][0]) if res.best["pit_laps"] else None),
            "tyre_optimal": res.tyre_optimal_label,
            "life": {x["compound"]: round(float(x["life_laps"]), 1) for _, x in res.life.iterrows()},
            "deg_full_push": {x["compound"]: round(float(x["deg_s_per_lap"]), 4) for _, x in res.life.iterrows()}}


def main() -> None:
    offline()
    out, rows = {}, []
    for key in EVENTS:
        ev = get_event(key)
        m = meta(key)
        cal = get_calibration(ev)
        plans = driver_plans(race_table(key), ev.n_race_laps)
        seqs = plans["seq"].value_counts()
        mode_stops = int(plans["n_stops"].mode().iloc[0])
        green = [p[0] for p, s in zip(plans["pit_laps"], plans["first_sc"]) if p and not s]
        fmed = float(np.median(green)) if green else None
        sc_set = bool(plans["first_sc"].mean() > 0.4)
        f_ship = BayesFit.load(DATA_PROCESSED / f"posterior_{key}.npz")
        f_prac = BayesFit.load(DATA_PROCESSED / f"posterior_{key}_practice.npz")
        old = Calibration()   # the config constants of the previous build
        variants = {
            "full": run(key, m, f_ship, cal),
            "no_position": run(key, m, f_ship, cal, lam=0.0),
            "no_plan_prior": run(key, m, f_ship, cal, tau=0.0),
            "no_pace_cal": run(key, m, f_ship, cal, pace_cal=False),
            "config_constants": run(key, m, f_ship, old, lam=0.0, tau=0.0, pace_cal=False, budgets=GRIP_BUDGET_S),
            "practice_only": run(key, m, f_prac, cal),
            "old_budget": run(key, m, f_ship, cal, budgets=GRIP_BUDGET_S),
        }
        r = {}
        for tag, v in variants.items():
            v["seq_share"] = float(seqs.get(v["seq"], 0) / max(len(plans), 1))
            v["stops_match_mode"] = bool(v["n_stops"] == mode_stops)
            v["first_minus_field"] = ((v["first"] - fmed) if (fmed is not None and v["first"] is not None and not sc_set) else None)
            r[tag] = v
        # -- the ladder vs the race -------------------------------------------
        rt = race_table(key)
        rc = rt[rt["is_accurate"] & ~rt["pit_in"] & ~rt["pit_out"] & (rt["track_status"].astype(str) == "1")]
        fe = {c: round(v["slope"], 4) for c, v in race_deg_slopes(rc, ev.fuel_effect_s_per_lap).items()}
        rm = measure_regime(key)
        sfe = {c: round(v["race_s_per_lap"], 4) for c, v in rm.per_compound.items()}
        prac = {c: round(v["practice_s_per_lap"], 4) for c, v in rm.per_compound.items()}
        model_rates = {c: round(float((f_ship.deg_loss(c, np.array([1.0, 10.0]))[:, 1] - f_ship.deg_loss(c, np.array([1.0, 10.0]))[:, 0]).mean() / 9), 4)
                       for c in f_ship.compounds}
        r["ladder"] = {"race_lap_fe": fe, "race_lap_fe_ordered": ordered(fe),
                       "race_stint_fe": sfe, "race_stint_fe_ordered": ordered(sfe),
                       "practice_stint_fe": prac, "practice_stint_fe_ordered": ordered(prac),
                       "model_rate": model_rates, "model_ordered": ordered(model_rates),
                       "circuit_ladder": (m.get("compound_ladder") or {}).get("circuit_ladder"),
                       "fit_variants": {k: {"slopes": {s["compound"]: round(s["slope_s_per_lap"], 4) for s in v["slopes"]}}
                                        for k, v in ((m.get("bayes") or {}).get("variants") or {}).items()}}
        r["field"] = {"mode_stops": mode_stops, "modal_seq": seqs.index[0] if len(seqs) else None,
                      "first_median_green": fmed, "sc_set": sc_set}
        out[key] = r
        rows.append({"event": key, **{f"{tag}": v["best"] for tag, v in variants.items()},
                     "field_modal": r["field"]["modal_seq"], "field_first": fmed,
                     **{f"{tag}_first_err": v["first_minus_field"] for tag, v in variants.items()},
                     **{f"{tag}_share": round(v["seq_share"], 2) for tag, v in variants.items()},
                     "race_lap_fe": fe, "ordered_lapfe": ordered(fe), "model_ordered": ordered(model_rates)})
        print(f"{key:16s} " + " | ".join(f"{tag} {v['best']} (share {v['seq_share']:.0%}, first {v['first_minus_field']})"
                                          for tag, v in variants.items()), flush=True)
        print(f"{'':16s} race rates lap-FE {fe} ordered={ordered(fe)} | model {model_rates} ordered={ordered(model_rates)}", flush=True)
    dump("ablation.json", out)
    pd.DataFrame(rows).to_csv(dump("ablation_table.json", []).with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
