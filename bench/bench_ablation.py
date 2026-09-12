"""Ablation: which of the changes moved the decision, and by how much.

For every weekend the same search is run on the shipped posterior with one
term switched off at a time, and the answer compared with the field:

  full                    the shipped objective (V4: calibrated constants, pace
                          calibration, the race-state first-stop term, undercut
                          exposure on later stops, plan prior through the
                          Pirelli nominations, per-circuit dirty air,
                          censoring-aware grip budgets; no first-stop prior)
  no_race_state           V4's race-state term switched off: the V3 objective
                          exactly (undercut exposure and the first-stop prior
                          time the first stop)
  race_state_no_cover     the race state with rivals that never cover
  race_state_lead_lap_value  a place valued at the lead-lap finishing interval
                          (the definition first implemented) instead of every
                          classified finisher's
  race_state_undiscounted a place valued at the full finishing interval (psi = 1:
                          every pit-cycle order assumed to survive to the flag)
  no_first_stop_prior     kappa = 0 (in V4 the prior is already off: equals full)
  no_plan_prior           plan-prior tau = 0
  no_nomination_mapping   the plan prior built from the letters as recorded
                          (`plan_prior_for(cp, use_nominations=False)`) — V2's
                          behaviour, the Barcelona defect
  no_circuit_regime_prior the regime factor from the 2026 donor pool alone; the
                          practice posterior is re-folded with it, so the curve
                          and the scaling agree, and its stint-rate MAE is
                          scored as well as its decision
  no_dirty_air_circuit    the pooled dirty-air cost instead of this circuit's
  no_cliff_budgets        V2's grip budgets (bench/v2/processed/calibration.json)
  no_percar               per-car plans with no driver terms at all
  no_position             undercut lambda = 0
  no_pace_cal             pace offsets as fitted (the ladder gate not enforced)
  config_constants        the hand-calibrated constants of V1
  practice_only           the practice posterior without the circuit history
  old_budget              grip budget 3.8 s for every compound

Metrics per variant: the sequence run by anyone, the start compound against the
majority, the stop count against the mode, the share of the field on the plan,
the first stop against the field's green-flag median (the three non-safety-car
weekends only) and the predicted tyre life over the longest stint run.

`full` is expected to reproduce the shipped `meta["strategy"]["best"]`.  It is
reported, not asserted: the pipeline searches 500 draws and the ablation 300, so
two plans within a few tenths of each other can swap, and a swap is a fact about
how tight the decision is rather than a failure of the benchmark.

Every variant but `no_race_state` and `config_constants` keeps the race state
on (they switch off one *other* term of the shipped objective); `config_constants`
is the V1 baseline and has no race state either.

plus the ladder-vs-race check: per-compound race degradation measured two
ways against the model's ordering.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd

from common import (OUT, NON_SC_EVENTS, arg_events, cp_for, dirty_air_of, driver_plans, dump,  # noqa: E402
                    first_stop_tables, fs_kwargs, kappa_of, memoise_regime, meta, offline,
                    race_table, v2_calibration)
from src import racestate, strategy as strat
from src.calibration import Calibration, get_calibration
from src.config import DATA_PROCESSED, GRIP_BUDGET_S, get_event
from src.history import apply_circuit_prior, plan_prior_for, race_deg_slopes
from src.laps import clean_laps
from src.model_bayes import BayesFit
from src.regime import RegimeFactor, measure_regime, regime_prior
from src.tyre import TyreModel

ORDER = ["SOFT", "MEDIUM", "HARD"]
N_DRAWS = 300


def ordered(rates: dict) -> bool | None:
    cs = [c for c in ORDER if c in rates]
    if len(cs) < 2:
        return None
    return all(rates[a] > rates[b] for a, b in zip(cs, cs[1:]))


def run(key: str, m: dict, fit: BayesFit, cal: Calibration, *, lam=None, tau=None, pace_cal=True,
        budgets=None, grid=None, dirty=None, regime=None, plan_prior=None, fs_tables=None,
        kappa=None, race_state="shipped", race_state_cover: bool = True, n_draws: int = N_DRAWS) -> dict:
    """One search.  Every switched-off term is a keyword whose default is the
    shipped value, so a variant names exactly what it changed.

    `race_state="shipped"` is V4's term with this weekend's leave-one-out
    constants (and the first-stop prior off, as the pipeline runs it); `None`
    is the V3 objective; a `RaceStateConstants` is that variant's constants."""
    ev = get_event(key)
    if isinstance(race_state, str):
        race_state = racestate.measure_constants(exclude=key)
    if race_state is not None and kappa is None:
        kappa = 0.0
    if regime is None:
        regime = RegimeFactor(ratio=float(m["regime"]["ratio"]), ln_sd=float(m["regime"]["ln_sd"]))
    caps = {c: int(v) for c, v in (m.get("circuit_history") or {}).get("stint_cap", {}).items()} or None
    support = {k: float(v) for k, v in m["age_support_by_compound"].items()}
    total = fit.posterior["lin"].shape[0]
    idx = np.random.default_rng(0).choice(total, size=min(n_draws, total), replace=False)
    model = TyreModel.from_fit(fit, draws=idx, budget=(budgets if budgets is not None else cal.budgets),
                               manage_floor=cal.manage_wear_floor, manage_cost_s=cal.manage_cost_s)
    kw = dict(regime=regime, support=support, max_per_compound=m["allocation"]["caps"], max_stint=caps,
              undercut_lambda=(cal.undercut_lambda if lam is None else lam),
              plan_prior=(m.get("plan_prior") or {} if plan_prior is None else plan_prior),
              plan_prior_tau_s=(cal.plan_prior_tau_s if tau is None else tau),
              traffic_s_per_lap=(dirty_air_of(cal, ev.circuit) if dirty is None else dirty),
              grid_penalty_s=(cal.grid_start_penalty_s if grid is None else grid), step=1)
    kw.update(fs_kwargs(strat.simulate_model, fs_tables, kappa_of(cal) if kappa is None else kappa))
    if race_state is not None:
        kw.update(race_state=race_state, race_state_cover=race_state_cover)
    ns = m.get("net_step") or {}
    if pace_cal and ns.get("measured") is not None:
        _, res, _ = strat.search_with_pace_calibration(model, ev, float(m["pit_loss_s"]), net_step_s=float(ns["measured"]),
                                                       net_step_se_s=float(ns.get("se") or 0.0), **kw)
    else:
        res = strat.simulate_model(model, ev, float(m["pit_loss_s"]), **kw)
    return {"best": res.best_label, "p_stops": res.p_stops, "push": res.best["push"],
            "n_stops": res.best["n_stops"], "seq": "-".join(res.best["compounds"]),
            "start": res.best["compounds"][0],
            "first": (int(res.best["pit_laps"][0]) if res.best["pit_laps"] else None),
            "pit_laps": list(res.best["pit_laps"]),
            "first_stop_s": res.best.get("first_stop_s"),
            "race_state_s": res.best.get("race_state_s"),
            "race_state_group": ((res.race_state or {}).get("groups") or {}).get((res.race_state or {}).get("best_group")),
            "tyre_optimal": res.tyre_optimal_label,
            "life": {x["compound"]: round(float(x["life_laps"]), 1) for _, x in res.life.iterrows()},
            "deg_full_push": {x["compound"]: round(float(x["deg_s_per_lap"]), 4) for _, x in res.life.iterrows()}}


def percar(key: str, m: dict, fit: BayesFit, cal: Calibration, drivers: list, *, race_factors=None,
           fs_tables=None, n_draws: int = N_DRAWS) -> dict:
    """Per-car plans with whatever driver terms the caller leaves in.

    `race_factors=None` *and* a model with its per-driver deviation stripped is
    the real "no per-car" ablation: `TyreModel.for_driver` reads `driver_dev`
    from the model, so passing no factors alone would still leave the practice
    deviation in.
    """
    ev = get_event(key)
    caps = {c: int(v) for c, v in (m.get("circuit_history") or {}).get("stint_cap", {}).items()} or None
    total = fit.posterior["lin"].shape[0]
    idx = np.random.default_rng(0).choice(total, size=min(n_draws, total), replace=False)
    model = TyreModel.from_fit(fit, draws=idx, budget=cal.budgets, manage_floor=cal.manage_wear_floor,
                               manage_cost_s=cal.manage_cost_s)
    if race_factors is None:
        model = dataclasses.replace(model, driver_dev={})
    kw = dict(regime=RegimeFactor(ratio=float(m["regime"]["ratio"]), ln_sd=float(m["regime"]["ln_sd"])),
              support={k: float(v) for k, v in m["age_support_by_compound"].items()},
              max_per_compound=m["allocation"]["caps"], max_stint=caps,
              undercut_lambda=cal.undercut_lambda, plan_prior=m.get("plan_prior") or {},
              plan_prior_tau_s=cal.plan_prior_tau_s, traffic_s_per_lap=dirty_air_of(cal, ev.circuit),
              grid_penalty_s=cal.grid_start_penalty_s)
    # the shipped per-car plans carry the race state (and no first-stop prior)
    kw.update(race_state=racestate.measure_constants(exclude=key))
    try:
        pdf = strat.per_driver_plans(model, ev, float(m["pit_loss_s"]), drivers,
                                     race_factors=(race_factors or None), **kw)
    except Exception as exc:
        return {"error": str(exc)}
    if pdf.empty:
        return {"n": 0}
    fs = pd.to_numeric(pdf["first_stop"], errors="coerce")
    return {"n": int(len(pdf)), "share_same_shape_as_field": float(pdf["same_shape_as_field"].mean()),
            "first_stop_spread": (float(fs.max() - fs.min()) if fs.notna().any() else None),
            "first_stop_median": (float(fs.median()) if fs.notna().any() else None),
            "n_distinct_plans": int(pdf["compounds"].nunique())}


def accuracy_spearman(key: str) -> dict:
    """The per-car Spearman of each rate-scale variant, from accuracy.json."""
    p = OUT / "accuracy.json"
    if not p.exists():
        return {"source": "accuracy.json has not been written yet"}
    try:
        var = (((json.loads(p.read_text()).get("per_event") or {}).get(key) or {}).get("variants") or {})
    except Exception as exc:
        return {"source": f"accuracy.json unreadable: {exc}"}
    out = {"source": "bench/out/accuracy.json"}
    for name in ("sealed", "sealed_driver_hist", "sealed_driver_practice_dev", "sealed_driver_team_pooled"):
        if name in var:
            out[name] = var[name].get("spearman")
    return out


def life_ratios(life: dict, obs_max: dict) -> dict:
    """Predicted life over the longest stint any finisher ran, per compound."""
    out = {c: (round(float(v) / obs_max[c], 2) if obs_max.get(c) else None) for c, v in (life or {}).items()}
    vals = [v for v in out.values() if v is not None]
    return {"by_compound": out, "median": (float(np.median(vals)) if vals else None),
            "n_over_1": int(sum(v > 1.0 for v in vals))}


def main() -> None:
    args = arg_events(__doc__)
    offline()
    memoise_regime()
    out, rows = {}, []
    v2cal = v2_calibration()
    for key in args.events:
        ev = get_event(key)
        m = meta(key)
        cal = get_calibration(ev)
        race = race_table(key)
        plans = driver_plans(race, ev.n_race_laps)
        seqs = plans["seq"].value_counts()
        starts = plans["seq"].str.split("-").str[0].value_counts()
        mode_stops = int(plans["n_stops"].mode().iloc[0])
        green = [p[0] for p, s in zip(plans["pit_laps"], plans["first_sc"]) if p and not s]
        fmed = float(np.median(green)) if green else None
        sc_set = bool(plans["first_sc"].mean() > 0.4)
        obs_max: dict = {}
        for _, p in plans.iterrows():
            for c, L in zip(p["seq"].split("-"), p["stint_lens"]):
                obs_max[c] = max(obs_max.get(c, 0), int(L))
        f_ship = BayesFit.load(DATA_PROCESSED / f"posterior_{key}.npz")
        f_prac = BayesFit.load(DATA_PROCESSED / f"posterior_{key}_practice.npz")
        cp = cp_for(key, m)
        fs_tables = first_stop_tables(cp, ev, list(f_ship.compounds))
        old = Calibration()   # V1's config constants

        # -- the regime without this circuit's own practice-vs-race history ----
        reg_nc, f_nc, mae_nc = None, f_ship, None
        try:
            reg_nc = regime_prior(ev, use_circuit_history=False)
            if cp.available and cp.rate_prior:
                f_nc, _ = apply_circuit_prior(f_prac, ev, reg_nc, cp)
            else:
                f_nc = f_prac
            from bench_accuracy import AGES, stint_rates, summarise
            rm = reg_nc.draws(f_nc.posterior["lin"].shape[0], seed=5)
            draws = {c: f_nc.deg_loss(c, AGES) * rm[:, None] for c in f_nc.compounds}
            mae_nc = summarise(stint_rates(draws, clean_laps(race), ev, float(cal.sigma_race_lap_s)))
        except Exception as exc:
            print(f"  no_circuit_regime_prior unavailable for {key}: {exc}")

        v2_budgets = ((v2cal.get("loo") or {}).get(key) or v2cal.get("global") or {}).get("grip_budget_by_compound")
        pp_letters = plan_prior_for(cp, use_nominations=False) if cp is not None else {}

        rs_c = racestate.measure_constants(exclude=key)
        variants = {
            "full": run(key, m, f_ship, cal, fs_tables=fs_tables),
            "no_race_state": run(key, m, f_ship, cal, fs_tables=fs_tables, race_state=None),
            "race_state_no_cover": run(key, m, f_ship, cal, fs_tables=fs_tables, race_state_cover=False),
            "race_state_lead_lap_value": run(key, m, f_ship, cal, fs_tables=fs_tables,
                                             race_state=dataclasses.replace(rs_c, place_gap_s=rs_c.place_gap_lead_lap_s)),
            "race_state_undiscounted": run(key, m, f_ship, cal, fs_tables=fs_tables,
                                           race_state=dataclasses.replace(rs_c, persistence=1.0)),
            "no_first_stop_prior": run(key, m, f_ship, cal, fs_tables=fs_tables, kappa=0.0),
            "no_plan_prior": run(key, m, f_ship, cal, fs_tables=fs_tables, tau=0.0),
            "no_nomination_mapping": run(key, m, f_ship, cal, fs_tables=fs_tables, plan_prior=pp_letters),
            "no_dirty_air_circuit": run(key, m, f_ship, cal, fs_tables=fs_tables,
                                        dirty=float(cal.dirty_air_s_per_lap)),
            "no_position": run(key, m, f_ship, cal, fs_tables=fs_tables, lam=0.0),
            "no_pace_cal": run(key, m, f_ship, cal, fs_tables=fs_tables, pace_cal=False),
            "config_constants": run(key, m, f_ship, old, lam=0.0, tau=0.0, pace_cal=False,
                                    budgets=GRIP_BUDGET_S, kappa=0.0, race_state=None),
            "practice_only": run(key, m, f_prac, cal, fs_tables=fs_tables),
            "old_budget": run(key, m, f_ship, cal, fs_tables=fs_tables, budgets=GRIP_BUDGET_S),
        }
        if v2_budgets:
            variants["no_cliff_budgets"] = run(key, m, f_ship, cal, fs_tables=fs_tables,
                                               budgets={c: float(v) for c, v in v2_budgets.items()})
        if reg_nc is not None:
            variants["no_circuit_regime_prior"] = run(key, m, f_nc, cal, fs_tables=fs_tables, regime=reg_nc)

        r = {}
        for tag, v in variants.items():
            v["seq_share"] = float(seqs.get(v["seq"], 0) / max(len(plans), 1))
            v["seq_run_by_anyone"] = bool(seqs.get(v["seq"], 0) > 0)
            v["stops_match_mode"] = bool(v["n_stops"] == mode_stops)
            v["start_matches_majority"] = bool(len(starts) and starts.index[0] == v["start"])
            v["first_minus_field"] = ((v["first"] - fmed) if (fmed is not None and v["first"] is not None and not sc_set) else None)
            v["life_ratio"] = life_ratios(v["life"], obs_max)
            r[tag] = v
        if "no_circuit_regime_prior" in r:
            r["no_circuit_regime_prior"]["regime"] = {"ratio": float(reg_nc.ratio), "ln_sd": float(reg_nc.ln_sd),
                                                      "mode": (reg_nc.temperature or {}).get("mode"),
                                                      "shipped_ratio": float(m["regime"]["ratio"])}
            r["no_circuit_regime_prior"]["rate_mae"] = (mae_nc or {}).get("rate_mae")
            r["no_circuit_regime_prior"]["rate_bias"] = (mae_nc or {}).get("rate_bias")
            r["no_circuit_regime_prior"]["rate_cov90"] = (mae_nc or {}).get("rate_cov90")
            r["no_circuit_regime_prior"]["shipped_rate_mae"] = m["score"]["mae"]

        # -- per-car: the shipped plans against plans with no driver terms -------
        drivers = sorted(race["driver"].unique())
        shipped_pd = pd.DataFrame(m.get("per_driver") or [])
        fs_ship = pd.to_numeric(shipped_pd.get("first_stop"), errors="coerce") if len(shipped_pd) else pd.Series(dtype=float)
        r["percar"] = {
            "shipped": {"n": int(len(shipped_pd)),
                        "share_same_shape_as_field": (float(shipped_pd["same_shape_as_field"].mean()) if len(shipped_pd) else None),
                        "first_stop_spread": (float(fs_ship.max() - fs_ship.min()) if fs_ship.notna().any() else None),
                        "mode": getattr(cal, "percar_mode", None)},
            "no_percar": percar(key, m, f_ship, cal, drivers, race_factors=None, fs_tables=fs_tables),
            "spearman": accuracy_spearman(key)}

        # -- the `full` variant against what the pipeline shipped ---------------
        shipped = m["strategy"]["best"]
        r["full_vs_shipped"] = {"shipped": shipped, "ablation_full": variants["full"]["best"],
                                "matches": bool(variants["full"]["best"] == shipped),
                                "note": "the pipeline searches 500 draws, the ablation 300"}
        if not r["full_vs_shipped"]["matches"]:
            print(f"  NOTE {key}: `full` gives {variants['full']['best']} where the pipeline shipped {shipped}")

        # -- the ladder vs the race -------------------------------------------
        rt = race
        rc = rt[rt["is_accurate"] & ~rt["pit_in"] & ~rt["pit_out"] & (rt["track_status"].astype(str) == "1")]
        fe = {c: round(v["slope"], 4) for c, v in race_deg_slopes(rc, ev.fuel_effect_s_per_lap).items()}
        rm_meas = measure_regime(key)
        sfe = {c: round(v["race_s_per_lap"], 4) for c, v in rm_meas.per_compound.items()}
        prac = {c: round(v["practice_s_per_lap"], 4) for c, v in rm_meas.per_compound.items()}
        model_rates = {c: round(float((f_ship.deg_loss(c, np.array([1.0, 10.0]))[:, 1] - f_ship.deg_loss(c, np.array([1.0, 10.0]))[:, 0]).mean() / 9), 4)
                       for c in f_ship.compounds}
        r["ladder"] = {"race_lap_fe": fe, "race_lap_fe_ordered": ordered(fe),
                       "race_stint_fe": sfe, "race_stint_fe_ordered": ordered(sfe),
                       "practice_stint_fe": prac, "practice_stint_fe_ordered": ordered(prac),
                       "model_rate": model_rates, "model_ordered": ordered(model_rates),
                       "circuit_ladder": (m.get("compound_ladder") or {}).get("circuit_ladder"),
                       # V3's gate: the stint-fixed-effects baseline the posterior is checked against
                       "stint_fe_baseline": {k: v for k, v in (m.get("stint_fe_baseline") or {}).items()
                                            if k != "table"},
                       "bayes_vs_stint_fe_pooled": m.get("bayes_vs_stint_fe_pooled"),
                       "fit_variants": {k: {"slopes": {s["compound"]: round(s["slope_s_per_lap"], 4) for s in v["slopes"]}}
                                        for k, v in ((m.get("bayes") or {}).get("variants") or {}).items()}}
        r["field"] = {"mode_stops": mode_stops, "modal_seq": seqs.index[0] if len(seqs) else None,
                      "majority_start": (starts.index[0] if len(starts) else None),
                      "first_median_green": fmed, "sc_set": sc_set, "obs_max_stint": obs_max,
                      "n_classified": int(len(plans))}
        r["inputs"] = {"kappa_s": kappa_of(cal), "first_stop_prior": bool(fs_tables),
                       "dirty_air_circuit": dirty_air_of(cal, ev.circuit),
                       "dirty_air_pooled": float(cal.dirty_air_s_per_lap),
                       "plan_prior_source": (m.get("plan_prior") or {}).get("source"),
                       "plan_prior_letters_source": (pp_letters or {}).get("source"),
                       "nomination": getattr(cp, "nomination", None),
                       "v2_budgets": v2_budgets, "budgets": cal.budgets}
        out[key] = r
        rows.append({"event": key, **{f"{tag}": v["best"] for tag, v in variants.items()},
                     "field_modal": r["field"]["modal_seq"], "field_first": fmed,
                     **{f"{tag}_first_err": v["first_minus_field"] for tag, v in variants.items()},
                     **{f"{tag}_share": round(v["seq_share"], 2) for tag, v in variants.items()},
                     "race_lap_fe": fe, "ordered_lapfe": ordered(fe), "model_ordered": ordered(model_rates)})
        print(f"{key:16s} " + " | ".join(f"{tag} {v['best']} (share {v['seq_share']:.0%}, first {v['first_minus_field']})"
                                          for tag, v in variants.items()), flush=True)
        print(f"{'':16s} race rates lap-FE {fe} ordered={ordered(fe)} | model {model_rates} ordered={ordered(model_rates)}", flush=True)
        print(f"{'':16s} per-car shipped {r['percar']['shipped']} | no_percar {r['percar']['no_percar']}", flush=True)
        dump("ablation.json", out)

    # -- the pooled table the report quotes ----------------------------------
    tags = [t for t in out[args.events[0]] if isinstance(out[args.events[0]][t], dict) and "seq_run_by_anyone" in out[args.events[0]][t]] if out else []
    pooled = {}
    for t in tags:
        vs = [out[k][t] for k in out if t in out[k]]
        firsts = {k: out[k][t]["first_minus_field"] for k in out if t in out[k] and k in NON_SC_EVENTS}
        lifes = [out[k][t]["life_ratio"]["median"] for k in out if t in out[k] and out[k][t]["life_ratio"]["median"] is not None]
        pooled[t] = {"n_events": len(vs),
                     "seq_run_by_anyone": int(sum(bool(v["seq_run_by_anyone"]) for v in vs)),
                     "start_matches_majority": int(sum(bool(v["start_matches_majority"]) for v in vs)),
                     "stops_match_mode": int(sum(bool(v["stops_match_mode"]) for v in vs)),
                     "mean_seq_share": float(np.mean([v["seq_share"] for v in vs])),
                     "first_minus_field": firsts,
                     "mean_abs_first_err": (float(np.mean([abs(x) for x in firsts.values() if x is not None]))
                                            if any(x is not None for x in firsts.values()) else None),
                     "life_ratio_median": (float(np.median(lifes)) if lifes else None)}
    print("\n=== pooled ablation ===")
    print(pd.DataFrame(pooled).T.to_string())
    dump("ablation.json", {**out, "_pooled": pooled,
                           "_note": ("`_pooled` counts are out of the weekends benchmarked; "
                                     "first_minus_field is restricted to the non-safety-car weekends "
                                     f"{NON_SC_EVENTS}")})
    pd.DataFrame(rows).to_csv(dump("ablation_table.json", []).with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
