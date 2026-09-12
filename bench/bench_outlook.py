"""Outlook benchmark: the recommendation *before any practice has run*.

Composes the prior exactly as `src.outlook.load_base(force_prior=True)` does
on a weekend with no posterior (compound-ladder prior + the circuit's 2023-25
races pooled + transferred regime, allocation, pit loss, stint caps, the
plan-shape prior and the leave-one-out calibration), runs the same search the
sealed model runs (ladder gate enforced, position term), and compares the
answer with what the field did and with the sealed model's answer.

Two runs per weekend:

  prior_only          the shipped prior, whose plan-shape counts are read
                      through the Pirelli nominations (`plan_prior_for`)
  prior_only_letters  the identical prior with the plan shapes read
                      letter-for-letter, `plan_prior_for(cp, use_nominations=False)`
                      — V2's behaviour, so the report can show the prior-only
                      modal match with and without the C-number mapping

Nothing else differs between them: the same base model, the same regime, the
same calibration, the same search.  The difference is the mapping.

Caveat, stated in the report: the regime factor and pit-loss/allocation
priors pool every *other* 2026 weekend on disk, including ones raced after the
target - the honest pre-season number would be wider.
"""

from __future__ import annotations

import time
from dataclasses import fields as dc_fields, replace

import numpy as np
import pandas as pd

from common import arg_events, baseline_out, driver_plans, dump, memoise_regime, meta, offline, race_table, v2_out  # noqa: E402
from src import strategy as strat
from src.config import get_event
from src.history import CircuitPrior, plan_prior_for
from src.outlook import load_base, sim_kwargs


def cp_of(base) -> CircuitPrior | None:
    """The `CircuitPrior` `load_base` built, rebuilt from the dict it kept.

    `BaseModel.history` is `cp.as_dict()`, so the plan prior can be rebuilt from
    the very object the outlook used rather than from a second
    `circuit_prior()` call that might see a different cache.
    """
    d = dict(getattr(base, "history", None) or {})
    if not d:
        return None
    names = {f.name for f in dc_fields(CircuitPrior)}
    cp = CircuitPrior(**{k: v for k, v in d.items() if k in names})
    cp.nominations_by_year = {int(y): v for y, v in (cp.nominations_by_year or {}).items()}
    return cp


def one(key: str, ev, base, plans, m: dict, *, tag: str) -> dict:
    """One search on an already-composed base model."""
    t0 = time.perf_counter()
    net = base.net_step or {}
    sk = sim_kwargs(base)
    model, res, pc = strat.search_with_pace_calibration(
        base.model, ev, base.pit_loss_s,
        net_step_s=float(net.get("measured") if net.get("measured") is not None else np.nan),
        net_step_se_s=float(net.get("se") or 0.0), **sk)
    dt = time.perf_counter() - t0
    mode_stops = int(plans["n_stops"].mode().iloc[0])
    seqs = plans["seq"].value_counts()
    green = [p[0] for p, s in zip(plans["pit_laps"], plans["first_sc"]) if p and not s]
    fmed = float(np.median(green)) if green else None
    rec = res.best
    rec_seq = "-".join(rec["compounds"])
    life = {r["compound"]: round(float(r["life_laps"]), 1) for _, r in res.life.iterrows()}
    return {"variant": tag, "basis": base.prior_basis, "best": res.best_label,
            "tyre_optimal": res.tyre_optimal_label, "p_stops": res.p_stops, "push": rec["push"],
            "life_laps_at_push": life, "caps": base.stint_cap, "pit_loss": round(base.pit_loss_s, 1),
            "regime": round(base.regime.ratio, 3),
            "regime_mode": ((base.regime.temperature or {}).get("mode")),
            "plan_prior_source": base.plan_prior.get("source"),
            "plan_prior_mode": ((base.plan_prior.get("nomination") or {}).get("mode")),
            "plan_prior_top": dict(list((base.plan_prior.get("sequences") or {}).items())[:4]),
            "plan_prior_starts": base.plan_prior.get("starts"),
            "objective": getattr(getattr(base, "objective", None), "label", ""),
            "objective_version": getattr(getattr(base, "objective", None), "version", "v3"),
            "race_state_in_objective": bool(sk.get("race_state") is not None),
            # V4 hands the first-stop tables to the search for the *rivals*' stop
            # laps; charged on our own lap only while kappa > 0 (V3)
            "first_stop_in_objective": bool(sk.get("first_stop_prior") and float(sk.get("first_stop_kappa_s") or 0) > 0),
            "first_stop_kappa_s": sk.get("first_stop_kappa_s"),
            "race_state_s": rec.get("race_state_s"),
            "first_stop_s": rec.get("first_stop_s"),
            "pace_calibration": {k: v for k, v in pc.items() if k in ("applied", "model_net_before", "model_net_after")},
            "mode_stops": mode_stops, "stops_match_mode": rec["n_stops"] == mode_stops,
            "stops_share": float((plans["n_stops"] == rec["n_stops"]).mean()),
            "seq_share": float(seqs.get(rec_seq, 0) / len(plans)), "field_modal_seq": seqs.index[0],
            "seq_run_by_anyone": bool(seqs.get(rec_seq, 0) > 0),
            "matches_field_modal": bool(rec_seq == seqs.index[0]),
            "start_ok": bool(plans["seq"].str.split("-").str[0].value_counts().index[0] == rec["compounds"][0]),
            "first_minus_field": ((int(rec["pit_laps"][0]) - fmed) if (fmed is not None and rec["pit_laps"] and plans["first_sc"].mean() <= 0.4) else None),
            "sealed_best": m["strategy"]["best"],
            "same_as_sealed_stops": rec["n_stops"] == m["strategy"]["best_plan"]["n_stops"],
            "seconds": round(dt, 1)}


def main() -> None:
    args = arg_events(__doc__)
    offline()
    memoise_regime()
    out, rows = {}, []
    base_v1 = baseline_out("outlook.json") or {}
    base_v2 = v2_out("outlook.json") or {}
    for key in args.events:
        ev = get_event(key)
        m = meta(key)
        plans = driver_plans(race_table(key), ev.n_race_laps)
        base = load_base(ev, force_prior=True)
        r = one(key, ev, base, plans, m, tag="prior_only")
        # the same prior with the plan shapes read letter for letter
        cp = cp_of(base)
        letters = plan_prior_for(cp, use_nominations=False)
        rl = one(key, ev, replace(base, plan_prior=letters), plans, m, tag="prior_only_letters")
        r["prior_only_letters"] = rl
        r["baseline_prior_best"] = (base_v1.get(key) or {}).get("best")
        r["baseline_prior_seq_share"] = (base_v1.get(key) or {}).get("seq_share")
        r["v2_prior_best"] = (base_v2.get(key) or {}).get("best")
        r["v2_prior_seq_share"] = (base_v2.get(key) or {}).get("seq_share")
        r["v2_prior_matches_field_modal"] = ((base_v2.get(key) or {}).get("best") is not None
                                             and (base_v2.get(key) or {}).get("seq_share") is not None
                                             and (base_v2.get(key) or {}).get("field_modal_seq")
                                             in ((base_v2.get(key) or {}).get("best") or ""))
        out[key] = r
        rows.append({"event": key, **{k: r[k] for k in ("basis", "best", "tyre_optimal", "p_stops", "mode_stops",
                                                        "stops_share", "seq_share", "matches_field_modal", "start_ok",
                                                        "first_minus_field", "sealed_best", "seconds")},
                     "letters_best": rl["best"], "letters_seq_share": rl["seq_share"],
                     "letters_matches_field_modal": rl["matches_field_modal"],
                     "letters_start_ok": rl["start_ok"],
                     "v2_prior_best": r["v2_prior_best"], "v1_prior_best": r["baseline_prior_best"]})
        print(f"{key:16s} prior-only {r['best']:28s} (tyre-opt {r['tyre_optimal']}) seq share {r['seq_share']:.0%} "
              f"modal-match {r['matches_field_modal']} start {r['start_ok']} first {r['first_minus_field']} "
              f"[{r['seconds']}s]\n{'':16s} letters    {rl['best']:28s} seq share {rl['seq_share']:.0%} "
              f"modal-match {rl['matches_field_modal']} start {rl['start_ok']} first {rl['first_minus_field']}  "
              f"| prior source {r['plan_prior_source']} ({r['plan_prior_mode']}) | sealed {m['strategy']['best']} "
              f"| V2 prior {r['v2_prior_best']} | V1 prior {r['baseline_prior_best']}", flush=True)
        dump("outlook.json", out)
    t = pd.DataFrame(rows)
    n = len(t)
    summary = {"n_events": n,
               "modal_match": int(t["matches_field_modal"].sum()), "modal_match_letters": int(t["letters_matches_field_modal"].sum()),
               "start_ok": int(t["start_ok"].sum()), "start_ok_letters": int(t["letters_start_ok"].sum()),
               "mean_seq_share": float(t["seq_share"].mean()), "mean_seq_share_letters": float(t["letters_seq_share"].mean()),
               "stops_match_mode": int(sum(bool((out[k] or {}).get("stops_match_mode")) for k in out))}
    out["_summary"] = summary
    dump("outlook.json", out)
    print(t.to_string(index=False))
    print("\nprior-only, mapped vs letter-for-letter:", summary)


if __name__ == "__main__":
    main()
