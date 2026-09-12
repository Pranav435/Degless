"""Outlook benchmark: the recommendation *before any practice has run*.

Composes the prior exactly as `src.outlook.load_base(force_prior=True)` does
on a weekend with no posterior (compound-ladder prior + the circuit's 2023-25
races pooled + transferred regime, allocation, pit loss, stint caps, the
plan-shape prior and the leave-one-out calibration), runs the same search the
sealed model runs (ladder gate enforced, position term), and compares the
answer with what the field did and with the sealed model's answer.

Caveat, stated in the report: the regime factor and pit-loss/allocation
priors pool every *other* 2026 weekend on disk, including ones raced after the
target - the honest pre-season number would be wider.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import EVENTS, baseline_out, driver_plans, dump, meta, offline, race_table  # noqa: E402
from src import strategy as strat
from src.config import get_event
from src.outlook import load_base, sim_kwargs


def main() -> None:
    offline()
    out, rows = {}, []
    base_prev = baseline_out("outlook.json") or {}
    for key in EVENTS:
        ev = get_event(key)
        t0 = time.perf_counter()
        base = load_base(ev, force_prior=True)
        net = base.net_step or {}
        model, res, pc = strat.search_with_pace_calibration(
            base.model, ev, base.pit_loss_s,
            net_step_s=float(net.get("measured") if net.get("measured") is not None else np.nan),
            net_step_se_s=float(net.get("se") or 0.0), **sim_kwargs(base))
        dt = time.perf_counter() - t0
        plans = driver_plans(race_table(key), ev.n_race_laps)
        mode_stops = int(plans["n_stops"].mode().iloc[0])
        seqs = plans["seq"].value_counts()
        green = [p[0] for p, s in zip(plans["pit_laps"], plans["first_sc"]) if p and not s]
        fmed = float(np.median(green)) if green else None
        m = meta(key)
        rec = res.best
        rec_seq = "-".join(rec["compounds"])
        life = {r["compound"]: round(float(r["life_laps"]), 1) for _, r in res.life.iterrows()}
        r = {"basis": base.prior_basis, "best": res.best_label, "tyre_optimal": res.tyre_optimal_label,
             "p_stops": res.p_stops, "push": rec["push"],
             "life_laps_at_push": life, "caps": base.stint_cap, "pit_loss": round(base.pit_loss_s, 1),
             "regime": round(base.regime.ratio, 3), "plan_prior_source": base.plan_prior.get("source"),
             "pace_calibration": {k: v for k, v in pc.items() if k in ("applied", "model_net_before", "model_net_after")},
             "mode_stops": mode_stops, "stops_match_mode": rec["n_stops"] == mode_stops,
             "stops_share": float((plans["n_stops"] == rec["n_stops"]).mean()),
             "seq_share": float(seqs.get(rec_seq, 0) / len(plans)), "field_modal_seq": seqs.index[0],
             "start_ok": bool(plans["seq"].str.split("-").str[0].value_counts().index[0] == rec["compounds"][0]),
             "first_minus_field": ((int(rec["pit_laps"][0]) - fmed) if (fmed is not None and rec["pit_laps"] and plans["first_sc"].mean() <= 0.4) else None),
             "sealed_best": m["strategy"]["best"], "same_as_sealed_stops": rec["n_stops"] == m["strategy"]["best_plan"]["n_stops"],
             "baseline_prior_best": (base_prev.get(key) or {}).get("best"),
             "baseline_prior_seq_share": (base_prev.get(key) or {}).get("seq_share"),
             "seconds": round(dt, 1)}
        out[key] = r
        rows.append({"event": key, **{k: r[k] for k in ("basis", "best", "tyre_optimal", "p_stops", "mode_stops", "stops_share",
                                                       "seq_share", "start_ok", "first_minus_field", "sealed_best",
                                                       "baseline_prior_best", "seconds")}})
        print(f"{key:16s} prior-only {res.best_label:28s} (tyre-opt {res.tyre_optimal_label}) p_stops {res.p_stops}  life {life}  "
              f"field mode {mode_stops} ({r['stops_share']:.0%}) seq share {r['seq_share']:.0%}  sealed {m['strategy']['best']}  "
              f"prev prior {r['baseline_prior_best']}  [{dt:.1f}s]", flush=True)
    dump("outlook.json", out)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
