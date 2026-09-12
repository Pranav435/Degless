"""The apex-speed channel, now the diagnostic: does it earn its cost?

For every weekend a production-setting joint fit (4x1500x1500, lap times +
apex speeds, linear, soft ladder) is scored against the race exactly like the
shipped lap-time-only practice posterior (both x regime, no circuit history),
and both are run through the shipped strategy search.  The previous
benchmark ran this comparison the other way round, when the joint fit was
the product.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import EVENTS, clean_practice, dump, meta, offline, race_table  # noqa: E402
from bench_accuracy import AGES, stint_rates, summarise
from src import strategy as strat
from src.calibration import get_calibration
from src.config import DATA_PROCESSED, get_event
from src.history import circuit_prior, plan_prior_for
from src.laps import clean_laps
from src.model_bayes import BayesFit, fit_bayes
from src.regime import RegimeFactor
from src.telemetry import load_apex, select_corners
from src.tyre import TyreModel


def main() -> None:
    offline()
    out, rows = {}, []
    for key in EVENTS:
        ev = get_event(key)
        m = meta(key)
        cal = get_calibration(ev)
        clean = clean_practice(key)
        rc = clean_laps(race_table(key))
        rg = RegimeFactor(ratio=float(m["regime"]["ratio"]), ln_sd=float(m["regime"]["ln_sd"]))
        cp = circuit_prior(ev, probe_practice_temp=False)
        ladder = cp.ladder if cp.available else None
        apex = load_apex(ev)
        apex_use = None
        if len(apex):
            apex = apex.merge(clean[["driver", "lap_number"]].drop_duplicates(), on=["driver", "lap_number"])
            if len(apex):
                sel = select_corners(apex)
                apex_use = apex[apex["corner"].isin(sel.corners)]
        t0 = time.perf_counter()
        f_joint = fit_bayes(clean, ev, prior="2026", compound_prior="soft", circuit_ladder=ladder, apex=apex_use)
        t_joint = time.perf_counter() - t0
        f_lap = BayesFit.load(DATA_PROCESSED / f"posterior_{key}_practice.npz")
        res = {}
        caps = {c: int(v) for c, v in (m.get("circuit_history") or {}).get("stint_cap", {}).items()} or None
        for tag, f in (("lap_only", f_lap), ("joint", f_joint)):
            n = f.posterior["lin"].shape[0]
            rm = rg.draws(n, seed=5)
            draws = {c: f.deg_loss(c, AGES) * rm[:, None] for c in f.compounds}
            s = summarise(stint_rates(draws, rc, ev, float(cal.sigma_race_lap_s)))
            idx = np.random.default_rng(0).choice(n, size=min(300, n), replace=False)
            model = TyreModel.from_fit(f, draws=idx, budget=cal.budgets, manage_floor=cal.manage_wear_floor,
                                       manage_cost_s=cal.manage_cost_s)
            sim = strat.simulate_model(model, ev, float(m["pit_loss_s"]), regime=rg,
                                       support={k: float(v) for k, v in m["age_support_by_compound"].items()},
                                       max_per_compound=m["allocation"]["caps"], max_stint=caps,
                                       undercut_lambda=cal.undercut_lambda, plan_prior=plan_prior_for(cp),
                                       plan_prior_tau_s=cal.plan_prior_tau_s, traffic_s_per_lap=cal.dirty_air_s_per_lap,
                                       grid_penalty_s=cal.grid_start_penalty_s)
            span = np.array([1.0, 10.0])
            rates = {c: float(((f.deg_loss(c, span)[:, 1] - f.deg_loss(c, span)[:, 0]) / 9).mean()) for c in f.compounds}
            res[tag] = {"rate_mae": s.get("rate_mae"), "rate_bias": s.get("rate_bias"), "rate_cov90": s.get("rate_cov90"),
                        "rate_width90": s.get("rate_width90"), "rates": {c: round(v, 4) for c, v in rates.items()},
                        "k_track_rel_sd": float(f.posterior["k_track"].std() / f.posterior["k_track"].mean()),
                        "rhat": float(f.max_rhat), "div": int(f.n_divergences), "best": sim.best_label,
                        "p_stops": sim.p_stops, "n_apex": int(f.n_apex)}
        res["t_fit_joint_s"] = round(t_joint, 1)
        res["t_fit_lap_only_s"] = (m.get("timings") or {}).get("fit_production_s")
        out[key] = res
        rows.append({"event": key, "mae_lap_only": res["lap_only"]["rate_mae"], "mae_joint": res["joint"]["rate_mae"],
                     "width_lap_only": res["lap_only"]["rate_width90"], "width_joint": res["joint"]["rate_width90"],
                     "ktrack_relsd_lap_only": res["lap_only"]["k_track_rel_sd"], "ktrack_relsd_joint": res["joint"]["k_track_rel_sd"],
                     "best_lap_only": res["lap_only"]["best"], "best_joint": res["joint"]["best"],
                     "t_joint_s": res["t_fit_joint_s"], "t_lap_only_s": res["t_fit_lap_only_s"]})
        print(f"{key:16s} lap-only MAE {res['lap_only']['rate_mae']:.3f} (k_track relsd {res['lap_only']['k_track_rel_sd']:.3f}) "
              f"joint MAE {res['joint']['rate_mae']:.3f} (relsd {res['joint']['k_track_rel_sd']:.3f}, fit {t_joint:.0f}s) | "
              f"best lap-only {res['lap_only']['best']} joint {res['joint']['best']}", flush=True)
        dump("apex.json", out)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
