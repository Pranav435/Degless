"""Accuracy benchmark: sealed degradation curves vs the race, against baselines.

For every scored weekend the same scorer (`src.validate.score_race`, plus a
stint-rate calibration check) is run on:

  sealed             the product: practice posterior + circuit history (rate-only
                     fold-in, capped, floored), x the temperature-modelled regime
  practice_only      the practice posterior alone, x the same regime
  practice_no_regime the practice posterior with no practice->race transfer
  sealed_geomean     the sealed curve x the *old* pooled geometric-mean regime
  sealed_perfect_t   the sealed curve x the regime with the actual race temperature
                     (what a perfect race-day forecast would have given)
  sealed_driver      the sealed curve scaled per driver by the leave-one-out race
                     factor (per-car intelligence): its value shows in the Spearman
  history_only       the circuit's 2023-25 race degradation (race regime)
  mixedlm_x_regime   the frequentist baseline slope x the transferred regime factor
  season_loo         one number per compound: the other 2026 races' mean rate
  zero               "tyres do not degrade" (the floor any model must beat)
  oracle_race        this race's own measured rate (in-sample upper bound)

Metrics per variant: stint-rate MAE and bias (s/lap), per-lap MAE (s), 90%
per-lap coverage, 90% stint-rate coverage and width, Spearman rank
correlation between predicted and observed stint rates.  The same table for
the previous benchmark's sealed files is rebuilt from bench/baseline so the
comparison is like for like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import BASELINE, EVENTS, baseline_meta, baseline_out, clean_practice, dump, meta, offline, race_table, sealed_for  # noqa: E402
from src.calibration import get_calibration
from src.config import DATA_PROCESSED, SEALED_DIR, get_event
from src.fuel import get_prior
from src.history import race_deg_slopes, race_track_temp, practice_track_temp
from src.laps import clean_laps
from src.model_bayes import BayesFit
from src.regime import RegimeFactor, regime_prior
from src.validate import _curve_block, load_sealed, score_race

AGES = np.arange(0, 41, dtype=float)


def stint_rates(draws_by_comp: dict, race_clean: pd.DataFrame, ev, sigma_obs: float,
                driver_scale: dict | None = None) -> pd.DataFrame:
    """Per-stint observed vs predicted degradation rate, with the predictive
    interval of the rate taken from the curve draws (shape (n_draws, 41)).
    `driver_scale[driver]` multiplies the predicted rate for that driver."""
    fp = get_prior(ev, "2026")
    rows = []
    for uid, g in race_clean.groupby("stint_uid"):
        g = g.sort_values("lap_number")
        comp = g["compound"].iloc[0]
        if comp not in draws_by_comp or len(g) < 8:
            continue
        ages = g["tyre_age"].to_numpy(float)
        obs = g["lap_time_s"].to_numpy(float) - fp.term(g["lap_in_stint"].to_numpy(float))
        first, last = np.argsort(ages)[:3], np.argsort(ages)[-3:]
        d_age = float(ages[last].mean() - ages[first].mean())
        if d_age < 4:
            continue
        obs_rate = float(obs[last].mean() - obs[first].mean()) / d_age
        D = draws_by_comp[comp]                          # (n, 41)
        pred = np.stack([np.interp(ages, AGES, D[i]) for i in range(D.shape[0])])
        pr = (pred[:, last].mean(1) - pred[:, first].mean(1)) / d_age
        if driver_scale:
            pr = pr * float(driver_scale.get(g["driver"].iloc[0], 1.0))
        noise = sigma_obs * np.sqrt(2.0 / 3.0) / d_age
        rows.append({"stint_uid": uid, "driver": g["driver"].iloc[0], "compound": comp,
                     "n_laps": len(g), "obs_rate": obs_rate, "pred_rate": float(pr.mean()),
                     "pred_sd": float(np.sqrt(pr.std() ** 2 + noise ** 2)),
                     "pred_lo": float(np.quantile(pr, 0.05) - 1.645 * noise),
                     "pred_hi": float(np.quantile(pr, 0.95) + 1.645 * noise)})
    return pd.DataFrame(rows)


def summarise(ps: pd.DataFrame) -> dict:
    if ps.empty:
        return {}
    err = ps["obs_rate"] - ps["pred_rate"]
    inside = ((ps["obs_rate"] >= ps["pred_lo"]) & (ps["obs_rate"] <= ps["pred_hi"])).mean()
    rho = spearmanr(ps["obs_rate"], ps["pred_rate"]).correlation if ps["pred_rate"].std() > 1e-9 else float("nan")
    return {"n_stints": int(len(ps)), "rate_mae": float(err.abs().mean()), "rate_bias": float(err.mean()),
            "rate_rmse": float(np.sqrt((err ** 2).mean())),
            "rate_cov90": float(inside), "rate_width90": float((ps["pred_hi"] - ps["pred_lo"]).mean()),
            "spearman": float(rho),
            "mae_by_compound": ps.assign(e=err.abs()).groupby("compound")["e"].mean().round(4).to_dict()}


def variants_for(key: str) -> dict:
    """name -> (draws_by_comp, sigma, driver_scale)."""
    ev = get_event(key)
    m = meta(key)
    rg = RegimeFactor(ratio=float(m["regime"]["ratio"]), ln_sd=float(m["regime"]["ln_sd"]))
    cal = get_calibration(ev)
    sig = float(cal.sigma_race_lap_s)
    out = {}

    fit_final = BayesFit.load(DATA_PROCESSED / f"posterior_{key}.npz")
    fit_prac = BayesFit.load(DATA_PROCESSED / f"posterior_{key}_practice.npz")
    n = fit_final.posterior["lin"].shape[0]
    rm = rg.draws(n, seed=5)
    comps = list(fit_final.compounds)

    out["sealed"] = ({c: fit_final.deg_loss(c, AGES) * rm[:, None] for c in comps}, sig, None)
    out["practice_only"] = ({c: fit_prac.deg_loss(c, AGES) * rm[:, None] for c in comps}, sig, None)
    out["practice_no_regime"] = ({c: fit_prac.deg_loss(c, AGES) for c in comps}, sig, None)
    # the old pooling: geometric mean over donors, no temperature model
    geo = float((m["regime"].get("temperature") or {}).get("pooled_geomean_ratio", rg.ratio))
    rg_geo = RegimeFactor(ratio=geo, ln_sd=float(m["regime"]["ln_sd"]))
    out["sealed_geomean"] = ({c: fit_final.deg_loss(c, AGES) * rg_geo.draws(n, seed=5)[:, None] for c in comps}, sig, None)
    # a perfect race-day temperature forecast (the actual race track temperature, never used by the model)
    try:
        rg_perf = regime_prior(ev, race_temp_c=race_track_temp(ev))
        out["sealed_perfect_t"] = ({c: fit_final.deg_loss(c, AGES) * rg_perf.draws(n, seed=5)[:, None] for c in comps}, sig, None)
        out["_regime_perfect_t"] = {"ratio": rg_perf.ratio, "ln_sd": rg_perf.ln_sd, "t_race": race_track_temp(ev),
                                    "t_practice": practice_track_temp(ev)}
    except Exception as exc:  # pragma: no cover
        print(f"  perfect-temperature regime unavailable for {key}: {exc}")
    # per-car: the sealed curve scaled by each driver's leave-one-out race factor
    if cal.driver_factors:
        out["sealed_driver"] = ({c: fit_final.deg_loss(c, AGES) * rm[:, None] for c in comps}, sig,
                                {d: float(f) for d, f in cal.driver_factors.items()})

    def lin(rate_by_comp):
        return {c: (AGES * float(r))[None, :] for c, r in rate_by_comp.items() if np.isfinite(r)}

    hist = (m.get("circuit_history") or {}).get("rate_prior") or {}
    if hist:
        out["history_only"] = (lin({c: v["mean_s_per_lap"] for c, v in hist.items()}), sig, None)
    mlm = m["mixedlm"]["slopes"]
    out["mixedlm_x_regime"] = (lin({c: max(mlm.get(c, np.nan), 0.0) * rg.ratio for c in comps}), sig, None)

    loo = {}
    for other in EVENTS:
        if other == key:
            continue
        r = race_table(other)
        r = r[r["is_accurate"] & ~r["pit_in"] & ~r["pit_out"] & (r["track_status"].astype(str) == "1")]
        d = race_deg_slopes(r, get_event(other).fuel_effect_s_per_lap)
        for c, v in d.items():
            if v["slope"] > 0.003:
                loo.setdefault(c, []).append(np.log(v["slope"]))
    out["season_loo"] = (lin({c: float(np.exp(np.mean(v))) for c, v in loo.items()}), sig, None)
    out["zero"] = (lin({c: 0.0 for c in comps}), sig, None)

    r = race_table(key)
    r = r[r["is_accurate"] & ~r["pit_in"] & ~r["pit_out"] & (r["track_status"].astype(str) == "1")]
    orc = race_deg_slopes(r, ev.fuel_effect_s_per_lap)
    out["oracle_race"] = (lin({c: v["slope"] for c, v in orc.items()}), sig, None)
    out["_oracle_rates"] = {c: v["slope"] for c, v in orc.items()}
    out["_loo_rates"] = {c: float(np.exp(np.mean(v))) for c, v in loo.items()}
    return out


def sealed_dict(draws_by_comp: dict, sigma: float, knee: dict, cliff: dict | None = None) -> dict:
    return {"ages": AGES.tolist(), "race_curves": {c: _curve_block(d) for c, d in draws_by_comp.items()},
            "curves": {}, "sigma_obs": sigma, "sigma_race": sigma, "knee": knee, "cliff_history": cliff or {}}


def baseline_rows(key: str, rc: pd.DataFrame, ev, sig_new: float) -> list:
    """The previous benchmark's sealed and practice-only curves, rescored with
    both the old (practice sigma) and the new (race sigma) noise, from the
    frozen baseline posteriors."""
    bm = baseline_meta(key)
    if bm is None:
        return []
    rows = []
    try:
        fit_b = BayesFit.load(BASELINE / "processed" / f"posterior_{key}.npz")
        fit_bp = BayesFit.load(BASELINE / "processed" / f"posterior_{key}_practice.npz")
    except Exception as exc:
        print(f"  baseline posteriors unavailable for {key}: {exc}")
        return []
    rg = RegimeFactor(ratio=float(bm["regime"]["ratio"]), ln_sd=float(bm["regime"]["ln_sd"]))
    n = fit_b.posterior["lin"].shape[0]
    rm = rg.draws(n, seed=5)
    sig_old = float(fit_b.posterior["sigma_obs"].mean())
    for tag, f in (("baseline_sealed", fit_b), ("baseline_practice_only", fit_bp)):
        draws = {c: f.deg_loss(c, AGES) * rm[:, None] for c in f.compounds}
        for sname, sig in (("old_sigma", sig_old), ("race_sigma", sig_new)):
            sc = score_race(sealed_dict(draws, sig, {}), rc, ev)
            s = summarise(stint_rates(draws, rc, ev, sig))
            s.update({"scorer_mae": float(sc.mae), "scorer_bias": float(sc.bias), "mae_lap": float(sc.mae_lap),
                      "lap_cov90": float(sc.coverage.get(0.9, np.nan)), "n_laps": int(sc.n_laps), "sigma": sig})
            rows.append({"event": key, "variant": f"{tag}[{sname}]", **{k: s.get(k) for k in
                         ("n_stints", "rate_mae", "rate_bias", "rate_cov90", "rate_width90", "spearman", "mae_lap", "lap_cov90")}})
    return rows


def main() -> None:
    offline()
    results = {}
    rows, brows = [], []
    for key in EVENTS:
        ev = get_event(key)
        m = meta(key)
        race = race_table(key)
        rc = clean_laps(race)
        v = variants_for(key)
        cliff = m.get("cliff_history") or {}
        sc_file = score_race(sealed_for(key, m), rc, ev)
        res = {"n_race_laps": ev.n_race_laps, "sealed_file_mae": sc_file.mae, "meta_mae": m["score"]["mae"],
               "sealed_rate_cov90": float(sc_file.rate_coverage90), "sealed_rate_width90": float(sc_file.rate_width90),
               "oracle_rates": v.pop("_oracle_rates"), "loo_rates": v.pop("_loo_rates"),
               "regime": {"ratio": m["regime"]["ratio"], "ln_sd": m["regime"]["ln_sd"],
                          "self": (m["regime"].get("self_measured") or {}).get("ratio"),
                          "geomean": (m["regime"].get("temperature") or {}).get("pooled_geomean_ratio"),
                          "perfect_t": v.pop("_regime_perfect_t", None)},
               "variants": {}}
        for name, (draws, sig, dscale) in v.items():
            sc = score_race(sealed_dict(draws, sig, {}, cliff), rc, ev)
            ps = stint_rates(draws, rc, ev, sig, driver_scale=dscale)
            s = summarise(ps)
            s.update({"scorer_mae": float(sc.mae), "scorer_bias": float(sc.bias), "mae_lap": float(sc.mae_lap),
                      "lap_cov90": float(sc.coverage.get(0.9, np.nan)), "lap_cov50": float(sc.coverage.get(0.5, np.nan)),
                      "n_laps": int(sc.n_laps), "sigma": sig})
            res["variants"][name] = s
            rows.append({"event": key, "variant": name, **{k: s.get(k) for k in
                         ("n_stints", "rate_mae", "rate_bias", "rate_cov90", "rate_width90", "spearman",
                          "mae_lap", "lap_cov90")}})
            print(f"{key:16s} {name:18s} stints {s.get('n_stints', 0):3d}  rateMAE {s.get('rate_mae', float('nan')):.3f}"
                  f"  bias {s.get('rate_bias', float('nan')):+.3f}  cov90 {s.get('rate_cov90', float('nan')):.2f}"
                  f"  width {s.get('rate_width90', float('nan')):.3f}  rho {s.get('spearman', float('nan')):+.2f}"
                  f"  lapMAE {s['mae_lap']:.2f}  lapcov90 {s['lap_cov90']:.2f}", flush=True)
        brows += baseline_rows(key, rc, ev, float(get_calibration(ev).sigma_race_lap_s))
        results[key] = res
    tbl = pd.DataFrame(rows)
    tbl.to_csv(dump("accuracy_table.json", []).with_suffix(".csv"), index=False)
    btbl = pd.DataFrame(brows)
    if not btbl.empty:
        btbl.to_csv(dump("accuracy_baseline_table.json", []).with_suffix(".csv"), index=False)
    pooled = (pd.concat([tbl, btbl], ignore_index=True).groupby("variant")
              .agg(rate_mae_mean=("rate_mae", "mean"), rate_mae_median=("rate_mae", "median"),
                   rate_mae_max=("rate_mae", "max"), bias_mean=("rate_bias", "mean"),
                   cov90_mean=("rate_cov90", "mean"), width_mean=("rate_width90", "mean"),
                   spearman_mean=("spearman", "mean"), lap_mae_mean=("mae_lap", "mean"),
                   n_pass=("rate_mae", lambda s: int((s < 0.15).sum())))
              .sort_values("rate_mae_mean"))
    print("\n=== pooled over weekends ===")
    print(pooled.round(3).to_string())
    dump("accuracy.json", {"per_event": results, "pooled": pooled.reset_index().to_dict("records")})
    print("\nlaps-weighted sealed rate MAE:",
          round(float(np.average(tbl[tbl.variant == "sealed"]["rate_mae"], weights=tbl[tbl.variant == "sealed"]["n_stints"])), 4))


if __name__ == "__main__":
    main()
