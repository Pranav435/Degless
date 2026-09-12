"""Accuracy benchmark: sealed degradation curves vs the race, against baselines.

For every scored weekend the same scorer (`src.validate.score_race`, plus a
stint-rate calibration check) is run on:

  sealed / sealed_v3    the product: practice posterior + circuit history
                        (rate-only fold-in, capped, floored), x the shipped
                        regime factor.  `sealed` is V2's key, kept so the
                        frozen comparison keeps working; `sealed_v3` is an
                        alias with the same numbers and the name the V3 report
                        uses
  practice_only         the practice posterior alone, x the same regime
  practice_no_regime    the practice posterior with no practice->race transfer

  the regime family     each one re-folds the *practice* posterior with
                        `history.apply_circuit_prior` under its own regime and
                        rescales by that regime's draws, so the fold-in and the
                        scaling always agree:
  regime_v2_temperature   V2: the archive race-day mean as the forecast
  regime_v3_pooled        donor median, no circuit history
  regime_v3_circuit       donor median + this circuit's own practice->race
                          history (what V3 ships: must reproduce `sealed`)
  regime_oracle_temperature  the actual race temperature as the forecast

  the per-car family    the sealed curve scaled per driver, `percar.rate_scale_table`:
  sealed_driver           V2's key: the leave-one-out race factors as shipped
  sealed_driver_hist      the same factors through `rate_scale_table("hist")`
  sealed_driver_practice_dev  this weekend's own `dev[d, c]` where it exists
  sealed_driver_team_pooled   the same, pooled over the team (V3 ships this)
  sealed_driver_hier      V4/WP-E: the hierarchical model's own scale
                          (`haascar.hier_rate_scale_table`) - the team pooling
                          weighted by each car's real clean-lap count, times
                          the precision-shrunk race-history prior.  It is the
                          scale the shipped per-car path applies, and it is
                          scored through the same code path as the others

Plus a Haas-only block (`per_event[key]["haas"]`, pooled under `haas`): OCO's
and BEA's stint-rate MAE and bias under `none`, `team_pooled` and `hier`.  Two
cars is not a population - a Spearman correlation over two points is not a
number - so the Haas block reports errors only, per car and per weekend.

  history_only          the circuit's 2023-25 race degradation (race regime)
  mixedlm_x_regime      the frequentist baseline slope x the regime factor
  season_loo            one number per compound: the other 2026 races' mean rate
  zero                  "tyres do not degrade" (the floor any model must beat)
  oracle_race           this race's own measured rate (in-sample upper bound)

Metrics per variant: stint-rate MAE and bias (s/lap), per-lap MAE (s), 90%
per-lap coverage, 90% *and* 95% stint-rate coverage and width, Spearman rank
correlation between predicted and observed stint rates.  The same table is
rebuilt from V1's frozen posteriors (`bench/baseline`) and V2's
(`bench/v2/processed`), so all three builds are scored by identical code.

The 196-stint population must not move between builds: the per-weekend
`n_stints` of the `sealed` variant and their sum are printed and written to
`accuracy.json` under `population`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (BASELINE, EVENTS, V2, arg_events, baseline_meta, clean_practice, cp_for,  # noqa: E402
                    dump, memoise_regime, meta, offline, race_table, sealed_for, v2_meta,
                    v2_processed)
from src.calibration import get_calibration
from src.config import DATA_PROCESSED, get_event
from src.fuel import get_prior
from src.history import apply_circuit_prior, race_deg_slopes, race_track_temp, practice_track_temp
from src.laps import clean_laps
from src.model_bayes import BayesFit
from src.regime import RegimeFactor, regime_prior
from src.validate import _curve_block, score_race

AGES = np.arange(0, 41, dtype=float)

# The per-car variants and the `rate_scale_table` kind each one asks for.
PERCAR_VARIANTS = {"sealed_driver_hist": "hist",
                   "sealed_driver_practice_dev": "practice_dev",
                   "sealed_driver_team_pooled": "team_pooled"}

REPORT_KEYS = ("n_stints", "rate_mae", "rate_bias", "rate_cov90", "rate_cov95", "rate_width90",
               "rate_width95", "spearman", "mae_lap", "lap_cov90")


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
                     "pred_hi": float(np.quantile(pr, 0.95) + 1.645 * noise),
                     "pred_lo95": float(np.quantile(pr, 0.025) - 1.960 * noise),
                     "pred_hi95": float(np.quantile(pr, 0.975) + 1.960 * noise)})
    return pd.DataFrame(rows)


def summarise(ps: pd.DataFrame) -> dict:
    if ps.empty:
        return {}
    err = ps["obs_rate"] - ps["pred_rate"]
    inside = ((ps["obs_rate"] >= ps["pred_lo"]) & (ps["obs_rate"] <= ps["pred_hi"])).mean()
    in95 = ((ps["obs_rate"] >= ps["pred_lo95"]) & (ps["obs_rate"] <= ps["pred_hi95"])).mean() \
        if "pred_lo95" in ps else float("nan")
    rho = spearmanr(ps["obs_rate"], ps["pred_rate"]).correlation if ps["pred_rate"].std() > 1e-9 else float("nan")
    return {"n_stints": int(len(ps)), "rate_mae": float(err.abs().mean()), "rate_bias": float(err.mean()),
            "rate_rmse": float(np.sqrt((err ** 2).mean())),
            "rate_cov90": float(inside), "rate_width90": float((ps["pred_hi"] - ps["pred_lo"]).mean()),
            "rate_cov95": float(in95),
            "rate_width95": (float((ps["pred_hi95"] - ps["pred_lo95"]).mean()) if "pred_lo95" in ps else float("nan")),
            "spearman": float(rho),
            "mae_by_compound": ps.assign(e=err.abs()).groupby("compound")["e"].mean().round(4).to_dict()}


def _teams(key: str) -> dict:
    r = race_table(key)
    if "team" not in r:
        return {}
    return r.drop_duplicates("driver").set_index("driver")["team"].astype(str).to_dict()


def _regime_variants(ev, m: dict, cp) -> dict:
    """name -> RegimeFactor, for the four regime ablations plus the shipped one.

    `regime_prior` is the same function the pipeline calls; only its keywords
    differ.  A variant that raises (no donors on disk, no circuit-regime cache)
    is dropped with a printed reason rather than failing the weekend.
    """
    out = {"regime_shipped": RegimeFactor(ratio=float(m["regime"]["ratio"]), ln_sd=float(m["regime"]["ln_sd"]))}
    asks = {"regime_v2_temperature": dict(temperature_model=True, use_circuit_history=False),
            "regime_v3_pooled": dict(use_circuit_history=False),
            "regime_v3_circuit": dict()}
    try:
        asks["regime_oracle_temperature"] = dict(race_temp_c=race_track_temp(ev))
    except Exception as exc:
        print(f"  race temperature unavailable for {ev.key}: {exc}")
    for name, kw in asks.items():
        try:
            out[name] = regime_prior(ev, **kw)
        except Exception as exc:
            print(f"  {name} unavailable for {ev.key}: {exc}")
    return out


def _refold(fit_prac, ev, rg, cp):
    """The practice posterior re-folded with the circuit history under `rg`.

    Every regime variant has to re-run the fold-in: `apply_circuit_prior`
    compares practice and history *in the race regime* and the regime factor is
    the link between them, so scoring a differently-scaled regime against a
    fold-in done with the shipped one would measure the mismatch, not the
    variant.
    """
    if not (getattr(cp, "available", False) and getattr(cp, "rate_prior", None)):
        return fit_prac
    new, _ = apply_circuit_prior(fit_prac, ev, rg, cp)
    return new


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
    cp = cp_for(key, m)

    sealed_draws = {c: fit_final.deg_loss(c, AGES) * rm[:, None] for c in comps}
    out["sealed"] = (sealed_draws, sig, None)
    out["sealed_v3"] = (sealed_draws, sig, None)          # the V3 report's name for the same curve
    out["practice_only"] = ({c: fit_prac.deg_loss(c, AGES) * rm[:, None] for c in comps}, sig, None)
    out["practice_no_regime"] = ({c: fit_prac.deg_loss(c, AGES) for c in comps}, sig, None)
    # the old pooling: geometric mean over donors, no temperature model
    geo = float((m["regime"].get("temperature") or {}).get("pooled_geomean_ratio", rg.ratio))
    rg_geo = RegimeFactor(ratio=geo, ln_sd=float(m["regime"]["ln_sd"]))
    out["sealed_geomean"] = ({c: fit_final.deg_loss(c, AGES) * rg_geo.draws(n, seed=5)[:, None] for c in comps}, sig, None)

    # -- the regime family: re-fold the practice posterior under each regime ----
    regs = _regime_variants(ev, m, cp)
    reg_detail, np_ = {}, fit_prac.posterior["lin"].shape[0]
    for name, rgv in regs.items():
        if name == "regime_shipped":
            continue
        f = _refold(fit_prac, ev, rgv, cp)
        draws = {c: f.deg_loss(c, AGES) * rgv.draws(np_, seed=5)[:, None] for c in f.compounds}
        out[name] = (draws, sig, None)
        reg_detail[name] = {"ratio": float(rgv.ratio), "ln_sd": float(rgv.ln_sd),
                            "mode": (rgv.temperature or {}).get("mode"),
                            "label": rgv.label,
                            "circuit_prior": bool((rgv.temperature or {}).get("circuit_prior"))}
    # V2's name for the perfect-forecast variant, kept so the frozen tables line up
    if "regime_oracle_temperature" in out:
        out["sealed_perfect_t"] = out["regime_oracle_temperature"]
        reg_detail["sealed_perfect_t"] = dict(reg_detail["regime_oracle_temperature"],
                                              t_race=race_track_temp(ev), t_practice=practice_track_temp(ev))
    # Does the shipped curve equal the regime variant V3 claims to ship?
    chk = {"checked": False}
    if "regime_v3_circuit" in out:
        a, b = sealed_draws, out["regime_v3_circuit"][0]
        shared = [c for c in a if c in b]
        if shared:
            d = max(float(np.abs(a[c].mean(0) - b[c].mean(0)).max()) for c in shared)
            chk = {"checked": True, "max_abs_diff_s": d, "matches": bool(d < 5e-3),
                   "shipped_ratio": float(rg.ratio),
                   "variant_ratio": float(regs["regime_v3_circuit"].ratio)}
            if not chk["matches"]:
                print(f"  NOTE {key}: regime_v3_circuit does not reproduce the shipped curve "
                      f"(max |diff| {d:.4f} s on the mean curve; shipped regime {rg.ratio:.3f} vs "
                      f"variant {regs['regime_v3_circuit'].ratio:.3f}) — expected while the "
                      f"artefacts on disk are V2's")
    out["_regime_detail"] = reg_detail
    out["_regime_v3_circuit_check"] = chk
    out["_regime_perfect_t"] = {"ratio": regs.get("regime_oracle_temperature", rg).ratio,
                                "ln_sd": regs.get("regime_oracle_temperature", rg).ln_sd,
                                "t_race": race_track_temp(ev), "t_practice": practice_track_temp(ev)}

    # -- the per-car family ----------------------------------------------------
    if cal.driver_factors:
        out["sealed_driver"] = (sealed_draws, sig, {d: float(f) for d, f in cal.driver_factors.items()})
    teams = _teams(key)
    haas_scales = {"none": None}
    try:
        from src.percar import rate_scale_table
        scales = {}
        for name, kind in PERCAR_VARIANTS.items():
            tbl = rate_scale_table(kind, fit=fit_final, cal=cal, teams=teams)
            if tbl and any(abs(v - 1.0) > 1e-9 for v in tbl.values()):
                out[name] = (sealed_draws, sig, {d: float(v) for d, v in tbl.items()})
            scales[name] = {"n": len(tbl or {}),
                            "range": ([round(min(tbl.values()), 3), round(max(tbl.values()), 3)] if tbl else None)}
            if kind == "team_pooled":
                haas_scales["team_pooled"] = ({d: float(v) for d, v in tbl.items()} if tbl else None)
        # -- V4/WP-E: the hierarchical model's own scale ---------------------
        # The lap counts are the ones the pipeline pools with
        # (`clean.groupby("driver").size()`), so this is the scale the shipped
        # per-car model applies rather than an equal-weight approximation of it.
        try:
            from src.haascar import hier_rate_scale_table
            n_laps = clean_practice(key).groupby("driver").size().to_dict()
            tbl = hier_rate_scale_table(fit=fit_final, cal=cal, teams=teams,
                                        n_laps_by_driver={str(k): float(v) for k, v in n_laps.items()})
            if tbl and any(abs(v - 1.0) > 1e-9 for v in tbl.values()):
                out["sealed_driver_hier"] = (sealed_draws, sig, {d: float(v) for d, v in tbl.items()})
            haas_scales["hier"] = ({d: float(v) for d, v in tbl.items()} if tbl else None)
            scales["sealed_driver_hier"] = {
                "n": len(tbl or {}),
                "range": ([round(min(tbl.values()), 3), round(max(tbl.values()), 3)] if tbl else None),
                "n_laps_by_driver": {str(k): int(v) for k, v in n_laps.items()},
                "definition": ("team-pooled dev at the driver's own clean-lap count, times the "
                               "precision-shrunk race-history factor (percar.shrink_factor)")}
        except Exception as exc:
            print(f"  sealed_driver_hier unavailable for {key}: {exc}")
        out["_percar_scales"] = scales
    except ImportError:
        print(f"  src.percar unavailable: the per-car variants are skipped for {key}")
    out["_haas_scales"] = haas_scales

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


def haas_rows(sealed_draws: dict, rc: pd.DataFrame, ev, sig: float, scales: dict) -> dict:
    """OCO's and BEA's own stint-rate error under the three per-car scalings.

    The same `stint_rates` call the field variants are scored with, restricted
    to the two Haas cars afterwards, so the number is comparable to the pooled
    one line for line.  No Spearman: it would be a rank correlation over two
    stint populations of three or four stints each, which is not a measurement.
    """
    from src.haascar import HAAS_DRIVERS

    out = {"drivers": list(HAAS_DRIVERS), "variants": {},
           "note": ("stint-rate error for the two Haas cars only; a Spearman correlation over "
                    "two cars is not meaningful and is not reported")}
    for name in ("none", "team_pooled", "hier"):
        if name not in scales:
            continue
        scale = scales[name]
        ps = stint_rates(sealed_draws, rc, ev, sig, driver_scale=(scale or None))
        if ps.empty:
            continue
        h = ps[ps["driver"].isin(HAAS_DRIVERS)]
        err = h["obs_rate"] - h["pred_rate"]
        out["variants"][name] = {
            "n_stints": int(len(h)),
            "rate_mae": (float(err.abs().mean()) if len(h) else None),
            "rate_bias": (float(err.mean()) if len(h) else None),
            "scale": {d: round(float((scale or {}).get(d, 1.0)), 4) for d in HAAS_DRIVERS},
            "by_driver": {d: {"n_stints": int((h["driver"] == d).sum()),
                              "rate_mae": (float(err[h["driver"] == d].abs().mean())
                                           if (h["driver"] == d).any() else None),
                              "rate_bias": (float(err[h["driver"] == d].mean())
                                            if (h["driver"] == d).any() else None),
                              "obs_rate": {c: round(float(x), 4) for c, x in
                                           h[h["driver"] == d].set_index("compound")["obs_rate"].items()},
                              "pred_rate": {c: round(float(x), 4) for c, x in
                                            h[h["driver"] == d].set_index("compound")["pred_rate"].items()}}
                          for d in HAAS_DRIVERS},
        }
    return out


def sealed_dict(draws_by_comp: dict, sigma: float, knee: dict, cliff: dict | None = None) -> dict:
    return {"ages": AGES.tolist(), "race_curves": {c: _curve_block(d) for c, d in draws_by_comp.items()},
            "curves": {}, "sigma_obs": sigma, "sigma_race": sigma, "knee": knee, "cliff_history": cliff or {}}


def frozen_rows(tag: str, root, m_frozen: dict | None, key: str, rc: pd.DataFrame, ev,
                sig_new: float) -> list:
    """A frozen build's sealed and practice-only curves, re-scored by this code.

    Both noise settings are reported: the one that build used (its own practice
    `sigma_obs`) and the race noise every current variant is scored with, so the
    V1 -> V2 -> V3 column is a like-for-like comparison and the interval widths
    are not an artefact of the scorer changing under them.
    """
    if m_frozen is None:
        return []
    rows = []
    try:
        fit_b = BayesFit.load(root / f"posterior_{key}.npz")
        fit_bp = BayesFit.load(root / f"posterior_{key}_practice.npz")
    except Exception as exc:
        print(f"  {tag} posteriors unavailable for {key}: {exc}")
        return []
    rg = RegimeFactor(ratio=float(m_frozen["regime"]["ratio"]), ln_sd=float(m_frozen["regime"]["ln_sd"]))
    n = fit_b.posterior["lin"].shape[0]
    rm = rg.draws(n, seed=5)
    sig_old = float(fit_b.posterior["sigma_obs"].mean())
    for name, f in ((f"{tag}_sealed", fit_b), (f"{tag}_practice_only", fit_bp)):
        draws = {c: f.deg_loss(c, AGES) * rm[:, None] for c in f.compounds}
        for sname, sig in (("old_sigma", sig_old), ("race_sigma", sig_new)):
            sc = score_race(sealed_dict(draws, sig, {}), rc, ev)
            s = summarise(stint_rates(draws, rc, ev, sig))
            s.update({"scorer_mae": float(sc.mae), "scorer_bias": float(sc.bias), "mae_lap": float(sc.mae_lap),
                      "lap_cov90": float(sc.coverage.get(0.9, np.nan)), "n_laps": int(sc.n_laps), "sigma": sig})
            rows.append({"event": key, "variant": f"{name}[{sname}]", **{k: s.get(k) for k in REPORT_KEYS}})
    return rows


def main() -> None:
    args = arg_events(__doc__)
    offline()
    memoise_regime()
    results = {}
    rows, brows = [], []
    for key in args.events:
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
                          "mode": ((m["regime"].get("temperature") or {}).get("mode")),
                          "perfect_t": v.pop("_regime_perfect_t", None),
                          "variants": v.pop("_regime_detail", {})},
               "regime_v3_circuit_check": v.pop("_regime_v3_circuit_check", {}),
               "percar_scales": v.pop("_percar_scales", {}),
               "first_stop_prior": m.get("first_stop_prior"),
               "variants": {}}
        haas_scales = v.pop("_haas_scales", {})
        for name, (draws, sig, dscale) in v.items():
            sc = score_race(sealed_dict(draws, sig, {}, cliff), rc, ev)
            ps = stint_rates(draws, rc, ev, sig, driver_scale=dscale)
            s = summarise(ps)
            s.update({"scorer_mae": float(sc.mae), "scorer_bias": float(sc.bias), "mae_lap": float(sc.mae_lap),
                      "lap_cov90": float(sc.coverage.get(0.9, np.nan)), "lap_cov50": float(sc.coverage.get(0.5, np.nan)),
                      "n_laps": int(sc.n_laps), "sigma": sig})
            res["variants"][name] = s
            rows.append({"event": key, "variant": name, **{k: s.get(k) for k in REPORT_KEYS}})
            print(f"{key:16s} {name:26s} stints {s.get('n_stints', 0):3d}  rateMAE {s.get('rate_mae', float('nan')):.3f}"
                  f"  bias {s.get('rate_bias', float('nan')):+.3f}  cov90 {s.get('rate_cov90', float('nan')):.2f}"
                  f"  cov95 {s.get('rate_cov95', float('nan')):.2f}"
                  f"  width {s.get('rate_width90', float('nan')):.3f}  rho {s.get('spearman', float('nan')):+.2f}"
                  f"  lapMAE {s['mae_lap']:.2f}  lapcov90 {s['lap_cov90']:.2f}", flush=True)
        # -- the two Haas cars on their own ----------------------------------
        try:
            res["haas"] = haas_rows(v["sealed"][0], rc, ev, v["sealed"][1], haas_scales)
            for name, blk in (res["haas"].get("variants") or {}).items():
                per = ", ".join(f"{d} {b['rate_mae']:.3f} (n={b['n_stints']}, x{blk['scale'][d]:.3f})"
                                if b["rate_mae"] is not None else f"{d} -"
                                for d, b in blk["by_driver"].items())
                # A weekend can leave the two cars with no scoreable stint at all
                # (Japan: neither car's stints clear `stint_rates`' 8-lap / 4-lap-of-age
                # bar), and that is a fact to print, not an exception to raise.
                mae = ("-" if blk["rate_mae"] is None else f"{blk['rate_mae']:.3f}")
                bias = ("-" if blk["rate_bias"] is None else f"{blk['rate_bias']:+.3f}")
                print(f"{key:16s} haas[{name:11s}] stints {blk['n_stints']:2d}  "
                      f"rateMAE {mae}  bias {bias}  {per}", flush=True)
        except Exception as exc:
            print(f"  haas block unavailable for {key}: {exc}")
        sig_new = float(get_calibration(ev).sigma_race_lap_s)
        brows += frozen_rows("baseline", BASELINE / "processed", baseline_meta(key), key, rc, ev, sig_new)
        brows += frozen_rows("v2", V2 / "processed", v2_meta(key), key, rc, ev, sig_new)
        results[key] = res
    tbl = pd.DataFrame(rows)
    tbl.to_csv(dump("accuracy_table.json", []).with_suffix(".csv"), index=False)
    btbl = pd.DataFrame(brows)
    if not btbl.empty:
        btbl.to_csv(dump("accuracy_baseline_table.json", []).with_suffix(".csv"), index=False)
    pooled = (pd.concat([tbl, btbl], ignore_index=True).groupby("variant")
              .agg(rate_mae_mean=("rate_mae", "mean"), rate_mae_median=("rate_mae", "median"),
                   rate_mae_max=("rate_mae", "max"), bias_mean=("rate_bias", "mean"),
                   cov90_mean=("rate_cov90", "mean"), cov95_mean=("rate_cov95", "mean"),
                   width_mean=("rate_width90", "mean"), width95_mean=("rate_width95", "mean"),
                   spearman_mean=("spearman", "mean"), lap_mae_mean=("mae_lap", "mean"),
                   n_pass=("rate_mae", lambda s: int((s < 0.15).sum())))
              .sort_values("rate_mae_mean"))
    print("\n=== pooled over weekends ===")
    print(pooled.round(3).to_string())

    # -- the population must not move between builds --------------------------
    sealed_n = {r["event"]: int(r["n_stints"] or 0) for r in rows if r["variant"] == "sealed"}
    population = {"per_event": sealed_n, "total": int(sum(sealed_n.values())),
                  "expected_total_all_seven": 196,
                  "complete": bool(sorted(sealed_n) == sorted(EVENTS))}
    print("\nsealed n_stints per weekend:", sealed_n, "-> total", population["total"],
          ("(all seven; V2 scored 196)" if population["complete"] else "(subset of weekends)"))
    checks = {k: results[k].get("regime_v3_circuit_check") for k in results}
    bad = [k for k, c in checks.items() if c.get("checked") and not c.get("matches")]
    if bad:
        print(f"regime_v3_circuit differs from the shipped curve on {bad} "
              f"(max |diff| { {k: round(checks[k]['max_abs_diff_s'], 4) for k in bad} })")
    # The frozen builds' per-weekend rows, re-scored by this code, so
    # `bench_compare` (and the report) can read V1 -> V2 -> V3 out of this one
    # JSON without reopening the CSVs or the frozen accuracy.json files.
    frozen = {}
    for r in brows:
        frozen.setdefault(r["variant"], {})[r["event"]] = {k: r.get(k) for k in REPORT_KEYS}
    # -- the Haas summary, pooled over the weekends that have one -------------
    haas = {"per_event": {k: results[k].get("haas") for k in results if results[k].get("haas")},
            "pooled": {}}
    for name in ("none", "team_pooled", "hier"):
        rows_h = [(k, b["variants"][name]) for k, b in haas["per_event"].items()
                  if name in (b.get("variants") or {})]
        if not rows_h:
            continue
        mae = [b["rate_mae"] for _, b in rows_h if b["rate_mae"] is not None]
        n = sum(b["n_stints"] for _, b in rows_h)
        haas["pooled"][name] = {
            "n_weekends": len(rows_h), "n_stints": int(n),
            "rate_mae_mean": (float(np.mean(mae)) if mae else None),
            "rate_mae_weighted": (float(np.average(mae, weights=[b["n_stints"] for _, b in rows_h
                                                                 if b["rate_mae"] is not None]))
                                  if mae else None),
            "bias_mean": float(np.mean([b["rate_bias"] for _, b in rows_h
                                        if b["rate_bias"] is not None])) if mae else None,
            "by_driver": {d: {"rate_mae_mean": float(np.mean(
                [b["by_driver"][d]["rate_mae"] for _, b in rows_h
                 if b["by_driver"].get(d, {}).get("rate_mae") is not None]))
                if any(b["by_driver"].get(d, {}).get("rate_mae") is not None for _, b in rows_h)
                else None,
                "n_stints": int(sum(b["by_driver"].get(d, {}).get("n_stints", 0) for _, b in rows_h))}
                for d in ("OCO", "BEA")},
        }
    if haas["pooled"]:
        print("\n=== Haas only (two cars; no Spearman) ===")
        for name, b in haas["pooled"].items():
            per = ", ".join(f"{d} {v['rate_mae_mean']:.3f} (n={v['n_stints']})"
                            if v["rate_mae_mean"] is not None else f"{d} -"
                            for d, v in b["by_driver"].items())
            print(f"  {name:11s} weekends {b['n_weekends']} stints {b['n_stints']:2d}  "
                  f"rateMAE mean {b['rate_mae_mean']:.4f}  laps-weighted {b['rate_mae_weighted']:.4f}  "
                  f"bias {b['bias_mean']:+.4f}  [{per}]")
    dump("accuracy.json", {"per_event": results, "pooled": pooled.reset_index().to_dict("records"),
                           "frozen_per_event": frozen, "population": population,
                           "regime_v3_circuit_check": checks, "haas": haas})
    sl = tbl[tbl.variant == "sealed"]
    if len(sl):
        print("\nlaps-weighted sealed rate MAE:",
              round(float(np.average(sl["rate_mae"], weights=sl["n_stints"])), 4))


if __name__ == "__main__":
    main()
