"""How much worse does the practice fit transfer when the stint runs past the
practice support?

This is the measurement WP-B's mechanism is scaled from, and it is taken before
the mechanism is written so that the scale is a number and not a preference.

**The question.**  A practice long run stops at age 14-25 laps.  Every stint the
strategy search plans beyond that age is priced off an *extrapolation* of the
fitted curve, and the search currently believes the extrapolated rate exactly as
firmly as the interpolated one.  If the practice->race transfer error grows with
how far past the support the stint runs, that growth is the missing uncertainty,
and its size is `extrap_ln_sd`.

**What is measured.**  For every race stint the pipeline already scored
(`data/processed/scorestint_<key>.parquet`, written by `src.validate.score_race`)
the scorer holds two numbers: `obs_rate`, the stint's measured degradation rate
from the mean of its first three laps to the mean of its last three, and
`pred_rate`, the same quantity read off the *sealed* race curve at the same ages.
Their log ratio

        e = ln(obs_rate / pred_rate)

is the transfer error on the scale the model's wear rate lives on - a
multiplicative error on a rate - which is the scale the widening has to be in.
Nothing here re-derives a rate: the comparison is the scorer's own, at the
scorer's push and regime, so the number cannot drift from the one on the
scorecard.

**The regressor** is the stint's extrapolation ratio

        r = (highest tyre age the stint reached) / (practice support for its
            compound, `fitstage_<key>.json["age_support_by_compound"]`)

so r <= 1 is a stint the practice data covers and r = 2 is a stint run to twice
the age practice ever saw.

**The estimator.**  Binning (what the plan asks for) is reported, but the scale
the mechanism needs is the *growth* of the dispersion, and the mechanism adds it
in quadrature to whatever the posterior already carries.  So the headline
estimate is the maximum-likelihood fit of exactly that shape,

        e_i ~ Normal(mu,  sigma_0^2 + a^2 * max(0, r_i - 1)^2 + nu_i^2)

with `a` = `extrap_ln_sd` (the extra log-sd at 2x support), `sigma_0` the
dispersion a stint inside the support already has, and `nu_i` the *measurement*
noise of the scorer's own rate estimate - two three-lap means of a racing lap,
`sigma_race * sqrt(2/3) / age_span`, divided by the rate to put it on the log
scale.  Deflating by `nu_i` matters: without it the fit charges the model for
the noise in the yardstick.

**The rule, fixed before the run** (WP-B's plan): if the dispersion grows with
the extrapolation, `extrap_ln_sd` is the measured growth `a`; if it does not,
`extrap_ln_sd` is the pooled transfer dispersion, charged at 2x support, and
the report says so.  On these seven weekends it does not grow - `a` comes out
at the 0 boundary with a 95% profile interval of 0-0.43 and the binned trend
is *negative* - so the fallback applies and the value is the pooled dispersion
deflated by the yardstick noise.  The report has to carry the reason the
measured zero is not shipped as zero: 31 of 184 stints ran past 1.5x support
and 12 past 2x, so this bounds the growth rather than measuring it, and a
stint only reaches 2x support when the tyre is behaving, which censors the
dispersion there downward.

Twelve of the 196 stints have a non-positive measured rate (the race stint got
faster: track evolution beating degradation), where the log ratio does not
exist.  They are dropped from the log-scale fit and counted, and the same
analysis is repeated on the relative error (obs - pred)/pred, which exists for
all of them, as the sensitivity that says whether dropping them moved anything.

Writes `bench/out/extrapolation.json`; prints the bin table, the fit and the
sensitivities.  Reads only what the pipeline wrote; fits nothing tyre-side.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (DATA_PROCESSED, SHORT, arg_events, cp_for, dirty_air_of, driver_plans,  # noqa: E402
                    dump, first_stop_tables, fs_kwargs, kappa_of, meta, offline, race_table,
                    sealed_for)

from src.config import SIGMA_RACE_LAP_S, get_event  # noqa: E402

# Bin edges on the extrapolation ratio, as WP-B's plan specifies them.  The
# open top bin is where the Belgium plan sat (28 laps on a 14-lap support).
BINS = [(0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, np.inf)]
BIN_LABELS = ["<=1x", "1-1.5x", "1.5-2x", ">2x"]

# Normalised MAD as the dispersion estimator throughout: three of the seven
# weekends have a stint whose measured rate is four sigma from its prediction
# (a stint that was actually a fuel-save or a traffic stint), and a plain sd
# would let one of those set the scale of every plan in the search.
MAD_TO_SD = 1.4826

# Asymptotic variance of the normalised MAD for a normal sample is
# ~1.36 sigma^2 / n (its relative efficiency against the sd is 37%).  Used only
# to weight the per-bin dispersions in the WLS trend the plan asks for.
MAD_VAR_FACTOR = 1.36


def robust_sd(x: np.ndarray) -> float:
    """1.4826 x MAD - the dispersion this whole script quotes."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(MAD_TO_SD * np.median(np.abs(x - np.median(x))))


# --------------------------------------------------------------------------
# The per-stint table
# --------------------------------------------------------------------------


def stint_table(key: str) -> pd.DataFrame:
    """One row per scored race stint, with its extrapolation ratio.

    `scorestint_*.parquet` carries the scorer's observed and predicted rate;
    the per-lap `score_*.parquet` carries the ages, which is where the stint's
    highest tyre age - how far past the support it ran - comes from.
    """
    st = pd.read_parquet(DATA_PROCESSED / f"scorestint_{key}.parquet")
    lap = pd.read_parquet(DATA_PROCESSED / f"score_{key}.parquet")
    fs = json.loads((DATA_PROCESSED / f"fitstage_{key}.json").read_text())
    support = {c: float(v) for c, v in (fs.get("age_support_by_compound") or {}).items()}
    try:
        sigma_race = float(sealed_for(key, None).get("sigma_race") or SIGMA_RACE_LAP_S)
    except Exception:
        sigma_race = float(SIGMA_RACE_LAP_S)

    age = lap.groupby("stint_uid")["tyre_age"].agg(max_age="max", min_age="min")
    d = st.join(age, on="stint_uid")
    d["event"] = key
    d["support"] = d["compound"].map(support)
    d["sigma_race"] = sigma_race
    d["ratio"] = d["max_age"] / d["support"]
    d["excess"] = np.maximum(0.0, d["ratio"] - 1.0)
    # The scorer's rate interval is curve uncertainty + the noise of two
    # three-lap means; only the second is measurement error in the yardstick.
    d["noise_rate_sd"] = d["sigma_race"] * np.sqrt(2.0 / 3.0) / d["age_span"]
    d["curve_rate_sd"] = np.sqrt(np.maximum(0.0, d["rate_sd"] ** 2 - d["noise_rate_sd"] ** 2))
    d["rel_err"] = (d["obs_rate"] - d["pred_rate"]) / d["pred_rate"]
    with np.errstate(divide="ignore", invalid="ignore"):
        d["log_ratio"] = np.where(d["obs_rate"] > 0, np.log(d["obs_rate"] / d["pred_rate"]), np.nan)
    # Measurement noise on the log scale (delta method at the prediction, which
    # is the stable denominator - `obs_rate` goes through zero).
    d["noise_ln_sd"] = d["noise_rate_sd"] / d["pred_rate"]
    return d


# --------------------------------------------------------------------------
# Binned description
# --------------------------------------------------------------------------


def bin_of(ratio: float, bins=BINS, labels=BIN_LABELS) -> str:
    for (lo, hi), lab in zip(bins, labels):
        if lo < ratio <= hi:
            return lab
    return labels[0]


def describe(x: np.ndarray, *, noise: np.ndarray | None = None) -> dict:
    """n, bias and dispersion of an error sample, raw and noise-deflated."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    x = x[ok]
    out = {"n": int(len(x))}
    if not len(x):
        return out
    disp = robust_sd(x)
    out.update({
        "bias_mean": float(np.mean(x)),
        "bias_median": float(np.median(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "disp_mad": disp,
        "share_positive": float(np.mean(x > 0)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
    })
    if noise is not None:
        nu = np.asarray(noise, dtype=float)[ok]
        med_nu = float(np.median(nu))
        out["noise_ln_sd_median"] = med_nu
        out["disp_excess"] = float(np.sqrt(max(0.0, disp ** 2 - med_nu ** 2)))
    # A one-sample t on the mean says whether the bin is *biased*, which is a
    # different statement from being dispersed and is the one that decides
    # whether the widening should be centred.
    if len(x) > 2:
        se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
        out["bias_se"] = se
        out["bias_t"] = float(out["bias_mean"] / se) if se > 0 else float("nan")
    return out


def bin_table(d: pd.DataFrame, col: str, labels=BIN_LABELS) -> pd.DataFrame:
    rows = []
    for lab in labels:
        g = d[d["bin"] == lab]
        r = describe(g[col].to_numpy(), noise=g["noise_ln_sd"].to_numpy() if col == "log_ratio" else None)
        rows.append({"bin": lab, "ratio_mean": (float(g["ratio"].mean()) if len(g) else float("nan")),
                     "len_mean": (float(g["max_age"].mean()) if len(g) else float("nan")), **r})
    pooled = describe(d[col].to_numpy(), noise=d["noise_ln_sd"].to_numpy() if col == "log_ratio" else None)
    rows.append({"bin": "pooled", "ratio_mean": float(d["ratio"].mean()),
                 "len_mean": float(d["max_age"].mean()), **pooled})
    return pd.DataFrame(rows)


def wls_trend(tab: pd.DataFrame, *, x_col: str = "ratio_mean", y_col: str = "disp_mad") -> dict:
    """Weighted least squares of the per-bin dispersion on the bin's ratio.

    The weights are the inverse asymptotic variance of a normalised MAD,
    `n / (1.36 sigma^2)`, so a bin with four stints cannot set the slope.  This
    is the trend statistic the plan asks for; the ML fit below is the estimator
    the mechanism is actually scaled from.
    """
    t = tab[(tab["bin"] != "pooled") & np.isfinite(tab[y_col]) & (tab["n"] >= 2)]
    if len(t) < 3:
        return {"n_bins": int(len(t)), "slope": None, "se": None,
                "note": "fewer than three usable bins"}
    x = t[x_col].to_numpy(dtype=float)
    y = t[y_col].to_numpy(dtype=float)
    n = t["n"].to_numpy(dtype=float)
    w = n / (MAD_VAR_FACTOR * np.maximum(y, 1e-6) ** 2)
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    beta = np.linalg.solve(XtWX, X.T @ W @ y)
    cov = np.linalg.inv(XtWX)
    resid = y - X @ beta
    # Scale the covariance by the weighted residual variance so that an
    # over-dispersed set of four bins does not report a two-decimal SE.
    dof = max(len(t) - 2, 1)
    scale = float((w * resid ** 2).sum() / dof)
    cov = cov * max(scale, 1.0)
    return {"n_bins": int(len(t)), "intercept": float(beta[0]), "slope": float(beta[1]),
            "se": float(np.sqrt(cov[1, 1])), "t": float(beta[1] / np.sqrt(cov[1, 1])),
            "disp_at_1x": float(beta[0] + beta[1]), "disp_at_2x": float(beta[0] + 2 * beta[1]),
            "x": x.tolist(), "y": y.tolist(), "weights": w.tolist(),
            "overdispersion_scale": scale}


# --------------------------------------------------------------------------
# The estimator the mechanism is scaled from
# --------------------------------------------------------------------------


def ml_fit(e: np.ndarray, u: np.ndarray, nu: np.ndarray | None = None,
           *, centred: bool = False) -> dict:
    """ML fit of `e ~ N(mu, sigma_0^2 + a^2 u^2 + nu^2)`; returns `a` and its SE.

    `u = max(0, r - 1)` is the extrapolation beyond the support, so `a` is the
    extra log-sd a stint acquires at 2x support - exactly `TyreModel`'s
    `extrap_ln_sd`.  `nu` (per stint) is the scorer's own measurement noise and
    is held fixed, so the fit attributes to the model only what the yardstick
    cannot explain; pass `None` to see what the estimate would be if the noise
    were charged to the model.  `centred=True` fixes `mu = 0`.

    SEs come from the numerical Hessian of the negative log-likelihood at the
    optimum, and `a` is parameterised as `log a` so the optimiser cannot walk
    it negative; the SE is transformed back by the delta method.
    """
    from scipy.optimize import minimize

    e = np.asarray(e, dtype=float)
    u = np.asarray(u, dtype=float)
    nu2 = np.zeros_like(e) if nu is None else np.asarray(nu, dtype=float) ** 2
    ok = np.isfinite(e) & np.isfinite(u) & np.isfinite(nu2)
    e, u, nu2 = e[ok], u[ok], nu2[ok]
    if len(e) < 6:
        return {"n": int(len(e)), "note": "too few stints to fit"}

    def nll(th):
        mu = 0.0 if centred else th[0]
        s0, a = np.exp(th[-2]), np.exp(th[-1])
        v = s0 ** 2 + (a * u) ** 2 + nu2
        return float(0.5 * np.sum(np.log(2 * np.pi * v) + (e - mu) ** 2 / v))

    start = ([] if centred else [float(np.median(e))]) + [np.log(max(robust_sd(e), 0.05)), np.log(0.2)]
    res = minimize(nll, np.array(start, dtype=float), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 20000, "maxfev": 20000})
    th = res.x
    mu = 0.0 if centred else float(th[0])
    s0, a = float(np.exp(th[-2])), float(np.exp(th[-1]))

    # numerical Hessian in the optimiser's parameterisation
    h = 1e-4
    k = len(th)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i] = h
            ej[j] = h
            H[i, j] = (nll(th + ei + ej) - nll(th + ei - ej)
                       - nll(th - ei + ej) + nll(th - ei - ej)) / (4 * h * h)
    try:
        cov = np.linalg.inv(H)
        se_log_a = float(np.sqrt(max(cov[-1, -1], 0.0)))
        se_log_s0 = float(np.sqrt(max(cov[-2, -2], 0.0)))
    except np.linalg.LinAlgError:
        se_log_a = se_log_s0 = float("nan")

    # profile-likelihood interval on `a`: the range where 2 x delta-nll <= 3.84
    def profile(a_fix):
        def nll_p(th2):
            mu_ = 0.0 if centred else th2[0]
            s0_ = np.exp(th2[-1])
            v = s0_ ** 2 + (a_fix * u) ** 2 + nu2
            return float(0.5 * np.sum(np.log(2 * np.pi * v) + (e - mu_) ** 2 / v))
        st = ([] if centred else [mu]) + [np.log(max(s0, 1e-3))]
        r = minimize(nll_p, np.array(st, dtype=float), method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 20000})
        return float(r.fun)

    best = float(res.fun)
    grid = np.concatenate([[0.0], np.geomspace(1e-3, 2.0, 60)])
    prof = np.array([profile(g) for g in grid])
    inside = grid[2 * (prof - best) <= 3.841]
    lo = float(inside.min()) if len(inside) else float("nan")
    hi = float(inside.max()) if len(inside) else float("nan")
    # likelihood-ratio test of a = 0 (no growth with extrapolation)
    lr = float(2 * (profile(0.0) - best))

    return {"n": int(len(e)), "mu": mu, "sigma_0": s0, "a": a,
            "se_a": (float(a * se_log_a) if np.isfinite(se_log_a) else float("nan")),
            "se_sigma_0": (float(s0 * se_log_s0) if np.isfinite(se_log_s0) else float("nan")),
            "a_ci95": [lo, hi], "lr_stat_a0": lr,
            "lr_p_a0": float(np.exp(-0.5 * lr)) if lr > 0 else 1.0,
            "noise_deflated": nu is not None, "centred": centred,
            "converged": bool(res.success),
            "sd_at_1x": s0, "sd_at_2x": float(np.sqrt(s0 ** 2 + a ** 2))}


def sign_test(d: pd.DataFrame, *, col: str = "log_ratio", threshold: float = 1.0) -> dict:
    """Is the error beyond the support *biased* (race faster than predicted)?

    The direction decides whether the widening should be centred: a symmetric
    widening of a well-located median is the right mechanism only if the errors
    beyond the support are not systematically one-sided.
    """
    from scipy.stats import binomtest

    beyond = d[(d["ratio"] > threshold) & np.isfinite(d[col])]
    inside = d[(d["ratio"] <= threshold) & np.isfinite(d[col])]
    out = {}
    for name, g in (("beyond_support", beyond), ("inside_support", inside)):
        x = g[col].to_numpy(dtype=float)
        if not len(x):
            out[name] = {"n": 0}
            continue
        k = int(np.sum(x > 0))
        bt = binomtest(k, len(x), 0.5)
        out[name] = {"n": int(len(x)), "n_faster_than_predicted": k,
                     "share_faster": float(k / len(x)), "binom_p": float(bt.pvalue),
                     "mean": float(np.mean(x)), "median": float(np.median(x)),
                     "sd": (float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"))}
    if out.get("beyond_support", {}).get("n") and out.get("inside_support", {}).get("n"):
        from scipy.stats import ttest_ind
        t = ttest_ind(beyond[col].to_numpy(dtype=float), inside[col].to_numpy(dtype=float),
                      equal_var=False)
        out["difference_of_means"] = {
            "beyond_minus_inside": float(beyond[col].mean() - inside[col].mean()),
            "welch_t": float(t.statistic), "p": float(t.pvalue)}
    return out


# --------------------------------------------------------------------------
# What the widening does to the recommendation (WP-B's B4)
# --------------------------------------------------------------------------


def observed_stints(key: str) -> dict:
    """Per compound, the stint lengths the field's classified finishers ran.

    `p10 - 2 <= L <= max` is `src.validate.strategy_backtest`'s own definition
    of "a stint length the compound was run to", i.e. the gate Belgium failed;
    it is reproduced here rather than re-invented so the two agree.
    """
    ev = get_event(key)
    plans = driver_plans(race_table(key), int(ev.n_race_laps))
    per: dict = {}
    for _, r in plans.iterrows():
        for c, L in zip(str(r["seq"]).split("-"), list(r["stint_lens"])):
            per.setdefault(c, []).append(int(L))
    return {c: {"n": len(v), "min": int(min(v)), "p10": float(np.quantile(v, 0.10)),
                "median": float(np.median(v)), "max": int(max(v))}
            for c, v in per.items()}


def inside_observed(obs: dict, compound: str, L: int) -> bool | None:
    o = obs.get(compound)
    if not o:
        return None
    return bool(o["p10"] - 2 <= L <= o["max"])


def _gap_to_inside(res, obs: dict) -> dict:
    """The cheapest scored plan all of whose stints the field actually ran, and
    what it costs over the recommendation."""
    t = res.table
    if t is None or t.empty:
        return {}
    ok = [all(inside_observed(obs, c, int(L)) for c, L in zip(str(r["compounds"]).split("-"),
                                                              list(r["stint_lens"])))
          for _, r in t.iterrows()]
    cand = t[pd.Series(ok, index=t.index)]
    if cand.empty:
        return {"best_inside_observed": None, "gap_to_inside_s": None}
    row = cand.sort_values("mean_s").iloc[0]
    return {"best_inside_observed": str(row["strategy"]),
            "gap_to_inside_s": float(row["mean_s"] - t["mean_s"].min())}


def effect_sweep(keys: list, levels: list, *, pace_cal: bool = False) -> dict:
    """The same search at each `extrap_ln_sd`, on the objective the ablation runs.

    The keywords are `bench_ablation.run`'s - this weekend's leave-one-out race
    state, the calibrated constants, the plan prior through the nominations, no
    first-stop prior - at 300 draws and a one-lap grid, so the only thing that
    changes down a column is the extrapolation widening.

    `pace_cal=False` calls `simulate_model` directly: one moving part, and the
    cleanest read of what the widening alone does.  `pace_cal=True` adds the
    ladder-gate iteration `search_with_pace_calibration` that the pipeline
    actually ships, so the recommendation is comparable with `meta`'s; the two
    disagree at Australia and Hungary whatever the widening is, which is a fact
    about the pace calibration and not about this mechanism.
    """
    from src import racestate, strategy as strat
    from src.calibration import get_calibration
    from src.model_bayes import BayesFit
    from src.regime import RegimeFactor
    from src.tyre import TyreModel

    out = {}
    for key in keys:
        ev = get_event(key)
        m = meta(key)
        cal = get_calibration(ev)
        post = DATA_PROCESSED / f"posterior_{key}.npz"
        if not post.exists():
            print(f"  {key}: no posterior on disk")
            continue
        fit = BayesFit.load(post)
        idx = np.random.default_rng(0).choice(fit.posterior["lin"].shape[0],
                                              size=min(300, fit.posterior["lin"].shape[0]),
                                              replace=False)
        support = {k: float(v) for k, v in m["age_support_by_compound"].items()}
        caps = {c: int(v) for c, v in (m.get("circuit_history") or {}).get("stint_cap", {}).items()} or None
        fs_tables = first_stop_tables(cp_for(key, m), ev, fit.compounds)
        rs = racestate.measure_constants(exclude=key)
        kw = dict(regime=RegimeFactor(ratio=float(m["regime"]["ratio"]), ln_sd=float(m["regime"]["ln_sd"])),
                  support=support, max_per_compound=m["allocation"]["caps"], max_stint=caps,
                  undercut_lambda=cal.undercut_lambda, plan_prior=(m.get("plan_prior") or {}),
                  plan_prior_tau_s=cal.plan_prior_tau_s,
                  traffic_s_per_lap=dirty_air_of(cal, ev.circuit),
                  grid_penalty_s=cal.grid_start_penalty_s, step=1)
        kw.update(fs_kwargs(strat.simulate_model, fs_tables, 0.0 if rs is not None else kappa_of(cal)))
        if rs is not None:
            kw.update(race_state=rs, race_state_cover=True)

        obs = observed_stints(key)
        rows = []
        for a in levels:
            model = TyreModel.from_fit(fit, draws=idx, budget=cal.budgets,
                                       manage_floor=cal.manage_wear_floor,
                                       manage_cost_s=cal.manage_cost_s,
                                       support=support, extrap_ln_sd=float(a))
            t0 = time.perf_counter()
            ns = m.get("net_step") or {}
            if pace_cal and ns.get("measured") is not None:
                _, res, _ = strat.search_with_pace_calibration(
                    model, ev, float(m["pit_loss_s"]), net_step_s=float(ns["measured"]),
                    net_step_se_s=float(ns.get("se") or 0.0), **kw)
            else:
                res = strat.simulate_model(model, ev, float(m["pit_loss_s"]), **kw)
            best = res.best
            lens = [int(x) for x in best["stint_lens"]]
            comps = list(best["compounds"])
            starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int)
            rows.append({
                "extrap_ln_sd": float(a), "best": res.best_label, "seq": "-".join(comps),
                "n_stops": int(best["n_stops"]), "push": float(best.get("push", float("nan"))),
                "first_stop": (int(best["pit_laps"][0]) if best["pit_laps"] else None),
                "pit_laps": [int(x) for x in best["pit_laps"]], "stint_lens": lens,
                "inside_observed": [inside_observed(obs, c, L) for c, L in zip(comps, lens)],
                "all_inside_observed": all(inside_observed(obs, c, L) for c, L in zip(comps, lens)),
                "extrap_ratio": [round(L / support[c], 2) if support.get(c) else None
                                 for c, L in zip(comps, lens)],
                "mean_s": float(best.get("mean_s", float("nan"))),
                "race_state_s": float(best.get("race_state_s", 0.0) or 0.0),
                "tyre_optimal": res.tyre_optimal_label,
                "life_risk": [(getattr(res, "model", None) or model).life_risk(
                    c, int(L), int(s), float(best.get("push", 1.0)), ev)
                    for c, L, s in zip(comps, lens, starts)],
                "search_s": round(time.perf_counter() - t0, 1),
                # How big a premium would the widening have to charge to make the
                # gate pass?  The cheapest scored plan whose every stint is a
                # length the compound was run to, and what it costs over the
                # recommendation.  Diagnostic: it says whether the mechanism is
                # the wrong size or the wrong mechanism.
                **_gap_to_inside(res, obs),
            })
            r = rows[-1]
            print(f"  {SHORT.get(key, key):4s} sd {a:.3f}: {r['best']:34s} "
                  f"first {str(r['first_stop']):>4s} lens {str(r['stint_lens']):16s} "
                  f"ratios {str(r['extrap_ratio']):16s} inside {str(r['inside_observed'])} "
                  f"({r['search_s']}s)", flush=True)
        out[key] = {"observed_stints": obs, "shipped": (m.get("strategy") or {}).get("best"),
                    "support": support, "levels": rows}
    return out


def main() -> None:
    args = arg_events(__doc__.splitlines()[0], extra=[
        (("--effect",), dict(action="store_true",
                             help="also run the search at 0, 1/2x, 1x and 2x the measured widening "
                                  "and write bench/out/extrapolation_effect.json (WP-B's B4)")),
        (("--effect-only",), dict(action="store_true", help="the sweep alone, reusing the measured value")),
    ])
    offline()

    if args.effect_only:
        from src.tyre import EXTRAP_LN_SD_MEASURED
        run_effect(list(args.events), float(EXTRAP_LN_SD_MEASURED),
                   "src.tyre.EXTRAP_LN_SD_MEASURED (not re-measured in this run)")
        return

    frames = []
    for key in args.events:
        try:
            frames.append(stint_table(key))
        except FileNotFoundError as exc:
            print(f"  {key}: {exc}")
    if not frames:
        print("no scored weekends found; run the pipeline's decide stage first")
        return
    d = pd.concat(frames, ignore_index=True)
    d["bin"] = d["ratio"].map(bin_of)

    n_dropped = int((~np.isfinite(d["log_ratio"])).sum())
    print(f"\n{len(d)} scored race stints over {d['event'].nunique()} weekends; "
          f"{n_dropped} with a non-positive measured rate (no log ratio)")
    print("support per weekend (laps): " + ", ".join(
        f"{SHORT.get(k, k)} " + "/".join(f"{c[0]}{int(v)}" for c, v in
                                         sorted(g.groupby('compound')['support'].first().items()))
        for k, g in d.groupby("event")))

    tab = bin_table(d, "log_ratio")
    rel = bin_table(d, "rel_err")

    cols = ["bin", "n", "ratio_mean", "len_mean", "bias_mean", "bias_median",
            "disp_mad", "noise_ln_sd_median", "disp_excess", "share_positive"]
    print("\n=== ln(race rate / sealed predicted rate) by extrapolation ratio ===")
    print(tab[cols].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print("\n=== the same on (obs - pred)/pred, which every stint has ===")
    print(rel[["bin", "n", "bias_mean", "bias_median", "disp_mad", "share_positive"]]
          .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    trend_raw = wls_trend(tab, y_col="disp_mad")
    trend_exc = wls_trend(tab, y_col="disp_excess")
    print("\n=== dispersion trend (WLS over the four bins, weights n / 1.36 sigma^2) ===")
    for name, t in (("raw dispersion", trend_raw), ("noise-deflated", trend_exc)):
        if t.get("slope") is None:
            print(f"  {name}: {t.get('note')}")
        else:
            print(f"  {name}: slope {t['slope']:+.3f} +/- {t['se']:.3f} per unit ratio "
                  f"(t {t['t']:+.2f}); {t['disp_at_1x']:.3f} at 1x -> {t['disp_at_2x']:.3f} at 2x")

    fit = ml_fit(d["log_ratio"].to_numpy(), d["excess"].to_numpy(), d["noise_ln_sd"].to_numpy())
    fit_nonoise = ml_fit(d["log_ratio"].to_numpy(), d["excess"].to_numpy(), None)
    fit_rel = ml_fit(d["rel_err"].to_numpy(), d["excess"].to_numpy(), None)
    print("\n=== ML fit  e ~ N(mu, sigma_0^2 + a^2 max(0, r-1)^2 + nu^2) ===")
    for name, f in (("log ratio, noise-deflated (headline)", fit),
                    ("log ratio, noise charged to the model", fit_nonoise),
                    ("relative error, all 196 stints", fit_rel)):
        if "a" not in f:
            print(f"  {name}: {f.get('note')}")
            continue
        se = f["se_a"]
        print(f"  {name}: mu {f['mu']:+.3f}, sigma_0 {f['sigma_0']:.3f}, "
              f"a {f['a']:.3f} +/- {'n/a (at the boundary)' if not np.isfinite(se) else format(se, '.3f')} "
              f"(95% profile {f['a_ci95'][0]:.3f}-{f['a_ci95'][1]:.3f}), "
              f"LR vs a=0 {f['lr_stat_a0']:.2f} (p {f['lr_p_a0']:.3f}); "
              f"sd {f['sd_at_1x']:.3f} at 1x -> {f['sd_at_2x']:.3f} at 2x")

    signs = sign_test(d)
    b = signs.get("beyond_support", {})
    i = signs.get("inside_support", {})
    print("\n=== sign of the error ===")
    if b.get("n"):
        print(f"  beyond support (r > 1): {b['n']} stints, {b['share_faster']:.0%} degraded FASTER "
              f"than predicted (binom p {b['binom_p']:.3f}), mean {b['mean']:+.3f}")
    if i.get("n"):
        print(f"  inside support (r <= 1): {i['n']} stints, {i['share_faster']:.0%} faster "
              f"(binom p {i['binom_p']:.3f}), mean {i['mean']:+.3f}")
    if "difference_of_means" in signs:
        dm = signs["difference_of_means"]
        print(f"  beyond - inside: {dm['beyond_minus_inside']:+.3f} ln (Welch t {dm['welch_t']:+.2f}, "
              f"p {dm['p']:.3f})")

    # -- sensitivities the report has to carry -------------------------------
    sens = {
        "bin_edges_1_1.25_1.5_2": None,
        "per_compound": {}, "per_event": {},
        "drop_leave_one_event_out": {},
    }
    alt_bins = [(0.0, 1.0), (1.0, 1.25), (1.25, 1.5), (1.5, np.inf)]
    alt_lab = ["<=1x", "1-1.25x", "1.25-1.5x", ">1.5x"]
    d2 = d.copy()
    d2["bin"] = [bin_of(r, alt_bins, alt_lab) for r in d2["ratio"]]
    sens["bin_edges_1_1.25_1.5_2"] = bin_table(d2, "log_ratio", alt_lab).to_dict("records")

    for c, g in d.groupby("compound"):
        sens["per_compound"][c] = {
            **describe(g["log_ratio"].to_numpy(), noise=g["noise_ln_sd"].to_numpy()),
            "ratio_max": float(g["ratio"].max()),
            "fit": ml_fit(g["log_ratio"].to_numpy(), g["excess"].to_numpy(), g["noise_ln_sd"].to_numpy()),
        }
    for k, g in d.groupby("event"):
        sens["per_event"][k] = {
            **describe(g["log_ratio"].to_numpy(), noise=g["noise_ln_sd"].to_numpy()),
            "ratio_max": float(g["ratio"].max()),
            # counted on the same rows `n` is, i.e. those with a log ratio
            "n_beyond_support": int(((g["ratio"] > 1) & np.isfinite(g["log_ratio"])).sum()),
        }
    # Leave-one-weekend-out on the headline estimate: the number wired into
    # `src/tyre.py` must not be one weekend's.
    for k in sorted(d["event"].unique()):
        g = d[d["event"] != k]
        gf = g[np.isfinite(g["log_ratio"])]
        f = ml_fit(g["log_ratio"].to_numpy(), g["excess"].to_numpy(), g["noise_ln_sd"].to_numpy())
        raw_k = robust_sd(gf["log_ratio"].to_numpy())
        nu_k = float(np.median(gf["noise_ln_sd"]))
        sens["drop_leave_one_event_out"][k] = {
            "a": f.get("a"), "se_a": f.get("se_a"), "sigma_0": f.get("sigma_0"), "n": f.get("n"),
            "pooled_disp_raw": raw_k,
            "pooled_disp_deflated": float(np.sqrt(max(0.0, raw_k ** 2 - nu_k ** 2)))}
    loo = {k: v for k, v in sens["drop_leave_one_event_out"].items() if v.get("a") is not None}
    loo_a = [v["a"] for v in loo.values()]
    loo_p = [v["pooled_disp_deflated"] for v in loo.values()]
    print("\n=== leave-one-weekend-out (growth term `a` / pooled deflated dispersion) ===")
    print("  " + (", ".join(f"{SHORT.get(k, k)} {v['a']:.3f}/{v['pooled_disp_deflated']:.3f}"
                            for k, v in loo.items())
                  or "one weekend benchmarked: nothing left to leave out"))
    if loo_a:
        print(f"  a: {min(loo_a):.3f}-{max(loo_a):.3f}; pooled deflated: "
              f"{min(loo_p):.3f}-{max(loo_p):.3f} (median {float(np.median(loo_p)):.3f})")

    print("\n=== per compound ===")
    for c, v in sens["per_compound"].items():
        f = v["fit"]
        print(f"  {c:7s} n {v['n']:3d} bias {v['bias_mean']:+.3f} disp {v['disp_mad']:.3f} "
              f"max ratio {v['ratio_max']:.2f} a {f.get('a', float('nan')):.3f}")

    # -- how much of the pooled dispersion is the model's? -------------------
    # Three deflations of the same pooled number, because which one becomes
    # `extrap_ln_sd` is the one judgement in this script:
    #   raw        everything, including the noise in the yardstick
    #   noise      minus the scorer's own three-lap-mean noise  <- the one used
    #   full       minus the sealed curve's whole rate interval, i.e. also the
    #              posterior and regime spread the sealed prediction carries
    # `full` is too aggressive here: the *strategy* model carries the practice
    # posterior but not the regime spread (see src/tyre.py's docstring), so the
    # transfer dispersion is uncertainty it genuinely does not have.
    fin = np.isfinite(d["log_ratio"])
    raw = robust_sd(d.loc[fin, "log_ratio"].to_numpy())
    nu_med = float(np.median(d.loc[fin, "noise_ln_sd"]))
    tot_med = float(np.median(d.loc[fin, "rate_sd"] / d.loc[fin, "pred_rate"]))
    ladder = {"raw": raw, "noise_ln_sd_median": nu_med, "total_rate_sd_over_pred_median": tot_med,
              "deflated_noise": float(np.sqrt(max(0.0, raw ** 2 - nu_med ** 2))),
              "deflated_full": float(np.sqrt(max(0.0, raw ** 2 - tot_med ** 2)))}
    print("\n=== pooled dispersion, deflated three ways ===")
    print(f"  raw {ladder['raw']:.3f}; minus yardstick noise ({nu_med:.3f}) -> "
          f"{ladder['deflated_noise']:.3f}; minus the whole sealed rate interval "
          f"({tot_med:.3f}) -> {ladder['deflated_full']:.3f}")

    # -- the recommendation --------------------------------------------------
    grows = bool(fit.get("a") and fit.get("lr_p_a0", 1.0) < 0.10
                 and (trend_exc.get("slope") or 0.0) > 0)
    recommended = float(fit["a"]) if grows else ladder["deflated_noise"]
    basis = ("the ML growth term `a`: dispersion grows with extrapolation" if grows else
             "the pooled practice->race transfer dispersion, deflated by the scorer's own "
             "three-lap-mean noise: no growth with extrapolation is detectable "
             f"(a = {fit.get('a', float('nan')):.3f}, 95% profile CI 0-{fit.get('a_ci95', [0, 0])[1]:.3f}, "
             f"noise-deflated WLS slope {trend_exc.get('slope', float('nan')):+.3f}), so the "
             "plan's fallback applies")
    print(f"\nEXTRAP_LN_SD_MEASURED = {recommended:.3f}  [{basis}]")

    payload = {
        "n_stints": int(len(d)), "n_events": int(d["event"].nunique()),
        "n_no_log_ratio": n_dropped,
        "events": sorted(d["event"].unique().tolist()),
        "definition": {
            "error": "ln(obs_rate / pred_rate) from scorestint_<key>.parquet "
                     "(src.validate.score_race: first-3 to last-3 lap means, stint-centred)",
            "ratio": "max tyre age in the stint / fitstage age_support_by_compound[compound]",
            "dispersion": "1.4826 x MAD",
            "measurement_noise": "sigma_race x sqrt(2/3) / age_span / pred_rate",
            "bins": BIN_LABELS,
        },
        "bins_log_ratio": tab.to_dict("records"),
        "bins_rel_err": rel.to_dict("records"),
        "trend_wls_raw": trend_raw,
        "trend_wls_noise_deflated": trend_exc,
        "ml_fit": fit,
        "ml_fit_noise_charged_to_model": fit_nonoise,
        "ml_fit_relative_error": fit_rel,
        "sign": signs,
        "sensitivity": sens,
        "pooled_dispersion_ladder": ladder,
        "recommended_extrap_ln_sd": recommended,
        "recommended_basis": basis,
        "per_stint": d[["event", "driver", "compound", "max_age", "support", "ratio",
                        "n_laps", "age_span", "obs_rate", "pred_rate", "err", "rate_sd",
                        "noise_ln_sd", "log_ratio", "rel_err", "bin"]].to_dict("records"),
    }
    p = dump("extrapolation.json", payload)
    print(f"\nwrote {p}")

    if args.effect:
        run_effect(list(args.events), recommended, basis)


def run_effect(keys: list, a: float, basis: str) -> None:
    """The B4 sweep at 0, 1/2x, 1x and 2x the measured widening, both ways."""
    levels = [0.0, 0.5 * a, a, 2.0 * a]
    print("\n=== the search at each level of the widening (no pace calibration) ===")
    raw = effect_sweep(keys, levels, pace_cal=False)
    print("\n=== the same with the shipped ladder-gate iteration ===")
    cal = effect_sweep(keys, levels, pace_cal=True)
    q = dump("extrapolation_effect.json",
             {"measured": a, "basis": basis, "levels": levels,
              "per_event": raw, "per_event_pace_calibrated": cal,
              "note": "simulate_model at 300 draws, 1-lap grid, bench_ablation.run's keywords, "
                      "leave-one-out race state; `per_event_pace_calibrated` adds the "
                      "search_with_pace_calibration iteration the pipeline ships"})
    print_effect(raw, "no pace calibration")
    print_effect(cal, "pace calibration (as shipped)")
    print(f"wrote {q}")


def print_effect(eff: dict, label: str = "") -> None:
    """The B4 table: what each level of the widening recommends."""
    rows = []
    for key, v in eff.items():
        for r in v["levels"]:
            rows.append({"event": SHORT.get(key, key), "sd": round(r["extrap_ln_sd"], 3),
                         "plan": r["best"], "first": r["first_stop"],
                         "lens": "/".join(str(x) for x in r["stint_lens"]),
                         "L_over_support": "/".join(str(x) for x in r["extrap_ratio"]),
                         "inside_observed": "".join("Y" if x else ("n" if x is False else "?")
                                                    for x in r["inside_observed"]),
                         "all_inside": r["all_inside_observed"],
                         "p_cliff": "/".join(f"{x['p_cliff']:.2f}" for x in r["life_risk"]),
                         "gap_to_inside_s": (round(r["gap_to_inside_s"], 2)
                                             if r.get("gap_to_inside_s") is not None else None),
                         "best_inside": r.get("best_inside_observed"),
                         "shipped": v["shipped"]})
    if not rows:
        return
    print(f"\n=== effect of the widening on the recommendation, {label} "
          "(300 draws, 1-lap grid, leave-one-out race state) ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
