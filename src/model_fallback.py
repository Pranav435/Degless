"""Degradation baselines — built first, on purpose.

These are the sanity checks that tell us the Bayesian posterior isn't lying.  If
they disagree by more than ~0.03 s/lap, the Bayesian model has a bug, not an
insight.

There are two, deliberately.  `fit_mixedlm` is the full mixed model: driver
random intercepts, stint-level variance components, compound slopes.  It is the
better estimator and the one the accuracy benchmark reports — and it is an
iterative optimisation that can fail to converge, wander, or return a slope
that is an artefact of its own variance structure, which is exactly the failure
a gate must not be blind to.  `stint_fe_baseline` is the other end: closed-form
least squares on within-stint deviations, no optimiser, no priors, no
distributional assumptions, nothing to converge.  It cannot be subtly wrong in
the way a hierarchical fit can, which is why it, and not the MixedLM's range, is
what the pipeline gates the posterior against (plan §3.3).  V2 gated on
"MixedLM MEDIUM slope in 0.12-0.35 s/lap", a hardcoded range that encodes what
2026 Barcelona happened to show and would fire on a genuinely low-degradation
circuit.

Uncertainty comes in both from a **block bootstrap that resamples whole
stints**, not laps.  Laps within a stint are strongly correlated (same fuel
load, same driver, same run plan); resampling laps would understate the
standard error by a large factor.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.config import BOOTSTRAP_N

log = logging.getLogger("degless.mixedlm")

FORMULA = "lap_time_corr ~ C(compound):tyre_age + C(compound)"
# Stint-level intercepts as a variance component nested inside driver.  Without
# them the unknown starting fuel load of each practice long run leaks into the
# age slope: long runs are typically started heavy *and* reach higher tyre age,
# so their level and their mean age move together and the slope biases upward
# (measured on Barcelona: MEDIUM 0.359 s/lap without, 0.295 s/lap with).  The
# Bayesian model carries a free `base[s]` per stint for exactly this reason, so
# the baseline has to match it for the cross-check to mean anything.
VC_FORMULA = {"stint": "0 + C(stint_uid)"}
_SLOPE_RE = re.compile(r"C\(compound\)\[(?:T\.)?([A-Z]+)\]:tyre_age")
_INTERCEPT_RE = re.compile(r"C\(compound\)\[(?:T\.)?([A-Z]+)\]$")

# The stint-FE baseline believes a compound it has seen at least this much of.
# Same thresholds as `regime._stint_fe_slope`, so the baseline slope and the
# regime estimator's practice slope are the same quantity on the same laps.
FE_MIN_STINTS = 2
FE_MIN_LAPS = 12


@dataclass
class MixedLMFit:
    slopes: dict = field(default_factory=dict)        # compound -> s/lap
    intercepts: dict = field(default_factory=dict)    # compound -> s
    boot_slopes: dict = field(default_factory=dict)   # compound -> ndarray
    n_laps: int = 0
    n_stints: int = 0
    n_boot_ok: int = 0
    stints_per_compound: dict = field(default_factory=dict)
    converged: bool = False
    summary_text: str = ""

    def ci(self, compound: str, level: float = 0.90) -> tuple[float, float]:
        b = self.boot_slopes.get(compound)
        if b is None or len(b) < 10:
            return (np.nan, np.nan)
        lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
        return (float(np.percentile(b, lo)), float(np.percentile(b, hi)))

    def table(self, level: float = 0.90) -> pd.DataFrame:
        rows = []
        for c in sorted(self.slopes):
            lo, hi = self.ci(c, level)
            ns = self.stints_per_compound.get(c, 0)
            rows.append(
                {"compound": c, "slope_s_per_lap": self.slopes[c],
                 "lo": lo, "hi": hi, "n_stints": ns,
                 "n_boot": len(self.boot_slopes.get(c, [])),
                 # A single stint gives the block bootstrap nothing to resample:
                 # every draw is the same stint, so the band collapses to zero
                 # width.  That is not confidence, it is absence of evidence.
                 "band_estimable": ns >= 2}
            )
        return pd.DataFrame(rows)


def _extract(params: pd.Series) -> tuple[dict, dict]:
    slopes, intercepts = {}, {}
    for name, val in params.items():
        m = _SLOPE_RE.search(name)
        if m:
            slopes[m.group(1)] = float(val)
            continue
        m = _INTERCEPT_RE.search(name)
        if m:
            intercepts[m.group(1)] = float(val)
    return slopes, intercepts


def _fit_once(df: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(FORMULA, df, groups=df["driver"], re_formula="~1",
                            vc_formula=VC_FORMULA)
        return model.fit(method="lbfgs", maxiter=200)


def fit_mixedlm(df: pd.DataFrame, *, n_boot: int = BOOTSTRAP_N,
                seed: int = 0) -> MixedLMFit:
    """Fit the baseline and bootstrap it over stints."""
    d = df.dropna(subset=["lap_time_corr", "tyre_age", "compound"]).copy()
    if d.empty:
        raise ValueError("no usable laps for MixedLM")

    res = _fit_once(d)
    slopes, intercepts = _extract(res.params)
    stints = d["stint_uid"].unique()

    out = MixedLMFit(
        slopes=slopes,
        intercepts=intercepts,
        n_laps=len(d),
        n_stints=len(stints),
        stints_per_compound=d.groupby("compound")["stint_uid"].nunique().to_dict(),
        converged=bool(getattr(res, "converged", True)),
        summary_text=str(res.summary()),
    )

    # -- block bootstrap over whole stints ---------------------------------
    rng = np.random.default_rng(seed)
    by_stint = {s: g for s, g in d.groupby("stint_uid")}
    boot: dict[str, list] = {c: [] for c in slopes}
    ok = 0
    for _ in range(n_boot):
        pick = rng.choice(stints, size=len(stints), replace=True)
        frames = []
        for rep, s in enumerate(pick):
            g = by_stint[s].copy()
            # Re-label so repeated draws stay distinct stints/groups.
            g["stint_uid"] = f"{s}#{rep}"
            frames.append(g)
        sample = pd.concat(frames, ignore_index=True)
        try:
            r = _fit_once(sample)
            s_b, _ = _extract(r.params)
        except Exception:
            continue
        if not s_b:
            continue
        ok += 1
        for c, v in s_b.items():
            if np.isfinite(v):
                boot.setdefault(c, []).append(v)

    out.boot_slopes = {c: np.asarray(v) for c, v in boot.items() if v}
    out.n_boot_ok = ok
    log.info("MixedLM: %d laps, %d stints, %d/%d bootstrap fits ok",
             out.n_laps, out.n_stints, ok, n_boot)
    return out


# --------------------------------------------------------------------------
# The stint fixed-effects baseline: what the pipeline gates the posterior on
# --------------------------------------------------------------------------


def _fe_slopes(df: pd.DataFrame, ycol: str, *, min_stints: int = FE_MIN_STINTS,
               min_laps: int = FE_MIN_LAPS) -> dict:
    """Per-compound slope of `ycol` on tyre age, within stints.

    Demeaning both sides inside a stint absorbs the stint's level — the unknown
    starting fuel load of a practice long run, the driver, the engine mode — so
    what is left is how lap time moves with age *inside* a run.  Closed form:
    `sum(x*y) / sum(x*x)` on the deviations, which is algebraically the OLS
    slope of a regression with one free intercept per stint.
    """
    out = {}
    for c in sorted(df["compound"].dropna().unique()):
        g = df[df["compound"] == c]
        if g["stint_uid"].nunique() < min_stints or len(g) < min_laps:
            continue
        y = g[ycol] - g.groupby("stint_uid")[ycol].transform("mean")
        x = g["tyre_age"] - g.groupby("stint_uid")["tyre_age"].transform("mean")
        denom = float(np.sum(x ** 2))
        if denom <= 0:
            continue
        out[c] = {"slope": float(np.sum(x * y) / denom), "n_laps": int(len(g)),
                  "n_stints": int(g["stint_uid"].nunique())}
    return out


def stint_fe_baseline(clean: pd.DataFrame, *, n_boot: int = 50, seed: int = 0,
                      ycol: str = "lap_time_corr", level: float = 0.90) -> dict:
    """Per-compound degradation with stint fixed effects, with a 90% band.

    `clean` is the clean practice lap table *after* both corrections — fuel
    (`fuel.add_fuel_correction`) and track evolution
    (`evolution.add_evolution_correction`) — so `lap_time_corr` is the channel
    the Bayesian model fits and the two are directly comparable.  Skipping the
    evolution correction is not a detail: at a low-degradation circuit it is
    most of the slope, and the baseline would gate the posterior against a
    number biased toward zero.

    The band is a block bootstrap over whole stints, the same resampling unit
    `fit_mixedlm` uses, because laps within a stint are not independent draws.

    Returns `{slopes, ci, n_stints, n_laps, pooled_slope, per_compound, table}`.
    `pooled_slope` is the laps-weighted mean over compounds — the single number
    the pipeline gate compares the posterior's pooled slope against.
    """
    need = {ycol, "tyre_age", "compound", "stint_uid"}
    missing = need - set(clean.columns)
    if missing:
        raise KeyError(f"stint_fe_baseline needs {sorted(missing)} on the lap table "
                       f"(fuel + evolution corrections applied?)")
    d = clean.dropna(subset=[ycol, "tyre_age", "compound"]).copy()
    if d.empty:
        raise ValueError("no usable laps for the stint-FE baseline")

    per = _fe_slopes(d, ycol)
    slopes = {c: v["slope"] for c, v in per.items()}
    stints = d["stint_uid"].unique()

    # -- block bootstrap over whole stints ---------------------------------
    rng = np.random.default_rng(seed)
    by_stint = {s: g for s, g in d.groupby("stint_uid")}
    boot: dict[str, list] = {c: [] for c in slopes}
    pooled_boot: list[float] = []
    for _ in range(max(int(n_boot), 0)):
        pick = rng.choice(stints, size=len(stints), replace=True)
        frames = []
        for rep, s in enumerate(pick):
            g = by_stint[s].copy()
            g["stint_uid"] = f"{s}#{rep}"   # repeated draws stay distinct stints
            frames.append(g)
        sample = pd.concat(frames, ignore_index=True)
        b = _fe_slopes(sample, ycol)
        if not b:
            continue
        for c, v in b.items():
            if c in boot and np.isfinite(v["slope"]):
                boot[c].append(v["slope"])
        wn = sum(v["n_laps"] for v in b.values())
        if wn:
            pooled_boot.append(sum(v["slope"] * v["n_laps"] for v in b.values()) / wn)

    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    ci = {}
    for c, vals in boot.items():
        a = np.asarray(vals, dtype=float)
        ci[c] = ((float(np.percentile(a, lo_q)), float(np.percentile(a, hi_q)))
                 if len(a) >= 10 else (float("nan"), float("nan")))

    w = sum(v["n_laps"] for v in per.values())
    pooled = float(sum(v["slope"] * v["n_laps"] for v in per.values()) / w) if w else float("nan")
    pb = np.asarray(pooled_boot, dtype=float)
    pooled_ci = ((float(np.percentile(pb, lo_q)), float(np.percentile(pb, hi_q)))
                 if len(pb) >= 10 else (float("nan"), float("nan")))

    rows = [{"compound": c, "slope_s_per_lap": per[c]["slope"],
             "lo": ci[c][0], "hi": ci[c][1], "n_stints": per[c]["n_stints"],
             "n_laps": per[c]["n_laps"], "n_boot": len(boot.get(c, [])),
             # One stint gives the block bootstrap nothing to resample, so a
             # zero-width band would be absence of evidence, not confidence.
             "band_estimable": per[c]["n_stints"] >= 2}
            for c in sorted(per)]
    table = pd.DataFrame(rows, columns=["compound", "slope_s_per_lap", "lo", "hi",
                                        "n_stints", "n_laps", "n_boot", "band_estimable"])
    log.info("stint-FE baseline: %d laps, %d stints, pooled %.4f s/lap, slopes %s",
             len(d), len(stints), pooled, {c: round(v, 4) for c, v in slopes.items()})
    return {
        "slopes": slopes, "ci": ci, "n_stints": int(len(stints)), "n_laps": int(len(d)),
        "pooled_slope": pooled, "pooled_ci": pooled_ci,
        "n_stints_by_compound": {c: per[c]["n_stints"] for c in per},
        "n_laps_by_compound": {c: per[c]["n_laps"] for c in per},
        "per_compound": per, "table": table, "n_boot": int(n_boot), "ycol": ycol,
        "estimator": "stint fixed effects (within-stint OLS), stint block bootstrap",
    }
