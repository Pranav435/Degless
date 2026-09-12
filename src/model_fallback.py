"""MixedLM degradation baseline — built first, on purpose.

This is the sanity check that tells us the Bayesian posterior isn't lying.  If
the two disagree by more than ~0.03 s/lap, the Bayesian model has a bug, not an
insight.

Uncertainty comes from a **block bootstrap that resamples whole stints**, not
laps.  Laps within a stint are strongly correlated (same fuel load, same
driver, same run plan); resampling laps would understate the standard error by
a large factor.
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
