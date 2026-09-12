"""Per-car tyre intelligence: what this weekend knows about this driver.

Two sources say how hard a particular car is on its tyres, and V2 used them
badly.

**The practice fit's own deviation.**  The Bayesian fit carries `dev[d, c]`,
the driver's additive slope deviation per compound in s/lap of age.  It is the
sharpest evidence available — measured this weekend, on this track, on this
car — but it exists only for drivers who did a long run on that compound, and
in a normal weekend that is a handful of cars per compound.  V2 added it where
it existed and did nothing where it did not, so half the grid ran on the field
model and the half that did not were pinned by whatever one long run happened
to show.

**Previous races.**  `history.race_driver_factors` measures a multiplicative
rate factor per driver from the races already run, pooled and shrunk by
`scripts/80_recalibrate.py`.  Stable, but stale by a weekend or more, and V2
applied it at full strength regardless of how precisely it was measured: a
driver whose factor was 1.32 +/- 0.25 moved the model as much as one measured
to +/-0.04.

**What this module adds.**  A team is the strongest grouping there is for tyre
behaviour — same car, same aero map, same suspension kinematics, same tyre
engineers, same set-up philosophy — and stronger than any pooling toward the
field.  `team_pooled_dev` therefore pools `dev` at team level and blends each
driver toward their own team's value with a weight that rises with the laps
behind their own measurement.  Two consequences: a driver with no long run on a
compound inherits their *team-mate's* deviation instead of the field's, and a
driver with one noisy long run is pulled most of the way back toward it.

`shrink_factor` does the matching job for the historical factor, on the log
scale, with the driver's own standard error deciding how much of it survives.

Nothing here writes a model; `TyreModel.for_driver` consumes both terms, and
`rate_scale_table` exposes the net per-driver rate multiplier so the accuracy
benchmark can rank the five variants (none / hist / practice_dev / team_pooled
/ combined) against the race's own per-driver rates.
"""

from __future__ import annotations

import numpy as np

__all__ = ["team_pooled_dev", "rate_scale_table", "shrink_factor", "RATE_SCALE_KINDS"]

POOL_K_LAPS = 20             # clean long-run laps at which a driver's own dev outweighs their team's
FACTOR_PRIOR_LN_SD = 0.10    # how far from the field a historical rate factor is believed to sit
RATE_FLOOR_S = 1e-3          # the rate floor `TyreModel.for_driver` applies, mirrored here

RATE_SCALE_KINDS = ("none", "hist", "practice_dev", "team_pooled", "combined")


# --------------------------------------------------------------------------
# Pooling the practice deviation at team level
# --------------------------------------------------------------------------


def team_pooled_dev(dev_by_driver: dict, teams, *, own_weight_laps=None,
                    k_laps: float = POOL_K_LAPS, n_laps_by_driver: dict | None = None) -> dict:
    """Per-driver, per-compound practice deviation, pooled over the team.

    `dev_by_driver` is `{driver: {compound: (draws,) array}}` — the shape
    `TyreModel.driver_dev` and `BayesFit.posterior["dev"]` carry.  `teams` maps
    driver to team (a dict or a `pd.Series`, e.g.
    `race.drop_duplicates("driver").set_index("driver")["team"]`).

    The blend is `w * own + (1 - w) * team` per compound with
    `w = n_own / (n_own + k_laps)`, the laps coming from `n_laps_by_driver`
    where the caller has them, else from `own_weight_laps` (a float or a dict
    standing in for them), else `k_laps` — equal weight, which is the honest
    default when nobody counted.  The blend is taken *per draw*, so the
    posterior correlation between a driver's deviation and their team-mate's is
    preserved and the strategy search's credible intervals stay meaningful.

    A driver with a team-mate's measurement but none of their own gets the team
    value outright; a driver whose team has no measurement at all is absent from
    the result and runs on the field model.
    """
    dev = {d: dict(v) for d, v in (dev_by_driver or {}).items() if v}
    tm = {str(k): str(v) for k, v in (dict(teams) if teams is not None else {}).items()}

    by_team: dict = {}
    for drv, per_c in dev.items():
        t = tm.get(drv)
        if t is None:
            continue
        for c, arr in per_c.items():
            by_team.setdefault(t, {}).setdefault(c, []).append(np.asarray(arr, dtype=float))
    team_dev = {t: {c: np.mean(np.stack(v), axis=0) for c, v in per_c.items()} for t, per_c in by_team.items()}

    def own_laps(drv: str) -> float:
        if n_laps_by_driver and drv in n_laps_by_driver:
            return float(n_laps_by_driver[drv])
        if isinstance(own_weight_laps, dict):
            return float(own_weight_laps.get(drv, k_laps))
        if own_weight_laps is not None:
            return float(own_weight_laps)
        return float(k_laps)

    out: dict = {}
    for drv in sorted(set(dev) | set(tm)):
        mine = dev.get(drv, {})
        theirs = team_dev.get(tm.get(drv, ""), {})
        if not mine and not theirs:
            continue
        if not mine:
            out[drv] = {c: np.asarray(v, dtype=float).copy() for c, v in theirs.items()}
            continue
        n_own = own_laps(drv)
        w = float(n_own / (n_own + float(k_laps))) if (n_own + k_laps) > 0 else 1.0
        blended = {}
        for c in sorted(set(mine) | set(theirs)):
            o, t = mine.get(c), theirs.get(c)
            if o is None:
                blended[c] = np.asarray(t, dtype=float).copy()
            elif t is None:
                blended[c] = np.asarray(o, dtype=float).copy()
            else:
                blended[c] = w * np.asarray(o, dtype=float) + (1.0 - w) * np.asarray(t, dtype=float)
        out[drv] = blended
    return out


# --------------------------------------------------------------------------
# Shrinking the historical factor
# --------------------------------------------------------------------------


def shrink_factor(factor: float, ln_sd: float | None, *, prior_ln_sd: float = FACTOR_PRIOR_LN_SD) -> float:
    """A rate factor pulled toward 1 by how loosely it was measured.

    On the log scale the posterior mean of a normal mean with a N(0, prior^2)
    prior is the measurement times its share of the total precision, which is
    `prior^2 / (prior^2 + se^2)`.  So a driver measured across seven races keeps
    most of their factor and one measured once keeps little of it, which is the
    behaviour V2's unshrunk factors lacked.
    """
    try:
        f = float(factor)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(f) or f <= 0:
        return 1.0
    p2 = float(prior_ln_sd) ** 2
    s2 = float(ln_sd) ** 2 if (ln_sd is not None and np.isfinite(ln_sd)) else 0.0
    w = p2 / (p2 + s2) if (p2 + s2) > 0 else 1.0
    return float(np.exp(np.log(f) * w))


# --------------------------------------------------------------------------
# The per-driver rate multiplier, for the benchmark's five variants
# --------------------------------------------------------------------------


def _dev_from(*, model=None, fit=None) -> dict:
    """`{driver: {compound: (draws,)}}` from a `TyreModel` or a `BayesFit`."""
    if model is not None and getattr(model, "driver_dev", None):
        return {d: dict(v) for d, v in model.driver_dev.items()}
    if fit is None:
        return {}
    D = np.asarray((getattr(fit, "posterior", {}) or {}).get("dev", []))
    drivers, comps = list(getattr(fit, "drivers", []) or []), list(getattr(fit, "compounds", []) or [])
    if D.ndim != 3 or D.shape[1] != len(drivers) or D.shape[2] != len(comps):
        return {}
    return {d: {c: D[:, i, j] for j, c in enumerate(comps)} for i, d in enumerate(drivers)}


def _rates(*, model=None, fit=None) -> dict:
    """Field degradation rate per compound at full push, s/lap."""
    if model is not None:
        return {c: float(model.rate(c).mean()) for c in model.compounds}
    if fit is None:
        return {}
    span = np.array([1.0, 10.0])
    out = {}
    for c in getattr(fit, "compounds", []) or []:
        d = fit.deg_loss(c, span)
        out[c] = float(((d[:, 1] - d[:, 0]) / (span[1] - span[0])).mean())
    return out


def _hist_factors(cal) -> dict:
    """`{driver: {"factor", "ln_sd"}}` from a `Calibration`, its JSON, or a plain map."""
    if cal is None:
        return {}
    get = (lambda k: cal.get(k)) if isinstance(cal, dict) else (lambda k: getattr(cal, k, None))
    detail = get("driver_factor_detail") or {}
    factors = get("driver_factors")
    sds = get("driver_factor_ln_sd") or {}
    if factors is None and not detail and isinstance(cal, dict) and cal and all(
            isinstance(v, (int, float)) for v in cal.values()):
        factors = cal          # a bare {driver: factor} map
    out = {}
    for drv, v in (detail or {}).items():
        out[drv] = {"factor": float(v.get("factor", 1.0)), "ln_sd": v.get("ln_sd")}
    for drv, v in (factors or {}).items():
        f = float(v.get("factor", 1.0)) if isinstance(v, dict) else float(v)
        sd = v.get("ln_sd") if isinstance(v, dict) else sds.get(drv)
        out.setdefault(drv, {"factor": f, "ln_sd": sd})
    return out


def _dev_scale(per_compound: dict | None, rates: dict) -> float:
    """`mean_c (rate_c + dev_c) / rate_c` over the compounds the driver supports.

    The same arithmetic `TyreModel.for_driver` does, rate floor included, so the
    number the benchmark ranks is the number the model actually applies.
    """
    if not per_compound or not rates:
        return 1.0
    vals = []
    for c, arr in per_compound.items():
        r = float(rates.get(c, float("nan")))
        if not np.isfinite(r) or r <= 0:
            continue
        vals.append(max(r + float(np.asarray(arr, dtype=float).mean()), RATE_FLOOR_S) / r)
    return float(np.mean(vals)) if vals else 1.0


def rate_scale_table(kind: str, *, fit=None, model=None, cal=None, teams=None,
                     n_laps_by_driver: dict | None = None, **kw) -> dict:
    """`{driver: multiplicative rate scale}` for one per-car variant.

        none          every car on the field model (the 1.0 baseline)
        hist          V2's pooled leave-one-out race factors, unshrunk
        practice_dev  this weekend's own `dev[d, c]`, where it exists
        team_pooled   the same, pooled over the team (shipped in V3)
        combined      team-pooled dev times the shrunk historical factor

    Every driver the inputs know about appears, so the benchmark's Spearman
    correlation against the race's measured per-driver rates is taken over the
    same population in all five variants.
    """
    k = str(kind or "none")
    if k not in RATE_SCALE_KINDS:
        raise ValueError(f"unknown per-car kind {kind!r}; known: {RATE_SCALE_KINDS}")
    tm = {str(a): str(b) for a, b in (dict(teams) if teams is not None else {}).items()}
    hist = _hist_factors(cal)
    dev = _dev_from(model=model, fit=fit)
    drivers = sorted(set(tm) | set(dev) | set(hist))
    if k == "none":
        return {d: 1.0 for d in drivers}
    if k == "hist":
        return {d: float(hist.get(d, {}).get("factor", 1.0)) for d in drivers}
    rates = _rates(model=model, fit=fit)
    used = dev if k == "practice_dev" else team_pooled_dev(dev, tm, n_laps_by_driver=n_laps_by_driver, **kw)
    out = {d: _dev_scale(used.get(d), rates) for d in drivers}
    if k == "combined":
        for d in drivers:
            h = hist.get(d, {})
            out[d] = float(out[d] * shrink_factor(h.get("factor", 1.0), h.get("ln_sd")))
    return out
