"""Track evolution: identified from push laps, applied to long runs.

The track gets faster through a practice session as rubber goes down.  Left in
the data, that improvement offsets tyre degradation within every long run and
the fitted slope comes out biased low — at a low-degradation circuit, below
zero.  The question is what observable identifies the evolution.

**Long runs cannot.**  Within a stint, session time and tyre age advance
together, so with a free per-stint level (needed to absorb the unknown fuel
load) evolution and degradation are collinear.  Across stints the level
differences are dominated by run plans that are correlated with time — early
low-fuel runs, late heavy race simulations — so a driver's long-run pace can
*rise* through the session while the track improves, and a monotone fit to it
returns nothing (Monza 2026: 0.00 s) or too much (Barcelona 2026: 2.9 s of
run-plan effect read as track).

**Push laps can.**  A lap within ~1.5 s of a driver's session best is a
low-fuel, fresh-tyre effort by construction, so push laps at different times
are comparable on exactly the two things long runs are not.  After the
driver's level and the compound ladder's pace step are removed, what is left is
the track, fitted as saturating rubber-in `-A (1 - exp(-t/T))` per session
(`fit_evolution_push`).  Measured this way every 2026 session evolves by
0.7-1.2 s, which is the size the physics suggests.  The long-run backfit is
kept as a fallback for a session with too few push laps and as a diagnostic
(`fit_evolution_auto` reports both).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger("degless.evolution")

N_BACKFIT_ITERS = 3
BIN_SECONDS = 300.0        # 5-minute bins: ~15 knots over a practice session
PUSH_MARGIN_S = 1.5        # a lap within this of the driver's session best is a push lap
PUSH_MIN_LAPS = 12         # per session, else fall back to the long-run backfit
PUSH_MIN_BINS = 3
PUSH_AGE_S_PER_LAP = 0.04  # nominal tyre-age correction on a push lap (ages 1-6)
MIN_LAPS_FOR_EVO = 20      # below this a session gets a flat (zero) curve
MAX_PLAUSIBLE_EVO_S = 5.0  # sanity guard; warns rather than fails


@dataclass
class EvolutionFit:
    """Per-session isotonic curves plus the degradation slopes they were fitted against."""

    models: dict = field(default_factory=dict)      # session -> IsotonicRegression | None
    deg_slopes: dict = field(default_factory=dict)  # compound -> s/lap
    iterations: list = field(default_factory=list)  # diagnostics per backfit pass
    skipped: list = field(default_factory=list)     # sessions with too few laps

    def predict(self, session, elapsed) -> np.ndarray:
        session = pd.Series(session).reset_index(drop=True)
        elapsed = pd.Series(elapsed).reset_index(drop=True).astype(float)
        out = np.zeros(len(session), dtype=float)
        for name, model in self.models.items():
            if model is None:
                continue
            m = (session == name).to_numpy()
            if m.any():
                out[m] = model.predict(elapsed[m].to_numpy())
        return out


def _design(df: pd.DataFrame) -> tuple[np.ndarray, list, list]:
    """Stint fixed effects + per-compound age slopes."""
    stints = sorted(df["stint_uid"].unique())
    compounds = sorted(df["compound"].unique())
    n = len(df)
    X = np.zeros((n, len(stints) + len(compounds)))
    s_idx = {s: i for i, s in enumerate(stints)}
    X[np.arange(n), df["stint_uid"].map(s_idx).to_numpy()] = 1.0
    age = df["tyre_age"].to_numpy(dtype=float)
    for j, c in enumerate(compounds):
        m = (df["compound"] == c).to_numpy()
        X[m, len(stints) + j] = age[m]
    return X, stints, compounds


def _binned_isotonic(tau: np.ndarray, target: np.ndarray,
                     bin_s: float) -> IsotonicRegression | None:
    """Decreasing isotonic on coarse time bins (weighted by bin occupancy)."""
    if len(tau) == 0:
        return None
    bins = np.floor(tau / bin_s).astype(int)
    order = np.argsort(bins)
    b, t = bins[order], target[order]
    uniq, start = np.unique(b, return_index=True)
    means = np.array([t[s:e].mean() for s, e in
                      zip(start, list(start[1:]) + [len(t)])])
    counts = np.array([e - s for s, e in
                       zip(start, list(start[1:]) + [len(t)])], dtype=float)
    if len(uniq) < 2:
        return None
    centres = (uniq + 0.5) * bin_s
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(centres, means, sample_weight=counts)
    return iso


def fit_evolution(df: pd.DataFrame, *, n_iter: int = N_BACKFIT_ITERS,
                  bin_s: float = BIN_SECONDS,
                  min_laps: int = MIN_LAPS_FOR_EVO) -> EvolutionFit:
    """Backfit isotonic track evolution against per-compound degradation.

    `df` must be clean laps carrying `lap_time_fuel_corr`, `lap_in_stint`,
    `tyre_age`, `compound`, `stint_uid`, `driver`, `session`, `lap_start_s`.
    """
    d = df.reset_index(drop=True).copy()
    y = d["lap_time_fuel_corr"].to_numpy(dtype=float)
    tau = d["lap_start_s"].to_numpy(dtype=float)
    X, stints, compounds = _design(d)
    n_s = len(stints)

    fit = EvolutionFit()
    evo = np.zeros(len(d))
    deg = np.zeros(len(d))

    for it in range(n_iter):
        # -- step 1: evolution explains what degradation does not -----------
        # Driver-level (not stint-level) centring: pooling a driver's stints is
        # what makes the level of the track curve identifiable.
        resid = pd.Series(y - deg, index=d.index)
        centred = (resid - resid.groupby(d["driver"]).transform("mean")).to_numpy()

        models, evo = {}, np.zeros(len(d))
        fit.skipped = []
        for sess, grp in d.groupby("session", sort=False):
            idx = grp.index.to_numpy()
            if len(idx) < min_laps:
                models[sess] = None
                fit.skipped.append(sess)
                continue
            iso = _binned_isotonic(tau[idx], centred[idx], bin_s)
            models[sess] = iso
            if iso is not None:
                e = iso.predict(tau[idx])
                evo[idx] = e - e.mean()  # level belongs to the intercepts
        fit.models = models

        # -- step 2: degradation explains what evolution does not -----------
        beta, *_ = np.linalg.lstsq(X, y - evo, rcond=None)
        deg = X[:, n_s:] @ beta[n_s:]

        slopes = {c: float(beta[n_s + j]) for j, c in enumerate(compounds)}
        fit.deg_slopes = slopes
        fit.iterations.append(
            {
                "iter": it + 1,
                "rss": float(np.sum((y - evo - X @ beta) ** 2)),
                "evo_range_s": float(evo.max() - evo.min()) if len(evo) else 0.0,
                **slopes,
            }
        )

    rng = fit.iterations[-1]["evo_range_s"] if fit.iterations else 0.0
    if rng > MAX_PLAUSIBLE_EVO_S:
        log.warning("track evolution range %.2f s exceeds %.1f s — check the fit",
                    rng, MAX_PLAUSIBLE_EVO_S)
    if fit.skipped:
        log.info("flat evolution for sessions with <%d clean laps: %s",
                 min_laps, fit.skipped)
    return fit


# --------------------------------------------------------------------------
# Push laps: the observable that identifies evolution
# --------------------------------------------------------------------------
#
# **Why the long-run backfit fails at Monza-type sessions.**  It centres a
# driver's long-run laps and asks the isotonic to explain the residual drift
# over session time.  But run plans are not random in time: the early "long
# runs" are low-fuel installation and set-up runs, the late ones are heavy
# race simulations, so a driver's fuel-corrected long-run pace *rises* through
# the session even as the track improves.  The unknown per-stint fuel load is
# exactly what the estimator cannot see, and it is correlated with time.  A
# decreasing isotonic fitted to a rising signal returns a flat line: zero
# evolution, every tenth of it absorbed into the degradation slope, which
# comes out biased low.  Measured at Monza 2026: 0.00 s of evolution from the
# backfit, against ~1 s of improvement plainly visible in the push laps.
#
# **Push laps do not have that problem.**  A lap within ~1.5 s of a driver's
# session best is a low-fuel, fresh-tyre effort by construction — the only way
# to set it — so push laps at different times are comparable on the two things
# a long run is not: fuel load and tyre age.  What remains after removing the
# driver's level and the compound step is the track.  Track evolution is a
# property of the tarmac, not of the run plan, so what push laps measure
# applies to the long runs that are then fitted for degradation.


class ExpEvolution:
    """Track evolution as saturating rubber-in: evo(tau) = -A (1 - exp(-tau / T)).

    A parametric shape rather than a free isotonic because push laps are few
    and noisy (~0.3 s), and because the isotonic clips to a constant after the
    last push lap — precisely where the long runs are, so any evolution still
    happening there would be read as negative degradation.  Grip builds as
    rubber is laid and saturates; one amplitude and one time constant, both
    fitted, the amplitude constrained non-negative.
    """

    def __init__(self, A: float, T: float, c: float = 0.0):
        self.A, self.T, self.c = float(A), float(T), float(c)

    def predict(self, tau):
        tau = np.asarray(tau, dtype=float)
        return self.c - self.A * (1.0 - np.exp(-tau / self.T))

    MAX_AMPLITUDE_S = 3.0   # no practice session has evolved more than ~3 s

    @classmethod
    def fit(cls, tau: np.ndarray, rel: np.ndarray, *, T_grid=(600, 900, 1200, 1800, 2700, 3600, 5400)):
        best = None
        for T in T_grid:
            f = 1.0 - np.exp(-tau / T)
            X = np.column_stack([np.ones_like(tau), -f])
            beta, *_ = np.linalg.lstsq(X, rel, rcond=None)
            c, A = float(beta[0]), float(np.clip(beta[1], 0.0, cls.MAX_AMPLITUDE_S))
            if A != beta[1]:   # amplitude clipped: re-fit the level for that amplitude
                c = float(np.mean(rel + A * f))
            rss = float(np.sum((rel - (c - A * f)) ** 2))
            if best is None or rss < best[0]:
                best = (rss, A, T, c)
        _, A, T, c = best
        return cls(A, T, c)


def fit_evolution_push(laps_all: pd.DataFrame, event=None, *, bin_s: float = BIN_SECONDS,
                       margin_s: float = PUSH_MARGIN_S, min_laps: int = PUSH_MIN_LAPS,
                       min_bins: int = PUSH_MIN_BINS) -> EvolutionFit:
    """Isotonic track evolution per session from push laps.

    `laps_all` is the full lap table from `src.laps.build_lap_table` (all
    stints, before the long-run filter); `event` supplies the compound ladder's
    pace offsets so a session that switches to the SOFT late is not read as
    the track improving by a compound step.
    """
    fit = EvolutionFit()
    d = laps_all[laps_all["ok_accurate"] & laps_all["ok_not_pit"] & laps_all["ok_green"]
                 & laps_all["ok_compound"] & laps_all["lap_time_s"].notna()].copy()
    if d.empty:
        return fit
    offsets = {}
    if event is not None:
        try:
            from src.compounds import compound_prior

            cp = compound_prior(event)
            offsets = cp.pace_offsets(sorted(d["compound"].unique()))
        except Exception:
            offsets = {}
    d["y"] = (d["lap_time_s"] - d["compound"].map(offsets).fillna(0.0)
              - PUSH_AGE_S_PER_LAP * (pd.to_numeric(d["tyre_age"], errors="coerce").fillna(2.0) - 2.0).clip(0, 8))
    for sess, g in d.groupby("session", sort=False):
        best = g.groupby("driver")["y"].transform("min")
        push = g[g["y"] <= best + margin_s]
        push = push.assign(rel=push["y"] - push.groupby("driver")["y"].transform("mean"))
        tau = push["lap_start_s"].to_numpy(dtype=float)
        n_bins = len(np.unique(np.floor(tau / bin_s))) if len(tau) else 0
        if len(push) < min_laps or n_bins < min_bins:
            fit.models[sess] = None
            fit.skipped.append(sess)
            continue
        model = ExpEvolution.fit(tau, push["rel"].to_numpy(dtype=float))
        fit.models[sess] = model
        e = model.predict(tau)
        fit.iterations.append({"session": sess, "n_push_laps": int(len(push)), "n_bins": int(n_bins),
                               "evo_range_s": float(e.max() - e.min()), "A_s": model.A, "T_s": model.T,
                               "estimator": "push laps, exponential"})
    return fit


def fit_evolution_auto(clean: pd.DataFrame, laps_all: pd.DataFrame | None = None, event=None,
                       **kw) -> EvolutionFit:
    """Push-lap evolution where a session has enough push laps; the long-run
    backfit for the others.  The backfit's own range is kept as a diagnostic."""
    back = fit_evolution(clean, **kw)
    if laps_all is None:
        return back
    push = fit_evolution_push(laps_all, event)
    out = EvolutionFit(deg_slopes=dict(back.deg_slopes))
    for sess in sorted(set(list(back.models) + list(push.models))):
        m = push.models.get(sess)
        out.models[sess] = m if m is not None else back.models.get(sess)
        if m is None:
            out.skipped.append(sess)
    # degradation slopes with the chosen evolution removed (diagnostic)
    d = clean.reset_index(drop=True)
    if len(d):
        y = d["lap_time_fuel_corr"].to_numpy(dtype=float) - out.predict(d["session"], d["lap_start_s"])
        X, stints, compounds = _design(d)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.deg_slopes = {c: float(beta[len(stints) + j]) for j, c in enumerate(compounds)}
    rng_push = {it["session"]: it["evo_range_s"] for it in push.iterations}
    rng_back = back.iterations[-1]["evo_range_s"] if back.iterations else 0.0
    out.iterations = list(back.iterations) + [
        {"iter": "push", "estimator": "push laps", "rss": float("nan"),
         "evo_range_s": float(max(rng_push.values()) if rng_push else 0.0),
         "per_session": rng_push, "backfit_range_s": float(rng_back), **out.deg_slopes}]
    log.info("evolution: push-lap ranges %s (backfit %.2f s), sessions on backfit: %s",
             {k: round(v, 2) for k, v in rng_push.items()}, rng_back, out.skipped)
    return out


def add_evolution_correction(df: pd.DataFrame, fit: EvolutionFit) -> pd.DataFrame:
    """Attach `evo_s` and the fully corrected pace channel."""
    out = df.copy()
    out["evo_s"] = fit.predict(out["session"], out["lap_start_s"])
    out["lap_time_corr"] = out["lap_time_fuel_corr"] - out["evo_s"]
    return out


def evolution_curve(fit: EvolutionFit, session: str, t_min: float,
                    t_max: float, n: int = 200) -> pd.DataFrame:
    """Sampled curve for plotting."""
    t = np.linspace(t_min, t_max, n)
    model = fit.models.get(session)
    e = np.zeros(n) if model is None else model.predict(t)
    return pd.DataFrame({"lap_start_s": t, "evo_s": e - e.mean()})
