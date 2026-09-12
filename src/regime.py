"""Practice -> race regime transfer: what drivers actually do with a tyre.

A practice long run and a race stint are not the same experiment, and treating
them as the same is the single largest error in a naive strategy model.

In a long run the driver is *gathering data*: full push on every lap, on a hot
afternoon track, in clean air, with the engineers explicitly asking for a
representative worst case.  In the race the same driver on the same compound
lifts and coasts into the braking zones, short-shifts, manages surface
temperature through the first two laps of a stint, runs in another car's
turbulence, and is working to a fuel and energy-deployment plan.  Every one of
those reduces the energy going through the contact patch, and degradation is
driven by exactly that energy.

Measured here, with the same stint-fixed-effects estimator applied to both
sessions:

    Barcelona 2026    practice          race           race/practice
      SOFT            +0.261 s/lap      +0.124 s/lap       0.47
      MEDIUM          +0.295 s/lap      +0.087 s/lap       0.29
                                        pooled             0.37

Left uncorrected, a 3x overstatement of degradation raises the cumulative cost
of a stint quadratically in its length while pit loss stays flat, so the
optimiser answers by pitting as early as the rules allow and taking the maximum
number of stops.  That is precisely the failure this module removes.

**The firewall still holds.**  The factor for a weekend is measured from *other*
weekends' races and never from the target weekend's — the same thing a race
team does when it carries a correlation offset from the last round.  For the
first weekend of a season, with no donor race available, the module falls back
to a documented default with a wide credible interval, and says so.

The factor is returned as a *distribution*, not a point estimate, so the extra
uncertainty it introduces widens the strategy answer honestly instead of being
laundered into a suspiciously confident recommendation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DEFAULT_RACE_REGIME_LN_SD,
    DEFAULT_RACE_REGIME_RATIO,
    EVENTS,
    REGIME_MIN_PRACTICE_SLOPE,
    VALID_COMPOUNDS,
    Event,
    get_event,
)
from src.fuel import get_prior

log = logging.getLogger("degless.regime")

MIN_STINT_LAPS_FOR_RATE = 6
SLOW_LAP_MARGIN_S = 2.5  # tighter than the practice cascade: races have traffic
REGIME_RATIO_BAND = (0.05, 1.20)  # a donor ratio outside this is not believed


# --------------------------------------------------------------------------
# The estimator, applied identically to both sessions
# --------------------------------------------------------------------------


def _stint_fe_slope(df: pd.DataFrame, ycol: str) -> dict:
    """Per-compound degradation slope with stint fixed effects.

    Demeaning within a stint absorbs the stint's level — its fuel load, engine
    mode, track state and driver — so what is left is purely how lap time moves
    with tyre age *inside* a stint.  It is the same quantity in practice and in
    the race, which is the whole point: the ratio of the two is then a clean
    measure of the regime difference and not of anything else.
    """
    out = {}
    for c in sorted(df["compound"].dropna().unique()):
        g = df[df["compound"] == c]
        if g["stint_uid"].nunique() < 2 or len(g) < 12:
            continue
        y = g[ycol] - g.groupby("stint_uid")[ycol].transform("mean")
        x = g["tyre_age"] - g.groupby("stint_uid")["tyre_age"].transform("mean")
        denom = float(np.sum(x ** 2))
        if denom <= 0:
            continue
        out[c] = {
            "slope": float(np.sum(x * y) / denom),
            "n_laps": int(len(g)),
            "n_stints": int(g["stint_uid"].nunique()),
        }
    return out


def race_track_evolution(d: pd.DataFrame, ev: Event) -> pd.Series:
    """Track evolution during a race, in seconds, indexed like `d`.

    **This function exists because its absence was the largest single error in
    this project's math.**  The regime factor is a ratio of a practice
    degradation slope to a race one, and the practice side has always had track
    evolution removed (`src.evolution` backfits it).  The race side did not.
    Within a stint, race lap and tyre age advance together, so an uncorrected
    race slope absorbs the whole evolution drift and comes out biased low by
    roughly that amount on every compound - measured at -0.087 s/lap at
    Barcelona 2026 and -0.079 s/lap at Hungary 2026 after fuel correction.  At
    Hungary that was enough to drive the measured race degradation *negative*
    (SOFT -0.022 s/lap), which is not a thing tyres do.  Dividing a corrected
    numerator by an uncorrected denominator gave 0.37 where the honest answer
    is 0.57, and every downstream stint length inherited the error.

    Unlike a practice session, a race identifies evolution cleanly and needs no
    backfitting.  All cars run the same lap at the same moment but at
    *different tyre ages*, because they pit at different times, so a race-lap
    fixed effect and a tyre-age slope are separately identified from the
    cross-section.  Measured on both 2026 weekends, 40-57% of the variance in
    tyre age survives projecting out driver and lap effects, and the median
    within-lap spread of tyre age is 3-7 laps: ample.

    The lap effect absorbs fuel burn as well as evolution, which is why `d["y"]`
    must be the *uncorrected* lap time here - correcting for fuel first and
    then removing a lap effect would remove it twice.  What is returned is the
    lap effect with the known fuel term added back, so the caller is left with
    evolution alone.
    """
    fp = get_prior(ev, "2026")
    if d.empty or d["lap_number"].nunique() < 5:
        return pd.Series(0.0, index=d.index)
    drv = pd.get_dummies(d["driver"], drop_first=True).astype(float)
    lapc = pd.Categorical(d["lap_number"].astype(int))
    lap = pd.get_dummies(lapc, drop_first=True).astype(float)
    comp = pd.get_dummies(d["compound"]).astype(float)
    age = d["tyre_age"].to_numpy(dtype=float)
    ageX = np.column_stack([comp[c].to_numpy() * age for c in comp.columns])
    X = np.column_stack([np.ones(len(d)), ageX, comp.to_numpy()[:, 1:],
                         drv.to_numpy(), lap.to_numpy()])
    y = d["lap_time_s"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    n_lap = lap.shape[1]
    fe = np.concatenate([[0.0], beta[len(beta) - n_lap:]])
    levels = np.asarray(lapc.categories, dtype=float)
    # Add the known fuel effect back so what is left is evolution alone, and
    # centre it: only the shape matters, the level is absorbed downstream.
    evo_by_lap = fe + fp.s_per_lap * (ev.n_race_laps - levels)
    evo_by_lap = evo_by_lap - evo_by_lap.mean()
    # Evolution is physically smooth; the raw lap effects carry lap-to-lap
    # noise from whoever happened to be on track.  Smooth on a quadratic.
    coef = np.polyfit(levels, evo_by_lap, 2)
    return pd.Series(np.polyval(coef, d["lap_number"].to_numpy(dtype=float)),
                     index=d.index)


def _race_frame(race: pd.DataFrame, ev: Event) -> pd.DataFrame:
    """Green-flag, non-pit, non-outlier race laps from stints long enough to score.

    Both corrections the practice side gets are applied here too - fuel *and*
    track evolution - so that `_stint_fe_slope` measures the same quantity in
    both regimes and their ratio measures the regime difference and nothing
    else.  See `race_track_evolution` for why the second one matters so much.
    """
    fp = get_prior(ev, "2026")
    d = race[
        race["is_accurate"].fillna(False).astype(bool)
        & ~race["pit_in"].fillna(False).astype(bool)
        & ~race["pit_out"].fillna(False).astype(bool)
        & (race["track_status"].astype(str) == "1")
        & race["compound"].isin(VALID_COMPOUNDS)
        & race["lap_time_s"].notna()
    ].copy()
    if d.empty:
        return d
    med = d.groupby("driver")["lap_time_s"].transform("median")
    d = d[d["lap_time_s"] < med + SLOW_LAP_MARGIN_S].copy()
    n = d.groupby("stint_uid")["lap_number"].transform("size")
    d = d[n >= MIN_STINT_LAPS_FOR_RATE].copy()
    if d.empty:
        return d
    d["evo_s"] = race_track_evolution(d, ev)
    laps_left = ev.n_race_laps - d["lap_number"].astype(float)
    d["y"] = d["lap_time_s"] - d["evo_s"] + fp.s_per_lap * laps_left
    return d


def _practice_frame(clean: pd.DataFrame, ev: Event) -> pd.DataFrame:
    """Clean practice laps with fuel and track evolution taken back out.

    Both corrections are the ones the fitter itself applies, so the practice
    slope measured here is the same quantity the posterior reports.
    """
    fp = get_prior(ev, "2026")
    d = clean.copy()
    evo = d["evo_s"] if "evo_s" in d else 0.0
    d["y"] = d["lap_time_s"] - evo + fp.s_per_lap * d["lap_in_stint"]
    return d


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass
class RegimeMeasurement:
    event: str = ""
    ratio: float = np.nan
    per_compound: dict = field(default_factory=dict)
    n_race_stints: int = 0
    n_practice_stints: int = 0
    usable_compounds: list = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        rows = [{"compound": c, **v} for c, v in sorted(self.per_compound.items())]
        return pd.DataFrame(rows)


def measure_regime(event: Event | str, *, race: pd.DataFrame | None = None,
                   practice: pd.DataFrame | None = None) -> RegimeMeasurement:
    """Race-to-practice degradation ratio for one weekend, per compound and pooled.

    Compounds whose practice slope is below `REGIME_MIN_PRACTICE_SLOPE` are
    excluded: dividing by a near-zero denominator produces a ratio with no
    information in it, and Hungary 2026 degrades slowly enough that every
    compound there is in that regime.
    """
    ev = get_event(event) if isinstance(event, str) else event
    if race is None:
        p = DATA_PROCESSED / f"laps_{ev.key}_race.parquet"
        if not p.exists():
            return RegimeMeasurement(event=ev.key)
        race = pd.read_parquet(p)
    if practice is None:
        p = DATA_PROCESSED / f"clean_{ev.key}_practice.parquet"
        if not p.exists():
            return RegimeMeasurement(event=ev.key)
        practice = pd.read_parquet(p)

    rf, pf = _race_frame(race, ev), _practice_frame(practice, ev)
    if rf.empty or pf.empty:
        return RegimeMeasurement(event=ev.key)

    rs, ps = _stint_fe_slope(rf, "y"), _stint_fe_slope(pf, "y")

    per, w_num, w_den = {}, 0.0, 0.0
    for c in sorted(set(rs) & set(ps)):
        pslope, rslope = ps[c]["slope"], rs[c]["slope"]
        usable = pslope >= REGIME_MIN_PRACTICE_SLOPE
        per[c] = {
            "practice_s_per_lap": pslope,
            "race_s_per_lap": rslope,
            "ratio": (rslope / pslope) if usable else np.nan,
            "n_practice_laps": ps[c]["n_laps"],
            "n_race_laps": rs[c]["n_laps"],
            "usable": bool(usable),
        }
        if usable:
            # Pool on the *slopes*, not on the ratios: a ratio whose denominator
            # is small is wildly noisy, and averaging ratios would let it
            # dominate.  This is the slope-weighted ratio, which is the total
            # degradation the race showed over the total practice predicted.
            w = ps[c]["n_laps"]
            w_num += rslope * w
            w_den += pslope * w

    ratio = float(w_num / w_den) if w_den > 0 else float("nan")
    return RegimeMeasurement(
        event=ev.key, ratio=ratio, per_compound=per,
        n_race_stints=int(rf["stint_uid"].nunique()),
        n_practice_stints=int(pf["stint_uid"].nunique()),
        usable_compounds=[c for c, v in per.items() if v["usable"]],
    )


# --------------------------------------------------------------------------
# The transferable prior
# --------------------------------------------------------------------------


@dataclass
class RegimeFactor:
    """A distribution over the practice -> race degradation multiplier."""

    ratio: float = DEFAULT_RACE_REGIME_RATIO
    ln_sd: float = DEFAULT_RACE_REGIME_LN_SD
    sources: list = field(default_factory=list)
    measured: bool = False
    label: str = "default (no donor race)"
    derivation: str = ""
    donor_detail: list = field(default_factory=list)

    def draws(self, n: int, *, seed: int = 0) -> np.ndarray:
        """`n` LogNormal draws — the multiplier applied per posterior draw.

        Sampling rather than multiplying by the mean is what keeps the
        strategy's credible intervals honest: the regime factor is genuinely
        uncertain, and a plan that only wins under an optimistic factor should
        show a lower win probability, not the same one.
        """
        rng = np.random.default_rng(seed)
        return np.exp(rng.normal(np.log(max(self.ratio, 1e-6)), self.ln_sd, size=n))

    @property
    def p05(self) -> float:
        return float(self.ratio * np.exp(-1.645 * self.ln_sd))

    @property
    def p95(self) -> float:
        return float(self.ratio * np.exp(1.645 * self.ln_sd))

    def as_dict(self) -> dict:
        return {
            "ratio": float(self.ratio), "ln_sd": float(self.ln_sd),
            "p05": self.p05, "p95": self.p95,
            "measured": bool(self.measured), "label": self.label,
            "sources": list(self.sources), "derivation": self.derivation,
            "donors": list(self.donor_detail),
        }


def regime_prior(target: Event | str, *, donors: list | None = None) -> RegimeFactor:
    """Pool the regime factor from every weekend *except* the target.

    Excluding the target is not a formality.  The factor multiplies the
    degradation curve that the strategy recommendation is built on, so fitting
    it on the target race would let race data set the answer through the back
    door — which is exactly what the sealed-prediction protocol exists to
    prevent.
    """
    ev = get_event(target) if isinstance(target, str) else target
    keys = list(donors) if donors is not None else [k for k in EVENTS if k != ev.key]
    keys = [k for k in keys if k != ev.key]

    ms, detail = [], []
    for k in keys:
        m = measure_regime(k)
        # A ratio outside this band is a broken measurement, not a regime
        # difference: below it the race showed no degradation to divide by
        # (Hungary 2026 pools to -0.11), above it the race would have to
        # degrade faster than a flat-out practice run, which does not happen.
        if np.isfinite(m.ratio) and REGIME_RATIO_BAND[0] <= m.ratio <= REGIME_RATIO_BAND[1]:
            ms.append(m)
            detail.append({
                "event": k, "ratio": round(float(m.ratio), 4),
                "compounds": m.usable_compounds,
                "n_race_stints": m.n_race_stints,
            })

    if not ms:
        return RegimeFactor(
            derivation=(
                f"no weekend other than {ev.key} has both practice and race data "
                f"on disk, so the default {DEFAULT_RACE_REGIME_RATIO:.2f}x applies "
                f"with a wide +/-{DEFAULT_RACE_REGIME_LN_SD:.2f} log-scale band"
            ),
        )

    logs = np.log([m.ratio for m in ms])
    ratio = float(np.exp(logs.mean()))
    # With one donor there is no between-weekend spread to measure, so the
    # default width is kept rather than reporting a fictitious zero.
    spread = float(logs.std(ddof=1)) if len(logs) > 1 else DEFAULT_RACE_REGIME_LN_SD
    ln_sd = float(max(spread, 0.15))
    return RegimeFactor(
        ratio=ratio, ln_sd=ln_sd, sources=[m.event for m in ms], measured=True,
        label=f"measured on {', '.join(m.event for m in ms)}",
        derivation=(
            "race/practice degradation ratio, stint-fixed-effects estimator on "
            "both sessions, geometric mean over "
            + ", ".join(f"{m.event} {m.ratio:.2f}x" for m in ms)
            + f"; applied to {ev.key}, whose own race is never used"
        ),
        donor_detail=detail,
    )
