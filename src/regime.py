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

Measured with the same stint-fixed-effects estimator applied to both sessions,
the race/practice ratio on the seven scored 2026 weekends runs 0.32 (Belgium)
to 0.81 (Italy).  Pooled as one geometric mean it transferred 0.50-0.60 to
every weekend and sat below the self-measured value on six of seven, with a
mean absolute log error of 0.31 (+/-36%).  It is not a constant across
circuits, and part of the reason is visible in the weather: the practice long
runs and the race are run at different track temperatures, and degradation
is thermally sensitive at +2.5% per degree (`history.thermal_sensitivity`,
fitted on 99 circuit-compound-years of the archive).

**The transfer is now modelled, not pooled.**  For every donor weekend

    log(ratio_w) = log(r0) + beta * (T_race_w - T_practice_w) + e_w

with `beta` the measured thermal sensitivity, so what is pooled across donors
is the temperature-corrected residual `log(r0)` - the management part - and
it is pooled with the **median**, so one Belgium cannot drag every other
weekend 15-30% low.

**The thermal term now needs a forecast to exist** (`temperature_model="auto"`,
the default).  V2 stood the correction up on the circuit's archive race-day
mean temperature whenever no forecast was given, which is a prediction of the
race day dressed as a measurement: it moved the factor by up to 20% on the
strength of what the weather did in 2023-2025, and the archive mean is
systematically hotter than a practice afternoon at some circuits and cooler at
others.  With no forecast in hand the honest statement is that the temperature
is unknown, so the donors' raw log ratios are pooled with the median and no
temperature enters at all (`temperature["mode"] == "none"`).  Pass
`race_temp_c=` - a real forecast, on the day - and the full correction applies
(`"forecast"`).  `temperature_model=True` restores V2's archive behaviour
(`"archive"`) for the ablation; `False` forces plain median pooling.

**What the circuit itself has done is now evidence.**  The same ratio can be
measured on *previous years* of this circuit - historical practice against the
historical race, the pre-2026 fuel physics, the same stint-fixed-effects
estimator (`src.regime_history`).  That prior knows that Spa is a
management circuit and Monza is not, which no pool of other 2026 weekends can
tell you.  It is combined with the donor pool by precision weights on the log
scale, and carries a 0.15 ln term for the car/tyre generation change.

**The firewall still holds.**  The factor for a weekend is measured from *other*
weekends' races and never from the target weekend's; the target's own race
temperature and its own race laps are never read.  For the first weekend of a
season, with no donor race available, the module falls back to a documented
default with a wide credible interval, and says so.

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
REGIME_LN_SD_FLOOR = 0.15
RACE_TEMP_FORECAST_SD_C = 4.0     # least uncertainty a race-day forecast from the archive carries


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


def _fuel_and_distance(ev: Event | None, fuel_s_per_lap: float | None,
                       n_race_laps: int | None, d: pd.DataFrame | None = None) -> tuple:
    """Resolve the fuel term and the race distance from an event or explicitly.

    A 2026 weekend has both on its `Event`; a historical race (`src.regime_history`)
    has neither - it ran on different regulations and a different distance - so
    both are passed in.  The explicit values always win, which is what keeps the
    2026 path bit-identical: its callers pass none.
    """
    if fuel_s_per_lap is None:
        if ev is None:
            raise ValueError("regime: either an Event or an explicit fuel_s_per_lap is required")
        fuel_s_per_lap = get_prior(ev, "2026").s_per_lap
    if n_race_laps is None:
        if ev is not None:
            n_race_laps = ev.n_race_laps
        elif d is not None and len(d):
            n_race_laps = int(pd.to_numeric(d["lap_number"], errors="coerce").max())
        else:
            raise ValueError("regime: either an Event or an explicit n_race_laps is required")
    return float(fuel_s_per_lap), int(n_race_laps)


def race_track_evolution(d: pd.DataFrame, ev: Event | None = None, *,
                         fuel_s_per_lap: float | None = None,
                         n_race_laps: int | None = None) -> pd.Series:
    """Track evolution during a race, in seconds, indexed like `d`.

    The regime factor is a ratio of a practice degradation slope to a race
    one, and the practice side has always had track evolution removed
    (`src.evolution` backfits it).  The race side must too: within a stint,
    race lap and tyre age advance together, so an uncorrected race slope
    absorbs the whole evolution drift and comes out biased low by roughly
    that amount on every compound - measured at -0.087 s/lap at Barcelona
    2026 and -0.079 s/lap at Hungary 2026 after fuel correction.

    Unlike a practice session, a race identifies evolution cleanly and needs no
    backfitting.  All cars run the same lap at the same moment but at
    *different tyre ages*, because they pit at different times, so a race-lap
    fixed effect and a tyre-age slope are separately identified from the
    cross-section.

    The lap effect absorbs fuel burn as well as evolution, which is why `d["y"]`
    must be the *uncorrected* lap time here - correcting for fuel first and
    then removing a lap effect would remove it twice.  What is returned is the
    lap effect with the known fuel term added back, so the caller is left with
    evolution alone.
    """
    fuel, n_laps = _fuel_and_distance(ev, fuel_s_per_lap, n_race_laps, d)
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
    evo_by_lap = fe + fuel * (n_laps - levels)
    evo_by_lap = evo_by_lap - evo_by_lap.mean()
    # Evolution is physically smooth; the raw lap effects carry lap-to-lap
    # noise from whoever happened to be on track.  Smooth on a quadratic.
    coef = np.polyfit(levels, evo_by_lap, 2)
    return pd.Series(np.polyval(coef, d["lap_number"].to_numpy(dtype=float)),
                     index=d.index)


def _race_frame(race: pd.DataFrame, ev: Event | None = None, *,
                fuel_s_per_lap: float | None = None, n_race_laps: int | None = None,
                evo: pd.Series | None = None) -> pd.DataFrame:
    """Green-flag, non-pit, non-outlier race laps from stints long enough to score.

    Both corrections the practice side gets are applied here too - fuel *and*
    track evolution - so that `_stint_fe_slope` measures the same quantity in
    both regimes and their ratio measures the regime difference and nothing
    else.  See `race_track_evolution` for why the second one matters so much.

    `fuel_s_per_lap` / `n_race_laps` override what the `Event` says (a
    historical race under the previous regulations: `history.FUEL_S_PER_LAP_PRE2026`),
    and `evo` an already-computed per-lap evolution series indexed like the
    filtered frame.  With none of them given the behaviour is V2's exactly.
    """
    fuel, n_laps = _fuel_and_distance(ev, fuel_s_per_lap, n_race_laps, race)
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
    d["evo_s"] = (race_track_evolution(d, ev, fuel_s_per_lap=fuel, n_race_laps=n_laps)
                  if evo is None else pd.Series(evo).reindex(d.index).astype(float).fillna(0.0))
    laps_left = n_laps - d["lap_number"].astype(float)
    d["y"] = d["lap_time_s"] - d["evo_s"] + fuel * laps_left
    return d


def _practice_frame(clean: pd.DataFrame, ev: Event | None = None, *,
                    fuel_s_per_lap: float | None = None) -> pd.DataFrame:
    """Clean practice laps with fuel and track evolution taken back out.

    Both corrections are the ones the fitter itself applies, so the practice
    slope measured here is the same quantity the posterior reports.  A
    historical practice session passes its own `fuel_s_per_lap`; its `evo_s`
    must already be on the frame (`evolution.add_evolution_correction`).
    """
    if fuel_s_per_lap is None and ev is None:
        raise ValueError("regime: either an Event or an explicit fuel_s_per_lap is required")
    fuel = (float(fuel_s_per_lap) if fuel_s_per_lap is not None
            else get_prior(ev, "2026").s_per_lap)
    d = clean.copy()
    evo = d["evo_s"] if "evo_s" in d else 0.0
    d["y"] = d["lap_time_s"] - evo + fuel * d["lap_in_stint"]
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
    return measure_regime_frames(race, practice, ev=ev, event=ev.key)


def measure_regime_frames(race: pd.DataFrame, practice: pd.DataFrame, *,
                          ev: Event | None = None, event: str = "",
                          fuel_s_per_lap: float | None = None,
                          n_race_laps: int | None = None,
                          race_evo: pd.Series | None = None) -> RegimeMeasurement:
    """The ratio from two already-built lap tables, with the physics passed in.

    `measure_regime` is this function plus "read the 2026 weekend's parquets and
    take the fuel term and the distance off its `Event`"; `src.regime_history`
    is this function plus "load a 2023-2025 weekend from FastF1 and hand it the
    pre-2026 fuel physics".  One estimator, two regimes of input, so the
    historical ratios are comparable with the 2026 ones by construction.
    """
    rf = _race_frame(race, ev, fuel_s_per_lap=fuel_s_per_lap, n_race_laps=n_race_laps,
                     evo=race_evo)
    pf = _practice_frame(practice, ev, fuel_s_per_lap=fuel_s_per_lap)
    if rf.empty or pf.empty:
        return RegimeMeasurement(event=event)

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
        event=event, ratio=ratio, per_compound=per,
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
    temperature: dict = field(default_factory=dict)

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
            "donors": list(self.donor_detail), "temperature": dict(self.temperature),
        }


def _donor_temps(ev: Event) -> tuple:
    from src.history import practice_track_temp, race_track_temp
    return practice_track_temp(ev), race_track_temp(ev)


def _thermal_mode(temperature_model, race_temp_c) -> tuple:
    """Resolve `temperature_model` into (apply the thermal term, mode label).

    `"auto"` (the default) applies it only against a supplied race-day forecast.
    `True` is V2: fall back to the circuit's archive race-day mean when no
    forecast is given - which is a prediction, not a measurement, which is why
    it is no longer the default.  `False` never applies it.
    """
    if temperature_model is True:
        return True, ("forecast" if race_temp_c is not None else "archive")
    if temperature_model is False or temperature_model is None:
        return False, "none"
    if isinstance(temperature_model, str) and temperature_model.lower() == "auto":
        return (race_temp_c is not None), ("forecast" if race_temp_c is not None else "none")
    raise ValueError(f"temperature_model must be 'auto', True or False, not {temperature_model!r}")


def regime_prior(target: Event | str, *, donors: list | None = None,
                 race_temp_c: float | None = None, practice_temp_c: float | None = None,
                 temperature_model: bool | str = "auto", clean: pd.DataFrame | None = None,
                 circuit_prior_weight: float = 1.0,
                 use_circuit_history: bool = True) -> RegimeFactor:
    """The regime factor for a weekend from every weekend *except* the target.

    Excluding the target is not a formality.  The factor multiplies the
    degradation curve that the strategy recommendation is built on, so fitting
    it on the target race would let race data set the answer through the back
    door — which is exactly what the sealed-prediction protocol exists to
    prevent.  The target's own race temperature is not read either: `race_temp_c`
    is a *forecast*, supplied by the caller on the day, and without one no
    temperature term enters at all.

    `use_circuit_history` adds this circuit's own previous years (practice
    against race, measured by `src.regime_history` with the same estimator and
    the pre-2026 fuel physics) as a second arm, combined by precision weights on
    the log scale; `circuit_prior_weight` scales its precision (0 disables it).
    Both are cache-only reads: no network, and never the target's own race.
    """
    from src.history import circuit_prior, practice_track_temp, thermal_sensitivity

    ev = get_event(target) if isinstance(target, str) else target
    keys = list(donors) if donors is not None else [k for k in EVENTS if k != ev.key]
    keys = [k for k in keys if k != ev.key]
    thermal, mode = _thermal_mode(temperature_model, race_temp_c)
    beta = float(thermal_sensitivity().get("beta_per_c", 0.025)) if thermal else 0.0

    # -- this circuit's own previous years, if they have been measured --------
    cprior = None
    if use_circuit_history and circuit_prior_weight > 0:
        from src.regime_history import circuit_regime_prior
        try:
            cprior = circuit_regime_prior(ev)
        except Exception as exc:   # a corrupt cache entry must not break a weekend
            log.warning("circuit regime prior unavailable for %s: %s", ev.key, exc)

    ms, detail, resid = [], [], []
    for k in keys:
        m = measure_regime(k)
        # A ratio outside this band is a broken measurement, not a regime
        # difference: below it the race showed no degradation to divide by,
        # above it the race would have to degrade faster than a flat-out
        # practice run, which does not happen.
        if not (np.isfinite(m.ratio) and REGIME_RATIO_BAND[0] <= m.ratio <= REGIME_RATIO_BAND[1]):
            continue
        dev = EVENTS[k]
        t_p, t_r = _donor_temps(dev) if thermal else (None, None)
        dT = (t_r - t_p) if (t_p is not None and t_r is not None) else None
        r = float(np.log(m.ratio) - (beta * dT if dT is not None else 0.0))
        ms.append(m)
        resid.append(r)
        detail.append({"event": k, "ratio": round(float(m.ratio), 4), "compounds": m.usable_compounds,
                       "n_race_stints": m.n_race_stints, "t_practice_c": t_p, "t_race_c": t_r,
                       "delta_t_c": (round(dT, 1) if dT is not None else None),
                       "residual_ratio": round(float(np.exp(r)), 4)})

    if not ms:
        if cprior:
            # No donor race anywhere (the first weekend of a season), but this
            # circuit has been measured before: that is a real prior, not a default.
            ratio = float(np.clip(cprior["ratio"], 0.15, 1.2))
            return RegimeFactor(
                ratio=ratio, ln_sd=float(max(cprior["ln_sd"], REGIME_LN_SD_FLOOR)), measured=True,
                sources=[f"{ev.circuit} {y}" for y in cprior["years"]],
                label=f"circuit history only ({len(cprior['years'])} years)",
                derivation=(
                    f"no other 2026 weekend has both practice and race data on disk; "
                    f"{ev.circuit}'s own practice->race ratio in {cprior['years']} "
                    f"({', '.join(f'{y} {v:.2f}x' for y, v in cprior['by_year'].items())}) "
                    f"gives {ratio:.2f}x, widened for the car/tyre generation change"),
                temperature={"mode": "none", "modelled": False, "beta_per_c": 0.0,
                             "circuit_prior": dict(cprior, weight_on_circuit=1.0)},
            )
        return RegimeFactor(
            temperature={"mode": "none", "modelled": False, "beta_per_c": 0.0},
            derivation=(
                f"no weekend other than {ev.key} has both practice and race data "
                f"on disk, so the default {DEFAULT_RACE_REGIME_RATIO:.2f}x applies "
                f"with a wide +/-{DEFAULT_RACE_REGIME_LN_SD:.2f} log-scale band"
            ),
        )

    resid = np.array(resid)
    logs = np.log([m.ratio for m in ms])
    centre = float(np.median(resid))
    if len(resid) > 1:
        mad = float(np.median(np.abs(resid - centre))) * 1.4826
        spread = float(max(mad, np.std(resid, ddof=1) * 0.8))
    else:
        spread = DEFAULT_RACE_REGIME_LN_SD
    # -- the target's own temperatures: practice measured, race forecast --------
    # With no thermal term in play nothing here is looked up at all: that keeps
    # the default path offline and fast (no FastF1 weather reads) as well as honest.
    t_prac = practice_temp_c if practice_temp_c is not None else (practice_track_temp(ev, clean) if thermal else None)
    t_race, t_src, t_sd = race_temp_c, "forecast supplied", 0.0
    if not thermal:
        t_race, t_src = None, ("temperature model off" if temperature_model is False
                              else "no race-day forecast supplied: no temperature term")
    elif t_race is None:
        cp = circuit_prior(ev, probe_practice_temp=False)
        if cp.race_temps:
            t_race = float(np.mean(cp.race_temps))
            t_sd = float(max(np.std(cp.race_temps, ddof=1) if len(cp.race_temps) > 1 else RACE_TEMP_FORECAST_SD_C,
                             RACE_TEMP_FORECAST_SD_C))
            t_src = f"archive race-day mean at {ev.circuit} {cp.years}"
        else:
            t_src = "no forecast: practice temperature assumed"
    dT = (t_race - t_prac) if (t_prac is not None and t_race is not None) else 0.0
    mu_pool = centre + beta * dT
    sd_pool = float(np.sqrt(max(spread, REGIME_LN_SD_FLOOR) ** 2 + (beta * t_sd) ** 2))

    # -- combine the donor pool with this circuit's own history ----------------
    # Two independent measurements of the same quantity on the log scale: the
    # management regime other 2026 weekends show, and what this circuit itself
    # showed in 2023-2025.  Precision weights, so a circuit measured in three
    # years pulls harder than one measured in one, and a tight donor pool
    # (Belgium: 0.21 ln) is not overridden by a wide historical one.
    mu, ln_sd, cp_detail = mu_pool, sd_pool, None
    if cprior and np.isfinite(cprior.get("ratio", np.nan)) and cprior["ratio"] > 0:
        mu_c = float(np.log(cprior["ratio"]))
        sd_c = float(max(cprior["ln_sd"], REGIME_LN_SD_FLOOR))
        w_p = 1.0 / sd_pool ** 2
        w_c = float(circuit_prior_weight) / sd_c ** 2
        mu = float((w_p * mu_pool + w_c * mu_c) / (w_p + w_c))
        ln_sd = float(np.sqrt(1.0 / (w_p + w_c)))
        cp_detail = {"ratio": float(cprior["ratio"]), "ln_sd": sd_c, "years": list(cprior["years"]),
                     "n": int(cprior.get("n", len(cprior["years"]))),
                     "by_year": dict(cprior.get("by_year", {})),
                     "weight": float(circuit_prior_weight),
                     "weight_on_circuit": float(w_c / (w_p + w_c))}
    # The factor is never known better than +/-15%: two arms agreeing is not a
    # reason to tell the optimiser the regime is certain.
    ln_sd = float(max(ln_sd, REGIME_LN_SD_FLOOR))
    ratio = float(np.clip(np.exp(mu), 0.15, 1.2))
    geo = float(np.exp(logs.mean()))
    temp = {"mode": mode, "beta_per_c": beta, "t_practice_c": t_prac, "t_race_expected_c": t_race,
            "t_race_source": t_src, "t_race_forecast_sd_c": t_sd, "delta_t_c": float(dT),
            "residual_median": float(np.exp(centre)), "residual_spread_ln": float(spread),
            "pooled_ratio": float(np.exp(mu_pool)), "pooled_ln_sd": sd_pool,
            "pooled_geomean_ratio": geo, "modelled": bool(thermal)}
    if cp_detail:
        temp["circuit_prior"] = cp_detail
    label = (f"median of {len(ms)} donors, temperature-corrected" if thermal
             else f"median of {len(ms)} donors")
    if cp_detail:
        label += " + circuit history"
    return RegimeFactor(
        ratio=ratio, ln_sd=ln_sd, measured=True, label=label,
        sources=[m.event for m in ms] + ([f"{ev.circuit} {y}" for y in cp_detail["years"]] if cp_detail else []),
        derivation=(
            "race/practice degradation ratio, stint-fixed-effects estimator on both sessions; "
            + ("each donor's log ratio corrected by the thermal sensitivity "
               f"({beta:+.3f}/degC) times its race-minus-practice track temperature, " if thermal else "")
            + "residuals pooled with the median: "
            + ", ".join(f"{d['event']} {d['ratio']:.2f}x"
                        + (f" (dT {d['delta_t_c']:+.0f} degC -> {d['residual_ratio']:.2f})" if d.get("delta_t_c") is not None else "")
                        for d in detail)
            + f" -> management residual {np.exp(centre):.2f}x (spread {spread:.2f} ln); "
            + (f"at {ev.key} practice ran at {t_prac:.0f} degC and the race is expected at {t_race:.0f} degC "
               f"({t_src}), so the pool gives {np.exp(mu_pool):.2f}x" if (t_prac is not None and t_race is not None)
               else f"no temperature term ({t_src}); the pool gives {np.exp(mu_pool):.2f}x")
            + (f"; {ev.circuit}'s own practice->race ratio in {cp_detail['years']} is "
               f"{cp_detail['ratio']:.2f}x (+/-{cp_detail['ln_sd']:.2f} ln), which carries "
               f"{cp_detail['weight_on_circuit']:.0%} of the precision, so {ratio:.2f}x applies"
               if cp_detail else f", so {ratio:.2f}x applies")
            + f"; the old pooled geometric mean would have given {geo:.2f}x. {ev.key}'s own race is never used"
        ),
        donor_detail=detail, temperature=temp,
    )
