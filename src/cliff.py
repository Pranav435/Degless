"""Within-stint pace collapse, and the grip budget read off it.

V2 estimated the grip budget — the seconds of lap time a tyre surrenders
before it falls off — as *race degradation rate times the longest stint the
compound was run to*.  That product is almost never the budget.  It is the
budget only when that longest stint actually reached the cliff, and at a
low-degradation circuit the longest stint ends for strategic reasons tens of
laps before it: Australia's MEDIUM went 0.025 s/lap x 28 laps = 0.7 s, which
says nothing about the MEDIUM's grip budget except that it exceeds 0.7 s.
V2 knew this and took the largest such product across weekends as a lower
bound, which is honest but wasteful: on six of seven weekends the bound was
below the 3.0 s informativeness threshold, so the budget the strategy model
shipped with was the *prior*, 3.80/3.83/3.84 s, on every compound.

The missing measurement is the cliff itself.  A stint that ends *because the
tyre was finished* looks nothing like a stint that ends because the pit wall
called a stop: the last few laps break away from the stint's own trend.  That
break is observable on every race stint long enough to have a trend, it is
what the grip budget is defined by, and nothing in V2 looked for it.

**The detector.**  On one stint's green, accurate, non-pit laps, fuel- and
evolution-corrected and levelled on its own first two laps, fit

    linear   y = b0 + b1 * age                       (the null: on-trend)
    hinge    y = b0 + b1 * age + b2 * max(0, age-k)  (k scanned, 1-lap grid)

A **collapse** is called only when all four of these hold:

    the hinge cuts the residual sum of squares by >= 25 %     (a real break)
    slope_post - slope_pre >= 0.15 s/lap                      (a steep break)
    the last three laps sit >= 0.6 s above the pre-knee trend  (a costly break)
    the stint ended within 3 laps of the knee                  (they boxed)

The last condition is what separates a cliff from a slow puncture, a damaged
floor or a long spell in traffic: a real cliff ends the stint, because no team
leaves a car out on a tyre that is 2 s a lap down.  A stint whose last three
laps sit within +/-0.4 s of its trend is the opposite case, **strategic**: the
tyre was still on its trend when the car came in, so the stop was a decision
and the stint's total loss is a *lower bound* on the budget.  Everything else
is **undetermined** — and a stint that ran to the chequered flag is censored by
the race, not by the tyre, so it never gets a collapse call however its last
laps looked.

**What a lower bound is allowed to be.**  A bound is only a bound if the stint
was on a *trend* when it reached that loss, so `cum_loss_at_end_s` is read off
the trend and, where the profile broke away, is read at the knee rather than at
the last lap.  And a stint whose trend is steeper than `TREND_RATE_MAX_S`
(0.5 s/lap) is not a tyre degrading at all — it is a damaged car, a fuel save,
or twenty laps in a DRS train — so it contributes neither an observation nor a
bound.  Without those two rules one Belgium stint claims a 5.5 s grip budget and
sets the SOFT's for every weekend.

**The estimator.**  Collapse stints give the budget directly; strategic and
undetermined stints give a lower bound.  That is a textbook right-censored
sample, so `grip_budget_estimate` maximises the censored log-normal likelihood

    sum_obs log phi((log o - log b)/s) + sum_bound log(1 - Phi((log d - log b)/s))

plus a log-normal prior on `b`, on a grid over the plausible band.  With no
collapse observation the likelihood has no mode of its own and the estimator
falls back to exactly V2's rule — the largest lower bound if it is informative,
otherwise the prior — so the two are nested and the ablation is meaningful.

The budget is floored at the largest lower bound: a tyre that demonstrably gave
up 4.1 s without falling off cannot have a 3.8 s budget, whatever the
collapse observations say.  That constraint is the whole reason the censored
likelihood is worth writing down rather than averaging the observations.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import GRIP_BUDGET_S, VALID_COMPOUNDS, Event, get_event

__all__ = [
    "detect_stint_collapse",
    "race_collapses",
    "grip_budget_estimate",
    "grip_budgets_by_compound",
    "budget_ratio_metrics",
    "COLLAPSE_ROW_COLUMNS",
]

MIN_STINT_LAPS = 8           # a stint with fewer green laps has no trend to break away from
RSS_DROP_MIN = 0.25          # the hinge must cut the linear model's RSS by this share
SLOPE_BREAK_MIN_S = 0.15     # s/lap the post-knee segment must add to the pre-knee one
DELTA_LAST3_MIN_S = 0.6      # s the last three laps must sit above the pre-knee trend
END_WITHIN_LAPS = 3          # a cliff ends the stint: the box must follow the knee this closely
ON_TREND_TOL_S = 0.4         # within this of the trend, the stint was still on it: a decision, not a cliff
OUTLIER_MARGIN_S = 8.0       # a lap this far above the driver's race median is an off, not degradation
RSS_FLOOR = 1e-9             # a noiseless profile cannot be "improved" on: no hinge selection
TREND_RATE_MAX_S = 0.5       # a "trend" steeper than this is a damaged car or a fuel save, not a tyre

BUDGET_BAND_S = (2.5, 5.0)   # outside this a "budget" is a broken measurement, not a tyre
BUDGET_INFORMATIVE_S = 3.0   # V2's threshold: below it a lower bound says nothing useful
BUDGET_OBS_LN_SD = 0.20      # spread of per-stint collapse losses around the compound's budget
BUDGET_GRID_STEP_S = 0.01

COLLAPSE_ROW_COLUMNS = [
    "driver", "team", "compound", "stint", "n", "kind", "knee_age", "collapse",
    "censored_by_flag", "slope_pre", "slope_post", "delta_last3_s",
    "cum_loss_at_knee_s", "cum_loss_at_end_s", "rate_s_per_lap", "in_lap", "sc_stop",
]


# --------------------------------------------------------------------------
# One stint
# --------------------------------------------------------------------------


def _ols(X: np.ndarray, y: np.ndarray) -> tuple:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid @ resid)


def _blank(n: int, why: str) -> dict:
    return {"n": int(n), "slope_pre": float("nan"), "slope_post": float("nan"),
            "knee_age": float("nan"), "delta_last3_s": float("nan"),
            "collapse": False, "kind": "undetermined",
            "cum_loss_at_knee_s": float("nan"), "cum_loss_at_end_s": float("nan"),
            "rate_s_per_lap": float("nan"), "rss_drop": float("nan"), "why": why}


def detect_stint_collapse(stint_df: pd.DataFrame, *, fuel_s_per_lap: float,
                          evo=None, min_laps: int = MIN_STINT_LAPS) -> dict:
    """Did this stint end on the cliff, or did the pit wall end it?

    `stint_df` is one stint's green, non-pit, accurate laps with `tyre_age`,
    `lap_time_s` and (where available) `lap_number`.  `evo` is an optional
    per-lap track-evolution series in seconds — a `pd.Series` indexed like
    `stint_df`, or an array in its row order — which is subtracted.

    The profile the decision is taken on is the *corrected, stint-centred*
    one: fuel burn added back (within a stint the car is getting lighter at a
    known rate, which would otherwise cancel part of the degradation),
    evolution removed (the track is getting faster, same problem), and the
    level set by the stint's own first two laps so that what is left is the
    lap time this tyre has surrendered since it was fresh.  Cumulative loss
    measured any other way is not a grip budget.

    Returns the keys listed in the module docstring plus `rss_drop` (the share
    of the linear model's residual sum of squares the hinge removed) and `why`
    (a sentence for the live engine's alert text and for the report).
    """
    need = ("tyre_age", "lap_time_s")
    if stint_df is None or len(stint_df) == 0 or any(c not in stint_df for c in need):
        return _blank(0 if stint_df is None else len(stint_df), "no usable laps")
    d = stint_df.dropna(subset=list(need))
    if evo is not None:
        ev_ser = evo if isinstance(evo, pd.Series) else pd.Series(np.asarray(evo, dtype=float),
                                                                 index=stint_df.index)
        d = d.assign(_evo=ev_ser.reindex(d.index).astype(float).fillna(0.0))
    else:
        d = d.assign(_evo=0.0)
    d = d.sort_values("tyre_age")
    n = len(d)
    if n < int(min_laps):
        return _blank(n, f"only {n} green laps, fewer than the {int(min_laps)} a trend needs")

    a = d["tyre_age"].to_numpy(dtype=float)
    # Fuel burn is a function of the *race* lap, so correct on it where the
    # column is there; inside a stint tyre age advances with it anyway.
    burnt = (d["lap_number"].to_numpy(dtype=float) - float(d["lap_number"].min())
             if "lap_number" in d else a - a[0])
    y = d["lap_time_s"].to_numpy(dtype=float) + float(fuel_s_per_lap) * burnt - d["_evo"].to_numpy(dtype=float)
    y = y - float(np.mean(y[:2]))

    # -- the null: one straight line through the whole stint ---------------
    ones = np.ones(n)
    b_lin, rss_lin = _ols(np.column_stack([ones, a]), y)

    # -- the pre-knee trend, fitted where a collapse cannot have reached ---
    # A collapse call requires the box within 3 laps of the knee, so the first
    # n-4 laps of a collapsing stint are pre-knee by construction.  That is
    # what makes this window the right place to measure the break from.
    m = min(n, max(5, n - 4))
    b_pre, _ = _ols(np.column_stack([ones[:m], a[:m]]), y[:m])
    last3 = slice(max(n - 3, 0), n)
    delta_last3 = float(np.mean(y[last3] - (b_pre[0] + b_pre[1] * a[last3])))

    # -- the hinge: where does the stint break away from itself? -----------
    lo, hi = int(math.ceil(a[0])) + 3, int(math.floor(a[-1])) - 1
    grid = [k for k in range(lo, hi + 1) if int((a <= k).sum()) >= 3 and int((a > k).sum()) >= 2]
    best_beta, best_rss, knee = None, np.inf, float("nan")
    for k in grid:
        X = np.column_stack([ones, a, np.maximum(a - k, 0.0)])
        beta, rss = _ols(X, y)
        if rss < best_rss:
            best_beta, best_rss, knee = beta, rss, float(k)
    if best_beta is None:
        out = _blank(n, "no knee the stint is long enough to scan")
        out["rate_s_per_lap"] = out["slope_pre"] = float(b_lin[1])
        out["delta_last3_s"] = delta_last3
        if float(b_lin[1]) <= TREND_RATE_MAX_S:
            out["cum_loss_at_end_s"] = float(max(b_lin[0] + b_lin[1] * a[-1], 0.0))
            out["kind"] = "strategic" if abs(delta_last3) <= ON_TREND_TOL_S else "undetermined"
        return out

    slope_pre, slope_post = float(best_beta[1]), float(best_beta[1] + best_beta[2])
    rss_drop = float(1.0 - best_rss / rss_lin) if rss_lin > RSS_FLOOR * max(n, 1) else 0.0
    hinge_selected = bool(rss_drop >= RSS_DROP_MIN)

    def hinge_at(age: float) -> float:
        return float(best_beta[0] + best_beta[1] * age + best_beta[2] * max(age - knee, 0.0))

    # The two cumulative losses, both on the corrected, stint-centred profile.
    # `cum_loss_at_end_s` is the loss the stint reached *while still on its own
    # trend*: where the profile broke away, the trend is only credible up to the
    # knee, so the value is taken there.  That is what makes it a sound
    # right-censored bound — "the tyre gave up this much and was still behaving
    # like a tyre" — rather than a measure of whatever ended the stint.  Reading
    # it to the last lap instead lets one car stuck in a DRS train, or one
    # retirement limping home, claim a 5.5 s grip budget.
    cum_knee = float(max(hinge_at(knee), 0.0))
    cum_end = cum_knee if hinge_selected else float(max(b_lin[0] + b_lin[1] * a[-1], 0.0))

    trend = slope_pre if hinge_selected else float(b_lin[1])
    if trend > TREND_RATE_MAX_S:
        # Nothing this module measures means anything on a stint whose *trend*
        # is 0.5 s/lap: no tyre degrades that way, and reading a grip budget off
        # it would let one damaged car or one fuel-save stint set the compound's
        # budget.  The slopes are reported, the losses are not.
        out = _blank(n, f"trend of {trend:+.2f} s/lap is not degradation: no budget read off this stint")
        out.update({"slope_pre": slope_pre, "slope_post": slope_post, "knee_age": knee,
                    "delta_last3_s": delta_last3, "rate_s_per_lap": trend, "rss_drop": rss_drop})
        return out

    collapse = bool(hinge_selected
                    and (slope_post - slope_pre) >= SLOPE_BREAK_MIN_S - 1e-9
                    and delta_last3 >= DELTA_LAST3_MIN_S - 1e-9
                    and (a[-1] - knee) <= END_WITHIN_LAPS)
    if collapse:
        kind, why = "collapse", (f"pace broke away at age {knee:.0f}: {slope_pre:+.3f} -> {slope_post:+.3f} s/lap, "
                                 f"last three laps {delta_last3:+.2f} s off the trend, boxed "
                                 f"{a[-1] - knee:.0f} laps later")
    elif abs(delta_last3) <= ON_TREND_TOL_S:
        kind, why = "strategic", (f"still on its trend at the stop ({delta_last3:+.2f} s over the last three laps): "
                                 f"the {cum_end:.2f} s it had given up is a lower bound on the budget")
    else:
        kind, why = "undetermined", (f"last three laps {delta_last3:+.2f} s off the trend but "
                                     + ("no clean break in the profile" if not hinge_selected
                                        else f"boxed {a[-1] - knee:.0f} laps after the break"))
    return {"n": int(n), "slope_pre": slope_pre, "slope_post": slope_post, "knee_age": knee,
            "delta_last3_s": delta_last3, "collapse": collapse, "kind": kind,
            "cum_loss_at_knee_s": cum_knee, "cum_loss_at_end_s": cum_end,
            "rate_s_per_lap": (slope_pre if hinge_selected else float(b_lin[1])),
            "rss_drop": rss_drop, "why": why}


# --------------------------------------------------------------------------
# A whole race
# --------------------------------------------------------------------------


def _correction_by_lap(race: pd.DataFrame, ev: Event):
    """A race-lap -> correction-seconds callable, or None.

    What is taken from `regime._race_frame` is its *whole* correction, read off
    as `y - lap_time_s`, not its `evo_s` column alone.  The regime module's
    fuel and evolution terms are entangled — `race_track_evolution` adds the
    fuel term to a lap fixed effect that already contains it, and `_race_frame`
    cancels the excess when it adds the fuel term back with the opposite sign —
    so rebuilding either half here would double-count.  Reading the net
    correction keeps this module measuring degradation on exactly the profile
    the race degradation slope and the regime factor are measured on, whatever
    the two halves do internally.  (Checked at Barcelona 2026: the net
    correction steepens a within-stint slope by +0.055 s/lap and the corrected
    stint-FE slopes, 0.125/0.173/0.209, match `history.race_deg_slopes`'
    independent two-way fit, 0.110/0.157/0.172.)

    Both terms are smooth in race lap — a quadratic evolution shape plus a
    linear fuel term — so a quadratic through the laps the frame kept
    reproduces it and extends it to the laps it dropped, which matters here
    because the laps it drops as "slow" are precisely the collapsed ones.
    """
    from src.regime import _race_frame

    try:
        rf = _race_frame(race, ev)
    except Exception:
        return None
    if rf is None or rf.empty or "y" not in rf:
        return None
    corr = (rf["y"] - rf["lap_time_s"]).groupby(rf["lap_number"]).mean()
    if len(corr) < 3:
        return None
    coef = np.polyfit(corr.index.to_numpy(dtype=float), corr.to_numpy(dtype=float), 2)
    return lambda laps: np.polyval(coef, np.asarray(laps, dtype=float))


def race_collapses(race_laps: pd.DataFrame, *, event: Event | str | None = None,
                   fuel_s_per_lap: float | None = None, n_race_laps: int | None = None,
                   min_laps: int = MIN_STINT_LAPS) -> pd.DataFrame:
    """One row per race stint long enough to score: did the tyre end it?

    With `event` the fuel physics and the track-evolution correction come from
    the event (the same two corrections `regime._race_frame` applies, so a
    stint's loss here is the same quantity the regime factor is measured on);
    without it only a plain fuel correction is applied and `fuel_s_per_lap`
    must be supplied.

    Unlike `_race_frame` this keeps the slow laps.  Dropping laps more than
    2.5 s above a driver's median is right when measuring a *slope* and fatal
    when looking for a *cliff*, since a tyre past its cliff is 2-4 s off.  The
    only laps dropped are those `OUTLIER_MARGIN_S` beyond it, which are offs
    and near-misses rather than degradation.

    Columns: `COLLAPSE_ROW_COLUMNS`.  `in_lap` is the lap the car pitted at the
    end of the stint (NaN if it did not), `sc_stop` whether that stop was taken
    under a safety car, and `censored_by_flag` whether the stint ran to the
    chequered flag — the three facts that say whether a stint's end is evidence
    about the tyre at all.
    """
    from src.strategy import is_sc_status

    ev = None
    if event is not None:
        ev = get_event(event) if isinstance(event, str) else event
        if fuel_s_per_lap is None:
            from src.fuel import get_prior
            fuel_s_per_lap = float(get_prior(ev, "2026").s_per_lap)
        if n_race_laps is None:
            n_race_laps = int(ev.n_race_laps)
    if fuel_s_per_lap is None:
        fuel_s_per_lap = 0.0
    if race_laps is None or race_laps.empty:
        return pd.DataFrame(columns=COLLAPSE_ROW_COLUMNS)
    if n_race_laps is None:
        n_race_laps = int(race_laps["lap_number"].max())

    raw = race_laps.copy()
    raw["stint"] = raw["stint"].astype(float)
    ends = raw.groupby(["driver", "stint"])["lap_number"].max()
    status = raw.set_index(["driver", "lap_number"])["track_status"].astype(str).to_dict()
    corr_fn = _correction_by_lap(raw, ev) if ev is not None else None

    d = raw[
        raw["is_accurate"].fillna(False).astype(bool)
        & ~raw["pit_in"].fillna(False).astype(bool)
        & ~raw["pit_out"].fillna(False).astype(bool)
        & (raw["track_status"].astype(str) == "1")
        & raw["compound"].isin(VALID_COMPOUNDS)
        & raw["lap_time_s"].notna()
    ].copy()
    if d.empty:
        return pd.DataFrame(columns=COLLAPSE_ROW_COLUMNS)
    med = d.groupby("driver")["lap_time_s"].transform("median")
    d = d[d["lap_time_s"] < med + OUTLIER_MARGIN_S].copy()
    if corr_fn is not None:
        # `evo` is subtracted, so the frame's correction enters negated; the
        # detector's own fuel term is switched off because the frame carries it.
        d["_corr"] = -corr_fn(d["lap_number"].to_numpy(dtype=float))

    key = "stint_uid" if "stint_uid" in d else None
    groups = d.groupby(key if key else ["driver", "stint"], sort=True)
    rows = []
    for _, g in groups:
        if len(g) < int(min_laps):
            continue
        res = detect_stint_collapse(g, fuel_s_per_lap=(0.0 if corr_fn is not None else float(fuel_s_per_lap)),
                                   evo=(g["_corr"] if corr_fn is not None else None),
                                   min_laps=min_laps)
        drv = str(g["driver"].iloc[0])
        stint = float(g["stint"].iloc[0])
        end_lap = float(ends.get((drv, stint), g["lap_number"].max()))
        has_next = (drv, stint + 1.0) in ends.index
        censored = bool(not has_next and end_lap >= n_race_laps - 1)
        in_lap = float(end_lap) if has_next else float("nan")
        sc = (bool(is_sc_status(status.get((drv, in_lap), status.get((drv, in_lap - 1.0), "1"))))
              if has_next else False)
        kind, collapse = res["kind"], res["collapse"]
        if not has_next:
            # No stop ended this stint: the chequered flag did, or the car
            # stopped.  Either way nobody decided the tyre was finished, so
            # there is no collapse to call and no on-trend stop to read — but
            # the loss it had already taken is still a lower bound on the
            # budget.  Only the flag case is `censored_by_flag`; a retirement
            # is a different censoring and is left unflagged.
            kind, collapse = "undetermined", False
        rows.append({
            "driver": drv, "team": str(g["team"].iloc[0]) if "team" in g else "",
            "compound": str(g["compound"].iloc[0]), "stint": int(stint), "n": int(res["n"]),
            "kind": kind, "knee_age": res["knee_age"], "collapse": bool(collapse),
            "censored_by_flag": censored, "slope_pre": res["slope_pre"], "slope_post": res["slope_post"],
            "delta_last3_s": res["delta_last3_s"], "cum_loss_at_knee_s": res["cum_loss_at_knee_s"],
            "cum_loss_at_end_s": res["cum_loss_at_end_s"], "rate_s_per_lap": res["rate_s_per_lap"],
            "in_lap": in_lap, "sc_stop": sc,
        })
    if not rows:
        return pd.DataFrame(columns=COLLAPSE_ROW_COLUMNS)
    return pd.DataFrame(rows)[COLLAPSE_ROW_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------
# The censored estimator
# --------------------------------------------------------------------------


def _log_sf(z: np.ndarray) -> np.ndarray:
    """log(1 - Phi(z)) — the probability a censored stint's budget exceeds its bound."""
    erfc = np.vectorize(math.erfc, otypes=[float])
    return np.log(np.clip(0.5 * erfc(np.asarray(z, dtype=float) / math.sqrt(2.0)), 1e-300, 1.0))


def _split_rows(rows) -> tuple:
    """Collapse observations and right-censored lower bounds from cliff rows."""
    if rows is None:
        return np.array([]), np.array([])
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if df.empty:
        return np.array([]), np.array([])
    hit = df["collapse"].fillna(False).astype(bool) if "collapse" in df else (df["kind"] == "collapse")
    obs = pd.to_numeric(df.loc[hit, "cum_loss_at_knee_s"], errors="coerce").dropna()
    bnd = pd.to_numeric(df.loc[~hit, "cum_loss_at_end_s"], errors="coerce").dropna()
    return obs[obs > 0].to_numpy(dtype=float), bnd[bnd > 0].to_numpy(dtype=float)


def grip_budget_estimate(rows, *, prior: float = GRIP_BUDGET_S, prior_ln_sd: float = 0.15,
                         band: tuple = BUDGET_BAND_S, grid: np.ndarray | None = None,
                         obs_ln_sd: float = BUDGET_OBS_LN_SD) -> dict:
    """The grip budget from a right-censored sample of stint losses.

    `rows` are `race_collapses` rows for **one compound**, pooled over as many
    races as the caller wants (use `grip_budgets_by_compound` to do the split).
    Collapse stints contribute `cum_loss_at_knee_s` as observations of the
    budget; every other stint contributes `cum_loss_at_end_s` as a lower bound,
    because the tyre reached that loss and was still on its trend.

    A lower bound can only *raise* the budget.  With no (consistent) collapse
    observation the likelihood is monotone and has no mode of its own, so the
    estimate is the prior or the largest bound, whichever is larger: a stint
    that gave up 4.2 s and was still on its trend rules out a 3.8 s budget,
    while a largest bound of 3.3 s says nothing against a 3.8 s prior.  (V2's
    rule took any bound above 3.0 s as the estimate even when it sat *below*
    the prior, which is how a held-out Barcelona was handed a 3.27 s MEDIUM -
    another circuit's lower bound used as a point estimate - and with it a
    21-lap MEDIUM life that put the plan's 24-lap stints past the cliff.)

    **Consistency.**  A collapse observation is a measurement of the budget
    only if it is a tyre cliff.  Across the seven 2026 races the detector finds
    four collapses at 0.9-2.5 s of cumulative loss against more than a hundred
    stints that ran past 3-4 s on their trend; a loss that small cannot be the
    same tyre's cliff, so an observation below the **median** lower bound of
    the same compound is flagged inconsistent (damage, traffic, a driver
    problem - something other than the tyre) and excluded from the likelihood.
    It is counted in `n_obs_excluded` so the report can say so.

    `se_s` is the standard deviation of the normalised likelihood-times-prior
    over the grid rather than a curvature approximation: the surface is
    routinely one-sided (all bound, no observation) and a curvature standard
    error there is meaningless.
    """
    obs, bnd = _split_rows(rows)
    lo, hi = float(band[0]), float(band[1])
    largest = float(bnd.max()) if len(bnd) else float("nan")
    if grid is None:
        grid = np.arange(lo, hi + BUDGET_GRID_STEP_S / 2, BUDGET_GRID_STEP_S)
    grid = np.asarray(grid, dtype=float)

    n_excluded = 0
    if len(obs) and len(bnd):
        consistent = obs >= float(np.median(bnd))
        n_excluded = int((~consistent).sum())
        obs = obs[consistent]

    if not len(obs):
        b = float(np.clip(max(float(prior), largest) if np.isfinite(largest) else float(prior), lo, hi))
        if np.isfinite(largest) and largest > float(prior):
            src = (f"largest lower bound {largest:.2f} s over {len(bnd)} stints on their trend exceeds the "
                   f"prior {float(prior):.2f} s; no consistent collapse observation")
        else:
            src = (f"prior {float(prior):.2f} s: no consistent collapse observation and the largest lower bound "
                   f"({largest:.2f} s over {len(bnd)} stints) does not exceed it" if np.isfinite(largest) else
                   f"prior {float(prior):.2f} s: no stint measured")
        if n_excluded:
            src += f"; {n_excluded} collapse observation(s) below the median bound excluded as not a tyre cliff"
        return {"budget_s": b, "n_obs": 0, "n_obs_excluded": n_excluded, "n_censored": int(len(bnd)),
                "se_s": float("nan"), "largest_bound_s": largest, "source": src}

    s = float(obs_ln_sd)
    if len(obs) >= 3:
        # let the observations speak about their own spread where there are
        # enough of them to; never tighter than the prior assumption
        s = float(max(s, np.std(np.log(obs), ddof=1)))
    lg = np.log(grid)
    ll = np.zeros_like(grid)
    for o in obs:
        z = (math.log(o) - lg) / s
        ll += -0.5 * z ** 2 - math.log(s)
    for dd in bnd:
        ll += _log_sf((math.log(dd) - lg) / s)
    ll += -0.5 * ((lg - math.log(float(prior))) / float(prior_ln_sd)) ** 2

    w = np.exp(ll - ll.max())
    w = w / w.sum()
    b = float(grid[int(np.argmax(ll))])
    se = float(np.sqrt(max(np.sum(w * (grid - np.sum(w * grid)) ** 2), 0.0)))
    floored = False
    if np.isfinite(largest) and largest > b:
        # a loss the tyre survived is a hard constraint on the budget, and the
        # censored term alone is too soft to enforce it
        b, floored = float(np.clip(largest, lo, hi)), True
    src = (f"censored log-normal MLE on {len(obs)} collapse observations "
           f"({', '.join(f'{o:.2f}' for o in np.sort(obs)[::-1][:6])} s) and {len(bnd)} lower bounds "
           f"(largest {largest:.2f} s), obs ln-sd {s:.2f}, prior {float(prior):.2f} +/-{float(prior_ln_sd):.2f} ln"
           + ("; floored at the largest lower bound" if floored else ""))
    if n_excluded:
        src += f"; {n_excluded} collapse observation(s) below the median bound excluded as not a tyre cliff"
    return {"budget_s": float(np.clip(b, lo, hi)), "n_obs": int(len(obs)), "n_obs_excluded": n_excluded,
            "n_censored": int(len(bnd)), "se_s": se, "largest_bound_s": largest, "source": src}


def grip_budgets_by_compound(rows, *, compounds=None, **kw) -> dict:
    """`grip_budget_estimate` per compound — what the calibration ships."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows or []))
    comps = list(compounds) if compounds is not None else (
        [c for c in VALID_COMPOUNDS if c in set(df["compound"])] if not df.empty else [])
    out = {}
    for c in comps:
        sub = df[df["compound"] == c] if not df.empty else df
        out[c] = grip_budget_estimate(sub, **kw)
    return out


# --------------------------------------------------------------------------
# The benchmark's view: does the budget predict where the cliff landed?
# --------------------------------------------------------------------------


def budget_ratio_metrics(pred_budget_by_compound: dict, race_rate_by_compound: dict,
                         collapse_rows) -> dict:
    """Predicted collapse lap (budget / rate) against the observed knees.

    This is the only end-to-end check the grip-budget invariant has: the model
    ships `budget_c` and a rate, their quotient is where it thinks the compound
    falls off, and the detector says where it actually did.  `over` counts the
    stints the model thought would last longer than they did (the failure that
    lets the optimiser run a SOFT for 33 laps), `under` the opposite.

    `race_rate_by_compound` must be a rate in the *same push regime the budget
    is defined in* — the full-push rate the model ships, or a race-measured
    slope divided by the regime factor.  Handing it a raw race slope answers a
    different question ("how long would this tyre last at the push the field
    ran?") and the predicted laps come out at twice the race distance.  The
    nested `own_rate` block sidesteps the choice: it prices each collapsed stint
    against *its own* measured pre-knee rate, which has no regime assumption in
    it at all, and is the honest test of the invariant.
    """
    df = collapse_rows if isinstance(collapse_rows, pd.DataFrame) else pd.DataFrame(list(collapse_rows or []))
    out, all_err = {}, []
    comps = sorted(set(pred_budget_by_compound) | (set(df["compound"]) if not df.empty else set()))
    for c in comps:
        b = pred_budget_by_compound.get(c)
        r = race_rate_by_compound.get(c)
        b = float(b.get("budget_s")) if isinstance(b, dict) else (float(b) if b is not None else float("nan"))
        r = float(r.get("slope")) if isinstance(r, dict) else (float(r) if r is not None else float("nan"))
        pred = float(b / r) if (np.isfinite(b) and np.isfinite(r) and r > 0) else float("nan")
        sub = df[(df["compound"] == c) & df["collapse"].fillna(False).astype(bool)] if not df.empty else df
        knees = pd.to_numeric(sub["knee_age"], errors="coerce").dropna().to_numpy(dtype=float) if len(sub) else np.array([])
        err = (pred - knees) if (np.isfinite(pred) and len(knees)) else np.array([])
        all_err.extend(err.tolist())
        own_r = (pd.to_numeric(sub["rate_s_per_lap"], errors="coerce").to_numpy(dtype=float)
                 if len(sub) else np.array([]))
        own_pred = np.where(own_r > 0, b / np.where(own_r > 0, own_r, 1.0), np.nan) if len(own_r) else np.array([])
        own_err = (own_pred - knees) if len(own_pred) else np.array([])
        own_err = own_err[np.isfinite(own_err)]
        out[c] = {"budget_s": b, "rate_s_per_lap": r, "predicted_lap": pred,
                  "n": int(len(knees)), "observed_knees": [float(x) for x in knees],
                  "observed_mean_lap": float(knees.mean()) if len(knees) else float("nan"),
                  "mean_error_laps": float(err.mean()) if len(err) else float("nan"),
                  "mean_abs_error_laps": float(np.abs(err).mean()) if len(err) else float("nan"),
                  "over": int((err > 0).sum()), "under": int((err < 0).sum()),
                  "own_rate": {
                      "mean_predicted_lap": float(np.nanmean(own_pred)) if len(own_err) else float("nan"),
                      "mean_error_laps": float(own_err.mean()) if len(own_err) else float("nan"),
                      "mean_abs_error_laps": float(np.abs(own_err).mean()) if len(own_err) else float("nan"),
                      "over": int((own_err > 0).sum()), "under": int((own_err < 0).sum())}}
    e = np.asarray(all_err, dtype=float)
    out["_pooled"] = {"n": int(len(e)),
                      "mean_error_laps": float(e.mean()) if len(e) else float("nan"),
                      "mean_abs_error_laps": float(np.abs(e).mean()) if len(e) else float("nan"),
                      "over": int((e > 0).sum()), "under": int((e < 0).sum())}
    return out
