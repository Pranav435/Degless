"""Haas's two cars, estimated from this weekend and nothing else.

`#31 OCO` and `#87 BEA` run the same car, and almost everything the strategy
model knows about that car is measured on the field: the degradation rates come
from a fit pooled over twenty drivers, the warm-up cost is a calibrated
constant, the dirty-air cost is a per-circuit average.  A race engineer sitting
on the Haas pit wall needs the two cars separated - which of them is harder on
its tyres, which of them loses more on the out-lap, which of them suffers more
behind a car it cannot pass - because the two decisions they take are
different decisions.

The danger in doing that is inventing it.  There is a well-worn genre of
"driver profile" that is really a sentence someone wrote down once ("Ocon is
gentle on his tyres"), and a model carrying such a sentence will happily move a
pit stop three laps on the strength of it.  So every number in a `CarState`
here is one of three things, and it says which:

    driver   measured on this weekend's data for this car, and there is enough
             of it that the car's own measurement outweighs its team's
    team     the team's value (both cars pooled), because this car's own
             evidence is too thin to outweigh it
    field    the field's value, because the team's evidence is thin too

The shrinkage is the standard one, `w = n / (n + k)` with `k` a pseudo-count in
the same units as `n`, applied driver -> team and then team -> field; `k` is
stated for every quantity below with the reason for its size.  Two properties
follow and are tested: a car with no evidence at all gets exactly the team
value at `shrink_weight = 0`, and a car with a lot of evidence keeps most of
its own.  Nothing is ever filled in from a previous weekend's *behaviour*: the
only historical input is the race-history rate factor
(`Calibration.driver_factors` with `driver_factor_ln_sd`), and it enters as a
prior through `percar.shrink_factor` - precision-weighted, so a factor measured
across one noisy race barely moves anything - and never as evidence.

What is measured, and how:

`pace_offset_s`         the level of the car's clean long-run laps against the
                        field's own compound/age trend (`lap_time_corr`, so
                        fuel and track evolution are already out).  Negative is
                        quicker than the field.  k = 8 laps: a practice long
                        run is 8-12 clean laps, so one full long run is worth
                        about half the team's value.
`pace_vs_teammate_s`    the same level minus the team-mate's, the number the
                        pit wall actually asks for.  Zero when either car has
                        no long run.
`deg_rate_by_compound`  the field rate plus the fit's own per-driver, per-
                        compound deviation `dev[d, c]`, pooled over the team by
                        `percar.team_pooled_dev` at k = 20 clean laps (its
                        `POOL_K_LAPS`, shared so that this module and the
                        shipped per-car model agree by construction).  The
                        team -> field step for this one quantity is *already*
                        done, by the fit: it samples `dev = sigma_dev *
                        dev_raw` with `sigma_dev ~ HalfNormal(0.02)`, a
                        hierarchical prior centred on the field, so shrinking a
                        second time here would double-count it.  `note` says so.
`age_sensitivity`       the same deviation as one number, s/lap per lap of tyre
                        age above the field, with the multiplicative rate scale
                        the model applies in `detail`.
`warmup_s`              the mean excess of a stint's first laps
                        (`lap_in_stint` 1 and 2) over that stint's own fitted
                        trend.  In a practice table lap 1 is the out-lap and is
                        filtered out, so this is normally the first flying lap.
                        k = 3 stints: the excess is one number per stint, and
                        three stints is a weekend's worth of long runs.
`consistency_s`         the residual SD about the stint trend, over the laps
                        the trend was fitted on.  k = 12 laps.
`traffic_sensitivity`   the car's excess lap time within `TRAFFIC_GAP_S` of the
                        car ahead, over the same stint trend, divided by the
                        field's same excess - so 1.0 is a car that loses what
                        everyone loses.  k = 6 in-traffic laps, shrunk toward
                        1.  Practice gives very few such laps (Hungary: one for
                        OCO, none for BEA), which is exactly why it shrinks.
`push_response`         *not identifiable*.  A practice long run is a full-push
                        experiment by construction - that is what the run is
                        for - so there is no within-weekend variation in push
                        from which a per-car management response could be
                        estimated.  The team value (the calibrated
                        `manage_cost_s`) is used and `source` says `team` at
                        `shrink_weight = 0`.
`sector_deg`            per-sector degradation, s/lap of age, from the sector
                        times in the FastF1 cache (the lap tables carry none).
                        `None` with `source = "unavailable"` where the cache
                        has no sector times - never a fabricated number.

`race_laps` is optional and is the live/post-race path: when it is passed, the
within-stint quantities (warm-up, consistency, traffic) are measured on the
race stints as well as the practice ones, and the pace level pools the practice
estimate with a race estimate taken against the field's own lap-by-lap median
(which removes fuel, evolution and neutralisations).  Degradation stays on the
posterior - folding a measured race rate back into the tyre model is the live
engine's job, not this module's - and the measured race rates are reported in
`detail` as a diagnostic.

Three things leave this module:

    `car_terms(state)`  the per-car keywords the strategy search and the live
                        engine take: `warmup_s`, `traffic_mult`, `race_factor`
                        and `dev_override`.
    `explain_difference` why the two cars' recommendations differ, in numbers,
                        or the statement that the difference is within noise.
    `haas_block`        the JSON-clean summary `meta["haas"]` carries.

Run `python -m src.haascar --events hungary-2026` for the evidence report: both
cars' states with their counts, weights and sources, and the per-car plans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import percar
from src.config import (
    MANAGE_COST_S,
    OUT_LAP_PENALTY_S,
    TRAFFIC_GAP_S,
    VALID_COMPOUNDS,
    Event,
    get_event,
)

__all__ = [
    "Estimate",
    "CarState",
    "HaasCarModel",
    "car_terms",
    "explain_difference",
    "haas_block",
    "hier_rate_scale_table",
    "stint_trend",
    "HAAS_DRIVERS",
    "HAAS_TEAM",
]

HAAS_TEAM = "Haas F1 Team"
HAAS_DRIVERS = ("OCO", "BEA")

# -- pseudo-counts, in the units of each quantity's own evidence -------------
# Every one of these is a *prior strength*, not a fitted constant: it says how
# much of its own evidence a car needs before its own number outweighs its
# team's.  None of them is tuned on a benchmark.
PACE_K_LAPS = 8.0          # a practice long run is 8-12 clean laps -> one run ~ half weight
POOL_K_LAPS = percar.POOL_K_LAPS      # 20; the shipped dev pooling, shared so the two agree
WARMUP_K_STINTS = 3.0      # the excess is one number per stint; 3 stints ~ a weekend of long runs
CONSISTENCY_K_LAPS = 12.0  # an SD needs a long run's worth of laps before it is the car's own
TRAFFIC_K_LAPS = 6.0       # in-traffic laps; practice rarely gives more than a handful
SECTOR_K_LAPS = 12.0       # per-sector slopes are noisier than the lap-level one
TEAM_K_MULT = 2.0          # the team -> field pseudo-count is `TEAM_K_MULT * k`: a team is two
                           # cars, so it needs about twice one car's evidence to stand alone

# -- estimator guards (engineering constants, each with its reason) ----------
TREND_FROM_LAP = 3         # the stint trend is fitted from the third lap on, so the warm-up
                           # laps it is measured against do not set it
MIN_TREND_LAPS = 3         # a two-parameter line needs three points to leave a residual
WARMUP_LAPS = (1, 2)       # lap 1 is the out-lap (usually filtered), lap 2 the first flying lap
MIN_FIELD_TRAFFIC_S = 0.05      # below this the field's own in-traffic excess is not measurable,
                                # so a ratio to it means nothing and the estimate stays at 1.0
MIN_FIELD_TRAFFIC_LAPS = 20
MIN_WARMUP_STINTS = 2           # the warm-up excess is one number per stint, read off a backward
                                # extrapolation of that stint's trend, and a single stint gives no
                                # way to tell a warm-up cost from one bad lap.  A pool below this
                                # contributes nothing - at any level.  Measured: the per-stint
                                # excesses run from -2.3 s to +2.0 s across the seven weekends,
                                # so one stint is not an estimate.
WARMUP_DELTA_BAND_S = 0.5       # and the *term* the search is charged only ever moves the
                                # calibrated field warm-up by this much: a per-car warm-up
                                # difference larger than half a second per stint is not something
                                # one or two practice stints can establish
MIN_OWN_TRAFFIC_LAPS = 3        # one lap in dirty air cannot be told apart from one lap with a
                                # mistake, and with n < 3 the spread is not estimable either, so a
                                # pool below this contributes nothing - at any level.  Without it,
                                # OCO's single in-traffic practice lap at Hungary would set a
                                # x1.56 dirty-air multiplier for itself and x1.21 for BEA.
TRAFFIC_MULT_BAND = (0.5, 2.0)  # a multiplier outside this band is thin-data noise, not a car

# -- materiality, for `explain_difference` -----------------------------------
# "Material" = large enough that it can move a pit stop by a lap.  Each is
# stated in the quantity's own units with what it costs over a stint.
MATERIAL_RATE_S = 0.005        # s/lap of age; 1.6 s over a 25-lap stint
MATERIAL_PACE_S = 0.05         # s/lap; 1.2 s over a 25-lap stint
MATERIAL_WARMUP_S = 0.15       # s per stint; half of the live decision hysteresis margin
MATERIAL_CONSISTENCY_S = 0.10  # s
MATERIAL_TRAFFIC_MULT = 0.15   # 15% of the dirty-air cost
MATERIAL_FACTOR = 0.05         # 5% of the rate


# --------------------------------------------------------------------------
# One estimated quantity
# --------------------------------------------------------------------------


@dataclass
class Estimate:
    """One number (or one dict of numbers) with where it came from.

    `value` is what the model should use.  `own` is the car's own unshrunk
    measurement (`None` when it has none), `team_value` the team level the
    estimate is shrunk toward and `field_value` the field level behind that.
    `shrink_weight` is the weight the car's own measurement carried, so
    `value == team_value` exactly when `shrink_weight == 0`.
    """

    value: object = None
    n_evidence: int = 0
    shrink_weight: float = 0.0
    source: str = "field"          # driver | team | field | unavailable
    own: object = None
    team_value: object = None
    field_value: object = None
    units: str = ""
    note: str = ""
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return _json_clean({
            "value": self.value, "n_evidence": int(self.n_evidence),
            "shrink_weight": round(float(self.shrink_weight), 3), "source": self.source,
            "own": self.own, "team_value": self.team_value, "field_value": self.field_value,
            "units": self.units, "note": self.note, "detail": self.detail,
        })


def _w(n: float, k: float) -> float:
    n = float(n or 0.0)
    k = float(k)
    return float(n / (n + k)) if (n + k) > 0 else 0.0


def _mix(a, b, w: float):
    """`w * a + (1 - w) * b` for floats, per-key for dicts, None-safe."""
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, dict) or isinstance(b, dict):
        da = a if isinstance(a, dict) else {}
        db = b if isinstance(b, dict) else {}
        return {k: _mix(da.get(k), db.get(k), w) for k in sorted(set(da) | set(db))}
    return float(w) * float(a) + (1.0 - float(w)) * float(b)


def _shrink(own, team, field_value, *, n_own: float, k: float,
            n_team: float | None = None, k_team: float | None = None,
            units: str = "", note: str = "", detail: dict | None = None) -> Estimate:
    """`w * own + (1 - w) * team_level`, driver -> team -> field.

    The team level is itself `w_t * team + (1 - w_t) * field` when the team's
    own count and pseudo-count are given, so a team with one thin long run does
    not pin both its cars; `k_team=None` takes the team value as it stands
    (used where the team -> field step is done elsewhere, e.g. by the fit's own
    hierarchical prior on `dev`).  `own=None` is no evidence: the result is the
    team level at `shrink_weight = 0`.
    """
    n_own = float(n_own or 0.0)
    team_level = field_value if team is None else team
    if team is not None and k_team is not None:
        team_level = _mix(team, field_value, _w(n_team if n_team is not None else 0.0, k_team))
    w = _w(n_own, k) if own is not None else 0.0
    value = _mix(own, team_level, w) if own is not None else team_level
    if own is not None and w >= 0.5:
        source = "driver"
    elif team is not None:
        source = "team"
    else:
        source = "field"
    return Estimate(value=value, n_evidence=int(round(n_own)), shrink_weight=w, source=source,
                    own=own, team_value=team_level, field_value=field_value, units=units,
                    note=note, detail=dict(detail or {}))


def _unavailable(reason: str, *, units: str = "") -> Estimate:
    return Estimate(value=None, n_evidence=0, shrink_weight=0.0, source="unavailable",
                    units=units, note=reason)


# --------------------------------------------------------------------------
# JSON hygiene: numpy and pandas do not survive `json.dumps`
# --------------------------------------------------------------------------


def _json_clean(obj):
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [_json_clean(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [_json_clean(r) for r in obj.to_dict("records")]
    if isinstance(obj, pd.Series):
        return _json_clean(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_clean(x) for x in obj]
    return str(obj)


# --------------------------------------------------------------------------
# The lap-table primitives: a stint's trend, and what sits above it
# --------------------------------------------------------------------------


def _ycol(df: pd.DataFrame) -> str:
    """`lap_time_corr` where the table carries it, else the raw lap time.

    The corrected column has the fuel term and the track-evolution term taken
    out (`src.evolution`), which is what a *level* comparison across sessions
    needs.  A within-stint comparison does not: the per-stint linear trend
    fitted below absorbs the fuel burn along with the degradation, so the raw
    column is a valid fallback and is what the uncleaned lap tables carry.
    """
    for c in ("lap_time_corr", "lap_time_s"):
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return "lap_time_s"


def _flag(df: pd.DataFrame, name: str, default: bool = True) -> pd.Series:
    if name in df.columns:
        return df[name].fillna(False).astype(bool)
    return pd.Series(default, index=df.index)


def _valid(df: pd.DataFrame, *, traffic: bool | None = False, long_runs: bool = True) -> pd.DataFrame:
    """The laps an estimator may use.

    Rules 1-3 of `src.laps.RULES` (accurate, not an in/out lap, green) plus a
    known compound and the slow-lap filter; `long_runs` adds rule 4 (the stint
    is a long run).  `traffic=False` keeps the laps with more than
    `TRAFFIC_GAP_S` to the car ahead (the clean set), `traffic=True` keeps only
    the laps inside it, `None` keeps both.  Any flag the frame does not carry
    is treated as passed, so a hand-built frame (a test, a synthetic stint)
    needs only the columns the estimator reads.
    """
    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame()
    keep = (_flag(df, "ok_accurate") & _flag(df, "ok_not_pit") & _flag(df, "ok_green")
            & _flag(df, "ok_compound") & _flag(df, "ok_not_slow"))
    if long_runs:
        keep &= _flag(df, "ok_stint_len")
    if traffic is not None:
        gap = pd.to_numeric(df.get("gap_ahead_s", pd.Series(np.inf, index=df.index)),
                            errors="coerce").fillna(np.inf)
        keep &= (gap > TRAFFIC_GAP_S) if not traffic else (gap <= TRAFFIC_GAP_S)
    out = df[keep].copy()
    return out


def stint_trend(g: pd.DataFrame, *, ycol: str | None = None, trend_from: int = TREND_FROM_LAP,
                min_trend_laps: int = MIN_TREND_LAPS, warmup_laps=WARMUP_LAPS) -> dict:
    """One stint's fitted trend, its warm-up excess and its residual spread.

    `y = a + b * lap_in_stint` is fitted by least squares on the laps from
    `trend_from` on - the warm-up laps are deliberately excluded, because a
    trend fitted through them would absorb the very excess this measures - and
    then

        warmup[l] = y(l) - (a + b l)      for l in `warmup_laps`
        resid_sd  = sqrt(SSR / (n - 2))   over the laps the trend was fitted on

    Returns `{}` when the stint has fewer than `min_trend_laps` trend laps.
    """
    if g is None or len(g) == 0:
        return {}
    y_name = ycol or _ycol(g)
    d = g.dropna(subset=[y_name]) if y_name in g.columns else pd.DataFrame()
    if d.empty or "lap_in_stint" not in d.columns:
        return {}
    x_all = pd.to_numeric(d["lap_in_stint"], errors="coerce").to_numpy(float)
    y_all = pd.to_numeric(d[y_name], errors="coerce").to_numpy(float)
    ok = np.isfinite(x_all) & np.isfinite(y_all)
    x_all, y_all = x_all[ok], y_all[ok]
    m = x_all >= float(trend_from)
    if int(m.sum()) < int(min_trend_laps):
        return {}
    x, y = x_all[m], y_all[m]
    A = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = max(len(x) - 2, 0)
    out = {"intercept": float(beta[0]), "slope": float(beta[1]), "n_trend": int(len(x)),
           "resid_ss": float(resid @ resid), "dof": int(dof),
           "resid_sd": (float(np.sqrt(resid @ resid / dof)) if dof > 0 else float("nan")),
           "ycol": y_name, "warmup": {}}
    for lap in warmup_laps:
        sel = np.isclose(x_all, float(lap))
        if sel.any():
            out["warmup"][int(lap)] = float(np.mean(y_all[sel] - (beta[0] + beta[1] * float(lap))))
    out["warmup_s"] = (float(np.mean(list(out["warmup"].values()))) if out["warmup"] else None)
    return out


def _trend_raw(df: pd.DataFrame, *, ycol: str | None = None) -> dict:
    """Per driver, the raw accumulators: warm-up excesses, residual sums, counts."""
    out: dict = {}
    if df is None or len(df) == 0 or "stint_uid" not in df.columns:
        return out
    y_name = ycol or _ycol(df)
    for (drv, uid), g in df.groupby(["driver", "stint_uid"], sort=False):
        t = stint_trend(g, ycol=y_name)
        if not t:
            continue
        o = out.setdefault(str(drv), {"warmup": [], "resid_ss": 0.0, "dof": 0, "n_laps": 0,
                                      "n_stints": 0, "slopes": []})
        o["n_stints"] += 1
        o["n_laps"] += int(t["n_trend"])
        o["resid_ss"] += float(t["resid_ss"])
        o["dof"] += int(t["dof"])
        o["slopes"].append(float(t["slope"]))
        if t.get("warmup_s") is not None:
            o["warmup"].append(float(t["warmup_s"]))
    return out


def _finalise_trend(raw: dict) -> dict:
    for o in raw.values():
        o["warmup_s"] = (float(np.mean(o["warmup"])) if o["warmup"] else None)
        o["n_warmup_stints"] = int(len(o["warmup"]))
        o["consistency_s"] = (float(np.sqrt(o["resid_ss"] / o["dof"])) if o["dof"] > 0 else None)
    return raw


def _merge_raw(parts: list, list_keys=("warmup", "slopes", "excess"),
               sum_keys=("resid_ss", "dof", "n_laps", "n_stints", "n")) -> dict:
    """Accumulators from several tables, added per driver.

    Each lap table is measured on its own lap-time column - the clean practice
    table carries `lap_time_corr`, the race table does not - so they have to be
    walked separately and pooled here.  Concatenating them first would make
    `_ycol` pick one column and silently drop every row the other table holds.
    """
    out: dict = {}
    for part in parts:
        for drv, o in (part or {}).items():
            tgt = out.setdefault(drv, {})
            for k, v in o.items():
                if k in list_keys:
                    tgt.setdefault(k, []).extend(list(v))
                elif k in sum_keys:
                    tgt[k] = tgt.get(k, 0) + v
    return out


def _trend_stats(df: pd.DataFrame, *, ycol: str | None = None) -> dict:
    """Per driver: the warm-up excesses, the pooled residual spread, the counts.

    The warm-up excess is pooled over *stints* (one number each) and the
    residual spread over *laps*, which is why the two carry different
    pseudo-counts.
    """
    return _finalise_trend(_trend_raw(df, ycol=ycol))


def _trend_stats_frames(frames: list) -> dict:
    """`_trend_stats` over several lap tables, each on its own column."""
    return _finalise_trend(_merge_raw([_trend_raw(_valid(f, traffic=False)) for f in frames]))


def _pool_trend(stats: dict, drivers) -> dict:
    """The same statistics pooled over a set of drivers (a team, or the field)."""
    warm, ss, dof, laps, stints = [], 0.0, 0, 0, 0
    for d in drivers:
        o = stats.get(d)
        if not o:
            continue
        warm += list(o["warmup"])
        ss += float(o["resid_ss"])
        dof += int(o["dof"])
        laps += int(o["n_laps"])
        stints += int(o["n_stints"])
    return {"warmup_s": (float(np.mean(warm)) if warm else None), "n_warmup_stints": len(warm),
            "consistency_s": (float(np.sqrt(ss / dof)) if dof > 0 else None),
            "n_laps": laps, "n_stints": stints}


def _traffic_raw(df: pd.DataFrame, *, ycol: str | None = None) -> dict:
    """Per driver: the mean excess, over the stint's own trend, of the laps run
    within `TRAFFIC_GAP_S` of the car ahead.

    The trend is fitted on the *clean* laps of the same stint, so the excess is
    measured against what that car was doing on that tyre at that age with
    nobody in front of it, and the warm-up laps are excluded from both sides.
    """
    out: dict = {}
    if df is None or len(df) == 0 or "gap_ahead_s" not in df.columns:
        return out
    y_name = ycol or _ycol(df)
    base = _valid(df, traffic=None)
    if base.empty:
        return out
    gap = pd.to_numeric(base["gap_ahead_s"], errors="coerce").fillna(np.inf)
    lis = pd.to_numeric(base["lap_in_stint"], errors="coerce")
    for (drv, uid), g in base.groupby(["driver", "stint_uid"], sort=False):
        gg, ll = gap.loc[g.index], lis.loc[g.index]
        clean = g[(gg > TRAFFIC_GAP_S) & (ll >= TREND_FROM_LAP)]
        dirty = g[(gg <= TRAFFIC_GAP_S) & (ll >= TREND_FROM_LAP)]
        if len(dirty) == 0:
            continue
        t = stint_trend(clean, ycol=y_name, trend_from=-np.inf)
        if not t:
            continue
        x = pd.to_numeric(dirty["lap_in_stint"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(dirty[y_name], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            continue
        ex = y[ok] - (t["intercept"] + t["slope"] * x[ok])
        o = out.setdefault(str(drv), {"excess": [], "n": 0, "n_stints": 0})
        o["excess"] += [float(v) for v in ex]
        o["n"] += int(ok.sum())
        o["n_stints"] += 1
    return out


def _finalise_traffic(raw: dict) -> dict:
    for o in raw.values():
        o["excess_s"] = float(np.mean(o["excess"])) if o["excess"] else None
    return raw


def _traffic_excess(df: pd.DataFrame, *, ycol: str | None = None) -> dict:
    return _finalise_traffic(_traffic_raw(df, ycol=ycol))


def _traffic_excess_frames(frames: list) -> dict:
    """`_traffic_excess` over several lap tables, each on its own column."""
    return _finalise_traffic(_merge_raw([_traffic_raw(f) for f in frames]))


def _pool_traffic(stats: dict, drivers) -> dict:
    ex, n = [], 0
    for d in drivers:
        o = stats.get(d)
        if not o:
            continue
        ex += list(o["excess"])
        n += int(o["n"])
    return {"excess_s": (float(np.mean(ex)) if ex else None), "n": n}


def _levels_practice(clean: pd.DataFrame) -> dict:
    """Per driver: the level of its clean long-run laps against the field's own
    compound/age trend, in seconds per lap, centred on the field.

    Two passes, on purpose.  The field trend (a session effect, a compound
    effect and a compound-specific age slope) is fitted without driver terms,
    and each driver's level is then the *median* residual of its own laps.  A
    single design with driver dummies is the textbook answer but is
    rank-deficient the moment a driver runs one compound in one session, which
    at Haas is the normal case; the two-pass version is always solvable, and
    the median makes one bad lap in a nine-lap sample harmless.
    """
    out = {"levels": {}, "n": {}, "n_by_compound": {}, "ycol": None, "n_laps": 0}
    d = _valid(clean, traffic=False)
    if d is None or len(d) == 0:
        return out
    y_name = _ycol(d)
    d = d.dropna(subset=[y_name, "compound"])
    if d.empty:
        return out
    age = pd.to_numeric(d.get("tyre_age", d.get("lap_in_stint")), errors="coerce").fillna(0.0).to_numpy(float)
    y = pd.to_numeric(d[y_name], errors="coerce").to_numpy(float)
    comp = pd.get_dummies(d["compound"].astype(str)).astype(float)
    sess = pd.get_dummies(d["session"].astype(str)).astype(float) if "session" in d else None
    cols = [comp.to_numpy(), comp.to_numpy() * age[:, None]]
    if sess is not None and sess.shape[1] > 1:
        cols.append(sess.to_numpy()[:, 1:])
    A = np.column_stack(cols)
    ok = np.isfinite(y) & np.isfinite(A).all(1)
    if int(ok.sum()) < 4:
        return out
    beta, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
    resid = np.full(len(y), np.nan)
    resid[ok] = y[ok] - A[ok] @ beta
    d = d.assign(_resid=resid).dropna(subset=["_resid"])
    lv = d.groupby("driver")["_resid"].median()
    n = d.groupby("driver")["_resid"].size()
    centre = float(np.average(lv.to_numpy(float), weights=n.reindex(lv.index).to_numpy(float)))
    out["levels"] = {str(k): float(v - centre) for k, v in lv.items()}
    out["n"] = {str(k): int(v) for k, v in n.items()}
    out["n_by_compound"] = {str(k): {str(c): int(v) for c, v in g["compound"].value_counts().items()}
                            for k, g in d.groupby("driver")}
    out["ycol"] = y_name
    out["n_laps"] = int(len(d))
    return out


def _levels_race(race: pd.DataFrame) -> dict:
    """Per driver: the level of its green race laps against the field's own
    median on the same lap, centred on the field.

    The per-lap field median is the cleanest control a race gives: every car on
    that lap carries the same fuel load, the same track and the same
    neutralisation, so the residual is pace.  It is *not* corrected for tyre
    age or compound - in a race those are part of the strategy, not a nuisance
    - so the number is the level as run, which is what the pit wall reads.
    """
    out = {"levels": {}, "n": {}, "ycol": None, "n_laps": 0}
    d = _valid(race, traffic=None, long_runs=False)
    if d is None or len(d) == 0:
        return out
    y_name = _ycol(d)
    d = d.dropna(subset=[y_name])
    if d.empty or "lap_number" not in d.columns:
        return out
    med = d.groupby("lap_number")[y_name].transform("median")
    cnt = d.groupby("lap_number")[y_name].transform("size")
    d = d.assign(_resid=pd.to_numeric(d[y_name], errors="coerce") - med)
    d = d[cnt >= 5].dropna(subset=["_resid"])       # a median over fewer than five cars is noise
    if d.empty:
        return out
    lv = d.groupby("driver")["_resid"].median()
    n = d.groupby("driver")["_resid"].size()
    centre = float(np.average(lv.to_numpy(float), weights=n.reindex(lv.index).to_numpy(float)))
    out["levels"] = {str(k): float(v - centre) for k, v in lv.items()}
    out["n"] = {str(k): int(v) for k, v in n.items()}
    out["ycol"] = y_name
    out["n_laps"] = int(len(d))
    return out


def _race_stint_rates(race: pd.DataFrame) -> dict:
    """Per driver and compound: the measured race stint rate, s/lap of age.

    A diagnostic only - it is reported in `deg_rate_by_compound.detail` and
    never folded into the value, because the tyre model's rates are a posterior
    the live engine re-prices (WP-B/WP-D), not something this module rewrites.
    Each stint's rate is the mean of its last three laps minus the mean of its
    first three, over the age between them, on the stint's own trend column.
    """
    out: dict = {}
    d = _valid(race, traffic=False, long_runs=False)
    if d is None or len(d) == 0 or "stint_uid" not in d.columns:
        return out
    y_name = _ycol(d)
    for (drv, uid), g in d.groupby(["driver", "stint_uid"], sort=False):
        g = g.dropna(subset=[y_name])
        if len(g) < 8:
            continue
        age = pd.to_numeric(g.get("tyre_age", g.get("lap_in_stint")), errors="coerce").to_numpy(float)
        y = pd.to_numeric(g[y_name], errors="coerce").to_numpy(float)
        o = np.argsort(age)
        first, last = o[:3], o[-3:]
        d_age = float(age[last].mean() - age[first].mean())
        if not np.isfinite(d_age) or d_age < 4:
            continue
        comp = str(g["compound"].iloc[0])
        out.setdefault(str(drv), {}).setdefault(comp, []).append(
            float((y[last].mean() - y[first].mean()) / d_age))
    return {d: {c: round(float(np.mean(v)), 4) for c, v in per.items()} for d, per in out.items()}


# --------------------------------------------------------------------------
# Sector degradation, from the FastF1 cache
# --------------------------------------------------------------------------


def sector_slopes(event: Event | str, *, sessions=None, min_laps: int = 6) -> dict:
    """Per driver: the slope of each sector time against tyre age, s/lap.

    The lap tables this project builds carry no sector times (`src.ingest`
    loads none), so they are read straight from the FastF1 cache - offline, the
    way `src.history._fastf1` does it - and joined to nothing: the session's own
    stints, compounds and tyre life come along with them.  A stint's slope is
    the least-squares slope of that sector over the stint, and a driver's slope
    is the lap-weighted mean of its stints'.  Laps are filtered the way the lap
    table filters them (accurate, green, not an in/out lap, inside the slow-lap
    margin) using FastF1's own columns.

    Returns `{driver: {"S1": ..., "S2": ..., "S3": ..., "n_laps": n,
    "n_stints": k}}`, or `{}` when the cache has no sector times for this
    weekend - in which case the caller must report the quantity as unavailable
    rather than substitute the lap-level number.
    """
    ev = get_event(event) if isinstance(event, str) else event
    try:
        import logging

        import fastf1

        from src.config import FASTF1_CACHE
        fastf1.Cache.enable_cache(str(FASTF1_CACHE))
        fastf1.Cache.offline_mode(True)
        logging.getLogger("fastf1").setLevel(logging.ERROR)
    except Exception:
        return {}
    names = tuple(sessions or getattr(ev, "practice_sessions", ("Practice 1", "Practice 2", "Practice 3")))
    frames = []
    for name in names:
        try:
            s = fastf1.get_session(ev.ff1_year, ev.ff1_round, name)
            s.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = s.laps
        except Exception:
            continue
        cols = ["Driver", "LapNumber", "Stint", "Compound", "TyreLife", "LapTime", "IsAccurate",
                "PitInTime", "PitOutTime", "TrackStatus", "Sector1Time", "Sector2Time", "Sector3Time"]
        have = [c for c in cols if c in getattr(laps, "columns", [])]
        if not all(c in have for c in ("Sector1Time", "Sector2Time", "Sector3Time", "Driver", "TyreLife")):
            continue
        f = pd.DataFrame(laps[have]).copy()
        f["session"] = name
        frames.append(f)
    if not frames:
        return {}
    d = pd.concat(frames, ignore_index=True)
    for c in ("LapTime", "Sector1Time", "Sector2Time", "Sector3Time"):
        d[c] = pd.to_timedelta(d[c], errors="coerce").dt.total_seconds()
    d = d[d.get("IsAccurate", True).fillna(False).astype(bool)] if "IsAccurate" in d else d
    if "PitInTime" in d:
        d = d[d["PitInTime"].isna()]
    if "PitOutTime" in d:
        d = d[d["PitOutTime"].isna()]
    if "TrackStatus" in d:
        ts = d["TrackStatus"].astype(str).str.strip()
        d = d[(ts == "1") | (ts == "") | (ts == "nan")]
    d = d.dropna(subset=["LapTime", "TyreLife"])
    if d.empty:
        return {}
    med = d.groupby("session")["LapTime"].transform("median")
    d = d[d["LapTime"] <= med + 5.0]                 # `SLOW_LAP_MARGIN_S`, on FastF1's own columns
    if "Compound" in d:
        d = d[d["Compound"].astype(str).str.upper().isin(VALID_COMPOUNDS)]
    if d.empty:
        return {}
    out: dict = {}
    keys = ["Driver", "session", "Stint"] if "Stint" in d else ["Driver", "session"]
    for k, g in d.groupby(keys, sort=False):
        if len(g) < int(min_laps):
            continue
        drv = str(k[0])
        age = pd.to_numeric(g["TyreLife"], errors="coerce").to_numpy(float)
        o = out.setdefault(drv, {"S1": [], "S2": [], "S3": [], "w": [], "n_laps": 0, "n_stints": 0})
        ok_any = False
        for i, col in enumerate(("Sector1Time", "Sector2Time", "Sector3Time"), start=1):
            y = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(age)
            if int(ok.sum()) < int(min_laps) or np.ptp(age[ok]) < 3:
                o[f"S{i}"].append(np.nan)
                continue
            A = np.column_stack([np.ones(int(ok.sum())), age[ok]])
            beta, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
            o[f"S{i}"].append(float(beta[1]))
            ok_any = True
        if ok_any:
            o["w"].append(float(len(g)))
            o["n_laps"] += int(len(g))
            o["n_stints"] += 1
        else:
            for i in (1, 2, 3):
                o[f"S{i}"].pop()
    res: dict = {}
    for drv, o in out.items():
        if not o["w"]:
            continue
        row = {"n_laps": int(o["n_laps"]), "n_stints": int(o["n_stints"])}
        w = np.asarray(o["w"], dtype=float)
        for i in (1, 2, 3):
            v = np.asarray(o[f"S{i}"], dtype=float)
            m = np.isfinite(v)
            row[f"S{i}"] = float(np.average(v[m], weights=w[m])) if m.any() else None
        res[drv] = row
    return res


def _pool_sectors(slopes: dict, drivers) -> dict | None:
    rows = [slopes[d] for d in drivers if d in slopes]
    if not rows:
        return None
    out = {}
    for i in (1, 2, 3):
        vals = [(r[f"S{i}"], r["n_laps"]) for r in rows if r.get(f"S{i}") is not None]
        out[f"S{i}"] = (float(np.average([v for v, _ in vals], weights=[n for _, n in vals]))
                        if vals else None)
    return out


# --------------------------------------------------------------------------
# The car state
# --------------------------------------------------------------------------


@dataclass
class CarState:
    """What this weekend knows about one Haas, quantity by quantity.

    Every field but the identifiers is an `Estimate`, so nothing can be read
    off this object without also being able to read how much evidence stands
    behind it and whether it is the car's own number or its team's.
    """

    driver: str = ""
    team: str = HAAS_TEAM
    event: str = ""
    teammate: str | None = None

    pace_offset_s: Estimate = field(default_factory=Estimate)
    deg_rate_by_compound: Estimate = field(default_factory=Estimate)
    age_sensitivity: Estimate = field(default_factory=Estimate)
    warmup_s: Estimate = field(default_factory=Estimate)
    consistency_s: Estimate = field(default_factory=Estimate)
    traffic_sensitivity: Estimate = field(default_factory=Estimate)
    push_response: Estimate = field(default_factory=Estimate)
    sector_deg: Estimate = field(default_factory=Estimate)
    pace_vs_teammate_s: Estimate = field(default_factory=Estimate)

    race_factor: dict = field(default_factory=dict)      # the history *prior*, {"factor", "ln_sd"}
    dev_draws: dict = field(default_factory=dict, repr=False)   # {compound: (draws,)} pooled dev
    evidence: dict = field(default_factory=dict)
    source: str = ""

    QUANTITIES = ("pace_offset_s", "deg_rate_by_compound", "age_sensitivity", "warmup_s",
                  "consistency_s", "traffic_sensitivity", "push_response", "sector_deg",
                  "pace_vs_teammate_s")

    def estimates(self) -> dict:
        return {name: getattr(self, name) for name in self.QUANTITIES}

    def as_dict(self) -> dict:
        return _json_clean({
            "driver": self.driver, "team": self.team, "event": self.event,
            "teammate": self.teammate, "race_factor": self.race_factor,
            "evidence": self.evidence, "source": self.source,
            **{name: est.as_dict() for name, est in self.estimates().items()},
        })

    def table(self) -> pd.DataFrame:
        rows = []
        for name, est in self.estimates().items():
            rows.append({"quantity": name, "value": est.value, "n_evidence": est.n_evidence,
                         "shrink_weight": round(float(est.shrink_weight), 3), "source": est.source,
                         "own": est.own, "team_value": est.team_value, "units": est.units})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Building it
# --------------------------------------------------------------------------


class HaasCarModel:
    """Both Haas cars' states for one weekend, and the terms they imply."""

    def __init__(self, event: str, team: str, states: dict, field_state: dict, source: str = ""):
        self.event = event
        self.team = team
        self.states = states
        self.field = field_state
        self.source = source

    # -- access ----------------------------------------------------------

    @property
    def drivers(self) -> list:
        return list(self.states)

    def __getitem__(self, driver: str) -> CarState:
        return self.states[driver]

    def terms(self, **kw) -> dict:
        return {d: car_terms(s, **kw) for d, s in self.states.items()}

    def explain(self, plans: dict | None = None) -> list:
        """`explain_difference` for the two cars, with plans `{driver: plan}`."""
        ds = self.drivers
        if len(ds) < 2:
            return ["only one Haas car has a state this weekend: nothing to compare"]
        a, b = ds[0], ds[1]
        p = plans or {}
        return explain_difference(self.states[a], self.states[b], p.get(a), p.get(b))

    def table(self) -> pd.DataFrame:
        return pd.concat([s.table().assign(driver=d) for d, s in self.states.items()],
                         ignore_index=True)

    def as_dict(self) -> dict:
        return _json_clean({"event": self.event, "team": self.team, "source": self.source,
                            "field": self.field,
                            "cars": {d: s.as_dict() for d, s in self.states.items()}})

    # -- construction ----------------------------------------------------

    @classmethod
    def from_weekend(cls, event: Event | str, model, clean_practice: pd.DataFrame, *,
                     race_laps: pd.DataFrame | None = None, calibration=None,
                     drivers=HAAS_DRIVERS, team: str = HAAS_TEAM,
                     practice_laps: pd.DataFrame | None = None,
                     sector_times: bool = True) -> "HaasCarModel":
        """Estimate both cars from this weekend's data.

        `model` is the weekend's `TyreModel` (its `driver_dev` is the fit's own
        per-driver deviation and its `rate(c)` the field rate), `clean_practice`
        the clean practice lap table and `calibration` the weekend's
        leave-one-out `Calibration` - whose `driver_factors` enter as a *prior*
        only, through `percar.shrink_factor`.

        `practice_laps` is the *uncleaned* practice lap table, and it is
        optional for one reason: the clean cascade drops every lap run within
        `TRAFFIC_GAP_S` of the car ahead, so the clean table contains no
        evidence at all about traffic sensitivity.  Pass it and the traffic
        estimate is measured; leave it out and the estimate stays at the field
        value of 1.0 and says so.  `race_laps` adds the race's own stints (the
        live and post-race path).  `sector_times=False` skips the FastF1 cache
        read, which is what a test that has no cache wants.
        """
        ev = get_event(event) if isinstance(event, str) else event
        drivers = [str(d) for d in drivers]
        cal = calibration
        clean = clean_practice if clean_practice is not None else pd.DataFrame()

        # -- who is in which team, from this weekend's own tables ----------
        teams = {}
        for df in (clean, practice_laps, race_laps):
            if df is not None and len(df) and "team" in df.columns:
                teams.update(df.drop_duplicates("driver").set_index("driver")["team"].astype(str).to_dict())
        teams = {str(k): str(v) for k, v in teams.items()}
        # A car that ran no laps at all is absent from every table, and its team
        # is not a guess - it is the identity the caller asked for.  Without this
        # the team pooling would drop it and it would run on the field model.
        for d in drivers:
            teams.setdefault(d, team)
        mates = {d: next((o for o in teams if o != d and teams.get(o) == teams.get(d, team)), None)
                 for d in drivers}
        team_drivers = sorted({d for d, t in teams.items() if t == team} | set(drivers))
        field_drivers = sorted(set(teams) | set(drivers))

        # -- the evidence tables -------------------------------------------
        # Warm-up, consistency and traffic are within-stint quantities and are
        # measured on whatever stints exist: practice always, the race too when
        # a race table is passed.  The warm-up and the residual spread are
        # taken from the *clean* table, because it carries `lap_time_corr` -
        # the excess of the first flying lap is a one-lap quantity read off a
        # backward extrapolation of the stint trend, and doing that on raw lap
        # times puts the stint's fuel burn into it.  Traffic has to use the
        # uncleaned table, since the clean cascade drops every in-traffic lap.
        prac_all = practice_laps if practice_laps is not None else clean
        within = [df for df in (clean, race_laps) if df is not None and len(df)]
        traffic_src = [df for df in (prac_all, race_laps) if df is not None and len(df)]

        trend = _trend_stats_frames(within)
        traffic = _traffic_excess_frames(traffic_src)
        lv_p = _levels_practice(clean)
        lv_r = _levels_race(race_laps) if (race_laps is not None and len(race_laps)) else {"levels": {}, "n": {}}
        sectors = sector_slopes(ev) if sector_times else {}
        race_rates = _race_stint_rates(race_laps) if (race_laps is not None and len(race_laps)) else {}

        # -- degradation: the fit's own deviation, pooled over the team -----
        n_clean = _valid(clean, traffic=False).groupby("driver").size().to_dict() \
            if (clean is not None and len(clean)) else {}
        dev_by_driver = {d: dict(v) for d, v in (getattr(model, "driver_dev", None) or {}).items()}
        pooled = percar.team_pooled_dev(dev_by_driver, teams,
                                        n_laps_by_driver={str(k): float(v) for k, v in n_clean.items()})
        rates = {c: float(np.asarray(model.rate(c)).mean()) for c in getattr(model, "compounds", [])}
        team_dev = _team_dev_mean(dev_by_driver, team_drivers)

        field_trend = _pool_trend(trend, field_drivers)
        field_traffic = _pool_traffic(traffic, field_drivers)
        field_state = {
            "pace_offset_s": 0.0,
            "deg_rate_by_compound": {c: round(v, 4) for c, v in rates.items()},
            "warmup_s": field_trend["warmup_s"],
            "n_warmup_stints": field_trend["n_warmup_stints"],
            "consistency_s": field_trend["consistency_s"],
            "traffic_excess_s": field_traffic["excess_s"],
            "n_traffic_laps": field_traffic["n"],
            "n_drivers": len(field_drivers),
            "sector_deg": _pool_sectors(sectors, field_drivers),
            "push_response_s": float(getattr(model, "manage_cost_s", MANAGE_COST_S)),
        }
        team_trend = _pool_trend(trend, team_drivers)
        team_traffic = _pool_traffic(traffic, team_drivers)
        team_sectors = _pool_sectors(sectors, team_drivers)

        states = {}
        for drv in drivers:
            states[drv] = _build_state(
                drv, ev=ev, team=team, mate=mates.get(drv), model=model, cal=cal,
                rates=rates, dev_own=dev_by_driver.get(drv), dev_pooled=pooled.get(drv),
                team_dev=team_dev, n_clean=n_clean, trend=trend, team_trend=team_trend,
                traffic=traffic, team_traffic=team_traffic, lv_p=lv_p, lv_r=lv_r,
                sectors=sectors, team_sectors=team_sectors, field_state=field_state,
                race_rates=race_rates, have_practice_all=(practice_laps is not None),
                have_race=(race_laps is not None and len(race_laps) > 0))
        src = (f"{ev.key}: practice long runs"
               + (" + the uncleaned practice table (traffic)" if practice_laps is not None else "")
               + (" + race laps" if (race_laps is not None and len(race_laps)) else "")
               + (f"; sector times from the FastF1 cache ({len(sectors)} cars)" if sectors
                  else "; no sector times in the FastF1 cache"))
        return cls(ev.key, team, states, field_state, source=src)


def _team_dev_mean(dev_by_driver: dict, team_drivers) -> dict | None:
    """The team's own mean `dev[d, c]` per compound, as posterior draws."""
    per: dict = {}
    for d in team_drivers:
        for c, arr in (dev_by_driver.get(d) or {}).items():
            per.setdefault(c, []).append(np.asarray(arr, dtype=float))
    if not per:
        return None
    return {c: np.mean(np.stack(v), axis=0) for c, v in per.items()}


def _build_state(drv: str, *, ev, team, mate, model, cal, rates, dev_own, dev_pooled, team_dev,
                 n_clean, trend, team_trend, traffic, team_traffic, lv_p, lv_r, sectors,
                 team_sectors, field_state, race_rates, have_practice_all, have_race) -> CarState:
    n_own_clean = float(n_clean.get(drv, 0.0))

    # -- pace level ----------------------------------------------------------
    own_p, n_p = lv_p["levels"].get(drv), float(lv_p["n"].get(drv, 0))
    own_r, n_r = lv_r["levels"].get(drv), float(lv_r["n"].get(drv, 0))
    if own_p is None and own_r is None:
        own_pace, n_pace = None, 0.0
    elif own_r is None:
        own_pace, n_pace = own_p, n_p
    elif own_p is None:
        own_pace, n_pace = own_r, n_r
    else:
        n_pace = n_p + n_r
        own_pace = (n_p * own_p + n_r * own_r) / n_pace if n_pace > 0 else own_p
    mate_p = lv_p["levels"].get(mate) if mate else None
    mate_r = lv_r["levels"].get(mate) if mate else None
    n_mate = float(lv_p["n"].get(mate, 0) + lv_r["n"].get(mate, 0)) if mate else 0.0
    mate_pace = None
    if mate_p is not None or mate_r is not None:
        a, na = (mate_p, float(lv_p["n"].get(mate, 0))) if mate_p is not None else (0.0, 0.0)
        b, nb = (mate_r, float(lv_r["n"].get(mate, 0))) if mate_r is not None else (0.0, 0.0)
        mate_pace = (na * a + nb * b) / (na + nb) if (na + nb) > 0 else None
    team_pace = None
    if own_pace is not None or mate_pace is not None:
        vals = [(v, n) for v, n in ((own_pace, n_pace), (mate_pace, n_mate)) if v is not None and n > 0]
        team_pace = (float(sum(v * n for v, n in vals) / sum(n for _, n in vals)) if vals else None)
    pace = _shrink(own_pace, team_pace, 0.0, n_own=n_pace, k=PACE_K_LAPS,
                   n_team=(n_pace + n_mate), k_team=PACE_K_LAPS * TEAM_K_MULT, units="s/lap",
                   note=("the level of the car's clean long-run laps against the field's own "
                         "compound/age trend; negative is quicker than the field.  It is "
                         "confounded by the run programme - practice long runs differ in fuel "
                         "load and in how hard they are being driven, and neither is observable "
                         "here - so it is reported, not charged: no cost in the objective reads "
                         "it (see car_terms)"
                         + (" (practice and race laps pooled by lap count)" if (own_p is not None and own_r is not None) else "")),
                   detail={"practice_s": own_p, "n_practice_laps": int(n_p),
                           "race_s": own_r, "n_race_laps": int(n_r),
                           "ycol": lv_p.get("ycol"),
                           "n_by_compound": (lv_p.get("n_by_compound") or {}).get(drv, {})})

    # -- pace vs team-mate ---------------------------------------------------
    if own_pace is None or mate_pace is None or mate is None:
        vs_mate = Estimate(value=0.0, n_evidence=0, shrink_weight=0.0, source="team", own=None,
                           team_value=0.0, field_value=0.0, units="s/lap",
                           note=("no comparison this weekend: "
                                 + ("this car has no clean long run" if own_pace is None else
                                    "the team-mate has no clean long run" if mate else
                                    "no team-mate in this weekend's tables")))
    else:
        vs_mate = _shrink(float(own_pace - mate_pace), 0.0, 0.0,
                          n_own=min(n_pace, n_mate), k=PACE_K_LAPS, units="s/lap",
                          note=(f"this car's level minus {mate}'s, shrunk toward 0 (the two cars "
                                f"are the same car) by the thinner of the two samples"),
                          detail={"teammate": mate, "n_teammate_laps": int(n_mate),
                                  "raw_s": float(own_pace - mate_pace)})

    # -- degradation ---------------------------------------------------------
    dev_pooled = dev_pooled or {}
    own_dev_m = ({c: float(np.asarray(v).mean()) for c, v in dev_own.items()} if dev_own else None)
    team_dev_m = ({c: float(np.asarray(v).mean()) for c, v in team_dev.items()} if team_dev else None)
    pooled_dev_m = {c: float(np.asarray(v).mean()) for c, v in dev_pooled.items()}
    eff = {c: max(rates[c] + pooled_dev_m.get(c, 0.0), percar.RATE_FLOOR_S) for c in rates}
    team_eff = ({c: max(rates[c] + (team_dev_m or {}).get(c, 0.0), percar.RATE_FLOOR_S) for c in rates}
                if team_dev_m else None)
    w_dev = _w(n_own_clean, POOL_K_LAPS) if own_dev_m else 0.0
    deg = Estimate(value={c: round(v, 4) for c, v in eff.items()},
                   n_evidence=int(n_own_clean), shrink_weight=w_dev,
                   source=("driver" if w_dev >= 0.5 else "team" if team_dev_m else "field"),
                   own=({c: round(max(rates[c] + own_dev_m.get(c, 0.0), percar.RATE_FLOOR_S), 4)
                         for c in rates} if own_dev_m else None),
                   team_value=({c: round(v, 4) for c, v in team_eff.items()} if team_eff else
                               {c: round(v, 4) for c, v in rates.items()}),
                   field_value={c: round(v, 4) for c, v in rates.items()}, units="s/lap of age",
                   note=("the field rate plus the fit's own dev[d, c], pooled driver -> team by "
                         f"percar.team_pooled_dev at k = {POOL_K_LAPS:.0f} clean laps; the "
                         "team -> field step is the fit's own hierarchical prior on dev "
                         "(sigma_dev ~ HalfNormal(0.02)) and is not applied twice here"),
                   detail={"dev_own_s_per_lap": ({c: round(v, 5) for c, v in own_dev_m.items()}
                                                 if own_dev_m else None),
                           "dev_team_s_per_lap": ({c: round(v, 5) for c, v in team_dev_m.items()}
                                                  if team_dev_m else None),
                           "dev_pooled_s_per_lap": {c: round(v, 5) for c, v in pooled_dev_m.items()},
                           "race_measured_s_per_lap": (race_rates.get(drv) if have_race else None),
                           "n_clean_laps_by_compound": (lv_p.get("n_by_compound") or {}).get(drv, {})})
    scale = (float(np.mean([eff[c] / rates[c] for c in rates if rates[c] > 0])) if rates else 1.0)
    age = Estimate(value=float(np.mean([pooled_dev_m.get(c, 0.0) for c in rates])) if rates else 0.0,
                   n_evidence=int(n_own_clean), shrink_weight=w_dev, source=deg.source,
                   own=(float(np.mean([own_dev_m.get(c, 0.0) for c in rates])) if own_dev_m and rates else None),
                   team_value=(float(np.mean([(team_dev_m or {}).get(c, 0.0) for c in rates]))
                               if team_dev_m and rates else 0.0),
                   field_value=0.0, units="s/lap per lap of age",
                   note=("the same pooled deviation as one number: how much more lap time per lap "
                         "of tyre age this car gives away than the field"),
                   detail={"rate_scale": round(scale, 4)})

    # -- warm-up and consistency --------------------------------------------
    o_t = trend.get(drv) or {}
    n_warm_own = float(o_t.get("n_warmup_stints", 0))
    n_warm_team = float(team_trend.get("n_warmup_stints", 0))
    warm = _shrink((o_t.get("warmup_s") if n_warm_own >= MIN_WARMUP_STINTS else None),
                   (team_trend.get("warmup_s") if n_warm_team >= MIN_WARMUP_STINTS else None),
                   field_state["warmup_s"],
                   n_own=n_warm_own, k=WARMUP_K_STINTS,
                   n_team=n_warm_team, k_team=WARMUP_K_STINTS * TEAM_K_MULT, units="s per stint",
                   note=(f"mean excess of laps {'/'.join(str(x) for x in WARMUP_LAPS)} of a stint "
                         f"over that stint's own trend (fitted from lap {TREND_FROM_LAP}); in a "
                         f"practice table lap 1 is the out-lap and is filtered out, so this is "
                         f"normally the first flying lap.  A pool of fewer than "
                         f"{MIN_WARMUP_STINTS} stints contributes nothing"),
                   detail={"n_stints": int(o_t.get("n_stints", 0)),
                           "n_warmup_stints": int(n_warm_own),
                           "per_stint_s": [round(v, 3) for v in (o_t.get("warmup") or [])],
                           "own_measured_s": o_t.get("warmup_s"),
                           "team_n_warmup_stints": int(n_warm_team),
                           "min_stints": MIN_WARMUP_STINTS,
                           "field_s": field_state["warmup_s"]})
    cons = _shrink(o_t.get("consistency_s"), team_trend.get("consistency_s"),
                   field_state["consistency_s"], n_own=float(o_t.get("n_laps", 0)),
                   k=CONSISTENCY_K_LAPS, n_team=float(team_trend.get("n_laps", 0)),
                   k_team=CONSISTENCY_K_LAPS * TEAM_K_MULT, units="s",
                   note="residual SD about the stint trend, over the laps the trend was fitted on",
                   detail={"n_laps": int(o_t.get("n_laps", 0)),
                           "field_s": field_state["consistency_s"]})

    # -- traffic sensitivity -------------------------------------------------
    f_ex, f_n = field_state["traffic_excess_s"], field_state["n_traffic_laps"]
    o_tr = traffic.get(drv) or {}
    measurable = (f_ex is not None and f_ex >= MIN_FIELD_TRAFFIC_S and f_n >= MIN_FIELD_TRAFFIC_LAPS)
    if not measurable:
        why = ("the uncleaned lap table was not passed, so no lap inside "
               f"{TRAFFIC_GAP_S:.0f} s of the car ahead is available"
               if not have_practice_all else
               f"the field's own in-traffic excess is {('%.3f' % f_ex) if f_ex is not None else 'not measurable'} s/lap "
               f"over {int(f_n)} laps, below the {MIN_FIELD_TRAFFIC_S:.2f} s / "
               f"{MIN_FIELD_TRAFFIC_LAPS} lap floor at which a ratio to it means anything")
        traf = Estimate(value=1.0, n_evidence=int(o_tr.get("n", 0)), shrink_weight=0.0,
                        source="field", own=None, team_value=1.0, field_value=1.0, units="x",
                        note=f"not measurable this weekend: {why}",
                        detail={"field_excess_s": f_ex, "field_n_laps": int(f_n),
                                "own_excess_s": o_tr.get("excess_s"), "own_n_laps": int(o_tr.get("n", 0))})
    else:
        n_own_tr = float(o_tr.get("n", 0))
        n_team_tr = float(team_traffic.get("n", 0))
        own_ratio = (float(o_tr["excess_s"] / f_ex)
                     if (o_tr.get("excess_s") is not None and n_own_tr >= MIN_OWN_TRAFFIC_LAPS)
                     else None)
        t_ex = team_traffic.get("excess_s")
        team_ratio = (float(t_ex / f_ex)
                      if (t_ex is not None and n_team_tr >= MIN_OWN_TRAFFIC_LAPS) else None)
        traf = _shrink(own_ratio, team_ratio, 1.0, n_own=n_own_tr, k=TRAFFIC_K_LAPS,
                       n_team=n_team_tr, k_team=TRAFFIC_K_LAPS * TEAM_K_MULT,
                       units="x",
                       note=(f"the car's excess over the stint trend within {TRAFFIC_GAP_S:.0f} s of "
                             f"the car ahead, divided by the field's same excess "
                             f"({f_ex:.3f} s/lap over {int(f_n)} laps); 1.0 is a car that loses "
                             f"what everyone loses"),
                       detail={"own_excess_s": o_tr.get("excess_s"), "own_n_laps": int(n_own_tr),
                               "team_excess_s": t_ex, "team_n_laps": int(n_team_tr),
                               "field_excess_s": f_ex, "field_n_laps": int(f_n),
                               "min_own_laps": MIN_OWN_TRAFFIC_LAPS})
        if own_ratio is None and n_own_tr > 0:
            traf.note += (f"; this car's {int(n_own_tr)} in-traffic lap(s) are below the "
                          f"{MIN_OWN_TRAFFIC_LAPS}-lap floor and contribute nothing")
        lo, hi = TRAFFIC_MULT_BAND
        if traf.value is not None and not (lo <= float(traf.value) <= hi):
            traf.detail["clipped_from"] = round(float(traf.value), 3)
            traf.value = float(min(max(float(traf.value), lo), hi))
            traf.note += f"; clipped to the [{lo}, {hi}] band (outside it the sample is noise, not a car)"

    # -- push response: not identifiable ------------------------------------
    push = Estimate(value=float(getattr(model, "manage_cost_s", MANAGE_COST_S)), n_evidence=0,
                    shrink_weight=0.0, source="team", own=None,
                    team_value=float(getattr(model, "manage_cost_s", MANAGE_COST_S)),
                    field_value=float(getattr(model, "manage_cost_s", MANAGE_COST_S)),
                    units="s/lap at push 0",
                    note=("not identifiable pre-race: a practice long run is a full-push "
                          "experiment by construction, so this weekend carries no within-car "
                          "variation in push from which a per-car management response could be "
                          "estimated.  The calibrated team/field value is used, fully shrunk"
                          + ("; the race's own stint rates are in "
                             "deg_rate_by_compound.detail.race_measured_s_per_lap as a diagnostic"
                             if have_race else "")),
                    detail={"manage_floor": float(getattr(model, "manage_floor", float("nan")))})

    # -- sector degradation --------------------------------------------------
    own_sec = sectors.get(drv)
    if not sectors:
        sec = _unavailable("the FastF1 cache carries no sector times for this weekend's practice "
                           "sessions, and the project's lap tables carry none either "
                           "(src.ingest loads no sector columns)", units="s/lap of age")
    else:
        sec = _shrink(({k: own_sec[k] for k in ("S1", "S2", "S3")} if own_sec else None),
                      team_sectors, field_state["sector_deg"],
                      n_own=float((own_sec or {}).get("n_laps", 0)), k=SECTOR_K_LAPS,
                      n_team=float(sum((sectors.get(d) or {}).get("n_laps", 0)
                                       for d in (drv, mate) if d)),
                      k_team=SECTOR_K_LAPS * TEAM_K_MULT, units="s/lap of age",
                      note="per-sector slope against tyre age over this car's long runs, from the "
                           "FastF1 cache's sector times",
                      detail={"n_stints": int((own_sec or {}).get("n_stints", 0)),
                              "n_cars_with_sectors": len(sectors)})

    rf = _race_factor(cal, drv)
    return CarState(
        driver=drv, team=team, event=getattr(ev, "key", str(ev)), teammate=mate,
        pace_offset_s=pace, deg_rate_by_compound=deg, age_sensitivity=age, warmup_s=warm,
        consistency_s=cons, traffic_sensitivity=traf, push_response=push, sector_deg=sec,
        pace_vs_teammate_s=vs_mate, race_factor=rf, dev_draws=dev_pooled,
        evidence={"n_clean_practice_laps": int(n_own_clean),
                  "n_practice_long_run_stints": int(o_t.get("n_stints", 0)),
                  "n_in_traffic_laps": int((traffic.get(drv) or {}).get("n", 0)),
                  "n_race_laps": int(lv_r["n"].get(drv, 0)),
                  "n_sector_laps": int((own_sec or {}).get("n_laps", 0))},
        source=("every quantity above is measured on this weekend; the race-history rate factor "
                "is a prior and enters only through percar.shrink_factor"))


def _race_factor(cal, drv: str) -> dict:
    """The driver's race-history rate factor as a *prior*: the measurement, how
    precisely it was measured, and what survives `percar.shrink_factor`."""
    if cal is None:
        return {"factor": 1.0, "ln_sd": None, "shrunk": 1.0,
                "role": "prior (no calibration passed)"}
    factors = getattr(cal, "driver_factors", None) or {}
    sds = getattr(cal, "driver_factor_ln_sd", None) or {}
    f = factors.get(drv, 1.0)
    f = float(f.get("factor", 1.0)) if isinstance(f, dict) else float(f)
    sd = sds.get(drv)
    return {"factor": f, "ln_sd": (float(sd) if sd is not None else None),
            "shrunk": float(percar.shrink_factor(f, sd)),
            "team_factor": float((getattr(cal, "team_factors", None) or {}).get(HAAS_TEAM, float("nan"))),
            "role": ("prior from previous races, precision-weighted by percar.shrink_factor; "
                     "never evidence about this weekend"),
            "source": getattr(cal, "source", "")}


# --------------------------------------------------------------------------
# What the search and the live engine take
# --------------------------------------------------------------------------


def car_terms(state: CarState, *, base_warmup_s: float = OUT_LAP_PENALTY_S) -> dict:
    """The per-car keywords: `warmup_s`, `traffic_mult`, `race_factor`, `dev_override`.

    Two of the four are *field-relative on purpose*.  The warm-up cost the
    objective charges (`OUT_LAP_PENALTY_S`) and the dirty-air cost it charges
    (`traffic_s_per_lap`) are calibrated field numbers; substituting a
    practice-measured absolute for either would move a car's plan for reasons
    that have nothing to do with the car.  So the car only carries the
    *difference* from the field:

        warmup_s    = base + (this car's warm-up - the field's), floored at 0
        traffic_mult= this car's in-traffic excess / the field's

    A car with no evidence therefore gets exactly `base_warmup_s` and 1.0, and
    the search reproduces the field objective bit for bit.  The warm-up
    difference is held inside `WARMUP_DELTA_BAND_S`: a practice sample of one
    or two stints cannot establish a bigger per-car difference than that, and
    the term is charged on every stint of every plan, so an unbounded one would
    decide the stop count.
    """
    w = state.warmup_s
    dw = 0.0
    fv = w.field_value if w is not None else None
    if w is not None and w.value is not None and fv is not None:
        dw = float(w.value) - float(fv)
        dw = float(min(max(dw, -WARMUP_DELTA_BAND_S), WARMUP_DELTA_BAND_S))
    mult = 1.0
    if state.traffic_sensitivity is not None and state.traffic_sensitivity.value is not None:
        mult = float(state.traffic_sensitivity.value)
    return {"warmup_s": float(max(float(base_warmup_s) + dw, 0.0)),
            "traffic_mult": mult,
            "race_factor": {"factor": float((state.race_factor or {}).get("factor", 1.0)),
                            "ln_sd": (state.race_factor or {}).get("ln_sd")},
            "dev_override": dict(state.dev_draws or {})}


# --------------------------------------------------------------------------
# Why the two cars' recommendations differ
# --------------------------------------------------------------------------


def _plan_label(plan) -> str:
    if plan is None:
        return "-"
    if isinstance(plan, str):
        return plan
    for k in ("best", "label", "best_label"):
        v = plan.get(k) if hasattr(plan, "get") else None
        if isinstance(v, str) and v:
            return v
    comps = plan.get("compounds") if hasattr(plan, "get") else None
    pits = plan.get("pit_laps") if hasattr(plan, "get") else None
    if comps is not None:
        c = "-".join(comps) if isinstance(comps, (list, tuple)) else str(comps)
        return f"{c} @ {pits}"
    return str(plan)


def _first_stop(plan) -> int | None:
    if plan is None or isinstance(plan, str):
        return None
    v = plan.get("first_stop") if hasattr(plan, "get") else None
    if v is not None and not (isinstance(v, float) and math.isnan(v)):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    pits = plan.get("pit_laps") if hasattr(plan, "get") else None
    if isinstance(pits, str):
        pits = [int(x) for x in pits.strip("[]").replace(" ", "").split(",") if x]
    if isinstance(pits, (list, tuple)) and pits:
        return int(pits[0])
    return None


def _shape(plan) -> str | None:
    if plan is None or isinstance(plan, str):
        return None
    c = plan.get("compounds") if hasattr(plan, "get") else None
    if isinstance(c, (list, tuple)):
        return "-".join(str(x) for x in c)
    return str(c) if c else None


def _ev_tag(est: Estimate) -> str:
    return f"n={est.n_evidence}, w={est.shrink_weight:.2f}, {est.source}"


def explain_difference(state_a: CarState, state_b: CarState, plan_a=None, plan_b=None) -> list:
    """Why these two cars are given different plans - in numbers, or not at all.

    The rule the pit wall needs is that a difference between the two cars'
    recommendations is either traceable to a stated, measured difference
    between the two cars, or it is noise.  So this walks the car states,
    reports every difference big enough to move a stop by a lap
    (`MATERIAL_*`), each with both cars' numbers and the evidence behind them,
    and - when nothing is material - says in as many words that the difference
    between the recommendations is within noise.

    Nothing here is a stored sentence about a driver: every line is generated
    from a number in the two states or the two plans.
    """
    a, b = state_a, state_b
    da, db = a.driver or "A", b.driver or "B"
    out = []

    fa, fb = _first_stop(plan_a), _first_stop(plan_b)
    sa, sb = _shape(plan_a), _shape(plan_b)
    la, lb = _plan_label(plan_a), _plan_label(plan_b)
    d_first = (fa - fb) if (fa is not None and fb is not None) else None
    if plan_a is not None or plan_b is not None:
        head = f"{da}: {la}; {db}: {lb}"
        if d_first is not None:
            head += (f" - first stops differ by {abs(d_first)} lap"
                     f"{'s' if abs(d_first) != 1 else ''} ({da} lap {fa}, {db} lap {fb})"
                     if d_first else f" - the same first stop (lap {fa})")
        if sa and sb:
            head += f"; {'the same plan shape' if sa == sb else f'different shapes ({sa} vs {sb})'}"
        out.append(head)
    plans_differ = bool((d_first not in (None, 0)) or (sa and sb and sa != sb)
                        or (la != lb and la != "-" and lb != "-"))

    causes = []

    # -- degradation, per compound ------------------------------------------
    ra = (a.deg_rate_by_compound.value or {}) if a.deg_rate_by_compound else {}
    rb = (b.deg_rate_by_compound.value or {}) if b.deg_rate_by_compound else {}
    for c in sorted(set(ra) & set(rb)):
        d = float(ra[c]) - float(rb[c])
        if abs(d) >= MATERIAL_RATE_S:
            harder, gentler = (da, db) if d > 0 else (db, da)
            causes.append((abs(d) / MATERIAL_RATE_S,
                           f"degradation on {c}: {da} {ra[c]:.4f} vs {db} {rb[c]:.4f} s/lap of age "
                           f"({d:+.4f}); {harder} pays {abs(d) * 25:.1f} s more than {gentler} over a "
                           f"25-lap stint, which pulls its stop earlier "
                           f"[{da}: {_ev_tag(a.deg_rate_by_compound)} | {db}: {_ev_tag(b.deg_rate_by_compound)}]"))

    # -- the scalar quantities ----------------------------------------------
    def scalar(name: str, label: str, thresh: float, fmt: str, unit: str, why: str):
        ea, eb = getattr(a, name, None), getattr(b, name, None)
        if ea is None or eb is None or ea.value is None or eb.value is None:
            return
        if isinstance(ea.value, dict) or isinstance(eb.value, dict):
            return
        d = float(ea.value) - float(eb.value)
        if abs(d) >= thresh:
            causes.append((abs(d) / thresh,
                           f"{label}: {da} {format(float(ea.value), fmt)}{unit} vs {db} "
                           f"{format(float(eb.value), fmt)}{unit} ({d:+.3f}); {why} "
                           f"[{da}: {_ev_tag(ea)} | {db}: {_ev_tag(eb)}]"))

    scalar("pace_offset_s", "pace level", MATERIAL_PACE_S, ".3f", " s/lap",
           "the quicker car reaches any given gap to the car ahead sooner, which moves the lap its "
           "stop is worth taking")
    scalar("age_sensitivity", "age sensitivity", MATERIAL_RATE_S, ".4f", " s/lap per lap of age",
           "the more age-sensitive car gains more from a fresh set")
    scalar("warmup_s", "warm-up", MATERIAL_WARMUP_S, ".2f", " s per stint",
           "a bigger warm-up cost makes every extra stop dearer, which pushes toward fewer stops "
           "and a later first one")
    scalar("consistency_s", "consistency", MATERIAL_CONSISTENCY_S, ".2f", " s",
           "a wider spread about the stint trend is uncertainty, not a cost: it widens the window "
           "rather than moving the lap")
    scalar("traffic_sensitivity", "traffic sensitivity", MATERIAL_TRAFFIC_MULT, ".2f", "x",
           "the more traffic-sensitive car pays more for rejoining into the pack, which pushes its "
           "stop toward a clearer lap")

    # -- the history prior --------------------------------------------------
    ha = float((a.race_factor or {}).get("shrunk", 1.0))
    hb = float((b.race_factor or {}).get("shrunk", 1.0))
    if abs(ha - hb) >= MATERIAL_FACTOR:
        causes.append((abs(ha - hb) / MATERIAL_FACTOR,
                       f"race-history rate factor (a prior, not this weekend's evidence): {da} "
                       f"x{ha:.3f} vs {db} x{hb:.3f} after precision shrinkage "
                       f"(raw x{float((a.race_factor or {}).get('factor', 1.0)):.3f} / "
                       f"x{float((b.race_factor or {}).get('factor', 1.0)):.3f}, ln_sd "
                       f"{(a.race_factor or {}).get('ln_sd')} / {(b.race_factor or {}).get('ln_sd')}); "
                       f"it scales the whole rate, so the higher factor stops earlier"))

    # -- sector degradation, where both cars have it -------------------------
    seca = (a.sector_deg.value or {}) if a.sector_deg else {}
    secb = (b.sector_deg.value or {}) if b.sector_deg else {}
    for s in ("S1", "S2", "S3"):
        va, vb = seca.get(s), secb.get(s)
        if va is None or vb is None:
            continue
        d = float(va) - float(vb)
        if abs(d) >= MATERIAL_RATE_S:
            causes.append((abs(d) / MATERIAL_RATE_S,
                           f"{s} degradation: {da} {float(va):+.4f} vs {db} {float(vb):+.4f} s/lap of "
                           f"age ({d:+.4f}); it says *where* the loss is, and enters no cost "
                           f"[{da}: {_ev_tag(a.sector_deg)} | {db}: {_ev_tag(b.sector_deg)}]"))

    for _, line in sorted(causes, key=lambda t: -t[0]):
        out.append(line)

    if not causes:
        biggest = _largest_gap(a, b)
        out.append(("no car-state difference is material"
                    + (f" (the largest is {biggest})" if biggest else "")
                    + f": {'the difference between the two recommendations is' if plans_differ else 'any difference between the two recommendations would be'}"
                    + " within noise, and the two cars should be given the same plan unless their "
                      "track position differs"))
    elif not plans_differ and (plan_a is not None or plan_b is not None):
        out.append("the two recommendations agree: the differences above are not big enough to "
                   "separate the plans")
    return out


def _largest_gap(a: CarState, b: CarState) -> str | None:
    """The biggest car-state difference as a share of its materiality threshold."""
    best = None
    pairs = [("pace_offset_s", MATERIAL_PACE_S, "s/lap"), ("age_sensitivity", MATERIAL_RATE_S, "s/lap"),
             ("warmup_s", MATERIAL_WARMUP_S, "s"), ("consistency_s", MATERIAL_CONSISTENCY_S, "s"),
             ("traffic_sensitivity", MATERIAL_TRAFFIC_MULT, "x")]
    for name, thresh, unit in pairs:
        ea, eb = getattr(a, name, None), getattr(b, name, None)
        if ea is None or eb is None or ea.value is None or eb.value is None:
            continue
        if isinstance(ea.value, dict) or isinstance(eb.value, dict):
            continue
        d = abs(float(ea.value) - float(eb.value))
        if best is None or d / thresh > best[0]:
            best = (d / thresh, f"{name} {d:.3f} {unit} against a {thresh:g} {unit} threshold")
    ra = (a.deg_rate_by_compound.value or {}) if a.deg_rate_by_compound else {}
    rb = (b.deg_rate_by_compound.value or {}) if b.deg_rate_by_compound else {}
    for c in sorted(set(ra) & set(rb)):
        d = abs(float(ra[c]) - float(rb[c]))
        if best is None or d / MATERIAL_RATE_S > best[0]:
            best = (d / MATERIAL_RATE_S,
                    f"{c} degradation {d:.4f} s/lap against a {MATERIAL_RATE_S:g} s/lap threshold")
    return best[1] if best else None


# --------------------------------------------------------------------------
# `meta["haas"]`
# --------------------------------------------------------------------------


def _plan_rows(per_driver, drivers) -> dict:
    """`{driver: plan dict}` from a `per_driver_plans` frame, a list of its rows,
    or a `{driver: plan}` map - whatever the caller has."""
    if per_driver is None:
        return {}
    if isinstance(per_driver, pd.DataFrame):
        if per_driver.empty or "driver" not in per_driver.columns:
            return {}
        rows = per_driver.to_dict("records")
    elif isinstance(per_driver, dict):
        if all(isinstance(v, dict) for v in per_driver.values()):
            return {str(k): v for k, v in per_driver.items() if str(k) in set(drivers)}
        rows = [per_driver]
    else:
        rows = list(per_driver)
    return {str(r.get("driver")): r for r in rows if str(r.get("driver")) in set(drivers)}


def haas_block(model_states, per_driver_df=None, race_state_block=None, *,
               base_warmup_s: float = OUT_LAP_PENALTY_S) -> dict:
    """The JSON-clean Haas summary for `meta["haas"]`.

    `model_states` is a `HaasCarModel`, a `{driver: CarState}` map or a list of
    `CarState`s; `per_driver_df` the per-car plans (the `per_driver_plans`
    frame, its records, or a `{driver: plan}` map); `race_state_block` the
    weekend's race-state record (`scripts/10_pipeline.py::race_state_block`),
    of which only the summary fields are carried - the per-lap curves stay
    where they are.

    The result is a plain dict of str/int/float/bool/None, so `json.dumps` can
    write it without a custom encoder, and every number in it is one a reader
    can trace: each car's state carries its own counts, weights and sources.
    """
    if isinstance(model_states, HaasCarModel):
        states = dict(model_states.states)
        event, team, src, field_state = (model_states.event, model_states.team,
                                         model_states.source, model_states.field)
    elif isinstance(model_states, dict):
        states = {str(k): v for k, v in model_states.items()}
        event, team, src, field_state = "", HAAS_TEAM, "", {}
    else:
        states = {s.driver: s for s in (model_states or [])}
        event, team, src, field_state = "", HAAS_TEAM, "", {}
    if states and not event:
        first = next(iter(states.values()))
        event, team = first.event, first.team

    plans = _plan_rows(per_driver_df, list(states))
    cars = {}
    for drv, st in states.items():
        t = car_terms(st, base_warmup_s=base_warmup_s)
        cars[drv] = {
            "state": st.as_dict(),
            "terms": {"warmup_s": round(float(t["warmup_s"]), 3),
                      "traffic_mult": round(float(t["traffic_mult"]), 3),
                      "race_factor": t["race_factor"],
                      "dev_override_s_per_lap": {c: round(float(np.asarray(v).mean()), 5)
                                                 for c, v in (t["dev_override"] or {}).items()},
                      "note": ("warmup_s and traffic_mult are field-relative: a car with no "
                               "evidence gets the calibrated field values unchanged")},
            "plan": plans.get(drv),
        }
    ds = list(states)
    expl = []
    if len(ds) >= 2:
        expl = explain_difference(states[ds[0]], states[ds[1]], plans.get(ds[0]), plans.get(ds[1]))
    rsb = race_state_block or {}
    rs = {k: rsb.get(k) for k in ("enabled", "group", "first_stop",
                                  "tyre_optimal_first_stop_in_group", "pack_first_stop_median",
                                  "pack_first_stop_iqr", "race_state_s", "converged")
          if k in rsb}
    if rsb.get("constants"):
        rs["constants"] = rsb["constants"]
    return _json_clean({
        "event": event, "team": team, "drivers": ds, "source": src,
        "field": field_state, "cars": cars, "explanation": expl, "race_state": rs,
        "evidence_rule": ("every quantity is measured on this weekend and shrunk driver -> team "
                          "-> field with a stated pseudo-count; the race-history driver factor is "
                          "a prior and enters only through percar.shrink_factor"),
    })


# --------------------------------------------------------------------------
# The accuracy benchmark's hierarchical per-car variant
# --------------------------------------------------------------------------


def hier_rate_scale_table(*, fit=None, model=None, cal=None, teams=None,
                          n_laps_by_driver: dict | None = None,
                          use_history_prior: bool = True) -> dict:
    """`{driver: multiplicative rate scale}` for the hierarchical per-car model.

    This is the scale the shipped per-car path actually applies - the one
    `scripts/10_pipeline.py` builds and this module's `car_terms` hands to
    `TyreModel.for_driver` - so scoring it against the race's own per-driver
    rates measures the model, not an approximation of it.  Two things separate
    it from `percar.rate_scale_table("team_pooled")`, which the benchmark
    already scores:

    1. the driver -> team pooling is weighted by each car's *own* clean lap
       count (`n_laps_by_driver`), where `team_pooled` as the benchmark calls it
       has no counts and so weights every car equally at `w = 0.5`;
    2. the race-history rate factor enters as a precision-shrunk prior
       (`percar.shrink_factor`), which is the top level of the hierarchy.

    Returned in exactly the shape `percar.rate_scale_table` returns, so the
    benchmark scores it through the same code path as every other variant.
    """
    kind = "combined" if use_history_prior else "team_pooled"
    return percar.rate_scale_table(kind, fit=fit, model=model, cal=cal, teams=teams,
                                   n_laps_by_driver=n_laps_by_driver)


# --------------------------------------------------------------------------
# The evidence report (E4): no tuning, no fitting, just what the weekend says
# --------------------------------------------------------------------------


def _load_weekend(key: str, *, n_draws: int = 300, seed: int = 0):
    """This weekend's tyre model, lap tables and calibration, off the disk.

    The model is built exactly as `bench/bench_ablation.py::percar` builds it -
    the shipped posterior, the calibrated budgets, a fixed 300-draw subsample -
    so the plans this report prints are the plans that benchmark would print.
    """
    import json

    from src.calibration import get_calibration
    from src.config import DATA_PROCESSED
    from src.model_bayes import BayesFit
    from src.tyre import TyreModel

    ev = get_event(key)
    m = json.loads((DATA_PROCESSED / f"meta_{key}.json").read_text())
    cal = get_calibration(ev)
    fit = BayesFit.load(DATA_PROCESSED / f"posterior_{key}.npz")
    total = fit.posterior["lin"].shape[0]
    idx = np.random.default_rng(seed).choice(total, size=min(n_draws, total), replace=False)
    model = TyreModel.from_fit(fit, draws=idx, budget=cal.budgets,
                               manage_floor=cal.manage_wear_floor, manage_cost_s=cal.manage_cost_s)
    clean = pd.read_parquet(DATA_PROCESSED / f"clean_{key}_practice.parquet")
    prac = pd.read_parquet(DATA_PROCESSED / f"laps_{key}_practice.parquet")
    return ev, m, cal, fit, model, clean, prac


def _sim_kwargs(key: str, m: dict, cal, fit, *, race_state: bool = True) -> dict:
    """The per-car search keywords `bench/bench_ablation.py::percar` uses."""
    from src import racestate
    from src.regime import RegimeFactor

    ev = get_event(key)
    caps = {c: int(v) for c, v in (m.get("circuit_history") or {}).get("stint_cap", {}).items()} or None
    kw = dict(regime=RegimeFactor(ratio=float(m["regime"]["ratio"]), ln_sd=float(m["regime"]["ln_sd"])),
              support={k: float(v) for k, v in m["age_support_by_compound"].items()},
              max_per_compound=m["allocation"]["caps"], max_stint=caps,
              undercut_lambda=cal.undercut_lambda, plan_prior=m.get("plan_prior") or {},
              plan_prior_tau_s=cal.plan_prior_tau_s,
              traffic_s_per_lap=cal.dirty_air_for(ev.circuit),
              grid_penalty_s=cal.grid_start_penalty_s)
    if race_state:
        kw["race_state"] = racestate.measure_constants(exclude=key)
    return kw


def evidence_report(keys, *, plans: bool = True, n_draws: int = 300, step: int = 2,
                    shortlist: int = 800, out_path=None) -> dict:
    """E4: build the Haas model for each weekend and print what moved.

    For every weekend: both `CarState`s (value, `n_evidence`, `shrink_weight`,
    `source` for all nine quantities), the shipped per-car plans out of
    `meta_<key>.json["per_driver"]`, and - with `plans=True` - the per-car
    search run twice, once with the new terms and once without, so the report
    can say whether the terms changed a recommendation.  Nothing is tuned here
    and nothing is written to `data/processed/`.
    """
    import json
    import time

    from src import strategy as strat

    out = {"weekends": {}, "moved": []}
    for key in keys:
        t0 = time.time()
        ev, m, cal, fit, model, clean, prac = _load_weekend(key, n_draws=n_draws)
        hm = HaasCarModel.from_weekend(ev, model, clean, calibration=cal, practice_laps=prac)
        shipped = _plan_rows(m.get("per_driver"), list(hm.states))
        print(f"\n=== {key} ===  {hm.source}")
        for drv, st in hm.states.items():
            print(f"  {drv} ({st.team}, team-mate {st.teammate}), evidence {st.evidence}")
            for name, est in st.estimates().items():
                v = est.value
                vs = ("-" if v is None else
                      ("{" + ", ".join(f"{k}: {float(x):.4f}" for k, x in sorted(v.items())
                                       if x is not None) + "}") if isinstance(v, dict) else
                      f"{float(v):+.4f}")
                own = ("-" if est.own is None else
                       ("{" + ", ".join(f"{k}: {float(x):.4f}" for k, x in sorted(est.own.items())
                                        if x is not None) + "}") if isinstance(est.own, dict) else
                       f"{float(est.own):+.4f}")
                print(f"     {name:22s} {vs:60s} n={est.n_evidence:3d} w={est.shrink_weight:.2f} "
                      f"{est.source:11s} own={own}")
            print(f"     race_factor (prior)   {st.race_factor}")
        rec = {"source": hm.source, "states": {d: s.as_dict() for d, s in hm.states.items()},
               "shipped_plans": _json_clean(shipped), "moved": _moved(hm)}
        print(f"  off the team/field value: {rec['moved'] or 'nothing'}")

        if plans:
            kw = _sim_kwargs(key, m, cal, fit)
            drivers = list(hm.states)
            terms = hm.terms()
            base = strat.per_driver_plans(
                model, ev, float(m["pit_loss_s"]), drivers,
                race_factors={d: terms[d]["race_factor"] for d in drivers},
                dev_by_driver={d: terms[d]["dev_override"] for d in drivers},
                factor_shrink=None, step=step, shortlist=shortlist, **kw)
            withx = strat.per_driver_plans(
                model, ev, float(m["pit_loss_s"]), drivers,
                race_factors={d: terms[d]["race_factor"] for d in drivers},
                dev_by_driver={d: terms[d]["dev_override"] for d in drivers},
                factor_shrink=None, step=step, shortlist=shortlist,
                warmup_by_driver={d: terms[d]["warmup_s"] for d in drivers},
                traffic_mult_by_driver={d: terms[d]["traffic_mult"] for d in drivers}, **kw)
            cols = ["driver", "best", "first_stop", "same_shape_as_field", "push"]
            print("  per-car plans, no car terms:")
            print("   " + base[cols].to_string(index=False).replace("\n", "\n   ") if not base.empty else "   (none)")
            print("  per-car plans, with warm-up and traffic terms:")
            print("   " + withx[cols].to_string(index=False).replace("\n", "\n   ") if not withx.empty else "   (none)")
            pm = {str(r["driver"]): r for r in withx.to_dict("records")} if not withx.empty else {}
            expl = hm.explain(pm)
            print("  explain_difference:")
            for line in expl:
                print(f"   - {line}")
            rec["plans_no_terms"] = _json_clean(base.to_dict("records")) if not base.empty else []
            rec["plans_with_terms"] = _json_clean(withx.to_dict("records")) if not withx.empty else []
            rec["explanation"] = expl
        rec["seconds"] = round(time.time() - t0, 1)
        out["weekends"][key] = rec
        print(f"  ({rec['seconds']}s)")
    if out_path:
        from pathlib import Path
        Path(out_path).write_text(json.dumps(out, indent=1))
        print(f"\nwritten {out_path}")
    return out


def _moved(hm: "HaasCarModel") -> list:
    """Which estimates actually left the team/field value, and by how much."""
    moved = []
    for drv, st in hm.states.items():
        for name, est in st.estimates().items():
            if est.value is None or est.team_value is None or est.shrink_weight <= 0:
                continue
            if isinstance(est.value, dict):
                for c in sorted(est.value):
                    tv = (est.team_value or {}).get(c)
                    if tv is None or est.value[c] is None:
                        continue
                    d = float(est.value[c]) - float(tv)
                    if abs(d) > 1e-6:
                        moved.append(f"{drv}.{name}[{c}] {d:+.5f} (n={est.n_evidence}, w={est.shrink_weight:.2f})")
            else:
                d = float(est.value) - float(est.team_value)
                if abs(d) > 1e-6:
                    moved.append(f"{drv}.{name} {d:+.4f} (n={est.n_evidence}, w={est.shrink_weight:.2f})")
    return moved


def main(argv=None) -> None:
    import argparse

    from src.config import DATA_PROCESSED

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--events", nargs="+", default=["australia-2026", "japan-2026", "barcelona-2026",
                                                   "austria-2026", "belgium-2026", "hungary-2026",
                                                   "italy-2026"])
    p.add_argument("--no-plans", action="store_true", help="states only; skip the per-car searches")
    p.add_argument("--draws", type=int, default=300)
    p.add_argument("--json", default=None, help="write the report to this path")
    a = p.parse_args(argv)
    keys = [k for k in a.events if (DATA_PROCESSED / f"meta_{k}.json").exists()]
    missing = [k for k in a.events if k not in keys]
    if missing:
        print(f"no meta on disk for {missing}; skipped")
    evidence_report(keys, plans=not a.no_plans, n_draws=a.draws, out_path=a.json)


if __name__ == "__main__":
    main()
