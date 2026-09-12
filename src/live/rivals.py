"""Live competitor selection, the rejoin projection and the pit response.

Task 1 answered "who are we racing?" with the four cars nearest in *virtual*
race position within 6 s (`racestate.relevant_rivals`) and "where do we rejoin?"
with today's gaps pushed back by the pit loss (`racestate.rejoin_traffic`).
Both are static reads of the timing screen, and both miss things a pit wall
does not:

* the car **directly ahead or behind on track** is the one place that can change
  on this lap, whatever its virtual gap says.  A car 8 s behind on the road is
  outside the 6 s virtual band, but it is the car that takes our place if we
  lose four seconds in the pit lane;
* a car **a lap down** cannot exchange a place with us, and neither can one
  whose remaining stops put it a whole pit cycle away.  Task 1 excluded the
  first only as a side effect (a lapped car usually has no numeric gap to the
  leader on the feed, and the term skipped cars more than a lap from ours) and
  the second not at all: a car twenty seconds behind that had stopped once more
  than us sat 2 s away in virtual position whether or not it still owed a stop
  of its own;
* on rejoin, the cars ahead are **not where they are now**.  In four laps a car
  half a second a lap slower has dropped two seconds, and a car that boxes on
  the lap we box is 22 s further back.  Projecting each car's race time forward
  on its own stint pace and the stop the engine gave it last lap is the same
  arithmetic the position term already does against our rivals - it was simply
  not being done for the traffic.

Everything here is a pure function of the `racestate.CarView` table the engine
builds each tick, so it can be unit-tested without a feed, and every number it
reports comes from the state (a gap, a pace, a stop lap, a pit loss, a
measured sigma) rather than from a rule of thumb.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, ndtr

from src import racestate
from src.racestate import CHOICE_TEMPER_S, DENSITY_MEAN, RELEVANT_RANGE_S, TRAFFIC_BAND_S, CarView

# A car we can physically lose the place to on this lap is at least as
# strategically relevant as one a full pit-cycle sigma away in virtual gap, so
# its selection score is capped here.  Engineering constant, and it orders the
# set rather than choosing it: it can only change *which* cars are picked when
# more than `k` cars survive the filters and an on-track neighbour is further
# than this many sigma away in virtual gap.  On both replays the surviving set
# averages 2.5 cars against k = 4, so the cap rarely binds at all.
ON_TRACK_SCORE = 1.0

# Beyond this many pit-cycle sigmas the order after the cycle is not in doubt
# (Phi(3) = 0.9987), so the rival's term is a constant in our stop lap and no
# place can change hands: with `sigma_rel` = 2.7 s and the 6 s virtual band
# that is 14 s of race time.  Statistical, not fitted.  Measured effect
# (`bench/bench_wpd_live.py --selection`): dropping the filter raises the mean
# rival set 2.5 -> 2.9 cars and costs 0.02 of the within-3-laps stop-call share
# at Hungary; it is what separates the V4 set from Task 1's.
INTERACTION_BAND_Z = 3.0

# Two cars on the same racing lap can differ by one completed lap while one of
# them has not yet crossed the line, so "a lap down" needs a two-lap gap in
# completed laps unless the feed says so in words ("1 L").
LAP_DOWN_TOL = 2

# A car more than this per lap off the field's median is showing a lap that is
# not its stint pace (traffic, a lock-up, damage, a lap the filters did not
# catch); the projection clips the deviation to it.  Physical: the spread of
# green racing laps between the fastest and slowest car in the 2026 races is
# under 2 s a lap, and over the four-lap horizon this bounds one car's drift at
# 8 s, a third of a pit loss.  Sensitivity (clip 1 / 2 / 4 / no clip on both
# replays): the mean rejoin-position error moves inside 0.84-0.97 places at
# Hungary and 1.33-1.41 at Barcelona, the median error, the bias and the
# stop-call metrics do not move at all, and 4 s is already the no-clip answer -
# the clip is a guard against a lap the filters missed, not a tuning knob.
PACE_DEV_CLIP_S = 2.0

# The projection needs a pace for our car and for most of the field; below this
# share of the cars carrying a pace estimate it falls back to the gap-based
# `racestate.rejoin_traffic`.  Engineering constant: with fewer than half the
# cars projected the rejoin position is a count of a different field.
PACE_COVERAGE_MIN = 0.5


# --------------------------------------------------------------------------
# Competitor selection
# --------------------------------------------------------------------------


@dataclass
class Rival:
    """One selected rival, with the reason it is in the set."""

    number: str
    code: str
    gap_s: float                   # race-time gap now, rival ahead positive
    virtual_gap_s: float           # ... corrected by a pit loss per stop of difference
    cycle_gap_s: float             # ... projected through the stops both still owe
    score: float                   # |virtual gap| / sigma_rel, capped for on-track neighbours
    on_track: str | None           # "ahead" | "behind" on the road, else None
    why: str
    tags: tuple = ()

    def as_triple(self) -> tuple:
        """The `(number, gap, virtual gap)` triple `racestate.live_position_term` takes."""
        return (self.number, self.gap_s, self.virtual_gap_s)


def _is_lapped(me: CarView, c: CarView, row: dict | None) -> bool:
    """Is this car a lap or more away from us?

    The feed says so in words on the gap to the leader ("1 L"), and
    `parse_gap` returns None for it, so a lapped car usually arrives here with
    no race-time gap at all.  The completed-lap count is the back-stop for the
    laps where the text has not caught up.
    """
    txt = str((row or {}).get("gap_leader") or "")
    if "L" in txt.upper() and any(ch.isdigit() for ch in txt):
        return True
    return abs(int(c.cur_lap) - int(me.cur_lap)) >= LAP_DOWN_TOL


def _pending(c: CarView) -> bool:
    """Does this car still owe a stop, by its own plan?

    Its own cost curve answers it: if running to the end on this set is no
    dearer than its best remaining stop, the car is done.  (The position term
    keeps its own, relative, `pending` flag - a car that has stopped more often
    than us is not in *our* round - and this is the car's own state, which is
    what the selection and the projection need.)
    """
    if c.extra.get("pending_own") is not None:
        return bool(c.extra["pending_own"])
    best_stop = float(np.min(c.curve_cost)) if len(c.curve_cost) else float("inf")
    done = np.isfinite(c.stay_cost) and c.stay_cost <= best_stop
    return not bool(done)


def _why(gap: float, vgap: float, cycle: float, stops_diff: int, on_track: str | None,
         pending: bool, in_pit: bool) -> str:
    """The reason this car is in the set, as numbers from the state."""
    parts = []
    if on_track:
        parts.append(f"{abs(gap):.1f} s {on_track} on track")
        if abs(vgap - gap) >= 0.5:
            parts.append(f"{abs(vgap):.1f} s {'ahead' if vgap > 0 else 'behind'} virtually")
    else:
        parts.append(f"{abs(vgap):.1f} s {'ahead' if vgap > 0 else 'behind'} virtually")
    if stops_diff:
        n = abs(int(stops_diff))
        parts.append(f"{'one' if n == 1 else str(n)} stop{'' if n == 1 else 's'} "
                     f"{'more' if stops_diff > 0 else 'fewer'}")
    if not pending:
        parts.append("no stop left")
    elif in_pit:
        parts.append("in the pit lane")
    if abs(cycle - vgap) >= 1.0:
        parts.append(f"{abs(cycle):.1f} s {'ahead' if cycle > 0 else 'behind'} after the cycle")
    return ", ".join(parts)


def strategic_rivals(me: CarView, cars: dict, order_rows: list | None, pit_loss_s: float,
                     sigma_rel_s: float, *, k: int = racestate.N_RIVALS,
                     rng_s: float = RELEVANT_RANGE_S) -> list:
    """The cars that can exchange a place with us through the next pit cycle.

    The candidates are the cars within `rng_s` of us in virtual race position
    (the gap corrected by a pit loss per stop of difference, exactly as
    `racestate.relevant_rivals`) **plus** the car directly ahead and the car
    directly behind on the road, whatever their gap.  Removed from that set:

    * cars a lap down (`_is_lapped`) - no place to exchange;
    * cars whose remaining stops put the order beyond doubt: with `pend` the
      stops each car still owes in this round, the gap once both have taken
      them is `gap + pit_loss (pend_me - pend_rival)`, and if neither that nor
      the gap now is inside `rng_s + INTERACTION_BAND_Z sigma` (and the two
      have the same sign, so no crossing) the rival's term cannot change with
      our stop lap.

    What is left is ranked by |virtual gap| / sigma_rel - strategic distance in
    units of the noise of two pit cycles - with an on-track neighbour's score
    capped at `ON_TRACK_SCORE`, then by closeness on the road, and cut at `k`.
    Every rival carries `why`.
    """
    if me.gap_leader_s is None or not np.isfinite(me.gap_leader_s) or k <= 0:
        return []
    rows = {r.get("driver_number"): r for r in (order_rows or [])}
    my_pos = me.position if me.position is not None else (rows.get(me.number) or {}).get("position")
    band = float(rng_s) + INTERACTION_BAND_Z * float(max(sigma_rel_s, 1e-6))
    pend_me = 1 if _pending(me) else 0
    out = []
    for num, c in cars.items():
        if num == me.number or c.gap_leader_s is None or not np.isfinite(c.gap_leader_s):
            continue
        row = rows.get(num)
        if _is_lapped(me, c, row):
            continue
        gap = float(me.gap_leader_s - c.gap_leader_s)
        vgap = gap + float(pit_loss_s) * (c.stops - me.stops)
        pend_r = 1 if _pending(c) else 0
        cycle = gap + float(pit_loss_s) * (pend_me - pend_r)
        on_track = None
        pos = c.position if c.position is not None else (row or {}).get("position")
        if my_pos is not None and pos is not None:
            if int(pos) == int(my_pos) - 1:
                on_track = "ahead"
            elif int(pos) == int(my_pos) + 1:
                on_track = "behind"
        in_band = abs(vgap) <= rng_s
        if not (in_band or on_track):
            continue
        # can a place still change hands through the cycle?
        reachable = (abs(gap) <= band or abs(cycle) <= band
                     or np.sign(gap) != np.sign(cycle))
        if not reachable:
            continue
        score = abs(vgap) / float(max(sigma_rel_s, 1e-6))
        if on_track:
            score = min(score, ON_TRACK_SCORE)
        tags = tuple(t for t in (("on_track_" + on_track) if on_track else None,
                                 "virtual_band" if in_band else None,
                                 None if pend_r else "no_stop_left") if t)
        out.append((score, abs(gap), Rival(
            number=num, code=c.code or num, gap_s=gap, virtual_gap_s=vgap, cycle_gap_s=cycle,
            score=score, on_track=on_track,
            why=_why(gap, vgap, cycle, int(c.stops - me.stops), on_track, bool(pend_r),
                     bool(c.in_pit)),
            tags=tags)))
    out.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in out[:int(k)]]


# --------------------------------------------------------------------------
# The rejoin projection
# --------------------------------------------------------------------------


def pace_reference(me: CarView, cars: dict) -> tuple:
    """`(reference pace, coverage)`: the field's median stint pace and the share
    of cars that have one.

    Only *differences* from this reference are used, so whatever common terms
    the pace carries (fuel correction, track evolution) cancel between two cars
    projected to the same lap.
    """
    vals, n = [], 0
    for num, c in cars.items():
        n += 1
        p = c.extra.get("pace_s")
        if p is not None and np.isfinite(p):
            vals.append(float(p))
    if not vals:
        return None, 0.0
    return float(np.median(vals)), (len(vals) / max(n, 1))


@dataclass
class FieldProjection:
    """Every car's projected race-time line, as arrays, so one lap costs one pass.

    Built once per car per tick (`field_projection`) and evaluated at each
    candidate rejoin lap: the projection is the same field arithmetic for every
    lap, and re-deriving it per lap in Python was a fifth of the tick.
    """

    numbers: list
    codes: list
    g0: np.ndarray                 # gap to the leader now
    rel: np.ndarray                # its stint pace less the field's, clipped
    cur: np.ndarray                # laps completed
    stop: np.ndarray               # the stop the engine gave it last lap (-1: none pending)
    in_pit: np.ndarray
    me_g0: float
    me_rel: float
    me_cur: int
    reference: float
    coverage: float


def field_projection(me: CarView, cars: dict, *, reference: float | None = None) -> FieldProjection | None:
    """The arrays `project_rejoin` needs, or None when there is not enough pace.

    A car with no pace estimate of its own is carried at the field's median (it
    is the least-informative choice that keeps it in the count); below
    `PACE_COVERAGE_MIN` of the field, or with no pace for us, there is no
    projection at all and the caller keeps Task 1's gaps."""
    if me.gap_leader_s is None or not np.isfinite(me.gap_leader_s):
        return None
    ref, coverage = (reference, 1.0) if reference is not None else pace_reference(me, cars)
    my_pace = me.extra.get("pace_s")
    if ref is None or my_pace is None or not np.isfinite(my_pace) or coverage < PACE_COVERAGE_MIN:
        return None
    nums, codes, g0, pace, cur, stop, pit = [], [], [], [], [], [], []
    for num, c in cars.items():
        if num == me.number or c.gap_leader_s is None or not np.isfinite(c.gap_leader_s):
            continue
        p = c.extra.get("pace_s")
        ns = c.extra.get("next_stop")
        nums.append(num)
        codes.append(c.code or num)
        g0.append(float(c.gap_leader_s))
        pace.append(float(p) if (p is not None and np.isfinite(p)) else ref)
        cur.append(int(c.cur_lap))
        stop.append(int(ns) if (ns is not None and _pending(c)) else -1)
        pit.append(bool(c.in_pit))
    clip = float(PACE_DEV_CLIP_S)
    return FieldProjection(
        numbers=nums, codes=codes,
        g0=np.asarray(g0, dtype=float),
        rel=np.clip(np.asarray(pace, dtype=float) - ref, -clip, clip),
        cur=np.asarray(cur, dtype=int), stop=np.asarray(stop, dtype=int),
        in_pit=np.asarray(pit, dtype=bool),
        me_g0=float(me.gap_leader_s),
        me_rel=float(min(clip, max(-clip, float(my_pace) - ref))),
        me_cur=int(me.cur_lap), reference=float(ref), coverage=float(coverage))


def project_rejoin(me: CarView, cars: dict, stop_lap: int, *, pit_loss_s: float,
                   pit_now_s: float | None = None, now_lap: int | None = None,
                   sigma_rel_s: float, dirty_air_s_per_lap: float, laps_per_stop: float,
                   band_s: float = TRAFFIC_BAND_S, reference: float | None = None,
                   proj: "FieldProjection | None" = None) -> dict | None:
    """Where we rejoin if we box on lap `stop_lap`, from every car's projected race time.

    Each car is carried forward to the end of lap `M = stop_lap + 1` - the lap
    both we and anyone stopping with us are out of the pit lane - on

        gap_to_leader(now) + (its pace - the field's) x laps to M + pit loss if
        it is projected to stop in between

    with its pace the median of its last three fuel-corrected clean laps (its
    last lap if it has fewer) and its stop the one the engine gave it on the
    previous lap.  Our own line adds the pit loss of the stop being priced
    (the safety-car factor when the stop is this lap).  The difference between
    two projected lines is how far ahead of us a car will be at the rejoin, so
    the expected rejoin position is `1 + sum Phi(lead / sigma_rel)` and a car
    `0-band_s` ahead is traffic, charged on V3's measured close-following cost
    exactly as `racestate.rejoin_traffic` does.

    Returns None when there is no pace for us or for at least
    `PACE_COVERAGE_MIN` of the field; the caller then keeps the gap-based
    fallback.  The projection deliberately carries no degradation drift over
    the two to four laps of the horizon: on this field that is 0.1-0.3 s, an
    order of magnitude under the pit loss it is being added to, and it would
    need a fresh-set model for the cars that stop inside the window.
    """
    fp = proj if proj is not None else field_projection(me, cars, reference=reference)
    if fp is None:
        return None
    M = int(stop_lap) + 1
    pit = float(pit_loss_s)
    pit_now = float(pit if pit_now_s is None else pit_now_s)
    now_lap = int(fp.me_cur + 1 if now_lap is None else now_lap)
    g_me = (fp.me_g0 + fp.me_rel * (M - fp.me_cur)
            + (pit_now if int(stop_lap) == now_lap else pit))
    if not len(fp.g0):
        return {"traffic_s": 0.0, "p_traffic": 0.0, "rejoin_position": 1, "rejoin_position_exp": 1.0,
                "gap_ahead_s": None, "ahead": None, "gap_behind_s": None, "behind": None,
                "band": [], "n_cars": 0, "stop_lap": int(stop_lap), "rejoin_lap": M,
                "source": "projected: no other car with a race-time gap"}
    stops_in = (fp.stop >= 0) & (fp.cur < fp.stop) & (fp.stop <= M)
    g = (fp.g0 + fp.rel * (M - fp.cur)
         + np.where(stops_in, np.where(fp.stop == now_lap, pit_now, pit), 0.0)
         # a car in the pit lane is taking its loss right now and the screen's
         # gap to the leader lags it by up to a lap
         + np.where(fp.in_pit & ~stops_in, pit, 0.0))
    x = g_me - g                                     # > 0: that car is ahead of us at the rejoin
    sig = float(max(sigma_rel_s, 1e-6))
    on_track = ~fp.in_pit
    p = (ndtr((float(band_s) - x) / sig) - ndtr((0.0 - x) / sig)) * on_track
    p_any = float(1.0 - np.prod(1.0 - np.clip(p, 0.0, 1.0)))
    ahead_exp = float(ndtr(x / sig).sum())
    ahead = np.flatnonzero(x > 0)
    behind = np.flatnonzero(x <= 0)
    j_a = int(ahead[np.argmin(x[ahead])]) if len(ahead) else None
    j_b = int(behind[np.argmax(x[behind])]) if len(behind) else None
    in_band = np.flatnonzero((x > 0) & (x <= float(band_s)) & on_track)
    band = [{"driver": fp.codes[i], "driver_number": fp.numbers[i], "gap_s": round(float(x[i]), 2),
             "p_in_band": round(float(p[i]), 3)} for i in in_band[np.argsort(x[in_band])]]
    return {"traffic_s": float(laps_per_stop * dirty_air_s_per_lap * p_any / DENSITY_MEAN),
            "p_traffic": p_any,
            "rejoin_position": int(round(ahead_exp)) + 1,
            "rejoin_position_exp": ahead_exp + 1.0,
            "gap_ahead_s": (round(float(x[j_a]), 2) if j_a is not None else None),
            "ahead": (fp.codes[j_a] if j_a is not None else None),
            "gap_behind_s": (round(float(-x[j_b]), 2) if j_b is not None else None),
            "behind": (fp.codes[j_b] if j_b is not None else None),
            "band": band, "n_cars": int(len(x)), "stop_lap": int(stop_lap), "rejoin_lap": M,
            "source": f"projected on {len(x) + 1} cars' stint pace and planned stops"}


def rejoin_or_fallback(me: CarView, cars: dict, stop_lap: int, *, pit_loss_s: float,
                       pit_now_s: float | None = None, now_lap: int | None = None,
                       sigma_rel_s: float, dirty_air_s_per_lap: float, laps_per_stop: float,
                       reference: float | None = None,
                       proj: "FieldProjection | None" = None) -> dict:
    """`project_rejoin`, falling back to `racestate.rejoin_traffic`'s gaps.

    The fallback is the Task 1 calculation, unchanged, so a car with no pace
    estimate (the first laps of a race, a car that has run nothing clean) is
    priced exactly as it was.
    """
    out = project_rejoin(me, cars, stop_lap, pit_loss_s=pit_loss_s, pit_now_s=pit_now_s,
                         now_lap=now_lap, sigma_rel_s=sigma_rel_s,
                         dirty_air_s_per_lap=dirty_air_s_per_lap, laps_per_stop=laps_per_stop,
                         reference=reference, proj=proj)
    if out is not None:
        return out
    now_lap = int(me.cur_lap + 1 if now_lap is None else now_lap)
    pit = float(pit_now_s if (pit_now_s is not None and int(stop_lap) == now_lap) else pit_loss_s)
    others = [c.gap_leader_s for num, c in cars.items() if num != me.number and not c.in_pit]
    rj = racestate.rejoin_traffic(me.gap_leader_s, others, pit,
                                  dirty_air_s_per_lap=dirty_air_s_per_lap,
                                  laps_per_stop=laps_per_stop, sigma_s=sigma_rel_s)
    return {**rj, "rejoin_position_exp": (float(rj["rejoin_position"])
                                          if rj.get("rejoin_position") is not None else None),
            "gap_ahead_s": None, "ahead": None, "gap_behind_s": None, "behind": None,
            "band": [], "n_cars": len(others), "stop_lap": int(stop_lap),
            "rejoin_lap": int(stop_lap) + 1,
            "source": "today's gaps (no stint pace yet)"}


# --------------------------------------------------------------------------
# The pit response: what a rival does if we box now
# --------------------------------------------------------------------------


def cover_response(detail: dict, place_value_s: float) -> dict | None:
    """One rival's response to our PIT NOW, from the cover terms of the position term.

    `racestate.live_position_term` reports, per rival, `rho` (the probability it
    boxes the lap after us instead of the lap it had planned, from
    `racestate.expected_ahead`: it covers when the place it would lose is worth
    more than moving its stop), the probability it is ahead afterwards if it
    covers and if it does not.  Turning that into the pit-wall answer:

        p_cover        the chance it reacts to our stop
        p_ahead_after  where it ends up if it does
        places_delta   the places that costs us, against it sticking to its plan
    """
    cov = (detail or {}).get("cover")
    if not cov:
        return None
    p_cover = float(cov.get("p_cover", float("nan")))
    p_cov_ahead = float(cov.get("p_ahead_if_cover", float("nan")))
    p_plan_ahead = float(cov.get("p_ahead_if_plan", float("nan")))
    if not np.isfinite(p_cover):
        return None
    delta = p_cov_ahead - p_plan_ahead
    return {"driver": detail.get("driver"), "driver_number": detail.get("driver_number"),
            "p_cover": p_cover, "p_ahead_after": p_cov_ahead, "p_ahead_if_not": p_plan_ahead,
            "places_delta": float(delta), "cost_s": float(place_value_s) * float(delta),
            "expected_cost_s": float(place_value_s) * float(delta) * p_cover}


def cover_summary(details: list, place_value_s: float) -> dict | None:
    """The field's pit response to our PIT NOW: the likeliest coverer and the total."""
    rows = [r for r in ((d.get("if_cover") or cover_response(d, place_value_s))
                        for d in (details or [])) if r]
    if not rows:
        return None
    top = max(rows, key=lambda r: (r["p_cover"], max(r["places_delta"], 0.0)))
    return {**top, "n_rivals": len(rows),
            "expected_places": round(float(sum(r["p_cover"] * r["places_delta"] for r in rows)), 6),
            "expected_s": round(float(sum(r["expected_cost_s"] for r in rows)), 4)}


def cover_probability(P: np.ndarray, q: np.ndarray, T_rival: np.ndarray, value_s: float,
                      cover_col: np.ndarray, later: np.ndarray,
                      temper: float = CHOICE_TEMPER_S) -> np.ndarray:
    """`rho[i, j]`: the chance the rival covers our option `i` instead of taking its `j`.

    The same expression as `racestate.expected_ahead` - which is the definition
    - kept here only so a caller that wants the probability itself does not
    have to re-price the round.  `tests/test_v4_live.py::
    test_cover_probability_matches_expected_ahead` asserts the two agree, so
    the two copies cannot drift apart silently.
    """
    T_rival = np.where(np.isfinite(T_rival), T_rival, 1e6)
    ok = cover_col >= 0
    col = np.where(ok, cover_col, 0)
    Pc = np.take_along_axis(P, col[None, :, None], axis=2)
    ben = float(value_s) * (Pc - P)
    cst = T_rival[col][:, None] - T_rival[None, :]
    return expit((ben - cst[None]) / float(temper)) * (later & ok[:, None])[None]
