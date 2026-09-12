"""Race state: when to stop, decided by the cars around you.

V3 timed the first stop on one car's cost surface - its tyre, the pit lane,
a generic undercut-exposure term and the circuit's historical first-stop
density - and that surface is flat for five laps either side of its minimum,
so the history prior ended up choosing the lap.  It chose late: the 2026 field
stopped 3-6 laps before the tool at every non-safety-car weekend, and before
any historical year at Barcelona and Austria.  Measured on the races
themselves, even a tyre model that knows the race's true degradation rates
puts the time-optimal first stop later than the field did within the families
the field ran.  The field is not optimising race time.  It is protecting and
taking track position, and nothing in V3 could see the cars that make that a
decision.

So the pit call here is made the way a pit wall makes it: for this car, from
where it is, against the three to five cars it is actually racing, comparing

    PIT NOW | STAY OUT 1 LAP | STAY OUT 2 LAPS | STAY OUT 3 LAPS | PIT AT THE EDGE OF THE WINDOW

and choosing the one with the best expected race outcome.  Every term is in
seconds of race time and every constant is measured on the 2026 races (never
the one being decided):

* **the tyre and the pit lane** - V3's cost tables, unchanged: what staying out
  k more laps costs on this set, what the rest of the race costs from each stop
  lap, the pit loss (x0.4 / x0.55 under a safety car / VSC this lap), the
  out-lap and the traffic the V3 density model charges on rejoin;
* **track position against each relevant rival** - for our stop lap `s` and
  its stop lap `l`, both cars' race time is priced lap by lap from their own
  tyres (stay-out on the set they are on, a fresh set after the stop) until
  both are out of the pits, and the rival is ahead afterwards with probability

        P = Phi((gap + D(s, l)) / sigma)

  `gap` is the race-time gap now (rival ahead positive), `D` our extra race
  time over those laps - which is where the undercut, the overcut, the
  compounds, the tyre ages, an undercut already in progress (the rival has
  pitted) and a safety-car stop all live - and `sigma` is the pit-cycle noise
  of two cars, measured from the spread of green-flag pit losses;
* **what a place is worth** - the median race-time interval between adjacent
  classified finishers (a place at the flag costs that much race time to win
  back),
  discounted by how often the order two adjacent cars leave a pit cycle in is
  still the order at the flag (`psi`, measured): a place won in the cycle is
  worth `V (2 psi - 1)`;
* **what the rival will do** - it chooses its own stop lap on its own cost
  curve, choosing among near-equal laps the way the pit wall treats them as
  equal (a logit at the engine's 1 s window tolerance), and it *covers* - boxes
  the lap after we do - when the place it would save is worth more than what
  the earlier stop costs it.

**Live**, the rivals are the four cars nearest in *virtual* race position (the
gap corrected for stops already made, so a car that pitted 20 s behind us is
two seconds ahead), with their real gaps, compounds, tyre ages, stops made,
pit status and the plan the engine gave them on the previous lap.  **Before
the race** there is no race state, so it is simulated: a pack of four rivals
at gaps drawn from the first-stint intervals the 2026 races actually showed,
all on the same plan family, each choosing its stop against the others.  The
first stop is the symmetric fixed point of that pack - every car's stop lap a
best response to every other car's - and the race-state term it produces is
what the plan search charges on the first stop, in place of V3's undercut
exposure and first-stop history prior.

History is kept for what it is good at: the plan-family prior still decides
which sequences are plausible, and the circuit's stint caps still bound the
edge of the window.  It no longer decides the lap.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr

from src.config import DATA_PROCESSED, get_event

log = logging.getLogger("degless.racestate")

# The scored 2026 weekends whose races the constants are measured on.  Every
# constant is measured leave-one-out: a weekend's own race never informs its
# own decision.
DONOR_EVENTS = ("australia-2026", "japan-2026", "barcelona-2026", "austria-2026",
                "belgium-2026", "hungary-2026", "italy-2026")

N_RIVALS = 4                 # the relevant rivals: the four nearest in virtual race position
RELEVANT_RANGE_S = 6.0       # ... within this much race time (the engine's undercut range)
PACK_LAPS = (5, 15)          # first-stint laps the pack intervals are measured on
PACK_POSITIONS = (2, 15)     # the pack, not the leader's clear air
CYCLE_PAIR_LAPS = 5          # two first stops this close are one pit cycle
GAP_QUANTILES = 8            # each pack rival's gap, as this many equal-mass quantiles
CHOICE_TEMPER_S = 1.0        # a rival chooses among laps within ~1 s as the pit wall does (WINDOW_TOL_S)
FIXED_POINT_ITERS = 80
FIXED_POINT_TOL = 1e-5
DAMPING = 0.5
LIVE_HORIZON_LAPS = 25       # a rival's stop distribution is carried this far ahead
TRAFFIC_BAND_S = 3.0         # a car this close ahead on rejoin is traffic (V3's density definition)
DENSITY_MEAN = 0.4488        # mean of V3's traffic-density quadratic (`strategy.traffic_density`)

ACTIONS = ("PIT NOW", "STAY OUT 1 LAP", "STAY OUT 2 LAPS", "STAY OUT 3 LAPS", "PIT AT EDGE OF WINDOW")

# Pooled over the seven 2026 races (measured 2026-09-13); used only when no
# donor race is on disk, and said so in `source`.
_FALLBACK = {"place_gap_s": 3.0, "persistence": 0.74, "cycle_sd_s": 1.9,
             "pack_gaps_s": (0.4, 0.7, 0.9, 1.1, 1.4, 1.8, 2.4, 3.4)}


# --------------------------------------------------------------------------
# The measured constants
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RaceStateConstants:
    """Everything the race-state term needs that is not the tyre model.

    `place_gap_s`   median race-time interval between adjacent classified
                    finishers, at the last lap both completed
    `persistence`   share of adjacent pairs whose order out of a pit cycle is the
                    order at the flag
    `cycle_sd_s`    robust SD of one car's green-flag pit loss about its race's
                    median - the in-lap, the stop and the out-lap together
    `pack_gaps_s`   the first-stint intervals between consecutive cars in the pack
    """

    place_gap_s: float
    persistence: float
    cycle_sd_s: float
    pack_gaps_s: tuple
    donors: tuple = ()
    n_finish_gaps: int = 0
    n_cycle_pairs: int = 0
    n_pit_stops: int = 0
    n_pack_gaps: int = 0
    place_gap_lead_lap_s: float = float("nan")      # diagnostic: lead-lap finishers only
    source: str = ""

    @property
    def place_value_s(self) -> float:
        """Seconds of race time a place won in a pit cycle is worth at the flag."""
        return float(self.place_gap_s * max(0.0, 2.0 * self.persistence - 1.0))

    @property
    def sigma_rel_s(self) -> float:
        """Noise on the race-time difference between two cars' pit cycles."""
        return float(math.sqrt(2.0) * self.cycle_sd_s)

    def as_dict(self) -> dict:
        return {"place_gap_s": round(self.place_gap_s, 3),
                "place_gap_lead_lap_s": (round(self.place_gap_lead_lap_s, 3)
                                         if np.isfinite(self.place_gap_lead_lap_s) else None),
                "persistence": round(self.persistence, 3),
                "place_value_s": round(self.place_value_s, 3), "cycle_sd_s": round(self.cycle_sd_s, 3),
                "sigma_rel_s": round(self.sigma_rel_s, 3),
                "pack_gap_median_s": round(float(np.median(self.pack_gaps_s)), 3),
                "pack_gap_p25_p75_s": [round(float(np.quantile(self.pack_gaps_s, q)), 3) for q in (0.25, 0.75)],
                "donors": list(self.donors), "n_finish_gaps": self.n_finish_gaps,
                "n_cycle_pairs": self.n_cycle_pairs, "n_pit_stops": self.n_pit_stops,
                "n_pack_gaps": self.n_pack_gaps, "n_rivals": N_RIVALS,
                "choice_temper_s": CHOICE_TEMPER_S, "source": self.source}


def order_table(race: pd.DataFrame) -> pd.DataFrame:
    """Race order at the end of every lap: cumulative time, position, intervals.

    A lap's end is its start plus its time, or the next lap's start where the
    time is missing (an in-lap or an out-lap FastF1 did not time)."""
    r = race.sort_values(["driver", "lap_number"]).copy()
    nxt = r.groupby("driver")["lap_start_s"].shift(-1)
    r["t_end"] = (r["lap_start_s"] + r["lap_time_s"]).fillna(nxt)
    out = []
    for _, g in r.groupby("lap_number"):
        g = g.dropna(subset=["t_end"]).sort_values("t_end")
        out.append(g.assign(pos=np.arange(1, len(g) + 1), gap_ahead=g["t_end"].diff(),
                            gap_behind=-g["t_end"].diff(-1)))
    return pd.concat(out, ignore_index=True) if out else r.iloc[0:0]


def _race_measurements(key: str) -> dict | None:
    """The four race-time quantities measured on one race's lap table."""
    from src.strategy import is_sc_status

    p = DATA_PROCESSED / f"laps_{key}_race.parquet"
    if not p.exists():
        return None
    ev = get_event(key)
    race = pd.read_parquet(p)
    o = order_table(race)
    if o.empty:
        return None
    green = o["track_status"].astype(str) == "1"
    pack = o[o["lap_number"].between(*PACK_LAPS) & o["pos"].between(*PACK_POSITIONS)
             & ~o["pit_in"] & ~o["pit_out"] & green]
    gaps = pack["gap_behind"].replace([np.inf, -np.inf], np.nan).dropna()
    gaps = gaps[gaps > 0].to_numpy(dtype=float)
    # the value of a place: the race-time interval between adjacent classified
    # finishers, measured at the last lap both completed so that lapped
    # midfield cars - the ones a place is usually contested among - count too.
    # (The lead-lap-only interval is kept as a diagnostic: it is a front-runner
    # sample, four or five cars at some races.)
    last = race.groupby("driver")["lap_number"].max()
    cls = set(last[last >= ev.n_race_laps - 2].index)
    tail = o.sort_values("lap_number").groupby("driver").tail(1)
    tail = tail[tail["driver"].isin(cls)].sort_values(["lap_number", "t_end"], ascending=[False, True])
    fpos = {d: i for i, d in enumerate(tail["driver"])}
    t_end = o.set_index(["driver", "lap_number"])["t_end"].to_dict()
    ds_fin = tail["driver"].tolist()
    fin = []
    for a, b in zip(ds_fin, ds_fin[1:]):
        lap = float(min(last[a], last[b]))
        ta, tb = t_end.get((a, lap)), t_end.get((b, lap))
        if ta is not None and tb is not None and np.isfinite(ta) and np.isfinite(tb) and tb > ta:
            fin.append(tb - ta)
    fin = np.asarray(fin, dtype=float)
    lead = o[o["lap_number"] == ev.n_race_laps].sort_values("t_end")
    fin_lead = lead["t_end"].diff().dropna().to_numpy(dtype=float)
    # persistence of the order a pit cycle produces
    pos = o.set_index(["driver", "lap_number"])["pos"].to_dict()
    status = o.set_index(["driver", "lap_number"])["track_status"].astype(str).to_dict()
    first = {}
    for d, g in race[race["pit_in"]].groupby("driver"):
        lap = int(g["lap_number"].min())
        if d in cls and not is_sc_status(status.get((d, float(lap)), "1")):
            first[d] = lap
    pairs = kept = 0
    ds = sorted(first)
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            la, lb = first[a], first[b]
            if abs(la - lb) > CYCLE_PAIR_LAPS:
                continue
            l0 = min(la, lb) - 1
            pa, pb = pos.get((a, float(l0))), pos.get((b, float(l0)))
            if pa is None or pb is None or abs(pa - pb) != 1:
                continue
            l1 = max(la, lb) + 2
            qa, qb = pos.get((a, float(l1))), pos.get((b, float(l1)))
            if qa is None or qb is None:
                continue
            pairs += 1
            kept += int((qa < qb) == (fpos[a] < fpos[b]))
    stops = np.zeros(0)
    pl = DATA_PROCESSED / f"pitloss_{key}.parquet"
    if pl.exists():
        x = pd.read_parquet(pl)["loss_s"].to_numpy(dtype=float)
        x = x[(x > 5) & (x < 60)]
        if len(x) >= 3:
            stops = x - np.median(x)
    return {"gaps": gaps, "fin": fin, "fin_lead": fin_lead, "pairs": pairs, "kept": kept, "stops": stops}


@lru_cache(maxsize=32)
def _measure(donors: tuple) -> RaceStateConstants:
    got = {k: _race_measurements(k) for k in donors}
    got = {k: v for k, v in got.items() if v is not None}
    if not got:
        return RaceStateConstants(**_FALLBACK, source="fallback: no donor race on disk "
                                  "(pooled 2026 values measured 2026-09-13)")
    gaps = np.concatenate([v["gaps"] for v in got.values()])
    fin = np.concatenate([v["fin"] for v in got.values()])
    fin_lead = np.concatenate([v["fin_lead"] for v in got.values()])
    stops = np.concatenate([v["stops"] for v in got.values()])
    pairs = sum(v["pairs"] for v in got.values())
    kept = sum(v["kept"] for v in got.values())
    place_gap = float(np.median(fin)) if len(fin) >= 5 else _FALLBACK["place_gap_s"]
    psi = float(kept / pairs) if pairs >= 10 else _FALLBACK["persistence"]
    sd = float(np.median(np.abs(stops)) * 1.4826) if len(stops) >= 10 else _FALLBACK["cycle_sd_s"]
    pack = tuple(float(x) for x in np.sort(gaps)) if len(gaps) >= 50 else _FALLBACK["pack_gaps_s"]
    return RaceStateConstants(place_gap_s=place_gap, persistence=psi, cycle_sd_s=sd, pack_gaps_s=pack,
                              donors=tuple(got), n_finish_gaps=int(len(fin)), n_cycle_pairs=int(pairs),
                              n_pit_stops=int(len(stops)), n_pack_gaps=int(len(gaps)),
                              place_gap_lead_lap_s=(float(np.median(fin_lead)) if len(fin_lead) else float("nan")),
                              source=f"measured on {len(got)} 2026 races: {', '.join(got)}")


def measure_constants(exclude: str | None = None, donors=DONOR_EVENTS) -> RaceStateConstants:
    """The race-state constants, measured on every donor race except `exclude`.

    Leave-one-out by construction: the decide stage for a scored weekend passes
    its own key, so its race never informs its own race-state term; a new
    weekend (no race yet) excludes nothing it could have used anyway."""
    return _measure(tuple(k for k in donors if k != exclude))


# --------------------------------------------------------------------------
# Rival behaviour: a stop-lap distribution and the cover response
# --------------------------------------------------------------------------


def softmin(cost: np.ndarray, temper: float = CHOICE_TEMPER_S) -> np.ndarray:
    """Choice probabilities over options by cost: a logit at `temper` seconds."""
    c = np.asarray(cost, dtype=float)
    ok = np.isfinite(c)
    if not ok.any():
        return np.full(len(c), 1.0 / max(len(c), 1))
    z = np.where(ok, np.exp(-(np.where(ok, c, 0.0) - c[ok].min()) / float(temper)), 0.0)
    return z / z.sum()


def expected_ahead(P: np.ndarray, q: np.ndarray, T_rival: np.ndarray | None, value_s: float,
                   cover_col: np.ndarray | None, later: np.ndarray | None,
                   temper: float = CHOICE_TEMPER_S) -> np.ndarray:
    """Expected P(rival ahead after the round) for each of our stop options.

    `P[g, i, j]` is the probability the rival is ahead once both are out of the
    pits if we stop on our option `i` and it on its option `j`, at gap scenario
    `g`.  `q[j]` is the rival's own choice over its options.  With `T_rival`
    (the rival's cost of each of its options), `cover_col[i]` (the rival option
    that boxes the lap after our option `i`, -1 if none) and `later[i, j]` (the
    rival's option `j` is after that cover lap), a rival that planned later
    covers us instead with probability

        rho = expit((V * (P_cover - P_planned) - (T[cover] - T[planned])) / temper)

    - it boxes early when the place it would otherwise lose is worth more than
    the stop it moves.  Returns shape (G, n_ours)."""
    if T_rival is None or cover_col is None or later is None or value_s <= 0:
        return np.einsum("gij,j->gi", P, q)
    # an option the rival cannot take carries a large finite cost rather than
    # inf: inf - inf in the cover comparison is NaN, and NaN x a zero choice
    # probability is still NaN
    T_rival = np.where(np.isfinite(T_rival), T_rival, 1e6)
    ok = cover_col >= 0
    col = np.where(ok, cover_col, 0)
    Pc = np.take_along_axis(P, col[None, :, None], axis=2)                  # (G, I, 1)
    ben = value_s * (Pc - P)                                                  # (G, I, J)
    cst = T_rival[col][:, None] - T_rival[None, :]                            # (I, J)
    rho = expit((ben - cst[None]) / float(temper)) * (later & ok[:, None])[None]
    return np.einsum("gij,j->gi", (1.0 - rho) * P + rho * Pc, q)


# --------------------------------------------------------------------------
# Before the race: the pack
# --------------------------------------------------------------------------


def pack_slots(const: RaceStateConstants, n_q: int = GAP_QUANTILES) -> tuple:
    """Gap scenarios for the four pack rivals, rival ahead positive.

    Two ahead and two behind; the nearer one at a pack interval, the further
    one at the sum of two, each as `n_q` equal-mass quantiles of the measured
    distribution.  Returns `(gaps (4 n_q,), slot index (4 n_q,))`."""
    g = np.asarray(const.pack_gaps_s, dtype=float)
    qs = (np.arange(n_q) + 0.5) / n_q
    one = np.quantile(g, qs)
    two = np.quantile((g[:, None] + g[None, ::max(1, len(g) // 200)]).ravel(), qs)
    gaps = np.concatenate([one, two, -one, -two])
    slot = np.repeat(np.arange(4), n_q)
    return gaps, slot


def pack_equilibrium(T: np.ndarray, laps: np.ndarray, stay_cum: np.ndarray, fresh_cum: np.ndarray,
                     const: RaceStateConstants, *, cover: bool = True,
                     temper: float = CHOICE_TEMPER_S, iters: int = FIXED_POINT_ITERS) -> dict:
    """The first stop of a car in a pack of four rivals running the same plan.

    `T[i]` is the plan family's race-time cost with its first stop on
    `laps[i]` (everything else in the plan at its best), `stay_cum[k]` the
    expected cost of the first `k` race laps on the start set and
    `fresh_cum[s, j]` that of `j` laps on the next set started after `s` laps
    (warm-up included), all at the family's push.  Every rival is a copy of the
    car: its stop lap is a logit choice on the same cost *including* the
    race-state term, so the pack is iterated to its symmetric fixed point
    (damped; it settles in 5-15 iterations).

    Returns the per-lap `tyre_s`, expected `places` (rivals ahead after the
    round, of four), `position_s = V * places`, `cost_s`, the first-stop
    `term_s` normalised to 0 at the chosen lap, the pack's stop distribution
    and the chosen `best_lap`."""
    laps = np.asarray(laps, dtype=int)
    T = np.asarray(T, dtype=float)
    ok = np.isfinite(T)
    laps, T = laps[ok], T[ok]
    S = len(laps)
    V = const.place_value_s
    if S == 0:
        return {}
    if S == 1 or V <= 0:
        # a place worth nothing (or one lap to choose from): the tyre decides
        q = softmin(T, temper)
        cq = np.cumsum(q)
        best = int(laps[int(np.argmin(T))])
        return {"laps": laps.tolist(), "tyre_s": T.tolist(), "places": [2.0] * S,
                "position_s": [2.0 * V] * S, "cost_s": T.tolist(), "term_s": [0.0] * S,
                "q": q.tolist(), "best_lap": best, "tyre_best_lap": best,
                "q_median": int(laps[min(int(np.searchsorted(cq, 0.5)), S - 1)]),
                "q_p25_p75": [int(laps[min(int(np.searchsorted(cq, 0.25)), S - 1)]),
                              int(laps[min(int(np.searchsorted(cq, 0.75)), S - 1)])],
                "iterations": 0, "converged": True, "place_value_s": float(V),
                "sigma_rel_s": const.sigma_rel_s}
    L_i, L_j = np.meshgrid(laps, laps, indexing="ij")
    M = np.maximum(L_i, L_j) + 1                                    # both out of the pits
    jmax = fresh_cum.shape[1] - 1
    A_me = stay_cum[L_i] + fresh_cum[L_i, np.clip(M - L_i, 0, jmax)]
    A_r = stay_cum[L_j] + fresh_cum[L_j, np.clip(M - L_j, 0, jmax)]
    D = A_me - A_r                                                  # our extra race time
    gaps, slot = pack_slots(const)
    P = ndtr((gaps[:, None, None] + D[None]) / const.sigma_rel_s)   # (G, S, S)
    w = np.full(len(gaps), 1.0 / GAP_QUANTILES)                    # per slot, quantiles equal mass
    cover_col = np.searchsorted(laps, laps + 1)
    cover_col = np.where((cover_col < S) & (laps[np.clip(cover_col, 0, S - 1)] == laps + 1), cover_col, -1)
    later = L_j > L_i + 1
    q = softmin(T, temper)
    places = np.zeros(S)
    converged, it = False, 0
    for it in range(1, iters + 1):
        pbar = expected_ahead(P, q, T + V * places if it > 1 else T, V,
                              cover_col if cover else None, later if cover else None, temper)
        places = (pbar * w[:, None]).sum(0)
        cost = T + V * places
        q_next = DAMPING * q + (1.0 - DAMPING) * softmin(cost, temper)
        if np.max(np.abs(q_next - q)) < FIXED_POINT_TOL:
            q = q_next
            converged = True
            break
        q = q_next
    cost = T + V * places
    i_best = int(np.argmin(cost))
    cq = np.cumsum(q)
    return {"laps": laps.tolist(), "tyre_s": T.tolist(), "places": places.tolist(),
            "position_s": (V * places).tolist(), "cost_s": cost.tolist(),
            "term_s": (V * (places - places[i_best])).tolist(), "q": q.tolist(),
            "best_lap": int(laps[i_best]), "tyre_best_lap": int(laps[int(np.argmin(T))]),
            "q_median": int(laps[min(int(np.searchsorted(cq, 0.5)), S - 1)]),
            "q_p25_p75": [int(laps[min(int(np.searchsorted(cq, 0.25)), S - 1)]),
                          int(laps[min(int(np.searchsorted(cq, 0.75)), S - 1)])],
            "iterations": int(it), "converged": bool(converged), "place_value_s": float(V),
            "sigma_rel_s": const.sigma_rel_s}


def term_by_lap(pack: dict | None, n_laps: int) -> np.ndarray | None:
    """A pack result's first-stop term as a (n_laps + 1,) array indexed by lap,
    held at its edge values outside the laps it was solved on."""
    if not pack or not pack.get("laps"):
        return None
    laps = np.asarray(pack["laps"], dtype=int)
    term = np.asarray(pack["term_s"], dtype=float)
    out = np.interp(np.arange(n_laps + 1), laps, term)
    return out


# --------------------------------------------------------------------------
# The decision itself: five actions on a race-state cost curve
# --------------------------------------------------------------------------


def action_table(laps, cost, parts: dict | None = None, *, now_lap: int, window_hi: int | None,
                 extra: dict | None = None) -> dict:
    """PIT NOW / STAY OUT 1-3 / PIT AT EDGE OF WINDOW on one cost curve.

    `laps`, `cost` are the candidate stop laps and their expected race-time
    cost (anything additive; only differences are reported), `parts` optional
    same-length components (tyre, position, traffic ...).  The edge is the last
    lap of the window if it lies beyond STAY OUT 3, else STAY OUT 3 itself.
    Returns the rows, the chosen action (lowest cost) and its lap."""
    laps = [int(x) for x in laps]
    idx = {l: i for i, l in enumerate(laps)}
    cost = np.asarray(cost, dtype=float)
    edge = int(window_hi) if (window_hi is not None and window_hi > now_lap + 3) else now_lap + 3
    targets = [now_lap, now_lap + 1, now_lap + 2, now_lap + 3, edge]
    rows, best = [], None
    for name, lap in zip(ACTIONS, targets):
        i = idx.get(lap)
        if i is None or not np.isfinite(cost[i]):
            rows.append({"action": name, "lap": lap, "legal": False})
            continue
        r = {"action": name, "lap": lap, "legal": True, "cost_s": float(cost[i])}
        for k, v in (parts or {}).items():
            r[k] = float(np.asarray(v, dtype=float)[i])
        for k, v in (extra or {}).items():
            r[k] = v[i] if isinstance(v, (list, np.ndarray)) else v
        rows.append(r)
        if best is None or r["cost_s"] < best["cost_s"]:
            best = r
    base = best["cost_s"] if best else 0.0
    for r in rows:
        if r.get("legal"):
            r["delta_s"] = float(r["cost_s"] - base)
    return {"actions": rows, "decision": (best["action"] if best else None),
            "decision_lap": (best["lap"] if best else None)}


# --------------------------------------------------------------------------
# Live: the nearest rivals, from the feed
# --------------------------------------------------------------------------


@dataclass
class CarView:
    """One car as the race-state term sees it on this lap.

    `cont[k]` is the expected race-time cost of `k` more laps on the set it is
    on (k = 0..R); `fresh_rows[s, j]` the expected cost of `j` laps on the set
    it would fit if it stopped after lap `s` (warm-up included); `curve_*` its
    own expected cost by next-stop lap (for its stop distribution and its cover
    decision) and `stay_cost` the no-further-stop option's."""

    number: str
    code: str
    cur_lap: int                         # laps completed
    gap_leader_s: float | None
    position: int | None
    stops: int
    in_pit: bool
    compound: str | None
    tyre_age: float | None
    cont: np.ndarray
    fresh_rows: np.ndarray | None
    next_compound: str | None
    curve_laps: np.ndarray
    curve_cost: np.ndarray
    stay_cost: float = float("nan")
    pending: bool = True                  # still has a stop to make in this round
    extra: dict = field(default_factory=dict)


def relevant_rivals(me: CarView, cars: dict, pit_loss_s: float, *, k: int = N_RIVALS,
                    rng_s: float = RELEVANT_RANGE_S) -> list:
    """The `k` cars nearest in *virtual* race position, within `rng_s`.

    Virtual: the race-time gap corrected by a pit loss per stop of difference,
    so a car that has pitted and rejoined behind is counted where it will be
    once we have stopped too.  Rivals are returned with `gap_s` (rival ahead
    positive) and `virtual_gap_s`."""
    if me.gap_leader_s is None or not np.isfinite(me.gap_leader_s):
        return []
    out = []
    for num, c in cars.items():
        if num == me.number or c.gap_leader_s is None or not np.isfinite(c.gap_leader_s):
            continue
        gap = float(me.gap_leader_s - c.gap_leader_s)
        v = gap + pit_loss_s * (c.stops - me.stops)
        if abs(v) <= rng_s:
            out.append((abs(v), num, gap, v))
    out.sort()
    return [(num, gap, v) for _, num, gap, v in out[:k]]


def live_position_term(me: CarView, rivals: list, cars: dict, S: np.ndarray, *, pit_now_s: float,
                       pit_s: float, now_lap: int, const: RaceStateConstants, cover: bool = True,
                       horizon: int = LIVE_HORIZON_LAPS) -> tuple:
    """Expected places lost to the relevant rivals, x V, for each of our next-stop laps `S`
    (and, in the last slot, for not stopping again).

    Returns `(position_s (len(S) + 1,), per-rival detail list)`."""
    V = const.place_value_s
    sig = const.sigma_rel_s
    S = np.asarray(S, dtype=int)
    n_opt = len(S) + 1
    total_me = me.cur_lap + len(me.cont) - 1
    places = np.zeros(n_opt)
    detail = []
    if V <= 0 or not rivals:
        return places, detail

    def a_car(c: CarView, stop_laps: np.ndarray, M: np.ndarray, pit_at_now: float) -> np.ndarray:
        """Race time from now to the end of lap M for car `c` stopping at `stop_laps`
        (-1 = no stop before M)."""
        R = len(c.cont) - 1
        tot = c.cur_lap + R
        Mc = np.clip(M, c.cur_lap, tot)
        stop = stop_laps >= 0
        s = np.where(stop, stop_laps, Mc)
        k_stay = np.clip(s - c.cur_lap, 0, R)
        out = c.cont[k_stay].astype(float)
        if stop.any() and c.fresh_rows is not None:
            tbl = c.fresh_rows
            j = np.clip(Mc - s, 0, tbl.shape[1] - 1)
            si = np.clip(s, 0, tbl.shape[0] - 1)
            out = out + np.where(stop, tbl[si, j] + np.where(s == now_lap, pit_at_now, pit_s), 0.0)
        return out

    for num, gap, vgap in rivals:
        r = cars[num]
        # the rival's options: stop at each lap of its horizon, or not within it
        if r.pending and len(r.curve_laps):
            lo = max(now_lap, r.cur_lap + 1)
            Lr = np.arange(lo, min(lo + horizon, total_me) + 1)
            cmap = dict(zip(r.curve_laps.tolist(), r.curve_cost.tolist()))
            Tr = np.array([cmap.get(int(l), np.inf) for l in Lr] + [r.stay_cost if np.isfinite(r.stay_cost)
                                                                     else np.inf], dtype=float)
            beyond = [float(c) for l, c in cmap.items() if l > Lr[-1]]
            if beyond:
                Tr[-1] = min(Tr[-1], min(beyond))
            if not np.isfinite(Tr).any():
                Lr, Tr = np.zeros(0, int), np.array([0.0])
        else:
            Lr, Tr = np.zeros(0, int), np.array([0.0])
        q = softmin(Tr)
        J = len(Lr) + 1
        # our options x its options: when is the round over, and what did each car run?
        ours = np.concatenate([S, [-1]])
        theirs = np.concatenate([Lr, [-1]])
        Oi, Tj = np.meshgrid(ours, theirs, indexing="ij")
        end = total_me
        M = np.where((Oi >= 0) & (Tj >= 0), np.maximum(Oi, Tj) + 1,
                     np.where(Oi >= 0, Oi + 1, np.where(Tj >= 0, Tj + 1, end)))
        M = np.minimum(M, end)
        A_me = a_car(me, Oi.ravel(), M.ravel(), pit_now_s).reshape(Oi.shape)
        A_r = a_car(r, Tj.ravel(), M.ravel(), pit_now_s).reshape(Oi.shape)
        # a rival whose stop lies beyond the horizon still owes this round's pit loss
        if r.pending:
            A_r = A_r + np.where(Tj < 0, pit_s, 0.0)
        D = A_me - A_r
        P = ndtr((gap + D[None]) / sig)                                        # (1, I, J)
        cover_col = None
        later = None
        if cover and len(Lr):
            pos_of = {int(l): j for j, l in enumerate(Lr)}
            cover_col = np.array([pos_of.get(int(s) + 1, -1) for s in S] + [-1])
            later = (Tj > (Oi + 1)) & (Tj >= 0) & (Oi >= 0)
        pa = expected_ahead(P, q, Tr, V, cover_col, later)[0]                  # (I,)
        places += pa
        detail.append({"driver": r.code or num, "driver_number": num, "gap_s": round(gap, 2),
                       "virtual_gap_s": round(vgap, 2), "compound": r.compound,
                       "tyre_age": (None if r.tyre_age is None else float(r.tyre_age)),
                       "stops": int(r.stops), "in_pit": bool(r.in_pit), "pending_stop": bool(r.pending),
                       "p_stop_next_3": float(q[:min(3, len(Lr))].sum()) if len(Lr) else 0.0,
                       "stop_lap_median": (int(Lr[min(int(np.searchsorted(np.cumsum(q[:-1]), 0.5)), len(Lr) - 1)])
                                           if len(Lr) and q[:-1].sum() > 0.5 else None),
                       "p_ahead": pa})
    return V * places, detail


def rejoin_traffic(me_gap_leader: float | None, others_gap_leader: list, pit_s: float, *,
                   dirty_air_s_per_lap: float, laps_per_stop: float, sigma_s: float) -> dict:
    """Traffic on rejoin from the gaps on the timing screen.

    After a stop the car is `pit_s` further back; every car that does not stop
    is where it is.  A car that would then be 0-3 s ahead is traffic (V3's
    density definition), so the chance of rejoining in traffic is
    `1 - prod(1 - P(car c in the band))` and the time it costs is V3's measured
    excess close-following laps per stop, scaled from the field's mean density
    to this rejoin: `laps_per_stop * dirty_air * P / 0.4488`.  Also returns the
    expected rejoin position among the cars counted."""
    if me_gap_leader is None or not np.isfinite(me_gap_leader):
        return {"traffic_s": float("nan"), "p_traffic": float("nan"), "rejoin_position": None}
    x = np.array([me_gap_leader + pit_s - g for g in others_gap_leader
                  if g is not None and np.isfinite(g)], dtype=float)      # each car's lead over us on rejoin
    if len(x) == 0:
        return {"traffic_s": 0.0, "p_traffic": 0.0, "rejoin_position": 1}
    p = ndtr((TRAFFIC_BAND_S - x) / sigma_s) - ndtr((0.0 - x) / sigma_s)
    p_any = float(1.0 - np.prod(1.0 - np.clip(p, 0.0, 1.0)))
    ahead = float(ndtr(x / sigma_s).sum())
    return {"traffic_s": float(laps_per_stop * dirty_air_s_per_lap * p_any / DENSITY_MEAN),
            "p_traffic": p_any, "rejoin_position": int(round(ahead)) + 1}
