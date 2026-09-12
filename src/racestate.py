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
each choosing its stop against the others.  The first stop is the fixed point
of that pack - every car's stop lap a best response to every other car's - and
the race-state term it produces is what the plan search charges on the first
stop, in place of V3's undercut exposure and first-stop history prior.

**V4-final: the pack is heterogeneous** (`rival_field`, the default whenever a
race state is passed).  Task 1's pack gave every rival *our* plan, and a field
of clones cannot produce the car that moves first: Barcelona's 2026 field
stopped at 11-15 because five of thirteen cars started on the SOFT and the
MEDIUM runners covered them, and a symmetric pack of MEDIUM two-stoppers has
nobody to cover.  So the pack is filled from a distribution over rival
**types** - a plan group (start compound, second compound, stop count) crossed
with a degradation level - whose family weights come from the model's own cost
of each group blended with how often this circuit's field has run it, and whose
stop laps come from their own cost curves blended with the circuit's historical
first-stop density.  The field is the mean-field fixed point over types (an
exact expectation over the type distribution, not a Monte Carlo draw), and our
own stop is the best response to it.  History enters as *the rivals'* plausible
behaviour and never as a term on our own lap.

**The value of a place** rests on the thinnest sample in the model - 25-35
adjacent pit-cycle pairs pooled over six races - so `psi` is estimated by
empirical-Bayes shrinkage of the per-race ratios (`estimator="regularized"`,
the production default) rather than as one pooled ratio, and the place value
carries a bootstrap standard error and interval.  `estimator="task1"`
reproduces the Task 1 measurement exactly; `bench/bench_place_value.py` audits
all four against each other and against the measurement windows.

History is kept for what it is good at: the plan-family prior still decides
which sequences are plausible, the rivals' behaviour is informed by what this
circuit's field has done, and the circuit's stint caps still bound the edge of
the window.  It no longer decides *our* lap.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr, ndtri

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
CYCLE_PAIR_ADJACENCY = 1     # ... and they were this many positions apart the lap before
GAP_QUANTILES = 8            # each pack rival's gap, as this many equal-mass quantiles
CHOICE_TEMPER_S = 1.0        # a rival chooses among laps within ~1 s as the pit wall does (WINDOW_TOL_S)
FIXED_POINT_ITERS = 80
FIXED_POINT_TOL = 1e-5
DAMPING = 0.5
LIVE_HORIZON_LAPS = 25       # a rival's stop distribution is carried this far ahead
TRAFFIC_BAND_S = 3.0         # a car this close ahead on rejoin is traffic (V3's density definition)
DENSITY_MEAN = 0.4488        # mean of V3's traffic-density quadratic (`strategy.traffic_density`)

# The place value's uncertainty (A2).  The donor *races* are the exchangeable
# unit - a race is one draw of a field, a track and a pit lane - so the
# bootstrap resamples races, not pairs.  2000 resamples put the Monte Carlo
# error on a 5/95 % quantile below a hundredth of a second, which is two orders
# below the 0.3 s the leave-one-out folds themselves move the place value by.
PLACE_BOOTSTRAP = 2000
PLACE_CI = (0.05, 0.95)
BOOTSTRAP_SEED = 4            # fixed so a fold's constants are reproducible
MIDFIELD_POSITIONS = (7, 16)  # the finishing band a place is usually contested in (diagnostic)
JEFFREYS_MEAN = 0.5           # Beta(1/2, 1/2): the reference prior for a binomial
JEFFREYS_WEIGHT = 1.0         # ... and its weight in pseudo-pairs (a + b = 1)
ESTIMATORS = ("task1", "lead_lap", "all_classified", "regularized")

ACTIONS = ("PIT NOW", "STAY OUT 1 LAP", "STAY OUT 2 LAPS", "STAY OUT 3 LAPS", "PIT AT EDGE OF WINDOW")

# Pooled over the seven 2026 races (measured 2026-09-13); used only when no
# donor race is on disk, and said so in `source`.
_FALLBACK = {"place_gap_s": 3.0, "persistence": 0.74, "cycle_sd_s": 1.9,
             "pack_gaps_s": (0.4, 0.7, 0.9, 1.1, 1.4, 1.8, 2.4, 3.4)}
_DONOR_ORDER = {k: i for i, k in enumerate(DONOR_EVENTS)}


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

    The V4 fields say *how well measured* the place value is, which matters
    because it rests on the model's thinnest sample: `estimator` names the
    treatment of that sample (see `measure_constants`), `persistence_raw` is the
    pooled ratio the shrinkage moved away from, `persistence_by_donor` and
    `n_cycle_pairs_by_donor` are the per-race counts behind it, and
    `place_value_sd_s` / `place_value_ci_s` are the bootstrap spread of
    `V (2 psi - 1)` over the donor races.  They describe the fit that produced
    the point values, so a `dataclasses.replace` that overrides `persistence` or
    `place_gap_s` (the ablation variants do) leaves them describing the fit, not
    the override.
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
    estimator: str = "task1"
    persistence_raw: float = float("nan")           # pooled kept / pairs, unshrunk
    persistence_by_donor: dict = field(default_factory=dict)    # key -> {pairs, kept}
    n_cycle_pairs_by_donor: dict = field(default_factory=dict)  # key -> pairs
    place_value_sd_s: float = float("nan")
    place_value_ci_s: tuple = ()                    # (5 %, 95 %) bootstrap over donor races
    place_gap_midfield_s: float = float("nan")      # diagnostic: finishers P7-P16

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
                "place_gap_midfield_s": (round(self.place_gap_midfield_s, 3)
                                         if np.isfinite(self.place_gap_midfield_s) else None),
                "persistence": round(self.persistence, 3),
                "persistence_raw": (round(self.persistence_raw, 3)
                                    if np.isfinite(self.persistence_raw) else None),
                "persistence_by_donor": {k: dict(v) for k, v in self.persistence_by_donor.items()},
                "n_cycle_pairs_by_donor": dict(self.n_cycle_pairs_by_donor),
                "place_value_s": round(self.place_value_s, 3),
                "place_value_sd_s": (round(self.place_value_sd_s, 3)
                                     if np.isfinite(self.place_value_sd_s) else None),
                "place_value_ci_s": ([round(float(x), 3) for x in self.place_value_ci_s]
                                     if len(self.place_value_ci_s) else None),
                "cycle_sd_s": round(self.cycle_sd_s, 3),
                "sigma_rel_s": round(self.sigma_rel_s, 3),
                "pack_gap_median_s": round(float(np.median(self.pack_gaps_s)), 3),
                "pack_gap_p25_p75_s": [round(float(np.quantile(self.pack_gaps_s, q)), 3) for q in (0.25, 0.75)],
                "donors": list(self.donors), "n_finish_gaps": self.n_finish_gaps,
                "n_cycle_pairs": self.n_cycle_pairs, "n_pit_stops": self.n_pit_stops,
                "n_pack_gaps": self.n_pack_gaps, "n_rivals": N_RIVALS,
                "estimator": self.estimator,
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


@lru_cache(maxsize=64)
def race_measurements(key: str, *, window: int = CYCLE_PAIR_LAPS,
                      adjacency: int = CYCLE_PAIR_ADJACENCY) -> dict | None:
    """The race-time quantities the constants are measured from, for one race.

    `window` and `adjacency` define a pit *cycle*: two first stops within
    `window` laps of each other, made by cars that were within `adjacency`
    positions the lap before the earlier one.  Both are reported settings rather
    than physical constants, so `bench/bench_place_value.py` varies them (3/5/8
    laps, 1 vs 2 positions) to show how much of `psi` they own.

    Returns the pack intervals, the three finishing-gap samples (all classified,
    lead lap only, the P7-P16 midfield band), the cycle pairs as individual
    outcomes (`pair_kept`, so a leave-one-pair-out influence can be computed),
    the green pit losses centred on the race's median, and whether the race's
    first-stop phase was safety-car affected."""
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
    # sample, four or five cars at some races.  The midfield band is the other
    # diagnostic: the gaps between the cars this tool actually decides for.)
    last = race.groupby("driver")["lap_number"].max()
    cls = set(last[last >= ev.n_race_laps - 2].index)
    tail = o.sort_values("lap_number").groupby("driver").tail(1)
    tail = tail[tail["driver"].isin(cls)].sort_values(["lap_number", "t_end"], ascending=[False, True])
    fpos = {d: i for i, d in enumerate(tail["driver"])}
    t_end = o.set_index(["driver", "lap_number"])["t_end"].to_dict()
    ds_fin = tail["driver"].tolist()
    fin, fin_mid = [], []
    lo_mid, hi_mid = MIDFIELD_POSITIONS
    for i, (a, b) in enumerate(zip(ds_fin, ds_fin[1:])):
        lap = float(min(last[a], last[b]))
        ta, tb = t_end.get((a, lap)), t_end.get((b, lap))
        if ta is not None and tb is not None and np.isfinite(ta) and np.isfinite(tb) and tb > ta:
            fin.append(tb - ta)
            if i + 1 >= lo_mid and i + 2 <= hi_mid:        # 1-based finishing positions
                fin_mid.append(tb - ta)
    fin = np.asarray(fin, dtype=float)
    lead = o[o["lap_number"] == ev.n_race_laps].sort_values("t_end")
    fin_lead = lead["t_end"].diff().dropna().to_numpy(dtype=float)
    # persistence of the order a pit cycle produces
    pos = o.set_index(["driver", "lap_number"])["pos"].to_dict()
    status = o.set_index(["driver", "lap_number"])["track_status"].astype(str).to_dict()
    first, first_any = {}, []
    for d, g in race[race["pit_in"]].groupby("driver"):
        lap = int(g["lap_number"].min())
        first_any.append(lap)
        if d in cls and not is_sc_status(status.get((d, float(lap)), "1")):
            first[d] = lap
    pair_kept = []
    ds = sorted(first)
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            la, lb = first[a], first[b]
            if abs(la - lb) > int(window):
                continue
            l0 = min(la, lb) - 1
            pa, pb = pos.get((a, float(l0))), pos.get((b, float(l0)))
            if pa is None or pb is None or not 1 <= abs(pa - pb) <= int(adjacency):
                continue
            l1 = max(la, lb) + 2
            qa, qb = pos.get((a, float(l1))), pos.get((b, float(l1)))
            if qa is None or qb is None:
                continue
            pair_kept.append(int((qa < qb) == (fpos[a] < fpos[b])))
    # A race whose first-stop phase ran under a safety car is a different
    # experiment: the stops are cheap and the order is reset behind the car.
    sc = False
    if first_any:
        span = o[o["lap_number"].between(min(first_any), max(first_any))]
        sc = bool(span["track_status"].astype(str).map(is_sc_status).any())
    stops = np.zeros(0)
    pl = DATA_PROCESSED / f"pitloss_{key}.parquet"
    if pl.exists():
        x = pd.read_parquet(pl)["loss_s"].to_numpy(dtype=float)
        x = x[(x > 5) & (x < 60)]
        if len(x) >= 3:
            stops = x - np.median(x)
    return {"gaps": gaps, "fin": fin, "fin_lead": fin_lead, "fin_mid": np.asarray(fin_mid, dtype=float),
            "pairs": len(pair_kept), "kept": int(sum(pair_kept)),
            "pair_kept": tuple(pair_kept), "stops": stops, "sc_affected": sc}


def persistence_eb(pairs, kept) -> dict:
    """Empirical-Bayes (beta-binomial) persistence from per-race pair counts.

    `psi` is the thinnest measurement in the model - 25-35 adjacent pit-cycle
    pairs pooled over six races, 3-9 of them per race - and the races are not
    replicates of one experiment: a street circuit's pit-cycle order survives to
    the flag more often than Spa's.  The pooled ratio ignores that
    heterogeneity (and lets the race with the most pairs set the number); the
    per-race ratios are far too noisy to use.  The standard treatment of that
    situation is beta-binomial shrinkage:

        the races' true rates ~ Beta(a, b) with mean `m` and variance `tau2`,
        estimated by method of moments - `m` the unweighted mean of the per-race
        ratios (a race is one draw), `tau2` the sample variance of those ratios
        less the average within-race binomial variance (DerSimonian-Laird);
        the prior's weight in pairs is `M = m (1 - m) / tau2 - 1`, and

            psi = (sum kept + m M) / (sum pairs + M)

    which lies between the pooled ratio and `m`, nearer `m` the thinner the
    pooled sample is relative to the between-race spread.

    `tau2 <= 0` means the races are consistent with one common rate and there is
    nothing to shrink toward but ignorance, so the **Jeffreys** prior
    Beta(1/2, 1/2) - the reference prior for a binomial, mean 1/2, weight 1
    pair - is used instead, pulling `psi` toward "a place in the cycle is worth
    nothing".  `M <= 0` (a between-race spread at or above the binomial ceiling)
    leaves the pooled ratio alone.

    Vectorised over leading axes (the last axis is the race axis) so the
    bootstrap can resample races in one call.  Returns `psi`, the pooled ratio
    it moved from, and the prior that moved it."""
    pairs = np.asarray(pairs, dtype=float)
    kept = np.asarray(kept, dtype=float)
    ok = pairs > 0
    n_ok = ok.sum(-1)
    p = np.where(ok, kept / np.where(ok, pairs, 1.0), 0.0)
    m = p.sum(-1) / np.maximum(n_ok, 1.0)
    s2 = (np.where(ok, (p - m[..., None]) ** 2, 0.0).sum(-1) / np.maximum(n_ok - 1.0, 1.0))
    within = (np.where(ok, p * (1.0 - p) / np.maximum(pairs - 1.0, 1.0), 0.0).sum(-1)
              / np.maximum(n_ok, 1.0))
    tau2 = s2 - within
    K = np.where(ok, kept, 0.0).sum(-1)
    N = np.where(ok, pairs, 0.0).sum(-1)
    ceiling = m * (1.0 - m)
    use_eb = (tau2 > 0.0) & (ceiling > 0.0) & (n_ok >= 2)
    M = np.where(use_eb, ceiling / np.where(tau2 > 0.0, tau2, 1.0) - 1.0, JEFFREYS_WEIGHT)
    M = np.maximum(M, 0.0)
    prior_mean = np.where(use_eb, m, JEFFREYS_MEAN)
    psi = (K + prior_mean * M) / np.maximum(N + M, 1e-9)
    raw = np.where(N > 0, K / np.maximum(N, 1e-9), np.nan)
    return {"psi": psi, "raw": raw, "prior_mean": prior_mean, "prior_pairs": M,
            "tau2": tau2, "between_race_mean": m, "n_races": n_ok,
            "method": np.where(use_eb, "beta-binomial MoM", "Jeffreys Beta(1/2,1/2)")}


def bootstrap_place_value(fins: list, pairs, kept, *, draws: int = PLACE_BOOTSTRAP,
                          seed: int = BOOTSTRAP_SEED, ci=PLACE_CI) -> dict:
    """Bootstrap the place value over the donor *races*.

    `fins[i]` is race `i`'s finishing-gap sample and `pairs[i]`/`kept[i]` its
    cycle-pair counts.  Races are resampled with replacement (a race is the
    exchangeable unit: one field, one track, one pit lane), `V` is recomputed as
    the median of the pooled gaps and `psi` by `persistence_eb` on the resampled
    counts, so the place value's spread carries both sources of error and their
    correlation.  Returns the sd and `ci` quantiles of `V`, `psi` and
    `V (2 psi - 1)`."""
    pairs = np.asarray(pairs, dtype=float)
    kept = np.asarray(kept, dtype=float)
    n = len(pairs)
    if n == 0 or not any(len(f) for f in fins):
        return {}
    width = max(1, max(len(f) for f in fins))
    padded = np.full((n, width), np.nan)
    for i, f in enumerate(fins):
        padded[i, :len(f)] = f
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(draws), n))
    with np.errstate(invalid="ignore"):
        V = np.nanmedian(padded[idx].reshape(int(draws), -1), axis=1)
    psi = persistence_eb(pairs[idx], kept[idx])["psi"]
    pv = V * np.maximum(2.0 * psi - 1.0, 0.0)

    def stat(x: np.ndarray) -> dict:
        x = x[np.isfinite(x)]
        if not len(x):
            return {"sd": float("nan"), "ci": ()}
        return {"sd": float(np.std(x, ddof=1)),
                "ci": tuple(float(v) for v in np.quantile(x, list(ci)))}

    return {"place_gap": stat(V), "persistence": stat(psi), "place_value": stat(pv),
            "draws": int(draws), "seed": int(seed), "ci_levels": tuple(ci)}


def _ordered(donors: Iterable) -> tuple:
    """Donor keys in the canonical `DONOR_EVENTS` order, unknown keys last.

    The measurement cache is keyed on the donor *set* and the estimator, so a
    fold is measured once however the caller ordered it; every pooled statistic
    below is order-invariant (a median, a sum, a sort) and only `source` and
    `donors` read the order, which is why it is canonicalised rather than
    ignored."""
    return tuple(sorted(set(donors), key=lambda k: (_DONOR_ORDER.get(k, len(DONOR_EVENTS)), str(k))))


@lru_cache(maxsize=64)
def _measure(donors: frozenset, estimator: str) -> RaceStateConstants:
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown place-value estimator {estimator!r}; expected one of {ESTIMATORS}")
    got = {k: race_measurements(k) for k in _ordered(donors)}
    got = {k: v for k, v in got.items() if v is not None}
    if not got:
        return RaceStateConstants(**_FALLBACK, estimator=estimator,
                                  source="fallback: no donor race on disk "
                                         "(pooled 2026 values measured 2026-09-13)")
    gaps = np.concatenate([v["gaps"] for v in got.values()])
    fin = np.concatenate([v["fin"] for v in got.values()])
    fin_lead = np.concatenate([v["fin_lead"] for v in got.values()])
    fin_mid = np.concatenate([v["fin_mid"] for v in got.values()])
    stops = np.concatenate([v["stops"] for v in got.values()])
    pairs = sum(v["pairs"] for v in got.values())
    kept = sum(v["kept"] for v in got.values())
    n_pairs = np.array([v["pairs"] for v in got.values()], dtype=float)
    n_kept = np.array([v["kept"] for v in got.values()], dtype=float)
    # The thin-sample guards are Task 1's, unchanged: below them the pooled
    # 2026 fallback is a better estimate than the donors on disk.
    place_gap = float(np.median(fin)) if len(fin) >= 5 else _FALLBACK["place_gap_s"]
    if estimator == "lead_lap":
        place_gap = float(np.median(fin_lead)) if len(fin_lead) >= 5 else _FALLBACK["place_gap_s"]
    raw = float(kept / pairs) if pairs >= 10 else _FALLBACK["persistence"]
    psi = raw
    eb: dict = {}
    boot: dict = {}
    if estimator == "regularized":
        eb = persistence_eb(n_pairs, n_kept)
        psi = float(eb["psi"]) if pairs >= 10 else _FALLBACK["persistence"]
        boot = bootstrap_place_value([v["fin"] for v in got.values()], n_pairs, n_kept)
    sd = float(np.median(np.abs(stops)) * 1.4826) if len(stops) >= 10 else _FALLBACK["cycle_sd_s"]
    pack = tuple(float(x) for x in np.sort(gaps)) if len(gaps) >= 50 else _FALLBACK["pack_gaps_s"]
    src = f"measured on {len(got)} 2026 races: {', '.join(got)}"
    if eb:
        src += (f"; psi shrunk from {raw:.3f} toward {float(eb['prior_mean']):.3f} "
                f"with {float(eb['prior_pairs']):.1f} prior pairs ({eb['method']})")
    return RaceStateConstants(place_gap_s=place_gap, persistence=psi, cycle_sd_s=sd, pack_gaps_s=pack,
                              donors=tuple(got), n_finish_gaps=int(len(fin)), n_cycle_pairs=int(pairs),
                              n_pit_stops=int(len(stops)), n_pack_gaps=int(len(gaps)),
                              place_gap_lead_lap_s=(float(np.median(fin_lead)) if len(fin_lead) else float("nan")),
                              place_gap_midfield_s=(float(np.median(fin_mid)) if len(fin_mid) else float("nan")),
                              estimator=estimator, persistence_raw=raw,
                              persistence_by_donor={k: {"pairs": int(v["pairs"]), "kept": int(v["kept"])}
                                                    for k, v in got.items()},
                              n_cycle_pairs_by_donor={k: int(v["pairs"]) for k, v in got.items()},
                              place_value_sd_s=float((boot.get("place_value") or {}).get("sd", float("nan"))),
                              place_value_ci_s=tuple((boot.get("place_value") or {}).get("ci", ())),
                              source=src)


def measure_constants(exclude: str | Iterable | None = None, donors=DONOR_EVENTS,
                      estimator: str = "regularized") -> RaceStateConstants:
    """The race-state constants, measured on every donor race except `exclude`.

    Leave-one-out by construction: the decide stage for a scored weekend passes
    its own key, so its race never informs its own race-state term; a new
    weekend (no race yet) excludes nothing it could have used anyway.  `exclude`
    takes a key or any iterable of keys, because the recalibration has to drop
    two - the weekend being scored *and* the donor whose objective is being
    searched - and a one-element set must give exactly what the bare key gives.

    `estimator` selects the treatment of the two samples the place value is
    built from (`bench/bench_place_value.py` reports all four side by side):

    * `task1` - every classified finisher's gap, `psi` the pooled ratio.  Task 1
      exactly, kept reproducible;
    * `all_classified` - the same estimator under its descriptive name;
    * `lead_lap` - `V` from lead-lap finishers only (Task 1's first
      implementation, a front-runner sample of four or five cars at some races);
    * `regularized` - **the V4 production default**: `V` as before, `psi` by
      empirical-Bayes shrinkage of the per-race ratios (`persistence_eb`), with
      a bootstrap standard error and 5-95 % interval on the place value.  It is
      chosen on statistical grounds - the standard treatment of a small binomial
      sample pooled over heterogeneous races - and not by any benchmark score.
    """
    ex = ({exclude} if isinstance(exclude, str) else set() if exclude is None else set(exclude))
    return _measure(frozenset(k for k in donors if k not in ex), str(estimator))


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
                   temper: float = CHOICE_TEMPER_S, *, rho_out: dict | None = None) -> np.ndarray:
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
    the stop it moves.  Returns shape (G, n_ours).

    `rho_out`, if given, receives `rho` itself under the key `"rho"` (None when
    there is no cover response), so the live engine can report the pit response
    per rival without re-pricing the round."""
    if T_rival is None or cover_col is None or later is None or value_s <= 0:
        if rho_out is not None:
            rho_out["rho"] = None
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
    if rho_out is not None:
        rho_out["rho"] = rho
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


# --------------------------------------------------------------------------
# Before the race: the heterogeneous rival field
# --------------------------------------------------------------------------


# The field logit's temperature.  Unlike `CHOICE_TEMPER_S` (the pit wall's own
# window tolerance, measured) this one says how sharply a *field* of nineteen
# cars sorts itself onto the cheapest plan family, and nothing in the data
# identifies it before WP-F calibrates it: 3 s is the engineering choice that
# leaves a family 3 s off the best with about a third of the best family's
# weight.  Sensitivity: at 1 s the field collapses onto one family (the
# symmetric pack again), at 10 s it is nearly uniform over families.
FAMILY_TEMPER_S = 3.0
# Pseudo-counts: a historical cell with `k0` stops in it gets half the weight.
# Same form and the same order as `firststop.MIN_COMPOUND_N` (5) and
# `PLAN_PRIOR_ALPHA`; 10 is deliberately more conservative than the first-stop
# prior's own back-off because this weight multiplies a *distribution* rather
# than backing one cell off to another.
HISTORY_WEIGHT_K0 = 10.0
FAMILY_PRIOR_WEIGHT_K0 = 10.0
# A rival type with less than this share of the modal type's weight puts under
# 0.04 of a car in the four pack slots, which moves the expected places by less
# than 0.01 of a place: it is dropped so the type x type tensor stays small.
FIELD_WEIGHT_FLOOR = 0.01
# Fallback spread of the rival field's degradation rate, in log units, when no
# calibration is on disk: 0.12 is the sd of the log per-team rate factors in
# `calibration.json` (0.12-0.14 across the leave-one-out folds, 11 teams).
RATE_LN_SD_FALLBACK = 0.12
# The expected-places curve is tabulated this finely and interpolated linearly;
# its second derivative is under 0.1 places/s^2, so the interpolation error is
# below 1e-5 places against a term measured in tenths of a second.
AHEAD_GRID_S = 0.02


@dataclass(frozen=True)
class RivalFieldConfig:
    """Who the four cars around us are, before the race has run.

    Task 1 filled the pack with copies of our own car on our own plan.  That
    cannot produce the car that moves first, and the car that moves first is
    what makes a field stop early: Barcelona 2026's field boxed at 11-15
    because five of thirteen cars started on the SOFT and the MEDIUM runners
    covered them.  So the pack is drawn from a distribution over rival
    **types** - a plan group (start compound, second compound, stop count)
    crossed with a degradation level - and solved as a mean-field equilibrium.

    * `mode` - `"hetero"` (the default) or `"symmetric"`, which is Task 1's
      `pack_equilibrium` unchanged.
    * `family_temper_s` - how sharply the field sorts onto the cheaper plan
      families (see `FAMILY_TEMPER_S`; WP-F calibrates it).
    * `use_history_prior` / `history_weight_k0` - the rivals' stop laps are
      blended toward the circuit's historical first-stop density at weight
      `n_cell / (n_cell + k0)`.  `False` sets that weight to zero, which is the
      ablation that answers "is the field's early stop the model's cost surface
      or the circuit's habit?".  It never touches *our* lap: history is the
      rivals' behaviour, never a term on our own decision.
    * `family_prior_weight_k0` - the same back-off for the plan-family mix.
      `inf` puts no weight on history at all, which with `use_history_prior`
      off is the "no historical strategy prior" ablation: the field's plan mix
      and its stop laps then come only from the model's own costs.
    * `rate_levels` / `rate_ln_sd` - the field's tyre-rate spread, `None`
      meaning the sd of the calibration's log per-team factors.
    """

    mode: str = "hetero"
    family_temper_s: float = FAMILY_TEMPER_S
    history_weight_k0: float = HISTORY_WEIGHT_K0
    use_history_prior: bool = True
    rate_levels: int = 3
    rate_ln_sd: float | None = None
    family_prior_weight_k0: float = FAMILY_PRIOR_WEIGHT_K0

    @property
    def symmetric(self) -> bool:
        return str(self.mode).lower() == "symmetric"

    def as_dict(self) -> dict:
        return {"mode": str(self.mode), "family_temper_s": float(self.family_temper_s),
                "history_weight_k0": float(self.history_weight_k0),
                "use_history_prior": bool(self.use_history_prior),
                "rate_levels": int(self.rate_levels),
                "rate_ln_sd": (None if self.rate_ln_sd is None else float(self.rate_ln_sd)),
                # None rather than `inf`, which is not JSON
                "family_prior_weight_k0": (float(self.family_prior_weight_k0)
                                           if np.isfinite(self.family_prior_weight_k0) else None)}


@lru_cache(maxsize=16)
def field_rate_ln_sd(event_key: str | None = None) -> float:
    """Spread of the rival field's degradation rate, in log units.

    Measured, not assumed: the calibration's per-team rate factors are what the
    previous races say about how differently the field wears its tyres, and
    their log sd is 0.12-0.14 across the leave-one-out folds.  Falls back to
    `RATE_LN_SD_FALLBACK` when no calibration is on disk."""
    try:
        from src.calibration import get_calibration

        tf = getattr(get_calibration(event_key), "team_factors", None) or {}
        x = np.log(np.array([float(v) for v in tf.values() if float(v) > 0]))
        if len(x) >= 3:
            return float(np.std(x, ddof=1))
    except Exception as exc:                      # a malformed calibration is not fatal
        log.debug("field rate spread unavailable (%s); using the fallback", exc)
    return float(RATE_LN_SD_FALLBACK)


def rate_nodes(n_levels: int, ln_sd: float) -> tuple:
    """Degradation levels for the rival field: rate factors and their weights.

    A rival is not our car, and the field's own rate spread is what makes two
    cars on the same plan stop four laps apart.  It is discretised as
    `n_levels` equal-mass quantile midpoints of a lognormal with log sd
    `ln_sd`, exactly as `pack_slots` discretises the gap distribution: three
    levels then sit at -0.97, 0 and +0.97 sd with weight 1/3 each (the plan's
    "-1, 0, +1 sd") and the middle level is our own model, factor 1.  An even
    number of levels has no node at 1 and the level nearest it then stands in
    for our own car, which is why the default is odd."""
    n = max(1, int(n_levels))
    z = ndtri((np.arange(n) + 0.5) / n)
    return np.exp(z * float(ln_sd)), np.full(n, 1.0 / n)


def family_weights(cost_s, neglogp, *, n_prior: int, cfg: RivalFieldConfig) -> np.ndarray:
    """How much of the field runs each plan group.

    `q_f(g) ~ exp(-C_g / family_temper_s) * p_hist(g) ** w_f`, with `C_g` the
    group's best tyre + pit lane + traffic + safety-car cost (no priors, so the
    field is not being told what to run by the same prior twice), `p_hist(g)`
    the circuit's smoothed frequency of that shape as `-log p` in `neglogp`
    (`strategy.plan_prior_penalty` at `tau = 1`), and
    `w_f = n / (n + family_prior_weight_k0)` the weight its `n` classified
    finishers earn it.  A circuit with no history of its own arrives here with
    the season pool the pipeline already substitutes, and `n_prior = 0` leaves
    the model's own costs to decide.

    Because `exp(-C/theta) p^w = exp(-(C + theta w (-log p))/theta)`, the blend
    is a logit on the cost plus `theta * w_f` seconds per nat of rarity - the
    same shape as the plan prior's own penalty, at the field's temperature."""
    cost = np.asarray(cost_s, dtype=float)
    nlp = np.zeros_like(cost) if neglogp is None else np.asarray(neglogp, dtype=float)
    n = max(0.0, float(n_prior or 0.0))
    w_f = n / (n + float(cfg.family_prior_weight_k0)) if n > 0 else 0.0
    theta = max(float(cfg.family_temper_s), 1e-6)
    return softmin(cost + theta * w_f * nlp, theta)


def history_weight(n_cell, cfg: RivalFieldConfig) -> float:
    """Weight the circuit's historical first-stop density earns for a rival's
    stop lap: `n_cell / (n_cell + k0)`, and zero with the prior switched off."""
    if not cfg.use_history_prior:
        return 0.0
    n = max(0.0, float(n_cell or 0.0))
    return float(n / (n + float(cfg.history_weight_k0))) if n > 0 else 0.0


def _ahead_curve(const: RaceStateConstants, lo: float, hi: float, *, n_q: int = GAP_QUANTILES,
                 step: float = AHEAD_GRID_S) -> tuple:
    """`F(x)`: expected rivals ahead, of the four pack slots, at race-time
    difference `x`, tabulated on a uniform grid over `[lo, hi]`.

    The same quantity `pack_equilibrium` forms as `(P * w).sum(0)` - the pack's
    four slots at `n_q` equal-mass gap quantiles each - but as a function of the
    one argument it depends on, so the heterogeneous field can look it up
    instead of evaluating the normal integral once per type pair."""
    gaps, _ = pack_slots(const, n_q)
    x = np.arange(float(lo), float(hi) + step, step)
    F = ndtr((gaps[:, None] + x[None, :]) / const.sigma_rel_s).sum(0) / n_q
    return F.astype(np.float32), float(lo), float(1.0 / step)


def _ahead_lookup(F: np.ndarray, lo: float, inv_step: float, x: np.ndarray) -> np.ndarray:
    """`F` at `x` by linear interpolation on its uniform grid (clipped at both
    ends, where `F` is flat at 0 and 4 anyway)."""
    u = np.clip((x - np.float32(lo)) * np.float32(inv_step), 0.0, len(F) - 1.001)
    i = u.astype(np.int32)
    f = u - i
    return F[i] * (1.0 - f) + F[i + 1] * f


def _softmin_rows(cost: np.ndarray, temper: float) -> np.ndarray:
    """`softmin` along the last axis, non-finite options excluded."""
    ok = np.isfinite(cost)
    base = np.where(ok, cost, np.inf).min(-1, keepdims=True)
    z = np.where(ok, np.exp(-(np.where(ok, cost, 0.0) - base) / float(temper)), 0.0)
    s = z.sum(-1, keepdims=True)
    return np.where(s > 0, z / np.where(s > 0, s, 1.0), 1.0 / cost.shape[-1])


def _pack_from_curve(laps: np.ndarray, T: np.ndarray, places: np.ndarray, q: np.ndarray,
                     const: RaceStateConstants, *, iterations: int, converged: bool) -> dict:
    """One car's first stop, read off its cost curve and the places it expects.

    The shape `scripts/10_pipeline.py::race_state_block`, `bench_ablation.py`
    and `term_by_lap` consume, identical for the symmetric pack and for a type
    in the heterogeneous field."""
    V = float(const.place_value_s)
    cost = T + V * places
    i_best = int(np.argmin(cost))
    S = len(laps)
    cq = np.cumsum(q)
    return {"laps": laps.tolist(), "tyre_s": T.tolist(), "places": places.tolist(),
            "position_s": (V * places).tolist(), "cost_s": cost.tolist(),
            "term_s": (V * (places - places[i_best])).tolist(), "q": q.tolist(),
            "best_lap": int(laps[i_best]), "tyre_best_lap": int(laps[int(np.argmin(T))]),
            "q_median": int(laps[min(int(np.searchsorted(cq, 0.5)), S - 1)]),
            "q_p25_p75": [int(laps[min(int(np.searchsorted(cq, 0.25)), S - 1)]),
                          int(laps[min(int(np.searchsorted(cq, 0.75)), S - 1)])],
            "iterations": int(iterations), "converged": bool(converged),
            "place_value_s": V, "sigma_rel_s": const.sigma_rel_s}


def rival_field(types: list, laps, const: RaceStateConstants, cfg: RivalFieldConfig, *,
                cover: bool = True, temper: float = CHOICE_TEMPER_S,
                iters: int = FIXED_POINT_ITERS) -> dict:
    """The mean-field equilibrium of a heterogeneous pack, and our best response.

    `types` is the rival field as a list of dicts, one per (plan group,
    degradation level); `laps` the shared candidate-lap grid they are all
    costed on.  Each type carries

        T          (S,) its race-time cost with its first stop on each lap,
                   `inf` where that lap is not legal for its group
        stay_cum   (L+1,) cost of the first k laps on its start compound
        fresh_cum  (n+1, L+1) cost of j laps on its next set, fitted after s laps
        weight     its share of the field (family weight x level weight)
        hist       (S,) `-log p` of each lap in the circuit's first-stop density
                   for its (start compound, stop count) cell, or None
        hist_w     the weight that density has earned (`history_weight`)
        ours       True for the one level that is our own car's rate

    Every type faces the *type-weighted field* on the four pack slots - the
    expectation over the field's types and their stop laps is taken exactly,
    not sampled - and chooses its own stop lap by a logit on its own cost plus
    the places it expects to lose, blended toward the circuit's history.  The
    map is iterated damped to its fixed point exactly as `pack_equilibrium`
    iterates the symmetric pack.  Our own stop is then the best response to the
    converged field: the same row, minimised on cost *without* the history term
    - a rival's habit is evidence about a rival, never a charge on our lap.

    Vectorised over the whole type x type x lap x lap tensor, which is built
    once: the field's tables do not change as the fixed point iterates, only
    the choice probabilities do, so each iteration is one contraction plus the
    cover logit.  Returns `{"packs": {family label: pack}, "types": [...],
    "iterations", "converged"}` with one pack per `ours` type, in the shape
    `pack_equilibrium` returns."""
    laps = np.asarray(laps, dtype=int)
    S, nT = len(laps), len(types)
    V = float(const.place_value_s)
    if not nT or not S:
        return {}
    T = np.stack([np.asarray(t["T"], dtype=float) for t in types])             # (nT, S)
    w = np.asarray([max(float(t["weight"]), 0.0) for t in types], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(nT, 1.0 / nT)
    # The rivals' history term, in seconds at their own logit temperature: a
    # logit on cost blended with `p_hist ** hist_w` is a logit on the cost plus
    # `temper * hist_w * (-log p_hist)`.  `use_history_prior` is honoured here
    # as well as where the weight is built, so the switch holds whatever a
    # caller put in the types.
    on = float(bool(cfg.use_history_prior))
    hist = np.stack([(np.zeros(S) if t.get("hist") is None else np.asarray(t["hist"], dtype=float))
                     * float(temper) * float(t.get("hist_w") or 0.0) * on for t in types])
    ours = [i for i, t in enumerate(types) if t.get("ours")]

    def packs_of(places: np.ndarray, q: np.ndarray, it: int, ok_fp: bool) -> dict:
        out = {}
        for i in ours:
            m = np.isfinite(T[i])
            if not m.any():
                continue
            qi = q[i][m]
            qi = qi / qi.sum() if qi.sum() > 0 else np.full(int(m.sum()), 1.0 / int(m.sum()))
            out[types[i]["label"]] = _pack_from_curve(laps[m], T[i][m], places[i][m], qi, const,
                                                     iterations=it, converged=ok_fp)
        return out

    q = _softmin_rows(T + hist, temper)
    if S == 1 or V <= 0:
        # a place worth nothing (or one lap to choose from): the tyre decides,
        # and every type expects the two cars ahead of it to stay there
        places = np.full((nT, S), 2.0)
        return {"packs": packs_of(places, q, 0, True), "iterations": 0, "converged": True,
                "types": _type_table(types, laps, q, w), "n_types": nT}

    # `B[x, i, j]`: type x's race time to the lap both cars are out of the pits,
    # for its stop on laps[i] against a rival stopping on laps[j].  Our extra
    # race time over the cycle is then `D[t, i, u, j] = B[t, i, j] - B[u, j, i]`.
    L_i, L_j = np.meshgrid(laps, laps, indexing="ij")
    M = np.maximum(L_i, L_j) + 1
    B = np.empty((nT, S, S), dtype=np.float32)
    for x, t in enumerate(types):
        stay = np.asarray(t["stay_cum"], dtype=float)
        fresh = np.asarray(t["fresh_cum"], dtype=float)
        jmax = fresh.shape[1] - 1
        B[x] = stay[np.clip(L_i, 0, len(stay) - 1)] + fresh[np.clip(L_i, 0, fresh.shape[0] - 1),
                                                            np.clip(M - L_i, 0, jmax)]
    Bs = np.ascontiguousarray(B.transpose(2, 0, 1))                 # Bs[i, u, j] = B[u, j, i]
    # Beyond the widest pack gap plus eight pit-cycle sigmas the normal
    # integral is 0 or 1 to machine precision, so the lookup's clip is exact
    # there and the table only has to span the band where it is not.
    span = float(np.max(np.abs(const.pack_gaps_s)) * 2.0 + 8.0 * const.sigma_rel_s)
    F, lo, inv = _ahead_curve(const, -span, span)

    cover_col = np.searchsorted(laps, laps + 1)
    cover_col = np.where((cover_col < S) & (laps[np.clip(cover_col, 0, S - 1)] == laps + 1), cover_col, -1)
    ok_cov = cover_col >= 0
    col = np.where(ok_cov, cover_col, 0)
    later = (L_j > L_i + 1) & ok_cov[:, None]                        # (S, S)

    # P[t, i, u, j], the places a cover would save the rival across the four
    # slots (`X`) and what the place is worth to *one* rival (`Xv`, seconds):
    # all fixed for the whole fixed point, because only the choice
    # probabilities move as it iterates.  `Xv` divides by the slots because the
    # cover is one car's decision about its own place, while `X` is the change
    # in the number of cars we expect to be ahead of.
    P = np.empty((nT, S, nT, S), dtype=np.float32)
    X = np.zeros((nT, S, nT, S), dtype=np.float32) if cover else None
    for t in range(nT):
        P[t] = _ahead_lookup(F, lo, inv, B[t][:, None, :] - Bs)
        if cover:
            Pc = np.take_along_axis(P[t], np.broadcast_to(col[:, None, None], (S, nT, 1)), axis=2)
            X[t] = np.where(later[:, None, :], Pc - P[t], np.float32(0.0))
    Xv = (np.float32(V / N_RIVALS) * X) if cover else None
    del Bs

    tmpr = np.float32(temper)
    places = np.zeros((nT, S))
    C = np.where(np.isfinite(T), T, 1e6).astype(np.float32)          # an illegal lap is not chosen
    converged, it = False, 0
    for it in range(1, int(iters) + 1):
        if cover:
            # `cst[i, u, j]`: what boxing the lap after us costs type u against
            # the lap it planned.  It covers when the place is worth more.
            cst = C[:, col].T[:, :, None] - C[None, :, :]
            Z = P + X * expit((Xv - cst[None]) / tmpr)
        else:
            Z = P
        Q = (w[:, None] * q).astype(np.float32)
        places = np.tensordot(Z, Q, axes=([2, 3], [0, 1])).astype(float)
        cost = T + V * places
        C = np.where(np.isfinite(cost), cost, 1e6).astype(np.float32)
        q_next = DAMPING * q + (1.0 - DAMPING) * _softmin_rows(cost + hist, temper)
        if np.max(np.abs(q_next - q)) < FIXED_POINT_TOL:
            q = q_next
            converged = True
            break
        q = q_next
    return {"packs": packs_of(places, q, it, converged), "iterations": int(it),
            "converged": bool(converged), "types": _type_table(types, laps, q, w), "n_types": nT}


def _type_table(types: list, laps: np.ndarray, q: np.ndarray, w: np.ndarray) -> list:
    """The rival field as a reader sees it: who runs what, and when they box."""
    out = []
    for i, t in enumerate(types):
        qi = np.asarray(q[i], dtype=float)
        s = qi.sum()
        qi = qi / s if s > 0 else np.full(len(laps), 1.0 / max(len(laps), 1))
        cq = np.cumsum(qi)
        n = len(laps)
        out.append({"family": t["label"], "rate_level": int(t.get("level", 0)),
                    "rate_factor": round(float(t.get("rate_factor", 1.0)), 3),
                    "weight": round(float(w[i]), 4),
                    "q_family": round(float(t.get("q_family", float("nan"))), 4),
                    "history_weight": round(float(t.get("hist_w") or 0.0), 3),
                    "stop_lap_median": int(laps[min(int(np.searchsorted(cq, 0.5)), n - 1)]),
                    "stop_lap_p25_p75": [int(laps[min(int(np.searchsorted(cq, 0.25)), n - 1)]),
                                         int(laps[min(int(np.searchsorted(cq, 0.75)), n - 1)])],
                    "ours": bool(t.get("ours"))})
    return out


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
        rho_out: dict = {}
        pa = expected_ahead(P, q, Tr, V, cover_col, later, rho_out=rho_out)[0]  # (I,)
        places += pa
        # the pit response to our PIT NOW: the chance it boxes the lap after us
        # instead of the lap it planned, and where that leaves it
        cov = None
        rho = rho_out.get("rho")
        if rho is not None and len(S):
            i0 = int(np.searchsorted(S, now_lap))
            if i0 < len(S) and int(S[i0]) == int(now_lap) and cover_col[i0] >= 0:
                j = int(cover_col[i0])
                cov = {"p_cover": float((rho[0, i0] * q).sum()),
                       "p_ahead_if_cover": float(P[0, i0, j]),
                       "p_ahead_if_plan": float((P[0, i0] * q).sum()),
                       "cover_lap": int(now_lap) + 1}
        detail.append({"driver": r.code or num, "driver_number": num, "gap_s": round(gap, 2),
                       "cover": cov,
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
