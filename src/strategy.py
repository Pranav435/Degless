"""Decisions, not curves.

Everything here consumes the posterior, so every answer is a *distribution*
over outcomes rather than a point estimate.  "Two-stop M-H-M beats one-stop M-H
with 78% probability, expected gain 6.2 s" is a sentence a point-estimate
optimiser physically cannot say.  That is the whole reason for going Bayesian.

**The decision is a pair: how long, and how hard.**  A driver can always buy
tyre life with lap time - lift and coast, short-shift, roll speed through the
corner instead of attacking the entry - and the question a strategy actually
answers is whether that trade beats stopping again.  So a plan here is a
sequence of stints *and* a push level, and both are optimised.  `src.tyre`
holds the trade-off; this module searches over it.

**What a stint costs.**

    cost(compound c, length L, start s, push p)
        =   sum over the stint of  grip_loss(wear so far)     the tyre
          + L * manage_cost(p)                                the driver
          + L * pace_offset[c]                                the rubber
          + warm-up                                           the out-lap

`grip_loss` is the cliff.  A tyre reaches it after surrendering a roughly fixed
amount of lap time - its grip budget, fitted per compound from the scored
races - so tyre life is *derived* from degradation rate rather than fitted
beside it, and a compound cannot come out both fast-degrading and long-lived.

**Order matters, because fuel load does.**  Wear accrues faster on a heavy car,
so the same (compound, length) pair costs more early in a race than late, and
where a stint *sits* is part of the decision.

**What a plan pays beyond the tyre.**  Pit-lane time, measured.  Dirty air: a
stop rejoins the car into traffic, worth ~1.2 excess close-following laps at
the measured dirty-air cost, scaled by how dense the field is at the rejoin
lap.  The safety car, as a credit rather than a charge - a stop not yet taken
is an option worth the chance a safety car arrives while it is still live.

**Track position, through the undercut.**  Benchmarked on the seven dry 2026
weekends, the tyre-only objective called every first stop 3-6 laps after the
field's median: the field is covering the undercut.  Every lap a car stays out
past the point where a rival on a fresh tyre would gain on it is a lap on
which it can lose the place, and the *exposure* of a stop lap is the
cumulative one-lap undercut gain a rival would have had over those laps
(`undercut_exposure_tables`).  It enters the objective at the calibrated
weight `undercut_lambda`, scaled by field density at the stop.  The
tyre-optimal plan (lambda = 0) is always reported beside the position-aware
one, because the two answer different questions and a strategist wants both.

**The field's plan shapes as a prior.**  A compound sequence nobody has run at
this circuit needs a large time gain to be recommended: the circuit's
historical sequence and start-compound frequencies enter as
`tau * (-log p(family))` (`plan_prior_penalty`), so the sealed model can no
longer do worse on plan shape than its own history-only prior did.

**The circuit's first-stop history, on the first stop only.**  The same
argument applies to *when* the field stops, and the first stop is the decision
the cost surface is worst at: benchmarked on the three non-safety-car 2026
weekends the position-aware objective was still 4-9 laps late.  So the
circuit's own green-flag first stops enter as a density over the first-stop lap
(`src.firststop`) at `kappa * neglogp[first stop]` seconds - zero at the modal
lap, a few tenths across the plausible window, several seconds where the
circuit has never stopped.  Only the first stop: everything after it is
re-decidable on the day, and a lap-2 prior would be pricing a decision the
race has already rewritten.  Green-flag stops only, and a safety-car stop is
held fixed wherever the term appears, so it cancels rather than being judged.

So the objective has **four race terms** beyond the tyre and the pit lane:
dirty air at the rejoin, the safety-car option's credit, the undercut exposure
of every stop lap at `lambda`, and the two history priors - `tau` on the plan
family and `kappa` on the first-stop lap.  Each is calibrated leave-one-out and
each can be switched off by setting its weight to zero, which is what the
ablations do; with all of them at zero the winner is the `tyre_optimal` plan,
reported beside the recommendation whatever the weights are.

What is still *not* priced: track position as a race-long state beyond the
undercut, the starting-tyre rule, and any interaction with what other cars
do.  Those are real and they are why a recommendation here is an input to a
decision rather than the decision.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    DIRTY_AIR_S_PER_LAP,
    GRID_START_PENALTY_S,
    GRIP_BUDGET_S,
    MAX_STINTS_PER_COMPOUND,
    MAX_STOPS,
    MAX_WEAR_LIMIT,
    SUPPORT_EXTRAPOLATION_LIMIT,
    MC_DRAWS,
    OUT_LAP_PENALTY_S,
    PIT_WINDOW_MARGIN,
    PLAN_PRIOR_ALPHA,
    PUSH_GRID,
    SC_PIT_LOSS_FRACTION,
    SC_RATE_PER_LAP,
    TRAFFIC_LAPS_PER_STOP,
    Event,
    get_event,
)
from src.compounds import hardness_rank
from src.regime import RegimeFactor
from src.tyre import TyreModel, grip_loss, load_profile, manage_cost, wear_multiplier

log = logging.getLogger("degless.strategy")

# `search_with_pace_calibration`: the calibration length and the plan it comes
# from have to be the same number, and the length has to sit where the net step
# is defined at all (inside the tyres' lives, before the cliff bends the loss).
MAX_PASSES = 4               # calibrate -> search passes before giving up on a fixed point
PACE_CAL_TOL_LAPS = 2.5      # two stint lengths this close are the same decision
LIFE_CAP_FRACTION = 0.9      # of the shortest mean compound life at the plan's push


def is_sc_status(status) -> bool:
    """Was the track under a safety car, virtual safety car or red flag?

    Track status strings can be composite ("126" = green + VSC + ...), so
    the digits are tested, not the whole string."""
    s = str(status or "")
    return any(ch in s for ch in "4567")


# --------------------------------------------------------------------------
# Pit loss, measured rather than assumed
# --------------------------------------------------------------------------


@dataclass
class PitLoss:
    seconds: float = np.nan
    n_stops: int = 0
    per_stop: pd.DataFrame | None = None

    def __float__(self) -> float:
        return float(self.seconds)


def measure_pit_loss(race: pd.DataFrame, *, window: int = 5) -> PitLoss:
    """(in-lap + out-lap) - 2 x median of the driver's nearby clean laps.

    Green-flag stops only: a stop under a safety car costs far less and would
    bias the estimate low.  Median across drivers.
    """
    d = race.copy()
    rows = []
    for drv, g in d.groupby("driver"):
        g = g.sort_values("lap_number").reset_index(drop=True)
        clean = g[g["is_clean"]] if "is_clean" in g else g
        if clean.empty:
            continue
        for i, row in g.iterrows():
            if not bool(row.get("pit_in")):
                continue
            nxt = g[g["lap_number"] == row["lap_number"] + 1]
            if nxt.empty or not bool(nxt.iloc[0].get("pit_out")):
                continue
            # green flag on both laps
            if str(row.get("track_status")) != "1" or str(nxt.iloc[0].get("track_status")) != "1":
                continue
            t_in = row["lap_time_s"]
            t_out = nxt.iloc[0]["lap_time_s"]
            if not (np.isfinite(t_in) and np.isfinite(t_out)):
                continue
            near = clean[
                (clean["lap_number"] >= row["lap_number"] - window)
                & (clean["lap_number"] <= row["lap_number"] + window)
            ]
            if len(near) < 3:
                near = clean
            if near.empty:
                continue
            ref = float(near["lap_time_s"].median())
            rows.append({"driver": drv, "lap": float(row["lap_number"]),
                         "in_lap": float(t_in), "out_lap": float(t_out),
                         "ref_lap": ref,
                         "loss_s": float(t_in + t_out - 2 * ref)})

    if not rows:
        log.warning("no green-flag pit stops found; pit loss unavailable")
        return PitLoss()
    per = pd.DataFrame(rows)
    # Trim obvious nonsense (penalties served, drive-throughs, damage).
    keep = per[(per["loss_s"] > 5) & (per["loss_s"] < 60)]
    if keep.empty:
        keep = per
    return PitLoss(seconds=float(keep["loss_s"].median()), n_stops=len(keep),
                   per_stop=per.sort_values("loss_s"))


# --------------------------------------------------------------------------
# What a plan pays that is not tyre wear: traffic, the safety car, position
# --------------------------------------------------------------------------


def traffic_density(event: Event | str) -> np.ndarray:
    """Relative density of the field, per race lap, normalised to mean 1.

    Measured directly: the fraction of green-flag race laps run within 3 s of
    the car ahead, pooled over both 2026 weekends and fitted as a quadratic in
    race fraction net of a post-stop term.  It runs 0.69 at the start, 0.38 by
    half distance and 0.48 at the flag - the field is bunched at the start and
    spreads out, which is why rejoining after an early stop is more expensive
    than rejoining after a late one.
    """
    ev = get_event(event) if isinstance(event, str) else event
    f = np.arange(1, ev.n_race_laps + 1, dtype=float) / ev.n_race_laps
    q = 0.692 - 1.037 * f + 0.826 * f ** 2
    return q / 0.4488          # mean of the same quadratic over [0, 1]


def traffic_cost(event: Event | str, pit_laps, *,
                 laps_per_stop: float = TRAFFIC_LAPS_PER_STOP,
                 s_per_lap: float = DIRTY_AIR_S_PER_LAP) -> float:
    """Seconds lost to dirty air because of the stops a plan makes.

    A stop does not only cost pit-lane time: the car rejoins into traffic and
    spends several laps in another car's wake before it is clear.  Measured on
    2026 race laps, a stop raises the probability of running within 3 s of the
    car ahead by 0.11-0.16 for about ten laps afterwards - about 1.2 excess
    close-following laps per stop - and it is scaled here by how dense the
    field is at the lap the car rejoins.
    """
    ev = get_event(event) if isinstance(event, str) else event
    if len(pit_laps) == 0:
        return 0.0
    dens = traffic_density(ev)
    idx = np.clip(np.asarray(pit_laps, dtype=int), 1, ev.n_race_laps) - 1
    return float(laps_per_stop * s_per_lap * dens[idx].sum())


def safety_car_credit(event: Event | str, pit_laps, pit_loss_s: float, *,
                      rate: float = SC_RATE_PER_LAP,
                      loss_fraction: float = SC_PIT_LOSS_FRACTION) -> float:
    """Expected seconds *saved* by still having a stop left when a car stops.

    A pit stop taken under a safety car costs roughly 40% of a green-flag one,
    because the field is slowed while the pit lane is not.  So a stop that has
    not yet been taken is an option with positive value, and its value is the
    probability that a safety car arrives while the option is still live.

    Modelled as a homogeneous Poisson process at `rate` per lap; the credit is
    taken against the *last* scheduled stop, which is the one still available
    for longest, and only one stop is credited.
    """
    ev = get_event(event) if isinstance(event, str) else event
    if len(pit_laps) == 0:
        return 0.0
    last = float(np.max(pit_laps))
    p_sc = 1.0 - np.exp(-float(rate) * last)
    return float(p_sc * (1.0 - float(loss_fraction)) * pit_loss_s)


def undercut_exposure_tables(model: TyreModel, event: Event | str, push: float, max_len: int, *,
                             out_lap_penalty: float = OUT_LAP_PENALTY_S) -> dict:
    """`expo[(old, new)][L]`: cumulative undercut exposure of running `old`
    for `L` laps and then stopping, when the rival fits `new`.

    On the lap a rival pits while you stay out, the rival runs a fresh tyre
    against your aged one.  What they take out of you on that lap is the pace
    your tyre has *lost since it was fresh* - the cumulative degradation at its
    age, which is what an aged tyre is slower by - less the fresh tyre's own
    first-lap loss, the cold out-lap, and whatever the new compound gives away
    on pace:

        gain1(a) = loss_old(a) - loss_new(1) - out_lap - (pace_new - pace_old)

    Every lap you stay out with `gain1 > 0` is a lap on which the place can be
    taken, so the exposure of stopping at age L is the sum of the positive
    gains up to L.  Posterior-mean quantities; the wear is at the plan's push
    and reference fuel load.  Shape (max_len + 1,), index 0 = no exposure.
    """
    ev = get_event(event) if isinstance(event, str) else event
    psi = model.psi(push)
    ages = np.arange(1, max_len + 1, dtype=float)
    loss_old = {c: grip_loss(model.wear_rate[c][:, None] * psi * ages[None, :],
                             budget=model.budget_of(c)).mean(0) for c in model.compounds}
    fresh = {c: float(grip_loss(model.wear_rate[c] * psi, budget=model.budget_of(c)).mean())
             for c in model.compounds}
    pace = {c: float(model.pace_offset[c].mean()) for c in model.compounds}
    out = {}
    for old in model.compounds:
        for new in model.compounds:
            gain1 = loss_old[old] - fresh[new] - out_lap_penalty - (pace[new] - pace[old])
            e = np.zeros(max_len + 1)
            e[1:] = np.cumsum(np.maximum(gain1, 0.0))
            out[(old, new)] = e
    return out


def plan_prior_penalty(seq, prior: dict | None, tau_s: float, *, alpha: float = PLAN_PRIOR_ALPHA) -> float:
    """Seconds of handicap a plan family carries for being rare at this circuit.

    `prior` is `history.plan_prior_for(...)`: sequence, start-compound and
    stop-count counts over classified finishers.  The smoothed probability of
    a family is `(n_seq + alpha * q) / (n + alpha)` with a back-off `q` that
    factorises into the start-compound and stop-count marginals, so a family
    never seen is penalised by how unusual its start and its stop count are
    rather than infinitely.  The modal family carries zero; everything else
    `tau * (log p_modal - log p_family)`.
    """
    if not prior or not tau_s or prior.get("n", 0) <= 0:
        return 0.0
    seq = [str(c) for c in seq]
    short = "-".join(c[0] for c in seq)
    n = float(prior["n"])
    seqs = prior.get("sequences") or {}
    starts = prior.get("starts") or {}
    stops = {int(k): v for k, v in (prior.get("stops") or {}).items()}

    def q_of(s: list) -> float:
        p_start = (starts.get(s[0], 0) + 0.5) / (n + 1.5)
        p_stops = (stops.get(len(s) - 1, 0) + 0.5) / (n + 0.5 * max(len(stops), 1))
        return p_start * p_stops

    def p_of(short_form: str, s: list) -> float:
        return (seqs.get(short_form, 0) + alpha * q_of(s)) / (n + alpha)

    names = {"S": "SOFT", "M": "MEDIUM", "H": "HARD"}
    p_max = max(p_of(k, [names.get(x, x) for x in k.split("-")]) for k in seqs) if seqs else p_of(short, seq)
    p = p_of(short, seq)
    return float(tau_s * max(np.log(p_max) - np.log(p), 0.0))


def first_stop_penalty(start_compound, first_stop_lap, prior: dict | None, kappa_s: float,
                       n_stops=None):
    """Seconds of handicap the first stop carries for being early or late here.

    `prior` is `firststop.first_stop_penalty_table(...)`:
    `prior[compound][n_stops]` is an array indexed by the in-lap, in nats above
    the circuit's modal first stop *for a plan with that many stops*, with
    `prior[compound]["any"]` the unconditional density.  `first_stop_lap` may be
    a scalar or an integer array (the whole family's stop laps at once); the
    result matches.  A plan with no stop, no prior, or `kappa_s = 0` pays
    nothing - which is the V2 objective exactly.

    Conditioning on the stop count matters because a first stop is the opening
    move of a plan: a circuit's two-stoppers box five to eight laps earlier than
    its one-stoppers, and charging a one-stop plan the pooled density pulls it
    toward a lap that only makes sense if you are stopping again.  Where the
    circuit has no history for that family the table's `"any"` entry backs it
    off; an old-style flat `{compound: array}` table is still accepted.

    Index 0 of each array is zero and laps outside the legal window carry the
    density's bounded floor, so no clipping beyond the array's own length is
    needed and a lap nobody has stopped on is expensive rather than impossible.
    """
    k = float(kappa_s or 0.0)
    if not prior or k == 0.0 or first_stop_lap is None:
        return 0.0
    per = prior.get(str(start_compound)) if hasattr(prior, "get") else None
    if per is None:
        return 0.0
    if isinstance(per, dict):
        tbl = None
        if n_stops is not None:
            try:
                tbl = per.get(int(n_stops))
            except (TypeError, ValueError):
                tbl = None
        if tbl is None:
            tbl = per.get("any")
        if tbl is None:
            return 0.0
    else:
        tbl = per                       # the pre-conditioning table shape
    tbl = np.asarray(tbl, dtype=float)
    lap = np.clip(np.asarray(first_stop_lap, dtype=int), 0, len(tbl) - 1)
    pen = k * tbl[lap]
    return float(pen) if np.ndim(first_stop_lap) == 0 else pen


# --------------------------------------------------------------------------
# Stint cost: the tyre model, evaluated over the posterior
# --------------------------------------------------------------------------


def regime_multipliers(regime: RegimeFactor | float | None, n_draws: int,
                       *, seed: int = 7) -> np.ndarray:
    """Kept for the reporting path and for `src.validate`.

    The strategy optimiser no longer multiplies degradation by an exogenous
    practice->race constant - it chooses a push level and pays for it in lap
    time, and the implied regime factor falls out of that choice.
    """
    if regime is None:
        return np.ones(n_draws)
    if isinstance(regime, RegimeFactor):
        return regime.draws(n_draws, seed=seed)
    return np.full(n_draws, float(regime))


def stint_cost_table(model: TyreModel, event: Event | str, max_len: int,
                     push: float, *, warmup_s: float = OUT_LAP_PENALTY_S) -> dict:
    """`cost[c][d, s, L]` in seconds - see `TyreModel.cost_table`."""
    return model.cost_table(event, max_len, push, warmup_s=warmup_s)


def life_caps(model: TyreModel, event: Event | str, *,
              push: float = None, wear_limit: float = MAX_WEAR_LIMIT,
              support: dict | None = None) -> dict:
    """Longest stint the search will consider, per compound.

    The binding limit is *the tyre*, not the extent of the practice data.  A
    stint is allowed out to `wear_limit` times the compound's life at the most
    conservative push the optimiser may use; past that the tyre is more than
    twice its pre-cliff loss rate and no engineer runs it.

    `support` is still accepted and, when given, is reported but not enforced;
    where the two disagree that gap is the honest limit of the analysis and it
    belongs in the output rather than silently in a bound.
    """
    ev = get_event(event) if isinstance(event, str) else event
    p = min(PUSH_GRID) if push is None else push
    out = {}
    for c in model.compounds:
        life = float(np.quantile(model.life_laps(c, p), 0.75))
        cap = wear_limit * life
        if support and c in support and np.isfinite(support[c]):
            cap = min(cap, SUPPORT_EXTRAPOLATION_LIMIT * float(support[c]))
        out[c] = int(max(PIT_WINDOW_MARGIN, min(ev.n_race_laps, round(cap))))
    return out


def _compositions(total: int, n: int, lo: int, hi: int) -> np.ndarray:
    """All ordered stint-length vectors of `n` parts summing to `total`."""
    if n == 1:
        return np.array([[total]], dtype=int) if lo <= total <= hi else np.zeros((0, 1), int)
    head = np.array(list(itertools.product(range(lo, hi + 1), repeat=n - 1)), dtype=int)
    if head.size == 0:
        return np.zeros((0, n), int)
    last = total - head.sum(1)
    ok = (last >= lo) & (last <= hi)
    return np.column_stack([head[ok], last[ok]])


def _stint_caps(compounds: list, max_stint) -> dict:
    """Per-compound maximum stint length, from an int, a dict, or nothing."""
    if max_stint is None:
        return {c: None for c in compounds}
    if isinstance(max_stint, dict):
        return {c: max_stint.get(c) for c in compounds}
    return {c: int(max_stint) for c in compounds}


def compound_life(model: TyreModel, event: Event | str, *,
                  push: float = 0.7, cap: int | dict | None = None,
                  support: dict | None = None) -> pd.DataFrame:
    """Useful life per compound, in laps, at full push and under management.

    `knee_lap` is the cliff at full push: budget / rate.  `life_laps` is the
    cliff at the managed push the optimiser is actually allowed to use, bounded
    by the race distance and by the longest stint the circuit's races have
    supported, with `longer_than_race` and `bound_by` saying which bound held.
    """
    caps = cap if isinstance(cap, dict) else None
    return model.life_table(event, push, caps=caps, support=support)


# --------------------------------------------------------------------------
# Strategy enumeration
# --------------------------------------------------------------------------


def enumerate_strategies(n_laps: int, compounds: list, *, max_stops: int = MAX_STOPS,
                         margin: int = PIT_WINDOW_MARGIN, step: int = 1,
                         max_per_compound: int | dict = MAX_STINTS_PER_COMPOUND,
                         max_stint: int | dict | None = None) -> tuple:
    """Legal plans: 1..max_stops stops, two-compound rule, 1-lap stint grid.

    Returns `(seqs, lens, starts)`.  `seqs[i]` is the ordered compound tuple of
    plan family `i`; `lens[i]` is an (M_i, n) integer array of stint-length
    vectors and `starts[i]` the matching laps-completed-before-each-stint.
    """
    if isinstance(max_per_compound, dict):
        def _alloc(c):
            return int(max_per_compound.get(c, MAX_STINTS_PER_COMPOUND))
    else:
        def _alloc(c):
            return int(max_per_compound)

    caps = _stint_caps(list(compounds), max_stint)
    ceiling = n_laps - margin
    have = [v for v in caps.values() if v is not None]
    hi = min(ceiling, max(have)) if have else ceiling
    if hi < margin:
        return [], [], []
    rank = hardness_rank(list(compounds))
    ordered = [c for _, c in sorted(zip(rank, compounds))]

    seqs, lens_all, starts_all = [], [], []
    for n_stint in range(2, max_stops + 2):
        comp = _compositions(n_laps, n_stint, margin, hi)
        if step > 1 and len(comp):
            comp = comp[(comp[:, :-1] % step == 0).all(1)]
        if not len(comp):
            continue
        for seq in itertools.product(ordered, repeat=n_stint):
            if len(set(seq)) < 2:  # FIA two-compound rule
                continue
            if any(seq.count(c) > _alloc(c) for c in set(seq)):
                continue  # more sets of one compound than the driver has
            keep = np.ones(len(comp), dtype=bool)
            for k, c in enumerate(seq):
                if caps.get(c) is not None:
                    keep &= comp[:, k] <= caps[c]
            sub = comp[keep]
            if not len(sub):
                continue
            starts = np.concatenate(
                [np.zeros((len(sub), 1), int), np.cumsum(sub, axis=1)[:, :-1]], axis=1)
            seqs.append(seq)
            lens_all.append(sub)
            starts_all.append(starts)
    return seqs, lens_all, starts_all


# --------------------------------------------------------------------------
# Monte Carlo race simulation
# --------------------------------------------------------------------------


@dataclass
class StrategyResult:
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    pit_loss_s: float = np.nan
    n_draws: int = 0
    n_strategies: int = 0
    n_scored: int = 0
    max_stint: int | dict | None = None
    best_label: str = ""
    best: dict = field(default_factory=dict)
    regime: dict = field(default_factory=dict)
    warmup_s: float = OUT_LAP_PENALTY_S
    life: pd.DataFrame | None = None
    by_stops: pd.DataFrame | None = None
    push_grid: tuple = PUSH_GRID
    implied_regime: float = np.nan
    model: TyreModel | None = None
    # (n_scored, n_draws) race-time cost of every scored plan on every draw,
    # rows aligned with `table`.
    times: np.ndarray | None = None
    # the plan the tyre alone would choose (no position term, no plan prior)
    tyre_optimal: dict = field(default_factory=dict)
    tyre_optimal_label: str = ""
    undercut_lambda: float = 0.0
    plan_prior_tau_s: float = 0.0
    plan_prior_source: str = ""
    first_stop_kappa_s: float = 0.0
    # V4: the race-state first-stop term, when the search carried one - the
    # measured constants, each plan group's pack equilibrium, and the group of
    # the recommended plan in full (`src.racestate.pack_equilibrium`)
    race_state: dict = field(default_factory=dict)

    def head(self, n: int = 10) -> pd.DataFrame:
        return self.table.head(n)

    @property
    def p_stops(self) -> dict:
        """Probability, over the posterior, that the fastest plan has k stops."""
        if self.by_stops is None or self.by_stops.empty:
            return {}
        return {int(r["n_stops"]): float(r["win_prob_any"]) for _, r in self.by_stops.iterrows()}


def _label(seq, pits, n_stops) -> str:
    return (f"{n_stops}-stop " + "-".join(c[0] for c in seq)
            + " @ " + ",".join(str(int(p)) for p in pits))


def simulate(fit, event: Event | str, pit_loss_s: float, *,
             regime: RegimeFactor | float | None = None,
             n_draws: int = MC_DRAWS, max_stops: int = MAX_STOPS,
             margin: int = PIT_WINDOW_MARGIN, step: int = 1,
             warmup_s: float = OUT_LAP_PENALTY_S, shortlist: int = 4000,
             max_per_compound: int | dict = MAX_STINTS_PER_COMPOUND,
             max_stint: int | dict | None = None,
             push_grid: tuple = PUSH_GRID,
             support: dict | None = None, grid_penalty_s: float = GRID_START_PENALTY_S,
             budget=GRIP_BUDGET_S, seed: int = 0, **kw) -> StrategyResult:
    """Rank every legal (plan, push level) pair from a sealed practice fit.

    Subsamples `n_draws` posterior draws into a `TyreModel` and hands over to
    `simulate_model`.  Extra keyword arguments (`undercut_lambda`,
    `plan_prior`, `plan_prior_tau_s`, `first_stop_prior`, `first_stop_kappa_s`,
    `traffic_s_per_lap`, ...) go through.
    """
    ev = get_event(event) if isinstance(event, str) else event
    total_draws = fit.posterior["lin"].shape[0]
    rng = np.random.default_rng(seed)
    draws = rng.choice(total_draws, size=min(n_draws, total_draws), replace=False)
    model = TyreModel.from_fit(fit, draws=draws, budget=budget)
    return simulate_model(model, ev, pit_loss_s, regime=regime, max_stops=max_stops,
                          margin=margin, step=step, warmup_s=warmup_s, shortlist=shortlist,
                          max_per_compound=max_per_compound, max_stint=max_stint,
                          push_grid=push_grid, support=support, grid_penalty_s=grid_penalty_s, **kw)


def simulate_model(model: TyreModel, event: Event | str, pit_loss_s: float, *,
                   regime: RegimeFactor | float | None = None,
                   max_stops: int = MAX_STOPS,
                   margin: int = PIT_WINDOW_MARGIN, step: int = 1,
                   warmup_s: float = OUT_LAP_PENALTY_S, shortlist: int = 4000,
                   max_per_compound: int | dict = MAX_STINTS_PER_COMPOUND,
                   max_stint: int | dict | None = None,
                   push_grid: tuple = PUSH_GRID,
                   support: dict | None = None, grid_penalty_s: float = GRID_START_PENALTY_S,
                   sc_rate: float = SC_RATE_PER_LAP,
                   traffic_s_per_lap: float = DIRTY_AIR_S_PER_LAP,
                   undercut_lambda: float = 0.0,
                   plan_prior: dict | None = None, plan_prior_tau_s: float = 0.0,
                   first_stop_prior: dict | None = None,
                   first_stop_kappa_s: float = 0.0,
                   race_state=None, race_state_cover: bool = True) -> StrategyResult:
    """Rank every legal (plan, push level) pair, then score a shortlist over
    the posterior draws carried by `model`.

    **V4: the race state times the first stop.**  With `race_state` (a
    `src.racestate.RaceStateConstants`) every plan group - start compound,
    second compound and stop count - is solved as a pack of four rivals running
    the same plan (`racestate.pack_equilibrium`), and the first stop of every
    plan in the group is charged that pack's race-state term: the seconds of
    track position, at the measured value of a place, a first stop on that lap
    gains or gives away against cars choosing theirs the same way.  The term
    *replaces* the undercut exposure on the first stop (it prices the same
    thing, car by car rather than as a generic exposure) and the caller is
    expected to switch the first-stop history prior off (`first_stop_kappa_s =
    0`); later stops keep `undercut_lambda`.  Without `race_state` this is the
    V3 objective, bit for bit.

    Two phases, because ordering the stints multiplies the search space by an
    order of magnitude and the full posterior does not fit alongside it:

    1. every legal plan is costed at every push level on the posterior *mean*
       cost, vectorised, and keeps its best push;
    2. the best `shortlist` plans by that cost are re-costed on every draw.

    The objective is the tyre cost plus pit lane, traffic and the safety-car
    option, plus three terms that are about the *race* rather than the tyre:
    `undercut_lambda` times the undercut exposure of every stop lap, the
    plan-family prior `plan_prior_tau_s * (-log p(family))`, and the circuit's
    first-stop density `first_stop_kappa_s * (-log p(first stop lap))` charged
    on the first stop alone (zero for a plan that never stops).  All three
    default to zero; the plan that wins with all of them at zero is reported as
    `tyre_optimal` whatever they are set to, so a reader always sees what the
    race terms changed.  The best plan at each stop count and for each starting
    compound is always scored, so "the one-stop is 9 s slower" is a sentence
    with a number in it.
    """
    ev = get_event(event) if isinstance(event, str) else event
    n_laps = ev.n_race_laps
    nd = model.n_draws
    lam = float(undercut_lambda or 0.0)
    tau = float(plan_prior_tau_s or 0.0)
    kappa = float(first_stop_kappa_s or 0.0)
    fsp = first_stop_prior if (first_stop_prior and kappa > 0) else None

    caps = life_caps(model, ev, push=min(push_grid), support=support)
    if isinstance(max_stint, dict):        # caller override, if any
        caps = {c: min(v, max_stint[c]) for c, v in caps.items() if c in max_stint} or caps
    elif max_stint is not None:
        caps = {c: min(v, int(max_stint)) for c, v in caps.items()}

    seqs, lens_all, starts_all = enumerate_strategies(
        n_laps, model.compounds, max_stops=max_stops, margin=margin, step=step,
        max_per_compound=max_per_compound, max_stint=caps)
    if not seqs:
        return StrategyResult(model=model)

    _rank = dict(zip(model.compounds, hardness_rank(list(model.compounds))))

    max_len = min(n_laps, max(caps.values()))
    tables = {}          # push -> cost dict, float32
    means = {}           # push -> {compound: (n_start+1, max_len+1)}
    expos = {}           # push -> {(old, new): (max_len+1,)}
    for p in push_grid:
        t = stint_cost_table(model, ev, max_len, p, warmup_s=warmup_s)
        tables[p] = {c: v.astype(np.float32) for c, v in t.items()}
        means[p] = {c: v.mean(0) for c, v in t.items()}
        expos[p] = undercut_exposure_tables(model, ev, p, max_len) if lam > 0 else None
        del t

    # -- phase 1: cost every plan at every push, keep the best push ---------
    fam_full, fam_tyre, fam_push, fam_pos, fam_prior, n_total = [], [], [], [], [], 0
    fam_first, fam_rs = [], []
    dens = traffic_density(ev)
    rs_info: dict = {}
    if race_state is not None:
        (fam_full, fam_tyre, fam_push, fam_pos, fam_prior, fam_first, fam_rs, n_total,
         rs_info) = _phase1_race_state(
            seqs, lens_all, starts_all, means=means, expos=expos, push_grid=push_grid,
            pit_loss_s=pit_loss_s, n_laps=n_laps, max_len=max_len, dens=dens,
            traffic_s_per_lap=traffic_s_per_lap, sc_rate=sc_rate, grid_penalty_s=grid_penalty_s,
            rank=_rank, plan_prior=plan_prior, tau=tau, fsp=fsp, kappa=kappa, lam=lam,
            race_state=race_state, cover=race_state_cover)
    for seq, lens, starts in (zip(seqs, lens_all, starts_all) if race_state is None else ()):
        pits = starts[:, 1:]
        # Terms that do not depend on the tyre model: pit lane, traffic, the
        # safety-car option value, the opening-stint grid penalty.
        fixed = np.full(len(lens), (len(seq) - 1) * pit_loss_s, dtype=np.float64)
        if pits.shape[1]:
            idx = np.clip(pits, 1, n_laps) - 1
            fixed += TRAFFIC_LAPS_PER_STOP * traffic_s_per_lap * dens[idx].sum(1)
            last = pits.max(1).astype(float)
            fixed -= ((1.0 - np.exp(-sc_rate * last))
                      * (1.0 - SC_PIT_LOSS_FRACTION) * pit_loss_s)
        fixed += grid_penalty_s * float(_rank[seq[0]])
        prior_pen = plan_prior_penalty(seq, plan_prior, tau) if tau > 0 else 0.0
        # The first-stop prior depends on the start compound, the stop count and
        # the first stop lap, none of which the push level moves, so it is priced
        # once per family here and carried through both phases unchanged.
        first_pen = (first_stop_penalty(seq[0], pits[:, 0], fsp, kappa, n_stops=len(seq) - 1)
                     if (fsp is not None and pits.shape[1]) else np.zeros(len(lens)))
        first_pen = np.broadcast_to(np.asarray(first_pen, dtype=float), (len(lens),))
        best_full = best_tyre = None
        for p in push_grid:
            t = fixed.copy()
            for k, c in enumerate(seq):
                t += means[p][c][starts[:, k], lens[:, k]]
            pos = np.zeros(len(lens))
            if lam > 0 and pits.shape[1]:
                for k in range(len(seq) - 1):
                    e = expos[p][(seq[k], seq[k + 1])]
                    pos += e[np.clip(lens[:, k], 0, max_len)] * dens[np.clip(pits[:, k], 1, n_laps) - 1]
            full = t + lam * pos + prior_pen + first_pen
            if best_full is None:
                best_full, bp, bpos, best_tyre = full, np.full(len(lens), p, dtype=float), pos, t
            else:
                take = full < best_full
                best_full = np.where(take, full, best_full)
                bp = np.where(take, p, bp)
                bpos = np.where(take, pos, bpos)
                best_tyre = np.minimum(best_tyre, t)
        fam_full.append(best_full)
        fam_tyre.append(best_tyre)
        fam_push.append(bp)
        fam_pos.append(bpos)
        fam_prior.append(np.full(len(lens), prior_pen))
        fam_first.append(np.asarray(first_pen, dtype=float))
        fam_rs.append(np.zeros(len(lens)))
        n_total += len(lens)

    flat = np.concatenate(fam_full)
    flat_tyre = np.concatenate(fam_tyre)
    flat_push = np.concatenate(fam_push)
    flat_pos = np.concatenate(fam_pos)
    flat_prior = np.concatenate(fam_prior)
    flat_first = np.concatenate(fam_first)
    flat_rs = np.concatenate(fam_rs)
    keep = min(shortlist, len(flat))
    order = np.argpartition(flat, keep - 1)[:keep]
    bounds = np.cumsum([0] + [len(x) for x in fam_full])
    stops_flat = np.concatenate([
        np.full(len(x), len(seq) - 1) for x, seq in zip(fam_full, seqs)])
    forced = [int(np.flatnonzero(stops_flat == n)[np.argmin(flat[stops_flat == n])])
              for n in np.unique(stops_flat)]
    first_flat = np.concatenate([np.full(len(x), _rank[seq[0]]) for x, seq in zip(fam_full, seqs)])
    forced += [int(np.flatnonzero(first_flat == r)[np.argmin(flat[first_flat == r])])
               for r in np.unique(first_flat)]
    # ...and the tyre-optimal plan, so its numbers exist under the full objective too
    tyre_idx = int(np.argmin(flat_tyre))
    forced.append(tyre_idx)
    order = np.unique(np.concatenate([order, np.array(forced, dtype=order.dtype)]))
    order = order[np.argsort(flat[order])]
    keep = len(order)

    fam_of = np.searchsorted(bounds, order, side="right") - 1
    row_of = order - bounds[fam_of]

    # -- phase 2: score the shortlist on every posterior draw --------------
    times = np.zeros((keep, nd), dtype=np.float64)
    rows = []
    for i, (fi, ri) in enumerate(zip(fam_of, row_of)):
        seq = seqs[fi]
        L = lens_all[fi][ri]
        st = starts_all[fi][ri]
        p = float(flat_push[order[i]])
        pits = [int(x) for x in st[1:]]
        fixed = (len(seq) - 1) * pit_loss_s
        fixed += traffic_cost(ev, pits, s_per_lap=traffic_s_per_lap)
        fixed -= safety_car_credit(ev, pits, pit_loss_s, rate=sc_rate)
        fixed += grid_penalty_s * float(_rank[seq[0]])
        fixed += lam * float(flat_pos[order[i]]) + float(flat_prior[order[i]])
        fixed += float(flat_first[order[i]])
        fixed += float(flat_rs[order[i]])
        t = np.full(nd, fixed, dtype=np.float64)
        for k, c in enumerate(seq):
            t += tables[p][c][:, st[k], L[k]]
        times[i] = t
        wear = [float(model.wear_at(c, [int(Lk)], [int(sk)], p, ev).mean())
                for c, Lk, sk in zip(seq, L, st)]
        rows.append({
            "strategy": _label(seq, pits, len(seq) - 1),
            "n_stops": len(seq) - 1,
            "compounds": "-".join(seq),
            "pit_laps": pits,
            "stint_lens": [int(x) for x in L],
            "push": p,
            "max_wear": float(np.max(wear)),
            "position_s": float(lam * flat_pos[order[i]]),
            "prior_s": float(flat_prior[order[i]]),
            "first_stop_s": float(flat_first[order[i]]),
            "race_state_s": float(flat_rs[order[i]]),
            "tyre_s": float(flat_tyre[order[i]]),
            "_flat": int(order[i]),
        })

    mean = times.mean(1)
    best_idx = int(np.argmin(mean))
    winner = np.argmin(times, axis=0)
    win_counts = np.bincount(winner, minlength=keep)

    tbl = pd.DataFrame(rows)
    tbl["mean_s"] = mean
    delta = times - times[best_idx][None, :]
    tbl["delta_s"] = delta.mean(1)
    tbl["delta_p05"] = np.quantile(delta, 0.05, axis=1)
    tbl["delta_p95"] = np.quantile(delta, 0.95, axis=1)
    tbl["p05_s"] = np.quantile(times, 0.05, axis=1)
    tbl["p95_s"] = np.quantile(times, 0.95, axis=1)
    tbl["win_prob"] = win_counts / nd
    tbl["p_beat_best"] = (delta < 0).mean(1)
    tbl.loc[best_idx, "p_beat_best"] = np.nan   # a plan does not beat itself
    tyre_row = tbl[tbl["_flat"] == tyre_idx].iloc[0]
    tbl = tbl.drop(columns="_flat")
    tbl = tbl.sort_values("mean_s")
    perm = tbl.index.to_numpy()
    tbl = tbl.reset_index(drop=True)
    times = times[perm]

    by_stops = (tbl.sort_values("mean_s").groupby("n_stops", as_index=False).first()
                .sort_values("mean_s").reset_index(drop=True))
    by_stops["delta_s"] = by_stops["mean_s"] - by_stops["mean_s"].min()
    by_stops["win_prob_any"] = [
        float(tbl.loc[tbl["n_stops"] == n, "win_prob"].sum()) for n in by_stops["n_stops"]
    ]

    top = tbl.iloc[0]
    rg = regime if isinstance(regime, RegimeFactor) else None
    tyre_plan = {"compounds": str(tyre_row["compounds"]).split("-"), "pit_laps": list(tyre_row["pit_laps"]),
                 "stint_lens": list(tyre_row["stint_lens"]), "n_stops": int(tyre_row["n_stops"]),
                 "push": float(tyre_row["push"]), "delta_s": float(tyre_row["mean_s"] - top["mean_s"]),
                 "tyre_s": float(tyre_row["tyre_s"]),
                 # what the tyre-optimal plan's own first stop would have cost under
                 # the prior it was chosen without: the ablation's headline number
                 "first_stop_s": float(tyre_row["first_stop_s"])}
    return StrategyResult(
        table=tbl, pit_loss_s=pit_loss_s, n_draws=nd,
        n_strategies=int(n_total), n_scored=keep, max_stint=caps,
        best_label=top["strategy"],
        best={"compounds": top["compounds"].split("-"),
              "pit_laps": list(top["pit_laps"]),
              "stint_lens": list(top["stint_lens"]),
              "n_stops": int(top["n_stops"]),
              "push": float(top["push"]),
              "position_s": float(top["position_s"]),
              "prior_s": float(top["prior_s"]),
              "first_stop_s": float(top["first_stop_s"]),
              **({"race_state_s": float(top["race_state_s"])} if race_state is not None else {})},
        regime=(rg.as_dict() if rg else {}), warmup_s=warmup_s,
        life=compound_life(model, ev, push=float(top["push"]), cap=caps, support=support),
        by_stops=by_stops, push_grid=tuple(push_grid),
        implied_regime=float(model.psi(float(top["push"]))), model=model,
        times=times, tyre_optimal=tyre_plan, tyre_optimal_label=str(tyre_row["strategy"]),
        undercut_lambda=lam, plan_prior_tau_s=tau, first_stop_kappa_s=kappa,
        plan_prior_source=str((plan_prior or {}).get("source", "")),
        race_state=_race_state_for_best(rs_info, list(top["compounds"].split("-"))))


def _group_label(g: tuple) -> str:
    c0, c1, n = g
    return f"{n}-stop {c0[0]}-{c1[0]}"


def _race_state_for_best(rs_info: dict, seq: list) -> dict:
    """`rs_info` with the recommended plan's own group pulled out as `best`."""
    if not rs_info:
        return {}
    label = _group_label((seq[0], seq[1], len(seq) - 1)) if len(seq) >= 2 else None
    return {**rs_info, "best_group": label, "best": (rs_info.get("packs") or {}).get(label) or {}}


def _phase1_race_state(seqs, lens_all, starts_all, *, means, expos, push_grid, pit_loss_s, n_laps,
                       max_len, dens, traffic_s_per_lap, sc_rate, grid_penalty_s, rank, plan_prior,
                       tau, fsp, kappa, lam, race_state, cover) -> tuple:
    """Phase 1 of `simulate_model` with the race-state first-stop term.

    Three passes.  (1) every plan at every push on everything but the race
    state - tyre, pit lane, traffic, the safety-car credit, the grid penalty,
    the undercut exposure of every stop *after* the first, the plan prior;
    (2) each plan group's cost by first-stop lap (the best plan of the group
    with its first stop on that lap, at its best push) is solved as a pack
    (`racestate.pack_equilibrium`); (3) the group's term is charged on every
    plan's first stop and the best push re-chosen.  A group whose best plan is
    further behind the overall best than the race-state term could ever make up
    (four places at the measured value, plus 5 s) is not solved and carries no
    term: it cannot win either way.
    """
    from src import racestate

    per_fam = []
    for seq, lens, starts in zip(seqs, lens_all, starts_all):
        pits = starts[:, 1:]
        fixed = np.full(len(lens), (len(seq) - 1) * pit_loss_s, dtype=np.float64)
        if pits.shape[1]:
            idx = np.clip(pits, 1, n_laps) - 1
            fixed += TRAFFIC_LAPS_PER_STOP * traffic_s_per_lap * dens[idx].sum(1)
            last = pits.max(1).astype(float)
            fixed -= ((1.0 - np.exp(-sc_rate * last))
                      * (1.0 - SC_PIT_LOSS_FRACTION) * pit_loss_s)
        fixed += grid_penalty_s * float(rank[seq[0]])
        prior_pen = plan_prior_penalty(seq, plan_prior, tau) if tau > 0 else 0.0
        first_pen = (first_stop_penalty(seq[0], pits[:, 0], fsp, kappa, n_stops=len(seq) - 1)
                     if (fsp is not None and pits.shape[1]) else np.zeros(len(lens)))
        first_pen = np.broadcast_to(np.asarray(first_pen, dtype=float), (len(lens),))
        base_p, tyre_p, posl_p = [], [], []
        for p in push_grid:
            t = fixed.copy()
            for k, c in enumerate(seq):
                t += means[p][c][starts[:, k], lens[:, k]]
            posl = np.zeros(len(lens))
            if lam > 0 and pits.shape[1] > 1:
                for k in range(1, len(seq) - 1):
                    e = expos[p][(seq[k], seq[k + 1])]
                    posl += e[np.clip(lens[:, k], 0, max_len)] * dens[np.clip(pits[:, k], 1, n_laps) - 1]
            base_p.append(t + lam * posl + prior_pen + first_pen)
            tyre_p.append(t)
            posl_p.append(posl)
        per_fam.append((np.stack(base_p), np.stack(tyre_p), np.stack(posl_p), prior_pen,
                        np.asarray(first_pen, dtype=float)))

    groups: dict = {}
    for fi, seq in enumerate(seqs):
        if len(seq) >= 2:
            groups.setdefault((seq[0], seq[1], len(seq) - 1), []).append(fi)
    curves = {}
    for g, fis in groups.items():
        Tg = np.full(n_laps + 1, np.inf)
        Pg = np.zeros(n_laps + 1, dtype=int)
        for fi in fis:
            base = per_fam[fi][0]
            pi = np.argmin(base, axis=0)
            val = base[pi, np.arange(base.shape[1])]
            first = starts_all[fi][:, 1]
            o = np.lexsort((val, first))
            uniq, at = np.unique(first[o], return_index=True)
            v = val[o][at]
            better = v < Tg[uniq]
            Tg[uniq[better]] = v[better]
            Pg[uniq[better]] = pi[o][at][better]
        laps = np.flatnonzero(np.isfinite(Tg))
        if len(laps):
            curves[g] = (laps, Tg[laps], Pg[laps])
    V = float(race_state.place_value_s)
    overall = min(float(c[1].min()) for c in curves.values()) if curves else 0.0
    margin = 4.0 * V + 5.0
    packs, terms = {}, {}
    for g, (laps, T, P) in curves.items():
        if float(T.min()) > overall + margin:
            continue
        p = push_grid[int(P[int(np.argmin(T))])]
        stay_cum = np.asarray(means[p][g[0]][0], dtype=float)
        fresh_cum = np.asarray(means[p][g[1]], dtype=float)
        laps_ok = laps[laps < len(stay_cum)]
        T_ok = T[laps < len(stay_cum)]
        pk = racestate.pack_equilibrium(T_ok, laps_ok, stay_cum, fresh_cum, race_state, cover=cover)
        if not pk:
            continue
        pk["push"] = float(p)
        packs[_group_label(g)] = pk
        terms[g] = racestate.term_by_lap(pk, n_laps)

    fam_full, fam_tyre, fam_push, fam_pos, fam_prior, fam_first, fam_rs = [], [], [], [], [], [], []
    n_total = 0
    for fi, (seq, lens, starts) in enumerate(zip(seqs, lens_all, starts_all)):
        base_p, tyre_p, posl_p, prior_pen, first_pen = per_fam[fi]
        g = (seq[0], seq[1], len(seq) - 1) if len(seq) >= 2 else None
        term = terms.get(g)
        rs_rows = term[np.clip(starts[:, 1], 0, n_laps)] if (term is not None and starts.shape[1] > 1) \
            else np.zeros(len(lens))
        full_p = base_p + rs_rows[None, :]
        pi = np.argmin(full_p, axis=0)
        cols = np.arange(len(lens))
        fam_full.append(full_p[pi, cols])
        fam_tyre.append(tyre_p.min(0))
        fam_push.append(np.asarray(push_grid, dtype=float)[pi])
        fam_pos.append(posl_p[pi, cols])
        fam_prior.append(np.full(len(lens), prior_pen))
        fam_first.append(first_pen)
        fam_rs.append(rs_rows)
        n_total += len(lens)
    info = {"constants": race_state.as_dict(), "cover": bool(cover),
            "groups": {k: {kk: v.get(kk) for kk in ("best_lap", "tyre_best_lap", "q_median", "q_p25_p75",
                                                   "iterations", "converged", "push")}
                       for k, v in packs.items()},
            "packs": packs, "n_groups": len(curves), "n_groups_solved": len(packs)}
    return fam_full, fam_tyre, fam_push, fam_pos, fam_prior, fam_first, fam_rs, n_total, info


def search_with_pace_calibration(model: TyreModel, event: Event | str, pit_loss_s: float, *,
                                 net_step_s: float, net_step_se_s: float = 0.0,
                                 seed: int = 0, **sim_kw) -> tuple:
    """The strategy search with the compound-ladder gate enforced.

    The fresh-tyre pace offsets a practice fit carries are prior-dominated;
    what the races identify is the *net* cost of one step harder over a stint
    of the length compounds are run to.  So: search, take the winner's median
    stint length and push, set the offsets so the model reproduces the measured
    net at that length (`calibrate_pace_offsets`), and search again.

    **The length has to be a fixed point.**  The calibration length comes from
    the plan and the plan comes from the calibration, and at Barcelona the pair
    oscillates: calibrated at 24 laps a three-stop wins, calibrated at that
    three-stop's 16.5 laps M-H-H wins again.  A fixed number of passes ships
    whichever one it stopped on - a model calibrated at 16.5 laps carrying a
    24-lap plan - and the ladder gate, which evaluates the net step at the
    *plan's* length, then fails on a model that is correct at a different
    length.  So this iterates to `|L_cal(k+1) - L_cal(k)| <= TOL` (at most
    `MAX_PASSES`), and if it will not settle it keeps the pass whose own plan and
    calibration length agree best rather than the last one tried.

    **And the length has to be inside the tyre's life.**  The net stint-level
    step is only defined while the loss is linear in age; past the cliff it is
    not, so a plan that runs a compound to 90% of its life and beyond gives a
    "net step" that is an artefact of the cliff's shape.  `L_cal` is therefore
    capped at `LIFE_CAP_FRACTION` of the shortest mean compound life at the
    plan's push, which is the band `calibrate_pace_offsets` can actually hit.

    Returns `(model, result, info)`; with no measured net the model is unchanged
    and `info` says so.  `info` carries `stint_len` (the length the shipped model
    is calibrated at - what the ladder gate must evaluate the net step at),
    `stint_len_plan`, `life_cap`, `passes` and `converged`.
    """
    from src.compounds import calibrate_pace_offsets

    ev = get_event(event) if isinstance(event, str) else event
    res0 = simulate_model(model, ev, pit_loss_s, **sim_kw)
    if res0.table.empty or not np.isfinite(net_step_s):
        return model, res0, {"applied": False, "why": "no plan or no measured net stint step"}

    def cal_length(res) -> tuple:
        """(L_cal, L_plan, cap) for a result: the plan's median stint, clipped to
        the shortest compound life at that plan's push."""
        lp = float(np.median(res.best["stint_lens"]))
        p_ = float(res.best["push"])
        lives = [float(np.mean(model.life_laps(c, p_))) for c in model.compounds]
        cap = LIFE_CAP_FRACTION * float(min(lives)) if lives else float("inf")
        return float(min(lp, cap)), lp, cap

    L, L_plan, cap = cal_length(res0)
    best, tried = None, []
    res, model2, info = res0, model, {"applied": False, "why": "no pass converged"}
    for k in range(MAX_PASSES):
        m_k, info_k = calibrate_pace_offsets(model, net_step_s, L, float(res.best["push"]), ev,
                                             net_se_s=net_step_se_s, seed=seed)
        if not info_k.get("applied"):
            return model, res0, {**info_k, "stint_len": L, "stint_len_plan": L_plan,
                                 "life_cap": cap, "passes": k + 1, "converged": False}
        res_k = simulate_model(m_k, ev, pit_loss_s, **sim_kw)
        if res_k.table.empty:
            break
        L_next, L_plan_k, cap_k = cal_length(res_k)
        tried.append({"pass": k + 1, "calibrated_at": L, "plan": res_k.best_label,
                      "plan_stint_len": L_plan_k, "next_length": L_next, "life_cap": cap_k})
        gap = abs(L_plan_k - L)
        if best is None or gap < best[0]:
            best = (gap, m_k, res_k, info_k, L, L_plan_k, cap_k)
        model2, res, info = m_k, res_k, info_k
        if abs(L_next - L) <= PACE_CAL_TOL_LAPS:
            info = {**info_k, "stint_len": L, "stint_len_plan": L_plan_k, "life_cap": cap_k,
                    "passes": k + 1, "converged": True, "passes_detail": tried}
            break
        L, L_plan, cap = L_next, L_plan_k, cap_k
    else:
        # Did not settle: ship the pass whose plan and calibration length agree
        # best, so the gate is evaluated on a model calibrated where its own plan
        # actually runs.
        if best is not None:
            _, model2, res, info_b, L_used, L_plan_b, cap_b = best
            info = {**info_b, "stint_len": L_used, "stint_len_plan": L_plan_b, "life_cap": cap_b,
                    "passes": MAX_PASSES, "converged": False, "passes_detail": tried}
    if "stint_len" not in info:
        info = {**info, "stint_len": L, "stint_len_plan": L_plan, "life_cap": cap,
                "passes": len(tried) or 1, "converged": False, "passes_detail": tried}
    info["best_before"] = res0.best_label
    info["best_after"] = res.best_label
    return model2, res, info


def best_by_start(res: StrategyResult) -> pd.DataFrame:
    """The best plan for each starting compound, with its cost against the
    overall best."""
    if res.table.empty:
        return pd.DataFrame()
    t = res.table.copy()
    t["start"] = t["compounds"].str.split("-").str[0]
    out = t.sort_values("mean_s").groupby("start", as_index=False).first()
    out["delta_s"] = out["mean_s"] - t["mean_s"].min()
    out["win_prob_any"] = [float(t.loc[t["start"] == s, "win_prob"].sum()) for s in out["start"]]
    rank = {c: i for i, c in enumerate(hardness_rank_order(list(out["start"])))}
    out["_o"] = out["start"].map(rank)
    return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def hardness_rank_order(compounds: list) -> list:
    """Compounds sorted softest-first."""
    r = hardness_rank(list(compounds))
    return [c for _, c in sorted(zip(r, compounds))]


def per_driver_plans(model: TyreModel, event: Event | str, pit_loss_s: float, drivers: list, *,
                     race_factors: dict | None = None, dev_by_driver: dict | None = None,
                     factor_ln_sd: dict | None = None, factor_shrink: float | None = 1.0,
                     warmup_by_driver: dict | None = None,
                     traffic_mult_by_driver: dict | None = None,
                     step: int = 2, shortlist: int = 800, **sim_kw) -> pd.DataFrame:
    """The best plan per car: the field model scaled by each driver's own
    tyre behaviour (`TyreModel.for_driver`), searched on a 2-lap grid.

    Four per-car terms, each optional and each backward compatible:

    * `race_factors` - the driver's measured rate factor from previous races.
      A bare `{driver: float}` map works as it always did; a
      `{driver: {"factor", "ln_sd"}}` map (or a plain factor plus
      `factor_ln_sd`) carries how precisely it was measured, and
      `factor_shrink=None` then shrinks it toward 1 by its own standard error
      (`percar.shrink_factor`) instead of applying it at face value.
    * `dev_by_driver` - a replacement for the practice fit's own per-driver
      deviation, normally `percar.team_pooled_dev(...)`, so a car with no long
      run on a compound inherits its team-mate's behaviour rather than the
      field's.  A driver absent from the map keeps the fit's own deviation.
    * `warmup_by_driver` - the stint warm-up cost for that car, in seconds,
      replacing the field's `warmup_s` for that car's search alone
      (`src.haascar.car_terms(state)["warmup_s"]`, which is field-relative: a
      car with no evidence of its own gets the field value back).
    * `traffic_mult_by_driver` - a multiplier on that car's dirty-air cost, so
      the search sees `traffic_s_per_lap * mult` for that car alone
      (`...["traffic_mult"]`, 1.0 for a car with no evidence).

    The last two default to `None`, which leaves the search exactly as it was:
    a keyword is inserted only for a driver the map actually names, and the
    field search (`base`) never sees either of them.

    One row per driver: the plan, the first-stop lap, how far it sits from the
    field plan, and the driver's rate factor with its sources.
    """
    ev = get_event(event) if isinstance(event, str) else event
    rf = race_factors or {}
    sds = factor_ln_sd or {}
    devs = dev_by_driver or {}
    base = simulate_model(model, ev, pit_loss_s, step=step, shortlist=shortlist, **sim_kw)
    base_traffic = float(sim_kw.get("traffic_s_per_lap", DIRTY_AIR_S_PER_LAP))
    rows = []
    for drv in drivers:
        v = rf.get(drv, 1.0)
        f = float(v.get("factor", 1.0)) if isinstance(v, dict) else float(v)
        ln_sd = v.get("ln_sd") if isinstance(v, dict) else sds.get(drv)
        dev_override = devs.get(drv)
        m_d = model.for_driver(drv, race_factor={"factor": f, "ln_sd": ln_sd},
                               dev_override=dev_override, factor_shrink=factor_shrink)
        dev = dev_override if dev_override is not None else model.driver_dev.get(drv, {})
        car_kw = dict(sim_kw)
        warm_d = (warmup_by_driver or {}).get(drv)
        mult_d = (traffic_mult_by_driver or {}).get(drv)
        if warm_d is not None:
            car_kw["warmup_s"] = float(warm_d)
        if mult_d is not None:
            car_kw["traffic_s_per_lap"] = base_traffic * float(mult_d)
        res = simulate_model(m_d, ev, pit_loss_s, step=step, shortlist=shortlist, **car_kw)
        if res.table.empty:
            continue
        rows.append({"driver": drv, "race_factor": f,
                     "race_factor_ln_sd": (float(ln_sd) if ln_sd is not None else None),
                     "dev_source": ("pooled" if dev_override is not None else "own practice"),
                     "practice_dev_s_per_lap": {c: round(float(np.asarray(x).mean()), 4) for c, x in dev.items()},
                     "eff_rate": {c: round(float(m_d.rate(c).mean()), 4) for c in m_d.compounds},
                     "best": res.best_label, "n_stops": res.best["n_stops"], "compounds": "-".join(res.best["compounds"]),
                     "pit_laps": list(res.best["pit_laps"]), "push": res.best["push"],
                     "first_stop": (int(res.best["pit_laps"][0]) if res.best["pit_laps"] else None),
                     "first_stop_s": float(res.best.get("first_stop_s", 0.0)),
                     "field_best": base.best_label,
                     "first_stop_vs_field": ((int(res.best["pit_laps"][0]) - int(base.best["pit_laps"][0]))
                                             if res.best["pit_laps"] and base.best["pit_laps"] else None),
                     "same_shape_as_field": bool("-".join(res.best["compounds"]) == "-".join(base.best["compounds"]))})
        # Only reported when the caller asked for them, so a default call's
        # frame - and the `meta["per_driver"]` block built from it - is unchanged.
        if warmup_by_driver is not None:
            rows[-1]["warmup_s"] = (float(warm_d) if warm_d is not None else None)
        if traffic_mult_by_driver is not None:
            rows[-1]["traffic_mult"] = (float(mult_d) if mult_d is not None else None)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pit window: where the decision is actually loose
# --------------------------------------------------------------------------


def pit_window(fit, event: Event | str, plan: dict, pit_loss_s: float, *,
               regime: RegimeFactor | float | None = None,
               n_draws: int = MC_DRAWS, margin: int = PIT_WINDOW_MARGIN,
               warmup_s: float = OUT_LAP_PENALTY_S,
               max_stint: int | dict | None = None, tolerance_s: float = 1.0,
               push: float | None = None,
               budget=GRIP_BUDGET_S, seed: int = 0, **kw) -> pd.DataFrame:
    """`pit_window_model` on a subsample of a sealed fit's draws.

    `first_stop_prior` / `first_stop_kappa_s` and the other objective weights go
    through `**kw`."""
    ev = get_event(event) if isinstance(event, str) else event
    if not plan:
        return pd.DataFrame()
    total = fit.posterior["lin"].shape[0]
    rng = np.random.default_rng(seed)
    draws = rng.choice(total, size=min(n_draws, total), replace=False)
    model = TyreModel.from_fit(fit, draws=draws, budget=budget)
    return pit_window_model(model, ev, plan, pit_loss_s, margin=margin, warmup_s=warmup_s,
                            max_stint=max_stint, tolerance_s=tolerance_s, push=push, **kw)


def pit_window_model(model: TyreModel, event: Event | str, plan: dict, pit_loss_s: float, *,
                     margin: int = PIT_WINDOW_MARGIN, warmup_s: float = OUT_LAP_PENALTY_S,
                     max_stint: int | dict | None = None, tolerance_s: float = 1.0,
                     push: float | None = None, sc_rate: float = SC_RATE_PER_LAP,
                     traffic_s_per_lap: float = DIRTY_AIR_S_PER_LAP,
                     undercut_lambda: float = 0.0,
                     first_stop_prior: dict | None = None,
                     first_stop_kappa_s: float = 0.0,
                     race_state_term: np.ndarray | None = None) -> pd.DataFrame:
    """Cost of moving one stop earlier or later, holding the others fixed.

    Sweeping each stop lap one at a time turns the single recommended lap into
    a *window* - the laps within `tolerance_s` of optimal - which is the form
    the decision is made in.  The push level is held at the plan's own, and the
    undercut-exposure and first-stop-prior terms are priced at the same weights
    the search used, so the window is the window of the objective that chose the
    plan: sweeping the first stop without its prior would report a window the
    recommendation does not live in.

    `race_state_term` (V4, seconds indexed by lap - `racestate.term_by_lap` of
    the plan group's pack) is charged on the first stop in place of its
    undercut exposure, exactly as `simulate_model` charged it.
    """
    ev = get_event(event) if isinstance(event, str) else event
    n_laps = ev.n_race_laps
    if not plan:
        return pd.DataFrame()
    nd = model.n_draws
    p = float(plan.get("push", 1.0) if push is None else push)
    lam = float(undercut_lambda or 0.0)
    kappa = float(first_stop_kappa_s or 0.0)
    fsp = first_stop_prior if (first_stop_prior and kappa > 0) else None
    caps = life_caps(model, ev, push=p)
    if isinstance(max_stint, dict):
        caps = {c: min(v, max_stint.get(c, v)) for c, v in caps.items()}
    max_len = min(n_laps, max(caps.values()))
    cost = stint_cost_table(model, ev, max_len, p, warmup_s=warmup_s)
    expo = undercut_exposure_tables(model, ev, p, max_len) if lam > 0 else None
    dens = traffic_density(ev)
    rst = None if race_state_term is None else np.asarray(race_state_term, dtype=float)

    seq, pits = list(plan["compounds"]), list(plan["pit_laps"])

    def race_time(pl):
        lens = np.diff([0, *pl, n_laps])
        if lens.min() < margin or lens.max() > max_len:
            return None
        if any(L > caps.get(c, max_len) for c, L in zip(seq, lens)):
            return None
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        t = np.full(nd, len(pl) * pit_loss_s, dtype=float)
        t += traffic_cost(ev, pl, s_per_lap=traffic_s_per_lap)
        t -= safety_car_credit(ev, pl, pit_loss_s, rate=sc_rate)
        for c, L, st in zip(seq, lens, starts):
            t += cost[c][:, int(st), int(L)]
        if expo is not None:
            for k in range(1 if rst is not None else 0, len(seq) - 1):
                t += lam * expo[(seq[k], seq[k + 1])][int(lens[k])] * dens[int(pl[k]) - 1]
        if fsp is not None and len(pl):
            t += first_stop_penalty(seq[0], int(pl[0]), fsp, kappa, n_stops=len(pl))
        if rst is not None and len(pl):
            t += float(rst[min(int(pl[0]), len(rst) - 1)])
        return t

    base = race_time(pits)
    if base is None:
        return pd.DataFrame()
    ref = float(base.mean())

    rows = []
    for k in range(len(pits)):
        for lap in range(margin, n_laps - margin + 1):
            pl = list(pits)
            pl[k] = lap
            if any(b - a < margin for a, b in zip(pl, pl[1:])):
                continue
            t = race_time(pl)
            if t is None:
                continue
            d = t - base
            rows.append({
                "stop": k + 1,
                "lap": lap,
                "loss_s": float(t.mean() - ref),
                "lo": float(np.quantile(d, 0.05)),
                "hi": float(np.quantile(d, 0.95)),
                "is_recommended": lap == pits[k],
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["loss_s"] = out["loss_s"] - out.groupby("stop")["loss_s"].transform("min")
    out["in_window"] = out["loss_s"] <= tolerance_s
    return out


def windows_from_sweep(pw: pd.DataFrame, plan: dict) -> list:
    """`[{stop, recommended, lo, hi}]` from a pit-window sweep."""
    if pw is None or pw.empty:
        return []
    out = []
    for k, g in pw.groupby("stop"):
        w = g[g["in_window"]]
        out.append({"stop": int(k), "recommended": int(plan["pit_laps"][int(k) - 1]),
                    "lo": int(w["lap"].min()), "hi": int(w["lap"].max())})
    return out


# --------------------------------------------------------------------------
# Undercut
# --------------------------------------------------------------------------


def undercut_window(fit, compound_old: str, compound_new: str, *,
                    event: Event | str = None, max_age: int = 40,
                    out_lap_penalty: float = OUT_LAP_PENALTY_S,
                    push: float = 0.85, budget=GRIP_BUDGET_S,
                    regime: RegimeFactor | float | None = None,
                    n_draws: int = MC_DRAWS) -> pd.DataFrame:
    """`undercut_window_model` on an evenly spaced subsample of a sealed fit."""
    total = fit.posterior["lin"].shape[0]
    idx = np.linspace(0, total - 1, min(n_draws, total)).astype(int)
    model = TyreModel.from_fit(fit, draws=idx, budget=budget)
    return undercut_window_model(model, compound_old, compound_new, max_age=max_age,
                                 out_lap_penalty=out_lap_penalty, push=push)


def undercut_window_model(model: TyreModel, compound_old: str, compound_new: str, *,
                          max_age: int = 40, out_lap_penalty: float = OUT_LAP_PENALTY_S,
                          push: float = 0.85) -> pd.DataFrame:
    """One-lap undercut gain as a function of the leader's tyre age.

    If you pit on lap L and the car ahead stays out one more lap, on that lap
    you run a fresh tyre while they run an aged one.  What you take out of
    them is the pace their tyre has lost *since it was fresh* - a 15-lap-old
    tyre is slower by its accumulated degradation, not by its last lap's
    increment - less your own fresh tyre's first-lap loss, the cold out-lap,
    and whatever your new compound gives away on pace.

    A previous version charged only the leader's *marginal* loss (the extra
    time their next lap costs over their last), which is the right quantity
    for "how much more do they lose by staying out one more lap" and the wrong
    one for what an undercut takes: it made the undercut worth less than the
    out-lap on every pre-cliff lap, i.e. never, which is not what happens at
    Barcelona.  This is the level-based gain, the same arithmetic as
    `undercut_exposure_tables` and the live engine.
    """
    ages = np.arange(1, max_age + 1, dtype=float)
    psi = model.psi(push)
    loss_old = grip_loss(model.wear_rate[compound_old][:, None] * psi * ages[None, :],
                         budget=model.budget_of(compound_old))
    w_new = model.wear_rate[compound_new][:, None] * psi
    fresh = grip_loss(w_new, budget=model.budget_of(compound_new))
    off = (model.pace_offset[compound_new] - model.pace_offset[compound_old])[:, None]
    gain = loss_old - fresh - out_lap_penalty - off
    return pd.DataFrame({
        "leader_tyre_age": ages,
        "gain_s": gain.mean(0),
        "lo": np.quantile(gain, 0.05, axis=0),
        "hi": np.quantile(gain, 0.95, axis=0),
        "p_positive": (gain > 0).mean(0),
    })


def undercut_duel(model: TyreModel, *, my_compound: str, my_age: float,
                  their_compound: str, their_age: float, gap_s: float,
                  new_compound: str, push: float = 0.85, laps: int = 5,
                  out_lap_penalty: float = OUT_LAP_PENALTY_S,
                  event: Event | str | None = None, lap_now: int | None = None) -> dict:
    """One specific duel: if the attacker pits now onto `new_compound` and the
    defender stays out `k` more laps, is the attacker ahead after `k` laps?

    Over each of those laps the attacker on a fresh tyre laps at the fresh
    tyre's loss while the defender laps at their aged tyre's *absolute* loss
    (relative to fresh), so the attacker gains the difference plus the
    compound difference, less the cold first lap once.  The undercut works
    when that cumulative gain exceeds the gap.  Reported per lap of exposure
    as a probability over the model's draws.  The overcut is the mirror
    image: it works on the laps where this gain is *negative*.
    """
    psi = model.psi(push)
    load = 1.0
    if event is not None and lap_now is not None:
        ev = get_event(event) if isinstance(event, str) else event
        lp = load_profile(ev)
        load = float(lp[min(max(int(lap_now), 1), ev.n_race_laps) - 1])
    ks = np.arange(1, laps + 1, dtype=float)
    rate_d = model.wear_rate[their_compound] * psi * load
    rate_n = model.wear_rate[new_compound] * psi * load
    w0 = model.wear_rate[their_compound] * psi * float(their_age)
    def_loss = grip_loss(w0[:, None] + rate_d[:, None] * ks[None, :], budget=model.budget_of(their_compound))
    att_loss = grip_loss(rate_n[:, None] * ks[None, :], budget=model.budget_of(new_compound))
    off = (model.pace_offset[their_compound] - model.pace_offset[new_compound])[:, None]
    per_lap = def_loss - att_loss + off
    cum = np.cumsum(per_lap, axis=1) - out_lap_penalty            # (n, laps)
    p_k = (cum > gap_s).mean(0)
    med = cum.mean(0)
    laps_needed = next((int(k + 1) for k in range(laps) if p_k[k] >= 0.5), None)
    return {"gap_s": float(gap_s), "p_by_lap": [float(x) for x in p_k],
            "gain_by_lap_s": [float(x) for x in med],
            "gain_lo": [float(x) for x in np.quantile(cum, 0.05, axis=0)],
            "gain_hi": [float(x) for x in np.quantile(cum, 0.95, axis=0)],
            "laps_needed": laps_needed,
            "p_undercut_1lap": float(p_k[0]), "p_undercut_3lap": float(p_k[min(2, laps - 1)]),
            "gap_beaten_after_3_s": float(med[min(2, laps - 1)])}


# --------------------------------------------------------------------------
# Counterfactual: "the moment"
# --------------------------------------------------------------------------


def race_stops(race: pd.DataFrame, event: Event | str, *, compounds: list | None = None) -> dict:
    """Per driver: compound sequence, actual in-laps, and whether each stop
    was taken under a safety car / VSC / red flag (the in-lap's track status,
    or the lap before it when the in-lap itself is missing)."""
    ev = get_event(event) if isinstance(event, str) else event
    out = {}
    status = race.set_index(["driver", "lap_number"])["track_status"].astype(str).to_dict()
    for drv, g in race.groupby("driver"):
        g = g.sort_values("lap_number")
        stints = (g.groupby("stint")
                  .agg(compound=("compound", "first"), start=("lap_number", "min"), end=("lap_number", "max"))
                  .sort_values("start").reset_index())
        if compounds is not None:
            stints = stints[stints["compound"].isin(compounds)]
        if len(stints) < 2:
            continue
        seq = stints["compound"].tolist()
        starts = stints["start"].to_numpy()[1:].astype(int)
        in_laps = [int(s) - 1 for s in starts]
        sc = [bool(is_sc_status(status.get((drv, float(p)), status.get((drv, float(p - 1)), "1"))))
              for p in in_laps]
        out[drv] = {"compounds": seq, "in_laps": in_laps, "sc": sc,
                    "finished_lap": int(g["lap_number"].max()), "classified": bool(g["lap_number"].max() >= ev.n_race_laps - 2)}
    return out


def counterfactual(fit_or_model, event: Event | str, race: pd.DataFrame,
                   pit_loss_s: float, *, margin: int = PIT_WINDOW_MARGIN,
                   regime: RegimeFactor | float | None = None,
                   warmup_s: float = OUT_LAP_PENALTY_S,
                   max_stint: int | dict | None = None, push: float = 0.7,
                   budget=GRIP_BUDGET_S, n_draws: int = 200,
                   race_factors: dict | None = None,
                   traffic_s_per_lap: float = DIRTY_AIR_S_PER_LAP,
                   undercut_lambda: float = 0.0,
                   first_stop_prior: dict | None = None, first_stop_kappa_s: float = 0.0,
                   include_unclassified: bool = False) -> pd.DataFrame:
    """For each driver: what their actual stop laps cost against the best
    alternative *with the same compounds and the same number of stops*.

    Holding the compound sequence fixed is deliberate - it isolates the timing
    decision, which is the one the pit wall actually owns on the day.

    **Safety-car aware.**  A stop taken under a safety car, VSC or red flag
    was not a timing decision on the model's cost surface: it was a cheap
    stop the race handed the driver.  Those stops are held at their actual
    laps (and charged the discounted pit loss) while the green-flag stops are
    optimised around them, and each row says how many of the driver's stops
    were of that kind.  A driver whose plan the model cannot price (three
    stints on one compound, a damaged car) is flagged, not dropped.

    **Per car.**  With `race_factors` the driver's own tyre model (the field
    model scaled by their measured race factor and the practice fit's
    per-driver deviation) prices their race, so "seconds lost" is against
    what *their* tyre would have allowed.

    **The first-stop prior.**  Charged on the driver's actual first stop and on
    every alternative's, at `first_stop_kappa_s`, so the comparison is made
    under the objective that produced the recommendation.  A *safety-car* first
    stop is held at its actual lap by the logic below, so the term is identical
    on both sides and cancels - a driver handed a cheap stop is not judged
    against the circuit's green-flag history.

    Vectorised: every legal placement of the free stops is costed on the
    posterior-mean tables at once; only the actual plan and the best
    alternative are priced draw by draw.
    """
    ev = get_event(event) if isinstance(event, str) else event
    n_laps = ev.n_race_laps
    if isinstance(fit_or_model, TyreModel):
        model = fit_or_model
    else:
        total = fit_or_model.posterior["lin"].shape[0]
        idx = np.linspace(0, total - 1, min(n_draws, total)).astype(int)
        model = TyreModel.from_fit(fit_or_model, draws=idx, budget=budget)
    nd = model.n_draws
    max_len = n_laps
    dens = traffic_density(ev)
    lam = float(undercut_lambda or 0.0)
    kappa = float(first_stop_kappa_s or 0.0)
    fsp = first_stop_prior if (first_stop_prior and kappa > 0) else None
    rf = race_factors or {}
    stops = race_stops(race, ev, compounds=list(model.compounds))
    grid = np.arange(margin, n_laps - margin + 1)

    cache: dict = {}

    def tables_for(drv):
        f = rf.get(drv, 1.0)
        f = float(f.get("factor", 1.0)) if isinstance(f, dict) else float(f)
        key = (round(f, 4), drv if model.driver_dev.get(drv) else "_")
        if key not in cache:
            m = model.for_driver(drv, race_factor=f) if (f != 1.0 or model.driver_dev.get(drv)) else model
            cost = stint_cost_table(m, ev, max_len, push, warmup_s=warmup_s)
            expo = undercut_exposure_tables(m, ev, push, max_len) if lam > 0 else None
            cache[key] = (m, cost, {c: v.mean(0) for c, v in cost.items()}, expo, f)
        return cache[key]

    def plan_fixed(pits, sc_flags, seq=None):
        pits = list(pits)
        t = 0.0
        for p_, s_ in zip(pits, sc_flags):
            t += pit_loss_s * (SC_PIT_LOSS_FRACTION if s_ else 1.0)
        t += traffic_cost(ev, pits, s_per_lap=traffic_s_per_lap)
        t -= safety_car_credit(ev, pits, pit_loss_s)
        if fsp is not None and seq and pits:
            t += first_stop_penalty(seq[0], int(pits[0]), fsp, kappa, n_stops=len(pits))
        return t

    rows = []
    for drv, info in stops.items():
        seq, actual, sc_flags = info["compounds"], info["in_laps"], info["sc"]
        if not info["classified"] and not include_unclassified:
            # a retirement is not a strategy: the plan was never completed,
            # and pricing its last stint to the flag invents a 150 s loss
            continue
        lens = np.diff(np.concatenate([[0], actual, [n_laps]])).astype(int)
        if lens.min() <= 0:
            continue
        m_d, cost, means, expo, factor = tables_for(drv)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        base = np.full(nd, plan_fixed(actual, sc_flags, seq), dtype=float)
        for c, L, st in zip(seq, lens, starts):
            base += cost[c][:, int(st), int(L)]
        if expo is not None:
            for k in range(len(seq) - 1):
                base += lam * expo[(seq[k], seq[k + 1])][int(lens[k])] * dens[int(actual[k]) - 1]
        wear = max(float(m_d.wear_at(c, [int(L)], [int(st)], push, ev).mean())
                   for c, L, st in zip(seq, lens, starts))
        # -- the alternatives: free stops on the grid, SC stops held --------
        free = [k for k, s_ in enumerate(sc_flags) if not s_]
        n_stops = len(actual)
        if len(free) > 3:
            # a six-stop race is a damaged car, not a strategy; enumerating
            # 50^6 placements is neither useful nor quick.  Optimise the first
            # three green-flag stops and hold the rest where they were.
            free = free[:3]
        if free:
            axes = [grid] * len(free)
            mesh = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(free))
            P = np.tile(np.asarray(actual, int), (len(mesh), 1))
            P[:, free] = mesh
            ok = np.all(np.diff(P, axis=1) >= margin, axis=1) if n_stops > 1 else np.ones(len(P), bool)
            ok &= (P[:, 0] >= margin) & (n_laps - P[:, -1] >= margin)
            P = P[ok]
        else:
            P = np.asarray([actual], int)
        if len(P) == 0:
            P = np.asarray([actual], int)
        Lm = np.diff(np.concatenate([np.zeros((len(P), 1), int), P, np.full((len(P), 1), n_laps)], axis=1), axis=1)
        Sm = np.concatenate([np.zeros((len(P), 1), int), np.cumsum(Lm, axis=1)[:, :-1]], axis=1)
        tm = np.zeros(len(P))
        for k, c in enumerate(seq):
            tm += means[c][Sm[:, k], np.clip(Lm[:, k], 0, max_len)]
        sc_loss = sum(pit_loss_s * (SC_PIT_LOSS_FRACTION if s_ else 1.0) for s_ in sc_flags)
        tm += sc_loss + TRAFFIC_LAPS_PER_STOP * traffic_s_per_lap * dens[np.clip(P, 1, n_laps) - 1].sum(1)
        last = P.max(1).astype(float)
        tm -= (1.0 - np.exp(-SC_RATE_PER_LAP * last)) * (1.0 - SC_PIT_LOSS_FRACTION) * pit_loss_s
        if expo is not None:
            for k in range(len(seq) - 1):
                tm += lam * expo[(seq[k], seq[k + 1])][np.clip(Lm[:, k], 0, max_len)] * dens[np.clip(P[:, k], 1, n_laps) - 1]
        if fsp is not None:
            # the driver's own stop count: the alternatives keep their plan's shape
            tm = tm + first_stop_penalty(seq[0], P[:, 0], fsp, kappa, n_stops=n_stops)
        j = int(np.argmin(tm))
        best_pits = [int(x) for x in P[j]]
        bl = np.diff(np.concatenate([[0], best_pits, [n_laps]])).astype(int)
        bs = np.concatenate([[0], np.cumsum(bl)[:-1]])
        best = np.full(nd, plan_fixed(best_pits, sc_flags, seq), dtype=float)
        for c, L, st in zip(seq, bl, bs):
            best += cost[c][:, int(st), int(L)]
        if expo is not None:
            for k in range(len(seq) - 1):
                best += lam * expo[(seq[k], seq[k + 1])][int(bl[k])] * dens[int(best_pits[k]) - 1]
        delta = base - best
        rows.append({
            "driver": drv,
            "compounds": "-".join(seq),
            "actual_pit_laps": list(map(int, actual)),
            "model_pit_laps": best_pits,
            "loss_s": float(max(delta.mean(), 0.0)),
            "lo": float(np.quantile(delta, 0.05)),
            "hi": float(np.quantile(delta, 0.95)),
            "p_loss_positive": float((delta > 0).mean()),
            "max_wear": wear,
            "n_sc_stops": int(sum(sc_flags)),
            "sc_stop_laps": [int(p_) for p_, s_ in zip(actual, sc_flags) if s_],
            "classified": bool(info["classified"]),
            "race_factor": factor,
            "flag": ("" if info["classified"] else "not classified: damage or retirement, not strategy")
                    + ("; stops under SC/VSC held fixed" if any(sc_flags) else ""),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("loss_s", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Evaluating *given* plans: the strategist's own candidates, under scenarios
# --------------------------------------------------------------------------


def scale_model(model: TyreModel, deg_mult: float = 1.0, pace_mult: float = 1.0) -> TyreModel:
    """The same draws with every wear rate scaled - a degradation scenario."""
    if deg_mult == 1.0 and pace_mult == 1.0:
        return model
    return model.copy_with(
        wear_rate={c: model.wear_rate[c] * float(deg_mult) for c in model.compounds},
        pace_offset={c: model.pace_offset[c] * float(pace_mult) for c in model.compounds},
        source=f"{model.source} x{deg_mult:.2f} deg")


def stint_lap_losses(model: TyreModel, event: Event | str, compound: str, start: int,
                     length: int, push: float, *, warmup_s: float = OUT_LAP_PENALTY_S) -> np.ndarray:
    """Per-lap cost of one stint, shape (draws, length) - the un-summed form of
    `TyreModel.cost_table`, so a plan can be drawn on the race clock lap by lap."""
    ev = get_event(event) if isinstance(event, str) else event
    n = ev.n_race_laps
    L = int(length)
    if L <= 0:
        return np.zeros((model.n_draws, 0))
    lf = load_profile(ev, exponent=model.load_exponent)
    laps = np.clip(np.arange(start + 1, start + L + 1), 1, n) - 1
    psi = model.psi(push)
    inc = model.wear_rate[compound][:, None] * psi * lf[laps][None, :]
    w = np.cumsum(inc, axis=1)
    loss = grip_loss(w - 0.5 * inc, budget=model.budget_of(compound))
    loss = loss + model.pace_offset[compound][:, None] + model.mcost(push)
    loss[:, 0] += float(warmup_s)
    return loss


def plan_label(compounds, pit_laps) -> str:
    return _label(list(compounds), list(pit_laps), len(list(compounds)) - 1)


def parse_plan_label(label: str) -> dict | None:
    """`"2-stop M-H-H @ 18,36"` -> `{compounds, pit_laps}`; None if it is not a plan label."""
    try:
        head, pits = label.split("@")
        seq = head.strip().split(" ", 1)[1].strip().split("-")
        names = {"S": "SOFT", "M": "MEDIUM", "H": "HARD"}
        comps = [names.get(s.strip(), s.strip()) for s in seq]
        pl = [int(x) for x in pits.strip().split(",") if x.strip()]
        return {"compounds": comps, "pit_laps": pl}
    except Exception:
        return None


def evaluate_plans(model: TyreModel, event: Event | str, plans: list, pit_loss_s: float, *,
                   push: float | None = None, push_grid: tuple = PUSH_GRID,
                   warmup_s: float = OUT_LAP_PENALTY_S,
                   grid_penalty_s: float = GRID_START_PENALTY_S,
                   sc_rate: float = SC_RATE_PER_LAP,
                   traffic_s_per_lap: float = DIRTY_AIR_S_PER_LAP,
                   allocation: dict | None = None, stint_cap: dict | None = None,
                   margin: int = PIT_WINDOW_MARGIN,
                   undercut_lambda: float = 0.0,
                   plan_prior: dict | None = None, plan_prior_tau_s: float = 0.0,
                   first_stop_prior: dict | None = None,
                   first_stop_kappa_s: float = 0.0) -> tuple:
    """Price a list of plans on every draw, with a lap-by-lap trace for each.

    Each plan is `{"compounds": [...], "pit_laps": [...], "push": optional,
    "label": optional}`.  Returns `(table, details)`.  The objective is the one
    `simulate_model` uses, including the position, plan-prior and first-stop
    terms at the weights given, so a plan built by hand is priced exactly as the
    optimiser would price it.  The first-stop penalty is charged on the plan's
    own first stop and reported as `first_stop_s`.
    """
    ev = get_event(event) if isinstance(event, str) else event
    n = ev.n_race_laps
    nd = model.n_draws
    dens = traffic_density(ev)
    rank = dict(zip(model.compounds, hardness_rank(list(model.compounds))))
    alloc = allocation or {}
    caps = stint_cap or {}
    lam = float(undercut_lambda or 0.0)
    tau = float(plan_prior_tau_s or 0.0)
    kappa = float(first_stop_kappa_s or 0.0)
    fsp = first_stop_prior if (first_stop_prior and kappa > 0) else None
    details, rows = [], []

    for i, pl in enumerate(plans):
        seq = [str(c).upper() for c in pl.get("compounds", [])]
        pits = [int(x) for x in pl.get("pit_laps", [])]
        label = pl.get("label") or plan_label(seq, pits)
        flags = []
        ok = (len(seq) >= 1 and len(pits) == len(seq) - 1
              and all(c in model.compounds for c in seq)
              and all(1 <= p_ <= n - 1 for p_ in pits)
              and all(b > a for a, b in zip(pits, pits[1:])))
        if not ok:
            details.append({"label": label, "compounds": seq, "pit_laps": pits, "valid": False,
                            "flags": ["stop laps must be in order, between lap 1 and the last lap, "
                                      "one fewer than the stints; compounds must be ones the model has"]})
            rows.append({"plan": label, "n_stops": len(pits), "compounds": "-".join(seq),
                         "pit_laps": pits, "valid": False, "flags": details[-1]["flags"][0]})
            continue
        lens = np.diff([0, *pits, n]).astype(int)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int)
        if len(set(seq)) < 2:
            flags.append("one compound only: a dry race needs two")
        counts = {c: seq.count(c) for c in set(seq)}
        for c, k in counts.items():
            if alloc.get(c) is not None and k > int(alloc[c]):
                flags.append(f"{k} stints on {c}: allocation allows {int(alloc[c])}")
        for k, (c, L) in enumerate(zip(seq, lens)):
            if L < margin:
                flags.append(f"stint {k + 1} is only {L} laps")
            if caps.get(c) is not None and L > int(caps[c]):
                flags.append(f"{c} {L} laps: this circuit has never supported more than {int(caps[c])}")
        prior_pen = plan_prior_penalty(seq, plan_prior, tau) if tau > 0 else 0.0
        first_pen = (first_stop_penalty(seq[0], int(pits[0]), fsp, kappa, n_stops=len(pits))
                     if (fsp is not None and pits) else 0.0)

        def cost_at(p_):
            per_lap = np.zeros((nd, n))
            for k, (c, L, s) in enumerate(zip(seq, lens, starts)):
                per_lap[:, s:s + L] = stint_lap_losses(model, ev, c, int(s), int(L), p_, warmup_s=warmup_s)
            for p_lap in pits:
                per_lap[:, p_lap - 1] += pit_loss_s + TRAFFIC_LAPS_PER_STOP * traffic_s_per_lap * dens[p_lap - 1]
            per_lap[:, 0] += grid_penalty_s * float(rank[seq[0]]) + prior_pen
            if first_pen:
                # on the stop it prices, so the lap-by-lap trace shows it there
                per_lap[:, int(pits[0]) - 1] += first_pen
            pos = 0.0
            if lam > 0 and pits:
                expo = undercut_exposure_tables(model, ev, p_, n)
                for k in range(len(seq) - 1):
                    e = lam * expo[(seq[k], seq[k + 1])][int(lens[k])] * dens[pits[k] - 1]
                    per_lap[:, pits[k] - 1] += e
                    pos += e
            credit = safety_car_credit(ev, pits, pit_loss_s, rate=sc_rate)
            return per_lap, credit, pos

        p_use = pl.get("push", push)
        if p_use is None:
            best = None
            for p_ in push_grid:
                per_lap, credit, pos = cost_at(float(p_))
                m = per_lap.sum(1).mean() - credit
                if best is None or m < best[0]:
                    best = (m, float(p_), per_lap, credit, pos)
            _, p_use, per_lap, credit, pos = best
        else:
            p_use = float(p_use)
            per_lap, credit, pos = cost_at(p_use)
        times = per_lap.sum(1) - credit
        wear_end = [model.wear_at(c, [int(L)], [int(s)], p_use, ev)[:, 0]
                    for c, L, s in zip(seq, lens, starts)]
        for k, w in enumerate(wear_end):
            if float(w.mean()) > 1.0:
                flags.append(f"stint {k + 1} ends past the cliff (wear {float(w.mean()):.2f})")
        cum = np.cumsum(per_lap, axis=1)
        details.append({
            "label": label, "compounds": seq, "pit_laps": pits, "stint_lens": [int(x) for x in lens],
            "push": p_use, "valid": True, "flags": flags, "times": times, "sc_credit_s": float(credit),
            "position_s": float(pos), "prior_s": float(prior_pen), "first_stop_s": float(first_pen),
            "trace_mean": cum.mean(0), "trace_lo": np.quantile(cum, 0.05, axis=0),
            "trace_hi": np.quantile(cum, 0.95, axis=0), "per_lap_mean": per_lap.mean(0),
            "wear_end_mean": [float(w.mean()) for w in wear_end],
            "wear_end_p90": [float(np.quantile(w, 0.9)) for w in wear_end],
        })
        rows.append({"plan": label, "n_stops": len(pits), "compounds": "-".join(seq), "pit_laps": pits,
                     "stint_lens": [int(x) for x in lens], "push": p_use, "valid": True,
                     "mean_s": float(times.mean()), "position_s": float(pos), "prior_s": float(prior_pen),
                     "first_stop_s": float(first_pen),
                     "max_wear": float(max(w.mean() for w in wear_end)),
                     "wear_end": [round(float(w.mean()), 2) for w in wear_end],
                     "flags": "; ".join(flags)})

    tbl = pd.DataFrame(rows)
    valid = [d for d in details if d.get("valid")]
    if valid:
        T = np.stack([d["times"] for d in valid])
        best_i = int(np.argmin(T.mean(1)))
        delta = T - T[best_i][None, :]
        fastest = np.bincount(np.argmin(T, axis=0), minlength=len(valid)) / T.shape[1]
        j = 0
        for idx, d in enumerate(details):
            if not d.get("valid"):
                continue
            d["delta_s"] = float(delta[j].mean())
            d["delta_p05"] = float(np.quantile(delta[j], 0.05))
            d["delta_p95"] = float(np.quantile(delta[j], 0.95))
            d["p_fastest"] = float(fastest[j])
            tbl.loc[idx, "delta_s"] = d["delta_s"]
            tbl.loc[idx, "delta_p05"] = d["delta_p05"]
            tbl.loc[idx, "delta_p95"] = d["delta_p95"]
            tbl.loc[idx, "p_fastest"] = d["p_fastest"]
            j += 1
    return tbl, details


def deg_crossover(model: TyreModel, event: Event | str, plan_a: dict, plan_b: dict,
                  pit_loss_s: float, *, mults=None, **kw) -> dict:
    """At what degradation multiplier does plan B overtake plan A?"""
    ev = get_event(event) if isinstance(event, str) else event
    mults = np.round(np.arange(0.6, 2.61, 0.1), 2) if mults is None else np.asarray(mults, float)
    curve = []
    for m in mults:
        tbl, det = evaluate_plans(scale_model(model, float(m)), ev, [plan_a, plan_b], pit_loss_s, **kw)
        if len(det) < 2 or not (det[0].get("valid") and det[1].get("valid")):
            return {"mult": None, "curve": [], "note": "one of the plans is not valid"}
        d = float(det[1]["times"].mean() - det[0]["times"].mean())
        p = float((det[1]["times"] < det[0]["times"]).mean())
        curve.append({"mult": float(m), "b_minus_a_s": d, "p_b_faster": p})
    base = next((c for c in curve if abs(c["mult"] - 1.0) < 1e-9), curve[0])
    b_better_now = base["b_minus_a_s"] < 0
    cross = None
    if not b_better_now:
        for c in curve:
            if c["mult"] >= 1.0 and c["b_minus_a_s"] < 0:
                cross = c["mult"]
                break
        direction = "up"
    else:
        for c in reversed([c for c in curve if c["mult"] <= 1.0]):
            if c["b_minus_a_s"] > 0:
                cross = c["mult"]
                break
        direction = "down"
    return {"mult": cross, "direction": direction, "b_better_at_base": bool(b_better_now), "curve": curve}


def value_of_information(res: StrategyResult, *, n_bins: int = 4) -> dict:
    """Which uncertainty the decision actually hinges on (EVPI per compound)."""
    if res.times is None or res.table.empty or res.model is None:
        return {}
    T = res.times
    mean = T.mean(1)
    i_star = int(np.argmin(mean))
    best_per_draw = T.min(0)
    evpi = float(np.mean(T[i_star] - best_per_draw))
    out = {"evpi_s": evpi, "best": str(res.table.iloc[i_star]["strategy"]), "by_compound": {}}
    for c in res.model.compounds:
        r = res.model.rate(c)
        edges = np.quantile(r, np.linspace(0, 1, n_bins + 1))
        bin_of = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, n_bins - 1)
        gain = np.zeros(T.shape[1])
        bests = []
        for b in range(n_bins):
            m = bin_of == b
            if not m.any():
                bests.append(None)
                continue
            i_b = int(np.argmin(T[:, m].mean(1)))
            gain[m] = T[i_star, m] - T[i_b, m]
            bests.append(str(res.table.iloc[i_b]["strategy"]))
        out["by_compound"][c] = {
            "gain_s": float(gain.mean()),
            "share": float(gain.mean() / evpi) if evpi > 1e-9 else 0.0,
            "rate_bins_s_per_lap": [[float(edges[b]), float(edges[b + 1])] for b in range(n_bins)],
            "best_by_rate_bin": bests,
            "decision_moves": len({b for b in bests if b}) > 1,
        }
    return out


def sc_playbook(model: TyreModel, event: Event | str, plan: dict, pit_loss_s: float, *,
                push: float | None = None, allocation: dict | None = None,
                stint_cap: dict | None = None, margin: int = PIT_WINDOW_MARGIN,
                sc_fraction: float = SC_PIT_LOSS_FRACTION,
                warmup_s: float = OUT_LAP_PENALTY_S,
                traffic_s_per_lap: float = DIRTY_AIR_S_PER_LAP) -> pd.DataFrame:
    """If a safety car comes out on lap L, do we box this lap?

    For every lap of the race: the cost of staying on the plan from here
    against the cost of pitting *now*, under the safety car, at the discounted
    pit loss, and then running the best continuation with the tyres that are
    left.  Only the decision *this lap* is priced.
    """
    ev = get_event(event) if isinstance(event, str) else event
    n = ev.n_race_laps
    nd = model.n_draws
    seq = [str(c).upper() for c in plan["compounds"]]
    pits = [int(x) for x in plan["pit_laps"]]
    p = float(plan.get("push", 1.0) if push is None else push)
    lens = np.diff([0, *pits, n]).astype(int)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int)
    alloc = allocation or {}
    caps = stint_cap or {}
    dens = traffic_density(ev)
    per_lap = np.zeros((nd, n))
    for c, L, s in zip(seq, lens, starts):
        per_lap[:, s:s + L] = stint_lap_losses(model, ev, c, int(s), int(L), p, warmup_s=warmup_s)
    for p_lap in pits:
        per_lap[:, p_lap - 1] += pit_loss_s + TRAFFIC_LAPS_PER_STOP * traffic_s_per_lap * dens[p_lap - 1]
    tables = model.cost_table(ev, n, p, warmup_s=warmup_s)
    means = {c: tables[c].mean(0) for c in model.compounds}

    def legal_seq(used: dict, extra: list) -> bool:
        counts = dict(used)
        for c in extra:
            counts[c] = counts.get(c, 0) + 1
        if any(alloc.get(c) is not None and k > int(alloc[c]) for c, k in counts.items()):
            return False
        return len(counts) >= 2

    def cap_ok(c, L):
        return caps.get(c) is None or L <= int(caps[c])

    rows = []
    for L in range(1, n - margin + 1):
        k = int(np.searchsorted(starts, L, side="right") - 1)          # stint containing lap L
        age = L - int(starts[k])
        used = {c: seq[:k + 1].count(c) for c in set(seq[:k + 1])}
        rem = seq[k + 1:]
        stay = per_lap[:, L:].sum(1)
        planned_here = L in pits
        R = n - L
        cands = []
        avail = [c for c in model.compounds if used.get(c, 0) < int(alloc.get(c, 99))]
        first = ([rem[0]] if rem and rem[0] in avail else []) + [c for c in avail if not rem or c != rem[0]]
        for c_new in first:
            if R >= margin and cap_ok(c_new, R) and legal_seq(used, [c_new]):
                cands.append((float(means[c_new][L, R]), [c_new], [], c_new))
            c2s = [rem[1]] if len(rem) >= 2 and rem[1] in avail else avail
            for c2 in c2s:
                if not legal_seq(used, [c_new, c2]):
                    continue
                qs = np.arange(L + margin, n - margin + 1)
                if len(qs) == 0:
                    continue
                L1 = qs - L
                L2 = n - qs
                ok = np.array([cap_ok(c_new, a) and cap_ok(c2, b) for a, b in zip(L1, L2)])
                if not ok.any():
                    continue
                cost = (means[c_new][L, L1] + means[c2][qs, L2] + pit_loss_s
                        + TRAFFIC_LAPS_PER_STOP * traffic_s_per_lap * dens[qs - 1])
                cost = np.where(ok, cost, np.inf)
                j = int(np.argmin(cost))
                if np.isfinite(cost[j]):
                    cands.append((float(cost[j]), [c_new, c2], [int(qs[j])], c_new))
            if len(rem) >= 3:
                c2, c3 = rem[1], rem[2]
                if legal_seq(used, [c_new, c2, c3]):
                    best2 = None
                    for q1 in range(L + margin, n - 2 * margin + 1):
                        q2 = np.arange(q1 + margin, n - margin + 1)
                        if len(q2) == 0:
                            continue
                        L1, L2, L3 = q1 - L, q2 - q1, n - q2
                        ok = np.array([cap_ok(c_new, L1) and cap_ok(c2, b) and cap_ok(c3, d) for b, d in zip(L2, L3)])
                        if not ok.any():
                            continue
                        cost = (means[c_new][L, L1] + means[c2][q1, L2] + means[c3][q2, L3] + 2 * pit_loss_s
                                + TRAFFIC_LAPS_PER_STOP * traffic_s_per_lap * (dens[q1 - 1] + dens[q2 - 1]))
                        cost = np.where(ok, cost, np.inf)
                        j = int(np.argmin(cost))
                        if np.isfinite(cost[j]) and (best2 is None or cost[j] < best2[0]):
                            best2 = (float(cost[j]), [c_new, c2, c3], [int(q1), int(q2[j])], c_new)
                    if best2:
                        cands.append(best2)
        if not cands:
            continue
        cost_mean, comps, stops, c_new = min(cands, key=lambda t: t[0])
        cont = np.full(nd, pit_loss_s * sc_fraction + TRAFFIC_LAPS_PER_STOP * traffic_s_per_lap * dens[L - 1])
        bounds = [L, *stops, n]
        for c, a, b in zip(comps, bounds[:-1], bounds[1:]):
            cont += tables[c][:, a, b - a]
        cont += len(stops) * pit_loss_s + TRAFFIC_LAPS_PER_STOP * traffic_s_per_lap * sum(dens[q - 1] for q in stops)
        gain = stay - cont
        g = float(gain.mean())
        p_pit = float((gain > 0).mean())
        if planned_here:
            verdict, g, p_pit = "PLANNED", float((1 - sc_fraction) * pit_loss_s), 1.0
        elif g > 1.0 and p_pit >= 0.6:
            verdict = "PIT"
        elif g < -1.0 and p_pit <= 0.4:
            verdict = "STAY"
        else:
            verdict = "MARGINAL"
        label = " → ".join(f"{c} {'to the flag' if i == len(comps) - 1 else f'until lap {stops[i]}'}"
                           for i, c in enumerate(comps))
        rows.append({"lap": L, "stint": k + 1, "compound": seq[k], "age_on_set": age,
                     "gain_s": g, "p_pit": p_pit, "verdict": verdict, "continuation": label,
                     "new_compound": c_new, "further_stops": stops})
    return pd.DataFrame(rows)


def playbook_ranges(pb: pd.DataFrame) -> list:
    """Compress a lap-by-lap playbook into `[{from, to, verdict, gain_s}]` runs."""
    if pb is None or pb.empty:
        return []
    out = []
    cur = None
    for _, r in pb.iterrows():
        if cur is None or r["verdict"] != cur["verdict"]:
            if cur is not None:
                out.append(cur)
            cur = {"from": int(r["lap"]), "to": int(r["lap"]), "verdict": r["verdict"],
                   "gains": [float(r["gain_s"])], "continuation": r["continuation"]}
        else:
            cur["to"] = int(r["lap"])
            cur["gains"].append(float(r["gain_s"]))
    if cur is not None:
        out.append(cur)
    for o in out:
        o["gain_s"] = float(np.mean(o.pop("gains")))
    return out
