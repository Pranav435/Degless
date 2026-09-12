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

This replaces an earlier design in which degradation was multiplied by a fixed
practice->race "regime" constant of 0.40.  That constant is a description of an
equilibrium, not a law: it is what you observe when a driver chooses to manage
a tyre.  Fixing it had two consequences, both visible in the output.  The model
could not represent the manage-or-stop trade at all, and - because the constant
was measured by an estimator that removed track evolution from the practice
side but not from the race side - it was too small by roughly half, so
degradation was under-predicted by 1.3x at Barcelona 2026 and 3.6x at Hungary
2026 against those weekends' own races.  An optimiser fed degradation that is
three times too low answers with too few stops and stints that are too long,
which is exactly what it did: a 33-lap stint on the SOFT at Hungary.

**What a stint costs.**

    cost(compound c, length L, start s, push p)
        =   sum over the stint of  grip_loss(wear so far)     the tyre
          + L * manage_cost(p)                                the driver
          + L * pace_offset[c]                                the rubber
          + warm-up                                           the out-lap

`grip_loss` is the cliff.  A tyre reaches it after surrendering a roughly fixed
amount of lap time - its grip budget, ~3.8 s, measured to within +/-9% across
three compounds at Barcelona 2026 - so tyre life is *derived* from degradation
rate rather than fitted beside it, and a compound cannot come out both
fast-degrading and long-lived.  This is what puts a physical limit on stint
length.  The old linear-plus-hinge curve had none: its hinge was unidentified,
because practice long runs stop at age 15-22 laps, before any compound reaches
its cliff, so the posterior was its prior and the curve was effectively a
straight line.  Nothing about a straight line says when a tyre is finished.

**Order matters, because fuel load does.**  Wear accrues faster on a heavy car,
so the same (compound, length) pair costs more early in a race than late, and
where a stint *sits* is part of the decision.  This is what reproduces the
pattern every real race shows and an order-invariant model cannot: stints get
longer as the race goes on.

**What a plan pays beyond the tyre.**  Pit-lane time, measured.  Dirty air: a
stop rejoins the car into traffic, worth ~1.2 excess close-following laps at
~0.45 s each, scaled by how dense the field is at the rejoin lap.  And the
safety car, as a credit rather than a charge - a stop not yet taken is an
option worth the chance a safety car arrives while it is still live, which is
why holding a stop in reserve has value and a model blind to it over-values
plans that have used all of theirs.

What is still *not* priced: track position as a race-long state, the
starting-tyre rule, and any interaction with what other cars do.  Those are
real and they are why a recommendation here is an input to a decision rather
than the decision.
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
# What a plan pays that is not tyre wear: traffic, and the safety car
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
    close-following laps per stop, worth ~0.55 s at the measured 0.45 s/lap
    cost of dirty air - and it is scaled here by how dense the field is at the
    lap the car rejoins.

    Without this term an extra stop is priced as pit-lane time alone, with the
    laps spent recovering afterwards free.  It is small next to pit loss, but
    it is signed the right way and it is the only thing in the objective that
    knows an early stop rejoins into a denser field than a late one.
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
    for longest, and only one stop is credited - a second safety car inside one
    race is rare enough that pricing it would be false precision.

    This is why a plan is not simply "fewer stops is better once the tyres
    last": holding a stop in reserve is worth real time, and a model without it
    systematically over-values plans that have already used all of theirs.
    """
    ev = get_event(event) if isinstance(event, str) else event
    if len(pit_laps) == 0:
        return 0.0
    last = float(np.max(pit_laps))
    p_sc = 1.0 - np.exp(-float(rate) * last)
    return float(p_sc * (1.0 - float(loss_fraction)) * pit_loss_s)


# --------------------------------------------------------------------------
# Stint cost: the tyre model, evaluated over the posterior
# --------------------------------------------------------------------------


def regime_multipliers(regime: RegimeFactor | float | None, n_draws: int,
                       *, seed: int = 7) -> np.ndarray:
    """Kept for the reporting path and for `src.validate`.

    The strategy optimiser no longer multiplies degradation by an exogenous
    practice->race constant - it chooses a push level and pays for it in lap
    time, and the implied regime factor falls out of that choice (see
    `src.tyre`).  This helper survives because the sealed-prediction and
    scoring code still reports a measured regime factor alongside the model's
    own, which is the cross-check that the old design could not offer.
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
    twice its pre-cliff loss rate and no engineer runs it, so evaluating plans
    out there only spends search on answers nobody would give.

    This replaces the previous rule, which capped at 1.5x the oldest tyre age
    practice happened to reach.  That rule had nothing to do with the tyre: at
    Hungary 2026 it let the SOFT run to 33 laps against its own fitted useful
    life of 26, because the practice support happened to be 22 laps.  The cap
    was doing the opposite of its job - the *least*-supported compound got the
    most extrapolation.

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
        # Second bound, and a different kind of claim.  The wear bound says
        # "the tyre is finished"; this one says "the curve is no longer
        # evidence".  The degradation rate is fitted on practice long runs, and
        # extrapolating it to a stint twice as long as anything practice ran is
        # asserting something the data does not carry - most visibly at a
        # low-degradation circuit, where the wear bound alone permits a stint
        # longer than the race.  Both are reported so a reader can see which
        # one binds, because where they disagree that gap is the honest limit
        # of the analysis.
        if support and c in support and np.isfinite(support[c]):
            cap = min(cap, SUPPORT_EXTRAPOLATION_LIMIT * float(support[c]))
        out[c] = int(max(PIT_WINDOW_MARGIN, min(ev.n_race_laps, round(cap))))
    return out


def _compositions(total: int, n: int, lo: int, hi: int) -> np.ndarray:
    """All ordered stint-length vectors of `n` parts summing to `total`.

    Ordered, not multisets: wear accrues faster on a heavy car, so a plan's
    cost depends on which stint runs when, and `S-H` and `H-S` are genuinely
    different plans.
    """
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

    `knee_lap` is the cliff: where the tyre has spent its grip budget and pace
    loss starts running away.  Unlike the old free `knee` parameter it is not
    fitted independently of the degradation rate - it is `budget / rate`, so a
    compound cannot come out both fast-degrading and long-lived, which is the
    contradiction the old fit kept producing.

    `life_laps` is the cliff at the managed push level the optimiser is
    actually allowed to use, which is the number an engineer wants: how long
    can this tyre go if the driver looks after it.
    """
    rows = []
    for c in model.compounds:
        full = model.life_laps(c, 1.0)
        man = model.life_laps(c, push)
        rows.append({"compound": c,
                     "knee_lap": float(full.mean()),
                     "life_laps": float(man.mean()),
                     "life_lo": float(np.quantile(man, 0.05)),
                     "life_hi": float(np.quantile(man, 0.95)),
                     "deg_s_per_lap": float((model.wear_rate[c] * model.budget).mean()),
                     "max_stint_laps": float((cap or {}).get(c, np.nan))
                     if isinstance(cap, dict) else float("nan"),
                     "practice_support_laps": float((support or {}).get(c, np.nan))})
    return pd.DataFrame(rows)


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
    vectors and `starts[i]` the matching laps-completed-before-each-stint.  The
    array form is what keeps the search vectorised: Barcelona at 1-lap
    resolution is ~1.1M ordered plans, which is fine as numpy and hopeless as a
    Python list of dicts.

    `max_per_compound` is the tyre allocation: a driver reaches the grid with
    roughly two usable sets of each compound, and 43 of the 44 driver-races
    across both weekends used at most two stints on any one compound.
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
    # Softest first, so a plan's printed order matches the conventional reading.
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
    # rows aligned with `table`.  Kept so downstream code can ask questions
    # the summary columns cannot answer - which uncertainty the decision
    # actually hinges on, or how the ranking moves under a scenario.
    times: np.ndarray | None = None

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
             budget: float = GRIP_BUDGET_S, seed: int = 0) -> StrategyResult:
    """Rank every legal (plan, push level) pair from a sealed practice fit.

    Subsamples `n_draws` posterior draws into a `TyreModel` and hands over to
    `simulate_model`, which does the work - the same search the live outlook
    runs on a prior-only model before any practice has happened.
    """
    ev = get_event(event) if isinstance(event, str) else event
    total_draws = fit.posterior["lin"].shape[0]
    rng = np.random.default_rng(seed)
    draws = rng.choice(total_draws, size=min(n_draws, total_draws), replace=False)
    model = TyreModel.from_fit(fit, draws=draws, budget=budget)
    return simulate_model(model, ev, pit_loss_s, regime=regime, max_stops=max_stops,
                          margin=margin, step=step, warmup_s=warmup_s, shortlist=shortlist,
                          max_per_compound=max_per_compound, max_stint=max_stint,
                          push_grid=push_grid, support=support, grid_penalty_s=grid_penalty_s)


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
                   traffic_s_per_lap: float = DIRTY_AIR_S_PER_LAP) -> StrategyResult:
    """Rank every legal (plan, push level) pair, then score a shortlist over
    the posterior draws carried by `model`.

    The decision variable is a *pair*.  How long to run a stint and how hard to
    run it are one decision, not two: a driver who nurses a tyre buys stint
    length with lap time, and the whole question a strategy answers is whether
    that trade is worth taking against stopping again.  A model that fixes the
    push level - which is what an exogenous practice->race regime constant
    does - cannot represent the trade at all, and answers every question with
    whatever stint length its assumed degradation rate implies.

    Two phases, because ordering the stints multiplies the search space by an
    order of magnitude and the full posterior does not fit alongside it:

    1. every legal plan is costed at every push level on the posterior *mean*
       cost, vectorised, and keeps its best push;
    2. the best `shortlist` plans by that cost are re-costed on every draw.

    Win probability is "fastest among the shortlist", and a plan outside the
    top few thousand on the mean cost has no realistic path to being fastest on
    a draw - but it is stated rather than implied, which is why `n_strategies`
    (searched) and `n_scored` (on the posterior) are reported separately.

    `sc_rate` and `traffic_s_per_lap` are exposed so a scenario ("a safety car
    is twice as likely here", "the pit lane is 3 s longer") is the same search
    with one number changed, not a different code path.
    """
    ev = get_event(event) if isinstance(event, str) else event
    n_laps = ev.n_race_laps
    nd = model.n_draws

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

    # Rank each compound against the *whole* set present this weekend.
    # `hardness_rank` ranks densely within whatever list it is handed, so it
    # must be called once on all compounds, not per-stint on a list of one.
    _rank = dict(zip(model.compounds, hardness_rank(list(model.compounds))))

    max_len = min(n_laps, max(caps.values()))
    tables = {}          # push -> cost dict, float32
    means = {}           # push -> {compound: (n_start+1, max_len+1)}
    for p in push_grid:
        t = stint_cost_table(model, ev, max_len, p, warmup_s=warmup_s)
        tables[p] = {c: v.astype(np.float32) for c, v in t.items()}
        means[p] = {c: v.mean(0) for c, v in t.items()}
        del t

    # -- phase 1: cost every plan at every push, keep the best push ---------
    fam_mean, fam_push, n_total = [], [], 0
    dens = traffic_density(ev)
    for seq, lens, starts in zip(seqs, lens_all, starts_all):
        pits = starts[:, 1:]
        # Terms that do not depend on the tyre model: pit lane, traffic, the
        # safety-car option value.
        fixed = np.full(len(lens), (len(seq) - 1) * pit_loss_s, dtype=np.float64)
        if pits.shape[1]:
            idx = np.clip(pits, 1, n_laps) - 1
            fixed += TRAFFIC_LAPS_PER_STOP * traffic_s_per_lap * dens[idx].sum(1)
            last = pits.max(1).astype(float)
            fixed -= ((1.0 - np.exp(-sc_rate * last))
                      * (1.0 - SC_PIT_LOSS_FRACTION) * pit_loss_s)
        # The opening stint starts from the grid, not from a pit exit: cold
        # tyres, a standing start, and the densest traffic of the race.  A
        # harder compound is worse at all three, and that cost is paid once.
        grid = grid_penalty_s * float(_rank[seq[0]])
        best = None
        for p in push_grid:
            t = fixed + grid
            for k, c in enumerate(seq):
                t += means[p][c][starts[:, k], lens[:, k]]
            if best is None:
                best, bp = t, np.full(len(lens), p, dtype=float)
            else:
                take = t < best
                best = np.where(take, t, best)
                bp = np.where(take, p, bp)
        fam_mean.append(best)
        fam_push.append(bp)
        n_total += len(lens)

    flat = np.concatenate(fam_mean)
    flat_push = np.concatenate(fam_push)
    keep = min(shortlist, len(flat))
    order = np.argpartition(flat, keep - 1)[:keep]
    # The best plan at each stop count is always scored, even when a shortlist
    # taken purely by mean cost would be swept clean by one stop count.  "The
    # one-stop is 9 s slower" is a far more useful sentence than a table with
    # no one-stop in it at all.
    bounds = np.cumsum([0] + [len(x) for x in fam_mean])
    stops_flat = np.concatenate([
        np.full(len(x), len(seq) - 1) for x, seq in zip(fam_mean, seqs)])
    forced = [int(np.flatnonzero(stops_flat == n)[np.argmin(flat[stops_flat == n])])
              for n in np.unique(stops_flat)]
    # ...and the best plan for each *starting* compound, for the same reason:
    # the grid-position question a strategist asks first is "what if we start
    # on the hard", and the answer should be a number, not an absence.
    first_flat = np.concatenate([np.full(len(x), _rank[seq[0]]) for x, seq in zip(fam_mean, seqs)])
    forced += [int(np.flatnonzero(first_flat == r)[np.argmin(flat[first_flat == r])])
               for r in np.unique(first_flat)]
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
        })

    mean = times.mean(1)
    best_idx = int(np.argmin(mean))
    winner = np.argmin(times, axis=0)
    win_counts = np.bincount(winner, minlength=keep)

    tbl = pd.DataFrame(rows)
    tbl["mean_s"] = mean
    # The decision-relevant spread is the spread of the *difference* against
    # the best plan, draw by draw, not the spread of each plan's absolute cost.
    # Every plan is scored on the same posterior draws, so most of the absolute
    # uncertainty - how fast this tyre degrades at all - is common to all of
    # them and cancels in the comparison.  Reporting the marginal spread next
    # to a 0.2 s delta invites reading a +/-40 s band as decision uncertainty
    # when the plans are in fact ranked identically on almost every draw.
    delta = times - times[best_idx][None, :]
    tbl["delta_s"] = delta.mean(1)
    tbl["delta_p05"] = np.quantile(delta, 0.05, axis=1)
    tbl["delta_p95"] = np.quantile(delta, 0.95, axis=1)
    tbl["p05_s"] = np.quantile(times, 0.05, axis=1)
    tbl["p95_s"] = np.quantile(times, 0.95, axis=1)
    tbl["win_prob"] = win_counts / nd
    tbl["p_beat_best"] = (delta < 0).mean(1)
    tbl.loc[best_idx, "p_beat_best"] = np.nan   # a plan does not beat itself
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
    return StrategyResult(
        table=tbl, pit_loss_s=pit_loss_s, n_draws=nd,
        n_strategies=int(n_total), n_scored=keep, max_stint=caps,
        best_label=top["strategy"],
        best={"compounds": top["compounds"].split("-"),
              "pit_laps": list(top["pit_laps"]),
              "stint_lens": list(top["stint_lens"]),
              "n_stops": int(top["n_stops"]),
              "push": float(top["push"])},
        regime=(rg.as_dict() if rg else {}), warmup_s=warmup_s,
        life=compound_life(model, ev, push=float(top["push"]), cap=caps, support=support),
        by_stops=by_stops, push_grid=tuple(push_grid),
        implied_regime=float(wear_multiplier(float(top["push"]))), model=model,
        times=times)


def best_by_start(res: StrategyResult) -> pd.DataFrame:
    """The best plan for each starting compound, with its cost against the
    overall best.  Track position is not in the objective, so this is the
    table a strategist reads when the grid, not the tyre, decides the start."""
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


# --------------------------------------------------------------------------
# Pit window: where the decision is actually loose
# --------------------------------------------------------------------------


def pit_window(fit, event: Event | str, plan: dict, pit_loss_s: float, *,
               regime: RegimeFactor | float | None = None,
               n_draws: int = MC_DRAWS, margin: int = PIT_WINDOW_MARGIN,
               warmup_s: float = OUT_LAP_PENALTY_S,
               max_stint: int | dict | None = None, tolerance_s: float = 1.0,
               push: float | None = None,
               budget: float = GRIP_BUDGET_S, seed: int = 0) -> pd.DataFrame:
    """`pit_window_model` on a subsample of a sealed fit's draws."""
    ev = get_event(event) if isinstance(event, str) else event
    if not plan:
        return pd.DataFrame()
    total = fit.posterior["lin"].shape[0]
    rng = np.random.default_rng(seed)
    draws = rng.choice(total, size=min(n_draws, total), replace=False)
    model = TyreModel.from_fit(fit, draws=draws, budget=budget)
    return pit_window_model(model, ev, plan, pit_loss_s, margin=margin, warmup_s=warmup_s,
                            max_stint=max_stint, tolerance_s=tolerance_s, push=push)


def pit_window_model(model: TyreModel, event: Event | str, plan: dict, pit_loss_s: float, *,
                     margin: int = PIT_WINDOW_MARGIN, warmup_s: float = OUT_LAP_PENALTY_S,
                     max_stint: int | dict | None = None, tolerance_s: float = 1.0,
                     push: float | None = None, sc_rate: float = SC_RATE_PER_LAP,
                     traffic_s_per_lap: float = DIRTY_AIR_S_PER_LAP) -> pd.DataFrame:
    """Cost of moving one stop earlier or later, holding the others fixed.

    A ranking that says "pit on lap 20" is not actionable on its own: on the
    day the car is in traffic, or the stop is a lap late, and what the wall
    needs to know is how much that costs.  Sweeping each stop lap one at a time
    turns the single recommended lap into a *window* - the laps within
    `tolerance_s` of optimal - which is the form the decision is made in.

    The push level is held at the plan's own, so the window answers "when
    should I stop, running the race I intend to run" rather than silently
    re-optimising the driving to suit each candidate lap.
    """
    ev = get_event(event) if isinstance(event, str) else event
    n_laps = ev.n_race_laps
    if not plan:
        return pd.DataFrame()
    nd = model.n_draws
    p = float(plan.get("push", 1.0) if push is None else push)
    caps = life_caps(model, ev, push=p)
    if isinstance(max_stint, dict):
        caps = {c: min(v, max_stint.get(c, v)) for c, v in caps.items()}
    max_len = min(n_laps, max(caps.values()))
    cost = stint_cost_table(model, ev, max_len, p, warmup_s=warmup_s)

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
    # Re-reference each stop to its own best lap, so "0" means "the best lap
    # for this stop" even where the plan's other stops are what is suboptimal.
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
                    push: float = 0.85, budget: float = GRIP_BUDGET_S,
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
    """Per-lap undercut gain as a function of the leader's tyre age.

    If you pit on lap L and the car ahead stays out one more lap, on that lap
    you run a fresh tyre while they run an aged one.  The time you take out of
    them is the pace they are losing to wear, less the cold-tyre cost of your
    out-lap and less whatever your new compound gives away on pace.

    The gain is computed against the *marginal* loss the leader is carrying -
    the extra second of lap time their next lap costs over their last - not
    against their cumulative loss since the stint began.  Those are different
    quantities and only the marginal one is what an undercut takes out of
    them; charging the cumulative figure is what makes an undercut window open
    several laps too early.

    Because the loss curve steepens sharply past the cliff, the window opens
    slowly and then all at once, which is the shape race engineers describe.
    """
    ages = np.arange(1, max_age + 1, dtype=float)
    psi = float(wear_multiplier(push))

    # Marginal pace loss on the leader's next lap, at each current age.
    w_old = model.wear_rate[compound_old][:, None] * psi * ages[None, :]
    step = model.wear_rate[compound_old][:, None] * psi
    marginal = grip_loss(w_old + step, budget=model.budget) - grip_loss(w_old, budget=model.budget)
    # ...and what your own fresh tyre is already losing on its first lap.
    w_new = model.wear_rate[compound_new][:, None] * psi
    fresh = grip_loss(w_new, budget=model.budget) - grip_loss(np.zeros_like(w_new), budget=model.budget)

    off = (model.pace_offset[compound_new] - model.pace_offset[compound_old])[:, None]
    gain = marginal - fresh - out_lap_penalty - off
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

    The attacker gains, each lap, the pace the defender's ageing tyre keeps
    losing plus the compound difference, less what its own fresh tyre loses
    and, once, the cold first lap.  The undercut works when that cumulative
    gain exceeds the gap.  Reported per lap of exposure as a probability over
    the model's draws - the same arithmetic the live engine runs on the day,
    made available before it.

    Both directions are answered from the one call: `attacker="me"` is my
    undercut on the car ahead, and the same numbers read the other way are the
    car behind's undercut on me.  The overcut - staying out while they pit -
    is the mirror image: it works on the laps where this gain is *negative*.
    """
    psi = float(wear_multiplier(push))
    load = 1.0
    if event is not None and lap_now is not None:
        ev = get_event(event) if isinstance(event, str) else event
        lp = load_profile(ev)
        load = float(lp[min(max(int(lap_now), 1), ev.n_race_laps) - 1])
    ks = np.arange(1, laps + 1, dtype=float)
    rate_d = model.wear_rate[their_compound] * psi * load
    rate_n = model.wear_rate[new_compound] * psi * load
    w0 = model.wear_rate[their_compound] * psi * float(their_age)
    loss0 = grip_loss(w0, budget=model.budget)
    def_loss = grip_loss(w0[:, None] + rate_d[:, None] * ks[None, :], budget=model.budget) - loss0[:, None]
    att_loss = grip_loss(rate_n[:, None] * ks[None, :], budget=model.budget)
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


def counterfactual(fit, event: Event | str, race: pd.DataFrame,
                   pit_loss_s: float, *, margin: int = PIT_WINDOW_MARGIN,
                   regime: RegimeFactor | float | None = None,
                   warmup_s: float = OUT_LAP_PENALTY_S,
                   max_stint: int | dict | None = None, push: float = 0.7,
                   budget: float = GRIP_BUDGET_S, n_draws: int = 200) -> pd.DataFrame:
    """For each driver: what their actual stop laps cost against the best
    alternative *with the same compounds and the same number of stops*.

    Holding the compound sequence fixed is deliberate - it isolates the timing
    decision, which is the one the pit wall actually owns on the day.

    Every driver is scored.  The old version skipped anyone whose race included
    a stint longer than the model's cap, which quietly dropped exactly the
    drivers whose strategy was most worth examining; the wear model is defined
    at any age, so a long stint is now costed rather than excluded, and its
    wear state is reported so a reader can see when a plan was past the cliff.
    """
    ev = get_event(event) if isinstance(event, str) else event
    n_laps = ev.n_race_laps
    total = fit.posterior["lin"].shape[0]
    idx = np.linspace(0, total - 1, min(n_draws, total)).astype(int)
    model = TyreModel.from_fit(fit, draws=idx, budget=budget)
    max_len = n_laps
    cost = stint_cost_table(model, ev, max_len, push, warmup_s=warmup_s)

    def total_cost(seq, lens):
        pits = list(np.cumsum(lens)[:-1])
        t = np.full(len(idx), (len(seq) - 1) * pit_loss_s, dtype=float)
        t += traffic_cost(ev, pits)
        t -= safety_car_credit(ev, pits, pit_loss_s)
        start = 0
        for c, L in zip(seq, lens):
            if c not in cost or L <= 0 or L > max_len:
                return None
            t += cost[c][:, int(start), int(L)]
            start += int(L)
        return t

    rows = []
    for drv, g in race.groupby("driver"):
        g = g.sort_values("lap_number")
        stints = (g.groupby("stint")
                  .agg(compound=("compound", "first"),
                       start=("lap_number", "min"),
                       end=("lap_number", "max"))
                  .sort_values("start").reset_index())
        stints = stints[stints["compound"].isin(model.compounds)]
        if len(stints) < 2:
            continue
        seq = stints["compound"].tolist()
        actual_pits = stints["start"].to_numpy()[1:].astype(int)
        lens = np.diff(np.concatenate([[0], actual_pits, [n_laps]])).astype(int)
        if lens.min() <= 0:
            continue
        base = total_cost(seq, lens)
        if base is None:
            continue
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        wear = max(float(model.wear_at(c, [int(L)], [int(st)], push, ev).mean())
                   for c, L, st in zip(seq, lens, starts))

        # Search the same compound sequence over all legal pit-lap sets.
        best, best_pits = base, tuple(actual_pits)
        grid = range(margin, n_laps - margin + 1)
        n_stops = len(actual_pits)
        for pits in itertools.combinations(grid, n_stops):
            if any(b - a < margin for a, b in zip(pits, pits[1:])):
                continue
            l2 = np.diff(np.concatenate([[0], list(pits), [n_laps]])).astype(int)
            if l2.min() < margin or l2.max() > max_len:
                continue
            c2 = total_cost(seq, l2)
            if c2 is not None and c2.mean() < best.mean():
                best, best_pits = c2, pits

        delta = base - best
        rows.append({
            "driver": drv,
            "compounds": "-".join(seq),
            "actual_pit_laps": list(map(int, actual_pits)),
            "model_pit_laps": list(map(int, best_pits)),
            "loss_s": float(delta.mean()),
            "lo": float(np.quantile(delta, 0.05)),
            "hi": float(np.quantile(delta, 0.95)),
            "p_loss_positive": float((delta > 0).mean()),
            "max_wear": wear,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("loss_s", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Evaluating *given* plans: the strategist's own candidates, under scenarios
# --------------------------------------------------------------------------
#
# `simulate_model` answers "what is the best plan?".  A strategist also needs
# the other direction: "here are the two or three plans we are actually
# choosing between - what does each cost, where do they cross, and what
# happens to that if degradation runs hot or the pit lane is slow?"  The
# functions below score plans handed to them, on the same posterior draws and
# with the same cost terms as the search, so a plan built by hand is priced
# exactly as the optimiser would price it.


def scale_model(model: TyreModel, deg_mult: float = 1.0, pace_mult: float = 1.0) -> TyreModel:
    """The same draws with every wear rate scaled - a degradation scenario.

    Multiplying every compound by one factor is what a hotter track, a
    greener surface or a heavier car does to first order (the thermal
    sensitivity measured on the archive is +2.5% per degree, so x1.4 is
    roughly a 13 degree hotter race day); it keeps the compound ordering and
    the draw-to-draw correlation intact, which is what makes plans scored on
    the scaled model comparable with the base case.
    """
    if deg_mult == 1.0 and pace_mult == 1.0:
        return model
    return TyreModel(
        compounds=list(model.compounds),
        wear_rate={c: model.wear_rate[c] * float(deg_mult) for c in model.compounds},
        pace_offset={c: model.pace_offset[c] * float(pace_mult) for c in model.compounds},
        budget=model.budget, load_exponent=model.load_exponent, n_draws=model.n_draws,
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
    psi = float(wear_multiplier(push))
    inc = model.wear_rate[compound][:, None] * psi * lf[laps][None, :]
    w = np.cumsum(inc, axis=1)
    loss = grip_loss(w - 0.5 * inc, budget=model.budget)
    loss = loss + model.pace_offset[compound][:, None] + float(manage_cost(push))
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
                   margin: int = PIT_WINDOW_MARGIN) -> tuple:
    """Price a list of plans on every draw, with a lap-by-lap trace for each.

    Each plan is `{"compounds": [...], "pit_laps": [...], "push": optional,
    "label": optional}`.  Returns `(table, details)`: the table has one row
    per plan with its expected cost, its expected loss against the best plan
    *in this list* (draw-by-draw quantiles, so the shared uncertainty
    cancels), the share of draws on which it is the fastest of the list, the
    wear each stint ends at, and any rule it bends; `details[i]` carries the
    per-draw times and the cumulative cost trace for plotting.

    Plans that break the two-compound rule, exceed the allocation or run a
    stint past its circuit cap are priced anyway and *flagged* - the
    strategist may know something the caps do not - but a plan whose stop
    laps are not in order is not a plan and is returned with no numbers.
    """
    ev = get_event(event) if isinstance(event, str) else event
    n = ev.n_race_laps
    nd = model.n_draws
    dens = traffic_density(ev)
    rank = dict(zip(model.compounds, hardness_rank(list(model.compounds))))
    alloc = allocation or {}
    caps = stint_cap or {}
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

        def cost_at(p_):
            per_lap = np.zeros((nd, n))
            for k, (c, L, s) in enumerate(zip(seq, lens, starts)):
                per_lap[:, s:s + L] = stint_lap_losses(model, ev, c, int(s), int(L), p_, warmup_s=warmup_s)
            for p_lap in pits:
                per_lap[:, p_lap - 1] += pit_loss_s + TRAFFIC_LAPS_PER_STOP * traffic_s_per_lap * dens[p_lap - 1]
            per_lap[:, 0] += grid_penalty_s * float(rank[seq[0]])
            credit = safety_car_credit(ev, pits, pit_loss_s, rate=sc_rate)
            return per_lap, credit

        p_use = pl.get("push", push)
        if p_use is None:
            best = None
            for p_ in push_grid:
                per_lap, credit = cost_at(float(p_))
                m = per_lap.sum(1).mean() - credit
                if best is None or m < best[0]:
                    best = (m, float(p_), per_lap, credit)
            _, p_use, per_lap, credit = best
        else:
            p_use = float(p_use)
            per_lap, credit = cost_at(p_use)
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
            "trace_mean": cum.mean(0), "trace_lo": np.quantile(cum, 0.05, axis=0),
            "trace_hi": np.quantile(cum, 0.95, axis=0), "per_lap_mean": per_lap.mean(0),
            "wear_end_mean": [float(w.mean()) for w in wear_end],
            "wear_end_p90": [float(np.quantile(w, 0.9)) for w in wear_end],
        })
        rows.append({"plan": label, "n_stops": len(pits), "compounds": "-".join(seq), "pit_laps": pits,
                     "stint_lens": [int(x) for x in lens], "push": p_use, "valid": True,
                     "mean_s": float(times.mean()),
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
    """At what degradation multiplier does plan B overtake plan A?

    The one number that turns a pre-race choice into a race-day rule: the live
    engine measures each car's degradation as a multiple of the model's rate,
    so "switch to the two-stop if the live multiplier passes 1.35" is a trigger
    the wall can act on without re-running anything.
    """
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
    """Which uncertainty the decision actually hinges on.

    The expected value of perfect information (EVPI) is the time a strategist
    would save, on average, by knowing the true degradation before choosing:
    the gap between the best plan on the posterior mean and the best plan on
    each draw.  Splitting the draws by each compound's own rate and asking how
    much of that gap knowing *that* compound would close gives a per-compound
    share - and that is the practice programme: the compound with the largest
    share is the one whose long run is worth the most.
    """
    if res.times is None or res.table.empty or res.model is None:
        return {}
    T = res.times
    mean = T.mean(1)
    i_star = int(np.argmin(mean))
    best_per_draw = T.min(0)
    evpi = float(np.mean(T[i_star] - best_per_draw))
    out = {"evpi_s": evpi, "best": str(res.table.iloc[i_star]["strategy"]), "by_compound": {}}
    for c in res.model.compounds:
        r = res.model.wear_rate[c] * res.model.budget
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
    left - the planned remaining sequence with its stops re-optimised, or one
    stop fewer if the new tyre can go to the flag.  The answer is a lap-by-lap
    verdict with the seconds at stake and the share of draws that agree, which
    is the sheet a strategist wants taped to the desk before the lights go out
    rather than computed under a yellow flag.

    Only the decision *this lap* is priced: the safety car is treated as an
    opportunity that exists now, not as a window whose length is known.
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
            # 0 further stops
            if R >= margin and cap_ok(c_new, R) and legal_seq(used, [c_new]):
                cands.append((float(means[c_new][L, R]), [c_new], [], c_new))
            # 1 further stop
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
            # 2 further stops - only if the plan itself still has them
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
        # per-draw cost of the chosen continuation
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
