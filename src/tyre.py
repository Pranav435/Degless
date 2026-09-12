"""The tyre model: a grip budget, a wear clock, and a push level the driver chooses.

This module replaces the previous "linear slope + softplus hinge, multiplied by
an exogenous practice->race regime constant" formulation.  That formulation had
three defects that between them produced the strategies this project was
getting wrong, and all three come from the same root cause: it treated
*degradation* as a property of the tyre alone, when it is really the outcome of
a decision the driver is making every lap.

    1. The cliff was unidentified.  A practice long run stops at age 15-22 laps,
       before any compound reaches its cliff, so the hinge posterior was just
       its prior and the fitted curve was effectively linear.  A linear
       degradation curve has no tyre-imposed optimum stint length at all - the
       only thing setting stint length is the pit-loss trade-off, which is why
       the optimiser was happy to run a SOFT for 33 laps.

    2. The regime factor was an exogenous constant.  "Race stints degrade at
       0.4x the practice rate" is a description of an *equilibrium*, not a law.
       It is what you get when a driver chooses to manage the tyre; it is not
       something the tyre does.  Bolting it on as a constant means the model
       cannot answer the question that actually decides a strategy: is it worth
       managing this tyre to make it last, or pushing it and stopping again?

    3. Nothing connected degradation rate to tyre life.  The two were fitted as
       independent parameters (`lin` and `knee`), so the model could believe a
       compound both degraded quickly and lasted a long time.

**The grip budget.**  The invariant this module is built on is that a tyre
reaches its cliff after it has surrendered a roughly fixed amount of lap time,
not after a fixed number of laps.  A tyre is a finite quantity of rubber in a
working temperature window; the cliff is what happens when the tread is worn
through and the carcass starts overheating, and by then the tyre is some 2-4 s
off its fresh pace regardless of which compound it is or how quickly it got
there.  Measured on Barcelona 2026 race stints - degradation rate from a
two-way fixed-effects fit, stint length from the longest stint each compound
was actually run to:

        SOFT    0.186 s/lap x 21 laps = 3.91 s
        MEDIUM  0.155 s/lap x 26 laps = 4.03 s
        HARD    0.110 s/lap x 31 laps = 3.40 s

Three compounds with a 1.7x spread in degradation rate and a 1.5x spread in
stint length agree on the budget to within +/-9%.  That is the invariant.
Across the seven scored 2026 weekends the same quantity runs 1.9-3.8 s, so the
budget is now fitted per compound by `scripts/80_recalibrate.py` from every
scored race but the target's and carried on the model as `budgets`; the
`GRIP_BUDGET_S` constant is the fallback.

The consequence is that tyre life is *derived*, not fitted:

        life_c = budget_c / degradation_rate_c

which forces the ordering the old model kept getting backwards - a compound
cannot both degrade quickly and last a long time - and gives the cliff a
location on a compound that was never run near its cliff in practice.  Where
that quotient exceeds the race distance the honest statement is "longer than
the race", and `life_table` says so rather than printing 2,900 laps.

**The push level.**  A driver can trade lap time for tyre life: lift and coast
into the braking zones, short-shift, roll more speed through the middle of the
corner instead of attacking the entry.  All of it reduces the energy going
through the contact patch, and all of it costs lap time.  So a stint is
described by a *pair* - how long, and how hard - and the strategy optimiser
must choose both.

    push p in [0, 1]:  p = 1 is a practice long run (full attack, which is what
    a long run is *for*); p < 1 is management.

    lap time cost of managing:  MANAGE_COST_S * (1 - p)
    wear rate multiplier:       wear_multiplier(p), rising from
                                MANAGE_WEAR_FLOOR at p=0 to 1.0 at p=1

This is what makes the model self-checking.  The old code *assumed* a regime
factor of 0.40.  This model instead *predicts* one: it is whatever
`wear_multiplier(p*)` comes out to at the optimiser's chosen push, and it can
be compared against the measured value.

**Fuel load.**  Wear accumulates faster on a heavy car, because wear is driven
by the energy through the contact patch and that scales with vertical load.
The exponent here is ~1.6 (see `config.TYRE_LOAD_EXPONENT`), a load sensitivity
rather than the 5.0 the previous version used - 5.0 was fitted by regressing
stint degradation on car mass across race phase, which is confounded with track
evolution and with the fact that teams conventionally run softer compounds
early, and it is not a number any tyre physics supports.

Everything here is vectorised over posterior draws so the strategy module can
carry uncertainty through without a Python loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import (
    CLIFF_EXPONENT,
    CLIFF_SHARPNESS,
    CLIFF_SMOOTH,
    GRIP_BUDGET_S,
    MANAGE_COST_S,
    MANAGE_WEAR_FLOOR,
    MANAGE_WEAR_EXPONENT,
    PUSH_GRID,
    TYRE_LOAD_EXPONENT,
    Event,
    get_event,
)

__all__ = [
    "wear_multiplier",
    "manage_cost",
    "grip_loss",
    "load_profile",
    "TyreModel",
]


# --------------------------------------------------------------------------
# The push <-> wear trade-off
# --------------------------------------------------------------------------


def wear_multiplier(push: np.ndarray | float,
                    *, floor: float = MANAGE_WEAR_FLOOR,
                    exponent: float = MANAGE_WEAR_EXPONENT) -> np.ndarray:
    """How fast the tyre wears at push level `p`, relative to full attack.

        wear(1) = 1.0     a practice long run, which is what the curve is fitted on
        wear(0) = floor   maximum management

    Convex in `p` (`exponent` > 1): the first increment of management is worth
    a lot of tyre life and costs little lap time, and the returns fall off
    after that.  That convexity is what gives the optimiser an interior
    optimum rather than pushing it to one extreme.
    """
    p = np.clip(np.asarray(push, dtype=float), 0.0, 1.0)
    return floor + (1.0 - floor) * p ** float(exponent)


def manage_cost(push: np.ndarray | float, *, cost_s: float = MANAGE_COST_S) -> np.ndarray:
    """Lap time given up, per lap, to run at push level `p`.

    Linear: lifting and coasting earlier in the braking zone costs time roughly
    in proportion to how much earlier you lift.  The convexity in the trade-off
    lives in `wear_multiplier`, not here.
    """
    p = np.clip(np.asarray(push, dtype=float), 0.0, 1.0)
    return float(cost_s) * (1.0 - p)


# --------------------------------------------------------------------------
# Wear -> pace loss
# --------------------------------------------------------------------------


def grip_loss(wear: np.ndarray, *, budget: float = GRIP_BUDGET_S,
              kappa: float = CLIFF_SHARPNESS, q: float = CLIFF_EXPONENT,
              smooth: float = CLIFF_SMOOTH) -> np.ndarray:
    """Pace loss, in seconds, at wear state `w` (1.0 = the cliff).

        loss(w) = budget * [ w + kappa * softmax(w - 1) ** q ]

    Below the cliff this is linear in wear and reaches exactly `budget` at
    w = 1, which is the definition of the grip budget: the tyre has by then
    given up its whole allowance of lap time.  Past the cliff the second term
    takes over and the loss runs away - at w = 1.5 the tyre is about 3x its
    pre-cliff rate, which is the point at which a real stint is over.

    `softmax` here is a softplus rather than a hard `max(0, .)` so the function
    is smooth, which keeps the optimiser's cost surface differentiable and
    stops a plan flipping discontinuously as a stint crosses the knee by one
    lap.
    """
    w = np.asarray(wear, dtype=float)
    s = float(smooth)
    excess = s * np.logaddexp(0.0, (w - 1.0) / s)   # smooth max(0, w-1)
    return float(budget) * (w + float(kappa) * excess ** float(q))


# --------------------------------------------------------------------------
# Fuel load
# --------------------------------------------------------------------------


def load_profile(event: Event | str, *, exponent: float = TYRE_LOAD_EXPONENT) -> np.ndarray:
    """Per-race-lap wear multiplier from fuel mass, normalised at mid-race mass.

    Length `n_race_laps`, indexed by lap-1.  Normalised so it averages ~1 over
    the race: the load term redistributes wear between the start and the end of
    a race rather than inflating the total, which is what makes it a *shape*
    correction and keeps it from double-counting the degradation level.
    """
    ev = get_event(event) if isinstance(event, str) else event
    from src.fuel import mass_profile

    m = mass_profile(ev)
    return (m / ev.m_total_kg) ** float(exponent)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


def _budget_map(budget, compounds: list) -> tuple:
    """`(pooled float, {compound: float})` from a float or a per-compound dict."""
    if isinstance(budget, dict):
        vals = {c: float(budget.get(c, budget.get("_pooled", GRIP_BUDGET_S))) for c in compounds}
        pooled = float(budget.get("_pooled", np.mean(list(vals.values())) if vals else GRIP_BUDGET_S))
        return pooled, vals
    b = float(budget)
    return b, {c: b for c in compounds}


@dataclass
class TyreModel:
    """A calibrated tyre model: per-compound wear rate, life and pace offset.

    `wear_rate[c]` has shape (draws,) and is the fraction of the tyre's life
    consumed per lap at full push and reference (mid-race) fuel load.  It is
    the reciprocal of the compound's full-push life in laps, and it comes from
    the practice degradation fit divided by the grip budget - which is exactly
    the statement "life = budget / rate" rearranged.

    `pace_offset[c]` has shape (draws,) and is the compound's intrinsic pace
    deficit in s/lap relative to the softest compound, before any degradation.

    `budgets[c]` is the compound's grip budget in seconds; `budget` is the
    pooled value kept for callers that need one number.
    """

    compounds: list = field(default_factory=list)
    wear_rate: dict = field(default_factory=dict)    # compound -> (draws,)
    pace_offset: dict = field(default_factory=dict)  # compound -> (draws,)
    budget: float = GRIP_BUDGET_S
    load_exponent: float = TYRE_LOAD_EXPONENT
    n_draws: int = 0
    source: str = ""
    budgets: dict = field(default_factory=dict)      # compound -> seconds
    # per-draw additive slope deviation per driver and compound, s/lap per lap
    # of age, from the practice fit (`dev[d, c]`); empty when not carried
    driver_dev: dict = field(default_factory=dict)   # driver -> {compound: (draws,)}
    # the management trade-off this model runs with (recalibrated per weekend)
    manage_floor: float = MANAGE_WEAR_FLOOR
    manage_cost_s: float = MANAGE_COST_S

    def __post_init__(self):
        if not self.budgets:
            self.budgets = {c: float(self.budget) for c in self.compounds}
        else:
            self.budgets = {c: float(self.budgets.get(c, self.budget)) for c in self.compounds}

    def psi(self, push: float) -> float:
        """Wear multiplier at `push` under this model's management trade-off."""
        return float(wear_multiplier(push, floor=self.manage_floor))

    def mcost(self, push: float) -> float:
        """Lap-time cost of managing at `push` under this model's trade-off."""
        return float(manage_cost(push, cost_s=self.manage_cost_s))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_fit(cls, fit, *, draws: np.ndarray | None = None,
                 budget=GRIP_BUDGET_S,
                 load_exponent: float = TYRE_LOAD_EXPONENT,
                 rate_floor: dict | None = None,
                 carry_drivers: bool = True,
                 manage_floor: float = MANAGE_WEAR_FLOOR,
                 manage_cost_s: float = MANAGE_COST_S) -> "TyreModel":
        """Build from a `BayesFit` of *practice* long runs.

        A practice long run is a full-push experiment by construction - that is
        what the run is for - so the fitted slope is the p = 1 degradation rate
        and needs no regime correction.  The regime factor is not applied here
        and is not applied anywhere else either; it is *derived* later from the
        push level the strategy optimiser chooses.

        The rate is read off the fitted curve as its average slope over the
        span practice actually supports, rather than from the `lin` parameter
        alone, so that whatever curvature the fit did find is included.

        `budget` may be one number or a per-compound dict (the recalibrated
        values).  `rate_floor[c]`, in s/lap at full push, is the smallest rate
        the model is allowed to believe for a compound - the circuit's own race
        history says a tyre never degrades slower than that here - and it is
        what stops a flat practice fit deriving a 2,900-lap tyre.
        """
        total = fit.posterior["lin"].shape[0]
        idx = np.arange(total) if draws is None else np.asarray(draws)
        span = np.array([1.0, 10.0])
        pooled, bmap = _budget_map(budget, list(fit.compounds))
        wear, pace = {}, {}
        for c in fit.compounds:
            d = fit.deg_loss(c, span)[idx]                 # (nd, 2)
            rate = (d[:, 1] - d[:, 0]) / (span[1] - span[0])
            # A non-positive fitted slope is physically impossible and would
            # make life infinite; floor it well below any real measurement,
            # and at the circuit's historical minimum where one is known.
            floor = max(1e-3, float((rate_floor or {}).get(c, 0.0) or 0.0))
            rate = np.maximum(rate, floor)
            wear[c] = rate / bmap[c]
            j = fit.compounds.index(c)
            pace[c] = fit.posterior["comp_offset"][idx, j]
        dev = {}
        if carry_drivers and "dev" in fit.posterior and getattr(fit, "drivers", None):
            D = np.asarray(fit.posterior["dev"])           # (draws, n_drv, n_comp)
            if D.ndim == 3 and D.shape[1] == len(fit.drivers) and D.shape[2] == len(fit.compounds):
                for i, drv in enumerate(fit.drivers):
                    dev[drv] = {c: D[idx, i, j] for j, c in enumerate(fit.compounds)}
        return cls(compounds=list(fit.compounds), wear_rate=wear, pace_offset=pace,
                   budget=pooled, budgets=bmap, load_exponent=float(load_exponent),
                   n_draws=len(idx), source="practice long runs (full push)", driver_dev=dev,
                   manage_floor=float(manage_floor), manage_cost_s=float(manage_cost_s))

    def copy_with(self, *, wear_rate: dict | None = None, pace_offset: dict | None = None,
                  source: str | None = None, n_draws: int | None = None,
                  driver_dev: dict | None = None, budgets: dict | None = None,
                  manage_floor: float | None = None, manage_cost_s: float | None = None) -> "TyreModel":
        """The same model with some arrays replaced; budgets and exponent carried."""
        wr = wear_rate if wear_rate is not None else {c: self.wear_rate[c].copy() for c in self.compounds}
        po = pace_offset if pace_offset is not None else {c: self.pace_offset[c].copy() for c in self.compounds}
        b = dict(self.budgets) if budgets is None else {c: float(budgets.get(c, self.budget)) for c in self.compounds}
        return TyreModel(compounds=list(self.compounds), wear_rate=wr, pace_offset=po,
                         budget=(self.budget if budgets is None else float(np.mean(list(b.values())))),
                         budgets=b, load_exponent=self.load_exponent,
                         n_draws=(self.n_draws if n_draws is None else int(n_draws)),
                         source=(self.source if source is None else source),
                         driver_dev=(self.driver_dev if driver_dev is None else driver_dev),
                         manage_floor=(self.manage_floor if manage_floor is None else float(manage_floor)),
                         manage_cost_s=(self.manage_cost_s if manage_cost_s is None else float(manage_cost_s)))

    def subsample(self, idx: np.ndarray, *, source: str | None = None) -> "TyreModel":
        idx = np.asarray(idx, dtype=int)
        dev = {d: {c: v[idx] for c, v in dc.items()} for d, dc in self.driver_dev.items()}
        return self.copy_with(wear_rate={c: self.wear_rate[c][idx] for c in self.compounds},
                              pace_offset={c: self.pace_offset[c][idx] for c in self.compounds},
                              n_draws=len(idx), source=source, driver_dev=dev)

    def for_driver(self, driver: str, *, race_factor: float = 1.0, practice_dev: bool = True) -> "TyreModel":
        """This model as it applies to one car.

        Two per-car terms.  `race_factor` is the driver's multiplicative rate
        factor measured on previous 2026 races (how much harder or gentler
        than the field this car is on its tyres); the practice fit's own
        per-driver, per-compound slope deviation `dev[d, c]` is added where
        the driver ran long runs this weekend.  Both are shrunk toward the
        field by their estimators, so an unknown driver gets the field model.
        """
        wr = {}
        dd = self.driver_dev.get(driver, {}) if practice_dev else {}
        for c in self.compounds:
            rate = self.wear_rate[c] * self.budgets[c]
            if c in dd:
                rate = np.maximum(rate + dd[c], 1e-3)
            wr[c] = rate * float(race_factor) / self.budgets[c]
        return self.copy_with(wear_rate=wr, source=f"{self.source} [{driver} x{race_factor:.2f}]")

    # -- derived quantities ------------------------------------------------

    def budget_of(self, compound: str) -> float:
        return float(self.budgets.get(compound, self.budget))

    def rate(self, compound: str) -> np.ndarray:
        """Degradation rate at full push, s/lap, per draw."""
        return self.wear_rate[compound] * self.budget_of(compound)

    def life_laps(self, compound: str, push: float = 1.0) -> np.ndarray:
        """Laps to the cliff at push level `p` and reference load, per draw."""
        return 1.0 / (self.wear_rate[compound] * self.psi(push))

    def cost_table(self, event: Event | str, max_len: int, push: float,
                   *, warmup_s: float = 0.0) -> dict:
        """`cost[c][d, s, L]` - seconds an L-lap stint on `c` costs, starting
        after `s` laps have been run, on posterior draw `d`, at push `p`.

        Three terms, all of them per-lap and summed over the stint:

            degradation   grip_loss(wear so far)   - the tyre
            management    manage_cost(p)           - the driver
            compound      pace_offset[c]           - the rubber

        plus a fixed warm-up cost for the stint's first flying lap.

        Terms that every strategy pays identically - the base lap time, the
        race-long fuel *pace* effect - are omitted: they cancel in the ranking
        and carrying them would only make the reported numbers look like race
        times when they are not.

        The `s` axis is what makes stint *order* matter.  Wear accrues faster
        on a heavy car, so an early stint burns more of its grip budget per lap
        than a late one, and the same (compound, length) pair is genuinely a
        different decision depending on where in the race it sits.  This is the
        mechanism that reproduces the pattern every real race shows and an
        order-invariant optimiser cannot: stints get longer as the race goes on.
        """
        ev = get_event(event) if isinstance(event, str) else event
        n_laps = ev.n_race_laps
        nd = self.n_draws
        lf = load_profile(ev, exponent=self.load_exponent)
        psi = self.psi(push)
        mcost = self.mcost(push)

        # Wear multiplier of the a-th lap of a stint that starts after s laps.
        idx = np.clip(np.arange(n_laps + 1)[:, None] + np.arange(1, max_len + 1)[None, :],
                      1, n_laps) - 1
        lap_load = lf[idx]                                   # (n_start+1, max_len)

        out = {}
        lens = np.arange(max_len + 1, dtype=float)
        for c in self.compounds:
            rate = self.wear_rate[c][:, None, None] * psi    # (nd,1,1)
            # Wear state *after* each lap of the stint.
            w = np.cumsum(np.broadcast_to(lap_load[None, :, :], (nd,) + lap_load.shape)
                          * rate, axis=2)                    # (nd, n_start+1, max_len)
            # Pace loss is evaluated at the wear carried *during* the lap, i.e.
            # the midpoint of the lap's wear interval - integrating the loss
            # over the lap rather than sampling it at one end. Sampling at the
            # end over-charges every stint by half a lap of degradation.
            dwear = np.empty_like(w)
            dwear[:, :, 0] = w[:, :, 0]
            dwear[:, :, 1:] = np.diff(w, axis=2)
            loss = grip_loss(w - 0.5 * dwear, budget=self.budget_of(c))
            cum = np.cumsum(loss, axis=2)

            cost = np.zeros((nd, n_laps + 1, max_len + 1), dtype=np.float64)
            cost[:, :, 1:] = cum
            cost += lens[None, None, :] * (self.pace_offset[c][:, None, None] + mcost)
            cost[:, :, 1:] += warmup_s
            out[c] = cost
        return out

    def wear_at(self, compound: str, lengths: np.ndarray, starts: np.ndarray,
                push: float, event: Event | str) -> np.ndarray:
        """Wear state reached at the end of a stint - the feasibility check.

        Returns shape (draws, len(lengths)).  A plan whose stints end well past
        w = 1 is not a plan; it is an extrapolation.
        """
        ev = get_event(event) if isinstance(event, str) else event
        lf = load_profile(ev, exponent=self.load_exponent)
        psi = self.psi(push)
        out = np.zeros((self.n_draws, len(lengths)))
        for i, (L, s) in enumerate(zip(np.asarray(lengths, int), np.asarray(starts, int))):
            laps = np.clip(np.arange(s + 1, s + L + 1), 1, ev.n_race_laps) - 1
            out[:, i] = self.wear_rate[compound] * psi * lf[laps].sum()
        return out

    def life_table(self, event: Event | str, push: float = 1.0, *,
                   caps: dict | None = None, support: dict | None = None):
        """Per-compound life, honestly stated.

        `life_laps` is budget / rate at the given push; `life_capped` is the
        same number bounded by the race distance and by the longest stint the
        circuit's races have supported (`caps`), and `longer_than_race` says
        whether the quotient exceeded the race - the case where the number an
        engineer wants is not "112 laps" but "this tyre is not what limits the
        stint here".
        """
        import pandas as pd

        ev = get_event(event) if isinstance(event, str) else event
        n = int(ev.n_race_laps)
        rows = []
        for c in self.compounds:
            full = self.life_laps(c, 1.0)
            man = self.life_laps(c, push)
            cap = int((caps or {}).get(c, n) or n)
            bound = min(n, cap)
            mean_life = float(man.mean())
            rows.append({
                "compound": c,
                "deg_s_per_lap": float(self.rate(c).mean()),
                "deg_lo": float(np.quantile(self.rate(c), 0.05)),
                "deg_hi": float(np.quantile(self.rate(c), 0.95)),
                "knee_lap": float(min(full.mean(), n)),
                "life_laps": float(min(mean_life, bound)),
                "life_lo": float(min(np.quantile(man, 0.05), bound)),
                "life_hi": float(min(np.quantile(man, 0.95), bound)),
                "life_model_uncapped": mean_life,
                "life_full_push": float(full.mean()),
                "longer_than_race": bool(mean_life >= n),
                "bound_by": ("race distance" if mean_life >= n and n <= cap else
                             "circuit history" if mean_life >= cap else "the tyre"),
                "max_stint_laps": float(cap) if caps else float("nan"),
                "practice_support_laps": float((support or {}).get(c, np.nan)),
                "pace_offset_s": float(self.pace_offset[c].mean()),
                "grip_budget_s": self.budget_of(c),
            })
        return pd.DataFrame(rows)

    def summary(self, event: Event | str, push: float = 1.0):
        """Per-compound table: full-push life, managed life, and the rates."""
        t = self.life_table(event, push)
        return t.rename(columns={"deg_s_per_lap": "deg_s_per_lap_full_push",
                                 "life_full_push": "life_laps_full_push",
                                 "life_laps": "life_laps_at_push"})
