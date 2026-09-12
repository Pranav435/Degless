"""The compound ladder — the second unidentifiable half of the problem.

`src.fuel` exists because tyre age and fuel burn are collinear *within* a
stint.  This module exists for the mirror-image problem *between* stints.

Two things the strategy simulator must know, and practice data cannot tell it:

**1. Which compound is quicker, and by how much.**  A practice long run starts
on an unknown fuel load in an unknown engine mode, so the per-stint intercept
that absorbs those also absorbs the entire compound pace difference.  Fitted
freely on Barcelona 2026 the model returned MEDIUM 0.10 s/lap *quicker* than
SOFT — a physical impossibility, and on a scale of ~0.1 s where the measured
step is ~0.22 s.

**2. Which compound degrades faster.**  The compounds are not run in comparable
conditions in practice: Barcelona 2026 gives 96 clean SOFT laps from short
early runs and 9 HARD laps.  Fitted freely the model returned SOFT degrading
*more slowly* than MEDIUM.

Both answers are physically impossible.  A softer compound has more grip, which
means more energy through the contact patch, which means both a quicker lap and
a faster-wearing tyre.  That ordering is a property of how the tyres are
constructed, not something to be re-discovered from 200 laps of running.

So the ladder is **pinned by construction and sized by the data**: the model is
parameterised in strictly positive steps between adjacent compounds, so the
ordering can never invert, while the magnitude of each step has an informative
prior the likelihood is free to move.  This is the same move the fuel prior
makes, applied to the other confound.

Together these two fixes are what stopped the optimiser recommending two 23-lap
SOFT stints at Barcelona.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import (
    COMPOUND_DEG_RATIO,
    COMPOUND_DEG_RATIO_LN_SD,
    COMPOUND_KNEE_STEP_LAPS,
    COMPOUND_KNEE_STEP_LN_SD,
    COMPOUND_ORDER,
    COMPOUND_PACE_STEP_FRAC,
    COMPOUND_PACE_STEP_REL_SD,
    Event,
    get_event,
)


def hardness_rank(compounds: list) -> np.ndarray:
    """Rank of each compound by hardness: 0 = softest present, increasing.

    Ranks are *dense over the compounds actually present*, so a weekend where
    only MEDIUM and HARD were run gets one step between them rather than two.
    Anything outside `COMPOUND_ORDER` sorts to the hard end.
    """
    order = {c: i for i, c in enumerate(COMPOUND_ORDER)}
    keyed = sorted(range(len(compounds)),
                   key=lambda j: order.get(compounds[j], len(COMPOUND_ORDER)))
    rank = np.empty(len(compounds), dtype=int)
    for r, j in enumerate(keyed):
        rank[j] = r
    return rank


@dataclass(frozen=True)
class CompoundPrior:
    """Prior on the size of one step down the compound ladder.

    Every field describes an *adjacent-compound step*; the model builds the
    absolute levels by accumulating steps, which is what makes the ordering
    structural rather than hoped for.
    """

    label: str
    pace_step_s: float          # harder compound is this much slower, per step
    pace_step_ln_sd: float
    deg_ratio: float            # softer degrades this many times faster, per step
    deg_ratio_ln_sd: float
    knee_step_laps: float       # harder compound's cliff arrives this much later
    knee_step_ln_sd: float
    derivation: str = ""

    # -- deterministic ladders, for reporting and for the fallback fit -----

    def pace_offsets(self, compounds: list) -> dict:
        """Prior-mean lap-time offset per compound, softest = 0."""
        r = hardness_rank(compounds)
        return {c: float(r[j] * self.pace_step_s) for j, c in enumerate(compounds)}

    def deg_multipliers(self, compounds: list) -> dict:
        """Prior-mean degradation multiplier per compound, softest = 1."""
        r = hardness_rank(compounds)
        return {c: float(self.deg_ratio ** (-r[j])) for j, c in enumerate(compounds)}

    def as_row(self) -> dict:
        return {
            "prior": self.label,
            "pace_step_s": round(self.pace_step_s, 4),
            "deg_ratio": round(self.deg_ratio, 3),
            "knee_step_laps": round(self.knee_step_laps, 2),
            "derivation": self.derivation,
        }


def compound_prior(event: Event | str, *,
                   pace_step_s: float | None = None,
                   pace_step_note: str = "") -> CompoundPrior:
    """Derived from the circuit's lap time, not hardcoded per track.

        pace step = 0.25% of lap time     (measured on adjacent race stints;
                                           `pace_step_s` overrides it with a
                                           donor weekend's estimate)
        deg ratio = 1.8x per step         (measured on Barcelona 2026 race
                                           stints: 1.42 and 2.30)
        knee step = 3 laps per step       (a harder tyre reaches its cliff later)
    """
    ev = get_event(event) if isinstance(event, str) else event
    if pace_step_s is None:
        step = COMPOUND_PACE_STEP_FRAC * ev.t_lap_ref_s
        note = (f"pace step = {COMPOUND_PACE_STEP_FRAC:.3%} x "
                f"{ev.t_lap_ref_s:.0f} s = {step:.3f} s per compound")
    else:
        step = float(pace_step_s)
        note = pace_step_note or f"pace step = {step:.3f} s per compound (supplied)"
    return CompoundPrior(
        label="2026 compound ladder",
        pace_step_s=step,
        pace_step_ln_sd=COMPOUND_PACE_STEP_REL_SD,
        deg_ratio=COMPOUND_DEG_RATIO,
        deg_ratio_ln_sd=COMPOUND_DEG_RATIO_LN_SD,
        knee_step_laps=COMPOUND_KNEE_STEP_LAPS,
        knee_step_ln_sd=COMPOUND_KNEE_STEP_LN_SD,
        derivation=(
            f"{note};  deg ratio = {COMPOUND_DEG_RATIO:.2f}x per compound;  "
            f"cliff arrives {COMPOUND_KNEE_STEP_LAPS:.0f} laps later per compound"
        ),
    )


def flat_prior(event: Event | str, **_) -> CompoundPrior:
    """The ladder switched off — the sensitivity comparison.

    Steps shrink to ~0 with a wide scale, which is as close as a strictly
    positive parameterisation gets to "no ordering assumption".  Kept so the
    Curves tab can show what the fit does without the ladder, the same way the
    "no fuel prior" variant shows what it does without the physics.
    """
    return CompoundPrior(
        label="no compound ladder",
        pace_step_s=0.01,
        pace_step_ln_sd=2.0,
        deg_ratio=1.001,
        deg_ratio_ln_sd=1.0,
        knee_step_laps=0.1,
        knee_step_ln_sd=2.0,
        derivation="ordering left almost unconstrained; the identifiability trap",
    )


PRIORS = {"ladder": compound_prior, "flat": flat_prior}


def get_compound_prior(event: Event | str, which: str = "ladder", *,
                       pace_step_s: float | None = None,
                       pace_step_note: str = "") -> CompoundPrior:
    if which not in PRIORS:
        raise KeyError(f"unknown compound prior {which!r}; known: {sorted(PRIORS)}")
    return PRIORS[which](event, pace_step_s=pace_step_s,
                         pace_step_note=pace_step_note)


def summary_table(event: Event | str, compounds: list | None = None,
                  *, pace_step_s: float | None = None):
    """One row per compound — the ladder as the model will apply it."""
    import pandas as pd

    ev = get_event(event) if isinstance(event, str) else event
    cp = compound_prior(ev, pace_step_s=pace_step_s)
    comps = list(compounds) if compounds else list(COMPOUND_ORDER)
    pace = cp.pace_offsets(comps)
    mult = cp.deg_multipliers(comps)
    rank = hardness_rank(comps)
    rows = [
        {
            "compound": c,
            "hardness_rank": int(rank[j]),
            "pace_offset_s": round(pace[c], 3),
            "deg_multiplier": round(mult[c], 3),
        }
        for j, c in enumerate(comps)
    ]
    return pd.DataFrame(rows).sort_values("hardness_rank").reset_index(drop=True)


# --------------------------------------------------------------------------
# Measuring the pace step, with the race-phase bias cancelled
# --------------------------------------------------------------------------
#
# The ladder's *ordering* is structural, but the size of each step is a real
# quantity that should be measured wherever it can be.  The obvious estimator
# — compare compounds across a race — does not work: compounds are assigned to
# race phases, not at random.  MEDIUMs run in the opening stint on a green
# track in a bunched field, HARDs run later on rubbered-in tarmac.  Regressed
# naively, the same Barcelona data says a HARD is 1.2 s/lap *quicker* than a
# MEDIUM going one way and 1.7 s/lap slower going the other.
#
# The fix is a symmetric design.  For two *adjacent* stints of one driver,
#
#     y2 - y1 = step * (rank2 - rank1)  -  phase
#
# where `phase` is however much faster the later stint is for reasons that have
# nothing to do with the tyre.  It enters as a constant, because it applies to
# every pair regardless of which way along the ladder the compound moved.
# Regressing the pace difference on the *rank* difference therefore identifies
# the step from the slope and dumps the confound into the intercept, where it
# shows up as the ~1.5 s/stint of track evolution it actually is.
#
# Both stints are read on fresh rubber (tyre age 2-6) so degradation has barely
# started, and fuel-corrected to a common race distance.


MIN_STINT_LAPS_FOR_STEP = 6
PACE_STEP_BAND = (0.02, 1.50)  # a donor step outside this is not believed


@dataclass
class PaceStepMeasurement:
    event: str = ""
    step_s: float = float("nan")
    se: float = float("nan")
    phase_bias_s: float = float("nan")
    n_laps: int = 0


def measure_pace_step(event: Event | str, race=None) -> PaceStepMeasurement:
    """The **net** stint-level compound effect: what one step harder costs over
    a whole stint, holding stint length and race phase fixed.

    This is the compound quantity that is actually identified by a race, and it
    is not the one a cost model consumes directly.  Two different things:

      (a) pace at *equal tyre age* - how much faster a fresh SOFT is than a
          fresh HARD.  Real, and estimable at +0.35 +/- 0.07 s per step on
          Barcelona 2026 with driver and race-lap fixed effects.
      (b) pace over a *whole stint* at the length each compound is run to.
          Measured here.  Barcelona 2026: +0.027 s/step.  Hungary 2026: +0.042.
          Essentially zero - the SOFT gives its fresh-tyre advantage back
          through faster degradation before the stint ends.

    The cost model needs (a), because it prices degradation separately.  But
    (a) is only meaningful *relative to* the degradation ladder sitting beside
    it: charge (a) on every lap and price degradation too, and the SOFT's
    advantage is counted twice.  Deriving (a) from (b) requires knowing what
    the degradation model already charges over a stint, and that is sharply
    sensitive to the stint length assumed - on Barcelona 2026 the implied
    fresh-tyre step runs from 0.20 s at a 14-lap stint to 0.30 s at 22 laps.

    So this function no longer tries to return (a).  It returns (b), which is
    what the data supports, and (a) is a calibrated constant
    (`COMPOUND_PACE_STEP_FRAC`) that the pipeline then *checks* by asking
    whether the model reproduces (b).  That is the honest division: measure
    what is identified, calibrate what is not, and gate the combination.
    """
    import pandas as pd

    from src.config import DATA_PROCESSED, VALID_COMPOUNDS
    from src.fuel import get_prior

    ev = get_event(event) if isinstance(event, str) else event
    if race is None:
        path = DATA_PROCESSED / f"laps_{ev.key}_race.parquet"
        if not path.exists():
            return PaceStepMeasurement(event=ev.key)
        race = pd.read_parquet(path)

    fp = get_prior(ev, "2026")
    d = race[
        race["is_accurate"].fillna(False).astype(bool)
        & ~race["pit_in"].fillna(False).astype(bool)
        & ~race["pit_out"].fillna(False).astype(bool)
        & (race["track_status"].astype(str) == "1")
        & race["compound"].isin(VALID_COMPOUNDS)
        & race["lap_time_s"].notna()
    ].copy()
    if d.empty or d["compound"].nunique() < 2:
        return PaceStepMeasurement(event=ev.key)
    med = d.groupby("driver")["lap_time_s"].transform("median")
    d = d[d["lap_time_s"] < med + 3.0].copy()
    d["y"] = d["lap_time_s"] + fp.s_per_lap * (
        ev.n_race_laps - d["lap_number"].astype(float))
    # Relative to the driver's own race, so car and driver quality drop out.
    d["rel"] = d["y"] - d.groupby("driver")["y"].transform("mean")

    st = (d.groupby(["driver", "stint"])
          .agg(compound=("compound", "first"), n=("y", "size"),
               rel=("rel", "mean"), midlap=("lap_number", "mean"))
          .reset_index())
    st = st[st["n"] >= MIN_STINT_LAPS_FOR_STEP]
    comps = [c for c in COMPOUND_ORDER if c in set(st["compound"])]
    if len(st) < 12 or len(comps) < 2:
        return PaceStepMeasurement(event=ev.key, n_laps=len(st))

    dum = pd.get_dummies(st["compound"]).astype(float)
    X = np.column_stack([dum[comps].to_numpy(),
                         st["n"].to_numpy(dtype=float),
                         st["midlap"].to_numpy(dtype=float) / ev.n_race_laps])
    y = st["rel"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    cov = np.linalg.pinv(X.T @ X) * float(resid @ resid / dof)

    rank = {c: i for i, c in enumerate(COMPOUND_ORDER)}
    xr = np.array([rank[c] - rank[comps[0]] for c in comps[1:]], dtype=float)
    off = np.array([beta[i + 1] - beta[0] for i in range(len(comps) - 1)])
    se = float(np.sqrt(np.mean([cov[i + 1, i + 1] + cov[0, 0]
                                for i in range(len(comps) - 1)])))
    net = float((xr @ off) / (xr @ xr)) if (xr @ xr) else float("nan")
    return PaceStepMeasurement(event=ev.key, step_s=net, se=se,
                               phase_bias_s=float(st["n"].median()),
                               n_laps=int(len(st)))


def model_net_stint_step(life: dict, stint_len: float, pace_step_s: float) -> float:
    """What the *model* says one step harder costs over a stint of `stint_len`.

    The counterpart to `measure_pace_step`, and the thing that has to match it.
    Pace offset works one way, degradation the other; this returns the net.
    """
    from src.config import GRIP_BUDGET_S
    from src.tyre import grip_loss

    comps = [c for c in COMPOUND_ORDER if c in life and life[c] and life[c] > 0]
    if len(comps) < 2:
        return float("nan")
    ages = np.arange(1, max(int(round(stint_len)), 1) + 1, dtype=float)
    rank = {c: i for i, c in enumerate(COMPOUND_ORDER)}
    steps = []
    for c in comps[1:]:
        dr = rank[c] - rank[comps[0]]
        deg_soft = float(grip_loss(ages / life[comps[0]] - 0.5 / life[comps[0]],
                                   budget=GRIP_BUDGET_S).mean())
        deg_hard = float(grip_loss(ages / life[c] - 0.5 / life[c],
                                   budget=GRIP_BUDGET_S).mean())
        steps.append((pace_step_s * dr - (deg_soft - deg_hard)) / dr)
    return float(np.mean(steps))


def pace_step_prior(target: Event | str, *, donors: list | None = None) -> dict:
    """Fresh-tyre pace step for a weekend, with the donor races' *net* stint
    effect carried alongside as the check on it.

    The fresh-tyre step is a calibrated constant, not a per-weekend
    measurement, and that is a deliberate retreat from what this function used
    to claim.  What a race identifies is the **net** cost of one step harder
    over a whole stint; splitting that net into "fresh-tyre pace" and
    "degradation" requires the degradation ladder, and the split is sharply
    sensitive to the stint length assumed - on Barcelona 2026 the implied
    fresh-tyre step runs 0.20 s at a 14-lap stint and 0.30 s at 22 laps.
    Reporting a per-weekend number for it was false precision, and acting on
    one produced a recommendation that used no HARD at either weekend, at
    circuits where the field ran 63% and 48% of its race laps on the HARD.

    So: the fresh step comes from `COMPOUND_PACE_STEP_FRAC`, calibrated so the
    model reproduces the measured net; the donors supply that measured net, and
    the pipeline gates the model against it.  The firewall is unchanged - the
    target weekend's own race sets nothing here.
    """
    from src.config import EVENTS

    ev = get_event(target) if isinstance(target, str) else target
    keys = list(donors) if donors is not None else [k for k in EVENTS if k != ev.key]
    keys = [k for k in keys if k != ev.key]

    ms = [m for m in (measure_pace_step(k) for k in keys) if np.isfinite(m.step_s)]
    step = COMPOUND_PACE_STEP_FRAC * ev.t_lap_ref_s
    out = {
        "step_s": float(step),
        "measured": False,
        "calibrated": True,
        "label": f"calibrated ({COMPOUND_PACE_STEP_FRAC:.2%} of lap time)",
        "sources": [m.event for m in ms],
        "detail": [{"event": m.event, "net_stint_step_s": round(m.step_s, 4),
                    "se": round(m.se, 4), "n_stints": m.n_laps,
                    "median_stint_laps": round(m.phase_bias_s, 1)} for m in ms],
    }
    if ms:
        net = float(np.mean([m.step_s for m in ms]))
        out["net_stint_step_measured"] = net
        out["derivation"] = (
            f"fresh-tyre step is the calibrated "
            f"{COMPOUND_PACE_STEP_FRAC:.2%} of lap time = {step:.3f} s at "
            f"{ev.key}. What the donor races identify is the *net* cost of one "
            f"step harder over a whole stint, holding stint length and race "
            f"phase fixed: "
            + "; ".join(f"{m.event} {m.step_s:+.3f} +/- {m.se:.3f} s "
                        f"({m.n_laps} stints, median {m.phase_bias_s:.0f} laps)"
                        for m in ms)
            + " - near zero, because the SOFT gives its fresh-tyre advantage "
              "back through degradation before the stint ends. The model is "
              "gated against reproducing that net, which is the combination "
              "the data actually pins.")
    else:
        out["net_stint_step_measured"] = float("nan")
        out["derivation"] = (
            f"fresh-tyre step is the calibrated "
            f"{COMPOUND_PACE_STEP_FRAC:.2%} of lap time = {step:.3f} s; no donor "
            f"race is available to measure the net stint-level step against")
    return out


ALLOCATION_COVERAGE = 0.85   # cap must cover this share of classified finishers


def measure_allocation(event: Event | str, race=None) -> dict:
    """How many stints of each compound a driver can actually run.

    Not a rule but a resource.  Pirelli's dry allocation is 13 sets, but they
    are consumed through practice and qualifying, and what survives to the grid
    is lopsided: the HARD is barely touched before Sunday, the MEDIUM is used
    for long runs, and the SOFT is spent in qualifying.  What reaches the race
    is roughly two hards, one medium and a scrubbed soft or two - and that, not
    lap time, is why the field's plans look the way they do.

    Measured from a race by counting stints per compound per *classified
    finisher* (a retirement truncates the count and would bias every cap down),
    and taking the smallest cap covering `ALLOCATION_COVERAGE` of them.  Pooled
    over the 2026 weekends that gives SOFT 2, MEDIUM 1, HARD 2:

        SOFT     41% of finishers ran none, 81% ran at most one
        MEDIUM   97% ran exactly one; at Hungary 2026 nobody ran two
        HARD     only 34% ran fewer than two

    A single global cap of 2 misses all of this.  It let the optimiser build
    plans from two mediums, or from two softs with no medium at all, neither of
    which is a set of tyres a driver has.
    """
    import pandas as pd

    from src.config import DATA_PROCESSED, MAX_STINTS_PER_COMPOUND, VALID_COMPOUNDS

    ev = get_event(event) if isinstance(event, str) else event
    if race is None:
        path = DATA_PROCESSED / f"laps_{ev.key}_race.parquet"
        if not path.exists():
            return {}
        race = pd.read_parquet(path)

    st = (race.groupby(["driver", "stint"])
          .agg(compound=("compound", "first"), n=("lap_number", "size"),
               end=("lap_number", "max"))
          .reset_index())
    st = st[(st["n"] >= 3) & st["compound"].isin(VALID_COMPOUNDS)]
    if st.empty:
        return {}
    finished = st.groupby("driver")["end"].max()
    classified = finished[finished >= ev.n_race_laps - 2].index
    st = st[st["driver"].isin(classified)]
    if st["driver"].nunique() < 6:
        return {}

    counts = (st.groupby(["driver", "compound"]).size().unstack(fill_value=0)
              .reindex(columns=list(VALID_COMPOUNDS), fill_value=0))
    out = {}
    for c in VALID_COMPOUNDS:
        k = counts[c].to_numpy()
        cap = next((j for j in range(0, int(k.max()) + 1)
                    if float((k <= j).mean()) >= ALLOCATION_COVERAGE), int(k.max()))
        out[c] = int(max(1, min(cap, MAX_STINTS_PER_COMPOUND + 1)))
    return out


def allocation_prior(target: Event | str, *, donors: list | None = None) -> dict:
    """Per-compound stint allowance, measured on *other* weekends' races."""
    from src.config import EVENTS, MAX_STINTS_PER_COMPOUND, VALID_COMPOUNDS

    ev = get_event(target) if isinstance(target, str) else target
    keys = [k for k in (donors if donors is not None else EVENTS) if k != ev.key]
    got = [(k, measure_allocation(k)) for k in keys]
    got = [(k, a) for k, a in got if a]
    if not got:
        return {"caps": {c: MAX_STINTS_PER_COMPOUND for c in VALID_COMPOUNDS},
                "measured": False, "sources": [],
                "derivation": (f"no donor race available; a flat "
                               f"{MAX_STINTS_PER_COMPOUND} stints per compound applies")}
    caps = {c: int(max(a.get(c, MAX_STINTS_PER_COMPOUND) for _, a in got))
            for c in VALID_COMPOUNDS}
    return {
        "caps": caps, "measured": True, "sources": [k for k, _ in got],
        "detail": {k: a for k, a in got},
        "derivation": (
            "stints per compound per classified finisher, "
            + "; ".join(f"{k} {a}" for k, a in got)
            + f" -> {caps} at {ev.key} (the most permissive donor, so the cap "
              f"never forbids something a donor field actually did)"),
    }
