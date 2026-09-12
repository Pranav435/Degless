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

Both answers are physically impossible *as a rule*, and the first version of
this module pinned the ordering by construction.  Seven scored weekends later
the races disagree with the rule more often than not: measured with driver and
race-lap fixed effects, softer-degrades-faster holds on 3 of 7 weekends, and a
model whose ordering is pinned cannot tell whether the estimator or the
assumption is wrong.  So the ladder is now a **soft prior sized per circuit**:
the fresh-tyre pace step and the degradation ratio between adjacent compounds
come from the circuit's own 2023-25 races (the compound intercepts and slopes
of the same regression, see `history.race_deg_slopes`), the practice fit
starts from those with a finite width, and a race that contradicts the
ordering can move it.  The hard ladder survives as `compound_prior="ladder"`.

**The gate is binding.**  What a race identifies is not the pace step or the
degradation ratio separately but their *net* over a stint - what one step
harder costs over the length compounds are actually run to.  On every scored
weekend the model's net had the wrong sign (harder was *faster* over a stint
by 0.06-0.32 s/lap, against a measured +0.03 to +0.11), and that gate was
reported but not enforced.  `calibrate_pace_offsets` now sets the fresh-tyre
offsets so that the model reproduces the measured net at the stint length it
recommends, draw by draw with the measurement's own uncertainty, before any
plan is priced.
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

SOFT_PACE_STEP_SD_FLOOR_S = 0.08   # a soft prior on the fresh-tyre step is at least this wide (s)
SOFT_DEG_RATIO_LN_SD_FLOOR = 0.15
PACE_OFFSET_STEP_BAND_S = (-0.15, 1.0)   # calibrated fresh-tyre offset per hardness step, physically plausible band


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
    absolute levels by accumulating steps.  With `soft=False` the steps are
    strictly positive (the ordering is structural); with `soft=True` they are
    Normal / LogNormal around the same means with a finite width, so the
    ordering is a prior the data may overturn.
    """

    label: str
    pace_step_s: float          # harder compound is this much slower, per step
    pace_step_ln_sd: float
    deg_ratio: float            # softer degrades this many times faster, per step
    deg_ratio_ln_sd: float
    knee_step_laps: float       # harder compound's cliff arrives this much later
    knee_step_ln_sd: float
    derivation: str = ""
    soft: bool = False
    pace_step_sd_s: float = 0.0  # absolute width of the soft pace-step prior (s)

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
            "soft": bool(self.soft),
            "derivation": self.derivation,
        }


def _ladder_values(ev: Event, pace_step_s, pace_step_note, circuit_ladder: dict | None):
    """Prior means for the pace step and the degradation ratio: the circuit's
    own races where they exist, the season-wide constants otherwise."""
    cl = circuit_ladder or {}
    if pace_step_s is not None:
        step = float(pace_step_s)
        note = pace_step_note or f"pace step = {step:.3f} s per compound (supplied)"
        step_se = float(cl.get("pace_step_se", 0.0) or 0.0)
    elif cl.get("pace_step_usable") and cl.get("pace_step_s", 0) > 0:
        step = float(cl["pace_step_s"])
        step_se = float(cl.get("pace_step_se", 0.0) or 0.0)
        note = (f"pace step = {step:.3f} +/- {step_se:.3f} s per compound at equal tyre age, "
                f"from this circuit's races ({cl.get('n_races', '?')} races)")
    else:
        step = COMPOUND_PACE_STEP_FRAC * ev.t_lap_ref_s
        step_se = 0.0
        note = (f"pace step = {COMPOUND_PACE_STEP_FRAC:.3%} x "
                f"{ev.t_lap_ref_s:.0f} s = {step:.3f} s per compound (season constant)")
    if cl.get("deg_ratio") and 0.5 < cl["deg_ratio"] < 4.0:
        ratio = float(cl["deg_ratio"])
        ratio_sd = float(max(cl.get("deg_ratio_ln_sd", COMPOUND_DEG_RATIO_LN_SD), SOFT_DEG_RATIO_LN_SD_FLOOR))
        rnote = f"deg ratio = {ratio:.2f}x per compound from this circuit's races"
    else:
        ratio, ratio_sd = COMPOUND_DEG_RATIO, COMPOUND_DEG_RATIO_LN_SD
        rnote = f"deg ratio = {ratio:.2f}x per compound (season constant)"
    return step, step_se, note, ratio, ratio_sd, rnote


def compound_prior(event: Event | str, *,
                   pace_step_s: float | None = None,
                   pace_step_note: str = "",
                   circuit_ladder: dict | None = None,
                   soft: bool = False) -> CompoundPrior:
    """The ladder prior for a weekend.

        pace step   this circuit's races (compound intercepts at equal tyre
                    age) where they exist, else 0.21% of lap time
        deg ratio   this circuit's races (adjacent-compound slope ratio) where
                    they exist, else 1.30x per step
        knee step   3 laps per step (a harder tyre reaches its cliff later)

    `soft=True` gives the same means with finite widths on the *signed* steps,
    so a weekend whose data contradict the ordering can invert it.
    """
    ev = get_event(event) if isinstance(event, str) else event
    step, step_se, note, ratio, ratio_sd, rnote = _ladder_values(ev, pace_step_s, pace_step_note, circuit_ladder)
    sd_abs = max(SOFT_PACE_STEP_SD_FLOOR_S, 2.0 * step_se, COMPOUND_PACE_STEP_REL_SD * abs(step))
    return CompoundPrior(
        label=("2026 compound ladder (soft, per circuit)" if soft else "2026 compound ladder"),
        pace_step_s=step,
        pace_step_ln_sd=COMPOUND_PACE_STEP_REL_SD,
        deg_ratio=ratio,
        deg_ratio_ln_sd=ratio_sd,
        knee_step_laps=COMPOUND_KNEE_STEP_LAPS,
        knee_step_ln_sd=COMPOUND_KNEE_STEP_LN_SD,
        derivation=(f"{note};  {rnote} (ln-sd {ratio_sd:.2f});  "
                    f"cliff arrives {COMPOUND_KNEE_STEP_LAPS:.0f} laps later per compound"
                    + (f";  soft ordering: pace step sd {sd_abs:.3f} s" if soft else "")),
        soft=soft,
        pace_step_sd_s=float(sd_abs),
    )


def soft_prior(event: Event | str, **kw) -> CompoundPrior:
    kw.pop("soft", None)
    return compound_prior(event, soft=True, **kw)


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


PRIORS = {"ladder": compound_prior, "soft": soft_prior, "flat": flat_prior}


def get_compound_prior(event: Event | str, which: str = "soft", *,
                       pace_step_s: float | None = None,
                       pace_step_note: str = "",
                       circuit_ladder: dict | None = None) -> CompoundPrior:
    if which not in PRIORS:
        raise KeyError(f"unknown compound prior {which!r}; known: {sorted(PRIORS)}")
    return PRIORS[which](event, pace_step_s=pace_step_s,
                         pace_step_note=pace_step_note, circuit_ladder=circuit_ladder)


def summary_table(event: Event | str, compounds: list | None = None,
                  *, pace_step_s: float | None = None, circuit_ladder: dict | None = None):
    """One row per compound — the ladder as the model will apply it."""
    import pandas as pd

    ev = get_event(event) if isinstance(event, str) else event
    cp = compound_prior(ev, pace_step_s=pace_step_s, circuit_ladder=circuit_ladder)
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
# Measuring the net stint-level compound step, with the race-phase bias cancelled
# --------------------------------------------------------------------------
#
# The obvious estimator - compare compounds across a race - does not work:
# compounds are assigned to race phases, not at random.  MEDIUMs run in the
# opening stint on a green track in a bunched field, HARDs run later on
# rubbered-in tarmac.  Regressed naively, the same Barcelona data says a HARD
# is 1.2 s/lap *quicker* than a MEDIUM going one way and 1.7 s/lap slower
# going the other.
#
# So the stint-mean pace, relative to the driver's own race, is regressed on
# compound dummies with stint length and race phase held fixed; the compound
# contrast per hardness step is the net cost of one step harder over a whole
# stint, at the lengths compounds are actually run to.


MIN_STINT_LAPS_FOR_STEP = 6
PACE_STEP_BAND = (0.02, 1.50)  # a donor step outside this is not believed
NET_SE_FLOOR_S = 0.03          # no single measurement is believed tighter than this


@dataclass
class PaceStepMeasurement:
    event: str = ""
    step_s: float = float("nan")
    se: float = float("nan")
    phase_bias_s: float = float("nan")
    n_laps: int = 0


def net_stint_step_from_laps(race, *, fuel_s_per_lap: float, n_race_laps: int,
                             event: str = "") -> PaceStepMeasurement:
    """The net stint-level compound effect from one race's lap table.

    Columns needed: driver, stint, lap_number, compound, lap_time_s,
    is_accurate, pit_in, pit_out, track_status.  Works for a 2026 lap table
    and for a canonical FastF1 frame of an earlier season alike.
    """
    import pandas as pd

    from src.config import VALID_COMPOUNDS

    d = race[
        race["is_accurate"].fillna(False).astype(bool)
        & ~race["pit_in"].fillna(False).astype(bool)
        & ~race["pit_out"].fillna(False).astype(bool)
        & (race["track_status"].astype(str) == "1")
        & race["compound"].isin(VALID_COMPOUNDS)
        & race["lap_time_s"].notna()
    ].copy()
    if d.empty or d["compound"].nunique() < 2:
        return PaceStepMeasurement(event=event)
    med = d.groupby("driver")["lap_time_s"].transform("median")
    d = d[d["lap_time_s"] < med + 3.0].copy()
    d["y"] = d["lap_time_s"] + float(fuel_s_per_lap) * (n_race_laps - d["lap_number"].astype(float))
    # Relative to the driver's own race, so car and driver quality drop out.
    d["rel"] = d["y"] - d.groupby("driver")["y"].transform("mean")

    st = (d.groupby(["driver", "stint"])
          .agg(compound=("compound", "first"), n=("y", "size"),
               rel=("rel", "mean"), midlap=("lap_number", "mean"))
          .reset_index())
    st = st[st["n"] >= MIN_STINT_LAPS_FOR_STEP]
    comps = [c for c in COMPOUND_ORDER if c in set(st["compound"])]
    if len(st) < 12 or len(comps) < 2:
        return PaceStepMeasurement(event=event, n_laps=len(st))

    dum = pd.get_dummies(st["compound"]).astype(float)
    X = np.column_stack([dum[comps].to_numpy(),
                         st["n"].to_numpy(dtype=float),
                         st["midlap"].to_numpy(dtype=float) / n_race_laps])
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
    return PaceStepMeasurement(event=event, step_s=net, se=se,
                               phase_bias_s=float(st["n"].median()),
                               n_laps=int(len(st)))


def measure_pace_step(event: Event | str, race=None) -> PaceStepMeasurement:
    """The **net** stint-level compound effect of one 2026 race: what one step
    harder costs over a whole stint, holding stint length and race phase fixed.

    Two different quantities exist and only this one is identified by a race:

      (a) pace at *equal tyre age* - how much faster a fresh SOFT is than a
          fresh HARD (+0.35 +/- 0.07 s per step on Barcelona 2026 with driver
          and race-lap fixed effects; `history.race_deg_slopes(offsets=True)`).
      (b) pace over a *whole stint* at the length each compound is run to.
          Measured here.  Barcelona 2026: +0.08 s/step.  Hungary 2026: +0.10.

    The cost model charges (a) per lap and degradation separately, and the
    two together must reproduce (b); `calibrate_pace_offsets` enforces that.
    """
    import pandas as pd

    from src.config import DATA_PROCESSED
    from src.fuel import get_prior

    ev = get_event(event) if isinstance(event, str) else event
    if race is None:
        path = DATA_PROCESSED / f"laps_{ev.key}_race.parquet"
        if not path.exists():
            return PaceStepMeasurement(event=ev.key)
        race = pd.read_parquet(path)
    fp = get_prior(ev, "2026")
    return net_stint_step_from_laps(race, fuel_s_per_lap=fp.s_per_lap, n_race_laps=ev.n_race_laps, event=ev.key)


def model_net_stint_step(life: dict, stint_len: float, pace_step_s: float,
                         budgets: dict | None = None) -> float:
    """What the *model* says one step harder costs over a stint of `stint_len`.

    The counterpart to `measure_pace_step`, and the thing that has to match it.
    Pace offset works one way, degradation the other; this returns the net.
    `life` is laps to the cliff at the push the plan runs; `budgets` the
    per-compound grip budget (the config constant where not given).
    """
    from src.config import GRIP_BUDGET_S
    from src.tyre import grip_loss

    comps = [c for c in COMPOUND_ORDER if c in life and life[c] and life[c] > 0]
    if len(comps) < 2:
        return float("nan")
    ages = np.arange(1, max(int(round(stint_len)), 1) + 1, dtype=float)
    rank = {c: i for i, c in enumerate(COMPOUND_ORDER)}
    b = budgets or {}
    steps = []
    for a, c in zip(comps, comps[1:]):
        dr = rank[c] - rank[a]
        deg_soft = float(grip_loss(ages / life[a] - 0.5 / life[a], budget=b.get(a, GRIP_BUDGET_S)).mean())
        deg_hard = float(grip_loss(ages / life[c] - 0.5 / life[c], budget=b.get(c, GRIP_BUDGET_S)).mean())
        steps.append((pace_step_s * dr - (deg_soft - deg_hard)) / dr)
    return float(np.mean(steps))


def model_net_step_draws(model, stint_len: float, push: float, event: Event | str) -> np.ndarray:
    """Per-draw net cost of one step harder over a `stint_len`-lap stint at
    `push`, from a `TyreModel`: the pace offsets net of the degradation
    difference, averaged over adjacent-compound pairs.  Shape (draws,)."""
    from src.tyre import grip_loss, wear_multiplier

    comps = [c for c in COMPOUND_ORDER if c in model.compounds]
    if len(comps) < 2:
        return np.full(model.n_draws, np.nan)
    ages = np.arange(1, max(int(round(stint_len)), 1) + 1, dtype=float)
    psi = float(wear_multiplier(push))
    rank = {c: i for i, c in enumerate(COMPOUND_ORDER)}

    def mean_loss(c):
        w = model.wear_rate[c][:, None] * psi * (ages[None, :] - 0.5)
        return grip_loss(w, budget=model.budget_of(c)).mean(1)

    out = np.zeros(model.n_draws)
    for a, c in zip(comps, comps[1:]):
        dr = rank[c] - rank[a]
        out += ((model.pace_offset[c] - model.pace_offset[a]) - (mean_loss(a) - mean_loss(c))) / dr
    return out / (len(comps) - 1)


def calibrate_pace_offsets(model, measured_net_s: float, stint_len: float, push: float,
                           event: Event | str, *, net_se_s: float = 0.0, seed: int = 0):
    """Set the fresh-tyre pace offsets so the model reproduces the measured net.

    For each adjacent pair (softer `a`, harder `c`) the model's net over an
    `L`-lap stint at push `p` is

        net = (offset_c - offset_a) - (mean loss_a - mean loss_c) over L laps

    and the race-measured net is the one quantity the compound ladder is
    identified through.  This solves for `offset_c - offset_a` draw by draw,
    with the measurement's own standard error added as noise, and returns the
    model with those offsets and the table of what changed.  The softest
    compound keeps offset 0; the degradation ladder is untouched.
    """
    from src.tyre import grip_loss, wear_multiplier

    if not np.isfinite(measured_net_s):
        return model, {"applied": False, "why": "no measured net stint step"}
    comps = [c for c in COMPOUND_ORDER if c in model.compounds]
    if len(comps) < 2:
        return model, {"applied": False, "why": "fewer than two compounds"}
    rng = np.random.default_rng(seed)
    ages = np.arange(1, max(int(round(stint_len)), 1) + 1, dtype=float)
    psi = float(wear_multiplier(push))
    rank = {c: i for i, c in enumerate(COMPOUND_ORDER)}

    def mean_loss(c):
        w = model.wear_rate[c][:, None] * psi * (ages[None, :] - 0.5)
        return grip_loss(w, budget=model.budget_of(c)).mean(1)

    before = {c: float(model.pace_offset[c].mean()) for c in comps}
    net_draw = float(measured_net_s) + rng.normal(0.0, max(float(net_se_s), 0.0), size=model.n_draws)
    pace = {c: model.pace_offset[c].copy() for c in model.compounds}
    pace[comps[0]] = np.zeros(model.n_draws)
    clipped = False
    for a, c in zip(comps, comps[1:]):
        dr = rank[c] - rank[a]
        step = (net_draw * dr + (mean_loss(a) - mean_loss(c))) / dr
        # A harder compound is never much quicker than a softer one on fresh
        # rubber, and Pirelli's design target caps the other side; a net that
        # asks for more than this is the estimator's phase bias, not the tyre.
        lo, hi = PACE_OFFSET_STEP_BAND_S
        if np.any(step < lo) or np.any(step > hi):
            clipped = True
        pace[c] = pace[a] + np.clip(step, lo, hi) * dr
    new = model.copy_with(pace_offset=pace, source=f"{model.source} + pace offsets calibrated to the measured net step")
    after = {c: float(pace[c].mean()) for c in comps}
    return new, {"applied": True, "measured_net_s": float(measured_net_s), "net_se_s": float(net_se_s),
                 "stint_len": float(stint_len), "push": float(push), "clipped": clipped,
                 "offsets_before": before, "offsets_after": after,
                 "model_net_before": float(model_net_step_draws(model, stint_len, push, event).mean()),
                 "model_net_after": float(model_net_step_draws(new, stint_len, push, event).mean())}


def pace_step_prior(target: Event | str, *, donors: list | None = None, circuit=None) -> dict:
    """Fresh-tyre pace step for a weekend, and the measured net stint-level
    step the model is calibrated to reproduce.

    The fresh step's prior mean comes from the circuit's own races where they
    exist (`circuit.ladder`), else the season constant.  The **net** is pooled
    with inverse-variance weights from every measurement available that does
    not use the target's race: the circuit's 2023-25 races (the same circuit,
    older tyres) and the other 2026 weekends (the same tyres, other circuits).
    """
    from src.config import EVENTS

    ev = get_event(target) if isinstance(target, str) else target
    keys = list(donors) if donors is not None else [k for k in EVENTS if k != ev.key]
    keys = [k for k in keys if k != ev.key]

    ms = [m for m in (measure_pace_step(k) for k in keys) if np.isfinite(m.step_s)]
    cl = (circuit.ladder if circuit is not None and getattr(circuit, "available", False) else {}) or {}
    hist_nets = list(getattr(circuit, "net_steps", []) or []) if circuit is not None else []
    if cl.get("pace_step_s") is not None and cl.get("pace_step_s", 0) > 0:
        step = float(cl["pace_step_s"])
        label = f"this circuit's races: {step:.3f} +/- {cl.get('pace_step_se', float('nan')):.3f} s at equal tyre age"
        measured = True
    else:
        step = COMPOUND_PACE_STEP_FRAC * ev.t_lap_ref_s
        label = f"calibrated ({COMPOUND_PACE_STEP_FRAC:.2%} of lap time)"
        measured = False
    vals, ws, detail = [], [], []
    for h in hist_nets:
        if np.isfinite(h.get("step_s", np.nan)) and PACE_STEP_BAND[0] - 0.5 <= h["step_s"] <= PACE_STEP_BAND[1]:
            se = max(float(h.get("se", 0.1)), NET_SE_FLOOR_S)
            vals.append(float(h["step_s"])); ws.append(1.0 / se ** 2)
            detail.append({"event": f"{ev.circuit} {h['year']}", "net_stint_step_s": round(h["step_s"], 4),
                           "se": round(se, 4), "n_stints": h.get("n_stints"),
                           "median_stint_laps": h.get("median_stint_laps"), "kind": "circuit history"})
    for m in ms:
        se = max(float(m.se), NET_SE_FLOOR_S)
        vals.append(float(m.step_s)); ws.append(1.0 / se ** 2)
        detail.append({"event": m.event, "net_stint_step_s": round(m.step_s, 4), "se": round(se, 4),
                       "n_stints": m.n_laps, "median_stint_laps": round(m.phase_bias_s, 1), "kind": "2026 donor"})
    out = {"step_s": float(step), "measured": measured, "calibrated": not measured, "label": label,
           "sources": [d["event"] for d in detail], "detail": detail,
           "circuit_ladder": cl}
    if vals:
        vals, ws = np.array(vals), np.array(ws)
        net = float(np.sum(vals * ws) / ws.sum())
        se_pool = float(np.sqrt(1.0 / ws.sum()))
        # between-measurement spread, so one precise but atypical race cannot claim the answer
        spread = float(np.sqrt(np.average((vals - net) ** 2, weights=ws))) if len(vals) > 1 else 0.05
        out["net_stint_step_measured"] = net
        out["net_stint_step_se"] = float(max(se_pool, spread / np.sqrt(len(vals))))
        out["derivation"] = (
            f"fresh-tyre step prior {step:.3f} s ({label}). The races identify the *net* cost of one "
            f"step harder over a whole stint, holding stint length and race phase fixed: "
            + "; ".join(f"{d['event']} {d['net_stint_step_s']:+.3f} +/- {d['se']:.3f} s" for d in detail)
            + f" -> pooled {net:+.3f} +/- {out['net_stint_step_se']:.3f} s per step. The model's fresh-tyre "
              "offsets are calibrated so that, with its own degradation ladder, it reproduces this net at the "
              "stint length it recommends.")
    else:
        out["net_stint_step_measured"] = float("nan")
        out["net_stint_step_se"] = float("nan")
        out["derivation"] = f"fresh-tyre step prior {step:.3f} s ({label}); no race is available to measure the net stint-level step"
    return out


# --------------------------------------------------------------------------
# Dirty air, measured
# --------------------------------------------------------------------------


def measure_dirty_air(race, event: Event | str, *, close_s: float = 3.0) -> dict:
    """Lap time lost running within `close_s` of the car ahead, with driver and
    race-lap fixed effects and a per-compound tyre-age slope, on green-flag
    racing laps.  Returns `{s_per_lap, se, n_laps, share_close}`."""
    import pandas as pd

    from src.config import VALID_COMPOUNDS

    ev = get_event(event) if isinstance(event, str) else event
    gap_col = next((c for c in ("gap_ahead_s", "gap_to_ahead_s", "interval_s") if c in race.columns), None)
    if gap_col is None:
        return {}
    d = race[
        race["is_accurate"].fillna(False).astype(bool)
        & ~race["pit_in"].fillna(False).astype(bool)
        & ~race["pit_out"].fillna(False).astype(bool)
        & (race["track_status"].astype(str) == "1")
        & race["compound"].isin(VALID_COMPOUNDS)
        & race["lap_time_s"].notna() & race["tyre_age"].notna()
    ].copy()
    if len(d) < 100:
        return {}
    med = d.groupby("driver")["lap_time_s"].transform("median")
    d = d[d["lap_time_s"] < med + 4.0].copy()
    g = d[gap_col].to_numpy(dtype=float)
    close = (np.isfinite(g) & (g < close_s)).astype(float)
    comps = sorted(d["compound"].unique())
    comp = pd.get_dummies(d["compound"]).astype(float)[comps]
    age = d["tyre_age"].to_numpy(dtype=float)
    ageX = np.column_stack([comp[c].to_numpy() * age for c in comps])
    drv = pd.get_dummies(d["driver"], drop_first=True).astype(float)
    lap = pd.get_dummies(d["lap_number"].astype(int), drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(d)), close, ageX, comp.to_numpy()[:, 1:], drv.to_numpy(), lap.to_numpy()])
    y = d["lap_time_s"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    try:
        se = float(np.sqrt(max(np.linalg.pinv(X.T @ X)[1, 1] * float(resid @ resid / dof), 0)))
    except Exception:
        se = float("nan")
    return {"s_per_lap": float(beta[1]), "se": se, "n_laps": int(len(d)), "share_close": float(close.mean()),
            "event": ev.key}


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
    and taking the smallest cap covering `ALLOCATION_COVERAGE` of them.
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
