"""The 2026 fuel/mass physics prior — the centrepiece of the identifiability story.

Within a stint, tyre age and fuel burn are *perfectly collinear*: both advance
+1 per lap.  A free regression therefore splits "car getting lighter" from
"tyres getting older" arbitrarily.  Pinning the fuel half with physics is what
makes the tyre half identifiable.

Only the fuel *change* within a stint is identifiable.  The unknown starting
fuel load of a practice long run is absorbed into that stint's intercept, so
the fuel term is driven by `lap_in_stint`, never by an absolute fuel estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    ALPHA_MASS,
    TYRE_LOAD_EXPONENT,
    FUEL_ALLOWANCE_2026_KG,
    FUEL_BURN_2025_KG_PER_LAP,
    K_TRACK_2025_S_PER_KG,
    K_TRACK_PRIOR_REL_SD,
    MID_STINT_FUEL_KG,
    MIN_CAR_WEIGHT_2026_KG,
    Event,
    get_event,
)


@dataclass(frozen=True)
class FuelPrior:
    """A fully specified fuel model: how much mass goes per lap, and what it's worth."""

    label: str
    burn_kg_per_lap: float
    k_track_s_per_kg: float
    rel_sd: float = K_TRACK_PRIOR_REL_SD
    derivation: str = ""

    @property
    def s_per_lap(self) -> float:
        """Lap time gained per lap purely from burning fuel."""
        return self.burn_kg_per_lap * self.k_track_s_per_kg

    def term(self, lap_in_stint: np.ndarray | pd.Series) -> np.ndarray:
        """Signed contribution of fuel to lap time (negative: the car gets faster)."""
        return -self.s_per_lap * np.asarray(lap_in_stint, dtype=float)

    def correct(self, lap_time_s, lap_in_stint) -> np.ndarray:
        """Remove the fuel effect: what the lap would have been at constant mass."""
        return np.asarray(lap_time_s, dtype=float) - self.term(lap_in_stint)


# --------------------------------------------------------------------------
# Fuel load as a degradation multiplier
# --------------------------------------------------------------------------
#
# The fuel prior above answers "how much quicker is a lighter car?".  This
# answers a different question the strategy simulator also needs: "how much
# harder does a heavier car wear its tyres?"
#
# Tyre wear is load-sensitive — abrasion and bulk heating both rise faster than
# linearly with the load through the contact patch — so the same compound
# degrades faster on lap 3 with a full tank than on lap 60 on fumes.  This is
# the mechanism behind a pattern every race shows and no order-invariant model
# can reproduce: real stints get *longer* through a race.  Barcelona 2026's
# M-H-H runners went 13-16, then 22-24, then 25-31 laps.
#
# Including it is also what makes stint *order* matter at all.  Without it a
# plan is fully described by its multiset of (compound, length) pairs and the
# simulator has no opinion on which tyre to start on.


def mass_profile(event: Event | str, laps: np.ndarray | None = None) -> np.ndarray:
    """Car mass in kg on each race lap, from the regulated minimum plus fuel.

    Lap 1 starts on the full allowance and the tank is empty at the flag, so
    the profile is a straight line from `min weight + allowance` down to
    `min weight`.
    """
    ev = get_event(event) if isinstance(event, str) else event
    n = ev.n_race_laps
    laps = np.arange(1, n + 1, dtype=float) if laps is None else np.asarray(laps, float)
    burned = FUEL_ALLOWANCE_2026_KG * np.clip((laps - 1) / max(n - 1, 1), 0.0, 1.0)
    return MIN_CAR_WEIGHT_2026_KG + FUEL_ALLOWANCE_2026_KG - burned


def deg_load_multiplier(event: Event | str, laps: np.ndarray | None = None,
                        *, exponent: float = TYRE_LOAD_EXPONENT) -> np.ndarray:
    """Degradation multiplier per race lap: (mass / mid-race mass) ** exponent.

    Normalised at the mid-race mass — the same reference `k_track` linearises
    about — so the multiplier averages near 1 across a race and the correction
    redistributes degradation between early and late stints rather than
    inflating the total.
    """
    ev = get_event(event) if isinstance(event, str) else event
    m = mass_profile(ev, laps)
    return (m / ev.m_total_kg) ** float(exponent)


def prior_2026(event: Event | str) -> FuelPrior:
    """Derived from the regulations, not hardcoded.

        burn = 70 kg / n_race_laps
        k    = alpha * t_lap_ref / (768 kg + mid-stint fuel)
    """
    ev = get_event(event) if isinstance(event, str) else event
    burn = FUEL_ALLOWANCE_2026_KG / ev.n_race_laps
    k = ALPHA_MASS * ev.t_lap_ref_s / (MIN_CAR_WEIGHT_2026_KG + MID_STINT_FUEL_KG)
    return FuelPrior(
        label="2026 physics",
        burn_kg_per_lap=burn,
        k_track_s_per_kg=k,
        derivation=(
            f"burn = {FUEL_ALLOWANCE_2026_KG:.0f} kg / {ev.n_race_laps} laps "
            f"= {burn:.3f} kg/lap;  "
            f"k = {ALPHA_MASS:.2f} x {ev.t_lap_ref_s:.0f} s / "
            f"{MIN_CAR_WEIGHT_2026_KG + MID_STINT_FUEL_KG:.0f} kg "
            f"= {k:.4f} s/kg"
        ),
    )


def prior_2025(event: Event | str) -> FuelPrior:
    """Last year's physics — the sensitivity slide.

    Heavier cars burning far more fuel per lap.  Applying it to a 2026 car
    over-corrects the fuel effect and mis-prices the tyre.
    """
    return FuelPrior(
        label="2025 physics",
        burn_kg_per_lap=FUEL_BURN_2025_KG_PER_LAP,
        k_track_s_per_kg=K_TRACK_2025_S_PER_KG,
        derivation=(
            f"burn = {FUEL_BURN_2025_KG_PER_LAP} kg/lap (110 kg allowance), "
            f"k = {K_TRACK_2025_S_PER_KG} s/kg (800 kg min weight)"
        ),
    )


def prior_none(event: Event | str) -> FuelPrior:
    """No physics at all — the free regression that the trap catches."""
    return FuelPrior(
        label="no prior",
        burn_kg_per_lap=0.0,
        k_track_s_per_kg=0.0,
        rel_sd=1e-6,
        derivation="fuel effect assumed zero; age and fuel burn stay collinear",
    )


PRIORS = {"2026": prior_2026, "2025": prior_2025, "none": prior_none}


def get_prior(event: Event | str, which: str = "2026") -> FuelPrior:
    if which not in PRIORS:
        raise KeyError(f"unknown prior {which!r}; known: {sorted(PRIORS)}")
    return PRIORS[which](event)


def add_fuel_correction(df: pd.DataFrame, event: Event | str,
                        which: str = "2026") -> pd.DataFrame:
    """Attach `fuel_term_s` and `lap_time_fuel_corr` to a clean lap table."""
    p = get_prior(event, which)
    out = df.copy()
    out["fuel_prior"] = p.label
    out["fuel_term_s"] = p.term(out["lap_in_stint"])
    out["lap_time_fuel_corr"] = out["lap_time_s"] - out["fuel_term_s"]
    return out


def summary_table(event: Event | str) -> pd.DataFrame:
    """One row per prior — the numbers that go on the physics slide."""
    ev = get_event(event) if isinstance(event, str) else event
    rows = []
    for name in ("2026", "2025", "none"):
        p = get_prior(ev, name)
        rows.append(
            {
                "prior": p.label,
                "burn_kg_per_lap": round(p.burn_kg_per_lap, 4),
                "k_track_s_per_kg": round(p.k_track_s_per_kg, 5),
                "fuel_effect_s_per_lap": round(p.s_per_lap, 4),
                "derivation": p.derivation,
            }
        )
    return pd.DataFrame(rows)
