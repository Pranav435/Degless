"""Lap-table assembly and the clean-lap cascade.

Every filter is its own boolean column so the Decompose tab can show laps
falling away rule by rule.  All rules are **absolute** — never relative to the
stint minimum, which selects on the outcome and inflates degradation slopes
(the stint minimum usually falls early in the stint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    GREEN_FLAG,
    MIN_STINT_LAPS,
    SLOW_LAP_MARGIN_S,
    TRAFFIC_GAP_S,
    VALID_COMPOUNDS,
    Event,
    get_event,
)

# The cascade, in the order the plan numbers it and the Decompose tab peels it.
# The order matters: stint length is evaluated on what survives the *validity*
# rules (1-3), before the situational rules (traffic, slow laps) are applied.
# Requiring 6 laps to survive traffic filtering too would discard whole long
# runs over a single lap spent behind another car.
RULES = [
    ("ok_accurate", "FastF1 IsAccurate"),
    ("ok_not_pit", "not an in-lap or out-lap"),
    ("ok_green", "track status all-green"),
    ("ok_stint_len", f"stint >= {MIN_STINT_LAPS} valid laps (long run)"),
    ("ok_traffic", f"gap to car ahead > {TRAFFIC_GAP_S:.1f}s"),
    ("ok_not_slow", f"lap time < session median + {SLOW_LAP_MARGIN_S:.0f}s"),
    ("ok_compound", "known compound"),
]


def stint_uid(df: pd.DataFrame) -> pd.Series:
    return (
        df["event"].astype(str)
        + "|"
        + df["session"].astype(str)
        + "|"
        + df["driver"].astype(str)
        + "|"
        + df["stint"].astype("Int64").astype(str)
    )


def _gap_to_car_ahead(group: pd.DataFrame) -> pd.Series:
    """Smallest positive LapStartTime difference to any *other* car, per lap.

    A car starting its lap right behind another is in traffic: dirty air costs
    tenths that have nothing to do with the tyre.
    """
    g = group.sort_values("lap_start_s")
    times = g["lap_start_s"].to_numpy(dtype=float)
    drivers = g["driver"].to_numpy()
    gaps = np.full(len(g), np.inf)
    n_back = 40  # more than the field size; bounds the inner loop
    for i in range(len(g)):
        ti = times[i]
        if not np.isfinite(ti):
            continue
        for j in range(i - 1, max(-1, i - n_back - 1), -1):
            if drivers[j] == drivers[i]:
                continue
            d = ti - times[j]
            if np.isfinite(d) and d > 0:
                gaps[i] = d
                break
    return pd.Series(gaps, index=g.index)


def build_lap_table(raw: pd.DataFrame, event: Event | str | None = None) -> pd.DataFrame:
    """Apply the clean-lap cascade and derive stint-relative coordinates."""
    df = raw.copy().reset_index(drop=True)
    if df.empty:
        return df

    if event is not None:
        ev = get_event(event) if isinstance(event, str) else event
        df["event"] = ev.key

    df["compound"] = df["compound"].astype(str).str.upper().replace({"NAN": None, "NONE": None})
    df["lap_time_s"] = pd.to_numeric(df["lap_time_s"], errors="coerce")
    df["stint_uid"] = stint_uid(df)

    # -- stint-relative coordinates ---------------------------------------
    stint_start = df.groupby("stint_uid")["lap_number"].transform("min")
    df["lap_in_stint"] = df["lap_number"] - stint_start + 1.0

    # Tyre age: prefer FastF1's TyreLife, else reconstruct from the stint.
    tyre_life = pd.to_numeric(df.get("tyre_life"), errors="coerce")
    age0 = (
        tyre_life.groupby(df["stint_uid"]).transform("min").fillna(1.0)
        - df.groupby("stint_uid")["lap_in_stint"].transform("min")
    )
    df["tyre_age"] = tyre_life.where(tyre_life.notna(), df["lap_in_stint"] + age0.fillna(0.0))

    # -- rule 1: FastF1's own validity flag --------------------------------
    df["ok_accurate"] = df["is_accurate"].fillna(False).astype(bool) & df["lap_time_s"].notna()

    # -- rule 2: pit in/out laps -------------------------------------------
    df["ok_not_pit"] = ~(df["pit_in"].fillna(False).astype(bool)
                         | df["pit_out"].fillna(False).astype(bool))

    # -- rule 3: track status must be all-green (MANDATORY) ----------------
    # Barcelona 2026 ends under a safety car; without this every late-stint fit
    # is poisoned by +20 s laps.
    ts = df["track_status"].astype(str).str.strip()
    df["ok_green"] = (ts == GREEN_FLAG) | (ts == "") | (ts == "nan")
    df.loc[ts.isin(["", "nan", "None"]), "ok_green"] = True

    # -- rule 7: valid compound --------------------------------------------
    df["ok_compound"] = df["compound"].isin(VALID_COMPOUNDS)

    # -- rule 6: absurdly slow laps (cooldowns, aborted runs) --------------
    base = df["ok_accurate"] & df["ok_not_pit"]
    med = (
        df[base].groupby(["event", "session"])["lap_time_s"].median()
        .rename("session_median")
    )
    df = df.merge(med, on=["event", "session"], how="left")
    df["ok_not_slow"] = df["lap_time_s"] <= df["session_median"] + SLOW_LAP_MARGIN_S
    df.loc[df["lap_time_s"].isna(), "ok_not_slow"] = False

    # -- rule 5: traffic ----------------------------------------------------
    gaps = pd.Series(np.inf, index=df.index)
    for _, grp in df.groupby(["event", "session"], sort=False):
        gaps.loc[grp.index] = _gap_to_car_ahead(grp)
    df["gap_ahead_s"] = gaps
    df["ok_traffic"] = df["gap_ahead_s"] > TRAFFIC_GAP_S

    # -- rule 4: long runs only --------------------------------------------
    # Counted over laps passing the validity rules (1-3) only.
    valid = df["ok_accurate"] & df["ok_not_pit"] & df["ok_green"]
    n_valid = valid.groupby(df["stint_uid"]).transform("sum")
    df["stint_valid_laps"] = n_valid
    df["ok_stint_len"] = n_valid >= MIN_STINT_LAPS

    df["is_clean"] = np.logical_and.reduce(
        [df[c].fillna(False).astype(bool) for c, _ in RULES]
    )
    df["stint_clean_laps"] = df["is_clean"].groupby(df["stint_uid"]).transform("sum")
    return df


def cascade_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Laps remaining after each rule is applied in turn — the Decompose story."""
    rows = [{"step": "raw laps", "rule": "—", "laps": int(len(df)),
             "dropped": 0}]
    keep = pd.Series(True, index=df.index)
    for col, label in RULES:
        before = int(keep.sum())
        keep = keep & df[col].fillna(False).astype(bool)
        after = int(keep.sum())
        rows.append({"step": col, "rule": label, "laps": after,
                     "dropped": before - after})
    return pd.DataFrame(rows)


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_clean"]].copy().reset_index(drop=True)


def compound_summary(df: pd.DataFrame) -> pd.DataFrame:
    c = clean_laps(df)
    if c.empty:
        return pd.DataFrame(columns=["compound", "laps", "stints", "median_lap_s"])
    return (
        c.groupby("compound")
        .agg(laps=("lap_time_s", "size"),
             stints=("stint_uid", "nunique"),
             median_lap_s=("lap_time_s", "median"))
        .reset_index()
        .sort_values("laps", ascending=False)
    )
