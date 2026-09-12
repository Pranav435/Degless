"""Corner apex speeds — the second observation channel.

Why this breaks the collinearity: fuel mass slows the car roughly uniformly and
dominates accelerating zones, while grip loss hits apex *minimum* speeds
disproportionately.  Two channels with different sensitivity signatures to the
same two latent causes make the split identifiable — something a single
lap-time channel mathematically cannot do.  Apex speeds are also robust to
practice engine modes, which wreck lap times but barely touch minimum corner
speeds.

This is the most time-expensive component in the pipeline: it needs per-lap
telemetry, so it wants a warm FastF1 cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    CORNER_WINDOW_M,
    DATA_PROCESSED,
    N_APEX_CORNERS,
    SLOW_CORNER_FRAC,
    Event,
    get_event,
)
from src.ingest import load_session

log = logging.getLogger("degless.telemetry")

APEX_SCHEMA = ["event", "session", "driver", "lap_number", "corner",
               "apex_speed", "compound", "tyre_age", "stint_uid"]


def _corner_apexes(tel: pd.DataFrame, corners: pd.DataFrame,
                   window_m: float) -> dict[int, float]:
    """Minimum speed within +/- window of each corner marker."""
    dist = tel["Distance"].to_numpy(dtype=float)
    speed = tel["Speed"].to_numpy(dtype=float)
    out: dict[int, float] = {}
    for _, row in corners.iterrows():
        d0 = float(row["Distance"])
        m = (dist >= d0 - window_m) & (dist <= d0 + window_m)
        if m.sum() >= 3:
            v = float(np.nanmin(speed[m]))
            if np.isfinite(v) and v > 0:
                out[int(row["Number"])] = v
    return out


def extract_apex_speeds(clean: pd.DataFrame, event: Event | str,
                        *, window_m: float = CORNER_WINDOW_M,
                        max_laps: int | None = None) -> pd.DataFrame:
    """Per-lap apex speed at every corner, for the clean laps of `clean`.

    Returns a long table keyed by (driver, lap_number, corner).
    """
    ev = get_event(event) if isinstance(event, str) else event
    frames = []
    for sess_name, grp in clean.groupby("session", sort=False):
        try:
            session = load_session(ev, sess_name, telemetry=True)
            corners = session.get_circuit_info().corners
        except Exception as exc:
            log.warning("no telemetry/circuit info for %s %s: %s",
                        ev.key, sess_name, exc)
            continue
        if corners is None or corners.empty:
            log.warning("no corner markers for %s %s", ev.key, sess_name)
            continue

        wanted = set(zip(grp["driver"], grp["lap_number"]))
        meta = grp.set_index(["driver", "lap_number"])
        laps = session.laps
        n_ok = n_fail = 0
        rows = []
        for _, lap in laps.iterrows():
            key = (str(lap["Driver"]), float(lap["LapNumber"]))
            if key not in wanted:
                continue
            try:
                tel = lap.get_car_data().add_distance()
            except Exception:
                n_fail += 1
                continue
            if tel is None or tel.empty or "Distance" not in tel:
                n_fail += 1
                continue
            apex = _corner_apexes(tel, corners, window_m)
            if not apex:
                n_fail += 1
                continue
            n_ok += 1
            m = meta.loc[key]
            for corner, v in apex.items():
                rows.append({
                    "event": ev.key, "session": sess_name, "driver": key[0],
                    "lap_number": key[1], "corner": corner, "apex_speed": v,
                    "compound": m["compound"], "tyre_age": float(m["tyre_age"]),
                    "stint_uid": m["stint_uid"],
                })
            if max_laps is not None and n_ok >= max_laps:
                break
        log.info("apex: %s %s -> %d laps ok, %d failed, %d corners",
                 ev.key, sess_name, n_ok, n_fail, len(corners))
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=APEX_SCHEMA)
    return pd.concat(frames, ignore_index=True)[APEX_SCHEMA]


@dataclass
class CornerSelection:
    corners: list
    table: pd.DataFrame


def select_corners(apex: pd.DataFrame, *, n: int = N_APEX_CORNERS,
                   slow_frac: float = SLOW_CORNER_FRAC) -> CornerSelection:
    """Keep the `n` highest-variance *slow* corners.

    Slow corners are where grip, not power, sets the speed, so they carry the
    most tyre information; among those, the ones whose apex speed varies most
    across laps carry the most *signal*.
    """
    if apex.empty:
        return CornerSelection([], pd.DataFrame())
    vmax = float(apex["apex_speed"].max())
    stats = (
        apex.groupby("corner")["apex_speed"]
        .agg(median="median", var="var", n="size")
        .reset_index()
    )
    stats["is_slow"] = stats["median"] < slow_frac * vmax
    pool = stats[stats["is_slow"] & (stats["n"] >= 20)]
    if pool.empty:
        log.warning("no slow corners passed selection; falling back to all corners")
        pool = stats[stats["n"] >= 20]
    picked = pool.nlargest(n, "var")["corner"].tolist()
    stats["selected"] = stats["corner"].isin(picked)
    return CornerSelection(sorted(picked), stats.sort_values("var", ascending=False))


def apex_path(event: Event | str) -> "object":
    ev = get_event(event) if isinstance(event, str) else event
    return DATA_PROCESSED / f"apex_{ev.key}.parquet"


def save_apex(apex: pd.DataFrame, event: Event | str) -> "object":
    p = apex_path(event)
    apex.to_parquet(p, index=False)
    return p


def load_apex(event: Event | str) -> pd.DataFrame:
    p = apex_path(event)
    if not p.exists():
        return pd.DataFrame(columns=APEX_SCHEMA)
    return pd.read_parquet(p)
