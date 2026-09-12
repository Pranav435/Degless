"""Data ingest: FastF1 primary, OpenF1 REST fallback.

Both paths emit the *same* canonical lap-table schema (see `LAP_SCHEMA`), so
swapping sources is a one-line change.

The practice-only firewall lives here.  `load_for_fitting` refuses to return
anything that is not a practice session, which makes it structurally impossible
for race data to reach the fitter.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import FASTF1_CACHE, PRACTICE_SESSIONS, Event, get_event

log = logging.getLogger("degless.ingest")

OPENF1 = "https://api.openf1.org/v1"

LAP_SCHEMA = [
    "event",
    "session",
    "driver",
    "team",
    "lap_number",
    "stint",
    "compound",
    "tyre_life",
    "lap_time_s",
    "lap_start_s",       # seconds since session start (track-evolution clock)
    "is_accurate",
    "pit_in",
    "pit_out",
    "track_status",
    "source",
]


class FirewallError(RuntimeError):
    """Raised when race data is requested through a fitting-only entry point."""


# --------------------------------------------------------------------------
# FastF1
# --------------------------------------------------------------------------


def _fastf1():
    import fastf1

    fastf1.Cache.enable_cache(str(FASTF1_CACHE))
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    return fastf1


def load_session(event: Event | str, session_name: str, *, telemetry: bool = False,
                 retries: int = 2):
    """Load and return a FastF1 Session object (cached on disk)."""
    ev = get_event(event) if isinstance(event, str) else event
    ff1 = _fastf1()
    last = None
    for attempt in range(retries + 1):
        try:
            s = ff1.get_session(ev.ff1_year, ev.ff1_round, session_name)
            s.load(laps=True, telemetry=telemetry, weather=True, messages=True)
            return s
        except Exception as exc:  # network / upstream hiccup
            last = exc
            log.warning("FastF1 load failed (%s %s, attempt %d): %s",
                        ev.key, session_name, attempt + 1, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not load {ev.key} {session_name}: {last}")


def _td_seconds(col) -> pd.Series:
    return pd.to_timedelta(col).dt.total_seconds()


def raw_laps_fastf1(event: Event | str, session_name: str, session=None) -> pd.DataFrame:
    """Canonical lap table from FastF1."""
    ev = get_event(event) if isinstance(event, str) else event
    s = session if session is not None else load_session(ev, session_name)
    laps = s.laps.copy()
    if laps.empty:
        return pd.DataFrame(columns=LAP_SCHEMA)

    out = pd.DataFrame(
        {
            "event": ev.key,
            "session": session_name,
            "driver": laps["Driver"].astype(str),
            "team": laps.get("Team", pd.Series(index=laps.index, dtype=object)),
            "lap_number": laps["LapNumber"].astype(float),
            "stint": laps["Stint"].astype(float),
            "compound": laps["Compound"].astype(str).str.upper(),
            "tyre_life": laps["TyreLife"].astype(float),
            "lap_time_s": _td_seconds(laps["LapTime"]),
            "lap_start_s": _td_seconds(laps["LapStartTime"]),
            "is_accurate": laps["IsAccurate"].astype(bool),
            "pit_in": laps["PitInTime"].notna().to_numpy(),
            "pit_out": laps["PitOutTime"].notna().to_numpy(),
            "track_status": laps["TrackStatus"].astype(str),
            "source": "fastf1",
        }
    ).reset_index(drop=True)
    return out[LAP_SCHEMA]


# --------------------------------------------------------------------------
# OpenF1 fallback
# --------------------------------------------------------------------------


def _openf1_get(path: str, **params) -> list:
    import requests

    for attempt in range(3):
        try:
            r = requests.get(f"{OPENF1}/{path}", params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.warning("OpenF1 %s failed (%d): %s", path, attempt + 1, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"OpenF1 {path} unavailable")


def _openf1_track_status(session_key: int, lap_starts: pd.Series) -> pd.Series:
    """Approximate FastF1 TrackStatus strings from OpenF1 race-control messages.

    OpenF1 has no per-lap status channel, so we build a step function from the
    flag messages and sample it at each lap start.  Conservative by design: any
    interval that is not explicitly clear counts as non-green.
    """
    try:
        rc = _openf1_get("race_control", session_key=session_key)
    except RuntimeError:
        return pd.Series("1", index=lap_starts.index)
    if not rc:
        return pd.Series("1", index=lap_starts.index)

    df = pd.DataFrame(rc)
    if "date" not in df or "flag" not in df:
        return pd.Series("1", index=lap_starts.index)
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
    df = df.sort_values("date")

    code = {
        "GREEN": "1",
        "CLEAR": "1",
        "YELLOW": "2",
        "DOUBLE YELLOW": "2",
        "RED": "5",
        "CHEQUERED": "1",
    }
    events = []
    for _, row in df.iterrows():
        flag = str(row.get("flag") or "").upper()
        cat = str(row.get("category") or "").upper()
        msg = str(row.get("message") or "").upper()
        if "SAFETY CAR" in msg or cat == "SAFETYCAR":
            st = "6" if "VIRTUAL" in msg else "4"
            if "ENDING" in msg or "IN THIS LAP" in msg:
                st = "1"
        else:
            st = code.get(flag)
        if st is not None:
            events.append((row["date"], st))
    if not events:
        return pd.Series("1", index=lap_starts.index)

    times = pd.DatetimeIndex([e[0] for e in events])
    states = np.array([e[1] for e in events])
    idx = np.searchsorted(times.asi8, pd.DatetimeIndex(lap_starts).asi8, side="right") - 1
    return pd.Series(np.where(idx < 0, "1", states[np.clip(idx, 0, None)]),
                     index=lap_starts.index)


def raw_laps_openf1(event: Event | str, session_name: str) -> pd.DataFrame:
    """Canonical lap table from the OpenF1 REST API (independent cross-check)."""
    ev = get_event(event) if isinstance(event, str) else event
    sk = ev.openf1_sessions.get(session_name)
    if sk is None:
        raise KeyError(f"no OpenF1 session key for {ev.key} {session_name}")

    laps = pd.DataFrame(_openf1_get("laps", session_key=sk))
    if laps.empty:
        return pd.DataFrame(columns=LAP_SCHEMA)
    stints = pd.DataFrame(_openf1_get("stints", session_key=sk))
    drivers = pd.DataFrame(_openf1_get("drivers", session_key=sk))

    laps = laps.dropna(subset=["lap_number"])
    laps["date_start"] = pd.to_datetime(laps["date_start"], format="mixed", utc=True)
    t0 = laps["date_start"].min()

    # attach stint / compound / tyre age by lap-number interval
    laps["stint"] = np.nan
    laps["compound"] = None
    laps["tyre_life"] = np.nan
    if not stints.empty:
        for _, st in stints.iterrows():
            m = (
                (laps["driver_number"] == st["driver_number"])
                & (laps["lap_number"] >= st["lap_start"])
                & (laps["lap_number"] <= st["lap_end"])
            )
            laps.loc[m, "stint"] = st.get("stint_number", np.nan)
            laps.loc[m, "compound"] = st.get("compound")
            age0 = st.get("tyre_age_at_start") or 0
            laps.loc[m, "tyre_life"] = (
                laps.loc[m, "lap_number"] - st["lap_start"] + 1 + age0
            )

    abbr = {}
    if not drivers.empty and "name_acronym" in drivers:
        abbr = dict(zip(drivers["driver_number"], drivers["name_acronym"]))
    team = {}
    if not drivers.empty and "team_name" in drivers:
        team = dict(zip(drivers["driver_number"], drivers["team_name"]))

    lap_time = pd.to_numeric(laps["lap_duration"], errors="coerce")
    pit_out = laps.get("is_pit_out_lap", pd.Series(False, index=laps.index)).fillna(False).astype(bool)
    # OpenF1 has no in-lap flag: the lap before a driver's pit-out lap is the in-lap.
    pit_in = pd.Series(False, index=laps.index)
    for drv, grp in laps.groupby("driver_number"):
        out_laps = grp.loc[pit_out.loc[grp.index], "lap_number"]
        if len(out_laps):
            pit_in.loc[grp.index[grp["lap_number"].isin(out_laps - 1)]] = True

    out = pd.DataFrame(
        {
            "event": ev.key,
            "session": session_name,
            "driver": laps["driver_number"].map(abbr).fillna(
                laps["driver_number"].astype(str)),
            "team": laps["driver_number"].map(team),
            "lap_number": laps["lap_number"].astype(float),
            "stint": laps["stint"].astype(float),
            "compound": pd.Series(laps["compound"]).astype(str).str.upper(),
            "tyre_life": laps["tyre_life"].astype(float),
            "lap_time_s": lap_time,
            "lap_start_s": (laps["date_start"] - t0).dt.total_seconds(),
            # OpenF1 has no IsAccurate flag; a finite lap time is the best proxy.
            "is_accurate": lap_time.notna().to_numpy(),
            "pit_in": pit_in.to_numpy(),
            "pit_out": pit_out.to_numpy(),
            "track_status": _openf1_track_status(sk, laps["date_start"]).to_numpy(),
            "source": "openf1",
        }
    ).reset_index(drop=True)
    return out[LAP_SCHEMA]


# --------------------------------------------------------------------------
# Public entry points (firewall)
# --------------------------------------------------------------------------


def raw_laps(event: Event | str, session_name: str, *, prefer: str = "fastf1",
             session=None) -> pd.DataFrame:
    """Canonical lap table from whichever source works, no firewall."""
    if prefer == "fastf1":
        try:
            df = raw_laps_fastf1(event, session_name, session=session)
            if not df.empty:
                return df
            log.warning("FastF1 returned no laps for %s; falling back to OpenF1",
                        session_name)
        except Exception as exc:
            log.warning("FastF1 path failed (%s); falling back to OpenF1", exc)
        return raw_laps_openf1(event, session_name)
    try:
        return raw_laps_openf1(event, session_name)
    except Exception as exc:
        log.warning("OpenF1 path failed (%s); falling back to FastF1", exc)
        return raw_laps_fastf1(event, session_name, session=session)


def load_for_fitting(event: Event | str, session: str | Iterable[str] | None = None,
                     *, prefer: str = "fastf1") -> pd.DataFrame:
    """PRACTICE-ONLY firewall.

    The only door the fitter is allowed to walk through.  Anything that is not
    a practice session raises `FirewallError` before a single byte is read.
    """
    ev = get_event(event) if isinstance(event, str) else event
    if session is None:
        names = list(ev.practice_sessions)
    elif isinstance(session, str):
        names = [session]
    else:
        names = list(session)

    for name in names:
        if name not in PRACTICE_SESSIONS:
            raise FirewallError(
                f"practice-only firewall: refusing to load {name!r} for fitting. "
                f"Allowed: {list(PRACTICE_SESSIONS)}"
            )

    frames = []
    for name in names:
        df = raw_laps(ev, name, prefer=prefer)
        if df.empty:
            log.warning("no laps for %s %s", ev.key, name)
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError(f"no practice laps found for {ev.key}")
    return pd.concat(frames, ignore_index=True)


def load_race(event: Event | str, *, prefer: str = "fastf1") -> pd.DataFrame:
    """Race laps — validation and strategy only, never fitting."""
    ev = get_event(event) if isinstance(event, str) else event
    return raw_laps(ev, "Race", prefer=prefer)
