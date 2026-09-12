"""The practice -> race regime, measured on this circuit's *previous* years.

`src.regime` measures the ratio of race degradation to practice degradation on
the 2026 weekends and pools it across weekends.  That pool says what a 2026 car
and tyre do on average; it cannot say that **Spa is a management circuit and
Monza is not**.  Measured per weekend the 2026 ratio runs 0.32 at Spa to 0.81 at
Monza, and a pooled 0.5-0.6 transferred to every weekend was V2's single
largest degradation error.

The circuit itself has the missing information, three years deep: FP1-FP3 of
2023, 2024 and 2025, against those years' races, on the same circuit, measured
with the **same estimator** - stint fixed effects on fuel- and
evolution-corrected laps - and the same clean-lap cascade.  Only the physics
changes: those cars burned 1.67 kg/lap at ~0.033 s/kg
(`history.FUEL_S_PER_LAP_PRE2026`) rather than 2026's ~70 kg over a race
distance, and the race distance is that year's.  Both are passed explicitly into
`regime.measure_regime_frames`, which is the same code path the 2026 weekends
take, so a historical ratio and a 2026 ratio are the same quantity.

**What it is not.**  A 2023 Pirelli on a 2023 car is not a 2026 tyre on a 2026
car: the ratio transfers the *driving*, not the rubber.  That assumption is
priced, not hidden - `circuit_regime_prior` adds `TYRE_GENERATION_LN_SD`
(0.15 ln, about +/-16%) to the spread of the years, which is the honest width for
"teams manage this circuit like that, with a different tyre".  It is then
combined with the 2026 donor pool by precision weights, never substituted for
it.

**Cost and caching.**  The FastF1 cache that ships with this repo holds races
only, so each practice session is one network load of ~17 s
(`scripts/05_history_practice.py` does it once, deliberately, and prints what it
got).  Every measurement is then summarised to
`data/processed/history/regime_<year>_<circuit>.json` and `circuit_regime_prior`
reads nothing but those files - so `make history` and every weekend build stay
offline, and a circuit nobody has fetched simply has no prior.

A rain-flagged race cannot be compared with a dry practice at all, so it is
cached as missing; a rain-flagged practice session is dropped, and if that
leaves no practice the year is missing too.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Event, get_event
from src.evolution import add_evolution_correction, fit_evolution_auto
from src.history import (
    FUEL_S_PER_LAP_PRE2026,
    HIST_DIR,
    MISSING_RETRY_S,
    YEARS,
    _fastf1,
    same_circuit,
)
from src.ingest import LAP_SCHEMA
from src.laps import build_lap_table, clean_laps
from src.regime import REGIME_LN_SD_FLOOR, REGIME_RATIO_BAND, measure_regime_frames

log = logging.getLogger("degless.regime_history")

REGIME_HISTORY_VERSION = 1
DEFAULT_SESSIONS = ("Practice 1", "Practice 2", "Practice 3")
# The assumption this module rests on, stated as a number: a previous
# generation's tyre and car on the same circuit tell you the management regime
# to within about +/-16%.  Added in quadrature to the spread between years.
TYRE_GENERATION_LN_SD = 0.15
CIRCUIT_SPREAD_LN_FLOOR = 0.20     # one year alone is never tighter than this


def slug(circuit: str) -> str:
    """The cache-file spelling of a circuit, as `history.summarise_race` uses."""
    return str(circuit).lower().replace(" ", "-")


def cache_path(year: int, circuit: str) -> Path:
    return HIST_DIR / f"regime_{int(year)}_{slug(circuit)}.json"


# --------------------------------------------------------------------------
# One circuit-year
# --------------------------------------------------------------------------


def _canonical(session, *, event: str, session_name: str) -> pd.DataFrame:
    """`ingest.LAP_SCHEMA` for an arbitrary FastF1 session.

    `ingest.raw_laps_fastf1` needs a 2026 `Event` to name the weekend; a 2023
    race has no `Event`, so the same columns are built here with the event
    spelled `"<year>-<circuit>"`.  `lap_start_s` is `LapStartTime`, the clock
    `src.evolution` fits the track curve against.
    """
    laps = session.laps.copy()
    if laps.empty:
        return pd.DataFrame(columns=LAP_SCHEMA)
    out = pd.DataFrame(
        {
            "event": event,
            "session": session_name,
            "driver": laps["Driver"].astype(str),
            "team": laps.get("Team", pd.Series(index=laps.index, dtype=object)),
            "lap_number": laps["LapNumber"].astype(float),
            "stint": laps["Stint"].astype(float),
            "compound": laps["Compound"].astype(str).str.upper(),
            "tyre_life": laps["TyreLife"].astype(float),
            "lap_time_s": pd.to_timedelta(laps["LapTime"]).dt.total_seconds(),
            "lap_start_s": pd.to_timedelta(laps["LapStartTime"]).dt.total_seconds(),
            "is_accurate": laps["IsAccurate"].astype(bool),
            "pit_in": laps["PitInTime"].notna().to_numpy(),
            "pit_out": laps["PitOutTime"].notna().to_numpy(),
            "track_status": laps["TrackStatus"].astype(str),
            "source": "fastf1",
        }
    ).reset_index(drop=True)
    return out[LAP_SCHEMA]


def _weather(session) -> tuple:
    """(rain flag, median track temperature) of a loaded session."""
    w = getattr(session, "weather_data", None)
    if w is None or not len(w):
        return False, None
    rain = bool(w["Rainfall"].any()) if "Rainfall" in w else False
    temp = (float(w["TrackTemp"].median())
            if "TrackTemp" in w and w["TrackTemp"].notna().any() else None)
    return rain, temp


def _fuel_correct_pre2026(clean: pd.DataFrame) -> pd.DataFrame:
    """The pre-2026 fuel physics, written straight onto the frame.

    `fuel.add_fuel_correction` is keyed to a 2026 `Event` (it derives the burn
    from the 2026 allowance and that weekend's distance), so the old
    regulations' flat 1.67 kg/lap x 0.033 s/kg is applied here instead.  The
    columns are the ones `src.evolution` and `regime._practice_frame` consume,
    with the same sign convention as `fuel.FuelPrior.correct`.
    """
    out = clean.copy()
    out["fuel_prior"] = "pre-2026 physics"
    out["fuel_term_s"] = -FUEL_S_PER_LAP_PRE2026 * out["lap_in_stint"].astype(float)
    out["lap_time_fuel_corr"] = out["lap_time_s"] - out["fuel_term_s"]
    return out


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _write(p: Path, payload: dict) -> dict:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    payload = _nan_to_none(payload)
    p.write_text(json.dumps(payload, indent=1))
    return payload


def measure_regime_history(year: int, circuit: str, *, sessions=DEFAULT_SESSIONS,
                           offline: bool = False, force: bool = False) -> dict | None:
    """The practice -> race degradation ratio of one previous year, cached as JSON.

    Returns the cached summary, a `{"missing": True, "why": ...}` marker, or
    `None` when nothing could be established and nothing was cached (an offline
    run with no cache entry).  `offline=True` puts FastF1 in offline mode for the
    rest of the process, exactly as `scripts/10_pipeline.py --offline` does, and
    a failure under it is never written to the cache - it is a missing download,
    not a missing race.
    """
    p = cache_path(year, circuit)
    if p.exists() and not force:
        try:
            d = json.loads(p.read_text())
        except Exception:
            d = None
        if d is not None and int(d.get("version", 0)) >= REGIME_HISTORY_VERSION:
            stale = (d.get("missing") and d.get("retry")
                     and time.time() - float(d.get("checked", 0)) > MISSING_RETRY_S)
            if not stale:
                return d
    if offline:
        ff1 = _fastf1()
        ff1.Cache.offline_mode(True)

    event = f"{int(year)}-{slug(circuit)}"
    base = {"version": REGIME_HISTORY_VERSION, "year": int(year), "circuit": circuit,
            "event": event, "checked": time.time()}

    # -- the race ----------------------------------------------------------
    ff1 = _fastf1()
    try:
        rs = ff1.get_session(int(year), circuit, "Race")
        if not same_circuit(circuit, rs.event):
            # FastF1 matches event names fuzzily and hands back *some* race for
            # a name it does not know; a new circuit has no history.
            raise LookupError(f"nearest {year} event is {rs.event.get('EventName')} at "
                              f"{rs.event.get('Location')}, not {circuit}")
        rs.load(laps=True, telemetry=False, weather=True, messages=False)
        race_raw = _canonical(rs, event=event, session_name="Race")
    except Exception as exc:
        log.info("no %s race at %s: %s", year, circuit, str(exc)[:90])
        if offline:
            return None
        return _write(p, base | {"missing": True, "why": "no race", "retry": True,
                                 "error": str(exc)[:160]})
    race_rain, t_race = _weather(rs)
    if race_rain:
        return _write(p, base | {"missing": True, "why": "rain", "retry": False,
                                 "rain": True, "track_temp_race_c": t_race,
                                 "note": "a wet race cannot be compared with a dry practice"})
    if race_raw.empty:
        return _write(p, base | {"missing": True, "why": "no race laps", "retry": True})
    n_race_laps = int(getattr(rs, "total_laps", 0) or race_raw["lap_number"].max())
    race_tbl = build_lap_table(race_raw, None)

    # -- the practice sessions --------------------------------------------
    frames, used, skipped, rainy, temps = [], [], [], [], {}
    for name in sessions:
        try:
            ps = ff1.get_session(int(year), circuit, name)
            if not same_circuit(circuit, ps.event):
                raise LookupError("different circuit")
            ps.load(laps=True, telemetry=False, weather=True, messages=False)
            df = _canonical(ps, event=event, session_name=name)
        except Exception as exc:
            # A sprint weekend has one practice session, not three.
            log.info("%s %s %s unavailable: %s", year, circuit, name, str(exc)[:80])
            skipped.append(name)
            continue
        if df.empty:
            skipped.append(name)
            continue
        rain, temp = _weather(ps)
        if rain:
            rainy.append(name)
            continue
        frames.append(df)
        used.append(name)
        temps[name] = temp
    if not frames:
        why = "rain" if rainy and not used else "no practice"
        if offline and why == "no practice":
            return None
        return _write(p, base | {"missing": True, "why": why,
                                 "retry": bool(why == "no practice"),
                                 "rain": bool(rainy), "sessions_rain": rainy,
                                 "sessions_missing": skipped, "track_temp_race_c": t_race})

    laps = build_lap_table(pd.concat(frames, ignore_index=True), None)
    clean = clean_laps(laps)
    if clean.empty:
        return _write(p, base | {"missing": True, "why": "no clean practice laps", "retry": True,
                                 "sessions_used": used, "sessions_missing": skipped,
                                 "sessions_rain": rainy})
    clean = _fuel_correct_pre2026(clean)
    # `event=None`: the compound pace offsets `fit_evolution_push` would remove
    # are 2026 quantities (src.compounds is keyed to a 2026 Event), and a
    # historical nomination is a different ladder.  The push-lap curve is
    # fitted without them, which is the same approximation the 2026 path makes
    # for a session whose compounds do not change.
    evo = fit_evolution_auto(clean, laps_all=laps, event=None)
    clean = add_evolution_correction(clean, evo)

    m = measure_regime_frames(race_tbl, clean, event=event,
                              fuel_s_per_lap=FUEL_S_PER_LAP_PRE2026,
                              n_race_laps=n_race_laps)
    # Laps-weighted practice temperature over the sessions that actually
    # contributed clean long-run laps, as `history.practice_track_temp` does.
    w = clean["session"].value_counts()
    num = den = 0.0
    for name, n in w.items():
        t = temps.get(str(name))
        if t is not None:
            num += t * float(n)
            den += float(n)
    t_prac = float(num / den) if den > 0 else None

    out = base | {
        "ratio": float(m.ratio) if np.isfinite(m.ratio) else None,
        "per_compound": m.per_compound,
        "n_race_stints": int(m.n_race_stints), "n_practice_stints": int(m.n_practice_stints),
        "usable_compounds": list(m.usable_compounds),
        "n_race_laps": n_race_laps, "n_clean_practice_laps": int(len(clean)),
        "sessions_used": used, "sessions_missing": skipped, "sessions_rain": rainy,
        "rain": False,
        "track_temp_practice_c": t_prac, "track_temp_race_c": t_race,
        "evolution_range_s": float(evo.iterations[-1].get("evo_range_s", float("nan")))
        if evo.iterations else None,
    }
    if not np.isfinite(m.ratio):
        out |= {"missing": True, "retry": False,
                "why": "no compound with a usable practice slope"}
    return _write(p, out)


# --------------------------------------------------------------------------
# Pooled over the years: the prior `regime_prior` consumes
# --------------------------------------------------------------------------


def circuit_regime_prior(event: Event | str, *, years=YEARS) -> dict | None:
    """This circuit's practice -> race ratio pooled over the cached years.

    Cache-only by construction: a circuit-year that
    `scripts/05_history_practice.py` has not fetched is simply absent, so a
    weekend build never waits on the network and never fails for want of
    history.  Ratios outside `regime.REGIME_RATIO_BAND` are discarded exactly as
    2026 donors are - outside it the measurement is broken, not informative.

    `ln_sd` is the spread between years (MAD-based, floored at
    `CIRCUIT_SPREAD_LN_FLOOR`) with `TYRE_GENERATION_LN_SD` added in quadrature.
    """
    ev = get_event(event) if isinstance(event, str) else event
    circuit = ev.circuit
    by_year, dropped = {}, {}
    for y in years:
        p = cache_path(y, circuit)
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception as exc:
            log.warning("unreadable %s: %s", p.name, exc)
            continue
        if d.get("missing"):
            dropped[int(y)] = d.get("why", "missing")
            continue
        r = d.get("ratio")
        if r is None or not np.isfinite(float(r)):
            dropped[int(y)] = "no ratio"
            continue
        if not (REGIME_RATIO_BAND[0] <= float(r) <= REGIME_RATIO_BAND[1]):
            dropped[int(y)] = f"ratio {float(r):.2f} outside the believable band"
            continue
        by_year[int(y)] = float(r)
    if not by_year:
        return None
    logs = np.log(np.array(sorted(by_year.values()), dtype=float))
    centre = float(np.median(logs))
    if len(logs) > 1:
        mad = float(np.median(np.abs(logs - centre))) * 1.4826
        spread = float(max(mad, np.std(logs, ddof=1) * 0.8))
    else:
        spread = 0.0
    ln_sd = float(np.sqrt(max(spread, CIRCUIT_SPREAD_LN_FLOOR) ** 2 + TYRE_GENERATION_LN_SD ** 2))
    return {
        "ratio": float(np.exp(centre)),
        "ln_sd": float(max(ln_sd, REGIME_LN_SD_FLOOR)),
        "years": sorted(by_year),
        "by_year": {y: round(v, 4) for y, v in sorted(by_year.items())},
        "n": len(by_year),
        "spread_ln": float(spread),
        "circuit": circuit,
        "dropped": dropped,
        "source": f"{circuit} practice->race ratio, {sorted(by_year)}, "
                  f"pre-2026 fuel physics, same stint-FE estimator",
    }


def circuit_regime_table(events, *, years=YEARS) -> pd.DataFrame:
    """One row per event: the pooled prior and the years behind it (reporting)."""
    rows = []
    for key in events:
        ev = get_event(key) if isinstance(key, str) else key
        cp = circuit_regime_prior(ev, years=years)
        row = {"event": ev.key, "circuit": ev.circuit}
        if cp:
            row |= {"ratio": round(cp["ratio"], 4), "ln_sd": round(cp["ln_sd"], 4),
                    "years": cp["years"], "by_year": cp["by_year"], "n": cp["n"]}
        else:
            row |= {"ratio": np.nan, "ln_sd": np.nan, "years": [], "by_year": {}, "n": 0}
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["measure_regime_history", "circuit_regime_prior", "circuit_regime_table",
           "cache_path", "slug", "REGIME_HISTORY_VERSION", "TYRE_GENERATION_LN_SD",
           "DEFAULT_SESSIONS"]
