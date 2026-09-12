"""Tests for the live layer.

Run: .venv/bin/python -m pytest tests -q

The parser tests replay archived sessions (data/raw/livetiming/*) and compare
against FastF1's own lap tables from the local cache; they are skipped when
either is missing.  The engine tests replay the first part of a race through
the identical code path the live daemon uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live.merge import merge  # noqa: E402
from src.live.sources import RecordedSource  # noqa: E402
from src.live.state import LiveState  # noqa: E402
from src.live.streams import inflate, parse_gap, parse_laptime, parse_stream_line  # noqa: E402

FP2 = ROOT / "data" / "raw" / "livetiming" / "2026_italy_fp2"
RACE = ROOT / "data" / "raw" / "livetiming" / "2026_hungary_race"


# --------------------------------------------------------------------------
# merge / streams
# --------------------------------------------------------------------------


def test_merge_dict_list_and_deleted():
    base = {"Lines": {"1": {"Sectors": [{"Value": "1"}, {"Value": "2"}, {"Value": "3"}]}}}
    merge(base, {"Lines": {"1": {"Sectors": {"2": {"Value": "9", "PersonalFastest": True}}}}})
    assert base["Lines"]["1"]["Sectors"][2] == {"Value": "9", "PersonalFastest": True}
    assert base["Lines"]["1"]["Sectors"][0] == {"Value": "1"}
    merge(base, {"Lines": {"1": {"Sectors": {"4": {"Value": "x"}}}}})
    assert len(base["Lines"]["1"]["Sectors"]) == 5
    merge(base, {"Lines": {"_deleted": ["1"], "2": {"a": 1}}})
    assert "1" not in base["Lines"] and base["Lines"]["2"] == {"a": 1}
    assert merge(None, {"a": {"b": 1}}) == {"a": {"b": 1}}
    assert merge({"a": 1}, {"a": [1, 2]}) == {"a": [1, 2]}


def test_parsers():
    assert parse_laptime("1:23.456") == pytest.approx(83.456)
    assert parse_laptime("16:39.662") == pytest.approx(999.662)
    assert parse_laptime("") is None
    assert parse_gap("+1.234") == pytest.approx(1.234)
    assert parse_gap("1 L") is None and parse_gap("LAP 1") is None and parse_gap("") is None
    m = parse_stream_line("TrackStatus", '00:06:07.109{"Status":"1","Message":"AllClear"}\n')
    assert m.t_session == pytest.approx(367.109) and m.payload["Status"] == "1"
    import base64, json, zlib
    raw = json.dumps({"Entries": [{"Utc": "2026-01-01T00:00:00Z", "Cars": {"1": {"Channels": {"2": 300}}}}]}).encode()
    co = zlib.compressobj(wbits=-15)
    blob = base64.b64encode(co.compress(raw) + co.flush()).decode()
    assert inflate(blob)["Entries"][0]["Cars"]["1"]["Channels"]["2"] == 300


# --------------------------------------------------------------------------
# state vs FastF1
# --------------------------------------------------------------------------


def _fastf1_laps(rnd: int, session: str) -> pd.DataFrame | None:
    try:
        import fastf1
        fastf1.Cache.enable_cache(str(ROOT / "data" / "raw" / "fastf1_cache"))
        import logging
        logging.getLogger("fastf1").setLevel(logging.ERROR)
        s = fastf1.get_session(2026, rnd, session)
        s.load(laps=True, telemetry=False, weather=False, messages=False)
        ff = s.laps.copy()
    except Exception:
        return None
    ff["lap_time_s"] = ff["LapTime"].dt.total_seconds()
    ff = ff.rename(columns={"Driver": "driver", "LapNumber": "lap_number"})
    ff["pit_in"] = ff["PitInTime"].notna()
    ff["pit_out"] = ff["PitOutTime"].notna()
    return ff


def _replay(session_dir: Path) -> LiveState:
    st = LiveState()
    for m in RecordedSource(session_dir, include_zipped=False):
        st.apply(m)
    return st


@pytest.mark.parametrize("session_dir,rnd,session,min_match", [
    (FP2, 13, "Practice 2", 0.995),
    (RACE, 11, "Race", 0.97),
])
def test_state_matches_fastf1(session_dir, rnd, session, min_match):
    if not session_dir.exists():
        pytest.skip("archived streams missing")
    ff = _fastf1_laps(rnd, session)
    if ff is None:
        pytest.skip("FastF1 cache missing")
    st = _replay(session_dir)
    live = st.laps_df()
    m = ff.merge(live, on=["driver", "lap_number"], how="inner", suffixes=("_ff", "_lv"))
    assert len(m) >= 0.98 * len(ff)
    both = m[m["lap_time_s_ff"].notna() & m["lap_time_s_lv"].notna()]
    assert ((both["lap_time_s_ff"] - both["lap_time_s_lv"]).abs() < 1e-6).mean() > 0.999
    assert (m["Compound"].str.upper() == m["compound"]).mean() >= min_match
    assert (m["TyreLife"] == m["tyre_life"]).mean() >= min_match
    assert (m["pit_in_ff"] == m["pit_in_lv"]).mean() >= 0.999
    assert (m["pit_out_ff"] == m["pit_out_lv"]).mean() >= 0.999
    assert (m["IsAccurate"] == m["is_accurate"]).mean() >= 0.99


def test_state_race_session_facts():
    if not RACE.exists():
        pytest.skip("archived streams missing")
    st = _replay(RACE)
    assert st.is_race
    assert st.lap_count == {"current": 70, "total": 70}
    assert st.session_status in ("Finalised", "Ends", "Finished")
    codes = {c for _, c, _ in st.track_status_log}
    assert "1" in codes
    assert len(st.drivers) >= 20
    snap = st.field_snapshot()
    assert snap[0]["position"] == 1
    assert all(r["compound"] in ("SOFT", "MEDIUM", "HARD") for r in snap if r["compound"])


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------


def test_race_engine_replay_to_lap_25():
    if not RACE.exists():
        pytest.skip("archived streams missing")
    from src.live.engine import RaceEngine, WeekendModel

    wm = WeekendModel.load("hungary-2026")
    eng = RaceEngine(wm)
    st = LiveState()
    ticks = 0
    for m in RecordedSource(RACE, include_zipped=False):
        st.apply(m)
        evs = st.drain_events()
        if any(k == "lap" for _, k, _ in evs) and st.lap_count.get("current") in (5, 12, 20, 25):
            snap = eng.tick(st)
            ticks += 1
        if (st.lap_count.get("current") or 0) > 25:
            break
    assert ticks >= 3
    field = snap["field"]
    assert len(field) >= 18
    lead = field[0]
    assert lead["position"] == 1
    assert 0 <= lead["wear"] < 1.5
    assert 0 <= lead["p_past_cliff"] <= 1
    plan = lead["plan"]
    assert plan is not None and plan["best"]
    assert plan["now_lap"] == int(st.tracks[lead["driver_number"]].current["lap_number"])
    assert plan["n_options"] > 10
    assert np.isfinite(plan["delta_box_now_s"])
    for w in plan["window"]:
        assert w["lap"] >= plan["now_lap"] and w["loss_s"] >= 0
    m = snap["meta"]["regime_multiplier"]
    assert 0.05 < m["mean"] < 2.0 and m["p05"] <= m["mean"] <= m["p95"]
    assert 10 < snap["meta"]["pit_loss_s"] < 40
    assert len(snap["meta"]["evolution_s"]) == snap["meta"]["total_laps"]
    # history-worthy: every field row serialises
    from src.live.store import _clean
    import json
    json.dumps(_clean(snap))


def test_practice_engine_replay():
    if not FP2.exists():
        pytest.skip("archived streams missing")
    from src.live.engine import PracticeEngine, WeekendModel

    wm = WeekendModel.load("italy-2026")
    eng = PracticeEngine(wm)
    st = _replay(FP2)
    snap = eng.tick(st)
    assert snap["engine"] == "practice"
    assert len(snap["board"]) >= 10
    for row in snap["board"]:
        assert row["n_laps"] >= 4 and np.isfinite(row["slope_s_per_lap"])
    assert set(snap["prior"]) == {"SOFT", "MEDIUM", "HARD"}


def test_prior_model_without_fit():
    from src.live.engine import WeekendModel
    from src.config import get_event
    import numpy as np
    wm_model = WeekendModel.prior_model(get_event("azerbaijan-2026"), 200, np.random.default_rng(1))
    life = {c: float(wm_model.life_laps(c, 1.0).mean()) for c in wm_model.compounds}
    assert life["SOFT"] < life["MEDIUM"] < life["HARD"]


def test_engine_respects_circuit_stint_caps():
    """A compound the circuit has never run long is never planned long."""
    if not RACE.exists():
        pytest.skip("archived streams missing")
    from src.live.engine import RaceEngine, WeekendModel

    wm = WeekendModel.load("hungary-2026")
    wm.stint_cap = {"SOFT": 12, "MEDIUM": 30, "HARD": 40}
    eng = RaceEngine(wm)
    st = LiveState()
    for m in RecordedSource(RACE, include_zipped=False):
        st.apply(m)
        if (st.lap_count.get("current") or 0) >= 20:
            break
    snap = eng.tick(st)
    checked = 0
    for r in snap["field"]:
        p = r.get("plan")
        if not p:
            continue
        checked += 1
        total = snap["meta"]["total_laps"]
        for o in p["options"]:
            lab = o["label"]
            if lab.startswith("1 stop"):
                c2 = lab.split(":")[1].split(" on lap ")[0].strip()
                stop = int(lab.split("lap ")[-1])
                assert total - stop <= wm.stint_cap[c2], lab
                assert stop - p["now_lap"] + p["stint_len_now"] <= wm.stint_cap[r["compound"]] + 1 \
                    or stop - p["now_lap"] <= 2, (lab, p["stint_len_now"])
            if lab == "stay out":
                assert p["stint_len_now"] + p["laps_remaining"] <= wm.stint_cap[r["compound"]] + 1
    assert checked >= 10
