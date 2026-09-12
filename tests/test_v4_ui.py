"""The Haas tab against a synthetic live decision.

No recorded session under `data/live/` carries a `plan["decision"]` yet - they
were all captured before WP-D wired the live race-execution call into the
snapshot - so this builds one lap's snapshot directly from the keys
`src.live.engine.RaceEngine._decide`/`_plan_finish` write (verified against
that module: `plan["decision"]` with `action`, `confidence`,
`projected_position`, `delta_vs_alternative_s`, `rivals`, `life`; the action
table under `plan["race_state"]["actions"]`), then checks the Haas tab's
panels print the model's own action, confidence and rival for one car, and
"-" wherever a field the other car has no value for.

Run: .venv/bin/python -m pytest tests/test_v4_ui.py -q
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app.haas_tab as haas_tab  # noqa: E402
import app.live_tab as live_tab  # noqa: E402
import src.config as config  # noqa: E402
import src.live.store as store  # noqa: E402

SCRIPT = ROOT / "tests" / "_haas_ui_app.py"
SESSION = "synth"


def _car_row(*, driver, number, position, compound, tyre_age, last_lap, deg_now, interval, plan=None) -> dict:
    """The keys `src.live.state.LiveState.field_snapshot` writes for one car,
    plus the per-tick additions `RaceEngine.tick` layers on (`plan`, minus the
    fields this test does not touch)."""
    row = {
        "driver_number": str(number), "driver": driver, "team": "Haas F1 Team", "position": position,
        "gap_leader_s": position * 1.4, "gap_leader": f"+{position * 1.4:.1f}",
        "interval_s": interval, "interval": (f"+{interval:.1f}" if interval is not None else ""),
        "in_pit": False, "retired": False, "stopped": False, "laps_complete": 20,
        "current_lap": 20, "compound": compound, "tyre_age": tyre_age, "stint": 1,
        "n_pit_stops": 0, "last_lap_s": last_lap, "best_lap_s": last_lap - 0.3,
        "wear": 0.4, "p_past_cliff": 0.0, "laps_to_cliff_p10": 10.0, "laps_to_cliff_p50": 14.0,
        "laps_to_cliff_p90": 18.0, "deg_now_s_per_lap": deg_now, "level_s": 90.0, "n_clean": 6,
        "proj": [], "cliff_alarm": False, "pace_collapse": False, "wear_alarm": False,
    }
    if plan is not None:
        row["plan"] = plan
        row["undercut"] = {"threat": None, "opportunity": None}
        row["rejoin_if_box_now"] = None
    return row


def _synthetic_snapshot() -> dict:
    oco_rival = {
        "driver": "RUS", "driver_number": "63", "gap_s": -2.3, "virtual_gap_s": -2.6, "on_track": "behind",
        "compound": "HARD", "tyre_age": 20.0, "stops": 1, "in_pit": False, "pending_stop": True,
        "stop_lap_median": 25, "if_cover": None,
        "why": "2.3 s behind on track, its own stop is due on lap 25",
    }
    oco_plan = {
        "best": "2-stop M-H-H @ 16,42", "window_lo": 16, "window_hi": 19,
        "laps_remaining": 44, "now_lap": 16,
        "decision": {
            "action": "PIT NOW", "lap": 16, "wait_laps": 0, "box_by_lap": None,
            "confidence": 0.71, "projected_position": 7, "projected_position_if_now": 7,
            "delta_vs_alternative_s": 0.4, "rivals": [oco_rival],
            "if_cover": None,
            "life": {"p_cliff_before_stop": 0.0, "life_p10": 10.0, "life_p50": 14.0, "life_p90": 18.0},
        },
        "race_state": {
            "actions": [
                {"action": "PIT NOW", "lap": 16, "legal": True, "cost_s": 160.0,
                 "tyre_s": 1.0, "position_s": 2.0, "traffic_s": 0.1},
                {"action": "STAY OUT 1 LAP", "lap": 17, "legal": True, "cost_s": 160.4,
                 "tyre_s": 0.5, "position_s": 2.6, "traffic_s": 0.2},
            ],
            "rivals": [oco_rival], "place_value_s": 3.0, "sigma_rel_s": 2.7,
        },
    }
    bea_plan = {
        "best": "2-stop M-H-H @ 16,42", "window_lo": 22, "window_hi": 25,
        "laps_remaining": 40, "now_lap": 20,
        "decision": {
            "action": "WAIT 2 LAPS", "lap": 22, "wait_laps": 2, "box_by_lap": None,
            "confidence": 0.55, "projected_position": None, "projected_position_if_now": None,
            "delta_vs_alternative_s": None, "rivals": [],
            "if_cover": None,
            "life": {"p_cliff_before_stop": 0.0, "life_p10": 12.0, "life_p50": 16.0, "life_p90": 20.0},
        },
        "race_state": {
            "actions": [
                {"action": "PIT NOW", "lap": 20, "legal": True, "cost_s": 205.0,
                 "tyre_s": 2.0, "position_s": 1.0, "traffic_s": 0.1},
                {"action": "WAIT 2 LAPS", "lap": 22, "legal": True, "cost_s": 204.0,
                 "tyre_s": 0.8, "position_s": 1.5, "traffic_s": 0.05},
            ],
            "rivals": [], "place_value_s": 3.0, "sigma_rel_s": 2.7,
        },
    }
    rus_row = _car_row(driver="RUS", number=63, position=7, compound="HARD", tyre_age=20.0,
                       last_lap=81.0, deg_now=0.02, interval=2.3)
    oco_row = _car_row(driver="OCO", number=31, position=6, compound="MEDIUM", tyre_age=16.0,
                       last_lap=82.4, deg_now=0.05, interval=1.1, plan=oco_plan)
    bea_row = _car_row(driver="BEA", number=87, position=12, compound="HARD", tyre_age=24.0,
                       last_lap=83.9, deg_now=0.03, interval=0.6, plan=bea_plan)
    return {
        "engine": "race",
        "meta": {"lap_count": {"current": 20, "total": 70}, "track_status": "1", "sc_active": False,
                 "tick_utc": "2026-09-13T12:00:00+00:00"},
        "field": [rus_row, oco_row, bea_row],
        "alerts": [],
    }


def _texts(node, out: list) -> list:
    proto = getattr(node, "proto", None)
    if proto is not None:
        out.append(str(proto))
    for child in (getattr(node, "children", None) or {}).values():
        _texts(child, out)
    return out


def _no_leaked_none(at: AppTest) -> None:
    """No literal Python `None` printed in place of a "-": chart element ids
    and encoded chart data carry unrelated "...-None" fragments, so those are
    stripped first, exactly as `tests/test_app_smoke.py`'s jargon check does."""
    text = "\n".join(_texts(at._tree, []))
    text = re.sub(r'\bid: "[^"]*"', "", text)
    text = re.sub(r"(?:[A-Za-z0-9+/=]|\\\\u002f){40,}", "", text)
    assert "None" not in text


def _metrics_by_label(at: AppTest) -> dict:
    out: dict = {}
    for m in at.metric:
        out.setdefault(m.label, []).append(m.value)
    return out


@pytest.fixture
def live_session(tmp_path, monkeypatch):
    sdir = tmp_path / SESSION
    sdir.mkdir(parents=True)
    (sdir / "snapshot.json").write_text(json.dumps(_synthetic_snapshot()))
    pd.DataFrame({
        "driver": ["OCO"] * 5, "lap_number": [12, 13, 14, 15, 16],
        "lap_time_s": [82.9, 82.7, 82.5, 82.6, 82.4], "compound": ["MEDIUM"] * 5,
        "pit_in": [False, False, False, False, False],
    }).to_parquet(sdir / "laps.parquet", index=False)
    # As the freshness test does for `config.DATA_PROCESSED`: point every
    # module's own `LIVE_DIR` name at the temporary directory, since each
    # holds its own copy from `from src.live.store import LIVE_DIR`.
    monkeypatch.setattr(store, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(haas_tab, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(live_tab, "LIVE_DIR", tmp_path)
    return tmp_path


def test_haas_tab_shows_the_live_decision(live_session):
    if not (config.DATA_PROCESSED / "meta_hungary-2026.json").exists():
        pytest.skip("no processed data for hungary-2026")
    at = AppTest.from_file(str(SCRIPT), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    metrics = _metrics_by_label(at)
    subheaders = [s.value for s in at.subheader]

    # OCO: a fully populated decision - the action, its confidence and the
    # relevant rival all come from the snapshot, verbatim.
    assert any("PIT NOW" in s for s in subheaders)
    assert "71%" in metrics.get("Confidence", [])
    assert any(v.startswith("RUS") for v in metrics.get("Relevant rival", []))

    # BEA: a real action and confidence, but no rejoin position, no expected
    # delta and no rival in this snapshot - every one of those prints "-",
    # never a fabricated number and never a leaked `None`.
    assert any("WAIT 2 LAPS" in s for s in subheaders)
    assert "55%" in metrics.get("Confidence", [])
    assert "—" in metrics.get("Relevant rival", [])
    assert "—" in metrics.get("Race-time delta", [])
    assert "—" in metrics.get("Rejoin position", []) or "—" in metrics.get("Projected position", [])

    _no_leaked_none(at)

    # The single-driver view must show the same live call, and its lap chart
    # (this session's synthetic `laps.parquet` has five OCO laps).
    at.button_group(key="haas_view_hungary-2026").set_value("Ocon").run()
    assert not at.exception, [e.value for e in at.exception]
    metrics = _metrics_by_label(at)
    assert "71%" in metrics.get("Confidence", [])
    assert any(v.startswith("RUS") for v in metrics.get("Relevant rival", []))
    subheaders = [s.value for s in at.subheader]
    assert any("lap times so far" in s for s in subheaders)
    _no_leaked_none(at)


def test_haas_tab_falls_back_to_the_pre_race_plan(monkeypatch, tmp_path):
    """No live session at all: the panels read the pre-race plan instead, and
    still never fabricate a live-only field (position, tyre age, gaps)."""
    if not (config.DATA_PROCESSED / "meta_hungary-2026.json").exists():
        pytest.skip("no processed data for hungary-2026")
    monkeypatch.setattr(store, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(haas_tab, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(live_tab, "LIVE_DIR", tmp_path)
    at = AppTest.from_file(str(SCRIPT), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    _no_leaked_none(at)
    text = "\n".join(_texts(at._tree, []))
    assert "pre-race plan" in text.lower() or "Pre-race plan" in text
