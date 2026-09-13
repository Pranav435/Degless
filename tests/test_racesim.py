"""The race simulation: the synthetic feed reproduces the truth inside the
live state, the engine drives the two Haas cars through it, the result is
JSON-safe and paired across modes, and the engine's evolution estimate reads
the race the right way round.

Run: .venv/bin/python -m pytest tests/test_racesim.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.config as config  # noqa: E402
from src.racesim import HAAS, SimConfig, RaceSim, _engine_validation, load_grid  # noqa: E402

EVENT = "spain-2026"


def _have_model() -> bool:
    return (config.DATA_PROCESSED / f"posterior_{EVENT}.npz").exists() and \
        (config.DATA_PROCESSED / f"weekend_{EVENT}.json").exists()


@pytest.fixture(scope="module")
def sim_result():
    if not _have_model():
        pytest.skip("no sealed model for spain-2026")
    from src.live.engine import WeekendModel

    wm = WeekendModel.load(EVENT)
    grid = load_grid(EVENT)
    sim = RaceSim(SimConfig(event=EVENT, seed=0), wm=wm, grid=grid)
    res = sim.run()
    return sim, res


def test_the_feed_reproduces_the_truth_in_the_live_state(sim_result):
    sim, res = sim_result
    laps = sim.state.laps_df(include_current=False)
    laps = laps[laps["is_complete"]]
    assert laps["driver"].nunique() == len(sim.cars)
    for car in sim.cars:
        d = laps[laps["driver"] == car.code].sort_values("lap_number")
        assert len(d) == sim.n, car.code
        assert np.allclose(d["lap_time_s"].to_numpy(), np.round(car.lap_times, 3), atol=2e-3), car.code
        assert list(d["compound"]) == car.compounds, car.code
        assert list(d["tyre_life"].astype(int)) == car.ages, car.code
        assert sorted(d[d["pit_in"]]["lap_number"].astype(int)) == [s["lap"] for s in car.stops], car.code
        assert sorted(d[d["pit_out"]]["lap_number"].astype(int)) == [s["lap"] + 1 for s in car.stops], car.code
    # a racing lap is accurate; in-laps and out-laps are not
    assert laps["is_accurate"].mean() > 0.9
    assert not laps[laps["pit_in"] | laps["pit_out"]]["is_accurate"].any()


def test_the_engine_drives_both_haas_cars_and_the_result_serialises(sim_result):
    sim, res = sim_result
    json.dumps(res)
    assert res["engine_in_loop"] and set(res["engine"]) == set(HAAS)
    for code in HAAS:
        recs = res["engine"][code]
        assert len(recs) == sim.n            # one record per tick (lap 0 .. n-1 completed)
        acts = [r["action_kind"] for r in recs if r.get("action_kind")]
        assert acts and set(acts) <= {"PIT_NOW", "STAY_OUT", "WAIT", "BOX_BY"}
        car = next(c for c in res["cars"] if c["code"] == code)
        assert len(car["compounds"]) >= 2 and len(car["stops"]) >= 1
        assert not car["forced_stop"], "the engine, not the fallback, made the stop"
        # the stop the car took is the lap the engine named on its in-lap
        first = car["stops"][0]["lap"]
        rec = next(r for r in recs if r["lap"] == first)
        assert rec["action_kind"] in ("PIT_NOW", "WAIT", "BOX_BY") and int(rec["dec_lap"]) == first
    val = _engine_validation(res)
    for code in HAAS:
        assert val[code]["first_stop"] is not None
        assert val[code]["rejoin"] is not None and abs(val[code]["rejoin"]["error"]) <= 3
    tm = res["tick_ms"]
    assert tm["n_ticks"] == sim.n - 1 and tm["median"] < 1000


def test_the_three_modes_share_one_truth():
    if not _have_model():
        pytest.skip("no sealed model for spain-2026")
    from src.live.engine import WeekendModel

    wm = WeekendModel.load(EVENT)
    grid = load_grid(EVENT)
    a = RaceSim(SimConfig(event=EVENT, seed=3, haas_mode="plan"), wm=wm, grid=grid)
    b = RaceSim(SimConfig(event=EVENT, seed=3, haas_mode="mirror"), wm=wm, grid=grid)
    assert a.truth.as_dict() == b.truth.as_dict()
    assert np.array_equal(a.noise, b.noise)
    ra, rb = a.run(), b.run()
    assert not ra["engine_in_loop"] and not rb["engine_in_loop"]
    # the field's plans are the same in both; only the Haas cars' stops may differ
    for ca, cb in zip(sorted(ra["cars"], key=lambda c: c["code"]), sorted(rb["cars"], key=lambda c: c["code"])):
        if ca["code"] not in HAAS:
            assert [s["lap"] for s in ca["stops"]] == [s["lap"] for s in cb["stops"]], ca["code"]


def test_the_engine_reads_evolution_the_right_way_round():
    """Synthetic race laps with a known fuel effect and a known evolution: the
    engine's estimate must have the evolution's slope, not twice the fuel's."""
    if not _have_model():
        pytest.skip("no sealed model for spain-2026")
    from src.live.engine import RaceEngine, WeekendModel

    wm = WeekendModel.load(EVENT)
    eng = RaceEngine(wm)
    n = wm.event.n_race_laps
    s = eng.fp.s_per_lap
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        stop = 24 + (i % 7)                       # offset stints identify age from lap
        for lap in range(2, n + 1):
            age = lap if lap <= stop else lap - stop
            comp = "MEDIUM" if lap <= stop else "HARD"
            t = 95.0 + s * (n - lap) - 0.03 * (lap - 1) + 0.01 * age + rng.normal(0, 0.3)
            rows.append({"driver_number": str(i), "lap_number": float(lap), "lap_time_s": t, "tyre_life": float(age),
                         "compound": comp, "is_accurate": True, "track_status": "1", "is_complete": True,
                         "pit_in": lap == stop, "pit_out": lap == stop + 1})
    laps = pd.DataFrame(rows)
    laps.loc[laps["pit_in"] | laps["pit_out"], "is_accurate"] = False
    evo = eng._race_evolution(laps, n)
    slope = np.polyfit(np.arange(2, n + 1), evo[1:], 1)[0]
    assert -0.06 < slope < -0.01, slope          # the truth is -0.03; twice the fuel would be -0.11
    # and the fuel correction takes the on-board fuel off, so late laps are not made slower
    y = eng._fuel_corrected(np.array([95.0, 95.0]), np.array([1.0, float(n)]), n)
    assert y[0] < y[1]


def test_the_live_options_carry_the_sealed_plan_family_handicap(sim_result):
    sim, res = sim_result
    if not sim.wm.plan_prior or sim.wm.plan_prior_tau_s <= 0:
        pytest.skip("this weekend's model carries no plan-family prior")
    # on the first stint the option that fits the modal family is charged nothing
    # and a rarer family something: the engine's records name the modal one
    recs = res["engine"]["OCO"]
    firsts = [r["best"] for r in recs[:10] if r.get("best")]
    assert firsts and all("HARD" in b or "MEDIUM" in b for b in firsts)
