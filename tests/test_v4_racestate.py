"""V4 Task 1: the race state times the pit stop.

The failure these guard against is the one V3 shipped: a first stop chosen on
one car's flat cost surface by the circuit's history, 3-6 laps after the field.
So the tests check the mechanism, not a number fitted to a benchmark:

* the constants are measured on the other races (leave-one-out) and a place is
  worth `V (2 psi - 1)`;
* a car in a pack stops no later than its tyre alone would, and the term it
  pays is zero at the lap it chooses;
* rivals cover when the place is worth more than the stop they move, and never
  when a place is worth nothing;
* the live call compares PIT NOW, STAY OUT 1/2/3 and the edge of the window
  against at most four rivals, and switching the race state off is V3.

Run: .venv/bin/python -m pytest tests/test_v4_racestate.py -v
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import racestate  # noqa: E402
from src.config import DATA_PROCESSED  # noqa: E402

RACE = ROOT / "data" / "raw" / "livetiming" / "2026_hungary_race"


def _synthetic_pack(rate: float = 0.08, n: int = 60, s_star: int = 24):
    """A plan family with a convex cost in its first-stop lap and linear wear."""
    laps = np.arange(8, 40)
    T = 0.02 * (laps - s_star) ** 2
    k = np.arange(n + 1, dtype=float)
    stay_cum = np.cumsum(np.concatenate([[0.0], rate * np.arange(1, n + 1)]))   # loss grows with age
    fresh_cum = np.zeros((n + 1, n + 1))
    for s in range(n + 1):
        j = np.arange(n + 1, dtype=float)
        fresh_cum[s] = 0.7 * (j > 0) + 0.05 * j * (j + 1) / 2 + 0.3 * j      # warm-up, wear, harder set
    return laps, T, stay_cum, fresh_cum


def _const(**kw):
    base = racestate.RaceStateConstants(place_gap_s=4.5, persistence=0.75, cycle_sd_s=1.8,
                                        pack_gaps_s=tuple(np.linspace(0.3, 3.5, 200)), source="test")
    return replace(base, **kw)


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------


def test_constants_are_leave_one_out_and_in_physical_ranges():
    if not (DATA_PROCESSED / "laps_hungary-2026_race.parquet").exists():
        pytest.skip("race lap tables not on disk")
    c = racestate.measure_constants(exclude="hungary-2026")
    assert "hungary-2026" not in c.donors and len(c.donors) >= 3
    assert 1.0 < c.place_gap_s < 15.0                 # seconds between adjacent finishers
    assert 0.5 < c.persistence <= 1.0                 # most pit-cycle orders survive to the flag
    assert 0.5 < c.cycle_sd_s < 4.0
    assert 0.3 < float(np.median(c.pack_gaps_s)) < 3.0
    assert c.place_value_s == pytest.approx(c.place_gap_s * (2 * c.persistence - 1))
    assert c.sigma_rel_s == pytest.approx(np.sqrt(2) * c.cycle_sd_s)
    assert c.n_cycle_pairs >= 10 and c.n_finish_gaps >= 20


# --------------------------------------------------------------------------
# the pack: before the race
# --------------------------------------------------------------------------


def test_pack_stops_no_later_than_the_tyre_and_pays_nothing_at_its_lap():
    laps, T, stay, fresh = _synthetic_pack()
    pk = racestate.pack_equilibrium(T, laps, stay, fresh, _const())
    assert pk["converged"]
    assert pk["best_lap"] <= pk["tyre_best_lap"]
    i = pk["laps"].index(pk["best_lap"])
    assert pk["term_s"][i] == pytest.approx(0.0, abs=1e-9)
    # expected rivals ahead after the round: 0..4, fewer the earlier you stop
    places = np.asarray(pk["places"])
    assert places.min() >= 0 and places.max() <= 4
    assert places[0] < places[-1]


def test_a_worthless_place_leaves_the_tyre_to_decide():
    laps, T, stay, fresh = _synthetic_pack()
    pk = racestate.pack_equilibrium(T, laps, stay, fresh, _const(persistence=0.5))   # 2 psi - 1 = 0
    assert pk["best_lap"] == pk["tyre_best_lap"]
    assert np.allclose(pk["term_s"], 0.0)


def test_a_stronger_undercut_pulls_the_stop_earlier():
    laps, T, stay_lo, fresh = _synthetic_pack(rate=0.04)
    _, _, stay_hi, _ = _synthetic_pack(rate=0.12)
    lo = racestate.pack_equilibrium(T, laps, stay_lo, fresh, _const())
    hi = racestate.pack_equilibrium(T, laps, stay_hi, fresh, _const())
    assert hi["best_lap"] <= lo["best_lap"]


# --------------------------------------------------------------------------
# rival behaviour
# --------------------------------------------------------------------------


def test_rival_covers_only_when_the_place_is_worth_the_stop():
    # one gap scenario; our options: stop on lap 10 or 11; rival: 10, 11, 12, 13
    ours = np.array([10, 11])
    theirs = np.array([10, 11, 12, 13])
    # P(rival ahead): falls the later it stops after us
    P = np.array([[[0.5, 0.3, 0.15, 0.05], [0.7, 0.5, 0.3, 0.15]]])
    q = np.array([0.0, 0.0, 0.0, 1.0])                         # it planned lap 13
    T_r = np.array([0.4, 0.2, 0.1, 0.0])                        # lap 13 is its cheapest
    cover = np.array([1, 2])                                    # the lap after ours
    later = theirs[None, :] > ours[:, None] + 1
    no = racestate.expected_ahead(P, q, T_r, 0.0, cover, later)
    yes = racestate.expected_ahead(P, q, T_r, 5.0, cover, later)
    assert no[0, 0] == pytest.approx(0.05)                      # a worthless place: no cover
    assert yes[0, 0] > no[0, 0]                                 # a valuable one: it covers on lap 11
    # impossible options carry no NaN into the expectation
    T_inf = np.array([np.inf, 0.2, np.inf, 0.0])
    out = racestate.expected_ahead(P, np.array([0.0, 0.5, 0.0, 0.5]), T_inf, 5.0, cover, later)
    assert np.isfinite(out).all()


def test_relevant_rivals_use_the_virtual_gap_and_cap_at_four():
    def view(num, gl, stops):
        return racestate.CarView(number=num, code=num, cur_lap=20, gap_leader_s=gl, position=None, stops=stops,
                                 in_pit=False, compound="MEDIUM", tyre_age=10.0, cont=np.zeros(40),
                                 fresh_rows=None, next_compound="HARD", curve_laps=np.zeros(0, int),
                                 curve_cost=np.zeros(0))
    me = view("me", 30.0, 0)
    cars = {"me": me, "a": view("a", 29.0, 0), "b": view("b", 31.5, 0),
            "pitted": view("pitted", 50.0, 1),                  # 20 s behind on track, one stop more: 2 s ahead virtually
            "far": view("far", 45.0, 0), "c": view("c", 27.0, 0), "d": view("d", 33.0, 0), "e": view("e", 34.0, 0)}
    riv = racestate.relevant_rivals(me, cars, pit_loss_s=22.0)
    names = [r[0] for r in riv]
    assert len(riv) == 4 and "far" not in names and "pitted" in names
    v = dict((r[0], r[2]) for r in riv)
    assert v["pitted"] == pytest.approx(30.0 - 50.0 + 22.0)


def test_rejoin_traffic_sees_the_train_it_would_rejoin_behind():
    busy = racestate.rejoin_traffic(10.0, [31.0, 60.0], 22.0, dirty_air_s_per_lap=0.3, laps_per_stop=1.2, sigma_s=1.0)
    clear = racestate.rejoin_traffic(10.0, [45.0, 60.0], 22.0, dirty_air_s_per_lap=0.3, laps_per_stop=1.2, sigma_s=1.0)
    assert busy["p_traffic"] > 0.5 > clear["p_traffic"]
    assert busy["traffic_s"] > clear["traffic_s"] >= 0
    assert busy["rejoin_position"] >= 1


def test_action_table_names_five_actions_and_picks_the_cheapest():
    laps = np.arange(15, 30)
    cost = (laps - 21.0) ** 2 / 10
    t = racestate.action_table(laps, cost, now_lap=17, window_hi=24)
    names = [a["action"] for a in t["actions"]]
    assert names == list(racestate.ACTIONS)
    assert [a["lap"] for a in t["actions"]] == [17, 18, 19, 20, 24]
    assert t["decision"] == "STAY OUT 3 LAPS" and t["decision_lap"] == 20
    t = racestate.action_table(laps, cost, now_lap=21, window_hi=23)
    assert t["decision"] == "PIT NOW"


# --------------------------------------------------------------------------
# the search and the live engine
# --------------------------------------------------------------------------


def _small_model(key: str = "hungary-2026", n: int = 80):
    from src.calibration import get_calibration
    from src.config import get_event
    from src.model_bayes import BayesFit
    from src.tyre import TyreModel

    p = DATA_PROCESSED / f"posterior_{key}.npz"
    if not p.exists():
        pytest.skip("posterior not on disk")
    ev = get_event(key)
    cal = get_calibration(ev)
    f = BayesFit.load(p)
    idx = np.random.default_rng(0).choice(f.posterior["lin"].shape[0], size=n, replace=False)
    return ev, cal, TyreModel.from_fit(f, draws=idx, budget=cal.budgets, manage_floor=cal.manage_wear_floor,
                                       manage_cost_s=cal.manage_cost_s)


def test_search_charges_the_race_state_on_the_first_stop_only():
    from src import strategy as strat

    ev, cal, model = _small_model()
    if not (DATA_PROCESSED / "laps_hungary-2026_race.parquet").exists():
        pytest.skip("race lap tables not on disk")
    kw = dict(step=2, shortlist=600, undercut_lambda=cal.undercut_lambda, max_per_compound={"SOFT": 2, "MEDIUM": 2, "HARD": 2})
    v3 = strat.simulate_model(model, ev, 22.0, **kw)
    v4 = strat.simulate_model(model, ev, 22.0, race_state=racestate.measure_constants(exclude=ev.key), **kw)
    assert v3.race_state == {} and "race_state_s" not in v3.best
    best = v4.race_state["best"]
    assert best and v4.best["pit_laps"][0] == best["best_lap"]
    assert best["best_lap"] <= best["tyre_best_lap"]
    assert abs(v4.best["race_state_s"]) < 1e-6              # zero at the lap it chose
    assert v4.race_state["constants"]["donors"] and ev.key not in v4.race_state["constants"]["donors"]


def test_live_engine_compares_five_actions_and_is_v3_when_switched_off():
    if not RACE.exists():
        pytest.skip("archived streams missing")
    from src.live.engine import RaceEngine, WeekendModel
    from src.live.sources import RecordedSource
    from src.live.state import LiveState

    wm = WeekendModel.load("hungary-2026")
    on, off = RaceEngine(wm), RaceEngine(wm, race_state=False)
    st = LiveState()
    snaps = None
    for m in RecordedSource(RACE, include_zipped=False):
        st.apply(m)
        evs = st.drain_events()
        if any(k == "lap" for _, k, _ in evs) and st.lap_count.get("current") in (8, 12, 16):
            snaps = (on.tick(st), off.tick(st))
        if (st.lap_count.get("current") or 0) > 16:
            break
    assert snaps is not None
    s_on, s_off = snaps
    assert s_on["meta"]["race_state"]["place_value_s"] > 0 and s_off["meta"]["race_state"] is None
    n_rs = 0
    for row in s_on["field"]:
        plan = row.get("plan") or {}
        rs = plan.get("race_state")
        if not rs:
            continue
        n_rs += 1
        assert [a["action"] for a in rs["actions"]] == list(racestate.ACTIONS)
        assert rs["decision"] in racestate.ACTIONS
        assert len(rs["rivals"]) <= racestate.N_RIVALS
        assert plan["first_stop_prior_applies"] is False
    assert n_rs >= 10
    for row in s_off["field"]:
        assert (row.get("plan") or {}).get("race_state") is None
    from src.live.store import _clean
    import json
    json.dumps(_clean(s_on))
