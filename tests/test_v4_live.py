"""V4 WP-D: the live race-execution engine - rivals, rejoin, the call, the reasons.

The failures these guard against are the ones Task 1 left in the live path:

* the rival set came from the virtual gap alone, so the car *directly behind on
  the road* - the one that takes the place if the stop goes wrong - was missing
  whenever it sat outside the 6 s band, and a car a lap down was in the set
  whenever the arithmetic happened to put it there;
* the rejoin was priced on today's gaps, so a car that was about to pit with us
  counted as if it were staying out;
* the recommendation had no memory, so it could flip every lap with nothing
  having happened;
* nothing said *why*.

So the tests check the mechanism on state built by hand (no feed, no fit) plus
one real replay tick for the wiring, and never a number fitted to a benchmark.

Run: .venv/bin/python -m pytest tests/test_v4_live.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import explain, racestate  # noqa: E402
from src.live import rivals  # noqa: E402
from src.live.engine import DriverTyre, RaceEngine  # noqa: E402
from src.live.store import _clean  # noqa: E402

RACE = ROOT / "data" / "raw" / "livetiming" / "2026_hungary_race"
PIT = 22.0
SIGMA = 2.7


# --------------------------------------------------------------------------
# helpers: a field of `CarView`s, no feed and no fit
# --------------------------------------------------------------------------


def view(num, gap_leader, *, position=None, stops=0, cur_lap=20, pace=90.0, next_stop=None,
         pending=True, in_pit=False, compound="MEDIUM", age=10.0, code=None):
    return racestate.CarView(
        number=num, code=code or num, cur_lap=cur_lap, gap_leader_s=gap_leader, position=position,
        stops=stops, in_pit=in_pit, compound=compound, tyre_age=age, cont=np.zeros(30),
        fresh_rows=np.zeros((60, 60)), next_compound="HARD", curve_laps=np.arange(21, 30),
        curve_cost=np.linspace(10.0, 12.0, 9), stay_cost=float("inf"),
        extra={"pace_s": pace, "pace_source": "test", "next_stop": next_stop,
               "pending_own": pending})


def rows_for(views: dict) -> list:
    return [{"driver_number": n, "driver": v.code, "position": v.position,
             "gap_leader": ("" if v.gap_leader_s is None else f"+{v.gap_leader_s:.1f}"),
             "team": "Haas F1 Team"} for n, v in views.items()]


# --------------------------------------------------------------------------
# D1: who are we racing?
# --------------------------------------------------------------------------


def test_selection_keeps_the_car_behind_on_track_and_drops_a_lapped_car():
    me = view("me", 30.0, position=10)
    cars = {
        "me": me,
        # 8 s behind on the road: outside the 6 s virtual band, but it is the car
        # that inherits the place if the stop costs us four seconds
        "behind": view("behind", 38.0, position=11, code="BEH"),
        "ahead": view("ahead", 29.2, position=9, code="AHD"),
        # a lap down by the completed-lap count, and again by the feed's words
        "lapped": view("lapped", 31.0, position=12, cur_lap=18, code="LAP"),
        "far": view("far", 90.0, position=16, code="FAR"),
    }
    order = rows_for(cars)
    picks = rivals.strategic_rivals(me, cars, order, PIT, SIGMA, k=4)
    names = [p.number for p in picks]
    assert "behind" in names and "ahead" in names
    assert "lapped" not in names and "far" not in names
    beh = next(p for p in picks if p.number == "behind")
    assert beh.on_track == "behind"
    assert "8.0 s behind on track" in beh.why
    # the feed's own words are enough on their own
    cars["lapped"] = view("lapped", 31.0, position=12, cur_lap=20, code="LAP")
    order = rows_for(cars)
    order = [r | {"gap_leader": "1 L"} if r["driver_number"] == "lapped" else r for r in order]
    assert "lapped" not in [p.number for p in rivals.strategic_rivals(me, cars, order, PIT, SIGMA, k=4)]
    assert "lapped" in [p.number for p in rivals.strategic_rivals(me, cars, rows_for(cars), PIT, SIGMA, k=4)]


def test_selection_drops_a_car_the_cycle_puts_out_of_reach_and_ranks_by_sigma():
    me = view("me", 30.0, position=10, pending=True)
    cars = {
        "me": me,
        # 2 s ahead once we have both stopped: it has pitted, we have not
        "cycled": view("cycled", 50.0, position=14, stops=1, pending=False, code="CYC"),
        # the same 20 s behind, but it still owes a stop of its own: it cannot
        # come back to us through this cycle
        "gone": view("gone", 50.0, position=15, stops=1, pending=True, code="GON"),
        "near": view("near", 31.0, position=11, code="NEA"),
    }
    picks = rivals.strategic_rivals(me, cars, rows_for(cars), PIT, SIGMA, k=4)
    names = [p.number for p in picks]
    assert "cycled" in names and "gone" not in names
    # ranked by |virtual gap| / sigma, closest first
    assert names[0] == "near"
    assert picks[0].score == pytest.approx(1.0 / SIGMA, rel=1e-6)
    assert "one stop more" in next(p for p in picks if p.number == "cycled").why


def test_selection_cap_is_the_smallest_k():
    me = view("me", 30.0, position=10)
    cars = {"me": me}
    for i in range(1, 7):
        cars[str(i)] = view(str(i), 30.0 + 0.7 * i, position=10 + i, code=f"C{i}")
    for k in (3, 4, 5, 6):
        assert len(rivals.strategic_rivals(me, cars, rows_for(cars), PIT, SIGMA, k=k)) == k


# --------------------------------------------------------------------------
# D2: the rejoin projection
# --------------------------------------------------------------------------


def _rejoin(cars, me, stop_lap=21):
    return rivals.project_rejoin(me, cars, stop_lap, pit_loss_s=PIT, pit_now_s=PIT, now_lap=21,
                                 sigma_rel_s=SIGMA, dirty_air_s_per_lap=0.45, laps_per_stop=1.2)


def test_the_projection_moves_a_car_that_is_stopping():
    me = view("me", 30.0, position=10)
    staying = {"me": me, "a": view("a", 20.0, position=9, code="AAA", next_stop=None, pending=False)}
    stopping = {"me": me, "a": view("a", 20.0, position=9, code="AAA", next_stop=21, pending=True)}
    a = _rejoin(staying, me)
    b = _rejoin(stopping, me)
    # it is 10 s up the road now; after our stop it is 32 s ahead if it stays
    # out and 10 s ahead if it boxes with us
    assert a["gap_ahead_s"] == pytest.approx(32.0, abs=0.01)
    assert b["gap_ahead_s"] == pytest.approx(10.0, abs=0.01)
    assert b["gap_ahead_s"] < a["gap_ahead_s"] - PIT / 2
    assert a["ahead"] == b["ahead"] == "AAA"
    assert a["source"].startswith("projected")


def test_the_projection_carries_pace_and_counts_the_traffic_band():
    me = view("me", 30.0, position=10, pace=90.0)
    # a car 2 s up the road but a second a lap slower: in four laps it is ours
    slow = {"me": me, "a": view("a", 28.0, position=9, code="SLO", pace=91.0, pending=False)}
    same = {"me": me, "a": view("a", 28.0, position=9, code="SAM", pace=90.0, pending=False)}
    far = rivals.project_rejoin(me, slow, 24, pit_loss_s=0.0, now_lap=21, sigma_rel_s=SIGMA,
                                dirty_air_s_per_lap=0.45, laps_per_stop=1.2)
    flat = rivals.project_rejoin(me, same, 24, pit_loss_s=0.0, now_lap=21, sigma_rel_s=SIGMA,
                                 dirty_air_s_per_lap=0.45, laps_per_stop=1.2)
    assert flat["gap_ahead_s"] == pytest.approx(2.0, abs=0.01)
    assert far["gap_behind_s"] == pytest.approx(3.0, abs=0.01)      # 5 laps at 1 s = 3 s behind us
    # a car inside the 3 s band is traffic and is named
    band = rivals.project_rejoin(me, {"me": me, "a": view("a", 28.0, position=9, code="BND",
                                                          pending=False)},
                                 21, pit_loss_s=0.0, now_lap=21, sigma_rel_s=SIGMA,
                                 dirty_air_s_per_lap=0.45, laps_per_stop=1.2)
    assert [c["driver"] for c in band["band"]] == ["BND"]
    # one car 2 s up the road, sigma 2.7: Phi((3-2)/sigma) - Phi(-2/sigma)
    assert band["p_traffic"] == pytest.approx(0.415, abs=0.01)
    assert band["traffic_s"] > 0
    clear = rivals.project_rejoin(                     # the same car eight seconds up the road
        me, {"me": me, "a": view("a", 22.0, position=9, code="BND", pending=False)}, 21,
        pit_loss_s=0.0, now_lap=21, sigma_rel_s=SIGMA, dirty_air_s_per_lap=0.45, laps_per_stop=1.2)
    assert clear["band"] == [] and clear["traffic_s"] < band["traffic_s"]


def test_the_projection_falls_back_to_the_gaps_without_a_pace():
    me = view("me", 30.0, position=10, pace=None)
    cars = {"me": me, "a": view("a", 20.0, position=9, pace=None)}
    assert rivals.field_projection(me, cars) is None
    out = rivals.rejoin_or_fallback(me, cars, 21, pit_loss_s=PIT, now_lap=21, sigma_rel_s=SIGMA,
                                    dirty_air_s_per_lap=0.45, laps_per_stop=1.2)
    assert out["source"].startswith("today's gaps")
    assert out["rejoin_position"] is not None


# --------------------------------------------------------------------------
# D3: the pit response
# --------------------------------------------------------------------------


def test_cover_probability_matches_expected_ahead():
    """The rho reported per rival is the rho the position term prices with."""
    P = np.array([[[0.5, 0.3, 0.15, 0.05], [0.7, 0.5, 0.3, 0.15]]])
    q = np.array([0.0, 0.1, 0.2, 0.7])
    T_r = np.array([0.4, 0.2, 0.1, 0.0])
    cover = np.array([1, 2])
    later = np.array([10, 11])[:, None] + 1 < np.array([10, 11, 12, 13])[None, :]
    V = 5.0
    rho: dict = {}
    ref = racestate.expected_ahead(P, q, T_r, V, cover, later, rho_out=rho)
    mine = rivals.cover_probability(P, q, T_r, V, cover, later)
    assert np.allclose(rho["rho"], mine)
    Pc = np.take_along_axis(P, cover[None, :, None], axis=2)
    assert np.allclose(ref, np.einsum("gij,j->gi", (1 - mine) * P + mine * Pc, q))


def test_cover_response_reports_the_places_it_costs():
    detail = {"driver": "RUS", "driver_number": "63",
              "cover": {"p_cover": 0.4, "p_ahead_if_cover": 0.8, "p_ahead_if_plan": 0.3, "cover_lap": 21}}
    out = rivals.cover_response(detail, 2.5)
    assert out["p_cover"] == 0.4 and out["p_ahead_after"] == 0.8
    assert out["places_delta"] == pytest.approx(0.5)
    assert out["cost_s"] == pytest.approx(1.25)              # 0.5 places x 2.5 s
    assert out["expected_cost_s"] == pytest.approx(0.5)
    s = rivals.cover_summary([detail, {"driver": "X", "cover": None}], 2.5)
    assert s["driver"] == "RUS" and s["n_rivals"] == 1
    assert rivals.cover_summary([], 2.5) is None


# --------------------------------------------------------------------------
# D4: the call, and its memory
# --------------------------------------------------------------------------


class _Green:
    track_status = "1"


def _bare_engine():
    """A `RaceEngine` with only the state `_decide` reads: no fit, no feed."""
    eng = RaceEngine.__new__(RaceEngine)
    eng._decision = {}
    return eng


def _tyre():
    dt = DriverTyre(number="me", code="OCO", compound="MEDIUM")
    dt.ltc = np.linspace(2.0, 30.0, 200)
    dt.weights = np.full(200, 1.0 / 200)
    dt.laps_to_cliff = (5.0, 12.0, 25.0)
    dt.deg_now_s_per_lap = 0.12
    return dt


def _call(eng, costs: dict, *, now_lap: int, rivals_detail: list, window_hi: int = 30,
          cur_lap: int | None = None, seed: int = 0):
    """Run `_decide` on a cost curve given by lap, with draws around it."""
    cur_lap = now_lap - 1 if cur_lap is None else cur_lap
    laps = np.array(sorted(costs), dtype=int)
    cost = np.array([costs[int(l)] for l in laps], dtype=float)
    tbl = racestate.action_table(
        laps, cost, {"tyre_s": cost - cost.min(), "position_s": np.zeros(len(laps)),
                     "traffic_s": np.zeros(len(laps))},
        now_lap=now_lap, window_hi=window_hi,
        extra={"rejoin_position": [10 + i for i in range(len(laps))]})
    rng = np.random.default_rng(seed)
    T = cost[:, None] + rng.normal(0, 0.4, size=(len(laps), 200))
    rs = {"place_value_s": 2.5, "rivals": rivals_detail, "position_stay_s": 0.0,
          "me": {"position": 12, "compound": "MEDIUM", "tyre_age": 18.0, "stops": 0,
                 "gap_leader_s": 30.0},
          "rejoin_detail": {int(now_lap) + i: {"rejoin_position": 12 + i, "gap_ahead_s": 1.8,
                                               "ahead": "RUS", "band": [], "p_traffic": 0.2,
                                               "traffic_s": 0.3, "source": "projected"}
                            for i in range(4)},
          "if_cover": None, "rejoin_gaps_now": {"rejoin_position": 13}}
    return eng._decide(_Green(), "me", _tyre(), rs, tbl, now_lap=now_lap, cur_lap=cur_lap, R=30,
                       stops_taken=0, window_hi=window_hi, stay_legal=False, T=T,
                       pos_of={i: i for i in range(len(laps))}, price=lambda i: T[i],
                       li=np.arange(len(laps)), st_l=laps, mean_all=cost, stay_i=None)


RIV = [{"driver": "RUS", "driver_number": "63", "gap_s": -1.9, "virtual_gap_s": -1.9,
        "compound": "MEDIUM", "tyre_age": 17.0, "stops": 0, "in_pit": False, "pending_stop": True,
        "p_stop_next_3": 0.4, "stop_lap_median": 23, "p_ahead_now": 0.3, "p_ahead_stay_3": 0.6,
        "why": "1.9 s behind on track", "on_track": "behind",
        "if_cover": {"driver": "RUS", "p_cover": 0.35, "p_ahead_after": 0.7, "p_ahead_if_not": 0.3,
                     "places_delta": 0.4, "cost_s": 1.0, "expected_cost_s": 0.35}}]


def test_the_call_names_one_of_four_actions_with_its_confidence():
    eng = _bare_engine()
    # boxing now is the cheapest lap
    d = _call(eng, {20: 0.0, 21: 1.0, 22: 2.0, 23: 3.0, 30: 8.0}, now_lap=20, rivals_detail=RIV)
    assert d["action"] == "PIT NOW" and d["action_kind"] == "PIT_NOW" and d["wait_laps"] == 0
    assert 0.0 < d["confidence"] <= 1.0 and d["n_actions"] == 5
    assert d["delta_vs_alternative_s"] == pytest.approx(1.0)
    assert d["projected_position"] == 12
    assert d["life"]["p_cliff_before_stop"] is not None and d["life"]["life_p50"] == 12.0
    # two laps out, the call is WAIT 2 LAPS; the window edge is BOX BY LAP x
    d2 = _call(eng, {20: 2.0, 21: 1.0, 22: 0.0, 23: 3.0, 30: 8.0}, now_lap=20, rivals_detail=RIV)
    assert d2["action"] == "WAIT 2 LAPS" and d2["wait_laps"] == 2 and d2["lap"] == 22
    eng2 = _bare_engine()
    d3 = _call(eng2, {20: 5.0, 21: 4.0, 22: 3.0, 23: 2.0, 30: 0.0}, now_lap=20, rivals_detail=RIV)
    assert d3["action"] == "BOX BY LAP 30" and d3["box_by_lap"] == 30 and d3["wait_laps"] is None


def test_the_call_is_held_inside_the_margin_and_released_by_a_rival_stopping():
    from src.live import engine as E

    eng = _bare_engine()
    first = _call(eng, {20: 3.0, 21: 2.0, 22: 1.0, 23: 0.0, 30: 6.0}, now_lap=20, rivals_detail=RIV)
    assert first["action"] == "WAIT 3 LAPS" and first["lap"] == 23 and first["changed"] is True
    # next lap: lap 22 is now better than the lap we are holding, but only just
    held = _call(eng, {21: 3.0, 22: 0.0, 23: 0.2, 24: 1.0, 30: 6.0}, now_lap=21, rivals_detail=RIV)
    assert held["lap"] == 23 and held["action"] == "WAIT 2 LAPS"
    assert held["held_by_hysteresis"] is True and held["changed"] is False
    assert held["held_since_lap"] == 20
    assert held["delta_vs_alternative_s"] == pytest.approx(-0.2)
    assert 0.0 < E.DECISION_HYSTERESIS_S < E.BOX_NOW_TOL_S
    # ... and a margin wider than the hold releases it
    eng_b = _bare_engine()
    _call(eng_b, {20: 3.0, 21: 2.0, 22: 1.0, 23: 0.0, 30: 6.0}, now_lap=20, rivals_detail=RIV)
    moved = _call(eng_b, {21: 3.0, 22: 0.0, 23: 0.9, 24: 1.0, 30: 6.0}, now_lap=21, rivals_detail=RIV)
    assert moved["lap"] == 22 and moved["held_by_hysteresis"] is False and moved["changed"] is True
    # ... and so does the rival stopping, even inside the margin
    eng_c = _bare_engine()
    _call(eng_c, {20: 3.0, 21: 2.0, 22: 1.0, 23: 0.0, 30: 6.0}, now_lap=20, rivals_detail=RIV)
    stopped = _call(eng_c, {21: 3.0, 22: 0.0, 23: 0.2, 24: 1.0, 30: 6.0}, now_lap=21,
                    rivals_detail=[{**RIV[0], "stops": 1}])
    assert stopped["lap"] == 22 and stopped["held_by_hysteresis"] is False
    assert any("has stopped" in s for s in stopped["state_change"])


def test_the_call_serialises_and_reads_as_a_sentence():
    eng = _bare_engine()
    d = _call(eng, {20: 0.0, 21: 1.0, 22: 2.0, 23: 3.0, 30: 8.0}, now_lap=20, rivals_detail=RIV)
    json.dumps(_clean(d))                      # the snapshot writer must take it
    assert d["headline"].startswith("PIT NOW") or "PIT NOW" in d["headline"]
    assert d["principal"] and len(d["reasons"]) >= 2
    assert any("RUS" in r for r in d["reasons"])


# --------------------------------------------------------------------------
# D5: the explanation, on a synthetic plan
# --------------------------------------------------------------------------


SYNTH = {
    "best": "1 stop: HARD on lap 22",
    "race_state": {
        "actions": [
            {"action": "PIT NOW", "lap": 20, "legal": True, "cost_s": 101.4, "tyre_s": 0.0,
             "position_s": 1.6, "traffic_s": 0.6, "delta_s": 0.4, "rejoin_position": 15},
            {"action": "STAY OUT 1 LAP", "lap": 21, "legal": True, "cost_s": 101.2, "tyre_s": 0.7,
             "position_s": 1.0, "traffic_s": 0.5, "delta_s": 0.2, "rejoin_position": 15},
            {"action": "STAY OUT 2 LAPS", "lap": 22, "legal": True, "cost_s": 101.0, "tyre_s": 1.4,
             "position_s": 0.0, "traffic_s": 0.4, "delta_s": 0.0, "rejoin_position": 14},
            {"action": "STAY OUT 3 LAPS", "lap": 23, "legal": True, "cost_s": 101.9, "tyre_s": 2.3,
             "position_s": 0.2, "traffic_s": 0.4, "delta_s": 0.9, "rejoin_position": 14},
            {"action": "PIT AT EDGE OF WINDOW", "lap": 26, "legal": False},
        ],
        "decision": "STAY OUT 2 LAPS", "decision_lap": 22,
        "rivals": RIV, "place_value_s": 2.5, "position_s_stay": 3.1,
    },
    "decision": {
        "action": "WAIT 2 LAPS", "action_kind": "WAIT", "lap": 22, "wait_laps": 2, "box_by_lap": None,
        "confidence": 0.78, "projected_position": 15, "projected_position_if_now": 16,
        "delta_vs_alternative_s": 0.2, "held_by_hysteresis": False, "hysteresis_s": 0.3,
        "rejoin": {"rejoin_position": 15, "gap_ahead_s": 2.1, "ahead": "HAM",
                   "band": [{"driver": "HAM", "gap_s": 2.1}], "p_traffic": 0.5},
        "life": {"p_cliff_before_stop": 0.18, "life_p10": 4.0, "life_p50": 9.0, "life_p90": 17.0,
                 "tyre_age": 21.0, "compound": "MEDIUM"},
        "if_cover": RIV[0]["if_cover"], "rivals": RIV,
    },
}


def test_explain_decision_speaks_in_the_numbers_of_the_state():
    out = explain.explain_decision(SYNTH, {"driver": "OCO", "compound": "MEDIUM", "position": 15},
                                   {"laps_remaining": 30, "sc_active": False})
    text = " | ".join([out["headline"], out["principal"]] + out["reasons"])
    assert "OCO: WAIT 2 LAPS" in out["headline"]
    assert "box on lap 22" in out["headline"] and "78 %" in out["headline"]
    assert "RUS" in text                                   # the rival
    assert "1.4 s of tyre" in text                         # the tyre cost of waiting
    assert "P15" in text                                   # the rejoin position
    assert "0.18" in text and "not decisive" in text       # the life risk, shown not used
    assert "covers a stop now with probability 0.35" in text
    assert 2 <= len(out["reasons"]) <= explain.MAX_REASONS
    # nothing is invented when the numbers are missing
    bare = explain.explain_decision({"decision": {"action": "PIT NOW"}}, None, None)
    assert bare["headline"] == "PIT NOW" and bare["reasons"] == []


def test_explain_actions_reads_the_pre_race_table():
    row = {"lap": 18, "decision": "PIT NOW", "decision_lap": 18,
           "actions": [
               {"action": "PIT NOW", "lap": 18, "legal": True, "cost_s": 160.0, "tyre_s": 1.95,
                "position_s": 0.0, "places_ahead": 1.93, "delta_s": 0.0},
               {"action": "STAY OUT 1 LAP", "lap": 19, "legal": True, "cost_s": 160.01,
                "tyre_s": 1.33, "position_s": 0.63, "places_ahead": 2.18, "delta_s": 0.01},
               {"action": "STAY OUT 2 LAPS", "lap": 20, "legal": False}]}
    out = explain.explain_actions(row)
    assert out["headline"] == "Lap 18: PIT NOW (stop on lap 18)"
    assert len(out["lines"]) == 2 and "1.9 s of tyre" in out["lines"][0]
    assert "the cheapest action" in out["lines"][0] and "+0.01 s against the best" in out["lines"][1]
    assert "1.93 rivals ahead" in out["lines"][0]
    assert out["principal"] and any("track position" in r or "tyre" in r for r in out["reasons"])
    assert explain.explain_actions(None)["lines"] == []


# --------------------------------------------------------------------------
# the wiring, on one real tick
# --------------------------------------------------------------------------


def _replay_to(lap: int, **kw):
    from src.live.engine import WeekendModel
    from src.live.sources import RecordedSource
    from src.live.state import LiveState

    wm = WeekendModel.load("hungary-2026")
    eng = RaceEngine(wm, **kw)
    st = LiveState()
    snap, last = None, None
    for m in RecordedSource(RACE, include_zipped=False):
        st.apply(m)
        evs = st.drain_events()
        cur = st.lap_count.get("current")
        if any(k == "lap" for _, k, _ in evs) and cur and cur != last:
            last = cur
            snap = eng.tick(st)
            if cur >= lap:
                break
    return snap


@pytest.mark.skipif(not RACE.exists(), reason="archived streams missing")
def test_the_live_decision_agrees_with_the_action_table_and_serialises():
    snap = _replay_to(18)
    json.dumps(_clean(snap))
    n = 0
    for row in snap["field"]:
        plan = row.get("plan") or {}
        rs, dec = plan.get("race_state"), plan.get("decision")
        if not rs or not dec:
            continue
        n += 1
        assert dec["action_kind"] in ("PIT_NOW", "STAY_OUT", "WAIT", "BOX_BY")
        assert len(dec["rivals"]) <= racestate.N_RIVALS
        for r in dec["rivals"]:
            assert r["why"]
        # the call is the action table's decision unless it is being held, or
        # unless running to the end without stopping again is the cheapest
        if not dec["held_by_hysteresis"] and dec["action_kind"] != "STAY_OUT":
            assert dec["lap"] == rs["decision_lap"]
        # every legal row carries both fields; they are None only where the row
        # shares its lap with another (the edge of the window can be STAY OUT
        # 3's lap) or the car has no cliff posterior yet
        shares = [a["p_best"] for a in rs["actions"] if a.get("legal")]
        assert len(shares) >= 2 and any(p is not None for p in shares)
        assert all(p is None or 0.0 <= p <= 1.0 for p in shares)
        assert sum(p for p in shares if p is not None) <= 1.0 + 1e-9
        for a in rs["actions"]:
            if a.get("legal"):
                assert a["p_cliff_before_stop"] is None or 0.0 <= a["p_cliff_before_stop"] <= 1.0
        assert row.get("team") is not None and row.get("driver_number")
    assert n >= 10


@pytest.mark.skipif(not RACE.exists(), reason="archived streams missing")
def test_the_v3_path_has_no_decision():
    snap = _replay_to(14, race_state=False)
    assert snap["meta"]["race_state"] is None
    for row in snap["field"]:
        plan = row.get("plan")
        if plan is None:
            continue
        assert plan["race_state"] is None
        assert plan["decision"] is None
        assert plan["best"] and np.isfinite(plan["delta_box_now_s"])
