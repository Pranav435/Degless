"""WP-C: the position-aware strategic metric and the live decision-stability block.

Both are benchmark code, so both are tested the way a benchmark has to be
tested - on synthetic inputs whose right answer is known by hand, never on the
numbers the seven weekends happen to produce.  The metric's definition is in
`docs/v4_methodology.md`; these are the three arithmetic facts it rests on:

  * a plan that stops when the field stops, on the field's own compounds, loses
    no places - `D = 0` exactly, so `L = 0` exactly;
  * a plan that stops eight laps after everyone else loses places, and loses
    more of them the longer it stays out;
  * `R_pos >= 0` for every candidate and `R_pos(best) = 0`.

and the two the stability block rests on: a change of action with a material
state change is explained, one without is not, and a build whose plans carry no
`decision` at all scores nothing rather than scoring perfectly.

Run: .venv/bin/python -m pytest tests/test_v4_bench.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from src import racestate  # noqa: E402

bench_strategy = pytest.importorskip("bench_strategy")
bench_live = pytest.importorskip("bench_live")

# A synthetic weekend: a place worth 5 s at the flag, kept 3 times in 4, a pit
# cycle whose race-time difference is noisy at 2 s per car, and a pack whose
# first-stint intervals run 0.5-4.0 s.
CONST = racestate.RaceStateConstants(
    place_gap_s=5.0, persistence=0.75, cycle_sd_s=2.0,
    pack_gaps_s=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    source="synthetic (tests/test_v4_bench.py)")

N_LAPS = 60
# Two synthetic compounds whose stint cost is quadratic in the stint length and
# flat in where the stint sits, so every `D` below can be worked out by hand:
#   START: 0.20 s/lap + 0.020 s/lap^2      NEXT: 0.15 s/lap + 0.010 s/lap^2
RATES = {"START": (0.20, 0.020), "NEXT": (0.15, 0.010)}


def _means() -> dict:
    L = np.arange(N_LAPS + 1, dtype=float)
    out = {}
    for c, (a, b) in RATES.items():
        row = a * L + b * L ** 2
        out[c] = np.broadcast_to(row[None, :], (N_LAPS + 1, N_LAPS + 1)).copy()
    return out


def _model(field_stops, second_by_start=None) -> bench_strategy.PositionModel:
    return bench_strategy.PositionModel(_means(), CONST, pit_loss_s=22.0,
                                        field_stops=field_stops,
                                        second_by_start=second_by_start or {"START": "NEXT"})


def _cost(c: str, n: int) -> float:
    a, b = RATES[c]
    return a * n + b * n ** 2


# --------------------------------------------------------------------------
# D(s, l)
# --------------------------------------------------------------------------


def test_delta_is_zero_when_both_cars_stop_on_the_same_lap():
    """Same start set, same lap, same next set: no race-time difference at all."""
    pm = _model([20])
    assert pm.delta("START", "NEXT", 20, np.array([20])) == pytest.approx(0.0, abs=1e-12)


def test_delta_matches_the_hand_calculation_for_a_later_stop():
    """`D(28, 20) = (stay[28] + fresh[28, 1]) - (stay[20] + fresh[20, 9])`."""
    pm = _model([20])
    want = (_cost("START", 28) + _cost("NEXT", 1)) - (_cost("START", 20) + _cost("NEXT", 9))
    got = float(pm.delta("START", "NEXT", 28, np.array([20]))[0])
    assert got == pytest.approx(want, rel=1e-12)
    assert got > 0, "staying out eight laps on a worn set must cost race time"


# --------------------------------------------------------------------------
# L(p)
# --------------------------------------------------------------------------


def test_stopping_when_the_field_stops_loses_no_places():
    """The headline invariant: `L = 0` when we do exactly what the field did."""
    pm = _model([18, 19, 20, 21, 22], second_by_start={"START": "NEXT"})
    L = pm.places_lost("START", "NEXT", 20)
    # the field is spread over five laps, so the mean is not identically zero,
    # but it is a fraction of a place and centred on zero
    assert abs(L) < 0.15
    pm_one = _model([20])
    assert pm_one.places_lost("START", "NEXT", 20) == pytest.approx(0.0, abs=1e-12)


def test_stopping_eight_laps_late_loses_places_and_more_of_them_the_later_it_is():
    pm = _model([20])
    on_time = pm.places_lost("START", "NEXT", 20)
    late = pm.places_lost("START", "NEXT", 28)
    later = pm.places_lost("START", "NEXT", 34)
    assert on_time == pytest.approx(0.0, abs=1e-12)
    assert late > 0.25, f"eight laps late should cost a real fraction of a place, got {late}"
    assert later > late
    assert late <= 4.0 and later <= 4.0, "there are only four pack slots to lose"


def test_stopping_earlier_than_the_field_gains_places():
    """The undercut, in the same arithmetic: a negative L is places gained."""
    pm = _model([28])
    assert pm.places_lost("START", "NEXT", 20) < 0


def test_places_lost_is_none_without_a_green_field_stop_or_a_known_compound():
    pm = _model([])
    assert pm.places_lost("START", "NEXT", 20) is None
    pm2 = _model([20])
    assert pm2.places_lost("START", "MISSING", 20) is None
    assert pm2.places_lost("MISSING", "NEXT", 20) is None


def test_p_retain_is_one_minus_a_quarter_of_L_clipped():
    pm = _model([20])
    assert pm.p_retain(0.0) == pytest.approx(1.0)
    assert pm.p_retain(1.0) == pytest.approx(0.75)
    assert pm.p_retain(-2.0) == pytest.approx(1.0)      # places gained: clipped at 1
    assert pm.p_retain(9.0) == pytest.approx(0.0)       # clipped at 0
    assert pm.p_retain(None) is None


def test_j_is_time_plus_the_place_value_times_L():
    pm = _model([20])
    assert pm.V == pytest.approx(5.0 * (2 * 0.75 - 1))
    assert pm.j(100.0, 0.5) == pytest.approx(100.0 + pm.V * 0.5)
    assert pm.j(None, 0.5) is None and pm.j(100.0, None) is None


def test_the_four_pack_slots_carry_a_total_mass_of_four():
    """`L` is a number of places out of four, so the slot weights must sum to 4."""
    pm = _model([20])
    assert float(pm.weight.sum()) == pytest.approx(4.0)
    assert len(np.unique(pm.slot)) == bench_strategy.N_PACK_SLOTS


# --------------------------------------------------------------------------
# R_pos
# --------------------------------------------------------------------------


def test_r_pos_is_non_negative_and_zero_at_the_best_candidate():
    """End to end on the synthetic weekend: T from the tyre, L from the pack."""
    pm = _model([20])
    # a candidate per first-stop lap, its race time the two stints plus one stop
    cand = {}
    for s in range(10, 41):
        T = _cost("START", s) + _cost("NEXT", N_LAPS - s) + pm.pit_loss_s
        cand[f"family@{s}"] = pm.j(T, pm.places_lost("START", "NEXT", s))
    R, best = bench_strategy.pos_regret(cand)
    assert best is not None
    assert R[best] == pytest.approx(0.0)
    assert min(R.values()) == pytest.approx(0.0)
    assert all(v >= -1e-12 for v in R.values()), "a regret cannot be negative"
    # the position term must actually move the answer: the pure-time optimum and
    # the position-aware optimum are not the same lap here
    pure = min(cand, key=lambda k: cand[k] - pm.V * pm.places_lost("START", "NEXT", int(k.split("@")[1])))
    assert pure != best


def test_pos_regret_ignores_missing_and_non_finite_candidates():
    R, best = bench_strategy.pos_regret({"a": 10.0, "b": None, "c": float("nan"), "d": 4.0})
    assert best == "d" and R == {"a": pytest.approx(6.0), "d": pytest.approx(0.0)}
    assert bench_strategy.pos_regret({}) == ({}, None)
    assert bench_strategy.pos_regret({"a": None}) == ({}, None)


def test_field_second_by_start_is_the_modal_second_compound():
    import pandas as pd

    plans = pd.DataFrame({"seq": ["MEDIUM-HARD", "MEDIUM-HARD", "MEDIUM-SOFT", "SOFT-HARD-HARD"]})
    got = bench_strategy.field_second_by_start(plans)
    assert got == {"MEDIUM": "HARD", "SOFT": "HARD"}
    assert bench_strategy.field_second_by_start(pd.DataFrame({"seq": []})) == {}


def test_per_driver_pit_laps_survive_the_meta_round_trip():
    """`meta["per_driver"]["pit_laps"]` is written as a string; C2 reads both."""
    assert bench_strategy._plan_pits({"pit_laps": "[18, 44]"}) == [18, 44]
    assert bench_strategy._plan_pits({"pit_laps": [18, 44]}) == [18, 44]
    assert bench_strategy._plan_pits({}) == []
    assert bench_strategy._plan_pits({"pit_laps": "not a list"}) == []


# --------------------------------------------------------------------------
# the live decision-stability block
# --------------------------------------------------------------------------


def _dec(action, delta=2.0, rivals=("RUS",)):
    return {"decision": {"action": action, "delta_vs_alternative_s": delta,
                         "rivals": [{"driver": r} for r in rivals]}}


def test_a_held_action_is_not_a_change():
    by_lap = {l: {"OCO": _dec("STAY OUT")} for l in range(1, 6)}
    s = bench_live.decision_stability(by_lap, {}, {l: False for l in range(1, 6)})
    assert s["n"] == 5 and s["n_pairs"] == 4
    assert s["n_changes"] == 0 and s["n_unexplained"] == 0
    assert s["share_changed"] == pytest.approx(0.0)
    assert s["share_unexplained"] == pytest.approx(0.0)
    assert s["share_of_changes_unexplained"] is None


def test_a_change_with_nothing_material_is_unexplained():
    by_lap = {1: {"OCO": _dec("STAY OUT")}, 2: {"OCO": _dec("PIT NOW")}}
    s = bench_live.decision_stability(by_lap, {}, {1: False, 2: False})
    assert s["n_pairs"] == 1 and s["n_changes"] == 1 and s["n_unexplained"] == 1
    assert s["share_unexplained"] == pytest.approx(1.0)
    assert s["examples"][0] == {"driver": "OCO", "lap": 2, "from": "STAY OUT", "to": "PIT NOW"}


@pytest.mark.parametrize("stops,sc,delta,reason", [
    ({"RUS": [2]}, {1: False, 2: False}, 2.0, "pit_rival"),
    ({"OCO": [2]}, {1: False, 2: False}, 2.0, "pit_self"),
    ({}, {1: False, 2: True}, 2.0, "sc_change"),
    ({}, {1: False, 2: False}, 0.5, "margin"),
])
def test_a_change_the_state_accounts_for_is_not_churn(stops, sc, delta, reason):
    by_lap = {1: {"OCO": _dec("STAY OUT", delta=2.0)}, 2: {"OCO": _dec("PIT NOW", delta=delta)}}
    s = bench_live.decision_stability(by_lap, stops, sc)
    assert s["n_changes"] == 1, reason
    assert s["n_unexplained"] == 0, f"{reason} should explain the change"
    assert s["reasons"][reason] == 1


def test_a_rival_not_in_the_decision_does_not_explain_a_change():
    by_lap = {1: {"OCO": _dec("STAY OUT", rivals=("RUS",))},
              2: {"OCO": _dec("PIT NOW", rivals=("RUS",))}}
    s = bench_live.decision_stability(by_lap, {"NOR": [2]}, {1: False, 2: False})
    assert s["n_unexplained"] == 1


def test_a_gap_in_a_car_s_laps_is_not_scored_as_a_pair():
    """A car that dropped out of the field for a lap has no previous decision."""
    by_lap = {1: {"OCO": _dec("STAY OUT")}, 2: {}, 3: {"OCO": _dec("PIT NOW")}}
    s = bench_live.decision_stability(by_lap, {}, {1: False, 2: False, 3: False})
    assert s["n"] == 2 and s["n_pairs"] == 0 and s["n_changes"] == 0


def test_plans_without_a_decision_block_score_nothing():
    """Task 1's plans have no `decision`: n = 0 and nulls, never a flattering 0."""
    by_lap = {1: {"OCO": {"plan_best": "1-stop M-H @ 20"}}, 2: {"OCO": {"box_now": 0.4}}}
    s = bench_live.decision_stability(by_lap, {}, {1: False, 2: False})
    assert s["n"] == 0 and s["n_pairs"] == 0
    assert s["n_changes"] is None and s["share_unexplained"] is None
    assert "no decision block" in s["note"]


def test_decision_rivals_are_read_out_of_any_shape_the_engine_writes():
    assert bench_live._decision_rivals({"rivals": [{"driver": "RUS"}]}) == {"RUS"}
    assert bench_live._decision_rivals({"rivals": ["RUS", "NOR"]}) == {"RUS", "NOR"}
    assert bench_live._decision_rivals({"rivals": [{"driver_number": "63"}]}) == {"63"}
    assert bench_live._decision_rivals({}) == set()


def test_haas_stop_rows_score_only_the_two_haas_cars():
    import pandas as pd

    per = pd.DataFrame([
        {"driver": "OCO", "in_lap": 20, "rec_3_before": 18, "in_window": True,
         "err_laps": 2.0, "box_now_delta_1_before": 0.5},
        {"driver": "BEA", "in_lap": 24, "rec_3_before": 18, "in_window": False,
         "err_laps": 6.0, "box_now_delta_1_before": 1.5},
        {"driver": "RUS", "in_lap": 22, "rec_3_before": 22, "in_window": True,
         "err_laps": 0.0, "box_now_delta_1_before": 0.1},
    ])
    h = bench_live.haas_stop_rows(per)
    assert h["n"] == 2 and h["drivers"] == ["OCO", "BEA"]
    assert h["share_err_within_3"] == pytest.approx(0.5)
    assert h["median_abs_err_laps"] == pytest.approx(4.0)
    assert h["median_box_now_delta_1_before"] == pytest.approx(1.0)
    assert h["by_driver"]["OCO"]["n"] == 1 and h["by_driver"]["BEA"]["median_abs_err_laps"] == 6.0
    empty = bench_live.haas_stop_rows(pd.DataFrame())
    assert empty["n"] == 0 and empty["median_abs_err_laps"] is None
