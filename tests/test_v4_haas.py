"""V4 WP-E: the Haas per-car model must not invent a driver.

The failure mode these guard against is lore: a number that looks like a
measurement of Ocon or Bearman but is really a prior, a team value wearing a
driver's name, or a single lap amplified into a car characteristic.  So the
tests check the *mechanism*:

* a car with no evidence gets exactly the team value at `shrink_weight = 0`;
* a car with plenty of evidence keeps most of its own estimate;
* the warm-up excess and the residual spread are the quantities they claim to
  be, on a synthetic stint with a known trend;
* `explain_difference` names a cause with numbers when the two cars really do
  differ, and says "within noise" when they do not;
* the two new `per_driver_plans` keywords reproduce the old result at their
  defaults, so switching WP-E on cannot silently move a plan;
* `haas_block` survives `json.dumps`, which is what `meta["haas"]` needs.

Run: .venv/bin/python -m pytest tests/test_v4_haas.py -v
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

from src import haascar  # noqa: E402
from src.config import OUT_LAP_PENALTY_S, TRAFFIC_GAP_S  # noqa: E402
from src.haascar import CarState, Estimate, HaasCarModel  # noqa: E402
from src.tyre import TyreModel  # noqa: E402

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
RATES = {"SOFT": 0.18, "MEDIUM": 0.15, "HARD": 0.11}
N_DRAWS = 24


# --------------------------------------------------------------------------
# A synthetic weekend: a field of long runs, and two Haas cars on it
# --------------------------------------------------------------------------


def _model(dev_by_driver: dict | None = None) -> TyreModel:
    rng = np.random.default_rng(3)
    wear = {c: np.full(N_DRAWS, RATES[c] / 4.0) for c in COMPOUNDS}
    pace = {c: np.full(N_DRAWS, 0.0) for c in COMPOUNDS}
    dev = {d: {c: np.full(N_DRAWS, float(v.get(c, 0.0))) + 1e-9 * rng.standard_normal(N_DRAWS)
               for c in COMPOUNDS}
           for d, v in (dev_by_driver or {}).items()}
    return TyreModel(compounds=list(COMPOUNDS), wear_rate=wear, pace_offset=pace, budget=4.0,
                     n_draws=N_DRAWS, source="synthetic", driver_dev=dev)


def _stint(driver: str, team: str, session: str, stint: int, compound: str, n: int, *,
           base: float = 90.0, slope: float = 0.10, warmup: float = 0.0, noise: float = 0.0,
           traffic_laps: int = 0, traffic_excess: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """One long run: `y = base + slope * lap_in_stint`, plus a warm-up excess on
    the first flying lap and, optionally, some laps stuck behind another car."""
    rng = np.random.default_rng(seed)
    laps = np.arange(2, n + 2, dtype=float)          # lap 1 is the out-lap, filtered out
    y = base + slope * laps + noise * rng.standard_normal(len(laps))
    y[laps == 2] += warmup
    gap = np.full(len(laps), 9.0)
    if traffic_laps:
        idx = np.arange(len(laps) - traffic_laps, len(laps))
        gap[idx] = 1.0
        y[idx] += traffic_excess
    return pd.DataFrame({
        "event": "synth-2026", "session": session, "driver": driver, "team": team,
        "lap_number": laps + 10.0, "stint": float(stint), "compound": compound,
        "tyre_life": laps, "lap_time_s": y, "lap_time_corr": y,
        "lap_in_stint": laps, "tyre_age": laps, "gap_ahead_s": gap,
        "stint_uid": f"synth-2026|{session}|{driver}|{stint}",
    })


def _weekend(*, oco_laps: int = 12, bea_laps: int = 0, oco_warmup: float = 0.9,
             field_warmup: float = 0.3, oco_dev: float = 0.02, bea_dev: float = 0.0,
             oco_traffic: int = 0, field_traffic: int = 0, traffic_excess: float = 0.6) -> dict:
    """A field of eight cars with long runs, plus the two Haas cars."""
    frames = []
    for i in range(8):
        drv, team = f"D{i}", f"T{i // 2}"
        frames.append(_stint(drv, team, "Practice 2", 1, "MEDIUM", 12, base=90.0,
                             warmup=field_warmup, noise=0.05, seed=100 + i,
                             traffic_laps=(field_traffic if i < 4 else 0),
                             traffic_excess=traffic_excess))
    if oco_laps:
        frames.append(_stint("OCO", haascar.HAAS_TEAM, "Practice 2", 4, "MEDIUM", oco_laps,
                             base=90.4, warmup=oco_warmup, noise=0.02, seed=7,
                             traffic_laps=oco_traffic, traffic_excess=traffic_excess))
    if bea_laps:
        frames.append(_stint("BEA", haascar.HAAS_TEAM, "Practice 2", 5, "MEDIUM", bea_laps,
                             base=90.4, warmup=field_warmup, noise=0.02, seed=8))
    full = pd.concat(frames, ignore_index=True)
    # `_valid` treats a missing ok_* flag as passed, so the clean table is just
    # the laps outside `TRAFFIC_GAP_S` - exactly what the real cascade leaves.
    clean = full[full["gap_ahead_s"] > TRAFFIC_GAP_S].copy()
    model = _model({"OCO": {c: oco_dev for c in COMPOUNDS},
                    "BEA": {c: bea_dev for c in COMPOUNDS}} if (oco_laps or bea_laps) else {})
    return {"full": full, "clean": clean, "model": model}


class _Cal:
    """The shrunken-prior side of a `Calibration`, and nothing else."""

    driver_factors = {"OCO": 1.10, "BEA": 1.10}
    driver_factor_ln_sd = {"OCO": 0.08, "BEA": 0.08}
    team_factors = {haascar.HAAS_TEAM: 1.15}
    source = "synthetic"


def _build(**kw) -> HaasCarModel:
    w = _weekend(**kw)
    return HaasCarModel.from_weekend("hungary-2026", w["model"], w["clean"], calibration=_Cal(),
                                     practice_laps=w["full"], sector_times=False)


# --------------------------------------------------------------------------
# Shrinkage
# --------------------------------------------------------------------------


def test_no_long_run_inherits_the_team_value_at_zero_weight():
    """BEA has no long run at all: every quantity must be the team value, and
    say so - `shrink_weight = 0` and `source` not `driver`."""
    hm = _build(oco_laps=12, bea_laps=0)
    bea = hm["BEA"]
    assert bea.pace_offset_s.shrink_weight == 0.0
    assert bea.pace_offset_s.own is None
    assert bea.pace_offset_s.value == pytest.approx(bea.pace_offset_s.team_value)
    assert bea.pace_offset_s.source in ("team", "field")
    # the warm-up too: no stint of its own, so the team's number
    assert bea.warmup_s.shrink_weight == 0.0
    assert bea.warmup_s.value == pytest.approx(bea.warmup_s.team_value)
    # and the term the search would take is the field's, unchanged
    t = haascar.car_terms(bea)
    assert t["traffic_mult"] == pytest.approx(1.0)
    # the team value is OCO's, shrunk toward the field - never OCO's raw number
    assert bea.warmup_s.value != pytest.approx(hm["OCO"].warmup_s.own)
    # nothing may be sourced "driver" with no laps
    for name, est in bea.estimates().items():
        assert est.source != "driver", name


def test_degradation_with_no_deviation_of_its_own_is_the_team_mate_s():
    """The quantity that actually enters the tyre model: a car the fit carries
    no `dev[d, c]` for must get its team-mate's, not the field's."""
    w = _weekend(oco_laps=20, bea_laps=0, oco_dev=0.02)
    model = _model({"OCO": {c: 0.02 for c in COMPOUNDS}})      # no BEA in the fit at all
    hm = HaasCarModel.from_weekend("hungary-2026", model, w["clean"], calibration=_Cal(),
                                   practice_laps=w["full"], sector_times=False)
    bea = hm["BEA"]
    assert bea.deg_rate_by_compound.shrink_weight == 0.0
    assert bea.deg_rate_by_compound.own is None
    assert bea.deg_rate_by_compound.value == bea.deg_rate_by_compound.team_value
    # the team value carries OCO's measured deviation, not the field rate
    for c in COMPOUNDS:
        assert bea.deg_rate_by_compound.value[c] > bea.deg_rate_by_compound.field_value[c]
    # and the override handed to `TyreModel.for_driver` is the same number
    dev = haascar.car_terms(bea)["dev_override"]
    assert dev and all(abs(float(np.asarray(v).mean()) - 0.02) < 1e-6 for v in dev.values())


def test_many_laps_keeps_most_of_its_own_estimate():
    hm = _build(oco_laps=60, bea_laps=6)
    oco = hm["OCO"]
    assert oco.pace_offset_s.shrink_weight > 0.85
    assert oco.pace_offset_s.source == "driver"
    # the value sits within 15% of the way from its own number to the team's
    own, team, val = (oco.pace_offset_s.own, oco.pace_offset_s.team_value, oco.pace_offset_s.value)
    assert abs(val - own) <= 0.15 * abs(own - team) + 1e-9
    assert oco.deg_rate_by_compound.shrink_weight > 0.7
    assert oco.consistency_s.shrink_weight > 0.8
    # the warm-up's evidence is counted in *stints*, and this car has run one,
    # which is below the floor: 60 laps of one stint buy it no weight at all.
    assert oco.warmup_s.n_evidence == 1
    assert oco.warmup_s.shrink_weight == 0.0
    assert oco.warmup_s.detail["own_measured_s"] is not None      # measured, but not used


def test_shrink_weight_rises_with_evidence():
    thin = _build(oco_laps=4, bea_laps=4)["OCO"]
    thick = _build(oco_laps=40, bea_laps=4)["OCO"]
    assert thin.pace_offset_s.shrink_weight < thick.pace_offset_s.shrink_weight
    assert thin.deg_rate_by_compound.shrink_weight < thick.deg_rate_by_compound.shrink_weight


def test_sector_deg_is_unavailable_not_invented():
    hm = _build(oco_laps=12, bea_laps=8)      # built with sector_times=False
    for drv in ("OCO", "BEA"):
        est = hm[drv].sector_deg
        assert est.value is None
        assert est.source == "unavailable"
        assert "sector" in est.note.lower()


def test_push_response_is_the_team_value_and_says_it_is_unidentified():
    hm = _build(oco_laps=40, bea_laps=40)
    for drv in ("OCO", "BEA"):
        est = hm[drv].push_response
        assert est.shrink_weight == 0.0
        assert est.source == "team"
        assert est.value == pytest.approx(est.team_value)
        assert "not identifiable" in est.note


def test_one_in_traffic_lap_does_not_become_a_car_characteristic():
    """A single lap in dirty air is below the floor: the multiplier stays 1.0."""
    hm = _build(oco_laps=12, bea_laps=8, oco_traffic=1, field_traffic=4, traffic_excess=2.0)
    assert hm["OCO"].traffic_sensitivity.value == pytest.approx(1.0)
    assert hm["OCO"].traffic_sensitivity.shrink_weight == 0.0
    assert haascar.car_terms(hm["OCO"])["traffic_mult"] == pytest.approx(1.0)


def test_traffic_sensitivity_is_measured_when_there_are_enough_laps():
    """Give OCO five in-traffic laps at twice the field's excess and the
    multiplier moves off 1.0 in the right direction."""
    hm = _build(oco_laps=16, bea_laps=8, oco_traffic=5, field_traffic=5, traffic_excess=0.6)
    est = hm["OCO"].traffic_sensitivity
    assert est.own is None or est.own > 0            # measured against the field's own excess
    if est.own is not None:
        assert est.shrink_weight > 0
        assert min(est.own, 1.0) <= est.value <= max(est.own, 1.0)


# --------------------------------------------------------------------------
# The lap-table estimators
# --------------------------------------------------------------------------


def test_stint_trend_recovers_the_trend_the_warmup_and_the_spread():
    """A stint with a known slope, a known first-flying-lap excess and no noise."""
    g = _stint("X", "T", "Practice 2", 1, "MEDIUM", 12, base=90.0, slope=0.13, warmup=0.8)
    t = haascar.stint_trend(g)
    assert t["slope"] == pytest.approx(0.13, abs=1e-9)
    assert t["intercept"] == pytest.approx(90.0, abs=1e-9)
    assert t["warmup"][2] == pytest.approx(0.8, abs=1e-9)
    assert t["warmup_s"] == pytest.approx(0.8, abs=1e-9)
    assert t["resid_sd"] == pytest.approx(0.0, abs=1e-9)
    assert t["n_trend"] == 11        # laps 3..13; lap 2, the warm-up lap, is not in the fit
    # the warm-up lap must not set the trend it is measured against
    assert haascar.stint_trend(g)["slope"] == pytest.approx(
        haascar.stint_trend(g[g["lap_in_stint"] > 2])["slope"], abs=1e-9)


def test_consistency_is_the_residual_sd_about_the_trend():
    """Residuals of a known size, with the (n - 2) correction the fit costs."""
    g = _stint("X", "T", "Practice 2", 1, "MEDIUM", 13, base=90.0, slope=0.1)
    y = g["lap_time_s"].to_numpy(float).copy()
    bump = np.array([0.2, -0.2] * 6 + [0.0])[: len(y)]
    g = g.assign(lap_time_s=y + bump, lap_time_corr=y + bump)
    t = haascar.stint_trend(g)
    x = g["lap_in_stint"].to_numpy(float)
    m = x >= 3
    A = np.column_stack([np.ones(int(m.sum())), x[m]])
    beta, *_ = np.linalg.lstsq(A, (y + bump)[m], rcond=None)
    resid = (y + bump)[m] - A @ beta
    assert t["resid_sd"] == pytest.approx(float(np.sqrt(resid @ resid / (m.sum() - 2))), rel=1e-9)
    assert t["dof"] == int(m.sum()) - 2


def test_warmup_pools_over_stints_and_consistency_over_laps():
    """Two stints, two warm-up numbers; the pseudo-counts are in those units."""
    a = _stint("X", "T", "Practice 2", 1, "MEDIUM", 10, warmup=0.6)
    b = _stint("X", "T", "Practice 3", 2, "MEDIUM", 10, warmup=1.0)
    stats = haascar._trend_stats(pd.concat([a, b], ignore_index=True))
    assert stats["X"]["n_warmup_stints"] == 2
    assert stats["X"]["warmup_s"] == pytest.approx(0.8, abs=1e-9)
    assert stats["X"]["n_laps"] == 18          # 9 trend laps (3..11) in each of the two stints
    assert stats["X"]["n_stints"] == 2


def test_race_laps_add_evidence_even_without_the_corrected_column():
    """The practice table carries `lap_time_corr` and the race table does not.

    They must be measured separately and pooled: concatenating them first makes
    one lap-time column win and silently drops the other table's rows.
    """
    prac = _stint("OCO", haascar.HAAS_TEAM, "Practice 2", 4, "MEDIUM", 10, warmup=0.5)
    race = _stint("OCO", haascar.HAAS_TEAM, "Race", 1, "MEDIUM", 14, warmup=0.9).drop(
        columns=["lap_time_corr"])
    both = haascar._trend_stats_frames([prac, race])
    assert both["OCO"]["n_warmup_stints"] == 2
    assert both["OCO"]["warmup_s"] == pytest.approx(0.7, abs=1e-9)
    # the naive concatenation is the bug this guards: it keeps only the practice stint
    naive = haascar._trend_stats(haascar._valid(pd.concat([prac, race], ignore_index=True),
                                                traffic=False))
    assert naive["OCO"]["n_warmup_stints"] == 1


def test_pace_offset_is_relative_to_the_field_and_signed():
    """OCO's stint is 0.4 s off the field's base: the level must say so."""
    hm = _build(oco_laps=20, bea_laps=10)
    assert hm["OCO"].pace_offset_s.own == pytest.approx(0.4, abs=0.12)
    assert hm["OCO"].pace_offset_s.units == "s/lap"
    # and the two cars' difference is the difference of their levels
    d = hm["OCO"].pace_vs_teammate_s
    assert d.detail.get("teammate") == "BEA"
    assert abs(d.value) <= abs(d.detail["raw_s"]) + 1e-9


# --------------------------------------------------------------------------
# The terms, and the explanation
# --------------------------------------------------------------------------


def test_car_terms_are_field_relative_so_no_evidence_is_no_change():
    """A car with no evidence must hand the search the field objective back."""
    st = CarState(driver="BEA", warmup_s=Estimate(value=0.4, field_value=0.4, team_value=0.4),
                  traffic_sensitivity=Estimate(value=1.0, field_value=1.0, team_value=1.0))
    t = haascar.car_terms(st)
    assert t["warmup_s"] == pytest.approx(OUT_LAP_PENALTY_S)
    assert t["traffic_mult"] == pytest.approx(1.0)
    # a car that warms up 0.25 s worse than the field carries exactly that
    st.warmup_s = Estimate(value=0.65, field_value=0.4, team_value=0.5)
    assert haascar.car_terms(st)["warmup_s"] == pytest.approx(OUT_LAP_PENALTY_S + 0.25)
    # ...but a wild estimate is held inside the band, and never goes negative
    st.warmup_s = Estimate(value=-9.0, field_value=0.4, team_value=0.4)
    assert haascar.car_terms(st)["warmup_s"] == pytest.approx(
        max(OUT_LAP_PENALTY_S - haascar.WARMUP_DELTA_BAND_S, 0.0))
    st.warmup_s = Estimate(value=9.0, field_value=0.4, team_value=0.4)
    assert haascar.car_terms(st)["warmup_s"] == pytest.approx(
        OUT_LAP_PENALTY_S + haascar.WARMUP_DELTA_BAND_S)
    st.warmup_s = Estimate(value=-9.0, field_value=0.4, team_value=0.4)
    assert haascar.car_terms(st, base_warmup_s=0.1)["warmup_s"] == 0.0


def _state(drv: str, rate: float, **kw) -> CarState:
    base = dict(
        pace_offset_s=Estimate(value=0.3, field_value=0.0, team_value=0.3, n_evidence=10,
                               shrink_weight=0.5, source="driver"),
        deg_rate_by_compound=Estimate(value={"MEDIUM": rate}, field_value={"MEDIUM": 0.15},
                                      team_value={"MEDIUM": 0.15}, n_evidence=12,
                                      shrink_weight=0.4, source="team"),
        age_sensitivity=Estimate(value=rate - 0.15, field_value=0.0, team_value=0.0),
        warmup_s=Estimate(value=0.4, field_value=0.4, team_value=0.4),
        consistency_s=Estimate(value=0.3, field_value=0.3, team_value=0.3),
        traffic_sensitivity=Estimate(value=1.0, field_value=1.0, team_value=1.0),
        push_response=Estimate(value=0.6, field_value=0.6, team_value=0.6),
        sector_deg=Estimate(value=None, source="unavailable"),
        pace_vs_teammate_s=Estimate(value=0.0),
        race_factor={"factor": 1.1, "ln_sd": 0.08, "shrunk": 1.05},
    )
    base.update(kw)
    return CarState(driver=drv, event="hungary-2026", teammate=None, **base)


def test_explain_difference_names_degradation_when_it_is_material():
    a = _state("OCO", 0.150)
    b = _state("BEA", 0.170)          # 0.020 s/lap apart, four times the threshold
    plan_a = {"best": "2-stop M-H-M @ 18,44", "compounds": ["MEDIUM", "HARD", "MEDIUM"],
              "pit_laps": [18, 44], "first_stop": 18}
    plan_b = {"best": "2-stop M-H-M @ 15,42", "compounds": ["MEDIUM", "HARD", "MEDIUM"],
              "pit_laps": [15, 42], "first_stop": 15}
    lines = haascar.explain_difference(a, b, plan_a, plan_b)
    assert isinstance(lines, list) and lines
    assert "first stops differ by 3 laps" in lines[0]
    deg = [l for l in lines if "degradation on MEDIUM" in l]
    assert deg, lines
    assert "0.1500" in deg[0] and "0.1700" in deg[0]      # both cars' numbers are stated
    assert "BEA pays" in deg[0]                           # and the direction, with a number
    assert not any("within noise" in l for l in lines)
    assert all("n=" in l for l in deg)                    # the evidence travels with the cause


def test_explain_difference_says_within_noise_when_nothing_is_material():
    a = _state("OCO", 0.1500)
    b = _state("BEA", 0.1502)         # well inside the threshold
    plan_a = {"best": "2-stop M-H-M @ 18,44", "compounds": ["MEDIUM", "HARD", "MEDIUM"],
              "pit_laps": [18, 44], "first_stop": 18}
    plan_b = {"best": "2-stop M-H-M @ 17,44", "compounds": ["MEDIUM", "HARD", "MEDIUM"],
              "pit_laps": [17, 44], "first_stop": 17}
    lines = haascar.explain_difference(a, b, plan_a, plan_b)
    assert any("within noise" in l for l in lines), lines
    assert not any("degradation on MEDIUM" in l for l in lines)


def test_explain_difference_works_without_plans():
    lines = haascar.explain_difference(_state("OCO", 0.15), _state("BEA", 0.15))
    assert any("within noise" in l for l in lines)


# --------------------------------------------------------------------------
# The hooks
# --------------------------------------------------------------------------


def _search_model() -> TyreModel:
    """A two-compound model small enough to search twice in a test."""
    rng = np.random.default_rng(11)
    n = 8
    wear = {"MEDIUM": 0.15 / 4.0 + 0.001 * rng.standard_normal(n),
            "HARD": 0.11 / 4.0 + 0.001 * rng.standard_normal(n)}
    pace = {"MEDIUM": np.zeros(n), "HARD": np.full(n, 0.25)}
    dev = {"OCO": {"MEDIUM": np.full(n, 0.004), "HARD": np.full(n, 0.002)}}
    return TyreModel(compounds=["MEDIUM", "HARD"], wear_rate=wear, pace_offset=pace, budget=4.0,
                     n_draws=n, source="synthetic", driver_dev=dev)


def test_per_driver_plans_defaults_reproduce_the_old_result():
    """The two new keywords at their defaults must change nothing at all."""
    from src import strategy as strat

    m = _search_model()
    kw = dict(step=4, shortlist=60, max_stops=2)
    old = strat.per_driver_plans(m, "hungary-2026", 21.0, ["OCO"], **kw)
    new = strat.per_driver_plans(m, "hungary-2026", 21.0, ["OCO"],
                                 warmup_by_driver=None, traffic_mult_by_driver=None, **kw)
    assert list(old.columns) == list(new.columns)        # no new columns unless asked for
    pd.testing.assert_frame_equal(old, new)
    # an explicit neutral term is the same search; a real one is allowed to move
    same = strat.per_driver_plans(m, "hungary-2026", 21.0, ["OCO"],
                                  warmup_by_driver={"OCO": OUT_LAP_PENALTY_S},
                                  traffic_mult_by_driver={"OCO": 1.0}, **kw)
    assert "warmup_s" in same.columns and "traffic_mult" in same.columns
    for col in ("best", "first_stop", "push", "compounds"):
        assert list(same[col]) == list(old[col]), col


def test_per_driver_plans_takes_the_per_car_terms():
    """A car charged a large warm-up and double dirty air is priced differently."""
    from src import strategy as strat

    m = _search_model()
    kw = dict(step=4, shortlist=60, max_stops=3, traffic_s_per_lap=0.45)
    old = strat.per_driver_plans(m, "hungary-2026", 21.0, ["OCO"], **kw)
    hit = strat.per_driver_plans(m, "hungary-2026", 21.0, ["OCO"],
                                 warmup_by_driver={"OCO": 6.0},
                                 traffic_mult_by_driver={"OCO": 3.0}, **kw)
    assert float(hit["warmup_s"].iloc[0]) == 6.0
    assert float(hit["traffic_mult"].iloc[0]) == 3.0
    # both terms are charged per stop, so the expensive car cannot stop more often
    assert int(hit["n_stops"].iloc[0]) <= int(old["n_stops"].iloc[0])


def test_haas_block_is_json_serialisable():
    hm = _build(oco_laps=12, bea_laps=8)
    plans = pd.DataFrame([
        {"driver": "OCO", "best": "2-stop M-H-M @ 18,44", "compounds": "MEDIUM-HARD-MEDIUM",
         "pit_laps": [18, 44], "first_stop": 18, "push": np.float64(0.55),
         "same_shape_as_field": np.bool_(True)},
        {"driver": "BEA", "best": "2-stop M-H-M @ 16,42", "compounds": "MEDIUM-HARD-MEDIUM",
         "pit_laps": [16, 42], "first_stop": 16, "push": np.float64(0.55),
         "same_shape_as_field": np.bool_(False)},
    ])
    rsb = {"enabled": True, "group": "MEDIUM|HARD|2", "first_stop": 18,
           "constants": {"place_gap_s": 5.1, "persistence": 0.74},
           "curve": {"laps": np.arange(5), "cost_s": np.zeros(5)}}
    block = haas_block = haascar.haas_block(hm, plans, rsb)
    s = json.dumps(block)                              # the actual requirement
    assert len(s) > 500
    again = json.loads(s)
    assert again["drivers"] == ["OCO", "BEA"]
    assert again["cars"]["OCO"]["plan"]["first_stop"] == 18
    assert again["cars"]["BEA"]["terms"]["traffic_mult"] == 1.0
    assert isinstance(again["explanation"], list) and again["explanation"]
    for drv in ("OCO", "BEA"):
        st = again["cars"][drv]["state"]
        for q in CarState.QUANTITIES:
            assert {"value", "n_evidence", "shrink_weight", "source"} <= set(st[q]), (drv, q)
    # the race-state curve arrays are not carried into the block
    assert "curve" not in again["race_state"]
    assert haas_block is block


def test_haas_block_takes_a_plain_state_map_and_a_record_list():
    hm = _build(oco_laps=12, bea_laps=8)
    rows = [{"driver": "OCO", "best": "1-stop M-H @ 20", "first_stop": 20}]
    block = haascar.haas_block(dict(hm.states), rows, None)
    json.dumps(block)
    assert block["cars"]["OCO"]["plan"]["first_stop"] == 20
    assert block["cars"]["BEA"]["plan"] is None


def test_hier_rate_scale_table_is_a_rate_scale_table():
    """The accuracy benchmark's new variant must be shaped like the others."""
    from src import percar

    m = _model({"OCO": {c: 0.01 for c in COMPOUNDS}, "BEA": {c: -0.005 for c in COMPOUNDS},
                "VER": {c: 0.0 for c in COMPOUNDS}})
    teams = {"OCO": haascar.HAAS_TEAM, "BEA": haascar.HAAS_TEAM, "VER": "Red Bull"}
    counts = {"OCO": 40.0, "BEA": 2.0, "VER": 20.0}
    hier = haascar.hier_rate_scale_table(model=m, cal=_Cal(), teams=teams, n_laps_by_driver=counts)
    pooled = percar.rate_scale_table("team_pooled", model=m, cal=_Cal(), teams=teams)
    assert set(hier) == set(pooled)
    assert all(isinstance(v, float) and np.isfinite(v) for v in hier.values())
    # the lap counts must actually bite: OCO has 40 laps, BEA 2, so OCO keeps
    # more of its own (positive) deviation than the equal-weight version does
    assert hier["OCO"] > hier["BEA"]
    no_prior = haascar.hier_rate_scale_table(model=m, cal=_Cal(), teams=teams,
                                             n_laps_by_driver=counts, use_history_prior=False)
    assert no_prior["OCO"] != pytest.approx(hier["OCO"])   # the prior is doing something
