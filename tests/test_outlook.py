"""Tests for the outlook, the plan tools and the committed-plan store.

Run: .venv/bin/python -m pytest tests/test_outlook.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import strategy as strat  # noqa: E402
from src.config import DATA_PROCESSED, get_event  # noqa: E402
from src.history import apply_rate_prior_to_model, same_circuit  # noqa: E402
from src.regime import RegimeFactor  # noqa: E402


# --------------------------------------------------------------------------
# history: circuit matching and the pooled prior
# --------------------------------------------------------------------------


def test_same_circuit_rejects_fuzzy_mismatch():
    sg = {"Location": "Marina Bay", "EventName": "Singapore Grand Prix", "Country": "Singapore"}
    assert not same_circuit("Madring", sg)
    assert same_circuit("Singapore", sg)
    assert same_circuit("Monza", {"Location": "Monza", "EventName": "Italian Grand Prix", "Country": "Italy"})
    assert same_circuit("Monte Carlo", {"Location": "Monaco", "EventName": "Monaco Grand Prix", "Country": "Monaco"})
    assert same_circuit("Interlagos", {"Location": "São Paulo", "EventName": "São Paulo Grand Prix", "Country": "Brazil"})
    assert same_circuit("Yas Marina Circuit", {"Location": "Yas Island", "EventName": "Abu Dhabi Grand Prix", "Country": "UAE"})
    assert not same_circuit("Shanghai", {"Location": "Budapest", "EventName": "Hungarian Grand Prix", "Country": "Hungary"})


def _ladder_model(n=200):
    from src.live.engine import WeekendModel

    return WeekendModel.prior_model(get_event("spain-2026"), n, np.random.default_rng(1))


def test_pooled_prior_keeps_the_ladder_ordered():
    model = _ladder_model()
    # a history that, read per compound, would invert the ladder
    prior = {"SOFT": {"mean_s_per_lap": 0.03, "ln_sd": 0.4},
             "MEDIUM": {"mean_s_per_lap": 0.12, "ln_sd": 0.4},
             "HARD": {"mean_s_per_lap": 0.09, "ln_sd": 0.4}}
    rg = RegimeFactor(ratio=0.6, ln_sd=0.35)
    pooled, rows = apply_rate_prior_to_model(model, get_event("spain-2026"), rg, prior, seed=0, pooled=True)
    r = {c: float((pooled.wear_rate[c] * pooled.budget).mean()) for c in pooled.compounds}
    assert r["SOFT"] > r["MEDIUM"] > r["HARD"]
    assert len(rows) == 3 and all(rows["pooled"])
    per, _ = apply_rate_prior_to_model(model, get_event("spain-2026"), rg, prior, seed=0, pooled=False)
    rp = {c: float((per.wear_rate[c] * per.budget).mean()) for c in per.compounds}
    assert rp["SOFT"] < rp["MEDIUM"]      # the per-compound reading does invert - which is the point


def test_same_regime_fold_moves_toward_the_measurement():
    model = _ladder_model()
    before = float((model.wear_rate["MEDIUM"] * model.budget).mean())
    live = {"MEDIUM": {"mean_s_per_lap": 0.25, "ln_sd": 0.3}}
    after, _ = apply_rate_prior_to_model(model, get_event("spain-2026"), RegimeFactor(), live, same_regime=True)
    a = float((after.wear_rate["MEDIUM"] * after.budget).mean())
    assert before < a < 0.25
    # the other compounds move with the ladder
    assert float((after.wear_rate["HARD"] * after.budget).mean()) > float((model.wear_rate["HARD"] * model.budget).mean())


# --------------------------------------------------------------------------
# strategy: the model-based search and the plan tools
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def italy():
    post = DATA_PROCESSED / "posterior_italy-2026.npz"
    if not post.exists():
        pytest.skip("no sealed Italy posterior on disk")
    from src.model_bayes import BayesFit

    return get_event("italy-2026"), BayesFit.load(post)


def test_simulate_model_equals_simulate_from_fit(italy):
    ev, fit = italy
    kw = dict(n_draws=200, support={"SOFT": 20.0, "MEDIUM": 22.0, "HARD": 22.0},
              max_stint={"SOFT": 10, "MEDIUM": 51, "HARD": 56})
    res = strat.simulate(fit, ev, 25.3, **kw)
    assert res.times is not None and res.times.shape == (len(res.table), 200)
    assert np.allclose(res.times.mean(1), res.table["mean_s"])
    assert abs(sum(res.p_stops.values()) - 1.0) < 1e-9
    assert res.best_label.startswith("1-stop")
    # evaluating the winning plan by hand prices it exactly as the search did
    tbl, det = strat.evaluate_plans(res.model, ev, [res.best], 25.3)
    assert det[0]["valid"]
    assert np.abs(det[0]["times"] - res.times[0]).max() < 1e-4
    assert len(det[0]["trace_mean"]) == ev.n_race_laps


def test_evaluate_plans_flags_and_invalid(italy):
    ev, fit = italy
    model = strat.TyreModel.from_fit(fit, draws=np.arange(0, fit.posterior["lin"].shape[0], max(fit.posterior["lin"].shape[0] // 100, 1)))
    plans = [{"compounds": ["MEDIUM", "MEDIUM"], "pit_laps": [25]},            # one compound
             {"compounds": ["SOFT", "HARD"], "pit_laps": [40, 10]},              # out of order
             {"compounds": ["MEDIUM", "HARD"], "pit_laps": [24], "push": 0.7}]
    tbl, det = strat.evaluate_plans(model, ev, plans, 25.3, allocation={"SOFT": 2, "MEDIUM": 2, "HARD": 2},
                                    stint_cap={"SOFT": 10, "MEDIUM": 51, "HARD": 56})
    assert "one compound" in det[0]["flags"][0]
    assert det[1]["valid"] is False
    assert det[2]["valid"] and det[2]["push"] == 0.7
    assert tbl.loc[2, "p_fastest"] >= 0


def test_sc_playbook_verdicts(italy):
    ev, fit = italy
    model = strat.TyreModel.from_fit(fit, draws=np.arange(0, fit.posterior["lin"].shape[0], max(fit.posterior["lin"].shape[0] // 100, 1)))
    plan = {"compounds": ["MEDIUM", "HARD"], "pit_laps": [24], "push": 1.0}
    pb = strat.sc_playbook(model, ev, plan, 25.3, allocation={"SOFT": 2, "MEDIUM": 2, "HARD": 2})
    assert set(pb["verdict"]) <= {"PIT", "STAY", "MARGINAL", "PLANNED"}
    assert pb.loc[pb["lap"] == 24, "verdict"].iloc[0] == "PLANNED"
    # a safety car just before the planned stop is a free stop
    assert pb.loc[pb["lap"] == 20, "verdict"].iloc[0] == "PIT"
    ranges = strat.playbook_ranges(pb)
    assert ranges and ranges[0]["from"] == 1 and all(r["to"] >= r["from"] for r in ranges)


def test_crossover_and_duel(italy):
    ev, fit = italy
    model = strat.TyreModel.from_fit(fit, draws=np.arange(0, fit.posterior["lin"].shape[0], max(fit.posterior["lin"].shape[0] // 100, 1)))
    a = {"compounds": ["MEDIUM", "HARD"], "pit_laps": [24], "push": 1.0}
    b = {"compounds": ["MEDIUM", "HARD", "MEDIUM"], "pit_laps": [17, 37], "push": 1.0}
    cx = strat.deg_crossover(model, ev, a, b, 25.3, mults=[1.0, 2.0, 3.0, 4.0])
    assert cx["curve"][0]["b_minus_a_s"] > 0
    assert cx["mult"] is None or cx["mult"] > 1.0
    d = strat.undercut_duel(model, my_compound="MEDIUM", my_age=20, their_compound="MEDIUM", their_age=20,
                            gap_s=0.2, new_compound="SOFT", push=1.0, event=ev, lap_now=21)
    assert len(d["p_by_lap"]) == 5 and 0 <= d["p_undercut_3lap"] <= 1


def test_value_of_information(italy):
    ev, fit = italy
    res = strat.simulate(fit, ev, 25.3, n_draws=150, max_stint={"SOFT": 10, "MEDIUM": 51, "HARD": 56})
    voi = strat.value_of_information(res)
    assert voi["evpi_s"] >= 0
    assert set(voi["by_compound"]) == set(res.model.compounds)
    assert all(v["gain_s"] <= voi["evpi_s"] + 1e-9 for v in voi["by_compound"].values())


# --------------------------------------------------------------------------
# the outlook end to end, and the plan store
# --------------------------------------------------------------------------


def test_outlook_prior_only_new_circuit(tmp_path, monkeypatch):
    from src import outlook

    # Madring may have a sealed fit by now (the supervisor refits during the
    # weekend); the pre-practice picture is what this test is about
    out = outlook.build("spain-2026", quick=True, n_draws=120, write=False, force_prior=True)
    assert out["stage"] == "prior"
    st = out["strategy"]
    assert st["best"] and abs(sum(st["p_stops"].values()) - 1.0) < 1e-9
    life = st["life"]
    assert life["SOFT"]["deg_s_per_lap"] > life["MEDIUM"]["deg_s_per_lap"] > life["HARD"]["deg_s_per_lap"]
    assert out["programme"] and out["sc_playbook"]["ranges"]
    assert "plan_b" in out["alternatives"] or "plan_c" in out["alternatives"]


def test_outlook_live_board_folds_in(tmp_path):
    from src import outlook

    ev = get_event("spain-2026")
    base = outlook.load_base(ev, n_draws=120, force_prior=True)
    pooled = {"MEDIUM": {"slope_s_per_lap": 0.30, "se": 0.03, "n_stints": 5, "n_laps": 40},
              "SOFT": {"slope_s_per_lap": 0.005, "se": 0.02, "n_stints": 1, "n_laps": 6}}
    model, rows = outlook.fold_live_board(base, pooled, session_name="Practice 1")
    used = {r["compound"]: r["used"] for r in rows}
    assert used["MEDIUM"] and not used["SOFT"]
    before = float((base.model.wear_rate["MEDIUM"] * base.model.budget).mean())
    after = float((model.wear_rate["MEDIUM"] * model.budget).mean())
    assert after > before


def test_plan_store_roundtrip(tmp_path, monkeypatch):
    from src import plans

    monkeypatch.setattr(plans, "PLANS_DIR", tmp_path)
    p = plans.commit_plan("spain-2026", compounds=["MEDIUM", "HARD"], pit_laps=[27], push=0.55,
                          driver="NOR", n_race_laps=57, windows=[{"stop": 1, "recommended": 27, "lo": 23, "hi": 30}])
    q = plans.commit_plan("spain-2026", compounds=["SOFT", "HARD"], pit_laps=[20], push=0.7)
    got = plans.load_plans("spain-2026")
    assert {x["id"] for x in got} == {p["id"], q["id"]}
    assert plans.plan_for_driver(got, "NOR")["id"] == p["id"]
    assert plans.plan_for_driver(got, "VER")["id"] == q["id"]     # falls back to the team default
    assert p["stint_lens"] == [27, 30]
    md = plans.as_markdown(p, "Spain 2026", 57)
    assert "Stop 1: lap **27**" in md and "window 23–30" in md
    plans.remove_plan("spain-2026", p["id"])
    assert [x["id"] for x in plans.load_plans("spain-2026")] == [q["id"]]
    assert plans.next_stop(q, 12) == (0, 20) and plans.next_stop(q, 30) == (None, None)
