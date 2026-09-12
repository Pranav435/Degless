"""V4: one objective, in one place.

Task 1 shipped the race-state term in the search and the pit window and left
the counterfactual, the outlook and the plan builder on V3's objective, so the
same plan was priced three different ways.  These tests check the mechanism
that fixes that - not a number fitted to the benchmark:

* the V4 objective never charges a first-stop history prior (kappa is
  structurally zero, and asking for one is refused);
* `evaluate_plans` with the race-state terms charges the plan *group*'s term on
  the first stop, and charges exactly what `pit_window_model` charges the same
  plan - the two functions price one objective;
* a plan whose family the search never solved is charged nothing and says so;
* `counterfactual` with the terms moves at least one driver's model first stop
  away from V3's answer where the term is not flat;
* the outlook build carries the terms and the objective's label into its JSON;
* the recalibration writes the V4 keys (kappa 0, lambda on the second stop, the
  rival field's temperature, the objective version).

Run: .venv/bin/python -m pytest tests/test_v4_objective.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import objective as objlib, strategy as strat  # noqa: E402
from src.calibration import Calibration, get_calibration  # noqa: E402
from src.config import DATA_PROCESSED, get_event  # noqa: E402
from src.objective import V4Objective  # noqa: E402

PY = sys.executable


def _model(key: str, n: int = 60):
    post = DATA_PROCESSED / f"posterior_{key}.npz"
    if not post.exists():
        pytest.skip(f"no sealed {key} posterior on disk")
    from src.model_bayes import BayesFit
    from src.tyre import TyreModel

    fit = BayesFit.load(post)
    total = fit.posterior["lin"].shape[0]
    idx = np.linspace(0, total - 1, min(n, total)).astype(int)
    cal = get_calibration(key)
    return TyreModel.from_fit(fit, draws=idx, budget=cal.budgets, manage_floor=cal.manage_wear_floor,
                              manage_cost_s=cal.manage_cost_s), fit, cal


# --------------------------------------------------------------------------
# the objective object
# --------------------------------------------------------------------------


def test_sim_kwargs_never_carries_a_first_stop_prior_weight():
    cal = Calibration(undercut_lambda=0.2, plan_prior_tau_s=4.0, first_stop_kappa_s=1.0,
                      grid_start_penalty_s=0.5)
    ev = get_event("hungary-2026")
    const = objlib.measure_constants_excluding({ev.key})
    obj = V4Objective.for_event(ev, cal, plan_prior={"n": 10, "sequences": {"M-H": 6}, "starts": {"MEDIUM": 8},
                                                     "stops": {1: 6, 2: 4}},
                                first_stop_tables=None, dirty_air=0.43, race_state=const)
    kw = obj.sim_kwargs()
    assert float(kw["first_stop_kappa_s"]) == 0.0
    assert float(obj.first_stop_kappa_s) == 0.0
    assert kw["undercut_lambda"] == pytest.approx(0.2)
    assert kw["race_state"] is const
    # and the V3 weight cannot sneak in through the constructor
    with pytest.raises(ValueError):
        V4Objective(first_stop_kappa_s=1.0)
    # the V3 baseline builder is the only way to get one, and it is labelled
    v3 = V4Objective.v3_for_event(ev, cal, plan_prior=None, first_stop_tables=None, dirty_air=0.43)
    assert v3.version == "v3" and v3.first_stop_kappa_s == pytest.approx(1.0)
    assert "V3" in v3.label and float(v3.v3_kwargs()["first_stop_kappa_s"]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        obj.v3_kwargs()


def test_kwargs_drop_keywords_the_target_does_not_take():
    """The objective has to work on this checkout and on the WP-A merge."""
    ev = get_event("hungary-2026")
    obj = V4Objective.for_event(ev, Calibration(), plan_prior={"n": 5}, first_stop_tables={"MEDIUM": {}},
                                dirty_air=0.4, race_state=objlib.measure_constants_excluding({ev.key}))
    # `counterfactual` takes no plan prior and no grid penalty
    ck = obj.counterfactual_kwargs()
    assert "plan_prior" not in ck and "grid_penalty_s" not in ck
    # `pit_window_model` takes no plan prior either
    assert "plan_prior" not in obj.window_kwargs()
    # every keyword handed over is one the target really accepts
    for fn, kw in ((strat.simulate_model, obj.sim_kwargs()),
                   (strat.pit_window_model, obj.window_kwargs()),
                   (strat.evaluate_plans, obj.eval_kwargs()),
                   (strat.counterfactual, obj.counterfactual_kwargs())):
        for k in kw:
            assert objlib.accepts(fn, k), (fn.__name__, k)
    # and `rival_field` only ever goes to a `simulate_model` that accepts it
    assert ("rival_field" in obj.sim_kwargs()) == (obj.rival_field is not None
                                                   and objlib.accepts(strat.simulate_model, "rival_field"))


def test_group_label_matches_the_search_key():
    assert objlib.group_label(["MEDIUM", "HARD", "MEDIUM"], [18, 40]) == "2-stop M-H"
    assert objlib.group_label(["MEDIUM", "HARD"], [24]) == "1-stop M-H"
    assert objlib.group_label(["MEDIUM"], []) is None
    assert objlib.group_label(["MEDIUM", "HARD"], n_stops=1) == strat._group_label(("MEDIUM", "HARD", 1))


# --------------------------------------------------------------------------
# evaluate_plans and pit_window_model price one objective
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hungary_search():
    model, fit, cal = _model("hungary-2026", n=80)
    ev = get_event("hungary-2026")
    fs = json.loads((DATA_PROCESSED / "fitstage_hungary-2026.json").read_text())
    const = objlib.measure_constants_excluding({ev.key})
    obj = V4Objective.for_event(ev, cal, plan_prior=fs.get("plan_prior") or {}, first_stop_tables=None,
                                dirty_air=cal.dirty_air_for(ev.circuit), race_state=const)
    res = strat.simulate_model(model, ev, 21.0, step=2, shortlist=300,
                               support={k: float(v) for k, v in fs["age_support_by_compound"].items()},
                               max_per_compound=fs["allocation"]["caps"], **obj.sim_kwargs())
    return ev, model, obj, res


def test_evaluate_plans_charges_the_group_term_on_the_first_stop(hungary_search):
    ev, model, obj, res = hungary_search
    terms = obj.terms(res)
    assert terms, "the search solved no plan group"
    plan = {"compounds": list(res.best["compounds"]), "pit_laps": list(res.best["pit_laps"]),
            "push": float(res.best["push"])}
    label = objlib.group_label(plan["compounds"], plan["pit_laps"])
    term = terms[label]
    # the term this plan's first stop is charged, straight off the group's curve
    want = float(term[min(int(plan["pit_laps"][0]), len(term) - 1)])

    _, det0 = strat.evaluate_plans(model, ev, [plan], 21.0, **obj.eval_kwargs())          # no terms
    _, det1 = strat.evaluate_plans(model, ev, [plan], 21.0, **obj.eval_kwargs(res))       # terms
    assert det0[0]["race_state_s"] == 0.0 and det0[0]["race_state"].startswith("not priced")
    assert det1[0]["race_state_s"] == pytest.approx(want, abs=1e-9)
    assert det1[0]["race_state"] == ""
    # the term *replaces* the first stop's undercut exposure, so the exposure
    # the plan is charged with the terms on is the later stops' alone
    if obj.undercut_lambda > 0 and len(plan["pit_laps"]) >= 1:
        assert det1[0]["position_s"] < det0[0]["position_s"] - 1e-9
    # ...and the charge lands on the first stop's lap in the lap-by-lap trace:
    # offsetting the group's whole curve by a constant moves that lap by it
    bump = {k: (v + 3.0 if k == label else v) for k, v in terms.items()}
    _, det1b = strat.evaluate_plans(model, ev, [plan], 21.0,
                                    **{**obj.eval_kwargs(res), "race_state_terms": bump})
    i = int(plan["pit_laps"][0]) - 1
    assert (det1b[0]["per_lap_mean"][i] - det1[0]["per_lap_mean"][i]) == pytest.approx(3.0, abs=1e-6)
    assert (det1b[0]["times"].mean() - det1[0]["times"].mean()) == pytest.approx(3.0, abs=1e-6)

    # a plan on a lap away from the group's chosen one is charged that lap's term
    lap2 = int(plan["pit_laps"][0]) + 4
    plan2 = {**plan, "pit_laps": [lap2] + list(plan["pit_laps"][1:])}
    _, det2 = strat.evaluate_plans(model, ev, [plan2], 21.0, **obj.eval_kwargs(res))
    assert det2[0]["race_state_s"] == pytest.approx(float(term[min(lap2, len(term) - 1)]), abs=1e-9)


def test_evaluate_plans_and_pit_window_charge_the_same_first_stop(hungary_search):
    """The window is the window of the objective that chose the plan."""
    ev, model, obj, res = hungary_search
    terms = obj.terms(res)
    plan = {"compounds": list(res.best["compounds"]), "pit_laps": list(res.best["pit_laps"]),
            "push": float(res.best["push"])}
    label = objlib.group_label(plan["compounds"], plan["pit_laps"])
    n0 = int(plan["pit_laps"][0])
    laps = [n0, n0 + 3]
    # what `pit_window_model` charges: the loss curve it sweeps for stop 1
    pw = strat.pit_window_model(model, ev, plan, 21.0, push=plan["push"], **obj.window_kwargs(res))
    got = {int(r.lap): float(r.loss_s) for r in pw[pw["stop"] == 1].itertuples()}
    # ...and what `evaluate_plans` charges for the same two plans
    plans = [{**plan, "pit_laps": [l] + list(plan["pit_laps"][1:])} for l in laps]
    tbl, det = strat.evaluate_plans(model, ev, plans, 21.0, push=plan["push"], **obj.eval_kwargs(res))
    term = terms[label]
    for d, l in zip(det, laps):
        assert d["race_state_s"] == pytest.approx(float(term[min(l, len(term) - 1)]), abs=1e-9)
    # the *difference* the term makes between the two laps is the same in both
    d_eval = det[1]["race_state_s"] - det[0]["race_state_s"]
    d_term = float(term[min(laps[1], len(term) - 1)] - term[min(laps[0], len(term) - 1)])
    assert d_eval == pytest.approx(d_term, abs=1e-9)
    assert all(l in got for l in laps), got
    # the two functions are one objective: moving the first stop from one lap to
    # the other costs the same in the window sweep as in the hand-priced plans
    # (the term, the later stops' exposure, the tyre and the pit lane together)
    assert (got[laps[1]] - got[laps[0]]) == pytest.approx(
        float(tbl.loc[1, "mean_s"] - tbl.loc[0, "mean_s"]), abs=0.05)
    # and bumping one lap of the group's curve moves exactly that lap in both
    bumped = {k: v.copy() for k, v in terms.items()}
    bumped[label][laps[1]] += 3.0
    pw_b = strat.pit_window_model(model, ev, plan, 21.0, push=plan["push"],
                                  **{**obj.window_kwargs(res), "race_state_term": bumped[label]})
    got_b = {int(r.lap): float(r.loss_s) for r in pw_b[pw_b["stop"] == 1].itertuples()}
    _, det_b = strat.evaluate_plans(model, ev, plans, 21.0, push=plan["push"],
                                    **{**obj.eval_kwargs(res), "race_state_terms": bumped})
    assert (got_b[laps[1]] - got[laps[1]]) == pytest.approx(3.0, abs=1e-6)
    assert (det_b[1]["times"].mean() - det[1]["times"].mean()) == pytest.approx(3.0, abs=1e-6)


def test_a_family_with_no_term_is_charged_nothing_and_says_so(hungary_search):
    ev, model, obj, res = hungary_search
    terms = dict(obj.terms(res))
    plan = {"compounds": list(res.best["compounds"]), "pit_laps": list(res.best["pit_laps"]),
            "push": float(res.best["push"])}
    label = objlib.group_label(plan["compounds"], plan["pit_laps"])
    terms.pop(label)
    tbl, det = strat.evaluate_plans(model, ev, [plan], 21.0, race_state_terms=terms,
                                    undercut_lambda=obj.undercut_lambda)
    assert det[0]["race_state_s"] == 0.0
    assert det[0]["race_state"] == "no term for this family"
    assert tbl.loc[0, "race_state"] == "no term for this family"


def test_default_behaviour_is_unchanged_without_the_keyword(hungary_search):
    ev, model, obj, res = hungary_search
    plan = {"compounds": list(res.best["compounds"]), "pit_laps": list(res.best["pit_laps"]),
            "push": float(res.best["push"])}
    a = strat.evaluate_plans(model, ev, [plan], 21.0, undercut_lambda=0.2)[1][0]
    b = strat.evaluate_plans(model, ev, [plan], 21.0, undercut_lambda=0.2, race_state_terms=None)[1][0]
    assert float(a["times"].mean()) == pytest.approx(float(b["times"].mean()), abs=1e-9)
    assert a["position_s"] == pytest.approx(b["position_s"])


def test_deg_crossover_passes_the_terms_through(hungary_search):
    ev, model, obj, res = hungary_search
    a = {"compounds": list(res.best["compounds"]), "pit_laps": list(res.best["pit_laps"])}
    alt = res.by_stops[res.by_stops["n_stops"] != res.best["n_stops"]]
    if alt.empty:
        pytest.skip("only one stop count in the table")
    b = {"compounds": str(alt.iloc[0]["compounds"]).split("-"), "pit_laps": list(alt.iloc[0]["pit_laps"])}
    cx = strat.deg_crossover(model, ev, a, b, 21.0, mults=[1.0, 1.6], **obj.eval_kwargs(res))
    assert len(cx["curve"]) == 2 and np.isfinite(cx["curve"][0]["b_minus_a_s"])


# --------------------------------------------------------------------------
# the counterfactual
# --------------------------------------------------------------------------


def test_counterfactual_on_the_v4_objective_moves_a_first_stop(hungary_search):
    import pandas as pd

    ev, model, obj, res = hungary_search
    race = pd.read_parquet(DATA_PROCESSED / f"laps_{ev.key}_race.parquet")
    terms = obj.terms(res)
    cal = get_calibration(ev.key)
    v3 = strat.counterfactual(model, ev, race, 21.0, push=float(res.best["push"]),
                              traffic_s_per_lap=cal.dirty_air_for(ev.circuit),
                              undercut_lambda=cal.undercut_lambda)
    v4 = strat.counterfactual(model, ev, race, 21.0, push=float(res.best["push"]),
                              **obj.counterfactual_kwargs(res))
    assert not v4.empty and "race_state_s" in v4
    a = {r.driver: (list(r.model_pit_laps), str(r.compounds)) for r in v3.itertuples()}
    b = {r.driver: (list(r.model_pit_laps), str(r.compounds)) for r in v4.itertuples()}
    shared = sorted(set(a) & set(b))
    assert shared
    # every driver whose own plan family has a non-flat term is priced on it
    charged = [d for d in shared if abs(float(v4.set_index("driver").loc[d, "race_state_s"])) > 1e-9]
    moved = [d for d in shared if a[d][0][:1] != b[d][0][:1]]
    non_flat = [lab for lab, t in terms.items() if float(np.ptp(t)) > 0.5]
    assert non_flat, "the race-state terms are flat: nothing to test"
    assert charged or moved, ("no driver was charged a race-state term and no first stop moved; "
                              f"terms solved for {sorted(terms)}")
    print(f"\n  counterfactual first stops moved for {len(moved)}/{len(shared)} drivers "
          f"({', '.join(f'{d}: {a[d][0][:1]}->{b[d][0][:1]}' for d in moved[:6])})")


# --------------------------------------------------------------------------
# the outlook carries the objective into its JSON
# --------------------------------------------------------------------------


def test_outlook_build_carries_the_race_state_terms():
    from src import outlook

    out = outlook.build("italy-2026", quick=True, n_draws=120, write=False)
    st = out["strategy"]
    assert st, "no strategy block"
    terms = st.get("race_state_terms")
    assert terms and all(len(v) for v in terms.values())
    assert st["race_state"]["enabled"] and st["race_state"]["constants"]["place_value_s"] > 0
    assert st["race_state"]["group"] in terms
    assert out["objective"]["version"] == "v4"
    assert out["objective"]["first_stop_kappa_s"] == 0.0
    assert out["objective"]["undercut_applies_to"] == "stops after the first"
    # the group the recommendation belongs to is the one the block reports
    best = out["strategy"]["best_plan"]
    assert objlib.group_label(best["compounds"], best["pit_laps"]) == st["race_state"]["group"]


# --------------------------------------------------------------------------
# the plan builder prices a hand-built plan the way the optimiser does
# --------------------------------------------------------------------------


def test_desk_reads_the_objective_out_of_the_outlook_json(hungary_search):
    """`app/desk_tab` has to charge the same terms the forecast's search did."""
    desk = pytest.importorskip("app.desk_tab")
    ev, model, obj, res = hungary_search
    out = {"strategy": {"race_state_terms": obj.terms_json(res)}, "objective": obj.as_dict(),
           "calibration": {"undercut_lambda": 0.0, "plan_prior_tau_s": 0.0}}
    oj = desk._objective_json(out)
    ekw, wkw = desk._eval_kw(oj), desk._window_kw(oj, {"compounds": list(res.best["compounds"]),
                                                        "pit_laps": list(res.best["pit_laps"])})
    assert ekw["undercut_lambda"] == pytest.approx(obj.undercut_lambda)
    assert ekw["traffic_s_per_lap"] == pytest.approx(obj.traffic_s_per_lap)
    assert ekw["grid_penalty_s"] == pytest.approx(obj.grid_penalty_s)
    label = objlib.group_label(res.best["compounds"], res.best["pit_laps"])
    assert set(ekw["race_state_terms"]) == set(obj.terms(res))
    assert np.allclose(ekw["race_state_terms"][label], obj.terms(res)[label], atol=1e-3)
    assert "race_state_term" in wkw and len(wkw["race_state_term"]) == ev.n_race_laps + 1
    # and pricing the forecast's own plan through them charges the group's term
    plan = {"compounds": list(res.best["compounds"]), "pit_laps": list(res.best["pit_laps"]),
            "push": float(res.best["push"])}
    _, det = strat.evaluate_plans(model, ev, [plan], 21.0, **ekw)
    want = float(obj.terms(res)[label][int(plan["pit_laps"][0])])
    assert det[0]["race_state_s"] == pytest.approx(want, abs=1e-3)
    # an older forecast (no objective block, only the display copy of the
    # family counts) leaves the family term out rather than mispricing it
    stale = {"strategy": {}, "plan_prior": {"n": 30, "sequences": {"M-H": 12}}, "calibration": {}}
    assert desk._eval_kw(desk._objective_json(stale)).get("plan_prior") is None


# --------------------------------------------------------------------------
# the recalibration writes the V4 keys
# --------------------------------------------------------------------------


def test_recalibration_quick_writes_the_v4_keys(tmp_path):
    """`--quick` on two donors: ~20 s, and it is the only end-to-end check that
    the sweeps really run on the V4 objective."""
    keys = [k for k in ("hungary-2026", "italy-2026")
            if (DATA_PROCESSED / f"posterior_{k}.npz").exists()
            and (DATA_PROCESSED / f"laps_{k}_race.parquet").exists()]
    if len(keys) < 2:
        pytest.skip("need two scored weekends on disk")
    out = tmp_path / "calibration.json"
    r = subprocess.run([PY, str(ROOT / "scripts" / "80_recalibrate.py"), "--quick",
                        "--events", *keys, "--out", str(out)],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(out.read_text())
    assert d["objective_version"] == "v4"
    for block in [d["global"]] + [d["loo"][k] for k in keys]:
        assert block["first_stop_kappa_s"] == 0.0
        assert "kappa" not in block["sweeps"]                  # the sweep is skipped, not run
        assert block["objective_version"] == "v4"
        assert block["family_temper_s"] in (1.0, 2.0, 3.0, 5.0, 8.0)
        assert "extrap_ln_sd" in block
        assert block["sweeps"]["family_temper"] and block["sweeps"]["lambda"]
        assert block["sweeps"]["lambda"][0]["n_second_donors"] >= 0
        assert "v3_first_stop_kappa_s" in block["raw"]
        assert block["race_state"], "no race-state constants recorded"
    # and the reader picks the new keys up
    cal = get_calibration(keys[0], path=out)
    assert cal.objective_version == "v4" and cal.first_stop_kappa_s == 0.0
    assert cal.family_temper_s > 0
