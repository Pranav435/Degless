"""V4 WP-A: the heterogeneous rival field and the place-value estimator.

Task 1's pack was four copies of our own car on our own plan, and the two
things that cost it were both structural: a field of clones has no car that
moves first (Barcelona's soft-starters), and the value of a place was a pooled
ratio over 26 pit-cycle pairs with no statement of how well it was known.  So
these tests check the mechanisms, never a benchmark number:

* a field of one type is the symmetric pack (the tensor algebra is right);
* a SOFT-start type boxes before a MEDIUM-start one on the same tyre model;
* the family temperature is what sorts the field onto the cheaper plans;
* switching the rivals' history off moves the rivals' stop laps and nothing
  about which families they run;
* `mode="symmetric"` is Task 1 exactly, and the `task1` estimator reproduces
  Task 1's shipped constants;
* the regularised persistence lies between the pooled ratio and its prior, and
  its interval covers the point estimate;
* a set exclusion is a single-key exclusion when the set has one key.

Run: .venv/bin/python -m pytest tests/test_v4_rivals.py -v
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import racestate  # noqa: E402
from src.config import DATA_PROCESSED  # noqa: E402

FROZEN_META = ROOT / "bench" / "v4_task1" / "processed" / "meta_hungary-2026.json"
N_LAPS = 50


def _const(**kw):
    base = racestate.RaceStateConstants(place_gap_s=4.5, persistence=0.75, cycle_sd_s=1.8,
                                        pack_gaps_s=tuple(np.linspace(0.3, 3.5, 200)), source="test")
    return base if not kw else __import__("dataclasses").replace(base, **kw)


def _tables(rate: float, pace: float, n: int = N_LAPS):
    """Cumulative stint cost of one compound: linear wear on a flat pace offset.

    `stay[k]` is what `k` laps on the start set cost and `fresh[s, j]` what `j`
    laps on the next set cost - the shape `pack_equilibrium` and `rival_field`
    take, built so that a faster-degrading compound's cost is convex in the
    stint length rather than asserted to peak anywhere."""
    k = np.arange(n + 1, dtype=float)
    stay = pace * k + rate * k * (k + 1) / 2.0
    fresh = np.broadcast_to(0.7 + pace * k + rate * k * (k + 1) / 2.0, (n + 1, n + 1)).copy()
    fresh[:, 0] = 0.0
    return stay, fresh


def _one_stop_curve(stay, fresh, laps, *, pit_loss_s: float = 22.0, n: int = N_LAPS):
    """A one-stop plan's race-time cost by first-stop lap, from its own tables."""
    return np.array([stay[l] + fresh[l, n - l] + pit_loss_s for l in laps], dtype=float)


def _type(label, stay, fresh, laps, grid, *, weight=1.0, hist=None, hist_w=0.0, ours=True,
          level=0, rate_factor=1.0):
    T = np.full(len(grid), np.inf)
    idx = {int(l): i for i, l in enumerate(grid)}
    T[[idx[int(l)] for l in laps]] = _one_stop_curve(stay, fresh, laps)
    return {"label": label, "T": T, "stay_cum": stay, "fresh_cum": fresh, "weight": weight,
            "hist": hist, "hist_w": hist_w, "ours": ours, "level": level,
            "rate_factor": rate_factor, "q_family": weight}


# --------------------------------------------------------------------------
# the tensor algebra: a field of one type is the symmetric pack
# --------------------------------------------------------------------------


def test_a_field_of_one_type_is_the_symmetric_pack():
    """Same game, two implementations: the mean-field tensor must reproduce
    `pack_equilibrium` exactly when the field is a copy of us.

    With the cover response on the two differ by construction and by a stated
    amount: `pack_equilibrium` decides the cover per gap scenario, while the
    field decides it on the place a rival expects to save averaged over the
    pack's gaps (there is no tensor that keeps 32 gap scenarios across every
    type pair inside the search's time budget).  The test pins that gap small
    rather than pretending it is zero."""
    grid = np.arange(6, 40)
    stay, fresh = _tables(0.05, 0.0)
    T = _one_stop_curve(stay, fresh, grid)
    const = _const()
    for cover, tol in ((False, 1e-4), (True, 0.1)):
        sym = racestate.pack_equilibrium(T, grid, stay, fresh, const, cover=cover)
        out = racestate.rival_field([_type("g", stay, fresh, grid, grid)], grid, const,
                                    racestate.RivalFieldConfig(), cover=cover)
        het = out["packs"]["g"]
        assert het["laps"] == sym["laps"]
        assert np.abs(np.asarray(het["places"]) - np.asarray(sym["places"])).max() < tol
        assert abs(het["best_lap"] - sym["best_lap"]) <= (0 if not cover else 1)


def test_a_worthless_place_leaves_the_tyre_to_decide_in_the_field_too():
    grid = np.arange(6, 40)
    stay, fresh = _tables(0.05, 0.0)
    out = racestate.rival_field([_type("g", stay, fresh, grid, grid)], grid,
                                _const(persistence=0.5), racestate.RivalFieldConfig())
    pk = out["packs"]["g"]
    assert pk["best_lap"] == pk["tyre_best_lap"]
    assert np.allclose(pk["term_s"], 0.0)


# --------------------------------------------------------------------------
# heterogeneity: who moves first
# --------------------------------------------------------------------------


def test_a_soft_start_type_boxes_before_a_medium_start_type():
    """The mechanism the symmetric pack could not have: a car on a softer start
    set reaches its own optimum earlier, so it is the one that moves first and
    the one the rest of the field has to answer."""
    grid = np.arange(4, 40)
    soft = _tables(0.10, -0.35)          # wears faster, quicker while it lasts
    med = _tables(0.05, 0.0)
    hard, _ = _tables(0.03, 0.35)
    fresh_hard = _tables(0.03, 0.35)[1]
    types = [_type("1-stop S-H", soft[0], fresh_hard, grid, grid, weight=0.5),
             _type("1-stop M-H", med[0], fresh_hard, grid, grid, weight=0.5)]
    out = racestate.rival_field(types, grid, _const(), racestate.RivalFieldConfig())
    tbl = {r["family"]: r for r in out["types"]}
    assert tbl["1-stop S-H"]["stop_lap_median"] < tbl["1-stop M-H"]["stop_lap_median"]
    assert out["packs"]["1-stop S-H"]["best_lap"] < out["packs"]["1-stop M-H"]["best_lap"]


def test_a_hotter_family_temperature_flattens_the_field():
    """`family_temper_s` is the only thing that decides how sharply the field
    sorts onto the cheaper plan families, so it has to be visible as exactly
    that: cold, everybody runs the best plan; hot, the field spreads."""
    cost = [0.0, 3.0, 6.0, 12.0]
    cold = racestate.family_weights(cost, None, n_prior=0,
                                    cfg=racestate.RivalFieldConfig(family_temper_s=1.0))
    warm = racestate.family_weights(cost, None, n_prior=0,
                                    cfg=racestate.RivalFieldConfig(family_temper_s=3.0))
    hot = racestate.family_weights(cost, None, n_prior=0,
                                   cfg=racestate.RivalFieldConfig(family_temper_s=10.0))
    assert cold.max() > warm.max() > hot.max()
    assert cold.min() < warm.min() < hot.min()
    for q in (cold, warm, hot):
        assert q.sum() == pytest.approx(1.0)
        assert np.all(np.diff(q) < 0)                 # the cheaper family is always likelier


def test_the_plan_prior_moves_the_field_toward_what_the_circuit_has_run():
    """History enters the field as *what rivals plausibly run*, at the weight
    its own sample earns: a family the circuit has never seen loses weight, and
    with no history at all (or none allowed) the costs decide alone."""
    cost = [0.0, 1.0]
    nlp = [3.0, 0.0]                                  # the cheaper family is the rarer one
    cfg = racestate.RivalFieldConfig()
    none = racestate.family_weights(cost, nlp, n_prior=0, cfg=cfg)
    thin = racestate.family_weights(cost, nlp, n_prior=2, cfg=cfg)
    thick = racestate.family_weights(cost, nlp, n_prior=90, cfg=cfg)
    off = racestate.family_weights(cost, nlp, n_prior=90,
                                   cfg=racestate.RivalFieldConfig(family_prior_weight_k0=float("inf")))
    assert none[0] > thin[0] > thick[0]               # more history, less weight on the rare family
    assert off == pytest.approx(none)                 # the weight, not the history, is switched off
    assert thick[1] > thick[0]                        # enough history overturns a 1 s cost edge


def test_switching_the_rivals_history_off_moves_their_stop_laps_only():
    """`use_history_prior=False` is the ablation that asks whether the field's
    early stop is the model's cost surface or the circuit's habit.  It must
    move the rivals' stop-lap distribution and nothing else about the field -
    not which families they run, not our own tyre cost."""
    grid = np.arange(4, 40)
    stay, fresh = _tables(0.05, 0.0)
    # a circuit whose field has always boxed within a lap or two of 12: the
    # `-log p` of a Gaussian density with sd 1.5 laps about lap 12
    early = (grid - 12.0) ** 2 / (2 * 1.5 ** 2)
    types = [_type("1-stop M-H", stay, fresh, grid, grid, weight=1.0, hist=early, hist_w=0.6)]
    on = racestate.rival_field(types, grid, _const(), racestate.RivalFieldConfig())
    off = racestate.rival_field(types, grid, _const(),
                                racestate.RivalFieldConfig(use_history_prior=False))
    # `hist_w` is the caller's (the weight `history_weight` hands it), so the
    # switch is checked where it lives as well as end to end
    assert racestate.history_weight(20, racestate.RivalFieldConfig()) > 0
    assert racestate.history_weight(20, racestate.RivalFieldConfig(use_history_prior=False)) == 0.0
    a, b = on["types"][0], off["types"][0]
    assert a["weight"] == b["weight"] and a["q_family"] == b["q_family"]
    assert on["packs"]["1-stop M-H"]["tyre_s"] == off["packs"]["1-stop M-H"]["tyre_s"]
    types_off = [dict(t, hist_w=0.0) for t in types]
    same = racestate.rival_field(types_off, grid, _const(), racestate.RivalFieldConfig())
    assert same["types"][0]["stop_lap_median"] == b["stop_lap_median"]
    assert a["stop_lap_median"] < b["stop_lap_median"]


# --------------------------------------------------------------------------
# the search: symmetric mode is Task 1, and the keyword goes everywhere
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
    return ev, cal, TyreModel.from_fit(f, draws=idx, budget=cal.budgets,
                                       manage_floor=cal.manage_wear_floor,
                                       manage_cost_s=cal.manage_cost_s)


def test_symmetric_mode_is_the_task_1_pack_equilibrium():
    """The ablations and the frozen Task 1 run depend on `mode="symmetric"`
    being the old code, so the shipped pack is re-solved here from the group's
    own curve and has to come back identical."""
    from src import strategy as strat

    if not (DATA_PROCESSED / "laps_hungary-2026_race.parquet").exists():
        pytest.skip("race lap tables not on disk")
    ev, cal, model = _small_model()
    const = racestate.measure_constants(exclude=ev.key, estimator="task1")
    kw = dict(step=2, shortlist=600, undercut_lambda=cal.undercut_lambda,
              max_per_compound={"SOFT": 2, "MEDIUM": 2, "HARD": 2})
    res = strat.simulate_model(model, ev, 22.0, race_state=const,
                               rival_field=racestate.RivalFieldConfig(mode="symmetric"), **kw)
    assert res.race_state["mode"] == "symmetric"
    pack = res.race_state["best"]
    max_len = min(ev.n_race_laps, max(res.max_stint.values()))
    means = {c: v.mean(0) for c, v in strat.stint_cost_table(model, ev, max_len,
                                                             float(pack["push"])).items()}
    seq = res.best["compounds"]
    again = racestate.pack_equilibrium(np.asarray(pack["tyre_s"]), np.asarray(pack["laps"]),
                                       means[seq[0]][0], means[seq[1]], const)
    assert again["best_lap"] == pack["best_lap"]
    assert again["iterations"] == pack["iterations"]
    assert np.allclose(again["term_s"], pack["term_s"])


def test_the_heterogeneous_field_is_the_default_and_reports_its_types():
    from src import strategy as strat

    if not (DATA_PROCESSED / "laps_hungary-2026_race.parquet").exists():
        pytest.skip("race lap tables not on disk")
    ev, cal, model = _small_model()
    const = racestate.measure_constants(exclude=ev.key)
    kw = dict(step=2, shortlist=600, undercut_lambda=cal.undercut_lambda,
              max_per_compound={"SOFT": 2, "MEDIUM": 2, "HARD": 2})
    res = strat.simulate_model(model, ev, 22.0, race_state=const, **kw)
    rs = res.race_state
    assert rs["mode"] == "hetero"
    fld = rs["rival_field"]
    assert fld["converged"] and fld["n_types"] >= 3
    assert fld["rate_ln_sd"] > 0
    levels = {r["rate_level"] for r in fld["types"]}
    assert levels == {-1, 0, 1}                        # three degradation levels, ours in the middle
    assert sum(r["weight"] for r in fld["types"]) == pytest.approx(1.0, abs=1e-3)   # 4 dp, reported
    ours = [r for r in fld["types"] if r["ours"]]
    assert ours and all(r["rate_factor"] == 1.0 for r in ours)
    # our own group still pays nothing at the lap it chose, as in Task 1
    assert res.best["pit_laps"][0] == rs["best"]["best_lap"]
    assert abs(res.best["race_state_s"]) < 1e-6
    # and with no race state at all this is still V3
    v3 = strat.simulate_model(model, ev, 22.0, **kw)
    assert v3.race_state == {} and "race_state_s" not in v3.best


def test_the_rival_field_keyword_reaches_every_search_entry_point():
    """`per_driver_plans` and `search_with_pace_calibration` hand their extra
    keywords to `simulate_model`; this pins that they still do, because a
    silently dropped `rival_field` would leave the per-car plans on a different
    rival field from the field plan's."""
    from src import strategy as strat

    assert "rival_field" in inspect.signature(strat.simulate_model).parameters
    for fn in (strat.simulate, strat.search_with_pace_calibration, strat.per_driver_plans):
        params = inspect.signature(fn).parameters
        assert "rival_field" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), fn.__name__


# --------------------------------------------------------------------------
# the place-value estimator
# --------------------------------------------------------------------------


def test_set_exclusion_equals_single_key_exclusion():
    if not (DATA_PROCESSED / "laps_hungary-2026_race.parquet").exists():
        pytest.skip("race lap tables not on disk")
    one = racestate.measure_constants(exclude="hungary-2026")
    for other in ({"hungary-2026"}, ["hungary-2026"], ("hungary-2026",), iter(["hungary-2026"])):
        assert racestate.measure_constants(exclude=other) == one
    two = racestate.measure_constants(exclude=["hungary-2026", "italy-2026"])
    assert "italy-2026" not in two.donors and "hungary-2026" not in two.donors
    assert racestate.measure_constants(exclude=None).donors == racestate.measure_constants().donors


def test_the_regularised_persistence_sits_between_the_ratio_and_its_prior():
    if not (DATA_PROCESSED / "laps_hungary-2026_race.parquet").exists():
        pytest.skip("race lap tables not on disk")
    c = racestate.measure_constants(exclude="hungary-2026")
    assert c.estimator == "regularized"
    lo, hi = sorted((c.persistence_raw, _prior_mean(c)))
    assert lo <= c.persistence <= hi
    assert 0.5 < c.persistence <= 1.0
    assert c.place_value_sd_s > 0
    a, b = c.place_value_ci_s
    assert a <= c.place_value_s <= b
    assert sum(c.n_cycle_pairs_by_donor.values()) == c.n_cycle_pairs
    assert sum(v["kept"] for v in c.persistence_by_donor.values()) / max(c.n_cycle_pairs, 1) \
        == pytest.approx(c.persistence_raw, abs=1e-9)


def _prior_mean(c) -> float:
    pairs = np.array([v["pairs"] for v in c.persistence_by_donor.values()], dtype=float)
    kept = np.array([v["kept"] for v in c.persistence_by_donor.values()], dtype=float)
    return float(racestate.persistence_eb(pairs, kept)["prior_mean"])


def test_shrinkage_is_strongest_where_the_pooled_sample_is_thinnest():
    """The estimator's reason for existing.  The pooled ratio weights a race by
    its pair count, which is only right if every race has the same true rate;
    the shrinkage moves it toward the mean over *races*, by an amount that
    shrinks away as the pooled sample grows."""
    pairs, kept = [3.0, 30.0], [3.0, 21.0]
    thin = racestate.persistence_eb(pairs, kept)
    thick = racestate.persistence_eb(np.array(pairs) * 10, np.array(kept) * 10)
    lo, hi = sorted((float(thin["raw"]), float(thin["prior_mean"])))
    assert lo < float(thin["psi"]) < hi                  # strictly between ratio and prior
    assert float(thin["prior_pairs"]) > 0
    assert abs(thin["psi"] - thin["raw"]) > abs(thick["psi"] - thick["raw"])
    # no spread between the races at all: the Jeffreys prior takes over
    flat = racestate.persistence_eb([10.0, 10.0], [8.0, 8.0])
    assert flat["prior_mean"] == pytest.approx(racestate.JEFFREYS_MEAN)
    assert flat["psi"] < flat["raw"]                     # pulled toward "a place is worth nothing"
    assert flat["psi"] == pytest.approx((16.0 + 0.5) / (20.0 + 1.0))
    assert str(flat["method"]).startswith("Jeffreys")


def test_the_task1_estimator_reproduces_the_shipped_task1_constants():
    if not FROZEN_META.exists():
        pytest.skip("the frozen Task 1 meta is not in this checkout")
    if not (DATA_PROCESSED / "laps_hungary-2026_race.parquet").exists():
        pytest.skip("race lap tables not on disk")
    want = json.loads(FROZEN_META.read_text())["race_state"]["constants"]
    got = racestate.measure_constants(exclude="hungary-2026", estimator="task1").as_dict()
    for k in ("place_gap_s", "place_gap_lead_lap_s", "persistence", "place_value_s", "cycle_sd_s",
              "sigma_rel_s", "pack_gap_median_s"):
        assert got[k] == pytest.approx(want[k], abs=5e-4), k
    for k in ("n_finish_gaps", "n_cycle_pairs", "n_pit_stops", "n_pack_gaps"):
        assert got[k] == want[k], k
    assert got["donors"] == want["donors"]


def test_the_estimators_differ_only_where_they_should():
    if not (DATA_PROCESSED / "laps_hungary-2026_race.parquet").exists():
        pytest.skip("race lap tables not on disk")
    t1 = racestate.measure_constants(exclude="hungary-2026", estimator="task1")
    allc = racestate.measure_constants(exclude="hungary-2026", estimator="all_classified")
    lead = racestate.measure_constants(exclude="hungary-2026", estimator="lead_lap")
    reg = racestate.measure_constants(exclude="hungary-2026", estimator="regularized")
    assert allc.place_value_s == t1.place_value_s        # the same estimator, named twice
    assert lead.place_gap_s == pytest.approx(t1.place_gap_lead_lap_s)
    assert lead.place_gap_s < t1.place_gap_s             # lead-lap gaps are a front-runner sample
    assert reg.place_gap_s == t1.place_gap_s             # only psi is regularised
    assert reg.persistence != t1.persistence
    assert np.isfinite(t1.place_gap_midfield_s)
    with pytest.raises(ValueError):
        racestate.measure_constants(estimator="whatever-scores-best")
