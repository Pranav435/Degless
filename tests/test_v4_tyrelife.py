"""V4 WP-B: a stint planned past the practice support is priced as one.

The failure these guard against is the one V4 Task 1 shipped at Belgium: the
plan ran a SOFT for 28 laps on a 14-lap practice support, paid no cliff cost
for it (the extrapolated quotient said the tyre was good for 38.6 laps), and
the longest SOFT anyone ran in the race was 26.  The cause generalises - a
stint beyond the support is priced as if the extrapolated rate were as certain
as an interpolated one - so the tests check the *mechanism* and not Belgium's
answer:

* `extrap_ln_sd = 0` is the pre-V4 model to the last bit, whether or not the
  model carries a support;
* a compound with a long model life and a short support pays a premium that
  grows with `L / support`, pays nothing at the support itself, and pays more
  when the widening is larger;
* `life_risk` reports a probability, and it is monotone in the stint length;
* on Belgium's own posterior a 28-lap SOFT (support 14) pays a strictly
  positive premium, and more than a 28-lap HARD (support 18) does.  The
  assertion is on the premium, not on which plan the search then picks.

Run: .venv/bin/python -m pytest tests/test_v4_tyrelife.py -v
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, get_event  # noqa: E402
from src.tyre import EXTRAP_LN_SD_MEASURED, TyreModel  # noqa: E402

# Columns `life_table` adds purely to *describe* the extrapolation; they are
# expected to differ between a model that knows its support and one that does
# not, and they price nothing.
INFO_COLS = ["practice_support_laps", "life_extrap_ln_sd", "life_extrap_note"]


def _synthetic(support: dict | None = None, extrap_ln_sd: float = 0.0, n: int = 400,
               rate: dict | None = None) -> TyreModel:
    """A two-compound model whose LONGLIFE compound lasts ~34 laps at full push
    but was only run to 14 in practice, and whose SHORTLIFE compound was run to
    as long as it lasts.  Wear rate draws are lognormal about the mean so the
    posterior has the spread a real one does."""
    rng = np.random.default_rng(3)
    rate = rate or {"LONGLIFE": 0.112, "SHORTLIFE": 0.150}   # s/lap at full push
    budget = 3.8
    wear, pace = {}, {}
    for c, r in rate.items():
        draws = np.exp(rng.normal(np.log(r), 0.12, size=n))
        wear[c] = draws / budget
        pace[c] = np.zeros(n)
    return TyreModel(compounds=list(rate), wear_rate=wear, pace_offset=pace, budget=budget,
                     n_draws=n, source="synthetic", support=dict(support or {}),
                     extrap_ln_sd=float(extrap_ln_sd))


def _belgium(n: int = 500):
    """Belgium's shipped posterior, its calibration and its practice support."""
    from src.calibration import get_calibration
    from src.model_bayes import BayesFit

    post = DATA_PROCESSED / "posterior_belgium-2026.npz"
    fs = DATA_PROCESSED / "fitstage_belgium-2026.json"
    if not post.exists() or not fs.exists():
        pytest.skip("Belgium posterior / fit stage not on disk")
    ev = get_event("belgium-2026")
    cal = get_calibration("belgium-2026")
    fit = BayesFit.load(post)
    idx = np.random.default_rng(0).choice(fit.posterior["lin"].shape[0],
                                          size=min(n, fit.posterior["lin"].shape[0]), replace=False)
    support = {k: float(v) for k, v in
               json.loads(fs.read_text())["age_support_by_compound"].items()}
    kw = dict(draws=idx, budget=cal.budgets, manage_floor=cal.manage_wear_floor,
              manage_cost_s=cal.manage_cost_s)
    return ev, support, kw, fit


def _premium(model_off: TyreModel, model_on: TyreModel, ev, compound: str, L: int, start: int,
             push: float, max_len: int | None = None) -> float:
    """Mean seconds the widening adds to one (compound, length, start) stint."""
    m = int(max_len or L)
    off = model_off.cost_table(ev, m, push)[compound][:, start, L].mean()
    on = model_on.cost_table(ev, m, push)[compound][:, start, L].mean()
    return float(on - off)


# --------------------------------------------------------------------------
# (i) extrap_ln_sd = 0 is the pre-V4 model
# --------------------------------------------------------------------------


def test_zero_extrap_ln_sd_is_bit_for_bit_the_previous_cost_table():
    ev, support, kw, fit = _belgium(n=200)
    bare = TyreModel.from_fit(fit, **kw)                                    # as V3 built it
    with_support = TyreModel.from_fit(fit, support=support, extrap_ln_sd=0.0, **kw)
    for push in (0.55, 1.0):
        a = bare.cost_table(ev, 30, push, warmup_s=0.6)
        b = with_support.cost_table(ev, 30, push, warmup_s=0.6)
        for c in bare.compounds:
            assert np.array_equal(a[c], b[c]), f"{c} at push {push} is not bit for bit"
    # and the derived quantities, bar the two columns that only *describe* the
    # support the model now knows about
    la = bare.life_table(ev, 0.7).drop(columns=INFO_COLS)
    lb = with_support.life_table(ev, 0.7).drop(columns=INFO_COLS)
    assert la.equals(lb)
    for c in bare.compounds:
        assert np.array_equal(bare.life_draws(c, 0.7), bare.life_laps(c, 0.7))
        assert np.array_equal(with_support.life_draws(c, 0.7), with_support.life_laps(c, 0.7))
        assert np.array_equal(bare.wear_at(c, [24], [10], 0.7, ev),
                              with_support.wear_at(c, [24], [10], 0.7, ev))
    assert bare.extrap_multiplier("SOFT", np.arange(1, 30)) is None
    assert with_support.extrap_multiplier("SOFT", np.arange(1, 30)) is None


def test_a_model_without_a_support_never_widens():
    """A model that cannot say where its evidence ends does not get to charge
    for running past it - otherwise every V2/V3 artefact would acquire a
    premium out of a missing key."""
    ev = get_event("belgium-2026")
    no_sup = _synthetic(support=None, extrap_ln_sd=EXTRAP_LN_SD_MEASURED)
    off = _synthetic(support=None, extrap_ln_sd=0.0)
    assert no_sup.extrap_multiplier("LONGLIFE", np.arange(1, 40)) is None
    assert _premium(off, no_sup, ev, "LONGLIFE", 28, 10, 0.7) == 0.0


# --------------------------------------------------------------------------
# (ii) the mechanism: a premium that grows with L / support
# --------------------------------------------------------------------------


def test_premium_is_zero_inside_the_support_and_grows_beyond_it():
    ev = get_event("belgium-2026")
    support = {"LONGLIFE": 14.0, "SHORTLIFE": 14.0}
    off = _synthetic(support=support, extrap_ln_sd=0.0)
    on = _synthetic(support=support, extrap_ln_sd=EXTRAP_LN_SD_MEASURED)
    assert on.life_laps("LONGLIFE", 1.0).mean() > 30          # long life, 14-lap support

    prem = [_premium(off, on, ev, "LONGLIFE", L, 8, 0.85, max_len=30) for L in (10, 14, 18, 22, 26, 28)]
    assert prem[0] == 0.0 and prem[1] == 0.0, f"a stint inside the support pays {prem[:2]}"
    beyond = prem[2:]
    assert all(b > 0 for b in beyond), f"a stint past the support pays nothing: {beyond}"
    assert all(x < y for x, y in zip(beyond, beyond[1:])), f"not monotone in L/support: {beyond}"


def test_premium_grows_with_extrap_ln_sd():
    ev = get_event("belgium-2026")
    support = {"LONGLIFE": 14.0, "SHORTLIFE": 14.0}
    off = _synthetic(support=support, extrap_ln_sd=0.0)
    prems = [_premium(off, _synthetic(support=support, extrap_ln_sd=a), ev, "LONGLIFE", 28, 8, 0.85)
             for a in (0.0, 0.25 * EXTRAP_LN_SD_MEASURED, 0.5 * EXTRAP_LN_SD_MEASURED,
                       EXTRAP_LN_SD_MEASURED, 2 * EXTRAP_LN_SD_MEASURED)]
    assert prems[0] == 0.0
    assert all(x < y for x, y in zip(prems, prems[1:])), prems


def test_a_shorter_support_pays_more_for_the_same_stint():
    """The point of the mechanism: two compounds with the same rate and the same
    28-lap stint differ only in how far past the evidence that stint is."""
    ev = get_event("belgium-2026")
    rate = {"LONGLIFE": 0.112, "SHORTLIFE": 0.112}
    off = _synthetic(support={"LONGLIFE": 14.0, "SHORTLIFE": 22.0}, extrap_ln_sd=0.0, rate=rate)
    on = _synthetic(support={"LONGLIFE": 14.0, "SHORTLIFE": 22.0},
                    extrap_ln_sd=EXTRAP_LN_SD_MEASURED, rate=rate)
    thin = _premium(off, on, ev, "LONGLIFE", 28, 8, 0.85)
    thick = _premium(off, on, ev, "SHORTLIFE", 28, 8, 0.85)
    assert thin > thick > 0, (thin, thick)


def test_the_fields_survive_every_copy():
    support = {"LONGLIFE": 14.0, "SHORTLIFE": 20.0}
    m = _synthetic(support=support, extrap_ln_sd=EXTRAP_LN_SD_MEASURED)
    for name, other in (("copy_with", m.copy_with(source="x")),
                        ("subsample", m.subsample(np.arange(0, 200))),
                        ("for_driver", m.for_driver("OCO", race_factor=1.1)),
                        ("replace", replace(m, driver_dev={}))):
        assert other.support == support, name
        assert other.extrap_ln_sd == EXTRAP_LN_SD_MEASURED, name
    from src.strategy import scale_model
    scaled = scale_model(m, deg_mult=1.2)
    assert scaled.support == support and scaled.extrap_ln_sd == EXTRAP_LN_SD_MEASURED


def test_life_interval_widens_beyond_the_support():
    ev = get_event("belgium-2026")
    support = {"LONGLIFE": 14.0, "SHORTLIFE": 40.0}
    off = _synthetic(support=support, extrap_ln_sd=0.0)
    on = _synthetic(support=support, extrap_ln_sd=EXTRAP_LN_SD_MEASURED)
    a = off.life_table(ev, 0.85).set_index("compound")
    b = on.life_table(ev, 0.85).set_index("compound")
    # the extrapolated compound's interval widens; the headline life does not move
    assert b.loc["LONGLIFE", "life_lo"] < a.loc["LONGLIFE", "life_lo"]
    assert b.loc["LONGLIFE", "life_laps"] == a.loc["LONGLIFE", "life_laps"]
    assert b.loc["LONGLIFE", "life_extrap_ln_sd"] > 0
    assert "practice support" in b.loc["LONGLIFE", "life_extrap_note"]
    # the compound whose life is inside its support keeps its interval and says
    # so (to rounding: its widened life comes off the crossing grid rather than
    # the quotient, because a handful of its draws do run past 40 laps)
    assert b.loc["SHORTLIFE", "life_lo"] == pytest.approx(a.loc["SHORTLIFE", "life_lo"], rel=1e-9)
    assert b.loc["SHORTLIFE", "life_extrap_ln_sd"] == 0.0
    assert "inside the evidence" in b.loc["SHORTLIFE", "life_extrap_note"]
    # `--no-extrap` (support known, widening off) must not claim practice ran
    # the tyre this far - the interval is silent about the extrapolation, and
    # the note has to say that rather than "practice supported this length"
    assert "widening is off" in a.loc["LONGLIFE", "life_extrap_note"]


# --------------------------------------------------------------------------
# (iii) life_risk
# --------------------------------------------------------------------------


def test_life_risk_is_a_probability_and_monotone_in_the_stint_length():
    ev = get_event("belgium-2026")
    m = _synthetic(support={"LONGLIFE": 14.0, "SHORTLIFE": 14.0},
                   extrap_ln_sd=EXTRAP_LN_SD_MEASURED)
    rows = [m.life_risk("LONGLIFE", L, 6, 0.85, ev) for L in range(6, 34, 2)]
    for r in rows:
        assert set(("p_cliff", "expected_excess_s", "life_p10", "life_p50", "life_p90")) <= set(r)
        assert 0.0 <= r["p_cliff"] <= 1.0
        assert r["expected_excess_s"] >= 0.0
        assert r["life_p10"] <= r["life_p50"] <= r["life_p90"]
    p = [r["p_cliff"] for r in rows]
    assert all(x <= y for x, y in zip(p, p[1:])), p
    assert p[-1] > p[0], "a 32-lap stint must be no safer than a 6-lap one"
    x = [r["expected_excess_s"] for r in rows]
    assert all(a <= b + 1e-12 for a, b in zip(x, x[1:])), x
    # the widening is what makes the long stint riskier than the point estimate
    off = _synthetic(support={"LONGLIFE": 14.0, "SHORTLIFE": 14.0}, extrap_ln_sd=0.0)
    assert m.life_risk("LONGLIFE", 28, 6, 0.85, ev)["p_cliff"] > \
        off.life_risk("LONGLIFE", 28, 6, 0.85, ev)["p_cliff"]


# --------------------------------------------------------------------------
# (iv) the Belgium regression, on the real posterior
# --------------------------------------------------------------------------


def test_belgium_28_lap_soft_pays_for_its_extrapolation():
    """Belgium 2026: the shipped plan's second stint was a 28-lap SOFT taken
    after 16 laps, on a 14-lap SOFT practice support (2.0x - exactly the
    search's own `SUPPORT_EXTRAPOLATION_LIMIT`).  The HARD's support was 18
    laps, so the same 28-lap stint is 1.56x its evidence.  The mechanism has to
    charge the first more than the second; which plan the search then picks is
    a benchmark result, not a test.
    """
    ev, support, kw, fit = _belgium()
    assert support["SOFT"] == 14.0 and support["HARD"] == 18.0, support
    off = TyreModel.from_fit(fit, support=support, extrap_ln_sd=0.0, **kw)
    on = TyreModel.from_fit(fit, support=support, extrap_ln_sd=EXTRAP_LN_SD_MEASURED, **kw)
    # the uncapped SOFT life is what made the stint look free
    life = off.life_table(ev, 0.55).set_index("compound")
    assert life.loc["SOFT", "life_model_uncapped"] > 28

    soft = _premium(off, on, ev, "SOFT", 28, 16, 0.55, max_len=30)
    hard = _premium(off, on, ev, "HARD", 28, 16, 0.55, max_len=30)
    assert soft > 0, f"the 28-lap SOFT stint pays nothing for its extrapolation ({soft})"
    assert soft > hard, f"SOFT (support 14) {soft:.3f} s should pay more than HARD (18) {hard:.3f} s"
    # and the risk the report has to state
    r = on.life_risk("SOFT", 28, 16, 0.55, ev)
    assert r["p_cliff"] > off.life_risk("SOFT", 28, 16, 0.55, ev)["p_cliff"]
    assert r["extrap_ln_sd_at_end"] == pytest.approx(EXTRAP_LN_SD_MEASURED, rel=1e-9)


def test_belgium_pipeline_model_carries_the_support():
    """The wiring, not the mechanism: the production model is built with the
    weekend's own support, so nothing downstream has to remember to pass it."""
    from src import outlook

    ev, support, _, _ = _belgium()
    if not (DATA_PROCESSED / "meta_belgium-2026.json").exists():
        pytest.skip("meta not on disk")
    base = outlook.load_base(ev, n_draws=60)
    assert base.model.support == pytest.approx(support)
    assert base.model.extrap_ln_sd == EXTRAP_LN_SD_MEASURED
