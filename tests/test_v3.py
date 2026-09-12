"""V3 regression tests: the seven checks of docs/v3_plan.md Sec.7, plus the
population invariant it appends.

V2's failures were never caught by a unit test: a Barcelona plan prior pooled
on the letters instead of the C-numbers and told the optimiser to start on a
tyre nobody used; a grip budget read off the longest stint regardless of
whether that stint ever reached the cliff; a first-stop prior that did not
exist at all. Each test here reads the real cached history or the real
calibration file and asserts the number an analyst would check by hand - not
a mock of the pipeline.

V3 is being built by several agents in parallel and this suite is written
before their work has all landed and before `make history` / `make
benchmark` have been re-run on the finished code. So every test here either

  (a) needs only code + the FastF1/history cache that is already on disk
      (nominations, the Barcelona plan prior, the regime fallback, the cliff
      detector: plan Sec.7's own list of what "must run now"), or
  (b) reads an artefact (`calibration.json`, a `fitstage_*.json`,
      `bench/out/*.json`) that only exists in its V3 shape after
      `scripts/80_recalibrate.py` / the full pipeline / the benchmark suite
      have run, and skips cleanly - never fails - when that shape is not
      there yet.

Run: .venv/bin/python -m pytest tests/test_v3.py -v
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

from src.config import DATA_PROCESSED  # noqa: E402

BENCH_OUT = ROOT / "bench" / "out"
CALIBRATION_PATH = DATA_PROCESSED / "calibration.json"

SCORED = ["australia-2026", "japan-2026", "barcelona-2026", "austria-2026",
          "belgium-2026", "hungary-2026", "italy-2026"]
IDENTICAL_NOMINATION_WEEKENDS = ["japan-2026", "austria-2026", "belgium-2026", "hungary-2026", "italy-2026"]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _calibration() -> dict | None:
    return _load_json(CALIBRATION_PATH)


# --------------------------------------------------------------------------
# 1. first-stop prior leave-one-out (plan Sec.7.1)
# --------------------------------------------------------------------------


def test_first_stop_prior_is_leave_one_out():
    """The circuit's first-stop history that fed a scored weekend's fit must
    hold only years before 2026 - a weekend's own race cannot be its own
    prior - and the recalibration's donor list for that weekend must not
    include the weekend itself either.  Skips (V2 shape) until
    `scripts/80_recalibrate.py` has written `first_stop_kappa_s`."""
    cal = _calibration()
    if not cal or "first_stop_kappa_s" not in (cal.get("global") or {}):
        pytest.skip("calibration.json has no first_stop_kappa_s yet: V3 recalibration has not run")

    checked = 0
    for key in SCORED:
        fs = _load_json(DATA_PROCESSED / f"fitstage_{key}.json")
        if not fs:
            continue
        fsg = (fs.get("circuit_history") or {}).get("first_stop_green")
        if not fsg:
            continue
        by_year = fsg.get("by_year") or fsg.get("years")
        assert by_year, f"{key}: first_stop_green carries neither by_year nor years"
        years = by_year.keys() if isinstance(by_year, dict) else by_year
        future = [y for y in years if int(y) >= 2026]
        assert not future, f"{key}: first_stop_green history includes {future}, not held out"

        donors = ((cal.get("loo") or {}).get(key) or {}).get("donors")
        if donors is not None:
            assert key not in donors, f"{key}: calibration.loo lists itself among its own donors"
        checked += 1
    if not checked:
        pytest.skip("no scored weekend has a V3 (first_stop_green-carrying) fitstage file yet")


# --------------------------------------------------------------------------
# 2. nominations.map_sequence (plan Sec.7.2)
# --------------------------------------------------------------------------


def test_map_sequence_translates_by_cnumber_and_flags_clamp():
    """Barcelona's own trap, in miniature: a sequence recorded in one year's
    roles must translate through the shared C-numbers into another year's,
    an identical nomination must be a no-op, and a compound outside the
    target's range must clamp and say so."""
    from src.nominations import map_sequence

    letters, info = map_sequence(["S", "M", "S"], ["C1", "C2", "C3"], ["C2", "C3", "C4"])
    assert letters == ["M", "H", "M"]
    assert info["clamped"] is False

    ident_letters, ident_info = map_sequence(["S", "M", "H"], ["C2", "C3", "C4"], ["C2", "C3", "C4"])
    assert ident_letters == ["S", "M", "H"]
    assert ident_info["clamped"] is False

    # 2023 Barcelona's HARD (C1) is harder than anything 2026 (C2/C3/C4) runs.
    clamp_letters, clamp_info = map_sequence(["H"], ["C1", "C2", "C3"], ["C3", "C4", "C5"])
    assert clamp_letters == ["H"]
    assert clamp_info["clamped"] is True


# --------------------------------------------------------------------------
# 3. Barcelona plan prior: the V2 defect, pinned (plan Sec.7.3)
# --------------------------------------------------------------------------


def test_barcelona_plan_prior_pins_the_v2_defect():
    """Pooled on the letters (V2), Barcelona's history says the field starts
    on the SOFT - which is 2026's MEDIUM, one step softer than anything those
    races used.  Mapped through the C-numbers, MEDIUM is the majority start
    and M-H-H (not any S-...) is the modal 2-stop family.  The five weekends
    whose nomination has not moved since 2023-2025 must be unaffected either
    way."""
    pytest.importorskip("fastf1")
    import fastf1

    fastf1.Cache.enable_cache(str(ROOT / "data" / "raw" / "fastf1_cache"))
    fastf1.Cache.offline_mode(True)
    import logging
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    from src.history import circuit_prior, plan_prior_for

    cp = circuit_prior("barcelona-2026", probe_practice_temp=False)
    if not cp.available:
        pytest.skip("no cached Barcelona history on disk")

    mapped = plan_prior_for(cp)
    two_stop = {k: v for k, v in mapped["sequences"].items() if k.count("-") == 2}
    assert two_stop, "no 2-stop (3-stint) family in the mapped plan prior"
    modal_family = max(two_stop, key=two_stop.get)
    assert modal_family == "M-H-H", f"modal 2-stop family is {modal_family}, not M-H-H"
    assert mapped["starts"]["MEDIUM"] > mapped["starts"].get("SOFT", 0)

    letters_only = plan_prior_for(cp, use_nominations=False)
    assert letters_only["starts"]["SOFT"] > letters_only["starts"]["MEDIUM"]

    for key in IDENTICAL_NOMINATION_WEEKENDS:
        cpk = circuit_prior(key, probe_practice_temp=False)
        if not cpk.available:
            continue
        with_nom = plan_prior_for(cpk)
        without_nom = plan_prior_for(cpk, use_nominations=False)
        assert with_nom["sequences"] == without_nom["sequences"], f"{key}: nominations moved the sequence counts"
        assert with_nom["starts"] == without_nom["starts"], f"{key}: nominations moved the start counts"


# --------------------------------------------------------------------------
# 4. regime_prior: forecast in, archive out (plan Sec.7.4)
# --------------------------------------------------------------------------


def test_regime_prior_temperature_modes():
    """With no race-day forecast the thermal correction must not apply at
    all (`mode == "none"`, no `delta_t_c` effect) rather than silently
    falling back to the archive's race-day mean, which is a prediction
    dressed as a measurement.  A real forecast switches it on and moves the
    ratio; `temperature_model=True` restores the V2 archive behaviour for
    the ablation."""
    from src.config import EVENTS
    from src.regime import regime_prior

    donors = [k for k in EVENTS if k != "belgium-2026" and (DATA_PROCESSED / f"laps_{k}_race.parquet").exists()]
    if len(donors) < 2:
        pytest.skip("fewer than two donor races on disk for belgium-2026")

    no_forecast = regime_prior("belgium-2026")
    assert no_forecast.temperature["mode"] == "none"
    assert no_forecast.temperature.get("delta_t_c") == 0.0

    forecast = regime_prior("belgium-2026", race_temp_c=31.0)
    assert forecast.temperature["mode"] == "forecast"
    assert forecast.ratio != no_forecast.ratio

    archive = regime_prior("belgium-2026", temperature_model=True, use_circuit_history=False)
    assert archive.temperature["mode"] == "archive"


# --------------------------------------------------------------------------
# 5. dirty air, per circuit (plan Sec.7.5)
# --------------------------------------------------------------------------


def test_dirty_air_by_circuit():
    """Running in traffic costs +0.43 s/lap at Budapest and buys back -0.20 s
    (the tow) at Monza; a single pooled constant prices an extra stop at
    Monza as if it cost the Hungarian penalty.  `Calibration.dirty_air_for`
    must return the circuit's own value and fall back to the pooled one for
    a circuit with no history of its own."""
    cal_json = _calibration()
    dabc = ((cal_json or {}).get("global") or {}).get("dirty_air_by_circuit")
    if not dabc:
        pytest.skip("calibration.json has no dirty_air_by_circuit yet: V3 recalibration has not run")
    assert "Budapest" in dabc and "Monza" in dabc, f"missing circuits in dirty_air_by_circuit: {sorted(dabc)}"
    assert float(dabc["Budapest"]) > float(dabc["Monza"])

    from src.calibration import get_calibration

    cal = get_calibration()
    assert hasattr(cal, "dirty_air_for"), "Calibration has no dirty_air_for method yet"
    assert cal.dirty_air_for("Budapest") == pytest.approx(float(dabc["Budapest"]))
    assert cal.dirty_air_for("a circuit with no history") == pytest.approx(cal.dirty_air_s_per_lap)


# --------------------------------------------------------------------------
# 6. within-stint cliff detector + censored grip budget (plan Sec.7.6)
# --------------------------------------------------------------------------


def _synthetic_stint(n_laps: int, *, trend: float = 0.1, knee: float = 15.0,
                     extra: float = 0.6, base: float = 90.0) -> pd.DataFrame:
    """Fuel-corrected lap times: a `trend` s/lap line with `extra` s/lap of
    additional slope from `knee` onward.  `fuel_s_per_lap=0.0` is passed to
    `detect_stint_collapse` so these numbers are used exactly as given."""
    age = np.arange(1, n_laps + 1, dtype=float)
    y = base + trend * age + np.where(age > knee, extra * (age - knee), 0.0)
    return pd.DataFrame({"lap_number": age, "tyre_age": age, "lap_time_s": y})


def test_cliff_detector_and_grip_budget_floor():
    """A stint that breaks away from its own trend and boxes soon after is a
    collapse; the same profile stopped before it ever breaks away is
    strategic.  And the estimator's floor is the largest *right-censored
    bound*, clipped into the budget band - not the largest raw observation,
    and not whatever a small handful of low collapse observations would pull
    the censored MLE toward on their own."""
    from src.cliff import BUDGET_BAND_S, detect_stint_collapse, grip_budget_estimate

    collapsing = _synthetic_stint(17)
    res = detect_stint_collapse(collapsing, fuel_s_per_lap=0.0)
    assert res["kind"] == "collapse"
    assert abs(res["knee_age"] - 15.0) <= 1.0

    strategic_stint = _synthetic_stint(17).iloc[:12].reset_index(drop=True)
    res_strategic = detect_stint_collapse(strategic_stint, fuel_s_per_lap=0.0)
    assert res_strategic["kind"] == "strategic"

    # Three low collapse observations (~2.6 s) would pull an unfloored MLE
    # toward them; a 4.9 s lower bound - a stint that gave up 4.9 s and was
    # still on its trend - must floor the estimate at 4.9, not below it.
    rows = pd.DataFrame([
        {"collapse": True, "kind": "collapse", "cum_loss_at_knee_s": 2.60, "cum_loss_at_end_s": 2.60},
        {"collapse": True, "kind": "collapse", "cum_loss_at_knee_s": 2.55, "cum_loss_at_end_s": 2.55},
        {"collapse": True, "kind": "collapse", "cum_loss_at_knee_s": 2.65, "cum_loss_at_end_s": 2.65},
        {"collapse": False, "kind": "strategic", "cum_loss_at_knee_s": np.nan, "cum_loss_at_end_s": 4.90},
    ])
    gb = grip_budget_estimate(rows)
    floor = min(gb["largest_bound_s"], BUDGET_BAND_S[1])
    assert gb["budget_s"] >= floor - 1e-9
    assert gb["budget_s"] == pytest.approx(4.90)

    # A bound outside the band clips into it rather than dragging the
    # estimate past the band's own ceiling.
    rows_over_band = pd.DataFrame([
        {"collapse": True, "kind": "collapse", "cum_loss_at_knee_s": 2.6, "cum_loss_at_end_s": 2.6},
        {"collapse": False, "kind": "strategic", "cum_loss_at_knee_s": np.nan, "cum_loss_at_end_s": 6.0},
    ])
    gb2 = grip_budget_estimate(rows_over_band)
    assert gb2["budget_s"] == pytest.approx(BUDGET_BAND_S[1])


# --------------------------------------------------------------------------
# 7. reproducibility: bench/out vs meta_*.json (plan Sec.7.7)
# --------------------------------------------------------------------------


def test_bench_output_reproduces_meta():
    """The benchmark's own recorded numbers must still match what the
    pipeline sealed: `bench/out/strategy.json`'s first-stop recommendation is
    `meta_*.json`'s `strategy.best_plan.pit_laps[0]` and its accuracy MAE is
    the same score the pipeline computed scoring the sealed file against the
    race, to floating-point precision.  Skips if the benchmark output is
    missing or predates the meta files it should have been built from."""
    strat_path = BENCH_OUT / "strategy.json"
    acc_path = BENCH_OUT / "accuracy.json"
    strat = _load_json(strat_path)
    acc = _load_json(acc_path)
    if not strat or not acc:
        pytest.skip("bench/out/strategy.json or accuracy.json missing")

    meta_files = {k: DATA_PROCESSED / f"meta_{k}.json" for k in SCORED}
    meta_files = {k: p for k, p in meta_files.items() if p.exists()}
    if not meta_files:
        pytest.skip("no meta_*.json on disk")

    newest_meta = max(p.stat().st_mtime for p in meta_files.values())
    if strat_path.stat().st_mtime < newest_meta or acc_path.stat().st_mtime < newest_meta:
        pytest.skip("bench/out predates the meta files: stale benchmark output")

    per_event = acc.get("per_event", {})
    checked = 0
    for key, p in meta_files.items():
        if key not in strat or key not in per_event:
            continue
        m = _load_json(p)
        pit_laps = ((m.get("strategy") or {}).get("best_plan") or {}).get("pit_laps") or []
        want_first_stop = int(pit_laps[0]) if pit_laps else None
        got_first_stop = strat[key]["first_stop"]["recommended"]
        assert got_first_stop == want_first_stop, (
            f"{key}: bench/out/strategy.json first_stop.recommended {got_first_stop} "
            f"!= meta strategy.best_plan.pit_laps[0] {want_first_stop}")

        # bench_accuracy.py records the sealed *file*'s own re-scored MAE as
        # `sealed_file_mae`; fall back to the freshly-simulated "sealed"
        # variant's score if some future version renames it.
        got_mae = per_event[key].get("sealed_file_mae")
        if got_mae is None:
            got_mae = (per_event[key].get("variants", {}).get("sealed", {}) or {}).get("scorer_mae")
        assert got_mae is not None, (
            f"{key}: accuracy.json has neither sealed_file_mae nor variants.sealed.scorer_mae")
        assert got_mae == pytest.approx(m["score"]["mae"], abs=1e-6), (
            f"{key}: accuracy.json sealed MAE {got_mae} != meta score.mae {m['score']['mae']}")
        checked += 1
    if not checked:
        pytest.skip("no weekend common to bench/out and the meta files")


# --------------------------------------------------------------------------
# Population: the scored weekends' rate-stint count must not move
# --------------------------------------------------------------------------


def test_population_n_rate_stints_sums_to_196():
    """V3 changes how stints are *priced*, not which ones exist: the same
    green, classified, long-enough-to-score stints must still be there.  If
    this sum moves, something upstream started dropping or admitting stints,
    not just weighing them differently."""
    total, found = 0, 0
    for key in SCORED:
        m = _load_json(DATA_PROCESSED / f"meta_{key}.json")
        if not m:
            continue
        total += int(m["score"]["n_rate_stints"])
        found += 1
    if found < len(SCORED):
        pytest.skip(f"only {found}/{len(SCORED)} scored weekends on disk")
    assert total == 196, f"n_rate_stints sums to {total} over {found} weekends, expected 196"
