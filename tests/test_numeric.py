"""Numeric regression tests: the model's numbers on the scored weekends.

The unit tests cover the live parser and the engine plumbing.  Nothing there
would have caught the hinge amplification that broke two of seven sealed
curves, or the compound ladder shipping with the wrong sign on every weekend.
These tests read the sealed files and race tables already on disk and assert
the numbers an analyst would notice; they skip cleanly on a machine without
the scored weekends.

Run: .venv/bin/python -m pytest tests/test_numeric.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, SEALED_DIR, get_event  # noqa: E402

SCORED = ["australia-2026", "japan-2026", "barcelona-2026", "austria-2026",
          "belgium-2026", "hungary-2026", "italy-2026"]
POOLED_RATE_MAE_MAX = 0.06     # s/lap, mean over weekends (practice-only floor is ~0.037)
PER_WEEKEND_RATE_MAE_MAX = 0.15
LADDER_TOL_S = 0.12


def _have(key: str) -> bool:
    return ((DATA_PROCESSED / f"meta_{key}.json").exists()
            and (DATA_PROCESSED / f"laps_{key}_race.parquet").exists())


def _scored():
    keys = [k for k in SCORED if _have(k)]
    if len(keys) < 3:
        pytest.skip("scored weekends not on disk")
    return keys


@pytest.fixture(scope="module")
def scores():
    import pandas as pd

    from src.laps import clean_laps
    from src.validate import load_sealed, score_race

    out = {}
    for key in _scored():
        m = json.loads((DATA_PROCESSED / f"meta_{key}.json").read_text())
        p = SEALED_DIR / m["sealed_file"]
        if not p.exists():
            continue
        sealed = load_sealed(p)
        race = pd.read_parquet(DATA_PROCESSED / f"laps_{key}_race.parquet")
        out[key] = (m, score_race(sealed, clean_laps(race), get_event(key)))
    if len(out) < 3:
        pytest.skip("sealed files not on disk")
    return out


def test_stint_rate_mae_per_weekend(scores):
    bad = {k: round(sc.mae, 3) for k, (m, sc) in scores.items() if not sc.mae < PER_WEEKEND_RATE_MAE_MAX}
    assert not bad, f"stint-rate MAE over target on {bad}"


def test_pooled_stint_rate_mae(scores):
    pooled = float(np.mean([sc.mae for _, sc in scores.values()]))
    assert pooled < POOLED_RATE_MAE_MAX, f"pooled stint-rate MAE {pooled:.3f} s/lap"


def test_no_under_coverage(scores):
    under = {k: round(sc.coverage.get(0.9, 1.0), 2) for k, (m, sc) in scores.items() if sc.coverage_direction == "under"}
    assert not under, f"90% per-lap interval under-covers on {under}"


def test_sealed_curves_are_not_amplified(scores):
    """The circuit-history fold-in may not move a practice rate by more than
    its cap: the sealed race curve at age 40 stays within a physical band."""
    for k, (m, sc) in scores.items():
        p = SEALED_DIR / m["sealed_file"]
        sealed = json.loads(p.read_text())
        for c, blk in sealed["race_curves"].items():
            loss40 = float(blk["mean"][-1])
            assert 0.0 <= loss40 <= 12.0, f"{k} {c}: sealed race curve reaches {loss40:.1f} s at age 40"


def test_ladder_gate_is_enforced():
    """The shipped model's net stint-level compound step matches the measured
    net within tolerance on every weekend - or, where the calibration was
    clipped to the physical band (a measured net that says a harder compound
    is *faster* over a stint), it at least moved toward the measurement and
    never carries the old wrong sign when the measurement is positive."""
    for key in _scored():
        m = json.loads((DATA_PROCESSED / f"meta_{key}.json").read_text())
        lc = m.get("ladder_check") or {}
        pc = m.get("pace_calibration") or {}
        model_net, meas = lc.get("model_net_step_s"), lc.get("measured_net_step_s")
        if model_net is None or meas is None or not np.isfinite(meas):
            continue
        if abs(float(model_net) - float(meas)) < LADDER_TOL_S:
            continue
        before = pc.get("model_net_before")
        assert pc.get("clipped") and before is not None and \
            abs(float(model_net) - float(meas)) < abs(float(before) - float(meas)), \
            f"{key}: model net {model_net:+.3f} vs measured {meas:+.3f} s/lap (before {before})"
        if float(meas) > 0:
            assert float(model_net) > 0, f"{key}: the model still prices a harder compound as faster over a stint"


def test_tyre_life_is_bounded_by_the_race():
    for key in _scored():
        m = json.loads((DATA_PROCESSED / f"meta_{key}.json").read_text())
        n = int(m["n_race_laps"])
        for r in (m.get("strategy") or {}).get("life", []):
            assert r["life_laps"] <= n + 1e-6, f"{key} {r['compound']}: life {r['life_laps']:.0f} > race {n}"


def test_stop_count_is_one_the_field_ran():
    for key in _scored():
        m = json.loads((DATA_PROCESSED / f"meta_{key}.json").read_text())
        bt = m.get("backtest") or {}
        if bt:
            assert bt.get("stops_observed_share", 0) > 0, f"{key}: nobody ran {bt.get('recommended_stops')} stops"
