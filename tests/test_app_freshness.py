"""The dashboard must re-read a file the supervisor rewrote underneath it.

`scripts/run.py` refits the weekend model after every practice session, so the
processed parquet/JSON change while the dashboard is open.  The loaders in
`app/dashboard.py` are `st.cache_data`-backed, and a cache keyed on the
filename alone froze every tab on whatever was on disk at first render — the
Evidence tab still showing FP1/FP2 long after FP3 had been fitted in.  These
tests pin the mtime key that fixes it.

Run: .venv/bin/python -m pytest tests/test_app_freshness.py -q
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, EVENTS  # noqa: E402

DASHBOARD = ROOT / "app" / "dashboard.py"


def _weekend_with_practice() -> str:
    """A weekend whose clean practice laps and cascade are both on disk."""
    for k in EVENTS:
        if ((DATA_PROCESSED / f"clean_{k}_practice.parquet").exists()
                and (DATA_PROCESSED / f"cascade_{k}.parquet").exists()):
            return k
    pytest.skip("no processed weekend on disk")


@pytest.fixture
def app():
    at = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(
        str(DASHBOARD), default_timeout=300)
    return at


def _evidence(at):
    return next(t for t in at.tabs if t.label == "Evidence")


def test_evidence_tab_picks_up_a_refit(app, tmp_path):
    """A session added to the clean parquet must appear without a restart."""
    key = _weekend_with_practice()
    pq = DATA_PROCESSED / f"clean_{key}_practice.parquet"
    backup = tmp_path / pq.name
    shutil.copy2(pq, backup)
    try:
        app.run()
        app.sidebar.selectbox[0].set_value(key).run()
        before = list(_evidence(app).selectbox[0].options)
        assert before, "the Evidence tab lists no practice session at all"

        # what a refit after the next practice session looks like on disk
        df = pd.read_parquet(pq)
        extra = df[df["session"] == before[-1]].copy()
        extra["session"] = "Practice 9"
        pd.concat([df, extra], ignore_index=True).to_parquet(pq, index=False)

        app.run()  # same server process: exactly like leaving the page open
        after = list(_evidence(app).selectbox[0].options)
        assert "Practice 9" in after, (
            f"stale read: Evidence still shows {after} after the parquet gained "
            f"a session — the loader cache is not keyed on the file's mtime")
    finally:
        shutil.copy2(backup, pq)


def test_missing_practice_session_is_stated(app):
    """A session the fit skipped is named on the page, not silently absent."""
    key = _weekend_with_practice()
    app.run()
    app.sidebar.selectbox[0].set_value(key).run()
    ev = _evidence(app)
    assert any("Practice fitted" in m.value for m in ev.markdown), (
        "the Evidence tab does not say which practice sessions reached the fit")


def test_loaders_are_keyed_on_mtime():
    """Every dashboard disk loader takes the file's mtime as a cache key."""
    import ast

    tree = ast.parse(DASHBOARD.read_text())
    cached = {n.name for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and any("cache_data" in ast.unparse(d) or "cache_resource" in ast.unparse(d)
                      for d in n.decorator_list)}
    readers = {n for n in cached if n.startswith("_read_")}
    assert readers, "no cached reader found in app/dashboard.py"
    for name in readers:
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        assert "mtime" in [a.arg for a in fn.args.args], (
            f"{name} is cached without an mtime key: it will serve stale data "
            f"after the supervisor rewrites the file")
