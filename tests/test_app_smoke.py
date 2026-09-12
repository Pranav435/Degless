"""The dashboard renders for every kind of weekend without an exception, and
none of the statistics jargon the redesign removed shows up on screen."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

import src.config as config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "dashboard.py")
BANNED = re.compile(
    r"posterior|credible|\bMAE\b|r_hat|\brhat\b|sha256|regime|grip budget|\bdraws\b|EVPI|regret|isotonic|k_track|"
    r"sigma_obs|outlook|supervisor|snapshot|counterfactual|MixedLM|divergence|\bpush\b|\bprior\b|donor|cliff|"
    r"allocation|\bdeg\b",
    re.I,
)


def _texts(node, out: list) -> list:
    proto = getattr(node, "proto", None)
    if proto is not None:
        out.append(str(proto))
    for child in (getattr(node, "children", None) or {}).values():
        _texts(child, out)
    return out


def _render(event: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.sidebar.selectbox[0].set_value(event).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _jargon(at: AppTest) -> list:
    hits = set()
    for text in _texts(at._tree, []):
        # Element ids, encoded chart data, chart-spec field names and literal
        # commands (<code>make outlook</code>) are not wording the reader has to parse.
        text = re.sub(r'\bid: "[^"]*"', "", text)
        text = re.sub(r"(?:[A-Za-z0-9+/=]|\\\\u002f){40,}", "", text)
        text = re.sub(r"<code>.*?</code>", "", text)
        text = re.sub(r'"(customdata|meta|uid)"', "", text)
        hits.update(m.group(0).lower() for m in BANNED.finditer(text))
    return sorted(hits)


@pytest.mark.parametrize("event", ["hungary-2026", "italy-2026", "barcelona-2026", "spain-2026"])
def test_weekend_renders_without_jargon(event):
    if not (config.DATA_PROCESSED / f"meta_{event}.json").exists() and \
            not (config.DATA_PROCESSED / f"outlook_{event}.json").exists():
        pytest.skip(f"no processed data for {event}")
    at = _render(event)
    assert at.tabs, "tabs missing"
    assert _jargon(at) == []


@pytest.mark.parametrize("event", ["hungary-2026", "italy-2026"])
def test_haas_tab_renders(event):
    if not (config.DATA_PROCESSED / f"meta_{event}.json").exists() and \
            not (config.DATA_PROCESSED / f"outlook_{event}.json").exists():
        pytest.skip(f"no processed data for {event}")
    at = _render(event)
    labels = [t.label for t in at.tabs]
    assert labels and labels[0] == "Haas", labels
    assert not at.exception, [e.value for e in at.exception]
    assert _jargon(at) == []
    for view in ("Ocon", "Bearman", "Haas Overview"):
        at.button_group(key=f"haas_view_{event}").set_value(view).run()
        assert not at.exception, (view, [e.value for e in at.exception])
        assert _jargon(at) == [], view


def test_plan_builder_sections():
    event = "spain-2026"
    if not (config.DATA_PROCESSED / f"outlook_{event}.npz").exists():
        pytest.skip("no forecast samples for spain-2026")
    at = _render(event)
    for section in ["Undercut", "Safety car", "Commit", "Practice focus", "Compare"]:
        at.button_group(key=f"desk_{event}_section").set_value(section).run()
        assert not at.exception, (section, [e.value for e in at.exception])
        assert _jargon(at) == [], section


def test_before_the_race_branch(tmp_path, monkeypatch):
    src = config.DATA_PROCESSED
    if not (src / "weekend_italy-2026.json").exists():
        pytest.skip("no pre-race model for italy-2026")
    for p in src.glob("*italy-2026*"):
        if p.is_file() and not p.name.startswith("meta_"):
            shutil.copy(p, tmp_path / p.name)
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    import streamlit as st
    st.cache_data.clear()
    try:
        at = _render("italy-2026")
        assert any("Before the race" in t for t in _texts(at._tree, []))
    finally:
        st.cache_data.clear()


def test_no_off_palette_widgets():
    offenders = []
    for f in (ROOT / "app").glob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"st\.(checkbox|toggle|radio|info|success|warning|error)\(", line):
                offenders.append(f"{f.name}:{i}")
    assert offenders == []
