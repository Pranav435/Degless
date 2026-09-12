"""Test-only entry point: renders just the Haas tab, standalone, so
`tests/test_v4_ui.py` can point `AppTest.from_file` at it without pulling in
the sidebar, the event selector or every other tab `app/dashboard.py` builds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.haas_tab import render_haas  # noqa: E402
from src.config import DATA_PROCESSED, EVENTS  # noqa: E402

EVENT = "hungary-2026"
SESSION_KEY = "synth"

_meta = json.loads((DATA_PROCESSED / f"meta_{EVENT}.json").read_text())
render_haas(EVENT, EVENTS[EVENT], _meta, None, SESSION_KEY)
