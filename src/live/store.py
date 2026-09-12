"""Snapshots on disk: the daemon writes, the app reads.

The feed process and the dashboard are separate processes on purpose — a slow
render must never stall the feed, and a feed hiccup must never take the app
down.  They meet in `data/live/<session>/`:

* `snapshot.json` — the engine's latest output (field table, plans, alerts),
  written atomically (temp file + rename) a few times a minute;
* `laps.parquet` — every lap so far, in the canonical schema;
* `history.jsonl` — one line per lap of the race with the field table as the
  engine saw it *at that moment*, so the live recommendations can be scored
  after the flag ("the cliff alarm fired on lap 19; the car pitted on 23").
* `status.json` — feed health (source, connected, messages, last message).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT

LIVE_DIR = ROOT / "data" / "live"


def _clean(obj):
    """JSON-safe: numpy scalars/arrays, NaN → None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class SnapshotStore:
    def __init__(self, session_key: str, root: Path = LIVE_DIR):
        self.dir = Path(root) / str(session_key)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._last_hist_lap = None
        self._last_laps_write = 0.0
        (root / "current.json").parent.mkdir(parents=True, exist_ok=True)
        atomic_write(Path(root) / "current.json", json.dumps({"session": str(session_key),
                                                              "updated": time.time()}))

    def write_snapshot(self, snap: dict) -> None:
        atomic_write(self.dir / "snapshot.json", json.dumps(_clean(snap), separators=(",", ":")))
        lap = (snap.get("meta") or {}).get("lap_count", {}).get("current")
        if snap.get("engine") == "race" and lap is not None and lap != self._last_hist_lap:
            self._last_hist_lap = lap
            rec = {"lap": lap, "utc": snap["meta"].get("tick_utc"),
                   "track_status": snap["meta"].get("track_status"),
                   "field": [{k: v for k, v in r.items() if k not in ("plan", "proj")}
                             | {"plan_best": (r.get("plan") or {}).get("best"),
                                "plan_next_stop": (r.get("plan") or {}).get("next_stop"),
                                "plan_window": [(r.get("plan") or {}).get("window_lo"),
                                                (r.get("plan") or {}).get("window_hi")],
                                "delta_box_now_s": (r.get("plan") or {}).get("delta_box_now_s")}
                             for r in snap.get("field", [])]}
            with open(self.dir / "history.jsonl", "a") as f:
                f.write(json.dumps(_clean(rec), separators=(",", ":")) + "\n")

    def write_laps(self, laps: pd.DataFrame, *, min_interval_s: float = 5.0) -> None:
        now = time.time()
        if now - self._last_laps_write < min_interval_s:
            return
        self._last_laps_write = now
        if laps is None or laps.empty:
            return
        tmp = self.dir / "laps.parquet.tmp"
        df = laps.copy()
        if "utc_end" in df:
            df["utc_end"] = df["utc_end"].astype(str)
        df.to_parquet(tmp, index=False)
        os.replace(tmp, self.dir / "laps.parquet")

    def write_status(self, status: dict) -> None:
        atomic_write(self.dir / "status.json", json.dumps(_clean(status)))


def read_current(root: Path = LIVE_DIR) -> str | None:
    p = Path(root) / "current.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("session")
    except Exception:
        return None


def read_snapshot(session_key: str, root: Path = LIVE_DIR) -> dict:
    p = Path(root) / str(session_key) / "snapshot.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def read_status(session_key: str, root: Path = LIVE_DIR) -> dict:
    p = Path(root) / str(session_key) / "status.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def read_laps(session_key: str, root: Path = LIVE_DIR) -> pd.DataFrame:
    p = Path(root) / str(session_key) / "laps.parquet"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()


def read_history(session_key: str, root: Path = LIVE_DIR) -> list:
    p = Path(root) / str(session_key) / "history.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out
