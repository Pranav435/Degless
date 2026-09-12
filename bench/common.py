"""Shared helpers for the benchmark suite (bench/).

Everything here reads artifacts the pipeline already wrote under
data/processed/ and predictions/sealed/, and the frozen baseline the previous
benchmark left under bench/baseline/.  Nothing fits a model.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, SEALED_DIR, VALID_COMPOUNDS, get_event  # noqa: E402

OUT = ROOT / "bench" / "out"
OUT.mkdir(parents=True, exist_ok=True)
BASELINE = ROOT / "bench" / "baseline"

EVENTS = ["australia-2026", "japan-2026", "barcelona-2026", "austria-2026",
          "belgium-2026", "hungary-2026", "italy-2026"]


def offline() -> None:
    """Keep FastF1 off the network: every session the benchmark needs is cached,
    and the Ergast mirror's timeouts were most of a five-minute stage last time."""
    try:
        import fastf1
        from src.config import FASTF1_CACHE
        fastf1.Cache.enable_cache(str(FASTF1_CACHE))
        fastf1.Cache.offline_mode(True)
    except Exception:
        pass


class Timer:
    def __init__(self):
        self.rows = []

    def __call__(self, name):
        t = self

        class _C:
            def __enter__(s):
                s.t0 = time.perf_counter()
                return s

            def __exit__(s, *a):
                t.rows.append({"stage": name, "seconds": round(time.perf_counter() - s.t0, 3)})
        return _C()

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def meta(key: str) -> dict:
    return json.loads((DATA_PROCESSED / f"meta_{key}.json").read_text())


def baseline_meta(key: str) -> dict | None:
    p = BASELINE / "processed" / f"meta_{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def baseline_out(name: str):
    p = BASELINE / "out" / name
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return pd.read_csv(p)


def race_table(key: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / f"laps_{key}_race.parquet")


def clean_practice(key: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / f"clean_{key}_practice.parquet")


def sealed_for(key: str, m: dict | None = None) -> dict:
    from src.validate import load_sealed

    m = m or meta(key)
    p = SEALED_DIR / m["sealed_file"]
    d = load_sealed(p)
    d["_file"] = p.name
    return d


def classified(race: pd.DataFrame, n_race_laps: int) -> list:
    fin = race.groupby("driver")["lap_number"].max()
    return list(fin[fin >= n_race_laps - 2].index)


def stints_of(race: pd.DataFrame, min_laps: int = 2) -> pd.DataFrame:
    st = (race.groupby(["driver", "stint"])
          .agg(compound=("compound", "first"), start=("lap_number", "min"),
               end=("lap_number", "max"), n=("lap_number", "size"))
          .reset_index().sort_values(["driver", "start"]))
    return st[(st["n"] >= min_laps) & st["compound"].isin(VALID_COMPOUNDS)]


def driver_plans(race: pd.DataFrame, n_race_laps: int) -> pd.DataFrame:
    """One row per classified finisher: compound sequence, in-laps, stop count,
    and whether each stop was taken under a safety car / VSC / red flag."""
    from src.strategy import is_sc_status

    st = stints_of(race)
    cls = classified(race, n_race_laps)
    status = race.set_index(["driver", "lap_number"])["track_status"].astype(str).to_dict()
    rows = []
    for drv, g in st[st["driver"].isin(cls)].groupby("driver"):
        g = g.sort_values("start")
        seq = g["compound"].tolist()
        pits = [int(x) - 1 for x in g["start"].tolist()[1:]]   # in-lap = lap before the new stint starts
        sc = [bool(is_sc_status(status.get((drv, float(p)), status.get((drv, float(p - 1)), "1")))) for p in pits]
        rows.append({"driver": drv, "seq": "-".join(seq), "short": "-".join(c[0] for c in seq),
                     "n_stops": len(seq) - 1, "pit_laps": pits, "sc_stops": sc,
                     "first_sc": (sc[0] if sc else False),
                     "stint_lens": g["n"].tolist()})
    return pd.DataFrame(rows)


def race_results(key: str) -> pd.DataFrame | None:
    """Finishing order from the FastF1 cache (Abbreviation, Position)."""
    try:
        import fastf1
        import logging
        from src.config import FASTF1_CACHE
        fastf1.Cache.enable_cache(str(FASTF1_CACHE))
        logging.getLogger("fastf1").setLevel(logging.ERROR)
        ev = get_event(key)
        s = fastf1.get_session(ev.ff1_year, ev.ff1_round, "Race")
        s.load(laps=False, telemetry=False, weather=False, messages=False)
        r = s.results[["Abbreviation", "Position", "Status"]].copy()
        r["Position"] = pd.to_numeric(r["Position"], errors="coerce")
        return r.rename(columns={"Abbreviation": "driver", "Position": "position", "Status": "status"})
    except Exception as exc:  # pragma: no cover
        print(f"  results unavailable for {key}: {exc}")
        return None


def dump(name: str, obj) -> Path:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1, default=_default))
    return p


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    return str(o)
