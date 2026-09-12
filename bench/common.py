"""Shared helpers for the benchmark suite (bench/).

Everything here reads artifacts the pipeline already wrote under
data/processed/ and predictions/sealed/, and the two frozen builds the previous
benchmarks left under bench/baseline/ (V1) and bench/v2/ (V2).  Nothing fits a
model and **nothing in bench/ writes to data/processed/ or predictions/sealed/**:
the one exception is `bench_speed`, which times `seal_predictions` and then
deletes the file it sealed (`discard_sealed`), so the sealed directory stays
exactly as the pipeline left it.

Three builds, one vocabulary:

    baseline_*   V1, the first benchmark (bench/baseline/)
    v2_*         V2, the "after the fifteen recommendations" build (bench/v2/)
    the bare     V3, whatever is on disk now under data/processed/
    readers

The V3 helpers below (`cp_for`, `first_stop_tables`, `kappa_of`,
`dirty_air_of`, `fs_kwargs`) all degrade to V2 behaviour when the artefact or
the keyword they need does not exist yet, so every script in this directory
runs against a V2 pipeline output as well as a V3 one.
"""

from __future__ import annotations

import argparse
import inspect
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
BASELINE = ROOT / "bench" / "baseline"      # V1
V2 = ROOT / "bench" / "v2"                  # V2

EVENTS = ["australia-2026", "japan-2026", "barcelona-2026", "austria-2026",
          "belgium-2026", "hungary-2026", "italy-2026"]

SHORT = {"australia-2026": "AUS", "japan-2026": "JPN", "barcelona-2026": "BCN", "austria-2026": "AUT",
         "belgium-2026": "BEL", "hungary-2026": "HUN", "italy-2026": "ITA"}

# The three weekends whose first stops were not set by a safety car.  Every
# first-stop timing metric in the suite is restricted to these, as V2's was.
NON_SC_EVENTS = ["barcelona-2026", "austria-2026", "hungary-2026"]


def arg_events(description: str = "", extra=None) -> argparse.Namespace:
    """`--events` on every script, defaulting to all seven scored weekends.

    Smoke-testing one weekend is the difference between a 20-second edit cycle
    and a twenty-minute one, and the full suite still runs with no arguments.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--events", nargs="+", default=list(EVENTS), metavar="KEY",
                   help="event keys to benchmark (default: all seven scored weekends)")
    for args, kw in (extra or []):
        p.add_argument(*args, **kw)
    ns = p.parse_args()
    unknown = [k for k in ns.events if k not in EVENTS]
    if unknown:
        print(f"  note: {unknown} are not scored weekends; comparisons against V1/V2 will be empty")
    return ns


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
    """V1's meta for a weekend."""
    p = BASELINE / "processed" / f"meta_{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def baseline_out(name: str):
    """V1's benchmark output."""
    return _read_out(BASELINE / "out" / name)


def baseline_processed(path_name: str) -> Path:
    return BASELINE / "processed" / path_name


def v2_meta(key: str) -> dict | None:
    """V2's meta for a weekend (frozen under bench/v2/processed/)."""
    p = V2 / "processed" / f"meta_{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def v2_out(name: str):
    """V2's benchmark output (frozen under bench/v2/out/)."""
    return _read_out(V2 / "out" / name)


def v2_processed(path_name: str) -> Path:
    """A path inside V2's frozen data/processed/ copy — posteriors, calibration,
    the per-weekend parquets.  Returned as a path, not read, because the callers
    want npz, parquet and json alike."""
    return V2 / "processed" / path_name


def v2_calibration() -> dict:
    p = v2_processed("calibration.json")
    return json.loads(p.read_text()) if p.exists() else {}


def _read_out(p: Path):
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


# --------------------------------------------------------------------------
# The V3 objective's new terms, read defensively
#
# Every helper here answers "what does this build give me?" rather than
# asserting what it must give, so the suite runs unchanged against a V2
# pipeline output (no first-stop prior, no per-circuit dirty air, no kappa) and
# against a V3 one.  A missing term degrades to V2's objective, which is
# exactly the `no_*` ablation, so a silent miss shows up as an ablation that
# cannot tell itself apart from `full` rather than as a crash.
# --------------------------------------------------------------------------


def accepts(fn, name: str) -> bool:
    """Does `fn` take a keyword called `name`?  (Or **kwargs.)"""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def cp_for(key: str, m: dict | None = None):
    """This weekend's `CircuitPrior`, rebuilt from its own meta where possible.

    The meta block is what the pipeline actually decided on, so rebuilding from
    it keeps the benchmark honest about which history entered the shipped
    answer.  A V2 meta has no `first_stop_green`/`dirty_air`/`nomination`, and
    `circuit_prior` itself is the fallback — it is a cache-only read, offline,
    and already what bench_ablation/bench_apex/bench_stability call.
    """
    from dataclasses import fields as dc_fields

    from src.history import CircuitPrior, circuit_prior

    names = {f.name for f in dc_fields(CircuitPrior)}
    block = dict((m or meta(key)).get("circuit_history") or {})
    if block.get("first_stop_green"):
        cp = CircuitPrior(**{k: v for k, v in block.items() if k in names})
        # A meta round-trip stringifies the integer year keys of
        # `nominations_by_year`; `plan_prior_for` reads both spellings, but the
        # benchmark prints them, so normalise here once.
        cp.nominations_by_year = {int(y): v for y, v in (cp.nominations_by_year or {}).items()}
        return cp
    return circuit_prior(key, probe_practice_temp=False)


def first_stop_tables(cp, ev, compounds):
    """The V3 first-stop penalty table for this weekend, or None.

    **Opaque by contract.**  Whatever `firststop.first_stop_penalty_table`
    returns is handed to the objective unchanged — nothing in bench/ indexes it,
    reads its keys or assumes its depth.  It has already been one shape
    (`{compound: (n+1,) array}`) and is becoming another
    (`{compound: {n_stops: array, "any": array}}`); the benchmark's only
    legitimate interest is "is there one, and does the search take it", so the
    only thing read off it here is truthiness.
    """
    try:
        from src.firststop import first_stop_penalty_table
    except ImportError:
        return None
    fsg = getattr(cp, "first_stop_green", None) or None
    if not fsg:
        return None
    try:
        return first_stop_penalty_table(fsg, int(ev.n_race_laps), list(compounds))
    except Exception as exc:   # a malformed history must not take the suite down
        print(f"  first-stop table unavailable for {getattr(ev, 'key', ev)}: {exc}")
        return None


def kappa_of(cal) -> float:
    """The calibrated first-stop weight, 0 on a V2 calibration (= no term)."""
    return float(getattr(cal, "first_stop_kappa_s", 0.0) or 0.0)


def dirty_air_of(cal, circuit: str | None) -> float:
    """The circuit's dirty-air cost where V3 has one, the pooled value otherwise."""
    fn = getattr(cal, "dirty_air_for", None)
    if callable(fn) and circuit:
        try:
            return float(fn(circuit))
        except Exception:
            pass
    return float(cal.dirty_air_s_per_lap)


def fs_kwargs(fn, tables, kappa: float) -> dict:
    """The first-stop keywords `fn` will accept, and nothing it will not.

    `tables` passes through by reference and is never inspected beyond being
    non-empty (see `first_stop_tables`): the objective owns its shape, and a
    benchmark that reached into it would have to be re-edited every time the
    penalty gains a dimension.
    """
    if not tables or not (kappa and np.isfinite(kappa)):
        return {}
    out = {}
    if accepts(fn, "first_stop_prior"):
        out["first_stop_prior"] = tables
    if accepts(fn, "first_stop_kappa_s"):
        out["first_stop_kappa_s"] = float(kappa)
    return out


def objective_kwargs(key: str, *, cal=None, m: dict | None = None, cp=None, compounds=None,
                     fn=None) -> dict:
    """Everything V3 adds to a search's keywords for this weekend.

    Returned as a dict so a caller can splat it into `simulate_model`,
    `search_with_pace_calibration` or `pit_window_model`; pass `fn` and the
    keywords the target does not take are dropped.
    """
    from src.calibration import get_calibration

    ev = get_event(key)
    m = m or meta(key)
    cal = cal if cal is not None else get_calibration(ev)
    cp = cp if cp is not None else cp_for(key, m)
    comps = list(compounds) if compounds is not None else list((m.get("bayes") or {}).get("comp_offset") or VALID_COMPOUNDS)
    tables = first_stop_tables(cp, ev, comps)
    kw = {"traffic_s_per_lap": dirty_air_of(cal, ev.circuit)}
    kw.update(fs_kwargs(fn or _sim(), tables, kappa_of(cal)))
    return kw


def _sim():
    from src import strategy as strat
    return strat.simulate_model


def memoise_regime() -> None:
    """Cache `regime.measure_regime(key)` inside this benchmark process.

    It is a pure read: two parquets and a stint-fixed-effects fit, the same
    answer every time.  The accuracy benchmark asks for five regime variants per
    weekend and each one re-pools six donors, so without this the suite spends
    most of its time re-measuring the same six races 35 times.  Calls that pass
    frames are handed straight through.
    """
    import src.regime as R

    if getattr(R, "_bench_memoised", False):
        return
    orig, cache = R.measure_regime, {}

    def wrapped(event, **kw):
        if kw:
            return orig(event, **kw)
        k = event if isinstance(event, str) else event.key
        if k not in cache:
            cache[k] = orig(event)
        return cache[k]

    R.measure_regime, R._bench_measure_regime_uncached, R._bench_memoised = wrapped, orig, True


def discard_sealed(path) -> None:
    """Delete a sealed file the benchmark wrote only to time the sealing.

    `bench_speed` times `validate.seal_predictions`, which is the one call in
    the suite that writes outside bench/.  Removing the file and its sha256
    side-car keeps `predictions/sealed/` exactly as `make history` left it, so
    the next `validate.latest_sealed` does not pick up a benchmark artefact.
    """
    p = Path(path)
    for q in (p, p.with_suffix(".json.sha256")):
        try:
            if q.exists() and q.is_relative_to(SEALED_DIR):
                q.unlink()
        except Exception as exc:   # pragma: no cover
            print(f"  could not remove the discarded seal {q.name}: {exc}")


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
