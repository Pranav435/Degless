"""Live tick time, race state on vs off, paired lap by lap in one process.

`bench_live` times one engine per process, so two runs minutes apart on a
laptop with other work on it measure the machine as much as the engine.  Here
both engines see the same `LiveState` on the same lap, ticked back to back in
alternating order, so the difference is the race-state term and nothing else.

Usage: python bench/bench_tick_paired.py [--events hungary-2026 barcelona-2026]
Writes bench/out/tick_paired.json.
"""

from __future__ import annotations

import time

import numpy as np

from common import ROOT, arg_events, dump, offline  # noqa: E402
from src.live.engine import RaceEngine, WeekendModel
from src.live.sources import RecordedSource
from src.live.state import LiveState

ARCHIVES = {"hungary-2026": ROOT / "data/raw/livetiming/2026_hungary_race",
            "barcelona-2026": ROOT / "data/raw/livetiming/2026_barcelona_race"}


def paired(key: str) -> dict:
    wm = WeekendModel.load(key)
    eng = {"on": RaceEngine(wm, race_state=True), "off": RaceEngine(wm, race_state=False)}
    st = LiveState()
    t = {"on": [], "off": []}
    last, flip = None, False
    for msg in RecordedSource(ARCHIVES[key], include_zipped=False):
        st.apply(msg)
        evs = st.drain_events()
        cur = st.lap_count.get("current")
        if any(k == "lap" for _, k, _ in evs) and cur and cur != last:
            last = cur
            order = ("off", "on") if flip else ("on", "off")
            flip = not flip
            for name in order:
                t0 = time.perf_counter()
                eng[name].tick(st)
                t[name].append(1000 * (time.perf_counter() - t0))
    on, off = np.array(t["on"]), np.array(t["off"])
    out = {"n_ticks": int(len(on)),
           "on_ms": {"mean": round(float(on.mean()), 1), "median": round(float(np.median(on)), 1),
                     "p95": round(float(np.quantile(on, 0.95)), 1), "max": round(float(on.max()), 1)},
           "off_ms": {"mean": round(float(off.mean()), 1), "median": round(float(np.median(off)), 1),
                      "p95": round(float(np.quantile(off, 0.95)), 1), "max": round(float(off.max()), 1)},
           "ratio_mean": round(float(on.mean() / off.mean()), 3),
           "extra_ms_median": round(float(np.median(on - off)), 1)}
    print(f"== {key}: race state on {out['on_ms']} | off {out['off_ms']} | on/off {out['ratio_mean']} "
          f"| extra per tick (median) {out['extra_ms_median']} ms")
    return out


def main() -> None:
    args = arg_events(__doc__)
    offline()
    keys = [k for k in args.events if k in ARCHIVES] or list(ARCHIVES)
    dump("tick_paired.json", {k: paired(k) for k in keys if ARCHIVES[k].exists()})


if __name__ == "__main__":
    main()
