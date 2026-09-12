"""Measure the practice -> race regime ratio on previous years, once, and cache it.

    .venv/bin/python scripts/05_history_practice.py --events hungary-2026
    .venv/bin/python scripts/05_history_practice.py            # the 7 scored weekends + spain-2026
    .venv/bin/python scripts/05_history_practice.py --offline   # cache only, no network

**This is the one script that is expected to use the network.**  The FastF1 cache
that ships with the repo holds races only, so every historical practice session
is a fresh load of roughly 17 s (a circuit-year is therefore ~1 minute, the full
set ~20 minutes).  Each circuit-year is then summarised into
`data/processed/history/regime_<year>_<circuit>.json`, and from that point
`src.regime_history.circuit_regime_prior` - and so every weekend build,
`make history` and the whole benchmark - reads nothing but those files.

Failures are normal and reported rather than fatal: a sprint weekend has one
practice session instead of three, a wet race cannot be compared with a dry
practice at all, and a circuit that did not exist in 2023 has no 2023 race.  A
circuit with no usable year simply has no prior, and the regime factor falls
back to the 2026 donor pool alone.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_event  # noqa: E402
from src.history import YEARS  # noqa: E402
from src.regime import REGIME_RATIO_BAND  # noqa: E402
from src.regime_history import (  # noqa: E402
    DEFAULT_SESSIONS,
    circuit_regime_prior,
    measure_regime_history,
)

BENCHMARK_EVENTS = ["australia-2026", "japan-2026", "barcelona-2026", "austria-2026",
                    "belgium-2026", "hungary-2026", "italy-2026", "spain-2026"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", nargs="*", default=BENCHMARK_EVENTS,
                    help="weekend keys whose circuits to measure (default: the 7 scored + spain-2026)")
    ap.add_argument("--years", nargs="*", type=int, default=list(YEARS))
    ap.add_argument("--sessions", nargs="*", default=list(DEFAULT_SESSIONS))
    ap.add_argument("--offline", action="store_true",
                    help="keep FastF1 off the network: only circuit-years already cached are measured")
    ap.add_argument("--force", action="store_true", help="re-measure even if a summary is cached")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    circuits: list[tuple[str, str]] = []
    for key in args.events:
        ev = get_event(key)
        if ev.circuit not in [c for c, _ in circuits]:
            circuits.append((ev.circuit, ev.key))

    ok = failed = 0
    for circuit, key in circuits:
        print(f"\n{circuit}  ({key})", flush=True)
        for year in args.years:
            t0 = time.time()
            try:
                d = measure_regime_history(year, circuit, sessions=tuple(args.sessions),
                                           offline=args.offline, force=args.force)
            except Exception as exc:
                print(f"  {year} {circuit:20s} FAILED  {exc}", flush=True)
                traceback.print_exc()
                failed += 1
                continue
            dt = time.time() - t0
            if d is None:
                print(f"  {year} {circuit:20s} not cached (offline)  {dt:5.1f}s", flush=True)
                failed += 1
            elif d.get("missing"):
                extra = (f" rain in {d['sessions_rain']}" if d.get("sessions_rain") else "")
                print(f"  {year} {circuit:20s} missing: {d.get('why')}{extra}  {dt:5.1f}s", flush=True)
                failed += 1
            else:
                r = d.get("ratio")
                band = "" if REGIME_RATIO_BAND[0] <= r <= REGIME_RATIO_BAND[1] else "  OUT OF BAND: discarded"
                print(f"  {year} {circuit:20s} ratio {r:.3f}{band}  race stints {d['n_race_stints']:3d}  "
                      f"practice stints {d['n_practice_stints']:3d}  "
                      f"{'/'.join(c[0] for c in d['usable_compounds']) or '-':5s}  "
                      f"{', '.join(d['sessions_used'])}  "
                      f"T {d.get('track_temp_practice_c') or float('nan'):.0f}->"
                      f"{d.get('track_temp_race_c') or float('nan'):.0f} degC  {dt:5.1f}s", flush=True)
                ok += 1

    print(f"\n{ok} circuit-years measured, {failed} unavailable\n")
    print("pooled circuit regime prior (what regime_prior will combine):")
    for key in args.events:
        ev = get_event(key)
        cp = circuit_regime_prior(ev, years=tuple(args.years))
        if cp:
            print(f"  {ev.key:16s} {ev.circuit:20s} {cp['ratio']:.3f}x  +/-{cp['ln_sd']:.3f} ln  "
                  f"years {cp['years']}  by year {cp['by_year']}")
        else:
            print(f"  {ev.key:16s} {ev.circuit:20s} no usable year: the donor pool alone applies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
