"""Pre-cache every session we need, including telemetry.

Telemetry downloads dominate wall clock, so this runs once in the background at
the start of the build and is never re-pulled afterwards.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EVENTS, get_event  # noqa: E402
from src.ingest import load_session  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="*", default=["barcelona-2026", "hungary-2026"])
    ap.add_argument("--no-telemetry", action="store_true")
    args = ap.parse_args()

    ok, failed = [], []
    for key in args.events:
        ev = get_event(key)
        for sess in list(ev.practice_sessions) + ["Race"]:
            t0 = time.time()
            label = f"{key}/{sess}"
            try:
                s = load_session(ev, sess, telemetry=not args.no_telemetry)
                n = 0 if s.laps is None else len(s.laps)
                print(f"[ok]   {label:32s} {n:4d} laps  {time.time()-t0:6.1f}s",
                      flush=True)
                ok.append(label)
            except Exception as exc:
                print(f"[FAIL] {label:32s} {exc}", flush=True)
                traceback.print_exc()
                failed.append(label)
    print(f"\ncached {len(ok)} sessions, {len(failed)} failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
