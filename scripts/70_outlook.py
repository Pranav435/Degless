"""Build the outlook for a weekend: the best current strategy picture, from
whatever is known - priors, the sealed practice fit, the live long-run board.

    .venv/bin/python scripts/70_outlook.py --event spain-2026
    .venv/bin/python scripts/70_outlook.py --event spain-2026 --session 11362   # fold in a live practice board
    .venv/bin/python scripts/70_outlook.py --event spain-2026 --quick           # skip the scenario matrix

The supervisor (`scripts/run.py`) runs this by itself: every 30 minutes
between sessions, every 3 minutes while a practice session is live, and
immediately after every refit.  Outputs go to data/processed/outlook_<key>.*
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_event  # noqa: E402
from src.outlook import build  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--session", default=None, help="data/live/<session> whose practice snapshot to fold in")
    ap.add_argument("--quick", action="store_true", help="skip the scenario matrix")
    ap.add_argument("--draws", type=int, default=300)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ev = get_event(args.event)
    out = build(ev, session=args.session, quick=args.quick, n_draws=args.draws)
    st = out.get("strategy") or {}
    print(f"{ev.name}: {out['stage_label']}")
    for s in out.get("sources", []):
        print(f"  - {s}")
    if st:
        print(f"  plan: {st['best']}  P(stops) {st['p_stops']}  push {st['push']:.2f}")
        for c, v in st["life"].items():
            print(f"    {c:7s} {v['deg_s_per_lap']:.3f} s/lap  life {v['life_laps']:.0f} [{v['life_lo']:.0f}-{v['life_hi']:.0f}]")
        for w in st.get("pit_windows", []):
            print(f"    stop {w['stop']}: lap {w['recommended']} (window {w['lo']}-{w['hi']})")
        alt = (out.get("alternatives") or {}).get("plan_b")
        if alt:
            print(f"  plan B: {alt['label']} ({alt['delta_s']:+.1f} s), overtakes A at deg x{alt.get('switch_mult')}")
        sc = out.get("scenarios") or {}
        if sc:
            print(f"  robust across scenarios: {sc.get('robust')} (max regret {sc.get('robust_max_regret_s', float('nan')):.1f} s; "
                  f"base plan {sc.get('base_max_regret_s', float('nan')):.1f} s)")
        for r in out.get("programme", [])[:3]:
            print(f"  practice: {r['text']}")
        for r in (out.get("sc_playbook") or {}).get("ranges", []):
            print(f"  SC laps {r['from']}-{r['to']}: {r['verdict']} ({r['gain_s']:+.1f} s) -> {r['continuation']}")
    print(f"  ({out.get('runtime_s')} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
