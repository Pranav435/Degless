"""Commit the weekend model's own per-car plans as decision cards.

    .venv/bin/python scripts/85_plans.py --event spain-2026            # OCO and BEA
    .venv/bin/python scripts/85_plans.py --event spain-2026 --drivers OCO

The supervisor runs this after every practice refit, so the live race view
always tracks the two Haas cars against the freshest sealed plan.  Cards go to
data/live/plans/<event>.json (the same store the Plan builder's Commit writes).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cards import HAAS_DRIVERS, commit_model_plans  # noqa: E402
from src.plans import as_markdown  # noqa: E402
from src.config import get_event  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--drivers", nargs="*", default=list(HAAS_DRIVERS))
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ev = get_event(args.event)
    plans = commit_model_plans(ev.key, args.drivers, note=args.note)
    if not plans:
        print(f"{ev.name}: no weekend model (or no forecast) to commit plans from yet")
        return 1
    for p in plans:
        print(as_markdown(p, ev.name, ev.n_race_laps))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
