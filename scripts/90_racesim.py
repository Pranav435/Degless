"""Simulate the race with the live engine in the loop, for the Race sim tab.

    .venv/bin/python scripts/90_racesim.py --event spain-2026
    .venv/bin/python scripts/90_racesim.py --event spain-2026 --quick     # the base scenario only
    .venv/bin/python scripts/90_racesim.py --event spain-2026 --grid      # refresh the grid from qualifying first

Writes data/processed/racesim_<key>.json (every scenario: the engine-driven
race, the plan-blind and copy-the-field baselines, the engine's lap-by-lap
record for the two Haas cars, the tick times) and data/processed/grid_<key>.json.
The supervisor runs this after every practice refit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_event  # noqa: E402
from src.racesim import DEFAULT_SCENARIOS, build, load_grid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--quick", action="store_true", help="the base scenario only")
    ap.add_argument("--grid", action="store_true", help="refresh the grid from the qualifying archive")
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ev = get_event(args.event)
    t0 = time.time()
    g = load_grid(ev, refresh=args.grid)
    print(f"{ev.name}: grid from {g['source']}")
    haas = [c for c in g["cars"] if c["team"] == "Haas F1 Team"]
    print("  " + " · ".join(f"{c['code']} P{c['grid']}" + (f" ({c['quali_best_s']:.3f})" if c.get("quali_best_s") else " (no time)")
                            for c in haas))
    scen = DEFAULT_SCENARIOS[:1] if args.quick else DEFAULT_SCENARIOS
    build(ev, scen, n_draws=args.draws, baselines=not args.no_baselines)
    print(f"  written in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
