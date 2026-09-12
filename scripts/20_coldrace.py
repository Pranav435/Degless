"""Cold race: one shot, live, no retuning.

Runs the *identical* pipeline on a weekend the model was never developed
against — a different track and, at Hungary 2026, a genuinely different
temperature regime (FP2 ran at 27-34 C track, FP3 at 49-55 C).  No parameter,
prior or filter is allowed to change between the dev race and this one; that is
the entire point.  Run it once and freeze the numbers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import COLD_EVENT, DATA_PROCESSED, DEV_EVENT  # noqa: E402


def headline(key: str) -> dict:
    p = DATA_PROCESSED / f"meta_{key}.json"
    if not p.exists():
        return {}
    m = json.loads(p.read_text())
    return {
        "event": m["event_name"],
        "clean practice laps": m["n_clean_laps"],
        "max r_hat": round(m["bayes"]["max_rhat"], 4),
        "divergences": m["bayes"]["n_divergences"],
        "MEDIUM slope s/lap": round(
            next((r["slope_s_per_lap"] for r in m["bayes"]["slopes"]
                  if r["compound"] == "MEDIUM"), float("nan")), 3),
        "stint-rate MAE s/lap": round(m["score"]["mae"], 3),
        "bias s/lap": round(m["score"].get("bias", float("nan")), 3),
        "90% coverage": f"{m['score']['coverage'].get('0.9', float('nan')):.1%}",
        "pit loss s": round(m["pit_loss_s"], 2),
        "plan": m["strategy"]["best"],
        "push chosen": round(m["strategy"].get("push", float("nan")), 2),
        "implied regime": round(m["strategy"].get("implied_regime", float("nan")), 2),
        "stops the field ran": m.get("backtest", {}).get("observed_stop_counts", {}),
        "stints inside observed range": m.get("backtest", {}).get(
            "all_stints_inside_observed_range"),
        "gates passed": f"{sum(g['pass'] for g in m['gates'])}/{len(m['gates'])}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default=COLD_EVENT)
    ap.add_argument("--compare-to", default=DEV_EVENT)
    args, rest = ap.parse_known_args()

    print(f"=== COLD RACE: {args.event} — one shot, no retuning ===\n", flush=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "10_pipeline.py"),
           "--event", args.event, *rest]
    rc = subprocess.call(cmd)

    print("\n=== dev race vs cold race ===")
    dev, cold = headline(args.compare_to), headline(args.event)
    if dev and cold:
        keys = list(dev.keys())
        w = max(len(k) for k in keys)
        print(f"{'':{w}}   {'dev':>22} {'cold':>22}")
        for k in keys:
            print(f"{k:{w}}   {str(dev[k]):>22} {str(cold[k]):>22}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
