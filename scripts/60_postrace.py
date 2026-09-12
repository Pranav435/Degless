"""After the flag: score the sealed prediction, and score what the live engine said.

Two receipts:

1. **The model** — the full retrospective pipeline (`10_pipeline.py`) on the
   weekend now that the race exists: sealed practice curves against the race
   stints, calibration, counterfactual, backtest.  Same script as before;
   nothing about the race reaches a fit.

2. **The live engine** — `data/live/<session>/history.jsonl` holds the field
   table as the engine saw it on every lap.  This scores those calls against
   what actually happened: did the **pace-collapse** alarm precede the pit stop,
   how far were the recommended in-laps from the real ones, did the undercut
   threats materialise, how did the live degradation multiplier track the value
   the offline estimator measures with the whole race in hand.

   The alarm scored here is the within-stint collapse detection (`cliff_alarm`
   on the field row, which is `pace_collapse`), not the old wear-based flag —
   `collapse_*` in the output, on the same keys.

    .venv/bin/python scripts/60_postrace.py --event italy-2026 --session 11361
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, get_event  # noqa: E402
from src.live.store import LIVE_DIR, read_history, read_laps  # noqa: E402


def score_live(session_key: str, ev) -> dict:
    hist = read_history(session_key)
    laps = read_laps(session_key)
    if not hist or laps.empty:
        return {"note": "no live history for this session"}
    laps = laps[laps["is_complete"]] if "is_complete" in laps else laps
    # actual stops: in-lap numbers per driver
    stops = {}
    for drv, g in laps[laps["pit_in"]].groupby("driver"):
        stops[drv] = sorted(int(x) for x in g["lap_number"])
    # per driver, per stint: the engine's calls in the laps before each real stop
    rows = []
    by_lap = {h["lap"]: {f["driver"]: f for f in h.get("field", [])} for h in hist}
    for drv, in_laps in stops.items():
        for p in in_laps:
            # what did the engine recommend 1, 3 and 5 laps before the stop?
            calls = {}
            for back in (1, 3, 5):
                f = (by_lap.get(p - back) or {}).get(drv)
                if f:
                    calls[back] = {"plan": f.get("plan_best"), "next": f.get("plan_next_stop"),
                                   "window": f.get("plan_window"), "collapse": f.get("cliff_alarm"),
                                   "p_past_cliff": f.get("p_past_cliff"), "box_now": f.get("delta_box_now_s")}
            # the first lap the collapse detector called this tyre, at an age where
            # a collapse is a tyre rather than a cold out-lap
            first_collapse = next((h["lap"] for h in hist if (by_lap[h["lap"]].get(drv) or {}).get("cliff_alarm")
                                   and h["lap"] <= p and (by_lap[h["lap"]].get(drv) or {}).get("tyre_age", 0) > 3), None)
            c3 = calls.get(3, {})
            win = c3.get("window") or [None, None]
            rows.append({"driver": drv, "actual_in_lap": p,
                         "rec_in_lap_3_before": c3.get("next"),
                         "window_3_before": f"{win[0]}-{win[1]}" if win and win[0] is not None else None,
                         "in_window": (win[0] is not None and win[0] <= p <= (win[1] or 0)),
                         "err_laps": (p - c3["next"]) if c3.get("next") is not None else None,
                         "box_now_delta_1_before": calls.get(1, {}).get("box_now"),
                         "collapse_lap": first_collapse,
                         "collapse_lead_laps": (p - first_collapse) if first_collapse else None})
    per = pd.DataFrame(rows)
    out = {"n_stops": int(len(per)),
           "share_in_window": float(per["in_window"].mean()) if len(per) else float("nan"),
           "median_abs_err_laps": float(per["err_laps"].abs().median()) if per["err_laps"].notna().any() else float("nan"),
           "collapses_before_stop": int(per["collapse_lap"].notna().sum()),
           "median_collapse_lead_laps": (float(per["collapse_lead_laps"].median())
                                         if per["collapse_lead_laps"].notna().any() else float("nan")),
           "per_stop": per.to_dict("records")}
    # live multiplier trajectory
    m_traj = [(h["lap"], np.nanmean([f.get("m_mean") or np.nan for f in h.get("field", [])])) for h in hist]
    out["m_field_by_lap"] = [(int(l), round(float(m), 3)) for l, m in m_traj if np.isfinite(m)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--session", default=None, help="live session key under data/live (default: latest race)")
    ap.add_argument("--skip-pipeline", action="store_true")
    args = ap.parse_args()
    ev = get_event(args.event)

    if not args.skip_pipeline:
        print(f"=== scoring the sealed model on the {ev.name} race (10_pipeline.py) ===", flush=True)
        rc = subprocess.call([sys.executable, str(ROOT / "scripts" / "10_pipeline.py"), "--event", ev.key])
        if rc != 0:
            print("pipeline failed", flush=True)

    sk = args.session
    if sk is None:
        cands = sorted((p for p in LIVE_DIR.iterdir() if (p / "history.jsonl").exists()),
                       key=lambda p: p.stat().st_mtime)
        sk = cands[-1].name if cands else None
    if sk is None:
        print("no live session history to score"); return 0
    print(f"\n=== scoring the live engine's calls: data/live/{sk} ===", flush=True)
    res = score_live(sk, ev)
    print(json.dumps({k: v for k, v in res.items() if k != "per_stop"}, indent=1, default=str))
    if res.get("per_stop"):
        print(pd.DataFrame(res["per_stop"]).to_string(index=False))
    (DATA_PROCESSED / f"livescore_{ev.key}_{sk}.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
