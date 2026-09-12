"""Live-engine benchmark: replay archived races through RaceEngine, one tick per
lap, and score the calls against what happened.

  * latency: wall time per tick (all cars), split by lap number, worst case
  * regime multiplier: the engine's field estimate by lap vs the value the
    offline estimator measures with the whole race in hand
  * stop calls: for every real green-flag stop, the engine's recommended
    next-stop lap and window 3 laps before it; share of real stops inside the
    window; median |error|; cliff alarms that preceded a stop
  * plan stability: how often the leader's recommended plan changed shape

Usage: python bench/bench_live.py [hungary-2026 barcelona-2026]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from common import ROOT, dump, meta  # noqa: E402
from src.live.engine import RaceEngine, WeekendModel
from src.live.sources import RecordedSource
from src.live.state import LiveState

ARCHIVES = {"hungary-2026": ROOT / "data/raw/livetiming/2026_hungary_race",
            "barcelona-2026": ROOT / "data/raw/livetiming/2026_barcelona_race"}


def replay(key: str) -> dict:
    wm = WeekendModel.load(key)
    eng = RaceEngine(wm)
    st = LiveState()
    ticks, hist = [], []
    last_lap = None
    t_parse = 0.0
    src = RecordedSource(ARCHIVES[key], include_zipped=False)
    t0 = time.perf_counter()
    for msg in src:
        tp = time.perf_counter()
        st.apply(msg)
        evs = st.drain_events()
        t_parse += time.perf_counter() - tp
        cur = st.lap_count.get("current")
        if any(k == "lap" for _, k, _ in evs) and cur and cur != last_lap:
            last_lap = cur
            t1 = time.perf_counter()
            snap = eng.tick(st)
            dt = time.perf_counter() - t1
            ticks.append({"lap": int(cur), "seconds": dt, "n_cars": len(snap["field"]),
                          "n_options_leader": (snap["field"][0].get("plan") or {}).get("n_options")})
            hist.append({"lap": int(cur), "m": snap["meta"]["regime_multiplier"], "sc": snap["meta"]["sc_active"],
                         "pit_loss": snap["meta"]["pit_loss_s"],
                         "field": {f["driver"]: {"plan_best": (f.get("plan") or {}).get("best"),
                                                 "next": (f.get("plan") or {}).get("next_stop"),
                                                 "window": [(f.get("plan") or {}).get("window_lo"), (f.get("plan") or {}).get("window_hi")],
                                                 "box_now": (f.get("plan") or {}).get("delta_box_now_s"),
                                                 "alarm": f.get("cliff_alarm"), "p_cliff": f.get("p_past_cliff"),
                                                 "wear": f.get("wear"), "m": f.get("m_mean"),
                                                 "tyre_age": f.get("tyre_age"), "compound": f.get("compound"),
                                                 "position": f.get("position")} for f in snap["field"]}})
    total = time.perf_counter() - t0
    laps = st.laps_df(include_current=False)
    laps = laps[laps["is_complete"]]
    # real stops
    stops = {d: sorted(int(x) for x in g["lap_number"]) for d, g in laps[laps["pit_in"]].groupby("driver")}
    by_lap = {h["lap"]: h["field"] for h in hist}
    rows = []
    for drv, in_laps in stops.items():
        for p in in_laps:
            f3 = (by_lap.get(p - 3) or {}).get(drv) or {}
            f1 = (by_lap.get(p - 1) or {}).get(drv) or {}
            win = f3.get("window") or [None, None]
            alarm = next((L for L in range(max(1, p - 15), p + 1)
                          if ((by_lap.get(L) or {}).get(drv) or {}).get("alarm")), None)
            rows.append({"driver": drv, "in_lap": p, "rec_3_before": f3.get("next"),
                         "window_3_before": win, "in_window": (win[0] is not None and win[0] <= p <= (win[1] or -1)),
                         "err_laps": (p - f3["next"]) if f3.get("next") is not None else None,
                         "box_now_delta_1_before": f1.get("box_now"), "p_cliff_1_before": f1.get("p_cliff"),
                         "wear_1_before": f1.get("wear"), "alarm_lap": alarm,
                         "position_at_stop": f1.get("position")})
    per = pd.DataFrame(rows)
    per_top = per[per["position_at_stop"].fillna(99) <= 10] if len(per) else per
    m_self = (meta(key).get("regime", {}).get("self_measured") or {}).get("ratio")
    m_traj = [(h["lap"], round(h["m"]["mean"], 3), round(h["m"]["p05"], 3), round(h["m"]["p95"], 3)) for h in hist]
    # plan-shape churn for the eventual top 3
    lat = pd.DataFrame(ticks)
    out = {
        "event": key, "n_ticks": len(ticks), "replay_wall_s": round(total, 1), "parse_wall_s": round(t_parse, 1),
        "tick_ms": {"mean": round(1000 * lat["seconds"].mean(), 1), "median": round(1000 * lat["seconds"].median(), 1),
                    "p95": round(1000 * lat["seconds"].quantile(0.95), 1), "max": round(1000 * lat["seconds"].max(), 1),
                    "first_tick": round(1000 * lat["seconds"].iloc[0], 1),
                    "by_third": [round(1000 * lat.iloc[i::3]["seconds"].mean(), 1) for i in range(0)] or
                                [round(1000 * lat[(lat.lap <= q)]["seconds"].mean(), 1) for q in (10, 30, 100)]},
        "n_options_leader_median": float(lat["n_options_leader"].median()),
        "stops": {"n": int(len(per)), "share_in_window": float(per["in_window"].mean()) if len(per) else None,
                  "median_abs_err_laps": float(per["err_laps"].abs().median()) if per["err_laps"].notna().any() else None,
                  "share_err_within_3": float((per["err_laps"].abs() <= 3).mean()) if per["err_laps"].notna().any() else None,
                  "n_with_recommendation": int(per["err_laps"].notna().sum()),
                  "alarms_before_stop": int(per["alarm_lap"].notna().sum()),
                  "median_box_now_delta_1_before": float(per["box_now_delta_1_before"].median()) if per["box_now_delta_1_before"].notna().any() else None,
                  "share_box_now_le_1s": float((per["box_now_delta_1_before"] <= 1.0).mean()) if per["box_now_delta_1_before"].notna().any() else None,
                  "median_p_cliff_1_before": float(per["p_cliff_1_before"].median()) if per["p_cliff_1_before"].notna().any() else None,
                  "median_wear_1_before": float(per["wear_1_before"].median()) if per["wear_1_before"].notna().any() else None},
        "stops_top10": {"n": int(len(per_top)), "share_in_window": float(per_top["in_window"].mean()) if len(per_top) else None,
                        "median_abs_err_laps": float(per_top["err_laps"].abs().median()) if len(per_top) and per_top["err_laps"].notna().any() else None},
        "regime": {"self_measured_offline": m_self, "prior_mean": hist[0]["m"]["prior_mean"] if hist else None,
                   "by_lap": m_traj[::5] + [m_traj[-1]], "final": m_traj[-1] if m_traj else None},
        "pit_loss_by_lap": [(h["lap"], round(h["pit_loss"], 2)) for h in hist][::10],
        "per_stop": per.to_dict("records"),
        "latency_by_lap": lat.to_dict("records"),
    }
    print(f"\n== {key}: {len(ticks)} ticks in {total:.1f}s (parse {t_parse:.1f}s); tick ms mean {out['tick_ms']['mean']} "
          f"median {out['tick_ms']['median']} p95 {out['tick_ms']['p95']} max {out['tick_ms']['max']}")
    print(f"   stops: {out['stops']}")
    print(f"   top10 stops: {out['stops_top10']}")
    print(f"   regime m: prior {out['regime']['prior_mean']}, final {out['regime']['final']}, offline self-measured {m_self}")
    print(f"   m by lap: {m_traj[::10]}")
    return out


def main() -> None:
    keys = sys.argv[1:] or list(ARCHIVES)
    res = {k: replay(k) for k in keys if ARCHIVES[k].exists()}
    dump("live.json", res)


if __name__ == "__main__":
    main()
