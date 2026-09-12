"""Live-engine benchmark: replay archived races through RaceEngine, one tick per
lap, and score the calls against what happened.

  * latency: wall time per tick (all cars), split by lap number, worst case
  * regime multiplier: the engine's field estimate by lap vs the value the
    offline estimator measures with the whole race in hand
  * stop calls: for every real green-flag stop, the engine's recommended
    next-stop lap and window 3 laps before it; share of real stops inside the
    window; median |error|
  * three signals, each scored as a classifier over *every car-lap* of the
    replay rather than only the laps before a stop:

        window    the engine's pit window was open on that lap
        box_now   boxing now cost <= 1.0 s against the best green plan
        collapse  the engine raised `cliff_alarm` (from V3 on, the within-stint
                  pace-collapse detector rather than the wear estimate)

    A signal on lap L is a true positive if that car pits on one of laps
    L+1..L+3 and a false positive otherwise; recall is the share of real stops
    with at least one such signal in the three laps before them.  Precision is
    the number that was missing from V2: an alarm that fires every lap catches
    every stop and says nothing.
  * plan stability: how often the leader's recommended plan changed shape
  * Haas (V4): `stops_haas` — the same stop-call metrics restricted to OCO and
    BEA, the two cars the tool is built for
  * decision stability (V4): the share of car-laps whose recommended *action*
    (`plan["decision"]["action"]`) changed from the previous lap with nothing
    material having changed — no stop by the car or one of its listed rivals,
    no safety-car change, and a best-vs-runner-up margin that moved by less
    than a second.  A pit wall that changes its mind for no reason is not
    trusted, and this is the number that says whether it does.

Usage: python bench/bench_live.py [--events hungary-2026 barcelona-2026] [--no-race-state]

`--no-race-state` replays the same archives through the engine with the V4
race-state term switched off (V3's objective) and writes `live_no_race_state.json`
beside `live.json`: the ablation, scored identically.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import ROOT, arg_events, dump, meta, offline  # noqa: E402
from src.live.engine import RaceEngine, WeekendModel
from src.live.sources import RecordedSource
from src.live.state import LiveState

ARCHIVES = {"hungary-2026": ROOT / "data/raw/livetiming/2026_hungary_race",
            "barcelona-2026": ROOT / "data/raw/livetiming/2026_barcelona_race"}

LOOKAHEAD = 3          # laps: a signal counts as a call if the stop follows within this many
BOX_NOW_MAX_S = 1.0    # "boxing now is free": the threshold V2 reported as a median

HAAS_DRIVERS = ("OCO", "BEA")     # #31 and #87, Haas F1 Team, in every 2026 lap table
# A best-vs-runner-up margin that moved by this much is new information, so an
# action change it explains is not churn.  Engineering constant of the same
# order as the engine's own window tolerance (`BOX_NOW_TOL_S`), stated in
# docs/v4_methodology.md; the block reports the raw counts too, so a different
# threshold can be applied to the recorded numbers without a re-run.
DECISION_MARGIN_S = 1.0


def _signals(f: dict, lap: int) -> dict:
    """The three call signals as the engine stated them on one car-lap."""
    win = f.get("window") or [None, None]
    lo, hi = win[0], win[1]
    box = f.get("box_now")
    return {"window": bool(lo is not None and hi is not None and lo <= lap <= hi),
            "box_now": bool(box is not None and np.isfinite(box) and box <= BOX_NOW_MAX_S),
            "collapse": bool(f.get("alarm"))}


def signal_scores(by_lap: dict, stops: dict) -> dict:
    """Precision and recall per signal over every car-lap of the replay."""
    names = ("window", "box_now", "collapse")
    tp = {n: 0 for n in names}
    fp = {n: 0 for n in names}
    fired_before = {n: set() for n in names}
    n_car_laps = 0
    for lap, field in by_lap.items():
        for drv, f in (field or {}).items():
            n_car_laps += 1
            nxt = [p for p in stops.get(drv, []) if 1 <= p - lap <= LOOKAHEAD]
            sig = _signals(f, int(lap))
            for n in names:
                if not sig[n]:
                    continue
                if nxt:
                    tp[n] += 1
                    fired_before[n].add((drv, nxt[0]))
                else:
                    fp[n] += 1
    n_stops = sum(len(v) for v in stops.values())
    out = {"n_car_laps": n_car_laps, "n_stops": n_stops, "lookahead_laps": LOOKAHEAD}
    for n in names:
        fired = tp[n] + fp[n]
        out[n] = {"n_signals": fired, "tp": tp[n], "fp": fp[n],
                  "precision": (tp[n] / fired if fired else None),
                  "stops_called": len(fired_before[n]),
                  "recall": (len(fired_before[n]) / n_stops if n_stops else None)}
    return out


def haas_stop_rows(per: pd.DataFrame) -> dict:
    """`stops_haas`: the two Haas cars' real stops against the engine's call.

    The same four numbers the pooled `stops` block reports — how many stops,
    the share called within three laps, the median absolute error and the
    box-now cost the lap before — restricted to OCO and BEA, plus each car on
    its own, because a team benchmark that pools twenty cars says nothing about
    the two the engineer is sitting behind.
    """
    def _rows(g: pd.DataFrame) -> dict:
        err = g["err_laps"] if len(g) else pd.Series(dtype=float)
        box = g["box_now_delta_1_before"] if len(g) else pd.Series(dtype=float)
        return {"n": int(len(g)),
                "n_with_recommendation": int(err.notna().sum()) if len(g) else 0,
                "share_in_window": (float(g["in_window"].mean()) if len(g) else None),
                "share_err_within_3": (float((err.abs() <= 3).mean()) if err.notna().any() else None),
                "median_abs_err_laps": (float(err.abs().median()) if err.notna().any() else None),
                "median_box_now_delta_1_before": (float(box.median()) if box.notna().any() else None),
                "in_laps": [int(x) for x in g["in_lap"]] if len(g) else [],
                "recommended_3_before": [(None if pd.isna(x) else int(x)) for x in g["rec_3_before"]] if len(g) else []}

    empty = pd.DataFrame()
    if not len(per) or "driver" not in per.columns:
        return {"drivers": list(HAAS_DRIVERS), **_rows(empty),
                "by_driver": {d: _rows(empty) for d in HAAS_DRIVERS}}
    return {"drivers": list(HAAS_DRIVERS), **_rows(per[per["driver"].isin(HAAS_DRIVERS)]),
            "by_driver": {d: _rows(per[per["driver"] == d]) for d in HAAS_DRIVERS}}


def _decision_rivals(dec: dict) -> set:
    """The driver codes (or numbers) the decision says it was priced against."""
    out = set()
    for r in (dec.get("rivals") or []):
        if isinstance(r, dict):
            for k in ("driver", "code", "driver_number", "number"):
                if r.get(k) is not None:
                    out.add(str(r[k]))
        elif r is not None:
            out.add(str(r))
    return out


def decision_stability(by_lap: dict, stops: dict, sc_by_lap: dict) -> dict:
    """Action churn: changes of mind nothing in the state accounts for.

    Over every car-lap that carries a `decision` and had one on the previous
    lap, the action either held or changed.  A change is *material* — and so
    not churn — when

      * the car pitted, with an in-lap on this lap or the previous one, or
      * a driver in either lap's `decision["rivals"]` pitted on those laps, or
      * `meta["sc_active"]` changed between the two laps, or
      * the best-vs-runner-up margin `delta_vs_alternative_s` moved by at least
        `DECISION_MARGIN_S`.

    Everything else is an unexplained change.  Task 1's plans carry no
    `decision` key at all, so the block reports `n = 0` and nulls rather than a
    zero that could be mistaken for perfect stability.
    """
    laps = sorted(int(l) for l in by_lap)
    prev: dict = {}
    pairs = changes = unexplained = 0
    n_with_decision = 0
    reasons = {"pit_self": 0, "pit_rival": 0, "sc_change": 0, "margin": 0}
    per_driver: dict = {}
    examples = []
    for lap in laps:
        field = by_lap.get(lap) or {}
        for drv, f in field.items():
            dec = (f or {}).get("decision")
            if not isinstance(dec, dict) or dec.get("action") is None:
                continue
            n_with_decision += 1
            p = prev.get(drv)
            prev[drv] = (lap, dec)
            if p is None or p[0] != lap - 1:
                continue
            plap, pdec = p
            d = per_driver.setdefault(drv, {"pairs": 0, "changes": 0, "unexplained": 0})
            pairs += 1
            d["pairs"] += 1
            if str(dec.get("action")) == str(pdec.get("action")):
                continue
            changes += 1
            d["changes"] += 1
            mine = set(stops.get(drv, []))
            pit_self = bool(mine & {lap, plap})
            rivals = _decision_rivals(dec) | _decision_rivals(pdec)
            pit_rival = any(set(stops.get(r, [])) & {lap, plap} for r in rivals)
            sc_change = bool(sc_by_lap.get(lap)) != bool(sc_by_lap.get(plap))
            a, b = dec.get("delta_vs_alternative_s"), pdec.get("delta_vs_alternative_s")
            margin = (a is not None and b is not None
                      and np.isfinite(float(a)) and np.isfinite(float(b))
                      and abs(float(a) - float(b)) >= DECISION_MARGIN_S)
            for name, hit in (("pit_self", pit_self), ("pit_rival", pit_rival),
                              ("sc_change", sc_change), ("margin", margin)):
                reasons[name] += int(bool(hit))
            if not (pit_self or pit_rival or sc_change or margin):
                unexplained += 1
                d["unexplained"] += 1
                if len(examples) < 10:
                    examples.append({"driver": drv, "lap": lap,
                                     "from": pdec.get("action"), "to": dec.get("action")})
    if not n_with_decision:
        return {"n": 0, "n_pairs": 0, "n_changes": None, "n_unexplained": None,
                "share_changed": None, "share_unexplained": None,
                "share_of_changes_unexplained": None, "by_driver": {}, "reasons": {},
                "margin_s": DECISION_MARGIN_S,
                "note": "the plans carry no decision block (Task 1): nothing to score"}
    for d in per_driver.values():
        d["share_unexplained"] = (d["unexplained"] / d["pairs"]) if d["pairs"] else None
    return {"n": int(n_with_decision), "n_pairs": int(pairs), "n_changes": int(changes),
            "n_unexplained": int(unexplained),
            "share_changed": (changes / pairs if pairs else None),
            "share_unexplained": (unexplained / pairs if pairs else None),
            "share_of_changes_unexplained": (unexplained / changes if changes else None),
            "margin_s": DECISION_MARGIN_S, "reasons": reasons,
            "by_driver": {k: per_driver[k] for k in per_driver if k in HAAS_DRIVERS} or per_driver,
            "haas": {d: per_driver.get(d) for d in HAAS_DRIVERS},
            "examples": examples}


def replay(key: str, race_state: bool = True) -> dict:
    wm = WeekendModel.load(key)
    eng = RaceEngine(wm, race_state=race_state)
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
                                                 "position": f.get("position"),
                                                 # V4 (WP-D): absent on a Task 1 plan, read defensively
                                                 "decision": (f.get("plan") or {}).get("decision")}
                                 for f in snap["field"]}})
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
            # the three laps before the stop, as the engine saw them
            before = [(L, ((by_lap.get(L) or {}).get(drv) or {})) for L in range(p - LOOKAHEAD, p)]
            sig = [_signals(f, L) for L, f in before if f]
            boxes = [f.get("box_now") for _, f in before
                     if f.get("box_now") is not None and np.isfinite(f.get("box_now"))]
            rows.append({"driver": drv, "in_lap": p, "rec_3_before": f3.get("next"),
                         "window_3_before": win, "in_window": (win[0] is not None and win[0] <= p <= (win[1] or -1)),
                         "err_laps": (p - f3["next"]) if f3.get("next") is not None else None,
                         "box_now_delta_1_before": f1.get("box_now"), "p_cliff_1_before": f1.get("p_cliff"),
                         "wear_1_before": f1.get("wear"), "alarm_lap": alarm,
                         # V3: did each signal fire on any of the three laps before the stop?
                         "window_signal": bool(any(s["window"] for s in sig)),
                         "box_now_signal": bool(any(s["box_now"] for s in sig)),
                         "collapse_signal": bool(any(s["collapse"] for s in sig)),
                         "box_now_min_3_before": (float(min(boxes)) if boxes else None),
                         "position_at_stop": f1.get("position")})
    per = pd.DataFrame(rows)
    scores = signal_scores(by_lap, stops)
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
                  "median_wear_1_before": float(per["wear_1_before"].median()) if per["wear_1_before"].notna().any() else None,
                  # V3: each signal's recall over the real stops, 3-lap lookahead
                  "share_window_signal": float(per["window_signal"].mean()) if len(per) else None,
                  "share_box_now_signal": float(per["box_now_signal"].mean()) if len(per) else None,
                  "share_collapse_signal": float(per["collapse_signal"].mean()) if len(per) else None,
                  "median_box_now_min_3_before": (float(per["box_now_min_3_before"].median())
                                                  if per["box_now_min_3_before"].notna().any() else None)},
        "signals": scores,
        "stops_top10": {"n": int(len(per_top)), "share_in_window": float(per_top["in_window"].mean()) if len(per_top) else None,
                        "median_abs_err_laps": float(per_top["err_laps"].abs().median()) if len(per_top) and per_top["err_laps"].notna().any() else None},
        # V4: the two Haas cars, and whether the recommended action holds
        "stops_haas": haas_stop_rows(per),
        "decision_stability": decision_stability(by_lap, stops, {h["lap"]: h["sc"] for h in hist}),
        "regime": {"self_measured_offline": m_self, "prior_mean": hist[0]["m"]["prior_mean"] if hist else None,
                   "by_lap": m_traj[::5] + [m_traj[-1]], "final": m_traj[-1] if m_traj else None},
        "pit_loss_by_lap": [(h["lap"], round(h["pit_loss"], 2)) for h in hist][::10],
        "per_stop": per.to_dict("records"),
        "latency_by_lap": lat.to_dict("records"),
    }
    print(f"\n== {key}: {len(ticks)} ticks in {total:.1f}s (parse {t_parse:.1f}s); tick ms mean {out['tick_ms']['mean']} "
          f"median {out['tick_ms']['median']} p95 {out['tick_ms']['p95']} max {out['tick_ms']['max']}")
    print(f"   stops: {out['stops']}")
    print(f"   signals over {scores['n_car_laps']} car-laps (stop within {LOOKAHEAD} laps = true positive): "
          + " | ".join(f"{n} prec {('%.2f' % scores[n]['precision']) if scores[n]['precision'] is not None else '-'}"
                       f" recall {('%.2f' % scores[n]['recall']) if scores[n]['recall'] is not None else '-'}"
                       f" ({scores[n]['n_signals']} fired)" for n in ("window", "box_now", "collapse")))
    print(f"   top10 stops: {out['stops_top10']}")
    print(f"   haas stops: {[(k, v) for k, v in out['stops_haas'].items() if k not in ('by_driver',)]}")
    print(f"   decision stability: { {k: v for k, v in out['decision_stability'].items() if k not in ('examples', 'by_driver', 'haas')} }")
    print(f"   regime m: prior {out['regime']['prior_mean']}, final {out['regime']['final']}, offline self-measured {m_self}")
    print(f"   m by lap: {m_traj[::10]}")
    return out


def main() -> None:
    args = arg_events(__doc__, extra=[(("--no-race-state",), {"action": "store_true",
                                       "help": "the V3 objective (race-state term off): writes live_no_race_state.json"})])
    offline()        # the replay is a recorded archive; nothing here may reach the network
    keys = [k for k in args.events if k in ARCHIVES] or list(ARCHIVES)
    missing = [k for k in keys if not ARCHIVES[k].exists()]
    if missing:
        print(f"  no archived feed for {missing}; skipped")
    res = {k: replay(k, race_state=not args.no_race_state) for k in keys if ARCHIVES[k].exists()}
    dump("live_no_race_state.json" if args.no_race_state else "live.json", res)


if __name__ == "__main__":
    main()
