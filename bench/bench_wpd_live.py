"""WP-D: the live race-execution experiments, on the two archived races.

`bench_live.py` scores the calls; this script asks the questions that choosing
the live mechanism needs answered, using `bench_live.replay` itself for the
scoring so the numbers are the benchmark's, not a second implementation:

* **how many rivals** (`--k 3 4 5 6`): the stop-call accuracy and the tick time
  at each cap, to pick the smallest `k` that is not worse;
* **which rivals** (`--selection`): Task 1's virtual-gap set, the V4 set without
  the reachability filter, and the V4 set as shipped;
* **how sticky the call is** (`--hysteresis 0.0 0.3 0.6`): the share of car-laps
  on which the call changes with no material state change, against the accuracy
  metrics, for each margin;
* **how good the rejoin projection is**: for every real green stop, the rejoin
  position the engine projected on the lap of the stop against the car's actual
  position two laps after it rejoined - for the projection and for Task 1's
  gap arithmetic, both recorded on the same replay
  (`decision.rejoin_if_now` and `decision.rejoin_if_now_gaps`).

Usage: python bench/bench_wpd_live.py [--events hungary-2026 barcelona-2026]
                                      [--k 3 4 5 6] [--hysteresis 0.0 0.3 0.6] [--selection]
Writes bench/out/wpd_live.json.  Nothing here writes outside bench/out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import arg_events, dump, offline  # noqa: E402

import bench_live  # noqa: E402
from src import racestate
from src.live import engine as E
from src.live import rivals as R

REC: list = []


def _record_ticks() -> None:
    """Keep, per car-lap, the decision the engine made (it is not in `bench_live`'s
    history, which predates `plan["decision"]`)."""
    if getattr(E.RaceEngine, "_wpd_wrapped", False):
        return
    orig = E.RaceEngine.tick

    def tick(self, state, **kw):
        snap = orig(self, state, **kw)
        lap = (snap.get("meta") or {}).get("lap_count", {}).get("current")
        for r in snap.get("field", []):
            d = (r.get("plan") or {}).get("decision") or {}
            rj = d.get("rejoin_if_now") or {}
            rjg = d.get("rejoin_if_now_gaps") or {}
            REC.append({"lap": lap, "driver": r.get("driver"), "num": r.get("driver_number"),
                        "team": r.get("team"), "position": r.get("position"),
                        "kind": d.get("kind") or d.get("action_kind"), "target": d.get("lap"),
                        "state_change": bool(d.get("state_change")),
                        "held": bool(d.get("held_by_hysteresis")),
                        "conf": d.get("confidence"), "n_rivals": len(d.get("rivals") or []),
                        "projected": rj.get("rejoin_position"), "gaps": rjg.get("rejoin_position"),
                        "has_decision": bool(d)})
        return snap

    E.RaceEngine.tick = tick
    E.RaceEngine._wpd_wrapped = True


class _Task1Rival:
    """Task 1's rival triple, in the shape `strategic_rivals` returns."""

    def __init__(self, num, gap, v, code):
        self.number, self.gap_s, self.virtual_gap_s, self.code = num, gap, v, code
        self.cycle_gap_s, self.score, self.on_track = v, abs(v), None
        self.why = "within 6 s in virtual position (the Task 1 rule)"

    def as_triple(self):
        return (self.number, self.gap_s, self.virtual_gap_s)


def task1_selection(me, cars, order_rows, pit_loss_s, sigma_rel_s, *, k=racestate.N_RIVALS, **kw):
    same = {n: c for n, c in cars.items() if abs(c.cur_lap - me.cur_lap) <= 1}
    return [_Task1Rival(n, g, v, same[n].code)
            for n, g, v in racestate.relevant_rivals(me, same, pit_loss_s, k=k)]


def stability(rec: pd.DataFrame) -> dict:
    """How often the call changes, and how often it changes for no reason.

    The *call* is the instruction, not the label: PIT NOW and WAIT k are the
    lap, and BOX BY LAP x is "not yet" - the edge of the window slides by a lap
    every lap, and counting that as a new decision would report a flip on nine
    car-laps in ten while the instruction never changed."""
    d = rec[rec["has_decision"]].copy()
    if d.empty:
        return {"n_car_laps": 0}
    d["call"] = [f"{k}:{t}" if k in ("PIT_NOW", "WAIT") else str(k)
                 for k, t in zip(d["kind"], d["target"])]
    d = d.sort_values(["num", "lap"])
    prev = d.groupby("num")["call"].shift()
    has_prev = prev.notna()
    changed = has_prev & (d["call"] != prev)
    quiet = ~d["state_change"].astype(bool)
    return {"n_car_laps": int(has_prev.sum()),
            "call_changed": float(changed[has_prev].mean()),
            "call_changed_no_state_change": float((changed & quiet)[has_prev & quiet].mean()),
            "held_by_hysteresis": float(d["held"].mean()),
            "mean_confidence": float(d["conf"].dropna().mean()),
            "mean_n_rivals": float(d["n_rivals"].mean()),
            "calls": {str(k): int(v) for k, v in d["kind"].value_counts().items()}}


def rejoin_quality(rec: pd.DataFrame, per_stop: list) -> dict:
    """The projected rejoin position against what happened.

    For a stop with in-lap `p` the car rejoins on lap `p + 1`, so "two laps
    after the rejoin" is its position on lap `p + 3`.  The engine's projection
    for boxing on lap `p` is the one it published on the tick of lap `p`."""
    d = rec[rec["has_decision"]]
    proj = {(n, l): (a, b) for n, l, a, b in zip(d["num"], d["lap"], d["projected"], d["gaps"])}
    pos = {(n, l): p for n, l, p in zip(rec["num"], rec["lap"], rec["position"])}
    num_of = dict(zip(rec["driver"], rec["num"]))
    rows = []
    for s in per_stop:
        n, p = num_of.get(s["driver"]), int(s["in_lap"])
        r = proj.get((n, p), proj.get((n, p + 1)))
        actual = pos.get((n, p + 3))
        if r is None or actual is None or not np.isfinite(actual):
            continue
        rows.append({"driver": s["driver"], "in_lap": p, "actual": float(actual),
                     "projected": r[0], "gaps": r[1]})
    df = pd.DataFrame(rows)
    out = {"n": int(len(df)), "rows": rows}
    for key, name in (("projected", "projected"), ("gaps", "todays_gaps")):
        if len(df) and df[key].notna().any():
            e = (df[key] - df["actual"]).dropna()
            out[name] = {"n": int(len(e)), "median_abs_err": float(e.abs().median()),
                         "mean_abs_err": float(e.abs().mean()),
                         "median_signed_err": float(e.median()),
                         "share_within_1": float((e.abs() <= 1).mean()),
                         "share_within_2": float((e.abs() <= 2).mean())}
    return out


def arm(label: str, events: list, *, k: int | None = None, hysteresis: float | None = None,
        select=None, band_z: float | None = None) -> dict:
    """One setting, both replays: `bench_live`'s own scoring plus the two WP-D rows."""
    if k is not None:
        E.STRATEGIC_RIVALS_K = int(k)
    if hysteresis is not None:
        E.DECISION_HYSTERESIS_S = float(hysteresis)
    if band_z is not None:
        R.INTERACTION_BAND_Z = float(band_z)
    E.rivals_mod.strategic_rivals = select or R.strategic_rivals
    out = {}
    for key in events:
        REC.clear()
        r = bench_live.replay(key)
        rec = pd.DataFrame(REC)
        out[key] = {"stops": r["stops"], "tick_ms": r["tick_ms"],
                    "signals": {n: {m: r["signals"][n][m] for m in ("precision", "recall")}
                                for n in ("window", "box_now", "collapse")},
                    "stability": stability(rec), "rejoin": rejoin_quality(rec, r["per_stop"])}
        s, st = out[key]["stops"], out[key]["stability"]
        rj = out[key]["rejoin"]
        print(f"  {label:18s} {key:16s} within3 {s['share_err_within_3']:.3f} "
              f"median {s['median_abs_err_laps']} window {s['share_in_window']:.3f} "
              f"box-now {s['median_box_now_delta_1_before']:.3f} tick {r['tick_ms']['median']:.0f} ms "
              f"rivals {st['mean_n_rivals']:.2f} call-changed {st['call_changed']:.3f} "
              f"(quiet {st['call_changed_no_state_change']:.3f}) held {st['held_by_hysteresis']:.3f} "
              f"rejoin |err| {(rj.get('projected') or {}).get('mean_abs_err', float('nan')):.2f} "
              f"vs gaps {(rj.get('todays_gaps') or {}).get('mean_abs_err', float('nan')):.2f}", flush=True)
    return out


def main() -> None:
    args = arg_events(__doc__, extra=[
        (("--k",), {"nargs": "+", "type": int, "default": [], "help": "rival caps to compare"}),
        (("--hysteresis",), {"nargs": "+", "type": float, "default": [],
                             "help": "decision hold margins (s) to compare"}),
        (("--selection",), {"action": "store_true", "help": "compare the rival-selection rules"}),
    ])
    offline()
    _record_ticks()
    events = [k for k in args.events if k in bench_live.ARCHIVES and bench_live.ARCHIVES[k].exists()]
    if not events:
        events = [k for k, p in bench_live.ARCHIVES.items() if p.exists()]
    k0, h0, z0 = E.STRATEGIC_RIVALS_K, E.DECISION_HYSTERESIS_S, R.INTERACTION_BAND_Z
    res = {"shipped": arm(f"shipped (k={k0})", events, k=k0, hysteresis=h0)}
    for k in args.k:
        res[f"k={k}"] = arm(f"k={k}", events, k=k, hysteresis=h0)
    for h in args.hysteresis:
        res[f"hysteresis={h}"] = arm(f"hysteresis={h}", events, k=k0, hysteresis=h)
    if args.selection:
        res["selection=task1"] = arm("selection task1", events, k=k0, hysteresis=h0,
                                     select=task1_selection)
        res["selection=no_reach_filter"] = arm("no reach filter", events, k=k0, hysteresis=h0,
                                               band_z=1e9)
        R.INTERACTION_BAND_Z = z0
    dump("wpd_live.json", res)


if __name__ == "__main__":
    main()
