"""V4 Task 1 against V3, by the V3 report's own metric definitions.

Reads what the unchanged benchmark scripts wrote - V3's under `bench/v3/out/`
(the V3 code, re-run on this machine), V4's under `bench/out/` - plus the live
ablation (`bench/out/live_no_race_state.json`) and each weekend's
`meta["race_state"]`.  Nothing here re-scores anything: every number is the
same field of the same JSON, pooled the way `bench_compare` pools it.

    first-stop error     mean |recommended first stop - field green-flag median|,
                         the three non-safety-car weekends (bench_strategy)
    window share         mean over weekends of the share of the field's green
                         first stops inside the model's stop-1 window
    oracle regret        mean over the seven weekends of the tool's regret
    sequence / start /   counts of seven: run by anyone, the majority's start,
    stop count           the field's modal stop count
    live                 mean over the two replays: stops inside the window 3
                         laps before, within 3 laps, median |error|, box-now cost
                         the lap before, tick mean / p95, signal precision/recall

Writes bench/out/v4_compare.json and prints the tables.
"""

from __future__ import annotations

import json

import numpy as np

from common import NON_SC_EVENTS, OUT, ROOT, dump  # noqa: E402
from src.config import DATA_PROCESSED

V3_OUT = ROOT / "bench" / "v3" / "out"
V3_PROC = ROOT / "bench" / "v3" / "processed"
EVENTS = ["australia-2026", "japan-2026", "barcelona-2026", "austria-2026", "belgium-2026",
          "hungary-2026", "italy-2026"]
LIVE = ["hungary-2026", "barcelona-2026"]


def _read(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _mean(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return float(np.mean(xs)) if xs else None


def strategy_metrics(s: dict) -> dict:
    fs = {k: (s[k]["first_stop"] or {}).get("rec_minus_field") for k in NON_SC_EVENTS if k in s}
    win = {k: ((s[k].get("pit_windows") or [{}])[0] or {}).get("share_inside") for k in EVENTS if k in s}
    reg = {k: (s[k].get("oracle") or {}).get("regret_s", {}).get("tool") for k in EVENTS if k in s}
    return {
        "first_stop_error_laps": _mean([abs(v) for v in fs.values() if v is not None]),
        "first_stop_signed_laps": _mean(fs.values()),
        "first_stop_by_event": fs,
        "window_share": _mean(win.values()), "window_share_by_event": win,
        "oracle_regret_s": _mean(reg.values()), "oracle_regret_by_event": reg,
        "regret_field_modal_s": _mean([(s[k].get("oracle") or {}).get("regret_s", {}).get("field_modal") for k in s]),
        "seq_run_by_anyone": sum(bool(s[k].get("rec_seq_run_by_anyone")) for k in EVENTS if k in s),
        "start_matches_majority": sum(bool(s[k].get("start_matches_majority")) for k in EVENTS if k in s),
        "stops_match_mode": sum(bool(s[k].get("stops_match_mode")) for k in EVENTS if k in s),
        "mean_seq_share": _mean([s[k].get("rec_seq_share") for k in EVENTS if k in s]),
        "plans": {k: s[k].get("recommended") for k in EVENTS if k in s},
        "per_car_same_shape": _mean([(s[k].get("per_driver") or {}).get("share_same_shape_as_field") for k in s]),
        "per_car_first_stop_spread": _mean([(s[k].get("per_driver") or {}).get("first_stop_spread") for k in s]),
        "counterfactual_over_30s": sum(int((s[k].get("counterfactual") or {}).get("n_over_30s") or 0) for k in s),
    }


def live_metrics(L: dict | None) -> dict | None:
    if not L:
        return None
    ev = [k for k in LIVE if k in L]
    g = lambda f: _mean([f(L[k]) for k in ev])  # noqa: E731
    out = {"tick_mean_ms": g(lambda v: v["tick_ms"]["mean"]), "tick_p95_ms": g(lambda v: v["tick_ms"]["p95"]),
           "tick_max_ms": g(lambda v: v["tick_ms"]["max"]),
           "share_in_window": g(lambda v: v["stops"]["share_in_window"]),
           "share_within_3": g(lambda v: v["stops"]["share_err_within_3"]),
           "median_abs_err_laps": g(lambda v: v["stops"]["median_abs_err_laps"]),
           "box_now_1_before_s": g(lambda v: v["stops"]["median_box_now_delta_1_before"]),
           "top10_median_abs_err": g(lambda v: (v.get("stops_top10") or {}).get("median_abs_err_laps")),
           "top10_share_in_window": g(lambda v: (v.get("stops_top10") or {}).get("share_in_window"))}
    for sig in ("window", "box_now", "collapse"):
        out[f"{sig}_precision"] = g(lambda v: v["signals"][sig]["precision"])
        out[f"{sig}_recall"] = g(lambda v: v["signals"][sig]["recall"])
    out["by_event"] = {k: {"tick_mean_ms": L[k]["tick_ms"]["mean"], "tick_p95_ms": L[k]["tick_ms"]["p95"],
                           **{kk: L[k]["stops"].get(kk) for kk in ("share_in_window", "share_err_within_3",
                                                                    "median_abs_err_laps", "median_box_now_delta_1_before")},
                           "window_pr": [L[k]["signals"]["window"]["precision"], L[k]["signals"]["window"]["recall"]],
                           "box_now_pr": [L[k]["signals"]["box_now"]["precision"], L[k]["signals"]["box_now"]["recall"]]}
                       for k in ev}
    return out


def first_stop_table(s3: dict, s4: dict) -> list:
    rows = []
    for k in EVENTS:
        m4 = _read(DATA_PROCESSED / f"meta_{k}.json") or {}
        rs = m4.get("race_state") or {}
        f3 = (s3.get(k) or {}).get("first_stop") or {}
        f4 = (s4.get(k) or {}).get("first_stop") or {}
        w3 = ((s3.get(k) or {}).get("pit_windows") or [{}])[0] or {}
        w4 = ((s4.get(k) or {}).get("pit_windows") or [{}])[0] or {}
        rows.append({"event": k, "sc_set": f4.get("sc_set"), "field_median_green": f4.get("field_median_green"),
                     "v3_plan": (s3.get(k) or {}).get("recommended"), "v4_plan": (s4.get(k) or {}).get("recommended"),
                     "v3_first": f3.get("recommended"), "v4_first": f4.get("recommended"),
                     "v3_err": f3.get("rec_minus_field"), "v4_err": f4.get("rec_minus_field"),
                     "v3_window": [w3.get("lo"), w3.get("hi")], "v4_window": [w4.get("lo"), w4.get("hi")],
                     "v3_window_share": w3.get("share_inside"), "v4_window_share": w4.get("share_inside"),
                     "tyre_first_in_group": rs.get("tyre_optimal_first_stop_in_group"),
                     "pack_median": rs.get("pack_first_stop_median"), "pack_iqr": rs.get("pack_first_stop_iqr"),
                     "place_value_s": (rs.get("constants") or {}).get("place_value_s"),
                     "persistence": (rs.get("constants") or {}).get("persistence"),
                     "place_gap_s": (rs.get("constants") or {}).get("place_gap_s"),
                     "sigma_rel_s": (rs.get("constants") or {}).get("sigma_rel_s"),
                     "regret_v3": (s3.get(k) or {}).get("oracle", {}).get("regret_s", {}).get("tool"),
                     "regret_v4": (s4.get(k) or {}).get("oracle", {}).get("regret_s", {}).get("tool"),
                     "oracle_best": (s4.get(k) or {}).get("oracle", {}).get("oracle_best")})
    return rows


def main() -> None:
    s3, s4 = _read(V3_OUT / "strategy.json"), _read(OUT / "strategy.json")
    l3, l4, l4off = _read(V3_OUT / "live.json"), _read(OUT / "live.json"), _read(OUT / "live_no_race_state.json")
    a4 = _read(OUT / "ablation.json") or {}
    a3 = _read(V3_OUT / "ablation.json") or {}
    if not (s3 and s4):
        raise SystemExit("need bench/v3/out/strategy.json and bench/out/strategy.json")
    out = {"strategy": {"v3": strategy_metrics(s3), "v4": strategy_metrics(s4)},
           "live": {"v3": live_metrics(l3), "v4": live_metrics(l4), "v4_no_race_state": live_metrics(l4off)},
           "ablation_pooled": {"v4": a4.get("_pooled"), "v3": a3.get("_pooled")},
           "first_stop": first_stop_table(s3, s4)}
    pt3, pt4 = _read(V3_OUT / "runtime.json") or {}, _read(OUT / "runtime.json") or {}
    out["runtime"] = {"v3": pt3, "v4": pt4}
    dump("v4_compare.json", out)

    def row(name, a, b, nd=3):
        f = (lambda x: "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x)))
        print(f"| {name} | {f(a)} | {f(b)} |")

    S3, S4 = out["strategy"]["v3"], out["strategy"]["v4"]
    print("| Metric | V3 | V4 |\n|---|---|---|")
    row("Mean |first stop - field green median| (laps, non-SC)", S3["first_stop_error_laps"], S4["first_stop_error_laps"])
    row("Share of field first stops inside the window", S3["window_share"], S4["window_share"])
    row("Oracle regret of the tool's plan (s)", S3["oracle_regret_s"], S4["oracle_regret_s"])
    row("Sequence run by anyone (of 7)", S3["seq_run_by_anyone"], S4["seq_run_by_anyone"])
    row("Start compound = majority (of 7)", S3["start_matches_majority"], S4["start_matches_majority"])
    row("Stop count = mode (of 7)", S3["stops_match_mode"], S4["stops_match_mode"])
    row("Mean field share on the recommended sequence", S3["mean_seq_share"], S4["mean_seq_share"])
    for tag, L in (("V3", out["live"]["v3"]), ("V4", out["live"]["v4"]), ("V4 no race state", out["live"]["v4_no_race_state"])):
        if L:
            print(f"live {tag}: " + ", ".join(f"{k} {v:.3f}" for k, v in L.items() if isinstance(v, float)))
    print("\nfirst stops:")
    for r in out["first_stop"]:
        print(r)


if __name__ == "__main__":
    main()
