"""V4 final against V4 Task 1 and V3, by the report's own metric definitions.

Three columns, three builds, all of them read off what the unchanged benchmark
scripts wrote:

    V3        `bench/v3/out/` + `bench/v3/processed/`   (the V3 code, re-run here)
    Task 1    `bench/v4_task1/out/` + `bench/v4_task1/processed/`  (frozen)
    V4        `bench/out/` + `data/processed/`          (whatever is on disk now)

plus the live ablation (`bench/out/live_no_race_state.json`).  Nothing here
re-scores anything: every number is the same field of the same JSON, pooled the
way `bench_compare` pools it.

    first-stop error     mean |recommended first stop - field green-flag median|,
                         the three non-safety-car weekends (bench_strategy)
    window share         mean over weekends of the share of the field's green
                         first stops inside the model's stop-1 window
    oracle regret        mean over the seven weekends of the tool's regret -
                         pure race time, on a tyre model that knows this race
    R_pos                the same regret with the first pit cycle's track
                         position priced in (`docs/v4_methodology.md`): reported
                         for the tool, the tyre-optimal plan and the field's
                         modal plan, with L(tool) and P_retain
    sequence / start /   counts of seven: run by anyone, the majority's start,
    stop count           the field's modal stop count
    Haas                 OCO and BEA: the per-car plan's first stop against the
                         driver's own green first stop, the shape match, L, and
                         live, the two cars' own stop calls
    live                 mean over the two replays: stops inside the window 3
                         laps before, within 3 laps, median |error|, box-now cost
                         the lap before, tick mean / p95, signal precision/recall,
                         and the share of action changes nothing explains
    estimators           the per-fold place-value table from
                         `bench/out/place_value.json` (WP-A), if it has run
    experiments          the E0-E8 rows from `bench/out/experiments.json`
                         (integration), if it has run

Writes bench/out/v4_compare.json and prints the tables as Markdown.
"""

from __future__ import annotations

import json

import numpy as np

from common import NON_SC_EVENTS, OUT, ROOT, dump  # noqa: E402
from src.config import DATA_PROCESSED

V3_OUT = ROOT / "bench" / "v3" / "out"
V3_PROC = ROOT / "bench" / "v3" / "processed"
T1_OUT = ROOT / "bench" / "v4_task1" / "out"
T1_PROC = ROOT / "bench" / "v4_task1" / "processed"
EVENTS = ["australia-2026", "japan-2026", "barcelona-2026", "austria-2026", "belgium-2026",
          "hungary-2026", "italy-2026"]
LIVE = ["hungary-2026", "barcelona-2026"]
HAAS_DRIVERS = ("OCO", "BEA")

# Fields worth carrying out of a place-value fold, if WP-A's file has them.
PV_FIELDS = ("place_value_s", "place_gap_s", "place_gap_lead_lap_s", "persistence",
             "persistence_raw", "place_value_sd_s", "place_value_ci_s", "sigma_rel_s",
             "n_cycle_pairs", "n_finish_gaps", "pairs", "kept", "n_pairs")


def _read(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _mean(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return float(np.mean(xs)) if xs else None


def _pos(s: dict, k: str) -> dict:
    """A weekend's `position_aware` block, or {} on a build that has none."""
    return (((s.get(k) or {}).get("oracle") or {}).get("position_aware") or {})


def _haas(s: dict, k: str) -> dict:
    return ((s.get(k) or {}).get("haas") or {})


def position_metrics(s: dict) -> dict:
    """The position-aware rows, pooled over the weekends that have an `L`.

    A build with no `position_aware` block (V3, Task 1 as frozen) returns Nones,
    which is what the comparison prints as "-": the metric is new, not worse.
    """
    have = [k for k in EVENTS if _pos(s, k).get("R_pos")]
    rp = {lab: {k: _pos(s, k)["R_pos"].get(lab) for k in have} for lab in
          ("tool", "tyre_optimal", "field_modal", "winner", "oracle_opt")}
    L = {lab: {k: (_pos(s, k).get("L") or {}).get(lab) for k in have} for lab in
         ("tool", "field_modal", "tyre_optimal")}
    return {
        "n_weekends": len([k for k in have if rp["tool"].get(k) is not None]),
        "rpos_tool_s": _mean(rp["tool"].values()),
        "rpos_tyre_optimal_s": _mean(rp["tyre_optimal"].values()),
        "rpos_field_modal_s": _mean(rp["field_modal"].values()),
        "rpos_winner_s": _mean(rp["winner"].values()),
        "L_tool": _mean(L["tool"].values()),
        "L_field_modal": _mean(L["field_modal"].values()),
        "L_tyre_optimal": _mean(L["tyre_optimal"].values()),
        "P_retain_tool": _mean([_pos(s, k).get("P_retain_tool") for k in have]),
        "place_value_s": _mean([_pos(s, k).get("V_s") for k in have]),
        "estimator": next((_pos(s, k).get("estimator") for k in have
                           if _pos(s, k).get("estimator")), None),
        "rpos_tool_by_event": rp["tool"], "L_tool_by_event": L["tool"],
        "best_candidate_by_event": {k: _pos(s, k).get("best_candidate") for k in have},
    }


def haas_metrics(s: dict) -> dict:
    """OCO and BEA pre-race: first-stop error, shape match and L per weekend."""
    have = [k for k in EVENTS if _haas(s, k).get("drivers")]
    rows = {}
    for k in have:
        for drv, d in (_haas(s, k).get("drivers") or {}).items():
            rows.setdefault(drv, {})[k] = d
    errs = [d.get("first_minus_actual_green") for v in rows.values() for d in v.values()]
    shape = [d.get("same_shape_as_field") for v in rows.values() for d in v.values()
             if d.get("same_shape_as_field") is not None]
    Ls = [d.get("L") for v in rows.values() for d in v.values()]
    return {
        "n_weekends": len(have), "n_car_weekends": sum(len(v) for v in rows.values()),
        "n_scored_green": len([e for e in errs if e is not None]),
        "first_stop_error_laps": _mean([abs(e) for e in errs if e is not None]),
        "first_stop_signed_laps": _mean([e for e in errs if e is not None]),
        "share_same_shape_as_field": (float(np.mean([bool(x) for x in shape])) if shape else None),
        "L_mean": _mean(Ls),
        "by_event": {k: {drv: {kk: (rows.get(drv, {}).get(k) or {}).get(kk)
                               for kk in ("plan", "first_stop", "actual_first_stop",
                                          "actual_first_sc", "first_minus_actual_green",
                                          "same_shape_as_field", "L")}
                         for drv in HAAS_DRIVERS} for k in have},
    }


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
        # V4 (WP-C): additive, and None on a build whose bench_strategy did not write them
        "position_aware": position_metrics(s),
        "haas": haas_metrics(s),
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
    # V4 (WP-C2): the two Haas cars' own stop calls, and decision churn.  Both
    # blocks are read defensively: a Task 1 live.json has neither.
    have_haas = [k for k in ev if isinstance(L[k].get("stops_haas"), dict)]
    have_dec = [k for k in ev if isinstance(L[k].get("decision_stability"), dict)]
    out["stops_haas"] = {
        "n": (sum(int((L[k]["stops_haas"].get("n") or 0)) for k in have_haas) if have_haas else None),
        "share_in_window": g(lambda v: (v.get("stops_haas") or {}).get("share_in_window")),
        "share_within_3": g(lambda v: (v.get("stops_haas") or {}).get("share_err_within_3")),
        "median_abs_err_laps": g(lambda v: (v.get("stops_haas") or {}).get("median_abs_err_laps")),
        "box_now_1_before_s": g(lambda v: (v.get("stops_haas") or {}).get("median_box_now_delta_1_before")),
        "by_event": {k: {kk: (L[k].get("stops_haas") or {}).get(kk)
                         for kk in ("n", "share_in_window", "share_err_within_3",
                                    "median_abs_err_laps", "median_box_now_delta_1_before")}
                     for k in ev},
    }
    n_dec = sum(int((L[k]["decision_stability"].get("n") or 0)) for k in have_dec)
    out["decision_stability"] = {
        "n": (n_dec if have_dec else None),
        "n_pairs": (sum(int((L[k]["decision_stability"].get("n_pairs") or 0)) for k in have_dec)
                    if have_dec else None),
        "n_changes": (sum(int((L[k]["decision_stability"].get("n_changes") or 0)) for k in have_dec)
                      if n_dec else None),
        "n_unexplained": (sum(int((L[k]["decision_stability"].get("n_unexplained") or 0)) for k in have_dec)
                          if n_dec else None),
        "share_changed": g(lambda v: (v.get("decision_stability") or {}).get("share_changed")),
        "share_unexplained": g(lambda v: (v.get("decision_stability") or {}).get("share_unexplained")),
        "share_of_changes_unexplained": g(lambda v: (v.get("decision_stability") or {}).get("share_of_changes_unexplained")),
        "by_event": {k: {kk: (L[k].get("decision_stability") or {}).get(kk)
                         for kk in ("n", "n_pairs", "n_changes", "n_unexplained",
                                    "share_unexplained", "note")} for k in ev},
    }
    out["by_event"] = {k: {"tick_mean_ms": L[k]["tick_ms"]["mean"], "tick_p95_ms": L[k]["tick_ms"]["p95"],
                           **{kk: L[k]["stops"].get(kk) for kk in ("share_in_window", "share_err_within_3",
                                                                    "median_abs_err_laps", "median_box_now_delta_1_before")},
                           "window_pr": [L[k]["signals"]["window"]["precision"], L[k]["signals"]["window"]["recall"]],
                           "box_now_pr": [L[k]["signals"]["box_now"]["precision"], L[k]["signals"]["box_now"]["recall"]]}
                       for k in ev}
    return out


def first_stop_table(s3: dict, s1: dict, s4: dict) -> list:
    """One row per weekend: the three builds' first stop, window and regret.

    The race-state numbers come from each build's own `meta_*.json` - V4's from
    `data/processed/`, Task 1's from its frozen copy - so the row says what that
    build actually decided on, not what the current pipeline would decide.
    """
    rows = []
    for k in EVENTS:
        m4 = _read(DATA_PROCESSED / f"meta_{k}.json") or {}
        m1 = _read(T1_PROC / f"meta_{k}.json") or {}
        rs, rs1 = m4.get("race_state") or {}, m1.get("race_state") or {}
        f3 = (s3.get(k) or {}).get("first_stop") or {}
        f1 = ((s1 or {}).get(k) or {}).get("first_stop") or {}
        f4 = (s4.get(k) or {}).get("first_stop") or {}
        w3 = ((s3.get(k) or {}).get("pit_windows") or [{}])[0] or {}
        w1 = (((s1 or {}).get(k) or {}).get("pit_windows") or [{}])[0] or {}
        w4 = ((s4.get(k) or {}).get("pit_windows") or [{}])[0] or {}
        p4 = _pos(s4, k)
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
                     "oracle_best": (s4.get(k) or {}).get("oracle", {}).get("oracle_best"),
                     # V4 final vs Task 1
                     "task1_plan": ((s1 or {}).get(k) or {}).get("recommended"),
                     "task1_first": f1.get("recommended"), "task1_err": f1.get("rec_minus_field"),
                     "task1_window": [w1.get("lo"), w1.get("hi")],
                     "task1_window_share": w1.get("share_inside"),
                     "task1_regret": ((s1 or {}).get(k) or {}).get("oracle", {}).get("regret_s", {}).get("tool"),
                     "task1_place_value_s": (rs1.get("constants") or {}).get("place_value_s"),
                     "task1_pack_median": rs1.get("pack_first_stop_median"),
                     # the position-aware metric (V4 only; the frozen builds have none)
                     "rpos_tool": (p4.get("R_pos") or {}).get("tool"),
                     "rpos_tyre_optimal": (p4.get("R_pos") or {}).get("tyre_optimal"),
                     "rpos_field_modal": (p4.get("R_pos") or {}).get("field_modal"),
                     "L_tool": (p4.get("L") or {}).get("tool"),
                     "L_field_modal": (p4.get("L") or {}).get("field_modal"),
                     "P_retain_tool": p4.get("P_retain_tool"),
                     "pos_best_candidate": p4.get("best_candidate"),
                     "pos_V_s": p4.get("V_s"), "pos_n_field_stops": p4.get("n_field_first_stops")})
    return rows


def _scalars(d: dict) -> dict:
    """A row's scalar fields, in the writer's own order - nothing interpreted."""
    return {k: v for k, v in d.items()
            if v is None or isinstance(v, (int, float, str, bool))}


def place_value_rows(pv) -> list | None:
    """The per-fold estimator table from `bench/out/place_value.json` (WP-A).

    WP-A owns that file's shape and it does not exist yet, so every level is
    probed rather than assumed: a dict of estimators, a list of rows, or a bare
    dict of folds all flatten to `{estimator, fold, ...}`, and anything else
    returns None so the comparison prints "not available" instead of crashing.
    """
    if not isinstance(pv, (dict, list)):
        return None
    rows = []

    def fields(v):
        return {k: v.get(k) for k in PV_FIELDS if isinstance(v, dict) and k in v}

    def folds_of(v):
        for name in ("by_fold", "folds", "per_fold", "by_event", "per_event"):
            if isinstance(v, dict) and isinstance(v.get(name), dict):
                return v[name]
        return {}

    src = None
    if isinstance(pv, dict):
        for name in ("by_estimator", "estimators", "per_estimator", "estimator"):
            if name in pv:
                src = pv[name]
                break
        if src is None:
            src = {k: v for k, v in pv.items() if isinstance(v, dict) and not k.startswith("_")}
    else:
        src = pv
    if isinstance(src, dict):
        for est, v in src.items():
            if not isinstance(v, dict):
                continue
            for fold, f in folds_of(v).items():
                if isinstance(f, dict):
                    rows.append({"estimator": est, "fold": fold, **fields(f)})
            f = fields(v)
            if f:
                rows.append({"estimator": est, "fold": "pooled", **f})
    elif isinstance(src, list):
        for v in src:
            if isinstance(v, dict):
                rows.append({"estimator": v.get("estimator"), "fold": v.get("fold") or v.get("event") or "pooled",
                             **fields(v)})
    return rows or None


def experiment_rows(ex) -> list | None:
    """The E0-E8 rows from `bench/out/experiments.json` (integration writes it).

    Read defensively for the same reason: only the scalar fields of each row are
    carried, in the writer's own order, so a row that gains or loses a metric
    still prints.
    """
    if not isinstance(ex, (dict, list)):
        return None
    items = (list(ex.items()) if isinstance(ex, dict)
             else [(str((r or {}).get("exp") or i), r) for i, r in enumerate(ex)])
    rows = []
    for name, v in items:
        if not isinstance(v, dict) or str(name).startswith("_"):
            continue
        row = {"exp": str(v.get("exp") or name)}
        row.update({k: x for k, x in _scalars(v).items() if k != "exp"})
        if len(row) > 1:
            rows.append(row)
    return rows or None


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------


def _f(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return "–" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def md_table(header: list, rows: list) -> None:
    print("| " + " | ".join(str(h) for h in header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(_f(x) for x in r) + " |")


def print_rows_table(rows: list, title: str, max_cols: int = 12) -> None:
    if not rows:
        print(f"\n{title}: not available")
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    cols = cols[:max_cols]
    print(f"\n{title}:")
    md_table(cols, [[r.get(c) for c in cols] for r in rows])


def main() -> None:
    s3, s1, s4 = _read(V3_OUT / "strategy.json"), _read(T1_OUT / "strategy.json"), _read(OUT / "strategy.json")
    l3, l1 = _read(V3_OUT / "live.json"), _read(T1_OUT / "live.json")
    l4, l4off = _read(OUT / "live.json"), _read(OUT / "live_no_race_state.json")
    a4 = _read(OUT / "ablation.json") or {}
    a1 = _read(T1_OUT / "ablation.json") or {}
    a3 = _read(V3_OUT / "ablation.json") or {}
    pv = _read(OUT / "place_value.json")
    ex = _read(OUT / "experiments.json")
    if not (s3 and s4):
        raise SystemExit("need bench/v3/out/strategy.json and bench/out/strategy.json")
    out = {"builds": {"v3": str(V3_OUT.relative_to(ROOT)), "task1": str(T1_OUT.relative_to(ROOT)),
                      "v4": str(OUT.relative_to(ROOT))},
           "strategy": {"v3": strategy_metrics(s3),
                        "task1": (strategy_metrics(s1) if s1 else None),
                        "v4": strategy_metrics(s4)},
           "live": {"v3": live_metrics(l3), "task1": live_metrics(l1), "v4": live_metrics(l4),
                    "v4_no_race_state": live_metrics(l4off)},
           "ablation_pooled": {"v4": a4.get("_pooled"), "task1": a1.get("_pooled"), "v3": a3.get("_pooled")},
           "first_stop": first_stop_table(s3, s1, s4),
           "place_value": place_value_rows(pv),
           "place_value_source": ("bench/out/place_value.json" if pv else "not written yet (WP-A)"),
           "experiments": experiment_rows(ex),
           "experiments_source": ("bench/out/experiments.json" if ex else "not written yet (WP-F/integration)")}
    pt3, pt1, pt4 = (_read(V3_OUT / "runtime.json") or {}, _read(T1_OUT / "runtime.json") or {},
                     _read(OUT / "runtime.json") or {})
    out["runtime"] = {"v3": pt3, "task1": pt1, "v4": pt4}
    dump("v4_compare.json", out)

    S3 = out["strategy"]["v3"]
    S1 = out["strategy"]["task1"] or {}
    S4 = out["strategy"]["v4"]
    P3, P1, P4 = S3["position_aware"], (S1.get("position_aware") or {}), S4["position_aware"]
    H3, H1, H4 = S3["haas"], (S1.get("haas") or {}), S4["haas"]

    def row(name, a, b, c, nd=3):
        print(f"| {name} | {_f(a, nd)} | {_f(b, nd)} | {_f(c, nd)} |")

    print("| Metric | V3 | Task 1 | V4 |\n|---|---|---|---|")
    for name, key in (("Mean |first stop - field green median| (laps, non-SC)", "first_stop_error_laps"),
                      ("Signed first-stop error (laps, non-SC)", "first_stop_signed_laps"),
                      ("Share of field first stops inside the window", "window_share"),
                      ("Oracle regret of the tool's plan (s)", "oracle_regret_s"),
                      ("Sequence run by anyone (of 7)", "seq_run_by_anyone"),
                      ("Start compound = majority (of 7)", "start_matches_majority"),
                      ("Stop count = mode (of 7)", "stops_match_mode"),
                      ("Mean field share on the recommended sequence", "mean_seq_share"),
                      ("Per-car plans sharing the field plan's shape", "per_car_same_shape")):
        row(name, S3.get(key), S1.get(key), S4.get(key))
    # the position-aware metric (new in V4; "-" on a build that has no block)
    for name, key in (("R_pos of the tool's plan (s)", "rpos_tool_s"),
                      ("R_pos of the tyre-optimal plan (s)", "rpos_tyre_optimal_s"),
                      ("R_pos of the field's modal plan (s)", "rpos_field_modal_s"),
                      ("L(tool): places lost in the first cycle", "L_tool"),
                      ("L(field modal)", "L_field_modal"),
                      ("P_retain(tool)", "P_retain_tool"),
                      ("Place value V used (s)", "place_value_s")):
        row(name, P3.get(key), P1.get(key), P4.get(key))
    for name, key in (("Haas: mean |per-car first stop - own green stop| (laps)", "first_stop_error_laps"),
                      ("Haas: signed first-stop error (laps)", "first_stop_signed_laps"),
                      ("Haas: share of per-car plans on the field plan's shape", "share_same_shape_as_field"),
                      ("Haas: mean L of the per-car plan", "L_mean")):
        row(name, H3.get(key), H1.get(key), H4.get(key))

    print("\n| Live metric | V3 | Task 1 | V4 | V4 no race state |\n|---|---|---|---|---|")
    LV = (out["live"]["v3"], out["live"]["task1"], out["live"]["v4"], out["live"]["v4_no_race_state"])
    for name, get in (("Real stops inside the window 3 laps earlier", lambda v: v.get("share_in_window")),
                      ("Stops within 3 laps of the recommendation", lambda v: v.get("share_within_3")),
                      ("Median |recommended - actual| (laps)", lambda v: v.get("median_abs_err_laps")),
                      ("Box-now cost the lap before the real stop (s)", lambda v: v.get("box_now_1_before_s")),
                      ("Window signal precision", lambda v: v.get("window_precision")),
                      ("Window signal recall", lambda v: v.get("window_recall")),
                      ("Box-now signal precision", lambda v: v.get("box_now_precision")),
                      ("Box-now signal recall", lambda v: v.get("box_now_recall")),
                      ("Tick mean (ms)", lambda v: v.get("tick_mean_ms")),
                      ("Tick p95 (ms)", lambda v: v.get("tick_p95_ms")),
                      ("Haas stops: n", lambda v: (v.get("stops_haas") or {}).get("n")),
                      ("Haas stops within 3 laps", lambda v: (v.get("stops_haas") or {}).get("share_within_3")),
                      ("Haas median |error| (laps)", lambda v: (v.get("stops_haas") or {}).get("median_abs_err_laps")),
                      ("Haas box-now cost the lap before (s)", lambda v: (v.get("stops_haas") or {}).get("box_now_1_before_s")),
                      ("Decision changes scored (car-lap pairs)", lambda v: (v.get("decision_stability") or {}).get("n_pairs")),
                      ("Action changed from the previous lap", lambda v: (v.get("decision_stability") or {}).get("share_changed")),
                      ("Action changed with nothing material changed", lambda v: (v.get("decision_stability") or {}).get("share_unexplained"))):
        print(f"| {name} | " + " | ".join(_f(get(v) if v else None) for v in LV) + " |")

    print("\nfirst stops (V3 / Task 1 / V4, and the position-aware regret):")
    md_table(["event", "SC", "field", "V3", "T1", "V4", "V3 err", "T1 err", "V4 err",
              "V4 window share", "R_pos tool", "L tool", "best candidate"],
             [[r["event"], r["sc_set"], r["field_median_green"], r["v3_first"], r["task1_first"], r["v4_first"],
               r["v3_err"], r["task1_err"], r["v4_err"], r["v4_window_share"], r["rpos_tool"], r["L_tool"],
               r["pos_best_candidate"]] for r in out["first_stop"]])

    print("\nHaas, per weekend (per-car plan vs the driver's own green first stop):")
    hb = H4.get("by_event") or {}
    md_table(["event", "driver", "plan", "first", "actual", "SC", "err", "same shape", "L"],
             [[k, drv, (d or {}).get("plan"), (d or {}).get("first_stop"), (d or {}).get("actual_first_stop"),
               (d or {}).get("actual_first_sc"), (d or {}).get("first_minus_actual_green"),
               (d or {}).get("same_shape_as_field"), (d or {}).get("L")]
              for k, v in hb.items() for drv, d in v.items()])

    print_rows_table(out["place_value"], "place-value estimators (bench/out/place_value.json)")
    print_rows_table(out["experiments"], "experiments E0-E8 (bench/out/experiments.json)")


if __name__ == "__main__":
    main()
