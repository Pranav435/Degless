"""V1 -> V2 -> V3: every headline metric of the three builds, with a verdict.

Reads the current outputs under bench/out/, the frozen V2 run under bench/v2/
and the frozen V1 run under bench/baseline/ (each one's bench/out plus its
meta/posterior files), and writes

    bench/out/compare.json           every number the report quotes, with a
                                     `summary` block a report writer can use
                                     directly: per metric {v1, v2, v3,
                                     delta_v3_v2, verdict}
    bench/out/compare.md             the comparison tables, as Markdown
    bench/out/fig/fig1_accuracy.png  stint-rate MAE per weekend V1/V2/V3 + width
    bench/out/fig/fig2_firststop.png first stop - field: tyre-optimal, V2,
                                     V3 without the prior, V3
    bench/out/fig/fig3_strategy.png  sequence/start/stop-count match counts
    bench/out/fig/fig4_life.png      predicted life / longest stint
    bench/out/fig/fig5_live.png      live tick latency and stop calls
    bench/out/fig/fig6_calibration.png the leave-one-out sweeps, kappa included

**The verdict rule**, stated in the JSON as `summary_rule` and applied by
`verdict()`: a metric that moved by less than the smaller of 5 % of its V2
value and one weekend's worth is "not meaningfully changed"; an exactly equal
pair is "unchanged"; otherwise "improved" or "regressed" by direction — lower is
better for errors, biases, widths, regrets and latencies, higher for match
counts and field shares, and for a coverage metric "better" means closer to its
target band.  One weekend's worth is 1 for a count out of the weekends
benchmarked, |V2| / n for a mean over n weekends, and undefined (so the 5 % test
alone decides) for a single figure like peak memory.

Charts follow the dataviz method: one axis, thin marks, a legend for two or
more series, direct labels only where they carry the story, text in ink
tokens, a validated categorical palette (slot 1 blue = V3, slot 2 orange = V2,
slot 3 aqua = V1, slot 4 violet = the V3 variant a figure contrasts), hairline
gridlines.
"""

from __future__ import annotations

import json
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (BASELINE, EVENTS, NON_SC_EVENTS, OUT, SHORT, V2, arg_events, baseline_meta,  # noqa: E402
                    baseline_out, dump, meta, v2_meta, v2_out)
from src.config import DATA_PROCESSED  # noqa: E402

FIG = OUT / "fig"
FIG.mkdir(parents=True, exist_ok=True)

# V2 numbered its figures differently (fig3_life, fig4_live, fig5_calibration and
# an earlier fig2_windows).  Leaving those in place next to V3's six would let the
# report pick up a figure from the previous build by its old name, so they are
# cleared here; the originals stay in bench/v2/out/fig/ and bench/baseline/out/fig/.
STALE_FIGS = ("fig2_windows.png", "fig3_life.png", "fig3_live.png", "fig4_live.png",
              "fig5_calibration.png")

# the validated reference palette (dataviz skill): slot 1 blue, 2 orange, 3 aqua, 4 violet
V3C, V2C, V1C, FOURTH = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
NEW, OLD, THIRD = V3C, V2C, V1C          # the names the V2 figures used
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
COMPOUND = {"SOFT": "#c2334a", "MEDIUM": "#eda100", "HARD": "#2a78d6"}

COV90_BAND = (0.88, 0.93)
COV95_BAND = (0.93, 0.97)

SUMMARY_RULE = (
    "verdict per metric: 'unchanged' if V3 equals V2 exactly; 'not meaningfully changed' if "
    "|V3 - V2| < min(0.05 * |V2|, one weekend's worth), where one weekend's worth is 1 for a "
    "count over the weekends benchmarked, |V2| / n for a mean over n weekends, and undefined "
    "(5% alone) for a single figure; otherwise 'improved' or 'regressed' by direction. Lower is "
    "better for MAE, absolute bias, interval width, first-stop error, oracle regret, stop-call "
    "error, tick latency, runtime, memory and failing gates; higher is better for match counts, "
    "field shares, coverage-in-band, precision and recall. A coverage metric is judged by its "
    "distance to its target band (90%: 0.88-0.93, 95%: 0.93-0.97), not by being larger. "
    "'unavailable' means one of the three builds does not report the metric."
)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False, "legend.fontsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def _num(x):
    """A float, or None for anything that is not a finite number."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _mean(xs):
    v = [_num(x) for x in xs]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def _median(xs):
    v = [_num(x) for x in xs]
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def _count(xs):
    return int(sum(bool(x) for x in xs))


def _dist_to_band(x, band):
    lo, hi = band
    if lo <= x <= hi:
        return 0.0
    return float(min(abs(x - lo), abs(x - hi)))


def verdict(v2, v3, *, lower_better=True, kind="mean", n=7, band=None, one_weekend=None) -> str:
    """The V2 -> V3 verdict, by the rule in `SUMMARY_RULE`."""
    a, b = _num(v2), _num(v3)
    if a is None or b is None:
        return "unavailable"
    d = b - a
    if d == 0:
        return "unchanged"
    ow = one_weekend
    if ow is None:
        if kind == "count":
            ow = 1.0
        elif kind == "mean" and n:
            ow = abs(a) / float(n)
    thr = min(0.05 * abs(a), ow) if ow is not None else 0.05 * abs(a)
    if abs(d) < thr:
        return "not meaningfully changed"
    if band is not None:
        da, db = _dist_to_band(a, band), _dist_to_band(b, band)
        if db < da:
            return "improved"
        if db > da:
            return "regressed"
        return "not meaningfully changed"
    return "improved" if ((d < 0) == bool(lower_better)) else "regressed"


class Summary:
    """The `summary` block, built one metric at a time."""

    def __init__(self):
        self.rows: dict = {}

    def add(self, key, label, v1, v2, v3, *, lower_better=True, kind="mean", n=7, band=None,
            one_weekend=None, unit="", note="") -> dict:
        a, b = _num(v2), _num(v3)
        row = {"label": label, "v1": _num(v1), "v2": a, "v3": b,
               "delta_v3_v2": (b - a if (a is not None and b is not None) else None),
               "delta_v3_v1": (b - _num(v1) if (b is not None and _num(v1) is not None) else None),
               "unit": unit, "kind": kind, "n": n,
               "lower_is_better": (None if band else bool(lower_better)),
               "target_band": list(band) if band else None,
               "verdict": verdict(v2, v3, lower_better=lower_better, kind=kind, n=n, band=band,
                                  one_weekend=one_weekend)}
        if note:
            row["note"] = note
        self.rows[key] = row
        return row


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def _bars(ax, labels, series, *, ylabel, title, fmt="{:.3f}", ylim=None, ref=None, ref_label=None,
          label_bars=True, rotate=90, legend_loc="upper right", legend_kw=None):
    """Grouped bars, one group per label, `series` a list of (name, colour, values).

    `rotate=0` for short labels (counts, signed lap offsets), 90 for the
    three-decimal values that would otherwise collide across seven groups.
    """
    x = np.arange(len(labels))
    k = max(len(series), 1)
    w = 0.8 / k
    for i, (name, colour, vals) in enumerate(series):
        off = (i - (k - 1) / 2) * w
        v = [(_num(z) if _num(z) is not None else np.nan) for z in vals]
        ax.bar(x + off, v, w * 0.9, color=colour, label=name, zorder=3)
        if label_bars:
            for xi, z in enumerate(v):
                if np.isfinite(z):
                    ax.text(xi + off, z, (" " if rotate else "") + fmt.format(z), ha="center",
                            va="bottom" if z >= 0 else "top", fontsize=7, color=INK2, rotation=rotate)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(len(labels) - 0.5, ref, f" {ref_label}", va="bottom", ha="right", fontsize=8, color=MUTED)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        flat = [_num(z) for _, _, vs in series for z in vs]
        flat = [z for z in flat if z is not None] + ([ref] if ref is not None else [])
        if flat:
            top = max(flat)
            ax.set_ylim(min(0, min(flat) * 1.15), top * 1.30 if top > 0 else 1)
    ax.legend(loc=legend_loc, **(legend_kw or {}))


def fig1_accuracy(per_event: list) -> None:
    labels = [SHORT.get(r["event"], r["event"]) for r in per_event]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.7), gridspec_kw={"width_ratios": [3, 2]})
    _bars(axes[0], labels,
          [("V1", V1C, [r["v1_mae"] for r in per_event]),
           ("V2", V2C, [r["v2_mae"] for r in per_event]),
           ("V3", V3C, [r["v3_mae"] for r in per_event])],
          ylabel="stint-rate MAE (s/lap)", title="Sealed curves vs the race, per weekend",
          ref=0.15, ref_label="target 0.15")
    ax = axes[1]
    x = np.arange(len(labels))
    for name, colour, k in (("V1", V1C, "v1_width"), ("V2", V2C, "v2_width"), ("V3", V3C, "v3_width")):
        ax.plot(x, [_num(r[k]) for r in per_event], color=colour, lw=1.8, marker="o", ms=4.5,
                label=name, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("90% interval width on the stint rate (s/lap)")
    ax.set_title("Interval width (coverage in the table)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_accuracy.png", dpi=170)
    plt.close(fig)


def fig2_firststop(rows: list) -> None:
    labels = [SHORT.get(r["event"], r["event"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    _bars(ax, labels,
          [("tyre-optimal (no prior, no position)", V1C, [r["tyre"] for r in rows]),
           ("V2", V2C, [r["v2"] for r in rows]),
           ("V3 without the first-stop prior", FOURTH, [r["v3_no_prior"] for r in rows]),
           ("V3", V3C, [r["v3"] for r in rows])],
          ylabel="recommended first stop − field median (laps)",
          title="First stop against the field's green-flag median", fmt="{:+.0f}",
          label_bars=True, rotate=0, legend_loc="upper center",
          legend_kw={"ncol": 2, "bbox_to_anchor": (0.5, -0.16)})
    ax.axhline(0, color="#c3c2b7", lw=1, zorder=2)
    vals = [_num(v) for r in rows for v in (r["tyre"], r["v2"], r["v3_no_prior"], r["v3"])]
    vals = [v for v in vals if v is not None] or [1.0]
    top, bot = max(vals), min(vals)
    ax.set_ylim(min(-2.5, bot * 1.35), max(1.0, top * 1.35))
    for xi, r in enumerate(rows):
        if r.get("sc_set"):
            ax.text(xi, max(1.0, top * 1.32), "SC set\nthe stops", ha="center", va="top",
                    fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_firststop.png", dpi=170)
    plt.close(fig)


def fig3_strategy(counts: dict, n_events: int) -> None:
    names = [("sequence_run_by_anyone", "Sequence run\nby a finisher"),
             ("start_match", "Start compound\n= the majority's"),
             ("stop_count_match", "Stop count\n= the field's mode")]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    _bars(ax, [lab for _, lab in names],
          [("V1", V1C, [counts[k]["v1"] for k, _ in names]),
           ("V2", V2C, [counts[k]["v2"] for k, _ in names]),
           ("V3", V3C, [counts[k]["v3"] for k, _ in names])],
          ylabel=f"weekends out of {n_events}", title="Does the recommendation match the field?",
          fmt="{:.0f}", rotate=0, ylim=(0, n_events + 1.4))
    ax.set_yticks(range(0, n_events + 1))
    fig.tight_layout()
    fig.savefig(FIG / "fig3_strategy.png", dpi=170)
    plt.close(fig)


def fig4_life(L: pd.DataFrame) -> None:
    if L.empty:
        return
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    xs = np.arange(len(L))
    cap = 4.0
    for i, (name, colour, k) in enumerate((("V1", V1C, "v1"), ("V2", V2C, "v2"), ("V3", V3C, "v3"))):
        off = (i - 1) * 0.27
        v = pd.to_numeric(L[k], errors="coerce").clip(upper=cap)
        ax.bar(xs + off, v, 0.25, color=colour, label=name, zorder=3)
        for xi, raw in enumerate(pd.to_numeric(L[k], errors="coerce")):
            if np.isfinite(raw) and raw > cap:
                ax.text(xi + off, cap, f"{raw:.0f}×", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(len(L) - 0.5, 1.0, " longest stint run", va="bottom", ha="right", fontsize=8, color=MUTED)
    ax.set_xticks(xs, [f"{SHORT.get(e, e)}\n{c[0]}" for e, c in zip(L["event"], L["compound"])], fontsize=7.5)
    ax.set_ylabel("predicted life ÷ longest stint run")
    ax.set_title("Tyre life against the longest stint any finisher ran")
    ax.set_ylim(0, cap + 0.35)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_life.png", dpi=170)
    plt.close(fig)


def fig5_live(rows: list) -> None:
    if not rows:
        return
    labs = [SHORT.get(r["event"], r["event"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.3))
    _bars(axes[0], labs,
          [("V1", V1C, [r["tick_v1"] for r in rows]), ("V2", V2C, [r["tick_v2"] for r in rows]),
           ("V3", V3C, [r["tick_v3"] for r in rows])],
          ylabel="mean tick (ms, whole field)", title="Live engine latency per lap", fmt="{:.0f}")
    axes[0].set_yscale("log")
    _bars(axes[1], labs,
          [("V1", V1C, [r["inwin_v1"] for r in rows]), ("V2", V2C, [r["inwin_v2"] for r in rows]),
           ("V3", V3C, [r["inwin_v3"] for r in rows])],
          ylabel="share of real stops inside the window", title="Stop calls: in the window",
          fmt="{:.2f}", rotate=0, ylim=(0, 1.18))
    _bars(axes[2], labs,
          [("V1", V1C, [r["within3_v1"] for r in rows]), ("V2", V2C, [r["within3_v2"] for r in rows]),
           ("V3", V3C, [r["within3_v3"] for r in rows])],
          ylabel="share of real stops within 3 laps of the call", title="Stop calls: within 3 laps",
          fmt="{:.2f}", rotate=0, ylim=(0, 1.18))
    fig.tight_layout()
    fig.savefig(FIG / "fig5_live.png", dpi=170)
    plt.close(fig)


def fig6_calibration(cal: dict) -> None:
    g = cal.get("global") or {}
    sw = g.get("sweeps") or {}
    panels = [("lambda", "undercut-exposure weight λ", "mean |first stop − field| (laps)", "mean_abs_first_err",
               "λ against the field's first stops", g.get("undercut_lambda")),
              ("kappa", "first-stop prior weight κ (s per nat)", "mean |first stop − field| (laps)", "mean_abs_first_err",
               "κ against the field's first stops", g.get("first_stop_kappa_s")),
              ("tau", "plan-prior weight τ (s per nat)", "mean share of the field on the plan", "mean_seq_share",
               "τ against the field's plan shapes", g.get("plan_prior_tau_s")),
              ("grid", "grid-start penalty (s per step)", "mean share of the field on the plan", "mean_seq_share",
               "Grid penalty", g.get("grid_start_penalty_s"))]
    have = [p for p in panels if sw.get(p[0])]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(3.5 * len(have), 3.1), squeeze=False)
    for ax, (name, xlab, ylab, ykey, title, chosen) in zip(axes[0], have):
        rows = sw.get(name) or []
        xs = [_num(r.get(name)) for r in rows]
        ys = [_num(r.get(ykey)) for r in rows]
        ax.plot(xs, ys, color=V3C, lw=1.8, marker="o", ms=4.5, zorder=3)
        if _num(chosen) is not None:
            ax.axvline(float(chosen), color=MUTED, lw=1, ls=(0, (4, 3)))
            ax.text(float(chosen), ax.get_ylim()[0], f" chosen {float(chosen):g}", fontsize=8,
                    color=MUTED, va="bottom")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_calibration.png", dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
# gathering
# --------------------------------------------------------------------------


def pooled_of(acc: dict | None, variant: str) -> dict:
    for p in ((acc or {}).get("pooled") or []):
        if p.get("variant") == variant:
            return p
    return {}


def accuracy_block(acc, acc_v2, acc_v1, events: list) -> tuple:
    """Per-weekend and pooled stint-rate accuracy for the three builds.

    The V1 and V2 columns come from *this* run's re-scoring of their frozen
    posteriors wherever it exists (`accuracy.json`'s `frozen_per_event` and the
    `*_sealed[race_sigma]` pooled rows), so all three builds are measured by the
    same code and the same noise; each build's own reported number is carried
    alongside as `*_reported` for the reader who wants the published figure.
    """
    frozen = (acc or {}).get("frozen_per_event") or {}
    per = []
    for k in events:
        v3 = (((acc or {}).get("per_event") or {}).get(k) or {}).get("variants", {}).get("sealed", {})
        fz2 = (frozen.get("v2_sealed[race_sigma]") or {}).get(k) or {}
        fz1 = (frozen.get("baseline_sealed[race_sigma]") or {}).get(k) or {}
        own2 = (((acc_v2 or {}).get("per_event") or {}).get(k) or {}).get("variants", {}).get("sealed", {})
        own1 = (((acc_v1 or {}).get("per_event") or {}).get(k) or {}).get("variants", {}).get("sealed", {})
        oracle = (((acc or {}).get("per_event") or {}).get(k) or {}).get("variants", {}).get("oracle_race", {})
        reg = (((acc or {}).get("per_event") or {}).get(k) or {}).get("regime") or {}
        per.append({
            "event": k,
            "v1_mae": fz1.get("rate_mae", own1.get("rate_mae")), "v1_mae_reported": own1.get("rate_mae"),
            "v2_mae": fz2.get("rate_mae", own2.get("rate_mae")), "v2_mae_reported": own2.get("rate_mae"),
            "v3_mae": v3.get("rate_mae"),
            "v1_width": fz1.get("rate_width90", own1.get("rate_width90")),
            "v2_width": fz2.get("rate_width90", own2.get("rate_width90")),
            "v3_width": v3.get("rate_width90"),
            "v1_cov90": fz1.get("rate_cov90", own1.get("rate_cov90")),
            "v2_cov90": fz2.get("rate_cov90", own2.get("rate_cov90")),
            "v3_cov90": v3.get("rate_cov90"),
            "v1_cov95": fz1.get("rate_cov95"), "v2_cov95": fz2.get("rate_cov95"),
            "v3_cov95": v3.get("rate_cov95"),
            "v1_bias": fz1.get("rate_bias", own1.get("rate_bias")),
            "v2_bias": fz2.get("rate_bias", own2.get("rate_bias")),
            "v3_bias": v3.get("rate_bias"),
            "v3_spearman": v3.get("spearman"), "v2_spearman": own2.get("spearman"),
            "oracle_mae": oracle.get("rate_mae"),
            "v3_practice_mae": (((acc or {}).get("per_event") or {}).get(k) or {}).get("variants", {}).get("practice_only", {}).get("rate_mae"),
            "regime_v3": reg.get("ratio"), "regime_mode": reg.get("mode"),
            "regime_self": reg.get("self"),
            "regime_v2": (v2_meta(k) or {}).get("regime", {}).get("ratio"),
            "regime_v1": (baseline_meta(k) or {}).get("regime", {}).get("ratio"),
            "n_stints": v3.get("n_stints"),
        })
    pooled = {"v3": pooled_of(acc, "sealed"),
              "v2_rescored": pooled_of(acc, "v2_sealed[race_sigma]"),
              "v1_rescored": pooled_of(acc, "baseline_sealed[race_sigma]"),
              "v2_reported": pooled_of(acc_v2, "sealed"),
              "v1_reported": pooled_of(acc_v1, "sealed"),
              "all_variants_v3": {p["variant"]: p for p in ((acc or {}).get("pooled") or [])}}
    return per, pooled


def strategy_rows(stg, stg_v2, stg_v1, abl, events: list) -> list:
    """One row per weekend with the three builds' decision against the field.

    V1's and V2's plans are re-scored against the field *inside this run*
    (`strategy.json`'s `baseline` and `v2` sub-blocks), which is what makes the
    share and match columns comparable; their own frozen runs are the fallback
    when a sub-block is missing.
    """
    rows = []
    for k in events:
        s = (stg or {}).get(k) or {}
        s2, s1 = (stg_v2 or {}).get(k) or {}, (stg_v1 or {}).get(k) or {}
        fs = s.get("first_stop") or {}
        fs2 = s2.get("first_stop") or {}
        inrun2, inrun1 = s.get("v2") or {}, s.get("baseline") or {}
        a = (abl or {}).get(k) or {}
        rows.append({
            "event": k,
            "v1_plan": s1.get("recommended"), "v2_plan": s2.get("recommended"), "v3_plan": s.get("recommended"),
            "tyre_optimal": s.get("tyre_optimal"),
            "field_modal": s.get("field_modal_seq"), "winner": s.get("winner_seq"),
            "v1_share": inrun1.get("seq_share", s1.get("rec_seq_share")),
            "v2_share": inrun2.get("seq_share", s2.get("rec_seq_share")),
            "v3_share": s.get("rec_seq_share"),
            "v1_run": (inrun1.get("seq_share", s1.get("rec_seq_share")) or 0) > 0,
            "v2_run": inrun2.get("seq_run_by_anyone", (inrun2.get("seq_share") or 0) > 0),
            "v3_run": s.get("rec_seq_run_by_anyone"),
            "v1_start": inrun1.get("start_matches_majority", s1.get("start_matches_majority")),
            "v2_start": inrun2.get("start_matches_majority", s2.get("start_matches_majority")),
            "v3_start": s.get("start_matches_majority"),
            "v1_stops": inrun1.get("stops_match_mode", s1.get("stops_match_mode")),
            "v2_stops": inrun2.get("stops_match_mode", s2.get("stops_match_mode")),
            "v3_stops": s.get("stops_match_mode"),
            "sc_set": fs.get("sc_set"), "field_first": fs.get("field_median_green"),
            "v1_first": fs.get("baseline_minus_field"), "v2_first": fs.get("v2_minus_field"),
            "v3_first": fs.get("rec_minus_field"), "tyre_first": fs.get("tyre_minus_field"),
            "v2_tyre_first": fs2.get("tyre_minus_field"),
            "v3_no_prior_first": ((a.get("no_first_stop_prior") or {}).get("first_minus_field")),
            "first_stop_s": fs.get("first_stop_s"), "kappa": fs.get("first_stop_kappa_s"),
            "v1_win1": ((s1.get("pit_windows") or [{}])[0] or {}).get("share_inside"),
            "v2_win1": ((s2.get("pit_windows") or [{}])[0] or {}).get("share_inside"),
            "v3_win1": ((s.get("pit_windows") or [{}])[0] or {}).get("share_inside"),
            "regret_v3": (s.get("oracle") or {}).get("regret_s", {}).get("tool"),
            "regret_v2": (s.get("oracle") or {}).get("regret_s", {}).get("v2_tool"),
            "regret_v1": (s.get("oracle") or {}).get("regret_s", {}).get("baseline_tool"),
            "regret_field": (s.get("oracle") or {}).get("regret_s", {}).get("field_modal"),
            "regret_winner": (s.get("oracle") or {}).get("regret_s", {}).get("winner"),
            "regret_tyre": (s.get("oracle") or {}).get("regret_s", {}).get("tyre_optimal"),
            "regret_v2_own": (s2.get("oracle") or {}).get("regret_s", {}).get("tool"),
            "regret_v1_own": (s1.get("oracle") or {}).get("regret_s", {}).get("tool"),
            "life": s.get("life") or [],
            "per_driver": s.get("per_driver") or {}, "per_driver_v2": s2.get("per_driver") or {},
            "per_driver_v1": s1.get("per_driver") or {},
            "per_driver_variants": s.get("per_driver_variants") or {},
            "cliff_detector": s.get("cliff_detector") or {},
            "stint_fe_baseline": s.get("stint_fe_baseline") or {},
            "bayes_vs_stint_fe_pooled": s.get("bayes_vs_stint_fe_pooled") or {},
            "calibration_used": s.get("calibration") or {},
            "cf": s.get("counterfactual") or {},
            "gates_failed": s.get("gates_failed") or [],
            "gates_failed_v2": s2.get("gates_failed") or [], "gates_failed_v1": s1.get("gates_failed") or [],
        })
    return rows


def live_rows(live, live_v2, live_v1) -> list:
    keys = [k for k in ("hungary-2026", "barcelona-2026") if (live or {}).get(k)]
    rows = []
    for k in keys:
        n, o, p = (live or {}).get(k) or {}, (live_v2 or {}).get(k) or {}, (live_v1 or {}).get(k) or {}
        sg = n.get("signals") or {}
        rows.append({
            "event": k,
            "tick_v3": (n.get("tick_ms") or {}).get("mean"), "tick_v2": (o.get("tick_ms") or {}).get("mean"),
            "tick_v1": (p.get("tick_ms") or {}).get("mean"),
            "p95_v3": (n.get("tick_ms") or {}).get("p95"), "p95_v2": (o.get("tick_ms") or {}).get("p95"),
            "p95_v1": (p.get("tick_ms") or {}).get("p95"),
            "max_v3": (n.get("tick_ms") or {}).get("max"), "max_v2": (o.get("tick_ms") or {}).get("max"),
            "max_v1": (p.get("tick_ms") or {}).get("max"),
            "inwin_v3": (n.get("stops") or {}).get("share_in_window"),
            "inwin_v2": (o.get("stops") or {}).get("share_in_window"),
            "inwin_v1": (p.get("stops") or {}).get("share_in_window"),
            "within3_v3": (n.get("stops") or {}).get("share_err_within_3"),
            "within3_v2": (o.get("stops") or {}).get("share_err_within_3"),
            "within3_v1": (p.get("stops") or {}).get("share_err_within_3"),
            "err_v3": (n.get("stops") or {}).get("median_abs_err_laps"),
            "err_v2": (o.get("stops") or {}).get("median_abs_err_laps"),
            "err_v1": (p.get("stops") or {}).get("median_abs_err_laps"),
            "box_v3": (n.get("stops") or {}).get("median_box_now_delta_1_before"),
            "box_v2": (o.get("stops") or {}).get("median_box_now_delta_1_before"),
            "box_v1": (p.get("stops") or {}).get("median_box_now_delta_1_before"),
            "n_stops": (n.get("stops") or {}).get("n"),
            "alarms_v3": (n.get("stops") or {}).get("alarms_before_stop"),
            "alarms_v2": (o.get("stops") or {}).get("alarms_before_stop"),
            "signals": sg,
            # V2 reported only "an alarm preceded the stop", i.e. a recall with no precision
            "collapse_recall_v2": (((o.get("stops") or {}).get("alarms_before_stop") or 0)
                                   / ((o.get("stops") or {}).get("n") or np.nan)
                                   if (o.get("stops") or {}).get("n") else None),
            "m_final_v3": ((n.get("regime") or {}).get("final") or [None, None])[1],
            "m_final_v2": ((o.get("regime") or {}).get("final") or [None, None])[1],
            "m_self": (n.get("regime") or {}).get("self_measured_offline"),
            "n_options_v3": n.get("n_options_leader_median"), "n_options_v2": o.get("n_options_leader_median"),
        })
    return rows


def _stage(speed: dict | None, *needles) -> float | None:
    """The first stage whose name contains every needle of one needle group."""
    for r in ((speed or {}).get("stages") or []):
        name = str(r.get("stage", "")).lower()
        for group in needles:
            if all(t.lower() in name for t in ([group] if isinstance(group, str) else group)):
                return _num(r.get("seconds"))
    return None


def _pytest_counts(path) -> dict:
    """`{passed, failed, skipped}` from pytest's own summary line.

    The last `N passed` line is the summary; a run with failures reads
    "1 failed, 37 passed, ...", so the pass count alone would hide the failure —
    which is the whole reason the failed and skipped counts are reported beside
    it and the suite's exit code beside them.
    """
    out = {"passed": None, "failed": None, "skipped": None}
    if not path.exists():
        return out
    txt = path.read_text()
    lines = [ln for ln in txt.splitlines() if re.search(r"\d+ (passed|failed|error)", ln)]
    if not lines:
        return out
    last = lines[-1]
    for key in ("passed", "failed", "skipped"):
        m = re.search(rf"(\d+) {key}", last)
        if m:
            out[key] = int(m.group(1))
    if out["passed"] is not None:
        out["failed"] = out["failed"] or 0
        out["skipped"] = out["skipped"] or 0
    return out


def _runtime(root) -> dict:
    """A suite run's `runtime.json` (per-stage seconds, exit codes, total)."""
    p = (root / "out" / "runtime.json") if root is not None else (OUT / "runtime.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# V1 and V2 predate `runtime.json`; their chain logs are unreliable (the frozen
# copies carry each other's timestamps), so the suite runtime for those builds is
# quoted from the report that published it rather than re-derived from a log.
V2_SUITE_SECONDS = 1500.0        # results_updated.md §6: "the whole benchmark suite ... runs in 25 minutes"
SUITE_NOTE_V2 = "V2's figure is from the V2 report (results_updated.md: 25 minutes); it predates runtime.json"


# --------------------------------------------------------------------------


def main() -> None:
    args = arg_events(__doc__)
    for name in STALE_FIGS:
        p = FIG / name
        if p.exists():
            p.unlink()
            print(f"  removed the previous build's {name} (kept in bench/v2/out/fig/)")
    acc, acc_v2, acc_v1 = _load("accuracy.json"), v2_out("accuracy.json"), baseline_out("accuracy.json")
    stg, stg_v2, stg_v1 = _load("strategy.json"), v2_out("strategy.json"), baseline_out("strategy.json")
    # Compare only the weekends this run actually produced.  A three-way count
    # of "6/7" against a V3 run that benchmarked one weekend is not a
    # comparison, it is an artefact of the subset, so the subset is the
    # population for every aggregate here and the JSON says so.
    present = set(((acc or {}).get("per_event") or {})) | set(stg or {})
    events = [k for k in args.events if k in present] or list(args.events)
    n_ev = len(events)
    partial = len(events) < len(EVENTS)
    if partial:
        print(f"  partial run: aggregates cover {events} only; the frozen builds' own pooled rows "
              f"(per-car Spearman, prior-only outlook) still cover all seven weekends")
    live, live_v2, live_v1 = _load("live.json"), v2_out("live.json"), baseline_out("live.json")
    outl, outl_v2, outl_v1 = _load("outlook.json"), v2_out("outlook.json"), baseline_out("outlook.json")
    abl, abl_v2 = _load("ablation.json"), v2_out("ablation.json")
    speed, speed_v2, speed_v1 = _load("speed.json"), v2_out("speed.json"), baseline_out("speed.json")
    stab, stab_v2, stab_v1 = _load("stability.json"), v2_out("stability.json"), baseline_out("stability.json")
    apex, apex_v2, apex_v1 = _load("apex.json"), v2_out("apex.json"), baseline_out("apex.json")
    calp = DATA_PROCESSED / "calibration.json"
    cal = json.loads(calp.read_text()) if calp.exists() else {}
    cal_v2 = json.loads((V2 / "processed" / "calibration.json").read_text()) \
        if (V2 / "processed" / "calibration.json").exists() else {}

    S = Summary()
    out: dict = {"events": events, "n_events": n_ev, "partial_run": partial,
                 "all_scored_weekends": list(EVENTS), "summary_rule": SUMMARY_RULE}
    md: list = []

    # ------------------------------------------------------------- accuracy
    per, pooled = accuracy_block(acc, acc_v2, acc_v1, events)
    out["accuracy"] = {"per_event": per, "pooled": pooled,
                       "population": (acc or {}).get("population"),
                       "regime_v3_circuit_check": (acc or {}).get("regime_v3_circuit_check")}
    p3, p2, p1 = pooled["v3"], pooled["v2_rescored"], pooled["v1_rescored"]
    S.add("deg_mae_mean", "Degradation-rate error vs the race (stint-rate MAE, s/lap, mean over weekends)",
          p1.get("rate_mae_mean"), p2.get("rate_mae_mean"), p3.get("rate_mae_mean"), n=n_ev, unit="s/lap")
    S.add("deg_mae_max", "Worst weekend's stint-rate MAE (s/lap)",
          p1.get("rate_mae_max"), p2.get("rate_mae_max"), p3.get("rate_mae_max"), n=n_ev, unit="s/lap")
    S.add("deg_bias_mean", "Stint-rate bias (s/lap, mean over weekends; 0 is best)",
          p1.get("bias_mean"), p2.get("bias_mean"), p3.get("bias_mean"), band=(0.0, 0.0), n=n_ev, unit="s/lap")
    S.add("cov90", "90% interval coverage on the stint rate",
          p1.get("cov90_mean"), p2.get("cov90_mean"), p3.get("cov90_mean"), band=COV90_BAND, n=n_ev)
    S.add("cov95", "95% interval coverage on the stint rate",
          p1.get("cov95_mean"), p2.get("cov95_mean"), p3.get("cov95_mean"), band=COV95_BAND, n=n_ev,
          note=("computed here from the per-stint tables of all three builds' posteriors; the frozen "
                "V1/V2 runs did not report it themselves"))
    S.add("width", "90% interval width on the stint rate (s/lap)",
          p1.get("width_mean"), p2.get("width_mean"), p3.get("width_mean"), n=n_ev, unit="s/lap")
    S.add("practice_only_mae", "Practice posterior x regime, no circuit history (stint-rate MAE)",
          pooled_of(acc, "baseline_practice_only[race_sigma]").get("rate_mae_mean"),
          pooled_of(acc, "v2_practice_only[race_sigma]").get("rate_mae_mean"),
          pooled["all_variants_v3"].get("practice_only", {}).get("rate_mae_mean"), n=n_ev, unit="s/lap")
    S.add("mixedlm_mae", "MixedLM slope x regime (the frequentist baseline)",
          None, None, pooled["all_variants_v3"].get("mixedlm_x_regime", {}).get("rate_mae_mean"),
          n=n_ev, unit="s/lap")
    per_weekend = {}
    for r in per:
        per_weekend[r["event"]] = {
            "v1": _num(r["v1_mae"]), "v2": _num(r["v2_mae"]), "v3": _num(r["v3_mae"]),
            "delta_v3_v2": ((_num(r["v3_mae"]) - _num(r["v2_mae"]))
                            if (_num(r["v3_mae"]) is not None and _num(r["v2_mae"]) is not None) else None),
            "verdict": verdict(r["v2_mae"], r["v3_mae"], kind="value")}
    S.rows["per_weekend_mae"] = {"label": "Stint-rate MAE per weekend (s/lap)", "unit": "s/lap",
                                 "lower_is_better": True, "per_event": per_weekend,
                                 "verdict": ("improved" if _count(v["verdict"] == "improved" for v in per_weekend.values())
                                             > _count(v["verdict"] == "regressed" for v in per_weekend.values())
                                             else ("regressed" if _count(v["verdict"] == "regressed" for v in per_weekend.values())
                                                   > _count(v["verdict"] == "improved" for v in per_weekend.values())
                                                   else "not meaningfully changed")),
                                 "n_improved": _count(v["verdict"] == "improved" for v in per_weekend.values()),
                                 "n_regressed": _count(v["verdict"] == "regressed" for v in per_weekend.values())}
    fig1_accuracy(per)

    md.append("### Accuracy per weekend (stint-rate MAE, s/lap; every build re-scored by this run's code)\n")
    md.append("| Weekend | V1 | V2 | V3 | V3 practice-only | Oracle | V3 bias | 90% coverage V1→V2→V3 | Width V1→V2→V3 | Regime V2→V3 (self) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")

    def f(v, d=3, plus=False):
        x = _num(v)
        if x is None:
            return "–"
        return f"{x:+.{d}f}" if plus else f"{x:.{d}f}"

    for r in per:
        md.append(f"| {r['event'].split('-')[0].title()} | {f(r['v1_mae'])} | {f(r['v2_mae'])} | **{f(r['v3_mae'])}** | "
                  f"{f(r['v3_practice_mae'])} | {f(r['oracle_mae'])} | {f(r['v3_bias'], 3, True)} | "
                  f"{f(r['v1_cov90'], 2)} → {f(r['v2_cov90'], 2)} → {f(r['v3_cov90'], 2)} | "
                  f"{f(r['v1_width'])} → {f(r['v2_width'])} → {f(r['v3_width'])} | "
                  f"{f(r['regime_v2'], 2)} → {f(r['regime_v3'], 2)} ({f(r['regime_self'], 2)}) |")
    md.append(f"| **Mean** | {f(_mean([r['v1_mae'] for r in per]))} | {f(_mean([r['v2_mae'] for r in per]))} | "
              f"**{f(_mean([r['v3_mae'] for r in per]))}** | {f(_mean([r['v3_practice_mae'] for r in per]))} | "
              f"{f(_mean([r['oracle_mae'] for r in per]))} | {f(_mean([r['v3_bias'] for r in per]), 3, True)} | "
              f"{f(_mean([r['v1_cov90'] for r in per]), 2)} → {f(_mean([r['v2_cov90'] for r in per]), 2)} → "
              f"{f(_mean([r['v3_cov90'] for r in per]), 2)} | {f(_mean([r['v1_width'] for r in per]))} → "
              f"{f(_mean([r['v2_width'] for r in per]))} → {f(_mean([r['v3_width'] for r in per]))} | |\n")

    md.append("### Pooled over the benchmarked weekends, every curve variant (V3 run)\n")
    md.append("| Curve source | Rate MAE mean | max | Bias | 90% cov | 95% cov | Width | Spearman |")
    md.append("|---|---|---|---|---|---|---|---|")
    order = ["oracle_race", "sealed", "regime_v3_circuit", "regime_v3_pooled", "regime_v2_temperature",
             "regime_oracle_temperature", "sealed_geomean", "sealed_driver_team_pooled",
             "sealed_driver_practice_dev", "sealed_driver_hist", "practice_only", "practice_no_regime",
             "mixedlm_x_regime", "history_only", "season_loo", "zero",
             "v2_sealed[race_sigma]", "v2_practice_only[race_sigma]", "baseline_sealed[race_sigma]",
             "baseline_practice_only[race_sigma]"]
    names = {"oracle_race": "Oracle (this race's own rates, in-sample)",
             "sealed": "**V3 sealed** (shipped)",
             "regime_v3_circuit": "V3 regime: donor median + this circuit's history (re-folded)",
             "regime_v3_pooled": "V3 regime: donor median only (no circuit history)",
             "regime_v2_temperature": "V2 regime: archive race-day temperature as the forecast",
             "regime_oracle_temperature": "Regime with the actual race temperature (a perfect forecast)",
             "sealed_geomean": "V3 sealed with V2's pooled geometric-mean regime",
             "sealed_driver_team_pooled": "V3 sealed, per-car scale team-pooled (shipped per-car mode)",
             "sealed_driver_practice_dev": "V3 sealed, per-car scale from this weekend's own dev",
             "sealed_driver_hist": "V3 sealed, per-car scale from the LOO race factors (V2's)",
             "practice_only": "Practice posterior x regime (no history)",
             "practice_no_regime": "Practice posterior, no regime transfer",
             "mixedlm_x_regime": "MixedLM slope x regime", "history_only": "Circuit history 2023-25 only",
             "season_loo": "Other 2026 races' mean rate", "zero": "Zero degradation",
             "v2_sealed[race_sigma]": "V2 sealed, re-scored here", "v2_practice_only[race_sigma]": "V2 practice-only, re-scored here",
             "baseline_sealed[race_sigma]": "V1 sealed, re-scored here",
             "baseline_practice_only[race_sigma]": "V1 practice-only, re-scored here"}
    for v in order:
        p = pooled["all_variants_v3"].get(v)
        if not p:
            continue
        md.append(f"| {names.get(v, v)} | {f(p.get('rate_mae_mean'))} | {f(p.get('rate_mae_max'))} | "
                  f"{f(p.get('bias_mean'), 3, True)} | {f(p.get('cov90_mean'), 2)} | {f(p.get('cov95_mean'), 2)} | "
                  f"{f(p.get('width_mean'))} | {f(p.get('spearman_mean'), 2, True)} |")
    md.append("")

    # ------------------------------------------------------------ decisions
    D = strategy_rows(stg, stg_v2, stg_v1, abl, events)
    out["decisions"] = D
    counts = {
        "sequence_run_by_anyone": S.add("sequence_run_by_anyone", "Compound sequence run by at least one finisher",
                                        _count(r["v1_run"] for r in D), _count(r["v2_run"] for r in D),
                                        _count(r["v3_run"] for r in D), lower_better=False, kind="count", n=n_ev),
        "start_match": S.add("start_match", "Start compound = the majority's",
                             _count(r["v1_start"] for r in D), _count(r["v2_start"] for r in D),
                             _count(r["v3_start"] for r in D), lower_better=False, kind="count", n=n_ev),
        "stop_count_match": S.add("stop_count_match", "Stop count = the field's mode",
                                  _count(r["v1_stops"] for r in D), _count(r["v2_stops"] for r in D),
                                  _count(r["v3_stops"] for r in D), lower_better=False, kind="count", n=n_ev),
    }
    S.add("mean_field_share", "Mean share of the field that ran the recommended sequence",
          _mean([r["v1_share"] for r in D]), _mean([r["v2_share"] for r in D]),
          _mean([r["v3_share"] for r in D]), lower_better=False, n=n_ev)
    fig3_strategy(counts, n_ev)

    non_sc = [r for r in D if r["event"] in NON_SC_EVENTS and not r["sc_set"]]
    n_fs = len(non_sc) or 1

    def absmean(k, rows=None):
        return _mean([abs(x) for x in (_num(r[k]) for r in (rows if rows is not None else non_sc)) if x is not None])

    S.add("first_stop_abs_err", "Mean |first stop − field green-flag median| (laps, non-SC weekends)",
          absmean("v1_first"), absmean("v2_first"), absmean("v3_first"), n=n_fs, unit="laps",
          note=f"over {[r['event'] for r in non_sc]}")
    S.add("first_stop_bias", "Mean signed first stop − field median (laps, non-SC weekends)",
          _mean([r["v1_first"] for r in non_sc]), _mean([r["v2_first"] for r in non_sc]),
          _mean([r["v3_first"] for r in non_sc]), band=(0.0, 0.0), n=n_fs, unit="laps")
    S.add("first_stop_abs_err_tyre_optimal", "Same, the tyre-optimal plan (no prior, no position term)",
          None, absmean("v2_tyre_first"), absmean("tyre_first"), n=n_fs, unit="laps")
    S.add("first_stop_abs_err_v3_no_prior", "Same, V3 with the first-stop prior switched off (kappa = 0)",
          None, absmean("v2_first"), absmean("v3_no_prior_first"), n=n_fs, unit="laps",
          note="the V2 column is V2's shipped answer, the closest published equivalent")
    S.add("first_stops_inside_window", "Share of the field's first stops inside the model's window",
          _mean([r["v1_win1"] for r in D]), _mean([r["v2_win1"] for r in D]),
          _mean([r["v3_win1"] for r in D]), lower_better=False, n=n_ev)
    fs_rows = [{"event": r["event"], "tyre": r["tyre_first"], "v2": r["v2_first"],
                "v3_no_prior": r["v3_no_prior_first"], "v3": r["v3_first"], "sc_set": r["sc_set"]} for r in D]
    fig2_firststop(fs_rows)

    S.add("oracle_regret_tool", "Oracle regret of the tool's plan (s, true race rates, same cost structure)",
          _mean([r["regret_v1"] for r in D]), _mean([r["regret_v2"] for r in D]),
          _mean([r["regret_v3"] for r in D]), n=n_ev, unit="s",
          note="all three plans priced on this run's oracle, so the comparison is like for like")
    fld = {tag: _mean([((((s or {}).get(k) or {}).get("oracle") or {}).get("regret_s") or {}).get("field_modal")
                       for k in events]) for tag, s in (("v1", stg_v1), ("v2", stg_v2))}
    wnr = {tag: _mean([((((s or {}).get(k) or {}).get("oracle") or {}).get("regret_s") or {}).get("winner")
                       for k in events]) for tag, s in (("v1", stg_v1), ("v2", stg_v2))}
    S.add("oracle_regret_field", "Oracle regret of the field's modal plan (s)",
          fld["v1"], fld["v2"], _mean([r["regret_field"] for r in D]), n=n_ev, unit="s",
          note=("each build prices the field's plan on its own oracle (the oracle inherits that build's "
                "pace offsets), so this row says how hard the race was, not how good the tool is"))
    S.add("oracle_regret_winner", "Oracle regret of the winner's plan (s)",
          wnr["v1"], wnr["v2"], _mean([r["regret_winner"] for r in D]), n=n_ev, unit="s",
          note="as above: a property of the race and of the build's oracle, not of the recommendation")
    S.add("oracle_beats_field", "Weekends where the tool's plan beats the field's modal plan on the oracle",
          _count((_num(r["regret_v1"]) is not None and _num(r["regret_field"]) is not None
                  and _num(r["regret_v1"]) < _num(r["regret_field"])) for r in D),
          _count((_num(r["regret_v2"]) is not None and _num(r["regret_field"]) is not None
                  and _num(r["regret_v2"]) < _num(r["regret_field"])) for r in D),
          _count((_num(r["regret_v3"]) is not None and _num(r["regret_field"]) is not None
                  and _num(r["regret_v3"]) < _num(r["regret_field"])) for r in D),
          lower_better=False, kind="count", n=n_ev,
          note="all three plans priced on this run's single oracle")

    lrows = []
    for r in D:
        for lf in (r["life"] or []):
            lrows.append({"event": r["event"], "compound": lf.get("compound"),
                          "v3": lf.get("ratio_to_max"), "v2": lf.get("v2_ratio_to_max"),
                          "v1": lf.get("baseline_ratio_to_max"), "bound_by": lf.get("bound_by"),
                          "obs_max": lf.get("obs_max"), "life": lf.get("pred_life_at_push"),
                          "uncapped": lf.get("pred_life_uncapped"),
                          "grip_budget_s": lf.get("grip_budget_s"), "v2_grip_budget_s": lf.get("v2_grip_budget_s")})
    L = pd.DataFrame(lrows)
    out["life"] = lrows
    fig4_life(L)
    n_cw = len(lrows) or 1
    ow_cw = n_cw / max(n_ev, 1)        # one weekend's worth of compound-weekends
    S.add("tyre_life_overstatements", f"Compound-weekends where predicted life exceeds the longest stint run (of {n_cw})",
          _count(_num(x["v1"]) is not None and _num(x["v1"]) > 1.0 for x in lrows),
          _count(_num(x["v2"]) is not None and _num(x["v2"]) > 1.0 for x in lrows),
          _count(_num(x["v3"]) is not None and _num(x["v3"]) > 1.0 for x in lrows),
          kind="count", n=n_ev, one_weekend=ow_cw)
    S.add("tyre_life_understatements", f"Compound-weekends where it is shorter than the longest stint run (of {n_cw})",
          _count(_num(x["v1"]) is not None and _num(x["v1"]) < 1.0 for x in lrows),
          _count(_num(x["v2"]) is not None and _num(x["v2"]) < 1.0 for x in lrows),
          _count(_num(x["v3"]) is not None and _num(x["v3"]) < 1.0 for x in lrows),
          kind="count", n=n_ev, one_weekend=ow_cw,
          note="an under-statement is a different failure from an over-statement, not its mirror")
    S.add("tyre_life_median_ratio", "Median predicted life ÷ longest stint run (1.0 is honest)",
          _median([x["v1"] for x in lrows]), _median([x["v2"] for x in lrows]),
          _median([x["v3"] for x in lrows]), band=(1.0, 1.0), n=n_ev)

    pv = [r["per_driver_variants"] for r in D]
    S.add("percar_same_shape_share", "Share of cars whose own plan shares the field plan's shape",
          _mean([(r["per_driver_v1"] or {}).get("share_same_shape_as_field") for r in D]),
          _mean([(r["per_driver_v2"] or {}).get("share_same_shape_as_field") for r in D]),
          _mean([(r["per_driver"] or {}).get("share_same_shape_as_field") for r in D]),
          lower_better=False, n=n_ev,
          note="informative about the field, not about the car: a high share means the per-car terms moved little")
    S.add("percar_first_stop_spread", "Per-car first-stop spread within a weekend (laps)",
          _mean([(r["per_driver_v1"] or {}).get("first_stop_spread") for r in D]),
          _mean([(r["per_driver_v2"] or {}).get("first_stop_spread") for r in D]),
          _mean([(r["per_driver"] or {}).get("first_stop_spread") for r in D]),
          lower_better=False, n=n_ev, unit="laps",
          note="wider is only better if it is right; see the Spearman rows")
    v2_rho = pooled_of(acc_v2, "sealed_driver").get("spearman_mean")
    for key, variant, label in (("percar_spearman_hist", "sealed_driver_hist", "LOO race factors (V2's per-car term)"),
                                ("percar_spearman_practice_dev", "sealed_driver_practice_dev", "this weekend's own dev[d,c]"),
                                ("percar_spearman_team_pooled", "sealed_driver_team_pooled", "team-pooled dev (V3 ships this)")):
        S.add(key, f"Spearman, predicted vs observed stint rate — per-car scale: {label}",
              None, v2_rho, pooled["all_variants_v3"].get(variant, {}).get("spearman_mean"),
              lower_better=False, n=n_ev,
              note="V2's column is its own `sealed_driver` variant, the only per-car scale it had")
    S.add("stint_rank_spearman", "Spearman, predicted vs observed stint rate (field model)",
          pooled_of(acc, "baseline_sealed[race_sigma]").get("spearman_mean"),
          pooled_of(acc, "v2_sealed[race_sigma]").get("spearman_mean"),
          p3.get("spearman_mean"), lower_better=False, n=n_ev)

    cd = [r["cliff_detector"] for r in D if r["cliff_detector"]]
    out["cliff_detector"] = {r["event"]: r["cliff_detector"] for r in D}
    S.add("collapse_pred_abs_err_laps", "Predicted collapse lap (budget ÷ rate) vs the detected knee (laps)",
          None, None, _mean([c.get("mean_abs_error_laps") for c in cd]), n=n_ev, unit="laps",
          note="V3 only: V1 and V2 had no within-stint collapse detector")
    S.add("collapse_n_detected", "Stints the detector calls a pace collapse",
          None, None, sum(int(c.get("n_collapse") or 0) for c in cd), kind="count", n=n_ev, lower_better=False,
          note="V3 only")
    def _fe_gap(r):
        b = r["bayes_vs_stint_fe_pooled"] or {}
        d = _num(b.get("diff"))
        if d is not None:
            return abs(d)
        x, y = _num(b.get("bayes")), _num(b.get("stint_fe"))
        return abs(x - y) if (x is not None and y is not None) else None

    S.add("bayes_vs_stint_fe_slope_gap", "|Bayes pooled slope − stint-FE baseline| (s/lap; the V3 gate, ≤ 0.06)",
          None, None, _mean([_fe_gap(r) for r in D]), n=n_ev, unit="s/lap",
          note="V3 only: it replaces V2's MixedLM range gate, which failed at three low-degradation circuits")
    S.add("counterfactual_over_30s", "Counterfactual 'seconds lost' figures above 30 s",
          sum(int((r["cf"] or {}).get("baseline_n_over_30s") or 0) for r in D),
          sum(int((r["cf"] or {}).get("v2_n_over_30s") or 0) for r in D),
          sum(int((r["cf"] or {}).get("n_over_30s") or 0) for r in D), kind="count", n=n_ev)

    md.append("### Decisions per weekend\n")
    md.append("| Weekend | V1 plan | V2 plan | V3 plan | Tyre-optimal | Field modal (share V1→V2→V3) | Winner | "
              "Stops=mode | Start=majority | First stop − field: tyre / V2 / V3-no-prior / V3 | "
              "Regret V1 / V2 / V3 / field / winner (s) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")

    def yn(v):
        return "–" if v is None else ("yes" if v else "no")

    for r in D:
        sc = " (SC)" if r["sc_set"] else ""
        md.append(f"| {r['event'].split('-')[0].title()} | {r['v1_plan']} | {r['v2_plan']} | **{r['v3_plan']}** | "
                  f"{r['tyre_optimal']} | {r['field_modal']} ({f(r['v1_share'], 2)}→{f(r['v2_share'], 2)}→{f(r['v3_share'], 2)}) | "
                  f"{r['winner']} | {yn(r['v1_stops'])}→{yn(r['v2_stops'])}→{yn(r['v3_stops'])} | "
                  f"{yn(r['v1_start'])}→{yn(r['v2_start'])}→{yn(r['v3_start'])} | "
                  f"{f(r['tyre_first'], 0, True)} / {f(r['v2_first'], 0, True)} / "
                  f"{f(r['v3_no_prior_first'], 0, True)} / {f(r['v3_first'], 0, True)}{sc} | "
                  f"{f(r['regret_v1'], 1)} / {f(r['regret_v2'], 1)} / {f(r['regret_v3'], 1)} / "
                  f"{f(r['regret_field'], 1)} / {f(r['regret_winner'], 1)} |")
    md.append("")

    # ----------------------------------------------------------------- live
    LR = live_rows(live, live_v2, live_v1)
    out["live"] = LR
    fig5_live(LR)
    S.add("live_tick_mean_ms", "Live engine tick, whole field (ms, mean over the replays)",
          _mean([r["tick_v1"] for r in LR]), _mean([r["tick_v2"] for r in LR]),
          _mean([r["tick_v3"] for r in LR]), n=max(len(LR), 1), unit="ms")
    S.add("live_tick_p95_ms", "Live engine tick, p95 (ms)",
          _mean([r["p95_v1"] for r in LR]), _mean([r["p95_v2"] for r in LR]),
          _mean([r["p95_v3"] for r in LR]), n=max(len(LR), 1), unit="ms")
    S.add("live_tick_max_ms", "Live engine tick, worst lap (ms)",
          _mean([r["max_v1"] for r in LR]), _mean([r["max_v2"] for r in LR]),
          _mean([r["max_v3"] for r in LR]), n=max(len(LR), 1), unit="ms")
    S.add("live_stops_in_window", "Share of real stops inside the engine's window 3 laps earlier",
          _mean([r["inwin_v1"] for r in LR]), _mean([r["inwin_v2"] for r in LR]),
          _mean([r["inwin_v3"] for r in LR]), lower_better=False, n=max(len(LR), 1))
    S.add("live_stops_within_3_laps", "Share of real stops within 3 laps of the recommendation",
          _mean([r["within3_v1"] for r in LR]), _mean([r["within3_v2"] for r in LR]),
          _mean([r["within3_v3"] for r in LR]), lower_better=False, n=max(len(LR), 1))
    S.add("live_median_stop_err_laps", "Median |recommended in-lap − actual| (laps)",
          _mean([r["err_v1"] for r in LR]), _mean([r["err_v2"] for r in LR]),
          _mean([r["err_v3"] for r in LR]), n=max(len(LR), 1), unit="laps")
    S.add("live_box_now_cost_s", "'Box now' cost the lap before the real stop (s, median)",
          _mean([r["box_v1"] for r in LR]), _mean([r["box_v2"] for r in LR]),
          _mean([r["box_v3"] for r in LR]), n=max(len(LR), 1), unit="s")
    for sig in ("window", "box_now", "collapse"):
        S.add(f"live_{sig}_precision", f"Live signal precision: {sig} (a stop within 3 laps = true positive)",
              None, None, _mean([(r["signals"].get(sig) or {}).get("precision") for r in LR]),
              lower_better=False, n=max(len(LR), 1),
              note="V3 only: V1 and V2 reported no precision for any signal")
        S.add(f"live_{sig}_recall", f"Live signal recall: {sig}",
              None, (_mean([r["collapse_recall_v2"] for r in LR]) if sig == "collapse" else None),
              _mean([(r["signals"].get(sig) or {}).get("recall") for r in LR]),
              lower_better=False, n=max(len(LR), 1),
              note=("V2's number is its 'alarm preceded the stop' share, the only recall it reported"
                    if sig == "collapse" else "V3 only"))

    md.append("### Live engine (archived replays)\n")
    md.append("| | " + " | ".join(f"{SHORT.get(r['event'], r['event'])} V1→V2→V3" for r in LR) + " |")
    md.append("|---|" + "---|" * len(LR))
    for lab, k1, k2, k3, d in (("Tick mean (ms)", "tick_v1", "tick_v2", "tick_v3", 0),
                               ("Tick p95 (ms)", "p95_v1", "p95_v2", "p95_v3", 0),
                               ("Stops inside the window", "inwin_v1", "inwin_v2", "inwin_v3", 2),
                               ("Stops within 3 laps", "within3_v1", "within3_v2", "within3_v3", 2),
                               ("Median stop error (laps)", "err_v1", "err_v2", "err_v3", 1),
                               ("Box-now cost 1 lap before (s)", "box_v1", "box_v2", "box_v3", 1)):
        md.append(f"| {lab} | " + " | ".join(f"{f(r[k1], d)} → {f(r[k2], d)} → **{f(r[k3], d)}**" for r in LR) + " |")
    if LR:
        md.append("| Signal precision / recall (V3) | " + " | ".join(
            ", ".join(f"{s} {f((r['signals'].get(s) or {}).get('precision'), 2)}/"
                      f"{f((r['signals'].get(s) or {}).get('recall'), 2)}"
                      for s in ("window", "box_now", "collapse")) for r in LR) + " |")
    md.append("")

    # ---------------------------------------------------------- calibration
    g = (cal.get("global") or {})
    g2 = (cal_v2.get("global") or {})
    out["calibration"] = {
        "global": {k: v for k, v in g.items() if k not in ("sweeps", "final_scores", "driver_factor_detail", "raw")},
        "global_v2": {k: v for k, v in g2.items() if k not in ("sweeps", "final_scores", "driver_factor_detail", "raw")},
        "loo": {k: {kk: vv for kk, vv in v.items()
                    if kk in ("grip_budget_by_compound", "undercut_lambda", "plan_prior_tau_s",
                              "first_stop_kappa_s", "grid_start_penalty_s", "dirty_air_s_per_lap",
                              "manage_cost_s", "manage_wear_floor", "percar_mode")}
                for k, v in (cal.get("loo") or {}).items()},
        "dirty_air_by_circuit": g.get("dirty_air_by_circuit"),
        "dirty_air_used_per_weekend": {r["event"]: {"used": (r["calibration_used"] or {}).get("dirty_air_used"),
                                                   "source": (r["calibration_used"] or {}).get("dirty_air_source")}
                                       for r in D},
        "grip_budget_detail": g.get("grip_budget_detail"),
        "grip_budget_v2_estimator": (g.get("raw") or {}).get("budget_v2"),
        "first_stop_kappa_s": g.get("first_stop_kappa_s"),
        "team_factors": g.get("team_factors"),
        "n_driver_factor_ln_sd": len(g.get("driver_factor_ln_sd") or {}),
        "percar_mode": g.get("percar_mode"),
        "raw": g.get("raw"), "sweeps": g.get("sweeps"), "per_weekend": cal.get("per_weekend"),
        "n_driver_factors": len(g.get("driver_factors") or {}),
        "driver_factors_top": sorted(((k, v) for k, v in (g.get("driver_factors") or {}).items()),
                                     key=lambda t: t[1])[:3]
        + sorted(((k, v) for k, v in (g.get("driver_factors") or {}).items()), key=lambda t: -t[1])[:3]}
    fig6_calibration(cal)
    if cal.get("loo"):
        md.append("### Leave-one-out calibration (V3)\n")
        md.append("| Held out | Grip budget S / M / H (s) | Manage cost, floor | Grid | Dirty air | λ | τ | κ |")
        md.append("|---|---|---|---|---|---|---|---|")
        for k, v in list((cal.get("loo") or {}).items()) + [("global", g)]:
            b = v.get("grip_budget_by_compound") or {}
            md.append(f"| {k} | {f(b.get('SOFT'), 2)} / {f(b.get('MEDIUM'), 2)} / {f(b.get('HARD'), 2)} | "
                      f"{f(v.get('manage_cost_s'), 2)}, {f(v.get('manage_wear_floor'), 2)} | "
                      f"{f(v.get('grid_start_penalty_s'), 2)} | {f(v.get('dirty_air_s_per_lap'), 3)} | "
                      f"{f(v.get('undercut_lambda'), 3)} | {f(v.get('plan_prior_tau_s'), 2)} | "
                      f"{f(v.get('first_stop_kappa_s'), 2)} |")
        md.append("")

    # --------------------------------------------------- outlook / ablation
    out["outlook"] = {"v3": outl, "v2": outl_v2, "v1": outl_v1}
    osum = (outl or {}).get("_summary") or {}
    S.add("outlook_modal_match", "Prior-only outlook matching the field's modal sequence",
          _count((((outl_v1 or {}).get(k) or {}).get("best") or "") and
                 ((outl_v1 or {}).get(k) or {}).get("field_modal_seq") in (((outl_v1 or {}).get(k) or {}).get("best") or "")
                 for k in events),
          _count((((outl_v2 or {}).get(k) or {}).get("best") or "") and
                 ((outl_v2 or {}).get(k) or {}).get("field_modal_seq") in (((outl_v2 or {}).get(k) or {}).get("best") or "")
                 for k in events),
          osum.get("modal_match"), lower_better=False, kind="count", n=n_ev)
    S.add("outlook_modal_match_letters", "Same, with the plan prior read letter-for-letter (no C-number mapping)",
          None, osum.get("modal_match_letters"), osum.get("modal_match_letters"),
          lower_better=False, kind="count", n=n_ev,
          note="the V2 behaviour, reproduced inside the V3 run as `prior_only_letters`")
    S.add("outlook_mean_seq_share", "Prior-only outlook: mean share of the field on its plan",
          _mean([((outl_v1 or {}).get(k) or {}).get("seq_share") for k in events]),
          _mean([((outl_v2 or {}).get(k) or {}).get("seq_share") for k in events]),
          osum.get("mean_seq_share"), lower_better=False, n=n_ev)
    out["ablation"] = abl
    out["ablation_v2"] = abl_v2
    apooled = (abl or {}).get("_pooled") or {}
    if apooled:
        md.append("### Ablation — the same search with one term switched off (V3)\n")
        md.append("| Variant | Sequence run by anyone | Start = majority | Stops = mode | Mean field share | "
                  "Mean |first stop − field| (non-SC) | Life ratio (median) |")
        md.append("|---|---|---|---|---|---|---|")
        for t, v in apooled.items():
            md.append(f"| {t} | {v.get('seq_run_by_anyone')}/{v.get('n_events')} | "
                      f"{v.get('start_matches_majority')}/{v.get('n_events')} | "
                      f"{v.get('stops_match_mode')}/{v.get('n_events')} | {f(v.get('mean_seq_share'), 2)} | "
                      f"{f(v.get('mean_abs_first_err'), 1)} | {f(v.get('life_ratio_median'), 2)} |")
        md.append("")

    # ---------------------------------------------- speed / stability / apex
    out["speed"] = {"v3": speed, "v2": speed_v2, "v1": speed_v1}
    out["stability"] = {"v3": stab, "v2": stab_v2, "v1": stab_v1}
    out["apex"] = {"v3": apex, "v2": apex_v2, "v1": apex_v1}
    out["speed_table"] = {tag: {r["stage"]: r["seconds"] for r in ((sp or {}).get("stages") or [])}
                          for tag, sp in (("v3", speed), ("v2", speed_v2), ("v1", speed_v1))}
    S.add("fit_time_s", "Production posterior fit (s)",
          _stage(speed_v1, "production"), _stage(speed_v2, "production"), _stage(speed, "production"),
          kind="value", unit="s")
    S.add("refit_time_s", "Weekend refit fit, 2x800x800 (s)",
          _stage(speed_v1, ("2x800x800", "quick"), "2x800x800"),
          _stage(speed_v2, ("2x800x800", "weekend refit"), "2x800x800"),
          _stage(speed, ("2x800x800", "weekend refit"), "2x800x800"), kind="value", unit="s")
    S.add("search_time_s", "Strategy search, 500 draws, 1-lap grid, full objective (s)",
          _stage(speed_v1, ("strategy search 500 draws, 1-lap", "")),
          _stage(speed_v2, ("strategy search 500 draws, 1-lap", "")),
          _stage(speed, ("strategy search 500 draws, 1-lap", "")), kind="value", unit="s")
    S.add("peak_memory_mb", "Peak resident memory of the speed benchmark (MB)",
          (speed_v1 or {}).get("peak_rss_mb"), (speed_v2 or {}).get("peak_rss_mb"),
          (speed or {}).get("peak_rss_mb"), kind="value", unit="MB")
    rt3, rt2, rt1 = _runtime(None), _runtime(V2), _runtime(BASELINE)
    suite_note = SUITE_NOTE_V2 if not rt2 else ""
    if not rt1:
        suite_note = (suite_note + "; " if suite_note else "") + "V1 published no suite runtime"
    S.add("benchmark_runtime_s", "Whole benchmark suite, wall clock (s)",
          rt1.get("total_seconds"), rt2.get("total_seconds", V2_SUITE_SECONDS),
          rt3.get("total_seconds"), kind="value", unit="s", note=suite_note)
    out["runtime"] = {"v3": rt3, "v2": rt2 or {"total_seconds": V2_SUITE_SECONDS, "source": SUITE_NOTE_V2},
                      "v1": rt1 or {"total_seconds": None, "source": "V1 published no suite runtime"}}

    t3 = _pytest_counts(OUT / "pytest.log")
    t2 = _pytest_counts(V2 / "out" / "pytest.log")
    t1 = _pytest_counts(BASELINE / "out" / "pytest.log")
    pytest_exit = ((rt3.get("pytest") or {}).get("exit"))
    fail_note = ""
    if t3["failed"]:
        fail_note = f"{t3['failed']} test(s) FAILING in the V3 run"
    if pytest_exit not in (None, 0):
        fail_note = (fail_note + "; " if fail_note else "") + f"pytest exit {pytest_exit}"
    S.add("test_count", "Tests passing", t1["passed"], t2["passed"], t3["passed"],
          lower_better=False, kind="count", n=1, note=fail_note)
    S.add("tests_failed", "Tests failing", t1["failed"], t2["failed"], t3["failed"],
          kind="count", n=1, note="from the same pytest summary line as the pass count")
    S.add("tests_skipped", "Tests skipped", t1["skipped"], t2["skipped"], t3["skipped"],
          kind="count", n=1)
    out["tests"] = {"v3": {**t3, "exit": pytest_exit, "log": "bench/out/pytest.log"},
                    "v2": t2, "v1": t1}
    if fail_note:
        print(f"  WARNING: {fail_note} (see bench/out/pytest.log)")
    S.add("gates_failing", "Gates failing across the benchmarked weekends",
          sum(len(r["gates_failed_v1"]) for r in D), sum(len(r["gates_failed_v2"]) for r in D),
          sum(len(r["gates_failed"]) for r in D), kind="count", n=n_ev)
    out["gates"] = {r["event"]: {"v1": r["gates_failed_v1"], "v2": r["gates_failed_v2"], "v3": r["gates_failed"]}
                    for r in D}

    # The planner's lambda x kappa diagnostic, if it has been run: carried
    # through as written, because it answers a question the suite does not —
    # whether *any* weight on either term puts the first stop on the field's lap.
    fsg = _load("firststop_grid.json")
    if fsg:
        S.rows["first_stop_grid"] = {
            "label": "First-stop λ × κ diagnostic: mean |first stop − field| over the non-SC weekends",
            "unit": "laps", "lower_is_better": True, "verdict": "diagnostic",
            "mean_abs_first_stop_error": fsg.get("mean_abs_first_stop_error"),
            "reading": fsg.get("reading"), "note": fsg.get("note"),
            "source": "bench/out/firststop_grid.json"}
        out["first_stop_grid"] = fsg
        md.append("### First stop: λ × κ diagnostic\n")
        md.append("| λ / κ | mean \\|first stop − field\\| (laps) |")
        md.append("|---|---|")
        for k, v in (fsg.get("mean_abs_first_stop_error") or {}).items():
            md.append(f"| {k} | {f(v, 2)} |")
        md.append(f"\n{fsg.get('reading', '')}\n")

    # -------------------------------------------------------------- summary
    out["summary"] = S.rows
    head = [f"### V1 → V2 → V3, every headline metric ({n_ev} weekend"
            f"{'s' if n_ev != 1 else ''}: {', '.join(SHORT.get(k, k) for k in events)})\n",
            "| Metric | V1 | V2 | V3 | Δ V3−V2 | Verdict |", "|---|---|---|---|---|---|"]
    body = []
    for k, r in S.rows.items():
        if "per_event" in r or "v3" not in r:
            continue        # per-weekend blocks and diagnostics are not three-way rows
        if r.get("kind") == "count":
            def g(v):
                x = _num(v)
                return "–" if x is None else f"{x:g}"
            body.append(f"| {r['label']} | {g(r['v1'])} | {g(r['v2'])} | **{g(r['v3'])}** | "
                        f"{g(r['delta_v3_v2'])} | {r['verdict']} |")
            continue
        big = r.get("unit") in ("ms", "MB", "s") and abs(_num(r["v3"]) or 0) >= 10
        d = 1 if big else 3
        body.append(f"| {r['label']} | {f(r['v1'], d)} | {f(r['v2'], d)} | **{f(r['v3'], d)}** | "
                    f"{f(r['delta_v3_v2'], d, True)} | {r['verdict']} |")
    note = ([f"*Partial run: the aggregates above cover {events} only. The frozen builds' own pooled "
             f"rows (per-car Spearman, prior-only outlook) still cover all seven weekends, so those "
             f"three-way comparisons are not like for like until the whole suite has run.*", ""]
            if partial else [])
    md[0:0] = head + body + [""] + note + [f"*{SUMMARY_RULE}*", ""]

    dump("compare.json", out)
    (OUT / "compare.md").write_text("\n".join(md) + "\n")
    print("\n".join(head + body))
    print(f"\nwrote {OUT / 'compare.json'}, {OUT / 'compare.md'} and six figures under {FIG}")
    tally = {}
    for r in S.rows.values():
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    print("verdicts:", tally)


if __name__ == "__main__":
    main()
