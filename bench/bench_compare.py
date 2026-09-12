"""Before / after: the updated benchmark against the previous one.

Reads the new outputs under bench/out/ and the frozen previous run under
bench/baseline/ (its bench/out and its meta/posterior files), and writes

    bench/out/compare.json          every number the report quotes
    bench/out/compare.md            the comparison tables, as Markdown
    bench/out/fig/fig1_accuracy.png stint-rate MAE per weekend, before/after
    bench/out/fig/fig2_firststop.png first-stop offset vs the field, before/after
    bench/out/fig/fig3_life.png     predicted life / longest stint, before/after
    bench/out/fig/fig4_live.png     live-engine stop calls, before/after
    bench/out/fig/fig5_calibration.png the leave-one-out sweeps

Charts follow the dataviz method: one axis, thin marks, a legend for two or
more series, direct labels only where they carry the story, text in ink
tokens, a validated categorical palette (slot 1 blue = updated build, slot 2
orange = previous build), hairline gridlines.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import BASELINE, EVENTS, OUT, baseline_meta, baseline_out, dump, meta  # noqa: E402
from src.config import DATA_PROCESSED  # noqa: E402

FIG = OUT / "fig"
FIG.mkdir(parents=True, exist_ok=True)

# the validated reference palette (dataviz skill): slot 1 blue, slot 2 orange, slot 3 aqua
NEW, OLD, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
COMPOUND = {"SOFT": "#c2334a", "MEDIUM": "#eda100", "HARD": "#2a78d6"}
SHORT = {"australia-2026": "AUS", "japan-2026": "JPN", "barcelona-2026": "BCN", "austria-2026": "AUT",
         "belgium-2026": "BEL", "hungary-2026": "HUN", "italy-2026": "ITA"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False, "legend.fontsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def _bar_pair(ax, labels, old, new, *, ylabel, title, fmt="{:.3f}", ylim=None, ref=None, ref_label=None):
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, old, w, color=OLD, label="previous build", zorder=3)
    ax.bar(x + w / 2, new, w, color=NEW, label="updated build", zorder=3)
    for xi, (o, n) in enumerate(zip(old, new)):
        if o is not None and np.isfinite(o):
            ax.text(xi - w / 2, o, " " + fmt.format(o), ha="center", va="bottom", fontsize=7.5, color=INK2, rotation=90)
        if n is not None and np.isfinite(n):
            ax.text(xi + w / 2, n, " " + fmt.format(n), ha="center", va="bottom", fontsize=7.5, color=INK2, rotation=90)
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
        top = np.nanmax([v for v in list(old) + list(new) if v is not None] + [ref or 0])
        ax.set_ylim(0, top * 1.28)
    ax.legend(loc="upper right")


def accuracy_tables(acc, acc_prev) -> tuple:
    rows, per = [], {}
    for k in EVENTS:
        new = ((acc or {}).get("per_event") or {}).get(k) or {}
        old = ((acc_prev or {}).get("per_event") or {}).get(k) or {}
        nv, ov = new.get("variants", {}), old.get("variants", {})
        r = {"event": k,
             "old_sealed": ov.get("sealed", {}).get("rate_mae"), "new_sealed": nv.get("sealed", {}).get("rate_mae"),
             "old_practice": ov.get("practice_only", {}).get("rate_mae"), "new_practice": nv.get("practice_only", {}).get("rate_mae"),
             "oracle": nv.get("oracle_race", {}).get("rate_mae"),
             "new_bias": nv.get("sealed", {}).get("rate_bias"), "old_bias": ov.get("sealed", {}).get("rate_bias"),
             "new_cov90": nv.get("sealed", {}).get("rate_cov90"), "old_cov90": ov.get("sealed", {}).get("rate_cov90"),
             "new_width": nv.get("sealed", {}).get("rate_width90"), "old_width": ov.get("sealed", {}).get("rate_width90"),
             "new_geomean": nv.get("sealed_geomean", {}).get("rate_mae"), "new_perfect_t": nv.get("sealed_perfect_t", {}).get("rate_mae"),
             "new_driver_rho": nv.get("sealed_driver", {}).get("spearman"), "new_rho": nv.get("sealed", {}).get("spearman"),
             "old_rho": ov.get("sealed", {}).get("spearman"),
             "regime_new": (new.get("regime") or {}).get("ratio"), "regime_old": (baseline_meta(k) or {}).get("regime", {}).get("ratio"),
             "regime_self": (new.get("regime") or {}).get("self"),
             "n_stints": nv.get("sealed", {}).get("n_stints")}
        rows.append(r)
        per[k] = r
    t = pd.DataFrame(rows)
    pooled_new = {p["variant"]: p for p in ((acc or {}).get("pooled") or [])}
    pooled_old = {p["variant"]: p for p in ((acc_prev or {}).get("pooled") or [])}
    return t, per, pooled_new, pooled_old


def main() -> None:
    acc, acc_prev = _load("accuracy.json"), baseline_out("accuracy.json")
    stg, stg_prev = _load("strategy.json"), baseline_out("strategy.json")
    live, live_prev = _load("live.json"), baseline_out("live.json")
    outl, outl_prev = _load("outlook.json"), baseline_out("outlook.json")
    abl = _load("ablation.json")
    speed, speed_prev = _load("speed.json"), baseline_out("speed.json")
    stab, stab_prev = _load("stability.json"), baseline_out("stability.json")
    apex, apex_prev = _load("apex.json"), baseline_out("apex.json")
    cal = json.loads((DATA_PROCESSED / "calibration.json").read_text()) if (DATA_PROCESSED / "calibration.json").exists() else {}
    out = {}
    md = []

    # ---------------------------------------------------------------- accuracy
    t, per, pooled_new, pooled_old = accuracy_tables(acc, acc_prev)
    out["accuracy"] = {"per_event": per, "pooled_new": pooled_new, "pooled_old": pooled_old}
    labels = [SHORT[k] for k in t["event"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    _bar_pair(axes[0], labels, t["old_sealed"].tolist(), t["new_sealed"].tolist(), ylabel="stint-rate MAE (s/lap)",
              title="Sealed curves vs the race, per weekend", ref=0.15, ref_label="target 0.15")
    ax = axes[1]
    x = np.arange(len(labels))
    ax.plot(x, t["old_width"], color=OLD, lw=2, marker="o", ms=5, label="previous build", zorder=3)
    ax.plot(x, t["new_width"], color=NEW, lw=2, marker="o", ms=5, label="updated build", zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("90% interval width on the stint rate (s/lap)")
    ax.set_title("Interval width (coverage in the table)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_accuracy.png", dpi=170)
    plt.close(fig)

    md.append("### Accuracy per weekend (stint-rate MAE, s/lap)\n")
    md.append("| Weekend | Previous sealed | Updated sealed | Updated practice-only | Oracle | Updated bias | 90% rate coverage prev → new | Width prev → new | Regime prev → new (self) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in t.iterrows():
        def f(v, d=3):
            return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"
        md.append(f"| {r['event'].split('-')[0].title()} | {f(r['old_sealed'])} | **{f(r['new_sealed'])}** | {f(r['new_practice'])} | {f(r['oracle'])} | "
                  f"{f(r['new_bias'])} | {f(r['old_cov90'], 2)} → {f(r['new_cov90'], 2)} | {f(r['old_width'])} → {f(r['new_width'])} | "
                  f"{f(r['regime_old'], 2)} → {f(r['regime_new'], 2)} ({f(r['regime_self'], 2)}) |")
    md.append(f"| **Mean** | {t['old_sealed'].mean():.3f} | **{t['new_sealed'].mean():.3f}** | {t['new_practice'].mean():.3f} | {t['oracle'].mean():.3f} | "
              f"{t['new_bias'].mean():+.3f} | {t['old_cov90'].mean():.2f} → {t['new_cov90'].mean():.2f} | {t['old_width'].mean():.3f} → {t['new_width'].mean():.3f} | |\n")
    md.append("### Pooled over the 7 weekends, every variant\n")
    md.append("| Curve source | Rate MAE mean | max | Bias | 90% coverage | Width | Spearman |")
    md.append("|---|---|---|---|---|---|---|")
    order = ["oracle_race", "sealed", "sealed_perfect_t", "sealed_geomean", "sealed_driver", "practice_only", "practice_no_regime",
             "mixedlm_x_regime", "history_only", "season_loo", "zero",
             "baseline_sealed[race_sigma]", "baseline_sealed[old_sigma]", "baseline_practice_only[race_sigma]"]
    names = {"oracle_race": "Oracle (this race's own rates, in-sample)", "sealed": "**Updated sealed** (history fold-in fixed, temperature regime)",
             "sealed_perfect_t": "Updated sealed with a perfect race-temperature forecast", "sealed_geomean": "Updated sealed with the old pooled regime",
             "sealed_driver": "Updated sealed, scaled per driver (LOO race factor)", "practice_only": "Practice posterior × regime (no history)",
             "practice_no_regime": "Practice posterior, no regime transfer", "mixedlm_x_regime": "MixedLM slope × regime",
             "history_only": "Circuit history 2023–25 only", "season_loo": "Other 2026 races' mean rate", "zero": "Zero degradation",
             "baseline_sealed[race_sigma]": "Previous sealed, rescored with the race noise", "baseline_sealed[old_sigma]": "Previous sealed, as reported before",
             "baseline_practice_only[race_sigma]": "Previous practice-only, race noise"}
    for v in order:
        p = pooled_new.get(v)
        if not p:
            continue
        md.append(f"| {names.get(v, v)} | {p['rate_mae_mean']:.3f} | {p['rate_mae_max']:.3f} | {p['bias_mean']:+.3f} | {p['cov90_mean']:.2f} | {p['width_mean']:.3f} | {p['spearman_mean']:+.2f} |")
    md.append("")

    # ---------------------------------------------------------------- decisions
    drows = []
    for k in EVENTS:
        s = (stg or {}).get(k) or {}
        sp = (stg_prev or {}).get(k) or {}
        fs = s.get("first_stop") or {}
        drows.append({"event": k, "prev": sp.get("recommended"), "new": s.get("recommended"), "tyre_opt": s.get("tyre_optimal"),
                      "modal": s.get("field_modal_seq"), "winner": s.get("winner_seq"),
                      "prev_stops_ok": sp.get("stops_match_mode"), "new_stops_ok": s.get("stops_match_mode"),
                      "prev_share": sp.get("rec_seq_share"), "new_share": s.get("rec_seq_share"),
                      "prev_start_ok": sp.get("start_matches_majority"), "new_start_ok": s.get("start_matches_majority"),
                      "prev_first": fs.get("baseline_minus_field"), "new_first": fs.get("rec_minus_field"), "tyre_first": fs.get("tyre_minus_field"),
                      "sc_set": fs.get("sc_set"), "field_first": fs.get("field_median_green"),
                      "regret_prev": (s.get("oracle") or {}).get("regret_s", {}).get("baseline_tool"),
                      "regret_new": (s.get("oracle") or {}).get("regret_s", {}).get("tool"),
                      "regret_field": (s.get("oracle") or {}).get("regret_s", {}).get("field_modal"),
                      "regret_winner": (s.get("oracle") or {}).get("regret_s", {}).get("winner"),
                      "win1_prev": ((sp.get("pit_windows") or [{}])[0] or {}).get("share_inside"),
                      "win1_new": ((s.get("pit_windows") or [{}])[0] or {}).get("share_inside"),
                      "life": s.get("life"), "per_driver": s.get("per_driver"), "cf": s.get("counterfactual"),
                      "gates_failed": s.get("gates_failed")})
    D = pd.DataFrame(drows)
    out["decisions"] = drows
    # fig 2: first-stop offset vs the field
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(D))
    w = 0.26
    ax.bar(x - w, D["prev_first"].astype(float), w, color=OLD, label="previous build", zorder=3)
    ax.bar(x, D["tyre_first"].astype(float), w, color=THIRD, label="updated, tyre-optimal", zorder=3)
    ax.bar(x + w, D["new_first"].astype(float), w, color=NEW, label="updated, position-aware", zorder=3)
    ax.axhline(0, color="#c3c2b7", lw=1, zorder=2)
    ymax = np.nanmax(np.abs(np.concatenate([D["prev_first"].astype(float), D["tyre_first"].astype(float), D["new_first"].astype(float)])))
    ax.set_ylim(min(-2.5, ax.get_ylim()[0]), ymax * 1.25)
    for xi, sc in enumerate(D["sc_set"]):
        if sc:
            ax.text(xi, ymax * 1.22, "SC set\nthe stops", ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.set_xticks(x, [SHORT[k] for k in D["event"]])
    ax.set_ylabel("recommended first stop − field median (laps)")
    ax.set_title("First stop against the field's green-flag median")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(FIG / "fig2_firststop.png", dpi=170)
    plt.close(fig)
    # fig 3: life ratio
    lrows = []
    for r in drows:
        for lf in (r["life"] or []):
            lrows.append({"event": r["event"], "compound": lf["compound"], "new": lf.get("ratio_to_max"), "prev": lf.get("baseline_ratio_to_max"),
                          "bound_by": lf.get("bound_by"), "obs_max": lf.get("obs_max"), "life": lf.get("pred_life_at_push"),
                          "uncapped": lf.get("pred_life_uncapped")})
    L = pd.DataFrame(lrows)
    out["life"] = lrows
    if not L.empty:
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        xs = np.arange(len(L))
        ax.bar(xs - 0.2, L["prev"].astype(float).clip(upper=4.0), 0.36, color=OLD, label="previous build", zorder=3)
        ax.bar(xs + 0.2, L["new"].astype(float), 0.36, color=NEW, label="updated build", zorder=3)
        for xi, v in enumerate(L["prev"]):
            if v is not None and np.isfinite(v) and v > 4.0:
                ax.text(xi - 0.2, 4.0, f"{v:.0f}×", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(len(L) - 0.5, 1.0, " longest stint run", va="bottom", ha="right", fontsize=8, color=MUTED)
        ax.set_xticks(xs, [f"{SHORT[e]}\n{c[0]}" for e, c in zip(L["event"], L["compound"])], fontsize=7.5)
        ax.set_ylabel("predicted life ÷ longest stint run")
        ax.set_title("Tyre life against the longest stint any finisher ran")
        ax.set_ylim(0, 4.3)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG / "fig3_life.png", dpi=170)
        plt.close(fig)

    md.append("### Decisions per weekend\n")
    md.append("| Weekend | Previous plan | Updated plan | Tyre-optimal | Field modal (share prev → new) | Winner | Stops = mode prev → new | Start = majority prev → new | First stop − field: prev / tyre / new | Regret prev / new / field / winner (s) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")

    def yn(v):
        return "–" if v is None else ("yes" if v else "no")

    def fv(v, d=0):
        return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:+.{d}f}" if d == 0 else f"{v:.{d}f}"

    for r in drows:
        sc = " (SC)" if r["sc_set"] else ""
        md.append(f"| {r['event'].split('-')[0].title()} | {r['prev']} | **{r['new']}** | {r['tyre_opt']} | {r['modal']} ({fv(r['prev_share'], 2)} → {fv(r['new_share'], 2)}) | {r['winner']} | "
                  f"{yn(r['prev_stops_ok'])} → {yn(r['new_stops_ok'])} | {yn(r['prev_start_ok'])} → {yn(r['new_start_ok'])} | "
                  f"{fv(r['prev_first'])} / {fv(r['tyre_first'])} / {fv(r['new_first'])}{sc} | "
                  f"{fv(r['regret_prev'], 1)} / {fv(r['regret_new'], 1)} / {fv(r['regret_field'], 1)} / {fv(r['regret_winner'], 1)} |")
    md.append("")
    summ = {
        "stops_match_prev": int(sum(bool(r["prev_stops_ok"]) for r in drows)), "stops_match_new": int(sum(bool(r["new_stops_ok"]) for r in drows)),
        "seq_run_prev": int(sum((r["prev_share"] or 0) > 0 for r in drows)), "seq_run_new": int(sum((r["new_share"] or 0) > 0 for r in drows)),
        "seq_share_prev": float(np.mean([r["prev_share"] or 0 for r in drows])), "seq_share_new": float(np.mean([r["new_share"] or 0 for r in drows])),
        "start_prev": int(sum(bool(r["prev_start_ok"]) for r in drows)), "start_new": int(sum(bool(r["new_start_ok"]) for r in drows)),
        "first_err_prev": float(np.nanmean([abs(r["prev_first"]) for r in drows if r["prev_first"] is not None and not r["sc_set"]])),
        "first_err_new": float(np.nanmean([abs(r["new_first"]) for r in drows if r["new_first"] is not None and not r["sc_set"]])),
        "first_err_tyre": float(np.nanmean([abs(r["tyre_first"]) for r in drows if r["tyre_first"] is not None and not r["sc_set"]])),
        "first_bias_prev": float(np.nanmean([r["prev_first"] for r in drows if r["prev_first"] is not None and not r["sc_set"]])),
        "first_bias_new": float(np.nanmean([r["new_first"] for r in drows if r["new_first"] is not None and not r["sc_set"]])),
        "regret_prev": float(np.nanmean([r["regret_prev"] for r in drows if r["regret_prev"] is not None])),
        "regret_new": float(np.nanmean([r["regret_new"] for r in drows if r["regret_new"] is not None])),
        "regret_field": float(np.nanmean([r["regret_field"] for r in drows if r["regret_field"] is not None])),
        "beats_field_new": int(sum((r["regret_new"] or 9e9) < (r["regret_field"] or -9e9) for r in drows if r["regret_new"] is not None and r["regret_field"] is not None)),
        "beats_winner_new": int(sum((r["regret_new"] or 9e9) < (r["regret_winner"] or -9e9) for r in drows if r["regret_new"] is not None and r["regret_winner"] is not None)),
        "win1_prev": float(np.nanmean([r["win1_prev"] for r in drows if r["win1_prev"] is not None])),
        "win1_new": float(np.nanmean([r["win1_new"] for r in drows if r["win1_new"] is not None])),
        "life_over_prev": int(sum((x["prev"] or 0) > 1.0 for x in lrows)), "life_over_new": int(sum((x["new"] or 0) > 1.0 for x in lrows)),
        "life_median_prev": float(np.nanmedian([x["prev"] for x in lrows if x["prev"] is not None])),
        "life_median_new": float(np.nanmedian([x["new"] for x in lrows if x["new"] is not None])),
        "life_n": len(lrows),
        "cf_over30_prev": int(sum((r["cf"] or {}).get("baseline_n_over_30s") or 0 for r in drows)),
        "cf_over30_new": int(sum((r["cf"] or {}).get("n_over_30s") or 0 for r in drows)),
        "cf_n_new": int(sum((r["cf"] or {}).get("n") or 0 for r in drows)),
        "gates_failed_new": int(sum(len(r["gates_failed"] or []) for r in drows)),
        "per_driver": {r["event"]: r["per_driver"] for r in drows},
    }
    out["decision_summary"] = summ

    # ---------------------------------------------------------------- live
    lrows2 = []
    for k in ("hungary-2026", "barcelona-2026"):
        n, o = (live or {}).get(k) or {}, (live_prev or {}).get(k) or {}
        lrows2.append({"event": k,
                       "tick_ms_prev": (o.get("tick_ms") or {}).get("mean"), "tick_ms_new": (n.get("tick_ms") or {}).get("mean"),
                       "tick_p95_prev": (o.get("tick_ms") or {}).get("p95"), "tick_p95_new": (n.get("tick_ms") or {}).get("p95"),
                       "inwin_prev": (o.get("stops") or {}).get("share_in_window"), "inwin_new": (n.get("stops") or {}).get("share_in_window"),
                       "err_prev": (o.get("stops") or {}).get("median_abs_err_laps"), "err_new": (n.get("stops") or {}).get("median_abs_err_laps"),
                       "within3_prev": (o.get("stops") or {}).get("share_err_within_3"), "within3_new": (n.get("stops") or {}).get("share_err_within_3"),
                       "boxnow_prev": (o.get("stops") or {}).get("median_box_now_delta_1_before"), "boxnow_new": (n.get("stops") or {}).get("median_box_now_delta_1_before"),
                       "alarm_prev": (o.get("stops") or {}).get("alarms_before_stop"), "alarm_new": (n.get("stops") or {}).get("alarms_before_stop"),
                       "n_stops": (n.get("stops") or {}).get("n"),
                       "m_final_prev": ((o.get("regime") or {}).get("final") or [None, None])[1], "m_final_new": ((n.get("regime") or {}).get("final") or [None, None])[1],
                       "m_self": (n.get("regime") or {}).get("self_measured_offline"),
                       "n_options_prev": o.get("n_options_leader_median"), "n_options_new": n.get("n_options_leader_median")})
    out["live"] = lrows2
    if live:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
        labs = [SHORT[r["event"]] for r in lrows2]
        _bar_pair(axes[0], labs, [r["tick_ms_prev"] for r in lrows2], [r["tick_ms_new"] for r in lrows2],
                  ylabel="mean tick (ms, whole field)", title="Live engine latency per lap", fmt="{:.0f}")
        _bar_pair(axes[1], labs, [r["inwin_prev"] for r in lrows2], [r["inwin_new"] for r in lrows2],
                  ylabel="share of real stops inside the window (3 laps before)", title="Stop calls", fmt="{:.2f}", ylim=(0, 1))
        fig.tight_layout()
        fig.savefig(FIG / "fig4_live.png", dpi=170)
        plt.close(fig)

    # ---------------------------------------------------------------- calibration sweeps
    if cal:
        g = cal.get("global") or {}
        sw = g.get("sweeps") or {}
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
        for ax, name, ylab, key in ((axes[0], "lambda", "mean |first stop − field| (laps)", "mean_abs_first_err"),
                                    (axes[1], "tau", "mean share of the field on the plan", "mean_seq_share"),
                                    (axes[2], "grid", "mean share of the field on the plan", "mean_seq_share")):
            rows = sw.get(name) or []
            if not rows:
                continue
            xs = [r[name] for r in rows]
            ys = [r.get(key) for r in rows]
            ax.plot(xs, ys, color=NEW, lw=2, marker="o", ms=5, zorder=3)
            chosen = {"lambda": g.get("undercut_lambda"), "tau": g.get("plan_prior_tau_s"), "grid": g.get("grid_start_penalty_s")}[name]
            if chosen is not None:
                ax.axvline(chosen, color=MUTED, lw=1, ls=(0, (4, 3)))
                ax.text(chosen, ax.get_ylim()[0], f" chosen {chosen:g}", fontsize=8, color=MUTED, va="bottom")
            ax.set_xlabel({"lambda": "undercut-exposure weight λ", "tau": "plan-prior weight τ (s per nat)", "grid": "grid-start penalty (s per step)"}[name])
            ax.set_ylabel(ylab)
            ax.set_title({"lambda": "λ against the field's first stops", "tau": "τ against the field's plan shapes", "grid": "Grid penalty"}[name])
        fig.tight_layout()
        fig.savefig(FIG / "fig5_calibration.png", dpi=170)
        plt.close(fig)
        out["calibration"] = {"global": {k: v for k, v in g.items() if k not in ("sweeps", "final_scores", "driver_factor_detail", "raw")},
                              "loo": {k: {kk: vv for kk, vv in v.items() if kk in ("grip_budget_by_compound", "undercut_lambda", "plan_prior_tau_s",
                                                                                 "grid_start_penalty_s", "dirty_air_s_per_lap", "manage_cost_s", "manage_wear_floor")}
                                      for k, v in (cal.get("loo") or {}).items()},
                              "raw": g.get("raw"), "sweeps": sw, "per_weekend": cal.get("per_weekend"),
                              "n_driver_factors": len(g.get("driver_factors") or {}),
                              "driver_factors_top": sorted(((k, v) for k, v in (g.get("driver_factors") or {}).items()), key=lambda t: t[1])[:3]
                              + sorted(((k, v) for k, v in (g.get("driver_factors") or {}).items()), key=lambda t: -t[1])[:3]}
        md.append("### Leave-one-out calibration\n")
        md.append("| Held out | Grip budget S / M / H (s) | Manage cost, floor | Grid penalty | Dirty air | λ | τ |")
        md.append("|---|---|---|---|---|---|---|")
        for k, v in list((cal.get("loo") or {}).items()) + [("global", g)]:
            b = v.get("grip_budget_by_compound") or {}
            md.append(f"| {k} | {b.get('SOFT', float('nan')):.2f} / {b.get('MEDIUM', float('nan')):.2f} / {b.get('HARD', float('nan')):.2f} | "
                      f"{v.get('manage_cost_s'):.2f}, {v.get('manage_wear_floor'):.2f} | {v.get('grid_start_penalty_s'):.2f} | "
                      f"{v.get('dirty_air_s_per_lap'):.3f} | {v.get('undercut_lambda'):.3f} | {v.get('plan_prior_tau_s'):.2f} |")
        md.append("")

    # ---------------------------------------------------------------- outlook / ablation / speed / stability / apex
    out["outlook"] = {k: {"new": (outl or {}).get(k), "prev": (outl_prev or {}).get(k)} for k in EVENTS}
    out["ablation"] = abl
    out["speed"] = {"new": speed, "prev": speed_prev}
    out["stability"] = {"new": stab, "prev": stab_prev}
    out["apex"] = {"new": apex, "prev": apex_prev}
    if speed:
        st_new = {r["stage"]: r["seconds"] for r in speed.get("stages", [])}
        st_old = {r["stage"]: r["seconds"] for r in (speed_prev or {}).get("stages", [])}
        out["speed_table"] = {"new": st_new, "prev": st_old}

    dump("compare.json", out)
    (OUT / "compare.md").write_text("\n".join(md))
    print("\n".join(md))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
