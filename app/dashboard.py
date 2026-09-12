"""degless — tyre degradation intelligence.

The app fits nothing.  Everything here is read from parquet/JSON written by
`scripts/10_pipeline.py`, so cold start is fast and nothing can break on stage.

Every view leads with the answer — one headline and a few number tiles — then
the charts behind it, each on a black card whose title says what it shows.
Explanations live in hover tooltips and a collapsed "How this works".  The look
(TGR Haas red, black and white) is defined in `app/theme.py` and `app/theme.css`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, EVENTS, current_event  # noqa: E402
from app.desk_tab import render_desk  # noqa: E402
from app.live_tab import render_now, render_strip, render_system, weekend_status  # noqa: E402
from app.theme import (  # noqa: E402
    BLACK, DEFS, DIM, FAINT, HAIR, LADDER, LETTER, MUTED, WHITE, badge, badges, card, ccol, chart, finite,
    fmt, headline, how, in_ladder, inject_css, ink_on, label_text, more, notice, palette, pct, rgba,
    saving_word, stops_word, style, tiles, tyre_pill, window_line,
)
from src.outlook import load_outlook, load_timeline  # noqa: E402
from src.engineer import ask as engineer_ask  # noqa: E402
from src.engineer import brief as engineer_brief  # noqa: E402
from src.engineer import credential_source, model_name  # noqa: E402

st.set_page_config(page_title="degless", layout="wide", page_icon="🏁")
inject_css()


# --------------------------------------------------------------------------
# Cached loaders (pure disk reads)
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_meta(key: str) -> dict | None:
    p = DATA_PROCESSED / f"meta_{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_pq(name: str) -> pd.DataFrame:
    p = DATA_PROCESSED / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_weekend(key: str) -> dict | None:
    p = DATA_PROCESSED / f"weekend_{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def available_events() -> list:
    """Scored weekends, weekends with a pre-race model, and weekends with an outlook only."""
    return [k for k in EVENTS if (DATA_PROCESSED / f"meta_{k}.json").exists()
            or (DATA_PROCESSED / f"weekend_{k}.json").exists()
            or (DATA_PROCESSED / f"outlook_{k}.json").exists()]


def _event_kind(k: str) -> str:
    if (DATA_PROCESSED / f"meta_{k}.json").exists():
        return "scored"
    if (DATA_PROCESSED / f"weekend_{k}.json").exists():
        return "pre-race model"
    return "outlook"


KIND_WORD = {"scored": "after the race", "pre-race model": "before the race", "outlook": "forecast only"}


def _shim_weekend(w: dict) -> dict:
    """A weekend-only model has no race half yet; give the page neutral values
    for everything the scored view reads, and say so on screen."""
    m = dict(w)
    nan = float("nan")
    m.setdefault("score", {"mae": nan, "mae_lap": nan, "rmse": nan, "n_rate_stints": 0, "n_laps": 0,
                           "n_stints": 0, "mae_by_compound": {}, "coverage": {"0.9": nan},
                           "cliff": {}, "bias": nan, "bias_by_compound": {}, "regime_label": "race regime",
                           "passes_mae": False, "passes_coverage": False})
    m.setdefault("pit_stops_measured", 0)
    m.setdefault("counterfactual_top", [])
    m.setdefault("backtest", {})
    b = m.setdefault("bayes", {})
    b.setdefault("k_track_laponly_rel_sd", nan)
    slopes = {r["compound"]: r["slope_s_per_lap"] for r in b.get("slopes", [])}
    m.setdefault("prior_sensitivity", {c: {"2026": v, "2025": nan, "none": nan} for c, v in slopes.items()})
    le = m.setdefault("load_effect", {"exponent": 1.6})
    le.setdefault("start_multiplier", (838 / 803) ** le.get("exponent", 1.6))
    le.setdefault("flag_multiplier", (768 / 803) ** le.get("exponent", 1.6))
    rg = m.setdefault("regime", {})
    rg.setdefault("self_measured", {"ratio": None, "per_compound": {}})
    lad = m.setdefault("compound_ladder", {})
    lad.setdefault("self_measured", {"step_s": nan, "se": nan, "n_laps": 0})
    lad.setdefault("unladdered_slopes", {})
    m.setdefault("max_stint_laps", {})
    st_ = m.setdefault("strategy", {})
    for k, v in (("n_strategies", 0), ("n_scored", 0), ("n_draws", 0), ("pit_windows", []),
                 ("by_stops", []), ("best_plan", {}), ("life", []), ("ordering_spread_s", nan),
                 ("traffic_s_per_stop", 0.0), ("safety_car_credit_s", 0.0), ("warmup_s", 0.7),
                 ("grip_budget_s", 3.8), ("push", nan), ("implied_regime", nan), ("best", "—")):
        st_.setdefault(k, v)
    m["_weekend_only"] = True
    return m


def _shim_outlook(o: dict, key: str) -> dict:
    """A weekend with no practice fit at all: the outlook is the only model.
    Give the page the shape of a weekend model, from the outlook's numbers,
    and mark it so the tabs that need a fit say so instead of drawing air."""
    nan = float("nan")
    st_ = o.get("strategy") or {}
    life = st_.get("life") or {}
    caps = st_.get("max_stint") or {}
    life_rows = [{"compound": c, "knee_lap": v.get("life_full_push", nan), "life_laps": v.get("life_laps", nan),
                  "life_lo": v.get("life_lo", nan), "life_hi": v.get("life_hi", nan),
                  "deg_s_per_lap": v.get("deg_s_per_lap", nan),
                  "max_stint_laps": float(caps.get(c, nan)), "practice_support_laps": nan}
                 for c, v in life.items()]
    ev_ = EVENTS[key]
    m = {
        "event": key, "event_name": o.get("event_name", ev_.name), "n_race_laps": o.get("n_race_laps", ev_.n_race_laps),
        "sessions_used": [], "sealed_file": "", "sealed_sha256": "",
        "pit_loss_s": float(o.get("pit_loss_s", nan)), "pit_loss_source": o.get("pit_loss_source", ""),
        "pit_stops_measured": 0, "n_clean_laps": 0, "n_raw_laps": 0, "compound_counts": [], "gates": [],
        "bayes": {"max_rhat": nan, "n_divergences": 0, "slopes": [], "k_track_mean": nan, "k_track_sd": nan,
                  "k_track_rel_sd": nan, "comp_offset": {}},
        "regime": o.get("regime") or {}, "circuit_history": o.get("history") or {},
        "history_combination": o.get("combination") or [],
        "allocation": {"caps": o.get("allocation") or {}}, "max_stint_laps": caps,
        "age_support_by_compound": o.get("support") or {},
        "physics": {"derivation": "", "fuel_effect_s_per_lap": nan}, "evolution": {"range_s": nan},
        "strategy": {"n_strategies": st_.get("n_strategies", 0), "n_scored": st_.get("n_scored", 0),
                     "n_draws": st_.get("n_draws", 0), "best": st_.get("best", "—"),
                     "best_plan": st_.get("best_plan", {}), "push": st_.get("push", nan),
                     "implied_regime": st_.get("implied_regime", nan), "grip_budget_s": st_.get("grip_budget_s", 3.8),
                     "by_stops": st_.get("by_stops", []), "life": life_rows, "pit_windows": st_.get("pit_windows", [])},
    }
    m = _shim_weekend(m)
    m["_outlook_only"] = True
    return m


# --------------------------------------------------------------------------
# Sidebar and page top
# --------------------------------------------------------------------------

events = available_events()
if not events:
    st.title("degless")
    notice("No processed data found. Run <code>make pipeline</code> first.", "alert")
    st.stop()

WEEKEND = weekend_status()
_cur = WEEKEND.get("event_key") or current_event().key
with st.sidebar:
    st.markdown("## degless")
    st.caption("Tyre wear and race strategy · 2026")
    key = st.selectbox(
        "Race weekend", events, index=(events.index(_cur) if _cur in events else len(events) - 1),
        format_func=lambda k: f"{EVENTS[k].name} · {KIND_WORD[_event_kind(k)]}",
    )
meta = load_meta(key)
WEEKEND_ONLY = meta is None
OUTLOOK = load_outlook(key)
OUTLOOK_ONLY = False
if WEEKEND_ONLY:
    _w = load_weekend(key)
    if _w is not None:
        meta = _shim_weekend(_w)
    else:
        meta = _shim_outlook(OUTLOOK or {}, key)
        OUTLOOK_ONLY = True
ev = EVENTS[key]
sc = meta["score"]
strat_meta = meta["strategy"]
best_plan = strat_meta.get("best_plan", {})
regime = meta.get("regime", {})
ladder = meta.get("compound_ladder", {})
n_pass = sum(g["pass"] for g in meta["gates"])
n_gate = len(meta["gates"])

with st.sidebar:
    if not OUTLOOK_ONLY:
        badge(f"Model checks {n_pass}/{n_gate}", "ok" if n_pass == n_gate else "alert",
              help="Automatic checks on the data, the fit and the plan. The Accuracy tab lists them.")
        if not WEEKEND_ONLY:
            st.metric("Average error", f"{sc['mae']:.2f} s/lap",
                      delta="on target" if sc["passes_mae"] else "over target", delta_color="off", delta_arrow="off",
                      help="How far the predicted tyre wear per lap was from the race, averaged over every "
                           "stint. Target: under 0.15 s/lap.")
    st.divider()
    render_system(WEEKEND)

if OUTLOOK_ONLY:
    DATA_BADGE = ("Forecast only · no practice yet", "info")
elif WEEKEND_ONLY:
    DATA_BADGE = ("Before the race · built on " + ", ".join(meta.get("sessions_used", [])), "info")
else:
    DATA_BADGE = ("After the race · prediction checked", "info")
st.title(ev.name, anchor=False)
render_strip(WEEKEND, DATA_BADGE)


def _event_names(keys) -> str:
    return ", ".join(EVENTS[k].name if k in EVENTS else str(k) for k in keys)


# ==========================================================================
# RACE PLAN — the decision
# ==========================================================================


def _tbl(kind: str) -> pd.DataFrame:
    # An outlook-only weekend has the same tables, written by the outlook.
    return load_pq(f"outlook_{key}_{kind}.parquet" if OUTLOOK_ONLY else f"{kind}_{key}.parquet")


def _window_title(windows: list) -> str:
    if not windows:
        return "Cost of stopping a lap early or late"
    if len(windows) == 1:
        w = windows[0]
        return f"Pit lap {w['recommended']} — laps {w['lo']}–{w['hi']} cost under 1 s"
    return "Under 1 s lost anywhere in " + " and ".join(f"laps {w['lo']}–{w['hi']}" for w in windows)


def _stops_title(b: pd.DataFrame) -> str:
    b = b.sort_values("delta_s")
    first = b.iloc[0]
    rest = " · ".join(f"{int(r['n_stops'])}-stop +{r['delta_s']:.1f} s" for _, r in b.iloc[1:].iterrows())
    return f"{int(first['n_stops'])}-stop fastest" + (f" · {rest}" if rest else "")


def _stint_chart(lanes: list, windows: list, pits: list):
    fig = go.Figure()
    for i, (label, df) in enumerate(lanes):
        y = len(lanes) - 1 - i
        alpha = 0.95 if i == 0 else 0.55
        for _, r in df.iterrows():
            L = int(r["laps"])
            comp = str(r["compound"])
            col = ccol(comp)
            letter = LETTER.get(comp, comp[:1])
            fig.add_trace(go.Bar(
                x=[L], y=[y], base=[int(r["start_lap"]) - 1], orientation="h",
                marker=dict(color=rgba(col, alpha), line=dict(color=BLACK, width=2)),
                width=0.62 if i == 0 else 0.46,
                # A label only where it fits upright; plotly rotates text that
                # overflows its bar, which is unreadable.
                text=(f"{letter} {L}" if L >= 9 else (letter if L >= 4 else "")),
                textposition="inside", insidetextanchor="middle", textangle=0, cliponaxis=False,
                constraintext="none", textfont=dict(color=ink_on(col, alpha), size=12), showlegend=False,
                hovertemplate=(f"<b>{label}</b><br>{comp.title()}, laps {int(r['start_lap'])}–"
                               f"{int(r['end_lap'])} ({L} laps)<extra></extra>")))
    for w in windows:
        fig.add_vrect(x0=w["lo"] - 1, x1=w["hi"], line_width=0, fillcolor=rgba(WHITE, 0.08), layer="below")
    for p in pits:
        fig.add_vline(x=p - 1, line=dict(color=WHITE, width=2, dash="dot"))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(lanes)))[::-1],
                     ticktext=[lab for lab, _ in lanes], showgrid=False)
    fig.update_xaxes(range=[0, ev.n_race_laps], dtick=10)
    # Overlay, not stack: each bar carries its own `base` (the stint's start lap).
    fig.update_layout(barmode="overlay", bargap=0.25)
    return style(fig, 90 + 42 * len(lanes), "", "race lap", legend=False)


def _pit_window_chart(pw: pd.DataFrame):
    fig = go.Figure()
    for k_stop, g in pw.groupby("stop"):
        g = g.sort_values("lap")
        fig.add_trace(go.Scatter(
            x=g["lap"], y=g["loss_s"], mode="lines", line=dict(color=WHITE, width=2), showlegend=False,
            hovertemplate=f"Stop {int(k_stop)} on lap %{{x}}<br>%{{y:.1f}} s lost<extra></extra>"))
        win = g[g["in_window"]]
        if not win.empty:
            fig.add_vrect(x0=win["lap"].min(), x1=win["lap"].max(), line_width=0,
                          fillcolor=rgba(WHITE, 0.08), layer="below")
        rec = g[g["is_recommended"]]
        if not rec.empty:
            fig.add_trace(go.Scatter(
                x=rec["lap"], y=rec["loss_s"], mode="markers", showlegend=False, hoverinfo="skip",
                marker=dict(color=WHITE, size=11, line=dict(color=BLACK, width=2))))
            fig.add_annotation(x=float(rec["lap"].iloc[0]), y=float(rec["loss_s"].iloc[0]), text=f"Stop {int(k_stop)}",
                               showarrow=False, yshift=18, font=dict(color=WHITE, size=12))
    fig.add_hline(y=1.0, line=dict(color=HAIR, dash="dash", width=1))
    fig.update_yaxes(range=[0, 12])
    return style(fig, 320, "time lost vs the best lap (s)", "pit lap", legend=False)


def _stops_chart(b: pd.DataFrame):
    b = b.sort_values("n_stops").reset_index(drop=True)
    best = float(b["delta_s"].min())
    ys = [f"{int(n)}-stop" for n in b["n_stops"]]
    fig = go.Figure()
    # A lollipop, not bars: the best plan sits at zero, and a bar of zero
    # length is an invisible mark carrying the most important row.
    for yv, d in zip(ys, b["delta_s"]):
        fig.add_shape(type="line", x0=0, x1=d, y0=yv, y1=yv, line=dict(color=HAIR, width=2))
    fig.add_trace(go.Scatter(
        x=b["delta_s"], y=ys, mode="markers+text", showlegend=False,
        marker=dict(size=15, color=[WHITE if d == best else DIM for d in b["delta_s"]],
                    line=dict(color=BLACK, width=2)),
        text=[("   fastest" if d == best else f"   +{d:.1f} s") + f" · wins {w:.0%}"
              for d, w in zip(b["delta_s"], b["win_prob_any"])],
        textposition="middle right", textfont=dict(color=WHITE, size=12),
        hovertemplate="%{y}: %{x:.1f} s slower than the best plan<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[-0.6, max(float(b["delta_s"].max()) * 2.1, 6)])
    return style(fig, 320, "", "slower than the best plan (s)", legend=False)


def _undercut_chart(uc: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(uc["leader_tyre_age"]) + list(uc["leader_tyre_age"])[::-1],
        y=list(uc["hi"]) + list(uc["lo"])[::-1], fill="toself", fillcolor=rgba(WHITE, 0.12),
        line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=uc["leader_tyre_age"], y=uc["gain_s"], mode="lines", line=dict(color=WHITE, width=2.5), showlegend=False,
        hovertemplate="Car ahead on %{x:.0f}-lap-old tyres<br>you gain %{y:+.2f} s a lap<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=HAIR, width=1))
    opens = uc[uc["p_positive"] > 0.5]
    if not opens.empty:
        fig.add_vrect(x0=float(opens["leader_tyre_age"].iloc[0]), x1=float(uc["leader_tyre_age"].max()),
                      line_width=0, fillcolor=rgba(WHITE, 0.06), layer="below", annotation_text="undercut works",
                      annotation_position="top left", annotation_font=dict(color=MUTED, size=11))
    return style(fig, 280, "gain per lap (s)", "age of the car ahead's tyres (laps)", legend=False)


def _laps_text(s) -> str:
    return str(s).strip("[]").replace(" ", "")


def _tyres_text(s) -> str:
    return " → ".join(x.title() for x in str(s).split("-"))


def _ranking_table(t: pd.DataFrame) -> pd.DataFrame:
    t = t.head(30)
    out = pd.DataFrame({"Plan": [label_text(s) for s in t["strategy"]]})
    if "push" in t:
        out["Tyre saving"] = [saving_word(p) for p in t["push"]]
    if "max_wear" in t:
        out["Deepest tyre use"] = t["max_wear"].clip(0, 1.2).to_numpy()
    out["Slower by (s)"] = t["delta_s"].round(1).to_numpy()
    if {"delta_p05", "delta_p95"} <= set(t.columns):
        out["Likely range (s)"] = [f"{a:+.1f} to {b:+.1f}" for a, b in zip(t["delta_p05"], t["delta_p95"])]
    if "win_prob" in t:
        out["Chance fastest"] = t["win_prob"].to_numpy()
    return out


def _ranking_cols() -> dict:
    # Built per render: the progress bar takes the active mode's ink.
    return {
        "Tyre saving": st.column_config.TextColumn(help=DEFS["saving"]),
        "Deepest tyre use": st.column_config.ProgressColumn(
            min_value=0, max_value=1, format="percent", color=palette()["ink"],
            help="The most of a tyre's usable life any stint uses. 100% is the drop-off."),
        "Slower by (s)": st.column_config.NumberColumn(help="Race time lost against the top plan, on average."),
        "Likely range (s)": st.column_config.TextColumn(help=DEFS["likely_range"]),
        "Chance fastest": st.column_config.NumberColumn(format="percent", help=DEFS["sims"]),
    }


def _tab_race_plan() -> None:
    plan, bystops, pw, field = _tbl("plan"), _tbl("bystops"), _tbl("pitwindow"), _tbl("fieldplan")
    uc, cf, ranking = _tbl("undercut"), _tbl("counterfactual"), _tbl("strategy")
    comps = list(best_plan.get("compounds", []))
    lens = list(best_plan.get("stint_lens", []))
    pits = list(best_plan.get("pit_laps", []))
    windows = strat_meta.get("pit_windows", [])
    if not comps:
        notice("No legal plan came out of the search for this weekend yet.", "alert")
        return
    n_stops = int(best_plan.get("n_stops", len(pits)))
    p_best = float(bystops.iloc[0]["win_prob_any"]) if not bystops.empty else float("nan")

    headline(f"{stops_word(n_stops)} " + " → ".join(tyre_pill(c) for c in comps), window_line(windows, pits),
             eyebrow="Forecast plan · no practice yet" if OUTLOOK_ONLY else "Recommended race plan")
    tiles("plan", [
        ("Stops", str(n_stops), DEFS["sims"], f"wins {pct(p_best)} of simulated races" if finite(p_best) else None),
        ("Pit on", " · ".join(f"lap {p}" for p in pits) or "—", DEFS["pit_window"],
         " · ".join(f"{w['lo']}–{w['hi']}" for w in windows) or None),
        ("Stints", " · ".join(f"{LETTER.get(c, c[:1])} {L}" for c, L in zip(comps, lens)), DEFS["stint"],
         f"{ev.n_race_laps} laps"),
        ("Pit loss", fmt(meta.get("pit_loss_s"), 1, " s"), DEFS["pit_loss"],
         f"{meta['pit_stops_measured']} stops measured" if meta.get("pit_stops_measured") else None),
    ])

    if not plan.empty:
        lanes = [("Our plan", plan)]
        if not field.empty:
            fin = field[field["finished_lap"] >= ev.n_race_laps - 2]
            lanes += [(d, fin[fin["driver"] == d]) for d in sorted(fin["driver"].unique())[:6]]
        words = " → ".join(f"{LETTER.get(str(r['compound']), '?')} {int(r['start_lap'])}–{int(r['end_lap'])}"
                           for _, r in plan.iterrows())
        with card("plan-clock", "Our plan vs how the field ran it" if len(lanes) > 1 else "Our plan on the race clock",
                  tip="Shaded bands are the pit windows; dotted lines are the planned stops."
                      + (" The rows below are drivers who finished this race." if len(lanes) > 1 else ""),
                  sub=words):
            chart(_stint_chart(lanes, windows, pits))

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        if not pw.empty:
            tight = [int(k) for k, g in pw.groupby("stop") if g["lap"].nunique() <= 3]
            tip = DEFS["pit_window"] + (
                f" Stop {', '.join(map(str, tight))} has almost no freedom: any other lap would run a tyre "
                "further than it has been run here." if tight else "")
            with card("plan-window", _window_title(windows), tip=tip):
                chart(_pit_window_chart(pw))
    with c2:
        if not bystops.empty:
            missing = sorted({1, 2, 3} - {int(n) for n in bystops["n_stops"]})
            tip = ("Race time lost against the best plan, on average over the simulated races. "
                   "“Wins” is how often that stop count came out fastest.")
            if missing:
                tip += (f" {', '.join(f'{m}-stop' for m in missing)} has no legal plan: the stints can't cover "
                        "the race without running a tyre further than it has been run here.")
            with card("plan-stops", _stops_title(bystops), tip=tip):
                chart(_stops_chart(bystops))

    with more():
        if not ranking.empty:
            st.markdown("**Every plan, ranked**")
            st.dataframe(_ranking_table(ranking), width="stretch", hide_index=True, column_config=_ranking_cols())
        if not uc.empty:
            caps = meta.get("max_stint_laps", {})
            cap = min([v for k, v in caps.items() if k in ("MEDIUM", "SOFT")]
                      or [float(uc["leader_tyre_age"].max())])
            uc = uc[uc["leader_tyre_age"] <= cap]
            opens = uc[uc["p_positive"] > 0.5]
            title = (f"The undercut works once the car ahead's tyres are {opens['leader_tyre_age'].iloc[0]:.0f}+ laps old"
                     if not opens.empty else "The undercut doesn't pay within the tyres' life")
            with card("plan-undercut", title,
                      tip=DEFS["undercut"] + " Here the car ahead is on MEDIUMs and you pit for SOFTs. "
                          "The band is the likely range."):
                chart(_undercut_chart(uc))
        if not cf.empty:
            top = cf.head(10).copy()
            lead = top.iloc[0]
            with card("plan-timing", f"{lead['driver']} lost the most to stop timing: {lead['loss_s']:.1f} s",
                      tip="Each driver's own tyres and stop count, with only the stop laps moved to the best ones. "
                          "This isolates the timing call the pit wall makes on the day."):
                fig = go.Figure(go.Bar(
                    x=top["loss_s"], y=top["driver"], orientation="h", width=0.55, marker=dict(color=DIM),
                    error_x=dict(type="data", symmetric=False, array=top["hi"] - top["loss_s"],
                                 arrayminus=top["loss_s"] - top["lo"], color=MUTED, thickness=1.2, width=4),
                    customdata=np.stack([top["compounds"].map(_tyres_text), top["actual_pit_laps"].map(_laps_text),
                                         top["model_pit_laps"].map(_laps_text)], axis=-1),
                    hovertemplate="<b>%{y}</b> (%{customdata[0]})<br>stopped on lap %{customdata[1]}<br>"
                                  "best was lap %{customdata[2]}<br>%{x:.1f} s lost<extra></extra>",
                    showlegend=False))
                fig.update_yaxes(autorange="reversed")
                chart(style(fig, 320, "", "seconds lost to stop timing", legend=False))
                st.dataframe(pd.DataFrame({
                    "Driver": top["driver"], "Tyres": top["compounds"].map(_tyres_text),
                    "Stopped on": top["actual_pit_laps"].map(_laps_text),
                    "Best laps": top["model_pit_laps"].map(_laps_text),
                    "Time lost (s)": top["loss_s"].round(1),
                    "Likely range (s)": [f"{a:.1f}–{b:.1f}" for a, b in zip(top["lo"], top["hi"])],
                }), width="stretch", hide_index=True)

    how(
        "Every legal plan is timed over hundreds of simulated races that share the same tyre-wear samples, "
        "so plans are compared like for like.",
        f"Included: tyre wear up to the drop-off, pace differences between tyres, pit loss "
        f"({fmt(meta.get('pit_loss_s'), 1, ' s')}), a slow first lap on cold tyres, heavier fuel early on, "
        "traffic after a stop, and the chance of a safety car.",
        "Not included: track position, the starting-tyre choice and the cars around you. Plans within about "
        "1 s of each other are effectively a tie — the window matters more than the exact lap.",
    )


# ==========================================================================
# TYRES
# ==========================================================================


def _wear_chart(curves: pd.DataFrame, life: pd.DataFrame, with_practice: bool):
    sub = curves[curves["variant"] == "2026_race"]
    if sub.empty:
        sub = curves[curves["variant"] == "2026"]
    fig = go.Figure()
    for cmp_ in in_ladder(sub["compound"].unique()):
        g = sub[sub["compound"] == cmp_].sort_values("tyre_age")
        col = ccol(cmp_)
        fig.add_trace(go.Scatter(
            x=list(g["tyre_age"]) + list(g["tyre_age"])[::-1], y=list(g["hi"]) + list(g["lo"])[::-1],
            fill="toself", fillcolor=rgba(col, 0.14), line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=g["tyre_age"], y=g["mean"], mode="lines", line=dict(color=col, width=2.5), showlegend=False,
            hovertemplate=f"{cmp_.title()} · lap %{{x:.0f}} of the stint<br>%{{y:.2f}} s slower than new<extra></extra>"))
        lab = g[g["tyre_age"] <= 34]
        if not lab.empty:
            lab = lab.iloc[-1]
            fig.add_annotation(x=lab["tyre_age"], y=lab["mean"], text=cmp_.title(), showarrow=False,
                               xanchor="center", yshift=13, font=dict(color=WHITE, size=12))
    if with_practice:
        pr = curves[curves["variant"] == "2026"]
        for cmp_ in in_ladder(pr["compound"].unique()):
            g = pr[pr["compound"] == cmp_].sort_values("tyre_age")
            fig.add_trace(go.Scatter(
                x=g["tyre_age"], y=g["mean"], mode="lines", showlegend=False,
                line=dict(color=ccol(cmp_), width=1.4, dash="dot"),
                hovertemplate=f"{cmp_.title()} in practice · lap %{{x:.0f}}<br>%{{y:.2f}} s<extra></extra>"))
    if not life.empty:
        for _, r in life.iterrows():
            fig.add_vline(x=r["knee_lap"], line=dict(color=rgba(ccol(r["compound"]), 0.5), width=1, dash="dash"))
    sup = meta.get("age_support_by_compound", {})
    x_max = 40
    if sup:
        edge = float(max(sup.values()))
        if edge < x_max:
            fig.add_vrect(x0=edge, x1=x_max, line_width=0, fillcolor=rgba(WHITE, 0.05), layer="below",
                          annotation_text="beyond practice data", annotation_position="top right",
                          annotation_font=dict(color=MUTED, size=11))
    fig.update_xaxes(range=[0, x_max])
    return style(fig, 420, "time lost vs a new tyre (s)", "laps on the tyre", legend=False)


def _life_chart(lf: pd.DataFrame):
    fig = go.Figure()
    for i, r in lf.iterrows():
        y = len(lf) - 1 - i
        c = str(r["compound"])
        fig.add_trace(go.Bar(
            x=[r["life_laps"]], y=[y], orientation="h", width=0.5, showlegend=False,
            marker=dict(color=rgba(ccol(c), 0.85), line=dict(color=BLACK, width=2)),
            hovertemplate=(f"{c.title()}: {r['life_laps']:.0f} laps with the plan's tyre saving "
                           f"(likely {r['life_lo']:.0f}–{r['life_hi']:.0f})<extra></extra>")))
        fig.add_trace(go.Scatter(
            x=[r["knee_lap"]], y=[y], mode="markers", showlegend=False,
            marker=dict(color=WHITE, size=11, symbol="diamond", line=dict(color=BLACK, width=2)),
            hovertemplate=f"{c.title()} flat out: drops off around lap {r['knee_lap']:.0f}<extra></extra>"))
        fig.add_annotation(x=r["life_laps"], y=y, xanchor="left", xshift=8, showarrow=False,
                           text=f"{r['life_laps']:.0f} with saving · {r['knee_lap']:.0f} flat out",
                           font=dict(color=MUTED, size=11))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(lf)))[::-1],
                     ticktext=[str(c).title() for c in lf["compound"]], showgrid=False)
    fig.update_xaxes(range=[0, float(lf["life_laps"].max()) * 1.75])
    return style(fig, 250, "", "laps", legend=False)


def _tyre_tiles(name: str, rows: list) -> None:
    """rows: dicts with compound, life, lo, hi, full_push, deg."""
    tiles(name, [(r["compound"], f"{r['life']:.0f} laps",
                  f"Likely range {r['lo']:.0f}–{r['hi']:.0f} laps with the plan's tyre saving. "
                  f"Driven flat out it drops off around lap {r['full_push']:.0f}.",
                  f"loses {r['deg']:.2f} s a lap") for r in rows])


def _tab_tyres() -> None:
    curves, knee, life = load_pq(f"curves_{key}.parquet"), load_pq(f"knee_{key}.parquet"), load_pq(f"life_{key}.parquet")
    lf = (life.set_index("compound").reindex(in_ladder(life["compound"])).reset_index()
          if not life.empty else life)
    if not lf.empty:
        longest = lf.sort_values("life_laps", ascending=False)
        headline(" · ".join(f"{r['compound']} ~{r['life_laps']:.0f} laps" for _, r in longest.iterrows()),
                 "How far each tyre goes at race pace with the plan's tyre saving", eyebrow="Tyre life")
        _tyre_tiles("tyres", [{"compound": r["compound"], "life": r["life_laps"], "lo": r["life_lo"],
                               "hi": r["life_hi"], "full_push": r["knee_lap"], "deg": r["deg_s_per_lap"]}
                              for _, r in lf.iterrows()])

    view = st.segmented_control("Show", ["Race", "Race + practice"], default="Race", required=True,
                                key="ty_view", label_visibility="collapsed")
    sub = curves[curves["variant"] == "2026_race"]
    if sub.empty:
        sub = curves[curves["variant"] == "2026"]
    at20 = {c: float(np.interp(20, g.sort_values("tyre_age")["tyre_age"], g.sort_values("tyre_age")["mean"]))
            for c, g in sub.groupby("compound")}
    with card("tyres-wear", "After 20 laps: " + " · ".join(f"{c.title()} +{at20[c]:.1f} s" for c in in_ladder(at20)),
              tip=DEFS["tyre_wear"] + " Solid lines are race pace, dotted lines the same tyres in practice. "
                  "Dashed verticals mark where each tyre drops off driven flat out; the shaded area is past the "
                  "oldest tyre anyone ran in practice."):
        chart(_wear_chart(curves, life, view == "Race + practice"))

    if not lf.empty:
        top = lf.loc[lf["life_laps"].idxmax()]
        with card("tyres-life", f"{str(top['compound']).title()} lasts longest: ~{top['life_laps']:.0f} laps with saving",
                  tip="Bars: how far each tyre goes with the plan's tyre saving. Diamonds: where it drops off "
                      "driven flat out."):
            chart(_life_chart(lf))

    with more():
        r = regime
        per = (r.get("self_measured", {}) or {}).get("per_compound", {})
        tiles("race-wear", [
            ("Race wear vs practice", f"{fmt(r.get('ratio'), 2)}×", DEFS["race_vs_practice"],
             f"likely {fmt(r.get('p05'), 2)}–{fmt(r.get('p95'), 2)}"),
            ("Measured on", "other races" if r.get("measured") else "default",
             "Taken from other 2026 races, never this weekend's: " + (_event_names(r.get("sources", [])) or "none")),
            ("This race, afterwards", f"{fmt((r.get('self_measured', {}) or {}).get('ratio'), 2)}×",
             "Measured on this weekend's race after the flag. Shown for comparison; never used for the prediction."),
        ])
        if per:
            df = pd.DataFrame([{"Tyre": c.title(), "Practice (s/lap)": round(v["practice_s_per_lap"], 3),
                                "Race (s/lap)": round(v["race_s_per_lap"], 3),
                                "Race vs practice": (f"{v['ratio']:.2f}×" if v.get("usable") else "—"),
                                "_o": LADDER.index(c) if c in LADDER else 9} for c, v in per.items()])
            st.dataframe(df.sort_values("_o").drop(columns="_o"), width="stretch", hide_index=True)

        slopes = {row["compound"]: row["slope_s_per_lap"] for row in meta["bayes"]["slopes"]}
        offs = ladder.get("fitted_offsets", meta["bayes"]["comp_offset"])
        if slopes:
            order = in_ladder(slopes)
            st.markdown("**Tyre pace order**", help="Softer tyres are always quicker and wear faster; "
                                                    "the model builds that order in and measures the size of each step.")
            st.dataframe(pd.DataFrame({
                "Tyre": [c.title() for c in order],
                "Wear in practice (s/lap)": [round(slopes.get(c, np.nan), 3) for c in order],
                "Pace vs softest (s/lap)": [round(offs.get(c, np.nan), 3) for c in order],
                "Drops off, flat out (lap)": [round(next((row["knee_lap"] for row in meta["bayes"]["slopes"]
                                                          if row["compound"] == c), np.nan), 1) for c in order],
            }), width="stretch", hide_index=True)
        ul = ladder.get("unladdered_slopes", {})
        if ul and not ladder.get("unladdered_ordered", True):
            notice("Without the built-in tyre order, these laps would make a softer tyre last longer than a "
                   "harder one — which is why the order is enforced.", "alert")

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            le = meta.get("load_effect", {})
            laps = np.arange(1, ev.n_race_laps + 1)
            mass = 768.0 + 70.0 * (1 - (laps - 1) / (ev.n_race_laps - 1))
            mult = (mass / 803.0) ** le.get("exponent", 5.0)
            with card("tyres-fuel", f"A full tank wears tyres {le.get('start_multiplier', 1):.2f}× as fast as mid-race",
                      tip="Heavier cars wear tyres faster, so stints get longer as the race goes on, and the plan "
                          "doesn't split the distance evenly."):
                fig = go.Figure(go.Scatter(x=laps, y=mult, mode="lines", line=dict(color=WHITE, width=2.5),
                                           showlegend=False, hovertemplate="lap %{x}<br>%{y:.2f}× wear<extra></extra>"))
                fig.add_hline(y=1.0, line=dict(color=HAIR, width=1))
                chart(style(fig, 240, "wear vs mid-race (×)", "race lap", legend=False))
        with c2:
            ks = knee[knee["variant"] == "2026"] if not knee.empty else knee
            if not ks.empty:
                with card("tyres-dropoff", "When each tyre drops off, driven flat out",
                          tip="Spread of the drop-off lap across the simulated races."):
                    fig = go.Figure()
                    for cmp_ in in_ladder(ks["compound"].unique()):
                        fig.add_trace(go.Histogram(x=ks[ks["compound"] == cmp_]["knee"], name=cmp_.title(),
                                                   opacity=0.62, nbinsx=45, marker_color=ccol(cmp_)))
                    fig.update_layout(barmode="overlay")
                    chart(style(fig, 240, "simulated races", "drop-off lap"))

        ps = pd.DataFrame(meta["prior_sensitivity"]).T
        if not ps.empty:
            order = in_ladder(ps.index)
            names = {"2026": "2026 fuel physics (used)", "2025": "2025 fuel physics", "none": "no fuel adjustment"}
            with card("tyres-fuelprior", "Without the fuel adjustment, tyre wear comes out wildly wrong",
                      tip="Tyre age and fuel burn rise together through a run, so without knowing how much fuel "
                          "costs per lap, the model can't tell tyre wear from the car getting lighter."):
                fig = go.Figure()
                for v, colr in (("2026", WHITE), ("2025", DIM), ("none", FAINT)):
                    fig.add_trace(go.Bar(x=[c.title() for c in order], y=[ps.loc[c, v] for c in order], name=names[v],
                                         marker=dict(color=colr, line=dict(color=BLACK, width=2)),
                                         hovertemplate="%{x} · " + names[v] + "<br>%{y:.2f} s/lap<extra></extra>"))
                fig.update_yaxes(type="log", tickmode="array", tickvals=[0.1, 0.2, 0.5, 1, 2, 5],
                                 ticktext=["0.1", "0.2", "0.5", "1", "2", "5"])
                chart(style(fig, 280, "tyre wear (s/lap, log scale)", ""))

    how(
        "Wear is measured on practice long runs with fuel burn and the track getting faster taken out, then "
        f"scaled to race conditions: races wear tyres about {fmt(regime.get('ratio'), 2)}× as fast as practice, "
        "measured on other 2026 races, never this one.",
        "Softer tyres always come out quicker and faster-wearing. That order is built in; the size of each step "
        "comes from the data.",
        "A tyre's life is how much time it can lose before it drops off, divided by how fast it wears — so a "
        "fast-wearing tyre can't also come out long-lasting.",
    )


def _outlook_tyre_fallback() -> None:
    notice("No practice yet: these tyre numbers come from the race forecast. Measured curves appear after the "
           "first practice session.")
    life = ((OUTLOOK or {}).get("strategy") or {}).get("life") or {}
    order = [c for c in LADDER if c in life]
    if not order:
        return
    headline(" · ".join(f"{c} ~{life[c]['life_laps']:.0f} laps" for c in sorted(order, key=lambda c: -life[c]["life_laps"])),
             "How far each tyre goes at race pace with the plan's tyre saving", eyebrow="Tyre life · forecast")
    _tyre_tiles("tyres-fc", [{"compound": c, "life": life[c]["life_laps"], "lo": life[c]["life_lo"],
                              "hi": life[c]["life_hi"], "full_push": life[c]["life_full_push"],
                              "deg": life[c]["deg_s_per_lap"]} for c in order])
    with card("tyres-fc-table", "Forecast by tyre"):
        st.dataframe(pd.DataFrame([{
            "Tyre": c.title(), "Wear (s/lap)": round(life[c]["deg_s_per_lap"], 3),
            "Likely wear (s/lap)": f"{life[c]['deg_lo']:.3f}–{life[c]['deg_hi']:.3f}",
            "Life with saving (laps)": round(life[c]["life_laps"]),
            "Likely life (laps)": f"{life[c]['life_lo']:.0f}–{life[c]['life_hi']:.0f}",
            "Life flat out (laps)": round(life[c]["life_full_push"]),
            "Pace vs softest (s/lap)": round(life[c]["pace_offset_s"], 3)} for c in order]),
            width="stretch", hide_index=True)


# ==========================================================================
# ACCURACY — how good the prediction was, and the data behind it
# ==========================================================================

GATE_WORDS = {
    "firewall blocks race data": "Race data locked out of the fit",
    "clean laps in 120-320": "Enough clean practice laps",
    "no non-green lap survives": "No yellow-flag or safety-car laps used",
    "all compounds valid": "Every tyre has usable data",
    "regime factor is plausible (0.15-1.0)": "Race-vs-practice wear factor is realistic",
    "track evolution identified & plausible": "Track-getting-faster effect is realistic",
    "MixedLM MEDIUM slope in 0.12-0.35": "Medium wear agrees with a simple cross-check",
    "apex channel available": "Corner-speed data available",
    "convergence: r_hat < 1.01 and zero divergences": "Model fit is stable",
    "Bayes vs MixedLM pooled degradation within 0.06 s/lap": "Main model agrees with the simple cross-check",
    "compound ladder ordered (softer = quicker and higher deg)": "Softer tyres come out quicker and wear faster",
    "sealed file verifies against its sha256": "Locked prediction file is untouched",
    "practice->race regime transfer removes the bias": "Race-vs-practice factor removes the lean",
    "stint degradation-rate MAE < 0.15 s/lap": "Average error under 0.15 s/lap",
    "90% coverage not below 0.80 (under-coverage is the failure)": "Likely range held often enough",
    "pit loss measured and plausible (12-35s)": "Pit loss measured and realistic",
    "recommended plan has realistic stint lengths": "Plan uses realistic stint lengths",
    "no stint in the recommended plan runs past the cliff": "No stint runs past the drop-off",
    "model does not assume more tyre management than was observed": "No more tyre saving than teams really do",
    "does not open the race on the hardest available compound": "Doesn't start on the hardest tyre",
    "model reproduces the measured net stint-level compound step": "Tyre pace gap matches the race",
    "recommended stop count is one the field actually ran": "Stop count is one the field ran",
    "every recommended stint length is one the compound was run to": "Every stint length has been run before",
}

CASCADE_WORDS = {
    "raw laps": "All practice laps", "ok_accurate": "Timing marked accurate", "ok_not_pit": "Not an in- or out-lap",
    "ok_green": "Green flag only", "ok_stint_len": "Part of a 6+ lap run", "ok_traffic": "Clear air (2 s+ gap)",
    "ok_not_slow": "Not a slow lap", "ok_compound": "Known tyre",
}


def _accuracy_scored() -> None:
    cov90 = sc["coverage"].get("0.9", float("nan"))
    bias = sc["bias"]
    lean = ("balanced" if abs(bias) < 0.02 else
            ("race wore tyres more" if bias > 0 else "race wore tyres less"))
    headline(f"Tyre wear predicted within {sc['mae']:.2f} s/lap",
             f"Target under 0.15 s/lap · checked on {sc['n_rate_stints']} race stints",
             eyebrow="How accurate the prediction was")
    tiles("acc", [
        ("Average error", f"{sc['mae']:.3f} s/lap",
         "How far the predicted tyre wear per lap was from what each race stint showed, averaged over all stints.",
         "✓ under 0.15" if sc["passes_mae"] else "⚠ over 0.15"),
        ("Lean", f"{bias:+.3f} s/lap",
         "The average of actual minus predicted wear. Positive: the race wore tyres more than predicted.", lean),
        ("Range held", pct(cov90),
         "How often a real stint landed inside the model's likely range. Aim for about 90%; much higher means "
         "the range is on the cautious side.", "aim 90%"),
        ("Checks", f"{n_pass}/{n_gate}", "Automatic checks on the data, the fit and the plan (listed under More detail).",
         "all passed" if n_pass == n_gate else f"⚠ {n_gate - n_pass} failed"),
    ])
    badge("Locked before the race", "ok",
          help=f"The prediction was saved to {meta.get('sealed_file', '')} with a fingerprint before any race lap "
               "was read, and race data can't be loaded while fitting. Scoring checks the fingerprint first.")

    ps_ = load_pq(f"scorestint_{key}.parquet")
    if not ps_.empty:
        with card("acc-stints", f"Predicted vs actual tyre wear on {len(ps_)} race stints",
                  tip="Each dot is one race stint. On the diagonal, the prediction was exact.",
                  sub="Average error by tyre: " + " · ".join(f"{k.title()} {v:.3f} s/lap"
                                                             for k, v in sc["mae_by_compound"].items())):
            lim = [min(ps_["pred_rate"].min(), ps_["obs_rate"].min()) - 0.05,
                   max(ps_["pred_rate"].max(), ps_["obs_rate"].max()) + 0.05]
            fig = go.Figure(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=HAIR, dash="dash", width=1),
                                       name="exact", hoverinfo="skip"))
            for cmp_ in in_ladder(ps_["compound"].unique()):
                g = ps_[ps_["compound"] == cmp_]
                fig.add_trace(go.Scatter(
                    x=g["pred_rate"], y=g["obs_rate"], mode="markers", name=cmp_.title(), text=g["driver"],
                    marker=dict(size=10, color=ccol(cmp_), line=dict(width=2, color=BLACK)),
                    hovertemplate="%{text}<br>predicted %{x:.3f} · actual %{y:.3f} s/lap<extra></extra>"))
            chart(style(fig, 380, "actual wear (s/lap)", "predicted wear (s/lap)"))

    with more("More detail: checks and range"):
        badges([(GATE_WORDS.get(g["gate"], g["gate"]), "ok" if g["pass"] else "alert") for g in meta["gates"]])
        cov = pd.DataFrame({"nominal": [float(k) for k in sc["coverage"]],
                            "held": list(sc["coverage"].values())}).sort_values("nominal")
        with card("acc-range", f"The likely range held {pct(cov90)} of the time (aim 90%)",
                  tip="Above the diagonal the range is wider than it needs to be: practice laps are noisier than "
                      "race stints, so the model stays cautious. That is disclosed rather than tuned away."):
            fig = go.Figure(go.Scatter(x=[0.4, 1], y=[0.4, 1], mode="lines", name="perfect",
                                       line=dict(color=HAIR, dash="dash", width=1), hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=cov["nominal"], y=cov["held"], mode="lines+markers", name="degless",
                                     line=dict(color=WHITE, width=2.5),
                                     marker=dict(size=10, color=WHITE, line=dict(color=BLACK, width=2)),
                                     hovertemplate="range set to %{x:.0%}<br>held %{y:.0%}<extra></extra>"))
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickformat=".0%")
            chart(style(fig, 320, "how often it held", "range setting"))
        if sc.get("cliff"):
            rows = [{"Tyre": c.title(), "Predicted drop-off (lap)": fmt(v.get("predicted_lap"), 0),
                     "Seen in the race (lap)": fmt(v.get("observed_lap"), 0) if v.get("detected") else "not reached",
                     "Off by (laps)": fmt(v.get("error_laps"), 1)} for c, v in sc["cliff"].items()]
            st.markdown("**Drop-off timing**")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _practice_data() -> None:
    casc, clean = load_pq(f"cascade_{key}.parquet"), load_pq(f"clean_{key}_practice.parquet")
    if not clean.empty:
        c1, c2 = st.columns([1, 2])
        with c1:
            sess = st.selectbox("Session", sorted(clean["session"].unique()), key="dec_s")
        with c2:
            stage = st.segmented_control("Lap times", ["Raw", "Fuel-adjusted", "Fuel + track adjusted"],
                                         default="Fuel + track adjusted", required=True, key="dec_stage")
        g = clean[clean["session"] == sess]
        col = {"Raw": "lap_time_s", "Fuel-adjusted": "lap_time_fuel_corr", "Fuel + track adjusted": "lap_time_corr"}[stage]
        with card("acc-peel", f"{meta['n_raw_laps']} practice laps → {meta['n_clean_laps']} clean long-run laps",
                  tip="Lap time against tyre age. Fuel-adjusted takes out the car getting lighter; track adjusted "
                      "also takes out the track getting faster. Dotted lines are straight-line fits per tyre."):
            fig = go.Figure()
            for cmp_ in in_ladder(g["compound"].unique()):
                sub = g[g["compound"] == cmp_]
                c = ccol(cmp_)
                fig.add_trace(go.Scatter(
                    x=sub["tyre_age"], y=sub[col], mode="markers", name=cmp_.title(),
                    marker=dict(color=c, size=8, line=dict(width=2, color=BLACK)),
                    hovertemplate=f"{cmp_.title()} · lap %{{x:.0f}} of the stint<br>%{{y:.2f}} s<extra></extra>"))
                if len(sub) > 3:
                    b = np.polyfit(sub["tyre_age"], sub[col], 1)
                    xs = np.linspace(sub["tyre_age"].min(), sub["tyre_age"].max(), 20)
                    fig.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines", showlegend=False,
                                             line=dict(color=c, width=2, dash="dot"),
                                             hovertemplate=f"{cmp_.title()}: {b[0]:+.3f} s a lap<extra></extra>"))
            chart(style(fig, 400, "lap time (s)", "laps on the tyre"))

    with more("More detail: which laps were used"):
        if not casc.empty:
            labels = [CASCADE_WORDS.get(s, r) for s, r in zip(casc["step"], casc["rule"])]
            with card("acc-cascade", f"Lap filtering: {int(casc['laps'].iloc[0])} → {int(casc['laps'].iloc[-1])} laps",
                      tip="Each rule is absolute (never relative to a stint's best lap), so no rule can quietly "
                          "remove late, slow laps and make tyres look better than they are."):
                fig = go.Figure(go.Bar(
                    x=casc["laps"], y=labels, orientation="h", width=0.55, showlegend=False,
                    marker=dict(color=[WHITE] + [DIM] * (len(casc) - 1), line=dict(color=BLACK, width=2)),
                    text=[f"  {n}  (−{d})" if d else f"  {n}" for n, d in zip(casc["laps"], casc["dropped"])],
                    textposition="outside", textfont=dict(color=WHITE),
                    hovertemplate="%{y}<br>%{x} laps left<extra></extra>"))
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(range=[0, casc["laps"].max() * 1.28])
                chart(style(fig, 320, "", "laps left", legend=False))
        comp = pd.DataFrame(meta["compound_counts"])
        if not comp.empty:
            comp["_o"] = comp["compound"].map({c: i for i, c in enumerate(LADDER)})
            comp = comp.sort_values("_o")
            sup, caps = meta.get("age_support_by_compound", {}), meta.get("max_stint_laps", {})
            st.markdown("**Clean laps by tyre**")
            st.dataframe(pd.DataFrame({
                "Tyre": comp["compound"].str.title(), "Clean laps": comp["laps"], "Runs": comp["stints"],
                "Oldest tyre run (laps)": [sup.get(c, np.nan) for c in comp["compound"]],
                "Longest stint allowed (laps)": [caps.get(c, np.nan) for c in comp["compound"]],
            }), width="stretch", hide_index=True)
            thin = comp.loc[comp["laps"].idxmin()]
            notice(f"<b>{str(thin['compound']).title()} has the least data: {int(thin['laps'])} laps from "
                   f"{int(thin['stints'])} run(s).</b> That's why its likely range is widest and its longest "
                   "allowed stint the tightest.", "info")


def _tab_accuracy() -> None:
    if OUTLOOK_ONLY:
        notice("No practice data yet. Accuracy and the practice data appear once there is a model to check.")
        return
    if WEEKEND_ONLY:
        notice("The race hasn't been run yet. The prediction is locked now and gets checked against the race "
               "after the flag.")
    else:
        _accuracy_scored()
    _practice_data()
    how(
        "Before the race, the practice-only prediction is saved with a fingerprint; after the race it's read back, "
        "the fingerprint checked, and the prediction compared with every clean race stint.",
        "Each stint is compared on its shape — how much slower lap 15 is than lap 3 — because practice can't know "
        "race fuel loads or engine modes.",
        "The race-vs-practice wear factor comes from other weekends, never this one, so this is a genuine "
        "out-of-sample test.",
    )


# ==========================================================================
# REPLAY
# ==========================================================================


def _tab_replay() -> None:
    if WEEKEND_ONLY:
        notice("Replay appears after the race. During the race, the Now tab is live.")
        return
    rp = load_pq(f"replay_{key}.parquet")
    if rp.empty:
        notice("No replay for this weekend.")
        return
    drivers = sorted(rp["driver"].unique())
    c1, c2 = st.columns([1, 3])
    with c1:
        drv = st.selectbox("Driver", drivers, key="rp_drv")
    d = rp[rp["driver"] == drv].sort_values("lap_number")
    with c2:
        lo_l, hi_l = int(d["lap_number"].min()), int(d["lap_number"].max())
        # Key on the driver: each driver ran a different number of laps, so a
        # slider value carried over can fall outside the new range.
        lap = st.slider("Lap", lo_l, max(hi_l, lo_l + 1), lo_l, key=f"rp_lap_{drv}")
    upto = d[d["lap_number"] <= lap]
    now = upto.iloc[-1] if len(upto) else None
    alarm = bool(now["cliff_alarm"]) if now is not None else False
    if now is not None:
        tiles("replay", [
            ("Tyre", f"{str(now.get('compound', '—')).title()} · {now['tyre_age']:.0f} laps"),
            ("Laps seen", f"{int(now['laps_seen'])}", "Clean laps on this set so far; the estimate firms up as they come in."),
            ("Drop-off risk", pct(now["p_past_cliff"]), DEFS["drop_off"], "⚠ alarm" if alarm else "✓ fine"),
        ])
        if alarm:
            badge(f"Drop-off alarm on lap {int(now['lap_number'])}: {pct(now['p_past_cliff'])} chance this tyre is "
                  f"past its drop-off · wearing {now['deg_now_s']:.2f} s a lap", "alert")

    fig = go.Figure()
    for cmp_ in in_ladder(d["compound"].dropna().unique()):
        g = d[d["compound"] == cmp_]
        fig.add_trace(go.Scatter(
            x=g["lap_number"], y=g["lap_time_s"], mode="markers", name=str(cmp_).title(),
            marker=dict(size=7, color=rgba(ccol(cmp_), 0.6), line=dict(width=1, color=BLACK)),
            hovertemplate=f"{str(cmp_).title()} · lap %{{x}}<br>%{{y:.2f}} s<extra></extra>"))
    band = upto.dropna(subset=["band_lo", "band_hi"])
    if len(band):
        fig.add_trace(go.Scatter(
            x=list(band["lap_number"]) + list(band["lap_number"])[::-1],
            y=list(band["band_hi"]) + list(band["band_lo"])[::-1], fill="toself", fillcolor=rgba(WHITE, 0.14),
            line=dict(width=0), name="likely range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=upto["lap_number"], y=upto["stint_pace_est"], mode="lines",
                             name="model's pace estimate", line=dict(color=WHITE, width=2.5)))
    for _, r in d[d["is_pit_in"]].iterrows():
        fig.add_vline(x=r["lap_number"], line=dict(color=HAIR, width=1))
    fig.add_vline(x=lap, line=dict(color=WHITE, width=2, dash="dash"))
    # In-laps, out-laps and safety-car laps run 20-30 s long; left in the
    # y-range they compress every racing lap into a few pixels.
    lt = d["lap_time_s"].dropna()
    if len(lt) > 5:
        lo_y, hi_y = float(lt.quantile(0.01)) - 1.0, float(lt.quantile(0.93)) + 2.5
        span = [c for c in [upto["band_lo"].min(), upto["band_hi"].max()] if np.isfinite(c)]
        if span:
            lo_y, hi_y = min(lo_y, min(span) - 0.5), max(hi_y, max(span) + 0.5)
        fig.update_yaxes(range=[lo_y, hi_y])
    with card("replay", f"{drv}, lap {lap}: " + ("drop-off alarm" if alarm else "tyre still working"),
              tip="Dots are real lap times; the white line is the model's pace estimate for the current tyres and "
                  "the band its likely range, which narrows as laps come in. Thin verticals are pit stops; in-laps "
                  "and safety-car laps sit off the top."):
        chart(style(fig, 420, "lap time (s)", "race lap"))


# ==========================================================================
# ENGINEER
# ==========================================================================


@st.cache_data(show_spinner=False)
def _brief(k: str):
    r = engineer_brief(k)
    return r.text, r.source


def _tab_engineer() -> None:
    st.markdown("Ask about this weekend. Answers only use numbers the model produced — the engineer never "
                "estimates its own.")
    cred = credential_source()
    if cred:
        badge(f"Connected · {cred} · {model_name()}", "ok")
    else:
        badge("Offline briefing · add GEMINI_API_KEY to .env for live answers", "info")
    c1, c2, c3 = st.columns([1, 4, 1], vertical_alignment="bottom")
    with c1:
        do_brief = st.button("Brief me", type="primary", key="eng_brief", width="stretch")
    with c2:
        q = st.text_input("Ask a question", placeholder="e.g. should we have committed to a one-stop?",
                          key="eng_q", label_visibility="collapsed")
    with c3:
        do_ask = st.button("Ask", key="eng_ask", width="stretch")
    if do_brief:
        with st.spinner("Briefing…"):
            st.session_state["eng_out"] = _brief(key)
    if do_ask and q.strip():
        with st.spinner("Thinking…"):
            r = engineer_ask(key, q.strip())
        st.session_state["eng_out"] = (r.text, r.source)
    if "eng_out" in st.session_state:
        txt, src = st.session_state["eng_out"]
        with card("engineer", "Race engineer"):
            st.markdown(txt)
        if src == "offline":
            notice("Offline mode: a fixed briefing built from the same facts. Everything else works the same.")


# ==========================================================================
# Tabs
# ==========================================================================

TAB_NAMES = ["Now", "Race plan", "Plan builder", "Tyres", "Accuracy", "Replay", "Engineer"]
tab = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))

with tab["Now"]:
    render_now(WEEKEND, meta, ev, outlook=OUTLOOK, timeline=load_timeline(key))
with tab["Race plan"]:
    _tab_race_plan()
with tab["Plan builder"]:
    render_desk(key, ev, OUTLOOK)
with tab["Tyres"]:
    if OUTLOOK_ONLY:
        _outlook_tyre_fallback()
    else:
        _tab_tyres()
with tab["Accuracy"]:
    _tab_accuracy()
with tab["Replay"]:
    _tab_replay()
with tab["Engineer"]:
    _tab_engineer()
