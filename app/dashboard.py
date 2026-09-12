"""degless — tyre degradation intelligence.

The app fits nothing.  Everything here is read from parquet/JSON written by
`scripts/10_pipeline.py`, so cold start is fast and nothing can break on stage.

**Layout principle.**  The first thing on screen is the decision — the plan, the
pit windows, and what it is worth — because that is what anyone opening this is
here for.  The evidence for it sits behind it in the order a sceptic would ask
for it: the tyre model, then the raw data it came from, then the receipts.
Every long explanation lives in a collapsed "why" block, so the page reads as
charts and numbers and expands into prose only where a reader asks it to.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, EVENTS, SEALED_DIR, current_event  # noqa: E402
from app.desk_tab import render_desk  # noqa: E402
from app.live_tab import render_now, render_strip, weekend_status  # noqa: E402
from src.outlook import load_outlook, load_timeline  # noqa: E402
from src.engineer import ask as engineer_ask  # noqa: E402
from src.engineer import brief as engineer_brief  # noqa: E402
from src.engineer import credential_source, model_name  # noqa: E402

st.set_page_config(page_title="degless", layout="wide", page_icon="🏁")

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
#
# Compound colours keep the identities every reader already knows — soft is
# red, medium is yellow, hard is the cool one — but stepped so they survive the
# checks the raw tyre colours fail.  Pirelli's own yellow (#F0E442) sits at
# 1.29:1 against a light page and its white is invisible outright.
#
# Both sets validated with the dataviz palette checker: light passes the
# lightness band, the chroma floor, adjacent-pair CVD separation (worst 21.4
# deutan) and the normal-vision floor; dark likewise (worst 9.3 deutan).  The
# light amber carries a sub-3:1 contrast warning against the page, so every
# compound mark in this app is *also* directly labelled and backed by a table —
# colour is never the only channel carrying compound identity.

THEMES = {
    "light": {
        "compound": {"SOFT": "#c2334a", "MEDIUM": "#eda100", "HARD": "#2a78d6"},
        "accent": "#2a78d6", "good": "#1a7f4b", "warn": "#b3541e", "bad": "#b3261e",
        "ink": "#111111", "muted": "#5c5c5c",
        "grid": "rgba(120,120,120,0.16)", "surface": "#ffffff",
        "panel": "rgba(120,120,120,0.06)", "hairline": "rgba(120,120,120,0.28)",
    },
    "dark": {
        "compound": {"SOFT": "#e8607a", "MEDIUM": "#c98500", "HARD": "#3987e5"},
        "accent": "#3987e5", "good": "#3fbf7f", "warn": "#e08a3c", "bad": "#e5645c",
        "ink": "#f2f2f2", "muted": "#a8a8a8",
        "grid": "rgba(150,150,150,0.18)", "surface": "#0e1117",
        "panel": "rgba(180,180,180,0.07)", "hairline": "rgba(180,180,180,0.28)",
    },
}


def _theme_name() -> str:
    try:
        t = getattr(st.context, "theme", None)
        if t is not None and getattr(t, "type", None) in THEMES:
            return t.type
    except Exception:
        pass
    return "light"


T = THEMES[_theme_name()]
COMPOUND_COLORS = T["compound"]
GREY = T["muted"]
GRID = T["grid"]
# Softest first — the order the ladder is built in, and the order every
# compound legend, axis and table in this app uses.
LADDER = ["SOFT", "MEDIUM", "HARD"]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def ccol(c: str) -> str:
    return COMPOUND_COLORS.get(c, GREY)


def in_ladder(compounds) -> list:
    """Compounds sorted softest-first, unknowns appended."""
    known = [c for c in LADDER if c in set(compounds)]
    return known + [c for c in compounds if c not in LADDER]


# --------------------------------------------------------------------------
# Cached loaders (pure disk reads)
# --------------------------------------------------------------------------
#
# Every loader is keyed on the file's mtime, the way the Strategy desk already
# keys its posterior.  The supervisor refits the weekend model after each
# practice session and rewrites these files underneath an open dashboard; a
# cache keyed on the filename alone would pin every tab to whatever was on disk
# when the page first rendered — the Evidence tab showing FP1/FP2 long after
# FP3 had been fitted in.


def _stamp(p: Path) -> float | None:
    """The file's mtime, or None when it is not there."""
    try:
        return p.stat().st_mtime
    except OSError:
        return None


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False)
def _read_pq(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_meta(key: str) -> dict | None:
    p = DATA_PROCESSED / f"meta_{key}.json"
    m = _stamp(p)
    return _read_json(str(p), m) if m is not None else None


def load_pq(name: str) -> pd.DataFrame:
    p = DATA_PROCESSED / name
    m = _stamp(p)
    return _read_pq(str(p), m) if m is not None else pd.DataFrame()


def load_sealed_json(fname: str) -> dict:
    p = SEALED_DIR / fname
    m = _stamp(p)
    return _read_json(str(p), m) if m is not None else {}


def load_weekend(key: str) -> dict | None:
    p = DATA_PROCESSED / f"weekend_{key}.json"
    m = _stamp(p)
    return _read_json(str(p), m) if m is not None else None


def _built_at(name: str) -> str:
    """Local time a processed file was last written — when its evidence was built."""
    m = _stamp(DATA_PROCESSED / name)
    return datetime.fromtimestamp(m).strftime("%H:%M") if m is not None else "—"


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
# Chart chrome
# --------------------------------------------------------------------------


def style(fig, height: int = 430, ytitle: str = "", xtitle: str = "",
          legend: bool = True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=34, b=8),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["ink"], size=13),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
        xaxis_title=xtitle, yaxis_title=ytitle,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=T["hairline"],
                     title_font=dict(color=T["muted"]),
                     tickfont=dict(color=T["muted"]))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=T["hairline"],
                     title_font=dict(color=T["muted"]),
                     tickfont=dict(color=T["muted"]))
    return fig


def why(title: str, body: str) -> None:
    """A collapsed explanation.  The page stays visual; the argument stays available."""
    with st.expander(title):
        st.markdown(body)


def callout(text: str, kind: str = "accent") -> None:
    col = T.get(kind, T["accent"])
    st.markdown(
        f"<div style='border-left:3px solid {col};padding:10px 14px;"
        f"background:{rgba(col, 0.08)};border-radius:4px;font-size:0.94rem;"
        f"margin:2px 0 10px 0'>{text}</div>",
        unsafe_allow_html=True,
    )


def chip(label: str, value: str, color: str | None = None, sub: str = "") -> str:
    col = color or T["accent"]
    return (
        f"<div style='display:inline-block;min-width:118px;padding:9px 14px;"
        f"margin:0 8px 8px 0;border-radius:8px;background:{T['panel']};"
        f"border-left:4px solid {col}'>"
        f"<div style='font-size:0.72rem;letter-spacing:.05em;color:{T['muted']};"
        f"text-transform:uppercase'>{label}</div>"
        f"<div style='font-size:1.32rem;font-weight:650;color:{T['ink']};"
        f"line-height:1.25'>{value}</div>"
        + (f"<div style='font-size:0.74rem;color:{T['muted']}'>{sub}</div>"
           if sub else "")
        + "</div>"
    )


def compound_pill(c: str, text: str = "") -> str:
    col = ccol(c)
    return (f"<span style='display:inline-block;padding:2px 9px;border-radius:11px;"
            f"background:{rgba(col, 0.16)};border:1px solid {col};color:{T['ink']};"
            f"font-size:0.8rem;font-weight:600;margin-right:5px'>"
            f"{text or c}</span>")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

events = available_events()
if not events:
    st.title("degless")
    st.error("No processed data found. Run `make pipeline` first.")
    st.stop()

st.sidebar.title("degless")
st.sidebar.caption("tyre degradation intelligence · 2026 regulations")
WEEKEND = weekend_status()
_cur = WEEKEND.get("event_key") or current_event().key
key = st.sidebar.selectbox(
    "Race weekend", events, index=(events.index(_cur) if _cur in events else len(events) - 1),
    format_func=lambda k: f"{EVENTS[k].name} · {_event_kind(k)}",
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

st.sidebar.markdown("---")
st.sidebar.markdown("**Model health**")
gate_col = T["good"] if n_pass == n_gate else T["warn"]
if OUTLOOK_ONLY:
    _o = OUTLOOK or {}
    st.sidebar.markdown(
        chip("Knowledge", "prior only", T["warn"], (_o.get("prior_basis") or "no practice yet")[:40])
        + chip("Outlook", (_o.get("strategy") or {}).get("best", "—"), T["accent"],
               "updated " + (_o.get("updated_utc") or "")[:16].replace("T", " ") + " UTC"),
        unsafe_allow_html=True)
    st.sidebar.caption(f"No practice fit yet: pit loss {meta['pit_loss_s']:.1f} s ({meta.get('pit_loss_source', '')[:36]}). "
                       "The sealed model appears after the first practice session.")
else:
  st.sidebar.markdown(
      chip("Gates", f"{n_pass}/{n_gate}", gate_col)
      + (chip("Stint MAE", f"{sc['mae']:.3f}",
              T["good"] if sc["passes_mae"] else T["bad"], "s/lap · target <0.15")
         + chip("Bias", f"{sc['bias']:+.3f}",
                T["good"] if abs(sc["bias"]) < 0.06 else T["warn"], "s/lap")
         if not WEEKEND_ONLY else
         chip("Race", "not yet run", T["warn"], f"fitted on {', '.join(meta.get('sessions_used', []))}"))
      + chip("Max r̂", f"{meta['bayes']['max_rhat']:.4f}",
             T["good"] if meta["bayes"]["max_rhat"] < 1.01 else T["warn"],
             f"{meta['bayes']['n_divergences']} divergences"),
    unsafe_allow_html=True,
  )
  st.sidebar.markdown("---")
  st.sidebar.caption(
    f"{meta['n_clean_laps']} clean practice laps · pit loss "
    f"{meta['pit_loss_s']:.1f} s\n\nSealed `{meta['sealed_file']}`\n\n"
    f"sha256 `{meta['sealed_sha256'][:20]}…`"
  )

st.title(f"degless — {ev.name}")
render_strip(WEEKEND, T, chip, callout)
if OUTLOOK_ONLY:
    callout("<b>No practice data yet.</b> Everything on this page is the outlook: the compound ladder, the "
            "circuit's (or the season's) race history, and the transferred priors, searched exactly as the sealed "
            "model will be. It refreshes by itself and the sealed fit replaces it after the first practice session.",
            "accent")
elif WEEKEND_ONLY:
    callout(f"<b>Pre-race model.</b> Fitted on {', '.join(meta.get('sessions_used', []))} and sealed "
            f"<code>{meta.get('sealed_file', '')}</code>; race data has not been read. The Live tab drives "
            "this model against the feed during the session.", "accent")

TAB_NAMES = ["Now", "Strategy desk", "Race plan", "Tyre model", "Evidence", "Validation", "Replay", "Engineer"]
tab = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))

# ==========================================================================
# 0. NOW — the pit wall, and between sessions the outlook
# ==========================================================================
with tab["Now"]:
    render_now(WEEKEND, T, ccol, chip, callout, style, rgba, compound_pill, meta, ev,
               outlook=OUTLOOK, timeline=load_timeline(key))


# ==========================================================================
# 0b. STRATEGY DESK — build, compare, stress-test, commit
# ==========================================================================
with tab["Strategy desk"]:
    render_desk(key, ev, OUTLOOK, T, ccol, chip, callout, style, rgba, compound_pill)


# ==========================================================================
# 1. RACE PLAN — the decision
# ==========================================================================
with tab["Race plan"]:
    def _tbl(kind: str) -> pd.DataFrame:
        # An outlook-only weekend has the same tables, written by the outlook.
        return load_pq(f"outlook_{key}_{kind}.parquet" if OUTLOOK_ONLY else f"{kind}_{key}.parquet")

    if OUTLOOK_ONLY:
        callout("This is the <b>outlook's</b> plan — priors and the strategy search, no practice fit. "
                "It is the same search the sealed model runs; the numbers tighten as practice arrives.", "warn")
    plan = _tbl("plan")
    bystops = _tbl("bystops")
    pw = _tbl("pitwindow")
    field = _tbl("fieldplan")
    life = _tbl("life")
    uc = _tbl("undercut")
    cf = _tbl("counterfactual")

    comps = list(best_plan.get("compounds", []))
    lens = list(best_plan.get("stint_lens", []))
    pits = list(best_plan.get("pit_laps", []))
    windows = strat_meta.get("pit_windows", [])
    p_best = float(bystops.iloc[0]["win_prob_any"]) if not bystops.empty else float("nan")

    # -- the recommendation ------------------------------------------------
    st.markdown(
        "<div style='font-size:0.78rem;letter-spacing:.09em;color:%s;"
        "text-transform:uppercase;margin-bottom:2px'>Recommended race plan</div>"
        % T["muted"], unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:2.0rem;font-weight:700;line-height:1.15;margin-bottom:8px'>"
        + f"{best_plan.get('n_stops', '—')}-stop &nbsp;"
        + "&nbsp;".join(compound_pill(c) for c in comps)
        + "</div>", unsafe_allow_html=True)

    win_txt = ", ".join(
        f"lap {w['recommended']} (window {w['lo']}–{w['hi']})" for w in windows
    ) or ", ".join(f"lap {p}" for p in pits)
    st.markdown(
        chip("Stops", str(best_plan.get("n_stops", "—")), T["accent"],
             f"{p_best:.0%} of posterior draws")
        + chip("Stint lengths", " · ".join(str(x) for x in lens), T["accent"],
               f"{ev.n_race_laps} race laps")
        + chip("Pit on", win_txt, T["accent"], "±1.0 s window")
        + chip("Pit loss", f"{meta['pit_loss_s']:.1f} s", T["muted"],
               f"measured, {meta['pit_stops_measured']} green-flag stops")
        + chip("Plans searched", f"{strat_meta['n_strategies']:,}", T["muted"],
               f"{strat_meta['n_scored']:,} scored on {strat_meta['n_draws']} draws"),
        unsafe_allow_html=True,
    )

    # -- stint timeline ----------------------------------------------------
    st.markdown("#### The plan on the race clock")
    if not plan.empty:
        fig = go.Figure()
        rows = [("degless plan", plan)]
        # A few real strategies underneath, so the plan is read against what
        # the field actually did rather than in a vacuum.
        field_rows = []
        if not field.empty:
            fin = field[field["finished_lap"] >= ev.n_race_laps - 2]
            for drv in sorted(fin["driver"].unique())[:6]:
                field_rows.append((drv, fin[fin["driver"] == drv]))
        lanes = rows + field_rows
        for i, (label, df) in enumerate(lanes):
            y = len(lanes) - 1 - i
            for _, r in df.iterrows():
                L = int(r["laps"])
                col = ccol(r["compound"])
                fig.add_trace(go.Bar(
                    x=[L], y=[y], base=[int(r["start_lap"]) - 1], orientation="h",
                    marker=dict(color=rgba(col, 0.85 if i == 0 else 0.55),
                                line=dict(color=T["surface"], width=2)),
                    width=0.62 if i == 0 else 0.46,
                    # A label only where it fits upright; plotly rotates text
                    # that overflows its bar, which is unreadable.
                    text=(f"{r['compound'][0]} {L}" if L >= 9
                          else (r["compound"][0] if L >= 4 else "")),
                    textposition="inside", insidetextanchor="middle",
                    textangle=0, cliponaxis=False, constraintext="none",
                    textfont=dict(color="#ffffff" if i == 0 else T["ink"], size=12),
                    showlegend=False,
                    hovertemplate=(f"<b>{label}</b><br>{r['compound']}<br>"
                                   f"laps {int(r['start_lap'])}–{int(r['end_lap'])}"
                                   f" ({L} laps)<extra></extra>")))
        for w in windows:
            fig.add_vrect(x0=w["lo"] - 1, x1=w["hi"], line_width=0,
                          fillcolor=rgba(T["accent"], 0.10), layer="below")
        for p in pits:
            fig.add_vline(x=p - 1, line=dict(color=T["accent"], width=2, dash="dot"))
        fig.update_yaxes(tickmode="array",
                         tickvals=list(range(len(lanes)))[::-1],
                         ticktext=[lab for lab, _ in lanes],
                         showgrid=False)
        fig.update_xaxes(range=[0, ev.n_race_laps], dtick=10)
        # Overlay, not stack: each bar carries its own `base` (the stint's start
        # lap), and stacking would ignore it and pile the stints on top of one
        # another at the origin.
        fig.update_layout(barmode="overlay", bargap=0.25)
        st.plotly_chart(style(fig, 90 + 42 * len(lanes), "", "race lap",
                              legend=False), width="stretch")
        # Colour is never the only channel: the same plan as text.
        st.caption(
            "Shaded bands are the ±1.0 s pit windows; dotted lines are the "
            "recommended stops. Plan in words: "
            + " → ".join(f"{r['compound']} laps {int(r['start_lap'])}–"
                         f"{int(r['end_lap'])} ({int(r['laps'])})"
                         for _, r in plan.iterrows())
            + (f". Field lanes are the {len(field_rows)} classified finishers shown "
               "for comparison." if field_rows else "")
        )

    st.markdown("---")
    c1, c2 = st.columns([1.25, 1])

    # -- pit window --------------------------------------------------------
    with c1:
        st.markdown("#### How much a lap early or late costs")
        if not pw.empty:
            fig = go.Figure()
            for k_stop, g in pw.groupby("stop"):
                g = g.sort_values("lap")
                col = T["accent"] if k_stop == 1 else T["warn"]
                fig.add_trace(go.Scatter(
                    x=g["lap"], y=g["loss_s"], mode="lines",
                    name=f"stop {int(k_stop)}",
                    line=dict(color=col, width=2),
                    hovertemplate="stop %s · lap %%{x}<br>%%{y:.2f} s lost"
                                  "<extra></extra>" % int(k_stop)))
                win = g[g["in_window"]]
                if not win.empty:
                    fig.add_vrect(x0=win["lap"].min(), x1=win["lap"].max(),
                                  line_width=0, fillcolor=rgba(col, 0.10),
                                  layer="below")
                rec = g[g["is_recommended"]]
                if not rec.empty:
                    fig.add_trace(go.Scatter(
                        x=rec["lap"], y=rec["loss_s"], mode="markers",
                        marker=dict(color=col, size=11,
                                    line=dict(color=T["surface"], width=2)),
                        showlegend=False, hoverinfo="skip"))
            fig.add_hline(y=1.0, line=dict(color=GREY, dash="dash", width=1))
            fig.update_yaxes(range=[0, 12])
            st.plotly_chart(style(fig, 360, "time lost vs the best lap (s)",
                                  "pit lap"), width="stretch")
            note = (
                "The flat floor is the decision's slack. Anywhere inside a shaded "
                "band costs under a second, so a lap lost to traffic is not a lap "
                "lost to the race — the dashed line is that 1 s threshold."
            )
            tight = [int(k) for k, g in pw.groupby("stop") if g["lap"].nunique() <= 3]
            if tight:
                note += (
                    " " + ", ".join(f"**Stop {k}**" for k in tight)
                    + (" has" if len(tight) == 1 else " have")
                    + " almost no legal freedom: every other lap would push a "
                    "stint past its compound's stint cap — the lower of what "
                    "the tyre can take and what the practice data supports. "
                    "That is a constraint on the plan, not a confident "
                    "recommendation."
                )
            st.caption(note)
        else:
            st.info("No pit-window sweep available.")

    # -- stop count --------------------------------------------------------
    with c2:
        st.markdown("#### One stop, two, or three?")
        if not bystops.empty:
            b = bystops.sort_values("n_stops").reset_index(drop=True)
            # A lollipop, not bars: the best plan sits at zero, and a bar of
            # zero length is an invisible mark carrying the most important row.
            fig = go.Figure()
            ys = [f"{int(n)}-stop" for n in b["n_stops"]]
            for yv, d in zip(ys, b["delta_s"]):
                fig.add_shape(type="line", x0=0, x1=d, y0=yv, y1=yv,
                              line=dict(color=T["hairline"], width=2))
            fig.add_trace(go.Scatter(
                x=b["delta_s"], y=ys, mode="markers+text",
                marker=dict(size=15,
                            color=[T["good"] if d == b["delta_s"].min()
                                   else T["accent"] for d in b["delta_s"]],
                            line=dict(color=T["surface"], width=2)),
                text=[f"   best · wins {w:.0%}" if d <= 0
                      else f"   +{d:.1f} s · wins {w:.0%}"
                      for d, w in zip(b["delta_s"], b["win_prob_any"])],
                textposition="middle right", textfont=dict(color=T["ink"], size=12),
                hovertemplate="%{y}<br>%{x:.1f} s slower in expectation<extra></extra>",
                showlegend=False))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(range=[-0.6, max(b["delta_s"].max() * 2.1, 6)])
            st.plotly_chart(style(fig, 360, "", "expected loss vs the best plan (s)",
                                  legend=False), width="stretch")
            missing = sorted({1, 2, 3} - set(int(n) for n in b["n_stops"]))
            if missing:
                st.caption(
                    ", ".join(f"**{m}-stop**" for m in missing)
                    + (" has no legal plan here: no combination of stint lengths "
                       "fits the race distance inside every compound's own stint "
                       "cap — the lower of its grip budget and its practice "
                       "support (see *How long each tyre lasts*). It is absent "
                       "because it was ruled out, not because it lost.")
                )
            alt = b[b["delta_s"] > 0]
            if not alt.empty:
                a = alt.loc[alt["delta_s"].idxmin()]
                st.caption(
                    f"Closest alternative: **{a['strategy']}**, {a['delta_s']:.1f} s "
                    f"slower in expectation but fastest on {a['win_prob_any']:.0%} of "
                    f"posterior draws. That second number is the one a point-estimate "
                    f"optimiser cannot produce."
                )

    strat_tbl = _tbl("strategy")
    if not strat_tbl.empty:
        with st.expander("The full ranking"):
            st.dataframe(
                strat_tbl.head(30)[[c for c in
                                    ["strategy", "n_stops", "compounds",
                                     "stint_lens", "push", "max_wear", "delta_s",
                                     "delta_p05", "delta_p95", "win_prob",
                                     "p_beat_best"]
                                    if c in strat_tbl.columns]].round(3),
                width="stretch", hide_index=True)
            st.caption(
                "`push` is the driving level the plan assumes — 1.0 is a "
                "practice long run, lower means managing the tyre and paying "
                "lap time for it. `max_wear` is the deepest any stint goes into "
                "its grip budget; 1.0 is the cliff. `delta_s` is expected loss "
                "against the top plan and `delta_p05`–`delta_p95` its spread "
                "**draw by draw** — every plan is scored on the same posterior "
                "samples, so the shared uncertainty about how fast this tyre "
                "degrades at all cancels in the comparison. That paired band is "
                "the decision-relevant one; each plan's own ±20 s absolute "
                "spread is mostly common-mode and would badly overstate how "
                "uncertain the *choice* is. Near-ties at the top are the honest "
                "answer — the decision is a window, not a lap."
            )

    why("Why this plan, and what it does not know",
        f"""
**The decision is a pair: how long, and how hard.** A driver can always buy
tyre life with lap time — lift and coast, short-shift, roll speed through the
corner. So a plan here is a sequence of stints *and* a push level, and both are
optimised. This plan assumes push
**{strat_meta.get('push', float('nan')):.2f}**, which implies a practice→race
degradation factor of **{strat_meta.get('implied_regime', float('nan')):.2f}**.
That number is *predicted*, not assumed: measured independently from other
weekends' races it comes out at {meta['regime']['ratio']:.2f}. An earlier
version of this model was handed 0.40 as a constant, and being handed it was
the problem — degradation came out roughly a third of the truth and the
optimiser answered with stints far longer than any team runs.

**What is priced.** Degradation, as a grip budget of
{strat_meta.get('grip_budget_s', 3.8):.1f} s that a tyre spends before its
cliff — so tyre life is *derived* from degradation rate rather than fitted
beside it, and a compound cannot come out both fast-degrading and long-lived.
The compound pace ladder. Measured pit loss ({meta['pit_loss_s']:.1f} s). A
{strat_meta.get('warmup_s', 0.7):.1f} s cold-tyre penalty on each stint's first
flying lap. The fuel-load effect on wear, which is why stint lengths are not
equal. Dirty air after a stop (~{strat_meta.get('traffic_s_per_stop', 0.0):.1f} s
per stop, scaled by how bunched the field is at the rejoin lap). And the safety
car as a *credit* of {strat_meta.get('safety_car_credit_s', 0.0):.1f} s — a stop
not yet taken is an option worth the chance a safety car arrives while it is
still live, which is why holding a stop in reserve has value.

**What is not priced — and this matters.** Track position as a race-long state,
the starting-tyre choice, and the undercut battle with the specific cars around
you. The compound *ordering* is the weakest part of the answer: the best plan
for each ordering of the same compounds spans only
{strat_meta.get('ordering_spread_s', float('nan')):.1f} s, well inside the
posterior's own width, and what decides a starting compound in reality is
position off the line, which is absent from this objective. The **number of
stops** and the **stint lengths** are what this is claiming.

**Read the seconds as a scale, not a verdict.** {strat_meta['n_strategies']:,}
legal plans were costed on the posterior-mean curve at each push level and the
best {strat_meta['n_scored']:,} re-costed on all {strat_meta['n_draws']} draws,
so a "win probability" is the share of draws on which a plan is fastest *among
that shortlist*.
""")

    # -- tyre life ---------------------------------------------------------
    st.markdown("---")
    c1, c2 = st.columns([1, 1.15])
    with c1:
        st.markdown("#### How long each tyre lasts")
        if not life.empty:
            lf = life.set_index("compound").reindex(in_ladder(life["compound"]))
            lf = lf.reset_index()
            fig = go.Figure()
            for i, r in lf.iterrows():
                y = len(lf) - 1 - i
                col = ccol(r["compound"])
                fig.add_trace(go.Bar(
                    x=[r["life_laps"]], y=[y], orientation="h", width=0.5,
                    marker=dict(color=rgba(col, 0.75),
                                line=dict(color=T["surface"], width=2)),
                    showlegend=False,
                    hovertemplate=(f"{r['compound']}<br>useful life "
                                   f"{r['life_laps']:.0f} laps at race push "
                                   f"[{r['life_lo']:.0f}–{r['life_hi']:.0f}]"
                                   "<extra></extra>")))
                fig.add_trace(go.Scatter(
                    x=[r["knee_lap"]], y=[y], mode="markers",
                    marker=dict(color=T["ink"], size=16, symbol="line-ns",
                                line=dict(color=T["ink"], width=2.5)),
                    showlegend=False,
                    hovertemplate=f"{r['compound']} cliff at full push, lap "
                                  f"{r['knee_lap']:.0f}<extra></extra>"))
                fig.add_annotation(
                    x=r["life_laps"], y=y, xanchor="left", xshift=8,
                    text=f"managed {r['life_laps']:.0f} · full push {r['knee_lap']:.0f}",
                    showarrow=False, font=dict(color=T["muted"], size=11))
            fig.update_yaxes(tickmode="array", tickvals=list(range(len(lf)))[::-1],
                             ticktext=list(lf["compound"]), showgrid=False)
            fig.update_xaxes(range=[0, float(lf["life_laps"].max()) * 1.75])
            st.plotly_chart(style(fig, 300, "", "laps", legend=False),
                            width="stretch")
            st.caption(
                "Bar is how far the tyre goes when the driver manages it; the "
                "tick is where it reaches its cliff at full attack. Both are "
                "the grip budget divided by the degradation rate — life is "
                "derived from rate here, not fitted next to it, which is what "
                "stops a compound coming out both fast-degrading and durable.")
    with c2:
        st.markdown("#### Undercut window")
        if not uc.empty:
            # The curve is only meaningful out to where the compounds have
            # support; past that it is the hinge extrapolating, not a tyre.
            caps = meta.get("max_stint_laps", {})
            cap = min([v for k, v in caps.items() if k in ("MEDIUM", "SOFT")]
                      or [float(uc["leader_tyre_age"].max())])
            uc = uc[uc["leader_tyre_age"] <= cap]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(uc["leader_tyre_age"]) + list(uc["leader_tyre_age"])[::-1],
                y=list(uc["hi"]) + list(uc["lo"])[::-1], fill="toself",
                fillcolor=rgba(T["accent"], 0.16), line=dict(width=0),
                name="90% credible", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=uc["leader_tyre_age"], y=uc["gain_s"], mode="lines",
                name="expected gain on the out-lap",
                line=dict(color=T["accent"], width=2.5),
                hovertemplate="leader on a %{x:.0f}-lap tyre<br>"
                              "%{y:+.2f} s/lap<extra></extra>"))
            fig.add_hline(y=0, line=dict(color=GREY, width=1))
            opens = uc[uc["p_positive"] > 0.5]
            if not opens.empty:
                x0 = float(opens["leader_tyre_age"].iloc[0])
                fig.add_vrect(x0=x0, x1=float(uc["leader_tyre_age"].max()),
                              line_width=0, fillcolor=rgba(T["good"], 0.10),
                              layer="below",
                              annotation_text="window open",
                              annotation_position="top left",
                              annotation_font=dict(color=T["muted"], size=11))
            st.plotly_chart(style(fig, 300, "gain per lap (s)",
                                  "leader's tyre age (laps)"), width="stretch")
            if not opens.empty:
                st.caption(
                    f"MEDIUM ahead, SOFT out of the pits. The undercut turns "
                    f"positive once the car ahead is on a "
                    f"**{opens['leader_tyre_age'].iloc[0]:.0f}-lap-old** tyre."
                )

    # -- counterfactual ----------------------------------------------------
    if not cf.empty:
        st.markdown("---")
        st.markdown("#### The moment — actual stop lap vs model-optimal")
        top = cf.head(10).copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top["loss_s"], y=top["driver"], orientation="h", width=0.55,
            marker=dict(color=rgba(T["warn"], 0.8)),
            error_x=dict(type="data", symmetric=False,
                         array=top["hi"] - top["loss_s"],
                         arrayminus=top["loss_s"] - top["lo"],
                         color=T["muted"], thickness=1.2, width=4),
            customdata=np.stack([top["compounds"], top["actual_pit_laps"],
                                 top["model_pit_laps"]], axis=-1),
            hovertemplate="<b>%{y}</b> (%{customdata[0]})<br>"
                          "stopped %{customdata[1]}<br>"
                          "model wanted %{customdata[2]}<br>"
                          "%{x:.1f} s<extra></extra>",
            showlegend=False))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 340, "", "seconds lost to stop timing",
                              legend=False), width="stretch")
        st.dataframe(
            top[["driver", "compounds", "actual_pit_laps", "model_pit_laps",
                 "loss_s", "lo", "hi"]].round(1),
            width="stretch", hide_index=True)
        st.caption(
            "Compound sequence and stop count held at what the driver actually "
            "ran, so this isolates the *timing* decision — the one the pit wall "
            "owns on the day. Bars carry 90% credible intervals."
        )

# ==========================================================================
# 2. TYRE MODEL
# ==========================================================================
def _tab_tyre_model():
    curves = load_pq(f"curves_{key}.parquet")
    knee = load_pq(f"knee_{key}.parquet")
    life = load_pq(f"life_{key}.parquet")

    st.markdown("#### Degradation, in the regime the race is run in")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        show_practice = st.toggle(
            "Overlay the practice-regime curve", value=True, key="tm_prac",
            help="What a flat-out practice long run does, before the "
                 "practice→race transfer is applied.")
        sub = curves[curves["variant"] == "2026_race"]
        if sub.empty:
            sub = curves[curves["variant"] == "2026"]
        fig = go.Figure()
        for cmp_ in in_ladder(sub["compound"].unique()):
            g = sub[sub["compound"] == cmp_].sort_values("tyre_age")
            col = ccol(cmp_)
            fig.add_trace(go.Scatter(
                x=list(g["tyre_age"]) + list(g["tyre_age"])[::-1],
                y=list(g["hi"]) + list(g["lo"])[::-1], fill="toself",
                fillcolor=rgba(col, 0.14), line=dict(width=0),
                name=f"{cmp_} 90%", hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=g["tyre_age"], y=g["mean"], mode="lines", name=cmp_,
                line=dict(color=col, width=2.5),
                hovertemplate=f"{cmp_} · age %{{x:.0f}}<br>"
                              "%{y:.2f} s lost<extra></extra>"))
            # Direct label *inside* the plot area — identity is never carried
            # by colour alone, and a label hung off the right edge gets clipped
            # by the chart margin.
            lab = g[g["tyre_age"] <= 34].iloc[-1]
            fig.add_annotation(x=lab["tyre_age"], y=lab["mean"], text=cmp_,
                               showarrow=False, xanchor="center", yshift=13,
                               font=dict(color=col, size=12))
        if show_practice:
            pr = curves[curves["variant"] == "2026"]
            for cmp_ in in_ladder(pr["compound"].unique()):
                g = pr[pr["compound"] == cmp_].sort_values("tyre_age")
                fig.add_trace(go.Scatter(
                    x=g["tyre_age"], y=g["mean"], mode="lines",
                    name=f"{cmp_} (practice)", line=dict(color=ccol(cmp_), width=1.4,
                                                         dash="dot"),
                    showlegend=False, hoverinfo="skip"))
        if not life.empty:
            for _, r in life.iterrows():
                fig.add_vline(x=r["knee_lap"],
                              line=dict(color=rgba(ccol(r["compound"]), 0.5),
                                        width=1, dash="dash"))
        # Everything to the right of the oldest tyre actually run in practice
        # is the hinge extrapolating. Saying so on the chart is cheaper than
        # saying it in a caption nobody reads.
        sup = meta.get("age_support_by_compound", {})
        x_max = 40
        if sup:
            edge = float(max(sup.values()))
            fig.add_vrect(x0=edge, x1=x_max, line_width=0,
                          fillcolor=rgba(T["muted"], 0.10), layer="below",
                          annotation_text="beyond practice support",
                          annotation_position="top right",
                          annotation_font=dict(color=T["muted"], size=11))
        fig.update_xaxes(range=[0, x_max])
        st.plotly_chart(style(fig, 430, "time lost vs a fresh tyre (s)",
                              "tyre age (laps)"), width="stretch")
        st.caption(
            "Solid: race regime — the curve the plan is built on. Dotted: the same "
            "posterior as fitted on practice long runs. Vertical dashes are each "
            "compound's fitted cliff; the shaded right-hand region is past the "
            "oldest tyre any practice long run reached ("
            + ", ".join(f"{c} {sup.get(c, float('nan')):.0f}"
                        for c in in_ladder(sup)) + " laps)."
        )
    with c2:
        st.markdown("**The compound ladder**")
        lt = pd.DataFrame(ladder.get("table", []))
        slopes = {r["compound"]: r["slope_s_per_lap"] for r in meta["bayes"]["slopes"]}
        offs = ladder.get("fitted_offsets", meta["bayes"]["comp_offset"])
        if not lt.empty:
            order = in_ladder(lt["compound"])
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[slopes.get(c, np.nan) for c in order], y=order, orientation="h",
                marker=dict(color=[ccol(c) for c in order]), width=0.5,
                text=[f"  {slopes.get(c, float('nan')):.3f} s/lap" for c in order],
                textposition="outside", textfont=dict(color=T["ink"]),
                showlegend=False,
                hovertemplate="%{y}<br>%{x:.3f} s/lap<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(range=[0, max(slopes.values()) * 1.55])
            st.plotly_chart(style(fig, 200, "",
                                  "practice degradation (s/lap)", legend=False),
                            width="stretch")
            tbl = pd.DataFrame({
                "compound": order,
                "deg (s/lap)": [round(slopes.get(c, np.nan), 3) for c in order],
                "pace vs softest (s/lap)": [round(offs.get(c, np.nan), 3)
                                            for c in order],
                # the cliff the sealed file reports: the circuit's longest / p90
                # stint where the fit carries no hinge, the posterior knee otherwise
                "cliff (lap)": [round(float((meta.get("cliff_history") or {}).get(c, {}).get("p90_stint",
                                      next((r.get("knee_lap", np.nan) for r in meta["bayes"]["slopes"]
                                            if r["compound"] == c), np.nan))), 1) for c in order],
            })
            st.dataframe(tbl, width="stretch", hide_index=True)
        ul = ladder.get("unladdered_slopes", {})
        if ul:
            bad = not ladder.get("unladdered_ordered", True)
            callout(
                "<b>Without the ladder, the same laps give</b> "
                + ", ".join(f"{c} {ul.get(c, float('nan')):.3f}" for c in in_ladder(ul))
                + (" — the ordering inverts, and a softer tyre comes out lasting "
                   "longer than a harder one." if bad else
                   " — ordered here, but on a knife edge: it is not enforced, so "
                   "it is not guaranteed."),
                "warn" if bad else "muted")

    why("Why the ladder is built in rather than fitted",
        f"""
A practice long run starts on an unknown fuel load in an unknown engine mode, so
the free per-stint level that absorbs those also absorbs the whole compound pace
difference. And the compounds are not run in comparable conditions:
{", ".join(f"{r['compound']} {r['laps']} laps" for r in meta["compound_counts"])}
at this weekend. Fitted freely, Barcelona 2026 returned MEDIUM *quicker* than
SOFT and SOFT degrading *more slowly* than MEDIUM — both physically impossible,
and together they are why an earlier build recommended two 23-lap SOFT stints.

So `lin`, `comp_offset` and `knee` are not sampled per compound. They are built
by accumulating **strictly positive steps** down a hardness ladder, so a harder
compound can never come out quicker or faster-degrading than a softer one, while
the *size* of every step stays free for the data to set. It is the same move as
the fuel prior — pin what the data cannot identify, estimate what it can.

**Step sizes.** Pace: {ladder.get('pace_step_s', float('nan')):.3f} s per compound
({ladder.get('label', '—')}). Measured on this weekend's own race by the
adjacent-stint estimator: {ladder.get('self_measured', {}).get('step_s', float('nan')):+.3f}
± {ladder.get('self_measured', {}).get('se', float('nan')):.3f} s over
{ladder.get('self_measured', {}).get('n_laps', 0)} race laps — reported here,
never used to fit this weekend.
""")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### The practice → race transfer")
        r = regime
        st.markdown(
            chip("Applied factor", f"{r.get('ratio', float('nan')):.2f}×",
                 T["accent"], f"90% {r.get('p05', 0):.2f}–{r.get('p95', 0):.2f}")
            + chip("Source", "measured" if r.get("measured") else "default",
                   T["good"] if r.get("measured") else T["warn"],
                   ", ".join(r.get("sources", [])) or "no donor race")
            + chip("This weekend", f"{(r.get('self_measured', {}) or {}).get('ratio') or float('nan'):.2f}×",
                   T["muted"], "measured, never used"),
            unsafe_allow_html=True)
        per = (regime.get("self_measured", {}) or {}).get("per_compound", {})
        if per:
            rows = [{"compound": c,
                     "practice (s/lap)": round(v["practice_s_per_lap"], 3),
                     "race (s/lap)": round(v["race_s_per_lap"], 3),
                     "ratio": (round(v["ratio"], 2) if v.get("usable") else "—")}
                    for c, v in per.items()]
            df = pd.DataFrame(rows)
            df["_o"] = df["compound"].map({c: i for i, c in enumerate(LADDER)})
            st.dataframe(df.sort_values("_o").drop(columns="_o"),
                         width="stretch", hide_index=True)
    with c2:
        st.markdown("#### Fuel load makes early tyres wear faster")
        le = meta.get("load_effect", {})
        laps = np.arange(1, ev.n_race_laps + 1)
        m = 768.0 + 70.0 * (1 - (laps - 1) / (ev.n_race_laps - 1))
        mult = (m / 803.0) ** le.get("exponent", 5.0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=laps, y=mult, mode="lines",
                                 line=dict(color=T["accent"], width=2.5),
                                 name="degradation multiplier",
                                 hovertemplate="lap %{x}<br>%{y:.2f}×<extra></extra>"))
        fig.add_hline(y=1.0, line=dict(color=GREY, width=1))
        st.plotly_chart(style(fig, 260, "degradation multiplier", "race lap",
                              legend=False), width="stretch")
        st.caption(
            f"A full tank wears the tyre {le.get('start_multiplier', 1):.2f}× as fast "
            f"as mid-race, an empty one {le.get('flag_multiplier', 1):.2f}×. This is "
            "why real stints get longer through a race — and why the plan above "
            "does not split the distance evenly."
        )

    why("Where the regime factor comes from — and why it is no longer an input",
        f"""
{regime.get('derivation', '')}

A practice long run and a race stint are not the same experiment. In a long run
the driver pushes every lap to gather data; in the race the same driver lifts
and coasts, manages surface temperature and is on an energy plan. Every one of
those puts less energy through the contact patch, and degradation is driven by
exactly that.

**But this is a choice, not a law.** It is what you observe *when a driver
decides to manage a tyre* — and whether to make that trade is precisely the
question a strategy has to answer. So the strategy model no longer multiplies
degradation by this factor. It optimises the push level directly, paying for
management in lap time, and the regime factor falls out as a *prediction*
({strat_meta.get('implied_regime', float('nan')):.2f} here) that can be checked
against this measurement ({regime.get('ratio', float('nan')):.2f}). The number
above is now a cross-check, not a dial.

**The estimator behind it was also wrong, and that mattered more.** It removed
track evolution from the practice side and not from the race side, then divided
one by the other. Within a stint, race lap and tyre age advance together, so the
race slope absorbed the whole evolution drift — about −0.08 s/lap at both 2026
weekends — and came out biased low on every compound. At Hungary that drove the
measured race degradation *negative*, which is not a thing tyres do. Corrected,
with driver and race-lap fixed effects on both sides, Barcelona reads 0.57–0.68
and Hungary 0.72–0.88, against the 0.37 the old estimator returned. Everything
downstream inherited that error: degradation was under-predicted by 1.3× at
Barcelona and 3.6× at Hungary, and an optimiser fed degradation three times too
low answers with too few stops and stints far longer than any team runs.

**The firewall still holds.** The factor for a weekend is measured on other
weekends' races, never on the target weekend's.
""")

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### The cliff, as a distribution")
        _kv = "2026" if (not knee.empty and (knee["variant"] == "2026").any()) else "hinge"
        ks = knee[knee["variant"] == _kv] if not knee.empty else knee
        if ks.empty:
            st.caption("The production fit carries no hinge: the knee is unidentified on every "
                       "scored weekend, and the cliff is reported from the circuit's race history instead.")
        else:
            fig = go.Figure()
            for cmp_ in in_ladder(ks["compound"].unique()):
                g = ks[ks["compound"] == cmp_]
                fig.add_trace(go.Histogram(
                    x=g["knee"], name=cmp_, opacity=0.62, nbinsx=45,
                    marker_color=ccol(cmp_)))
            fig.update_layout(barmode="overlay")
            st.plotly_chart(style(fig, 330, "posterior draws", "cliff lap"),
                            width="stretch")
            if _kv == "hinge":
                st.caption("Diagnostic hinge variant: the knee posterior equals its prior, which is why "
                           "the production fit is linear and the cliff comes from the circuit's races.")
    with c2:
        st.markdown("#### What the fuel prior is worth")
        ps = pd.DataFrame(meta["prior_sensitivity"]).T
        order = in_ladder(ps.index)
        fig = go.Figure()
        names = {"2026": "2026 physics (ours)", "2025": "2025 physics",
                 "none": "no fuel prior"}
        for i, v in enumerate(["2026", "2025", "none"]):
            fig.add_trace(go.Bar(
                x=order, y=[ps.loc[c, v] for c in order], name=names[v],
                marker=dict(color=[T["accent"], T["warn"], T["bad"]][i],
                            line=dict(color=T["surface"], width=2)),
                hovertemplate="%{x} · " + names[v] + "<br>%{y:.2f} s/lap<extra></extra>"))
        fig.update_yaxes(type="log", tickmode="array",
                         tickvals=[0.1, 0.2, 0.5, 1, 2, 5],
                         ticktext=["0.1", "0.2", "0.5", "1", "2", "5"])
        st.plotly_chart(style(fig, 330, "fitted slope (s/lap, log scale)", ""),
                        width="stretch")
        b = meta["bayes"]
        st.caption(
            f"With no fuel prior the free regression runs away — age and fuel burn "
            f"are collinear, so nothing stops it. `k_track` posterior "
            f"{b['k_track_mean']:.4f} ± {b['k_track_sd']:.4f} s/kg, a "
            f"{b['k_track_rel_sd']:.0%} spread against the 25% prior: the lap-time "
            f"channel carries almost no information about the split, so the prior "
            f"is doing the work — and it is derived from the regulations."
        )

# ==========================================================================
# 3. EVIDENCE
# ==========================================================================
def _tab_evidence():
    casc = load_pq(f"cascade_{key}.parquet")
    clean = load_pq(f"clean_{key}_practice.parquet")
    if casc.empty:
        st.warning(f"No clean-lap cascade on disk for {ev.name}. Build one with "
                   f"`make weekend EVENT={key}`.")
        return

    # Which practice sessions actually reached the fit.  The pipeline skips a
    # session whose timing archive is not published yet — the normal state of
    # FP3 on a Saturday morning — and says so only on its own stdout, so
    # without this the tab simply looks short of laps for no stated reason.
    used = list(meta.get("sessions_used") or [])
    if not used and not clean.empty:
        used = list(clean["session"].unique())
    missing = [s for s in ev.practice_sessions if s not in used]
    st.markdown(
        chip("Practice fitted", f"{len(used)}/{len(ev.practice_sessions)}",
             T["warn"] if missing else T["good"], ", ".join(used) or "none")
        + chip("Built", _built_at(f"clean_{key}_practice.parquet"), T["muted"],
               "from the files on disk"),
        unsafe_allow_html=True)
    if missing:
        callout(
            f"<b>{', '.join(missing)} is not in this fit.</b> Its timing archive had not been "
            f"published when the model was last built, so the pipeline fitted on "
            f"{', '.join(used) or 'nothing'} alone. The supervisor folds it in at the next refit; "
            f"<code>make weekend EVENT={key}</code> forces one now.", "warn")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("#### Clean-lap cascade — laps falling away, rule by rule")
        fig = go.Figure(go.Bar(
            x=casc["laps"], y=casc["rule"], orientation="h", width=0.55,
            marker=dict(color=[T["accent"]] + [rgba(T["accent"], 0.45)]
                        * (len(casc) - 1), line=dict(color=T["surface"], width=2)),
            text=[f"  {n}  (−{d})" if d else f"  {n}" for n, d in
                  zip(casc["laps"], casc["dropped"])],
            textposition="outside", textfont=dict(color=T["ink"]),
            showlegend=False,
            hovertemplate="%{y}<br>%{x} laps remain<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, casc["laps"].max() * 1.28])
        st.plotly_chart(style(fig, 360, "", "laps remaining", legend=False),
                        width="stretch")
    with c2:
        st.markdown("#### Surviving laps by compound")
        comp = pd.DataFrame(meta["compound_counts"])
        comp["_o"] = comp["compound"].map({c: i for i, c in enumerate(LADDER)})
        comp = comp.sort_values("_o").drop(columns="_o")
        sup = meta.get("age_support_by_compound", {})
        caps = meta.get("max_stint_laps", {})
        comp["age support"] = [sup.get(c, np.nan) for c in comp["compound"]]
        comp["stint cap"] = [caps.get(c, np.nan) for c in comp["compound"]]
        fig = go.Figure(go.Bar(
            x=comp["compound"], y=comp["laps"],
            marker=dict(color=[ccol(c) for c in comp["compound"]],
                        line=dict(color=T["surface"], width=2)),
            width=0.5, text=comp["laps"], textposition="outside",
            textfont=dict(color=T["ink"]), cliponaxis=False, showlegend=False,
            hovertemplate="%{x}<br>%{y} clean laps<extra></extra>"))
        fig.update_yaxes(range=[0, float(comp["laps"].max()) * 1.22])
        st.plotly_chart(style(fig, 250, "clean laps", "", legend=False),
                        width="stretch")
        st.dataframe(comp.round(1), width="stretch", hide_index=True)
        thin = comp.loc[comp["laps"].idxmin()]
        callout(
            f"<b>{thin['compound']} is thin: {int(thin['laps'])} laps from "
            f"{int(thin['stints'])} stint(s).</b> This is not hidden — it is why "
            f"that compound's credible band is wide, why its stint cap is the "
            f"tightest, and why the ladder rather than the likelihood sets its "
            f"position.", "warn")

    st.markdown("---")
    st.markdown("#### The peel: raw → fuel-corrected → evolution-corrected")
    if not clean.empty:
        c1, c2 = st.columns([1, 2])
        with c1:
            sess = st.selectbox("Session", sorted(clean["session"].unique()),
                                key="dec_s")
        with c2:
            stage = st.radio("Stage",
                             ["raw lap times", "fuel-corrected",
                              "fuel + evolution corrected"],
                             horizontal=True, key="dec_stage")
        g = clean[clean["session"] == sess]
        col = {"raw lap times": "lap_time_s",
               "fuel-corrected": "lap_time_fuel_corr",
               "fuel + evolution corrected": "lap_time_corr"}[stage]
        fig = go.Figure()
        for cmp_ in in_ladder(g["compound"].unique()):
            sub = g[g["compound"] == cmp_]
            c = ccol(cmp_)
            fig.add_trace(go.Scatter(
                x=sub["tyre_age"], y=sub[col], mode="markers", name=cmp_,
                marker=dict(color=c, size=8,
                            line=dict(width=2, color=T["surface"])),
                hovertemplate=f"{cmp_} · age %{{x:.0f}}<br>%{{y:.2f}} s<extra></extra>"))
            if len(sub) > 3:
                b = np.polyfit(sub["tyre_age"], sub[col], 1)
                xs = np.linspace(sub["tyre_age"].min(), sub["tyre_age"].max(), 20)
                fig.add_trace(go.Scatter(
                    x=xs, y=np.polyval(b, xs), mode="lines", showlegend=False,
                    line=dict(color=c, width=2, dash="dot"),
                    hovertemplate=f"{cmp_}: {b[0]:+.3f} s/lap<extra></extra>"))
        st.plotly_chart(style(fig, 420, "lap time (s)", "tyre age (laps)"),
                        width="stretch")

    ph = meta["physics"]
    callout(
        f"<b>{meta['n_raw_laps']} raw practice laps</b> reduce to "
        f"<b>{meta['n_clean_laps']} clean long-run laps</b> through seven "
        f"<i>absolute</i> rules. The fuel prior is derived, not assumed: "
        f"{ph['derivation']}, worth <b>{ph['fuel_effect_s_per_lap']:.3f} s/lap</b>. "
        f"Track evolution is then backfitted with a monotone-decreasing isotonic "
        f"curve spanning {meta['evolution']['range_s']:.2f} s.")

    why("Why every rule is absolute, never relative to the stint minimum",
        f"""
A `t ≤ 1.03 × stint_min` filter selects on the outcome: the minimum usually
falls early in the stint, so the rule preferentially deletes late laps and
*inflates* the very degradation slope it is meant to measure.

**Track status is mandatory, not optional.** The {ev.name} race ends under a
safety car — final stints show +17 to +21 s in the last three laps for *every*
driver. Without that filter those laps poison every late-stint fit.
""")

def _outlook_tyre_fallback():
    st.info("No practice fit yet. The outlook's tyre picture is below; the fitted curves, the compound ladder "
            "and the practice→race transfer appear here after the first practice session.")
    st_ = (OUTLOOK or {}).get("strategy") or {}
    life = st_.get("life") or {}
    if life:
        rows = [{"compound": c, "deg (s/lap)": round(v["deg_s_per_lap"], 3),
                 "deg 90%": f"{v['deg_lo']:.3f} – {v['deg_hi']:.3f}",
                 "life at plan push (laps)": round(v["life_laps"]), "life 90%": f"{v['life_lo']:.0f} – {v['life_hi']:.0f}",
                 "life at full push": round(v["life_full_push"]), "pace vs softest (s/lap)": round(v["pace_offset_s"], 3)}
                for c in ("SOFT", "MEDIUM", "HARD") if (v := life.get(c))]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Built on: " + "; ".join((OUTLOOK or {}).get("sources", [])))


with tab["Tyre model"]:
    if OUTLOOK_ONLY:
        _outlook_tyre_fallback()
    else:
        _tab_tyre_model()

with tab["Evidence"]:
    if OUTLOOK_ONLY:
        st.info("No practice laps yet: the clean-lap cascade and the peel appear after the first practice session.")
    else:
        _tab_evidence()


# ==========================================================================
# 4. VALIDATION
# ==========================================================================
with tab["Validation"]:
    if WEEKEND_ONLY:
        st.info('The race has not been run yet: the sealed prediction will be scored after the flag (scripts/60_postrace.py).')
    else:
        fw = next((g for g in meta["gates"] if "firewall" in g["gate"]), None)
        cov90 = sc["coverage"].get("0.9", float("nan"))

        st.markdown(
            chip("Stint-rate MAE", f"{sc['mae']:.3f}",
                 T["good"] if sc["passes_mae"] else T["bad"], "s/lap · target < 0.15")
            + chip("Bias", f"{sc['bias']:+.3f}",
                   T["good"] if abs(sc["bias"]) < 0.06 else T["warn"], "s/lap")
            + chip("90% coverage", f"{cov90:.0%}",
                   T["good"] if sc["passes_coverage"] else T["bad"],
                   "conservative" if cov90 > 0.97 else "in band")
            + chip("Per-lap MAE", f"{sc.get('mae_lap', float('nan')):.2f}", T["muted"],
                   "s · noise-bounded")
            + chip("Scored on", f"{sc['n_rate_stints']}", T["muted"],
                   "clean race stints"),
            unsafe_allow_html=True)
        callout(
            f"Scored against the <b>{sc.get('regime_label', 'race regime')}</b> curves — "
            f"the sealed practice fit scaled by the transferred "
            f"{regime.get('ratio', float('nan')):.2f}× regime factor. That is the curve "
            f"that makes a claim about the race, so that is the one under test.",
            "good" if abs(sc["bias"]) < 0.06 else "warn")

        st.markdown(f"#### Gates — {n_pass} of {n_gate} passing")
        gdf = pd.DataFrame(meta["gates"])
        # A status list, not a bar chart: every bar would be the same length, so
        # length would encode nothing while looking like it encoded something.
        rows = []
        for _, g in gdf.iterrows():
            col = T["good"] if g["pass"] else T["bad"]
            rows.append(
                f"<div style='display:flex;gap:10px;align-items:baseline;"
                f"padding:6px 10px;border-radius:6px;margin-bottom:3px;"
                f"background:{rgba(col, 0.07)};border-left:3px solid {col}'>"
                f"<span style='color:{col};font-weight:700;font-size:0.72rem;"
                f"min-width:34px'>{'PASS' if g['pass'] else 'FAIL'}</span>"
                f"<span style='color:{T['ink']};font-size:0.86rem;min-width:330px'>"
                f"{g['gate']}</span>"
                f"<span style='color:{T['muted']};font-size:0.79rem'>{g['detail']}</span>"
                f"</div>")
        st.markdown("".join(rows), unsafe_allow_html=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Calibration — nominal vs empirical coverage")
            cov = pd.DataFrame({"nominal": [float(k) for k in sc["coverage"]],
                                "empirical": list(sc["coverage"].values())}
                               ).sort_values("nominal")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0.4, 1], y=[0.4, 1], mode="lines",
                                     line=dict(color=GREY, dash="dash", width=1),
                                     name="perfect calibration"))
            fig.add_trace(go.Scatter(x=cov["nominal"], y=cov["empirical"],
                                     mode="lines+markers", name="degless",
                                     line=dict(color=T["accent"], width=2.5),
                                     marker=dict(size=10,
                                                 line=dict(color=T["surface"], width=2)),
                                     hovertemplate="nominal %{x:.0%}<br>"
                                                   "empirical %{y:.1%}<extra></extra>"))
            st.plotly_chart(style(fig, 380, "empirical", "nominal"), width="stretch")
            st.caption(
                "Above the line is over-coverage: the intervals are wider than they "
                "need to be. `sigma_obs` is *practice* per-lap noise (~0.9 s of traffic, "
                "engine modes, fuel saving) while race stints are scored centred, which "
                "removes most of that — and the regime factor's own spread widens it "
                "further. Conservative, disclosed, not tuned away."
            )
        with c2:
            st.markdown("#### Predicted vs observed stint degradation rate")
            ps_ = load_pq(f"scorestint_{key}.parquet")
            if not ps_.empty:
                fig = go.Figure()
                lim = [min(ps_["pred_rate"].min(), ps_["obs_rate"].min()) - 0.05,
                       max(ps_["pred_rate"].max(), ps_["obs_rate"].max()) + 0.05]
                fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="perfect",
                                         line=dict(color=GREY, dash="dash", width=1)))
                for cmp_ in in_ladder(ps_["compound"].unique()):
                    g = ps_[ps_["compound"] == cmp_]
                    fig.add_trace(go.Scatter(
                        x=g["pred_rate"], y=g["obs_rate"], mode="markers", name=cmp_,
                        marker=dict(size=10, color=ccol(cmp_),
                                    line=dict(width=2, color=T["surface"])),
                        text=g["driver"],
                        hovertemplate="%{text}<br>pred %{x:.3f} · obs %{y:.3f}"
                                      "<extra></extra>"))
                st.plotly_chart(style(fig, 380, "observed (s/lap)", "predicted (s/lap)"),
                                width="stretch")
                st.caption(f"MAE by compound: "
                           + ", ".join(f"**{k}** {v:.3f}"
                                       for k, v in sc["mae_by_compound"].items()))

        if fw:
            st.success(
                f"**Practice-only firewall — enforced, not promised.** "
                f"`load_for_fitting(session=\"Race\")` raises before a single byte is "
                f"read: *{fw['detail']}*"
            )
        st.caption(
            f"Predictions sealed to `{meta['sealed_file']}` with sha256 "
            f"`{meta['sealed_sha256'][:32]}…` **before** any race lap was opened. "
            "Scoring reads the sealed file back and verifies the hash."
        )

        why("What the validation is actually testing",
            f"""
    A practice fit cannot know the base pace of a race stint — different fuel load,
    engine mode, track state. What it *can* predict is the **shape** of the stint:
    how much slower lap 15 is than lap 3. So each race stint is centred on its own
    mean and compared to the centred prediction. That concedes exactly one degree of
    freedom per stint and scores the thing the model actually claims to know.

    **The headline number moved for a reason.** An earlier build scored the raw
    practice curves against race stints. On Barcelona that produced a stint-rate MAE
    of 0.174 s/lap and a systematic **−0.174 s/lap** bias in the same direction on
    every compound — a regime difference, not noise. That gap is now modelled
    explicitly (`src/regime.py`) rather than reported and lived with, and Barcelona's
    residual is 0.051 s/lap at **+0.036**. The transfer factor came from a *different*
    weekend, so this is an out-of-sample test of it, not a fit.

    **The transfer is not free.** It is a single multiplicative factor, so it helps
    most where degradation is high and over-corrects where it is already low — and
    the spread between what the two weekends measure (0.57–0.68 at Barcelona,
    0.72–0.88 at Hungary) is itself the point: this was never a transferable
    constant. That is why the strategy optimiser derives it from the push level it
    chooses rather than being handed it. The direction of
    that trade is real and worth knowing before trusting the factor at a weekend that
    looks nothing like the donor. This weekend's own residual is
    {sc['bias']:+.3f} s/lap.

    **Two errors, measuring different things.** `mae_stint_rate` ({sc['mae']:.3f} s/lap)
    is the headline and the one strategy depends on. `mae_lap`
    ({sc.get('mae_lap', float('nan')):.2f} s) is bounded below by the per-lap noise of a
    racing lap and no tyre model drives it to 0.15 — reported for honesty, not as a
    target.
    """)

        if sc.get("cliff"):
            st.markdown("**Cliff error — posterior knee vs observed race collapse**")
            st.dataframe(pd.DataFrame(sc["cliff"]).T.round(2), width="stretch")

# ==========================================================================
# 5. REPLAY
# ==========================================================================
with tab["Replay"]:
    if WEEKEND_ONLY:
        st.info('Replay appears after the race; during the race use the Live tab.')
    else:
        rp = load_pq(f"replay_{key}.parquet")
        if rp.empty:
            st.info("No replay data.")
        else:
            drivers = sorted(rp["driver"].unique())
            c1, c2 = st.columns([1, 3])
            with c1:
                drv = st.selectbox("Driver", drivers, key="rp_drv")
            d = rp[rp["driver"] == drv].sort_values("lap_number")
            with c2:
                lo_l, hi_l = int(d["lap_number"].min()), int(d["lap_number"].max())
                # Key on the driver: each driver ran a different number of laps, so a
                # slider value carried over can fall outside the new range.
                lap = st.slider("Lap", lo_l, max(hi_l, lo_l + 1), lo_l,
                                key=f"rp_lap_{drv}")
            upto = d[d["lap_number"] <= lap]
            now = upto.iloc[-1] if len(upto) else None

            if now is not None:
                past = float(now["p_past_cliff"])
                st.markdown(
                    chip("Tyre age", f"{now['tyre_age']:.0f}", T["accent"], "laps")
                    + chip("Compound", str(now.get("compound", "—")),
                           ccol(str(now.get("compound", ""))))
                    + chip("Laps of evidence", f"{int(now['laps_seen'])}", T["muted"])
                    + chip("Uncertainty", f"±{now['stint_pace_sd']:.3f}"
                           if np.isfinite(now["stint_pace_sd"]) else "—", T["muted"], "s")
                    + chip("P(past cliff)", f"{past:.0%}",
                           T["bad"] if bool(now["cliff_alarm"]) else T["good"]),
                    unsafe_allow_html=True)
                if bool(now["cliff_alarm"]):
                    callout(
                        f"<b>PACE COLLAPSE — lap {int(now['lap_number'])}.</b> "
                        f"This stint's own laps have broken away from their trend (the within-stint "
                        f"detector, not the wear estimate); {past:.0%} posterior probability the grip "
                        f"budget is spent. Degradation so far this stint: {now['deg_now_s']:.2f} s/lap.",
                        "bad")
                else:
                    callout(f"Tyre on trend — no pace collapse detected; {past:.0%} past-cliff "
                            f"probability.", "good")

            fig = go.Figure()
            for cmp_ in in_ladder(d["compound"].dropna().unique()):
                g = d[d["compound"] == cmp_]
                fig.add_trace(go.Scatter(
                    x=g["lap_number"], y=g["lap_time_s"], mode="markers", name=cmp_,
                    marker=dict(size=7, color=rgba(ccol(cmp_), 0.55),
                                line=dict(width=1, color=T["surface"])),
                    hovertemplate=f"{cmp_} · lap %{{x}}<br>%{{y:.2f}} s<extra></extra>"))
            band = upto.dropna(subset=["band_lo", "band_hi"])
            if len(band):
                fig.add_trace(go.Scatter(
                    x=list(band["lap_number"]) + list(band["lap_number"])[::-1],
                    y=list(band["band_hi"]) + list(band["band_lo"])[::-1],
                    fill="toself", fillcolor=rgba(T["accent"], 0.16),
                    line=dict(width=0), name="90% band", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=upto["lap_number"], y=upto["stint_pace_est"], mode="lines",
                name="stint pace estimate", line=dict(color=T["accent"], width=2.5)))
            for _, r in d[d["is_pit_in"]].iterrows():
                fig.add_vline(x=r["lap_number"],
                              line=dict(color=T["hairline"], width=1))
            fig.add_vline(x=lap, line=dict(color=T["warn"], width=2))
            # In-laps, out-laps and safety-car laps run 20-30 s long; left in the
            # y-range they compress every racing lap into a few pixels.  Clip to
            # the racing band and let the outliers sit off the top.
            lt = d["lap_time_s"].dropna()
            if len(lt) > 5:
                lo_y = float(lt.quantile(0.01)) - 1.0
                hi_y = float(lt.quantile(0.93)) + 2.5
                span = [c for c in [upto["band_lo"].min(), upto["band_hi"].max()]
                        if np.isfinite(c)]
                if span:
                    lo_y, hi_y = min(lo_y, min(span) - 0.5), max(hi_y, max(span) + 0.5)
                fig.update_yaxes(range=[lo_y, hi_y])
            st.plotly_chart(style(fig, 430, "lap time (s)", "race lap"),
                            width="stretch")
            st.caption(
                "Pure playback of a precomputed parquet — the app fits nothing at "
                "runtime. As laps accumulate the estimate sharpens and the band "
                "narrows as 1/√n; the alarm fires when the posterior probability the "
                "tyre has spent its grip budget crosses 50%. Thin verticals are actual pit "
                "laps; the y-axis is clipped to racing pace, so in-laps and "
                "safety-car laps sit off the top of the chart."
            )

# ==========================================================================
# 6. ENGINEER
# ==========================================================================
with tab["Engineer"]:
    st.markdown("#### AI race engineer")
    st.caption(
        "Grounded strictly on the sealed analysis. The model is handed a fact "
        "sheet computed by the pipeline and instructed never to estimate a "
        "number — an LLM guessing degradation figures would undo the point of "
        "everything in the other five tabs."
    )

    @st.cache_data(show_spinner=False)
    def _brief(k: str):
        r = engineer_brief(k)
        return r.text, r.source

    _cred = credential_source()
    if _cred:
        st.caption(f"Connected — using **{_cred}**, model `{model_name()}`.")
    else:
        st.caption(
            "Not connected — running the offline briefing. Put `GEMINI_API_KEY=...` in `.env` "
            "next to the Makefile, then restart."
        )

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Brief me", type="primary", key="eng_brief",
                     width="stretch"):
            with st.spinner("Briefing…"):
                txt, src = _brief(key)
            st.session_state["eng_out"] = (txt, src)
    with c2:
        q = st.text_input("Ask a question about this weekend's analysis",
                          placeholder="e.g. should we have committed to a one-stop?",
                          key="eng_q", label_visibility="collapsed")
    if st.button("Ask", key="eng_ask") and q.strip():
        with st.spinner("Thinking…"):
            r = engineer_ask(key, q.strip())
        st.session_state["eng_out"] = (r.text, r.source)

    if "eng_out" in st.session_state:
        txt, src = st.session_state["eng_out"]
        st.markdown(txt)
        if src == "offline":
            callout(
                "<b>Offline mode.</b> No Gemini key in `.env`, so this is the "
                "deterministic briefing assembled from the same fact sheet. The "
                "dashboard behaves identically either way — which is what makes it "
                "safe to demo.", "warn")
