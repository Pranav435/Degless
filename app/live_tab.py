"""The Now tab: what the pit wall sees during a session, and between sessions
the race forecast.

Reads the snapshots the daemon (`scripts/50_live.py`) writes under
`data/live/<session>/` and re-renders every few seconds.  The app never
touches the feed or the model; if the daemon dies the last snapshot stays on
screen with its age shown, and the app itself cannot fall over.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.theme import (
    BLACK, DEFS, DIM, FAINT, HAIR, LADDER, LEVEL_KIND, MUTED, WHITE, age, badge, badges, card, ccol, chart,
    finite, fmt, headline, label_text, more, notice, pct, pit_loss_text, plan_text, rgba, stops_word, style, tiles,
    tyre_pill, verdict_word, window_line,
)
from src import plans as plan_store
from src import schedule
from src.live.store import LIVE_DIR, read_history, read_laps, read_snapshot, read_status

SHORT = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Qualifying": "Quali",
         "Sprint Qualifying": "Sprint quali", "Sprint Shootout": "Sprint quali"}
TRACK = {"1": ("Green flag", "ok"), "2": ("Yellow flag", "flag"), "4": ("Safety car", "flag"),
         "5": ("Red flag", "alert"), "6": ("VSC", "flag"), "7": ("VSC ending", "flag")}
ALERT_KIND = {"bad": "alert", "warn": "alert", "good": "ok", "accent": "info"}
ALERT_ICON = {"bad": "⚠", "warn": "⚠", "good": "✓", "accent": "●"}


def read_supervisor() -> dict:
    p = LIVE_DIR / "supervisor.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        d["_age_s"] = (datetime.now(timezone.utc) - datetime.fromisoformat(d["updated"])).total_seconds()
        return d
    except Exception:
        return {}


def weekend_status() -> dict:
    """Schedule from the supervisor if it is running, else computed here."""
    sup = read_supervisor()
    if sup and sup.get("_age_s", 1e9) < 120:
        st_ = sup["schedule"]
        st_["_supervisor"] = sup
        return st_
    st_ = schedule.status()
    st_["_supervisor"] = None
    return st_


# --------------------------------------------------------------------------
# Page top and sidebar status
# --------------------------------------------------------------------------


def render_strip(wk: dict, data_badge: tuple | None = None) -> None:
    """The line everybody reads first: what is on, what is next, and the weekend's sessions."""
    st.markdown(f"**{schedule.describe(wk)}**")
    items = []
    for s_ in wk.get("weekend", []):
        name = SHORT.get(s_["name"], s_["name"])
        if s_["state"] == "live":
            items.append((f"{name} · live", "alert", ":material/radio_button_checked:"))
        elif s_["state"] == "finished":
            items.append((name, "ok"))
        else:
            when = datetime.fromisoformat(s_["start"]).astimezone().strftime("%a %H:%M")
            items.append((f"{name} · {when}", "info"))
    if data_badge:
        items.append(data_badge)
    badges(items)


def render_system(wk: dict) -> None:
    """Sidebar: is the supervisor running the weekend by itself?"""
    sup = wk.get("_supervisor")
    if sup is None:
        badge("Auto-updates off", "alert",
              help="Nothing starts by itself. Run `make run` (dashboard plus the live feed, refits and scoring) "
                   "instead of `make app`.")
        return
    badge("Auto-updating", "ok", help=f"The supervisor checked in {sup['_age_s']:.0f} s ago.")
    lines = []
    feed = sup.get("feed")
    if feed:
        lines.append(f"Live feed {'running' if feed.get('alive') else 'stopped'} (session {feed.get('session')})")
    if sup.get("tasks"):
        lines.append("Working on: " + ", ".join(sup["tasks"]))
    if sup.get("history"):
        lines.append("Last: " + sup["history"][-1]["text"])
    ol = sup.get("outlook") or {}
    if ol.get("event"):
        nxt = ("refreshing now" if ol.get("running") else
               (f"next refresh in {ol['next_in_s'] / 60:.0f} min" if ol.get("next_in_s") is not None else "idle"))
        lines.append(f"Race forecast for {ol['event']}: {nxt}" + (f", updated {age(ol.get('updated'))}" if ol.get("updated") else ""))
    with st.popover("System details", icon=":material/settings:", width="stretch"):
        st.markdown("\n".join(f"- {ln}" for ln in lines) if lines else "Idle, watching the calendar.")


# --------------------------------------------------------------------------
# Now: live session, or the race forecast
# --------------------------------------------------------------------------


def render_now(wk: dict, meta: dict, ev, outlook: dict | None = None, timeline: list | None = None) -> None:
    """Live view when a session is on; otherwise the countdown and the race forecast."""
    live = wk.get("live")
    if live:
        sk = str(live["key"])
        if (LIVE_DIR / sk / "snapshot.json").exists():
            render_session(sk, event_key=live.get("event_key"))
        else:
            notice(f"{live['label']} is live — waiting for the first data. The feed starts by itself; otherwise "
                   f"run <code>make live EVENT={live.get('event_key')}</code>.")
        return
    nxt = wk.get("next")
    if nxt:
        start_local = datetime.fromisoformat(nxt["start"]).astimezone().strftime("%A %H:%M")
        notice(f"<b>Nothing on track.</b> Next: {nxt['label']}, {start_local} local. "
               "This page goes live by itself 15 minutes before.")
    if outlook:
        render_outlook(outlook, timeline or [], ev)
    elif (meta.get("strategy") or {}).get("best_plan"):
        _render_fit_plan(meta)
    render_history(meta)
    sessions = _sessions()
    if sessions:
        with st.expander("Recorded sessions", icon=":material/history:"):
            sk = st.selectbox("Session", sessions, key="now_replay")
            render_session(sk, event_key=ev.key)


def _render_fit_plan(meta: dict) -> None:
    """No forecast for this weekend: the practice fit's own plan."""
    s_ = meta.get("strategy") or {}
    plan, wins = s_.get("best_plan") or {}, s_.get("pit_windows") or []
    comps = plan.get("compounds", [])
    headline(f"{stops_word(len(comps) - 1)} " + " → ".join(tyre_pill(c) for c in comps),
             window_line(wins, plan.get("pit_laps", [])),
             eyebrow="Race plan · from " + (", ".join(meta.get("sessions_used", [])) or "practice"))
    rows = []
    for r in s_.get("life") or []:
        capped = finite(r.get("max_stint_laps")) and r["life_laps"] > r["max_stint_laps"]
        rows.append((str(r["compound"]), f"≤ {int(r['max_stint_laps'])} laps" if capped else f"{r['life_laps']:.0f} laps",
                     "Capped at the longest stint anyone has run here." if capped else DEFS["tyre_wear"],
                     f"loses {r['deg_s_per_lap']:.2f} s a lap"))
    tiles("fit-plan", rows)
    st.caption("Details in the Race plan and Tyres tabs.")


def _stage_text(o: dict) -> str:
    if str(o.get("stage", "")).startswith("prior"):
        s = "before any practice"
    else:
        s = "from " + (", ".join(o.get("sessions_used") or []) or "practice so far")
    live = o.get("live") or {}
    if live.get("used"):
        s += f" + live {live.get('session_name')}"
    return s


def _plan_b(o: dict):
    alts = o.get("alternatives") or {}
    alt = alts.get("plan_b") or alts.get("plan_c")
    if not alt:
        return None
    sw = alt.get("switch_mult")
    if alt["delta_s"] < 0:
        sub = "already faster"
    elif sw and alt.get("switch_direction") == "up":
        sub = f"better if tyres wear {sw - 1:.0%} faster"
    else:
        sub = "rarely better"
    beta = max((o.get("thermal") or {}).get("beta_per_c", 0.025), 1e-6)
    hlp = f"{label_text(alt['label'])} is {alt['delta_s']:+.1f} s against the plan on average."
    if sw and alt.get("switch_direction") == "up":
        hlp += f" It takes over if tyres wear ×{sw:.2f} the forecast rate (about {np.log(sw) / beta:+.0f} °C of track)."
    return ("Plan B", plan_text(alt.get("compounds", [])), hlp, sub)


def _life_range_chart(life: dict):
    order = [c for c in LADDER if c in life]
    fig = go.Figure()
    for i, c in enumerate(order):
        v, y = life[c], len(order) - 1 - i
        col = ccol(c)
        fig.add_trace(go.Bar(x=[v["life_hi"] - v["life_lo"]], y=[y], base=[v["life_lo"]], orientation="h", width=0.5,
                             marker=dict(color=rgba(col, 0.35), line=dict(color=BLACK, width=2)), showlegend=False,
                             hovertemplate=f"{c.title()}: likely {v['life_lo']:.0f}–{v['life_hi']:.0f} laps<extra></extra>"))
        fig.add_trace(go.Scatter(x=[v["life_laps"]], y=[y], mode="markers", showlegend=False,
                                 marker=dict(color=col, size=14, line=dict(color=BLACK, width=2)),
                                 hovertemplate=f"{c.title()}: {v['life_laps']:.0f} laps with the plan's tyre saving<extra></extra>"))
        fig.add_annotation(x=v["life_hi"], y=y, xanchor="left", xshift=8, showarrow=False,
                           text=f"{v['life_laps']:.0f} laps (likely {v['life_lo']:.0f}–{v['life_hi']:.0f})",
                           font=dict(color=MUTED, size=11))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order)))[::-1], ticktext=[c.title() for c in order],
                     showgrid=False)
    fig.update_xaxes(range=[0, max(v["life_hi"] for v in life.values()) * 1.45])
    fig.update_layout(barmode="overlay")
    return style(fig, 100 + 55 * len(order), "", "laps", legend=False)


def render_outlook(o: dict, timeline: list, ev) -> None:
    st_ = o.get("strategy") or {}
    if not st_:
        notice("No legal plan came out of the race forecast for this weekend yet.", "alert")
        return
    best = st_.get("best_plan") or {}
    comps, wins = best.get("compounds", []), st_.get("pit_windows", [])
    n = len(comps) - 1
    p_stops = st_.get("p_stops") or {}
    headline(f"{stops_word(n)} " + " → ".join(tyre_pill(c) for c in comps), window_line(wins, best.get("pit_laps", [])),
             eyebrow=f"Race forecast · {_stage_text(o)} · updated {age(o.get('updated_utc'))}")
    tiles("forecast", [
        ("Stops", str(n), DEFS["sims"], f"{pct(p_stops.get(str(n)))} chance it's best" if p_stops else None),
        ("Pit lap", " · ".join(str(w["recommended"]) for w in wins) or "—", DEFS["pit_window"],
         " · ".join(f"{w['lo']}–{w['hi']}" for w in wins) or None),
        _plan_b(o),
        ("Pit loss", fmt(o.get("pit_loss_s"), 1, " s"), DEFS["pit_loss"], None),
    ])

    life = st_.get("life") or {}
    if life:
        top = max(life, key=lambda c: life[c]["life_laps"])
        with card("forecast-life", f"{top.title()} lasts longest here: ~{life[top]['life_laps']:.0f} laps",
                  tip="Dots: how far each tyre goes with the plan's tyre saving. Bars: the likely range."):
            chart(_life_range_chart(life))

    ranges = (o.get("sc_playbook") or {}).get("ranges") or []
    if ranges:
        st.caption("If a safety car comes out")
        kind = {"PIT": "alert", "STAY": "info", "MARGINAL": "info", "PLANNED": "ok"}
        badges([(f"Laps {r['from']}–{r['to']}: {verdict_word(r['verdict']).lower()}"
                 + (f" ({r['gain_s']:+.0f} s)" if r["verdict"] in ("PIT", "STAY") else ""),
                 kind.get(r["verdict"], "info"), "") for r in ranges])

    with more():
        c1, c2 = st.columns([1, 1], gap="medium")
        with c1:
            lines = []
            for b in st_.get("by_start") or []:
                lines.append(f"Start on {b['start'].title()}: {label_text(b['label'])} "
                             f"({'best' if b['delta_s'] == 0 else f'+{b['delta_s']:.1f} s'}, best in {b['win_prob_any']:.0%})")
            alts = o.get("alternatives") or {}
            if alts.get("plan_c"):
                c_ = alts["plan_c"]
                lines.append(f"One stop fewer: {label_text(c_['label'])} ({c_['delta_s']:+.1f} s)")
            prog, voi = o.get("programme") or [], o.get("voi") or {}
            if prog:
                lines.append(f"Up to {voi.get('evpi_s', 0):.1f} s rides on the plan choice. Most useful practice runs: "
                             + "; ".join(f"{r['compound'].title()} for {r['target_laps']}+ laps "
                                         f"({r['share']:.0%} of the doubt)" for r in prog[:2]
                                         if r.get("decision_moves") or r.get("share", 0) > 0.2))
            if lines:
                st.markdown("\n".join(f"- {ln}" for ln in lines))
            st.caption("Build, compare and stress-test plans in the Plan builder tab.")
        with c2:
            _outlook_timeline_chart(timeline)
        basis = _forecast_basis(o)
        if basis:
            st.markdown("**What the forecast is built on**")
            st.markdown("\n".join(f"- {s}" for s in basis))
        comb = o.get("combination") or []
        if comb and str(o.get("stage", "")).startswith("prior"):
            st.dataframe(pd.DataFrame([{
                "Tyre": str(r.get("compound")).title(),
                "Past races here (s/lap)": fmt(r.get("history_race"), 3),
                "Tyre order estimate (s/lap)": fmt(r.get("practice_as_race"), 3),
                "Combined (s/lap)": fmt(r.get("combined_race"), 3),
                "Weight on past races": fmt((r.get("weight_on_history") or 0) * 100, 0, "%")} for r in comb]),
                width="stretch", hide_index=True)


def _forecast_basis(o: dict) -> list:
    """What the forecast rests on, from its structured fields (older saved text uses the old wording)."""
    out = []
    r = o.get("regime") or {}
    if finite(r.get("ratio")):
        out.append(f"Race wear vs practice: {r['ratio']:.2f}× (likely {fmt(r.get('p05'), 2)}–{fmt(r.get('p95'), 2)}), "
                   "measured on other 2026 races")
    h, sea = o.get("history") or {}, o.get("season") or {}
    if h:
        out.append(f"Past races at {h.get('circuit', 'this circuit')}: {', '.join(str(y) for y in h.get('years', []))}")
    elif sea.get("n_circuits"):
        out.append(f"No past races here: tyre wear from the 2026 season, pooled over {sea['n_circuits']} circuits "
                   f"({', '.join(sea.get('circuits', []))})")
    if finite(o.get("pit_loss_s")):
        out.append(f"Pit loss {o['pit_loss_s']:.1f} s ({pit_loss_text(o.get('pit_loss_source'))})")
    if o.get("allocation"):
        out.append("Tyre sets: " + ", ".join(f"{str(c).title()} {v}" for c, v in o["allocation"].items()))
    if o.get("n_practice_laps"):
        out.append(f"{o['n_practice_laps']} clean practice laps")
    lv = o.get("live") or {}
    if lv.get("used"):
        out.append(f"Live {lv.get('session_name')}: {lv.get('n_long_runs', 0)} long runs so far, replaced by the "
                   "refit after the session")
    return out


def _outlook_timeline_chart(timeline: list) -> None:
    pts = [t for t in timeline if t.get("p_stops")]
    if len(pts) < 2:
        st.caption("How the forecast moves as data arrives charts itself here from the second update on.")
        return
    x = [datetime.fromisoformat(t["utc"]) for t in pts]
    last = pts[-1]["p_stops"] or {}
    lead = max(last, key=lambda k: float(last[k])) if last else "1"
    with card("forecast-history", f"How the forecast has moved: {len(pts)} updates",
              tip="Chance each stop count is best, at every forecast update."):
        fig = go.Figure()
        keys = sorted({k for t in pts for k in (t["p_stops"] or {})})
        colours = {lead: WHITE, **dict(zip([k for k in keys if k != lead], (DIM, FAINT, FAINT)))}
        for k in keys:
            ys = [float((t["p_stops"] or {}).get(k, 0.0)) for t in pts]
            if max(ys) <= 0:
                continue
            colr = colours[k]
            fig.add_trace(go.Scatter(x=x, y=ys, mode="lines+markers", name=f"{k}-stop",
                                     line=dict(color=colr, width=2), marker=dict(size=7, color=colr, line=dict(color=BLACK, width=1)),
                                     hovertemplate=f"{k}-stop · %{{x|%a %H:%M}}<br>%{{y:.0%}} chance best<extra></extra>"))
            fig.add_annotation(x=x[-1], y=ys[-1], text=f"{k}-stop", showarrow=False, xanchor="left", xshift=6,
                               font=dict(color=WHITE, size=11))
        for t in pts:
            if t.get("stage", "").startswith("sealed") and not t.get("live_used"):
                fig.add_vline(x=datetime.fromisoformat(t["utc"]), line=dict(color=HAIR, width=1))
        fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
        chart(style(fig, 240, "chance best", "", legend=False))


def render_history(meta: dict) -> None:
    """What this circuit has done before."""
    h = meta.get("circuit_history") or {}
    if not h:
        return
    with st.expander(f"Past races at {h.get('circuit', 'this circuit')} ({', '.join(str(y) for y in h.get('years', []))})",
                     icon=":material/flag:"):
        stops = h.get("stops") or {}
        tot = sum(stops.values()) or 1
        items = [(f"{k}-stop", f"{v / tot:.0%}", "Share of finishers who ran this many stops here.", f"{v} drivers")
                 for k, v in stops.items()]
        caps, typ = h.get("stint_cap") or {}, h.get("stint_typical") or {}
        items += [(f"Longest {c.lower()} stint", f"{caps[c]} laps", "The longest this tyre has been run here; "
                   "plans never go past it.", f"typical {typ[c]['p50']:.0f}") for c in LADDER if c in caps and c in typ]
        if h.get("pit_loss_s"):
            items.append(("Pit loss here", f"{h['pit_loss_s']:.1f} s", DEFS["pit_loss"]))
        th = h.get("thermal") or {}
        if th.get("track_temp_now") is not None and th.get("track_temp_hist") is not None:
            items.append(("Track temperature", f"{th['track_temp_now']:.0f} °C",
                          "Hotter track, faster tyre wear.", f"{th['track_temp_hist']:.0f} °C back then · wear ×{th['multiplier']:.2f}"))
        tiles("history", items)
        plans = h.get("plans") or {}
        if plans:
            st.caption("Plans the finishers ran: " + ", ".join(f"{label_text(k)} ×{v}" for k, v in plans.items())
                       + ("" if h.get("soft_race_tyre", True) else " · the soft has not been a race tyre here"))
        comb = meta.get("history_combination") or []
        if comb:
            st.dataframe(pd.DataFrame([{
                "Tyre": str(r["compound"]).title(),
                "Practice, in race trim (s/lap)": fmt(r.get("practice_as_race"), 3),
                "Past races here (s/lap)": fmt(r.get("history_race"), 3),
                "Combined (s/lap)": fmt(r.get("combined_race"), 3),
                "Weight on past races": fmt((r.get("weight_on_history") or 0) * 100, 0, "%")} for r in comb]),
                width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# Live session
# --------------------------------------------------------------------------


def render_session(sk: str, every: int = 3, event_key: str | None = None) -> None:
    """The live view for one session directory, auto-refreshing."""

    @st.fragment(run_every=f"{every}s")
    def body():
        snap = read_snapshot(sk)
        status = read_status(sk)
        if not snap:
            notice("Waiting for the first data…")
            return
        _render_header(snap, status)
        engine = snap.get("engine")
        if engine == "race":
            _render_race(snap, sk, event_key=event_key)
        elif engine == "practice":
            _render_practice(snap)
        else:
            st.json(snap.get("meta", {}))

    body()


def _render_header(snap: dict, status: dict) -> None:
    meta = snap.get("meta", {})
    sess = meta.get("session", {}) or {}
    lc = meta.get("lap_count", {}) or {}
    weather = meta.get("weather") or {}
    track, kind = TRACK.get(str(meta.get("track_status", "1")), (str(meta.get("track_status")), "info"))
    total = meta.get("total_laps") or lc.get("total")
    items = [
        (f"{sess.get('meeting') or ''} · {sess.get('Name') or ''}".strip(" ·"), "info"),
        (track, kind),
        (f"Lap {lc.get('current') or '—'}" + (f" / {total}" if total else ""), "info"),
        (f"Track {fmt(weather.get('TrackTemp'), 0, ' °C')} · air {fmt(weather.get('AirTemp'), 0, ' °C')}"
         + (" · rain" if weather.get("Rainfall") else ""), "info"),
        (f"Updated {age(meta.get('tick_utc'))}", "info"),
    ]
    if snap.get("engine") == "race":
        items.append((f"Pit loss {fmt(meta.get('pit_loss_s'), 1, ' s')}", "info"))
        rm = meta.get("regime_multiplier")
        if rm:
            items.append((f"Race wear vs practice {rm['mean']:.2f}×", "info"))
    badges(items)


def _plan_status(r: dict, p: dict, lap_now: int) -> dict:
    """One car against its committed plan, for the tracker."""
    # A new set of tyres is what the plan counts, so the stint index (from
    # TimingAppData) is the count; the pit-lane counter is the fallback.
    n_stops_done = int(r["stint"]) - 1 if r.get("stint") else int(r.get("n_pit_stops") or 0)
    idx, nxt = plan_store.next_stop(p, lap_now)
    wins = {w["stop"]: w for w in p.get("windows", [])}
    comps = p.get("compounds", [])
    planned_c = comps[min(n_stops_done, len(comps) - 1)] if comps else None
    on_c = r.get("compound")
    n_planned = len(p.get("pit_laps", []))
    level, status = "good", "on plan"
    if n_stops_done > n_planned:
        level, status = "warn", f"off plan: {n_stops_done} stops made, {n_planned} planned"
    elif nxt is None:
        status = "final stint, to the flag"
    else:
        w = wins.get((idx or 0) + 1)
        if w and w["lo"] <= lap_now <= w["hi"]:
            level, status = "accent", f"window open: stop {idx + 1} due lap {nxt} ({w['lo']}–{w['hi']})"
        elif w and lap_now > w["hi"]:
            level, status = "bad", f"stop {idx + 1} overdue: window closed lap {w['hi']}"
        else:
            status = f"stop {idx + 1} in {nxt - lap_now} laps (lap {nxt})"
    if planned_c and on_c and on_c != planned_c and n_stops_done <= n_planned:
        level = "warn" if level == "good" else level
        status += f" · on {str(on_c).title()}, plan said {str(planned_c).title()}"
    alt = p.get("alternative") or {}
    m = r.get("m_mean")
    trig = alt.get("switch_mult")
    switch = ""
    if trig and m is not None and np.isfinite(m):
        if m >= trig:
            level, switch = "warn", f"wear ×{m:.2f} is past the ×{trig:.2f} trigger: consider {label_text(alt.get('label'))}"
        else:
            switch = f"wear ×{m:.2f}, switch at ×{trig:.2f}"
    return {"level": level, "status": status, "next_stop": nxt, "stop_idx": idx, "switch": switch,
            "window": wins.get((idx or 0) + 1) if nxt is not None else None}


FIELD_KEY = ["Pos", "Driver", "Tyre", "Life used", "Laps left", "Plan from here", "Window", "Alert"]
FIELD_COLS = {
    "Tyre": st.column_config.TextColumn(help="Tyre and how many laps it has done."),
    "Life used": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent", color="#FFFFFF",
                                                 help=DEFS["life_used"]),
    "Laps left": st.column_config.TextColumn(help="Laps before the drop-off at the current rate of wear."),
    "Plan from here": st.column_config.TextColumn(help="The best remaining plan for this car from now on."),
    "Window": st.column_config.TextColumn(help=DEFS["pit_window"]),
    "Wear (s/lap)": st.column_config.TextColumn(help="Pace the tyre is expected to lose on its next lap."),
    "Box now costs": st.column_config.TextColumn(help="Race time lost by pitting at the end of this lap instead "
                                                      "of following the best plan."),
    "Drop-off risk": st.column_config.NumberColumn(format="percent", help=DEFS["drop_off"]),
    "Undercut threat": st.column_config.TextColumn(help="The car behind, and its chance of being ahead three laps "
                                                        "after pitting now."),
    "Undercut chance": st.column_config.TextColumn(help="Your chance of being ahead of the car in front three laps "
                                                        "after pitting now."),
}


def _field_rows(field: list) -> pd.DataFrame:
    rows = []
    for r in field:
        p = r.get("plan") or {}
        uc = r.get("undercut") or {}
        th, op = uc.get("threat") or {}, uc.get("opportunity") or {}
        alert = ("⚠ drop-off" if r.get("cliff_alarm") else
                 ("⚠ undercut threat" if th and th.get("p_undercut_3lap", 0) >= 0.5 else ""))
        rows.append({
            "Pos": r.get("position"), "Driver": r.get("driver"),
            "Tyre": f"{str(r.get('compound') or '—').title()} {r.get('tyre_age') if r.get('tyre_age') is not None else ''}".strip(),
            "Life used": float(min(r["wear"], 1.0)) if finite(r.get("wear")) else None,
            "Laps left": (f"{r['laps_to_cliff_p50']:.0f}" if finite(r.get("laps_to_cliff_p50")) and r["laps_to_cliff_p50"] < 60
                          else "60+"),
            "Plan from here": label_text(p.get("best")) if p.get("best") else ("in pit" if r.get("in_pit") else "—"),
            "Window": f"{p['window_lo']}–{p['window_hi']}" if p.get("window_lo") is not None else "—",
            "Alert": alert,
            "Gap": "" if r.get("position") == 1 else (r.get("gap_leader") or ""),
            "Interval": "" if r.get("position") == 1 else (r.get("interval") or ""),
            "Last lap": fmt(r.get("last_lap_s"), 3),
            "Wear (s/lap)": fmt(r.get("deg_now_s_per_lap"), 3),
            "Box now costs": fmt(p.get("delta_box_now_s"), 1, " s"),
            "Drop-off risk": float(r.get("p_past_cliff") or 0),
            "Undercut threat": f"{th['driver']} {th['p_undercut_3lap']:.0%}" if th else "",
            "Undercut chance": f"{op['driver']} {op['p_undercut_3lap']:.0%}" if op else "",
        })
    return pd.DataFrame(rows)


def _render_race(snap: dict, sk: str, event_key: str | None = None) -> None:
    field = snap.get("field", [])
    alerts = snap.get("alerts", [])
    plans = plan_store.load_plans(event_key) if event_key else []
    lap_meta = (snap.get("meta", {}).get("lap_count") or {}).get("current") or 0

    cols = st.segmented_control("Columns", ["Key", "All"], default="Key", required=True, key=f"live_cols_{sk}",
                                label_visibility="collapsed")
    c1, c2 = st.columns([2.3, 1], gap="medium")
    with c1:
        df = _field_rows(field)
        with card(f"live-field-{sk}", "Field"):
            st.dataframe(df[FIELD_KEY] if cols == "Key" and not df.empty else df, width="stretch", hide_index=True,
                         height=min(60 + 36 * len(df), 820), column_config=FIELD_COLS)
    with c2:
        with card(f"live-alerts-{sk}", "Latest alerts"):
            if not alerts:
                st.caption("None yet.")
            for a in reversed(alerts[-6:]):
                st.markdown(f"{ALERT_ICON.get(a.get('level'), '●')} **L{a.get('lap')} {a.get('driver')}** · {a.get('text')}")

    if plans:
        items, rows = [], []
        for r in field:
            p_ = plan_store.plan_for_driver(plans, r.get("driver"))
            if p_ is None or (p_.get("driver") is None and r.get("position") is not None and r["position"] > 10
                              and len(plans) > 1):
                continue        # the team default only for the cars nobody has committed for, kept short
            s = _plan_status(r, p_, int(r.get("current_lap") or lap_meta))
            items.append((f"{r.get('driver')}: {s['status']}", LEVEL_KIND.get(s["level"], "info")))
            rows.append({"Driver": r.get("driver"), "Plan": label_text(p_.get("label")), "For": p_.get("driver") or "team",
                         "Next stop": f"lap {s['next_stop']}" if s["next_stop"] else "—",
                         "Window": f"{s['window']['lo']}–{s['window']['hi']}" if s.get("window") else "—",
                         "Status": s["status"], "Wear vs trigger": s["switch"],
                         "Model's plan now": label_text((r.get("plan") or {}).get("best"))})
        if items:
            st.caption("Committed plans against the race")
            badges(items)
            with more("Committed plans in detail"):
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    drivers = [r["driver"] for r in field]
    if not drivers:
        return
    c1, c2 = st.columns([1, 3])
    with c1:
        drv = st.selectbox("Driver", drivers, key=f"live_drv_{sk}")
    r = next((x for x in field if x["driver"] == drv), None)
    if r is None:
        return
    p = r.get("plan") or {}
    p_committed = plan_store.plan_for_driver(plans, drv) if plans else None
    with c2:
        if p_committed:
            s = _plan_status(r, p_committed, int(r.get("current_lap") or lap_meta))
            trig = p_committed.get("triggers") or {}
            extra = trig.get(f"stop_{(s['stop_idx'] or 0) + 1}", "")
            notice(f"<b>{label_text(p_committed.get('label'))}</b> ({p_committed.get('driver') or 'team default'}) · "
                   f"{s['status']}" + (f" · {s['switch']}" if s["switch"] else "")
                   + (f"<br>{extra}" if extra else ""), LEVEL_KIND.get(s["level"], "info"))
    rj = r.get("rejoin_if_box_now") or {}
    tiles(f"live-drv-{sk}", [
        ("Tyre", f"{str(r.get('compound') or '—').title()} · {r.get('tyre_age') or '—'} laps", None,
         f"stint {r.get('stint') or '—'}"),
        ("Life used", pct(r.get("wear")), DEFS["life_used"], f"drop-off risk {pct(r.get('p_past_cliff') or 0)}"),
        ("Wear vs forecast", f"{fmt(r.get('m_mean'), 2)}×", DEFS["wear_vs_forecast"],
         f"likely {fmt(r.get('m_lo'), 2)}–{fmt(r.get('m_hi'), 2)}"),
        ("Box now costs", fmt(p.get("delta_box_now_s"), 1, " s"),
         "Race time lost by pitting at the end of this lap instead of following the best plan from here.",
         f"rejoin P{rj.get('position', '?')}" + (f" behind {rj['behind']}" if rj.get("behind") else "")),
    ])

    laps = read_laps(sk)
    if not laps.empty:
        d = laps[laps["driver"] == drv].sort_values("lap_number")
        with card(f"live-laps-{sk}", f"{drv}: best plan from here is {label_text(p.get('best')).lower()}"
                  if p.get("best") else f"{drv}: lap times",
                  tip="Dots are lap times by tyre; the dotted white line is the next laps' forecast from tyre wear "
                      "alone. The shaded band is the pit window."):
            fig = go.Figure()
            for cmp_ in d["compound"].dropna().unique():
                g = d[d["compound"] == cmp_]
                fig.add_trace(go.Scatter(x=g["lap_number"], y=g["lap_time_s"], mode="markers", name=str(cmp_).title(),
                                         marker=dict(size=7, color=rgba(ccol(cmp_), 0.8), line=dict(width=1, color=BLACK)),
                                         hovertemplate=f"{str(cmp_).title()} · lap %{{x}}<br>%{{y:.3f}} s<extra></extra>"))
            proj = r.get("proj") or []
            if proj and r.get("last_lap_s") and r.get("level_s"):
                last = d["lap_number"].max()
                fig.add_trace(go.Scatter(x=[last + j for j in range(1, len(proj) + 1)],
                                         y=[float(r["last_lap_s"]) + v for v in proj], mode="lines+markers",
                                         name="forecast", line=dict(color=WHITE, dash="dot", width=2)))
            for _, pr in d[d["pit_in"]].iterrows():
                fig.add_vline(x=pr["lap_number"], line=dict(color=HAIR, width=1))
            if p.get("window_lo") is not None:
                fig.add_vrect(x0=p["window_lo"] - 0.5, x1=p["window_hi"] + 0.5, line_width=0,
                              fillcolor=rgba(WHITE, 0.10), layer="below", annotation_text="pit window",
                              annotation_position="top left", annotation_font=dict(color=MUTED, size=11))
            lt = d["lap_time_s"].dropna()
            if len(lt) > 4:
                fig.update_yaxes(range=[float(lt.quantile(0.02)) - 1.0, float(lt.quantile(0.9)) + 3.0])
            chart(style(fig, 360, "lap time (s)", "lap"))

    with more():
        win = p.get("window") or []
        if win:
            w = pd.DataFrame(win)
            with card(f"live-inlap-{sk}", "Cost of each in-lap", tip=DEFS["pit_window"]):
                fig = go.Figure(go.Scatter(x=w["lap"], y=w["loss_s"], mode="lines", line=dict(color=WHITE, width=2),
                                           hovertemplate="in-lap %{x}<br>+%{y:.2f} s<extra></extra>", showlegend=False))
                fig.add_hline(y=1.0, line=dict(color=HAIR, dash="dash", width=1))
                fig.update_yaxes(range=[0, 12])
                chart(style(fig, 260, "time lost vs the best in-lap (s)", "in-lap", legend=False))
        opts = p.get("options") or []
        if opts:
            o = pd.DataFrame(opts)
            st.dataframe(pd.DataFrame({"Plan": o["label"].map(label_text), "Slower by (s)": o["delta_s"].round(1),
                                       "Chance best": o["win_prob"]}),
                         width="stretch", hide_index=True,
                         column_config={"Chance best": st.column_config.NumberColumn(format="percent", help=DEFS["sims"])})
        uc = r.get("undercut") or {}
        for key_, title in (("threat", "Undercut threat from behind"), ("opportunity", "Undercut on the car ahead")):
            u = uc.get(key_)
            if u:
                st.markdown(f"**{title}: {u['driver']}** — {u['gap_s']:.1f} s gap, on a fresh "
                            f"{str(u['new_compound']).title()}: "
                            + ", ".join(f"ahead after {k + 1} lap{'s' if k else ''} {pk:.0%}" for k, pk in enumerate(u["p_by_lap"][:4])))
        hist = read_history(sk)
        if hist:
            rows = []
            for h in hist:
                x = next((f for f in h.get("field", []) if f.get("driver") == drv), None)
                if x:
                    rows.append({"Lap": h["lap"], "Track": TRACK.get(str(h.get("track_status")), (h.get("track_status"), ""))[0],
                                 "Tyre": str(x.get("compound") or "").title(), "Age": x.get("tyre_age"),
                                 "Life used": x.get("wear"), "Drop-off risk": x.get("p_past_cliff"),
                                 "Wear vs forecast": x.get("m_mean"), "Plan": label_text(x.get("plan_best")),
                                 "Window": x.get("plan_window"), "Box now costs (s)": x.get("delta_box_now_s"),
                                 "Alarm": "⚠" if x.get("cliff_alarm") else ""})
            st.markdown("**Lap by lap: what the model said**")
            st.dataframe(pd.DataFrame(rows).round(2), width="stretch", hide_index=True, height=360)


# --------------------------------------------------------------------------
# Practice
# --------------------------------------------------------------------------


def _render_practice(snap: dict) -> None:
    board, pooled, prior, field = snap.get("board", []), snap.get("pooled", {}), snap.get("prior", {}), snap.get("field", [])
    items = [("Long runs", str(len(board)), "Runs of 4+ laps on one set of tyres so far.")]
    for c in LADDER:
        po, pr = pooled.get(c) or {}, prior.get(c) or {}
        if po or pr:
            items.append((c, f"{po['slope_s_per_lap']:+.3f} s/lap" if po else "—",
                          "Tyre wear from this session's long runs, fuel-adjusted. The track getting faster is not yet "
                          "taken out, so live numbers read low; the refit after the session does that.",
                          f"forecast {pr['rate_s_per_lap']:.3f}" if pr else None))
    tiles("practice", items)
    if board:
        with card("practice-runs", f"{len(board)} long runs so far, adjusted for fuel",
                  tip="Each line is one run: lap time against laps on the tyre, with the fuel effect taken out."):
            fig = go.Figure()
            for b in board[:12]:
                col = ccol(b["compound"])
                fig.add_trace(go.Scatter(x=b["ages"], y=b["y"], mode="lines+markers", showlegend=False,
                                         line=dict(color=rgba(col, 0.75), width=1.5), marker=dict(size=5, color=col),
                                         hovertemplate=f"{b['driver']} {str(b['compound']).title()}<br>lap %{{x:.0f}} of the run · %{{y:.2f}} s<extra></extra>"))
                if len(b["ages"]):
                    fig.add_annotation(x=b["ages"][-1], y=b["y"][-1], text=b["driver"], showarrow=False, xanchor="left",
                                       xshift=5, font=dict(color=WHITE, size=10))
            chart(style(fig, 400, "lap time, fuel-adjusted (s)", "laps on the tyre", legend=False))
    else:
        st.caption("No long runs yet.")

    with more():
        if field:
            st.markdown("**Timing**")
            st.dataframe(pd.DataFrame([{"P": r.get("position"), "Driver": r.get("driver"),
                                        "Tyre": str(r.get("compound") or "").title(), "Age": r.get("tyre_age"),
                                        "Best": fmt(r.get("best_lap_s"), 3), "Last": fmt(r.get("last_lap_s"), 3),
                                        "Gap": r.get("gap_leader") or "", "Laps": r.get("laps_complete"),
                                        "Pit": "in" if r.get("in_pit") else ""} for r in field]),
                         width="stretch", hide_index=True, height=min(60 + 36 * len(field), 600))
        if board:
            st.markdown("**Long runs**")
            st.dataframe(pd.DataFrame([{"Driver": b["driver"], "Run": b["stint"], "Tyre": str(b["compound"]).title(),
                                        "Laps": b["n_laps"], "Tyre age": f"{b['age_from']:.0f}–{b['age_to']:.0f}",
                                        "Wear (s/lap)": f"{b['slope_s_per_lap']:+.3f} ± {b['se']:.3f}",
                                        "Best, fuel-adjusted": fmt(b["best_s"], 3)} for b in board]),
                         width="stretch", hide_index=True, height=min(60 + 36 * len(board), 600))
        rows = [{"Tyre": c.title(),
                 "Live wear (s/lap)": (f"{pooled[c]['slope_s_per_lap']:+.3f} ± {pooled[c]['se']:.3f} "
                                       f"({pooled[c]['n_stints']} runs)" if pooled.get(c) else "—"),
                 "Forecast (s/lap)": (f"{prior[c]['rate_s_per_lap']:.3f} (likely {prior[c]['lo']:.3f}–{prior[c]['hi']:.3f})"
                                      if prior.get(c) else "—")} for c in LADDER]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        runs = snap.get("alerts", [])[-10:]
        if runs:
            st.markdown("**Latest runs**")
            st.markdown("\n".join(f"- L{a.get('lap')} {a.get('text')}" for a in reversed(runs)))


def _sessions() -> list:
    if not LIVE_DIR.exists():
        return []
    out = []
    for p in sorted(LIVE_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if p.is_dir() and (p / "snapshot.json").exists():
            out.append(p.name)
    return out
