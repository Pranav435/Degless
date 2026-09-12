"""The Live tab: what the pit wall sees during a session.

Reads the snapshots the daemon (`scripts/50_live.py`) writes under
`data/live/<session>/` and re-renders every couple of seconds.  The app never
touches the feed or the model; if the daemon dies the last snapshot stays on
screen with its age shown, and the app itself cannot fall over.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.live.store import (
    LIVE_DIR,
    read_current,
    read_history,
    read_laps,
    read_snapshot,
    read_status,
)
from src import plans as plan_store
from src import schedule


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
        st = sup["schedule"]
        st["_supervisor"] = sup
        return st
    st = schedule.status()
    st["_supervisor"] = None
    return st


def render_strip(wk: dict, T: dict, chip, callout) -> None:
    """The one line everybody reads first: what is on, what is next."""
    sup = wk.get("_supervisor")
    live, nxt = wk.get("live"), wk.get("next")
    head = schedule.describe(wk)
    col = T["bad"] if live else T["accent"]
    st_html = (f"<div style='font-size:1.15rem;font-weight:700;color:{col};margin:2px 0 6px 0'>{head}</div>")
    st.markdown(st_html, unsafe_allow_html=True)
    # weekend sessions as chips
    chips = ""
    for s_ in wk.get("weekend", []):
        state = s_["state"]
        c = T["bad"] if state == "live" else (T["muted"] if state == "finished" else T["accent"])
        when = datetime.fromisoformat(s_["start"]).astimezone().strftime("%a %H:%M")
        sub = {"live": "LIVE", "finished": "done", "upcoming": f"in {s_['minutes_to_start']/60:.1f} h"
               if s_["minutes_to_start"] > 90 else f"in {s_['minutes_to_start']:.0f} min"}[state]
        chips += chip(s_["name"], when, c, sub)
    if chips:
        st.markdown(chips, unsafe_allow_html=True)
    if sup is None:
        callout("The supervisor is not running, so nothing starts by itself. Run <code>make run</code> "
                "(dashboard + automatic feed, refits and scoring) instead of <code>make app</code>.", "warn")
    else:
        feed = sup.get("feed")
        tasks = sup.get("tasks") or {}
        bits = []
        if feed:
            bits.append(f"feed {'running' if feed.get('alive') else 'stopped'} for session {feed.get('session')}")
        if tasks:
            bits.append("running: " + ", ".join(tasks))
        if sup.get("history"):
            bits.append("last: " + sup["history"][-1]["text"])
        ol = sup.get("outlook") or {}
        if ol.get("event"):
            bits.append(f"outlook for {ol['event']}: {ol.get('stage') or 'building'}"
                        + (f", updated {_age(ol.get('updated'))}" if ol.get("updated") else "")
                        + (" (refreshing now)" if ol.get("running") else
                           (f", next in {ol['next_in_s'] / 60:.0f} min" if ol.get("next_in_s") is not None else "")))
        st.caption("Supervisor · " + (" · ".join(bits) if bits else "idle, watching the calendar")
                   + f" · updated {sup['_age_s']:.0f} s ago")
        a = sup.get("auth") or {}
        if a.get("status") not in (None, "ok"):
            callout(f"F1TV car telemetry is off: {a.get('detail', 'no valid token')}. Timing, "
                    f"degradation and strategy are unaffected — run <code>make login</code> "
                    f"when you want telemetry back.", "warn")


def render_now(wk: dict, T: dict, ccol, chip, callout, style, rgba, compound_pill, meta: dict, ev,
               outlook: dict | None = None, timeline: list | None = None) -> None:
    """Live view when a session is on; otherwise the countdown and the outlook."""
    live = wk.get("live")
    if live:
        sk = str(live["key"])
        if (LIVE_DIR / sk / "snapshot.json").exists():
            render_session(sk, T, ccol, chip, callout, style, rgba, compound_pill, event_key=live.get("event_key"))
        else:
            st.info(f"{live['label']} is live; waiting for the feed's first snapshot "
                    f"(the supervisor starts it automatically; otherwise `make live EVENT={live.get('event_key')}`).")
        return
    nxt = wk.get("next")
    if nxt:
        start_local = datetime.fromisoformat(nxt["start"]).astimezone().strftime("%A %H:%M")
        callout(f"<b>Nothing is on track.</b> Next up: {nxt['label']} at {start_local} local. "
                "The feed starts by itself 15 minutes before; this page switches to the live view then.", "accent")
    if outlook:
        render_outlook(outlook, timeline or [], ev, T, ccol, chip, callout, style, rgba)
    plan = (meta.get("strategy") or {}).get("best_plan") or {}
    if plan and not outlook:
        wins = (meta.get("strategy") or {}).get("pit_windows") or []
        life = (meta.get("strategy") or {}).get("life") or []
        st.markdown("#### The plan for the race, from practice so far")
        st.markdown(
            chip("Plan", f"{plan.get('n_stops', '—')}-stop " + "-".join(c[0] for c in plan.get("compounds", [])),
                 T["good"], " · ".join(f"{c} {L}" for c, L in zip(plan.get("compounds", []), plan.get("stint_lens", []))))
            + "".join(chip(f"Stop {w['stop']}", f"lap {w['recommended']}", T["accent"], f"window {w['lo']}–{w['hi']}") for w in wins)
            + "".join(chip(r["compound"],
                           (f"≤ {int(r['max_stint_laps'])} laps" if np.isfinite(r.get("max_stint_laps", float("nan")))
                            and r["life_laps"] > r["max_stint_laps"] else f"{r['life_laps']:.0f} laps"),
                           ccol(r["compound"]),
                           (f"stint cap binds · {r['deg_s_per_lap']:.3f} s/lap" if np.isfinite(r.get("max_stint_laps", float("nan")))
                            and r["life_laps"] > r["max_stint_laps"] else f"managed life · {r['deg_s_per_lap']:.3f} s/lap"))
                      for r in life),
            unsafe_allow_html=True)
        st.caption(f"Fitted on {', '.join(meta.get('sessions_used', []) or ['practice'])}. Details in the Race plan and Tyre model tabs. "
                   "Rehearse the live view any time: `make replay EVENT=hungary-2026 DIR=data/raw/livetiming/2026_hungary_race`.")
    render_history(meta, T, ccol, chip, callout)
    # replays and past sessions, out of the way
    sessions = _sessions()
    if sessions:
        with st.expander("Open a recorded or replayed session"):
            sk = st.selectbox("Session", sessions, key="now_replay")
            render_session(sk, T, ccol, chip, callout, style, rgba, compound_pill, event_key=ev.key)


# --------------------------------------------------------------------------
# The outlook: the next race, from everything known so far
# --------------------------------------------------------------------------


def render_outlook(o: dict, timeline: list, ev, T: dict, ccol, chip, callout, style, rgba) -> None:
    st_ = o.get("strategy") or {}
    st.markdown(f"#### Outlook for {o.get('event_name', ev.name)} — {o.get('stage_label', '')}")
    live = o.get("live") or {}
    st.caption(f"Updated {_age(o.get('updated_utc'))} · {o.get('n_practice_laps', 0)} clean practice laps in the fit"
               + (f" · live {live.get('session_name')} board folded in ({live.get('n_long_runs', 0)} long runs)"
                  if live.get("used") else "")
               + (f" · track {live['track_temp_c']:.0f} °C" if live.get("track_temp_c") is not None else "")
               + f" · {st_.get('n_strategies', 0):,} plans searched on {o.get('n_draws', 0)} draws"
               + " · refreshes by itself as data arrives")
    if not st_:
        callout("No legal plan came out of the search for this weekend yet.", "warn")
        return
    best = st_.get("best_plan") or {}
    p_stops = st_.get("p_stops") or {}
    top_k = max(p_stops, key=p_stops.get) if p_stops else None
    life = st_.get("life") or {}
    order = [c for c in ("SOFT", "MEDIUM", "HARD") if c in life]
    sc = o.get("scenarios") or {}
    chips = (
        chip("Plan", st_.get("best", "—"), T["good"],
             " · ".join(f"{c} {L}" for c, L in zip(best.get("compounds", []), best.get("stint_lens", [])))
             + f" · push {st_.get('push', float('nan')):.2f}")
        + "".join(chip(f"{k}-stop", f"{v:.0%}", T["accent"] if k == top_k else T["muted"],
                       "of posterior draws" if k == top_k else "") for k, v in p_stops.items())
        + "".join(chip(f"Stop {w['stop']}", f"lap {w['recommended']}", T["accent"], f"window {w['lo']}–{w['hi']}")
                  for w in st_.get("pit_windows", []))
        + "".join(chip(c, f"{life[c]['life_laps']:.0f} laps", ccol(c),
                       f"{life[c]['life_lo']:.0f}–{life[c]['life_hi']:.0f} · {life[c]['deg_s_per_lap']:.3f} s/lap")
                  for c in order)
        + chip("Pit loss", f"{o.get('pit_loss_s', float('nan')):.1f} s", T["muted"], (o.get("pit_loss_source") or "")[:40])
        + (chip("Robust plan", sc.get("robust", "—"), T["warn"] if sc.get("robust") != st_.get("best") else T["good"],
                f"worst case {sc.get('robust_max_regret_s', float('nan')):.1f} s across 9 scenarios") if sc else "")
    )
    st.markdown(chips, unsafe_allow_html=True)
    srcs = o.get("sources") or []
    with st.expander("What this is built on"):
        for s in srcs:
            st.markdown(f"- {s}")
        comb = o.get("combination") or []
        if comb and o.get("stage", "").startswith("prior"):
            rows = [{"Tyre": r.get("compound"), "History, race regime (s/lap)": _fmt(r.get("history_race"), 3),
                     "Ladder prior, as race (s/lap)": _fmt(r.get("practice_as_race"), 3),
                     "Combined, race regime (s/lap)": _fmt(r.get("combined_race"), 3),
                     "Weight on history": _fmt((r.get("weight_on_history") or 0) * 100, 0, "%")} for r in comb]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("Before any practice, the compound ladder and the race history are combined into one "
                       "circuit-severity scale (so the ladder ordering cannot invert). The sealed fit after FP1 "
                       "replaces the ladder side with measured long runs.")
        lv = (o.get("live") or {}).get("rows") or []
        if lv:
            st.dataframe(pd.DataFrame(lv), width="stretch", hide_index=True)
    # what would move it, and the alternatives
    c1, c2 = st.columns([1.2, 1])
    with c1:
        alts = o.get("alternatives") or {}
        lines = []
        for name, alt in alts.items():
            tag = "Plan B (one more stop)" if name == "plan_b" else "Plan C (one fewer stop)"
            sw = alt.get("switch_mult")
            lines.append(f"**{tag}: {alt['label']}** — {alt['delta_s']:+.1f} s in expectation"
                         + (f"; becomes faster if degradation runs ×{sw:.2f} the outlook's rate"
                            f" ({np.log(sw) / max((o.get('thermal') or {}).get('beta_per_c', 0.025), 1e-6):+.0f} °C of track)"
                            if sw and alt.get("switch_direction") == "up" else
                            (f"; is the faster plan already (by {-alt['delta_s']:.1f} s)" if alt["delta_s"] < 0 else
                             "; does not overtake within ×2.6 degradation")))
        bs = st_.get("by_start") or []
        if bs:
            lines.append("**By starting tyre:** " + "; ".join(
                f"{b['start']} → {b['label']} ({'best' if b['delta_s'] == 0 else f'+{b['delta_s']:.1f} s'}, "
                f"{b['win_prob_any']:.0%})" for b in bs))
        prog = o.get("programme") or []
        if prog:
            voi = o.get("voi") or {}
            lines.append(f"**What would settle it** (decision uncertainty {voi.get('evpi_s', 0):.1f} s): "
                         + " ".join(r["text"].split(" - ")[0] + f" ({r['share']:.0%})." for r in prog[:2]
                                    if r.get("decision_moves") or r["share"] > 0.2))
        pb = (o.get("sc_playbook") or {}).get("ranges") or []
        if pb:
            lines.append("**Safety car:** " + "; ".join(
                f"laps {r['from']}–{r['to']} {r['verdict'].lower()}" + (f" ({r['gain_s']:+.0f} s)" if r["verdict"] in ("PIT", "STAY") else "")
                for r in pb))
        for ln in lines:
            st.markdown(ln)
        st.caption("Build, compare and stress-test plans, run the undercut calculator and commit the decision card in "
                   "the **Strategy desk** tab.")
    with c2:
        _outlook_timeline_chart(timeline, T, ccol, style, rgba)


def _outlook_timeline_chart(timeline: list, T: dict, ccol, style, rgba) -> None:
    pts = [t for t in timeline if t.get("p_stops")]
    if len(pts) < 2:
        st.caption("How the outlook moves as data arrives will chart itself here from the second build on.")
        return
    x = [datetime.fromisoformat(t["utc"]) for t in pts]
    fig = go.Figure()
    shades = {1: 0.45, 2: 0.75, 3: 1.0}
    for k in ("1", "2", "3"):
        ys = [float((t["p_stops"] or {}).get(k, 0.0)) for t in pts]
        if max(ys) <= 0:
            continue
        fig.add_trace(go.Scatter(x=x, y=ys, mode="lines+markers", name=f"{k}-stop",
                                 line=dict(color=rgba(T["accent"], shades[int(k)]), width=2),
                                 marker=dict(size=7, line=dict(color=T["surface"], width=1)),
                                 hovertemplate=f"{k}-stop · %{{x|%a %H:%M}}<br>P %{{y:.0%}}<extra></extra>"))
    for t in pts:
        if t.get("stage", "").startswith("sealed") and not t.get("live_used"):
            fig.add_vline(x=datetime.fromisoformat(t["utc"]), line=dict(color=T["hairline"], width=1))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    st.plotly_chart(style(fig, 230, "P(best plan has k stops)", ""), width="stretch")
    fig2 = go.Figure()
    for c in ("SOFT", "MEDIUM", "HARD"):
        ys = [(t.get("life") or {}).get(c, {}).get("mean") for t in pts]
        if not any(v is not None for v in ys):
            continue
        lo = [(t.get("life") or {}).get(c, {}).get("lo") for t in pts]
        hi = [(t.get("life") or {}).get(c, {}).get("hi") for t in pts]
        fig2.add_trace(go.Scatter(x=x + x[::-1], y=hi + lo[::-1], fill="toself", fillcolor=rgba(ccol(c), 0.12),
                                  line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig2.add_trace(go.Scatter(x=x, y=ys, mode="lines+markers", name=c, line=dict(color=ccol(c), width=2),
                                  marker=dict(size=6, line=dict(color=T["surface"], width=1)),
                                  hovertemplate=f"{c} · %{{x|%a %H:%M}}<br>life %{{y:.0f}} laps<extra></extra>"))
    st.plotly_chart(style(fig2, 230, "tyre life at the plan's push (laps)", ""), width="stretch")
    st.caption(f"{len(pts)} builds so far: {pts[0]['stage_label']} → {pts[-1]['stage_label']}. Bands are 90% credible.")


def render_history(meta: dict, T: dict, ccol, chip, callout) -> None:
    """What this circuit has done before, and how much of the model it now carries."""
    h = meta.get("circuit_history") or {}
    if not h:
        return
    st.markdown(f"#### What {h.get('circuit', 'this circuit')} has done before ({', '.join(str(y) for y in h.get('years', []))})")
    stops = h.get("stops") or {}
    tot = sum(stops.values()) or 1
    chips = "".join(chip(f"{k}-stop", f"{v/tot:.0%}", T["accent"], f"{v} finishers") for k, v in stops.items())
    chips += "".join(chip(c, f"cap {h['stint_cap'][c]} laps", ccol(c),
                          f"typical {h['stint_typical'][c]['p50']:.0f}, p90 {h['stint_typical'][c]['p90']:.0f}")
                     for c in ("SOFT", "MEDIUM", "HARD") if c in (h.get("stint_cap") or {}) and c in (h.get("stint_typical") or {}))
    if h.get("pit_loss_s"):
        chips += chip("Pit loss here", f"{h['pit_loss_s']:.1f} s", T["muted"], "measured on this pit lane")
    sea = h.get("season") or {}
    if sea.get("n_circuits"):
        chips += chip("2026 vs 2025 deg", f"{sea['factor']:.2f}×", T["muted"], f"{sea['n_circuits']} shared circuits")
    th = h.get("thermal") or {}
    if th.get("track_temp_now") is not None and th.get("track_temp_hist") is not None:
        chips += chip("Track temp", f"{th['track_temp_now']:.0f}°C", T["warn"] if th["multiplier"] > 1.1 else T["muted"],
                      f"vs {th['track_temp_hist']:.0f}°C then · deg ×{th['multiplier']:.2f}")
    st.markdown(chips, unsafe_allow_html=True)
    plans = h.get("plans") or {}
    if plans:
        st.caption("Plans run by classified finishers: " + ", ".join(f"{k} ×{v}" for k, v in plans.items())
                   + ("" if h.get("soft_race_tyre", True) else " — the SOFT has not been a race tyre here."))
    comb = meta.get("history_combination") or []
    if comb:
        rows = [{"Tyre": r["compound"], "Practice fit, in race trim (s/lap)": _fmt(r.get("practice_as_race"), 3),
                 "History, this circuit (race s/lap)": _fmt(r.get("history_race"), 3),
                 "Combined, race trim (s/lap)": _fmt(r.get("combined_race"), 3),
                 "Weight on history": _fmt((r.get("weight_on_history") or 0) * 100, 0, "%")} for r in comb]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Practice measures the rate; history says how far a tyre is actually taken here and how "
                   "fast it degrades in race trim. The two are combined on the log scale, weighted by their "
                   "uncertainty; the stint caps above are hard bounds in every plan search.")


def render_session(sk: str, T: dict, ccol, chip, callout, style, rgba, compound_pill, every: int = 3,
                   event_key: str | None = None) -> None:
    """The live view for one session directory, auto-refreshing."""

    @st.fragment(run_every=f"{every}s")
    def body():
        snap = read_snapshot(sk)
        status = read_status(sk)
        if not snap:
            st.info("Waiting for the first snapshot…")
            return
        _render_header(snap, status, T, chip)
        engine = snap.get("engine")
        if engine == "race":
            _render_race(snap, sk, T, ccol, chip, callout, style, rgba, compound_pill, event_key=event_key)
        elif engine == "practice":
            _render_practice(snap, sk, T, ccol, chip, callout, style, rgba, compound_pill)
        else:
            st.json(snap.get("meta", {}))

    body()


def _render_header(snap, status, T, chip):
    meta = snap.get("meta", {})
    engine = snap.get("engine")
    sess = meta.get("session", {}) or {}
    lc = meta.get("lap_count", {}) or {}
    ts = meta.get("track_status", "1")
    ts_label = {"1": "GREEN", "2": "YELLOW", "4": "SAFETY CAR", "5": "RED", "6": "VSC", "7": "VSC ENDING"}.get(ts, ts)
    ts_col = {"1": T["good"], "2": T["warn"], "4": T["bad"], "5": T["bad"], "6": T["warn"], "7": T["warn"]}.get(ts, T["muted"])
    st.markdown(
        chip("Session", f"{sess.get('meeting') or ''} · {sess.get('Name') or ''}".strip(" ·"), T["accent"],
             f"{meta.get('status') or '—'} · feed {status.get('source', '?')}"
             + _auth_note(status) + f" · {_age(meta.get('tick_utc'))}")
        + chip("Track", ts_label, ts_col, f"lap {lc.get('current') or '—'} / {meta.get('total_laps') or lc.get('total') or '—'}")
        + (chip("Pit loss", _fmt(meta.get("pit_loss_s"), 1, " s"), T["muted"], meta.get("pit_loss_source", "")[:48])
           if engine == "race" else "")
        + (chip("Race deg vs practice", f"{meta['regime_multiplier']['mean']:.2f}×", T["accent"],
                f"90% {meta['regime_multiplier']['p05']:.2f}–{meta['regime_multiplier']['p95']:.2f} · prior {meta['regime_multiplier']['prior_mean']:.2f}")
           if engine == "race" and meta.get("regime_multiplier") else "")
        + chip("Weather", f"{_fmt((meta.get('weather') or {}).get('TrackTemp'), 0, '°C')} track",
               T["muted"], f"air {_fmt((meta.get('weather') or {}).get('AirTemp'), 0, '°C')}"
               + (" · RAIN" if (meta.get('weather') or {}).get('Rainfall') else "")),
        unsafe_allow_html=True)
    st.caption(f"Model: {meta.get('model_source', '—')}. Messages {status.get('messages', 0):,}, laps {status.get('n_laps', 0)}.")


def _fmt(x, nd=1, suffix=""):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}{suffix}"


def _age(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        s = (datetime.now(timezone.utc) - t).total_seconds()
        return f"{s:.0f} s ago" if s < 120 else f"{s/60:.0f} min ago"
    except Exception:
        return "—"


def _auth_note(status: dict) -> str:
    """What the F1TV login is doing, including when it has quietly lapsed.

    The token only unlocks car telemetry, so an expired one degrades the feed
    rather than stopping it — which is exactly why it has to be shown.
    """
    if status.get("auth"):
        return " · F1TV auth"
    return {"expired": " · F1TV token expired — make login",
            "invalid": " · F1TV token invalid — make login",
            "none": " · no F1TV login"}.get(status.get("auth_status"), "")


def _sessions() -> list:
    if not LIVE_DIR.exists():
        return []
    out = []
    for p in sorted(LIVE_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if p.is_dir() and (p / "snapshot.json").exists():
            out.append(p.name)
    return out


def render_live(T: dict, ccol, chip, callout, style, rgba, compound_pill) -> None:
    sessions = _sessions()
    if not sessions:
        callout("No live session yet. Start the daemon: "
                "<code>.venv/bin/python scripts/50_live.py --event italy-2026</code> "
                "(or <code>--source recorded --session-dir data/raw/livetiming/2026_hungary_race --speed 20</code> "
                "to replay a race at 20x).", "warn")
        return
    cur = read_current()
    default = sessions.index(cur) if cur in sessions else 0
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sk = st.selectbox("Session", sessions, index=default, key="live_session")
    with c2:
        auto = st.toggle("Auto-refresh", value=True, key="live_auto")
    with c3:
        every = st.select_slider("Every", options=[2, 5, 10, 30], value=5, key="live_every")

    @st.fragment(run_every=(f"{every}s" if auto else None))
    def body():
        snap = read_snapshot(sk)
        status = read_status(sk)
        if not snap:
            st.info("Waiting for the first snapshot…")
            return
        meta = snap.get("meta", {})
        engine = snap.get("engine")
        sess = meta.get("session", {}) or {}
        lc = meta.get("lap_count", {}) or {}
        ts = meta.get("track_status", "1")
        ts_label = {"1": "GREEN", "2": "YELLOW", "4": "SAFETY CAR", "5": "RED", "6": "VSC", "7": "VSC ENDING"}.get(ts, ts)
        ts_col = {"1": T["good"], "2": T["warn"], "4": T["bad"], "5": T["bad"], "6": T["warn"], "7": T["warn"]}.get(ts, T["muted"])
        st.markdown(
            chip("Session", f"{sess.get('meeting') or ''} · {sess.get('Name') or ''}".strip(" ·"), T["accent"],
                 f"{meta.get('status') or '—'} · feed {status.get('source', '?')}"
                 + _auth_note(status) + f" · {_age(meta.get('tick_utc'))}")
            + chip("Track", ts_label, ts_col, f"lap {lc.get('current') or '—'} / {meta.get('total_laps') or lc.get('total') or '—'}")
            + (chip("Pit loss", _fmt(meta.get("pit_loss_s"), 1, " s"), T["muted"], meta.get("pit_loss_source", "")[:48])
               if engine == "race" else "")
            + (chip("Race deg vs practice", f"{meta['regime_multiplier']['mean']:.2f}×", T["accent"],
                    f"90% {meta['regime_multiplier']['p05']:.2f}–{meta['regime_multiplier']['p95']:.2f} · prior {meta['regime_multiplier']['prior_mean']:.2f}")
               if engine == "race" and meta.get("regime_multiplier") else "")
            + chip("Weather", f"{_fmt((meta.get('weather') or {}).get('TrackTemp'), 0, '°C')} track",
                   T["muted"], f"air {_fmt((meta.get('weather') or {}).get('AirTemp'), 0, '°C')}"
                   + (" · RAIN" if (meta.get('weather') or {}).get('Rainfall') else "")),
            unsafe_allow_html=True)
        st.caption(f"Model: {meta.get('model_source', '—')}. Messages {status.get('messages', 0):,}, "
                   f"laps {status.get('n_laps', 0)}.")

        if engine == "race":
            _render_race(snap, sk, T, ccol, chip, callout, style, rgba, compound_pill)
        elif engine == "practice":
            _render_practice(snap, sk, T, ccol, chip, callout, style, rgba, compound_pill)
        else:
            st.json(meta)

    body()


# --------------------------------------------------------------------------
# Race
# --------------------------------------------------------------------------


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
    level, status = "good", "on plan"
    if n_stops_done > len(p.get("pit_laps", [])):
        level, status = "warn", f"{n_stops_done} stops made, plan had {len(p.get('pit_laps', []))}: off plan"
    elif nxt is None:
        status = "final stint: to the flag"
    else:
        w = wins.get((idx or 0) + 1)
        if w and w["lo"] <= lap_now <= w["hi"]:
            level, status = "accent", f"WINDOW OPEN for stop {idx + 1} (plan lap {nxt}, window {w['lo']}–{w['hi']})"
        elif w and lap_now > w["hi"]:
            level, status = "bad", f"stop {idx + 1} OVERDUE: window closed at lap {w['hi']}"
        else:
            status = f"stop {idx + 1} on lap {nxt}: {nxt - lap_now} laps away" + (f" (window {w['lo']}–{w['hi']})" if w else "")
    if planned_c and on_c and on_c != planned_c and n_stops_done <= len(p.get("pit_laps", [])):
        level = "warn" if level == "good" else level
        status += f" · on {on_c}, plan said {planned_c}"
    alt = p.get("alternative") or {}
    m = r.get("m_mean")
    trig = alt.get("switch_mult")
    switch = ""
    if trig and m is not None and np.isfinite(m):
        if m >= trig:
            level, switch = "warn", f"deg ×{m:.2f} ≥ trigger ×{trig:.2f}: consider {alt.get('label')}"
        else:
            switch = f"deg ×{m:.2f} < trigger ×{trig:.2f}"
    return {"level": level, "status": status, "next_stop": nxt, "stop_idx": idx, "switch": switch,
            "window": wins.get((idx or 0) + 1) if nxt is not None else None}


def _render_race(snap, sk, T, ccol, chip, callout, style, rgba, compound_pill, event_key: str | None = None):
    field = snap.get("field", [])
    alerts = snap.get("alerts", [])
    plans = plan_store.load_plans(event_key) if event_key else []
    rows = []
    for r in field:
        p = r.get("plan") or {}
        uc = r.get("undercut") or {}
        th, op = uc.get("threat") or {}, uc.get("opportunity") or {}
        rows.append({
            "P": r.get("position"), "Drv": r.get("driver"), "Tyre": r.get("compound"),
            "Age": r.get("tyre_age"),
            "Gap": ("" if r.get("position") == 1 else (r.get("gap_leader") or "")),
            "Int": ("" if r.get("position") == 1 else (r.get("interval") or "")),
            "Last": _fmt(r.get("last_lap_s"), 3), "Deg s/lap": _fmt(r.get("deg_now_s_per_lap"), 3),
            "Wear": _fmt(r.get("wear"), 2), "P(cliff)": _fmt((r.get("p_past_cliff") or 0) * 100, 0, "%"),
            "Laps left": (f"{r['laps_to_cliff_p50']:.0f} ({r['laps_to_cliff_p10']:.0f}–{r['laps_to_cliff_p90']:.0f})"
                          if r.get("laps_to_cliff_p50") is not None and r["laps_to_cliff_p50"] < 60 else ">60"),
            "Plan": p.get("best") or ("—" if not r.get("in_pit") else "in pit"),
            "Window": (f"{p['window_lo']}–{p['window_hi']}" if p.get("window_lo") is not None else "—"),
            "Box now Δ": _fmt(p.get("delta_box_now_s"), 1, " s"),
            "Threat": (f"{th['driver']} {th['p_undercut_3lap']:.0%}" if th else ""),
            "Undercut": (f"{op['driver']} {op['p_undercut_3lap']:.0%}" if op else ""),
            "!": ("CLIFF" if r.get("cliff_alarm") else ""),
        })
    df = pd.DataFrame(rows)

    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.markdown("#### Field")
        st.dataframe(df, width="stretch", hide_index=True, height=min(60 + 36 * len(df), 820))
        st.caption("Deg is the tyre's expected pace loss on its next lap; Wear is the share of the grip budget "
                   "spent (1.0 = cliff); Laps left is to the cliff at the current rate (median, 10–90%). "
                   "Plan is the best remaining strategy from now on the car's own posterior; Window is the "
                   "in-lap range within 1 s of optimal; Box now Δ is the cost of pitting at the end of this "
                   "lap versus that plan. Threat/Undercut: probability the car behind jumps you (or you jump "
                   "the car ahead) after three laps if the attacker pits now.")
    with c2:
        st.markdown("#### Alerts")
        if not alerts:
            st.caption("None yet.")
        for a in reversed(alerts[-14:]):
            col = {"bad": T["bad"], "warn": T["warn"], "good": T["good"], "accent": T["accent"]}.get(a.get("level"), T["muted"])
            st.markdown(
                f"<div style='border-left:3px solid {col};padding:6px 10px;margin:0 0 6px 0;"
                f"background:{rgba(col, 0.07)};border-radius:4px;font-size:0.85rem'>"
                f"<b>L{a.get('lap')} {a.get('driver')}</b> · {a.get('text')}</div>",
                unsafe_allow_html=True)

    if plans:
        st.markdown("#### Committed plans against the race")
        rows = []
        for r in field:
            p_ = plan_store.plan_for_driver(plans, r.get("driver"))
            if p_ is None or (p_.get("driver") is None and not any(x.get("driver") is None for x in plans)):
                continue
            if p_.get("driver") is None and r.get("position") is not None and r["position"] > 10 and len(plans) > 1:
                continue        # the team default only for the cars nobody has committed for, kept short
            lap_now = int(r.get("current_lap") or (snap.get("meta", {}).get("lap_count") or {}).get("current") or 0)
            s = _plan_status(r, p_, lap_now)
            eng = r.get("plan") or {}
            rows.append({"Drv": r.get("driver"), "P": r.get("position"), "Plan": p_.get("label"),
                         "For": p_.get("driver") or "team", "Lap": lap_now,
                         "Tyre": f"{r.get('compound') or '—'} {r.get('tyre_age') or '—'}",
                         "Next stop": (f"lap {s['next_stop']}" if s["next_stop"] else "—"),
                         "Window": (f"{s['window']['lo']}–{s['window']['hi']}" if s.get("window") else "—"),
                         "Status": s["status"], "Deg vs trigger": s["switch"] or "",
                         "Engine box-now Δ": _fmt(eng.get("delta_box_now_s"), 1, " s"),
                         "Engine says": eng.get("best") or ""})
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=min(60 + 36 * len(rows), 420))
            st.caption("Each committed car against its decision card: the next planned stop and window, whether "
                       "the car is on the planned compound, the live degradation multiplier against the switch "
                       "trigger, and what the engine would do from here. Commit or change plans in the Strategy desk.")

    st.markdown("---")
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
    if p_committed:
        lap_now = int(r.get("current_lap") or (snap.get("meta", {}).get("lap_count") or {}).get("current") or 0)
        s = _plan_status(r, p_committed, lap_now)
        trig = (p_committed.get("triggers") or {})
        stop_line = trig.get(f"stop_{(s['stop_idx'] or 0) + 1}", "")
        callout(f"<b>Committed plan {p_committed.get('label')}</b> ({p_committed.get('driver') or 'team default'}) · "
                f"{s['status']}" + (f" · {s['switch']}" if s["switch"] else "")
                + (f"<br><span style='font-size:0.85rem'>{stop_line}</span>" if stop_line else "")
                + (f"<br><span style='font-size:0.85rem'>{trig.get('safety_car')}</span>" if trig.get("safety_car") else ""),
                s["level"])
    with c2:
        st.markdown(
            chip("Tyre", f"{r.get('compound') or '—'} · {r.get('tyre_age') or '—'} laps", ccol(str(r.get("compound"))),
                 f"stint {r.get('stint')} · {r.get('n_clean', 0)} clean laps used")
            + chip("Wear", _fmt(r.get("wear"), 2), T["bad"] if r.get("cliff_alarm") else T["accent"],
                   f"P(past cliff) {_fmt((r.get('p_past_cliff') or 0) * 100, 0, '%')}")
            + chip("This car's deg", f"{_fmt(r.get('m_mean'), 2)}×", T["accent"],
                   f"of practice rate · 90% {_fmt(r.get('m_lo'), 2)}–{_fmt(r.get('m_hi'), 2)}")
            + chip("Best from here", p.get("best") or "—", T["good"],
                   f"wins {p.get('win_prob', 0) * 100:.0f}% of draws · stay out {_fmt(p.get('delta_stay_out_s'), 1, ' s')}")
            + chip("Box now", _fmt(p.get("delta_box_now_s"), 1, " s"),
                   T["good"] if (p.get("delta_box_now_s") or 9) < 1.0 else T["warn"],
                   (f"rejoin P{(r.get('rejoin_if_box_now') or {}).get('position', '?')} behind "
                    f"{(r.get('rejoin_if_box_now') or {}).get('behind') or '—'}")),
            unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 1])
    laps = read_laps(sk)
    with c1:
        st.markdown("#### Lap times and the model's projection")
        if not laps.empty:
            d = laps[laps["driver"] == drv].sort_values("lap_number")
            fig = go.Figure()
            for cmp_ in d["compound"].dropna().unique():
                g = d[d["compound"] == cmp_]
                fig.add_trace(go.Scatter(x=g["lap_number"], y=g["lap_time_s"], mode="markers", name=str(cmp_),
                                         marker=dict(size=7, color=rgba(ccol(str(cmp_)), 0.8),
                                                     line=dict(width=1, color=T["surface"])),
                                         hovertemplate=f"{cmp_} · lap %{{x}}<br>%{{y:.3f}} s<extra></extra>"))
            proj = r.get("proj") or []
            if proj and r.get("last_lap_s") and r.get("level_s"):
                last = d["lap_number"].max()
                base = float(r["last_lap_s"])
                xs = [last + j for j in range(1, len(proj) + 1)]
                ys = [base + v for v in proj]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="projected (deg only)",
                                         line=dict(color=T["accent"], dash="dot", width=2)))
            for _, pr in d[d["pit_in"]].iterrows():
                fig.add_vline(x=pr["lap_number"], line=dict(color=T["hairline"], width=1))
            if p.get("window_lo") is not None:
                fig.add_vrect(x0=p["window_lo"] - 0.5, x1=p["window_hi"] + 0.5, line_width=0,
                              fillcolor=rgba(T["accent"], 0.10), layer="below",
                              annotation_text="pit window", annotation_position="top left")
            lt = d["lap_time_s"].dropna()
            if len(lt) > 4:
                fig.update_yaxes(range=[float(lt.quantile(0.02)) - 1.0, float(lt.quantile(0.9)) + 3.0])
            st.plotly_chart(style(fig, 380, "lap time (s)", "lap"), width="stretch")
    with c2:
        st.markdown("#### Cost of each in-lap")
        win = p.get("window") or []
        if win:
            w = pd.DataFrame(win)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=w["lap"], y=w["loss_s"], mode="lines", line=dict(color=T["accent"], width=2),
                                     hovertemplate="in-lap %{x}<br>+%{y:.2f} s<extra></extra>", name="loss vs best"))
            fig.add_hline(y=1.0, line=dict(color=T["muted"], dash="dash", width=1))
            fig.update_yaxes(range=[0, 12])
            st.plotly_chart(style(fig, 300, "time lost vs best in-lap (s)", "in-lap", legend=False), width="stretch")
        opts = p.get("options") or []
        if opts:
            st.dataframe(pd.DataFrame(opts).rename(columns={"label": "plan", "delta_s": "Δ s", "win_prob": "P(best)"}).round(2),
                         width="stretch", hide_index=True)
        uc = r.get("undercut") or {}
        for key_, title in (("threat", "Undercut threat from behind"), ("opportunity", "Undercut on the car ahead")):
            u = uc.get(key_)
            if u:
                st.markdown(f"**{title}: {u['driver']}** — gap {u['gap_s']:.1f} s, on a fresh {u['new_compound']}: "
                            + ", ".join(f"after {k+1}: {pk:.0%}" for k, pk in enumerate(u["p_by_lap"][:4])))

    # -- history: what the engine said lap by lap --------------------------
    hist = read_history(sk)
    if hist:
        with st.expander("What the engine said, lap by lap (for scoring after the flag)"):
            rows = []
            for h in hist:
                x = next((f for f in h.get("field", []) if f.get("driver") == drv), None)
                if x:
                    rows.append({"lap": h["lap"], "track": h.get("track_status"), "tyre": x.get("compound"),
                                 "age": x.get("tyre_age"), "wear": x.get("wear"), "P(cliff)": x.get("p_past_cliff"),
                                 "m": x.get("m_mean"), "plan": x.get("plan_best"), "window": x.get("plan_window"),
                                 "box now Δ": x.get("delta_box_now_s"), "alarm": x.get("cliff_alarm")})
            st.dataframe(pd.DataFrame(rows).round(2), width="stretch", hide_index=True, height=400)


# --------------------------------------------------------------------------
# Practice
# --------------------------------------------------------------------------


def _render_practice(snap, sk, T, ccol, chip, callout, style, rgba, compound_pill):
    board = snap.get("board", [])
    pooled = snap.get("pooled", {})
    prior = snap.get("prior", {})
    field = snap.get("field", [])
    if field:
        rows = [{"P": r.get("position"), "Drv": r.get("driver"), "Tyre": r.get("compound"), "Age": r.get("tyre_age"),
                 "Best": _fmt(r.get("best_lap_s"), 3), "Last": _fmt(r.get("last_lap_s"), 3),
                 "Gap": r.get("gap_leader") or "", "Laps": r.get("laps_complete"),
                 "Pit": "in" if r.get("in_pit") else ""} for r in field]
        with st.expander("Timing", expanded=(snap.get("meta", {}).get("session", {}).get("Type") == "Qualifying")):
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=min(60 + 36 * len(rows), 600))
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown("#### Long-run board")
        if board:
            df = pd.DataFrame([{"Drv": b["driver"], "Stint": b["stint"], "Tyre": b["compound"], "Laps": b["n_laps"],
                                "Age": f"{b['age_from']:.0f}–{b['age_to']:.0f}",
                                "Deg s/lap": f"{b['slope_s_per_lap']:+.3f} ± {b['se']:.3f}",
                                "Best fuel-corr": _fmt(b["best_s"], 3)} for b in board])
            st.dataframe(df, width="stretch", hide_index=True, height=min(60 + 36 * len(df), 700))
            st.caption("Fuel-corrected with the 2026 physics prior, in-/out-laps, non-green and traffic laps "
                       "removed, stints of 4+ laps. Not yet corrected for track evolution — the offline fit "
                       "does that; a rubbering-in track makes every live slope read low.")
        else:
            st.caption("No long runs yet.")
    with c2:
        st.markdown("#### Pooled by compound vs the prior")
        rows = []
        for c in ("SOFT", "MEDIUM", "HARD"):
            pr = prior.get(c, {})
            po = pooled.get(c, {})
            rows.append({"Tyre": c,
                         "Live (s/lap)": (f"{po['slope_s_per_lap']:+.3f} ± {po['se']:.3f} ({po['n_stints']} stints)" if po else "—"),
                         "Model prior": (f"{pr['rate_s_per_lap']:.3f} [{pr['lo']:.3f}–{pr['hi']:.3f}]" if pr else "—")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown("#### Runs")
        for a in reversed(snap.get("alerts", [])[-10:]):
            st.markdown(f"- L{a.get('lap')} {a.get('text')}")
    if board:
        st.markdown("#### Long runs, fuel-corrected")
        fig = go.Figure()
        for b in board[:12]:
            col = ccol(b["compound"])
            fig.add_trace(go.Scatter(x=b["ages"], y=b["y"], mode="lines+markers", name=f"{b['driver']} {b['compound']} S{b['stint']}",
                                     line=dict(color=rgba(col, 0.75), width=1.5), marker=dict(size=5, color=col),
                                     hovertemplate=f"{b['driver']} {b['compound']}<br>age %{{x:.0f}} · %{{y:.2f}} s<extra></extra>"))
        st.plotly_chart(style(fig, 420, "fuel-corrected lap time (s)", "tyre age (laps)"), width="stretch")
