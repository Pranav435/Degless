"""The Haas tab: what OCO and BEA should do right now, and why.

Two protagonists, one question each: *what should this car do?*  Live, the
answer and its reasons come straight from the race-execution engine's own
state (`plan["decision"]`, `plan["race_state"]`, the field row) through
`src.explain`; before a session is live, the same vocabulary is built from the
pre-race plan (`meta["per_driver"]`, `meta["race_state"]["decisions"]`,
`meta["haas"]` where the pipeline has written it).  Nothing here is a stock
sentence - every number is a value already sitting in the snapshot or the
meta, and a value that is not there prints as "-".

Everything model-internal (the cost curves, the tyre-pack constants, the full
search) stays in the Race plan, Plan builder and Tyres tabs; the Advanced
expander at the bottom only points there.
"""

from __future__ import annotations

import ast

import pandas as pd
import streamlit as st

from app.live_tab import car_lap_chart
from app.theme import age, card, chart, finite, fmt, headline, pct, plan_text, tiles
from src import objective as objlib
from src.explain import explain_actions, explain_decision
from src.haascar import HAAS_DRIVER_META, HAAS_DRIVERS, HAAS_TEAM
from src.live.store import LIVE_DIR, read_laps, read_snapshot

DRIVERS = list(HAAS_DRIVERS)
DRIVER_META = HAAS_DRIVER_META
TEAM = HAAS_TEAM
VIEWS = ["Haas Overview", "Ocon", "Bearman"]
VIEW_DRIVER = {"Ocon": "OCO", "Bearman": "BEA"}


# --------------------------------------------------------------------------
# Small parsers: some tables store a plan's compounds/pit laps as a real list,
# others (round-tripped through a wide parquet -> JSON meta) as their str().
# --------------------------------------------------------------------------


def _compounds_list(x) -> list:
    if isinstance(x, list):
        return [str(c) for c in x]
    if not x:
        return []
    return [c for c in str(x).split("-") if c]


def _int_list(x) -> list:
    if isinstance(x, list):
        return [int(v) for v in x]
    if not x:
        return []
    try:
        v = ast.literal_eval(str(x))
        return [int(i) for i in v] if isinstance(v, (list, tuple)) else []
    except Exception:
        return []


def _has_live(sk) -> bool:
    return bool(sk) and (LIVE_DIR / str(sk) / "snapshot.json").exists()


def _row_at_position(field: list, pos) -> dict | None:
    if pos is None:
        return None
    return next((r for r in field if r.get("position") == pos), None)


def _rival_text(rv: dict | None) -> str:
    if not rv:
        return "—"
    code = rv.get("driver") or rv.get("driver_number")
    if not code:
        return "—"
    gap = rv.get("gap_s")
    if gap is None:
        return str(code)
    return f"{code} · {abs(gap):.1f} s {'ahead' if gap > 0 else 'behind'}"


# --------------------------------------------------------------------------
# Pre-race sourcing: the car's own plan, the team window, the recommended
# lap's decision row - never a number this weekend's model didn't produce.
# --------------------------------------------------------------------------


def _start_compound(per_driver: dict | None) -> str | None:
    comps = _compounds_list((per_driver or {}).get("compounds"))
    return comps[0] if comps else None


def _degradation_pre_race(per_driver: dict | None, haas_car: dict | None, comp: str | None):
    if comp and per_driver and isinstance(per_driver.get("eff_rate"), dict):
        v = per_driver["eff_rate"].get(comp)
        if v is not None:
            return v
    dv = (((haas_car or {}).get("state") or {}).get("deg_rate_by_compound") or {}).get("value")
    if comp and isinstance(dv, dict) and comp in dv:
        return dv[comp]
    return None


def _team_window(meta: dict, stop: int = 1) -> dict | None:
    for w in (meta.get("strategy") or {}).get("pit_windows") or []:
        if int(w.get("stop", -1)) == stop:
            return w
    return None


def _win_prob_for_stops(meta: dict, n_stops):
    if n_stops is None:
        return None
    for r in (meta.get("strategy") or {}).get("by_stops") or []:
        if int(r.get("n_stops", -1)) == int(n_stops):
            return r.get("win_prob_any", r.get("win_prob"))
    return None


def _decision_delta(dec_row: dict | None):
    """The next-best action's cost above the recommended one, from the same
    lap's action table `explain_actions` reads - the pre-race equivalent of
    the live decision's `delta_vs_alternative_s`."""
    if not dec_row:
        return None
    rows = [r for r in (dec_row.get("actions") or []) if r.get("legal")]
    pick = dec_row.get("decision")
    chosen = next((r for r in rows if r.get("action") == pick), None)
    alts = [r for r in rows if r is not chosen and r.get("delta_s") is not None]
    return min((r["delta_s"] for r in alts), default=None)


def _pre_race_context(drv: str, meta: dict) -> dict:
    per_driver = next((r for r in (meta.get("per_driver") or []) if r.get("driver") == drv), None)
    haas_car = ((meta.get("haas") or {}).get("cars") or {}).get(drv) or {}
    decisions = (meta.get("race_state") or {}).get("decisions") or []
    dec_row = decisions[-1] if decisions else None
    comp = _start_compound(per_driver)
    w = _team_window(meta, 1)
    n_stops = (per_driver or {}).get("n_stops")
    return {
        "live": False, "row": None,
        "position": None, "compound": comp, "tyre_age": None, "pace": None,
        "degradation": _degradation_pre_race(per_driver, haas_car, comp),
        "gap_ahead": None, "gap_behind": None,
        "action": dec_row.get("decision") if dec_row else None,
        "window_text": (f"{w['lo']}–{w['hi']}" if w else "—"),
        "confidence": _win_prob_for_stops(meta, n_stops),
        "rejoin_position": None,
        "delta_s": _decision_delta(dec_row),
        "rival": None,
        "plan_text": (plan_text(_compounds_list((per_driver or {}).get("compounds")),
                                _int_list((per_driver or {}).get("pit_laps")))
                     if per_driver else "—"),
        "why": explain_actions(dec_row),
    }


# --------------------------------------------------------------------------
# Live sourcing: the field row and its plan/decision, exactly as the engine
# wrote them this tick.
# --------------------------------------------------------------------------


def _live_context(row: dict, field: list, snap_meta: dict) -> dict:
    plan = row.get("plan") or {}
    dec = plan.get("decision") or {}
    behind = _row_at_position(field, (row.get("position") or 0) + 1)
    exp_meta = {"laps_remaining": plan.get("laps_remaining"), "now_lap": plan.get("now_lap"),
               "sc_active": snap_meta.get("sc_active")}
    rejoin_pos = dec.get("projected_position")
    if rejoin_pos is None:
        rejoin_pos = dec.get("projected_position_if_now")
    rivals = dec.get("rivals") or []
    return {
        "live": True, "row": row,
        "position": row.get("position"), "compound": row.get("compound"),
        "tyre_age": row.get("tyre_age"), "pace": row.get("last_lap_s"),
        "degradation": row.get("deg_now_s_per_lap"),
        "gap_ahead": row.get("interval"), "gap_behind": (behind or {}).get("interval"),
        "action": dec.get("action"),
        "window_text": (f"{plan['window_lo']}–{plan['window_hi']}" if plan.get("window_lo") is not None else "—"),
        "confidence": dec.get("confidence"),
        "rejoin_position": rejoin_pos,
        "delta_s": dec.get("delta_vs_alternative_s"),
        "rival": rivals[0] if rivals else None,
        "plan_text": plan.get("best") or "—",
        "why": explain_decision(plan, row, exp_meta),
    }


def _car_context(drv: str, field: list, snap_meta: dict, meta: dict, live: bool) -> dict:
    if live:
        row = next((r for r in field if r.get("driver") == drv), None)
        if row is not None:
            return _live_context(row, field, snap_meta)
    return _pre_race_context(drv, meta)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def _overview_panel(drv: str, ctx: dict) -> None:
    info = DRIVER_META[drv]
    label = "Live" if ctx["live"] else "Pre-race plan"
    with card(f"haas-ov-{drv}", f"#{info['number']} {info['name']}", sub=label):
        tiles(f"haas-ov1-{drv}", [
            ("Position", f"P{int(ctx['position'])}" if finite(ctx["position"]) else "—"),
            ("Compound", ctx["compound"].title() if ctx.get("compound") else "—"),
            ("Tyre age", f"{ctx['tyre_age']:.0f} laps" if finite(ctx["tyre_age"]) else "—"),
            ("Pace", fmt(ctx["pace"], 3, " s")),
            ("Degradation", fmt(ctx["degradation"], 3, " s/lap")),
        ])
        tiles(f"haas-ov2-{drv}", [
            ("Gap ahead", ctx["gap_ahead"] or "—"),
            ("Gap behind", ctx["gap_behind"] or "—"),
            ("Action", ctx["action"] or "—"),
            ("Pit window", ctx["window_text"]),
            ("Confidence", pct(ctx["confidence"])),
        ])
        tiles(f"haas-ov3-{drv}", [
            ("Rejoin position", f"P{int(ctx['rejoin_position'])}" if finite(ctx["rejoin_position"]) else "—"),
            ("Race-time delta", fmt(ctx["delta_s"], 1, " s")),
            ("Relevant rival", _rival_text(ctx["rival"])),
        ])


def _why_block(why: dict) -> None:
    principal = why.get("principal") or ""
    reasons = [r for r in (why.get("reasons") or []) if r != principal]
    if not principal and not reasons:
        st.caption("Not enough of this weekend's data yet to explain a call.")
        return
    if principal:
        st.markdown(f"**{principal}**")
    if reasons:
        st.markdown("\n".join(f"- {r}" for r in reasons))


def _pit_wall_card(drv: str, ctx: dict) -> None:
    info = DRIVER_META[drv]
    title = f"{ctx['action']} — {info['name']}" if ctx.get("action") else f"No call yet — {info['name']}"
    with card(f"haas-pw-{drv}", title):
        tiles(f"haas-pwt-{drv}", [
            ("Confidence", pct(ctx["confidence"])),
            ("Projected position", f"P{int(ctx['rejoin_position'])}" if finite(ctx["rejoin_position"]) else "—"),
            ("Race-time delta", fmt(ctx["delta_s"], 1, " s")),
        ])
        _why_block(ctx["why"])


def _pit_wall_section(ctxs: dict) -> None:
    with card("haas-pitwall", "Haas Pit Wall"):
        cols = st.columns(2, gap="medium")
        for col, drv in zip(cols, DRIVERS):
            with col:
                _pit_wall_card(drv, ctxs[drv])


def _race_field_section(field: list, focus_rows: list, live: bool) -> None:
    with card("haas-field", "Race field"):
        if not live or not focus_rows:
            st.caption("The race field appears once the session is live.")
            return
        codes: set = set()
        why_map: dict = {}
        for row in focus_rows:
            d = row.get("driver")
            if d:
                codes.add(d)
            dec = (row.get("plan") or {}).get("decision") or {}
            rivals = dec.get("rivals") or []
            if rivals:
                for rv in rivals:
                    c = rv.get("driver") or rv.get("driver_number")
                    if c:
                        codes.add(c)
                        why_map[c] = rv.get("why")
            else:
                pos = row.get("position")
                if pos is not None:
                    for r in field:
                        p = r.get("position")
                        if p is not None and 1 <= abs(int(p) - int(pos)) <= 2:
                            codes.add(r.get("driver"))
        rows = [r for r in field if r.get("driver") in codes]
        rows.sort(key=lambda r: (r.get("position") is None, r.get("position") or 99))
        out = []
        for r in rows:
            d = r.get("driver")
            tyre_age = r.get("tyre_age")
            out.append({
                "Pos": r.get("position"), "Driver": d,
                "Gap to leader": r.get("gap_leader") or "—", "Interval": r.get("interval") or "—",
                "Tyre": f"{(r.get('compound') or '—').title()}"
                       + (f" {tyre_age:.0f}" if finite(tyre_age) else ""),
                "Pit status": "in pit" if r.get("in_pit") else f"{int(r.get('n_pit_stops') or 0)} stop(s)",
                "Why": "this car" if d in DRIVERS else (why_map.get(d) or "—"),
            })
        st.dataframe(pd.DataFrame(out), width="stretch", hide_index=True)


def _strategy_rows(meta: dict) -> list:
    by_stops = (meta.get("strategy") or {}).get("by_stops") or []
    rs = meta.get("race_state") or {}
    curve = rs.get("curve") or {}
    laps_c, places_c = curve.get("laps"), curve.get("places")
    group_now = rs.get("group")
    out = []
    for r in by_stops:
        comps = _compounds_list(r.get("compounds"))
        pits = _int_list(r.get("pit_laps"))
        n_stops = int(r.get("n_stops", len(pits)))
        places = None
        if laps_c and places_c and pits:
            grp = objlib.group_label(comps, pits, n_stops)
            if grp and grp == group_now and pits[0] in laps_c:
                places = places_c[laps_c.index(pits[0])]
        out.append({"n_stops": n_stops, "compounds": comps, "pit_laps": pits,
                   "delta_s": r.get("delta_s"), "delta_p05": r.get("delta_p05"), "delta_p95": r.get("delta_p95"),
                   "win_prob_any": r.get("win_prob_any", r.get("win_prob")), "places_first_cycle": places})
    out.sort(key=lambda x: x["n_stops"])
    return out


def _strategy_section(meta: dict) -> None:
    rows = _strategy_rows(meta)
    with card("haas-strategy", "Strategy comparison"):
        if not rows:
            st.caption("No candidate plans for this weekend yet.")
            return
        out = [{
            "Stops": r["n_stops"], "Sequence": plan_text(r["compounds"], r["pit_laps"]),
            "Pit laps": ", ".join(str(p) for p in r["pit_laps"]) or "—",
            "Race-time delta (s)": fmt(r["delta_s"], 1),
            "Likely range (s)": (f"{r['delta_p05']:+.1f} to {r['delta_p95']:+.1f}"
                                 if finite(r.get("delta_p05")) and finite(r.get("delta_p95")) else "—"),
            "Position change, first cycle": (f"{r['places_first_cycle']:+.2f}"
                                             if finite(r.get("places_first_cycle")) else "—"),
            "Win probability": pct(r.get("win_prob_any")),
        } for r in rows]
        st.dataframe(pd.DataFrame(out), width="stretch", hide_index=True)


def _lap_chart_section(drv: str, name: str, sk: str, ctx: dict) -> None:
    laps = read_laps(sk, root=LIVE_DIR)
    if laps.empty:
        return
    d = laps[laps["driver"] == drv].sort_values("lap_number")
    if d.empty:
        return
    row = ctx.get("row") or {}
    plan = row.get("plan") or {}
    with card(f"haas-laps-{drv}", f"{name}: lap times so far",
             tip="Dots are lap times by tyre; the dotted line is the tyre-wear forecast from here; the shaded "
                 "band is the pit window."):
        chart(car_lap_chart(d, row, plan))


# --------------------------------------------------------------------------
# The tab
# --------------------------------------------------------------------------


def render_haas(key: str, ev, meta: dict, outlook: dict | None, live_session_key: str | None = None) -> None:
    sk = live_session_key if _has_live(live_session_key) else None
    snap = read_snapshot(sk, root=LIVE_DIR) if sk else {}
    field = snap.get("field") or []
    snap_meta = snap.get("meta") or {}
    live = bool(sk and field)

    view = st.segmented_control("Haas view", VIEWS, default="Haas Overview", required=True,
                                key=f"haas_view_{key}", label_visibility="collapsed")

    ctxs = {drv: _car_context(drv, field, snap_meta, meta, live) for drv in DRIVERS}

    if view == "Haas Overview":
        sub = " · ".join(f"{drv}: {ctxs[drv]['action'] or '—'}" for drv in DRIVERS)
        headline("Live pit-wall calls for both cars" if live else "Pre-race plan for both cars", sub,
                eyebrow=f"{TEAM} · {ev.name}")
        cols = st.columns(2, gap="medium")
        for col, drv in zip(cols, DRIVERS):
            with col:
                _overview_panel(drv, ctxs[drv])
        _pit_wall_section(ctxs)
        _race_field_section(field, [ctxs[d]["row"] for d in DRIVERS if ctxs[d].get("row")], live)
        _strategy_section(meta)
        expl = (meta.get("haas") or {}).get("explanation") or []
        if expl:
            with card("haas-explain", "Why the two cars' plans differ"):
                st.markdown("\n".join(f"- {e}" for e in expl))
    else:
        drv = VIEW_DRIVER[view]
        ctx = ctxs[drv]
        info = DRIVER_META[drv]
        headline(f"What should {info['name']} do right now?", ctx["action"] or ctx.get("plan_text") or "—",
                eyebrow=f"#{info['number']} {TEAM} · {ev.name}")
        _overview_panel(drv, ctx)
        _pit_wall_card(drv, ctx)
        _race_field_section(field, [ctx["row"]] if ctx.get("row") else [], live)
        if live and sk:
            _lap_chart_section(drv, info["name"], sk, ctx)

    with st.expander("Advanced", icon=":material/tune:"):
        st.markdown(
            "- Cost curves, the tyre model and the full search are in **Race plan** and **Plan builder**.\n"
            "- Tyre wear and life ranges are in **Tyres**.\n"
            "- Every automatic check on the data and the fit is in **Evidence**."
        )
