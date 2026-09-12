"""The Plan builder: where a strategist builds, compares, stress-tests and
commits a plan - and gets the card to take to the pit wall.

Everything here runs on the race forecast's simulated races
(`outlook_<key>.npz`), so a plan typed in by hand is priced exactly as the
optimiser prices its own, on the same samples, with the same cost terms.
Nothing is fitted: the heavy work is a few hundred samples through the tyre
model, which takes milliseconds and is cached per input.

One section runs at a time, picked from a switch, in the order a strategist
uses them: Compare, Undercut, Safety car, Commit, Practice focus.  The plans
and the "what if" settings above the switch feed every section.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.theme import (
    BLACK, DEFS, DIM, FAINT, HAIR, LETTER, RED, WHITE, age, badge, badges, card, ccol, chart, how, ink_on,
    label_text, more, notice, palette, rgba, saving_word, style, tiles, verdict_word,
)
from src import plans as plan_store
from src import strategy as strat
from src.config import PUSH_GRID, SC_RATE_PER_LAP, get_event
from src.outlook import draws_path, load_model
from src.tyre import TyreModel

SECTIONS = ["Compare", "Undercut", "Safety car", "Commit", "Practice focus"]
SAVING_TO_PUSH = {saving_word(p): float(p) for p in sorted(PUSH_GRID, reverse=True)}
SC_WORDS = {0.0: "none", 0.5: "half", 1.0: "normal", 2.0: "double", 3.0: "triple"}


# --------------------------------------------------------------------------
# Cached model and computations
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _model_cached(key: str, mtime: float) -> TyreModel | None:
    return load_model(key)


def _model(key: str) -> TyreModel | None:
    p = draws_path(key)
    return _model_cached(key, p.stat().st_mtime) if p.exists() else None


def _plans_sig(plans: list) -> str:
    return json.dumps(plans, sort_keys=True, default=str)


@st.cache_data(show_spinner=False)
def _evaluate(key: str, mtime: float, plans_json: str, deg_mult: float, pit_loss: float, sc_mult: float,
              alloc_json: str, caps_json: str):
    model = _model_cached(key, mtime)
    ev = get_event(key)
    plans = json.loads(plans_json)
    tbl, det = strat.evaluate_plans(strat.scale_model(model, deg_mult), ev, plans, pit_loss,
                                    sc_rate=SC_RATE_PER_LAP * sc_mult, allocation=json.loads(alloc_json),
                                    stint_cap=json.loads(caps_json))
    return tbl, det


@st.cache_data(show_spinner=False)
def _playbook(key: str, mtime: float, plan_json: str, deg_mult: float, pit_loss: float, alloc_json: str, caps_json: str):
    model = _model_cached(key, mtime)
    ev = get_event(key)
    plan = json.loads(plan_json)
    pb = strat.sc_playbook(strat.scale_model(model, deg_mult), ev, plan, pit_loss, push=plan.get("push"),
                           allocation=json.loads(alloc_json), stint_cap=json.loads(caps_json))
    return pb, strat.playbook_ranges(pb)


@st.cache_data(show_spinner=False)
def _card_numbers(key: str, mtime: float, plan_json: str, others_json: str, pit_loss: float,
                  alloc_json: str, caps_json: str):
    """Windows, switch triggers, undercut exposure and the expected wear at each stop."""
    model = _model_cached(key, mtime)
    ev = get_event(key)
    plan = json.loads(plan_json)
    others = json.loads(others_json)
    caps = json.loads(caps_json)
    alloc = json.loads(alloc_json)
    tbl, det = strat.evaluate_plans(model, ev, [plan], pit_loss, allocation=alloc, stint_cap=caps)
    d = det[0]
    if not d.get("valid"):
        return None
    p_use = float(d["push"])
    pw = strat.pit_window_model(model, ev, {"compounds": plan["compounds"], "pit_laps": plan["pit_laps"], "push": p_use},
                                pit_loss, max_stint=(caps or None), push=p_use)
    windows = strat.windows_from_sweep(pw, plan)
    switches = []
    for o in others:
        cx = strat.deg_crossover(model, ev, {"compounds": plan["compounds"], "pit_laps": plan["pit_laps"]},
                                 {"compounds": o["compounds"], "pit_laps": o["pit_laps"]}, pit_loss,
                                 allocation=alloc, stint_cap=caps)
        base = next((c for c in cx.get("curve", []) if abs(c["mult"] - 1.0) < 1e-9), None)
        switches.append({"label": o.get("label", plan_store.short_label(o["compounds"], o["pit_laps"])),
                         "mult": cx.get("mult"), "direction": cx.get("direction"),
                         "b_better_at_base": cx.get("b_better_at_base"),
                         "delta_at_base_s": (base["b_minus_a_s"] if base else None)})
    exposure = []
    for i, p_lap in enumerate(plan["pit_laps"]):
        c_now, c_next = plan["compounds"][i], plan["compounds"][i + 1]
        age_ = int(d["stint_lens"][i])
        duel = strat.undercut_duel(model, my_compound=c_now, my_age=age_, their_compound=c_now, their_age=age_,
                                   gap_s=0.0, new_compound=c_next, push=p_use, event=ev, lap_now=p_lap)
        exposure.append({"stop": i + 1, "lap": p_lap, "new_compound": c_next,
                         "gain_1": duel["gain_by_lap_s"][0], "gain_3": duel["gain_by_lap_s"][2],
                         "wear_end": d["wear_end_mean"][i], "wear_p90": d["wear_end_p90"][i]})
    return {"push": p_use, "windows": windows, "switches": switches, "exposure": exposure,
            "wear_end": d["wear_end_mean"], "mean_s": float(d["times"].mean()), "flags": d["flags"],
            "stint_lens": d["stint_lens"]}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _slot(d: dict) -> str:
    return d["label"].split(" · ")[0]


def _stints_text(compounds, lens) -> str:
    return " · ".join(f"{LETTER.get(c, c[0])} {L}" for c, L in zip(compounds, lens))


def _plan_editor(slot: str, default: dict | None, comps: list, n: int, key: str) -> dict:
    k = f"desk_{key}_{slot}"
    d = default or {"compounds": [comps[min(1, len(comps) - 1)], comps[-1]], "pit_laps": [n // 2]}
    with st.container(key=f"editor-{slot}"):
        st.markdown(f"**Plan {slot}**")
        n_stops = st.segmented_control("Stops", [1, 2, 3], default=min(max(len(d["pit_laps"]), 1), 3), required=True,
                                       format_func=lambda v: f"{v} stop{'s' if v > 1 else ''}", key=f"{k}_stops")
        cols = st.columns(n_stops + 1)
        seq, pits = [], []
        for i in range(n_stops + 1):
            with cols[i]:
                dc = d["compounds"][i] if i < len(d["compounds"]) else comps[-1]
                seq.append(st.selectbox(f"Stint {i + 1}", comps, index=(comps.index(dc) if dc in comps else 0),
                                        format_func=str.title, key=f"{k}_c{i}"))
                if i < n_stops:
                    dp = d["pit_laps"][i] if i < len(d["pit_laps"]) else int(round(n * (i + 1) / (n_stops + 1)))
                    pits.append(int(st.number_input(f"Stop {i + 1} lap", min_value=1, max_value=n - 1,
                                                    value=int(min(max(dp, 1), n - 1)), key=f"{k}_p{i}")))
        saving = st.selectbox("Tyre saving", ["Auto (best)"] + list(SAVING_TO_PUSH), key=f"{k}_push", help=DEFS["saving"])
    plan = {"compounds": seq, "pit_laps": pits, "label": f"{slot} · {plan_store.short_label(seq, pits)}"}
    if saving in SAVING_TO_PUSH:
        plan["push"] = SAVING_TO_PUSH[saving]
    return plan


def _gantt(valid: list, n: int):
    fig = go.Figure()
    for i, d in enumerate(valid):
        y = len(valid) - 1 - i
        bounds = [0, *d["pit_laps"], n]
        for c, a, b in zip(d["compounds"], bounds[:-1], bounds[1:]):
            L = b - a
            col = ccol(c)
            letter = LETTER.get(c, c[0])
            fig.add_trace(go.Bar(x=[L], y=[y], base=[a], orientation="h", width=0.55,
                                 marker=dict(color=rgba(col, 0.9), line=dict(color=BLACK, width=2)),
                                 text=(f"{letter} {L}" if L >= 6 else (letter if L >= 3 else "")),
                                 textposition="inside", insidetextanchor="middle", textangle=0, cliponaxis=False,
                                 constraintext="none", textfont=dict(color=ink_on(col, 0.9), size=12), showlegend=False,
                                 hovertemplate=f"<b>Plan {_slot(d)}</b><br>{c.title()} laps {a + 1}–{b} ({L})<extra></extra>"))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(valid)))[::-1],
                     ticktext=[f"Plan {_slot(d)}" for d in valid], showgrid=False)
    fig.update_xaxes(range=[0, n], dtick=10)
    fig.update_layout(barmode="overlay", bargap=0.25)
    return style(fig, 80 + 44 * max(len(valid), 1), "", "race lap", legend=False)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def _section_compare(det: list, valid: list, n: int, outlook: dict) -> None:
    tiles("desk-compare", [
        (f"Plan {_slot(d)}", "fastest" if d["delta_s"] <= 0 else f"+{d['delta_s']:.1f} s",
         f"{label_text(d['label'].split(' · ')[1])}. Against the best of these plans: likely "
         f"{d['delta_p05']:+.1f} to {d['delta_p95']:+.1f} s.", f"{d['p_fastest']:.0%} chance fastest")
        for d in valid])
    c1, c2 = st.columns([1.5, 1], gap="medium")
    with c1:
        ref = valid[0]
        best = min(valid, key=lambda d: d["delta_s"])
        with card("desk-cumulative", f"Plan {_slot(best)} is quickest by the flag" if len(valid) > 1
                  else "Race time through the race",
                  tip=f"Race time against Plan {_slot(ref)}, lap by lap. Above zero is slower. A plan's line jumps at "
                      "its own stop and claws the time back as the fresher tyre pays off."):
            fig = go.Figure()
            fig.add_hline(y=0, line=dict(color=HAIR, width=1))
            for d, (colr, dash) in zip(valid, ((WHITE, "solid"), (DIM, "dash"), (FAINT, "dot"))):
                y = np.asarray(d["trace_mean"]) - np.asarray(ref["trace_mean"])
                fig.add_trace(go.Scatter(x=np.arange(1, n + 1), y=y, mode="lines", name=f"Plan {_slot(d)}",
                                         line=dict(color=colr, width=2, dash=dash),
                                         hovertemplate=f"Plan {_slot(d)} · lap %{{x}}: %{{y:+.1f}} s vs Plan {_slot(ref)}<extra></extra>"))
                fig.add_annotation(x=n, y=float(y[-1]), text=f"Plan {_slot(d)}", showarrow=False, xanchor="left",
                                   xshift=4, font=dict(color=WHITE, size=11))
                for p_lap in d["pit_laps"]:
                    fig.add_vline(x=p_lap, line=dict(color=rgba(WHITE, 0.2), width=1))
            chart(style(fig, 320, f"race time vs Plan {_slot(ref)} (s)", "race lap", legend=False))
    with c2:
        with card("desk-gantt", "The plans on the race clock",
                  sub=" · ".join(f"Plan {_slot(d)}: " + " → ".join(f"{c.title()} {L}" for c, L in zip(d["compounds"], d["stint_lens"]))
                                 for d in valid)):
            chart(_gantt(valid, n))

    with more():
        rows = []
        for d in det:
            if not d.get("valid"):
                rows.append({"Plan": f"Plan {_slot(d)}", "Notes": d["flags"][0]})
                continue
            rows.append({"Plan": f"Plan {_slot(d)}", "Stints": _stints_text(d["compounds"], d["stint_lens"]),
                         "Tyre saving": saving_word(d["push"]), "Slower by (s)": f"{d['delta_s']:+.1f}",
                         "Likely range (s)": f"{d['delta_p05']:+.1f} to {d['delta_p95']:+.1f}",
                         "Chance fastest": f"{d['p_fastest']:.0%}",
                         "Tyre life used at each stop": " · ".join(f"{w:.0%}" for w in d["wear_end_mean"]),
                         "Notes": "; ".join(d["flags"])})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
            "Tyre saving": st.column_config.TextColumn(help=DEFS["saving"]),
            "Likely range (s)": st.column_config.TextColumn(help=DEFS["likely_range"]),
            "Chance fastest": st.column_config.TextColumn(help=DEFS["sims"]),
            "Tyre life used at each stop": st.column_config.TextColumn(help=DEFS["life_used"]),
            "Notes": st.column_config.TextColumn(help="A rule the plan bends (tyre sets, the longest stint run here, the "
                                                      "two-tyre rule). It's priced anyway, because the wall may know better."),
        })
        sc = outlook.get("scenarios") or {}
        if sc.get("matrix"):
            m = pd.DataFrame(sc["matrix"])
            piv = m.pivot(index="deg_mult", columns="pit_delta_s", values="best")
            reg = m.pivot(index="deg_mult", columns="pit_delta_s", values="base_plan_regret_s")
            pal = palette()
            html = "<table style='border-collapse:collapse;font-size:0.84rem;width:100%'><tr><th></th>"
            for col in piv.columns:
                html += f"<th style='padding:4px 10px;text-align:left'>pit loss {col:+.0f} s</th>"
            html += "</tr>"
            for dm in piv.index:
                html += f"<tr><td style='padding:4px 10px'>tyre wear ×{dm:.1f}</td>"
                for col in piv.columns:
                    r = float(reg.loc[dm, col]) if pd.notna(reg.loc[dm, col]) else 0.0
                    bg = rgba(RED, 0.55) if r > 1.0 else pal["paper2"]
                    html += (f"<td style='padding:6px 10px;background:{bg};border:2px solid {pal['css']['card']}'>"
                             f"<b>{label_text(piv.loc[dm, col])}</b><br>forecast plan +{r:.1f} s</td>")
                html += "</tr>"
            html += "</table>"
            with card("desk-grid", f"Safest all-round plan: {label_text(sc.get('robust'))}",
                      tip=f"Best plan under each what-if. Red cells: the forecast plan loses over 1 s there. The safest "
                          f"plan is never more than {sc.get('robust_max_regret_s', float('nan')):.1f} s off the best in "
                          f"any of them; the forecast plan at worst {sc.get('base_max_regret_s', float('nan')):.1f} s."):
                st.markdown(html, unsafe_allow_html=True)


def _section_undercut(key: str, ev, model, comps: list, n: int, deg_mult: float, p_use: float) -> None:
    u = st.columns([1, 0.8, 1, 0.8, 0.8, 1, 0.8])
    my_c = u[0].selectbox("My tyre", comps, index=min(1, len(comps) - 1), format_func=str.title, key=f"desk_{key}_uc_myc")
    my_age = u[1].number_input("My tyre age", 1, 60, 15, key=f"desk_{key}_uc_mya")
    th_c = u[2].selectbox("Their tyre", comps, index=min(1, len(comps) - 1), format_func=str.title, key=f"desk_{key}_uc_thc")
    th_age = u[3].number_input("Their tyre age", 1, 60, 15, key=f"desk_{key}_uc_tha")
    gap = u[4].number_input("Gap (s)", 0.0, 30.0, 1.5, 0.1, key=f"desk_{key}_uc_gap")
    new_c = u[5].selectbox("New tyre fitted", comps, index=len(comps) - 1, format_func=str.title, key=f"desk_{key}_uc_new")
    lap_now = u[6].number_input("Lap now", 1, n, min(20, n), key=f"desk_{key}_uc_lap")
    sm = strat.scale_model(model, float(deg_mult))
    att = strat.undercut_duel(sm, my_compound=my_c, my_age=float(my_age), their_compound=th_c, their_age=float(th_age),
                              gap_s=float(gap), new_compound=new_c, push=p_use, event=ev, lap_now=int(lap_now))
    dfn = strat.undercut_duel(sm, my_compound=th_c, my_age=float(th_age), their_compound=my_c, their_age=float(my_age),
                              gap_s=float(gap), new_compound=new_c, push=p_use, event=ev, lap_now=int(lap_now))
    a1, a2 = st.columns(2, gap="medium")
    with a1:
        v = att["laps_needed"]
        notice(f"<b>Undercut works:</b> pit now onto the {new_c.title()} and you're ahead after {v} lap(s) — "
               f"{att['p_undercut_3lap']:.0%} chance after 3." if v else
               f"<b>Undercut doesn't work:</b> a {gap:.1f} s gap doesn't close within 5 laps "
               f"({att['p_undercut_3lap']:.0%} chance after 3). Staying out — the overcut — is better on these tyres.",
               "ok" if v else "alert")
    with a2:
        v = dfn["laps_needed"]
        notice(f"<b>You're exposed:</b> if they pit and you stay out, they're ahead after {v} lap(s). Cover the stop."
               if v else "<b>Not exposed:</b> their new tyre doesn't close the gap within 5 laps — the overcut is yours.",
               "alert" if v else "ok")
    with card("desk-undercut", "Chance of being ahead, lap by lap",
              tip=DEFS["undercut"] + " " + DEFS["overcut"] + " Your tyre keeps losing pace as it ages, the new one loses "
                  "little, minus a slow first lap and the pace gap between tyres."):
        laps = list(range(1, 6))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=laps, y=att["p_by_lap"], mode="lines+markers", name="You pit, they stay out",
                                 line=dict(color=WHITE, width=2.5), marker=dict(size=9, color=WHITE, line=dict(color=BLACK, width=2)),
                                 hovertemplate="%{x} laps after your stop: %{y:.0%} chance you're ahead<extra></extra>"))
        fig.add_trace(go.Scatter(x=laps, y=dfn["p_by_lap"], mode="lines+markers", name="They pit, you stay out",
                                 line=dict(color=DIM, width=2, dash="dash"), marker=dict(size=9, color=DIM, line=dict(color=BLACK, width=2)),
                                 hovertemplate="%{x} laps after their stop: %{y:.0%} chance they're ahead<extra></extra>"))
        fig.add_hline(y=0.5, line=dict(color=HAIR, dash="dash", width=1))
        fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
        fig.update_xaxes(dtick=1)
        chart(style(fig, 280, "chance the car that pitted is ahead", "laps after the stop"))
    with more():
        st.dataframe(pd.DataFrame({"Laps after the stop": laps,
                                   "You pit: gain on them (s)": [f"{g:+.2f}" for g in att["gain_by_lap_s"]],
                                   "Likely range (s)": [f"{lo:+.1f} to {hi:+.1f}" for lo, hi in zip(att["gain_lo"], att["gain_hi"])],
                                   "You pit: chance ahead": [f"{p:.0%}" for p in att["p_by_lap"]],
                                   "They pit: their gain (s)": [f"{g:+.2f}" for g in dfn["gain_by_lap_s"]],
                                   "They pit: chance they're ahead": [f"{p:.0%}" for p in dfn["p_by_lap"]]}),
                     width="stretch", hide_index=True)


def _section_safety_car(key: str, mtime: float, valid: list, deg_mult: float, pit_loss: float,
                        alloc_json: str, caps_json: str) -> None:
    labels = [d["label"] for d in valid]
    pick = st.selectbox("For plan", labels, format_func=lambda s: f"Plan {s.split(' · ')[0]} · {label_text(s.split(' · ')[1])}",
                        key=f"desk_{key}_pb_plan")
    dsel = next(d for d in valid if d["label"] == pick)
    plan_sel = {"compounds": dsel["compounds"], "pit_laps": dsel["pit_laps"], "push": float(dsel["push"])}
    pb, ranges = _playbook(key, mtime, json.dumps(plan_sel), float(deg_mult), float(pit_loss), alloc_json, caps_json)
    if pb.empty:
        notice("No safety-car playbook: this plan leaves no lap where a stop is possible.")
        return
    kind = {"PIT": "alert", "STAY": "info", "MARGINAL": "info", "PLANNED": "ok"}
    badges([(f"Laps {r['from']}–{r['to']}: {verdict_word(r['verdict']).lower()} ({r['gain_s']:+.1f} s)", kind.get(r["verdict"], "info"), "")
            for r in ranges])
    box = [r for r in ranges if r["verdict"] == "PIT"]
    title = ("Box under a safety car on laps " + ", ".join(f"{r['from']}–{r['to']}" for r in box)) if box \
        else "Stay on plan whenever a safety car comes"
    with card("desk-sc", title, tip=DEFS["safety_car"] + " Above zero, boxing that lap beats carrying on with the plan; "
              "the tyres left are then used in the best way from there. Only the call on that lap is priced."):
        fig = go.Figure()
        for verdict, colr in (("PIT", WHITE), ("MARGINAL", DIM), ("STAY", FAINT), ("PLANNED", None)):
            g = pb[pb["verdict"] == verdict]
            if g.empty:
                continue
            marker = (dict(color=colr, line=dict(color=BLACK, width=1)) if colr
                      else dict(color="rgba(0,0,0,0)", line=dict(color=WHITE, width=1.5)))
            fig.add_trace(go.Bar(x=g["lap"], y=g["gain_s"], name=verdict_word(verdict), marker=marker,
                                 customdata=np.stack([g["p_pit"], g["continuation"]], axis=-1),
                                 hovertemplate="Safety car on lap %{x}<br>boxing: %{y:+.1f} s against staying on plan<br>"
                                               "%{customdata[0]:.0%} chance boxing is better<br>then: %{customdata[1]}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=HAIR, width=1))
        fig.update_layout(barmode="overlay", bargap=0.15)
        chart(style(fig, 300, "seconds saved by boxing", "lap the safety car comes out"))
    with more("Lap by lap"):
        st.dataframe(pd.DataFrame({"Lap": pb["lap"], "Stint": pb["stint"], "Tyre": pb["compound"].str.title(),
                                   "Tyre age": pb["age_on_set"], "Boxing saves (s)": pb["gain_s"].round(1),
                                   "Chance boxing is better": pb["p_pit"].round(2),
                                   "Call": pb["verdict"].map(verdict_word), "Then": pb["continuation"]}),
                     width="stretch", hide_index=True, height=360)


def _triggers(nums: dict, ranges: list) -> tuple[dict, dict]:
    """Short trigger lines for the card, and the plan B block the live view tracks."""
    triggers = {}
    for i, w in enumerate(nums["windows"]):
        ex = nums["exposure"][i] if i < len(nums["exposure"]) else None
        line = f"Stop {w['stop']}: lap {w['recommended']} (window {w['lo']}–{w['hi']})"
        if ex:
            line += (f". Tyre life used above {min(ex['wear_p90'] + 0.05, 1.0):.0%} before lap "
                     f"{max(w['lo'], w['recommended'] - 3)}: stop on lap {w['lo']}")
            line += (f". A car within {max(ex['gain_3'], 0):.1f} s behind on new {ex['new_compound'].title()}s gets "
                     f"ahead in 3 laps if you stay out past lap {w['recommended']}" if ex["gain_3"] > 0 else
                     ". No undercut risk at this tyre age")
        triggers[f"stop_{w['stop']}"] = line
    alt_block = {}
    for s in nums["switches"]:
        name = label_text(s["label"])
        if s.get("b_better_at_base"):
            txt = f"{name} is already {-s['delta_at_base_s']:.1f} s faster at the forecast's tyre wear"
        elif s.get("mult"):
            txt = (f"Switch to {name} if this car's tyre wear reaches ×{s['mult']:.2f} the forecast "
                   f"({s['delta_at_base_s']:+.1f} s slower at ×1.0)")
        else:
            txt = f"{name} never overtakes this plan (tested up to ×2.6 tyre wear)"
        triggers[f"switch_{s['label']}"] = txt
        if not alt_block and s.get("mult") and not s.get("b_better_at_base"):
            alt_block = {"label": s["label"], "delta_s": s["delta_at_base_s"], "switch_mult": s["mult"],
                         "when": "Watch the 'Wear vs forecast' tile for this car on the Now tab"}
    if ranges:
        box = [r for r in ranges if r["verdict"] == "PIT"]
        stay = [r for r in ranges if r["verdict"] == "STAY"]
        triggers["safety_car"] = (("Safety car: box on laps " + ", ".join(f"{r['from']}–{r['to']}" for r in box)
                                   + f" (saves ~{np.mean([r['gain_s'] for r in box]):.0f} s)"
                                   + ("; stay out on laps " + ", ".join(f"{r['from']}–{r['to']}" for r in stay) if stay else ""))
                                  if box else "Safety car: stay on plan whenever it comes")
    return triggers, alt_block


def _section_commit(key: str, ev, mtime: float, outlook: dict, plans: list, valid: list, pit_base: float,
                    alloc_json: str, caps_json: str) -> None:
    n = ev.n_race_laps
    labels = [d["label"] for d in valid]
    k1, k2, k3 = st.columns([1.2, 1, 2])
    with k1:
        card_pick = st.selectbox("Commit plan", labels, format_func=lambda s: f"Plan {s.split(' · ')[0]} · {label_text(s.split(' · ')[1])}",
                                 key=f"desk_{key}_card_plan")
    with k2:
        driver = st.text_input("Driver (3 letters, blank = whole team)", "", key=f"desk_{key}_card_drv",
                               max_chars=3).strip().upper() or None
    with k3:
        note = st.text_input("Note", "", key=f"desk_{key}_card_note", placeholder="e.g. cover VER if within 2 s at the stop")
    dcard = next(d for d in valid if d["label"] == card_pick)
    plan_card = {"compounds": dcard["compounds"], "pit_laps": dcard["pit_laps"]}
    if "push" in next(p for p in plans if p["label"] == card_pick):
        plan_card["push"] = float(dcard["push"])
    others = [{"compounds": d["compounds"], "pit_laps": d["pit_laps"], "label": d["label"].split(" · ")[1]}
              for d in valid if d["label"] != card_pick]
    nums = _card_numbers(key, mtime, json.dumps(plan_card), json.dumps(others), float(pit_base), alloc_json, caps_json)
    if nums is None:
        notice("The chosen plan isn't valid.", "alert")
        return
    # The card's safety-car line is for this plan, at the forecast's own wear and pit loss.
    _, ranges = _playbook(key, mtime, json.dumps({**plan_card, "push": nums["push"]}), 1.0, float(pit_base),
                          alloc_json, caps_json)
    triggers, alt_block = _triggers(nums, ranges)
    preview = {"driver": driver, "label": plan_store.short_label(plan_card["compounds"], plan_card["pit_laps"]),
               "compounds": plan_card["compounds"], "pit_laps": plan_card["pit_laps"], "push": nums["push"],
               "windows": nums["windows"], "triggers": triggers, "alternative": alt_block, "note": note}

    with card("desk-card", f"Decision card · {driver or 'team default'} · {label_text(preview['label'])}"):
        tiles("desk-card", [
            ("Stops", " · ".join(f"lap {w['recommended']}" for w in nums["windows"]) or "none", DEFS["pit_window"],
             " · ".join(f"{w['lo']}–{w['hi']}" for w in nums["windows"]) or None),
            ("Stints", _stints_text(plan_card["compounds"], nums["stint_lens"]), DEFS["stint"]),
            ("Tyre saving", saving_word(nums["push"]), DEFS["saving"]),
            ("Plan B", label_text(alt_block["label"]) if alt_block else "—",
             "The alternative this plan hands over to if tyre wear runs high.",
             f"switch at ×{alt_block['switch_mult']:.2f} wear" if alt_block else None),
        ])
        st.markdown("\n".join(f"- {v}" for v in triggers.values()))
        if nums["flags"]:
            st.caption("Note: " + "; ".join(nums["flags"]))
    md = plan_store.as_markdown(preview, ev.name, n)
    b1, b2, _ = st.columns([1, 1, 3])
    with b1:
        if st.button("Commit this plan", type="primary", key=f"desk_{key}_commit", width="stretch"):
            plan_store.commit_plan(key, compounds=plan_card["compounds"], pit_laps=plan_card["pit_laps"],
                                   push=nums["push"], driver=driver, note=note, windows=nums["windows"],
                                   triggers=triggers, alternative=alt_block, n_race_laps=n,
                                   source=f"plan builder · {outlook.get('stage_label')}")
            st.toast(f"Committed for {driver or 'the team'}. The live race view now tracks the car against it.")
            st.rerun()
    with b2:
        st.download_button("Download card", md, file_name=f"{key}_{driver or 'team'}_card.md", mime="text/markdown",
                           key=f"desk_{key}_dl", width="stretch")

    committed = plan_store.load_plans(key)
    if committed:
        st.markdown("**Committed plans**")
        for p in committed:
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            with c1:
                st.markdown(f"**{p.get('driver') or 'Team default'}** · {label_text(p.get('label'))} · "
                            f"{saving_word(p.get('push', 1))} · {(p.get('committed_utc') or '')[:16].replace('T', ' ')} UTC"
                            + (f" · _{p['note']}_" if p.get("note") else ""))
            with c2:
                if st.button("Remove", key=f"desk_{key}_rm_{p['id']}", width="stretch"):
                    plan_store.remove_plan(key, p["id"])
                    st.rerun()


def _section_practice(outlook: dict) -> None:
    voi, prog = outlook.get("voi") or {}, outlook.get("programme") or []
    if not prog:
        notice("This forecast has no practice priorities.")
        return
    bc = voi.get("by_compound") or {}
    tiles("desk-voi", [("At stake", f"{voi.get('evpi_s', 0):.1f} s",
                        "Race time a better-informed choice of plan would save on average, if tyre wear were known "
                        "exactly. Practice long runs buy some of it back.")]
          + [(r["compound"], f"{r['gain_s']:.1f} s", f"A {r['target_laps']}+ lap run on this tyre is worth this much "
              "to the decision.", f"{r['share']:.0%} of what's at stake") for r in prog[:3]])
    lines = []
    for r in prog:
        v = bc.get(r["compound"]) or {}
        bins = v.get("best_by_rate_bin") or []
        tail = (f"; if it wears slowly the plan is {label_text(bins[0])}, if fast {label_text(bins[-1])}"
                if bins and r.get("decision_moves") else "; the plan doesn't hinge on this tyre")
        lines.append(f"**{r['compound'].title()}:** a long run of {r['target_laps']}+ laps — worth {r['gain_s']:.1f} s{tail}")
    st.markdown("\n".join(f"- {ln}" for ln in lines))
    how("The simulated races are split by how fast each tyre wears, and the best plan re-chosen in each group. "
        "The time that saves, on average, is what a clean long run on that tyre is worth.",
        "Pit loss and the safety car can't be learned in practice; the race measures them.")


# --------------------------------------------------------------------------
# The tab
# --------------------------------------------------------------------------


def render_desk(key: str, ev, outlook: dict | None) -> None:
    n = ev.n_race_laps
    model = _model(key)
    if not outlook or model is None:
        notice("No race forecast for this weekend yet, so there's nothing to price plans on. <code>make run</code> "
               f"builds one by itself, or run <code>make outlook EVENT={key}</code>.")
        return
    mtime = draws_path(key).stat().st_mtime
    st_ = outlook.get("strategy") or {}
    alloc = outlook.get("allocation") or {}
    caps = outlook.get("stint_cap") or {}
    alloc_json, caps_json = json.dumps(alloc), json.dumps(caps)
    comps = [c for c in ("SOFT", "MEDIUM", "HARD") if c in model.compounds]
    pit_base = float(outlook.get("pit_loss_s", 22.0))
    beta = float((outlook.get("thermal") or {}).get("beta_per_c", 0.025))
    p_stops = st_.get("p_stops") or {}

    tiles("desk-top", [
        ("Forecast plan", label_text(st_.get("best")), f"From the race forecast, updated {age(outlook.get('updated_utc'))}.",
         " · ".join(f"{k}-stop {v:.0%}" for k, v in p_stops.items() if v > 0) or None),
        ("Pit loss", f"{pit_base:.1f} s", DEFS["pit_loss"]),
        ("Tyre sets", " · ".join(f"{LETTER.get(c, c[0])}{v}" for c, v in alloc.items()) or "—",
         "How many stints each tyre can do with the sets allowed."),
        ("Longest stints here", " · ".join(f"{LETTER.get(c, c[0])}{v}" for c, v in caps.items()) or "—",
         "The longest each tyre has been run at this circuit. Plans past it get a note."),
    ])

    best = st_.get("best_plan") or {}
    alts = outlook.get("alternatives") or {}
    by_start = st_.get("by_start") or []
    d_a = {"compounds": best.get("compounds"), "pit_laps": best.get("pit_laps")} if best.get("compounds") else None
    pb_ = alts.get("plan_b") or alts.get("plan_c")
    d_b = {"compounds": pb_["compounds"], "pit_laps": pb_["pit_laps"]} if pb_ else None
    other_start = next((b for b in by_start if best.get("compounds") and b["start"] != best["compounds"][0]), None)
    d_c = {"compounds": other_start["compounds"], "pit_laps": other_start["pit_laps"]} if other_start else None

    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    with c1:
        extra = st.pills("Also compare", ["B", "C"], selection_mode="multi",
                         default=[s for s, d in (("B", d_b), ("C", d_c)) if d], format_func=lambda s: f"Plan {s}",
                         key=f"desk_{key}_slots",
                         help="Plan A starts as the forecast's plan, B as the best plan with one more stop, C as the "
                              "best plan on a different starting tyre. Edit any of them.")
    with c2:
        if st.button("Reset to the forecast's plans", key=f"desk_{key}_reset", width="stretch"):
            for k_ in [k_ for k_ in st.session_state if str(k_).startswith(f"desk_{key}_")]:
                del st.session_state[k_]
            st.rerun()
    slots = [("A", d_a)] + [(s, d) for s, d in (("B", d_b), ("C", d_c)) if s in (extra or [])]
    cols = st.columns(len(slots), gap="small")
    plans = []
    for col, (slot, dflt) in zip(cols, slots):
        with col:
            plans.append(_plan_editor(slot, dflt, comps, n, key))

    w1, w2, w3, w4 = st.columns([1.4, 1.2, 1.2, 1], vertical_alignment="bottom")
    with w1:
        deg_mult = st.slider("Tyre wear ×", 0.6, 2.0, 1.0, 0.05, key=f"desk_{key}_deg",
                             help="1.0 is the forecast's own wear rate. Every tyre scales together.")
    with w2:
        pit_delta = st.slider("Pit loss ± (s)", -5.0, 5.0, 0.0, 0.5, key=f"desk_{key}_pit")
    with w3:
        sc_mult = st.select_slider("Safety-car chance", options=list(SC_WORDS), value=1.0, format_func=SC_WORDS.get,
                                   key=f"desk_{key}_sc", help="How likely a safety car is, against a normal race.")
    with w4:
        dT = np.log(deg_mult) / beta if beta > 0 else float("nan")
        badge(f"≈ {dT:+.0f} °C track · pit loss {pit_base + pit_delta:.1f} s", "info",
              help="The wear setting expressed as a change in track temperature: hotter track, faster wear.")
    pit_loss = pit_base + pit_delta

    tbl, det = _evaluate(key, mtime, _plans_sig(plans), float(deg_mult), float(pit_loss), float(sc_mult),
                         alloc_json, caps_json)
    valid = [d for d in det if d.get("valid")]
    section = st.segmented_control("Section", SECTIONS, default="Compare", required=True, key=f"desk_{key}_section",
                                   label_visibility="collapsed")
    if section == "Practice focus":
        _section_practice(outlook)
        return
    if not valid:
        notice("None of the plans is valid: stop laps must be in order, one fewer than the stints.", "alert")
        return
    if section == "Compare":
        _section_compare(det, valid, n, outlook)
    elif section == "Undercut":
        _section_undercut(key, ev, model, comps, n, deg_mult, float(valid[0]["push"]))
    elif section == "Safety car":
        _section_safety_car(key, mtime, valid, deg_mult, pit_loss, alloc_json, caps_json)
    else:
        _section_commit(key, ev, mtime, outlook, plans, valid, pit_base, alloc_json, caps_json)
