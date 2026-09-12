"""The Strategy desk: where a strategist builds, compares, stress-tests and
commits a plan - and gets the sheet to take to the pit wall.

Everything here runs on the outlook's posterior draws (`outlook_<key>.npz`),
so a plan typed in by hand is priced exactly as the optimiser prices its own,
on the same draws, with the same cost terms.  Nothing is fitted: the heavy
work is a few hundred draws through the tyre model, which takes milliseconds
and is cached per input.

The sections, in the order a strategist uses them:

1. **Compare plans** - up to three candidate plans side by side: expected
   cost, the draw-by-draw spread of the difference, the share of draws each
   is fastest on, where each stint ends on its grip budget, and the
   lap-by-lap crossover chart.
2. **Stress test** - one row of scenario controls (degradation multiplier,
   pit-lane time, safety-car likelihood) that re-prices the same plans; the
   outlook's precomputed matrix and the minimax-regret plan alongside.
3. **Undercut and overcut** - a specific duel, both directions.
4. **Safety-car playbook** - box or not, lap by lap, for the plan chosen.
5. **Decision card** - windows, triggers, plan B and its switch point;
   commit it (the live race view then tracks the car against it) and
   download it as text.
6. **Practice programme** - which long run would narrow the decision most.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import plans as plan_store
from src import strategy as strat
from src.config import PUSH_GRID, SC_RATE_PER_LAP, get_event
from src.outlook import draws_path, load_model
from src.tyre import TyreModel

LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H"}
VERDICT_COL = {"PIT": "good", "STAY": "muted", "MARGINAL": "warn", "PLANNED": "accent"}


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
    push = plan.get("push")
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
        age = int(d["stint_lens"][i])
        duel = strat.undercut_duel(model, my_compound=c_now, my_age=age, their_compound=c_now, their_age=age,
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


def _stints_text(compounds, lens) -> str:
    return " · ".join(f"{LETTER.get(c, c[0])} {L}" for c, L in zip(compounds, lens))


def _plan_editor(slot: str, default: dict | None, comps: list, n: int, key: str, *, optional: bool) -> dict | None:
    k = f"desk_{key}_{slot}"
    if optional:
        on = st.checkbox(f"Plan {slot}", value=default is not None, key=f"{k}_on")
        if not on:
            return None
    else:
        st.markdown(f"**Plan {slot}**")
    d = default or {"compounds": [comps[min(1, len(comps) - 1)], comps[-1]], "pit_laps": [n // 2]}
    n_stops = st.selectbox("Stops", [1, 2, 3], index=min(max(len(d["pit_laps"]), 1) - 1, 2), key=f"{k}_stops")
    cols = st.columns(n_stops + 1)
    seq, pits = [], []
    for i in range(n_stops + 1):
        with cols[i]:
            dc = d["compounds"][i] if i < len(d["compounds"]) else comps[-1]
            seq.append(st.selectbox(f"Stint {i + 1}", comps, index=(comps.index(dc) if dc in comps else 0),
                                    key=f"{k}_c{i}"))
            if i < n_stops:
                dp = d["pit_laps"][i] if i < len(d["pit_laps"]) else int(round(n * (i + 1) / (n_stops + 1)))
                pits.append(int(st.number_input(f"Stop {i + 1} lap", min_value=1, max_value=n - 1,
                                                value=int(min(max(dp, 1), n - 1)), key=f"{k}_p{i}")))
    push_opt = st.selectbox("Push", ["auto (best of the grid)"] + [f"{p:.2f}" for p in PUSH_GRID], key=f"{k}_push",
                            help="1.0 is a practice long run, flat out; lower is tyre management, paid in lap time.")
    plan = {"compounds": seq, "pit_laps": pits, "label": f"{slot} · {plan_store.short_label(seq, pits)}"}
    if not push_opt.startswith("auto"):
        plan["push"] = float(push_opt)
    return plan


def _gantt(plans_det: list, n: int, T: dict, ccol, rgba, style, windows_by_plan: dict | None = None):
    fig = go.Figure()
    valid = [d for d in plans_det if d.get("valid")]
    for i, d in enumerate(valid):
        y = len(valid) - 1 - i
        bounds = [0, *d["pit_laps"], n]
        for c, a, b in zip(d["compounds"], bounds[:-1], bounds[1:]):
            L = b - a
            fig.add_trace(go.Bar(x=[L], y=[y], base=[a], orientation="h", width=0.55,
                                 marker=dict(color=rgba(ccol(c), 0.8), line=dict(color=T["surface"], width=2)),
                                 text=(f"{LETTER.get(c, c[0])} {L}" if L >= 6 else (LETTER.get(c, c[0]) if L >= 3 else "")),
                                 textposition="inside", insidetextanchor="middle", textangle=0, cliponaxis=False,
                                 constraintext="none", textfont=dict(color="#ffffff", size=12), showlegend=False,
                                 hovertemplate=f"<b>{d['label']}</b><br>{c} laps {a + 1}–{b} ({L})<extra></extra>"))
        for w in (windows_by_plan or {}).get(d["label"], []):
            fig.add_shape(type="rect", x0=w["lo"] - 1, x1=w["hi"], y0=y - 0.42, y1=y + 0.42, line_width=0,
                          fillcolor=rgba(T["accent"], 0.14), layer="below")
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(valid)))[::-1],
                     ticktext=[d["label"].split(" · ")[0] for d in valid], showgrid=False)
    fig.update_xaxes(range=[0, n], dtick=10)
    fig.update_layout(barmode="overlay", bargap=0.25)
    return style(fig, 80 + 44 * max(len(valid), 1), "", "race lap", legend=False)


# --------------------------------------------------------------------------
# The tab
# --------------------------------------------------------------------------


def render_desk(key: str, ev, outlook: dict | None, T: dict, ccol, chip, callout, style, rgba, compound_pill) -> None:
    n = ev.n_race_laps
    model = _model(key)
    if not outlook or model is None:
        callout("No outlook for this weekend yet. The supervisor builds one by itself (<code>make run</code>); "
                f"or run <code>make outlook EVENT={key}</code>.", "warn")
        return
    mtime = draws_path(key).stat().st_mtime
    st_ = outlook.get("strategy") or {}
    alloc = outlook.get("allocation") or {}
    caps = outlook.get("stint_cap") or {}
    alloc_json, caps_json = json.dumps(alloc), json.dumps(caps)
    comps = [c for c in ("SOFT", "MEDIUM", "HARD") if c in model.compounds]
    pit_base = float(outlook.get("pit_loss_s", 22.0))
    beta = float((outlook.get("thermal") or {}).get("beta_per_c", 0.025))

    st.markdown(
        chip("Model", outlook.get("stage_label", "—"), T["accent"],
             f"{outlook.get('n_draws', 0)} draws · updated {_age(outlook.get('updated_utc'))}")
        + chip("Outlook plan", st_.get("best", "—"), T["good"],
               "P(stops) " + " · ".join(f"{k}: {v:.0%}" for k, v in (st_.get("p_stops") or {}).items()))
        + chip("Pit loss", f"{pit_base:.1f} s", T["muted"], (outlook.get("pit_loss_source") or "")[:44])
        + chip("Sets", " · ".join(f"{LETTER.get(c, c[0])}{v}" for c, v in alloc.items()), T["muted"],
               "stints per compound the allocation allows")
        + (chip("Stint caps", " · ".join(f"{LETTER.get(c, c[0])}{v}" for c, v in caps.items()), T["muted"],
                "longest this circuit has supported") if caps else ""),
        unsafe_allow_html=True)

    # ------------------------------------------------------------------ 1. plans
    st.markdown("#### 1 · Compare plans")
    st.caption("Plan A starts as the outlook's recommendation, B as the best plan with one more stop, C as the best "
               "plan on a different starting compound. Edit any of them; every plan is priced on the same posterior "
               "draws, so the differences are honest and the shared uncertainty cancels.")
    best = st_.get("best_plan") or {}
    alts = outlook.get("alternatives") or {}
    by_start = st_.get("by_start") or []
    d_a = {"compounds": best.get("compounds"), "pit_laps": best.get("pit_laps")} if best.get("compounds") else None
    pb_ = alts.get("plan_b") or alts.get("plan_c")
    d_b = {"compounds": pb_["compounds"], "pit_laps": pb_["pit_laps"]} if pb_ else None
    other_start = next((b for b in by_start if best.get("compounds") and b["start"] != best["compounds"][0]), None)
    d_c = {"compounds": other_start["compounds"], "pit_laps": other_start["pit_laps"]} if other_start else None
    if st.button("Reset plans to the outlook's", key=f"desk_{key}_reset"):
        for k_ in [k_ for k_ in st.session_state if str(k_).startswith(f"desk_{key}_")]:
            del st.session_state[k_]
        st.rerun()
    ca, cb, cc = st.columns(3)
    with ca:
        plan_a = _plan_editor("A", d_a, comps, n, key, optional=False)
    with cb:
        plan_b = _plan_editor("B", d_b, comps, n, key, optional=True)
    with cc:
        plan_c = _plan_editor("C", d_c, comps, n, key, optional=True)
    plans = [p for p in (plan_a, plan_b, plan_c) if p]

    # ------------------------------------------------------------------ 2. scenario row
    st.markdown("#### 2 · Stress test")
    s1, s2, s3, s4 = st.columns([1.4, 1.2, 1, 1.4])
    with s1:
        deg_mult = st.slider("Degradation × outlook", 0.6, 2.0, 1.0, 0.05, key=f"desk_{key}_deg",
                             help="1.0 is the outlook's own rate. Every compound scales together.")
    with s2:
        pit_delta = st.slider("Pit loss delta (s)", -5.0, 5.0, 0.0, 0.5, key=f"desk_{key}_pit")
    with s3:
        sc_mult = st.select_slider("Safety-car likelihood", options=[0.0, 0.5, 1.0, 2.0, 3.0], value=1.0,
                                   key=f"desk_{key}_sc", help="multiplier on the per-lap safety-car rate")
    with s4:
        dT = np.log(deg_mult) / beta if beta > 0 else float("nan")
        st.markdown(chip("Equivalent", f"{dT:+.0f} °C track", T["warn"] if abs(dT) > 6 else T["muted"],
                         f"at {beta:+.3f}/°C · pit {pit_base + pit_delta:.1f} s"), unsafe_allow_html=True)
    pit_loss = pit_base + pit_delta
    tbl, det = _evaluate(key, mtime, _plans_sig(plans), float(deg_mult), float(pit_loss), float(sc_mult),
                         alloc_json, caps_json)
    valid = [d for d in det if d.get("valid")]
    if not valid:
        callout("None of the plans is valid: stop laps must be in order, one fewer than the stints.", "bad")
        return
    rows = []
    for d in det:
        if not d.get("valid"):
            rows.append({"Plan": d["label"], "Notes": d["flags"][0]})
            continue
        rows.append({"Plan": d["label"], "Stints": _stints_text(d["compounds"], d["stint_lens"]),
                     "Push": f"{d['push']:.2f}",
                     "Δ vs best (s)": f"{d['delta_s']:+.1f}", "Δ 5–95%": f"{d['delta_p05']:+.1f} … {d['delta_p95']:+.1f}",
                     "P(fastest)": f"{d['p_fastest']:.0%}",
                     "Stint-end wear": " · ".join(f"{w:.2f}" for w in d["wear_end_mean"]),
                     "Notes": "; ".join(d["flags"])})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Δ is the expected race-time loss against the best of these plans on the scenario above, with the "
               "5–95% band of that difference draw by draw. Stint-end wear is the share of the grip budget used "
               "when the tyre comes off (1.0 = the cliff). Notes flag a rule the plan bends — the allocation, a "
               "circuit stint cap, the two-compound rule — priced anyway, because the wall may know better.")

    c1, c2 = st.columns([1.5, 1])
    with c1:
        ref = valid[0]
        fig = go.Figure()
        fig.add_hline(y=0, line=dict(color=T["hairline"], width=1))
        dashes = ["solid", "dash", "dot"]
        for i, d in enumerate(valid):
            y = np.asarray(d["trace_mean"]) - np.asarray(ref["trace_mean"])
            x = np.arange(1, n + 1)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=d["label"].split(" · ")[0],
                                     line=dict(color=T["accent"] if i == 0 else T["ink"], width=2, dash=dashes[i % 3]),
                                     hovertemplate=f"{d['label']}<br>lap %{{x}}: %{{y:+.1f}} s vs {ref['label'].split(' · ')[0]}<extra></extra>"))
            fig.add_annotation(x=n, y=float(y[-1]), text=d["label"].split(" · ")[0], showarrow=False, xanchor="left",
                               xshift=4, font=dict(color=T["muted"], size=11))
            for p_lap in d["pit_laps"]:
                fig.add_vline(x=p_lap, line=dict(color=rgba(T["muted"], 0.35), width=1))
        st.plotly_chart(style(fig, 330, f"cumulative time vs {ref['label'].split(' · ')[0]} (s)", "race lap"),
                        width="stretch")
        st.caption("Positive is slower than Plan A at that point of the race. Thin verticals are the stops; a "
                   "plan's line jumps at its own stop and claws the time back as the fresher tyre pays off.")
    with c2:
        st.plotly_chart(_gantt(valid, n, T, ccol, rgba, style), width="stretch")
        st.caption("Plan in words: " + " — ".join(
            f"**{d['label'].split(' · ')[0]}** " + " → ".join(f"{c} {L}" for c, L in zip(d["compounds"], d["stint_lens"]))
            for d in valid))

    sc = outlook.get("scenarios") or {}
    if sc.get("matrix"):
        with st.expander("The outlook's own scenario matrix: the best plan under each scenario, and the plan that regrets least"):
            m = pd.DataFrame(sc["matrix"])
            piv = m.pivot(index="deg_mult", columns="pit_delta_s", values="best")
            reg = m.pivot(index="deg_mult", columns="pit_delta_s", values="base_plan_regret_s")
            html = "<table style='border-collapse:collapse;font-size:0.84rem'><tr><th></th>"
            for col in piv.columns:
                html += f"<th style='padding:4px 10px;color:{T['muted']}'>pit {col:+.0f} s</th>"
            html += "</tr>"
            for dm in piv.index:
                html += f"<tr><td style='padding:4px 10px;color:{T['muted']}'>deg ×{dm:.1f}</td>"
                for col in piv.columns:
                    r = float(reg.loc[dm, col]) if pd.notna(reg.loc[dm, col]) else 0.0
                    bg = rgba(T["warn"], 0.18) if r > 1.0 else rgba(T["good"], 0.10)
                    html += (f"<td style='padding:6px 10px;background:{bg};border:2px solid {T['surface']}'>"
                             f"<b>{piv.loc[dm, col]}</b><br><span style='color:{T['muted']}'>base plan +{r:.1f} s</span></td>")
                html += "</tr>"
            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)
            st.markdown(f"Least-regret plan across all nine: **{sc.get('robust')}** (worst case "
                        f"{sc.get('robust_max_regret_s', float('nan')):.1f} s); the outlook's plan is at worst "
                        f"{sc.get('base_max_regret_s', float('nan')):.1f} s off in any of them.")

    # ------------------------------------------------------------------ 3. undercut
    st.markdown("#### 3 · Undercut and overcut")
    u1, u2, u3, u4, u5, u6, u7 = st.columns([1, 0.8, 1, 0.8, 0.8, 1, 0.8])
    with u1:
        my_c = st.selectbox("My tyre", comps, index=min(1, len(comps) - 1), key=f"desk_{key}_uc_myc")
    with u2:
        my_age = st.number_input("My age", 1, 60, 15, key=f"desk_{key}_uc_mya")
    with u3:
        th_c = st.selectbox("Their tyre", comps, index=min(1, len(comps) - 1), key=f"desk_{key}_uc_thc")
    with u4:
        th_age = st.number_input("Their age", 1, 60, 15, key=f"desk_{key}_uc_tha")
    with u5:
        gap = st.number_input("Gap (s)", 0.0, 30.0, 1.5, 0.1, key=f"desk_{key}_uc_gap")
    with u6:
        new_c = st.selectbox("Fresh tyre fitted", comps, index=len(comps) - 1, key=f"desk_{key}_uc_new")
    with u7:
        lap_now = st.number_input("Lap now", 1, n, min(20, n), key=f"desk_{key}_uc_lap")
    p_use = float(valid[0]["push"])
    sm = strat.scale_model(model, float(deg_mult))
    att = strat.undercut_duel(sm, my_compound=my_c, my_age=float(my_age), their_compound=th_c, their_age=float(th_age),
                              gap_s=float(gap), new_compound=new_c, push=p_use, event=ev, lap_now=int(lap_now))
    dfn = strat.undercut_duel(sm, my_compound=th_c, my_age=float(th_age), their_compound=my_c, their_age=float(my_age),
                              gap_s=float(gap), new_compound=new_c, push=p_use, event=ev, lap_now=int(lap_now))
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(f"**I pit now onto the {new_c}; they stay out.** Am I ahead after…")
        st.dataframe(pd.DataFrame({"laps": [1, 2, 3, 4, 5],
                                   "gain on them (s)": [f"{g:+.2f}" for g in att["gain_by_lap_s"]],
                                   "90% band": [f"{lo:+.1f} … {hi:+.1f}" for lo, hi in zip(att["gain_lo"], att["gain_hi"])],
                                   "P(ahead)": [f"{p:.0%}" for p in att["p_by_lap"]]}),
                    width="stretch", hide_index=True)
        v = att["laps_needed"]
        callout(f"Undercut works after <b>{v} lap(s)</b> — P(ahead after 3) {att['p_undercut_3lap']:.0%}." if v
                else f"The undercut does not clear a {gap:.1f} s gap within 5 laps (P after 3: {att['p_undercut_3lap']:.0%}). "
                     "Staying out — the overcut — is the better move on these tyres.",
                "good" if v else "warn")
    with a2:
        st.markdown(f"**They pit now onto the {new_c}; I stay out.** Are they ahead of me after…")
        st.dataframe(pd.DataFrame({"laps": [1, 2, 3, 4, 5],
                                   "their gain (s)": [f"{g:+.2f}" for g in dfn["gain_by_lap_s"]],
                                   "P(they jump me)": [f"{p:.0%}" for p in dfn["p_by_lap"]]}),
                    width="stretch", hide_index=True)
        v = dfn["laps_needed"]
        callout(f"Exposed: they are ahead after <b>{v} lap(s)</b> if I stay out — cover the stop." if v
                else "Not exposed within 5 laps: their fresh tyre does not close the gap; the overcut is mine.",
                "bad" if v else "good")
    st.caption("Same arithmetic as the live engine's undercut threat/opportunity: the defender's ageing tyre keeps "
               "losing pace, the attacker's fresh one loses little, less the cold first lap and the compound "
               "pace difference. Probabilities are over the posterior draws at the scenario above.")

    # ------------------------------------------------------------------ 4. safety car
    st.markdown("#### 4 · Safety-car playbook")
    labels = [d["label"] for d in valid]
    pick = st.selectbox("For plan", labels, key=f"desk_{key}_pb_plan")
    dsel = next(d for d in valid if d["label"] == pick)
    plan_sel = {"compounds": dsel["compounds"], "pit_laps": dsel["pit_laps"], "push": float(dsel["push"])}
    pb, ranges = _playbook(key, mtime, json.dumps(plan_sel), float(deg_mult), float(pit_loss), alloc_json, caps_json)
    if pb.empty:
        st.info("No playbook: the plan leaves no lap on which a stop is possible.")
    else:
        chips = ""
        for r in ranges:
            col = T[VERDICT_COL.get(r["verdict"], "muted")]
            sub = (f"{r['gain_s']:+.1f} s" + (f" · {r['continuation']}" if r["verdict"] in ("PIT", "MARGINAL", "PLANNED") else ""))
            chips += chip(f"SC on laps {r['from']}–{r['to']}", r["verdict"].title(), col, sub[:70])
        st.markdown(chips, unsafe_allow_html=True)
        fig = go.Figure()
        for verdict in ("PIT", "MARGINAL", "STAY", "PLANNED"):
            g = pb[pb["verdict"] == verdict]
            if g.empty:
                continue
            fig.add_trace(go.Bar(x=g["lap"], y=g["gain_s"], name=verdict.title(),
                                 marker=dict(color=rgba(T[VERDICT_COL[verdict]], 0.8), line=dict(color=T["surface"], width=1)),
                                 customdata=np.stack([g["p_pit"], g["continuation"]], axis=-1),
                                 hovertemplate="SC on lap %{x}<br>box now: %{y:+.1f} s vs staying on plan<br>"
                                               "P(box is better) %{customdata[0]:.0%}<br>%{customdata[1]}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=T["hairline"], width=1))
        fig.update_layout(barmode="overlay", bargap=0.15)
        st.plotly_chart(style(fig, 300, "seconds saved by boxing under the SC", "lap the safety car appears"), width="stretch")
        st.caption("Above zero, boxing this lap at the discounted pit loss beats continuing the plan; the "
                   "continuation is the best use of the tyres left (remaining stops re-optimised, or one stop "
                   "fewer if the new tyre can reach the flag). Only the decision *this lap* is priced.")
        with st.expander("Lap by lap"):
            st.dataframe(pb[["lap", "stint", "compound", "age_on_set", "gain_s", "p_pit", "verdict", "continuation"]]
                         .rename(columns={"age_on_set": "age", "gain_s": "box gain (s)", "p_pit": "P(box better)"})
                         .round(2), width="stretch", hide_index=True, height=360)

    # ------------------------------------------------------------------ 5. decision card
    st.markdown("#### 5 · Decision card — commit the plan the wall will run")
    k1, k2, k3 = st.columns([1.2, 1, 2])
    with k1:
        card_pick = st.selectbox("Commit plan", labels, key=f"desk_{key}_card_plan")
    with k2:
        driver = st.text_input("Driver (TLA, blank = team default)", "", key=f"desk_{key}_card_drv",
                               max_chars=3).strip().upper() or None
    with k3:
        note = st.text_input("Note", "", key=f"desk_{key}_card_note",
                             placeholder="e.g. cover VER if within 2 s at the stop")
    dcard = next(d for d in valid if d["label"] == card_pick)
    plan_card = {"compounds": dcard["compounds"], "pit_laps": dcard["pit_laps"]}
    if "push" in [p for p in plans if p["label"] == card_pick][0]:
        plan_card["push"] = float(dcard["push"])
    others = [{"compounds": d["compounds"], "pit_laps": d["pit_laps"], "label": d["label"].split(" · ")[1]}
              for d in valid if d["label"] != card_pick]
    nums = _card_numbers(key, mtime, json.dumps(plan_card), json.dumps(others), float(pit_base), alloc_json, caps_json)
    if nums is None:
        st.info("The chosen plan is not valid.")
        return
    triggers = {}
    for i, w in enumerate(nums["windows"]):
        ex = nums["exposure"][i] if i < len(nums["exposure"]) else None
        line = (f"Stop {w['stop']}: lap {w['recommended']}, window {w['lo']}–{w['hi']} (within 1 s of optimal)")
        if ex:
            line += (f". Expect wear {ex['wear_end']:.2f} at the stop (p90 {ex['wear_p90']:.2f}); live wear above "
                     f"{min(ex['wear_p90'] + 0.05, 1.0):.2f} before lap {max(w['lo'], w['recommended'] - 3)} means the tyre is "
                     f"running hot: go to lap {w['lo']}")
            line += (f". Undercut exposure: a car within {max(ex['gain_3'], 0):.1f} s behind on a fresh {ex['new_compound']} "
                     f"is ahead after 3 laps if you stay out past lap {w['recommended']}"
                     if ex["gain_3"] > 0 else
                     f". Undercut exposure: nil — a fresh {ex['new_compound']} behind does not gain on you at this age")
        triggers[f"stop_{w['stop']}"] = line
    alt_block = {}
    for s in nums["switches"]:
        if s.get("b_better_at_base"):
            txt = f"{s['label']} is already faster by {-s['delta_at_base_s']:.1f} s at the outlook's degradation"
        elif s.get("mult"):
            txt = (f"Switch to {s['label']} if the live degradation multiplier reaches ×{s['mult']:.2f} "
                   f"(it is {s['delta_at_base_s']:+.1f} s slower at ×1.0)")
        else:
            txt = f"{s['label']} does not overtake this plan within ×2.6 degradation ({s['delta_at_base_s']:+.1f} s at ×1.0)"
        triggers[f"switch_{s['label']}"] = txt
        if not alt_block and s.get("mult") and not s.get("b_better_at_base"):
            alt_block = {"label": s["label"], "delta_s": s["delta_at_base_s"], "switch_mult": s["mult"],
                         "when": "the live engine's per-car degradation multiplier is the 'This car's deg' chip on the Now tab"}
    if ranges:
        pit_ranges = [r for r in ranges if r["verdict"] == "PIT"]
        stay_ranges = [r for r in ranges if r["verdict"] == "STAY"]
        triggers["safety_car"] = ("Safety car: box on laps " + ", ".join(f"{r['from']}–{r['to']}" for r in pit_ranges)
                                  + (f" (saves ~{np.mean([r['gain_s'] for r in pit_ranges]):.0f} s)" if pit_ranges else "")
                                  + ("; stay out on laps " + ", ".join(f"{r['from']}–{r['to']}" for r in stay_ranges)
                                     if stay_ranges else "")) if pit_ranges else "Safety car: stay on plan whenever it comes"
    committed = plan_store.load_plans(key)
    preview = {"driver": driver, "label": plan_store.short_label(plan_card["compounds"], plan_card["pit_laps"]),
               "compounds": plan_card["compounds"], "pit_laps": plan_card["pit_laps"], "push": nums["push"],
               "windows": nums["windows"], "triggers": triggers, "alternative": alt_block, "note": note}
    md = plan_store.as_markdown(preview, ev.name, n)
    b1, b2, b3 = st.columns([1, 1, 3])
    with b1:
        if st.button("Commit this plan", type="primary", key=f"desk_{key}_commit", width="stretch"):
            plan_store.commit_plan(key, compounds=plan_card["compounds"], pit_laps=plan_card["pit_laps"],
                                   push=nums["push"], driver=driver, note=note, windows=nums["windows"],
                                   triggers=triggers, alternative=alt_block, n_race_laps=n,
                                   source=f"desk · {outlook.get('stage_label')}")
            st.success(f"Committed for {driver or 'the team'}. The live race view now tracks the car against it.")
            st.rerun()
    with b2:
        st.download_button("Download card (.md)", md, file_name=f"{key}_{driver or 'team'}_card.md", mime="text/markdown",
                           key=f"desk_{key}_dl", width="stretch")
    with b3:
        st.caption(f"Expected cost {nums['mean_s']:.1f} s at push {nums['push']:.2f}"
                   + (" · " + "; ".join(nums["flags"]) if nums["flags"] else ""))
    st.markdown(md)
    if committed:
        st.markdown("**Committed plans**")
        for p in committed:
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"- **{p.get('driver') or 'team default'}** · {p.get('label')} · push {p.get('push', 1):.2f} · "
                            f"{(p.get('committed_utc') or '')[:16].replace('T', ' ')} UTC"
                            + (f" · _{p['note']}_" if p.get("note") else ""))
            with c2:
                if st.button("Remove", key=f"desk_{key}_rm_{p['id']}"):
                    plan_store.remove_plan(key, p["id"])
                    st.rerun()

    # ------------------------------------------------------------------ 6. practice programme
    st.markdown("#### 6 · Practice programme — what would narrow the decision")
    voi = outlook.get("voi") or {}
    prog = outlook.get("programme") or []
    if prog:
        st.markdown(chip("Decision uncertainty", f"{voi.get('evpi_s', 0):.1f} s", T["accent"],
                         "expected regret of choosing now vs knowing the true degradation"), unsafe_allow_html=True)
        for r in prog:
            st.markdown(f"- {r['text']}")
        bc = voi.get("by_compound") or {}
        rows = [{"Compound": c, "Worth (s)": round(v["gain_s"], 2), "Share": f"{v['share']:.0%}",
                 "Rate range (s/lap)": f"{v['rate_bins_s_per_lap'][0][0]:.3f} – {v['rate_bins_s_per_lap'][-1][1]:.3f}",
                 "Best plan at the slow end": (v["best_by_rate_bin"] or [None])[0],
                 "…at the fast end": (v["best_by_rate_bin"] or [None])[-1]} for c, v in bc.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Value of information: the draws are split by each compound's own degradation rate and the "
                   "best plan re-chosen within each quarter; the time that would save, averaged, is what a "
                   "clean long run on that compound is worth to the decision. Pit loss and the safety car are "
                   "not learnable in practice — the race measures them.")
    else:
        st.caption("No value-of-information breakdown in this outlook.")


def _age(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        s = (datetime.now(timezone.utc) - datetime.fromisoformat(iso.replace("Z", "+00:00"))).total_seconds()
        return f"{s:.0f} s ago" if s < 120 else (f"{s / 60:.0f} min ago" if s < 7200 else f"{s / 3600:.1f} h ago")
    except Exception:
        return "—"
