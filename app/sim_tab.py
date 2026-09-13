"""The Race sim tab: the whole Grand Prix, lap by lap, with the live strategy
engine on the pit wall for both Haas cars.

Everything here is read from `data/processed/racesim_<key>.json`, written by
`scripts/90_racesim.py` (the supervisor runs it after every practice refit).
Each scenario holds one synthetic race run three times on the same truth: the
engine deciding for Ocon and Bearman, the sealed plan followed blindly, and a
no-model baseline that copies the car ahead.  The tab shows the race as an
analyst would read it (race trace, positions, stints), then one car at a time:
the call the engine made on every lap and how it settled, what it believed
about the tyre against what the tyre was really doing, who it was racing and
where it said the car would rejoin, and its own reasons - and how long every
lap's decision took to compute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.theme import (
    BLACK,
    DEFS,
    HAIR,
    MUTED,
    WHITE,
    badges,
    card,
    ccol,
    chart,
    finite,
    fmt,
    headline,
    how,
    in_ladder,
    more,
    notice,
    palette,
    pct,
    rgba,
    style,
    tiles,
)
from src.racesim import HAAS, load_result

NAMES = {"OCO": "Esteban Ocon", "BEA": "Ollie Bearman"}
NUMBERS = {"OCO": 31, "BEA": 87}
ACTION_COLOUR = {"PIT_NOW": "#FF2E45", "BOX_BY": "#FFD12E", "WAIT": "#FFD12E", "STAY_OUT": "#9C9FA2"}
MODE_WORD = {"engine": "Engine on the wall", "plan": "Sealed plan, followed blindly", "mirror": "No model: copy the car ahead"}


def _car(res: dict, code: str) -> dict | None:
    return next((c for c in res.get("cars", []) if c.get("code") == code), None)


def _stops_text(stops: list) -> str:
    if not stops:
        return "no stop"
    return ", ".join(f"lap {s[0]} → {str(s[1]).title()}" if isinstance(s, (list, tuple))
                     else f"lap {s['lap']} → {str(s['to']).title()}" for s in stops)


def _pos_word(p) -> str:
    return f"P{int(p)}" if finite(p) else "—"


def _delta_places(a, b) -> str:
    """Places gained by a over b, as a word."""
    if not (finite(a) and finite(b)):
        return "—"
    d = int(b) - int(a)
    return "level" if d == 0 else (f"{d:+d} place{'s' if abs(d) != 1 else ''}")


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def _race_trace(res: dict, focus: set):
    """The race history chart: every car against the winner's average lap."""
    cars = res["cars"]
    n = res["n_laps"]
    winner = cars[0]
    ref = float(winner["race_time_s"]) / max(len(winner["laps"]["lap_time_s"]), 1)
    fig = go.Figure()
    pal = palette()
    for c in cars:
        lt = np.asarray(c["laps"]["lap_time_s"], dtype=float)
        cum = np.cumsum(lt)
        laps = np.arange(1, len(lt) + 1)
        y = ref * laps - cum
        is_focus = c["code"] in focus
        col = (pal["accent"] if c["code"] == "OCO" else pal["ink"]) if is_focus else rgba(pal["ink"], 0.22)
        fig.add_trace(go.Scatter(x=laps, y=y, mode="lines", name=c["code"],
                                 line=dict(color=col, width=3 if is_focus else 1),
                                 opacity=1.0 if is_focus else 0.9, showlegend=is_focus,
                                 hovertemplate=f"{c['code']} · lap %{{x}}<br>%{{y:.1f}} s vs the winner's average<extra></extra>"))
        if is_focus:
            for s in c["stops"]:
                k = int(s["lap"]) - 1
                if 0 <= k < len(y):
                    fig.add_trace(go.Scatter(x=[s["lap"]], y=[y[k]], mode="markers", showlegend=False,
                                             marker=dict(size=11, color=ccol(s["to"]), line=dict(color=BLACK, width=2)),
                                             hovertemplate=f"{c['code']} stops lap {s['lap']} → {str(s['to']).title()}<extra></extra>"))
    for L in res.get("sc_laps") or []:
        fig.add_vrect(x0=L - 0.5, x1=L + 0.5, line_width=0, fillcolor=rgba("#FFD12E", 0.10), layer="below")
    fig.update_xaxes(range=[1, n])
    return style(fig, 420, "seconds against the winner's average lap", "race lap")


def _positions_chart(res: dict, code: str, rivals: list):
    n = res["n_laps"]
    pal = palette()
    fig = go.Figure()
    me = _car(res, code)
    for rv in rivals:
        c = _car(res, rv)
        if c is None:
            continue
        fig.add_trace(go.Scatter(x=np.arange(1, n + 1), y=c["laps"]["position"], mode="lines", name=rv,
                                 line=dict(color=rgba(pal["ink"], 0.35), width=1.5),
                                 hovertemplate=f"{rv} · lap %{{x}}<br>P%{{y}}<extra></extra>"))
    if me is not None:
        fig.add_trace(go.Scatter(x=np.arange(1, n + 1), y=me["laps"]["position"], mode="lines", name=code,
                                 line=dict(color=pal["accent"], width=3),
                                 hovertemplate=f"{code} · lap %{{x}}<br>P%{{y}}<extra></extra>"))
        for s in me["stops"]:
            fig.add_vline(x=s["lap"], line=dict(color=WHITE, width=1.5, dash="dot"))
    for L in res.get("sc_laps") or []:
        fig.add_vrect(x0=L - 0.5, x1=L + 0.5, line_width=0, fillcolor=rgba("#FFD12E", 0.10), layer="below")
    fig.update_yaxes(autorange="reversed", dtick=2)
    fig.update_xaxes(range=[1, n])
    return style(fig, 320, "position", "race lap")


def _call_chart(res: dict, code: str, recs: list):
    """What the engine called on each lap: the stop lap it named, the window it
    priced, and the lap the car actually stopped."""
    n = res["n_laps"]
    me = _car(res, code)
    fig = go.Figure()
    laps = [r["lap"] for r in recs]
    lo = [r.get("window_lo") for r in recs]
    hi = [r.get("window_hi") for r in recs]
    ok = [i for i, (a, b) in enumerate(zip(lo, hi)) if a is not None and b is not None]
    if ok:
        fig.add_trace(go.Scatter(x=[laps[i] for i in ok] + [laps[i] for i in ok][::-1],
                                 y=[hi[i] for i in ok] + [lo[i] for i in ok][::-1], fill="toself",
                                 fillcolor=rgba(WHITE, 0.12), line=dict(width=0), name="pit window", hoverinfo="skip"))
    for kind, word in (("BOX_BY", "Box by lap"), ("WAIT", "Wait, then box"), ("PIT_NOW", "Pit now")):
        pts = [(r["lap"], r["dec_lap"], r.get("confidence")) for r in recs
               if r.get("action_kind") == kind and r.get("dec_lap") is not None]
        if pts:
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", name=word,
                                     marker=dict(size=9, color=ACTION_COLOUR[kind], line=dict(color=BLACK, width=1)),
                                     customdata=[p[2] if p[2] is not None else float("nan") for p in pts],
                                     hovertemplate=f"lap %{{x}}: {word.lower()} %{{y}}<br>confidence %{{customdata:.0%}}<extra></extra>"))
    stay = [r["lap"] for r in recs if r.get("action_kind") == "STAY_OUT"]
    if stay:
        fig.add_trace(go.Scatter(x=stay, y=[n] * len(stay), mode="markers", name="Stay out to the flag",
                                 marker=dict(size=7, color=ACTION_COLOUR["STAY_OUT"], symbol="line-ew", line=dict(width=2, color=ACTION_COLOUR["STAY_OUT"])),
                                 hovertemplate="lap %{x}: stay out, no further stop<extra></extra>"))
    if me is not None:
        for s in me["stops"]:
            fig.add_hline(y=s["lap"], line=dict(color=ccol(s["to"]), width=2, dash="dash"),
                          annotation_text=f"stopped lap {s['lap']} → {str(s['to']).title()}",
                          annotation_position="bottom right", annotation_font=dict(color=MUTED, size=11))
    fig.add_trace(go.Scatter(x=[1, n], y=[1, n], mode="lines", line=dict(color=HAIR, width=1, dash="dot"),
                             showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(range=[1, n])
    fig.update_yaxes(range=[0, n + 2])
    return style(fig, 340, "the lap the call names", "race lap")


def _lap_chart(res: dict, code: str):
    me = _car(res, code)
    if me is None:
        return None
    L = me["laps"]
    lt = np.asarray(L["lap_time_s"], dtype=float)
    laps = np.arange(1, len(lt) + 1)
    comps = L["compound"]
    fig = go.Figure()
    for cmp_ in dict.fromkeys(comps):
        m = np.array([c == cmp_ for c in comps])
        fig.add_trace(go.Scatter(x=laps[m], y=lt[m], mode="markers", name=str(cmp_).title(),
                                 marker=dict(size=7, color=rgba(ccol(cmp_), 0.85), line=dict(width=1, color=BLACK)),
                                 hovertemplate=f"{str(cmp_).title()} · lap %{{x}}<br>%{{y:.3f}} s<extra></extra>"))
    for s in me["stops"]:
        fig.add_vline(x=s["lap"], line=dict(color=HAIR, width=1))
    for Ls in res.get("sc_laps") or []:
        fig.add_vrect(x0=Ls - 0.5, x1=Ls + 0.5, line_width=0, fillcolor=rgba("#FFD12E", 0.10), layer="below")
    clean = lt[(np.asarray(L["tyre_age"]) > 1) & ~np.isin(laps, list(L["pit_in"]) + list(L["pit_out"]))]
    if len(clean) > 4:
        fig.update_yaxes(range=[float(np.quantile(clean, 0.02)) - 0.8, float(np.quantile(clean, 0.95)) + 2.0])
    return style(fig, 320, "lap time (s)", "race lap")


def _wear_chart(res: dict, code: str, recs: list):
    """Life used: what the engine believed, lap by lap, against the truth."""
    pal = palette()
    fig = go.Figure()
    x = [r["lap"] for r in recs]
    est = [r.get("wear") for r in recs]
    tru = [r.get("true_wear") for r in recs]
    fig.add_trace(go.Scatter(x=x, y=[v if v is not None else float("nan") for v in tru], mode="lines",
                             name="what the tyre was really doing", line=dict(color=rgba(pal["ink"], 0.5), width=2, dash="dot"),
                             hovertemplate="lap %{x}<br>true life used %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=[v if v is not None else float("nan") for v in est], mode="lines",
                             name="engine's estimate", line=dict(color=pal["accent"], width=2.5),
                             hovertemplate="lap %{x}<br>estimated life used %{y:.0%}<extra></extra>"))
    risk = [r.get("p_past_cliff") for r in recs]
    if any(v for v in risk if v):
        fig.add_trace(go.Bar(x=x, y=[v if v is not None else 0 for v in risk], name="drop-off risk",
                             marker=dict(color=rgba("#FFD12E", 0.35)), hovertemplate="lap %{x}<br>drop-off risk %{y:.0%}<extra></extra>"))
    me = _car(res, code)
    if me is not None:
        for s in me["stops"]:
            fig.add_vline(x=s["lap"], line=dict(color=HAIR, width=1))
    fig.update_yaxes(range=[0, 1.0], tickformat=".0%")
    return style(fig, 300, "share of the tyre's life", "race lap")


def _tick_chart(res: dict):
    t = pd.DataFrame(res.get("ticks") or [])
    if t.empty:
        return None
    t = t[t["lap"] >= 1]
    fig = go.Figure(go.Bar(x=t["lap"], y=t["engine_ms"], marker=dict(color=rgba(palette()["ink"], 0.55)),
                           hovertemplate="lap %{x}<br>%{y:.0f} ms for the whole field<extra></extra>"))
    fig.update_yaxes(rangemode="tozero")
    return style(fig, 220, "milliseconds per lap", "race lap", legend=False)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def _impact_table(sc: dict) -> pd.DataFrame:
    rows = []
    for code in HAAS:
        row = sc.get("impact", {}).get(code) or {}
        for mode in ("engine", "plan", "mirror"):
            r = row.get(mode)
            if not r:
                continue
            rows.append({"Driver": code, "Strategy": MODE_WORD[mode], "Grid": f"P{r['grid']}", "Finish": f"P{r['finish']}",
                         "Gap to the winner": f"{r['gap_to_winner_s']:.1f} s", "Stops": _stops_text(r["stops"]),
                         "vs the engine": ("—" if mode == "engine" else
                                           f"{-row.get(f'places_vs_{mode}', 0):+d} places, {-row.get(f'time_vs_{mode}_s', 0.0):+.1f} s")})
    return pd.DataFrame(rows)


def _classification(res: dict) -> pd.DataFrame:
    rows = []
    for c in res["cars"]:
        rows.append({"Finish": c["finish"], "Driver": c["code"], "Team": c["team"], "Grid": c["grid"],
                     "Change": int(c["grid"]) - int(c["finish"]),
                     "Gap": ("winner" if c["finish"] == 1 else f"+{c['gap_to_winner_s']:.1f} s"),
                     "Stops": _stops_text(c["stops"]), "Tyres": " → ".join(str(x).title() for x in c["compounds"])})
    return pd.DataFrame(rows)


def _scenario_table(data: dict) -> pd.DataFrame:
    rows = []
    for sid in data.get("order", []):
        sc = data["scenarios"].get(sid) or {}
        imp = sc.get("impact") or {}
        e = sc.get("engine") or {}
        row = {"Scenario": sc.get("label", sid)}
        for code in HAAS:
            r = imp.get(code) or {}
            eng, pl, mi = r.get("engine") or {}, r.get("plan") or {}, r.get("mirror") or {}
            row[f"{code} engine"] = _pos_word(eng.get("finish"))
            row[f"{code} plan"] = _pos_word(pl.get("finish"))
            row[f"{code} copy"] = _pos_word(mi.get("finish"))
            row[f"{code} vs plan"] = (f"{r.get('time_vs_plan_s', 0):+.1f} s" if "time_vs_plan_s" in r else "—")
        row["Engine, ms a lap"] = f"{(e.get('tick_ms') or {}).get('median', float('nan')):.0f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _key_laps(recs: list, stops: list) -> list:
    """The laps worth reading the engine's reasons on: the first call, every
    change of call, the in-lap of every stop."""
    out, prev = [], None
    for r in recs:
        a = r.get("action")
        if a and (prev is None or (r.get("changed") and a != prev)):
            out.append(r)
        prev = a or prev
    for s in stops:
        r = next((x for x in recs if x["lap"] == s["lap"]), None)
        if r and r not in out:
            out.append(r)
    out.sort(key=lambda r: r["lap"])
    return out[:8]


# --------------------------------------------------------------------------
# Tyre degradation, one Haas car at a time
# --------------------------------------------------------------------------


def _life_word(laps, n_race: int) -> str:
    """Laps a set would last at the rate it was really wearing."""
    if not finite(laps) or laps <= 0:
        return "—"
    return f"{laps:.0f} laps" if laps < 2 * max(n_race, 1) else "far beyond the race"


def _cliff_word(r: dict) -> str:
    """The engine's laps-to-the-drop-off at the end of a stint, with its band."""
    p50, p10, p90 = r.get("laps_to_cliff_p50"), r.get("laps_to_cliff_p10"), r.get("laps_to_cliff_p90")
    if not finite(p50):
        return "—"
    if p50 >= 60:
        return "60+"
    if finite(p10) and finite(p90) and p90 < 60:
        return f"{p50:.0f} ({p10:.0f}–{p90:.0f})"
    return f"{p50:.0f}"


def _car_mult(res: dict, code: str) -> float:
    """This car's own multiplier on the tyre's wear rate in this scenario's truth."""
    me = _car(res, code) or {}
    if finite(me.get("car_mult")):
        return float(me["car_mult"])
    cm = ((res.get("truth") or {}).get("car_mult") or {}).get(code)
    return float(cm) if finite(cm) else 1.0


def _deg_stints(res: dict, code: str) -> list:
    """One row per set of tyres this car ran: what the tyre really did on this
    car, against what the engine had read off the lap times by the end of it.

    The wear rate comes from the simulation's own wear trace rather than being
    re-derived from the compound's rate, so it carries the fuel-load shape and
    the extrapolation the truth actually ran with.
    """
    me = _car(res, code)
    if me is None:
        return []
    L = me["laps"]
    ages, comps, wear = list(L["tyre_age"]), list(L["compound"]), list(L["wear"])
    budget = (res.get("truth") or {}).get("budget") or {}
    recs = {r["lap"]: r for r in ((res.get("engine") or {}).get(code) or [])}
    n_race = int(res.get("n_laps") or len(ages))
    # a new set is on whenever the age counter restarts
    starts = [i for i in range(len(ages)) if i == 0 or float(ages[i]) <= float(ages[i - 1])]
    rows = []
    for n, i0 in enumerate(starts, start=1):
        i1 = (starts[n] if n < len(starts) else len(ages)) - 1
        n_laps = i1 - i0 + 1
        c = str(comps[i1] or "")
        used = float(wear[i1])              # a set goes on with none of its life used
        rate = used / n_laps if n_laps else float("nan")
        r = recs.get(i1 + 1) or {}
        eff = finite(r.get("m_eff"))
        m, m_lo, m_hi = ((r.get("m_eff"), r.get("m_eff_lo"), r.get("m_eff_hi")) if eff
                         else (r.get("m_mean"), r.get("m_lo"), r.get("m_hi")))
        rows.append({
            "Stint": n,
            "Tyre": c.title(),
            "Laps": f"{i0 + 1}–{i1 + 1} ({n_laps})",
            "Wear per lap": (fmt(rate * float(budget.get(c, 3.8)), 3, " s") if finite(rate) else "—"),
            "Life used": (float(min(used, 1.0)) if finite(used) else None),
            "Set would last": _life_word((1.0 / rate) if finite(rate) and rate > 0 else None, n_race),
            "Engine's wear vs forecast": (f"{m:.2f}× ({m_lo:.2f}–{m_hi:.2f})"
                                          if finite(m) and finite(m_lo) and finite(m_hi)
                                          else (f"{m:.2f}×" if finite(m) else "—")),
            "Engine's wear per lap": fmt(r.get("deg_now_s_per_lap"), 3, " s"),
            "Clean laps read": (int(r["n_clean"]) if finite(r.get("n_clean")) else None),
            "Laps to the drop-off": _cliff_word(r),
            "Drop-off risk": (float(r["p_past_cliff"]) if finite(r.get("p_past_cliff")) else None),
        })
    return rows


def _deg_compounds(res: dict, code: str) -> list:
    """What each tyre in the scenario's truth would do on this car."""
    me, tr = _car(res, code), (res.get("truth") or {})
    if me is None or not tr.get("rate"):
        return []
    cm = _car_mult(res, code)
    ran = list(me["laps"]["compound"])
    n_race = int(res.get("n_laps") or 0)
    rows = []
    for c in in_ladder(list(tr["rate"])):
        rate = float(tr["rate"][c]) * cm
        if not (finite(rate) and rate > 0):
            continue
        rows.append({
            "Tyre": str(c).title(),
            "Wear per lap": fmt(rate * float((tr.get("budget") or {}).get(c, 3.8)), 3, " s"),
            "Set would last": _life_word(1.0 / rate, n_race),
            "Pace offset": f"{float((tr.get('pace_offset') or {}).get(c, 0.0)):+.2f} s",
            "Laps run": sum(1 for x in ran if x == c),
        })
    return rows


def _deg_section(res: dict, val: dict, code: str, key: str) -> None:
    """This car's tyre degradation: the truth it ran on, and what the engine
    made of it.  The field-wide version of the same numbers is in "The tyres
    this race ran on"; every rate here carries this car's own multiplier."""
    me = _car(res, code)
    if me is None:
        return
    tr, v = (res.get("truth") or {}), (val.get(code) or {})
    stints, comps = _deg_stints(res, code), _deg_compounds(res, code)
    if not stints and not comps:
        return
    last = NAMES.get(code, code).split()[-1]
    with card(f"sim-tyrelife-{code}-{key}", f"{last}'s tyre degradation",
              tip="Left: one row per set of tyres the car ran — what the tyre really gave up per lap on this "
                  "car, how much of its life had gone by the end of the stint, and what the engine had read off "
                  "the lap times by then. Right: what each tyre in this scenario's truth would have done on this "
                  "car. " + DEFS["tyre_wear"]):
        c1, c2 = st.columns([1.6, 1], gap="medium")
        with c1:
            st.dataframe(pd.DataFrame(stints), width="stretch", hide_index=True, column_config={
                "Laps": st.column_config.TextColumn(help=DEFS["stint"]),
                "Wear per lap": st.column_config.TextColumn(
                    help="Seconds this set gave up per lap over the stint, on this car, in this race."),
                "Life used": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent",
                                                             color=palette()["ink"], help=DEFS["life_used"]),
                "Set would last": st.column_config.TextColumn(
                    help="Laps to the drop-off at the rate the set was really wearing."),
                "Engine's wear vs forecast": st.column_config.TextColumn(
                    help=DEFS["wear_vs_forecast"] + " Median and the 10th–90th band, at the end of the stint."),
                "Engine's wear per lap": st.column_config.TextColumn(
                    help="What the engine expected the tyre to cost on its next lap, at the end of the stint."),
                "Clean laps read": st.column_config.NumberColumn(
                    help="Green, traffic-free laps the engine had to measure this stint's wear from."),
                "Laps to the drop-off": st.column_config.TextColumn(
                    help=DEFS["drop_off"] + " The engine's estimate at the end of the stint, median (10th–90th)."),
                "Drop-off risk": st.column_config.NumberColumn(format="percent", help=DEFS["drop_off"]),
            })
        with c2:
            st.dataframe(pd.DataFrame(comps), width="stretch", hide_index=True, column_config={
                "Wear per lap": st.column_config.TextColumn(help="Over a stint on this car, at reference fuel load."),
                "Set would last": st.column_config.TextColumn(help="Laps to the drop-off on this car."),
                "Pace offset": st.column_config.TextColumn(help="Lap time this tyre costs against the softest, fresh."),
                "Laps run": st.column_config.NumberColumn(help="Laps this car actually ran on the tyre in this race."),
            })
        bits = [f"{last}'s tyres wore **{_car_mult(res, code):.2f}×** the field's in this race"]
        if finite(v.get("wear_mae")):
            bits.append(f"the engine tracked life used to within {100 * float(v['wear_mae']):.1f} points of the truth")
        mult = float(tr.get("regime_mult", float("nan"))) * float((res.get("config") or {}).get("deg_mult", 1.0))
        if finite(mult):
            bits.append(f"the scenario itself ran at {mult:.2f}× the forecast wear")
        st.caption(" · ".join(bits) + ". Fuel load moves wear between the start and the end of the race, "
                   "not its total; the engine never sees the truth, it has to find it from lap times.")


def _driver_section(res: dict, val: dict, code: str, key: str) -> None:
    recs = (res.get("engine") or {}).get(code) or []
    me = _car(res, code)
    if me is None or not recs:
        st.caption("No engine record for this car in this scenario.")
        return
    v = val.get(code) or {}
    stops = me["stops"]
    first = stops[0] if stops else None
    rec_in = next((r for r in recs if first and r["lap"] == first["lap"]), None)
    life = v.get("life_at_stop") or {}
    rj = v.get("rejoin") or {}
    name = NAMES.get(code, code)
    headline(f"{name}: P{me['grid']} on the grid, P{me['finish']} at the flag",
             (f"{_stops_text(stops)} · the call settled on lap {v['call_settled_lap']}" if v.get("call_settled_lap") else _stops_text(stops)),
             eyebrow=f"#{NUMBERS.get(code, '')} · engine on the wall")
    tiles(f"sim-drv-{code}-{key}", [
        ("Stop taken", f"lap {first['lap']}" if first else "—", "The in-lap the engine sent the car in on.",
         (f"window {rec_in['window_lo']}–{rec_in['window_hi']} that lap" if rec_in and rec_in.get("window_lo") is not None else None)),
        ("Call settled", f"lap {v['call_settled_lap']}" if v.get("call_settled_lap") else "—",
         "The first lap from which the lap the engine named stayed within one lap of the lap the car stopped on.",
         f"{v.get('n_call_changes', 0)} changes of call"),
        ("Confidence at the stop", pct(rec_in.get("confidence")) if rec_in else "—", DEFS["sims"]),
        ("Rejoin", (f"P{rj['projected']} said, P{rj['actual']} got" if rj else "—"),
         "Where the engine said the car would come out of the pit lane, against where it did."),
        ("Life used at the stop", (f"{life['wear_true']:.0%} really" if life.get("wear_true") is not None else "—"),
         DEFS["life_used"], (f"engine said {life['wear_est']:.0%}" if life.get("wear_est") is not None else None)),
        ("Wear vs forecast", (f"{v['regime_est_at_stop']:.2f}×" if finite(v.get("regime_est_at_stop")) else "—"),
         DEFS["wear_vs_forecast"], (f"truth {v['regime_true']:.2f}×" if finite(v.get("regime_true")) else None)),
    ])
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        with card(f"sim-call-{code}-{key}", "The call, lap by lap",
                  tip="Each dot is the lap the engine named on that lap of the race; the band is the pit window it "
                      "priced; the dashed line is the lap the car actually stopped. A call that converges early and "
                      "holds is a call a pit wall can plan around."):
            chart(_call_chart(res, code, recs))
    with c2:
        with card(f"sim-wear-{code}-{key}", "What it believed about the tyre",
                  tip="The engine's estimate of life used, lap by lap, against the truth the simulation ran on. "
                      "Bars are the engine's drop-off risk."):
            chart(_wear_chart(res, code, recs))
    _deg_section(res, val, code, key)
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        rivals = []
        if rec_in:
            rivals = [r.get("driver") for r in (rec_in.get("rivals") or []) if r.get("driver")]
        if not rivals:
            pos = me["laps"]["position"]
            around = [c["code"] for c in res["cars"] if c["code"] != code and abs(int(c["finish"]) - int(me["finish"])) <= 2]
            rivals = around[:4]
        with card(f"sim-pos-{code}-{key}", f"Position through the race, with {', '.join(rivals) if rivals else 'the cars around'}",
                  tip="Dotted verticals are this car's stops; yellow bands a safety car."):
            chart(_positions_chart(res, code, rivals))
    with c2:
        with card(f"sim-laps-{code}-{key}", "Lap times by tyre"):
            fig = _lap_chart(res, code)
            if fig is not None:
                chart(fig)
    # who it was racing at the stop, and where it said it would come out
    if rec_in:
        with card(f"sim-rivals-{code}-{key}", f"Lap {rec_in['lap']}: {rec_in.get('action') or 'the call'}",
                  sub=rec_in.get("headline") or ""):
            if rec_in.get("principal"):
                st.markdown(f"**{rec_in['principal']}**")
            reasons = [r for r in (rec_in.get("reasons") or []) if r != rec_in.get("principal")]
            if reasons:
                st.markdown("\n".join(f"- {r}" for r in reasons))
            rv = rec_in.get("rivals") or []
            if rv:
                st.dataframe(pd.DataFrame([{
                    "Rival": r.get("driver"), "Gap": (f"{abs(r['gap_s']):.1f} s {'ahead' if r['gap_s'] > 0 else 'behind'}" if finite(r.get("gap_s")) else "—"),
                    "Tyre": (f"{str(r.get('compound') or '—').title()} {int(r['tyre_age'])}" if finite(r.get("tyre_age")) else "—"),
                    "Stops": r.get("stops"), "Their stop due": (f"lap {int(r['stop_lap_median'])}" if finite(r.get("stop_lap_median")) else "—"),
                    "Covers us": pct(r.get("p_cover")), "Why it matters": r.get("why") or "—"} for r in rv]),
                    width="stretch", hide_index=True)
            acts = rec_in.get("actions") or []
            if acts:
                st.dataframe(pd.DataFrame([{"Option": a.get("action"), "Lap": a.get("lap"),
                                            "Slower by": fmt(a.get("delta_s"), 1, " s"), "Chance best": pct(a.get("p_best"))}
                                           for a in acts]), width="stretch", hide_index=True)
    with more("Every lap: what the engine said"):
        key_laps = _key_laps(recs, stops)
        for r in key_laps:
            st.markdown(f"**Lap {r['lap']} · {r.get('action') or '—'}** — {r.get('headline') or ''}")
            for line in (r.get("reasons") or [])[:3]:
                st.markdown(f"- {line}")
        rows = [{"Lap": r["lap"], "Pos": r.get("position"), "Tyre": f"{str(r.get('compound') or '').title()} {int(r['tyre_age']) if finite(r.get('tyre_age')) else ''}",
                 "Life used": r.get("wear"), "Really": r.get("true_wear"), "Laps left": (f"{r['laps_to_cliff_p50']:.0f}" if finite(r.get("laps_to_cliff_p50")) and r["laps_to_cliff_p50"] < 60 else "60+"),
                 "Plan from here": r.get("best"), "Window": (f"{r['window_lo']}–{r['window_hi']}" if r.get("window_lo") is not None else "—"),
                 "Call": r.get("action") or "", "Confidence": r.get("confidence"), "Box now costs": fmt(r.get("delta_box_now_s"), 1, " s"),
                 "Rejoin if now": _pos_word(r.get("projected_position_if_now"))} for r in recs]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=380,
                     column_config={"Life used": st.column_config.NumberColumn(format="percent"),
                                    "Really": st.column_config.NumberColumn(format="percent"),
                                    "Confidence": st.column_config.NumberColumn(format="percent")})
        al = [a for a in res.get("alerts") or [] if a.get("driver") == code]
        if al:
            st.markdown("**Alerts raised for this car**")
            st.markdown("\n".join(f"- L{a.get('lap')} · {a.get('text')}" for a in al[:12]))


def _tyre_section(res: dict, key: str) -> None:
    tr = res.get("truth") or {}
    rows = []
    for c in ("SOFT", "MEDIUM", "HARD"):
        if c not in (tr.get("rate") or {}):
            continue
        rate = float(tr["rate"][c])
        budget = float((tr.get("budget") or {}).get(c, 3.8))
        rows.append({"Tyre": c.title(), "Wear per lap (truth)": f"{rate * budget:.3f} s", "Life to the drop-off": (f"{1 / rate:.0f} laps" if rate > 0 else "—"),
                     "Pace offset": f"{float((tr.get('pace_offset') or {}).get(c, 0)):+.2f} s/lap",
                     "Longest stint in the field": (max([len([x for x in cc["laps"]["compound"] if x == c]) for cc in res["cars"]] or [0]))})
    with card(f"sim-tyres-{key}", "The tyres this race ran on",
              tip="The truth of this scenario: one of the sealed model's own possible tyres, scaled by the scenario. "
                  "The engine never sees these numbers; it has to find them from lap times."):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(f"Wear vs the forecast {tr.get('regime_mult', float('nan')):.2f}× · pit lane {tr.get('pit_loss_s', float('nan')):.1f} s · "
                   f"track getting faster by {abs(tr.get('evo_s_per_lap', 0)):.2f} s a lap.")


# --------------------------------------------------------------------------
# The tab
# --------------------------------------------------------------------------


def render_sim(key: str, ev, meta: dict | None = None) -> None:
    data = load_result(key)
    if not data or not data.get("scenarios"):
        notice("No race simulation for this weekend yet. <code>make run</code> builds one after every practice refit, "
               f"or run <code>make racesim EVENT={key}</code>.")
        return
    order = data.get("order") or list(data["scenarios"])
    labels = {sid: data["scenarios"][sid].get("label", sid) for sid in order}
    pick = st.segmented_control("Scenario", order, format_func=lambda s: labels[s], default=order[0], required=True,
                                key=f"sim_scen_{key}", label_visibility="collapsed")
    sc = data["scenarios"][pick]
    res = sc["engine"]
    imp = sc.get("impact") or {}
    val = sc.get("validation") or {}
    tm = res.get("tick_ms") or {}
    oco, bea = _car(res, "OCO"), _car(res, "BEA")

    def _line(code, c):
        r = imp.get(code) or {}
        pl = (r.get("plan") or {}).get("finish")
        return f"{NAMES[code].split()[-1]} P{c['grid']} → P{c['finish']}" + (f" (plan alone P{pl})" if pl else "")

    headline(" · ".join(_line(code, c) for code, c in (("OCO", oco), ("BEA", bea)) if c),
             f"{labels[pick]} · {res['n_laps']} laps · every lap decided for all {len(res['cars'])} cars in "
             f"{tm.get('median', float('nan')):.0f} ms (a lap is {tm.get('lap_time_s', 95):.0f} s)",
             eyebrow=f"Race simulation · {ev.name} · {res.get('grid_source', '')}")
    sum_tiles = []
    for code, c in (("OCO", oco), ("BEA", bea)):
        if not c:
            continue
        r = imp.get(code) or {}
        pl, mi = r.get("plan") or {}, r.get("mirror") or {}
        sum_tiles.append((f"{NAMES[code].split()[-1]}: engine", f"P{c['finish']}", "Finish with the engine making the calls.",
                          _stops_text(c["stops"])))
        sum_tiles.append((f"vs the plan", _delta_places(c["finish"], pl.get("finish")),
                          "Places against following the sealed pre-race plan to the lap, on the same race.",
                          (f"{r.get('time_vs_plan_s', 0):+.1f} s race time" if "time_vs_plan_s" in r else None)))
        sum_tiles.append((f"vs no model", _delta_places(c["finish"], mi.get("finish")),
                          "Places against a wall with no model, that covers the car ahead when it stops.",
                          (f"{r.get('time_vs_mirror_s', 0):+.1f} s race time" if "time_vs_mirror_s" in r else None)))
    tiles(f"sim-top-{key}", sum_tiles)
    badges([(f"Engine {tm.get('median', float('nan')):.0f} ms a lap, worst {tm.get('max', float('nan')):.0f} ms", "ok"),
            (f"{tm.get('n_msgs', 0)} timing messages through the live feed path", "info"),
            (f"Safety car laps {res['sc_laps'][0]}–{res['sc_laps'][-1]}" if res.get("sc_laps") else "No safety car", "flag" if res.get("sc_laps") else "info"),
            (f"Sealed model {res.get('sealed_file') or '—'}", "info")])

    with card(f"sim-trace-{key}", "The race",
              tip="The race-history chart analysts read: every car against the winner's average lap, so a line "
                  "climbing is a car gaining. Haas cars in colour, stops marked with the tyre fitted; yellow bands "
                  "are safety-car laps."):
        chart(_race_trace(res, set(HAAS)))

    view = st.segmented_control("Car", ["Ocon", "Bearman"], default="Ocon", required=True, key=f"sim_car_{key}",
                                label_visibility="collapsed")
    _driver_section(res, val, "OCO" if view == "Ocon" else "BEA", key)

    with card(f"sim-impact-{key}", "What the engine was worth on this race",
              tip="The same race three times: same tyres, same field, same luck. Only what the two Haas cars were "
                  "told differs."):
        st.dataframe(_impact_table(sc), width="stretch", hide_index=True)

    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        with card(f"sim-speed-{key}", "Real-time budget",
                  tip="Time the engine took to price every car's options and make both calls after each lap. The "
                      "feed daemon ticks every two seconds; a lap is over ninety."):
            tiles(f"sim-speed-t-{key}", [
                ("Median", f"{tm.get('median', float('nan')):.0f} ms"),
                ("95th percentile", f"{tm.get('p95', float('nan')):.0f} ms"),
                ("Worst lap", f"{tm.get('max', float('nan')):.0f} ms"),
                ("Share of a lap", f"{100 * tm.get('median', 0) / (1000 * tm.get('lap_time_s', 95)):.2f} %"),
            ])
            fig = _tick_chart(res)
            if fig is not None:
                chart(fig)
    with c2:
        _tyre_section(res, key)

    with card(f"sim-class-{key}", "Classification at the flag"):
        st.dataframe(_classification(res), width="stretch", hide_index=True, height=min(60 + 36 * len(res["cars"]), 500))

    if len(order) > 1:
        with card(f"sim-scen-{key}", "Across scenarios",
                  tip="Each row is a different truth: the model's central tyre, two other tyres it thinks possible, "
                      "a safety car, and tyres wearing faster or slower than forecast."):
            st.dataframe(_scenario_table(data), width="stretch", hide_index=True)

    how("The race is simulated lap by lap for all 22 cars: pace from the qualifying gap to pole, fuel burn, a tyre "
        "the sealed model thinks possible, the track getting faster, close-following losses and a pass threshold, "
        "the standing start, pit-lane time on the in-lap and out-lap, and a safety car that bunches the field.",
        "Every lap is turned into the timing messages the official feed sends and applied to the same live state and "
        "the same engine the pit wall runs on race day. Ocon and Bearman do what the engine says; the other cars run "
        "the plan the weekend model gave each of them, cover an undercut, and box under a safety car when a stop is near.",
        "The same race is run again with the sealed plan followed blindly and with no model at all, so the "
        "difference is the engine's.",
        f"Built {data.get('written_utc', '')[:16].replace('T', ' ')} UTC on the sealed model "
        f"{data.get('sealed_file') or ''}.")
