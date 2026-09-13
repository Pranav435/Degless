"""Why the model made this call, in the numbers it made it with.

Every sentence here is assembled from a quantity that is already in the state:
a term of the action table (tyre, track position, rejoin traffic), a rival's
gap, compound, tyre age, stop distribution or cover probability, the rejoin
projection, or the tyre-life posterior.  Nothing is a stock phrase about
strategy - if a number is missing the sentence is not produced, so the pit wall
never reads an explanation the model cannot support.

Two entry points, one vocabulary:

* `explain_decision(plan, row, meta)` for the live call (`plan["decision"]` and
  `plan["race_state"]` from `src.live.engine`);
* `explain_actions(decisions_row)` for the pre-race decision table
  (`meta["race_state"]["decisions"]`, written by `scripts/10_pipeline.py`).

Both are pure functions of dictionaries: no model, no feed, no I/O.  That is
what makes them testable on a synthetic plan, and it is why the live engine can
call them inside a tick.
"""

from __future__ import annotations

import math

MAX_REASONS = 5          # the principal reason plus up to four supporting ones fits a pit-wall panel
# Below this chance of reaching the cliff before the stop the life line is
# noise on a quantity no practice long run identifies, so it is not said at all
# (it is always in `decision["life"]` for anyone who wants the number).
LIFE_RISK_SHOW_P = 0.05


# --------------------------------------------------------------------------
# formatting: a number or nothing
# --------------------------------------------------------------------------


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _s(x, dp: int = 1, sign: bool = False) -> str | None:
    v = _num(x)
    if v is None:
        return None
    return f"{v:+.{dp}f} s" if sign else f"{v:.{dp}f} s"


def _pct(x) -> str | None:
    v = _num(x)
    return None if v is None else f"{100 * v:.0f} %"


def _laps(k) -> str:
    n = _num(k)
    if n is None:
        return "some laps"
    n = int(round(n))
    return "1 lap" if n == 1 else f"{n} laps"


# --------------------------------------------------------------------------
# the live call
# --------------------------------------------------------------------------


def _rows(plan: dict) -> list:
    return [r for r in ((plan.get("race_state") or {}).get("actions") or []) if r.get("legal")]


def _row_at(rows: list, lap) -> dict | None:
    if lap is None:
        return None
    return next((r for r in rows if r.get("lap") == lap), None)


def _now_row(rows: list) -> dict | None:
    return next((r for r in rows if r.get("action") == "PIT NOW"), None)


def _part_deltas(chosen: dict | None, alt: dict | None) -> list:
    """`(name, delta, phrase)` per term of the action table: how much more the
    alternative costs than the chosen action in tyre, position and traffic."""
    if not chosen or not alt:
        return []
    out = []
    for key, name in (("tyre_s", "tyre"), ("position_s", "track position"), ("traffic_s", "rejoin traffic"),
                      ("family_s", "plan-shape handicap")):
        a, c = _num(alt.get(key)), _num(chosen.get(key))
        if a is None or c is None:
            continue
        out.append((name, a - c))
    out.sort(key=lambda t: -abs(t[1]))
    return out


def _headline(dec: dict, row: dict) -> str:
    action = str(dec.get("action") or "—")
    conf = _pct(dec.get("confidence"))
    pos = dec.get("projected_position")
    pos_now = dec.get("projected_position_if_now")
    bits = []
    if action.startswith("PIT NOW") and pos:
        bits.append(f"projected to rejoin P{int(pos)}")
    elif action.startswith("WAIT") and dec.get("lap"):
        bits.append(f"box on lap {int(dec['lap'])}" + (f", rejoin P{int(pos)}" if pos else ""))
    elif action.startswith("BOX BY") and pos_now:
        bits.append(f"rejoin P{int(pos_now)} if taken this lap")
    elif action == "STAY OUT":
        bits.append("no further stop")
    d = _num(dec.get("delta_vs_alternative_s"))
    if d is not None and d > 0:
        bits.append(f"{d:.1f} s better than the next option")
    if conf:
        bits.append(f"{conf} confidence")
    who = row.get("driver")
    head = f"{who}: {action}" if who else action
    return head + (" — " + ", ".join(bits) if bits else "")


def explain_decision(plan: dict, row: dict | None = None, meta: dict | None = None) -> dict:
    """The principal reason for the live call and two to four supporting ones.

    `plan` is one car's plan dict (`plan["decision"]`, `plan["race_state"]`),
    `row` its field row (driver, compound, tyre age, position) and `meta` the
    snapshot meta (laps remaining, safety car).  Returns
    `{"headline", "principal", "reasons"}`; every string is built from a number
    in those three dictionaries, and a missing number simply drops its
    sentence."""
    plan = plan or {}
    row = dict(row or {})
    meta = dict(meta or {})
    dec = plan.get("decision") or {}
    rs = plan.get("race_state") or {}
    rows = _rows(plan)
    chosen = _row_at(rows, dec.get("lap")) if not str(dec.get("action", "")).startswith("STAY OUT") else None
    now_row = _now_row(rows)
    alts = [r for r in rows if r is not chosen]
    alt = min(alts, key=lambda r: _num(r.get("cost_s")) if _num(r.get("cost_s")) is not None else math.inf,
              default=None)
    reasons: list = []

    # 1. the term that decides it: what the best alternative costs more of
    principal = ""
    parts = _part_deltas(chosen, alt)
    if dec.get("held_by_hysteresis"):
        d = _num(dec.get("delta_vs_alternative_s"))
        h = _num(dec.get("hysteresis_s"))
        if d is not None and h is not None:
            principal = (f"The call is held: {alt.get('action', 'the alternative') if alt else 'the alternative'} "
                         f"is only {abs(d):.2f} s better, inside the {h:.1f} s a call is held for")
    if not principal and parts and alt is not None:
        name, d = parts[0]
        if d > 0:
            principal = f"{alt.get('action', 'the alternative')} costs {d:.1f} s more of {name}"
        else:
            principal = (f"{alt.get('action', 'the alternative')} is {abs(d):.1f} s better on {name}, "
                         f"and worse everywhere else")
    if not principal and chosen is not None:
        c = _num(chosen.get("cost_s"))
        principal = (f"{chosen.get('action')} has the lowest expected race time"
                     + (f" ({c:.1f} s)" if c is not None else ""))
    if principal:
        reasons.append(principal)

    # 2. the tyre: what staying out costs on this set
    if now_row is not None:
        later = [r for r in rows if (r.get("lap") or 0) > (now_row.get("lap") or 0)
                 and _num(r.get("tyre_s")) is not None]
        ref = None
        for r in later:
            if (r.get("lap") or 0) - (now_row.get("lap") or 0) == 2:
                ref = r
        ref = ref or (later[-1] if later else None)
        t_ref = None if ref is None else _num(ref.get("tyre_s"))
        t_now = _num(now_row.get("tyre_s"))
        if t_ref is not None and t_now is not None:
            d = t_ref - t_now
            k = int((ref.get("lap") or 0) - (now_row.get("lap") or 0))
            verb = "costs" if d >= 0 else "saves"
            reasons.append(f"Staying out {_laps(k)} {verb} {abs(d):.1f} s of tyre against boxing now")

    # 3. who we are racing, and what it does if we box
    rivals = rs.get("rivals") or []
    if rivals:
        r0 = rivals[0]
        who = r0.get("driver") or r0.get("driver_number")
        gap = _num(r0.get("gap_s"))
        if who and gap is not None:
            side = "ahead" if gap > 0 else "behind"
            txt = f"{who} {side} is {abs(gap):.1f} s away"
            if r0.get("on_track"):
                txt += " on track"
            age, comp = _num(r0.get("tyre_age")), r0.get("compound")
            if comp and age is not None:
                txt += f" on a {int(age)}-lap-old {comp}"
            sl = r0.get("stop_lap_median")
            if sl:
                txt += f" and its own stop is due on lap {int(sl)}"
            cov = r0.get("if_cover") or None
            pc = _num((cov or {}).get("p_cover"))
            pa = _num((cov or {}).get("p_ahead_after"))
            dp = _num((cov or {}).get("places_delta"))
            if pc is not None and pa is not None:
                txt += (f"; it covers a stop now with probability {pc:.2f}, and then holds the place "
                        f"with probability {pa:.2f}")
                if dp is not None and dp > 0:
                    txt += f" ({dp:.2f} places)"
            reasons.append(txt)

    # 4. where we come out
    rj = dec.get("rejoin") or dec.get("rejoin_if_now") or {}
    pos = dec.get("projected_position")
    at_lap = bool(pos and dec.get("rejoin"))
    pos = pos if at_lap else dec.get("projected_position_if_now")
    if pos:
        txt = (f"{dec.get('action', 'The stop')} is projected to rejoin P{int(pos)}" if at_lap
               else f"Boxing this lap is projected to rejoin P{int(pos)}")
        g, who = _num(rj.get("gap_ahead_s")), rj.get("ahead")
        if g is not None and who:
            txt += f", {g:.1f} s behind {who}"
        band = rj.get("band") or []
        if band:
            txt += f", with {len(band)} car{'' if len(band) == 1 else 's'} inside the traffic band"
        conf = _pct(dec.get("confidence"))
        if conf:
            txt += f" ({conf} confidence)"
        reasons.append(txt)

    # 5. tyre-life risk: reported, never decisive
    life = dec.get("life") or {}
    pc = _num(life.get("p_cliff_before_stop"))
    p50 = _num(life.get("life_p50"))
    if pc is not None and p50 is not None and pc >= LIFE_RISK_SHOW_P:
        age, comp = _num(life.get("tyre_age")), life.get("compound") or row.get("compound")
        head = (f"The {comp} is {int(age)} laps old; " if (comp and age is not None) else "")
        reasons.append(head + f"the drop-off is {p50:.0f} laps away (p10 {_num(life.get('life_p10')) or 0:.0f}, "
                              f"p90 {_num(life.get('life_p90')) or 0:.0f}); P(drop-off before the stop) "
                              f"{pc:.2f} — shown, not decisive")
    if meta.get("sc_active"):
        reasons.append("A safety car is out: the pit loss is discounted this lap")
    return {"headline": _headline(dec, row), "principal": principal, "reasons": reasons[:MAX_REASONS]}


# --------------------------------------------------------------------------
# the pre-race decision table
# --------------------------------------------------------------------------


def explain_actions(decisions_row: dict | None) -> dict:
    """The same vocabulary for one lap of the pre-race table.

    `decisions_row` is one entry of `meta["race_state"]["decisions"]`:
    `{"lap", "actions": [...], "decision", "decision_lap"}` with each action
    carrying `cost_s`, `delta_s` and the `tyre_s` / `position_s` /
    `places_ahead` parts.  Returns `{"headline", "principal", "reasons",
    "lines"}` - `lines` is one sentence per action, in the table's order."""
    d = decisions_row or {}
    rows = [r for r in (d.get("actions") or []) if r.get("legal")]
    lap = d.get("lap")
    pick = d.get("decision")
    pick_lap = d.get("decision_lap")
    chosen = next((r for r in rows if r.get("action") == pick), None)
    alts = [r for r in rows if r is not chosen]
    alt = min(alts, key=lambda r: _num(r.get("cost_s")) if _num(r.get("cost_s")) is not None else math.inf,
              default=None)
    lines = []
    for r in rows:
        bits = []
        for key, name in (("tyre_s", "of tyre"), ("position_s", "of track position"),
                          ("traffic_s", "of rejoin traffic")):
            v = _s(r.get(key))
            if v is not None:
                bits.append(f"{v} {name}")
        pl = _num(r.get("places_ahead"))
        if pl is not None:
            bits.append(f"{pl:.2f} rivals ahead after the round")
        dv = _num(r.get("delta_s"))
        delta = ("the cheapest action" if (dv is not None and abs(dv) < 0.005)
                 else (f"{dv:+.2f} s against the best" if dv is not None else ""))
        lines.append(f"{r.get('action')} (lap {r.get('lap')}): " + delta
                     + (" — " + ", ".join(bits) if bits else ""))
    reasons = []
    principal = ""
    parts = _part_deltas(chosen, alt)
    if parts and alt is not None:
        name, dv = parts[0]
        principal = (f"{alt.get('action')} costs {dv:.1f} s more of {name}" if dv > 0
                     else f"{alt.get('action')} is {abs(dv):.1f} s better on {name} and worse elsewhere")
        reasons.append(principal)
    if chosen is not None:
        t, p = _num(chosen.get("tyre_s")), _num(chosen.get("position_s"))
        if t is not None and p is not None:
            reasons.append(f"{chosen.get('action')} pays {t:.1f} s of tyre and {p:.1f} s of track position")
    if alt is not None and chosen is not None:
        ca, cc = _num(alt.get("cost_s")), _num(chosen.get("cost_s"))
        if ca is not None and cc is not None:
            reasons.append(f"the margin over {alt.get('action')} is {ca - cc:+.2f} s of expected race time")
    head = (f"Lap {int(lap)}: {pick}" if (lap is not None and pick) else (pick or "—"))
    if pick_lap is not None:
        head += f" (stop on lap {int(pick_lap)})"
    return {"headline": head, "principal": principal, "reasons": reasons[:MAX_REASONS], "lines": lines}
