"""Decision cards: a plan priced on the forecast's own draws, with its windows,
switch triggers, undercut exposure and safety-car rule - and the commit of the
model's own per-car plans for the two Haas drivers.

The Plan builder tab built these numbers inline; they live here so that the
supervisor can commit the weekend model's per-car plans after every refit
(`scripts/85_plans.py`) and the live race view tracks the cars against them
without anyone having to click.  Nothing here reads race data for the target
weekend: the model is the outlook's (sealed practice fit, or the prior stage),
the plans are the weekend fit's per-driver search.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import numpy as np

from src import objective as objlib
from src import plans as plan_store
from src import strategy as strat
from src.config import DATA_PROCESSED, Event, get_event
from src.outlook import load_model, load_outlook
from src.tyre import TyreModel

HAAS_DRIVERS = ("OCO", "BEA")
_FROM_LETTER = {"S": "SOFT", "M": "MEDIUM", "H": "HARD"}
_LABEL = re.compile(r"^(\d+)-stop ([SMH](?:-[SMH])*)(?: @ ([\d,]+))?$")


def label_words(label) -> str:
    """'2-stop M-H-H @ 17,37' -> '2-stop M → H → H · laps 17, 37' (plain words, no app import)."""
    m = _LABEL.match(str(label or "").strip())
    if not m:
        s = str(label or "—")
        return s[:1].upper() + s[1:]
    seq = " → ".join(m.group(2).split("-"))
    laps = [int(x) for x in m.group(3).split(",")] if m.group(3) else []
    s = f"{m.group(1)}-stop {seq}"
    if laps:
        s += (" · laps " if len(laps) > 1 else " · lap ") + ", ".join(str(p) for p in laps)
    return s


# --------------------------------------------------------------------------
# The objective: a plan priced exactly as the forecast's own search priced it
# --------------------------------------------------------------------------


def objective_json(outlook: dict) -> str:
    """The forecast's own cost terms, as a cache key and a payload.

    Everything the forecast's search charged beyond the tyre and the pit lane:
    the exposure weight on the stops after the first, the plan-family weight and
    its counts, the circuit's close-following cost, the start-tyre step and -
    the one that times the first stop - each plan family's track-position term
    by lap (`strategy.evaluate_plans(race_state_terms=...)`)."""
    st = outlook.get("strategy") or {}
    obj = outlook.get("objective") or {}
    cal = outlook.get("calibration") or {}
    # An older forecast on disk carries only the *display* copy of the family
    # counts (its sequences truncated, no stop counts), which would price the
    # family term differently from the search that wrote it.  Charge it only
    # from a complete set of counts; otherwise leave the family term out.
    pp = obj.get("plan_prior") or outlook.get("plan_prior") or {}
    if not (pp.get("n") and pp.get("sequences") and pp.get("stops")):
        pp = {}
    return json.dumps({"undercut_lambda": float(obj.get("undercut_lambda", cal.get("undercut_lambda", 0.0)) or 0.0),
                       "plan_prior": pp,
                       "plan_prior_tau_s": float(obj.get("plan_prior_tau_s", cal.get("plan_prior_tau_s", 0.0)) or 0.0)
                       if pp else 0.0,
                       "traffic_s_per_lap": float(obj.get("traffic_s_per_lap", cal.get("dirty_air_used", 0.0)) or 0.0),
                       "grid_penalty_s": float(obj.get("grid_penalty_s", cal.get("grid_start_penalty_s", 0.0)) or 0.0),
                       "race_state_terms": st.get("race_state_terms") or {}},
                      sort_keys=True, default=float)


def eval_kw(obj_json: str) -> dict:
    """`objective_json` back into `evaluate_plans` keywords."""
    o = json.loads(obj_json or "{}")
    kw = {k: o[k] for k in ("undercut_lambda", "plan_prior_tau_s", "traffic_s_per_lap", "grid_penalty_s") if k in o}
    if o.get("plan_prior"):
        kw["plan_prior"] = o["plan_prior"]
    terms = objlib.terms_from_json(o.get("race_state_terms"))
    if terms:
        kw["race_state_terms"] = terms
    return kw


def window_kw(obj_json: str, plan: dict) -> dict:
    """...and into `pit_window_model` keywords, with this plan's own family term."""
    o = json.loads(obj_json or "{}")
    kw = {k: o[k] for k in ("undercut_lambda", "traffic_s_per_lap") if k in o}
    terms = objlib.terms_from_json(o.get("race_state_terms"))
    lab = objlib.group_label(plan.get("compounds"), plan.get("pit_laps"))
    if lab and lab in terms:
        kw["race_state_term"] = terms[lab]
    return kw


# --------------------------------------------------------------------------
# The card's numbers
# --------------------------------------------------------------------------


def card_numbers(model: TyreModel, ev: Event, plan: dict, others: list, pit_loss: float,
                 alloc: dict, caps: dict, obj_json: str = "{}") -> dict | None:
    """Windows, switch triggers, undercut exposure and the expected wear at each stop.

    `plan` is `{compounds, pit_laps[, push]}`; `others` the alternatives whose
    crossover with it is wanted (`{compounds, pit_laps, label}`)."""
    ekw = eval_kw(obj_json)
    tbl, det = strat.evaluate_plans(model, ev, [plan], pit_loss, allocation=alloc, stint_cap=caps, **ekw)
    d = det[0]
    if not d.get("valid"):
        return None
    p_use = float(d["push"])
    pw = strat.pit_window_model(model, ev, {"compounds": plan["compounds"], "pit_laps": plan["pit_laps"], "push": p_use},
                                pit_loss, max_stint=(caps or None), push=p_use, **window_kw(obj_json, plan))
    windows = strat.windows_from_sweep(pw, plan)
    switches = []
    for o in others:
        cx = strat.deg_crossover(model, ev, {"compounds": plan["compounds"], "pit_laps": plan["pit_laps"]},
                                 {"compounds": o["compounds"], "pit_laps": o["pit_laps"]}, pit_loss,
                                 allocation=alloc, stint_cap=caps, **ekw)
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
            "stint_lens": d["stint_lens"], "race_state_s": float(d.get("race_state_s", 0.0))}


def playbook_ranges(model: TyreModel, ev: Event, plan: dict, pit_loss: float, alloc: dict, caps: dict) -> list:
    pb = strat.sc_playbook(model, ev, plan, pit_loss, push=plan.get("push"), allocation=alloc, stint_cap=caps)
    return strat.playbook_ranges(pb)


def triggers(nums: dict, ranges: list, label_fn=label_words) -> tuple[dict, dict]:
    """Short trigger lines for the card, and the plan B block the live view tracks."""
    out = {}
    for i, w in enumerate(nums["windows"]):
        ex = nums["exposure"][i] if i < len(nums["exposure"]) else None
        line = f"Stop {w['stop']}: lap {w['recommended']} (window {w['lo']}–{w['hi']})"
        if ex:
            line += (f". Tyre life used above {min(ex['wear_p90'] + 0.05, 1.0):.0%} before lap "
                     f"{max(w['lo'], w['recommended'] - 3)}: stop on lap {w['lo']}")
            line += (f". A car within {max(ex['gain_3'], 0):.1f} s behind on new {ex['new_compound'].title()}s gets "
                     f"ahead in 3 laps if you stay out past lap {w['recommended']}" if ex["gain_3"] > 0 else
                     ". No undercut risk at this tyre age")
        out[f"stop_{w['stop']}"] = line
    alt_block = {}
    for s in nums["switches"]:
        name = label_fn(s["label"])
        if s.get("b_better_at_base"):
            txt = f"{name} is already {-s['delta_at_base_s']:.1f} s faster at the forecast's tyre wear"
        elif s.get("mult"):
            txt = (f"Switch to {name} if this car's tyre wear reaches ×{s['mult']:.2f} the forecast "
                   f"({s['delta_at_base_s']:+.1f} s slower at ×1.0)")
        else:
            txt = f"{name} never overtakes this plan (tested up to ×2.6 tyre wear)"
        out[f"switch_{s['label']}"] = txt
        if not alt_block and s.get("mult") and not s.get("b_better_at_base"):
            alt_block = {"label": s["label"], "delta_s": s["delta_at_base_s"], "switch_mult": s["mult"],
                         "when": "Watch the 'Wear vs forecast' tile for this car on the Now tab"}
    if ranges:
        box = [r for r in ranges if r["verdict"] == "PIT"]
        stay = [r for r in ranges if r["verdict"] == "STAY"]
        out["safety_car"] = (("Safety car: box on laps " + ", ".join(f"{r['from']}–{r['to']}" for r in box)
                              + f" (saves ~{np.mean([r['gain_s'] for r in box]):.0f} s)"
                              + ("; stay out on laps " + ", ".join(f"{r['from']}–{r['to']}" for r in stay) if stay else ""))
                             if box else "Safety car: stay on plan whenever it comes")
    return out, alt_block


# --------------------------------------------------------------------------
# The model's own per-car plans, committed
# --------------------------------------------------------------------------


def _plan_of(row: dict) -> dict | None:
    comps = row.get("compounds")
    comps = [c for c in str(comps).split("-") if c] if not isinstance(comps, list) else [str(c) for c in comps]
    pits = row.get("pit_laps")
    if not isinstance(pits, list):
        try:
            pits = json.loads(str(pits))
        except Exception:
            return None
    if not comps or len(comps) != len(pits) + 1:
        return None
    return {"compounds": [c.upper() for c in comps], "pit_laps": [int(p) for p in pits],
            "push": float(row.get("push", 1.0) or 1.0)}


def model_plans(event_key: str) -> dict:
    """`{driver: plan}` from the weekend fit's per-driver search, plus the field's under `'*'`."""
    p = DATA_PROCESSED / f"weekend_{event_key}.json"
    if not p.exists():
        return {}
    w = json.loads(p.read_text())
    out = {}
    for r in w.get("per_driver") or []:
        pl = _plan_of(r)
        if pl:
            out[str(r.get("driver"))] = pl
    bp = (w.get("strategy") or {}).get("best_plan") or {}
    if bp.get("compounds") and bp.get("pit_laps"):
        out["*"] = {"compounds": [str(c) for c in bp["compounds"]], "pit_laps": [int(x) for x in bp["pit_laps"]],
                    "push": float(bp.get("push", 1.0) or 1.0)}
    return out


def commit_model_plans(event_key: str, drivers=HAAS_DRIVERS, *, note: str = "") -> list:
    """Commit the weekend model's own plan for each driver as a decision card.

    The card's windows, wear, undercut exposure, switch triggers and safety-car
    rule are priced on the outlook's draws - the same numbers the Plan builder
    shows - so a card committed here and one committed by hand read the same.
    Returns the committed plans (empty when the weekend has no model yet)."""
    ev = get_event(event_key)
    plans = model_plans(event_key)
    outlook = load_outlook(event_key) or {}
    model = load_model(event_key)
    if not plans or model is None or not outlook:
        return []
    alloc = outlook.get("allocation") or {}
    caps = outlook.get("stint_cap") or {}
    pit_loss = float(outlook.get("pit_loss_s", 22.0))
    obj_json = objective_json(outlook)
    alt = ((outlook.get("alternatives") or {}).get("plan_b") or {})
    w = json.loads((DATA_PROCESSED / f"weekend_{event_key}.json").read_text())
    sealed = w.get("sealed_file", "")
    out = []
    for drv in drivers:
        plan = plans.get(drv) or plans.get("*")
        if not plan:
            continue
        others = []
        team = plans.get("*")
        if team and (team["compounds"], team["pit_laps"]) != (plan["compounds"], plan["pit_laps"]):
            others.append({**team, "label": plan_store.short_label(team["compounds"], team["pit_laps"])})
        if alt.get("compounds") and alt.get("pit_laps") and \
                (list(alt["compounds"]), [int(x) for x in alt["pit_laps"]]) != (plan["compounds"], plan["pit_laps"]):
            others.append({"compounds": list(alt["compounds"]), "pit_laps": [int(x) for x in alt["pit_laps"]],
                           "label": alt.get("label") or plan_store.short_label(alt["compounds"], alt["pit_laps"])})
        nums = card_numbers(model, ev, plan, others, pit_loss, alloc, caps, obj_json)
        if nums is None:
            continue
        ranges = playbook_ranges(model, ev, {**plan, "push": nums["push"]}, pit_loss, alloc, caps)
        trig, alt_block = triggers(nums, ranges)
        src = (f"weekend model {sealed or event_key} · per-car plan for {drv} · "
               f"{outlook.get('stage_label', '')} · {datetime.now(timezone.utc).isoformat(timespec='minutes')}")
        out.append(plan_store.commit_plan(event_key, compounds=plan["compounds"], pit_laps=plan["pit_laps"],
                                          push=nums["push"], driver=drv, note=note, windows=nums["windows"],
                                          triggers=trig, alternative=alt_block, n_race_laps=ev.n_race_laps,
                                          source=src))
    return out
