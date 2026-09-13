"""AI race engineer — LLM-grounded briefing and Q&A (P2).

This is the plan's P2 component and first on the cut line: nothing else depends
on it.  (The plan specifies a Claude-grounded engineer; this build uses Gemini
Flash at the operator's request.  Nothing else in the pipeline is affected —
no model output ever feeds back into a fit.)

**Remit.** It is Haas's tyre-degradation engineer, not a general commentator:
it engineers #31 Ocon and #87 Bearman, leads on what the tyres are doing, and
treats the rest of the grid as the field those two race in.  The `haas` block
of the fact sheet carries each car separately — the two drivers wear tyres at
measurably different rates, and averaging them into one team number would throw
away the only thing a pit wall cares about here.

**Grounding.** The model is never asked to estimate anything. Every number it
is allowed to say is computed by the pipeline and handed to it as a fact sheet;
the system prompt forbids inventing figures and requires it to say so when the
fact sheet does not cover a question. An LLM guessing tyre degradation numbers
would undo the entire point of the rest of this codebase.  Each per-car
quantity travels with the evidence behind it (`n_evidence`, `shrink_weight`,
`source`, and what the driver / team / field levels each said), because on a
thin weekend a car's "own" rate is mostly its team-mate's, and a briefing that
hides that is worse than no briefing.

**Provider.** Gemini, keyed by `GEMINI_API_KEY` in the project's `.env` (loaded
by `src.config`).  Without a key, `brief()` and `ask()` fall back to a
deterministic briefing assembled from the same fact sheet. The
dashboard therefore behaves identically with or without an API key, which is
what keeps it safe to demo.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

from src.config import DATA_PROCESSED, Event, get_event
from src.haascar import HAAS_DRIVER_META, HAAS_DRIVERS, HAAS_TEAM

log = logging.getLogger("degless.engineer")

# Flash, per the operator's request.  Overridable without a code change; note
# that gemini-2.0-flash has been shut down, so an older default would fail.
DEFAULT_MODEL = "gemini-3.7-flash"

# `max_output_tokens` covers the model's *thinking* as well as its answer, and
# on this fact sheet the thoughts alone run to about 1900 tokens: a 2000-token
# budget left 73 for the briefing and every answer stopped mid-sentence.  Six
# paragraphs need roughly 1200, so the budget has to carry both.
DEFAULT_MAX_TOKENS = 8000


def max_tokens() -> int:
    try:
        return max(256, int(os.environ.get("ENGINEER_MAX_TOKENS", DEFAULT_MAX_TOKENS)))
    except ValueError:
        return DEFAULT_MAX_TOKENS


def model_name() -> str:
    if provider() == "claude":
        return os.environ.get("CLAUDE_MODEL", "claude-opus-5")
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

SYSTEM = """You are the tyre-degradation race engineer on the Haas F1 Team pit \
wall. You engineer two cars and only two: car 31, Esteban Ocon (OCO), and car \
87, Ollie Bearman (BEA). You are briefing your own strategist on what the \
tyres are doing and what that means for each car's stop.

The analysis in front of you comes from a tool called degless, which fits tyre \
degradation curves to practice long-run data under a 2026-regulation fuel-mass \
prior, seals its predictions before the race, and scores them against the race \
afterwards. The FACT SHEET below is its output. The `haas` block is your own \
two cars; everything else is the field they race in.

Rules you must follow:
1. Every number you state must come from the FACT SHEET below. Never estimate, \
interpolate, or recall a number from elsewhere. If the fact sheet does not \
contain something, say plainly that the analysis does not cover it. In \
particular, never derive a tyre life, a stint length or a crossover lap the \
fact sheet does not already give you - a degradation rate is not a tyre life \
until the model has capped it.
2. Lead with the tyre. The rate it is degrading at, how much of the set is \
gone, what that costs per lap, and what it does to the stop - in that order - \
before any talk of track position.
3. Engineer the two cars separately. OCO and BEA wear tyres at different rates \
and the fact sheet gives each car its own; never quote one car's number for \
the other, and never average them into a Haas number the fact sheet does not \
state. Name the driver every time you give a number.
4. Say where each number comes from. The tyre quantities under each car carry \
`n_evidence`, `shrink_weight` and `source`: source "own" is that driver's own \
laps, "team" means the value has been shrunk toward the other car, "field" \
toward the whole grid, and `own`/`team_value`/`field_value` show what each \
level said. A car with a handful of clean laps is being described mostly by \
its team-mate and the field - say so rather than presenting the number as that \
driver's own measured rate.
5. Report uncertainty wherever the fact sheet gives it. A credible interval is \
not decoration, and the race-history rate factor carries an `ln_sd` for a \
reason.
6. Do not oversell. If the validation shows the model is biased, or the race \
has not been run and scored yet, say so in the same breath as any \
recommendation that depends on it.
7. Be concise and concrete, the way a race engineer on the radio is concise. \
No preamble, no restating the question."""


@dataclass
class EngineerReply:
    text: str
    grounded: bool = True
    source: str = "gemini"  # "claude" | "gemini" | "offline"
    model: str = ""


# --------------------------------------------------------------------------
# Fact sheet
# --------------------------------------------------------------------------


# The provenance fields haascar attaches to every shrunk quantity.  The prose
# `note`/`detail` that travel with them are dropped: rule 4 of the system prompt
# already says how to read `source`, and the fact sheet is large enough.
_SHRUNK_KEEP = ("value", "n_evidence", "shrink_weight", "source",
                "own", "team_value", "field_value", "units")


def _shrunk(x):
    """One of haascar's driver->team->field estimates, with its provenance."""
    if not isinstance(x, dict):
        return x
    return {k: v for k, v in x.items() if k in _SHRUNK_KEEP}


def _haas_facts(m: dict, live: dict) -> dict:
    """The two cars this pit wall actually engineers.

    Kept per driver rather than per team on purpose: OCO and BEA degrade tyres
    at measurably different rates, and a weekend thin on clean laps fills the
    gap by shrinking a car toward its team-mate.  Each quantity therefore
    travels with the evidence behind it, so a team-shaped number is never read
    back as the driver's own.
    """
    h = m.get("haas") or {}
    cars = h.get("cars") or {}
    if not cars:
        return {}
    live_by_code = {r.get("driver"): r for r in (live.get("field") or []) if r.get("driver")}
    out = {}
    for code in HAAS_DRIVERS:
        c = cars.get(code) or {}
        state, plan = (c.get("state") or {}), (c.get("plan") or {})
        who = HAAS_DRIVER_META.get(code, {})
        out[code] = {
            "driver": code,
            "number": who.get("number"),
            "name": who.get("name"),
            "teammate": state.get("teammate"),
            "evidence_this_weekend": state.get("evidence"),
            "tyre": {
                "deg_rate_by_compound_s_per_lap": _shrunk(state.get("deg_rate_by_compound")),
                "effective_race_rate_s_per_lap": plan.get("eff_rate"),
                "practice_deviation_s_per_lap": plan.get("practice_dev_s_per_lap"),
                "race_history_rate_factor": state.get("race_factor"),
                "age_sensitivity": _shrunk(state.get("age_sensitivity")),
                "warmup_s": _shrunk(state.get("warmup_s")),
                "traffic_sensitivity": _shrunk(state.get("traffic_sensitivity")),
                "sector_degradation": _shrunk(state.get("sector_deg")),
                "push_response": _shrunk(state.get("push_response")),
            },
            "pace": {
                "offset_vs_field_s_per_lap": _shrunk(state.get("pace_offset_s")),
                "vs_teammate_s_per_lap": _shrunk(state.get("pace_vs_teammate_s")),
                "consistency_s": _shrunk(state.get("consistency_s")),
            },
            "plan": plan,
            "live_now": live_by_code.get(code),
        }
    return {
        "team": h.get("team", HAAS_TEAM),
        "source": h.get("source"),
        "evidence_rule": h.get("evidence_rule"),
        "field_reference": h.get("field"),
        "difference_between_the_cars": h.get("explanation"),
        "cars": out,
        "note": ("`tyre.deg_rate_by_compound_s_per_lap` is what this weekend's practice "
                 "measured for the car; `effective_race_rate_s_per_lap` is the rate the "
                 "plan is built on, which is that rate scaled by the car's race-history "
                 "factor. Tyre lives and stint caps are field-level and live under "
                 "`strategy.life`; there is no per-driver tyre life in this analysis. "
                 "A car's committed decision card, where one has been committed, is in "
                 "the top-level `committed_plans`, keyed by `driver`."),
    }


def fact_sheet(event: Event | str) -> dict:
    """Everything the model is allowed to talk about, straight from the pipeline."""
    ev = get_event(event) if isinstance(event, str) else event
    p = DATA_PROCESSED / f"meta_{ev.key}.json"
    if not p.exists():
        p = DATA_PROCESSED / f"weekend_{ev.key}.json"
    outlook = _outlook_facts(ev)
    if not p.exists():
        if not outlook:
            raise FileNotFoundError(f"no pipeline output or outlook for {ev.key}; run the pipeline or the outlook")
        m = _empty_meta(ev, outlook)
    else:
        m = json.loads(p.read_text())
    if "score" not in m:   # a pre-race weekend model: no race half yet
        m = dict(m)
        m["score"] = {"note": "race not yet run; nothing scored"}
        m.setdefault("prior_sensitivity", {})
        m.setdefault("counterfactual_top", [])
        m["pit_stops_measured"] = 0
        m.setdefault("mixedlm", {"slopes": {}})
        m["bayes"] = {**m.get("bayes", {}), "k_track_laponly_rel_sd": float("nan")}
    # -- live session, when one is running -----------------------------------
    live = {}
    try:
        from src.live.store import read_current, read_snapshot

        sk = read_current()
        snap = read_snapshot(sk) if sk else {}
        if snap:
            meta_l = snap.get("meta", {})
            live = {"session": meta_l.get("session"), "status": meta_l.get("status"),
                    "lap": meta_l.get("lap_count"), "track_status": meta_l.get("track_status"),
                    "pit_loss_s": meta_l.get("pit_loss_s"), "regime_multiplier": meta_l.get("regime_multiplier"),
                    "field": [{k: v for k, v in r.items() if k in (
                        "position", "driver", "compound", "tyre_age", "gap_leader", "interval", "wear",
                        "p_past_cliff", "laps_to_cliff_p50", "deg_now_s_per_lap", "m_mean", "cliff_alarm")}
                              | {"plan": {k: v for k, v in (r.get("plan") or {}).items() if k in (
                                  "best", "window_lo", "window_hi", "delta_box_now_s", "delta_stay_out_s", "win_prob")},
                                 "undercut_threat": ((r.get("undercut") or {}).get("threat") or {}).get("driver"),
                                 "undercut_p3": ((r.get("undercut") or {}).get("threat") or {}).get("p_undercut_3lap")}
                              for r in snap.get("field", [])[:22]],
                    "alerts": [a.get("text") for a in snap.get("alerts", [])[-12:]],
                    "practice_board": snap.get("board", [])[:15], "pooled": snap.get("pooled", {})}
    except Exception:
        live = {}

    committed = _committed_plans(ev)
    return {
        "event": m.get("event_name", ev.name),
        "race_laps": m.get("n_race_laps", ev.n_race_laps),
        "data": {
            "raw_practice_laps": m.get("n_raw_laps"),
            "clean_practice_laps": m.get("n_clean_laps"),
            "compound_counts": m.get("compound_counts", []),
            "fitted_on": "practice sessions only (firewall enforced)",
        },
        "physics_2026": m.get("physics", {}),
        "prior_sensitivity_slope_s_per_lap": m.get("prior_sensitivity", {}),
        "degradation": m.get("bayes", {}).get("slopes", []),
        "fuel_sensitivity_k_track": {
            "mean_s_per_kg": m["bayes"]["k_track_mean"],
            "sd_s_per_kg": m["bayes"]["k_track_sd"],
            "relative_sd": m["bayes"]["k_track_rel_sd"],
            "note": ("the prior's relative SD was 0.25; a posterior close to that "
                     "means the lap-time channel carried almost no information "
                     "about the fuel/age split"),
        },
        "convergence": {
            "max_rhat": m["bayes"]["max_rhat"],
            "divergences": m["bayes"]["n_divergences"],
        },
        "mixedlm_baseline_slopes": m.get("mixedlm", {}).get("slopes", {}),
        "compound_ladder": {
            **{k: v for k, v in m.get("compound_ladder", {}).items()
               if k in ("pace_step_s", "label", "derivation", "table",
                        "fitted_offsets", "unladdered_slopes", "ordered")},
            "note": ("compound pace and degradation ordering is imposed by "
                     "construction (softer is always quicker and degrades "
                     "faster); only the size of each step is fitted, because "
                     "practice data cannot identify the ordering"),
        },
        "practice_to_race_regime": {
            **{k: v for k, v in m.get("regime", {}).items()
               if k in ("ratio", "p05", "p95", "measured", "label",
                        "sources", "derivation")},
            "note": ("race stints degrade more slowly than practice long runs "
                     "because drivers manage tyres. This factor is measured on "
                     "OTHER weekends' races, never this one's, and it is a "
                     "CROSS-CHECK, not an input: the strategy optimiser chooses "
                     "a push level and the implied factor falls out of it"),
        },
        "fuel_load_on_wear": m.get("load_effect", {}),
        "validation": m.get("score", {}),
        "pit_loss_s": m.get("pit_loss_s"),
        "strategy": m.get("strategy", {}),
        "counterfactual_top": m.get("counterfactual_top", []),
        "gates": m.get("gates", []),
        "live_session": live,
        "outlook": outlook,
        "committed_plans": committed,
        "haas": _haas_facts(m, live),
        "limits": {
            "age_support_laps": m.get("age_support_laps"),
            "age_support_by_compound": m.get("age_support_by_compound"),
            "max_stint_considered": m.get("max_stint_laps"),
            "max_stints_per_compound": m.get("max_stints_per_compound"),
            "not_modelled": ("track position as a race-long state, the "
                             "starting-tyre choice, and the undercut battle "
                             "with specific cars are NOT priced. Dirty air "
                             "after a stop and the safety car ARE priced. The "
                             "claim is the stop count and the stint lengths, "
                             "not the compound running order, whose best plans "
                             "are within about a second of each other"),
            "note": ("degradation beyond each compound's own practice age "
                     "support is extrapolation and is capped in the search"),
        },
    }


def _outlook_facts(ev: Event) -> dict:
    """The outlook, trimmed to what a briefing can use: the plan and its
    probabilities, tyre lives, the alternatives and their switch triggers, the
    scenario matrix, the practice programme and the safety-car playbook."""
    try:
        from src.outlook import load_outlook

        o = load_outlook(ev.key)
    except Exception:
        o = None
    if not o:
        return {}
    st = o.get("strategy") or {}
    sc = o.get("scenarios") or {}
    return {
        "what_it_is": ("the best current strategy picture for this race from everything known so far; "
                       "it updates as practice runs and after every refit"),
        "stage": o.get("stage_label"), "updated_utc": o.get("updated_utc"), "sources": o.get("sources", []),
        "n_practice_laps": o.get("n_practice_laps"), "live": {k: v for k, v in (o.get("live") or {}).items()
                                                               if k in ("session_name", "used", "n_long_runs", "track_temp_c", "note")},
        "plan": st.get("best"), "plan_detail": st.get("best_plan"), "p_stops": st.get("p_stops"),
        "pit_windows": st.get("pit_windows"), "push": st.get("push"), "implied_regime": st.get("implied_regime"),
        "tyres": st.get("life"), "by_stops": [{k: v for k, v in b.items() if k in ("label", "delta_s", "win_prob_any", "stint_lens", "push")}
                                             for b in st.get("by_stops", [])],
        "by_start_compound": [{k: v for k, v in b.items() if k in ("start", "label", "delta_s", "win_prob_any")}
                              for b in st.get("by_start", [])],
        "alternatives": {k: {kk: vv for kk, vv in v.items() if kk in ("label", "delta_s", "switch_mult", "switch_direction", "stint_lens")}
                         for k, v in (o.get("alternatives") or {}).items()},
        "scenarios": {"robust_plan": sc.get("robust"), "robust_max_regret_s": sc.get("robust_max_regret_s"),
                      "base_plan_max_regret_s": sc.get("base_max_regret_s"),
                      "matrix": [{k: v for k, v in c.items() if k in ("deg_mult", "pit_delta_s", "best", "n_stops", "base_plan_regret_s")}
                                 for c in sc.get("matrix", [])]},
        "value_of_information": {"evpi_s": (o.get("voi") or {}).get("evpi_s"),
                                 "practice_programme": [r.get("text") for r in o.get("programme", [])]},
        "safety_car_playbook": [{k: v for k, v in r.items() if k in ("from", "to", "verdict", "gain_s", "continuation")}
                                for r in (o.get("sc_playbook") or {}).get("ranges", [])],
        "pit_loss_s": o.get("pit_loss_s"), "pit_loss_source": o.get("pit_loss_source"),
        "stint_cap": o.get("stint_cap"), "allocation": o.get("allocation"), "regime": o.get("regime"),
        "history": o.get("history"),
    }


def _committed_plans(ev: Event) -> list:
    try:
        from src.plans import load_plans

        return [{k: v for k, v in p.items() if k in ("driver", "label", "compounds", "pit_laps", "push", "windows",
                                                     "triggers", "alternative", "note", "committed_utc")}
                for p in load_plans(ev.key)]
    except Exception:
        return []


def _empty_meta(ev: Event, outlook: dict) -> dict:
    """A meta-like shell for a weekend that has no practice fit yet."""
    nan = float("nan")
    return {"event_name": ev.name, "n_race_laps": ev.n_race_laps, "n_raw_laps": 0, "n_clean_laps": 0,
            "compound_counts": [], "physics": {}, "prior_sensitivity": {},
            "bayes": {"slopes": [], "k_track_mean": nan, "k_track_sd": nan, "k_track_rel_sd": nan,
                      "max_rhat": nan, "n_divergences": 0},
            "mixedlm": {"slopes": {}}, "compound_ladder": {}, "regime": outlook.get("regime") or {},
            "load_effect": {}, "pit_loss_s": outlook.get("pit_loss_s"), "strategy": {}, "gates": [],
            "max_stint_laps": outlook.get("stint_cap"), "age_support_by_compound": {}}


# --------------------------------------------------------------------------
# Offline fallback
# --------------------------------------------------------------------------


def _offline_note(reason: str | None, question: str | None, what: str = "briefing") -> str:
    """The italic footer saying why this is the offline text and not a live answer."""
    why = reason or "no GEMINI_API_KEY in .env"
    tail = (f" rather than an answer to \u201c{question}\u201d" if question else "")
    fix = "" if reason else " Set one for grounded Q&A."
    return f"*(Offline mode: {why}, so this is the standard {what}{tail}.{fix})*"


def _plural(n, word: str) -> str:
    n = int(n or 0)
    return f"{n} {word}" + ("" if n == 1 else "s")


def _shrink_phrase(q: dict) -> str:
    """How much of a shrunk quantity is really the car's own evidence."""
    w = q.get("shrink_weight")
    src = q.get("source") or "field"
    if w is None:
        return "provenance not recorded"
    if w >= 0.99 or src == "own":
        return "all of it this car's own laps"
    return f"shrunk toward the {src}, with {w:.0%} weight on this car's own laps"


def _haas_lines(facts: dict) -> list:
    """The two cars, in the offline briefing: each one's own measured rate, the
    rate its plan is built on, and how much of that is really its own evidence."""
    haas = facts.get("haas") or {}
    cars = haas.get("cars") or {}
    if not cars:
        return []
    out = []
    for code, c in cars.items():
        t = c.get("tyre") or {}
        rate = (t.get("deg_rate_by_compound_s_per_lap") or {})
        eff = t.get("effective_race_rate_s_per_lap") or {}
        rf = t.get("race_history_rate_factor") or {}
        ev = c.get("evidence_this_weekend") or {}
        plan = c.get("plan") or {}
        measured = ", ".join(f"{k.title()} {v:.4f}" for k, v in (rate.get("value") or {}).items())
        raced = ", ".join(f"{k.title()} {v:.4f}" for k, v in eff.items())
        out.append(
            f"**{c.get('name') or code}, car {c.get('number') or '?'}.** Measured this "
            f"weekend: {measured or 'no rate'} s/lap of age, from "
            f"{ev.get('n_clean_practice_laps', 0)} clean practice laps over "
            f"{_plural(ev.get('n_practice_long_run_stints', 0), 'long-run stint')} — "
            + _shrink_phrase(rate) + ". "
            f"The plan runs on {raced or 'no rate'} s/lap after the race-history factor "
            f"x{rf.get('factor', float('nan')):.3f} (shrunk to x{rf.get('shrunk', float('nan')):.3f}, "
            f"ln_sd {rf.get('ln_sd', float('nan')):.3f}), which is from previous races and "
            f"is not evidence about this weekend. Warm-up "
            f"{(t.get('warmup_s') or {}).get('value', float('nan')):.2f} s a stint, traffic "
            f"x{(t.get('traffic_sensitivity') or {}).get('value', float('nan')):.2f}. "
            f"Plan {plan.get('best') or '—'}"
            + (f", first stop lap {plan['first_stop']}" if plan.get("first_stop") else "")
            + (f" ({plan['first_stop_vs_field']:+d} laps against the field's)"
               if plan.get("first_stop_vs_field") else "")
            + ".")
    diff = (haas.get("difference_between_the_cars") or [])[:3]
    if diff:
        out.append("**Between the cars.** " + " ".join(d.rstrip(".") + "." for d in diff))
    return out


def _validation_line(v: dict) -> str:
    """The Validation paragraph.  There is nothing to report until the race has
    been run and the sealed prediction scored against it, which is the normal
    state of every weekend before its own Sunday."""
    if v.get("mae") is None:
        return ("**Validation.** " + str(v.get("note") or "nothing scored yet")
                + ". The sealed prediction is scored against the race after the flag; until "
                  "then none of the numbers above have been checked against a race, and the "
                  "receipts in the Validation tab are the previous weekends'.")
    cov = v.get("coverage") or {}
    c90 = cov.get("0.9")
    return (f"**Validation.** Stint-rate MAE {v['mae']:.3f} s/lap over "
            f"{v.get('n_rate_stints', 0)} race stints against a 0.15 target, bias "
            f"{v.get('bias', 0):+.3f} s/lap, scored against the "
            f"{v.get('regime_label', 'race regime')} curves. 90% predictive coverage "
            + (f"{c90:.1%}" if c90 is not None else "not recorded")
            + (" — over-covering, so the intervals are conservative rather than "
               "overconfident." if (c90 or 0) > 0.97 else "."))


def _offline_outlook(facts: dict, question: str | None, reason: str | None = None) -> EngineerReply:
    """The briefing when there is no practice fit yet: the outlook is the analysis."""
    o = facts["outlook"]
    tyres = o.get("tyres") or {}
    lines = [f"**{facts['event']} - outlook, {o.get('stage', 'prior')}.** Built on: "
             + "; ".join(o.get("sources", [])) + ".",
             "",
             "**Tyres.** " + "; ".join(
                 f"{c} {v['deg_s_per_lap']:.3f} s/lap [{v['deg_lo']:.3f}-{v['deg_hi']:.3f}], life "
                 f"{v['life_laps']:.0f} laps [{v['life_lo']:.0f}-{v['life_hi']:.0f}] at the plan's push"
                 for c, v in tyres.items()) + ".",
             "",
             f"**The plan.** {o.get('plan')} - stints {(o.get('plan_detail') or {}).get('stint_lens')} at push "
             f"{o.get('push', float('nan')):.2f}; P(stops) {o.get('p_stops')}. Pit "
             + (", ".join(f"stop {w['stop']} lap {w['recommended']} (window {w['lo']}-{w['hi']})"
                          for w in o.get("pit_windows") or []) or "see plan")
             + f". Pit loss {o.get('pit_loss_s', float('nan')):.1f} s ({o.get('pit_loss_source', '')})."]
    alt = (o.get("alternatives") or {}).get("plan_b")
    if alt:
        lines += ["", f"**Plan B.** {alt.get('label')} is {alt.get('delta_s', 0):+.1f} s in expectation"
                      + (f" and becomes faster if live degradation exceeds x{alt['switch_mult']:.2f}."
                         if alt.get("switch_mult") else ".")]
    sc = o.get("scenarios") or {}
    if sc.get("robust_plan"):
        lines += ["", f"**Robustness.** Across degradation x0.7-x1.4 and pit loss +/-3 s the least-regret plan is "
                      f"{sc['robust_plan']} (worst case {sc.get('robust_max_regret_s', 0):.1f} s); the base plan's worst case "
                      f"is {sc.get('base_plan_max_regret_s', 0):.1f} s."]
    prog = (o.get("value_of_information") or {}).get("practice_programme") or []
    if prog:
        lines += ["", "**What to learn in practice.** " + " ".join(prog[:2])]
    pb = o.get("safety_car_playbook") or []
    if pb:
        lines += ["", "**Safety car.** " + "; ".join(f"laps {r['from']}-{r['to']}: {r['verdict'].lower()}"
                                                     f" ({r['gain_s']:+.1f} s)" for r in pb)]
    if facts.get("committed_plans"):
        lines += ["", "**Committed.** " + "; ".join(f"{p.get('driver') or 'team'}: {p.get('label')}"
                                                    for p in facts["committed_plans"])]
    if question or reason:
        lines += ["", _offline_note(reason, question, "outlook briefing")]
    return EngineerReply("\n".join(lines), grounded=True, source="offline")


def _offline(facts: dict, question: str | None, reason: str | None = None) -> EngineerReply:
    d = facts["degradation"]
    v = facts["validation"]
    if not d and facts.get("outlook"):
        return _offline_outlook(facts, question, reason)
    ph = facts["physics_2026"]
    k = facts["fuel_sensitivity_k_track"]
    st_ = facts["strategy"]
    rg = facts.get("practice_to_race_regime", {})
    lad = facts.get("compound_ladder", {})
    plan = st_.get("best_plan", {})
    wins = ", ".join(
        f"stop {w['stop']} lap {w['recommended']} (window {w['lo']}-{w['hi']})"
        for w in st_.get("pit_windows", [])) or "see plan"
    lines = [
        f"**{facts['event']} — {facts['data']['clean_practice_laps']} clean "
        f"practice laps, practice data only.**",
        "",
        "**Tyres.** " + "; ".join(
            f"{r['compound']} {r['deg_s_per_lap']:.3f} s/lap at full push, "
            f"cliff at lap {r['knee_lap']:.0f} pushing / {r['life_laps']:.0f} "
            f"managed [{r['life_lo']:.0f}–{r['life_hi']:.0f}]"
            for r in st_.get("life", [])) + ". Tyre life is the grip budget "
        f"({st_.get('grip_budget_s', float('nan')):.1f} s of lap time a tyre "
        "surrenders before its cliff) divided by the degradation rate, so a "
        "compound cannot be both fast-degrading and long-lived.",
        "",
        *[x for line in _haas_lines(facts) for x in (line, "")],
        f"**Fuel.** {ph['derivation']}, worth {ph['fuel_effect_s_per_lap']:.3f} "
        f"s/lap. k_track posterior {k['mean_s_per_kg']:.4f} ± "
        f"{k['sd_s_per_kg']:.4f} s/kg ({k['relative_sd']:.0%} relative) — "
        "essentially the prior, so the physics assumption is carrying that split, "
        "not the data.",
        "",
        f"**Regime.** The plan assumes push {st_.get('push', float('nan')):.2f} "
        f"(1.0 = a practice long run), which implies race stints degrading at "
        f"{st_.get('implied_regime', float('nan')):.2f}x the practice rate. That "
        f"is a prediction, not an input: measured on other weekends' races the "
        f"factor is {rg.get('ratio', float('nan')):.2f} "
        f"[{rg.get('p05', float('nan')):.2f}-{rg.get('p95', float('nan')):.2f}], "
        f"{rg.get('label', 'source unknown')}. Drivers buy tyre life with lap "
        f"time; the optimiser chooses how much of that trade to make rather "
        f"than being handed the answer.",
        "",
        _validation_line(v),
        "",
        f"**The plan.** {st_['best']} — stints "
        f"{plan.get('stint_lens', [])} over {facts['race_laps']} laps. Pit "
        f"{wins}. Pit loss measured at {facts['pit_loss_s']:.1f} s from "
        f"green-flag stops; {st_.get('n_strategies', 0):,} legal plans searched.",
    ]
    for b in st_.get("by_stops", [])[1:]:
        lines += ["", f"**Alternative.** {b['strategy']} is "
                      f"{b['delta_s']:.1f} s slower in expectation but fastest on "
                      f"{b['win_prob_any']:.0%} of posterior draws."]
        break
    lines += ["", "**What this does not price.** " + facts["limits"]["not_modelled"]
              + "."]
    if facts["counterfactual_top"]:
        c = facts["counterfactual_top"][0]
        lines += ["", f"**The moment.** {c['driver']} stopped on "
                      f"{c['actual_pit_laps']} where the model wanted "
                      f"{c['model_pit_laps']} — {c['loss_s']:.1f} s, holding "
                      f"their compounds and stop count fixed so only the timing "
                      f"is scored."]
    if question or reason:
        lines += ["", _offline_note(reason, question)]
    return EngineerReply("\n".join(lines), grounded=True, source="offline")


def credential_source() -> str | None:
    """Which credential will be used.  Gemini (`GEMINI_API_KEY`, then
    `GOOGLE_API_KEY`) is the provider; Claude is used only when
    `ENGINEER_PROVIDER=claude` is set alongside `ANTHROPIC_API_KEY`."""
    if os.environ.get("ENGINEER_PROVIDER", "").lower() == "claude" and os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY"
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            return var
    return None


def provider() -> str:
    src = credential_source()
    if src == "ANTHROPIC_API_KEY":
        return "claude"
    if src:
        return "gemini"
    return "offline"


CLAUDE_MODEL = "claude-opus-5"


def _ask_claude(facts: dict, prompt: str, max_tokens: int = 4000) -> EngineerReply:
    """Grounded answer from Claude.  Streams so a long briefing cannot time out;
    the fact sheet is cached across questions in the same session."""
    import anthropic

    client = anthropic.Anthropic()
    model = os.environ.get("CLAUDE_MODEL", CLAUDE_MODEL)
    with client.beta.messages.stream(
        model=model,
        max_tokens=max_tokens,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=[{"type": "text", "text": SYSTEM},
                {"type": "text",
                 "text": "FACT SHEET (the only numbers you may use):\n" + json.dumps(facts, indent=1, default=str),
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"TASK: {prompt}"}],
    ) as stream:
        resp = stream.get_final_message()
    if resp.stop_reason == "refusal":
        return EngineerReply("The model declined to answer that.", grounded=True, source="claude", model=model)
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:
        return EngineerReply(f"The model returned no text (stop reason {resp.stop_reason}).",
                             grounded=True, source="claude", model=model)
    return EngineerReply(text, grounded=True, source="claude", model=resp.model or model)


def _have_credentials() -> bool:
    return credential_source() is not None


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------


# 429/500/503 mean the model is busy, not that the request is wrong: the large
# briefing prompt draws them where a one-line prompt goes straight through.  The
# budget is deliberately short - someone is watching a spinner, and the caller
# falls back to the offline briefing on the way out.
RETRY_STATUS = (429, 500, 503)
RETRY_BACKOFF_S = (1.0, 3.0, 6.0)


def _generate_with_retry(client, model: str, msg: str, config):
    """`generate_content`, retrying only the codes that a wait can fix."""
    for wait in RETRY_BACKOFF_S:
        try:
            return client.models.generate_content(model=model, contents=msg, config=config)
        except Exception as exc:
            if getattr(exc, "code", None) not in RETRY_STATUS:
                raise
            log.warning("%s busy (%s %s); retrying in %.0f s",
                        model, getattr(exc, "code", "?"), getattr(exc, "status", ""), wait)
            time.sleep(wait)
    return client.models.generate_content(model=model, contents=msg, config=config)


def _ask_gemini(facts: dict, prompt: str, budget: int | None = None) -> EngineerReply:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")
                          or os.environ.get("GOOGLE_API_KEY"))
    msg = (
        "FACT SHEET (the only numbers you may use):\n"
        f"{json.dumps(facts, indent=2, default=str)}\n\n"
        f"TASK: {prompt}"
    )
    model = model_name()
    resp = _generate_with_retry(
        client, model, msg,
        types.GenerateContentConfig(
            system_instruction=SYSTEM,
            max_output_tokens=budget or max_tokens(),
            # Low but non-zero: this is a factual briefing, not creative writing.
            temperature=0.2,
        ),
    )

    # A blocked prompt comes back as a normal response with no candidates, not
    # as an exception — read it before touching .text.
    fb = getattr(resp, "prompt_feedback", None)
    if fb is not None and getattr(fb, "block_reason", None):
        return EngineerReply(
            f"The model declined to answer that ({fb.block_reason}).",
            grounded=True, source="gemini", model=model)

    text = ""
    try:
        text = (resp.text or "").strip()
    except Exception:  # no text part at all
        text = ""
    reason = None
    if getattr(resp, "candidates", None):
        reason = getattr(resp.candidates[0], "finish_reason", None)
    if not text:
        # MAX_TOKENS here means the budget went entirely on the model's thinking
        # and the answer was cut before a word of it was emitted.
        return EngineerReply(
            f"The model returned no usable text (finish reason: {reason}).",
            grounded=True, source="gemini", model=model)
    if str(reason).endswith("MAX_TOKENS"):
        # Half a briefing read as a whole one is worse than no briefing.
        text += ("\n\n*(Cut short at the output limit. Raise `ENGINEER_MAX_TOKENS` "
                 "in `.env` — it is currently "
                 f"{budget or max_tokens()} — or ask a narrower question.)*")
    return EngineerReply(text, grounded=True, source="gemini", model=model)


def list_models() -> list:
    """Model ids this key can actually use — handy when a default 404s."""
    from google import genai

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")
                          or os.environ.get("GOOGLE_API_KEY"))
    out = []
    for m in client.models.list():
        name = getattr(m, "name", "") or ""
        out.append(name.removeprefix("models/"))
    return sorted(out)


def _call_failed(exc: Exception) -> str:
    """Why the live call did not happen, short enough to read in the app."""
    detail = " ".join(str(exc).split())
    return f"the {provider()} call failed ({detail[:160] or type(exc).__name__})"


def brief(event: Event | str) -> EngineerReply:
    """A race-engineer briefing on the weekend's analysis."""
    facts = fact_sheet(event)
    if not _have_credentials():
        return _offline(facts, None)
    try:
        return (_ask_claude if provider() == "claude" else _ask_gemini)(
            facts,
            "Brief the strategist on the tyre picture for both our cars. Take 31 "
            "Ocon and 87 Bearman in turn: the degradation rate on each compound and "
            "how much of it is that driver's own evidence rather than the team's, how "
            "the car compares with its team-mate and with the field reference, its "
            "warm-up and traffic sensitivity, the plan and pit window that rate "
            "implies, and what would make you change the call. Then one paragraph on "
            "what the field and the outlook are doing that threatens those two calls "
            "(stop-count probabilities, plan B and its switch trigger, the safety-car "
            "playbook), and one on what we do not know: the practice support behind "
            "each compound's rate, and whether the sealed prediction has been scored "
            "against a race yet. If live_session is populated, finish with what you "
            "would say on the radio to each car right now. Six short paragraphs at "
            "most.",
        )
    except Exception as exc:
        log.warning("%s briefing failed (%s); using offline briefing", provider(), exc)
        return _offline(facts, None, _call_failed(exc))


def ask(event: Event | str, question: str) -> EngineerReply:
    """Grounded Q&A over the sealed analysis."""
    facts = fact_sheet(event)
    if not _have_credentials():
        return _offline(facts, question)
    try:
        return (_ask_claude if provider() == "claude" else _ask_gemini)(
            facts, f"Answer the strategist's question: {question}")
    except Exception as exc:
        log.warning("%s Q&A failed (%s); using offline briefing", provider(), exc)
        return _offline(facts, question, _call_failed(exc))
