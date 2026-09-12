"""AI race engineer — LLM-grounded briefing and Q&A (P2).

This is the plan's P2 component and first on the cut line: nothing else depends
on it.  (The plan specifies a Claude-grounded engineer; this build uses Gemini
Flash at the operator's request.  Nothing else in the pipeline is affected —
no model output ever feeds back into a fit.)

**Grounding.** The model is never asked to estimate anything. Every number it
is allowed to say is computed by the pipeline and handed to it as a fact sheet;
the system prompt forbids inventing figures and requires it to say so when the
fact sheet does not cover a question. An LLM guessing tyre degradation numbers
would undo the entire point of the rest of this codebase.

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
from dataclasses import dataclass

from src.config import DATA_PROCESSED, Event, get_event

log = logging.getLogger("degless.engineer")

# Flash, per the operator's request.  Overridable without a code change; note
# that gemini-2.0-flash has been shut down, so an older default would fail.
DEFAULT_MODEL = "gemini-3.7-flash"


def model_name() -> str:
    if provider() == "claude":
        return os.environ.get("CLAUDE_MODEL", "claude-opus-5")
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

SYSTEM = """You are a Formula 1 race engineer briefing a race strategist. You \
are speaking about a real analysis produced by a tool called degless, which \
fits tyre degradation curves to practice long-run data under a 2026-regulation \
fuel-mass prior, seals its predictions, and then scores them against the race.

Rules you must follow:
1. Every number you state must come from the FACT SHEET below. Never estimate, \
interpolate, or recall a number from elsewhere. If the fact sheet does not \
contain something, say plainly that the analysis does not cover it.
2. Report uncertainty wherever the fact sheet gives it. A credible interval is \
not decoration.
3. Do not oversell. If the validation shows the model is biased, say so in the \
same breath as any recommendation that depends on it.
4. Be concise and concrete, the way a race engineer on the radio is concise. \
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
        "committed_plans": _committed_plans(ev),
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


def _offline_outlook(facts: dict, question: str | None) -> EngineerReply:
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
    if question:
        lines += ["", f"*(Offline mode: no GEMINI_API_KEY in .env, so this is the standard outlook briefing "
                      f"rather than an answer to \u201c{question}\u201d.)*"]
    return EngineerReply("\n".join(lines), grounded=True, source="offline")


def _offline(facts: dict, question: str | None) -> EngineerReply:
    d = facts["degradation"]
    v = facts["validation"]
    if not d and facts.get("outlook"):
        return _offline_outlook(facts, question)
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
        f"**Validation.** Stint-rate MAE {v['mae']:.3f} s/lap over "
        f"{v.get('n_rate_stints', 0)} race stints against a 0.15 target, bias "
        f"{v.get('bias', 0):+.3f} s/lap, scored against the "
        f"{v.get('regime_label', 'race regime')} curves. 90% predictive coverage "
        f"{v['coverage'].get('0.9', float('nan')):.1%}"
        + (" — over-covering, so the intervals are conservative rather than "
           "overconfident."
           if v['coverage'].get('0.9', 0) > 0.97 else "."),
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
    if question:
        lines += ["", f"*(Offline mode: no GEMINI_API_KEY in .env, "
                      f"so this is the standard briefing rather than an answer to "
                      f"“{question}”. Set one for grounded Q&A.)*"]
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


def _ask_gemini(facts: dict, prompt: str, max_tokens: int = 2000) -> EngineerReply:
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
    resp = client.models.generate_content(
        model=model,
        contents=msg,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            max_output_tokens=max_tokens,
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
    if not text:
        reason = None
        if getattr(resp, "candidates", None):
            reason = getattr(resp.candidates[0], "finish_reason", None)
        # MAX_TOKENS here means the answer was cut before any text was emitted.
        return EngineerReply(
            f"The model returned no usable text (finish reason: {reason}).",
            grounded=True, source="gemini", model=model)
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


def brief(event: Event | str) -> EngineerReply:
    """A race-engineer briefing on the weekend's analysis."""
    facts = fact_sheet(event)
    if not _have_credentials():
        return _offline(facts, None)
    try:
        return (_ask_claude if provider() == "claude" else _ask_gemini)(
            facts,
            "Give the strategist a briefing on this weekend: what the tyres are "
            "doing, what the physics prior is and is not buying us, how the "
            "sealed prediction scored against the race (or that it has not run yet), "
            "what the outlook recommends (plan, stop-count probabilities, pit windows, "
            "plan B and its switch trigger, the robust plan across scenarios, what to "
            "learn in practice, the safety-car playbook) and — if live_session is "
            "populated — what the live engine is saying right now: who is near the "
            "cliff, whose window is open, undercut threats, and what you would call "
            "over the radio. Six short paragraphs at most.",
        )
    except Exception as exc:
        log.warning("Gemini briefing failed (%s); using offline briefing", exc)
        return _offline(facts, None)


def ask(event: Event | str, question: str) -> EngineerReply:
    """Grounded Q&A over the sealed analysis."""
    facts = fact_sheet(event)
    if not _have_credentials():
        return _offline(facts, question)
    try:
        return (_ask_claude if provider() == "claude" else _ask_gemini)(
            facts, f"Answer the strategist's question: {question}")
    except Exception as exc:
        log.warning("Gemini Q&A failed (%s); using offline briefing", exc)
        return _offline(facts, question)
