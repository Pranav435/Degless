"""The outlook: the best current strategy picture for the next race, kept
fresh as information arrives.

Between races the model knows nothing about the coming weekend from practice,
because there has been none - but it is not ignorant.  It has the compound
ladder, three years of races at the circuit (or, for a circuit nobody has
raced, the 2026 season so far), a practice->race regime factor measured on
every scored weekend, a pit-loss and allocation prior from the same, the
thermal sensitivity of degradation, the circuit's revealed plan shapes and
the leave-one-out calibration of the plan-deciding constants.  That is enough
to run the same strategy search the sealed model runs, with honest (wide)
uncertainty, and to say which of that uncertainty the decision hinges on.

From then on every new piece of information tightens the picture, in order:

    prior            ladder + circuit history / 2026 season pooled
    live board       long runs as they happen, folded in during a session
    sealed fit       the refit after each practice session replaces the prior
    live board again the next session, on top of the sealed fit
    track temperature  from the live weather feed, through the thermal prior

Every build appends a line to a timeline, so the app can show how the
recommended stop count, the stint lengths and the tyre lives have moved as
data came in.

What a build produces, beyond the plan itself:

* **the tyre-optimal plan** beside the position-aware one, so the reader
  sees what the undercut-exposure term and the plan-shape prior changed;
* **scenarios** - the search re-run with degradation x0.7 / x1.4 and the pit
  lane 3 s quicker / slower, and the minimax-regret plan across them;
* **value of information** - which compound's uncertainty the decision hinges
  on, which is the practice programme;
* **safety-car playbook** - lap by lap, box under a safety car or not;
* **plan B** - the best plan with one more stop and the live degradation
  multiplier at which it overtakes plan A: the switch trigger.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import firststop, strategy as strat
from src.calibration import Calibration, get_calibration
from src.compounds import allocation_prior, pace_step_prior
from src.config import (
    DATA_PROCESSED, DIRTY_AIR_S_PER_LAP, MAX_STINTS_PER_COMPOUND, VALID_COMPOUNDS, Event, get_event,
)
from src.history import (
    THERMAL_BETA_DEFAULT, apply_rate_prior_to_model, circuit_prior, plan_prior_for, season_prior,
    stint_caps_for, summarise_race, thermal_sensitivity, YEARS,
)
from src.live.store import LIVE_DIR, atomic_write, read_snapshot
from src.regime import RegimeFactor, regime_prior
from src.tyre import EXTRAP_LN_SD_MEASURED, TyreModel

log = logging.getLogger("degless.outlook")

N_OUTLOOK_DRAWS = 300
LIVE_SLOPE_LN_SD_FLOOR = 0.30   # a live long-run slope is not evolution-corrected: at least this wide
MIN_LIVE_SLOPE = 0.015          # s/lap below which a live slope says nothing about the rate
MIN_LIVE_STINTS = 2
DEG_SCENARIOS = (0.7, 1.0, 1.4)
PIT_SCENARIOS = (-3.0, 0.0, 3.0)
SCENARIO_DRAWS = 150


def outlook_path(key: str) -> Path:
    return DATA_PROCESSED / f"outlook_{key}.json"


def draws_path(key: str) -> Path:
    return DATA_PROCESSED / f"outlook_{key}.npz"


def timeline_path(key: str) -> Path:
    return DATA_PROCESSED / f"outlook_{key}_timeline.jsonl"


# --------------------------------------------------------------------------
# The base model: sealed fit if there is one, the composed prior otherwise
# --------------------------------------------------------------------------


@dataclass
class BaseModel:
    model: TyreModel
    regime: RegimeFactor
    stage: str                       # "prior" | "sealed"
    sources: list = field(default_factory=list)
    pit_loss_s: float = 22.0
    pit_loss_source: str = ""
    allocation: dict = field(default_factory=dict)
    stint_cap: dict = field(default_factory=dict)
    support: dict | None = None
    sessions_used: list = field(default_factory=list)
    sealed_file: str = ""
    history: dict = field(default_factory=dict)
    season: dict = field(default_factory=dict)
    thermal: dict = field(default_factory=dict)
    combination: list = field(default_factory=list)
    n_practice_laps: int = 0
    prior_basis: str = ""
    calibration: Calibration = field(default_factory=Calibration)
    plan_prior: dict = field(default_factory=dict)
    net_step: dict = field(default_factory=dict)      # {measured, se, derivation}
    pace_calibration: dict = field(default_factory=dict)
    first_stop_table: dict | None = None              # firststop.first_stop_penalty_table
    first_stop_prior: dict = field(default_factory=dict)   # its summary, for the JSON
    dirty_air: float = DIRTY_AIR_S_PER_LAP            # this circuit's value, not the pooled one


def _regime_from_meta(rg: dict) -> RegimeFactor:
    return RegimeFactor(ratio=float(rg.get("ratio", 0.65)), ln_sd=float(rg.get("ln_sd", 0.35)),
                        sources=list(rg.get("sources", [])), measured=bool(rg.get("measured", False)),
                        label=str(rg.get("label", "")), derivation=str(rg.get("derivation", "")),
                        donor_detail=list(rg.get("donors", [])), temperature=dict(rg.get("temperature", {})))


def _history_temps(ev: Event) -> list:
    out = []
    for y in YEARS:
        r = summarise_race(y, ev.circuit)
        if r and not r.get("rain") and r.get("track_temp_c") is not None:
            out.append(float(r["track_temp_c"]))
    return out


def sim_kwargs(base: BaseModel) -> dict:
    """The keyword arguments every search on this base model shares."""
    cal = base.calibration
    return dict(regime=base.regime, support=base.support, max_per_compound=base.allocation,
                max_stint=(base.stint_cap or None), undercut_lambda=cal.undercut_lambda,
                plan_prior=base.plan_prior, plan_prior_tau_s=cal.plan_prior_tau_s,
                first_stop_prior=base.first_stop_table, first_stop_kappa_s=cal.first_stop_kappa_s,
                traffic_s_per_lap=base.dirty_air, grid_penalty_s=cal.grid_start_penalty_s)


def eval_kwargs(base: BaseModel) -> dict:
    cal = base.calibration
    return dict(allocation=base.allocation, stint_cap=base.stint_cap, undercut_lambda=cal.undercut_lambda,
                plan_prior=base.plan_prior, plan_prior_tau_s=cal.plan_prior_tau_s,
                first_stop_prior=base.first_stop_table, first_stop_kappa_s=cal.first_stop_kappa_s,
                traffic_s_per_lap=base.dirty_air, grid_penalty_s=cal.grid_start_penalty_s)


def load_base(ev: Event, *, track_temp_c: float | None = None, n_draws: int = N_OUTLOOK_DRAWS,
              seed: int = 0, force_prior: bool = False) -> BaseModel:
    """`force_prior=True` composes the pre-practice prior even when a sealed
    fit exists - the benchmark's "what would the outlook have said on Thursday"."""
    from src.live.engine import WeekendModel, backfill_circuit_history, pit_loss_prior

    rng = np.random.default_rng(seed)
    cal = get_calibration(ev)
    post = DATA_PROCESSED / f"posterior_{ev.key}.npz"
    meta_p = DATA_PROCESSED / f"weekend_{ev.key}.json"
    if not meta_p.exists():
        meta_p = DATA_PROCESSED / f"meta_{ev.key}.json"
    if post.exists() and meta_p.exists() and not force_prior:
        from src.model_bayes import BayesFit

        meta = json.loads(meta_p.read_text())
        fit = BayesFit.load(post)
        total = fit.posterior["lin"].shape[0]
        idx = rng.choice(total, size=min(n_draws, total), replace=False)
        # V4/WP-B: the practice age support this weekend reached, so the outlook
        # prices a stint past it as an extrapolation rather than a measurement.
        # A metadata file that predates the key leaves `support` empty, and the
        # widening is then simply absent rather than wrong.
        sup = {k: float(v) for k, v in (meta.get("age_support_by_compound") or {}).items()}
        model = TyreModel.from_fit(fit, draws=idx, budget=cal.budgets,
                                   manage_floor=cal.manage_wear_floor, manage_cost_s=cal.manage_cost_s,
                                   support=sup, extrap_ln_sd=EXTRAP_LN_SD_MEASURED)
        # `first_stop_green` / `dirty_air` from the sibling file if this one
        # predates them; both are pure 2023-25 history (see the helper).
        hist = backfill_circuit_history(ev, meta.get("circuit_history") or {}, meta_p)
        caps = {k: int(v) for k, v in (hist.get("stint_cap") or {}).items()}
        used = list(meta.get("sessions_used", []))
        sources = [f"sealed practice fit on {', '.join(used) or 'practice'} "
                   f"({meta.get('n_clean_laps', fit.n_laps)} clean long-run laps, {post.name})"]
        if hist:
            sources.append(f"circuit history {hist.get('years')} folded into the fit")
        th = dict(hist.get("thermal") or {})
        if track_temp_c is not None:
            th["track_temp_live"] = float(track_temp_c)
        pp = meta.get("plan_prior") or plan_prior_for(None)
        ns = meta.get("net_step") or {}
        sources.append(f"calibration: {cal.source}")
        if pp:
            sources.append(f"plan-shape prior from {pp.get('source')}")
        # The circuit's first-stop density and its own dirty-air cost, both from
        # the races already run there - never from this weekend's.
        fs_table = firststop.first_stop_penalty_table(hist.get("first_stop_green"), ev.n_race_laps,
                                                      model.compounds)
        _pp_stops = (pp or {}).get("stops") or {}
        _pp_starts = (pp or {}).get("starts") or {}
        fs_sum = meta.get("first_stop_prior") or firststop.first_stop_summary(
            hist.get("first_stop_green"), ev.n_race_laps,
            start_compound=(max(_pp_starts, key=_pp_starts.get) if _pp_starts else None),
            n_stops=(int(max(_pp_stops, key=_pp_stops.get)) if _pp_stops else None)) or {}
        dirty = cal.dirty_air_for(ev.circuit)
        if fs_sum:
            sources.append(f"first-stop prior: mode lap {fs_sum['mode']}, "
                           f"{fs_sum['p25']:.0f}-{fs_sum['p75']:.0f} at kappa {cal.first_stop_kappa_s:.2f} s/nat")
        sources.append(f"dirty air {dirty:.2f} s/lap "
                       f"({'this circuit' if ev.circuit in cal.dirty_air_by_circuit else 'pooled over the 2026 races'})")
        return BaseModel(model=model, regime=_regime_from_meta(meta.get("regime", {})), stage="sealed",
                         sources=sources, pit_loss_s=float(meta.get("pit_loss_s", 22.0)),
                         pit_loss_source=str(meta.get("pit_loss_source", "")),
                         allocation=dict((meta.get("allocation") or {}).get("caps")
                                         or {c: MAX_STINTS_PER_COMPOUND for c in VALID_COMPOUNDS}),
                         stint_cap=caps, support=sup,
                         sessions_used=used, sealed_file=str(meta.get("sealed_file", "")), history=hist,
                         thermal=th, combination=list(meta.get("history_combination") or []),
                         n_practice_laps=int(meta.get("n_clean_laps", 0)), prior_basis="sealed fit",
                         calibration=cal, plan_prior=pp, net_step=ns,
                         first_stop_table=fs_table, first_stop_prior=fs_sum, dirty_air=dirty)

    # -- no practice yet: compose the prior --------------------------------
    model = WeekendModel.prior_model(ev, n_draws, rng, calibration=cal)
    # The live weather feed reports the track temperature *now*, during a
    # practice session - it is not a race-day forecast, and handing it to the
    # regime model as one would claim to know Sunday's track from Friday's.  So
    # it enters as `practice_temp_c` only; the temperature mode stays "none"
    # unless a real forecast is supplied.
    regime = regime_prior(ev, practice_temp_c=track_temp_c)
    sources = [f"compound-ladder prior (MEDIUM 0.10 s/lap, factor-2 spread), regime factor "
               f"{regime.ratio:.2f}x [{regime.p05:.2f}-{regime.p95:.2f}] {regime.label}"]
    temps = _history_temps(ev)
    t_hist = float(np.mean(temps)) if temps else None
    cp = circuit_prior(ev, track_temp_c=(track_temp_c if track_temp_c is not None else t_hist),
                       probe_practice_temp=False)
    combination, season, basis = [], {}, "compound ladder only"
    if cp.available and cp.rate_prior:
        model, moved = apply_rate_prior_to_model(model, ev, regime, cp.rate_prior, rng=rng, label="circuit history", pooled=True)
        combination = moved.to_dict("records")
        basis = f"circuit history {cp.years}"
        sources.append(f"{ev.circuit} races {cp.years}: race degradation per compound x season factor "
                       f"{cp.season.get('factor', 1):.2f} ({cp.season.get('n_circuits', 0)} shared circuits)"
                       + (f", thermal x{cp.thermal.get('multiplier', 1):.2f} at {track_temp_c:.0f} degC"
                          if track_temp_c is not None and cp.thermal.get("multiplier") else ""))
    else:
        season = season_prior()
        if season.get("rate_prior"):
            model, moved = apply_rate_prior_to_model(model, ev, regime, season["rate_prior"], rng=rng, label="2026 season", pooled=True)
            combination = moved.to_dict("records")
            basis = f"2026 season pooled ({season.get('n_circuits', 0)} circuits)"
            sources.append(f"no race history for {ev.circuit}: 2026 race degradation pooled over "
                           f"{', '.join(season.get('circuits', []))}")
        else:
            sources.append(f"no race history for {ev.circuit} and no 2026 season measurement yet")
    alloc = allocation_prior(ev)
    pit, pit_src = pit_loss_prior(ev)
    if cp.available and cp.pit_loss_s:
        pit, pit_src = float(cp.pit_loss_s), f"this pit lane, {cp.years}"
    caps = stint_caps_for(ev, cp) if cp.available else {}
    if caps:
        sources.append("stint caps from this circuit's races: " + ", ".join(f"{c} {v}" for c, v in caps.items()))
    sources.append(f"pit loss {pit:.1f} s ({pit_src}); allocation {alloc['caps']}")
    pp = plan_prior_for(cp if cp.available else None)
    if pp:
        sources.append(f"plan-shape prior from {pp.get('source')}: " + ", ".join(f"{k} {v}" for k, v in list(pp['sequences'].items())[:4]))
    fs_table = firststop.first_stop_penalty_table(
        cp.first_stop_green if cp.available else None, ev.n_race_laps, model.compounds)
    _pp_stops = (pp or {}).get("stops") or {}
    _pp_starts = (pp or {}).get("starts") or {}
    fs_sum = firststop.first_stop_summary(
        cp.first_stop_green if cp.available else None, ev.n_race_laps,
        start_compound=(max(_pp_starts, key=_pp_starts.get) if _pp_starts else None),
        n_stops=(int(max(_pp_stops, key=_pp_stops.get)) if _pp_stops else None)) or {}
    if fs_sum:
        sources.append(f"first-stop prior: mode lap {fs_sum['mode']}, {fs_sum['p25']:.0f}-{fs_sum['p75']:.0f} "
                       f"from {fs_sum['n']} historical green first stops at kappa {cal.first_stop_kappa_s:.2f} s/nat")
    dirty = cal.dirty_air_for(ev.circuit)
    sources.append(f"dirty air {dirty:.2f} s/lap "
                   f"({'this circuit' if ev.circuit in cal.dirty_air_by_circuit else 'pooled over the 2026 races'})")
    ps = pace_step_prior(ev, circuit=cp if cp.available else None)
    ns = {"measured": ps.get("net_stint_step_measured"), "se": ps.get("net_stint_step_se"),
          "derivation": ps.get("derivation", "")}
    sources.append(f"calibration: {cal.source}")
    return BaseModel(model=model, regime=regime, stage="prior", sources=sources, pit_loss_s=float(pit),
                     pit_loss_source=pit_src, allocation=dict(alloc["caps"]), stint_cap=caps, support=None,
                     history=(cp.as_dict() if cp.available else {}), season=season,
                     thermal=dict(cp.thermal) if cp.available else {}, combination=combination,
                     n_practice_laps=0, prior_basis=basis, calibration=cal, plan_prior=pp, net_step=ns,
                     first_stop_table=fs_table, first_stop_prior=fs_sum, dirty_air=dirty)


# --------------------------------------------------------------------------
# Folding the live long-run board in
# --------------------------------------------------------------------------


def fold_live_board(base: BaseModel, pooled: dict, *, session_name: str = "", seed: int = 1) -> tuple:
    """Update the model with the live practice engine's pooled slopes.

    A live slope is fuel-corrected but not evolution-corrected (a rubbering-in
    track makes every live slope read low), so it enters with a wide floor on
    its log-scale width and the sealed refit after the session replaces it.
    Both the model and a long run are in the practice regime: no transfer.
    """
    rate_prior, rows = {}, []
    for c, v in (pooled or {}).items():
        if c not in base.model.compounds:
            continue
        slope, se, n_st = float(v.get("slope_s_per_lap", 0)), float(v.get("se", 0)), int(v.get("n_stints", 0))
        if slope < MIN_LIVE_SLOPE or n_st < MIN_LIVE_STINTS:
            rows.append({"compound": c, "live_slope": slope, "se": se, "n_stints": n_st,
                         "used": False, "why": "slope too small or too few stints to inform the rate"})
            continue
        ln_sd = float(np.sqrt((se / slope) ** 2 + LIVE_SLOPE_LN_SD_FLOOR ** 2))
        rate_prior[c] = {"mean_s_per_lap": slope, "ln_sd": ln_sd}
        rows.append({"compound": c, "live_slope": slope, "se": se, "n_stints": n_st, "n_laps": int(v.get("n_laps", 0)),
                     "ln_sd_used": ln_sd, "used": True})
    if not rate_prior:
        return base.model, rows
    model, moved = apply_rate_prior_to_model(base.model, base.regime, base.regime, rate_prior,
                                             rng=np.random.default_rng(seed), same_regime=True,
                                             label=f"live {session_name or 'practice'} board")
    mv = {r["compound"]: r for r in moved.to_dict("records")}
    for r in rows:
        m = mv.get(r["compound"])
        if m and r.get("used"):
            r.update({"model_before": m.get("practice"), "model_after": m.get("combined"),
                      "weight_on_live": m.get("weight_on_history")})
    return model, rows


# --------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------


def _life_summary(model: TyreModel, ev: Event, push: float, caps: dict | None) -> dict:
    out = {}
    t = model.life_table(ev, push, caps=caps)
    for r in t.to_dict("records"):
        out[r["compound"]] = {"deg_s_per_lap": r["deg_s_per_lap"], "deg_lo": r["deg_lo"], "deg_hi": r["deg_hi"],
                              "life_laps": r["life_laps"], "life_lo": r["life_lo"], "life_hi": r["life_hi"],
                              "life_full_push": r["life_full_push"], "life_model_uncapped": r["life_model_uncapped"],
                              "longer_than_race": r["longer_than_race"], "bound_by": r["bound_by"],
                              # V4/WP-B: how far past the practice evidence the
                              # quoted life is, so the desk can say "28, and
                              # that is 2x anything we measured" rather than "28"
                              "practice_support_laps": r["practice_support_laps"],
                              "life_extrap_ln_sd": r["life_extrap_ln_sd"],
                              "life_extrap_note": r["life_extrap_note"],
                              "pace_offset_s": r["pace_offset_s"], "grip_budget_s": r["grip_budget_s"]}
    return out


def _plan_dict(row) -> dict:
    return {"label": str(row["strategy"]), "compounds": str(row["compounds"]).split("-"),
            "pit_laps": [int(x) for x in row["pit_laps"]], "stint_lens": [int(x) for x in row["stint_lens"]],
            "push": float(row["push"]), "n_stops": int(row["n_stops"]),
            "delta_s": float(row.get("delta_s", 0.0)), "win_prob": float(row.get("win_prob", 0.0)),
            "position_s": float(row.get("position_s", 0.0)), "prior_s": float(row.get("prior_s", 0.0)),
            "first_stop_s": float(row.get("first_stop_s", 0.0))}


def _scenarios(model: TyreModel, ev: Event, base: BaseModel, best_plan: dict, sim_kw: dict) -> dict:
    """The search under degradation and pit-lane scenarios, and the plan that
    regrets least across all of them."""
    rng = np.random.default_rng(3)
    idx = np.sort(rng.choice(model.n_draws, size=min(SCENARIO_DRAWS, model.n_draws), replace=False))
    small = model.subsample(idx)
    cells, cands = [], {best_plan["label"]: best_plan}
    for dm in DEG_SCENARIOS:
        for dp in PIT_SCENARIOS:
            sm = strat.scale_model(small, dm)
            res = strat.simulate_model(sm, ev, base.pit_loss_s + dp, step=2, shortlist=1500, **sim_kw)
            if res.table.empty:
                continue
            top = _plan_dict(res.table.iloc[0])
            cands.setdefault(top["label"], top)
            cells.append({"deg_mult": dm, "pit_delta_s": dp, "best": top["label"], "n_stops": top["n_stops"],
                          "p_stops": {str(k): v for k, v in res.p_stops.items()},
                          "stint_lens": top["stint_lens"], "push": top["push"]})
    # regret of every candidate in every scenario, on the full draw set
    plans = list(cands.values())
    regret = {p["label"]: [] for p in plans}
    ek = eval_kwargs(base)
    for cell in cells:
        sm = strat.scale_model(model, cell["deg_mult"])
        tbl, det = strat.evaluate_plans(sm, ev, plans, base.pit_loss_s + cell["pit_delta_s"], **ek)
        means = {d["label"]: float(d["times"].mean()) for d in det if d.get("valid")}
        floor = min(means.values())
        for lab, m in means.items():
            regret[lab].append(m - floor)
        cell["base_plan_regret_s"] = float(means.get(best_plan["label"], np.nan) - floor)
        cell["deg_equiv_temp_delta_c"] = float(np.log(cell["deg_mult"]) / THERMAL_BETA_DEFAULT)
    worst = {lab: (max(r) if r else np.nan) for lab, r in regret.items()}
    robust = min(worst, key=worst.get) if worst else best_plan["label"]
    return {"matrix": cells,
            "candidates": [{**p, "max_regret_s": float(worst.get(p["label"], np.nan)),
                            "mean_regret_s": float(np.mean(regret[p["label"]])) if regret[p["label"]] else None}
                           for p in plans],
            "robust": robust, "robust_max_regret_s": float(worst.get(robust, np.nan)),
            "base_max_regret_s": float(worst.get(best_plan["label"], np.nan)),
            "deg_scenarios": list(DEG_SCENARIOS), "pit_scenarios": list(PIT_SCENARIOS)}


def practice_programme(voi: dict, base: BaseModel, life: dict) -> list:
    """Turn the value-of-information ranking into runs worth doing."""
    out = []
    if not voi or not voi.get("by_compound"):
        return out
    ranked = sorted(voi["by_compound"].items(), key=lambda kv: -kv[1]["gain_s"])
    for c, v in ranked:
        sup = (base.support or {}).get(c)
        target = int(np.clip(round(0.4 * life.get(c, {}).get("life_laps", 20)), 8, 18))
        bins = v.get("best_by_rate_bin") or []
        lo, hi = bins[0], bins[-1]
        moves = v.get("decision_moves")
        text = (f"{c}: a long run of {target}+ laps"
                + (f" (practice so far reaches age {sup:.0f})" if sup else " (no practice support yet)")
                + f" - worth {v['gain_s']:.1f} s of expected regret ({v['share']:.0%} of the total)")
        if moves and lo and hi and lo != hi:
            text += f"; at the slow end of its rate the plan is {lo}, at the fast end {hi}"
        elif not moves:
            text += "; the decision does not turn on this compound"
        out.append({"compound": c, "gain_s": float(v["gain_s"]), "share": float(v["share"]),
                    "target_laps": target, "text": text, "decision_moves": bool(moves)})
    return out


def build(event: Event | str, *, session: str | None = None, n_draws: int = N_OUTLOOK_DRAWS,
          quick: bool = False, seed: int = 0, write: bool = True, force_prior: bool = False) -> dict:
    """Build the outlook for a weekend and (by default) write it to disk.
    `force_prior=True` ignores a sealed fit and builds the pre-practice picture."""
    ev = get_event(event) if isinstance(event, str) else event
    t0 = time.time()
    # -- what the live feed knows, if a practice session is on or just ended
    snap = read_snapshot(session) if session else {}
    live_meta = (snap.get("meta") or {}) if snap else {}
    live_session = (live_meta.get("session") or {}) if live_meta else {}
    track_temp = None
    try:
        tt = (live_meta.get("weather") or {}).get("TrackTemp")
        track_temp = float(tt) if tt is not None else None
    except (TypeError, ValueError):
        track_temp = None
    base = load_base(ev, track_temp_c=track_temp, n_draws=n_draws, seed=seed, force_prior=force_prior)
    model = base.model
    live_rows, live_used, live_note = [], False, ""
    sess_name = str(live_session.get("Name") or "")
    if snap and snap.get("engine") == "practice":
        if base.stage == "sealed" and sess_name and sess_name in base.sessions_used:
            live_note = f"{sess_name} is already in the sealed fit; live board not folded in again"
        else:
            model, live_rows = fold_live_board(base, snap.get("pooled") or {}, session_name=sess_name)
            live_used = any(r.get("used") for r in live_rows)
            if live_used:
                base.sources.append(f"live {sess_name or 'practice'} long-run board folded in: "
                                    + ", ".join(f"{r['compound']} {r['live_slope']:.3f}+/-{r['se']:.3f} s/lap "
                                                f"({r['n_stints']} stints)" for r in live_rows if r.get("used"))
                                    + " (not evolution-corrected; the refit after the session replaces this)")
    stage = base.stage + ("+live" if live_used else "")
    stage_label = ({"prior": f"prior only - {base.prior_basis}",
                    "sealed": f"sealed fit on {', '.join(base.sessions_used) or 'practice'}"}[base.stage]
                   + (f" + live {sess_name or 'practice'} board" if live_used else ""))

    # -- the search, with the ladder gate enforced --------------------------
    sim_kw = sim_kwargs(base)
    net = base.net_step or {}
    model, res, pace_cal = strat.search_with_pace_calibration(
        model, ev, base.pit_loss_s, net_step_s=float(net.get("measured") if net.get("measured") is not None else np.nan),
        net_step_se_s=float(net.get("se") or 0.0), **sim_kw)
    base.pace_calibration = pace_cal
    cal = base.calibration
    out = {"event": ev.key, "event_name": ev.name, "circuit": ev.circuit, "n_race_laps": ev.n_race_laps,
           "updated_utc": datetime.now(timezone.utc).isoformat(), "stage": stage, "stage_label": stage_label,
           "prior_basis": base.prior_basis, "sources": base.sources, "sessions_used": base.sessions_used,
           "sealed_file": base.sealed_file, "n_practice_laps": base.n_practice_laps,
           "live": {"session": session, "session_name": sess_name, "used": live_used, "rows": live_rows,
                    "note": live_note, "n_long_runs": len(snap.get("board", [])) if snap else 0,
                    "track_temp_c": track_temp},
           "regime": base.regime.as_dict(), "pit_loss_s": base.pit_loss_s, "pit_loss_source": base.pit_loss_source,
           "allocation": base.allocation, "stint_cap": base.stint_cap, "support": base.support,
           "thermal": {**base.thermal, "beta_per_c": float(thermal_sensitivity().get("beta_per_c", THERMAL_BETA_DEFAULT))},
           "history": {k: v for k, v in base.history.items() if k in ("circuit", "years", "stops", "plans", "starts",
                                                                       "stint_typical", "stint_cap", "stint_longest",
                                                                       "pit_loss_s", "sc_share", "soft_race_tyre",
                                                                       "season", "thermal", "ladder", "first_stop")},
           "season": {k: v for k, v in base.season.items() if k != "rate_prior"},
           "combination": base.combination, "n_draws": model.n_draws,
           "calibration": {**{k: v for k, v in cal.as_dict().items() if k != "driver_factors"}, "source": cal.source,
                           "dirty_air_used": float(base.dirty_air)},
           "plan_prior": {"source": base.plan_prior.get("source"), "n": base.plan_prior.get("n"),
                          "sequences": dict(list((base.plan_prior.get("sequences") or {}).items())[:8]),
                          "starts": base.plan_prior.get("starts"),
                          "nomination": base.plan_prior.get("nomination")},
           "first_stop_prior": base.first_stop_prior,
           "net_step": net, "pace_calibration": {k: v for k, v in pace_cal.items() if k != "model_net_draws"}}
    if res.table.empty:
        out["strategy"] = {}
        out["runtime_s"] = round(time.time() - t0, 1)
        if write:
            _write(ev, out, model, res, pd.DataFrame(), pd.DataFrame())
        return out
    best = _plan_dict(res.table.iloc[0])
    push = best["push"]
    pw = strat.pit_window_model(model, ev, res.best, base.pit_loss_s, max_stint=res.max_stint, push=push,
                                undercut_lambda=cal.undercut_lambda, traffic_s_per_lap=base.dirty_air,
                                first_stop_prior=base.first_stop_table,
                                first_stop_kappa_s=cal.first_stop_kappa_s)
    windows = strat.windows_from_sweep(pw, res.best)
    life = _life_summary(model, ev, push, res.max_stint if isinstance(res.max_stint, dict) else None)
    comps_present = [c for c in ("SOFT", "MEDIUM", "HARD") if c in model.compounds]
    uc_old = "MEDIUM" if "MEDIUM" in comps_present else comps_present[-1]
    uc_new = "SOFT" if "SOFT" in comps_present else comps_present[0]
    uc_age = int(min(res.max_stint.get(uc_old, 40), 40))
    uc = strat.undercut_window_model(model, uc_old, uc_new, max_age=max(uc_age, 8), push=push)
    by_stops = [_plan_dict(r) | {"win_prob_any": float(r["win_prob_any"]), "delta_s": float(r["delta_s"])}
                for _, r in res.by_stops.iterrows()]
    by_start = [_plan_dict(r) | {"start": str(r["start"]), "win_prob_any": float(r["win_prob_any"]),
                                 "delta_s": float(r["delta_s"])}
                for _, r in strat.best_by_start(res).iterrows()]
    # plan B (one more stop) and plan C (one fewer), with the deg multiplier at which each overtakes A
    alts = {}
    ek = eval_kwargs(base)
    for name, k in (("plan_b", best["n_stops"] + 1), ("plan_c", best["n_stops"] - 1)):
        row = res.by_stops[res.by_stops["n_stops"] == k]
        if row.empty:
            continue
        alt = _plan_dict(row.iloc[0])
        cx = strat.deg_crossover(model, ev, {"compounds": res.best["compounds"], "pit_laps": res.best["pit_laps"]},
                                 {"compounds": alt["compounds"], "pit_laps": alt["pit_laps"]},
                                 base.pit_loss_s, **ek)
        alt["switch_mult"] = cx.get("mult")
        alt["switch_direction"] = cx.get("direction")
        alt["crossover_curve"] = cx.get("curve", [])
        alts[name] = alt
    voi = strat.value_of_information(res)
    programme = practice_programme(voi, base, life)
    # `sc_playbook` prices only the decision *this lap* under a safety car, so the
    # first-stop prior has nothing to say about it and is deliberately absent.
    pb = strat.sc_playbook(model, ev, res.best, base.pit_loss_s, push=push, allocation=base.allocation,
                           stint_cap=base.stint_cap, traffic_s_per_lap=base.dirty_air)
    out["strategy"] = {
        "best": best["label"], "best_plan": {**res.best, "label": best["label"]}, "push": push,
        "implied_regime": float(res.implied_regime), "n_strategies": int(res.n_strategies),
        "n_scored": int(res.n_scored), "n_draws": int(res.n_draws), "max_stint": res.max_stint,
        "p_stops": {str(k): v for k, v in res.p_stops.items()}, "by_stops": by_stops, "by_start": by_start,
        "pit_windows": windows, "life": life,
        "top": [_plan_dict(r) for _, r in res.table.head(12).iterrows()],
        "grip_budget_s": float(model.budget), "grip_budgets": dict(model.budgets),
        "tyre_optimal": {**res.tyre_optimal, "label": res.tyre_optimal_label},
        "position_s": float(res.best.get("position_s", 0.0)), "prior_s": float(res.best.get("prior_s", 0.0)),
        "first_stop_s": float(res.best.get("first_stop_s", 0.0)),
        "undercut_lambda": float(res.undercut_lambda), "plan_prior_tau_s": float(res.plan_prior_tau_s),
        "first_stop_kappa_s": float(res.first_stop_kappa_s),
    }
    out["alternatives"] = alts
    out["voi"] = voi
    out["programme"] = programme
    out["sc_playbook"] = {"ranges": strat.playbook_ranges(pb),
                          "rows": [{k: (v if k != "further_stops" else list(v)) for k, v in r.items()}
                                   for r in pb.to_dict("records")]}
    out["scenarios"] = {} if quick else _scenarios(model, ev, base, best, sim_kw)
    out["runtime_s"] = round(time.time() - t0, 1)
    if write:
        _write(ev, out, model, res, pw, uc)
    return out


def _write(ev: Event, out: dict, model: TyreModel, res, pw: pd.DataFrame, uc: pd.DataFrame) -> None:
    key = ev.key
    atomic_write(outlook_path(key), json.dumps(out, indent=1, default=_json_default))
    arrays = {f"wear_{c}": model.wear_rate[c] for c in model.compounds}
    arrays.update({f"pace_{c}": model.pace_offset[c] for c in model.compounds})
    arrays.update({f"budget_{c}": np.array(model.budget_of(c)) for c in model.compounds})
    np.savez_compressed(draws_path(key), compounds=np.array(model.compounds), budget=np.array(model.budget),
                        load_exponent=np.array(model.load_exponent),
                        manage_floor=np.array(model.manage_floor), manage_cost_s=np.array(model.manage_cost_s), **arrays)
    if res is not None and not res.table.empty:
        t = res.table.head(400).copy()
        t["pit_laps"] = t["pit_laps"].astype(str)
        t["stint_lens"] = t["stint_lens"].astype(str)
        t.to_parquet(DATA_PROCESSED / f"outlook_{key}_strategy.parquet", index=False)
        b = res.by_stops.copy()
        b["pit_laps"] = b["pit_laps"].astype(str)
        b["stint_lens"] = b["stint_lens"].astype(str)
        b.to_parquet(DATA_PROCESSED / f"outlook_{key}_bystops.parquet", index=False)
        res.life.to_parquet(DATA_PROCESSED / f"outlook_{key}_life.parquet", index=False)
        rows, lap0 = [], 0
        for i, (comp, L) in enumerate(zip(res.best["compounds"], res.best["stint_lens"])):
            rows.append({"stint": i + 1, "compound": comp, "start_lap": lap0 + 1, "end_lap": lap0 + L, "laps": int(L)})
            lap0 += L
        pd.DataFrame(rows).to_parquet(DATA_PROCESSED / f"outlook_{key}_plan.parquet", index=False)
        if pw is not None and not pw.empty:
            pw.to_parquet(DATA_PROCESSED / f"outlook_{key}_pitwindow.parquet", index=False)
        if uc is not None and not uc.empty:
            uc.to_parquet(DATA_PROCESSED / f"outlook_{key}_undercut.parquet", index=False)
    # the timeline: one compact line per build
    st = out.get("strategy") or {}
    rec = {"utc": out["updated_utc"], "stage": out["stage"], "stage_label": out["stage_label"],
           "session": (out.get("live") or {}).get("session_name") or None,
           "live_used": bool((out.get("live") or {}).get("used")),
           "n_practice_laps": out.get("n_practice_laps", 0),
           "n_long_runs": (out.get("live") or {}).get("n_long_runs", 0),
           "best": st.get("best"), "n_stops": (st.get("best_plan") or {}).get("n_stops"),
           "stint_lens": (st.get("best_plan") or {}).get("stint_lens"), "push": st.get("push"),
           "tyre_optimal": (st.get("tyre_optimal") or {}).get("label"),
           "p_stops": st.get("p_stops", {}),
           "life": {c: {"mean": v["life_laps"], "lo": v["life_lo"], "hi": v["life_hi"]} for c, v in (st.get("life") or {}).items()},
           "deg": {c: v["deg_s_per_lap"] for c, v in (st.get("life") or {}).items()},
           "pit_loss_s": out.get("pit_loss_s"), "evpi_s": (out.get("voi") or {}).get("evpi_s"),
           "voi": {c: v["gain_s"] for c, v in ((out.get("voi") or {}).get("by_compound") or {}).items()},
           "track_temp_c": (out.get("live") or {}).get("track_temp_c"),
           "robust": (out.get("scenarios") or {}).get("robust")}
    with open(timeline_path(key), "a") as f:
        f.write(json.dumps(rec, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


# --------------------------------------------------------------------------
# Readers, for the app
# --------------------------------------------------------------------------


def load_outlook(key: str) -> dict | None:
    p = outlook_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_model(key: str) -> TyreModel | None:
    p = draws_path(key)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        comps = [str(c) for c in z["compounds"]]
        budgets = {c: float(z[f"budget_{c}"]) for c in comps if f"budget_{c}" in z.files}
        return TyreModel(compounds=comps,
                         wear_rate={c: z[f"wear_{c}"] for c in comps},
                         pace_offset={c: z[f"pace_{c}"] for c in comps},
                         budget=float(z["budget"]), budgets=budgets, load_exponent=float(z["load_exponent"]),
                         n_draws=int(len(z[f"wear_{comps[0]}"])), source=f"outlook {key}",
                         manage_floor=float(z["manage_floor"]) if "manage_floor" in z.files else 0.45,
                         manage_cost_s=float(z["manage_cost_s"]) if "manage_cost_s" in z.files else 0.9)


def load_timeline(key: str) -> list:
    p = timeline_path(key)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def outlook_events() -> list:
    """Event keys with an outlook on disk."""
    return sorted(p.name[len("outlook_"):-len(".json")] for p in DATA_PROCESSED.glob("outlook_*.json")
                  if not p.name.endswith("_timeline.json"))
