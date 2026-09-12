"""Strategy-intelligence benchmark: was the recommendation the right decision?

Per weekend, against the classified finishers of the real race:

  * stop count: recommended vs the field's mode and vs the winner
  * compound sequence: is the recommended sequence one anybody ran?  the
    winner's?  what share of the field ran it?  what did the top 5 run?
  * starting compound: matches the winner / the majority?
  * first stop, safety-car aware: the recommended lap, the tyre-optimal lap
    and V2's recommendation against the field's median *green-flag* first
    stop; the seconds of first-stop prior the chosen plan carries; share of
    the field's in-laps inside the model's window
  * the cliff detector: where the shipped grip budget says each compound
    collapses against where `cliff.race_collapses` says it did, and the
    collapse/strategic/undetermined split per compound
  * tyre life: predicted life per compound (bounded by the race and the
    circuit's history, as the tool now states it) and the model's own
    uncapped quotient, vs the longest and p90 stint actually run
  * oracle regret: every candidate plan re-priced on a tyre model whose
    degradation rates are the ones *measured on this race*, with the model's
    own grip budgets and pace offsets
  * the position-aware regret `R_pos` (V4, `docs/v4_methodology.md`): the same
    oracle race time plus what the first pit cycle does to track position
    against the field as it actually stopped, at the weekend's leave-one-out
    place value — the one term the pure-time oracle has no notion of
  * Haas: OCO's and BEA's own per-car plans against their own green first
    stops, and each car's expected places lost through the first cycle
  * per-car plans: how many drivers' own plans share the field plan's shape,
    and whether the per-driver first-stop spread tracks what those drivers did
  * counterfactual sanity: implausible "seconds lost" figures, and how many
    drivers' stops were held fixed for a safety car
"""

from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd
from scipy.special import ndtr

from common import (OUT, arg_events, baseline_meta, baseline_processed, driver_plans, dump,  # noqa: E402
                    memoise_regime, meta, offline, race_results, race_table, v2_meta, v2_processed)
from src import racestate, strategy as strat
from src.calibration import get_calibration
from src.compounds import hardness_rank
from src.config import (DATA_PROCESSED, DIRTY_AIR_S_PER_LAP, GRID_START_PENALTY_S,
                        SC_PIT_LOSS_FRACTION, SC_RATE_PER_LAP, TRAFFIC_LAPS_PER_STOP, get_event)
from src.history import race_deg_slopes
from src.model_bayes import BayesFit
from src.tyre import TyreModel

# The team the tool is built for, and its two cars in every 2026 lap table.
HAAS_TEAM = "Haas F1 Team"
HAAS_DRIVERS = ("OCO", "BEA")

# The plans the position-aware block reports on; every one of them is also in
# the candidate set `C`, so `R_pos >= 0` holds by construction.
POS_LABELS = ("tool", "tyre_optimal", "field_modal", "winner", "oracle_opt")
N_PACK_SLOTS = 4          # two rivals ahead, two behind (`racestate.pack_slots`)


def accuracy_percar(key: str) -> dict:
    """The three per-car Spearman variants, from `accuracy.json` if it has run.

    Stint ranking is an accuracy question measured on 28-43 stints a weekend;
    re-deriving it here would be the same arithmetic on the same curves, so the
    strategy report quotes it rather than recomputing it, and says so when the
    accuracy stage has not run yet.
    """
    p = OUT / "accuracy.json"
    if not p.exists():
        return {"source": "accuracy.json has not been written yet"}
    try:
        v = (json.loads(p.read_text()).get("per_event") or {}).get(key) or {}
    except Exception as exc:
        return {"source": f"accuracy.json unreadable: {exc}"}
    var = v.get("variants") or {}
    out = {"source": "bench/out/accuracy.json"}
    for name in ("sealed", "sealed_driver_hist", "sealed_driver_practice_dev", "sealed_driver_team_pooled"):
        if name in var:
            out[name] = {"spearman": var[name].get("spearman"), "rate_mae": var[name].get("rate_mae")}
    out["scales"] = v.get("percar_scales")
    return out


def cliff_rows(m: dict, key: str, ev) -> dict:
    """The cliff detector's race rows and the budget's predicted collapse lap.

    Read from `meta["cliff_detector"]` where the V3 decide stage wrote it; a V2
    meta has no such block, and `cliff.race_collapses` is then run here on the
    race lap table so the metric exists for both builds (and so the numbers can
    be compared with the ones the pipeline recorded).
    """
    block = dict(m.get("cliff_detector") or {})
    rows = block.get("rows")
    source = "meta[cliff_detector]"
    if not rows:
        try:
            from src.cliff import race_collapses
            rows = race_collapses(race_table(key), event=ev).to_dict("records")
            source = "bench: cliff.race_collapses on the race lap table"
        except Exception as exc:
            return {"source": f"unavailable: {exc}"}
    df = pd.DataFrame(rows or [])
    if df.empty:
        return {"source": source, "n": 0}
    by_comp = {}
    for c, g in df.groupby("compound"):
        kinds = g["kind"].value_counts().to_dict()
        coll = g[g["collapse"].fillna(False).astype(bool)] if "collapse" in g else g.iloc[0:0]
        by_comp[str(c)] = {"n_stints": int(len(g)),
                           "kinds": {str(k): int(v) for k, v in kinds.items()},
                           "n_collapse": int(len(coll)),
                           "observed_knee_mean": (float(pd.to_numeric(coll["knee_age"], errors="coerce").mean())
                                                  if len(coll) else None),
                           "observed_knees": [float(x) for x in pd.to_numeric(coll.get("knee_age"), errors="coerce").dropna()]
                           if len(coll) else [],
                           "cum_loss_at_knee_mean_s": (float(pd.to_numeric(coll["cum_loss_at_knee_s"], errors="coerce").mean())
                                                       if len(coll) else None)}
    out = {"source": source, "n": int(len(df)),
           "kinds": {str(k): int(v) for k, v in df["kind"].value_counts().items()},
           "n_collapse": int(df["collapse"].fillna(False).astype(bool).sum()) if "collapse" in df else 0,
           "by_compound": by_comp}
    # predicted collapse lap = budget / the model's full-push rate, per compound.
    # The decide stage writes it as `budget_vs_observed`; `budget_ratio_metrics`
    # is the function's own name and is accepted too.
    ratios = block.get("budget_vs_observed") or block.get("budget_ratio_metrics")
    if not ratios:
        try:
            from src.cliff import budget_ratio_metrics
            budgets = (m["strategy"].get("grip_budgets") or {})
            life = {x["compound"]: x for x in m["strategy"]["life"]}
            rates = {c: life.get(c, {}).get("deg_s_per_lap") for c in budgets}
            rates = {c: v for c, v in rates.items() if v}
            if budgets and rates:
                ratios = budget_ratio_metrics(budgets, rates, df)
                out["budget_ratio_source"] = "bench: cliff.budget_ratio_metrics on the shipped budgets and life rates"
        except Exception as exc:
            out["budget_ratio_source"] = f"unavailable: {exc}"
    else:
        out["budget_ratio_source"] = "meta[cliff_detector][budget_vs_observed]"
    out["budget_ratio_metrics"] = ratios
    if ratios:
        pooled = ratios.get("_pooled") or {}
        out["over"] = pooled.get("over")
        out["under"] = pooled.get("under")
        out["mean_abs_error_laps"] = pooled.get("mean_abs_error_laps")
    return out


def oracle_model(key: str, m: dict, n: int = 200, seed: int = 0) -> tuple:
    ev = get_event(key)
    r = race_table(key)
    r = r[r["is_accurate"] & ~r["pit_in"] & ~r["pit_out"] & (r["track_status"].astype(str) == "1")]
    d = race_deg_slopes(r, ev.fuel_effect_s_per_lap)
    rng = np.random.default_rng(seed)
    budgets = (m["strategy"].get("grip_budgets") or {})
    pace = (m.get("pace_calibration") or {}).get("offsets_after") or m["bayes"]["comp_offset"]
    comps = [c for c in m["bayes"]["comp_offset"] if c in d]
    wear, po = {}, {}
    for c in comps:
        s = max(d[c]["slope"], 0.005)
        draws = np.exp(rng.normal(np.log(s), max(d[c]["se"], 0.005) / s, size=n))
        wear[c] = draws / float(budgets.get(c, m["strategy"]["grip_budget_s"]))
        po[c] = np.full(n, float(pace.get(c, 0.0)))
    return TyreModel(compounds=comps, wear_rate=wear, pace_offset=po, budget=float(m["strategy"]["grip_budget_s"]),
                     budgets={c: float(budgets.get(c, m["strategy"]["grip_budget_s"])) for c in comps},
                     n_draws=n, source="race-measured (oracle)"), d


# --------------------------------------------------------------------------
# The position-aware strategic metric (WP-C1; definition: docs/v4_methodology.md)
#
# The pure-time oracle regret above prices every candidate on a tyre model that
# knows this race's measured degradation rates - and nothing else.  It has no
# notion of a place, so a plan that spends two seconds of race time to keep
# track position is a regression by its measure, which is precisely the trade
# V4's race state exists to make.  This block adds the missing term:
#
#     J(p)     = T_oracle(p) + V * L(p)
#     R_pos(p) = J(p) - min over the candidate set of J
#
# `L(p)` is the expected number of places the first pit cycle costs against the
# field as it actually stopped, on the four measured pack slots.  Nothing here
# touches the pure-time block: it is a second score over the same candidates,
# written into `r["oracle"]["position_aware"]`.
# --------------------------------------------------------------------------


def race_state_constants(key: str):
    """This weekend's leave-one-out race-state constants (V and sigma_rel).

    `estimator="regularized"` is WP-A's V4 production estimator.  A checkout
    whose `measure_constants` does not take the keyword yet (Task 1's does not),
    or spells the estimator differently, is asked without it, so the metric
    scores either build - and the estimator actually used is reported as
    `estimator` beside the numbers rather than assumed.
    """
    try:
        return racestate.measure_constants(exclude=key, estimator="regularized")
    except Exception:
        return racestate.measure_constants(exclude=key)


def mean_cost_tables(model: TyreModel, ev, push: float = 1.0, max_len: int | None = None) -> dict:
    """`means[c][s, L]`: expected seconds an L-lap stint on `c` costs when it
    starts after `s` laps, at `push` - `strategy.stint_cost_table` averaged over
    the posterior draws.  The same table `_phase1_race_state` prices its pack on.
    """
    n = int(ev.n_race_laps)
    max_len = n if max_len is None else int(max_len)
    tab = strat.stint_cost_table(model, ev, max_len, float(push))
    return {c: np.asarray(v, dtype=float).mean(0) for c, v in tab.items()}


def field_second_by_start(plans: pd.DataFrame) -> dict:
    """The modal second compound the field ran from each start compound.

    This is the rival's set in `D(s, l)`: a rival that started where we did and
    then did what most of the field that started there did.
    """
    out = {}
    if plans is None or not len(plans):
        return out
    parts = [str(s).split("-") for s in plans["seq"]]
    seconds = {}
    for p in parts:
        if len(p) >= 2:
            seconds.setdefault(p[0], []).append(p[1])
    for start, xs in seconds.items():
        out[start] = pd.Series(xs).value_counts().index[0]
    return out


class PositionModel:
    """`D`, `L` and `J` for one weekend, on the oracle tyre model at push 1.

    `means` is the oracle's mean stint-cost table, `const` the weekend's
    leave-one-out race-state constants, `field_stops` the field's green-flag
    first stops (classified finishers, in-laps not under a safety car) and
    `second_by_start` the rival's second compound per start compound.

    The pack slots are `racestate.pack_slots`: two rivals ahead and two behind
    at the measured first-stint intervals, each as equal-mass quantiles, so a
    slot contributes the mean over its quantiles and the four slots sum - the
    same weighting `pack_equilibrium` uses.
    """

    def __init__(self, means: dict, const, pit_loss_s: float, field_stops, second_by_start: dict):
        self.means = means
        self.pit_loss_s = float(pit_loss_s)
        self.field_stops = np.asarray(sorted(int(x) for x in (field_stops or [])), dtype=int)
        self.second_by_start = dict(second_by_start or {})
        self.const = const
        self.V = float(const.place_value_s)
        self.sigma = float(const.sigma_rel_s)
        gaps, slot = racestate.pack_slots(const)
        self.gaps = np.asarray(gaps, dtype=float)
        self.slot = np.asarray(slot, dtype=int)
        n_q = max(1, int(np.bincount(self.slot).max()))
        self.weight = np.full(len(self.gaps), 1.0 / n_q)
        self.base_ahead = ndtr(self.gaps / self.sigma)

    # -- D(s, l) ----------------------------------------------------------
    def delta(self, start: str, second: str, s: int, l, rival_second: str | None = None):
        """`D(s, l) = A_me(s) - A_r(l)`, both cars' race time from lap 0 to
        `max(s, l) + 1` - the lap both are out of the pits - each on its own set,
        pit loss included (it cancels: both cars stop inside the cycle)."""
        stay = self.means[start][0]                      # the start set, from lap 0
        fresh_me = self.means[second]
        fresh_r = self.means[rival_second or second]
        s = int(s)
        l = np.asarray(l, dtype=int)
        M = np.maximum(s, l) + 1
        a_me = (stay[np.clip(s, 0, len(stay) - 1)]
                + fresh_me[np.clip(s, 0, fresh_me.shape[0] - 1),
                           np.clip(M - s, 0, fresh_me.shape[1] - 1)]
                + self.pit_loss_s)
        a_r = (stay[np.clip(l, 0, len(stay) - 1)]
               + fresh_r[np.clip(l, 0, fresh_r.shape[0] - 1),
                         np.clip(M - l, 0, fresh_r.shape[1] - 1)]
               + self.pit_loss_s)
        return a_me - a_r

    # -- L(p) -------------------------------------------------------------
    def places_lost(self, start: str, second: str, s: int, *, rival_second: str | None = None,
                    field_stops=None) -> float | None:
        """`L(p)`: expected places lost through the first pit cycle, of four.

        Negative is places gained.  None when the weekend has no green first
        stop to score against, or the plan's compounds are not in the model.
        """
        stops = self.field_stops if field_stops is None else np.asarray(field_stops, dtype=int)
        if start not in self.means or second not in self.means or not len(stops):
            return None
        rs = rival_second or self.second_by_start.get(start) or second
        if rs not in self.means:
            rs = second
        D = np.atleast_1d(self.delta(start, second, int(s), stops, rs))
        P = ndtr((self.gaps[:, None] + D[None, :]) / self.sigma)
        return float(((P - self.base_ahead[:, None]) * self.weight[:, None]).sum(0).mean())

    def j(self, T, L) -> float | None:
        """`J(p) = T_oracle(p) + V L(p)`."""
        if T is None or L is None:
            return None
        return float(T) + self.V * float(L)

    def p_retain(self, L) -> float | None:
        """`1 - L/4`, clipped: the share of the four pack slots we hold."""
        if L is None:
            return None
        return float(np.clip(1.0 - float(L) / float(N_PACK_SLOTS), 0.0, 1.0))


def pos_regret(J: dict) -> tuple:
    """`R_pos(p) = J(p) - min over the candidate set of J`, and the best label.

    Every candidate's `R_pos` is >= 0 and the best candidate's is exactly 0,
    which is what makes the number a regret rather than a score.
    """
    js = {}
    for k, v in (J or {}).items():
        if v is None:
            continue
        v = float(v)
        if np.isfinite(v):
            js[k] = v
    if not js:
        return {}, None
    best = min(js, key=js.get)
    return {k: float(v - js[best]) for k, v in js.items()}, best


def family_first_stop_candidates(model: TyreModel, ev, seq, pit_loss_s: float, means: dict, *,
                                 alloc=None, caps=None, push: float = 1.0,
                                 traffic_s_per_lap: float | None = None) -> list:
    """One plan per legal first-stop lap of `seq`'s family, later stops re-optimised.

    The stint-length grid is `strategy.enumerate_strategies` on the family's own
    compounds with the search's margin, allocation and circuit stint caps; each
    first-stop lap keeps the composition with the lowest pure-time oracle cost
    (tyre + pit lane + rejoin traffic + safety-car credit + grid penalty, no
    prior and no position term - the terms `simulate_model` prices in phase 1).
    The chosen plans are re-priced exactly by `evaluate_plans` afterwards, so
    this sweep only ever *selects* the later stops.
    """
    seq = [str(c).upper() for c in seq]
    if len(seq) < 2 or any(c not in means for c in seq):
        return []
    n = int(ev.n_race_laps)
    seqs, lens_all, starts_all = strat.enumerate_strategies(
        n, sorted(set(seq)), max_stops=len(seq) - 1,
        max_per_compound=(alloc if alloc else {}), max_stint=(caps or None))
    idx = next((i for i, s in enumerate(seqs) if list(s) == seq), None)
    if idx is None:
        return []
    lens, starts = lens_all[idx], starts_all[idx]
    pits = starts[:, 1:]
    if not pits.shape[1]:
        return []
    dens = strat.traffic_density(ev)
    rank = dict(zip(model.compounds, hardness_rank(list(model.compounds))))
    traffic = DIRTY_AIR_S_PER_LAP if traffic_s_per_lap is None else float(traffic_s_per_lap)
    T = np.full(len(lens), (len(seq) - 1) * float(pit_loss_s), dtype=float)
    T += TRAFFIC_LAPS_PER_STOP * traffic * dens[np.clip(pits, 1, n) - 1].sum(1)
    T -= ((1.0 - np.exp(-SC_RATE_PER_LAP * pits.max(1).astype(float)))
          * (1.0 - SC_PIT_LOSS_FRACTION) * float(pit_loss_s))
    T += GRID_START_PENALTY_S * float(rank.get(seq[0], 0))
    for k, c in enumerate(seq):
        T += means[c][starts[:, k], lens[:, k]]
    out = []
    for s in np.unique(pits[:, 0]):
        sel = np.flatnonzero(pits[:, 0] == s)
        i = int(sel[int(np.argmin(T[sel]))])
        out.append({"label": f"family@{int(s)}", "compounds": list(seq),
                    "pit_laps": [int(x) for x in pits[i]], "push": float(push)})
    return out


def position_block(key: str, ev, orc: TyreModel, cand: list, plans: pd.DataFrame,
                   pit_loss_s: float, alloc, caps, green_first: list) -> tuple:
    """`r["oracle"]["position_aware"]`, and the `PositionModel` it was built on.

    `cand` is the pure-time block's candidate list (same dicts, same labels);
    the candidate set `C` is every legal first-stop lap of the tool's family
    with later stops re-optimised on the oracle, plus every reported plan, so
    `R_pos >= 0` and the best candidate's `R_pos` is 0 by construction.
    """
    const = race_state_constants(key)
    means = mean_cost_tables(orc, ev, push=1.0)
    pm = PositionModel(means, const, pit_loss_s, green_first, field_second_by_start(plans))
    named = {c["label"]: c for c in cand if c.get("label") in POS_LABELS}
    tool = named.get("tool")
    fam = (family_first_stop_candidates(orc, ev, tool["compounds"], pit_loss_s, means,
                                        alloc=alloc, caps=caps) if tool else [])
    pool = {**{f["label"]: f for f in fam}, **named}           # a named plan wins a label clash
    _, det = strat.evaluate_plans(orc, ev, list(pool.values()), pit_loss_s, push=1.0,
                                  allocation=alloc, stint_cap=caps or {})
    T = {d["label"]: float(d["times"].mean()) for d in det if d.get("valid")}
    L, J = {}, {}
    for lab, p in pool.items():
        cs = [str(c).upper() for c in p.get("compounds") or []]
        pl = list(p.get("pit_laps") or [])
        if len(cs) < 2 or not pl or lab not in T:
            continue
        L[lab] = pm.places_lost(cs[0], cs[1], int(pl[0]))
        j = pm.j(T[lab], L[lab])
        if j is not None:
            J[lab] = j
    R, best_lab = pos_regret(J)
    ref = J.get(best_lab) if best_lab is not None else None

    def _r(x, nd=3):
        return None if x is None else round(float(x), nd)

    out = {
        "definition": "J(p) = T_oracle(p) + V L(p); R_pos(p) = J(p) - min_C J  (docs/v4_methodology.md)",
        "estimator": getattr(const, "estimator", None),
        "constants_source": const.source,
        "V_s": _r(pm.V), "sigma_rel_s": _r(pm.sigma),
        "place_gap_s": _r(const.place_gap_s), "persistence": _r(const.persistence),
        "n_field_first_stops": int(len(pm.field_stops)),
        "field_first_stops": [int(x) for x in pm.field_stops],
        "pack_gaps_s": [_r(g, 2) for g in pm.gaps[::max(1, len(pm.gaps) // 8)]],
        "rival_second_by_start": pm.second_by_start,
        "family": (list(tool["compounds"]) if tool else None),
        "n_candidates": len(J), "n_family_candidates": len(fam),
        "T_oracle_s": {k: _r(T.get(k), 2) for k in POS_LABELS},
        "first_stop": {k: (int(named[k]["pit_laps"][0]) if (k in named and named[k].get("pit_laps"))
                           else None) for k in POS_LABELS},
        "L": {k: _r(L.get(k), 4) for k in POS_LABELS},
        "J": {k: _r(J.get(k), 2) for k in POS_LABELS},
        "R_pos": {k: _r(R.get(k), 2) for k in POS_LABELS},
        "P_retain_tool": _r(pm.p_retain(L.get("tool")), 4),
        "best_candidate": best_lab,
        "best_candidate_first_stop": (int(pool[best_lab]["pit_laps"][0])
                                      if (best_lab and pool[best_lab].get("pit_laps")) else None),
        "best_candidate_J_s": _r(ref, 2),
        "family_sweep": [{"first_stop": int(f["pit_laps"][0]), "pit_laps": f["pit_laps"],
                          "T_s": _r(T.get(f["label"]), 2), "L": _r(L.get(f["label"]), 4),
                          "J_s": _r(J.get(f["label"]), 2), "R_pos_s": _r(R.get(f["label"]), 2)}
                         for f in fam if f["label"] in J],
    }
    if not len(pm.field_stops):
        out["note"] = ("no green-flag first stop among the classified finishers: "
                       "L is undefined for this weekend")
    return out, pm


def _plan_pits(row: dict) -> list:
    """`meta["per_driver"]`'s `pit_laps`, which a meta round-trip stringifies."""
    v = (row or {}).get("pit_laps")
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    try:
        return [int(x) for x in ast.literal_eval(str(v))]
    except Exception:
        return []


def haas_block(m: dict, plans: pd.DataFrame, race: pd.DataFrame, pm: PositionModel | None) -> dict:
    """OCO and BEA: their own per-car plan against their own race (WP-C2).

    For each car: the per-car plan's first stop, the driver's actual *green*
    first stop, whether the per-car plan shares the field plan's shape, and the
    car's own `L` - the same first-cycle place cost as C1, evaluated at the
    per-car plan's start compound, second compound and first stop.
    """
    per = {str(r.get("driver")): r for r in (m.get("per_driver") or [])}
    act = plans.set_index("driver") if (plans is not None and len(plans)) else None
    teams = (race.groupby("driver")["team"].first().to_dict() if "team" in race.columns else {})
    drivers = {}
    for drv in HAAS_DRIVERS:
        row = per.get(drv) or {}
        comps = [c for c in str(row.get("compounds") or "").split("-") if c]
        first = row.get("first_stop")
        first = int(first) if first is not None and np.isfinite(float(first)) else None
        a = (act.loc[drv].to_dict() if (act is not None and drv in act.index) else {})
        a_pits = [int(x) for x in (a.get("pit_laps") or [])]
        a_first = a_pits[0] if a_pits else None
        a_sc = bool(a.get("first_sc")) if a_pits else None
        L = (pm.places_lost(comps[0], comps[1], first)
             if (pm is not None and len(comps) >= 2 and first is not None) else None)
        drivers[drv] = {
            "team": teams.get(drv), "in_per_driver": bool(row), "classified": bool(a),
            "plan": row.get("best"), "compounds": row.get("compounds"),
            "n_stops": row.get("n_stops"), "pit_laps": _plan_pits(row),
            "first_stop": first, "push": row.get("push"),
            "same_shape_as_field": row.get("same_shape_as_field"),
            "race_factor": row.get("race_factor"),
            "actual_seq": a.get("seq"), "actual_stops": (int(a["n_stops"]) if a else None),
            "actual_pit_laps": a_pits, "actual_first_stop": a_first, "actual_first_sc": a_sc,
            # the timing metric: green first stops only, as everywhere else in the suite
            "first_minus_actual_green": ((first - a_first)
                                         if (first is not None and a_first is not None and not a_sc)
                                         else None),
            "first_minus_actual": ((first - a_first)
                                   if (first is not None and a_first is not None) else None),
            "seq_match_actual": ((str(row.get("compounds")) == str(a.get("seq")))
                                 if (row and a) else None),
            "stops_match_actual": ((int(row["n_stops"]) == int(a["n_stops"]))
                                   if (row.get("n_stops") is not None and a) else None),
            "L": (round(float(L), 4) if L is not None else None),
            "P_retain": (round(float(pm.p_retain(L)), 4) if (pm is not None and L is not None) else None),
        }
    ok = [d["first_minus_actual_green"] for d in drivers.values() if d["first_minus_actual_green"] is not None]
    return {"team": HAAS_TEAM, "drivers": drivers,
            "n_scored_green": len(ok),
            "mean_abs_first_stop_err": (float(np.mean([abs(x) for x in ok])) if ok else None),
            "n_same_shape_as_field": int(sum(bool(d["same_shape_as_field"]) for d in drivers.values())),
            "mean_L": (float(np.mean([d["L"] for d in drivers.values() if d["L"] is not None]))
                       if any(d["L"] is not None for d in drivers.values()) else None)}


def window_rows(pw: list, same: pd.DataFrame, best: dict) -> list:
    rows = []
    for w in pw:
        k = w["stop"] - 1
        cand = same[[len(p) > k for p in same["pit_laps"]]]
        # green-flag stops only for the first stop: a safety-car stop was not a timing decision
        if k == 0:
            cand = cand[~cand["first_sc"]]
        actual = [p[k] for p in cand["pit_laps"]]
        if not actual:
            continue
        inside = np.mean([(w["lo"] <= a <= w["hi"]) for a in actual])
        rows.append({"stop": w["stop"], "recommended": w["recommended"], "lo": w["lo"], "hi": w["hi"],
                     "n_actual": len(actual), "share_inside": float(inside),
                     "median_abs_err": float(np.median([abs(a - w["recommended"]) for a in actual])),
                     "median_err": float(np.median([w["recommended"] - a for a in actual])),
                     "actual_median": float(np.median(actual))})
    return rows


def _first(plan: dict | None) -> int | None:
    p = (plan or {}).get("pit_laps") or []
    return int(p[0]) if p else None


def main() -> None:
    args = arg_events(__doc__)
    offline()
    memoise_regime()
    out = {}
    rows = []
    for key in args.events:
        ev = get_event(key)
        m = meta(key)
        bm = baseline_meta(key)          # V1
        v2m = v2_meta(key)               # V2
        cal = get_calibration(ev)
        race = race_table(key)
        plans = driver_plans(race, ev.n_race_laps)
        res = race_results(key)
        if res is not None:
            plans = plans.merge(res, on="driver", how="left").sort_values("position")
        st = m["strategy"]
        best = st["best_plan"]
        tyre_opt = st.get("tyre_optimal") or {}
        rec_seq = "-".join(best["compounds"])
        n_cls = len(plans)
        winner = plans.iloc[0] if res is not None and len(plans) else None
        top5 = plans.head(5) if res is not None else plans.iloc[0:0]

        stop_counts = plans["n_stops"].value_counts().to_dict()
        mode_stops = int(plans["n_stops"].mode().iloc[0]) if n_cls else None
        seq_counts = plans["seq"].value_counts()
        modal_seq = seq_counts.index[0] if len(seq_counts) else None
        start_counts = plans["seq"].str.split("-").str[0].value_counts()
        green_first = [p[0] for p, s in zip(plans["pit_laps"], plans["first_sc"]) if p and not s]
        sc_share_first = float(plans["first_sc"].mean()) if n_cls else None
        fmed = float(np.median(green_first)) if green_first else None

        r = {
            "recommended": st["best"], "rec_stops": int(best["n_stops"]), "rec_seq": rec_seq,
            "rec_push": float(best.get("push", np.nan)),
            "tyre_optimal": tyre_opt.get("label"), "tyre_optimal_delta_s": tyre_opt.get("delta_s"),
            "position_s": st.get("position_s"), "prior_s": st.get("prior_s"),
            "n_classified": n_cls, "field_stop_counts": {int(k): int(v) for k, v in stop_counts.items()},
            "mode_stops": mode_stops, "stops_match_mode": bool(best["n_stops"] == mode_stops),
            "stops_share": float(stop_counts.get(best["n_stops"], 0) / max(n_cls, 1)),
            "field_modal_seq": modal_seq, "field_seq_counts": {k: int(v) for k, v in seq_counts.head(6).items()},
            "rec_seq_share": float(seq_counts.get(rec_seq, 0) / max(n_cls, 1)),
            "rec_seq_run_by_anyone": bool(seq_counts.get(rec_seq, 0) > 0),
            "start_compound": best["compounds"][0],
            "field_start_counts": {k: int(v) for k, v in start_counts.items()},
            "start_matches_majority": bool(start_counts.index[0] == best["compounds"][0]) if len(start_counts) else None,
            "winner": (winner["driver"] if winner is not None else None),
            "winner_seq": (winner["seq"] if winner is not None else None),
            "winner_stops": (int(winner["n_stops"]) if winner is not None else None),
            "winner_pits": (list(winner["pit_laps"]) if winner is not None else None),
            "stops_match_winner": (bool(best["n_stops"] == winner["n_stops"]) if winner is not None else None),
            "seq_match_winner": (bool(rec_seq == winner["seq"]) if winner is not None else None),
            "top5_seqs": (top5["seq"].tolist() if len(top5) else []),
            "seq_in_top5": bool(rec_seq in set(top5["seq"])) if len(top5) else None,
            "first_stop": {"field_median_green": fmed, "share_under_sc": sc_share_first,
                           "sc_set": bool(sc_share_first is not None and sc_share_first > 0.4),
                           "recommended": _first(best),
                           "tyre_optimal": _first(tyre_opt),
                           "rec_minus_field": ((_first(best) - fmed) if (fmed is not None and best["pit_laps"]) else None),
                           "tyre_minus_field": ((_first(tyre_opt) - fmed) if (fmed is not None and tyre_opt.get("pit_laps")) else None),
                           # V1
                           "baseline_rec": (_first(bm["strategy"]["best_plan"]) if bm else None),
                           "baseline_minus_field": ((_first(bm["strategy"]["best_plan"]) - fmed)
                                                    if (bm and fmed is not None and bm["strategy"]["best_plan"]["pit_laps"]) else None),
                           # V2
                           "v2_rec": (_first(v2m["strategy"]["best_plan"]) if v2m else None),
                           "v2_minus_field": ((_first(v2m["strategy"]["best_plan"]) - fmed)
                                              if (v2m and fmed is not None and v2m["strategy"]["best_plan"]["pit_laps"]) else None),
                           # V3's new term: the seconds of first-stop prior the chosen plan pays,
                           # and the prior's own shape.  Absent on a V2 meta (kappa = 0 there).
                           "first_stop_s": (best.get("first_stop_s") if best.get("first_stop_s") is not None
                                            else st.get("first_stop_s")),
                           "first_stop_kappa_s": (st.get("first_stop_kappa_s")
                                                  if st.get("first_stop_kappa_s") is not None
                                                  else getattr(cal, "first_stop_kappa_s", None)),
                           "prior": m.get("first_stop_prior"),
                           "prior_from_history": ((m.get("circuit_history") or {}).get("first_stop_green") or {}).get("median_lap")},
            "baseline": ({"recommended": bm["strategy"]["best"], "seq_share": float(seq_counts.get("-".join(bm["strategy"]["best_plan"]["compounds"]), 0) / max(n_cls, 1)),
                          "start_matches_majority": bool(start_counts.index[0] == bm["strategy"]["best_plan"]["compounds"][0]) if len(start_counts) else None,
                          "stops_match_mode": bool(bm["strategy"]["best_plan"]["n_stops"] == mode_stops)} if bm else None),
            "v2": ({"recommended": v2m["strategy"]["best"],
                    "seq_share": float(seq_counts.get("-".join(v2m["strategy"]["best_plan"]["compounds"]), 0) / max(n_cls, 1)),
                    "seq_run_by_anyone": bool(seq_counts.get("-".join(v2m["strategy"]["best_plan"]["compounds"]), 0) > 0),
                    "start_matches_majority": bool(start_counts.index[0] == v2m["strategy"]["best_plan"]["compounds"][0]) if len(start_counts) else None,
                    "stops_match_mode": bool(v2m["strategy"]["best_plan"]["n_stops"] == mode_stops)} if v2m else None),
        }

        # -- pit windows vs actual in-laps of drivers on the same stop count --
        same = plans[plans["n_stops"] == best["n_stops"]]
        r["pit_windows"] = window_rows(st.get("pit_windows", []), same, best)

        # -- tyre life vs the longest stint actually run -----------------------
        life = {x["compound"]: x for x in st["life"]}
        lens = []
        for _, p in plans.iterrows():
            for c, L in zip(p["seq"].split("-"), p["stint_lens"]):
                lens.append({"compound": c, "L": int(L)})
        lens = pd.DataFrame(lens)
        life_rows = []
        for c, g in lens.groupby("compound"):
            lf = life.get(c, {})
            blf = ({x["compound"]: x for x in bm["strategy"]["life"]}.get(c, {}) if bm else {})
            vlf = ({x["compound"]: x for x in v2m["strategy"]["life"]}.get(c, {}) if v2m else {})
            life_rows.append({"compound": c, "pred_life_at_push": lf.get("life_laps"),
                              "pred_life_uncapped": lf.get("life_model_uncapped"), "bound_by": lf.get("bound_by"),
                              "longer_than_race": lf.get("longer_than_race"),
                              "pred_knee_full_push": lf.get("knee_lap"),
                              "grip_budget_s": lf.get("grip_budget_s"), "deg_s_per_lap": lf.get("deg_s_per_lap"),
                              "baseline_life": blf.get("life_laps"), "v2_life": vlf.get("life_laps"),
                              "v2_bound_by": vlf.get("bound_by"), "v2_grip_budget_s": vlf.get("grip_budget_s"),
                              "obs_max": int(g["L"].max()), "obs_p90": float(g["L"].quantile(0.9)),
                              "obs_median": float(g["L"].median()), "n_stints": int(len(g)),
                              "ratio_to_max": (float(lf["life_laps"] / g["L"].max()) if lf.get("life_laps") else None),
                              "baseline_ratio_to_max": (float(blf["life_laps"] / g["L"].max()) if blf.get("life_laps") else None),
                              "v2_ratio_to_max": (float(vlf["life_laps"] / g["L"].max()) if vlf.get("life_laps") else None)})
        r["life"] = life_rows

        # -- oracle regret --------------------------------------------------------
        orc, rates = oracle_model(key, m)
        alloc = m["allocation"]["caps"]
        caps = {c: int(v) for c, v in (m.get("circuit_history") or {}).get("stint_cap", {}).items()} or None
        pit_loss = float(m["pit_loss_s"])
        sim = strat.simulate_model(orc, ev, pit_loss, push_grid=(1.0,), max_per_compound=alloc,
                                   max_stint=caps, shortlist=2000)
        cand = [{"label": "tool", "compounds": best["compounds"], "pit_laps": best["pit_laps"], "push": 1.0}]
        if tyre_opt.get("pit_laps"):
            cand.append({"label": "tyre_optimal", "compounds": tyre_opt["compounds"], "pit_laps": tyre_opt["pit_laps"], "push": 1.0})
        if bm:
            cand.append({"label": "baseline_tool", "compounds": bm["strategy"]["best_plan"]["compounds"],
                         "pit_laps": bm["strategy"]["best_plan"]["pit_laps"], "push": 1.0})
        if v2m:
            cand.append({"label": "v2_tool", "compounds": v2m["strategy"]["best_plan"]["compounds"],
                         "pit_laps": v2m["strategy"]["best_plan"]["pit_laps"], "push": 1.0})
        if modal_seq is not None:
            ms = plans[plans["seq"] == modal_seq]
            k = int(ms["n_stops"].iloc[0])
            med_pits = [int(np.median([p[i] for p in ms["pit_laps"]])) for i in range(k)]
            cand.append({"label": "field_modal", "compounds": modal_seq.split("-"), "pit_laps": med_pits, "push": 1.0})
        if winner is not None:
            cand.append({"label": "winner", "compounds": winner["seq"].split("-"), "pit_laps": list(winner["pit_laps"]), "push": 1.0})
        if not sim.table.empty:
            cand.append({"label": "oracle_opt", "compounds": sim.best["compounds"], "pit_laps": sim.best["pit_laps"], "push": 1.0})
        cand = [c for c in cand if all(x in orc.compounds for x in c["compounds"])]
        tbl, det = strat.evaluate_plans(orc, ev, cand, pit_loss, push=1.0, allocation=alloc, stint_cap=caps or {})
        costs = {d["label"]: float(d["times"].mean()) for d in det if d.get("valid")}
        ref = costs.get("oracle_opt", min(costs.values()))
        r["oracle"] = {"rates": {c: round(v["slope"], 4) for c, v in rates.items()},
                       "oracle_best": sim.best_label if not sim.table.empty else None,
                       "costs_s": costs, "regret_s": {k: round(v - ref, 2) for k, v in costs.items()},
                       "tool_beats_field_modal": (costs.get("tool", np.inf) < costs.get("field_modal", np.inf)),
                       "tool_beats_winner": (costs.get("tool", np.inf) < costs.get("winner", np.inf))}
        # -- the position-aware metric (additive; the block above is untouched) --
        pos, pm = position_block(key, ev, orc, cand, plans, pit_loss, alloc, caps, green_first)
        r["oracle"]["position_aware"] = pos
        # -- Haas: OCO and BEA against their own races ------------------------
        r["haas"] = haas_block(m, plans, race, pm)
        # -- per-car plans ----------------------------------------------------
        pd_rows = m.get("per_driver") or []
        if pd_rows:
            pdf = pd.DataFrame(pd_rows).merge(plans[["driver", "n_stops", "seq", "pit_laps", "first_sc"]].rename(
                columns={"n_stops": "actual_stops", "seq": "actual_seq", "pit_laps": "actual_pits"}), on="driver", how="inner")
            pdf["first_actual"] = [p[0] if p else np.nan for p in pdf["actual_pits"]]
            same_shape = float(pdf["same_shape_as_field"].mean()) if len(pdf) else None
            stops_ok = float((pdf["n_stops"] == pdf["actual_stops"]).mean()) if len(pdf) else None
            seq_ok = float((pdf["compounds"] == pdf["actual_seq"]).mean()) if len(pdf) else None
            g = pdf[(~pdf["first_sc"]) & pdf["first_stop"].notna() & pdf["first_actual"].notna()]
            from scipy.stats import spearmanr
            rho = float(spearmanr(g["first_stop"], g["first_actual"]).correlation) if len(g) >= 5 and g["first_stop"].std() > 0 else None
            r["per_driver"] = {"n": int(len(pdf)), "share_same_shape_as_field": same_shape, "stops_match_actual": stops_ok,
                               "seq_match_actual": seq_ok, "first_stop_spearman_vs_actual": rho,
                               "first_stop_spread": (float(pdf["first_stop"].max() - pdf["first_stop"].min()) if pdf["first_stop"].notna().any() else None),
                               "race_factor_range": [float(pdf["race_factor"].min()), float(pdf["race_factor"].max())],
                               "mode": getattr(cal, "percar_mode", None)}
        else:
            r["per_driver"] = {}
        # the three per-car rate-scale variants, ranked against the race's own stint rates
        r["per_driver_variants"] = accuracy_percar(key)
        # -- the cliff detector ------------------------------------------------
        r["cliff_detector"] = cliff_rows(m, key, ev)
        # -- counterfactual sanity ------------------------------------------
        cf = pd.read_parquet(DATA_PROCESSED / f"counterfactual_{key}.parquet")

        def _over30(p):
            return int((pd.read_parquet(p)["loss_s"] > 30).sum()) if p.exists() else None

        r["counterfactual"] = {"n": int(len(cf)), "n_over_30s": int((cf["loss_s"] > 30).sum()),
                               "n_with_sc_stops": int((cf["n_sc_stops"] > 0).sum()) if "n_sc_stops" in cf else None,
                               "n_over_30s_classified": int(((cf["loss_s"] > 30) & cf["classified"]).sum()) if "classified" in cf else None,
                               "median_loss_s": float(cf["loss_s"].median()),
                               "top": cf.head(3)[["driver", "compounds", "loss_s"]].round(1).to_dict("records"),
                               "baseline_n_over_30s": _over30(baseline_processed(f"counterfactual_{key}.parquet")),
                               "v2_n_over_30s": _over30(v2_processed(f"counterfactual_{key}.parquet"))}
        r["cliff"] = m["score"]["cliff"]
        r["ladder_check"] = m.get("ladder_check")
        r["pace_calibration"] = {k: v for k, v in (m.get("pace_calibration") or {}).items() if k in ("applied", "measured_net_s", "model_net_before", "model_net_after", "clipped")}
        r["gates_failed"] = [g["gate"] for g in m["gates"] if not g["pass"]]
        r["calibration"] = {k: v for k, v in (m.get("calibration") or {}).items()
                            if k in ("grip_budget_by_compound", "undercut_lambda", "plan_prior_tau_s",
                                     "grid_start_penalty_s", "dirty_air_s_per_lap", "manage_cost_s",
                                     "manage_wear_floor",
                                     # V3: the first-stop weight, the circuit's own dirty air, the
                                     # censored budget estimate and which per-car term was applied
                                     "first_stop_kappa_s", "dirty_air_used", "dirty_air_source",
                                     "grip_budget_detail", "percar_mode", "source")}
        r["stint_fe_baseline"] = {k: v for k, v in (m.get("stint_fe_baseline") or {}).items() if k != "table"}
        r["bayes_vs_stint_fe_pooled"] = m.get("bayes_vs_stint_fe_pooled")
        out[key] = r
        win = r["pit_windows"]
        rows.append({"event": key, "recommended": st["best"], "tyre_optimal": tyre_opt.get("label"),
                     "baseline": (bm["strategy"]["best"] if bm else None),
                     "v2": (v2m["strategy"]["best"] if v2m else None),
                     "winner": r["winner_seq"], "winner_pits": r["winner_pits"],
                     "field_modal": modal_seq, "stops_match_mode": r["stops_match_mode"], "stops_share": round(r["stops_share"], 2),
                     "seq_share": round(r["rec_seq_share"], 2), "seq_in_top5": r["seq_in_top5"],
                     "start_ok": r["start_matches_majority"],
                     "first_rec_minus_field": r["first_stop"]["rec_minus_field"],
                     "first_tyre_minus_field": r["first_stop"]["tyre_minus_field"],
                     "first_baseline_minus_field": r["first_stop"]["baseline_minus_field"],
                     "first_v2_minus_field": r["first_stop"]["v2_minus_field"],
                     "first_stop_s": r["first_stop"]["first_stop_s"],
                     "sc_set_first": r["first_stop"]["sc_set"],
                     "regret_tool": r["oracle"]["regret_s"].get("tool"), "regret_tyre_opt": r["oracle"]["regret_s"].get("tyre_optimal"),
                     "regret_baseline": r["oracle"]["regret_s"].get("baseline_tool"),
                     "regret_v2": r["oracle"]["regret_s"].get("v2_tool"),
                     "regret_field": r["oracle"]["regret_s"].get("field_modal"), "regret_winner": r["oracle"]["regret_s"].get("winner"),
                     "win1_inside": (win[0]["share_inside"] if win else None), "win1_err": (win[0]["median_abs_err"] if win else None),
                     "collapse_n": (r["cliff_detector"] or {}).get("n_collapse"),
                     "collapse_abs_err_laps": (r["cliff_detector"] or {}).get("mean_abs_error_laps"),
                     # V4: the position-aware regret and the places the first cycle costs
                     "rpos_tool": pos["R_pos"].get("tool"), "rpos_tyre_opt": pos["R_pos"].get("tyre_optimal"),
                     "rpos_field": pos["R_pos"].get("field_modal"),
                     "L_tool": pos["L"].get("tool"), "L_field": pos["L"].get("field_modal")})
        print(f"\n== {key}: tool {st['best']} | tyre-optimal {tyre_opt.get('label')} | V2 {v2m['strategy']['best'] if v2m else '-'} "
              f"| V1 {bm['strategy']['best'] if bm else '-'} "
              f"| winner {r['winner']} {r['winner_seq']} @ {r['winner_pits']} | modal {modal_seq}")
        print(f"   stops: rec {best['n_stops']} mode {mode_stops} ({r['stops_share']:.0%} of field); seq share {r['rec_seq_share']:.0%}; "
              f"in top5 {r['seq_in_top5']}; start {best['compounds'][0]} vs field {dict(start_counts)}")
        print(f"   first stop: field median (green) {fmed}, {sc_share_first:.0%} under SC; rec {r['first_stop']['recommended']} "
              f"tyre-opt {r['first_stop']['tyre_optimal']} V2 {r['first_stop']['v2_rec']} V1 {r['first_stop']['baseline_rec']}; "
              f"prior penalty paid {r['first_stop']['first_stop_s']} s (kappa {r['first_stop']['first_stop_kappa_s']}), "
              f"prior {(r['first_stop']['prior'] or {}).get('mode') if isinstance(r['first_stop']['prior'], dict) else None}")
        print(f"   oracle rates {r['oracle']['rates']}  regret (s): {r['oracle']['regret_s']}")
        print(f"   position-aware (V {pos['V_s']} s, sigma_rel {pos['sigma_rel_s']} s, "
              f"{pos['n_field_first_stops']} green field first stops, {pos['n_candidates']} candidates): "
              f"L {pos['L']}")
        print(f"     R_pos (s): {pos['R_pos']}; P_retain(tool) {pos['P_retain_tool']}; "
              f"best candidate {pos['best_candidate']} @ {pos['best_candidate_first_stop']}")
        for drv, h in r["haas"]["drivers"].items():
            print(f"   haas {drv}: plan {h['plan']} first {h['first_stop']} vs actual "
                  f"{h['actual_first_stop']} ({h['actual_seq']}, SC {h['actual_first_sc']}) "
                  f"-> green err {h['first_minus_actual_green']}; shape {h['same_shape_as_field']}; "
                  f"L {h['L']} (P_retain {h['P_retain']})")
        print(f"   windows: {win}")
        print(f"   life: {[(x['compound'], x['pred_life_at_push'], x['bound_by'], x['obs_p90'], x['obs_max']) for x in life_rows]}")
        print(f"   per-driver: {r['per_driver']}")
        print(f"   per-driver variants: { {k: v for k, v in (r['per_driver_variants'] or {}).items() if k != 'scales'} }")
        print(f"   cliff detector: n {(r['cliff_detector'] or {}).get('n')} kinds {(r['cliff_detector'] or {}).get('kinds')} "
              f"collapses {(r['cliff_detector'] or {}).get('n_collapse')} "
              f"predicted-vs-observed |err| {(r['cliff_detector'] or {}).get('mean_abs_error_laps')} laps "
              f"(over {(r['cliff_detector'] or {}).get('over')} / under {(r['cliff_detector'] or {}).get('under')})")
        print(f"   counterfactual >30s: {r['counterfactual']['n_over_30s']}/{r['counterfactual']['n']} "
              f"(V2 {r['counterfactual']['v2_n_over_30s']}, V1 {r['counterfactual']['baseline_n_over_30s']}); "
              f"SC-held drivers {r['counterfactual']['n_with_sc_stops']}")
    dump("strategy.json", out)
    t = pd.DataFrame(rows)
    t.to_csv(dump("strategy_table.json", []).with_suffix(".csv"), index=False)
    print("\n" + t.to_string(index=False))


if __name__ == "__main__":
    main()
