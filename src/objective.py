"""One objective, assembled once.

Every strategy decision in the repo - the offline search, the pit window, the
per-car plans, the counterfactual, the plan builder's hand-built plans, the
degradation crossover, the recalibration's sweeps - has to be scored on the
*same* cost, or the numbers a reader compares are not comparable and a
constant calibrated on one of them does not apply to the others.  V3 had the
objective's parameters spelled out at eight call sites; V4 Task 1 added the
race-state term to two of them and left the rest on V3, which is the state
`results_v4_task1.md` §5 records as "not converted".

`V4Objective` is that objective's parameters, in one place:

    tyre + pit lane + traffic + safety-car credit          (always, from the model)
  + race state on the first stop                           `race_state`, `rival_field`
  + undercut exposure on the *later* stops at `lambda`     `undercut_lambda`
  + plan-family prior at `tau`                             `plan_prior`, `plan_prior_tau_s`
  + grid penalty per step of start-compound hardness       `grid_penalty_s`

and nothing else.  The first-stop history prior (`first_stop_kappa_s`) is
structurally **zero** under V4: the race state times the first stop, and
history enters only as the plan-family prior, the stint caps and - through
`first_stop_prior`, which is still handed to the search - the *rivals'*
plausible stop laps (WP-A's rival field).  A V4 objective with a non-zero
kappa is a contradiction and is refused.

The four `*_kwargs` methods splat the parameters into the four scoring
functions in `src.strategy`, dropping any keyword the target does not accept,
so one objective works on this checkout and on the merged one (WP-A adds
`rival_field` to `simulate_model`; before it lands the keyword is dropped and
the search runs Task 1's symmetric pack).

    obj = V4Objective.for_event(ev, cal, plan_prior=pp, first_stop_tables=fs,
                                dirty_air=dirty, race_state=const)
    model, res, _ = strat.search_with_pace_calibration(model, ev, pit_loss, **obj.sim_kwargs())
    pw  = strat.pit_window_model(model, ev, res.best, pit_loss, **obj.window_kwargs(res))
    cf  = strat.counterfactual(model, ev, race, pit_loss, **obj.counterfactual_kwargs(res))
    tbl, det = strat.evaluate_plans(model, ev, plans, pit_loss, **obj.eval_kwargs(res))

`V4Objective.v3_for_event` builds the V3 objective (kappa on, no race state)
for the explicitly labelled baselines and ablations - `--no-race-state`,
`bench_ablation`'s variants - and `v3_kwargs()` refuses to hand V3 kwargs to
anything that is not labelled a V3 baseline.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, replace

import numpy as np

from src import racestate, strategy as strat
from src.config import DIRTY_AIR_S_PER_LAP, GRID_START_PENALTY_S, SC_RATE_PER_LAP, Event, get_event

V4_LABEL = ("V4 objective: race state ({mode} rival field), lambda on later stops, "
            "tau plan prior, no first-stop prior")
V4_LABEL_NO_RS = ("V4 objective without a race state: lambda on every stop, tau plan prior, "
                  "no first-stop prior")
V3_LABEL = ("V3 baseline objective: undercut exposure on every stop at lambda, tau plan prior, "
            "first-stop history prior at kappa")

# The rival field's family temperature when nothing has calibrated one: the
# same default WP-A's `RivalFieldConfig` carries (3.0 s), so a checkout without
# the dataclass and one with it price the same field.
FAMILY_TEMPER_S_DEFAULT = 3.0


# --------------------------------------------------------------------------
# Keyword plumbing: one objective, four call signatures, two checkouts
# --------------------------------------------------------------------------


def accepts(fn, name: str) -> bool:
    """Does `fn` take a keyword called `name`?  (Or `**kwargs`.)

    The same test `bench/common.accepts` makes, kept here so `src` does not
    import from `bench`."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):       # a builtin or a C function: assume it does
        return True
    if name in sig.parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _filtered(kw: dict, target, base) -> dict:
    """`kw` less every keyword `base` does not accept (and, when `target` is a
    wrapper such as `search_with_pace_calibration`, every keyword it rejects).

    `base` is the function that actually consumes the keywords, which is what
    makes this safe for wrappers that forward `**kw`: `per_driver_plans` accepts
    anything and hands it to `simulate_model`, so the signature that decides is
    `simulate_model`'s."""
    out = {}
    for k, v in kw.items():
        if not accepts(base, k):
            continue
        if target is not None and target is not base and not accepts(target, k):
            continue
        out[k] = v
    return out


def terms_by_group(res_or_packs, n_laps: int) -> dict:
    """`{group label: (n_laps + 1,) array}` - each plan group's race-state
    first-stop term, indexed by first-stop lap.

    Takes a `StrategyResult` or the `packs` dict of its `race_state`.  The
    labels are `strategy._group_label((start, second, n_stops))`, e.g.
    "2-stop M-H", which is the key `evaluate_plans` / `counterfactual` look a
    plan's own group up by."""
    rs = getattr(res_or_packs, "race_state", res_or_packs) or {}
    packs = (rs.get("packs") if isinstance(rs, dict) else None) or {}
    out = {}
    for label, pack in packs.items():
        term = racestate.term_by_lap(pack, int(n_laps))
        if term is not None:
            out[str(label)] = np.asarray(term, dtype=float)
    return out


def group_label(compounds, pit_laps=None, n_stops=None) -> str | None:
    """The race-state group label of a plan - `"2-stop M-H"` - or `None` for a
    plan with no stop or one stint.

    The same key `simulate_model` files each group's pack under, so a consumer
    that has only the plan (the desk, a committed plan, a benchmark row) can
    look its term up."""
    seq = [str(c).upper() for c in (compounds or [])]
    n = int(n_stops if n_stops is not None else len(pit_laps or []))
    if len(seq) < 2 or n < 1:
        return None
    return strat._group_label((seq[0], seq[1], n))


def terms_from_json(terms: dict | None) -> dict:
    """`{group label: list of floats}` (as the outlook JSON stores it) back to
    `{group label: array}` for `evaluate_plans` / `counterfactual`."""
    return {str(k): np.asarray(v, dtype=float) for k, v in (terms or {}).items() if v is not None and len(v)}


def race_state_block(res, const, pw=None, n_laps: int | None = None) -> dict:
    """The race-state record the pipeline, the weekend script and the outlook
    all write: the recommended plan group's pack equilibrium and, for every lap
    from six before the first stop to the stop itself, the five actions it
    compared and the one it chose.

    Moved here from `scripts/10_pipeline.py` so that every writer of a
    `race_state` block writes the same shape (WP-F/F3).  `n_laps` is accepted
    and unused: the curve is already indexed by the laps it was solved on."""
    rs = getattr(res, "race_state", None) or {}
    best = rs.get("best") or {}
    if not best or const is None:
        return {"enabled": const is not None, "constants": (const.as_dict() if const is not None else None)}
    laps = best["laps"]
    cost = np.asarray(best["cost_s"], dtype=float)
    tyre = np.asarray(best["tyre_s"], dtype=float)
    pos = np.asarray(best["position_s"], dtype=float)
    pit_laps = (res.best or {}).get("pit_laps") or []
    first = int(pit_laps[0]) if len(pit_laps) else None
    w_hi = None
    if pw is not None and hasattr(pw, "empty") and not pw.empty:
        g1 = pw[(pw["stop"] == 1) & pw["in_window"]]
        w_hi = int(g1["lap"].max()) if len(g1) else None
    decisions = []
    if first is not None:
        for lap in range(max(int(laps[0]), first - 6), first + 1):
            t = racestate.action_table(laps, cost, {"tyre_s": tyre - tyre.min(), "position_s": pos - pos.min(),
                                                    "places_ahead": np.asarray(best["places"])},
                                       now_lap=lap, window_hi=w_hi)
            decisions.append({"lap": lap, **t})
    return {"enabled": True, "constants": const.as_dict(), "group": rs.get("best_group"),
            "mode": rs.get("mode"), "rival_field": rs.get("rival_field"),
            "first_stop": first, "tyre_optimal_first_stop_in_group": best.get("tyre_best_lap"),
            "pack_first_stop_median": best.get("q_median"), "pack_first_stop_iqr": best.get("q_p25_p75"),
            "iterations": best.get("iterations"), "converged": best.get("converged"), "push": best.get("push"),
            "race_state_s": float((res.best or {}).get("race_state_s", 0.0)),
            "curve": {k: best.get(k) for k in ("laps", "tyre_s", "places", "position_s", "cost_s", "term_s", "q")},
            "groups": rs.get("groups"), "n_groups_solved": rs.get("n_groups_solved"),
            "decisions": decisions}


def measure_constants_excluding(keys, **kw):
    """`racestate.measure_constants` with a *set* of weekends held out.

    WP-A widens `exclude` to any iterable of keys; before it lands the keyword
    takes one string and a set would silently exclude nothing, so the exclusion
    is done on the donor list instead - which is the same measurement either
    way, because the constants are cached on the donor tuple.  The recalibration
    needs this: a sweep on donor `d` inside the block that holds `h` out must
    see neither race."""
    keys = {str(k) for k in (keys or ()) if k}
    fn = racestate.measure_constants
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None
    extra = {k: v for k, v in kw.items() if sig is None or accepts(fn, k)}
    ann = str(sig.parameters["exclude"].annotation) if (sig and "exclude" in sig.parameters) else ""
    if keys and any(w in ann for w in ("Iterable", "Collection", "Sequence", "set")):
        return fn(exclude=(keys if len(keys) > 1 else next(iter(keys))), **extra)
    donors = tuple(k for k in racestate.DONOR_EVENTS if k not in keys)
    return fn(donors=donors, **extra)


def rival_field_default(cal=None, **over):
    """WP-A's `RivalFieldConfig` at the calibrated family temperature, or `None`
    on a checkout that has no such dataclass yet (where `simulate_model`'s
    symmetric pack is the only field there is)."""
    cls = getattr(racestate, "RivalFieldConfig", None)
    if cls is None:
        return None
    kw = {"mode": str(getattr(cal, "rival_field_mode", "hetero") or "hetero"),
          "family_temper_s": float(getattr(cal, "family_temper_s", FAMILY_TEMPER_S_DEFAULT)
                                   or FAMILY_TEMPER_S_DEFAULT)}
    kw.update(over)
    names = set(getattr(cls, "__dataclass_fields__", {}))
    return cls(**{k: v for k, v in kw.items() if not names or k in names})


def _extrap_ln_sd(cal=None) -> float:
    """The tyre-life extrapolation width: the calibration's if it carries one,
    else WP-B's measured module constant, else 0 (which is V3 bit for bit)."""
    v = getattr(cal, "extrap_ln_sd", None)
    if v is None:
        from src import tyre as _tyre
        v = getattr(_tyre, "EXTRAP_LN_SD_MEASURED", 0.0)
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------
# The objective
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class V4Objective:
    """The parameters of the one objective, and how to hand them to each scorer.

    `race_state`   `racestate.RaceStateConstants` - the measured place value,
                   persistence, pit-cycle noise and pack gaps (leave-one-out).
                   `None` is the objective without a race state.
    `rival_field`  WP-A's `RivalFieldConfig`; `None` means "the default field"
                   (hetero once WP-A lands, Task 1's symmetric pack before).
    `undercut_lambda`  V3's undercut exposure, charged on the stops **after the
                   first** only: the race-state term prices the first stop's
                   exposure car by car.
    `first_stop_kappa_s`  0 in production; non-zero only on a V3 baseline.
    `first_stop_prior`  the circuit's green first-stop density tables.  Handed
                   to the search for the *rivals'* stop laps (WP-A), never
                   charged on our own lap unless kappa > 0 (V3).
    `extrap_ln_sd` WP-B's tyre-life extrapolation width, carried here so the
                   model builders can read it off the objective.
    """

    race_state: object | None = None
    rival_field: object | None = None
    undercut_lambda: float = 0.0
    plan_prior: dict | None = None
    plan_prior_tau_s: float = 0.0
    first_stop_kappa_s: float = 0.0
    traffic_s_per_lap: float = DIRTY_AIR_S_PER_LAP
    grid_penalty_s: float = GRID_START_PENALTY_S
    sc_rate: float = SC_RATE_PER_LAP
    extrap_ln_sd: float = 0.0
    first_stop_prior: dict | None = None
    label: str = ""
    version: str = "v4"
    # the race distance, so `window_kwargs(res)` can index the group term by lap
    n_race_laps: int = 0
    event_key: str = ""

    def __post_init__(self):
        if self.version == "v4" and float(self.first_stop_kappa_s or 0.0) != 0.0:
            raise ValueError("the V4 objective does not charge a first-stop history prior: "
                             f"first_stop_kappa_s={self.first_stop_kappa_s} (use V4Objective.v3_for_event "
                             "for an explicitly labelled V3 baseline)")
        if not self.label:
            object.__setattr__(self, "label", self._default_label())

    # -- builders ----------------------------------------------------------

    def _default_label(self) -> str:
        if self.version != "v4":
            return V3_LABEL
        if self.race_state is None:
            return V4_LABEL_NO_RS
        mode = getattr(self.rival_field, "mode", None) or ("hetero" if self.rival_field is not None else "default")
        return V4_LABEL.format(mode=mode)

    @classmethod
    def for_event(cls, event: Event | str, cal, *, plan_prior: dict | None = None,
                  first_stop_tables: dict | None = None, dirty_air: float | None = None,
                  race_state=None, rival_field=None, extrap_ln_sd: float | None = None) -> "V4Objective":
        """The production objective for a weekend: the calibration's weights,
        this circuit's dirty air, this weekend's plan prior and first-stop
        tables, and the race state the caller measured (leave-one-out).

        `rival_field=None` asks for the default field; pass
        `racestate.RivalFieldConfig(mode="symmetric")` for Task 1's pack."""
        ev = get_event(event) if isinstance(event, str) else event
        return cls(race_state=race_state,
                   rival_field=(rival_field if rival_field is not None
                                else (rival_field_default(cal) if race_state is not None else None)),
                   undercut_lambda=float(getattr(cal, "undercut_lambda", 0.0) or 0.0),
                   plan_prior=plan_prior or {},
                   plan_prior_tau_s=float(getattr(cal, "plan_prior_tau_s", 0.0) or 0.0),
                   first_stop_kappa_s=0.0,
                   traffic_s_per_lap=float(dirty_air if dirty_air is not None
                                           else getattr(cal, "dirty_air_for", lambda _c=None: DIRTY_AIR_S_PER_LAP)(ev.circuit)),
                   grid_penalty_s=float(getattr(cal, "grid_start_penalty_s", 0.0) or 0.0),
                   sc_rate=SC_RATE_PER_LAP,
                   extrap_ln_sd=(float(extrap_ln_sd) if extrap_ln_sd is not None else _extrap_ln_sd(cal)),
                   first_stop_prior=first_stop_tables, version="v4",
                   n_race_laps=int(ev.n_race_laps), event_key=str(ev.key))

    @classmethod
    def v3_for_event(cls, event: Event | str, cal, *, plan_prior: dict | None = None,
                     first_stop_tables: dict | None = None, dirty_air: float | None = None) -> "V4Objective":
        """The V3 objective - undercut exposure on every stop, the first-stop
        history prior at the calibrated kappa, no race state.

        For explicitly labelled baselines and ablations only (`--no-race-state`,
        `bench_ablation`'s variants, the V3 columns of a comparison)."""
        obj = cls.for_event(event, cal, plan_prior=plan_prior, first_stop_tables=first_stop_tables,
                            dirty_air=dirty_air, race_state=None, rival_field=None)
        return replace(obj, version="v3", label=V3_LABEL, race_state=None, rival_field=None,
                       first_stop_kappa_s=float(getattr(cal, "first_stop_kappa_s", 0.0) or 0.0),
                       extrap_ln_sd=0.0)

    def without_race_state(self) -> "V4Objective":
        """The same weights with the race state off (the `no_race_state` ablation
        of the V4 objective, which is *not* V3: kappa stays at zero)."""
        return replace(self, race_state=None, rival_field=None, label=V4_LABEL_NO_RS)

    # -- the four call sites -----------------------------------------------

    def sim_kwargs(self, fn=None) -> dict:
        """Keywords for `simulate_model` / `search_with_pace_calibration` /
        `per_driver_plans` / `simulate`.

        `first_stop_prior` goes through even at kappa = 0: WP-A's rival field
        reads it for the *rivals'* stop-lap prior, and `simulate_model` ignores
        it on our own lap while kappa is zero."""
        kw = {"undercut_lambda": float(self.undercut_lambda),
              "plan_prior": self.plan_prior or None,
              "plan_prior_tau_s": float(self.plan_prior_tau_s),
              "first_stop_prior": self.first_stop_prior,
              "first_stop_kappa_s": float(self.first_stop_kappa_s),
              "traffic_s_per_lap": float(self.traffic_s_per_lap),
              "grid_penalty_s": float(self.grid_penalty_s),
              "sc_rate": float(self.sc_rate)}
        if self.race_state is not None:
            kw["race_state"] = self.race_state
            if self.rival_field is not None:
                kw["rival_field"] = self.rival_field
        return _filtered(kw, fn, strat.simulate_model)

    def v3_kwargs(self, fn=None) -> dict:
        """`sim_kwargs` for a V3 baseline - refused on anything else."""
        if self.version != "v3":
            raise ValueError("v3_kwargs is for explicitly labelled V3 baselines and diagnostics only; "
                             f"this objective is {self.version!r} ({self.label})")
        return self.sim_kwargs(fn)

    def window_kwargs(self, res=None, n_laps: int | None = None, fn=None) -> dict:
        """Keywords for `pit_window_model`: the same weights, plus the
        recommended plan group's race-state term indexed by first-stop lap."""
        kw = {"undercut_lambda": float(self.undercut_lambda),
              "traffic_s_per_lap": float(self.traffic_s_per_lap),
              "sc_rate": float(self.sc_rate)}
        if float(self.first_stop_kappa_s) > 0:
            kw["first_stop_prior"] = self.first_stop_prior
            kw["first_stop_kappa_s"] = float(self.first_stop_kappa_s)
        term = self.best_term(res, n_laps)
        if term is not None:
            kw["race_state_term"] = term
        return _filtered(kw, fn, strat.pit_window_model)

    def eval_kwargs(self, res=None, n_laps: int | None = None, fn=None) -> dict:
        """Keywords for `evaluate_plans` / `deg_crossover`: the same weights,
        plus every plan group's race-state term, so a plan built by hand is
        priced exactly as the optimiser priced its own."""
        kw = {"undercut_lambda": float(self.undercut_lambda),
              "plan_prior": self.plan_prior or None,
              "plan_prior_tau_s": float(self.plan_prior_tau_s),
              "traffic_s_per_lap": float(self.traffic_s_per_lap),
              "grid_penalty_s": float(self.grid_penalty_s),
              "sc_rate": float(self.sc_rate)}
        if float(self.first_stop_kappa_s) > 0:
            kw["first_stop_prior"] = self.first_stop_prior
            kw["first_stop_kappa_s"] = float(self.first_stop_kappa_s)
        terms = self.terms(res, n_laps)
        if terms:
            kw["race_state_terms"] = terms
        return _filtered(kw, fn, strat.evaluate_plans)

    def counterfactual_kwargs(self, res=None, n_laps: int | None = None, fn=None) -> dict:
        """Keywords for `counterfactual`: what each driver's actual stop laps
        cost under the objective that produced the recommendation."""
        kw = {"undercut_lambda": float(self.undercut_lambda),
              "traffic_s_per_lap": float(self.traffic_s_per_lap)}
        if float(self.first_stop_kappa_s) > 0:
            kw["first_stop_prior"] = self.first_stop_prior
            kw["first_stop_kappa_s"] = float(self.first_stop_kappa_s)
        terms = self.terms(res, n_laps)
        if terms:
            kw["race_state_terms"] = terms
        return _filtered(kw, fn, strat.counterfactual)

    def model_kwargs(self, fn=None) -> dict:
        """The objective's tyre-model parameters (WP-B's `extrap_ln_sd`), dropped
        on a checkout whose `TyreModel` has no such field."""
        from src.tyre import TyreModel

        kw = {"extrap_ln_sd": float(self.extrap_ln_sd)}
        names = set(getattr(TyreModel, "__dataclass_fields__", {}))
        return {k: v for k, v in kw.items() if k in names} if names else {}

    # -- the race-state terms ----------------------------------------------

    def terms(self, res=None, n_laps: int | None = None) -> dict:
        """`{group label: term by lap}` for every group the search solved."""
        if res is None or self.race_state is None:
            return {}
        return terms_by_group(res, int(n_laps or self.n_race_laps or 0))

    def best_term(self, res=None, n_laps: int | None = None):
        """The recommended plan group's term, as `pit_window_model` wants it."""
        if res is None or self.race_state is None:
            return None
        rs = getattr(res, "race_state", None) or {}
        return racestate.term_by_lap(rs.get("best"), int(n_laps or self.n_race_laps or 0))

    # -- reporting ---------------------------------------------------------

    def as_dict(self) -> dict:
        """The objective as JSON: the weights, the label and the version."""
        rf = self.rival_field
        return {"label": self.label, "version": self.version,
                "undercut_lambda": float(self.undercut_lambda),
                "undercut_applies_to": ("stops after the first" if self.race_state is not None else "every stop"),
                "plan_prior_tau_s": float(self.plan_prior_tau_s),
                "first_stop_kappa_s": float(self.first_stop_kappa_s),
                "traffic_s_per_lap": float(self.traffic_s_per_lap),
                "grid_penalty_s": float(self.grid_penalty_s),
                "sc_rate": float(self.sc_rate),
                "extrap_ln_sd": float(self.extrap_ln_sd),
                # the plan-family counts in full, so a consumer of the JSON (the
                # plan builder) can charge `tau` exactly as the search did
                "plan_prior": (self.plan_prior or None),
                "race_state": (self.race_state.as_dict() if self.race_state is not None else None),
                "rival_field": (asdict(rf) if rf is not None and hasattr(rf, "__dataclass_fields__") else None),
                "first_stop_prior_in_objective": bool(self.first_stop_prior and float(self.first_stop_kappa_s) > 0),
                "first_stop_prior_for_rivals": bool(self.first_stop_prior is not None and self.race_state is not None),
                "event": self.event_key, "n_race_laps": int(self.n_race_laps)}

    def block(self, res=None, pw=None) -> dict:
        """The `race_state` block for a meta / outlook file."""
        return race_state_block(res, self.race_state, pw, self.n_race_laps)

    def terms_json(self, res=None) -> dict:
        """`race_state_terms` as JSON: group label -> list of floats by lap."""
        return {k: [round(float(x), 4) for x in v] for k, v in self.terms(res).items()}
