"""The live strategy engine: sealed tyre model in, decisions out, every lap.

Two engines share one interface (`tick(state) -> dict`):

* `RaceEngine` for races and sprints.  For every car it maintains a posterior
  over how fast *its* tyre is wearing (the sealed practice fit, re-weighted by
  the laps the car has actually run this stint), and from that posterior it
  answers the questions a pit wall asks: how far is this tyre from its cliff,
  what is the best plan from here, when does the window open and close, what
  does boxing *now* cost, can the car behind undercut us, is a safety-car stop
  worth taking.
* `PracticeEngine` for free practice.  It keeps a live long-run board — the
  same fuel-corrected, stint-fixed-effect degradation estimate the offline
  pipeline uses, updated as laps arrive — so the model's prior can be checked
  against the day's running before it is sealed.

**How the live update works.**  The sealed fit gives `n` posterior draws of
each compound's wear rate (practice regime, full push).  A race stint runs in
a different regime — managed, cooler, in traffic — which the offline work
measures as a *multiplier* on the practice rate with a wide distribution (the
regime factor).  Every draw is paired with a multiplier drawn from that prior;
each car's clean laps this stint then re-weight the draws by likelihood, with
the stint's level profiled out (a practice fit cannot know a race stint's base
pace, only its shape).  That is a particle approximation to the posterior, it
costs microseconds per lap, and everything downstream — wear, cliff
probability, plan costs, undercut gains — is evaluated on the weighted draws,
so the uncertainty shown on screen is the model's, not a heuristic.

**The plan search is the offline objective, vectorised.**  Every one-stop and
two-stop continuation is costed in a handful of array operations over the
cached fresh-stint cost tables, with the same undercut-exposure term the
offline search carries (at the weight the weekend was calibrated with), so
the live call and the pre-race plan cannot disagree about what a late stop
costs.  The circuit's first-stop prior is charged too, but **only on a car that
has not stopped yet**: it prices where this circuit's field has historically
taken its first stop, and a car on its second set is past the decision the
prior is about.  About 0.1-0.2 s per tick for the whole field.

**The cliff alarm is now a measurement.**  It used to fire on
`P(wear >= grip budget)` — a quantity the practice fit barely identifies, since
no practice long run reaches a cliff, so the alarm largely reported the budget
prior.  It now fires on `cliff.detect_stint_collapse` run on the car's own
fuel-corrected clean laps this stint: a break in the stint's own pace trend,
steep enough and costly enough to be a tyre falling off.  `p_past_cliff` and
`laps_to_cliff_*` are still computed and still shown — they are the model's
forecast — but the alarm is the observation.  The output key stays
`cliff_alarm` (the app reads it); its meaning is the collapse detection, and
`pace_collapse` carries the same flag under its own name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import cliff, firststop, percar, racestate
from src.calibration import get_calibration
from src.config import (
    DATA_PROCESSED,
    DEFAULT_RACE_REGIME_LN_SD,
    DEFAULT_RACE_REGIME_RATIO,
    GRIP_BUDGET_S,
    MAX_STINTS_PER_COMPOUND,
    OUT_LAP_PENALTY_S,
    PIT_WINDOW_MARGIN,
    SC_PIT_LOSS_FRACTION,
    TRAFFIC_LAPS_PER_STOP,
    VALID_COMPOUNDS,
    Event,
    get_event,
)
from src.compounds import hardness_rank
from src.fuel import get_prior
from src.live.state import LiveState
from src.tyre import TyreModel, grip_loss, load_profile

log = logging.getLogger("degless.live.engine")

N_DRAWS = 300
SIGMA_RACE_LAP_S = 0.5       # per-lap noise of a clean racing lap, s
DIRTY_SIGMA_MULT = 1.6       # a lap started 1-2 s behind another car is noisier, not useless
EVO_PRIOR_S_PER_LAP = -0.06  # track evolution over a race before the data can say (donor races: -0.08)
FIELD_TEMPER = 0.5           # other cars' laps count half: they manage their tyres differently
MIN_ESS = 40.0               # floor on the effective number of posterior draws (of N_DRAWS)
TRAFFIC_INTERVAL_S = 1.0     # a lap started this close behind another car is dirty (DRS range)
TRAFFIC_SOFT_INTERVAL_S = 2.0 # ...and this close is downweighted
SLOW_LAP_MARGIN_S = 3.0      # a lap this much slower than the stint median is not a racing lap
MIN_LAPS_FOR_UPDATE = 3
WINDOW_TOL_S = 1.0
VSC_PIT_LOSS_FRACTION = 0.55
UNDERCUT_RANGE_S = 6.0       # only cars this close can be undercut / undercut you
CLIFF_ALARM_P = 0.5
ALERT_COOLDOWN_LAPS = 3
PLAN_SHORTLIST = 48          # options priced draw by draw per car per lap (the rest on the expected cost)
COLLAPSE_MIN_LAPS = 6        # green laps a stint needs before the collapse detector will call it
BOX_NOW_TOL_S = 0.5          # box-now within this of the best plan is a live decision, not a hypothetical


def _temper_for_ess(ll: np.ndarray, min_ess: float) -> float:
    """Largest temperature in (0, 1] at which exp(t*ll) keeps ESS >= min_ess."""
    def ess(t):
        w = np.exp(t * (ll - ll.max()))
        w /= w.sum()
        return 1.0 / np.sum(w ** 2)
    if ess(1.0) >= min_ess:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if ess(mid) >= min_ess:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------
# The weekend model: what the offline pipeline sealed, or a prior if it didn't
# --------------------------------------------------------------------------


@dataclass
class WeekendModel:
    event: Event
    model: TyreModel                     # n draws, practice regime
    m_prior: np.ndarray                  # (n,) regime multiplier draws paired with the tyre draws
    pit_loss_s: float
    pit_loss_source: str
    allocation: dict                     # compound -> max stints
    stint_cap: dict = field(default_factory=dict)   # compound -> longest stint this circuit has supported
    history: dict = field(default_factory=dict)
    source: str = ""
    sealed_file: str = ""
    n_draws: int = N_DRAWS
    undercut_lambda: float = 0.0
    dirty_air_s_per_lap: float = 0.45
    calibration_source: str = ""
    first_stop_kappa_s: float = 0.0
    first_stop_table: dict | None = None             # compound -> (n+1,) nats by first-stop lap
    driver_dev_pooled: dict = field(default_factory=dict)   # driver -> {compound: (n,) draws}
    # V4: the race-state constants (every other 2026 race); None switches the
    # race state off and the engine runs the V3 objective
    race_state: object | None = None

    @classmethod
    def load(cls, event: Event | str, *, n_draws: int = N_DRAWS, seed: int = 0) -> "WeekendModel":
        ev = get_event(event) if isinstance(event, str) else event
        rng = np.random.default_rng(seed)
        cal = get_calibration(ev)
        post = DATA_PROCESSED / f"posterior_{ev.key}.npz"
        regime_ratio, regime_sd, sealed = DEFAULT_RACE_REGIME_RATIO, DEFAULT_RACE_REGIME_LN_SD, ""
        meta_p = DATA_PROCESSED / f"weekend_{ev.key}.json"
        if not meta_p.exists():
            meta_p = DATA_PROCESSED / f"meta_{ev.key}.json"
        meta = {}
        if meta_p.exists():
            import json
            meta = json.loads(meta_p.read_text())
            rg = meta.get("regime", {})
            regime_ratio = float(rg.get("ratio", regime_ratio))
            regime_sd = float(rg.get("ln_sd", regime_sd))
            sealed = meta.get("sealed_file", "")
        if post.exists():
            from src.model_bayes import BayesFit

            fit = BayesFit.load(post)
            total = fit.posterior["lin"].shape[0]
            idx = rng.choice(total, size=min(n_draws, total), replace=False)
            model = TyreModel.from_fit(fit, draws=idx, budget=cal.budgets,
                                       manage_floor=cal.manage_wear_floor, manage_cost_s=cal.manage_cost_s)
            # the weekend script's calibrated pace offsets, if it wrote them
            pc = (meta.get("pace_calibration") or {}).get("offsets_after")
            if pc and all(c in pc for c in model.compounds):
                model = model.copy_with(pace_offset={c: np.full(model.n_draws, float(pc[c])) for c in model.compounds})
            source = f"sealed fit {post.name} ({fit.prior_label}, {fit.n_laps} practice laps)"
        else:
            model = cls.prior_model(ev, n_draws, rng, calibration=cal)
            source = "prior only: no practice fit for this weekend yet"
        n = model.n_draws
        m_prior = np.exp(rng.normal(np.log(max(regime_ratio, 1e-3)), regime_sd, size=n))
        pit, pit_src = pit_loss_prior(ev, meta)
        hist = meta.get("circuit_history") or {}
        if hist.get("pit_loss_s") and not str(pit_src).startswith("measured at"):
            pit, pit_src = float(hist["pit_loss_s"]), f"this pit lane, {hist.get('years')}"
        alloc = (meta.get("allocation") or {}).get("caps") or \
            {c: MAX_STINTS_PER_COMPOUND for c in VALID_COMPOUNDS}
        caps = {k: int(v) for k, v in (hist.get("stint_cap") or {}).items()}
        if hist and post.exists():
            source += f"; circuit history {hist.get('years')}"
        # The circuit's first-stop density, from `circuit_history.first_stop_green`
        # (its 2023-25 races).  A V2 metadata file has no such key, so the table
        # is None and the term is simply absent rather than wrong.
        hist = backfill_circuit_history(ev, hist, meta_p)
        fs_table = firststop.first_stop_penalty_table(hist.get("first_stop_green"),
                                                      ev.n_race_laps, model.compounds)
        # Team-pooled practice deviation, computed once: a car with no long run on
        # a compound inherits its team-mate's behaviour rather than the field's.
        # The team map comes from this weekend's practice lap table, which is the
        # only place the engine can get it before the DriverList arrives.
        pooled_dev = {}
        prac = DATA_PROCESSED / f"laps_{ev.key}_practice.parquet"
        if model.driver_dev and prac.exists():
            try:
                lt = pd.read_parquet(prac, columns=["driver", "team"])
                teams = lt.drop_duplicates("driver").set_index("driver")["team"].to_dict()
                pooled_dev = percar.team_pooled_dev(model.driver_dev, teams)
                if pooled_dev:
                    source += f"; per-car deviation pooled over {len(set(teams.values()))} teams"
            except Exception:
                log.exception("team-pooled practice deviation unavailable for %s", ev.key)
        # The race state: measured on every 2026 race but this one, so a replay
        # of a scored race never learns from itself.
        try:
            rs_const = racestate.measure_constants(exclude=ev.key)
        except Exception:
            log.exception("race-state constants unavailable for %s; the V3 objective runs", ev.key)
            rs_const = None
        return cls(event=ev, model=model, m_prior=m_prior, pit_loss_s=pit,
                   pit_loss_source=pit_src, allocation=alloc, stint_cap=caps, history=hist,
                   source=source, sealed_file=sealed, n_draws=n,
                   undercut_lambda=float(cal.undercut_lambda),
                   dirty_air_s_per_lap=float(cal.dirty_air_for(ev.circuit)),
                   calibration_source=cal.source,
                   first_stop_kappa_s=float(cal.first_stop_kappa_s), first_stop_table=fs_table,
                   driver_dev_pooled=pooled_dev, race_state=rs_const)

    @staticmethod
    def prior_model(ev: Event, n: int, rng: np.random.Generator, calibration=None) -> TyreModel:
        """A tyre model from the compound ladder alone, for a weekend with no fit.

        Wide on purpose: the MEDIUM rate is LogNormal around 0.10 s/lap with a
        factor-2 spread, which spans every 2026 weekend measured so far
        (0.07-0.30 s/lap at full push).
        """
        from src.config import COMPOUND_DEG_RATIO, COMPOUND_PACE_STEP_FRAC

        cal = calibration if calibration is not None else get_calibration(ev)
        comps = list(VALID_COMPOUNDS)
        rank = dict(zip(comps, hardness_rank(comps)))
        med = np.exp(rng.normal(np.log(0.10), 0.5, size=n))
        ratio = np.exp(rng.normal(np.log(COMPOUND_DEG_RATIO), 0.25, size=n))
        step = COMPOUND_PACE_STEP_FRAC * ev.t_lap_ref_s
        budgets = cal.budgets
        wear, pace = {}, {}
        for c in comps:
            r = rank[c] - rank["MEDIUM"]
            rate = med * ratio ** (-r)
            wear[c] = rate / budgets[c]
            pace[c] = np.full(n, step * rank[c]) + rng.normal(0, 0.05, size=n)
        return TyreModel(compounds=comps, wear_rate=wear, pace_offset=pace,
                         budget=float(np.mean(list(budgets.values()))), budgets=budgets, n_draws=n,
                         source="compound-ladder prior", manage_floor=cal.manage_wear_floor,
                         manage_cost_s=cal.manage_cost_s)


HISTORY_ONLY_KEYS = ("first_stop_green", "dirty_air")


def backfill_circuit_history(ev: Event, hist: dict | None, meta_path) -> dict:
    """`circuit_history`, with the two V3 history blocks back-filled if missing.

    `weekend_<key>.json` is preferred over `meta_<key>.json` everywhere because
    it is the *pre-race* state, but a weekend file written by an older version
    has no `first_stop_green` and no `dirty_air`, and on a circuit whose history
    is already on disk that would silently switch the first-stop prior off —
    a failure nobody would see, since the term's absence looks exactly like a
    circuit with no history.  So those two keys, and only those two, are taken
    from the sibling file when the chosen one predates them.

    Safe by construction: both are measured on the circuit's 2023-25 races and
    contain nothing about the current weekend, which is why they can come from
    the retrospective file without breaking the firewall.  Everything else in
    the block — the thermal term, the season pool, the rate prior the fit used —
    stays as the chosen file wrote it.
    """
    out = dict(hist or {})
    if all(out.get(k) for k in HISTORY_ONLY_KEYS):
        return out
    name = getattr(meta_path, "name", "")
    other = DATA_PROCESSED / (f"meta_{ev.key}.json" if name.startswith("weekend_")
                              else f"weekend_{ev.key}.json")
    if not other.exists():
        return out
    try:
        import json

        och = json.loads(other.read_text()).get("circuit_history") or {}
    except Exception:
        log.exception("could not read the circuit history from %s", other.name)
        return out
    for k in HISTORY_ONLY_KEYS:
        if not out.get(k) and och.get(k):
            out[k] = och[k]
    return out


def pit_loss_prior(ev: Event, meta: dict | None = None) -> tuple:
    """Pit-lane time loss for this circuit: measured here if we have it,
    else the median of every donor race measured so far."""
    meta = meta or {}
    if meta.get("pit_loss_s") and np.isfinite(meta["pit_loss_s"]) and meta.get("pit_stops_measured", 0) >= 3:
        return float(meta["pit_loss_s"]), f"measured at {ev.key} ({meta['pit_stops_measured']} stops)"
    vals, names = [], []
    for p in sorted(DATA_PROCESSED.glob("meta_*.json")):
        import json
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        v = m.get("pit_loss_s")
        if v and np.isfinite(v) and 10 < v < 40 and m.get("event") != ev.key:
            vals.append(float(v))
            names.append(m.get("event"))
    if vals:
        return float(np.median(vals)), f"median of {len(vals)} donor races ({', '.join(names)})"
    return 22.0, "default 22 s (no donor race measured)"


# --------------------------------------------------------------------------
# Per-driver tyre posterior
# --------------------------------------------------------------------------


@dataclass
class DriverTyre:
    number: str
    code: str = ""                         # the TLA, which is how the fit names drivers
    compound: str | None = None
    stint_index: int | None = None
    stint_first_lap: int | None = None
    ages: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))       # fuel-corrected clean lap times
    laps_used: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_clean: int = 0
    weights: np.ndarray | None = None      # over draws
    m_mean: float = float("nan")
    m_lo: float = float("nan")
    m_hi: float = float("nan")
    level_s: float = float("nan")
    wear: np.ndarray | None = None         # (n,) wear now
    p_past_cliff: float = 0.0
    laps_to_cliff: tuple = (float("nan"),) * 3
    proj: list = field(default_factory=list)   # next laps' expected pace loss vs now
    deg_now_s_per_lap: float = float("nan")
    collapse: dict = field(default_factory=dict)   # cliff.detect_stint_collapse on this stint
    pace_collapse: bool = False


class RaceEngine:
    """Decisions for a race, recomputed every tick from the live state.

    **V4: the race state makes the pit call.**  Every tick runs in two passes.
    The first prices every car's options on its own tyre and the pit lane,
    exactly as V3 did.  The second re-prices each option's *next* stop against
    the car's four nearest rivals in virtual race position - their gaps, sets,
    tyre ages, stops made, pit status and the plan the engine gave them on the
    previous lap - at the measured value of a place (`src.racestate`), and
    swaps V3's generic traffic charge for the traffic the timing screen shows at
    the rejoin point over the next four laps.  The undercut exposure of the
    current set and the first-stop history prior come out of the next stop,
    because the race state prices what they approximated.  Each plan carries the
    explicit comparison of PIT NOW, STAY OUT 1/2/3 LAPS and PIT AT THE EDGE OF
    THE WINDOW (`plan["race_state"]`).  With `race_state=False`, or a weekend
    model without the constants, the engine is V3's, number for number.
    """

    def __init__(self, wm: WeekendModel, *, seed: int = 0, race_state: bool = True):
        self.wm = wm
        self.ev = wm.event
        self.n_laps = wm.event.n_race_laps
        self.model = wm.model
        self.n = wm.model.n_draws
        self.m_prior = wm.m_prior
        self.rng = np.random.default_rng(seed)
        self.load = load_profile(self.ev)
        self.fp = get_prior(self.ev, "2026")
        self.tyres: dict[str, DriverTyre] = {}
        self.field_weights = np.ones(self.n) / self.n
        self.pit_loss_s = wm.pit_loss_s
        self.pit_loss_source = wm.pit_loss_source
        self.pit_loss_live: list = []
        self.alerts: list = []
        self._alert_last: dict = {}
        self._plans: dict = {}
        self._last_plan_lap: dict = {}
        self._cost_cache: dict = {}
        self.tick_no = 0
        self.lam = float(getattr(wm, "undercut_lambda", 0.0) or 0.0)
        self.dirty_air = float(getattr(wm, "dirty_air_s_per_lap", 0.45) or 0.45)
        self.kappa = float(getattr(wm, "first_stop_kappa_s", 0.0) or 0.0)
        self.first_stop_table = getattr(wm, "first_stop_table", None)
        self.pooled_dev = dict(getattr(wm, "driver_dev_pooled", {}) or {})
        self._rate_cache: dict = {}
        self.race_state = getattr(wm, "race_state", None) if race_state else None
        self._ctx: dict = {}            # num -> the car's last plan context (phase 1)
        self._rs_curves: dict = {}      # num -> (laps, cost, stay) from the previous tick's race-state plan

    # -- helpers ------------------------------------------------------------

    def _rates(self, code: str) -> dict:
        """`{compound: (n,) wear rate}` for one car: the field model with this
        driver's team-pooled practice deviation folded in.

        The deviation enters the car's *own* quantities - its likelihood, its
        wear, its cliff, the cost of continuing on the set it is on - while the
        fresh-stint cost tables in `_plan` stay field-level.  That is an
        approximation, and a deliberate one: those tables are
        (compounds x laps x laps) over every draw, built once per race, and
        rebuilding them per car per lap was 95% of a 1.1 s tick for a second-order
        effect on a tyre the car has not fitted yet.  What the per-car term
        changes is the decision about the tyre on the car now, which is the
        decision the engine is asked for.
        """
        if not code or not self.pooled_dev.get(code):
            return self.model.wear_rate
        if code not in self._rate_cache:
            self._rate_cache[code] = self.model.for_driver(
                code, dev_override=self.pooled_dev.get(code), factor_shrink=None).wear_rate
        return self._rate_cache[code]

    def _first_stop_pen(self, compound: str | None, laps, n_stops=None):
        """`kappa * neglogp[lap]`, or 0 - the circuit's first-stop prior.

        `n_stops` is the *total* stop count of the option being priced, because
        the density is conditioned on the plan's family: a car pricing a one-stop
        continuation must not be charged the circuit's two-stop first stops.
        """
        if not self.first_stop_table or self.kappa <= 0 or not compound:
            return 0.0
        from src.strategy import first_stop_penalty

        return first_stop_penalty(compound, laps, self.first_stop_table, self.kappa, n_stops=n_stops)

    def _total_laps(self, state: LiveState) -> int:
        t = state.lap_count.get("total")
        return int(t) if t else self.n_laps

    def _load_vec(self, laps: np.ndarray) -> np.ndarray:
        idx = np.clip(np.asarray(laps, dtype=int), 1, self.n_laps) - 1
        return self.load[idx]

    def _fuel_corrected(self, lap_time: np.ndarray, lap_number: np.ndarray, total: int) -> np.ndarray:
        # Race fuel: the car gets lighter as laps pass; add back what has burned.
        return lap_time + self.fp.s_per_lap * (total - lap_number)

    # -- tyre posterior per driver ------------------------------------------

    def _race_evolution(self, laps: pd.DataFrame, total: int) -> np.ndarray:
        """Track evolution per race lap (s, centred), from the cross-section.

        All cars run the same lap at the same moment on tyres of different
        ages, so a race-lap fixed effect and per-compound age slopes are
        separately identified — the estimator `src.regime` uses offline.  The
        lap effect absorbs fuel burn too, so the known fuel term is added back
        and what remains is evolution.  Early in the race, when few laps exist,
        the estimate is shrunk toward a linear prior.
        """
        n = int(total)
        prior = EVO_PRIOR_S_PER_LAP * (np.arange(1, n + 1) - 1.0)
        prior -= prior.mean()
        d = laps[laps["is_accurate"] & (laps["track_status"] == "1") & laps["compound"].isin(self.model.compounds)
                 & laps["lap_time_s"].notna() & laps["tyre_life"].notna()]
        n_lap = d["lap_number"].nunique() if len(d) else 0
        if len(d) < 60 or n_lap < 8:
            return prior
        drv = pd.get_dummies(d["driver_number"], drop_first=True).astype(float)
        lapc = pd.Categorical(d["lap_number"].astype(int))
        lap = pd.get_dummies(lapc, drop_first=True).astype(float)
        comp = pd.get_dummies(d["compound"]).astype(float)
        age = d["tyre_life"].to_numpy(dtype=float)
        ageX = np.column_stack([comp[c].to_numpy() * age for c in comp.columns])
        X = np.column_stack([np.ones(len(d)), ageX, comp.to_numpy()[:, 1:], drv.to_numpy(), lap.to_numpy()])
        y = d["lap_time_s"].to_numpy(dtype=float)
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            return prior
        fe = np.concatenate([[0.0], beta[len(beta) - lap.shape[1]:]])
        levels = np.asarray(lapc.categories, dtype=float)
        evo = fe + self.fp.s_per_lap * (n - levels)
        evo -= evo.mean()
        coef = np.polyfit(levels, evo, 2 if n_lap >= 15 else 1)
        est = np.polyval(coef, np.arange(1, n + 1, dtype=float))
        est -= est.mean()
        w = n_lap / (n_lap + 15.0)
        return w * est + (1 - w) * prior

    def _update_tyres(self, state: LiveState, laps: pd.DataFrame, total: int) -> None:
        rate_draws = {c: self.model.wear_rate[c] for c in self.model.compounds}
        self.evo = self._race_evolution(laps, total)
        ll_by_driver: dict = {}
        for num, tr in state.tracks.items():
            st = tr.current_stint
            dt = self.tyres.get(num)
            if dt is None:
                dt = DriverTyre(number=num)
                self.tyres[num] = dt
            dt.code = state.driver_label(num)
            if st is None or st.get("compound") not in rate_draws:
                dt.compound = (st or {}).get("compound")
                dt.weights = None
                continue
            if dt.stint_index != st["index"]:
                dt.stint_index = st["index"]
                dt.compound = st["compound"]
                dt.stint_first_lap = st.get("first_lap")
                dt.weights = None
                dt.collapse, dt.pace_collapse = {}, False
            g = laps[(laps["driver_number"] == num) & (laps["is_complete"])]
            g = g[g["lap_number"] >= (dt.stint_first_lap or 1)] if dt.stint_first_lap else g.iloc[0:0]
            c = dt.compound
            rate = self._rates(dt.code)[c]            # (n,) this car's own wear rate
            first = int(dt.stint_first_lap or 1)
            cur_lap = int(tr.current["lap_number"]) if tr.current else first
            lap_axis = np.arange(first, cur_lap + 1)
            cum_load = np.cumsum(self._load_vec(lap_axis))          # (L,)
            # --- likelihood from racing laps -------------------------------
            clean = g[g["is_accurate"] & (g["track_status"] == "1") & ~g["pit_in"] & ~g["pit_out"]]
            if len(clean) >= 3:
                med = clean["lap_time_s"].median()
                clean = clean[clean["lap_time_s"] < med + SLOW_LAP_MARGIN_S]
            iv = clean["interval_s"].to_numpy(dtype=float)
            pos = clean["position"].to_numpy(dtype=float)
            dirty = np.isfinite(iv) & (iv < TRAFFIC_INTERVAL_S) & (pos > 1)
            soft = np.isfinite(iv) & (iv < TRAFFIC_SOFT_INTERVAL_S) & (pos > 1) & ~dirty
            sig = np.where(soft, SIGMA_RACE_LAP_S * DIRTY_SIGMA_MULT, SIGMA_RACE_LAP_S)[~dirty]
            clean = clean[~dirty]
            dt.n_clean = len(clean)
            ll = None
            if len(clean) >= MIN_LAPS_FOR_UPDATE:
                ln = clean["lap_number"].to_numpy(dtype=float)
                y = self._fuel_corrected(clean["lap_time_s"].to_numpy(dtype=float), ln, total)
                y = y - self.evo[np.clip(ln.astype(int), 1, len(self.evo)) - 1]
                k = np.clip(ln.astype(int) - first, 0, len(cum_load) - 1)
                w_mid = (cum_load[k][None, :] - 0.5 * self._load_vec(ln)[None, :]) \
                    * (rate * self.m_prior)[:, None]
                D = grip_loss(w_mid, budget=self.model.budget_of(c))            # (n, L)
                resid = y[None, :] - D
                wt = 1.0 / sig ** 2
                level = (resid * wt).sum(1, keepdims=True) / wt.sum()
                ll = -(((resid - level) ** 2) * wt).sum(1) / 2.0
                dt._ll = ll
                dt._level = level[:, 0]
                dt.ages = clean["tyre_life"].to_numpy(dtype=float)
                dt.y = y
                dt.laps_used = ln
                ll_by_driver[num] = ll
            else:
                dt._ll = None
            # -- has this tyre fallen off? ---------------------------------
            # On the car's own green racing laps this stint, uncorrected: the
            # detector adds the fuel burn back itself, and the track-evolution
            # term is deliberately left in.  Evolution makes a stint look
            # *flatter* than it is, so leaving it raises the bar a collapse has
            # to clear — the right direction for an alarm that interrupts a pit
            # wall.  Safe to call every lap: a few OLS fits on <= 40 rows.
            if len(clean) >= COLLAPSE_MIN_LAPS:
                sub = clean[["lap_number", "lap_time_s"]].assign(
                    tyre_age=clean["tyre_life"].to_numpy(dtype=float))
                try:
                    dt.collapse = cliff.detect_stint_collapse(
                        sub, fuel_s_per_lap=self.fp.s_per_lap, evo=None, min_laps=COLLAPSE_MIN_LAPS)
                except Exception:
                    log.exception("collapse detector failed for %s", num)
                    dt.collapse = {}
                dt.pace_collapse = bool(dt.collapse.get("collapse"))
        # Field posterior: every car's laps, tempered.  Driver posterior: own
        # laps at full weight plus the rest of the field tempered.
        ll_field = np.zeros(self.n)
        for ll in ll_by_driver.values():
            ll_field += FIELD_TEMPER * ll
        self.field_temperature = _temper_for_ess(ll_field, MIN_ESS)
        ll_field = ll_field * self.field_temperature
        fw = np.exp(ll_field - ll_field.max())
        self.field_weights = fw / fw.sum()
        for num, tr in state.tracks.items():
            dt = self.tyres.get(num)
            if dt is None or dt.compound not in rate_draws:
                continue
            own = getattr(dt, "_ll", None)
            tot = ll_field.copy()
            if own is not None:
                tot += (1.0 - FIELD_TEMPER) * own
            tot = tot * _temper_for_ess(tot, MIN_ESS)
            w = np.exp(tot - tot.max())
            w /= w.sum()
            dt.weights = w
            if own is not None:
                dt.level_s = float((dt._level * w).sum())
            # --- wear now, cliff, projection -------------------------------
            c = dt.compound
            rate = self._rates(dt.code)[c]
            b = self.model.budget_of(c)
            first = int(dt.stint_first_lap or 1)
            cur_lap = int(tr.current["lap_number"]) if tr.current else first
            cum_load = np.cumsum(self._load_vec(np.arange(first, cur_lap + 1)))
            m = self.m_prior
            wear_now = cum_load[-1] * rate * m                                   # (n,)
            dt.wear = wear_now
            dt.m_mean = float((m * w).sum())
            order = np.argsort(m)
            cw = np.cumsum(w[order])
            dt.m_lo = float(m[order][np.searchsorted(cw, 0.05)])
            dt.m_hi = float(m[order][min(np.searchsorted(cw, 0.95), self.n - 1)])
            dt.p_past_cliff = float(((wear_now >= 1.0) * w).sum())
            per_lap = rate * m * self._load_vec(np.array([cur_lap]))[0]
            ltc = np.where(per_lap > 0, (1.0 - wear_now) / np.maximum(per_lap, 1e-9), np.inf)
            ltc = np.clip(ltc, 0, 99)
            o = np.argsort(ltc)
            cw = np.cumsum(w[o])
            dt.laps_to_cliff = tuple(float(ltc[o][min(np.searchsorted(cw, q), self.n - 1)])
                                     for q in (0.1, 0.5, 0.9))
            loss_now = grip_loss(wear_now, budget=b)
            dt.proj = [float(((grip_loss(wear_now + per_lap * j, budget=b) - loss_now) * w).sum())
                       for j in range(1, 6)]
            dt.deg_now_s_per_lap = float(((grip_loss(wear_now + per_lap, budget=b) - loss_now) * w).sum())

    def _resample(self, weights: np.ndarray | None) -> np.ndarray:
        """Draw indices proportional to weights (systematic resampling)."""
        if weights is None:
            return np.arange(self.n)
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        return np.searchsorted(np.cumsum(weights), pos).clip(0, self.n - 1)

    # -- costs --------------------------------------------------------------

    def _future_cost_tables(self, total: int) -> tuple:
        """cost[c][d, s, L] for fresh stints on *every* draw (draw d paired with
        its regime multiplier), plus the undercut-exposure tables.

        Built once per race: the tables depend only on the sealed model, the
        regime prior and the race distance, none of which change between
        laps.  Each car's own posterior enters as a *weight vector* over the
        draws - its weighted-mean table prices every option in one gather,
        and only the shortlist is priced draw by draw.  A previous version
        rebuilt the table per car per lap on that car's resampled draws, which
        was 95% of a 1.1 s tick.
        """
        key = ("tables", int(total))
        if self._cost_cache.get("key") == key:
            return self._cost_cache["tables"], self._cost_cache["expo"]
        scaled = self.model.copy_with(
            wear_rate={c: self.model.wear_rate[c] * self.m_prior for c in self.model.compounds},
            driver_dev={})
        ev = self.ev
        if total != ev.n_race_laps:
            from dataclasses import replace
            ev = replace(ev, n_race_laps=int(total))
        tables = {c: v.astype(np.float32) for c, v in
                  scaled.cost_table(ev, int(total), 1.0, warmup_s=OUT_LAP_PENALTY_S).items()}
        expo = None
        if self.lam > 0:
            from src.strategy import undercut_exposure_tables
            expo = undercut_exposure_tables(scaled, ev, 1.0, int(total))
        self._cost_cache = {"key": key, "tables": tables, "expo": expo}
        return tables, expo

    def _continue_cost(self, dt: DriverTyre, idx: np.ndarray, cur_lap: int, n_more: int,
                       total: int) -> np.ndarray:
        """(len(idx), n_more+1): cumulative cost of running this tyre k more laps.

        On this car's own rate (team-pooled practice deviation folded in), the
        same rate its wear and cliff were computed from."""
        c = dt.compound
        rate = (self._rates(dt.code)[c] * self.m_prior)[idx]
        pace = self.model.pace_offset[c][idx]
        w0 = dt.wear[idx] if dt.wear is not None else np.zeros(len(idx))
        laps = np.arange(cur_lap, cur_lap + max(n_more, 0)) + 1
        loads = self._load_vec(laps) if n_more > 0 else np.zeros(0)
        inc = rate[:, None] * loads[None, :]
        w_end = w0[:, None] + np.cumsum(inc, axis=1)
        loss = grip_loss(w_end - 0.5 * inc, budget=self.model.budget_of(c)) + pace[:, None]
        out = np.zeros((len(idx), n_more + 1))
        out[:, 1:] = np.cumsum(loss, axis=1)
        return out

    # -- plans --------------------------------------------------------------

    def _plan(self, state: LiveState, num: str, dt: DriverTyre, tr, total: int,
              sc_active: bool) -> dict | None:
        """One car's plan on its own cost surface alone (V3): no race state."""
        ctx = self._plan_prepare(state, num, dt, tr, total)
        return self._plan_finish(state, num, dt, tr, total, ctx) if ctx else None

    def _plan_prepare(self, state: LiveState, num: str, dt: DriverTyre, tr, total: int) -> dict | None:
        """Phase 1 of a car's plan: every option on its expected cost.

        Returned as a context rather than a decision so that the race-state term
        (`_race_state_extra`), which needs *every* car's continuation and
        fresh-set costs, can re-price the next stop of each option before the
        choice is made.  The V3 objective's next-stop terms (the undercut
        exposure of the current set and the first-stop prior) are carried per
        option in `xv3`, and the next stop's traffic charge in `tv3`, so the
        race state can take them out and put its own in."""
        cur = tr.current
        if cur is None or dt.compound not in self.model.compounds:
            return None
        cur_lap = int(cur["lap_number"]) - 1          # laps completed
        R = total - cur_lap
        if R <= 0:
            return None
        n = self.n
        w = dt.weights if dt.weights is not None else self.field_weights          # (n,) this car's posterior
        tables, expo = self._future_cost_tables(total)
        margin = PIT_WINDOW_MARGIN
        used = {}
        for s in tr.stints:
            if s.get("compound") and s.get("first_lap") is not None:
                used[s["compound"]] = used.get(s["compound"], 0) + 1
        compounds_used = set(used)
        avail = [c for c in self.model.compounds
                 if used.get(c, 0) < int(self.wm.allocation.get(c, MAX_STINTS_PER_COMPOUND))]
        if not avail:
            avail = list(self.model.compounds)
        cont = self._continue_cost(dt, np.arange(n), cur_lap, R, total)      # (n, R+1) on every draw
        cont_m = w @ cont                                                     # (R+1,) this car's expectation
        mean_t = {c: np.tensordot(w, tables[c], axes=1) for c in avail}      # (total+1, total+1) per compound
        pit_now_factor = (SC_PIT_LOSS_FRACTION if state.track_status == "4" else
                          VSC_PIT_LOSS_FRACTION if state.track_status in ("6", "7") else 1.0)
        pit = self.pit_loss_s
        dens = self._density(total)
        traffic = TRAFFIC_LAPS_PER_STOP * self.dirty_air
        caps = self.wm.stint_cap
        stint_len_now = cur_lap - int(dt.stint_first_lap or 1) + 1      # laps run on this set so far
        BIG = 10 ** 6
        cap_cur = caps.get(dt.compound, BIG)
        over_cap = stint_len_now > cap_cur + 1
        now_lap = cur_lap + 1
        cur_c = dt.compound

        def cont_ok_vec(k):   # continuing this set k more laps
            k = np.asarray(k)
            return (k <= 2) if over_cap else ((stint_len_now + k) <= cap_cur + 1)

        def cap_ok_vec(c, L):
            return np.asarray(L) <= caps.get(c, BIG)

        def expo_cont(c_new, k):
            """Exposure added by staying out k more laps before fitting `c_new`."""
            if expo is None:
                return np.zeros(np.shape(k))
            e = expo[(cur_c, c_new)]
            a0 = min(stint_len_now, len(e) - 1)
            a1 = np.clip(stint_len_now + np.asarray(k), 0, len(e) - 1)
            return self.lam * (e[a1] - e[a0]) * dens[np.clip(cur_lap + np.asarray(k), 1, total) - 1]

        # The circuit's first-stop prior, and only while the stop being priced
        # still *is* the first one: a car on its opening set (stint index 0, one
        # stint on record) is taking the decision the prior is about.  Later
        # stints get nothing - the term would be pricing history against a
        # decision the race has already overwritten.
        first_stint = bool(len(tr.stints) <= 1 or (dt.stint_index or 0) == 0)
        stops_taken = max(len([s for s in tr.stints if s.get("first_lap") is not None]) - 1, 0)

        def first_stop_pen(p_laps, more_stops: int):
            """Nats-weighted seconds on a first stop at lap(s) `p_laps`.

            `more_stops` is how many stops the option still makes (1 for a
            one-stop continuation, 2 for a two-stop), so the family the density is
            conditioned on is the plan's *total* stop count - what the car has
            already taken plus what it is about to.  On a first-stint car that is
            simply `more_stops`.
            """
            if not first_stint:
                return 0.0
            return self._first_stop_pen(cur_c, p_laps, n_stops=stops_taken + int(more_stops))

        # -- phase 1: every option on this car's expected cost -------------------
        # per option: expected cost, kind, stop lap, legality, and the recipe
        # (compounds and laps) to price it on the draws if it makes the shortlist
        means, kinds, stops, legals, recipes = [], [], [], [], []
        xv3s, tv3s, nxcs = [], [], []
        cidx = {c: i for i, c in enumerate(self.model.compounds)}
        legal0 = bool((len(compounds_used) >= 2 or not state.is_race) and cont_ok_vec(R))
        means.append(np.array([cont_m[R]])); kinds.append(np.array([0])); stops.append(np.array([-1]))
        legals.append(np.array([legal0])); recipes.append([("stay", None, None, None, None)])
        xv3s.append(np.zeros(1)); tv3s.append(np.zeros(1)); nxcs.append(np.array([-1]))
        P1 = np.arange(now_lap, total - margin + 1)
        K1 = P1 - cur_lap
        keep = K1 <= R - margin
        P1, K1 = P1[keep], K1[keep]
        if len(P1):
            rem = total - P1
            fsp1 = first_stop_pen(P1, 1)
            fixed1 = (pit * np.where(P1 == now_lap, pit_now_factor, 1.0)
                      + traffic * dens[np.clip(P1, 1, total) - 1] + fsp1)
            fsp1 = np.broadcast_to(np.asarray(fsp1, dtype=float), P1.shape)
            for c2 in avail:
                legal = (~((c2 == cur_c) and len(compounds_used) < 2)) & cont_ok_vec(K1) & cap_ok_vec(c2, rem)
                ex1 = expo_cont(c2, K1)
                means.append(cont_m[K1] + fixed1 + mean_t[c2][P1, rem] + ex1)
                kinds.append(np.full(len(P1), 1)); stops.append(P1)
                legals.append(np.asarray(legal, bool) & np.ones(len(P1), bool))
                recipes.append([(c2, int(p), int(r), None, None) for p, r in zip(P1, rem)])
                xv3s.append(fsp1 + ex1); tv3s.append(traffic * dens[np.clip(P1, 1, total) - 1])
                nxcs.append(np.full(len(P1), cidx[c2]))
        step = 2 if R > 30 else 1
        p1s = np.arange(now_lap, total - 2 * margin + 1, step)
        pairs = [(p1, p2) for p1 in p1s for p2 in range(p1 + margin, total - margin + 1, step)]
        if pairs:
            P1p = np.array([a for a, _ in pairs]); P2p = np.array([b for _, b in pairs])
            K1p = P1p - cur_lap
            fsp2 = first_stop_pen(P1p, 2)
            fixed2 = (pit * np.where(P1p == now_lap, pit_now_factor, 1.0) + pit
                      + traffic * (dens[P1p - 1] + dens[P2p - 1]) + fsp2)
            fsp2 = np.broadcast_to(np.asarray(fsp2, dtype=float), P1p.shape)
            for c2 in avail:
                e1 = expo_cont(c2, K1p)
                for c3 in avail:
                    if c2 == c3 and used.get(c2, 0) + 2 > int(self.wm.allocation.get(c2, MAX_STINTS_PER_COMPOUND)):
                        continue
                    legal = cont_ok_vec(K1p) & cap_ok_vec(c2, P2p - P1p) & cap_ok_vec(c3, total - P2p)
                    if not np.any(legal):
                        continue
                    m2 = cont_m[K1p] + fixed2 + mean_t[c2][P1p, P2p - P1p] + mean_t[c3][P2p, total - P2p] + e1
                    if expo is not None:
                        e2 = expo[(c2, c3)]
                        m2 = m2 + self.lam * e2[np.clip(P2p - P1p, 0, len(e2) - 1)] * dens[P2p - 1]
                    means.append(m2); kinds.append(np.full(len(P1p), 2)); stops.append(P1p)
                    legals.append(np.asarray(legal, bool))
                    recipes.append([(c2, int(a), int(b - a), c3, int(total - b)) for a, b in pairs])
                    xv3s.append(fsp2 + e1); tv3s.append(traffic * dens[P1p - 1])
                    nxcs.append(np.full(len(P1p), cidx[c2]))
        return {"cur_lap": cur_lap, "R": R, "now_lap": now_lap, "w": w, "cont": cont, "cont_m": cont_m,
                "mean_t": mean_t, "tables": tables, "expo": expo, "pit": pit, "pit_now_factor": pit_now_factor,
                "dens": dens, "traffic": traffic, "caps": caps, "legal0": legal0,
                "compounds_used": compounds_used, "avail": avail, "cur_c": cur_c,
                "stint_len_now": stint_len_now, "first_stint": first_stint, "stops_taken": stops_taken,
                "mean_all": np.concatenate(means), "kind": np.concatenate(kinds),
                "stop": np.concatenate(stops), "legal": np.concatenate(legals),
                "recipe": [r for blk in recipes for r in blk],
                "xv3": np.concatenate(xv3s), "tv3": np.concatenate(tv3s), "nxc": np.concatenate(nxcs),
                "first_stop_pen": first_stop_pen, "expo_cont": expo_cont}

    def _plan_finish(self, state: LiveState, num: str, dt: DriverTyre, tr, total: int, ctx: dict,
                     rs: dict | None = None) -> dict:
        """Phase 2 of a car's plan: choose, price the shortlist on the draws, report.

        With `rs` (from `_race_state_extra`) each option's next stop is charged
        the race state - the expected places lost to the relevant rivals at the
        measured value of a place, and the traffic the timing screen says the car
        would rejoin into - in place of V3's exposure and first-stop prior."""
        cur_lap, R, now_lap, w = ctx["cur_lap"], ctx["R"], ctx["now_lap"], ctx["w"]
        cont, tables, expo = ctx["cont"], ctx["tables"], ctx["expo"]
        pit, pit_now_factor, dens, traffic = ctx["pit"], ctx["pit_now_factor"], ctx["dens"], ctx["traffic"]
        caps, legal0, compounds_used, avail = ctx["caps"], ctx["legal0"], ctx["compounds_used"], ctx["avail"]
        stint_len_now, first_stint, stops_taken = ctx["stint_len_now"], ctx["first_stint"], ctx["stops_taken"]
        first_stop_pen, expo_cont = ctx["first_stop_pen"], ctx["expo_cont"]
        n = self.n
        kind, stop, legal, recipe = ctx["kind"], ctx["stop"], ctx["legal"], ctx["recipe"]
        if rs is None:
            mean_all = ctx["mean_all"]
            rs_adj = None
        else:
            rs_adj = rs["opt_extra"] - ctx["xv3"]
            mean_all = ctx["mean_all"] + rs_adj
        if not legal.any():
            legal = np.ones(len(stop), bool)
        li = np.flatnonzero(legal)
        ml = mean_all[li]
        j_best = int(np.argmin(ml))
        best_i = int(li[j_best])
        # next-stop window: best expected cost per candidate stop lap
        st_l = stop[li]
        has_stop = st_l >= 0
        window, in_win = [], []
        if has_stop.any():
            laps_u = np.unique(st_l[has_stop])
            best_by_lap = np.full(len(laps_u), np.inf)
            np.minimum.at(best_by_lap, np.searchsorted(laps_u, st_l[has_stop]), ml[has_stop])
            floor = float(best_by_lap.min())
            window = [{"lap": int(p), "loss_s": float(v - floor)} for p, v in zip(laps_u, best_by_lap)]
            in_win = [x["lap"] for x in window if x["loss_s"] <= WINDOW_TOL_S]
        now_mask = st_l == now_lap
        box_i = int(li[np.flatnonzero(now_mask)[np.argmin(ml[now_mask])]]) if now_mask.any() else None
        stay_i = int(li[0]) if (kind[li[0]] == 0) else None

        # -- phase 2: the shortlist on the draws -------------------------------
        # the best options by expected cost, plus box-now and stay-out, priced
        # draw by draw on this car's resampled posterior for the win probabilities
        top_j = np.argsort(ml)[:PLAN_SHORTLIST]
        short = list(dict.fromkeys([int(li[j]) for j in top_j] + [x for x in (box_i, stay_i) if x is not None]))
        idx = self._resample(w)
        rows = []
        for i in short:
            c2, p, r1, c3, r2 = recipe[i]
            if c2 == "stay":
                t = cont[idx, R]
            elif c3 is None:
                k = p - cur_lap
                t = (cont[idx, k] + float(pit * (pit_now_factor if p == now_lap else 1.0) + traffic * dens[min(p, total) - 1])
                     + tables[c2][idx, p, r1] + float(expo_cont(c2, np.array([k]))[0])
                     + float(first_stop_pen(p, 1)))
            else:
                k = p - cur_lap
                p2 = p + r1
                t = (cont[idx, k] + float(pit * (pit_now_factor if p == now_lap else 1.0) + pit
                                          + traffic * (dens[p - 1] + dens[p2 - 1]))
                     + tables[c2][idx, p, r1] + tables[c3][idx, p2, r2] + float(expo_cont(c2, np.array([k]))[0])
                     + float(first_stop_pen(p, 2)))
                if expo is not None:
                    e2 = expo[(c2, c3)]
                    t = t + self.lam * e2[min(r1, len(e2) - 1)] * dens[p2 - 1]
            if rs_adj is not None:
                t = t + float(rs_adj[i])
            rows.append(np.asarray(t, dtype=np.float64))
        T = np.stack(rows)                                                   # (n_short, n)
        win_short = np.bincount(np.argmin(T, axis=0), minlength=len(short)) / n
        win_of = dict(zip(short, win_short))
        pos_of = {i: j for j, i in enumerate(short)}
        best_t = T[pos_of[best_i]]
        delta_box_now = float(mean_all[box_i] - mean_all[best_i]) if box_i is not None else float("nan")
        delta_stay = float(mean_all[stay_i] - mean_all[best_i]) if stay_i is not None else float("nan")
        p_best_vs_stay = (float(((best_t - T[pos_of[stay_i]]) < 0).mean())
                          if stay_i is not None and stay_i != best_i else float("nan"))

        def label_of(i):
            c2, p, r1, c3, r2 = recipe[i]
            if c2 == "stay":
                return "stay out", []
            if c3 is None:
                return f"1 stop: {c2} on lap {p}", [c2]
            return f"2 stops: {c2} lap {p}, {c3} lap {p + r1}", [c2, c3]

        best_label, best_comps = label_of(best_i)
        rs_report = None
        if rs is not None:
            # the five actions, on the race-state curve: cost by next-stop lap,
            # best continuation at each lap
            S = np.asarray(rs["laps"], dtype=int)
            cur_best = np.full(len(S), np.inf)
            core_best = np.full(len(S), np.inf)
            st_l_all = stop[li]
            pos_in = np.searchsorted(S, st_l_all)
            okm = (st_l_all >= 0) & (pos_in < len(S))
            okm[okm] &= S[pos_in[okm]] == st_l_all[okm]
            np.minimum.at(cur_best, pos_in[okm], ml[okm])
            core_l = (ctx["mean_all"] - ctx["xv3"])[li]
            np.minimum.at(core_best, pos_in[okm], core_l[okm])
            fin = np.isfinite(cur_best)
            pos_l = np.asarray(rs["position_s"], dtype=float)
            trf_l = np.asarray(rs["traffic_s"], dtype=float)
            tbl = racestate.action_table(
                S[fin], cur_best[fin],
                {"tyre_s": core_best[fin] - np.nanmin(core_best[fin]) if fin.any() else core_best[fin],
                 "position_s": pos_l[fin] - np.nanmin(pos_l[fin]) if fin.any() else pos_l[fin],
                 "traffic_s": trf_l[fin]},
                now_lap=now_lap, window_hi=(max(in_win) if in_win else None),
                extra={"rejoin_position": [rs["rejoin"].get(int(s)) for s in S[fin]]})
            rs_report = {**tbl, "rivals": rs["rivals"], "place_value_s": rs["place_value_s"],
                         "sigma_rel_s": rs["sigma_rel_s"], "n_rivals": len(rs["rivals"]),
                         "position_s_stay": rs["position_stay_s"]}
        return {
            "race_state": rs_report,
            "best": best_label, "best_kind": int(kind[best_i]),
            "next_stop": (int(stop[best_i]) if stop[best_i] >= 0 else None),
            "next_compound": (best_comps[0] if best_comps else None),
            "compounds": list(best_comps), "win_prob": float(win_of.get(best_i, 0.0)),
            "window_lo": (min(in_win) if in_win else None), "window_hi": (max(in_win) if in_win else None),
            "window": window[:60],
            "delta_box_now_s": delta_box_now, "delta_stay_out_s": delta_stay,
            "p_best_vs_stay": p_best_vs_stay,
            "stay_out_legal": bool(legal0), "compounds_used": sorted(compounds_used),
            "available": avail, "pit_now_factor": pit_now_factor,
            "options": [{"label": label_of(int(li[j]))[0], "delta_s": float(ml[j] - ml[j_best]),
                         "win_prob": float(win_of.get(int(li[j]), 0.0))} for j in top_j[:6]],
            "laps_remaining": int(R), "n_options": int(len(li)), "now_lap": int(now_lap),
            "stint_cap": caps.get(dt.compound), "stint_len_now": int(stint_len_now),
            "box_now_label": (label_of(box_i)[0] if box_i is not None else None),
            "undercut_lambda": self.lam,
            "first_stop_kappa_s": self.kappa,
            "first_stop_prior_applies": bool(rs is None and first_stint and self.first_stop_table is not None
                                             and self.kappa > 0),
            "first_stop_s": (float(first_stop_pen(int(stop[best_i]), max(int(kind[best_i]), 1)))
                             if (stop[best_i] >= 0 and rs is None) else 0.0),
            "first_stop_n_stops": (stops_taken + int(kind[best_i]) if stop[best_i] >= 0 else None),
        }

    # -- race state ---------------------------------------------------------

    def _curve_of(self, ctx: dict, mean_all: np.ndarray | None = None) -> tuple:
        """A car's expected cost by next-stop lap (best legal option at each lap)
        and its no-further-stop cost: what its stop distribution is drawn from."""
        m = ctx["mean_all"] if mean_all is None else mean_all
        stop, legal = ctx["stop"], ctx["legal"]
        ok = legal & (stop >= 0)
        if not ok.any():
            return np.zeros(0, int), np.zeros(0), (float(m[0]) if ctx["legal0"] else float("inf"))
        laps = np.unique(stop[ok])
        best = np.full(len(laps), np.inf)
        np.minimum.at(best, np.searchsorted(laps, stop[ok]), m[ok])
        return laps, best, (float(m[0]) if ctx["legal0"] else float("inf"))

    def _fresh_rows(self, ctx: dict, total: int) -> tuple:
        """`rows[s, j]`: the expected cost of j laps on the set the car would fit
        if it stopped after lap s (its best option's compound at that lap)."""
        core = ctx["mean_all"] - ctx["xv3"]
        stop, legal, nxc = ctx["stop"], ctx["legal"], ctx["nxc"]
        ok = legal & (stop >= 0)
        rows = np.zeros((total + 1, total + 1))
        comp_at = {}
        if not ok.any():
            return rows, comp_at
        o = np.lexsort((core[ok], stop[ok]))
        s_sorted, c_sorted = stop[ok][o], nxc[ok][o]
        uniq, at = np.unique(s_sorted, return_index=True)
        for s, ci in zip(uniq, c_sorted[at]):
            c = self.model.compounds[int(ci)]
            tbl = ctx["mean_t"].get(c)
            if tbl is not None and s <= total:
                rows[int(s), :tbl.shape[1]] = tbl[int(s)]
                comp_at[int(s)] = c
        return rows, comp_at

    def _car_views(self, state: LiveState, order: list, ctxs: dict, total: int) -> dict:
        """`racestate.CarView` for every car with a plan context this tick."""
        rows = {r["driver_number"]: r for r in order}
        views = {}
        for num, ctx in ctxs.items():
            if ctx is None:
                continue
            r = rows.get(num) or {}
            fresh, comp_at = self._fresh_rows(ctx, total)
            prev = self._rs_curves.get(num)
            if prev is not None:
                c_laps, c_cost, stay = prev
            else:
                c_laps, c_cost, stay = self._curve_of(ctx)
            plan = self._plans.get(num) or {}
            nxt = plan.get("next_compound")
            if not nxt and comp_at:
                nxt = comp_at[min(comp_at)]
            gl = 0.0 if r.get("position") == 1 else r.get("gap_leader_s")
            views[num] = racestate.CarView(
                number=num, code=state.driver_label(num), cur_lap=int(ctx["cur_lap"]),
                gap_leader_s=(float(gl) if gl is not None else None),
                position=r.get("position"), stops=int(ctx["stops_taken"]), in_pit=bool(r.get("in_pit")),
                compound=ctx["cur_c"], tyre_age=r.get("tyre_age"), cont=np.asarray(ctx["cont_m"], dtype=float),
                fresh_rows=fresh, next_compound=nxt, curve_laps=np.asarray(c_laps, dtype=int),
                curve_cost=np.asarray(c_cost, dtype=float), stay_cost=float(stay))
        return views

    def _race_state_extra(self, state: LiveState, num: str, ctx: dict, views: dict, order: list,
                          total: int) -> dict | None:
        """The race-state charge on each of this car's options' next stop.

        The four cars nearest in virtual race position, their gaps, sets, tyre
        ages, stops made, pit status and last lap's plans (`racestate.
        live_position_term`), plus the traffic the timing screen says the car
        would rejoin into over the next four laps (`racestate.rejoin_traffic`,
        replacing the V3 density model's charge there).  None when the car has
        no race-time gap to price it with (lapped, or before the first timing
        line) - the plan then keeps V3's terms."""
        me = views.get(num)
        if me is None or me.gap_leader_s is None or not np.isfinite(me.gap_leader_s):
            return None
        stop = ctx["stop"]
        S = np.unique(stop[stop >= 0])
        if not len(S):
            return None
        pit, factor, now_lap = ctx["pit"], ctx["pit_now_factor"], ctx["now_lap"]
        same_lap = {k: v for k, v in views.items() if abs(v.cur_lap - me.cur_lap) <= 1}
        riv = racestate.relevant_rivals(me, same_lap, pit)
        for k, _, _ in riv:
            r = same_lap[k]
            best_stop = float(np.min(r.curve_cost)) if len(r.curve_cost) else float("inf")
            done = np.isfinite(r.stay_cost) and r.stay_cost <= best_stop
            r.pending = bool(r.stops <= me.stops and not done)
        pos, detail = racestate.live_position_term(me, riv, same_lap, S, pit_now_s=pit * factor, pit_s=pit,
                                                   now_lap=now_lap, const=self.race_state)
        # traffic on rejoin, from the screen, over the laps it can be seen for
        others = [v.gap_leader_s for k, v in same_lap.items() if k != num and not v.in_pit]
        traffic_l = np.array([float(ctx["traffic"] * ctx["dens"][int(s) - 1]) for s in S])
        rejoin = {}
        for i, s in enumerate(S):
            if s > now_lap + 3:
                break
            rj = racestate.rejoin_traffic(me.gap_leader_s, others, pit * (factor if s == now_lap else 1.0),
                                          dirty_air_s_per_lap=self.dirty_air,
                                          laps_per_stop=TRAFFIC_LAPS_PER_STOP, sigma_s=self.race_state.sigma_rel_s)
            if np.isfinite(rj["traffic_s"]):
                traffic_l[i] = rj["traffic_s"]
                rejoin[int(s)] = rj["rejoin_position"]
        tadj = traffic_l - np.array([float(ctx["traffic"] * ctx["dens"][int(s) - 1]) for s in S])
        base = float(np.min(pos))
        pos_s = pos[:-1] - base
        at = np.searchsorted(S, np.clip(stop, S[0], S[-1]))
        opt = np.where(stop >= 0, pos_s[at] + tadj[at], pos[-1] - base)
        for d in detail:
            d["p_ahead_now"] = float(d["p_ahead"][0]) if len(d["p_ahead"]) else None
            d["p_ahead_stay_3"] = float(d["p_ahead"][min(3, len(d["p_ahead"]) - 1)]) if len(d["p_ahead"]) else None
            d["p_ahead"] = None
        return {"opt_extra": opt, "laps": [int(s) for s in S], "position_s": [float(x) for x in pos_s],
                "traffic_s": [float(x) for x in traffic_l], "rejoin": rejoin, "rivals": detail,
                "place_value_s": float(self.race_state.place_value_s),
                "sigma_rel_s": float(self.race_state.sigma_rel_s), "position_stay_s": float(pos[-1] - base)}

    def _density(self, total: int) -> np.ndarray:
        f = np.arange(1, total + 1, dtype=float) / total
        q = 0.692 - 1.037 * f + 0.826 * f ** 2
        return q / 0.4488

    # -- undercut ----------------------------------------------------------

    def _undercut(self, state: LiveState, order: list, num: str, dt: DriverTyre) -> dict:
        """Threat from behind and opportunity on the car ahead.

        If the attacker pits now and the defender stays out `k` more laps
        before its own stop, the attacker gains, over those laps, the pace the
        defender's aged tyre is *slower than fresh by* plus the compound
        difference, less what its own fresh tyre loses and the cold first lap.
        The undercut works if that cumulative gain exceeds the gap.  Level-
        based, like the offline calculator: a 15-lap-old tyre is slower by its
        accumulated degradation, not by its last lap's increment.
        """
        out = {"threat": None, "opportunity": None}
        pos = {r["driver_number"]: r for r in order}
        me = pos.get(num)
        if me is None or me.get("position") is None:
            return out
        w_me = dt.weights if dt.weights is not None else np.ones(self.n) / self.n
        K = 5

        def cum_gain(attacker: DriverTyre, defender: DriverTyre, new_c: str) -> np.ndarray:
            """(n, K): attacker's cumulative gain after k laps on the fresh tyre."""
            if defender.wear is None or defender.compound not in self.model.compounds:
                return np.zeros((self.n, K))
            rate_d = self.model.wear_rate[defender.compound] * self.m_prior
            rate_n = self.model.wear_rate[new_c] * self.m_prior
            w0 = defender.wear
            ks = np.arange(1, K + 1)
            def_loss = grip_loss(w0[:, None] + rate_d[:, None] * ks[None, :], budget=self.model.budget_of(defender.compound))
            att_loss = grip_loss(rate_n[:, None] * ks[None, :], budget=self.model.budget_of(new_c))
            off = (self.model.pace_offset[defender.compound] - self.model.pace_offset[new_c])[:, None]
            per_lap = def_loss - att_loss + off
            return np.cumsum(per_lap, axis=1) - OUT_LAP_PENALTY_S

        def summarise(gain: np.ndarray, gap: float) -> dict:
            p_k = [float(((gain[:, k] > gap) * w_me).sum()) for k in range(K)]
            med = [float(np.sum(gain[:, k] * w_me)) for k in range(K)]
            laps_needed = next((k + 1 for k in range(K) if p_k[k] >= 0.5), None)
            return {"gap_s": float(gap), "p_by_lap": p_k, "gain_by_lap_s": med,
                    "laps_needed": laps_needed, "p_undercut_1lap": p_k[0], "p_undercut_3lap": p_k[2],
                    "gain_per_lap_s": float(med[1] - med[0]) if K > 1 else float(med[0])}

        behind = [r for r in order if r.get("position") == me["position"] + 1]
        if behind:
            b = behind[0]
            g = b.get("interval_s")
            bt = self.tyres.get(b["driver_number"])
            new_c = self._likely_new_compound(state, b["driver_number"])
            if g is not None and 0 < g < UNDERCUT_RANGE_S and bt is not None \
                    and dt.compound in self.model.compounds and new_c:
                out["threat"] = {"driver": b["driver"], "new_compound": new_c,
                                 **summarise(cum_gain(bt, dt, new_c), g)}
        ahead = [r for r in order if r.get("position") == me["position"] - 1]
        if ahead:
            a = ahead[0]
            g = me.get("interval_s")
            at = self.tyres.get(a["driver_number"])
            new_c = self._likely_new_compound(state, num)
            if g is not None and 0 < g < UNDERCUT_RANGE_S and at is not None \
                    and at.compound in self.model.compounds and new_c:
                out["opportunity"] = {"driver": a["driver"], "new_compound": new_c,
                                      **summarise(cum_gain(dt, at, new_c), g)}
        return out

    def _likely_new_compound(self, state: LiveState, num: str) -> str | None:
        plan = self._plans.get(num)
        if plan and plan.get("next_compound"):
            return plan["next_compound"]
        tr = state.tracks.get(num)
        c = (tr.current_stint or {}).get("compound") if tr else None
        for cand in ("HARD", "MEDIUM", "SOFT"):
            if cand in self.model.compounds and cand != c:
                return cand
        return None

    # -- rejoin --------------------------------------------------------------

    def _rejoin(self, order: list, num: str, extra_s: float) -> dict | None:
        me = next((r for r in order if r["driver_number"] == num), None)
        if me is None or me.get("position") is None:
            return None
        g0 = me.get("gap_leader_s") if me.get("position", 1) > 1 else 0.0
        if g0 is None:
            return None
        g_new = g0 + extra_s
        ahead = []
        for r in order:
            if r["driver_number"] == num:
                continue
            g = r.get("gap_leader_s") if (r.get("position") or 99) > 1 else 0.0
            if g is None:
                continue
            ahead.append((g, r))
        ahead_sorted = sorted(ahead, key=lambda t: t[0])
        n_ahead = sum(1 for g, _ in ahead_sorted if g < g_new)
        car_ahead = next((r for g, r in reversed(ahead_sorted) if g < g_new), None)
        car_behind = next((r for g, r in ahead_sorted if g >= g_new), None)
        return {"position": n_ahead + 1,
                "behind": (car_ahead["driver"] if car_ahead else None),
                "gap_to_ahead_s": (float(g_new - (car_ahead["gap_leader_s"] or 0.0)) if car_ahead else None),
                "ahead_of": (car_behind["driver"] if car_behind else None),
                "gap_to_behind_s": (float((car_behind["gap_leader_s"] or 0.0) - g_new) if car_behind else None)}

    # -- pit loss, measured live -------------------------------------------

    def _measure_pit_loss(self, laps: pd.DataFrame) -> None:
        rows = []
        for num, g in laps[laps["is_complete"]].groupby("driver_number"):
            g = g.sort_values("lap_number")
            clean = g[g["is_accurate"]]
            if clean.empty:
                continue
            for i, r in g.iterrows():
                if not r["pit_in"]:
                    continue
                nxt = g[g["lap_number"] == r["lap_number"] + 1]
                if nxt.empty or not bool(nxt.iloc[0]["pit_out"]):
                    continue
                if r["track_status"] != "1" or nxt.iloc[0]["track_status"] != "1":
                    continue
                t_in, t_out = r["lap_time_s"], nxt.iloc[0]["lap_time_s"]
                if not (np.isfinite(t_in) and np.isfinite(t_out)):
                    continue
                near = clean[(clean["lap_number"] >= r["lap_number"] - 5) & (clean["lap_number"] <= r["lap_number"] + 6)]
                if len(near) < 3:
                    continue
                loss = t_in + t_out - 2 * float(near["lap_time_s"].median())
                if 5 < loss < 60:
                    rows.append({"driver": r["driver"], "lap": int(r["lap_number"]), "loss_s": float(loss)})
        self.pit_loss_live = rows
        if len(rows) >= 2:
            self.pit_loss_s = float(np.median([r["loss_s"] for r in rows]))
            self.pit_loss_source = f"measured live from {len(rows)} green-flag stops"
        else:
            self.pit_loss_s = self.wm.pit_loss_s
            self.pit_loss_source = self.wm.pit_loss_source

    # -- alerts ----------------------------------------------------------------

    def _alert(self, state: LiveState, num: str, kind: str, level: str, text: str, lap: int) -> None:
        key = (num, kind)
        last = self._alert_last.get(key)
        if last is not None and lap - last < ALERT_COOLDOWN_LAPS:
            return
        self._alert_last[key] = lap
        self.alerts.append({"t_session": state.t_now, "utc": (state.utc_now.isoformat() if state.utc_now else None),
                            "lap": lap, "driver": state.driver_label(num), "driver_number": num,
                            "kind": kind, "level": level, "text": text})
        self.alerts = self.alerts[-200:]

    # -- tick ---------------------------------------------------------------

    def tick(self, state: LiveState, *, replan: bool = True) -> dict:
        self.tick_no += 1
        total = self._total_laps(state)
        laps = state.laps_df(include_current=False)
        if not laps.empty:
            self._measure_pit_loss(laps)
            self._update_tyres(state, laps, total)
        order = state.field_snapshot()
        sc = state.track_status in ("4", "6", "7")
        field = []
        todo = []
        # -- pass 1: each car on its own tyre and the pit lane ---------------------
        for r in order:
            num = r["driver_number"]
            tr = state.tracks[num]
            dt = self.tyres.get(num)
            row = dict(r)
            row.update({"m_mean": float("nan"), "m_lo": float("nan"), "m_hi": float("nan"),
                        "wear": float("nan"), "p_past_cliff": float("nan"),
                        "laps_to_cliff_p10": float("nan"), "laps_to_cliff_p50": float("nan"),
                        "laps_to_cliff_p90": float("nan"), "deg_now_s_per_lap": float("nan"),
                        "level_s": float("nan"), "n_clean": 0, "proj": [], "cliff_alarm": False,
                        "pace_collapse": False, "collapse_kind": None, "collapse_knee_age": float("nan"),
                        "collapse_slope_post": float("nan"), "collapse_why": None,
                        "wear_alarm": False})
            if dt is not None and dt.wear is not None:
                w = dt.weights if dt.weights is not None else np.ones(self.n) / self.n
                row.update({"m_mean": dt.m_mean, "m_lo": dt.m_lo, "m_hi": dt.m_hi,
                            "wear": float((dt.wear * w).sum()), "p_past_cliff": dt.p_past_cliff,
                            "laps_to_cliff_p10": dt.laps_to_cliff[0],
                            "laps_to_cliff_p50": dt.laps_to_cliff[1],
                            "laps_to_cliff_p90": dt.laps_to_cliff[2],
                            "deg_now_s_per_lap": dt.deg_now_s_per_lap, "level_s": dt.level_s,
                            "n_clean": int(dt.n_clean), "proj": dt.proj,
                            # `cliff_alarm` is the key the app reads; it now carries
                            # the *measured* collapse.  The model's wear-based
                            # forecast is still here, under its own name.
                            "cliff_alarm": bool(dt.pace_collapse),
                            "pace_collapse": bool(dt.pace_collapse),
                            "collapse_kind": dt.collapse.get("kind"),
                            "collapse_knee_age": float(dt.collapse.get("knee_age", float("nan"))),
                            "collapse_slope_post": float(dt.collapse.get("slope_post", float("nan"))),
                            "collapse_why": dt.collapse.get("why"),
                            "wear_alarm": bool(dt.p_past_cliff >= CLIFF_ALARM_P
                                               or dt.laps_to_cliff[1] <= 2.0)})
            lap_no = int(tr.current["lap_number"]) if tr.current else 0
            eligible = bool(dt is not None and dt.wear is not None and not tr.retired and state.is_race
                            and state.session_status not in ("Finished", "Finalised", "Ends"))
            needs = bool(eligible and replan and (self._last_plan_lap.get(num) != lap_no or sc))
            if needs:
                try:
                    self._ctx[num] = self._plan_prepare(state, num, dt, tr, total)
                except Exception:
                    log.exception("plan failed for %s", num)
                    self._ctx[num] = None
            todo.append((row, num, tr, dt, lap_no, eligible, needs))
            field.append(row)
        # -- pass 2: the race state against the nearest rivals, then the call -------
        views = None
        if self.race_state is not None:
            ctxs = {t[1]: self._ctx.get(t[1]) for t in todo if t[6] and self._ctx.get(t[1]) is not None}
            try:
                views = self._car_views(state, order, ctxs, total)
            except Exception:
                log.exception("race-state views failed; this tick prices the V3 objective")
                views = None
        new_curves = {}
        for row, num, tr, dt, lap_no, eligible, needs in todo:
            if not eligible:
                continue
            if needs:
                ctx = self._ctx.get(num)
                if ctx is None:
                    self._plans[num] = None
                else:
                    try:
                        rs = None
                        if views is not None:
                            try:
                                rs = self._race_state_extra(state, num, ctx, views, order, total)
                            except Exception:
                                log.exception("race state failed for %s; its plan keeps V3's terms", num)
                                rs = None
                        self._plans[num] = self._plan_finish(state, num, dt, tr, total, ctx, rs)
                        if rs is not None:
                            new_curves[num] = self._curve_of(ctx, ctx["mean_all"] - ctx["xv3"] + rs["opt_extra"])
                    except Exception:
                        log.exception("plan failed for %s", num)
                        self._plans[num] = None
                self._last_plan_lap[num] = lap_no
            plan = self._plans.get(num)
            row["plan"] = plan
            uc = self._undercut(state, order, num, dt)
            row["undercut"] = uc
            if plan is not None:
                factor = plan.get("pit_now_factor", 1.0)
                row["rejoin_if_box_now"] = self._rejoin(order, num, self.pit_loss_s * factor)
                if row["pace_collapse"]:
                    self._alert(state, num, "collapse", "bad",
                                f"PACE COLLAPSE on the {dt.compound}: pace broke away at age "
                                f"{row['collapse_knee_age']:.0f}, now {row['collapse_slope_post']:+.2f} s/lap "
                                f"({row['collapse_kind']}); the model still gives it "
                                f"~{row['laps_to_cliff_p50']:.0f} laps", lap_no)
                # Box-now is within a tenth of the best plan: the decision is
                # live this lap.  Fired under a safety car too - the numbers
                # say the same thing - beside the SC-specific message below.
                if np.isfinite(plan["delta_box_now_s"]) and plan["delta_box_now_s"] <= BOX_NOW_TOL_S \
                        and plan["best_kind"] > 0:
                    rj = row.get("rejoin_if_box_now") or {}
                    self._alert(state, num, "box_now", "warn",
                                f"BOX NOW is live: it costs {plan['delta_box_now_s']:+.1f} s against "
                                f"{plan['best']}; rejoin P{rj.get('position', '?')}", lap_no)
                if sc and np.isfinite(plan["delta_box_now_s"]) and plan["delta_box_now_s"] < 0.5 \
                        and plan["best_kind"] > 0:
                    rj = row["rejoin_if_box_now"] or {}
                    self._alert(state, num, "sc_box", "warn",
                                f"{'SC' if state.track_status == '4' else 'VSC'} — box now: costs "
                                f"{plan['delta_box_now_s']:+.1f} s vs the best green plan"
                                f" (pit loss x{factor:.2f}); rejoin P{rj.get('position', '?')}", lap_no)
                if plan.get("window_lo") is not None and plan["window_lo"] <= lap_no <= (plan["window_hi"] or 0) \
                        and plan["best_kind"] > 0:
                    self._alert(state, num, "window", "accent",
                                f"Pit window open: laps {plan['window_lo']}-{plan['window_hi']} "
                                f"({plan['best']}, box-now costs {plan['delta_box_now_s']:+.1f} s)", lap_no)
                th = uc.get("threat")
                if th and th["p_undercut_3lap"] > 0.5:
                    self._alert(state, num, "undercut_threat", "warn",
                                f"UNDERCUT THREAT from {th['driver']} (gap {th['gap_s']:.1f} s): on a fresh "
                                f"{th['new_compound']} they are ahead after "
                                f"{th['laps_needed'] or '>5'} lap(s) if you stay out — "
                                f"P(1 lap) {th['p_undercut_1lap']:.0%}, P(3 laps) {th['p_undercut_3lap']:.0%}", lap_no)
                op = uc.get("opportunity")
                if op and op["p_undercut_3lap"] > 0.5:
                    self._alert(state, num, "undercut_opp", "good",
                                f"UNDERCUT ON {op['driver']} (gap {op['gap_s']:.1f} s): box now and you are "
                                f"ahead after {op['laps_needed'] or '>5'} lap(s) if they stay out — "
                                f"P(3 laps) {op['p_undercut_3lap']:.0%}", lap_no)
        self._rs_curves.update(new_curves)
        m_field = self.m_prior
        fw = self.field_weights
        o = np.argsort(m_field)
        cw = np.cumsum(fw[o])
        m_summary = {"mean": float((m_field * fw).sum()),
                     "p05": float(m_field[o][np.searchsorted(cw, 0.05)]),
                     "p95": float(m_field[o][min(np.searchsorted(cw, 0.95), self.n - 1)]),
                     "prior_mean": float(self.m_prior.mean())}
        return {
            "engine": "race",
            "meta": {**state.summary(), "total_laps": total, "pit_loss_s": self.pit_loss_s,
                     "evolution_s": [round(float(x), 3) for x in getattr(self, "evo", [])],
                     "pit_loss_source": self.pit_loss_source, "pit_stops_seen": len(self.pit_loss_live),
                     "regime_multiplier": m_summary, "model_source": self.wm.source,
                     "field_temperature": float(getattr(self, "field_temperature", 1.0)),
                     "sealed_file": self.wm.sealed_file, "n_draws": self.n,
                     "sc_active": sc, "tick": self.tick_no, "undercut_lambda": self.lam,
                     "dirty_air_s_per_lap": self.dirty_air,
                     "first_stop_kappa_s": self.kappa,
                     "first_stop_prior": bool(self.first_stop_table is not None and self.race_state is None),
                     "race_state": (self.race_state.as_dict() if self.race_state is not None else None),
                     "percar_pooled_drivers": len(self.pooled_dev),
                     "tick_utc": datetime.now(timezone.utc).isoformat()},
            "field": field,
            "alerts": list(self.alerts[-60:]),
            "compounds": {c: {"life_full_push": float(self.model.life_laps(c, 1.0).mean()),
                              "pace_offset_s": float(self.model.pace_offset[c].mean()),
                              "grip_budget_s": float(self.model.budget_of(c))}
                          for c in self.model.compounds},
        }


# --------------------------------------------------------------------------
# Practice: the live long-run board
# --------------------------------------------------------------------------


class PracticeEngine:
    """Per-stint degradation from the laps run so far, the way the pipeline does it."""

    def __init__(self, wm: WeekendModel):
        self.wm = wm
        self.ev = wm.event
        self.fp = get_prior(self.ev, "2026")
        self.model = wm.model
        self.tick_no = 0
        self.alerts: list = []
        self._seen: set = set()

    def tick(self, state: LiveState, **_) -> dict:
        self.tick_no += 1
        laps = state.laps_df(include_current=False)
        board, pooled = [], {}
        if not laps.empty:
            d = laps[laps["is_accurate"] & (laps["track_status"] == "1") & ~laps["pit_in"] & ~laps["pit_out"]
                     & laps["compound"].isin(VALID_COMPOUNDS) & laps["lap_time_s"].notna()].copy()
            d = d.sort_values("lap_start_s")
            starts = d["lap_start_s"].to_numpy(dtype=float)
            drivers = d["driver_number"].to_numpy()
            gap = np.full(len(d), np.inf)
            for i in range(len(d)):
                for j in range(i - 1, max(-1, i - 40), -1):
                    if drivers[j] != drivers[i] and np.isfinite(starts[i]) and np.isfinite(starts[j]):
                        gap[i] = starts[i] - starts[j]
                        break
            d["gap_ahead_s"] = gap
            d = d[d["gap_ahead_s"] > 2.0]
            rows = []
            for (num, stint), g in d.groupby(["driver_number", "stint"]):
                g = g.sort_values("lap_number")
                med = g["lap_time_s"].median()
                g = g[g["lap_time_s"] < med + SLOW_LAP_MARGIN_S]
                if len(g) < 4:
                    continue
                lis = g["lap_number"] - g["lap_number"].min() + 1
                y = g["lap_time_s"].to_numpy(dtype=float) - self.fp.term(lis.to_numpy(dtype=float))
                a = g["tyre_life"].to_numpy(dtype=float)
                if np.ptp(a) < 3:
                    continue
                A = np.column_stack([np.ones_like(a), a])
                beta, res, *_ = np.linalg.lstsq(A, y, rcond=None)
                resid = y - A @ beta
                se = float(np.sqrt(resid @ resid / max(len(y) - 2, 1) / np.sum((a - a.mean()) ** 2)))
                comp = g["compound"].iloc[0]
                rows.append({"driver": g["driver"].iloc[0], "driver_number": num, "stint": int(stint),
                             "compound": comp, "n_laps": int(len(g)), "age_from": float(a.min()),
                             "age_to": float(a.max()), "slope_s_per_lap": float(beta[1]), "se": se,
                             "level_s": float(beta[0] + beta[1] * a.mean()), "best_s": float(y.min()),
                             "laps": [int(x) for x in g["lap_number"]],
                             "y": [round(float(v), 3) for v in y], "ages": [float(x) for x in a]})
                key = (num, int(stint), int(len(g)))
                if key not in self._seen and len(g) >= 6:
                    self._seen.add(key)
                    self.alerts.append({"t_session": state.t_now, "lap": int(g["lap_number"].max()),
                                        "driver": g["driver"].iloc[0], "driver_number": num,
                                        "kind": "long_run", "level": "accent",
                                        "text": f"{g['driver'].iloc[0]} {len(g)}-lap {comp} run: "
                                                f"{beta[1]:+.3f} ± {se:.3f} s/lap (age {a.min():.0f}-{a.max():.0f})"})
            board = sorted(rows, key=lambda r: -r["n_laps"])
            if rows:
                for comp in VALID_COMPOUNDS:
                    sub = [r for r in rows if r["compound"] == comp]
                    if len(sub) < 2:
                        continue
                    xs, ys = [], []
                    for r in sub:
                        a = np.array(r["ages"]); y = np.array(r["y"])
                        xs.append(a - a.mean()); ys.append(y - y.mean())
                    x = np.concatenate(xs); y = np.concatenate(ys)
                    den = float(np.sum(x ** 2))
                    if den > 0:
                        slope = float(np.sum(x * y) / den)
                        resid = y - slope * x
                        se = float(np.sqrt(resid @ resid / max(len(y) - len(sub) - 1, 1) / den))
                        pooled[comp] = {"slope_s_per_lap": slope, "se": se, "n_stints": len(sub),
                                        "n_laps": int(len(y))}
        prior = {c: {"rate_s_per_lap": float(self.model.rate(c).mean()),
                     "lo": float(np.quantile(self.model.rate(c), 0.05)),
                     "hi": float(np.quantile(self.model.rate(c), 0.95))}
                 for c in self.model.compounds}
        return {
            "engine": "practice",
            "meta": {**state.summary(), "model_source": self.wm.source, "tick": self.tick_no,
                     "tick_utc": datetime.now(timezone.utc).isoformat(),
                     "fuel_effect_s_per_lap": self.fp.s_per_lap},
            "field": state.field_snapshot(),
            "board": board[:40], "pooled": pooled, "prior": prior,
            "alerts": list(self.alerts[-60:]),
        }


def make_engine(state: LiveState, wm: WeekendModel):
    if state.is_race:
        return RaceEngine(wm)
    return PracticeEngine(wm)
