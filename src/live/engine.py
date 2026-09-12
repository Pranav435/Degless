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
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DEFAULT_RACE_REGIME_LN_SD,
    DEFAULT_RACE_REGIME_RATIO,
    DIRTY_AIR_S_PER_LAP,
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

    @classmethod
    def load(cls, event: Event | str, *, n_draws: int = N_DRAWS, seed: int = 0) -> "WeekendModel":
        ev = get_event(event) if isinstance(event, str) else event
        rng = np.random.default_rng(seed)
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
            model = TyreModel.from_fit(fit, draws=idx, budget=GRIP_BUDGET_S)
            source = f"sealed fit {post.name} ({fit.prior_label}, {fit.n_laps} practice laps)"
        else:
            model = cls.prior_model(ev, n_draws, rng)
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
        return cls(event=ev, model=model, m_prior=m_prior, pit_loss_s=pit,
                   pit_loss_source=pit_src, allocation=alloc, stint_cap=caps, history=hist,
                   source=source, sealed_file=sealed, n_draws=n)

    @staticmethod
    def prior_model(ev: Event, n: int, rng: np.random.Generator) -> TyreModel:
        """A tyre model from the compound ladder alone, for a weekend with no fit.

        Wide on purpose: the MEDIUM rate is LogNormal around 0.10 s/lap with a
        factor-2 spread, which spans every 2026 weekend measured so far
        (0.07-0.30 s/lap at full push).
        """
        from src.config import COMPOUND_DEG_RATIO, COMPOUND_PACE_STEP_FRAC

        comps = list(VALID_COMPOUNDS)
        rank = dict(zip(comps, hardness_rank(comps)))
        med = np.exp(rng.normal(np.log(0.10), 0.5, size=n))
        ratio = np.exp(rng.normal(np.log(COMPOUND_DEG_RATIO), 0.25, size=n))
        step = COMPOUND_PACE_STEP_FRAC * ev.t_lap_ref_s
        wear, pace = {}, {}
        for c in comps:
            r = rank[c] - rank["MEDIUM"]
            rate = med * ratio ** (-r)
            wear[c] = rate / GRIP_BUDGET_S
            pace[c] = np.full(n, step * rank[c]) + rng.normal(0, 0.05, size=n)
        return TyreModel(compounds=comps, wear_rate=wear, pace_offset=pace,
                         budget=GRIP_BUDGET_S, n_draws=n, source="compound-ladder prior")


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


class RaceEngine:
    """Decisions for a race, recomputed every tick from the live state."""

    def __init__(self, wm: WeekendModel, *, seed: int = 0):
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

    # -- helpers ------------------------------------------------------------

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
            if st is None or st.get("compound") not in rate_draws:
                dt.compound = (st or {}).get("compound")
                dt.weights = None
                continue
            if dt.stint_index != st["index"]:
                dt.stint_index = st["index"]
                dt.compound = st["compound"]
                dt.stint_first_lap = st.get("first_lap")
                dt.weights = None
            g = laps[(laps["driver_number"] == num) & (laps["is_complete"])]
            g = g[g["lap_number"] >= (dt.stint_first_lap or 1)] if dt.stint_first_lap else g.iloc[0:0]
            c = dt.compound
            rate = rate_draws[c]                      # (n,)
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
                D = grip_loss(w_mid, budget=self.model.budget)            # (n, L)
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
        # Field posterior: every car's laps, tempered.  Driver posterior: own
        # laps at full weight plus the rest of the field tempered, which is a
        # cheap stand-in for a hierarchical model and keeps one coherent set of
        # draws per car for both the tyre it is on and the ones it will fit.
        ll_field = np.zeros(self.n)
        for ll in ll_by_driver.values():
            ll_field += FIELD_TEMPER * ll
        # A particle approximation degenerates once a few hundred laps of
        # likelihood pile onto one draw.  Temper the field's evidence so the
        # effective sample size stays above a floor: the posterior is then
        # honestly wider than the raw likelihood would make it, which is the
        # right side to err on for a model that ignores driver-to-driver
        # differences in management.
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
            rate = rate_draws[c]
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
            loss_now = grip_loss(wear_now, budget=self.model.budget)
            dt.proj = [float(((grip_loss(wear_now + per_lap * j, budget=self.model.budget) - loss_now) * w).sum())
                       for j in range(1, 6)]
            dt.deg_now_s_per_lap = float(((grip_loss(wear_now + per_lap, budget=self.model.budget)
                                           - loss_now) * w).sum())

    def _resample(self, weights: np.ndarray | None) -> np.ndarray:
        """Draw indices proportional to weights (systematic resampling)."""
        if weights is None:
            return np.arange(self.n)
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        return np.searchsorted(np.cumsum(weights), pos).clip(0, self.n - 1)

    # -- costs --------------------------------------------------------------

    def _future_cost_tables(self, total: int, idx: np.ndarray) -> dict:
        """cost[c][d, s, L] for fresh stints, on the field-weighted draws `idx`.

        Cached per tick on the resampled index set: the same table serves every
        driver's plan search.
        """
        key = (self.tick_no, total, int(idx[:8].sum()), int(idx[-8:].sum()), len(idx))
        if self._cost_cache.get("key") == key:
            return self._cost_cache["tables"]
        scaled = TyreModel(
            compounds=list(self.model.compounds),
            wear_rate={c: self.model.wear_rate[c][idx] * self.m_prior[idx] for c in self.model.compounds},
            pace_offset={c: self.model.pace_offset[c][idx] for c in self.model.compounds},
            budget=self.model.budget, load_exponent=self.model.load_exponent, n_draws=len(idx))
        ev = self.ev
        if total != ev.n_race_laps:
            from dataclasses import replace
            ev = replace(ev, n_race_laps=int(total))
        tables = scaled.cost_table(ev, int(total), 1.0, warmup_s=OUT_LAP_PENALTY_S)
        self._cost_cache = {"key": key, "tables": tables, "idx": idx}
        return tables

    def _continue_cost(self, dt: DriverTyre, idx: np.ndarray, cur_lap: int, n_more: int,
                       total: int) -> np.ndarray:
        """(len(idx), n_more+1): cumulative cost of running this tyre k more laps."""
        c = dt.compound
        rate = (self.model.wear_rate[c] * self.m_prior)[idx]
        pace = self.model.pace_offset[c][idx]
        w0 = dt.wear[idx] if dt.wear is not None else np.zeros(len(idx))
        laps = np.arange(cur_lap, cur_lap + max(n_more, 0)) + 1
        loads = self._load_vec(laps) if n_more > 0 else np.zeros(0)
        inc = rate[:, None] * loads[None, :]
        w_end = w0[:, None] + np.cumsum(inc, axis=1)
        loss = grip_loss(w_end - 0.5 * inc, budget=self.model.budget) + pace[:, None]
        out = np.zeros((len(idx), n_more + 1))
        out[:, 1:] = np.cumsum(loss, axis=1)
        return out

    # -- plans --------------------------------------------------------------

    def _plan(self, state: LiveState, num: str, dt: DriverTyre, tr, total: int,
              sc_active: bool) -> dict | None:
        cur = tr.current
        if cur is None or dt.compound not in self.model.compounds:
            return None
        cur_lap = int(cur["lap_number"]) - 1          # laps completed
        R = total - cur_lap
        if R <= 0:
            return None
        # One coherent set of draws per car, for the tyre it is on and the
        # ones it will fit: its own posterior (which already carries the field's
        # laps, tempered).
        idx = self._resample(dt.weights if dt.weights is not None else self.field_weights)
        idx_d = idx
        tables = self._future_cost_tables(total, idx)
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
        cont = self._continue_cost(dt, idx_d, cur_lap, R, total)      # (n, R+1)
        n = len(idx)
        pit_now_factor = (SC_PIT_LOSS_FRACTION if state.track_status == "4" else
                          VSC_PIT_LOSS_FRACTION if state.track_status in ("6", "7") else 1.0)
        pit = self.pit_loss_s
        dens = self._density(total)

        options = []   # (label, times(n,), kind, next_stop_lap, compounds)
        caps = self.wm.stint_cap
        stint_len_now = cur_lap - int(dt.stint_first_lap or 1) + 1      # laps run on this set so far
        BIG = 10 ** 6

        def cap_ok(c: str, L: int) -> bool:
            return L <= caps.get(c, BIG)

        over_cap = stint_len_now > caps.get(dt.compound, BIG) + 1

        def cont_ok(k: int) -> bool:   # continuing this set k more laps
            if over_cap:               # already past what this circuit has ever supported: stop within 2 laps
                return k <= 2
            return (stint_len_now + k) <= caps.get(dt.compound, BIG) + 1
        # 0 stops
        legal0 = (len(compounds_used) >= 2 or not state.is_race) and cont_ok(R)
        t0 = cont[:, R].copy()
        options.append({"label": "stay out", "kind": 0, "stop": None, "compounds": [], "times": t0,
                        "legal": legal0})
        # 1 stop with lap p as the in-lap (p = cur_lap + 1 is "box at the end
        # of this lap"); the car reaches the pit having completed p laps.
        now_lap = cur_lap + 1
        for p in range(now_lap, total - margin + 1):
            k = p - cur_lap
            if k > R - margin:
                break
            rem = total - p
            for c2 in avail:
                legal = not (c2 == dt.compound and len(compounds_used) < 2) and cont_ok(k) and cap_ok(c2, rem)
                loss = pit * (pit_now_factor if p == now_lap else 1.0)
                t = cont[:, k] + loss + TRAFFIC_LAPS_PER_STOP * DIRTY_AIR_S_PER_LAP * dens[min(p, total) - 1] \
                    + tables[c2][:, p, rem]
                options.append({"label": f"1 stop: {c2} on lap {p}", "kind": 1, "stop": p,
                                "compounds": [c2], "times": t, "legal": legal})
        # 2 stops, coarse grid
        step = 2 if R > 30 else 1
        for p1 in range(now_lap, total - 2 * margin + 1, step):
            k1 = p1 - cur_lap
            for p2 in range(p1 + margin, total - margin + 1, step):
                for c2, c3 in itertools.product(avail, repeat=2):
                    if c2 == c3 and used.get(c2, 0) + 2 > int(self.wm.allocation.get(c2, MAX_STINTS_PER_COMPOUND)):
                        continue
                    if not (cont_ok(k1) and cap_ok(c2, p2 - p1) and cap_ok(c3, total - p2)):
                        continue
                    loss1 = pit * (pit_now_factor if p1 == now_lap else 1.0)
                    t = (cont[:, k1] + loss1 + pit
                         + TRAFFIC_LAPS_PER_STOP * DIRTY_AIR_S_PER_LAP * (dens[p1 - 1] + dens[p2 - 1])
                         + tables[c2][:, p1, p2 - p1] + tables[c3][:, p2, total - p2])
                    options.append({"label": f"2 stops: {c2} lap {p1}, {c3} lap {p2}", "kind": 2,
                                    "stop": p1, "compounds": [c2, c3], "times": t, "legal": True})
        legal = [o for o in options if o["legal"]]
        if not legal:
            legal = options
        means = np.array([o["times"].mean() for o in legal])
        best_i = int(np.argmin(means))
        best = legal[best_i]
        T = np.stack([o["times"] for o in legal])                      # (n_opt, n)
        win = np.bincount(np.argmin(T, axis=0), minlength=len(legal)) / n
        # next-stop window: best cost per candidate stop lap, over 1-stop and 2-stop plans
        by_lap: dict = {}
        for o, mu in zip(legal, means):
            if o["stop"] is None:
                continue
            if o["stop"] not in by_lap or mu < by_lap[o["stop"]]:
                by_lap[o["stop"]] = mu
        window = []
        if by_lap:
            floor = min(by_lap.values())
            window = [{"lap": int(p), "loss_s": float(v - floor)} for p, v in sorted(by_lap.items())]
        in_win = [w["lap"] for w in window if w["loss_s"] <= WINDOW_TOL_S]
        cands = [(o, mu) for o, mu in zip(legal, means) if o["stop"] == now_lap]
        box_now = min(cands, key=lambda t: t[1])[0] if cands else None
        stay = legal[0] if legal[0]["kind"] == 0 else None
        delta_box_now = float(box_now["times"].mean() - best["times"].mean()) if box_now else float("nan")
        delta_stay = float(stay["times"].mean() - best["times"].mean()) if stay else float("nan")
        # p(best beats stay out) draw by draw
        p_best_vs_stay = float(((best["times"] - stay["times"]) < 0).mean()) if stay and stay is not best else float("nan")
        top = sorted(zip(legal, means, win), key=lambda t: t[1])[:6]
        return {
            "best": best["label"], "best_kind": best["kind"], "next_stop": best["stop"],
            "next_compound": (best["compounds"][0] if best["compounds"] else None),
            "compounds": best["compounds"], "win_prob": float(win[best_i]),
            "window_lo": (min(in_win) if in_win else None), "window_hi": (max(in_win) if in_win else None),
            "window": window[:60],
            "delta_box_now_s": delta_box_now, "delta_stay_out_s": delta_stay,
            "p_best_vs_stay": p_best_vs_stay,
            "stay_out_legal": bool(legal0), "compounds_used": sorted(compounds_used),
            "available": avail, "pit_now_factor": pit_now_factor,
            "options": [{"label": o["label"], "delta_s": float(mu - means[best_i]),
                         "win_prob": float(w)} for o, mu, w in top],
            "laps_remaining": int(R), "n_options": len(legal), "now_lap": int(now_lap),
            "stint_cap": caps.get(dt.compound), "stint_len_now": int(stint_len_now),
            "box_now_label": (box_now["label"] if box_now else None),
        }

    def _density(self, total: int) -> np.ndarray:
        f = np.arange(1, total + 1, dtype=float) / total
        q = 0.692 - 1.037 * f + 0.826 * f ** 2
        return q / 0.4488

    # -- undercut ----------------------------------------------------------

    def _undercut(self, state: LiveState, order: list, num: str, dt: DriverTyre) -> dict:
        """Threat from behind and opportunity on the car ahead.

        If the attacker pits now and the defender stays out `k` more laps
        before its own stop, the attacker gains, over those laps, the pace the
        defender's tyre keeps losing plus the compound difference, less what
        its own fresh tyre loses and the cold first lap.  The undercut works if
        that cumulative gain exceeds the gap.  Reported per lap of exposure,
        as a probability over the driver's posterior draws.
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
            loss0 = grip_loss(w0, budget=self.model.budget)
            ks = np.arange(1, K + 1)
            def_loss = grip_loss(w0[:, None] + rate_d[:, None] * ks[None, :], budget=self.model.budget) - loss0[:, None]
            att_loss = grip_loss(rate_n[:, None] * ks[None, :], budget=self.model.budget)
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
                # lapped car: behind everyone with a time gap
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
        for r in order:
            num = r["driver_number"]
            tr = state.tracks[num]
            dt = self.tyres.get(num)
            row = dict(r)
            row.update({"m_mean": float("nan"), "m_lo": float("nan"), "m_hi": float("nan"),
                        "wear": float("nan"), "p_past_cliff": float("nan"),
                        "laps_to_cliff_p10": float("nan"), "laps_to_cliff_p50": float("nan"),
                        "laps_to_cliff_p90": float("nan"), "deg_now_s_per_lap": float("nan"),
                        "level_s": float("nan"), "n_clean": 0, "proj": [], "cliff_alarm": False})
            if dt is not None and dt.wear is not None:
                w = dt.weights if dt.weights is not None else np.ones(self.n) / self.n
                row.update({"m_mean": dt.m_mean, "m_lo": dt.m_lo, "m_hi": dt.m_hi,
                            "wear": float((dt.wear * w).sum()), "p_past_cliff": dt.p_past_cliff,
                            "laps_to_cliff_p10": dt.laps_to_cliff[0],
                            "laps_to_cliff_p50": dt.laps_to_cliff[1],
                            "laps_to_cliff_p90": dt.laps_to_cliff[2],
                            "deg_now_s_per_lap": dt.deg_now_s_per_lap, "level_s": dt.level_s,
                            "n_clean": int(dt.n_clean), "proj": dt.proj,
                            "cliff_alarm": bool(dt.p_past_cliff >= CLIFF_ALARM_P
                                                or dt.laps_to_cliff[1] <= 2.0)})
            lap_no = int(tr.current["lap_number"]) if tr.current else 0
            # plans: recompute on a new lap (or when asked), reuse otherwise
            if dt is not None and dt.wear is not None and not tr.retired and state.is_race \
                    and state.session_status not in ("Finished", "Finalised", "Ends"):
                if replan and (self._last_plan_lap.get(num) != lap_no or sc):
                    try:
                        self._plans[num] = self._plan(state, num, dt, tr, total, sc)
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
                    # -- alerts ------------------------------------------------
                    if row["cliff_alarm"]:
                        self._alert(state, num, "cliff", "bad",
                                    f"CLIFF: P(past grip budget) {row['p_past_cliff']:.0%}, "
                                    f"~{row['laps_to_cliff_p50']:.0f} laps left on the {dt.compound}", lap_no)
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
            field.append(row)
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
                     "sc_active": sc, "tick": self.tick_no,
                     "tick_utc": datetime.now(timezone.utc).isoformat()},
            "field": field,
            "alerts": list(self.alerts[-60:]),
            "compounds": {c: {"life_full_push": float(self.model.life_laps(c, 1.0).mean()),
                              "pace_offset_s": float(self.model.pace_offset[c].mean())}
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
            # traffic from lap start times, as the offline cascade does
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
            # pooled per compound with stint fixed effects
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
        prior = {c: {"rate_s_per_lap": float((self.model.wear_rate[c] * self.model.budget).mean()),
                     "lo": float(np.quantile(self.model.wear_rate[c] * self.model.budget, 0.05)),
                     "hi": float(np.quantile(self.model.wear_rate[c] * self.model.budget, 0.95))}
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
