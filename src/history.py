"""What this circuit has done before: previous races as a prior.

A practice fit knows the tyre's *rate* of degradation on this weekend's track,
and only over the ages practice ran.  It does not know how long a tyre is
actually run here, which is set by things practice cannot see — thermal
degradation on a low-wear circuit, graining, the pit-lane length, the safety
car rate, and what every team learned the last three Septembers.  Monza is the
canonical case: practice shows almost no degradation, so a rate-based life
model derives a tyre good for hundreds of laps, while in three real races
nobody has run the soft for more than nine laps.

This module turns the circuit's previous races (FastF1, 2023-2025) into two
things the model consumes:

1. **A prior on the race-regime degradation rate per compound**, measured with
   the same driver + race-lap fixed-effects estimator `src.regime` uses (so it
   is evolution- and fuel-corrected), pooled across years, and scaled by a
   **season factor** — the 2026/2025 ratio of the same estimator on the circuits
   that have a race in both seasons.  2026 is a new car and a new tyre; last
   year's number is transferred, not copied.
2. **Stint-length caps per compound** — the longest stint the compound has been
   run to here, scaled to this year's race distance with a small margin.  Not a
   model output: a fact about the circuit, and the hard bound the optimiser and
   the live engine respect.

Everything is cached under `data/processed/history/` so a weekend build does
not re-read fifty races.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, EVENTS, FASTF1_CACHE, VALID_COMPOUNDS, Event, get_event

log = logging.getLogger("degless.history")

HIST_DIR = DATA_PROCESSED / "history"
YEARS = (2023, 2024, 2025)
# Fuel physics of the previous regulations, for the evolution/fuel split of old races.
FUEL_S_PER_LAP_PRE2026 = 1.67 * 0.033
MIN_STINT = 3
CAP_MARGIN = 1.10          # stint cap = historical max x margin, scaled to race length
RATE_PRIOR_LN_SD_FLOOR = 0.35
THERMAL_BETA_DEFAULT = 0.025   # d log(deg) / d track temp, per °C, measured within circuit x compound 2023-2025
MISSING_RETRY_S = 7 * 24 * 3600   # how long a 'no such race' marker is believed
SEASON_PRIOR_LN_SD_FLOOR = 0.45   # a circuit nobody has raced: at least this wide


# --------------------------------------------------------------------------
# One race
# --------------------------------------------------------------------------


def _fastf1():
    import fastf1
    import logging as _l

    fastf1.Cache.enable_cache(str(FASTF1_CACHE))
    _l.getLogger("fastf1").setLevel(_l.ERROR)
    return fastf1


def race_deg_slopes(laps: pd.DataFrame, fuel_s_per_lap: float) -> dict:
    """Per-compound degradation, s/lap, with driver and race-lap fixed effects.

    The lap effect absorbs track evolution and fuel burn alike; a tyre-age
    slope per compound is identified from the cross-section of cars at
    different ages on the same lap.  Returns {compound: {slope, se, n_laps}}.
    """
    d = laps.dropna(subset=["lap_time_s", "tyre_life", "compound"])
    d = d[d["compound"].isin(VALID_COMPOUNDS)]
    if len(d) < 60 or d["lap_number"].nunique() < 8:
        return {}
    comps = sorted(d["compound"].unique())
    drv = pd.get_dummies(d["driver"], drop_first=True).astype(float)
    lap = pd.get_dummies(d["lap_number"].astype(int), drop_first=True).astype(float)
    comp = pd.get_dummies(d["compound"]).astype(float)[comps]
    age = d["tyre_life"].to_numpy(dtype=float)
    ageX = np.column_stack([comp[c].to_numpy() * age for c in comps])
    X = np.column_stack([np.ones(len(d)), ageX, comp.to_numpy()[:, 1:], drv.to_numpy(), lap.to_numpy()])
    y = d["lap_time_s"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    try:
        cov = np.linalg.pinv(X.T @ X) * float(resid @ resid / dof)
        ses = np.sqrt(np.clip(np.diag(cov)[1:1 + len(comps)], 0, None))
    except Exception:
        ses = np.full(len(comps), np.nan)
    out = {}
    for i, c in enumerate(comps):
        n = int((d["compound"] == c).sum())
        if n >= 30:
            out[c] = {"slope": float(beta[1 + i]), "se": float(ses[i]), "n_laps": n}
    return out


# Circuits whose FastF1 location or event name shares no word with the name
# the calendar uses for them.
CIRCUIT_ALIASES = {
    "monte carlo": ["monaco"], "interlagos": ["sao paulo", "brazil"], "kuala lumpur": ["sepang", "malaysia"],
    "yas marina circuit": ["yas island", "abu dhabi"], "madring": ["madrid"], "singapore": ["marina bay"],
    "spielberg": ["red bull ring", "austria"], "montreal": ["gilles villeneuve", "canada"],
    "mexico city": ["hermanos rodriguez", "mexico"], "lusail": ["losail", "qatar"], "austin": ["cota", "americas"],
}


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    return " ".join(w for w in s.lower().replace("-", " ").split() if w not in ("grand", "prix", "circuit", "de", "the"))


def same_circuit(circuit: str, event) -> bool:
    """Does a FastF1 event row describe `circuit`?  Word overlap between the
    calendar's name (plus its aliases) and the event's location, name and
    country, with accents stripped."""
    want = {_norm(circuit)} | {_norm(a) for a in CIRCUIT_ALIASES.get(_norm(circuit), [])}
    want_tokens = set()
    for w in want:
        want_tokens |= set(w.split())
    have = " ".join(_norm(event.get(k, "")) for k in ("Location", "EventName", "OfficialEventName", "Country"))
    have_tokens = set(have.split())
    if any(w and w in have for w in want):
        return True
    return bool(want_tokens & have_tokens)


def _canonical(session) -> pd.DataFrame:
    laps = session.laps.copy()
    out = pd.DataFrame({
        "driver": laps["Driver"].astype(str),
        "lap_number": laps["LapNumber"].astype(float),
        "stint": laps["Stint"].astype(float),
        "compound": laps["Compound"].astype(str).str.upper(),
        "tyre_life": laps["TyreLife"].astype(float),
        "lap_time_s": laps["LapTime"].dt.total_seconds(),
        "is_accurate": laps["IsAccurate"].astype(bool),
        "pit_in": laps["PitInTime"].notna(),
        "pit_out": laps["PitOutTime"].notna(),
        "track_status": laps["TrackStatus"].astype(str),
    })
    return out


def summarise_race(year: int, circuit: str, *, force: bool = False) -> dict | None:
    """Everything the priors need from one race, cached as JSON."""
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{year}_{circuit.lower().replace(' ', '-')}"
    p = HIST_DIR / f"{key}.json"
    if p.exists() and not force:
        d = json.loads(p.read_text())
        if d.get("missing"):
            # A circuit with no race that year (Madring in 2023, say).  The
            # marker expires so a transient network failure cannot become a
            # permanent "no history"; the outlook re-asks once a week.
            import time as _t
            if _t.time() - float(d.get("checked", 0)) < MISSING_RETRY_S:
                return None
        else:
            return d
    ff1 = _fastf1()
    import time as _t
    try:
        s = ff1.get_session(year, circuit, "Race")
        if not same_circuit(circuit, s.event):
            # FastF1 matches event names fuzzily and will hand back *some*
            # race for a name it does not know - "Madring" became the
            # Singapore Grand Prix.  A new circuit has no history; say so.
            raise LookupError(f"nearest {year} event is {s.event.get('EventName')} at "
                              f"{s.event.get('Location')}, not {circuit}")
        s.load(laps=True, telemetry=False, weather=True, messages=True)
    except Exception as exc:
        log.info("no %s race for %s: %s", year, circuit, str(exc)[:80])
        p.write_text(json.dumps({"year": year, "circuit": circuit, "missing": True,
                                 "checked": _t.time(), "error": str(exc)[:160]}))
        return None
    laps = _canonical(s)
    n_laps = int(s.total_laps or laps["lap_number"].max())
    fin = laps.groupby("driver")["lap_number"].max()
    classified = fin[fin >= n_laps - 2].index
    st = (laps.groupby(["driver", "stint"])
          .agg(compound=("compound", "first"), n=("lap_number", "size"), start=("lap_number", "min"))
          .reset_index())
    st = st[st["driver"].isin(classified) & (st["n"] >= MIN_STINT) & st["compound"].isin(VALID_COMPOUNDS)]
    stops = (st.groupby("driver").size() - 1)
    plans = (st.sort_values(["driver", "start"]).groupby("driver")["compound"]
             .apply(lambda x: "-".join(v[0] for v in x)))
    per_comp = {}
    for c, g in st.groupby("compound"):
        per_comp[c] = {"n_stints": int(len(g)), "p10": float(g["n"].quantile(0.1)), "p50": float(g["n"].median()),
                       "p90": float(g["n"].quantile(0.9)), "max": int(g["n"].max()),
                       "share_of_laps": float(g["n"].sum() / st["n"].sum())}
    clean = laps[laps["is_accurate"] & ~laps["pit_in"] & ~laps["pit_out"] & (laps["track_status"] == "1")]
    deg = race_deg_slopes(clean, FUEL_S_PER_LAP_PRE2026 if year < 2026 else get_event_fuel(circuit))
    # pit loss: (in + out) - 2 x nearby clean median, green flag both laps
    rows = []
    for drv, g in laps.groupby("driver"):
        g = g.sort_values("lap_number")
        cl = g[g["is_accurate"]]
        for _, r in g[g["pit_in"]].iterrows():
            nxt = g[g["lap_number"] == r["lap_number"] + 1]
            if nxt.empty or not bool(nxt.iloc[0]["pit_out"]) or r["track_status"] != "1" or nxt.iloc[0]["track_status"] != "1":
                continue
            near = cl[(cl["lap_number"] >= r["lap_number"] - 5) & (cl["lap_number"] <= r["lap_number"] + 6)]
            if len(near) < 3 or not (np.isfinite(r["lap_time_s"]) and np.isfinite(nxt.iloc[0]["lap_time_s"])):
                continue
            loss = r["lap_time_s"] + nxt.iloc[0]["lap_time_s"] - 2 * near["lap_time_s"].median()
            if 5 < loss < 60:
                rows.append(loss)
    sc_laps = float((laps["track_status"].isin(["4", "6", "7"])).groupby(laps["lap_number"]).any().mean()) if len(laps) else 0.0
    weather = s.weather_data
    out = {
        "year": year, "circuit": circuit, "event": str(s.event["EventName"]), "location": str(s.event["Location"]),
        "date": str(s.event["EventDate"])[:10], "n_laps": n_laps, "n_classified": int(len(classified)),
        "stops": {int(k): int(v) for k, v in stops.value_counts().sort_index().items()},
        "plans": {k: int(v) for k, v in plans.value_counts().head(6).items()},
        "compounds": per_comp, "deg": deg,
        "pit_loss_s": (float(np.median(rows)) if len(rows) >= 2 else None), "n_pit_stops": len(rows),
        "sc_share_of_laps": sc_laps,
        "track_temp_c": (float(weather["TrackTemp"].median()) if weather is not None and len(weather) else None),
        "rain": bool(weather["Rainfall"].any()) if weather is not None and len(weather) else False,
    }
    p.write_text(json.dumps(out, indent=1))
    return out


def get_event_fuel(circuit: str) -> float:
    ev = next((e for e in EVENTS.values() if e.circuit.lower() == circuit.lower()), None)
    return ev.fuel_effect_s_per_lap if ev else 0.031


# --------------------------------------------------------------------------
# Season factor: 2026 vs 2025 on the circuits that have both
# --------------------------------------------------------------------------


def season_factor(*, force: bool = False) -> dict:
    """Geometric-mean ratio of 2026 to 2025 race degradation on shared circuits.

    The 2026 side comes from the weekends already run this season (their race
    lap tables are on disk once the pipeline has scored them); the 2025 side
    from the archive.  Laps-weighted per compound, then pooled.
    """
    p = HIST_DIR / "season_factor.json"
    if p.exists() and not force:
        d = json.loads(p.read_text())
        if d.get("n_circuits", 0) >= 3:
            return d
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    ratios, detail = [], []
    for k, ev in EVENTS.items():
        rp = DATA_PROCESSED / f"laps_{k}_race.parquet"
        if not rp.exists() or not ev.donor_ok:
            continue
        r26 = pd.read_parquet(rp)
        r26 = r26[r26["is_accurate"] & ~r26["pit_in"] & ~r26["pit_out"] & (r26["track_status"].astype(str) == "1")]
        d26 = race_deg_slopes(r26.rename(columns={"tyre_life": "tyre_life"}), ev.fuel_effect_s_per_lap)
        h25 = summarise_race(2025, ev.circuit)
        if not d26 or not h25 or not h25.get("deg"):
            continue
        num = den = 0.0
        for c in d26:
            if c in h25["deg"] and h25["deg"][c]["slope"] > 0.01 and d26[c]["slope"] > 0.005:
                w = min(d26[c]["n_laps"], h25["deg"][c]["n_laps"])
                num += w * d26[c]["slope"]
                den += w * h25["deg"][c]["slope"]
        if den > 0 and num > 0:
            ratios.append(num / den)
            detail.append({"circuit": ev.circuit, "ratio": round(num / den, 3),
                           "deg_2026": {c: round(v["slope"], 3) for c, v in d26.items()},
                           "deg_2025": {c: round(v["slope"], 3) for c, v in h25["deg"].items()}})
    if not ratios:
        out = {"factor": 1.0, "ln_sd": 0.5, "n_circuits": 0, "detail": [], "note": "no shared circuit yet; 1.0 assumed"}
    else:
        logs = np.log(ratios)
        out = {"factor": float(np.exp(logs.mean())), "ln_sd": float(max(logs.std(ddof=1) if len(logs) > 1 else 0.4, 0.25)),
               "n_circuits": len(ratios), "detail": detail,
               "note": "2026/2025 race degradation, driver + lap fixed effects, laps-weighted per compound"}
    p.write_text(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------
# The prior for a weekend
# --------------------------------------------------------------------------


def thermal_sensitivity() -> dict:
    """How much faster a tyre degrades on a hotter track, from the archive.

    Fitted once from every cached race (`data/processed/history/thermal.json`):
    the within-circuit, within-compound slope of log race degradation on the
    median track temperature.  +0.025 per °C — a 10 °C hotter race day means
    ~30% more degradation — with a standard error a quarter of that.
    """
    p = HIST_DIR / "thermal.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"beta_per_c": THERMAL_BETA_DEFAULT, "se": 0.007, "n": 0, "note": "default"}


def practice_track_temp(ev: Event) -> float | None:
    """Median track temperature of the practice session run closest to race
    time of day (FP2 on a conventional weekend), from the FastF1 cache."""
    ff1 = _fastf1()
    for name in ("Practice 2", "Practice 3", "Practice 1"):
        if name not in ev.practice_sessions:
            continue
        try:
            s = ff1.get_session(ev.ff1_year, ev.ff1_round, name)
            s.load(laps=False, telemetry=False, weather=True, messages=False)
            w = s.weather_data
            if w is not None and len(w) and w["TrackTemp"].notna().any():
                return float(w["TrackTemp"].median())
        except Exception:
            continue
    return None


@dataclass
class CircuitPrior:
    event: str
    circuit: str
    years: list = field(default_factory=list)
    races: list = field(default_factory=list)            # the per-race summaries
    rate_prior: dict = field(default_factory=dict)       # compound -> {mean_s_per_lap, ln_sd, source}
    stint_cap: dict = field(default_factory=dict)        # compound -> max laps this year
    stint_typical: dict = field(default_factory=dict)    # compound -> {p10,p50,p90} scaled
    stops: dict = field(default_factory=dict)            # pooled distribution over classified finishers
    plans: dict = field(default_factory=dict)
    pit_loss_s: float | None = None
    sc_share: float = 0.0
    season: dict = field(default_factory=dict)
    thermal: dict = field(default_factory=dict)          # {beta_per_c, track_temp_now, track_temp_hist, multiplier}
    soft_race_tyre: bool = True                          # was the SOFT run for real stints here?

    @property
    def available(self) -> bool:
        return bool(self.races)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "races"} | {
            "races": [{k: r[k] for k in ("year", "event", "n_laps", "stops", "plans", "compounds", "deg",
                                          "pit_loss_s", "sc_share_of_laps", "track_temp_c")} for r in self.races]}


def circuit_prior(event: Event | str, years=YEARS, *, track_temp_c: float | None = None,
                  probe_practice_temp: bool = True) -> CircuitPrior:
    """`probe_practice_temp=False` skips the FastF1 lookup of this weekend's
    practice temperature (which does not exist before the weekend starts and
    is slow to discover that); the thermal multiplier is then 1.0 unless a
    temperature is passed in."""
    ev = get_event(event) if isinstance(event, str) else event
    races = [r for r in (summarise_race(y, ev.circuit) for y in years) if r and not r.get("rain")]
    cp = CircuitPrior(event=ev.key, circuit=ev.circuit, years=[r["year"] for r in races], races=races)
    if not races:
        return cp
    sf = season_factor()
    cp.season = sf
    # -- thermal proxy: this weekend's track temperature against the archive's
    th = thermal_sensitivity()
    temps = [r["track_temp_c"] for r in races if r.get("track_temp_c") is not None]
    t_now = track_temp_c if track_temp_c is not None else (practice_track_temp(ev) if probe_practice_temp else None)
    mult = 1.0
    if temps and t_now is not None:
        mult = float(np.exp(th["beta_per_c"] * (t_now - float(np.mean(temps)))))
        mult = float(np.clip(mult, 0.6, 1.7))
    cp.thermal = {"beta_per_c": th["beta_per_c"], "track_temp_now": t_now,
                  "track_temp_hist": (float(np.mean(temps)) if temps else None), "multiplier": mult}
    # -- degradation rate prior (race regime), pooled over years -----------
    for c in VALID_COMPOUNDS:
        vals, ws = [], []
        for r in races:
            d = r.get("deg", {}).get(c)
            if d and d["slope"] > 0.003:
                vals.append(np.log(d["slope"]))
                ws.append(d["n_laps"])
        if not vals:
            continue
        vals, ws = np.array(vals), np.array(ws, float)
        mu = float(np.average(vals, weights=ws))
        spread = float(np.sqrt(np.average((vals - mu) ** 2, weights=ws))) if len(vals) > 1 else 0.3
        ln_sd = float(np.sqrt(max(spread, RATE_PRIOR_LN_SD_FLOOR) ** 2 + sf.get("ln_sd", 0.4) ** 2))
        cp.rate_prior[c] = {"mean_s_per_lap": float(np.exp(mu) * sf.get("factor", 1.0) * mult), "ln_sd": ln_sd,
                            "years": [r["year"] for r in races if r.get("deg", {}).get(c)],
                            "raw_mean_s_per_lap": float(np.exp(mu)), "thermal_multiplier": mult}
    # -- stint caps and typical lengths, scaled to this year's distance ------
    for c in VALID_COMPOUNDS:
        mx, p10, p50, p90, n = [], [], [], [], 0
        for r in races:
            s = r.get("compounds", {}).get(c)
            if not s:
                continue
            scale = ev.n_race_laps / max(r["n_laps"], 1)
            mx.append(s["max"] * scale)
            p10.append(s["p10"] * scale); p50.append(s["p50"] * scale); p90.append(s["p90"] * scale)
            n += s["n_stints"]
        if mx:
            cp.stint_cap[c] = int(round(max(mx) * CAP_MARGIN))
            cp.stint_typical[c] = {"p10": float(np.mean(p10)), "p50": float(np.mean(p50)),
                                   "p90": float(np.mean(p90)), "n_stints": n}
    soft = cp.stint_typical.get("SOFT")
    cp.soft_race_tyre = bool(soft and soft["n_stints"] >= 6 and soft["p50"] >= 8)
    # -- the field's revealed strategy -----------------------------------
    stops: dict = {}
    for r in races:
        for k, v in r.get("stops", {}).items():
            stops[int(k)] = stops.get(int(k), 0) + int(v)
    cp.stops = dict(sorted(stops.items()))
    plans: dict = {}
    for r in races:
        for k, v in r.get("plans", {}).items():
            plans[k] = plans.get(k, 0) + int(v)
    cp.plans = dict(sorted(plans.items(), key=lambda t: -t[1])[:6])
    pls = [r["pit_loss_s"] for r in races if r.get("pit_loss_s")]
    cp.pit_loss_s = float(np.median(pls)) if pls else None
    cp.sc_share = float(np.mean([r.get("sc_share_of_laps", 0.0) for r in races]))
    return cp


# --------------------------------------------------------------------------
# Folding the prior into a fit
# --------------------------------------------------------------------------


def _combine_lognormal(lr: np.ndarray, mu_h: float, sd_h: float, lr_ratio: float,
                       regime_ln_sd: float, rng: np.random.Generator) -> tuple:
    """Precision-weighted combination of log-rate draws with a log-normal prior.

    `lr` are log rates in the *model's* regime; the prior `(mu_h, sd_h)` is in
    the race regime, `lr_ratio` (log of the practice->race factor) links the
    two and `regime_ln_sd` is its uncertainty - zero when both are already in
    the same regime.  Returns the rescaled draws (rank-preserving, so the
    correlation with everything else survives) and a row of what moved.
    """
    lr = np.asarray(lr, dtype=float)
    mu_p, sd_p = float(lr.mean()), float(max(lr.std(), 0.05))
    mu_p_race = mu_p + lr_ratio
    sd_p_race = float(np.sqrt(sd_p ** 2 + regime_ln_sd ** 2))
    prec = 1 / sd_p_race ** 2 + 1 / sd_h ** 2
    mu_c_race = (mu_p_race / sd_p_race ** 2 + mu_h / sd_h ** 2) / prec
    sd_c_race = float(np.sqrt(1 / prec))
    mu_c = mu_c_race - lr_ratio
    sd_c = float(np.sqrt(max(sd_c_race ** 2 - regime_ln_sd ** 2, 0.15 ** 2)))
    order = np.argsort(lr)
    target = np.sort(rng.normal(mu_c, sd_c, size=len(lr)))
    new_lr = np.empty_like(lr)
    new_lr[order] = target
    row = {"practice": float(np.exp(mu_p)), "practice_as_race": float(np.exp(mu_p_race)),
           "practice_sd_ln": sd_p_race, "history_race": float(np.exp(mu_h)), "history_sd_ln": sd_h,
           "combined_race": float(np.exp(mu_c_race)), "combined": float(np.exp(mu_c)),
           "combined_sd_ln": sd_c, "weight_on_history": float((1 / sd_h ** 2) / prec)}
    return new_lr, row


def season_prior(*, force: bool = False) -> dict:
    """A degradation prior for a circuit nobody has raced: the 2026 season so far.

    Pooled per compound over every 2026 race the season-factor machinery has
    measured (driver + race-lap fixed effects, so evolution- and
    fuel-corrected), on the log scale, with the between-circuit spread as the
    width - floored, because a new circuit is at least as uncertain as the
    spread between known ones.  Race regime, like the circuit prior.
    """
    sf = season_factor(force=force)
    vals: dict = {}
    for d in sf.get("detail", []):
        for c, v in (d.get("deg_2026") or {}).items():
            if v and v > 0.003:
                vals.setdefault(c, []).append(np.log(float(v)))
    out = {}
    for c, ls in vals.items():
        ls = np.array(ls)
        spread = float(ls.std(ddof=1)) if len(ls) > 1 else SEASON_PRIOR_LN_SD_FLOOR
        out[c] = {"mean_s_per_lap": float(np.exp(ls.mean())), "ln_sd": float(max(spread, SEASON_PRIOR_LN_SD_FLOOR)),
                  "n_circuits": int(len(ls)), "raw_mean_s_per_lap": float(np.exp(ls.mean()))}
    return {"rate_prior": out, "n_circuits": int(len(sf.get("detail", []))),
            "circuits": [d.get("circuit") for d in sf.get("detail", [])],
            "note": "2026 race degradation pooled over the circuits raced so far; race regime"}


def apply_rate_prior_to_model(model, ev: Event, regime, rate_prior: dict, *,
                              rng: np.random.Generator | None = None, seed: int = 0,
                              same_regime: bool = False, label: str = "history",
                              pooled: bool = False):
    """`apply_circuit_prior` for a `TyreModel` rather than a Bayes fit.

    Used by the outlook before any practice exists (the ladder prior combined
    with the circuit's or the season's race history) and while practice is
    running (the live long-run board folded in with `same_regime=True`, since
    a long run and the model are both in the practice regime).  Returns a new
    model and the table of what moved.

    `pooled=True` reads the history as one statement about the *circuit's
    severity* rather than three about the compounds: each compound's history
    is converted to what it implies for the reference compound through the
    model's own ladder, the implications are precision-pooled, and one common
    scale is applied to every compound.  The ladder ordering therefore cannot
    invert, which per-compound combination of two or three noisy races can do
    (and did).  It is the right reading when the model's only knowledge of the
    compounds is the ladder itself, i.e. before any practice has run.
    """
    from src.tyre import TyreModel

    rng = np.random.default_rng(seed) if rng is None else rng
    lr_ratio = 0.0 if same_regime else float(np.log(max(regime.ratio, 0.05)))
    rg_sd = 0.0 if same_regime else float(regime.ln_sd)
    wear = {c: model.wear_rate[c].copy() for c in model.compounds}
    rows, scales, pending = [], {}, []
    if pooled:
        present = [c for c in model.compounds if rate_prior.get(c)]
        if not present:
            return model, pd.DataFrame()
        ref = "MEDIUM" if "MEDIUM" in model.compounds else model.compounds[0]
        lr_ref = np.log(np.maximum(model.wear_rate[ref] * model.budget, 1e-4))
        mu_ref = float(lr_ref.mean())
        ys, ws = [], []
        for c in present:
            pr = rate_prior[c]
            mu_c = float(np.log(np.maximum(model.wear_rate[c] * model.budget, 1e-4)).mean())
            # what this compound's history says about the reference compound
            y = float(np.log(pr["mean_s_per_lap"])) - (mu_c - mu_ref)
            w = 1.0 / float(pr["ln_sd"]) ** 2
            ys.append(y); ws.append(w)
            rows.append({"compound": c, "history_race": float(pr["mean_s_per_lap"]), "history_sd_ln": float(pr["ln_sd"]),
                         "implied_reference_race": float(np.exp(y)), "source": label})
        ys, ws = np.array(ys), np.array(ws)
        mu_h = float(np.sum(ys * ws) / ws.sum())
        sd_h = float(np.sqrt(1.0 / ws.sum()))
        new_lr, row = _combine_lognormal(lr_ref, mu_h, sd_h, lr_ratio, rg_sd, rng)
        common = np.exp(new_lr - lr_ref)
        for c in model.compounds:
            wear[c] = model.wear_rate[c] * common
        for r in rows:
            r.update({"practice": row["practice"], "practice_as_race": row["practice_as_race"],
                      "combined_race": row["combined_race"], "combined": row["combined"],
                      "weight_on_history": row["weight_on_history"], "pooled": True, "reference": ref})
        new = TyreModel(compounds=list(model.compounds), wear_rate=wear,
                        pace_offset={c: model.pace_offset[c].copy() for c in model.compounds},
                        budget=model.budget, load_exponent=model.load_exponent, n_draws=model.n_draws,
                        source=f"{model.source} + {label} (pooled)")
        return new, pd.DataFrame(rows)
    for c in model.compounds:
        lr = np.log(np.maximum(model.wear_rate[c] * model.budget, 1e-4))
        pr = rate_prior.get(c)
        if not pr:
            pending.append(c)
            continue
        new_lr, row = _combine_lognormal(lr, float(np.log(pr["mean_s_per_lap"])), float(pr["ln_sd"]),
                                         lr_ratio, rg_sd, rng)
        scale = np.exp(new_lr - lr)
        scales[c] = scale
        wear[c] = model.wear_rate[c] * scale
        rows.append({"compound": c, **row, "source": label})
    if scales and pending:
        common = np.exp(np.mean([np.log(v) for v in scales.values()], axis=0))
        for c in pending:
            wear[c] = model.wear_rate[c] * common
            rows.append({"compound": c, "practice": float((model.wear_rate[c] * model.budget).mean()),
                         "history_race": None, "combined": float((wear[c] * model.budget).mean()),
                         "weight_on_history": None, "source": label,
                         "note": f"no {label} for this compound; moved with the ladder"})
    new = TyreModel(compounds=list(model.compounds), wear_rate=wear,
                    pace_offset={c: model.pace_offset[c].copy() for c in model.compounds},
                    budget=model.budget, load_exponent=model.load_exponent, n_draws=model.n_draws,
                    source=f"{model.source} + {label}")
    return new, pd.DataFrame(rows)


def apply_circuit_prior(fit, ev: Event, regime, cp: CircuitPrior, *, seed: int = 0):
    """Combine the practice posterior with the circuit's history, per compound.

    The history prior is on the *race-regime* rate; the practice fit is in the
    practice regime, and `regime.ratio` links the two.  On the log scale both
    are (approximately) normal, so the combination is the precision-weighted
    normal — the standard conjugate update — and each posterior draw's
    degradation curve is rescaled so the draws follow the combined law while
    keeping their ranks (the correlation with everything else is preserved).

    Returns a new fit object, plus a table of what moved.
    """
    from copy import deepcopy

    new = deepcopy(fit)
    rng = np.random.default_rng(seed)
    span = np.array([1.0, 10.0])
    rows = []
    scales: dict = {}       # compound -> per-draw multiplicative scale applied
    pending = []            # compounds with no history of their own
    for j, c in enumerate(fit.compounds):
        d = fit.deg_loss(c, span)
        rate = np.maximum((d[:, 1] - d[:, 0]) / 9.0, 1e-4)          # practice-regime rate per draw
        lr = np.log(rate)
        mu_p, sd_p = float(lr.mean()), float(max(lr.std(), 0.05))
        pr = cp.rate_prior.get(c)
        if not pr:
            pending.append((j, c, mu_p))
            continue
        # Compare in the *race* regime, where the decision lives.  The
        # practice->race transfer is what is uncertain about practice, so its
        # spread attaches to the practice side; history is already race trim.
        # The combination goes back to the practice regime for the draws; the
        # live engine and the optimiser add the regime uncertainty again
        # themselves, so it is taken out.  Rank-preserving rescale of the draws.
        new_lr, row = _combine_lognormal(lr, float(np.log(pr["mean_s_per_lap"])), float(pr["ln_sd"]),
                                         float(np.log(max(regime.ratio, 0.05))), float(regime.ln_sd), rng)
        scale = np.exp(new_lr - lr)
        scales[c] = scale
        new.posterior["lin"][:, j] = fit.posterior["lin"][:, j] * scale
        if "hinge" in new.posterior:
            new.posterior["hinge"][:, j] = fit.posterior["hinge"][:, j] * scale
        rows.append({"compound": c, **row})
    # A compound the circuit has no race history for (Monza's SOFT, say) moves
    # with the others: the compound ladder is a property of the tyre range, so
    # the history's correction to the track's severity applies to it too.  Its
    # own ordering relative to the compounds that did move is thereby kept.
    if scales and pending:
        common = np.exp(np.mean([np.log(v) for v in scales.values()], axis=0))
        for j, c, mu_p in pending:
            new.posterior["lin"][:, j] = fit.posterior["lin"][:, j] * common
            if "hinge" in new.posterior:
                new.posterior["hinge"][:, j] = fit.posterior["hinge"][:, j] * common
            rows.append({"compound": c, "practice": float(np.exp(mu_p)), "history_race": None,
                         "combined": float(np.exp(mu_p) * common.mean()),
                         "weight_on_history": None, "note": "no history for this compound here; moved with the ladder"})
    elif pending:
        for j, c, mu_p in pending:
            rows.append({"compound": c, "practice": float(np.exp(mu_p)), "history_race": None,
                         "combined": float(np.exp(mu_p))})
    new.prior_label = f"{fit.prior_label} + circuit history"
    return new, pd.DataFrame(rows)


def stint_caps_for(ev: Event, cp: CircuitPrior, base: dict | None = None) -> dict:
    """Per-compound stint cap: the tighter of the model's own and history's."""
    out = dict(base or {})
    for c, cap in cp.stint_cap.items():
        out[c] = int(min(out.get(c, 10 ** 6), cap))
    return out
