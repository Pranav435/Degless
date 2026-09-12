"""What this circuit has done before: previous races as a prior.

A practice fit knows the tyre's *rate* of degradation on this weekend's track,
and only over the ages practice ran.  It does not know how long a tyre is
actually run here, which is set by things practice cannot see — thermal
degradation on a low-wear circuit, graining, the pit-lane length, the safety
car rate, and what every team learned the last three Septembers.  Monza is the
canonical case: practice shows almost no degradation, so a rate-based life
model derives a tyre good for hundreds of laps, while in three real races
nobody has run the soft for more than nine laps.

This module turns the circuit's previous races (FastF1, 2023-2025) into what
the model consumes:

1. **A prior on the race-regime degradation rate per compound**, measured with
   the same driver + race-lap fixed-effects estimator `src.regime` uses (so it
   is evolution- and fuel-corrected), pooled across years, and scaled by a
   **season factor** — the 2026/2025 ratio of the same estimator on the circuits
   that have a race in both seasons.  2026 is a new car and a new tyre; last
   year's number is transferred, not copied.
2. **Stint-length caps per compound** — the longest stint the compound has been
   run to here, scaled to this year's race distance with a small margin.  Not a
   model output: a fact about the circuit, and the hard bound the optimiser and
   the live engine respect.  The same numbers, without the margin, are the
   **cliff** the sealed file reports: practice cannot identify a knee, and the
   longest and p90 stints the compound has actually been run to here are the
   honest statement of where it ends.
3. **The compound ladder as this circuit's races show it** — the fresh-tyre
   pace step (compound intercepts of the same regression, i.e. pace at equal
   tyre age) and the degradation ratio between adjacent compounds — which is
   the prior the practice fit's ladder now starts from, instead of one
   season-wide constant.
4. **The field's revealed plan shapes** — every classified finisher's compound
   sequence and starting compound, which the strategy search reads as a prior
   over plan families so a sequence nobody has run here needs a large time
   gain to be recommended.  Recorded in roles (SOFT/MEDIUM/HARD) but counted
   in C-numbers: Pirelli's nomination moves between seasons, so every plan is
   translated through `src.nominations` into the compounds the target year
   actually has before it is pooled.
5. **When the field stopped, under green flags**, and **what running in dirty
   air costs here** — the two circuit facts practice cannot show and the 2026
   donors cannot transfer, which the first-stop prior and the position model
   consume.

Everything is cached under `data/processed/history/` so a weekend build does
not re-read fifty races.  The per-race summary carries a version; an older
cache entry is recomputed on first use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COMPOUND_ORDER, DATA_PROCESSED, EVENTS, FASTF1_CACHE, VALID_COMPOUNDS, Event, get_event
from src.nominations import comparable, map_sequence, nomination, role_of

log = logging.getLogger("degless.history")

HIST_DIR = DATA_PROCESSED / "history"
YEARS = (2023, 2024, 2025)
SUMMARY_VERSION = 4
# Fuel physics of the previous regulations, for the evolution/fuel split of old races.
FUEL_S_PER_LAP_PRE2026 = 1.67 * 0.033
MIN_STINT = 3
CAP_MARGIN = 1.10          # stint cap = historical max x margin, scaled to race length
RATE_PRIOR_LN_SD_FLOOR = 0.35
THERMAL_BETA_DEFAULT = 0.025   # d log(deg) / d track temp, per °C, measured within circuit x compound 2023-2025
MISSING_RETRY_S = 7 * 24 * 3600   # how long a 'no such race' marker is believed
SEASON_PRIOR_LN_SD_FLOOR = 0.45   # a circuit nobody has raced: at least this wide
MAX_HISTORY_SCALE = 3.0           # the history may not move a practice rate by more than this factor per draw
RATE_FLOOR_SIGMA = 1.5            # floor = the pooled race rate's lower bound this many ln-sd below its mean
PACE_STEP_SE_MAX_S = 0.25         # a circuit's fresh-tyre step is only used as a prior mean when this precise
HIST_NET_SE_FLOOR_S = 0.10        # a pre-2026 net stint step (older tyres, older cars) is believed no tighter than this
# A pre-2026 net stint step measured less precisely than this is not used at
# all.  Melbourne is why: 2024's -0.46 +/- 0.24 s/step dragged the pooled net
# at Australia to -0.07 on its own, against +0.06 to +0.17 on every 2026
# donor, and a negative net asks the optimiser to believe a harder tyre is
# quicker over a stint.  A measurement that wide is the estimator's phase bias
# (19 stints, three red flags), not the tyre range.
HIST_NET_SE_MAX_S = 0.15
# Recency weights for pooling a circuit's own races: the tyre range and the
# cars moved every winter, so 2023 is half a race and 2025 is a whole one.
HIST_RECENCY_WEIGHT = {2023: 0.5, 2024: 0.7, 2025: 1.0}


# --------------------------------------------------------------------------
# One race
# --------------------------------------------------------------------------


def _offline_mode() -> bool:
    """Is FastF1 refusing the network (`Cache.offline_mode(True)`)?

    It has no getter, so the cached session's setting is read directly.  The
    answer decides two things: whether our own `enable_cache` call is allowed
    to cancel the caller's offline mode (see `_fastf1`), and whether a failed
    load is recorded as "this race does not exist" (see `summarise_race`) -
    offline, a failure means only that this race is not in the cache.
    """
    try:
        import fastf1

        s = fastf1.Cache._requests_session_cached
        return bool(s is not None and s.settings.only_if_cached)
    except Exception:
        return False


def _fastf1():
    import fastf1
    import logging as _l

    # `enable_cache` builds a *new* requests-cache session, which silently
    # clears an offline mode the caller asked for - so a build that was told
    # not to touch the network would quietly touch it.  Re-apply it.
    offline = _offline_mode()
    fastf1.Cache.enable_cache(str(FASTF1_CACHE))
    if offline:
        fastf1.Cache.offline_mode(True)
    _l.getLogger("fastf1").setLevel(_l.ERROR)
    return fastf1


def race_deg_slopes(laps: pd.DataFrame, fuel_s_per_lap: float, *, offsets: bool = False) -> dict:
    """Per-compound degradation, s/lap, with driver and race-lap fixed effects.

    The lap effect absorbs track evolution and fuel burn alike; a tyre-age
    slope per compound is identified from the cross-section of cars at
    different ages on the same lap.  Returns {compound: {slope, se, n_laps}}.

    With `offsets=True` each compound also carries `offset_s`: its pace at
    equal tyre age relative to the softest compound present (the compound
    intercept of the same regression), with its standard error.  That is the
    fresh-tyre pace step the strategy model charges per lap, measured jointly
    with the degradation it charges separately - so the two together
    reproduce this race's net stint-level compound effect by construction,
    which is the combination the optimiser actually consumes.
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
        oses = np.sqrt(np.clip(np.diag(cov)[1 + len(comps):2 * len(comps)], 0, None))
    except Exception:
        ses = np.full(len(comps), np.nan)
        oses = np.full(max(len(comps) - 1, 0), np.nan)
    out = {}
    # compound intercepts relative to the alphabetically-first compound
    raw_off = {comps[0]: 0.0}
    raw_se = {comps[0]: 0.0}
    for i, c in enumerate(comps[1:]):
        raw_off[c] = float(beta[1 + len(comps) + i])
        raw_se[c] = float(oses[i]) if i < len(oses) else float("nan")
    softest = next((c for c in COMPOUND_ORDER if c in comps), comps[0])
    for i, c in enumerate(comps):
        n = int((d["compound"] == c).sum())
        if n >= 30:
            out[c] = {"slope": float(beta[1 + i]), "se": float(ses[i]), "n_laps": n}
            if offsets:
                out[c]["offset_s"] = float(raw_off[c] - raw_off[softest])
                out[c]["offset_se"] = float(np.sqrt(raw_se[c] ** 2 + raw_se[softest] ** 2))
    return out


def race_driver_factors(laps: pd.DataFrame, *, min_laps: int = 25) -> dict:
    """How much harder than the field each driver is on the tyre, this race.

    The same regression as `race_deg_slopes` with a per-driver age slope
    added: `y = ... + slope[c] * age + delta[d] * age + driver FE + lap FE`.
    `delta[d]` is the driver's extra degradation per lap of age, in s/lap,
    over the field's compound rate.  Returned per driver as a multiplicative
    factor on the rate, `1 + delta / slope_bar` where `slope_bar` is the
    field's rate on the compounds that driver ran, with its standard error
    and lap count - the inputs the recalibration script pools and shrinks.
    """
    d = laps.dropna(subset=["lap_time_s", "tyre_life", "compound"])
    d = d[d["compound"].isin(VALID_COMPOUNDS)]
    if len(d) < 80 or d["lap_number"].nunique() < 8:
        return {}
    comps = sorted(d["compound"].unique())
    drivers = sorted(d["driver"].unique())
    drv = pd.get_dummies(d["driver"]).astype(float)[drivers]
    lap = pd.get_dummies(d["lap_number"].astype(int), drop_first=True).astype(float)
    comp = pd.get_dummies(d["compound"]).astype(float)[comps]
    age = d["tyre_life"].to_numpy(dtype=float)
    ageX = np.column_stack([comp[c].to_numpy() * age for c in comps])
    # driver x age, sum-to-zero over drivers so the compound slopes stay the field's
    dage = drv.to_numpy() * age[:, None]
    dage = dage[:, :-1] - dage[:, -1:]
    X = np.column_stack([np.ones(len(d)), ageX, dage, comp.to_numpy()[:, 1:], drv.to_numpy()[:, 1:], lap.to_numpy()])
    y = d["lap_time_s"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    try:
        cov = np.linalg.pinv(X.T @ X) * float(resid @ resid / dof)
    except Exception:
        return {}
    k = 1 + len(comps)
    deltas = np.concatenate([beta[k:k + len(drivers) - 1], [-beta[k:k + len(drivers) - 1].sum()]])
    dvar = np.concatenate([np.diag(cov)[k:k + len(drivers) - 1], [np.sum(cov[k:k + len(drivers) - 1, k:k + len(drivers) - 1])]])
    slopes = {c: float(beta[1 + i]) for i, c in enumerate(comps)}
    out = {}
    for i, drv_name in enumerate(drivers):
        g = d[d["driver"] == drv_name]
        if len(g) < min_laps:
            continue
        w = g["compound"].value_counts()
        sbar = float(sum(slopes[c] * n for c, n in w.items()) / w.sum())
        if sbar <= 0.005:
            continue
        out[drv_name] = {"delta_s_per_lap": float(deltas[i]), "se": float(np.sqrt(max(dvar[i], 0))),
                         "field_rate": sbar, "factor": float(1.0 + deltas[i] / sbar),
                         "factor_se": float(np.sqrt(max(dvar[i], 0)) / sbar), "n_laps": int(len(g))}
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
    """An old race's laps in the column names the 2026 estimators expect.

    `lap_start_s`, `gap_ahead_s` and `tyre_age` are here so the canonical frame
    is a drop-in for a 2026 lap table: the gap to the car ahead is what
    `compounds.measure_dirty_air` regresses on (the same
    `laps._gap_to_car_ahead` the clean-lap cascade uses, over the whole race
    rather than per session), and `tyre_age` is the name it and the cliff
    detector use for `tyre_life`.
    """
    from src.laps import _gap_to_car_ahead

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
        "lap_start_s": (laps["LapStartTime"].dt.total_seconds()
                        if "LapStartTime" in laps else np.nan),
    })
    out["tyre_age"] = out["tyre_life"]
    out["gap_ahead_s"] = (_gap_to_car_ahead(out) if out["lap_start_s"].notna().any()
                          else np.inf)
    return out


def _ladder_from_deg(deg: dict) -> dict:
    """Adjacent-compound pace step and degradation ratio from one race's regression."""
    comps = [c for c in COMPOUND_ORDER if c in deg]
    rank = {c: i for i, c in enumerate(COMPOUND_ORDER)}
    steps, ratios, wsteps, wratios = [], [], [], []
    for a, b in zip(comps, comps[1:]):
        dr = rank[b] - rank[a]
        da, db = deg[a], deg[b]
        if "offset_s" in da and "offset_s" in db and np.isfinite(db.get("offset_se", np.nan)):
            steps.append((db["offset_s"] - da["offset_s"]) / dr)
            wsteps.append(1.0 / max(db["offset_se"] ** 2 + da["offset_se"] ** 2, 1e-4))
        if da["slope"] > 0.003 and db["slope"] > 0.003:
            ratios.append(np.log(da["slope"] / db["slope"]) / dr)
            wratios.append(min(da["n_laps"], db["n_laps"]))
    out = {}
    if steps:
        out["pace_step_s"] = float(np.average(steps, weights=wsteps))
        out["pace_step_se"] = float(np.sqrt(1.0 / np.sum(wsteps)))
    if ratios:
        out["deg_ratio"] = float(np.exp(np.average(ratios, weights=wratios)))
        out["n_pairs"] = len(ratios)
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
        elif int(d.get("version", 1)) >= SUMMARY_VERSION:
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
        # The marker says "there is no such race", and it is believed for a
        # week.  Offline, a failed load means only "not in this cache", so the
        # marker is not written - it would blind every later build for a week
        # over a race that is simply absent from the cache.  A LookupError is
        # the exception: it is raised above from the schedule itself, which is
        # cached, so it is a real answer even offline.
        if not _offline_mode() or isinstance(exc, LookupError):
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
    starts = st.sort_values(["driver", "start"]).groupby("driver")["compound"].first()
    per_comp = {}
    for c, g in st.groupby("compound"):
        per_comp[c] = {"n_stints": int(len(g)), "p10": float(g["n"].quantile(0.1)), "p50": float(g["n"].median()),
                       "p90": float(g["n"].quantile(0.9)), "max": int(g["n"].max()),
                       "share_of_laps": float(g["n"].sum() / st["n"].sum())}
    clean = laps[laps["is_accurate"] & ~laps["pit_in"] & ~laps["pit_out"] & (laps["track_status"] == "1")]
    fuel = FUEL_S_PER_LAP_PRE2026 if year < 2026 else get_event_fuel(circuit)
    deg = race_deg_slopes(clean, fuel, offsets=True)
    ladder = _ladder_from_deg(deg)
    # net stint-level compound step, the quantity the ladder is gated against
    from src.compounds import net_stint_step_from_laps
    ns = net_stint_step_from_laps(laps, fuel_s_per_lap=fuel, n_race_laps=n_laps)
    # first stops: in-lap of the first stop per classified finisher that stopped
    # (a finisher who never stopped has no first stop), the opening stint's
    # compound and length, and whether the stop fell under a safety car.  The
    # compound is the *letter of that year* - the C-number comes from the
    # nomination table at read time, so this JSON stays nomination-agnostic.
    # `n_stops` is the driver's total stop count in that race, off the same stint
    # table `stops` is counted from.  It is what makes the first-stop prior
    # conditionable: a one-stopper's first stop and a two-stopper's are answers to
    # different questions, and pooling them is how a circuit with a two-stop year
    # pulls a one-stop plan ten laps early.
    status_by = laps.set_index(["driver", "lap_number"])["track_status"].to_dict()
    first_rows = []
    for drv, g in st.sort_values(["driver", "start"]).groupby("driver"):
        if len(g) < 2:
            continue
        in_lap = int(g["start"].iloc[1]) - 1
        first_rows.append({"driver": drv, "compound": str(g["compound"].iloc[0]),
                           "laps": int(g["n"].iloc[0]), "in_lap": in_lap,
                           "n_stops": int(len(g) - 1),
                           "sc": str(status_by.get((drv, float(in_lap)), "1")) != "1"})
    fr = pd.DataFrame(first_rows)
    first_stop = ({"median_lap": float(fr["in_lap"].median()), "p25": float(fr["in_lap"].quantile(0.25)),
                   "p75": float(fr["in_lap"].quantile(0.75)), "n": int(len(fr)),
                   "sc_share": float(fr["sc"].mean())} if len(fr) else {})
    # Green-flag first stops only: a stop taken because the safety car came out
    # says nothing about when the tyre was done, and pooling the two is how the
    # first-stop prior ends up 4 laps early at a circuit with a high SC rate.
    fg = fr[~fr["sc"]] if len(fr) else fr
    first_stop_green = ({"median_lap": float(fg["in_lap"].median()), "p25": float(fg["in_lap"].quantile(0.25)),
                         "p75": float(fg["in_lap"].quantile(0.75)), "n": int(len(fg)),
                         "in_laps": [int(v) for v in sorted(fg["in_lap"])]} if len(fg) else {})
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
    # Dirty air, measured on this race rather than transferred from the 2026
    # donors: the cost of running within 3 s of the car ahead is a property of
    # the circuit (Hungary +0.43 s/lap, Monza -0.20) and the recalibration needs
    # a per-circuit value that never reads the target's own race.
    from src.compounds import measure_dirty_air
    ev_for_circuit = event_for_circuit(circuit)
    dirty = {}
    if ev_for_circuit is not None and laps["lap_start_s"].notna().any():
        try:
            dirty = measure_dirty_air(laps, ev_for_circuit)
        except Exception as exc:       # a race whose design matrix is singular
            log.info("no dirty-air measurement for %s %s: %s", year, circuit, str(exc)[:80])
    weather = s.weather_data
    out = {
        "version": SUMMARY_VERSION,
        "year": year, "circuit": circuit, "event": str(s.event["EventName"]), "location": str(s.event["Location"]),
        "date": str(s.event["EventDate"])[:10], "n_laps": n_laps, "n_classified": int(len(classified)),
        "stops": {int(k): int(v) for k, v in stops.value_counts().sort_index().items()},
        "plans": {k: int(v) for k, v in plans.value_counts().items()},
        "starts": {k: int(v) for k, v in starts.value_counts().items()},
        "first_stop": first_stop,
        "first_stints": first_rows,
        "first_stop_green": first_stop_green,
        "dirty_air": dirty,
        "nomination": nomination(year, circuit),
        "compounds": per_comp, "deg": deg, "ladder": ladder,
        "net_step": ({"step_s": float(ns.step_s), "se": float(ns.se), "n_stints": int(ns.n_laps),
                      "median_stint_laps": float(ns.phase_bias_s)} if np.isfinite(ns.step_s) else {}),
        "pit_loss_s": (float(np.median(rows)) if len(rows) >= 2 else None), "n_pit_stops": len(rows),
        "sc_share_of_laps": sc_laps,
        "track_temp_c": (float(weather["TrackTemp"].median()) if weather is not None and len(weather) else None),
        "air_temp_c": (float(weather["AirTemp"].median()) if weather is not None and len(weather) else None),
        "rain": bool(weather["Rainfall"].any()) if weather is not None and len(weather) else False,
    }
    p.write_text(json.dumps(out, indent=1))
    return out


def event_for_circuit(circuit: str) -> Event | None:
    """The 2026 event at this circuit, if the calendar has one."""
    return next((e for e in EVENTS.values() if e.circuit.lower() == str(circuit).lower()), None)


def get_event_fuel(circuit: str) -> float:
    ev = event_for_circuit(circuit)
    return ev.fuel_effect_s_per_lap if ev else 0.031


# --------------------------------------------------------------------------
# Track temperatures of this season's sessions, cached
# --------------------------------------------------------------------------


def session_track_temp(ev: Event, session_name: str) -> float | None:
    """Median track temperature of one 2026 session from the FastF1 cache,
    memoised in `history/temps_<key>.json` so the regime prior does not load
    a weather table every time it is asked."""
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    p = HIST_DIR / f"temps_{ev.key}.json"
    cache = {}
    if p.exists():
        try:
            cache = json.loads(p.read_text())
        except Exception:
            cache = {}
    if session_name in cache:
        return cache[session_name]
    ff1 = _fastf1()
    val = None
    try:
        s = ff1.get_session(ev.ff1_year, ev.ff1_round, session_name)
        s.load(laps=False, telemetry=False, weather=True, messages=False)
        w = s.weather_data
        if w is not None and len(w) and w["TrackTemp"].notna().any():
            val = float(w["TrackTemp"].median())
    except Exception as exc:
        log.info("no weather for %s %s: %s", ev.key, session_name, str(exc)[:60])
        return None
    cache[session_name] = val
    p.write_text(json.dumps(cache, indent=1))
    return val


def practice_track_temp(ev: Event, clean: pd.DataFrame | None = None) -> float | None:
    """Track temperature the practice long runs were done at.

    Laps-weighted over the sessions that contributed clean long-run laps when
    a clean lap table is given (or on disk), so a hot FP3 and a cool FP2 are
    weighted by what the fit actually saw; otherwise the session run closest
    to race time of day (FP2 on a conventional weekend)."""
    if clean is None:
        cp = DATA_PROCESSED / f"clean_{ev.key}_practice.parquet"
        if cp.exists():
            try:
                clean = pd.read_parquet(cp, columns=["session"])
            except Exception:
                clean = None
    if clean is not None and "session" in clean and len(clean):
        w = clean["session"].value_counts()
        num = den = 0.0
        for name, n in w.items():
            t = session_track_temp(ev, str(name))
            if t is not None:
                num += t * n
                den += n
        if den > 0:
            return float(num / den)
    for name in ("Practice 2", "Practice 3", "Practice 1"):
        if name not in ev.practice_sessions:
            continue
        t = session_track_temp(ev, name)
        if t is not None:
            return t
    return None


def race_track_temp(ev: Event) -> float | None:
    """This weekend's race track temperature - a *post-race* fact, used only
    to measure donor weekends and never to predict the target's own race."""
    return session_track_temp(ev, "Race")


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


@dataclass
class CircuitPrior:
    event: str
    circuit: str
    years: list = field(default_factory=list)
    races: list = field(default_factory=list)            # the per-race summaries
    rate_prior: dict = field(default_factory=dict)       # compound -> {mean_s_per_lap, ln_sd, source}
    rate_floor: dict = field(default_factory=dict)       # compound -> smallest race rate seen here, season-scaled
    stint_cap: dict = field(default_factory=dict)        # compound -> max laps this year (with margin)
    stint_longest: dict = field(default_factory=dict)    # compound -> longest stint run here, scaled, no margin
    stint_typical: dict = field(default_factory=dict)    # compound -> {p10,p50,p90} scaled
    stops: dict = field(default_factory=dict)            # pooled distribution over classified finishers
    plans: dict = field(default_factory=dict)            # top plan families, for display
    plans_all: dict = field(default_factory=dict)        # every plan family with its count
    starts: dict = field(default_factory=dict)           # start compound -> count
    first_stop: dict = field(default_factory=dict)       # {median_lap (scaled), p25, p75, n, sc_share}
    # green only: {median_lap, p25, p75, n, in_laps, by_compound, by_stops,
    # by_compound_stops, by_year} - see `circuit_prior`
    first_stop_green: dict = field(default_factory=dict)
    ladder: dict = field(default_factory=dict)           # {pace_step_s, pace_step_se, deg_ratio, deg_ratio_ln_sd, n_races}
    net_steps: list = field(default_factory=list)        # per race: {year, step_s, se, n_stints}
    dirty_air: dict = field(default_factory=dict)        # {s_per_lap, se, n_races, by_year} - not clipped positive
    nomination: list | None = None                       # the target year's C-numbers, hard -> soft
    nominations_by_year: dict = field(default_factory=dict)
    pit_loss_s: float | None = None
    sc_share: float = 0.0
    season: dict = field(default_factory=dict)
    thermal: dict = field(default_factory=dict)          # {beta_per_c, track_temp_now, track_temp_hist, multiplier}
    race_temps: list = field(default_factory=list)       # median race track temperature per year
    soft_race_tyre: bool = True                          # was the SOFT run for real stints here?

    @property
    def available(self) -> bool:
        return bool(self.races)

    def cliff(self) -> dict:
        """The cliff as the circuit's races state it: longest and p90 stint per compound."""
        out = {}
        for c in VALID_COMPOUNDS:
            if c in self.stint_longest:
                out[c] = {"longest_stint": float(self.stint_longest[c]),
                          "p90_stint": float(self.stint_typical.get(c, {}).get("p90", np.nan)),
                          "p50_stint": float(self.stint_typical.get(c, {}).get("p50", np.nan)),
                          "n_stints": int(self.stint_typical.get(c, {}).get("n_stints", 0)),
                          "source": f"{self.circuit} races {self.years}, scaled to {self.event}'s distance"}
        return out

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "races"} | {
            "races": [{k: r.get(k) for k in ("year", "event", "n_laps", "stops", "plans", "starts", "first_stop",
                                              "first_stop_green", "dirty_air", "nomination",
                                              "compounds", "deg", "ladder", "net_step", "pit_loss_s",
                                              "sc_share_of_laps", "track_temp_c")} for r in self.races]}


def circuit_prior(event: Event | str, years=YEARS, *, track_temp_c: float | None = None,
                  probe_practice_temp: bool = True) -> CircuitPrior:
    """`probe_practice_temp=False` skips the FastF1 lookup of this weekend's
    practice temperature (which does not exist before the weekend starts and
    is slow to discover that); the thermal multiplier is then 1.0 unless a
    temperature is passed in."""
    ev = get_event(event) if isinstance(event, str) else event
    races = [r for r in (summarise_race(y, ev.circuit) for y in years) if r and not r.get("rain")]
    cp = CircuitPrior(event=ev.key, circuit=ev.circuit, years=[r["year"] for r in races], races=races)
    # The nominations first, so a circuit with no history at all (Madring) still
    # tells `plan_prior_for` which compounds the season pool must be mapped to.
    cp.nomination = nomination(ev.ff1_year, ev.circuit)
    cp.nominations_by_year = {int(y): nomination(y, ev.circuit) for y in (*years, ev.ff1_year)}
    if not races:
        return cp
    sf = season_factor()
    cp.season = sf
    # -- thermal proxy: this weekend's track temperature against the archive's
    th = thermal_sensitivity()
    temps = [r["track_temp_c"] for r in races if r.get("track_temp_c") is not None]
    cp.race_temps = [float(t) for t in temps]
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
        # The floor: the lower end of what the circuit's races support.  The
        # pooled race rate's 1.5-sigma lower bound on the log scale (about
        # half the mean at the usual widths), so a flat practice fit cannot
        # claim a tyre that degrades slower than every race here has shown,
        # while the floor can never override the combination itself.
        cp.rate_floor[c] = float(cp.rate_prior[c]["mean_s_per_lap"] * np.exp(-RATE_FLOOR_SIGMA * ln_sd))
    if cp.rate_floor:
        # a compound with no history of its own can never degrade slower than the slowest one that has
        lo = min(cp.rate_floor.values())
        for c in VALID_COMPOUNDS:
            cp.rate_floor.setdefault(c, lo)
    # -- the ladder as this circuit's races show it --------------------------
    # Weighted by recency as well as precision: 2023 ran a different tyre range
    # and a different car, and at Melbourne the old races are also the noisy
    # ones (see HIST_RECENCY_WEIGHT and HIST_NET_SE_MAX_S).
    steps, sw, ratios, rw = [], [], [], []
    for r in races:
        L = r.get("ladder") or {}
        rec = HIST_RECENCY_WEIGHT.get(int(r["year"]), 1.0)
        if "pace_step_s" in L and np.isfinite(L.get("pace_step_se", np.nan)) and 0 < L["pace_step_se"] <= 2 * PACE_STEP_SE_MAX_S:
            steps.append(L["pace_step_s"]); sw.append(rec / max(L["pace_step_se"], 0.03) ** 2)
        if "deg_ratio" in L and L["deg_ratio"] > 0:
            ratios.append(np.log(L["deg_ratio"])); rw.append(rec * float(L.get("n_pairs", 1)))
    if steps or ratios:
        cp.ladder = {"n_races": len(races), "recency_weights": [HIST_RECENCY_WEIGHT.get(int(r["year"]), 1.0) for r in races]}
        if steps:
            cp.ladder["pace_step_s"] = float(np.average(steps, weights=sw))
            cp.ladder["pace_step_se"] = float(np.sqrt(1.0 / np.sum(sw)))
            cp.ladder["pace_step_by_year"] = [round(s, 3) for s in steps]
            cp.ladder["pace_step_usable"] = bool(cp.ladder["pace_step_se"] <= PACE_STEP_SE_MAX_S
                                                 and cp.ladder["pace_step_s"] > 0)
        if ratios:
            cp.ladder["deg_ratio"] = float(np.exp(np.average(ratios, weights=rw)))
            spread = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.25
            cp.ladder["deg_ratio_ln_sd"] = float(np.clip(spread, 0.15, 0.5))
            cp.ladder["deg_ratio_by_year"] = [round(float(np.exp(x)), 3) for x in ratios]
    # Every measurement is carried, with the raw standard error beside the
    # floored one: `compounds.pace_step_prior` is where the se > HIST_NET_SE_MAX_S
    # drop happens, so it can report what it dropped and why.
    cp.net_steps = [{"year": r["year"], **r["net_step"],
                     "se": float(max(r["net_step"].get("se", HIST_NET_SE_FLOOR_S), HIST_NET_SE_FLOOR_S)),
                     "se_raw": float(r["net_step"].get("se", float("nan"))),
                     "recency": HIST_RECENCY_WEIGHT.get(int(r["year"]), 1.0)}
                    for r in races if r.get("net_step")]
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
            cp.stint_longest[c] = float(max(mx))
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
    cp.plans_all = dict(sorted(plans.items(), key=lambda t: -t[1]))
    cp.plans = dict(list(cp.plans_all.items())[:6])
    starts: dict = {}
    for r in races:
        for k, v in (r.get("starts") or {}).items():
            starts[k] = starts.get(k, 0) + int(v)
    cp.starts = dict(sorted(starts.items(), key=lambda t: -t[1]))
    fs = [(r["first_stop"], ev.n_race_laps / max(r["n_laps"], 1)) for r in races if r.get("first_stop")]
    if fs:
        cp.first_stop = {"median_lap": float(np.mean([f["median_lap"] * s for f, s in fs])),
                         "p25": float(np.mean([f["p25"] * s for f, s in fs])),
                         "p75": float(np.mean([f["p75"] * s for f, s in fs])),
                         "n": int(sum(f["n"] for f, _ in fs)),
                         "sc_share": float(np.mean([f["sc_share"] for f, _ in fs]))}
    # -- green-flag first stops, as a sample rather than three quantiles -----
    # The first-stop prior (WP-C) is a density over laps, so the pooled in-laps
    # themselves are what it needs: each scaled to this year's distance, and
    # keyed by the compound the stint was run on *as the target year names it*
    # (Barcelona's 2025 C3 openers were "SOFT" then and are this year's MEDIUM).
    # `by_stops` and `by_compound_stops` condition on the plan the stop belonged
    # to.  A first stop is the opening move of a *plan*, and the same circuit
    # supports very different ones: Spa's 2024 two-stoppers came in on lap 11 and
    # its one-stoppers on lap 19, so the unconditional density has a mode between
    # them that belongs to neither family.  Keyed "<letter>|<n_stops>" so the
    # JSON round-trip keeps them as strings.
    in_laps, by_compound, by_year = [], {}, {}
    by_stops, by_compound_stops = {}, {}
    for r in races:
        g = r.get("first_stop_green") or {}
        if not g.get("in_laps"):
            continue
        scale = ev.n_race_laps / max(r["n_laps"], 1)
        scaled = [float(v) * scale for v in g["in_laps"]]
        in_laps += scaled
        by_year[int(r["year"])] = {"median_lap": float(np.median(scaled)), "n": len(scaled),
                                   "raw_median_lap": float(g.get("median_lap", np.nan)),
                                   "n_race_laps": int(r["n_laps"])}
        nom_y = r.get("nomination") or cp.nominations_by_year.get(int(r["year"]))
        how = comparable(nom_y, cp.nomination)
        rows = [x for x in (r.get("first_stints") or []) if not x.get("sc")]
        if not rows:
            continue
        for x in rows:
            lap = float(x["in_lap"]) * scale
            ns = x.get("n_stops")
            # The stop count needs no nomination mapping - how many times the
            # field stopped is a fact about the pit lane - so it is pooled even
            # where the compounds cannot be compared.
            if ns is not None:
                by_stops.setdefault(str(int(ns)), []).append(lap)
            if how in ("unknown", "disjoint"):
                continue
            letters, _ = map_sequence([x["compound"]], nom_y, cp.nomination)
            by_compound.setdefault(letters[0], []).append(lap)
            if ns is not None:
                by_compound_stops.setdefault(f"{letters[0]}|{int(ns)}", []).append(lap)
    if in_laps:
        a = np.sort(np.array(in_laps, dtype=float))
        cp.first_stop_green = {"median_lap": float(np.median(a)), "p25": float(np.quantile(a, 0.25)),
                               "p75": float(np.quantile(a, 0.75)), "n": int(len(a)),
                               "in_laps": [float(v) for v in a],
                               "by_compound": {k: sorted(v) for k, v in by_compound.items()},
                               "by_stops": {k: sorted(v) for k, v in sorted(by_stops.items())},
                               "by_compound_stops": {k: sorted(v) for k, v in sorted(by_compound_stops.items())},
                               "by_year": by_year,
                               "source": f"{cp.circuit} green-flag first stops {cp.years}, "
                                         f"scaled to {ev.n_race_laps} laps"}
    # -- dirty air, pooled by precision over the years ----------------------
    # Deliberately not clipped at zero: Monza's measurement is negative (a car
    # in the tow is quicker there) and pretending otherwise is how a pooled
    # constant ends up charging Monza for the slipstream.
    da_v, da_w, da_by = [], [], {}
    for r in races:
        d = r.get("dirty_air") or {}
        v, se = d.get("s_per_lap"), d.get("se")
        if v is None or not np.isfinite(float(v)) or se is None or not np.isfinite(float(se)):
            continue
        da_by[int(r["year"])] = {"s_per_lap": float(v), "se": float(se), "n_laps": int(d.get("n_laps", 0)),
                                 "share_close": float(d.get("share_close", np.nan))}
        da_v.append(float(v)); da_w.append(1.0 / max(float(se), 0.02) ** 2)
    if da_v:
        w = np.array(da_w)
        cp.dirty_air = {"s_per_lap": float(np.sum(np.array(da_v) * w) / w.sum()),
                        "se": float(np.sqrt(1.0 / w.sum())), "n_races": len(da_v), "by_year": da_by,
                        "source": f"{cp.circuit} races {sorted(da_by)}, within 3 s of the car ahead"}
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
                              pooled: bool = False, max_scale: float = MAX_HISTORY_SCALE,
                              rate_floor: dict | None = None):
    """`apply_circuit_prior` for a `TyreModel` rather than a Bayes fit.

    Used by the outlook before any practice exists (the ladder prior combined
    with the circuit's or the season's race history) and while practice is
    running (the live long-run board folded in with `same_regime=True`, since
    a long run and the model are both in the practice regime).  Returns a new
    model and the table of what moved.

    Only the *rate* moves: each compound's wear rate is rescaled per draw by
    the ratio of the combined law to the model's own, capped at `max_scale`
    either way, and floored at `rate_floor[c]` (practice regime, s/lap) where
    the circuit's history gives one.  Nothing else about the model changes.

    `pooled=True` reads the history as one statement about the *circuit's
    severity* rather than three about the compounds: each compound's history
    is converted to what it implies for the reference compound through the
    model's own ladder, the implications are precision-pooled, and one common
    scale is applied to every compound.  The ladder ordering therefore cannot
    invert, which per-compound combination of two or three noisy races can do
    (and did).  It is the right reading when the model's only knowledge of the
    compounds is the ladder itself, i.e. before any practice has run.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    lr_ratio = 0.0 if same_regime else float(np.log(max(regime.ratio, 0.05)))
    rg_sd = 0.0 if same_regime else float(regime.ln_sd)
    wear = {c: model.wear_rate[c].copy() for c in model.compounds}
    floors = rate_floor or {}

    def _finish(c, new_rate):
        r = np.asarray(new_rate, float)
        if c in floors and floors[c]:
            r = np.maximum(r, float(floors[c]))
        return r / model.budget_of(c)

    rows, scales, pending = [], {}, []
    if pooled:
        present = [c for c in model.compounds if rate_prior.get(c)]
        if not present:
            return model, pd.DataFrame()
        ref = "MEDIUM" if "MEDIUM" in model.compounds else model.compounds[0]
        lr_ref = np.log(np.maximum(model.rate(ref), 1e-4))
        mu_ref = float(lr_ref.mean())
        ys, ws = [], []
        for c in present:
            pr = rate_prior[c]
            mu_c = float(np.log(np.maximum(model.rate(c), 1e-4)).mean())
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
        common = np.clip(np.exp(new_lr - lr_ref), 1.0 / max_scale, max_scale)
        for c in model.compounds:
            wear[c] = _finish(c, model.rate(c) * common)
        for r in rows:
            r.update({"practice": row["practice"], "practice_as_race": row["practice_as_race"],
                      "combined_race": row["combined_race"], "combined": row["combined"],
                      "weight_on_history": row["weight_on_history"], "pooled": True, "reference": ref,
                      "scale_mean": float(common.mean())})
        return model.copy_with(wear_rate=wear, source=f"{model.source} + {label} (pooled)"), pd.DataFrame(rows)
    for c in model.compounds:
        rate = np.maximum(model.rate(c), 1e-4)
        lr = np.log(rate)
        pr = rate_prior.get(c)
        if not pr:
            pending.append(c)
            continue
        new_lr, row = _combine_lognormal(lr, float(np.log(pr["mean_s_per_lap"])), float(pr["ln_sd"]),
                                         lr_ratio, rg_sd, rng)
        scale = np.clip(np.exp(new_lr - lr), 1.0 / max_scale, max_scale)
        scales[c] = scale
        wear[c] = _finish(c, rate * scale)
        raw = np.exp(new_lr - lr)
        rows.append({"compound": c, **row, "source": label, "scale_mean": float(scale.mean()),
                     "capped_share": float(np.mean((raw > max_scale) | (raw < 1 / max_scale)))})
    if scales and pending:
        common = np.exp(np.mean([np.log(v) for v in scales.values()], axis=0))
        for c in pending:
            wear[c] = _finish(c, model.rate(c) * common)
            rows.append({"compound": c, "practice": float(model.rate(c).mean()),
                         "history_race": None, "combined": float((wear[c] * model.budget_of(c)).mean()),
                         "weight_on_history": None, "source": label,
                         "note": f"no {label} for this compound; moved with the ladder"})
    return model.copy_with(wear_rate=wear, source=f"{model.source} + {label}"), pd.DataFrame(rows)


def apply_circuit_prior(fit, ev: Event, regime, cp: CircuitPrior, *, seed: int = 0,
                        max_scale: float = MAX_HISTORY_SCALE, floor: bool = True):
    """Combine the practice posterior with the circuit's history, per compound.

    The history prior is on the *race-regime* rate; the practice fit is in the
    practice regime, and `regime.ratio` links the two.  On the log scale both
    are (approximately) normal, so the combination is the precision-weighted
    normal — the standard conjugate update — and each posterior draw's
    degradation *rate* is rescaled so the draws follow the combined law while
    keeping their ranks (the correlation with everything else is preserved).

    **Only the rate moves.**  The rate is the curve's average slope over the
    span practice supports (ages 1-10, the same reading `TyreModel.from_fit`
    takes), and the whole change is carried by the linear term.  A previous
    version rescaled the hinge by the same factor, and at a circuit where
    practice shows almost no degradation the factor is large: at Australia
    2026 a post-knee slope of 0.03-0.05 s/lap became 2.5-4.5 s/lap, the
    sealed curve collapsed after lap 17 and the race score went from 0.05 to
    0.43 s/lap.  The per-draw factor is now capped at `max_scale` either way
    and the combined rate is floored at the smallest race rate the circuit
    has ever shown (transferred to the practice regime), so a flat practice
    fit cannot derive a 2,900-lap tyre either.

    Returns a new fit object, plus a table of what moved.
    """
    from copy import deepcopy

    new = deepcopy(fit)
    rng = np.random.default_rng(seed)
    span = np.array([1.0, 10.0])
    ratio = float(max(regime.ratio, 0.05))
    rows = []
    scales: dict = {}       # compound -> per-draw multiplicative scale applied to the rate
    pending = []            # compounds with no history of their own

    def _set_rate(j, c, target):
        """Write a per-draw rate back into `lin`, leaving any hinge untouched."""
        lin = fit.posterior["lin"][:, j]
        d = fit.deg_loss(c, span)
        rate = np.maximum((d[:, 1] - d[:, 0]) / 9.0, 1e-4)
        hinge_part = rate - lin                    # what a hinge adds over the span (0 without one)
        t = np.asarray(target, float)
        if floor and cp.rate_floor.get(c):
            t = np.maximum(t, float(cp.rate_floor[c]) / ratio)
        new.posterior["lin"][:, j] = np.maximum(t - hinge_part, 1e-4)

    for j, c in enumerate(fit.compounds):
        d = fit.deg_loss(c, span)
        rate = np.maximum((d[:, 1] - d[:, 0]) / 9.0, 1e-4)          # practice-regime rate per draw
        lr = np.log(rate)
        mu_p = float(lr.mean())
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
                                         float(np.log(ratio)), float(regime.ln_sd), rng)
        raw_scale = np.exp(new_lr - lr)
        scale = np.clip(raw_scale, 1.0 / max_scale, max_scale)
        scales[c] = scale
        _set_rate(j, c, rate * scale)
        rows.append({"compound": c, **row, "scale_mean": float(scale.mean()),
                     "capped_share": float(np.mean((raw_scale > max_scale) | (raw_scale < 1 / max_scale))),
                     "floor_practice": (float(cp.rate_floor[c]) / ratio if cp.rate_floor.get(c) else None),
                     "floor_binds": bool(cp.rate_floor.get(c) and float(np.mean(rate * scale)) < float(cp.rate_floor[c]) / ratio)})
    # A compound the circuit has no race history for (Monza's SOFT, say) moves
    # with the others: the compound ladder is a property of the tyre range, so
    # the history's correction to the track's severity applies to it too.  Its
    # own ordering relative to the compounds that did move is thereby kept.
    if scales and pending:
        common = np.exp(np.mean([np.log(v) for v in scales.values()], axis=0))
        for j, c, mu_p in pending:
            d = fit.deg_loss(c, span)
            rate = np.maximum((d[:, 1] - d[:, 0]) / 9.0, 1e-4)
            _set_rate(j, c, rate * common)
            rows.append({"compound": c, "practice": float(np.exp(mu_p)), "history_race": None,
                         "combined": float(np.exp(mu_p) * common.mean()), "scale_mean": float(common.mean()),
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


def _start_key(letter: str) -> str:
    """The start-compound marginal is keyed by the full compound name - what the
    race summaries record and what `strategy.plan_prior_penalty` looks up."""
    return role_of(letter) or str(letter)


def _season_plan_pool(target_nomination: list | None = None, *, exclude: str | None = None) -> dict:
    """Plan shapes from the 2026 races run so far — the prior for a circuit
    nobody has raced (Madring) or whose own races cannot be compared.

    With a `target_nomination` each donor's sequences are translated out of its
    own 2026 nomination and into the target's, so Melbourne's C4 MEDIUM arrives
    at Madring as the SOFT it is there; a donor whose nomination is not in the
    table is skipped entirely rather than pooled on the letters.  `exclude` is
    the target event, which may never inform its own prior.
    """
    seqs, starts, stops, n, used, per_race = {}, {}, {}, 0, [], []
    for k, ev in EVENTS.items():
        rp = DATA_PROCESSED / f"laps_{k}_race.parquet"
        if k == exclude or not rp.exists() or not ev.donor_ok:
            continue
        nom = nomination(ev.ff1_year, ev.circuit) if target_nomination else None
        how = comparable(nom, target_nomination) if target_nomination else "letters"
        row = {"year": int(ev.ff1_year), "event": k, "nomination": nom, "comparable": how,
               "n_mapped": 0, "n_dropped": 0, "n_clamped": 0}
        if target_nomination and how in ("unknown", "disjoint"):
            per_race.append(row)
            continue
        r = pd.read_parquet(rp)
        st = (r.groupby(["driver", "stint"]).agg(compound=("compound", "first"), n=("lap_number", "size"),
                                                 start=("lap_number", "min"), end=("lap_number", "max")).reset_index())
        st = st[(st["n"] >= MIN_STINT) & st["compound"].isin(VALID_COMPOUNDS)]
        fin = st.groupby("driver")["end"].max()
        cls = fin[fin >= ev.n_race_laps - 2].index
        st = st[st["driver"].isin(cls)].sort_values(["driver", "start"])
        for _, g in st.groupby("driver"):
            src = [str(c)[0] for c in g["compound"]]
            if target_nomination:
                letters, info = map_sequence(src, nom, target_nomination)
            else:
                letters, info = src, {"clamped": False}
            stops[len(g) - 1] = stops.get(len(g) - 1, 0) + 1
            n += 1
            if len(set(letters)) == 1 and len(set(src)) > 1:
                # the same rule as the circuit prior: a plan that collapses onto
                # one compound under the target nomination keeps its stop count
                # only - its start is as uninterpretable as its sequence
                row["n_dropped"] += 1
                continue
            starts[_start_key(letters[0])] = starts.get(_start_key(letters[0]), 0) + 1
            seq = "-".join(letters)
            seqs[seq] = seqs.get(seq, 0) + 1
            row["n_mapped"] += 1
            row["n_clamped"] += int(bool(info["clamped"]))
        used.append(k)
        per_race.append(row)
    if not n:
        return {}
    note = " mapped by C-number" if target_nomination else ""
    return {"sequences": dict(sorted(seqs.items(), key=lambda t: -t[1])),
            "starts": dict(sorted(starts.items(), key=lambda t: -t[1])),
            "stops": dict(sorted(stops.items())), "n": int(sum(seqs.values())) if seqs else int(n),
            "source": f"2026 season pooled ({', '.join(used)}){note}", "per_race": per_race}


def plan_prior_for(cp: CircuitPrior | None, *, season_fallback: bool = True,
                   use_nominations: bool = True, target_nomination: list | None = None) -> dict:
    """The field's revealed plan shapes as a prior: `{"sequences", "starts",
    "stops", "n", "source", "nomination"}` in the short form the strategy
    search uses ("M-H-H"), from the circuit's own races, or - for a circuit
    nobody has raced - from every 2026 race summarised so far.

    The counts are stated in the **target year's** compounds, not in the
    letters the old races were recorded with (see `src.nominations`):

    1. an identical nomination is the letters unchanged;
    2. a shifted one maps every stint by C-number.  A sequence that collapses
       onto a single compound under the mapping is not a plan anybody can run
       at this nomination - Melbourne 2023's nine M-H one-stoppers are C3-C2,
       both of them 2026 HARDs - so it contributes its **stop count only**: it
       leaves the sequence counts *and* the start-compound counts.  Its start is
       as uninterpretable as its sequence, because what the team chose was the
       first half of a two-compound plan that does not exist at this nomination,
       and crediting the mapped letter would say the field opened on a tyre it
       never ran a comparable race on.  How many times the field stops here is
       a fact about the pit lane, so the stop count survives;
    3. a race whose nomination is disjoint or unverified contributes its stop
       count only, and the sequences fall back in order to the circuit's other
       years, the 2026 season pool (itself mapped into this nomination), and
       finally the letters as recorded, flagged `role-fallback`;
    4. the target's own 2026 race is never read.

    `use_nominations=False` is V2 exactly - the letters pooled as recorded -
    which is the ablation the benchmark compares against.
    """
    target = list(target_nomination) if target_nomination else (
        list(cp.nomination) if cp is not None and cp.nomination else None)
    if cp is not None and cp.available and cp.plans_all:
        letters_only = {"sequences": dict(cp.plans_all), "starts": dict(cp.starts),
                        "stops": {int(k): int(v) for k, v in cp.stops.items()},
                        "n": int(sum(cp.plans_all.values())), "source": f"{cp.circuit} races {cp.years}"}
        if not use_nominations:
            return letters_only
        stops = {int(k): int(v) for k, v in cp.stops.items()}
        seqs, starts, per_race, moved = {}, {}, [], False
        nby = cp.nominations_by_year or {}
        for r in cp.races:
            y = int(r["year"])
            # `str(y)` because a CircuitPrior rebuilt from a meta JSON has string
            # keys; the race's own entry is the first choice either way.
            nom = r.get("nomination") or nby.get(y) or nby.get(str(y))
            how = comparable(nom, target)
            row = {"year": y, "nomination": nom, "comparable": how,
                   "n_mapped": 0, "n_dropped": 0, "n_clamped": 0}
            per_race.append(row)
            if how not in ("identical", "shifted"):
                continue
            # Both counters come off the same object - `plans` is one entry per
            # classified finisher - so a dropped plan drops its start with it and
            # the two marginals stay over the same population.  (Reading
            # `r["starts"]` instead, as V2 did, keeps the start of a plan whose
            # sequence was thrown away.)
            for short, cnt in (r.get("plans") or {}).items():
                src = short.split("-")
                letters, info = map_sequence(src, nom, target)
                moved = moved or "-".join(letters) != short
                if len(set(letters)) == 1 and len(set(src)) > 1:
                    row["n_dropped"] += int(cnt)
                    continue
                seq = "-".join(letters)
                seqs[seq] = seqs.get(seq, 0) + int(cnt)
                key = _start_key(letters[0])
                starts[key] = starts.get(key, 0) + int(cnt)
                row["n_mapped"] += int(cnt)
                row["n_clamped"] += int(cnt) if info["clamped"] else 0
        if seqs:
            note = " mapped by C-number" if moved else ""
            return {"sequences": dict(sorted(seqs.items(), key=lambda t: -t[1])),
                    "starts": dict(sorted(starts.items(), key=lambda t: -t[1])),
                    "stops": dict(sorted(stops.items())), "n": int(sum(seqs.values())),
                    "source": f"{cp.circuit} races {cp.years}{note}",
                    "nomination": {"target": target, "per_race": per_race, "mode": "c-number"}}
        pool = _season_plan_pool(target, exclude=cp.event) if (season_fallback and target) else {}
        if pool:
            # The circuit's own stop counts survive - how many times the field
            # stops here is a fact about the pit lane, not about the compounds.
            return {"sequences": pool["sequences"], "starts": pool["starts"],
                    "stops": dict(sorted(stops.items())), "n": pool["n"],
                    "source": f"{cp.circuit} stop counts {cp.years}; shapes from {pool['source']}",
                    "nomination": {"target": target, "per_race": per_race + pool["per_race"],
                                   "mode": "c-number"}}
        return letters_only | {"source": f"{cp.circuit} races {cp.years} (role-fallback: "
                                         f"no comparable nomination)",
                               "nomination": {"target": target, "per_race": per_race, "mode": "letters"}}
    if not season_fallback:
        return {}
    pool = _season_plan_pool(target if use_nominations else None,
                             exclude=(cp.event if cp is not None else None))
    if not pool:
        return {}
    out = {k: v for k, v in pool.items() if k != "per_race"}
    if use_nominations:
        out["nomination"] = {"target": target, "per_race": pool["per_race"],
                             "mode": "c-number" if target else "letters"}
    return out
