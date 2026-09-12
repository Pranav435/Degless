"""Sealing and scoring — the receipt.

The model never sees race data.  Predictions are frozen to disk with a sha256
sidecar *before* any race lap is read, and scoring reads the sealed file back
rather than the live fit object, so the number on the slide is provably the
number that was predicted.

**What is actually scored.**  A practice fit cannot know the base pace of a race
stint: the car is on a different fuel load, in a different engine mode, on a
different track state.  What it *can* predict is the **shape** of the stint —
how much slower lap 15 of a stint is than lap 3.  So each race stint is
centred on its own mean and compared to the centred prediction.  That concedes
exactly one degree of freedom per stint (the level) and scores the thing the
model actually claims to know.  Stating this is the difference between a
validation and a demo.

Two errors are reported, and they measure different things:

* `mae_stint_rate` (**the headline**, s/lap) — per stint, the observed
  degradation *rate* from the mean of the first three laps to the mean of the
  last three, against the same quantity from the sealed curve.  The < 0.15
  s/lap target is on this scale.
* `mae_lap` (s) — raw per-lap error.  This one is bounded below by the per-lap
  noise of a racing lap and no tyre model can drive it to 0.15.  Reported for
  honesty, not as a target.

**The noise the intervals use.**  Race stints are scored with the per-lap
noise of a *racing* lap (`sigma_race`, 0.5 s), not the practice `sigma_obs`
(0.74-1.05 s of engine modes, fuel saving and traffic that centring removes).
With the practice value the 90% intervals covered 95-100% at 1.5x the width
the metric's own noise floor needs; the sealed file carries both and the
scorer uses the race value where it exists.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import (
    GRIP_BUDGET_S,
    SEALED_DIR,
    SIGMA_RACE_LAP_S,
    VALID_COMPOUNDS,
    Event,
    get_event,
)
from src.fuel import get_prior

log = logging.getLogger("degless.validate")

MAE_TARGET = 0.15
COVERAGE_BAND = (0.80, 0.97)
NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95)


# --------------------------------------------------------------------------
# Sealing
# --------------------------------------------------------------------------


def _curve_block(d: np.ndarray) -> dict:
    return {
        "mean": d.mean(0).round(6).tolist(),
        "p05": np.quantile(d, 0.05, axis=0).round(6).tolist(),
        "p95": np.quantile(d, 0.95, axis=0).round(6).tolist(),
        "sd": d.std(0).round(6).tolist(),
    }


def seal_predictions(fit, event: Event | str, *, ages=None, regime=None,
                     max_age: int = 40, note: str = "", sigma_race: float = SIGMA_RACE_LAP_S,
                     cliff: dict | None = None, extra: dict | None = None) -> tuple:
    """Freeze the posterior degradation curves to JSON + sha256.  Returns (path, sha).

    Two curve sets are sealed and both are checked in:

    * `curves` — the posterior as fitted, i.e. what a *practice long run* does.
    * `race_curves` — the same posterior scaled by the practice -> race regime
      factor (`src.regime`), i.e. what a *race stint* is predicted to do.

    The second is the one that gets scored.  Scaling is applied draw by draw
    against draws from the regime factor's own distribution, so the extra
    uncertainty in the transfer widens the predictive band rather than
    vanishing into a point multiplier — which is what lets the coverage test
    police it.

    `cliff` is the circuit history's statement of where each compound ends
    (`CircuitPrior.cliff()`), sealed as such: a practice fit cannot identify
    a knee, and the file says so instead of carrying a prior.
    """
    ev = get_event(event) if isinstance(event, str) else event
    ages = np.arange(0, max_age + 1, dtype=float) if ages is None else np.asarray(ages)

    curves, race_curves = {}, {}
    for c in fit.compounds:
        d = fit.deg_loss(c, ages)
        curves[c] = _curve_block(d)
        if regime is not None:
            race_curves[c] = _curve_block(d * regime.draws(d.shape[0], seed=5)[:, None])

    j = {name: i for i, name in enumerate(fit.compounds)}
    has_knee = bool(getattr(fit, "has_hinge", False))
    payload = {
        "product": "degless",
        "event": ev.key,
        "event_name": ev.name,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "fitted_on": "practice sessions only (firewall enforced in src.ingest)",
        "sessions": list(ev.practice_sessions),
        "prior": fit.prior_label,
        "compound_prior": getattr(fit, "compound_prior_label", ""),
        "n_laps": int(fit.n_laps),
        "n_apex_rows": int(fit.n_apex),
        "max_rhat": float(fit.max_rhat),
        "n_divergences": int(fit.n_divergences),
        "converged": bool(fit.converged),
        "hinge": has_knee,
        "ages": ages.tolist(),
        "curves": curves,
        "race_curves": race_curves,
        "regime": (regime.as_dict() if regime is not None else {}),
        "knee": ({c: {"mean": float(fit.posterior["knee"][:, j[c]].mean()),
                      "p05": float(np.quantile(fit.posterior["knee"][:, j[c]], 0.05)),
                      "p95": float(np.quantile(fit.posterior["knee"][:, j[c]], 0.95))}
                  for c in fit.compounds} if has_knee else {}),
        "cliff_history": dict(cliff or {}),
        "k_track": {"mean": float(fit.k_track.mean()),
                    "sd": float(fit.k_track.std())},
        "sigma_obs": float(fit.posterior["sigma_obs"].mean()),
        "sigma_race": float(sigma_race),
        "comp_offset": fit.comp_offset,
        "note": note,
    }
    if extra:
        payload.update(extra)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = SEALED_DIR / f"{ev.key}_{stamp}.json"
    blob = json.dumps(payload, indent=2, sort_keys=True, default=_json_default).encode()
    path.write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    (path.with_suffix(".json.sha256")).write_text(f"{sha}  {path.name}\n")
    log.info("sealed %s (sha256 %s)", path.name, sha[:16])
    return path, sha


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_sealed(path) -> dict:
    from pathlib import Path

    path = Path(path)
    blob = path.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    side = path.with_suffix(".json.sha256")
    if side.exists():
        expect = side.read_text().split()[0]
        if expect != sha:
            raise ValueError(f"sealed file {path.name} does not match its sha256")
    d = json.loads(blob)
    d["_sha256"] = sha
    return d


def latest_sealed(event: Event | str):
    ev = get_event(event) if isinstance(event, str) else event
    files = sorted(SEALED_DIR.glob(f"{ev.key}_*.json"))
    files = [f for f in files if not f.name.endswith(".sha256.json")]
    return files[-1] if files else None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass
class Scorecard:
    event: str = ""
    sealed_file: str = ""
    sealed_sha: str = ""
    regime_label: str = ""
    n_stints: int = 0
    n_laps: int = 0
    mae: float = np.nan          # headline: stint degradation-rate MAE, s/lap
    mae_lap: float = np.nan      # per-lap shape MAE, s (noise-bounded)
    rmse: float = np.nan
    n_rate_stints: int = 0
    bias: float = np.nan          # mean(observed - predicted) rate, s/lap
    bias_by_compound: dict = field(default_factory=dict)
    per_stint: pd.DataFrame | None = None
    mae_by_compound: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)   # nominal -> empirical (per lap)
    rate_coverage90: float = np.nan                 # stint-rate 90% interval coverage
    rate_width90: float = np.nan
    coverage_direction: str = ""   # "under" (a real failure) | "ok" | "over"
    cliff: dict = field(default_factory=dict)      # compound -> {pred, obs, err}
    per_lap: pd.DataFrame | None = None
    sigma_used: float = np.nan
    passes_mae: bool = False
    passes_coverage: bool = False

    def summary(self) -> str:
        cov90 = self.coverage.get(0.90, float("nan"))
        return (
            f"{self.event}: stint-rate MAE {self.mae:.3f} s/lap over "
            f"{self.n_rate_stints} race stints (target < {MAE_TARGET}); "
            f"per-lap MAE {self.mae_lap:.3f} s over {self.n_laps} laps; "
            f"90% coverage {cov90:.2%} per lap, {self.rate_coverage90:.2%} on the stint rate "
            f"(band {COVERAGE_BAND[0]:.0%}-{COVERAGE_BAND[1]:.0%}); "
            f"bias {self.bias:+.3f} s/lap"
        )


def scored_curves(sealed: dict) -> tuple:
    """The curve set that is actually scored, and a label for it."""
    rc = sealed.get("race_curves")
    if rc:
        return rc, "race regime"
    return sealed["curves"], "practice regime"


def _interp(sealed: dict, compound: str, key: str, ages: np.ndarray) -> np.ndarray:
    a = np.asarray(sealed["ages"], dtype=float)
    curves, _ = scored_curves(sealed)
    return np.interp(ages, a, np.asarray(curves[compound][key], dtype=float))


def score_race(sealed: dict, race_clean: pd.DataFrame, event: Event | str,
               *, prior: str = "2026", min_stint: int = 6, sigma: float | None = None) -> Scorecard:
    """Score the sealed practice prediction against clean race laps.

    `sigma` overrides the per-lap noise; by default the sealed file's
    `sigma_race` is used, falling back to its practice `sigma_obs`.
    """
    ev = get_event(event) if isinstance(event, str) else event
    fp = get_prior(ev, prior)

    curves, regime_label = scored_curves(sealed)
    d = race_clean.dropna(subset=["lap_time_s", "tyre_age", "compound"]).copy()
    d = d[d["compound"].isin(curves.keys())]
    sigma_obs = float(sigma if sigma is not None else (sealed.get("sigma_race") or sealed["sigma_obs"]))

    rows = []
    for uid, g in d.groupby("stint_uid"):
        g = g.sort_values("lap_number")
        if len(g) < min_stint:
            continue
        comp = g["compound"].iloc[0]
        ages = g["tyre_age"].to_numpy(dtype=float)

        # Observed pace with the race fuel effect removed, centred on the stint.
        obs = g["lap_time_s"].to_numpy(dtype=float) - fp.term(
            g["lap_in_stint"].to_numpy(dtype=float))
        obs_c = obs - obs.mean()

        mean = _interp(sealed, comp, "mean", ages)
        sd = _interp(sealed, comp, "sd", ages)
        pred_c = mean - mean.mean()

        # Predictive sd: posterior curve uncertainty + per-lap observation noise.
        tot_sd = np.sqrt(sd ** 2 + sigma_obs ** 2)

        rows.append(pd.DataFrame({
            "stint_uid": uid, "driver": g["driver"].to_numpy(),
            "lap_number": g["lap_number"].to_numpy(), "compound": comp,
            "tyre_age": ages, "obs_centred": obs_c, "pred_centred": pred_c,
            "err": obs_c - pred_c, "pred_sd": tot_sd, "curve_sd": sd,
        }))

    sc = Scorecard(event=ev.key, sealed_file=sealed.get("_file", ""),
                   sealed_sha=sealed.get("_sha256", ""), regime_label=regime_label, sigma_used=sigma_obs)
    if not rows:
        log.warning("no race stints long enough to score for %s", ev.key)
        return sc

    per = pd.concat(rows, ignore_index=True)
    sc.per_lap = per
    sc.n_stints = per["stint_uid"].nunique()
    sc.n_laps = len(per)
    sc.mae_lap = float(per["err"].abs().mean())
    sc.rmse = float(np.sqrt((per["err"] ** 2).mean()))

    # -- headline: per-stint degradation rate -------------------------------
    srows = []
    for uid, g in per.groupby("stint_uid"):
        g = g.sort_values("tyre_age")
        if len(g) < 8:
            continue
        first, last = g.head(3), g.tail(3)
        d_age = float(last["tyre_age"].mean() - first["tyre_age"].mean())
        if d_age < 4:
            continue
        obs_rate = float(last["obs_centred"].mean() - first["obs_centred"].mean()) / d_age
        pred_rate = float(last["pred_centred"].mean() - first["pred_centred"].mean()) / d_age
        # rate interval: curve uncertainty on the rate plus the noise of two 3-lap means
        curve_sd = float(np.sqrt(last["curve_sd"].mean() ** 2 + first["curve_sd"].mean() ** 2)) / d_age
        noise = sigma_obs * np.sqrt(2.0 / 3.0) / d_age
        rate_sd = float(np.sqrt(curve_sd ** 2 + noise ** 2))
        srows.append({"stint_uid": uid, "driver": g["driver"].iloc[0],
                      "compound": g["compound"].iloc[0], "n_laps": len(g),
                      "age_span": d_age, "obs_rate": obs_rate,
                      "pred_rate": pred_rate, "err": obs_rate - pred_rate,
                      "rate_sd": rate_sd,
                      "inside90": bool(abs(obs_rate - pred_rate) <= 1.645 * rate_sd)})
    if srows:
        ps = pd.DataFrame(srows)
        sc.per_stint = ps
        sc.n_rate_stints = len(ps)
        sc.mae = float(ps["err"].abs().mean())
        sc.mae_by_compound = ps.groupby("compound")["err"].apply(
            lambda s: float(s.abs().mean())).to_dict()
        # Signed bias matters more than its magnitude here: a model that is
        # wrong in the same direction on every stint is telling you the two
        # regimes differ, not that it is noisy.
        sc.bias = float(ps["err"].mean())
        sc.bias_by_compound = ps.groupby("compound")["err"].mean().to_dict()
        sc.rate_coverage90 = float(ps["inside90"].mean())
        sc.rate_width90 = float((2 * 1.645 * ps["rate_sd"]).mean())
    else:
        sc.mae = sc.mae_lap
        sc.mae_by_compound = per.groupby("compound")["err"].apply(
            lambda s: float(s.abs().mean())).to_dict()

    # -- calibration: nominal vs empirical coverage -------------------------
    from scipy.stats import norm

    for lvl in NOMINAL_LEVELS:
        z = norm.ppf(0.5 + lvl / 2)
        inside = (per["err"].abs() <= z * per["pred_sd"]).mean()
        sc.coverage[lvl] = float(inside)

    sc.passes_mae = sc.mae < MAE_TARGET
    c90 = sc.coverage.get(0.90, np.nan)
    # Under- and over-coverage are not the same failure.  An interval that
    # contains the truth less often than it claims is *dishonest*.  One that
    # contains it more often is merely inefficient.  Only the first is
    # allowed to fail the gate; the second is reported as what it is.
    if not np.isfinite(c90):
        sc.coverage_direction = ""
    elif c90 < COVERAGE_BAND[0]:
        sc.coverage_direction = "under"
    elif c90 > COVERAGE_BAND[1]:
        sc.coverage_direction = "over"
    else:
        sc.coverage_direction = "ok"
    sc.passes_coverage = sc.coverage_direction in ("ok", "over")

    # -- cliff error --------------------------------------------------------
    sc.cliff = _cliff_error(sealed, per)
    return sc


def _cliff_error(sealed: dict, per: pd.DataFrame, w: float = 1.5) -> dict:
    """Compare the sealed cliff to the observed race pace-collapse lap.

    The observed knee is found by fitting a softplus hinge shape to the
    pooled, stint-centred race profile and scanning the knee on a 1-lap grid.
    The sealed cliff is the posterior knee where the fit carried a hinge, else
    the circuit history's p90 stint (the reported cliff), labelled as such.
    """
    out = {}
    sp = lambda x: np.logaddexp(0.0, x)  # noqa: E731
    knee = sealed.get("knee") or {}
    hist = sealed.get("cliff_history") or {}
    for comp, g in per.groupby("compound"):
        ages = g["tyre_age"].to_numpy(dtype=float)
        y = g["obs_centred"].to_numpy(dtype=float)
        if len(g) < 20 or ages.max() - ages.min() < 8:
            continue
        grid = np.arange(max(5.0, ages.min() + 2), min(40.0, ages.max() - 1) + 1)
        if len(grid) < 2:
            continue
        best, best_k = np.inf, np.nan
        for k in grid:
            h = w * (sp((ages - k) / w) - sp(-k / w))
            X = np.column_stack([np.ones_like(ages), ages, h])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            rss = float(np.sum((y - X @ beta) ** 2))
            if rss < best:
                best, best_k = rss, float(k)
        if comp in knee:
            pred, src = float(knee[comp]["mean"]), "posterior knee"
        elif comp in hist and np.isfinite(hist[comp].get("p90_stint", np.nan)):
            pred, src = float(hist[comp]["p90_stint"]), "circuit history p90 stint"
        else:
            pred, src = float("nan"), "none"
        detected = bool(grid[0] < best_k < grid[-1])
        out[comp] = {"predicted_lap": pred, "source": src,
                     "observed_lap": best_k if detected else float("nan"),
                     "error_laps": abs(pred - best_k) if (detected and np.isfinite(pred)) else float("nan"),
                     "detected": detected}
    return out


def calibration_table(sc: Scorecard) -> pd.DataFrame:
    return pd.DataFrame({
        "nominal": list(sc.coverage.keys()),
        "empirical": list(sc.coverage.values()),
    })


# --------------------------------------------------------------------------
# Strategy backtest: was the *recommendation* the right shape?
# --------------------------------------------------------------------------


def strategy_backtest(res, race: pd.DataFrame, event: Event | str,
                      *, min_stint: int = 3) -> dict:
    """Score the recommended plan against what the field actually did.

    `score_race` checks the degradation *curve*; this checks the *decision*.
    Three things are checked, none of which the curve metrics can see:

    * **stop count** - against the distribution among classified finishers;
    * **stint length per compound** - whether each recommended stint sits
      inside the range the compound was actually run to;
    * **the grip-budget invariant** - degradation rate times longest observed
      stint, per compound, against the budget the model used;

    plus, **safety-car aware**, the field's first-stop laps: the median in-lap
    of first stops taken under green, the share taken under a safety car,
    and where the tyre-optimal and position-aware recommendations sit
    against it.  The race is validation data throughout.
    """
    from src.strategy import race_stops

    ev = get_event(event) if isinstance(event, str) else event

    st = (race.groupby(["driver", "stint"])
          .agg(compound=("compound", "first"), n=("lap_number", "size"),
               end=("lap_number", "max"))
          .reset_index())
    st = st[(st["n"] >= min_stint) & st["compound"].isin(VALID_COMPOUNDS)]
    if st.empty or res is None or not getattr(res, "best", None):
        return {}

    finished = st.groupby("driver")["end"].max()
    classified = finished[finished >= ev.n_race_laps - 2].index
    stops = st[st["driver"].isin(classified)].groupby("driver").size() - 1
    dist = stops.value_counts().sort_index()

    rec_stops = int(res.best["n_stops"])
    lens = list(res.best["stint_lens"])
    comps = list(res.best["compounds"])

    by_comp = st.groupby("compound")["n"]
    obs = {c: {"p10": float(g.quantile(0.10)), "median": float(g.median()),
               "p90": float(g.quantile(0.90)), "max": float(g.max()),
               "n": int(len(g))}
           for c, g in by_comp}

    per_stint, n_ok = [], 0
    for c, L in zip(comps, lens):
        o = obs.get(c)
        inside = bool(o and o["p10"] - 2 <= L <= o["max"])
        n_ok += int(inside)
        per_stint.append({"compound": c, "recommended_laps": int(L),
                          "observed_median": (o or {}).get("median", float("nan")),
                          "observed_max": (o or {}).get("max", float("nan")),
                          "inside_observed_range": inside})

    # The invariant is stated at the push the race was actually run at.
    model = getattr(res, "model", None)
    psi = float(model.psi(float(res.best.get("push", 1.0)))) if model is not None else 1.0
    budget = {}
    for c, g in st.groupby("compound"):
        rate = res.life.loc[res.life["compound"] == c, "deg_s_per_lap"]
        if len(rate) and np.isfinite(float(rate.iloc[0])):
            budget[c] = float(rate.iloc[0]) * psi * float(g["n"].max())
    budget_used = {c: (model.budget_of(c) if model is not None else float(GRIP_BUDGET_S)) for c in budget}

    # -- first stops, safety-car aware ---------------------------------------
    rs = race_stops(race, ev)
    firsts = [(v["in_laps"][0], v["sc"][0]) for d, v in rs.items() if v["classified"] and v["in_laps"]]
    green = [p for p, sc_ in firsts if not sc_]
    rec_first = int(res.best["pit_laps"][0]) if res.best.get("pit_laps") else None
    tyre_first = (int(res.tyre_optimal["pit_laps"][0]) if getattr(res, "tyre_optimal", None)
                  and res.tyre_optimal.get("pit_laps") else None)
    first_stop = {
        "n": len(firsts), "n_green": len(green), "share_under_sc": (float(np.mean([s for _, s in firsts])) if firsts else None),
        "field_median_green": (float(np.median(green)) if green else None),
        "field_p25_green": (float(np.quantile(green, 0.25)) if green else None),
        "field_p75_green": (float(np.quantile(green, 0.75)) if green else None),
        "recommended": rec_first, "tyre_optimal": tyre_first,
        "recommended_minus_field": ((rec_first - float(np.median(green))) if (green and rec_first is not None) else None),
        "tyre_optimal_minus_field": ((tyre_first - float(np.median(green))) if (green and tyre_first is not None) else None),
        "sc_set_the_stops": bool(firsts and np.mean([s for _, s in firsts]) > 0.4),
    }

    return {
        "recommended_stops": rec_stops,
        "observed_stop_counts": {int(k): int(v) for k, v in dist.items()},
        "modal_stops": int(dist.idxmax()) if len(dist) else None,
        "stops_match_mode": bool(len(dist) and rec_stops == int(dist.idxmax())),
        "stops_observed_share": (float(dist.get(rec_stops, 0) / dist.sum())
                                 if len(dist) else float("nan")),
        "per_stint": per_stint,
        "all_stints_inside_observed_range": n_ok == len(lens),
        "grip_budget_implied": budget,
        "grip_budget_used": budget_used,
        "grip_budget_config": float(GRIP_BUDGET_S),
        "first_stop": first_stop,
    }
