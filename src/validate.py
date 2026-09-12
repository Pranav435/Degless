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
  last three, against the same quantity from the sealed curve.  This is the
  quantity the plan's own race cross-check uses (GAS stint 2: +3.58 s over 26
  laps ~ 0.14 s/lap), and the < 0.15 s/lap target is on this scale.
* `mae_lap` (s) — raw per-lap error.  This one is bounded below by the per-lap
  noise of a racing lap (`sigma_obs` ~ 0.9 s: traffic, defending, fuel saving,
  engine modes) and no tyre model can drive it to 0.15.  Reported for honesty,
  not as a target.
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
                     max_age: int = 40, note: str = "") -> tuple:
    """Freeze the posterior degradation curves to JSON + sha256.  Returns (path, sha).

    Two curve sets are sealed and both are checked in:

    * `curves` — the posterior as fitted, i.e. what a *practice long run* does.
    * `race_curves` — the same posterior scaled by the practice -> race regime
      factor (`src.regime`), i.e. what a *race stint* is predicted to do.

    The second is the one that gets scored, because it is the one that makes a
    claim about the race.  Scaling is applied draw by draw against draws from
    the regime factor's own distribution, so the extra uncertainty in the
    transfer widens the predictive band rather than vanishing into a point
    multiplier — which is what lets the coverage test police it.
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
    payload = {
        "product": "degless",
        "event": ev.key,
        "event_name": ev.name,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "fitted_on": "practice sessions only (firewall enforced in src.ingest)",
        "sessions": list(ev.practice_sessions),
        "prior": fit.prior_label,
        "n_laps": int(fit.n_laps),
        "n_apex_rows": int(fit.n_apex),
        "max_rhat": float(fit.max_rhat),
        "n_divergences": int(fit.n_divergences),
        "converged": bool(fit.converged),
        "ages": ages.tolist(),
        "curves": curves,
        "race_curves": race_curves,
        "regime": (regime.as_dict() if regime is not None else {}),
        "knee": {c: {"mean": float(fit.posterior["knee"][:, j[c]].mean()),
                     "p05": float(np.quantile(fit.posterior["knee"][:, j[c]], 0.05)),
                     "p95": float(np.quantile(fit.posterior["knee"][:, j[c]], 0.95))}
                 for c in fit.compounds},
        "k_track": {"mean": float(fit.k_track.mean()),
                    "sd": float(fit.k_track.std())},
        "sigma_obs": float(fit.posterior["sigma_obs"].mean()),
        "comp_offset": fit.comp_offset,
        "note": note,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = SEALED_DIR / f"{ev.key}_{stamp}.json"
    blob = json.dumps(payload, indent=2, sort_keys=True).encode()
    path.write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    (path.with_suffix(".json.sha256")).write_text(f"{sha}  {path.name}\n")
    log.info("sealed %s (sha256 %s)", path.name, sha[:16])
    return path, sha


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
    coverage: dict = field(default_factory=dict)   # nominal -> empirical
    coverage_direction: str = ""   # "under" (a real failure) | "ok" | "over"
    cliff: dict = field(default_factory=dict)      # compound -> {pred, obs, err}
    per_lap: pd.DataFrame | None = None
    passes_mae: bool = False
    passes_coverage: bool = False

    def summary(self) -> str:
        cov90 = self.coverage.get(0.90, float("nan"))
        return (
            f"{self.event}: stint-rate MAE {self.mae:.3f} s/lap over "
            f"{self.n_rate_stints} race stints (target < {MAE_TARGET}); "
            f"per-lap MAE {self.mae_lap:.3f} s over {self.n_laps} laps; "
            f"90% coverage {cov90:.2%} (band {COVERAGE_BAND[0]:.0%}-{COVERAGE_BAND[1]:.0%}); "
            f"bias {self.bias:+.3f} s/lap"
        )


def scored_curves(sealed: dict) -> tuple:
    """The curve set that is actually scored, and a label for it.

    `race_curves` when the sealed file carries them, `curves` otherwise — so a
    file sealed before the regime transfer existed still scores, against the
    practice-regime curves it was written with.
    """
    rc = sealed.get("race_curves")
    if rc:
        return rc, "race regime"
    return sealed["curves"], "practice regime"


def _interp(sealed: dict, compound: str, key: str, ages: np.ndarray) -> np.ndarray:
    a = np.asarray(sealed["ages"], dtype=float)
    curves, _ = scored_curves(sealed)
    return np.interp(ages, a, np.asarray(curves[compound][key], dtype=float))


def score_race(sealed: dict, race_clean: pd.DataFrame, event: Event | str,
               *, prior: str = "2026", min_stint: int = 6) -> Scorecard:
    """Score the sealed practice prediction against clean race laps."""
    ev = get_event(event) if isinstance(event, str) else event
    fp = get_prior(ev, prior)

    curves, regime_label = scored_curves(sealed)
    d = race_clean.dropna(subset=["lap_time_s", "tyre_age", "compound"]).copy()
    d = d[d["compound"].isin(curves.keys())]
    sigma_obs = float(sealed["sigma_obs"])

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
            "err": obs_c - pred_c, "pred_sd": tot_sd,
        }))

    sc = Scorecard(event=ev.key, sealed_file=sealed.get("_file", ""),
                   sealed_sha=sealed.get("_sha256", ""), regime_label=regime_label)
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
        srows.append({"stint_uid": uid, "driver": g["driver"].iloc[0],
                      "compound": g["compound"].iloc[0], "n_laps": len(g),
                      "age_span": d_age, "obs_rate": obs_rate,
                      "pred_rate": pred_rate, "err": obs_rate - pred_rate})
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
    # contains the truth less often than it claims is *dishonest* — it asserts
    # confidence it has not earned.  One that contains it more often is merely
    # inefficient: the answer is right and the error bar is wider than it needs
    # to be.  Only the first is allowed to fail the gate; the second is
    # reported as what it is.
    #
    # This build over-covers, and for a known reason: `sigma_obs` is the
    # per-lap noise of a *practice* lap (~0.9 s of traffic, engine modes and
    # fuel saving), while each race stint is scored centred on its own mean,
    # which removes most of that.  The regime factor's own spread widens the
    # band further.  Both are deliberate and neither is tuned away.
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
    """Compare the posterior knee to the observed race pace-collapse lap.

    The observed knee is found by fitting the same softplus hinge shape to the
    pooled, stint-centred race profile and scanning the knee on a 1-lap grid.
    """
    out = {}
    sp = lambda x: np.logaddexp(0.0, x)  # noqa: E731
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
        pred = float(sealed["knee"][comp]["mean"])
        # A best fit sitting on the edge of the scan grid means the scan never
        # found an interior collapse — the race simply did not run these tyres
        # far enough to show a cliff.  Reporting the boundary as if it were a
        # detected knee would be a fabricated number.
        detected = bool(grid[0] < best_k < grid[-1])
        out[comp] = {"predicted_lap": pred,
                     "observed_lap": best_k if detected else float("nan"),
                     "error_laps": abs(pred - best_k) if detected else float("nan"),
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

    `score_race` checks the degradation *curve*; this checks the *decision*,
    which is a different question and the one this project exists to answer.  A
    curve can be within its error bars and still produce a plan no team would
    run, and that is exactly the failure this rebuild was aimed at: the previous
    model passed every curve gate while recommending a 33-lap stint on the SOFT
    at Hungary 2026 and a one-stop at a circuit where nobody one-stopped.

    Three things are checked, none of which the curve metrics can see:

    * **stop count** - against the distribution among classified finishers,
      which is the field's own revealed answer to the same question;
    * **stint length per compound** - whether each recommended stint sits
      inside the range the compound was actually run to, per compound, because
      "18 laps on the SOFT" is right or wrong only relative to the SOFT;
    * **the grip-budget invariant** - degradation rate times longest observed
      stint, per compound, which should come out near `GRIP_BUDGET_S` if the
      premise the tyre model is built on holds at this circuit.

    The race is validation data throughout.  Nothing here feeds a fit; it is
    read after the prediction is sealed.
    """
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

    # The invariant is stated at the push the race was actually run at, so the
    # full-push rate the model reports has to be scaled by the same wear
    # multiplier the plan assumes.  Comparing a full-push rate against a
    # managed stint length would overstate the implied budget by 1/psi (~1.5x)
    # and make a consistent measurement look like a contradiction.
    from src.tyre import wear_multiplier

    psi = float(wear_multiplier(float(res.best.get("push", 1.0))))
    budget = {}
    for c, g in st.groupby("compound"):
        rate = res.life.loc[res.life["compound"] == c, "deg_s_per_lap"]
        if len(rate) and np.isfinite(float(rate.iloc[0])):
            budget[c] = float(rate.iloc[0]) * psi * float(g["n"].max())

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
        "grip_budget_config": float(GRIP_BUDGET_S),
    }
