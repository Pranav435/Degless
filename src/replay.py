"""Pre-computed per-lap replay states.

The slider in the app is pure playback of a parquet computed offline.  It looks
live and it cannot break on stage.

At lap L of a stint the state is what a race engineer would have had on the
pit wall at that moment: the stint's pace level estimated from the laps run so
far, an uncertainty band that shrinks as evidence accumulates, the model's
projection for the next few laps, and a cliff alarm.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, GRIP_BUDGET_S, Event, get_event
from src.fuel import get_prior
from src.tyre import TyreModel

log = logging.getLogger("degless.replay")

LOOKAHEAD = 5  # laps projected ahead of the current lap
CLIFF_ALARM_P = 0.5  # P(tyre has spent its grip budget) above which the alarm fires


def build_replay(fit, event: Event | str, race: pd.DataFrame, *,
                 prior: str = "2026", n_draws: int = 400,
                 push: float = 0.7) -> pd.DataFrame:
    """One row per (driver, lap) with the sequential state at that lap.

    The cliff alarm is the probability that the tyre has spent its grip budget,
    computed from the same wear model the strategy optimiser uses, so the
    replay and the recommendation cannot disagree about when a tyre is done.
    It used to read the fitted `knee` parameter instead, which is the one part
    of the curve practice cannot identify - long runs stop at age 15-22 laps,
    before any compound reaches its cliff, so that posterior was very close to
    its prior and the alarm was largely reporting an assumption.
    """
    ev = get_event(event) if isinstance(event, str) else event
    fp = get_prior(ev, prior)

    total = fit.posterior["lin"].shape[0]
    idx = np.linspace(0, total - 1, min(n_draws, total)).astype(int)
    model = TyreModel.from_fit(fit, draws=idx, budget=GRIP_BUDGET_S)
    from src.tyre import load_profile, wear_multiplier
    lf = load_profile(ev)
    psi = float(wear_multiplier(push))
    sigma_obs = float(fit.posterior["sigma_obs"].mean())

    rows = []
    for uid, g in race.groupby("stint_uid"):
        g = g.sort_values("lap_number")
        comp = g["compound"].iloc[0]
        if comp not in fit.compounds:
            continue
        ages = g["tyre_age"].to_numpy(dtype=float)
        laps = g["lap_number"].to_numpy(dtype=float)
        drv = g["driver"].iloc[0]
        clean = g["is_clean"].to_numpy() if "is_clean" in g else np.ones(len(g), bool)

        # Pace with the race fuel effect removed; the model explains what's left.
        pace = g["lap_time_s"].to_numpy(dtype=float) - fp.term(
            g["lap_in_stint"].to_numpy(dtype=float))
        deg = fit.deg_loss(comp, ages)[idx]          # (draws, n_laps)
        deg_mean = deg.mean(0)

        seen_sum, seen_n = 0.0, 0
        for i in range(len(g)):
            if clean[i] and np.isfinite(pace[i]):
                seen_sum += pace[i] - deg_mean[i]
                seen_n += 1
            level = seen_sum / seen_n if seen_n else np.nan
            # Level uncertainty shrinks as 1/sqrt(n) of the laps seen so far.
            level_sd = sigma_obs / np.sqrt(seen_n) if seen_n else np.nan

            k = min(i + LOOKAHEAD, len(g) - 1)
            proj = level + deg_mean[k] if seen_n else np.nan
            proj_sd = float(np.sqrt(deg[:, k].var() + (level_sd ** 2 if seen_n else 0.0)))

            # Wear accumulated over the laps this stint has actually run,
            # weighted by the fuel load the car carried on each of them.
            lap_idx = np.clip(laps[:i + 1].astype(int), 1, ev.n_race_laps) - 1
            wear = model.wear_rate[comp] * psi * lf[lap_idx].sum()
            p_cliff = float((wear >= 1.0).mean())
            rows.append({
                "event": ev.key, "driver": drv, "stint_uid": uid,
                "lap_number": laps[i], "tyre_age": ages[i], "compound": comp,
                "lap_time_s": g["lap_time_s"].to_numpy(dtype=float)[i],
                "is_clean": bool(clean[i]),
                "laps_seen": seen_n,
                "stint_pace_est": level,
                "stint_pace_sd": level_sd,
                "band_lo": level - 1.645 * level_sd if seen_n else np.nan,
                "band_hi": level + 1.645 * level_sd if seen_n else np.nan,
                "proj_lap": laps[k],
                "proj_pace": proj,
                "proj_lo": proj - 1.645 * proj_sd if seen_n else np.nan,
                "proj_hi": proj + 1.645 * proj_sd if seen_n else np.nan,
                "deg_now_s": float(deg_mean[i]),
                "p_past_cliff": p_cliff,
                "cliff_alarm": bool(p_cliff > CLIFF_ALARM_P),
                "is_pit_in": bool(g["pit_in"].to_numpy()[i]),
                "is_pit_out": bool(g["pit_out"].to_numpy()[i]),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["lap_number", "driver"]).reset_index(drop=True)


def replay_path(event: Event | str):
    ev = get_event(event) if isinstance(event, str) else event
    return DATA_PROCESSED / f"replay_{ev.key}.parquet"


def save_replay(df: pd.DataFrame, event: Event | str):
    p = replay_path(event)
    df.to_parquet(p, index=False)
    return p


def load_replay(event: Event | str) -> pd.DataFrame:
    p = replay_path(event)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()
