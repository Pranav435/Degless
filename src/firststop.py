"""The first-stop prior: where this circuit has always made teams stop.

The first stop is the one decision a strategy model gets graded on hardest and
the one it has least information about.  Everything after it is re-decidable —
the pit wall sees the race and re-optimises — but the first stop is taken out
of a clean-air, full-fuel state that practice never reproduces, and by then the
model's only evidence is a degradation curve fitted on 15-22 lap practice runs.
V2's answer came purely from the cost surface: pit loss against cumulative
degradation, nudged by an undercut-exposure term.  On the three 2026 weekends
whose first stops were not safety-car-set it was 4-9 laps late.

What it was missing is the thing every strategist has: the circuit's own
history.  Barcelona's field has stopped at a median lap 13.5 (2023) and 16.5
(2025) scaled to the 2026 distance, and did stop at 13 in 2026.  That is
information, and it is available before the weekend starts.

**Why a density and not a target lap.**  The between-year spread is +/-4 laps,
so the honest object is a distribution, not a number.  Pushing the optimiser
toward a *lap* would be worse than leaving it alone: it would override the cost
surface on a weekend whose tyres genuinely behave differently.  A density added
to the objective as `kappa * neglogp[lap]` seconds is a soft prior — it costs
nothing at the modal lap, a few tenths across the plausible window, and several
seconds only where the circuit has never stopped.  `kappa` (seconds per nat) is
calibrated leave-one-out like every other constant, so a weekend whose history
is unhelpful can set it near zero.

**Four ingredients.**

    a Gaussian KDE over every scaled historical green first stop;
    the same KDE restricted to stops that opened a plan with the same number of
      *stops*, mixed in with weight n_s / (n_s + min_compound_n);
    the same again restricted to the same start compound *and* stop count,
      mixed in with weight n_cs / (n_cs + min_compound_n);
    a uniform floor at weight `floor` over the legal pit window, so a lap the
    circuit has never stopped on is expensive, not impossible.

**Why condition on the stop count.**  A first stop is the opening move of a
plan, not a free-standing decision: Spa's 2024 two-stoppers boxed on lap 11 and
its one-stoppers on lap 19, and an unconditional density over both has its mode
between them, belonging to neither.  Charged against a one-stop plan that
density is simply wrong — it pulls the stop toward a lap that only makes sense
if you are stopping again — and the benchmark caught exactly that: Belgium's
one-stop M-S was dragged to lap 12 by two-stop history, Australia's to 18.  The
fix is the same structure the plan prior already has: condition on the family.
`by_stops` needs no nomination mapping (how often the field stops here is a fact
about the pit lane); `by_compound_stops` does, which is why it is keyed by the
*target* year's letter.

The compound matters for the same reason — a MEDIUM start is a different
decision from a SOFT start, and it is the 2026 nomination's letter that counts,
which is why `CircuitPrior.first_stop_green` maps the history through the
Pirelli nominations first — so the finest cell is (compound, stops) and the
coarser ones back it off when it is thin.

Only green-flag first stops enter.  A safety-car stop is a cheap stop the race
handed the team, it is not evidence about tyre life or about the trade-off the
model is pricing, and four of the seven 2026 weekends were safety-car set.
"""

from __future__ import annotations

import numpy as np

__all__ = ["first_stop_prior", "first_stop_penalty_table", "first_stop_summary"]

BANDWIDTH_LAPS = 2.5         # at a 60-lap race; scaled with the distance
REFERENCE_RACE_LAPS = 60.0
UNIFORM_FLOOR = 0.05         # share of the density a lap with no history still gets
MIN_COMPOUND_N = 5           # start-compound stops needed before that KDE gets half the weight
WINDOW_MARGIN = 6            # laps from either end the strategy search will not stop inside anyway


def _clean_laps(values, n_race_laps: int) -> np.ndarray:
    x = np.asarray([v for v in (values or []) if v is not None], dtype=float)
    if not x.size:
        return x
    x = x[np.isfinite(x)]
    return np.clip(x, 1.0, float(n_race_laps))


def _stops_laps(by_stops: dict | None, n_stops, n_race_laps: int) -> np.ndarray:
    """The in-laps of first stops that opened an `n_stops`-stop plan.

    Keys arrive as strings from JSON and as ints from a freshly built
    `CircuitPrior`; both are accepted rather than pinning one, because a silent
    key miss would leave the prior unconditional with nothing to notice.
    """
    if not by_stops or n_stops is None:
        return np.array([])
    try:
        k = int(n_stops)
    except (TypeError, ValueError):
        return np.array([])
    for key in (str(k), k):
        if key in by_stops:
            return _clean_laps(by_stops[key], n_race_laps)
    return np.array([])


def _compound_stops_laps(by_cs: dict | None, start_compound: str | None, n_stops,
                         n_race_laps: int) -> np.ndarray:
    """The `"<letter>|<n_stops>"` cell, however the history spelled the compound."""
    if not by_cs or not start_compound or n_stops is None:
        return np.array([])
    try:
        k = int(n_stops)
    except (TypeError, ValueError):
        return np.array([])
    c = str(start_compound).upper()
    for letter in (c[:1], c):
        key = f"{letter}|{k}"
        if key in by_cs:
            return _clean_laps(by_cs[key], n_race_laps)
    for key, v in by_cs.items():
        parts = str(key).split("|")
        if len(parts) == 2 and parts[0].upper()[:1] == c[:1] and parts[1] == str(k):
            return _clean_laps(v, n_race_laps)
    return np.array([])


def _compound_laps(by_compound: dict | None, start_compound: str | None, n_race_laps: int) -> np.ndarray:
    """The start-compound's in-laps, however the history keyed them.

    `CircuitPrior.first_stop_green["by_compound"]` is built by mapping each
    historical stint's compound through the Pirelli nominations, and
    `nominations.map_sequence` returns *letters* ("M"), while everything else in
    the model speaks full names ("MEDIUM").  Both are accepted here rather than
    pinning one: a silent key miss would leave the prior field-wide with no
    error to notice, which is the worst of the three outcomes.
    """
    if not by_compound or not start_compound:
        return np.array([])
    c = str(start_compound).upper()
    for key in (c, c[:1]):
        if key in by_compound:
            return _clean_laps(by_compound[key], n_race_laps)
    for k, v in by_compound.items():
        if str(k).upper()[:1] == c[:1]:
            return _clean_laps(v, n_race_laps)
    return np.array([])


def _kde(x: np.ndarray, laps: np.ndarray, h: float) -> np.ndarray:
    """Normalised Gaussian KDE of `x` evaluated on the integer grid `laps`."""
    u = (laps[None, :] - x[:, None]) / float(h)
    d = np.exp(-0.5 * u ** 2).sum(axis=0)
    s = d.sum()
    return d / s if s > 0 else np.full(len(laps), 1.0 / len(laps))


def _quantile(laps: np.ndarray, p: np.ndarray, q: float) -> float:
    c = np.cumsum(p)
    return float(laps[int(np.searchsorted(c, q * c[-1], side="left"))])


def first_stop_prior(first_stop_green: dict | None, n_race_laps: int, *,
                     start_compound: str | None = None,
                     n_stops: int | None = None,
                     bandwidth_laps: float = BANDWIDTH_LAPS,
                     floor: float = UNIFORM_FLOOR,
                     min_compound_n: int = MIN_COMPOUND_N,
                     margin: int = WINDOW_MARGIN) -> dict | None:
    """A density over the first-stop lap, as a per-lap penalty in nats.

    `first_stop_green` is `CircuitPrior.first_stop_green`: the circuit's
    green-flag first stops pooled over years and already scaled to this event's
    distance, with `in_laps` and (optionally) `by_compound`, `by_stops` and
    `by_compound_stops`.  Absent or empty history returns `None` — no prior, no
    penalty, exactly V2's objective.

    The mixture backs off from the finest cell to the coarsest, each stage
    weighted by its own count: all stops → the same stop count → the same start
    compound *and* stop count.  `n_stops=None` reproduces the unconditional
    compound-only prior, so every older caller still gets what it asked for.

    `neglogp` is shifted so the modal lap costs nothing; it is what the
    objective multiplies by `kappa` seconds per nat.  Laps outside
    `[margin, n_race_laps - margin]` carry the floor's penalty: the strategy
    search does not place stops there, so those entries exist only to keep the
    array indexable by lap number without an out-of-range branch.
    """
    if not first_stop_green:
        return None
    n = int(n_race_laps)
    x_all = _clean_laps(first_stop_green.get("in_laps"), n)
    if not x_all.size:
        return None
    laps = np.arange(1, n + 1)
    h = float(bandwidth_laps) * n / REFERENCE_RACE_LAPS

    p = _kde(x_all, laps, h)
    n_comp = n_stop = n_cs = 0
    w_stop = w_cs = 0.0
    if n_stops is None:
        # the old behaviour: compound only
        if start_compound:
            x_c = _compound_laps(first_stop_green.get("by_compound"), start_compound, n)
            n_comp = int(x_c.size)
            if n_comp:
                w = n_comp / (n_comp + float(min_compound_n))
                p = (1.0 - w) * p + w * _kde(x_c, laps, h)
    else:
        x_s = _stops_laps(first_stop_green.get("by_stops"), n_stops, n)
        n_stop = int(x_s.size)
        if n_stop:
            w_stop = n_stop / (n_stop + float(min_compound_n))
            p = (1.0 - w_stop) * p + w_stop * _kde(x_s, laps, h)
        x_cs = _compound_stops_laps(first_stop_green.get("by_compound_stops"),
                                    start_compound, n_stops, n)
        n_cs = int(x_cs.size)
        if n_cs:
            w_cs = n_cs / (n_cs + float(min_compound_n))
            p = (1.0 - w_cs) * p + w_cs * _kde(x_cs, laps, h)
        if start_compound and not n_cs:
            # No car has opened an n-stop plan on this compound here, so the
            # compound's own history is the best remaining evidence about the
            # start - back off to it rather than to the field-wide density.
            x_c = _compound_laps(first_stop_green.get("by_compound"), start_compound, n)
            n_comp = int(x_c.size)
            if n_comp:
                w = n_comp / (n_comp + float(min_compound_n))
                p = (1.0 - w) * p + w * _kde(x_c, laps, h)

    lo, hi = int(margin), int(n - margin)
    if hi <= lo:
        lo, hi = 1, n
    inside = (laps >= lo) & (laps <= hi)
    unif = np.where(inside, 1.0 / max(int(inside.sum()), 1), 0.0)
    f = float(np.clip(floor, 1e-6, 0.5))
    p = (1.0 - f) * p + f * unif
    # Outside the legal window the mixture has no floor to stand on and the KDE
    # can be arbitrarily small; pin those laps at the floor's own level so the
    # penalty is bounded everywhere.
    p = np.where(inside, p, f / max(int(inside.sum()), 1))

    neglogp = -np.log(np.clip(p, 1e-300, None))
    neglogp = neglogp - neglogp.min()
    mode = int(laps[int(np.argmax(p))])
    src = (f"Gaussian KDE (bandwidth {h:.1f} laps) over {x_all.size} historical green first stops "
           f"scaled to {n} laps")
    if n_stop:
        src += f", blended {w_stop:.2f} toward the {n_stop} that opened a {int(n_stops)}-stop plan"
    if n_cs:
        src += f", then {w_cs:.2f} toward the {n_cs} {start_compound}-start {int(n_stops)}-stop ones"
    if n_comp:
        w = n_comp / (n_comp + float(min_compound_n))
        src += f", blended {w:.2f} toward the {n_comp} {start_compound}-start stops"
    src += f", with a {f:.0%} uniform floor on laps {lo}-{hi}"
    return {"laps": laps, "neglogp": neglogp,
            "mode_lap": mode,
            "median_lap": _quantile(laps, p, 0.50),
            "p25": _quantile(laps, p, 0.25),
            "p75": _quantile(laps, p, 0.75),
            "n": int(x_all.size), "n_compound": n_comp,
            "n_stops": (None if n_stops is None else int(n_stops)),
            "n_by_stops": n_stop, "n_by_compound_stops": n_cs, "source": src}


def first_stop_penalty_table(first_stop_green: dict | None, n_race_laps: int,
                             compounds, n_stops=(1, 2, 3), **kw) -> dict | None:
    """`table[compound][n_stops][lap]` — nats for stopping on `lap` first.

    One array per (start compound, stop count), each of shape
    `(n_race_laps + 1,)` so a caller can index it with the in-lap straight out of
    a plan; index 0 is unused and zero.  `table[compound]["any"]` is the
    unconditional (compound-only) density, which is what a caller that does not
    know the plan's stop count — or one whose stop count is outside `n_stops` —
    falls back to.  This is the object the objective and the live engine take:
    `kappa * table[seq[0]][len(seq) - 1][pit_laps[0]]`.
    """
    if not first_stop_green:
        return None
    n = int(n_race_laps)

    def arr(pr):
        t = np.zeros(n + 1, dtype=float)
        t[1:] = pr["neglogp"]
        return t

    out = {}
    for c in list(compounds or []):
        pr_any = first_stop_prior(first_stop_green, n, start_compound=c, **kw)
        if pr_any is None:
            return None
        per = {"any": arr(pr_any)}
        for k in list(n_stops or []):
            pr = first_stop_prior(first_stop_green, n, start_compound=c, n_stops=int(k), **kw)
            if pr is None:
                continue
            if not pr.get("n_by_stops") and not pr.get("n_by_compound_stops"):
                # A family this circuit's field has never run (a one-stop at
                # Melbourne, whose 2023-24 races were all two-stops) has no
                # first-stop history of its own, and another family's first
                # stops say nothing about where its stop belongs - a two-stop's
                # lap-9 first stops would drag a one-stop to the same lap.  No
                # information is no penalty, not a borrowed one.
                per[int(k)] = np.zeros(n + 1, dtype=float)
                continue
            per[int(k)] = arr(pr)
        out[c] = per
    return out or None


def first_stop_summary(first_stop_green: dict | None, n_race_laps: int, *,
                       start_compound: str | None = None, n_stops: int | None = None,
                       **kw) -> dict | None:
    """The prior's shape as the fitstage JSON and the report state it.

    Reports the compound-only density and, when `n_stops` is given, the
    compound x stops cell beside it under `by_stops`, so a reader can see how far
    conditioning on the family moved the mode.
    """
    pr = first_stop_prior(first_stop_green, n_race_laps, start_compound=start_compound, **kw)
    if pr is None:
        return None
    out = {"mode": pr["mode_lap"], "median": pr["median_lap"], "p25": pr["p25"], "p75": pr["p75"],
           "n": pr["n"], "n_compound": pr["n_compound"], "start_compound": start_compound,
           "source": pr["source"]}
    if n_stops is not None:
        cs = first_stop_prior(first_stop_green, n_race_laps, start_compound=start_compound,
                              n_stops=int(n_stops), **kw)
        if cs is not None:
            out["by_stops"] = {"n_stops": int(n_stops), "mode": cs["mode_lap"], "median": cs["median_lap"],
                               "p25": cs["p25"], "p75": cs["p75"],
                               "n_by_stops": cs["n_by_stops"], "n_by_compound_stops": cs["n_by_compound_stops"],
                               "source": cs["source"]}
    return out
