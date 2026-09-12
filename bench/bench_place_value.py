"""What is a place worth, and how well do we know it?

The place value `V (2 psi - 1)` is the constant the whole race-state term is
proportional to, and it rests on the thinnest samples in the model: ~100
finishing gaps and **25-35 adjacent pit-cycle pairs** pooled over six races,
three to ten pairs per race.  Task 1 shipped it as a pooled ratio and disclosed
that changing one definition (lead-lap gaps vs every classified finisher's)
moved the benchmark's first-stop error by 0.7 laps.  A number that influential,
measured that thinly, has to be audited rather than asserted.

So this script reports, per leave-one-out fold and pooled over all seven races:

  n           the cycle pairs and finishing gaps behind the fold;
  psi         the pooled ratio, the empirical-Bayes shrunk value, the prior it
              shrank toward and a bootstrap interval over the donor races;
  V           the median finishing gap on three definitions - every classified
              finisher (Task 1's), lead-lap finishers only (Task 1's first
              implementation) and the P7-P16 midfield band (the cars this tool
              actually decides for) - with a bootstrap interval;
  place value under each of the four estimators `racestate.measure_constants`
              offers, so the estimator's effect on the constant is visible
              before any search is run;

and five sensitivities, each of which is a definition somebody chose:

  window      a pit cycle is two first stops within 3, 5 (shipped) or 8 laps;
  adjacency   the two cars were 1 (shipped) or up to 2 positions apart;
  lapped cars lead-lap finishers vs every classified finisher (the V column);
  SC races    races whose first-stop phase ran under a safety car against the
              green-only ones - a safety-car cycle is a different experiment;
  influence   the leave-one-pair-out range of psi: how much one observation of
              one pair at one race moves the constant.

**It chooses nothing.**  The production estimator (`regularized`) was fixed by
`docs/v4_plan.md` on statistical grounds - beta-binomial shrinkage is the
standard treatment of a small binomial sample pooled over heterogeneous
races - before any of these numbers existed, and nothing here is a benchmark
score.  Offline and read-only: `data/processed/laps_*_race.parquet` and
`pitloss_*.parquet`, nothing else.

Writes `bench/out/place_value.json`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import SHORT, arg_events, dump, offline  # noqa: E402
from src import racestate as rst

WINDOWS = (3, 5, 8)            # laps between two first stops that make one cycle
ADJACENCY = (1, 2)             # positions apart the lap before the earlier stop


def counts(keys: list, **kw) -> dict:
    """Per-race measurements for `keys`, dropping the races not on disk."""
    got = {k: rst.race_measurements(k, **kw) for k in keys}
    return {k: v for k, v in got.items() if v is not None}


def psi_block(got: dict) -> dict:
    """psi three ways - pooled, shrunk, and one pair at a time - for one fold."""
    keys = list(got)
    pairs = np.array([got[k]["pairs"] for k in keys], dtype=float)
    kept = np.array([got[k]["kept"] for k in keys], dtype=float)
    eb = rst.persistence_eb(pairs, kept)
    # per-observation influence: drop each cycle pair in turn - every one of
    # them, at the race it was observed at - and see how far the shrunk psi
    # moves.  One pair out of 26 is the unit this constant is measured in.
    lo = hi = float(eb["psi"])
    for i, k in enumerate(keys):
        for outcome in got[k]["pair_kept"]:
            pr, kp = pairs.copy(), kept.copy()
            pr[i] -= 1
            kp[i] -= outcome
            v = float(rst.persistence_eb(pr, kp)["psi"])
            lo, hi = min(lo, v), max(hi, v)
    return {"n_pairs": int(pairs.sum()), "n_kept": int(kept.sum()),
            "n_races_with_pairs": int((pairs > 0).sum()),
            "psi_raw": float(eb["raw"]), "psi_shrunk": float(eb["psi"]),
            "prior_mean": float(eb["prior_mean"]), "prior_pairs": float(eb["prior_pairs"]),
            "tau2": float(eb["tau2"]), "between_race_mean": float(eb["between_race_mean"]),
            "method": str(eb["method"]),
            "psi_loo_pair": [round(lo, 4), round(hi, 4)], "psi_loo_pair_range": round(hi - lo, 4),
            "by_race": {k: {"pairs": int(got[k]["pairs"]), "kept": int(got[k]["kept"])} for k in keys}}


def v_block(got: dict) -> dict:
    """The place gap on its three definitions, with a bootstrap interval."""
    keys = list(got)

    def med(name: str) -> float:
        x = np.concatenate([got[k][name] for k in keys]) if keys else np.zeros(0)
        return float(np.median(x)) if len(x) else float("nan")

    pairs = np.array([got[k]["pairs"] for k in keys], dtype=float)
    kept = np.array([got[k]["kept"] for k in keys], dtype=float)
    boot = rst.bootstrap_place_value([got[k]["fin"] for k in keys], pairs, kept)
    return {"n_gaps": int(sum(len(got[k]["fin"]) for k in keys)),
            "V_all_classified": med("fin"), "V_lead_lap": med("fin_lead"),
            "V_midfield_p7_p16": med("fin_mid"),
            "V_ci": [round(float(x), 3) for x in (boot.get("place_gap") or {}).get("ci", ())],
            "psi_ci": [round(float(x), 4) for x in (boot.get("persistence") or {}).get("ci", ())],
            "place_value_sd": round(float((boot.get("place_value") or {}).get("sd", float("nan"))), 3),
            "place_value_ci": [round(float(x), 3) for x in (boot.get("place_value") or {}).get("ci", ())]}


def fold(keys: list, *, exclude=None) -> dict:
    """One leave-one-out fold: the constants and the audit behind them."""
    donors = [k for k in keys if k != exclude]
    got = counts(donors)
    out = {"exclude": exclude, "donors": donors, **psi_block(got), **v_block(got)}
    for est in rst.ESTIMATORS:
        c = rst.measure_constants(exclude=exclude, donors=tuple(keys), estimator=est)
        out[f"place_value_{est}"] = round(c.place_value_s, 3)
        out[f"place_gap_{est}"] = round(c.place_gap_s, 3)
        out[f"persistence_{est}"] = round(c.persistence, 4)
    return out


def sensitivities(keys: list) -> dict:
    """The five definitional choices, pooled over every donor race."""
    out: dict = {"cycle_window": {}, "adjacency": {}, "safety_car": {}}
    for w in WINDOWS:
        got = counts(keys, window=w)
        b = psi_block(got)
        out["cycle_window"][str(w)] = {k: b[k] for k in ("n_pairs", "psi_raw", "psi_shrunk",
                                                         "psi_loo_pair_range")}
    for a in ADJACENCY:
        got = counts(keys, adjacency=a)
        b = psi_block(got)
        out["adjacency"][str(a)] = {k: b[k] for k in ("n_pairs", "psi_raw", "psi_shrunk",
                                                      "psi_loo_pair_range")}
    got = counts(keys)
    for tag, want in (("sc_affected", True), ("green_only", False)):
        sub = {k: v for k, v in got.items() if bool(v["sc_affected"]) is want}
        out["safety_car"][tag] = ({"races": list(sub), **psi_block(sub), **v_block(sub)} if sub
                                  else {"races": [], "note": "no race in this class"})
    out["safety_car"]["definition"] = ("a race counts as safety-car affected when any lap between "
                                       "the earliest and the latest first stop was run under SC, VSC "
                                       "or a red flag")
    return out


def main() -> None:
    args = arg_events(__doc__)
    offline()
    keys = list(args.events)
    out = {"donor_events": keys,
           "pooled": fold(keys),
           "folds": {k: fold(keys, exclude=k) for k in keys},
           "sensitivity": sensitivities(keys),
           "production_estimator": "regularized",
           "note": ("the estimator is fixed by docs/v4_plan.md on statistical grounds; nothing here "
                    "is a benchmark score and no race's own result informs its fold")}

    rows = []
    for tag, b in [("pooled", out["pooled"])] + [(f"-{SHORT.get(k, k)}", v) for k, v in out["folds"].items()]:
        rows.append({"fold": tag, "n_pairs": b["n_pairs"], "n_gaps": b["n_gaps"],
                     "psi_raw": round(b["psi_raw"], 3), "psi_shrunk": round(b["psi_shrunk"], 3),
                     "prior": round(b["prior_mean"], 3), "prior_pairs": round(b["prior_pairs"], 1),
                     "psi_ci": b["psi_ci"], "psi_pair_range": b["psi_loo_pair_range"],
                     "V_all": round(b["V_all_classified"], 2), "V_lead": round(b["V_lead_lap"], 2),
                     "V_mid": round(b["V_midfield_p7_p16"], 2), "V_ci": b["V_ci"],
                     "pv_task1": b["place_value_task1"], "pv_lead": b["place_value_lead_lap"],
                     "pv_reg": b["place_value_regularized"], "pv_sd": b["place_value_sd"],
                     "pv_ci": b["place_value_ci"]})
    tbl = pd.DataFrame(rows)
    print("\n=== place value, per leave-one-out fold (fold '-XXX' = that weekend held out) ===")
    print(tbl.to_string(index=False))
    print("\n=== sensitivity: what a pit cycle is ===")
    print(pd.DataFrame(out["sensitivity"]["cycle_window"]).T.to_string())
    print("\n=== sensitivity: how adjacent the two cars were ===")
    print(pd.DataFrame(out["sensitivity"]["adjacency"]).T.to_string())
    print("\n=== sensitivity: safety-car first-stop phases against green ones ===")
    sc = {k: v for k, v in out["sensitivity"]["safety_car"].items() if isinstance(v, dict)}
    print(pd.DataFrame({k: {kk: v.get(kk) for kk in ("races", "n_pairs", "psi_raw", "psi_shrunk",
                                                     "V_all_classified")} for k, v in sc.items()}).T.to_string())
    print(f"\nproduction estimator: {out['production_estimator']}")
    p = dump("place_value.json", out)
    tbl.to_csv(p.with_name("place_value_table.csv"), index=False)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
