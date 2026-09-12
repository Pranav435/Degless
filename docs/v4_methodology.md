# Degless V4 — benchmark methodology (WP-C)

*Frozen 2026-09-13, before any V4 benchmark was run, from `docs/v4_plan.md` §2/WP-C and §4. It defines the position-aware strategic metric `R_pos`, the Haas rows, the live decision-stability block, and the procedure the final numbers must come out of. Implemented in `bench/bench_strategy.py`, `bench/bench_live.py` and `bench/bench_v4_compare.py`; tested in `tests/test_v4_bench.py`. Nothing here re-defines an existing metric: the pure-time oracle regret, the first-stop error, the window share, the sequence/start/stop matches and every live metric keep their V3 definitions, their code path and their JSON keys, and `R_pos` is reported beside them.*

## 1. Why a second regret

The oracle regret the suite has reported since V2 prices every candidate plan on a tyre model whose degradation rates are the ones **measured on the race being scored** (`bench_strategy.oracle_model`), with the model's own grip budgets and pace offsets, and takes the difference from the best plan that model can find. It is the right question for a tyre model and the wrong one for a pit wall: it is a pure race-time number, and a race-time oracle has no notion of a place. Measured on the 2026 races, even the oracle puts the time-optimal first stop *later* than the field stopped, so V4's race state — which moves the first stop earlier to protect track position — is a regression by that metric by construction (Task 1: 9.40 s → 10.92 s, `results_v4_task1.md` §5).

`R_pos` adds the missing term and nothing else. It is the same oracle race time plus the value of the track position the first pit cycle wins or loses, at the weekend's own leave-one-out place value. A plan that spends two seconds to keep a place is then scored on whether the place was worth two seconds.

## 2. Definition

For a candidate plan `p` with first stop on lap `s_p`, on the weekend's oracle tyre model at push 1:

```
J(p)      = T_oracle(p) + V · L(p)

L(p)      = mean over the field's green first stops l_r
            of  Σ_slots [ Φ((g_slot + D(s_p, l_r)) / σ_rel) − Φ(g_slot / σ_rel) ]

D(s, l)   = A_me(s) − A_r(l)
A_me(s)   = cost of s laps on p's start compound
              + cost of (max(s,l)+1 − s) laps on p's second compound, that set fitted after s laps
              + pit loss
A_r(l)    = cost of l laps on p's start compound
              + cost of (max(s,l)+1 − l) laps on the rival's second compound, fitted after l laps
              + pit loss

R_pos(p)  = J(p) − min over q in C of J(q)

P_retain(p) = clip(1 − L(p)/4, 0, 1)
```

* **`T_oracle(p)`** is exactly the number the pure-time block already reports (`r["oracle"]["costs_s"]`): `strategy.evaluate_plans` on the oracle model at push 1, with the same allocation and stint caps, so `J` and the existing regret differ by one term and nothing else. It is reported per plan as `position_aware.T_oracle_s`.
* **The field's green first stops `l_r`** are the in-laps of the first stop of every classified finisher whose first stop was *not* taken under a safety car — `common.driver_plans`' `pit_laps[0]` where `first_sc` is false, the same list the first-stop error and the window share are measured against. `n_field_first_stops` records how many there were, and a weekend with none (Italy 2026: every first stop was under a safety car) reports `L`, `J` and `R_pos` as `null` with a note rather than a number built on nothing.
* **`D(s, l)`** is both cars' race time from lap 0 to the lap both are out of the pits, `max(s, l) + 1`, each on its own set, priced on the oracle's mean stint-cost table (`strategy.stint_cost_table` at push 1, averaged over the posterior draws — the same table `strategy._phase1_race_state` prices its pack on). Our car runs `p`'s start compound then `p`'s second. The rival runs **the same start compound as `p`** and the **modal second compound among the field's plans that started on that compound** (`position_aware.rival_second_by_start`), falling back to `p`'s own second where the field never started there. The pit loss is carried in both terms, where it cancels: both cars stop inside the cycle by construction.
* **The pack slots `g_slot`** are `racestate.pack_slots(const)`: two rivals ahead and two behind, the nearer at one measured first-stint interval and the further at the sum of two, each as eight equal-mass quantiles of the measured interval distribution. A slot therefore contributes the mean over its quantiles, and the four slots sum — the same weighting `racestate.pack_equilibrium` uses, so the four weights total 4.0 and `L` is a number of places out of four.
* **`V` and `σ_rel`** are the weekend's **leave-one-out** race-state constants: `racestate.measure_constants(exclude=<this weekend>, estimator="regularized")`, WP-A's V4 production estimator, with the keyword dropped on a checkout whose `measure_constants` does not take it yet (Task 1's does not). The estimator actually used is recorded as `position_aware.estimator` and its donor list as `constants_source`. `V = place_gap · max(0, 2ψ − 1)`, `σ_rel = √2 · cycle_sd`.
* **The candidate set `C`** is every legal first-stop lap of `p`'s family — the tool's compound sequence, with the later stops re-optimised on the oracle — together with `{tool, tyre_optimal, field_modal, winner, oracle_opt}`. The family sweep takes its stint-length grid from `strategy.enumerate_strategies` on the family's own compounds with the search's margin (6 laps), the weekend's tyre allocation and the circuit's stint caps; for each first-stop lap it keeps the composition with the lowest pure-time oracle cost (tyre + pit lane + rejoin traffic + safety-car credit + grid penalty — the phase-1 terms, no prior and no position term), and the chosen plans are then re-priced exactly by `evaluate_plans`, so the sweep only ever *selects* the later stops. Because every reported plan is itself in `C`, `R_pos ≥ 0` for all of them and `R_pos(best_candidate) = 0`.
* **`L(p)` is signed**: positive is places lost, negative is places gained (an undercut that works). `P_retain` is the share of the four pack slots the plan holds, clipped into `[0, 1]`; it is a reporting convenience, not a probability the model computes anywhere.

Reported per weekend in `bench/out/strategy.json` under `r["oracle"]["position_aware"]` — `V_s`, `sigma_rel_s`, `n_field_first_stops`, `T_oracle_s`, `L`, `J`, `R_pos`, `P_retain_tool`, `best_candidate`, the whole `family_sweep` — in `bench/out/strategy_table.csv` as `rpos_tool`, `rpos_tyre_opt`, `rpos_field`, `L_tool`, `L_field`, and pooled over the weekends by `bench/bench_v4_compare.py`.

### Deviations from the plan's text

* The plan's `C` is the family sweep ∪ `{oracle_opt, field_modal, winner, tyre_optimal}`; the implementation also puts `tool` in `C`. Without it `R_pos(tool)` could be negative by a rounding-sized amount whenever the tool's own later stops beat the sweep's pick at the same first-stop lap, and a regret that can go negative is not a regret. It changes no other plan's `R_pos` by more than the tool's own margin.
* The plan says the rival's second compound is "the modal second compound among the field's plans with that start, else `p`'s second". The modal second is taken over **all** classified finishers that started on that compound, not only those whose first stop was green: the compound choice is a plan, not a timing decision, so a safety-car stop does not disqualify it.
* `P_retain` is defined on `L` of the plan being scored, so `P_retain_tool` is reported for the tool and a `P_retain` per Haas car; no `P_retain` is reported for the other candidates, which would only restate their `L`.

## 3. What it measures — and what it does not

**It measures** the first pit cycle. For one plan, on one weekend, against the field as it actually stopped: how much race time the plan costs against the best plan available on a tyre model that knows the race, plus what those laps do to the plan's place among four nearby cars through that one exchange, valued at the measured worth of a place.

**It does not measure:**

* **Anything after the first cycle.** The second and third stops carry no position term at all. A plan that wins the first cycle and loses the race is scored as a good plan.
* **A real grid position.** There is no starting position anywhere in the metric. The four rivals sit at the *measured pack-gap quantiles* — the first-stint intervals the 2026 races showed at P2–P15 on laps 5–15 — not at this car's actual gaps in this race. `L` is therefore what the cycle does to a car in a typical midfield pack, not to OCO from P14 at Barcelona.
* **The field's response to us.** The field's stops are taken as fixed at what they were. There is no cover response (unlike `racestate.pack_equilibrium`, which has one) and no reaction to our stop: we are scored against history, and history did not have to race us.
* **Who the rivals are.** They are four anonymous cars on our own start compound and the field's modal second compound, not the drivers who were actually around us — the same rivals at every candidate lap, and the same at every position.
* **Overtaking, defending, or whether the place could be taken back on track.** Position is a probability from a normal cushion on the race-time difference, `Φ((g + D)/σ_rel)`; `σ_rel` is measured from the spread of green-flag pit losses and nothing else. The persistence discount `ψ` inside `V` is the only thing that prices "the order out of a cycle is not the order at the flag".
* **Its own inputs' uncertainty.** `V` rests on `ψ`, and `ψ` rests on 25–35 first-pit-cycle pairs per fold; leave-one-out it swings 0.68–0.81 and takes `V` with it, 2.0–3.6 s (`results_v4_task1.md` §5). A 0.4-place difference in `L` is inside that. WP-A's `bench_place_value.py` reports the CI and the estimator sensitivity; `R_pos` should be read with it.
* **Safety-car weekends.** Where the field's first stops were set by a safety car there is little or no green field stop to score against; at Italy 2026 there is none, and the metric declines to report.

Both regrets are therefore kept and reported side by side. Neither is the verdict on its own: the pure-time regret is the cost of the trade and `R_pos` is the trade priced, and where they disagree the report says so and says by how much.

## 4. Haas rows

**Pre-race** (`bench_strategy.py`, `r["haas"]`), for `#31 OCO` and `#87 BEA`, Haas F1 Team:

* the per-car plan from `meta["per_driver"]` — label, compounds, stop count, first stop, push, `race_factor`;
* the driver's own **green** first stop from the race lap table (`common.driver_plans`), with `actual_first_sc` stated: `first_minus_actual_green` is `null` when the driver's own first stop was under a safety car, because that stop was not a timing decision, and `first_minus_actual` keeps the raw difference beside it;
* `same_shape_as_field` as the pipeline recorded it, plus `seq_match_actual` and `stops_match_actual` against the driver's real race;
* the car's own **`L`** — §2's `L`, evaluated on the oracle model at the per-car plan's start compound, second compound and first stop, against the same field stops — and its `P_retain`. The plan is the car's; the place cost is the weekend's.

Pooled by `bench_v4_compare.py` as `strategy.<build>.haas`: mean absolute and signed first-stop error over the scored car-weekends, the share of per-car plans on the field plan's shape, and the mean `L`.

**Live** (`bench_live.py`):

* `stops_haas` — the pooled stop-call metrics restricted to OCO and BEA: `n`, the share of their real stops called within three laps, the median absolute error, the median box-now cost the lap before the stop, the share inside the window three laps earlier, and each car on its own under `by_driver`.

## 5. Live decision stability

`bench_live.py`'s `decision_stability`, over every car-lap of both replays. A car-lap is *scored* when the engine's plan carries a `decision` block (`plan["decision"]["action"]`, WP-D) and the same car had one on the previous lap. The action either held or changed. A change is **material** — and so not churn — when any of:

1. the car pitted, with an in-lap on this lap or the previous one;
2. a driver listed in either lap's `decision["rivals"]` pitted on those laps;
3. `meta["sc_active"]` differs between the two laps;
4. the best-vs-runner-up margin `decision["delta_vs_alternative_s"]` moved by at least `DECISION_MARGIN_S = 1.0 s`.

Anything else is an **unexplained change**. Reported as `n` (scored car-laps), `n_pairs`, `n_changes`, `n_unexplained`, `share_changed`, `share_unexplained` (of pairs), `share_of_changes_unexplained`, the count of each material reason, the two Haas cars separately, and up to ten examples.

`DECISION_MARGIN_S = 1.0 s` is an engineering constant of the same order as the engine's own window tolerance (`BOX_NOW_TOL_S`) and WP-D's `DECISION_HYSTERESIS_S = 0.3 s`; it is not fitted to anything. The raw counts are recorded, so a different threshold can be applied to the stored numbers without a re-run.

The `decision` key is read defensively. Task 1's plans carry none, so the block reports `n = 0` and nulls rather than a zero share that would read as perfect stability.

## 6. Benchmark procedure (plan §4, frozen)

1. `.venv/bin/python -m pytest tests -q` green.
2. `scripts/80_recalibrate.py` (V4 objective) → `data/processed/calibration.json`; then `scripts/10_pipeline.py --event <key> --stage decide --offline` on the seven scored weekends (the tyre posteriors are **not** refitted in V4); `scripts/40_weekend.py` / the outlook for the live weekends as the repo's commands require.
3. `bench/run_all.sh` (accuracy, strategy, ablation, outlook, live, stability, speed, apex, pytest, compare), then `bench/bench_live.py --no-race-state`, `bench/bench_tick_paired.py`, `bench/bench_experiments.py`, `bench/bench_place_value.py`, `bench/bench_extrapolation.py`, `bench/bench_v4_compare.py`.
4. Any code fix after step 3 restarts from step 2. The report quotes only the last complete run.
5. `results_v4.md` → `results_v4.pdf` (`bench/md2pdf.py`).

Every metric is leave-one-out: a weekend's own race never informs its own decision, and that includes `V` and `σ_rel` here. The three builds compared are V3 (`bench/v3/`), V4 Task 1 (`bench/v4_task1/`, frozen) and V4 final (`bench/out/` + `data/processed/`); `bench/bench_v4_compare.py` prints the three columns and writes `bench/out/v4_compare.json`.

### Reproducibility note

Task 1's frozen `bench/out/*.json` were produced on a different machine. Re-running the unchanged `bench_strategy.py` here reproduces every integer, boolean, count and rounded figure exactly, and the unrounded `oracle.costs_s` floats to ~1 part in 10¹² (float summation order under a different BLAS). Tick latencies in `live.json` are machine-dependent by nature and are compared only through `bench_tick_paired.py`, which measures both objectives in one process.
