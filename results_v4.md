# Degless V4 — Haas race strategy: the race-state system completed

*13 September 2026. Same seven dry 2026 weekends, same 196 race stints, same benchmark scripts and definitions as V3 and V4 Task 1. V3's numbers are its own code re-run on this machine (`bench/v3/`); Task 1's are frozen as it left them (`bench/v4_task1/`, measured on a slower laptop, so its tick times are not comparable and are quoted only in the paired form); the final numbers are one uninterrupted run of `bench/run_all.sh` (954 s, 16 stages, all exit 0) on the code at the last commit, after the leave-one-out recalibration and the seven decide stages. Tests: 156 pass, 0 skip. The tyre posteriors were not refitted.*

**Bottom line.** V4-final is the Task 1 race-state concept made whole: one objective everywhere, tyre life stated as a range, a per-car model for the two Haas drivers, a live engine that names its rivals, projects the rejoin, holds its call and explains it, a position-aware regret beside the pure-time one, and a Haas front end. On the metric Task 1 was built for it improves again: the mean first-stop error on the three green-flag weekends falls from **1.67 to 1.33 laps** (signed +1.00 → 0.00; V3 4.33) and the share of the field's first stops inside the model's window rises from 0.448 to **0.530**. Live, the median stop-call error falls from 3.0 to **2.5 laps** and the tick costs **92 ms** on this machine (race state on vs off, paired: +8 and +12.5 ms). The price is in the strategy-family layer, and it is not the race state's doing: the V4 recalibration's per-fold constants move two families — Australia to a 2-stop (τ = 6 s in that fold) and Barcelona to a 3-stop (τ = 2.5 s, chosen over 4.0 s by a one-flag tie) — so the sequence and stop-count matches fall from 6/7 and 7/7 to **5/7 and 5/7**, the oracle regret rises from 10.92 to **12.66 s** and the new position-aware regret reads 13.5 s against 11.5 s for Task 1's frozen plans. The same code under Task 1's per-fold constants (E6b) keeps 6/7/7 at 10.94 s regret and 11.6 s `R_pos`. The tie-break that produced the constants is V3's pre-registered rule and it stands; changing it after seeing these numbers is the pattern Task 1 was told not to repeat. The heterogeneous rival field was built and validated and is **not** shipped for the pre-race call: it predicts the field's stops better than the pack of clones in every fold and prices our own stop worse (§4, §5). Belgium's stint gate is diagnosed, not closed, and Austria gains one.

## 1. What changed since Task 1

Task 1's mechanism is kept whole: the first stop is timed against the cars being raced, at the measured value of a place, and the plan family is still chosen by the tyre model under the plan-family prior. V4-final changes what surrounds it.

- **One objective** (`src/objective.py`). Every strategy cost in production — the search, the pit window, the counterfactual, the outlook, the plan builder, the per-car plans, the recalibration sweeps — is assembled from one `V4Objective`: the race-state term on the first stop, the undercut exposure at `λ` on the stops after it, the plan-family prior at `τ`, no first-stop history prior (`κ = 0` is asserted), the circuit's dirty air, the measured pit loss and the tyre-life extrapolation width. Task 1 had left the counterfactual, the outlook and the plan builder on V3's objective; `docs/v4_objective_audit.md` lists the 247 occurrences and what each became. The counterfactual on the V4 objective moves 18 of 19 Hungarian drivers' best first stops (COL 24 → 18, ANT 22 → 17) and lowers the median seconds lost at Barcelona from 8.6 to 5.8 and at Hungary from 4.0 to 2.4.
- **The recalibration runs on the V4 objective** (`scripts/80_recalibrate.py`, 539 s). Every sweep search carries the race state, measured without both the held-out weekend and the donor being searched. `κ` is not swept: it is structurally zero. `λ` is swept against the second stop (the first is the race state's), `τ` against the plan shapes as before, and two new constants — the rival field's family temperature and the rival mode — against how well each rival model predicts the field's own green first stops (§4).
- **Tyre life is a range beyond the practice support** (`src/tyre.py`). The practice-to-race rate transfer error was measured on all scored stints (184 with a positive measured rate): pooled dispersion 0.60 in log rate, 0.46 after removing the scorer's own three-lap noise, and it does **not** grow with a stint's extrapolation beyond the practice support (bins ≤ 1×, 1–1.5×, 1.5–2×, > 2× of support: dispersion 0.70, 0.50, 0.64, 0.31 on n = 77, 76, 19, 12; a maximum-likelihood growth term sits at zero, 95 % interval 0–0.43; 44 % of stints beyond support degraded faster than predicted, binomial p = 0.25). The shipped mechanism widens each posterior draw's wear rate by a fixed seeded lognormal factor `exp(z · 0.46 · max(0, L/support − 1))` beyond the support, so a stint pays a risk premium only through the cliff's convexity and only where it extrapolates. Every reported life carries `life_lo`/`life_hi` and a censored-share note; the live action table carries `p_cliff_before_stop`. Bit-identical to Task 1 at width 0; the width is measured, not tuned, and it changes no plan at 0, ½×, 1× or 2× (E5).
- **A regularised place value** (`src/racestate.py`, `bench/bench_place_value.py`). Persistence `ψ` is shrunk by an empirical-Bayes beta-binomial across the donor races; the place gap keeps Task 1's all-classified-finisher definition; both carry a bootstrap 5–95 % interval, and so does the place value, which every meta and live snapshot now reports with its standard deviation (§4, Issue 2).
- **A heterogeneous rival field, built, validated, not shipped for the pre-race call** (`racestate.rival_field`, `strategy._hetero_field`). A rival *type* is a plan group (start compound, second compound, stop count) crossed with one of three degradation levels; types are weighted by a logit on the model's own family costs blended with the circuit's plan history; each type chooses its stop lap on its own cost curve blended with the circuit's first-stop density for its cell; the field is solved as a mean-field equilibrium on the four pack slots, and our stop is the best response with the cover on. It is in the code (`RivalFieldConfig(mode="hetero")`), the recalibration and the benchmark.
- **A Haas car model** (`src/haascar.py`). For #31 Ocon and #87 Bearman, from this weekend's practice alone: pace level, degradation and age sensitivity (the fit's own per-driver deviation, team-pooled), warm-up, long-run consistency, traffic sensitivity, sector degradation (from the FastF1 cache), pace against the team-mate; each shrunk driver → team → field with a documented pseudo-count and reported with its evidence count, shrink weight and source. The cars' warm-up and traffic terms enter their own plan searches and the live engine as differences from the field; `explain_difference` traces every material difference between the two cars' recommendations to a cause with the numbers, or says it is within noise. `meta["haas"]` carries all of it.
- **The live race-execution engine** (`src/live/rivals.py`, `src/explain.py`, `src/live/engine.py`). Rivals are the cars that can exchange a place with us through the next cycle — the virtual band as before, plus the car directly ahead and behind on track, minus lapped cars and cars a whole cycle away — ranked by strategic relevance and capped at four. The rejoin is projected from every car's stint pace and the stop the engine gave it last lap, not from today's gaps. Each rival carries the probability it covers our stop and the outcome if it does. Every plan carries a `decision`: PIT NOW / STAY OUT / WAIT k LAPS / BOX BY LAP x, a confidence (the share of the car's posterior draws on which the chosen action is the cheapest of the five), the projected rejoin position, the margin to the runner-up, the rivals, the tyre-life risk per action, and reasons generated from those numbers. A 0.3 s hysteresis holds the previous call unless the new one beats it by that margin or the state changed (a stop, a rival's stop, an SC/VSC, the window closing).
- **A position-aware strategy metric** (`bench/bench_strategy.py`, `docs/v4_methodology.md`). `R_pos(p) = J(p) − min_C J`, `J(p) = T_oracle(p) + V · L(p)`, where `L(p)` is the expected number of places lost through the first pit cycle against the field's actual green first stops on the four measured pack slots, priced on the same race-measured oracle tyre model as the pure-time regret, which is kept unchanged beside it.
- **The Haas front end** (`app/haas_tab.py`): a first tab, *Degless — Haas Race Strategy*, with the Haas Overview (two car panels), the Pit Wall (the call, its confidence, the projected position, the delta, the principal reason and the Why block from `src.explain`), the race field around the two cars and the 1-/2-/3-stop comparison; driver views for Ocon and Bearman; every value from the snapshot or the meta, "—" where there is none. Model internals stay in the existing tabs.
- Also: `bench/bench_extrapolation.py`, `bench/bench_place_value.py`, `bench/bench_experiments.py` (E0–E8), `bench/bench_wpd_live.py` (the live choices), `bench/bench_v4_compare.py` (three builds), eight new ablation variants, `docs/v4_plan.md`, `docs/v4_methodology.md`, `docs/v4_feature_audit.md`, `docs/v4_objective_audit.md`; 103 new tests.

## 2. Architecture

```
telemetry / lap tables
  → tyre state           src/tyre.py (posterior, cliff, extrapolation width), src/cliff.py
  → Haas car state       src/haascar.py (driver → team → field), src/percar.py
  → race state           src/racestate.py (constants; symmetric pack, rival field), src/live/rivals.py
  → candidate actions    src/strategy.py (families and laps), src/live/engine.py (options)
  → expected outcomes    src/objective.py (one V4Objective for every path)
  → PIT / STAY / WINDOW  racestate.action_table, engine._decide (hysteresis, confidence)
  → explanation          src/explain.py, haascar.explain_difference
  → presentation         app/haas_tab.py, app/live_tab.py
```

The strategy-family layer (nomination, plan prior at `τ`, tyre life and its width, stint caps, race distance) decides what is plausible; the race-execution layer (rivals, gaps, tyre state, traffic, rejoin, SC/VSC) decides when. History enters the execution layer only as the rivals' behaviour, never as a term on our own lap.

## 3. Final against Task 1 and V3

### 3.1 Pre-race decisions

| Metric (definitions unchanged) | V3 | Task 1 | V4 final | |
|---|---|---|---|---|
| Mean \|first stop − field green median\| (laps, non-SC) | 4.333 | 1.667 | **1.333** | improved |
| Signed first-stop error (laps) | +4.33 | +1.00 | **0.00** | improved |
| Share of field first stops inside the window | 0.361 | 0.448 | **0.530** | improved |
| Oracle regret, pure race time (s) | 9.399 | 10.917 | 12.664 | **regressed** |
| `R_pos` of the tool (s, 6 weekends) | – | 11.54 | 13.49 | **regressed** |
| `R_pos` of the tyre-optimal plan (s, 5) | – | 6.77 | 7.11 | – |
| `R_pos` of the field's modal plan (s, 6) | – | 18.48 | 18.75 | – |
| `L(tool)`: places lost through the first cycle | – | +0.03 | **−0.12** | improved |
| `P_retain(tool)` | – | 0.947 | 0.965 | improved |
| Sequence run by anyone (of 7) | 6 | 6 | 5 | **regressed** |
| Start compound = majority (of 7) | 7 | 7 | 7 | unchanged |
| Stop count = mode (of 7) | 7 | 7 | 5 | **regressed** |
| Mean field share on the recommended sequence | 0.350 | 0.350 | 0.217 | **regressed** |
| Per-car plans sharing the field plan's shape | 0.862 | 0.721 | 0.735 | – |
| Gates failing | 3 | 4 | 5 | **regressed** (Austria, §6) |
| Tests | 43 pass, 2 skip | 53 pass, 2 skip | **156 pass** | +103 |
| Benchmark suite wall time | – | 1420 s (laptop) | 954 s (16 stages) | – |

Task 1's `R_pos` is its frozen plans scored by the new metric. The pure-time regret and `R_pos` agree on the direction: V4-final's plans give away race time at Australia (2-stop) and gain it at Barcelona (the 3-stop is 3.7 s from the oracle and loses no place, but nobody ran it).

| Weekend | Field green median | V3 | Task 1 | V4 final | V4 window (share inside) | pack median | tyre alone | regret T1 → V4 (s) | `R_pos` V4 | `L` V4 |
|---|---|---|---|---|---|---|---|---|---|---|
| Australia (SC) | 25 | M-H @ 25 | M-H @ 21 | **M-H-H @ 14,36** | 12–17 (0.00) | 14 | 17 | 11.3 → 28.3 | 29.0 | +0.27 |
| Japan (SC) | 18 | M-H @ 23 | M-H @ 19 | M-H @ 19 | 16–22 (1.00) | 19 | 23 | 0.6 → 0.6 | 0.7 | +0.16 |
| Barcelona | 13 | M-H-H @ 19,42 | M-H-H @ 15,40 | **M-S-M-S @ 12,30,47** | 11–14 (0.67) | 12 | 17 | 7.6 → 3.7 | 0.3 | −0.92 |
| Austria | 18 | M-H-M @ 22,46 | M-H-M @ 20,45 | M-H-M @ 20,43 | 18–22 (0.85) | 21 | 25 | 16.5 → 15.4 | 16.4 | +0.40 |
| Belgium (SC) | 16 | M-S @ 18 | M-S @ 16 | M-S @ 16 | 16–17 (0.33) | 16 | 21 | 12.1 → 12.1 | 10.1 | −0.20 |
| Hungary | 19 | M-H-M @ 22,45 | M-H-M @ 18,43 | M-H-M @ 18,43 | 16–21 (0.33) | 19 | 24 | 23.6 → 23.9 | 24.4 | −0.39 |
| Italy (SC) | – | M-H @ 22 | M-H @ 21 | M-H @ 21 | 18–24 (–) | 21 | 26 | 4.7 → 4.7 | – | – |

- The race state moves every first stop earlier than the tyre alone (by 3–5 laps) and, on the three green-flag weekends, lands on −1, +2, −1.
- **Australia** and **Barcelona** change family. Both are the recalibration's per-fold constants, not the race state: with Task 1's constants the same code returns Task 1's families on every weekend (E6b, §4). Australia's fold took `τ = 6 s` (its shape objective is flat between 6 and 8 and one flag better than at 4; Melbourne's 2023–24 history is 12 of 23 on M-H-H, and the 2026 field split 8 one-stops to 6 two-stops); Barcelona's took `τ = 2.5 s` over the best value 4.0 s because the two differ by exactly the one-flag tolerance of the parsimony rule, and at 2.5 s the tyre model's 3-stop beats the field's M-H-H.
- Belgium's plan is Task 1's (the 28-lap SOFT stint and its gate); Austria's second stop moves 45 → 43 under its fold's `λ = 0.45` and the last MEDIUM stint runs to 28 laps against an observed maximum of 27, a new gate failure.

### 3.2 Live (the two archived replays, one tick per lap)

| Metric | V3 | Task 1 | V4 final | V4, race state off |
|---|---|---|---|---|
| Real stops inside the window 3 laps before | 0.421 | 0.443 | **0.459** | 0.420 |
| Stops within 3 laps of the recommendation | 0.472 | 0.547 | 0.545 | 0.502 |
| Median \|recommended − actual\| (laps) | 3.5 | 3.0 | **2.5** | 3.0 |
| Box-now cost the lap before the real stop (s) | 1.755 | 1.030 | 1.235 | 1.777 |
| Window signal precision / recall | 0.180 / 0.516 | 0.175 / 0.566 | 0.174 / 0.558 | 0.174 / 0.494 |
| Box-now signal precision / recall | 0.193 / 0.496 | 0.199 / 0.604 | 0.196 / 0.555 | 0.194 / 0.493 |
| Tick, mean / p95 (ms; V3 and Task 1 on the laptop) | 311 / 360 | 591 / 740 | **92 / 102** | 88 / 101 |
| Tick, paired on/off medians, BCN / HUN (ms) | – | +13.5 / +18.4 | **87.3 vs 77.5 / 94.6 vs 80.4** | – |
| Haas cars (OCO, BEA; 10 stops): within 3 laps / median error / box-now | – | – | 0.33 / 4.0 / 2.09 s | 0.50 / 3.5 / 1.85 s |
| Action changes nothing in the state explains (share of car-laps) | – | – | 0.172 | – |

Per replay, Task 1 → V4: Hungary in-window 0.565 → 0.522, within-3 0.565 → 0.543, median error 3 → 3, box-now 0.30 → 0.51 s; Barcelona 0.321 → 0.396, 0.528 → 0.547, 3 → 2, 1.76 → 1.96 s. The race state is worth +0.04 in-window, +0.04 within-3, half a lap of median error and 0.5 s of box-now cost against the same engine with it off. The Haas rows are ten stops (four at Hungary, six at Barcelona) and are worse with the race state on than off; Bearman's Barcelona stops at 18, 39 and 60 were called for 22, 48 and never. Every live number beyond `tick` is unaffected by the machine.

The live choices (`bench/out/wpd_live.json`, Barcelona): rivals k = 3 / 4 / 5 / 6 give within-3-laps 0.528 / 0.547 / 0.547 / 0.547 and in-window 0.415 / 0.377 / 0.415 / 0.434 at mean sets of 2.2 / 2.5 / 2.7 / 2.9 cars; k = 4 shipped as the smallest cap that is not worse on the primary metric. Hysteresis 0.0 / 0.3 / 0.6 s: unexplained changes 27.8 % / 24.8 % / 18.9 % of car-laps with no change in any accuracy metric; 0.3 s shipped (the engine's own window tolerance). The reachability filter (a car a whole cycle away is not a rival) is what separates the V4 set from Task 1's: 2.5 vs 2.9 cars and +0.02 within-3-laps at Hungary. The rejoin projection beats the gap arithmetic on the real stops of both replays (mean rejoin-position error 0.84–0.97 places at Hungary and 1.33–1.41 at Barcelona across the clip settings, versus the static gaps' larger errors on the same stops).

### 3.3 Accuracy, the tyre model and the Haas cars

The tyre model is unchanged (196 stints, the same population, MAE 0.0353 s/lap pooled; the `test_population_n_rate_stints_sums_to_196` invariant holds). The per-car variants: `team_pooled` 0.0362 s/lap, the hierarchical `sealed_driver_hier` 0.0372; on the fifteen scoreable Haas stints `none` 0.0410, `team_pooled` 0.0431, `hier` 0.0415 (better on Bearman, worse on Ocon). The Haas model's own terms change neither car's plan on any weekend; the pre-race per-car plans put Ocon's first stop at −1 (Japan), +5 (Barcelona), +2 (Austria), 0 (Belgium), −1 (Hungary) and Bearman's at −6 (Hungary) against their own green first stops, mean absolute 2.5 laps over the six scored car-weekends (Task 1's plans: 2.2 over seven).

## 4. The experiments (E0–E8) and the ablations

All leave-one-out, 300 draws on the shipped posterior, the same search and the same three scorings the strategy benchmark applies (`bench/bench_experiments.py`); E0 is Task 1 as frozen at the pipeline's 500 draws.

| Exp | Variant | first-stop err | signed | window | oracle regret | `R_pos` | `L` | seq/start/stops | BCN/AUT/HUN |
|---|---|---|---|---|---|---|---|---|---|
| E0 | Task 1 (frozen) | 1.67 | +1.00 | 0.448 | 10.92 | 11.54 | +0.03 | 6/7/7 | +2/+2/−1 |
| E1 | heterogeneous field, Task 1's estimator, no widening | 1.67 | +1.67 | 0.477 | 12.47 | 14.56 | +0.37 | 5/7/5 | 0/+5/0 |
| E1b | heterogeneous, no historical strategy prior | 1.67 | +1.67 | 0.440 | 12.31 | 14.28 | +0.32 | 5/7/5 | 0/+5/0 |
| E1c | symmetric pack, final code | **1.33** | 0.00 | 0.502 | 12.66 | **13.47** | −0.11 | 5/7/5 | −1/+2/−1 |
| E2 | place value: Task 1's estimator | 1.33 | 0.00 | 0.488 | 12.66 | 13.47 | −0.11 | 5/7/5 | −1/+2/−1 |
| E2 | place value: lead-lap | 1.33 | +1.33 | 0.447 | 12.50 | 14.01 | +0.13 | 5/7/5 | 0/+3/+1 |
| E2 | place value: regularised (shipped) | 1.33 | 0.00 | 0.502 | 12.66 | 13.47 | −0.11 | 5/7/5 | −1/+2/−1 |
| E3 | no family logit (rivals' mix from history alone) | 1.67 | +1.67 | 0.516 | 12.39 | 14.56 | +0.37 | 5/7/5 | +1/+4/0 |
| E3 | one degradation level | 1.33 | 0.00 | 0.502 | 12.66 | 13.47 | −0.11 | 5/7/5 | −1/+2/−1 |
| E5 | widening 0 / ½× / 2× | 1.33 | 0.00 | 0.530 / 0.530 / 0.502 | 12.70 / 12.70 / 12.76 | 13.53 / 13.53 / 13.59 | −0.11 | 5/7/5 | −1/+2/−1 |
| E8 | full V4 (shipped) | 1.33 | 0.00 | 0.502 | 12.66 | 13.47 | −0.11 | 5/7/5 | −1/+2/−1 |
| E6b | **diagnostic:** V4 objective, Task 1's per-fold `λ`, `τ` | 1.67 | +1.00 | 0.479 | **10.94** | **11.61** | +0.03 | **6/7/7** | +2/+2/−1 |

- **E1.** The heterogeneous field does not generalise on our decision: Austria +5 against the pack's +2, `R_pos` 14.6 against 13.5. Spreading the four rivals over plan families that do not interact with our first cycle takes pressure off the stop. That is why the pack of clones ships (§5, Issue 1).
- **E2.** The estimator choice changes nothing but the lead-lap definition, which is worse — as Task 1 found and disclosed.
- **E3, E5.** One degradation level, the family logit and the widening change no decision; the widening moves the window share by ±0.03.
- **E6b.** Everything V4 changed in the mechanisms — the objective, the estimator, the widening, the pack on the final code — under Task 1's constants keeps every family and improves both regrets. The regression in §3.1 is the recalibration's.

Ablations (`bench/bench_ablation.py`, pooled, first-stop error on BCN/AUT/HUN): `full` 1.33 (−1/+2/−1); `no_race_state` 3.67 (+3/+4/+4, the V3 objective on the V4 calibration); `symmetric_pack` = `full`; `hetero_pack` 1.67 (0/+5/0); `rivals_no_history` 1.67; `no_family_logit` 1.67; `rival_rate_levels_1`, `extrap_off`, `place_value_task1`, `no_first_stop_prior`, `no_dirty_air_circuit`, `old_budget`, `no_cliff_budgets` = `full`; `race_state_no_cover` 1.33 (−1/+1/−2); `race_state_lead_lap_value` 1.33 (0/+3/+1); `race_state_undiscounted` 3.00 and 4/6/5; `no_position` 1.00; `no_plan_prior` 2/2/5; `no_pace_cal` 0.67 and 7/7/6; `config_constants` 3/1/7. Per car, dropping the historical driver factors (`no_hist_driver_factors`) raises the share of cars on the field plan's shape from 0.45 to 0.75 at Australia and 0.59 to 1.00 at Barcelona.

### 4.1 The calibration (leave-one-out, V4 objective)

| Fold | rival mode | family temper (s) | `λ` (second stop) | `τ` (s) | ll/stop: hetero | symmetric | hetero, no history | uniform | n stops |
|---|---|---|---|---|---|---|---|---|---|
| global (a new weekend) | symmetric | 5.0 | 0.20 | 4.0 | −4.844 | −5.107 | −4.823 | −4.146 | 67 |
| −Australia | symmetric | 3.0 | 0.45 | 6.0 | −4.796 | −5.127 | −4.774 | −4.150 | 64 |
| −Japan | symmetric | 2.0 | 0.45 | 4.0 | −4.756 | −5.110 | −4.738 | −4.173 | 58 |
| −Barcelona | symmetric | 8.0 | 0.10 | 2.5 | −4.969 | −5.534 | −4.945 | −4.136 | 54 |
| −Austria | symmetric | 2.0 | 0.45 | 4.0 | −4.940 | −5.389 | −4.924 | −4.110 | 51 |
| −Belgium | symmetric | 8.0 | 0.15 | 4.0 | −4.856 | −5.088 | −4.839 | −4.188 | 60 |
| −Hungary | symmetric | 2.0 | 0.15 | 4.0 | −4.780 | −4.957 | −4.762 | −4.106 | 48 |
| −Italy | symmetric | 5.0 | 0.00 | 6.0 | −4.844 | −5.047 | −4.820 | −4.146 | 67 |

`κ = 0` in every fold (V3 had 1.0 in five). `λ` is identified by the second-stop objective in every fold but on two or three donors, and it moves 0.0–0.45; `τ` sits on a one-flag tolerance (`tol = 0.25 / n_donors`). The likelihood columns are the rival-model validation: the heterogeneous field beats the symmetric pack by 0.18–0.57 nats per stop in every fold, both are 0.6–0.8 nats per stop worse than a uniform stop lap over the race, and the history prior on the rivals' laps changes the likelihood by less than 0.03. The rival mode is forced to `symmetric` by the E1 result (`--rival-mode symmetric`); the family temperature is identified inside the widened grid (2–8 s; on the plan's 1–8 s grid it sat on the edge in every fold).

### 4.2 The place value (`bench/out/place_value.json`)

| Fold | pairs | `ψ` raw | `ψ` shrunk | prior pairs | `ψ` 5–95 % | `V` all | lead-lap | midfield P7–16 | `V` 5–95 % | value, Task 1 | value, regularised | sd | 5–95 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | 35 | 0.743 | 0.741 | 3.7 | 0.57–0.85 | 5.49 | 3.86 | 5.85 | 4.30–8.01 | 2.67 | 2.65 | 1.12 | 0.85–4.58 |
| −Australia | 34 | 0.735 | 0.726 | 5.9 | 0.54–0.83 | 5.16 | 3.87 | 5.16 | 4.22–7.74 | 2.43 | 2.33 | 0.96 | 0.38–3.72 |
| −Japan | 29 | 0.724 | 0.723 | 2.7 | 0.51–0.86 | 5.75 | 4.19 | 8.01 | 4.30–9.69 | 2.58 | 2.56 | 1.42 | 0.08–5.12 |
| −Barcelona | 30 | 0.767 | 0.766 | 2.0 | 0.57–0.88 | 4.58 | 3.48 | 5.16 | 4.22–5.87 | 2.44 | 2.44 | 1.01 | 0.67–4.18 |
| −Austria | 25 | 0.680 | 0.682 | 3.5 | 0.50–0.81 | 5.65 | 3.87 | 4.50 | 4.30–9.65 | 2.04 | 2.05 | 1.28 | 0.00–4.48 |
| −Belgium | 31 | 0.806 | 0.797 | 1.0 (Jeffreys) | 0.71–0.88 | 5.85 | 3.86 | 6.45 | 4.30–9.69 | 3.58 | 3.47 | 0.88 | 2.61–5.42 |
| −Hungary | 26 | 0.731 | 0.730 | 2.5 | 0.49–0.87 | 5.65 | 3.86 | 5.51 | 4.58–9.51 | 2.61 | 2.60 | 1.42 | 0.00–5.02 |
| −Italy | 35 | 0.743 | 0.741 | 3.7 | 0.57–0.85 | 5.16 | 3.74 | 5.51 | 4.22–9.69 | 2.51 | 2.49 | 1.18 | 0.88–4.78 |

Sensitivities (pooled): a 3 / 5 / 8-lap cycle window gives `ψ` 0.77 / 0.74 / 0.70 on 31 / 35 / 46 pairs; adjacency 1 vs ≤ 2 positions 0.74 vs 0.81 (35 vs 64 pairs); the five safety-car-affected races give `ψ` 0.76 and `V` 5.0 s on 21 pairs against 0.71 and 9.7 s for the two green races on 14; leaving any one pair out moves `ψ` by 0.03–0.06. The regularised estimator moves the constant by at most 0.11 s at any fold; its bootstrap standard deviation is 0.9–1.4 s. The constant is not precise, and the model no longer pretends it is.

## 5. The eleven issues, one by one

**Issue 1 — the symmetric pack.** Built the heterogeneous field (§1), validated it, and did not ship it for the pre-race call. Leave-one-out, on the field's own green first stops, the type field scores 0.18–0.57 nats per stop better than the pack of clones in every fold, and its family temperature is now identified inside the grid. But both rival models score 0.6–0.8 nats per stop *worse* than a stop lap drawn uniformly over the race: they put the field's median within 1–2 laps on the green-flag weekends (Barcelona 14 vs 13, Austria 18 vs 18, Hungary 18 vs 19) and are far too narrow about the rest of the field. On the specification's own test (E1) the heterogeneous field moves our first stop later — Austria +5 against the pack's +2 — because spreading the four rivals over plan families that do not interact with our first cycle takes pressure off the stop; it predicts the rivals better and prices our decision worse. The pre-race term therefore runs on the symmetric pack, as the specification says to do when the richer model does not generalise; the field stays in the code, the recalibration and the benchmark. The next rival model has to fix the dispersion, not the centre. Live, the rivals were always real cars and are now the strategically relevant ones (Issue 8).

**Issue 2 — the value of a place.** The audit is §4.2. The regularised estimator was fixed by the plan before any benchmark ran, on statistical grounds; the four estimators are compared in E2 and only the lead-lap definition changes a decision, for the worse. Every meta and every live snapshot carries `place_value_sd_s` and `place_value_ci_s`.

**Issue 3 — oracle regret.** The pure-time regret is unchanged and reported; `R_pos` is defined in `docs/v4_methodology.md` and reported beside it for the tool, the tyre-optimal plan and the field's modal plan (§3.1, §4). What the two say together: Task 1's earlier stops gave away 1.5 s of pure time for 0.03 places; V4-final's plans lose no place (`L = −0.12`, `P_retain` 0.965) and the tool's `R_pos` (13.5 s) sits between the tyre-optimal plan's (7.1 s) and the field's modal plan's (18.8 s) — the position term does not, on its own, vindicate the earlier stop against the oracle; it does vindicate it against what the field did.

**Issue 4 — one objective.** Done (§1); `docs/v4_objective_audit.md` is the audit. The recalibration's own weakness is the finding of the release: the second-stop objective rests on two or three donors per fold and `λ` moves 0.0–0.45 between folds; `τ`'s shape objective sits on a one-flag tolerance, and in the Barcelona fold the best value (4.0 s) and the chosen one (2.5 s) differ by exactly that one flag — the parsimony tie-break picks the smaller, and that alone turns Barcelona's plan into a 3-stop nobody ran. The tie-break is V3's rule, pre-registered in the plan, and it stands; E6b shows what the identification problem costs.

**Issue 5 — the two Haas cars.** The car model exists, is estimated from the weekend alone, and mostly shrinks to the team: over the fourteen car-weekends the pace level moves off the team value on every one (0.16–0.54 s/lap), the degradation deviation by at most 0.02 s/lap of age, warm-up on two car-weekends (the only two with two or more warm-up stints), traffic sensitivity on two (Barcelona Ocon ×1.27, Austria Bearman ×0.74), sector degradation on all seven weekends, push response never (not identifiable pre-race, and said so). Turning the cars' own terms on changes neither car's plan on any weekend, and the hierarchical rate scaling is no more accurate than the team-pooled one (§3.3) — so the rate scaling stays team-pooled, exactly as the specification asks when driver-specific intelligence does not improve out-of-sample performance. What the model does deliver is the explanation: when the two cars' recommendations differ, `explain_difference` names the cause with both numbers and their evidence (Australia: pace level −0.31 vs +0.71 s/lap on 7 / 16 laps, MEDIUM degradation 0.032 vs 0.018, a 0.3 s difference over a 25-lap stint), and when they do not, it says the difference is within noise.

**Issue 6 — fewer variables.** `docs/v4_feature_audit.md` inventories every input with file and line. Removed or demoted: the grid-start penalty (calibrated to 0.0 in every fold, inert), the dead regime-multiplier path, the historical driver factors (now a shrunk prior only, and `no_hist_driver_factors` measures them). Humidity, wind and straight-line speed do not exist in the code; apex features are diagnostic only. Kept as physically necessary even though seven weekends cannot rank them: the grip budget and cliff, fuel mass, track evolution, pit loss, the safety-car credit, per-circuit dirty air, the stint caps. Eight new ablations (§4).

**Issue 7 — tyre life as a range.** Measured, then built (§1). Every stint beyond the practice support carries a widening measured on the seven weekends; the life table states `life_lo`/`life_hi` and the censored share; the live action table carries `p_cliff_before_stop` and the life quantiles per action, shown and never decisive. At the measured width it changes no plan (E5); Belgium's 28-lap SOFT stint carries `p_cliff` 0.066 and a 0.22 s premium.

**Issue 8 — live.** Competitor selection, the rejoin projection, the pit response, the decision with its confidence and hysteresis, and the explanations are in (§1, §3.2). The five questions — pit now, wait, how long, why, where we rejoin, who we are racing, what if they cover — are answered in `plan["decision"]` and read out by `src/explain.py`. Tick 92 ms on this machine, +8–13 ms for the race state.

**Issue 9 — Belgium.** Diagnosed, not patched. The 28-lap SOFT stint sits at exactly twice the practice support (14 laps) on a fitted SOFT rate (0.158 s/lap) below MEDIUM's (0.179); the model's SOFT cliff is 38.6 laps away, so the stint pays no cliff cost, and the field's M-H (18 of 19) loses to M-S by 5.5 s of tyre. The measured widening adds 0.22 s where 2.3 s would move the stop; a physical ladder projection of the rates (softer degrades at least as fast) changes no plan either. The cause is a tyre-model level error on a 103-clean-lap practice weekend (its clean-lap gate also fails), and the generalisable statement is a rule, not a cap: a compound whose fitted rate inverts the ladder on a thin practice is not to be trusted at the edge of its support. `tests/test_v4_tyrelife.py` pins the mechanism on the real Belgium posterior.

**Issue 10 — family vs execution.** Preserved in the objective (§2): history enters the execution layer only as the rivals' behaviour. Where the family layer moved in V4-final it moved because of the recalibrated `τ`, not because of the race state (E6b).

**Issue 11 — integrity.** The methodology was frozen before the run (`docs/v4_methodology.md`, with its three pre-run amendments and the E1 outcome); every constant carries its source in the code; the estimator and the extrapolation width were fixed before the benchmark; the rival mode was decided by the experiment the specification prescribes; no circuit-specific rule was added; the one post-hoc temptation — a different `τ` tie-break at Barcelona — was declined and is reported instead. Two of the implementation agents were cut off by an API rate limit mid-task (the live engine, after its own tests had passed; the UI, mid-verification); their work was recovered from their worktrees and is covered by the final test run.

## 6. Failures and remaining weaknesses

- **The strategy-family layer regressed, through the calibration.** Sequence 6 → 5, stop count 7 → 5, oracle regret 10.9 → 12.7 s, field share on the sequence 0.35 → 0.22. Every one of these is Australia (τ = 6) and Barcelona (τ = 2.5). The second-stop objective for `λ` has two or three donors per fold; the shape objective for `τ` decides on one flag. Seven weekends do not identify these constants and the release says so; the next calibration needs a criterion that does (a hierarchical pooling across folds, or a plan-family likelihood over all finishers rather than the recommended plan's flags).
- **Gates: 5 failing (Task 1: 4).** Australia's two fit-stage gates and Belgium's two are Task 1's; Austria's is new (a 28-lap final MEDIUM stint against an observed maximum of 27, from `λ = 0.45` moving the second stop earlier).
- **Belgium** is not closed (Issue 9).
- **The rival models under-disperse.** Neither beats a uniform stop lap on likelihood; the heterogeneous field is in the code for the day that is fixed.
- **Live:** Hungary's stop calls are a little worse than Task 1's (in-window 0.565 → 0.522, box-now 0.30 → 0.51 s) while Barcelona's are better; the ten Haas stops are worse with the race state on than off; 17 % of car-laps still change the call for no reason the state records; the box-now cost the lap before a real stop rose from 1.03 to 1.24 s.
- **The Haas model is mostly a mirror of the team.** Warm-up and traffic sensitivity are identifiable on two car-weekends each; push response never. The value it adds pre-race is the explanation, not the plan.
- **Not done:** the live engine still assembles its objective from four `WeekendModel` fields rather than a `V4Objective` (behaviourally identical, stated in the audit); `bench_stability`/`bench_apex` spell their keywords out; the outlook's sealed-fit path for the two live weekends was not re-sealed (the fits are Task 1's, by design).

## 7. Reproduce

```
.venv/bin/python scripts/80_recalibrate.py --rival-mode symmetric      # 539 s
for e in australia-2026 japan-2026 barcelona-2026 austria-2026 belgium-2026 hungary-2026 italy-2026; do
  .venv/bin/python scripts/10_pipeline.py --event $e --stage decide --offline; done   # 15-26 s each
zsh bench/run_all.sh                                                    # 954 s; then bench/out/v4_compare.json
.venv/bin/python bench/md2pdf.py results_v4.md results_v4.pdf
```

`bench/bench_experiments.py --variants E1 E6b` re-runs any experiment row; `bench/bench_wpd_live.py --k 3 4 5 6 --hysteresis 0.0 0.3 0.6 --selection` the live choices; `bench/bench_place_value.py`, `bench/bench_extrapolation.py` the two measurements. The Task 1 baseline is under `bench/v4_task1/`, V3 under `bench/v3/`.
