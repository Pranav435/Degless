# Degless V4 — Complete the Race-State Strategy System

<mission>

You are the senior F1 strategy/modeling engineer responsible for completing Degless V4.

Task 1 has already been implemented successfully. Your job is NOT to rebuild Task 1.

Your job is to:

1. inspect the current repository and Task 1 implementation,
2. identify and resolve the remaining weaknesses,
3. make the race-state model more realistic and statistically defensible,
4. reduce unnecessary/noisy model inputs,
5. make Haas and its two cars first-class strategic protagonists,
6. make every strategy-facing code path use the same coherent objective,
7. improve live and offline strategy quality without sacrificing Task 1's first-stop improvement,
8. update the UI so it communicates the Haas race-strategy system properly,
9. run the complete benchmark,
10. compare the final result directly against V4 Task 1,
11. produce the final `results_v4.md`,
12. produce the final `results_v4.pdf`.

Use:

- **Fable** for planning, decomposition, architecture review, and identifying dependencies before implementation.
- **Opus** for the majority of investigation, modeling, algorithm design, coding, refactoring, and difficult debugging.
- **Sonnet** for testing, skeptical review, benchmark verification, and documentation/report generation.

Do not stop at a plan.

MAKE THE CHANGES.

</mission>


<authoritative_baseline>

Treat the existing V4 Task 1 result as the immediate baseline.

Task 1 was run on the same:

- seven dry conventional 2026 weekends,
- 196 race stints,
- benchmark methodology,
- benchmark scripts.

Task 1 achieved:

| Metric | V3 | V4 Task 1 |
|---|---:|---:|
| Mean first-stop error | 4.333 laps | **1.667 laps** |
| Signed first-stop error | +4.33 laps | **+1.00 lap** |
| Field stops inside model window | 0.361 | **0.448** |
| Live stops within ±3 laps | 0.472 | **0.547** |
| Live median stop error | 3.5 laps | **3.0 laps** |
| Live box-now cost before actual stop | 1.755s | **1.030s** |
| Sequence match | 6/7 | 6/7 |
| Start compound match | 7/7 | 7/7 |
| Stop-count match | 7/7 | 7/7 |
| Oracle regret | 9.399s | **10.917s** |
| Live tick, BCN/HUN median | 287/317ms | **298/341ms** |
| Tests | 43 pass, 2 skip | **53 pass, 2 skip** |
| Gates failing | 3 | **4** |
| Per-car plan-shape agreement | 0.862 | **0.721** |

The most important Task 1 result is:

> Disabling race state reproduces V3's first-stop error of 4.33 laps. Full race state achieves 1.67 laps.

Therefore the Task 1 race-state concept is valuable and must be preserved.

Do not sacrifice this improvement merely to simplify the code.

</authoritative_baseline>


<task1_architecture>

Task 1 introduced `src/racestate.py`.

Its core concept is:

For a candidate stop lap for Haas and candidate stop laps for relevant rivals, price the race-time consequences of each combination and estimate the probability of retaining/losing track position.

The model currently considers:

- tyre cost,
- pit loss,
- compound,
- tyre age,
- undercut,
- overcut,
- traffic,
- SC/VSC,
- competitor response,
- expected position loss,
- value of a position.

The live engine evaluates:

- PIT NOW,
- STAY OUT 1 LAP,
- STAY OUT 2 LAPS,
- STAY OUT 3 LAPS,
- PIT AT EDGE OF WINDOW.

The pre-race system simulates a pack of four rivals.

Read the actual implementation before changing any of this.

Do not assume the report perfectly describes the code.

The code is authoritative for implementation details.
The benchmark is authoritative for measured performance.
</task1_architecture>


# Remaining issues to resolve

<issue_1_heterogeneous_competitor_model>

## ISSUE 1 — The pre-race rival pack is unrealistically symmetric

This is the most important modeling weakness remaining in Task 1.

Task 1 simulates a pack in which all rivals run the same strategy family and converge toward a symmetric fixed point.

That is useful for estimating strategic pressure, but it cannot reproduce heterogeneous race behaviour.

For example, at Barcelona:

- soft starters stopped earlier,
- medium starters covered,
- the field did not behave like four identical copies of our strategy.

The symmetric pack therefore cannot naturally create a first mover.

### Required change

Replace the assumption:

> all rivals run our strategy family

with a heterogeneous competitor model.

The pre-race competitor simulation must sample or construct plausible rival strategy states from the empirical distribution available without race leakage.

At minimum distinguish:

- starting compound,
- likely strategy family,
- first-stop distribution,
- pit-window distribution,
- tyre-life distribution,
- pace/degradation profile.

Do not give every rival an identical plan.

For each simulated rival, independently sample from an appropriate leave-one-out distribution.

The model should be able to produce situations such as:

- Rival A starts SOFT and is likely to stop early.
- Rival B starts MEDIUM and is likely to extend.
- Rival C is an undercut threat.
- Rival D is unlikely to react immediately.

The model must not know what actually happened in the benchmark race.

### Important

Do NOT hard-code:

> Barcelona soft starters pit on lap X.

The model must learn this from eligible historical/current-weekend information.

Use compound-conditioned and strategy-family-conditioned distributions where sample sizes permit.

Shrink aggressively when sample size is weak.

### Required experiment

Compare:

- Task 1 symmetric pack
- heterogeneous pack
- heterogeneous pack with no historical strategy prior
- final model

Measure:

- first-stop error,
- field window share,
- oracle regret,
- sequence/start/stops,
- per-weekend first-stop error.

If heterogeneous modeling does not generalize, document why and keep the simpler model where appropriate.

</issue_1_heterogeneous_competitor_model>


<issue_2_place_value>

## ISSUE 2 — The value-of-position estimate is statistically fragile

Task 1 currently estimates:

`place value = V × (2ψ − 1)`

with approximately:

- V = 4.6–5.9s
- ψ = 0.68–0.81
- place value = 2.0–3.6s

The persistence estimate is based on only approximately 25–35 cycle pairs.

This is a major concern because the position-value term directly influences pit timing.

### Required change

Audit the estimator statistically.

Determine:

- exact sample count per leave-one-out fold,
- variance,
- sensitivity to individual observations,
- sensitivity to lapped cars,
- sensitivity to race position,
- sensitivity to SC/green-flag conditions,
- sensitivity to the definition of a "cycle pair."

Do NOT select the definition based on whichever produces the best first-stop benchmark.

The Task 1 report explicitly disclosed that the first implementation used lead-lap finishers, produced 2.33 laps error, and was then changed to all classified finishers.

Do not repeat this pattern.

### Required solution

Develop a defensible estimator using some combination of:

- robust pooling,
- shrinkage,
- confidence intervals,
- sample weighting,
- position-aware estimates,
- classified-finisher information.

The exact method should be chosen from evidence.

The model must not pretend a 25-observation estimate is highly precise.

Represent uncertainty if appropriate.

### Required experiment

Compare:

1. current Task 1 estimator,
2. lead-lap-only estimator,
3. all-classified-finisher estimator,
4. statistically regularized estimator.

Do not select purely by benchmark score.

Choose the estimator that is most statistically defensible while retaining strategy performance.

</issue_2_place_value>


<issue_3_oracle_regret>

## ISSUE 3 — Oracle regret increased from 9.40s to 10.92s

Do not blindly optimize this metric.

Understand why it happened.

Task 1 deliberately introduced track-position value while the oracle is based on pure race time.

Therefore the increase may represent a legitimate trade:

> sacrifice some theoretical pure race time to preserve a position that matters in an actual race.

However, the benchmark currently cannot determine whether that trade is actually beneficial.

### Required change

Add at least one additional strategy-quality metric that evaluates position-aware performance.

Possible metrics include:

- expected finishing position,
- actual finishing position relative to strategy counterfactual,
- position-weighted regret,
- race-time regret plus validated position value,
- expected positions lost/gained at pit cycle,
- position retention probability.

Do not invent a metric merely to make V4 look better.

Define the metric mathematically.

Explain exactly what it measures.

Keep the existing pure-time oracle regret for continuity with V3.

The final report must show BOTH:

- pure-time oracle regret,
- position-aware strategic regret/quality.

Do not delete the original metric.

</issue_3_oracle_regret>


<issue_4_objective_inconsistency>

## ISSUE 4 — Some parts of the application still use the V3 objective

Task 1 explicitly identified that:

- outlook,
- counterfactual,
- desk `evaluate_plans`

still price V3's objective.

`calibration.json` was also not recalibrated for the new strategy objective.

This is unacceptable for the final V4 architecture.

### Required change

Audit every strategy-facing path.

Search the repository for:

- old V3 objective,
- old undercut exposure,
- first-stop history prior,
- old strategy score,
- V3 `evaluate_plans`,
- duplicated strategy scoring.

Classify every occurrence as:

- production decision path,
- live decision path,
- diagnostic,
- benchmark baseline,
- historical compatibility.

Production strategy paths must use one coherent V4 objective.

V3 scoring may remain only where it is explicitly labelled as a V3 baseline/diagnostic.

Ensure:

- offline strategy,
- live RaceEngine,
- counterfactual analysis,
- strategy desk,
- outlook,
- UI recommendation,
- benchmark output

do not silently disagree about what "best strategy" means.

### Calibration

Determine whether V4 requires recalibration of:

- τ,
- λ,
- κ,
- position value,
- strategy-family parameters,
- any other objective-dependent parameter.

Use proper leave-one-out calibration.

Do not calibrate on the benchmark race being evaluated.

</issue_4_objective_inconsistency>


<issue_5_per_car_haas_model>

## ISSUE 5 — Per-car strategy divergence increased

Task 1's same-shape share fell:

`0.862 → 0.721`

This is NOT automatically a failure.

The correct Haas system should sometimes make different decisions for Ocon and Bearman.

But divergence must be caused by meaningful race-state/car-state differences, not noise.

### Required change

Build a compact hierarchical Haas car model.

First-class cars:

- #31 Esteban Ocon
- #87 Ollie Bearman

Do not hard-code driver characteristics.

For each Haas car estimate from current-weekend evidence:

- current tyre pace,
- degradation,
- tyre-age sensitivity,
- tyre warm-up,
- long-run consistency,
- traffic sensitivity,
- push/management response,
- sector degradation,
- pace relative to teammate.

Use hierarchical shrinkage:

`driver/car evidence → Haas team evidence → field evidence`

When evidence is insufficient, converge toward the team/field estimate.

### Critical requirement

A difference between Ocon and Bearman should be explainable.

For every materially different recommendation, the system should be able to identify the cause, e.g.:

- Ocon is 0.7s behind a rival with 4 laps older tyres,
- Bearman has a 2.1s gap to a car whose pit window is opening,
- Ocon's tyre degradation is materially higher this stint,
- Bearman is projected to rejoin in traffic.

Do not create "Ocon profile" or "Bearman profile" lore without data.

### Required benchmark

Measure whether driver-specific intelligence improves:

- individual stop timing,
- individual race-state decisions,
- stint prediction,
- position outcome.

If it does not improve out-of-sample performance, shrink or remove it.

Do not force independence for its own sake.

</issue_5_per_car_haas_model>


<issue_6_variable_noise>

## ISSUE 6 — Reduce unnecessary variables

The goal is NOT maximum feature count.

Audit the entire feature set.

Prioritize variables that directly influence race decisions.

### CORE

Keep strong emphasis on:

- compound,
- tyre age,
- degradation,
- current tyre pace,
- near-term tyre pace,
- tyre-life uncertainty,
- fuel/mass,
- track evolution,
- sector-level pace,
- push/management,
- current-weekend Haas evidence,
- current position,
- gap ahead,
- gap behind,
- competitor compound,
- competitor tyre age,
- competitor pit status,
- rejoin position,
- traffic,
- pit loss,
- SC/VSC.

### SUPPORTING

Use when evidence exists:

- apex degradation,
- circuit-specific dirty air,
- race/practice degradation regime,
- compound nomination mapping,
- historical strategy distributions,
- temperature when race-day temperature information is actually available.

### DEPRIORITIZE

Do not allow these to become major strategy drivers without evidence:

- humidity,
- wind,
- generic straight-line speed,
- historical global driver factors,
- excessive apex features,
- generic temperature corrections,
- duplicate traffic features,
- arbitrary historical bonuses,
- arbitrary constants.

For every removed/demoted feature, run an ablation where practical.

Do not remove physically necessary features merely because they are weak in a seven-weekend benchmark.

</issue_6_variable_noise>


<issue_7_tyre_life_uncertainty>

## ISSUE 7 — Tyre life should be uncertainty-aware

V3/V4 correctly recognize that many long stints provide lower bounds rather than exact tyre-life measurements.

Preserve this.

Do not represent:

`28 laps`

as if it means:

> "the tyre will definitely be optimal for 28 laps."

### Required change

Propagate tyre-life uncertainty into strategy.

For each candidate action estimate something equivalent to:

- expected tyre cost,
- probability of excessive degradation,
- probability of reaching a cliff,
- uncertainty in remaining useful life.

The strategy should distinguish:

`28 ± 1 laps`

from:

`28 ± 5 laps`.

The pit decision should therefore be based on expected outcome and risk, not only a point estimate.

Do not make cliff probability the primary decision signal.

The primary signal remains:

> expected value of PIT NOW versus STAY OUT.

</issue_7_tyre_life_uncertainty>


<issue_8_live_strategy>

## ISSUE 8 — Improve live strategy without making the engine unnecessarily expensive

Task 1 improved live strategy accuracy but increased tick cost approximately 6–9%.

Task 1 currently:

- re-prices against four nearby cars,
- uses real gaps,
- uses tyre/stop information,
- looks ahead approximately four laps for traffic.

### Required change

Preserve the useful live race-state calculation.

Improve:

- competitor selection,
- rejoin prediction,
- traffic prediction,
- pit-response modeling,
- recommendation stability.

Investigate whether four rivals is sufficient.

Use the nearest strategically relevant cars rather than blindly increasing the number of cars.

Do NOT increase computational complexity without benchmark evidence.

Keep live strategy comfortably below:

**200 ms per tick**

on the established benchmark environment.

### Live recommendation requirements

The live engine must clearly answer:

- PIT NOW?
- WAIT?
- HOW LONG?
- WHY?
- WHAT POSITION WILL WE REJOIN IN?
- WHO ARE WE RACING?
- WHAT HAPPENS IF THEY COVER?

Do not generate unstable recommendations where the answer flips every lap without a meaningful state change.

</issue_8_live_strategy>


<issue_9_belgium_gate_and_strategy_feasibility>

## ISSUE 9 — Fix the new Belgium failure without hard-coding Belgium

Task 1 introduced a gate failure because a lap-16 stop implies a 28-lap SOFT stint while the longest observed Belgian run was 26 laps.

Investigate the actual cause.

Do not simply raise/lower the Belgian tyre-life cap.

Determine whether the issue is:

- compound feasibility,
- tyre-life uncertainty,
- strategy-family generation,
- nomination mapping,
- circuit stint cap,
- race-state movement,
- pack simulation.

The fix must generalize to other circuits.

Add a regression test reproducing the failure.

</issue_9_belgium_gate_and_strategy_feasibility>


<issue_10_strategy_family_vs_execution>

## ISSUE 10 — Separate "what strategy" from "when to execute it"

Task 1 correctly preserved V3's strategy family:

- sequence = 6/7
- start compound = 7/7
- stop count = 7/7

The major improvement was moving the execution timing earlier.

Preserve this hierarchy:

### Strategy-family layer

Answers:

> What broad tyre strategy is plausible?

Uses:

- compound nomination,
- historical strategy distribution,
- tyre life,
- degradation,
- race distance,
- circuit constraints.

### Race-execution layer

Answers:

> When should Haas execute the next stop?

Uses:

- current position,
- nearby competitors,
- tyre state,
- traffic,
- undercut/overcut,
- rejoin,
- SC/VSC,
- current race state.

Do not let historical strategy priors override the race-execution layer.

</issue_10_strategy_family_vs_execution>


<issue_11_benchmark_integrity>

## ISSUE 11 — Protect benchmark integrity

The Task 1 report contains a disclosure that one place-value definition was changed after observing the first result.

Do not repeat that process.

For all V4 work:

- define methodology before final benchmark execution,
- perform tuning only with eligible leave-one-out data,
- do not inspect target race outcomes while tuning,
- do not change benchmark definitions after seeing V4 results,
- do not create circuit-specific exceptions based on benchmark failures,
- do not use actual race stops to calibrate the same race.

Every parameter must have a documented source:

- leave-one-out measurement,
- physical constant,
- prior,
- externally fixed value,
- or explicitly justified engineering constant.

If an engineering constant is introduced, document why it exists and test sensitivity.

</issue_11_benchmark_integrity>


# Haas product requirements

<haas_identity>

The final application must be explicitly built around Haas.

The product should communicate:

> **Degless — Haas Race Strategy**

The two protagonists are:

- **#31 Esteban Ocon**
- **#87 Ollie Bearman**

The system's strategic antagonists are:

- traffic,
- tyre degradation,
- undercuts,
- overcuts,
- midfield compression,
- pit loss,
- Safety Cars/VSCs,
- competitors,
- bad timing,
- uncertainty.

Do not turn this into a fantasy game.

The tone should be a professional Haas race-engineering tool.

</haas_identity>


<ui>

Inspect the existing UI and make the necessary changes.

At minimum implement:

## Haas Overview

Two prominent live car panels.

### Ocon

Show:

- position
- compound
- tyre age
- current pace
- degradation
- gap ahead
- gap behind
- recommended action
- pit window
- confidence
- projected rejoin position
- expected time delta
- relevant rival

### Bearman

Same structure.

Do not fabricate values.

## Haas Pit Wall

Central decision panel.

For each car:

- PIT NOW
- STAY OUT
- WAIT X LAPS
- BOX BY LAP X
- confidence
- projected position
- expected race-time delta
- principal reason

## Why?

Every recommendation must explain the actual decision.

Example:

> **BOX OCON**
>
> Staying out 2 laps costs 1.4s in tyre performance.
> Mercedes behind is entering its pit window.
> Pitting now is projected to retain P15.
> Confidence: 78%.

These explanations must be generated from model state.

Never hard-code strategy explanations.

## Race field

Show:

- Ocon
- Bearman
- nearby rivals
- positions
- gaps
- tyre compounds
- pit status

## Strategy comparison

Show candidate:

- 1-stop
- 2-stop
- 3-stop

with:

- expected race time,
- expected position,
- pit laps,
- compound sequence,
- uncertainty/risk.

## Driver views

Allow:

- Haas Overview
- Ocon
- Bearman

Each driver view should answer:

> What should this Haas car do right now?

Keep model internals in an advanced/debug view.
</ui>


# Architecture requirements

<architecture>

Do not create an enormous monolithic strategy function.

Maintain clear conceptual separation:

## Tyre model

Answers:

> How is the tyre behaving?

## Car/Haas model

Answers:

> How is this Haas car behaving?

## Race-state model

Answers:

> What is happening around us?

## Strategy-family model

Answers:

> What broad strategy is plausible?

## Race-execution model

Answers:

> What should Haas do now?

## Presentation/UI

Answers:

> How do we communicate the decision?

The final recommendation should flow approximately as:

`telemetry/data`

→ `tyre state`

→ `Haas car state`

→ `race state`

→ `candidate actions`

→ `expected race outcomes`

→ `PIT/STAY/WINDOW recommendation`

→ `explanation`

Do not duplicate strategy scoring in multiple places.

</architecture>


# Required experimental program

<experiments>

Do not implement changes without measuring them.

At minimum run these controlled experiments:

## E0 — Task 1 baseline

Current Task 1 code.

## E1 — Heterogeneous rivals

Replace symmetric pack with independently sampled competitor states.

## E2 — Position-value estimator

Compare current and statistically regularized position value.

## E3 — Reduced feature set

Remove/demote low-value features.

## E4 — Haas hierarchical model

Add current-weekend Ocon/Bearman intelligence.

## E5 — Tyre-life uncertainty

Propagate uncertainty into action scoring.

## E6 — Objective unification

Replace remaining V3 production scoring with the V4 objective.

## E7 — Live competitor/rejoin improvements

Improve live competitor and rejoin modeling.

## E8 — Full V4

Combine only the components that survive validation.

For each experiment record:

- first-stop error,
- signed first-stop error,
- field window share,
- oracle regret,
- position-aware metric,
- sequence match,
- start compound match,
- stop count,
- live ±3-lap accuracy,
- live median error,
- box-now cost,
- live precision/recall,
- runtime,
- tests.

Do not retain a feature simply because it improves one weekend.

</experiments>


# Required benchmark procedure

<benchmark>

After implementation:

1. Run the complete existing test suite.
2. Run the complete seven-weekend benchmark.
3. Use exactly the same benchmark methodology as V3 and Task 1.
4. Run leave-one-out calibration where required.
5. Run the live benchmarks.
6. Run all required ablations.
7. Compare final V4 directly against Task 1.
8. Re-run the final benchmark after all code fixes.

The final benchmark MUST correspond to the final code.

Do not benchmark, modify code, and report the old benchmark as final.

Use the repository's established commands, including where applicable:

```bash
make history
make benchmark