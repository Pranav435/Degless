# degless benchmark — accuracy, intelligence, speed

*Benchmarked 12 September 2026 on the 7 dry, conventional 2026 weekends the tool has scored (Australia, Japan, Barcelona, Austria, Belgium, Hungary, Italy). Reproduce with `bench/run_chain.sh`; raw outputs are in `bench/out/`.*

## 1. Summary

| Metric | Result | Verdict |
|---|---|---|
| Degradation-rate error vs the race (stint-rate MAE, shipped curves) | mean 0.109 s/lap, median 0.041; 5/7 weekends under the 0.15 target | Two weekends (Australia 0.43, Italy 0.15) are broken by one step of the pipeline |
| Same, with the circuit-history step removed | mean 0.037 s/lap; 7/7 under target; equal to an in-sample oracle (0.036) | The practice fit is at the noise floor of the metric |
| Bayesian model vs a MixedLM-times-regime baseline | 0.037 vs 0.040 s/lap | The Bayesian machinery buys intervals, not accuracy |
| 90 % intervals on the stint rate | 95–100 % coverage, width 1.5× the noise floor | Honest but ~40 % too wide |
| Stop count vs the field's mode | 7/7 | Strong |
| Compound sequence run by at least one finisher | 3/7 (0 % of the field at Australia, Japan, Belgium, Hungary) | Weakest decision |
| First-stop lap vs field median (non-safety-car weekends) | model later by 3–6 laps; 14–78 % of real stops inside the window | Systematic late bias |
| Tyre life vs longest stint actually run | over-estimated on 19/19 compound-weekends, median 1.7× | Life numbers are not usable as stated |
| Oracle regret of the tool's plan (true race degradation) | mean 9 s; beats the field's modal plan 6/6, the winner's plan 5/6 | Good, under the model's own cost structure |
| Prior-only outlook (no practice) matching the field's modal plan | 5/7, vs 3/7 after practice | Practice data makes the compound choice worse |
| Live engine (Hungary and Barcelona replays) | ~1 s per lap for the field; 25–48 % of real stops inside its window, median miss 3.5–4 laps; cliff alarm before 0/46 stops at Hungary | Usable, same late bias as offline, cliff not informative |
| Apex-speed channel (the model's second data channel) | +0.001 s/lap accuracy; fuel-sensitivity posterior unchanged (sd = prior); moves 6/7 plans onto a SOFT start; 3× fit time | Net negative for the decision |
| Lap-time-only practice fit, no history, no apex | matches the field's modal plan 6/7, stop count 6/7 | Best plan-shape result of any variant tested |
| Stability | plan fixed after FP2 at Barcelona and after FP2 at Hungary; three seeds move slopes < 0.001 s/lap; quick fit (2×800) = production accuracy | Strong |
| Speed | production fit 30 s clean (247 s under disk contention); everything after the fit < 6 s; strategy search 0.6 s; live tick ~1 s; full retrospective pipeline 4.5–11 min (5 fits) | Fit is the only slow stage |
| Test suite | 20/20 pass (97 s) | |

**Bottom line.** The degradation *rate* is predicted as well as it can be measured, the stop count is right every time, and the answer is stable across seeds and settles after FP2. What limits the tool for an analyst is (a) one defective step (the circuit-history fold-in) that corrupts two of seven weekends, (b) a compound-choice model that is contradicted by every race it has seen and made worse by the apex-speed channel, (c) tyre-life and cliff numbers that are physically implausible at low-degradation circuits, and (d) no notion of track position, which makes every first-stop call late. All four are fixable with the data already on disk; the simplest configuration tested (lap-time-only fit, no history fold-in) already beats the shipped one on every axis.

## 2. What was benchmarked and how

* **Data.** Practice long runs (103–269 clean laps per weekend) in, sealed pre-race curves out, scored against 196 race stints (3,313 clean race laps) that the fitter never saw. Race data are read only through the scoring and backtest paths.
* **Accuracy protocol.** The project's own scorer (`src.validate.score_race`: per-stint degradation rate from the first three to the last three laps, stint-centred) plus a stint-rate 90 % interval check and Spearman rank correlation, run on the shipped curves and on seven alternatives with identical inputs: practice-only posterior, practice posterior with no practice-to-race transfer, circuit history alone, MixedLM slope × transferred regime factor, leave-one-out season mean rate, zero degradation, and an in-sample oracle (this race's own measured rates).
* **Decision protocol.** Recommended plan vs classified finishers (stop count, compound sequence, start compound, first-stop window), predicted tyre life vs the longest stint run, and *oracle regret*: every plan re-priced on a tyre model whose rates are the ones measured on that race, so the tool's plan, the field's modal plan and the winner's plan are compared in seconds under the truth.
* **Live protocol.** The archived Hungary and Barcelona race feeds replayed through `RaceEngine`, one tick per lap; per-tick latency, and every real pit stop compared with the engine's call 3 laps and 1 lap earlier.
* **Ablations.** The circuit-history step and the apex-speed channel each switched off, on all 7 weekends, scored and run through the strategy search; the prior-only outlook (no practice) built for each weekend.
* **Speed and stability.** Every pipeline stage timed in-process on Barcelona (the largest search), the slow stages twice; quick fits on FP1, FP1+FP2, FP1–3 and three seeds at Barcelona and Hungary.
* **Hardware.** Apple M4, 10 cores, 16 GB, JAX on CPU, Python 3.13.5. A Time Machine backup and Spotlight indexing ran during the timing session; process start-up (imports) was inflated by it, in-process stage timings were not.

## 3. Accuracy: the degradation curves

![](bench/out/fig/fig1_accuracy.png)

**Per weekend, shipped curves (practice posterior + circuit history, × regime factor):**

| Weekend | Clean practice laps | Race stints | Rate MAE (s/lap) | Rate bias | Per-lap MAE (s) | 90 % rate coverage | Same, practice-only posterior |
|---|---|---|---|---|---|---|---|
| Australia | 233 | 23 | **0.434** | −0.417 | 4.52 | 0.91 | 0.054 |
| Japan | 269 | 19 | 0.024 | −0.021 | 0.27 | 1.00 | 0.020 |
| Barcelona | 225 | 31 | 0.041 | +0.013 | 0.38 | 1.00 | 0.041 |
| Austria | 262 | 40 | 0.032 | −0.010 | 0.35 | 1.00 | 0.030 |
| Belgium | 103 | 19 | 0.030 | −0.009 | 0.29 | 1.00 | 0.038 |
| Hungary | 184 | 43 | 0.046 | +0.017 | 0.42 | 0.95 | 0.049 |
| Italy | 262 | 21 | **0.154** | −0.153 | 1.75 | 0.86 | 0.029 |
| **Mean** | | 196 | **0.109** | −0.083 | 1.14 | 0.96 | **0.037** |

**Against the baselines, pooled over the 7 weekends:**

| Curve source | Rate MAE mean | Rate MAE max | Bias | 90 % coverage | Interval width | Spearman (stint ranking) |
|---|---|---|---|---|---|---|
| Oracle (this race's own rates, in-sample) | 0.036 | 0.048 | +0.008 | 0.83 | 0.16 | +0.10 |
| Practice posterior × regime (no history) | **0.037** | 0.054 | +0.011 | 0.96 | 0.24 | −0.13 |
| MixedLM slope × regime (frequentist baseline) | 0.040 | 0.064 | −0.003 | 0.83 | 0.16 | +0.03 |
| Circuit history 2023–25 only | 0.042 | 0.052 | −0.004 | 0.82 | 0.15 | +0.07 |
| Other 2026 races' mean rate (one number per compound) | 0.047 | 0.087 | +0.021 | 0.76 | 0.16 | −0.04 |
| Practice posterior, no regime transfer | 0.054 | 0.088 | −0.029 | 0.87 | 0.23 | −0.12 |
| Zero degradation | 0.068 | 0.119 | +0.065 | 0.54 | 0.16 | — |
| **Shipped (practice + history) × regime** | **0.109** | 0.434 | −0.083 | 0.96 | 0.53 | −0.07 |

Point baselines carry only the scorer's observation noise as their interval, so 0.16 is the width floor of this metric.

**Findings**

1. **The circuit-history fold-in is net negative.** It is within ±0.01 s/lap of the practice-only curve on five weekends and destroys the other two. Mechanism, verified in the saved posteriors: at Australia and Italy the practice fit finds almost no degradation (linear term 0.001–0.010 s/lap, post-knee hinge 0.03–0.05 s/lap). `apply_circuit_prior` then rescales *both* the linear term and the hinge by the same 30–80× factor, producing a post-knee slope of 2.5–4.5 s/lap at Australia (0.4–0.8 at Italy). The sealed race curve collapses after lap ~17, the prediction is 0.15–0.42 s/lap too pessimistic, per-lap error reaches 4.5 s and the 90 % interval under-covers.
2. **The practice fit is at the noise floor.** Without the history step it matches the in-sample oracle (0.037 vs 0.036) on every weekend, including Belgium with only 103 clean laps.
3. **The Bayesian model adds intervals, not accuracy.** MixedLM × regime is within 0.003 s/lap of it. The regime transfer itself is worth 0.017 s/lap and removes a −0.03 s/lap bias.
4. **Stint ranking is not predicted.** Spearman correlation between predicted and observed stint rates is ≈ 0 for every method, including the oracle: within a weekend the stint-to-stint spread is driver, traffic and management, and the model carries none of it into the strategy.
5. **Intervals are honest but wide.** Rate-level 90 % intervals cover 95–100 % at 1.5× the noise-floor width; per-lap 90 % coverage is 97–100 % on the good weekends. `sigma_obs` is the practice per-lap noise (0.74–1.05 s) while race stints are scored centred; the regime factor's log-sd (0.35–0.48) is wider than the measured spread (0.31).
6. **The cliff is not predicted.** The knee posterior equals its prior on every weekend (SOFT 13–19, MEDIUM 16–22, HARD 20–25 laps regardless of circuit); where a race knee could be detected the mean error is 7.0 laps (11 cases, range 2–14).
7. **The transferred regime factor is biased low.** Transferred 0.50–0.60 vs self-measured 0.58–0.81 on six of seven weekends (Belgium 0.32 is the exception that drags the geometric mean); mean absolute log error 0.31, i.e. ±36 %. The factor is not a constant across circuits and is currently pooled as if it were.
8. **The apex-speed channel does not do what it is there for.** Refitting every weekend on lap times alone (same settings, same priors) gives a rate MAE of 0.039 vs 0.037 with the 900 apex rows, and the fuel-sensitivity posterior has a relative sd of 0.25–0.26 either way, identical to its prior: the fuel/age split is set by the prior, not identified by the data, with or without the second channel.

| Practice-only fit | Rate MAE mean | k_track relative sd (prior 0.25) | Fit time | Plans matching the field's modal sequence |
|---|---|---|---|---|
| Lap times + apex speeds (shipped structure) | 0.037 | 0.25 | 30 s (247 s under disk load) | 1/7 |
| Lap times only | 0.039 | 0.26 | 9–11 s | 6/7 |

## 4. Intelligence: the decisions

![](bench/out/fig/fig2_windows.png)

| Weekend | Tool | Field modal plan (share) | Winner | Stops = mode | Sequence run by anyone | Start compound = majority | Oracle regret: tool / field / winner (s) |
|---|---|---|---|---|---|---|---|
| Australia | 1-stop H-M @ 36 | M-H (53 %) | M-H @ 12 | yes | no | no (HARD) | 6.5 / 13.8 / 18.2 |
| Japan | 1-stop S-H @ 21 | M-H (85 %) | M-H @ 22 | yes | no (SOFT not raced) | no (SOFT) | n/a / 1.2 / 1.5 |
| Barcelona | 2-stop M-H-H @ 19,43 | M-H-H (46 %) | S-H-M-H | yes | yes | yes | 1.3 / 5.1 / 6.7 |
| Austria | 2-stop M-H-H @ 23,47 | M-H-H (29 %) | M-H-H @ 19,43 | yes | yes | yes | 18.0 / 20.9 / 20.9 |
| Belgium | 1-stop S-M @ 24 | M-H (58 %) | M-S @ 18 | yes | no | no (SOFT) | 10.8 / 17.3 / 10.7 |
| Hungary | 2-stop M-H-M @ 23,47 | M-H-H (32 %) | M-H-H-S | yes | no | yes | 15.9 / 27.5 / 26.7 |
| Italy | 1-stop M-H @ 23 | M-H (32 %) | H-M-M @ 3,28 | yes | yes | yes | 1.5 / 10.9 / 30.6 |

**Findings**

1. **Stop count is reliable: 7/7 match the field's mode**, and on average 70 % of classified finishers ran the recommended count.
2. **Compound choice is not.** On 4 of 7 weekends the recommended sequence was run by nobody; at Japan the tool starts on a SOFT that no car raced, at Belgium on a SOFT one car in nineteen started on, at Australia on the HARD. The project's own "net stint-level compound step" gate fails on all 7 weekends with the wrong sign: the model prices one step harder at −0.06 to −0.32 s/lap net (harder is *faster* over a stint), the races measure +0.03 to +0.11 (harder is slower). The 1.0 s grid-start penalty exists to paper over this and does not suffice.
3. **The ladder's ordering is not what the races show.** Race degradation rates measured with the project's own estimators order softer-faster on only 3/7 weekends (lap fixed effects) and 3/7 (stint fixed effects); practice long runs order that way on 2/7. Whether the estimator or the assumption is wrong, the model cannot tell, because the ordering is pinned by construction.
4. **First stops are called late.** Excluding the two weekends where a safety car set the first stops (Italy lap 3, Australia), the recommended first-stop lap is 3–6 laps after the field median and 14–78 % of real first stops fall inside the model's window. The field is covering the undercut; the objective has no track-position or undercut term (the undercut calculator exists but is not in the plan cost).
5. **Tyre life is over-stated everywhere.** Predicted life at the chosen push exceeds the longest stint any finisher ran on 19/19 compound-weekends, median ratio 1.7 (Australia HARD 105 vs 46 laps; Italy HARD 93 vs 50). Life is `grip budget / rate`, so a low-degradation practice fit produces nonsense: the practice-only Australia posterior gives 2,900 laps, the Italy outlook quotes a 112-lap HARD for a 53-lap race. The implied grip budget from the races is 1.9–3.7 s, not the 3.8 s constant.
6. **Under true degradation the tool's plan is good.** Oracle regret is 1.3–18 s (mean 9 s) and beats the field's consensus plan 6/6 and the winner's actual plan 5/6. This holds only inside the model's own cost structure (grip budget, pace offsets, no traffic position, no safety car), so it measures the optimiser, not the world.
7. **The counterfactual has no safety-car awareness.** 17 of 145 per-driver "seconds lost" figures exceed 30 s; 13 of them are Italy drivers who pitted on lap 3–4 under the safety car and are charged 30–158 s for a "wrong" stop. Australia lists a 210 s loss for a three-stint SOFT-only race that was a damaged car.
8. **The prior-only outlook beats the post-practice model on compound choice.** With no practice at all (compound ladder + the circuit's 2023–25 races), the outlook picks the field's modal sequence on 5/7 weekends (Japan M-H, run by 85 %) against 3/7 after practice, and the stop mode on 6/7. It builds in 0.5 s. Practice data currently improves the rate and worsens the plan shape.
9. **Per-driver intelligence is absent offline.** The fit estimates a per-driver management term but the strategy, pit window and counterfactual use one field-average curve; the Spearman result above shows the cost of that.
10. **The simplest fit gives the most plausible plans.** The lap-time-only practice posterior, with neither the history fold-in nor the apex channel, recommends M-H at Australia, Japan, Belgium and Italy and M-H-H at Barcelona and Austria: the field's modal plan on 6 of 7 weekends (the miss is a 1-stop M-H at Hungary, where the field two-stopped). The same fit with the apex channel starts on the SOFT at 6 of 7 (S-M-M, S-H-H, S-M, S-H, M-S) and matches the field once. The extra channel is where the SOFT bias enters.

## 5. Live engine

Archived Hungary and Barcelona race feeds replayed through `RaceEngine`, one tick per lap, sealed models from those weekends' practice.

| | Hungary (70 laps, 46 real stops) | Barcelona (66 laps, 53 real stops) |
|---|---|---|
| Tick latency, whole field: mean / p95 / max | 1.16 / 1.38 / 1.43 s | 0.98 / 1.32 / 1.39 s |
| Feed parsing, whole race | 0.3 s | 0.3 s |
| Real stops inside the engine's window 3 laps earlier | 48 % | 25 % |
| Median error, recommended in-lap vs actual | 4.0 laps | 3.5 laps |
| Stops within 3 laps of the recommendation | 33 % | 47 % |
| Cliff alarm raised before the stop | 0 of 46 | 9 of 53 |
| Engine's wear estimate at the moment the car stopped (median, 1 = cliff) | 0.31 | 0.70 |
| "Box now" cost the lap before the real stop (median) | +1.8 s | +3.2 s |
| Regime multiplier: prior → last lap (offline truth) | 0.53 → 0.67 (0.64) | 0.63 → 0.63 (0.70) |

![](bench/out/fig/fig3_live.png)

**Findings**

1. **Latency is fine for a pit wall but has no headroom.** About 1 s per lap for 20 cars, flat from lap 2 to the flag; parsing is negligible, the plan enumeration (≈2,000–3,000 options per car in Python loops) is the whole cost.
2. **Stop calls carry the same late bias as the offline plan.** The engine says "boxing now costs 1.8–3.2 s" on the lap before teams actually box; its window catches a quarter to a half of real stops and misses by 3.5–4 laps in the median.
3. **The cliff does not predict stops.** The alarm preceded 0 of 46 stops at Hungary; teams stopped with the engine's wear estimate at 31 % (Hungary) and 70 % (Barcelona) of the grip budget. The cliff as parameterised is not the constraint teams are managing to.
4. **The regime multiplier converges but wanders.** It ends within 0.03–0.07 of the offline truth, but sits at 0.98 on lap 12 and 0.52 on lap 52 at Barcelona: early in a race, track evolution and traffic are read as tyre behaviour, and the per-car update inherits that.

## 6. Speed

Barcelona 2026 (225 clean laps, 900 apex rows, 620k candidate plans), in-process timers. A Time Machine backup and Spotlight indexing saturated the disk during the first pass; the slow stages were re-timed 20 minutes later. Both numbers are given where they differ.

| Stage | Seconds (first pass, disk contended) | Re-timed |
|---|---|---|
| Import JAX + NumPyro | 48 | 0.6 |
| Load practice laps from the FastF1 cache | 49 | — |
| Lap table, clean cascade, fuel, track evolution | 3.4 | |
| MixedLM baseline, 50 stint bootstraps | 1.6 | |
| **NUTS 4 chains × 1500 + 1500, joint lap-time + apex (production)** | **247** | **30** |
| NUTS 4 × 1500 + 1500, lap-time only | 8.6 | 11 (9–13 across the 7 weekends) |
| NUTS 2 × 800 + 800, joint (`--quick`) | 21 | 12–29 (stability runs) |
| NUTS 2 × 800 + 800, lap-time only | 6.2 | |
| Circuit-history combination, seal, score | < 0.1 | |
| Strategy search, 620k plans, 500 draws, 1-lap grid | 0.64 | |
| Pit-window sweep / undercut window | 0.06 / 0.001 | |
| Counterfactual, all drivers | 4.7 | |
| Strategy-desk calls (evaluate 3 plans, crossover, SC playbook, VOI, duel) | 0.01–0.17 each | |
| Outlook build (300 draws) | 0.5–3.3 | |
| App cold start (Streamlit `AppTest`, project gate < 3 s) | 420 | 1.3 |
| Peak memory | 1.3 GB | |
| Full retrospective pipeline, recorded on quiet runs (5 fits, 4 of them joint) | 270–667 per weekend, mean 447 | |
| Weekend refit script, recorded | 32–37 quick, 199 full | |

**Findings**

1. **The fit is the only slow stage, and it is fast enough.** 30 s for the production joint fit on a quiet disk, 9–11 s lap-time only; the retrospective pipeline's 4.5–11 minutes come from running five fits, four of them joint, plus a 200-bootstrap MixedLM.
2. **Everything downstream is interactive.** The complete decision layer (search, windows, counterfactual, desk tools) runs in under 6 s; the outlook rebuilds in seconds; the dashboard starts in 1.3 s.
3. **The tool is sensitive to disk contention in a way an analyst will notice.** Under a running backup the same fit took 8× longer and the dashboard 7 minutes; JAX compilation and the FastF1 SQLite cache are both file-heavy.

## 7. Stability

Quick fits (2 chains × 800 + 800, apex channel on, circuit history folded in as the weekend script does), 300-draw strategy search, scored against the race.

| Fit | Clean laps | Slopes S / M / H (s/lap, practice) | Best plan | Rate MAE |
|---|---|---|---|---|
| Barcelona, FP1 only | 83 | 0.192 / 0.156 / — | 2-stop S-M-M @ 20,43 | 0.086 |
| Barcelona, FP1+FP2 | 222 | 0.266 / 0.202 / 0.152 | 2-stop M-H-H @ 19,43 | 0.044 |
| Barcelona, FP1–FP3 | 225 | 0.268 / 0.203 / 0.153 | 2-stop M-H-H @ 19,43 | 0.044 |
| Barcelona, FP1–FP3, seeds 1 and 2 | 225 | ±0.001 | identical | 0.044 |
| Hungary, FP2 only | 155 | 0.196 / 0.133 / 0.115 | 2-stop M-H-M @ 23,47 | 0.048 |
| Hungary, FP2+FP3 | 184 | 0.174 / 0.126 / 0.110 | 2-stop M-H-M @ 23,47 | 0.045 |
| Hungary, FP2+FP3, seeds 1 and 2 | 184 | ±0.002 | identical | 0.045 |

**Findings**

1. **Seeds do not matter.** Three seeds move every slope by less than 0.002 s/lap and never change the plan; all fits converge (r-hat ≤ 1.016, zero divergences).
2. **The answer is set after the second practice session.** FP3 changes nothing at Barcelona (3 extra laps) and only the third decimal at Hungary. FP1 alone is not enough: 83 laps, no HARD run, a 0.086 s/lap error and a plan built on two MEDIUM stints.
3. **The quick fit is as accurate as the production fit** (0.044 vs 0.041 s/lap at Barcelona, 0.045 vs 0.046 at Hungary) in 12–29 s instead of 30–250 s.

## 8. Engineering observations

* **Gates.** 23 gates per weekend; failures: Australia 7, Italy 6, Belgium 3, Japan 3, others 1. The ladder gate fails everywhere and is not blocking, so a wrong-sign compound model ships.
* **Constants.** 55 numeric constants in `src/config.py` and 15 in the live engine. The ones that decide the plan (grip budget 3.8 s, pace step 0.21 %, manage cost 0.9 s, wear floor 0.45, grid-start penalty 1.0 s, dirty air 0.45 s) were calibrated on Barcelona and Hungary alone; seven scored weekends now exist and none of them has been re-fitted leave-one-out.
* **Sealing works.** Every sealed file verifies against its sha256; the practice-only firewall raises on Race, Qualifying and Sprint; race data reach nothing but the scorer.
* **Tests.** 20/20 pass; coverage is the live parser and engine plumbing, not the model's numbers (no test would have caught the hinge amplification or the wrong-sign ladder).

## 9. Recommendations

Ranked by expected effect on what an analyst sees. Effort: S = hours, M = days, L = a week or more.

### Accuracy

1. **Fix the history fold-in (S).** In `apply_circuit_prior`, rescale only the rate that enters the tyre model (as `apply_rate_prior_to_model` already does for the outlook) and never the hinge; cap the per-draw scale at 3× and route the combination through `TyreModel` rather than the posterior arrays. Expected: Australia 0.43→0.05, Italy 0.15→0.03 s/lap, pooled mean 0.109→≈0.04, 7/7 under target.
2. **Model the regime factor, do not pool it (M).** Regress the per-weekend race/practice ratio on practice-vs-race track temperature (the thermal sensitivity of +2.5 %/°C is already measured) and pool the residual; use the median, not the geometric mean, so one Belgium cannot pull every weekend 15–30 % low.
3. **Calibrate the intervals (S).** Score with a race per-lap noise (≈0.5 s, the live engine's value) instead of the practice `sigma_obs`, and set the regime log-sd to the measured 0.31. Target 90 % rate coverage of 88–93 % at ~30 % narrower bands.
4. **Drop the hinge from the practice fit (S).** It is unidentified on every weekend and costs sampler time; take the cliff from the circuit's race history (longest and p90 stint per compound, already cached) and report it as such.

### Intelligence

5. **Recalibrate the compound ladder per circuit (M).** Estimate the fresh-tyre pace step and the degradation ratio from the circuit's own 2023–25 races with the lap-fixed-effects regression that `race_deg_slopes` already runs (add a compound intercept), and make the net-stint-step gate blocking. Replace the hard ordering with a soft prior so a race that contradicts it can move it. This is the single change most likely to stop recommending compounds nobody runs.
6. **Put track position in the objective (M).** Add the undercut exposure of each stop lap (the calculator exists) and a cover-the-undercut term fitted to the measured 3–6 lap early bias of the field; report the tyre-optimal lap and the position-optimal lap separately.
7. **Make tyre life honest (S).** Floor the rate used for life at the circuit-history minimum, cap life at the race distance, and print "longer than the race" instead of 2,900 laps; fit the grip budget per compound from the seven races (implied 1.9–3.8 s) instead of the 3.8 s constant.
8. **Safety-car awareness in the backtest and counterfactual (S).** Exclude or flag stops taken under status 4/6/7 or within two laps of a red flag; the lap table already carries track status.
9. **Field prior on plan shape (M).** Combine the circuit's historical start-compound and sequence frequencies with the model's cost as a prior over plan families, so a sequence with zero historical support needs a large time gain to be recommended. The prior-only outlook already does better than the sealed model on this axis; the sealed model should not be allowed to do worse than its own prior.
10. **Per-car recommendations (L).** Carry the per-driver management term (`dev[d,c]`) and a per-team offset from previous 2026 races into the strategy and counterfactual; the live engine already does this per car and should be the template.

### Speed

11. **Lap-time-only fits on race weekends (S).** The apex channel costs 3× the fit time, adds 0.002 s/lap of error, leaves the fuel term exactly at its prior, and is the source of the SOFT-start recommendations (Section 4, finding 10). Make lap-time-only the default for the weekend refit and the outlook; keep the joint fit in the retrospective as a diagnostic until it can be shown to change a decision for the better.
12. **Cut the race-day pipeline to one fit (S).** The 2025-prior, no-prior and no-ladder fits are sensitivity slides; run them only in the retrospective. With 11 and the quick sampler settings, the weekend refit drops from 3–11 min to under 30 s.
13. **Vectorise the live plan search and the counterfactual (M).** Enumerate the 2-stop grid as arrays over the cost tables already cached per tick (as `simulate_model` does offline) instead of Python loops per driver; target 0.2 s per tick and 0.5 s for the counterfactual. Persist the JAX compilation cache and import Streamlit pages lazily so cold start meets the 3 s gate.

### Process

14. **Leave-one-out recalibration (M).** Re-fit the six plan-deciding constants on all seven weekends with each weekend held out, store per-circuit values where they differ, and make the recalibration a script so the eighth weekend updates them automatically.
15. **Numeric regression tests (S).** Assert the pooled stint-rate MAE and the ladder sign on the seven scored weekends in `make test`, so the next change that breaks a curve is caught before a race.

## Appendix: reproducing

All scripts run from the project root with `.venv/bin/python`; `bench/run_chain.sh` runs them in order. Nothing they do modifies `data/processed/` or the sealed predictions.

| Script | What it measures | Output |
|---|---|---|
| `bench/bench_accuracy.py` | curves vs race, 8 variants, 7 weekends | `bench/out/accuracy.json`, `accuracy_table.csv` |
| `bench/bench_strategy.py` | decisions vs field, oracle regret, windows, life | `bench/out/strategy.json`, `strategy_table.csv` |
| `bench/bench_outlook.py` | prior-only outlook vs field | `bench/out/outlook.json` |
| `bench/bench_ablation.py` | history fold-in ablation, ladder vs races | `bench/out/ablation.json` |
| `bench/bench_live.py` | `RaceEngine` replay: latency and stop calls | `bench/out/live.json` |
| `bench/bench_speed.py` | stage timings on Barcelona | `bench/out/speed.json` |
| `bench/bench_stability.py` | seeds and partial practice | `bench/out/stability.json` |
| `bench/bench_apex.py` | apex channel on/off, re-timings | `bench/out/apex.json` |
| `bench/md2pdf.py` | this report to PDF (headless Chrome) | `results.pdf` |
