# degless benchmark, updated — after the fifteen recommendations

*Benchmarked 12 September 2026 on the same 7 dry, conventional 2026 weekends as the previous run (Australia, Japan, Barcelona, Austria, Belgium, Hungary, Italy), against the same 196 race stints. Every recommendation of the previous report (`results.md`) was implemented and the retrospective pipeline, the leave-one-out recalibration and the benchmark suite were re-run from scratch. The previous run is frozen under `bench/baseline/` and every comparison below is computed from it by `bench/bench_compare.py`. Reproduce with `make history && make benchmark`.*

## 1. Summary — previous build → updated build

| Metric | Previous | Updated | Verdict |
|---|---|---|---|
| Degradation-rate error vs the race (stint-rate MAE, shipped curves) | mean 0.109 s/lap, max 0.434; 5/7 under the 0.15 target | **mean 0.040, max 0.052; 7/7 under target** | Fixed. Australia 0.434 → 0.049, Italy 0.154 → 0.030; the pooled error now sits 0.004 above the in-sample oracle (0.036) |
| Same, practice-only posterior (no history) | 0.037 | 0.045 | The lap-time-only linear fit is as accurate as the joint one (apex benchmark: 0.045 vs 0.044); the 0.008 is the new regime factor missing at Belgium and Barcelona, which the history fold-in absorbs |
| Bayesian model vs the MixedLM × regime baseline | 0.037 vs 0.040 | **0.040 vs 0.047** | The shipped curve now beats the frequentist baseline, because the circuit history is in it |
| 90 % intervals on the stint rate | 96 % coverage at width 0.531 (previous practice-only: 96 % at 0.240) | **95 % coverage at width 0.188** | 65 % narrower than the shipped previous band, 22 % narrower than the previous practice-only band, still honest; the target was 88–93 % |
| Stop count vs the field's mode | 7/7 | **7/7** | Held |
| Compound sequence run by at least one finisher | 3/7 (mean field share 15 %) | **6/7 (mean field share 33 %)** | The plan-shape prior did this; the miss is Barcelona, where the history is on a different compound nomination |
| Start compound = the majority's | 4/7 | **6/7** | No SOFT-start recommendation survives; the miss is Barcelona |
| First-stop lap vs the field's green-flag median (3 non-safety-car weekends) | +6, +5, +4 laps (mean +5.0) | **position-aware +8, +4, +3 (mean +5.0); tyre-optimal +3, −1, +6 (mean +3.3)** | Not fixed. The undercut-exposure term moves stops by 0–1 lap; the plan-shape prior trades timing for shape (Barcelona) |
| Share of the field's first stops inside the model's window | 23 % | **30 %** | Modest |
| Tyre life vs the longest stint any finisher ran | over-stated on 20/20 compound-weekends, median 1.68× | **13/20, median 1.09×** | Bounded by the race and the circuit's history; where it is now *under*-stated it is the circuit's SOFT cap (Australia, Italy, Hungary) under-stating this year's soft |
| Oracle regret of the tool's plan (true race rates, same cost structure) | mean 12.1 s; beats the field 6/6, the winner 5/6 | **mean 10.5 s; beats the field 5/7, the winner 5/7** | Comparable; the field's own plan regrets 15.6 s on average under the same oracle |
| Counterfactual "seconds lost" figures above 30 s | 17 of 145 (13 at Italy, stops under the safety car) | **2 of 122 classified drivers (0 at Italy)** | Safety-car stops are held fixed; retirements are no longer priced to the flag |
| Prior-only outlook (no practice) matching the field's modal sequence | 5/7 | 3/7 (run by someone 6/7; stop mode 6/7) | Worse on the modal match: the plan-shape prior imports Barcelona's 2023/2025 soft-heavy plans |
| Live engine, Hungary and Barcelona replays | 1.16 / 0.98 s per lap; 48 % / 25 % of real stops inside the window; median miss 4.0 / 3.5 laps | **0.073 / 0.068 s per lap; 54 % / 34 %; median miss 3 / 2 laps; 50 % / 57 % of stops within 3 laps** | 15× faster, better calls; the cliff alarm still precedes 0 of 46 stops at Hungary |
| Apex-speed channel | shipped; +0.001 s/lap, 3× fit time, 6/7 SOFT starts | **diagnostic only**: 0.044 vs 0.045 lap-only, fuel term still at its prior (rel sd 0.25 either way), fit 11–19 s vs 7 s | Retired from production, kept as a variant |
| Stability | plan fixed after FP2; seeds move slopes < 0.002 | **plan fixed after FP2 (Barcelona) / FP2 (Hungary); seeds move slopes < 0.002 and never the plan; weekend fit 2–4 s** | Held |
| Speed | production fit 30 s (247 s contended); full retrospective 4.5–11 min; weekend refit 32–199 s; live tick ~1 s | **production fit 6.6 s; retrospective fit stage 40–60 s of compute (85–364 s wall with FastF1 loads); weekend refit 16–27 s wall (fit 4.6–6.4 s); live tick 0.07 s; app cold start 1.4 s** | Every target of the previous report met except where the disk or the network intervened |
| Gates failing across the 7 weekends | 22 | **7** (3 are the MixedLM range check, which is broken at low-degradation circuits) | |
| Test suite | 20/20 in 97 s | **27/27 in 10 s** (7 numeric regression tests added; the replay tests got the engine speed-up) | |

**Bottom line.** The two defective weekends are repaired and the rate is predicted at the noise floor everywhere, with intervals a third of the width they were. The compound choice is no longer contradicted by the races: the recommended sequence was run by the field on six weekends instead of three and the start compound matches the majority on six instead of four, and the mechanism that did it is the field's own plan-shape history plus the enforced ladder gate, not the practice data. Tyre life is stated honestly. The live engine went from one second a lap to seventy milliseconds and calls stops better. What did *not* move is the first-stop timing: the single-car undercut-exposure term changes the answer by at most a lap, and where the plan-shape prior helps the sequence it can push the first stop later still (Barcelona). That, and a plan-shape prior that does not know the compounds were renamed between seasons, are the two places the next week of work should go.

## 2. What was changed

Each recommendation of the previous report, what was built, where it lives, and what the benchmark measured. Effort as estimated before: S = hours, M = days, L = a week.

| # | Recommendation | What was done | Where | Measured effect |
|---|---|---|---|---|
| 1 | Fix the history fold-in (S) | The circuit history rescales only the *rate* (the curve's slope over ages 1–10); the hinge is never touched; the per-draw factor is capped at 3× either way; the combined rate is floored at the history's 1.5-σ lower bound | `history.apply_circuit_prior`, `apply_rate_prior_to_model` | Australia 0.434 → 0.049, Italy 0.154 → 0.030 s/lap; pooled 0.109 → 0.040 |
| 2 | Model the regime factor, do not pool it (M) | Each donor's log ratio is corrected by the thermal sensitivity times its race-minus-practice track temperature; the residuals are pooled with the **median**; the target's practice temperature is measured from the sessions that supplied the long runs and its race temperature is the archive's race-day mean (forecast sd added to the width). Session temperatures are cached | `regime.regime_prior`, `history.session_track_temp` | Neutral on accuracy: 0.040 vs 0.038 with the old pooling, 0.038 with the actual race temperature. See §3 finding 3 |
| 3 | Calibrate the intervals (S) | Race stints are scored with the racing-lap noise (0.5 s, sealed as `sigma_race`) instead of the practice `sigma_obs`; the rate interval carries the curve's own uncertainty | `validate.seal_predictions`, `score_race` | Width 0.531 → 0.188 at 95 % coverage |
| 4 | Drop the hinge from the practice fit (S) | Production fit is linear in age; the sealed file carries the circuit's longest and p90 stint as the cliff, labelled as such; the hinge survives as a diagnostic variant | `model_bayes` (`use_hinge`), `CircuitPrior.cliff` | Fit 30 → 7 s; no accuracy change |
| 5 | Recalibrate the ladder per circuit, make the gate blocking (M) | The compound intercepts of the race regression give the fresh-tyre step and the adjacent-compound degradation ratio per circuit; the fit's ladder is a **soft** prior around them (ordering can invert, and does at Austria and Belgium as the races show); after the search the pace offsets are set so the model reproduces the measured net stint-level step, draw by draw with its standard error, inside a physical band | `history.race_deg_slopes(offsets=True)`, `compounds` (soft prior, `calibrate_pace_offsets`), `strategy.search_with_pace_calibration` | Model net step now matches the measurement on 6/7 weekends (before: wrong sign on 7/7); Hungary's sequence M-H-M (0 %) → M-H-H (32 %) |
| 6 | Put track position in the objective (M) | The undercut exposure of every stop lap (cumulative one-lap undercut gain a rival would have had, level-based) enters at weight λ, scaled by field density; the pit window and the live engine carry the same term; the tyre-optimal plan (λ = 0, no prior) is always reported beside the position-aware one | `strategy.undercut_exposure_tables`, `simulate_model`, `pit_window_model`, `live.engine` | λ calibrated leave-one-out to 0.1–0.3; first stops move 0–1 lap; no net change against the field (§4 finding 4) |
| 7 | Make tyre life honest (S) | Life is bounded by the race distance and the circuit's longest stint, with `bound_by` and `longer_than_race` stated; the rate is floored at the history's lower bound; grip budgets are fitted per compound leave-one-out with a censoring-aware estimator | `TyreModel.life_table`, `scripts/80_recalibrate.py` | Over-stated 20/20 → 13/20, median 1.68× → 1.09×; the races only *bound* the budget from below (§8) |
| 8 | Safety-car awareness (S) | Stops taken under status 4/5/6/7 are flagged from the lap table; the counterfactual holds them at their actual laps at the discounted pit loss and optimises the green-flag stops; the backtest and the benchmark use green-flag first stops; retirements are not priced to the flag | `strategy.race_stops`, `counterfactual`, `validate.strategy_backtest` | > 30 s figures 17 → 2; Italy 13 → 0 |
| 9 | Field prior on plan shape (M) | The circuit's sequence and start-compound frequencies enter the objective as τ·(−log p) with a smoothed, backed-off frequency; a new circuit uses the 2026 season pooled | `history.plan_prior_for`, `strategy.plan_prior_penalty` | The single change that fixed plan shape: without it the sequence is run by anyone on 2/7 and the start matches on 2/7 (§4 ablation) |
| 10 | Per-car recommendations (L) | The fit's per-driver management term is persisted and applied; each driver's race degradation factor is measured on other 2026 races with driver × age effects, pooled and shrunk leave-one-out; per-car plans and a per-car counterfactual are produced | `TyreModel.for_driver`, `history.race_driver_factors`, `strategy.per_driver_plans` | Per-car plans share the field plan's shape on 58–100 % of cars; the factors do not improve stint ranking (Spearman +0.05 → +0.02) |
| 11 | Lap-time-only fits on race weekends (S) | Default everywhere; apex is `--apex` / `--joint` | `40_weekend.py`, `10_pipeline.py` | Fit 30 → 7 s; accuracy equal; SOFT starts gone |
| 12 | One fit on race day (S) | The weekend script runs one quick lap-only fit; the sensitivity variants live in the retrospective at the quick settings | `40_weekend.py` | Weekend refit 16–27 s wall; the supervisor's own Spain refit during this weekend's FP2: 47 s |
| 13 | Vectorise the live search and the counterfactual; JAX cache; lazy app (M) | Cost tables are built once per race on every draw; each car prices every option from its weighted-mean table in one gather and only a shortlist per draw; the counterfactual enumerates placements as arrays; the JAX compilation cache persists under `data/raw/jax_cache` | `live.engine.RaceEngine._plan`, `strategy.counterfactual`, `model_bayes._setup_jax` | Tick 1.16 → 0.073 s; counterfactual 4.7 → 1.3 s; import 0.46 s; app cold start 1.4 s on a quiet disk |
| 14 | Leave-one-out recalibration (M) | Grip budgets, dirty air, driver factors, the management trade-off, λ, τ and the grid penalty are re-derived with each weekend held out and written to `data/processed/calibration.json`; every consumer reads its weekend's set | `scripts/80_recalibrate.py`, `src/calibration.py`, `make recalibrate` | 5 minutes for all 8 sets; values in §8 |
| 15 | Numeric regression tests (S) | Pooled and per-weekend stint-rate MAE, no under-coverage, no amplified sealed curve, the enforced ladder gate, life bounded by the race, stop count run by the field | `tests/test_numeric.py` | 7 tests, in `make test` |

## 3. Accuracy: the degradation curves

![](bench/out/fig/fig1_accuracy.png)

**Per weekend, shipped curves (practice posterior + circuit history, × regime):**

| Weekend | Previous sealed | Updated sealed | Updated practice-only | Oracle | Updated bias | 90 % rate coverage prev → new | Width prev → new | Regime prev → new (self-measured) |
|---|---|---|---|---|---|---|---|---|
| Australia | 0.434 | **0.049** | 0.055 | 0.045 | +0.012 | 0.91 → 0.78 | 1.767 → 0.149 | 0.57 → 0.61 (0.75) |
| Japan | 0.024 | **0.029** | 0.018 | 0.017 | −0.029 | 1.00 → 1.00 | 0.295 → 0.199 | 0.50 → 0.63 (0.73) |
| Barcelona | 0.041 | **0.040** | 0.050 | 0.040 | +0.010 | 1.00 → 1.00 | 0.335 → 0.251 | 0.52 → 0.51 (0.70) |
| Austria | 0.032 | **0.033** | 0.036 | 0.030 | −0.017 | 1.00 → 0.97 | 0.306 → 0.199 | 0.51 → 0.69 (0.58) |
| Belgium | 0.030 | **0.052** | 0.083 | 0.048 | −0.046 | 1.00 → 1.00 | 0.191 → 0.199 | 0.58 → 0.80 (0.32) |
| Hungary | 0.046 | **0.044** | 0.045 | 0.042 | −0.004 | 0.95 → 0.91 | 0.240 → 0.199 | 0.52 → 0.83 (0.64) |
| Italy | 0.154 | **0.030** | 0.031 | 0.032 | −0.009 | 0.86 → 0.95 | 0.581 → 0.123 | 0.60 → 0.49 (0.81) |
| **Mean** | 0.109 | **0.040** | 0.045 | 0.036 | −0.012 | 0.96 → 0.95 | 0.531 → 0.188 | |

**Against the baselines, pooled over the 7 weekends** (every variant scored with the race noise; the previous build's curves re-scored from its frozen posteriors):

| Curve source | Rate MAE mean | max | Bias | 90 % coverage | Width | Spearman |
|---|---|---|---|---|---|---|
| Oracle (this race's own rates, in-sample) | 0.036 | 0.048 | +0.008 | 0.71 | 0.096 | +0.10 |
| **Updated sealed** (history fold-in fixed, temperature-modelled regime) | **0.040** | 0.052 | −0.012 | 0.95 | 0.188 | +0.05 |
| Updated sealed with the actual race temperature (a perfect forecast) | 0.038 | 0.049 | −0.015 | 0.92 | 0.178 | +0.06 |
| Updated sealed with the old pooled geometric-mean regime | 0.038 | 0.049 | −0.010 | 0.92 | 0.187 | +0.04 |
| Updated sealed, scaled per driver by the leave-one-out race factor | 0.044 | 0.061 | −0.018 | 0.91 | 0.197 | +0.02 |
| Practice posterior × regime (no history) | 0.045 | 0.083 | −0.012 | 0.86 | 0.202 | +0.02 |
| MixedLM slope × regime | 0.047 | 0.073 | −0.018 | 0.50 | 0.096 | +0.05 |
| Circuit history 2023–25 only | 0.044 | 0.062 | −0.008 | 0.58 | 0.092 | −0.01 |
| Other 2026 races' mean rate | 0.047 | 0.087 | +0.021 | 0.59 | 0.096 | −0.04 |
| Practice posterior, no regime transfer | 0.068 | 0.140 | −0.045 | 0.77 | 0.198 | −0.01 |
| Zero degradation | 0.068 | 0.119 | +0.065 | 0.34 | 0.096 | — |
| Previous sealed, re-scored with the race noise | 0.109 | 0.434 | −0.083 | 0.88 | 0.466 | −0.07 |
| Previous practice-only, re-scored with the race noise | 0.037 | 0.054 | +0.011 | 0.92 | 0.175 | −0.13 |

**Findings**

1. **The fold-in is fixed and now adds accuracy.** With the rate-only, capped and floored combination the sealed curve is within 0.004 s/lap of the practice posterior on five weekends and *better* than it on the other two (Belgium 0.083 → 0.052, Barcelona 0.050 → 0.040), which is what a history prior should do. Australia and Italy went from the worst two weekends to ordinary ones. On the saved posteriors the per-draw scale is capped on 100 % of Australia's and Italy's draws (practice shows 0.004–0.024 s/lap there) and the floor binds on both of Australia's history compounds and on Italy's MEDIUM; the result is a race-regime rate of 0.03–0.05 s/lap at both, against the 0.02–0.08 the lap-fixed-effects estimator measures on those races. At Japan, Barcelona, Austria, Belgium and Hungary the cap touches at most 2 % of draws and the floor never binds.
2. **The rate is at the noise floor.** 0.040 pooled against an in-sample oracle of 0.036 and a 0.041–0.052 range where the previous run had 0.024–0.434. Every weekend passes the 0.15 gate; the largest error is Belgium (0.052), see finding 3.
3. **The temperature-modelled regime factor did not earn its keep.** Pooled accuracy is 0.040 with it, 0.038 with the old geometric mean and 0.038 with the *actual* race temperature. Against the self-measured ratio the mean absolute log error is 0.36 (modelled, archive forecast), 0.24 (median pooling, no temperature term) and 0.28 (perfect forecast): the between-weekend variation in the regime ratio (0.32 at Belgium to 0.81 at Italy) is not thermal. What the model does do is widen honestly (ln-sd 0.21–0.44 including the forecast spread) and it helps where the forecast is right (Belgium with the actual 31 °C: 0.052 → 0.031). The archive's race-day mean is the weak link (Belgium's single 2024 race at 43 °C forecast a race run at 31 °C). Recommendation 3 in §10.
4. **The intervals are calibrated.** 95 % coverage on the stint rate at a width of 0.188 s/lap, against 0.096 for a point estimate carrying only the scorer's noise; the previous shipped band was 0.531 at 96 %, and re-scored with the same race noise it would have covered 88 % at 0.466. Australia (78 %) and Hungary (91 %) are the two weekends inside the target band; the rest over-cover.
5. **Stint ranking is still not predicted, and per-car factors do not help.** Spearman between predicted and observed stint rates is +0.05 for the sealed curve and +0.02 with each driver's leave-one-out race factor applied (+0.10 for the oracle). The factors are real on the races they are measured on (0.91–1.70 after shrinkage, Bottas hardest on his tyres, Lindblad, Antonelli and Hülkenberg gentlest) but they do not transfer stint to stint.
6. **The cliff is now reported, not fitted.** The sealed file carries the circuit's longest and p90 stint per compound. Against the race collapse laps the scorer can detect (10 cases), the p90 stint is 2–22 laps late (mean 10); the previous posterior knee was 7 laps off on 11 cases. Neither is a cliff model; §10 item 5.
7. **The frequentist baseline is no longer a match.** MixedLM × regime scores 0.047 to the sealed 0.040, and the MixedLM range gate fails at Australia (−0.11 s/lap), Japan (0.116) and Italy (0.021): the baseline itself is broken at low-degradation circuits.

## 4. Intelligence: the decisions

![](bench/out/fig/fig2_firststop.png)

| Weekend | Previous plan | Updated plan | Tyre-optimal (no position term, no prior) | Field modal (share prev → new) | Winner | Start = majority prev → new | First stop − field median: prev / tyre-optimal / updated | Oracle regret prev / updated / field / winner (s) |
|---|---|---|---|---|---|---|---|---|
| Australia | 1-stop H-M @ 36 | **1-stop M-H @ 23** | 1-stop H-M @ 34 | M-H (0 % → 53 %) | M-H @ 12 | no → yes | +11 / +9 / −2 (SC set the stops) | 11.2 / 9.5 / 20.5 / 25.8 |
| Japan | 1-stop S-H @ 21 | **1-stop M-H @ 23** | 1-stop M-S @ 31 | M-H (0 % → 85 %) | M-H @ 22 | no → yes | +3 / +13 / +5 (SC) | n/a / 0.5 / 0.4 / 0.4 |
| Barcelona | 2-stop M-H-H @ 19,43 | **2-stop S-M-S @ 21,43** | 3-stop M-M-S-S @ 16,32,49 | M-H-H (46 % → 0 %) | S-H-M-H @ 11,27,41 | yes → no | +6 / +3 / +8 | 5.3 / 5.5 / 10.6 / 8.1 |
| Austria | 2-stop M-H-H @ 23,47 | **2-stop M-H-M @ 22,46** | 3-stop M-S-S-M @ 17,35,53 | M-H-H (29 % → 24 % on M-H-M) | M-H-H @ 19,43 | yes → yes | +5 / −1 / +4 | 22.4 / 13.4 / 25.8 / 25.8 |
| Belgium | 1-stop S-M @ 24 | **1-stop M-S @ 20** | 1-stop S-M @ 24 | M-H (0 % → 5 % on M-S) | M-S @ 18 | no → yes | +8 / +8 / +4 (SC) | 13.9 / 16.0 / 10.6 / 14.4 |
| Hungary | 2-stop M-H-M @ 23,47 | **2-stop M-H-H @ 22,46** | 2-stop M-M-S @ 25,51 | M-H-H (0 % → 32 %) | M-H-H-S @ 17,39,56 | yes → yes | +4 / +6 / +3 | 16.8 / 26.3 / 28.8 / 27.4 |
| Italy | 1-stop M-H @ 23 | **1-stop M-H @ 25** | 1-stop H-M @ 28 | M-H (32 % → 32 %) | H-M-M @ 3,28 | yes → yes | all stops under the SC | 2.7 / 2.1 / 12.8 / 30.7 |

**Ablation — the same search on the shipped posterior with one term switched off:**

| Variant | Sequence run by anyone | Start = majority | Stops = mode | Mean field share on the plan | First stop − field (Barcelona, Austria, Hungary) |
|---|---|---|---|---|---|
| **Full objective** | 6/7 | 6/7 | 7/7 | 0.33 | +8, +4, +3 |
| No position term (λ = 0) | 6/7 | 6/7 | 7/7 | 0.33 | +8, +5, +3 |
| No plan-shape prior (τ = 0) | 2/7 | 2/7 | 5/7 | 0.12 | +3, 0, 0 |
| No pace calibration (ladder gate not enforced) | 6/7 | 6/7 | 7/7 | 0.29 | +5, +3, +3 |
| Previous build's constants, no new terms | 4/7 | 2/7 | 7/7 | 0.11 | +5, +3, 0 |
| Practice posterior, no history | 5/7 | 6/7 | 5/7 | 0.32 | +3, +4, +2 |

![](bench/out/fig/fig3_life.png)

**Findings**

1. **Stop count holds at 7/7**, and on average 70 % of classified finishers ran the recommended count (Japan 90 %, Belgium 95 %).
2. **The compound choice is fixed on six weekends, by the field's own history.** The ablation is unambiguous: with the plan-shape prior off, the search drifts back to sequences nobody ran (2/7) and SOFT starts (2/7 match); with it on, 6/7 and 6/7. The enforced ladder gate is what moved Hungary from M-H-M to the modal M-H-H, and no weekend any longer prices a harder compound as faster over a stint where the races say it is slower (model net +0.05 to +0.09 s/lap per step against measured +0.04 to +0.07; Australia is the exception, where the pooled measurement is itself negative and the calibration is clipped).
3. **Barcelona is the miss, and it is a data-labelling problem.** The circuit's 2023 and 2025 races were run on a softer nomination; their S-M-S and S-H-M plans are read letter for letter, so the prior favours a SOFT start (31 of 37 historical finishers) where the 2026 field went M-H-H. The tyre-optimal plan there is a 3-stop with earlier stops; the position-aware plan is a 2-stop S-M-S with a 21-lap first stint, 8 laps after the field. The prior-only outlook has the same problem (its Barcelona pick fell from M-H-H, 46 % of the field, to S-M-S, 0 %). §10 item 2.
4. **First stops are still late, and the position term is not the lever.** The undercut-exposure term is flat until λ ≈ 0.3 (mean absolute first-stop error 3.3 → 2.7 laps in the calibration sweep) and moves the shipped first stop by 0–1 lap. The tyre-optimal plan is closer to the field's green-flag median (+3.3 laps mean) than the position-aware one (+5.0), because the plan-shape prior prefers two-stop families with longer opening stints. The field's 13-lap first stops at Barcelona are not explained by a single car's undercut arithmetic at these degradation rates; they are a positional race between many cars. §10 item 1.
5. **Tyre life is honest.** Predicted life at the plan's push exceeds the longest stint any finisher ran on 13 of 20 compound-weekends (median ratio 1.09, was 1.68); on 6 it is bounded by the circuit's history, on 2 by the tyre, on none by the race distance any more. The remaining over-statements are 1.4–1.8× on the MEDIUM at Australia, Austria, Belgium and Hungary and the SOFT at Barcelona; the under-statements (0.42–0.56) are the circuit's SOFT cap of 10–19 laps against a 2026 soft the field ran for 24–34.
6. **Under true degradation the tool's plan is still good, and no better than before.** Mean regret 10.5 s (previous 12.1 on the six comparable weekends), beating the field's consensus 5/7 and the winner 5/7. Hungary's 26 s is the M-H-H plan priced with the calibrated HARD offset; the field's own M-H-H regrets 29 s under the same oracle, so the comparison stands, but the oracle inherits the model's pace offsets and is not an independent judge of them.
7. **The counterfactual is usable.** Two figures above 30 s remain (both Australia: a damaged car on three SOFT stints, and a driver whose plan the model cannot price inside the allocation); Italy's thirteen safety-car "losses" are gone because those stops are held where they were. 122 classified drivers are scored; retirements are flagged and not priced to the flag.
8. **Per-car plans exist and are informative about the field, not about the car.** Each driver's own plan shares the field plan's shape on 58 % (Hungary) to 100 % (Japan, Italy) of cars; per-car first stops spread 2–10 laps within a weekend; their rank against what the drivers actually did is uncorrelated (Spearman −0.75 to +0.28). The leave-one-out race factors span 0.85–1.72 but, as the accuracy section shows, do not transfer.
9. **The prior-only outlook got worse on the modal match (5/7 → 3/7) for the Barcelona reason above**, and is otherwise unchanged: Japan M-H (85 %), Hungary M-H-H, Italy M-H, Austria M-H-M (24 %), Belgium M-S (5 %, the winner's plan), Australia M-H-H (7 %, a two-stop where the field one-stopped). It builds in 1–4 s.

## 5. Live engine

Archived Hungary and Barcelona race feeds replayed through `RaceEngine`, one tick per lap, the sealed models from those weekends' practice, the calibrated constants and the position term.

| | Hungary (70 laps, 46 real stops) prev → new | Barcelona (66 laps, 53 real stops) prev → new |
|---|---|---|
| Tick latency, whole field: mean / p95 / max | 1156 / 1375 / 1428 ms → **73 / 81 / 140 ms** | 985 / 1320 / 1390 ms → **68 / 74 / 118 ms** |
| Options priced per car per lap | 277 → 601 | 731 → 665 |
| Real stops inside the engine's window 3 laps earlier | 48 % → **54 %** | 25 % → **34 %** |
| Median error, recommended in-lap vs actual | 4.0 → **3.0** laps | 3.5 → **2.0** laps |
| Stops within 3 laps of the recommendation | 33 % → **50 %** | 47 % → **57 %** |
| "Box now" cost the lap before the real stop (median) | +1.8 → **+0.7 s** | +3.2 → +2.3 s |
| Cliff alarm raised before the stop | 0 of 46 → 0 of 46 | 9 of 53 → 2 of 53 |
| Regime multiplier: prior → last lap (offline truth) | 0.53 → 0.67 (0.64) becomes 0.87 → 0.78 (0.64) | 0.63 → 0.63 (0.70) becomes 0.55 → 0.57 (0.70) |

![](bench/out/fig/fig4_live.png)

**Findings**

1. **Fifteen times faster, with headroom.** The whole cost of a tick was one thing: the fresh-stint cost table was rebuilt for every car on that car's resampled draws (17 times a lap). It is now built once per race and each car reads its expectation of it; the per-draw work is a shortlist of 48 options. 73 ms for twenty cars leaves the target of 200 ms a lap three times over.
2. **The calls improved with the objective.** Half of Hungary's stops and 57 % of Barcelona's now fall within 3 laps of the recommendation, and the engine says "boxing now costs 0.7 s" the lap before Hungary's teams box (was 1.8 s).
3. **The cliff still does not predict stops.** 0 of 46 at Hungary, where teams stop with the wear estimate at 39 % of the budget. The window and box-now logic is the useful signal; the wear-based alarm should be retired (§10 item 9).
4. **The regime multiplier at Hungary starts high** (0.87 from the temperature-modelled prior against 0.64 measured after the race) and walks down to 0.78 by the flag; Barcelona's starts at 0.55 and ends at 0.57 against 0.70. The live update corrects the prior's direction but not fully within a race: the same finding as §3, seen from the pit wall.

## 6. Speed

Barcelona 2026, in-process timers on a quiet disk (the previous run's contended figures in brackets where they differ).

| Stage | Previous (s) | Updated (s) |
|---|---|---|
| Import JAX + NumPyro | 0.6 (48) | 0.46 |
| Load practice laps from the FastF1 cache | 49 | 0.78 |
| **Production fit** (was NUTS 4 × 1500 joint; now 4 × 1500 lap-time only, linear, soft ladder) | **30 (247)** | **6.6** |
| Weekend refit fit (2 × 800 lap-time only) | 12–29 (joint) | 3.6 |
| Diagnostic: hinge variant (2 × 800) | — | 4.3 |
| Diagnostic: joint lap-time + apex (4 × 1500) | 30 | 12.6 |
| Strategy search, 620k plans, 500 draws, 1-lap grid, full objective | 0.64 | 0.68 |
| Search with the ladder gate enforced (2–3 searches) | — | 1.3 |
| Pit-window sweep with the position term | 0.06 | 0.06 |
| Counterfactual, all classified drivers, safety-car aware, per car | 4.7 | 1.3 |
| Per-car plans, 20 drivers, 2-lap grid | — | 7.1 |
| Outlook build, 300 draws, with scenarios | 0.5–3.3 | 3.4 |
| Live engine tick, whole field | ~1.0 | 0.07 |
| App cold start (Streamlit `AppTest`, quiet disk) | 1.3 (420) | 1.4 (266 during the benchmark run) |
| Full retrospective pipeline, per weekend, recorded | 270–667 | 85–364 wall; fit stage 40–60 of compute, decide stage 11–42 |
| Weekend refit script, recorded | 32–37 quick, 199 full | 16 (Hungary), 21 (Barcelona), 27 (Italy); 47 for the supervisor's live Spain FP1+FP2 refit |
| Leave-one-out recalibration, 7 weekends × 8 sets | — | 268 |
| Peak memory | 1.3 GB | 2.1 GB |

**Findings**

1. **Every slow stage of the previous report is gone.** The production fit is 6.6 s, the weekend script is under 30 s, the live tick is 70 ms, and the whole benchmark suite (accuracy to comparison, tests included) runs in 25 minutes.
2. **What remains slow is not the model.** The retrospective's wall time is dominated by FastF1 loads whose Ergast-mirror requests time out (Australia's practice load took 305 s of a 364 s pipeline; Barcelona's race load 103 s) and by the first JAX compilation of a new data shape on a contended disk (463 s once in the stability run, then 3–4 s). A `--offline` flag now keeps the cached weekends off the network; the compilation cache persists between processes.
3. **The per-car search and the diagnostics are the new cost**, 7 s and 40 s respectively, and both are optional.
4. **Memory rose to 2.1 GB** because the live tables are kept for every draw; the app and the daemon do not share a process, so this is the benchmark's figure, not the pit wall's.

## 7. Stability

Weekend-setting fits (2 chains × 800 + 800, lap-time only, soft ladder, circuit history folded in as the weekend script does), 300-draw search with the full objective, scored against the race.

| Fit | Clean laps | Slopes S / M / H (s/lap, practice) | Best plan | Rate MAE | Fit (s) |
|---|---|---|---|---|---|
| Barcelona, FP1 only | 83 | 0.222 / 0.140 / — | 2-stop S-M-S @ 20,44 | 0.082 | 463 (first compile, contended disk) |
| Barcelona, FP1+FP2 | 222 | 0.240 / 0.212 / 0.159 | 2-stop S-M-S @ 21,43 | 0.043 | 4.4 |
| Barcelona, FP1–FP3 | 225 | 0.244 / 0.212 / 0.159 | 2-stop S-M-S @ 21,43 | 0.043 | 3.9 |
| Barcelona, FP1–FP3, seeds 1 and 2 | 225 | ±0.001 | identical | 0.042 | 2.9 / 2.8 |
| Hungary, FP2 only | 155 | 0.153 / 0.114 / 0.094 | 2-stop M-H-H @ 21,45 | 0.049 | 3.7 |
| Hungary, FP2+FP3 | 184 | 0.129 / 0.108 / 0.091 | 2-stop M-H-H @ 22,45 | 0.045 | 3.8 |
| Hungary, FP2+FP3, seeds 1 and 2 | 184 | ±0.002 | identical | 0.045 | 2.0 / 2.5 |

Seeds do not matter, the answer is set after the second session, and a quick fit scores the same as the production fit (0.043 vs 0.040 at Barcelona, 0.045 vs 0.044 at Hungary). FP1 alone is still not enough (83 laps, no HARD run, 0.082 s/lap).

## 8. Calibration, leave one out

![](bench/out/fig/fig5_calibration.png)

| Held out | Grip budget S / M / H (s) | Manage cost (s), wear floor | Grid penalty (s) | Dirty air (s/lap) | λ | τ (s/nat) |
|---|---|---|---|---|---|---|
| Australia | 3.80 / 3.83 / 3.84 | 0.90, 0.45 | 0 | 0.067 | 0.30 | 2.5 |
| Japan | 3.80 / 3.83 / 3.84 | 0.90, 0.35 | 0 | 0.137 | 0.20 | 6.0 |
| Barcelona | 3.80 / 3.80 / 3.84 | 0.90, 0.35 | 0 | 0.067 | 0.10 | 4.0 |
| Austria | 3.80 / 3.83 / 3.84 | 0.60, 0.55 | 0 | 0.084 | 0.15 | 4.0 |
| Belgium | 3.80 / 3.83 / 3.84 | 0.60, 0.55 | 0 | 0.137 | 0.15 | 4.0 |
| Hungary | 3.80 / 3.83 / 3.27 | 0.60, 0.55 | 0 | 0.067 | 0.15 | 2.5 |
| Italy | 3.80 / 3.83 / 3.84 | 0.60, 0.45 | 0 | 0.137 | 0.30 | 4.0 |
| **Global (a new weekend)** | 3.80 / 3.83 / 3.84 | 0.60, 0.45 | 0 | 0.120 | 0.30 | 4.0 |

**Findings**

1. **The races only bound the grip budget from below.** Race rate × longest stint is 0.5–2.0 s at every low-degradation weekend because the longest stint ends for strategic reasons long before the cliff (Australia's MEDIUM: 0.025 s/lap × 28 laps = 0.7 s); a first version of the estimator took the upper quartile of those, got 2.1 s for the MEDIUM, and would have given a 14-lap MEDIUM life at Barcelona where the field ran 26. The censoring-aware estimate is the largest lower bound any weekend showed: 3.83 s (MEDIUM, Barcelona) and 3.84 s (HARD, Hungary), i.e. the Barcelona value of the previous build, and the SOFT is never run near its cliff (largest bound 2.1 s), so its prior stays. The previous report's "implied 1.9–3.8 s" was mostly the inflated Australia and Italy rates.
2. **The management trade-off is looser than assumed**: management costs 0.6 s/lap at full effect, not 0.9, with the same wear floor of 0.45; the optimiser's implied practice-to-race factor is 0.59–0.71 against measured 0.32–0.81.
3. **Dirty air is smaller and circuit-specific**: +0.43 s/lap at Hungary and +0.25 at Australia, but −0.09 at Suzuka and −0.20 at Monza, where a car within 3 s is in a tow. The pooled 0.12 replaces the 0.45 constant, and the sign change is a real effect the objective should carry per circuit.
4. **The grid-start penalty is redundant** once the plan-shape prior carries the start-compound frequency: every sweep chose 0.
5. **λ and τ are weakly identified.** The λ objective is flat from 0 to 0.2 and drops one step at 0.3 (the sweep's edge is 0.45); τ saturates at 4. Both are set by three and seven weekends respectively, and the leave-one-out spread (λ 0.1–0.3, τ 2.5–6) is the honest statement of that.
6. **23 drivers have race factors**, 0.91 (Lindblad, Antonelli, Hülkenberg) to 1.70 (Bottas) after shrinkage toward 1 with a 0.15 log-sd prior.

## 9. Engineering observations

* **Gates.** 7 failures over 7 weekends (was 22): the MixedLM range check at Australia, Japan and Italy (the baseline is broken at low-degradation circuits, not the model); Australia's ladder gate (the pooled measured net is negative and the calibration is clipped to the physical band); Barcelona's stint-length range (21- and 23-lap SOFT stints against an observed maximum of 21); Belgium's 103 clean laps; one divergence in Italy's 6,000 draws.
* **Sealing and the firewall are unchanged**; the sealed file additionally carries the race noise, the circuit's cliff and the compound prior's label.
* **Tests.** 27 pass in 10 s: the 20 previous ones plus 7 numeric regression tests that would have caught both defects the previous report found.
* **It ran on a live weekend.** The Spanish Grand Prix was in progress during this benchmark; the user's supervisor refit Madring on FP1+FP2 with the new weekend script in 47 s (1-stop M-H @ 33, sealed) and rebuilt the outlook from it while the benchmark was running.
* **Constants.** The six hand-calibrated numbers of the previous report now come from `data/processed/calibration.json`, per weekend, with the config values as the fallback for a season with no scored race.

## 10. Where further improvements can be made

Ranked by expected effect on what an analyst sees. Effort: S = hours, M = days, L = a week or more.

1. **First-stop timing (M).** The single-car undercut exposure is the wrong lever: its objective is flat below λ = 0.3 and it moves the first stop by a lap. The circuit's own first-stop distribution is already computed (`CircuitPrior.first_stop`: median 8.5–23 laps, scaled) and should enter as a direct prior on the stop lap the way the plan-shape prior enters on the family; beyond that, a rival-relative model that prices losing the place to the specific car behind (the live engine's duel arithmetic, applied to the grid) is what the field is actually doing. Target: mean first-stop error ≤ 2 laps on the non-safety-car weekends, from 5.
2. **Plan-shape prior through compound nominations (M).** Read the historical sequences through Pirelli's C-numbers rather than S/M/H letters, so Barcelona's 2023/2025 soft-heavy plans map onto 2026's nomination; keep the letter prior only where the nomination is unchanged. Expected: Barcelona's sequence share 0 → 46 %, the prior-only outlook's modal match back to 5/7 or better.
3. **Regime factor: forecast in, archive out (S).** Make the temperature term act only when a race-day forecast is supplied (`40_weekend.py --race-temp`), pooling the median residual otherwise (mean absolute log error vs self-measured 0.24 against 0.36 now), and add a circuit-level regime prior from the circuit's own practice-versus-race history so Belgium's 0.32 is not a surprise. Belgium's 0.052 s/lap (bias −0.046) is this.
4. **Identify the grip budget from race stints (M).** The races bound it from below; the scorer already detects pace collapses at 16–19 laps where the history's p90 stint says 21–29. A within-stint cliff detector on the race lap tables would fit the budget per compound instead of taking the Barcelona value everywhere.
5. **Per-car intelligence from within the weekend (M).** Leave-one-out race factors do not transfer (Spearman +0.05 → +0.02). The information that does exist is the practice fit's per-driver term and the live engine's per-car posterior; pool by team rather than driver and update the per-car plan from the live posterior instead of the pre-race factor.
6. **The compound ladder at Melbourne (S).** The pooled net step mixes 2023–24 Melbourne (harder faster by 0.5–0.7 s/lap, a phase-bias artefact) with 2026 donors (+0.03 to +0.11); weight recent seasons and drop pre-2026 measurements with a standard error above 0.15. Australia's ladder gate would then pass.
7. **Replace the MixedLM gate (S)** with a stint-fixed-effects estimator with the evolution correction, which is the quantity the posterior reports; the current baseline fails on three weekends for reasons unrelated to the model.
8. **Dirty air per circuit (S).** The estimate changes sign between Hungary (+0.43) and Monza (−0.20); the calibration should carry it per circuit from the circuit's own races, as it does the ladder.
9. **Retire the wear-based cliff alarm in the live engine (S)** in favour of the window and box-now signals, which now catch half the stops within 3 laps while the alarm precedes none.
10. **Operations (S).** Run the retrospective with `--offline` once the weekends are cached (5 minutes of Ergast timeouts at Australia), warm the JAX compilation cache after `make cache`, and re-time anything quoted here on a quiet disk before quoting it again; the app's 266 s cold start in the benchmark run was 1.4 s an hour later.
11. **Italy's single divergence (S)**: raise `target_accept` or reparameterise the per-driver deviation scale.
12. **Widen the calibration grids (S)**: λ's sweep ended at its edge; τ saturated. Cheap to extend now that each set costs 35 s.

## Appendix: reproducing

All scripts run from the project root with `.venv/bin/python`; `make history` runs the fit stage on every scored weekend, the leave-one-out recalibration and the decide stage; `make benchmark` runs `bench/run_all.sh`. Nothing in `bench/` modifies `data/processed/` or the sealed predictions; the previous run's artifacts are frozen under `bench/baseline/`.

| Script | What it measures | Output |
|---|---|---|
| `scripts/10_pipeline.py --stage fit / decide` | the retrospective, split so the calibration can sit between the fits and the decisions | `data/processed/fitstage_*.json`, `meta_*.json` |
| `scripts/80_recalibrate.py` | leave-one-out calibration of the plan-deciding constants | `data/processed/calibration.json` |
| `bench/bench_accuracy.py` | curves vs race, 11 variants + the frozen baseline re-scored, 7 weekends | `bench/out/accuracy.json`, `accuracy_table.csv`, `accuracy_baseline_table.csv` |
| `bench/bench_strategy.py` | decisions vs field, tyre-optimal vs position-aware, oracle regret, per-car plans, safety-car-aware windows and counterfactual | `bench/out/strategy.json`, `strategy_table.csv` |
| `bench/bench_ablation.py` | each new term switched off; ladder vs races | `bench/out/ablation.json` |
| `bench/bench_outlook.py` | prior-only outlook vs field, vs the previous prior | `bench/out/outlook.json` |
| `bench/bench_live.py` | `RaceEngine` replay: latency and stop calls | `bench/out/live.json` |
| `bench/bench_speed.py` | stage timings on Barcelona | `bench/out/speed.json` |
| `bench/bench_stability.py` | seeds and partial practice | `bench/out/stability.json` |
| `bench/bench_apex.py` | the joint fit as a diagnostic against the shipped lap-time fit | `bench/out/apex.json` |
| `bench/bench_compare.py` | every before/after table and figure in this report | `bench/out/compare.json`, `compare.md`, `fig/` |
| `bench/md2pdf.py` | this report to PDF (headless Chrome) | `results_updated.pdf` |
