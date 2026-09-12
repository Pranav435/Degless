### V1 → V2 → V3, every headline metric (7 weekends: AUS, JPN, BCN, AUT, BEL, HUN, ITA)

| Metric | V1 | V2 | V3 | Δ V3−V2 | Verdict |
|---|---|---|---|---|---|
| Degradation-rate error vs the race (stint-rate MAE, s/lap, mean over weekends) | 0.109 | 0.040 | **0.035** | -0.004 | improved |
| Worst weekend's stint-rate MAE (s/lap) | 0.434 | 0.052 | **0.049** | -0.003 | improved |
| Stint-rate bias (s/lap, mean over weekends; 0 is best) | -0.083 | -0.011 | **-0.004** | +0.008 | improved |
| 90% interval coverage on the stint rate | 0.879 | 0.945 | **0.912** | -0.033 | not meaningfully changed |
| 95% interval coverage on the stint rate | 0.944 | 0.959 | **0.940** | -0.019 | not meaningfully changed |
| 90% interval width on the stint rate (s/lap) | 0.466 | 0.188 | **0.164** | -0.024 | improved |
| Practice posterior x regime, no circuit history (stint-rate MAE) | 0.037 | 0.045 | **0.038** | -0.007 | improved |
| MixedLM slope x regime (the frequentist baseline) | – | – | **0.043** | – | unavailable |
| Compound sequence run by at least one finisher | 3 | 6 | **6** | 0 | unchanged |
| Start compound = the majority's | 4 | 6 | **7** | 1 | improved |
| Stop count = the field's mode | 7 | 7 | **7** | 0 | unchanged |
| Mean share of the field that ran the recommended sequence | 0.153 | 0.329 | **0.350** | +0.021 | improved |
| Mean |first stop − field green-flag median| (laps, non-SC weekends) | 5.000 | 5.000 | **4.333** | -0.667 | improved |
| Mean signed first stop − field median (laps, non-SC weekends) | 5.000 | 5.000 | **4.333** | -0.667 | improved |
| Same, the tyre-optimal plan (no prior, no position term) | – | 3.333 | **3.000** | -0.333 | improved |
| Same, V3 with the first-stop prior switched off (kappa = 0) | – | 5.000 | **4.333** | -0.667 | improved |
| Share of the field's first stops inside the model's window | 0.230 | 0.303 | **0.361** | +0.058 | improved |
| Oracle regret of the tool's plan (s, true race rates, same cost structure) | 13.165 | 11.781 | **9.399** | -2.383 | improved |
| Oracle regret of the field's modal plan (s) | 13.8 | 15.6 | **17.5** | +1.8 | regressed |
| Oracle regret of the winner's plan (s) | 16.5 | 18.9 | **20.0** | +1.0 | regressed |
| Weekends where the tool's plan beats the field's modal plan on the oracle | 5 | 5 | **5** | 0 | unchanged |
| Compound-weekends where predicted life exceeds the longest stint run (of 20) | 20 | 13 | **12** | -1 | improved |
| Compound-weekends where it is shorter than the longest stint run (of 20) | 0 | 7 | **8** | 1 | regressed |
| Median predicted life ÷ longest stint run (1.0 is honest) | 1.679 | 1.094 | **1.084** | -0.010 | not meaningfully changed |
| Share of cars whose own plan shares the field plan's shape | – | 0.795 | **0.862** | +0.067 | improved |
| Per-car first-stop spread within a weekend (laps) | – | 8.857 | **5.714** | -3.143 | regressed |
| Spearman, predicted vs observed stint rate — per-car scale: LOO race factors (V2's per-car term) | – | 0.025 | **0.049** | +0.024 | improved |
| Spearman, predicted vs observed stint rate — per-car scale: this weekend's own dev[d,c] | – | 0.025 | **-0.020** | -0.045 | regressed |
| Spearman, predicted vs observed stint rate — per-car scale: team-pooled dev (V3 ships this) | – | 0.025 | **-0.031** | -0.056 | regressed |
| Spearman, predicted vs observed stint rate (field model) | -0.073 | 0.048 | **0.042** | -0.006 | regressed |
| Predicted collapse lap (budget ÷ rate) vs the detected knee (laps) | – | – | **23.381** | – | unavailable |
| Stints the detector calls a pace collapse | – | – | **4** | – | unavailable |
| |Bayes pooled slope − stint-FE baseline| (s/lap; the V3 gate, ≤ 0.06) | – | – | **0.017** | – | unavailable |
| Counterfactual 'seconds lost' figures above 30 s | 17 | 2 | **1** | -1 | improved |
| Live engine tick, whole field (ms, mean over the replays) | 1070.2 | 70.8 | **74.8** | +4.0 | regressed |
| Live engine tick, p95 (ms) | 1347.5 | 77.2 | **81.8** | +4.7 | regressed |
| Live engine tick, worst lap (ms) | 1408.6 | 128.6 | **134.4** | +5.8 | not meaningfully changed |
| Share of real stops inside the engine's window 3 laps earlier | 0.362 | 0.442 | **0.421** | -0.020 | not meaningfully changed |
| Share of real stops within 3 laps of the recommendation | 0.399 | 0.533 | **0.472** | -0.061 | regressed |
| Median |recommended in-lap − actual| (laps) | 3.750 | 2.500 | **3.500** | +1.000 | regressed |
| 'Box now' cost the lap before the real stop (s, median) | 2.516 | 1.478 | **1.755** | +0.277 | regressed |
| Live signal precision: window (a stop within 3 laps = true positive) | – | – | **0.180** | – | unavailable |
| Live signal recall: window | – | – | **0.516** | – | unavailable |
| Live signal precision: box_now (a stop within 3 laps = true positive) | – | – | **0.193** | – | unavailable |
| Live signal recall: box_now | – | – | **0.496** | – | unavailable |
| Live signal precision: collapse (a stop within 3 laps = true positive) | – | – | **0.171** | – | unavailable |
| Live signal recall: collapse | – | 0.019 | **0.050** | +0.031 | improved |
| Prior-only outlook matching the field's modal sequence | 0 | 0 | **3** | 3 | improved |
| Same, with the plan prior read letter-for-letter (no C-number mapping) | – | 3 | **3** | 0 | unchanged |
| Prior-only outlook: mean share of the field on its plan | 0.329 | 0.262 | **0.276** | +0.013 | improved |
| Production posterior fit (s) | 247.372 | 6.648 | **6.156** | -0.492 | improved |
| Weekend refit fit, 2x800x800 (s) | 21.297 | 3.580 | **2.932** | -0.648 | improved |
| Strategy search, 500 draws, 1-lap grid, full objective (s) | 0.636 | 0.680 | **0.652** | -0.028 | not meaningfully changed |
| Peak resident memory of the speed benchmark (MB) | 1310.0 | 2090.0 | **1981.0** | -109.0 | improved |
| Whole benchmark suite, wall clock (s) | – | 1500.0 | **371.0** | -1129.0 | improved |
| Tests passing | 20 | 27 | **38** | 11 | improved |
| Tests failing | 0 | 0 | **0** | 0 | unchanged |
| Tests skipped | 0 | 0 | **0** | 0 | unchanged |
| Gates failing across the benchmarked weekends | 22 | 7 | **3** | -4 | improved |

*verdict per metric: 'unchanged' if V3 equals V2 exactly; 'not meaningfully changed' if |V3 - V2| < min(0.05 * |V2|, one weekend's worth), where one weekend's worth is 1 for a count over the weekends benchmarked, |V2| / n for a mean over n weekends, and undefined (5% alone) for a single figure; otherwise 'improved' or 'regressed' by direction. Lower is better for MAE, absolute bias, interval width, first-stop error, oracle regret, stop-call error, tick latency, runtime, memory and failing gates; higher is better for match counts, field shares, coverage-in-band, precision and recall. A coverage metric is judged by its distance to its target band (90%: 0.88-0.93, 95%: 0.93-0.97), not by being larger. 'unavailable' means one of the three builds does not report the metric.*

### Accuracy per weekend (stint-rate MAE, s/lap; every build re-scored by this run's code)

| Weekend | V1 | V2 | V3 | V3 practice-only | Oracle | V3 bias | 90% coverage V1→V2→V3 | Width V1→V2→V3 | Regime V2→V3 (self) |
|---|---|---|---|---|---|---|---|---|---|
| Australia | 0.434 | 0.049 | **0.049** | 0.055 | 0.045 | +0.011 | 0.70 → 0.78 → 0.78 | 1.676 → 0.149 → 0.146 | 0.61 → 0.67 (0.75) |
| Japan | 0.024 | 0.029 | **0.020** | 0.018 | 0.017 | -0.017 | 1.00 → 1.00 → 1.00 | 0.184 → 0.199 → 0.154 | 0.63 → 0.63 (0.73) |
| Barcelona | 0.041 | 0.040 | **0.040** | 0.051 | 0.040 | +0.001 | 1.00 → 1.00 → 0.94 | 0.247 → 0.252 → 0.213 | 0.51 → 0.60 (0.70) |
| Austria | 0.032 | 0.033 | **0.032** | 0.033 | 0.030 | -0.013 | 1.00 → 0.97 → 0.95 | 0.226 → 0.199 → 0.176 | 0.69 → 0.65 (0.58) |
| Belgium | 0.030 | 0.052 | **0.029** | 0.033 | 0.048 | -0.011 | 1.00 → 1.00 → 1.00 | 0.185 → 0.199 → 0.162 | 0.80 → 0.47 (0.32) |
| Hungary | 0.046 | 0.044 | **0.046** | 0.047 | 0.042 | +0.019 | 0.84 → 0.91 → 0.81 | 0.193 → 0.196 → 0.159 | 0.83 → 0.53 (0.64) |
| Italy | 0.154 | 0.030 | **0.032** | 0.030 | 0.032 | -0.016 | 0.62 → 0.95 → 0.90 | 0.549 → 0.124 → 0.140 | 0.49 → 0.65 (0.81) |
| **Mean** | 0.109 | 0.040 | **0.035** | 0.038 | 0.036 | -0.004 | 0.88 → 0.95 → 0.91 | 0.466 → 0.188 → 0.164 | |

### Pooled over the benchmarked weekends, every curve variant (V3 run)

| Curve source | Rate MAE mean | max | Bias | 90% cov | 95% cov | Width | Spearman |
|---|---|---|---|---|---|---|---|
| Oracle (this race's own rates, in-sample) | 0.036 | 0.048 | +0.008 | 0.71 | 0.76 | 0.096 | +0.10 |
| **V3 sealed** (shipped) | 0.035 | 0.049 | -0.004 | 0.91 | 0.94 | 0.164 | +0.04 |
| V3 regime: donor median + this circuit's history (re-folded) | 0.035 | 0.049 | -0.004 | 0.91 | 0.94 | 0.164 | +0.04 |
| V3 regime: donor median only (no circuit history) | 0.039 | 0.049 | -0.012 | 0.92 | 0.95 | 0.179 | +0.05 |
| V2 regime: archive race-day temperature as the forecast | 0.040 | 0.052 | -0.011 | 0.95 | 0.96 | 0.187 | +0.05 |
| Regime with the actual race temperature (a perfect forecast) | 0.035 | 0.049 | -0.003 | 0.91 | 0.95 | 0.164 | +0.04 |
| V3 sealed with V2's pooled geometric-mean regime | 0.039 | 0.056 | -0.010 | 0.92 | 0.94 | 0.169 | +0.05 |
| V3 sealed, per-car scale team-pooled (shipped per-car mode) | 0.036 | 0.053 | -0.003 | 0.91 | 0.94 | 0.163 | -0.03 |
| V3 sealed, per-car scale from this weekend's own dev | 0.036 | 0.053 | -0.003 | 0.91 | 0.94 | 0.163 | -0.02 |
| V3 sealed, per-car scale from the LOO race factors (V2's) | 0.038 | 0.049 | -0.010 | 0.90 | 0.94 | 0.171 | +0.05 |
| Practice posterior x regime (no history) | 0.038 | 0.055 | +0.002 | 0.86 | 0.93 | 0.167 | +0.02 |
| Practice posterior, no regime transfer | 0.066 | 0.127 | -0.043 | 0.79 | 0.89 | 0.197 | -0.01 |
| MixedLM slope x regime | 0.043 | 0.082 | -0.009 | 0.64 | 0.72 | 0.096 | +0.03 |
| Circuit history 2023-25 only | 0.044 | 0.062 | -0.008 | 0.58 | 0.64 | 0.092 | -0.01 |
| Other 2026 races' mean rate | 0.047 | 0.087 | +0.021 | 0.59 | 0.67 | 0.096 | -0.04 |
| Zero degradation | 0.068 | 0.119 | +0.065 | 0.34 | 0.39 | 0.096 | – |
| V2 sealed, re-scored here | 0.040 | 0.052 | -0.011 | 0.95 | 0.96 | 0.188 | +0.05 |
| V2 practice-only, re-scored here | 0.045 | 0.083 | -0.012 | 0.86 | 0.94 | 0.203 | -0.01 |
| V1 sealed, re-scored here | 0.109 | 0.434 | -0.083 | 0.88 | 0.94 | 0.466 | -0.07 |
| V1 practice-only, re-scored here | 0.037 | 0.054 | +0.011 | 0.92 | 0.95 | 0.175 | -0.13 |

### Decisions per weekend

| Weekend | V1 plan | V2 plan | V3 plan | Tyre-optimal | Field modal (share V1→V2→V3) | Winner | Stops=mode | Start=majority | First stop − field: tyre / V2 / V3-no-prior / V3 | Regret V1 / V2 / V3 / field / winner (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Australia | 1-stop H-M @ 36 | 1-stop M-H @ 23 | **1-stop M-H @ 25** | 1-stop H-M @ 33 | MEDIUM-HARD (0.00→0.53→0.53) | MEDIUM-HARD | yes→yes→yes | no→yes→yes | +8 / -2 / – / +0 (SC) | 10.7 / 9.0 / 7.0 / 19.8 / 24.9 |
| Japan | 1-stop S-H @ 21 | 1-stop M-H @ 23 | **1-stop M-H @ 23** | 1-stop H-S @ 33 | MEDIUM-HARD (0.00→0.85→0.85) | MEDIUM-HARD | yes→yes→yes | no→yes→yes | +15 / +5 / – / +5 (SC) | – / 0.6 / 0.6 / 0.4 / 0.5 |
| Barcelona | 2-stop M-H-H @ 19,43 | 2-stop S-M-S @ 21,43 | **2-stop M-H-H @ 19,42** | 3-stop M-M-S-S @ 16,32,49 | MEDIUM-HARD-HARD (0.46→0.00→0.46) | SOFT-HARD-MEDIUM-HARD | yes→yes→yes | yes→no→yes | +3 / +8 / +6 / +6 | 5.6 / 6.9 / 5.6 / 11.3 / 9.1 |
| Austria | 2-stop M-H-H @ 23,47 | 2-stop M-H-M @ 22,46 | **2-stop M-H-M @ 22,46** | 3-stop S-S-M-M @ 18,36,54 | MEDIUM-HARD-HARD (0.29→0.24→0.24) | MEDIUM-HARD-HARD | yes→yes→yes | yes→yes→yes | +0 / +4 / +4 / +4 | 25.1 / 14.9 / 14.9 / 28.7 / 28.8 |
| Belgium | 1-stop S-M @ 24 | 1-stop M-S @ 20 | **1-stop M-S @ 18** | 1-stop S-M @ 23 | MEDIUM-HARD (0.00→0.05→0.05) | MEDIUM-SOFT | yes→yes→yes | no→yes→yes | +7 / +4 / – / +2 (SC) | 12.8 / 14.9 / 13.2 / 12.2 / 13.2 |
| Hungary | 2-stop M-H-M @ 23,47 | 2-stop M-H-H @ 22,46 | **2-stop M-H-M @ 22,45** | 2-stop M-M-S @ 25,51 | MEDIUM-HARD-HARD (0.00→0.32→0.00) | MEDIUM-HARD-HARD-SOFT | yes→yes→yes | yes→yes→yes | +6 / +3 / +3 / +3 | 20.8 / 32.6 / 20.2 / 35.3 / 31.7 |
| Italy | 1-stop M-H @ 23 | 1-stop M-H @ 25 | **1-stop M-H @ 22** | 1-stop H-M @ 28 | MEDIUM-HARD (0.32→0.32→0.32) | HARD-MEDIUM-MEDIUM | yes→yes→yes | yes→yes→yes | – / – / – / – (SC) | 4.0 / 3.5 / 4.3 / 14.5 / 31.7 |

### Live engine (archived replays)

| | HUN V1→V2→V3 | BCN V1→V2→V3 |
|---|---|---|
| Tick mean (ms) | 1156 → 73 → **80** | 985 → 68 → **69** |
| Tick p95 (ms) | 1375 → 81 → **88** | 1320 → 74 → **75** |
| Stops inside the window | 0.48 → 0.54 → **0.52** | 0.25 → 0.34 → **0.32** |
| Stops within 3 laps | 0.33 → 0.50 → **0.43** | 0.47 → 0.57 → **0.51** |
| Median stop error (laps) | 4.0 → 3.0 → **4.0** | 3.5 → 2.0 → **3.0** |
| Box-now cost 1 lap before (s) | 1.8 → 0.7 → **0.6** | 3.2 → 2.3 → **2.9** |
| Signal precision / recall (V3) | window 0.20/0.67, box_now 0.19/0.65, collapse 0.14/0.04 | window 0.16/0.36, box_now 0.19/0.34, collapse 0.20/0.06 |

### Leave-one-out calibration (V3)

| Held out | Grip budget S / M / H (s) | Manage cost, floor | Grid | Dirty air | λ | τ | κ |
|---|---|---|---|---|---|---|---|
| australia-2026 | 3.84 / 4.09 / 4.27 | 0.60, 0.45 | 0.00 | 0.067 | 0.200 | 4.00 | 1.00 |
| japan-2026 | 3.84 / 4.09 / 4.30 | 0.60, 0.45 | 0.00 | 0.137 | 0.200 | 4.00 | 1.00 |
| barcelona-2026 | 3.80 / 3.80 / 4.19 | 0.60, 0.45 | 0.00 | 0.067 | 0.000 | 4.00 | 0.00 |
| austria-2026 | 3.84 / 4.09 / 4.30 | 0.60, 0.45 | 0.00 | 0.084 | 0.200 | 4.00 | 1.00 |
| belgium-2026 | 3.84 / 4.09 / 4.27 | 0.60, 0.45 | 0.00 | 0.137 | 0.200 | 4.00 | 1.00 |
| hungary-2026 | 3.84 / 4.09 / 4.10 | 0.60, 0.45 | 0.00 | 0.067 | 0.200 | 4.00 | 0.00 |
| italy-2026 | 3.84 / 4.09 / 4.30 | 0.60, 0.45 | 0.00 | 0.137 | 0.200 | 4.00 | 1.00 |
| global | 3.84 / 4.09 / 4.30 | 0.60, 0.45 | 0.00 | 0.120 | 0.200 | 4.00 | 1.00 |

### Ablation — the same search with one term switched off (V3)

| Variant | Sequence run by anyone | Start = majority | Stops = mode | Mean field share | Mean |first stop − field| (non-SC) | Life ratio (median) |
|---|---|---|---|---|---|---|
| full | 6/7 | 7/7 | 7/7 | 0.35 | 4.3 | 1.10 |
| no_first_stop_prior | 6/7 | 7/7 | 7/7 | 0.35 | 4.3 | 1.10 |
| no_plan_prior | 3/7 | 4/7 | 5/7 | 0.09 | 1.3 | 1.10 |
| no_nomination_mapping | 5/7 | 6/7 | 6/7 | 0.28 | 3.3 | 1.10 |
| no_dirty_air_circuit | 6/7 | 7/7 | 7/7 | 0.35 | 4.3 | 1.10 |
| no_position | 6/7 | 7/7 | 7/7 | 0.35 | 5.0 | 1.10 |
| no_pace_cal | 6/7 | 7/7 | 7/7 | 0.35 | 3.7 | 1.10 |
| config_constants | 3/7 | 1/7 | 7/7 | 0.06 | 1.7 | 1.10 |
| practice_only | 4/7 | 7/7 | 5/7 | 0.28 | 3.3 | 1.04 |
| old_budget | 6/7 | 7/7 | 7/7 | 0.35 | 4.3 | 1.10 |
| no_cliff_budgets | 6/7 | 7/7 | 7/7 | 0.35 | 4.3 | 1.10 |
| no_circuit_regime_prior | 6/7 | 7/7 | 7/7 | 0.35 | 4.7 | 1.10 |

### First stop: λ × κ diagnostic

| λ / κ | mean \|first stop − field\| (laps) |
|---|---|
| lambda 0.0 / kappa 0.0 | 4.67 |
| lambda 0.0 / kappa 1.5 | 4.00 |
| lambda 0.0 / kappa 3.0 | 3.67 |
| lambda 0.0 / kappa 6.0 | 3.33 |
| lambda 0.15 / kappa 0.0 | 4.33 |
| lambda 0.15 / kappa 1.5 | 3.67 |
| lambda 0.15 / kappa 3.0 | 3.33 |
| lambda 0.15 / kappa 6.0 | 3.00 |
| lambda 0.3 / kappa 0.0 | 4.33 |
| lambda 0.3 / kappa 1.5 | 3.67 |
| lambda 0.3 / kappa 3.0 | 3.00 |
| lambda 0.3 / kappa 6.0 | 2.67 |
| lambda 0.6 / kappa 0.0 | 1.33 |
| lambda 0.6 / kappa 1.5 | 2.67 |
| lambda 0.6 / kappa 3.0 | 2.67 |
| lambda 0.6 / kappa 6.0 | 2.33 |

kappa moves Hungary onto the field's lap (23 -> 19) because the circuit's history (mode 18) agrees with the 2026 field (19); at Barcelona and Austria no weight moves the first stop below the history's own mode (17; 21-25) while the 2026 field stopped at 13 and 18, earlier than any historical year. lambda 0.6 with kappa 0 reaches 1.33 laps only by switching Austria and Hungary onto SOFT-start families (S-M-H) that 1 of 17 and 0 of 19 finishers ran. The residual first-stop error is between the circuit's history and the 2026 field, not in the objective.

