### Accuracy per weekend (stint-rate MAE, s/lap)

| Weekend | Previous sealed | Updated sealed | Updated practice-only | Oracle | Updated bias | 90% rate coverage prev → new | Width prev → new | Regime prev → new (self) |
|---|---|---|---|---|---|---|---|---|
| Australia | 0.434 | **0.049** | 0.055 | 0.045 | 0.012 | 0.91 → 0.78 | 1.767 → 0.149 | 0.57 → 0.61 (0.75) |
| Japan | 0.024 | **0.029** | 0.018 | 0.017 | -0.029 | 1.00 → 1.00 | 0.295 → 0.199 | 0.50 → 0.63 (0.73) |
| Barcelona | 0.041 | **0.040** | 0.050 | 0.040 | 0.010 | 1.00 → 1.00 | 0.335 → 0.251 | 0.52 → 0.51 (0.70) |
| Austria | 0.032 | **0.033** | 0.036 | 0.030 | -0.017 | 1.00 → 0.97 | 0.306 → 0.199 | 0.51 → 0.69 (0.58) |
| Belgium | 0.030 | **0.052** | 0.083 | 0.048 | -0.046 | 1.00 → 1.00 | 0.191 → 0.199 | 0.58 → 0.80 (0.32) |
| Hungary | 0.046 | **0.044** | 0.045 | 0.042 | -0.004 | 0.95 → 0.91 | 0.240 → 0.199 | 0.52 → 0.83 (0.64) |
| Italy | 0.154 | **0.030** | 0.031 | 0.032 | -0.009 | 0.86 → 0.95 | 0.581 → 0.123 | 0.60 → 0.49 (0.81) |
| **Mean** | 0.109 | **0.040** | 0.045 | 0.036 | -0.012 | 0.96 → 0.95 | 0.531 → 0.188 | |

### Pooled over the 7 weekends, every variant

| Curve source | Rate MAE mean | max | Bias | 90% coverage | Width | Spearman |
|---|---|---|---|---|---|---|
| Oracle (this race's own rates, in-sample) | 0.036 | 0.048 | +0.008 | 0.71 | 0.096 | +0.10 |
| **Updated sealed** (history fold-in fixed, temperature regime) | 0.040 | 0.052 | -0.012 | 0.95 | 0.188 | +0.05 |
| Updated sealed with a perfect race-temperature forecast | 0.038 | 0.049 | -0.015 | 0.92 | 0.178 | +0.06 |
| Updated sealed with the old pooled regime | 0.038 | 0.049 | -0.010 | 0.92 | 0.187 | +0.04 |
| Updated sealed, scaled per driver (LOO race factor) | 0.044 | 0.061 | -0.018 | 0.91 | 0.197 | +0.02 |
| Practice posterior × regime (no history) | 0.045 | 0.083 | -0.012 | 0.86 | 0.202 | +0.02 |
| Practice posterior, no regime transfer | 0.068 | 0.140 | -0.045 | 0.77 | 0.198 | -0.01 |
| MixedLM slope × regime | 0.047 | 0.073 | -0.018 | 0.50 | 0.096 | +0.05 |
| Circuit history 2023–25 only | 0.044 | 0.062 | -0.008 | 0.58 | 0.092 | -0.01 |
| Other 2026 races' mean rate | 0.047 | 0.087 | +0.021 | 0.59 | 0.096 | -0.04 |
| Zero degradation | 0.068 | 0.119 | +0.065 | 0.34 | 0.096 | +nan |
| Previous sealed, rescored with the race noise | 0.109 | 0.434 | -0.083 | 0.88 | 0.466 | -0.07 |
| Previous sealed, as reported before | 0.109 | 0.434 | -0.083 | 0.96 | 0.531 | -0.07 |
| Previous practice-only, race noise | 0.037 | 0.054 | +0.011 | 0.92 | 0.175 | -0.13 |

### Decisions per weekend

| Weekend | Previous plan | Updated plan | Tyre-optimal | Field modal (share prev → new) | Winner | Stops = mode prev → new | Start = majority prev → new | First stop − field: prev / tyre / new | Regret prev / new / field / winner (s) |
|---|---|---|---|---|---|---|---|---|---|
| Australia | 1-stop H-M @ 36 | **1-stop M-H @ 23** | 1-stop H-M @ 34 | MEDIUM-HARD (0.00 → 0.53) | MEDIUM-HARD | yes → yes | no → yes | +11 / +9 / -2 (SC) | 11.2 / 9.5 / 20.5 / 25.8 |
| Japan | 1-stop S-H @ 21 | **1-stop M-H @ 23** | 1-stop M-S @ 31 | MEDIUM-HARD (0.00 → 0.85) | MEDIUM-HARD | yes → yes | no → yes | +3 / +13 / +5 (SC) | – / 0.5 / 0.4 / 0.4 |
| Barcelona | 2-stop M-H-H @ 19,43 | **2-stop S-M-S @ 21,43** | 3-stop M-M-S-S @ 16,32,49 | MEDIUM-HARD-HARD (0.46 → 0.00) | SOFT-HARD-MEDIUM-HARD | yes → yes | yes → no | +6 / +3 / +8 | 5.3 / 5.5 / 10.6 / 8.1 |
| Austria | 2-stop M-H-H @ 23,47 | **2-stop M-H-M @ 22,46** | 3-stop M-S-S-M @ 17,35,53 | MEDIUM-HARD-HARD (0.29 → 0.24) | MEDIUM-HARD-HARD | yes → yes | yes → yes | +5 / -1 / +4 | 22.4 / 13.4 / 25.8 / 25.8 |
| Belgium | 1-stop S-M @ 24 | **1-stop M-S @ 20** | 1-stop S-M @ 24 | MEDIUM-HARD (0.00 → 0.05) | MEDIUM-SOFT | yes → yes | no → yes | +8 / +8 / +4 (SC) | 13.9 / 16.0 / 10.6 / 14.4 |
| Hungary | 2-stop M-H-M @ 23,47 | **2-stop M-H-H @ 22,46** | 2-stop M-M-S @ 25,51 | MEDIUM-HARD-HARD (0.00 → 0.32) | MEDIUM-HARD-HARD-SOFT | yes → yes | yes → yes | +4 / +6 / +3 | 16.8 / 26.3 / 28.8 / 27.4 |
| Italy | 1-stop M-H @ 23 | **1-stop M-H @ 25** | 1-stop H-M @ 28 | MEDIUM-HARD (0.32 → 0.32) | HARD-MEDIUM-MEDIUM | yes → yes | yes → yes | – / – / – (SC) | 2.7 / 2.1 / 12.8 / 30.7 |

### Leave-one-out calibration

| Held out | Grip budget S / M / H (s) | Manage cost, floor | Grid penalty | Dirty air | λ | τ |
|---|---|---|---|---|---|---|
| australia-2026 | 3.80 / 3.83 / 3.84 | 0.90, 0.45 | 0.00 | 0.067 | 0.300 | 2.50 |
| japan-2026 | 3.80 / 3.83 / 3.84 | 0.90, 0.35 | 0.00 | 0.137 | 0.200 | 6.00 |
| barcelona-2026 | 3.80 / 3.80 / 3.84 | 0.90, 0.35 | 0.00 | 0.067 | 0.100 | 4.00 |
| austria-2026 | 3.80 / 3.83 / 3.84 | 0.60, 0.55 | 0.00 | 0.084 | 0.150 | 4.00 |
| belgium-2026 | 3.80 / 3.83 / 3.84 | 0.60, 0.55 | 0.00 | 0.137 | 0.150 | 4.00 |
| hungary-2026 | 3.80 / 3.83 / 3.27 | 0.60, 0.55 | 0.00 | 0.067 | 0.150 | 2.50 |
| italy-2026 | 3.80 / 3.83 / 3.84 | 0.60, 0.45 | 0.00 | 0.137 | 0.300 | 4.00 |
| global | 3.80 / 3.83 / 3.84 | 0.60, 0.45 | 0.00 | 0.120 | 0.300 | 4.00 |
