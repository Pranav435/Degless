# V4 Task 1 — race state times the pit stop

*13 September 2026. Same seven weekends, same 196 stints, same benchmark scripts as V3. The V3 numbers below come from V3's own code (commit `3cf72aa`), re-run on this machine in a separate git worktree (a Sonnet 5 agent ran that baseline). Every V3 decision and quality metric reproduced the committed Mac run exactly; only wall-clock timings differ. V3's outputs are frozen under `bench/v3/`. The tyre/degradation model and `calibration.json` are unchanged.*

**Bottom line.** The mean first-stop error on the three non-safety-car weekends falls from **4.33 to 1.67 laps**, meeting the ≤2 target. The share of the field's first stops inside the model's window rises from 0.36 to 0.45. The live stop calls improve on every accuracy metric. Sequence, start-compound and stop-count matches are unchanged: V4 picks V3's plan family at every weekend and only moves the first stop, 1–4 laps earlier. The cost is **oracle regret, up from 9.40 to 10.92 s**. The oracle prices pure race time and cannot see track position, and V4 deliberately trades race time for track position.

## 1. What changed

**The decision.** A new module, `src/racestate.py`, makes the pit call against the cars being raced rather than on one car's cost surface. For each candidate stop lap `s` of our car and stop lap `l` of each relevant rival, both cars' race time is priced lap by lap from their own tyres until both are out of the pits. The rival is then ahead with probability `Φ((gap + D(s,l)) / σ)`. The undercut, the overcut, compounds, tyre ages, an undercut already in progress and a safety-car stop all sit inside `D`. Each rival picks its own stop from its own cost curve, and covers (boxes the lap after us) when the place it would lose is worth more than the stop it moves. The expected places lost, times the value of a place, is added to the V3 tyre, pit-lane and traffic cost. Every term is in seconds.

- **Live** (`src/live/engine.py`): each tick now runs in two passes. The first is V3's per-car costing. The second re-prices each option's *next* stop against the 4 cars nearest in *virtual* position (gap corrected by a pit loss per stop of difference, within 6 s). It uses their real gaps, compounds, tyre ages, stops made, pit status and the plans the engine gave them on the previous lap. It also takes rejoin traffic from the timing screen for the next 4 laps, and the SC/VSC pit factor applies to both us and rivals. Every plan carries the explicit comparison **PIT NOW / STAY OUT 1 / 2 / 3 LAPS / PIT AT EDGE OF WINDOW** (`plan["race_state"]`), with tyre, position and traffic parts, the expected rejoin position and the per-rival detail.
- **Pre-race** (`src/strategy.py`, `scripts/10_pipeline.py`, `scripts/40_weekend.py`): there is no race state before the race, so it is simulated. Each plan group (start compound, second compound, stop count) is solved as a pack of four rivals at gaps drawn from the measured first-stint intervals. All run the same plan, and the pack is iterated to its symmetric fixed point (every car's stop a best response to the others'). That term replaces V3's undercut exposure *and* first-stop history prior on the first stop. Later stops keep V3's undercut term. The plan-family prior τ and the circuit's stint caps (the edge of the window) stay: history now says what is plausible, not when to box. `meta["race_state"]` records the pack and the five-action table for every lap from six before the stop to the stop.

**Constants.** All are measured leave-one-out on the other 2026 races; none is tuned to the benchmark.

| Constant | Race-time meaning | Value (LOO range) |
|---|---|---|
| place gap `V` | median interval between adjacent classified finishers | 4.6–5.9 s |
| persistence `ψ` | share of adjacent pit-cycle pairs whose order holds to the flag (25–35 pairs) | 0.68–0.81 |
| **value of a place** | `V (2ψ − 1)`: a place won in the cycle, discounted by how often it is overturned | **2.0–3.6 s** |
| `σ` | √2 × robust SD of green-flag pit loss (two cars' pit cycles) | 2.5–2.9 s |
| pack gaps | intervals, laps 5–15, P2–P15 | median 1.27–1.30 s |

No weight is fitted. The choices that are not measurements:

- 4 rivals (the brief says 3–5).
- Rivals choose among near-equal laps at the engine's existing 1 s window tolerance.
- The measurement windows: pack gaps on laps 5–15 at P2–P15; cycle pairs are first stops within 5 laps.
- Numerical settings: 8 gap quantiles per rival, a 0.5-damped fixed point, a 25-lap live horizon, and V3's 3 s traffic band.

**Disclosure.** The first implementation measured `V` on lead-lap finishers only. That gave 2.33 laps. I switched to all classified finishers (gap at the last lap both completed) *after seeing that result*. My reason: the lead-lap sample is a front-runner sample (4–5 cars at Melbourne and Barcelona) and excludes the lapped midfield this tool decides for. Both versions are in the ablation.

**Also:** `--no-race-state` flags in `10_pipeline.py` and `bench_live.py`. 10 new tests (`tests/test_v4_racestate.py`). New ablation variants. `bench/bench_tick_paired.py` (on/off tick time in one process) and `bench/bench_v4_compare.py`. Windows portability fixes: sealed files restored from git with LF (autocrlf had broken their sha256); a peak-memory fallback in `bench_speed.py`; suites run with `PYTHONUTF8=1`.

## 2. V3 vs Task 1

| Metric (definitions unchanged) | V3 | V4 | |
|---|---|---|---|
| Mean \|first stop − field green-flag median\| (laps, non-SC) | 4.333 | **1.667** | improved (target ≤2 met) |
| Signed first-stop error (laps) | +4.33 | +1.00 | improved |
| Share of field first stops inside the model's window | 0.361 | **0.448** | improved |
| Oracle regret of the tool's plan (s) | 9.399 | **10.917** | **regressed** |
| Sequence run by anyone / start = majority / stops = mode (of 7) | 6 / 7 / 7 | 6 / 7 / 7 | unchanged |
| Mean field share on the recommended sequence | 0.350 | 0.350 | unchanged |
| Live: real stops inside the window 3 laps earlier | 0.421 | 0.443 | improved |
| Live: stops within 3 laps of the recommendation | 0.472 | **0.547** | improved |
| Live: median \|recommended − actual\| (laps) | 3.5 | **3.0** | improved |
| Live: box-now cost the lap before the real stop (s) | 1.755 | **1.030** | improved |
| Live signal precision / recall: window | 0.180 / 0.516 | 0.175 / 0.566 | recall up, precision −0.005 |
| Live signal precision / recall: box-now | 0.193 / 0.496 | 0.199 / 0.604 | improved |
| Live tick, BCN / HUN median (ms), one process, race state off (= V3's algorithm) vs on | 287 / 317 | 298 / 341 | **+13.5 / +18.4 ms (+6–9%)** |
| Tests (this machine) | 43 pass, 2 skip | 53 pass, 2 skip | +10 |
| Gates failing | 3 | 4 | regressed (Belgium, §5) |
| Per-car plans sharing the field plan's shape | 0.862 | 0.721 | regressed |

Per live replay (Hungary / Barcelona): the in-window share goes 0.52 → 0.57 / 0.32 → 0.32. Within 3 laps goes 0.43 → 0.57 / 0.51 → 0.53. The median error goes 4 → 3 / 3 → 3. The box-now cost goes 0.57 → 0.30 / 2.94 → 1.76 s.

On tick time: this laptop (1.3 GHz, with the dashboard process running in the background) runs about 4× slower than the Mac V3 was measured on (75 ms). Separate runs minutes apart gave V3 311 ms and V4 591 ms mean. But V4 *with the race state off* gave 600 ms in the same window, so that gap is the machine, not the term. The paired run is the fair measure. Everything stays well inside a lap.

## 3. Ablation (race state disabled, and its parts)

Same 300-draw search as V3's ablation; first-stop error on the three non-SC weekends.

| Variant | Seq / start / stops (of 7) | \|first stop − field\| | BCN / AUT / HUN |
|---|---|---|---|
| **full (V4)** | 6 / 7 / 7 | **1.67** | +2 / +2 / −1 |
| **no_race_state** (= V3 objective) | 6 / 7 / 7 | 4.33 | +6 / +4 / +3 |
| race_state_no_cover (rivals never cover) | 6 / 7 / 7 | 1.33 | +1 / +1 / −2 |
| race_state_lead_lap_value (first definition) | 6 / 7 / 7 | 2.33 | +3 / +3 / +1 |
| race_state_undiscounted (ψ = 1) | **5 / 6** / 7 | 2.00 | 0 / −2 / −4 (Austria flips to S-H-M) |
| no_plan_prior | 3 / 4 / 5 | 1.67 | — |
| no_position (λ = 0 on later stops) | 6 / 7 / 7 | 1.33 | — |

- `no_race_state` reproduces V3 exactly (19,42 / 22,46 / 22,45). The live replays with the race state off reproduce V3's live metrics exactly.
- The race state alone is responsible for the whole 2.67-lap gain.
- Dropping the cover response scores better here (1.33), but it is kept: rivals do cover, and removing it would be tuning to three weekends.
- Removing the persistence discount over-unravels. Austria moves onto a SOFT start nobody ran, which is what the discount exists to prevent.

## 4. First-stop analysis

| Weekend | Field green median | V3 | V4 | Tyre alone (same group) | Pack median (IQR) | V4 window | Place value |
|---|---|---|---|---|---|---|---|
| Barcelona | 13 | 19 (+6) | **15 (+2)** | 19 | 16 (15–17) | 14–17 | 2.44 s |
| Austria | 18 | 22 (+4) | **20 (+2)** | 24 | 20 (18–21) | 18–22 | 2.04 s |
| Hungary | 19 | 22 (+3) | **18 (−1)** | 24 | 19 (17–20) | 16–21 | 2.61 s |
| Japan (SC) | 18 | 23 | 19 | 23 | 19 (17–21) | 16–22 | 2.58 s |
| Belgium (SC) | 16 | 18 | 16 | 21 | 16 (16–17) | 16–16 | 3.59 s |
| Australia (SC) | 25 | 25 | 21 | 26 | 21 (20–23) | 18–24 | 2.43 s |
| Italy (SC) | — | 22 | 21 | 26 | 21 (19–23) | 18–24 | 2.51 s |

- The race state moves every first stop 1–4 laps earlier than the tyre alone would, never later, and never changes the family.
- Mechanism, Hungary pre-race lap 18: PIT NOW costs 1.95 s of tyre against the tyre optimum but leaves 1.93 of 4 rivals expected ahead. STAY OUT 1 LAP saves 0.62 s of tyre and loses 0.63 s of position, so PIT NOW wins by 0.01 s.
- **Barcelona stays 2 laps late.** Its 2026 field stopped at 11–15, earlier than even the pack's interquartile range: soft-starters boxed first and the medium-runners covered. The symmetric pack gives every rival the same plan, so it cannot generate that first mover.
- **Austria stays 2 laps late** because its leave-one-out persistence is the lowest (0.68): its own race, with 9 of 10 pairs kept, is excluded. That makes its place value the smallest.

## 5. Failures and remaining weaknesses

- **Oracle regret regressed by 1.52 s** (Australia +4.3, Hungary +3.4, Barcelona +2.1, Austria +1.6, Belgium −1.0). Within V3's own families, both the oracle and the tyre model price later stops as cheaper. The oracle has no notion of a place. Whether the trade is right is exactly what this benchmark cannot judge; by its own definition it is a regression.
- **The place-value definition was changed after the first result** (§1). With lead-lap gaps the first-stop error is 2.33 laps, just outside target.
- **ψ rests on 25–35 cycle pairs.** Leave-one-out it swings 0.68–0.81, and it is the most influential constant: place value 2.0–3.6 s.
- **Symmetric pack.** Pre-race rivals all run our family, so no soft-starter first movers (Barcelona). Only the first stop gets a pre-race race-state term; the second still uses V3's exposure.
- **Window share fell at three weekends** despite the overall rise: Australia 1.0 → 0.0 (only 2 green stops, both late), Belgium 0.33 → 0.22 (its window collapsed to one lap), Hungary 0.42 → 0.33 (2-stoppers' first stops spread from lap 13 to 34).
- **New gate failure at Belgium.** Stopping on lap 16 leaves a 28-lap SOFT stint, and the longest run there was 26.
- **Live:** our own policy is evaluated open-loop (rivals cover us; we only re-decide next lap). Rejoin traffic uses current gaps and only 4 laps ahead. Lapped cars are ignored. Window-signal precision dips 0.005. Tick +6–9%.
- **Per-car plans:** the same-shape share falls 0.86 → 0.72; each driver's search now runs its own pack.
- **Not converted:** the outlook, the counterfactual and the desk's `evaluate_plans` still price V3's objective. `calibration.json` was not re-run, so τ was calibrated alongside V3's λ and κ.

**Reproduce:** `python scripts/10_pipeline.py --event <k> --stage decide --offline` for the seven weekends. Then the suite: `bench/run_all.sh`'s stages; on Windows run with `PYTHONUTF8=1`. Then `bench/bench_live.py --no-race-state`, `bench/bench_tick_paired.py`, and `bench/bench_v4_compare.py` → `bench/out/v4_compare.json`.
