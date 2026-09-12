# V4 feature audit (WP-H)

*Documentation only, no code changed. Scope: every input that reaches a strategy decision in the repository as it stands at commit `04fa10a` — i.e. the Task 1 codebase, before WP-A (hetero rival field), WP-B (tyre-life uncertainty), WP-C (position metric), WP-D (`src/live/rivals.py`, `src/explain.py`), WP-E (`src/haascar.py`) or WP-F (`src/objective.py`) exist. None of those four files exist yet (checked by listing). This audit is therefore the baseline the other work packages change, not a description of the target architecture. Classification (CORE / SUPPORTING / DEPRIORITISED) is per `v4_prompt.md`, "ISSUE 6 — Reduce unnecessary variables". Entry-mode vocabulary follows `docs/v4_plan.md` §0/§2(e): measured this weekend / measured LOO on donors / historical prior / physical constant / calibrated constant / engineering constant.*

*Values quoted for calibrated constants are read from `data/processed/calibration.json` (`written_utc` 2026-09-12T17:22Z, global block, seven-weekend LOO) via `.venv/bin/python`; race-state constants are `src.racestate.measure_constants` evaluated live for four example exclusions.*

## 0. Legend

| Class | Meaning (from the prompt) |
|---|---|
| **CORE** | compound, tyre age, degradation, current/near-term tyre pace, tyre-life uncertainty, fuel/mass, track evolution, sector pace, push/management, current-weekend Haas evidence, position, gap ahead/behind, competitor compound/age/pit status, rejoin position, traffic, pit loss, SC/VSC |
| **SUPPORTING** | apex degradation, circuit-specific dirty air, race/practice regime, compound nomination mapping, historical strategy distributions, temperature *when race-day information is actually available* |
| **DEPRIORITISED** | humidity, wind, generic straight-line speed, historical global driver factors, excessive apex features, generic temperature corrections, duplicate traffic features, arbitrary historical bonuses, arbitrary constants |

Decision tags: **Family** (which compounds/how many stops), **First-stop** (when to take it), **Later-stops**, **Live** (in-race call), **Per-car** (Ocon/Bearman divergence).

---

## 1. Offline strategy search

### 1.1 Tyre model core — `src/tyre.py`, `src/config.py` (physics, ~53–378)

| Input | What | Entry mode | Value / source | Decision | Class | Ablation |
|---|---|---|---|---|---|---|
| Compound identity | SOFT/MEDIUM/HARD ordering, ladder | physical + calibrated | `COMPOUND_ORDER` config.py:141; pace step `COMPOUND_PACE_STEP_FRAC=0.0021` config.py:156 (net-of-degradation calibration, config.py:159–198); deg ratio `COMPOUND_DEG_RATIO=1.30` config.py:217 (FE-corrected measurement, config.py:200–216) | Family | CORE | — |
| Degradation rate `wear_rate[c]` | practice fit's slope ÷ grip budget | measured this weekend (practice fit) | `TyreModel.from_fit` tyre.py:264–315; combined with circuit history via `history.apply_circuit_prior` (§2.1) | Family, First-stop, Later-stops | CORE | `practice_only` (existing) |
| Grip budget | seconds of pace a tyre surrenders to its cliff | calibrated (per compound, LOO, censored estimate) | pooled default `GRIP_BUDGET_S=3.8` config.py:335 (Barcelona-only derivation, config.py:316–334); **calibrated** `grip_budget_by_compound` = `{SOFT 3.84, MEDIUM 4.09, HARD 4.30}` (calibration.json, global) | Family, all stops | CORE | `no_cliff_budgets`, `old_budget` (existing) |
| Cliff shape | `CLIFF_SHARPNESS=8.0`, `CLIFF_EXPONENT=2.0`, `CLIFF_SMOOTH=0.05` | engineering constant | config.py:350–352, rationale 345–349 (target shape: ~1.5x loss at wear 1.25, ~3x at 1.5) — **the numbers 8.0/2.0 themselves have no fit behind them, only the target shape**; see §6 | All stops | CORE mechanism / **arbitrary constant** | sensitivity sweep (proposed, none exists) |
| Push/management trade-off | `MANAGE_COST_S`, `MANAGE_WEAR_FLOOR`, `MANAGE_WEAR_EXPONENT`, `PUSH_GRID` | calibrated (cost/floor) + engineering (exponent, grid) | config default 0.9/0.45/1.7 (config.py:371–378); **calibrated** `manage_cost_s=0.6`, `manage_wear_floor=0.45` (calibration.json, stable across all 7 LOO folds) | Family (push level chosen per plan) | CORE | — |
| Fuel/mass load profile | per-lap wear multiplier from car mass | physical (2026 regs) | `TYRE_LOAD_EXPONENT=1.6` config.py:88 (derivation 69–87, superseding an unidentified 5.0); `fuel.mass_profile` fuel.py:78–89, `tyre.load_profile` tyre.py:186–198 | Family (stint order) | CORE | — |
| Warm-up cost | cold out-lap penalty | engineering constant | `OUT_LAP_PENALTY_S=0.7` config.py:442 (439–441) | First-stop, Later-stops | CORE | — |
| Per-driver deviation `dev[d,c]` | practice fit's own slope deviation, this car, this weekend | measured this weekend | `TyreModel.for_driver` tyre.py:341–380; `driver_dev` from `BayesFit.posterior["dev"]` | Per-car | CORE | `no_percar` (existing) |

### 1.2 Race-time costs beyond the tyre — `src/strategy.py`

| Input | What | Entry mode | Value / source | Decision | Class | Ablation |
|---|---|---|---|---|---|---|
| Pit loss | in-lap + out-lap − 2×median clean lap, green-flag stops only | measured this weekend | `measure_pit_loss` strategy.py:147–196 | All stops | CORE | — |
| Traffic / dirty air | rejoin-lap dirty-air cost from a quadratic field-density curve | measured (pooled 2026) + calibrated (per circuit) | `traffic_density` strategy.py:204–217 (quadratic `0.692−1.037f+0.826f²`); `traffic_cost` strategy.py:220–237; config default `DIRTY_AIR_S_PER_LAP=0.45` config.py:390, `TRAFFIC_LAPS_PER_STOP=1.2` config.py:397; **calibrated per circuit**, e.g. Budapest 0.41, Monza 0.10 s/lap (`dirty_air_by_circuit`, calibration.json) — a 4× spread, so pooling would misprice every circuit | Family, all stops | CORE (traffic) / SUPPORTING (per-circuit value) | `no_dirty_air_circuit` (existing) |
| Safety-car credit | expected value of a stop not yet taken, under a Poisson SC arrival | physical model + measured rate | `safety_car_credit` strategy.py:240–259; `SC_RATE_PER_LAP=0.006`, `SC_PIT_LOSS_FRACTION=0.40` config.py:404–405 | Family (last scheduled stop) | CORE | — |
| Undercut exposure λ | generic one-lap undercut gain, cumulative, charged on **stops after the first** when race-state is on (on the first stop too when it is off) | calibrated LOO | `undercut_exposure_tables` strategy.py:262–296; default `UNDERCUT_EXPOSURE_LAMBDA=0.15` config.py:459; **calibrated** `undercut_lambda=0.2` (0.0 for the Barcelona fold) | Later-stops (First-stop when race-state off) | CORE | `no_position` (existing) |
| Plan-family prior τ | `tau*(−log p(family))` from the circuit's sequence/start/stop-count history | historical prior, calibrated weight | `plan_prior_penalty` strategy.py:299–330; default `PLAN_PRIOR_TAU_S=2.5` config.py:469; **calibrated** `plan_prior_tau_s=4.0` (stable, all 7 folds) | Family | SUPPORTING | `no_plan_prior` (existing) |
| First-stop history prior κ | `kappa*neglogp[lap]` from the circuit's green first-stop KDE | historical prior, calibrated weight | `first_stop_penalty` strategy.py:333–378, mechanism in `src/firststop.py` (§2); default `FIRST_STOP_KAPPA_S=1.0` config.py:486; **calibrated** 1.0 (0.0 for Barcelona and Hungary folds) — **but structurally forced to 0 in production whenever a race state is passed** (`scripts/10_pipeline.py:603` `kappa_used = 0.0 if use_rs else cal.first_stop_kappa_s`) | First-stop (only where race-state is off) | SUPPORTING, **currently inert in the shipped path** | `no_race_state`, `no_first_stop_prior` (existing) |
| Grid-start penalty | 1 s/step-of-hardness on the opening stint only, from revealed preference (0/32 finishers started HARD) | engineering constant, "reported as a calibrated preference, not a measurement" (config.py:431–436) | default `GRID_START_PENALTY_S=1.0` config.py:437; **calibrated `grid_start_penalty_s = 0.0` in every one of the 7 LOO folds** | Family (opening compound) | CORE mechanism / **arbitrary constant, currently inert** | see §6/§8 |

### 1.3 Race state / rival pack — `src/racestate.py`

| Input | What | Entry mode | Value / source | Decision | Class | Ablation |
|---|---|---|---|---|---|---|
| Place value `V` | median race-time gap between adjacent classified finishers | measured LOO (donors) | racestate.py:117–164, `_race_measurements` 183–253; **example LOO values** (computed live): exclude Barcelona → V=4.58s (n=103 gaps); exclude Hungary → 5.65s (97); exclude Belgium → 5.85s (97); exclude Australia → 5.16s (101) — matches the plan's quoted 4.6–5.9s range | First-stop | CORE | `place_value_task1` (misnomer — this *is* task1; see §8) |
| Persistence ψ | share of adjacent first-pit-cycle pairs whose order 2 laps later survives to the flag | measured LOO (donors), **n=26–34 pairs** | racestate.py:229–245 | First-stop | CORE, **small-n** | `place_value_lead_lap`, `race_state_undiscounted` (existing, bench_ablation.py:15–18) |
| σ_rel (pit-cycle noise) | √2 × robust SD of green pit loss | measured LOO | racestate.py:148–150 | First-stop | CORE | — |
| Pack gaps | first-stint interval quantiles, P2–P15, laps 5–15 | measured LOO | `pack_slots` racestate.py:341–353; `PACK_LAPS=(5,15)`, `PACK_POSITIONS=(2,15)` racestate.py:92–93 | First-stop | CORE | — |
| Rival pack structure | **every rival assumed to run our own plan family** (symmetric fixed point) | model assumption, not measured | `pack_equilibrium` racestate.py:356–432; explicit design note racestate.py:56–62 "a pack of four rivals ... all on the same plan family" | First-stop (dominant driver) | **CORE mechanism, but the central weakness Issue 1 names** — currently the *only* mode | `symmetric_pack` (proposed — this literally names current production behaviour; WP-A's hetero field is the alternative) |
| `N_RIVALS=4`, `RELEVANT_RANGE_S=6.0` | pack/live rival-set size and range | engineering constant, not sensitivity-tested here | racestate.py:90–91; mirrored by live's `UNDERCUT_RANGE_S=6.0` engine.py:95 | First-stop, Live | CORE mechanism / engineering magnitude | Issue 8 calls for a k=3–6 sweep; not yet run in this codebase |
| `CHOICE_TEMPER_S=1.0` | rival's own indifference window when choosing a stop lap | engineering constant, tied to `WINDOW_TOL_S=1.0` (engine.py:93) | racestate.py:96 | First-stop | CORE mechanism | — |
| Cover mechanism | a rival boxes early if the place it saves is worth more than the stop it moves | model (logit), value-weighted | `expected_ahead` racestate.py:304–333 | First-stop only (offline) | CORE | `race_state_no_cover` (existing) |

### 1.4 Search bounds and enumeration — `src/strategy.py`, `src/config.py`

| Input | What | Entry mode | Value / source | Decision | Class | Ablation |
|---|---|---|---|---|---|---|
| `life_caps` | longest stint the search considers per compound | physical (wear bound) + engineering (extrapolation limit) | `life_caps` strategy.py:407–430; `MAX_WEAR_LIMIT=1.35` config.py:517 (rationale 505–516); `SUPPORT_EXTRAPOLATION_LIMIT=2.0` config.py:525 (519–524) | Family (stint length ceiling) | CORE | this is the Belgium-gate mechanism (see below) |
| Tyre-life uncertainty (extrapolation-aware) | **does not exist in this codebase.** `life_lo`/`life_hi` (tyre.py:501–502) come only from posterior-draw quantiles (parameter uncertainty at a fixed length); nothing widens with distance beyond practice support — confirmed by `grep -rn "extrap_ln_sd"` returning nothing | — | This is exactly the WP-B/Issue 7 gap and the documented mechanism of the Belgium gate failure (`docs/v4_plan.md` §0: "a stint planned at the edge of the practice support ... is priced as if the extrapolation were certain") | Family, First-stop | **CORE, currently missing** | `extrap_off` (proposed; would be a no-op today since the mechanism doesn't exist — becomes meaningful once WP-B ships) |
| `MAX_STINTS_PER_COMPOUND=2` | at most 2 sets of one compound | measured (43/44 driver-races) | config.py:503 (496–502) | Family | CORE | — |
| `MC_DRAWS=500`, `PIT_WINDOW_MARGIN=6`, `MAX_STOPS=3`, `PUSH_GRID=(0.55,0.7,0.85,1.0)` | search resolution, not a price | engineering constant | config.py:411–413, 378 | Family (search only) | engineering, not a decision input | — |

---

## 2. Pre-race fit/priors consumed by the search

### 2.1 Circuit and season history — `src/history.py`

| Input | What | Entry mode | Value / source | Decision | Class | Ablation |
|---|---|---|---|---|---|---|
| `circuit_prior.rate_prior[c]` | degradation-rate prior per compound from the circuit's 2023–25 races | historical prior | history.py:714–730; blended into the practice posterior by `apply_circuit_prior` history.py:1051–1140 (rate only, capped `MAX_HISTORY_SCALE=3.0` history.py:76, floored at `rate_floor`) | Family, all stops | SUPPORTING | `no_circuit_regime_prior`, `practice_only` (existing) |
| `season_factor()` | 2026/2025 race-degradation ratio on shared circuits | historical, measured | history.py:563–606 | Family (scales `rate_prior`) | SUPPORTING | folded into `no_circuit_regime_prior` |
| Thermal multiplier (circuit history) | scales `rate_prior[c]["mean_s_per_lap"]` by `exp(β·(T_practice,now − T_race,hist))`, clipped [0.6, 1.7] | **measured this weekend** (practice track temp, real) vs historical race temps | history.py:703–713; `β=0.025`/°C from `thermal_sensitivity()` history.py:614–628 (99 circuit-compound-years); **always on** once circuit history exists and practice has run — not gated on a race-day forecast | Family, all stops | SUPPORTING | `no_temperature_mode` (proposed; new — see §7) |
| Regime factor `RegimeFactor` (fit-time use) | practice→race degradation ratio, donor-pooled + circuit's own regime history, precision-combined | measured LOO (donors) + historical (circuit) | `regime_prior` regime.py:422–591; feeds `apply_circuit_prior(f, ev, regime, cp)` (10_pipeline.py:314) — **this materially rescales the fitted wear rate** | Family, all stops | SUPPORTING | `no_circuit_regime_prior` (existing) |
| Regime factor, *thermal term* (`regime_prior`'s own correction) | corrects each donor's log ratio by β×ΔT before pooling | **gated on a race-day forecast** (`temperature_model="auto"`); off (`mode="none"`) in every retrospective/backtest run in this repo since none pass `race_temp_c` | regime.py:405–419, 34–45; `scripts/40_weekend.py:173` is the only call site that can supply `--race-temp` | Family, all stops | SUPPORTING (correctly gated on availability, per the prompt's own wording) | `no_temperature_mode` (proposed) |
| Regime factor, *strategy-search use* | `simulate_model(regime=...)` / `strategy.regime_multipliers` | **dead code for pricing**: `regime_multipliers` (strategy.py:386–398) is defined but never called anywhere in the repo (`grep -rn regime_multipliers` returns only its own definition); `simulate_model`'s `regime` kwarg is stored only as `res.regime` for reporting (strategy.py:868) and never scales a cost table | — | Reporting only | **DEPRIORITISED (arbitrary/dead)** | none needed — recommend deleting or wiring it |
| Regime factor, *validation use* | `src.validate.seal_predictions` scales the sealed race-curve by `regime.draws(...)` to score the model against the actual race (accuracy gates) | measured LOO | validate.py:77–128 | Diagnostic (accuracy gate), not a strategy decision | out of scope for CORE/SUPPORTING | — |
| Plan-family history (`plan_prior_for`) | sequence/start/stop-count counts, mapped through Pirelli nominations before pooling | historical, mapped | history.py used via `plan_prior_for` (called at 10_pipeline.py:226); nomination mapping is `src/nominations.py` (below) | Family | SUPPORTING | `no_nomination_mapping`, `no_plan_prior` (existing) |
| Stint caps / cliff | longest + p90 stint per compound, scaled to this year's distance | historical, measured | `CircuitPrior.cliff()` history.py:666–676; `stint_caps_for` history.py:1143–1148; `CAP_MARGIN=1.10` history.py:71 | Family (feasibility bound) | CORE (a physical/feasibility fact about the circuit, not a soft prior) | — |
| Compound ladder (circuit) | fresh-tyre pace step and deg-ratio as this circuit's races show them | historical, recency-weighted | history.py:742–766, `HIST_RECENCY_WEIGHT={2023:0.5,2024:0.7,2025:1.0}` history.py:89 | Family | SUPPORTING | folded into `no_circuit_regime_prior`/`practice_only` |

### 2.2 First-stop density — `src/firststop.py`

| Input | What | Entry mode | Value / source | Decision | Class | Ablation |
|---|---|---|---|---|---|---|
| First-stop KDE, conditioned on (start compound, stop count) | Gaussian KDE over historical green first stops, backed off through 3 tiers, uniform floor | historical prior | `first_stop_prior` firststop.py:156–256; `BANDWIDTH_LAPS=2.5`, `MIN_COMPOUND_N=5`, `UNIFORM_FLOOR=0.05` firststop.py:66–70 | First-stop (**only where race-state is off — see §1.2**) | SUPPORTING | `no_race_state` (existing) exercises the whole mechanism vs its replacement |

### 2.3 Compound nomination mapping — `src/nominations.py`

| Input | What | Entry mode | Value / source | Decision | Class | Ablation |
|---|---|---|---|---|---|---|
| Pirelli C-number mapping | translates a historical role (SOFT/MEDIUM/HARD) through that year's C-numbers into the target year's roles, clamping at the ends | hand-verified reference table (`data/pirelli_nominations.json`) | `map_sequence` nominations.py:213–238, `cnumber_letter` 142–164; absent nomination → `None`, never guessed (nominations.py:26–29) | Family (via `plan_prior_for`), First-stop (via `firststop`'s `by_compound`) | SUPPORTING | `no_nomination_mapping` (existing) — this is the Barcelona-defect ablation named in `docs/v4_plan.md` §0 |

### 2.4 Per-car practice pooling — `src/percar.py`

Covered fully in §4 (per-car layer) since it is exactly that layer's input.

### 2.5 Apex/telemetry — `src/telemetry.py`, `bench/bench_apex.py`

| Input | What | Entry mode | Value / source | Decision | Class | Ablation |
|---|---|---|---|---|---|---|
| Corner apex speeds | minimum speed in a ±60 m window per corner, top-4 highest-variance slow corners | measured (telemetry), **diagnostic only** | telemetry.py:39–147; `CORNER_WINDOW_M=60`, `SLOW_CORNER_FRAC=0.6`, `N_APEX_CORNERS=4` config.py:104–106 | **None in production**: `stage_fit` builds the production posterior (`fits["2026"]`, 10_pipeline.py:301) with no `apex=` argument; apex only enters when `--joint` is passed (10_pipeline.py:278, a diagnostic branch that fits a separate, unshipped posterior) | SUPPORTING per the prompt's own list, and correctly **not** a production driver today | `bench/bench_apex.py` already *is* the ablation (joint vs lap-only, scored against the race) — no further action needed; matches the prompt's "excessive apex features" caution by construction (kept to 4 corners, diagnostic) |

### 2.6 Humidity / wind / weather — repo-wide grep

`grep -rn -i "humid" src scripts bench` → **no matches**. `grep -rn -i "wind"` (word-boundary, excluding "window") → **no matches**. Weather usage is exclusively track/air temperature (`TrackTemp`/`AirTemp` from FastF1's weather channel) and a boolean rain flag used only to classify a weekend `dry`/`wet` for the calendar (`config.py` `Event.dry`, set from a manual 2026-09-04 survey, config.py:616–621) and to exclude wet races from the circuit-history pool (`circuit_prior` history.py:693, `not r.get("rain")`). **Humidity and wind are absent from the codebase entirely — the DEPRIORITIZE list's two most explicit items are non-issues because the features do not exist**, not because they were demoted.

---

## 3. Live engine — `src/live/engine.py`

*No `src/live/rivals.py` or `src/explain.py` exist yet (WP-D). Everything below is the single-file Task 1 engine.*

| Input | What | Entry mode | Value / source | Decision | Class | Ablation |
|---|---|---|---|---|---|---|
| `N_DRAWS=300` | live posterior particle count | engineering constant | engine.py:83 | Live (all) | engineering | — |
| `SIGMA_RACE_LAP_S=0.5`, `DIRTY_SIGMA_MULT=1.6` | clean/dirty lap noise for the likelihood | engineering constant (clean value shared with config.py:494; the 1.6 multiplier itself uncited) | engine.py:84–85 | Live tyre posterior | CORE mechanism / **1.6 is an arbitrary constant** | — |
| `EVO_PRIOR_S_PER_LAP=-0.06` | track-evolution fallback before in-race data can say | measured (donor races: −0.08) | engine.py:86; `_race_evolution` engine.py:452–490 | Live (all cars' pace) | CORE | — |
| `FIELD_TEMPER=0.5` | other cars' laps count half toward a car's own tyre posterior | engineering constant, plausible rationale, no fit | engine.py:87 | Live tyre posterior | **arbitrary constant** | — |
| `MIN_ESS=40.0` | floor on effective posterior draws (tempering) | engineering constant | engine.py:88, `_temper_for_ess` engine.py:103–118 | Live (all) | engineering | — |
| `TRAFFIC_INTERVAL_S=1.0`/`TRAFFIC_SOFT_INTERVAL_S=2.0` | DRS-range dirty-lap detection for the likelihood | physically motivated (DRS range) | engine.py:89–90 | Live tyre posterior | CORE | — |
| `SLOW_LAP_MARGIN_S=3.0` (live) vs `5.0` (config.py:96, offline clean-lap rule) vs `2.5` (regime.py:89) | three different numeric thresholds for "a lap too slow to be racing" | engineering constant, **not unified** | engine.py:91; config.py:96; regime.py:89 | Live vs offline data cleaning | **arbitrary constant (inconsistent across contexts)** | — |
| `_update_tyres` — per-car wear posterior | resamples the sealed posterior by this car's own fuel/evolution-corrected clean laps this stint | measured live, this car, this lap | engine.py:492–620 | Live, Per-car | CORE | — |
| Team-pooled practice deviation, live | `pooled_dev` folded into `_rates()` | measured this weekend (from `WeekendModel.load`, engine.py:200–214) via `percar.team_pooled_dev` | engine.py:404–423 | Live, Per-car | CORE | — |
| `_plan_prepare`/`_plan_finish` — option costs | every 1- and 2-stop continuation, vectorised over the field's cached fresh-stint tables | model (offline objective replayed live) | engine.py:689–981 | Live | CORE | — |
| `_race_state_extra` — relevant rivals + place value | 4 nearest rivals in *virtual* position, real gaps/compounds/ages/stops, `racestate.live_position_term` | measured live (this weekend's timing screen) + LOO constants (`self.race_state`) | engine.py:1047–1100; rivals via `racestate.relevant_rivals` racestate.py:521–540 | Live (next-stop pricing, any stop) | CORE | `no_race_state`-equivalent: `race_state=False` at `RaceEngine.__init__` (engine.py:371) |
| Rejoin traffic (live) | projected traffic band from the *current* timing-screen gaps, replacing the density model on the option's own next stop | measured live | `racestate.rejoin_traffic` racestate.py:629–650, called at engine.py:1082–1087 | Live | CORE | see §5 (duplicate discussion) |
| `_density` (V3 quadratic, live) | the same `0.692−1.037f+0.826f²` curve as `strategy.traffic_density`, recomputed in engine.py | measured (pooled 2026) | engine.py:1102–1105 | Live, for every option **except** the one the race-state term re-prices | CORE | see §5 |
| `_undercut` — threat/opportunity display | one-lap undercut gain vs the car directly ahead/behind, `UNDERCUT_RANGE_S=6.0` | model, live gaps | engine.py:1109–1181 | Live (display, not the plan search itself) | CORE | — |
| `_rejoin` | naive projected rejoin position from current gaps + a fixed extra time | measured live | engine.py:1185–1209 | Live (display) | CORE | — |
| `_measure_pit_loss` (live) | live-measured pit loss, falls back to the pre-race prior below n=2 | measured live | engine.py:1213–1243 | Live, all cars | CORE | — |
| `WINDOW_TOL_S=1.0`, `VSC_PIT_LOSS_FRACTION=0.55`, `CLIFF_ALARM_P=0.5`, `ALERT_COOLDOWN_LAPS=3`, `PLAN_SHORTLIST=48`, `COLLAPSE_MIN_LAPS=6`, `BOX_NOW_TOL_S=0.5` | live decision/alert tolerances | engineering constant | engine.py:93–100; `BOX_NOW_TOL_S` gates the box-now flag at engine.py:1366, `CLIFF_ALARM_P` gates the wear alarm at engine.py:1303 | Live | engineering (several are decision-adjacent, e.g. `BOX_NOW_TOL_S`) | Issue 8/WP-D names `DECISION_HYSTERESIS_S≈0.3s` as the constant to test — not implemented yet (no hysteresis exists in this engine; every tick re-decides from scratch) |
| Pit-loss discount under SC/VSC, live | `pit_now_factor` = `SC_PIT_LOSS_FRACTION` or `VSC_PIT_LOSS_FRACTION` when `state.track_status` is 4/6/7 | physical (measured pit-loss fraction) + live flag | engine.py:722–723 | Live | CORE | distinct from the *offline* SC credit (§1.2) — see note below |

**Live decision-stability note.** Nothing in the Task 1 live engine holds a decision between ticks: `_plan_finish` recomputes the best option from scratch every tick with no memory of the previous lap's call (confirmed — no state keyed by "held_since_lap" or a hysteresis band exists anywhere in `engine.py`). Issue 8's "do not generate unstable recommendations where the answer flips every lap" and WP-D's `plan["decision"]` (hysteresis, `changed`, `held_since_lap`) are unbuilt; today's engine can flip on noise alone.

---

## 4. Per-car layer — `src/percar.py`, `TyreModel.for_driver`, `strategy.per_driver_plans`, calibration factors

| Input | What | Entry mode | Value / source | Decision | Class | Ablation |
|---|---|---|---|---|---|---|
| `team_pooled_dev` | this weekend's practice deviation, blended toward the team at `w=n_own/(n_own+20 laps)` | measured this weekend | percar.py:58–121, `POOL_K_LAPS=20` percar.py:46 | Per-car | CORE (exactly "current-weekend Haas evidence") | `no_percar` (existing) |
| `TyreModel.for_driver` | applies `race_factor` (shrunk) and `dev_override` (team-pooled) to one car's rates | current-weekend measured + shrunk historical | tyre.py:341–380 | Per-car | CORE | `no_percar` |
| `shrink_factor` | pulls a historical rate factor toward 1 by `p²/(p²+se²)`, `FACTOR_PRIOR_LN_SD=0.10` | engineering (prior width) applied to a measured factor | percar.py:129–147 | Per-car | mechanism is CORE-adjacent (correctly shrinks a DEPRIORITISED input, see next row) | — |
| **`cal.driver_factors`** — historical, cross-circuit rate factor per driver | a multiplicative factor from **previous races**, pooled/shrunk in `scripts/80_recalibrate.py`, applied via `factor_shrink=None` | historical (LOO across circuits, not this weekend) | calibration.json `driver_factors`: e.g. `OCO 1.115±0.078 ln`, `BEA 1.318±0.077 ln`; **actively consumed** in production at `scripts/10_pipeline.py:699` (`per_driver_plans`, the shipped per-car plan for Ocon/Bearman) and `:682` (counterfactual), and `scripts/40_weekend.py:335` | Per-car (first-stop timing, plan shape) | **This is the prompt's DEPRIORITISE item "historical global driver factors" — and it is currently a live driver of the shipped Haas per-car plans, not a demoted one.** | `no_hist_driver_factors` (proposed — set `factor_shrink=1.0` with `race_factors={}` / pass no `cal.driver_factors`, see §7) |
| `cal.team_factors` | team-pooled version of the above (`Haas F1 Team: 1.21`) | historical, computed | calibration.json `team_factors`; written by `scripts/80_recalibrate.py:526` | **None** — `grep -rn "team_factors"` shows it is only read by `bench/bench_compare.py` for reporting; no strategy or live code path consumes it | not currently a decision input (dead constant, reserved for WP-E) | — |
| `percar_mode` | which per-car variant ships (`team_pooled` in V3/V4) | calibrated selection | calibration.json `percar_mode="team_pooled"`; alternatives `hist`, `none` scored in `percar.rate_scale_table` percar.py:220–252 | Per-car | CORE (mechanism), see driver_factors row for the DEPRIORITISED sub-input it still carries | `no_percar` (existing) |

**What is not yet built at this layer** (Issue 5/WP-E, `src/haascar.py` absent): tyre-age sensitivity, warm-up excess, long-run consistency, traffic sensitivity, push/management response and sector degradation are **not separately estimated per car** anywhere in this codebase — the only per-car quantities that exist today are the practice-deviation blend and the shrunk historical rate factor above. `haascar.explain_difference` (a causal explanation for an Ocon/Bearman divergence) does not exist; nothing today can answer "why does the tool recommend something different for the two cars" beyond "one has a different rate factor/practice deviation."

---

## 5. Duplicates flagged

**V3 density traffic vs live rejoin traffic.** Not a literal duplicate — they are explicitly swapped. In `_plan_prepare`/`_plan_finish` (engine.py:689–981), every option's expected cost is built with the V3 quadratic-density traffic charge baked in (`ctx["xv3"]` excludes it, `ctx["tv3"]` tracks it separately but it lives inside `fixed1`/`fixed2`, engine.py:789,808). When `_race_state_extra` prices the option currently being evaluated (its *own* next stop), `rs_adj = rs["opt_extra"] - ctx["xv3"]` is added (engine.py:859), and `rs["opt_extra"]` already nets out the old density traffic against the new rejoin-projected traffic (`tadj`, racestate section of engine.py:1088–1092). So for **the stop being priced**, the swap is clean. **But** a 2-stop live option's *second* stop keeps the plain density-model traffic charge unconditionally (engine.py:901–907, `traffic*dens[p2-1]`) — there is no rejoin projection for a stop beyond the one currently being decided. This is by design (the race-state term only ever re-prices "the next stop"), but it means the two traffic models coexist within the same option at different stop indices, which is worth stating explicitly rather than leaving implicit.

**Offline, the generic traffic cost and the race-state position term are *not* swapped — they stack.** In `_phase1_race_state` (strategy.py:891–1005), every plan's `fixed` cost always includes the quadratic-density traffic charge (strategy.py:912–915, unconditional), and the race-state `term_by_lap` is added on top of that at the first stop (strategy.py:989, `rs_rows = term[...]`; `full_p = base_p + rs_rows`). The two mechanisms price physically distinct things (dirty-air lap-time loss vs. probability of losing a classified position), so this is not literally double-counting the same seconds, but they are correlated proxies for "what happens when you rejoin near other cars," and nothing in the current code checks that they don't jointly over-price a crowded rejoin. **Recommend an ablation before WP-C's position-aware metric is trusted**: compare `full` against a variant with the generic traffic term zeroed at the first stop only, to see how much of its effect the race-state term already captures.

**Undercut exposure (λ) vs race-state cover.** These price the same underlying phenomenon — a rival's strategic pit response to protect track position — through two different mechanisms that are scoped to different stops by construction, not by measurement: offline, `undercut_lambda` prices every stop *after* the first (`_phase1_race_state` strategy.py:930–936, `range(1, len(seq)-1)`) while the race-state pack's `expected_ahead` cover logic (racestate.py:304–333) prices only the first. Live, the scoping is the same but re-evaluated every tick: `_race_state_extra` covers whichever stop is "next" for the car (engine.py:1047), while a live 2-stop option's second leg still carries `undercut_lambda` unconditionally (engine.py:905–907). Nothing currently verifies that the boundary between the two mechanisms is priced consistently — a car on its second stop gets the *statistical* undercut-exposure treatment for what is, live, its next actual pit decision, while a car on its first stop gets the *rival-specific* cover treatment for the same kind of decision. **Recommend the existing `no_position` ablation (`undercut_lambda=0`) to quantify how much later-stop cost currently comes from λ alone**, before WP-A's hetero field decides whether cover should extend past the first stop.

**First-stop history prior (κ) vs the race-state term.** Correctly *not* a duplicate: `kappa_used = 0.0 if use_rs else cal.first_stop_kappa_s` (10_pipeline.py:603) and the live engine's `first_stint` gating (engine.py:756–770) enforce mutual exclusion. The residual problem is Issue 4's, not a duplicate-pricing one: `strategy.counterfactual` (10_pipeline.py:684, kappa = `cal.first_stop_kappa_s`, **no race-state term at all**), `src/outlook.py` and `app/desk_tab.py::evaluate_plans` still price the V3 objective side by side with the race-state-aware search — see `docs/v4_plan.md` §0 for the full list of affected call sites; WP-H does not re-derive that list, it is already correctly identified there.

---

## 6. Arbitrary constants flagged (no cited numeric derivation)

| Constant | Value | File:line | Why flagged |
|---|---|---|---|
| `CLIFF_SHARPNESS`, `CLIFF_EXPONENT` | 8.0, 2.0 | config.py:350–351 | Chosen to hit a qualitative target shape (comment states the *resulting* loss ratios at wear 1.25/1.5/1.75); no fit or measurement produces 8.0/2.0 specifically |
| `MANAGE_WEAR_EXPONENT` | 1.7 | config.py:373 | Justified only by a downstream cross-check (reproduces a 0.57 measured regime ratio); the exponent itself is not independently fitted |
| `GRID_START_PENALTY_S` | 1.0 (default), **0.0 calibrated in all 7 folds** | config.py:437; calibration.json | File itself calls this "a calibrated preference, not a measurement" (config.py:434–436); now measured to be zero everywhere — see §8 |
| `MAX_HISTORY_SCALE` | 3.0 | history.py:76 | A safety cap on how far circuit history may rescale a practice rate; no cited sensitivity for why 3× and not 2× or 5× |
| `FIELD_TEMPER` | 0.5 | engine.py:87 | "Other cars' laps count half: they manage their tyres differently" — plausible but no fit behind 0.5 specifically |
| `DIRTY_SIGMA_MULT` | 1.6 | engine.py:85 | Same pattern: directionally justified, magnitude not derived |
| `SLOW_LAP_MARGIN_S` | 5.0 (config.py:96) / 2.5 (regime.py:89) / 3.0 (engine.py:91) | three files | Same concept (a lap too slow to be a racing lap), three different uncoordinated thresholds |
| `N_RIVALS` | 4 | racestate.py:90, engine.py (mirrored `UNDERCUT_RANGE_S`) | Issue 8 explicitly asks for a k=3–6 sweep; not run in this codebase, so 4 is currently asserted rather than chosen |
| `RELEVANT_RANGE_S`/`UNDERCUT_RANGE_S` | 6.0 s | racestate.py:91, engine.py:95 | Internally consistent with each other but no cited measurement for why 6 s is the right range |
| `regime_multipliers`/`simulate_model(regime=...)` | — | strategy.py:386–398 | Dead for pricing (§2.1) — an input that looks load-bearing from the signature but is not; the closest thing to a pure "arbitrary/unused constant" in the objective |

---

## 7. DEPRIORITIZE-list items: current status

| Item | Status in this codebase |
|---|---|
| Humidity | **Absent** (§2.6). Compliant. |
| Wind | **Absent** (§2.6). Compliant. |
| Generic straight-line speed | **Absent** — no speed-trap or straight-line feature exists anywhere in `src/`. Compliant. |
| **Historical global driver factors** | **Present and active** in the shipped per-car plans and counterfactual (§4, `cal.driver_factors`). This is the one DEPRIORITIZE item that is currently a real driver of a production decision, not merely latent risk. |
| Excessive apex features | **Present but diagnostic-only**, capped at 4 corners, not in the production posterior (§2.5). Compliant with the prompt's caution by construction. |
| Generic temperature corrections | **Two distinct, non-generic mechanisms**, each gated on a real measurement (this weekend's practice temperature, or an explicit race-day forecast) rather than a blanket constant (§2.1). Borderline only in the sense that circuit_prior's thermal multiplier is *not* gated on a forecast (it uses practice temperature, which is always available once practice runs) — recommend `no_temperature_mode` before concluding this is compliant rather than merely well-intentioned. |
| Duplicate traffic features | **Confirmed, characterised in §5.** Live is a clean swap; offline the two mechanisms stack without a check. |
| Arbitrary historical bonuses | `GRID_START_PENALTY_S` is the clearest instance — self-described as a preference, calibrated to zero (§6). |
| Arbitrary constants | Catalogued in §6. |

---

## 8. Ranked recommendations

**Demote or remove now (evidence already in hand):**
1. `GRID_START_PENALTY_S` — calibrated to 0.0 in all seven LOO folds (calibration.json); the mechanism is inert in every production path. Either remove the term or leave it at the calibrated 0.0 and delete the config-default rationale that no longer describes what ships.
2. `strategy.regime_multipliers` / `simulate_model`'s `regime=` kwarg for pricing — confirmed dead code for the objective (§2.1, §6). Either wire it to something (there is nothing left for it to scale, since the push/wear trade-off replaced the exogenous regime factor per `src/tyre.py`'s own docstring) or delete it; keeping an unused parameter with this name invites a future contributor to believe the regime factor prices the search when it does not.
3. `cal.driver_factors` as currently applied to `per_driver_plans`/`counterfactual` — this is squarely Issue 5's target and the DEPRIORITIZE list's "historical global driver factors." Do not remove outright (it is currently the *only* per-car timing signal besides practice deviation), but it should be superseded by WP-E's hierarchical Haas model, and its marginal contribution should be measured first (see below) rather than assumed harmful.

**Keep as physically necessary even if weak on seven weekends:**
1. Grip budget / cliff mechanism (`src/tyre.py`) — derived from 2026 regulation physics plus a measured budget; the whole point of the module is that a stint's length is *derived*, not fitted, so it should not be judged by whether it moves the benchmark on seven races.
2. Fuel/mass load profile (`src/fuel.py`, `TYRE_LOAD_EXPONENT`) — regulation-derived, makes stint order matter at all; removing it returns the model to the pre-V2 defect (order-invariant plans).
3. Track evolution correction (`src/evolution.py`) — physically necessary to avoid biasing degradation low at low-wear circuits; already cross-checked against a long-run backfit.
4. Measured pit loss, SC credit, per-circuit dirty air — all measured, all real effects with documented sensitivity (dirty air alone spans 4× across circuits).
5. Circuit-history stint caps/cliff (`history.CircuitPrior.cliff`) — a feasibility fact about the circuit, not a soft prior; this is what actually bounds a runaway extrapolation today, pending WP-B's uncertainty-based mechanism.

**Needs an ablation before a decision:**
1. **Symmetric rival pack** (`racestate.pack_equilibrium`) — Issue 1's central concern and currently the *only* rival model that exists. Run `symmetric_pack` against WP-A's hetero field once built; until then, every first-stop recommendation in this codebase rests on an assumption the prompt itself calls unrealistic.
2. **Undercut λ vs race-state cover boundary** (§5) — run `no_position` (λ=0) with race-state on to see how much of "later-stop" cost currently comes from the generic exposure term alone, before deciding whether cover should extend past the first stop.
3. **Offline traffic-density stacking with the position term** (§5) — no existing ablation isolates this; propose zeroing the density-traffic term at the first stop only (holding it elsewhere) and compare against `full`.
4. **Thermal/temperature mechanism** (§2.1, §7) — run the proposed `no_temperature_mode` (switch off both `circuit_prior`'s thermal multiplier and `regime_prior`'s thermal correction) to quantify real effect before deciding this stays SUPPORTING rather than sliding toward "generic temperature correction."
5. **`cal.driver_factors`** (§4, §8-demote) — run the proposed `no_hist_driver_factors` (per-car plans with `factor_shrink=1.0` and no historical factor, practice deviation only) against the current shipped per-car plans, to measure what the historical factor is actually buying before WP-E replaces it.
6. **Rival rate levels / family logit** (WP-A's `rate_levels`, `family_temper_s`) — not built yet; `rival_rate_levels_1` and `no_family_logit` are prospective ablations for when WP-A ships, not actionable today.
7. **Apex** — already effectively ablated by `bench/bench_apex.py` (joint vs lap-only fit, scored against the race); no further action needed, listed here only for completeness against the prompt's DEPRIORITIZE item.
