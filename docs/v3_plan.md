# Degless V3 — implementation plan

*Planned 2026-09-12 (Fable). Source of truth for the engineering agents. The specification is the user's V3 prompt plus `results_updated.md` (V2). V1 is frozen under `bench/baseline/`, V2 under `bench/v2/` (frozen 2026-09-12T12:27Z, see `bench/v2/FROZEN.txt`). Nothing under `bench/baseline/` or `bench/v2/` may be modified. `results_updated.md/.pdf` and `results.md/.pdf` are read-only.*

## 0. Facts the plan rests on (verified)

* Tests at the start of V3: **30/30 pass in 30 s** (`.venv/bin/python -m pytest tests -q`); V2's report said 27, three were added by later commits. V3 must keep all 30 and add its own.
* Hardware: Apple M4, 17 GB, Python 3.13.5, JAX on CPU; headless Chrome at `/Applications/Google Chrome.app` for the PDF.
* The FastF1 cache holds **race sessions only** for 2023–2025 (18–21 MB/year) and full 2026 weekends. Historical *practice* sessions must be fetched over the network once (a 2025 FP2 loads in ~17 s) and then summarised to JSON so `make history` never needs the network again.
* The 2026 lap tables carry a `team` column; the posterior carries `dev` (draws × drivers × compounds), the per-driver slope deviation.
* Pirelli nominations (hard/medium/soft), verified on press.pirelli.com / formula1.com:

| Circuit | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| Melbourne | C2/C3/C4 | C3/C4/C5 | C3/C4/C5 (wet, excluded) | C3/C4/C5 |
| Suzuka | C1/C2/C3 | C1/C2/C3 | C1/C2/C3 | C1/C2/C3 |
| Barcelona | C1/C2/C3 | C1/C2/C3 (rain-flagged, excluded) | C1/C2/C3 | **C2/C3/C4** |
| Spielberg | C3/C4/C5 (wet) | C3/C4/C5 (wet) | C3/C4/C5 | C3/C4/C5 |
| Spa-Francorchamps | C2/C3/C4 (wet) | C2/C3/C4 | C1/C3/C4 (wet) | C2/C3/C4 |
| Budapest | C3/C4/C5 | C3/C4/C5 | C3/C4/C5 | C3/C4/C5 |
| Monza | C3/C4/C5 | C3/C4/C5 | C3/C4/C5 | C3/C4/C5 |
| Other 2026 | Shanghai C2/C3/C4; Miami, Montreal, Monte Carlo C3/C4/C5; Zandvoort, Madring C2/C3/C4; Silverstone unknown | | | |

  So the C-number mapping changes only **Barcelona (2023, 2025: one step harder than 2026)** and **Melbourne 2023 (one step harder than 2026)**; every other used history year has the 2026 nomination. Barcelona's mapped history: 2023/2025 "SOFT" = C3 = 2026 MEDIUM; S-M-M/S-H-M/S-H-H/S-M-H → M-H-H (10 of 37 finishers, the modal 2-stop family), S-M-S → M-H-M, S-M-S-S → M-H-M-M; 31 of 37 start on 2026's MEDIUM. The 2026 Barcelona field: M-H-H 6/13 (46 %), MEDIUM start 8/13.
* Historical green-flag opening stints, scaled to the 2026 distance, by C-number (from the cached races): Barcelona C3 median 13.5 (2023, n=16) / 16.5 (2025, n=16) against the 2026 field's 13 on the same C3; Budapest C4 16.5 / 16 / 20.5 vs 2026's 19; Spielberg 2025 C4 24.3 (overall 20.3) vs 2026's 18; Monza C4 19.7 / 14.5 / 31.5. The between-year spread is ±4 laps: the first-stop prior must be wide and its weight calibrated leave-one-out; a ≤2-lap mean error is the target, not a promise.
* 2026 field green-flag first-stop medians (from `meta_*.json` backtests): Barcelona 13 (p25 12, p75 14), Austria 18 (17.75–19.25), Hungary 19 (16–22); Australia, Japan, Belgium, Italy are safety-car set and excluded from timing metrics, exactly as in V2.
* V2 leave-one-out calibration: λ 0.1–0.3, τ 2.5–6, dirty air pooled 0.12 (per-weekend measurements +0.25 AUS, −0.09 JPN, +0.15 BCN, +0.12 AUT, +0.01 BEL, +0.43 HUN, −0.20 ITA), budgets 3.80/3.83/3.84 (HARD 3.27 with Hungary held out).

## 1. Work packages, ownership, order

Files are owned by exactly one work package per phase so agents can run in parallel without merge conflicts.

### Phase 1 (parallel, independent files)

**WP-A — nominations, plan prior through C-numbers, history summary v3, Melbourne ladder rule.** Files: `src/nominations.py` (new), `src/history.py`, `src/compounds.py` (only `pace_step_prior` and `_ladder`-related pooling), `data/raw/pirelli_nominations.json` (new).

**WP-B — regime factor: forecast-only temperature + circuit-level practice→race prior; stint-FE baseline gate.** Files: `src/regime.py`, `src/regime_history.py` (new), `scripts/05_history_practice.py` (new), `src/model_fallback.py`.

**WP-C — within-stint cliff detector + censoring-aware grip budgets; first-stop prior density; team-pooled per-car model.** Files: `src/cliff.py` (new), `src/firststop.py` (new), `src/tyre.py`, `src/percar.py` (new).

### Phase 2 (sequential, one agent)

**WP-D — wiring.** Files: `src/strategy.py`, `src/calibration.py`, `scripts/80_recalibrate.py`, `scripts/10_pipeline.py`, `scripts/40_weekend.py`, `src/outlook.py`, `src/live/engine.py`, `src/config.py`, `Makefile`, `src/replay.py`.

### Phase 3 (parallel)

**WP-E — benchmark suite.** Files: `bench/*.py`, `bench/run_all.sh`. **WP-F — regression tests.** Files: `tests/test_v3.py` (new), existing tests only if a signature changed.

### Phase 4 — run `make history && make benchmark` from scratch; ablations; fix. Phase 5 — `results_v3.md` / `results_v3.pdf`; quality gate.

## 2. WP-A specification

### 2.1 `src/nominations.py` + `data/raw/pirelli_nominations.json`

* JSON: `{"2023": {"Melbourne": ["C2","C3","C4"], ...}, "2024": {...}, "2025": {...}, "2026": {...}}` keyed by **circuit name as used in `src/config.py` `_CAL`** (Melbourne, Shanghai, Suzuka, Miami, Montreal, Monte Carlo, Barcelona, Spielberg, Silverstone, Spa-Francorchamps, Budapest, Zandvoort, Monza, Madring, Baku, Kuala Lumpur, Singapore, Austin, Mexico City, Interlagos, Las Vegas, Lusail, Yas Marina Circuit, plus Sakhir, Jeddah, Imola for 2023–25). Order is hard → soft. Every entry carries the verified values from §0; entries the planner could not verify (2026 Silverstone; 2026 rounds after Madring) are **absent**, not guessed. Include a `"_sources"` note.
* API: `nomination(year, circuit) -> list[str] | None`; `map_sequence(seq_letters_or_names, from_nom, to_nom) -> tuple[list[str], dict]` returns the target-letter sequence plus `{"clamped": bool, "shift": int, "shared": int}`; a compound harder than the target's hardest maps to the target HARD, softer than the softest to SOFT (clamped); `comparable(from_nom, to_nom) -> str` in `{"identical", "shifted", "disjoint", "unknown"}` (shifted = at least two shared C-numbers).
* Rules for the plan prior (`history.plan_prior_for`):
  1. identical nomination → literal letters (unchanged behaviour);
  2. shifted → map every stint by C-number; a mapped sequence that collapses to a single compound (e.g. 2023 Melbourne M-H = C3-C2 → H-H) is **not** a legal 2026 plan: drop it from the sequence counts, keep it in the stop-count marginal, and keep its *mapped* start compound in the start marginal; clamped plans keep full weight;
  3. disjoint or unknown nomination → that race contributes nothing to the sequence/start counts (stop-count marginal only) and the prior falls back, in order, to the other years of the circuit, then the 2026 season pool (itself mapped by C-number into the target nomination), then role letters with `"comparable": "role-fallback"` in the source string;
  4. smoothing/back-off (`strategy.plan_prior_penalty`, α = 2) unchanged;
  5. never read the target's own 2026 race (the 2026 season pool excludes the target event, as `plan_prior_for` already does).
* `plan_prior_for(cp, *, season_fallback=True, use_nominations=True, target_nomination=None)` — `use_nominations=False` reproduces V2 exactly (the ablation needs it); the returned dict gains `"nomination": {"target": [...], "per_race": [{"year", "nomination", "comparable", "n_mapped", "n_dropped", "n_clamped"}], "mode": "c-number"|"letters"}`.

### 2.2 `history.summarise_race` → `SUMMARY_VERSION = 3`

Add, per race (keep every existing key):
* `"first_stints": [{"driver", "compound", "laps", "in_lap", "sc"}]` for classified finishers (compound = letter of that year; the C-number is derived from the nomination table at read time so the JSON stays nomination-agnostic);
* `"first_stop_green": {"median_lap", "p25", "p75", "n", "in_laps": [...]}` over green-flag first stops only;
* `"dirty_air": measure_dirty_air(...)` on the canonical frame — `_canonical` must add `lap_start_s` (from `LapStartTime`) and the gap to the car ahead (reuse `src.laps._gap_to_car_ahead` on the whole race), and a `tyre_age` column (= `tyre_life`); the estimator is `src.compounds.measure_dirty_air` unchanged; store `{s_per_lap, se, n_laps, share_close}` or `{}`;
* `"nomination": [...]` from the table (or null).

Bumping the version recomputes the ~24 cached summaries used by the 7 weekends on first use (offline from the FastF1 cache; `summarise_race` must call `fastf1.Cache.offline_mode(True)` only when the caller asked for it — do not change the network behaviour of the function itself). Run `.venv/bin/python -c "from src.history import circuit_prior; [circuit_prior(k, probe_practice_temp=False) for k in ['australia-2026','japan-2026','barcelona-2026','austria-2026','belgium-2026','hungary-2026','italy-2026']]"` once at the end of WP-A so the cache is warm.

### 2.3 `CircuitPrior` additions (populated in `circuit_prior`)

* `first_stop_green: dict` — pooled over years: `{"median_lap", "p25", "p75", "n"}` (scaled to the event's distance) and `"in_laps": [...]` (scaled), `"by_compound": {target_letter: [scaled in-laps]}` where the compound is mapped through the nominations (WP-C's `src/firststop.py` consumes this);
* `dirty_air: dict` — precision-weighted mean of the years' `dirty_air.s_per_lap` with `se`, `n_races`, `by_year`; **not clipped to be positive**;
* `nomination: list | None` for the target year and `nominations_by_year`.

### 2.4 Melbourne ladder rule

In `circuit_prior` (ladder pooling) and `compounds.pace_step_prior` (net-step pooling): weight by recency (2025: 1.0, 2024: 0.7, 2023: 0.5) and **drop any pre-2026 net-step measurement whose standard error exceeds 0.15 s** (`HIST_NET_SE_MAX_S = 0.15`); keep the existing `HIST_NET_SE_FLOOR_S`. Report what was dropped in `pace_step_prior()["detail"]` (`"used": false, "why": ...`). Check with `pace_step_prior("australia-2026", circuit=circuit_prior("australia-2026"))` that the pooled net is no longer negative; if it still is, say so in the WP report — do not tune further.

### 2.5 Acceptance (WP-A)

`plan_prior_for(circuit_prior("barcelona-2026"))["sequences"]` has `M-H-H` as its modal 2-stop family and `starts["MEDIUM"] > starts["SOFT"]`; with `use_nominations=False` it equals V2's `{'S-M-S-S': 8, 'S-H-M': 4, ...}`. Japan/Austria/Belgium/Hungary/Italy priors are byte-identical to V2 (identical nominations). The existing 30 tests pass.

## 3. WP-B specification

### 3.1 `regime_prior` — forecast in, archive out

* New signature: `regime_prior(target, *, donors=None, race_temp_c=None, practice_temp_c=None, temperature_model="auto", clean=None, circuit_prior_weight=1.0, use_circuit_history=True)`. `"auto"` = apply the thermal correction **only when `race_temp_c` is supplied** (the forecast); otherwise the donors' raw log ratios are pooled with the median and no temperature enters. `temperature_model=True` reproduces V2 (archive race-day mean) for the ablation; `False` = plain median pooling.
* `RegimeFactor.temperature["mode"]` ∈ `{"forecast", "none", "archive"}`; `derivation` says which.

### 3.2 Circuit-level practice→race regime prior (`src/regime_history.py`)

* `measure_regime_history(year, circuit, *, sessions=("Practice 1","Practice 2","Practice 3"), offline=False) -> dict | None`, cached as `data/processed/history/regime_<year>_<circuit>.json`: loads the historical practice sessions and the race from FastF1, builds the canonical frames (`ingest`-style columns, `build_lap_table` with `event="<year>-<circuit>"` set on the frame, `clean_laps`), applies the **pre-2026 fuel physics** (`history.FUEL_S_PER_LAP_PRE2026`) and the evolution correction (`fit_evolution_auto` with `laps_all`), and measures the same stint-fixed-effects ratio `regime.measure_regime` measures (factor `_race_frame`/`_practice_frame`/`_stint_fe_slope` so they accept `fuel_s_per_lap` and `n_race_laps` explicitly; the 2026 path must be unchanged). Output: `{year, circuit, ratio, per_compound, n_race_stints, n_practice_stints, usable_compounds, sessions_used, rain, version}`; a rain-flagged practice or race → `{"missing": true, ...}` like `summarise_race`.
* `circuit_regime_prior(event) -> dict | None`: pooled over available years: `{"ratio": exp(median log ratio), "ln_sd": sqrt(max(spread, 0.20)^2 + 0.15^2), "years": [...], "by_year": {...}}` (0.15 is the tyre/car-generation term; document it as an assumption). Ratios outside `REGIME_RATIO_BAND` are discarded exactly as donors are.
* `regime_prior` combines the donor pool (median residual, spread → `ln_sd_pool`) and the circuit prior with precision weights on the log scale; `temperature["circuit_prior"]` records what entered; `label` says `"... + circuit history"`. With `use_circuit_history=False` the V3 result equals the donor pool alone (ablation).
* `scripts/05_history_practice.py --events <keys> [--offline]`: fetches and caches the summaries for the circuits of the given events for `history.YEARS` (default: the 7 benchmark weekends + spain-2026), printing one line per race. Network use is expected here and nowhere else. Run it at the end of WP-B for the 7 weekends (`--events australia-2026 japan-2026 barcelona-2026 austria-2026 belgium-2026 hungary-2026 italy-2026`) and report which circuit-years were obtained; failures are acceptable if reported, and `circuit_regime_prior` must return `None` cleanly for a circuit without any.

### 3.3 Stint-fixed-effects baseline gate (`src/model_fallback.py`)

* `stint_fe_baseline(clean) -> dict` — per-compound slope with stint fixed effects on `lap_time_corr` (fuel- **and** evolution-corrected clean practice laps), with a stint block bootstrap (`n_boot` param, default 50) for a 90 % band; returns `{slopes, ci, n_stints, pooled_slope}` with the laps-weighted pooled slope.
* The MixedLM fit stays (the accuracy benchmark reports it) but is no longer gated on a range; WP-D replaces the `"MixedLM MEDIUM slope in 0.12-0.35"` gate with two: `"stint-FE baseline pooled slope is finite and >= 0"` and `"Bayes pooled slope within 0.06 s/lap of the stint-FE baseline"`.

### 3.4 Acceptance (WP-B)

`regime_prior("belgium-2026")` (no forecast) reports `mode == "none"`, `regime_prior("belgium-2026", race_temp_c=31.0)` reports `mode == "forecast"`, and `regime_prior(..., temperature_model=True)` reproduces V2's 0.80 at Belgium to 3 decimals (the fitstage JSON carries V2's numbers). `circuit_regime_prior` returns a dict for at least four of the seven circuits or the WP report explains which loads failed. Existing tests pass.

## 4. WP-C specification

### 4.1 `src/cliff.py` — within-stint cliff detector and censoring-aware grip budget

* `detect_stint_collapse(stint: pd.DataFrame, *, fuel_s_per_lap, evo=None, min_laps=8) -> dict` on one race stint's green, non-pit, accurate laps (`lap_number`, `tyre_age`, `lap_time_s`; `evo` an optional per-lap evolution series to subtract): fuel-corrected lap time vs age; fit a linear trend on the first max(5, n−4) laps and a two-segment (hinge) model with the knee scanned on the integer grid `[4, n−2]`; report `{n, slope_pre, slope_post, knee_age, delta_last3_s (mean residual of the last 3 laps vs the pre-knee trend), collapse: bool, kind: "collapse"|"strategic"|"undetermined", cum_loss_at_knee_s, cum_loss_at_end_s}`. **Collapse** requires all of: hinge model reduces RSS by ≥ 25 % vs linear, `slope_post − slope_pre ≥ 0.15 s/lap`, `delta_last3_s ≥ 0.6 s`, and the stint ended within 3 laps of the knee. **Strategic** = the stint ended with the last three laps within ±0.4 s of the trend (no collapse, pitted on-trend). Otherwise undetermined. A stint that ends because the race ends is `censored_by_flag` (no collapse call).
* `race_collapses(race_laps, event_or_params) -> pd.DataFrame` one row per stint (`driver, compound, stint, n, kind, knee_age, collapse, cum_loss_at_knee_s, cum_loss_at_end_s, rate_s_per_lap`) using `regime.race_track_evolution` for the evolution term (needs the frame the regime module builds; reuse `regime._race_frame` if practical).
* `grip_budget_estimate(rows_by_weekend, *, prior=GRIP_BUDGET_S, prior_sd=0.6, band=(2.5, 5.0)) -> dict` per compound: a **censored** estimate — collapse stints contribute `cum_loss_at_knee_s` as observations, strategic/undetermined stints contribute `cum_loss_at_end_s` as lower bounds (right-censored); maximise the log-normal likelihood `Σ log φ(obs) + Σ log(1 − Φ(bound))` plus the prior, on a 1-D grid; return `{budget_s, n_obs, n_censored, se_s, source}`. With no collapse observation it degenerates to "largest lower bound, else prior" (V2 behaviour).
* `budget_ratio_metrics(...)`: helper for the benchmark — predicted collapse lap per compound (budget / race rate) vs observed knee, over/under counts.

### 4.2 `src/firststop.py` — the first-stop prior density

* `first_stop_prior(cp_first_stop_green: dict, n_race_laps: int, *, start_compound=None, bandwidth_laps=2.5, floor=0.05, min_compound_n=5) -> dict`: a Gaussian KDE (bandwidth scaled by `n_race_laps/60`) over the circuit's green-flag first-stop in-laps (already scaled), mixed with the start-compound-specific KDE with weight `n_c/(n_c + min_compound_n)` when `start_compound` is given, then mixed with a uniform over `[margin, n_race_laps − margin]` at weight `floor`. Returns `{"laps": np.arange(1, n+1), "neglogp": array (min 0), "mode_lap", "median_lap", "p25", "p75", "n", "n_compound", "source"}`.
* `first_stop_penalty_table(cp_first_stop_green, n_race_laps, compounds, **kw) -> dict[compound, np.ndarray]` — per start compound a `(n+1,)` array indexed by the first-stop lap (index 0 unused), which WP-D adds as `kappa * table[seq[0]][pits[:, 0]]` in `simulate_model` phase 1, phase 2, `pit_window_model`, `evaluate_plans`, `counterfactual` (first stop only, green-flag) and the live engine.
* Empty/absent history → `None` (no penalty).

### 4.3 `src/percar.py` + `src/tyre.py` — per-car intelligence from the weekend

* `team_pooled_dev(fit_or_model_dev: dict, teams: dict[driver, team], *, shrink_laps=None) -> dict`: per driver and compound, the practice deviation pooled at team level: `dev_team = mean over the team's drivers`, driver value = precision-weighted blend of own and team deviation (own weight `n_own/(n_own + k)`, `k` = 20 clean laps by default; when the fit's `dev` posterior sd is available use its precision instead), returned as `{driver: {compound: (draws,) array}}`.
* `TyreModel.for_driver(driver, *, race_factor=1.0, practice_dev=True, dev_override=None, factor_ln_shrink=None)` — `dev_override` replaces `self.driver_dev[driver]`; `race_factor` may now be a dict `{"factor", "ln_sd"}` and is shrunk toward 1 by `factor_ln_shrink` (multiplying its log by `k/(k+ln_sd²)`... keep simple: `log f · (1 − w)` with `w` given). Backward compatible for existing callers.
* `percar.rate_scale_table(kind, fit, cal, teams) -> dict[driver, float]` for the accuracy benchmark's Spearman variants: `kind ∈ {"none", "hist" (V2 LOO factors), "practice_dev", "team_pooled", "combined"}`.

### 4.4 Acceptance (WP-C)

Unit checks in the WP report: a synthetic stint with a linear trend of 0.1 s/lap and a collapse of +0.6 s/lap from lap 15 that ends at lap 17 → `collapse=True, knee_age≈15`; the same stint truncated at lap 12 → `strategic`; `grip_budget_estimate` with three lower bounds (1.5, 2.0, 2.2) and no observation → the largest bound or the prior per V2 rule; with observations (3.6, 3.9) and bound 2.0 → ≈3.75. `first_stop_prior` on Barcelona's history has its mode between 12 and 17. Existing tests pass.

## 5. WP-D specification (wiring)

1. **Objective** (`strategy.simulate_model`, `pit_window_model`, `evaluate_plans`, `counterfactual`, `search_with_pace_calibration`, `per_driver_plans`): new keyword `first_stop_prior: dict[compound, ndarray] | None` and `first_stop_kappa_s: float` (κ, seconds per nat). Phase 1: `full = t + lam*pos + prior_pen + kappa * fsp[seq[0]][pits[:, 0]]` (zero for plans with no stop); phase 2 fixed term likewise; `StrategyResult` gains `first_stop_kappa_s`, `best["first_stop_s"]` (the penalty the chosen plan carries) and the table a `first_stop_s` column; `tyre_optimal` stays κ = 0, λ = 0, τ = 0. Pit window: same term on the first stop. Counterfactual: the term on the driver's actual first stop and on the alternatives, green-flag only (a safety-car first stop is held fixed anyway). `evaluate_plans` and `sc_playbook`: the term on the first stop of the plan (`sc_playbook` prices only the decision now; leave it unchanged).
2. **Calibration** (`src/calibration.py`): fields `first_stop_kappa_s` (default `config.FIRST_STOP_KAPPA_S = 1.0`), `dirty_air_by_circuit: dict` (circuit → s/lap; `dirty_air_for(circuit)` returns it or the pooled value), `grip_budget_detail`, `driver_factor_ln_sd: dict`, `percar_mode: str` (`"team_pooled"` shipped; `"hist"` = V2). `_KEYS` extended; missing keys fall back to defaults so a V2 `calibration.json` still loads.
3. **`scripts/80_recalibrate.py`**: (a) budgets via `cliff.grip_budget_estimate` over the donors' race stints (`cliff.race_collapses`), leave-one-out as now, with the V2 estimator kept as `"budget_v2"` in `raw` for the report; (b) dirty air per circuit from `CircuitPrior.dirty_air` (historical races only — never the target's 2026 race) with the pooled 2026 donor median as the fallback and a global `dirty_air_by_circuit` map for every circuit that has one; the sweeps use each donor's own circuit value; (c) κ sweep `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` on the first-stop objective, iterated with λ and τ as now (order: manage → λ → κ → τ → grid, twice); widen λ's grid to `[..., 0.45, 0.6]` and τ's to `[..., 6, 8]` (V2 item 12); (d) driver factors: also emit `driver_factor_ln_sd` and pool per team (`team_factors`); (e) per-weekend block records the circuit's dirty air, the first-stop prior summary (mode/median) and the collapse counts. `--quick` keeps coarse grids. Keep runtime under ~10 min.
4. **`scripts/10_pipeline.py`** fit stage: `regime_prior(ev, clean=clean)` (auto mode → no temperature, circuit prior on), gate rename `"regime factor is plausible"` unchanged; replace the MixedLM range gate as in §3.3; write `first_stop_prior` summary, `plan_prior` (mapped), `circuit_history.dirty_air`, `nomination` and `regime.temperature.mode` into `fitstage_*.json`. Decide stage: build the penalty tables (`firststop.first_stop_penalty_table` from `cp.first_stop_green`), pass κ, per-circuit dirty air, per-car mode; `per_driver_plans` with `percar` (team-pooled dev + shrunk hist factors); record `cliff` collapse rows (`cliff.race_collapses` on the race) in `meta["cliff_detector"]` and the per-compound predicted collapse lap (budget/race rate at the plan's push) vs observed; keep every existing meta key. Add `--offline` to the Makefile `history` target (`OFFLINE ?= --offline`).
5. **`scripts/40_weekend.py`, `src/outlook.py`**: same objective terms (`--race-temp` is the forecast path; the outlook passes the live weather track temperature as the *practice* temperature only — never as a race forecast); `sim_kwargs` carries the first-stop tables and κ.
6. **`src/live/engine.py`**: (a) `WeekendModel` carries `first_stop_kappa_s` and the penalty tables (from meta `circuit_history.first_stop_green`); in `_plan`, for a car still in its **first** stint, add `kappa * table[c_now][P1]` to every option whose first stop is at lap P1 (kinds 1 and 2), nothing for later stints; (b) **retire the wear-based cliff alarm**: `cliff_alarm` becomes `pace_collapse` — the within-stint detector (`cliff.detect_stint_collapse`) on the car's own fuel/evolution-corrected clean laps this stint (min 6 laps), and the alert text says "PACE COLLAPSE" with the measured post-knee slope; keep `p_past_cliff` and `laps_to_cliff_*` as displayed quantities; add a `box_now` alert (level "warn") when `delta_box_now_s <= 0.5` and `best_kind > 0`, cooldown 3 laps; `bench_live` counts window, box-now and collapse signals separately; (c) per-car practice deviation (team-pooled) enters the car's likelihood and continuation cost (`_continue_cost`) while the fresh-stint tables stay field-level (document the approximation). `src/replay.py`'s `cliff_alarm` should mirror the new definition or be renamed consistently (the app reads `cliff_alarm`; keep the key, change the meaning, update the two app captions in `app/live_tab.py` from "CLIFF" to "COLLAPSE" — the only app edits allowed).
7. **Smoke test before handing over**: `scripts/10_pipeline.py --event hungary-2026 --stage fit --no-diagnostics --offline --boot 20`, then `scripts/80_recalibrate.py --quick --events hungary-2026 barcelona-2026 --out /tmp/cal_smoke.json`, then `--stage decide` for hungary-2026 with the smoke calibration copied into place is **not** allowed (it would overwrite the real calibration); instead run decide with the current `calibration.json` and confirm it runs end to end. Then run the full `make history` (all seven, `--offline`) and report the per-weekend first-stop error, sequence share and rate MAE from the meta files. All 30 tests must pass.

## 6. WP-E — benchmark suite

* Three-way comparison: `bench/common.py` gains `V2 = ROOT/"bench"/"v2"`, `v2_meta(key)`, `v2_out(name)`; every `baseline_*` hook keeps meaning V1. `bench_compare.py` writes the V1 → V2 → V3 tables and figures (`fig1` degradation MAE V1/V2/V3, `fig2` first-stop error tyre-optimal/V2/V3-no-prior/V3, `fig3` strategy-match counts, `fig4` life ratio, `fig5` live latency and stop calls, `fig6` calibration sweeps incl. κ) and a `verdict` column per metric ∈ {improved, unchanged, regressed, not meaningfully changed} with the rule: changed by less than the smaller of 5 % or one weekend's worth → "not meaningfully changed".
* `bench_accuracy.py` variants: keep V2's; add `sealed_v3` (the shipped V3 curve), `regime_v2_temperature` (archive-temperature regime, V2 style), `regime_v3_pooled` (donor median, no circuit prior), `regime_v3_circuit` (shipped: donor median + circuit prior), `regime_oracle_temperature` (actual race temperature as the forecast), `practice_no_regime`; each re-applies `apply_circuit_prior` on the practice posterior with that regime so the fold-in is consistent; per-car variants `sealed_driver_hist`, `sealed_driver_practice_dev`, `sealed_driver_team_pooled` (Spearman). Report pooled MAE, max, bias, 90/95 % coverage, width.
* `bench_strategy.py`: add the first-stop block for V3 (`rec`, `tyre_optimal`, V2's rec from `bench/v2`, and "V3 without the prior" from the ablation), the share of first stops inside the window, cliff-detector metrics (predicted vs observed collapse lap per compound, over/under counts, tyre-life ratio), per-car metrics (three variants).
* `bench_ablation.py` variants (each a full search on the shipped posterior with one thing off): `full`, `no_first_stop_prior` (κ=0), `no_plan_prior` (τ=0), `no_nomination_mapping` (plan prior with `use_nominations=False`), `no_circuit_regime_prior` (regime = donor pool only; re-fold the practice posterior), `no_dirty_air_circuit` (pooled dirty air), `no_cliff_budgets` (V2 budgets from `bench/v2/processed/calibration.json`), `no_percar` (per-car plans with no driver terms) plus V2's `no_position`, `config_constants`, `practice_only`. Metrics: sequence run by anyone, start match, stops match, field share, first-stop error (3 non-SC weekends), tyre-life ratio.
* `bench_live.py`: add `box_now` (≤ 1 s the lap before) precision/recall and `collapse` precision/recall against real stops (a signal within 3 laps before a stop = true positive; a signal with no stop within 3 laps = false positive), alongside the window metrics; keep tick latency.
* `bench/run_all.sh` unchanged in order; every stage must run offline.

## 7. WP-F — tests (`tests/test_v3.py`)

1. `first_stop_prior` for a scored weekend is built from `circuit_history.first_stop_green` (years < 2026 only) and `calibration.loo[key]` never lists the key among its donors — the held-out race is not in its own prior.
2. `nominations.map_sequence(["S","M","S"], C1C2C3 → C2C3C4) == ["M","H","M"]`; identical nominations map to themselves; a clamp is flagged.
3. Barcelona's mapped plan prior has a MEDIUM-start majority and `M-H-H` as its modal 2-stop family; with `use_nominations=False` the SOFT start dominates (the V2 defect, pinned).
4. `regime_prior` without a forecast has `temperature.mode == "none"` and no `delta_t_c` effect; with `race_temp_c` it is `"forecast"`.
5. The shipped calibration carries a circuit-specific dirty-air value for Hungary that is larger than Monza's and the objective uses it (`Calibration.dirty_air_for`).
6. `cliff.detect_stint_collapse` calls a synthetic on-trend stint `strategic` and a synthetic collapsing stint `collapse`; `grip_budget_estimate` never returns a value below the largest observed collapse loss.
7. Reproducibility: `bench/out/strategy.json` first-stop recommendations equal `meta_*.json` (`strategy.best_plan.pit_laps[0]`) for every weekend, and `bench/out/accuracy.json` `sealed` MAE per weekend equals `meta_*.json["score"]["mae"]` within 1e-6.
Plus: all seven weekends have `n_rate_stints` summing to 196 (the population must not change).

## 8. Benchmark protocol (Phase 4)

```
make history OFFLINE=--offline      # fit all 7 → 05_history_practice not needed (cached JSON) → 80_recalibrate → decide all 7
make benchmark                      # bench/run_all.sh → bench/out/
```
Then the report. Runtime budget ≈ 1 h per full cycle. Nothing in `bench/` may write to `data/processed/` or `predictions/sealed/`. If a benchmark definition must change, the old and new metric are both reported.

## 9. Report (`results_v3.md` → `results_v3.pdf` via `bench/md2pdf.py`)

Sections in the order the prompt lists (executive summary; V1→V2→V3; what changed; degradation; strategy; tyre life; per-car; live; ablations; engineering; per-weekend; remaining failures; next steps; exact commands). Every number from `bench/out/*.json`, `data/processed/calibration.json` and `meta_*.json` of the V3 run; V2 numbers from `bench/v2`; V1 from `bench/baseline`. The report must state plainly whether V3 improved first-stop timing and by how much on each of Barcelona, Austria, Hungary.
