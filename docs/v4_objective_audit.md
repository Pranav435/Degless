# V4 objective audit — every place a strategy cost is priced

*WP-F, 2026-09-13. Written against the code at the head of this branch (WP-F applied). Line numbers are from that state; `git log -p -- src/objective.py` is the change.*

**The one objective** (`src/objective.py::V4Objective`, label
`"V4 objective: race state (<mode> rival field), lambda on later stops, tau plan prior, no first-stop prior"`):

| term | where it comes from | charged on |
|---|---|---|
| tyre, pit lane, dirty air at the rejoin, safety-car credit | the tyre model and the measured constants | every stint / stop |
| **race state** `V·(places(s) − places(s*))` | `racestate.pack_equilibrium` / WP-A's rival field, per plan group | the **first** stop |
| undercut exposure × `undercut_lambda` | `strategy.undercut_exposure_tables` | the stops **after** the first |
| plan-family prior × `plan_prior_tau_s` | `history.plan_prior_for` | the plan |
| grid penalty × start-compound hardness | `calibration.grid_start_penalty_s` | the plan |
| first-stop history prior × `first_stop_kappa_s` | `firststop.first_stop_penalty_table` | **nothing** — structurally 0 under V4; the tables are still handed to the search for the *rivals'* stop laps (WP-A) |

`V4Objective(first_stop_kappa_s > 0)` raises. The only builder that produces a non-zero kappa is
`V4Objective.v3_for_event`, whose `version` is `"v3"` and whose label starts "V3 baseline objective"; `v3_kwargs()`
refuses to run on anything else.

## Classification

`production decision path` = a number the pipeline / weekend script / outlook ships ·
`live decision path` = the race engine · `diagnostic` = printed or written but not decided on ·
`benchmark baseline` = a deliberate V3/oracle comparison · `historical compatibility` = reading an older artefact.

### Production decision path — converted

| file:line | what was there (V3 / Task 1) | V4 |
|---|---|---|
| `src/objective.py:242-469` | — | **new.** `V4Objective` + `sim_kwargs / window_kwargs / eval_kwargs / counterfactual_kwargs / model_kwargs`, each dropping keywords the target does not accept (`accepts`, `_filtered`), so the same object runs before and after the WP-A merge. |
| `src/objective.py:106-121` | `racestate.term_by_lap` called ad hoc at two sites | `terms_by_group(res, n_laps)` → `{group label: term by lap}`, the one producer of the terms every scorer reads. |
| `src/objective.py:144-182` | `scripts/10_pipeline.py::race_state_block` (script-local) | **moved** into `src.objective`; the pipeline keeps the name as an alias (`scripts/10_pipeline.py:81`), `40_weekend.py:442` and `src/outlook.py:616` now write the same block. |
| `src/strategy.py:1694-1860` `evaluate_plans` | position term = `lambda ×` exposure on **every** stop; no race state; `first_stop_kappa_s` charged | accepts `race_state_terms`; the plan's own group term replaces the **first** stop's exposure (`strategy.py:1777-1787, 1802-1806`), later stops keep `lambda`, kappa forced to 0 (`strategy.py:1737-1740`), `race_state_s` + `race_state` note in every row and detail. Absent keyword ⇒ byte-identical V3 behaviour (tested). |
| `src/strategy.py:1873-1886` `deg_crossover` | forwarded `**kw` only | explicit `race_state_terms`, documented as *not* rescaled by the degradation multiplier. |
| `src/strategy.py:1440-1650` `counterfactual` | V3: exposure on every stop **and** `first_stop_kappa_s` on the actual and alternative first stops | accepts `race_state_terms`; each driver's own plan group's term on the first stop of the actual plan, of every enumerated alternative and of the best (`strategy.py:1549-1566, 1603-1605, 1615`), kappa forced to 0, `race_state_s` in the row, and a `flag` when the driver's family has no term. |
| `src/strategy.py:1202-1300` `pit_window_model` | already took `race_state_term` (Task 1) | unchanged; it is now the *definition* the other scorers are tested against (`tests/test_v4_objective.py::test_evaluate_plans_and_pit_window_charge_the_same_first_stop`). |
| `scripts/10_pipeline.py:571-596` | `sim_kw` spelled out by hand, `kappa_used = 0 if use_rs else cal.first_stop_kappa_s` | `obj = V4Objective.for_event(...)` (or `v3_for_event` under `--no-race-state`), `sim_kw = bounds + obj.sim_kwargs()`; the label is printed. |
| `scripts/10_pipeline.py:635-636` | `pit_window_model(..., first_stop_prior=..., race_state_term=...)` | `**obj.window_kwargs(res)`. |
| `scripts/10_pipeline.py:656-661` | `counterfactual(..., first_stop_kappa_s=cal.first_stop_kappa_s)` — **the Task 1 gap**: judged under a term the recommendation never paid | `**obj.counterfactual_kwargs(res)`. On hungary-2026 18 of 19 drivers' `model_pit_laps` first stop moves. |
| `scripts/10_pipeline.py:678-681, 885, 900` | per-car plans on `sim_kw`; meta carried `race_state_s` | unchanged call, now on the V4 `sim_kw`; meta gains `strategy.race_state_terms` and an `objective` block. |
| `scripts/40_weekend.py:278-289, 304-305, 434-445` | `sim_kw` by hand, `racestate.measure_constants(exclude=key)` inline, no `race_state` block in the meta | `V4Objective.for_event` + `obj.sim_kwargs()` / `obj.window_kwargs(res)`; `weekend_<key>.json` gains `race_state`, `objective` and `strategy.race_state_terms`. |
| `src/outlook.py:153-197` `objective_for / sim_kwargs / eval_kwargs` | both spelled the V3 weights out, kappa on, no race state | built from `V4Objective`; `objective_of(base)` keeps `base.plan_prior` authoritative so a caller may swap the prior on a copy (`bench_outlook`'s letter-for-letter variant). |
| `src/outlook.py:562-564, 579-590, 439` | pit window, plan-B crossover and the scenario regrets on V3's terms | `obj.window_kwargs(res)` / `eval_kwargs(base, res)`; the terms are the measured race state, so the same terms price every degradation scenario (documented at `outlook.py:436-438`). |
| `src/outlook.py:610-618` | `strategy.{position_s, prior_s, first_stop_s}` | adds `strategy.race_state_s`, `strategy.race_state` (the shared block) and `strategy.race_state_terms`, plus a top-level `objective` block with the label. |
| `app/desk_tab.py:45-85` | `evaluate_plans` / `pit_window_model` / `deg_crossover` called with **no** objective terms at all — a hand-built plan was priced on the tyre alone | `_objective_json(outlook)` → `_eval_kw` / `_window_kw`: `undercut_lambda`, `plan_prior` (+ `tau`), `traffic_s_per_lap`, `grid_penalty_s` and the plan's own family `race_state_term(s)`. |
| `app/desk_tab.py:107-146, 470, 624` | — | the three cached computations take `obj_json`; the compare table shows "Track position at the first stop (s)" (`desk_tab.py:279-291`), "—" where the family has no term. No banned jargon added (`tests/test_app_smoke.py`). |
| `src/calibration.py:74-92, 112-116, 160-165` | no V4 keys | `family_temper_s` (3.0, WP-A's default), `extrap_ln_sd` (0.0 = V3 bit for bit), `objective_version` (`"v4"`; a block written by the V3 script carries no key and is stamped `"v3"`). |
| `scripts/80_recalibrate.py` (whole) | see below | see below |

### Production decision path — recalibration (`scripts/80_recalibrate.py`)

| file:line | V3 | V4 |
|---|---|---|
| `:295-338` `Donor.objective / _sim_kw / search / calibrated_model` | no `race_state` in any sweep search; kappa swept | every search carries `V4Objective` with `race_state = measure_constants_excluding({held_out, donor})` (the global block excludes only the donor) and the block's `family_temper_s`; `kappa` asserted 0. |
| `:261-275` `Donor.race_state` | — | the constants per exclusion set, cached. |
| `:186-203` field measurements | first stops only | adds `second_by_stops` / `second_median_by_stops` / `second_n_by_stops` (green second stops among finishers with that stop count) and `field_stop_mode`. |
| `:340-371` `decision_scores` | `first_err` | adds `second_err` (the lambda objective), `second_lap`, `race_state_s`, `family_costs`. |
| `:698-710` `obj_second` | `obj_first` for both lambda and kappa | lambda's objective is the **later** stops: mean \|recommended second stop − field median green second stop among same-stop-count finishers\| over donors whose recommendation and field mode are ≥ 2 stops; parsimony `tol` = one lap over those donors; flat ⇒ smallest wins and `undercut_lambda_identified: false`. |
| `:737-756` the sweep loop | lambda, kappa, tau, grid × 2 | lambda, tau, grid × 2; **no kappa sweep** (`KAPPA_V4 = 0.0` at `:129-132`), V3's value kept at `raw.v3_first_stop_kappa_s` with a note. |
| `:375-490` `family_costs / family_logit / family_loglik / sweep_family_temper` | — | `family_temper_s` by maximum likelihood of the donors' start-compound and stop-count shares under `q(g) ∝ exp(−C_g/τ_f)·p_hist(g)^w`, `w = n/(n+10)`, grid {1,2,3,5,8} s, LOO. `C_g` = min `tyre_s` per (start, second, stops) group from `res.table` (or WP-A's own family table when the merge exposes costs). No extra searches. |
| `:766-815` the written block | — | `first_stop_kappa_s: 0.0`, `family_temper_s`, `extrap_ln_sd` (from `src.tyre.EXTRAP_LN_SD_MEASURED` if WP-B has it, else 0.0), `objective_version: "v4"`, `objective_label`, `objective_notes`, `undercut_lambda_identified`, `family_temper_identified`, `race_state` (the constants each donor was searched with), `held_out`. |

### Live decision path — **not** WP-F's to convert (WP-D)

| file:line | state | note |
|---|---|---|
| `src/live/engine.py:139-147, 225-229` | `WeekendModel` carries `undercut_lambda`, `first_stop_kappa_s`, `first_stop_table`, `race_state` as separate fields, assembled from `Calibration` | already V4-shaped in effect: with the V4 calibration `cal.first_stop_kappa_s` is 0, so nothing is charged. **Left alone by WP-F** — WP-D owns this file; integration should let it read `V4Objective` rather than four fields. |
| `src/live/engine.py:392-398, 432-436, 657-658` | `self.lam`, `self.kappa`, `undercut_exposure_tables`, `race_state` | the engine prices each option's *next* stop against the real rivals (`racestate.live_position_term`), and `first_stop_prior_applies` (`:976`) is already `rs is None and … kappa > 0`: history times nothing while the race state is on. |
| `src/live/engine.py:958, 974-981, 1414-1418` | reports `undercut_lambda`, `first_stop_kappa_s`, `race_state` per plan / snapshot | diagnostic reporting; unchanged. |

### Diagnostics — left as they are, and why

| file:line | what | why |
|---|---|---|
| `src/strategy.py:262-296` `undercut_exposure_tables` | the V3 exposure table itself | still the objective's later-stop term; not removed. |
| `src/strategy.py:333-385` `first_stop_penalty` | the V3 first-stop density term | kept, reachable only at `kappa > 0`: the V3 baselines, the ablations and `--no-race-state`. |
| `src/firststop.py` (whole, 7 hits) | the density tables and their summary | still built everywhere: the plan-family prior, the report's "the circuit prefers lap 18", and WP-A's rivals. Not a term on our own lap. |
| `src/strategy.py:854, 1810-1815` `tyre_optimal`, `first_stop_s` on the tyre-optimal plan | "what the plan would have paid" | diagnostic, printed beside the recommendation. |
| `scripts/10_pipeline.py:629-633, 666-667` | prints the first-stop prior term and the circuit's modal lap | diagnostic text; the number is 0 under V4 and the print says so. |
| `src/outlook.py:249-254, 324-328` (`sources`) | "circuit first-stop history … not a term on ours (kappa 0 under the race-state objective)" | was "first-stop prior … at kappa 1.00 s/nat", which is no longer true. |

### Benchmark baselines — deliberately V3, now labelled

| file:line | what | V4 |
|---|---|---|
| `bench/common.py:301-352` `kappa_of / fs_kwargs / sim_kwargs_for` | the V3 first-stop kwargs helper every bench script uses | untouched: it is what the V3 columns are produced with. |
| `bench/bench_ablation.py:89-131, 249-263` | `no_race_state`, `race_state_no_cover`, `race_state_lead_lap_value`, `race_state_undiscounted`, `config_constants` | untouched; `no_race_state` **is** the V3 objective and must stay reachable (WP-H adds the new variants). |
| `bench/bench_speed.py:228-243` | one timing row "desk: evaluate 3 plans" on V3's kwargs | now two rows: "desk: evaluate 3 plans (V4 objective, race-state terms)" and "... (V3 objective baseline)"; the crossover row and the counterfactual row likewise (`:214-221`, `:243`). |
| `bench/bench_strategy.py:324` `evaluate_plans(orc, …)` on the oracle model | the pure-time oracle regret | **left on pure time deliberately** — WP-C's `R_pos` adds the position-aware metric *beside* it, and changing this row would silently redefine every historical regret number. |
| `bench/bench_strategy.py:253-255, 373-411` | reports `first_stop_kappa_s` from the meta or the calibration | reads the shipped artefact; now reports 0. |
| `bench/bench_compare.py:352-354, 977-1007` | the V2/V3/V4 constants table, including κ | historical compatibility: it prints whatever each generation wrote. |
| `bench/bench_stability.py:75-78`, `bench/bench_apex.py:70`, `bench/bench_accuracy.py:350` | spell the weights out for their own search | benchmark-local searches; each already carries the race state where Task 1 added it. Integration may move them to `V4Objective`; no decision depends on them. |
| `bench/bench_outlook.py:84-92` | `first_stop_in_objective = bool(sk.get("first_stop_prior"))` | now `… and kappa > 0`, plus `objective`, `objective_version`, `race_state_in_objective`, `race_state_s`: the row says which objective the prior-only search ran. |

### Historical compatibility

| file:line | what |
|---|---|
| `src/calibration.py:160-165` | a V3 `calibration.json` has no `objective_version`, `family_temper_s` or `extrap_ln_sd`: the reader stamps `"v3"` and uses the dataclass defaults. |
| `src/config.py` `FIRST_STOP_KAPPA_S`, `UNDERCUT_EXPOSURE_LAMBDA` | the defaults with their derivations; `FIRST_STOP_KAPPA_S` is now referenced only by `Calibration`'s default and the V3 baseline builder. |
| `tests/test_v3.py` (4 hits), `tests/test_outlook.py:97-135` | V3-objective tests: they call the scorers without `race_state_terms`, which is why the keyword's absence has to stay byte-identical. |

## What remains unconverted after WP-F (stated, not hidden)

1. **`src/live/engine.py`** assembles the objective from four `WeekendModel` fields rather than a `V4Objective`. Behaviourally V4 already (kappa 0 through the calibration, race state on), but it is a second spelling of the same object. WP-D owns the file; integration should wire `V4Objective` in.
2. **`bench/bench_strategy.py`'s oracle regret** is pure race time by design (WP-C).
3. **`bench_stability` / `bench_apex` / `bench_accuracy`** still spell their search kwargs out.
4. **`sc_playbook`** takes no first-stop term at all, by design: it prices only this lap's safety-car decision (`outlook.py:597-599`).
5. **`extrap_ln_sd`** is carried through the objective and the calibration but is 0 until WP-B lands `src.tyre.EXTRAP_LN_SD_MEASURED`; `V4Objective.model_kwargs()` is the hook and drops itself on a checkout whose `TyreModel` has no such field.
6. **`rival_field`** is `None` (Task 1's symmetric pack) until WP-A lands `racestate.RivalFieldConfig`; `objective.rival_field_default` returns it the moment the dataclass exists, and `sim_kwargs` only passes the keyword to a `simulate_model` that accepts it.
