# BOX BOX — MVP Build Plan (single day, ~7h, solo + agent)

## Context

`updated_structure.md` describes a 36-hour, 4-person hackathon build for a tyre-degradation
intelligence product. Today is **idea submission day**: we need a working proof of concept that
runs live, uses real F1 data, and shows real physics — not slideware. Deliverable is a **Streamlit
app running locally** plus a **recorded demo video**. No deploy, no repo polish.

The pitch's whole differentiator is the **identifiability trap**: within a stint, tyre age and fuel
burn are perfectly collinear (both +1/lap), so a free regression splits "car getting lighter" from
"tyres getting older" arbitrarily. The MVP must *demonstrate that split moving* under a physics
prior, with honest uncertainty, and then cash the curves in for a race decision.

### Pre-flight findings (verified live during planning — do not re-check)

| Check | Result |
|---|---|
| 2026 season data on F1 livetiming (FastF1's source) | ✅ All streams present for 2026 rounds: `TimingData`, `TyreStintSeries`, `CarData.z`, `Position.z`, `WeatherData`, `TrackStatus`, `DriverList` |
| FastF1 | `3.8.3` (2026-04-29), `requires_python >=3.10`, classifies 3.13 ✅ |
| Local Python | Homebrew `python3.13` at `/opt/homebrew/bin/python3.13` (note: bare `python3` resolves to an unrelated venv — **use an explicit venv**) |
| Modeling stack on 3.13 | numpyro 0.21.0, jax 0.11.1, pymc 6.3.1, arviz 1.3.0, statsmodels 0.15.0, scikit-learn 1.9.0, streamlit 1.62.0, plotly 7.0.0, anthropic 1.2.0 — all support 3.13 ✅ |
| 2026 rounds with usable dry practice + race | Barcelona (Jun 14), Austria (Jun 28), Hungary (Jul 26), Suzuka (Mar 29) ✅ |
| **2026 Bahrain & Saudi** | ❌ **Missing** from both the livetiming index and OpenF1 lap data. Do not use. |
| Barcelona 2026 FP1–FP3 clean long-run laps | 281 raw → **189 after filtering** (92 MEDIUM / 88 SOFT / 9 HARD) |
| Naive vs fuel-corrected slope (real fit, Barcelona FP1–3) | MEDIUM 0.204 → 0.233 s/lap; SOFT 0.144 → 0.174 s/lap. **The prior moves the slope ~15–20%** — the identifiability story is real and quantified. |
| Barcelona 2026 race deg cross-check | GAS stint 2 (HARD, 26L): +3.58 s first-3 vs last-3 → ~0.14 s/lap. Matches the practice fit. ✅ |
| **Late-race safety car at Barcelona 2026** | Final stints show +17 to +21 s in the last 3 laps for *all* drivers. **TrackStatus filtering is mandatory**, not optional. |
| Naive relative-to-stint-min outlier filters | Inflate slopes (selection bias — the min usually falls early in the stint). **Use absolute rules**, not `t <= 1.03 * stint_min`. |

### Race selection (locked)

- **Dev race: Barcelona 2026** (meeting 1287 — FP1 `11300`, FP2 `11301`, FP3 `11302`, Race `11307`, 66 laps). Dry, track temp 44–50 °C, 35 long runs in FP2 alone.
- **Cold race: Hungary 2026** (meeting 1291 — FP2 `11336`, FP3 `11337`, Race `11342`, 70 laps). Dry, *much* cooler FP2 (27–34 °C) vs FP3 (49–55 °C) — a genuine cross-track, cross-temperature generalization test.
- **Backup: Austria 2026** (FP2 `11309`, FP3 `11310`, Race `11315`) if either above misbehaves.

---

## Architecture

**Primary data source: FastF1 3.8.3** (gives `get_circuit_info().corners` for the apex-speed
channel, `pick_track_status()`, merged telemetry with a `Distance` channel — none of which OpenF1
provides). **Fallback / independent cross-check: OpenF1 REST** (`/laps`, `/stints`, `/weather`),
already proven to work end-to-end during planning. Both paths feed the *same* lap-table schema, so
the fallback is a one-line source swap.

```
degless/
  plan.md
  requirements.txt
  Makefile                    # make cache / make pipeline / make app
  data/raw/fastf1_cache/      # FastF1 disk cache (big; populated at H0 in background)
  data/processed/             # laps_<event>_<session>.parquet, apex_<event>.parquet
  predictions/sealed/         # frozen JSON + sha256
  src/
    config.py                 # EVENTS, track constants, fuel/mass priors, corner picks
    ingest.py                 # FastF1 pulls + PRACTICE-ONLY firewall + OpenF1 fallback
    laps.py                   # lap table assembly, clean-lap flags, traffic gap, pit loss
    telemetry.py              # corner selection + per-lap apex speed extraction
    fuel.py                   # 2026 fuel/mass physics prior
    evolution.py              # isotonic track evolution (backfitted)
    model_fallback.py         # statsmodels MixedLM + bootstrap  (BUILD FIRST)
    model_bayes.py            # NumPyro hierarchical + softplus hinge + apex channel
    strategy.py               # pit-loss, strategy enumeration, MC race sim, undercut
    replay.py                 # precomputed per-lap sequential states
    validate.py               # sealed predictions, MAE, calibration coverage, cliff error
    engineer.py               # (P2) Claude-grounded briefing/Q&A
  app/dashboard.py            # Streamlit: Decompose · Curves · Validate · Replay · Strategy
  scripts/
    00_cache.py               # background pre-cache of all 8 sessions
    10_pipeline.py            # end-to-end: laps -> physics -> fits -> sealed json
    20_coldrace.py            # one-shot cold-race run
```

**Engine choice: NumPyro over PyMC.** PyMC needs PyTensor C compilation, which is the classic
macOS/Python-3.13 time sink. NumPyro's NUTS is pure JAX CPU, installs from wheels, and samples
~300 laps in well under a minute. PyMC stays as a documented alternative; it is not on the
critical path. The **MixedLM baseline is built first regardless** — it is the sanity check that
tells us the Bayesian posterior isn't lying.

---

## Physics: the fuel prior (this is the centerpiece — get it exactly right)

2026 regulations, both verified: **race fuel allowance 70 kg** (down from 110 kg) and **minimum
car weight 768 kg** (down from 800 kg).

**Burn rate** (`src/fuel.py`):
```
burn_kg_per_lap = FUEL_ALLOWANCE_2026 / n_race_laps
  Barcelona: 70 / 66 = 1.061 kg/lap
  Hungary:   70 / 70 = 1.000 kg/lap
```

**Mass sensitivity**, derived rather than hardcoded — a ~1 % mass increase costs ~α of lap time,
with α ≈ 0.30 (dimensionless, the standard aero/mechanical rule of thumb):
```
k_track [s/kg] = alpha * t_lap_ref / m_total,   alpha = 0.30
  m_total = 768 kg (min weight, incl. driver) + mid-stint fuel (~35 kg) ≈ 803 kg
  Barcelona: 0.30 * 78 s / 803 = 0.0291 s/kg
  Hungary:   0.30 * 77 s / 803 = 0.0288 s/kg
```
This lands on the well-known ~0.03 s/kg figure *from first principles*, which is exactly the slide
that sells Move 1. Encode it as an informative prior, **not** a fixed constant:
```
k_track ~ LogNormal(log(k_hat), 0.25)      # 25% relative SD, widened for 2026 novelty
```

**Only the fuel *change* within a stint is identifiable** — the unknown starting fuel load of a
practice long run is absorbed into that stint's intercept. So the fuel term is:
```
fuel_term(i) = -k_track * burn_kg_per_lap * lap_index_within_stint(i)
```
Do not attempt to estimate absolute practice fuel loads. That is a trap and it costs hours.

**Sensitivity slide (free, high value):** re-run the fit with the *2025* prior
(1.67 kg/lap × 0.033 s/kg) and show the degradation slope shift. Already measured in planning:
MEDIUM 0.233 → 0.371 s/lap. "Use last year's physics and you misprice the tyre by 60 %."

---

## Clean-lap rules (absolute, never relative-to-min)

In `src/laps.py`, one boolean column per rule plus an `is_clean` conjunction, so the Decompose tab
can show laps falling away rule by rule:

1. `not lap.IsAccurate` → drop (FastF1's own validity flag).
2. Pit in-lap / out-lap → drop (`PitInTime`/`PitOutTime` non-null).
3. **Track status ≠ "1" (all-green)** → drop. Covers yellow, SC, VSC, red. **Mandatory** — the
   Barcelona 2026 race ends under a safety car and will otherwise poison every late-stint fit.
4. Stint length ≥ 6 laps after the above (long runs only, excludes quali sims).
5. **Traffic:** for each lap, compute gap to the car ahead from `LapStartTime` across all drivers —
   the smallest positive `LapStartTime` difference to any other car. Flag `< 2.0 s` → drop.
6. Lap time > `session_median + 5 s` → drop (in-lap cooldowns, aborted runs, box-box laps).
7. `Compound` in {SOFT, MEDIUM, HARD} and non-null.

Tyre age: prefer FastF1's `TyreLife`; fall back to `lap_number - stint_start + tyre_age_at_start`.
Expect ~190 clean laps per weekend from FP1–FP3. **HARD will be thin (~10 laps) — do not hide
this. Wide credible bands on HARD are the honest-uncertainty story, not a bug.**

**Pit loss, measured not assumed** (`src/strategy.py`): for every green-flag race pit stop,
`(in_lap + out_lap) − 2 × median(driver's clean laps ±5 laps)`. Take the median across drivers.

---

## Track evolution (backfitted isotonic)

Naive isotonic-on-raw-pace absorbs degradation signal. Fit it **jointly, by backfitting**, 3 iterations:

```
resid = lap_time - stint_intercept - fuel_term
for _ in range(3):
    evo   = IsotonicRegression(increasing=False, out_of_bounds="clip")
              .fit(session_elapsed_s, resid - deg_hat)
    deg_hat = OLS(resid - evo(session_elapsed_s) ~ age by compound + stint FE)
```
Monotone-decreasing is physically justified in a dry session (rubber only goes down). Both target
weekends are verified dry, so the constraint holds. Fit across **all cars pooled** — it is a track
property, not a car property.

---

## Second channel: corner apex speeds (Move 1's real leverage)

`src/telemetry.py`. This is the **most time-expensive component** — telemetry downloads dominate
wall clock. It is P1 with a hard cut rule (below).

1. `ci = session.get_circuit_info()` → `ci.corners` DataFrame with `Number` and `Distance`.
2. For each clean lap: `tel = lap.get_car_data().add_distance()`.
3. For each corner, `min(Speed)` within `Distance ∈ [corner_dist − 60 m, corner_dist + 60 m]`.
4. Pick the **4 corners** with the highest across-lap *variance* in apex speed among slow corners
   (`median apex < 0.6 × session max speed`) — these carry the most grip information.
5. Persist to `data/processed/apex_<event>.parquet` keyed by `(driver, lap_number, corner)`.

**Why it breaks the collinearity:** fuel mass slows the car roughly uniformly and dominates
accelerating zones; grip loss hits apex minimum speeds disproportionately. Two channels with
*different sensitivity signatures* to the same two latent causes make the split identifiable —
something a single lap-time channel mathematically cannot do. It is also robust to practice engine
modes, which wreck lap times but barely touch apex speeds.

---

## Models

### `src/model_fallback.py` — MixedLM baseline (build first, ~30 min)

```python
smf.mixedlm("lap_time_corr ~ C(compound):tyre_age + C(compound)",
            df, groups=df["driver"], re_formula="~1").fit()
```
where `lap_time_corr = lap_time − fuel_term − evo_term`. 200× block bootstrap resampling **whole
stints** (not laps — laps within a stint are correlated) for uncertainty bands. This is the number
the Bayesian fit must agree with; if they diverge by more than ~0.03 s/lap, the Bayesian model has
a bug, not an insight.

### `src/model_bayes.py` — NumPyro hierarchical, hinge cliff, two channels

For lap *i* with driver *d*, compound *c*, stint *s*, tyre age *a*, session elapsed *τ*:

```
lap_time[i] ~ Normal( base[s] + evo(τ) - k_track*burn*a_in_stint
                      + deg[c](a) + dev[d,c]*a ,  sigma_obs )

deg[c](a) = lin[c] * a  +  hinge[c] * w * softplus( (a - knee[c]) / w )
```

Priors:
| Parameter | Prior | Rationale |
|---|---|---|
| `base[s]` (per stint) | `Normal(median_lap, 1.5)` | absorbs unknown practice fuel load + engine mode |
| `mu_lin` | `HalfNormal(0.06)` | grid-level linear deg |
| `lin[c]` | `Normal(mu_lin, sigma_lin)`, `sigma_lin ~ HalfNormal(0.03)` | **partial pooling across compounds** |
| `knee[c]` | `TruncatedNormal(18, 6, low=5, high=40)` | the cliff, as a learned parameter |
| `hinge[c]` | `HalfNormal(0.10)` | post-cliff extra slope |
| `w` | fixed `1.5` laps | softplus smoothness — sampler-friendly |
| `k_track` | `LogNormal(log(k_hat), 0.25)` | **the physics prior — Move 1** |
| `dev[d,c]` | `Normal(0, sigma_dev)`, `sigma_dev ~ HalfNormal(0.02)` | per-driver tyre-management deviation |
| `sigma_obs` | `HalfNormal(0.3)` | |

**Apex channel** (joint likelihood, shared latent grip):
```
apex[j] ~ Normal( v0[d, corner] - lambda[corner] * grip_loss[c](a) , sigma_apex )
grip_loss[c](a) = deg[c](a)                      # same latent curve, different observable
lambda[corner] ~ HalfNormal(20)                  # km/h lost per second-per-lap of grip loss
```
Sampling `deg[c]` against *both* likelihoods is what pins the fuel/age split. **This is the single
most defensible technical claim in the pitch.**

NUTS: 4 chains × 1000 warmup × 1000 draws, `target_accept_prob=0.9`. Expect < 90 s on ~200 laps.
Save the InferenceData. **Convergence gate: all `r_hat < 1.01`, zero divergences.** If it
diverges, non-center `dev[d,c]` first; if it still fights, ship MixedLM and say so honestly.

**Outputs consumed by everything downstream:** per-compound posterior curves with 90 % credible
bands, cliff posterior (`knee[c]` histogram → "cliff at lap 21 ± 2"), per-driver offsets, and a
posterior-predictive function `predict_stint(compound, start_age, n_laps) -> (n_draws, n_laps)`.

---

## Validation protocol (the receipt)

1. **Practice-only firewall** in `src/ingest.py`: `load_for_fitting()` asserts
   `session_name in {"Practice 1","Practice 2","Practice 3"}` and raises otherwise. Race data
   *cannot* reach the fitter. Show this assert on screen in the demo — it costs 5 seconds and buys
   total credibility.
2. **Seal** predictions to `predictions/sealed/<event>_<utc>.json` + a `sha256` sidecar, *before*
   opening race data.
3. **Score** on Barcelona 2026 race clean laps (`src/validate.py`):
   - Stint pace **MAE** — target < 0.15 s/lap.
   - **Calibration**: fraction of race laps inside the 90 % predictive interval. Plot nominal vs
     empirical coverage across 50/80/90/95 %. Almost nobody brings this.
   - **Cliff error**: posterior `knee[c]` vs observed race pace-collapse lap.
4. **Cold race: Hungary 2026, run once, live, no retuning.** Different track, different
   temperature regime. Freeze all numbers and screenshots after.

---

## Decisions, not curves

`src/strategy.py`:
- **Pit loss** measured from the race data (above).
- **Enumerate** legal strategies: 1–2 stops, compound sequences satisfying the two-compound rule,
  pit laps on a 1-lap grid within [8, n_laps−8]. Search space is ~small thousands — brute force in
  numpy, no optimizer needed.
- **Race time** = Σ over laps of `base + k_track*burn*(n_laps−lap) + deg[c](age)` + `n_stops × pit_loss`.
- **Monte Carlo over 500 posterior draws** → a *distribution* over strategy rankings, so we can
  say "Two-stop M–H–M beats one-stop M–H with 78 % probability, expected gain 6.2 s". A
  point-estimate optimizer physically cannot say that. This is the whole reason we went Bayesian.
- **Undercut window**: lap-by-lap `undercut_gain(L) = deg_ahead(age) × out_lap_advantage − pit_loss_delta`;
  report when the window opens and when you're exposed to the car behind.
- **Counterfactual — "the moment"**: for each driver, compare actual pit lap vs model-optimal pit
  lap and quantify the loss in seconds. Barcelona 2026 has long HARD stints (GAS s2: 26 laps,
  +3.58 s of deg) — a real finding is highly likely to be there. Search all drivers, pick the
  largest defensible loss, and **verify by hand** before it goes in the video.

`src/replay.py`: precompute per-lap sequential states offline into one parquet. The slider is pure
playback — it looks live and **cannot break on stage**. State per lap: posterior stint-pace
estimate, shrinking uncertainty band, cliff-alarm boolean, actual pit laps overlaid.

---

## App

Streamlit 1.62 + Plotly 7. Colorblind-safe compound palette (SOFT `#D55E00`, MEDIUM `#F0E442`,
HARD `#0072B2` — Okabe–Ito). Every tab carries a plain-text summary block, so the whole analysis is
consumable without reading a chart — a genuine accessibility story.

| Tab | Content |
|---|---|
| **Decompose** | Raw practice laps (noise) → clean-lap filter cascade (laps falling away, rule by rule) → fuel-corrected → evolution-corrected → residuals. The "peel the confounds" animation. |
| **Curves** | Per-compound posterior curves with 90 % credible bands + cliff posterior histogram. Toggle: 2026 prior vs 2025 prior vs no prior — **the identifiability slide, live**. |
| **Validate** | Sealed prediction vs actual race laps, MAE, calibration plot, cliff error. Cold-race panel. |
| **Replay** | Lap slider, uncertainty shrinking, cliff alarm firing, actual pit lap overlaid, counterfactual seconds lost. |
| **Strategy** | MC strategy ranking with win probabilities, undercut window chart. |

Everything reads precomputed parquet/JSON. **No model fitting inside the app** — it must start in
under 3 seconds.

---

## Schedule (7 hours, solo + agent)

| Time | Work | Exit criterion |
|---|---|---|
| **H0:00–0:20** | venv on `python3.13`, `requirements.txt`, `src/config.py`. **Launch `scripts/00_cache.py` in the background immediately** — it pre-caches all 8 sessions incl. telemetry (20–40 min). This is the long pole; everything else is written while it runs. | background cache running |
| **H0:20–1:20** | `ingest.py` (+ firewall), `laps.py` (clean-lap rules, traffic gap, tyre age), OpenF1 fallback path. | one tidy parquet per session; ~190 clean laps at Barcelona |
| **H1:20–2:10** | `fuel.py`, `evolution.py` (backfitted isotonic). | clean-pace residual channel exists |
| **H2:10–2:40** | `model_fallback.py` MixedLM + stint bootstrap. | **first real deg numbers with bands** — a shippable demo already exists here |
| **H2:40–3:40** | `model_bayes.py` NumPyro, lap-time channel only. Convergence gate. | posterior curves + cliff posterior, r_hat < 1.01 |
| **H3:40–4:20** | `telemetry.py` apex speeds + join the second channel into the model. **CUT RULE: if not converging by H4:20, ship lap-time-only and present apex speeds as a diagnostic chart instead.** | joint model or a clean fallback |
| **H4:20–5:00** | `validate.py`: seal → score Barcelona race. MAE + calibration. Then **cold race Hungary, one shot**. | headline MAE + coverage numbers exist |
| **H5:00–6:00** | `strategy.py` (pit loss, MC, undercut) + counterfactual search + `replay.py` precompute. | one verified "driver X lost Y seconds" finding |
| **H6:00–6:40** | `app/dashboard.py`, five tabs. | click-through works end to end |
| **H6:40–7:00** | Record the demo video. Freeze. | video exists |

**Cut line if time runs short** — drop in this order, and only this order:
1. AI Race Engineer (`engineer.py`) — P2, never on the critical path.
2. Apex-speed second channel → present as a diagnostic chart, keep the physics prior.
3. Cold race → single-track validation only, and *say* that's a limitation.
4. Strategy MC → keep the counterfactual, which is the emotional beat.

**Never cut:** the practice-only firewall, the clean-lap cascade, credible bands, the
2026-vs-2025 prior toggle. Those four *are* the differentiator.

---

## Verification

Each stage is verifiable standalone — run these as you go, don't batch them to the end:

```bash
make cache          # python scripts/00_cache.py   (background, H0)
make pipeline       # python scripts/10_pipeline.py --event barcelona-2026
make coldrace       # python scripts/20_coldrace.py --event hungary-2026
make app            # streamlit run app/dashboard.py
```

Assertions to write as you build (cheap, catch the real bugs):
1. **Firewall:** `load_for_fitting(session="Race")` raises. Test it explicitly.
2. **Lap counts:** Barcelona FP1–FP3 yields ~180–200 clean laps, compound split roughly
   92 MED / 88 SOFT / 9 HARD. A big deviation means a filter regressed.
3. **Slope sanity:** MixedLM MEDIUM slope lands in **0.12–0.30 s/lap**. Planning measured 0.204
   naive / 0.233 fuel-corrected. Outside that range, something is wrong.
4. **Race cross-check:** predicted MEDIUM/HARD deg reproduces the observed ~0.14 s/lap on GAS's
   26-lap HARD stint in the Barcelona race.
5. **Safety-car guard:** assert no lap with `TrackStatus != "1"` survives into any fit. The
   Barcelona race SC produces +20 s laps — if slopes explode, this filter broke.
6. **Convergence:** `az.summary(idata).r_hat.max() < 1.01` and zero divergences, asserted in the
   pipeline, not eyeballed.
7. **Bayes vs MixedLM:** posterior mean slope within 0.03 s/lap of the MixedLM point estimate.
8. **Calibration:** empirical 90 % coverage in [0.80, 0.97]. Outside that, report it honestly —
   a miscalibrated model shown honestly still beats a point estimate shown confidently.
9. **App cold start:** `streamlit run` to interactive in < 3 s (proves nothing fits at runtime).

**Final gate before recording:** full pipeline from a cleared `data/processed/` on Barcelona, then
Hungary cold, then click every tab. Record only after that passes.

---

## Notes

- Use an explicit venv: `/opt/homebrew/bin/python3.13 -m venv .venv`. Bare `python3` on this
  machine resolves to an unrelated project's venv.
- FastF1 telemetry pulls are slow and rate-limited. Cache once at H0, never re-pull during the
  build. If a session 404s, fall back to the OpenF1 path (verified working) rather than debugging.
- 2026 Bahrain and Saudi are missing upstream. Do not use them, do not spend time investigating.
- On approval, this plan is written to `degless/plan.md` as the working document.
