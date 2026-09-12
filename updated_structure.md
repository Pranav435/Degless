# BOX BOX — Tyre Degradation Intelligence, Upgraded

**Theme:** AI Motorsport Intelligence
**Pitch in 15 seconds:** Every team will strip noise with a regression and show a curve. We solve the identification problem that regression silently fudges, attach honest uncertainty to the curves, and then cash them in for race decisions — pit calls, undercut windows, strategy win-probabilities — validated cold on a race the model never saw.

---

## 1. Why the baseline plan won't win

The original plan (fuel/traffic/evolution corrections → curve fit → dashboard) is competent, but it is the *default* solution. Expect five other teams to bring FastF1 + XGBoost + Streamlit. It also has three real weaknesses, and each one becomes our differentiator:

1. **The identifiability trap.** Within a stint, tyre age and fuel load are almost perfectly collinear — both change by exactly one unit per lap. A free regression can split "car getting lighter" vs "tyres getting older" almost arbitrarily, so the "clean" degradation curve is partly an artifact of that arbitrary split. The two effects are the same order of magnitude (fuel ≈ 0.04–0.06 s/lap of gain, degradation ≈ 0.03–0.15 s/lap of loss), so this is not a rounding error — it is *the* central statistical problem, and most teams will never mention it. We make solving it the centerpiece.
2. **No uncertainty.** Practice long-runs are tiny samples: a handful of 8–15 lap runs per driver, on different compounds, at different times of day. Point-estimate curves overfit, and a claim like "the cliff is at lap 22" is not honest without a ±. Judges with a stats background will notice who has error bars and who doesn't.
3. **Curves aren't the product.** F1 teams don't pay for degradation curves; they pay for *decisions*: box now or extend, undercut or cover, one stop or two. A static chart demos poorly against a model that visibly wins a race it wasn't shown.

---

## 2. The three winning moves

### Move 1 — Solve identifiability with physics and a second observable

Break the fuel–age collinearity with information a naive regression doesn't use:

- **Physics prior on fuel, not a free coefficient.** Fuel burn per lap is approximately known per track (≈1.4–1.8 kg/lap), and lap-time sensitivity to mass is well characterized (≈0.03 s per kg). Encode both as informative Bayesian priors with realistic uncertainty instead of letting the model estimate fuel effect freely. The prior pins down the fuel share of pace loss, and whatever the data adds beyond it flows to degradation. (If using 2026-regulation races, widen the priors — the new cars carry less fuel energy.)
- **A second measurement channel: corner apex speeds.** Extract per-lap minimum speeds at 3–5 fixed corners from FastF1 telemetry. Fuel mass slows the car everywhere and dominates the accelerating zones; grip loss shows up disproportionately at the apexes. Two observed channels with *different sensitivity signatures* give the model leverage that a single lap-time channel mathematically cannot have. This is also robust to engine-mode games in practice, which pollute lap times but barely touch apex speeds.
- **Track evolution as a shape constraint, not a guess.** In a dry session, grip only improves. Fit session-wide pace vs session time across *all* cars with isotonic (monotone-decreasing) regression — a constraint the physics justifies, which stops the evolution term from absorbing degradation signal.
- **Traffic:** flag laps with gap-to-car-ahead under ~2.0 s; exclude from fitting or carry as an explicit penalty term. Also drop in/out laps, safety-car/VSC laps, and obvious push-lap vs long-run mode mismatches (detectable from fuel-corrected pace clusters).

### Move 2 — Hierarchical Bayes: pool the grid, keep the uncertainty

- **Partial pooling across the field.** No single driver has enough clean long-run laps, but 20 cars together do. A hierarchical model shares each compound's curve *shape* across the grid while giving every car/driver its own pace offset and its own mild deviation. This is the statistically correct answer to sparse practice data — and "we used all 20 cars as replicates of the same tyre physics" is a sentence judges remember.
- **The cliff as a learned parameter.** Model degradation as linear-plus-hinge (smooth softplus knee for sampler friendliness). The posterior over the knee location *is* the cliff prediction: "cliff at lap 21 ± 2," not a vibes-based annotation.
- **Outputs:** per-compound degradation curves with credible bands, per-car offsets, cliff posteriors, and a full predictive distribution for any hypothetical stint.
- **Fallback ladder (de-risking):** if PyMC fights back, drop to statsmodels MixedLM (driver as random effect) + bootstrap for uncertainty bands. Same story, cheaper machinery. Build the MixedLM baseline *first* regardless — it's the sanity check.

### Move 3 — Ship decisions, not curves

This is the demo layer that separates a science project from a product:

- **Strategy optimizer.** Enumerate legal strategies (compound rules, pit windows, 1–2 stops), compute race time as the integral of base pace + fuel + degradation + pit loss, then Monte Carlo over the posterior curves. Output: "Two-stop Medium–Hard–Medium beats one-stop Medium–Hard with 78% probability, expected gain 6.2 s." Probabilities, because we have a posterior — point-estimate optimizers can't say that.
- **Undercut window calculator.** From the curves + measured pit loss: lap-by-lap undercut gain against the car ahead, when the window opens, and when you're safe from the car behind.
- **Counterfactual replay — "the moment" of the demo.** Take the real race, re-simulate a team's actual strategy against the model-optimal one, and quantify the error: "Driver X lost 7.4 seconds by running 4 laps past the cliff." Finding one real, defensible strategy mistake in the validation race is worth more than any chart.
- **Race replay mode.** A lap slider scrubs through the race: the model's stint-pace estimate updates lap by lap, uncertainty visibly shrinks, and a cliff alarm fires *before* the pace collapse is obvious to the eye — overlaid with when the team actually pitted. All replay states are precomputed offline; the slider is just playback, so it looks live but can't break on stage.
- **AI Race Engineer (optional, high leverage).** An LLM layer (e.g. Claude API) strictly grounded on the model's JSON outputs — it may only quote numbers the model produced. It generates radio-style pre-race briefing sheets per driver and answers judge questions live ("should Norris cover the undercut on lap 18?"). This doubles as a text-first accessible mode: the entire analysis is consumable without reading a single chart, which is a genuine inclusion story most teams won't have.

---

## 3. Proof, not vibes — the validation protocol

Most hackathon validation is "we eyeballed it." Ours is a protocol:

1. **Practice-only firewall.** The data loader for model fitting has a hard whitelist: FP1/FP2/FP3 only. Race data physically cannot leak into training.
2. **Sealed predictions.** Before the race data is opened, freeze the predicted stint models to a timestamped JSON (hash it — cheap theater, real hygiene).
3. **Cold race.** One entire weekend at a *different track* stays untouched until the final hours. Running the full pipeline on it once, live, is the generalization headline.
4. **Metrics that mean something:**
   - Stint pace MAE vs actual race laps (target: under ~0.15 s/lap on clean laps).
   - **Calibration:** did the 90% predictive intervals cover ~90% of race laps? A calibration plot is a rigor signal almost nobody brings.
   - Cliff-lap error in laps (predicted knee vs observed pace collapse / actual pit response).
   - Strategy sanity: does the optimizer's ranking of strategies match how the race actually shook out for cars that ran them?

---

## 4. Five-minute demo script

1. **The mess (30 s).** Raw practice lap times, all cars: pure noise. "Which tyre is degrading? You can't tell. Neither can a regression — here's why."
2. **The trap (45 s).** One slide: tyre age and fuel are collinear within a stint; same order of magnitude; a free fit splits them arbitrarily. "Every simple model of this problem is secretly making this split at random."
3. **The fix (60 s).** Peel confounds live: raw lap → fuel-corrected (physics prior) → evolution-corrected (isotonic) → traffic-filtered → clean residuals. Then the money chart: per-compound degradation curves *with credible bands* and a cliff posterior.
4. **The receipt (60 s).** Sealed practice-only predictions overlaid on actual race laps; MAE and the calibration plot. Then the cold-track race, run once, live.
5. **The product (90 s).** Replay mode: cliff alarm fires at lap 19, team pits at 23 — counterfactual shows 7 s lost. Strategy optimizer with win probabilities. Ask the AI engineer one what-if out loud.
6. **Close (15 s).** "Practice data in, race calls out, error bars all the way down."

---

## 5. Repo layout

- `data/raw/` — cached FastF1 sessions (pre-cache before the event; pulls are slow)
- `data/processed/` — lap-level dataframes, one row per driver-lap, with clean-lap flags
- `predictions/sealed/` — frozen pre-race prediction JSONs + hashes
- `src/ingest.py` — FastF1 pulls, caching, session whitelist firewall
- `src/laps.py` — lap-table assembly, traffic/in-out/SC filtering
- `src/telemetry.py` — corner detection on a reference lap, per-lap apex-speed extraction
- `src/evolution.py` — isotonic session-evolution fit
- `src/fuel.py` — per-track burn-rate priors and mass-sensitivity handling
- `src/model_bayes.py` — hierarchical PyMC model (compound curves, hinge cliff, car offsets)
- `src/model_fallback.py` — MixedLM + bootstrap baseline (built first)
- `src/strategy.py` — strategy enumeration, Monte Carlo race-time simulation, undercut windows
- `src/replay.py` — per-lap sequential state updates for race replay (precomputed)
- `src/validate.py` — sealed-prediction scoring: MAE, calibration, cliff error
- `src/engineer.py` — LLM briefing/Q&A layer, grounded on model JSON only
- `app/dashboard.py` — Streamlit, four tabs: Decompose · Curves · Replay · Strategy
- `notebooks/01_explore.ipynb`, `02_identification.ipynb`, `03_validation.ipynb`
- `pitch/` — deck, demo script, backup screenshots and a screen-recorded demo video

---

## 6. Build order with clocks (36-hour plan, team of 4)

**Before the event (if rules allow):** cache all target-session FastF1 data, set up the Python env (PyMC solved and tested), push the repo skeleton, dry-run one MixedLM fit on old data.

- **Hours 0–4 — Foundation (all hands).** Ingest, lap-level dataframe, clean-lap filters, quick EDA. Exit criterion: one tidy parquet per session.
- **Hours 4–10 — Confounds (split).** Person A: fuel priors + isotonic evolution + traffic filter. Person B: corner extraction from telemetry. Exit: residual "clean pace" channel + apex-speed channel per lap.
- **Hours 8–16 — Models.** MixedLM baseline first (hours 8–11), then hierarchical PyMC with hinge cliff (hours 10–16). Exit: posterior curves per compound with bands.
- **Hours 14–22 — Strategy + validation harness.** Optimizer, Monte Carlo, undercut calculator; sealed-prediction pipeline scored on the dev race. Exit: headline MAE + calibration numbers exist.
- **Hours 20–28 — App.** Dashboard tabs, replay-state precompute, LLM engineer if on schedule. Exit: click-through demo works end to end.
- **Hours 28–33 — Cold race.** Run the untouched weekend once, freeze all numbers and screenshots. No model changes after this point.
- **Hours 33–36 — Pitch.** Deck, two full demo dry-runs, record the backup video.

**Role split:** 1 data/ingest, 1 modeling, 1 strategy/validation, 1 app/demo/pitch. Everyone owns the demo script.

---

## 7. Data plan

- **Race selection criteria:** dry, non-sprint weekends at high-degradation tracks (the Bahrain / Barcelona / Suzuka profile), so practice long-runs exist and strategy actually varied. Avoid sprint weekends (one practice session) and wet sessions (evolution monotonicity breaks).
- **Use 2024–25 seasons** for the core build (stable, well-understood car physics for the fuel priors); optionally run the most recent 2026 round at the end for recency flair, with widened priors.
- **Dev race and cold race at different tracks** — cross-track generalization is the headline, not a stretch goal.
- **Measure pit loss per track from the data** (in-lap + out-lap deltas vs normal laps) rather than hardcoding it.

---

## 8. How this maps to judging criteria

- **Innovation:** identification via physics priors + a second telemetry channel; learned cliff with uncertainty; counterfactual strategy replay. None of these are in the default solution.
- **Technical depth:** hierarchical Bayesian model, calibration analysis, sealed cold-track validation protocol.
- **Feasibility:** every risky component has a fallback (MixedLM for PyMC, static counterfactual for replay, optional LLM layer), and the replay demo is precomputed so it cannot fail live.
- **Impact / story:** "we found the seven-second mistake" — a concrete, quantified race-day error no one else will have.
- **Presentation:** a demo that *moves* (replay + alarm), a live judge Q&A via the AI engineer, an accessible text-first briefing mode, and a backup video if the venue Wi-Fi dies.

---

## 9. Stretch goals (only after the cold race is scored)

- Double-ML cross-check: estimate the degradation slope with orthogonalized ML (EconML) as a robustness slide — two very different methods agreeing is powerful.
- Driver tyre-management fingerprints: per-driver deviation from the compound curve as a "who saves their tyres" ranking.
- Cross-season transfer: last year's same-track posterior as this year's prior.
- Thermal proxies: track-temperature interaction on the degradation slope.

---

## 10. Tech stack

- **Data:** FastF1 (laps + telemetry), pandas, pyarrow
- **Confounds:** scikit-learn (isotonic regression), numpy
- **Modeling:** PyMC (hierarchical + hinge cliff); statsmodels MixedLM + bootstrap as fallback
- **Strategy:** pure numpy Monte Carlo (the search space is small — no fancy optimizer needed)
- **App:** Streamlit + Plotly (colorblind-safe compound palette, text-summary mode on every tab)
- **AI engineer:** any LLM API (e.g. Claude), grounded strictly on model-output JSON
