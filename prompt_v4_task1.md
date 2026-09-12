# V4 Task 1 — Fix Race-State and Pit Strategy

<role>
Act as the lead race-strategy engineer for Degless V4.

Use Opus 5 for investigation, modeling, and implementation. Use Sonnet 5 for testing and verification.

Do not redesign the entire algorithm. Focus exclusively on fixing Degless's biggest V3 weakness: pit-stop timing and race-state decision making.
</role>

<context>
V3's key failure is first-stop timing:

- V3 first-stop error: 4.33 laps
- Target: <=2 laps
- Field first stops inside model window: 36.1%
- V3 oracle regret: 9.399s

The historical plan prior is useful, but it currently has too much influence over WHEN a car pits.

Race state must become the primary decision-maker.
</context>

<task>

Inspect the existing strategy engine and redesign the pit decision around:

- current lap
- laps remaining
- Haas position
- gap ahead
- gap behind
- nearby competitor positions
- nearby competitor tyre compounds
- nearby competitor tyre ages
- competitor pit status
- current Haas tyre age
- current tyre pace
- predicted next 1–5 lap pace
- degradation
- pit-lane loss
- out-lap penalty
- traffic on rejoin
- expected rejoin position
- undercut exposure
- overcut opportunity
- SC/VSC state
- remaining tyre life

For every relevant lap, explicitly compare:

- PIT NOW
- STAY OUT 1 LAP
- STAY OUT 2 LAPS
- STAY OUT 3 LAPS
- PIT AT EDGE OF WINDOW

Choose the action with the best expected race outcome.

Do not simply move the existing first-stop prior earlier.

Historical strategy should be a prior for what is plausible, not the primary reason for a pit call.

Do not add large numbers of arbitrary weights. Every new strategy term must have a clear race-time interpretation.

Model the nearest 3–5 relevant competitors rather than giving every car equal strategic importance.

Preserve the V3 tyre/degradation model unless it is necessary to support this strategy work.
</task>

<validation>

After implementation:

1. Run all tests.
2. Run the complete existing benchmark.
3. Compare V4 against V3 using exactly the same benchmark methodology.
4. Run an ablation with the new race-state strategy disabled.
5. Check specifically:
   - first-stop error
   - field stops inside model window
   - oracle regret
   - sequence match
   - start-compound match
   - live stop-call accuracy
   - live tick time

Do not change benchmark definitions to improve V4's numbers.
</validation>

<deliverable>

Create a short file:

`results_v4_task1.md`

Include:
- what changed
- V3 vs Task-1 metrics
- ablation results
- first-stop analysis
- failures and remaining weaknesses

Do not move on to unrelated UI work.
</deliverable>