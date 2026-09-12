# degless — tyre degradation intelligence, live

Practice data in, race calls out, error bars all the way down — and now on the
pit wall in real time.

* **Offline**: fit a hierarchical Bayesian tyre model on practice long runs
  (fuel-mass physics prior, isotonic track evolution, corner apex speeds as a
  second channel, compound ladder), seal it before the race, score it after.
* **Live**: consume the official F1 live timing feed, keep a lap-level view of
  the session, and turn the sealed model into decisions every lap: cliff
  probability, best remaining plan, pit window, box-now cost, undercut threats,
  safety-car stops, rejoin position.
* **Outlook**: between sessions the tool is not idle.  From the compound
  ladder, the circuit's previous races (or the 2026 season, for a circuit
  nobody has raced), the transferred regime, pit-loss and allocation priors
  and the live long-run board while practice runs, it keeps a strategy
  picture for the *next* race - plan, stop-count probabilities, pit windows,
  tyre lives, plan B and its switch trigger, a scenario matrix, a safety-car
  playbook and the practice programme that would narrow the decision - and
  rebuilds it as every new piece of data arrives, charting how it moved.
* **Strategy desk**: build up to three plans, price them on the same posterior
  draws, stress-test them (degradation, pit lane, safety car), run the
  undercut/overcut calculator, read the safety-car playbook, and commit a
  decision card per driver that the live race view then tracks the car
  against.
* **App**: one Streamlit dashboard — Now tab first, then the Strategy desk,
  the race plan, the tyre model, the evidence, the validation receipts, the
  replay, and an AI race engineer grounded strictly on the model's numbers
  (the outlook and the committed plans included).

See `docs/RUNBOOK.md` for race-day operation and `docs/DATA_SOURCES.md` for
the data-source research.

## Layout

```
src/                 model (config, ingest, laps, fuel, evolution, telemetry, model_bayes,
                     model_fallback, compounds, regime, tyre, strategy, validate, replay, engineer)
src/outlook.py       the next race's strategy picture from everything known so far, kept fresh
src/plans.py         committed decision cards (data/live/plans/<event>.json)
src/live/            live layer: streams, merge, state, sources, engine, store
scripts/00_cache.py  pre-cache FastF1 sessions (telemetry included)
scripts/10_pipeline.py   full retrospective pipeline on a weekend with a race (fit + seal + score)
scripts/40_weekend.py    pre-race model: fit on practice so far, seal, posterior for the live engine
scripts/50_live.py       the live daemon (SignalR / static polling / recorded replay)
scripts/60_postrace.py   score the sealed model and the live engine's calls after the flag
scripts/70_outlook.py    build the outlook (the supervisor runs it by itself)
app/dashboard.py     Streamlit; app/live_tab.py is the Now tab, app/desk_tab.py the Strategy desk
tests/               parser vs FastF1, engine replay
data/raw/            FastF1 cache, archived livetiming streams, OpenF1 session index
data/processed/      fits, posteriors, plans, sealed metadata
data/live/<session>/ snapshots the daemon writes and the app reads
predictions/sealed/  frozen pre-race predictions + sha256
```

## Quick start

```bash
/opt/homebrew/bin/python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY=... for the AI engineer
make cache && make history      # once; ~2 GB of FastF1 data, then the donor weekends
make run                        # everything else: open http://localhost:8501
```

`make run` starts the dashboard and looks after the weekend by itself: it
watches the F1 calendar, starts the live feed 15 minutes before every session
and stops it after, refits the weekend model after each practice session,
scores the race afterwards, and keeps the outlook for the next race fresh
(every 30 minutes between sessions, every 3 minutes during practice, and
right after every refit). The first tab always shows what is on now, what is
next and when - and, between sessions, the outlook. Rehearse on a recorded
race any time:

```bash
make run REHEARSE=data/raw/livetiming/2026_hungary_race
```

Individual steps remain available (`make weekend`, `make live`, `make postrace`,
`make outlook EVENT=spain-2026`, `make test`; see `make help`).

## Status of the 2026 data

All dry, conventional 2026 weekends are cached and scored: Australia, Japan,
Barcelona, Austria, Belgium, Hungary. Sprint weekends (China, Canada, Britain)
are race donors only; Miami, Monaco and Zandvoort were wet and are excluded.
Bahrain and Saudi Arabia are missing upstream. Italy has a pre-race model on
FP1+FP2; its post-race scoring was interrupted and the supervisor re-runs it
at its next start. Spain (Madring, a new circuit with no race history) has a
prior-only outlook built from the 2026 season until practice starts.
