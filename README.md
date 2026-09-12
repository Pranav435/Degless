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
src/live/auth.py     F1TV token: freshness, silent refresh from the saved browser profile
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
.venv/bin/python -m playwright install chromium   # for the F1TV sign-in
cp .env.example .env            # add GEMINI_API_KEY=... for the AI engineer
make cache && make history      # once; ~2 GB of FastF1 data, then the donor weekends
make run                        # everything else: open http://localhost:8501
```

`make run` is the only command you need on a race weekend. It starts the
dashboard and looks after the weekend by itself: it watches the F1 calendar,
starts the live feed 15 minutes before every session and stops it after,
refits the weekend model after each practice session, scores the race
afterwards, and keeps the outlook for the next race fresh (every 30 minutes
between sessions, every 3 minutes during practice, and right after every
refit). The first tab always shows what is on now, what is next and when -
and, between sessions, the outlook.

It also keeps you signed in to F1TV. Subscription tokens last about four days
and Formula 1 publishes no refresh endpoint, so one lapses most weeks — and it
lapses quietly, because every *timing* topic works without it and only car
telemetry does not. So `make run`:

* checks the token on startup and every 30 minutes, treating "expires within
  12 hours" as already due, so a session never starts on a token that will die
  halfway through it;
* **refreshes it silently** from the Chromium profile under `data/raw/browser`
  — the sign-in cookie there outlives the token, so the usual case costs you
  nothing and you never see a prompt;
* only if that profile has lapsed too, **asks** whether to sign in, and opens
  the window for you. Answer `n`, or say nothing for three minutes, and it
  carries on without telemetry rather than blocking;
* never prompts when nothing is attached to the terminal (a pipe, a service,
  a closed laptop), so an unattended start always proceeds.

`make run` never stops because a part of it failed: a supervisor tick that
raises is logged and retried, the live feed is restarted if it exits, and the
dashboard is restarted up to five times before it is left down with a pointer
to `data/live/logs/app.log` — the feed and refits carry on either way.

Rehearse on a recorded race any time:

```bash
make run REHEARSE=data/raw/livetiming/2026_hungary_race
```

Individual steps remain available (`make weekend`, `make live`, `make postrace`,
`make outlook EVENT=spain-2026`, `make test`; see `make help`).

| command | what it does |
| --- | --- |
| `make run` | everything: dashboard, feed, refits, scoring, outlook, F1TV login |
| `make run NOLOGIN=1` | same, but never prompt for a sign-in |
| `make login` | sign in deliberately (silent refresh first, browser if needed) |
| `make login PASTE=1` | paste the `login-session` cookie instead of opening a browser |
| `make login STATUS=1` | just report what the stored token is worth |

## Running on Windows

There is no `make` on Windows by default, so call the same scripts directly —
the supervisor launches Streamlit as `python -m streamlit` and shuts children
down without POSIX-only signals, so it behaves the same way there.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m playwright install chromium
copy .env.example .env
.venv\Scripts\python scripts\run.py            # the one command; http://localhost:8501
```

The rest map one to one:

| Makefile target | Windows |
| --- | --- |
| `make run` | `.venv\Scripts\python scripts\run.py` |
| `make run NOLOGIN=1` | `.venv\Scripts\python scripts\run.py --no-login` |
| `make run REHEARSE=<dir>` | `.venv\Scripts\python scripts\run.py --rehearse <dir>` |
| `make login` | `.venv\Scripts\python scripts\f1login.py` |
| `make login PASTE=1` | `.venv\Scripts\python scripts\f1login.py --paste` |
| `make cache` | `.venv\Scripts\python scripts\00_cache.py --events australia-2026 japan-2026 barcelona-2026 austria-2026 belgium-2026 hungary-2026 italy-2026` |
| `make weekend EVENT=<k>` | `.venv\Scripts\python scripts\40_weekend.py --event <k>` |
| `make live EVENT=<k>` | `.venv\Scripts\python scripts\50_live.py --event <k>` |
| `make postrace EVENT=<k>` | `.venv\Scripts\python scripts\60_postrace.py --event <k>` |
| `make test` | `.venv\Scripts\python -m pytest tests -q` |

If you would rather keep the Makefile, `make` works under WSL, Git Bash, or
`winget install GnuWin32.Make`. Stop `run.py` with Ctrl-C as usual.

## Live feed: what changed (2026-09-12)

Four defects found during Spain FP3, where the Now tab showed `Aborted` and no
practice data. Three were real bugs; the fourth was the feed being honest about
something it had no way to do.

**The live feed had no session clock.** SignalR messages carry wall clock only
(`t_session` is `None`); the archived `.jsonStream` files carry a session
clock, and `LiveState.apply` only advanced time from the latter. So on every
live session `t_now` stayed `0.0`, every `lap_start_s` was `0`, and the
practice board's traffic filter (`gap_ahead_s > 2.0`, which needs the gap to
the car ahead) discarded every lap. The live long-run board could never
populate. Replaying a real 618-lap FP2 through the live message shape:

| | before | after |
| --- | --- | --- |
| `t_now` | 0.0 | 5042.9 |
| distinct `lap_start_s` | 1 | 618 |
| board rows | **0** | **24** |

This is the trap the replays hid: `RecordedSource` supplies a session clock,
so every test passed while the live path produced nothing. `LiveState.apply`
now derives a session clock from the first message's wall time when the source
carries none, and the two paths agree exactly.

**The weekend model was never reloaded.** The supervisor starts the live feed
and the initial weekend refit in the same tick, so on the first session of a
weekend the feed loads its model seconds before the fit lands and then runs
the whole session on `prior only: no practice fit for this weekend yet` —
ignoring the FP1/FP2 data already on disk. `50_live.py` now watches the fit
artifacts and swaps the model in mid-session, keeping the alerts the engine
has already raised. It deliberately declines mid-race if the race engine has
already built its per-driver posterior, which is sized to the model it started
with.

**Backfill was silently a no-op.** F1 serves `static/<path>/<Topic>.jsonStream`
as `403 AccessDenied` until a session's `ArchiveStatus` reaches `Complete` —
during and for a while after the session, every topic is denied. So
`backfill: 1 archived messages` was 13 topics being refused, and `--source
static` cannot work live at all. The daemon now says so plainly. The late-join
path that *does* work is the SignalR subscribe keyframe, which is rich (~36 KB
of `TimingData`, the full `DriverList`, per-driver stints) and is what
populates the field on a mid-session join.

**An expired F1TV token failed silently.** It is now detected with its expiry
time, refreshed without a prompt where possible, surfaced on the Now tab
("F1TV token expired — make login") and in `data/live/<session>/status.json`,
and explained rather than just dropped. See the login behaviour under Quick
start.

`Aborted` itself was not a bug: FP3 was genuinely red-flagged
(`SESSION WILL BE TEMPORARILY STOPPED`, 11:14:23 UTC) and never restarted.
`Aborted` is Formula 1's own term for a stopped session.

## Status of the 2026 data

All dry, conventional 2026 weekends are cached and scored: Australia, Japan,
Barcelona, Austria, Belgium, Hungary. Sprint weekends (China, Canada, Britain)
are race donors only; Miami, Monaco and Zandvoort were wet and are excluded.
Bahrain and Saudi Arabia are missing upstream. Italy has a pre-race model on
FP1+FP2; its post-race scoring was interrupted and the supervisor re-runs it
at its next start. Spain (Madring, a new circuit with no race history) has a
prior-only outlook built from the 2026 season until practice starts.
