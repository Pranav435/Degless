# Runbook

## The one command

```bash
make run
```

Then open http://localhost:8501. That is all there is to operate.

What `make run` does (`scripts/run.py`):

* starts the dashboard;
* watches the F1 calendar (every session of the year: practice, sprint
  qualifying, sprints, qualifying, races) and shows what is live now, what is
  next and when, and the whole weekend's timetable at the top of every page;
* 15 minutes before any session it starts the live feed for the right weekend,
  backfills anything already run, and stops the feed 25 minutes after the
  scheduled end (or when the session status says it is over);
* after every practice session it refits the weekend model on all practice so
  far, so the race engine always runs on the freshest sealed model;
* about 75 minutes after a race it scores the sealed prediction and the live
  engine's own calls;
* keeps the **outlook** for the next race fresh: every 30 minutes between
  sessions, every 3 minutes while a practice session is live (the live
  long-run board is folded in), and immediately after every refit;
* writes what it is doing to `data/live/supervisor.json`, which the app shows
  under the status line.

Leave it running for the weekend. Ctrl-C stops everything cleanly.

## First time

```bash
/opt/homebrew/bin/python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # put GEMINI_API_KEY=... in it for the AI engineer
make cache                # FastF1 cache of the dry 2026 weekends (~2 GB)
make history              # donor weekends for the transferred priors
make login                # optional: F1TV login → car telemetry on the live feed
```

The live feed needs no login for timing data; the F1TV token only adds car
telemetry and positions.

## What the app shows

**Haas** (first tab): what #31 Esteban Ocon and #87 Ollie Bearman should each
do right now, with a driver-view switch — Haas Overview / Ocon / Bearman.
Haas Overview leads with a panel per car: position, compound, tyre age,
current pace, degradation, the gaps ahead and behind, the recommended action,
pit window, confidence, projected rejoin position, expected race-time delta
and the relevant rival. Below it, the Haas Pit Wall states the call for each
car (PIT NOW / STAY OUT / WAIT k laps / BOX BY LAP x) with its confidence,
projected position and expected delta, then the Why block underneath it —
the principal reason and two to four supporting ones, e.g. "Staying out 2
laps costs 1.4 s of tyre; the car behind is entering its window; pitting now
is projected to retain P15 (78% confidence)". Every one of those sentences is
built from a number already in the model's state; nothing here is a stock
phrase, and a value the state does not carry prints as "—" rather than being
guessed at. Race field lists both cars and their strategic rivals with
position, gaps, compounds, tyre age and pit status; Strategy comparison lines
up the 1-, 2- and 3-stop candidates with their expected race time, expected
position change through the first pit cycle, pit laps, compound sequence and
uncertainty. The Ocon and Bearman views are the same panel and Why block for
one car, plus its live lap chart when a session is on.

Live, every number comes from that lap's snapshot — the field row for OCO or
BEA and the race-execution engine's `decision` for it. Before a session goes
live, the same panels read the pre-race plan instead (the car's own plan, the
team's pit window and recommended lap, and — once the pipeline has written
it — the per-car model of pace, degradation and warm-up), and say so under
each panel's title. Curves, tyre-pack constants and the full search stay in
the other tabs; the Haas tab's Advanced expander only links to them.

**Now**: the status line and weekend timetable; when a session is
live, the live view for it — during practice the long-run degradation board,
during qualifying the timing, during a race the pit wall: per car tyre age,
live degradation, wear, cliff probability, laps to the cliff, best remaining
plan, pit window, box-now cost, undercut threat and opportunity, alerts, the
committed plans tracked against the race (next planned stop and window, on
the planned compound or not, live degradation against the switch trigger),
and per driver the lap-time chart with the projection, the in-lap cost curve,
the option table and the rejoin estimate. Between sessions: the countdown and
the **outlook** for the next race — the recommended plan and the probability
of each stop count, the pit windows, tyre life per compound, plan B and the
degradation multiplier at which it takes over, the least-regret plan across
the scenario matrix, what to run in practice to narrow the decision, the
safety-car playbook, and two charts of how all of it has moved as data came in.

**Strategy desk**: the actionables. (1) Build up to three plans (compounds,
stop laps, push) and price them side by side on the same posterior draws:
expected loss against the best, its 5–95% band, P(fastest), the wear each
stint ends at, and the lap-by-lap crossover chart. (2) Stress-test them with
one row of controls — degradation multiplier (shown as the equivalent track
temperature), pit-lane time, safety-car likelihood — next to the outlook's
precomputed matrix. (3) The undercut/overcut calculator for a specific duel,
both directions. (4) The safety-car playbook for any plan: box or stay, lap by
lap, with the seconds at stake. (5) The decision card: windows, the wear to
expect at each stop, undercut exposure, the switch triggers to the other
plans, the safety-car rule; commit it for a driver (or as the team default)
and download it as text. (6) The practice programme: which compound's long
run is worth the most to the decision.

The outlook is the same search the sealed model runs; before any practice it
runs on the compound ladder combined with the circuit's race history (or the
2026 season for a new circuit) and the transferred priors, with the
uncertainty that implies. It never reads race data for the target weekend.

**Race sim**: the whole race, lap by lap, with the engine on the wall for
both cars, on a known truth.  Pick a scenario (the forecast tyre, a safety car
on lap 22, tyres wearing twice or five times the forecast, slower tyres, other
tyres the model thinks possible).  The headline gives each car's grid and
finish with the engine against the sealed plan followed blindly; the race
trace and classification show the whole field; per car, the call lap by lap
(the lap it named, the window it priced, the lap it stopped), what it believed
about the tyre against the truth, that car's own tyre degradation set by set
(what each set really gave up per lap on this car, how much of its life had
gone by the end of the stint, and what the engine had read off the lap times by
then) next to what each compound would have done on it, positions against its
rivals, the reasons it gave on the in-lap, and every lap's record.  The
real-time card times every lap's decision for the whole field.  `make racesim
EVENT=<key>` rebuilds it (`QUICK=1` for the base scenario); `make run` does so
after every refit, and commits the model's per-car decision cards (`make plans
EVENT=<key>`) at the same time.

**Race plan / Tyre model / Evidence**: the sealed weekend model and how it was
built (for a weekend with no practice yet, the Race plan tab shows the
outlook's plan). **Validation / Replay**: the receipts, once the race has been
scored. **Engineer**: Haas's tyre-degradation race engineer for #31 Ocon and
#87 Bearman — ask questions, or take the briefing, and it leads on what each
car's tyres are doing before anything about track position. Answers are
grounded on the model's numbers, each car's own degradation rates and the
evidence behind them, the outlook, the committed plans and the live snapshot;
it never averages the two drivers into one Haas number, and it says when a
car's rate is really its team-mate's. Needs `GEMINI_API_KEY` in `.env`
(`GEMINI_MODEL` and `ENGINEER_MAX_TOKENS` override the model and the per-answer
output budget). Without a key, or if the provider is busy, the same briefing is
assembled offline from the identical fact sheet and says which of the two it
is.

## Rehearsal, any time

```bash
make run REHEARSE=data/raw/livetiming/2026_hungary_race
```

Presents the recorded Hungary race as live at 20x, through the identical code
path, so the whole tool can be watched end to end without a session on.

## If something breaks

* Feed logs: `data/live/logs/feed_<session>.log`; refit and scoring logs are
  next to them. The status line says which task is running and the last event.
* If the official hub is down: `make live-static EVENT=<key>` polls F1's
  archive files instead (free, higher latency); the app does not care which
  source is running.
* `make test` runs the parser-vs-FastF1 and engine replay tests, and the
  outlook, plan-tool and plan-store tests.
* `make outlook EVENT=<key>` rebuilds the outlook by hand (`SESSION=<key>` to
  fold a live practice board in); its log is `data/live/logs/outlook_<key>.log`.
