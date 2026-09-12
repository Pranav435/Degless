# Live and historical F1 data sources (verified 2026-09-04, Italian GP weekend)

## What degless uses

| Purpose | Source | Auth | Notes |
|---|---|---|---|
| **Live feed (primary)** | Official F1 live timing, SignalR Core hub `wss://livetiming.formula1.com/signalrcore` | None for timing topics; F1TV token for `CarData.z` / `Position.z` | `src/live/sources.py::SignalRSource`. Uses the token FastF1 stores after its one-time browser login (`fastf1.internals.f1auth`), falls back to no auth. Sub-second latency. |
| **Live fallback** | The same feed's archive files, polled by byte range: `https://livetiming.formula1.com/static/<path>/<Topic>.jsonStream` | None | `StaticPollSource`. Written by F1 as the session runs; CDN latency unverified. Also used to **backfill** a late join. |
| **History / fitting** | FastF1 3.8.3 (laps, telemetry, weather, track status, circuit corners) | None | Cache under `data/raw/fastf1_cache`. All dry 2026 weekends cached. |
| **History cross-check** | OpenF1 REST (`api.openf1.org/v1`) | None for data ≥30 min after a session | Same canonical lap schema through `src/ingest.py`. Session keys in `data/raw/openf1_sessions_2026.json`. |
| **Replay / tests** | Archived `.jsonStream` files under `data/raw/livetiming/<session>/` | None | `RecordedSource` replays a whole session through the identical code path the live feed uses. |

## Options considered

1. **F1 SignalR Core (chosen).** Protocol moved from `/signalr` to `/signalrcore` in May 2025 (old hub → 401). Unauthenticated `Subscribe` returns `TimingData`, `TimingAppData`, `TyreStintSeries`, `TrackStatus`, `WeatherData`, `RaceControlMessages`, `LapCount`, `DriverList`, `PitLaneTimeCollection`, … Car telemetry and positions are withheld without an F1TV Access/Pro/Premium token. The stream list and `TimingData` keys are unchanged from 2025; `CarData.z` lost channel 45 (DRS) in 2026.
2. **OpenF1 sponsor tier** (€9.90/month): live REST + MQTT/WebSocket (`mqtt.openf1.org:8883`), ~3 s latency, 6 req/s. Clean JSON, but a middleman that itself depends on an F1TV token. Free tier is historical only.
3. **Unauthenticated static-stream polling**: free, same data, latency unknown; kept as the fallback.
4. **FastF1 `livetiming` client**: record-only; its auth flow and `signalrcore` usage are what `SignalRSource` reuses.
5. **livef1 (PyPI)**: live path hardcodes the dead `/signalr/` hub; historical path fine. Not used.
6. **Commercial** (Sportradar, Sportmonks €69–79/month, API-Sports): no telemetry, thin or unverifiable tyre data. Not used.
7. **Jolpica** (Ergast successor): results only, no live.
8. **MultiViewer local GraphQL**: a personal proxy tied to a GUI app. Not used.

## Feed facts the parser relies on

* Messages are patches; `src/live/merge.py` deep-merges them (dict-by-index into lists, `_deleted` keys).
* Lap boundaries: `TimingData.Lines[n].NumberOfLaps` increments; `LastLapTime`, sector 3 and speed traps arriving within 5 s belong to the lap just completed (FastF1's rule). `InPit` true/false marks in-/out-laps.
* Stints: `TimingAppData.Lines[n].Stints[i]` gives `Compound`, `New`, `StartLaps` (age when fitted), `TotalLaps`.
* Track status codes: 1 clear, 2 yellow, 4 safety car, 5 red, 6 VSC, 7 VSC ending.
* `CarData.z` / `Position.z`: base64 + raw deflate JSON; channels 0 RPM, 2 speed, 3 gear, 4 throttle, 5 brake.
* Timestamps: SignalR messages carry UTC; archive lines carry a session clock; `Heartbeat.Utc` links the two.
