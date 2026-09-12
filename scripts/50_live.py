"""The live daemon: feed in, decisions out, snapshots to disk for the app.

    # during a session, official feed (F1TV token used if FastF1 has one):
    .venv/bin/python scripts/50_live.py --event italy-2026

    # replay an archived session at 20x real time through the identical path:
    .venv/bin/python scripts/50_live.py --event hungary-2026 --source recorded \
        --session-dir data/raw/livetiming/2026_hungary_race --speed 20

    # replay instantly (tests, post-session analysis):
    .venv/bin/python scripts/50_live.py --event hungary-2026 --source recorded \
        --session-dir data/raw/livetiming/2026_hungary_race --speed 0

    # free fallback if the hub is down: poll the archive files as they are written
    .venv/bin/python scripts/50_live.py --event italy-2026 --source static

Every message from a live source is also recorded to
data/live/<session>/feed.jsonl in the format FastF1's `LiveTimingData` reads.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_PROCESSED, current_event, get_event  # noqa: E402
from src.live.engine import PracticeEngine, RaceEngine, WeekendModel  # noqa: E402
from src.live.sources import (  # noqa: E402
    JsonlSource, RecordedSource, Recorder, SignalRSource, StaticPollSource,
    current_session_info, download_session, session_path_from_info,
)
from src.live.state import LiveState  # noqa: E402
from src.live.store import LIVE_DIR, SnapshotStore  # noqa: E402

log = logging.getLogger("degless.live")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default=None, help="event key (default: the current weekend)")
    ap.add_argument("--source", default="signalr", choices=["signalr", "static", "recorded", "jsonl"])
    ap.add_argument("--session-dir", default=None, help="recorded: directory of .jsonStream files")
    ap.add_argument("--file", default=None, help="jsonl: recording to replay")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed; 0 = instant")
    ap.add_argument("--session-type", default=None, help="Race | Practice (default: from the feed)")
    ap.add_argument("--tick", type=float, default=2.0, help="seconds between engine ticks")
    ap.add_argument("--no-auth", action="store_true", help="signalr: do not use the F1TV token")
    ap.add_argument("--no-backfill", action="store_true",
                    help="signalr: do not replay the archive files before going live")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--session-key", default=None, help="name of the data/live/<key> directory")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this many wall seconds")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ev = get_event(args.event) if args.event else current_event()
    wm = WeekendModel.load(ev)
    print(f"event: {ev.name} ({ev.n_race_laps} laps)\nmodel: {wm.source}\n"
          f"pit loss prior: {wm.pit_loss_s:.1f} s ({wm.pit_loss_source})", flush=True)

    def model_stamp() -> tuple:
        """Modification times of everything `WeekendModel.load` reads."""
        out = []
        for p in (DATA_PROCESSED / f"posterior_{ev.key}.npz",
                  DATA_PROCESSED / f"weekend_{ev.key}.json",
                  DATA_PROCESSED / f"meta_{ev.key}.json"):
            try:
                out.append(p.stat().st_mtime)
            except OSError:
                out.append(0.0)
        return tuple(out)

    model_seen = model_stamp()

    def reload_model() -> None:
        """A refit landed while we were running: pick it up.

        The supervisor starts this daemon and the weekend refit in the same
        tick, so on the first session of a weekend the fit lands seconds after
        we have already loaded the prior-only model.  Without this the whole
        session runs on the prior and ignores the practice already on disk.
        """
        nonlocal wm, engine
        try:
            fresh = WeekendModel.load(ev)
        except Exception:
            log.exception("could not reload the weekend model")
            return
        if fresh.source == wm.source:
            return
        if isinstance(engine, RaceEngine) and engine.tyres:
            # Mid-race the engine carries a per-driver posterior sized to the
            # model it started with; swapping it out would throw that away.
            log.warning("a new fit landed mid-race; keeping the running model")
            return
        wm = fresh
        print(f"model reloaded: {wm.source}", flush=True)
        if engine is not None:
            new = type(engine)(wm)
            for attr in ("alerts", "_seen", "tick_no"):   # keep what it has already said
                if hasattr(engine, attr):
                    setattr(new, attr, getattr(engine, attr))
            engine = new

    state = LiveState(session_type=args.session_type)
    engine = None
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    # -- source -----------------------------------------------------------
    recorder = None
    speed = None if args.speed == 0 else args.speed
    session_key = args.session_key
    backfill_msgs = []
    if args.source == "recorded":
        src = RecordedSource(args.session_dir, speed=speed)
        session_key = session_key or Path(args.session_dir).name
    elif args.source == "jsonl":
        src = JsonlSource(args.file, speed=speed)
        session_key = session_key or Path(args.file).stem
    elif args.source == "static":
        src = StaticPollSource(interval_s=max(args.tick, 3.0))
        try:
            info = current_session_info()
            session_key = session_key or str(info.get("Key") or "static")
        except Exception:
            session_key = session_key or "static"
    else:
        try:
            info = current_session_info()
            path = session_path_from_info(info)
            session_key = session_key or str(info.get("Key") or "live")
            print(f"current session: {info.get('Meeting', {}).get('Name')} — {info.get('Name')} "
                  f"({info.get('Type')}), path {path}", flush=True)
            archive = (info.get("ArchiveStatus") or {}).get("Status", "")
            if archive == "Complete":
                print("that session is finished; waiting for the next one (no backfill)", flush=True)
            if not args.no_backfill and path and archive != "Complete":
                bdir = LIVE_DIR / str(session_key) / "backfill"
                got = download_session(path, bdir, topics=[
                    "SessionInfo", "SessionStatus", "SessionData", "DriverList", "TrackStatus",
                    "LapCount", "TimingData", "TimingAppData", "TyreStintSeries", "WeatherData",
                    "RaceControlMessages", "Heartbeat", "ExtrapolatedClock"])
                denied = got.pop("_denied", 0)
                if denied and not any(k != "SessionInfo" for k in got):
                    # F1 publishes the per-topic files only once the session's
                    # archive is complete; until then every one of them is 403.
                    print(f"backfill: this session's archive is not published yet "
                          f"({denied} topics denied); the live keyframe carries the "
                          f"current state instead", flush=True)
                elif got:
                    backfill_msgs = RecordedSource(bdir).messages()
                    print(f"backfill: {len(backfill_msgs)} archived messages "
                          f"({sum(got.values())/1e6:.1f} MB)", flush=True)
        except Exception as exc:
            log.warning("could not read the current session info: %s", exc)
            session_key = session_key or "live"
        src = SignalRSource(use_auth=not args.no_auth)
        if not args.no_record:
            recorder = Recorder(LIVE_DIR / str(session_key) / "feed.jsonl")
    if args.max_seconds and hasattr(src, "deadline"):
        src.deadline = time.time() + args.max_seconds
    store = SnapshotStore(session_key)
    print(f"writing snapshots to {store.dir}", flush=True)

    # -- loop -------------------------------------------------------------
    t_start = time.time()
    last_tick = 0.0
    n = 0
    status = {"source": args.source, "event": ev.key, "session_key": session_key,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "auth": bool(getattr(src, "token", None)),
              "auth_status": getattr(src, "auth_status", "n/a"),
              "auth_detail": getattr(src, "auth_detail", ""), "messages": 0}
    if status["auth_status"] in ("expired", "invalid", "none"):
        print(f"F1TV: {status['auth_detail']} — timing is unaffected, "
              f"car telemetry is not available", flush=True)

    instant = speed is None and args.source in ("recorded", "jsonl")

    def do_tick(force: bool = False):
        nonlocal engine, last_tick, state, store, model_seen
        now = time.time()
        if instant and not force:
            return          # instant replay: only the lap-driven ticks
        if not force and now - last_tick < args.tick:
            return
        last_tick = now
        stamp = model_stamp()
        if stamp != model_seen:
            model_seen = stamp
            reload_model()
        if engine is None:
            if state.session_type is None and state.n_messages < 50:
                return
            engine = RaceEngine(wm) if state.is_race else PracticeEngine(wm)
            print(f"engine: {type(engine).__name__} (session type {state.session_type})", flush=True)
        try:
            snap = engine.tick(state)
        except Exception:
            log.exception("engine tick failed")
            return
        store.write_snapshot(snap)
        store.write_laps(state.laps_df(include_current=True))
        status.update(messages=n, last_message_utc=(state.utc_now.isoformat() if state.utc_now else None),
                      connected=bool(getattr(src, "connected", True)), t_session=state.t_now,
                      session_status=state.session_status, engine=type(engine).__name__,
                      model_source=wm.source,
                      n_laps=int(sum(t.n_complete for t in state.tracks.values())))
        store.write_status(status)
        if not args.quiet:
            lc = state.lap_count
            lead = next((r for r in snap.get("field", []) if r.get("position") == 1), None)
            extra = ""
            if snap.get("engine") == "race" and lead:
                extra = (f" | P1 {lead.get('driver')} {lead.get('compound')} age {lead.get('tyre_age')} "
                         f"wear {lead.get('wear', float('nan')):.2f} plan: {(lead.get('plan') or {}).get('best')}")
            print(f"[{state.t_now:8.1f}s] {state.session_status:9s} status {state.track_status} "
                  f"lap {lc.get('current')}/{lc.get('total')} msgs {n} laps "
                  f"{status['n_laps']} alerts {len(snap.get('alerts', []))}{extra}", flush=True)

    try:
        for m in backfill_msgs:
            state.apply(m)
            n += 1
        if backfill_msgs:
            do_tick(force=True)
        for m in src:
            if stop["flag"]:
                break
            if m.topic == "SessionInfo" and isinstance(m.payload, dict) and m.payload.get("Key"):
                new_key = str(m.payload["Key"])
                old_key = str(state.session_info.get("Key") or "")
                if old_key and new_key != old_key:
                    # A new session started on the same connection: start clean.
                    print(f"session changed {old_key} -> {new_key}: resetting", flush=True)
                    state = LiveState(session_type=args.session_type)
                    engine = None
                    session_key = new_key
                    store = SnapshotStore(session_key)
                    status.update(session_key=session_key)
                    if recorder is not None:
                        recorder.close()
                        recorder = Recorder(LIVE_DIR / str(session_key) / "feed.jsonl")
            state.apply(m)
            n += 1
            if recorder is not None:
                recorder.write(m)
            evs = state.drain_events()
            lap_event = any(k in ("lap", "pit_in", "pit_out", "track_status", "session_status")
                            for _, k, _ in evs)
            do_tick(force=lap_event and (instant or (time.time() - last_tick) > 0.5))
            if args.max_seconds and time.time() - t_start > args.max_seconds:
                break
        do_tick(force=True)
    finally:
        if recorder is not None:
            recorder.close()
        if hasattr(src, "stop"):
            src.stop()
    print(f"done: {n} messages, {status.get('n_laps')} laps", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
