"""degless, the whole tool, from one command:

    .venv/bin/python scripts/run.py          (or: make run)

It starts the dashboard and then looks after the weekend by itself:

* watches the F1 calendar; when a session goes live it starts the live feed
  daemon for the right weekend, and stops it when the session ends;
* after every practice session it refits the weekend model on all the practice
  run so far, so the race engine always has the freshest sealed model;
* after a race it scores the sealed prediction and the live engine's calls once
  the archive data is available;
* keeps the **outlook** for the next race fresh (`scripts/70_outlook.py`):
  every 30 minutes between sessions, every 3 minutes while a practice session
  is live (folding the live long-run board in), and immediately after every
  refit - so the strategy picture on the Now tab and the Strategy desk always
  reflects everything known so far;
* keeps `data/live/supervisor.json` up to date so the app can show what it is
  doing, what is live, and what is next.

Nothing here needs a person to press anything at the right time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import schedule  # noqa: E402
from src.config import DATA_PROCESSED  # noqa: E402
from src.live.store import LIVE_DIR, atomic_write  # noqa: E402

log = logging.getLogger("degless.run")
PY = sys.executable
LOG_DIR = LIVE_DIR / "logs"
STATE_FILE = LIVE_DIR / "supervisor.json"


class Job:
    """A child process with a log file."""

    def __init__(self, name: str, cmd: list, log_path: Path, quiet: bool = False):
        self.name = name
        self.cmd = cmd
        self.log_path = log_path
        self.quiet = quiet
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(log_path, "a")
        self.proc = subprocess.Popen(cmd, stdout=self._f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        self.started = time.time()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, grace: float = 10.0) -> None:
        if not self.alive():
            return
        self.proc.send_signal(signal.SIGINT)
        t0 = time.time()
        while self.alive() and time.time() - t0 < grace:
            time.sleep(0.2)
        if self.alive():
            self.proc.kill()
        try:
            self._f.close()
        except Exception:
            pass

    def as_dict(self) -> dict:
        return {"name": self.name, "alive": self.alive(), "returncode": self.proc.poll(),
                "started": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
                "log": str(self.log_path.relative_to(ROOT))}


class Supervisor:
    def __init__(self, *, app: bool = True, port: int = 8501, poll_s: float = 20.0,
                 rehearse: str | None = None, speed: float = 20.0):
        self.app = app
        self.rehearse = Path(rehearse) if rehearse else None
        self.speed = speed
        self.port = port
        self.poll_s = poll_s
        self.feed: Job | None = None
        self.feed_session: int | None = None
        self.tasks: dict[str, Job] = {}       # name -> job (refits, scoring)
        self.done: set = set()                 # task names already run
        self._outlook_last: dict = {}          # event_key -> (time, posterior mtime, session key)
        self._outlook_target: tuple = (None, None)
        self.history: list = []
        self.streamlit: Job | None = None
        self._stop = False

    # -- helpers ------------------------------------------------------------

    def _note(self, text: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{stamp}] {text}", flush=True)
        self.history.append({"utc": datetime.now(timezone.utc).isoformat(), "text": text})
        self.history = self.history[-60:]

    def _write_state(self, st: dict) -> None:
        state = {
            "updated": datetime.now(timezone.utc).isoformat(),
            "schedule": st, "headline": schedule.describe(st),
            "feed": self.feed.as_dict() | {"session": self.feed_session} if self.feed else None,
            "tasks": {k: v.as_dict() for k, v in self.tasks.items()},
            "done": sorted(self.done), "history": self.history[-20:],
            "app_url": f"http://localhost:{self.port}" if self.app else None,
            "outlook": self._outlook_state(),
            "pid": os.getpid(),
        }
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write(STATE_FILE, json.dumps(state, default=str))

    def _has_weekend_model(self, event_key: str) -> bool:
        return (DATA_PROCESSED / f"posterior_{event_key}.npz").exists()

    # -- actions ------------------------------------------------------------

    def start_feed(self, s: dict) -> None:
        ev = s.get("event_key")
        cmd = [PY, str(ROOT / "scripts" / "50_live.py"), "--session-key", str(s["key"]), "--quiet"]
        if ev:
            cmd += ["--event", ev]
        if s.get("rehearsal"):
            cmd += ["--source", "recorded", "--session-dir", str(self.rehearse), "--speed", str(self.speed)]
        self.feed = Job(f"feed:{s['label']}", cmd, LOG_DIR / f"feed_{s['key']}.log")
        self.feed_session = s["key"]
        self._note(f"feed started for {s['label']} (session {s['key']})")

    def stop_feed(self, why: str) -> None:
        if self.feed is not None:
            self.feed.stop()
            self._note(f"feed stopped ({why})")
        self.feed = None
        self.feed_session = None

    def run_task(self, name: str, cmd: list, *, quiet: bool = False) -> None:
        if name in self.tasks and self.tasks[name].alive():
            return
        self.tasks[name] = Job(name, cmd, LOG_DIR / f"{name.replace(':', '_')}.log", quiet=quiet)
        if not quiet:
            self._note(f"started {name}")

    def _reap(self) -> None:
        for name, job in list(self.tasks.items()):
            if not job.alive():
                rc = job.proc.poll()
                self.done.add(name)
                if not job.quiet or rc != 0:
                    self._note(f"{name} finished (exit {rc})")
                del self.tasks[name]

    # -- the outlook: the next race's strategy picture, kept fresh ------------

    def _outlook_target_for(self, st: dict) -> tuple:
        """`(event_key, practice session dict or None)`: the weekend whose
        outlook to keep fresh, and the practice session whose live board the
        sealed model does not yet include (live, or finished but not refitted)."""
        live, nxt = st.get("live"), st.get("next")
        if live and live.get("is_race"):
            return None, None          # the race engine owns the race
        focus = live or nxt
        ek = focus.get("event_key") if focus else None
        if not ek:
            return None, None
        post = DATA_PROCESSED / f"posterior_{ek}.npz"
        sess = None
        for s in st.get("weekend", []):
            if s.get("event_key") != ek or s.get("type") != "Practice" or s.get("state") == "upcoming":
                continue
            end_ts = datetime.fromisoformat(s["end"]).timestamp()
            if s["state"] == "live" or not post.exists() or post.stat().st_mtime < end_ts:
                sess = s
        return ek, sess

    def maybe_outlook(self, st: dict) -> None:
        ek, sess = self._outlook_target_for(st)
        self._outlook_target = (ek, sess["key"] if sess else None)
        if not ek:
            return
        name = f"outlook:{ek}"
        if name in self.tasks and self.tasks[name].alive():
            return
        live_practice = bool(sess and sess["state"] == "live" and self.feed is not None)
        every = OUTLOOK_LIVE_S if live_practice else OUTLOOK_IDLE_S
        post = DATA_PROCESSED / f"posterior_{ek}.npz"
        post_m = post.stat().st_mtime if post.exists() else 0.0
        sk = sess["key"] if sess else None
        last = self._outlook_last.get(ek)
        out = DATA_PROCESSED / f"outlook_{ek}.json"
        if last is None:
            # first sight of this weekend since the supervisor started: build
            # unless a fresh outlook is already on disk
            due = (not out.exists()) or time.time() - out.stat().st_mtime > every
        else:
            due = (time.time() - last[0] >= every          # the cadence
                   or last[1] != post_m                    # a refit landed: rebuild now
                   or (sk is not None and last[2] != sk))  # a new practice session's board to fold in
        if not due:
            return
        cmd = [PY, str(ROOT / "scripts" / "70_outlook.py"), "--event", ek]
        if sk is not None and (LIVE_DIR / str(sk) / "snapshot.json").exists():
            cmd += ["--session", str(sk)]
        self.run_task(name, cmd, quiet=True)
        self._outlook_last[ek] = (time.time(), post_m, sk)

    def _outlook_state(self) -> dict:
        ek, sk = self._outlook_target
        out = {"event": ek, "session": sk, "running": any(n.startswith("outlook:") and j.alive()
                                                            for n, j in self.tasks.items())}
        if ek:
            p = DATA_PROCESSED / f"outlook_{ek}.json"
            if p.exists():
                try:
                    o = json.loads(p.read_text())
                    st = o.get("strategy") or {}
                    out.update({"updated": o.get("updated_utc"), "stage": o.get("stage_label"),
                                "best": st.get("best"), "p_stops": st.get("p_stops")})
                except Exception:
                    pass
            last = self._outlook_last.get(ek)
            if last:
                out["next_in_s"] = max(0.0, (OUTLOOK_LIVE_S if sk else OUTLOOK_IDLE_S) - (time.time() - last[0]))
        return out

    # -- the loop -----------------------------------------------------------

    def _rehearsal_session(self) -> dict | None:
        """A recorded session presented as live, so the whole tool can be
        exercised on a quiet Tuesday: `make run REHEARSE=data/raw/livetiming/2026_hungary_race`."""
        if self.rehearse is None:
            return None
        if self.feed is not None and not self.feed.alive():
            return None   # replay finished: back to the real calendar
        name = self.rehearse.name                      # e.g. 2026_hungary_race
        parts = name.split("_")
        ev = next((k for k in schedule.EVENTS if len(parts) > 1 and k.startswith(parts[1])), None)
        now = datetime.now(timezone.utc)
        return {"key": f"rehearsal_{name}", "meeting_key": 0, "name": parts[-1].upper() if parts else "REPLAY",
                "type": "Race" if "race" in name else "Practice", "start": now.isoformat(),
                "end": now.isoformat(), "circuit": name, "country": "", "event_key": ev,
                "label": f"REHEARSAL · {name} at {self.speed:.0f}x", "state": "live", "is_race": "race" in name,
                "minutes_to_start": -1.0, "minutes_to_end": 999.0, "rehearsal": True}

    def tick(self) -> None:
        sessions = schedule.load()
        st = schedule.status(sessions=sessions)
        reh = self._rehearsal_session()
        if reh is not None:
            st["live"] = reh
            st["event_key"] = reh["event_key"] or st.get("event_key")
        live = st.get("live")
        # 1. feed
        if live and (self.feed is None or not self.feed.alive() or self.feed_session != live["key"]):
            if self.feed is not None and self.feed_session != live["key"]:
                self.stop_feed("session changed")
            elif self.feed is not None and not self.feed.alive():
                self._note("feed exited; restarting")
            self.start_feed(live)
        if not live and self.feed is not None:
            self.stop_feed("session over")
        # 2. after a practice session: refit the weekend model
        last = st.get("last")
        if last and last.get("event_key") and last["type"] == "Practice":
            name = f"weekend:{last['event_key']}:{last['key']}"
            post = DATA_PROCESSED / f"posterior_{last['event_key']}.npz"
            end_ts = datetime.fromisoformat(last["end"]).timestamp()
            fresh = post.exists() and post.stat().st_mtime > end_ts
            if fresh:
                self.done.add(name)
            elif name not in self.done and name not in self.tasks and last["minutes_to_end"] < -LIVE_TAIL_MIN_REFIT:
                self.run_task(name, [PY, str(ROOT / "scripts" / "40_weekend.py"), "--event", last["event_key"]])
        # 3. a weekend with no model yet but practice already run: fit now
        ek = st.get("event_key")
        if ek and not self._has_weekend_model(ek):
            ran = [s for s in st["weekend"] if s["type"] == "Practice" and s["state"] == "finished"]
            name = f"weekend:{ek}:initial"
            if ran and name not in self.done and name not in self.tasks:
                self.run_task(name, [PY, str(ROOT / "scripts" / "40_weekend.py"), "--event", ek])
        # 4. after a race: score it (the archive lags the flag by ~an hour)
        if last and last.get("event_key") and last["name"] == "Race" and last["minutes_to_end"] < -75:
            name = f"postrace:{last['event_key']}"
            if name not in self.done and name not in self.tasks:
                self.run_task(name, [PY, str(ROOT / "scripts" / "60_postrace.py"), "--event", last["event_key"]])
        # 5. the outlook for the next race, from everything known so far
        self.maybe_outlook(st)
        self._reap()
        self._write_state(st)

    def run(self) -> int:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        if self.app:
            self.streamlit = Job("app", [str(ROOT / ".venv" / "bin" / "streamlit"), "run",
                                         str(ROOT / "app" / "dashboard.py"), "--server.port", str(self.port),
                                         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                                 LOG_DIR / "app.log")
            self._note(f"dashboard at http://localhost:{self.port}")
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_stop", True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop", True))
        try:
            while not self._stop:
                try:
                    self.tick()
                except Exception:
                    log.exception("supervisor tick failed")
                for _ in range(int(self.poll_s * 5)):
                    if self._stop:
                        break
                    time.sleep(0.2)
        finally:
            self.stop_feed("shutting down")
            for j in self.tasks.values():
                j.stop()
            if self.streamlit is not None:
                self.streamlit.stop()
            self._note("stopped")
        return 0


LIVE_TAIL_MIN_REFIT = 5   # minutes after the scheduled end before a practice refit starts
OUTLOOK_IDLE_S = 30 * 60  # between sessions: the priors only change when something is scored
OUTLOOK_LIVE_S = 3 * 60   # during practice: fold the live long-run board in as it grows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-app", action="store_true", help="feed and refits only, no dashboard")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--rehearse", default=None, metavar="SESSION_DIR",
                    help="present a recorded session as live (e.g. data/raw/livetiming/2026_hungary_race)")
    ap.add_argument("--speed", type=float, default=20.0, help="rehearsal replay speed")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    st = schedule.status()
    print(f"degless — {schedule.describe(st)}", flush=True)
    return Supervisor(app=not args.no_app, port=args.port, poll_s=args.poll,
                      rehearse=args.rehearse, speed=args.speed).run()


if __name__ == "__main__":
    raise SystemExit(main())
