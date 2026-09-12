"""Where live timing messages come from.

Every source is an iterator of `Message`s, so the state machine and the
strategy engine never know whether they are fed by the live SignalR hub, by a
recording of it, or by the archive files F1 publishes for every session.

* `SignalRSource` — the official feed at `wss://livetiming.formula1.com/signalrcore`.
  Uses the F1TV subscription token FastF1 stores after its browser login when
  one is present (that unlocks `CarData.z` / `Position.z`); connects without
  it otherwise, which still carries every timing topic.  Every message is
  recorded to a FastF1-compatible file so the session can be re-loaded with
  `fastf1.livetiming.data.LiveTimingData` afterwards.
* `RecordedSource` — the per-topic `.jsonStream` archive files (one directory
  per session), merged into one time-ordered stream.  With `speed` set it
  paces playback in real time (or faster); without it, it replays instantly.
  This is the test harness for everything downstream.
* `JsonlSource` — replays our own recordings (`[topic, payload, utc]` lines).
* `StaticPollSource` — polls the same archive files by byte range while a
  session is running.  Unauthenticated and free, but the CDN adds latency and
  F1 could stop updating them mid-session; kept as a last resort.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import requests

from src.live.streams import (
    TOPICS,
    Message,
    decode_payload,
    parse_stream_line,
    parse_utc,
)

log = logging.getLogger("degless.live.sources")

LIVETIMING = "https://livetiming.formula1.com"
STATIC = f"{LIVETIMING}/static"
SIGNALR_WS = "wss://livetiming.formula1.com/signalrcore"
SIGNALR_NEGOTIATE = f"{LIVETIMING}/signalrcore/negotiate"
HEADERS = {"User-Agent": "BestHTTP", "Accept-Encoding": "gzip, identity"}


# --------------------------------------------------------------------------
# Recording (FastF1-compatible)
# --------------------------------------------------------------------------


class Recorder:
    """Append every message as a JSON line `[topic, payload, utc]`.

    FastF1's `LiveTimingData` reads exactly this shape (it `json.loads` each
    line after a quote fix-up), so a recording can be loaded into a FastF1
    session after the fact even if the official archive is slow to appear.
    """

    def __init__(self, path: str | Path, mode: str = "a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, mode, encoding="utf-8")
        self.n = 0

    def write(self, msg: Message) -> None:
        if msg.meta.get("snapshot"):
            return  # a keyframe is not a feed message; FastF1 would misread it
        utc = msg.utc.isoformat().replace("+00:00", "Z") if msg.utc else ""
        payload = msg.raw_payload if hasattr(msg, "raw_payload") else msg.meta.get("wire", msg.payload)
        self._f.write(json.dumps([msg.topic, payload, utc], separators=(",", ":")) + "\n")
        self.n += 1
        if self.n % 50 == 0:
            self._f.flush()

    def close(self) -> None:
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Archive replay
# --------------------------------------------------------------------------


def session_path_from_info(info: dict) -> str:
    return str(info.get("Path", "")).strip("/")


def download_session(path: str, out_dir: str | Path, topics: list | None = None,
                     timeout: int = 90) -> dict:
    """Fetch every topic's `.jsonStream` for a session path into `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    got = {}
    denied = []
    r = requests.get(f"{STATIC}/{path}/SessionInfo.json", headers=HEADERS, timeout=timeout)
    if r.ok:
        (out / "SessionInfo.json").write_bytes(r.content)
        got["SessionInfo"] = len(r.content)
    for t in (topics or TOPICS):
        url = f"{STATIC}/{path}/{t}.jsonStream"
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
        except requests.RequestException as exc:
            log.warning("%s: %s", t, exc)
            continue
        if r.ok and r.content:
            (out / f"{t}.jsonStream").write_bytes(r.content)
            got[t] = len(r.content)
        elif r.status_code == 403:
            denied.append(t)
    if denied and not any(k != "SessionInfo" for k in got):
        # F1 serves the per-topic files only once the session's archive is
        # complete; during and just after a session every one of them is 403.
        log.warning("the archive for %s is not published yet (%d topics denied): "
                    "a late join starts from the live keyframe instead", path, len(denied))
        got["_denied"] = len(denied)
    return got


class RecordedSource:
    """Replay a directory of `.jsonStream` files as one ordered message stream."""

    def __init__(self, session_dir: str | Path, *, speed: float | None = None,
                 topics: list | None = None, start_at_s: float = 0.0,
                 stop_at_s: float | None = None, include_zipped: bool = True):
        self.dir = Path(session_dir)
        self.speed = speed
        self.topics = topics
        self.start_at_s = start_at_s
        self.stop_at_s = stop_at_s
        self.include_zipped = include_zipped
        self.info = {}
        p = self.dir / "SessionInfo.json"
        if p.exists():
            try:
                self.info = json.loads(p.read_text(encoding="utf-8-sig"))
            except Exception:
                self.info = {}

    def _epoch(self) -> datetime | None:
        """Wall clock at session-clock zero, from the first Heartbeat."""
        p = self.dir / "Heartbeat.jsonStream"
        if not p.exists():
            return None
        with open(p, encoding="utf-8-sig") as f:
            for line in f:
                m = parse_stream_line("Heartbeat", line)
                if m and isinstance(m.payload, dict) and m.payload.get("Utc"):
                    u = parse_utc(m.payload["Utc"])
                    if u:
                        return u - timedelta(seconds=m.t_session)
        return None

    def messages(self) -> list:
        msgs = []
        if self.info:
            msgs.append(Message("SessionInfo", self.info, t_session=0.0))
        files = sorted(self.dir.glob("*.jsonStream"))
        for f in files:
            topic = f.name[: -len(".jsonStream")]
            if self.topics and topic not in self.topics:
                continue
            if not self.include_zipped and topic.endswith(".z"):
                continue
            with open(f, encoding="utf-8-sig") as fh:
                for i, line in enumerate(fh):
                    m = parse_stream_line(topic, line)
                    if m is None:
                        continue
                    m.meta["seq"] = i
                    msgs.append(m)
        msgs.sort(key=lambda m: (m.t_session, m.meta.get("seq", 0)))
        epoch = self._epoch()
        if epoch is not None:
            for m in msgs:
                m.utc = epoch + timedelta(seconds=m.t_session)
        return msgs

    def __iter__(self) -> Iterator[Message]:
        msgs = self.messages()
        t0_wall = time.monotonic()
        t0_sess = None
        for m in msgs:
            if m.t_session < self.start_at_s:
                continue
            if self.stop_at_s is not None and m.t_session > self.stop_at_s:
                break
            if self.speed:
                if t0_sess is None:
                    t0_sess = m.t_session
                due = t0_wall + (m.t_session - t0_sess) / self.speed
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, 5.0))
            yield m


class JsonlSource:
    """Replay a `Recorder` file (or FastF1's own live-timing recording)."""

    def __init__(self, path: str | Path, *, speed: float | None = None):
        self.path = Path(path)
        self.speed = speed

    def __iter__(self) -> Iterator[Message]:
        epoch = None
        t0_wall = time.monotonic()
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    topic, payload, utc = json.loads(line)
                except Exception:
                    try:
                        topic, payload, utc = json.loads(
                            line.replace("'", '"').replace("True", "true").replace("False", "false"))
                    except Exception:
                        continue
                u = parse_utc(utc) if utc else None
                if epoch is None and u is not None:
                    epoch = u
                t = (u - epoch).total_seconds() if (u and epoch) else None
                m = Message(topic, decode_payload(topic, payload), utc=u, t_session=t)
                if self.speed and t is not None:
                    due = t0_wall + t / self.speed
                    d = due - time.monotonic()
                    if d > 0:
                        time.sleep(min(d, 5.0))
                yield m


# --------------------------------------------------------------------------
# Live: SignalR Core
# --------------------------------------------------------------------------


def f1tv_token_status() -> tuple[str | None, str, str]:
    """`(token, status, detail)` for the token FastF1's browser login stored.

    `status` is one of `ok`, `expired`, `invalid`, `none`.  The token only
    unlocks the car telemetry topics (`CarData.z` / `Position.z`); every timing
    topic works without it, so an expired token degrades the feed rather than
    breaking it — but it has to be said out loud, because the tokens last about
    four days and the failure is otherwise silent.
    """
    try:
        from fastf1.internals.f1auth import AUTH_DATA_FILE, JWKS_URL, _verify_jwt
    except Exception as exc:
        return None, "none", f"FastF1 auth support unavailable ({exc})"
    try:
        tok = Path(AUTH_DATA_FILE).read_text().strip()
    except FileNotFoundError:
        return None, "none", "no F1TV login stored; run `make login`"
    except Exception as exc:
        return None, "none", f"could not read the stored login ({exc})"
    if not tok:
        return None, "none", "no F1TV login stored; run `make login`"
    try:
        _verify_jwt(tok, JWKS_URL)
    except Exception as exc:
        detail = f"{exc}"
        expiry = _token_expiry(tok)
        if expiry is not None:
            ago = (datetime.now(timezone.utc) - expiry).total_seconds() / 3600.0
            if ago > 0:
                detail = (f"expired {expiry:%Y-%m-%d %H:%M} UTC ({ago:.0f} h ago); "
                          f"run `make login` to sign in again")
                return None, "expired", detail
        return None, "invalid", f"{detail}; run `make login` to sign in again"
    return tok, "ok", "F1TV token valid"


def _token_expiry(tok: str) -> datetime | None:
    """The `exp` claim, without verifying the signature (it may be expired)."""
    try:
        import jwt

        exp = jwt.decode(tok, options={"verify_signature": False}).get("exp")
        return datetime.fromtimestamp(float(exp), timezone.utc) if exp else None
    except Exception:
        return None


def stored_f1tv_token() -> str | None:
    """The F1TV subscription token FastF1's browser login stored, if valid."""
    tok, status, detail = f1tv_token_status()
    if tok is None:
        log.warning("F1TV telemetry not available: %s (timing topics are unaffected)", detail)
    return tok


class SignalRSource:
    """The official live feed.

    Runs the `signalrcore` connection in a background thread and hands messages
    over a queue.  The `Subscribe` completion carries a full keyframe for every
    topic, which is emitted first as snapshot messages (`meta.snapshot=True`),
    so a late join starts from the current state.
    """

    def __init__(self, topics: list | None = None, *, token: str | None = None,
                 use_auth: bool = True, timeout_s: float = 0.0,
                 reconnect: bool = True):
        self.topics = list(topics or TOPICS)
        if token is not None:
            self.token, self.auth_status, self.auth_detail = token, "ok", "token supplied"
        elif use_auth:
            self.token, self.auth_status, self.auth_detail = f1tv_token_status()
            if self.token is None:
                log.warning("F1TV telemetry not available: %s (timing topics are unaffected)",
                            self.auth_detail)
        else:
            self.token, self.auth_status, self.auth_detail = None, "off", "--no-auth"
        self.timeout_s = timeout_s
        self.reconnect = reconnect
        self._q: queue.Queue = queue.Queue()
        self._conn = None
        self._connected = threading.Event()
        self._stop = threading.Event()
        self.last_message_at = None
        self.errors = 0
        self.keyframes = 0        # Subscribe completions: one per (re)connect

    # -- callbacks ----------------------------------------------------------

    def _on_feed(self, args):
        try:
            if isinstance(args, dict):  # completion of Subscribe: keyframe
                now = datetime.now(timezone.utc)   # one instant for the whole keyframe
                for topic, state in args.items():
                    self._q.put(Message(topic, decode_payload(topic, state),
                                        utc=now, meta={"snapshot": True}))
                self.last_message_at = time.time()
                self.keyframes += 1
                return
            topic, payload, utc = args[0], args[1], (args[2] if len(args) > 2 else "")
            u = parse_utc(utc) or datetime.now(timezone.utc)
            m = Message(topic, decode_payload(topic, payload), utc=u,
                        meta={"wire": payload})
            self._q.put(m)
            self.last_message_at = time.time()
        except Exception:
            self.errors += 1
            log.exception("bad feed message")

    def _on_completion(self, msg):
        # signalrcore CompletionMessage for the Subscribe invocation
        try:
            result = getattr(msg, "result", None)
            if isinstance(result, dict):
                self._on_feed(result)
        except Exception:
            log.exception("bad completion")

    def _connect(self):
        from signalrcore.hub_connection_builder import HubConnectionBuilder

        headers = dict(HEADERS)
        try:
            r = requests.options(SIGNALR_NEGOTIATE, headers=headers, timeout=20)
            if "AWSALBCORS" in r.cookies:
                headers["Cookie"] = f"AWSALBCORS={r.cookies['AWSALBCORS']}"
        except requests.RequestException as exc:
            log.warning("pre-negotiate failed: %s", exc)
        options = {"verify_ssl": True, "headers": headers}
        if self.token:
            options["access_token_factory"] = lambda: self.token
        conn = HubConnectionBuilder().with_url(SIGNALR_WS, options=options) \
            .configure_logging(logging.WARNING).build()
        conn.on_open(lambda: (self._connected.set(), log.info("signalr open (%s)",
                                                              "auth" if self.token else "no auth")))
        conn.on_close(lambda: (self._connected.clear(), log.warning("signalr closed")))
        conn.on_error(lambda e: log.warning("signalr error: %s", e))
        conn.on("feed", self._on_feed)
        conn.start()
        for _ in range(300):
            if self._connected.is_set():
                break
            time.sleep(0.1)
        if not self._connected.is_set():
            raise ConnectionError("signalr did not connect")
        conn.send("Subscribe", [self.topics], on_invocation=self._on_completion)
        self._conn = conn

    def _run(self):
        backoff = 2.0
        while not self._stop.is_set():
            try:
                self._connect()
                backoff = 2.0
                while not self._stop.is_set() and self._connected.is_set():
                    time.sleep(0.5)
                    if self.timeout_s and self.last_message_at and \
                            time.time() - self.last_message_at > self.timeout_s:
                        log.warning("no data for %.0fs; reconnecting", self.timeout_s)
                        break
            except Exception as exc:
                self.errors += 1
                log.warning("signalr connection failed: %s", exc)
            finally:
                try:
                    if self._conn is not None:
                        self._conn.stop()
                except Exception:
                    pass
                self._conn = None
                self._connected.clear()
            if not self.reconnect or self._stop.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
        self._q.put(None)

    def start(self) -> "SignalRSource":
        threading.Thread(target=self._run, name="signalr", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    deadline: float | None = None      # wall-clock time after which iteration ends

    def __iter__(self) -> Iterator[Message]:
        if self._conn is None and not self._connected.is_set():
            self.start()
        while True:
            if self.deadline is not None and time.time() > self.deadline:
                self.stop()
                return
            try:
                m = self._q.get(timeout=1.0)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if m is None:
                return
            yield m


# --------------------------------------------------------------------------
# Live fallback: poll the archive files by byte range
# --------------------------------------------------------------------------


def current_session_info(timeout: int = 20) -> dict:
    r = requests.get(f"{STATIC}/SessionInfo.json", headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return json.loads(r.content.decode("utf-8-sig"))


class StaticPollSource:
    """Poll `{STATIC}/{path}/{Topic}.jsonStream` with HTTP Range requests.

    Unverified for latency during a live session; the archive files are
    written by F1 as the session runs, but through a CDN whose caching we do
    not control.  A useful last resort, and the same code that backfills a
    late join to the SignalR feed.
    """

    def __init__(self, path: str | None = None, *, interval_s: float = 5.0,
                 topics: list | None = None, once: bool = False, timeout: int = 30):
        self.path = path
        self.interval_s = interval_s
        self.topics = [t for t in (topics or TOPICS)]
        self.once = once
        self.timeout = timeout
        self.offsets: dict = {}
        self.partial: dict = {}
        self.info: dict = {}
        self.epoch: datetime | None = None

    def _resolve(self):
        if self.path is None:
            self.info = current_session_info(self.timeout)
            self.path = session_path_from_info(self.info)
        else:
            try:
                r = requests.get(f"{STATIC}/{self.path}/SessionInfo.json", headers=HEADERS,
                                 timeout=self.timeout)
                if r.ok:
                    self.info = json.loads(r.content.decode("utf-8-sig"))
            except requests.RequestException:
                pass

    def _fetch_new(self, topic: str) -> list:
        url = f"{STATIC}/{self.path}/{topic}.jsonStream"
        off = self.offsets.get(topic, 0)
        headers = dict(HEADERS)
        headers["Range"] = f"bytes={off}-"
        try:
            r = requests.get(url, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            log.debug("%s: %s", topic, exc)
            return []
        if r.status_code == 416 or r.status_code == 404:
            return []
        if r.status_code == 200 and off > 0:
            body = r.content[off:]     # server ignored the range
        elif r.status_code in (200, 206):
            body = r.content
        else:
            return []
        if not body:
            return []
        self.offsets[topic] = off + len(body)
        text = self.partial.get(topic, "") + body.decode("utf-8-sig", errors="replace")
        lines = text.split("\n")
        self.partial[topic] = lines[-1]
        out = []
        for line in lines[:-1]:
            m = parse_stream_line(topic, line)
            if m is not None:
                out.append(m)
        return out

    def __iter__(self) -> Iterator[Message]:
        self._resolve()
        if self.info:
            yield Message("SessionInfo", self.info, t_session=0.0, utc=datetime.now(timezone.utc))
        while True:
            batch = []
            for t in self.topics:
                batch.extend(self._fetch_new(t))
            batch.sort(key=lambda m: m.t_session)
            for m in batch:
                if m.topic == "Heartbeat" and self.epoch is None and isinstance(m.payload, dict):
                    u = parse_utc(m.payload.get("Utc", ""))
                    if u:
                        self.epoch = u - timedelta(seconds=m.t_session)
                if self.epoch is not None:
                    m.utc = self.epoch + timedelta(seconds=m.t_session)
                yield m
            if self.once:
                return
            time.sleep(self.interval_s)


def make_source(kind: str, **kw):
    kind = kind.lower()
    if kind == "signalr":
        return SignalRSource(**kw)
    if kind == "recorded":
        return RecordedSource(**kw)
    if kind == "jsonl":
        return JsonlSource(**kw)
    if kind == "static":
        return StaticPollSource(**kw)
    raise ValueError(f"unknown source {kind!r}")
