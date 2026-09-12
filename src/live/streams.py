"""Stream names, message envelope and decoders for the F1 live timing feed.

The feed is a set of named topics.  Over SignalR each message is
`[topic, payload, utc]`; in the archived `.jsonStream` files the same payloads
are prefixed with a session-relative `HH:MM:SS.mmm` clock.  Both are normalised
here into a `Message` carrying the topic, the decoded payload, the wall-clock
time if known and a session-relative time in seconds.

Compressed topics (`CarData.z`, `Position.z`) are base64 + raw deflate; the
decoder is the one FastF1 uses.
"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

# Topics we subscribe to.  The order is cosmetic.
TOPICS = [
    "Heartbeat", "SessionInfo", "SessionStatus", "SessionData",
    "ExtrapolatedClock", "LapCount", "TrackStatus", "DriverList",
    "TimingData", "TimingAppData", "TyreStintSeries", "TimingStats",
    "WeatherData", "RaceControlMessages", "CarData.z", "Position.z",
    "TopThree", "RcmSeries", "TeamRadio", "ChampionshipPrediction",
    "PitLaneTimeCollection", "PitStopSeries",
]

# Topics whose payload arrives compressed.
ZIPPED = {"CarData.z", "Position.z"}

# Track status codes as FastF1 reports them.
TRACK_STATUS = {
    "1": "AllClear", "2": "Yellow", "4": "SCDeployed", "5": "Red",
    "6": "VSCDeployed", "7": "VSCEnding",
}
TRACK_STATUS_FROM_MESSAGE = {v: k for k, v in TRACK_STATUS.items()}


@dataclass
class Message:
    topic: str
    payload: Any
    utc: datetime | None = None          # wall clock, when known
    t_session: float | None = None       # seconds since the stream's epoch
    raw: str = ""                        # original text, for recording
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def inflate(text: str) -> Any:
    """Decode a `.z` payload: base64 → raw deflate → JSON."""
    if isinstance(text, (dict, list)):
        return text
    text = text.strip().strip('"')
    raw = zlib.decompress(base64.b64decode(text), -zlib.MAX_WBITS)
    return json.loads(raw.decode("utf-8-sig"))


def decode_payload(topic: str, payload: Any) -> Any:
    if topic in ZIPPED and isinstance(payload, str):
        try:
            return inflate(payload)
        except Exception:
            return None
    if isinstance(payload, str):
        s = payload.strip()
        if s[:1] in "{[":
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return payload
    return payload


def parse_clock(text: str) -> float | None:
    """`HH:MM:SS.mmm` (session-relative) → seconds."""
    try:
        h, m, s = text.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return None


def parse_utc(text: str) -> datetime | None:
    """ISO-8601 with up to 7 fractional digits and a trailing Z."""
    if not text:
        return None
    t = text.strip().rstrip("Z")
    if "." in t:
        head, frac = t.split(".", 1)
        frac = (frac + "000000")[:6]
        t = f"{head}.{frac}"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def parse_laptime(text: Any) -> float | None:
    """`1:23.456` / `83.456` / `16:39.662` → seconds; '' → None."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip()
    if not s:
        return None
    try:
        parts = s.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return None


def parse_gap(text: Any) -> float | None:
    """`+1.234` / `1.234` / `+1 LAP` / `LAP 12` → seconds (None if not a time)."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip().replace("+", "")
    if not s or "LAP" in s.upper():
        return None
    try:
        return float(s)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# jsonStream lines
# --------------------------------------------------------------------------


def parse_stream_line(topic: str, line: str) -> Message | None:
    """One line of an archived `.jsonStream` file → `Message` (session clock only)."""
    line = line.lstrip("﻿").rstrip("\r\n")
    if len(line) < 13:
        return None
    t = parse_clock(line[:12])
    if t is None:
        return None
    payload = decode_payload(topic, line[12:])
    if payload is None:
        return None
    return Message(topic=topic, payload=payload, t_session=t, raw=line)


def session_epoch(t_session: float, anchor_utc: datetime, anchor_t: float) -> datetime:
    return anchor_utc + timedelta(seconds=t_session - anchor_t)
