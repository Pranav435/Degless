"""The session, reconstructed lap by lap from the live timing patches.

`LiveState.apply(message)` merges every patch into a per-topic snapshot and,
for the topics that matter to strategy, maintains tidy derived structures:

* **laps** — one record per driver per lap, in the canonical schema the
  offline pipeline uses (`src.ingest.LAP_SCHEMA`) plus sectors, speed traps,
  position and the gap to the car ahead at the line.  Lap boundaries follow the
  same rules FastF1's archive parser uses (`fastf1._api._laps_data_driver`), so
  a session replayed through here reproduces FastF1's lap table — that is the
  test `tests/test_live_state.py` runs.
* **stints** — from `TimingAppData`: compound, whether the set was new, its
  age when fitted, and the lap it started on.
* **track status** timeline, session status, lap count, weather, race control
  messages, the driver list, and the latest position/gap/interval per driver.
* **car telemetry** — the latest `CarData.z` sample per driver and a short
  ring buffer, enough for a live minimum-corner-speed channel.

The state is deliberately plain Python (dicts, lists, floats): it is updated
tens of times a second and must never block on pandas.  DataFrame views are
built on demand and cached until the next relevant patch.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.live.merge import deep_copy, indexed, merge
from src.live.streams import (
    TRACK_STATUS_FROM_MESSAGE,
    Message,
    parse_gap,
    parse_laptime,
    parse_utc,
)

log = logging.getLogger("degless.live.state")

LATE_VALUE_WINDOW_S = 5.0     # values this soon after a boundary belong to the previous lap
MAX_LAP_TIME_S = 150.0        # longer "laps" are session breaks, not laps (FastF1 rule)
SECTOR_SUM_TOL_S = 0.0035     # |s1+s2+s3 - lap| above this marks the lap inaccurate (FastF1: 3 ms)
CAR_BUFFER = 600              # ~2.5 minutes of 3.7 Hz car telemetry per driver

LAP_COLUMNS = [
    "driver_number", "driver", "team", "lap_number", "stint", "compound",
    "tyre_life", "tyre_new", "lap_time_s", "lap_start_s", "lap_end_s",
    "s1_s", "s2_s", "s3_s", "speed_i1", "speed_i2", "speed_fl", "speed_st",
    "is_accurate", "pit_in", "pit_out", "track_status", "position",
    "gap_leader_s", "interval_s", "n_pit_stops", "is_complete", "utc_end",
]


def _new_lap(number: int, start_s: float | None, status: str) -> dict:
    return {
        "lap_number": number, "lap_time_s": None, "lap_time_explicit_none": False,
        "lap_start_s": start_s, "lap_end_s": None, "utc_end": None,
        "s1_s": None, "s2_s": None, "s3_s": None,
        "speed_i1": None, "speed_i2": None, "speed_fl": None, "speed_st": None,
        "pit_in": False, "pit_out": False, "pit_in_s": None, "pit_out_s": None,
        "statuses": [status] if status else [], "position": None,
        "gap_leader_s": None, "interval_s": None, "n_pit_stops": 0,
        "stint": None, "is_complete": False,
    }


@dataclass
class DriverTrack:
    number: str
    laps: list = field(default_factory=list)      # finalised + the current lap (last)
    api_laps: int = 0                             # feed's NumberOfLaps, monotone
    out_of_pit: bool = False
    in_pit: bool = True
    pit_stops: int = -1                           # FastF1 convention: first exit -> 0
    last_boundary_s: float | None = None
    stints: list = field(default_factory=list)    # [{index, compound, new, start_laps, total_laps, started_s, first_lap}]
    # latest streamed values
    position: int | None = None
    gap_leader_s: float | None = None
    gap_leader_txt: str = ""
    interval_s: float | None = None
    interval_txt: str = ""
    last_lap_s: float | None = None
    best_lap_s: float | None = None
    retired: bool = False
    stopped: bool = False
    status_bits: int = 0
    line: int | None = None

    @property
    def current(self) -> dict | None:
        return self.laps[-1] if self.laps else None

    @property
    def n_complete(self) -> int:
        return sum(1 for l in self.laps if l["is_complete"])

    @property
    def current_stint(self) -> dict | None:
        return self.stints[-1] if self.stints else None


class LiveState:
    """Everything known about the session so far."""

    def __init__(self, session_type: str | None = None):
        self.snap: dict = {}                       # topic -> merged raw state
        self.drivers: dict[str, dict] = {}         # number -> {tla, name, team, colour}
        self.tracks: dict[str, DriverTrack] = {}
        self.session_type = session_type           # "Race" | "Practice" | "Qualifying" | None
        self.session_info: dict = {}
        self.session_status: str = ""
        self.session_started_s: float | None = None
        self.session_finished_s: float | None = None
        self.lap_count: dict = {"current": None, "total": None}
        self.track_status: str = "1"
        self.track_status_log: list = []           # [(t_session, status, message)]
        self.weather: dict = {}
        self.weather_log: list = []
        self.rcm: list = []                        # race control messages, in order
        self.clock: dict = {}                      # ExtrapolatedClock
        self.t_now: float = 0.0                    # latest session time seen
        self.utc_now: datetime | None = None
        self.epoch_utc: datetime | None = None     # utc at t_session = 0 (when known)
        self.car: dict[str, dict] = {}             # latest telemetry per driver
        self.car_buf: dict[str, deque] = {}
        self.pos: dict[str, dict] = {}
        self.pit_lane: dict = {}                   # PitLaneTimeCollection / PitStopSeries
        self.n_messages = 0
        self._version = 0
        self._laps_cache: tuple = (-1, None)
        self.events: list = []                     # (t, kind, payload) since last drain

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def apply(self, msg: Message) -> None:
        self.n_messages += 1
        if msg.t_session is not None:
            self.t_now = max(self.t_now, float(msg.t_session))
        if msg.utc is not None:
            self.utc_now = msg.utc
            if self.epoch_utc is None and msg.t_session is not None:
                from datetime import timedelta
                self.epoch_utc = msg.utc - timedelta(seconds=float(msg.t_session))
        topic, p = msg.topic, msg.payload
        t = float(msg.t_session) if msg.t_session is not None else self.t_now

        handler = getattr(self, f"_on_{topic.replace('.', '_')}", None)
        if topic not in ("CarData.z", "Position.z"):
            self.snap[topic] = merge(self.snap.get(topic), deep_copy(p) if isinstance(p, (dict, list)) else p)
        if handler is not None:
            try:
                handler(p, t, msg)
            except Exception:  # never let one bad patch kill the feed
                log.exception("handler for %s failed", topic)

    def drain_events(self) -> list:
        ev, self.events = self.events, []
        return ev

    def _emit(self, t: float, kind: str, **payload) -> None:
        self.events.append((t, kind, payload))
        self._version += 1

    # ------------------------------------------------------------------
    # Session-level topics
    # ------------------------------------------------------------------

    def _on_SessionInfo(self, p, t, msg):
        if isinstance(p, dict):
            self.session_info = self.snap.get("SessionInfo", p)
            typ = self.session_info.get("Type") or self.session_info.get("Name", "")
            if typ and self.session_type is None:
                self.session_type = str(typ)

    def _on_SessionStatus(self, p, t, msg):
        if not isinstance(p, dict) or "Status" not in p:
            return
        self.session_status = str(p["Status"])
        if self.session_status == "Started":
            if self.session_started_s is None:
                self.session_started_s = t
            if self.is_race:
                for tr in self.tracks.values():
                    self._race_start(tr, t)
        elif self.session_status in ("Finished", "Finalised", "Ends"):
            if self.session_finished_s is None:
                self.session_finished_s = t
        self._emit(t, "session_status", status=self.session_status)

    def _on_LapCount(self, p, t, msg):
        if isinstance(p, dict):
            if "CurrentLap" in p:
                self.lap_count["current"] = int(p["CurrentLap"])
            if "TotalLaps" in p:
                self.lap_count["total"] = int(p["TotalLaps"])
            self._emit(t, "lap_count", **self.lap_count)

    def _on_TrackStatus(self, p, t, msg):
        if not isinstance(p, dict):
            return
        status = str(p.get("Status") or TRACK_STATUS_FROM_MESSAGE.get(p.get("Message", ""), "1"))
        self.track_status = status
        self.track_status_log.append((t, status, p.get("Message", "")))
        for tr in self.tracks.values():
            if tr.current is not None and (not tr.current["statuses"] or tr.current["statuses"][-1] != status):
                tr.current["statuses"].append(status)
        self._emit(t, "track_status", status=status, message=p.get("Message", ""))

    def _on_WeatherData(self, p, t, msg):
        if isinstance(p, dict):
            w = {}
            for k, v in p.items():
                try:
                    w[k] = float(v) if k != "Rainfall" else (str(v) in ("1", "true", "True"))
                except (TypeError, ValueError):
                    w[k] = v
            self.weather.update(w)
            self.weather_log.append((t, dict(self.weather)))

    def _on_RaceControlMessages(self, p, t, msg):
        if not isinstance(p, dict):
            return
        msgs = p.get("Messages", p)
        for _, m in indexed(msgs):
            if isinstance(m, dict) and "Message" in m:
                m = dict(m)
                m["t_session"] = t
                self.rcm.append(m)
                self._emit(t, "rcm", **m)

    def _on_ExtrapolatedClock(self, p, t, msg):
        if isinstance(p, dict):
            self.clock.update(p)

    def _on_Heartbeat(self, p, t, msg):
        if isinstance(p, dict) and p.get("Utc") and msg.utc is None:
            u = parse_utc(p["Utc"])
            if u is not None:
                self.utc_now = u
                if self.epoch_utc is None:
                    from datetime import timedelta
                    self.epoch_utc = u - timedelta(seconds=t)

    def _on_DriverList(self, p, t, msg):
        if not isinstance(p, dict):
            return
        for num, d in p.items():
            if not isinstance(d, dict) or not str(num).isdigit():
                continue
            cur = self.drivers.setdefault(str(num), {"number": str(num)})
            if "Tla" in d:
                cur["tla"] = d["Tla"]
            if "TeamName" in d:
                cur["team"] = d["TeamName"]
            if "TeamColour" in d:
                cur["colour"] = d["TeamColour"]
            if "FullName" in d:
                cur["name"] = d["FullName"]
            if "Line" in d:
                cur["line"] = d["Line"]
            self._track(str(num))

    def _on_PitLaneTimeCollection(self, p, t, msg):
        if isinstance(p, dict):
            self.pit_lane = merge(self.pit_lane, deep_copy(p))

    def _on_PitStopSeries(self, p, t, msg):
        if isinstance(p, dict):
            self.pit_lane.setdefault("PitStopSeries", {})
            self.pit_lane["PitStopSeries"] = merge(self.pit_lane["PitStopSeries"], deep_copy(p))

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    @property
    def is_race(self) -> bool:
        st = (self.session_type or "").lower()
        return st.startswith("race") or st == "sprint" or "sprint" == st

    def _track(self, num: str) -> DriverTrack:
        tr = self.tracks.get(num)
        if tr is None:
            tr = DriverTrack(number=num)
            self.tracks[num] = tr
        return tr

    def _race_start(self, tr: DriverTrack, t: float) -> None:
        """At the lights: everyone is on lap 1 from the grid, pit counter at zero."""
        tr.out_of_pit = True
        tr.in_pit = False
        if tr.pit_stops < 0:
            tr.pit_stops = 0
        if not tr.laps:
            tr.laps.append(_new_lap(1, t, self.track_status))
        else:
            cur = tr.current
            cur["lap_start_s"] = t
            cur["pit_in"] = False
            cur["pit_in_s"] = None
            cur["statuses"] = [self.track_status]
            cur["pit_out"], cur["pit_out_s"] = False, None   # the lap to the grid is not an out-lap
        tr.current["n_pit_stops"] = 0
        tr.current["stint"] = tr.current_stint["index"] if tr.current_stint else None
        tr.last_boundary_s = t

    def _on_TimingData(self, p, t, msg):
        if not isinstance(p, dict):
            return
        lines = p.get("Lines")
        if not isinstance(lines, dict):
            return
        for num, resp in lines.items():
            if not isinstance(resp, dict):
                continue
            self._timing_line(self._track(str(num)), resp, t, msg)

    def _timing_line(self, tr: DriverTrack, resp: dict, t: float, msg: Message) -> None:
        # --- streamed values (position, gaps) ------------------------------
        if "Position" in resp:
            try:
                tr.position = int(resp["Position"])
            except (TypeError, ValueError):
                pass
        if "Line" in resp:
            tr.line = resp["Line"]
        if "GapToLeader" in resp:
            tr.gap_leader_txt = str(resp["GapToLeader"])
            tr.gap_leader_s = parse_gap(resp["GapToLeader"])
        if isinstance(resp.get("IntervalToPositionAhead"), dict) and "Value" in resp["IntervalToPositionAhead"]:
            tr.interval_txt = str(resp["IntervalToPositionAhead"]["Value"])
            tr.interval_s = parse_gap(resp["IntervalToPositionAhead"]["Value"])
        if "Retired" in resp:
            tr.retired = bool(resp["Retired"])
        if "Stopped" in resp:
            tr.stopped = bool(resp["Stopped"])
        if "Status" in resp and isinstance(resp["Status"], int):
            tr.status_bits = resp["Status"]
        if isinstance(resp.get("BestLapTime"), dict) and resp["BestLapTime"].get("Value"):
            v = parse_laptime(resp["BestLapTime"]["Value"])
            if v is not None:
                tr.best_lap_s = v

        # --- lap counting, FastF1 semantics ---------------------------------
        n_api = resp.get("NumberOfLaps")
        if isinstance(n_api, int) and n_api < tr.api_laps:
            return  # late data for a lap already processed: ignore the message

        if "InPit" in resp and resp["InPit"] is False:
            tr.out_of_pit = True

        # Values arriving within a few seconds of a boundary belong to the lap
        # that just ended, except the speed trap on the main straight.
        late = (tr.last_boundary_s is not None and len(tr.laps) >= 2
                and (t - tr.last_boundary_s) < LATE_VALUE_WINDOW_S)
        cur = tr.current
        if cur is None:
            cur = _new_lap(1, t if self.is_race else None, self.track_status)
            tr.laps.append(cur)
        target = tr.laps[-2] if late else cur

        sectors = resp.get("Sectors")
        if isinstance(sectors, dict):
            for key, col in (("0", "s1_s"), ("1", "s2_s"), ("2", "s3_s")):
                s = sectors.get(key)
                if isinstance(s, dict) and s.get("Value"):
                    v = parse_laptime(s["Value"])
                    if v is not None:
                        target[col] = v
        llt = resp.get("LastLapTime")
        if isinstance(llt, dict) and "Value" in llt:
            v = parse_laptime(llt["Value"])
            if v is None:
                target["lap_time_explicit_none"] = True
            elif v < MAX_LAP_TIME_S:
                target["lap_time_s"] = v
                tr.last_lap_s = v
        speeds = resp.get("Speeds")
        if isinstance(speeds, dict):
            for key, col in (("I1", "speed_i1"), ("I2", "speed_i2"), ("FL", "speed_fl"), ("ST", "speed_st")):
                s = speeds.get(key)
                if isinstance(s, dict) and s.get("Value"):
                    try:
                        (cur if key == "ST" else target)[col] = float(s["Value"])
                    except ValueError:
                        pass

        boundary = isinstance(n_api, int) and n_api > tr.api_laps

        if "InPit" in resp:
            if resp["InPit"] is True:
                tr.in_pit = True
                if tr.pit_stops >= 0:
                    cur["pit_in"] = True
                    cur["pit_in_s"] = t
                self._emit(t, "pit_in", driver=tr.number, lap=cur["lap_number"])
            elif resp["InPit"] is False:
                tr.in_pit = False
                tr.pit_stops += 1
                if boundary:
                    cur["_pit_out_next"] = t   # belongs to the lap about to start
                else:
                    cur["pit_out"] = True
                    cur["pit_out_s"] = t
                self._emit(t, "pit_out", driver=tr.number, lap=cur["lap_number"])
        if "PitOut" in resp and resp["PitOut"] is True and not cur["pit_out"] and not boundary:
            # Some sessions only flag PitOut; treat like InPit False if we
            # never saw the explicit transition.
            if not tr.out_of_pit:
                tr.out_of_pit = True
                tr.in_pit = False
                tr.pit_stops = max(tr.pit_stops, 0)
            cur["pit_out"] = True
            cur["pit_out_s"] = t

        if boundary:
            tr.api_laps = n_api
            if tr.out_of_pit and tr.pit_stops >= 0:
                self._close_lap(tr, cur, t, msg)

    @staticmethod
    def _has_data(lap: dict) -> bool:
        return any(lap[k] is not None for k in ("lap_time_s", "s1_s", "s2_s", "s3_s",
                                                  "speed_i1", "speed_i2", "speed_fl"))

    def _close_lap(self, tr: DriverTrack, cur: dict, t: float, msg: Message) -> None:
        if cur["lap_number"] == 1 and not self._has_data(cur) and not self.is_race:
            # The feed's first lap count arrives as the car leaves the garage:
            # a pseudo out-lap with nothing in it.  FastF1 drops it; so do we,
            # by restarting lap 1 from here and carrying the pit exit over.
            pit_out_s = cur.pop("_pit_out_next", None) or (cur["pit_out_s"] if cur["pit_out"] else None)
            fresh = _new_lap(1, t, self.track_status)
            if pit_out_s is not None:
                fresh["pit_out"], fresh["pit_out_s"] = True, pit_out_s
            fresh["n_pit_stops"] = tr.pit_stops
            fresh["stint"] = tr.current_stint["index"] if tr.current_stint else cur["stint"]
            tr.laps[-1] = fresh
            tr.last_boundary_s = t
            return
        cur["lap_end_s"] = t
        cur["utc_end"] = msg.utc
        cur["n_pit_stops"] = tr.pit_stops
        cur["position"] = tr.position
        cur["gap_leader_s"] = tr.gap_leader_s
        cur["interval_s"] = tr.interval_s
        cur["is_complete"] = True
        st = tr.current_stint
        cur["stint"] = st["index"] if st else None
        # A pit exit flagged in the same message as the boundary belongs to
        # the new lap; one flagged in the last seconds of the old lap too.
        move_pit_out = cur.pop("_pit_out_next", None)
        if move_pit_out is None and cur["pit_out"] and cur["pit_out_s"] is not None \
                and (t - cur["pit_out_s"]) < LATE_VALUE_WINDOW_S and cur["lap_number"] > 1:
            move_pit_out = cur["pit_out_s"]
            cur["pit_out"], cur["pit_out_s"] = False, None
        nxt = _new_lap(cur["lap_number"] + 1, t, self.track_status)
        if move_pit_out is not None:
            nxt["pit_out"], nxt["pit_out_s"] = True, move_pit_out
        nxt["n_pit_stops"] = tr.pit_stops
        nxt["stint"] = cur["stint"]
        tr.laps.append(nxt)
        tr.last_boundary_s = t
        self._emit(t, "lap", driver=tr.number, lap=cur["lap_number"])

    # ------------------------------------------------------------------
    # Stints
    # ------------------------------------------------------------------

    def _on_TimingAppData(self, p, t, msg):
        if not isinstance(p, dict):
            return
        lines = p.get("Lines")
        if not isinstance(lines, dict):
            return
        for num, resp in lines.items():
            if not isinstance(resp, dict):
                continue
            tr = self._track(str(num))
            stints = resp.get("Stints")
            if stints is None:
                continue
            for idx, s in indexed(stints):
                if not isinstance(s, dict):
                    continue
                self._stint_patch(tr, idx, s, t)

    def _stint_patch(self, tr: DriverTrack, idx: int, s: dict, t: float) -> None:
        rec = next((x for x in tr.stints if x["index"] == idx), None)
        if rec is None:
            if "Compound" not in s and not tr.stints:
                return  # a TotalLaps tick before we know the compound: wait
            rec = {"index": idx, "compound": None, "new": None, "start_laps": 0,
                   "total_laps": 0, "started_s": t, "first_lap": None,
                   "tyres_not_changed": None}
            tr.stints.append(rec)
            tr.stints.sort(key=lambda r: r["index"])
            cur = tr.current
            if cur is not None:
                # The stint appears during the out-lap (or, for a race, before
                # the start); that lap and all later ones belong to it.
                cur["stint"] = idx
                rec["first_lap"] = cur["lap_number"]
            changed = True
        else:
            changed = False
        if "Compound" in s:
            c = str(s["Compound"]).upper()
            if c != rec["compound"]:
                rec["compound"] = c
                changed = True
        if "New" in s:
            rec["new"] = str(s["New"]).lower() == "true"
        if "StartLaps" in s:
            try:
                rec["start_laps"] = int(s["StartLaps"])
            except (TypeError, ValueError):
                pass
        if "TotalLaps" in s:
            try:
                rec["total_laps"] = int(s["TotalLaps"])
            except (TypeError, ValueError):
                pass
        if "TyresNotChanged" in s:
            rec["tyres_not_changed"] = str(s["TyresNotChanged"]) == "1"
        if changed:
            self._emit(t, "stint", driver=tr.number, index=idx, compound=rec["compound"])

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _on_CarData_z(self, p, t, msg):
        if not isinstance(p, dict):
            return
        for e in p.get("Entries", []):
            utc = parse_utc(e.get("Utc", ""))
            for num, car in (e.get("Cars") or {}).items():
                ch = (car or {}).get("Channels") or {}
                try:
                    sample = {"utc": utc, "t": t, "rpm": ch.get("0"), "speed": ch.get("2"),
                              "gear": ch.get("3"), "throttle": ch.get("4"),
                              "brake": ch.get("5"), "drs": ch.get("45")}
                except AttributeError:
                    continue
                self.car[str(num)] = sample
                self.car_buf.setdefault(str(num), deque(maxlen=CAR_BUFFER)).append(sample)

    def _on_Position_z(self, p, t, msg):
        if not isinstance(p, dict):
            return
        for e in p.get("Position", []):
            utc = parse_utc(e.get("Timestamp", ""))
            for num, ent in (e.get("Entries") or {}).items():
                if isinstance(ent, dict):
                    self.pos[str(num)] = {"utc": utc, "t": t, "status": ent.get("Status"),
                                          "x": ent.get("X"), "y": ent.get("Y"), "z": ent.get("Z")}

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def driver_label(self, num: str) -> str:
        d = self.drivers.get(str(num), {})
        return d.get("tla") or str(num)

    def tyre_age(self, tr: DriverTrack, lap: dict) -> float | None:
        """Age of the tyre on this lap: laps run on the set at the end of it."""
        if lap.get("stint") is None:
            return None
        st = next((x for x in tr.stints if x["index"] == lap["stint"]), None)
        if st is None or st.get("first_lap") is None:
            return None
        return float(st["start_laps"] + (lap["lap_number"] - st["first_lap"] + 1))

    def lap_records(self, include_current: bool = True) -> list:
        rows = []
        for num, tr in self.tracks.items():
            d = self.drivers.get(num, {})
            stint_by_idx = {s["index"]: s for s in tr.stints}
            # Stints are numbered by order of use, like FastF1, so a set that
            # was declared and replaced before it turned a lap does not count.
            used = sorted({l["stint"] for l in tr.laps if l["stint"] is not None
                           and (l["is_complete"] or self._has_data(l))})
            rank = {idx: i + 1 for i, idx in enumerate(used)}
            prev_statuses = None
            for lap in tr.laps:
                if not lap["is_complete"]:
                    if not include_current and not (self._has_data(lap) or lap["pit_in"]):
                        continue
                st = stint_by_idx.get(lap["stint"]) if lap["stint"] is not None else None
                lt = lap["lap_time_s"]
                s1, s2, s3 = lap["s1_s"], lap["s2_s"], lap["s3_s"]
                if lt is None and lap["lap_time_explicit_none"] and None not in (s1, s2, s3):
                    lt = s1 + s2 + s3
                statuses = lap["statuses"] or ["1"]
                ts = "".join(statuses) if statuses != ["1"] else "1"
                accurate = (lap["is_complete"] and lt is not None and None not in (s1, s2, s3)
                            and abs((s1 + s2 + s3) - lt) < SECTOR_SUM_TOL_S
                            and not lap["pit_in"] and not lap["pit_out"]
                            and set(statuses) <= {"1", "2"}
                            and (prev_statuses is None or "4" not in prev_statuses))
                prev_statuses = statuses
                rows.append({
                    "driver_number": num, "driver": d.get("tla", num), "team": d.get("team"),
                    "lap_number": float(lap["lap_number"]),
                    "stint": float(rank.get(lap["stint"], len(used) + 1)) if lap["stint"] is not None else float("nan"),
                    "compound": (st or {}).get("compound"),
                    "tyre_life": self.tyre_age(tr, lap),
                    "tyre_new": (st or {}).get("new"),
                    "lap_time_s": lt, "lap_start_s": lap["lap_start_s"], "lap_end_s": lap["lap_end_s"],
                    "s1_s": s1, "s2_s": s2, "s3_s": s3,
                    "speed_i1": lap["speed_i1"], "speed_i2": lap["speed_i2"],
                    "speed_fl": lap["speed_fl"], "speed_st": lap["speed_st"],
                    "is_accurate": bool(accurate), "pit_in": bool(lap["pit_in"]),
                    "pit_out": bool(lap["pit_out"]), "track_status": ts,
                    "position": lap["position"], "gap_leader_s": lap["gap_leader_s"],
                    "interval_s": lap["interval_s"], "n_pit_stops": lap["n_pit_stops"],
                    "is_complete": bool(lap["is_complete"]), "utc_end": lap["utc_end"],
                })
        return rows

    def laps_df(self, include_current: bool = False) -> pd.DataFrame:
        key = (self._version, include_current)
        if self._laps_cache[0] == key and self._laps_cache[1] is not None:
            return self._laps_cache[1]
        rows = self.lap_records(include_current=include_current)
        df = pd.DataFrame(rows, columns=LAP_COLUMNS)
        if not df.empty:
            df["lap_time_s"] = pd.to_numeric(df["lap_time_s"], errors="coerce")
            df["tyre_life"] = pd.to_numeric(df["tyre_life"], errors="coerce")
            df = df.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
        self._laps_cache = (key, df)
        return df

    def stints_df(self) -> pd.DataFrame:
        rows = []
        for num, tr in self.tracks.items():
            for s in tr.stints:
                n_laps = sum(1 for l in tr.laps if l["stint"] == s["index"] and l["is_complete"])
                rows.append({"driver_number": num, "driver": self.driver_label(num),
                             "stint": s["index"] + 1, "compound": s["compound"],
                             "new": s["new"], "start_laps": s["start_laps"],
                             "total_laps": s["total_laps"], "first_lap": s["first_lap"],
                             "laps_complete": n_laps, "started_s": s["started_s"]})
        return pd.DataFrame(rows)

    def field_snapshot(self) -> list:
        """Latest per-driver state, ordered by position."""
        out = []
        for num, tr in self.tracks.items():
            d = self.drivers.get(num, {})
            st = tr.current_stint
            cur = tr.current
            out.append({
                "driver_number": num, "driver": d.get("tla", num), "team": d.get("team"),
                "colour": d.get("colour"), "position": tr.position, "line": tr.line,
                "gap_leader_s": tr.gap_leader_s, "gap_leader": tr.gap_leader_txt,
                "interval_s": tr.interval_s, "interval": tr.interval_txt,
                "in_pit": tr.in_pit, "retired": tr.retired, "stopped": tr.stopped,
                "laps_complete": tr.n_complete, "current_lap": cur["lap_number"] if cur else None,
                "compound": (st or {}).get("compound"),
                "tyre_age": (self.tyre_age(tr, cur) if cur else None),
                "stint": (st["index"] + 1) if st else None,
                "n_pit_stops": max(tr.pit_stops, 0),
                "last_lap_s": tr.last_lap_s, "best_lap_s": tr.best_lap_s,
                "speed": (self.car.get(num) or {}).get("speed"),
            })
        out.sort(key=lambda r: (r["position"] is None, r["position"] or 99))
        return out

    def summary(self) -> dict:
        return {
            "session": {k: self.session_info.get(k) for k in ("Name", "Type", "Path", "Key", "StartDate")}
            | {"meeting": (self.session_info.get("Meeting") or {}).get("Name")},
            "status": self.session_status, "track_status": self.track_status,
            "lap_count": dict(self.lap_count), "t_session": self.t_now,
            "utc": self.utc_now.isoformat() if self.utc_now else None,
            "weather": dict(self.weather), "n_drivers": len(self.tracks),
            "n_messages": self.n_messages, "clock": dict(self.clock),
        }
