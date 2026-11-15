"""Output sinks for plate events."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from anpr.stream.events import PlateEvent


class Sink(Protocol):
    """Anything that can receive plate events."""

    def emit(self, event: PlateEvent) -> None:
        """Receive one event."""

    def close(self) -> None:
        """Flush and release resources."""


class JsonlSink:
    """Append events to a JSON-lines file, flushing after every event."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, event: PlateEvent) -> None:
        """Write one event as a line of JSON."""
        self._fh.write(json.dumps(event.to_dict()) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the file."""
        self._fh.close()


class SqliteSink:
    """Store events in SQLite (a local, queryable audit trail)."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS plate_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT, track_id INTEGER, event TEXT, vehicle_class TEXT,
        plate TEXT, confidence REAL, valid INTEGER, n_reads INTEGER,
        first_seen REAL, last_seen REAL, state TEXT, plate_box TEXT
    )"""

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(self._SCHEMA)
        self._db.commit()

    def emit(self, event: PlateEvent) -> None:
        """Insert one event row."""
        d = event.to_dict()
        d["valid"] = int(d["valid"])
        d["plate_box"] = json.dumps(d["plate_box"])
        cols = ",".join(d)
        marks = ",".join("?" * len(d))
        with self._lock:
            self._db.execute(
                f"INSERT INTO plate_events ({cols}) VALUES ({marks})", list(d.values())
            )
            self._db.commit()

    def close(self) -> None:
        """Close the database."""
        self._db.close()


class CallbackSink:
    """Forward events to a function (webhook, MQTT publish, queue, ...)."""

    def __init__(self, fn: Callable[[PlateEvent], None]) -> None:
        self.fn = fn

    def emit(self, event: PlateEvent) -> None:
        """Call the wrapped function."""
        self.fn(event)

    def close(self) -> None:
        """Nothing to release."""


class MultiSink:
    """Fan events out to several sinks; one failing sink never blocks the rest."""

    def __init__(self, sinks: list[Sink]) -> None:
        self.sinks = sinks
        self.errors = 0

    def emit(self, event: PlateEvent) -> None:
        """Send the event to every sink."""
        for s in self.sinks:
            try:
                s.emit(event)
            except Exception:  # noqa: BLE001 - isolate sink failures
                self.errors += 1

    def close(self) -> None:
        """Close every sink."""
        for s in self.sinks:
            s.close()
