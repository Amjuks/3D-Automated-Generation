import contextlib
import sqlite3
import threading
import time
from pathlib import Path

from .util import canonical


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > 2:
            raise ValueError("database version is newer than this application")
        for migration in sorted((Path(__file__).parent / "migrations").glob("*.sql")):
            if int(migration.name.split("_")[0]) > version:
                self.db.executescript("BEGIN IMMEDIATE;\n" + migration.read_text() + "\nCOMMIT;")
        self.db.commit()

    def execute(self, sql, params=()):
        with self.lock, self.db:
            return self.db.execute(sql, params)

    def rows(self, sql, params=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, params).fetchall()]

    def one(self, sql, params=()):
        rows = self.rows(sql, params)
        return rows[0] if rows else None

    def call(self, **data):
        columns = list(data)
        self.execute(
            f"INSERT INTO llm_calls ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            tuple(data.values()),
        )

    def validation(self, scene_id, report):
        self.execute(
            "INSERT INTO validations(scene_id,created,report) VALUES (?,?,?)",
            (scene_id, time.time(), canonical(report)),
        )

    def close(self):
        self.db.close()


@contextlib.contextmanager
def run_lock(path):
    """Kernel-released advisory lock; interrupted runs never leave a stale lock."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("this run is already being processed") from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
