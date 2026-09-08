import sqlite3
from pathlib import Path

import pytest

from scene_generator.checkpoint import State, run_lock


def test_migration_preserves_existing_run(tmp_path):
    path = tmp_path / "state.sqlite"
    db = sqlite3.connect(path)
    migrations = Path(__file__).parents[1] / "scene_generator/migrations"
    db.executescript((migrations / "001_initial.sql").read_text())
    db.execute("INSERT INTO runs(id,config,status,stage,started) VALUES ('old','{}','pending','plan',1)")
    db.commit()
    db.close()
    state = State(path)
    try:
        assert state.one("SELECT id FROM runs")["id"] == "old"
        assert state.db.execute("PRAGMA user_version").fetchone()[0] == 2
        assert state.rows("SELECT * FROM asset_events") == []
    finally:
        state.close()


def test_run_lock_released_after_failure(tmp_path):
    path = tmp_path / ".lock"
    with run_lock(path):
        with pytest.raises(RuntimeError, match="already"):
            with run_lock(path):
                pass
    with run_lock(path):
        pass
