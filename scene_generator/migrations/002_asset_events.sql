CREATE TABLE IF NOT EXISTS asset_events(
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    scene_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    created REAL NOT NULL,
    reused INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    license TEXT NOT NULL,
    source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS asset_events_run ON asset_events(run_id, scene_id);
PRAGMA user_version=2;
