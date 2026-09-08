import time


def usage(state, run_id, scene_id=None):
    where, args = "run_id=?", [run_id]
    if scene_id:
        where += " AND scene_id=?"
        args.append(scene_id)
    rows = state.rows("SELECT * FROM llm_calls WHERE " + where, args)

    def aggregate(calls):
        result = {
            key: sum(r[key] for r in calls)
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "total_tokens", "cache_hit")
        }
        result.update(
            calls=len(calls),
            retries=sum(r["attempt"] > 0 for r in calls),
            errors=sum(r["error"] is not None for r in calls),
            duration_seconds=sum(r["duration"] for r in calls),
            estimated_cost=sum(r["estimated_cost"] for r in calls)
            if all(r["estimated_cost"] is not None for r in calls)
            else None,
        )
        return result

    result = aggregate(rows)
    result["by_component"] = {
        key or "planning": aggregate([r for r in rows if r["component_id"] == key])
        for key in sorted({r["component_id"] for r in rows}, key=lambda x: x or "")
    }
    return result


def run_summary(state, run_id):
    run = state.one("SELECT id,status,stage,started,ended,error FROM runs WHERE id=?", (run_id,))
    if not run:
        raise ValueError("run not found")
    run["duration_seconds"] = (run["ended"] or time.time()) - run["started"]
    run["scenes"] = state.rows(
        "SELECT id,category,status,stage,started,ended,error FROM scenes WHERE run_id=? ORDER BY ordinal,id", (run_id,)
    )
    run["components"] = state.rows(
        "SELECT c.status,COUNT(*) AS count FROM components c JOIN scenes s ON c.scene_id=s.id WHERE s.run_id=? GROUP BY c.status",
        (run_id,),
    )
    run["assets"] = state.one(
        "SELECT COUNT(*) AS uses,COALESCE(SUM(reused),0) AS reused,COALESCE(SUM(1-reused),0) AS downloaded,"
        "COALESCE(SUM(CASE WHEN reused=0 THEN bytes ELSE 0 END),0) AS downloaded_bytes FROM asset_events WHERE run_id=?",
        (run_id,),
    )
    run["files"] = state.one(
        "SELECT COUNT(*) AS count,COALESCE(SUM(f.bytes),0) AS bytes FROM files f JOIN scenes s ON f.scene_id=s.id WHERE s.run_id=?",
        (run_id,),
    )
    run["geometry"] = state.one(
        "SELECT COALESCE(SUM(json_extract(c.stats,'$.triangles')),0) AS triangles,"
        "COALESCE(SUM(json_extract(c.stats,'$.vertices')),0) AS vertices FROM components c JOIN scenes s ON c.scene_id=s.id WHERE s.run_id=?",
        (run_id,),
    )
    run["llm"] = usage(state, run_id)
    return run
