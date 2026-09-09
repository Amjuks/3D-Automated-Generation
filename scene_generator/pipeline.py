import json
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .assets import apply_assets, select_assets
from .checkpoint import State, run_lock
from .config import Config
from .exporters import EXPORT_VERSION, export_scene
from .generators import generate, load_geometry, save_geometry
from .llm import LLM
from .logging import Log
from .models import Component
from .repair import repair_components
from .statistics import run_summary, usage
from .util import canonical, digest, file_hash, read_json, slug, stable_seed, write_json
from .validation import validate


def component_path(scene_path, node):
    # IDs are compiler-generated; never map arbitrary user/LLM strings to paths.
    parts = node.id.split("/")[1:]
    if any(not p or p in {".", ".."} or "\\" in p for p in parts):
        raise ValueError("unsafe component id")
    return scene_path / "components" / Path(*parts)


def load_scene(scene_path):
    scene_path = Path(scene_path).resolve()
    manifest = read_json(scene_path / "scene.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported scene manifest version")
    nodes, paths = [], {}
    for relative in manifest["components"]:
        path = (scene_path / relative).resolve()
        if not path.is_relative_to(scene_path):
            raise ValueError("component path escapes project")
        node = Component.model_validate(read_json(path / "spec.json"))
        nodes.append(node)
        paths[node.id] = path
    return manifest, nodes, paths


def spec_hash(node):
    data = node.model_dump(mode="json")
    data.pop("validation_state")
    return digest(data)


class Pipeline:
    def __init__(self, run_path):
        self.path = Path(run_path).resolve()
        self.state = State(self.path / "state.sqlite")
        run = self.state.one("SELECT * FROM runs")
        if not run:
            self.state.close()
            raise ValueError("run checkpoint not found")
        self.run_id = run["id"]
        snapshot = json.loads(run["config"])
        snapshot["generation"].setdefault("workflow", "legacy")
        self.config = Config.model_validate(snapshot)
        self.generation = self.config.generation
        self.log = Log(self.path / "events.jsonl")
        self.llm = None

    @classmethod
    def create(cls, config):
        config = config.model_copy(deep=True)
        if config.generation.seed is None:
            config.generation.seed = secrets.randbits(63)
        categories = "-".join(f"{slug(k)}-{v}" for k, v in config.scenes.items())[:65]
        base = (
            slug(config.generation.output.name)
            if config.generation.output.name
            else f"{categories}-{datetime.now():%Y-%m-%d-%H%M%S}-s{config.generation.seed}"
        )
        root = config.generation.output.root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        suffix = 1
        while True:
            run_id = base if suffix == 1 else f"{base}-{suffix:02d}"
            path = root / run_id
            try:
                path.mkdir(exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
        # Resolve filesystem settings in the snapshot so resume is cwd-independent.
        config = config.model_copy(deep=True)
        config.generation.output.root = config.generation.output.root.resolve()
        config.generation.assets.cache_dir = config.generation.assets.cache_dir.resolve()
        if config.generation.assets.local_catalog:
            config.generation.assets.local_catalog = config.generation.assets.local_catalog.resolve()
        state = State(path / "state.sqlite")
        state.execute(
            "INSERT INTO runs(id,config,status,stage,started) VALUES (?,?,?,?,?)",
            (run_id, config.model_dump_json(), "pending", "plan", time.time()),
        )
        used_categories = set()
        for category, count in config.scenes.items():
            category_base = slug(category)
            category_name = category_base
            suffix = 2
            while category_name in used_categories:
                category_name = f"{category_base}-{suffix}"
                suffix += 1
            used_categories.add(category_name)
            for ordinal in range(count):
                scene_id = f"{category_name}-{ordinal + 1:02d}"
                seed = stable_seed(config.generation.seed, category, ordinal)
                state.execute(
                    "INSERT INTO scenes(id,run_id,category,ordinal,seed) VALUES (?,?,?,?,?)",
                    (scene_id, run_id, category, ordinal, seed),
                )
        state.close()
        write_json(path / "config.json", config.model_dump(mode="json"))
        return cls(path)

    def close(self):
        if self.llm:
            self.llm.close()
        self.state.close()

    def stage(self, scene_id, stage):
        self.state.execute("UPDATE runs SET stage=? WHERE id=?", (stage, self.run_id))
        self.state.execute("UPDATE scenes SET stage=? WHERE id=?", (stage, scene_id))
        self.log.event("stage", scene=scene_id, stage=stage)

    def run(self):
        with run_lock(self.path / ".lock"):
            self.state.execute("UPDATE runs SET status='running',ended=NULL,error=NULL WHERE id=?", (self.run_id,))
            self.log.event("run", id=self.run_id, path=str(self.path))
            try:
                if self.llm:
                    self.llm.close()
                self.llm = LLM(
                    self.generation.llm,
                    self.state,
                    self.run_id,
                    self.generation.output.root / ".llm-cache",
                    log=self.log,
                )
                scenes = self.state.rows(
                    "SELECT * FROM scenes WHERE run_id=? ORDER BY category,ordinal", (self.run_id,)
                )
                self.log.event("batch", scenes=len(scenes), categories=self.config.scenes)
                for index, row in enumerate(scenes, 1):
                    try:
                        self.log.event(
                            "scene",
                            id=row["id"],
                            index=index,
                            total=len(scenes),
                            variation=row["ordinal"] + 1,
                            category=row["category"],
                        )
                        self.process_scene(row)
                    except BaseException as exc:
                        self.state.execute(
                            "UPDATE scenes SET status=?,ended=?,error=? WHERE id=?",
                            (
                                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                time.time(),
                                type(exc).__name__,
                                row["id"],
                            ),
                        )
                        raise
                self.state.execute(
                    "UPDATE runs SET status='complete',stage='complete',ended=? WHERE id=?", (time.time(), self.run_id)
                )
            except BaseException as exc:
                self.state.execute(
                    "UPDATE runs SET status=?,ended=?,error=? WHERE id=?",
                    (
                        "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        time.time(),
                        type(exc).__name__,
                        self.run_id,
                    ),
                )
                self.log.event("run_stopped", error=type(exc).__name__, resume_id=self.run_id)
                raise
            finally:
                write_json(self.path / "summary.json", run_summary(self.state, self.run_id))
            self.log.event("complete", id=self.run_id)
        return run_summary(self.state, self.run_id)

    def process_scene(self, row):
        sid = row["id"]
        path = self.path / "scenes" / sid
        path.mkdir(parents=True, exist_ok=True)
        self.state.execute(
            "UPDATE scenes SET status='running',started=COALESCE(started,?),ended=NULL,error=NULL WHERE id=?",
            (time.time(), sid),
        )
        creative = self.generation.workflow == "creative"
        brief = designs = None
        requests = None
        if creative:
            from .assets import design_asset_requests
            from .brief import compatibility_plan, design_scene

            self.stage(sid, "brief")
            previous_concepts = [
                {"title": p["title"], "concept": p.get("concept", "")[:200]}
                for p in (
                    json.loads(r["plan"])
                    for r in self.state.rows(
                        "SELECT plan FROM scenes WHERE run_id=? AND id<>? AND plan IS NOT NULL ORDER BY ordinal DESC LIMIT 3",
                        (self.run_id, sid),
                    )
                )
            ]
            brief, designs = design_scene(
                row["category"],
                row["ordinal"],
                row["seed"],
                self.generation,
                self.llm,
                sid,
                path,
                self.log,
                previous_concepts,
                description=self.config.scene_descriptions.get(row["category"], ""),
            )
            plan = compatibility_plan(brief, row["category"])
            requests = design_asset_requests(brief, designs)
            write_json(path / "asset-requests.json", requests)
            self.state.execute("UPDATE scenes SET plan=? WHERE id=?", (plan.model_dump_json(), sid))
        else:
            from .legacy.models import ScenePlan
            from .legacy.planning import plan_scene

            self.stage(sid, "plan")
            if row["plan"]:
                plan = ScenePlan.model_validate_json(row["plan"])
            else:
                plan = plan_scene(row["category"], row["ordinal"], row["seed"], self.generation, self.llm, sid)
                self.state.execute("UPDATE scenes SET plan=? WHERE id=?", (plan.model_dump_json(), sid))
        write_json(path / "plan.json", plan.model_dump(mode="json"))
        if creative and not (path / "scene.json").exists():
            from .creative import compile_creative

            # Validate allocations before downloading assets or paying for object recipes.
            self.stage(sid, "layout")
            compile_creative(brief, designs, None, [], sid, row["seed"], self.generation)
        self.stage(sid, "assets")
        assets = select_assets(self.generation.assets, plan, path, self.log, requests=requests, seed=row["seed"])
        for asset in assets:
            self.state.execute(
                "INSERT INTO asset_events(run_id,scene_id,asset_id,created,reused,bytes,license,source) VALUES (?,?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    sid,
                    asset["id"],
                    time.time(),
                    asset.get("reused_this_pass", asset["reused"]),
                    asset.get("bytes", 0) + sum(f["bytes"] for f in asset.get("files", [])),
                    asset["license"],
                    asset["source"],
                ),
            )
            self.state.execute(
                "INSERT OR REPLACE INTO assets(id,scene_id,metadata,reused) VALUES (?,?,?,?)",
                (asset["id"], sid, canonical(asset), asset["reused"]),
            )
        self.stage(sid, "decompose")
        if (path / "scene.json").exists():
            manifest, nodes, paths = load_scene(path)
            connections = manifest["connections"]
        else:
            if creative:
                from .creative import compile_creative
                from .recipes import design_recipes

                self.stage(sid, "objects")
                recipes = design_recipes(
                    designs,
                    self.llm,
                    sid,
                    path,
                    self.log,
                    assets,
                    seed=row["seed"],
                    brief=brief,
                    quality=self.generation.quality,
                )
                nodes, connections, tokens, coverage = compile_creative(
                    brief, designs, recipes, assets, sid, row["seed"], self.generation
                )
                selected_roles = {a.get("target_role") for a in assets}
                write_json(
                    path / "asset-coverage.json",
                    {
                        "objects": coverage,
                        "unfulfilled_requests": [r for r in requests if r.get("target_role") not in selected_roles],
                    },
                )
            else:
                from .legacy.decomposition import decompose

                nodes, connections, tokens = decompose(
                    plan, sid, row["seed"], self.generation, self.llm, path / "room-designs"
                )
            apply_assets(nodes, assets, legacy=not creative)
            from .surfaces import texture_scene

            texture_scene(nodes, self.generation.output.root / ".surface-cache", self.generation.quality)
            paths = {n.id: component_path(path, n) for n in nodes}
            for node in nodes:
                write_json(paths[node.id] / "spec.json", node.model_dump(mode="json"))
                write_json(paths[node.id] / "contract.json", node.model_dump(mode="json"))
            write_json(path / "design-tokens.json", {k: v.model_dump(mode="json") for k, v in tokens.items()})
            manifest = {
                "schema_version": 1,
                "scene_id": sid,
                "run_id": self.run_id,
                "units": "meters",
                "up_axis": "Z",
                "plan": "plan.json",
                "components": [str(paths[n.id].relative_to(path)) for n in nodes],
                "connections": connections,
                "generation": self.generation.model_dump(mode="json"),
            }
            write_json(path / "scene.json", manifest)
        self.stage(sid, "generate")
        regenerated = self.generate_components(sid, nodes, paths)
        self.stage(sid, "validate")
        for attempt in range(self.generation.repair_attempts + 1):
            report = validate(
                nodes,
                connections,
                lambda n: load_geometry(paths[n.id] / "geometry.npz"),
                self.generation.max_scene_triangles,
                self.generation.max_texture_bytes,
            )
            self.state.validation(sid, report.model_dump(mode="json"))
            write_json(path / "validation.json", {**report.model_dump(mode="json"), "valid": report.valid})
            if report.valid:
                break
            if attempt == self.generation.repair_attempts:
                raise RuntimeError(
                    f"scene validation failed with {len(report.issues)} issues; see {path / 'validation.json'}"
                )
            nodes, invalidated = repair_components(nodes, report, paths)
            if not invalidated:
                raise RuntimeError(f"no safe automatic repair for validation issues; see {path / 'validation.json'}")
            self.log.event("repair", scene=sid, components=len(invalidated), attempt=attempt + 1)
            for node in nodes:
                if node.id in invalidated:
                    write_json(paths[node.id] / "spec.json", node.model_dump(mode="json"))
                    self.state.execute("UPDATE components SET status='pending' WHERE id=?", (node.id,))
            regenerated += self.generate_components(sid, nodes, paths)
        for node in nodes:
            if node.validation_state != "valid":
                node.validation_state = "valid"
                write_json(paths[node.id] / "spec.json", node.model_dump(mode="json"))
        self.indexes(path, nodes, connections, assets)
        for node in nodes:
            self.record_file(sid, paths[node.id] / "spec.json", "component_manifest", node.id)
        for artifact in path.glob("*.json"):
            self.record_file(sid, artifact, "manifest")
        for artifact in (path / "indexes").glob("*.json"):
            self.record_file(sid, artifact, "index")
        self.stage(sid, "export")
        export_report = path / "export.json"
        export = read_json(export_report) if export_report.exists() else None
        fingerprint = digest(
            {
                "export_version": EXPORT_VERSION,
                "components": [spec_hash(n) for n in nodes],
                "generation": self.generation.model_dump(mode="json"),
                "assets": [(a["id"], a["sha256"]) for a in assets],
            }
        )
        valid_export = (
            export
            and export.get("fingerprint") == fingerprint
            and Path(export["path"]).is_file()
            and file_hash(export["path"]) == export["sha256"]
        )
        if regenerated or not valid_export:
            export = export_scene(
                path,
                nodes,
                {n.id: paths[n.id] / "geometry.npz" for n in nodes if n.generator != "group"},
                self.generation,
                assets,
                plan,
            )
            export["fingerprint"] = fingerprint
            write_json(export_report, export)
        if creative:
            coverage = read_json(path / "asset-coverage.json")
            by_id = {n.id: n for n in nodes}
            for entry in coverage["objects"]:
                obj = by_id.get(entry["object"])
                if obj:
                    repairs = [
                        by_id[c].parameters["repair"] for c in obj.child_ids if by_id[c].parameters.get("repair")
                    ]
                    if repairs:
                        entry.update(source="placeholder", repairs=repairs)
            write_json(path / "asset-coverage.json", coverage)
            write_json(
                path / "quality.json",
                {
                    "geometry_valid": report.valid,
                    "renderer": export["backend"],
                    "preview_available": (path / "scene.png").exists(),
                    "downloaded_assets": len(assets),
                    "distinct_textures": len(
                        {m.base_color_texture for n in nodes for m in n.materials if m.base_color_texture}
                    ),
                    "asset_instances": sum(o["source"] == "asset" for o in coverage["objects"]),
                    "recipe_instances": sum(o["source"] == "recipe" for o in coverage["objects"]),
                    "placeholder_instances": sum(o["source"] == "placeholder" for o in coverage["objects"]),
                    "scaled_objects": sum(o["scale"] < 0.99 for o in coverage["objects"]),
                    "unfulfilled_asset_requests": len(coverage["unfulfilled_requests"]),
                    "limitations": export.get("limitations", [])
                    + [
                        "Procedural assemblies are approximations, not a guarantee of photorealism.",
                        "Weather is an art-direction brief; volumetric weather and character animation are not simulated.",
                        "Geometry validation does not assess visual realism or likeness.",
                    ],
                },
            )
            self.record_file(sid, path / "scene.md", "design_brief")
            self.record_file(sid, path / "asset-coverage.json", "manifest")
            self.record_file(sid, path / "quality.json", "manifest")
        self.record_file(sid, export_report, "manifest")
        self.record_file(sid, Path(export["path"]), "export")
        for preview in path.glob("*.png"):
            self.record_file(sid, preview, "preview")
        if (path / "scene.blend").exists():
            self.record_file(sid, path / "scene.blend", "blender_project")
        if self.generation.output.component_glbs:
            for node in nodes:
                target = paths[node.id] / "geometry.glb"
                if target.exists():
                    self.record_file(sid, target, "component_export", node.id)
        self.state.execute(
            "UPDATE scenes SET status='complete',stage='complete',ended=? WHERE id=?", (time.time(), sid)
        )
        scene_state = self.state.one("SELECT started,ended FROM scenes WHERE id=?", (sid,))
        write_json(
            path / "report.json",
            {
                "scene_id": sid,
                "status": "complete",
                "plan": plan.model_dump(mode="json"),
                "validation": {"valid": report.valid, **report.model_dump(mode="json")},
                "export": export,
                "llm": usage(self.state, self.run_id, sid),
                "assets": assets,
                "regenerated_components_this_pass": regenerated,
                "duration_seconds": scene_state["ended"] - scene_state["started"],
                "started": scene_state["started"],
                "ended": scene_state["ended"],
            },
        )
        self.log.event(
            "scene_complete",
            scene=sid,
            triangles=report.stats["triangles"],
            regenerated=regenerated,
            backend=export["backend"],
        )

    def generate_components(self, sid, nodes, paths):
        todo = []
        changed = set()
        for node in nodes:
            previous = self.state.one("SELECT * FROM components WHERE id=?", (node.id,))
            target = paths[node.id] / "geometry.npz"
            key = spec_hash(node)
            valid = (
                previous
                and previous["status"] == "complete"
                and previous["spec_hash"] == key
                and not any(d in changed for d in node.dependencies)
                and (node.generator == "group" or (target.exists() and file_hash(target) == previous["geometry_hash"]))
            )
            if valid:
                continue
            changed.add(node.id)
            self.state.execute(
                "INSERT INTO components(id,scene_id,parent_id,path,spec_hash,status,started) VALUES (?,?,?,?,?,'running',?) ON CONFLICT(id) DO UPDATE SET spec_hash=excluded.spec_hash,status='running',started=excluded.started,ended=NULL,error=NULL",
                (node.id, sid, node.parent_id, str(paths[node.id]), key, time.time()),
            )
            self.state.execute("DELETE FROM dependencies WHERE component_id=?", (node.id,))
            for dep in node.dependencies:
                self.state.execute("INSERT INTO dependencies VALUES (?,?)", (node.id, dep))
            if node.generator == "group":
                self.state.execute("UPDATE components SET status='complete',ended=? WHERE id=?", (time.time(), node.id))
            else:
                todo.append(node)

        def work(node):
            try:
                data = generate(node)
                save_geometry(paths[node.id] / "geometry.npz", data)
                return node, {"triangles": len(data["faces"]), "vertices": len(data["vertices"])}, None
            except Exception as exc:
                return node, {}, type(exc).__name__

        with ThreadPoolExecutor(max_workers=self.generation.workers) as pool:
            # Bounded submission batches avoid queuing a whole large scene in memory.
            for offset in range(0, len(todo), self.generation.workers * 4):
                for node, stats, error in pool.map(work, todo[offset : offset + self.generation.workers * 4]):
                    target = paths[node.id] / "geometry.npz"
                    if error:
                        target.unlink(missing_ok=True)
                        self.state.execute("DELETE FROM files WHERE path=?", (str(target),))
                    sha = file_hash(target) if not error else None
                    self.state.execute(
                        "UPDATE components SET status=?,geometry_hash=?,attempts=attempts+1,ended=?,stats=?,error=? WHERE id=?",
                        ("failed" if error else "complete", sha, time.time(), canonical(stats), error, node.id),
                    )
                    if not error:
                        self.record_file(sid, target, "geometry", node.id)
                if len(todo) > 100:
                    self.log.event(
                        "geometry_progress",
                        scene=sid,
                        processed=min(offset + self.generation.workers * 4, len(todo)),
                        total=len(todo),
                    )
        return len(todo)

    def record_file(self, sid, path, kind, component_id=None):
        self.state.execute(
            "INSERT OR REPLACE INTO files(path,scene_id,component_id,sha256,bytes,kind) VALUES (?,?,?,?,?,?)",
            (str(path), sid, component_id, file_hash(path), path.stat().st_size, kind),
        )

    def indexes(self, path, nodes, connections, assets):
        write_json(path / "indexes/spatial.json", {n.id: n.world_bounds.model_dump(mode="json") for n in nodes})
        write_json(path / "indexes/dependencies.json", {n.id: n.dependencies for n in nodes})
        write_json(path / "indexes/connections.json", connections)
        write_json(path / "indexes/assets.json", assets)


def _validate_project(path):
    manifest, nodes, paths = load_scene(path)
    from .config import Generation

    generation = Generation.model_validate(manifest["generation"])
    report = validate(
        nodes,
        manifest["connections"],
        lambda n: load_geometry(paths[n.id] / "geometry.npz"),
        generation.max_scene_triangles,
        generation.max_texture_bytes,
    )
    write_json(Path(path) / "validation.json", {"valid": report.valid, **report.model_dump(mode="json")})
    return report


def _export_project(path, fmt):
    path = Path(path).resolve()
    manifest, nodes, paths = load_scene(path)
    from .config import Generation

    generation = Generation.model_validate(manifest["generation"])
    report = _validate_project(path)
    if not report.valid:
        raise RuntimeError("validation failed; repair with resume before exporting")
    data = read_json(path / "plan.json")
    if data.get("workflow") == "creative":
        from .brief import SceneSummary

        plan = SceneSummary.model_validate(data)
    else:
        from .legacy.models import ScenePlan

        plan = ScenePlan.model_validate(data)
    assets = read_json(path / "assets.json")
    result = export_scene(
        path,
        nodes,
        {n.id: paths[n.id] / "geometry.npz" for n in nodes if n.generator != "group"},
        generation,
        assets,
        plan,
        fmt,
    )

    write_json(path / f"export-{fmt}.json", result)
    if fmt == generation.output.format:
        previous = read_json(path / "export.json") if (path / "export.json").exists() else {}
        write_json(path / "export.json", {**previous, **result})
    database = path.parent.parent / "state.sqlite"
    if database.exists():
        state = State(database)
        try:
            state.execute(
                "INSERT OR REPLACE INTO files(path,scene_id,component_id,sha256,bytes,kind) VALUES (?,?,NULL,?,?,?)",
                (result["path"], manifest["scene_id"], result["sha256"], result["bytes"], "export"),
            )
            state.validation(manifest["scene_id"], report.model_dump(mode="json"))
            write_json(path.parent.parent / "summary.json", run_summary(state, manifest["run_id"]))
        finally:
            state.close()
    return result


def project_lock(path):
    path = Path(path).resolve()
    root = path.parent.parent if (path.parent.parent / "state.sqlite").exists() else path
    return run_lock(root / ".lock")


def validate_project(path):
    with project_lock(path):
        return _validate_project(path)


def export_project(path, fmt):
    with project_lock(path):
        return _export_project(path, fmt)
