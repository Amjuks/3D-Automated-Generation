"""Versioned graph run orchestration, isolated from both prior planning compilers."""

import time
from pathlib import Path

import trimesh

from .assets import select_assets
from .backends.blender import BlenderBackend, discover_blender
from .backends.trimesh_backend import TrimeshBackend
from .design import DesignGraph, Diagnostic, RepairRecord
from .generators import load_geometry
from .graph_compile import compile_graph, validate_geometry
from .graph_planning import plan_graph
from .graph_render import render_options
from .graph_spatial import resolve
from .models import Issue, ValidationReport
from .policy import ScenePolicy
from .util import atomic_write, digest, file_hash, read_json, write_json

GRAPH_EXPORT_VERSION = 2


def choose_backend(generation):
    for name in generation.frameworks:
        if name == "blender" and discover_blender():
            return BlenderBackend(generation.blender_timeout)
        if name == "trimesh" and not generation.output.require_blender:
            return TrimeshBackend()
    raise RuntimeError("no configured graph export backend is available")


def write_report(path, report):
    write_json(path / "validation.json", {**report.model_dump(mode="json"), "valid": report.valid})


def require_valid(path, report):
    write_report(path, report)
    if not report.valid:
        write_json(
            path / "quality.json",
            {
                "schema_version": 2,
                "geometry_valid": False,
                "creative_fidelity": "not assessed",
                "diagnostics": [d.model_dump(mode="json") for d in report.diagnostics],
                "repairs": [r.model_dump(mode="json") for r in report.repairs],
            },
        )
        raise RuntimeError(f"graph validation failed; see {path / 'validation.json'}")


def graph_export(path, graph, resolved, nodes, paths, generation, report, fmt=None):
    fmt = fmt or generation.output.format
    if fmt not in {"glb", "obj", "ply"}:
        raise ValueError("unsupported export format")
    backend = choose_backend(generation)
    rendering = render_options(graph, resolved, backend.name, fmt, report)
    if generation.output.preview and not rendering["cameras"]:
        report.diagnostics.append(
            Diagnostic(
                layer="quality",
                code="preview_camera",
                severity="warning",
                message="preview unavailable: no supported design camera",
            )
        )
    require_valid(path, report)
    options = {
        "graph_render": rendering,
        "component_glbs": generation.output.component_glbs,
        "preview": generation.output.preview,
        "preview_samples": generation.output.preview_samples,
        "preview_width": generation.output.preview_width,
    }
    destination = path / ("scene." + fmt)
    result = backend.export(path, nodes, {n.id: paths[n.id] / "geometry.npz" for n in nodes}, destination, options)
    if fmt == "glb":
        loaded = trimesh.load(destination, force="scene", process=False)
        if nodes and not loaded.geometry:
            raise RuntimeError("export lost graph geometry")
        result["glb_reload_verified"] = True
    result.update(path=str(destination), sha256=file_hash(destination), format=fmt, schema_version=2)
    return result


def process_graph(pipeline, row, path):
    from .pipeline import component_path

    sid, generation = row["id"], pipeline.generation
    concept = pipeline.config.scene_descriptions.get(row["category"], row["category"])

    def search_candidates(design):
        requests = [
            {"type": "model", "query": design.assets[n.asset].query, "target_role": n.id}
            for n in design.nodes
            if n.asset in design.assets and n.count
        ]
        selected = select_assets(generation.assets, None, path, pipeline.log, requests=requests, seed=row["seed"])
        # Reuse flags are operational state, not part of the stable design request.
        return [
            {
                k: v
                for k, v in a.items()
                if k in {"id", "target_role", "license", "sha256", "triangles", "source_extents"}
            }
            for a in selected
        ]

    graph, policy = plan_graph(
        concept,
        row["seed"],
        generation,
        pipeline.llm,
        sid,
        path,
        lambda stage: pipeline.stage(sid, stage),
        asset_search=search_candidates,
    )
    prose = [f"# {graph.title}", "", graph.intent, "", "## Design nodes", ""]
    for node in graph.nodes:
        prose.extend(
            [
                f"- {node.id} ({node.kind}): {node.description}",
                f"  Parent: {node.parent_id or 'world'}; count: {node.count}; recipe: {node.recipe or 'none'}.",
            ]
        )
    prose.extend(["", "Geometry validity and creative fidelity are reported separately in quality.json."])
    atomic_write(path / "scene.md", ("\n".join(prose) + "\n").encode())
    pipeline.stage(sid, "resolve")
    resolved = resolve(graph, policy)
    write_json(path / "resolved.json", resolved.model_dump(mode="json"))
    require_valid(path, resolved.report)
    pipeline.stage(sid, "assets")
    requests = []
    for node in graph.nodes:
        request = graph.assets.get(node.asset)
        if request and node.count:
            requests.append({"type": "model", "query": request.query, "target_role": node.id})
    write_json(
        path / "asset-requests.json",
        {"schema_version": 2, "requests": {k: v.model_dump(mode="json") for k, v in graph.assets.items()}},
    )
    assets = select_assets(generation.assets, None, path, pipeline.log, requests=requests, seed=row["seed"])
    for asset in assets:
        pipeline.state.execute(
            "INSERT INTO asset_events(run_id,scene_id,asset_id,created,reused,bytes,license,source) VALUES (?,?,?,?,?,?,?,?)",
            (
                pipeline.run_id,
                sid,
                asset["id"],
                time.time(),
                asset.get("reused_this_pass", asset["reused"]),
                asset.get("bytes", 0) + sum(f["bytes"] for f in asset.get("files", [])),
                asset["license"],
                asset["source"],
            ),
        )
    pipeline.stage(sid, "compile")
    nodes, coverage, report = compile_graph(graph, resolved, policy, sid, row["seed"], assets)
    write_json(path / "asset-manifest.json", {"schema_version": 2, "requests": requests, "realizations": coverage})
    require_valid(path, report)
    paths = {n.id: component_path(path, n) for n in nodes}
    for node in nodes:
        spec_path = paths[node.id] / "spec.json"
        if spec_path.exists():
            try:
                saved_spec = (
                    read_json(spec_path)
                    if spec_path.stat().st_size <= policy.max_checkpoint_bytes
                    else "oversized checkpoint"
                )
            except (ValueError, OSError):
                saved_spec = "corrupt checkpoint"
            if saved_spec != node.model_dump(mode="json"):
                report.repairs.append(
                    RepairRecord(
                        node=node.design_node_id,
                        original=saved_spec,
                        new=node.model_dump(mode="json"),
                        reason="restored component from authoritative graph",
                        constraint="graph_component_contract",
                        intent_impact="design unchanged; checkpoint restored",
                    )
                )
        write_json(paths[node.id] / "spec.json", node.model_dump(mode="json"))
        write_json(paths[node.id] / "contract.json", node.model_dump(mode="json"))
    plan = {
        "workflow": "graph",
        "schema_version": 2,
        "title": graph.title,
        "concept": graph.intent,
        "category": row["category"],
    }
    write_json(path / "plan.json", plan)
    pipeline.state.execute("UPDATE scenes SET plan=? WHERE id=?", (__import__("json").dumps(plan), sid))
    manifest = {
        "schema_version": 2,
        "component_version": 2,
        "recipe_version": 2,
        "asset_manifest_version": 2,
        "scene_id": sid,
        "run_id": pipeline.run_id,
        "workflow": "graph",
        "units": "meters",
        "up_axis": "Z",
        "design_graph": "design-graph.json",
        "graph_hash": digest(graph.model_dump(mode="json")),
        "policy": policy.model_dump(mode="json"),
        "generation": generation.model_dump(mode="json"),
        "components": [str(paths[n.id].relative_to(path)) for n in nodes],
        "connections": [],
        "bounds": {k: v.model_dump(mode="json") for k, v in resolved.bounds.items()},
    }
    write_json(path / "scene.json", manifest)
    pipeline.stage(sid, "generate")
    regenerated = pipeline.generate_components(sid, nodes, paths)
    pipeline.stage(sid, "validate")
    validate_geometry(nodes, lambda n: load_geometry(paths[n.id] / "geometry.npz"), policy, report)
    require_valid(path, report)
    pipeline.state.validation(sid, report.model_dump(mode="json"))
    pipeline.stage(sid, "export")
    fingerprint = digest(
        {
            "version": GRAPH_EXPORT_VERSION,
            "manifest": manifest,
            "geometry": [(n.id, file_hash(paths[n.id] / "geometry.npz")) for n in nodes],
            "backend": choose_backend(generation).name,
            "assets": [{k: v for k, v in a.items() if k not in {"reused", "reused_this_pass"}} for a in assets],
        }
    )
    target = path / "export.json"
    export = None
    if target.exists():
        try:
            saved = read_json(target)
            if (
                saved.get("fingerprint") == fingerprint
                and Path(saved["path"]).is_file()
                and file_hash(saved["path"]) == saved["sha256"]
            ):
                export = saved
                render_options(graph, resolved, export["backend"], export["format"], report)
        except (ValueError, KeyError, OSError):
            pass
    if export is None:
        export = graph_export(path, graph, resolved, nodes, paths, generation, report)
        export["fingerprint"] = fingerprint
        write_json(target, export)
    write_report(path, report)
    write_json(
        path / "quality.json",
        {
            "schema_version": 2,
            "geometry_valid": report.valid,
            "creative_fidelity": "not assessed by geometric validation",
            "mock": generation.llm.mode == "mock",
            "realizations": coverage,
            "diagnostics": [d.model_dump(mode="json") for d in report.diagnostics],
            "repairs": [r.model_dump(mode="json") for r in report.repairs],
            "limitations": export.get("limitations", [])
            + (
                ["Mock mode exports an abstract fixture, not a semantic interpretation of the concept."]
                if generation.llm.mode == "mock"
                else []
            ),
        },
    )
    write_json(
        path / "planning/complete.json",
        {
            "schema_version": 2,
            "stage": "compile_validate_export",
            "request_id": fingerprint,
            "graph_hash": manifest["graph_hash"],
            "export_sha256": export["sha256"],
            "valid": report.valid,
        },
    )
    pipeline.indexes(path, nodes, [], assets)
    for artifact in path.glob("*.json"):
        pipeline.record_file(sid, artifact, "manifest")
    for artifact in (path / "planning").glob("*.json"):
        pipeline.record_file(sid, artifact, "checkpoint")
    pipeline.record_file(sid, Path(export["path"]), "export")
    pipeline.state.execute(
        "UPDATE scenes SET status='complete',stage='complete',ended=? WHERE id=?", (time.time(), sid)
    )
    write_json(
        path / "report.json",
        {
            "scene_id": sid,
            "status": "complete",
            "plan": plan,
            "export": export,
            "validation": {**report.model_dump(mode="json"), "valid": report.valid},
            "regenerated_components_this_pass": regenerated,
        },
    )
    pipeline.log.event(
        "scene_complete",
        scene=sid,
        triangles=report.stats["triangles"],
        regenerated=regenerated,
        backend=export["backend"],
    )


def validate_graph_project(path, nodes, paths, manifest):
    graph = DesignGraph.model_validate(read_json(Path(path) / "design-graph.json"))
    policy = ScenePolicy.model_validate(manifest["policy"])
    resolved = resolve(graph, policy)
    report = resolved.report
    if digest(graph.model_dump(mode="json")) != manifest["graph_hash"]:
        report.diagnostics.append(
            Diagnostic(layer="design", code="graph_hash", message="graph changed; resume to recompile")
        )
    from .pipeline import spec_hash

    expected, _, compiled = compile_graph(
        graph, resolved, policy, manifest["scene_id"], 0, read_json(Path(path) / "assets.json")
    )
    # Seeds are checkpointed per component; compare all other authoritative fields.
    actual = {n.id: n for n in nodes}
    if set(actual) != {n.id for n in expected}:
        report.diagnostics.append(
            Diagnostic(layer="design", code="components", message="component membership differs from graph")
        )
    for n in expected:
        saved = actual.get(n.id)
        if saved:
            n.seed = saved.seed
            if spec_hash(n) != spec_hash(saved):
                report.diagnostics.append(
                    Diagnostic(
                        layer="design",
                        code="component_contract",
                        nodes=[n.design_node_id],
                        message="component differs from graph; resume to rebuild",
                    )
                )
    report.diagnostics.extend(d for d in compiled.diagnostics if d not in report.diagnostics)
    validate_geometry(nodes, lambda n: load_geometry(paths[n.id] / "geometry.npz"), policy, report)
    write_report(Path(path), report)
    return ValidationReport(
        issues=[
            Issue(component_id=",".join(d.nodes), code=d.code, message=d.message, severity=d.severity)
            for d in report.diagnostics
        ],
        stats=report.stats,
        checks={"graph": "version 2 layers"},
    )


def export_graph_project(path, nodes, paths, manifest, generation, fmt):
    graph = DesignGraph.model_validate(read_json(path / "design-graph.json"))
    resolved = resolve(graph, ScenePolicy.model_validate(manifest["policy"]))
    result = graph_export(path, graph, resolved, nodes, paths, generation, resolved.report, fmt)
    write_json(path / f"export-{fmt}.json", result)
    return result
