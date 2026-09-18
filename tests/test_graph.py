"""Behavioral graph workflow tests; fixtures specify designs, not compiler categories."""

import sys

import numpy as np
import pytest
import trimesh

from scene_generator.design import Assembly, DesignGraph, DesignNode, Operation, Provenance
from scene_generator.generators import generate
from scene_generator.graph_compile import compile_graph, validate_geometry
from scene_generator.graph_planning import STAGES
from scene_generator.graph_spatial import resolve
from scene_generator.models import Bounds, Material, Transform
from scene_generator.pipeline import Pipeline, export_project, load_scene, validate_project
from scene_generator.policy import ScenePolicy
from scene_generator.util import file_hash, read_json, write_json


def node(identity, kind="object", **kwargs):
    return DesignNode(id=identity, kind=kind, description=identity, provenance=Provenance(source="user"), **kwargs)


def graph(nodes, size=(1, 1, 1)):
    return DesignGraph(
        id="design",
        title="User design",
        intent="Explicit test design",
        nodes=nodes,
        materials={"arbitrary": Material(name="iridescent woven moon dust", bevel=0)},
        recipes={
            "shape": Assembly(
                id="shape",
                visual_intent="test shape",
                parts=[Operation(capability="box", size=size, material="arbitrary")],
            )
        },
    )


def compile_test(g):
    p = ScenePolicy()
    resolved = resolve(g, p)
    nodes, coverage, report = compile_graph(g, resolved, p, "test", 42)
    validate_geometry(nodes, generate, p, report)
    return nodes, coverage, report


@pytest.mark.parametrize("kind", ["terrain", "surface", "object", "structure"])
def test_no_invented_buildings_rooms_paths_or_furniture(kind):
    g = graph([node("unfamiliar semantic form", kind, recipe="shape")], (12, 8, 0.3))
    nodes, _, report = compile_test(g)
    assert report.valid
    assert len(nodes) == 1
    assert nodes[0].kind == kind
    assert nodes[0].materials[0].name == "iridescent woven moon dust"
    assert not any(n.kind in {"room", "path"} for n in nodes)


def test_optional_empty_regions_and_zero_count():
    g = graph([node("empty", "region"), node("unused", count=0, recipe="shape")])
    nodes, coverage, report = compile_test(g)
    assert nodes == [] and report.valid
    assert coverage == [{"node": "unused", "source": "zero_count"}]


def test_nested_frames_rotation_scale_and_multiple_structures():
    g = graph(
        [
            node(
                "site",
                "region",
                spatial={"frame": {"contract_version": 2, "translation": (10, 0, 0), "rotation": (0, 0, 90)}},
            ),
            node(
                "level",
                "volume",
                parent_id="site",
                spatial={"frame": {"contract_version": 2, "translation": (2, 0, 5)}},
            ),
            node("a", "structure", parent_id="level", recipe="shape"),
            node(
                "b", "structure", recipe="shape", spatial={"frame": {"contract_version": 2, "translation": (-5, 0, 0)}}
            ),
        ]
    )
    nodes, _, report = compile_test(g)
    assert report.valid and len(nodes) == 2
    np.testing.assert_allclose(np.array(nodes[0].world_transform.matrix)[:3, 3], (10, 2, 5), atol=1e-8)
    assert nodes[0].world_bounds.min[2] == 5
    assert nodes[1].world_bounds.min[0] == -5


def test_anchor_attachment_and_non_stair_vertical_connection():
    g = graph(
        [
            node(
                "top",
                "volume",
                spatial={
                    "frame": {"contract_version": 2, "translation": (0, 0, 20)},
                    "anchors": [{"name": "dock", "position": (2, 3, 0)}],
                },
            ),
            node(
                "lift",
                "portal",
                spatial={"anchors": [{"name": "hook", "position": (0, 0, 1)}]},
                relationships=[
                    {"id": "hang", "kind": "attach", "target": "top", "anchor": "hook", "target_anchor": "dock"},
                    {"id": "travel", "kind": "connect", "target": "top"},
                ],
            ),
        ]
    )
    r = resolve(g, ScenePolicy())
    assert r.report.valid
    np.testing.assert_allclose(np.array(r.matrices["lift"])[:3, 3], (2, 3, 19))
    assert not any(n.recipe for n in g.nodes)


def test_irregular_polygon_containment_failure_not_aabb_only():
    g = graph(
        [
            node("triangular site", "region", spatial={"polygon": [(0, 0), (10, 0), (0, 10)], "height": 3}),
            node(
                "outside",
                recipe="shape",
                spatial={"frame": {"contract_version": 2, "translation": (8, 8, 0)}},
                relationships=[{"id": "must_fit", "kind": "contain", "target": "triangular site"}],
            ),
        ]
    )
    r = resolve(g, ScenePolicy())
    assert not r.report.valid
    d = next(d for d in r.report.diagnostics if d.constraint == "must_fit")
    assert d.nodes == ["outside", "triangular site"] and d.approximate


def test_negative_space_is_not_compiled_as_solid():
    g = graph(
        [node("void", "negative_space", spatial={"envelope": {"max": (2, 2, 2)}}), node("intrusion", recipe="shape")]
    )
    r = resolve(g, ScenePolicy())
    assert any(d.code == "negative_space" for d in r.report.diagnostics)
    nodes, _, _ = compile_graph(g, r, ScenePolicy(), "s", 1)
    assert len(nodes) == 1 and nodes[0].design_node_id == "intrusion"


def test_solver_conflicts_optional_relaxation_and_missing_anchor():
    g = graph(
        [
            node("a"),
            node("b"),
            node(
                "conflict",
                relationships=[
                    {"id": "left", "kind": "relative", "target": "a", "offset": (1, 0, 0)},
                    {"id": "right", "kind": "relative", "target": "b", "offset": (9, 0, 0)},
                    {"id": "optional", "kind": "anchor", "target": "a", "target_anchor": "absent", "required": False},
                ],
            ),
        ]
    )
    r = resolve(g, ScenePolicy(solver_iterations=3))
    assert not r.report.valid
    assert any(d.constraint == "left" for d in r.report.diagnostics)
    assert any(d.relaxed and d.compromise for d in r.report.diagnostics)


def test_sightline_blockage():
    g = graph(
        [
            node("eye"),
            node(
                "view",
                spatial={"frame": {"contract_version": 2, "translation": (10, 0, 0)}},
                relationships=[{"id": "sight", "kind": "visible", "target": "eye"}],
            ),
            node("blocker", recipe="shape", spatial={"frame": {"contract_version": 2, "translation": (4, -0.5, -0.5)}}),
        ]
    )
    r = resolve(g, ScenePolicy())
    assert any(d.constraint == "sight" and d.severity == "error" for d in r.report.diagnostics)


def test_explicit_repetition_and_determinism():
    g = graph(
        [
            node(
                "floating forms",
                recipe="shape",
                count=2,
                instances=[
                    Transform(contract_version=2, translation=(0, 0, 5)),
                    Transform(contract_version=2, translation=(3, 2, 9)),
                ],
            )
        ]
    )
    a, _, report = compile_test(g)
    b, _, _ = compile_test(g)
    assert report.valid and [n.model_dump() for n in a] == [n.model_dump() for n in b]
    assert [n.world_bounds.min[2] for n in a] == [5, 9]
    with pytest.raises(ValueError, match="explicit local frame"):
        node("invalid repetition", count=2)


@pytest.mark.parametrize(
    "fallback,source,valid", [("omit", "omitted", True), ("recipe", "recipe", True), ("error", None, False)]
)
def test_asset_misses_are_intentional_and_accounted(fallback, source, valid):
    from scene_generator.design import AssetRequest

    g = graph([node("unknown", recipe="shape", asset="request")])
    g.assets = {
        "request": AssetRequest(
            id="request",
            identity="unknown",
            role="focus",
            appearance="woven",
            material_intent="soft",
            placement_context="floating",
            fidelity="high",
            query="unknown",
            fallback=fallback,
        )
    }
    _, coverage, report = compile_test(g)
    assert report.valid == valid
    if source:
        assert coverage[0]["source"] == source
        assert coverage[0].get("reason") or coverage[0].get("fallback_reason")


def test_unsupported_operation_never_becomes_unrelated_box():
    g = graph([node("form", recipe="shape")])
    g.recipes["shape"].parts[0].capability = "unavailable_boolean"
    nodes, _, report = compile_test(g)
    assert nodes == [] and not report.valid
    assert any(d.code == "unsupported_operation" for d in report.diagnostics)


def test_user_site_and_budget_reject_without_resizing():
    g = graph([node("large", recipe="shape")], (20, 20, 20))
    r = resolve(g, ScenePolicy(site=Bounds(max=(2, 2, 2))))
    assert not r.report.valid
    assert r.bounds["large"].size == (20, 20, 20)
    p = ScenePolicy(max_triangles=1)
    nodes, _, report = compile_graph(g, resolve(g, p), p, "s", 1)
    assert not nodes and any(d.layer == "budget" for d in report.diagnostics)


def graph_config(config):
    config.generation.workflow = "graph"
    config.scenes = {"floating abstract environment": 1}
    return config


def test_graph_pipeline_resume_export_and_corruption(config):
    p = Pipeline.create(graph_config(config))
    try:
        assert p.run()["status"] == "complete"
        scene = next((p.path / "scenes").iterdir())
        assert validate_project(scene).valid
        manifest, nodes, paths = load_scene(scene)
        assert manifest["schema_version"] == 2
        assert trimesh.load(scene / "scene.glb", force="scene").geometry
        old = file_hash(scene / "scene.glb")
        calls = p.state.one("SELECT COUNT(*) AS n FROM llm_calls")["n"]
        assert calls == 9
        p.run()
        assert p.state.one("SELECT COUNT(*) AS n FROM llm_calls")["n"] == calls
        assert read_json(scene / "report.json")["regenerated_components_this_pass"] == 0
        assert file_hash(scene / "scene.glb") == old
        (scene / "planning/regions.json").write_text("{corrupt")
        (paths[nodes[0].id] / "geometry.npz").write_bytes(b"corrupt")
        p.run()
        assert validate_project(scene).valid
        assert read_json(scene / "report.json")["regenerated_components_this_pass"] == 1
        assert (scene / "planning/regions.invalid.json").exists()
        assert export_project(scene, "ply")["format"] == "ply"
        spec = nodes[0].model_dump(mode="json")
        spec["world_transform"]["affine_matrix"][0][3] = 999
        write_json(paths[nodes[0].id] / "spec.json", spec)
        assert not validate_project(scene).valid
        p.run()
        assert validate_project(scene).valid
    finally:
        p.close()


@pytest.mark.parametrize("stage", list(STAGES) + ["resolve", "assets", "compile", "generate", "validate", "export"])
def test_resume_at_every_planning_and_compilation_boundary(config, monkeypatch, stage):
    p = Pipeline.create(graph_config(config))
    original = p.stage

    def interrupt(sid, current):
        original(sid, current)
        if current == stage:
            raise KeyboardInterrupt()

    monkeypatch.setattr(p, "stage", interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            p.run()
        monkeypatch.setattr(p, "stage", original)
        assert p.run()["status"] == "complete"
        scene = next((p.path / "scenes").iterdir())
        assert len(list((scene / "planning").glob("*.json"))) == 10
        assert validate_project(scene).valid
    finally:
        p.close()


def test_graph_does_not_import_legacy(tmp_path):
    import subprocess

    code = """
import sys
from scene_generator.config import Config
from scene_generator.pipeline import Pipeline
c=Config.model_validate({'scenes':{'test':1},'generation':{'llm':{'mode':'mock'},'frameworks':['trimesh'],'output':{'root':sys.argv[1]}}})
p=Pipeline.create(c)
try: p.run()
finally: p.close()
assert not any(n.startswith('scene_generator.legacy') for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_unknown_versions_and_reference_cycles():
    from scene_generator.design import DesignGraph

    with pytest.raises(ValueError):
        DesignGraph.model_validate({"schema_version": 99, "id": "x", "title": "x", "intent": "x"})
    with pytest.raises(ValueError, match="cyclic"):
        graph([node("a", parent_id="b"), node("b", parent_id="a")])
    with pytest.raises(ValueError, match="missing"):
        graph([node("a", relationships=[{"id": "x", "kind": "contain", "target": "missing"}])])


def test_render_backend_compatibility():
    from scene_generator.graph_render import render_options

    g = graph(
        [
            node(
                "light",
                "light",
                properties={"light_type": "POINT", "color": (1, 1, 1), "intensity": 10, "required": True},
            )
        ]
    )
    r = resolve(g, ScenePolicy())
    render_options(g, r, "trimesh", "glb", r.report)
    assert not r.report.valid
    assert r.report.diagnostics[-1].layer == "capability"


def test_collision_permission_and_directed_separation():
    a = node("a", recipe="shape")
    b = node("b", recipe="shape")
    g = graph([a, b])
    assert any(d.code == "collision" for d in resolve(g, ScenePolicy()).report.diagnostics)
    b.relationships = [
        __import__("scene_generator.design", fromlist=["Constraint"]).Constraint(
            id="separate", kind="separate", target="a", offset=(1, 0, 0), distance=2
        )
    ]
    resolved = resolve(g, ScenePolicy())
    assert resolved.report.valid
    assert resolved.bounds["b"].min[0] == 3
    repair = resolved.report.repairs[0]
    assert repair.original != repair.new and repair.constraint == "separate" and repair.intent_impact
    b.relationships = [
        __import__("scene_generator.design", fromlist=["Constraint"]).Constraint(
            id="intentional", kind="overlap", target="a"
        )
    ]
    assert resolve(g, ScenePolicy()).report.valid


def test_explicit_metadata_migration_preserves_legacy_geometry():
    from scene_generator.contract_migration import migrate_component
    from scene_generator.models import Component

    old = Component(
        id="s/shape",
        name="shape",
        kind="arbitrary",
        bounds=Bounds(min=(2, 3, 4), max=(3, 4, 5)),
        local_transform=Transform(translation=(2, 3, 4)),
        world_transform=Transform(translation=(2, 3, 4)),
        seed=1,
    )
    migrated = migrate_component(old.model_dump())
    assert migrated.contract_version == 2 and migrated.id == old.id
    assert migrated.world_bounds == old.world_bounds
    assert migrated.local_transform.matrix == old.local_transform.matrix
    with pytest.raises(ValueError):
        migrate_component(migrated.model_dump(), 1)
    with pytest.raises(ValueError):
        migrate_component({**old.model_dump(), "contract_version": 99})


def test_explicit_irregular_mesh_and_different_scene_concepts():
    base = trimesh.creation.box(extents=(3, 2, 1))
    base.vertices -= base.bounds[0]
    # Sloping wedge, represented directly by its chosen geometry rather than a category compiler.
    base.vertices[:, 2] *= 0.3 + base.vertices[:, 0] / 3
    g = graph([node("landscape", "terrain", recipe="shape")])
    g.recipes["shape"].parts = [
        Operation(
            capability="mesh",
            size=(3, 2, 1.3),
            material="arbitrary",
            parameters={"vertices": base.vertices.tolist(), "faces": base.faces.tolist()},
        )
    ]
    landscape, _, report = compile_test(g)
    assert report.valid and len(landscape) == 1
    h = graph(
        [
            node(
                "floating installation",
                recipe="shape",
                count=3,
                instances=[Transform(contract_version=2, translation=(x, 0, z)) for x, z in [(0, 3), (4, 7), (8, 5)]],
            )
        ]
    )
    installation, _, report = compile_test(h)
    assert report.valid and len(installation) == 3
    assert {n.generator for n in landscape} != {n.generator for n in installation}


def test_empty_scene_glb_export(config, monkeypatch):
    import scene_generator.graph_planning as planning

    monkeypatch.setattr(planning, "mock_graph", lambda *_: graph([node("empty region", "region")]))
    p = Pipeline.create(graph_config(config))
    try:
        assert p.run()["status"] == "complete"
        scene = next((p.path / "scenes").iterdir())
        assert load_scene(scene)[1] == []
        assert validate_project(scene).valid
    finally:
        p.close()


@pytest.mark.parametrize("backend", ["trimesh", "blender"])
def test_graph_render_export_preserves_transforms_and_camera(config, monkeypatch, backend):
    import scene_generator.graph_planning as planning
    from scene_generator.backends.blender import discover_blender

    if backend == "blender" and not discover_blender():
        pytest.skip("Blender is not installed")
    g = graph(
        [
            node(
                "shape",
                recipe="shape",
                spatial={
                    "frame": {
                        "contract_version": 2,
                        "rotation": (0, 0, 45),
                        "scale": (2, 1, 1),
                        "translation": (4, 5, 6),
                    }
                },
            ),
            node(
                "camera",
                "camera",
                properties={"position": (12, 12, 12), "target": (4, 5, 6), "fov": 55, "required": True},
            ),
            node(
                "light",
                "light",
                properties={
                    "light_type": "POINT",
                    "position": (4, 5, 10),
                    "color": (1, 0.8, 0.5),
                    "intensity": 200,
                    "required": backend == "blender",
                },
            ),
        ]
    )
    monkeypatch.setattr(planning, "mock_graph", lambda *_: g)
    graph_config(config)
    config.generation.frameworks = [backend]
    p = Pipeline.create(config)
    try:
        assert p.run()["status"] == "complete"
        scene = next((p.path / "scenes").iterdir())
        loaded = trimesh.load(scene / "scene.glb", force="scene", process=False)
        assert loaded.geometry
        assert loaded.has_camera
        _, nodes, _ = load_scene(scene)
        expected = nodes[0].world_bounds
        # GLB basis conversion: (x,y,z) -> (x,z,-y).
        np.testing.assert_allclose(loaded.bounds[0], (expected.min[0], expected.min[2], -expected.max[1]), atol=1e-5)
        assert validate_project(scene).valid
    finally:
        p.close()


def test_typed_graph_retry_preserves_ids(tmp_path, monkeypatch):
    import json

    import httpx

    from scene_generator.checkpoint import State
    from scene_generator.config import LLMConfig
    from scene_generator.graph_planning import StageOutput
    from scene_generator.llm import LLM

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = State(tmp_path / "state.sqlite")
    calls = []
    valid = StageOutput(graph=graph([node("stable")]), rationale="preserve node").model_dump(mode="json")

    def handler(request):
        calls.append(json.loads(request.content))
        response = valid if len(calls) > 1 else {**valid, "graph": {**valid["graph"], "nodes": []}}
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(response)}}]}
        )

    llm = LLM(
        LLMConfig(mode="live", stream=False, retries=1, requests_per_minute=60000),
        state,
        "run",
        tmp_path / "cache",
        httpx.MockTransport(handler),
    )
    try:
        answer = llm.request(StageOutput, {}, "scene", validation_context={"prior_ids": ["stable"]})
        assert answer.graph.nodes[0].id == "stable" and len(calls) == 2
        assert "preserve existing stable IDs" in calls[1]["messages"][1]["content"]
    finally:
        llm.close()
        state.close()


def test_existing_zone_snapshot_remains_zone_workflow(config):
    config.generation.workflow = "creative"
    config.generation.quality = "draft"
    p = Pipeline.create(config)
    path = p.path
    p.close()
    resumed = Pipeline(path)
    try:
        assert resumed.generation.workflow == "creative"
    finally:
        resumed.close()


def test_pre_workflow_legacy_snapshot_is_not_reinterpreted(config):
    import json

    p = Pipeline.create(config)
    path = p.path
    snapshot = json.loads(p.state.one("SELECT config FROM runs")["config"])
    snapshot["generation"].pop("workflow")
    snapshot["generation"].pop("policy")
    p.state.execute("UPDATE runs SET config=?", (json.dumps(snapshot),))
    p.close()
    resumed = Pipeline(path)
    try:
        assert resumed.generation.workflow == "legacy"
        assert resumed.run()["status"] == "complete"
        scene = next((path / "scenes").iterdir())
        assert read_json(scene / "scene.json")["schema_version"] == 1
        assert export_project(scene, "ply")["format"] == "ply"
    finally:
        resumed.close()


def test_reusable_recipe_supports_resolve_locally():
    from scene_generator.design import Constraint

    g = graph(
        [
            node("assembly one", recipe="shape"),
            node("assembly two", recipe="shape", spatial={"frame": {"contract_version": 2, "translation": (10, 0, 8)}}),
        ]
    )
    g.recipes["shape"] = Assembly(
        id="shape",
        visual_intent="stacked assembly",
        parts=[
            Operation(id="base", capability="box", size=(2, 2, 1), material="arbitrary"),
            Operation(id="cap", capability="sphere", size=(1, 1, 1), material="arbitrary"),
        ],
        relationships=[Constraint(id="stack", source="cap", kind="support", target="base")],
    )
    nodes, _, report = compile_test(g)
    assert report.valid and len(nodes) == 4
    assert [n.world_bounds.min[2] for n in nodes] == [0, 1, 8, 9]
    assert report.repairs[0].node == "recipe/shape/cap"


def test_stale_planning_checkpoint_invalidates_dependents(config):
    p = Pipeline.create(graph_config(config))
    try:
        p.run()
        scene = next((p.path / "scenes").iterdir())
        checkpoint = read_json(scene / "planning/world.json")
        checkpoint["output"]["graph"]["intent"] = "untrusted changed checkpoint"
        write_json(scene / "planning/world.json", checkpoint)
        p.run()
        assert read_json(scene / "design-graph.json")["intent"] != "untrusted changed checkpoint"
        assert (scene / "planning/world.invalid.json").exists()
    finally:
        p.close()


def test_expansion_budget_precedes_spatial_allocation(monkeypatch):
    import scene_generator.graph_spatial as spatial

    g = graph(
        [
            node(
                "repeated",
                recipe="shape",
                count=2,
                instances=[Transform(contract_version=2), Transform(contract_version=2)],
            )
        ]
    )

    def forbidden(*args):
        raise AssertionError("spatial allocation ran before policy preflight")

    monkeypatch.setattr(spatial, "node_points", forbidden)
    resolved = spatial.resolve(g, ScenePolicy(max_parts=1))
    assert not resolved.report.valid
    assert resolved.report.diagnostics[0].code == "spatial_preflight"


def test_incremental_live_stage_updates_keep_existing_nodes():
    from scene_generator.graph_planning import StageDelta

    base = graph([node("existing", "region")])
    delta = StageDelta.model_validate(
        {
            "nodes": [node("new", recipe="shape", parent_id="existing").model_dump()],
            "rationale": "Add one object without retransmitting the region",
        },
        context={"previous_graph": base.model_dump()},
    )
    merged = delta.merge(base).graph
    assert [n.id for n in merged.nodes] == ["existing", "new"]
    assert merged.recipes == base.recipes
    with pytest.raises(ValueError, match="missing parent"):
        StageDelta.model_validate(
            {"nodes": [node("new", parent_id="absent").model_dump()], "rationale": "invalid"},
            context={"previous_graph": base.model_dump()},
        )
