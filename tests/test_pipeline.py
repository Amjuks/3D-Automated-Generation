import numpy as np
import pytest
import trimesh

from scene_generator.generators import load_geometry
from scene_generator.pipeline import Pipeline, export_project, load_scene, validate_project
from scene_generator.util import file_hash, read_json, write_json
from scene_generator.validation import validate


def run(config):
    pipeline = Pipeline.create(config)
    try:
        result = pipeline.run()
        return pipeline.path, result
    finally:
        pipeline.close()


def test_end_to_end_resume_and_corruption_repair(config):
    path, result = run(config)
    assert result["status"] == "complete"
    scene = next((path / "scenes").iterdir())
    assert validate_project(scene).valid
    loaded = trimesh.load(scene / "scene.glb", force="scene")
    assert len(loaded.geometry) > 5
    assert np.isfinite(loaded.bounds).all()
    manifest, nodes, paths = load_scene(scene)
    mesh_nodes = [n for n in nodes if n.generator != "group"]
    times = {n.id: (paths[n.id] / "geometry.npz").stat().st_mtime_ns for n in mesh_nodes}
    export_hash = file_hash(scene / "scene.glb")
    pipeline = Pipeline(path)
    try:
        pipeline.run()
        assert read_json(scene / "report.json")["regenerated_components_this_pass"] == 0
        assert file_hash(scene / "scene.glb") == export_hash
        # Corrupt one independent leaf: only that leaf must regenerate.
        victim = next(n for n in mesh_nodes if n.name == "relief-0")
        (paths[victim.id] / "geometry.npz").write_bytes(b"interrupted write")
        pipeline.run()
        assert read_json(scene / "report.json")["regenerated_components_this_pass"] == 1
        for n in mesh_nodes:
            if n.id != victim.id:
                assert (paths[n.id] / "geometry.npz").stat().st_mtime_ns == times[n.id]
        assert validate_project(scene).valid
        assert pipeline.state.one("SELECT COUNT(*) AS n FROM llm_calls")["n"] == 2
    finally:
        pipeline.close()
    assert export_project(scene, "ply")["format"] == "ply"


def test_interruption_checkpoint(config, monkeypatch):
    pipeline = Pipeline.create(config)
    original = pipeline.generate_components

    def interrupt(*args):
        original(*args)
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline, "generate_components", interrupt)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run()
    assert pipeline.state.one("SELECT status FROM runs")["status"] == "interrupted"
    path = pipeline.path
    pipeline.close()
    resumed = Pipeline(path)
    try:
        assert resumed.run()["status"] == "complete"
        assert resumed.state.one("SELECT COUNT(*) AS n FROM llm_calls")["n"] == 2
    finally:
        resumed.close()


def test_collision_floating_connectivity_and_invalid_mesh(config):
    path, _ = run(config)
    scene = next((path / "scenes").iterdir())
    manifest, nodes, paths = load_scene(scene)
    node = next(n for n in nodes if n.name.startswith("seat-"))
    node.world_transform.translation = tuple(
        x + (1 if i == 2 else 0) for i, x in enumerate(node.world_transform.translation)
    )
    report = validate(nodes, [])
    assert {"floating", "transform", "connectivity"} <= {i.code for i in report.issues}
    _, nodes, paths = load_scene(scene)
    leaf = next(n for n in nodes if n.generator == "box")
    arrays = load_geometry(paths[leaf.id] / "geometry.npz")
    arrays["faces"] = arrays["faces"][:-1]
    report = validate(
        nodes,
        manifest["connections"],
        lambda n: arrays if n.id == leaf.id else load_geometry(paths[n.id] / "geometry.npz"),
    )
    assert "manifold" in {i.code for i in report.issues}


def test_seed_reproducibility_and_variation(config):
    config.scenes = {"museum": 4}
    path, _ = run(config)
    scenes = sorted((path / "scenes").iterdir())
    plans = [read_json(p / "plan.json") for p in scenes]
    assert len({(p["columns"], p["rows"], p["collection"], p["room_width"]) for p in plans}) == 4
    reference = file_hash(scenes[0] / "scene.glb")
    path2, _ = run(config)
    assert file_hash(sorted((path2 / "scenes").iterdir())[0] / "scene.glb") == reference


def test_constrained_dimensions(config):
    config.generation.dimensions = (16, 16, 4)
    config.generation.bounding_space = __import__("scene_generator.models", fromlist=["Bounds"]).Bounds(
        min=(100, 200, 5), max=(116, 216, 9)
    )
    path, _ = run(config)
    scene = next((path / "scenes").iterdir())
    _, nodes, _ = load_scene(scene)
    assert nodes[0].world_transform.translation == (100, 200, 5)
    assert config.generation.bounding_space.contains(nodes[0].world_bounds)


def test_selective_contract_repair(config):
    path, _ = run(config)
    scene = next((path / "scenes").iterdir())
    _, nodes, paths = load_scene(scene)
    victim = next(n for n in nodes if n.name == "relief-0")
    victim.world_transform.translation = (999, 999, 999)
    write_json(paths[victim.id] / "spec.json", victim.model_dump(mode="json"))
    pipeline = Pipeline(path)
    try:
        assert pipeline.run()["status"] == "complete"
        assert validate_project(scene).valid
    finally:
        pipeline.close()


def test_texture_budget_repair_drops_texture_maps(config):
    config.generation.max_texture_bytes = 0
    path, result = run(config)
    assert result["status"] == "complete"
    scene = next((path / "scenes").iterdir())
    assert validate_project(scene).valid
    _, nodes, _ = load_scene(scene)
    assert all(
        not texture
        for node in nodes
        for material in node.materials
        for texture in (material.base_color_texture, material.roughness_texture, material.normal_texture)
    )


def test_minimum_site_and_wall_mount_check(config):
    config.generation.dimensions = (11, 14, 3.5)
    path, _ = run(config)
    scene = next((path / "scenes").iterdir())
    manifest, nodes, paths = load_scene(scene)
    assert validate_project(scene).valid
    art = next(n for n in nodes if n.parameters.get("mounted"))
    art.world_transform.translation = (999, 999, 999)
    report = validate(nodes, manifest["connections"])
    assert "mount" in {issue.code for issue in report.issues}


def test_resume_completed_llm_work_without_api_key(config, monkeypatch):
    path, _ = run(config)
    pipeline = Pipeline(path)
    live_snapshot = pipeline.config.model_copy(deep=True)
    live_snapshot.generation.llm.mode = "live"
    pipeline.state.execute("UPDATE runs SET config=?", (live_snapshot.model_dump_json(),))
    pipeline.close()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pipeline = Pipeline(path)
    try:
        assert pipeline.run()["status"] == "complete"
        assert pipeline.state.one("SELECT COUNT(*) AS n FROM llm_calls")["n"] == 2
    finally:
        pipeline.close()
