import pytest
import trimesh

from scene_generator.backends.blender import discover_blender
from scene_generator.pipeline import Pipeline
from scene_generator.util import read_json


def test_missing_blender_falls_back(config, monkeypatch):
    monkeypatch.setattr("scene_generator.exporters.discover_blender", lambda: None)
    config.generation.frameworks = ["blender", "trimesh"]
    pipeline = Pipeline.create(config)
    try:
        pipeline.run()
        scene = next((pipeline.path / "scenes").iterdir())
        assert read_json(scene / "export.json")["backend"] == "trimesh"
    finally:
        pipeline.close()


def test_blender_only_fails_with_checkpoint(config, monkeypatch):
    monkeypatch.setattr("scene_generator.exporters.discover_blender", lambda: None)
    config.generation.frameworks = ["blender"]
    pipeline = Pipeline.create(config)
    try:
        with pytest.raises(RuntimeError, match="Blender unavailable"):
            pipeline.run()
        row = pipeline.state.one("SELECT status,stage FROM scenes")
        assert row == {"status": "failed", "stage": "export"}
        assert pipeline.state.one("SELECT COUNT(*) AS n FROM components WHERE status!='complete'")["n"] == 0
    finally:
        pipeline.close()


def test_blender_timeout_falls_back_to_trimesh(config, monkeypatch):
    from scene_generator.backends.blender import BlenderTimeoutError

    config.generation.frameworks = ["blender", "trimesh"]
    monkeypatch.setattr(
        "scene_generator.backends.blender.BlenderBackend.export",
        lambda *args: (_ for _ in ()).throw(BlenderTimeoutError("timed out")),
    )
    pipeline = Pipeline.create(config)
    try:
        assert pipeline.run()["status"] == "complete"
        scene = next((pipeline.path / "scenes").iterdir())
        assert read_json(scene / "export.json")["backend"] == "trimesh"
        assert (scene / "scene.glb").exists()
    finally:
        pipeline.close()


@pytest.mark.skipif(not discover_blender(), reason="Blender is optional; set BLENDER_PATH to exercise headless export")
def test_blender_headless_glb(config):
    config.generation.frameworks = ["blender"]
    pipeline = Pipeline.create(config)
    try:
        assert pipeline.run()["status"] == "complete"
        scene = next((pipeline.path / "scenes").iterdir())
        assert read_json(scene / "export.json")["backend"] == "blender"
        assert len(trimesh.load(scene / "scene.glb", force="scene").geometry) > 5
    finally:
        pipeline.close()
