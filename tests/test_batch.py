import pytest
from pydantic import ValidationError

from scene_generator.cli import main
from scene_generator.config import load_config
from scene_generator.llm import LLM
from scene_generator.pipeline import Pipeline
from scene_generator.planning import plan_scene


def test_shorthand_and_existing_generation_settings(tmp_path):
    counts = tmp_path / "counts.yaml"
    counts.write_text("museum: 10\naquarium: 5\n")
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        "scenes: {room: 1}\ngeneration:\n  seed: 73\n  llm: {mode: mock, timeout: 120}\n  output: {name: old-run}\n"
    )
    config = load_config(counts, settings)
    assert config.scenes == {"museum": 10, "aquarium": 5}
    assert config.generation.seed == 73
    assert config.generation.llm.timeout == 120
    assert config.generation.output.name is None
    assert load_config(settings).generation.output.name == "old-run"
    assert load_config(counts).generation.llm.mode == "live"


@pytest.mark.parametrize("yaml", ["museum: true", "museum: 0", "museum: 1.5", "museum: many", "{}", "[]", ""])
def test_invalid_shorthand(tmp_path, yaml):
    path = tmp_path / "counts.yaml"
    path.write_text(yaml)
    with pytest.raises(ValidationError):
        load_config(path)


def test_fifteen_variations_have_distinct_seeds_and_reproducible_plans(config):
    config.scenes = {"museum": 10, "aquarium": 5}
    pipeline = Pipeline.create(config)
    path = pipeline.path
    llm = LLM(config.generation.llm, pipeline.state, pipeline.run_id, path / "test-cache")
    try:
        rows = pipeline.state.rows("SELECT * FROM scenes ORDER BY category,ordinal")
        assert len(rows) == 15
        assert len({r["seed"] for r in rows}) == 15
        assert {r["id"] for r in rows} == {
            *(f"museum-{i:02d}" for i in range(1, 11)),
            *(f"aquarium-{i:02d}" for i in range(1, 6)),
        }
        plans = [plan_scene(r["category"], r["ordinal"], r["seed"], config.generation, llm, r["id"]) for r in rows]
        assert len({p.model_dump_json() for p in plans}) == 15
        museums = [p for p in plans if p.category == "museum"]
        assert [p.style for p in museums[:4]] == ["neoclassical", "modern", "brutalist", "neoclassical"]
        assert museums[0].environment == "urban" and museums[3].environment == "forest"
        assert len({p.layout for p in museums}) == 3
    finally:
        llm.close()
        pipeline.close()
    resumed = Pipeline(path)
    try:
        assert resumed.config.scenes == config.scenes
        assert resumed.state.rows("SELECT * FROM scenes ORDER BY category,ordinal") == rows
    finally:
        resumed.close()


def test_cli_shorthand_settings_and_resume(tmp_path, capsys):
    counts = tmp_path / "counts.yaml"
    counts.write_text("room: 1\nstudio: 1\n")
    settings = tmp_path / "settings.yaml"
    root = tmp_path / "runs"
    settings.write_text(
        f"scenes: {{museum: 10}}\ngeneration:\n  quality: draft\n  frameworks: [trimesh]\n"
        f"  llm: {{mode: mock}}\n  output: {{root: '{root}', name: ignored}}\n"
    )
    assert main(["generate", "--config", str(counts), "--settings", str(settings), "--name", "batch-test"]) == 0
    path = root / "batch-test"
    assert (path / "scenes/room-01/scene.glb").is_file()
    assert (path / "scenes/studio-01/scene.glb").is_file()
    assert not (path / "scenes/museum-01").exists()
    output = capsys.readouterr().err
    assert "[batch] scenes=2" in output and "index=2 total=2" in output
    assert main(["resume", "--run-id", "batch-test", "--root", str(root)]) == 0
