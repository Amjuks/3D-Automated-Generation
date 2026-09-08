import pytest
from pydantic import ValidationError

from scene_generator.brief import SceneBrief, ZoneDesign, default_brief
from scene_generator.config import Config
from scene_generator.creative import compile_creative
from scene_generator.pipeline import Pipeline
from scene_generator.recipes import ObjectRecipe, object_key
from scene_generator.util import read_json
from scene_generator.validation import validate


def fixtures(category="orbital food farm", floors=1):
    brief = SceneBrief.model_validate(default_brief(category, 0, 123))
    brief.zones[1].floors = floors
    designs = [
        ZoneDesign(
            description="A unique space",
            floor_description="Natural ground",
            wall_description="Open edges",
            floor_texture="stone",
            wall_texture="brick",
            objects=[
                dict(
                    name="Helical vapor collector",
                    kind="vapor collector",
                    count=2,
                    material="metal",
                    asset_query="",
                    detail="A spiral mast",
                    width=1,
                    depth=1,
                    height=2,
                )
            ],
        )
        for _ in brief.zones
    ]
    recipe = ObjectRecipe.model_validate(
        {
            "description": "Spiral mast",
            "parts": [
                {"name": "support", "primitive": "cylinder", "position": [0, 0, 0], "size": [1, 1, 0.2]},
                {"name": "coil", "primitive": "torus", "position": [0.1, 0.1, 0.15], "size": [0.8, 0.8, 0.85]},
            ],
        }
    )
    return brief, designs, {object_key(designs[0].objects[0]): recipe}


def test_arbitrary_category_and_multi_floor_geometry():
    brief, designs, recipes = fixtures(floors=3)
    nodes, connections, _, coverage = compile_creative(
        brief, designs, recipes, [], "unknown-category", 123, Config(scenes={"anything": 1}).generation
    )
    report = validate(nodes, connections)
    assert report.valid, report.model_dump()
    assert any("stair-1-" in n.name for n in nodes)
    assert len([n for n in nodes if n.kind == "room" and not n.parameters.get("circulation")]) == 3
    assert all("museum" not in n.id for n in nodes)
    assert len(coverage) == 8


def test_freeform_scene_changes_physical_layout():
    brief, designs, recipes = fixtures()
    generation = Config(scenes={"anything": 1}).generation
    first = compile_creative(brief, designs, recipes, [], "x", 123, generation)[0]
    brief.composition = "freeform"
    brief.zones[0].x, brief.zones[0].y = 4, 2
    brief.zones[1].x, brief.zones[1].y = 28, 25
    second = compile_creative(brief, designs, recipes, [], "x", 123, generation)[0]
    assert first[0].bounds != second[0].bounds
    assert [n.world_transform for n in first if n.name == "zone-02"] != [
        n.world_transform for n in second if n.name == "zone-02"
    ]


def test_bad_recipe_and_overlapping_brief_rejected():
    with pytest.raises(ValidationError):
        ObjectRecipe.model_validate(
            {
                "description": "bad",
                "parts": [{"name": "oops", "primitive": "box", "position": [0.8, 0, 0], "size": [0.5, 1, 1]}],
            }
        )
    brief = default_brief("any", 0, 0)
    brief["composition"] = "freeform"
    with pytest.raises(ValidationError, match="3m"):
        SceneBrief.model_validate(brief)


def test_creative_resume_reuses_every_design_checkpoint(tmp_path):
    config = Config.model_validate(
        {
            "scenes": {"unlisted imaginary environment": 1},
            "generation": {
                "quality": "draft",
                "frameworks": ["trimesh"],
                "llm": {"mode": "mock"},
                "output": {"root": str(tmp_path)},
                "assets": {"mode": "mock"},
            },
        }
    )
    pipeline = Pipeline.create(config)
    try:
        assert pipeline.run()["status"] == "complete"
        scene = next((pipeline.path / "scenes").iterdir())
        assert (scene / "scene.md").exists()
        assert read_json(scene / "quality.json")["geometry_valid"]
        calls = pipeline.state.one("SELECT count(*) AS n FROM llm_calls")["n"]
        assert pipeline.run()["status"] == "complete"
        assert pipeline.state.one("SELECT count(*) AS n FROM llm_calls")["n"] == calls
        assert read_json(scene / "report.json")["regenerated_components_this_pass"] == 0
    finally:
        pipeline.close()


def test_all_material_roles_receive_distinct_textures():
    from scene_generator.assets import apply_assets

    brief, designs, recipes = fixtures()
    nodes, _, _, _ = compile_creative(brief, designs, recipes, [], "x", 123, Config(scenes={"anything": 1}).generation)
    apply_assets(
        nodes,
        [
            {"type": "texture", "id": "a", "path": "floor-a.png", "target_role": "zone-01/floor"},
            {"type": "texture", "id": "b", "path": "floor-b.png", "target_role": "zone-02/floor"},
            {"type": "texture", "id": "c", "path": "wall-c.png", "target_role": "zone-02/wall"},
        ],
    )
    for role, path in [
        ("zone-01/floor", "floor-a.png"),
        ("zone-02/floor", "floor-b.png"),
        ("zone-02/wall", "wall-c.png"),
    ]:
        selected = [n for n in nodes if n.parameters.get("material_role") == role]
        assert selected and all(n.materials[0].base_color_texture == path for n in selected)


def test_local_model_search_targets_arbitrary_roles(tmp_path):
    import trimesh

    from scene_generator.assets import select_assets
    from scene_generator.brief import compatibility_plan
    from scene_generator.logging import Log
    from scene_generator.util import write_json

    trimesh.creation.box().export(tmp_path / "source.glb")
    write_json(
        tmp_path / "catalog.json",
        [{"id": "vapor-collector", "type": "model", "tags": ["vapor collector"], "path": "source.glb"}],
    )
    cfg = Config(scenes={"anything": 1}).generation.assets
    cfg.mode = "local"
    cfg.local_catalog = tmp_path / "catalog.json"
    brief, _, _ = fixtures()
    records = select_assets(
        cfg,
        compatibility_plan(brief, "anything"),
        tmp_path,
        Log(tmp_path / "events.jsonl"),
        requests=[{"type": "model", "query": "vapor collector", "target_role": "zone-01/object-01"}],
    )
    assert len(records) == 1 and records[0]["target_role"] == "zone-01/object-01"


def test_required_renderer_does_not_silently_fallback(tmp_path, monkeypatch):
    from scene_generator.exporters import export_scene

    cfg = Config(scenes={"anything": 1}).generation
    cfg.output.require_blender = True
    monkeypatch.setattr("scene_generator.exporters.discover_blender", lambda: None)
    with pytest.raises(RuntimeError, match="requires Blender"):
        export_scene(tmp_path, [], {}, cfg, [], None)
