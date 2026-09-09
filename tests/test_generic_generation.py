import subprocess
import sys
from pathlib import Path

import pytest

from scene_generator.brief import SceneBrief, ZoneDesign, default_brief, design_scene
from scene_generator.config import Config
from scene_generator.creative import compile_creative
from scene_generator.generators import generate
from scene_generator.layout import object_placements
from scene_generator.logging import Log
from scene_generator.pipeline import Pipeline, export_project
from scene_generator.recipes import ObjectRecipe, design_recipes, object_key
from scene_generator.util import digest, file_hash, read_json, write_json
from scene_generator.validation import validate


def scene_data():
    brief = SceneBrief.model_validate(default_brief("unlisted structure", 0, 91))
    designs = [
        ZoneDesign(
            description="A zone",
            floor_description="",
            wall_description="",
            floor_texture="",
            wall_texture="",
            objects=[
                dict(
                    name="Resonator",
                    kind="custom resonator",
                    count=3,
                    material="metal",
                    detail="A shaped resonant shell",
                    width=1,
                    depth=1,
                    height=2,
                )
            ],
        )
        for _ in brief.zones
    ]
    recipe = ObjectRecipe(
        description="Custom shape",
        parts=[
            dict(
                name="shape",
                primitive="lathe",
                position=[0, 0, 0],
                size=[1, 1, 1],
                profile=[[0, 0], [0.5, 0], [0.25, 1], [0, 1]],
            )
        ],
    )
    return brief, designs, {object_key(designs[0].objects[0]): recipe}


def test_creative_pipeline_does_not_import_legacy_compilers():
    script = "from scene_generator.pipeline import Pipeline; from scene_generator.creative import compile_creative; import sys; assert not any(x.startswith('scene_generator.legacy') for x in sys.modules)"
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).parents[1])


def test_paths_follow_freeform_footprints_and_can_be_omitted():
    brief, designs, recipes = scene_data()
    brief.composition = "freeform"
    brief.zones[0].x, brief.zones[0].y = 1, 20
    brief.zones[1].x, brief.zones[1].y = 24, 3
    config = Config(scenes={"unlisted": 1}).generation
    nodes, connections, _, _ = compile_creative(brief, designs, recipes, [], "x", 17, config)
    zones = [n for n in nodes if n.name in {"zone-01", "zone-02"}]
    paths = [n for n in nodes if n.name.startswith("path-")]
    assert paths and all(not p.world_bounds.intersects(z.world_bounds) for p in paths for z in zones)
    assert validate(nodes, connections).valid
    brief.circulation = "none"
    brief.zones[1].windows = False
    brief.zones[1].roof = False
    designs[0].objects = []
    nodes, connections, _, coverage = compile_creative(brief, designs, recipes, [], "x", 17, config)
    assert not any(n.name.startswith("path-") or n.name in {"glazing", "ceiling"} for n in nodes)
    assert len(coverage) == 3 and validate(nodes, connections).valid


@pytest.mark.parametrize("arrangement", ["scattered", "clustered", "perimeter", "rows"])
def test_seeded_placements_are_valid_and_repeatable(arrangement):
    brief, designs, recipes = scene_data()
    designs[0].arrangement = arrangement
    zone = brief.zones[0]
    first = object_placements(zone, designs[0], 0, 123)
    assert first == object_placements(zone, designs[0], 0, 123)
    if arrangement != "rows":
        assert first != object_placements(zone, designs[0], 0, 124)
    for i, (_, _, _, size, _, x, y) in enumerate(first):
        assert x >= 0 and y >= 0 and x + size[0] <= zone.width and y + size[1] <= zone.depth
        for _, _, _, other, _, ox, oy in first[:i]:
            assert x + size[0] <= ox or ox + other[0] <= x or y + size[1] <= oy or oy + other[1] <= y


def test_object_counts_floor_targets_and_explicit_repetition():
    brief, designs, _ = scene_data()
    zone = brief.zones[1]
    zone.floors = 3
    assert sum(len(object_placements(zone, designs[1], f, 3)) for f in range(3)) == 3
    designs[1].objects[0].floor = 2
    assert not object_placements(zone, designs[1], 0, 3)
    assert len(object_placements(zone, designs[1], 2, 3)) == 3
    designs[1].objects[0].repeat_on_floors = True
    assert sum(len(object_placements(zone, designs[1], f, 3)) for f in range(3)) == 9


def test_new_primitives_create_valid_geometry():
    brief, designs, recipes = scene_data()
    config = Config(scenes={"unlisted": 1}).generation
    for kind in ["lathe", "cone", "capsule"]:
        recipe = next(iter(recipes.values())).model_copy(deep=True)
        recipe.parts[0].primitive = kind
        nodes, connections, _, _ = compile_creative(
            brief, designs, {object_key(designs[0].objects[0]): recipe}, [], "x", 17, config
        )
        arrays = {n.id: generate(n) for n in nodes if n.generator != "group"}
        report = validate(nodes, connections, lambda n: arrays[n.id])
        assert report.valid, report.model_dump()


def test_recipe_identity_context_and_legacy_checkpoint_reuse(tmp_path):
    brief, designs, recipes = scene_data()
    old = designs[0].objects[0]
    altered = old.model_copy(update={"material": "glass"})
    assert object_key(old) != object_key(altered)
    calls = []

    class Client:
        def request(self, schema, context, *args, **kwargs):
            calls.append(context)
            return next(iter(recipes.values()))

    for seed in [13, 14]:
        path = tmp_path / str(seed)
        path.mkdir()
        design_recipes(designs, Client(), "s", path, Log(path / "events.jsonl"), [], seed=seed, brief=brief)
    assert len(calls) == 2 and calls[0]["seed"] != calls[1]["seed"]
    assert calls[0]["object"]["material"] == "metal" and "count" not in calls[0]["object"]
    old_key = digest({"kind": old.kind, "detail": old.detail, "size": [old.width, old.depth, old.height]})[:20]
    path = tmp_path / "legacy"
    path.mkdir()
    write_json(path / "object-recipes" / f"{old_key}.json", next(iter(recipes.values())).model_dump(mode="json"))
    design_recipes(designs, Client(), "s", path, Log(path / "events.jsonl"), [], seed=15)
    assert len(calls) == 2


def test_input_preferences_and_nonhuman_population_are_data(tmp_path):
    brief, designs, _ = scene_data()
    brief.population = 2
    brief.population_kind = "broad mechanical glider"
    for design in designs:
        obj = design.objects[0]
        obj.population = True
        obj.count = 1
        obj.kind = brief.population_kind
        obj.width = 4
        obj.depth = 3
        obj.height = 0.4
    captured = []

    class Client:
        def request(self, schema, context, *args, **kwargs):
            captured.append(context)
            result = brief if schema is SceneBrief else designs[len(captured) - 2]
            return schema.model_validate(result.model_dump(), context=kwargs.get("validation_context"))

    config = Config.model_validate(
        {
            "scenes": {"unlisted": 1},
            "generation": {"variations": {"architectural_styles": ["woven"], "environments": ["subterranean"]}},
        }
    )
    out, detailed = design_scene(
        "unlisted",
        0,
        13,
        config.generation,
        Client(),
        "s",
        tmp_path,
        Log(tmp_path / "events.jsonl"),
        description="A specified habitat",
    )
    assert captured[0]["description"] == "A specified habitat" and captured[0]["style_preference"] == "woven"
    assert captured[0]["environment_preference"] == "subterranean"
    assert all(len(d.objects) == 1 and d.objects[0].width == 4 for d in detailed)
    assert sum(o.count for d in detailed for o in d.objects) == 2
    assert not any(o.kind in {"person", "human"} for d in detailed for o in d.objects)


def test_default_seed_variation_explicit_seed_replay_and_resume(tmp_path):
    cfg = Config.model_validate(
        {
            "scenes": {"invented installation": 1},
            "generation": {
                "quality": "draft",
                "frameworks": ["trimesh"],
                "llm": {"mode": "mock"},
                "output": {"root": str(tmp_path)},
            },
        }
    )
    seeds = []
    for _ in range(2):
        pipeline = Pipeline.create(cfg)
        seeds.append(pipeline.generation.seed)
        pipeline.close()
    assert seeds[0] != seeds[1] and cfg.generation.seed is None
    cfg.generation.seed = 99
    hashes = []
    for _ in range(2):
        pipeline = Pipeline.create(cfg)
        try:
            assert pipeline.run()["status"] == "complete"
            scene = next((pipeline.path / "scenes").iterdir())
            hashes.append(file_hash(scene / "scene.glb"))
            before = pipeline.state.one("SELECT count(*) n FROM llm_calls")["n"]
            assert pipeline.run()["status"] == "complete"
            assert pipeline.state.one("SELECT count(*) n FROM llm_calls")["n"] == before
            assert read_json(scene / "report.json")["regenerated_components_this_pass"] == 0
            assert export_project(scene, "ply")["format"] == "ply"
        finally:
            pipeline.close()
    assert hashes[0] == hashes[1]


def test_defaults_and_material_colors_survive_reload():
    from scene_generator.generators import as_mesh

    brief, designs, recipes = scene_data()
    obj = designs[0].objects[0]
    assert object_key(obj) == object_key(type(obj).model_validate_json(obj.model_dump_json()))
    nodes, _, _, _ = compile_creative(brief, designs, recipes, [], "x", 1, Config(scenes={"x": 1}).generation)
    node = next(n for n in nodes if n.generator == "box")
    node.materials[0] = node.materials[0].model_copy(update={"color": (1, 1, 1, 1)})
    mesh = as_mesh(node, generate(node))
    assert list(mesh.visual.material.baseColorFactor) == [255, 255, 255, 255]


def test_zero_count_objects_make_no_asset_or_recipe_requests(tmp_path):
    from scene_generator.assets import design_asset_requests

    brief, designs, _ = scene_data()
    for design in designs:
        design.objects[0].count = 0
        design.objects[0].asset_query = "unused search"
        design.objects[0].material = "brushed blue alloy"

    class NoCalls:
        def request(self, *args, **kwargs):
            raise AssertionError("zero-count object requested a recipe")

    assert design_recipes(designs, NoCalls(), "s", tmp_path, Log(tmp_path / "events.jsonl"), []) == {}
    assert not any(r["type"] == "model" for r in design_asset_requests(brief, designs))


def test_scene_constraints_are_validated_before_expensive_work():
    from pydantic import ValidationError

    brief, designs, _ = scene_data()
    with pytest.raises(ValidationError, match="dimensions"):
        SceneBrief.model_validate(brief.model_dump(), context={"dimensions": (15, 15, 4)})
    with pytest.raises(ValidationError, match="population"):
        ZoneDesign.model_validate(designs[0].model_dump(), context={"population": 3, "floors": 1})
    designs[0].objects[0].floor = 2
    with pytest.raises(ValidationError, match="floor"):
        ZoneDesign.model_validate(designs[0].model_dump(), context={"population": 0, "floors": 1})


def test_legacy_zone_count_semantics_survive_resume(tmp_path):
    brief, designs, _ = scene_data()
    brief.zones[1].floors = 3
    write_json(tmp_path / "brief.json", brief.model_dump(mode="json"))
    for i, design in enumerate(designs, 1):
        saved = design.model_dump(mode="json")
        for obj in saved["objects"]:
            obj.pop("repeat_on_floors")
        write_json(tmp_path / "zone-designs" / f"zone-{i:02d}.json", saved)
    _, restored = design_scene(
        "x", 0, 1, Config(scenes={"x": 1}).generation, None, "s", tmp_path, Log(tmp_path / "events.jsonl")
    )
    assert sum(len(object_placements(brief.zones[1], restored[1], f, 1)) for f in range(3)) == 9


def test_arbitrary_category_names_do_not_collide_after_slugging(tmp_path):
    cfg = Config.model_validate(
        {"scenes": {"a_b": 1, "a_b-2": 1, "a b": 1}, "generation": {"output": {"root": str(tmp_path)}}}
    )
    pipeline = Pipeline.create(cfg)
    try:
        rows = pipeline.state.rows("SELECT id FROM scenes")
        assert len({r["id"] for r in rows}) == 3
    finally:
        pipeline.close()
