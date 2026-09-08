"""Small, checkpointed design tasks; Markdown is a view of validated design data."""

from typing import Literal

from pydantic import Field, model_validator

from .models import Model, ScenePlan
from .util import atomic_write, read_json, stable_seed, write_json


class ZoneBrief(Model):
    name: str = Field(max_length=80)
    purpose: str = Field(max_length=300)
    enclosure: Literal["open", "building"] = "open"
    floors: int = Field(default=1, ge=1, le=3)
    x: float = Field(default=0, ge=0, le=160)
    y: float = Field(default=0, ge=0, le=160)
    width: float = Field(default=12, ge=8, le=24)
    depth: float = Field(default=12, ge=8, le=24)
    floor_height: float = Field(default=4, ge=3.5, le=40)
    ground: str = Field(default="stone", max_length=40, pattern=r"^[a-z][a-z0-9_-]*$")
    wall: str = Field(default="plaster", max_length=40, pattern=r"^[a-z][a-z0-9_-]*$")
    palette: list[str] = Field(min_length=1, max_length=6)


class SceneBrief(Model):
    title: str = Field(max_length=100)
    concept: str = Field(max_length=650)
    family: str = Field(max_length=80)
    composition: Literal["freeform", "campus", "linear", "staggered"]
    environment: str = Field(max_length=160)
    lighting: Literal["daylight", "sunset", "overcast", "night"]
    weather: str = Field(max_length=160)
    terrain: str = Field(max_length=60)
    population: int = Field(ge=0, le=24)
    population_story: str = Field(max_length=240)
    population_kind: str = Field(default="inhabitant", max_length=70)
    population_query: str = Field(default="", max_length=100)
    zones: list[ZoneBrief] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def feasible_zones(self):
        if any(z.enclosure == "building" and z.floor_height > 6 for z in self.zones):
            raise ValueError("Building floor heights must be <=6m; open zones can extend to 40m")
        if self.composition == "freeform":
            for i, z in enumerate(self.zones):
                for other in self.zones[:i]:
                    if min(z.x + z.width + 1.5, other.x + other.width + 1.5) > max(z.x - 1.5, other.x - 1.5) and min(
                        z.y + z.depth + 1.5, other.y + other.depth + 1.5
                    ) > max(z.y - 1.5, other.y - 1.5):
                        raise ValueError(
                            "Freeform zones must not overlap and must leave 3m gaps; adjust their x/y positions"
                        )
        return self


class ObjectSpec(Model):
    name: str = Field(max_length=70)
    kind: str = Field(max_length=70)
    count: int = Field(ge=1, le=24)
    material: str = Field(max_length=40, pattern=r"^[a-z][a-z0-9_-]*$")
    asset_query: str = Field(default="", max_length=100)
    detail: str = Field(max_length=220)
    width: float = Field(default=1.5, ge=0.2, le=5)
    depth: float = Field(default=1.5, ge=0.2, le=5)
    height: float = Field(default=2, ge=0.2, le=40)


class ZoneDesign(Model):
    arrangement: Literal["scattered", "clustered", "perimeter", "rows"] = "scattered"
    description: str = Field(max_length=500)
    floor_description: str = Field(max_length=300)
    wall_description: str = Field(max_length=300)
    floor_texture: str = Field(max_length=100)
    wall_texture: str = Field(max_length=100)
    objects: list[ObjectSpec] = Field(min_length=1, max_length=8)


def default_brief(category, ordinal, seed):
    # Offline fixtures exercise the compiler, not semantic intelligence.
    zones = [
        dict(
            name=f"{category} area {i + 1}",
            purpose=f"A distinct part of {category}.",
            enclosure="open" if i == 0 else "building",
            floors=1,
            width=10 + stable_seed(seed, i) % 5,
            depth=12,
            floor_height=4.2,
            ground=["stone", "wood"][i],
            wall="plaster",
            palette=[f"{category} focal object", "seating"],
        )
        for i in range(2)
    ]
    return dict(
        title=f"{category.title()} — {ordinal + 1}",
        concept=f"A varied interpretation of {category}.",
        family=category,
        composition=["campus", "linear", "staggered"][ordinal % 3],
        environment="Site-specific surroundings",
        lighting="daylight",
        weather="Clear air",
        terrain="grass",
        population=2,
        population_story="Visitors exploring the scene.",
        zones=zones,
    )


def markdown(brief, designs):
    lines = [
        f"# {brief.title}",
        "",
        brief.concept,
        "",
        f"- Scene family: {brief.family}",
        f"- Composition: {brief.composition}",
        f"- Environment: {brief.environment}",
        f"- Terrain: {brief.terrain}",
        f"- Light: {brief.lighting}; weather: {brief.weather}",
        f"- People: {brief.population}. {brief.population_story}",
        "",
        "## Structure and circulation",
        "Zones connect to a continuous 3 m promenade. Buildings have a front doorway, windows, an interior aisle and a stair bay when more than one floor is requested. Object placement preserves the aisle. Dimensions below are in meters.",
        "",
    ]
    for i, z in enumerate(brief.zones):
        lines += [
            f"## Zone {i + 1}: {z.name}",
            z.purpose,
            f"{z.enclosure}; {z.width} × {z.depth} m; {z.floors} floor(s), {z.floor_height} m per floor.",
            f"Floor/ground: {z.ground}. Wall: {z.wall}. Object palette: {', '.join(z.palette)}.",
        ]
        if i < len(designs):
            d = designs[i]
            lines += [
                d.description,
                f"Floors: {d.floor_description}",
                f"Walls: {d.wall_description}",
                f"Texture searches: {d.floor_texture}; {d.wall_texture}.",
            ]
            lines += [
                f"- {o.count} × {o.name} ({o.kind}, {o.material}): {o.detail}. Asset search: {o.asset_query or 'procedural'}."
                for o in d.objects
            ]
        else:
            lines += ["Detailed zone design pending."]
        lines += [""]
    lines += [
        "## Build contract",
        "This document is generated from brief.json and zone-designs/*.json. It is a design specification, not executable code. asset-coverage.json records acquired assets and procedural fallbacks; quality.json records final fidelity limits.",
    ]
    return "\n".join(lines) + "\n"


def design_scene(category, ordinal, seed, generation, llm, scene_id, path, log, previous_concepts=None):
    target = path / "brief.json"
    if target.exists():
        brief = SceneBrief.model_validate(read_json(target))
    else:
        # No global neoclassical museum style is imposed on unrelated categories.
        brief = llm.request(
            SceneBrief,
            {
                "task": "Design the overall scene, not individual objects yet.",
                "category": category,
                "category_is_authoritative": True,
                "variation": ordinal,
                "avoid_repeating": previous_concepts or [],
                "seed": seed,
                "composition_direction": "Invent a freeform composition with varied zone x/y positions and unequal dimensions; leave 3m gaps between zones. Avoid repeated grids.",
                "instruction": "Expand the supplied category imaginatively, without imposing a building or museum template. It can describe any real or fictional place, object collection or environment. Specify 1-6 coherent zones, their physical dimensions, building floors, weather, lighting and population. Open zones have one floor; their floor_height is the available vertical extent and may reach 40m for tall terrain, vegetation or objects. Building floor heights must stay <=6m. For freeform composition give nonoverlapping x/y positions in meters with at least 3m gaps; vary orientation of the overall arrangement through positions and dimensions. Invent arbitrary category-appropriate object names in the palette; the next task designs each zone separately. Favor contrasting silhouettes and purposeful negative space.",
            },
            scene_id,
            "brief",
            mock=lambda: default_brief(category, ordinal, seed),
        )
        for z in brief.zones:
            if z.enclosure == "open":
                z.floors = 1
        write_json(target, brief.model_dump(mode="json"))
    designs = []
    atomic_write(path / "scene.md", markdown(brief, designs).encode())
    for i, z in enumerate(brief.zones):
        checkpoint = path / "zone-designs" / f"zone-{i + 1:02d}.json"
        log.event("design_zone", scene=scene_id, index=i + 1, total=len(brief.zones))
        if checkpoint.exists():
            design = ZoneDesign.model_validate(read_json(checkpoint))
        else:

            def mock():
                return dict(
                    description=z.purpose,
                    floor_description=f"{z.ground} with worn edges and material variation.",
                    wall_description=f"{z.wall} with openings and visible construction joints."
                    if z.enclosure == "building"
                    else "Open views and planted edges.",
                    floor_texture=z.ground + " ground floor",
                    wall_texture=z.wall + " wall",
                    objects=[
                        dict(
                            name=k.title(),
                            kind=k,
                            count=2,
                            material={
                                "tree": "leaf",
                                "plant": "leaf",
                                "rock": "stone",
                                "tank": "glass",
                                "fish": "accent",
                                "voxel": "grass",
                            }.get(k, "wood" if k in {"bench", "table", "shelf", "crate"} else "metal"),
                            asset_query={
                                "tree": "pine tree",
                                "person": "standing person",
                                "sculpture": "statue bust",
                            }.get(k, k),
                            detail=f"{k.title()} with varied proportions, visible detail and natural wear.",
                        )
                        for k in z.palette
                    ],
                )

            design = llm.request(
                ZoneDesign,
                {
                    "task": "Detail only this zone. Do not redesign the whole site.",
                    "concept": brief.concept,
                    "family": brief.family,
                    "zone": z.model_dump(mode="json"),
                    "lighting": brief.lighting,
                    "neighbors": [n.name for n in brief.zones[max(0, i - 1) : i + 2] if n is not z],
                    "instruction": "Specify objects, material finishes and short asset-search phrases. Object kinds are free-form names: later tasks find assets or design a geometry recipe for each. No executable code. Keep objects relevant to this zone palette. Give precise visual details and varied quantities.",
                },
                scene_id,
                f"zone-{i + 1:02d}",
                mock=mock,
            )
            people = brief.population // len(brief.zones) + (i < brief.population % len(brief.zones))
            if people and not any(o.kind.lower() in {"person", "people", "human"} for o in design.objects):
                design.objects.append(
                    ObjectSpec(
                        name="Inhabitants of the area",
                        kind=brief.population_kind,
                        count=people,
                        material="fabric",
                        asset_query=brief.population_query,
                        detail=brief.population_story,
                        width=0.65,
                        depth=0.65,
                        height=1.75,
                    )
                )
            write_json(checkpoint, design.model_dump(mode="json"))
        designs.append(design)
        atomic_write(path / "scene.md", markdown(brief, designs).encode())
    return brief, designs


def compatibility_plan(brief, category):
    return ScenePlan(
        title=brief.title,
        category=category,
        style=brief.family[:60],
        environment=brief.environment[:60],
        collection="art",
        atmosphere={"night": "dramatic", "sunset": "warm", "overcast": "cool"}.get(brief.lighting, "daylight"),
        columns=len(brief.zones),
        rows=1,
        room_width=12,
        room_depth=12,
        height=6,
        accent=(0.2, 0.45, 0.5),
        concept=brief.concept[:450],
    )
