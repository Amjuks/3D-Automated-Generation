"""Small, checkpointed design tasks; Markdown is a view of validated design data."""

from typing import Literal

from pydantic import Field, ValidationInfo, model_validator

from .mock_design import default_brief as default_brief
from .models import Model
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
    ground: str = Field(default="ground", min_length=1, max_length=80)
    wall: str = Field(default="wall", min_length=1, max_length=80)
    palette: list[str] = Field(default_factory=list, max_length=6)
    windows: bool = True
    roof: bool = True


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
    circulation: Literal["none", "paths"] = "paths"
    zones: list[ZoneBrief] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def feasible_zones(self, info: ValidationInfo):
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
        if any(z.enclosure == "open" and z.floors != 1 for z in self.zones):
            raise ValueError("open zones must have one floor")
        limits = (info.context or {}).get("dimensions")
        if limits:
            from .layout import zone_layout

            _, extent = zone_layout(self)
            if any(a > b for a, b in zip(extent, limits)):
                raise ValueError("zone layout and margins exceed the supplied site dimensions")
        return self


class ObjectSpec(Model):
    name: str = Field(max_length=70)
    kind: str = Field(max_length=70)
    count: int = Field(ge=0, le=24)
    material: str = Field(min_length=1, max_length=80)
    asset_query: str = Field(default="", max_length=100)
    detail: str = Field(max_length=220)
    width: float = Field(default=1.5, ge=0.2, le=5)
    depth: float = Field(default=1.5, ge=0.2, le=5)
    height: float = Field(default=2, ge=0.2, le=40)
    # None distributes the requested count across floors. Old saved zone designs
    # are upgraded to repeat explicitly so a resume does not change their meaning.
    floor: int | None = Field(default=None, ge=0, le=2)
    repeat_on_floors: bool = False
    population: bool = False


class ZoneDesign(Model):
    arrangement: Literal["scattered", "clustered", "perimeter", "rows"] = "scattered"
    description: str = Field(max_length=500)
    floor_description: str = Field(max_length=300)
    wall_description: str = Field(max_length=300)
    floor_texture: str = Field(max_length=100)
    wall_texture: str = Field(max_length=100)
    objects: list[ObjectSpec] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def allocations(self, info: ValidationInfo):
        context = info.context or {}
        floors = context.get("floors")
        if floors is not None and any(o.floor is not None and o.floor >= floors for o in self.objects):
            raise ValueError("object floor must exist in its zone")
        population = context.get("population")
        actual = sum(o.count * ((floors or 1) if o.repeat_on_floors else 1) for o in self.objects if o.population)
        if population is not None and actual != population:
            raise ValueError("population objects must total the assigned population count across all floors")
        return self


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
        f"- Population: {brief.population}. {brief.population_story}",
        "",
        "## Structure and circulation",
        f"External circulation: {brief.circulation}. Building allocations reserve an entrance, interior aisle and floor connections. Openings and roofs follow each zone brief. Dimensions below are in meters.",
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


def design_scene(category, ordinal, seed, generation, llm, scene_id, path, log, previous_concepts=None, description=""):
    target = path / "brief.json"
    if target.exists():
        brief = SceneBrief.model_validate(read_json(target))
    else:
        styles = generation.variations.architectural_styles
        environments = generation.variations.environments
        limits = generation.dimensions
        if generation.bounding_space:
            limits = (
                tuple(min(a, b) for a, b in zip(limits, generation.bounding_space.size))
                if limits
                else generation.bounding_space.size
            )
        brief = llm.request(
            SceneBrief,
            {
                "task": "Design the overall scene, not individual objects yet.",
                "category": category,
                "description": description,
                "category_is_authoritative": True,
                "variation": ordinal,
                "avoid_repeating": previous_concepts or [],
                "seed": seed,
                "style_preference": styles[ordinal % len(styles)] if styles else None,
                "environment_preference": environments[(ordinal // max(1, len(styles))) % len(environments)]
                if environments
                else None,
                "dimensions": generation.dimensions,
                "bounding_space": generation.bounding_space.model_dump() if generation.bounding_space else None,
                "composition_direction": "Choose composition for the requested purpose. Vary positions and dimensions when appropriate, and preserve intentional repetition requested by the input.",
                "instruction": "Expand the supplied category and description imaginatively. Honor supplied preferences and site dimensions. Specify 1-6 coherent zones with dimensions, optional building floors, weather, lighting and population. Choose zero population when appropriate; inhabitants need not be human. Choose circulation none if paths are inappropriate. Open zones have one floor; floor_height is their vertical extent, up to 40m. Building floor heights must stay <=6m. Choose windows and roof for each building. Freeform zones need nonoverlapping x/y positions and 3m gaps. The site adds 7m margins to the maximum zone extents; fit within supplied dimensions including margins and 1m vertical clearance. Invent appropriate object names in the palette. Favor contrasting silhouettes and purposeful negative space.",
            },
            scene_id,
            "brief",
            mock=lambda: default_brief(category, ordinal, seed),
            validation_context={"dimensions": limits},
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
            saved = read_json(checkpoint)
            for obj in saved.get("objects", []):
                obj.setdefault("repeat_on_floors", True)
            design = ZoneDesign.model_validate(saved)
        else:
            population = brief.population // len(brief.zones) + (i < brief.population % len(brief.zones))

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
                            count=1 + stable_seed(seed, i, k) % 3,
                            material=z.ground,
                            asset_query=k,
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
                    "seed": stable_seed(seed, "zone", i),
                    "zone": z.model_dump(mode="json"),
                    "lighting": brief.lighting,
                    "population": {
                        "count": population,
                        "kind": brief.population_kind,
                        "story": brief.population_story,
                        "asset_query": brief.population_query,
                    },
                    "neighbors": [n.name for n in brief.zones[max(0, i - 1) : i + 2] if n is not z],
                    "instruction": "Specify objects, material finishes and short asset-search phrases. Object kinds are free-form names; later tasks find assets or design geometry recipes. Give precise visual details, dimensions and varied quantities appropriate to the palette. Empty zones are allowed. Object count is the total across floors; use floor to target one floor, or repeat_on_floors only for intentional repetition. Include the assigned population as ordinary objects with population=true and appropriate dimensions, material, appearance and kind. Respect its count, including zero; do not assume human bodies. No executable code.",
                },
                scene_id,
                f"zone-{i + 1:02d}",
                mock=mock,
                validation_context={"population": population, "floors": z.floors},
            )
            write_json(checkpoint, design.model_dump(mode="json"))
        designs.append(design)
        atomic_write(path / "scene.md", markdown(brief, designs).encode())
    return brief, designs


class SceneSummary(Model):
    workflow: Literal["creative"] = "creative"
    title: str
    category: str
    concept: str
    environment: str
    atmosphere: Literal["warm", "cool", "dramatic", "daylight"]
    layout: str


def compatibility_plan(brief, category):
    """Export/report metadata; content remains in the complete scene brief."""
    return SceneSummary(
        title=brief.title,
        category=category,
        concept=brief.concept,
        environment=brief.environment,
        layout=brief.composition,
        atmosphere={"night": "dramatic", "sunset": "warm", "overcast": "cool"}.get(brief.lighting, "daylight"),
    )
