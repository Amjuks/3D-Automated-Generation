"""Category-independent spatial compiler. Semantic objects arrive as assets or recipes."""

import math

from .builder import Builder
from .layout import circulation_segments, object_placements, zone_layout
from .models import Material, Socket
from .recipes import object_key
from .spatial import box
from .util import stable_seed


def compile_creative(brief, designs, recipes, assets, scene_id, seed, generation):
    tokens = {}

    def material(name):
        if name not in tokens:
            # Stable fallback colors for material names without explicit recipe colors.
            hue = stable_seed(seed, name)
            color = tuple(0.18 + ((hue >> (i * 8)) & 255) / 255 * 0.42 for i in range(3))
            colors = {
                "grass": (0.18, 0.28, 0.09),
                "glass": (0.6, 0.8, 0.86),
                "wood": (0.33, 0.19, 0.09),
                "stone": (0.53, 0.52, 0.47),
                "metal": (0.28, 0.3, 0.32),
                "plaster": (0.77, 0.74, 0.68),
            }
            tokens[name] = Material(
                name=name,
                color=(*colors.get(name, color), 0.2 if name == "glass" else 1),
                transmission=0.9 if name == "glass" else 0,
                roughness=0.12 if name == "glass" else 0.65,
                metallic=0.7 if name == "metal" else 0,
            )
        return name

    for name in ["ground", "path", "glass", "metal", "wall", "floor", brief.terrain]:
        material(name)
    for z in brief.zones:
        material(z.ground)
        material(z.wall)
    b = Builder(scene_id, seed, generation.quality, tokens)
    origins, extent = zone_layout(brief)
    width, depth, height = extent
    limits = generation.dimensions
    if generation.bounding_space:
        limits = (
            tuple(min(a, c) for a, c in zip(limits, generation.bounding_space.size))
            if limits
            else generation.bounding_space.size
        )
    if limits and any(a > c for a, c in zip(extent, limits)):
        raise ValueError(
            f"Designed site requires {extent} meters, exceeding configured dimensions {limits}; reduce brief zone dimensions before resuming"
        )
    root = b.add(
        None, "scene", "scene", generation.bounding_space.min if generation.bounding_space else (0, 0, 0), extent
    )
    root.parameters["camera"] = {
        "position": [-width * 0.12, -depth * 0.2, max(width, depth) * 0.58],
        "target": [width * 0.5, depth * 0.5, height * 0.2],
    }
    root.parameters["lighting"] = brief.lighting
    root.parameters["weather"] = brief.weather
    root.parameters["workflow"] = "creative"
    ground = b.solid(
        root,
        "terrain",
        (0, 0, 0),
        (width, depth, 0.2),
        material(brief.terrain),
        parameters={"material_role": "environment/terrain"},
    )
    if brief.circulation == "paths":
        for i, (origin, size) in enumerate(circulation_segments(brief.zones, origins)):
            b.solid(root, f"path-{i}", origin, size, material("path"), collidable=False)
    acquired = {r.get("target_role"): r for r in assets if r["type"] == "model"}
    coverage = []
    for zi, (zone, design) in enumerate(zip(brief.zones, designs)):
        origin = origins[zi]
        building = b.add(
            root,
            f"zone-{zi + 1:02d}",
            "building" if zone.enclosure == "building" else "zone",
            origin,
            (zone.width, zone.depth, zone.floor_height * zone.floors),
            support_id=ground.id,
        )
        building.parameters["description"] = design.description
        w, d, h = zone.width, zone.depth, zone.floor_height
        hall = None
        if zone.enclosure == "building":
            hall = b.add(
                building, "circulation", "room", (0, 0, 0), (w, 3, h * zone.floors), parameters={"circulation": True}
            )
            hall.sockets = [Socket(id="entrance", position=(w / 2, 0, 0.2), kind="door")]
            for f in range(zone.floors):
                b.solid(hall, f"landing-{f}", (0, 1.65, f * h), (w, 1.35, 0.18), material(zone.ground))
                if f < zone.floors - 1:
                    steps = math.ceil(h / 0.18)
                    run = (w - 1) / steps
                    for t in range(steps):
                        b.solid(
                            hall,
                            f"stair-{f}-{t}",
                            (0.5 + t * run, 0.2, f * h),
                            (run, 1.2, (t + 1) * h / steps),
                            material(zone.ground),
                        )
        for f in range(zone.floors):
            room_y = 3 if hall else 0
            floor = b.add(building, f"floor-{f}", "room" if hall else "zone", (0, room_y, f * h), (w, d - room_y, h))
            floor.parameters["camera"] = {"position": [w / 2, 0.7, 1.65], "target": [w / 2, d - room_y - 1, 1.4]}
            slab = b.solid(
                floor,
                "floor-surface",
                (0, 0, 0),
                (w, d - room_y, 0.18),
                material(zone.ground),
                parameters={"material_role": f"zone-{zi + 1:02d}/floor"},
            )
            if hall:
                gap = 2
                jamb = (w - gap) / 2
                for x in (0, jamb + gap):
                    b.solid(
                        floor,
                        f"door-jamb-{x}",
                        (x, 0, 0.18),
                        (jamb, 0.18, h - 0.3),
                        material(zone.wall),
                        parameters={"material_role": f"zone-{zi + 1:02d}/wall"},
                    )
                b.solid(floor, "lintel", (jamb, 0, 2.8), (gap, 0.18, h - 2.92), material(zone.wall))
                for x in (0, w - 0.18):
                    b.solid(
                        floor,
                        f"side-{x}",
                        (x, 0.18, 0.18),
                        (0.18, d - 3 - 0.36, h - 0.3),
                        material(zone.wall),
                        parameters={"material_role": f"zone-{zi + 1:02d}/wall"},
                    )
                # The brief chooses an opaque wall or an actual window opening.
                rd = d - 3
                bands = ((0.18, 0.8), (h - 0.8, 0.68)) if zone.windows else ((0.18, h - 0.3),)
                for z, hh in bands:
                    b.solid(
                        floor,
                        f"rear-{z}",
                        (0, rd - 0.18, z),
                        (w, 0.18, hh),
                        material(zone.wall),
                        parameters={"material_role": f"zone-{zi + 1:02d}/wall"},
                    )
                if zone.windows:
                    b.solid(floor, "glazing", (0.18, rd - 0.16, 0.98), (w - 0.36, 0.08, h - 1.78), material("glass"))
                if zone.roof or f < zone.floors - 1:
                    b.solid(floor, "ceiling", (0, 0, h - 0.12), (w, rd, 0.12), material(zone.wall))
                floor.reserved = [box((w / 2 - 0.9, 0.18, 0.18), (1.8, rd - 0.36, 2.4))]
                point = tuple(a + c for a, c in zip(floor.world_transform.translation, (w / 2, 0, 0.18)))
                b.connections.append({"from": floor.id, "to": hall.id, "width": 2, "position": point})
            for oi, spec, k, size, factor, x, y in object_placements(zone, design, f, stable_seed(seed, zi)):
                role = f"zone-{zi + 1:02d}/object-{oi + 1:02d}"
                obj = b.add(
                    floor,
                    f"object-{oi + 1:02d}-{k}",
                    "object",
                    (x, y, 0.18),
                    size,
                    support_id=slab.id,
                    parameters={
                        "asset_role": role,
                        "semantic_kind": spec.kind,
                        "requested_size": [spec.width, spec.depth, spec.height],
                        "scale_to_fit": factor,
                    },
                )
                if role in acquired:
                    record = acquired[role]
                    matname = material(spec.material)
                    node = b.add(
                        obj,
                        "model",
                        "object",
                        (0, 0, 0),
                        size,
                        "asset",
                        matname,
                        parameters={
                            "path": record["path"],
                            "sha256": record["sha256"],
                            "asset_id": record["id"],
                            "up_axis": record.get("up_axis", "Y" if record["source"].startswith("https:") else "Z"),
                        },
                    )
                    node.budget.triangles = max(2000, record.get("triangles", 2000))
                    if record.get("base_color_texture"):
                        node.materials = [
                            Material(
                                name="asset-" + record["id"],
                                color=(1, 1, 1, 1),
                                base_color_texture=record["base_color_texture"],
                            )
                        ]
                    coverage.append(
                        {"role": role, "object": obj.id, "source": "asset", "asset": record["id"], "scale": factor}
                    )
                elif recipes is not None:
                    recipe = recipes[object_key(spec)]
                    for pi, part in enumerate(recipe.parts):
                        matname = f"recipe-{object_key(spec)}-{pi}"
                        tokens[matname] = Material(
                            name=matname,
                            color=(*part.color, part.opacity),
                            transmission=part.transmission,
                            roughness=part.roughness,
                            metallic=part.metallic,
                            emission=part.emission,
                        )
                        partnode = b.add(
                            obj,
                            f"part-{pi:02d}",
                            "detail",
                            tuple(a * c for a, c in zip(part.position, size)),
                            tuple(a * c for a, c in zip(part.size, size)),
                            part.primitive,
                            matname,
                            parameters={
                                "assembly_id": obj.id,
                                "rotation_degrees": part.rotation,
                                "surface": spec.material,
                                "profile": part.profile,
                            },
                        )
                        partnode.budget.triangles = 50000
                    coverage.append(
                        {
                            "role": role,
                            "object": obj.id,
                            "source": "recipe",
                            "recipe": object_key(spec),
                            "scale": factor,
                        }
                    )
    # Population uses the same asset/recipe system, not a category-specific character generator.
    return b.nodes, b.connections, tokens, coverage
