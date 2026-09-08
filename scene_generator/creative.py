"""Category-independent spatial compiler. Semantic objects arrive as assets or recipes."""

import math
import random

from .decomposition import Builder
from .models import Material, Socket
from .recipes import object_key
from .spatial import box
from .util import stable_seed


def compile_creative(brief, designs, recipes, assets, scene_id, seed, generation):
    tokens = {}

    def material(name):
        if name not in tokens:
            # A per-scene palette, not the previous global museum palette.
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
    cols = len(brief.zones) if brief.composition == "linear" else 2
    cellw = max(z.width for z in brief.zones) + 5
    celld = max(z.depth for z in brief.zones) + 5
    rows = math.ceil(len(brief.zones) / cols)
    width, depth = cols * cellw + 4, rows * celld + 4
    if brief.composition == "freeform":
        width = max(z.x + z.width for z in brief.zones) + 7
        depth = max(z.y + z.depth for z in brief.zones) + 7
        for i, z in enumerate(brief.zones):
            for other in brief.zones[:i]:
                if min(z.x + z.width + 1.5, other.x + other.width + 1.5) > max(z.x - 1.5, other.x - 1.5) and min(
                    z.y + z.depth + 1.5, other.y + other.depth + 1.5
                ) > max(z.y - 1.5, other.y - 1.5):
                    raise ValueError(
                        "Freeform zones overlap or leave less than 3m circulation; correct brief.json and resume"
                    )
    height = max(z.floors * z.floor_height for z in brief.zones) + 1
    extent = (width, depth, height)
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
    # Continuous paths between independently allocated zones; intersections are decorative surfacing.
    for row in range(rows):
        b.solid(
            root, f"promenade-{row}", (0, row * celld + 0.3, 0.2), (width, 3, 0.06), material("path"), collidable=False
        )
    b.solid(root, "connecting-path", (0.3, 0, 0.2), (2.5, depth, 0.06), material("path"), collidable=False)
    acquired = {r.get("target_role"): r for r in assets if r["type"] == "model"}
    coverage = []
    for zi, (zone, design) in enumerate(zip(brief.zones, designs)):
        row, col = divmod(zi, cols)
        shift = 1.2 * (zi % 2) if brief.composition == "staggered" else 0
        origin = (3 + col * cellw + shift, 3.3 + row * celld, 0.2)
        if brief.composition == "freeform":
            origin = (3 + zone.x, 3.3 + zone.y, 0.2)
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
                # Rear window band, actual glass, sill and top; no opaque wall behind glass.
                rd = d - 3
                for z, hh in ((0.18, 0.8), (h - 0.8, 0.68)):
                    b.solid(
                        floor,
                        f"rear-{z}",
                        (0, rd - 0.18, z),
                        (w, 0.18, hh),
                        material(zone.wall),
                        parameters={"material_role": f"zone-{zi + 1:02d}/wall"},
                    )
                b.solid(floor, "glazing", (0.18, rd - 0.16, 0.98), (w - 0.36, 0.08, h - 1.78), material("glass"))
                b.solid(floor, "ceiling", (0, 0, h - 0.12), (w, rd, 0.12), material(zone.wall))
                floor.reserved = [box((w / 2 - 0.9, 0.18, 0.18), (1.8, rd - 0.36, 2.4))]
                point = tuple(a + c for a, c in zip(floor.world_transform.translation, (w / 2, 0, 0.18)))
                b.connections.append({"from": floor.id, "to": hall.id, "width": 2, "position": point})
            occupied = []
            rng = random.Random(stable_seed(seed, zi, f, "placement"))
            # Two side strips preserve an unobstructed center. Each object gets its own allocation.
            available_depth = d - room_y - 1.5
            requests = [(oi, spec, k) for oi, spec in enumerate(design.objects) for k in range(spec.count)]
            per_side = math.ceil(len(requests) / 2)
            slotd = available_depth / max(1, per_side)
            slotw = (w - 2.8) / 2
            for index, (oi, spec, k) in enumerate(requests):
                side = index % 2
                r = index // 2
                factor = min(1, slotw / spec.width, (slotd - 0.12) / spec.depth, (h - 0.6) / spec.height)
                if factor <= 0:
                    raise ValueError("Object density exceeds zone allocation")
                size = tuple(v * factor for v in (spec.width, spec.depth, spec.height))
                x = 0.45 if side == 0 else w - 0.45 - size[0]
                y = 0.7 + r * slotd
                if not hall and design.arrangement != "rows":
                    for _ in range(80):
                        px = rng.uniform(0.3, w - size[0] - 0.3)
                        py = rng.uniform(0.3, d - size[1] - 0.3)
                        if design.arrangement == "perimeter":
                            px = 0.3 if side == 0 else w - size[0] - 0.3
                        if all(
                            px + size[0] + 0.15 <= ox
                            or ox + ow + 0.15 <= px
                            or py + size[1] + 0.15 <= oy
                            or oy + od + 0.15 <= py
                            for ox, oy, ow, od in occupied
                        ):
                            x, y = px, py
                            break
                    else:
                        raise ValueError("Objects do not fit without overlap; reduce zone object counts or dimensions")
                occupied.append((x, y, size[0], size[1]))
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
                else:
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
