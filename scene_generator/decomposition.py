"""Parent-owned allocations; designers select a grammar but never calculate placement."""

from concurrent.futures import ThreadPoolExecutor

from .models import Bounds, Budget, Component, Material, RoomDesign, Socket, Transform
from .spatial import box
from .util import stable_seed


class Builder:
    def __init__(self, scene_id, seed, quality, tokens):
        self.scene_id, self.seed, self.quality = scene_id, seed, quality
        self.nodes = []
        self.by_id = {}
        self.tokens = tokens
        self.connections = []

    def add(self, parent, name, kind, origin, size, generator="group", material="wall", **kwargs):
        identity = (parent.id if parent else self.scene_id) + "/" + name
        origin = tuple(origin)
        world = tuple(a + b for a, b in zip(parent.world_transform.translation, origin)) if parent else origin
        node = Component(
            id=identity,
            parent_id=parent.id if parent else None,
            name=name,
            kind=kind,
            bounds=box(origin, size),
            local_transform=Transform(translation=origin),
            world_transform=Transform(translation=world),
            generator=generator,
            materials=[self.tokens[material]] if generator != "group" else [],
            seed=stable_seed(self.seed, identity),
            dependencies=[parent.id] if parent else [],
            budget=Budget(detail=self.quality),
            **kwargs,
        )
        if parent:
            if not Bounds(max=parent.bounds.size).contains(node.bounds):
                raise ValueError(f"allocation exceeds parent: {identity}")
            parent.child_ids.append(identity)
        self.nodes.append(node)
        self.by_id[identity] = node
        return node

    def solid(self, parent, name, origin, size, material="wall", generator="box", **kwargs):
        return self.add(
            parent,
            name,
            "structural" if material in {"wall", "floor", "trim", "glass"} else "detail",
            origin,
            size,
            generator,
            material,
            **kwargs,
        )


def design_tokens(plan):
    classic = plan.style.lower() in {"neoclassical", "classical", "baroque"}
    concrete = plan.style.lower() == "brutalist"
    warm = plan.atmosphere in {"warm", "daylight"}
    values = {
        "wall": (
            (0.77, 0.75, 0.68, 1) if classic else (0.48, 0.49, 0.47, 1) if concrete else (0.89, 0.89, 0.85, 1),
            0.88,
            0,
        ),
        "floor": ((0.40, 0.29, 0.19, 1) if warm else (0.30, 0.34, 0.37, 1), 0.4, 0),
        "trim": ((0.88, 0.84, 0.71, 1) if classic else (0.20, 0.22, 0.24, 1), 0.38, 0.12),
        "wood": ((0.29, 0.14, 0.065, 1), 0.48, 0),
        "metal": ((0.10, 0.12, 0.13, 1), 0.25, 0.8),
        "accent": ((*plan.accent, 1), 0.65, 0),
        "stone": ((0.70, 0.68, 0.59, 1), 0.45, 0.05),
        "glass": ((0.63, 0.78, 0.85, 0.24), 0.12, 0),
        "paper": ((0.90, 0.87, 0.73, 1), 0.85, 0),
        "ground": ((0.17, 0.25, 0.10, 1) if plan.environment == "forest" else (0.44, 0.43, 0.39, 1), 1, 0),
        "leaf": ((0.16, 0.32, 0.10, 1), 0.9, 0),
    }
    return {name: Material(name=name, color=c, roughness=r, metallic=m) for name, (c, r, m) in values.items()}


def furnish(b, room, design, floor_id):
    w, d, h = room.bounds.size
    # Two side strips leave a 1.5 m minimum central route and door approach.
    usable = d - 2.8
    count = min(design.density, max(1, int(usable / 1.7)))
    for side in (0, 1):
        for i in range(count):
            x = (0.4 if w >= 6 else 0.25) if side == 0 else w - (1.6 if w >= 6 else 1.45)
            y = 1.9 + i * usable / count
            if design.furnishing == "gallery":
                parent = b.add(
                    room, f"display-{side}-{i}", "furniture", (x, y, 0.18), (1.2, 1.2, 2.15), support_id=floor_id
                )
                pedestal = b.solid(parent, "pedestal", (0, 0, 0), (1.2, 1.2, 0.9), "stone")
                pedestal.sockets = [Socket(id="top", position=(0.6, 0.6, 0.9))]
                obj = b.add(
                    parent,
                    "collection-object",
                    "object",
                    (0.2, 0.2, 0.9),
                    (0.8, 0.8, 1.1),
                    design.centerpiece,
                    "accent",
                    support_id=pedestal.id,
                    attachment_socket="top",
                )
                obj.dependencies.append(pedestal.id)
                b.solid(parent, "label", (0.2, 0.03, 0.91), (0.5, 0.16, 0.025), "paper", support_id=pedestal.id)
            elif design.furnishing in {"library", "study"}:
                parent = b.add(
                    room, f"bookcase-{side}-{i}", "furniture", (x, y, 0.18), (1.2, 1.2, 2.3), support_id=floor_id
                )
                for sx in (0, 1.13):
                    b.solid(parent, f"side-{sx}", (sx, 0, 0), (0.07, 0.4, 2.3), "wood")
                b.solid(parent, "back", (0.07, 0, 0), (1.06, 0.04, 2.3), "wood")
                for level in range(5):
                    shelf = b.solid(parent, f"shelf-{level}", (0.07, 0.04, level * 0.45), (1.06, 0.36, 0.05), "wood")
                    for j in range(7):
                        book = b.add(
                            parent,
                            f"book-{level}-{j}",
                            "object",
                            (0.09 + j * 0.145, 0.08, level * 0.45 + 0.05),
                            (0.10, 0.28, 0.30 + 0.01 * (j % 3)),
                            "box",
                            "accent" if j % 2 else "paper",
                            support_id=shelf.id,
                        )
                        book.dependencies.append(shelf.id)
            else:
                parent = b.add(
                    room, f"seat-{side}-{i}", "furniture", (x, y, 0.18), (1.2, 1.2, 0.95), support_id=floor_id
                )
                for ix in (0, 1):
                    for iy in (0, 1):
                        b.solid(
                            parent, f"leg-{ix}-{iy}", (0.10 + ix * 0.90, 0.10 + iy * 0.75, 0), (0.1, 0.1, 0.36), "wood"
                        )
                b.solid(parent, "cushion", (0.05, 0.05, 0.36), (1.1, 1.0, 0.14), "accent")
                b.solid(parent, "back", (0.05, 0.95, 0.5), (1.1, 0.1, 0.45), "accent")
    if design.wall_art:
        # Back-wall frames and layered canvas remain inside the room allocation.
        art_width = min(1.3, w / 2 - 1.6)
        for i in range(2):
            frame = b.add(
                room,
                f"artwork-{i}",
                "object",
                (w * (0.25 + 0.5 * i) - art_width / 2, d - 0.28, 1.5),
                (art_width, 0.10, 1.15),
                collidable=False,
                parameters={"mounted": True},
            )
            b.solid(frame, "frame", (0, 0.04, 0), (art_width, 0.06, 1.15), "trim")
            b.solid(frame, "canvas", (0.07, 0.02, 0.07), (art_width - 0.14, 0.02, 1.01), "accent")
            for j in range(3):
                b.solid(
                    frame,
                    f"relief-{j}",
                    (0.12 + j * (art_width - 0.24) / 3, 0, 0.2 + 0.15 * (j % 2)),
                    ((art_width - 0.4) / 3, 0.02, 0.4),
                    "paper",
                )


def decompose(plan, scene_id, seed, generation, llm, room_cache=None):
    if (
        ("museum" in plan.category.lower() or plan.category.lower() in {"gallery", "art_center"})
        and plan.layout != "legacy"
        and plan.galleries
    ):
        from .museum import compile_museum

        return compile_museum(plan, scene_id, seed, generation, llm, room_cache)
    tokens = design_tokens(plan)
    b = Builder(scene_id, seed, generation.quality, tokens)
    cw, cd, h = plan.room_width, plan.room_depth, plan.height
    bw, bd = cw * plan.columns, cd * plan.rows + 3
    root_origin = generation.bounding_space.min if generation.bounding_space else (0, 0, 0)
    root = b.add(None, "scene", "scene", root_origin, (bw + 6, bd + 6, h + 0.5))
    site = b.add(root, "site", "site", (0, 0, 0), root.bounds.size)
    ground = b.solid(site, "terrain", (0, 0, 0), (bw + 6, bd + 6, 0.20), "ground")
    building = b.add(site, "building", "building", (3, 3, 0.20), (bw, bd, h), support_id=ground.id)
    floor = b.add(building, "floor-0", "floor", (0, 0, 0), (bw, bd, h))
    hall_y = cd if plan.rows == 2 else 0
    hall = b.add(floor, "circulation", "room", (0, hall_y, 0), (bw, 3, h))
    b.solid(hall, "slab", (0, 0, 0), (bw, 3, 0.18), "floor")
    hall.reserved = [box((0.18, 0.45, 0.18), (bw - 0.36, 2.1, 2.4))]
    hall.clearance = [box((0, 0.75, 0.18), (0.18, 1.5, 2.4))]
    hall.sockets = [Socket(id="entrance", position=(0, 1.5, 0.18), normal=(-1, 0, 0), kind="door")]
    b.solid(hall, "ceiling", (0, 0, h - 0.12), (bw, 3, 0.12), "wall")
    b.solid(hall, "end-wall", (bw - 0.18, 0, 0.18), (0.18, 3, h - 0.3), "wall")
    # Entrance opening in west face, lintel and jambs.
    for yy in (0, 2.25):
        b.solid(hall, f"entrance-jamb-{yy}", (0, yy, 0.18), (0.18, 0.75, h - 0.3), "wall")
    b.solid(hall, "entrance-lintel", (0, 0.75, 2.65), (0.18, 1.5, h - 2.77), "wall")
    if plan.rows == 1:
        b.solid(hall, "front-wall", (0.18, 0, 0.18), (bw - 0.36, 0.18, h - 0.3), "wall")
    rooms = []
    for row in range(plan.rows):
        for col in range(plan.columns):
            y = 0 if plan.rows == 2 and row == 0 else hall_y + 3
            room = b.add(floor, f"room-{row}-{col}", "room", (col * cw, y, 0), (cw, cd, h))
            slab = b.solid(room, "slab", (0, 0, 0), (cw, cd, 0.18), "floor")
            b.solid(room, "ceiling", (0, 0, h - 0.12), (cw, cd, 0.12), "wall")
            for x, suffix in ((0, "west"), (cw - 0.10, "east")):
                b.solid(room, f"wall-{suffix}", (x, 0.18, 0.18), (0.10, cd - 0.36, h - 0.30), "wall")
            door_y = cd - 0.18 if y == 0 and plan.rows == 2 else 0
            back_y = 0 if door_y else cd - 0.18
            gap = 1.5
            jamb = (cw - gap) / 2
            for x, suffix in ((0, "left"), (jamb + gap, "right")):
                b.solid(room, f"door-wall-{suffix}", (x, door_y, 0.18), (jamb, 0.18, h - 0.3), "wall")
            b.solid(room, "door-lintel", (jamb, door_y, 2.65), (gap, 0.18, h - 2.77), "wall")
            room.sockets = [
                Socket(
                    id="door",
                    position=(cw / 2, cd if door_y else 0, 0.18),
                    normal=(0, 1 if door_y else -1, 0),
                    kind="door",
                )
            ]
            room.reserved = [box((cw / 2 - 0.75, 0.18, 0.18), (1.5, cd - 0.36, 2.4))]
            room.clearance = [box((cw / 2 - 1.0, cd - 1.8 if door_y else 0.18, 0.18), (2.0, 1.62, 2.4))]
            # Actual window opening, glass and sill, rather than glass inside an opaque wall.
            b.solid(room, "back-base", (0, back_y, 0.18), (cw, 0.18, 0.9), "wall")
            b.solid(room, "back-top", (0, back_y, h - 1.0), (cw, 0.18, 0.88), "wall")
            # Opaque piers provide physical artwork mounts between window panes.
            centers = (cw * 0.25, cw * 0.75)
            boundaries = sorted({0.0, 0.7, cw - 0.7, cw, *(c + delta for c in centers for delta in (-0.75, 0.75))})
            for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
                middle = (left + right) / 2
                opaque = middle < 0.7 or middle > cw - 0.7 or any(abs(middle - c) < 0.75 for c in centers)
                b.solid(
                    room,
                    f"window-band-{index}",
                    (left, back_y if opaque else back_y + 0.065, 1.08),
                    (right - left, 0.18 if opaque else 0.035, h - 2.08),
                    "wall" if opaque else "glass",
                )
            if plan.style.lower() == "neoclassical":
                for x, suffix in ((0.12, "left"), (cw - 0.35, "right")):
                    b.solid(
                        room, f"pilaster-{suffix}", (x, 0.3, 0.18), (0.23, 0.26, h - 0.42), "trim", generator="cylinder"
                    )
                b.solid(room, "cornice", (0.12, 0.3, h - 0.24), (cw - 0.24, 0.20, 0.12), "trim")
            b.connections.append(
                {
                    "from": room.id,
                    "to": hall.id,
                    "socket": "door",
                    "width": gap,
                    "position": tuple(
                        a + c for a, c in zip(room.world_transform.translation, room.sockets[0].position)
                    ),
                }
            )
            rooms.append((room, slab))

    def request(pair):
        room, slab = pair
        ordinal = rooms.index(pair)

        def mock():
            residential = plan.collection == "residential"
            return {
                "furnishing": ["lounge", "library", "study", "dining"][ordinal % 4] if residential else "gallery",
                "centerpiece": {"antiquities": "vase", "science": "sphere", "sculpture": "cylinder"}.get(
                    plan.collection, "vase"
                ),
                "density": 1 if generation.quality == "draft" else 2 + (ordinal % 2),
                "wall_art": True,
            }

        context = {
            "component": room.name,
            "seed": room.seed,
            "quality": generation.quality,
            "bounds": room.bounds.size,
            "parent_contract": {"door_width": 1.5, "central_path": 1.5, "furniture_strip_width": 1.2},
            "neighbors": ["public circulation hall"],
            "tokens": {"style": plan.style, "collection": plan.collection, "atmosphere": plan.atmosphere},
            "instruction": "Choose reusable furnishing generators appropriate to this room.",
        }
        # Cache persisted per room before proceeding to other work.
        from .util import read_json, write_json

        cache_path = room_cache / (room.name + ".json") if room_cache else None
        if cache_path and cache_path.exists():
            return RoomDesign.model_validate(read_json(cache_path))
        result = llm.request(RoomDesign, context, scene_id, room.id, mock=mock)
        if cache_path:
            write_json(cache_path, result.model_dump(mode="json"))
        return result

    with ThreadPoolExecutor(max_workers=generation.llm.concurrency) as pool:
        designs = list(pool.map(request, rooms))
    for (room, slab), design in zip(rooms, designs):
        furnish(b, room, design, slab.id)
    # Landscape objects occupy the outer site margins, away from the west entrance.
    if plan.environment in {"forest", "coastal"}:
        for i in range(max(2, plan.columns * 2)):
            x = 1 + i * (bw + 3) / max(2, plan.columns * 2)
            tree = b.add(
                site, f"tree-{i}", "furniture", (x, bd + 3.7, 0.20), (1.4, 1.4, min(h, 4.2)), support_id=ground.id
            )
            trunk = b.solid(tree, "trunk", (0.6, 0.6, 0), (0.2, 0.2, 1.8), "wood", generator="cylinder")
            b.solid(
                tree,
                "canopy",
                (0, 0, 1.8),
                (1.4, 1.4, min(h, 4.2) - 1.8),
                "leaf",
                generator="sphere",
                support_id=trunk.id,
            )
    return b.nodes, b.connections, tokens
