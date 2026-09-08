"""Museum design compiler: unequal galleries, planted courts and detailed exhibits."""

import random
from concurrent.futures import ThreadPoolExecutor

from .models import Material, RoomDesign, Socket
from .spatial import box
from .util import read_json, write_json


def palette(plan, tokens):
    values = {
        "limestone": ((0.65, 0.61, 0.50, 1), 0.72, 0),
        "plaster": ((0.79, 0.75, 0.65, 1), 0.85, 0),
        "concrete": ((0.45, 0.46, 0.43, 1), 0.88, 0),
        "timber": ((0.24, 0.115, 0.045, 1), 0.5, 0),
        "bronze": ((0.28, 0.20, 0.095, 1), 0.37, 0.78),
        "ceramic": ((0.52, 0.29, 0.15, 1), 0.38, 0),
        "terrazzo": ((0.57, 0.55, 0.49, 1), 0.45, 0),
        "parquet": ((0.34, 0.20, 0.095, 1), 0.48, 0),
        "gravel": ((0.40, 0.38, 0.31, 1), 0.93, 0),
        "roof": ((0.15, 0.18, 0.18, 1), 0.65, 0.25),
        "fabric": ((0.24, 0.29, 0.25, 1), 0.92, 0),
        "water": ((0.21, 0.35, 0.34, 0.8), 0.10, 0.15),
    }
    for key, (color, rough, metal) in values.items():
        tokens[key] = Material(name=key, color=color, roughness=rough, metallic=metal)
    tokens["glass"] = Material(
        name="glass", color=(0.88, 0.94, 0.97, 0.28), roughness=0.08, transmission=0.85, bevel=0.002
    )
    tokens["trim"] = tokens["limestone"].model_copy(update={"name": "trim", "bevel": 0.012})
    return tokens


def detail(b, parent, name, origin, size, material, generator="box", **kw):
    node = b.add(parent, name, "detail", origin, size, generator, material, **kw)
    if generator in {"sculpture", "rock", "urn", "tree", "torus"}:
        node.budget.triangles = 30000
    return node


def floor_surface(b, room, finish, quality):
    w, d, _ = room.bounds.size
    floor = b.add(room, "finished-floor", "structural", (0, 0, 0), (w, d, 0.18))
    detail(b, floor, "subfloor", (0, 0, 0), (w, d, 0.14), "concrete")
    tile = 1.4 if finish != "parquet" else 0.7
    nx, ny = max(1, round(w / tile)), max(1, round(d / tile))
    if quality == "draft":
        nx, ny = 2, 2
    tw, td = w / nx, d / ny
    for ix in range(nx):
        for iy in range(ny):
            n = detail(
                b,
                floor,
                f"paving-{ix}-{iy}",
                (ix * tw + 0.002, iy * td + 0.002, 0.14),
                (tw - 0.004, td - 0.004, 0.04),
                finish,
            )
            # Discrete stone/board tone variation remains small and physically plausible.
            variation = random.Random(n.seed).choice((0.97, 1.0, 1.03))
            n.materials = [
                n.materials[0].model_copy(
                    update={"color": tuple(min(1, c * variation) for c in n.materials[0].color[:3]) + (1,)}
                )
            ]
    return floor


def perimeter(b, room, entry, finish, roof):
    w, d, _ = room.bounds.size
    h = room.parameters["wall_height"]
    # Walls are split around real openings. Long external faces have deep window reveals.
    for edge, length in (("south", w), ("north", w), ("west", d), ("east", d)):
        across_x = edge in {"south", "north"}
        if not across_x:
            length -= 0.44
        edge_origin = (0, 0) if edge in {"south", "west"} else ((0, d - 0.22) if edge == "north" else (w - 0.22, 0))
        if not across_x:
            edge_origin = (edge_origin[0], 0.22)

        def piece(name, start, z, width, height, depth=0.22, mat=finish, generator="box"):
            origin = (
                (edge_origin[0] + start, edge_origin[1], z) if across_x else (edge_origin[0], edge_origin[1] + start, z)
            )
            size = (width, depth, height) if across_x else (depth, width, height)
            return detail(b, room, f"{edge}-{name}", origin, size, mat, generator)

        if edge == entry:
            gap = 2.1
            jamb = (length - gap) / 2
            piece("jamb-left", 0, 0.18, jamb, h - 0.18)
            piece("jamb-right", jamb + gap, 0.18, jamb, h - 0.18)
            piece("lintel", jamb, 3.65, gap, h - 3.65)
            # The curved head is visible above a fully accessible rectangular portal.
            if across_x:
                piece("arch", jamb, 2.65, gap, 1.0, mat="limestone", generator="arch")
            else:
                piece("arch-lintel", jamb, 3.4, gap, 0.25, mat="limestone")
        else:
            # Opaque lower walls accept paintings; clerestories admit daylight above.
            piece("dado", 0, 0.18, length, 2.7)
            piece("head", 0, h - 0.35, length, 0.35)
            columns = max(2, int(length / 2.8))
            for k in range(columns):
                cell = length / columns
                piece(f"mullion-{k}", k * cell, 2.88, 0.20, h - 3.23)
                piece(f"glass-{k}", k * cell + 0.20, 2.88, cell - 0.20, h - 3.23, 0.035, "glass")
            # Dado cap, skirting, wall joints: tiny offsets cast real contact shadows.
            if across_x:
                y = 0.22 if edge == "south" else d - 0.265
                detail(b, room, f"{edge}-skirting", (0.22, y, 0.18), (w - 0.44, 0.045, 0.16), "trim")
                detail(b, room, f"{edge}-dado-cap", (0.22, y, 2.68), (w - 0.44, 0.045, 0.08), "trim")
            else:
                x = 0.22 if edge == "west" else w - 0.265
                detail(b, room, f"{edge}-skirting", (x, 0.275, 0.18), (0.045, d - 0.55, 0.16), "trim")
    # Explicit roof types change both the interior section and exterior silhouette.
    if roof == "vault":
        detail(b, room, "barrel-vault", (0.22, 0.22, h), (w - 0.44, d - 0.44, 1.4), "plaster", "vault")
        detail(b, room, "roof-south-end", (0.22, 0.0, h), (w - 0.44, 0.18, 1.4), "limestone", "arch")
        detail(b, room, "roof-north-end", (0.22, d - 0.18, h), (w - 0.44, 0.18, 1.4), "limestone", "arch")
    elif roof == "skylight":
        opening = w * 0.38
        side = (w - opening) / 2
        detail(b, room, "roof-west", (0, 0, h), (side, d, 0.16), "roof")
        detail(b, room, "roof-east", (side + opening, 0, h), (side, d, 0.16), "roof")
        detail(b, room, "skylight", (side, 0, h + 0.10), (opening, d, 0.05), "glass")
        for k in range(1, max(2, int(d / 2))):
            y = k * d / max(2, int(d / 2))
            detail(b, room, f"skylight-rib-{k}", (side, y, h + 0.16), (opening, 0.07, 0.12), "metal")
    else:
        detail(b, room, "roof-slab", (0, 0, h), (w, d, 0.20), "roof")
        for k in range(1, 4):
            detail(b, room, f"coffer-beam-{k}", (0.22, k * d / 4, h - 0.20), (w - 0.44, 0.16, 0.20), "timber")
    # Preserve shell volume separately from the upper roof allocation.
    room.parameters["wall_height"] = h


def label(b, parent, name, position, size, text):
    n = detail(b, parent, name, position, size, "paper", parameters={"surface": "label", "text": text})
    return n


def bench(b, room, name, x, y, width, floor_id):
    seat = b.add(room, name, "furniture", (x, y, 0.18), (width, 0.60, 0.47), support_id=floor_id)
    for k in (0, 1):
        detail(b, seat, f"leg-{k}", (0.10 + k * (width - 0.24), 0.08, 0), (0.14, 0.44, 0.35), "bronze")
    for j in range(5):
        detail(b, seat, f"timber-slat-{j}", (0, j * 0.12, 0.35), (width, 0.112, 0.12), "timber")


def furnish_gallery(b, room, brief, design, floor, entry):
    w, d, total_h = room.bounds.size
    h = room.parameters["wall_height"]
    rng = random.Random(room.seed)
    # Route direction follows the real portal; side galleries use a cross-room path.
    vertical = entry in {"north", "south"}
    path_length = d if vertical else w
    across = w if vertical else d
    count = 1 if b.quality == "draft" else min(3, max(2, int((path_length - 3) / 2.5)))
    if design.arrangement == "sparse":
        count = 1
    for side in (0, 1):
        for k in range(count):
            along = 1.6 + k * max(2.0, (path_length - 4) / count) + rng.uniform(-0.15, 0.15)
            cross = 0.75 if side == 0 else across - 2.0
            x, y = (cross, along) if vertical else (along, cross)
            width = rng.uniform(0.70, 1.05)
            depth = rng.uniform(0.65, 0.95)
            pedestal_height = rng.uniform(0.65, 0.95)
            exhibit_h = rng.uniform(0.6, 1.15)
            group = b.add(
                room, f"exhibit-{side}-{k}", "furniture", (x, y, 0.18), (1.25, 1.25, 2.35), support_id=floor.id
            )
            plinth = detail(
                b, group, "plinth", (0, 0, 0), (width, depth, pedestal_height), "limestone" if side == 0 else "timber"
            )
            theme = brief.theme
            choices = {
                "antiquities": ["urn", "sculpture", "urn"],
                "ceramics": ["urn", "vase", "urn"],
                "sculpture": ["sculpture", "torus", "sculpture"],
                "contemporary": ["torus", "rock", "sculpture"],
                "natural_history": ["rock", "rock", "urn"],
                "paintings": ["sculpture", "urn", "torus"],
            }
            recipe = choices[theme][(k + side) % 3]
            objw = min(width - 0.12, rng.uniform(0.45, 0.78))
            objd = min(depth - 0.12, 0.60)
            obj = detail(
                b,
                group,
                "collection-object",
                ((width - objw) / 2, (depth - objd) / 2, pedestal_height),
                (objw, objd, exhibit_h),
                "bronze"
                if recipe in {"torus", "sculpture"} and side
                else "ceramic"
                if recipe in {"urn", "vase"}
                else "stone",
                recipe,
                support_id=plinth.id,
                parameters={"theme": theme, "asset_candidate": k == 0 and side == 0},
            )
            obj.dependencies.append(plinth.id)
            label(
                b,
                group,
                "catalogue-label",
                (0.05, depth + 0.03, 0.04),
                (0.36, 0.035, 0.24),
                f"{brief.name} | Object {k + side * count + 1:02d}",
            )
            # Glass vitrines are thin individual panes, so artifacts do not collide with a solid glass box.
            if theme in {"antiquities", "ceramics", "natural_history"} and k % 2 == 0:
                z = pedestal_height
                tall = exhibit_h + 0.10
                for j, gx in enumerate((0.01, width - 0.022)):
                    detail(b, group, f"vitrine-side-{j}", (gx, 0.01, z), (0.012, depth - 0.02, tall), "glass")
                for j, gy in enumerate((0.01, depth - 0.022)):
                    detail(b, group, f"vitrine-face-{j}", (0.022, gy, z), (width - 0.044, 0.012, tall), "glass")
                detail(b, group, "vitrine-top", (0.01, 0.01, z + tall), (width - 0.02, depth - 0.02, 0.012), "glass")
    # Seating occupies the back corners, outside exhibit strips and the accessible route.
    if vertical and d > 8:
        bench(b, room, "rest-bench", 0.65, d - 1.0, min(2.0, w / 2 - 1.8), floor.id)
    elif not vertical and w > 8:
        bench(b, room, "rest-bench", w - 2.5, 0.6, 1.8, floor.id)
    # Paintings and curatorial panels live on actual opaque walls below clerestories.
    edges = ["north", "south"] if vertical else ["west", "east"]
    edge = next(e for e in edges if e != entry)
    for k in range(2 if brief.theme != "paintings" else 3):
        length = w if vertical else d
        span = min(1.4, (length - 3) / 3)
        p = 0.9 + k * (length - 2) / 3
        if vertical:
            origin = (p, 0.235 if edge == "south" else d - 0.30, 1.25)
            dims = (span, 0.065, 1.05)
        else:
            origin = (0.235 if edge == "west" else w - 0.30, p, 1.25)
            dims = (0.065, span, 1.05)
        frame = b.add(room, f"painting-{k}", "object", origin, dims, parameters={"mounted": True})
        if vertical:
            # Front face faces inward from the north wall.
            detail(b, frame, "gilded-frame", (0, 0.025, 0), (span, 0.04, 1.05), "bronze")
            detail(
                b,
                frame,
                "canvas",
                (0.055, 0, 0.055),
                (span - 0.11, 0.025, 0.94),
                "paper",
                parameters={"surface": "artwork"},
            )
        else:
            detail(b, frame, "gilded-frame", (0.025, 0, 0), (0.04, span, 1.05), "bronze")
            detail(
                b,
                frame,
                "canvas",
                (0, 0.055, 0.055),
                (0.025, span - 0.11, 0.94),
                "paper",
                parameters={"surface": "artwork", "uv_axes": [1, 2]},
            )
    # Visible track hardware and light sources share the same coordinates and targets.
    for k in range(3):
        x = w * (0.25 + 0.25 * k)
        detail(b, room, f"light-track-{k}", (x, 0.6, h - 0.28), (0.045, d - 1.2, 0.045), "metal")
        for j in range(2):
            y = d * (0.30 + 0.35 * j)
            detail(
                b,
                room,
                f"spot-{k}-{j}",
                (x - 0.07, y, h - 0.44),
                (0.14, 0.14, 0.16),
                "metal",
                "cylinder",
                parameters={
                    "light": {"type": "SPOT", "energy": 110, "color": [1, 0.83, 0.64], "target": [x, y + 1.0, 0.9]}
                },
            )
    room.parameters["camera"] = {"position": [w * 0.49, 1.15, 1.65], "target": [w * 0.34, d * 0.70, 1.35]}
    if entry == "north":
        room.parameters["camera"] = {"position": [w * 0.49, d - 1.15, 1.65], "target": [w * 0.32, d * 0.28, 1.35]}
    if not vertical:
        room.parameters["camera"] = {"position": [1.15, d * 0.49, 1.65], "target": [w * 0.72, d * 0.35, 1.35]}


def landscape(b, site, origin, size, name, floor_id=None):
    w, d = size
    bed = b.add(site, name, "landscape", (origin[0], origin[1], origin[2]), (w, d, 4.6))
    detail(b, bed, "mineral-bed", (0, 0, 0), (w, d, 0.12), "gravel")
    rng = random.Random(bed.seed)
    for i in range(max(3, int(w * d / 5))):
        x = rng.uniform(0.3, max(0.31, w - 0.9))
        y = rng.uniform(0.3, max(0.31, d - 0.9))
        detail(
            b, bed, f"shrub-{i}", (x, y, 0.12), (0.55, 0.65, rng.uniform(0.25, 0.6)), "leaf", "tree", collidable=False
        )
    if w > 4 and d > 4:
        x, y = w * 0.57, d * 0.47
        detail(b, bed, "tree-trunk", (x, y, 0.12), (0.23, 0.23, 2.9), "timber", "cylinder")
        for k in range(5):
            dx, dy = rng.uniform(-0.7, 0.4), rng.uniform(-0.6, 0.4)
            detail(
                b,
                bed,
                f"canopy-{k}",
                (x + dx, y + dy, 1.8 + 0.25 * k),
                (1.8, 1.8, 1.6),
                "leaf",
                "tree",
                collidable=False,
            )


def compile_museum(plan, scene_id, seed, generation, llm, room_cache):
    from .decomposition import Builder, design_tokens

    tokens = palette(plan, design_tokens(plan))
    b = Builder(scene_id, seed, generation.quality, tokens)
    briefs = plan.galleries
    sizes = [
        (max(6.8, plan.room_width * g.width), max(7.2, plan.room_depth * g.depth), max(4.0, plan.height * g.height))
        for g in briefs
    ]
    placements = []
    if plan.layout == "courtyard":
        sizes = sizes[:4]
        north, east, south, west = sizes
        cw = max(north[0], south[0]) + 6.4
        cd = max(east[1], west[1]) + 6.4
        cx, cy = west[0] + 4, south[1] + 4
        placements = [
            (cx + (cw - north[0]) / 2, cy + cd, "south"),
            (cx + cw, cy + (cd - east[1]) / 2, "west"),
            (cx + (cw - south[0]) / 2, cy - south[1], "north"),
            (cx - west[0], cy + (cd - west[1]) / 2, "east"),
        ]
        sw, sd = cx + cw + east[0] + 4, cy + cd + north[1] + 4
        circulation_origin = (cx, cy, 0)
        circulation_size = (cw, cd, max(s[2] for s in sizes) + 1.6)
    else:
        gap = 1.8 if plan.layout == "pavilions" else 0.5
        maxdepth = max(s[1] for s in sizes)
        hall_y = maxdepth + 4 if plan.layout == "pavilions" else 4
        cursor = 4.0
        for i, (w, d, h) in enumerate(sizes):
            south = plan.layout == "pavilions" and i % 2
            placements.append((cursor, hall_y - d if south else hall_y + 4.2, "north" if south else "south"))
            cursor += w + gap
        sw, sd = cursor + 4, hall_y + 4.2 + maxdepth + 4
        circulation_origin = (4, hall_y, 0)
        circulation_size = (cursor - 4, 4.2, max(s[2] for s in sizes) + 1.6)
    height = max(s[2] for s in sizes) + 1.9
    root_origin = generation.bounding_space.min if generation.bounding_space else (0, 0, 0)
    limits = generation.bounding_space.size if generation.bounding_space else generation.dimensions
    if limits and any(a > b for a, b in zip((sw, sd, height), limits)):
        raise ValueError(
            f"{plan.layout} museum needs at least {sw:.1f} x {sd:.1f} x {height:.1f} m; enlarge dimensions or use a smaller concept"
        )
    root = b.add(None, "scene", "scene", root_origin, (sw, sd, height))
    site = b.add(root, "site", "site", (0, 0, 0), (sw, sd, height))
    detail(b, site, "terrain", (0, 0, 0), (sw, sd, 0.18), "ground")
    building = b.add(site, "museum", "building", (0, 0, 0.18), (sw, sd, height - 0.18))
    floor = b.add(building, "ground-floor", "floor", (0, 0, 0), building.bounds.size)
    circ = b.add(
        floor,
        "cloister" if plan.layout == "courtyard" else "promenade",
        "room",
        circulation_origin,
        circulation_size,
        parameters={"circulation": True},
    )
    cw, cd, ch = circulation_size
    circ.sockets = [Socket(id="entrance", kind="door", position=(1.2, 0, 0.18), normal=(0, -1, 0))]
    if plan.layout == "courtyard":
        strips = [
            ((0, 0, 0), (cw, 3.1, 0.18)),
            ((0, cd - 3.1, 0), (cw, 3.1, 0.18)),
            ((0, 3.1, 0), (3.1, cd - 6.2, 0.18)),
            ((cw - 3.1, 3.1, 0), (3.1, cd - 6.2, 0.18)),
        ]
        circ.reserved = [
            box((0.3, 0.3, 0.18), (cw - 0.6, 2.0, 2.4)),
            box((0.3, cd - 2.3, 0.18), (cw - 0.6, 2.0, 2.4)),
            box((0.3, 2.3, 0.18), (2.0, cd - 4.6, 2.4)),
            box((cw - 2.3, 2.3, 0.18), (2.0, cd - 4.6, 2.4)),
        ]
        for i, (p, s) in enumerate(strips):
            detail(b, circ, f"arcade-floor-{i}", p, s, "limestone")
            detail(b, circ, f"arcade-roof-{i}", (p[0], p[1], 3.5), (s[0], s[1], 0.16), "limestone")
        for side, y in enumerate((2.55, cd - 3.0)):
            for k in range(max(2, int((cw - 6) / 3))):
                x = 3.3 + k * (cw - 7) / max(1, int((cw - 6) / 3) - 1)
                detail(b, circ, f"column-base-{side}-{k}", (x, y, 0.18), (0.45, 0.45, 0.20), "limestone")
                detail(
                    b,
                    circ,
                    f"column-shaft-{side}-{k}",
                    (x + 0.085, y + 0.085, 0.38),
                    (0.28, 0.28, 2.86),
                    "limestone",
                    "cylinder",
                )
                detail(b, circ, f"column-capital-{side}-{k}", (x, y, 3.24), (0.45, 0.45, 0.26), "limestone")
        landscape(b, circ, (3.3, 3.3, 0), (cw - 6.6, cd - 6.6), "courtyard-garden")
    else:
        detail(b, circ, "promenade-floor", (0, 0, 0), (cw, cd, 0.18), "limestone")
        detail(b, circ, "glazed-canopy", (0, 0, 3.6), (cw, cd, 0.08), "glass")
        circ.reserved = [box((0.25, 1.1, 0.18), (cw - 0.5, 2.0, 2.4))]
        for k in range(max(2, int(cw / 3))):
            x = 0.4 + k * (cw - 0.8) / max(1, int(cw / 3) - 1)
            for side, y in enumerate((0.25, cd - 0.45)):
                detail(b, circ, f"arcade-post-{side}-{k}", (x, y, 0.18), (0.16, 0.16, 3.42), "bronze")
    rooms = []
    for i, (brief, (w, d, h), (x, y, entry)) in enumerate(zip(briefs, sizes, placements)):
        room = b.add(
            floor,
            f"gallery-{i + 1:02d}",
            "room",
            (x, y, 0),
            (w, d, h + 1.55),
            parameters={"title": brief.name, "theme": brief.theme},
        )
        finished = floor_surface(b, room, plan.floor_finish, generation.quality)
        # Build the shell against its wall height; upper allocation remains available for roof geometry.
        room.parameters["wall_height"] = h
        # perimeter accepts wall height explicitly via the temporary parameter below.
        perimeter(b, room, entry, brief.finish, brief.roof)
        vertical = entry in {"north", "south"}
        if vertical:
            portal = (w / 2, 0 if entry == "south" else d, 0.18)
            room.reserved = [box((w / 2 - 0.9, 0.35, 0.18), (1.8, d - 0.70, 2.4))]
        else:
            portal = (0 if entry == "west" else w, d / 2, 0.18)
            room.reserved = [box((0.35, d / 2 - 0.9, 0.18), (w - 0.70, 1.8, 2.4))]
        room.sockets = [Socket(id="door", kind="door", position=portal)]
        b.connections.append(
            {
                "from": room.id,
                "to": circ.id,
                "socket": "door",
                "width": 2.1,
                "position": tuple(a + c for a, c in zip(room.world_transform.translation, portal)),
            }
        )
        rooms.append((room, brief, finished, entry))

    def design(item):
        room, brief, finished, entry = item
        context = {
            "gallery": brief.model_dump(),
            "seed": room.seed,
            "concept": plan.concept,
            "style": plan.style,
            "allocation_m": room.bounds.size,
            "door_edge": entry,
            "reserved_route_width_m": 1.8,
            "instruction": "Curate a distinctive gallery. Choose sparse/curated/clustered composition and lighting appropriate to this theme. Do not repeat a single exhibit throughout the museum. Label should be a short curatorial title.",
        }
        path = room_cache / (room.name + ".json") if room_cache else None
        if path and path.exists():
            return RoomDesign.model_validate(read_json(path))
        result = llm.request(
            RoomDesign,
            context,
            scene_id,
            room.id,
            mock=lambda: {
                "furnishing": "gallery",
                "centerpiece": "vase",
                "density": 2,
                "wall_art": True,
                "arrangement": "sparse" if brief.theme == "sculpture" else "curated",
                "exhibit_family": brief.theme
                if brief.theme in {"ceramics", "sculpture", "paintings", "natural_history"}
                else "mixed",
                "lighting": "soft_daylight" if brief.roof == "skylight" else "warm_spots",
                "label": brief.name,
            },
        )
        if path:
            write_json(path, result.model_dump(mode="json"))
        return result

    with ThreadPoolExecutor(max_workers=generation.llm.concurrency) as pool:
        designs = list(pool.map(design, rooms))
    for (room, brief, finished, entry), design_result in zip(rooms, designs):
        furnish_gallery(b, room, brief, design_result, finished, entry)
    return b.nodes, b.connections, tokens
