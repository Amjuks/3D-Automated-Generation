"""Generic bounded allocations and circulation around the actual zone footprints."""

import math
import random

import networkx as nx

from .util import stable_seed


def zone_layout(brief):
    count = len(brief.zones)
    cols = count if brief.composition == "linear" else math.ceil(math.sqrt(count))
    cellw = max(z.width for z in brief.zones) + 5
    celld = max(z.depth for z in brief.zones) + 5
    origins = []
    for i, zone in enumerate(brief.zones):
        if brief.composition == "freeform":
            x, y = zone.x, zone.y
        else:
            row, col = divmod(i, cols)
            x = col * cellw + (1.2 * (i % 2) if brief.composition == "staggered" else 0)
            y = row * celld
        origins.append((x + 3, y + 3.3, 0.2))
    extent = (
        max(p[0] + z.width for p, z in zip(origins, brief.zones)) + 4,
        max(p[1] + z.depth for p, z in zip(origins, brief.zones)) + 3.7,
        max(z.floors * z.floor_height for z in brief.zones) + 1,
    )
    return origins, extent


def circulation_segments(zones, origins):
    """Route a rectilinear tree through free space, connecting every zone entrance.

    The 1.2m surface plus clearance fits the brief's 3m zone gaps. No path
    crosses a zone interior, including in freeform compositions.
    """
    margin = 0.75
    obstacles = [
        (x - margin, y - margin, x + z.width + margin, y + z.depth + margin) for z, (x, y, _) in zip(zones, origins)
    ]
    ports = [(x + z.width / 2, y - margin) for z, (x, y, _) in zip(zones, origins)]
    start = (ports[0][0], 0.75)
    xs = sorted({p[0] for p in ports} | {v for a, _, b, _ in obstacles for v in (a, b)})
    ys = sorted({start[1]} | {v for _, a, _, b in obstacles for v in (a, b)})
    graph = nx.Graph()
    for x in xs:
        for y in ys:
            if not any(a < x < b and c < y < d for a, c, b, d in obstacles):
                graph.add_node((x, y))
    for x in xs:
        for a, b in zip(ys, ys[1:]):
            if (
                (x, a) in graph
                and (x, b) in graph
                and not any(left < x < right and low < (a + b) / 2 < high for left, low, right, high in obstacles)
            ):
                graph.add_edge((x, a), (x, b), weight=b - a)
    for y in ys:
        for a, b in zip(xs, xs[1:]):
            if (
                (a, y) in graph
                and (b, y) in graph
                and not any(low < y < high and left < (a + b) / 2 < right for left, low, right, high in obstacles)
            ):
                graph.add_edge((a, y), (b, y), weight=b - a)
    edges = set()
    for port in ports:
        try:
            route = nx.shortest_path(graph, start, port, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise ValueError("zone footprints leave no connected circulation route") from None
        edges.update(tuple(sorted((a, b))) for a, b in zip(route, route[1:]))
    segments = []
    for (ax, ay), (bx, by) in sorted(edges):
        if ax == bx:
            segments.append(((ax - 0.6, ay - 0.6, 0.2), (1.2, by - ay + 1.2, 0.06)))
        else:
            segments.append(((ax - 0.6, ay - 0.6, 0.2), (bx - ax + 1.2, 1.2, 0.06)))
    for (x, y), (_, zy, _) in zip(ports, origins):
        segments.append(((x - 0.6, y, 0.2), (1.2, zy - y, 0.06)))
    return segments


def object_placements(zone, design, floor, seed):
    """Seeded placements with a safe packing fallback and an explicit scale report."""
    requests = []
    for oi, spec in enumerate(design.objects):
        if spec.floor is not None and spec.floor >= zone.floors:
            raise ValueError("object targets a floor outside its zone")
        for k in range(spec.count):
            target = spec.floor if spec.floor is not None else k % zone.floors
            if spec.repeat_on_floors or target == floor:
                requests.append((oi, spec, k))
    if not requests:
        return []
    rng = random.Random(stable_seed(seed, floor, "placement"))
    if design.arrangement != "rows":
        rng.shuffle(requests)
    indoor = zone.enclosure == "building"
    w, d, h = zone.width, zone.depth - (3 if indoor else 0), zone.floor_height
    cols = 2 if indoor else max(1, min(len(requests), round(math.sqrt(len(requests) * w / d))))
    rows = math.ceil(len(requests) / cols)
    cellw = (w - (2.8 if indoor else 0.9)) / cols
    celld = (d - 1.4) / rows
    prepared = []
    for index, (oi, spec, k) in enumerate(requests):
        factor = min(1, (cellw - 0.15) / spec.width, (celld - 0.15) / spec.depth, (h - 0.6) / spec.height)
        if factor <= 0:
            raise ValueError("object density exceeds the zone allocation")
        size = tuple(v * factor for v in (spec.width, spec.depth, spec.height))
        col, row = index % cols, index // cols
        x = (0.45 if col == 0 else w - 0.45 - size[0]) if indoor else 0.45 + col * cellw
        y = 0.7 + row * celld
        prepared.append((oi, spec, k, size, factor, x, y))
    if design.arrangement == "rows":
        return prepared

    placed = []
    center = (rng.uniform(w * 0.35, w * 0.65), rng.uniform(d * 0.35, d * 0.65))
    for oi, spec, k, size, factor, sx, sy in prepared:
        for attempt in range(160):
            if design.arrangement == "clustered":
                x, y = rng.gauss(center[0], w * 0.18), rng.gauss(center[1], d * 0.18)
            else:
                x, y = rng.uniform(0.35, w - size[0] - 0.35), rng.uniform(0.35, d - size[1] - 0.35)
            if design.arrangement == "perimeter":
                edge = rng.randrange(4)
                if edge < 2:
                    x = 0.35 if edge == 0 else w - size[0] - 0.35
                else:
                    y = 0.35 if edge == 2 else d - size[1] - 0.35
            if not (0.3 <= x <= w - size[0] - 0.3 and 0.3 <= y <= d - size[1] - 0.3):
                continue
            if indoor and x + size[0] > w / 2 - 1.0 and x < w / 2 + 1.0:
                continue
            if all(
                x + size[0] + 0.15 <= ox or ox + os[0] + 0.15 <= x or y + size[1] + 0.15 <= oy or oy + os[1] + 0.15 <= y
                for _, _, _, os, _, ox, oy in placed
            ):
                placed.append((oi, spec, k, size, factor, x, y))
                break
        else:
            # Restart the whole allocation: mixing random placements with fallback
            # cells can introduce collisions with already placed objects.
            return prepared
    return placed
