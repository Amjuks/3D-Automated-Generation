"""Conservative deterministic checks; reports distinguish coverage from guarantees."""

from collections import defaultdict, deque

import numpy as np
import trimesh

from .models import Bounds, Issue, ValidationReport
from .spatial import SpatialIndex


def validate(nodes, connections, geometry=None, triangle_limit=2_000_000, texture_limit=256_000_000):
    result = ValidationReport()
    ids = {n.id: n for n in nodes}

    def issue(node, code, message, severity="error"):
        result.issues.append(Issue(component_id=node.id, code=code, message=message, severity=severity))

    if not nodes:
        result.issues.append(Issue(component_id="scene", code="empty", message="scene contains no components"))
        return result
    if len(ids) != len(nodes):
        issue(nodes[0], "duplicate_id", "component IDs must be unique")
    roots = [n for n in nodes if n.parent_id is None]
    if len(roots) != 1:
        issue(nodes[0], "hierarchy", "expected one root")
    for n in nodes:
        parent = ids.get(n.parent_id)
        if n.parent_id and not parent:
            issue(n, "hierarchy", "missing parent")
        if parent:
            if not Bounds(max=parent.bounds.size).contains(n.bounds):
                issue(n, "bounds", "component allocation exceeds parent")
            expected = np.array(parent.world_transform.translation) + n.local_transform.translation
            if not np.allclose(expected, n.world_transform.translation, atol=1e-7):
                issue(n, "transform", "world transform does not compose with parent")
            if n.id not in parent.child_ids:
                issue(n, "hierarchy", "parent does not reference child")
        if not np.allclose(n.bounds.min, n.local_transform.translation):
            issue(n, "transform", "allocation and local translation disagree")
        for child in n.child_ids:
            if child not in ids or ids[child].parent_id != n.id:
                issue(n, "hierarchy", "child relationship is not reciprocal")
        for dep in n.dependencies:
            if dep not in ids:
                issue(n, "dependency", "missing dependency")
        for volume in n.reserved + n.clearance:
            if not Bounds(max=n.bounds.size).contains(volume):
                issue(n, "bounds", "reserved volume exceeds allocation")
        if n.generator != "group" and not n.materials:
            issue(n, "material", "mesh has no material")
        if n.support_id:
            support = ids.get(n.support_id)
            if not support:
                issue(n, "support", "missing support component")
            else:
                gap = n.world_bounds.min[2] - support.world_bounds.max[2]
                a, b = n.world_bounds, support.world_bounds
                overlap = all(min(a.max[k], b.max[k]) - max(a.min[k], b.min[k]) > 0 for k in (0, 1))
                if abs(gap) > 0.031 or not overlap:
                    issue(n, "floating", "object does not contact its declared support")
                if n.attachment_socket:
                    socket = next((s for s in support.sockets if s.id == n.attachment_socket), None)
                    if socket is None:
                        issue(n, "attachment", "attachment socket does not exist")
                    else:
                        position = np.array(support.world_transform.translation) + socket.position
                        if not (
                            a.min[0] - 0.01 <= position[0] <= a.max[0] + 0.01
                            and a.min[1] - 0.01 <= position[1] <= a.max[1] + 0.01
                            and abs(position[2] - a.min[2]) < 0.031
                        ):
                            issue(n, "attachment", "socket does not meet object support face")
        if (
            n.kind == "room"
            and n.name != "circulation"
            and not n.parameters.get("circulation")
            and min(n.bounds.size[:2]) < 5
        ):
            issue(n, "ergonomics", "room is smaller than supported ergonomic grammar")
    # Kahn sort covers parent/dependency cycles without recursion limits.
    incoming = {n.id: set(n.dependencies + ([n.parent_id] if n.parent_id else [])) for n in nodes}
    reverse = defaultdict(set)
    for nid, deps in incoming.items():
        for dep in deps:
            reverse[dep].add(nid)
    ready = deque(k for k, v in incoming.items() if not v)
    visited = set()
    while ready:
        nid = ready.popleft()
        visited.add(nid)
        for child in reverse[nid]:
            incoming[child].discard(nid)
            if not incoming[child]:
                ready.append(child)
    if len(visited) != len(nodes):
        issue(nodes[0], "dependency_cycle", "hierarchy/dependency graph is cyclic or references missing nodes")
    spatial = SpatialIndex()
    solids = [n for n in nodes if n.generator != "group" and n.collidable]
    for n in solids:
        for other in spatial.query(n.world_bounds):
            if n.parameters.get("assembly_id") and n.parameters["assembly_id"] == ids[other].parameters.get(
                "assembly_id"
            ):
                continue  # Intentional joints within one object; collisions with other objects still checked.
            issue(n, "collision", "AABB intersection with " + other)
        spatial.add(n.id, n.world_bounds)
    for mounted in (n for n in nodes if n.parameters.get("mounted")):
        a = mounted.world_bounds
        nearby = Bounds(min=tuple(v - 0.025 for v in a.min), max=tuple(v + 0.025 for v in a.max))
        attached = False
        for other_id in spatial.query(nearby):
            wall = ids[other_id]
            if not wall.materials or wall.materials[0].name not in {
                "wall",
                "trim",
                "limestone",
                "plaster",
                "concrete",
                "timber",
            }:
                continue
            b = wall.world_bounds
            for axis in (0, 1):
                contact = min(abs(a.min[axis] - b.max[axis]), abs(a.max[axis] - b.min[axis])) < 0.025
                overlap = all(min(a.max[k], b.max[k]) - max(a.min[k], b.min[k]) > 0.1 for k in range(3) if k != axis)
                attached |= contact and overlap
        if not attached:
            issue(mounted, "mount", "wall-mounted object has no contact with an opaque structural surface")

    # Clearance applies to furniture groups as well as meshes; exclude enclosing structural groups.
    for room in (n for n in nodes if n.kind == "room"):
        for volume in room.reserved + room.clearance:
            world = volume.translated(room.world_transform.translation)
            for other in spatial.query(world):
                issue(ids[other], "clearance", "obstructs a reserved route or door approach in " + room.id)
    rooms = {n.id for n in nodes if n.kind == "room"}
    adjacency = defaultdict(set)
    for connection in connections:
        a, b = connection["from"], connection["to"]
        if a not in rooms or b not in rooms or connection.get("width", 0) < 1.2:
            issue(nodes[0], "connectivity", "invalid room connection or inaccessible doorway")
            continue
        position = connection.get("position")
        if position is None or not all(
            all(room.min[k] - 0.01 <= position[k] <= room.max[k] + 0.01 for k in range(3))
            for room in (ids[a].world_bounds, ids[b].world_bounds)
        ):
            issue(ids[a], "connectivity", "door does not meet both connected rooms")
        adjacency[a].add(b)
        adjacency[b].add(a)
    entrances = [n.id for n in nodes if n.kind == "room" and any(s.id == "entrance" for s in n.sockets)]
    seen = set(entrances)
    queue = deque(entrances)
    while queue:
        for other in adjacency[queue.popleft()] - seen:
            seen.add(other)
            queue.append(other)
    for missing in rooms - seen:
        issue(ids[missing], "connectivity", "room has no route to an exterior entrance")
    triangles = vertices = 0
    if geometry is not None:
        for n in nodes:
            if n.generator == "group":
                continue
            try:
                arrays = geometry(n)
                v, f, uv = arrays["vertices"], arrays["faces"], arrays["uv"]
                if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
                    raise ValueError("invalid vertices")
                if (
                    f.ndim != 2
                    or f.shape[1] != 3
                    or not len(f)
                    or not np.issubdtype(f.dtype, np.integer)
                    or f.min() < 0
                    or f.max() >= len(v)
                ):
                    raise ValueError("invalid faces")
                if np.any(v.min(axis=0) < -1e-5) or np.any(v.max(axis=0) > np.array(n.bounds.size) + 1e-5):
                    issue(n, "mesh_bounds", "geometry exceeds its allocation")
                if uv.shape != (len(v), 2) or not np.isfinite(uv).all():
                    issue(n, "uv", "missing or invalid UVs")
                welded = trimesh.Trimesh(vertices=v, faces=f, process=True)
                if not welded.is_watertight or not welded.is_winding_consistent:
                    issue(
                        n,
                        "manifold",
                        "mesh is open or inconsistently wound",
                        "warning" if n.generator in {"asset", "sculpture"} else "error",
                    )
                if np.any(welded.area_faces < 1e-12):
                    issue(n, "degenerate", "mesh contains zero-area triangles")
                if len(f) > n.budget.triangles:
                    issue(n, "polygon_budget", "component exceeds triangle budget")
                triangles += len(f)
                vertices += len(v)
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                issue(n, "geometry", "geometry checkpoint is missing or invalid")
    texture_paths = {
        p
        for n in nodes
        for m in n.materials
        for p in (m.base_color_texture, m.normal_texture, m.roughness_texture)
        if p
    }
    texture_bytes = 0
    from pathlib import Path

    for path in texture_paths:
        if not Path(path).is_file():
            issue(nodes[0], "texture", "referenced texture is missing")
        else:
            texture_bytes += Path(path).stat().st_size
    if triangles > triangle_limit:
        issue(nodes[0], "polygon_budget", "scene exceeds triangle budget")
    if texture_bytes > texture_limit:
        issue(nodes[0], "texture_budget", "scene exceeds texture byte budget")
    result.stats = {
        "components": len(nodes),
        "meshes": len(solids),
        "triangles": triangles,
        "vertices": vertices,
        "materials": len({m.name for n in nodes for m in n.materials}),
        "textures": len(texture_paths),
        "texture_bytes": texture_bytes,
    }
    result.checks = {
        "bounds_transforms": "checked",
        "dependencies": "checked",
        "collisions": "conservative AABB broad phase",
        "clearance": "reserved door and straight circulation volumes",
        "connectivity": "door graph, geometric portals, exterior reachability",
        "floating": "declared supports and sockets; not a physics simulation",
        "ergonomics": "template room and doorway minimums",
        "meshes_uv_materials": "checked" if geometry else "not checked: no geometry supplied",
        "budgets": "checked",
    }
    return result
