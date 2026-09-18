"""Relationship projection and diagnostics. AABBs are derived broad-phase data."""

import graphlib
import itertools
import time
from typing import Literal

import numpy as np
import trimesh
from pydantic import Field

from .design import Diagnostic, GraphReport, RepairRecord
from .models import Bounds, Model


class ResolvedScene(Model):
    schema_version: Literal[2] = 2
    matrices: dict[str, list[list[float]]]
    bounds: dict[str, Bounds]
    recipe_matrices: dict[str, dict[str, list[list[float]]]] = Field(default_factory=dict)
    report: GraphReport = Field(default_factory=GraphReport)


def corners(bounds):
    return np.array(list(itertools.product(*zip(bounds.min, bounds.max))))


def envelope(points):
    points = np.asarray(points)
    low, high = points.min(axis=0), points.max(axis=0)
    # Degenerate semantic envelopes are not represented as fake volume.
    return Bounds(min=tuple(low), max=tuple(high)) if np.all(high > low) else None


def transformed_bounds(bounds, matrix):
    return envelope(trimesh.transform_points(corners(bounds), matrix))


def node_points(node, graph, recipe_matrices=None):
    points = []
    if node.spatial.envelope:
        points.extend(corners(node.spatial.envelope))
    if node.spatial.polygon and node.spatial.height:
        points.extend((x, y, z) for x, y in node.spatial.polygon for z in (0, node.spatial.height))
    points.extend(node.spatial.points)
    if node.recipe in graph.recipes and node.count:
        for instance in node.instances or [None]:
            inst = np.eye(4) if instance is None else np.asarray(instance.matrix)
            for part in graph.recipes[node.recipe].parts:
                frame = (recipe_matrices or {}).get(node.recipe, {}).get(part.id, part.frame.matrix)
                points.extend(trimesh.transform_points(corners(Bounds(max=part.size)), inst @ frame))
    return np.asarray(points).reshape((-1, 3))


def point_inside_polygon(point, polygon):
    x, y = point
    inside = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if (
            abs(cross) < 1e-8
            and min(ax, bx) - 1e-8 <= x <= max(ax, bx) + 1e-8
            and min(ay, by) - 1e-8 <= y <= max(ay, by) + 1e-8
        ):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def segment_hits_box(start, end, bounds, eps):
    low, high = eps, 1 - eps
    delta = end - start
    for axis in range(3):
        if abs(delta[axis]) < eps:
            if not bounds.min[axis] < start[axis] < bounds.max[axis]:
                return False
        else:
            a, b = sorted(
                ((bounds.min[axis] - start[axis]) / delta[axis], (bounds.max[axis] - start[axis]) / delta[axis])
            )
            low, high = max(low, a), min(high, b)
    return low < high


def resolve(graph, policy, check_collisions=True):
    from .design import Assembly, DesignGraph, DesignNode, Provenance
    from .models import Transform

    expanded_parts = sum(n.count * len(graph.recipes[n.recipe].parts) for n in graph.nodes if n.recipe in graph.recipes)
    recipe_parts = sum(len(recipe.parts) for recipe in graph.recipes.values())
    if len(graph.nodes) > policy.max_nodes or max(expanded_parts, recipe_parts) > policy.max_parts:
        return ResolvedScene(
            matrices={},
            bounds={},
            report=GraphReport(
                diagnostics=[
                    Diagnostic(
                        layer="budget",
                        code="spatial_preflight",
                        message="node or expanded part budget exceeded before spatial allocation",
                    )
                ]
            ),
        )
    recipe_matrices = {}
    recipe_report = GraphReport()
    for identity, recipe in graph.recipes.items():
        if not recipe.relationships:
            continue
        parts = []
        recipes = {}
        for part in recipe.parts:
            operation = part.model_copy(deep=True)
            operation.frame = Transform(contract_version=2)
            recipes[part.id] = Assembly(id=part.id, visual_intent=recipe.visual_intent, parts=[operation])
            parts.append(
                DesignNode(
                    id=part.id,
                    kind="object",
                    description=recipe.visual_intent,
                    recipe=part.id,
                    spatial={"frame": part.frame},
                    provenance=Provenance(source="resolver"),
                    relationships=[c for c in recipe.relationships if c.source == part.id],
                )
            )
        local_graph = DesignGraph(
            id=identity,
            title=recipe.visual_intent,
            intent=recipe.visual_intent,
            nodes=parts,
            recipes=recipes,
            materials=graph.materials,
        )
        local_policy = policy.model_copy(update={"site": None, "required_reachable": [], "minimum_route_width": None})
        local = resolve(local_graph, local_policy, check_collisions=False)
        recipe_matrices[identity] = local.matrices
        for diagnostic in local.report.diagnostics:
            diagnostic.nodes = [f"recipe/{identity}/{part}" for part in diagnostic.nodes]
            recipe_report.diagnostics.append(diagnostic)
        for repair in local.report.repairs:
            repair.node = f"recipe/{identity}/{repair.node}"
            recipe_report.repairs.append(repair)

    by_id = {n.id: n for n in graph.nodes}
    order = list(
        graphlib.TopologicalSorter({n.id: [n.parent_id] if n.parent_id else [] for n in graph.nodes}).static_order()
    )
    locals_ = {n.id: np.asarray(n.spatial.frame.matrix) for n in graph.nodes}
    local_points = {n.id: node_points(n, graph, recipe_matrices) for n in graph.nodes}
    matrices, bounds = {}, {}
    report = recipe_report

    def refresh():
        bounds.clear()
        for identity in order:
            parent = by_id[identity].parent_id
            matrices[identity] = (matrices[parent] if parent else np.eye(4)) @ locals_[identity]
            pts = local_points[identity]
            if len(pts):
                box = envelope(trimesh.transform_points(pts, matrices[identity]))
                if box:
                    bounds[identity] = box
        # Only semantic groups without their own envelope derive bounds from children.
        for identity in reversed(order):
            parent = by_id[identity].parent_id
            if parent and not len(local_points[parent]) and identity in bounds:
                boxes = [bounds[identity]] + ([bounds[parent]] if parent in bounds else [])
                bounds[parent] = envelope(np.concatenate([corners(b) for b in boxes]))

    def anchor(node, name):
        p = next((a.position for a in node.spatial.anchors if a.name == name), None) if name else (0, 0, 0)
        if p is None:
            raise ValueError("missing anchor: " + str(name))
        return trimesh.transform_points([p], matrices[node.id])[0]

    originals = {k: v.copy() for k, v in locals_.items()}
    forcing = {}
    start = time.monotonic()
    refresh()
    for _ in range(policy.solver_iterations):
        movement = 0.0
        for node in graph.nodes:
            for c in node.relationships:
                if time.monotonic() - start > policy.solver_timeout:
                    break
                if c.kind not in {
                    "relative",
                    "anchor",
                    "attach",
                    "support",
                    "adjacent",
                    "orientation",
                    "contain",
                    "separate",
                    "keep_out",
                }:
                    continue
                target = by_id[c.target]
                a, b = bounds.get(node.id), bounds.get(c.target)
                try:
                    delta = np.zeros(3)
                    if c.kind in {"relative", "anchor", "attach"}:
                        delta = (
                            anchor(target, c.target_anchor)
                            + matrices[c.target][:3, :3] @ np.asarray(c.offset)
                            - anchor(node, c.anchor)
                        )
                    elif c.kind == "support" and a and b:
                        delta[2] = b.max[2] + c.distance - a.min[2]
                    elif c.kind == "adjacent" and a and b:
                        # Offset is the design-provided direction, never a compiler-chosen side.
                        direction = np.asarray(c.offset)
                        if not np.any(direction):
                            continue
                        axis = int(np.argmax(abs(direction)))
                        delta[axis] = (
                            (b.max[axis] + c.distance - a.min[axis])
                            if direction[axis] > 0
                            else (b.min[axis] - c.distance - a.max[axis])
                        )
                    elif c.kind == "contain" and a and b and not target.spatial.polygon:
                        # Project only when the declared envelope can contain the object.
                        if all(x <= y + policy.tolerance for x, y in zip(a.size, b.size)):
                            delta = np.maximum(0, np.asarray(b.min) - a.min) - np.maximum(0, np.asarray(a.max) - b.max)
                    elif c.kind in {"separate", "keep_out"} and a and b:
                        # A supplied direction controls which side to use. Without it,
                        # report a conflict rather than inventing a placement direction.
                        direction = np.asarray(c.offset)
                        if np.any(direction):
                            axis = int(np.argmax(abs(direction)))
                            if direction[axis] > 0:
                                delta[axis] = max(0, b.max[axis] + c.distance - a.min[axis])
                            else:
                                delta[axis] = min(0, b.min[axis] - c.distance - a.max[axis])
                    elif c.kind == "orientation":
                        from .models import Transform

                        desired = (
                            matrices[c.target][:3, :3]
                            @ np.asarray(Transform(contract_version=2, rotation=c.rotation).matrix)[:3, :3]
                        )
                        parent = matrices[node.parent_id][:3, :3] if node.parent_id else np.eye(3)
                        current = locals_[node.id][:3, :3].copy()
                        locals_[node.id][:3, :3] = np.linalg.solve(parent, desired) @ np.diag(node.spatial.frame.scale)
                        movement += float(np.max(abs(current - locals_[node.id][:3, :3])))
                    if np.any(delta):
                        parent = matrices[node.parent_id][:3, :3] if node.parent_id else np.eye(3)
                        locals_[node.id][:3, 3] += np.linalg.solve(parent, delta)
                        movement += float(np.linalg.norm(delta))
                    if np.any(delta) or c.kind == "orientation":
                        forcing[node.id] = c.id
                    refresh()
                except ValueError:
                    pass  # Report missing anchors in the final constraint pass.
        if movement < policy.tolerance or time.monotonic() - start > policy.solver_timeout:
            break
    refresh()
    for identity, original in originals.items():
        if not np.allclose(original, locals_[identity], atol=policy.tolerance):
            report.repairs.append(
                RepairRecord(
                    node=identity,
                    original=original.tolist(),
                    new=locals_[identity].tolist(),
                    reason="projected authored local frame onto declared spatial relationship",
                    constraint=forcing.get(identity, "parent_frame"),
                    intent_impact="authored placement changed; relationship takes precedence; final constraints checked separately",
                )
            )
    for node in graph.nodes:
        for c in node.relationships:
            a, b = bounds.get(node.id), bounds.get(c.target)
            approximate = False
            message = "constraint satisfied"
            valid = False
            try:
                if c.kind in {"relative", "anchor", "attach"}:
                    valid = (
                        np.linalg.norm(
                            anchor(by_id[c.target], c.target_anchor)
                            + matrices[c.target][:3, :3] @ np.asarray(c.offset)
                            - anchor(node, c.anchor)
                        )
                        <= policy.tolerance
                    )
                elif c.kind == "orientation":
                    from .models import Transform

                    expected = (
                        matrices[c.target][:3, :3]
                        @ np.asarray(Transform(contract_version=2, rotation=c.rotation).matrix)[:3, :3]
                    )
                    actual = matrices[node.id][:3, :3]
                    valid = np.allclose(
                        actual / np.linalg.norm(actual, axis=0),
                        expected / np.linalg.norm(expected, axis=0),
                        atol=policy.tolerance,
                    )
                elif c.kind in {"visible", "connect", "route"}:
                    p, q = anchor(node, c.anchor), anchor(by_id[c.target], c.target_anchor)
                    obstacles = [
                        v
                        for k, v in bounds.items()
                        if k not in {node.id, c.target} and by_id[k].recipe and by_id[k].count
                    ]
                    valid = not any(segment_hits_box(p, q, box, policy.tolerance) for box in obstacles)
                    approximate = True
                    message = "straight segment checked against broad-phase envelopes; curved routes and navigation not certified"
                elif a and b:
                    approximate = True
                    if c.kind == "contain":
                        pts = trimesh.transform_points(corners(a), np.linalg.inv(matrices[c.target]))
                        target = by_id[c.target]
                        valid = b.contains(a, policy.tolerance)
                        if target.spatial.polygon:
                            valid = valid and all(point_inside_polygon(p[:2], target.spatial.polygon) for p in pts)
                            message = "polygon containment sampled at envelope corners; concave edges not certified"
                    elif c.kind in {"separate", "keep_out"}:
                        gap = np.maximum(0, np.maximum(np.asarray(b.min) - a.max, np.asarray(a.min) - b.max))
                        valid = (
                            not a.intersects(b, policy.tolerance)
                            and np.linalg.norm(gap) + policy.tolerance >= c.distance
                        )
                    elif c.kind == "overlap":
                        valid = a.intersects(b, -policy.tolerance)
                    elif c.kind == "support":
                        valid = abs(a.min[2] - b.max[2] - c.distance) <= policy.tolerance and all(
                            min(a.max[k], b.max[k]) > max(a.min[k], b.min[k]) for k in (0, 1)
                        )
                    elif c.kind == "adjacent":
                        direction = np.asarray(c.offset)
                        if np.any(direction):
                            axis = int(np.argmax(abs(direction)))
                            gap = a.min[axis] - b.max[axis] if direction[axis] > 0 else b.min[axis] - a.max[axis]
                            valid = abs(gap - c.distance) <= policy.tolerance
                if not valid:
                    message = "unsatisfied " + c.kind + ": missing spatial data, conflict, or unavailable route"
            except ValueError as exc:
                message = str(exc)
            if not valid or approximate:
                report.diagnostics.append(
                    Diagnostic(
                        layer="support" if c.kind in {"support", "attach"} else "constraint",
                        code=c.kind,
                        nodes=[node.id, c.target],
                        constraint=c.id,
                        message=message,
                        severity="error" if not valid and c.required else "warning",
                        approximate=approximate,
                        relaxed=not valid and not c.required,
                        compromise="optional constraint left unsatisfied" if not valid and not c.required else None,
                    )
                )
    for identity, box in bounds.items():
        if max(abs(x) for x in (*box.min, *box.max)) > policy.max_extent:
            report.diagnostics.append(
                Diagnostic(layer="budget", code="extent", nodes=[identity], message="world extent exceeds policy")
            )
        if policy.site and not policy.site.contains(box, policy.tolerance):
            report.diagnostics.append(
                Diagnostic(layer="spatial", code="site", nodes=[identity], message="derived bounds exceed user site")
            )
    # Keep-out regions apply even without pairwise constraints; no geometry is manufactured for them.
    for node in graph.nodes:
        if node.kind != "negative_space" or node.id not in bounds:
            continue
        for other in graph.nodes:
            if other.recipe and other.count and other.id in bounds and bounds[node.id].intersects(bounds[other.id]):
                report.diagnostics.append(
                    Diagnostic(
                        layer="spatial",
                        code="negative_space",
                        nodes=[node.id, other.id],
                        message="geometry envelope intersects explicit negative space",
                        approximate=True,
                    )
                )
    # Distinct visual nodes require explicit permission to overlap. Assemblies
    # remain intentional local unions; no automatic placement repair is applied.
    solids = [n for n in graph.nodes if n.recipe and n.count and n.id in bounds] if check_collisions else []
    for index, a in enumerate(solids):
        for b in solids[index + 1 :]:
            intentional = any(c.target == b.id and c.kind in {"overlap", "attach", "contain"} for c in a.relationships)
            intentional |= any(c.target == a.id and c.kind in {"overlap", "attach", "contain"} for c in b.relationships)
            if not intentional and bounds[a.id].intersects(bounds[b.id], policy.tolerance):
                report.diagnostics.append(
                    Diagnostic(
                        layer="spatial",
                        code="collision",
                        nodes=[a.id, b.id],
                        message="visual envelopes intersect; declare intentional overlap or separate the design",
                        approximate=True,
                    )
                )
    import networkx as nx

    connectivity = nx.Graph()
    connectivity.add_nodes_from(by_id)
    connectivity.add_edges_from(
        (n.id, c.target) for n in graph.nodes for c in n.relationships if c.kind in {"connect", "route"}
    )
    for a, b in policy.required_reachable:
        if a not in connectivity or b not in connectivity or not nx.has_path(connectivity, a, b):
            report.diagnostics.append(
                Diagnostic(layer="spatial", code="reachability", nodes=[a, b], message="required connection is absent")
            )
    if policy.minimum_route_width:
        for node in graph.nodes:
            if node.kind in {"path", "portal"} and node.properties.get("width", 0) < policy.minimum_route_width:
                report.diagnostics.append(
                    Diagnostic(
                        layer="spatial",
                        code="accessibility",
                        nodes=[node.id],
                        message="declared route width below user policy",
                    )
                )
    return ResolvedScene(
        matrices={k: v.tolist() for k, v in matrices.items()},
        bounds=bounds,
        recipe_matrices=recipe_matrices,
        report=report,
    )
