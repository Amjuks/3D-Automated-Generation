"""Data-only graph realization; no category dispatch and no implicit architecture."""

import numpy as np

from .capabilities import GEOMETRY
from .design import Diagnostic, GraphReport
from .models import Bounds, Budget, Component, Transform
from .util import digest, stable_seed


def compile_graph(graph, resolved, policy, scene_id, seed, selected=()):
    report = GraphReport.model_validate(resolved.report.model_dump())
    nodes, coverage = [], []
    if not resolved.matrices and not report.valid:
        return nodes, coverage, report
    candidates = {a.get("target_role"): a for a in selected}
    total_estimate = 0

    def diagnostic(node, code, message, layer="design", severity="error"):
        report.diagnostics.append(
            Diagnostic(layer=layer, code=code, nodes=[node.id], message=message, severity=severity)
        )

    for node in graph.nodes:
        if not node.count:
            coverage.append({"node": node.id, "source": "zero_count"})
            continue
        if node.kind in {"camera", "light", "atmosphere"}:
            continue
        recipe = graph.recipes.get(node.recipe)
        request = graph.assets.get(node.asset)
        if node.recipe and not recipe:
            diagnostic(node, "recipe_reference", "missing recipe " + node.recipe)
        if node.asset and not request:
            diagnostic(node, "asset_reference", "missing asset request " + node.asset)
        candidate = candidates.get(node.id) if request else None
        fallback_reason = None
        if request:
            allowed = {s.lower().replace("-1.0", "") for s in request.licenses}
            if candidate and candidate["license"].lower().replace("-1.0", "") not in allowed:
                candidate = None
                fallback_reason = "candidate license does not match requested licensing"
            if not candidate:
                fallback_reason = fallback_reason or "asset search found no usable candidate"
                if request.fallback == "error":
                    diagnostic(node, "asset_miss", fallback_reason, "quality")
                    continue
                if request.fallback == "omit":
                    coverage.append({"node": node.id, "source": "omitted", "reason": fallback_reason})
                    diagnostic(node, "omission", fallback_reason, "quality", "warning")
                    continue
                if not recipe:
                    diagnostic(node, "fallback_recipe", "explicit recipe fallback is missing")
                    continue
        if not recipe and not candidate:
            if node.kind in {"object", "surface", "terrain", "structure"} and not any(
                n.parent_id == node.id for n in graph.nodes
            ):
                diagnostic(node, "unrealized", "visual leaf has no asset or explicit geometry recipe", "quality")
            continue
        if candidate:
            from .design import Operation

            if not node.spatial.envelope or request.material not in graph.materials:
                diagnostic(
                    node,
                    "asset_dimensions",
                    "asset realization needs an explicit envelope and asset request material ID",
                )
                continue
            parts = [
                Operation(
                    capability="asset",
                    size=node.spatial.envelope.size,
                    material=request.material,
                    frame=Transform(contract_version=2, translation=node.spatial.envelope.min),
                    parameters={"path": candidate["path"], "sha256": candidate["sha256"]},
                )
            ]
        else:
            parts = recipe.parts
            if not parts:
                diagnostic(node, "empty_recipe", "visual recipe contains no parts")
        coverage.append(
            {
                "node": node.id,
                "source": "asset" if candidate else "recipe",
                "fallback_reason": fallback_reason,
                "asset": candidate,
                "recipe": node.recipe,
                "conversion": "existing importer; uniform fit" if candidate else None,
                "scaling": (
                    min(a / b for a, b in zip(node.spatial.envelope.size, candidate["source_extents"]))
                    if candidate and candidate.get("source_extents")
                    else None
                ),
            }
        )
        for instance_index, instance in enumerate(node.instances or [Transform(contract_version=2)]):
            for index, part in enumerate(parts):
                if len(nodes) >= policy.max_parts:
                    diagnostic(node, "part_budget", "expanded part count exceeds policy", "budget")
                    return nodes, coverage, report
                if part.material not in graph.materials:
                    diagnostic(node, "material_reference", "missing material " + part.material)
                    continue
                try:
                    if part.capability == "asset" and candidate:
                        estimate = candidate["triangles"]
                    elif part.capability not in GEOMETRY or part.capability not in policy.allowed_operations:
                        diagnostic(
                            node, "unsupported_operation", "unsupported operation " + part.capability, "capability"
                        )
                        continue
                    else:
                        estimate = GEOMETRY[part.capability].validate(part, policy)
                        if part.capability == "box" and graph.materials[part.material].bevel > 0:
                            estimate = 44
                    if estimate > policy.max_component_triangles:
                        raise ValueError("component triangle estimate exceeds policy")
                except (ValueError, TypeError) as exc:
                    diagnostic(node, "operation", str(exc), "capability")
                    continue
                total_estimate += estimate
                if total_estimate > policy.max_triangles or total_estimate * 240 > policy.max_memory_bytes:
                    diagnostic(
                        node, "preflight_budget", "estimated geometry exceeds triangle or memory policy", "budget"
                    )
                    return nodes, coverage, report
                frame = (
                    resolved.recipe_matrices.get(node.recipe, {}).get(part.id, part.frame.matrix)
                    if not candidate
                    else part.frame.matrix
                )
                matrix = np.asarray(resolved.matrices[node.id]) @ instance.matrix @ frame
                identity = scene_id + "/graph/" + digest(node.id)[:24] + f"/i{instance_index}/p{index}"
                material = graph.materials[part.material].model_copy(deep=True)
                if candidate and candidate.get("base_color_texture"):
                    material.base_color_texture = candidate["base_color_texture"]
                nodes.append(
                    Component(
                        contract_version=2,
                        id=identity,
                        design_node_id=node.id,
                        provenance=node.provenance.model_dump(),
                        name=node.description,
                        kind=node.kind,
                        bounds=Bounds(max=part.size),
                        local_transform=Transform(contract_version=2, affine_matrix=matrix.tolist()),
                        world_transform=Transform(contract_version=2, affine_matrix=matrix.tolist()),
                        materials=[material],
                        generator=part.capability,
                        parameters=part.parameters,
                        budget=Budget(
                            triangles=policy.max_component_triangles, detail=recipe.detail if recipe else "standard"
                        ),
                        seed=stable_seed(seed, identity),
                    )
                )
    report.stats["estimated_triangles"] = total_estimate
    return nodes, coverage, report


def validate_geometry(nodes, geometry, policy, report):
    import trimesh

    triangles = memory = 0
    texture_paths = set()
    for node in nodes:

        def error(code, message, layer="geometry"):
            report.diagnostics.append(Diagnostic(layer=layer, code=code, nodes=[node.design_node_id], message=message))

        try:
            arrays = geometry(node)
            v, f, uv = arrays["vertices"], arrays["faces"], arrays["uv"]
            memory += sum(a.nbytes for a in arrays.values())
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
                raise ValueError("invalid triangles")
            if uv.shape != (len(v), 2) or not np.isfinite(uv).all():
                raise ValueError("invalid UV coordinates")
            if (v < -policy.tolerance).any() or (v > np.array(node.bounds.size) + policy.tolerance).any():
                raise ValueError("mesh exceeds recipe envelope")
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
            if not mesh.is_watertight or not mesh.is_winding_consistent or (mesh.area_faces < 1e-12).any():
                raise ValueError("open, degenerate or inconsistently wound geometry")
            triangles += len(f)
            if len(f) > node.budget.triangles:
                error("component_triangles", "component exceeds triangle policy", "budget")
            for m in node.materials:
                texture_paths.update(p for p in (m.base_color_texture, m.normal_texture, m.roughness_texture) if p)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            error("mesh", str(exc))
    from pathlib import Path

    texture_bytes = 0
    for p in texture_paths:
        if not Path(p).is_file():
            report.diagnostics.append(Diagnostic(layer="geometry", code="texture", message="missing texture"))
        else:
            texture_bytes += Path(p).stat().st_size
    for code, actual, maximum in [
        ("triangles", triangles, policy.max_triangles),
        ("memory", memory, policy.max_memory_bytes),
        ("texture_bytes", texture_bytes, policy.max_texture_bytes),
    ]:
        if actual > maximum:
            report.diagnostics.append(Diagnostic(layer="budget", code=code, message="resource policy exceeded"))
    report.stats.update(triangles=triangles, memory_bytes=memory, texture_bytes=texture_bytes, components=len(nodes))
    return report
