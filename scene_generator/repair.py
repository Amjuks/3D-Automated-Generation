"""Bounded repairs update failed components and explicitly invalidate dependents."""

from .models import Component
from .util import read_json


def repair_components(nodes, report, paths):
    by_id = {n.id: n for n in nodes}
    failures = {}
    for issue in report.issues:
        if issue.severity == "error":
            failures.setdefault(issue.component_id, set()).add(issue.code)
    changed = set()
    for identity, codes in failures.items():
        if identity not in paths:
            continue
        original = Component.model_validate(read_json(paths[identity] / "contract.json"))
        # Parent allocations are authoritative; restore corrupted or drifting specs first.
        if any(
            code in codes
            for code in {"bounds", "transform", "floating", "attachment", "clearance", "collision", "mount"}
        ):
            by_id[identity] = original
            changed.add(identity)
        if codes & {"geometry", "mesh_bounds", "manifold", "degenerate", "uv", "polygon_budget", "material"}:
            replacement = original.model_copy(deep=True)
            if replacement.generator == "asset":
                replacement.generator = "box"
                replacement.parameters = {"repair": "invalid asset replaced by bounding-box placeholder"}
            replacement.budget.detail = "draft"
            by_id[identity] = replacement
            changed.add(identity)
        if "texture_budget" in codes:
            for node in nodes:
                if not any(
                    texture
                    for material in node.materials
                    for texture in (
                        material.base_color_texture,
                        material.roughness_texture,
                        material.normal_texture,
                    )
                ):
                    continue
                replacement = node.model_copy(deep=True)
                replacement.materials = [
                    material.model_copy(
                        update={
                            "base_color_texture": None,
                            "roughness_texture": None,
                            "normal_texture": None,
                        }
                    )
                    for material in replacement.materials
                ]
                by_id[node.id] = replacement
                changed.add(node.id)
    # Only transitive dependents can be affected by a changed contract.
    invalidated = set(changed)
    while True:
        dependents = {n.id for n in nodes if any(d in invalidated for d in n.dependencies)}
        added = dependents - invalidated
        if not added:
            break
        invalidated.update(added)
    return [by_id[n.id] for n in nodes], invalidated
