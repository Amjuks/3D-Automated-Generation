"""Asset search defaults for saved legacy plans only."""


def asset_requests(plan):
    queries = [
        ("texture", "floor parquet wood" if plan.atmosphere in {"warm", "daylight"} else "floor marble stone"),
        (
            "hdri",
            {
                "urban": "city street outdoor",
                "forest": "forest woods outdoor",
                "coastal": "beach coast outdoor",
            }.get(plan.environment, plan.environment),
        ),
        (
            "model",
            {
                "antiquities": "vase pottery ancient",
                "science": "globe instrument apparatus",
                "natural_history": "rock fossil specimen",
            }.get(plan.collection, "sculpture statue bust vase"),
        ),
    ]
    return [{"type": kind, "query": query, "target_material": plan.floor_finish} for kind, query in queries]


def apply_models(nodes, records):
    models = [r for r in records if r["type"] == "model"]
    used_models = set()
    for node in nodes:
        if node.name == "collection-object" and node.parameters.get("asset_candidate", True):
            candidates = [r for r in models if r["id"] not in used_models]
            if not candidates:
                continue
            selected = candidates[0]
            used_models.add(selected["id"])
            node.generator = "asset"
            node.parameters = {"path": selected["path"], "sha256": selected["sha256"], "asset_id": selected["id"]}
            node.budget.triangles = max(node.budget.triangles, selected.get("triangles", 2000))
            if selected.get("base_color_texture"):
                from ..models import Material

                node.materials = [
                    Material(
                        name="asset-" + selected["id"],
                        color=(1, 1, 1, 1),
                        base_color_texture=selected["base_color_texture"],
                    )
                ]
