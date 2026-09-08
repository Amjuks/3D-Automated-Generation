import random

from .models import ScenePlan


def gallery_briefs(seed, style, count=4):
    rng = random.Random(seed)
    themes = ["antiquities", "sculpture", "paintings", "ceramics", "natural_history", "contemporary"]
    rng.shuffle(themes)
    names = {
        "antiquities": "Traces of the Ancient World",
        "sculpture": "The Human Form",
        "paintings": "Light and Landscape",
        "ceramics": "Earth and Fire",
        "natural_history": "Deep Time",
        "contemporary": "Material Dialogues",
    }
    return [
        {
            "name": names[t],
            "theme": t,
            "width": round(rng.uniform(0.8, 1.3), 2),
            "depth": round(rng.uniform(0.8, 1.4), 2),
            "height": round(rng.uniform(0.85, 1.2), 2),
            "roof": ["vault", "skylight", "coffered"][i % 3],
            "finish": "concrete" if style == "brutalist" else ["plaster", "limestone", "timber"][i % 3],
        }
        for i, t in enumerate(themes[:count])
    ]


def plan_scene(category, ordinal, seed, generation, llm, scene_id):
    styles = generation.variations.architectural_styles
    environments = generation.variations.environments
    style = styles[ordinal % len(styles)]
    environment = environments[(ordinal // len(styles)) % len(environments)]
    rng = random.Random(seed)
    small = category.lower() in {"room", "studio", "small_room"}
    museum = "museum" in category.lower() or category.lower() in {"gallery", "art_center"}
    residential = category.lower() in {"mansion", "house", "room", "studio", "small_room"}

    def mock():
        return dict(
            title=f"{style.title()} {category.title()} {ordinal + 1}",
            category=category,
            style=style,
            environment=environment,
            collection="residential"
            if residential
            else ["sculpture", "antiquities", "science", "art", "natural_history"][ordinal % 5],
            atmosphere=["warm", "cool", "dramatic", "daylight"][ordinal % 4],
            columns=1 if small else 2 + ordinal % 4,
            rows=1 if small else 1 + (ordinal // 2) % 2,
            room_width=round(rng.uniform(6.5, 10.5), 2),
            room_depth=round(rng.uniform(7, 11), 2),
            height=3.4 if residential else round(rng.uniform(4.2, 6), 2),
            accent=[round(rng.uniform(0.15, 0.65), 3) for _ in range(3)],
            layout=["courtyard", "pavilions", "enfilade"][ordinal % 3] if museum else "legacy",
            concept="A sequence of distinct galleries around a planted court, with warm mineral surfaces, daylight and a layered collection."
            if museum
            else "",
            galleries=gallery_briefs(seed, style, 4 if ordinal % 3 == 0 else 3 + ordinal % 3) if museum else [],
            floor_finish=["limestone", "parquet", "terrazzo"][ordinal % 3],
            age="restored" if style == "neoclassical" else "contemporary",
        )

    result = llm.request(
        ScenePlan,
        {
            "category": category,
            "variation": ordinal,
            "seed": seed,
            "style": style,
            "environment": environment,
            "dimensions": generation.dimensions,
            "bounding_space": generation.bounding_space.model_dump() if generation.bounding_space else None,
            "instruction": "Design a scene appropriate to the supplied category. Invent a named, site-specific concept. For museums choose courtyard (four galleries around a planted cloister), pavilions (staggered unequal wings linked by a glazed promenade), or enfilade (an asymmetric sequence of unequal galleries with a long public arcade). Never choose legacy for a museum. Supply exactly four distinct gallery briefs for courtyard, or three to six for other museum layouts, each with a different theme, width/depth/height ratio, roof and wall finish. Ratios influence actual built geometry. Avoid identical rooms or collections. Give each gallery a poetic but informative title. Use historic/restored/contemporary surfaces appropriate to the style. For all non-museum categories use legacy with an empty galleries list. Use provided style and environment. Consider real circulation, atmospheric lighting, material aging and diverse exhibit scales.",
        },
        scene_id,
        mock=mock,
    )
    # Category and selected variation are user contracts, not model suggestions.
    result = result.model_copy(update={"category": category, "style": style, "environment": environment})
    if museum and (result.layout == "legacy" or not result.galleries):
        result = result.model_copy(
            update={
                "layout": ["courtyard", "pavilions", "enfilade"][ordinal % 3],
                "galleries": ScenePlan.model_validate(mock()).galleries,
            }
        )
    if result.layout == "courtyard" and len(result.galleries) != 4:
        from .models import GalleryBrief

        result.galleries = (result.galleries + [GalleryBrief.model_validate(g) for g in gallery_briefs(seed, style)])[
            :4
        ]
    # Fit before any decomposition. Never shrink furniture/door ergonomics after allocation.
    available = generation.dimensions
    if generation.bounding_space:
        available = (
            tuple(min(a, b) for a, b in zip(available, generation.bounding_space.size))
            if available
            else generation.bounding_space.size
        )
    if available:
        w, d, h = available
        columns, rows = result.columns, result.rows
        while columns > 1 and (w - 6) / columns < 5:
            columns -= 1
        while rows > 1 and (d - 9) / rows < 5:
            rows -= 1
        result = ScenePlan.model_validate(
            {
                **result.model_dump(),
                "columns": columns,
                "rows": rows,
                "room_width": min(result.room_width, (w - 6) / columns),
                "room_depth": min(result.room_depth, (d - 9) / rows),
                "height": min(result.height, h - 0.5),
            }
        )
    return result
