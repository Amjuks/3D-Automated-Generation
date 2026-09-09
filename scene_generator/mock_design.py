"""Offline compiler fixtures. Live semantic decisions always come from the LLM."""

import random

from .util import stable_seed


def default_brief(category, ordinal, seed):
    # Offline fixtures exercise the compiler, not semantic intelligence.
    rng = random.Random(seed)
    zones = [
        dict(
            name=f"{category} area {i + 1}",
            purpose=f"A distinct part of {category}.",
            enclosure="open" if i == 0 else "building",
            floors=1,
            width=10 + stable_seed(seed, i) % 5,
            depth=12,
            floor_height=4.2,
            ground=["stone", "wood"][i],
            wall="plaster",
            palette=[f"{category} focal element", f"{category} secondary element"],
        )
        for i in range(2)
    ]
    return dict(
        title=f"{category.title()} — {ordinal + 1}",
        concept=f"A varied interpretation of {category}.",
        family=category,
        composition=rng.choice(["campus", "linear", "staggered"]),
        environment="Site-specific surroundings",
        lighting="daylight",
        weather="Clear air",
        terrain="ground",
        population=0,
        population_story="No population requested in this offline fixture.",
        zones=zones,
    )
