"""Shared parent-owned allocations, independent of scene semantics."""

from .models import Bounds, Budget, Component, Transform
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
