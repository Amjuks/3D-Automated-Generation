"""Technical limits and user constraints, separate from creative decisions."""

from typing import Literal

from pydantic import Field

from .models import Bounds, Model


class ScenePolicy(Model):
    schema_version: Literal[2] = 2
    site: Bounds | None = None
    max_nodes: int = Field(default=2000, ge=1)
    max_parts: int = Field(default=10000, ge=1)
    max_triangles: int = Field(default=2_000_000, ge=1)
    max_texture_bytes: int = Field(default=256_000_000, ge=0)
    max_memory_bytes: int = Field(default=1_000_000_000, ge=1024)
    max_checkpoint_bytes: int = Field(default=8_000_000, ge=1024)
    max_component_triangles: int = Field(default=100_000, ge=12)
    max_extent: float = Field(default=1_000_000, gt=0)
    tolerance: float = Field(default=0.001, gt=0)
    solver_iterations: int = Field(default=100, ge=1, le=1000)
    solver_timeout: float = Field(default=30, gt=0)
    allowed_operations: list[str] = Field(
        default_factory=lambda: ["box", "sphere", "cylinder", "cone", "capsule", "torus", "mesh"]
    )
    required_reachable: list[tuple[str, str]] = Field(default_factory=list)
    minimum_route_width: float | None = Field(default=None, gt=0)
    asset_availability: list[str] = Field(default_factory=list)

    @classmethod
    def from_generation(cls, generation):
        data = generation.policy.model_dump(exclude_unset=True)
        data.setdefault("max_triangles", generation.max_scene_triangles)
        data.setdefault("max_texture_bytes", generation.max_texture_bytes)
        if generation.bounding_space:
            data.setdefault("site", generation.bounding_space)
        elif generation.dimensions:
            data.setdefault("site", Bounds(max=generation.dimensions))
        return cls.model_validate(data)
