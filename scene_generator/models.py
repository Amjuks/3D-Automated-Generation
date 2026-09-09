from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Vec3 = tuple[float, float, float]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class Bounds(Model):
    """Axis-aligned box in meters, Z up; maximum faces are inclusive."""

    min: Vec3 = (0, 0, 0)
    max: Vec3

    @model_validator(mode="after")
    def ordered(self):
        if any(b <= a for a, b in zip(self.min, self.max)):
            raise ValueError("bounds must have strictly positive extent on all axes")
        return self

    @property
    def size(self) -> Vec3:
        return tuple(b - a for a, b in zip(self.min, self.max))

    def contains(self, other: Bounds, eps=1e-6):
        return all(a - eps <= c and d <= b + eps for a, b, c, d in zip(self.min, self.max, other.min, other.max))

    def translated(self, offset: Vec3):
        return Bounds(
            min=tuple(a + b for a, b in zip(self.min, offset)), max=tuple(a + b for a, b in zip(self.max, offset))
        )

    def intersects(self, other: Bounds, eps=1e-5):
        return all(min(b, d) - max(a, c) > eps for a, b, c, d in zip(self.min, self.max, other.min, other.max))


class Transform(Model):
    translation: Vec3 = (0, 0, 0)
    # Contract v1 deliberately permits translation only: all boxes are exact AABBs.
    rotation: Vec3 = (0, 0, 0)
    scale: Vec3 = (1, 1, 1)

    @model_validator(mode="after")
    def supported(self):
        if self.rotation != (0, 0, 0) or self.scale != (1, 1, 1):
            raise ValueError("contract v1 supports translation-only transforms")
        return self

    @property
    def matrix(self):
        x, y, z = self.translation
        return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]


class Material(Model):
    name: str
    color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1)
    roughness: float = Field(default=0.65, ge=0, le=1)
    metallic: float = Field(default=0, ge=0, le=1)
    base_color_texture: str | None = None
    roughness_texture: str | None = None
    normal_texture: str | None = None
    texture_scale: float = Field(default=1, gt=0, le=20)
    bevel: float = Field(default=0.006, ge=0, le=0.1)
    transmission: float = Field(default=0, ge=0, le=1)
    emission: float = Field(default=0, ge=0, le=20)

    @model_validator(mode="after")
    def color_range(self):
        if any(not 0 <= x <= 1 for x in self.color):
            raise ValueError("material color channels must lie in [0, 1]")
        return self


class Socket(Model):
    id: str
    position: Vec3
    normal: Vec3 = (0, 0, 1)
    kind: Literal["support", "door", "mount"] = "support"


class Budget(Model):
    triangles: int = Field(default=2000, ge=12)
    texture_bytes: int = Field(default=16_777_216, ge=0)
    detail: Literal["draft", "standard", "high"] = "standard"


class Component(Model):
    id: str
    name: str
    kind: str
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    bounds: Bounds  # Allocation in parent-local coordinates.
    local_transform: Transform
    world_transform: Transform
    sockets: list[Socket] = Field(default_factory=list)
    reserved: list[Bounds] = Field(default_factory=list)  # Component-local.
    clearance: list[Bounds] = Field(default_factory=list)
    materials: list[Material] = Field(default_factory=list)
    generator: Literal[
        "group",
        "box",
        "cylinder",
        "sphere",
        "vase",
        "asset",
        "vault",
        "arch",
        "urn",
        "rock",
        "sculpture",
        "torus",
        "tree",
        "organic",
        "bust",
        "cone",
        "capsule",
        "lathe",
    ] = "group"
    parameters: dict = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)
    seed: int
    validation_state: Literal["pending", "valid", "invalid", "repaired"] = "pending"
    support_id: str | None = None
    attachment_socket: str | None = None
    collidable: bool = True

    @property
    def world_bounds(self):
        return Bounds(max=self.bounds.size).translated(self.world_transform.translation)


class Issue(Model):
    component_id: str
    code: str
    message: str
    severity: Literal["error", "warning"] = "error"


class ValidationReport(Model):
    issues: list[Issue] = Field(default_factory=list)
    checks: dict[str, str] = Field(default_factory=dict)
    stats: dict = Field(default_factory=dict)

    @property
    def valid(self):
        return not any(i.severity == "error" for i in self.issues)


def __getattr__(name):
    # Older integrations may still import legacy contracts from this module.
    if name in {"GalleryBrief", "ScenePlan", "RoomDesign"}:
        from .legacy import models

        return getattr(models, name)
    raise AttributeError(name)
