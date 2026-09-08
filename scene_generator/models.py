from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Vec3 = tuple[float, float, float]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


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


class GalleryBrief(Model):
    name: str = Field(max_length=70)
    theme: Literal["antiquities", "sculpture", "paintings", "natural_history", "ceramics", "contemporary"]
    width: float = Field(default=1, ge=0.75, le=1.5)
    depth: float = Field(default=1, ge=0.7, le=1.6)
    height: float = Field(default=1, ge=0.8, le=1.35)
    roof: Literal["vault", "skylight", "coffered"] = "vault"
    finish: Literal["limestone", "plaster", "concrete", "timber"] = "plaster"


class ScenePlan(Model):
    title: str = Field(max_length=120)
    category: str = Field(max_length=60)
    style: str = Field(max_length=60)
    environment: str = Field(max_length=60)
    collection: Literal["sculpture", "antiquities", "science", "art", "natural_history", "residential"]
    atmosphere: Literal["warm", "cool", "dramatic", "daylight"]
    columns: int = Field(ge=1, le=12)
    rows: int = Field(ge=1, le=2)
    room_width: float = Field(ge=5, le=20)
    room_depth: float = Field(ge=5, le=20)
    height: float = Field(ge=3, le=9)
    accent: tuple[float, float, float]
    layout: Literal["legacy", "courtyard", "pavilions", "enfilade"] = "legacy"
    concept: str = Field(default="", max_length=450)
    galleries: list[GalleryBrief] = Field(default_factory=list, max_length=6)
    floor_finish: Literal["limestone", "terrazzo", "parquet"] = "limestone"
    age: Literal["historic", "restored", "contemporary"] = "restored"

    @model_validator(mode="after")
    def check_accent(self):
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in self.accent):
            raise ValueError("accent must be RGB in [0,1]")
        return self


class RoomDesign(Model):
    furnishing: Literal["gallery", "library", "lounge", "study", "dining"]
    centerpiece: Literal["vase", "sphere", "cylinder"]
    density: int = Field(ge=1, le=4)
    wall_art: bool
    arrangement: Literal["curated", "clustered", "sparse"] = "curated"
    exhibit_family: Literal["ceramics", "sculpture", "paintings", "natural_history", "mixed"] = "mixed"
    lighting: Literal["warm_spots", "soft_daylight", "dramatic"] = "warm_spots"
    label: str = Field(default="Selected works", max_length=80)


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
