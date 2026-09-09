"""Schemas for existing template-era checkpoints."""

import math
from typing import Literal

from pydantic import Field, model_validator

from ..models import Model


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
