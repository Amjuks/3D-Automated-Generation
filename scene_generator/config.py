from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from .models import Bounds, Model, Vec3
from .policy import ScenePolicy


class Variations(Model):
    architectural_styles: list[str] = Field(default_factory=list)
    environments: list[str] = Field(default_factory=list)

    @field_validator("architectural_styles", "environments")
    @classmethod
    def nonempty(cls, value):
        if any(not x.strip() or len(x) > 100 for x in value):
            raise ValueError("variation lists must contain nonempty short strings")
        return value


class LLMConfig(Model):
    mode: Literal["live", "mock"] = "live"
    response_format: Literal["json_schema", "json_object", "text"] = "json_object"
    temperature: float = Field(default=0.2, ge=0, le=2)
    stream: bool = True
    stream_usage: bool = True
    concurrency: int = Field(default=4, ge=1, le=32)
    requests_per_minute: float = Field(default=30, gt=0)
    retries: int = Field(default=3, ge=0, le=10)
    timeout: float = Field(default=60, gt=0)
    connect_timeout: float = Field(default=10, gt=0)
    write_timeout: float = Field(default=30, gt=0)
    pool_timeout: float = Field(default=5, gt=0)
    total_timeout: float = Field(default=600, gt=0)
    max_response_bytes: int = Field(default=2_000_000, ge=1024)
    max_tokens: int | None = Field(default=4000, ge=256, le=16000)
    token_limit_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)
    cached_input_cost_per_million: float | None = Field(default=None, ge=0)


class AssetConfig(Model):
    mode: Literal["mock", "local", "polyhaven"] = "mock"
    cache_dir: Path = Path(".cache/scene-generator/assets")
    local_catalog: Path | None = None
    offline: bool = False
    max_download_bytes: int = Field(default=50_000_000, gt=0)
    max_assets: int = Field(default=16, ge=0, le=100)
    texture_resolution: Literal["1k", "2k", "4k"] = "2k"
    max_total_bytes: int = Field(default=400_000_000, gt=0)
    required: bool = False


class OutputConfig(Model):
    format: Literal["glb", "ply", "obj"] = "glb"
    root: Path = Path("runs")
    component_glbs: bool = False
    preview: bool = False
    require_blender: bool = False
    name: str | None = Field(default=None, min_length=1, max_length=80)
    preview_samples: int = Field(default=64, ge=8, le=1024)
    preview_width: int = Field(default=1440, ge=320, le=3840)


class Generation(Model):
    workflow: Literal["graph", "creative", "legacy"] = "graph"
    policy: ScenePolicy = Field(default_factory=ScenePolicy)
    variations: Variations = Field(default_factory=Variations)
    frameworks: list[Literal["blender", "trimesh"]] = Field(default_factory=lambda: ["blender", "trimesh"])
    output: OutputConfig = Field(default_factory=OutputConfig)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    quality: Literal["draft", "standard", "high"] = "standard"
    dimensions: Vec3 | None = None
    bounding_space: Bounds | None = None
    llm: LLMConfig = Field(default_factory=LLMConfig)
    assets: AssetConfig = Field(default_factory=AssetConfig)
    workers: int = Field(default=4, ge=1, le=32)
    max_scene_triangles: int = Field(default=2_000_000, ge=1000)
    max_texture_bytes: int = Field(default=256_000_000, ge=0)
    repair_attempts: int = Field(default=2, ge=0, le=5)
    blender_timeout: float = Field(default=600, gt=0)

    @field_validator("frameworks")
    @classmethod
    def backends(cls, value):
        if not value or len(set(value)) != len(value):
            raise ValueError("provide at least one backend without duplicates")
        return value

    @field_validator("dimensions")
    @classmethod
    def dimensions_valid(cls, value):
        if value is not None and any(x <= 0 for x in value):
            raise ValueError("dimensions must be positive meters")
        return value


class Config(Model):
    scenes: dict[str, int]
    scene_descriptions: dict[str, str] = Field(default_factory=dict)
    generation: Generation = Field(default_factory=Generation)

    @model_validator(mode="after")
    def descriptions_valid(self):
        if any(k not in self.scenes or not v.strip() or len(v) > 4000 for k, v in self.scene_descriptions.items()):
            raise ValueError("scene_descriptions must reference batch categories and contain 1-4000 characters")
        return self

    @field_validator("scenes", mode="before")
    @classmethod
    def categories(cls, value):
        if not isinstance(value, dict) or not value:
            raise ValueError("scenes must be a nonempty category-to-count mapping")
        for key, count in value.items():
            if not isinstance(key, str) or not key.strip() or len(key) > 60:
                raise ValueError("category must be a short nonempty string")
            if type(count) is not int or not 1 <= count <= 10000:
                raise ValueError("scene counts must be integers from 1 to 10000")
        return value


def load_config(path: Path, settings: Path | None = None) -> Config:
    try:
        value = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        raise ValueError("invalid YAML configuration") from None
    # A plain category/count mapping is shorthand for the existing scenes block.
    if isinstance(value, dict) and "scenes" not in value and "generation" not in value:
        value = {"scenes": value}
    config = Config.model_validate(value)
    if settings is not None:
        # Reuse a complete existing configuration as a generation profile.
        # Its scene counts and run name belong to the source batch, not this one.
        config.generation = load_config(settings).generation.model_copy(deep=True)
        config.generation.output.name = None
    return config
