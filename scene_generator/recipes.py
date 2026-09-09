"""Object-local, data-only geometry recipes for arbitrary semantic object names."""

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .models import Model
from .util import digest, read_json, stable_seed, write_json

Unit = Annotated[float, Field(ge=0, le=1)]
PositiveUnit = Annotated[float, Field(gt=0, le=1)]


class Part(Model):
    name: str = Field(max_length=70)
    primitive: Literal[
        "box", "cylinder", "sphere", "cone", "capsule", "lathe", "vase", "urn", "rock", "bust", "torus", "organic"
    ]
    position: tuple[Unit, Unit, Unit]
    size: tuple[PositiveUnit, PositiveUnit, PositiveUnit]
    color: tuple[Unit, Unit, Unit] = (0.5, 0.5, 0.5)
    roughness: float = Field(default=0.6, ge=0, le=1)
    metallic: float = Field(default=0, ge=0, le=1)
    emission: float = Field(default=0, ge=0, le=20)
    opacity: float = Field(default=1, ge=0, le=1)
    transmission: float = Field(default=0, ge=0, le=1)
    rotation: tuple[float, float, float] = (0, 0, 0)
    profile: list[tuple[Unit, Unit]] = Field(default_factory=list, max_length=32)

    @field_validator("primitive", mode="before")
    @classmethod
    def legacy_names(cls, value):
        # These old names actually generate a classical bust and an organic blob.
        return {"sculpture": "bust", "tree": "organic"}.get(value, value)

    @model_validator(mode="after")
    def contained(self):
        if any(p + s > 1.000001 for p, s in zip(self.position, self.size)):
            raise ValueError("Each position + size must be <= 1 inside the object allocation")
        if self.primitive == "lathe" and (
            len(self.profile) < 4
            or self.profile[0][0] != 0
            or self.profile[-1][0] != 0
            or self.profile[-1][1] <= self.profile[0][1]
            or max(r for r, _ in self.profile) == 0
            or any(b[1] < a[1] for a, b in zip(self.profile, self.profile[1:]))
        ):
            raise ValueError(
                "lathe profile needs >=4 radius/height points, increasing height, positive radius, and axis endpoints"
            )
        return self


class ObjectRecipe(Model):
    description: str = Field(max_length=400)
    parts: list[Part] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def grounded(self):
        if min(p.position[2] for p in self.parts) > 0.001:
            raise ValueError("At least one part must contact z=0")
        return self


def object_key(spec):
    return digest(
        {
            "version": 2,
            "name": spec.name,
            "kind": spec.kind,
            "detail": spec.detail,
            "material": spec.material,
            "size": [spec.width, spec.depth, spec.height],
        }
    )[:20]


def design_recipes(designs, llm, scene_id, path, log, assets, seed=0, brief=None, quality="standard"):
    recipes = {}
    acquired = {r.get("target_role") for r in assets if r["type"] == "model"}
    for zi, design in enumerate(designs):
        for oi, spec in enumerate(design.objects):
            if spec.count == 0:
                continue
            role = f"zone-{zi + 1:02d}/object-{oi + 1:02d}"
            if role in acquired:
                continue
            key = object_key(spec)
            if key in recipes:
                continue
            target = path / "object-recipes" / f"{key}.json"
            old_key = digest({"kind": spec.kind, "detail": spec.detail, "size": [spec.width, spec.depth, spec.height]})[
                :20
            ]
            legacy = path / "object-recipes" / f"{old_key}.json"
            checkpoint = target if target.exists() else legacy
            if checkpoint.exists():
                # Preserve already completed designs from earlier runs. New global
                # cache entries include material, creative context and the scene seed.
                recipe = ObjectRecipe.model_validate(read_json(checkpoint))
            else:
                log.event("design_object", scene=scene_id, role=role, object=spec.kind)
                recipe = llm.request(
                    ObjectRecipe,
                    {
                        "task": "Model only this one object as a recognizable, detailed assembly.",
                        "object": spec.model_dump(
                            mode="json", include={"name", "kind", "material", "detail", "width", "depth", "height"}
                        ),
                        "seed": stable_seed(seed, key),
                        "quality": quality,
                        "concept": brief.concept if brief else design.description,
                        "instruction": "Return normalized parts inside [0,1]^3, Z up. Match detail to the requested quality and actual complexity; do not reduce a complex object to a single primitive. Use the parts needed for a recognizable silhouette, supports, joints and purposeful detail; 24 is a ceiling, not a target. Omit default properties to keep JSON short. Position is the lower corner; position+size <=1 on every axis. Rotation is degrees, baked inside each allocation. Overlap inside this assembly is allowed. At least one part touches z=0. Honor the requested material and use coherent colors. Lathe accepts radius/height profile points with nondecreasing height and axis endpoints; bust means a human bust, organic is a perturbed ellipsoid, and other primitives are geometric building blocks. Compose arbitrary objects from these shapes. Do not generate code.",
                    },
                    scene_id,
                    role,
                    mock=lambda: dict(
                        description="Offline placeholder assembly",
                        parts=[
                            dict(
                                name="base",
                                primitive="box",
                                position=[0, 0, 0],
                                size=[1, 1, 0.1],
                                color=[0.3, 0.2, 0.1],
                            ),
                            dict(
                                name="body",
                                primitive="sphere",
                                position=[0.1, 0.1, 0.1],
                                size=[0.8, 0.8, 0.9],
                                color=[0.4, 0.6, 0.3],
                            ),
                        ],
                    ),
                )
                write_json(target, recipe.model_dump(mode="json"))
            recipes[key] = recipe
    return recipes
