"""Object-local, data-only geometry recipes for arbitrary semantic object names."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .models import Model
from .util import digest, read_json, write_json

Unit = Annotated[float, Field(ge=0, le=1)]
PositiveUnit = Annotated[float, Field(gt=0, le=1)]


class Part(Model):
    name: str = Field(max_length=70)
    primitive: Literal["box", "cylinder", "sphere", "vase", "urn", "rock", "sculpture", "torus", "tree"]
    position: tuple[Unit, Unit, Unit]
    size: tuple[PositiveUnit, PositiveUnit, PositiveUnit]
    color: tuple[Unit, Unit, Unit] = (0.5, 0.5, 0.5)
    roughness: float = Field(default=0.6, ge=0, le=1)
    metallic: float = Field(default=0, ge=0, le=1)
    emission: float = Field(default=0, ge=0, le=20)
    opacity: float = Field(default=1, ge=0, le=1)
    transmission: float = Field(default=0, ge=0, le=1)
    rotation: tuple[float, float, float] = (0, 0, 0)

    @model_validator(mode="after")
    def contained(self):
        if any(p + s > 1.000001 for p, s in zip(self.position, self.size)):
            raise ValueError("Each position + size must be <= 1 inside the object allocation")
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
    return digest({"kind": spec.kind, "detail": spec.detail, "size": [spec.width, spec.depth, spec.height]})[:20]


def design_recipes(designs, llm, scene_id, path, log, assets):
    recipes = {}
    acquired = {r.get("target_role") for r in assets if r["type"] == "model"}
    for zi, design in enumerate(designs):
        for oi, spec in enumerate(design.objects):
            role = f"zone-{zi + 1:02d}/object-{oi + 1:02d}"
            if role in acquired:
                continue
            key = object_key(spec)
            if key in recipes:
                continue
            target = path / "object-recipes" / f"{key}.json"
            if target.exists():
                recipe = ObjectRecipe.model_validate(read_json(target))
            else:
                log.event("design_object", scene=scene_id, role=role, object=spec.kind)
                recipe = llm.request(
                    ObjectRecipe,
                    {
                        "task": "Model only this one object as a recognizable, detailed assembly.",
                        "object": spec.model_dump(mode="json"),
                        "instruction": "Return normalized parts inside [0,1]^3, Z up. Use silhouette, supports, joints and small details. Position is the lower corner and size is an allocation; position+size <=1 on every axis. Rotation is degrees, baked inside each allocation. Overlapping parts within this assembly are allowed. At least one part touches z=0. Use multiple natural material colors. Use emission for genuinely luminous parts, otherwise zero. Parts become real geometry; the description alone does not. Do not generate code.",
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
