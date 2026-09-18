"""Version 2 creative contracts. Semantic nodes never imply geometry or dimensions."""

from typing import Literal

from pydantic import Field, model_validator

from .models import Bounds, Material, Model, Transform, Vec3

ID = str


class Provenance(Model):
    source: Literal["user", "llm", "mock", "resolver", "migration"]
    request_id: str = ""
    explanation: str = ""


class Anchor(Model):
    name: str
    position: Vec3


class Constraint(Model):
    source: ID | None = None
    id: ID
    kind: Literal[
        "relative",
        "anchor",
        "adjacent",
        "contain",
        "separate",
        "overlap",
        "visible",
        "support",
        "attach",
        "connect",
        "route",
        "keep_out",
        "orientation",
    ]
    target: ID
    offset: Vec3 = (0, 0, 0)
    anchor: str | None = None
    target_anchor: str | None = None
    distance: float = Field(default=0, ge=0)
    rotation: Vec3 = (0, 0, 0)
    required: bool = True


class SpatialIntent(Model):
    frame: Transform = Field(default_factory=lambda: Transform(contract_version=2))
    anchors: list[Anchor] = Field(default_factory=list)
    # These are explicit design envelopes, never automatically supplied allocations.
    envelope: Bounds | None = None
    polygon: list[tuple[float, float]] = Field(default_factory=list)
    height: float | None = Field(default=None, gt=0)
    points: list[Vec3] = Field(default_factory=list)
    clearance: float = Field(default=0, ge=0)


class Operation(Model):
    id: ID | None = None
    capability: str
    size: Vec3
    frame: Transform = Field(default_factory=lambda: Transform(contract_version=2))
    material: ID
    parameters: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def positive(self):
        if any(x <= 0 for x in self.size):
            raise ValueError("operation dimensions must be positive")
        return self


class Assembly(Model):
    schema_version: Literal[2] = 2
    id: ID
    visual_intent: str
    parts: list[Operation] = Field(default_factory=list)
    relationships: list[Constraint] = Field(default_factory=list)
    detail: Literal["draft", "standard", "high"] = "standard"
    alternatives: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def part_references(self):
        for index, part in enumerate(self.parts):
            if part.id is None:
                part.id = f"part-{index}"
        ids = {part.id for part in self.parts}
        if len(ids) != len(self.parts):
            raise ValueError("duplicate recipe part IDs")
        for relationship in self.relationships:
            if relationship.source not in ids or relationship.target not in ids:
                raise ValueError("assembly relationships require source and target part IDs")
        return self


class AssetRequest(Model):
    schema_version: Literal[2] = 2
    id: ID
    identity: str
    role: str
    appearance: str
    material_intent: str
    material: ID | None = None
    placement_context: str
    fidelity: str
    query: str
    licenses: list[str] = Field(default_factory=lambda: ["CC0"])
    alternatives: list[str] = Field(default_factory=list)
    fallback: Literal["recipe", "omit", "error"] = "error"


class DesignNode(Model):
    contract_version: Literal[2] = 2
    revision: int = Field(default=1, ge=1)
    id: ID = Field(min_length=1, max_length=160)
    kind: Literal[
        "intent",
        "site",
        "region",
        "volume",
        "structure",
        "surface",
        "terrain",
        "portal",
        "path",
        "object",
        "group",
        "support",
        "attachment",
        "negative_space",
        "material",
        "camera",
        "light",
        "atmosphere",
    ]
    description: str
    parent_id: ID | None = None
    dependencies: list[ID] = Field(default_factory=list)
    relationships: list[Constraint] = Field(default_factory=list)
    provenance: Provenance
    spatial: SpatialIntent = Field(default_factory=SpatialIntent)
    recipe: ID | None = None
    asset: ID | None = None
    count: int = Field(default=1, ge=0)
    # Repetition has explicit offsets; the compiler never invents a grid.
    instances: list[Transform] = Field(default_factory=list)
    properties: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def explicit_instances(self):
        if self.count > 1 and len(self.instances) != self.count:
            raise ValueError("repetitions require one explicit local frame per instance")
        return self


class DesignGraph(Model):
    schema_version: Literal[2] = 2
    id: ID
    revision: int = Field(default=1, ge=1)
    title: str
    intent: str
    nodes: list[DesignNode] = Field(default_factory=list)
    materials: dict[str, Material] = Field(default_factory=dict)
    recipes: dict[str, Assembly] = Field(default_factory=dict)
    assets: dict[str, AssetRequest] = Field(default_factory=dict)

    @model_validator(mode="after")
    def references(self):
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate design node IDs")
        import graphlib

        deps = {}
        constraints = set()
        if any(key != value.id for collection in (self.recipes, self.assets) for key, value in collection.items()):
            raise ValueError("recipe and asset dictionary keys must match their stable IDs")
        for n in self.nodes:
            refs = n.dependencies + ([n.parent_id] if n.parent_id else [])
            for c in n.relationships:
                if c.source not in {None, n.id}:
                    raise ValueError("relationship source must match its owning node")
                if c.id in constraints:
                    raise ValueError("duplicate constraint ID: " + c.id)
                constraints.add(c.id)
                if c.target not in ids:
                    raise ValueError("missing constraint target: " + c.target)
            if any(r not in ids for r in refs):
                raise ValueError("missing parent or dependency: " + n.id)
            deps[n.id] = refs
        try:
            tuple(graphlib.TopologicalSorter(deps).static_order())
        except graphlib.CycleError as exc:
            raise ValueError("cyclic design hierarchy") from exc
        return self


class Diagnostic(Model):
    layer: Literal["design", "constraint", "spatial", "support", "geometry", "budget", "capability", "quality"]
    code: str
    nodes: list[ID] = Field(default_factory=list)
    constraint: ID | None = None
    message: str
    severity: Literal["error", "warning"] = "error"
    approximate: bool = False
    relaxed: bool = False
    compromise: str | None = None


class RepairRecord(Model):
    node: ID
    original: object
    new: object
    reason: str
    constraint: ID
    intent_impact: str


class GraphReport(Model):
    schema_version: Literal[2] = 2
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    repairs: list[RepairRecord] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)

    @property
    def valid(self):
        return not any(d.severity == "error" for d in self.diagnostics)
