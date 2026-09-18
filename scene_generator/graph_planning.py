"""Incremental structured planning with content-addressed, validated checkpoints."""

import json
from typing import Literal

from pydantic import Field, ValidationInfo, model_validator

from .capabilities import BACKENDS
from .design import Assembly, AssetRequest, DesignGraph, DesignNode
from .models import Material, Model
from .policy import ScenePolicy
from .util import digest, read_json, stable_seed, write_json

STAGES = (
    "intent",
    "world",
    "regions",
    "structure",
    "circulation",
    "objects",
    "appearance",
    "alternatives",
    "recipes",
)
DIRECTIONS = (
    "Interpret the user intent, non-goals, and required experience.",
    "Propose the overall world and spatial story. No prescribed scene category or template.",
    "Define optional regions, nested spaces, volumes, and their relationships.",
    "Define chosen structures, terrain, surfaces, portals, openings and negative space. All are optional.",
    "Define only appropriate circulation, sightlines, accessibility and vertical transitions. Stairs are optional.",
    "Define objects, reusable groups, explicit repetition frames, supports and attachments.",
    "Define material values, asset requests, lighting, atmosphere and camera nodes.",
    "Resolve asset and procedural alternatives with explicit fallback policies.",
    "Supply complete reusable geometry recipes using only available operations. Preserve the spatial story.",
)


class StageInput(Model):
    schema_version: Literal[2] = 2
    stage: str
    concept: str
    seed: int
    policy: ScenePolicy
    graph: DesignGraph
    instruction: str
    runtime_limits: dict = Field(default_factory=dict)


class StageOutput(Model):
    schema_version: Literal[2] = 2
    graph: DesignGraph
    rationale: str

    @model_validator(mode="after")
    def accumulated(self, info: ValidationInfo):
        context = info.context or {}
        prior = set(context.get("prior_ids", []))
        current = {n.id for n in self.graph.nodes}
        if prior - current:
            raise ValueError("preserve existing stable IDs; express omissions using count=0")
        if len(self.graph.nodes) > context.get("max_nodes", 2000):
            raise ValueError("design exceeds node policy")
        if context.get("complete"):
            for n in self.graph.nodes:
                if n.recipe and n.recipe not in self.graph.recipes:
                    raise ValueError("missing recipe: " + n.recipe)
                if n.asset and n.asset not in self.graph.assets:
                    raise ValueError("missing asset request: " + n.asset)
            for recipe in self.graph.recipes.values():
                for part in recipe.parts:
                    if part.material not in self.graph.materials:
                        raise ValueError("missing recipe material: " + part.material)
        return self


class StageDelta(Model):
    """Only changed records cross the wire; full validated snapshots stay on disk."""

    title: str | None = None
    intent: str | None = None
    nodes: list[DesignNode] = Field(default_factory=list)
    materials: dict[str, Material] = Field(default_factory=dict)
    recipes: dict[str, Assembly] = Field(default_factory=dict)
    assets: dict[str, AssetRequest] = Field(default_factory=dict)
    rationale: str

    def merge(self, graph, context=None):
        data = graph.model_dump(mode="json")
        for field in ("title", "intent"):
            if getattr(self, field) is not None:
                data[field] = getattr(self, field)
        nodes = {n["id"]: n for n in data["nodes"]}
        nodes.update({n.id: n.model_dump(mode="json") for n in self.nodes})
        data["nodes"] = list(nodes.values())
        for field in ("materials", "recipes", "assets"):
            data[field].update({key: value.model_dump(mode="json") for key, value in getattr(self, field).items()})
        return StageOutput.model_validate({"graph": data, "rationale": self.rationale}, context=context)

    @model_validator(mode="after")
    def valid_update(self, info: ValidationInfo):
        context = info.context or {}
        if "previous_graph" in context:
            self.merge(DesignGraph.model_validate(context["previous_graph"]), context)
        return self


class StageCheckpoint(Model):
    schema_version: Literal[2] = 2
    stage: str
    request_id: str
    output_hash: str
    output: StageOutput


def checkpoint_request(path, stage, context, llm, scene_id, mock, limit):
    key = digest(
        {
            "contract": 2,
            "input": context.model_dump(mode="json"),
            "schema": StageDelta.model_json_schema(),
            "llm": llm.config.model_dump(mode="json"),
            "endpoint": llm.base,
            "model": llm.model,
        }
    )
    target = path / "planning" / f"{stage}.json"
    validation_context = {
        "prior_ids": [n.id for n in context.graph.nodes],
        "max_nodes": context.policy.max_nodes,
        "complete": stage == "recipes",
        "previous_graph": context.graph.model_dump(mode="json"),
    }
    if target.exists():
        try:
            if target.stat().st_size > limit:
                raise ValueError("oversized checkpoint")
            saved = StageCheckpoint.model_validate(read_json(target), context=validation_context)
            if saved.request_id == key and saved.output_hash == digest(saved.output.model_dump(mode="json")):
                return saved.output
        except (ValueError, OSError):
            pass
        # Keep one damaged/stale copy for inspection; bounded disk usage.
        target.replace(target.with_suffix(".invalid.json"))

    def mock_delta():
        output = StageOutput.model_validate(mock())
        prior = {n.id: n for n in context.graph.nodes}
        return StageDelta(
            title=output.graph.title,
            intent=output.graph.intent,
            nodes=[n for n in output.graph.nodes if n.id not in prior or n != prior[n.id]],
            materials=output.graph.materials,
            recipes=output.graph.recipes,
            assets=output.graph.assets,
            rationale=output.rationale,
        )

    prompt = context.model_dump(mode="json", exclude_defaults=True)
    answer = llm.request(
        StageDelta, prompt, scene_id, f"graph/{stage}", mock=mock_delta, validation_context=validation_context
    ).merge(context.graph, validation_context)
    for node in answer.graph.nodes:
        if not node.provenance.request_id:
            node.provenance.request_id = key
    saved = StageCheckpoint(
        stage=stage, request_id=key, output=answer, output_hash=digest(answer.model_dump(mode="json"))
    )
    if (
        len((json.dumps(saved.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        > limit
    ):
        raise ValueError("planning checkpoint exceeds policy byte limit")
    write_json(target, saved.model_dump(mode="json"))
    return answer


def mock_graph(concept, seed):
    """Explicit abstract test fixture, not a natural-language design interpreter."""
    from .design import Assembly, DesignNode, Operation, Provenance
    from .models import Material, Transform

    value = stable_seed(seed, concept)
    return DesignGraph(
        id="design",
        title=concept[:120],
        intent=concept,
        materials={"sample": Material(name="mock sample", color=(0.2 + value % 6 / 10, 0.4, 0.6, 1), bevel=0)},
        recipes={
            "sample": Assembly(
                id="sample",
                visual_intent="abstract offline test form",
                parts=[Operation(capability="sphere", size=(1 + value % 3, 2, 1), material="sample")],
            )
        },
        nodes=[
            DesignNode(
                id="sample",
                kind="object",
                description="Abstract offline test fixture",
                recipe="sample",
                provenance=Provenance(source="mock"),
                spatial={"frame": Transform(contract_version=2, translation=(0, 0, 2)).model_dump()},
            )
        ],
    )


def plan_graph(concept, seed, generation, llm, scene_id, path, stage_callback, asset_search=None):
    policy = ScenePolicy.from_generation(generation)
    graph = DesignGraph(id="design", title=concept[:120], intent=concept)
    candidates = []
    for stage, direction in zip(STAGES, DIRECTIONS):
        stage_callback(stage)
        if stage == "alternatives" and asset_search:
            candidates = asset_search(graph)
        context = StageInput(
            stage=stage,
            concept=concept,
            seed=seed,
            policy=policy,
            graph=graph,
            runtime_limits={
                "assets": generation.assets.model_dump(mode="json"),
                "asset_candidates": candidates,
                "backend_capabilities": {name: sorted(BACKENDS[name]) for name in generation.frameworks},
                "workers": generation.workers,
                "response_bytes": generation.llm.max_response_bytes,
            },
            instruction=direction
            + " Return only changed/new records as a StageDelta, preserving stable IDs. Omit all unchanged records and default fields. "
            "Keep this stage concise and incremental. Intent/world stages describe the story and semantic regions only, no recipes yet. "
            "Preserve architectural or landscape ambition through reusable assemblies and instancing. Use explicit geometric primitives, "
            "do not output large raw triangle arrays when compact assemblies suffice. Node relationships control placement. "
            "Plan a camera and lighting for a rendered preview. Never generate code. Semantic names do not select geometry. Dimensions and frames must "
            "come from the design. Empty scenes/regions and zero object counts are legal. "
            "No mandatory rooms, floors, furniture, buildings, paths, roofs, or centered doors. "
            "Unsupported features must have an explicit supported alternative or be reported. "
            "Frames use meters, Z up, Euler XYZ degrees, positive scale. Graph references must exist. "
            "Render properties: camera uses position,target,fov; light uses light_type,position,color,intensity.",
        )
        answer = checkpoint_request(
            path,
            stage,
            context,
            llm,
            scene_id,
            lambda: StageOutput(
                graph=mock_graph(concept, seed) if stage == "objects" else graph,
                rationale="Offline fixture; no semantic interpretation.",
            ),
            policy.max_checkpoint_bytes,
        )
        if len(answer.graph.nodes) > policy.max_nodes:
            raise ValueError("design exceeds policy node limit")
        if answer.graph.id != graph.id:
            raise ValueError("planning changed stable graph identity")
        graph = answer.graph
    write_json(path / "design-graph.json", graph.model_dump(mode="json"))
    return graph, policy
