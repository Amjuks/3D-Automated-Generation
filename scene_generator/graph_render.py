"""Explicit render intentions and backend compatibility; no auto-created scene lighting."""

from typing import Literal

import numpy as np
import trimesh
from pydantic import Field, model_validator

from .design import Diagnostic
from .models import Model, Vec3


class CameraSpec(Model):
    position: Vec3
    target: Vec3
    fov: float = Field(default=60, gt=0, lt=179)
    required: bool = False

    @model_validator(mode="after")
    def direction(self):
        if np.linalg.norm(np.asarray(self.position) - self.target) < 1e-8:
            raise ValueError("camera target must differ from position")
        return self


class LightSpec(Model):
    light_type: Literal["POINT", "SUN", "SPOT"]
    position: Vec3 = (0, 0, 0)
    color: Vec3
    intensity: float = Field(ge=0, le=1_000_000)
    required: bool = False

    @model_validator(mode="after")
    def channels(self):
        if any(x < 0 or x > 1 for x in self.color):
            raise ValueError("light color channels outside [0,1]")
        return self


class AtmosphereSpec(Model):
    color: Vec3
    intensity: float = Field(ge=0, le=1000)
    required: bool = False


def render_options(graph, resolved, backend, fmt, report):
    output = {"cameras": [], "lights": [], "atmospheres": []}
    for node in graph.nodes:
        if node.kind not in {"camera", "light", "atmosphere"} or not node.count:
            continue
        if node.count > 1:
            report.diagnostics.append(
                Diagnostic(
                    layer="capability",
                    code="render_instances",
                    nodes=[node.id],
                    message="render instances require separate explicit camera or light nodes",
                )
            )
            continue
        schema = {"camera": CameraSpec, "light": LightSpec, "atmosphere": AtmosphereSpec}[node.kind]
        try:
            spec = schema.model_validate(node.properties).model_dump()
        except ValueError as exc:
            report.diagnostics.append(Diagnostic(layer="design", code="render_spec", nodes=[node.id], message=str(exc)))
            continue
        supported = backend == "blender" or (node.kind == "camera" and fmt == "glb")
        if fmt != "glb" and node.kind in {"light", "camera"}:
            supported = False
        if not supported:
            report.diagnostics.append(
                Diagnostic(
                    layer="capability",
                    code="render_feature",
                    nodes=[node.id],
                    severity="error" if spec["required"] else "warning",
                    message=f"{backend}/{fmt} cannot preserve {node.kind}",
                )
            )
            continue
        spec["id"] = node.id
        matrix = np.asarray(resolved.matrices[node.id])
        for key in ("position", "target"):
            if key in spec:
                spec[key] = trimesh.transform_points([spec[key]], matrix)[0].tolist()
        spec["matrix"] = matrix.tolist()
        output[{"camera": "cameras", "light": "lights", "atmosphere": "atmospheres"}[node.kind]].append(spec)
    if len(output["atmospheres"]) > 1:
        report.diagnostics.append(
            Diagnostic(layer="capability", code="multiple_atmospheres", message="backend supports one world atmosphere")
        )
    if backend == "trimesh" and len(output["cameras"]) > 1:
        report.diagnostics.append(
            Diagnostic(layer="capability", code="multiple_cameras", message="trimesh supports one active camera")
        )
    return output


def camera_matrix(position, target):
    position, target = np.asarray(position), np.asarray(target)
    z = position - target
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    matrix[:3, 3] = position
    return matrix
