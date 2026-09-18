"""Operation registry shared by planning, validation and compilation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometryCapability:
    name: str
    estimated_triangles: int
    parameters: frozenset[str] = frozenset()

    def validate(self, operation, policy):
        if set(operation.parameters) - self.parameters:
            raise ValueError("unsupported operation parameters")
        if max(operation.size) > policy.max_extent:
            raise ValueError("operation extent exceeds policy")
        if self.name == "mesh":
            v = np.asarray(operation.parameters.get("vertices"), dtype=float)
            f = np.asarray(operation.parameters.get("faces"))
            if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
                raise ValueError("invalid mesh vertices")
            if (
                f.ndim != 2
                or f.shape[1] != 3
                or not len(f)
                or not np.issubdtype(f.dtype, np.integer)
                or f.min() < 0
                or f.max() >= len(v)
            ):
                raise ValueError("invalid mesh faces")
            if (v < 0).any() or (v > operation.size).any():
                raise ValueError("mesh vertices exceed explicit recipe envelope")
            return len(f)
        return self.estimated_triangles


GEOMETRY = {
    name: GeometryCapability(name, triangles)
    for name, triangles in [
        ("box", 12),
        ("sphere", 1280),
        ("cylinder", 256),
        ("cone", 128),
        ("capsule", 32768),
        ("torus", 8192),
    ]
}
GEOMETRY["mesh"] = GeometryCapability("mesh", 0, frozenset({"vertices", "faces"}))
# Unsupported operations remain diagnosable; semantic categories are never registry keys.
BACKENDS = {
    "trimesh": {"geometry", "local_frames", "materials", "perspective_camera", "glb", "obj", "ply"},
    "blender": {"geometry", "local_frames", "materials", "perspective_camera", "light", "glb", "obj", "ply"},
}
