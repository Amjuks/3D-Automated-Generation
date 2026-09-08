"""Reusable generators in a unit box. Geometry checkpoints use pickle-free NPZ."""

import io
from functools import lru_cache

import numpy as np
import trimesh
from PIL import Image

from .util import atomic_write, digest, file_hash

GENERATOR_VERSION = 2


def signature(component):
    return digest(
        {
            "version": GENERATOR_VERSION,
            "generator": component.generator,
            "parameters": component.parameters,
            "seed": component.seed if component.generator in {"urn", "rock", "sculpture", "tree"} else 0,
            "quality": component.budget.detail,
            "size": component.bounds.size,
            "materials": [m.model_dump() for m in component.materials],
        }
    )


@lru_cache(maxsize=128)
def primitive(kind, quality):
    segments = {"draft": 12, "standard": 32, "high": 64}[quality]
    if kind == "box":
        mesh = trimesh.creation.box()
    elif kind == "cylinder":
        mesh = trimesh.creation.cylinder(radius=0.5, height=1, sections=segments)
    elif kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions={"draft": 1, "standard": 2, "high": 3}[quality])
    elif kind == "vase":
        profile = np.array(
            [
                [0, 0],
                [0.25, 0],
                [0.26, 0.06],
                [0.38, 0.22],
                [0.45, 0.48],
                [0.30, 0.72],
                [0.16, 0.85],
                [0.22, 0.97],
                [0.22, 1],
                [0, 1],
            ]
        )
        mesh = trimesh.creation.revolve(profile, sections=segments)
    else:
        raise ValueError(f"unregistered generator: {kind}")
    mesh.vertices -= mesh.bounds[0]
    mesh.vertices /= mesh.extents
    return mesh


@lru_cache(maxsize=16)
def asset_mesh(path, sha256):
    if file_hash(path) != sha256:
        raise ValueError("source asset checksum mismatch")
    return trimesh.load(path, force="scene", process=False).to_mesh()


def generate(component):
    if component.generator == "asset":
        mesh = asset_mesh(component.parameters["path"], component.parameters["sha256"]).copy()
        if component.parameters.get("up_axis") == "Y":
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        if not len(mesh.faces) or np.any(mesh.extents <= 0):
            raise ValueError("asset has no volumetric geometry")
        mesh.vertices -= mesh.bounds[0]
        # Uniform scale preserves asset proportions, centered on its support allocation.
        factor = min(np.array(component.bounds.size) / mesh.extents)
        mesh.vertices *= factor
        mesh.vertices[:, :2] += (np.array(component.bounds.size)[:2] - mesh.extents[:2]) / 2
    elif component.generator == "box":
        from .organic import rounded_box

        mesh = rounded_box(component.bounds.size, component.materials[0].bevel if component.materials else 0.004)
    elif component.generator in {"vault", "arch"}:
        from .organic import arc_mesh

        mesh = arc_mesh(
            component.bounds.size,
            component.parameters.get("thickness", 0.10),
            segments=24 if component.budget.detail == "draft" else 64,
        )
    elif component.generator in {"urn", "rock", "sculpture", "torus", "tree"}:
        from .organic import form

        mesh = form(component.generator, component.budget.detail, component.seed, component.parameters)
        mesh.vertices *= component.bounds.size
    else:
        mesh = primitive(component.generator, component.budget.detail).copy()
        mesh.vertices *= component.bounds.size
    rotation = component.parameters.get("rotation_degrees")
    if rotation and any(rotation) and component.generator != "asset":
        angles = np.radians(rotation)
        mesh.apply_transform(trimesh.transformations.euler_matrix(*angles))
        mesh.vertices -= mesh.bounds[0]
        mesh.vertices *= np.array(component.bounds.size) / mesh.extents
    if component.generator == "asset" and getattr(mesh.visual, "uv", None) is not None:
        return {
            "vertices": np.asarray(mesh.vertices),
            "faces": np.asarray(mesh.faces),
            "uv": np.asarray(mesh.visual.uv),
        }
    # Split per-face vertices for a deterministic box projection without UV seams across faces.
    vertices = mesh.vertices[mesh.faces].reshape((-1, 3))
    faces = np.arange(len(vertices)).reshape((-1, 3))
    normals = np.abs(mesh.face_normals)
    uv = np.empty((len(vertices), 2))
    for axis in range(3):
        chosen = np.flatnonzero(normals.argmax(axis=1) == axis)
        indices = (chosen[:, None] * 3 + np.arange(3)).reshape(-1)
        axes = [a for a in range(3) if a != axis]
        uv[indices] = vertices[indices][:, axes]  # 1 UV repeat per meter.
    if component.parameters.get("surface") in {"artwork", "label"}:
        axes = component.parameters.get("uv_axes", [0, 2])
        uv = vertices[:, axes] / np.array(component.bounds.size)[axes]
    elif component.materials:
        uv *= component.materials[0].texture_scale
    return {"vertices": vertices, "faces": faces, "uv": uv}


def save_geometry(path, arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    atomic_write(path, buf.getvalue())


def load_geometry(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in ("vertices", "faces", "uv")}


def as_mesh(component, arrays):
    mesh = trimesh.Trimesh(vertices=arrays["vertices"], faces=arrays["faces"], process=False)
    m = component.materials[0]
    kwargs = {}
    if m.base_color_texture:
        with Image.open(m.base_color_texture) as img:
            kwargs["baseColorTexture"] = img.convert("RGB")
    if m.normal_texture:
        with Image.open(m.normal_texture) as img:
            kwargs["normalTexture"] = img.convert("RGB")
    if m.roughness_texture:
        with Image.open(m.roughness_texture) as img:
            roughness = img.convert("L")
            kwargs["metallicRoughnessTexture"] = Image.merge(
                "RGB", (Image.new("L", roughness.size, 255), roughness, Image.new("L", roughness.size, 0))
            )
    material = trimesh.visual.material.PBRMaterial(
        name=m.name,
        baseColorFactor=m.color,
        roughnessFactor=m.roughness,
        metallicFactor=m.metallic,
        emissiveFactor=[min(1, c * m.emission) for c in m.color[:3]],
        alphaMode="BLEND" if m.color[3] < 1 else "OPAQUE",
        doubleSided=False,
        **kwargs,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=arrays["uv"], material=material)
    return mesh
