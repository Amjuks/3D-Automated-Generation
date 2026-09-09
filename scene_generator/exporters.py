import trimesh

from .backends.blender import BlenderBackend, BlenderTimeoutError, discover_blender
from .backends.trimesh_backend import TrimeshBackend
from .util import file_hash

EXPORT_VERSION = 2


def export_scene(scene_path, nodes, paths, generation, assets, plan, fmt=None):
    fmt = fmt or generation.output.format
    if fmt not in {"glb", "obj", "ply"}:
        raise ValueError("supported export formats: glb, obj, ply")
    if generation.output.require_blender and not discover_blender():
        raise RuntimeError("This quality profile requires Blender; install it or set BLENDER_PATH, then resume")
    backend = None
    for name in generation.frameworks:
        if name == "blender" and discover_blender():
            if len(paths) > 500 and "trimesh" in generation.frameworks:
                continue
            backend = BlenderBackend(generation.blender_timeout)
            break
        if name == "trimesh":
            backend = TrimeshBackend()
            break
    if backend is None:
        raise RuntimeError("Blender unavailable and no fallback configured; install Blender or enable trimesh")
    destination = scene_path / ("scene." + fmt)
    hdri = next((r["path"] for r in assets if r["type"] == "hdri"), None)
    options = {
        "hdri": hdri,
        "atmosphere": plan.atmosphere,
        "preview": generation.output.preview,
        "component_glbs": generation.output.component_glbs,
        "preview_samples": generation.output.preview_samples,
        "preview_width": generation.output.preview_width,
        "title": plan.title,
        "layout": plan.layout,
        "camera": nodes[0].parameters.get("camera"),
        "lighting": nodes[0].parameters.get("lighting"),
    }
    try:
        result = backend.export(scene_path, nodes, paths, destination, options)
    except BlenderTimeoutError:
        if "trimesh" not in generation.frameworks:
            raise
        backend = TrimeshBackend()
        result = backend.export(scene_path, nodes, paths, destination, options)
    if generation.output.preview and backend.name != "blender":
        result["preview"] = "unavailable: Blender required"
    if fmt == "glb":
        loaded = trimesh.load(destination, force="scene", process=False)
        if not loaded.geometry:
            raise RuntimeError("exported GLB contains no geometry")
        result["glb_reload_verified"] = True
    result.update(path=str(destination.resolve()), sha256=file_hash(destination), format=fmt)
    return result
