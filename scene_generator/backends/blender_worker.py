"""Executed by Blender's bundled Python; no scene_generator imports required."""

import json
import os
import sys
from pathlib import Path


def main():
    import math

    import bmesh
    import bpy
    import numpy as np
    from mathutils import Vector

    job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    materials = {}
    meshes = {}
    objects = {}
    for node in job["nodes"]:
        if node["generator"] == "group":
            obj = bpy.data.objects.new(node["id"], None)
        else:
            import hashlib

            signature = hashlib.sha256(
                json.dumps(
                    {k: node[k] for k in ("generator", "parameters", "materials", "budget", "seed")}, sort_keys=True
                ).encode()
                + str([b - a for a, b in zip(node["bounds"]["min"], node["bounds"]["max"])]).encode()
            ).hexdigest()
            if signature not in meshes:
                with np.load(job["geometry_paths"][node["id"]], allow_pickle=False) as arrays:
                    data = bpy.data.meshes.new(signature)
                    data.from_pydata(arrays["vertices"].tolist(), [], arrays["faces"].tolist())
                    data.update()
                    uv = data.uv_layers.new(name="UVMap")
                    for loop in data.loops:
                        uv.data[loop.index].uv = arrays["uv"][loop.vertex_index]
                if node["generator"] in {"asset", "sculpture", "organic", "tree", "rock"}:
                    bm = bmesh.new()
                    bm.from_mesh(data)
                    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
                    bm.to_mesh(data)
                    bm.free()
                for polygon in data.polygons:
                    polygon.use_smooth = node["generator"] not in {"box", "asset"}
                mat = node["materials"][0]
                key = json.dumps(mat, sort_keys=True)
                if key not in materials:
                    m = bpy.data.materials.new(mat["name"])
                    m.use_nodes = True
                    bsdf = m.node_tree.nodes.get("Principled BSDF")
                    bsdf.inputs["Base Color"].default_value = mat["color"]
                    bsdf.inputs["Roughness"].default_value = mat["roughness"]
                    bsdf.inputs["Metallic"].default_value = mat["metallic"]
                    bsdf.inputs["Alpha"].default_value = mat["color"][3]
                    bsdf.inputs["Transmission Weight"].default_value = mat.get("transmission", 0)
                    bsdf.inputs["IOR"].default_value = 1.46
                    bsdf.inputs["Emission Color"].default_value = mat["color"]
                    bsdf.inputs["Emission Strength"].default_value = mat.get("emission", 0)
                    for field, socket in (
                        ("base_color_texture", "Base Color"),
                        ("roughness_texture", "Roughness"),
                        ("normal_texture", "Normal"),
                    ):
                        if mat.get(field):
                            tex = m.node_tree.nodes.new("ShaderNodeTexImage")
                            tex.image = bpy.data.images.load(mat[field], check_existing=True)
                            output = tex.outputs["Color"]
                            if field != "base_color_texture":
                                tex.image.colorspace_settings.name = "Non-Color"
                            if field == "normal_texture":
                                normal = m.node_tree.nodes.new("ShaderNodeNormalMap")
                                m.node_tree.links.new(output, normal.inputs["Color"])
                                output = normal.outputs["Normal"]
                            m.node_tree.links.new(output, bsdf.inputs[socket])
                            if field == "base_color_texture" and mat.get("emission", 0) > 0:
                                m.node_tree.links.new(output, bsdf.inputs["Emission Color"])
                    materials[key] = m
                data.materials.append(materials[key])
                meshes[signature] = data
            obj = bpy.data.objects.new(node["id"], meshes[signature])
        bpy.context.collection.objects.link(obj)
        if node["parent_id"]:
            obj.parent = objects[node["parent_id"]]
        obj.location = node["local_transform"]["translation"]
        objects[node["id"]] = obj
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    background = nodes.get("Background")
    background.inputs["Strength"].default_value = 0.45
    if job["options"].get("hdri"):
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(job["options"]["hdri"])
        scene.world.node_tree.links.new(env.outputs["Color"], background.inputs["Color"])
    atmosphere = job["options"].get("atmosphere", "warm")
    color = (
        (1.0, 0.80, 0.60) if atmosphere == "warm" else (0.75, 0.85, 1.0) if atmosphere == "cool" else (1.0, 0.95, 0.86)
    )
    # A real daylight source creates directional shadows through clerestories and courts.
    sun_data = bpy.data.lights.new("Afternoon sun", "SUN")
    lighting = job["options"].get("lighting")
    sun_data.energy = {"night": 0.04, "overcast": 0.5, "sunset": 1.2}.get(lighting, 2.0)
    sun_data.angle = 0.08
    sun_data.color = (1.0, 0.55, 0.27) if lighting == "sunset" else (1.0, 0.91, 0.78)
    background.inputs["Strength"].default_value = 0.04 if lighting == "night" else 0.45
    sun = bpy.data.objects.new("Afternoon sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(28), math.radians(-20), math.radians(-35))
    # Fixture geometry and exported spots use the same parent-local specification.
    for n in job["nodes"]:
        spec = n["parameters"].get("light")
        if not spec:
            continue
        data = bpy.data.lights.new(n["name"] + "-source", spec["type"])
        data.energy = spec["energy"]
        data.color = spec["color"]
        data.shadow_soft_size = 0.07
        if spec["type"] == "SPOT":
            data.spot_size = math.radians(64)
            data.spot_blend = 0.55
        light = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(light)
        offset = Vector(n["world_transform"]["translation"])
        light.location = offset + Vector((0.07, 0.07, -0.03))
        parent = objects[n["parent_id"]]
        target = parent.matrix_world.translation + Vector(spec["target"])
        light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()
    # Point lights survive KHR_lights_punctual GLB export; area lights enhance previews.
    for n in job["nodes"]:
        if n["kind"] == "room" and not n["parameters"].get("circulation"):
            extent = [b - a for a, b in zip(n["bounds"]["min"], n["bounds"]["max"])]
            data = bpy.data.lights.new(n["name"] + "-light", "POINT")
            data.energy = 140 if n["parameters"].get("wall_height") else 550 if atmosphere == "dramatic" else 1100
            data.color = color
            data.shadow_soft_size = 0.5
            light = bpy.data.objects.new(data.name, data)
            scene.collection.objects.link(light)
            light.location = [
                a + b
                for a, b in zip(n["world_transform"]["translation"], (extent[0] / 2, extent[1] / 2, extent[2] - 0.4))
            ]
    destination = Path(job["destination"])
    temp = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    if destination.suffix == ".glb":
        bpy.ops.export_scene.gltf(filepath=str(temp), export_format="GLB", export_yup=True, export_lights=True)
    elif destination.suffix == ".obj":
        bpy.ops.wm.obj_export(filepath=str(temp), export_materials=False)
    elif destination.suffix == ".ply":
        bpy.ops.wm.ply_export(filepath=str(temp))
    else:
        raise ValueError("unsupported Blender export format")
    os.replace(temp, destination)
    if job["options"].get("component_glbs"):
        for node in job["nodes"]:
            if node["generator"] == "group":
                continue
            bpy.ops.object.select_all(action="DESELECT")
            obj = objects[node["id"]]
            obj.select_set(True)
            out = str(Path(job["geometry_paths"][node["id"]]).with_suffix(".glb"))
            bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True)
    if job["options"].get("preview"):
        camera_data = bpy.data.cameras.new("Overview")
        camera = bpy.data.objects.new("Overview", camera_data)
        scene.collection.objects.link(camera)
        camera_spec = job["options"].get("camera")
        if camera_spec:
            origin = Vector(job["nodes"][0]["world_transform"]["translation"])
            camera.location = origin + Vector(camera_spec["position"])
            target = origin + Vector(camera_spec["target"])
        else:
            room = next(
                (n for n in job["nodes"] if n["kind"] == "room" and not n["parameters"].get("circulation")),
                job["nodes"][0],
            )
            room_origin = Vector(room["world_transform"]["translation"])
            rw, rd, rh = [b - a for a, b in zip(room["bounds"]["min"], room["bounds"]["max"])]
            camera_spec = room["parameters"].get(
                "camera", {"position": [rw / 2, 0.7, 1.65], "target": [rw / 2, rd - 1, 1.4]}
            )
            camera.location = room_origin + Vector(camera_spec["position"])
            target = room_origin + Vector(camera_spec["target"])
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera_data.lens = 24
        scene.camera = camera
        scene.render.engine = "CYCLES"
        scene.cycles.samples = job["options"].get("preview_samples", 64)
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 8
        scene.view_settings.view_transform = "AgX"
        width = job["options"].get("preview_width", 1440)
        scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = (
            width,
            int(width * 2 / 3),
            100,
        )
        bpy.ops.wm.save_as_mainfile(filepath=str(destination.with_suffix(".blend")))
        scene.render.filepath = str(destination.with_suffix(".png"))
        bpy.ops.render.render(write_still=True)
        if job["options"].get("camera"):
            detail = next((n for n in job["nodes"][1:] if n["parameters"].get("camera")), None)
            if detail:
                origin = Vector(detail["world_transform"]["translation"])
                spec = detail["parameters"]["camera"]
                camera.location = origin + Vector(spec["position"])
                target = origin + Vector(spec["target"])
                camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
                scene.render.filepath = str(destination.with_name("detail.png"))
                bpy.ops.render.render(write_still=True)
        if job["options"].get("layout", "legacy") != "legacy":
            root = job["nodes"][0]
            size = Vector([b - a for a, b in zip(root["bounds"]["min"], root["bounds"]["max"])])
            origin = Vector(root["world_transform"]["translation"])
            target = origin + Vector((size.x * 0.5, size.y * 0.5, size.z * 0.25))
            camera.location = origin + Vector((-size.x * 0.24, -size.y * 0.40, max(size.x, size.y) * 0.72))
            camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
            camera_data.lens = 40
            scene.render.filepath = str(destination.with_name("exterior.png"))
            bpy.ops.render.render(write_still=True)
    Path(job["report"]).write_text(
        json.dumps(
            {
                "backend": "blender",
                "version": bpy.app.version_string,
                "unique_meshes": len(meshes),
                "instances": len(job["geometry_paths"]),
                "bytes": destination.stat().st_size,
                "limitations": ["HDRI affects preview lighting; glTF does not embed a world environment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
