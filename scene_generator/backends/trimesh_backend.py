from pathlib import Path

import numpy as np
import trimesh

from ..generators import as_mesh, load_geometry, signature
from ..util import atomic_write


class TrimeshBackend:
    name = "trimesh"

    def export(self, scene_path, nodes, geometry_paths, destination, options):
        scene = trimesh.Scene(base_frame="world")
        # glTF is Y up. Rotate the Z-up contract hierarchy exactly once at its root.
        axis = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        scene.graph.update(frame_from="world", frame_to="z_up", matrix=axis)
        meshes = {}
        for node in nodes:
            parent = node.parent_id or "z_up"
            if node.generator == "group":
                scene.graph.update(frame_from=parent, frame_to=node.id, matrix=node.local_transform.matrix)
                continue
            key = signature(node)
            if key not in meshes:
                mesh = as_mesh(node, load_geometry(geometry_paths[node.id]))
                meshes[key] = "mesh-" + key[:20]
                scene.add_geometry(
                    mesh,
                    geom_name=meshes[key],
                    node_name=node.id,
                    parent_node_name=parent,
                    transform=node.local_transform.matrix,
                )
            else:
                scene.graph.update(
                    frame_from=parent, frame_to=node.id, matrix=node.local_transform.matrix, geometry=meshes[key]
                )
        fmt = destination.suffix.lstrip(".")
        if fmt == "glb":
            data = scene.export(file_type="glb")
        else:
            data = scene.to_mesh().export(file_type=fmt)
        atomic_write(destination, data)
        if options.get("component_glbs"):
            for node in nodes:
                if node.generator != "group":
                    mesh = as_mesh(node, load_geometry(geometry_paths[node.id]))
                    atomic_write(
                        Path(geometry_paths[node.id]).with_suffix(".glb"), trimesh.Scene(mesh).export(file_type="glb")
                    )
        return {
            "backend": self.name,
            "unique_meshes": len(meshes),
            "instances": len(geometry_paths),
            "bytes": destination.stat().st_size,
            "limitations": ["HDRI and punctual lighting require Blender"],
        }
