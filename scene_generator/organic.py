"""Closed, curved mesh recipes. All geometry is fitted inside its allocated box."""

import numpy as np
import trimesh


def rounded_box(size, bevel):
    half = np.array(size) / 2
    bevel = min(bevel, min(half) * 0.35)
    points = []
    for signs in __import__("itertools").product((-1, 1), repeat=3):
        for axis in range(3):
            point = (half - bevel) * signs
            point[axis] = half[axis] * signs[axis]
            points.append(point)
    mesh = trimesh.convex.convex_hull(np.array(points))
    mesh.vertices += half
    return mesh


def arc_mesh(size, thickness=0.12, segments=48, arch=False):
    """Extruded elliptical vault shell: continuous mesh, closed end rims."""
    w, depth, height = size
    outer = []
    for y in (0, depth):
        for radius in (1.0, 1.0 - min(0.3, thickness / min(w / 2, height))):
            for angle in np.linspace(0, np.pi, segments + 1):
                outer.append((w / 2 + w / 2 * radius * np.cos(angle), y, height * radius * np.sin(angle)))
    n = segments + 1
    faces = []

    def quad(a, b, c, d):
        faces.extend(((a, b, c), (a, c, d)))

    for i in range(segments):
        quad(i, i + 1, 2 * n + i + 1, 2 * n + i)
        quad(n + i, 3 * n + i, 3 * n + i + 1, n + i + 1)
        quad(i, n + i, n + i + 1, i + 1)
        quad(2 * n + i, 2 * n + i + 1, 3 * n + i + 1, 3 * n + i)
    for i in (0, segments):
        quad(i, 2 * n + i, 3 * n + i, n + i)
    mesh = trimesh.Trimesh(outer, faces, process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def form(kind, quality, seed, parameters):
    rng = np.random.default_rng(seed)
    level = {"draft": 1, "standard": 2, "high": 3}[quality]
    if kind in {"rock", "tree"}:
        mesh = trimesh.creation.icosphere(subdivisions=level + (kind == "rock"))
        v = mesh.vertices
        perturb = 0.09 * np.sin(v[:, 0] * 7 + rng.random() * 6) * np.cos(v[:, 1] * 5) + 0.06 * np.sin(
            v[:, 2] * 11 + v[:, 0] * 4
        )
        mesh.vertices *= (1 + perturb)[:, None]
    elif kind == "urn":
        waist = rng.uniform(0.23, 0.40)
        profile = [
            [0, 0],
            [0.21, 0],
            [0.24, 0.035],
            [0.24, 0.07],
            [0.31, 0.12],
            [0.43, 0.32],
            [0.45, 0.50],
            [waist, 0.70],
            [0.16, 0.84],
            [0.16, 0.94],
            [0.23, 0.96],
            [0.23, 1],
            [0.18, 1],
            [0.13, 0.92],
            [0.13, 0.84],
            [waist - 0.045, 0.70],
            [0.40, 0.49],
            [0.37, 0.32],
            [0.26, 0.16],
            [0, 0.16],
            [0, 0],
        ]
        mesh = trimesh.creation.revolve(np.array(profile), sections=24 * (level + 1))
    elif kind == "torus":
        mesh = trimesh.creation.torus(
            major_radius=0.36, minor_radius=0.09, major_sections=32 * (level + 1), minor_sections=12
        )
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif kind == "sculpture":
        pieces = []

        def ellipsoid(center, radii):
            m = trimesh.creation.icosphere(subdivisions=level)
            m.vertices *= radii
            m.vertices += center
            pieces.append(m)

        # Classical bust: shaped shoulders, neck, head, nose and draped chest.
        ellipsoid((0, 0, 0.35), (0.34, 0.20, 0.34))
        ellipsoid((0, 0, 0.62), (0.12, 0.12, 0.17))
        ellipsoid((0.025, -0.012, 0.81), (0.18, 0.155, 0.235))
        ellipsoid((0.025, -0.16, 0.79), (0.045, 0.067, 0.09))
        for side in (-1, 1):
            ellipsoid((side * 0.30, 0, 0.43), (0.14, 0.17, 0.11))
        mesh = trimesh.util.concatenate(pieces)
    else:
        raise ValueError("unknown organic generator")
    mesh.vertices -= mesh.bounds[0]
    mesh.vertices /= mesh.extents
    return mesh
