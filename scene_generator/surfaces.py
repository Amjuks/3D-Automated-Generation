"""Deterministic tileable PBR fallback surfaces, exported as real texture maps."""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .util import atomic_write, digest, stable_seed

SURFACE_VERSION = 3


def png(path, array):
    image = Image.fromarray(np.uint8(np.clip(array, 0, 255)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    atomic_write(path, buffer.getvalue())


def noise(size, rng):
    out = np.zeros((size, size))
    # Periodic Fourier noise: seamlessly repeating plaster/stone grain, not pixel static.
    yy, xx = np.mgrid[:size, :size] / size * 2 * np.pi
    for octave in range(1, 7):
        frequency = 2**octave
        for _ in range(4):
            out += (
                np.sin(
                    xx * rng.integers(1, frequency + 1)
                    + yy * rng.integers(-frequency, frequency + 1)
                    + rng.uniform(0, 2 * np.pi)
                )
                / frequency
            )
    out /= max(out.std(), 0.001)
    return out


def font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def print_texture(path, text, seed, artwork=False):
    rng = np.random.default_rng(seed)
    size = 768 if artwork else 512
    canvas = Image.new("RGB", (size, size), (222, 215, 196) if artwork else (235, 231, 216))
    draw = ImageDraw.Draw(canvas)
    if artwork:
        # Layered abstract pigment studies, varied composition with a canvas grain.
        palette = [(42, 63, 61), (140, 71, 44), (182, 144, 91), (51, 66, 85), (195, 183, 152)]
        for _ in range(45):
            x, y = rng.integers(-120, size, 2)
            w, h = rng.integers(40, 420, 2)
            color = palette[int(rng.integers(len(palette)))]
            if rng.random() < 0.55:
                draw.ellipse((int(x), int(y), int(x + w), int(y + h)), fill=color)
            else:
                draw.rectangle((int(x), int(y), int(x + w), int(y + h)), fill=color)
        arr = np.array(canvas, dtype=float)
        yy, xx = np.mgrid[:size, :size]
        arr += (rng.normal(0, 2, (size, size)) + 0.8 * np.sin(xx * np.pi / 2) + 0.8 * np.sin(yy * np.pi / 2))[
            :, :, None
        ]
        png(path, arr)
    else:
        draw.line((35, 65, size - 35, 65), fill=(120, 88, 45), width=3)
        import textwrap

        for index, line in enumerate(textwrap.wrap(text, width=26)[:6]):
            draw.text((35, 95 + index * 42), line, fill=(35, 36, 32), font=font(25))
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        atomic_write(path, buf.getvalue())


def texture_scene(nodes, cache, quality):
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    size = {"draft": 128, "standard": 512, "high": 1024}[quality]
    cache_materials = {}
    for node in nodes:
        if not node.materials:
            continue
        mat = node.materials[0]
        if mat.name in {"glass", "water", "light"}:
            continue
        if mat.base_color_texture:
            continue
        motif = node.parameters.get("surface", mat.name)
        printable = motif in {"artwork", "label"}
        key = digest(
            {
                "version": SURFACE_VERSION,
                "motif": motif,
                "material": mat.model_dump(),
                "size": size,
                "seed": node.seed if printable else 0,
                "text": node.parameters.get("text", ""),
            }
        )[:24]
        if key in cache_materials:
            node.materials = [cache_materials[key]]
            continue
        directory = cache / key
        directory.mkdir(exist_ok=True)
        color_path, rough_path, normal_path = (directory / f"{name}.png" for name in ("color", "roughness", "normal"))
        if printable:
            if not color_path.exists():
                print_texture(color_path, node.parameters.get("text", node.name), node.seed, motif == "artwork")
            modified = mat.model_copy(
                update={"color": (1.0, 1.0, 1.0, 1.0), "base_color_texture": str(color_path), "texture_scale": 1}
            )
        else:
            if not all(p.exists() for p in (color_path, rough_path, normal_path)):
                rng = np.random.default_rng(stable_seed(motif, SURFACE_VERSION))
                n = noise(size, rng)
                yy, xx = np.mgrid[:size, :size] / size
                micro = rng.normal(0, 0.1, (size, size))
                if any(k in motif for k in ("wood", "timber", "parquet")):
                    grain = np.sin(2 * np.pi * (xx * 28 + 0.35 * n))
                    height = 0.5 * n + 0.35 * grain + micro
                    variation = 0.07 * n + 0.04 * grain
                elif any(k in motif for k in ("stone", "limestone", "marble")):
                    vein = np.exp(-np.abs(np.sin(2 * np.pi * (xx * 2 + yy + 0.15 * n))) * 35)
                    height = 0.3 * n + 0.8 * vein + micro
                    variation = 0.035 * n - 0.10 * vein
                elif "metal" in motif or "bronze" in motif:
                    height = 0.1 * n + micro
                    variation = 0.025 * n + 0.02 * micro
                elif motif == "terrazzo":
                    flecks = (rng.random((size, size)) > 0.985).astype(float)
                    from scipy.ndimage import gaussian_filter

                    flecks = gaussian_filter(flecks, 1.1) * 15
                    height = 0.03 * n + micro
                    variation = 0.03 * n + 0.15 * flecks
                else:
                    height = 0.5 * n + micro
                    variation = 0.035 * n + 0.01 * micro
                base = np.array(mat.color[:3]) * 255
                color = base[None, None, :] * (1 + variation[:, :, None])
                png(color_path, color)
                png(rough_path, (mat.roughness + 0.045 * n + 0.01 * micro) * 255)
                dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.8
                dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.8
                normal = np.dstack((-dx, -dy, np.ones_like(dx)))
                normal /= np.linalg.norm(normal, axis=2, keepdims=True)
                png(normal_path, (normal * 0.5 + 0.5) * 255)
            modified = mat.model_copy(
                update={
                    "color": (1.0, 1.0, 1.0, 1.0),
                    "base_color_texture": str(color_path),
                    "roughness_texture": str(rough_path),
                    "normal_texture": str(normal_path),
                }
            )
        cache_materials[key] = modified
        node.materials = [modified]
