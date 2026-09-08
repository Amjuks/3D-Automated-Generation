import hashlib

import httpx
import pytest

from scene_generator.polyhaven import PolyHaven


def test_download_integrity_and_offline_reuse(tmp_path):
    data = b"test asset payload"
    file = {
        "url": "https://dl.polyhaven.org/file/ph-assets/test.glb",
        "size": len(data),
        "md5": hashlib.md5(data).hexdigest(),
    }
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["User-Agent"].startswith("SceneGenerator/")
        return httpx.Response(200, content=data)

    client = PolyHaven(tmp_path, transport=httpx.MockTransport(handler))
    try:
        path, reused = client.download(file)
        assert not reused and path.read_bytes() == data
        client.offline = True
        assert client.download(file)[1]
        assert len(calls) == 1
        path.write_bytes(b"corrupt")
        with pytest.raises(RuntimeError, match="offline"):
            client.download(file)
    finally:
        client.close()


def test_budget_checksum_and_host_checks(tmp_path):
    client = PolyHaven(
        tmp_path, max_bytes=10, transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"abc"))
    )
    try:
        with pytest.raises(ValueError, match="official"):
            client.download({"url": "http://localhost/secret", "size": 3})
        with pytest.raises(ValueError, match="budget"):
            client.download({"url": "https://dl.polyhaven.org/a.glb", "size": 11})
        with pytest.raises(ValueError, match="checksum"):
            client.download({"url": "https://dl.polyhaven.org/a.glb", "size": 3, "md5": "bad"})
        assert not list(tmp_path.rglob("*.glb"))
    finally:
        client.close()


def test_model_bundle_traversal_rejected(tmp_path):
    import json

    data = json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": "../../secret.bin", "byteLength": 3}]}).encode()
    client = PolyHaven(tmp_path, transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data)))
    try:
        with pytest.raises(ValueError, match="unsafe"):
            client.download_model(
                {
                    "url": "https://dl.polyhaven.org/a.gltf",
                    "size": len(data),
                    "include": {"../../secret.bin": {"url": "https://dl.polyhaven.org/a.bin", "size": 3}},
                }
            )
    finally:
        client.close()


def test_local_textured_model_applied(config, tmp_path):
    import numpy as np
    import trimesh
    from PIL import Image

    from scene_generator.pipeline import Pipeline
    from scene_generator.util import read_json, write_json

    model = trimesh.creation.box()
    model.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(model.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (8, 8), "red")),
    )
    model.export(tmp_path / "model.glb")
    write_json(
        tmp_path / "catalog.json", [{"id": "local-box", "type": "model", "path": "model.glb", "license": "CC0-1.0"}]
    )
    config.scenes = {"museum": 1}
    config.generation.assets.mode = "local"
    config.generation.assets.local_catalog = tmp_path / "catalog.json"
    pipeline = Pipeline.create(config)
    try:
        pipeline.run()
        scene = next((pipeline.path / "scenes").iterdir())
        records = read_json(scene / "assets.json")
        assert records[0]["base_color_texture"]
        assert read_json(scene / "validation.json")["valid"]
    finally:
        pipeline.close()
