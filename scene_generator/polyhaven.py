"""Official Poly Haven API client with integrity-checked, offline-capable caching."""

import hashlib
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .util import atomic_write, digest, file_hash, read_json, stable_seed, write_json

CREDIT = "Powered by Poly Haven"
BASE_URL = "https://api.polyhaven.com"


class PolyHaven:
    def __init__(self, cache: Path, offline=False, max_bytes=50_000_000, transport=None):
        self.cache, self.offline, self.max_bytes = Path(cache), offline, max_bytes
        self.client = httpx.Client(
            headers={"User-Agent": "SceneGenerator/0.1 (procedural-3d-scenes)"},
            timeout=60,
            transport=transport,
            follow_redirects=False,
        )
        self.downloaded = 0
        self.reused = 0

    def close(self):
        self.client.close()

    def _get(self, url):
        for attempt in range(4):
            try:
                response = self.client.get(url)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code < 500
                    and exc.response.status_code != 429
                ):
                    raise RuntimeError(f"Poly Haven HTTP {exc.response.status_code}") from None
                if attempt == 3:
                    raise RuntimeError("Poly Haven request failed after retries") from None
                time.sleep(0.5 * 2**attempt)

    def json(self, endpoint):
        path = self.cache / "metadata" / (digest(endpoint) + ".json")
        if path.exists():
            try:
                return read_json(path)
            except (ValueError, OSError):
                if self.offline:
                    raise RuntimeError("cached Poly Haven metadata is invalid") from None
        if self.offline:
            raise RuntimeError("Poly Haven metadata unavailable offline; populate the cache first")
        value = self._get(BASE_URL + endpoint).json()
        write_json(path, value)
        return value

    def search(self, query, asset_type, limit=8, seed=0):
        catalog = self.json("/assets")
        type_id = {"hdri": 0, "texture": 1, "model": 2}[asset_type]
        words = list(dict.fromkeys(query.lower().replace("_", " ").split()))
        matches = []
        for identity, meta in catalog.items():
            if meta.get("type") not in (type_id, asset_type):
                continue
            text = str(
                [
                    identity,
                    meta.get("name"),
                    meta.get("tags"),
                    meta.get("categories"),
                    meta.get("category"),
                    meta.get("attributes"),
                    meta.get("description"),
                ]
            ).lower()
            score = sum((len(words) - index) for index, word in enumerate(words) if word in text)
            if score:
                matches.append((score, identity, meta))
        matches.sort(key=lambda row: (-row[0], stable_seed(seed, row[1]), row[1]))
        return [(identity, meta) for _, identity, meta in matches[:limit]]

    def files(self, identity):
        if not identity or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identity
        ):
            raise ValueError("invalid Poly Haven asset id")
        return self.json("/files/" + identity)

    @staticmethod
    def candidates(tree, prefix=()):
        if isinstance(tree, dict):
            if "url" in tree and "size" in tree:
                yield prefix, tree
            for key, value in tree.items():
                if isinstance(value, dict) and key != "include":
                    yield from PolyHaven.candidates(value, prefix + (key,))

    def download_model(self, file):
        """Assemble glTF dependencies without trusting their relative paths or URIs."""
        path, reused = self.download(file)
        if path.suffix != ".gltf":
            return path, reused
        from urllib.parse import unquote

        bundle = self.cache / "bundles" / digest(file["url"])
        bundle.mkdir(parents=True, exist_ok=True)
        includes = file.get("include", {})
        if sum(int(v["size"]) for v in includes.values()) + int(file["size"]) > self.max_bytes:
            raise ValueError("model bundle exceeds download budget")

        def safe_relative(value):
            decoded = unquote(value)
            target = (bundle / decoded).resolve()
            if not target.is_relative_to(bundle.resolve()) or ":" in decoded or "\\" in decoded:
                raise ValueError("unsafe glTF dependency path")
            return target

        model = read_json(path)
        uris = [entry.get("uri") for key in ("buffers", "images") for entry in model.get(key, [])]
        for uri in uris:
            if not uri or uri.startswith("data:"):
                continue
            safe_relative(uri)
            if uri not in includes:
                raise ValueError("glTF references an undeclared dependency")
        for relative, dependency in includes.items():
            target = safe_relative(relative)
            source, cached = self.download(dependency)
            reused = reused and cached
            target.parent.mkdir(parents=True, exist_ok=True)
            # Atomic copies avoid partially published dependencies after interruption.
            atomic_write(target, source.read_bytes())
        target = bundle / "model.gltf"
        atomic_write(target, path.read_bytes())
        # A single portable GLB seals the bundle and simplifies later integrity checks.
        import trimesh

        loaded = trimesh.load(target, force="scene", process=False)
        destination = bundle / "model.glb"
        atomic_write(destination, loaded.export(file_type="glb"))
        return destination.resolve(), reused

    def download(self, file):
        url = urlsplit(file["url"])
        if (
            url.scheme != "https"
            or url.hostname not in {"dl.polyhaven.org", "cdn.polyhaven.com"}
            or url.username
            or url.password
        ):
            raise ValueError("asset download URL must use an official HTTPS asset host")
        size = int(file["size"])
        if size <= 0 or size > self.max_bytes:
            raise ValueError("asset exceeds download budget")
        suffix = Path(url.path).suffix.lower()
        path = self.cache / "downloads" / (digest(file["url"]) + suffix)
        metadata = path.with_suffix(path.suffix + ".json")
        if path.exists() and metadata.exists():
            record = read_json(metadata)
            if path.stat().st_size == size and file_hash(path) == record["sha256"]:
                self.reused += 1
                return path.resolve(), True
        if self.offline:
            raise RuntimeError("asset file unavailable or corrupt in offline cache")
        # Stream to disk; enforce both advertised and actual byte limits.
        path.parent.mkdir(parents=True, exist_ok=True)
        import os
        import tempfile

        for attempt in range(4):
            fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".download-")
            try:
                count = 0
                md5 = hashlib.md5()
                with os.fdopen(fd, "wb") as output, self.client.stream("GET", file["url"]) as response:
                    response.raise_for_status()
                    for block in response.iter_bytes():
                        count += len(block)
                        if count > min(size, self.max_bytes):
                            raise ValueError("download exceeds declared size")
                        output.write(block)
                        md5.update(block)
                    output.flush()
                    os.fsync(output.fileno())
                if count != size or (file.get("md5") and md5.hexdigest() != file["md5"]):
                    raise ValueError("asset checksum or size mismatch")
                os.replace(temp, path)
                write_json(metadata, {"url": file["url"], "sha256": file_hash(path), "bytes": size})
                self.downloaded += 1
                return path.resolve(), False
            except httpx.HTTPError:
                if attempt == 3:
                    raise RuntimeError("asset download failed after retries") from None
                time.sleep(0.5 * 2**attempt)
            finally:
                if Path(temp).exists():
                    Path(temp).unlink()
        raise AssertionError("unreachable")
