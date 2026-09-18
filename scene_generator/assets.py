from pathlib import Path

from .polyhaven import CREDIT, PolyHaven
from .util import digest, file_hash, read_json, write_json


def select_assets(config, plan, scene_path, log, requests=None, seed=0):
    checkpoint = scene_path / "assets.json"
    request_key = digest(requests) if requests is not None else None
    request_checkpoint = scene_path / "asset-request-key.json"
    if checkpoint.exists() and (
        request_key is None or (request_checkpoint.exists() and read_json(request_checkpoint) == request_key)
    ):
        records = read_json(checkpoint)
        dependencies = [(r["path"], r["sha256"]) for r in records]
        dependencies += [(f["path"], f["sha256"]) for r in records for f in r.get("files", [])]
        if all(Path(p).exists() and file_hash(p) == sha for p, sha in dependencies):
            return [{**r, "reused_this_pass": True} for r in records]
    records = []
    errors = []
    if config.mode == "local" or config.local_catalog:
        if not config.local_catalog:
            raise ValueError("assets.local_catalog is required for local mode")
        catalog = read_json(config.local_catalog)
        entries = []
        if requests is None:
            entries = catalog[: config.max_assets]
        else:
            for request in requests[: config.max_assets]:
                words = set(request["query"].lower().replace("_", " ").split())
                ranked = []
                for entry in catalog:
                    if entry["type"] != request["type"]:
                        continue
                    text = str([entry.get("id"), entry.get("tags", []), entry.get("name", "")]).lower()
                    score = sum(word in text for word in words)
                    if score:
                        ranked.append((score, entry))
                if ranked:
                    chosen = max(ranked, key=lambda pair: pair[0])[1]
                    entries.append(
                        {
                            **chosen,
                            **{k: v for k, v in request.items() if k in {"target_role", "target_material", "query"}},
                        }
                    )
        local_bytes = 0
        for entry in entries[: config.max_assets]:
            path = (config.local_catalog.parent / entry["path"]).resolve()
            size = path.stat().st_size
            if size > config.max_download_bytes or local_bytes + size > config.max_total_bytes:
                errors.append({"id": entry.get("id"), "error": "local asset exceeds byte policy"})
                continue
            local_bytes += size
            if entry["type"] not in {"model", "texture", "hdri"}:
                raise ValueError("local asset type must be model, texture or hdri")
            records.append(
                {
                    **entry,
                    "path": str(path),
                    "sha256": file_hash(path),
                    "reused": True,
                    "license": entry.get("license", "user-provided"),
                    "source": "local",
                }
            )
    if config.mode == "polyhaven":
        log.event("asset_credit", credit=CREDIT)
        client = PolyHaven(config.cache_dir, config.offline, config.max_download_bytes)
        if requests is None:
            from .legacy.assets import asset_requests

            queries = asset_requests(plan)
        else:
            queries = requests
        local_roles = {r.get("target_role") for r in records}
        if requests is not None:
            queries = [r for r in queries if r.get("target_role") not in local_roles]
        used_bytes = 0
        try:
            for request in queries[: max(0, config.max_assets - len(records))]:
                kind, query = request["type"], request["query"]
                log.event("asset_search", type=kind, role=request.get("target_role", request.get("target_material")))
                try:
                    for identity, meta in client.search(query, kind, seed=seed):
                        candidates = []
                        for keys, file in client.candidates(client.files(identity)):
                            suffix = Path(file["url"].split("?")[0]).suffix.lower()
                            allowed = {
                                "texture": {".jpg", ".png"},
                                "hdri": {".hdr", ".exr"},
                                "model": {".glb", ".gltf"},
                            }[kind]
                            if suffix not in allowed or file["size"] + sum(
                                f["size"] for f in file.get("include", {}).values()
                            ) > min(config.max_download_bytes, config.max_total_bytes - used_bytes):
                                continue
                            channel = "base_color"
                            if kind == "texture":
                                key = " ".join(keys).lower()
                                if not any(s in key for s in ("diff", "color", "albedo")):
                                    continue
                            candidates.append((file["size"], keys, file, channel))
                        if not candidates:
                            continue
                        _, keys, file, channel = sorted(
                            candidates, key=lambda x: (config.texture_resolution not in x[1], x[0])
                        )[0]
                        path, reused = client.download_model(file) if kind == "model" else client.download(file)
                        record = {
                            "id": identity,
                            "type": kind,
                            "path": str(path),
                            "sha256": file_hash(path),
                            "reused": reused,
                            "source": "https://polyhaven.com/a/" + identity,
                            "license": "CC0-1.0",
                            "credit": CREDIT,
                            "metadata": meta,
                            "channel": channel,
                            "target_material": request.get("target_material"),
                            "target_role": request.get("target_role"),
                            "query": query,
                            "download": file,
                        }
                        if kind == "texture":
                            maps = {}
                            for map_kind, needles in {
                                "normal_texture": ("nor_gl",),
                                "roughness_texture": ("rough",),
                            }.items():
                                options = [
                                    (f["size"], f)
                                    for ks, f in client.candidates(client.files(identity))
                                    if config.texture_resolution in ks
                                    and any(n in " ".join(ks).lower() for n in needles)
                                    and Path(f["url"]).suffix.lower() in {".jpg", ".png"}
                                    and f["size"]
                                    <= min(
                                        config.max_download_bytes,
                                        config.max_total_bytes
                                        - used_bytes
                                        - file["size"]
                                        - sum(Path(p).stat().st_size for p in maps.values()),
                                    )
                                ]
                                if options:
                                    map_path, _ = client.download(min(options, key=lambda x: x[0])[1])
                                    maps[map_kind] = str(map_path)
                            record["maps"] = maps
                        record["selection"] = request
                        used_bytes += Path(path).stat().st_size + sum(
                            Path(p).stat().st_size for p in record.get("maps", {}).values()
                        )
                        records.append(record)
                        log.event("asset_selected", type=kind, id=identity, role=request.get("target_role"))
                        break
                    else:
                        errors.append({"type": kind, "error": "no compatible asset within budget"})
                except (RuntimeError, ValueError, OSError) as exc:
                    errors.append({"type": kind, "error": type(exc).__name__})
        finally:
            client.close()
    if errors:
        write_json(scene_path / "asset-errors.json", errors)
        log.event("asset_fallback", count=len(errors))
        if config.required:
            raise RuntimeError("required assets unavailable; see asset-errors.json")
    # Preserve model appearance as a packed PBR atlas with matching UV coordinates.
    import io

    import trimesh

    from .util import atomic_write

    rejected = set()
    for index, record in enumerate(records):
        if record["type"] != "model":
            continue
        try:
            mesh = trimesh.load(record["path"], force="scene", process=False).to_mesh()
            if not len(mesh.faces) or min(mesh.extents) <= 0 or len(mesh.faces) > 500_000:
                raise ValueError("model is empty, flat or exceeds geometry allowance")
        except (ValueError, OSError, TypeError, KeyError) as exc:
            errors.append({"id": record["id"], "type": "model", "error": type(exc).__name__})
            rejected.add(index)
            continue
        if getattr(mesh.visual, "kind", None) == "texture":
            material = mesh.visual.material
            tex = getattr(material, "baseColorTexture", None)
            if tex is None:
                tex = getattr(material, "image", None)
            if tex is not None:
                buffer = io.BytesIO()
                tex.save(buffer, format="PNG")
                atlas = scene_path / "asset-textures" / (digest(record["id"])[:20] + ".png")
                atomic_write(atlas, buffer.getvalue())
                record["base_color_texture"] = str(atlas.resolve())
        record["triangles"] = len(mesh.faces)
        record["source_extents"] = mesh.extents.tolist()
    records = [r for i, r in enumerate(records) if i not in rejected]
    if rejected:
        write_json(scene_path / "asset-errors.json", errors)
        log.event("asset_fallback", count=len(rejected))
        if config.required:
            raise RuntimeError("required models could not be imported; see asset-errors.json")
    for record in records:
        dependencies = list(record.get("maps", {}).values())
        if record.get("base_color_texture"):
            dependencies.append(record["base_color_texture"])
        record["files"] = [{"path": p, "sha256": file_hash(p), "bytes": Path(p).stat().st_size} for p in dependencies]
        record["bytes"] = Path(record["path"]).stat().st_size
    if request_key is not None:
        write_json(request_checkpoint, request_key)
    write_json(checkpoint, records)
    return records


def apply_assets(nodes, records, legacy=False):
    textures = [r for r in records if r["type"] == "texture"]
    for node in nodes:
        if node.materials:
            matches = [
                r
                for r in textures
                if (r.get("target_role") and r["target_role"] == node.parameters.get("material_role"))
                or (not r.get("target_role") and node.materials[0].name == r.get("target_material", "floor"))
            ]
            if matches:
                tex = matches[node.seed % len(matches)]
                node.materials = [
                    node.materials[0].model_copy(
                        update={"color": (1.0, 1.0, 1.0, 1.0), "base_color_texture": tex["path"], **tex.get("maps", {})}
                    )
                ]
    if legacy:
        from .legacy.assets import apply_models

        apply_models(nodes, records)


def design_asset_requests(brief, designs):
    requests = [{"type": "hdri", "query": brief.environment + " " + brief.lighting, "target_role": "environment"}]
    requests.append({"type": "texture", "query": brief.terrain + " ground", "target_role": "environment/terrain"})
    per_zone = []
    for zi, (zone, design) in enumerate(zip(brief.zones, designs)):
        prefix = f"zone-{zi + 1:02d}"
        queue = [
            {
                "type": "texture",
                "query": design.floor_texture,
                "target_material": zone.ground,
                "target_role": prefix + "/floor",
            }
        ]
        queue += [
            {"type": "model", "query": o.asset_query, "target_role": prefix + f"/object-{oi + 1:02d}"}
            for oi, o in enumerate(design.objects)
            if o.asset_query and o.count
        ]
        if zone.enclosure == "building":
            queue.insert(
                2,
                {
                    "type": "texture",
                    "query": design.wall_texture,
                    "target_material": zone.wall,
                    "target_role": prefix + "/wall",
                },
            )
        per_zone.append(queue)
    # Round robin prevents the first zone consuming the entire asset allowance.
    for index in range(max(map(len, per_zone), default=0)):
        requests.extend(q[index] for q in per_zone if index < len(q))
    return requests
