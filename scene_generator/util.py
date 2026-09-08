import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts):
    return int(digest(parts)[:15], 16)


def slug(value):
    return re.sub(r"[^a-z0-9_-]", "_", value.lower()).strip("_")[:50] or "scene"


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data.encode() if isinstance(data, str) else data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())
