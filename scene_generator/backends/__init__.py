from pathlib import Path
from typing import Protocol


class Backend(Protocol):
    name: str

    def export(self, scene_path: Path, nodes, geometry_paths, destination: Path, options: dict) -> dict: ...
