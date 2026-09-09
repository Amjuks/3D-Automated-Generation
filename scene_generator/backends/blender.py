import os
import shutil
import subprocess
from pathlib import Path

from ..util import read_json, write_json


class BlenderTimeoutError(RuntimeError):
    pass


def discover_blender():
    configured = os.getenv("BLENDER_PATH")
    if configured:
        return shutil.which(configured) or (
            configured if Path(configured).is_file() and os.access(configured, os.X_OK) else None
        )
    system = shutil.which("blender")
    if system:
        return system
    # Optional portable installation lives with this project, without modifying PATH.
    portable = Path(__file__).resolve().parents[2] / ".cache/tools/blender-4.5.1-linux-x64/blender"
    return str(portable) if portable.is_file() and os.access(portable, os.X_OK) else None


class BlenderBackend:
    name = "blender"

    def __init__(self, timeout=600):
        self.executable = discover_blender()
        self.timeout = timeout
        if not self.executable:
            raise RuntimeError("Blender unavailable; install Blender or include trimesh in generation.frameworks")

    def export(self, scene_path, nodes, geometry_paths, destination, options):
        job = scene_path / "blender-job.json"
        write_json(
            job,
            {
                "nodes": [n.model_dump(mode="json") for n in nodes],
                "geometry_paths": {k: str(Path(v).resolve()) for k, v in geometry_paths.items()},
                "destination": str(destination.resolve()),
                "options": options,
                "report": str((scene_path / "blender-stats.json").resolve()),
            },
        )
        script = Path(__file__).with_name("blender_worker.py")
        env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_")}
        with (scene_path / "blender.log").open("w") as log:
            try:
                process = subprocess.run(
                    [
                        self.executable,
                        "--background",
                        "--factory-startup",
                        "--disable-autoexec",
                        "--python-exit-code",
                        "1",
                        "--python",
                        str(script),
                        "--",
                        str(job.resolve()),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout,
                    env=env,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise BlenderTimeoutError("Blender timed out; export can be resumed") from None
        if process.returncode or not destination.exists():
            raise RuntimeError("Blender export failed; see scene blender.log")
        return read_json(scene_path / "blender-stats.json")
