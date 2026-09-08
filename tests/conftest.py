import pytest

from scene_generator.config import Config


@pytest.fixture
def config(tmp_path):
    return Config.model_validate(
        {
            "scenes": {"room": 1},
            "generation": {
                "workflow": "legacy",
                "frameworks": ["trimesh"],
                "llm": {"mode": "mock"},
                "output": {"root": str(tmp_path / "runs")},
                "assets": {"mode": "mock", "cache_dir": str(tmp_path / "assets")},
            },
        }
    )
