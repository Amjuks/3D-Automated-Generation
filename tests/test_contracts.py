import pytest
from pydantic import ValidationError

from scene_generator.config import Config
from scene_generator.models import Bounds, Transform
from scene_generator.spatial import SpatialIndex


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "1", 10001])
def test_reject_bad_counts(count):
    with pytest.raises(ValidationError):
        Config.model_validate({"scenes": {"museum": count}})


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1, 0])
def test_reject_invalid_bounds(value):
    with pytest.raises(ValidationError):
        Bounds(max=(value, 2, 3))


def test_touching_is_not_collision():
    a = Bounds(max=(1, 1, 1))
    b = Bounds(min=(1, 0, 0), max=(2, 1, 1))
    assert not a.intersects(b)
    assert a.contains(Bounds(min=(0.1, 0.1, 0.1), max=(0.9, 0.9, 0.9)))
    index = SpatialIndex()
    index.add("a", a)
    assert index.query(b) == []
    assert index.query(a) == ["a"]


def test_unsupported_transform_is_explicit():
    with pytest.raises(ValidationError):
        Transform(rotation=(0, 0, 90))
