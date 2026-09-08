"""Deterministic spatial allocation and broad-phase index."""

from collections import defaultdict
from itertools import product
from math import floor

from .models import Bounds


class SpatialIndex:
    def __init__(self, cell_size=4.0):
        self.cell_size = cell_size
        self.cells = defaultdict(set)
        self.boxes = {}

    def keys(self, box):
        return product(
            *(range(floor(a / self.cell_size), floor(b / self.cell_size) + 1) for a, b in zip(box.min, box.max))
        )

    def add(self, identity, box):
        self.boxes[identity] = box
        for key in self.keys(box):
            self.cells[key].add(identity)

    def query(self, box):
        ids = set()
        for key in self.keys(box):
            ids.update(self.cells.get(key, ()))
        return [i for i in sorted(ids) if self.boxes[i].intersects(box)]


def box(origin, size):
    return Bounds(min=tuple(origin), max=tuple(a + b for a, b in zip(origin, size)))
