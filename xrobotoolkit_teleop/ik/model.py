"""Geometry records shared by native model adapters; no solver dependencies."""

from dataclasses import dataclass
from operator import index
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class Geometry:
    name: str
    link: str
    type: str
    size: np.ndarray


@dataclass(frozen=True)
class Distance:
    objA: int
    objB: int
    parentA: int
    parentB: int
    min_distance: float
    pointA: np.ndarray
    pointB: np.ndarray


def checked_pair_indices(pair_indices, pair_count):
    """Accept integer pair IDs without silently truncating floats or strings."""
    indices = [index(value) for value in pair_indices]
    if any(value < 0 or value >= pair_count for value in indices):
        raise ValueError("Collision pair index out of range")
    return indices


def geometry_descriptions(urdf_path):
    result = {}
    for link in ET.parse(urdf_path).getroot().findall("link"):
        for i, collision in enumerate(link.findall("collision")):
            shape = list(collision.find("geometry"))[0]
            size = np.zeros(3)
            if shape.tag == "box":
                size[:] = np.fromstring(shape.get("size"), sep=" ")
            elif shape.tag in ("cylinder", "capsule"):
                size[:2] = float(shape.get("radius")), float(shape.get("length"))
            elif shape.tag == "sphere":
                size[0] = float(shape.get("radius"))
            name = f"{link.get('name')}_{i}"
            result[name] = Geometry(name, link.get("name"), shape.tag, size)
    return result


def validate_collision_links(urdf_path, pairs):
    geometries = geometry_descriptions(urdf_path)
    links = {g.link for g in geometries.values()}
    for link in {name for pair in pairs for name in pair}:
        if link not in links:
            raise ValueError(f"X2 IK URDF requires collision geometry for {link}")
    return geometries


class NativeState:
    """Expose NumPy configuration data without returning a native state object."""

    def __init__(self, native):
        self._native = native

    @property
    def q(self):
        return self._native.state.q

    @q.setter
    def q(self, value):
        self._native.state.q = value


def model_description(names, velocity):
    # Full geometry configurations use a leading world pose followed by scalar
    # joints. Controlled-joint problem vectors never contain these base entries.
    return SimpleNamespace(
        names=("universe", "root_joint", *names),
        nq=7 + len(names),
        nv=6 + len(names),
        njoints=2 + len(names),
        velocity_limit=velocity,
    )
