"""Read binary STL meshes and fit conservative collision primitives."""

from functools import lru_cache
from math import cos, sin
import struct

import numpy as np


@lru_cache(maxsize=None)
def mesh_triangles(path):
    """Read the bundled binary STL once; fitting happens only during generation."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Expected a binary STL: {path}")
    triangles = struct.unpack_from("<I", data, 80)[0]
    if not triangles or len(data) != 84 + 50 * triangles:
        raise ValueError(f"Expected a binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(data, dtype=dtype, offset=84)["vertices"].astype(float)


def urdf_quaternion(roll, pitch, yaw):
    """URDF fixed-axis RPY -> MuJoCo wxyz (independent of compiler eulerseq)."""
    cr, cp, cy = cos(roll / 2), cos(pitch / 2), cos(yaw / 2)
    sr, sp, sy = sin(roll / 2), sin(pitch / 2), sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quaternion_matrix(quaternion):
    w, x, y, z = quaternion
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def convex_hull_2d(points):
    points = np.unique(points, axis=0)
    if len(points) <= 2:
        return points

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])


def enclosing_circle(points):
    """Fit a deterministic enclosing circle to the projected mesh hull."""
    hull = convex_hull_2d(points)
    shuffled = hull[np.random.default_rng(0).permutation(len(hull))]
    center, radius_squared = shuffled[0].copy(), 0.0
    for i, p in enumerate(shuffled):
        if np.dot(p - center, p - center) <= radius_squared + 1e-14:
            continue
        center, radius_squared = p.copy(), 0.0
        for j, q in enumerate(shuffled[:i]):
            if np.dot(q - center, q - center) <= radius_squared + 1e-14:
                continue
            center = (p + q) / 2
            radius_squared = np.dot(p - center, p - center)
            for r in shuffled[:j]:
                if np.dot(r - center, r - center) <= radius_squared + 1e-14:
                    continue
                matrix = 2 * np.array([q - p, r - p])
                if abs(np.linalg.det(matrix)) < 1e-12:
                    a, b = max(((p, q), (p, r), (q, r)), key=lambda pair: np.sum((pair[0] - pair[1]) ** 2))
                    center = (a + b) / 2
                else:
                    # Solve in coordinates relative to p for numerical stability.
                    center = p + np.linalg.solve(matrix, [np.sum((q - p) ** 2), np.sum((r - p) ** 2)])
                radius_squared = max(np.sum((point - center) ** 2) for point in (p, q, r))
    # Recompute against the full hull: no fitting tolerance may omit vertices.
    return center, np.sqrt(np.max(np.sum((hull - center) ** 2, axis=1)))


def rotation_rpy(rotation):
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def fit_primitive(points):
    """Choose an enclosing primitive by volume, without dropping mesh points."""
    padding = 0.001
    bases = [np.eye(3)]
    bases.extend((np.eye(3)[:, [1, 2, 0]], np.eye(3)[:, [2, 0, 1]]))
    _, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
    for order in ((1, 0, 2), (2, 1, 0), (0, 2, 1)):
        basis = vectors[:, order].copy()
        if np.linalg.det(basis) < 0:
            basis[:, 0] *= -1
        bases.append(basis)
    candidates = []
    for basis in bases:
        local = points @ basis
        lower, upper = local.min(axis=0), local.max(axis=0)
        size = upper - lower + 2 * padding
        candidates.append((np.prod(size), "box", basis @ ((lower + upper) / 2), basis, size))
        center, radius = enclosing_circle(local[:, :2])
        radius += padding
        length = size[2]
        center = basis @ np.array([center[0], center[1], (lower[2] + upper[2]) / 2])
        candidates.append((np.pi * radius**2 * length, "cylinder", center, basis, np.array([radius, length, 0])))
    return min(candidates, key=lambda candidate: candidate[0] * (0.98 if candidate[1] == "cylinder" else 1))


def sliced_mesh_points(triangles, slices=None, with_limits=False, *, cuts=None):
    """Clip slabs using vertices and edge-plane intersections, covering faces."""
    vertices = triangles.reshape(-1, 3)
    if cuts is None:
        cuts = np.linspace(vertices[:, 2].min(), vertices[:, 2].max(), slices + 1)
    for lower, upper in zip(cuts[:-1], cuts[1:]):
        points = [vertices[(vertices[:, 2] >= lower) & (vertices[:, 2] <= upper)]]
        for plane in (lower, upper):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                start, end = triangles[:, a], triangles[:, b]
                dz = end[:, 2] - start[:, 2]
                crossing = (
                    (dz != 0)
                    & (np.minimum(start[:, 2], end[:, 2]) <= plane)
                    & (np.maximum(start[:, 2], end[:, 2]) >= plane)
                )
                t = (plane - start[crossing, 2]) / dz[crossing]
                points.append(start[crossing] + t[:, None] * (end[crossing] - start[crossing]))
        points = np.unique(np.concatenate(points), axis=0)
        if len(points):
            yield (lower, upper, points) if with_limits else points
