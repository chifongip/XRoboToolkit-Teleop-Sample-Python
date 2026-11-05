"""Exact Placo distances and conservative primitive-clearance checks."""

import numpy as np


class CollisionChecker:
    """Cache one control cycle's queries for a fixed set of collision pairs.

    Joint configurations may change freely. Recreate the checker when collision
    geometry or pairs change, and clear its cache at each control-cycle boundary.
    """

    MAX_CACHE_ENTRIES = 128

    def __init__(self, robot):
        self.robot = robot
        self.pair_count = len(robot.collision_model.collision_pairs)
        distances = robot.distances()
        if not distances or len(distances) != self.pair_count:
            raise RuntimeError("Collision model requires geometry for every configured pair")
        self.pair_indices = np.array([(distance.objA, distance.objB) for distance in distances])
        objects = robot.collision_model.objects
        supported = {"box": 0, "cylinder": 1, "sphere": 2}
        self.can_bound = all(obj.type in supported for obj in objects)
        self.shape_types = np.array([supported.get(obj.type, -1) for obj in objects])
        self.shape_sizes = np.array([obj.size for obj in objects])
        self.bounding_radii = np.linalg.norm(self.shape_sizes / 2, axis=1)
        cylinders = self.shape_types == 1
        self.bounding_radii[cylinders] = np.hypot(self.shape_sizes[cylinders, 0], self.shape_sizes[cylinders, 1] / 2)
        spheres = self.shape_types == 2
        self.bounding_radii[spheres] = self.shape_sizes[spheres, 0]
        self._distance_cache = {}
        self._reference_distances = None
        self._reference_transforms = None

    def clear_cache(self):
        self._distance_cache.clear()
        self._reference_distances = None
        self._reference_transforms = None

    def distances(self):
        if not np.all(np.isfinite(self.robot.state.q)):
            raise RuntimeError("non-finite collision configuration")
        self.robot.update_kinematics()
        key = self.robot.state.q.tobytes()
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached
        distances = self.robot.distances()
        if not distances or len(distances) != self.pair_count:
            raise RuntimeError("Collision model requires geometry for every configured pair")
        values = np.array([d.min_distance for d in distances])
        if not np.all(np.isfinite(values)):
            raise RuntimeError("non-finite collision distances")
        if len(self._distance_cache) >= self.MAX_CACHE_ENTRIES:
            self._distance_cache.clear()
        values.flags.writeable = False
        self._distance_cache[key] = values
        if self.can_bound:
            self._reference_distances = values
            self._reference_transforms = np.asarray(self.robot.collision_placements(self.robot.state.q))
        return values

    def _support_radii(self, indices, local_axes):
        """Projection radii for boxes, Z-axis cylinders, and spheres."""
        sizes = self.shape_sizes[indices]
        radii = np.einsum("pki,pi->pk", np.abs(local_axes), sizes / 2)
        cylinders = self.shape_types[indices] == 1
        if np.any(cylinders):
            axes = local_axes[cylinders]
            radii[cylinders] = sizes[cylinders, 0, None] * np.linalg.norm(axes[:, :, :2], axis=2) + (
                sizes[cylinders, 1, None] / 2
            ) * np.abs(axes[:, :, 2])
        spheres = self.shape_types[indices] == 2
        if np.any(spheres):
            radii[spheres] = sizes[spheres, 0, None] * np.linalg.norm(local_axes[spheres], axis=2)
        return radii

    def clearance_bounds(self, floor):
        """Conservatively certify distant primitives; query close pairs exactly.

        AABB separation and projection-axis gaps bound unsigned primitive distance
        from below. Zero never certifies clearance or penetration recovery.
        """
        if not self.can_bound:
            raise ValueError("Clearance bounds require boxes, cylinders, or spheres")
        if not np.all(np.isfinite(self.robot.state.q)):
            raise RuntimeError("non-finite collision configuration")
        transforms = np.asarray(self.robot.collision_placements(self.robot.state.q))
        rotation, center = transforms[:, :3, :3], transforms[:, :3, 3]
        a, b = self.pair_indices.T
        motion_bounds = None
        if self._reference_distances is not None:
            reference = self._reference_transforms
            movement = np.linalg.norm(center - reference[:, :3, 3], axis=1)
            # A rigid shape moves by at most its center translation plus the
            # rotation chord at its enclosing radius. Spheres ignore rotation.
            chord = np.linalg.norm(rotation - reference[:, :3, :3], axis=(1, 2)) / np.sqrt(2)
            chord[self.shape_types == 2] = 0
            movement += self.bounding_radii * chord
            motion_bounds = self._reference_distances - movement[a] - movement[b] - 1e-9
            if np.all(motion_bounds >= floor):
                return motion_bounds
        world_half = self._support_radii(np.arange(len(rotation)), rotation)
        delta = center[b] - center[a]
        bounds = np.linalg.norm(np.maximum(np.abs(delta) - world_half[a] - world_half[b], 0), axis=1)
        if motion_bounds is not None:
            bounds = np.maximum(bounds, motion_bounds)
        close = bounds < floor
        if np.any(close):
            ra, rb = rotation[a[close]], rotation[b[close]]
            aa, ab = ra.transpose(0, 2, 1), rb.transpose(0, 2, 1)
            cross = np.cross(aa[:, :, None, :], ab[:, None, :, :]).reshape(-1, 9, 3)
            axes = np.concatenate((aa, ab, cross, delta[close, None, :]), axis=1)
            norms = np.linalg.norm(axes, axis=2)
            axes = axes / np.maximum(norms[:, :, None], 1e-12)
            radius_a = self._support_radii(a[close], np.einsum("pkj,pji->pki", axes, ra))
            radius_b = self._support_radii(b[close], np.einsum("pkj,pji->pki", axes, rb))
            gaps = np.abs(np.einsum("pkj,pj->pk", axes, delta[close])) - radius_a - radius_b
            gaps[norms < 1e-8] = -np.inf
            bounds[close] = np.maximum(bounds[close], gaps.max(axis=1))
        return np.maximum(bounds - 1e-9, 0)

    def baseline(self, margin):
        if self.robot.state.q.tobytes() in self._distance_cache:
            return self.distances()
        if self.can_bound:
            bounds = self.clearance_bounds(np.full(self.pair_count, margin))
            if np.all(bounds >= margin):
                return bounds
        return self.distances()
