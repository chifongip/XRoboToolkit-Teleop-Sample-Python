"""Native model distances and conservative primitive-clearance checks."""

import numpy as np


class CollisionChecker:
    """Cache one control cycle's queries for a fixed set of collision pairs.

    Joint configurations may change freely. Recreate the checker when collision
    geometry or pairs change, and clear its cache at each control-cycle boundary.
    """

    MAX_CACHE_ENTRIES = 128
    BOUND_QUERY_GUARD = 1e-6

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
        self.clear_cache()

    def clear_cache(self):
        self._distance_cache.clear()
        self._clearance_cache = {}
        self._subset_witness_key = None
        self._subset_witnesses = {}
        self._bounded_reference = None
        self._reference_distances = None
        self._reference_transforms = None
        self._witness_key = None
        self._witnesses = None

    def distances(self, *, require_witnesses=False):
        if not np.all(np.isfinite(self.robot.state.q)):
            raise RuntimeError("non-finite collision configuration")
        self.robot.update_kinematics()
        key = self.robot.state.q.tobytes()
        cached = self._distance_cache.get(key)
        if cached is not None and (not require_witnesses or key == self._witness_key):
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
        # Retain only the latest witness query, rather than all geometry records
        # for every sampled configuration in the distance cache.
        self._witness_key = key
        self._witnesses = distances
        if self.can_bound:
            self._reference_distances = values
            self._reference_transforms = np.asarray(self.robot.collision_placements(self.robot.state.q))
        return values

    def clearances(self, threshold, *, require_witnesses=False):
        """Signed distances capped at threshold; certify distant pairs by bounds.

        Capped results never enter the exact distance or exact motion-reference
        cache. Unsupported bounds/subset queries return uncapped exact distances.
        """
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Collision clearance threshold must be finite and positive")
        if not np.all(np.isfinite(self.robot.state.q)):
            raise RuntimeError("non-finite collision configuration")
        subset = getattr(self.robot, "distance_subset", None)
        if not self.can_bound or not callable(subset):
            return self.distances(require_witnesses=require_witnesses)
        configuration_key = self.robot.state.q.tobytes()
        if configuration_key in self._distance_cache:
            return np.minimum(self.distances(require_witnesses=require_witnesses), threshold)
        key = (configuration_key, float(threshold))
        cached = self._clearance_cache.get(key)
        if cached is not None and (not require_witnesses or key == self._subset_witness_key):
            return cached
        if not require_witnesses:
            # A larger cap retains all exact values needed by a smaller one.
            # Do not reuse its witness key: those records may have been evicted.
            for (pose, cap), values in self._clearance_cache.items():
                if pose == configuration_key and cap >= threshold:
                    result = np.minimum(values, threshold)
                    result.flags.writeable = False
                    return result
        guarded = threshold + self.BOUND_QUERY_GUARD
        bounds = self.clearance_bounds(np.full(self.pair_count, guarded))
        if not np.isfinite(bounds).all():
            raise RuntimeError("non-finite collision bounds")
        indices = np.flatnonzero(bounds < guarded)
        records = subset(indices)
        if len(records) != len(indices):
            raise RuntimeError("Collision subset query returned incomplete results")
        exact = np.array([record.min_distance for record in records])
        if not np.isfinite(exact).all():
            raise RuntimeError("non-finite collision distances")
        values = np.full(self.pair_count, threshold, dtype=float)
        values[indices] = np.minimum(exact, threshold)
        values.flags.writeable = False
        if len(self._clearance_cache) >= self.MAX_CACHE_ENTRIES:
            self._clearance_cache.clear()
        self._clearance_cache[key] = values
        self._subset_witness_key = key
        self._subset_witnesses = dict(zip(indices, records, strict=True))
        if self._bounded_reference is None or threshold >= self._bounded_reference[0]:
            # These are lower bounds, not exact distances. A trigger-sized cap
            # retains enough slack to certify small subsequent path samples.
            self._bounded_reference = (
                threshold,
                values,
                np.asarray(self.robot.collision_placements(self.robot.state.q)),
            )
        return values

    def linearized_clearance(self, trigger):
        """Return nearby signed distances and their joint-velocity Jacobian.

        Reuse the exact witnesses from the baseline distance query. This is the
        same witness-point linearization as Placo's self-collision constraint,
        including its reversed normal for penetrating shapes.
        """
        values = (
            self.clearances(trigger, require_witnesses=True) if trigger > 0 else self.distances(require_witnesses=True)
        )
        nearby = np.flatnonzero(values < trigger)
        robot = self.robot
        robot.update_kinematics()
        jacobian = np.zeros((len(nearby), robot.model.nv))
        if nearby.size == 0:
            return values[nearby], jacobian
        key = (robot.state.q.tobytes(), float(trigger))
        witnesses_by_pair = (
            self._subset_witnesses
            if key == self._subset_witness_key and key[0] not in self._distance_cache
            else self._witnesses
        )
        witnesses = [witnesses_by_pair[int(index)] for index in nearby]
        a = np.array([distance.pointA for distance in witnesses])
        b = np.array([distance.pointB for distance in witnesses])
        parents = np.array([(distance.parentA, distance.parentB) for distance in witnesses])
        normal = b - a
        lengths = np.linalg.norm(normal, axis=1)
        if not np.all(np.isfinite(lengths)) or np.any(lengths < 1e-12):
            raise RuntimeError("undefined collision witness normal")
        normal /= lengths[:, None]
        normal[values[nearby] < 0] *= -1
        joint_jacobians = np.zeros((robot.model.njoints, 6, robot.model.nv))
        for parent in np.unique(parents):
            if parent != 0:
                joint_jacobians[parent] = robot.joint_jacobian(robot.model.names[int(parent)], "world")
        for sign, parent, point in ((-1, parents[:, 0], a), (1, parents[:, 1], b)):
            joint = joint_jacobians[parent]
            # WORLD twists are about the world origin. Shift their linear
            # velocity to the witness point before projecting on the normal.
            jacobian += sign * (
                np.einsum("pi,pij->pj", normal, joint[:, :3])
                + np.einsum("pi,pij->pj", np.cross(point, normal), joint[:, 3:])
            )
        if not np.all(np.isfinite(jacobian)):
            raise RuntimeError("non-finite collision Jacobian")
        return values[nearby], jacobian

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
        """Bound primitive clearance without querying exact distances.

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
        references = []
        if self._reference_distances is not None:
            references.append((self._reference_distances, self._reference_transforms))
        if self._bounded_reference is not None:
            references.append(self._bounded_reference[1:])
        for reference_distances, reference in references:
            movement = np.linalg.norm(center - reference[:, :3, 3], axis=1)
            # A rigid shape moves by at most its center translation plus the
            # rotation chord at its enclosing radius. Spheres ignore rotation.
            chord = np.linalg.norm(rotation - reference[:, :3, :3], axis=(1, 2)) / np.sqrt(2)
            chord[self.shape_types == 2] = 0
            movement += self.bounding_radii * chord
            candidate = reference_distances - movement[a] - movement[b] - 1e-9
            motion_bounds = candidate if motion_bounds is None else np.maximum(motion_bounds, candidate)
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
        if self.can_bound and callable(getattr(self.robot, "distance_subset", None)):
            return self.clearances(margin)
        if self.robot.state.q.tobytes() in self._distance_cache:
            return self.distances()
        if self.can_bound:
            bounds = self.clearance_bounds(np.full(self.pair_count, margin))
            if np.all(bounds >= margin):
                return bounds
        return self.distances()
