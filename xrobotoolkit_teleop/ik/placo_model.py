"""Full observed robot geometry using Placo's native model wrapper."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import placo

from .model import Distance, NativeState, checked_pair_indices, model_description, validate_collision_links


class PlacoModel:
    def __init__(self, spec):
        pairs = json.loads(Path(spec.collision_pairs_path).read_text())
        descriptions = validate_collision_links(spec.urdf_path, pairs)
        self._native = placo.RobotWrapper(spec.urdf_path)
        self._native.load_collision_pairs(spec.collision_pairs_path)
        self._names = tuple(self._native.joint_names())
        self.state = NativeState(self._native)
        self.model = model_description(self._names, self._native.model.velocity_limit.copy())
        self.collision_model = SimpleNamespace(
            objects=tuple(descriptions[obj.name] for obj in self._native.collision_model.objects),
            collision_pairs=tuple(self._native.collision_model.collision_pairs),
        )
        self._init_subset_queries()
        for name, position in spec.initial_positions.items():
            self.set_joint(name, position)
        self.update_kinematics()

    def _init_subset_queries(self):
        """Use Placo placements and the same Coal primitives, without another FK model."""
        if any(obj.type not in {"box", "cylinder", "sphere"} for obj in self.collision_model.objects):
            self.distance_subset = None
            return
        try:
            import coal
        except ImportError:
            # Placo-only installations may provide libcoal without its Python
            # bindings. The shared checker then keeps exhaustive behavior.
            self.distance_subset = None
            return
        self._coal = coal
        self._subset_shapes = []
        for obj in self.collision_model.objects:
            if obj.type == "box":
                shape = coal.Box(*obj.size)
            elif obj.type == "cylinder":
                shape = coal.Cylinder(*obj.size[:2])
            else:
                shape = coal.Sphere(obj.size[0])
            self._subset_shapes.append(shape)
        self._subset_request = coal.DistanceRequest()
        self._subset_request.enable_signed_distance = True
        self._subset_parents = [obj.parent_joint for obj in self._native.collision_model.objects]

    def distance_subset(self, pair_indices):
        """Exact signed distances and copied witnesses in requested pair order."""
        pairs = self.collision_model.collision_pairs
        indices = checked_pair_indices(pair_indices, len(pairs))
        if not indices:
            return []
        placements = self.collision_placements(self.state.q)
        transforms = {
            index: self._coal.Transform3s(placements[index, :3, :3], placements[index, :3, 3])
            for index in {obj for pair in indices for obj in pairs[pair]}
        }
        records = []
        for index in indices:
            a, b = pairs[index]
            result = self._coal.DistanceResult()
            self._coal.distance(
                self._subset_shapes[a],
                transforms[a],
                self._subset_shapes[b],
                transforms[b],
                self._subset_request,
                result,
            )
            records.append(
                Distance(
                    a,
                    b,
                    self._subset_parents[a],
                    self._subset_parents[b],
                    result.min_distance,
                    np.array(result.getNearestPoint1()),
                    np.array(result.getNearestPoint2()),
                )
            )
        return records

    def joint_names(self):
        return self._names

    def get_joint_offset(self, name):
        return self._native.get_joint_offset(name)

    def get_joint_v_offset(self, name):
        return self._native.get_joint_v_offset(name)

    def set_joint(self, name, value):
        self._native.set_joint(name, value)

    def get_joint(self, name):
        return self._native.get_joint(name)

    def set_joint_limits(self, name, lower, upper):
        self._native.set_joint_limits(name, lower, upper)

    def set_velocity_limit(self, name, value):
        self._native.set_velocity_limit(name, value)
        self.model.velocity_limit[self.get_joint_v_offset(name)] = value

    def update_kinematics(self):
        self._native.update_kinematics()

    def get_T_world_frame(self, name):
        return self._native.get_T_world_frame(name).copy()

    def joint_jacobian(self, name, frame):
        return self._native.joint_jacobian(name, frame)

    def collision_placements(self, q):
        return np.asarray(self._native.collision_placements(q))

    def distances(self):
        return [
            Distance(d.objA, d.objB, d.parentA, d.parentB, d.min_distance, np.array(d.pointA), np.array(d.pointB))
            for d in self._native.distances()
        ]
