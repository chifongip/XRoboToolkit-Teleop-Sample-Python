"""Full observed robot geometry using Pinocchio/Coal, independently of Placo."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import coal

from .model import Distance, checked_pair_indices, model_description, validate_collision_links


class PinocchioModel:
    def __init__(self, spec):
        pairs = json.loads(Path(spec.collision_pairs_path).read_text())
        descriptions = validate_collision_links(spec.urdf_path, pairs)
        self._native = pin.buildModelFromUrdf(spec.urdf_path)
        if self._native.nq != self._native.nv:
            raise ValueError("Fixed-base scalar robot joints are required")
        self._data = self._native.createData()
        self._geometry = pin.buildGeomFromUrdf(self._native, spec.urdf_path, pin.GeometryType.COLLISION)
        self._names = tuple(self._native.names[1:])
        self.model = model_description(self._names, np.r_[np.zeros(6), self._native.velocityLimit])
        self.state = SimpleNamespace(q=np.r_[np.zeros(6), 1.0, pin.neutral(self._native)])
        self._indices = {n: self._native.getJointId(n) for n in self._names}
        self._geometry.removeAllCollisionPairs()
        objects = []
        counts = {}
        link_objects = {}
        for i, geometry in enumerate(self._geometry.geometryObjects):
            link = self._native.frames[geometry.parentFrame].name
            count = counts.get(link, 0)
            counts[link] = count + 1
            objects.append(descriptions[f"{link}_{count}"])
            link_objects.setdefault(link, []).append(i)
        seen = set()
        for left, right in pairs:
            for a in link_objects[left]:
                for b in link_objects[right]:
                    pair = tuple(sorted((a, b)))
                    if pair not in seen:
                        self._geometry.addCollisionPair(pin.CollisionPair(*pair))
                        seen.add(pair)
        self.collision_model = SimpleNamespace(
            objects=tuple(objects), collision_pairs=tuple((p.first, p.second) for p in self._geometry.collisionPairs)
        )
        self._geometry_data = pin.GeometryData(self._geometry)
        self._pair_metadata = []
        for pair in self._geometry.collisionPairs:
            a, b = pair.first, pair.second
            pa = self._geometry.geometryObjects[a].parentJoint
            pb = self._geometry.geometryObjects[b].parentJoint
            self._pair_metadata.append((a, b, pa + 1 if pa else 0, pb + 1 if pb else 0))
        for request in self._geometry_data.distanceRequests:
            request.enable_signed_distance = True
            request.gjk_initial_guess = coal.GJKInitialGuess.CachedGuess
        for name, value in spec.initial_positions.items():
            self.set_joint(name, value)
        self.update_kinematics()

    def joint_names(self):
        return self._names

    def get_joint_offset(self, name):
        return 7 + self._native.idx_qs[self._indices[name]]

    def get_joint_v_offset(self, name):
        return 6 + self._native.idx_vs[self._indices[name]]

    def set_joint(self, name, value):
        self.state.q[self.get_joint_offset(name)] = value

    def get_joint(self, name):
        return float(self.state.q[self.get_joint_offset(name)])

    def set_joint_limits(self, name, lower, upper):
        index = self._native.idx_qs[self._indices[name]]
        self._native.lowerPositionLimit[index] = lower
        self._native.upperPositionLimit[index] = upper

    def set_velocity_limit(self, name, value):
        self.model.velocity_limit[self.get_joint_v_offset(name)] = value

    def update_kinematics(self):
        pin.computeJointJacobians(self._native, self._data, self.state.q[7:])
        pin.updateFramePlacements(self._native, self._data)

    def get_T_world_frame(self, name):
        index = self._native.getFrameId(name, pin.FrameType.BODY)
        if index >= self._native.nframes:
            index = self._native.getFrameId(name)
        if index >= self._native.nframes:
            raise ValueError(f"Unknown robot frame: {name}")
        return self._data.oMf[index].homogeneous.copy()

    def joint_jacobian(self, name, frame):
        if frame != "world":
            raise ValueError("Collision Jacobians must be expressed in world coordinates")
        J = pin.getJointJacobian(self._native, self._data, self._indices[name], pin.ReferenceFrame.WORLD)
        return np.c_[np.zeros((6, 6)), J]

    def collision_placements(self, q):
        pin.updateGeometryPlacements(self._native, self._data, self._geometry, self._geometry_data, q[7:])
        return np.array([T.homogeneous for T in self._geometry_data.oMg])

    def distances(self):
        pin.computeDistances(self._native, self._data, self._geometry, self._geometry_data, self.state.q[7:])
        return [self._distance_record(i, d) for i, d in enumerate(self._geometry_data.distanceResults)]

    def _distance_record(self, index, distance):
        return Distance(
            *self._pair_metadata[index],
            distance.min_distance,
            np.array(distance.getNearestPoint1()),
            np.array(distance.getNearestPoint2()),
        )

    def distance_subset(self, pair_indices):
        """Exact signed distances and copied witnesses, in requested pair order."""
        indices = checked_pair_indices(pair_indices, len(self._pair_metadata))
        if not indices:
            return []
        pin.updateGeometryPlacements(self._native, self._data, self._geometry, self._geometry_data, self.state.q[7:])
        return [
            self._distance_record(index, pin.computeDistance(self._geometry, self._geometry_data, index))
            for index in indices
        ]
