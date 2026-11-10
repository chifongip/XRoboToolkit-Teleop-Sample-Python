"""Placo differential QP engine; model ownership stays with its backend."""

import numpy as np
import placo

from .backend import IKSolver


class PlacoIKSolver(IKSolver):
    def __init__(self, spec, robot):
        self.spec = spec
        self.robot = robot
        self._native = placo.KinematicsSolver(self.robot._native)
        self._native.dt = spec.dt
        self._native.mask_fbase(True)
        self._native.enable_joint_limits(True)
        # Shared absolute command bounds enforce velocity independently of servo error.
        self._native.enable_velocity_limits(False)
        for name in self.robot.joint_names():
            if name not in spec.joints:
                self._native.mask_dof(name)
                self.robot.set_joint_limits(name, -1e6, 1e6)
        for name, bounds in spec.limits.items():
            self.robot.set_joint_limits(name, *bounds)
        self.tasks = {
            key: self._native.add_frame_task(frame, self.robot.get_T_world_frame(frame))
            for key, frame in spec.frames.items()
        }
        self.manipulability = [
            self._native.add_manipulability_task(frame, "both", 1.0) for frame in spec.frames.values()
        ]
        self.posture_task = self._native.add_joints_task()
        self._constraints = [
            self._native.add_joint_space_half_spaces_constraint(np.zeros((1, self.robot.model.nq)), np.ones(1))
            for _ in range(3)
        ]
        self.offsets = np.array([self.robot.get_joint_offset(n) for n in spec.joints])

    def _set_constraint(self, destination, A, b, priority):
        full = np.zeros((max(1, len(b)), self.robot.model.nq))
        if len(b):
            full[:, self.offsets] = A
        destination.A = full
        destination.b = b if len(b) else np.ones(1)
        destination.configure("ik_constraint", priority, 1e6)

    def solve(self, problem):
        for name in self.spec.joints:
            if name in problem.active:
                self._native.unmask_dof(name)
            else:
                self.robot.set_joint(name, problem.previous[name])
                self._native.mask_dof(name)
        self.robot.update_kinematics()
        self._native.dt = problem.dt
        homing = problem.joint_target is not None
        for key, task in self.tasks.items():
            task.T_world_frame = problem.targets[key]
            task.configure(key, "soft", 0.0 if homing else 10.0)
        for task in self.manipulability:
            task.configure("manipulability", "soft", 0.0 if homing else 1e-2)
        self.posture_task.set_joints(
            problem.joint_target if homing else {name: self.spec.initial_positions[name] for name in self.spec.joints}
        )
        self.posture_task.configure("posture", "soft", 1.0 if homing else 1e-4)
        n = len(self.spec.joints)
        self._set_constraint(
            self._constraints[0], np.r_[np.eye(n), -np.eye(n)], np.r_[problem.upper, -problem.lower], "hard"
        )
        self._set_constraint(self._constraints[1], problem.hard.A, problem.hard.b, "hard")
        self._set_constraint(self._constraints[2], problem.soft.A, problem.soft.b, "soft")
        self._native.solve(True)
        q = np.array([self.robot.get_joint(n) for n in self.spec.joints])
        return dict(zip(self.spec.joints, q))
