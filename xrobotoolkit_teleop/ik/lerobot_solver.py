"""LeRobot G1's nonlinear IK objective and filter for named scalar joints.

Algorithm reference: Hugging Face LeRobot's G1_29_ArmIK (Apache-2.0).
This module intentionally imports the optional native dependencies only on use.
"""

from collections import deque

import numpy as np

from .backend import IKSolver


def load_dependencies():
    try:
        import casadi
        import pinocchio
        from pinocchio import casadi as cpin

        if not casadi.has_nlpsol("ipopt"):
            raise ImportError("CasADi cannot load its IPOPT plugin")
    except (ImportError, AttributeError, RuntimeError) as exc:
        raise ImportError(
            "The lerobot backend requires compatible Pinocchio CasADi bindings and CasADi/IPOPT. "
            "See docs/lerobot_ik.md for environment setup; Placo remains available with --ik-backend placo."
        ) from exc
    return pinocchio, cpin, casadi


class G1IKSolver:
    """Reference G1 objective and filter with position bounds.

    Production LeRobotIKSolver adds velocity and collision constraints to this
    mathematical helper. Its backend owns transactions with the controller.

    solve() commits filter history only after a finite bounded solution. A failed
    solve raises RuntimeError so the controller can retain its servo commands.
    All public joint dictionaries use URDF joint names and radians.
    """

    FILTER_WEIGHTS = np.array([0.4, 0.3, 0.2, 0.1])  # oldest to newest, as in LeRobot

    def __init__(self, urdf_path, joint_names, position_limits, frames=None, groups=None):
        self.pin, cpin, ca = load_dependencies()
        self.joint_names = tuple(joint_names)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("LeRobot IK requires distinct controlled joints")
        full = self.pin.buildModelFromUrdf(str(urdf_path))
        self.full_model = full
        self.body_configuration = self.pin.neutral(full)
        self._full_indices = {name: full.idx_qs[full.getJointId(name)] for name in full.names[1:]}
        self._full_data = full.createData()
        missing = set(self.joint_names) - set(full.names)
        if missing:
            raise ValueError(f"IK model is missing arm joints: {sorted(missing)}")
        locked = [j for j in range(1, full.njoints) if full.names[j] not in self.joint_names]
        self.model = self.pin.buildReducedModel(full, locked, self.pin.neutral(full))
        if self.model.nq != len(self.joint_names) or self.model.nv != len(self.joint_names):
            raise ValueError("LeRobot IK requires scalar controlled joints")
        self.indices = np.array([self.model.idx_qs[self.model.getJointId(n)] for n in self.joint_names])
        self.arm_indices = {
            side: np.array([i for n, i in zip(self.joint_names, self.indices, strict=True) if n in names])
            for side, names in (
                groups
                or {side: [n for n in self.joint_names if n.startswith(f"{side}_")] for side in ("left", "right")}
            ).items()
        }
        self._active_keys = {key: key if groups is not None else f"{key}_arm" for key in self.arm_indices}
        if any(not len(indices) for indices in self.arm_indices.values()):
            raise ValueError("LeRobot IK requires nonempty control groups")
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        for name, index in zip(self.joint_names, self.indices, strict=True):
            lower, upper = position_limits[name]
            self.lower[index] = max(lower, self.lower[index])
            self.upper[index] = min(upper, self.upper[index])
        if not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower >= self.upper):
            raise ValueError("Incompatible finite joint position limits")
        frame_names = frames or {side: f"{side}_hand_center" for side in ("left", "right")}
        self.frame_ids = {key: self.model.getFrameId(frame) for key, frame in frame_names.items()}
        self._full_frame_ids = {key: full.getFrameId(frame) for key, frame in frame_names.items()}
        if any(i >= self.model.nframes for i in self.frame_ids.values()):
            raise ValueError("IK model is missing a configured end-effector frame")
        # Keep observed, uncontrolled joints in symbolic FK. Reducing them at
        # neutral would give the wrong targets when a deployment torso moves.
        cm = cpin.Model(full)
        cd = cm.createData()
        cq = ca.SX.sym("q", self.model.nq)
        body = ca.SX.sym("body", full.nq)
        configuration = ca.SX(body)
        for name, index in zip(self.joint_names, self.indices, strict=True):
            configuration[full.idx_qs[full.getJointId(name)]] = cq[index]
        targets = {side: ca.SX.sym(f"target_{side}", 4, 4) for side in self.frame_ids}
        cpin.framesForwardKinematics(cm, cd, configuration)
        translation = ca.vertcat(
            *[cd.oMf[fid].translation - targets[key][:3, 3] for key, fid in self._full_frame_ids.items()]
        )
        rotation = ca.vertcat(
            *[cpin.log3(cd.oMf[fid].rotation @ targets[key][:3, :3].T) for key, fid in self._full_frame_ids.items()]
        )
        # Pinocchio's symbolic log3 contains inactive branches with singular
        # derivatives at axis-aligned rotations. Reverse AD can propagate 0*NaN
        # through them; forward AD preserves the same objective without that path.
        self.errors = ca.Function(
            "ik_errors", [cq, body, *targets.values()], [translation, rotation], {"enable_reverse": False}
        )
        self._build_optimizer(ca)
        self._history = deque(maxlen=4)
        self._previous = None
        self._active = set()

    def _build_optimizer(self, ca):
        """Create a solver graph without changing accepted filter state."""
        self.opti = ca.Opti()
        self.q = self.opti.variable(self.model.nq)
        self.reference = self.opti.parameter(self.model.nq)
        self.targets = {side: self.opti.parameter(4, 4) for side in self.frame_ids}
        self.bound_lower = self.opti.parameter(self.model.nq)
        self.bound_upper = self.opti.parameter(self.model.nq)
        self.body = self.opti.parameter(self.full_model.nq)
        ep, er = self.errors(self.q, self.body, *self.targets.values())
        self.tracking_cost = (
            50 * ca.sumsqr(ep) + ca.sumsqr(er) + 0.02 * ca.sumsqr(self.q) + 0.1 * ca.sumsqr(self.q - self.reference)
        )
        self.opti.minimize(self.tracking_cost)
        self.opti.subject_to(self.opti.bounded(self.bound_lower, self.q, self.bound_upper))
        self.opti.solver(
            "ipopt",
            {
                "ipopt": {"print_level": 0, "sb": "yes", "max_iter": 50, "tol": 1e-6},
                "print_time": False,
                "calc_lam_p": False,
            },
        )

    def _vector(self, positions):
        q = np.empty(self.model.nq)
        for name, index in zip(self.joint_names, self.indices, strict=True):
            q[index] = positions[name]
        if not np.isfinite(q).all():
            raise ValueError("IK joint positions must be finite")
        return q

    def reset(self, joints=None):
        if joints is not None:
            self._active.difference_update(
                key
                for key, indices in self.arm_indices.items()
                if any(n in joints for n, i in zip(self.joint_names, self.indices, strict=True) if i in indices)
            )
            return
        self._history.clear()
        self._previous = None
        self._active.clear()

    def close(self):
        self.reset()

    def _full_configuration(self, positions):
        configuration = self.body_configuration.copy()
        # Require all controlled joints; omitted observed joints retain neutral.
        for name in self.joint_names:
            configuration[self._full_indices[name]] = positions[name]
        for name, value in positions.items():
            if name in self._full_indices:
                configuration[self._full_indices[name]] = value
        if not np.isfinite(configuration).all():
            raise ValueError("Observed body configuration must be finite")
        return configuration

    def forward_kinematics(self, positions):
        configuration = self._full_configuration(positions)
        self.pin.framesForwardKinematics(self.full_model, self._full_data, configuration)
        return {key: self._full_data.oMf[fid].homogeneous.copy() for key, fid in self._full_frame_ids.items()}

    def solve(self, measured_positions, previous_positions, targets, active_arms):
        measured = self._vector(measured_positions)
        held = self._vector(previous_positions)
        active = {side for side in self.frame_ids if active_arms.get(self._active_keys[side], False)}
        # Releases take effect even if this solve fails. New engagements are
        # committed only on success, so a retry still rebases from measurements.
        self._active.intersection_update(active)
        reference = held.copy() if self._previous is None else self._previous.copy()
        history = [q.copy() for q in self._history]
        # Rebase only changed arms, preserving the other arm's filter continuity.
        for side, indices in self.arm_indices.items():
            if side not in active:
                reference[indices] = held[indices]
                for sample in history:
                    sample[indices] = held[indices]
            elif side not in self._active:
                reference[indices] = measured[indices]
                for sample in history:
                    sample[indices] = measured[indices]
        lower, upper = self.lower.copy(), self.upper.copy()
        for group, indices in self.arm_indices.items():
            if group not in active:
                if np.any(held[indices] < lower[indices]) or np.any(held[indices] > upper[indices]):
                    raise RuntimeError(f"Held joints outside IK limits: {group}")
                lower[indices] = upper[indices] = held[indices]
        for side, parameter in self.targets.items():
            target = np.asarray(targets[side], dtype=float)
            if target.shape != (4, 4) or not np.isfinite(target).all():
                raise ValueError(f"Invalid {side} IK target")
            self.opti.set_value(parameter, target)
        self.opti.set_value(self.body, self._full_configuration(measured_positions))
        self.opti.set_initial(self.q, np.clip(reference, lower, upper))
        self.opti.set_value(self.reference, reference)
        self.opti.set_value(self.bound_lower, lower)
        self.opti.set_value(self.bound_upper, upper)
        try:
            solution = np.asarray(self.opti.solve().value(self.q)).reshape(self.model.nq)
        except RuntimeError as exc:
            raise RuntimeError("LeRobot IK did not converge; holding commands") from exc
        if not np.isfinite(solution).all() or np.any(solution < lower - 1e-6) or np.any(solution > upper + 1e-6):
            raise RuntimeError("LeRobot IK returned an invalid joint configuration")
        solution = np.clip(solution, lower, upper)
        if not history or not np.array_equal(solution, history[-1]):
            history = (history + [solution])[-4:]
        filtered = history[-1].copy() if len(history) < 4 else np.asarray(history).T @ self.FILTER_WEIGHTS
        filtered = np.clip(filtered, lower, upper)
        self._history = deque(history, maxlen=4)
        self._previous = filtered.copy()
        self._active = active
        return {n: float(filtered[i]) for n, i in zip(self.joint_names, self.indices, strict=True)}


class LeRobotIKSolver(G1IKSolver, IKSolver):
    """Nonlinear IK engine with bounded commands and growing collision capacity."""

    def __init__(self, spec):
        self.spec = spec
        super().__init__(spec.urdf_path, spec.joints, spec.limits, spec.frames, spec.groups)
        self._allocate_constraints(16)

    def _allocate_constraints(self, rows):
        # Capacity grows geometrically; every supplied row is retained. Keeping
        # distant pairs out of the NLP avoids thousands of inactive slack DOFs.
        _, _, ca = load_dependencies()
        self._build_optimizer(ca)
        self.mode = self.opti.parameter()
        self.home = self.opti.parameter(self.model.nq)
        self._rows = rows
        self.hard_A = self.opti.parameter(self._rows, self.model.nq)
        self.hard_b = self.opti.parameter(self._rows)
        self.soft_A = self.opti.parameter(self._rows, self.model.nq)
        self.soft_b = self.opti.parameter(self._rows)
        self.slack = self.opti.variable(self._rows)
        self.slack_upper = self.opti.parameter()
        self.opti.subject_to(self.hard_A @ self.q <= self.hard_b)
        self.opti.subject_to(self.soft_A @ self.q - self.soft_b <= self.slack)
        self.opti.subject_to(self.opti.bounded(0, self.slack, self.slack_upper))
        # Short-circuit the unused pose objective during joint-space homing.
        # Multiplying its rotation-log derivative by zero can still yield NaN.
        objective = ca.if_else(self.mode > 0.5, ca.sumsqr(self.q - self.home), self.tracking_cost, True)
        self.opti.minimize(objective + 1e6 * ca.sumsqr(self.slack))

    def capture_state(self):
        return (
            [q.copy() for q in self._history],
            None if self._previous is None else self._previous.copy(),
            self._active.copy(),
        )

    def restore_state(self, state):
        history, previous, active = state
        self._history = deque((q.copy() for q in history), maxlen=4)
        self._previous = None if previous is None else previous.copy()
        self._active = active.copy()

    def _constraint_parameters(self, constraint):
        if len(constraint.b) > self._rows:
            raise ValueError("Too many configured collision rows")
        A, b = np.zeros((self._rows, self.model.nq)), np.ones(self._rows)
        A[: len(constraint.b), self.indices] = constraint.A
        b[: len(constraint.b)] = constraint.b
        return A, b

    def solve(self, problem):
        required = max(len(problem.hard.b), len(problem.soft.b))
        if required > self._rows:
            self._allocate_constraints(2 ** (required - 1).bit_length())
        active = {key: any(n in problem.active for n in names) for key, names in self.spec.groups.items()}
        self._active.intersection_update(key for key, value in active.items() if value)
        lower, upper = self.lower.copy(), self.upper.copy()
        try:
            self.lower[self.indices] = np.maximum(self.lower[self.indices], problem.lower)
            self.upper[self.indices] = np.minimum(self.upper[self.indices], problem.upper)
            if np.any(self.lower > self.upper):
                raise RuntimeError("Incompatible joint command bounds")
            A, b = self._constraint_parameters(problem.hard)
            self.opti.set_value(self.hard_A, A)
            self.opti.set_value(self.hard_b, b)
            A, b = self._constraint_parameters(problem.soft)
            self.opti.set_value(self.soft_A, A)
            self.opti.set_value(self.soft_b, b)
            self.opti.set_value(self.slack_upper, 1000.0 if np.any(problem.soft.A) else 0.0)
            self.opti.set_initial(self.slack, 0)
            homing = problem.joint_target is not None
            self.opti.set_value(self.mode, float(homing))
            self.opti.set_value(self.home, self._vector(problem.joint_target or problem.previous))
            if homing:
                self.reset()
            return super().solve(problem.measured, problem.previous, problem.targets, active)
        finally:
            self.lower, self.upper = lower, upper

    def accept(self, positions):
        self._previous = self._vector(positions)
