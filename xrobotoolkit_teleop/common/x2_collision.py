"""Shared X2 collision constraints and bounded command-path sampling."""

import numpy as np
from numpy.polynomial.polynomial import polyval


class X2CollisionMixin:
    COLLISION_CHECK_STEP_RAD = 0.02
    MAX_COLLISION_SAMPLES = 4096

    def _update_collision_constraint(self, constraint):
        """Build the native clearance inequalities from cached exact witnesses.

        X2 has a masked floating base and scalar hinge joints, so velocity
        columns 6: map directly to configuration columns 7:. Nonlinear command
        validation and the finite-difference penetration recovery stay separate.
        """
        distances, jacobian = self.collision_checker.linearized_clearance(self.collision_trigger)
        configuration = self.placo_robot.state.q
        if distances.size == 0:
            # Placo rejects empty half-space matrices. A single tautology keeps
            # the persistent constraint valid until a nearby pair appears.
            constraint.A = np.zeros((1, len(configuration)))
            constraint.b = np.ones(1)
            return
        matrix = np.zeros((len(distances), len(configuration)))
        matrix[:, 7:] = -jacobian[:, 6:]
        constraint.A = matrix
        constraint.b = distances - self._solver_collision_margin + matrix @ configuration

    def _add_recovery_constraint(self, distances, solver=None):
        """Keep each nearby pair from worsening while soft avoidance separates it.

        Finite differences include signed primitive penetration, where separation of
        several pairs must be coordinated. Only gripped arms may move.
        """
        robot = self.placo_robot
        previous = robot.state.q.copy()
        nearby = distances < self.collision_trigger
        jacobian = np.zeros((np.count_nonzero(nearby), len(previous)))
        epsilon = 1e-5
        try:
            for name, joints in self.arm_joints.items():
                if not self.active[name]:
                    continue
                for joint in joints:
                    offset = robot.get_joint_offset(joint)
                    plus, minus = previous.copy(), previous.copy()
                    plus[offset] += epsilon
                    minus[offset] -= epsilon
                    robot.state.q = plus
                    d_plus = self.collision_checker.distances()[nearby]
                    robot.state.q = minus
                    d_minus = self.collision_checker.distances()[nearby]
                    jacobian[:, offset] = (d_plus - d_minus) / (2 * epsilon)
        finally:
            robot.state.q = previous
            robot.update_kinematics()
        # d(q + dq) >= min(d(q), margin), expressed as A q_new <= b.
        current = distances[nearby]
        floor = np.minimum(current, self.collision_margin)
        constraint = (self.solver if solver is None else solver).add_joint_space_half_spaces_constraint(
            -jacobian, current - floor - jacobian @ previous
        )
        constraint.configure("x2_recovery_clearance", "hard")
        return constraint

    def _check_collision_clearance(self):
        if np.any(self.collision_checker.distances() < self.collision_margin - 1e-6):
            raise RuntimeError("X2 self-collision clearance violated")

    def _collision_motion_samples(self, start, target, coupled_hands):
        """Sample coupled finger motion while preserving measured tracking error.

        A passive joint follows its polynomial plus a linearly decaying initial
        tracking error. Derivative bounds keep every joint's sample spacing
        within COLLISION_CHECK_STEP_RAD, including nonlinear passive motion.
        """
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
            raise RuntimeError("non-finite X2 collision configuration")
        delta = target - start
        rates = np.abs(delta)
        max_abs = np.maximum(np.abs(start), np.abs(target))
        corrections = []
        if coupled_hands:
            for passive, source, coefficients in self.hand_couplings:
                initial_error = start[passive] - polyval(start[source], coefficients)
                final_error = target[passive] - polyval(target[source], coefficients)
                slope = polyval(max_abs[source], np.arange(1, len(coefficients)) * np.abs(coefficients[1:]))
                rates[passive] = slope * rates[source] + abs(final_error - initial_error)
                max_abs[passive] = polyval(max_abs[source], np.abs(coefficients)) + max(
                    abs(initial_error), abs(final_error)
                )
                corrections.append((passive, source, coefficients, initial_error, final_error))
        max_rate = float(np.max(rates[7:]))
        if not np.isfinite(max_rate) or max_rate > self.MAX_COLLISION_SAMPLES * self.COLLISION_CHECK_STEP_RAD:
            raise RuntimeError("X2 collision motion exceeds sampling budget")
        steps = max(1, int(np.ceil(max_rate / self.COLLISION_CHECK_STEP_RAD)))
        fractions = np.linspace(0, 1, steps + 1)[1:]
        samples = start + fractions[:, None] * delta
        for passive, source, coefficients, initial_error, final_error in corrections:
            samples[:, passive] = (
                polyval(samples[:, source], coefficients) + (1 - fractions) * initial_error + fractions * final_error
            )
        if not np.all(np.isfinite(samples)):
            raise RuntimeError("non-finite X2 collision motion")
        return samples
