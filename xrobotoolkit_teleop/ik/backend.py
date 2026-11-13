"""Common solver lifecycle and transactional backend behavior."""

from abc import ABC, abstractmethod

import numpy as np

from .types import IKProblem, IKProposal, RobotAdapter, RobotSpec


class IKSolver(ABC):
    """Solver engines operate on controlled joints and own only solver state."""

    @abstractmethod
    def solve(self, problem: IKProblem) -> dict[str, float]: ...

    def capture_state(self) -> object:
        return None

    def restore_state(self, state: object) -> None:
        """Stateless engines have nothing to restore."""

    def accept(self, positions: dict[str, float]) -> None:
        """Stateless engines need no reference update after command acceptance."""

    def reset(self, joints: frozenset[str] | None = None) -> None:
        """Stateless engines have no history to reset."""

    def close(self) -> None:
        self.reset()


class TransactionalBackend:
    """Own one model adapter and one solver engine under a shared lifecycle.

    Proposing restores both model configuration and accepted solver history,
    even on failure. Only accept() commits a proposal and applied commands.
    """

    def __init__(self, spec: RobotSpec, robot: RobotAdapter, solver: IKSolver):
        self.spec = spec
        self.robot = robot
        self.solver = solver
        self._closed = False

    def _require_open(self):
        if self._closed:
            raise RuntimeError("IK backend is closed")

    def _positions(self, positions):
        if set(positions) != set(self.spec.joints):
            raise RuntimeError("IK commands must contain exactly the configured controlled joints")
        try:
            values = np.array([positions[name] for name in self.spec.joints], dtype=float)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("IK commands must be numeric scalars") from exc
        if values.shape != (len(self.spec.joints),):
            raise RuntimeError("IK commands must be scalar joint values")
        if not np.isfinite(values).all():
            raise RuntimeError("Nonfinite IK joint commands")
        return values

    def propose(self, problem: IKProblem) -> IKProposal:
        self._require_open()
        configuration = self.robot.state.q.copy()
        state = self.solver.capture_state()
        try:
            for name, value in problem.measured.items():
                self.robot.set_joint(name, value)
            positions = self._solve_problem(problem)
            values = self._positions(positions)
            if np.any(values < problem.lower - 1e-6) or np.any(values > problem.upper + 1e-6):
                raise RuntimeError("IK solver returned commands outside the requested bounds")
            positions = {
                name: float(value)
                for name, value in zip(self.spec.joints, np.clip(values, problem.lower, problem.upper))
            }
            return IKProposal(positions, self.solver.capture_state())
        finally:
            try:
                self.solver.restore_state(state)
            finally:
                self.robot.state.q = configuration
                self.robot.update_kinematics()

    def _solve_problem(self, problem):
        """Prepare the adapter for engines that read its numeric kinematics."""
        self.robot.update_kinematics()
        return self.solver.solve(problem)

    def accept(self, proposal: IKProposal, applied_positions: dict[str, float]) -> None:
        self._require_open()
        values = self._positions(applied_positions)
        applied_positions = {name: float(value) for name, value in zip(self.spec.joints, values)}
        self.solver.restore_state(proposal.pending)
        self.solver.accept(applied_positions)
        for name, value in applied_positions.items():
            self.robot.set_joint(name, value)
        self.robot.update_kinematics()

    def reject(self, proposal: IKProposal) -> None:
        # propose() already restored the accepted state for every solver.
        self._require_open()

    def reset(self, joints: frozenset[str] | None = None) -> None:
        self._require_open()
        self.solver.reset(joints)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.solver.close()
