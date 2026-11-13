"""LeRobot backend assembled from its model adapter and IPOPT solver engine."""

from .backend import TransactionalBackend
from .lerobot_solver import LeRobotIKSolver


class LeRobotBackend(TransactionalBackend):
    def __init__(self, spec):
        # Validate optional symbolic bindings before importing the model adapter.
        solver = LeRobotIKSolver(spec)
        from .pinocchio_model import PinocchioModel

        super().__init__(spec, PinocchioModel(spec), solver)

    def _solve_problem(self, problem):
        # The symbolic solver reads problem.measured through its own model.
        # propose() still refreshes adapter caches when restoring configuration,
        # including after failures that may have modified numeric kinematics.
        return self.solver.solve_in_transaction(problem)
