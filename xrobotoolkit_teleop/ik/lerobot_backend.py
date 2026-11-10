"""LeRobot backend assembled from its model adapter and IPOPT solver engine."""

from .backend import TransactionalBackend
from .lerobot_solver import LeRobotIKSolver


class LeRobotBackend(TransactionalBackend):
    def __init__(self, spec):
        # Validate optional symbolic bindings before importing the model adapter.
        solver = LeRobotIKSolver(spec)
        from .pinocchio_model import PinocchioModel

        super().__init__(spec, PinocchioModel(spec), solver)
