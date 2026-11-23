"""Placo backend assembled from its model adapter and QP solver engine."""

from .backend import TransactionalBackend
from .placo_model import PlacoModel
from .placo_solver import PlacoIKSolver


class PlacoBackend(TransactionalBackend):
    def __init__(self, spec, *, preferred_joint=None):
        robot = PlacoModel(spec)
        super().__init__(spec, robot, PlacoIKSolver(spec, robot, preferred_joint=preferred_joint))
