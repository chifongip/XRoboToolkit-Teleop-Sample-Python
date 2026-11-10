"""Backend-neutral IK contracts and lazy backend discovery."""

from .registry import BACKENDS, BackendName, create_backend
from .backend import IKSolver, TransactionalBackend
from .types import IKBackend, IKProblem, IKProposal, LinearConstraint, RobotAdapter, RobotSpec, VisualizationSnapshot

__all__ = [
    "BACKENDS",
    "BackendName",
    "create_backend",
    "IKSolver",
    "TransactionalBackend",
    "IKBackend",
    "IKProblem",
    "IKProposal",
    "LinearConstraint",
    "RobotAdapter",
    "RobotSpec",
    "VisualizationSnapshot",
]
