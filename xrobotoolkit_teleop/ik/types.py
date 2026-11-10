"""Native-library-free data exchanged by controllers and IK implementations."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .model import Distance, Geometry


class RobotState(Protocol):
    q: np.ndarray


class ModelDescription(Protocol):
    names: tuple[str, ...]
    nq: int
    nv: int
    njoints: int
    velocity_limit: np.ndarray


class CollisionModel(Protocol):
    objects: tuple[Geometry, ...]
    collision_pairs: tuple[tuple[int, int], ...]


class RobotAdapter(Protocol):
    """Numeric geometry interface; native solver objects remain in adapters.

    Full configurations have a leading world pose (7 entries), and full
    velocities have a leading world twist (6 entries). IKProblem vectors use
    only RobotSpec.joints, in their configured order.
    """

    state: RobotState
    model: ModelDescription
    collision_model: CollisionModel

    def joint_names(self) -> tuple[str, ...]: ...
    def get_joint_offset(self, name: str) -> int: ...
    def get_joint_v_offset(self, name: str) -> int: ...
    def get_joint(self, name: str) -> float: ...
    def set_joint(self, name: str, value: float) -> None: ...
    def set_joint_limits(self, name: str, lower: float, upper: float) -> None: ...
    def set_velocity_limit(self, name: str, value: float) -> None: ...
    def update_kinematics(self) -> None: ...
    def get_T_world_frame(self, name: str) -> np.ndarray: ...
    def joint_jacobian(self, name: str, frame: str) -> np.ndarray: ...
    def collision_placements(self, q: np.ndarray) -> np.ndarray: ...
    def distances(self) -> list[Distance]: ...


@dataclass(frozen=True)
class RobotSpec:
    urdf_path: str
    joints: tuple[str, ...]
    frames: dict[str, str]
    groups: dict[str, tuple[str, ...]]
    limits: dict[str, tuple[float, float]]
    initial_positions: dict[str, float]
    collision_pairs_path: str
    dt: float


@dataclass
class LinearConstraint:
    A: np.ndarray
    b: np.ndarray
    priority: str = "hard"


@dataclass(frozen=True)
class IKProblem:
    measured: dict[str, float]
    previous: dict[str, float]
    targets: dict[str, np.ndarray]
    active: frozenset[str]
    lower: np.ndarray
    upper: np.ndarray
    hard: LinearConstraint
    soft: LinearConstraint
    dt: float
    joint_target: dict[str, float] | None = None


@dataclass
class IKProposal:
    positions: dict[str, float]
    pending: object = None


@dataclass(frozen=True)
class VisualizationSnapshot:
    link_poses: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    positions: dict[str, float]
    status: str
    end_effectors: dict[str, np.ndarray] = field(default_factory=dict)


class IKBackend(Protocol):
    """propose() preserves model and accepted solver state on success or failure.

    accept() commits the final applied commands; reject() leaves state intact.
    reset() clears selected solver history. close() is idempotent and prevents
    further proposals, acceptance, rejection, or resets.
    """

    robot: RobotAdapter
    spec: RobotSpec

    def propose(self, problem: IKProblem) -> IKProposal: ...
    def accept(self, proposal: IKProposal, applied_positions: dict[str, float]) -> None: ...
    def reject(self, proposal: IKProposal) -> None: ...
    def reset(self, joints: frozenset[str] | None = None) -> None: ...
    def close(self) -> None: ...


@dataclass
class PoseTarget:
    T_world_frame: np.ndarray = field(default_factory=lambda: np.eye(4))
