"""Adding a backend changes this registry, not the controller."""

from dataclasses import dataclass
from enum import Enum
from importlib import import_module

from .types import IKBackend, RobotSpec


@dataclass(frozen=True)
class BackendRegistration:
    factory: str
    description: str
    capabilities: frozenset[str] = frozenset({"pose", "joint_target", "linear_constraints"})


BACKENDS = {
    "placo": BackendRegistration("xrobotoolkit_teleop.ik.placo_backend:PlacoBackend", "Differential QP IK"),
    "lerobot": BackendRegistration("xrobotoolkit_teleop.ik.lerobot_backend:LeRobotBackend", "Nonlinear IPOPT IK"),
}
BackendName = Enum("BackendName", {name: name for name in BACKENDS}, type=str)


def create_backend(name: str | Enum, spec: RobotSpec) -> IKBackend:
    name = name.value if isinstance(name, Enum) else name
    if name not in BACKENDS:
        raise ValueError(f"Unknown ik_backend {name!r}; choose from {', '.join(BACKENDS)}")
    registration = BACKENDS[name]
    if not {"pose", "joint_target", "linear_constraints"} <= registration.capabilities:
        raise ValueError(f"Backend {name!r} does not support the required objectives and collision constraints")
    module, symbol = registration.factory.split(":")
    try:
        factory = getattr(import_module(module), symbol)
        return factory(spec)
    except ImportError as exc:
        raise ImportError(f"Cannot initialize {name} IK: {exc}. See docs/lerobot_ik.md") from exc
