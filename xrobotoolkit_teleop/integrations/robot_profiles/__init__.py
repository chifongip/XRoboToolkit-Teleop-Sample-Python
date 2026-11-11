"""Lazy robot profiles for external-state joint-position teleoperation."""

from dataclasses import dataclass
from importlib import import_module

from xrobotoolkit_teleop.ik.types import RobotSpec


@dataclass(frozen=True)
class ControlGroup:
    joints: tuple[str, ...]
    frame: str
    controller: str
    grip: str


@dataclass(frozen=True)
class RobotProfile:
    spec: RobotSpec
    observed_joints: tuple[str, ...]
    groups: dict[str, ControlGroup]
    reference_frame: str
    velocities: dict[str, float]
    headset: str = "headset"
    home_button: str = "Y"

    def validate(self):
        import numpy as np

        joints = self.spec.joints
        grouped = [joint for group in self.groups.values() for joint in group.joints]
        if not joints or len(set(joints)) != len(joints) or set(joints) != set(grouped):
            raise ValueError("Profile groups must cover every controlled joint")
        if len(grouped) != len(set(grouped)) or not self.groups or any(not g.joints for g in self.groups.values()):
            raise ValueError("Profile control groups must be nonempty and disjoint")
        if len(set(self.observed_joints)) != len(self.observed_joints) or not set(joints) <= set(self.observed_joints):
            raise ValueError("Profile must observe every controlled joint exactly once")
        if self.spec.groups != {key: group.joints for key, group in self.groups.items()}:
            raise ValueError("Profile and IK groups disagree")
        if self.spec.frames != {key: group.frame for key, group in self.groups.items()}:
            raise ValueError("Profile and IK frames disagree")
        if set(self.spec.limits) != set(joints) or set(self.velocities) != set(joints):
            raise ValueError("Profile limits must cover controlled joints")
        if set(self.spec.initial_positions) != set(self.observed_joints):
            raise ValueError("Profile initial positions must cover observed joints")
        if not np.isfinite(list(self.spec.initial_positions.values())).all():
            raise ValueError("Profile initial positions must be finite")
        if not np.isfinite(self.spec.dt) or self.spec.dt <= 0:
            raise ValueError("Profile control period must be finite and positive")
        if not self.reference_frame or not self.headset or not self.home_button:
            raise ValueError("Profile reference frame and XR device names must be nonempty")
        if any(not g.frame or not g.controller or not g.grip for g in self.groups.values()):
            raise ValueError("Profile group frames and XR mappings must be nonempty")
        for name in joints:
            lower, upper = self.spec.limits[name]
            if (
                not np.isfinite([lower, upper, self.velocities[name]]).all()
                or lower >= upper
                or self.velocities[name] <= 0
            ):
                raise ValueError("Invalid profile joint limits")


# A factory(settings) returns a context manager yielding a RobotProfile.
PROFILES = {"x2": "xrobotoolkit_teleop.integrations.robot_profiles.x2:load_profile"}


def load_profile(name, settings):
    if name not in PROFILES:
        raise ValueError(f"Unknown robot {name!r}; choose from {', '.join(PROFILES)}")
    module, symbol = PROFILES[name].split(":")
    return getattr(import_module(module), symbol)(settings)
