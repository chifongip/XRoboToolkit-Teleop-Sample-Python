"""Validated settings shared by external-state retargeting and robot demos."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TeleopSettings:
    arm_velocity_limit: float = 8.0
    return_velocity_limit: float = 1.0
    hand_velocity_limit: float = 2.0
    control_frequency: float = 50.0
    scale_factor: float = 1.0
    collision_margin: float = 0.02
    collision_trigger: float = 0.08
    tracking_timeout_s: float = 0.25
    grip_threshold: float = 0.9

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.collision_trigger <= self.collision_margin:
            raise ValueError("collision_trigger must exceed collision_margin")
        if self.grip_threshold >= 1:
            raise ValueError("grip_threshold must be less than one")


DEFAULTS = TeleopSettings()
