"""Compatibility imports for the X2 external-state retargeter."""

from xrobotoolkit_teleop.integrations.retargeter import ExternalStateRetargeter, XRFrame
from xrobotoolkit_teleop.integrations.robot_profiles.x2 import deployment_urdf as deployment_urdf

X2XRFrame = XRFrame


class X2ArmRetargeter(ExternalStateRetargeter):
    """X2 convenience wrapper; new robots use ExternalStateRetargeter profiles."""

    def __init__(self, settings=None, *, ik_backend="placo"):
        from .x2_defaults import DEFAULTS

        super().__init__(DEFAULTS if settings is None else settings, robot="x2", ik_backend=ik_backend)
