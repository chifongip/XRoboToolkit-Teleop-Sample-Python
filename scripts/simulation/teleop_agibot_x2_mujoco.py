"""Run the fixed-body Agibot X2 dual-arm VR simulation."""

import math
import os
from pathlib import Path
from typing import Literal

import tyro
from xrobotoolkit_teleop.common.x2_defaults import DEFAULTS
from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import (
    AgibotX2TeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def main(
    xml_path: str = os.path.join(ASSET_PATH, "agibot/x2/scene.xml"),
    robot_urdf_path: str = os.path.join(ASSET_PATH, "agibot/x2/x2_fixed.urdf"),
    scale_factor: float = DEFAULTS.scale_factor,
    visualize_placo: bool = False,
    arm_velocity_limit: float = DEFAULTS.arm_velocity_limit,
    hand_velocity_limit: float = DEFAULTS.hand_velocity_limit,
    collision_margin: float = DEFAULTS.collision_margin,
    collision_trigger: float = DEFAULTS.collision_trigger,
    with_omnihand: bool = False,
    control_frequency: float = DEFAULTS.control_frequency,
    return_velocity_limit: float = DEFAULTS.return_velocity_limit,
    ik_backend: Literal["placo", "lerobot"] = "placo",
):
    """Hold a grip to move its hand center; release both grips to reset the control heading.

    Args:
        xml_path: MuJoCo scene with fixed body and 14 arm position actuators.
        robot_urdf_path: Matching fixed-body URDF for IK.
        scale_factor: Robot translation per unit of VR controller translation.
        visualize_placo: Also open the Placo visualization in a browser.
        arm_velocity_limit: Placo tracking joint speed cap in rad/s, bounded by each joint's URDF limit.
        return_velocity_limit: Return-to-default joint target speed cap in rad/s; also bounded by URDF limits.
        hand_velocity_limit: Actuated finger target speed cap in rad/s, bounded by each joint's URDF limit.
        collision_margin: Placo minimum clearance between IK collision shapes, in meters.
        collision_trigger: Placo collision activation distance in meters; greater than margin.
        with_omnihand: Use the bundled X2 + Omnihand Pro scene; triggers open/close hands.
        control_frequency: Target control frequency in Hz; default 50. Physics steps fit each control period.
        ik_backend: Placo constrained IK or LeRobot G1-style nonlinear IK (joint bounds only).
    """
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be finite and positive")
    if with_omnihand:
        if Path(xml_path).resolve() != Path(ASSET_PATH, "agibot/x2/scene.xml").resolve():
            raise ValueError("Use either --with-omnihand or a custom --xml-path")
        xml_path = os.path.join(ASSET_PATH, "agibot/x2/scene_omnihand.xml")
        if Path(robot_urdf_path).resolve() == Path(ASSET_PATH, "agibot/x2/x2_fixed.urdf").resolve():
            robot_urdf_path = os.path.join(ASSET_PATH, "agibot/x2/x2_omnihand_ik.urdf")
    controller = AgibotX2TeleopController(
        xml_path,
        robot_urdf_path,
        scale_factor,
        visualize_placo,
        arm_velocity_limit=arm_velocity_limit,
        hand_velocity_limit=hand_velocity_limit,
        collision_margin=collision_margin,
        collision_trigger=collision_trigger,
        control_frequency=control_frequency,
        return_velocity_limit=return_velocity_limit,
        ik_backend=ik_backend,
    )
    print(f"IK backend: {ik_backend}")
    if ik_backend == "lerobot":
        print("LeRobot tracking uses joint position limits and filtering; arm speed/collision gates are disabled.")
    print("X2 simulation: controller motion -> left/right hand centers (position and orientation)")
    print("Look forward when engaging a grip: heading locks until both grips are released.")
    print(f"Control frequency target: {control_frequency:g} Hz")
    print("Hold the corresponding grip to move; release to hold. Close the viewer or press Ctrl-C to exit.")
    print("Hold Y to return both arms to the default pose; release to hold.")
    if with_omnihand:
        print("Left/right triggers close the corresponding Omnihand Pro hand; release to open (contacts disabled).")
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
