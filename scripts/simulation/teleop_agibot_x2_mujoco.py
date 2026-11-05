"""Run the fixed-body Agibot X2 dual-arm VR simulation."""

import os
import math
from pathlib import Path

import tyro

from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def main(
    xml_path: str = os.path.join(ASSET_PATH, "agibot/x2/scene.xml"),
    robot_urdf_path: str = os.path.join(ASSET_PATH, "agibot/x2/x2_fixed.urdf"),
    scale_factor: float = 1.0,
    visualize_placo: bool = False,
    arm_velocity_limit: float = 8.0,
    hand_velocity_limit: float = 2.0,
    collision_margin: float = 0.02,
    collision_trigger: float = 0.08,
    with_omnihand: bool = False,
    control_frequency: float = 50.0,
    return_velocity_limit: float = 1.0,
):
    """Control each hand center in the VR tracking world frame while gripping; release to hold.

    Args:
        xml_path: MuJoCo scene with fixed body and 14 arm position actuators.
        robot_urdf_path: Matching fixed-body URDF for Placo IK.
        scale_factor: Robot translation per unit of VR controller translation.
        visualize_placo: Also open the Placo visualization in a browser.
        arm_velocity_limit: Arm IK joint speed cap in rad/s, bounded by each joint's URDF limit.
        return_velocity_limit: Return-to-default joint target speed cap in rad/s; also bounded by arm/URDF limits.
        hand_velocity_limit: Actuated finger target speed cap in rad/s, bounded by each joint's URDF limit.
        collision_margin: Minimum clearance between IK collision shapes, in meters.
        collision_trigger: Distance at which IK collision constraints activate, in meters; greater than margin.
        with_omnihand: Use the bundled X2 + Omnihand Pro scene; triggers open/close hands.
        control_frequency: Target control frequency in Hz; default 50. Physics steps fit each control period.
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
    )
    print("X2 simulation: VR tracking world-frame controllers -> left/right hand centers (position and orientation)")
    print(f"Control frequency target: {control_frequency:g} Hz")
    print("Hold the corresponding grip to move; release to hold. Close the viewer or press Ctrl-C to exit.")
    print("Hold Y to return both arms to the default pose; release to hold. Collision checks remain active.")
    if with_omnihand:
        print("Left/right triggers close the corresponding Omnihand Pro hand; release to open (contacts disabled).")
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
