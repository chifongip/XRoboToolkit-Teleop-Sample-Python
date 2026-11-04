"""Run the fixed-body Agibot X2 dual-arm VR simulation."""

import os
import math

import tyro

from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def main(
    xml_path: str = os.path.join(ASSET_PATH, "agibot/x2/scene.xml"),
    robot_urdf_path: str = os.path.join(ASSET_PATH, "agibot/x2/x2_fixed.urdf"),
    scale_factor: float = 1.0,
    visualize_placo: bool = False,
    velocity_limit: float = 8.0,
):
    """Control each wrist in the VR tracking world frame while gripping; release to hold.

    Args:
        xml_path: MuJoCo scene with fixed body and 14 arm position actuators.
        robot_urdf_path: Matching fixed-body URDF for Placo IK.
        scale_factor: Robot translation per unit of VR controller translation.
        visualize_placo: Also open the Placo visualization in a browser.
        velocity_limit: IK joint speed cap in rad/s, bounded by each joint's URDF limit.
    """
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be finite and positive")
    controller = AgibotX2TeleopController(
        xml_path, robot_urdf_path, scale_factor, visualize_placo, velocity_limit=velocity_limit
    )
    print("X2 simulation: VR tracking world-frame left/right controller -> left/right wrist (position and orientation)")
    print("Hold the corresponding grip to move; release to hold. Close the viewer or press Ctrl-C to exit.")
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
