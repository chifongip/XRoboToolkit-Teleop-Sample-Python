import tyro
from xrobotoolkit_teleop.hardware.piper_teleop_controller import (
    DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG,
    DEFAULT_DUAL_PIPER_URDF_PATH,
    PiperTeleopController,
)


def main(
    robot_urdf_path: str = DEFAULT_DUAL_PIPER_URDF_PATH,
    scale_factor: float = 1.5,
    enable_camera: bool = False,
    enable_log_data: bool = True,
    visualize_placo: bool = True,
    control_rate_hz: int = 50,
    log_dir: str = "logs/dual_piper",
):
    """
    Main function to run the dual Piper hardware teleoperation.

    Connects to two Piper arms over CAN (left: can0, right: can1 by default),
    initialises the XR client, and runs the IK-based teleoperation loop.
    """
    controller = PiperTeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG,
        scale_factor=scale_factor,
        enable_camera=enable_camera,
        enable_log_data=enable_log_data,
        visualize_placo=visualize_placo,
        control_rate_hz=control_rate_hz,
        log_dir=log_dir,
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
