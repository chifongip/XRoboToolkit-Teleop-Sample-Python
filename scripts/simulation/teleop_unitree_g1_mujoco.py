"""Run the fixed-base Unitree G1 dual-arm VR simulation."""

import tyro

from xrobotoolkit_teleop.common.teleop_settings import DEFAULTS, TrackingReference
from xrobotoolkit_teleop.ik import BackendName
from xrobotoolkit_teleop.simulation.unitree_g1_model import DEFAULT_G1_VARIANT, G1Variant
from xrobotoolkit_teleop.simulation.unitree_g1_teleop_controller import UnitreeG1TeleopController


def main(
    variant: G1Variant = DEFAULT_G1_VARIANT,
    xml_path: str | None = None,
    robot_urdf_path: str | None = None,
    scale_factor: float = DEFAULTS.scale_factor,
    arm_velocity_limit: float = DEFAULTS.arm_velocity_limit,
    collision_margin: float = DEFAULTS.collision_margin,
    collision_trigger: float = DEFAULTS.collision_trigger,
    control_frequency: float = DEFAULTS.control_frequency,
    return_velocity_limit: float = DEFAULTS.return_velocity_limit,
    ik_backend: BackendName = BackendName.placo,
    visualize_ik: bool = False,
    tracking_reference: TrackingReference = "fixed-heading",
):
    """Track the G1 palm poses with the corresponding VR controller grips."""
    controller = UnitreeG1TeleopController(
        xml_path,
        robot_urdf_path,
        variant=variant,
        scale_factor=scale_factor,
        arm_velocity_limit=arm_velocity_limit,
        collision_margin=collision_margin,
        collision_trigger=collision_trigger,
        control_frequency=control_frequency,
        return_velocity_limit=return_velocity_limit,
        ik_backend=ik_backend,
        visualize_ik=visualize_ik,
        tracking_reference=tracking_reference,
    )
    print(f"G1 {variant} | IK backend: {controller.ik_backend} | Tracking reference: {tracking_reference}")
    if tracking_reference == "fixed-heading":
        print("Look forward when engaging a grip: heading locks until both grips are released.")
    else:
        print("Waist-relative tracking: release both grips before engaging. Head pose is unused.")
    print(f"Control frequency target: {control_frequency:g} Hz")
    print("Hold a grip to move its palm. Release to hold. Close the viewer or press Ctrl-C to exit.")
    print("Hold Y to return both arms to home. Release both grips after homing to resume tracking.")
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
