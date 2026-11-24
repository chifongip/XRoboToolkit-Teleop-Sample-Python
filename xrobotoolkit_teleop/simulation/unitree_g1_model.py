"""Names and arm joint groups shared by G1 asset generation and teleoperation."""

from pathlib import Path
from typing import Literal, get_args

from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


G1Variant = Literal["23dof", "29dof"]
DEFAULT_G1_VARIANT: G1Variant = "29dof"
G1_ASSETS = Path(ASSET_PATH) / "unitree/g1"


def g1_model_stem(variant: G1Variant) -> str:
    if variant not in get_args(G1Variant):
        raise ValueError("variant must be 23dof or 29dof")
    return f"g1_{variant}_rev_1_0"


def g1_arm_joints(variant: G1Variant) -> dict[str, tuple[str, ...]]:
    g1_model_stem(variant)
    joints = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
    if variant == "29dof":
        joints += ("wrist_pitch", "wrist_yaw")
    return {f"{side}_arm": tuple(f"{side}_{name}_joint" for name in joints) for side in ("left", "right")}
