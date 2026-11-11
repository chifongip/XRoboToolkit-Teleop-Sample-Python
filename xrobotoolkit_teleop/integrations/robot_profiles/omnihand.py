"""O12 actuator ordering, trigger poses and nonlinear passive coupling."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from xrobotoolkit_teleop.simulation.agibot_x2_hand_config import HAND_CLOSED
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


@dataclass(frozen=True)
class Hands:
    active: tuple[str, ...]
    observed: tuple[str, ...]
    groups: dict[str, tuple[str, ...]]
    limits: dict[str, tuple[float, float]]
    velocities: dict[str, float]
    closed: dict[str, float]
    couplings: tuple[tuple[str, str, tuple[float, ...]], ...]


def load_hands(root):
    groups, limits, velocities, closed, couplings = {}, {}, {}, {}, []
    for side, prefix in (("left", "L"), ("right", "R")):
        groups[side] = tuple(f"{prefix}_{suffix}_joint" for suffix in HAND_CLOSED)
        for name, value in zip(groups[side], HAND_CLOSED.values(), strict=True):
            limit = root.find(f"joint[@name='{name}']/limit")
            limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
            velocities[name] = float(limit.get("velocity"))
            closed[name] = -value if side == "right" and name in {"R_thumb_roll_joint", "R_thumb_abad_joint"} else value
        model = ET.parse(Path(ASSET_PATH) / "omnihand_pro/MJCF" / f"omnihand_pro_{side}.xml").getroot()
        for equality in model.findall("equality/joint"):
            coefficients = tuple(map(float, equality.get("polycoef", "0 1 0 0 0").split()))
            couplings.append((equality.get("joint1"), equality.get("joint2"), coefficients))
    observed = tuple(
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("type") != "fixed" and joint.get("name").startswith(("L_", "R_"))
    )
    return Hands(
        tuple(n for names in groups.values() for n in names),
        observed,
        groups,
        limits,
        velocities,
        closed,
        tuple(couplings),
    )
