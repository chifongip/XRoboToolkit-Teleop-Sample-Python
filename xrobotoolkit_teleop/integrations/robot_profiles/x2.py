"""X2 deployment geometry; generated resources live as long as the bridge."""

import xml.etree.ElementTree as ET
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from xrobotoolkit_teleop.ik.types import RobotSpec
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

from . import ControlGroup, RobotProfile


def deployment_urdf(omnihand=False):
    """Restore measured body DOFs while retaining fitted collision geometry."""
    assets = Path(ASSET_PATH) / "agibot/x2"
    root = ET.parse(assets / ("x2_omnihand_ik.urdf" if omnihand else "x2_fixed.urdf")).getroot()
    arm_names = [
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("type") == "revolute" and not joint.get("name").startswith(("L_", "R_"))
    ]
    source = ET.parse(assets / "x2_ultra.urdf").getroot()
    for joint in source.findall("joint"):
        if joint.get("type") != "fixed" and joint.get("name") not in arm_names:
            old = root.find(f"joint[@name='{joint.get('name')}']")
            if old is None:
                raise ValueError("X2 deployment model does not match the fixed-body model")
            root.remove(old)
            root.append(deepcopy(joint))
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str((assets / mesh.get("filename")).resolve()))
    return root, arm_names


@contextmanager
def load_profile(settings, omnihand=False):
    root, arms = deployment_urdf(omnihand)
    observed = tuple(j.get("name") for j in root.findall("joint") if j.get("type") != "fixed")
    if len(arms) != 14 or len(observed) != (69 if omnihand else 31):
        raise ValueError("X2 requires 14 controlled arms and the matching body/finger observations")
    limits = {}
    velocities = {}
    for name in arms:
        limit = root.find(f"joint[@name='{name}']/limit")
        limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        velocities[name] = float(limit.get("velocity"))
    groups = {
        side: ControlGroup(
            tuple(n for n in arms if n.startswith(f"{side}_")),
            f"{side}_hand_center",
            f"{side}_controller",
            f"{side}_grip",
        )
        for side in ("left", "right")
    }
    with TemporaryDirectory(prefix="x2-teleop-ik-") as directory:
        path = Path(directory) / "x2.urdf"
        pairs = Path(directory) / "collisions.json"
        ET.ElementTree(root).write(path)
        pairs.write_bytes(
            (
                Path(ASSET_PATH) / "agibot/x2" / ("collisions_omnihand.json" if omnihand else "collisions.json")
            ).read_bytes()
        )
        spec = RobotSpec(
            str(path),
            tuple(arms),
            {k: g.frame for k, g in groups.items()},
            {k: g.joints for k, g in groups.items()},
            limits,
            dict.fromkeys(observed, 0.0),
            str(pairs),
            1 / settings.control_frequency,
        )
        if omnihand:
            from xrobotoolkit_teleop.integrations.robot_profiles.omnihand import load_hands

            hands = load_hands(root)
        else:
            hands = None
        yield RobotProfile(spec, observed, groups, "torso_link", velocities, hands=hands)


def load_omnihand_profile(settings):
    return load_profile(settings, omnihand=True)
