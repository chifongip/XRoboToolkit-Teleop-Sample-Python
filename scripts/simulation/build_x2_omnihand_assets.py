"""Rebuild the bundled, contact-disabled X2 + Omnihand Pro MJCF using local assets."""

from copy import deepcopy
from math import cos, sin
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


ASSETS = Path(__file__).resolve().parents[2] / "assets"


def mesh_center(path):
    """Geometric bounding-box center of a supplied binary STL, in mesh axes."""
    data = path.read_bytes()
    triangles = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * triangles:
        raise ValueError(f"Expected a binary STL: {path}")
    bounds = [[float("inf"), float("-inf")] for _ in range(3)]
    for triangle in struct.iter_unpack("<12fH", data[84:]):
        for start in (3, 6, 9):
            for axis in range(3):
                value = triangle[start + axis]
                bounds[axis][0] = min(bounds[axis][0], value)
                bounds[axis][1] = max(bounds[axis][1], value)
    return [(lower + upper) / 2 for lower, upper in bounds]


def set_hand_center(mjcf, urdf, side, center):
    """Add matching fixed frames with axes parallel to the wrist-roll frame."""
    name = f"{side}_hand_center"
    parent = f"{side}_wrist_roll_link"
    body = mjcf.find(f".//body[@name='{parent}']")
    previous = body.find(f"body[@name='{name}']")
    if previous is not None:
        body.remove(previous)
    xyz = " ".join(map(str, center))
    ET.SubElement(body, "body", name=name, pos=xyz)
    for tag, element_name in (("link", name), ("joint", f"{name}_joint")):
        previous = urdf.find(f"{tag}[@name='{element_name}']")
        if previous is not None:
            urdf.remove(previous)
    ET.SubElement(urdf, "link", name=name)
    joint = ET.SubElement(urdf, "joint", name=f"{name}_joint", type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=name)
    ET.SubElement(joint, "origin", xyz=xyz, rpy="0 0 0")


def urdf_quaternion(roll, pitch, yaw):
    """URDF fixed-axis RPY -> MuJoCo wxyz (independent of compiler eulerseq)."""
    cr, cp, cy = cos(roll / 2), cos(pitch / 2), cos(yaw / 2)
    sr, sp, sy = sin(roll / 2), sin(pitch / 2), sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def copy_urdf_inertial(body, link):
    """Copy the supplied URDF inertia into its MJCF body frame."""
    inertia = link.find("inertial")
    origin = inertia.find("origin")
    tensor = inertia.find("inertia")
    # MJCF fullinertia is expressed in body axes, unlike a URDF tensor whose
    # axes follow inertial/origin. Rotate it before MuJoCo diagonalizes it.
    rotation = quaternion_matrix(urdf_quaternion(*map(float, origin.get("rpy", "0 0 0").split())))
    matrix = [
        [float(tensor.get(k)) for k in row]
        for row in (("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz"))
    ]
    rotated = [
        [sum(rotation[i][k] * matrix[k][l] * rotation[j][l] for k in range(3) for l in range(3)) for j in range(3)]
        for i in range(3)
    ]
    previous = body.find("inertial")
    if previous is not None:
        body.remove(previous)
    ET.SubElement(
        body,
        "inertial",
        pos=origin.get("xyz", "0 0 0"),
        mass=inertia.find("mass").get("value"),
        fullinertia=" ".join(str(rotated[i][j]) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))),
    )


def quaternion_matrix(quaternion):
    w, x, y, z = quaternion
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def build_documents():
    """Assemble derived XML documents without modifying source or output files."""
    x2 = ASSETS / "agibot/x2"
    hand = ASSETS / "omnihand_pro"
    root = ET.parse(x2 / "x2_fixed.xml").getroot()
    ik = ET.parse(x2 / "x2_fixed.urdf").getroot()
    for side in ("left", "right"):
        mesh = root.find(f"asset/mesh[@name='{side}_wrist_roll_link']")
        center = mesh_center(x2 / "meshes" / Path(mesh.get("file")).name)
        set_hand_center(root, ik, side, center)
    default_model, default_ik = deepcopy(root), deepcopy(ik)
    root.set("model", "x2_fixed_omnihand_pro")
    # One compiler and one shared asset/default set; mesh files are relative to x2.
    root.find("compiler").set("meshdir", ".")
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", "meshes/" + mesh.get("file"))
    # Omnihand Pro mounts require the short v1.4 wrists, replacing the v1.3 end
    # geometry and its inertia. Keep wrist frames/joints for the existing arm IK.
    wrist_links = ET.parse(x2 / "omnihand_wrist_links.xml").getroot()
    for side in ("left", "right"):
        name = f"{side}_wrist_roll_link"
        link = wrist_links.find(f"link[@name='{name}']")
        mesh_file = Path(link.find("visual/geometry/mesh").get("filename")).name
        root.find(f"asset/mesh[@name='{name}']").set("file", f"meshes/{mesh_file}")
        body = root.find(f".//body[@name='{name}']")
        copy_urdf_inertial(body, link)
        body.find("geom").set("rgba", link.find("visual/material/color").get("rgba"))
        ik_link = ik.find(f"link[@name='{name}']")
        for tag in ("visual", "inertial"):
            ik_link.remove(ik_link.find(tag))
            ik_link.append(deepcopy(link.find(tag)))
    shared = ET.parse(hand / "MJCF/omnihand_pro_asset.xml").getroot()
    for asset in shared.find("asset"):
        if asset.tag == "mesh":
            asset.set("file", "../../omnihand_pro/meshes/" + asset.get("file"))
        root.find("asset").append(deepcopy(asset))
    for default in shared.find("default"):
        if default.get("class") == "hand_joint":
            # The supplied 0.1 rad limit margin repels fingers from zero and
            # saturates the motors at the open pose. Engage limits at endpoints.
            default.find("joint").set("margin", "0")
        # Keep source density/inertias, but prevent all hand contacts.
        for geom in default.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            if default.get("class") == "plastic_visual":
                geom.set("density", "0")
        root.find("default").append(deepcopy(default))
    original_joints = [j.get("name") for j in root.findall(".//body/joint")]
    original_actuators = [a.get("name") for a in root.find("actuator")]
    home = root.find("keyframe/key[@name='home']")
    qhome = dict(zip(original_joints, home.get("qpos").split()))
    chome = dict(zip(original_actuators, home.get("ctrl").split()))
    for side, prefix in (("left", "L"), ("right", "R")):
        model = ET.parse(hand / f"MJCF/omnihand_pro_{side}.xml").getroot()
        palm = model.find("worldbody/body")
        mount = wrist_links.find(f"joint[@name='{prefix}_palm_joint']")
        origin = mount.find("origin")
        palm.set("pos", origin.get("xyz"))
        quaternion = urdf_quaternion(*map(float, origin.get("rpy").split()))
        palm.set("quat", " ".join(map(str, quaternion)))
        # Express the palm mesh center in wrist axes, preserving target axes.
        center = mesh_center(hand / "meshes" / f"{prefix.lower()}_palm.STL")
        rotation = quaternion_matrix(quaternion)
        offset = list(map(float, origin.get("xyz").split()))
        center = [offset[i] + sum(rotation[i][j] * center[j] for j in range(3)) for i in range(3)]
        set_hand_center(root, ik, side, center)
        # The source MJCF infers palm mass from meshes. Use the matching Omnihand Pro
        # URDF's supplied inertia instead, avoiding visual/collision double mass.
        urdf = ET.parse(hand / f"omnihand_pro_{side}.urdf").getroot()
        copy_urdf_inertial(palm, urdf.find(f"link[@name='{prefix}_palm']"))
        parent = mount.find("parent").get("link")
        root.find(f".//body[@name='{parent}']").append(palm)
        for section in ("contact", "actuator", "equality"):
            target = root.find(section)
            if target is None:
                target = ET.SubElement(root, section)
            target.extend(deepcopy(list(model.find(section))))
    home.set("qpos", " ".join(qhome.get(j.get("name"), "0") for j in root.findall(".//body/joint")))
    home.set("ctrl", " ".join(chome.get(a.get("name"), "0") for a in root.find("actuator")))
    scene = ET.parse(x2 / "scene.xml")
    scene.getroot().find("include").set("file", "x2_omnihand.xml")
    scene.getroot().set("model", "X2 dual arm Omnihand Pro teleoperation")
    documents = {
        "x2_fixed.xml": default_model,
        "x2_fixed.urdf": default_ik,
        "x2_omnihand.xml": root,
        "x2_omnihand_ik.urdf": ik,
        "scene_omnihand.xml": scene.getroot(),
    }
    for document in documents.values():
        ET.indent(document)
    return documents


def build():
    for name, root in build_documents().items():
        ET.ElementTree(root).write(ASSETS / "agibot/x2" / name, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    build()
