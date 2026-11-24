"""Build fixed-base G1 teleop models from the bundled Unitree descriptions."""

from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from xrobotoolkit_teleop.simulation.unitree_g1_model import G1_ASSETS, G1Variant, g1_arm_joints, g1_model_stem
from xrobotoolkit_teleop.utils.mesh_fitting import (
    fit_primitive,
    mesh_triangles,
    quaternion_matrix,
    rotation_rpy,
    sliced_mesh_points,
    urdf_quaternion,
)


ROOT = Path(__file__).resolve().parents[2]
ASSETS = G1_ASSETS.resolve()
PALM_OFFSETS = {"23dof": (0.156, 0.010, -0.003), "29dof": (0.035, 0.005, -0.003)}
# The 29 DOF hand mesh origin expressed in the 23 DOF wrist mesh frame.
HAND_OFFSET_23 = (0.1215, 0.0055, 0.0015)
# Open the shoulders enough for both variants to clear the default IK margin.
HOME = {"shoulder_pitch": 0.3, "shoulder_roll": 0.45, "elbow": 1.0}
# Match the X2 position-servo gains for the corresponding joint roles.
GAINS = {
    "shoulder_pitch": (500, 32),
    "shoulder_roll": (500, 32),
    "shoulder_yaw": (500, 32),
    "elbow": (300, 20),
    "wrist_roll": (60, 6),
    "wrist_pitch": (60, 6),
    "wrist_yaw": (120, 6),
}


def numbers(values):
    return " ".join(f"{value:.12g}" for value in values)


def fixed_joint(urdf, name, parent, child, xyz):
    joint = ET.SubElement(urdf, "joint", name=name, type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz=xyz, rpy="0 0 0")


def add_palms(mjcf, urdf, variant):
    hands = {}
    for side, sign in (("left", 1), ("right", -1)):
        name = f"{side}_palm_center"
        x, y, z = PALM_OFFSETS[variant]
        offset = np.array([x, sign * y, z])
        if variant == "23dof":
            hand = parent = f"{side}_wrist_roll_rubber_hand"
            mj_offset = offset
            rotation = None
        else:
            hand = f"{side}_rubber_hand"
            parent = f"{side}_wrist_yaw_link"
            mount = urdf.find(f"joint[@name='{side}_hand_palm_joint']/origin")
            rotation = urdf_quaternion(*map(float, mount.get("rpy").split()))
            mj_offset = np.fromstring(mount.get("xyz"), sep=" ") + quaternion_matrix(rotation) @ offset
        body = ET.SubElement(mjcf.find(f".//body[@name='{parent}']"), "body", name=name, pos=numbers(mj_offset))
        if rotation is not None:
            body.set("quat", numbers(rotation))
        ET.SubElement(urdf, "link", name=name)
        fixed_joint(urdf, f"{name}_joint", hand, name, numbers(offset))
        hands[side] = hand
    return hands


def transform(origin, triangles):
    if origin is None:
        return triangles
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    rotation = np.asarray(quaternion_matrix(urdf_quaternion(*rpy)))
    return triangles @ rotation.T + xyz


def collision_meshes(urdf):
    """Group fixed covers with their housing so each body region keeps one pair entry."""
    meshes = {}
    for link in urdf.findall("link"):
        parts = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is not None:
                triangles = mesh_triangles(ASSETS / mesh.get("filename"))
                scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                parts.append(transform(visual.find("origin"), triangles * scale))
        if parts:
            meshes[link.get("name")] = np.concatenate(parts)
    meshes["pelvis_contour_link"] = np.concatenate((meshes["pelvis_contour_link"], meshes.pop("pelvis")))
    logo_joint = next(joint for joint in urdf.findall("joint") if joint.find("child").get("link") == "logo_link")
    meshes["torso_link"] = np.concatenate(
        (meshes["torso_link"], transform(logo_joint.find("origin"), meshes.pop("logo_link")))
    )
    return meshes


def primitive_bounds(shape):
    kind, center, basis, size = shape
    if kind == "box":
        extent = np.abs(basis) @ (size / 2)
    else:
        extent = size[0] * np.hypot(basis[:, 0], basis[:, 1]) + size[1] / 2 * np.abs(basis[:, 2])
    return center - extent, center + extent


def covered_by_primitives(points, primitives):
    covered = np.zeros(len(points), dtype=bool)
    for kind, center, basis, size in primitives:
        local = np.einsum("ij,jk->ik", points - center, basis)
        if kind == "box":
            inside = np.all(np.abs(local) <= size / 2, axis=1)
        else:
            inside = (np.hypot(local[:, 0], local[:, 1]) <= size[0]) & (np.abs(local[:, 2]) <= size[1] / 2)
        covered |= inside
    return covered


def fit_mesh(triangles, max_slices):
    """Select a conservative union from cuts along the mesh's local axes."""
    points = triangles.reshape(-1, 3)
    fits = [(fit_primitive(np.unique(points, axis=0))[1:],)]
    if max_slices == 1:
        return fits[0]
    axes = np.argsort(-np.ptp(points, axis=0))[:2]

    for axis in axes:
        basis = np.roll(np.eye(3), axis - 2, axis=0)
        local = triangles @ basis
        for count in range(2, max_slices + 1):
            fit = []
            for slab in sliced_mesh_points(local, count):
                _, kind, center, rotation, size = fit_primitive(slab)
                fit.append((kind, basis @ center, basis @ rotation, size))
            fits.append(tuple(fit))

    bounds = np.array([primitive_bounds(shape) for fit in fits for shape in fit])
    # Rank fitted unions over the same sampling domain. The coverage guarantee
    # comes from clipping complete triangles and enclosing each resulting slab.
    samples = np.random.default_rng(0).uniform(bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0), (16384, 3))
    return min(
        fits,
        key=lambda fit: np.count_nonzero(covered_by_primitives(samples, fit)) * (1 + 0.06 * (len(fit) - 1)),
    )


def fit_torso(triangles):
    """Place ten height bands where joining adjacent slabs adds the least volume."""
    slabs = list(sliced_mesh_points(triangles, 32, with_limits=True))
    lower, upper = np.array([(points.min(axis=0)[:2], points.max(axis=0)[:2]) for _, _, points in slabs]).transpose(
        1, 0, 2
    )
    heights = np.r_[slabs[0][0], [top for _, top, _ in slabs]]
    count, bands = len(slabs), 10
    boxes = {}
    volumes = np.full((count, count + 1), np.inf)
    for start in range(count):
        for end in range(start + 1, count + 1):
            lo = np.r_[lower[start:end].min(axis=0), heights[start]]
            hi = np.r_[upper[start:end].max(axis=0), heights[end]]
            size = hi - lo + 0.002
            boxes[start, end] = ("box", (lo + hi) / 2, np.eye(3), size)
            volumes[start, end] = np.prod(size)

    costs = np.full((bands + 1, count + 1), np.inf)
    costs[0, 0] = 0
    previous = np.zeros_like(costs, dtype=int)
    for band in range(1, bands + 1):
        for end in range(band, count + 1):
            candidates = costs[band - 1, :end] + volumes[:end, end]
            start = np.argmin(candidates)
            previous[band, end] = start
            costs[band, end] = candidates[start]

    fit = []
    end = count
    for band in range(bands, 0, -1):
        start = previous[band, end]
        fit.append(boxes[start, end])
        end = start
    return tuple(reversed(fit))


@lru_cache
def fit_rubber_hand(path):
    """Use the standalone 29 DOF hand mesh as the source for both variants."""
    return fit_mesh(mesh_triangles(path), max_slices=4)


def fit_wrist_and_hand(triangles, hand, side):
    x, y, z = HAND_OFFSET_23
    offset = np.array([x, y if side == "left" else -y, z])
    # The 23 DOF mesh also contains the forearm. Clip that section at the
    # rubber-hand mount and fit it independently of the transferred hand.
    basis = np.eye(3)[:, [1, 2, 0]]
    local = triangles @ basis
    cuts = np.linspace(local[:, :, 2].min(), x, 3)
    forearm = []
    for slab in sliced_mesh_points(local, cuts=cuts):
        _, kind, center, rotation, size = fit_primitive(slab)
        forearm.append((kind, basis @ center, basis @ rotation, size))
    return tuple(forearm) + tuple((kind, center + offset, rotation, size) for kind, center, rotation, size in hand)


def fit_collisions(urdf):
    """Fit the visible shells with a limited number of box and cylinder pieces."""
    meshes = collision_meshes(urdf)
    budgets = {
        "pelvis_contour_link": 3,
        "hip_pitch_link": 2,
        "hip_roll_link": 3,
        "hip_yaw_link": 3,
        "knee_link": 4,
        "ankle_roll_link": 3,
        "shoulder_roll_link": 2,
        "shoulder_yaw_link": 2,
        "elbow_link": 3,
        "wrist_pitch_link": 2,
    }
    head = meshes.pop("head_link")
    bottom, top = head[:, :, 2].min(), head[:, :, 2].max()
    # The wide bottom collar is fixed against the torso and shares the
    # torso's adjacent-link policy at the shoulder housings.
    head_cuts = [bottom, bottom + 0.032, bottom + 0.078, bottom + 0.113, bottom + 0.156, top]
    collar, *head_shell = [fit_primitive(slab)[1:] for slab in sliced_mesh_points(head, cuts=head_cuts)]
    head_origin = urdf.find("joint[@name='head_joint']/origin")
    kind, center, basis, size = collar
    rpy = np.fromstring(head_origin.get("rpy"), sep=" ")
    rotation = np.asarray(quaternion_matrix(urdf_quaternion(*rpy)))
    chest = fit_torso(meshes.pop("torso_link"))
    fits = {
        "head_link": head_shell,
        "torso_link": chest + ((kind, transform(head_origin, center), rotation @ basis, size),),
    }
    for name, triangles in meshes.items():
        part = name.removeprefix("left_").removeprefix("right_")
        if part in {"wrist_roll_rubber_hand", "rubber_hand"}:
            side = name.split("_", 1)[0]
            hand = fit_rubber_hand(ASSETS / "meshes" / f"{side}_rubber_hand.STL")
            fits[name] = fit_wrist_and_hand(triangles, hand, side) if part == "wrist_roll_rubber_hand" else hand
        else:
            fits[name] = fit_mesh(triangles, budgets.get(part, 1))

    for link in urdf.findall("link"):
        for collision in link.findall("collision"):
            link.remove(collision)
        for kind, center, basis, size in fits.get(link.get("name"), ()):
            primitive = ET.SubElement(link, "collision")
            ET.SubElement(primitive, "origin", xyz=numbers(center), rpy=numbers(rotation_rpy(basis)))
            geometry = ET.SubElement(primitive, "geometry")
            if kind == "cylinder":
                ET.SubElement(geometry, kind, radius=f"{size[0]:.12g}", length=f"{size[1]:.12g}")
            else:
                ET.SubElement(geometry, kind, size=numbers(size))


def collision_pairs(urdf, groups, hands):
    arms = {}
    for side, hand in hands.items():
        links = [urdf.find(f"joint[@name='{joint}']/child").get("link") for joint in groups[f"{side}_arm"]]
        if hand not in links:
            links.append(hand)
        arms[side] = links
    moving = set(arms["left"] + arms["right"])
    fixed = [
        link.get("name")
        for link in urdf.findall("link")
        if link.find("collision") is not None and link.get("name") not in moving
    ]
    pairs = list(product(arms["left"], arms["right"]))
    for links in arms.values():
        for index, link in enumerate(links):
            # Joint housings overlap locally, including the short 29 DOF wrist chain.
            pairs.extend((link, other) for other in links[index + 3 :] if "_wrist_" not in link)
            pairs.extend((link, body) for body in fixed if not (body == "torso_link" and index < 2))
    return pairs


def configure_scene(mjcf, urdf, arm_names):
    source_actuators = mjcf.find("actuator")
    source_actuators.clear()
    mjcf.insert(1, ET.Element("option", timestep="0.001", integrator="implicitfast"))
    pelvis = mjcf.find("worldbody/body[@name='pelvis']")
    ET.SubElement(urdf, "link", name="world")
    fixed_joint(urdf, "fixed_simulation_base", "world", "pelvis", pelvis.get("pos"))
    urdf_joints = {joint.get("name"): joint for joint in urdf.findall("joint")}
    for name, joint in urdf_joints.items():
        if name not in arm_names:
            joint.set("type", "fixed")
            for child in list(joint):
                if child.tag in {"axis", "limit", "dynamics"}:
                    joint.remove(child)

    home = []
    for body in pelvis.iter("body"):
        for joint in body.findall("joint"):
            name = joint.get("name")
            if name not in arm_names:
                body.remove(joint)
                continue
            limits = urdf_joints[name].find("limit")
            lower, upper = map(float, joint.get("range").split())
            lower = max(lower, float(limits.get("lower")))
            upper = min(upper, float(limits.get("upper")))
            effort = float(limits.get("effort"))
            side, kind = name.removesuffix("_joint").split("_", 1)
            kp, kv = GAINS[kind]
            joint.set("range", numbers((lower, upper)))
            joint.set("armature", "0.03")
            ET.SubElement(
                source_actuators,
                "position",
                name=name,
                joint=name,
                kp=str(kp),
                kv=str(kv),
                ctrlrange=numbers((lower, upper)),
                forcerange=numbers((-effort, effort)),
            )
            home.append(-HOME[kind] if kind == "shoulder_roll" and side == "right" else HOME.get(kind, 0.0))

    for body in pelvis.iter("body"):
        for geom in body.findall("geom"):
            # Stock MJCFs duplicate visual meshes for collision and add separate contact shapes.
            if geom.get("contype") != "0":
                body.remove(geom)

    # Reuse the same mocap target triads as the X2 viewer.
    source = ET.parse(ROOT / "assets/agibot/x2/scene.xml")
    worldbody = mjcf.find("worldbody")
    for side in ("left", "right"):
        worldbody.append(source.find(f"worldbody/body[@name='{side}_target']"))
    keyframe = ET.SubElement(mjcf, "keyframe")
    ET.SubElement(keyframe, "key", name="home", qpos=numbers(home), ctrl=numbers(home))


def build(variant: G1Variant):
    stem = g1_model_stem(variant)
    mjcf = ET.parse(ASSETS / f"{stem}.xml").getroot()
    urdf = ET.parse(ASSETS / f"{stem}.urdf").getroot()
    groups = g1_arm_joints(variant)
    arm_names = {joint for group in groups.values() for joint in group}
    configure_scene(mjcf, urdf, arm_names)
    hands = add_palms(mjcf, urdf, variant)
    fit_collisions(urdf)
    pairs = collision_pairs(urdf, groups, hands)
    model_dir = ASSETS / stem
    model_dir.mkdir(exist_ok=True)
    (model_dir / "collisions.json").write_text("[\n" + ",\n".join(f"  {json.dumps(pair)}" for pair in pairs) + "\n]\n")
    for mesh in urdf.iter("mesh"):
        mesh.set("filename", f"../{mesh.get('filename')}")
    for root, path, name in (
        (mjcf, ASSETS / f"{stem}_scene.xml", "model"),
        (urdf, model_dir / f"{stem}_ik.urdf", "name"),
    ):
        root.set(name, f"{stem}_teleop")
        ET.indent(root, space="  ")
        path.write_text(ET.tostring(root, encoding="unicode") + "\n")


if __name__ == "__main__":
    for variant in PALM_OFFSETS:
        build(variant)
