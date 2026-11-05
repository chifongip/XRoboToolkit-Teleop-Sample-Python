"""Rebuild the bundled, contact-disabled X2 + Omnihand Pro MJCF using local assets."""

from copy import deepcopy
from functools import lru_cache
from math import cos, sin
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np


ASSETS = Path(__file__).resolve().parents[2] / "assets"


@lru_cache(maxsize=None)
def mesh_triangles(path):
    """Read the bundled binary STL once; fitting happens only during generation."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Expected a binary STL: {path}")
    triangles = struct.unpack_from("<I", data, 80)[0]
    if not triangles or len(data) != 84 + 50 * triangles:
        raise ValueError(f"Expected a binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(data, dtype=dtype, offset=84)["vertices"].astype(float)


def mesh_bounds(path):
    points = mesh_triangles(path).reshape(-1, 3)
    return np.stack((points.min(axis=0), points.max(axis=0)), axis=1).tolist()


def mesh_center(path):
    return [(lower + upper) / 2 for lower, upper in mesh_bounds(path)]


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


def add_collision_geometry(urdf, omnihand=False):
    """Restore source X2 geometry as input to offline primitive fitting.

    Neighboring joint housings overlap by design; the pair file excludes them.
    The hand URDF chains and their per-link primitives are merged separately.
    """
    source = ET.parse(ASSETS / "agibot/x2/x2_ultra.urdf").getroot()
    wrists = ET.parse(ASSETS / "agibot/x2/omnihand_wrist_links.xml").getroot()
    for link in urdf.findall("link"):
        name = link.get("name")
        for old in link.findall("collision"):
            link.remove(old)
        original = source.find(f"link[@name='{name}']/collision")
        if omnihand and name.endswith("wrist_roll_link"):
            original = wrists.find(f"link[@name='{name}']/collision")
        if original is not None:
            link.append(deepcopy(original))
        elif name in ("waist_yaw_link", "waist_pitch_link"):
            # The upstream model omits waist collisions; include those stationary
            # housings using their matching visual meshes.
            collision = ET.SubElement(link, "collision")
            for tag in ("origin", "geometry"):
                collision.append(deepcopy(link.find(f"visual/{tag}")))


def collision_link_groups(urdf):
    arms = {
        side: [
            f"{side}_{part}_link"
            for part in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_yaw",
                "wrist_pitch",
                "wrist_roll",
            )
        ]
        for side in ("left", "right")
    }
    links = [link.get("name") for link in urdf.findall("link") if link.find("collision") is not None]
    hands = {
        side: [name for name in links if name.startswith(prefix + "_")]
        for prefix, side in (("L", "left"), ("R", "right"))
    }
    moving = set(arms["left"] + arms["right"] + hands["left"] + hands["right"])
    fixed = [name for name in links if name not in moving]
    return arms, hands, fixed


def collision_pairs(urdf):
    """Select moving-vs-body, opposite-arm, and non-neighbor same-arm pairs."""
    arms, _, fixed = collision_link_groups(urdf)
    pairs = [[a, b] for a in arms["left"] for b in arms["right"]]
    for links in arms.values():
        pairs.extend([a, b] for i, a in enumerate(links) for b in links[i + 3 :])
        for i, arm in enumerate(links):
            pairs.extend([arm, body] for body in fixed if not (body == "torso_link" and i < 2))
    return pairs


def convex_hull_2d(points):
    points = np.unique(points, axis=0)
    if len(points) <= 2:
        return points

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])


def enclosing_circle(points):
    """Fit a deterministic enclosing circle to the projected mesh hull."""
    hull = convex_hull_2d(points)
    shuffled = hull[np.random.default_rng(0).permutation(len(hull))]
    center, radius_squared = shuffled[0].copy(), 0.0
    for i, p in enumerate(shuffled):
        if np.dot(p - center, p - center) <= radius_squared + 1e-14:
            continue
        center, radius_squared = p.copy(), 0.0
        for j, q in enumerate(shuffled[:i]):
            if np.dot(q - center, q - center) <= radius_squared + 1e-14:
                continue
            center = (p + q) / 2
            radius_squared = np.dot(p - center, p - center)
            for r in shuffled[:j]:
                if np.dot(r - center, r - center) <= radius_squared + 1e-14:
                    continue
                matrix = 2 * np.array([q - p, r - p])
                if abs(np.linalg.det(matrix)) < 1e-12:
                    a, b = max(((p, q), (p, r), (q, r)), key=lambda pair: np.sum((pair[0] - pair[1]) ** 2))
                    center = (a + b) / 2
                else:
                    # Solve in coordinates relative to p for numerical stability.
                    center = p + np.linalg.solve(matrix, [np.sum((q - p) ** 2), np.sum((r - p) ** 2)])
                radius_squared = max(np.sum((point - center) ** 2) for point in (p, q, r))
    # Recompute against the full hull: no fitting tolerance may omit vertices.
    return center, np.sqrt(np.max(np.sum((hull - center) ** 2, axis=1)))


def rotation_rpy(rotation):
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def fit_primitive(points):
    """Choose an enclosing primitive by volume, without dropping mesh points."""
    padding = 0.001
    bases = [np.eye(3)]
    bases.extend((np.eye(3)[:, [1, 2, 0]], np.eye(3)[:, [2, 0, 1]]))
    _, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
    for order in ((1, 0, 2), (2, 1, 0), (0, 2, 1)):
        basis = vectors[:, order].copy()
        if np.linalg.det(basis) < 0:
            basis[:, 0] *= -1
        bases.append(basis)
    candidates = []
    for basis in bases:
        local = points @ basis
        lower, upper = local.min(axis=0), local.max(axis=0)
        size = upper - lower + 2 * padding
        candidates.append((np.prod(size), "box", basis @ ((lower + upper) / 2), basis, size))
        center, radius = enclosing_circle(local[:, :2])
        radius += padding
        length = size[2]
        center = basis @ np.array([center[0], center[1], (lower[2] + upper[2]) / 2])
        candidates.append((np.pi * radius**2 * length, "cylinder", center, basis, np.array([radius, length, 0])))
    return min(candidates, key=lambda candidate: candidate[0] * (0.98 if candidate[1] == "cylinder" else 1))


def sliced_mesh_points(triangles, slices, with_limits=False):
    """Clip slabs using vertices and edge-plane intersections, covering faces."""
    vertices = triangles.reshape(-1, 3)
    cuts = np.linspace(vertices[:, 2].min(), vertices[:, 2].max(), slices + 1)
    for lower, upper in zip(cuts[:-1], cuts[1:]):
        points = [vertices[(vertices[:, 2] >= lower) & (vertices[:, 2] <= upper)]]
        for plane in (lower, upper):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                start, end = triangles[:, a], triangles[:, b]
                dz = end[:, 2] - start[:, 2]
                crossing = (
                    (dz != 0)
                    & (np.minimum(start[:, 2], end[:, 2]) <= plane)
                    & (np.maximum(start[:, 2], end[:, 2]) >= plane)
                )
                t = (plane - start[crossing, 2]) / dz[crossing]
                points.append(start[crossing] + t[:, None] * (end[crossing] - start[crossing]))
        points = np.unique(np.concatenate(points), axis=0)
        if len(points):
            yield (lower, upper, points) if with_limits else points


def clip_polygon_half(points, axis, boundary, lower):
    """Clip a convex section hull to a half-plane, including crossing edges."""
    result = []
    previous = points[-1]
    previous_inside = previous[axis] <= boundary if lower else previous[axis] >= boundary
    for point in points:
        inside = point[axis] <= boundary if lower else point[axis] >= boundary
        if inside != previous_inside:
            fraction = (boundary - previous[axis]) / (point[axis] - previous[axis])
            result.append(previous + fraction * (point - previous))
        if inside:
            result.append(point)
        previous, previous_inside = point, inside
    return np.asarray(result)


def circle_union_area(circles):
    """Integrate exposed circle arcs to avoid counting overlaps more than once."""
    area = 0.0
    for i, (center, radius) in enumerate(circles):
        arcs = [(0.0, 2 * np.pi)]
        for j, (other, other_radius) in enumerate(circles):
            if i == j:
                continue
            delta = other - center
            distance = np.linalg.norm(delta)
            if distance + radius <= other_radius:
                if distance > 1e-12 or radius < other_radius or j < i:
                    arcs = []
                    break
            if distance >= radius + other_radius or distance <= abs(radius - other_radius):
                continue
            angle = np.arctan2(delta[1], delta[0])
            half = np.arccos(np.clip((distance**2 + radius**2 - other_radius**2) / (2 * distance * radius), -1, 1))
            start = (angle - half) % (2 * np.pi)
            end = start + 2 * half
            covered = [(start, min(end, 2 * np.pi))]
            if end > 2 * np.pi:
                covered.append((0, end - 2 * np.pi))
            for lo, hi in covered:
                remaining = []
                for a, b in arcs:
                    if hi <= a or lo >= b:
                        remaining.append((a, b))
                    else:
                        if lo > a:
                            remaining.append((a, lo))
                        if hi < b:
                            remaining.append((hi, b))
                arcs = remaining
        for a, b in arcs:
            area += 0.5 * (
                radius**2 * (b - a)
                + radius * center[0] * (np.sin(b) - np.sin(a))
                + radius * center[1] * (np.cos(a) - np.cos(b))
            )
    return area


def rounded_section(hull):
    """Greedily refine a section into at most four overlapping cylinders."""

    def circle(polygon):
        center, radius = enclosing_circle(polygon)
        return center, radius + 0.001

    polygons = [hull]
    circles = [circle(hull)]
    area = circle_union_area(circles)
    for _ in range(3):
        best = None
        for index, polygon in enumerate(polygons):
            for axis in (0, 1):
                lo, hi = polygon[:, axis].min(), polygon[:, axis].max()
                for fraction in (0.35, 0.5, 0.65):
                    boundary = lo + fraction * (hi - lo)
                    halves = [clip_polygon_half(polygon, axis, boundary, half) for half in (True, False)]
                    candidate = circles[:index] + [circle(half) for half in halves] + circles[index + 1 :]
                    candidate_area = circle_union_area(candidate)
                    if best is None or candidate_area < best[0]:
                        best = candidate_area, candidate, polygons[:index] + halves + polygons[index + 1 :]
        # Additional geometry must remove enough empty space to justify its cost.
        if best[0] >= area * 0.95:
            break
        area, circles, polygons = best
    return area, circles


def torso_primitives(triangles):
    """Fit oval chest sections, choosing compact single or compound cylinders."""
    padding = 0.001
    for lower, upper, points in sliced_mesh_points(triangles, 8, with_limits=True):
        hull = convex_hull_2d(points[:, :2])
        length = upper - lower + 2 * padding
        z = (lower + upper) / 2
        boxes = points.max(axis=0) - points.min(axis=0) + 2 * padding
        # Each partition encloses its complete clipped convex section hull.
        area, circles = rounded_section(hull)
        # Favor cylinders for equally compact fits; flat rectangular sections
        # remain boxes when rounding would add substantial empty space.
        if area <= boxes[0] * boxes[1] * 1.02:
            for c, r in circles:
                yield "cylinder", np.array([c[0], c[1], z]), np.eye(3), np.array([r, length, 0])
        else:
            yield "box", (points.min(axis=0) + points.max(axis=0)) / 2, np.eye(3), boxes


def use_collision_primitives(urdf, omnihand=False):
    """Replace CAD collision meshes with fitted cylinders and compact boxes."""
    segmented = (
        "shoulder_roll_link",
        "shoulder_yaw_link",
        "elbow_link",
        "wrist_yaw_link",
        "hip_roll_link",
        "hip_yaw_link",
        "knee_link",
    )
    for link in urdf.findall("link"):
        name = link.get("name")
        for collision in list(link.findall("collision")):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            triangles = mesh_triangles(ASSETS / "agibot/x2" / mesh.get("filename")) * scale
            origin = collision.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rotation = np.asarray(quaternion_matrix(urdf_quaternion(*map(float, origin.get("rpy", "0 0 0").split()))))
            if name.endswith("wrist_roll_link") and not omnihand:
                slices = 4
            elif name.endswith(segmented):
                slices = 2
            else:
                slices = 1
            link.remove(collision)
            if name == "torso_link":
                primitives = torso_primitives(triangles)
            else:
                primitives = (fit_primitive(points)[1:] for points in sliced_mesh_points(triangles, slices))
            for index, (kind, center, basis, size) in enumerate(primitives):
                combined = rotation @ basis
                collision = ET.SubElement(link, "collision", name=f"{name}_{kind}_{index}")
                ET.SubElement(
                    collision,
                    "origin",
                    xyz=" ".join(map(str, xyz + rotation @ center)),
                    rpy=" ".join(map(str, rotation_rpy(combined))),
                )
                geometry = ET.SubElement(collision, "geometry")
                if kind == "cylinder":
                    ET.SubElement(geometry, "cylinder", radius=str(size[0]), length=str(size[1]))
                else:
                    ET.SubElement(geometry, "box", size=" ".join(map(str, size)))


def add_articulated_hands(urdf):
    """Merge the bundled hand URDF chains; keep every finger joint observable.

    URDF mimic tags approximate coupling linearly. Remove them here because the
    measured MuJoCo joint states include the source MJCF's nonlinear coupling.
    Hand DOFs are masked in arm IK, not optimized independently.
    """
    mounts = ET.parse(ASSETS / "agibot/x2/omnihand_wrist_links.xml").getroot()
    for prefix, side in (("L", "left"), ("R", "right")):
        hand = ET.parse(ASSETS / "omnihand_pro" / f"omnihand_pro_{side}.urdf").getroot()
        materials = {material.get("name"): material for material in hand.findall("material")}
        for element in hand:
            if element.tag not in ("link", "joint") or element.get("name") == "base_link":
                continue
            element = deepcopy(element)
            if element.tag == "joint":
                if element.get("name") == f"{prefix}_palm_joint":
                    element = deepcopy(mounts.find(f"joint[@name='{prefix}_palm_joint']"))
                mimic = element.find("mimic")
                if mimic is not None:
                    element.remove(mimic)
            else:
                # Namespace material references so the merged model is standalone.
                for material in element.findall("visual/material"):
                    name = material.get("name")
                    if material.find("color") is None and name in materials:
                        for value in materials[name]:
                            material.append(deepcopy(value))
                    material.set("name", f"{prefix}_{name}")
                # Keep the supplied boxes, cylinders, and spheres in their
                # original link frames. Fill only missing housing geometry.
                if not element.findall("collision") and element.find("visual/geometry/mesh") is not None:
                    visual = element.find("visual")
                    collision = ET.SubElement(element, "collision")
                    for tag in ("origin", "geometry"):
                        collision.append(deepcopy(visual.find(tag)))
                for mesh in element.findall(".//mesh"):
                    mesh.set("filename", f"../../omnihand_pro/meshes/{Path(mesh.get('filename')).name}")
            urdf.append(element)


def articulated_collision_pairs(urdf):
    """Add external hand pairs, excluding structural wrist/palm neighbors."""
    arms, hands, fixed = collision_link_groups(urdf)
    pairs = collision_pairs(urdf)
    for side, links in hands.items():
        other = "right" if side == "left" else "left"
        pairs.extend([hand, body] for hand in links for body in fixed + arms[side][:-3] + arms[other])
    pairs.extend([a, b] for a in hands["left"] for b in hands["right"])
    # Same-hand neighbors intentionally touch/curl close together. External
    # avoidance uses the normal arm clearance; no 2 cm internal-finger margin.
    return pairs


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
    add_collision_geometry(default_ik)
    add_collision_geometry(ik, omnihand=True)
    use_collision_primitives(default_ik)
    add_articulated_hands(ik)
    use_collision_primitives(ik, omnihand=True)
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
    documents = build_documents()
    pair_files = {
        "collisions.json": collision_pairs(documents["x2_fixed.urdf"]),
        "collisions_omnihand.json": articulated_collision_pairs(documents["x2_omnihand_ik.urdf"]),
    }
    for name, pairs in pair_files.items():
        contents = "[\n" + ",\n".join("  " + json.dumps(pair) for pair in pairs) + "\n]\n"
        (ASSETS / "agibot/x2" / name).write_text(contents)
    for name, root in documents.items():
        ET.ElementTree(root).write(ASSETS / "agibot/x2" / name, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    build()
