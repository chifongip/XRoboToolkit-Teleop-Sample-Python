"""Headless G1 simulation checks with real IK and mocked XR input."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from meshcat import transformations as tf

from xrobotoolkit_teleop.common.waist_tracking import WaistSample
from xrobotoolkit_teleop.simulation import offline_replay
from xrobotoolkit_teleop.simulation.unitree_g1_model import G1_ASSETS, g1_model_stem
from xrobotoolkit_teleop.simulation.unitree_g1_teleop_controller import UnitreeG1TeleopController
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD


SIDES = ("left", "right")
VARIANTS = ("23dof", "29dof")
IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


class ReplayXR:
    def __init__(self):
        self.timestamp_ns = 1
        self.now = 1.0
        self.gripped = False
        self.homing = False
        self.poses = {side: IDENTITY_POSE.copy() for side in SIDES}
        self.closed = False

    def get_timestamp_ns(self):
        return self.timestamp_ns

    def get_key_value_by_name(self, name):
        return float(self.gripped and name.endswith("_grip"))

    def get_button_state_by_name(self, name):
        return name == "Y" and self.homing

    def get_pose_by_name(self, name):
        side = name.removesuffix("_controller")
        return self.poses.get(side, IDENTITY_POSE).copy()

    def get_waist_sample(self):
        return WaistSample(
            IDENTITY_POSE.copy(),
            self.poses,
            self.timestamp_ns,
            self.timestamp_ns,
            self.now,
            np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        )

    def close(self):
        self.closed = True


def mujoco_frames(model, data):
    result = {}
    for side in SIDES:
        body = data.body(f"{side}_palm_center")
        pose = np.eye(4)
        pose[:3, :3] = body.xmat.reshape(3, 3)
        pose[:3, 3] = body.xpos
        result[side] = pose
    return result


class G1SimulationTests(unittest.TestCase):
    def test_placo_loads_bundled_pairs_before_checking_neutral(self):
        # Capture native startup output as well as Python output.
        probe = """
import sys
import placo

robot = placo.RobotWrapper(sys.argv[1])
startup_pairs = robot.collision_model.collision_pairs
robot.load_collision_pairs(sys.argv[2])
assert startup_pairs
assert startup_pairs == robot.collision_model.collision_pairs
"""
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                stem = g1_model_stem(variant)
                model_dir = G1_ASSETS / stem
                urdf = model_dir / f"{stem}_ik.urdf"
                pairs = model_dir / "collisions.json"
                result = subprocess.run(
                    [sys.executable, "-c", probe, str(urdf), str(pairs)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, output)
                self.assertNotIn("self collisions in neutral position", output)

    def test_configured_collision_pairs_clear_neutral_and_home(self):
        backends = ("placo", "lerobot") if os.environ.get("X2_TEST_LEROBOT") == "1" else ("placo",)
        for variant in VARIANTS:
            pair_counts = []
            for backend in backends:
                with (
                    self.subTest(variant=variant, backend=backend),
                    patch.object(UnitreeG1TeleopController, "_create_xr_client"),
                    contextlib.closing(UnitreeG1TeleopController(variant=variant, ik_backend=backend)) as ctrl,
                ):
                    self.assertGreater(ctrl.collision_checker.distances().min(), ctrl.collision_margin)
                    for name in ctrl.robot.joint_names():
                        ctrl.robot.set_joint(name, 0.0)
                    ctrl.robot.update_kinematics()
                    distances = ctrl.robot.distances()
                    self.assertGreater(min(pair.min_distance for pair in distances), 0)
                    pair_counts.append(len(distances))
            with self.subTest(variant=variant):
                self.assertLessEqual(len(set(pair_counts)), 1)

    def test_variants_share_rubber_hand_collision_shapes(self):
        urdf23, urdf29 = (ET.parse(G1_ASSETS / stem / f"{stem}_ik.urdf") for stem in map(g1_model_stem, VARIANTS))
        for side, sign in (("left", 1), ("right", -1)):
            with self.subTest(side=side):
                wrist = urdf23.findall(f"link[@name='{side}_wrist_roll_rubber_hand']/collision")
                hand = urdf29.findall(f"link[@name='{side}_rubber_hand']/collision")
                self.assertTrue(hand)
                self.assertGreater(len(wrist), len(hand))
                offset = np.array([0.1215, sign * 0.0055, 0.0015])
                for combined, standalone in zip(wrist[-len(hand) :], hand, strict=True):
                    actual_shape = combined.find("geometry")[0]
                    expected_shape = standalone.find("geometry")[0]
                    self.assertEqual(
                        (actual_shape.tag, actual_shape.attrib), (expected_shape.tag, expected_shape.attrib)
                    )
                    actual_origin, expected_origin = combined.find("origin"), standalone.find("origin")
                    self.assertEqual(actual_origin.get("rpy"), expected_origin.get("rpy"))
                    np.testing.assert_allclose(
                        np.fromstring(actual_origin.get("xyz"), sep=" "),
                        np.fromstring(expected_origin.get("xyz"), sep=" ") + offset,
                        atol=1e-11,
                        rtol=0,
                    )

    def test_collision_primitives_cover_visual_meshes(self):
        grouped_links = {
            "pelvis": ("pelvis_contour_link",),
            "logo_link": ("torso_link",),
            "head_link": ("head_link", "torso_link"),
        }
        for variant in VARIANTS:
            with self.subTest(variant=variant), patch.object(UnitreeG1TeleopController, "_create_xr_client"):
                ctrl = UnitreeG1TeleopController(variant=variant)
                self.addCleanup(ctrl.close)
                model, data = ctrl.mj_model, ctrl.mj_data
                stock = ET.parse(G1_ASSETS / f"{g1_model_stem(variant)}.urdf")
                visual_links = {
                    Path(mesh.get("filename")).stem: link.get("name")
                    for link in stock.findall("link")
                    for mesh in link.findall("visual/geometry/mesh")
                }
                mesh_files = {
                    mesh.get("name"): Path(mesh.get("file")).stem
                    for mesh in ET.parse(ctrl.xml_path).findall("asset/mesh")
                }
                by_link = {}
                placements = ctrl.robot.collision_placements(ctrl.robot.state.q)
                for shape, pose in zip(ctrl.robot.collision_model.objects, placements, strict=True):
                    by_link.setdefault(shape.link, []).append((shape, pose))

                covered_meshes = set()
                for geom in range(model.ngeom):
                    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                        continue
                    mesh = model.mesh(model.geom_dataid[geom])
                    filename = mesh_files[mesh.name]
                    link = visual_links[filename]
                    addr, count = int(mesh.vertadr[0]), int(mesh.vertnum[0])
                    vertices = model.mesh_vert[addr : addr + count]
                    addr, count = int(mesh.faceadr[0]), int(mesh.facenum[0])
                    triangles = vertices[model.mesh_face[addr : addr + count]]
                    points = np.concatenate(
                        (vertices, triangles.mean(axis=1), 0.5 * (triangles[:, 0] + triangles[:, 1]))
                    )
                    points = points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                    covered = np.zeros(len(points), dtype=bool)
                    tolerance = 0.0001  # Allow for stock MJCF transform rounding.
                    for host in grouped_links.get(link, (link,)):
                        for shape, pose in by_link[host]:
                            local = (points - pose[:3, 3]) @ pose[:3, :3]
                            if shape.type == "box":
                                covered |= np.all(np.abs(local) <= shape.size / 2 + tolerance, axis=1)
                            else:
                                self.assertEqual(shape.type, "cylinder")
                                radius, length = shape.size[:2]
                                covered |= (np.hypot(local[:, 0], local[:, 1]) <= radius + tolerance) & (
                                    np.abs(local[:, 2]) <= length / 2 + tolerance
                                )
                    self.assertTrue(np.all(covered), f"{link}: {np.count_nonzero(~covered)} uncovered mesh samples")
                    covered_meshes.add(filename)
                self.assertEqual(covered_meshes, set(visual_links))

    def test_custom_model_validation_precedes_xr(self):
        model23 = g1_model_stem("23dof")
        model29 = g1_model_stem("29dof")
        xml = G1_ASSETS / f"{model23}_scene.xml"
        urdf = G1_ASSETS / model29 / f"{model29}_ik.urdf"
        with patch.object(UnitreeG1TeleopController, "_create_xr_client") as xr:
            with self.assertRaisesRegex(ValueError, "Provide both"):
                UnitreeG1TeleopController(xml, variant="23dof")
            with self.assertRaisesRegex(ValueError, "matching fixed-base models"):
                UnitreeG1TeleopController(xml, urdf, variant="23dof")
            xr.assert_not_called()

    def test_mismatched_wrist_axis_is_rejected_before_xr(self):
        stem = g1_model_stem("23dof")
        urdf = G1_ASSETS / stem / f"{stem}_ik.urdf"
        changed = ET.parse(G1_ASSETS / f"{stem}_scene.xml")
        changed.find("compiler").set("meshdir", str((G1_ASSETS / "meshes").resolve()))
        changed.find(".//worldbody//joint[@name='left_wrist_roll_joint']").set("axis", "-1 0 0")
        with TemporaryDirectory() as folder, patch.object(UnitreeG1TeleopController, "_create_xr_client") as xr:
            xml = Path(folder) / "wrong-wrist-axis.xml"
            changed.write(xml)
            with self.assertRaisesRegex(ValueError, "do not match for left_palm_center"):
                UnitreeG1TeleopController(xml, urdf, variant="23dof")
            xr.assert_not_called()

    def test_placo_tracking_matrix(self):
        for variant in VARIANTS:
            for mode in ("fixed-heading", "waist"):
                with self.subTest(variant=variant, tracking_reference=mode):
                    self.check_replay(variant, mode, "placo")

    @unittest.skipUnless(os.environ.get("X2_TEST_LEROBOT") == "1", "enable native LeRobot tests")
    def test_lerobot_tracking_matrix(self):
        for variant in VARIANTS:
            for mode in ("fixed-heading", "waist"):
                with self.subTest(variant=variant, tracking_reference=mode):
                    self.check_replay(variant, mode, "lerobot")

    def check_replay(self, variant, mode, backend):
        xr = ReplayXR()
        with (
            patch.object(UnitreeG1TeleopController, "_create_xr_client", return_value=xr),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            ctrl = UnitreeG1TeleopController(
                variant=variant, ik_backend=backend, tracking_reference=mode, control_frequency=60
            )
        try:
            with patch("xrobotoolkit_teleop.simulation.fixed_base_teleop_controller.time.monotonic", lambda: xr.now):
                self.run_replay(ctrl, xr, variant)
        finally:
            ctrl.close()
        self.assertTrue(xr.closed)

    def run_replay(self, ctrl, xr, variant):
        model, data = ctrl.mj_model, ctrl.mj_data
        home = model.key("home").qpos.copy()
        home_commands = model.key("home").ctrl.copy()
        reference = mujoco.MjData(model)
        self.assertEqual(model.njnt, 10 if variant == "23dof" else 14)
        self.assertEqual(ctrl.physics_substeps, 17)
        self.assertGreater(ctrl.collision_checker.distances().min(), ctrl.collision_margin)
        self.assertEqual(set(ctrl.target_mocap_idx.values()), {0, 1})
        initial = ctrl.robot.state.q.copy()
        joint_offsets = [ctrl.robot.get_joint_offset(name) for name in ctrl.arm_actuators.values()]
        velocity = np.array(
            [
                ctrl.robot.model.velocity_limit[ctrl.robot.get_joint_v_offset(name)]
                for name in ctrl.arm_actuators.values()
            ]
        )
        limits = np.array(list(ctrl.position_limits.values()))

        def tick(fresh=True):
            previous = data.ctrl.copy()
            before = data.time
            xr.now += ctrl.dt if fresh else 2 * ctrl.XR_TRACKING_TIMEOUT_S
            if fresh:
                xr.timestamp_ns += round(ctrl.dt * 1e9)
            ctrl.step_control()
            self.assertAlmostEqual(data.time - before, ctrl.dt, places=10)
            cap = np.minimum(velocity, ctrl.return_velocity_limit) if xr.homing else velocity
            self.assertTrue(np.all(np.abs(data.ctrl - previous) <= cap * ctrl.dt + 1e-7))
            self.assertTrue(np.all(data.ctrl >= limits[:, 0] - 1e-7))
            self.assertTrue(np.all(data.ctrl <= limits[:, 1] + 1e-7))

        target = home.copy()
        for side, sign in (("left", 1), ("right", -1)):
            deltas = (("shoulder_pitch", -0.10), ("elbow", 0.12), ("wrist_roll", 0.18 * sign))
            if variant == "29dof":
                deltas += (("wrist_pitch", -0.05), ("wrist_yaw", 0.10 * sign))
            for joint, delta in deltas:
                address = model.joint(f"{side}_{joint}_joint").qposadr[0]
                target[address] += delta
        for q in (home, target):
            reference.qpos[:] = q
            mujoco.mj_forward(model, reference)
            for name in ctrl.robot.joint_names():
                ctrl.robot.set_joint(name, q[model.joint(name).qposadr[0]])
            ctrl.robot.update_kinematics()
            for side, expected in mujoco_frames(model, reference).items():
                np.testing.assert_allclose(
                    ctrl.robot.get_T_world_frame(f"{side}_palm_center"), expected, atol=1e-4, rtol=0
                )
        ctrl.robot.state.q = initial
        ctrl.robot.update_kinematics()

        for _ in range(3):
            tick()
        xr.gripped = True
        tick()
        anchors = {side: ctrl.effector_task[f"{side}_arm"].T_world_frame.copy() for side in SIDES}
        basis = R_HEADSET_TO_WORLD
        for amount in np.linspace(0.1, 1, 10):
            reference.qpos[:] = home + amount * (target - home)
            mujoco.mj_forward(model, reference)
            for side, pose in mujoco_frames(model, reference).items():
                anchor = anchors[side]
                delta = np.eye(4)
                delta[:3, :3] = basis.T @ pose[:3, :3] @ anchor[:3, :3].T @ basis
                xr.poses[side] = np.r_[
                    basis.T @ (pose[:3, 3] - anchor[:3, 3]),
                    np.roll(tf.quaternion_from_matrix(delta), -1),
                ]
            previous = data.ctrl.copy()
            tick()
            self.assertTrue(ctrl._ik_valid, ctrl.status)
            for side, expected in mujoco_frames(model, reference).items():
                np.testing.assert_allclose(ctrl.effector_task[f"{side}_arm"].T_world_frame, expected, atol=1e-6)
            # Check the accepted command path independently of the shared policy.
            saved = ctrl.robot.state.q.copy()
            try:
                for command in np.linspace(previous, data.ctrl, 4):
                    q = saved.copy()
                    q[joint_offsets] = command
                    ctrl.robot.state.q = q
                    self.assertGreaterEqual(ctrl.collision_checker.distances().min(), ctrl.collision_margin - 1e-6)
            finally:
                ctrl.robot.state.q = saved
                ctrl.robot.update_kinematics()

        for _ in range(12):
            tick()
        visible = ctrl.visualization_snapshot()
        for side, desired in mujoco_frames(model, reference).items():
            arm = f"{side}_arm"
            side_mask = [name.startswith(side) for name in ctrl.arm_actuators.values()]
            self.assertGreater(np.linalg.norm((data.ctrl - home_commands)[side_mask]), 0.04)
            actual = visible.end_effectors[arm]
            self.assertLess(np.linalg.norm(actual[:3, 3] - desired[:3, 3]), 0.025)
            error = np.trace(actual[:3, :3].T @ desired[:3, :3])
            self.assertLess(np.arccos(np.clip((error - 1) / 2, -1, 1)), 0.15)

        if variant == "23dof":
            # A rotation-only request around a controller axis can be unattainable
            # at the commanded palm position. Both backends must respond within
            # joint, slew and collision constraints while retaining the target.
            poses = {side: pose.copy() for side, pose in xr.poses.items()}
            for angle in np.linspace(0.03, 0.30, 10):
                for side, pose in poses.items():
                    rotation = tf.quaternion_about_axis(angle, [0, 1, 0])
                    delta = tf.quaternion_multiply(rotation, np.roll(pose[3:], 1))
                    xr.poses[side][3:] = np.roll(delta, -1)
                tick()
                self.assertTrue(ctrl._ik_valid, ctrl.status)
                self.assertTrue(np.all(np.isfinite(data.ctrl)))

        held = data.ctrl.copy()
        xr.poses["left"][0] = np.nan
        tick()
        np.testing.assert_array_equal(data.ctrl, held)
        xr.poses["left"] = IDENTITY_POSE.copy()
        tick(fresh=False)
        self.assertFalse(ctrl._stream_fresh)
        np.testing.assert_array_equal(data.ctrl, held)

        xr.gripped = False
        tick()
        np.testing.assert_array_equal(data.ctrl, held)
        xr.gripped = True
        xr.homing = True
        for _ in range(45):
            tick()
            self.assertTrue(ctrl._ik_valid, ctrl.status)
        np.testing.assert_array_equal(data.ctrl, home_commands)
        xr.homing = False
        tick()
        self.assertFalse(ctrl._ik_valid)
        np.testing.assert_array_equal(data.ctrl, home_commands)
        xr.gripped = False
        tick()
        xr.gripped = True
        tick()
        tick()
        self.assertTrue(ctrl._ik_valid, ctrl.status)


class G1OfflineReplayTests(unittest.TestCase):
    def check_g1_backend(self, backend):
        for requested, variant, joint_count in (("23dof", "23dof", 10), (None, "29dof", 14)):
            with (
                self.subTest(backend=backend, variant=variant),
                patch(
                    "xrobotoolkit_sdk.init", side_effect=AssertionError("offline replay initialized the XR SDK")
                ) as sdk,
            ):
                result = offline_replay.run_replay(backend, robot="g1", variant=requested)
                self.assertTrue(result["passed"], result["violations"])
                self.assertEqual((result["robot"], result["variant"]), ("g1", variant))
                self.assertEqual(len(result["arm_joint_names"]), joint_count)
                self.assertEqual(result["failed_cycles"], 0)
                self.assertEqual([e["pose"] for e in result["endpoints"]], ["T", "home", "T", "home"])
                for endpoint in result["endpoints"][::2]:
                    self.assertLessEqual(endpoint["max_arm_t_pose_deviation_deg"], 10)
                sdk.assert_not_called()

    def test_g1_variants_with_placo(self):
        self.check_g1_backend("placo")

    @unittest.skipUnless(os.environ.get("X2_TEST_LEROBOT") == "1", "requires optional Pinocchio/CasADi build")
    def test_g1_variants_with_lerobot(self):
        self.check_g1_backend("lerobot")

    def test_robot_specific_options_are_validated(self):
        with self.assertRaisesRegex(ValueError, "--with-omnihand requires --robot x2"):
            offline_replay.run_replay(robot="g1", with_omnihand=True)
        with self.assertRaisesRegex(ValueError, "--variant requires --robot g1"):
            offline_replay.run_replay(variant="23dof")


if __name__ == "__main__":
    unittest.main()
