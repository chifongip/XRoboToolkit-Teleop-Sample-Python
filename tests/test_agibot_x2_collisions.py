"""Collision geometry, constrained IK, and rejected-command integration checks."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import placo
from meshcat import transformations as tf

from test_agibot_x2_simulation import ASSETS, FakeXR
from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController

THUMB_THIGH_POSE = {
    "left_elbow_joint": -1.7811800987455206,
    "left_shoulder_pitch_joint": 1.923722789280073,
    "left_shoulder_roll_joint": 2.4981127086550687,
    "left_shoulder_yaw_joint": 0.3528405628567013,
    "left_wrist_pitch_joint": -0.008333327148600134,
    "left_wrist_roll_joint": -0.006863725365831019,
    "left_wrist_yaw_joint": 2.4325615031041945,
}


class CollisionXR(FakeXR):
    def __init__(self):
        super().__init__()
        self.grips.update(left_trigger=0.0, right_trigger=0.0)


class X2CollisionTests(unittest.TestCase):
    @staticmethod
    def points_covered(shapes, points):
        covered = np.zeros(len(points), dtype=bool)
        for shape in shapes:
            origin = shape.find("origin")
            xyz = np.fromstring(origin.get("xyz"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            local = (points - xyz) @ tf.euler_matrix(*rpy)[:3, :3]
            geometry = shape.find("geometry")
            if geometry.find("box") is not None:
                half = np.fromstring(geometry.find("box").get("size"), sep=" ") / 2
                inside = np.all(np.abs(local) <= half + 1e-6, axis=1)
            elif geometry.find("cylinder") is not None:
                cylinder = geometry.find("cylinder")
                inside = (np.linalg.norm(local[:, :2], axis=1) <= float(cylinder.get("radius")) + 1e-6) & (
                    np.abs(local[:, 2]) <= float(cylinder.get("length")) / 2 + 1e-6
                )
            elif geometry.find("sphere") is not None:
                inside = np.linalg.norm(local, axis=1) <= float(geometry.find("sphere").get("radius")) + 1e-6
            else:
                raise AssertionError("Unexpected collision shape")
            covered |= inside
        return covered

    def assert_points_covered(self, shapes, points):
        covered = self.points_covered(shapes, points)
        self.assertTrue(np.all(covered), f"{np.count_nonzero(~covered)} mesh samples outside collision primitives")

    def make_controller(self, omnihand=False, **kwargs):
        scene, urdf = ("scene_omnihand.xml", "x2_omnihand_ik.urdf") if omnihand else ("scene.xml", "x2_fixed.urdf")
        with (
            patch("xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller.XrClient", CollisionXR),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            c = AgibotX2TeleopController(str(ASSETS / scene), str(ASSETS / urdf), **kwargs)
        self.addCleanup(c.close)
        return c

    def test_pairs_geometry_and_home_clearance_in_both_modes(self):
        pairs = json.loads((ASSETS / "collisions.json").read_text())
        self.assertEqual(len(pairs), len({tuple(sorted(pair)) for pair in pairs}))
        self.assertIn(["left_wrist_roll_link", "right_wrist_roll_link"], pairs)
        self.assertIn(["left_elbow_link", "torso_link"], pairs)
        self.assertNotIn(["left_elbow_link", "left_wrist_yaw_link"], pairs)
        for omnihand in (False, True):
            with self.subTest(omnihand=omnihand):
                c = self.make_controller(omnihand)
                self.assertGreaterEqual(c.collision_pair_count, len(pairs))
                self.assertEqual(c.collision_constraint.priority, "hard")
                self.assertGreater(min(d.min_distance for d in c.robot.distances()), c.collision_margin)
                urdf = ET.parse(c.robot_urdf_path).getroot()
                kinds = {list(shape.find("geometry"))[0].tag for shape in urdf.findall("link/collision")}
                self.assertIn("cylinder", kinds)
                self.assertTrue(kinds <= {"box", "cylinder", "sphere"})

    def test_cached_collision_linearization_matches_finite_differences(self):
        for omnihand in (False, True):
            c = self.make_controller(omnihand)
            robot = c.robot
            for pitch in (0.35, 1.7, 2.0):
                robot.set_joint("left_shoulder_pitch_joint", pitch)
                robot.update_kinematics()
                c.collision_checker.clear_cache()
                baseline = c.collision_checker.distances()
                nearby = baseline < c.collision_trigger
                values, jacobian = c.collision_checker.linearized_clearance(c.collision_trigger)
                np.testing.assert_array_equal(values, baseline[nearby])
                previous = robot.state.q.copy()
                for joint in c.arm_joints["left_arm"]:
                    offset = robot.get_joint_offset(joint)
                    plus, minus = previous.copy(), previous.copy()
                    plus[offset] += 1e-5
                    minus[offset] -= 1e-5
                    robot.state.q = plus
                    forward = c.collision_checker.distances()[nearby]
                    robot.state.q = minus
                    backward = c.collision_checker.distances()[nearby]
                    np.testing.assert_allclose(
                        jacobian[:, robot.get_joint_v_offset(joint)], (forward - backward) / 2e-5, atol=5e-3, rtol=0
                    )
                robot.state.q = previous

    def test_collision_linearization_reuses_current_witnesses_only(self):
        c = self.make_controller(True)
        checker, robot = c.collision_checker, c.robot
        checker.clear_cache()
        q = robot.state.q.copy()
        native_distances = robot.distances
        with patch.object(robot, "distances", side_effect=native_distances) as query:
            checker.distances()
            distances, jacobian = checker.linearized_clearance(c.collision_trigger)
            self.assertEqual(query.call_count, 1)
            robot.set_joint("left_shoulder_pitch_joint", robot.get_joint("left_shoulder_pitch_joint") + 0.001)
            checker.linearized_clearance(c.collision_trigger)
            self.assertEqual(query.call_count, 2)
            robot.state.q = q
            checker.distances()  # Scalar cache hit; witnesses still belong to the other pose.
            self.assertEqual(query.call_count, 2)
            restored_distances, restored_jacobian = checker.linearized_clearance(c.collision_trigger)
            self.assertEqual(query.call_count, 3)
            np.testing.assert_array_equal(restored_distances, distances)
            np.testing.assert_allclose(restored_jacobian, jacobian, atol=1e-12)
            empty_distances, empty_jacobian = checker.linearized_clearance(-1e6)
            self.assertEqual(empty_distances.shape, (0,))
            self.assertEqual(empty_jacobian.shape, (0, robot.model.nv))

    def test_ik_with_no_pairs_inside_collision_trigger(self):
        c = self.make_controller(collision_trigger=0.025)
        c.xr_client.grips.update(left_grip=1.0, right_grip=1.0)
        c._update_ik()
        self.assertTrue(c._ik_valid)
        np.testing.assert_array_equal(np.atleast_2d(c.collision_constraint.A), np.zeros((1, c.robot.model.nq)))
        np.testing.assert_array_equal(np.atleast_1d(c.collision_constraint.b), np.ones(1))

    def test_articulated_pairs_cover_external_contacts_without_structural_neighbors(self):
        pairs = json.loads((ASSETS / "collisions_omnihand.json").read_text())
        normalized = {tuple(sorted(pair)) for pair in pairs}
        self.assertEqual(len(pairs), len(normalized))
        for pair in (
            ("L_index_dip_link", "R_index_dip_link"),
            ("L_thumb_dip_link", "torso_link"),
            ("L_index_dip_link", "left_elbow_link"),
        ):
            self.assertIn(tuple(sorted(pair)), normalized)
        for a, b in pairs:
            self.assertFalse(a.startswith("L_") and b.startswith("L_"))
            self.assertFalse(a.startswith("R_") and b.startswith("R_"))
            for hand, arm in ((a, b), (b, a)):
                side = "left" if hand.startswith("L_") else "right" if hand.startswith("R_") else None
                if side:
                    self.assertNotIn(
                        arm, (f"{side}_wrist_yaw_link", f"{side}_wrist_pitch_link", f"{side}_wrist_roll_link")
                    )

    def test_safe_endpoints_with_colliding_intermediate_pose_are_rejected(self):
        c = self.make_controller()
        c.active.update(left_arm=True, right_arm=True)
        robot = c.robot
        home = robot.state.q.copy()
        joints = {
            "left_elbow_joint": -1.5901651064,
            "left_shoulder_pitch_joint": 1.0478224949,
            "left_shoulder_roll_joint": 0.9220900547,
            "left_shoulder_yaw_joint": -1.7941147610,
            "left_wrist_pitch_joint": 0.2078817563,
            "left_wrist_roll_joint": -0.5077870421,
            "left_wrist_yaw_joint": 1.5281786778,
        }
        for joint, value in joints.items():
            robot.set_joint(joint, value)
        target = robot.state.q.copy()
        robot.update_kinematics()
        self.assertGreater(min(d.min_distance for d in robot.distances()), c.collision_margin)
        # The penetrating portion is 40% along the path from the new home.
        robot.state.q = home + 0.4 * (target - home)
        robot.update_kinematics()
        self.assertLess(min(d.min_distance for d in robot.distances()), 0)
        robot.state.q = target.copy()
        with self.assertRaisesRegex(RuntimeError, "clearance violated"):
            c._validate_collision_motion(home)
        np.testing.assert_array_equal(robot.state.q, target)

    def test_invalid_margins_rejected_before_xr_connection(self):
        for kwargs in ({"collision_margin": v} for v in (0, -0.01, np.nan, np.inf)):
            with patch("xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller.XrClient") as xr:
                with self.assertRaises(ValueError):
                    AgibotX2TeleopController(str(ASSETS / "scene.xml"), str(ASSETS / "x2_fixed.urdf"), **kwargs)
                xr.assert_not_called()
        for value in (0.01, 0.02, np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.make_controller(collision_trigger=value)

    def test_missing_collision_shapes_fail_setup_and_close_xr(self):
        root = ET.parse(ASSETS / "x2_fixed.urdf").getroot()
        link = root.find("link[@name='left_elbow_link']")
        for collision in list(link.findall("collision")):
            link.remove(collision)
        for mesh in root.findall(".//mesh"):
            mesh.set("filename", str(ASSETS / mesh.get("filename")))
        xr = CollisionXR()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_collision.urdf"
            ET.ElementTree(root).write(path)
            with (
                patch(
                    "xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller.XrClient", return_value=xr
                ) as connect,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(ValueError, "collision geometry for left_elbow_link"):
                    AgibotX2TeleopController(str(ASSETS / "scene.xml"), str(path))
        connect.assert_not_called()

    def test_colliding_candidate_is_rejected_and_state_restored(self):
        for omnihand in (False, True):
            c = self.make_controller(omnihand)
            c.xr_client.grips["left_grip"] = 1.0
            c._update_ik()
            c._update_robot_state()
            previous = c.robot.state.q.copy()
            held = c.mj_data.ctrl.copy()
            # Find a real geometric collision, independent of mocked distances.
            rng = np.random.default_rng(14)
            for _ in range(100):
                c.robot.state.q = previous.copy()
                for joint in c.arm_joints["left_arm"]:
                    limits = c.mj_model.joint(joint).range
                    c.robot.set_joint(joint, rng.uniform(*limits))
                c.robot.update_kinematics()
                if min(d.min_distance for d in c.robot.distances()) < 0:
                    unsafe = c.robot.state.q.copy()
                    break
            else:
                self.fail("No colliding fixture found")
            c.robot.state.q = previous.copy()
            c.robot.update_kinematics()

            def unsafe_solve(*args):
                c.robot.state.q = unsafe.copy()

            with (
                patch.object(placo.KinematicsSolver, "solve", side_effect=unsafe_solve),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                c._update_ik()
                c._send_command()
            self.assertFalse(c._ik_valid)
            np.testing.assert_array_equal(c.mj_data.ctrl, held)
            np.testing.assert_allclose(c.robot.state.q, previous, atol=1e-12)

    def test_inward_teleop_preserves_or_recovers_clearance_of_accepted_commands(self):
        for omnihand in (False, True):
            c = self.make_controller(omnihand)
            c.xr_client.grips.update(left_grip=1.0, right_grip=1.0)
            c.step_control()
            initial = c.mj_data.ctrl.copy()
            # Lift and reach forward while moving inward, clearing the thighs
            # beneath the locomanipulation home pose.
            c.xr_client.poses["left_controller"][:3] = [0.5, 0.3, -0.2]
            c.xr_client.poses["right_controller"][:3] = [-0.5, 0.3, -0.2]
            accepted = 0
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(100):
                    held = c.mj_data.ctrl.copy()
                    c._update_robot_state()
                    measured_distances = c.collision_checker.distances()
                    c._update_ik()
                    c._send_command()
                    if c._ik_valid:
                        accepted += 1
                        # Test the outgoing targets, including held-arm targets.
                        for actuator, joint in c.arm_actuators.items():
                            c.robot.set_joint(joint, c.mj_data.ctrl[actuator])
                        c.robot.update_kinematics()
                        self.assertTrue(
                            np.all(
                                c.collision_checker.distances()
                                >= np.minimum(measured_distances, c.collision_margin) - 1e-6
                            )
                        )
                    else:
                        np.testing.assert_array_equal(c.mj_data.ctrl, held)
                    mujoco.mj_step(c.mj_model, c.mj_data, nstep=c.physics_substeps)
            self.assertGreater(accepted, 5)
            self.assertGreater(np.linalg.norm(c.mj_data.ctrl - initial), 0.05)

    def test_return_avoids_obstructed_direct_path_and_reaches_home(self):
        for omnihand, dynamics in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(omnihand=omnihand, dynamics=dynamics):
                c = self.make_controller(omnihand, return_velocity_limit=0.5)
                joints = {
                    "left_elbow_joint": -1.5901651064,
                    "left_shoulder_pitch_joint": 1.0478224949,
                    "left_shoulder_roll_joint": 0.9220900547,
                    "left_shoulder_yaw_joint": -1.7941147610,
                    "left_wrist_pitch_joint": 0.2078817563,
                    "left_wrist_roll_joint": -0.5077870421,
                    "left_wrist_yaw_joint": 1.5281786778,
                }
                for joint, value in joints.items():
                    c.mj_data.qpos[c.mj_model.joint(joint).qposadr] = value
                self._sync_return_test_commands(c)
                c._update_robot_state()
                initial = c.robot.state.q.copy()
                # Both endpoints are clear, but their straight interpolation is blocked.
                self.assertGreater(c.collision_checker.distances().min(), c.collision_margin)
                for joint, value in joints.items():
                    home = c.mj_model.key("home").qpos[c.mj_model.joint(joint).qposadr[0]]
                    c.robot.set_joint(joint, (home + value) / 2)
                self.assertLess(c.collision_checker.distances().min(), c.collision_margin)
                c.robot.state.q = initial
                c.xr_client.buttons["Y"] = True
                with patch.object(c, "_solve_return_avoidance", wraps=c._solve_return_avoidance) as avoid:
                    self._run_checked_return(c, dynamics=dynamics)
                self.assertGreater(avoid.call_count, 0)

    def test_return_recovers_existing_collision_without_worsening_it(self):
        for omnihand in (False, True):
            with self.subTest(omnihand=omnihand):
                c = self.make_controller(omnihand)
                for joint, value in (("left_shoulder_pitch_joint", 2.0), ("left_shoulder_roll_joint", -0.06)):
                    c.mj_data.qpos[c.mj_model.joint(joint).qposadr] = value
                self._sync_return_test_commands(c)
                c._update_robot_state()
                self.assertLess(c.collision_checker.distances().min(), 0)
                c.xr_client.buttons["Y"] = True
                self._run_checked_return(c)

    @staticmethod
    def _sync_return_test_commands(c):
        for actuator, joint in c.arm_actuators.items():
            c.mj_data.ctrl[actuator] = c.mj_data.qpos[c.mj_model.joint(joint).qposadr][0]
        mujoco.mj_forward(c.mj_model, c.mj_data)

    def _run_checked_return(self, c, dynamics=False):
        arms = list(c.arm_actuators)
        for _ in range(400):
            c._update_robot_state()
            measured = c.collision_checker.distances()
            previous = c.mj_data.ctrl.copy()
            c._update_ik()
            self.assertTrue(c._ik_valid)
            c._send_command()
            self.assertTrue(c._ik_valid)
            self.assertLessEqual(np.max(np.abs(c.mj_data.ctrl - previous)), c.return_velocity_limit * c.dt + 1e-8)
            self.assertTrue(np.all(c.collision_checker.distances() >= np.minimum(measured, c.collision_margin) - 1e-6))
            if dynamics:
                mujoco.mj_step(c.mj_model, c.mj_data, nstep=c.physics_substeps)
            else:
                # Ideal tracking isolates collision planning from servo dynamics.
                for actuator, joint in c.arm_actuators.items():
                    c.mj_data.qpos[c.mj_model.joint(joint).qposadr] = c.mj_data.ctrl[actuator]
                mujoco.mj_forward(c.mj_model, c.mj_data)
            if np.max(np.abs(c.mj_data.ctrl[arms] - c.mj_model.key("home").ctrl[arms])) < 1e-5:
                break
        else:
            self.fail("Return did not reach home")

    def test_recovery_from_margin_violation_and_penetration_respects_speed_limits(self):
        for omnihand in (False, True):
            for pitch, roll in ((1.7, 0.0), (2.0, -0.06)):
                with self.subTest(omnihand=omnihand, pitch=pitch):
                    c = self.make_controller(omnihand, arm_velocity_limit=0.5)
                    m, data, robot = c.mj_model, c.mj_data, c.robot
                    for joint, value in (("left_shoulder_roll_joint", roll), ("left_shoulder_pitch_joint", pitch)):
                        data.qpos[m.joint(joint).qposadr] = value
                    for actuator, joint in c.arm_actuators.items():
                        data.ctrl[actuator] = data.qpos[m.joint(joint).qposadr][0]
                    mujoco.mj_forward(m, data)
                    c._update_robot_state()
                    initial = c.collision_checker.distances()
                    self.assertLess(initial.min(), 0 if pitch == 2.0 else c.collision_margin)
                    held_right = {a: data.ctrl[a] for a, j in c.arm_actuators.items() if j.startswith("right_")}
                    c.xr_client.grips["left_grip"] = 1.0
                    for index in range(80):
                        c._update_robot_state()
                        before = robot.state.q.copy()
                        distances = c.collision_checker.distances()
                        c._update_ik()
                        self.assertTrue(c._ik_valid)
                        self.assertEqual(c.collision_constraint.priority, "hard")
                        c._send_command()
                        command_distances = c.collision_checker.distances()
                        self.assertTrue(np.all(command_distances >= np.minimum(distances, c.collision_margin) - 1e-6))
                        if index == 0:
                            # A speed-limited recovery step need not clear the
                            # entire margin, but must improve the existing deficit.
                            self.assertLess(command_distances.min(), c.collision_margin)
                            self.assertGreater(command_distances.min(), distances.min())
                        for actuator, joint in c.arm_actuators.items():
                            delta = abs(data.ctrl[actuator] - before[robot.get_joint_offset(joint)])
                            self.assertLessEqual(delta, c.arm_velocity_limit * c.dt + 1e-6)
                            if actuator in held_right:
                                self.assertEqual(data.ctrl[actuator], held_right[actuator])
                            # Ideal servo tracking isolates kinematic recovery
                            # from dynamics, gravity, and tracking error.
                            data.qpos[m.joint(joint).qposadr] = data.ctrl[actuator]
                        mujoco.mj_forward(m, data)
                        if command_distances.min() >= c.collision_margin:
                            break
                    else:
                        self.fail("Recovery did not restore clearance")

    def test_recovery_rejects_worsening_or_new_violations(self):
        c = self.make_controller()
        c.active.update(left_arm=True, right_arm=True)
        robot = c.robot
        robot.set_joint("left_shoulder_pitch_joint", 1.7)
        robot.set_joint("left_shoulder_roll_joint", 0.0)
        measured = robot.state.q.copy()
        for actuator, joint in c.arm_actuators.items():
            c.mj_data.ctrl[actuator] = measured[robot.get_joint_offset(joint)]
        held = next(i for i, joint in c.arm_actuators.items() if joint == "left_shoulder_roll_joint")
        c.mj_data.ctrl[held] = 0.1
        # Deeper incursion into an existing violation is rejected.
        robot.set_joint("left_shoulder_roll_joint", -0.06)
        with self.assertRaisesRegex(RuntimeError, "clearance violated"):
            c._validate_collision_motion(measured)
        # Improving the left arm cannot excuse a new right-arm violation.
        robot.state.q = measured.copy()
        robot.set_joint("left_shoulder_roll_joint", 0.02)
        robot.set_joint("right_shoulder_pitch_joint", 1.7)
        robot.set_joint("right_shoulder_roll_joint", 0.0)
        with self.assertRaisesRegex(RuntimeError, "clearance violated"):
            c._validate_collision_motion(measured)
        # A stationary command cannot be accepted as recovery.
        robot.state.q = measured.copy()
        with self.assertRaisesRegex(RuntimeError, "recovery made no progress"):
            c._validate_collision_motion(measured)

    def test_penetrating_previous_target_requires_signed_path_checks(self):
        c = self.make_controller()
        c.active.update(left_arm=True, right_arm=True)
        robot = c.robot
        measured = robot.state.q.copy()
        previous = c.mj_data.ctrl.copy()
        joint = "left_shoulder_roll_joint"
        actuator = next(i for i, name in c.arm_actuators.items() if name == joint)
        previous[actuator] = 1.0
        home_angle = robot.get_joint(joint)

        # A signed-distance fixture: the measured pose is clear, the old servo
        # target penetrates by 1 cm, and returning toward the measured pose
        # first deepens penetration. Its unsigned bound is valid but cannot
        # certify that the negative clearance floor is preserved.
        def distances():
            fraction = (robot.get_joint(joint) - home_angle) / (1.0 - home_angle)
            return np.array([0.05 - 0.2 * fraction + 0.14 * fraction**2])

        checker = SimpleNamespace(
            can_bound=True,
            pair_indices=c.collision_checker.pair_indices[:1],
            distances=distances,
            baseline=lambda margin: distances(),
            clearance_bounds=lambda floor: np.maximum(distances(), 0),
        )
        with patch.object(c, "collision_checker", checker):
            with self.assertRaisesRegex(RuntimeError, "clearance violated"):
                c._validate_collision_motion(measured, previous_commands=previous)
        np.testing.assert_array_equal(robot.state.q, measured)

    def test_partial_recovery_is_allowed_with_clear_previous_servo_targets(self):
        c = self.make_controller()
        c.active.update(left_arm=True, right_arm=False)
        robot = c.robot
        # The actual arm has drifted inside the margin while its old target
        # remains clear. Recovery must not require the new target to reach
        # the full margin in one timestep from that old, clear servo target.
        robot.set_joint("left_shoulder_pitch_joint", 1.7)
        robot.set_joint("left_shoulder_roll_joint", 0.0)
        measured = robot.state.q.copy()
        for actuator, joint in c.arm_actuators.items():
            c.mj_data.ctrl[actuator] = measured[robot.get_joint_offset(joint)]
        held = next(i for i, joint in c.arm_actuators.items() if joint == "left_shoulder_roll_joint")
        c.mj_data.ctrl[held] = 0.1
        robot.set_joint("left_shoulder_roll_joint", 0.01)
        target = robot.state.q.copy()
        self.assertLess(c.collision_checker.distances().min(), c.collision_margin)
        c._validate_collision_motion(measured)
        np.testing.assert_array_equal(robot.state.q, target)

    def test_simulation_recovers_from_actual_primitive_penetration(self):
        for omnihand in (False, True):
            c = self.make_controller(omnihand)
            m, data = c.mj_model, c.mj_data
            for joint, value in (("left_shoulder_roll_joint", -0.06), ("left_shoulder_pitch_joint", 2.0)):
                data.qpos[m.joint(joint).qposadr] = value
            for actuator, joint in c.arm_actuators.items():
                data.ctrl[actuator] = data.qpos[m.joint(joint).qposadr][0]
            mujoco.mj_forward(m, data)
            c._update_robot_state()
            self.assertLess(c.collision_checker.distances().min(), 0)
            c.xr_client.grips["left_grip"] = 1.0
            for _ in range(60):
                c.step_control()
            # Commands restore the margin. The gravity-loaded position servos
            # have tracking error, so actual motion must clear penetration but
            # is not expected to match the commanded clearance exactly.
            self.assertGreaterEqual(c.collision_checker.distances().min(), c.collision_margin - 1e-6)
            c._update_robot_state()
            self.assertGreater(c.collision_checker.distances().min(), 0)

    def test_articulated_hand_fk_collision_placements_and_masked_ik(self):
        c = self.make_controller(True)
        m, data, robot = c.mj_model, c.mj_data, c.robot
        root = ET.parse(c.robot_urdf_path).getroot()
        self.assertEqual(len(c.hand_joint_names), 38)
        self.assertFalse(any("envelope" in shape.get("name", "") for shape in root.findall("link/collision")))
        hand_links = [link for link in root.findall("link") if link.get("name").startswith(("L_", "R_"))]
        object_ids = {obj.name: i for i, obj in enumerate(robot.collision_model.objects)}
        first_tip = None
        for fraction in (0.0, 0.23, 0.51, 1.0):
            commands = data.ctrl.copy()
            for actuators in c.hand_actuators.values():
                for actuator, closed in actuators.items():
                    commands[actuator] = fraction * closed
            q = robot.state.q.copy()
            c._project_hand_commands(q, commands)
            for joint in c.hand_joint_names:
                data.qpos[m.joint(joint).qposadr] = q[robot.get_joint_offset(joint)]
            # Independent passive observation: this must not be overwritten by
            # URDF's approximate linear mimic relation in the collision model.
            data.qpos[m.joint("L_index_dip_joint").qposadr] += 0.07
            mujoco.mj_forward(m, data)
            c._update_robot_state()
            placements = robot.collision_placements(robot.state.q)
            for link in hand_links:
                name = link.get("name")
                body = m.body(name).id
                expected = np.eye(4)
                expected[:3, :3] = data.xmat[body].reshape(3, 3)
                expected[:3, 3] = data.xpos[body]
                np.testing.assert_allclose(robot.get_T_world_frame(name), expected, atol=2e-6)
                for index, collision in enumerate(link.findall("collision")):
                    origin = collision.find("origin")
                    local = tf.euler_matrix(*np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
                    local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                    np.testing.assert_allclose(placements[object_ids[f"{name}_{index}"]], expected @ local, atol=2e-6)
            tip = robot.get_T_world_frame("L_index_dip_link").copy()
            if first_tip is None:
                first_tip = tip
            hand_q = {joint: robot.get_joint(joint) for joint in c.hand_joint_names}
            c.xr_client.grips.update(left_grip=1.0, right_grip=1.0)
            c._update_ik()
            self.assertTrue(c._ik_valid)
            for joint, value in hand_q.items():
                self.assertAlmostEqual(robot.get_joint(joint), value, places=12)
        self.assertGreater(np.linalg.norm(tip[:3, 3] - first_tip[:3, 3]), 0.03)

    def test_trigger_motion_into_a_real_body_collision_is_rejected(self):
        c = self.make_controller(True)
        robot, model, data = c.robot, c.mj_model, c.mj_data
        for joint, value in THUMB_THIGH_POSE.items():
            data.qpos[model.joint(joint).qposadr] = value
        for actuator, joint in c.arm_actuators.items():
            data.ctrl[actuator] = data.qpos[model.joint(joint).qposadr][0]
        mujoco.mj_forward(model, data)
        c._update_robot_state()
        self.assertGreater(c.collision_checker.distances().min(), c.collision_margin)
        previous = data.ctrl.copy()
        # This pose clears the open thumb but the closed thumb penetrates the
        # thigh. Only the finger configuration changes; arms stay in place.
        for actuator, closed in c.hand_actuators["left"].items():
            data.ctrl[actuator] = closed
        closed_q = robot.state.q.copy()
        c._project_hand_commands(closed_q, data.ctrl)
        robot.state.q = closed_q
        self.assertLess(c.collision_checker.distances().min(), 0)
        c._update_robot_state()
        with contextlib.redirect_stdout(io.StringIO()):
            c._validate_combined_commands(previous)
        np.testing.assert_array_equal(data.ctrl, previous)
        self.assertGreater(c.collision_checker.distances().min(), c.collision_margin)

    def test_arm_motion_checks_unchanged_held_finger_targets(self):
        c = self.make_controller(True)
        robot, data = c.robot, c.mj_data
        measured = robot.state.q.copy()
        # Fingers are still open while the previously accepted close target is
        # pending. Moving the arm toward the thigh must consider that target.
        for actuator, closed in c.hand_actuators["left"].items():
            data.ctrl[actuator] = closed
        previous = data.ctrl.copy()
        for joint, value in THUMB_THIGH_POSE.items():
            robot.set_joint(joint, value)
        c.active.update(left_arm=True, right_arm=False)
        c._ik_valid = True
        c._stream_fresh = True
        c.xr_client.grips["left_trigger"] = 1.0
        with contextlib.redirect_stdout(io.StringIO()):
            c._send_command()
        np.testing.assert_array_equal(data.ctrl, previous)
        self.assertFalse(c._ik_valid)
        np.testing.assert_array_equal(robot.state.q, measured)

    def test_hand_motion_samples_nonlinear_coupling_with_tracking_error(self):
        c = self.make_controller(True)
        robot = c.robot
        source = robot.get_joint_offset("L_index_pip_joint")
        target = robot.get_joint_offset("L_index_dip_joint")
        c.hand_couplings = [(target, source, np.array([0.0, 0.0, 5.0, 0.0, 0.0]))]
        measured = robot.state.q.copy()
        measured[target] = 0.003  # Actual passive joint is off the coupling curve.
        robot.state.q = measured.copy()
        commands = c.mj_data.ctrl.copy()
        actuator = next(i for i, offset in c.hand_command_offsets.items() if offset == source)
        commands[actuator] = 0.1
        samples = []

        def distances():
            samples.append(robot.state.q.copy())
            return np.full(c.collision_pair_count, 0.1)

        with (
            patch.object(c.collision_checker, "distances", side_effect=distances),
            patch.object(c.collision_checker, "can_bound", False),
        ):
            c._validate_collision_motion(measured, commands)
        interior = []
        # The first two queries establish measured and held baselines. Inspect
        # the measured-to-command segment, up to its first endpoint.
        for q in samples[2:]:
            fraction = q[source] / 0.1
            self.assertAlmostEqual(q[target], 5 * q[source] ** 2 + (1 - fraction) * 0.003, places=10)
            if fraction >= 1 - 1e-10:
                break
            interior.append(q)
        self.assertTrue(interior)

    def test_hand_coupling_chains_are_ordered_and_invalid_constraints_rejected(self):
        c = self.make_controller(True)
        model, robot = c.mj_model, c.robot
        chain = ["L_thumb_dip_joint", "L_index_dip_joint", "L_middle_dip_joint", "L_ring_pip_joint", "L_ring_dip_joint"]
        source_name = "L_index_pip_joint"

        def coupling_model(edges, coefficients=None):
            return SimpleNamespace(
                neq=len(edges),
                eq_type=np.full(len(edges), mujoco.mjtEq.mjEQ_JOINT),
                eq_obj1id=np.array([model.joint(target).id for target, _ in edges]),
                eq_obj2id=np.array([model.joint(source).id if source is not None else -1 for _, source in edges]),
                eq_data=np.array(coefficients if coefficients is not None else [[0.1, 0.5, 0, 0, 0]] * len(edges)),
                joint=model.joint,
                actuator_trnid=model.actuator_trnid,
            )

        edges = list(zip(chain, [source_name] + chain[:-1]))
        with patch.object(c, "mj_model", coupling_model(list(reversed(edges)))):
            c._configure_hand_couplings()
        commands = c.mj_data.ctrl.copy()
        source = robot.get_joint_offset(source_name)
        actuator = next(i for i, offset in c.hand_command_offsets.items() if offset == source)
        commands[actuator] = 0.6
        q = robot.state.q.copy()
        c._project_hand_commands(q, commands)
        expected = 0.6
        for joint in chain:
            expected = 0.1 + 0.5 * expected
            self.assertAlmostEqual(q[robot.get_joint_offset(joint)], expected)

        invalid = [
            ([(chain[0], chain[1]), (chain[1], chain[0])], None, "Cyclic"),
            ([(chain[0], source_name), (chain[0], source_name)], None, "Duplicate"),
            ([(chain[0], None)], None, "two named joints"),
            ([(source_name, chain[0])], None, "passive hand joints"),
            ([(chain[0], source_name)], [[0, np.nan, 0, 0, 0]], "finite"),
        ]
        for edges, coefficients, message in invalid:
            with self.subTest(message=message), patch.object(c, "mj_model", coupling_model(edges, coefficients)):
                with self.assertRaisesRegex(ValueError, message):
                    c._configure_hand_couplings()

    def test_collision_samples_bound_nonlinear_passive_motion(self):
        c = self.make_controller(True)
        robot = c.robot
        source = robot.get_joint_offset("L_index_pip_joint")
        passive = robot.get_joint_offset("L_index_dip_joint")
        c.hand_couplings = [(passive, source, np.array([0, 0, 16.0, 0, 0]))]
        start = robot.state.q.copy()
        target = start.copy()
        target[source], target[passive] = 0.5, 4.0
        samples = c._collision_motion_samples(start, target, coupled_hands=True)
        np.testing.assert_allclose(samples[:, passive], 16 * samples[:, source] ** 2)
        np.testing.assert_array_equal(samples[-1], target)
        intervals = np.diff(np.vstack((start, samples)), axis=0)
        self.assertLessEqual(np.abs(intervals[:, 7:]).max(), c.COLLISION_CHECK_STEP_RAD + 1e-10)
        target[source] = 1000
        with self.assertRaisesRegex(RuntimeError, "sampling budget"):
            c._collision_motion_samples(start, target, coupled_hands=True)

    def test_primitive_clearance_bounds_do_not_certify_unsafe_pairs(self):
        for omnihand in (False, True):
            c = self.make_controller(omnihand)
            robot = c.robot
            home = robot.state.q.copy()
            rng = np.random.default_rng(15)
            for _ in range(12):
                robot.state.q = home.copy()
                for joint in robot.joint_names():
                    lower, upper = c.mj_model.joint(joint).range
                    robot.set_joint(joint, rng.uniform(lower, upper))
                bounds = c.collision_checker.clearance_bounds(np.full(c.collision_pair_count, c.collision_margin))
                exact = c.collision_checker.distances()
                self.assertTrue(np.all(bounds <= np.maximum(exact, 0) + 1e-7))
                certified = bounds >= c.collision_margin
                self.assertTrue(np.all(exact[certified] >= c.collision_margin))

    def test_motion_bounds_remain_conservative_near_exact_references(self):
        for omnihand in (False, True):
            c = self.make_controller(omnihand)
            robot, checker = c.robot, c.collision_checker
            home = robot.state.q.copy()
            rng = np.random.default_rng(23)
            for penetrating in (False, True):
                robot.state.q = home.copy()
                if penetrating:
                    robot.set_joint("left_shoulder_pitch_joint", 2.0)
                    robot.set_joint("left_shoulder_roll_joint", -0.06)
                checker.clear_cache()
                reference = checker.distances()
                self.assertEqual(np.min(reference) < 0, penetrating)
                for _ in range(8):
                    q = robot.state.q.copy()
                    for joint in robot.joint_names():
                        q[robot.get_joint_offset(joint)] += rng.uniform(-0.005, 0.005)
                    robot.state.q = q
                    bounds = checker.clearance_bounds(np.full(c.collision_pair_count, c.collision_margin))
                    exact = checker.distances()
                    self.assertTrue(np.all(bounds <= np.maximum(exact, 0) + 1e-7))
                    self.assertTrue(np.all(exact[bounds >= c.collision_margin] >= c.collision_margin))

    def test_fitted_primitives_cover_source_surfaces_and_reduce_empty_volume(self):
        c = self.make_controller()
        m = c.mj_model
        urdf = ET.parse(c.robot_urdf_path).getroot()
        source = ET.parse(ASSETS / "x2_ultra.urdf").getroot()
        for link in source.findall("link"):
            name = link.get("name")
            mesh_source = link.find("collision/geometry/mesh")
            if mesh_source is None:
                continue
            shapes = urdf.findall(f"link[@name='{name}']/collision")
            self.assertTrue(shapes, name)
            found = False
            for geom in range(m.ngeom):
                if m.body(m.geom_bodyid[geom]).name != name or m.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mesh = m.geom_dataid[geom]
                if m.mesh(mesh).name != name:
                    continue
                found = True
                vertices = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, m.geom_quat[geom])
                vertices = vertices @ rotation.reshape(3, 3).T + m.geom_pos[geom]
                vertices = vertices * np.fromstring(mesh_source.get("scale", "1 1 1"), sep=" ")
                faces = m.mesh_face[m.mesh_faceadr[mesh] : m.mesh_faceadr[mesh] + m.mesh_facenum[mesh]]
                triangles = vertices[faces]
                points = np.concatenate(
                    (
                        vertices,
                        triangles.mean(axis=1),
                        0.2 * triangles[:, 0] + 0.3 * triangles[:, 1] + 0.5 * triangles[:, 2],
                    )
                )
                with self.subTest(link=name):
                    self.assert_points_covered(shapes, points)
                if name == "torso_link":
                    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
                    source_volume = np.prod(upper - lower)
                    # Include cylinder bulges outside the original mesh AABB;
                    # otherwise clipping the sampling grid hides excess volume.
                    for shape in shapes:
                        origin = shape.find("origin")
                        center = np.fromstring(origin.get("xyz"), sep=" ")
                        rotation = tf.euler_matrix(*np.fromstring(origin.get("rpy"), sep=" "))[:3, :3]
                        cylinder = shape.find("geometry/cylinder")
                        if cylinder is not None:
                            half = float(cylinder.get("radius")) * np.linalg.norm(rotation[:, :2], axis=1)
                            half += float(cylinder.get("length")) / 2 * np.abs(rotation[:, 2])
                        else:
                            half = np.abs(rotation) @ (
                                np.fromstring(shape.find("geometry/box").get("size"), sep=" ") / 2
                            )
                        lower, upper = np.minimum(lower, center - half), np.maximum(upper, center + half)
                    axes = [np.linspace(a, b, 65) for a, b in zip(lower, upper)]
                    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
                    # Occupied union volume, not a sum that double-counts the
                    # intentionally overlapping chest cylinders.
                    union_volume = self.points_covered(shapes, grid).mean() * np.prod(upper - lower)
                    self.assertLess(union_volume / source_volume, 0.7)
                    self.assertGreater(
                        len([shape for shape in shapes if shape.find("geometry/cylinder") is not None]), 1
                    )
            self.assertTrue(found, name)

    def test_articulated_hand_preserves_source_collision_primitives(self):
        merged = ET.parse(ASSETS / "x2_omnihand_ik.urdf").getroot()
        for side in ("left", "right"):
            source = ET.parse(ASSETS.parents[1] / "omnihand_pro" / f"omnihand_pro_{side}.urdf").getroot()
            for link in source.findall("link"):
                if link.get("name") == "base_link":
                    continue
                actual = merged.findall(f"link[@name='{link.get('name')}']/collision")
                expected = link.findall("collision")
                if not expected:
                    self.assertTrue(actual)
                    continue
                self.assertEqual(len(actual), len(expected))
                for original, generated in zip(expected, actual):
                    self.assertEqual(original.find("origin").attrib, generated.find("origin").attrib)
                    self.assertEqual(list(original.find("geometry"))[0].tag, list(generated.find("geometry"))[0].tag)
                    self.assertEqual(
                        list(original.find("geometry"))[0].attrib, list(generated.find("geometry"))[0].attrib
                    )
