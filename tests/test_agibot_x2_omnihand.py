"""Headless integration checks for optional X2 Omnihand Pro hands; no XR service needed."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from test_agibot_x2_simulation import ASSETS, FakeXR
from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController
from xrobotoolkit_teleop.utils.mujoco_utils import calc_placo_q_from_mujoco_qpos


def load_script(name):
    # Load entrypoints by path; an installed dependency also owns a `scripts`
    # package, so importing scripts.simulation is ambiguous in this environment.
    path = ASSETS.parents[2] / "scripts/simulation" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script("build_x2_omnihand_assets")
entrypoint = load_script("teleop_agibot_x2_mujoco")


class HandXR(FakeXR):
    def __init__(self):
        super().__init__()
        self.grips.update(left_trigger=0.0, right_trigger=0.0)


class X2OmnihandProTests(unittest.TestCase):
    def make_controller(
        self, xml_path=ASSETS / "scene_omnihand.xml", robot_urdf_path=ASSETS / "x2_omnihand_ik.urdf", **kwargs
    ):
        with (
            patch("xrobotoolkit_teleop.common.base_teleop_controller.XrClient", HandXR),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            c = AgibotX2TeleopController(str(xml_path), str(robot_urdf_path), **kwargs)
        self.addCleanup(c.xr_client.close)
        return c

    @staticmethod
    def settle_commands(controller):
        for _ in range(100):
            controller.step_control()

    def thumb_finger_pairs(self, model):
        pairs = []
        for prefix in ("L", "R"):
            thumb, fingers = [], []
            for i in range(model.ngeom):
                # Query collision meshes even though contact forces are disabled.
                if model.geom_group[i] != 3 or model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                name = model.body(model.geom_bodyid[i]).name
                if name.startswith(f"{prefix}_thumb"):
                    thumb.append(i)
                elif name.startswith(tuple(f"{prefix}_{finger}" for finger in ("index", "middle", "ring", "pinky"))):
                    fingers.append(i)
            self.assertEqual((len(thumb), len(fingers)), (5, 12))
            pairs.extend((a, b) for a in thumb for b in fingers)
        return pairs

    def assert_thumb_clearance(self, controller, pairs):
        m, d = controller.mj_model, controller.mj_data
        for thumb, finger in pairs:
            distance = mujoco.mj_geomDistance(m, d, thumb, finger, 0.01, None)
            self.assertGreaterEqual(
                distance,
                -1e-4,
                f"{m.body(m.geom_bodyid[thumb]).name} overlaps {m.body(m.geom_bodyid[finger]).name}",
            )

    def test_structure_mounts_and_home(self):
        c = self.make_controller()
        m, d = c.mj_model, c.mj_data
        self.assertEqual((m.nq, m.nv, m.nu, m.neq), (52, 52, 38, 14))
        self.assertEqual(len(c.arm_actuators), 14)
        self.assertEqual([len(v) for v in c.hand_actuators.values()], [12, 12])
        for side, prefix, roll, abad in (("left", "L", -0.6, 0.6), ("right", "R", 0.6, -0.6)):
            targets = {m.joint(m.actuator_trnid[i, 0]).name: target for i, target in c.hand_actuators[side].items()}
            # Omnihand Pro mirrors only roll/abduction, not thumb MCP/PIP flexion.
            for suffix, expected in (("roll", roll), ("abad", abad), ("mcp", -0.05), ("pip", -0.6)):
                self.assertEqual(targets[f"{prefix}_thumb_{suffix}_joint"], expected)
            for i in c.hand_actuators[side]:
                self.assertEqual(m.jnt_margin[m.actuator_trnid[i, 0]], 0.0)
        source = ET.parse(ASSETS / "x2_fixed.xml").getroot()
        generated = ET.parse(ASSETS / "x2_omnihand.xml").getroot()
        wrist_links = ET.parse(ASSETS / "omnihand_wrist_links.xml").getroot()
        arm_home = dict(
            zip(
                (j.get("name") for j in source.findall(".//body/joint")),
                map(float, source.find("keyframe/key").get("qpos").split()),
            )
        )
        for side, prefix, y, yaw in (("left", "L", -0.001, 1.58), ("right", "R", 0.001, -1.58)):
            palm = m.body(f"{prefix}_palm")
            wrist = m.body(f"{side}_wrist_roll_link")
            self.assertEqual(palm.parentid[0], wrist.id)
            np.testing.assert_allclose(palm.pos, [0, y, -0.049])
            # Independently compose URDF Rz(yaw) Ry(pitch) Rx(roll).
            cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(3.14), np.sin(3.14)
            expected = np.array([[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0, cp]])
            actual = d.xmat[wrist.id].reshape(3, 3).T @ d.xmat[palm.id].reshape(3, 3)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            self.assertAlmostEqual(palm.mass[0], 0.831592534870634)
            # Short v1.4 wrist geometry replaces the old end geometry and mass.
            mesh_name = "left_wrist_roll_link10.stl" if side == "left" else "right_wrist_roll_link2.stl"
            self.assertEqual(
                generated.find(f"asset/mesh[@name='{side}_wrist_roll_link']").get("file"),
                f"meshes/{mesh_name}",
            )
            self.assertAlmostEqual(wrist.mass[0], 0.081)
            source_wrist = wrist_links.find(f"link[@name='{side}_wrist_roll_link']/inertial")
            np.testing.assert_allclose(wrist.ipos, np.fromstring(source_wrist.find("origin").get("xyz"), sep=" "))
            tensor = source_wrist.find("inertia")
            expected_inertia = np.array(
                [
                    [float(tensor.get(k)) for k in row]
                    for row in (("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz"))
                ]
            )
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, wrist.iquat)
            rotation = rotation.reshape(3, 3)
            np.testing.assert_allclose(rotation @ np.diag(wrist.inertia) @ rotation.T, expected_inertia, atol=1e-12)
        for joint, value in arm_home.items():
            self.assertAlmostEqual(d.qpos[m.joint(joint).qposadr[0]], value)
        for i, name in c.arm_actuators.items():
            self.assertAlmostEqual(d.ctrl[i], arm_home[name])
        # Only the floor may retain contact bits in this scene.
        for i in range(m.ngeom):
            if m.geom(i).name != "floor":
                self.assertEqual(m.geom_contype[i], 0)
                self.assertEqual(m.geom_conaffinity[i], 0)

    def test_generated_files_match_sources(self):
        documents = builder.build_documents()
        self.assertEqual(
            builder.collision_pairs(documents["x2_fixed.urdf"]), json.loads((ASSETS / "collisions.json").read_text())
        )
        self.assertEqual(
            builder.articulated_collision_pairs(documents["x2_omnihand_ik.urdf"]),
            json.loads((ASSETS / "collisions_omnihand.json").read_text()),
        )
        for name, root in documents.items():
            with self.subTest(file=name):
                self.assertEqual(
                    ET.tostring(root, encoding="utf-8", xml_declaration=True), (ASSETS / name).read_bytes()
                )

    def test_urdf_inertia_rotation(self):
        link = ET.fromstring(
            """<link name="example"><inertial>
            <origin xyz="0.01 0.02 0.03" rpy="0 0 1.5707963267948966"/>
            <mass value="4"/>
            <inertia ixx="1" iyy="2" izz="3" ixy="0" ixz="0" iyz="0"/>
        </inertial></link>"""
        )
        body = ET.Element("body")
        builder.copy_urdf_inertial(body, link)
        inertia = body.find("inertial")
        np.testing.assert_allclose(np.fromstring(inertia.get("fullinertia"), sep=" "), [2, 1, 3, 0, 0, 0], atol=1e-12)
        self.assertEqual(inertia.get("pos"), "0.01 0.02 0.03")
        self.assertEqual(inertia.get("mass"), "4")

    def test_cli_accepts_equivalent_default_scene_path(self):
        with (
            patch.object(entrypoint, "AgibotX2TeleopController") as factory,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            entrypoint.main(
                xml_path=str(ASSETS / "scene.xml"),
                with_omnihand=True,
                arm_velocity_limit=0.4,
                hand_velocity_limit=0.8,
                control_frequency=60.0,
            )
            self.assertEqual(Path(factory.call_args.args[0]).resolve(), ASSETS / "scene_omnihand.xml")
            self.assertEqual(Path(factory.call_args.args[1]).resolve(), ASSETS / "x2_omnihand_ik.urdf")
            self.assertEqual(factory.call_args.kwargs["arm_velocity_limit"], 0.4)
            self.assertEqual(factory.call_args.kwargs["hand_velocity_limit"], 0.8)
            self.assertEqual(factory.call_args.kwargs["control_frequency"], 60.0)
            factory.return_value.run.assert_called_once()
            factory.reset_mock()
            entrypoint.main()
            self.assertEqual(factory.call_args.kwargs["control_frequency"], 50.0)
            factory.reset_mock()
            with self.assertRaisesRegex(ValueError, "custom --xml-path"):
                entrypoint.main(xml_path="custom.xml", with_omnihand=True)
            factory.assert_not_called()

    def test_hand_centers_match_palm_geometry_ik_and_markers(self):
        c = self.make_controller()
        m, d = c.mj_model, c.mj_data
        home = d.qpos.copy()
        rng = np.random.default_rng(7)
        for _ in range(5):
            d.qpos[:] = home
            for joint in c.arm_actuators.values():
                j = m.joint(joint)
                d.qpos[j.qposadr[0]] = np.clip(home[j.qposadr[0]] + rng.uniform(-0.15, 0.15), *j.range)
            mujoco.mj_forward(m, d)
            c.placo_robot.state.q = calc_placo_q_from_mujoco_qpos(m, c.placo_robot, d.qpos)
            c.placo_robot.update_kinematics()
            for side, prefix in (("left", "L"), ("right", "R")):
                name = f"{side}_hand_center"
                frame, palm, wrist = (m.body(n).id for n in (name, f"{prefix}_palm", f"{side}_wrist_roll_link"))
                center_in_palm = d.xmat[palm].reshape(3, 3).T @ (d.xpos[frame] - d.xpos[palm])
                # Independently measured bounding-box centers of the palm meshes.
                expected = (
                    [-0.0013905745, 0.00004094094, 0.062206626]
                    if side == "left"
                    else [-0.0013897624, -0.000053508207, 0.062206626]
                )
                np.testing.assert_allclose(center_in_palm, expected, atol=1e-8)
                np.testing.assert_allclose(d.xmat[frame], d.xmat[wrist], atol=1e-12)
                ik_pose = c.placo_robot.get_T_world_frame(name)
                np.testing.assert_allclose(ik_pose[:3, 3], d.xpos[frame], atol=1e-4)
                np.testing.assert_allclose(ik_pose[:3, :3], d.xmat[frame].reshape(3, 3), atol=1e-3)
                c.effector_task[f"{side}_arm"].T_world_frame = ik_pose
            c._update_mocap_target()
            for side in ("left", "right"):
                marker = c.target_mocap_idx[f"{side}_arm"]
                np.testing.assert_allclose(d.mocap_pos[marker], d.xpos[m.body(f"{side}_hand_center").id], atol=1e-4)

    def test_wrong_ik_hand_center_model_is_rejected_and_closes_xr(self):
        xr = HandXR()
        with (
            patch("xrobotoolkit_teleop.common.base_teleop_controller.XrClient", return_value=xr),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(ValueError, "Omnihand Pro requires x2_omnihand_ik.urdf"):
                AgibotX2TeleopController(str(ASSETS / "scene_omnihand.xml"), str(ASSETS / "x2_fixed.urdf"))
        self.assertTrue(xr.closed)

    def test_controller_rotation_targets_the_hand_center_in_both_models(self):
        for scene, urdf in (("scene.xml", "x2_fixed.urdf"), ("scene_omnihand.xml", "x2_omnihand_ik.urdf")):
            with self.subTest(scene=scene):
                c = self.make_controller(ASSETS / scene, ASSETS / urdf)
                self.settle_commands(c)
                c.xr_client.grips.update(left_grip=1.0, right_grip=1.0)
                c.step_control()
                targets = {side: c.effector_task[f"{side}_arm"].T_world_frame.copy() for side in ("left", "right")}
                for side in ("left", "right"):
                    c.xr_client.poses[f"{side}_controller"][3:] = [0, np.sin(0.1), 0, np.cos(0.1)]
                self.settle_commands(c)
                for side, initial in targets.items():
                    task = c.effector_task[f"{side}_arm"].T_world_frame
                    np.testing.assert_allclose(task[:3, 3], initial[:3, 3], atol=1e-12)
                    self.assertGreater(np.linalg.norm(task[:3, :3] - initial[:3, :3]), 0.1)
                    center, _ = c._get_link_pose(f"{side}_hand_center")
                    self.assertLess(np.linalg.norm(center - initial[:3, 3]), 0.015)

    def test_invalid_custom_actuators_are_rejected(self):
        c = self.make_controller()
        arm_names = set(c.arm_actuators.values())
        m = c.mj_model
        # A position command has different meaning with a non-unit gear or a
        # torque motor; refuse either instead of silently applying wrong forces.
        for gear, bias in ((2.0, mujoco.mjtBias.mjBIAS_AFFINE), (1.0, mujoco.mjtBias.mjBIAS_NONE)):
            with self.subTest(gear=gear, bias=bias):
                m.actuator_gear[0, 0] = gear
                m.actuator_biastype[0] = bias
                with self.assertRaisesRegex(ValueError, "position servos"):
                    c._configure_actuators(arm_names)
        m.actuator_gear[0, 0] = 1.0
        m.actuator_biastype[0] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_trnid[0, 0] = m.actuator_trnid[1, 0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            c._configure_actuators(arm_names)

    def test_actuator_order_does_not_change_control(self):
        model = ET.parse(ASSETS / "x2_omnihand.xml").getroot()
        model.find("compiler").set("meshdir", str(ASSETS))
        actuators = model.find("actuator")
        actuators[:] = list(reversed(actuators))
        home = model.find("keyframe/key[@name='home']")
        home.set("ctrl", " ".join(reversed(home.get("ctrl").split())))
        scene = ET.parse(ASSETS / "scene_omnihand.xml")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            ET.ElementTree(model).write(directory / "robot.xml")
            scene.getroot().find("include").set("file", "robot.xml")
            scene.write(directory / "scene.xml")
            c = self.make_controller(directory / "scene.xml")
        initial = c.mj_data.ctrl.copy()
        c.xr_client.grips.update(left_grip=1.0, left_trigger=0.25, right_trigger=0.75)
        c.step_control()
        c.xr_client.poses["left_controller"][0] += 0.03
        for _ in range(100):
            c.step_control()
        self.assertTrue(c._ik_valid)
        for i, joint in c.arm_actuators.items():
            expected = c.placo_robot.get_joint(joint) if joint.startswith("left_") else initial[i]
            self.assertAlmostEqual(c.mj_data.ctrl[i], expected)
        for side, trigger in (("left", 0.25), ("right", 0.75)):
            for i, closed in c.hand_actuators[side].items():
                self.assertAlmostEqual(c.mj_data.ctrl[i], trigger * closed)

    def test_independent_triggers_and_arm_commands(self):
        c = self.make_controller()
        arms = list(c.arm_actuators)
        left, right = (list(c.hand_actuators[s]) for s in ("left", "right"))
        arm_home = c.mj_data.ctrl[arms].copy()
        c.xr_client.grips["left_trigger"] = 0.5
        self.settle_commands(c)
        np.testing.assert_allclose(c.mj_data.ctrl[left], np.array(list(c.hand_actuators["left"].values())) * 0.5)
        np.testing.assert_array_equal(c.mj_data.ctrl[right], 0)
        np.testing.assert_array_equal(c.mj_data.ctrl[arms], arm_home)
        # Arm IK updates cannot overwrite either hand, even with interleaved joints.
        c.xr_client.grips.update(left_grip=1.0, right_trigger=1.0)
        self.settle_commands(c)
        hands = c.mj_data.ctrl[left + right].copy()
        c.xr_client.poses["left_controller"][0] += 0.03
        for _ in range(20):
            c.step_control()
        self.assertTrue(c._ik_valid)
        self.assertGreater(np.max(np.abs(c.mj_data.ctrl[arms] - arm_home)), 0.001)
        np.testing.assert_array_equal(c.mj_data.ctrl[left + right], hands)
        c.xr_client.grips.update(left_grip=0.0, left_trigger=0.0, right_trigger=0.0)
        arm_hold = c.mj_data.ctrl[arms].copy()
        self.settle_commands(c)
        np.testing.assert_array_equal(c.mj_data.ctrl[left + right], 0)
        np.testing.assert_array_equal(c.mj_data.ctrl[arms], arm_hold)

    def test_stale_stream_invalid_triggers_and_clamping(self):
        c = self.make_controller()
        left, right = (list(c.hand_actuators[s]) for s in ("left", "right"))
        c.xr_client.grips.update(left_trigger=0.5, right_trigger=0.5)
        self.settle_commands(c)
        held = c.mj_data.ctrl.copy()
        for invalid in (np.nan, np.inf, -np.inf):
            c.xr_client.grips.update(left_trigger=invalid, right_trigger=0.25)
            self.settle_commands(c)
            np.testing.assert_array_equal(c.mj_data.ctrl[left], held[left])
            np.testing.assert_allclose(c.mj_data.ctrl[right], np.array(list(c.hand_actuators["right"].values())) * 0.25)
        c.xr_client.grips.update(left_trigger=10.0, right_trigger=-2.0)
        self.settle_commands(c)
        np.testing.assert_allclose(c.mj_data.ctrl[left], list(c.hand_actuators["left"].values()))
        np.testing.assert_array_equal(c.mj_data.ctrl[right], 0)
        held = c.mj_data.ctrl.copy()
        c.xr_client.advance_timestamps = False
        c.xr_client.grips.update(left_trigger=0.0, right_trigger=1.0)
        c._last_xr_update_at -= 1.0
        c.step_control()
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        c.xr_client.advance_timestamps = True
        c.step_control()
        for i, limit in c.hand_velocity_limits.items():
            self.assertLessEqual(abs(c.mj_data.ctrl[i] - held[i]), limit * c.dt + 1e-12)
        self.settle_commands(c)
        np.testing.assert_array_equal(c.mj_data.ctrl[left], 0)
        np.testing.assert_allclose(c.mj_data.ctrl[right], list(c.hand_actuators["right"].values()))

    def test_independent_hand_velocity_cap_and_urdf_limits(self):
        hand_limits = {
            joint.get("name"): float(joint.find("limit").get("velocity"))
            for side in ("left", "right")
            for joint in ET.parse(ASSETS.parents[1] / "omnihand_pro" / f"omnihand_pro_{side}.urdf")
            .getroot()
            .findall("joint")
            if joint.find("limit") is not None
        }
        for requested in (0.5, 2.0, 8.0):
            with self.subTest(hand_velocity_limit=requested):
                c = self.make_controller(arm_velocity_limit=0.25, hand_velocity_limit=requested)
                expected = {
                    i: min(requested, hand_limits[c.mj_model.joint(c.mj_model.actuator_trnid[i, 0]).name])
                    for actuators in c.hand_actuators.values()
                    for i in actuators
                }
                self.assertEqual(c.hand_velocity_limits, expected)
                for joint in c.arm_actuators.values():
                    offset = c.placo_robot.get_joint_v_offset(joint)
                    self.assertAlmostEqual(c.placo_robot.model.velocity_limit[offset], 0.25)
                for left, right in ((1.0, 0.5), (0.0, 1.0), (1.0, 0.0)):
                    c.xr_client.grips.update(left_trigger=left, right_trigger=right)
                    for _ in range(100):
                        previous = c.mj_data.ctrl.copy()
                        c.step_control()
                        for i, limit in expected.items():
                            self.assertLessEqual(abs(c.mj_data.ctrl[i] - previous[i]) / c.dt, limit + 1e-12)
                        for actuators in c.hand_actuators.values():
                            progress = [c.mj_data.ctrl[i] / closed for i, closed in actuators.items() if closed != 0]
                            np.testing.assert_allclose(progress, progress[0], atol=1e-12)
                self.assertFalse(np.any(c.mj_data.warning.number))

    def test_hand_speed_limits_follow_configured_control_frequency(self):
        for frequency in (25.0, 60.0, 100.0):
            with self.subTest(frequency=frequency):
                c = self.make_controller(control_frequency=frequency, hand_velocity_limit=1.0)
                c.xr_client.grips.update(left_trigger=1.0, right_trigger=1.0)
                for _ in range(3):
                    previous = c.mj_data.ctrl.copy()
                    c.step_control()
                    for actuator, limit in c.hand_velocity_limits.items():
                        self.assertLessEqual(
                            abs(c.mj_data.ctrl[actuator] - previous[actuator]), limit / frequency + 1e-12
                        )
                self.assertGreater(np.linalg.norm(c.mj_data.ctrl - previous), 0)

    def test_repeated_motion_is_stable_and_coupled(self):
        c = self.make_controller()
        m, d = c.mj_model, c.mj_data
        arms = list(c.arm_actuators.values())
        home = {j: d.qpos[m.joint(j).qposadr[0]] for j in arms}
        for trigger in (1.0, 0.0, 1.0, 0.0):
            c.xr_client.grips.update(left_trigger=trigger, right_trigger=trigger)
            for _ in range(300):
                c.step_control()
            self.assertTrue(np.all(np.isfinite(d.qpos)))
            self.assertTrue(np.all(np.isfinite(d.qvel)))
            self.assertEqual(d.ncon, 0)
            self.assertFalse(np.any(d.warning.number))
            for joint in arms:
                self.assertLess(abs(d.qpos[m.joint(joint).qposadr[0]] - home[joint]), 0.04)
            for side, actuators in c.hand_actuators.items():
                for i, closed in actuators.items():
                    joint = m.actuator_trnid[i, 0]
                    actual = d.qpos[m.jnt_qposadr[joint]]
                    self.assertLess(abs(actual - closed * trigger), 0.1, m.joint(joint).name)
            for eq in range(m.neq):
                q1 = d.qpos[m.jnt_qposadr[m.eq_obj1id[eq]]]
                q2 = d.qpos[m.jnt_qposadr[m.eq_obj2id[eq]]]
                expected = np.polynomial.polynomial.polyval(q2, m.eq_data[eq, :5])
                self.assertLess(abs(q1 - expected), 0.02)

    def test_thumb_does_not_penetrate_fingers_during_closing_and_opening(self):
        for speed in (0.5, 2.0, 8.0):
            with self.subTest(hand_velocity_limit=speed):
                c = self.make_controller(hand_velocity_limit=speed)
                pairs = self.thumb_finger_pairs(c.mj_model)
                for trigger in (1.0, 0.0):
                    c.xr_client.grips.update(left_trigger=trigger, right_trigger=trigger)
                    for step in range(300):
                        c.step_control()
                        if step % 5:
                            continue
                        self.assert_thumb_clearance(c, pairs)

    def test_partial_trigger_reversals_preserve_thumb_clearance(self):
        # Review reproducer: independent joint ramps caused 2.17 mm of overlap
        # at the default speed when each trigger value was held for 200 ms.
        sequence = (0.8, 0.8, 1, 1, 0.8, 1, 0.5, 0, 0.2, 0.5, 0.5, 0.8, 0.8, 1)
        for speed in (0.5, 2.0, 8.0):
            with self.subTest(hand_velocity_limit=speed):
                c = self.make_controller(hand_velocity_limit=speed)
                pairs = self.thumb_finger_pairs(c.mj_model)
                for trigger in sequence:
                    c.xr_client.grips.update(left_trigger=trigger, right_trigger=trigger)
                    for _ in range(20):
                        c.step_control()
                        self.assert_thumb_clearance(c, pairs)


if __name__ == "__main__":
    unittest.main()
