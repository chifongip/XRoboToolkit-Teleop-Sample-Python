"""Numeric transition validator and portable model identity, without an IK solver."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from xrobotoolkit_teleop.common.transition_validation import TransitionValidation
from xrobotoolkit_teleop.utils.collision_checks import CollisionChecker

CONTRACT = "robojudo_executor_v1"


def model_fingerprint(profile):
    root = ET.parse(profile.spec.urdf_path).getroot()
    # Generated URDFs contain host-specific absolute mesh paths. Hash contents.
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", hashlib.sha256(Path(mesh.get("filename")).read_bytes()).hexdigest())
    description = {
        "urdf": ET.tostring(root, encoding="unicode"),
        "pairs": json.loads(Path(profile.spec.collision_pairs_path).read_text()),
        "observed": profile.observed_joints,
        "arms": profile.spec.joints,
        "limits": profile.spec.limits,
        "velocities": profile.velocities,
        "hands": None if profile.hands is None else vars(profile.hands),
    }
    return hashlib.sha256(json.dumps(description, sort_keys=True, allow_nan=False).encode()).hexdigest()


class ExecutorModel(TransitionValidation):
    def __init__(self, profile, settings):
        self.profile = profile
        self.settings = settings
        self.fingerprint = model_fingerprint(profile)
        self.hand_names = list(profile.hands.active)
        self.collision_margin = settings.collision_margin
        self.robot = None
        self.collision_checker = None
        self.hand_couplings = []
        if settings.collision_checking:
            from xrobotoolkit_teleop.ik.pinocchio_model import PinocchioModel

            self.robot = PinocchioModel(profile.spec)
            self.hand_couplings = [
                (self.robot.get_joint_offset(a), self.robot.get_joint_offset(b), np.array(c))
                for a, b, c in profile.hands.couplings
            ]
            self.collision_checker = CollisionChecker(self.robot)

    def validate_transition(self, measured, previous, candidate):
        if not self.settings.collision_checking:
            return
        robot = self.robot
        if set(measured) != set(self.profile.observed_joints):
            raise ValueError("incomplete executor measurements")
        for name, value in measured.items():
            robot.set_joint(name, value)
        measured_q = robot.state.q.copy()
        previous_q, candidate_q = measured_q.copy(), measured_q.copy()
        for name in candidate:
            offset = robot.get_joint_offset(name)
            previous_q[offset], candidate_q[offset] = previous[name], candidate[name]
        self._project_hands(previous_q)
        self._project_hands(candidate_q)
        self.collision_checker.clear_cache()
        self._validate_motion(measured_q, previous_q, candidate_q)
