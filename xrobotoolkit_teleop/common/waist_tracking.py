"""Pelvis-relative tracking and arm clutches, shared by both X2 workflows."""

import time
from dataclasses import dataclass

import numpy as np

from xrobotoolkit_teleop.common.teleop_settings import DEFAULTS
from xrobotoolkit_teleop.utils.geometry import relative_target, xr_pose_to_matrix


@dataclass(frozen=True)
class WaistSample:
    pelvis_pose: np.ndarray
    controllers: dict[str, np.ndarray]
    body_timestamp_ns: int | None
    xr_timestamp_ns: int  # Shared SDK timestamp, not per-controller validity.
    observed_at: float  # Local getter observation time, not packet arrival time.
    hip_positions: np.ndarray | None = None  # XR-world positions in left, right order.
    body_head_timestamp_ns: int | None = None  # Body joint 15, not the separate headset pose.


def _pelvis_control_axes(hip_positions, pelvis_rotation):
    """Map pelvis-local directions to robot forward, left, and up."""
    hips = np.asarray(hip_positions, dtype=float)
    if hips.shape != (2, 3) or not np.isfinite(hips).all():
        raise ValueError("invalid hip positions for waist heading")
    # Use the PICO pelvis joint's local up axis. Projecting against room-up
    # at engagement would bake the operator's starting pitch/roll into the map.
    with np.errstate(over="ignore", invalid="ignore"):
        left = pelvis_rotation.T @ (hips[0] - hips[1])
        left[1] = 0.0
        width = np.linalg.norm(left)
    if not np.isfinite(width) or width < 1e-3:
        raise ValueError("hip positions do not define a lateral pelvis axis")
    left /= width
    up = np.array([0.0, 1.0, 0.0])
    forward = np.cross(left, up)
    return np.stack((forward, left, up))


class WaistTracking:
    # Conservative per-sample jump guards, not physical robot safety limits.
    MAX_TRANSLATION_STEP_M = 0.5
    MAX_ROTATION_STEP_RAD = 1.0

    def __init__(self, timeout_s=DEFAULTS.tracking_timeout_s):
        self.timeout_s = timeout_s
        self._last_stamps = None
        self._body_clock = None
        self._advanced_at = [None, None]
        self._last_observed_at = None
        self._previous = None
        self._relative = None
        self._pending_control_axes = None
        self._control_axes = None
        self.reason = "waist tracking not sampled"
        self.valid = False
        self.can_capture = False
        self._references = {}

    def clear(self):
        """Release arm clutches without discarding stream freshness history."""
        self._references.clear()

    def has_reference(self, arm):
        return arm in self._references

    def release(self, arm):
        """Return whether the arm had a captured reference."""
        return self._references.pop(arm, None) is not None

    def invalidate(self, reason):
        """Hold both arms; require fresh timestamp progression before recovery."""
        self.reason = reason
        self.valid = False
        self.can_capture = False
        self.clear()
        self._advanced_at = [None, None]
        self._last_observed_at = None
        self._relative = None
        return None

    def _fresh(self, sample, now):
        packet_stamp = sample.xr_timestamp_ns
        if type(packet_stamp) is not int or packet_stamp <= 0:
            raise ValueError("invalid XR packet timestamp")
        body_stamp = sample.body_timestamp_ns
        body_clock = "body"
        if type(body_stamp) is not int or body_stamp <= 0:
            body_stamp = sample.body_head_timestamp_ns
            body_clock = "body head joint"
        if type(body_stamp) is not int or body_stamp <= 0:
            raise ValueError("body frame timestamp unavailable")
        if not np.isfinite(sample.observed_at) or not 0 <= now - sample.observed_at <= self.timeout_s:
            raise ValueError("waist sample timed out")
        stamps = (packet_stamp, body_stamp)
        if (
            self._last_stamps is None
            or body_clock != self._body_clock
            or any(stamp < old for stamp, old in zip(stamps, self._last_stamps))
        ):
            self._last_stamps = stamps
            self._body_clock = body_clock
            self._previous = None
            self.invalidate("waiting for advancing XR packet and body frame timestamps")
            return False
        for i, (stamp, old) in enumerate(zip(stamps, self._last_stamps)):
            if stamp > old:
                self._advanced_at[i] = sample.observed_at
        self._last_stamps = stamps
        for stream, advanced_at in zip(("XR packet", "body frame"), self._advanced_at):
            if advanced_at is None:
                self.reason = f"waiting for {stream} timestamp progression"
                return False
            if now - advanced_at > self.timeout_s:
                self.invalidate(f"{stream} timestamp stopped advancing")
                return False
        if self._last_observed_at is not None and sample.observed_at - self._last_observed_at > self.timeout_s:
            self.invalidate("waist sample gap exceeded tracking timeout")
            return False
        return True

    def update(self, sample, error=None):
        """Return controller transforms in pelvis coordinates, or hold both arms."""
        self.valid = False
        self.can_capture = False
        try:
            if sample is None:
                return self.invalidate(error or "waist sample unavailable")
            previous_stamps = self._last_stamps
            if not self._fresh(sample, time.monotonic()):
                return None
            waist = xr_pose_to_matrix(sample.pelvis_pose)
            controllers = {side: xr_pose_to_matrix(sample.controllers[side]) for side in ("left", "right")}
            if self._relative is not None and self._last_stamps[1] == previous_stamps[1]:
                # Controllers can advance while the sender repeats a body frame.
                self._last_observed_at = sample.observed_at
                self.valid = True
                self.reason = None
                return self._relative
            rotation = waist[:3, :3]
            if self._control_axes is None:
                self._pending_control_axes = _pelvis_control_axes(sample.hip_positions, rotation)
            relative = {}
            for side, pose in controllers.items():
                transform = np.eye(4)
                transform[:3, :3] = rotation.T @ pose[:3, :3]
                with np.errstate(over="ignore", invalid="ignore"):
                    transform[:3, 3] = rotation.T @ (pose[:3, 3] - waist[:3, 3])
                if not np.isfinite(transform[:3, 3]).all():
                    return self.invalidate("invalid pelvis-relative controller position")
                relative[side] = transform
            # Raw poses detect common recenter jumps; relative poses detect hand jumps.
            current = {"waist": waist, **controllers, **{f"relative_{s}": p for s, p in relative.items()}}
            previous, self._previous = self._previous, current
            if previous is not None:
                for name, pose in current.items():
                    old = previous[name]
                    angle = np.arccos(np.clip((np.trace(pose[:3, :3] @ old[:3, :3].T) - 1) / 2, -1, 1))
                    if (
                        np.linalg.norm(pose[:3, 3] - old[:3, 3]) > self.MAX_TRANSLATION_STEP_M
                        or angle > self.MAX_ROTATION_STEP_RAD
                    ):
                        return self.invalidate("waist/controller pose discontinuity")
            self._last_observed_at = sample.observed_at
            self._relative = relative
            self.valid = True
            self.can_capture = True
            self.reason = None
            return relative
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return self.invalidate(str(exc))

    def target(self, arm, controller, hand, scale):
        """Apply the clutch delta with a new hand anchor on the first call for each grip."""
        if not self.valid:
            raise ValueError("waist tracking unavailable")
        if not self.has_reference(arm):
            if not self.can_capture:
                raise ValueError("waiting for a fresh body frame to engage")
            if self._control_axes is None:
                self._control_axes = self._pending_control_axes
            self._references[arm] = (controller.copy(), hand.copy())
            return hand.copy()
        reference, initial_hand = self._references[arm]
        return relative_target(controller, reference, initial_hand, self._control_axes, scale)


def read_waist_sample(client):
    try:
        return client.get_waist_sample(), None
    except (AttributeError, ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
        return None, f"waist sample unavailable: {exc}"
