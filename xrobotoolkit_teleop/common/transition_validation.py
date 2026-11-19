"""Combined command transition checks shared by the bridge and robot executor."""

import math
from itertools import product

import numpy as np
from xrobotoolkit_teleop.ik.safety import CollisionPolicy


class TransitionValidation(CollisionPolicy):
    def _application_samples(self, start, candidate, independent_offsets=()):
        samples = self._collision_motion_samples(start, candidate, bool(self.hand_couplings))
        if not self.hand_names and not independent_offsets:
            return samples
        # Arms and the two hand handles have independent application timing.
        # Include partial application, not just their synchronized diagonal.
        groups = []
        finger_offsets = set()
        hand_groups = self.profile.hands.groups.values() if self.hand_names else ()
        for names in hand_groups:
            offsets = {self.robot.get_joint_offset(n) for n in names}
            offsets.update(passive for passive, source, _ in self.hand_couplings if source in offsets)
            finger_offsets.update(offsets)
            groups.append(sorted(offsets))
        groups.insert(0, sorted(set(range(len(start))) - finger_offsets - set(independent_offsets)))
        groups.extend([offset] for offset in independent_offsets)
        paths = []
        for offsets in groups:
            if np.array_equal(start[offsets], candidate[offsets]):
                paths.append(start[None, offsets])
                continue
            group_samples = samples
            if independent_offsets:
                # Give each independently applied group its own sample spacing.
                group_target = start.copy()
                group_target[offsets] = candidate[offsets]
                group_samples = self._collision_motion_samples(start, group_target, bool(self.hand_couplings))
            paths.append(np.unique(np.vstack((start[offsets], group_samples[:, offsets])), axis=0))
        count = math.prod(len(path) for path in paths)
        if count > self.MAX_COLLISION_SAMPLES:
            raise RuntimeError("independent arm/hand motion exceeds sampling budget")

        def configurations():
            for parts in product(*paths):
                configuration = start.copy()
                for offsets, values in zip(groups, parts, strict=True):
                    configuration[offsets] = values
                yield configuration
            # Recovery progress must be assessed at the final target.
            yield candidate

        return configurations()

    def _validate_motion(self, measured, previous, candidate, independent_offsets=()):
        if not all(np.isfinite(configuration).all() for configuration in (measured, previous, candidate)):
            raise RuntimeError("non-finite robot command configuration")
        if not self.settings.collision_checking:
            return
        robot = self.robot
        try:
            robot.state.q = measured
            measured_distances = self.collision_checker.clearances(self.collision_margin)
            for start in (measured, previous):
                robot.state.q = start
                distances = self.collision_checker.clearances(self.collision_margin)
                floor = np.minimum(np.minimum(distances, measured_distances), self.collision_margin)
                for configuration in self._application_samples(start, candidate, independent_offsets):
                    robot.state.q = configuration
                    values = (
                        self.collision_checker.clearance_bounds(floor)
                        if self.collision_checker.can_bound and np.all(floor > 0)
                        else None
                    )
                    if values is None or np.any(values < floor):
                        values = self.collision_checker.clearances(self.collision_margin)
                    rejected = np.flatnonzero(values < floor - 1e-6)
                    if rejected.size:
                        index = rejected[0]
                        a, b = self.collision_checker.pair_indices[index]
                        objects = robot.collision_model.objects
                        raise RuntimeError(
                            "Robot command path violates self-collision clearance "
                            f"({objects[a].link} vs {objects[b].link})"
                        )
                    if not self.hand_names:
                        floor = np.maximum(floor, np.minimum(values, self.collision_margin))
                violating = measured_distances < self.collision_margin - 1e-6
                if (
                    start is measured
                    and np.any(violating)
                    and not np.any(values[violating] > measured_distances[violating] + 1e-8)
                ):
                    raise RuntimeError("Robot collision recovery made no progress")
        finally:
            robot.state.q = candidate
            robot.update_kinematics()

    def _project_hands(self, configuration):
        for passive, source, coefficients in self.hand_couplings:
            configuration[passive] = np.polynomial.polynomial.polyval(configuration[source], coefficients)
