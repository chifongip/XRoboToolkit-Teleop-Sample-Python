"""Replay fixed-base X2 or G1 arm motion through mocked XR input."""

import json
import sys

import tyro

from xrobotoolkit_teleop.ik import BackendName
from xrobotoolkit_teleop.simulation.offline_replay import ReplayRobot, run_replay
from xrobotoolkit_teleop.simulation.unitree_g1_model import G1Variant


def main(
    ik_backend: BackendName = BackendName.placo,
    with_omnihand: bool = False,
    cycles: int = 2,
    move_seconds: float = 2.0,
    hold_seconds: float = 1.0,
    visualize: bool = False,
    visualize_ik: bool = False,
    robot: ReplayRobot = "x2",
    variant: G1Variant | None = None,
):
    """Offline arm pose replay. Exits with status 1 on a failed check.

    Args:
        ik_backend: Registered IK backend to exercise.
        with_omnihand: Use the articulated X2 hand scene with fingers open.
        robot: Robot to replay.
        variant: G1 joint-count variant. Defaults to 29dof for G1.
        cycles: Number of complete home-to-target-and-back cycles.
        move_seconds: Simulated duration of each smooth transition.
        hold_seconds: Simulated settling time at each endpoint.
        visualize: Open the MuJoCo physics viewer and pace the replay in real time.
        visualize_ik: Open the shared browser IK viewer and pace the replay in real time.
    """
    try:
        result = run_replay(
            ik_backend=ik_backend,
            with_omnihand=with_omnihand,
            cycles=cycles,
            move_seconds=move_seconds,
            hold_seconds=hold_seconds,
            visualize=visualize,
            visualize_ik=visualize_ik,
            robot=robot,
            variant=variant,
        )
    except KeyboardInterrupt:
        print("Offline replay interrupted before completion.", file=sys.stderr)
        raise SystemExit(130) from None
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}))
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    tyro.cli(main)
