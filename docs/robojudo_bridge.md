# RoboJuDo bridge

The release deployment guide is maintained in the parent repository:
[XRoboToolkit integration](../../../docs/xrobotoolkit_teleop.md).

Use `python -m xrobotoolkit_teleop.integrations.robojudo --help` for the supported deployment options.
Both `--robot x2` and `--robot x2_omnihand` use robot-authoritative desired-target execution.
Configure collision checking only in RoboJuDo; the bridge adopts the validated robot policy.
Both peers must use matching release sources and assets.

The bridge and RoboJuDo print concise INFO summaries once per second, without extra flags:

```text
XR: send=49.9 Hz | completion RTT=32.4 ms avg / 48.1 ms max | tracking
RoboJuDo: receive=49.8 Hz | control=50.0 Hz | complete=31.2 Hz
```

`send` counts desired commands successfully published by the bridge, excluding visualization-only frames.
`receive` counts valid desired commands accepted by RoboJuDo; `control` counts completed pipeline iterations,
excluding idle feedback servicing; `complete` counts commands acknowledged after successful arm submission and,
for OmniHand, both hand writes. A running 50 Hz pipeline can complete fewer than 50 fresh commands per second.

Completion RTT runs from bridge publication to the first matching completion feedback received, including the
return feedback path. It uses only the bridge's monotonic clock and existing command identifiers. Duplicate
feedback is counted once; missing, superseded, and unobserved acknowledgements produce no latency sample.
The log displays `RTT=n/a` when a window has no samples. Pending timestamp storage is bounded and cleared on
session/generation changes; logging does not gate command execution.

RTT does not establish physical arrival or firmware acceptance. One-way publication-to-receipt latency requires
synchronized clocks and is not reported. These summaries replace the local bridge loop-processing latency log.

In either tracking mode, holding the XR HOME button requires a live XR timestamp
stream. HOME holds the fingers and clears the arm clutch references. Release both
grips before resuming teleoperation. A robot-requested HOME can run with XR offline.

## Waist-relative reference

Both X2 profiles support `--tracking-reference waist` (default: `fixed-heading`):

```bash
python -m xrobotoolkit_teleop.integrations.robojudo --tracking-reference waist
```

Complete PICO's native full-body calibration before starting. No extra calibration
file or SDK changes are required. See the [X2 reference behavior, tracking checks,
and diagnostic instructions](agibot_x2.md#waist-relative-reference). Waist mode
checks both the top-level XR packet timestamp and body-frame freshness.

Waist mode changes bridge retargeting only. Schema version 3, torso-relative
visualization, robot-owned execution, limits, and collision policy remain unchanged.
Startup, bridge restart, and robot generation changes require grip release
before engagement. Waist loss stops desired tracking commands; RoboJuDo's existing
hold/timeout behavior applies, so physical settling is not instantaneous. Recovery
captures fresh references without replaying outage motion. Physical robot
qualification remains pending.
