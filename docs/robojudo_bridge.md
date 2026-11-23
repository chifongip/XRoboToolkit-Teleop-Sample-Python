# XR Bridge for RoboJuDo-Plus

Use the [X2 XR teleoperation guide](../../../docs/xrobotoolkit_teleop.md) in your RoboJuDo-Plus checkout
for installation, preset selection, launch commands, hardware setup, and runtime logs. In a standalone
XRoboToolkit checkout, find the guide at `docs/xrobotoolkit_teleop.md` in the parent project,
[RoboJuDo-Plus](https://github.com/chifongip/RoboJuDo-Plus).

The bridge module is `xrobotoolkit_teleop.integrations.robojudo`. Both X2 profiles (`x2` and `x2_omnihand`)
support `fixed-heading`, `waist`, and `pelvis-ik` with Placo or LeRobot. Select the matching RoboJuDo preset for pelvis IK.
RoboJuDo supplies the validated model identity and collision policy over robot feedback. Use matching RoboJuDo
and XRoboToolkit revisions.

```bash
conda activate robojudo-xr
python -m xrobotoolkit_teleop.integrations.robojudo --help
```

For PICO calibration and body-frame diagnostics, see the [X2 waist-relative reference](agibot_x2.md#waist-relative-reference).
For backend and native dependency details, see the [IK backend guide](lerobot_ik.md).
