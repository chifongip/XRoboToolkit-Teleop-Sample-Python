# RoboJuDo bridge

The release deployment guide is maintained in the parent repository:
[XRoboToolkit integration](../../../docs/xrobotoolkit_teleop.md).

Use `python -m xrobotoolkit_teleop.integrations.robojudo --help` for the supported deployment options.
Both `--robot x2` and `--robot x2_omnihand` use robot-authoritative desired-target execution.
Configure collision checking only in RoboJuDo; the bridge adopts the validated robot policy.
Both peers must use matching release sources and assets.
