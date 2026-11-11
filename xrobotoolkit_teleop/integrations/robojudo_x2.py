"""Backward-compatible X2 bridge entrypoint."""

import tyro

from xrobotoolkit_teleop.integrations.robojudo import RoboJuDoBridge, main


class RoboJuDoX2Bridge(RoboJuDoBridge):
    """Existing X2 constructor; transport and retargeting are profile driven."""

    def __init__(
        self,
        command_endpoint="tcp://127.0.0.1:8559",
        feedback_endpoint="tcp://127.0.0.1:8571",
        settings=None,
        xr_client=None,
        *,
        ik_backend="placo",
    ):
        from xrobotoolkit_teleop.common.x2_defaults import DEFAULTS

        super().__init__(
            command_endpoint,
            feedback_endpoint,
            DEFAULTS if settings is None else settings,
            xr_client,
            robot="x2",
            ik_backend=ik_backend,
        )


if __name__ == "__main__":
    tyro.cli(main)
