"""Read-only pelvis/controller stream recorder; JSON lines on stdout."""

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from xrobotoolkit_teleop.common.waist_tracking import WaistTracking, read_waist_sample
from xrobotoolkit_teleop.common.xr_client import XrClient


def json_safe(value):
    """Keep diagnostic records valid JSON even when the SDK supplies NaN/Inf."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=50.0)
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.duration, args.rate)):
        parser.error("duration and rate must be finite and positive")
    tracking = WaistTracking()
    xr = XrClient()
    start = time.monotonic()
    previous = None
    advances = [0, 0]
    previous_pelvis_stamp = None
    pelvis_advances = 0
    try:
        deadline = start
        while time.monotonic() - start < args.duration:
            sample, error = read_waist_sample(xr)
            elapsed = time.monotonic() - start
            relative = tracking.update(sample, error)
            pelvis_stamp, pelvis_error = None, None
            if sample is not None:
                try:
                    stamp = xr.get_pelvis_timestamp_ns()
                    if type(stamp) is not int:
                        raise ValueError("expected an integer pelvis timestamp")
                    pelvis_stamp = stamp
                except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
                    pelvis_error = str(exc)
            if pelvis_stamp is not None and pelvis_stamp > 0:
                if previous_pelvis_stamp is not None and pelvis_stamp > previous_pelvis_stamp:
                    pelvis_advances += 1
                previous_pelvis_stamp = pelvis_stamp
            if sample:
                stamps = (sample.body_timestamp_ns, sample.xr_timestamp_ns)
                if previous:
                    advances = [
                        n + int(type(a) is int and type(b) is int and a > b)
                        for n, a, b in zip(advances, stamps, previous)
                    ]
                previous = stamps
            record = {
                "elapsed_s": elapsed,
                "sample": asdict(sample) if sample else None,
                "error": error,
                "hold_reason": tracking.reason,
                "observed_update_hz": [n / max(elapsed, 1e-9) for n in advances],
                "pelvis_timestamp_ns": pelvis_stamp,
                "pelvis_timestamp_error": pelvis_error,
                "pelvis_observed_update_hz": pelvis_advances / max(elapsed, 1e-9),
                "relative": relative,
            }
            print(json.dumps(json_safe(record), allow_nan=False), flush=True)
            deadline += 1 / args.rate
            time.sleep(max(0.0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        xr.close()


if __name__ == "__main__":
    main()
