"""Sample SDK timestamps without IK or ZMQ to diagnose stream interruptions."""

import argparse
import csv
import math
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReceiveRate:
    timeout_s: float = 0.25
    first_valid_at: float | None = None
    last_sample_at: float | None = None
    last_change_at: float | None = None
    last_stamp: int | None = None
    updates: int = 0
    invalid_samples: int = 0
    timestamp_resets: int = 0
    timeout_episodes: int = 0
    timed_out: bool = False
    update_gaps: deque[float] = field(default_factory=lambda: deque(maxlen=4096))
    max_update_gap_s: float = 0.0
    max_poll_gap_s: float = 0.0
    max_source_gap_s: float = 0.0

    def observe(self, stamp, now):
        sample_gap = 0.0 if self.last_sample_at is None else now - self.last_sample_at
        if self.last_sample_at is not None:
            self.max_poll_gap_s = max(self.max_poll_gap_s, sample_gap)
        self.last_sample_at = now
        if type(stamp) is not int or stamp <= 0:
            self.invalid_samples += 1
        elif self.last_stamp is None:
            self.first_valid_at = self.last_change_at = now
            self.last_stamp = stamp
        elif stamp != self.last_stamp:
            update_gap = now - self.last_change_at
            self.update_gaps.append(update_gap)
            self.max_update_gap_s = max(self.max_update_gap_s, update_gap)
            if stamp > self.last_stamp:
                self.max_source_gap_s = max(self.max_source_gap_s, (stamp - self.last_stamp) / 1e9)
                self.updates += 1
            else:
                self.timestamp_resets += 1
            self.last_change_at = now
            self.last_stamp = stamp
            self.timed_out = False
        age = None if self.last_change_at is None else now - self.last_change_at
        if age is not None and age > self.timeout_s and not self.timed_out:
            self.timeout_episodes += 1
            self.timed_out = True
        return age, sample_gap

    def report(self, now):
        elapsed = 0.0 if self.first_valid_at is None else now - self.first_valid_at
        rate = self.updates / elapsed if elapsed > 0 else 0.0
        age = None if self.last_change_at is None else round(1000 * (now - self.last_change_at), 1)
        gaps = sorted(self.update_gaps)
        p95 = gaps[math.ceil(0.95 * len(gaps)) - 1] if gaps else 0.0
        max_gap = max(self.max_update_gap_s, 0.0 if self.last_change_at is None else now - self.last_change_at)
        return (
            f"SDK updates={rate:.1f} Hz | age={age} ms | "
            f"update gap recent-p95/max={1000 * p95:.1f}/{1000 * max_gap:.1f} ms | "
            f"max poll gap={1000 * self.max_poll_gap_s:.1f} ms | "
            f"max source timestamp gap={1000 * self.max_source_gap_s:.1f} ms | "
            f"timeouts={self.timeout_episodes} resets={self.timestamp_resets} invalid={self.invalid_samples}"
        )


def _positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=_positive, default=30.0, help="measurement duration in seconds")
    parser.add_argument("--sample-hz", type=_positive, default=200.0, help="SDK polling frequency")
    parser.add_argument("--timeout", type=_positive, default=0.25, help="timestamp timeout threshold in seconds")
    parser.add_argument("--csv", type=Path, help="optional per-sample CSV output")
    args = parser.parse_args()
    print("Stop other XR SDK clients first. This probe samples timestamps; it sends no robot commands.", flush=True)
    # Lazy import keeps --help and statistics tests independent of the SDK.
    from xrobotoolkit_teleop.common.xr_client import XrClient

    probe = ReceiveRate(args.timeout)
    with ExitStack() as resources:
        client = XrClient()
        resources.callback(client.close)
        started = time.monotonic()
        next_report = started + 2.0
        writer = None
        if args.csv is not None:
            output = resources.enter_context(args.csv.open("w", newline=""))
            writer = csv.writer(output)
            writer.writerow(["elapsed_s", "timestamp_ns", "unchanged_age_s", "poll_gap_s"])
        try:
            while time.monotonic() - started < args.duration:
                sample_started = time.monotonic()
                stamp = client.get_timestamp_ns()
                now = time.monotonic()
                age, gap = probe.observe(stamp, now)
                if writer is not None:
                    writer.writerow([now - started, stamp, age, gap])
                if now >= next_report:
                    print(probe.report(now), flush=True)
                    next_report = now + 2.0
                time.sleep(max(0, 1 / args.sample_hz - (time.monotonic() - sample_started)))
        except KeyboardInterrupt:
            pass
        finally:
            print("Final: " + probe.report(time.monotonic()), flush=True)


if __name__ == "__main__":
    main()
