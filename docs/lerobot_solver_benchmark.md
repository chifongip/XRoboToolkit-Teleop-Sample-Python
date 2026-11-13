# LeRobot solver performance

The optimizer preserves the objective, filter, convergence tolerances, and command
validation. See [execution and warm-state behavior](lerobot_ik.md#ipopt-execution-optimizations).
Measurements below use Python 3.11.18, CasADi 3.6.7, and an Intel Core i9-14900HX in
`robojudo-xr`. They describe local synthetic replays, not hardware timing guarantees.

## Reproduce

Save the original solver and compare it with the current implementation, without
concurrent CPU workloads:

```bash
git show 5a84a10b6b2319b2c2b0bc37fcc744de46fac434:xrobotoolkit_teleop/ik/lerobot_solver.py > /tmp/lerobot_solver_baseline.py
PYTHONPATH=. python scripts/misc/benchmark_lerobot_solver.py --baseline-source /tmp/lerobot_solver_baseline.py --variants baseline all
```

Omit `--variants` to include the individual warm, slack, and function-call variants.
Those controls share the new wrapper and residual-based slack initialization, so
their timings do not attribute every difference solely to the named switch.

The benchmark uses three repeats of 40 measured frames per scenario and reverses
variant order on alternating repeats. Six warmup frames are excluded. Solver
records are assigned to measured frames explicitly; held frames or multiple
successful solve attempts fail the solver benchmark instead of shifting its
samples. Numeric FK telemetry is inside retarget timing but outside solver timing.
Every accepted command passes exhaustive signed-distance path validation outside
the timed region. Reported values are medians of the three run-level statistics.

## Reviewed implementation

The robustness review passed 142 tests and 960 measured collision replay cycles,
with zero held cycles and successful exhaustive path validation. The `review`
section of the [raw measurement artifact](lerobot_solver_benchmark.json) records
these runs separately from the initial implementation data.

| Scenario | Solver median, original → reviewed | Solver p95, original → reviewed | Retarget p95, original → reviewed |
| --- | ---: | ---: | ---: |
| Fingers stationary | 14.11 → 8.65 ms | 18.02 → 10.03 ms | 20.90 → 12.69 ms |
| Fingers closing | 15.18 → 8.99 ms | 17.04 → 10.92 ms | 21.28 → 15.20 ms |
| Servo lag | 14.53 → 8.85 ms | 18.24 → 10.08 ms | 21.28 → 12.97 ms |
| Repeated penetrating snapshot | 24.62 → 21.85 ms | 26.79 → 23.36 ms | 260.85 → 259.82 ms |

All comparison gates passed: at least 10% improvement in normal solver medians,
no scenario exceeding a 5% regression in the median of run-level solver/retarget
p95 values, and tracking-error increases below 1 mm and 0.01 rad. Recovery still
spends most of its time on collision processing.

Regression coverage includes constant soft inequalities requiring slack, invalid
constraint shapes/nonfinite inputs, graph-compilation failure and retry, cached
graph transitions, cold/warm direct-function parity, iteration exhaustion,
float32 acknowledgements, proposal rollback, offline pose cycles, and bridge
behavior. Benchmark accounting tests reject retries and held frames instead of
reusing successful solves from another frame.

The final review passed 144 tests and a further 48 functional replay cycles across
all four scenarios, with zero held cycles and successful exhaustive path checks.
It added coverage for slack-bound multiplier invalidation when padding becomes a
supplied row, non-real/nonnumeric inputs, and incompatible bounds before graph
selection. The artifact's `final_review` section records these checks; the timing
table above remains the three-repeat measurement from the preceding review.

## Initial implementation measurements

The [archived measurements](lerobot_solver_benchmark.json) precede the robustness
review. They used the last 40 successful solves for solver statistics rather than
explicit frame boundaries, and direct-call ablations used warm initialization even
without an accepted seed. They remain a record of that implementation, rather
than measurements of the reviewed code. All 135 tests, 2,400 measured collision
replay cycles, and 800 measured simulation cycles passed at that revision.

| Scenario | Solver median, original → optimized | Solver p95, original → optimized | Retarget p95, original → optimized |
| --- | ---: | ---: | ---: |
| Fingers stationary | 13.80 → 8.48 ms | 16.09 → 9.73 ms | 18.83 → 12.48 ms |
| Fingers closing | 15.36 → 8.29 ms | 16.70 → 9.46 ms | 20.85 → 13.51 ms |
| Servo lag | 13.90 → 8.34 ms | 16.52 → 9.61 ms | 19.38 → 12.27 ms |
| Repeated penetrating snapshot | 24.27 → 21.55 ms | 25.09 → 22.10 ms | 241.97 → 241.82 ms |

The 200-cycle simulation comparisons reported arms median/p95 of 10.73/15.31 →
8.49/10.14 ms and Omnihand 13.95/17.62 → 11.01/12.42 ms, with zero failures.
Setup times were 486 → 397 ms and 505 → 516 ms respectively; first-cycle times
were 60 → 17 ms and 55 → 18 ms. Compilation and capacity growth can add cold-cycle
latency even when steady-state percentiles improve.

## Derivative expansion and collision-disabled latency

The current solver differentiates the MX objective before expanding its gradient
and Hessian functions. Blanket NLP `expand=True` is avoided: broader validation
found NaN gradients during obstructed homing. Numeric kinematics are refreshed
when proposals restore their model, including on failure. Explicit state copies
and backend-owned proposal rollback remove redundant copying; the numeric adapter
computes collision Jacobians only when requested.

Empty collision rows have a dedicated cached graph containing only joint bounds.
Hard-only, soft-only, and combined collision graphs retain independent capacities,
so collision graph growth cannot make the empty graph retain padded inequalities.

The applied implementation was compared with the pre-change Git revision using
three alternating-order repeats of 200 measured frames for each scenario. Six
warmup frames were excluded. Collisions were disabled; diagnostic FK and command
acknowledgements were outside retarget timing for both implementations. The
baseline loads its own solver, transaction backend, and numeric adapter from Git.
The host was an Intel Core i9-14900HX, Python 3.11.18, CasADi 3.6.7. These are
synthetic workstation measurements, not Jetson timing guarantees.

| Scenario | Retarget median, before → applied | Retarget p95, before → applied |
| --- | ---: | ---: |
| Stationary fingers | 4.04 → 2.08 ms | 4.32 → 2.20 ms |
| Closing fingers | 4.14 → 2.07 ms | 4.39 → 2.18 ms |
| Servo lag | 4.03 → 2.09 ms | 4.29 → 2.21 ms |

All 3,600 measured comparison frames succeeded. The applied implementation had
zero 20 ms overruns across its 1,800 measured frames and a maximum retarget time
of 2.47 ms. Maximum command difference was `5.96e-8` rad, with mean tracking-error
changes below `1e-9` m/rad. See [raw measurements](lerobot_latency_benchmark.json).
The implementation passed 79 native regression tests, including obstructed
homing, proposal failure cache restoration, standalone solver rollback, empty
collision graph reuse after growth, and offline pose cycles.

Reproduce with the recorded `baseline_commit` from the raw measurements:

```bash
PYTHONPATH=. python scripts/misc/benchmark_lerobot_latency.py \
  --baseline-ref PRE_CHANGE_COMMIT --output /tmp/lerobot-latency.json
```

Omit `--baseline-ref` to measure the working tree alone, including on the Jetson.
The replay sends no robot commands. Collision-enabled geometry processing remains
a separate bottleneck: the preceding prototype validation did not improve every
scenario's end-to-end p95, even though solver time improved. Neither this replay
nor selecting 50 Hz establishes a hard real-time guarantee.
