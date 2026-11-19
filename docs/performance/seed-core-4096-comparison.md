# 4096-world evaluation throughput

Each run uses an NVIDIA A100-SXM4-80GB on cuda:3 after a 20-second zero-utilization check, with 50 warmup control steps and 500 measured control steps. Control runs at 20 ms with four 5 ms physics steps and sliding friction 1.0.

Background CUDA contexts present during the pre-check, by run: seed: 0, seed-newton50: 0, core: 0.

| Runtime and policy | Newton maximum | Constraint capacity | TF32 matmul | ms / vector step | Active transitions / s |
| --- | ---: | ---: | --- | ---: | ---: |
| SEED native checkpoint | 10 | 1024 | True | 42.782 | 95740.5 |
| SEED native checkpoint | 50 | 1024 | True | 43.026 | 95199.1 |
| Core / HEFT checkpoint | 50 | 4096 | False | 38.509 | 106364.4 |

The timed window includes actor inference, simulation, observation/reference updates and scoring. Preparation and initial reset are excluded. All runs use float32 actions without CUDA autocast. The line-search maximum is 20 and the contact capacity is 100 per world. Core also records pose traces. The policies and runtime configurations differ; the table compares complete evaluation throughput.

Core elapsed time divided by SEED elapsed time: 0.9001 with native Newton 10, and 0.8950 with Newton 50.

Detailed runtime versions, physics values, memory measurements, timing scope and numerical checks are recorded in [the JSON comparison](seed-core-4096-comparison.json).

The native SEED baseline loads its saved 380-dimensional command configuration: relative joint positions, body-frame root-position commands, and body-frame root-orientation commands. Its root-position command depends on actual global position; this native checkpoint is used only for the speed comparison. Core / HEFT uses the admitted observation contract without actual global position or linear velocity.

All three pre-checks found zero background CUDA contexts, zero allocated device memory, and zero GPU utilization. All runs ended with finite joint/root poses. Core retains overflow flags from every physics substep; the native SEED check covers the final control step.
