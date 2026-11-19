# MJWarp evaluator

See [protocol](PROTOCOL.md), [batching](BATCHING.md) and [adapter interface](ADAPTERS.md).

The locked runtime uses MuJoCo 3.11.0, mujoco-warp 3.11.0, Warp 1.18.0 and PyTorch 2.14.1. CUDA simulation and policy inference use float32; TensorFloat-32 is disabled. Metric accumulation uses float64.

Reference body poses are precomputed. Actual body poses are refreshed after each control interval. Capacity overflow and non-finite active simulation states invalidate a full-suite result.
