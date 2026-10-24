# Batched evaluation

Motion count, active slots, CPU threads and inference batch size are independent.
For example, a registered 10,000-motion suite can run with 2,048 active slots,
32 CPU threads and at most 256 policy samples per call:

```sh
uv run evaluate.py --jobs 2048 --threads 32 --inference-batch-size 256
```

The selected suite still determines motion count. LAFAN40 uses at most 40 active
slots. Sizes need not be powers of two or divide one another. A batch limit is a
maximum, not a promise of a full batch; the final batch and partial resets can be
smaller. Core runs complete motions and refills slots without splitting motions.

By default, batched adapters use up to four slots per available CPU (at most 64),
scalar adapters use up to eight, and physics uses up to eight threads. Motion
count caps the actual slots. Override these bounds for larger workloads; they
are conservative defaults, not a guarantee of optimal settings for every engine.

## Array interface

Declare `"array_batch_entrypoint": "adapter:create_batch"` in `runtime.json`.
The factory receives `(adapter_root, resolved_runtime)` and returns:

```python
class Policy:
    def reset_batch(self, slots, states, references, seeds):
        # Clear history for exactly these slot IDs. Copy any retained inputs.
        ...

    def act_batch(self, slots, states, references):
        # Build observations and call your engine here.
        # Return ControlBatch(target_q, target_qd, kp, kd, feedforward_torque).
        # Each output has shape [len(slots), number_of_joints].
        ...

    def close(self):
        ...
```

| Input | Shape |
| --- | --- |
| `StateBatch` joint positions / velocities | N × J |
| Base quaternion, linear / angular velocity | N × 4, N × 3 |
| `ReferenceBatch` joint positions / velocities | N × K × J |
| Reference root position / quaternion | N × K × 3, N × K × 4 |
| Reference body positions / quaternions | N × K × B × 3, N × K × B × 4 |

N is the current call size, J the canonical joint count, K the declared reference
window, B the reference body count. Quaternions use XYZW; units and axes are the
same as the packet SDK. Full field definitions are in `sdk.py`.

Input arrays are read-only views valid until the call returns. Copy values kept
in history. Calls can contain a reordered subset of slots; reset only those slots
and return controls in the supplied order. Never infer motion identity from slot
numbers. Random generators and recurrent state must be independent per slot.

Alternatively, `batch_entrypoint` provides tuples of SDK packets and expects a
`ControlCommand` per slot. The scalar interface creates independent controller
instances in the same process; it does not automatically combine model forwards.
Use adapter-local RNG state rather than process-global random state. All three
interfaces use the same simulation, observation boundary and scoring path.

## Resource limits

Core and policy share one evaluation process; MuJoCo uses its CPU thread pool.
There is no per-step IPC. An adapter may encapsulate a separate inference service
when its engine requires one.
Verification runs afterward with a CPU-bounded process pool. Memory still grows
with active simulator state, loaded reference motion data and adapter histories.
Choose slots and batch size to fit host memory and the adapter's device memory;
more slots are not always faster. Fixed-size engines may pad or split internally.
Core has no CUDA or TensorRT dependency.

Execution records include actual slots, physics threads and phase timings. A
short synthetic scaling test validates allocation, refill and transport; it is
not a throughput estimate for long motions or large neural policies.

## Timing

The result records native MuJoCo `step`, `forward`, `reset` and `jac` time,
reference/trace bookkeeping, metrics and policy calls separately. Policy timing
also separates input assembly, adapter execution and control validation. These
nested timers must not be added to their enclosing totals. Direct calls have
zero IPC time; input assembly can still copy arrays. Adapter execution includes
its observation construction and inference, not just the neural network.

Progress reports include elapsed time and cumulative phase timings every five
seconds. Reference timing separates native FK, interpolation, velocity work and
numeric storage. FK models and scratch data persist; array consumers reuse body
snapshots without reconstructing per-body objects.

Use complete motions for throughput comparisons, or explicitly label a duration
limit. Compare the same checkpoint, adapter, slots, CPU threads and workload;
record startup separately. The independent trace verifier remains part of the
end-to-end evaluation time.

For online results only, use `uv run evaluate.py --no-traces`. This skips trajectory
encoding, compression, writes and independent replay verification. The same
rollout still computes success, completion and both metric ranges. Results are
marked `completed_without_traces`, not `trace_verified`; the default keeps traces
for reproducible leaderboard evidence.
