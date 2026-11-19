# Model adapter interface

Each model repository provides runtime.mjwarp.json, admission.mjwarp.json, tensor_adapter.py, convert_weights.py, evaluate.py, pyproject.toml, uv.lock and a pinned Core submodule.

The tensor adapter constructor receives model root, runtime, canonical joint names, environment count, device and converted-weight directory. reset(slots) resets only selected environments. act(state, reference) returns a tensor shaped [environments, 5, joints]: target position, target velocity, proportional gain, derivative gain and feedforward torque.

The SDK exposes only allowed observations. References contain root/joint quantities; adapters own further reference computation. Histories and recurrent states remain independent per environment.

Conversion hooks receive root, runtime, source, output, artifacts and device. They write only to the supplied output/cache directory. Original checkpoints, auxiliary assets and converted artifacts are content-checked. Additional production files are pinned through tensor_assets_sha256.

Validation compares original inference, observation assembly, histories, reset behavior and complete 4096-environment inference. Keep development proofs immutable. Store evaluation/reproduction reports separately without changing admission identities.
