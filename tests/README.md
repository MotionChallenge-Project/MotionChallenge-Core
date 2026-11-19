# Protocol checks

```bash
uv sync --locked
uv run --locked python -m unittest discover -v
```

CPU checks cover original motion selection and conversion, reference sampling, kinematics, metric and failure calculations, strict policy observations, artifact integrity, complete trace verification, reproduction identity and public commands.

The four MJWarp simulation checks run when CUDA is available. Select their device with `MOTIONCHALLENGE_TEST_DEVICE=cuda:0`. They check active-world behavior, partial reset and refill, inactive-slot isolation, finite-state rejection and constraint-capacity overflow. CPU-only environments report these four checks as skipped.
