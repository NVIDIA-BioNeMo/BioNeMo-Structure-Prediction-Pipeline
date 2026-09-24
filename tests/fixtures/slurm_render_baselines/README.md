# SLURM Render Baselines

These fixtures are normalized golden outputs for RunSpec-driven SLURM rendering.
They cover the native archive worker script/report, the analysis metadata
finalizer script/report, and the scalar folding action scripts.

`folding_scalar/` freezes today's non-array scalar folding render:
one `.sbatch` per folding action (`msa-flatten-000001`, `split-000001`,
`preprocess-000001`, `fold-000001`, `canonical-pair-000001`), each rendered
to `--nodes=1` / `--ntasks=1` with `%j` logs and no `--array`. It is the byte
oracle that later renderer/topology changes (tasks-per-node, array packing)
must preserve for the scalar case.

Regenerate after an intentional rendering change with:

```bash
UPDATE_RENDER_BASELINES=1 .venv/bin/pytest tests/test_slurm/test_render_baselines.py
UPDATE_RENDER_BASELINES=1 .venv/bin/pytest tests/test_slurm/test_folding_render_baselines.py
```

The test normalizes temporary directories, RunSpec hashes, and render timestamps
before comparing or updating these files.
