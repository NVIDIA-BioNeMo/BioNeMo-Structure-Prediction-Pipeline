# Folding ColabFold Kernel Image

GPU scientific image for the `fold` action with the colabfold backend. Built
from the official ColabFold image `ghcr.io/sokrypton/colabfold:1.6.2-cuda12`.

Carries `colabfold_batch`, Contract+Control+Runtime wheels (via `--no-deps`), and the
CUDA compat entrypoint. The entrypoint reads `$BSPP_CUDA_COMPAT_DIR` and
degrades gracefully if the compat dir is absent (the colabfold base image may
carry its own CUDA runtime).

## Base-image fallback

If the colabfold base image's default Python is not 3.12, `build.sh` fails
with a clear message. Install Python 3.12 via `pyenv` or `conda` into a
separate prefix and use that interpreter for the BSPP wheel install and smoke.

## Build, smoke, push, import

```bash
containers/scripts/build.sh folding colabfold
containers/folding/colabfold/smoke-local.sh bspp-orchestration:folding-colabfold
containers/scripts/push.sh folding colabfold
sbatch containers/scripts/pull-sqsh.sh folding-colabfold
```

The base-image digest is pinned in `image-lock.json` and verified at build time.
