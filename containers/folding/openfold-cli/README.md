# Folding OpenFold-CLI Kernel Image

GPU scientific image for the `fold` action with the openfold-cli backend. Built
from the public Docker Hub base `nvidia/cuda:12.1.1-devel-ubuntu22.04` (no NGC,
no entitlement, anonymous pull). Installs torch via
`pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121`
(pins its own torch (cu121); no longer mirrors a variant), then the OpenFold scientific deps
and OpenFold itself from the pinned source commit.

Carries `run_pretrained_openfold.py`, Contract+Control+Runtime wheels (via `--no-deps`),
`hhsuite`/`hmmer`/`kalign`, and the CUDA compat entrypoint.

## numpy version note

The openfold-cli image pins `numpy>=1.26,<2` (required by OpenFold). The runtime
wheel declares `numpy>=2.0`, but `pip install --no-deps` means that declared
dependency is not enforced. `pip check` would report this mismatch; this is
acceptable and documented.

## Build, smoke, push, import

```bash
containers/scripts/build.sh folding openfold-cli
containers/folding/openfold-cli/smoke-local.sh bspp-orchestration:folding-openfold-cli
containers/scripts/push.sh folding openfold-cli
sbatch containers/scripts/pull-sqsh.sh folding-openfold-cli
```

The base-image digest is pinned in `image-lock.json` and verified at build time.
