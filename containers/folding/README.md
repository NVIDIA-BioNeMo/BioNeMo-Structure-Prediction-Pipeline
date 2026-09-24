# Folding Container Images

This directory defines **four images** for the folding Execution Runtime:

| Image | Directory | Purpose | Base image |
|---|---|---|---|
| folding-runtime | `runtime/` | Non-fold actions (msa-flatten, split, preprocess, canonical-pair) | `docker.io/nvidia/cuda:12.6.3-base-ubuntu24.04` |
| folding-colabfold | `colabfold/` | `fold` action with colabfold backend | `ghcr.io/sokrypton/colabfold:1.6.2-cuda12` |
| folding-openfold-cli | `openfold-cli/` | `fold` action with openfold-cli backend | `nvidia/cuda:12.1.1-devel-ubuntu22.04` (Docker Hub) |
| folding-bioir | `bioir/` | `fold` action with bioir backend | `nvidia/cuda:13.0.3-devel-ubuntu24.04` (Docker Hub) |

All base images are **public** (Docker Hub, GHCR, PyPI, GitHub only). No NGC
`nvcr.io` base images, no NGC entitlement, anonymous pull.

The **runtime image** carries the Contract+Control+Runtime wheels (via `--no-deps`),
`/usr/bin/lz4`, `/usr/bin/rsync`, `/usr/bin/tar`, `/usr/bin/flock`, and **s5cmd**
(for remote-bundle MSA inputs via the publish-to-swiftstack transport). It is a
folding-executor-only image, not a general `bspp-orchestration-runtime` image.
It downloads and sha256-verifies the pinned pixi binary (v0.80.0) inside the
Dockerfile and resolves its conda environment non-locked from the trimmed
`pixi.toml` (no colabfold). No host-side pixi and no committed `pixi.lock`.

The **kernel images** each install an explicit runtime dependency set
(click/pydantic/pyyaml, plus numpy for openfold-cli) before the `--no-deps`
wheel install, with a build-time import assertion. The openfold-cli image
pins `numpy>=1.26,<2` (required by OpenFold; see the numpy version
mismatch note).

`openfold-trt` remains a fail-closed contract value in `FOLDING_BACKENDS`. No
image is built for it. The executor raises
`OPENFOLD_TRT_DEFERRED_MODEL_FN_ERROR` before any publication.

## Build, smoke, push, import

```bash
# Build one image
containers/scripts/build.sh folding runtime
containers/scripts/build.sh folding colabfold
containers/scripts/build.sh folding openfold-cli
containers/scripts/build.sh folding bioir

# Build all four
containers/scripts/build.sh folding all

# Smoke
containers/folding/runtime/smoke-local.sh bspp-orchestration:folding-runtime
containers/folding/colabfold/smoke-local.sh bspp-orchestration:folding-colabfold
containers/folding/openfold-cli/smoke-local.sh bspp-orchestration:folding-openfold-cli
containers/folding/bioir/smoke-local.sh bspp-orchestration:folding-bioir

# Push
containers/scripts/push.sh folding runtime
containers/scripts/push.sh folding colabfold
containers/scripts/push.sh folding openfold-cli
containers/scripts/push.sh folding bioir

# Import on cluster
sbatch containers/scripts/pull-sqsh.sh folding-runtime
sbatch containers/scripts/pull-sqsh.sh folding-colabfold
sbatch containers/scripts/pull-sqsh.sh folding-openfold-cli
sbatch containers/scripts/pull-sqsh.sh folding-bioir
```

## Pinned base-image digests

The three kernel image `image-lock.json` files carry pinned base-image
digests for `base_image.linux_amd64_digest`. The base-image digests are
pinned in the image lock and verified at build time. The runtime
image-lock.json reuses a known-good real digest from the previous
monolithic lock.

## Configurable Install Mode

### `mount_orchestration_source` Cluster Profile field

When `mount_orchestration_source: true` is set on a Cluster Profile, the
folding renderer mounts `orchestration_repo` read-only at
`/workspace/bspp-orchestration` on every folding action and sets
`BSPP_ORCHESTRATION_DEV_MOUNT=1` in the srun environment. The container
entrypoint resolves the orchestration source baked-first:

- **Baked (default)** → uses the baked Contract+Control+Runtime wheels already
  installed in the image.
- **`BSPP_ORCHESTRATION_DEV_MOUNT=1` + valid mount** → override mode:
  editable-installs Contract+Control+Runtime with
  `pip install -e --no-build-isolation` (using hatchling baked into each
  image). The mounted commit must equal the baked commit unless
  `BSPP_ORCHESTRATION_DEV_MOUNT_ALLOW_MISMATCH=1` is set (fail-closed).
- **Mount present but malformed** → **fail-closed exit 1**. This includes a
  non-directory (e.g., a regular file) at the mount path.
- **`BSPP_INSTALL_MODE_SKIP=1`** → skips mount detection entirely, uses
  baked mode.

The helper exports `BSPP_ORCHESTRATION_SOURCE` (`"override"` or `"baked"`)
and, when a non-empty commit is determined,
`BSPP_ORCHESTRATION_PROVENANCE_COMMIT`.

### Default behavior

When `mount_orchestration_source` is `false` or absent (the **default**), no
mount is emitted and the container uses baked wheels. This preserves
baked-wheel determinism as the production default.

### Warning: absolute path required

When `mount_orchestration_source=true`, `orchestration_repo` must be an
**absolute path** to a valid orchestration source checkout containing both
`packages/orchestration-contract/pyproject.toml` and
`packages/orchestration-runtime/pyproject.toml`. A relative path fails at
**render time** (the renderer's `_validate_mount_value` rejects non-absolute
paths). An empty directory, partial checkout, or directory missing those
files will cause every folding action to **exit 1** at container startup. A
non-directory (e.g., a regular file) at the mount path also exits 1. A
non-git directory that nonetheless contains both pyproject.toml files will
enter override mode with **unset provenance** (not exit 1).

### Conflict warning

Do not combine `mount_orchestration_source=true` with an `extra_mounts`
entry targeting `/workspace/bspp-orchestration` — the renderer will reject
the conflicting mount.

### Benchmark path

The benchmark submit path (`bsppctl validate-run` /
`folding_benchmark_submit.py`) sets `BSPP_INSTALL_MODE_SKIP=1` via `env` in
the srun command (not the heredoc) to preserve baked-wheel determinism. The
helper skips mount detection and uses baked mode regardless of whether
`orchestration_repo` is mounted.

### Deliberate difference from postprocessing

The postprocessing entrypoint silently skips a malformed orchestration mount.
The folding helper fail-closes. This is intentional because the folding mount
is opt-in: if the operator explicitly requested it, a malformed mount is a
configuration error.

### Hatchling baking

All four folding images bake `hatchling>=1.25` (via pixi for the runtime image,
via `pip install` in the same RUN layer for the other three) so that
`pip install -e --no-build-isolation` works in network-constrained
Slurm/Pyxis environments.

See the configurable install mode section for the full architecture decision.
