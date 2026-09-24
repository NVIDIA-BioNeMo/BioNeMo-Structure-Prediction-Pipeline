# Folding Runtime Image

This directory defines the folding **runtime image** — the image used for
non-fold actions (`msa-flatten`, `split`, `preprocess`, `canonical-pair`). It
is `paths.image` in Cluster Profile examples. It is a
**folding-executor-only image**, not a general `bspp-orchestration-runtime`
image. Its scalar entrypoints require yaml, click, pydantic, and NumPy2. The
folding package anchor eagerly imports archive planning, whose deterministic
shuffle uses NumPy. The remaining general Runtime dependencies (duckdb,
google-cloud-storage, orjson, pyarrow, rich, submitit, zstandard) are not
installed. Scientific backends remain lazy and are not invoked in this image.

The image carries:

- Contract+Control+Runtime wheels via `pip install --no-deps`
- `/usr/bin/lz4`, `/usr/bin/rsync`, `/usr/bin/tar`, `/usr/bin/flock` (from pixi)
- **s5cmd** (for remote-bundle MSA inputs via the publish-to-swiftstack
  transport)
- A trimmed `pixi.toml` (no `[pypi-dependencies]` colabfold line)
- The CUDA 12.6.3 compatibility entrypoint

## pixi (in-image, non-locked)

The pinned pixi binary (v0.80.0) is downloaded and sha256-verified inside the
Dockerfile (via `--build-arg` from `image-lock.json`), and `pixi install`
resolves the conda environment non-locked from the trimmed `pixi.toml`. No
host-side pixi and no committed `pixi.lock` are required.

## Build, smoke, push, import

```bash
containers/scripts/build.sh folding runtime
containers/folding/runtime/smoke-local.sh bspp-orchestration:folding-runtime
containers/scripts/push.sh folding runtime
sbatch containers/scripts/pull-sqsh.sh folding-runtime
```

The build installs pixi inside the image, builds and hashes the Contract,
Control, and Runtime wheels, generates the in-image manifest, and writes
`dist/build-record.json`. The smoke runs with `--network none`, no source
mount, and the real image toolchain. It checks the pinned compatibility
symlink, effective loader ordering, the locked executables, the importability
of the Contract/Control/Runtime distributions, the presence of the Control
distribution, and the embedded image manifest. It proves image composition,
not live cluster or scientific-data acceptance.

The smoke also imports the actual folding executor, legacy MSA importer and
canonical index, runs executor help, and requires the real legacy import command
to reject a temporary empty handoff before publishing anything. This reaches
lazy command imports that root CLI help does not exercise. No target data,
model, network access or GPU is used by these entrypoint checks.
