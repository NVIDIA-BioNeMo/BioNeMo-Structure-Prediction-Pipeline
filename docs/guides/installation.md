# Installation

Install the Control package where you will operate the pipeline. Scientific
execution belongs in the phase-specific cluster images; installing Control does
not require CUDA, PyTorch, model checkpoints, or the Runtime dependency set.

## Choose an environment

The repository is a uv workspace. All three distributions require Python 3.12
(`>=3.12,<3.13`). Run the following commands from the repository root, with
Git, uv, and a supported Python available:

```bash
uv sync --frozen --package bspp-orchestration-control --no-dev
uv run --frozen --package bspp-orchestration-control --no-dev bsppctl --help
```

The installed executable is also available as `.venv/bin/bsppctl`. Using it
directly does not trigger another uv synchronization.

| Environment | What it installs | Intended use |
| --- | --- | --- |
| Control | Contract, Click, Pydantic, PyYAML, Tabulate | Profiles, Plans, lifecycle commands, evidence handling |
| Runtime | Contract plus data and execution dependencies, including PyArrow, DuckDB, NumPy, and Submitit | Runtime executors and data processing |
| Phase image | Baked orchestration wheels plus the selected tools and scientific backend | Execution through Slurm/Pyxis |
| Development workspace | All workspace packages and development tools | Source changes and repository checks |

For the development workspace, use:

```bash
uv sync --frozen --all-packages --group dev
```

Runtime's Python dependencies alone do not install every folding backend or its
weights. Conversely, the small folding Runtime image installs only the executor
dependencies it needs; it is not a general-purpose Runtime development image.
See [Containers](containers.md) for the supported image families.

## Preserve the dependency lock

The commands above consume the shipped `uv.lock` without changing it.
`--frozen` skips the check that the lock agrees with current project metadata;
it does **not** resolve updated dependencies. `--locked` instead checks that no
lock update is needed and refuses when the lock is stale. An ordinary `uv sync`
may update the lock.

Use `uv lock --check`, or replace `--frozen` with `--locked`, when checking
freshness for the selected checkout. A freshness refusal is distinct from a
package download or Python-version failure: compare the project metadata with
the intended lock before proceeding. Frozen mode is useful for replaying a
reviewed lock, not for proving that it matches changed dependencies. For an
intentional dependency change, regenerate and review the lock as a source
change, then build from the resulting clean commit.

## Reconstruct the public benchmark

`prepare-benchmark` belongs to Control, but writing its Parquet files needs
PyArrow. Add that dependency to the command environment without installing the
whole Runtime package:

```bash
uv run --frozen --package bspp-orchestration-control --no-dev --with pyarrow \
  bsppctl prepare-benchmark \
  --spec docs/benchmarks/pdb-temporal-2022-2025-v1/benchmark-spec.json \
  --reconstruct docs/benchmarks/pdb-temporal-2022-2025-v1/reconstruction-targets.jsonl \
  --output /path/to/new-benchmark-corpus \
  --workers 4
```

Replace the output path with a new directory outside the checkout. Reconstruction
downloads public RCSB records and produces the corpus and reference files; it
does not generate MSAs or require S3 credentials. Follow the
[public corpus instructions](../benchmarks/pdb-temporal-2022-2025-v1/README.md)
to verify the fingerprint and checksums. Preprocessing currently needs a `.fa`
filename; make the documented byte-identical copy of `targets.fasta` outside the
checked corpus directory.

Reference validation is a separate operation. The shipped `validate-run`
interface consumes an S3 corpus prefix; a locally reconstructed corpus does not
enable a local-directory validation flag. See
[Troubleshooting](troubleshooting.md#reference-validation-and-local-corpora).

## Prepare the cluster side

Control supports either SSH from an operator machine or `local-slurm` from a
host with the site's Slurm commands. For native Control, stage the same clean
source revision and install the Control-only environment there. Running Control
on a login host submits scheduled work; it is not permission to run scientific
kernels on that host.

Keep profiles, authority directories, inputs, outputs, and credentials outside
the clean source checkout. Record the source commit and retain build records
when deploying images. Continue with
[Cluster configuration](cluster-configuration.md) and
[Containers](containers.md); the illustrative YAML templates need real site
values before they can execute.
