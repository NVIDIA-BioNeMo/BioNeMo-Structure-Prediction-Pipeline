# `containers/` — BSPP orchestration container images

Repository-owned images for running `bspp-orchestration-runtime` on GPU
clusters. The common scripts under `containers/scripts/` are the supported
build, publish, and Enroot/Pyxis import interface for every pipeline image.

Every pipeline image bakes the orchestration source (Contract + Control +
Runtime wheels at the pinned commit). A bind-mount of the orchestration source
remains available as an explicit dev override
(`BSPP_ORCHESTRATION_DEV_MOUNT=1`); the baked wheels are the default. The
preprocessing image additionally bakes its locked Scientific Kernel into a
dedicated reproducible image.

The postprocessing variants, ceiling survey, and SM acceptance matrix apply
only to the `runtime-postprocessing` image family. They do not define dependency
ceilings or acceptance gates for `runtime-preprocessing` or future
`runtime-folding-*` families.

## Tooling targets

| Target | Pipeline | Definition | Local image | Registry tag | Default SquashFS |
|---|---|---|---|---|---|
| `postprocessing` | postprocessing | `containers/Dockerfile` + `variants/postprocessing.env` | `bspp-orchestration:postprocessing` | `:postprocessing` | `bspp-orchestration-postprocessing.sqsh` |
| `preprocessing` | preprocessing | `containers/preprocessing/Dockerfile` + locked dedicated builder | `bspp-orchestration:preprocessing` | `:preprocessing` | `bspp-orchestration-preprocessing.sqsh` |

The registry repository for all targets is
`registry.example.com/bspp/bspp-orchestration`. Registry tags
are mutable transport names. Qualification and execution bind the imported
image by immutable OCI and SquashFS content identities.

## Postprocessing variant

| Variant | Tag | CUDA | Torch | Python | torch-cluster source | Platforms |
|---------|-----|------|-------|--------|----------------------|-----------|
| postprocessing | `:postprocessing` | 13.0 | 2.11.0+cu130 | 3.12 | PyG `pt211cu130` wheel | linux/amd64 |

The single `postprocessing` variant consolidates the former `current` and
`latest` variants: it starts from the newest PyG prebuilt `torch_cluster` wheel
for CPython 3.12 / linux x86_64, then pins Torch, CUDA, and the base image
around that ABI.

No amd64 production variant builds `torch-cluster` from git/source. Stage 13
performance depends on the compiled `torch_cluster.radius_graph` CUDA path.

Currency check (2026-09-02): the newest PyG wheel pages are torch-2.13.0, but
`torch_cluster` prebuilts stop at torch 2.11.0+cu130 (`torch_scatter` /
`torch_sparse` already have `pt212cu130` cp312 wheels; `torch_cluster` lags),
so `postprocessing` sits exactly at its ceiling. Re-run
`containers/scripts/survey-runtime-postprocessing-ceiling.sh` before assuming a
bump is available (needs outbound internet; run on the host workstation). When
PyG publishes a `torch_cluster`
cp312 linux-x86_64 wheel for torch >= 2.12, bump the variant pins together
with the matching `cuda-compat-13-x` package (packages through 13-3 exist and
are compatible with driver 535+ on Data Center GPUs per NVIDIA's Application
Compatibility Support Matrix) and re-run the full acceptance loop in
SM_ACCEPTANCE.md.

## Baked Toolkit

The image bakes the public upstream toolkit at an exact pinned commit under
`/opt/afdb-toolkit`. The clone, checkout, `.git` removal, and provenance write
happen in a single `RUN` instruction so no credential-bearing metadata
survives into later layers.

### Source Contract

| Property | Value |
|----------|-------|
| Repository | `https://github.com/PDBeurope/AFDB-Integration-Kit.git` |
| Branch | `nvidia-postproc` |
| Commit | `e2fa757aa0cb2cec8e4a8382627fcbbca7599556` |
| Toolkit path | `/opt/afdb-toolkit` |

### Provenance

`/opt/afdb-toolkit/provenance.json` records the immutable source identity:

```json
{
  "schema_version": 1,
  "repository": "https://github.com/PDBeurope/AFDB-Integration-Kit.git",
  "branch": "nvidia-postproc",
  "commit": "e2fa757aa0cb2cec8e4a8382627fcbbca7599556",
  "checked_out_at": "<ISO8601 UTC>",
  "toolkit_path": "/opt/afdb-toolkit"
}
```

### Build-time Qualification

After the pixi environment is installed, the Dockerfile compiles iPSAE from the
baked source (`/opt/afdb-toolkit/afdb_integration_kit/ipsae/`) with `make -B -C . CXX=g++`
and runs `qualify-baked-toolkit.sh`. The qualification script creates the
canonical BSPP-RQ-PAIR fixture (identical to the runtime qualification contract),
runs the baked iPSAE binary against it, and checks that `ipsae_AB` and `ipsae_BA`
both equal `1.000000`. Any deviation fails the image build.

### Dependency and rollout scope

The scientific environment includes `jsonschema[format-nongpl]`, matching the
upstream toolkit's format-validation requirements. Its dependencies are resolved
by the existing Pixi build; the source pin alone does not lock every dependency.

After a pin change, rebuild the postprocessing image and retain its image and
SquashFS identities, then create a fresh source bundle and Runtime Qualification.
Existing qualification evidence remains tied to the old source/image. An
explicit Cluster Profile `afdb_toolkit_repo` override takes precedence over the
baked source and must be migrated deliberately.

This pin includes upstream [PR #33](https://github.com/PDBeurope/AFDB-Integration-Kit/pull/33),
which preserves truthful BioIR attribution through the toolkit CLI, JSON metadata
and ModelCIF conversion for one explicitly declared method per invocation:

- `OpenFold2 (BioNeMo IR) / AlphaFold-Multimer`
- `OpenFold2 (BioNeMo IR) / OpenFold-pTM`

Set both worker `tool_used` and `homodimer_tool_used` to the same BioIR method;
the adapter forwards both declarations to the toolkit. Every original score
JSON must contain the matching producer-written `bioir_model_source`.
The toolkit rejects missing or conflicting provenance before execution, even
with dry-run or resume. Runtime version remains unknown (`?`) unless supplied
from retained prediction-environment evidence; the postprocessing environment
cannot establish it. Do not relabel predictions or patch scores to invent
provenance. See the [upstream input requirements](https://github.com/PDBeurope/AFDB-Integration-Kit/blob/e2fa757aa0cb2cec8e4a8382627fcbbca7599556/scripts/README.md#homogeneous-bioir-predictions).

Mixed pTM/multimer policies require separate homogeneous runs or a future
per-model provenance bridge. Repinning alone does not establish orchestration
end-to-end acceptance: a fresh image, Runtime Qualification and a real BioIR
analysis/archive acceptance run are still required. Supported image
architectures are unchanged.

### Build Authentication

Generic image builds clone the pinned toolkit from an explicit public Git URL
(`TOOLKIT_REPO`); no SSH agent is required for `build.sh` or `push.sh`. Keys are never copied into any image.

## Layout

```text
containers/
  Dockerfile
  preprocessing/
    Dockerfile
    build.sh
    smoke-local.sh
  pyprojects/
    latest.toml
  variants/
    postprocessing.env
  scripts/
    build.sh
    entrypoint.sh
    preprocessing_*.py
    pull-sqsh.sh
    push.sh
    qualify-baked-toolkit.sh
    regen-locks.sh
    resolve-repo-root.sh
    slurm-job-monitor.sh
    slurm-smoke-gpu.sh
    smoke-gpu.sh
    survey-runtime-postprocessing-ceiling.sh
```

The pixi manifests are intentionally separate from the workspace member
`pyproject.toml` files: the host `bspp-orchestration-control` environment can stay
lightweight while the container environment carries the production
scientific/GPU stack needed by `bspp-orchestration-runtime`.

The underscore-named preprocessing support tools are standard-library
diagnostic and evidence validators. They are mounted read-only and executed by
the pinned preprocessing container Python on a Slurm worker; the login-node
handoff does not depend on a cluster-installed Python.

The sourceable `slurm-job-monitor.sh` is the phase-agnostic Slurm job monitor
shared by handoffs. It polls the
live queue until the job leaves it, then requires a complete top-level terminal
accounting row with an exact exit code, returning 0 only for `COMPLETED`.
Queue/accounting lag is retried a bounded number of times; it never synthesizes
a terminal result.

Within `bspp_monitor_job`, a failed `squeue` query makes live presence
unknown; it is not evidence of queue absence. The monitor stops live polling
and defers to its bounded `sacct` loop. Only a complete, recognized top-level
terminal accounting row publishes the immutable `monitor.result`; accounting
query failure remains a hard resumable failure, while accounting lag or
nonterminal exhaustion returns failure without a record and can be resumed
safely.

`preprocessing_mmseqs_db_layout.py` is the confined MMseqs-layout boundary for
the diagnostic evidence. It follows the pinned MMseqs 8cc5ce3 shard-first
discovery rule as implemented by
[`FileUtil::findDatafiles` lines 330--345](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/FileUtil.cpp#L330-L345),
refuses ambiguous base/shard layouts, records resolved confined
links and component provenance, and only publishes immutable `0400` manifests,
result-target-key files, or regular copied components. Publication paths must
be outside the source root with no symlinked parent; a nonempty copy destination
is reused only with its matching immutable completion manifest. Its bounded
`directory-inventory` operation records `lstat` evidence for the whole
preserved search-output directory before any result layout is opened. It caps
entry count and name length, never follows links, and records each entry's
type, mode, size, mtime, device, inode, and link target.

The generic `result-target-keys` operation emits a schema-2 manifest that
records the source basename, so `res` and `res_exp` are decoded independently. The
first field of each NUL-framed result row is a target-member key, not the
result database's primary key. Duplicate target keys across query records are
valid and retained; extraction is streaming and capped at one million rows.
Valid metadata with absent data produces explicit `data_absent` evidence and
no key file.

`preprocessing_padded_db_preflight.py` publishes schema-2 evidence with a
top-level outcome and enumerated structural failures. It parses each nonblank
`uniref30_2302_db_pad.lookup` row as exactly three TAB-separated fields after
removing only the line ending: column 1 is the padded-primary key, column 2 is
a nonempty name that may contain spaces, and column 3 is the base-primary key.
The exact structural gates are lookup column 1 equals `pad.index` and lookup
column 3 equals `db.index`, with duplicate identifiers rejected independently.
This contract follows the pinned implementation that writes the padded index
and constructs lookup entries from the padded id and original database key in
[`makepaddedseqdb.cpp` lines 90--128](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/util/makepaddedseqdb.cpp#L90-L128),
using the exact `id<TAB>entryName<TAB>fileNumber` serialization in
[`DBReader.cpp` lines 741--748](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/DBReader.cpp#L741-L748).

All other cross-namespace relationships are observational. `_aln.index` is
measured against both padded and base primary keys; `res` target keys are
measured against padded, base, and alignment keys; and `res_exp` target keys
are measured against `_seq.index`. Missing totals, distinct counts, and bounded
witnesses confirm or refute captured expectations without becoming structural
gates. In particular, the stock `_aln` is not renumbered by
`makepaddedseqdb`, and identifier magnitude alone never establishes a
correspondence. Every primary namespace is parsed and allocated from its own
maximum under an absolute bitmap cap and a range/cardinality-density guard;
phase release and an aggregate live-bitmap budget bound combined memory.

## CUDA Compatibility

`postprocessing` uses CUDA 13.0 and installs the NVIDIA forward-compatibility package
`cuda-compat-13-0` (driver user-space libraries under
`/usr/local/cuda-13.0/compat`). This is the production mechanism on clusters
whose host driver is older than the toolkit, not a fallback: NVIDIA's CUDA
Application Compatibility Support Matrix (docs.nvidia.com/deploy/cuda-compatibility)
lists every `cuda-compat-13-x` package (13.0–13.3) as compatible with driver
branch 535+ on NVIDIA Data Center GPUs. The target cluster runs driver 535.104.12 (native
CUDA ≤ 12.2), and the CUDA 13.0 `postprocessing` image completed the reference postprocessing
pipeline there with byte parity (the pinned reference outputs, validated
2026-09-02).

The entrypoint toggle only controls whether `${BSPP_CUDA_COMPAT_DIR}` is
prepended to `LD_LIBRARY_PATH`:

```bash
BSPP_CUDA_COMPAT=off    # default: do not modify LD_LIBRARY_PATH
BSPP_CUDA_COMPAT=auto   # use compat libs if the configured dir exists
BSPP_CUDA_COMPAT=force  # fail if the configured compat dir is absent
```

On Pyxis/Enroot clusters the container runtime resolves the image-shipped
compat libraries independently of this toggle, which is why target-cluster acceptance
rows record `Compat Mode: off` while still using CUDA 13 user-space on a 535
driver. Use `force` when you want an explicit, fail-closed check in non-Slurm
contexts.

CUDA compatibility broadens portability but does not remove the need to test a
cluster/GPU/driver combination. Always run the GPU smoke test before pipeline
work on a new cluster or after changing the Torch/CUDA pins.

## Build

```bash
# Postprocessing local single-arch builds
bash containers/scripts/build.sh postprocessing
bash containers/scripts/build.sh postprocessing
bash containers/scripts/build.sh postprocessing

# Dedicated preprocessing build through the same interface
bash containers/scripts/build.sh preprocessing

# Build + convert any target's local Docker image to squashfs
bash containers/scripts/build.sh postprocessing --squashfs
bash containers/scripts/build.sh preprocessing --squashfs

# Build + push any target to the GitLab registry
bash containers/scripts/push.sh postprocessing
bash containers/scripts/push.sh postprocessing
bash containers/scripts/push.sh postprocessing
bash containers/scripts/push.sh preprocessing

# Rebuild + smoke + push every image of this repository on this host
bash containers/scripts/push.sh all
# (see --dry-run, --include-generic)
```

The build currently uses the variant pixi manifest directly, with dependency
pins recorded in `containers/pyprojects/`. `containers/scripts/regen-locks.sh`
is retained as a helper for generating pixi lockfiles once pixi/network access
is available in the development environment.

The literal `preprocessing` target delegates to
`containers/preprocessing/build.sh`; that backend retains its clean-checkout,
artifact verification, baked-wheel, embedded-provenance, and linux/amd64
contracts. It is an implementation backend, not a second operator workflow.
Future pipeline images must extend the common `build.sh`, `push.sh`, and
`pull-sqsh.sh` target interface instead of adding issue-specific build,
registry, Enroot, or SquashFS drivers. `push.sh all` is the full-matrix canonical command: preflight checks, then
preprocessing + every `containers/folding/*/` image (dynamically discovered),
each freshly built, pushed, and smoke-verified, with evidence capture and a
fail-fast summary. The single generic `.env` variant (`postprocessing`) joins
with `--include-generic` (its toolkit source defaults to the public upstream).

`push.sh preprocessing` performs a fresh dedicated build before publishing.
Therefore a smoke performed before `push.sh` is not evidence for the image
that was pushed. For qualification work, push first, verify the local tag has
the `image_id` in `containers/preprocessing/dist/build-record.json`, and then
run the local smoke against `bspp-orchestration:preprocessing`. The push adds
the exact canonical registry `oci_digest` to that same build record without
using the local image ID as a substitute.

### Image Layout

```text
/opt/
  afdb-toolkit/           # baked public toolkit (exact pinned commit)
    provenance.json        # immutable source identity
    qualification-result.json  # build-time qualification pass
    ipsae_cpp              # compiled iPSAE binary
    afdb_integration_kit/  # full source working tree (without .git)
  bspp-orchestration-env/ # pixi environment
    .pixi/
    pixi.toml
  pixi-activate.sh
  bspp/
    lib/                   # orchestration-contract source
    execution_bootstrap.py
/usr/local/bin/
  entrypoint.sh
  bspp-container-smoke-gpu
  qualify-baked-toolkit.sh
```

## GPU Smoke Test

Run inside the container on a GPU node before submitting archive work:

```bash
bspp-container-smoke-gpu
```

The smoke test imports Torch and `torch_cluster`, executes `radius_graph` on
CUDA tensors, checks that the visible GPU SM is present in
`torch.cuda.get_arch_list()`, and checks that AFDB is not using the slow
fallback path when the mounted toolkit is present. It also encodes a small
Zstd RAW payload through `nvidia.nvcomp` and decodes it with the system `zstd`
CLI; `tar_compression: zstd-members` relies on that same container capability.

To record an acceptance artifact:

```bash
bspp-container-smoke-gpu /tmp/bspp-smoke-postprocessing.json
```

Use [SM_ACCEPTANCE.md](SM_ACCEPTANCE.md) to track which image has passed on
which cluster/GPU architecture. The `postprocessing` image is accepted per SM
after smoke, Stage 13 timing, and result-parity checks.

On SLURM:

```bash
IMAGE_TAG=postprocessing sbatch --partition=<partition> containers/scripts/slurm-smoke-gpu.sh
IMAGE_TAG=postprocessing sbatch --partition=<partition> containers/scripts/slurm-smoke-gpu.sh
```

The SLURM wrappers call `/usr/local/bin/entrypoint.sh` explicitly. Do not rely
on Pyxis/Enroot implicitly honoring Docker `ENTRYPOINT`; site defaults vary.
The wrappers also resolve the repository through `BSPP_ORCH` or
`SLURM_SUBMIT_DIR` because `sbatch` runs a spool copy where `$0` and
`BASH_SOURCE[0]` can point under Slurm's runtime directory instead of the repo.

The entrypoint exposes `AFDB-Integration-Kit` through `PYTHONPATH` instead of
installing it editable by default. Cluster jobs mount that repository read-only,
and editable installs update `*.egg-info` in the source tree. Set
`BSPP_INSTALL_TOOLKIT_EDITABLE=on` only for writable development mounts.

The orchestration source is baked into the image and is the default
(`BSPP_ORCHESTRATION_SOURCE=baked`). To iterate on a live checkout, mount it
at `/workspace/bspp-orchestration` and set `BSPP_ORCHESTRATION_DEV_MOUNT=1`;
the entrypoint then strict-editable-installs Contract, Control, and Runtime
from the mount. The mounted commit must equal the baked commit unless
`BSPP_ORCHESTRATION_DEV_MOUNT_ALLOW_MISMATCH=1` is set (fail-closed). A mount
present without the dev flag fails closed rather than silently overriding the
baked source.

## Run Dev Container

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)/../bspp/AFDB-Integration-Kit:/workspace/AFDB-Integration-Kit" \
  -v "$(pwd):/workspace/bspp-orchestration" \
  -e BSPP_ORCHESTRATION_DEV_MOUNT=1 \
  -w /workspace/bspp-orchestration \
  bspp-orchestration:postprocessing \
  bash
```

## Enroot / Pyxis Import

Use the tracked CPU Slurm importer for any published target:

```bash
sbatch containers/scripts/pull-sqsh.sh postprocessing
sbatch containers/scripts/pull-sqsh.sh postprocessing
sbatch containers/scripts/pull-sqsh.sh preprocessing
```

The importer reports the completed SquashFS path, size, and SHA-256. It uses
the no-port GitLab registry route required on NVIDIA clusters:

```bash
enroot import docker://registry.example.com/bspp/bspp-orchestration:postprocessing
```

Store private registry credentials outside the repo:

```text
~/.config/enroot/.credentials
machine registry.example.com login <registry-user> password <PAT-or-deploy-token>
```

The token needs `read_registry` scope.

```bash
chmod 700 ~/.config/enroot
chmod 600 ~/.config/enroot/.credentials
```

## Registry

```text
registry.example.com/bspp/bspp-orchestration:{postprocessing,preprocessing,folding-runtime,folding-colabfold,folding-openfold-cli,folding-bioir}
```
