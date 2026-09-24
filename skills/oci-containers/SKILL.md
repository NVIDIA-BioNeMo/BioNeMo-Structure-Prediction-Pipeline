---
name: oci-containers
description: Build, publish, import, and qualify the BSPP orchestration OCI container images.
---

# OCI containers

## When to use this skill

Use this skill to prepare and build the BSPP orchestration container images, publish them to a container registry, import them to SquashFS for
Enroot/Pyxis execution on a Slurm cluster, and run the qualification gates that
prove the baked toolchain works. All commands below are ordinary shell commands
with repository-relative script paths, run from the repository root.

The canonical scripts live under `containers/scripts/` and the dedicated
phase-specific image definitions live under `containers/preprocessing/` and
`containers/folding/`. This document covers the externally configurable
interface of `containers/scripts/push.sh` and `containers/scripts/pull-sqsh.sh`
(registry, repository, and import settings supplied through environment
variables), so the same instructions work on any workstation and any cluster.

## Container variants

There are three build targets: `postprocessing`, `preprocessing`, and `folding`
(four images), each with its own dedicated image definition. Every image bakes
the orchestration source (Contract + Control + Runtime wheels at the pinned
commit); a bind-mount of the orchestration source remains an explicit dev
override (`BSPP_ORCHESTRATION_DEV_MOUNT=1`).

### The `postprocessing` image

The single `containers/variants/postprocessing.env` declares the build inputs
consumed by `containers/scripts/build.sh` and `containers/scripts/push.sh`:

- `VARIANT_TAG` — the local image tag (`postprocessing`).
- `BASE_IMAGE` — the CUDA base image.
- `CUDA_VERSION` — the CUDA major/minor version.
- `PIXI_MANIFEST` — a `containers/pyprojects/*.toml` pixi manifest.
- `TORCH_VERSION` — the pinned PyTorch build.
- `TORCH_CLUSTER_VERSION` — the pinned `torch_cluster` build.
- `PLATFORMS` — the declared build platforms (e.g. `linux/amd64`).

The image is built from the shared `containers/Dockerfile` by the dedicated
`containers/postprocessing/build.sh` backend, which sources
`containers/variants/postprocessing.env` for the variant pins and bakes the
Contract + Control + Runtime wheels (with sha256s recorded in
`dist/build-record.json`). The former `current`, `latest`, and `baked-toolkit`
variants are consolidated into this single image.

The shared Dockerfile also takes the toolkit pin as build args:

- `TOOLKIT_REPO` — the toolkit Git URL. Defaults to the public upstream toolkit
  (`https://github.com/PDBeurope/AFDB-Integration-Kit.git`); the variant file
  deliberately does not carry it.
- `TOOLKIT_BRANCH` / `TOOLKIT_REF` — optional overrides. Unset selects the
  public `nvidia-postproc` branch at the exact commit pinned in the Dockerfile; exporting
  them makes `build.sh` forward them as build args, so an operator can build
  against a different public toolkit mirror, branch, or ref without editing
  files. The public build always uses the public toolkit source.

The image records the actually-built ref as `BSPP_EXPECTED_TOOLKIT_COMMIT`,
and the entrypoint refuses to start a container whose baked toolkit does not
match it — so a build with an overridden `TOOLKIT_REF` cannot silently produce
an image pinned to the default.

The pinned upstream toolkit preserves BioIR attribution through its CLI, JSON
metadata and ModelCIF conversion for one explicit method per invocation. Both
worker `tool_used` and `homodimer_tool_used` must equal either
`OpenFold2 (BioNeMo IR) / AlphaFold-Multimer` or
`OpenFold2 (BioNeMo IR) / OpenFold-pTM`, and every original score must carry the
matching producer-written `bioir_model_source`. A runtime version remains `?`
unless supported by retained prediction-environment evidence. Do not relabel
predictions or synthesize provenance. Mixed pTM/multimer policies need separate
homogeneous runs or a future per-model provenance bridge. See
[the current source and input requirements](../../containers/README.md#dependency-and-rollout-scope).
A source pin update requires a fresh image and Runtime Qualification, followed
by real BioIR analysis/archive acceptance before claiming orchestration
end-to-end support. An explicit `afdb_toolkit_repo` profile override continues
to take precedence.

### Dedicated builders (`preprocessing`, `postprocessing`, `folding`)

`preprocessing` and `postprocessing` each live under their own directory.
`folding` has **four images**:
`runtime/` (non-fold actions), `colabfold/`, `openfold-cli/`, and `bioir/` (the
`fold` action). Each has the same layout:

- `build.sh` — the dedicated build backend.
- `Dockerfile` — the image-specific image definition.
- `image-lock.json` — the pinned-artifact authority (see below).
- `entrypoint.sh` — the CUDA compat entrypoint.
- `image-smoke.py` — the in-image composition smoke.
- `smoke-local.sh` — a local composition smoke for the freshly built image.

The runtime image also has `pixi.toml` (the trimmed environment, resolved
non-locked in-image). All base images are public (Docker Hub, GHCR, PyPI,
GitHub only — no NGC, no entitlement, anonymous pull).

The generic `containers/scripts/build.sh` and `containers/scripts/push.sh`
delegate these targets to the dedicated backend builders. The dedicated
backends are implementation backends, not separate operator workflows: operators
always go through the canonical `containers/scripts/build.sh` and
`containers/scripts/push.sh` entry points.

### Target summary

| Target | Definition source | Local image name |
|---|---|---|
| `postprocessing` | `containers/postprocessing/` (+ `variants/postprocessing.env`) | `bspp-orchestration:postprocessing` |
| `preprocessing` | `containers/preprocessing/` | `bspp-orchestration:preprocessing` |
| `folding runtime` | `containers/folding/runtime/` | `bspp-orchestration:folding-runtime` |
| `folding colabfold` | `containers/folding/colabfold/` | `bspp-orchestration:folding-colabfold` |
| `folding openfold-cli` | `containers/folding/openfold-cli/` | `bspp-orchestration:folding-openfold-cli` |
| `folding bioir` | `containers/folding/bioir/` | `bspp-orchestration:folding-bioir` |

## Pinned-artifact locks (`image-lock.json`)

Each dedicated builder has a `containers/<phase>/image-lock.json` that pins every
non-PyPI/conda artifact baked into the image. The general rule is: every such
artifact (pixi, mmseqs, rsync) is pinned in `image-lock.json` and downloaded +
SHA-256 verified inside the Dockerfile, never on the host.

### Folding per-image locks and fail-closed sentinels

Each folding image has its own `containers/folding/<image>/image-lock.json`:

- `base_image` — a digest-selected public base image (Docker Hub or GHCR).
- Runtime image also pins `pixi` and `rsync`.
- Kernel images pin `runtime_deps` (documentation-only) and backend-specific
  fields (`colabfold_version`, `openfold_source_commit`, `bioir_version`,
  `torch`).

Each `build.sh` verifies the pinned `base_image.linux_amd64_digest` and
fails closed (exit 2) if it is unset or equals the 64-zero placeholder, so no
kernel image can be built from an unpinned base. All image locks currently
carry real, resolvable base digests; there is no placeholder left to replace.

**Operator action:** none for the base pins — they are verified at build time.

### Preprocessing lock

`containers/preprocessing/image-lock.json` pins `pixi`, `mmseqs`, `rsync`, and
`colabfold`. These entries are already digest-pinned (no placeholder sentinel),
so the preprocessing target is buildable today.

Its canonical image smoke also runs a tiny, CPU-only synthetic paired-row
regression through the pinned MMseqs `result2msa` and ColabFold assembler.
It checks row identity and sequence contents under scientific schema v3
without a search, external database, GPU, or network. This proves tool
composition; it does not establish scientific accuracy or benchmark success.

## Build

To rebuild, smoke-verify, and push **every** image of this repository on a
Docker-capable host in one job (preflight checks, dynamic image matrix,
evidence capture, fail-fast summary):

```bash
./containers/scripts/push.sh all            # preprocessing + postprocessing + every folding image
./containers/scripts/push.sh all --dry-run  # print the plan only
```

`push.sh all` covers the dedicated images (`preprocessing`, `postprocessing`,
and every `containers/folding/*/` image). The command requires `BSPP_REGISTRY`
to be set and the host to be logged in to the registry.

Build a single variant locally with `containers/scripts/build.sh`:

```bash
./containers/scripts/build.sh postprocessing
./containers/scripts/build.sh preprocessing
./containers/scripts/build.sh folding runtime
./containers/scripts/build.sh folding colabfold
./containers/scripts/build.sh folding openfold-cli
./containers/scripts/build.sh folding bioir
./containers/scripts/build.sh folding all
./containers/scripts/build.sh postprocessing --squashfs
./containers/scripts/build.sh postprocessing --squashfs /path/to/output.sqsh
./containers/scripts/build.sh postprocessing --no-cache
```

The `--squashfs` flag converts the freshly built local Docker image into a
SquashFS file via `enroot import` (default output
`bspp-orchestration-<tag>.sqsh`). Any unrecognized arguments after the variant
name are passed through to `docker build` (for example `--no-cache`).

Prerequisites:

- Docker.
- For `--squashfs`: Enroot installed on the workstation — the conversion runs
  `enroot import dockerd://…`, reading the freshly built image from the local
  Docker daemon.
- For the dedicated `preprocessing`/`postprocessing`/`folding` targets: a clean
  committed checkout, `uv`, and network access. No host-side `pixi` or `curl`
  is required — the pinned pixi/mmseqs binaries are downloaded and verified
  inside the Dockerfile. The backend refuses a dirty tree. The `postprocessing`
  toolkit is cloned from the public upstream over HTTPS (no SSH agent
  required).

## Publish

Publish a variant with `containers/scripts/push.sh`:

```bash
./containers/scripts/push.sh <postprocessing|preprocessing|folding <image>|all>
```

For `postprocessing`, `--include-postprocessing-internal` additionally builds the
postprocessing-internal image after the public image is pushed:

```bash
./containers/scripts/push.sh postprocessing --include-postprocessing-internal
```

The internal image is built and tagged locally via
`containers/nvidia/build-internal.sh` — build + tag only, **never pushed by design**.
Passing `--include-postprocessing-internal` with any other single variant
(`preprocessing`, `folding`) is a fail-fast error. In `all` mode the flag is
part of the matrix and has the same build-only-never-push semantics.

The publish interface is externally configurable:

- `BSPP_REGISTRY` — **required**. The script fails closed with a clear error if
  it is unset; there is no hard-coded internal registry.
- `BSPP_IMAGE_REPOSITORY` — generic default `bspp-orchestration`.

The pushed reference is `${BSPP_REGISTRY}/${BSPP_IMAGE_REPOSITORY}:<tag>`. For
example:

```bash
BSPP_REGISTRY=registry.example.com ./containers/scripts/push.sh postprocessing
```

`push.sh` performs a fresh build and then a push (single-arch), logging in to
the configured registry when it is not already authenticated. For the
`preprocessing`/`postprocessing`/`folding` targets it strictly tags the freshly
built image ID and atomically supplements
`containers/<phase>/dist/build-record.json` with `registry_image` and the
canonical `oci_digest`; it never treats a local image ID as a registry digest.
`all` runs the full matrix (see Build above).

## Cluster import (Enroot to SquashFS)

`containers/scripts/pull-sqsh.sh` is a Slurm job that runs `enroot import` of the
registry image into a `.sqsh` file for Pyxis/Enroot execution:

```bash
sbatch containers/scripts/pull-sqsh.sh
sbatch containers/scripts/pull-sqsh.sh postprocessing
```

The default tag is `postprocessing`, or a tag supplied from `containers/.env`.

The import interface is externally configurable:

- `BSPP_REGISTRY` — **required**. The script fails closed with a clear error if
  it is unset; there is no hard-coded internal registry.
- `BSPP_IMAGE_REPOSITORY` — generic default `bspp-orchestration`. Together
  these select the source image `<registry>/<repository>:<tag>`.
- `BASE_DIR` — output base directory; default `${SLURM_SUBMIT_DIR:-$PWD}`.
- `CONTAINER_DIR` — containers directory; default `${BASE_DIR}/containers`.
- `CONTAINER_IMAGE` — full output `.sqsh` path override; the derived default is
  `${CONTAINER_DIR}/bspp-orchestration-<tag>.sqsh`. None of these defaults is a
  site filesystem path.
- The script no longer hard-codes `#SBATCH --partition` or `#SBATCH --account`.
  Operators pass these to `sbatch` themselves, and must supply `BSPP_REGISTRY`:

```bash
BSPP_REGISTRY=registry.example.com sbatch \
  --partition=<cluster-partition> --account=<cluster-account> \
  containers/scripts/pull-sqsh.sh postprocessing
```

### Enroot registry credentials

For a private registry, Enroot reads a netrc-style credentials file at
`~/.config/enroot/.credentials`:

```text
machine <registry-host> login <user> password <token>
```

The token needs `read_registry` scope. Set permissions to `700` on the
directory and `600` on the file. Keep the registry host generic; never commit
credentials to the repository.

## Qualification

- `containers/scripts/smoke-gpu.sh` — the in-container GPU smoke (exposed inside
  the image as `bspp-container-smoke-gpu`). It verifies the Torch +
  `torch_cluster` CUDA path by running `radius_graph` on CUDA tensors, checks
  that the visible GPU SM is present in `torch.cuda.get_arch_list()`, confirms
  the AFDB fallback path is not active, and round-trips an nvCOMP Zstd RAW
  payload through `zstd`. It writes an optional JSON evidence record.
- `containers/scripts/slurm-smoke-gpu.sh` — the Slurm wrapper that runs the smoke
  inside a GPU allocation via `srun --container-image=<sqsh>`. Set `IMAGE_TAG`
  and submit it with `sbatch`. The wrapper resolves the repository root through
  `BSPP_ORCH`/`ORCH_DIR`/`SLURM_SUBMIT_DIR` because `sbatch` spools a copy of
  the script.
- `containers/scripts/qualify-baked-toolkit.sh` — the build-time baked-toolkit
  qualification invoked inside the Dockerfile (the name is retained; it is the
  scientific iPSAE gate, not the removed variant). It rebuilds iPSAE from baked
  source, runs a deterministic paired-PDB/PAE fixture identical to the runtime
  qualification contract, and requires `ipsae_AB` and `ipsae_BA` to equal
  `1.000000`. Any deviation exits non-zero and aborts the image build. Use the
  `BSPP_BAKED_TOOLKIT` override to point the script at a simulated toolkit root
  for sandbox testing.

## Registry credentials and cluster access

### Registry credentials

Obtain a deploy token or personal access token from the registry's web UI. For
`push.sh` (via `docker login <registry>`) the token needs registry write/push
scope; for `pull-sqsh.sh` (via the Enroot `~/.config/enroot/.credentials` netrc
file) `read_registry` scope is sufficient. Never store credentials inside the
repository or in `containers/.env` (which is gitignored). Use
normal HTTPS access for the public toolkit clone; no toolkit SSH key is required.

### Cluster access

Request an account and a suitable partition/queue from the cluster
administrator. Confirm that Enroot/Pyxis and the required GPU driver and
architecture are available. Pass `--partition`/`--account` to `sbatch` for the
import and smoke jobs. All names in this document are generic — substitute your
own registry host, cluster account, and partition.
