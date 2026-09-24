# Folding Benchmark Example

This directory is a schema-complete operator example for the folding Phase
Lifecycle. It demonstrates a complete folding Phase Plan and its
matching Cluster Profile for the three-phase chain
(preprocessing -> folding -> postprocessing).

The [public 1,000-target benchmark inputs](../../benchmarks/pdb-temporal-2022-2025-v1/README.md)
provide actual PDB IDs, assembly URLs and pinned sequences for reconstruction
with `bsppctl prepare-benchmark --reconstruct`. That input corpus is separate
from this directory's illustrative Phase Plans and image/location placeholders.
The input guide includes the byte-preserving `.fasta` to `.fa` copy needed by
the preprocessing parser.

| File | Purpose |
|---|---|
| `profiles.yaml` | User-local Cluster Profile example for a folding cluster. |
| `run-plan.yaml` | Phase-aware Run Plan example: a `FoldingPhasePlan` with `payload.transport: publish-to-s3` and a remote MSA-set input. |
| `folding-phase-plan.yaml` | Folding Phase Plan example: a `FoldingPhasePlan` with `payload.transport: local` and a local MSA-set input. |

Both `run-plan.yaml` and `folding-phase-plan.yaml` are `FoldingPhasePlan`
documents. Folding has no separate legacy Run Plan, so the operator-authored
"Run Plan" and the "Phase Plan" are one document. Both files carry executable
root-manifest authority (`payload.msa_set_manifest` binds the same artifact-set
identity as `payload.msa_set` and `input_location`); they differ only in the
per-seam transport policy and the matching input-location kind. Together they
demonstrate both transport branches:

- `run-plan.yaml` selects `publish-to-s3` and consumes a
  `VerifiedRemoteBundledArtifactLocation` (an `s3://` `bundle_uri`), the default
  maximal cross-cluster topology.
- `folding-phase-plan.yaml` selects `local` and consumes a
  `VerifiedLocalBundledArtifactLocation` (absolute cluster paths), the
  single-cluster shortcut.

## Deliberately non-runnable

This directory is intentionally non-runnable for two independent reasons:

1. **Placeholder identities and locations.** The checked-in identities,
   digests, sizes, and timestamps are illustrative placeholders, not production
   claims, exactly like the `runtime-qualification.json` shape in the
   postprocessing example. Both Phase Plan files parse successfully through
   the strict `FoldingPhasePlan` loader (`yaml.safe_load` +
   `phase_plan_family_from_mapping`), including their canonical
   `artifact_location_id` values.
2. **Placeholder image pins.** Executable composition is present: each rendered
   action invokes the job-local Runtime executor for `openfold-cli`, `colabfold`
   or `bioir`. `openfold-trt` fails closed with its deferred-model-factory error.
   The example image/location records are still illustrative; replace them with
   actual qualified image and input identities before execution. Build pins and
   schema checks do not prove a successful live benchmark. Live acceptance
   has not yet run on any real allocation; the benchmark flow documented here
   leads from qualification toward that operator milestone.

See [implementation status](../../status.md) for the current release boundary
and the [BioIR walkthrough](../../benchmarks/bioir-workflow.md) for the complete
operator sequence.

A remote placeholder record may pass materialization because materialization
trusts the previously verified remote-location authority and does not fetch its
bytes there. That does not make the example executable or prove the remote
object exists. Selecting `publish-to-s3` records the per-seam transport
policy only; it does not itself move bytes.

Replace the following placeholder values with independently established records
before materializing a new run:

- `payload.msa_set.artifact_set_id` (`sha256:` + 64 hex characters),
- `payload.msa_set_manifest.artifact_set_id` and its `chunks[].sha256` (the
  checked-in values are canonical for the placeholder chunk reference; recompute
  `payload.msa_set_manifest.artifact_set_id`, `payload.msa_set.artifact_set_id`,
  `input_location.artifact_set_id`, and `input_location.artifact_location_id`
  together whenever any manifest or location field changes),
- every `sha256` digest in `input_location` and its `members`,
- all `*_size_bytes` values,
- `input_location.artifact_location_id` (the checked-in value is canonical for
  the placeholder fields; recompute it from the physical representation
  whenever any placeholder digest, size, path, or member changes),
- `input_location.verified_at` (the real verification timestamp),
- `input_location.bundle_uri` / `tar_path` / `bundle_path` (the real location),
- `input_location.raw_tar_members` and `members` (the real tar inventory).

The `payload.transport`, `payload.backend`, and `target_cluster` values are the
real example choices and are not placeholders. The Cluster Profile paths and
object-store prefixes are illustrative example values; update them for the
target cluster before submission.

After replacing the illustrative records, follow the preferred Phase Lifecycle:
`phase materialize` -> `submit` -> `status` ->
`resume` -> (`cancel` / `retry` as needed) -> `finalize`. Give the run a
dedicated authority root outside the clean source checkout.

## Backend-selected image

`profiles.yaml` `paths.image` points at the folding **runtime image**
SquashFS (for non-fold actions). The folding release preset selects the backend-specific kernel
image at the Slurm job/step boundary; the `folding_backend_images` field wires
the three kernel images (`openfold-cli`, `colabfold`, `bioir`) built with
`containers/scripts/build.sh folding <image>`. All base images are public
(Docker Hub, GHCR, PyPI, GitHub only — no NGC, no entitlement, anonymous pull).

Per-backend image overrides are wired via the `folding_backend_images` field in
the Cluster Profile. Each entry is a `{backend, image}` pair; backends without
an explicit override fall back to `paths.image`.

The example also records `folding_backend_assets` for the three executable
backends (chain-manifest CSV, OpenFold model dir, ColabFold weights dir, BioIR
`.pt` checkpoint) with one exact read-only `extra_mounts` target per asset path
(rendered as Pyxis `:ro`). `openfold-trt` deliberately has no
asset record and no speculative factory field: selecting it fails closed with
the stable deferred-`model_fn` error. The per-backend kernel image locks in
`containers/folding/<image>/image-lock.json` carry pinned, build-verified base
digests.

## BioIR Release Preset

The `folding_release_preset` field selects the BioIR package source.
The folding release preset resolves to the public PyPI `bionemo-ir==0.1.0`
package from the public Docker Hub base
`nvidia/cuda:13.0.3-devel-ubuntu24.04`. The preset name is retained for
qualification identity compatibility. The preset flows through the folding
cluster snapshot into the qualification tuple identity, so different presets are
never conflated.
