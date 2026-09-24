# Configuration reference

A Phase Plan states the work and scientific choices. A Cluster Profile supplies
site resources and paths. Materialization combines them with verified inputs
into immutable attempt authority; editing an authored file afterward does not
rewrite that authority.

## Documents and ownership

| Document | Who creates it | Role |
| --- | --- | --- |
| Phase Plan | Operator | One preprocessing, folding or postprocessing intent; names `target_cluster`. |
| Cluster Profile | Operator/site owner | Transport, accounts, paths, images, resources, mounts and qualification configuration. |
| Legacy Run Plan | Operator, for postprocessing | Pinned input to a postprocessing Phase Plan; not a separate folding/preprocessing plan. |
| Phase RunSpec | Control during materialization | Immutable resolved input, cluster and action snapshot for one Attempt. |
| Runtime handoff | Runtime | Verified output identity and evidence consumed by finalization or a downstream phase. |
| Phase Receipt | Control during finalization | Attempt-bound accepted result that seals the Phase Run. |

Write plans and profiles; obtain measured input hashes, qualification records and
handoffs from their actual producers. Do not author a successful receipt or edit
generated authority to make an incompatible input appear valid.

## Start from the public examples

The commented examples document the detailed schema and its constraints:

- [Shared Cluster Profile](../../skills/examples/run-plans/cluster-profile.yaml)
- [Preprocessing Phase Plan](../../skills/examples/run-plans/preprocessing-phase-plan.yaml)
- [Folding Phase Plan](../../skills/examples/run-plans/folding-phase-plan.yaml)
- [Postprocessing Phase Plan](../../skills/examples/run-plans/postprocessing-phase-plan.yaml)
- [Postprocessing authority-document examples](../../skills/examples/run-plans/postprocessing/)

They are schema-loadable illustrations, **not runnable configurations**. Replace
placeholder identities, paths, sizes, hashes and qualification records with
values from your environment. The [configuring-runs skill](../../skills/configuring-runs/SKILL.md)
explains how to obtain them.

## Selecting and resolving a profile

The config file has a top-level `clusters:` mapping. A Plan's `target_cluster`
must name one of its entries. Select the file using
`bsppctl --config profiles.yaml …`, `BSPPCTL_CONFIG`, or the default
`~/.config/bsppctl/profiles.yaml`, in that precedence order.

| Profile area | Configuration |
| --- | --- |
| Identity and scheduler | `owner`, `account`, named `resources` entries. |
| Transport | `transport: ssh` requires `ssh_target`; `local-slurm` must omit it and runs Control where Slurm commands are available. |
| Filesystems | The example's `paths` block contains project/output/staging roots, source checkout, images, probe roots and qualification storage. Do not duplicate the same keys in both nested and flat forms. |
| Container mounts | `extra_mounts` entries use `source`, `target`, `read_only`; the default is writable. Explicitly mount immutable model/input assets read-only. |
| Folding images | `paths.image` serves non-fold actions; `folding_backend_images` selects the folding kernel image by backend. |
| Folding assets | `folding_backend_assets` gives container-visible checkpoints or model directories. Each selected asset needs matching read-only mount coverage. |
| Runtime evidence | Qualification roots, image policy, expiry and phase-specific provenance pins. |
| AWS file locations | `postprocessing_credential_mounts` identifies the credentials/config files; the same profile group is required by the current S3-based `validate-run`. |

Keep credentials outside plans, profiles, repositories and captured command
output. Profiles contain file locators, not credential contents.

`folding_release_preset` accepts `internal` and `public`; omission resolves to
`internal` on this source revision. Set `folding_release_preset: public`
explicitly when following the public BioIR examples. The preset is part of
qualification identity; changing it requires matching deployment evidence.
The normal image path uses baked orchestration packages
(`mount_orchestration_source: false` by default); a source path is not proof that
its code was installed into the selected image.

## Resources and packed folding

Resource entries include `partition`, `cpus_per_task`, `memory`, `time` and
optional scheduler fields `gres`, `array` and `nodelist`.
Relevant named entries include `control_cpu`, `gpu_worker`,
`analysis_finalize`, `acceptance_tar_payload_parity` and `acceptance_semantic`.

For scalar folding, omit the typed topology fields and use the site's scalar GPU
request, such as the example's `gres: gpu:1`. For packed folding, author
`nodes`, `tasks_per_node` and `gpus_per_task` together in `gpu_worker`, with
optional `max_parallel`; remove legacy `gres` and `array` strings.

Packed worker count is `nodes * tasks_per_node`, and `gpus_per_task` must be 1.
The renderer creates `nodes` one-node array elements, each running
`tasks_per_node` ranks. For example, 2 nodes × 8 tasks produces 16 ranks in an
array `0-1`, with global rank:

```text
SLURM_ARRAY_TASK_ID * tasks_per_node + SLURM_PROCID
```

`max_parallel` limits concurrent array elements and cannot exceed `nodes`.
Omitting it leaves no explicit array concurrency cap. This topology requests
capacity; it does not guarantee simultaneous starts or distinct physical hosts.
Verify actual allocations when those properties matter. The degenerate typed
1-node × 1-task shape is rejected; use scalar configuration instead.

## Phase-specific input

**Preprocessing:** the Plan binds a verified FASTA input, work plan, chunk
execution intent and Database Set selection. New scientific schema v3 fixes
ColabFold `--filter 1`; historical v1/v2 retain `--filter 2`. Changing scientific
schema changes identity and requires a new Phase rather than Retry.

Profiles map `(identifier, version)` Database Sets to real manifest paths.
Choose an allowed access policy explicitly: `direct`, `stage-required` or
`stage-preferred`. Staging requires the complete `database_cache_root`,
`database_cache_unix_user`, `database_cache_filesystem_type`,
`database_cache_reserve_bytes`, `database_lock_wait_seconds` group. Cache
ownership, filesystem/capacity and source identity are checked in the execution
environment. Use a complete manifest of the source tree. For placement details,
see [database placement](../preprocessing-database-placement-architecture.md).

`preprocessing_runtime` binds the imported image, OCI digest, image lock, three
wheel hashes, source commit/Source Bundle and tool versions. Fill the entire
group from canonical build/import and Source Bundle records, then qualify that
exact selection.

**Folding:** the Plan binds a verified `bspp.msa-set/v1` bundle and selects the
backend and seam transport. The schema names `openfold-cli`, `bioir`, `colabfold`
and `openfold-trt`; naming a backend is not evidence of an available executable
image (the TRT route fails closed in this release). Local artifact paths need
actual matching payload bytes wherever Control and Runtime verify them.

For mixed monomer/complex BioIR, explicitly author
`payload.bioir_model_policy` with `policy: expanded-chain-count-v1`, the
`openfold2_ptm_1` / `alphafold2_multimer_1` model sources, and measured checkpoint
sizes/hashes. Configure both `bioir_monomer_checkpoint` and `bioir_checkpoint`
in the profile. One expanded chain selects the monomer model; repeated chains
still count as multiple chains. Omitting the policy retains the historical
multimer-only behavior. A new model/checkpoint policy requires a fresh Plan/run.

`payload.evidence_profile: artifact-backed-v2` requires packed BioIR and an
explicit model policy. It keeps full scores on the execution filesystem while
Control fetches bounded indexed metadata. Omission retains legacy evidence.
The profile is fixed across Retry.

**Postprocessing:** the Phase Plan pins exact legacy Run Plan, acceptance policy,
logical-input inventory and Runtime Qualification documents, and declares its
output namespace. Keep the authored document bytes available; a filename alone
does not supply the pinned identity.

See [CLI commands](bsppctl.md) for qualification and
[phase lifecycle](../guides/phase-lifecycle.md) for the transition from authored
configuration to immutable authority and accepted handoffs.
