---
name: configuring-runs
description: How to prepare Run Plans, Phase Plans, and Cluster Profiles for BSPP phase runs.
---

# Configuring runs

## What this skill covers

This skill teaches you how to prepare the three configuration artifacts that drive an BSPP run:
the **Run Plan**, the **Phase Plan**, and the **Cluster Profile**. It explains how the three
artifacts divide responsibility, how the three phase envelopes (preprocessing, folding, and
postprocessing) shape each Phase Plan, where you retrieve each piece of site-specific information
you will need to fill in, and how a Phase Plan selects a Cluster Profile by name. Concrete,
schema-loadable starting points live in the public example files under
`skills/examples/run-plans/`; this skill points to them rather than duplicating their
field-by-field detail.

## The three configuration artifacts

The three artifacts have deliberately different jobs. Keep them separate: a plan states *what* you
want to run, while a profile states *where and how* it can run.

### Phase Plan

The **Phase Plan** is the modern, user-authored statement of intent for exactly one phase. It
declares the phase via its `phase_kind` field, names the target cluster via its top-level
`target_cluster` field, and carries the phase-specific payload (plus a verified `input_location`
for preprocessing and folding). The Control Plane command `bsppctl phase materialize` validates a
Phase Plan and turns it into an immutable Phase RunSpec, which is the authority that submission,
status, resume, retry, and finalization operate on. A Phase Plan is the primary document you write
today.

### Run Plan

The **Run Plan** is the legacy postprocessing intent document. Its fields include `run_kind`,
`workflow_template`, `dataset`, `references`, `worker`, `storage`, `acceptance`, and others. Today
it is not driven directly: the postprocessing Phase Plan pins the legacy Run Plan's exact bytes as
one of four authority documents, so the Run Plan remains an authored input but is consumed through
the Phase Plan rather than executed on its own.

The relationship between the two plan kinds differs by phase:

- **postprocessing** has a legacy Run Plan, pinned by exact bytes inside the Phase Plan.
- **folding** has no separate legacy Run Plan; its single Phase Plan serves both roles.
- **preprocessing** in the current slice is Phase-Plan-only.

### Cluster Profile

The **Cluster Profile** is the cluster-specific operational configuration. It records the owner,
transport, account, filesystem roots, container image, Slurm partition and resources, mounts,
database placement, folding backend assets, packed folding topology, and runtime qualification
roots. A Cluster Profile is shared and reusable across phases, and it is selected by name — never
inlined into a Phase Plan.
Because it is shared, editing one profile adapts every plan that references it.

## The three phase envelopes

Each phase has its own envelope: the same outer shape with a phase-specific payload. The exhaustive
field-by-field enumeration belongs to the example files; this section gives the high-level shape.

### preprocessing

Input is one verified FASTA file (`input_location`, a `.fa`). The payload is a bounded work plan
plus chunk execution intent plus a database set selection (identifier/version and access policy)
used by the MSA search.

New preprocessing plans use scoped scientific `schema_version: 3`. This fixes
ColabFold `--filter 1`: unpaired MSA filtering remains enabled, while paired
rows retain their shared alignment order. Historical scientific v1/v2 retain
`--filter 2` and their exact serialized identities. Switching an old plan to
v3 changes its scientific identity and requires a new Phase, rather than a
Retry of the old scientific run. Targets and model settings are preserved,
but paired-derived MSA rows and downstream features may change. The outer
Phase, adapter, and global Contract versions remain unchanged.

### folding

Input is a verified `bspp.msa-set/v1` artifact set (a local or remote bundled location). The
payload selects one of four backends (`openfold-cli`, `bioir`, `colabfold`, `openfold-trt`) and a
per-seam transport policy (publish to object storage vs. local pass-through). External backend
assets — the chain-manifest CSV, model/weights directories, and checkpoints — come from the
Cluster Profile, not from the Phase Plan.

Original HumanSTRING dimer IDs (`homo_<accession>` and
`hetero_<accession>_<accession>`) are supported by folding and public legacy-MSA
import. Preserve them exactly in `a3ms/<target_id>.a3m`; do not add AF prefixes
or create aliases. Their preprocessing scientific settings explicitly use
`require_afdb_model_id_stem: false`. The dedicated syntax accepts uppercase
six- or ten-character accessions, with two distinct accessions for `hetero_`.
Both forms have two expanded chains and retain the multimer model route.
HumanSTRING postprocessing is unsupported and fails at inventory discovery;
canonical folding acceptance and confidence/file-integrity audits do not
provide reference-structure accuracy validation.

### postprocessing

Input is the prediction handoff. The Phase Plan pins four authority documents by exact bytes — the
legacy Run Plan, the acceptance policy, the logical-input inventory, and the runtime qualification
record — and declares an `output_namespace`.

## Where to retrieve each input

You will need several pieces of site-specific information before you can fill in the plans and
profile. Obtain each from its owner; never invent values and never paste credentials into an
authored file.

### Object-storage endpoint and bucket

Obtain the endpoint URL and bucket name from the storage administrator or the platform's storage
console. Record them as `s3://<bucket>/<prefix>`-style values in the Run Plan `storage`/transport
fields. Access keys and other secret material live outside the authored files; plans and profiles
reference them by name or path only.

### Cluster partitions and accounts

Obtain the Slurm account and the list of partition names (with their CPU/GPU/memory/time limits)
from the cluster's scheduler configuration or the HPC support team. Record them in the Cluster
Profile `account` and `resources` fields (partition, cpus_per_task, memory, and time). For
`gpu_worker`, author either the legacy `gres` string or the optional typed packed-topology fields
`nodes`, `tasks_per_node`, `gpus_per_task`, `max_parallel` — never both, see the next subsection.

### Packed folding topology

The `gpu_worker` resource entry accepts four optional typed packed-topology fields:
`nodes`, `tasks_per_node`, `gpus_per_task`, and `max_parallel`.

- **Scalar default**: omit all four fields and materialization renders today's scalar fold action —
  `tasks_per_node=1`, no array, no `gpus_per_task`, with `gres: gpu:1` as the GPU request. This is
  the historical scalar action shape.
- **Packed**: author `nodes`, `tasks_per_node`, and `gpus_per_task` together (`max_parallel` is
  optional) and materialization renders a one-node Slurm array. `workers = nodes * tasks_per_node`;
  the topology is packed iff `workers > 1`. The array is `--array=0-(N-1)[%M]` with `--nodes=1`,
  `--ntasks-per-node=<tasks_per_node>`, and `--gpus-per-task=<gpus_per_task>`. Each array element's
  global rank is `SLURM_ARRAY_TASK_ID * tasks_per_node + SLURM_PROCID`.
- **All-or-none typed topology**: `tasks_per_node`, `gpus_per_task`, and `max_parallel` cannot be
  authored without `nodes`; typed topology requires `gpus_per_task`; and the degenerate
  `nodes=1, tasks_per_node=1` shape is rejected rather than silently rendering a GPU-less scalar
  action.
- **Optional cap**: `max_parallel` is optional. When supplied it must satisfy `max_parallel <= nodes`
  and renders `--array=0-(N-1)%M`; when omitted it renders the bare `--array=0-(N-1)` with no
  concurrency cap.
- **Exclusivity**: typed topology and the legacy `gres`/`array` strings are mutually exclusive
  Authoring `nodes` together with `gres` or `array` is rejected by
  `resolve_cluster_profile`, so remove the `gres` line when you author packed topology.
- **Constraints**: `nodes >= 1`, `tasks_per_node >= 1`, and packed fold requires `gpus_per_task`
  exactly 1.

See `skills/examples/run-plans/cluster-profile.yaml` for the concrete shape; the example keeps the
fields commented so the shared profile stays scalar-by-default and loadable.

### Backend assets

Obtain the model weights, the chain-manifest CSV, and the per-backend checkpoints from the
scientific-data or model-registry owner. Record their container-visible filesystem paths in the
Cluster Profile `folding_backend_assets` field, and cover each path with an exactly matching
read-only `extra_mounts` entry.

For a mixed monomer/complex BioIR run, explicitly author `payload.bioir_model_policy`
with `schema_version: 1`, `policy: expanded-chain-count-v1`, model sources
`openfold2_ptm_1` and `alphafold2_multimer_1`, and the measured SHA-256 and byte size
of both checkpoints. Single expanded chains use the native OpenFold pTM checkpoint;
two or more chains, including repeated copies of one polymer, retain the multimer
checkpoint. This is a scientific model choice recorded in the Plan and action digests.
Configure `bioir_monomer_checkpoint` alongside `bioir_checkpoint` in the selected
profile, each with an exact read-only mount. Native OpenFold `.pt` weights require
no AlphaFold parameter conversion. Preserve provider/revision/license provenance.
An omitted policy preserves historical multimer-only behavior; it does not enable
monomers. A policy change requires a fresh Phase Plan/run, not a Retry of old science.
Full finite PAE/pLDDT remain required; the pTM model provides genuine pTM and
undefined monomer ipTM is `null`.

## Selecting a Cluster Profile with `target_cluster`

A Phase Plan binds to a Cluster Profile by naming it in the Phase Plan's top-level `target_cluster`
field. That name must resolve to an entry in the shared Cluster Profile file. The shared public
example is `skills/examples/run-plans/cluster-profile.yaml`; the same profile is referenced by all
three phase examples, so a user edits one profile to adapt all three plans to a new site.

## Example starting points

The public example files live under `skills/examples/run-plans/`:

- `skills/examples/run-plans/preprocessing-phase-plan.yaml`
- `skills/examples/run-plans/folding-phase-plan.yaml`
- `skills/examples/run-plans/postprocessing-phase-plan.yaml`
- `skills/examples/run-plans/cluster-profile.yaml`

The existing internal reference directories under `docs/examples/` —
`docs/examples/folding-benchmark/` and `docs/examples/postprocessing-reference-run/` — offer further
concrete shapes. The public examples are schema-loadable but intentionally non-runnable until you
replace the placeholder identities, paths, and (for postprocessing) the runtime qualification
record.

## Related skills

- `skills/oci-containers/SKILL.md` — building, publishing, importing, and qualifying containers.
- `skills/control-plane/SKILL.md` — driving `bsppctl`.


### Bounded evidence for large packed BioIR runs

Author `payload.evidence_profile: artifact-backed-v2` only with packed BioIR and
an explicit `bioir_model_policy`. The existing policy is also valid
for all-dimer runs: expanded two-chain targets retain its multimer checkpoint.
The profile changes execution/evidence identity, keeps input and scientific
identity unchanged, and is frozen across Retry. Omission preserves the legacy
score-bearing records and receipts.

Full original scores, including finite square PAE, remain on the execution
filesystem. Control fetches four indexed metadata files; it does not fetch PAE.
Before launching a large campaign, measure the implemented serializers using
actual target IDs, sequences and planned paths. Limits are 16,384 targets,
1 GiB per score file, 128 MiB per metadata file, and 256 MiB per bundle, with a
1 MiB finalization index. Exceeding a bound rejects the action; it never truncates
science. Configure scalar memory for bounded metadata and one full score matrix.
Use an independent read-only original-file integrity audit after finalization;
without reference structures, do not describe it as reference validation.
