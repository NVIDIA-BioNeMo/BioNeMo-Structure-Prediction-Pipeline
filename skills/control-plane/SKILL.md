---
name: control-plane
description: How to maneuver bsppctl across the phase lifecycle, coordinators, benchmark tooling, and runtime qualification.
---

# control-plane

## What this skill covers

This skill maps the `bsppctl` command families a phase-run operator needs, with each
command tied to the moment you use it. It covers the generic phase lifecycle, the
postprocessing-only coordinators and evidence commands, the folding-only benchmark
tooling, runtime qualification and release approval, the phase-specific
publish/qualify/resolve commands, and the data-movement planning command. For producing
the container image that a Cluster Profile points at, see the oci-containers skill; for
authoring Phase Plans and Cluster Profiles, see the configuring-runs skill. This skill
references those two by name only and does not duplicate their content.

## Generic phase lifecycle

These commands drive one Phase Run through materialize → submit → status/resume → cancel/retry
→ finalize. All of them take `--authority-root <root>`, the caller-selected directory that
holds the durable Phase Run authority.

- `phase materialize <phase-plan.yaml> --authority-root <root> [--source-repo <repo>]`
  — use when you have a finished Phase Plan and want to create one scheduler-free, immutable
  Phase Run. It prints a new opaque `phase_run_id` as JSON to stdout. `--source-repo` points at
  the clean committed orchestration source represented by the qualification (defaults to the
  current directory).
- `phase submit <phase-run-id> --authority-root <root>`
  — use when a materialized attempt is ready to be dispatched to the cluster. It returns after
  scheduler assignment.
- `phase status <phase-run-id> --authority-root <root> [--format table|text|json]`
  — use when you want a read-only observation of durable Phase state and scheduler accounting
  without mutation. Default format is `table`.
- `phase resume <phase-run-id> --authority-root <root>`
  — use when scheduler state changed and you want one bounded reconciliation cycle. It never
  infers terminal results from sparse or missing accounting.
- `phase cancel <phase-run-id> --authority-root <root>`
  — use when you deliberately stop an active attempt. Completion is only confirmed through
  terminal scheduler accounting, not by the cancel call itself. For preprocessing and folding,
  sparse terminal JSON triggers the existing exact job-identity/exit-code accounting fallback.
  An unset signal stays unknown; missing or conflicting evidence leaves cancellation pending.
  Repeating `phase cancel` continues its recorded intent and confirms already-terminal jobs
  without another cancellation request when exact accounting is available.
- `phase retry <phase-run-id> --authority-root <root> [--profile <profile-id>] [--source-repo
  <repo>] [--carry-forward <selection.json>]`
  — use when the current attempt is durably failed or accounting-confirmed cancelled and you
  want one clean immutable successor attempt. It materializes the successor but never submits,
  so follow with `phase submit` on the returned successor id. `--profile` selects an explicit
  successor Cluster Profile id.
- `phase finalize <phase-run-id> --authority-root <root> [--scheduler-evidence <file>]
  [--action-evidence <file>] [--handoff <dir>] [--acceptance-adjudication <file>]`
  — use when all actions have durable successful accounting and you want the attempt-bound
  receipt and to permanently seal the Phase Run. The flags are family-specific: postprocessing
  uses all four. Legacy folding uses `--scheduler-evidence` and `--action-evidence`;
  artifact-backed folding also requires the indexed `--handoff` bundle described below.
  All folding rejects `--acceptance-adjudication`.

### Preprocessing-only `--carry-forward`

The `--carry-forward <selection.json>` option on `phase retry` is preprocessing-only and is
rejected for other phases. It is an opt-in strict JSON selection of already-verified predecessor
A3M members to adopt into the fresh successor workspace. Omitting the option means the exact
no-carry transition.

## Postprocessing coordinators and indexed evidence

The coordinators, diagnostics and scheduler export below are postprocessing-only.
`phase evidence fetch` also accepts the explicitly selected folding v2 profile.

- `phase run-postprocessing <phase-plan.yaml> --authority-root <root> --execution-root <root>
  [--source-repo <repo>] [--poll-interval <s>] [--timeout <s>]`
  — use for a normal postprocessing run. It is the restart-safe coordinator: it materializes,
  submits, polls with bounded resume, fetches and validates the Runtime handoff, exports
  scheduler evidence, and finalizes. Re-run the identical command to restart after a crash.
  Authority and execution roots must be separate and outside the source checkout.
- `phase retry-postprocessing <phase-run-id> --authority-root <root> --execution-root <root>
  [--source-repo <repo>] [--poll-interval <s>] [--timeout <s>]`
  — use for the explicit restart-safe Retry coordinator. It persists one exact
  predecessor-to-successor Retry event in a create-once intent, then drives only that successor
  to finalization. It uses a new dedicated execution root.
- `phase diagnostics <phase-run-id> --authority-root <root> --diagnostics-root <root>`
  — use when troubleshooting. It recaptures bounded scheduler, failed-job, and acceptance
  diagnostics into a warning-only, non-authoritative summary outside Phase authority.
- `phase evidence fetch <phase-run-id> --authority-root <root> --destination <dir>`
  — use when all actions have durable successful accounting and you want to fetch and atomically
  validate the bounded indexed Runtime finalization handoff.
- `phase evidence export-scheduler <phase-run-id> --authority-root <root> --output <file>`
  — use when you want to export successful scheduler evidence from durable postprocessing events
  to a create-once local JSON path.

## Folding benchmark tooling

- `prepare-benchmark --spec <spec.json> --output <dir> [--workers <n>] [--reconstruct
  <targets.jsonl>]`
  — use when curating the pinned folding benchmark corpus into a new output directory (offline
  operator handoff) or reconstructing it from a pinned JSONL target list. `--reconstruct` skips
  discovery and rebuilds from the pinned target list.
- `validate-run --run-dir <dir> --suite <suite.json> --index <index.json> --corpus <s3-prefix>
  --fingerprint <f> --profile <profile-id> --output-dir <dir> [--aws-profile <name>]
  [--poll-interval <s>] [--timeout <s>]`
  — use when judging a completed folding run against the pinned composition + validity suite.
  It submits and monitors the cluster-side validation worker. Note it judges composition and
  validity, not structural accuracy.

## Folding legacy-MSA import

- `legacy-msa-import --handoff-root <dir> --output-dir <dir> --profile <profile-id>
  [--lz4 <name>] [--poll-interval <s>] [--timeout <s>] [--resume-job-id <scalar-id>]`
  — use when a legacy MSA artifact set lacks producer-attested member lengths and you need a new
  enriched artifact-set ID/location to author a folding Plan. It submits and monitors
  one bounded Runtime Slurm job via the selected Cluster Profile, verifies the existing
  manifest/location/bundle identity, derives the member lengths, atomically publishes enriched
  records, fetches only the small result records, and returns the new artifact-set ID/location.
  Each input or result metadata record is limited to 16 MiB; this limit does not
  apply to the tar/LZ4 payload bytes verified in Runtime. Legacy records and payload bytes
  are never modified. `--profile <profile-id>` selects the
  Cluster Profile; the global `--config <file>` option selects the profile file.
  If the submission response was lost, use `--resume-job-id` with the exact assigned scalar ID
  and the original handoff/output/profile/lz4 arguments. This read-only recovery
  verifies accounting owner/name/SubmitLine and the immutable staged script against current
  public rendering, requires exact `sacct` COMPLETED `0:0`, and reuses result validation.
  It never stages or submits another job. An expired `squeue` entry is tolerated; unavailable
  accounting or changed/missing script bytes fail closed. The staged script is an immutable
  file binding, not a claim of Slurm-retained historical script bytes.

## Packed fold array lifecycle

- **Scalar default vs packed array**: when the Cluster Profile authors the typed topology fields,
  the fold action materializes as a one-node array `--array=0-(N-1)[%M]` (the `%M` concurrency cap
  is applied only when `max_parallel` is set) with the global-rank formula
  `SLURM_ARRAY_TASK_ID * tasks_per_node + SLURM_PROCID`; omitted topology keeps the scalar fold
  action. Typed topology and the legacy `gres`/`array` strings are mutually exclusive.
- **Array terminal evidence**: `phase resume` records the exact fold-array terminal fact — the
  parent job plus every `<parent>_<index>` task with conclusive `sacct` state and exact exit code.
  `phase status` reports that durable array observation on the fold action. Canonical-pair
  reduction and `phase finalize` proceed only after that fold-array dependency has succeeded; they do
  not independently re-query or re-record every array task. The outcome is never inferred from an
  arbitrary parent or child row.
- **Retry behavior**: a partial fold failure triggers an attempt-level `phase retry` that
  automatically derives a folding-specific carry-forward record from verified rank journals,
  omits the carry record when zero completions verify, and submits the
  unchanged full N-element successor array where each rank adopts/skips valid carried targets.
  There is no caller-authored carry subset for folding.
  A change to an explicit BioIR model policy or checkpoint content identity is a
  scientific change and requires a fresh Phase Plan/run. Retry cannot change that
  policy while carrying old predictions.

## Runtime qualification and release approval

- `runtime qualify --profile <profile-id> [--source-repo <repo>] [--source-package-identity
  <file>]`
  — use when a governed run needs a fresh current Runtime Qualification record before submission.
  It prints a YAML `runtime_qualification:` record with the smoke job id and record path.
- `runtime resolve --profile <profile-id> [--source-repo <repo>]`
  — use after the qualification smoke job reaches a terminal state. It promotes and authenticates
  the exact record, exiting zero only with `current: true` plus the tuple id, path, size, and
  SHA-256 (nonzero otherwise).
- `release approve-publication <acceptance> --evidence-root <root> --expected-runspec-sha256 <h>
  --destination <dest> --approval <file>`
  — use when independently revalidating release evidence and creating one durable,
  acceptance-bound publication approval. Creating an approval does not upload anything.

## Phase-specific publish / qualify / resolve

- `phase publish-preprocessing <phase-run-id> --authority-root <root> --handoff-path <dir>
  --evidence-dir <dir> --profile <profile-id>`
  — preprocessing-only. Publish a verified local MSA-set bundle to the object store. It requires
  the RunSpec publish transport `publish-to-s3` and a publish prefix, and writes
  `artifact-location-remote.json` plus upload evidence to `--evidence-dir` (which must be outside
  authority). The runtime publish command executes cluster-side inside a Slurm job + the
  imported runtime container (never the local workstation interpreter), reading the bundle
  from Lustre and uploading to S3. The profile provides the `control_cpu` scheduling class
  and `postprocessing_credential_mounts` for AWS shared-profile access.
- `phase publish-folding <phase-run-id> --authority-root <root> --bundles <json> --local-paths
  <json> --evidence-dir <dir> --profile <profile-id>`
  — folding-only. Publish operator-attested prediction bundles to the object store, writing
  `prediction-bundle-upload-evidence.json`. Same cluster-side execution model as
  `publish-preprocessing`.
- `runtime preprocessing stage-source-bundle --profile <profile-id> [--source-repo <repo>]
  [--build-dir <dir>] [--dry-run]`
  — preprocessing-only. Build the provenance-only Source Bundle
  (`bspp-orchestration-<commit>.tar.zst`) from a clean checkout and stage its exact bytes to the
  profile `source_bundle_root` over the profile transport. The artifact retains the historical
  `.tar.zst` suffix but is an uncompressed tar, staged as immutable provenance and never
  extracted. It prints a YAML record whose
  `source_commit` and `source_bundle_sha256` are the values to pin in the profile
  `preprocessing_runtime` block; run it before `runtime preprocessing qualify`. `--dry-run`
  builds and verifies without staging.
- `runtime preprocessing qualify --profile <profile-id> [--source-repo <repo>]` and
  `runtime preprocessing resolve --profile <profile-id> [--source-repo <repo>]`
  — preprocessing-only runtime qualification pair: submit the scheduled smoke, then resolve the
  exact qualified record. Qualify requires the Source Bundle already staged and pinned (see
  `runtime preprocessing stage-source-bundle`).

## Data-movement planning

- `data plan --phase-plan <phase-plan.yaml>` or `data plan --source <src> --destination
  <key> --size-bytes <n> --sha256 <h>`
  — use when you want the Control Plane to plan a data-movement transfer before executing
  it. Plan-referenced mode reads a FoldingPhasePlan; manual mode takes the verified source,
  full destination object key, size, and digest. Options: `--s3-prefix <prefix>`
  (required in plan-referenced mode unless `--override-prefix` is given),
  `--override-prefix <prefix>` (recovery mode), `--dry-run`/`--no-dry-run` (default
  dry-run), and `--format json|yaml`.

## Configuration and Cluster Profile selection

`--config <file>` is a global `bsppctl` option. Resolution order is: explicit `--config`, then
the `BSPPCTL_CONFIG` environment variable, then the default
`~/.config/bsppctl/profiles.yaml`.

A Phase Plan selects its target cluster via `target_cluster` (see the configuring-runs skill).
Commands that act on a named profile take `--profile <profile-id>`: `phase retry`,
`runtime qualify`, `runtime resolve`, `runtime preprocessing stage-source-bundle`,
`runtime preprocessing qualify`/`resolve`, and `validate-run`.

The profile file is a YAML `clusters:` mapping; each named cluster carries transport, roots,
image, account, and resource defaults. See configuring-runs for authoring guidance and
oci-containers for producing the image that `paths.image` points at.

## Output conventions

Most lifecycle, evidence, and benchmark commands emit JSON to stdout. There is no
`--write-json` flag; redirect stdout to capture the result. `runtime qualify`, `runtime resolve`,
`runtime preprocessing stage-source-bundle`, and `release approve-publication` emit YAML to
stdout instead.

`phase status --format table|text|json` (default `table`): `table` and `text` render the
human-readable ordered action table, while `json` emits the stable structured JSON report. In
that report, `null` means a field is known empty and `missing` means it was absent from that
status mapping.

Coordinators emit JSON event lines while polling, then a final JSON result, and exit nonzero when
the final status is not `accepted`. `runtime resolve` and `validate-run` also exit nonzero on
non-current / non-accepted results, so scripts can key off the exit code.


## Artifact-backed folding finalization

For a packed BioIR Phase sealed with `evidence_profile: artifact-backed-v2`, first
use public `phase resume` to record exact assigned canonical-job success
(`COMPLETED`, `0:0`). Then use the existing command:

```bash
bsppctl phase evidence fetch "$phase_run_id" --authority-root "$authority_root" --destination "$bundle"
bsppctl phase finalize "$phase_run_id" --authority-root "$authority_root" \
  --scheduler-evidence "$scheduler_evidence" --handoff "$bundle" \
  --action-evidence "$bundle/$canonical_action_id/action-evidence.json"
```

Use the actual sealed canonical action ID and genuine successful scheduler
record. Folding does not gain the postprocessing-only scheduler export command.
The create-once bundle contains the fold handoff, canonical handoff/index and
aggregate action evidence, plus the index published last. Fetch verifies bounded
stable bytes, exact hashes and terminal authority; no prediction files transfer.
Control authenticates Runtime content attestations, not remote PDB/PAE bytes.
Preserve original scores and separately audit them when claiming campaign closure.
BioIR's producer writes its selected `bioir_model_source` into each new canonical
score file; artifact-backed validation requires that value to match the sealed
model policy. Missing provenance is a failure, even when in-memory model metadata
is correct. Deploy the producer repair through reviewed images; preserve failed
outputs and use a fresh Phase when the new kernel image changes Retry's frozen
action parameters.

Retry preserves the selected evidence profile and normal automatic verified
carry-forward. V2 accepts native and adopted outputs only under the current
Attempt's contained paths. It never re-adopts worker-completed outputs during
canonicalization; full-carry attempts may adopt once when workers were skipped.
An automatic Slurm requeue does not authorize reuse or replacement of a partial
workspace. Preserve it and use the public Retry flow after terminal observation.
