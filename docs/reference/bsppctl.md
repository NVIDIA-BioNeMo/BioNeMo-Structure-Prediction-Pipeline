# `bsppctl` reference

`bsppctl` authors durable phase authority and coordinates scheduler work. Start
with the [phase lifecycle guide](../guides/phase-lifecycle.md) and
[configuration reference](configuration.md). This reference describes the
commands shipped in this release; use `bsppctl COMMAND --help` for option types
and defaults.

## Global configuration

Place the global `--config` option **before** the command group:

```bash
bsppctl --config profiles.yaml phase materialize phase-plan.yaml \
  --authority-root /path/to/authority --source-repo /path/to/checkout
```

The configuration file is selected in this order: explicit `--config FILE`,
`BSPPCTL_CONFIG`, then `~/.config/bsppctl/profiles.yaml`. Its `clusters:` mapping
contains named profiles. A Phase Plan selects one with `target_cluster`;
commands that operate directly on a profile use `--profile NAME`.

Paths and names in examples are placeholders. Plans, profiles and declarations
are operator-authored inputs; IDs, hashes, successful evidence and receipts must
come from the actual files and operations they describe.

## Phase commands

Every command in this table requires `--authority-root DIRECTORY`. `PLAN` is a
positional Phase Plan filename; `ID` is the actual `phase_run_id`, not a Slurm
job ID.

| Command after `bsppctl phase` | Additional arguments | Effect |
| --- | --- | --- |
| `materialize PLAN` | Optional `--source-repo DIRECTORY` | Validate inputs and create immutable attempt authority without submitting jobs. |
| `submit ID` | None | Dispatch the current materialized attempt and record scheduler assignments. |
| `status ID` | Optional `--format table\|text\|json` | Read durable state and scheduler observations without updating authority. |
| `resume ID` | None | Perform one bounded reconciliation with scheduler accounting. |
| `cancel ID` | None | Record cancellation intent and request cancellation; terminal accounting confirms the outcome. |
| `retry ID` | Optional `--profile NAME`, `--source-repo DIRECTORY`, `--carry-forward FILE` | Materialize a successor attempt after a durably failed or confirmed-cancelled attempt; does not submit. |
| `finalize ID` | Family-specific evidence below | Validate successful evidence, issue a receipt and seal the Phase Run. |
| `diagnostics ID` | Required `--diagnostics-root DIRECTORY` | Collect non-authoritative diagnostics; postprocessing only. |

Retry retains the Phase Run ID and advances its Attempt. Inspect the returned
IDs before submitting. `--carry-forward` is a
preprocessing-only selection of verified predecessor A3Ms. Folding derives its
own verified carry-forward from rank journals; it rejects a caller-authored
carry selection. Changing the scientific model policy requires a fresh Plan/run.

### Finalization inputs

Although Click lists these flags as optional, the selected phase requires the
combination below. Each flag takes an existing file, except `--handoff`, which
takes an existing directory.

| Phase / evidence profile | Required flags |
| --- | --- |
| Preprocessing | `--scheduler-evidence`, `--action-evidence`, `--handoff` |
| Folding, legacy evidence | `--scheduler-evidence`, `--action-evidence` |
| Folding, `artifact-backed-v2` | `--scheduler-evidence`, `--action-evidence`, `--handoff` |
| Postprocessing | All three above, plus `--acceptance-adjudication` |

Preprocessing and folding reject `--acceptance-adjudication`. Finalization
acceptance is distinct from publication and from reference-coordinate validation.

### Evidence commands

```text
bsppctl phase evidence fetch ID --authority-root DIRECTORY --destination DIRECTORY
bsppctl phase evidence export-scheduler ID --authority-root DIRECTORY --output FILE
```

`fetch` supports postprocessing and folding with `artifact-backed-v2`, after
successful accounting is durable. It validates indexed Runtime metadata into a
new local destination; it does not copy folding PDB/PAE payloads. Create its
parent directory first. It is not a preprocessing handoff-fetch command.

`export-scheduler` is **postprocessing-only** and writes a create-once JSON file.
Preprocessing and folding require a caller-provided, typed successful scheduler
record bound to genuine assigned-job accounting. There is no generic scheduler
export command for those phases; see the [lifecycle guide](../guides/phase-lifecycle.md).

### Postprocessing coordinators

```text
bsppctl phase run-postprocessing PLAN --authority-root DIRECTORY --execution-root DIRECTORY
bsppctl phase retry-postprocessing ID --authority-root DIRECTORY --execution-root DIRECTORY
```

Both accept `--source-repo DIRECTORY`, `--poll-interval SECONDS` (default 30) and
`--timeout SECONDS` (default 86400). They coordinate submission, bounded polling,
evidence fetch, scheduler export and finalization. Repeating the same coordinator
command resumes its recorded work. Retry uses a new dedicated execution root.
Authority and execution roots must be separate and outside the checkout.

## Qualification

Each command below uses the selected global config and requires `--profile NAME`.

| Command | Other options and prerequisites |
| --- | --- |
| `runtime qualify` | `--source-repo DIRECTORY`; `--source-package-identity FILE` is required for SSH profiles. Creates or refreshes governed Runtime Qualification evidence. |
| `runtime resolve` | `--source-repo DIRECTORY`. Resolves the exact qualification record; exits nonzero if it is not current. |
| `runtime preprocessing stage-source-bundle` | `--source-repo DIRECTORY`, `--build-dir DIRECTORY`, `--dry-run`. Build provenance from clean committed source; stage immutable bytes unless dry-run. |
| `runtime preprocessing qualify` | `--source-repo DIRECTORY`. Requires the staged Source Bundle and the exact image/source pins in `preprocessing_runtime`; submits the scheduled smoke. |
| `runtime preprocessing resolve` | `--source-repo DIRECTORY`. Resolve the qualified cluster record into Control state after the smoke completes. |

Qualification applies to its recorded source/image/site tuple. It does not stand
in for model-weight checks or a successful GPU inference. Source Bundle output
supplies the measured `source_commit` and `source_bundle_sha256` to pin before
preprocessing qualification. Despite its historical `.tar.zst` filename, that
provenance bundle is an uncompressed tar and is not extracted into the image.

## Benchmark and MSA commands

```text
bsppctl prepare-benchmark --output DIRECTORY [--spec FILE] [--reconstruct FILE] [--workers N]
bsppctl legacy-msa-import --handoff-root DIRECTORY --output-dir DIRECTORY --profile NAME
bsppctl validate-run --run-dir DIRECTORY --suite FILE --index FILE --corpus S3_PREFIX \
  --fingerprint HASH --profile NAME --output-dir DIRECTORY
```

`prepare-benchmark` creates a new corpus directory. `--reconstruct` selects a
pinned target JSONL instead of fresh discovery. The default spec is
`configs/benchmark.pdb-temporal-v1.json`; the default worker count is 12.

`legacy-msa-import` takes cluster-resident handoff and output directories and
schedules a bounded Runtime job to enrich a genuine MSA handoff with member
lengths. It preserves original records and archive bytes
and emits new artifact-set/location records. Optional flags are `--lz4 NAME`
(default `lz4`), `--poll-interval SECONDS`, `--timeout SECONDS`, and
`--resume-job-id SCALAR_ID`. The last flag verifies and monitors that exact
already-submitted job; it never submits another job. Keep the original inputs,
profile and output arguments when using it.

`validate-run` submits a separate cluster-side judge. Run, suite, index and
output paths are **cluster-resident**. This release requires an S3 corpus;
there is no `--corpus-dir` option. Select a validation-capable `paths.image`
(such as the postprocessing image) and configure the profile's AWS
credential/config file mounts, with optional `--aws-profile NAME`. The small
folding Runtime image lacks the validation worker's PyArrow dependency. It also accepts
`--poll-interval SECONDS` and `--timeout SECONDS` (30 and 86400 by default).
It writes `summary.json` and `validation.parquet`. The judge checks identity,
finite scores and coordinate coverage; it reports C-alpha RMSD without gating
on an RMSD accuracy threshold.

## Publication and seam derivation

These phase commands also require `--authority-root DIRECTORY`:

| Command | Other required arguments |
| --- | --- |
| `phase publish-preprocessing ID` | `--profile NAME --handoff-path DIRECTORY --evidence-dir DIRECTORY` |
| `phase publish-folding ID` | `--profile NAME --bundles FILE --local-paths FILE --evidence-dir DIRECTORY` |
| `phase derive-seam-parquets ID` | `--index FILE --evidence FILE --master-output FILE --tracking-output FILE --s3-output-prefix PREFIX --source-run NAME --archive-name NAME --evidence-dir DIRECTORY` |

Publishing requires the corresponding declared transport and verified artifacts;
evidence/output destinations must stay outside authority. Both publication
commands require a profile matching the frozen RunSpec and schedule a CPU Slurm
job inside the imported Runtime container. Configure `control_cpu` resources
and AWS credential/config file mounts. Control reads the local handoff or bundle
declarations, while the declared archive paths must be readable on the cluster.
The commands monitor the job and return small upload-evidence files to the local
`--evidence-dir`; archive payloads remain cluster-side until their S3 upload.
Their JSON result contains `job_id` and `evidence_files`.
Folding bundle and local-path files are operator-provided JSON declarations of
actual archives.
Seam derivation runs a local Runtime transform and additionally accepts
`--gcs-destination-prefix PREFIX` and `--force`. Its command is under `phase`,
not at the top level.

```text
bsppctl release approve-publication ACCEPTANCE --evidence-root DIRECTORY \
  --expected-runspec-sha256 HASH --destination DESTINATION --approval FILE
```

This independently revalidates acceptance evidence and creates a publication
approval. Creating the approval does not upload data.

## Data-movement planning

Choose one mode:

```text
bsppctl data plan --phase-plan FILE --s3-prefix S3_PREFIX
bsppctl data plan --source SOURCE --destination S3_OBJECT_KEY --size-bytes N --sha256 HASH
```

`--destination` is a full object key, not just a prefix. Both modes accept
`--override-prefix PREFIX`, `--dry-run` / `--no-dry-run` (dry-run is the default),
and `--format json|yaml` (default YAML). Plan mode requires `--s3-prefix` unless
`--override-prefix` is supplied. Supply measured source size and digest in manual
mode; this command is not Phase submission or acceptance.

## Output handling

Most phase and benchmark commands emit JSON. Runtime qualification/resolve,
Source Bundle staging and publication approval emit YAML. Status defaults to a
table; request `--format json` for structured output. Coordinators emit JSON
event lines before their final result. Capture stdout by shell redirection;
there is no `--write-json` flag. Check exit status as well as output, and retain
stderr separately when automating.

See the repository [control-plane skill](../../skills/control-plane/SKILL.md)
for the broader operator workflow.
