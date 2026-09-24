# Implementation status and release limits

This page describes the main implementation rooted at
`9af63428fb0b721d9c81f98c3e41d045d87184d9`. Documentation on another branch or a
locally patched image does not establish the behavior of this source snapshot.

## Implemented surfaces

| Surface | Available behavior | Execution requirements |
| --- | --- | --- |
| Control | `bsppctl` Phase materialization, submission, observation, reconciliation, cancellation, retry and finalization | Site-specific Profile, valid Plan and persistent authority directory |
| Preprocessing | Adapter-v3 execution and explicit database-placement policies | Search databases, compatible GPU Runtime and preprocessing qualification |
| BioIR folding | BioIR backend with explicit checkpoint/model policy and packed execution | Verified MSA artifacts, checkpoints, matching images and GPU resources |
| Other folding backends | `openfold-cli` and `colabfold` executor composition | Their respective images and assets; validate each selected environment |
| `openfold-trt` | Contract value only; execution fails closed | Deferred model factory support is not supplied |
| Postprocessing | Runtime actions and restart-safe run/retry coordinators | Qualified toolkit, predecessor artifacts and configured acceptance policy |
| Benchmark reconstruction | Public 1,000-target reconstruction and identity checks | Public download access and PyArrow |
| Benchmark validation | Scheduled completed-run validation | S3 corpus location, suite, canonical index and compatible Runtime |

This table describes code paths, not a promise of production readiness,
structural accuracy or validated performance on every GPU architecture.
Hardware and image qualification must match the execution environment; consult
[containers](guides/containers.md) before selecting an architecture.

## Operator boundaries

### Select the release preset explicitly

This revision supports both `internal` and `public` folding release presets.
Omitting `folding_release_preset` selects `internal`; set
`folding_release_preset: public` explicitly for the public BioIR configuration.
See the [configuration reference](reference/configuration.md) for the related
profile and backend-asset settings.

### Templates need verified values

The canonical [Plan templates](examples/README.md) contain fake paths and
identities chosen to illustrate the strict schema. Passing schema validation
does not establish that an image, input file, checkpoint or remote object
exists. Replace illustrative values with verified artifacts before execution.

### The workflow has explicit phase handoffs

There is no single public coordinator that executes the entire preprocessing →
folding → postprocessing benchmark. The postprocessing coordinators manage that
phase. Preprocessing and folding use the explicit lifecycle.

Several boundaries require operator-supplied artifacts: a verified database
inventory, materialized input identity, preprocessing handoff transfer and typed
scheduler evidence for finalization. `phase evidence fetch` supports
postprocessing and selected artifact-backed folding; it is not a generic
preprocessing download command. `phase evidence export-scheduler` is
postprocessing-only. See the [lifecycle guide](guides/phase-lifecycle.md) and
[BioIR walkthrough](benchmarks/bioir-workflow.md) for the supported command sequence.

### Public data access and validation transport are separate

`prepare-benchmark --reconstruct` downloads the public corpus without S3
credentials. The shipped `validate-run` still requires `--corpus`.
It does not accept a local-corpus argument. Downloading the public references
therefore does not, by itself, satisfy the validation command's transport setup.

### Completion has distinct meanings

A successful scheduler exit establishes job completion. Phase finalization
checks bound execution evidence and seals a receipt. Benchmark validation
checks completed predictions against its configured suite and corpus. Report
these outcomes separately; none alone establishes experimental accuracy.

### Packed execution does not retain per-target model latency

BioIR measures timing in its in-memory result metadata, but the current packed
executor's persisted score and rank-journal records omit those timing fields.
Allocation-based throughput can be measured. Exact sequence-length-versus-model-
time plots cannot be reconstructed from those artifacts; file modification
times are not a substitute. See [methodology](benchmarks/methodology.md).

### Qualification and recovery depend on exact evidence

A changed image, source, mount topology or resource configuration can invalidate
an existing qualification. Scheduler forecasts are advisory; an array limit
does not guarantee simultaneous node allocation. Missing terminal records,
partial dispatch or an absent per-rank journal can prevent automatic recovery.
Use [troubleshooting](guides/troubleshooting.md) and preserve the original
Attempt's records rather than fabricating completion evidence.

## Reporting executed validation

Use the [reporting guidance](benchmarks/reporting.md) to document exact
source/image/data identity, settings, hardware, timing boundaries, failures and
validation outcomes. Pending runs and native smoke checks should not be
presented as completed full benchmark results. Keep site access details and
raw operational records outside the public documentation.
