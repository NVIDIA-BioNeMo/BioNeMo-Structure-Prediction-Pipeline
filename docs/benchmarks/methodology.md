# BioIR benchmark measurement methodology

A useful benchmark binds a defined cohort and scientific configuration to
actual execution and clearly named timing boundaries. Follow the
[workflow](bioir-workflow.md) to produce accepted outputs, then use the
[report template](reporting.md) to disclose what was measured. Scheduling
a job, passing an image smoke, and accepting a Phase are different outcomes.

## Define the experiment before running it

Record the corpus fingerprint, ordered target IDs, sequence hashes, chain
counts, and lengths. For the public 1,000-target cohort, keep the assembly and
reference pins from the [dataset bundle](pdb-temporal-2022-2025-v1/README.md).
Use **total expanded residues** for target length: sum the length of every
modeled chain copy. A repeated polymer contributes once per copy; neither the
longest chain nor the unique-polymer length is the complex length.

Pin the following independently:

- Clean source commit and any deployment patch; package/wheel identities,
  immutable OCI digest, imported image hash, and installed image manifest.
- BioIR model policy, checkpoint content hashes/sizes, and scientific settings.
  The explicit mixed policy differs scientifically from legacy multimer-only
  execution.
- MSA source/database identity, preprocessing scientific schema and parameters,
  artifact-set/member identities, and exact tar/LZ4 payloads.
- Profile, Plan and materialized RunSpec identities; worker projection and
  per-rank target counts; actual GPU model, memory, driver and software stack.
- Suite, reference corpus, acceptance criteria, and any separately defined
  structural-accuracy thresholds.

State whether each run generated fresh MSAs or consumed a shared verified MSA
set. Fresh generation can yield identical member bytes; that does not mean its
preprocessing cost was zero. Conversely, consuming a previous set does not
measure a new preprocessing run. Compare sources and science before attributing
a difference solely to hardware or scheduling.

## Separate timing boundaries

Keep UTC timestamps and original scheduler records. Use a monotonic clock for
local command durations; do not subtract clocks from different hosts without a
verified common time basis.

| Measurement | Start and end | Include and disclose |
| --- | --- | --- |
| Submission command duration | Workstation invocation to CLI exit | Staging, transport and scheduler assignment; usually not completion of the assigned work. |
| Dependency or eligibility delay | Scheduler `Submit` to `Eligible` | Waiting for upstream actions or eligibility restrictions. |
| Eligible queue delay | `Eligible` to `Start` | Scheduler admission after eligibility. Unknown timestamps remain unknown. |
| MSA allocation wall time | Actual preprocessing allocation `Start` to `End` | Input fetch, database placement, native search, validation and publication within that allocation. |
| Database placement | Retained placement start/end and Result | Cold population, verified warm reuse, direct access, or failure. Do not label a reused cache as a fresh copy. |
| Native MSA search | Genuine native search start/end | Label separately from placement and post-search validation/publication. |
| Folding CPU preparation | Each `msa-flatten`, `split`, and `preprocess` allocation | CPU action time and queue delay separately. The action called `preprocess` here is not the upstream MSA Phase. |
| Folding allocation window | Earliest fold-element `Start` to latest fold-element `End` | Concurrent and staggered scheduling, model load, compilation, target processing and idle tail within allocations. |
| Canonicalization and acceptance | Canonical action, then workstation closeout | Canonical reduction, evidence fetch and public finalization are separate from GPU allocation time. |
| Benchmark validation | Scheduled validation allocation and its command duration | S3 corpus fetch/integrity checks and suite execution; report outside inference throughput. |
| Campaign end-to-end time | Explicitly chosen campaign start to accepted/validated end | State inclusion of builds, imports, qualification, queueing, retries, transfers and validation. |

Only subtract timestamps when they describe the same physical allocation epoch.
Preserve original `Submit`, `Eligible`, `Start`, `End`, `ElapsedRaw`, state,
exit code, node list, requested/allocated resources, and partition. Capture
terminal restart counts while the scheduler still retains them. A transient
`RUNNING` snapshot does not replace a later terminal record.

## Compute throughput and resource exposure

For physical fold allocation epoch `i`, let `e_i` be its actual elapsed seconds
and `g_i` its authenticated allocated GPU count. For a fixed device count within
each epoch:

```text
GPU-hours = sum(g_i * e_i) / 3600
GPU-normalized throughput = newly computed accepted targets / GPU-hours
Fold-window throughput = newly computed accepted targets /
                         (latest fold End - earliest fold Start)
```

State the units of the last rate, such as targets/second or targets/hour. The
allocation-based GPU metric includes initialization, compilation and any idle
tail inside the allocation; it is not pure model inference efficiency.
Report absolute elapsed time alongside normalized throughput.

Count every physical allocation once. Array parents, elements, batch steps,
extern steps and `srun` steps can describe overlapping execution; summing all
rows double-counts resources. Preserve the parent/element-to-physical-job mapping
and separate requeues or restarts into epochs when the evidence supports it.
Do not substitute the requested GPU count for a missing allocated count.
If scheduler accounting is insufficient, require independent allocation and
device-ownership evidence; otherwise leave GPU-normalized results unavailable.

Keep a complete ledger of preparation, diagnostics, failed attempts and retries.
For a no-retry successful run, the unique accepted cohort is the throughput
numerator. When Retry carries predictions, distinguish newly computed targets
from adopted outputs: dividing the entire accepted cohort by only the final
attempt's compute would overstate throughput. Report campaign cost including
all relevant attempts, and label any successful-attempt-only view separately.
Do not add queue duration to GPU-hours when no GPU was allocated.

Node-hours and CPU allocation time may be reported separately using actual
allocation records. Neither is interchangeable with GPU-hours. Scheduling
requests can differ from effective allocations, so record both.

## Prove parallelism

A Plan with two nodes and `max_parallel: 2` permits two array elements to run
concurrently. It does not prove that they did, that they occupied distinct
nodes, or that their GPU models matched.

For each actual element, retain its physical job identity, node, start/end
interval, restart history, and any measured rank/device bindings. The shipped
executor does not automatically log a GPU UUID for every rank. If device-level
claims require additional evidence, collect original GPU UUID/model observations
within the actual scheduled allocation and bind them to its job, node, and time.
Cross-check those observations against scheduler ownership and isolation. Do not
claim per-rank device mapping when it was not recorded. Distinct ranks must not
accidentally refer to the same device; visibility alone does not prove exclusive
allocation ownership.

For two intervals `[s0, e0]` and `[s1, e1]`, compute measured overlap as:

```text
overlap_seconds = max(0, min(e0, e1) - max(s0, s1))
```

Report overlap, the observed number of distinct simultaneous nodes/devices,
and any period when only one element ran. Do not infer parallelism from the
submission time, `%2` array limit, a test-only admission forecast, or total
requested GPU count. Preserve complete original logs containing initial device
records; a short tail may omit the only UUID evidence.

For hardware comparisons, report the exact GPU models and counts. If cohorts
run on different hardware, source revisions, worker projections or resource
limits, describe those differences explicitly instead of presenting a
single-variable speedup.

## Per-target timing has a narrower meaning

The BioIR session measures `fold_wall_seconds` around its processor call and
also receives upstream `model_inference_time` and `time_taken` fields. These
are different scopes. Model/session initialization can occur outside that
processor timer; the timer is not an allocation wall clock.

In the current packed executor, rank journals retain target identity and output
pins but not those timing fields. Reduction reconstructs model metadata rather
than preserving the session's quality/timing metadata. Therefore an accepted
packed run does **not** by itself provide measured per-target latency. Inspect
actual retained outputs/logs before claiming timing coverage; do not assume an
in-memory field was persisted.

For a sequence-length versus time plot, use only genuine measured durations
with a defined scope, units, target identity and source. State coverage and
missingness for each run. Pair exact shared target/sequence identities before
comparing cohorts. Do not turn file modification times, first-seen polling
times, differences between completion timestamps, or job-time averages into
per-target inference measurements.

Implementation boundaries:
[BioIR session](../../packages/orchestration-runtime/src/bspp/orchestration/runtime/folding/execution/bioir_session.py),
[packed executor](../../packages/orchestration-runtime/src/bspp/orchestration/runtime/folding/executor.py),
and [benchmark validator](../../packages/orchestration-runtime/src/bspp/orchestration/runtime/folding/benchmark/validation.py).

## Keep outcome claims distinct

Use separate report fields for:

1. Scheduler success and complete terminal/restart evidence.
2. Accepted Phase Receipt with exact output lineage and cohort coverage.
3. Independent integrity checks of original prediction files and scores.
4. Benchmark suite acceptance against the verified corpus and suite.
5. Any independently specified structural-accuracy evaluation.

The shipped suite validates composition, output validity, finite confidence
values, and reference C-alpha coverage. It reports C-alpha RMSD without applying
an RMSD pass threshold. Confidence scores alone do not establish accuracy.
An unavailable S3 validation path means validation was not run, even when
folding acceptance succeeded. Pending jobs contribute no completed benchmark
result; distinguish provisional progress from a reproducible final report.
