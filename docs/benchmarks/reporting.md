# Benchmark reporting guidance

This guide provides a template for documenting completed experiments.
It contains no benchmark results or claims that a run has completed.
Use [the workflow](bioir-workflow.md) and
[measurement methodology](methodology.md) when preparing a report.

Keep identifiers and immutable corpus pins in the
[dataset bundle](pdb-temporal-2022-2025-v1/README.md); link to it rather than
copying its reconstruction files. Keep canonical configuration examples in
[`skills/examples/run-plans`](../../skills/examples/run-plans/).

## Publication boundary

Follow [DEVELOPING.md](../../DEVELOPING.md). Public reports must exclude
internal hostnames, cluster/account names, user paths, storage locations,
credentials, and private operator handoffs. Do not copy raw private logs or
operational receipts into this tree. Retain originals in the appropriate
restricted evidence store and publish only an approved, independently checked
summary with public source/data/software identities.

Use neutral hardware labels and relative artifact names. A sanitized summary
must not imply that private evidence is publicly downloadable or that an
unpublished deployment patch is present in the release. State any reproducibility
limitation plainly. Preserve the original evidence and its provenance; sanitizing
a report is not permission to rewrite authoritative records.

## Suggested file layout

When publishing a completed, reviewed experiment, use a separate results
directory such as `results/YYYY-MM-DD-bioir-public-1000/README.md`. This path is
illustrative and does not identify an included result. Link associated
public tables or figures from that report, and identify their schema, units,
coverage, and generation method. Keep large generated predictions, MSAs,
reference coordinates and model weights outside Git; use approved immutable
artifact references when available.

Index published reports separately with a date, source revision, cohort,
hardware, and validation scope. Do not add a results-table row for a pending
run or fill missing measurements with estimates.

## Report template

Use the following sections. Replace every instruction with measured facts or
an explicit unavailable/not-run statement; do not present this template as data.

### Scope and outcome

State the experiment's question, date, complete/partial outcome, target count,
and whether it measures folding alone or the fresh end-to-end workflow.
Separate Phase acceptance, original-file integrity, benchmark validation, and
accuracy evaluation. Link the public dataset definition.

### Reproducibility manifest

| Item | Required report content |
| --- | --- |
| Source and software | Public source commit, deployment delta if any, package versions, immutable image identifiers, and installed image/source identity. |
| Cohort | Dataset fingerprint, ordered cohort/subset definition, sequence identity, expanded chain counts and total residues. |
| Science | BioIR policy, checkpoint provider/revision/hash/size, preprocessing scientific settings, and template/search settings. |
| MSA provenance | Fresh versus shared generation, database identity, member/content hashes, original/enriched artifact lineage, and cache mode. |
| Execution | Requested topology and actual nodes/GPUs, model/memory/driver, rank projection, per-rank work counts, resource/time limits and overlap. |
| Validation | Suite and corpus identity, exact checks performed, passed/failed/missing counts, and any preregistered accuracy criteria. |
| Evidence availability | Which sanitized artifacts are public, which originals are restricted, and the resulting limits on independent reproduction. |

Do not substitute a convenient source checkout for the source actually installed
in the images. Report each architecture or deployment source separately when
they differ.

### Measurements

Provide a table with values **and units**, the exact start/end boundary, evidence
source, and inclusion/exclusion policy. Include:

- Eligible queue delay and dependency delay separately.
- MSA placement/cache cost, native search, and total MSA allocation duration.
- CPU preparation and canonicalization durations.
- Folding allocation window, GPU-hours, target numerator, and both wall-clock
  and GPU-normalized throughput.
- Actual two-node overlap and observed device ownership/model evidence.
- Validation, transfer and closeout durations when reporting end-to-end cost.
- All failed/retried allocations and how carried predictions affect accounting.

For repeated trials, give individual measurements and the stated aggregation
method. One trial is one observation, not an estimate of run-to-run variance.
Compare only defined timing scopes. Explain any source, hardware, MSA, model,
cohort or worker-count difference between compared runs.

### Figures and per-target data

For each figure, give its input table, units, identity join, coverage, and missing
records. A length-time plot requires retained per-target measurements; packed
allocation time is not a substitute. Use expanded total residues for complex
length. Keep measured latency scopes separate and disclose initialization or
compilation treatment.

### Limitations and verification

State unresolved evidence gaps, excluded cases, unavailable validation, and
which conclusions the data supports. Report failures alongside successes.
Name the checks used to establish cohort identity, original-file integrity,
scheduler closure, actual parallelism, and acceptance. Structural RMSD reported
by the shipped suite is not an accuracy pass criterion.

Before publication, independently check the report's arithmetic and claims
against retained originals and confirm its links, tables, figures and public
metadata disclose no internal operational identities.
