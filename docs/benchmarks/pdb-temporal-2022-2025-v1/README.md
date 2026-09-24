# Public download list for the 1,000-target folding benchmark

This bundle describes the `pdb-temporal-2022-2025-v1` benchmark documented on
`bspp-orchestration` main: 750 monomers, 150 dimers, 50 trimers and 50 tetramers,
with 391,180 residues across all modeled chains. It contains identifiers and
small sequence/metadata pins, not reference coordinate files, MSAs or model
weights. Downloading its reference data from RCSB requires no storage credentials.

Files:

- `pdb-ids.txt`: 1,000 unique PDB entry IDs, one per line.
- `benchmark-targets.csv`: target/PDB/assembly IDs, public download URLs, chain
  lengths, strata and expected sequence/source/reference hashes.
- `assembly-urls.txt`: the corresponding 1,000 public mmCIF assembly URLs.
- `reconstruction-targets.jsonl`: ordered chain sequences and metadata required
  by the existing `bsppctl prepare-benchmark --reconstruct` command.
- `benchmark-spec.json`: the benchmark selection specification from main.
- `benchmark-provenance.json`: cohort counts and the expected dataset fingerprint.
- `verification.json`: checks performed on this export, including the scope of
  live public-download verification.
- `full-reconstruction-verification.json`: the completed 1,000-target wrapper
  execution and downstream format-validation results.
- `SHA256SUMS`: checksums of this bundle's files.

All targets select biological assembly 1. PDB IDs identify structure entries,
not UniProt accessions. An entry ID alone does not capture assembly expansion,
chain ordering or the exact sequence construct used for benchmarking. The
companion reconstruction file preserves those inputs. Repeated chains remain
repeated, so a homodimer is not accidentally turned into a monomer. The source
corpus's A/B/C/D chain labels are synthetic; this export does not present them
as native RCSB chain identifiers.

## Download the public structures

From this directory (`docs/benchmarks/pdb-temporal-2022-2025-v1`):

```bash
sha256sum -c SHA256SUMS
wget --input-file=assembly-urls.txt --directory-prefix=assemblies
```

These downloads are the original public assembly mmCIF files. They are not yet
the normalized C-alpha references and validation suites expected by BSPP.

## Reconstruct the benchmark with the existing wrapper

From the root of an installed orchestration checkout, replace the output path
below and use a new output directory outside the Git worktree:

```bash
uv run --isolated --with pyarrow bsppctl prepare-benchmark \
  --spec docs/benchmarks/pdb-temporal-2022-2025-v1/benchmark-spec.json \
  --reconstruct docs/benchmarks/pdb-temporal-2022-2025-v1/reconstruction-targets.jsonl \
  --output /path/to/new-benchmark-corpus \
  --workers 4
```

This command downloads public RCSB assemblies and entry metadata itself; the
separate `wget` step is optional and is not a cache for that command. No
object-store S3 credentials are involved. The wrapper produces target FASTA,
JSONL/Parquet metadata, normalized references, subsets and validation suites.
MSAs are generated in the subsequent preprocessing workflow.

The current preprocessing parser requires a `.fa` filename. Before using the
generated FASTA as preprocessing input, make a byte-identical copy outside the
generated corpus directory, leaving its checked manifest and files untouched:

```bash
cp /path/to/new-benchmark-corpus/targets.fasta /path/to/benchmark-input.fa
```

Use `benchmark-input.fa` for preprocessing. This filename adaptation does not
change target IDs, sequences or complex composition.

The export uses `total_length`, as required by the reconstruction CLI. The
published materialized corpus calls the same quantity `total_residues`.

Check the resulting dataset fingerprint before treating it as the same release:

```bash
python3 - /path/to/new-benchmark-corpus/dataset.json <<'PY'
import json, sys
expected = 'c04ec62e6eecf165eea010f82f7aa1ad72ddfd99ce462a9c4afeac286049a217'
with open(sys.argv[1]) as stream:
    actual = json.load(stream)['dataset_fingerprint']
if actual != expected:
    raise SystemExit(f'Corpus identity changed: expected {expected}, got {actual}')
print('Published benchmark fingerprint matches.')
PY
```

Public source records can be revised. The existing reconstruction code checks
the pinned sequences but recomputes source/reference hashes from the downloads;
it does not enforce expected hashes supplied in an input record. The fingerprint
check above detects changes relevant to the benchmark identity. Expected hashes
in the CSV support per-target diagnosis. Do not silently accept a changed corpus
as the historical benchmark.

## Full download and format validation

On 2026-09-21 UTC, all 1,000 targets were reconstructed from the public RCSB
endpoints through `bsppctl prepare-benchmark --reconstruct --workers 4`, using
source commit (pinned upstream revision) and these exact
specification and reconstruction-input bytes. The command finished
successfully in 105.8 seconds. The existing environment already had PyArrow, so
the installed `bsppctl` executable was invoked directly; `uv` environment
provisioning was not retested.

The Runtime recomputed and matched the original published dataset fingerprint
and checked every one of the 1,025 checksum-listed files, with complete manifest
coverage. All 1,000 target records and reference structures matched the original
corpus. The real preprocessing parser accepted all 1,000 sequences after the
`.fa` filename adaptation. Parquet schema/rows matched; all seven validation
suites loaded; all six nested throughput subsets passed ordering and membership
checks. Of 1,026 corpus files, 1,024 were byte-identical: only `dataset.json`'s
generation timestamp and the corresponding checksum manifest differed.

This records the completed public download and benchmark-input preparation
check. The `source_main_commit` in `benchmark-provenance.json` identifies the
upstream main revision inspected during export (a pinned upstream revision), rather than the
source revision that executed reconstruction. Both verification JSON files
retain their original historical contents: `verification.json` records the
initial four representative samples, and `full-reconstruction-verification.json`
records the subsequent complete reconstruction.

These public inputs are included on this branch. Generated coordinate files,
MSAs and model weights remain external. The historical scientific benchmark
fixes were deployed through a pinned upstream revision;
the reference integration combines those fixes with newer main changes and
is assessed separately by local checks. This input bundle does not claim that
newly integrated container images, GPU folding or MSA search were rerun.

Public sources and terms:

- [RCSB file download services](https://www.rcsb.org/docs/programmatic-access/file-download-services)
- [RCSB usage policy](https://www.rcsb.org/pages/usage-policy)
- `DATASET-LICENSE.md`: the release corpus's provenance and CC0 attribution note.
