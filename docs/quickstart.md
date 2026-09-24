# Quickstart: prepare the public BioIR benchmark

This walkthrough installs the Control CLI and reconstructs the repository's
1,000-target benchmark on your workstation. It uses public structure downloads
and does not submit a GPU job. Continue with the
[BioIR workflow](benchmarks/bioir-workflow.md) when the corpus is ready.

## 1. Install the Control CLI

Use a Linux environment with Python 3.12, uv and a checkout of this repository.
Run commands from the repository root:

```bash
uv sync --frozen --package bspp-orchestration-control --no-dev
uv run --frozen --package bspp-orchestration-control --no-dev bsppctl --help
```

Frozen mode preserves the shipped lock. It does not verify that the lock agrees
with the current dependency declarations. Read the
[installation guide](guides/installation.md) for freshness checks and the
distinction between Control and Runtime environments.

## 2. Verify the checked-in input bundle

The bundle contains public PDB identifiers, assembly URLs, sequence pins and
reconstruction metadata. It does not contain reference coordinate payloads,
MSAs, search databases or model weights.

```bash
(cd docs/benchmarks/pdb-temporal-2022-2025-v1 && sha256sum -c SHA256SUMS)
```

See the [dataset guide](benchmarks/pdb-temporal-2022-2025-v1/README.md) for exact
chain/assembly semantics and provenance.

## 3. Reconstruct the corpus

Choose a new output directory outside the checkout. Replace the example path
before running this command; its destination must not already contain a corpus.
PyArrow is added to this command's uv environment for Parquet output.

```bash
export BSPP_BENCHMARK_ROOT=/path/to/new-benchmark-workspace
mkdir -p "$BSPP_BENCHMARK_ROOT"

uv run --frozen --package bspp-orchestration-control --no-dev --with pyarrow \
  bsppctl prepare-benchmark \
  --spec docs/benchmarks/pdb-temporal-2022-2025-v1/benchmark-spec.json \
  --reconstruct docs/benchmarks/pdb-temporal-2022-2025-v1/reconstruction-targets.jsonl \
  --output "$BSPP_BENCHMARK_ROOT/corpus" \
  --workers 4
```

This downloads public RCSB assembly structures and entry metadata. Object-store
credentials are not required for reconstruction. Public records can change;
the following checks establish whether the result matches the pinned release.

## 4. Verify identity and prepare the FASTA filename

```bash
python3 - "$BSPP_BENCHMARK_ROOT/corpus/dataset.json" <<'PY'
import json
import sys

expected = "c04ec62e6eecf165eea010f82f7aa1ad72ddfd99ce462a9c4afeac286049a217"
with open(sys.argv[1]) as stream:
    actual = json.load(stream)["dataset_fingerprint"]
if actual != expected:
    raise SystemExit(f"Corpus identity changed: expected {expected}, got {actual}")
print("Published benchmark fingerprint matches.")
PY

(cd "$BSPP_BENCHMARK_ROOT/corpus" && sha256sum -c SHA256SUMS)
cp "$BSPP_BENCHMARK_ROOT/corpus/targets.fasta" "$BSPP_BENCHMARK_ROOT/benchmark-input.fa"
cmp "$BSPP_BENCHMARK_ROOT/corpus/targets.fasta" "$BSPP_BENCHMARK_ROOT/benchmark-input.fa"
```

The preprocessing parser requires `.fa`. This byte-identical copy preserves
the checksummed corpus and every target's chain composition.

Successful reconstruction produces FASTA, JSONL/Parquet metadata, normalized
references, validation suites and throughput subsets. The full cohort contains
750 monomers, 150 dimers, 50 trimers and 50 tetramers: 391,180 residues across
all modeled chains.

## 5. Continue to scheduled execution

The next steps require a configured cluster, prepared images, model checkpoints
and search databases:

1. Follow [cluster configuration](guides/cluster-configuration.md) and [container preparation](guides/containers.md).
2. Use the [BioIR workflow](benchmarks/bioir-workflow.md) to qualify preprocessing, generate fresh MSAs and execute folding.
3. Follow [benchmark methodology](benchmarks/methodology.md) when interpreting elapsed time and throughput.

The checked-in Plans are illustrative templates, not runnable jobs. The current
release requires operator preparation of several verified handoff/evidence
artifacts. Its final `bsppctl validate-run` command also requires an S3 corpus
location. These boundaries are documented in the walkthrough and
[implementation status](status.md).
