# BioNeMo Structure Prediction Pipeline

The BioNeMo Structure Prediction Pipeline is an open-source, GPU-accelerated
workflow that goes from protein sequence to predicted 3D structure at scale. It
builds multiple sequence alignments (MSAs), predicts structures for single
proteins and multi-chain complexes, and scores the confidence of each
prediction. Each phase runs as containerized jobs on a Slurm cluster with NVIDIA
GPUs.

NVIDIA and its collaborators used this pipeline to generate two datasets that
are openly available in the [AlphaFold Database](https://alphafold.ebi.ac.uk/):

- **Pandemic Preparedness / viral protein complex dataset:** predicted
  structures for the protein complexes of more than 2,800 viruses
  ([NVIDIA blog](https://blogs.nvidia.com/blog/open-protein-dataset/),
  [EMBL-EBI announcement](https://www.ebi.ac.uk/about/news/technology-and-innovation/alphafold-database-adds-viral-protein-complexes-to-support-pandemic-preparedness/),
  [Nature article](https://www.nature.com/articles/d41586-026-03022-1)).
- **Proteome-scale protein complexes:** about 1.8 million high-confidence
  complexes, selected from about 31 million predictions across 4,777 proteomes
  ([EMBL-EBI announcement](https://www.embl.org/news/science-technology/first-complexes-alphafold-database/),
  [manuscript](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/afdb.html)).

Researchers can run the same workflow on their own protein targets.

![Pipeline overview: the bsppctl command-line tool drives three containerized phases on a Slurm cluster.](docs/assets/workflow-overview.svg)

**Start here:** [Quickstart](docs/quickstart.md) · [Documentation](docs/index.md) ·
[Pipeline overview](docs/pipeline-overview.md) ·
[Architecture](docs/architecture/README.md) ·
[Release status and limitations](docs/status.md)

## How it works

| Phase | What it does | Built on |
| --- | --- | --- |
| Preprocessing | Searches sequence databases on the GPU to build an MSA for each target | [ColabFold](https://github.com/sokrypton/ColabFold) `colabfold_search` with [MMseqs2](https://github.com/soedinglab/MMseqs2) GPU search |
| Folding | Predicts 3D structures with per-residue confidence (pLDDT) and predicted aligned error (PAE) | [OpenFold2](https://github.com/aqlaboratory/openfold) with AlphaFold2-Multimer weights (optionally OpenFold2 pTM weights for single-chain targets), accelerated by [NVIDIA BioNeMo Inference Runtime](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime). The folding phase can also run standard [OpenFold2](https://github.com/aqlaboratory/openfold) or [ColabFold](https://github.com/sokrypton/ColabFold) (AlphaFold2); see [release status](docs/status.md) for what each backend supports |
| Postprocessing | Scores complex interfaces (ipSAE, pDockQ2), checks for steric clashes and exports ModelCIF and BinaryCIF files with Parquet manifests | [AFDB Integration Kit](https://github.com/PDBeurope/AFDB-Integration-Kit) |

`bsppctl`, the pipeline's command-line tool, runs on your workstation (over SSH)
or directly on a Slurm host and submits each phase as containerized Slurm jobs.
Each phase run is pinned to checksum-verified inputs and closed with a receipt
once its job records are verified. Phases hand off results through local
storage or S3-compatible object storage, so they can run on the same cluster or
on different ones. Folding can spread targets across multiple GPUs, balanced by
sequence length.

Phases are run and handed off individually rather than through a single
end-to-end command. See [release status and limitations](docs/status.md) for
supported configurations.

## Requirements

- A Slurm cluster with [Pyxis](https://github.com/NVIDIA/pyxis) and
  [Enroot](https://github.com/NVIDIA/enroot) whose nodes have x86-64 CPUs and
  NVIDIA GPUs with at least 80 GB of memory, such as the A100 80GB (Ampere) or
  H100 80GB (Hopper)
- Python 3.12, Git and [uv](https://docs.astral.sh/uv/) on the machine that runs `bsppctl`
- Docker, to build the container images
- Your input sequences (FASTA), MSA search databases, model weights and
  postprocessing reference data (UniProt mappings), which you supply
- Fast local storage, such as NVMe, is recommended for the MSA search
  databases; the UniRef30 search index alone is about 240 GB
- S3-compatible object storage for postprocessing and benchmark validation

## Install

From a checkout of this repository:

```bash
# Install the bsppctl command-line tool
uv sync --frozen --package bspp-orchestration-control --no-dev

# Confirm the install
uv run --frozen --package bspp-orchestration-control --no-dev bsppctl --help
```

`--frozen` installs from the shipped lock file without checking that the lock is
current; see [installation and lock handling](docs/guides/installation.md).

This installs only the lightweight control tool; scientific work runs inside the
container images below. Rebuilding the public benchmark also needs PyArrow (see
the [quickstart](docs/quickstart.md)), and [DEVELOPING.md](DEVELOPING.md) covers
the development setup.

## Public benchmark

The repository includes `pdb-temporal-2022-2025-v1`, a 1,000-target benchmark of
experimental structures released in the PDB from 2022 through 2025: 750
monomers, 150 dimers, 50 trimers and 50 tetramers. `bsppctl prepare-benchmark`
rebuilds it from public RCSB downloads with no credentials. The
[quickstart](docs/quickstart.md) covers this step, and the
[BioIR benchmark walkthrough](docs/benchmarks/bioir-workflow.md) continues
through MSA generation and folding.

## Concepts

- **Cluster Profile**: settings for one cluster, including connection, paths, container images, mounts, account and resource defaults.
- **Phase Plan**: what one phase should do, and which cluster it runs on.
- **Phase RunSpec**: the immutable record created from a Plan for a single run attempt.
- **Phase Receipt**: the sealed record of a successful attempt, issued after its outputs are verified.

The [phase lifecycle guide](docs/guides/phase-lifecycle.md) walks through each
step: materialize, submit, status, resume, cancel, retry and finalize.

## Agent skills and example run plans

The [skills/](skills/) directory contains agent skills that help AI coding
assistants configure runs, operate `bsppctl` and build containers. They use the
same example files as the documentation:

- [Cluster Profile](skills/examples/run-plans/cluster-profile.yaml)
- [Preprocessing Phase Plan](skills/examples/run-plans/preprocessing-phase-plan.yaml)
- [Folding Phase Plan](skills/examples/run-plans/folding-phase-plan.yaml)
- [Postprocessing Phase Plan](skills/examples/run-plans/postprocessing-phase-plan.yaml)

The examples use placeholder paths and identifiers. Replace them with your own
values before submitting a run; see the
[configuration reference](docs/reference/configuration.md).

## Containers

Each phase has its own container image; folding uses a shared runtime image
plus the image for your chosen backend. Build the images with Docker, push them to your registry (set
`BSPP_REGISTRY`) and import them on the cluster as SquashFS files for
Pyxis/Enroot. The [container guide](docs/guides/containers.md) covers each step.

## Commands

| Task | Command | Guide |
| --- | --- | --- |
| Run and manage a phase | `bsppctl phase …` | [Phase lifecycle](docs/guides/phase-lifecycle.md) |
| Publish phase outputs to object storage | `bsppctl phase publish-preprocessing` / `publish-folding` | [Phase lifecycle](docs/guides/phase-lifecycle.md) |
| Rebuild the public benchmark | `bsppctl prepare-benchmark` | [Quickstart](docs/quickstart.md) |
| Validate benchmark predictions | `bsppctl validate-run` | [BioIR walkthrough](docs/benchmarks/bioir-workflow.md) |
| Qualify preprocessing and postprocessing images on a cluster | `bsppctl runtime …` | [CLI reference](docs/reference/bsppctl.md) |

For problems during a run, see [troubleshooting](docs/guides/troubleshooting.md).

## Licensing

This repository is licensed under the Apache License 2.0 — see
[LICENSE](LICENSE). Third-party component licenses and notices are published
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The benchmark dataset under
[docs/benchmarks/pdb-temporal-2022-2025-v1](docs/benchmarks/pdb-temporal-2022-2025-v1/README.md)
is derived from RCSB Protein Data Bank records and is provided under the
CC0 1.0 Universal public-domain dedication; see its
[DATASET-LICENSE.md](docs/benchmarks/pdb-temporal-2022-2025-v1/DATASET-LICENSE.md).

## Contributing

This project is currently not accepting contributions.
