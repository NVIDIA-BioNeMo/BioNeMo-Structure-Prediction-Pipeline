# Folding BioIR Kernel Image

GPU scientific image for the `fold` action with the bioir backend. Built from
the public Docker Hub base `nvidia/cuda:13.0.3-devel-ubuntu24.04` (no NGC, no
entitlement, anonymous pull). Installs torch via
`pip install torch==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130`
(mirrors `containers/variants/postprocessing.env`), then `pip install bionemo-ir==0.1.0`
from public PyPI.

Carries `bionemo_ir`, Contract+Control+Runtime wheels (via `--no-deps`), and the CUDA
compat entrypoint.

Runtime binds the explicit checkpoint during both session construction and
processor calls, since BioIR can load the model lazily on the first call. The
scoped binding restores the prior environment on return or failure. The image
installs Python development headers at the installed interpreter's exact package
version: Triton needs them to compile extensions during model construction.
The local image smoke checks composition, imports, and compilation/import of a
small Python C extension against those headers; it does not initialize CUDA or
load a checkpoint. Model construction can itself compile and launch dummy GPU
kernels before any target or model forward pass.
Basic CUDA readiness and the lazy-binding regression likewise do not establish
checkpoint tensor compatibility or successful inference.

Mixed monomer/complex runs opt in through the Phase Plan's `bioir_model_policy`
(per the documented contract). One expanded chain selects BioIR's native `openfold2_ptm_1` model;
multiple chains select the existing `alphafold2_multimer_1` model, including
homomers represented by repeated chain IDs. The policy records both checkpoint
SHA-256 values and sizes; profile paths remain separately mounted read-only.
Each worker verifies selected checkpoint bytes before its first load and lazily
reuses at most one session per model. Native OpenFold `finetuning_ptm_1.pt` uses
`OPENFOLD2_PTM_1_CKPT`; no AlphaFold conversion or weight relabeling is involved.
This policy changes monomer model science and requires a fresh Phase run. Legacy
plans without it preserve multimer-only execution. Canonical provenance reports
OpenFold-pTM for monomers and AlphaFold-Multimer for complexes, with genuine full
PAE and unchanged finite-score checks; undefined monomer ipTM remains `null`.

## Build, smoke, push, import

```bash
containers/scripts/build.sh folding bioir
containers/folding/bioir/smoke-local.sh bspp-orchestration:folding-bioir
containers/scripts/push.sh folding bioir
sbatch containers/scripts/pull-sqsh.sh folding-bioir
```

The base-image digest is pinned in `image-lock.json` and verified at build time.
