# CUDA SM Acceptance Matrix

This project treats `torch-cluster` as the GPU performance gate. A variant is
accepted on a GPU architecture only after the container smoke test proves that
PyTorch was compiled for the visible SM and `torch_cluster.radius_graph` runs
on CUDA.

## Acceptance Command

Run inside the container on the target GPU node:

```bash
bspp-container-smoke-gpu /tmp/bspp-smoke-${IMAGE_TAG:-unknown}.json
```

Required pass signals:

```text
device_sm_supported_by_torch True
device_ptx_supported_by_torch <informational>
radius_graph_cuda_ok ...
nvcomp_zstd_raw_encode_ok ...
nvcomp_zstd_cli_decode_ok True
afdb_radius_graph_fallback False
```

If `device_sm_supported_by_torch` is false, the image is not valid for that
GPU, regardless of whether CUDA is visible. PTX support is recorded for
diagnostics only; production acceptance requires exact `sm_XX` support in
`torch.cuda.get_arch_list()`.

On SLURM, use the wrapper so results are stored with stable names:

```bash
IMAGE_TAG=postprocessing sbatch --partition=<partition> containers/scripts/slurm-smoke-gpu.sh
```

## Matrix To Fill

| Variant | Torch | CUDA | Cluster | Node/Partition | GPU | SM | Driver | Compat Mode | Image Digest / sqsh Hash | Orchestration Commit | Toolkit Commit | Smoke JSON | Smoke | Stage 13 Timing | Result Parity | Date |
|---------|-------|------|---------|----------------|-----|----|--------|-------------|--------------------------|----------------------|---------------|------------|-------|-----------------|---------------|------|
| current | 2.5.1+cu121 | 12.1 | Example cluster | gpu-partition / example-node | A100-SXM4-80GB | sm_80 | 535.104.12 | off | TBD | a619ae0 | TBD | containers/smoke-results/current-9584878-0.json | passed | TBD | TBD | 2026-05-05 |
| latest | 2.11.0+cu130 | 13.0 | Example cluster | gpu-partition / example-node | A100-SXM4-80GB | sm_80 | 535.104.12 | off | TBD | a619ae0 | TBD | containers/smoke-results/latest-9584436-0.json | passed | TBD | TBD | 2026-05-05 |
| latest | 2.11.0+cu130 | 13.0 | Example cluster | gpu-partition / TBD | A100-SXM4-80GB | sm_80 | 535.104.12 | off | `sha256:2d8b3a920cc341bf8695854ea7695278a8631d9d8f89b5caacf4ee5a0e29add3` | a21a024 | c8f824d | containers/smoke-results/latest-12911926-0.json | passed | completed (timing TBD) | passed; 3 pre-accepted residual models (canonical-provenance format deviations) | 2026-09-02 |
| latest | 2.11.0+cu130 | 13.0 | newer cluster | TBD | H100 | sm_90 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| latest | 2.11.0+cu130 | 13.0 | newer cluster | TBD | B100/B200 | sm_100 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| latest | 2.11.0+cu130 | 13.0 | newer cluster | TBD | RTX 50-series | sm_120 | TBD | TBD | TBD | TBD | TBD | TBD | informational | N/A | N/A | TBD |

### Compat Mode column semantics

The `Compat Mode` column records the `BSPP_CUDA_COMPAT` entrypoint toggle at
smoke time. It does **not** mean CUDA forward-compatibility libraries were
unused: Example cluster's host driver is 535.104.12 (native CUDA ≤ 12.2), so CUDA 13.x
variants execute on the image-shipped `cuda-compat-13-0` user-space libraries,
which the Pyxis/Enroot container runtime resolves independently of the toggle.
NVIDIA's CUDA Application Compatibility Support Matrix lists `cuda-compat-13-x`
(13.0–13.3) as compatible with driver branch 535+ on Data Center GPUs, so this
configuration is vendor-supported. End-to-end proof beyond the smoke gate: the
`latest` variant completed the pinned reference baseline pipeline on Example cluster with byte parity
(the pinned reference outputs, validated 2026-09-02).

## Policy

- `current` is the conservative Example cluster baseline.
- `latest` is accepted per GPU architecture, not globally.
- A successful Docker build is not an acceptance signal.
- A variant is not accepted for zstd-member tar delivery unless
  `nvcomp_zstd_cli_decode_ok` is true.
- RTX 50-series builder machines can provide informational smoke data, but
  production acceptance comes from Example cluster and other target clusters.

## Current Checkpoint

As of 2026-05-05, both `current` and `latest` pass the Example cluster A100 smoke gate:

```text
status: passed
device_sm_supported_by_torch: true
radius_graph_cuda_ok: true
nvcomp_zstd_raw_encode_ok: {...}
nvcomp_zstd_cli_decode_ok: true
afdb_radius_graph_fallback: false
```

The build-time baked-toolkit qualification (iPSAE paired-pdb-pae semantic
check) serves as a
pre-publication gate: a build that passes has proven the iPSAE binary matches
the expected scientific contract. This does not replace GPU smoke or Stage 13
timing, but is a necessary condition baked into the image.

Provenance attestation is recorded at `/opt/afdb-toolkit/provenance.json`
and `/opt/afdb-toolkit/qualification-result.json` in every image.

2026-09-02 update: the `latest` Example cluster parity pilot is complete. The
reference pipeline reproduced the pinned reference baseline
with byte parity, modulo three pre-accepted
residual models (all canonical-provenance format deviations).
Remaining acceptance work for `latest`: record Stage 13 timing and extend the
matrix to sm_90 / sm_100 / sm_120 clusters.
