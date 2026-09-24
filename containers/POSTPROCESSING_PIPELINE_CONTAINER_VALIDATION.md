# Postprocessing Pipeline Container Validation

This file records intermediate pilots proving that the postprocessing pipeline
can run correctly from the `bspp-orchestration` container. It is not the final
production validation record for the full dataset.

`SM_ACCEPTANCE.md` is the GPU/container smoke matrix. This log is for
containerized postprocessing runtime, output counts, and result parity.

## What To Record

For each pilot, keep the record brief but complete:

- RunSpec path, orchestration commit, toolkit commit, container variant, and
  sqsh path or hash.
- Dataset, archive array, `batch_size`, `shards_per_archive`, workers, output
  directory, and SwiftStack output prefix.
- SLURM job id, node, GPU, driver, final state, and elapsed time.
- Pipeline totals from `pipeline_results.json`: total runtime, Stage 13 runtime,
  models processed, and failed models.
- Native worker totals from stderr: extraction time, upload file count, upload time,
  and batch completion time.
- Validation outputs: expected object counts, actual object counts, and any
  parity diffs against the selected baseline.

Do not record transient blockers that were fixed permanently in code unless
they affect interpretation of a recorded run.

## Planned Pilot: Example cluster A100 Latest Container Batch 1000

Status: running/timing evidence collected for batch 1; strict byte parity needs
a fresh run after regenerating `provider.json` with the legacy copyright text.

| Field | Value |
|-------|-------|
| Purpose | First end-to-end postprocessing timing and output check from the `latest` pixi/PyG container |
| RunSpec | `/cluster/projects/example-account/users/example-user/bspp/bspp-proj/output/20251128_full_run_v2_orch_bridge_attempt_20260505_latest_batch1000/runspec.yaml` |
| Run ID | `20251128_full_run_v2_orch_bridge_attempt_20260505_latest_batch1000` |
| Dataset | `20251128_full_run_v2` |
| Archive array | `0-0` |
| Container | `latest`, `/cluster/projects/example-account/users/example-user/bspp/containers/bspp-orchestration-latest.sqsh` |
| Worker shape | 24 workers, `batch_size=1000`, `shards_per_archive=2`, self-upload enabled |
| Output dir | `/cluster/projects/example-account/users/example-user/bspp/bspp-proj/output/20251128_full_run_v2_orch_bridge_attempt_20260505_latest_batch1000` |
| SwiftStack prefix | `s3://example-bucket/users/example-user/postprocessed-test/20251128_full_run_v2_orch_bridge_attempt_20260505_latest_batch1000/` |
| Preflight | passed, no blockers |
| Submit command | Historical bridge pilot; bridge submission wrapper has since been removed |

### Results

Fill after completion:

| Field | Value |
|-------|-------|
| Job id | `9594338_0` |
| Node / GPU / driver | `example-node`, A100-SXM4-80GB, driver TBD |
| SLURM final state | running at last observation |
| SLURM elapsed | `00:10:04` at last observation |
| Archive extraction time | 67.7s |
| Batch count | `shard_0` has 2 batches; batch 1 reran cleanly, batch 2 was skipped from a stale marker |
| Total pipeline runtime per batch | batch 1: 91.77s for 1000 models |
| Stage 13 runtime per batch | batch 1: 10.09s for 1000 models |
| Upload file count / time | batch 1: 6983 files in 56.9s |
| Final validation report | TBD |
| Result parity vs baseline | 25-object sample had 6 byte mismatches, all explained by PDB copyright year; after normalizing only that year, mismatches were 0 |

### Notes

- The acceptance gate is native CUDA `torch_cluster.radius_graph`; the smoke
  test already passed for `latest` on Example cluster A100.
- Compare against `current` only after this `latest` run is complete and its
  output is structurally valid.
- The byte mismatch source was generated `provider.json`. The actual
  `POSTPROCESSING_PIPELINE.md` baseline at
  `s3://example-bucket/users/example-user/postprocessed-test/20251128_full_run_v2/`
  carries `Copyright 2024 NVIDIA. All rights reserved.`, so byte-parity
  RunSpecs must preserve that value.
- Upload is currently on the GPU allocation critical path. A stalled pilot
  showed `s5cmd --numworkers 256` plateauing at 6969/6983 batch objects while
  holding 5.9G and 8998 files in `/dev/shm`; subsequent byte-parity attempts
  should use lower `s5cmd_numworkers` before any larger run.
