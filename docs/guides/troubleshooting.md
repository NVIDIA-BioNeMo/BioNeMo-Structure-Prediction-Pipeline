# Troubleshooting

Start with the exact failed command, its exit code, the selected profile, and
the original logs. A queued job, a completed scheduler allocation, an accepted
Phase receipt, and a reference-validation result are different milestones.
Preserve their evidence separately.

## Installation and benchmark preparation

| Symptom | Check and supported action |
| --- | --- |
| `bsppctl` is missing | Use Python 3.12 and install the Control package as shown in [Installation](installation.md). Run `.venv/bin/bsppctl --help` to check the installed executable without a new sync. |
| uv refuses a stale lock | `--locked` checks freshness; `--frozen` consumes the shipped lock without checking project-metadata agreement. Verify the intended checkout and dependency declarations. Use frozen mode only to replay a reviewed lock; do not silently regenerate a benchmark deployment's lock. |
| `prepare-benchmark` needs PyArrow | Use the documented `uv run --frozen --package bspp-orchestration-control --no-dev --with pyarrow bsppctl prepare-benchmark ...` command. PyArrow is optional for ordinary Control use. |
| The input FASTA is rejected | The preprocessing input parser requires a `.fa` filename. Make the byte-identical copy described in the [public corpus guide](../benchmarks/pdb-temporal-2022-2025-v1/README.md); do not edit the checked corpus's sequences or manifest. |
| Corpus fingerprint differs | Preserve the downloaded files and compare them with the release manifest. Public upstream records can change; a new fingerprint must not be reported as the old benchmark. |

## Profile and container startup

An unknown field, incomplete cache group, or mixed scalar/packed GPU request is
a schema error. Compare with the
[canonical profile template](../../skills/examples/run-plans/cluster-profile.yaml).
YAML values are literal: `$VARIABLE` does not expand inside a path. Cluster
paths must exist on allocated nodes, not only on the operator's machine.

For `exec format error`, compare the image architecture with the allocated
node's architecture. Supplied images are Linux x86-64. Do not treat emulator
startup or successful image import as native GPU readiness.

For CUDA initialization or driver errors, retain the exact image, host driver,
visible devices, and container stderr from a scheduled allocation. A successful
host `nvidia-smi` does not prove that the container loads a compatible CUDA driver
library. The shipped entrypoint and compatibility libraries are part of image
identity; use a supported, tested image/driver combination and revalidate any
image change. Avoid ad hoc library replacement under an existing image pin.

## Registry and Enroot import failures

| Symptom | Check and supported action |
| --- | --- |
| Anonymous access or registry `401`/`403` | Check the actual Enroot import hostname against the external credential configuration. Push authentication and import authentication can use different clients or endpoints. Keep credentials out of logs and profiles. |
| Whiteout creation or extraction denied | Check the actual import temporary/cache filesystem. Use a site-supported job-local filesystem with adequate capacity; changing the final SquashFS destination alone does not move extraction. |
| Import is killed or reports OOM | Inspect scheduler state and memory allocation. Increase the CPU import job's memory based on that failure and use a fresh output filename. An import failure does not establish a scientific image failure. |
| Image exists but fails a manifest/hash check | Retain the rejected file and import records. Reconcile source, build, registry, and SquashFS identities instead of substituting a tag or bypassing the check. |

The [import script](../../containers/scripts/pull-sqsh.sh) deletes its chosen
output before importing. Never reuse a path pinned by active or historical
execution. See [Containers](containers.md#import-on-the-cluster).

## Database placement and shared memory

For direct database access failures, inspect the selected manifest and actual
mount view. Direct and staged placement enforce different filesystem and
ownership checks. If staged access is appropriate, author the complete cache
configuration and qualify it; do not rewrite manifests or bypass the guard.

A missing GPU-server shared-memory name does not by itself prove that the
server exited or identify another process as the cause. Retain both server and
client logs, the selected database paths, device visibility, and container mount
view. Site isolation can be configured with existing `extra_mounts`, but each
source must exist when the job starts. A separate job's `/dev/shm` directory may
not survive to the next allocation. Any private backing directory needs real
ownership, isolation, native server/client, and cross-job lifetime checks before
production use. Do not remove unrelated shared-memory files.

## Pending jobs and resource shape

Inspect the scheduler's pending reason, partition/account limits, per-node CPU
and memory demand, GPU availability, and time limit. Packed folding requests
CPU resources per task and memory per node. Its array concurrency limit is not
a reservation for two simultaneous nodes. Scheduler forecasts are advisory;
report parallelism from actual allocation overlap.

Use a fresh immutable operational selection when changing a profile. Preserve
the previous attempt and confirm its jobs are terminal before starting
replacement work. Lowering a Slurm CPU allocation does not necessarily change
the scientific command's thread count; inspect the rendered action rather than
assuming those values are linked.

## Dependent submission and slow transport

An `afterok` dependency can be rejected if its predecessor has already completed
and aged out of Slurm's active job records before a later submission arrives.
Distinguish this from a predecessor failure using the original submission
timestamps, assignments, accounting, and the site's retention policy. Long SSH
round trips can expose this boundary even when the computation succeeds.

The supported alternative is native Control with `transport: local-slurm` and
no `ssh_target`. Install Control from the same clean checkout on a site host,
then materialize using that profile. All scientific work still runs through
Slurm. See [Cluster configuration](cluster-configuration.md#select-the-profile-and-transport).

For a partial submission, retain all known assignments and diagnostics before
recovery. Use public reconciliation and cancellation where applicable; do not
manually remove dependencies, edit authority, or blindly repeat `sbatch`. If an
uncertain dispatch cannot be reconciled, establish which jobs actually exist
before launching separate replacement work.

## Evidence, cancellation, and retry

These public commands inspect or reconcile the selected Phase:

```bash
bsppctl phase status PHASE_RUN_ID --authority-root /path/to/authority --format json
bsppctl phase resume PHASE_RUN_ID --authority-root /path/to/authority
```

For postprocessing, `bsppctl phase diagnostics PHASE_RUN_ID --authority-root
/path/to/authority --diagnostics-root /path/to/diagnostics` also collects
non-authoritative diagnostics. This command does not support preprocessing or
folding.

`resume` performs one reconciliation pass. `cancel` must confirm termination
through accounting; an issued cancellation request alone is not closure.
`retry` creates a successor attempt using validated carry-forward evidence.
For folding, missing or invalid rank journals can prevent retry, including when
the fold never started. Preserve the refusal and original authority; do not
manufacture empty journals or use an unsupported force option.

Fetch evidence into a new destination and finalize only with the real
family-specific scheduler and action/handoff records. For detailed accounting,
retain terminal `scontrol` records promptly, before the site purges them, as
well as UTC `sacct --duplicates` records. Preserve array-element identities and
restart information. Missing restart history cannot be repaired by assuming
zero restarts from the final exit code.

## Reference validation and local corpora

The shipped `bsppctl validate-run` requires `--corpus s3://bucket/prefix` and the
expected `--fingerprint`. Its `--run-dir`, `--suite`, `--index`, and output paths
must be accessible to the scheduled validation job. Configure object-store
access there; successful local corpus reconstruction does not provide that
access automatically.

There is no shipped local-directory corpus option for this command. An accepted
folding receipt establishes the workflow's evidence contract, not agreement
with reference structures. Keep reference-validation status explicit until the
actual validation job and its result are available.
