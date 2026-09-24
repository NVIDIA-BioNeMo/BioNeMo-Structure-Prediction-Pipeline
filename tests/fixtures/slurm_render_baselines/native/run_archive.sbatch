#!/bin/bash
# Generated artifact: slurm/run_archive.sbatch
# Run ID: run
# Source RunSpec: <TMP>/runspec.yaml
# Source SHA256: <SOURCE_SHA256>
# Rendered at: <RENDERED_AT>
# DO NOT EDIT: regenerate this file from the RunSpec.
#SBATCH --job-name=bspp_ds1_native
#SBATCH --partition=gpu
#SBATCH --account=acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=<TMP>/output/logs/native_%A_%a.out
#SBATCH --error=<TMP>/output/logs/native_%A_%a.err
#SBATCH --gres=gpu:1

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DUCKDB_MEMORY_LIMIT=1GB

echo "BSPP native archive worker: ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
srun \
  --container-image=image.sqsh \
  --container-mounts=/lustre:/lustre,<TMP>/AFDB-Integration-Kit:/workspace/afdb-toolkit:ro,<TMP>/bspp-orchestration:/workspace/bspp-orchestration,<TMP>/runspec.yaml:/workspace/bspp-runspec/runspec.yaml:ro \
  --no-container-mount-home \
  env \
  BSPP_ORCHESTRATION_DEV_MOUNT=1 \
  /usr/local/bin/entrypoint.sh \
  bash <<'BSPP_NATIVE_IN_CONTAINER'
set -euo pipefail

TOOLKIT_ROOT=/workspace/afdb-toolkit
ORCHESTRATION_ROOT=/workspace/bspp-orchestration
CONTAINER_WORKDIR=/workspace/bspp-orchestration
RUNSPEC=/workspace/bspp-runspec/runspec.yaml
PRODUCTION_PIPELINE=/workspace/afdb-toolkit/scripts/production_pipeline.py
S5CMD_PATH=/bin/s5cmd
PYTHON_BIN="${BSPP_ORCHESTRATION_PYTHON:-/opt/bspp-orchestration-env/.pixi/envs/default/bin/python}"
fail() { echo "BSPP native worker preflight failed: $*" >&2; exit 127; }
require_dir() { [[ -d "$1" ]] || fail "missing $2: $1"; }
require_file() { [[ -f "$1" ]] || fail "missing $2: $1"; }
require_executable() { [[ -x "$1" ]] || fail "missing executable $2: $1"; }

require_dir "$TOOLKIT_ROOT" "toolkit mount"
require_dir "$ORCHESTRATION_ROOT" "orchestration mount"
require_dir "$CONTAINER_WORKDIR" "container workdir"
require_file "$RUNSPEC" "RunSpec"
require_file "$PRODUCTION_PIPELINE" "production_pipeline.py"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
require_executable "$PYTHON_BIN" "Python runtime"
command -v tar >/dev/null 2>&1 || fail "missing tar on PATH"
command -v lz4 >/dev/null 2>&1 || fail "missing lz4 on PATH"
command -v zstd >/dev/null 2>&1 || fail "missing zstd on PATH"
if [[ "$S5CMD_PATH" == */* ]]; then
  require_executable "$S5CMD_PATH" "configured s5cmd path"
else
  command -v "$S5CMD_PATH" >/dev/null 2>&1 || fail "missing configured s5cmd on PATH: $S5CMD_PATH"
fi
export PYTHONPATH="${TOOLKIT_ROOT}:${ORCHESTRATION_ROOT}/packages/orchestration-contract/src:${ORCHESTRATION_ROOT}/packages/orchestration-runtime/src${PYTHONPATH:+:${PYTHONPATH}}"
"$PYTHON_BIN" -c "from bspp.orchestration.runtime.cli import cli; cli()" --help >/dev/null
cd "$CONTAINER_WORKDIR"

"$PYTHON_BIN" \
  -c \
  'from bspp.orchestration.runtime.cli import cli; cli()' \
  worker \
  archive-task \
  --runspec \
  "$RUNSPEC"
BSPP_NATIVE_IN_CONTAINER
