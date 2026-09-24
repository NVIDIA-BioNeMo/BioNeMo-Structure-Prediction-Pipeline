#!/bin/bash
# Generated artifact: slurm/run_analysis_finalize.sbatch
# Run ID: run
# Source RunSpec: <TMP>/runspec.yaml
# Source SHA256: <SOURCE_SHA256>
# Rendered at: <RENDERED_AT>
# DO NOT EDIT: regenerate this file from the RunSpec.
#SBATCH --job-name=bspp_ds1_analysis_finalize
#SBATCH --partition=cpu_long
#SBATCH --account=acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=<TMP>/output/logs/analysis_metadata_finalize_%j.out
#SBATCH --error=<TMP>/output/logs/analysis_metadata_finalize_%j.err

set -euo pipefail

echo "BSPP analysis finalizer: ${SLURM_JOB_ID:-local}"
srun \
  --container-image=image.sqsh \
  --container-mounts=/lustre:/lustre,<TMP>/AFDB-Integration-Kit:/workspace/afdb-toolkit:ro,<TMP>/bspp-orchestration:/workspace/bspp-orchestration \
  --no-container-mount-home \
  env \
  BSPP_ORCHESTRATION_DEV_MOUNT=1 \
  /usr/local/bin/entrypoint.sh \
  bash <<'BSPP_FINALIZE_IN_CONTAINER'
set -euo pipefail

ORCHESTRATION_ROOT=/workspace/bspp-orchestration
CONTAINER_WORKDIR=/workspace/bspp-orchestration
S5CMD_PATH=/bin/s5cmd
PYTHON_BIN="${BSPP_ORCHESTRATION_PYTHON:-/opt/bspp-orchestration-env/.pixi/envs/default/bin/python}"
fail() { echo "BSPP finalizer preflight failed: $*" >&2; exit 127; }
require_dir() { [[ -d "$1" ]] || fail "missing $2: $1"; }
require_file() { [[ -f "$1" ]] || fail "missing $2: $1"; }
require_executable() { [[ -x "$1" ]] || fail "missing executable $2: $1"; }

require_dir "$ORCHESTRATION_ROOT" "orchestration mount"
require_dir "$CONTAINER_WORKDIR" "container workdir"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
require_executable "$PYTHON_BIN" "Python runtime"
export PYTHONPATH="${ORCHESTRATION_ROOT}/packages/orchestration-contract/src:${ORCHESTRATION_ROOT}/packages/orchestration-runtime/src${PYTHONPATH:+:${PYTHONPATH}}"
"$PYTHON_BIN" -c "import bspp.orchestration.runtime.postprocessing.analysis_finalizer" >/dev/null
cd "$CONTAINER_WORKDIR"

if [[ "$S5CMD_PATH" == */* ]]; then
  require_executable "$S5CMD_PATH" "configured s5cmd path"
else
  command -v "$S5CMD_PATH" >/dev/null 2>&1 || fail "missing configured s5cmd on PATH: $S5CMD_PATH"
fi

"$PYTHON_BIN" \
  -m \
  bspp.orchestration.runtime.postprocessing.analysis_finalizer \
  --csv \
  <TMP>/output/analysis_metadata.csv \
  --parquet \
  <TMP>/output/analysis_metadata.parquet \
  --selected-ids \
  <TMP>/output/high_quality_model_ids.txt \
  --local-tars-csv \
  <TMP>/output/uploaded_tars.csv \
  --model-tar-index \
  <TMP>/output/model_tar_index.csv \
  --high-quality-work-dir \
  <TMP>/hq-work \
  --high-quality-s3-prefix \
  s3://example-bucket/users/test/postprocessed/ds1_hq/ \
  --s5cmd-path \
  /bin/s5cmd \
  --s5cmd-numworkers \
  32

echo "BSPP high-quality extraction/upload completed for s3://example-bucket/users/test/postprocessed/ds1_hq/"
BSPP_FINALIZE_IN_CONTAINER
