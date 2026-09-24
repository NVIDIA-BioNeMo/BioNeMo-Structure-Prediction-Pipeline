#!/usr/bin/env bash
# BSPP folding Phase Runtime Action
# Phase Run: phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# Attempt: attempt-0001
# RunSpec digest: 7d375f04476ec43b96b6e12edd1ed6eb1e8fb1be4261b1736920b8e6c40295e3
#SBATCH --job-name=bspp-phase-ab2ebd4fe710b23c0535ed8428e87b417b3e9f4ce77340101d9c6b340f9a4308
#SBATCH --comment=bspp-phase-ab2ebd4fe710b23c0535ed8428e87b417b3e9f4ce77340101d9c6b340f9a4308
#SBATCH --partition=cpu
#SBATCH --account=bspp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/bspp/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001/actions/msa-flatten-000001/slurm-%j.out
#SBATCH --error=/lustre/bspp/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001/actions/msa-flatten-000001/slurm-%j.err

set -euo pipefail
PHASE_RUNSPEC=/lustre/bspp/staging/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001/phase-runspec.json
EXPECTED_RUNSPEC_DOCUMENT_SHA256=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
ACTION_EVIDENCE=/lustre/bspp/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001/actions/msa-flatten-000001/action-evidence.json
HANDOFF=/lustre/bspp/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001/actions/msa-flatten-000001/handoff.json
mkdir -p -- /lustre/bspp/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001/actions/msa-flatten-000001
[[ -f "$PHASE_RUNSPEC" ]] || { echo "missing staged Phase RunSpec: $PHASE_RUNSPEC" >&2; exit 127; }
_runspec_sha256="$(sha256sum "$PHASE_RUNSPEC" | awk '{print $1}')"
[[ "$_runspec_sha256" == "$EXPECTED_RUNSPEC_DOCUMENT_SHA256" ]] || { echo "Phase RunSpec document SHA-256 mismatch" >&2; exit 127; }
srun --container-image=registry/bspp-runtime:latest --container-mounts=/lustre/bspp/staging/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001/phase-runspec.json:/lustre/bspp/staging/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001/phase-runspec.json:ro,/lustre/bspp/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001:/lustre/bspp/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001,/bspp-fixture/msa-set/msa-set.tar:/bspp-fixture/msa-set/msa-set.tar:ro,/bspp-fixture/msa-set/msa-set.tar.lz4:/bspp-fixture/msa-set/msa-set.tar.lz4:ro --no-container-mount-home /usr/local/bin/entrypoint.sh python -m bspp.orchestration.runtime.folding.executor --phase-runspec /lustre/bspp/staging/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001/phase-runspec.json --action-id msa-flatten-000001 --action-evidence /lustre/bspp/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001/actions/msa-flatten-000001/action-evidence.json --handoff /lustre/bspp/bspp-phase-runs/phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempt-0001/actions/msa-flatten-000001/handoff.json
