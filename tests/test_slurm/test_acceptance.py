# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for acceptance comparator SLURM rendering."""

from __future__ import annotations

import json
from pathlib import Path

from bspp.orchestration.runtime.slurm.acceptance import (
    render_acceptance_semantic,
    render_acceptance_tar_payload_parity,
)
from tests.runspec_workflow_helpers import load_workflow_runspec, workflow_runspec_data


def test_render_acceptance_tar_payload_parity_dry_run_writes_nothing(tmp_path: Path) -> None:
    spec = load_workflow_runspec(tmp_path)

    plan = render_acceptance_tar_payload_parity(spec, dry_run=True)

    assert spec.submission is not None
    assert plan.dry_run is True
    assert plan.script_path == spec.submission.evidence_dir / "acceptance" / "tar_payload_parity" / (
        "run_tar_payload_parity.sbatch"
    )
    assert plan.report_path == spec.submission.evidence_dir / "acceptance" / "tar_payload_parity" / (
        "tar_payload_parity_report.json"
    )
    assert not plan.script_path.exists()
    assert not (spec.submission.evidence_dir / "acceptance").exists()


def test_render_acceptance_tar_payload_parity_writes_containerized_script(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    spec = load_workflow_runspec(tmp_path, data)

    plan = render_acceptance_tar_payload_parity(spec, dry_run=False)

    script = plan.script_path.read_text()
    paths = data["paths"]
    assert isinstance(paths, dict)
    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --account=acct" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=32G" in script
    assert "#SBATCH --time=00:30:00" in script
    assert f"--container-image={spec.container.image}" in script
    assert f"{paths['orchestration_repo']}:/workspace/bspp-orchestration" in script
    assert "/opt/bspp-orchestration-env/.pixi/envs/default/bin/python" in script
    assert (
        'export PYTHONPATH="${ORCHESTRATION_ROOT}/packages/orchestration-contract/src:'
        '${ORCHESTRATION_ROOT}/packages/orchestration-runtime/src${PYTHONPATH:+:${PYTHONPATH}}"'
    ) in script
    assert "validate \\\n  tar-payload-parity" in script
    assert f"--baseline-dir \\\n  {spec.acceptance.baseline_output_dir}" in script
    assert f"--candidate-dir \\\n  {spec.paths.output_dir}" in script
    assert "--relative-dir \\\n  local_tars" in script
    assert "--match-mode \\\n  by-tar" in script
    assert '--workers \\\n  "${SLURM_CPUS_PER_TASK}"' in script
    assert "--baseline-run-name \\\n  baseline" in script
    assert "--candidate-run-name \\\n  candidate" in script
    assert "--payload-sample-count \\\n  20" in script
    assert f"--write-report \\\n  {plan.evidence_dir}" in script
    assert "--strict" in script
    report = json.loads((plan.script_path.parent / "tar_payload_parity_sbatch_report.json").read_text())
    assert report["plan"]["report_path"] == str(plan.report_path)


def test_render_acceptance_semantic_writes_expected_flags_and_optional_booleans(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    acceptance = data["acceptance"]
    assert isinstance(acceptance, dict)
    acceptance["candidate_parquet_required"] = False
    acceptance["compare_failed_sets"] = False
    acceptance["compare_tar_manifest_rows"] = False
    acceptance["compare_analysis_model_rows"] = False
    spec = load_workflow_runspec(tmp_path, data)

    plan = render_acceptance_semantic(spec, dry_run=False)

    script = plan.script_path.read_text()
    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --account=acct" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --time=00:20:00" in script
    assert "validate \\\n  semantic-acceptance" in script
    assert f"--baseline-dir \\\n  {spec.acceptance.baseline_output_dir}" in script
    assert f"--candidate-dir \\\n  {spec.paths.output_dir}" in script
    assert "--expected-tar-count \\\n  1" in script
    assert "--expected-local-tars-rows \\\n  1" in script
    assert "--expected-failed-rows \\\n  0" in script
    assert "--expected-analysis-rows \\\n  10" in script
    assert "--expected-selected-ids \\\n  2" in script
    assert "--candidate-parquet-optional" in script
    assert "--no-compare-failed-sets" in script
    assert "--no-compare-tar-manifest-rows" in script
    assert "--no-compare-analysis-model-rows" in script
    assert f"--write-report \\\n  {plan.evidence_dir}" in script
    report = json.loads((plan.script_path.parent / "semantic_acceptance_sbatch_report.json").read_text())
    assert report["plan"]["report_path"] == str(plan.report_path)
