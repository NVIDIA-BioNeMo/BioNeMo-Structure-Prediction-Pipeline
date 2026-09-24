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

"""Postprocessing scheduler evidence contract and Control export tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
from click.testing import CliRunner

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_receipt import PostprocessingTaskReceiptEvidence
from bspp.orchestration.contract.postprocessing_scheduler_evidence import (
    PostprocessingSchedulerActionEvidence,
    PostprocessingSchedulerEvidence,
    postprocessing_scheduler_evidence_from_mapping,
)
from bspp.orchestration.control.cli import cli

RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
ATTEMPT_ID = "attempt-0001"
SHA = "1" * 64


def test_scheduler_evidence_strict_round_trip_binds_dependency_jobs() -> None:
    evidence = _evidence()

    assert postprocessing_scheduler_evidence_from_mapping(evidence.to_mapping()) == evidence

    changed = list(evidence.actions)
    changed[-1] = replace(changed[-1], dependency_job_ids=("1001", "1004", "1005"))
    with pytest.raises(ValueError, match="dependency job ids"):
        replace(evidence, actions=tuple(changed))


def test_scheduler_evidence_rejects_unknown_fields_and_id_tampering() -> None:
    evidence = _evidence()
    payload = evidence.to_mapping()
    payload["surprise"] = True
    with pytest.raises(ValueError, match="missing or extra"):
        postprocessing_scheduler_evidence_from_mapping(payload)
    with pytest.raises(ValueError, match="identity preimage"):
        replace(evidence, scheduler_evidence_id="2" * 64)


def test_phase_evidence_cli_exposes_fetch_and_scheduler_export() -> None:
    result = CliRunner().invoke(cli, ["phase", "evidence", "--help"])

    assert result.exit_code == 0
    assert "fetch" in result.output
    assert "export-scheduler" in result.output


def _evidence() -> PostprocessingSchedulerEvidence:
    action_specs = (
        ("postprocessing-01-preflight", (), "1001"),
        ("postprocessing-05-analysis-finalize", ("postprocessing-01-preflight",), "1002"),
        (
            "postprocessing-06-acceptance-tar-payload-parity",
            ("postprocessing-05-analysis-finalize",),
            "1003",
        ),
        ("postprocessing-07-acceptance-semantic", ("postprocessing-05-analysis-finalize",), "1004"),
        (
            "postprocessing-08-acceptance-verify-evidence",
            (
                "postprocessing-06-acceptance-tar-payload-parity",
                "postprocessing-07-acceptance-semantic",
            ),
            "1005",
        ),
        (
            "postprocessing-09-acceptance-adjudication",
            (
                "postprocessing-06-acceptance-tar-payload-parity",
                "postprocessing-07-acceptance-semantic",
                "postprocessing-08-acceptance-verify-evidence",
            ),
            "1006",
        ),
    )
    jobs = {action_id: job_id for action_id, _, job_id in action_specs}
    actions = tuple(
        PostprocessingSchedulerActionEvidence(
            action_id=action_id,
            runtime_action_digest=SHA,
            dependencies=dependencies,
            cluster_script_path=f"/staging/{action_id}.sbatch",
            script_sha256=SHA,
            scheduler_correlation_token="bspp-pp-" + "a" * 48,
            dependency_job_ids=tuple(jobs[item] for item in dependencies),
            parent_job_id=job_id,
            expected_task_indexes=(),
            tasks=(
                PostprocessingTaskReceiptEvidence(
                    scheduler_job_id=job_id,
                    state="COMPLETED",
                    exit_code="0:0",
                    source="sacct",
                    restarts=0,
                ),
            ),
        )
        for action_id, dependencies, job_id in action_specs
    )
    identity: dict[str, object] = {
        "schema_version": 1,
        "evidence_kind": "postprocessing-control-scheduler-evidence-v1",
        "phase_run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "phase_runspec_digest": SHA,
        "action_graph_digest": SHA,
        "submission_id": "postprocessing-submission-" + "b" * 64,
        "actions": [item.to_mapping() for item in actions],
    }
    return PostprocessingSchedulerEvidence(
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_runspec_digest=SHA,
        action_graph_digest=SHA,
        submission_id="postprocessing-submission-" + "b" * 64,
        actions=actions,
        scheduler_evidence_id=canonical_mapping_digest(identity),
    )
