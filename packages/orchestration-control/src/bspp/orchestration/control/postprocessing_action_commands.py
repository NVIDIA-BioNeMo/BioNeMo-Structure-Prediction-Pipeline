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

"""Shared V2 postprocessing Runtime Action command resolution."""

from __future__ import annotations

from bspp.orchestration.contract.postprocessing_action_contract import PostprocessingRuntimeAction
from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.control.workflow_rendering import resolve_workflow_step_command


def resolve_postprocessing_action_command(
    legacy_runspec: RunSpec,
    action: PostprocessingRuntimeAction,
) -> str:
    """Return the V2 command preimage for one stored postprocessing action."""
    if legacy_runspec.workflow is None:
        raise ValueError("postprocessing action command resolution requires a materialized workflow")
    step = next((item for item in legacy_runspec.workflow.steps if item.name == action.step_name), None)
    if step is None:
        raise ValueError(f"postprocessing action {action.action_id!r} has no workflow step")
    if action.step_name == "acceptance-verify-evidence":
        submission = legacy_runspec.submission
        if submission is None:
            raise ValueError("postprocessing acceptance verification requires submission.evidence_dir")
        return _phase_acceptance_verify_command(str(submission.evidence_dir))
    return resolve_workflow_step_command(legacy_runspec, step)


def _phase_acceptance_verify_command(evidence_root: str) -> str:
    """Render verification with report references relative to its attempt evidence root."""
    return "\n".join(
        (
            "\"$PYTHON_BIN\" - <<'BSPP_VERIFY_PHASE_ACCEPTANCE'",
            "import os",
            "from pathlib import Path",
            "from bspp.orchestration.runtime.validation.acceptance_evidence import (",
            "    verify_acceptance_evidence,",
            "    write_acceptance_evidence_report,",
            ")",
            f"evidence_root = Path({evidence_root!r})",
            "os.chdir(evidence_root)",
            "report = verify_acceptance_evidence(",
            "    parity_report_path=Path('acceptance/tar_payload_parity/tar_payload_parity_report.json'),",
            "    semantic_report_path=Path('acceptance/semantic_acceptance/semantic_acceptance_summary.json'),",
            ")",
            "write_acceptance_evidence_report(report, evidence_root / 'acceptance/verify_evidence')",
            "raise SystemExit(0 if report.ok else 1)",
            "BSPP_VERIFY_PHASE_ACCEPTANCE",
        )
    )


__all__ = ["resolve_postprocessing_action_command"]
