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

"""Control-side publication approval after independent release acceptance."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from bspp.orchestration.contract.release_acceptance import (
    PublicationApproval,
    PublicationApprovalStore,
    acceptance_evidence_from_bytes,
    validate_acceptance,
)
from bspp.orchestration.control.governed_submission import stable_read

MAX_ACCEPTANCE_RECORD_BYTES = 1024 * 1024


def approve_publication(
    acceptance_path: Path,
    *,
    evidence_root: Path,
    expected_runspec_sha256: str,
    destination: str,
    approval_path: Path,
) -> PublicationApproval:
    """Re-read and validate acceptance before creating one destination-bound approval."""
    evidence = acceptance_evidence_from_bytes(stable_read(acceptance_path, maximum_bytes=MAX_ACCEPTANCE_RECORD_BYTES))
    validation = validate_acceptance(
        evidence,
        evidence_root=evidence_root,
        expected_runspec_sha256=expected_runspec_sha256,
    )
    if not validation.ok:
        raise ValueError("release acceptance failed: " + "; ".join(validation.issues))
    approval = PublicationApproval.create_from_acceptance(validation, destination=destination)
    PublicationApprovalStore(approval_path).create(approval)
    return approval


def execute_approved_publication[PublicationResult](
    approval_path: Path,
    *,
    acceptance_sha256: str,
    destination: str,
    publish: Callable[[str], PublicationResult],
) -> PublicationResult:
    """Consume the exact approval before invoking an external publication adapter."""
    approval = PublicationApprovalStore(approval_path).consume(
        acceptance_sha256=acceptance_sha256,
        destination=destination,
    )
    return publish(approval.destination)


__all__ = ["approve_publication", "execute_approved_publication"]
