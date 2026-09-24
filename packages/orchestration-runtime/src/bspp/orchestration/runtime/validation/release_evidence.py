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

"""Compose independently observed release reports into strict acceptance evidence."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from bspp.orchestration.contract.release_acceptance import (
    AcceptanceEvidence,
    CandidateInventoryReport,
    LocalIntegrityReport,
    NoUploadReport,
    ProvenanceIdentityReport,
    SemanticValidationReport,
    TerminalFailureReport,
)


def build_release_acceptance_evidence(
    *,
    expected_candidates: tuple[str, ...],
    observed_candidates: tuple[str, ...],
    local_integrity: LocalIntegrityReport,
    semantic_validation: SemanticValidationReport,
    terminal_failures: TerminalFailureReport,
    no_upload: NoUploadReport,
    provenance_identity: ProvenanceIdentityReport,
) -> AcceptanceEvidence:
    """Build the versioned envelope without consulting worker-reported ok flags."""
    return AcceptanceEvidence(
        CandidateInventoryReport(1, expected_candidates, observed_candidates),
        local_integrity,
        semantic_validation,
        terminal_failures,
        no_upload,
        provenance_identity,
    )


def write_release_acceptance_evidence(
    evidence: AcceptanceEvidence,
    path: Path,
) -> Path:
    """Durably publish canonical acceptance evidence exactly once."""
    target = path.absolute()
    if target != target.resolve(strict=False):
        raise ValueError("acceptance evidence path must be canonical")
    parent = target.parent
    parent_before = parent.lstat()
    if not stat.S_ISDIR(parent_before.st_mode) or parent.is_symlink():
        raise ValueError("acceptance evidence parent must be a real directory")

    directory_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened_parent = os.fstat(directory_fd)
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            parent_before.st_dev,
            parent_before.st_ino,
        ):
            raise ValueError("acceptance evidence parent changed before publication")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        output_fd = os.open(target.name, flags, 0o444, dir_fd=directory_fd)
        try:
            remaining = memoryview(evidence.canonical_bytes())
            while remaining:
                written = os.write(output_fd, remaining)
                if written <= 0:
                    raise OSError("acceptance evidence write made no progress")
                remaining = remaining[written:]
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        os.fsync(directory_fd)
        parent_after = parent.lstat()
        if (parent_after.st_dev, parent_after.st_ino) != (
            opened_parent.st_dev,
            opened_parent.st_ino,
        ):
            raise ValueError("acceptance evidence parent changed during publication")
    finally:
        os.close(directory_fd)
    return target


__all__ = ["build_release_acceptance_evidence", "write_release_acceptance_evidence"]
