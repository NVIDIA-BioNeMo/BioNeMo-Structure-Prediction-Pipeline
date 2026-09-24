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

"""Strict adapter from authentic Runtime Qualification records to Phase authority."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_execution import (
    QualifiedPostprocessingRuntimeSelection,
)
from bspp.orchestration.control.profiles import ResolvedClusterProfile, resolve_cluster_profile
from bspp.orchestration.control.runtime_qualification import check_runtime_qualification
from bspp.orchestration.control.runtime_qualification_validation import (
    ValidatedRuntimeQualification,
    validate_promoted_runtime_qualification,
)
from bspp.orchestration.control.transport import CommandRunner, default_command_runner


def select_pinned_postprocessing_runtime(
    document: bytes,
    *,
    attempt_id: str,
    profile: ResolvedClusterProfile | None,
    source_repo: Path | None,
    observed_at: datetime | None,
) -> QualifiedPostprocessingRuntimeSelection:
    """Validate exact promoted producer bytes and freeze the operational selection."""
    validated = validate_promoted_runtime_qualification(
        document,
        profile=profile,
        source_repo=source_repo,
        observed_at=observed_at,
    )
    return _postprocessing_selection(validated, document=document, attempt_id=attempt_id)


def _postprocessing_selection(
    validated: ValidatedRuntimeQualification,
    *,
    document: bytes,
    attempt_id: str,
) -> QualifiedPostprocessingRuntimeSelection:
    image_mapping = validated.image.to_mapping()
    source_mapping = validated.source_package.to_mapping()
    selected_mapping = validated.selected_source.to_mapping()
    toolkit_mapping = validated.toolkit_package.to_mapping() if validated.toolkit_package is not None else None
    toolkit_identity = toolkit_mapping if toolkit_mapping is not None else image_mapping
    requeue_exit = validated.autorequeue_cap.requeue_exit if validated.autorequeue_cap is not None else None
    max_batch_requeue = validated.autorequeue_cap.max_batch_requeue if validated.autorequeue_cap is not None else None
    return QualifiedPostprocessingRuntimeSelection(
        tuple_id=validated.tuple_id,
        qualification_location=f"attempts/{attempt_id}/runtime-qualification.json",
        qualification_sha256=hashlib.sha256(document).hexdigest(),
        qualification_size_bytes=len(document),
        qualified_at=validated.qualified_at,
        expires_at=validated.expires_at,
        image_path=str(validated.image.path),
        image_sha256=validated.image.sha256,
        image_size_bytes=validated.image.size_bytes,
        image_policy=validated.image.policy,
        source_kind=validated.source_kind,
        source_revision=validated.selected_source.revision,
        source_package_path=str(validated.source_package.package_path),
        toolkit_package_path=(
            str(validated.toolkit_package.package_path) if validated.toolkit_package is not None else None
        ),
        runtime_ipsae_binary_path=str(Path(validated.remote_result_path).parent / validated.runtime_ipsae.binary.path),
        runtime_ipsae_binary_sha256=validated.runtime_ipsae.binary.sha256,
        runtime_ipsae_binary_size_bytes=validated.runtime_ipsae.binary.size_bytes,
        source_identity_digest=canonical_mapping_digest(selected_mapping),
        source_package_identity_digest=canonical_mapping_digest(source_mapping),
        toolkit_identity_digest=canonical_mapping_digest(toolkit_identity),
        bootstrap_sha256=validated.bootstrap_sha256,
        runtime_component_identity_digest=canonical_mapping_digest(validated.runtime_ipsae.to_mapping()),
        requeue_exit=requeue_exit,
        max_batch_requeue=max_batch_requeue,
    )


def replay_postprocessing_runtime(
    document: bytes,
    *,
    attempt_id: str,
) -> tuple[str, QualifiedPostprocessingRuntimeSelection]:
    """Rebuild selection using frozen record bytes without ambient state."""
    validated = validate_promoted_runtime_qualification(
        document,
        profile=None,
        source_repo=None,
        observed_at=None,
    )
    return validated.profile_name, _postprocessing_selection(validated, document=document, attempt_id=attempt_id)


def resolve_current_postprocessing_runtime(
    *,
    attempt_id: str,
    profile_name: str,
    config_path: Path,
    source_repo: Path,
    observed_at: datetime,
    runner: CommandRunner = default_command_runner,
) -> tuple[ResolvedClusterProfile, QualifiedPostprocessingRuntimeSelection, bytes]:
    """Resolve the current authentic producer record for a Retry Attempt."""
    profile = resolve_cluster_profile(profile_name, config_path=config_path)
    check = check_runtime_qualification(
        profile_name=profile_name,
        config_path=config_path,
        source_repo=source_repo,
        now=observed_at,
        runner=runner,
    )
    if not check.current:
        raise ValueError(f"postprocessing Retry requires current Runtime Qualification: {check.reason}")
    if check.snapshot is None:
        raise ValueError("postprocessing Retry qualification check did not return exact record authority")
    document = check.snapshot.document
    selection = select_pinned_postprocessing_runtime(
        document,
        attempt_id=attempt_id,
        profile=profile,
        source_repo=source_repo,
        observed_at=observed_at,
    )
    if selection.tuple_id != check.tuple_id:
        raise ValueError("postprocessing Retry qualification check and record tuple differ")
    return profile, selection, document


__all__ = [
    "replay_postprocessing_runtime",
    "resolve_current_postprocessing_runtime",
    "select_pinned_postprocessing_runtime",
]
