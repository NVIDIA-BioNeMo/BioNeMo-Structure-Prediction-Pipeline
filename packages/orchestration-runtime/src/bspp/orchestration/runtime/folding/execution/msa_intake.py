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

"""Folding MSA intake: dispatch local vs remote MSA bundle projection.

``prepare_folding_msa_input`` dispatches on the RunSpec's ``input_location``:

* **local** ``VerifiedLocalBundledArtifactLocation`` → ``project_msa_members``
  (the durable tar is already on disk).
* **remote** ``VerifiedRemoteBundledArtifactLocation`` → ``project_from_remote``
  with the concrete ``download_remote_msa_bundle`` production binding that
  calls ``s3_transfer.cp`` on ``remote_location.bundle_uri``.

The remote download binding rejects ``PlannedTransfer`` (dry-run) and fails
closed on any non-OK ``TransferResult``, mirroring the seam-transport
publisher's contract.
"""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.phase import FoldingPhaseRunSpec
from bspp.orchestration.contract.preprocessing_handoff import (
    MsaArtifactSetManifest,
    VerifiedLocalBundledArtifactLocation,
    VerifiedRemoteBundledArtifactLocation,
)
from bspp.orchestration.runtime.data_movement.common import PlannedTransfer
from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer

from .errors import FoldingBackendError
from .msa_projection import project_from_remote, project_msa_members


class MsaIntakeError(FoldingBackendError):
    """Raised when remote MSA bundle intake fails closed."""


def download_remote_msa_bundle(
    remote_location: VerifiedRemoteBundledArtifactLocation,
    destination: Path,
) -> None:
    """Concrete production binding: download ``remote_location.bundle_uri`` via ``s3_transfer.cp``.

    Rejects ``PlannedTransfer`` (dry-run) before ``.ok`` is checked.
    Fails closed on any non-OK ``TransferResult``.
    """
    result = s3_transfer.cp(remote_location.bundle_uri, str(destination))
    if isinstance(result, PlannedTransfer):
        raise MsaIntakeError("remote MSA bundle intake requires an executed transfer, not a dry-run plan")
    if not result.ok:
        raise MsaIntakeError(
            f"remote MSA bundle download failed with returncode {result.returncode}: {result.stderr_tail.strip()}"
        )


def prepare_folding_msa_input(
    runspec: FoldingPhaseRunSpec,
    workspace: Path,
    *,
    artifact_set: MsaArtifactSetManifest,
    lz4_argv: tuple[str, ...],
) -> tuple[VerifiedLocalBundledArtifactLocation, dict[str, Path]]:
    """Dispatch MSA intake based on the RunSpec's ``input_location`` kind.

    Returns a ``(VerifiedLocalBundledArtifactLocation, dict[str, Path])`` tuple:
    the (possibly derived) local artifact location and the projected A3M
    member path map.

    * **local** → ``project_msa_members``; the local location is the RunSpec's
      own ``input_location`` (already verified on disk).
    * **remote** → ``project_from_remote`` with ``download_remote_msa_bundle``
      as the concrete S3 transport binding.
    """
    consumption: MsaSetConsumption = runspec.payload.msa_set
    location = runspec.input_location
    if isinstance(location, VerifiedLocalBundledArtifactLocation):
        projected = project_msa_members(consumption, artifact_set, location, workspace)
        return location, projected
    if isinstance(location, VerifiedRemoteBundledArtifactLocation):
        return project_from_remote(
            consumption,
            artifact_set,
            location,
            workspace,
            download=download_remote_msa_bundle,
            lz4_argv=lz4_argv,
        )
    raise MsaIntakeError(f"unsupported folding input location type: {type(location).__name__}")


__all__ = ["MsaIntakeError", "download_remote_msa_bundle", "prepare_folding_msa_input"]
