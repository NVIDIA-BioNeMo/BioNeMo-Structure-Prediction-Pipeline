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

"""Immutable credential mount selection for postprocessing renderer authority."""

from __future__ import annotations

from bspp.orchestration.contract.postprocessing_execution import PostprocessingCredentialMountSnapshot
from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.control.profiles import ResolvedClusterProfile


def snapshot_postprocessing_credential_mounts(
    legacy_runspec: RunSpec,
    profile: ResolvedClusterProfile,
) -> PostprocessingCredentialMountSnapshot:
    """Freeze every currently script-affecting credential source locator."""
    reference = legacy_runspec.secrets.s3_credentials_ref
    if reference.scheme != "aws":
        return PostprocessingCredentialMountSnapshot(
            aws_shared_credentials_file=None,
            aws_config_file=None,
        )
    configured = profile.postprocessing_credential_mounts
    if configured is None:
        raise ValueError(
            "AWS postprocessing requires Cluster Profile postprocessing_credential_mounts "
            "with both cluster-visible shared-profile files"
        )
    return PostprocessingCredentialMountSnapshot(
        aws_shared_credentials_file=configured.aws_shared_credentials_file,
        aws_config_file=configured.aws_config_file,
    )


__all__ = ["snapshot_postprocessing_credential_mounts"]
