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

"""Contract tests for the shared public Cluster Profile example."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bspp.orchestration.control.profiles import UserClusterProfile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "skills/examples/run-plans/cluster-profile.yaml"

_FIELD_TOKENS = (
    "owner",
    "transport",
    "ssh_target",
    "account",
    "paths",
    "project_root",
    "output_root",
    "staging_root",
    "orchestration_repo",
    "image",
    "probe_root",
    "source_bundle_root",
    "runtime_image_cache_root",
    "runtime_qualification_root",
    "runtime_qualification_control_root",
    "governed_package_root",
    "runtime_image_policy",
    "runtime_qualification_expires_hours",
    "preprocessing_runtime",
    "cluster_image_path",
    "cluster_image_sha256",
    "oci_digest",
    "image_lock_sha256",
    "contract_wheel_sha256",
    "runtime_wheel_sha256",
    "control_wheel_sha256",
    "source_commit",
    "source_bundle_sha256",
    "colabfold_version",
    "mmseqs_version",
    "rsync_version",
    "cuda_version",
    "postprocessing_credential_mounts",
    "aws_shared_credentials_file",
    "aws_config_file",
    "database_sets",
    "identifier",
    "version",
    "manifest_path",
    "database_access_policies",
    "database_cache_root",
    "database_cache_namespace",
    "database_cache_unix_user",
    "database_cache_filesystem_type",
    "database_cache_reserve_bytes",
    "database_lock_wait_seconds",
    "extra_mounts",
    "source",
    "target",
    "read_only",
    "folding_release_preset",
    "folding_backend_images",
    "folding_backend_assets",
    "chain_manifest_csv",
    "openfold_model_dir",
    "colabfold_weights_dir",
    "bioir_checkpoint",
    "resources",
    "control_cpu",
    "gpu_worker",
    "analysis_finalize",
    "acceptance_tar_payload_parity",
    "acceptance_semantic",
    "partition",
    "cpus_per_task",
    "memory",
    "time",
    "gres",
)

_ENUM_TOKENS = (
    "ssh",
    "local-slurm",
    "digest-checked",
    "trusted-cache",
    "stage-required",
    "stage-preferred",
    "direct",
    "acceptance",
    "public",
    "openfold-cli",
    "colabfold",
    "bioir",
    "openfold-trt",
)


def _comment_text() -> str:
    lines = PROFILE.read_text().splitlines()
    return "\n".join(line for line in lines if line.strip().startswith("#"))


def test_shared_profile_loads() -> None:
    data = yaml.safe_load(PROFILE.read_text())
    assert isinstance(data, dict)
    clusters = data["clusters"]
    assert isinstance(clusters, dict)
    assert "my-cluster" in clusters

    UserClusterProfile.model_validate(clusters["my-cluster"])


@pytest.mark.parametrize("token", (*_FIELD_TOKENS, *_ENUM_TOKENS))
def test_profile_comment_coverage(token: str) -> None:
    assert token in _comment_text()
