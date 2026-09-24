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

"""Pure preprocessing package and durable-path planning.

Port Baseline:
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/msa_batch.sh:37-52.
"""

from __future__ import annotations

from bspp.orchestration.contract.preprocessing_execution import (
    ExpectedA3M,
    PreprocessingPackagePlan,
    PreprocessingSiteConfig,
)


def plan_preprocessing_package(
    *,
    chunk_name: str,
    expected_a3ms: tuple[ExpectedA3M, ...],
    site: PreprocessingSiteConfig,
    staging_directory: str,
) -> PreprocessingPackagePlan:
    """Render literal baseline tar/lz4 argv without staging or execution."""
    chunk_stem = chunk_name.removesuffix(".fa")
    scratch_tar_path = _join(staging_directory, f"{chunk_stem}.tar")
    scratch_lz4_path = f"{scratch_tar_path}.lz4"
    return PreprocessingPackagePlan(
        chunk_name=chunk_name,
        staging_directory=staging_directory,
        declared_stage_members=tuple(expected.member_name for expected in expected_a3ms),
        tar_member_scope=".",
        scratch_tar_path=scratch_tar_path,
        scratch_lz4_path=scratch_lz4_path,
        durable_tar_path=_join(site.finished_msa_root, f"{chunk_stem}.tar"),
        durable_lz4_path=_join(site.finished_msa_root, f"{chunk_stem}.tar.lz4"),
        completed_input_source_path=_join(site.split_input_root, chunk_name),
        completed_input_path=_join(site.finished_input_root, chunk_name),
        tar_argv=(site.tar_executable, "cf", scratch_tar_path, "-C", staging_directory, "."),
        lz4_argv=(site.lz4_executable, "-v", "-3", scratch_tar_path, scratch_lz4_path),
    )


def _join(root: str, *parts: str) -> str:
    return "/".join((root.rstrip("/"), *parts))


__all__ = ["plan_preprocessing_package"]
