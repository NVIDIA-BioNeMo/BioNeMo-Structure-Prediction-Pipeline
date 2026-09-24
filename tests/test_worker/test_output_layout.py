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

from __future__ import annotations

from pathlib import Path

import pytest

from bspp.orchestration.runtime.worker import (
    TransferPair,
    batch_done_marker_path,
    batch_tar_name,
    metadata_collection_filename,
    metadata_search_filename,
    metadata_tar_name,
    plan_file_upload_transfers,
    shard_dir,
    success_outputs_dir,
    uploaded_marker_path,
)
from bspp.orchestration.runtime.worker.output_layout import (
    join_destination_prefix,
    metadata_collection_relative_path,
    metadata_search_relative_path,
)


def test_shard_and_marker_paths_are_pure_conventions(tmp_path: Path) -> None:
    shard = shard_dir(tmp_path, 7)

    assert shard == tmp_path / "shard_7"
    assert success_outputs_dir(shard) == tmp_path / "shard_7" / "success_outputs"
    assert uploaded_marker_path(shard) == tmp_path / "shard_7" / ".uploaded"
    assert batch_done_marker_path(shard, 3) == tmp_path / "shard_7" / ".batch_3_done"
    assert not shard.exists()


def test_metadata_names_match_wp8a_one_based_shard_format() -> None:
    assert metadata_search_filename(0, 2, "wp8a") == "AF-metadata-1-of-2-wp8a.json"
    assert metadata_collection_filename(1, 2, "wp8a") == "AF-chain-metadata-2-of-2-wp8a.json"
    assert metadata_search_filename(0, 1) == "AF-metadata-1-of-1.json"
    assert metadata_collection_filename(0, 1, "") == "AF-chain-metadata-1-of-1.json"


def test_metadata_relative_paths_use_search_and_collection_directories() -> None:
    assert metadata_search_relative_path(0, 1, "dataset") == Path(
        "metadata/search/AF-metadata-1-of-1-dataset.json",
    )
    assert metadata_collection_relative_path(0, 1, "dataset") == Path(
        "metadata/collection/AF-chain-metadata-1-of-1-dataset.json",
    )


def test_plan_file_upload_transfers_requires_explicit_prefix(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        plan_file_upload_transfers(tmp_path / "success_outputs", ["AF-1-model_v1.cif"])  # type: ignore[call-arg]


def test_plan_file_upload_transfers_builds_sources_and_destinations(tmp_path: Path) -> None:
    success = tmp_path / "shard_0" / "success_outputs"

    transfers = plan_file_upload_transfers(
        success,
        [
            "AF-0000000000000001-model_v1.cif",
            Path("metadata/search/AF-metadata-1-of-2-wp8a.json"),
        ],
        prefix="s3://example-bucket-test/wp8a/",
    )

    assert transfers == (
        TransferPair(
            source=success / "AF-0000000000000001-model_v1.cif",
            destination="s3://example-bucket-test/wp8a/AF-0000000000000001-model_v1.cif",
        ),
        TransferPair(
            source=success / "metadata/search/AF-metadata-1-of-2-wp8a.json",
            destination="s3://example-bucket-test/wp8a/metadata/search/AF-metadata-1-of-2-wp8a.json",
        ),
    )
    assert transfers[0].s5cmd_cp_args() == (
        "cp",
        str(success / "AF-0000000000000001-model_v1.cif"),
        "s3://example-bucket-test/wp8a/AF-0000000000000001-model_v1.cif",
    )


def test_plan_file_upload_transfers_rejects_paths_outside_success_outputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plan_file_upload_transfers(tmp_path / "success_outputs", ["../escape.cif"], prefix="s3://bucket/run")


@pytest.mark.parametrize(
    "relative_key",
    [
        "metadata\\search\\AF-metadata-1-of-1.json",
        "metadata\\..\\escape.cif",
    ],
)
def test_plan_file_upload_transfers_rejects_backslash_relative_keys_on_posix(
    tmp_path: Path,
    relative_key: str,
) -> None:
    with pytest.raises(ValueError, match="backslashes"):
        plan_file_upload_transfers(tmp_path / "success_outputs", [relative_key], prefix="s3://bucket/run")


def test_join_destination_prefix_rejects_backslash_relative_key_on_posix() -> None:
    with pytest.raises(ValueError, match="backslash separators"):
        join_destination_prefix("s3://bucket/run", "metadata\\..\\escape.cif")


def test_tar_names_use_wp8a_local_tar_conventions() -> None:
    assert batch_tar_name(0, 1) == "shard_0_batch_1.tar"
    assert metadata_tar_name(7) == "shard_7_metadata.tar"
    assert batch_tar_name(2, 3, compression="zstd-members") == "shard_2_batch_3.tar"
    assert metadata_tar_name(2, compression="gzip") == "shard_2_metadata.tar.gz"


def test_package_root_re_exports_output_layout_symbols(tmp_path: Path) -> None:
    import bspp.orchestration.runtime.worker as worker

    transfer = worker.TransferPair(source=tmp_path / "file.cif", destination="s3://bucket/file.cif")

    assert transfer == TransferPair(source=tmp_path / "file.cif", destination="s3://bucket/file.cif")
    assert worker.plan_file_upload_transfers(tmp_path, ["file.cif"], prefix="s3://bucket") == (
        TransferPair(source=tmp_path / "file.cif", destination="s3://bucket/file.cif"),
    )
