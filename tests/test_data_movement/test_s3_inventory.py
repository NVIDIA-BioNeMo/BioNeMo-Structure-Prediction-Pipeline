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

"""Tests for s5cmd inventory argv builders and parsers."""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.runtime.data_movement.s3.client import S3Credentials
from bspp.orchestration.runtime.data_movement.s3.inventory import (
    S3InventoryObject,
    build_list_prefix_argv,
    build_sample_download_argv,
    parse_s5cmd_ls_output,
    read_inventory_csv,
    write_inventory_csv,
)

_CREDS = S3Credentials(access_key_id="key", secret_access_key="secret", endpoint_url="https://swiftstack.test")


def test_build_list_prefix_argv_includes_endpoint_workers_and_glob() -> None:
    argv = build_list_prefix_argv(
        "s3://example-bucket/users/example-user/hq",
        credentials=_CREDS,
        s5cmd_path="/opt/s5cmd",
        numworkers=16,
    )

    assert argv == (
        "/opt/s5cmd",
        "--endpoint-url",
        "https://swiftstack.test",
        "--numworkers",
        "16",
        "ls",
        "s3://example-bucket/users/example-user/hq/*",
    )


def test_build_sample_download_argv() -> None:
    argv = build_sample_download_argv(
        "s3://bucket/chunk_0000.tar",
        Path("/tmp/chunk_0000.tar"),
        credentials=_CREDS,
        s5cmd_path="s5cmd",
    )

    assert argv == (
        "s5cmd",
        "--endpoint-url",
        "https://swiftstack.test",
        "cp",
        "s3://bucket/chunk_0000.tar",
        "/tmp/chunk_0000.tar",
    )


def test_parse_s5cmd_ls_output_reads_objects_and_skips_dirs() -> None:
    output = "\n".join(
        [
            "2026/06/01 12:00:00                 7 s3://bucket/prefix/chunk_0000.tar",
            "2026/06/01 12:00:01               DIR s3://bucket/prefix/nested/",
            "2026/06/01 12:00:02                11 s3://bucket/prefix/chunk_0001.tar",
        ]
    )

    objects = parse_s5cmd_ls_output(output)

    assert objects == (
        S3InventoryObject(uri="s3://bucket/prefix/chunk_0000.tar", size_bytes=7),
        S3InventoryObject(uri="s3://bucket/prefix/chunk_0001.tar", size_bytes=11),
    )


def test_inventory_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "inventory.csv"
    objects = (S3InventoryObject(uri="s3://bucket/key", size_bytes=5),)

    write_inventory_csv(path, objects)

    assert read_inventory_csv(path) == objects
