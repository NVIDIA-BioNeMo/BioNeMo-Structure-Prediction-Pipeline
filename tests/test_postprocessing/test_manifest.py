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

"""Tests for bspp.orchestration.runtime.postprocessing.manifest."""

from __future__ import annotations

import json
from pathlib import Path

from bspp.orchestration.runtime.postprocessing.manifest import (
    DatasetConfig,
    ProviderConfig,
    filter_manifest_for_models,
    load_dataset_config,
    load_provider_config,
    read_archive_list,
    read_file_index,
    read_model_ids,
    write_archive_list,
    write_dataset_config,
    write_file_index,
    write_model_ids,
    write_provider_json,
)

# --- model IDs ---


def test_write_read_model_ids_roundtrip(tmp_path: Path) -> None:
    ids = ["AF-0000000000000003", "AF-0000000000000001", "AF-0000000000000002"]
    path = tmp_path / "model_ids.txt"
    write_model_ids(ids, path)

    loaded = read_model_ids(path)
    # write_model_ids sorts and deduplicates
    assert loaded == ["AF-0000000000000001", "AF-0000000000000002", "AF-0000000000000003"]


def test_write_model_ids_deduplicates(tmp_path: Path) -> None:
    ids = ["AF-001", "AF-001", "AF-002"]
    path = tmp_path / "model_ids.txt"
    write_model_ids(ids, path)
    loaded = read_model_ids(path)
    assert loaded == ["AF-001", "AF-002"]


def test_write_model_ids_empty(tmp_path: Path) -> None:
    path = tmp_path / "model_ids.txt"
    write_model_ids([], path)
    loaded = read_model_ids(path)
    assert loaded == []


# --- file index ---


def test_write_read_file_index_roundtrip(tmp_path: Path) -> None:
    index = {
        "AF-001": ["AF-001-model_v1.pdb", "AF-001-meta_v1.json"],
        "AF-002": ["AF-002-model_v1.pdb"],
    }
    path = tmp_path / "file_index.json"
    write_file_index(index, path)

    loaded = read_file_index(path)
    assert loaded == index


# --- archive list ---


def test_write_read_archive_list_roundtrip(tmp_path: Path) -> None:
    archives = ["batch_01.tar.lz4", "batch_02.tar.lz4"]
    path = tmp_path / "archive_list.json"
    write_archive_list(archives, path)

    loaded = read_archive_list(path)
    assert loaded == archives


# --- manifest filtering ---


def _write_manifest_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Helper to create a test manifest CSV."""
    import csv

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_entity_id", "chain_id", "uniprot_ac"])
        writer.writeheader()
        writer.writerows(rows)


def test_filter_manifest_for_models(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest_csv(
        manifest,
        [
            {"model_entity_id": "AF-001", "chain_id": "A", "uniprot_ac": "P12345"},
            {"model_entity_id": "AF-002", "chain_id": "A", "uniprot_ac": "P67890"},
            {"model_entity_id": "AF-003", "chain_id": "A", "uniprot_ac": "Q11111"},
            {"model_entity_id": "AF-002", "chain_id": "B", "uniprot_ac": "P99999"},
        ],
    )

    output = tmp_path / "filtered.csv"
    count = filter_manifest_for_models(manifest, {"AF-001", "AF-002"}, output)

    assert count == 3  # AF-001 (1 row) + AF-002 (2 rows)

    import csv

    with output.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 3
    ids = [r["model_entity_id"] for r in rows]
    assert "AF-003" not in ids
    assert ids.count("AF-002") == 2


def test_filter_manifest_empty_intersection(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest_csv(
        manifest,
        [
            {"model_entity_id": "AF-001", "chain_id": "A", "uniprot_ac": "P12345"},
            {"model_entity_id": "AF-002", "chain_id": "A", "uniprot_ac": "P67890"},
        ],
    )

    output = tmp_path / "filtered.csv"
    count = filter_manifest_for_models(manifest, {"AF-999"}, output)
    assert count == 0


# --- dataset config ---


def test_write_dataset_config(tmp_path: Path) -> None:
    path = tmp_path / "dataset_config.json"
    write_dataset_config(
        tool_used="AlphaFold",
        provider_id="TEST",
        output_path=path,
        model_created_date="2024-01-01T00:00:00Z",
    )

    data = json.loads(path.read_text())
    assert data["toolUsed"] == "AlphaFold"
    assert data["providerId"] == "TEST"
    assert data["entityType"] == "protein"
    assert data["isUniProt"] is True
    assert data["modelCreatedDate"] == "2024-01-01T00:00:00Z"
    assert data["versionTag"] == "v1"
    assert data["latestVersion"] == 1
    assert data["allVersions"] == [1]


def test_write_dataset_config_custom_fields(tmp_path: Path) -> None:
    path = tmp_path / "dataset_config.json"
    write_dataset_config(
        tool_used="OpenFold",
        provider_id="NVDA",
        output_path=path,
        entity_type="complex",
        version_tag="v2",
        custom_field="custom_value",
    )

    data = json.loads(path.read_text())
    assert data["entityType"] == "complex"
    assert data["versionTag"] == "v2"
    assert data["custom_field"] == "custom_value"


def test_load_dataset_config_model(tmp_path: Path) -> None:
    path = tmp_path / "dataset_config.json"
    write_dataset_config(
        tool_used="AlphaFold",
        provider_id="TEST",
        output_path=path,
        uniqueIdTemplate="{model_entity_id}",
    )

    config = load_dataset_config(path)
    assert isinstance(config, DatasetConfig)
    assert config.tool_used == "AlphaFold"
    assert config.provider_id == "TEST"
    assert config.to_wire_dict()["providerId"] == "TEST"
    assert config.to_compat_dict()["uniqueIdTemplate"] == "{model_entity_id}"


# --- provider JSON ---


def test_write_provider_json(tmp_path: Path) -> None:
    path = tmp_path / "provider.json"
    write_provider_json(
        provider_id="TEST",
        provider_name="Test Provider",
        output_path=path,
    )

    data = json.loads(path.read_text())
    assert data["providerId"] == "TEST"
    assert data["providerName"] == "Test Provider"
    assert data["providerUrl"] == "https://alphafold.ebi.ac.uk"
    assert len(data["copyrights"]) == 1
    assert "Test Provider" in data["copyrights"][0]


def test_write_provider_json_custom_url(tmp_path: Path) -> None:
    path = tmp_path / "provider.json"
    write_provider_json(
        provider_id="NVDA",
        provider_name="NVIDIA",
        output_path=path,
        provider_url="https://nvidia.com",
    )

    data = json.loads(path.read_text())
    assert data["providerUrl"] == "https://nvidia.com"


def test_load_provider_config_model(tmp_path: Path) -> None:
    path = tmp_path / "provider.json"
    write_provider_json(
        provider_id="NVDA",
        provider_name="NVIDIA",
        output_path=path,
        provider_url="https://nvidia.com",
    )

    config = load_provider_config(path)
    assert isinstance(config, ProviderConfig)
    assert config.provider_name == "NVIDIA"
    assert config.to_wire_dict()["providerUrl"] == "https://nvidia.com"


def test_load_provider_config_model_defaults_copyrights(tmp_path: Path) -> None:
    path = tmp_path / "provider.json"
    path.write_text(json.dumps({"providerId": "NVDA", "providerName": "NVIDIA"}))

    config = load_provider_config(path)

    assert config.copyrights == ["Copyright 2024 NVIDIA. All rights reserved."]
