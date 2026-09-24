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

"""Manifest operations: model ID lists, file indices, archive lists, and CSV filtering.

All write functions create parent directories as needed. All read functions
raise :class:`FileNotFoundError` if the path does not exist.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DatasetConfig",
    "ProviderConfig",
    "filter_manifest_for_models",
    "load_dataset_config",
    "load_provider_config",
    "read_archive_list",
    "read_file_index",
    "read_model_ids",
    "write_archive_list",
    "write_dataset_config",
    "write_file_index",
    "write_model_ids",
    "write_provider_json",
]


class DatasetConfig(BaseModel):
    """Typed BSPP dataset metadata config."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    tool_used: str = Field(alias="toolUsed")
    provider_id: str = Field(alias="providerId")
    entity_type: str = Field(default="protein", alias="entityType")
    is_uniprot: bool = Field(default=True, alias="isUniProt")
    model_created_date: str = Field(default="", alias="modelCreatedDate")
    version_tag: str = Field(default="v1", alias="versionTag")
    ordinal: int = 1
    latest_version: int = Field(default=1, alias="latestVersion")
    all_versions: list[int] = Field(default_factory=lambda: [1], alias="allVersions")

    def to_compat_dict(self) -> dict[str, Any]:
        """Return loaded fields using JSON aliases without adding defaults."""
        return self.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)

    def to_wire_dict(self) -> dict[str, Any]:
        """Return the complete emitted JSON object shape."""
        return self.model_dump(by_alias=True, exclude_none=True)


class ProviderConfig(BaseModel):
    """Typed BSPP provider metadata config."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    provider_id: str = Field(alias="providerId")
    provider_name: str = Field(alias="providerName")
    provider_url: str = Field(default="https://alphafold.ebi.ac.uk", alias="providerUrl")
    copyrights: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _default_copyrights(self) -> ProviderConfig:
        if not self.copyrights:
            self.copyrights = [f"Copyright 2024 {self.provider_name}. All rights reserved."]
        return self

    def to_compat_dict(self) -> dict[str, Any]:
        """Return loaded fields using JSON aliases without adding defaults."""
        return self.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)

    def to_wire_dict(self) -> dict[str, Any]:
        """Return the complete emitted JSON object shape."""
        return self.model_dump(by_alias=True, exclude_none=True)


def write_model_ids(model_ids: list[str], output_path: Path) -> None:
    """Write sorted, deduplicated model IDs one per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_ids = sorted(set(model_ids))
    output_path.write_text("\n".join(sorted_ids) + "\n" if sorted_ids else "")


def read_model_ids(path: Path) -> list[str]:
    """Read model IDs from a text file (one per line, blank lines skipped)."""
    text = path.read_text()
    return [line for line in text.splitlines() if line.strip()]


def write_file_index(index: dict[str, list[str]], output_path: Path) -> None:
    """Write a model-ID-to-filenames index as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, sort_keys=True))


def read_file_index(path: Path) -> dict[str, list[str]]:
    """Read a file index JSON back into a dict."""
    result: dict[str, list[str]] = json.loads(path.read_text())
    return result


def write_archive_list(archives: list[str], output_path: Path) -> None:
    """Write an archive name list as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(archives))


def read_archive_list(path: Path) -> list[str]:
    """Read an archive list JSON back into a list of strings."""
    result: list[str] = json.loads(path.read_text())
    return result


def filter_manifest_for_models(
    manifest_path: Path,
    model_ids: set[str],
    output_path: Path,
) -> int:
    """Filter a chain-mapping manifest CSV for rows matching *model_ids*.

    Reads the CSV at *manifest_path*, keeps only rows whose
    ``model_entity_id`` column value is in *model_ids*, and writes the
    filtered CSV to *output_path*.

    Returns the number of rows written (excluding the header).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with manifest_path.open(newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            msg = f"Manifest CSV has no header: {manifest_path}"
            raise ValueError(msg)

        with output_path.open("w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row.get("model_entity_id") in model_ids:
                    writer.writerow(row)
                    rows_written += 1

    return rows_written


def write_dataset_config(
    tool_used: str,
    provider_id: str,
    output_path: Path,
    **kwargs: Any,
) -> None:
    """Write a dataset configuration JSON file.

    Creates a JSON with standard BSPP dataset metadata fields. Any extra
    keyword arguments are merged into the config dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "toolUsed": tool_used,
        "providerId": provider_id,
        "entityType": kwargs.pop("entity_type", "protein"),
        "isUniProt": kwargs.pop("is_uniprot", True),
        "modelCreatedDate": kwargs.pop("model_created_date", ""),
        "versionTag": kwargs.pop("version_tag", "v1"),
        "ordinal": kwargs.pop("ordinal", 1),
        "latestVersion": kwargs.pop("latest_version", 1),
        "allVersions": kwargs.pop("all_versions", [1]),
    }
    payload.update(kwargs)
    config = DatasetConfig.model_validate(payload)
    output_path.write_text(json.dumps(config.to_wire_dict(), indent=2) + "\n")


def load_dataset_config(path: Path) -> DatasetConfig:
    """Read and validate a dataset configuration JSON file."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        msg = f"Expected a JSON object in {path}, got {type(payload).__name__}"
        raise TypeError(msg)
    return DatasetConfig.model_validate(payload)


def write_provider_json(
    provider_id: str,
    provider_name: str,
    output_path: Path,
    **kwargs: Any,
) -> None:
    """Write a provider configuration JSON file.

    Creates a JSON with standard BSPP provider metadata. Any extra keyword
    arguments are merged into the config dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "providerId": provider_id,
        "providerName": provider_name,
        "providerUrl": kwargs.pop("provider_url", "https://alphafold.ebi.ac.uk"),
        "copyrights": kwargs.pop(
            "copyrights",
            [f"Copyright 2024 {provider_name}. All rights reserved."],
        ),
    }
    payload.update(kwargs)
    provider = ProviderConfig.model_validate(payload)
    output_path.write_text(json.dumps(provider.to_wire_dict(), indent=2) + "\n")


def load_provider_config(path: Path) -> ProviderConfig:
    """Read and validate a provider configuration JSON file."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        msg = f"Expected a JSON object in {path}, got {type(payload).__name__}"
        raise TypeError(msg)
    return ProviderConfig.model_validate(payload)
