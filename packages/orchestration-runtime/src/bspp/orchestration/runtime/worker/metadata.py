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

"""Metadata isolation and finalization helpers for the native worker."""

from __future__ import annotations

import json
import shutil
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bspp.orchestration.runtime.worker.metadata_combine import combine_metadata_files
from bspp.orchestration.runtime.worker.output_layout import (
    metadata_collection_relative_path,
    metadata_search_relative_path,
)

META_JSON_SUFFIX = "-meta_v1.json"


class MetadataCommandRunner(Protocol):
    """Fakeable boundary for metadata combine invocations."""

    def __call__(self, plan: MetadataCombinePlan) -> int:
        """Run a metadata combine plan and return its exit code."""


@dataclass(frozen=True, slots=True)
class MetadataCombinePlan:
    """One metadata combine invocation."""

    input_dir: Path
    output_dir: Path
    output_filename: str


@dataclass(frozen=True, slots=True)
class MetadataFinalizePlan:
    """The search and collection combine plans for one shard."""

    search: MetadataCombinePlan | None
    collection: MetadataCombinePlan | None

    @property
    def plans(self) -> tuple[MetadataCombinePlan, ...]:
        """Return the concrete combine plans, skipping missing inputs."""

        return tuple(plan for plan in (self.search, self.collection) if plan is not None)


@dataclass(frozen=True, slots=True)
class MetadataFinalizeResult:
    """Result of executing metadata finalization."""

    exit_code: int
    output_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class PrefilterResult:
    """Input prefilter result for one processing batch."""

    good_model_ids: tuple[str, ...]
    bad_models: tuple[tuple[str, str], ...]


def clean_sub_shard_metadata_state(work_dir: Path) -> tuple[Path, ...]:
    """Delete metadata/cache state that must not leak between sub-shards."""

    removed: list[Path] = []
    for dirname in ("model_jsons", "chain_jsons"):
        path = work_dir / dirname
        if path.exists():
            shutil.rmtree(path)
            removed.append(path)
    cache_file = work_dir / ".pipeline_cache.json"
    if cache_file.exists():
        cache_file.unlink()
        removed.append(cache_file)
    return tuple(removed)


def clean_metadata_json_outputs(work_dir: Path, model_ids: Iterable[str]) -> int:
    """Remove per-model metadata JSONs for failed or fallback IDs."""

    removed = 0
    for dirname in ("model_jsons", "chain_jsons"):
        json_dir = work_dir / dirname
        if not json_dir.exists():
            continue
        for model_id in model_ids:
            path = json_dir / f"{model_id}.json"
            if path.exists():
                path.unlink()
                removed += 1
    return removed


def check_meta_json_for_null(path: Path, *, fast: bool = True) -> tuple[bool, str]:
    """Check one input meta JSON for null values that crash downstream scoring."""

    if not path.exists():
        return False, ""
    raw = path.read_text(encoding="utf-8")
    if fast:
        if ",null," in raw or "[null]" in raw or ",null]" in raw or "[null," in raw:
            return True, "null in input json"
        return False, ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return True, f"JSON parse error: {exc}"
    for key in ("pae", "plddt", "max_pae"):
        if key not in data:
            return True, f"missing key: {key}"
        if data[key] is None:
            return True, "null in input json"
    return False, ""


def prefilter_batch_inputs(
    model_ids: Iterable[str],
    input_dir: Path,
    *,
    file_index: dict[str, list[str]] | None = None,
    fast: bool = True,
) -> PrefilterResult:
    """Pre-validate input meta JSONs for nulls before pipeline execution."""

    good_model_ids: list[str] = []
    bad_models: list[tuple[str, str]] = []
    for model_id in model_ids:
        meta_path = _meta_json_path(model_id, input_dir, file_index)
        has_null, reason = check_meta_json_for_null(meta_path, fast=fast)
        if has_null and reason:
            bad_models.append((model_id, reason))
        else:
            good_model_ids.append(model_id)
    return PrefilterResult(good_model_ids=tuple(good_model_ids), bad_models=tuple(bad_models))


def metadata_destination_dir(work_dir: Path, shard_success_dir: Path, *, s3_upload_enabled: bool) -> Path:
    """Return the metadata combine destination for the current upload mode."""

    return work_dir / "_metadata_staging" if s3_upload_enabled else shard_success_dir


def plan_metadata_finalization(
    *,
    work_dir: Path,
    output_base_dir: Path,
    logical_shard_id: int,
    total_shards: int,
    dataset_tag: str = "",
) -> MetadataFinalizePlan:
    """Plan the search and chain-collection metadata combines."""

    search_plan = _plan_one_metadata_combine(
        input_dir=work_dir / "model_jsons",
        output_path=output_base_dir / metadata_search_relative_path(logical_shard_id, total_shards, dataset_tag),
    )
    collection_plan = _plan_one_metadata_combine(
        input_dir=work_dir / "chain_jsons",
        output_path=output_base_dir / metadata_collection_relative_path(logical_shard_id, total_shards, dataset_tag),
    )
    return MetadataFinalizePlan(search=search_plan, collection=collection_plan)


def finalize_metadata(
    plan: MetadataFinalizePlan,
    *,
    runner: MetadataCommandRunner | None = None,
) -> MetadataFinalizeResult:
    """Run planned metadata combines in legacy order."""

    output_files: list[Path] = []
    for combine_plan in plan.plans:
        combine_plan.output_dir.mkdir(parents=True, exist_ok=True)
        rc = runner(combine_plan) if runner is not None else _run_metadata_combine(combine_plan)
        if rc != 0:
            return MetadataFinalizeResult(exit_code=rc, output_files=tuple(output_files))
        output_files.append(combine_plan.output_dir / combine_plan.output_filename)
    return MetadataFinalizeResult(exit_code=0, output_files=tuple(output_files))


def _plan_one_metadata_combine(
    *,
    input_dir: Path,
    output_path: Path,
) -> MetadataCombinePlan | None:
    if not input_dir.exists():
        return None
    return MetadataCombinePlan(
        input_dir=input_dir,
        output_dir=output_path.parent,
        output_filename=output_path.name,
    )


def _run_metadata_combine(plan: MetadataCombinePlan) -> int:
    try:
        combine_metadata_files(
            input_dir=plan.input_dir,
            output_dir=plan.output_dir,
            output_filename=plan.output_filename,
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0


def _meta_json_path(model_id: str, input_dir: Path, file_index: dict[str, list[str]] | None) -> Path:
    if file_index:
        filenames = file_index.get(model_id)
        meta_name = next(
            (filename for filename in (filenames or []) if filename.endswith(META_JSON_SUFFIX)),
            f"{model_id}{META_JSON_SUFFIX}",
        )
        return input_dir / meta_name
    return input_dir / f"{model_id}{META_JSON_SUFFIX}"


__all__ = [
    "META_JSON_SUFFIX",
    "MetadataCombinePlan",
    "MetadataCommandRunner",
    "MetadataFinalizePlan",
    "MetadataFinalizeResult",
    "PrefilterResult",
    "check_meta_json_for_null",
    "clean_metadata_json_outputs",
    "clean_sub_shard_metadata_state",
    "finalize_metadata",
    "metadata_destination_dir",
    "plan_metadata_finalization",
    "prefilter_batch_inputs",
]
