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

"""Render archive-mode preprocess artifacts from a canonical RunSpec."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from bspp.orchestration.contract.runspec import RunSpec, render_artifact_header
from bspp.orchestration.runtime.inputs.archives import ARCHIVE_COLUMN, DATASET_COLUMNS, ArchiveCoverage
from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary
from bspp.orchestration.runtime.postprocessing.manifest import write_dataset_config, write_provider_json
from bspp.orchestration.runtime.postprocessing.shard_config import compute_archive_shard_config
from bspp.orchestration.runtime.toolkit import resolve_toolkit_for_container

_AF_ID_PATTERN = re.compile(r"AF-\d{16}")
_AFDB_MSA_PATTERN = re.compile(r"AFDB_(AF-\d{16})")
_PDB_ASSEMBLY_MSA_PATTERN = re.compile(r"pdb_[a-z0-9]+_assembly_\d+(?![a-z0-9_])")
_MODEL_ID_COLUMNS = ("model_entity_id", "af_id", "model_id", "msa_path")


@dataclass(frozen=True)
class AllowlistSummary:
    """Summary of per-archive allowlists rendered for archive mode."""

    directory: Path
    files: int
    total_ids: int
    min_ids: int
    max_ids: int
    empty_files: tuple[str, ...]
    archive_mismatches: tuple[str, ...]

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable summary data."""
        return {
            "directory": str(self.directory),
            "files": self.files,
            "total_ids": self.total_ids,
            "min_ids": self.min_ids,
            "max_ids": self.max_ids,
            "empty_files": list(self.empty_files),
            "archive_mismatches": list(self.archive_mismatches),
        }


@dataclass(frozen=True)
class ArchivePreprocessPlan:
    """Rendered archive-mode recipe/preprocess artifact paths and counts."""

    recipe_dir: Path
    output_dir: Path
    dry_run: bool
    archives: tuple[str, ...]
    array_range: str
    artifact_paths: tuple[Path, ...]
    allowlists: AllowlistSummary | None = None

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable plan data."""
        return {
            "recipe_dir": str(self.recipe_dir),
            "output_dir": str(self.output_dir),
            "dry_run": self.dry_run,
            "archives": list(self.archives),
            "archive_count": len(self.archives),
            "array_range": self.array_range,
            "artifact_paths": [str(path) for path in self.artifact_paths],
            "allowlists": self.allowlists.to_redacted_dict() if self.allowlists is not None else None,
        }


def recipe_dir_for_spec(spec: RunSpec) -> Path:
    """Return the exact recipe directory for rendered preprocess artifacts."""
    return spec.paths.recipe_dir or (spec.paths.output_dir / "rendered_recipe")


def normalized_archive_names(archives: ArchiveCoverage | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return deterministic archive filenames with ``.tar.lz4`` suffixes."""
    raw = archives.archives if isinstance(archives, ArchiveCoverage) else tuple(archives)
    names: list[str] = []
    for value in raw:
        name = Path(str(value)).name
        if not name.endswith(".tar.lz4"):
            name = f"{name}.tar.lz4"
        names.append(name)
    return tuple(sorted(dict.fromkeys(names)))


_MISSING = object()


def _optional_attr(source: object, name: str, default: object = _MISSING) -> object:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _set_optional(config: dict[str, Any], source: object, name: str, *, key: str | None = None) -> None:
    value = _optional_attr(source, name)
    if value is not _MISSING and value is not None:
        config[key or name] = _config_value(value)


def _config_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_config_value(item) for item in value]
    if isinstance(value, list):
        return [_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _config_value(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _config_value(model_dump())
    return value


def build_recipe_config(spec: RunSpec, archives: ArchiveCoverage | list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Build a legacy-compatible recipe config from a RunSpec."""
    archive_names = normalized_archive_names(archives)
    shard_config = compute_archive_shard_config(list(archive_names), spec.worker.shards_per_archive)
    gpu = spec.resources.get("gpu_worker")
    # Validate the toolkit override/mount mapping (raises ToolkitOverrideNotFoundError
    # on a declared-but-unmapped override); the resolved value is unused here.
    resolve_toolkit_for_container(spec)
    paths = {
        "input_dir": str(spec.paths.staging_dir.parent),
        "output_dir": str(spec.paths.output_dir),
        "log_dir": str(spec.paths.log_dir),
        "manifest_csv": str(spec.references.manifest_csv),
        "uniprot_db": str(spec.references.uniprot_duckdb),
        "staging_dir": str(spec.paths.staging_dir),
    }
    _set_optional(paths, spec.references, "heterodimer_id_manifest")

    worker = {
        "stages": spec.worker.stages,
        "workers": spec.worker.workers,
        "tool_used": spec.worker.tool_used,
        "heterodimers": bool(_optional_attr(spec.worker, "heterodimers", False)),
        "parallel_stages": bool(_optional_attr(spec.worker, "parallel_stages", False)),
        "batch_size": spec.worker.batch_size,
        "local_scratch": spec.worker.local_scratch,
        "scratch_dir": str(spec.worker.scratch_dir),
        "duckdb_memory_limit": spec.worker.duckdb_memory_limit,
    }
    for name in (
        "clash_device",
        "clash_batch_size",
        "dssp_algorithm",
        "retry_failed_only",
        "retry_metadata_delta_tag",
    ):
        _set_optional(worker, spec.worker, name)

    upload = {
        "self_upload": spec.worker.self_upload,
        "s3_prefix": spec.storage.s3_output_prefix if spec.worker.self_upload else "",
        "s5cmd_path": str(spec.worker.s5cmd_path),
        "max_upload_slots": spec.worker.upload_slots,
        "s5cmd_numworkers": spec.worker.s5cmd_numworkers,
    }
    upload_key_by_field = {
        "upload_mode": "mode",
        "s3_tar_prefix": "tar_prefix",
        "s3_tar_manifest_csv": "tar_manifest_csv",
        "local_tar_dir": "local_tar_dir",
        "local_tar_manifest_csv": "local_tar_manifest_csv",
        "tar_compression": "tar_compression",
    }
    for name, key in upload_key_by_field.items():
        _set_optional(upload, spec.storage, name, key=key)

    config: dict[str, Any] = {
        "cluster": spec.cluster.name,
        "job_name": f"bspp_{spec.dataset.name}",
        "run_name": spec.dataset.name,
        "slurm": {
            "partition": gpu.partition if gpu else None,
            "account": spec.cluster.account,
            "array_range": shard_config["array_range"],
            "cpus_per_task": gpu.cpus_per_task if gpu else 30,
            "memory": gpu.memory if gpu else "128G",
            "gres": gpu.gres if gpu else "gpu:1",
            "time": gpu.time if gpu else "04:00:00",
        },
        "paths": paths,
        "worker": worker,
        "environment": {
            "python_env": "",
            "modules": "",
        },
        "upload": upload,
        "monitoring": {
            "enabled": True,
            "refresh_interval": 10,
            "alert_threshold": 5.0,
        },
    }
    analysis_metadata = _optional_attr(spec, "analysis_metadata")
    if analysis_metadata is not _MISSING and analysis_metadata is not None:
        config["analysis_metadata"] = _config_value(analysis_metadata)
    return config


def render_archive_preprocess_artifacts(
    spec: RunSpec,
    archives: ArchiveCoverage | list[str] | tuple[str, ...],
    *,
    dry_run: bool = True,
    write_allowlists: bool = True,
) -> ArchivePreprocessPlan:
    """Render or plan archive-mode recipe, preprocess artifacts, and reports."""
    archive_names = normalized_archive_names(archives)
    shard_config = compute_archive_shard_config(list(archive_names), spec.worker.shards_per_archive)
    recipe_dir = recipe_dir_for_spec(spec)
    artifact_paths = (
        recipe_dir / "config.yaml",
        spec.paths.output_dir / "archive_list.json",
        spec.paths.output_dir / "shard_config.json",
        spec.paths.output_dir / "dataset_config.json",
        spec.paths.output_dir / "provider.json",
        spec.paths.output_dir / "wp4" / "preprocess_summary.json",
    )
    allowlists = None
    if not dry_run:
        recipe_dir.mkdir(parents=True, exist_ok=True)
        _write_recipe_config(spec, recipe_dir / "config.yaml", build_recipe_config(spec, archive_names))
        _write_json(list(archive_names), spec.paths.output_dir / "archive_list.json")
        if write_allowlists:
            allowlists = render_allowlists(spec, archive_names, dry_run=False)
            if allowlists.files > 0 and allowlists.total_ids == 0:
                msg = (
                    "rendered allowlists contain zero IDs for every archive; "
                    "use --no-allowlists for archive-inventory-only runs or fix the RunSpec dataset/tracking source"
                )
                raise ValueError(msg)
            shard_config["has_allowlists"] = True
        _write_json(shard_config, spec.paths.output_dir / "shard_config.json")
        write_dataset_config(
            spec.worker.tool_used,
            spec.worker.provider_id,
            spec.paths.output_dir / "dataset_config.json",
            model_created_date=_utc_now(),
        )
        write_provider_json(
            spec.worker.provider_id,
            spec.worker.provider_name,
            spec.paths.output_dir / "provider.json",
            provider_url=spec.worker.provider_url,
            copyrights=list(spec.worker.provider_copyrights),
        )
    elif write_allowlists:
        allowlists = render_allowlists(spec, archive_names, dry_run=True)
        if allowlists.files > 0 and allowlists.total_ids == 0:
            msg = (
                "rendered allowlists contain zero IDs for every archive; "
                "use --no-allowlists for archive-inventory-only runs or fix the RunSpec dataset/tracking source"
            )
            raise ValueError(msg)

    plan = ArchivePreprocessPlan(
        recipe_dir=recipe_dir,
        output_dir=spec.paths.output_dir,
        dry_run=dry_run,
        archives=archive_names,
        array_range=str(shard_config["array_range"]),
        artifact_paths=artifact_paths,
        allowlists=allowlists,
    )
    if not dry_run:
        report = {
            "run_id": spec.dataset.run_id,
            "dataset": spec.dataset.name,
            "source_runspec": str(spec.source_path) if spec.source_path is not None else None,
            "source_hash": spec.source_hash,
            "plan": plan,
        }
        write_json_report(report, spec.paths.output_dir / "wp4" / "preprocess_summary.json")
        write_text_summary(report, spec.paths.output_dir / "wp4" / "preprocess_summary.txt")
    return plan


def render_allowlists(
    spec: RunSpec,
    archives: list[str] | tuple[str, ...],
    *,
    dry_run: bool = True,
) -> AllowlistSummary:
    """Render or plan per-archive allowlists from the tracking/master parquet."""
    archive_names = normalized_archive_names(tuple(archives))
    ids_by_archive = _load_allowlist_ids(spec.references.tracking_parquet, spec.dataset.name)
    if not ids_by_archive:
        ids_by_archive = _load_allowlist_ids(spec.references.master_parquet, spec.dataset.name)
    output_dir = spec.paths.output_dir / "allowlists"
    counts: list[int] = []
    empty_files: list[str] = []
    files: list[tuple[str, list[str]]] = []
    normalized_keys = {_archive_key(name) for name in archive_names}
    mismatches = sorted(set(ids_by_archive) - normalized_keys)
    for archive_name in archive_names:
        ids = sorted(ids_by_archive.get(_archive_key(archive_name), set()))
        counts.append(len(ids))
        if not ids:
            empty_files.append(archive_name)
        files.append((archive_name, ids))
    if files and sum(counts) == 0:
        msg = (
            "rendered allowlists contain zero IDs for every archive; "
            "use --no-allowlists for archive-inventory-only runs or fix the RunSpec dataset/tracking source"
        )
        raise ValueError(msg)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        for archive_name, ids in files:
            (output_dir / f"{archive_name}.txt").write_text("\n".join(ids) + "\n" if ids else "")
    return AllowlistSummary(
        directory=output_dir,
        files=len(archive_names),
        total_ids=sum(counts),
        min_ids=min(counts) if counts else 0,
        max_ids=max(counts) if counts else 0,
        empty_files=tuple(empty_files),
        archive_mismatches=tuple(mismatches),
    )


def render_archive_preprocess_artifacts_report(spec: RunSpec, plan: ArchivePreprocessPlan) -> str:
    """Render a deterministic JSON report for an archive preprocess plan."""
    return report_to_json(
        {
            "run_id": spec.dataset.run_id,
            "dataset": spec.dataset.name,
            "source_runspec": str(spec.source_path) if spec.source_path is not None else None,
            "source_hash": spec.source_hash,
            "plan": plan,
        }
    )


def _write_recipe_config(spec: RunSpec, path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render_artifact_header(spec, "recipe/config.yaml") + yaml.safe_dump(config, sort_keys=False)
    _write_text(payload, path)


def _write_json(value: object, path: Path) -> None:
    _write_text(json.dumps(value, indent=2) + "\n", path)


def _write_text(payload: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _load_allowlist_ids(parquet_path: Path, dataset: str) -> dict[str, set[str]]:
    if not parquet_path.exists():
        return {}
    schema_names = _parquet_schema_names(parquet_path)
    if ARCHIVE_COLUMN not in schema_names:
        return {}
    dataset_column = _first_present(DATASET_COLUMNS, schema_names)
    model_columns = tuple(name for name in _MODEL_ID_COLUMNS if name in schema_names)
    if not model_columns:
        return {}

    table = pq.read_table(
        parquet_path,
        columns=_unique_columns((dataset_column, ARCHIVE_COLUMN, *model_columns)),
        filters=[(dataset_column, "=", dataset)] if dataset_column is not None else None,
    )
    archives = table.column(ARCHIVE_COLUMN).to_pylist()
    model_values = {column: table.column(column).to_pylist() for column in model_columns}
    result: dict[str, set[str]] = {}
    for row_index, archive in enumerate(archives):
        if not archive:
            continue
        model_id = _model_id_from_columns(model_values, row_index)
        if not model_id:
            continue
        result.setdefault(_archive_key(str(archive)), set()).add(model_id)
    return result


def _row_dataset(row: dict[str, object]) -> str | None:
    for key in ("dataset_name", "source_run", "dataset"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _row_model_id(row: dict[str, object]) -> str | None:
    for key in ("model_entity_id", "af_id", "model_id"):
        value = row.get(key)
        if value:
            match = _AF_ID_PATTERN.search(str(value))
            return match.group(0) if match else str(value)
    msa_path = row.get("msa_path")
    if msa_path:
        match = _AFDB_MSA_PATTERN.search(str(msa_path)) or _AF_ID_PATTERN.search(str(msa_path))
        if match:
            return match.group(1) if match.lastindex else match.group(0)
    return None


def _model_id_from_columns(model_values: dict[str, list[object]], row_index: int) -> str | None:
    for key in ("model_entity_id", "af_id", "model_id"):
        values = model_values.get(key)
        value = values[row_index] if values is not None else None
        if value:
            match = _AF_ID_PATTERN.search(str(value))
            return match.group(0) if match else str(value)
    values = model_values.get("msa_path")
    msa_path = values[row_index] if values is not None else None
    if msa_path:
        match = (
            _AFDB_MSA_PATTERN.search(str(msa_path))
            or _AF_ID_PATTERN.search(str(msa_path))
            or _PDB_ASSEMBLY_MSA_PATTERN.search(str(msa_path))
        )
        if match:
            return match.group(1) if match.lastindex else match.group(0)
    return None


def _parquet_schema_names(path: Path) -> tuple[str, ...]:
    return tuple(str(name) for name in pq.ParquetFile(path).schema_arrow.names)


def _first_present(candidates: tuple[str, ...], names: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _unique_columns(columns: tuple[str | None, ...]) -> list[str]:
    result: list[str] = []
    for column in columns:
        if column is not None and column not in result:
            result.append(column)
    return result


def _archive_key(value: str) -> str:
    name = Path(value).name
    return name.removesuffix(".tar.lz4")


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "AllowlistSummary",
    "ArchivePreprocessPlan",
    "build_recipe_config",
    "normalized_archive_names",
    "recipe_dir_for_spec",
    "render_allowlists",
    "render_archive_preprocess_artifacts",
    "render_archive_preprocess_artifacts_report",
]
