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

"""RunSpec-driven GCS preflight and delivery-boundary reports."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.contract.secrets import ResolvedSecret
from bspp.orchestration.runtime.data_movement.backends import BackendUnavailableError, get_backend
from bspp.orchestration.runtime.data_movement.common import PlannedTransfer
from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary
from bspp.orchestration.runtime.slurm.arrays import array_task_count


def _resolve_dm_backend() -> ModuleType:
    try:
        return get_backend("dm")
    except BackendUnavailableError as exc:
        raise BackendUnavailableError(
            "dm",
            hint="the NVIDIA Data Mover (dm) is not installed in this build; select --tool gcloud instead.",
        ) from exc


@dataclass(frozen=True)
class PrefixListPlan:
    """Prefix-list artifact rendered for a dm GCS copy."""

    path: Path
    origin: tuple[str, ...]
    destination: tuple[str, ...]

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable prefix-list data."""
        return {
            "path": str(self.path),
            "origin": list(self.origin),
            "destination": list(self.destination),
        }


@dataclass(frozen=True)
class GCSPreflightReport:
    """Dry-run GCS delivery boundary report."""

    dataset: str
    run_id: str
    source_prefix: str
    destination_prefix: str | None
    tool: str
    dry_run_only: bool
    production_prefixes_allowed: bool
    expected_objects: int | None
    commands: tuple[PlannedTransfer, ...]
    cleanup_commands: tuple[PlannedTransfer, ...]
    prefix_list: PrefixListPlan | None
    secret_statuses: tuple[ResolvedSecret, ...]
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.destination_prefix is not None and not self.blockers

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable preflight report data."""
        return {
            "dataset": self.dataset,
            "run_id": self.run_id,
            "source_prefix": self.source_prefix,
            "destination_prefix": self.destination_prefix,
            "tool": self.tool,
            "dry_run_only": self.dry_run_only,
            "production_prefixes_allowed": self.production_prefixes_allowed,
            "expected_objects": self.expected_objects,
            "ready": self.ready,
            "blockers": list(self.blockers),
            "commands": [_planned_transfer_to_dict(command) for command in self.commands],
            "cleanup_commands": [_planned_transfer_to_dict(command) for command in self.cleanup_commands],
            "prefix_list": self.prefix_list.to_redacted_dict() if self.prefix_list is not None else None,
            "secret_statuses": [status.as_redacted_mapping() for status in self.secret_statuses],
        }


def build_gcs_preflight_report(
    spec: RunSpec,
    *,
    tool: str = "gcloud",
    write_artifacts: bool = False,
    strict_secrets: bool = False,
    environ: dict[str, str] | None = None,
) -> GCSPreflightReport:
    """Build a dry-run GCS preflight report from *spec*."""
    if tool not in {"dm", "gcloud"}:
        msg = f"Unsupported GCS preflight tool: {tool}"
        raise ValueError(msg)
    secret_statuses = (
        spec.validate_required_secrets(environ=environ) if strict_secrets else spec.resolve_secrets(environ=environ)
    )
    destination = spec.storage.gcs_destination_prefix
    blockers: list[str] = []
    commands: list[PlannedTransfer] = []
    cleanup_commands: list[PlannedTransfer] = []
    prefix_list = None

    if destination is None:
        blockers.append("storage.gcs_destination_prefix is not configured")
    else:
        source_rel = _s3_relative_prefix(spec.storage.s3_output_prefix)
        bucket, destination_rel = _gs_bucket_and_prefix(destination)
        if tool == "dm":
            dm = _resolve_dm_backend()
            if bucket != dm.DM_GCS_BUCKET:
                blockers.append(f"dm GCS preflight supports gs://{dm.DM_GCS_BUCKET}/..., got gs://{bucket}/...")
            prefix_list = PrefixListPlan(
                path=spec.paths.output_dir / "wp7" / f"gcs_prefix_list_{spec.dataset.name}.json",
                origin=(source_rel,),
                destination=(_dm_destination_prefix(source_rel, destination_rel),),
            )
            commands.append(_dm_command(dm, prefix_list.path))
        else:
            commands.append(
                PlannedTransfer(
                    tool="gcloud",
                    argv=(
                        "gcloud",
                        "storage",
                        "rsync",
                        "--recursive",
                        spec.storage.s3_output_prefix,
                        destination,
                    ),
                    note="S3-to-GCS dry run preview",
                )
            )
        cleanup_commands.append(
            PlannedTransfer(
                tool="gcloud",
                argv=("gcloud", "storage", "rm", "--recursive", _cleanup_glob(destination)),
                note="isolated-prefix cleanup preview",
            )
        )

    if write_artifacts and prefix_list is not None:
        _write_prefix_list(prefix_list)

    return GCSPreflightReport(
        dataset=spec.dataset.name,
        run_id=spec.dataset.run_id,
        source_prefix=spec.storage.s3_output_prefix,
        destination_prefix=destination,
        tool=tool,
        dry_run_only=spec.validation.gcs_dry_run_only,
        production_prefixes_allowed=spec.storage.allow_production_prefixes,
        expected_objects=_expected_objects(spec),
        commands=tuple(commands),
        cleanup_commands=tuple(cleanup_commands),
        prefix_list=prefix_list,
        secret_statuses=secret_statuses,
        blockers=tuple(blockers),
    )


def render_gcs_preflight_report(report: GCSPreflightReport) -> str:
    """Render a deterministic JSON GCS preflight report."""
    return report_to_json(report)


def write_gcs_preflight_report(report: GCSPreflightReport, output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and text GCS preflight reports under *output_dir*."""
    json_path = write_json_report(report, output_dir / "wp7" / "gcs_preflight_report.json")
    text_path = write_text_summary(report, output_dir / "wp7" / "gcs_preflight_report.txt")
    return json_path, text_path


def _planned_transfer_to_dict(command: PlannedTransfer) -> dict[str, object]:
    return {"tool": command.tool, "argv": list(command.argv), "note": command.note}


def _dm_command(dm: ModuleType, prefix_list_path: Path) -> PlannedTransfer:
    return PlannedTransfer(
        tool="dm",
        argv=(
            "dm",
            "job",
            "copy",
            "--cluster",
            dm.DM_GCS_CLUSTER,
            dm.DM_SWIFTSTACK_SOURCE,
            dm.DM_GCS_DESTINATION,
            "-y",
            "--prefix-list",
            str(prefix_list_path),
        ),
        note="S3-to-GCS dry run preview",
    )


def _write_prefix_list(plan: PrefixListPlan) -> None:
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"origin": list(plan.origin), "destination": list(plan.destination)}
    tmp_path = plan.path.with_name(plan.path.name + ".tmp")
    with tmp_path.open("w") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, plan.path)


def _s3_relative_prefix(value: str) -> str:
    prefix = value.removeprefix("s3://")
    if "/" not in prefix:
        return ""
    return _ensure_trailing_slash(prefix.split("/", 1)[1])


def _gs_bucket_and_prefix(value: str) -> tuple[str, str]:
    if not value.startswith("gs://"):
        msg = f"Expected gs:// GCS destination prefix, got {value}"
        raise ValueError(msg)
    raw = value.removeprefix("gs://")
    if "/" not in raw:
        return raw, ""
    bucket, prefix = raw.split("/", 1)
    return bucket, _ensure_trailing_slash(prefix)


def _dm_destination_prefix(source_rel: str, destination_rel: str) -> str:
    if destination_rel == source_rel:
        return "/"
    return f"/{destination_rel}" if destination_rel else "/"


def _cleanup_glob(destination: str) -> str:
    return f"{_ensure_trailing_slash(destination)}**"


def _ensure_trailing_slash(value: str) -> str:
    return value if not value or value.endswith("/") else f"{value}/"


def _expected_objects(spec: RunSpec) -> int | None:
    archive_count = _archive_count_from_array(spec.dataset.array)
    if archive_count == 1:
        return spec.validation.expected_one_archive_objects
    return None


def _archive_count_from_array(value: str) -> int:
    return array_task_count(value)


__all__ = [
    "GCSPreflightReport",
    "PrefixListPlan",
    "build_gcs_preflight_report",
    "render_gcs_preflight_report",
    "write_gcs_preflight_report",
]
