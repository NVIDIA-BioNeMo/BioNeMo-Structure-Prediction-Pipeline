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

"""Stage dispatcher for uploading post-processed outputs.

Three stages, matching the legacy ``upload_and_track.py``:

=============  ===========================  ======================
 Stage          Source                       Destination / tools
=============  ===========================  ======================
 ``s3``         Lustre ``success_outputs``   S3 (s5cmd|dm)
 ``gcs``        S3 prefix         GCS bucket (gcloud|dm)
 ``gcs-direct`` Lustre ``success_outputs``   GCS bucket (gcloud only)
=============  ===========================  ======================

Each stage builds the relevant tool's command (prefix-list JSON for dm,
command file for s5cmd, argv for gcloud) via the subprocess wrappers in
:mod:`bspp.orchestration.runtime.data_movement`, optionally executes it, and
returns a :class:`StagePlan` describing what was done. Tracking parquet
updates remain the caller's responsibility — the dispatcher reports
success but does not mutate the parquet so callers can gate the update
on side-signals (e.g. dm job completion).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from bspp.orchestration.contract.data_placement import (
    DATA_PLACEMENT_TOOLS_BY_STAGE,
    DataPlacementRecord,
    DataPlacementStage,
    DataPlacementTool,
    validate_data_placement_tool_for_stage,
)
from bspp.orchestration.runtime.data_movement.backends import BackendUnavailableError, get_backend
from bspp.orchestration.runtime.data_movement.common import PlannedTransfer, TransferResult, format_argv
from bspp.orchestration.runtime.data_movement.gcs import transfer as gcs_transfer
from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer
from bspp.orchestration.runtime.postprocessing.failure_adapter import raise_transport_failure_if_audited
from bspp.orchestration.runtime.postprocessing.sharding import (
    iter_in_range_shard_dirs,
    read_required_shards,
)

Stage = DataPlacementStage
Tool = DataPlacementTool
ALLOWED_TOOLS = DATA_PLACEMENT_TOOLS_BY_STAGE


def _require_destination_prefix(value: str | None, scheme: str, *, name: str) -> str:
    """Require a non-empty ``scheme://`` destination prefix.

    The legacy internal destination defaults were removed. Every upload
    boundary must receive an explicit destination; a missing or malformed value
    fails closed before any file is written or backend is discovered.
    """
    if value is None or not value.strip():
        msg = f"{name} is required (a {scheme}:// destination prefix); no default is provided"
        raise ValueError(msg)
    return value


def _resolve_dm_backend() -> ModuleType:
    try:
        return get_backend("dm")
    except BackendUnavailableError as exc:
        raise BackendUnavailableError(
            "dm",
            hint=(
                "the NVIDIA Data Mover (dm) is not installed in this build; "
                "select --tool s5cmd (s3) or --tool gcloud (gcs) instead."
            ),
        ) from exc


class InvalidToolForStageError(ValueError):
    """Raised when a tool is not allowed for the requested stage."""


@dataclass(frozen=True)
class StagePlan:
    """Description of one stage's prepared transfers (possibly already run)."""

    stage: Stage
    tool: Tool
    dataset: str
    source: str | None = None
    destination: str | None = None
    commands: tuple[PlannedTransfer, ...] = field(default_factory=tuple)
    results: tuple[TransferResult, ...] = field(default_factory=tuple)
    shards_skipped: int = 0
    shards_queued: int = 0
    auxiliary_paths: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def executed(self) -> bool:
        return bool(self.results)

    @property
    def all_ok(self) -> bool:
        return self.executed and all(r.ok for r in self.results)

    def to_data_placement_record(
        self,
        *,
        payload_bytes_moved: bool | None = None,
        evidence_path: str | Path | None = None,
        commands: tuple[tuple[str, ...], ...] | None = None,
        job_id: str | None = None,
        terminal_status: str | None = None,
    ) -> DataPlacementRecord:
        """Return the shared contract evidence record for this stage."""
        if self.source is None or self.destination is None:
            msg = "StagePlan requires source and destination to render a DataPlacementRecord"
            raise ValueError(msg)
        return DataPlacementRecord(
            stage=self.stage,
            tool=self.tool,
            dataset=self.dataset,
            source=self.source,
            destination=self.destination,
            payload_bytes_moved=self.all_ok if payload_bytes_moved is None else payload_bytes_moved,
            evidence_path=str(evidence_path) if evidence_path is not None else None,
            commands=tuple(command.argv for command in self.commands) if commands is None else commands,
            job_id=job_id,
            terminal_status=terminal_status,
        )


def validate_tool_for_stage(tool: Tool, stage: Stage) -> None:
    """Raise :class:`InvalidToolForStageError` if *tool* is not valid for *stage*."""
    try:
        validate_data_placement_tool_for_stage(tool, stage)
    except ValueError as exc:
        raise InvalidToolForStageError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Shared filesystem helpers
# ---------------------------------------------------------------------------


def find_success_outputs(dataset_output_dir: Path) -> tuple[list[Path], int]:
    """Find ``shard_*/success_outputs`` dirs with content; skip self-uploaded.

    A shard directory is considered already self-uploaded if it contains
    a ``.uploaded`` marker file. Those are excluded from the bulk-upload
    list and counted separately.

    Stale ``shard_N/`` directories whose id is ``>= required_shards``
    (from ``shard_config.json``) are silently skipped — uploading stale
    leftovers from a prior larger run would contaminate the S3
    destination. Use ``validate count`` to surface stale anomalies;
    this helper is strictly the transfer path.

    Returns ``(dirs_to_upload, skipped_count)``.
    """
    dirs_to_upload: list[Path] = []
    skipped = 0
    required_shards = read_required_shards(dataset_output_dir)
    for _shard_id, shard_dir in iter_in_range_shard_dirs(dataset_output_dir, required_shards):
        success_dir = shard_dir / "success_outputs"
        if not success_dir.is_dir():
            continue
        if (shard_dir / ".uploaded").exists():
            skipped += 1
            continue
        if any(success_dir.iterdir()):
            dirs_to_upload.append(success_dir)
    return dirs_to_upload, skipped


def _ensure_trailing_slash(value: str) -> str:
    return value if not value or value.endswith("/") else f"{value}/"


def _split_uri(value: str, scheme: str) -> tuple[str, str]:
    expected = f"{scheme}://"
    if not value.startswith(expected):
        msg = f"Expected {expected} destination prefix, got {value}"
        raise ValueError(msg)
    raw = value.removeprefix(expected)
    if "/" not in raw:
        return raw, ""
    bucket, prefix = raw.split("/", 1)
    return bucket, _ensure_trailing_slash(prefix)


def _s3_bucket_and_prefix(value: str) -> tuple[str, str]:
    return _split_uri(value, "s3")


def _gcs_bucket_and_prefix(value: str) -> tuple[str, str]:
    return _split_uri(value, "gs")


def _s3_destination(prefix: str) -> tuple[str, str, str]:
    bucket, rel_prefix = _s3_bucket_and_prefix(prefix)
    return bucket, rel_prefix, f"s3://{bucket}/{rel_prefix}"


def _gcs_destination(prefix: str) -> tuple[str, str, str]:
    bucket, rel_prefix = _gcs_bucket_and_prefix(prefix)
    return bucket, rel_prefix, f"gs://{bucket}/{rel_prefix}"


def _dm_gcs_destination_prefix(source_rel: str, destination_rel: str) -> str:
    if destination_rel == source_rel:
        return "/"
    return f"/{destination_rel}" if destination_rel else "/"


# ---------------------------------------------------------------------------
# Stage s3 — Lustre -> S3
# ---------------------------------------------------------------------------


def upload_s3(
    dataset: str,
    dataset_output_dir: Path,
    *,
    tool: Tool = "s5cmd",
    data_dir: Path | None = None,
    s3_destination_prefix: str,
    execute: bool = False,
    dry_run: bool = False,
) -> StagePlan:
    """Plan (and optionally execute) the Lustre -> S3 upload.

    *data_dir* is where the generated prefix-list / command-file assets
    land; it defaults to ``<dataset_output_dir>/upload_logs_<dataset>``.
    ``s3_destination_prefix`` is mandatory: there is no legacy
    internal default.
    """
    validate_tool_for_stage(tool, "s3")
    s3_destination_prefix = _require_destination_prefix(s3_destination_prefix, "s3", name="s3_destination_prefix")
    dirs, skipped = find_success_outputs(dataset_output_dir)
    log_dir = data_dir if data_dir is not None else dataset_output_dir / f"upload_logs_{dataset}"
    bucket, s3_prefix, s3_url = _s3_destination(s3_destination_prefix)
    source = str(dataset_output_dir)

    if tool == "dm":
        dm = _resolve_dm_backend()
        destination = dm.DMEndpoint(f"{dm.SWIFTSTACK_PDX_ALIAS}:{bucket}")
        return _s3_via_dm(
            dataset,
            dirs,
            skipped,
            log_dir,
            s3_prefix,
            destination,
            dm=dm,
            source=source,
            destination_record=s3_url,
            execute=execute,
            dry_run=dry_run,
        )
    return _s3_via_s5cmd(
        dataset,
        dirs,
        skipped,
        log_dir,
        s3_url,
        source=source,
        destination_record=s3_url,
        execute=execute,
        dry_run=dry_run,
    )


def _s3_via_dm(
    dataset: str,
    dirs: Sequence[Path],
    skipped: int,
    log_dir: Path,
    s3_prefix: str,
    destination: object,
    *,
    dm: ModuleType,
    source: str,
    destination_record: str,
    execute: bool,
    dry_run: bool,
) -> StagePlan:
    commands: list[PlannedTransfer] = []
    results: list[TransferResult] = []
    aux: list[Path] = []

    chunks = dm.chunk_prefixes(dirs)
    for idx, chunk in enumerate(chunks, start=1):
        suffix = f"_part{idx}" if len(chunks) > 1 else ""
        prefix_list_path = log_dir / f"prefix_list_{dataset}{suffix}.json"
        if not dry_run:
            dm.write_prefix_list(chunk, [s3_prefix] * len(chunk), prefix_list_path)
            aux.append(prefix_list_path)
        planned = dm.job_copy(
            dm.DMEndpoint("local-filesystem"),
            destination,
            prefix_list_path,
            srun_extra_args="--mem=0",
            dry_run=True,
        )
        assert isinstance(planned, PlannedTransfer)
        commands.append(planned)
        if execute and not dry_run:
            res = dm.job_copy(
                dm.DMEndpoint("local-filesystem"),
                destination,
                prefix_list_path,
                srun_extra_args="--mem=0",
            )
            assert isinstance(res, TransferResult)
            raise_transport_failure_if_audited(res)
            results.append(res)

    return StagePlan(
        stage="s3",
        tool="dm",
        dataset=dataset,
        source=source,
        destination=destination_record,
        commands=tuple(commands),
        results=tuple(results),
        shards_skipped=skipped,
        shards_queued=len(dirs),
        auxiliary_paths=tuple(aux),
    )


def _s3_via_s5cmd(
    dataset: str,
    dirs: Sequence[Path],
    skipped: int,
    log_dir: Path,
    s3_url: str,
    *,
    source: str,
    destination_record: str,
    execute: bool,
    dry_run: bool,
) -> StagePlan:
    pairs = [(f"{d}/*", s3_url) for d in dirs]
    cmd_file = log_dir / f"s5cmd_upload_{dataset}.txt"

    if not dry_run and dirs:
        s3_transfer.write_cp_command_file(pairs, cmd_file)

    planned = s3_transfer.run_command_file(cmd_file, dry_run=True)
    assert isinstance(planned, PlannedTransfer)

    results: list[TransferResult] = []
    if execute and not dry_run and dirs:
        res = s3_transfer.run_command_file(cmd_file)
        assert isinstance(res, TransferResult)
        raise_transport_failure_if_audited(res)
        results.append(res)

    return StagePlan(
        stage="s3",
        tool="s5cmd",
        dataset=dataset,
        source=source,
        destination=destination_record,
        commands=(planned,),
        results=tuple(results),
        shards_skipped=skipped,
        shards_queued=len(dirs),
        auxiliary_paths=(cmd_file,) if not dry_run and dirs else (),
    )


# ---------------------------------------------------------------------------
# Stage gcs — S3 -> GCS
# ---------------------------------------------------------------------------


def upload_gcs(
    dataset: str,
    *,
    tool: Tool = "gcloud",
    data_dir: Path,
    s3_source_prefix: str,
    gcs_destination_prefix: str,
    execute: bool = False,
    dry_run: bool = False,
) -> StagePlan:
    """Plan (and optionally execute) S3 -> GCS.

    Both ``s3_source_prefix`` and ``gcs_destination_prefix`` are mandatory:
    there is no legacy internal default.
    """
    validate_tool_for_stage(tool, "gcs")
    s3_source_prefix = _require_destination_prefix(s3_source_prefix, "s3", name="s3_source_prefix")
    gcs_destination_prefix = _require_destination_prefix(gcs_destination_prefix, "gs", name="gcs_destination_prefix")
    source_bucket, ss_prefix = _s3_bucket_and_prefix(s3_source_prefix)
    gcs_bucket, gcs_prefix, gcs_url = _gcs_destination(gcs_destination_prefix)

    if tool == "dm":
        dm = _resolve_dm_backend()
        prefix_list_path = data_dir / f"gcs_prefix_list_{dataset}.json"
        dm_destination_prefix = _dm_gcs_destination_prefix(ss_prefix, gcs_prefix)
        if not dry_run:
            dm.write_prefix_list([ss_prefix], [dm_destination_prefix], prefix_list_path)

        planned = dm.job_copy(
            dm.DMEndpoint(f"{dm.SWIFTSTACK_SMS_ALIAS}:{source_bucket}"),
            dm.DMEndpoint(f"{dm.GCS_SMS_ALIAS}:{gcs_bucket}"),
            prefix_list_path,
            cluster=dm.GCS_CLUSTER,
            dry_run=True,
        )
        assert isinstance(planned, PlannedTransfer)
        results: list[TransferResult] = []
        if execute and not dry_run:
            res = dm.job_copy(
                dm.DMEndpoint(f"{dm.SWIFTSTACK_SMS_ALIAS}:{source_bucket}"),
                dm.DMEndpoint(f"{dm.GCS_SMS_ALIAS}:{gcs_bucket}"),
                prefix_list_path,
                cluster=dm.GCS_CLUSTER,
            )
            assert isinstance(res, TransferResult)
            raise_transport_failure_if_audited(res)
            results.append(res)
        return StagePlan(
            stage="gcs",
            tool="dm",
            dataset=dataset,
            source=s3_source_prefix,
            destination=gcs_url,
            commands=(planned,),
            results=tuple(results),
            auxiliary_paths=(prefix_list_path,) if not dry_run else (),
        )

    # tool == "gcloud" — client-side S3 list+download→GCS upload isn't
    # supported by the public gcloud adapter (gcloud storage rsync has no S3
    # source support; dm does server-side). Fail closed rather than pass an
    # s3:// URI to gcloud (which would fail at execution time). Operators must
    # use the internal dm backend, or stage data locally and use
    # upload-gcs-direct (Lustre → GCS).
    raise ValueError(
        "client-side S3→GCS is not supported by the public gcloud adapter; "
        "select --tool dm (internal) or stage locally and use upload-gcs-direct"
    )


# ---------------------------------------------------------------------------
# Stage gcs-direct — Lustre -> GCS (bypassing S3)
# ---------------------------------------------------------------------------


def upload_gcs_direct(
    dataset: str,
    dataset_output_dir: Path,
    *,
    tool: Tool = "gcloud",
    gcs_destination_prefix: str,
    execute: bool = False,
    dry_run: bool = False,
) -> StagePlan:
    """Plan (and optionally execute) Lustre -> GCS via gcloud rsync.

    ``gcs_destination_prefix`` is mandatory: there is no legacy
    internal default. Each shard directory is rsynced to the destination.
    """
    validate_tool_for_stage(tool, "gcs-direct")
    gcs_destination_prefix = _require_destination_prefix(gcs_destination_prefix, "gs", name="gcs_destination_prefix")
    dirs, skipped = find_success_outputs(dataset_output_dir)
    _gcs_bucket, _gcs_prefix, gcs_url = _gcs_destination(gcs_destination_prefix)

    commands: list[PlannedTransfer] = []
    results: list[TransferResult] = []
    for success_dir in dirs:
        planned = gcs_transfer.rsync(success_dir, gcs_url, dry_run=True)
        assert isinstance(planned, PlannedTransfer)
        commands.append(planned)
        if execute and not dry_run:
            res = gcs_transfer.rsync(success_dir, gcs_url)
            assert isinstance(res, TransferResult)
            raise_transport_failure_if_audited(res)
            results.append(res)

    return StagePlan(
        stage="gcs-direct",
        tool="gcloud",
        dataset=dataset,
        source=str(dataset_output_dir),
        destination=gcs_url,
        commands=tuple(commands),
        results=tuple(results),
        shards_skipped=skipped,
        shards_queued=len(dirs),
    )


def render_plan(plan: StagePlan) -> str:
    """Human-readable multi-line summary for CLI output."""
    lines = [
        f"Stage: {plan.stage}",
        f"Tool: {plan.tool}",
        f"Dataset: {plan.dataset}",
        f"Shards queued: {plan.shards_queued}",
        f"Shards skipped (already self-uploaded): {plan.shards_skipped}",
        "Commands:",
    ]
    for cmd in plan.commands:
        lines.append(f"  {format_argv(cmd.argv)}")
    if plan.results:
        lines.append("Results:")
        for res in plan.results:
            lines.append(f"  rc={res.returncode} elapsed={res.elapsed_s:.1f}s")
    return "\n".join(lines)


__all__ = [
    "ALLOWED_TOOLS",
    "InvalidToolForStageError",
    "Stage",
    "StagePlan",
    "Tool",
    "find_success_outputs",
    "render_plan",
    "upload_gcs",
    "upload_gcs_direct",
    "upload_s3",
    "validate_tool_for_stage",
]
