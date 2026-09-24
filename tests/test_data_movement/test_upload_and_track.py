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

"""Tests for the public upload_and_track stage dispatcher.

This file covers the public ``s5cmd``/``gcloud`` paths, the required
storage-configuration fail-closed boundary, the Python API defaults
that mirror ``DATA_PLACEMENT_TOOLS_BY_STAGE``, and the structured
``BackendUnavailableError`` seam for the explicitly selectable historical
``dm`` label.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.data_movement.backends import BackendUnavailableError
from bspp.orchestration.runtime.postprocessing import upload_and_track as uat

S3_PREFIX = "s3://example-bucket/postprocessed/dsA/"
GCS_PREFIX = "gs://example-gcs-bucket/postprocessed/dsA/"


@pytest.fixture(autouse=True)
def _tools_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def which(name: str) -> str | None:
        return f"/usr/local/bin/{name}" if name in {"dm", "s5cmd", "gcloud"} else None

    monkeypatch.setattr(shutil, "which", which)
    # Placeholder S3 creds so dry-run argv assembly works in tests.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://swiftstack.test")


def _make_shard(dataset_output_dir: Path, idx: int, *, with_files: bool, uploaded: bool = False) -> None:
    shard = dataset_output_dir / f"shard_{idx}"
    success = shard / "success_outputs"
    success.mkdir(parents=True, exist_ok=True)
    if with_files:
        (success / "model.pdb").write_text("dummy")
    if uploaded:
        (shard / ".uploaded").touch()


def test_find_success_outputs_skips_uploaded_and_empty(tmp_path: Path) -> None:
    _make_shard(tmp_path, 0, with_files=True)
    _make_shard(tmp_path, 1, with_files=True, uploaded=True)
    _make_shard(tmp_path, 2, with_files=False)

    dirs, skipped = uat.find_success_outputs(tmp_path)

    assert [d.parent.name for d in dirs] == ["shard_0"]
    assert skipped == 1


def test_validate_tool_for_stage_rejects_bad_combo() -> None:
    with pytest.raises(uat.InvalidToolForStageError):
        uat.validate_tool_for_stage("s5cmd", "gcs")

    with pytest.raises(uat.InvalidToolForStageError):
        uat.validate_tool_for_stage("dm", "gcs-direct")


# ---------------------------------------------------------------------------
# Required storage configuration
# ---------------------------------------------------------------------------


def test_upload_s3_requires_destination_prefix(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dsA"
    _make_shard(dataset_dir, 0, with_files=True)

    with pytest.raises(ValueError, match="s3_destination_prefix"):
        uat.upload_s3("dsA", dataset_dir, tool="s5cmd", s3_destination_prefix="", dry_run=False)

    # Fails before any command file is written.
    assert not list(tmp_path.rglob("s5cmd_upload_*.txt"))


def test_upload_s3_rejects_omitted_destination_argument(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dsA"
    _make_shard(dataset_dir, 0, with_files=True)

    with pytest.raises(TypeError):
        uat.upload_s3("dsA", dataset_dir, tool="s5cmd", dry_run=False)

    assert not list(tmp_path.rglob("s5cmd_upload_*.txt"))


def test_upload_gcs_requires_source_and_destination(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"

    with pytest.raises(ValueError, match="s3_source_prefix"):
        uat.upload_gcs(
            "dsA",
            tool="gcloud",
            data_dir=log_dir,
            s3_source_prefix="",
            gcs_destination_prefix=GCS_PREFIX,
            dry_run=False,
        )

    with pytest.raises(ValueError, match="gcs_destination_prefix"):
        uat.upload_gcs(
            "dsA", tool="gcloud", data_dir=log_dir, s3_source_prefix=S3_PREFIX, gcs_destination_prefix="", dry_run=False
        )


def test_upload_gcs_direct_requires_destination_prefix(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dsA"
    _make_shard(dataset_dir, 0, with_files=True)

    with pytest.raises(ValueError, match="gcs_destination_prefix"):
        uat.upload_gcs_direct("dsA", dataset_dir, gcs_destination_prefix="", dry_run=False)


def test_upload_s3_rejects_malformed_destination_scheme(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dsA"
    _make_shard(dataset_dir, 0, with_files=True)

    with pytest.raises(ValueError, match="Expected s3://"):
        uat.upload_s3("dsA", dataset_dir, tool="s5cmd", s3_destination_prefix="gs://wrong/", dry_run=False)


# ---------------------------------------------------------------------------
# Public s5cmd / gcloud paths
# ---------------------------------------------------------------------------


def test_upload_s3_s5cmd_writes_command_file(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dsA"
    _make_shard(dataset_dir, 0, with_files=True)
    log_dir = tmp_path / "logs"

    plan = uat.upload_s3(
        "dsA",
        dataset_dir,
        tool="s5cmd",
        data_dir=log_dir,
        s3_destination_prefix=S3_PREFIX,
        execute=False,
        dry_run=False,
    )

    assert plan.tool == "s5cmd"
    cmd_file = next(p for p in plan.auxiliary_paths if p.name.startswith("s5cmd_upload_"))
    assert cmd_file.read_text().strip().startswith("cp ")
    assert "example-bucket" in cmd_file.read_text()


def test_upload_gcs_gcloud_fails_closed(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"

    with pytest.raises(ValueError, match="S3→GCS"):
        uat.upload_gcs(
            "dsA",
            tool="gcloud",
            data_dir=log_dir,
            s3_source_prefix=S3_PREFIX,
            gcs_destination_prefix=GCS_PREFIX,
            execute=False,
            dry_run=False,
        )


def test_upload_s3_omitted_tool_defaults_to_s5cmd_without_dm_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dsA"
    _make_shard(dataset_dir, 0, with_files=True)
    log_dir = tmp_path / "logs"

    def fail_get_backend(name: str) -> object:
        raise AssertionError(f"get_backend({name!r}) must not be called for the public default")

    monkeypatch.setattr(uat, "get_backend", fail_get_backend)

    plan = uat.upload_s3(
        "dsA",
        dataset_dir,
        data_dir=log_dir,
        s3_destination_prefix=S3_PREFIX,
        execute=False,
        dry_run=False,
    )

    assert plan.tool == "s5cmd"
    cmd_file = next(p for p in plan.auxiliary_paths if p.name.startswith("s5cmd_upload_"))
    assert cmd_file.read_text().strip().startswith("cp ")


def test_upload_gcs_omitted_tool_defaults_to_gcloud_without_dm_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_dir = tmp_path / "logs"

    def fail_get_backend(name: str) -> object:
        raise AssertionError(f"get_backend({name!r}) must not be called for the public default")

    monkeypatch.setattr(uat, "get_backend", fail_get_backend)

    with pytest.raises(ValueError, match="S3→GCS"):
        uat.upload_gcs(
            "dsA",
            data_dir=log_dir,
            s3_source_prefix=S3_PREFIX,
            gcs_destination_prefix=GCS_PREFIX,
            execute=False,
            dry_run=False,
        )


def test_upload_gcs_direct_uses_gcloud_rsync(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dsA"
    _make_shard(dataset_dir, 0, with_files=True)
    _make_shard(dataset_dir, 1, with_files=True)

    plan = uat.upload_gcs_direct("dsA", dataset_dir, gcs_destination_prefix=GCS_PREFIX, execute=False, dry_run=False)

    assert plan.stage == "gcs-direct"
    assert plan.tool == "gcloud"
    assert plan.shards_queued == 2
    assert len(plan.commands) == 2
    for cmd in plan.commands:
        assert "rsync" in cmd.argv
        assert GCS_PREFIX in cmd.argv


def test_upload_gcs_direct_accepts_custom_destination(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dsA"
    _make_shard(dataset_dir, 0, with_files=True)

    plan = uat.upload_gcs_direct(
        "dsA",
        dataset_dir,
        gcs_destination_prefix="gs://example-gcs-bucket-dev/users/test/postprocessed/dsA/",
        execute=False,
        dry_run=False,
    )

    assert len(plan.commands) == 1
    assert "gs://example-gcs-bucket-dev/users/test/postprocessed/dsA/" in plan.commands[0].argv


def test_stage_plan_renders_data_placement_contract_record(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dsA"
    _make_shard(dataset_dir, 0, with_files=True)

    plan = uat.upload_gcs_direct(
        "dsA",
        dataset_dir,
        gcs_destination_prefix=GCS_PREFIX,
        execute=False,
        dry_run=False,
    )

    record = plan.to_data_placement_record()

    assert record.stage == "gcs-direct"
    assert record.tool == "gcloud"
    assert record.dataset == "dsA"
    assert record.source == str(dataset_dir)
    assert record.destination == GCS_PREFIX
    assert record.payload_bytes_moved is False


# ---------------------------------------------------------------------------
# Backend-unavailable error seam
# ---------------------------------------------------------------------------


def test_dm_resolver_preserves_structured_backend_unavailable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dsA"
    _make_shard(dataset_dir, 0, with_files=True)

    def fail_get_backend(name: str):
        raise BackendUnavailableError(name)

    monkeypatch.setattr(uat, "get_backend", fail_get_backend)

    with pytest.raises(BackendUnavailableError) as excinfo:
        uat.upload_s3(
            "dsA",
            dataset_dir,
            tool="dm",
            s3_destination_prefix=S3_PREFIX,
            execute=False,
            dry_run=False,
        )

    error = excinfo.value
    assert error.name == "dm"
    assert error.hint is not None
    assert "s5cmd" in error.hint
    assert "gcloud" in error.hint
    assert isinstance(error.__cause__, BackendUnavailableError)
    message = str(error)
    assert message.count("data-movement backend") == 1
    # No double-wrapped "backend '<sentence>'" text and no sentence in ``name``.
    assert "backend unavailable" not in error.name
    assert "backend unavailable" not in message


def test_gcs_preflight_dm_resolver_preserves_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from bspp.orchestration.runtime.postprocessing import gcs_preflight

    def fail_get_backend(name: str):
        raise BackendUnavailableError(name)

    monkeypatch.setattr(gcs_preflight, "get_backend", fail_get_backend)

    with pytest.raises(BackendUnavailableError) as excinfo:
        gcs_preflight._resolve_dm_backend()

    error = excinfo.value
    assert error.name == "dm"
    assert error.hint is not None
    assert "gcloud" in error.hint
    assert isinstance(error.__cause__, BackendUnavailableError)
    assert str(error).count("data-movement backend") == 1


# ---------------------------------------------------------------------------
# CLI forwarding and fail-closed behavior
# ---------------------------------------------------------------------------


def test_upload_s3_cli_forwards_custom_destination_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_base = tmp_path / "output"
    dataset_dir = output_base / "dsA"
    dataset_dir.mkdir(parents=True)
    captured: dict[str, object] = {}

    def fake_upload_s3(dataset: str, dataset_output_dir: Path, **kwargs: object) -> uat.StagePlan:
        captured.update(kwargs)
        captured["dataset"] = dataset
        captured["dataset_output_dir"] = dataset_output_dir
        return uat.StagePlan(stage="s3", tool="s5cmd", dataset=dataset)

    monkeypatch.setattr(uat, "upload_s3", fake_upload_s3)

    result = CliRunner().invoke(
        cli,
        [
            "upload-s3",
            "--dataset",
            "dsA",
            "--output-base",
            str(output_base),
            "--s3-destination-prefix",
            "s3://example-bucket-dev/users/test/postprocessed/dsA/",
        ],
    )

    assert result.exit_code == 0
    assert captured["dataset"] == "dsA"
    assert captured["dataset_output_dir"] == dataset_dir
    assert captured["s3_destination_prefix"] == "s3://example-bucket-dev/users/test/postprocessed/dsA/"


def test_upload_gcs_cli_forwards_custom_prefixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_upload_gcs(dataset: str, **kwargs: object) -> uat.StagePlan:
        captured.update(kwargs)
        captured["dataset"] = dataset
        return uat.StagePlan(stage="gcs", tool="gcloud", dataset=dataset)

    monkeypatch.setattr(uat, "upload_gcs", fake_upload_gcs)

    result = CliRunner().invoke(
        cli,
        [
            "upload-gcs",
            "--dataset",
            "dsA",
            "--data-dir",
            str(tmp_path / "logs"),
            "--s3-source-prefix",
            "s3://example-bucket-dev/users/test/postprocessed/dsA/",
            "--gcs-destination-prefix",
            "gs://example-gcs-bucket-dev/delivery/dsA/",
        ],
    )

    assert result.exit_code == 0
    assert captured["dataset"] == "dsA"
    assert captured["s3_source_prefix"] == "s3://example-bucket-dev/users/test/postprocessed/dsA/"
    assert captured["gcs_destination_prefix"] == "gs://example-gcs-bucket-dev/delivery/dsA/"


def test_upload_gcs_direct_cli_forwards_custom_destination_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_base = tmp_path / "output"
    dataset_dir = output_base / "dsA"
    dataset_dir.mkdir(parents=True)
    captured: dict[str, object] = {}

    def fake_upload_gcs_direct(dataset: str, dataset_output_dir: Path, **kwargs: object) -> uat.StagePlan:
        captured.update(kwargs)
        captured["dataset"] = dataset
        captured["dataset_output_dir"] = dataset_output_dir
        return uat.StagePlan(stage="gcs-direct", tool="gcloud", dataset=dataset)

    monkeypatch.setattr(uat, "upload_gcs_direct", fake_upload_gcs_direct)

    result = CliRunner().invoke(
        cli,
        [
            "upload-gcs-direct",
            "--dataset",
            "dsA",
            "--output-base",
            str(output_base),
            "--gcs-destination-prefix",
            "gs://example-gcs-bucket-dev/users/test/postprocessed/dsA/",
        ],
    )

    assert result.exit_code == 0
    assert captured["dataset"] == "dsA"
    assert captured["dataset_output_dir"] == dataset_dir
    assert captured["gcs_destination_prefix"] == "gs://example-gcs-bucket-dev/users/test/postprocessed/dsA/"


def test_upload_s3_cli_rejects_omitted_destination_prefix(tmp_path: Path) -> None:
    output_base = tmp_path / "output"
    dataset_dir = output_base / "dsA"
    dataset_dir.mkdir(parents=True)

    result = CliRunner().invoke(
        cli,
        ["upload-s3", "--dataset", "dsA", "--output-base", str(output_base)],
    )

    assert result.exit_code == 2
    assert "s3-destination-prefix" in result.output


def test_upload_gcs_cli_rejects_omitted_prefixes(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["upload-gcs", "--dataset", "dsA", "--data-dir", str(tmp_path / "logs")],
    )

    assert result.exit_code == 2
    assert "Missing option" in result.output
    assert "s3-source-prefix" in result.output


def test_upload_gcs_direct_cli_rejects_omitted_destination_prefix(tmp_path: Path) -> None:
    output_base = tmp_path / "output"
    dataset_dir = output_base / "dsA"
    dataset_dir.mkdir(parents=True)

    result = CliRunner().invoke(
        cli,
        ["upload-gcs-direct", "--dataset", "dsA", "--output-base", str(output_base)],
    )

    assert result.exit_code == 2
    assert "gcs-destination-prefix" in result.output


def test_upload_cli_help_exposes_required_prefixes_without_legacy_defaults() -> None:
    runner = CliRunner()

    upload_s3_help = runner.invoke(cli, ["upload-s3", "--help"])
    assert upload_s3_help.exit_code == 0
    assert "--s3-destination-prefix" in upload_s3_help.output
    assert "required" in upload_s3_help.output
    assert "legacy" not in upload_s3_help.output

    upload_gcs_help = runner.invoke(cli, ["upload-gcs", "--help"])
    assert upload_gcs_help.exit_code == 0
    assert "--s3-source-prefix" in upload_gcs_help.output
    assert "--gcs-destination-prefix" in upload_gcs_help.output
    assert "legacy" not in upload_gcs_help.output

    upload_gcs_direct_help = runner.invoke(cli, ["upload-gcs-direct", "--help"])
    assert upload_gcs_direct_help.exit_code == 0
    assert "--gcs-destination-prefix" in upload_gcs_direct_help.output
    assert "legacy" not in upload_gcs_direct_help.output


def test_upload_s3_cli_defaults_to_s5cmd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_base = tmp_path / "output"
    dataset_dir = output_base / "dsA"
    dataset_dir.mkdir(parents=True)
    captured: dict[str, object] = {}

    def fake_upload_s3(dataset: str, dataset_output_dir: Path, **kwargs: object) -> uat.StagePlan:
        captured.update(kwargs)
        captured["dataset"] = dataset
        captured["dataset_output_dir"] = dataset_output_dir
        return uat.StagePlan(stage="s3", tool="s5cmd", dataset=dataset)

    monkeypatch.setattr(uat, "upload_s3", fake_upload_s3)

    result = CliRunner().invoke(
        cli,
        [
            "upload-s3",
            "--dataset",
            "dsA",
            "--output-base",
            str(output_base),
            "--s3-destination-prefix",
            S3_PREFIX,
        ],
    )

    assert result.exit_code == 0
    assert captured["tool"] == "s5cmd"


def test_upload_gcs_cli_defaults_to_gcloud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_upload_gcs(dataset: str, **kwargs: object) -> uat.StagePlan:
        captured.update(kwargs)
        captured["dataset"] = dataset
        return uat.StagePlan(stage="gcs", tool="gcloud", dataset=dataset)

    monkeypatch.setattr(uat, "upload_gcs", fake_upload_gcs)

    result = CliRunner().invoke(
        cli,
        [
            "upload-gcs",
            "--dataset",
            "dsA",
            "--data-dir",
            str(tmp_path / "logs"),
            "--s3-source-prefix",
            S3_PREFIX,
            "--gcs-destination-prefix",
            GCS_PREFIX,
        ],
    )

    assert result.exit_code == 0
    assert captured["tool"] == "gcloud"
