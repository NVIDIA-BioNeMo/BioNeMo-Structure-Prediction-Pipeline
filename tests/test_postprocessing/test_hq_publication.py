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

"""Tests for local HQ chunk publication planning."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from bspp.orchestration.contract.runspec import runspec_from_mapping
from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.data_movement.s3.client import S3Credentials
from bspp.orchestration.runtime.data_movement.s3.inventory import (
    S3InventoryObject,
    read_inventory_csv,
    read_sample_hashes_csv,
)
from bspp.orchestration.runtime.postprocessing.hq_chunks import HqChunkBuildChunk, write_hq_chunk_manifest
from bspp.orchestration.runtime.postprocessing.hq_publication import (
    HqChunkPublicationExecutionError,
    HqChunkUploadRow,
    execute_hq_chunk_publication,
    list_hq_publication_remote_prefix,
    plan_hq_chunk_publication,
    read_hq_upload_manifest,
    write_hq_upload_manifest,
)


def test_plan_hq_chunk_publication_builds_deterministic_destinations(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0", b"chunk-1"))
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))

    plan = plan_hq_chunk_publication(spec, chunks_dir=chunks_dir)

    assert plan.ok
    assert [item.destination_uri for item in plan.items] == [
        "s3://example-bucket/users/example-user/hq-canary/chunk_0000.tar",
        "s3://example-bucket/users/example-user/hq-canary/chunk_0001.tar",
    ]
    assert [item.action for item in plan.items] == ["upload", "upload"]


def test_plan_hq_chunk_publication_requires_enabled_runspec(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary", enabled=False))

    with pytest.raises(ValueError, match="disabled"):
        plan_hq_chunk_publication(spec, chunks_dir=chunks_dir)


def test_plan_hq_chunk_publication_fails_closed_on_manifest_mismatch(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    (chunks_dir / "chunk_0000.tar").write_bytes(b"changed")
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))

    with pytest.raises(ValueError, match=r"size mismatch|sha256 mismatch"):
        plan_hq_chunk_publication(spec, chunks_dir=chunks_dir)


def test_plan_hq_chunk_publication_resolves_relative_manifest_tar_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    manifest_path = chunks_dir / "hq_chunks_manifest.csv"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        manifest_text.replace(str(chunks_dir / "chunk_0000.tar"), "chunk_0000.tar"), encoding="utf-8"
    )
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))

    plan = plan_hq_chunk_publication(spec, chunks_dir=chunks_dir)

    assert plan.ok
    assert plan.items[0].source_path == chunks_dir / "chunk_0000.tar"


def test_plan_hq_chunk_publication_resumes_only_verified_prior_upload(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))
    initial = plan_hq_chunk_publication(spec, chunks_dir=chunks_dir)
    item = initial.items[0]
    upload_manifest = tmp_path / "upload.csv"
    write_hq_upload_manifest(
        upload_manifest,
        (
            HqChunkUploadRow(
                chunk_index=item.chunk_index,
                source_path=item.source_path,
                destination_uri=item.destination_uri,
                size_bytes=item.size_bytes,
                sha256=item.sha256,
                payload_sha256=item.payload_sha256,
                status="uploaded",
                attempts=1,
                remote_size_bytes=item.size_bytes,
                timestamp_utc="2026-06-01T00:00:00+00:00",
            ),
        ),
    )

    resumed = plan_hq_chunk_publication(
        spec,
        chunks_dir=chunks_dir,
        upload_manifest_path=upload_manifest,
        remote_inventory=(S3InventoryObject(uri=item.destination_uri, size_bytes=item.size_bytes),),
    )

    assert resumed.ok
    assert resumed.items[0].action == "skip_existing"


def test_plan_hq_chunk_publication_fails_on_foreign_remote_objects(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))

    plan = plan_hq_chunk_publication(
        spec,
        chunks_dir=chunks_dir,
        remote_inventory=(
            S3InventoryObject(uri="s3://example-bucket/users/example-user/hq-canary/stray.tar", size_bytes=1),
        ),
    )

    assert not plan.ok
    assert "foreign object" in plan.failures[0]


def test_plan_hq_chunk_publication_requires_overwrite_for_unknown_destination_collision(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))
    destination = "s3://example-bucket/users/example-user/hq-canary/chunk_0000.tar"

    blocked = plan_hq_chunk_publication(
        spec,
        chunks_dir=chunks_dir,
        remote_inventory=(S3InventoryObject(uri=destination, size_bytes=7),),
    )
    overwrite = plan_hq_chunk_publication(
        runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary", overwrite=True)),
        chunks_dir=chunks_dir,
        remote_inventory=(S3InventoryObject(uri=destination, size_bytes=7),),
    )

    assert not blocked.ok
    assert blocked.items[0].action == "blocked"
    assert overwrite.ok
    assert overwrite.items[0].action == "overwrite"


def test_hq_chunks_publish_cli_dry_run_writes_reports(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    runspec_path = tmp_path / "runspec.yaml"
    runspec_path.write_text(
        yaml.safe_dump(_runspec("s3://example-bucket/users/example-user/hq-canary")), encoding="utf-8"
    )

    result = CliRunner().invoke(
        cli,
        [
            "hq-chunks",
            "publish",
            "--runspec",
            str(runspec_path),
            "--chunks-dir",
            str(chunks_dir),
            "--write-report",
            str(tmp_path / "evidence"),
        ],
    )

    assert result.exit_code == 0
    rendered = json.loads(result.output)
    assert rendered["target_prefix"] == "s3://example-bucket/users/example-user/hq-canary/"
    assert rendered["upload_count"] == 1
    assert (tmp_path / "evidence" / "hq_chunks_publication_plan.json").exists()
    assert (tmp_path / "evidence" / "hq_chunks_upload_manifest.csv").exists()


def test_execute_hq_chunk_publication_collects_upload_and_gate_evidence(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0", b"chunk-1"))
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))
    remote_objects: dict[str, bytes] = {}
    uploads: list[tuple[Path, str]] = []

    def list_prefix(prefix: str) -> tuple[S3InventoryObject, ...]:
        return tuple(
            S3InventoryObject(uri=uri, size_bytes=len(payload))
            for uri, payload in sorted(remote_objects.items())
            if uri.startswith(prefix)
        )

    def upload_object(source: Path, destination: str) -> None:
        uploads.append((source, destination))
        remote_objects[destination] = source.read_bytes()

    def download_object(uri: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(remote_objects[uri])

    report = execute_hq_chunk_publication(
        spec,
        chunks_dir=chunks_dir,
        output_dir=tmp_path / "evidence",
        list_prefix=list_prefix,
        upload_object=upload_object,
        download_object=download_object,
    )

    assert report.ok
    assert [destination for _, destination in uploads] == [
        "s3://example-bucket/users/example-user/hq-canary/chunk_0000.tar",
        "s3://example-bucket/users/example-user/hq-canary/chunk_0001.tar",
    ]
    rows = read_hq_upload_manifest(report.upload_manifest_path)
    assert [row.status for row in rows] == ["uploaded", "uploaded"]
    assert [row.attempts for row in rows] == [1, 1]
    assert [row.remote_size_bytes for row in rows] == [7, 7]
    assert read_inventory_csv(report.pre_upload_inventory_path) == ()
    assert [obj.uri for obj in read_inventory_csv(report.post_upload_inventory_path)] == [
        "s3://example-bucket/users/example-user/hq-canary/chunk_0000.tar",
        "s3://example-bucket/users/example-user/hq-canary/chunk_0001.tar",
    ]
    sampled = read_sample_hashes_csv(report.sampled_hashes_path)
    assert [item.uri for item in sampled] == [
        "s3://example-bucket/users/example-user/hq-canary/chunk_0000.tar",
        "s3://example-bucket/users/example-user/hq-canary/chunk_0001.tar",
    ]
    assert report.validation_report.ok
    assert (tmp_path / "evidence" / "hq_chunks_publication_gate_a.json").exists()


def test_execute_hq_chunk_publication_retries_stale_post_upload_inventory(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))
    remote_objects: dict[str, bytes] = {}
    list_calls = 0

    def list_prefix(prefix: str) -> tuple[S3InventoryObject, ...]:
        nonlocal list_calls
        list_calls += 1
        if list_calls == 2:
            return ()
        return tuple(
            S3InventoryObject(uri=uri, size_bytes=len(payload))
            for uri, payload in sorted(remote_objects.items())
            if uri.startswith(prefix)
        )

    def upload_object(source: Path, destination: str) -> None:
        remote_objects[destination] = source.read_bytes()

    def download_object(uri: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(remote_objects[uri])

    report = execute_hq_chunk_publication(
        spec,
        chunks_dir=chunks_dir,
        output_dir=tmp_path / "evidence",
        list_prefix=list_prefix,
        upload_object=upload_object,
        download_object=download_object,
        post_upload_inventory_poll_seconds=0,
    )

    assert report.ok
    assert list_calls == 3
    rows = read_hq_upload_manifest(report.upload_manifest_path)
    assert rows[0].status == "uploaded"
    assert read_inventory_csv(report.post_upload_inventory_path) == (
        S3InventoryObject(
            uri="s3://example-bucket/users/example-user/hq-canary/chunk_0000.tar",
            size_bytes=7,
        ),
    )


def test_execute_hq_chunk_publication_rejects_unverified_existing_prefix_before_upload(tmp_path: Path) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    spec = runspec_from_mapping(_runspec("s3://example-bucket/users/example-user/hq-canary"))
    uploads: list[tuple[Path, str]] = []

    def list_prefix(prefix: str) -> tuple[S3InventoryObject, ...]:
        assert prefix == "s3://example-bucket/users/example-user/hq-canary/"
        return (S3InventoryObject(uri="s3://example-bucket/users/example-user/hq-canary/stray.tar", size_bytes=1),)

    def upload_object(source: Path, destination: str) -> None:
        uploads.append((source, destination))

    with pytest.raises(HqChunkPublicationExecutionError, match="blocked"):
        execute_hq_chunk_publication(
            spec,
            chunks_dir=chunks_dir,
            output_dir=tmp_path / "evidence",
            list_prefix=list_prefix,
            upload_object=upload_object,
            download_object=lambda uri, destination: None,
        )

    assert uploads == []
    assert (tmp_path / "evidence" / "hq_chunks_remote_inventory_before.csv").exists()
    assert (tmp_path / "evidence" / "hq_chunks_publication_plan.json").exists()


def test_list_hq_publication_remote_prefix_treats_s5cmd_no_object_found_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_which(name: str) -> str | None:
        return "/opt/s5cmd" if name == "s5cmd" else None

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr='ERROR "ls s3://example-bucket/users/example-user/hq-publication-checks/fresh/*": no object found\n',
        )

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    objects = list_hq_publication_remote_prefix(
        "s3://example-bucket/users/example-user/hq-publication-checks/fresh/",
        credentials=S3Credentials(access_key_id="key", secret_access_key="secret", endpoint_url="https://swift"),
        numworkers=8,
    )

    assert objects == ()


def test_list_hq_publication_remote_prefix_normalizes_relative_s5cmd_ls_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_which(name: str) -> str | None:
        return "/opt/s5cmd" if name == "s5cmd" else None

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="2026/06/09 01:42:00         101939200  chunk_0000.tar\n",
            stderr="",
        )

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    objects = list_hq_publication_remote_prefix(
        "s3://example-bucket/users/example-user/hq-publication-checks/20260609T014106Z/",
        credentials=S3Credentials(access_key_id="key", secret_access_key="secret", endpoint_url="https://swift"),
        numworkers=8,
    )

    assert objects == (
        S3InventoryObject(
            uri="s3://example-bucket/users/example-user/hq-publication-checks/20260609T014106Z/chunk_0000.tar",
            size_bytes=101939200,
        ),
    )


def test_hq_chunks_publish_cli_execute_fails_when_validation_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    runspec_path = tmp_path / "runspec.yaml"
    runspec_path.write_text(
        yaml.safe_dump(_runspec("s3://example-bucket/users/example-user/hq-canary")), encoding="utf-8"
    )
    captured: dict[str, Any] = {}

    class FailedReport:
        ok = False

        def to_redacted_dict(self) -> dict[str, object]:
            return {"ok": False, "validation_report": {"ok": False, "failures": ["sampled sha256 mismatch"]}}

    def fake_execute(**kwargs: Any) -> FailedReport:
        captured.update(kwargs)
        return FailedReport()

    from bspp.orchestration.runtime.postprocessing import hq_publication as hq_publication_mod

    monkeypatch.setattr(hq_publication_mod, "execute_hq_chunk_publication", fake_execute)

    result = CliRunner().invoke(
        cli,
        [
            "hq-chunks",
            "publish",
            "--runspec",
            str(runspec_path),
            "--chunks-dir",
            str(chunks_dir),
            "--write-report",
            str(tmp_path / "evidence"),
            "--execute",
        ],
    )

    assert result.exit_code != 0
    assert captured["chunks_dir"] == chunks_dir
    assert captured["output_dir"] == tmp_path / "evidence"
    assert "validation gate failed" in result.output


def test_hq_chunks_publish_cli_execute_reports_publication_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks_dir = _write_chunks(tmp_path, (b"chunk-0",))
    runspec_path = tmp_path / "runspec.yaml"
    runspec_path.write_text(
        yaml.safe_dump(_runspec("s3://example-bucket/users/example-user/hq-canary")), encoding="utf-8"
    )

    def fake_execute(**kwargs: Any) -> object:
        raise HqChunkPublicationExecutionError("remote list failed")

    from bspp.orchestration.runtime.postprocessing import hq_publication as hq_publication_mod

    monkeypatch.setattr(hq_publication_mod, "execute_hq_chunk_publication", fake_execute)

    result = CliRunner().invoke(
        cli,
        [
            "hq-chunks",
            "publish",
            "--runspec",
            str(runspec_path),
            "--chunks-dir",
            str(chunks_dir),
            "--write-report",
            str(tmp_path / "evidence"),
            "--execute",
        ],
    )

    assert result.exit_code != 0
    assert "remote list failed" in result.output
    assert "Traceback" not in result.output


def _write_chunks(tmp_path: Path, payloads: tuple[bytes, ...]) -> Path:
    chunks_dir = tmp_path / "chunks"
    chunks: list[HqChunkBuildChunk] = []
    for index, payload in enumerate(payloads):
        path = chunks_dir / f"chunk_{index:04d}.tar"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        chunks.append(
            HqChunkBuildChunk(
                chunk_index=index,
                tar_path=path,
                file_count=1,
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                payload_sha256=hashlib.sha256(f"payload-{index}".encode()).hexdigest(),
            )
        )
    write_hq_chunk_manifest(chunks_dir / "hq_chunks_manifest.csv", tuple(chunks))
    return chunks_dir


def _runspec(target_prefix: str, *, overwrite: bool = False, enabled: bool = True) -> dict[str, object]:
    return {
        "dataset": {"name": "dataset", "run_id": "run", "mode": "archive", "array": "0-0"},
        "cluster": {"name": "example-cluster", "account": "acct"},
        "paths": {
            "project_root": "/proj",
            "staging_dir": "/proj/staging",
            "output_dir": "/proj/output",
            "log_dir": "/proj/logs",
            "legacy_repo": "/repo/legacy",
            "orchestration_repo": "/repo/orch",
        },
        "references": {
            "master_parquet": "/refs/master.parquet",
            "tracking_parquet": "/refs/tracking.parquet",
            "manifest_csv": "/refs/manifest.csv",
            "uniprot_duckdb": "/refs/uniprot.duckdb",
        },
        "container": {"image": "image", "workdir": "/workspace"},
        "resources": {"gpu_worker": {"partition": "p", "cpus_per_task": 1, "memory": "1G", "time": "00:10:00"}},
        "worker": {
            "stages": "metadata_export",
            "workers": 1,
            "batch_size": 1,
            "shards_per_archive": 1,
            "self_upload": False,
            "local_scratch": True,
            "scratch_dir": "/tmp",
            "s5cmd_path": "/bin/s5cmd",
        },
        "storage": {
            "s3_archive_prefix": "s3://example-bucket/structures/",
            "s3_output_prefix": "s3://example-bucket/users/example-user/output/",
            "allow_production_prefixes": False,
        },
        "analysis_metadata": {
            "high_quality_from_tars": {
                "publication": {
                    "enabled": enabled,
                    "target_prefix": target_prefix,
                    "overwrite": overwrite,
                }
            }
        },
        "secrets": {"s3_credentials_ref": "env:bspp/s3"},
    }
