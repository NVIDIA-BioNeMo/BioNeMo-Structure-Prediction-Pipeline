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

"""Public filesystem-authority seams for preprocessing action evidence."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from bspp.orchestration.contract.database_placement import DatabaseAccessPolicy
from bspp.orchestration.contract.database_set_provisioning import DatabaseSetIdentity
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingChunkActionEvidence,
    PreprocessingDatabasePlacementCommandFailureEvidence,
)
from bspp.orchestration.contract.preprocessing_runtime import PREPROCESSING_ADAPTER_VERSION
from bspp.orchestration.contract.preprocessing_state import (
    PreprocessingArchiveEvidence,
    PreprocessingPairedEvidence,
)
from bspp.orchestration.runtime.preprocessing import _database_placement_evidence_io as placement_evidence_io
from bspp.orchestration.runtime.preprocessing._action_evidence_io import (
    load_preprocessing_action_evidence,
    publish_preprocessing_action_evidence,
)


def _action_evidence(*, error: str = "placement authority could not be reconciled") -> PreprocessingChunkActionEvidence:
    chunk_name = "proteins_tranche00_00000.fa"
    phase_run_id = "phase-run-0123456789abcdef0123456789abcdef"
    attempt_id = "attempt-0001"
    phase_runspec_digest = "a" * 64
    action_id = "preprocessing-chunk-000000"
    placement_process_status = 17
    placement = PreprocessingDatabasePlacementCommandFailureEvidence(
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        phase_runspec_digest=phase_runspec_digest,
        action_id=action_id,
        database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"),
        requested_policy=DatabaseAccessPolicy.STAGE_REQUIRED,
        source_manifest_sha256="b" * 64,
        placement_process_status=placement_process_status,
        result=None,
        result_digest=None,
        science_started=False,
        classification="evidence-reconciliation-failed",
        error=error,
    )
    return PreprocessingChunkActionEvidence(
        adapter_version=PREPROCESSING_ADAPTER_VERSION,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        phase_runspec_digest=phase_runspec_digest,
        action_id=action_id,
        placement_process_status=placement_process_status,
        chunk_name=chunk_name,
        started_at="2026-08-28T10:00:00.000000Z",
        finished_at="2026-08-28T10:00:01.000000Z",
        outcome="failed",
        command_outcomes=(),
        paired_evidence=PreprocessingPairedEvidence(
            chunk_name=chunk_name,
            durable_record_path="/output/proteins_tranche00_00000.record",
            durable_log_path="/output/proteins_tranche00_00000.log",
            record_lines=None,
            log_lines=None,
        ),
        archive_evidence=PreprocessingArchiveEvidence(
            chunk_name=chunk_name,
            durable_tar_path="/output/proteins_tranche00_00000.tar",
            durable_lz4_path="/output/proteins_tranche00_00000.tar.lz4",
            tar_size_bytes=None,
            lz4_size_bytes=None,
            tar_members=None,
        ),
        output_hashes=(),
        error=error,
        database_placement=placement,
        raw_search_evidence=None,
        carry_forward_adoption=None,
    )


def _canonical_bytes(evidence: PreprocessingChunkActionEvidence) -> bytes:
    return (json.dumps(evidence.to_mapping(), indent=2, sort_keys=True) + "\n").encode()


def _write_authority(path: Path, evidence: PreprocessingChunkActionEvidence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(evidence))
    path.chmod(0o444)


def test_action_evidence_loader_accepts_exact_immutable_canonical_authority(tmp_path: Path) -> None:
    evidence = _action_evidence()
    path = tmp_path / "evidence" / "action.json"
    _write_authority(path, evidence)

    assert load_preprocessing_action_evidence(path) == evidence


@pytest.mark.parametrize("fault", ["symlink", "non-regular", "writable", "noncanonical"])
def test_action_evidence_loader_rejects_non_authoritative_files(fault: str, tmp_path: Path) -> None:
    evidence = _action_evidence()
    path = tmp_path / "evidence" / "action.json"
    path.parent.mkdir(parents=True)
    if fault == "symlink":
        target = path.with_name("target.json")
        _write_authority(target, evidence)
        path.symlink_to(target.name)
    elif fault == "non-regular":
        path.mkdir()
    elif fault == "writable":
        path.write_bytes(_canonical_bytes(evidence))
        path.chmod(0o644)
    else:
        path.write_bytes(json.dumps(evidence.to_mapping(), separators=(",", ":")).encode())
        path.chmod(0o444)

    with pytest.raises(ValueError):
        load_preprocessing_action_evidence(path)


def test_action_evidence_loader_rejects_file_change_while_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _action_evidence()
    forged = _action_evidence(error="forged placement reconciliation failure")
    path = tmp_path / "evidence" / "action.json"
    _write_authority(path, evidence)
    authority_inode = path.stat().st_ino
    real_fstat = placement_evidence_io.os.fstat
    changed = False

    def change_after_first_file_stat(descriptor: int) -> os.stat_result:
        nonlocal changed
        info = real_fstat(descriptor)
        if info.st_ino == authority_inode and not changed:
            changed = True
            path.chmod(0o600)
            path.write_bytes(_canonical_bytes(forged))
            path.chmod(0o444)
        return info

    monkeypatch.setattr(placement_evidence_io.os, "fstat", change_after_first_file_stat)

    with pytest.raises(ValueError):
        load_preprocessing_action_evidence(path)
    assert changed


def test_action_evidence_loader_rejects_parent_rebind_during_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _action_evidence()
    path = tmp_path / "evidence" / "action.json"
    displaced = tmp_path / "displaced-evidence"
    _write_authority(path, evidence)
    real_open = placement_evidence_io.os.open
    rebound = False

    def rebind_before_file_open(
        target: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal rebound
        if target == path.name and dir_fd is not None and not rebound:
            rebound = True
            path.parent.rename(displaced)
            path.parent.mkdir()
            _write_authority(path, replace(evidence, error="substitute authority"))
        return real_open(target, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(placement_evidence_io.os, "open", rebind_before_file_open)

    with pytest.raises(ValueError):
        load_preprocessing_action_evidence(path)
    assert rebound


def test_action_evidence_publisher_uses_anchored_durable_no_replace_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _action_evidence()
    path = tmp_path / "evidence" / "action.json"
    path.parent.mkdir(parents=True)
    real_open = placement_evidence_io.os.open
    real_link = placement_evidence_io.os.link
    real_fsync = placement_evidence_io.os.fsync
    real_unlink = placement_evidence_io.os.unlink
    opened: list[tuple[str, int, int | None]] = []
    linked: list[tuple[int | None, int | None, bool]] = []
    fsynced_kinds: list[int] = []
    unlinked: list[tuple[str, int | None]] = []

    def observe_open(
        target: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        opened.append((os.fspath(target), flags, dir_fd))
        return real_open(target, flags, mode, dir_fd=dir_fd)

    def observe_link(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        linked.append((src_dir_fd, dst_dir_fd, follow_symlinks))
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    def observe_fsync(descriptor: int) -> None:
        fsynced_kinds.append(stat.S_IFMT(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    def observe_unlink(target: os.PathLike[str] | str, *, dir_fd: int | None = None) -> None:
        unlinked.append((os.fspath(target), dir_fd))
        real_unlink(target, dir_fd=dir_fd)

    monkeypatch.setattr(placement_evidence_io.os, "open", observe_open)
    monkeypatch.setattr(placement_evidence_io.os, "link", observe_link)
    monkeypatch.setattr(placement_evidence_io.os, "fsync", observe_fsync)
    monkeypatch.setattr(placement_evidence_io.os, "unlink", observe_unlink)

    publish_preprocessing_action_evidence(evidence, path)

    assert path.read_bytes() == _canonical_bytes(evidence)
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert load_preprocessing_action_evidence(path) == evidence
    temporary_open = next(item for item in opened if item[0].startswith(f".{path.name}.tmp-"))
    assert temporary_open[1] & (os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW) == (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    )
    assert temporary_open[2] is not None
    assert linked == [(temporary_open[2], temporary_open[2], False)]
    assert stat.S_IFREG in fsynced_kinds
    assert stat.S_IFDIR in fsynced_kinds
    assert any(name == temporary_open[0] and dir_fd == temporary_open[2] for name, dir_fd in unlinked)
    assert not tuple(path.parent.glob(f".{path.name}.tmp-*"))


@pytest.mark.parametrize("existing", ["exact", "different"])
def test_action_evidence_publisher_never_replaces_existing_destination(existing: str, tmp_path: Path) -> None:
    evidence = _action_evidence()
    path = tmp_path / "evidence" / "action.json"
    preexisting = evidence if existing == "exact" else _action_evidence(error="different immutable authority")
    _write_authority(path, preexisting)
    before = path.read_bytes()
    before_mode = stat.S_IMODE(path.stat().st_mode)

    with pytest.raises(ValueError):
        publish_preprocessing_action_evidence(evidence, path)

    assert path.read_bytes() == before
    assert stat.S_IMODE(path.stat().st_mode) == before_mode == 0o444
    assert not tuple(path.parent.glob(f".{path.name}.tmp-*"))


def test_action_evidence_publisher_rejects_destination_substitution_before_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _action_evidence()
    forged = _action_evidence(error="substituted immutable authority")
    path = tmp_path / "evidence" / "action.json"
    path.parent.mkdir(parents=True)
    real_fsync = placement_evidence_io.os.fsync
    substituted = False

    def substitute_after_parent_fsync(descriptor: int) -> None:
        nonlocal substituted
        real_fsync(descriptor)
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and path.exists() and not substituted:
            substituted = True
            path.unlink()
            _write_authority(path, forged)

    monkeypatch.setattr(placement_evidence_io.os, "fsync", substitute_after_parent_fsync)

    with pytest.raises(ValueError):
        publish_preprocessing_action_evidence(evidence, path)

    assert substituted
    assert path.read_bytes() == _canonical_bytes(forged)
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert not tuple(path.parent.glob(f".{path.name}.tmp-*"))


@pytest.mark.parametrize("boundary", ["after-anchor", "after-link"])
def test_action_evidence_publisher_rejects_parent_rebind(
    boundary: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _action_evidence()
    path = tmp_path / "evidence" / "action.json"
    displaced = tmp_path / "displaced-evidence"
    path.parent.mkdir(parents=True)
    real_open = placement_evidence_io.os.open
    real_link = placement_evidence_io.os.link
    rebound = False

    def rebind_parent() -> None:
        nonlocal rebound
        rebound = True
        path.parent.rename(displaced)
        path.parent.mkdir()

    def rebind_after_anchor(
        target: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if (
            boundary == "after-anchor"
            and os.fspath(target).startswith(f".{path.name}.tmp-")
            and dir_fd is not None
            and not rebound
        ):
            rebind_parent()
        return real_open(target, flags, mode, dir_fd=dir_fd)

    def rebind_after_link(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        if boundary == "after-link" and not rebound:
            rebind_parent()

    monkeypatch.setattr(placement_evidence_io.os, "open", rebind_after_anchor)
    monkeypatch.setattr(placement_evidence_io.os, "link", rebind_after_link)

    with pytest.raises(ValueError):
        publish_preprocessing_action_evidence(evidence, path)

    assert rebound
    assert not path.exists()
    assert (displaced / path.name).read_bytes() == _canonical_bytes(evidence)
    assert not tuple(path.parent.glob(f".{path.name}.tmp-*"))
    assert not tuple(displaced.glob(f".{path.name}.tmp-*"))


@pytest.mark.parametrize(
    "fault",
    ["parent-fsync", "reload-open", "reload-parse", "reload-canonical", "temporary-cleanup"],
)
def test_action_evidence_publisher_reports_post_link_faults_without_claiming_success(
    fault: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _action_evidence()
    path = tmp_path / "evidence" / "action.json"
    path.parent.mkdir(parents=True)
    real_open = placement_evidence_io.os.open
    real_fsync = placement_evidence_io.os.fsync
    real_unlink = placement_evidence_io.os.unlink
    injected = False

    def faulting_fsync(descriptor: int) -> None:
        nonlocal injected
        if (
            fault in {"parent-fsync", "reload-parse", "reload-canonical"}
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and path.exists()
            and not injected
        ):
            injected = True
            if fault == "parent-fsync":
                raise OSError("injected parent fsync failure")
            if fault in {"reload-parse", "reload-canonical"}:
                path.unlink()
                if fault == "reload-parse":
                    path.write_bytes(b"{not-json\n")
                else:
                    path.write_bytes(json.dumps(evidence.to_mapping(), separators=(",", ":")).encode())
                path.chmod(0o444)
        real_fsync(descriptor)

    def faulting_open(
        target: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal injected
        if fault == "reload-open" and target == path.name and dir_fd is not None and path.exists() and not injected:
            injected = True
            raise OSError("injected strict reload open failure")
        return real_open(target, flags, mode, dir_fd=dir_fd)

    def faulting_unlink(target: os.PathLike[str] | str, *, dir_fd: int | None = None) -> None:
        nonlocal injected
        if (
            fault == "temporary-cleanup"
            and os.fspath(target).startswith(f".{path.name}.tmp-")
            and path.exists()
            and not injected
        ):
            injected = True
            raise OSError("injected temporary cleanup failure")
        real_unlink(target, dir_fd=dir_fd)

    monkeypatch.setattr(placement_evidence_io.os, "fsync", faulting_fsync)
    monkeypatch.setattr(placement_evidence_io.os, "open", faulting_open)
    monkeypatch.setattr(placement_evidence_io.os, "unlink", faulting_unlink)

    with pytest.raises(ValueError):
        publish_preprocessing_action_evidence(evidence, path)

    assert injected
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    if fault not in {"reload-parse", "reload-canonical"}:
        assert path.read_bytes() == _canonical_bytes(evidence)
    assert not tuple(path.parent.glob(f".{path.name}.tmp-*"))
