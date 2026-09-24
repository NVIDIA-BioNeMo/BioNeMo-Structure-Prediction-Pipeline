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

"""Cold Database Replica population and atomic-publication seams."""

from __future__ import annotations

import errno
import json
import os
import stat
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from bspp.orchestration.contract.database_replica import (
    DatabaseCapacityGate,
    DatabaseReplicaColdResult,
    database_replica_manifest_digest,
    database_replica_manifest_from_mapping,
)
from bspp.orchestration.contract.phase import phase_runspec_from_mapping
from bspp.orchestration.contract.runspec_policies import parse_slurm_time_seconds
from bspp.orchestration.runtime.preprocessing._database_replica_copy import population_worker_count
from bspp.orchestration.runtime.preprocessing._database_replica_evidence_io import (
    load_database_replica_cold_result,
)
from bspp.orchestration.runtime.preprocessing._database_replica_publication import (
    PopulationWorkspace,
    make_replica_immutable,
    publish_replica_noreplace,
    require_renameat2,
)
from bspp.orchestration.runtime.preprocessing.database_placement import DatabasePlacementError
from tests.support.database_cold_replica import (
    _manifest,
    _shared_source_fixture,
    _stage_required_fixture,
)
from tests.support.database_cold_replica import (
    cold_cache_root as cold_cache_root,
)
from tests.support.preprocessing_execution import preprocessing_execution_fixture


def test_stage_required_cold_population_accepts_exact_capacity_equality_and_publishes_complete_replica(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"

    result = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=result_path,
        failure_path=failure_path,
    )

    assert isinstance(result, DatabaseReplicaColdResult)
    assert result.capacity_gate == DatabaseCapacityGate(
        available_user_bytes=2,
        allocated_replica_bytes=2,
        reserved_bytes=0,
        required_bytes=2,
        decision="sufficient",
    )
    assert result.copy_evidence.selected_workers == 2
    assert result.copy_evidence.total_copied_bytes == 2
    assert result.copy_evidence.aggregate_bytes_per_second == (
        result.copy_evidence.total_copied_bytes * 1_000_000_000 // result.copy_evidence.elapsed_nanoseconds
    )
    final = cache_root / "replicas" / fixture.runspec.payload.database.source_manifest_sha256
    manifest = database_replica_manifest_from_mapping(json.loads((final / "replica-manifest.json").read_text()))
    assert database_replica_manifest_digest(manifest) == result.replica_manifest_sha256
    assert {path.name for path in final.iterdir()} == {
        "replica-manifest.json",
        *(member.logical_name for member in fixture.runspec.payload.database.source_manifest.members),
    }
    assert all((final / member.logical_name).read_bytes() == b"x" for member in manifest.members)
    assert stat.S_IMODE(final.stat().st_mode) == 0o555
    assert stat.S_IMODE((final / "replica-manifest.json").stat().st_mode) == 0o444
    assert all(stat.S_IMODE((final / member.logical_name).stat().st_mode) == 0o444 for member in manifest.members)
    assert result.source_mount.filesystem_type == "overlay"
    assert result.cache_mount.filesystem_type == "tmpfs"
    assert result.cache_mount.writable_verified is True
    assert not failure_path.exists()


def test_staging_lock_wait_must_fit_the_sole_action_wall_time(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _, _ = _stage_required_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    staging = fixture.runspec.payload.database.staging
    assert staging is not None
    wall_seconds = parse_slurm_time_seconds(fixture.runspec.payload.actions[0].resources.time)

    equal_database = replace(
        fixture.runspec.payload.database,
        staging=replace(staging, lock_wait_seconds=wall_seconds),
    )
    equal_payload = replace(fixture.runspec.payload, database=equal_database)
    assert equal_payload.database.staging is not None
    assert equal_payload.database.staging.lock_wait_seconds == wall_seconds

    excessive_database = replace(
        fixture.runspec.payload.database,
        staging=replace(staging, lock_wait_seconds=wall_seconds + 1),
    )
    with pytest.raises(ValueError, match="cannot exceed the Runtime Action wall time"):
        replace(fixture.runspec.payload, database=excessive_database)


def test_stage_required_runspec_rejects_missing_staging(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _, _ = _stage_required_fixture(tmp_path / "work", cold_cache_root, monkeypatch)

    with pytest.raises(ValueError, match="staged database branches require a complete profile staging snapshot"):
        replace(fixture.runspec.payload.database, staging=None)


def test_direct_runspec_without_staging_round_trips_with_the_same_digest(tmp_path: Path) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "direct", direct_policy=True)

    assert fixture.runspec.payload.database.staging is None
    reconstructed = phase_runspec_from_mapping(fixture.runspec.to_mapping())

    assert reconstructed == fixture.runspec
    assert reconstructed.digest == fixture.runspec.digest


@pytest.mark.parametrize(
    ("unique_source_count", "allocated_cpus", "expected"),
    [
        (5, 1, 1),
        (5, 6, 3),
        (3, 16, 3),
        (20, 16, 8),
    ],
)
def test_population_worker_count_obeys_floor_cpu_source_and_cap_bounds(
    unique_source_count: int,
    allocated_cpus: int,
    expected: int,
) -> None:
    assert (
        population_worker_count(
            unique_source_count=unique_source_count,
            allocated_cpus=allocated_cpus,
        )
        == expected
    )


def test_exact_visible_cold_result_takes_precedence_over_failure_evidence(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _ = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    system_link = os.link

    def publish_then_report_exists(source: object, destination: object, **kwargs: object) -> None:
        system_link(source, destination, **kwargs)
        if destination == result_path.name:
            raise FileExistsError(errno.EEXIST, "injected post-link ambiguity")

    monkeypatch.setattr(os, "link", publish_then_report_exists)

    with pytest.raises(DatabasePlacementError, match="cold Result"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    result = load_database_replica_cold_result(result_path)
    assert result.phase_run_id == fixture.runspec.phase_run_id
    assert result.phase_runspec_digest == fixture.runspec.digest
    assert not failure_path.exists()


def test_hidden_process_death_partial_is_ineligible_and_untouched_by_cold_population(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    replicas = cache_root / "replicas"
    replicas.mkdir(mode=0o700)
    orphan = replicas / ".population-orphan"
    orphan.mkdir(mode=0o700)
    marker = orphan / "partial"
    marker.write_text("incomplete")
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )

    result = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "placement" / "result.json",
        failure_path=tmp_path / "placement" / "failure.json",
    )

    assert isinstance(result, DatabaseReplicaColdResult)
    assert marker.read_text() == "incomplete"
    assert orphan.name != fixture.runspec.payload.database.source_manifest_sha256


def test_same_parent_renameat2_is_one_atomic_noreplace_transition(tmp_path: Path) -> None:
    replicas = tmp_path / "replicas"
    replicas.mkdir(mode=0o700)
    parent_descriptor = os.open(replicas, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    workspaces: list[PopulationWorkspace] = []
    try:
        for ordinal in range(2):
            name = f".population-racer-{ordinal}"
            temporary = replicas / name
            temporary.mkdir(mode=0o700)
            marker = temporary / "winner"
            marker.write_text(str(ordinal))
            marker.chmod(0o444)
            temporary.chmod(0o555)
            workspaces.append(
                PopulationWorkspace(
                    replicas_descriptor=parent_descriptor,
                    temporary_descriptor=os.open(
                        temporary,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    ),
                    temporary_name=name,
                )
            )
        renameat2 = require_renameat2()
        barrier = threading.Barrier(2)
        errors: list[DatabasePlacementError] = []

        def publish(workspace: PopulationWorkspace) -> None:
            barrier.wait()
            try:
                publish_replica_noreplace(renameat2, workspace, destination_name="manifest-digest")
            except DatabasePlacementError as exc:
                errors.append(exc)

        threads = [threading.Thread(target=publish, args=(workspace,)) for workspace in workspaces]
        assert not (replicas / "manifest-digest").exists()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        destination = replicas / "manifest-digest"
        assert destination.is_dir()
        assert (destination / "winner").read_text() in {"0", "1"}
        assert len(errors) == 1
        assert "EEXIST" in str(errors[0])
        assert len(tuple(replicas.glob(".population-racer-*"))) == 1
    finally:
        for workspace in workspaces:
            os.close(workspace.temporary_descriptor)
        os.close(parent_descriptor)
        for directory in (item for item in replicas.iterdir() if item.is_dir()):
            directory.chmod(0o700)


def test_immutable_publication_open_failure_never_closes_a_reused_foreign_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    for member in manifest.members:
        path = temporary / member.replica_path
        path.write_bytes(b"x" * member.size_bytes)
        os.utime(path, ns=(member.mtime_ns, member.mtime_ns))
    root_descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    real_open = os.open
    foreign_descriptor: int | None = None

    def fail_after_reusing_closed_descriptor(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal foreign_descriptor
        if path == "primary":
            foreign_descriptor = real_open("/dev/null", os.O_RDONLY)
            raise OSError("injected member open failure")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", fail_after_reusing_closed_descriptor)
    try:
        with pytest.raises(DatabasePlacementError, match="freeze"):
            make_replica_immutable(root_descriptor, manifest=manifest, members=manifest.members)
        assert foreign_descriptor is not None
        os.fstat(foreign_descriptor)
    finally:
        monkeypatch.undo()
        if foreign_descriptor is not None:
            os.close(foreign_descriptor)
        os.close(root_descriptor)


def test_duplicate_resolved_sources_run_one_pinned_rsync_and_publish_hardlink_aliases(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _shared_source_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    call_log = tmp_path / "rsync-calls"
    executable = tmp_path / "rsync-fixture"
    executable.write_text("#!/bin/sh\nprintf 'call\\n' >> " + str(call_log) + '\nexec /usr/bin/rsync "$@"\n')
    executable.chmod(0o755)
    fixture = replace(
        fixture,
        database_placement_paths=replace(fixture.database_placement_paths, rsync=executable),
    )
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=1, f_frsize=1),
    )

    result = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "placement" / "result.json",
        failure_path=tmp_path / "placement" / "failure.json",
    )

    assert isinstance(result, DatabaseReplicaColdResult)
    assert result.copy_evidence.source_count == 1
    assert result.copy_evidence.selected_workers == 1
    assert call_log.read_text().splitlines() == ["call"]
    outcome = result.copy_evidence.outcomes[0]
    assert outcome.logical_names == tuple(
        sorted(member.logical_name for member in fixture.runspec.payload.database.source_manifest.members)
    )
    assert outcome.argv[0] == "/usr/bin/rsync"
    assert "/replicas/.population-<private>/" in outcome.argv[-1]
    final = cache_root / "replicas" / fixture.runspec.payload.database.source_manifest_sha256
    aliases = [final / member.logical_name for member in fixture.runspec.payload.database.source_manifest.members]
    assert len({path.stat().st_ino for path in aliases}) == 1
    assert all(path.is_file() and not path.is_symlink() for path in aliases)
