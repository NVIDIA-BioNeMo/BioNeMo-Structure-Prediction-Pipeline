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

"""Cold Database Replica failure evidence and CLI dispatch seams."""

from __future__ import annotations

import ctypes
import errno
import os
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from click.testing import CliRunner

from bspp.orchestration.contract.database_placement import (
    DATABASE_CACHE_ROOT,
    DATABASE_SOURCE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
)
from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaColdResult,
    canonical_database_replica_cold_result_bytes,
    database_replica_manifest_digest,
)
from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.preprocessing._database_replica import (
    _place_cold_stage_required_database,
    _production_cold_replica_services,
)
from bspp.orchestration.runtime.preprocessing._database_replica_evidence_io import (
    load_database_replica_cold_failure,
)
from bspp.orchestration.runtime.preprocessing._database_replica_lock import LockWait
from bspp.orchestration.runtime.preprocessing.database_placement import DatabasePlacementError
from tests.support.database_cold_replica import (
    _cache_mount,
    _capacity,
    _manifest,
    _shared_source_fixture,
    _source_mount,
    _stage_required_fixture,
)
from tests.support.database_cold_replica import (
    cold_cache_root as cold_cache_root,
)


class _RenameAt2Function:
    def __init__(self, call: Callable[[int, bytes, int, bytes, int], int]) -> None:
        self._call = call
        self.argtypes: object = None
        self.restype: object = None

    def __call__(
        self,
        source_parent: int,
        source_name: bytes,
        destination_parent: int,
        destination_name: bytes,
        flags: int,
    ) -> int:
        return self._call(source_parent, source_name, destination_parent, destination_name, flags)


class _LibcWithRenameAt2:
    def __init__(self, library: object, renameat2: Callable[[int, bytes, int, bytes, int], int]) -> None:
        self._library = library
        self.renameat2 = _RenameAt2Function(renameat2)

    def __getattr__(self, name: str) -> object:
        return getattr(self._library, name)


def _replace_libc_renameat2(
    monkeypatch: pytest.MonkeyPatch,
    renameat2: Callable[[int, bytes, int, bytes, int], int],
) -> None:
    real_cdll = cast(Callable[..., object], ctypes.CDLL)

    def load_library(*args: object, **kwargs: object) -> _LibcWithRenameAt2:
        return _LibcWithRenameAt2(real_cdll(*args, **kwargs), renameat2)

    monkeypatch.setattr(ctypes, "CDLL", load_library)


def test_stage_required_insufficient_capacity_publishes_bounded_failure_and_never_falls_back(
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
        lambda _descriptor: SimpleNamespace(f_bavail=1, f_frsize=1),
    )
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError, match="insufficient"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "insufficient-capacity"
    assert failure.capacity_gate is not None
    assert failure.capacity_gate.decision == "insufficient"
    assert failure.science_started is False
    assert not result_path.exists()
    assert not (cache_root / "replicas" / fixture.runspec.payload.database.source_manifest_sha256).exists()
    assert not tuple((cache_root / "replicas").glob(".population-*"))


def test_capacity_observation_error_publishes_distinct_bounded_failure(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _ = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )

    def fail_capacity_observation(_descriptor: int) -> None:
        raise OSError("injected fstatvfs failure")

    monkeypatch.setattr(os, "fstatvfs", fail_capacity_observation)
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError, match="user-available"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "capacity-observation-failed"
    assert failure.capacity_gate is None
    assert failure.science_started is False
    assert not result_path.exists()


@pytest.mark.parametrize(
    ("mutation", "classification"),
    [
        ("source-rw", "source-mount-invalid"),
        ("cache-fstype", "cache-authority-invalid"),
        ("cache-not-writable", "cache-authority-invalid"),
        ("missing-cpus", "authority-invalid"),
    ],
)
def test_cold_population_fails_closed_on_mount_cache_and_allocation_authority(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    classification: str,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    mountinfo_path = manifest_path.parent / "mountinfo"
    if mutation == "source-rw":
        mountinfo_path.write_text(
            mountinfo_path.read_text().replace(
                f"{DATABASE_SOURCE_ROOT} ro -",
                f"{DATABASE_SOURCE_ROOT} rw -",
            )
        )
    elif mutation == "cache-fstype":
        mountinfo_path.write_text(mountinfo_path.read_text().replace("tmpfs shm rw", "xfs shm rw"))
    elif mutation == "cache-not-writable":
        cache_root.chmod(0o500)
    else:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert load_database_replica_cold_failure(failure_path).classification == classification
    assert not result_path.exists()


def test_cold_population_rejects_effective_user_mismatch_before_cache_mutation(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
        unix_user="nobody",
    )
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError, match="Unix user"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=tmp_path / "placement" / "result.json",
            failure_path=failure_path,
        )

    assert load_database_replica_cold_failure(failure_path).classification == "authority-invalid"
    assert tuple(cache_root.iterdir()) == ()


def test_place_database_cli_exposes_no_physical_path_override_flags() -> None:
    result = CliRunner().invoke(cli, ["preprocessing", "place-database", "--help"])

    assert result.exit_code == 0, result.output
    assert "--phase-runspec" in result.output
    assert "--source-manifest" in result.output
    assert "--source-root" not in result.output
    assert "--cache-root" not in result.output
    assert "--rsync" not in result.output


@pytest.mark.parametrize(("behavior", "message"), [("omit", "inventory"), ("exit", "exit 19"), ("extra", "extra")])
def test_copy_boundary_failures_publish_no_result_or_visible_replica(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    message: str,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    executable = tmp_path / "rsync-fixture"
    if behavior == "omit":
        body = "#!/bin/sh\nexit 0\n"
    elif behavior == "exit":
        body = "#!/bin/sh\nexit 19\n"
    else:
        body = '#!/bin/sh\n/usr/bin/rsync "$@" || exit $?\ntouch "$(dirname "$7")/extra"\n'
    executable.write_text(body)
    executable.chmod(0o755)
    fixture = replace(
        fixture,
        database_placement_paths=replace(fixture.database_placement_paths, rsync=executable),
    )
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError, match=message):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert load_database_replica_cold_failure(failure_path).classification == "copy-failed"
    assert not result_path.exists()
    replicas = cache_root / "replicas"
    assert not (replicas / fixture.runspec.payload.database.source_manifest_sha256).exists()
    assert not tuple(replicas.glob(".population-*"))


def test_source_drift_during_rsync_fails_before_replica_publication(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    executable = tmp_path / "rsync-fixture"
    executable.write_text('#!/bin/sh\n/usr/bin/rsync "$@" || exit $?\ntouch "$6"\n')
    executable.chmod(0o755)
    fixture = replace(
        fixture,
        database_placement_paths=replace(fixture.database_placement_paths, rsync=executable),
    )
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError, match="metadata"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=tmp_path / "placement" / "result.json",
            failure_path=failure_path,
        )

    assert load_database_replica_cold_failure(failure_path).classification == "source-inventory-invalid"
    assert not (cache_root / "replicas" / fixture.runspec.payload.database.source_manifest_sha256).exists()


def test_cache_without_hardlink_support_fails_with_distinct_classification(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _shared_source_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=1, f_frsize=1),
    )

    system_link = os.link

    def reject_hardlink(source: object, destination: object, **kwargs: object) -> None:
        if source == "colabfold_envdb_202108_db" and destination == "uniref30_2302_db":
            raise OSError("hard links unsupported")
        system_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", reject_hardlink)
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError, match="hard-link"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=tmp_path / "placement" / "result.json",
            failure_path=failure_path,
        )

    assert load_database_replica_cold_failure(failure_path).classification == "hard-links-unsupported"
    assert not (cache_root / "replicas" / fixture.runspec.payload.database.source_manifest_sha256).exists()


def test_source_and_cache_on_same_device_fail_with_complete_cache_authority_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _ = _stage_required_fixture(
        tmp_path / "work",
        tmp_path / "cache-profile",
        monkeypatch,
    )
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError, match="different mounted filesystems"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=tmp_path / "placement" / "result.json",
            failure_path=failure_path,
        )

    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "cache-authority-invalid"
    assert failure.source_mount is not None
    assert failure.cache_mount is not None
    assert (failure.source_mount.device_major, failure.source_mount.device_minor) == (
        failure.cache_mount.device_major,
        failure.cache_mount.device_minor,
    )


def test_source_mount_drift_after_copy_has_source_mount_classification(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _shared_source_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    mountinfo_path = manifest_path.parent / "mountinfo"
    executable = tmp_path / "rsync-fixture"
    executable.write_text(
        '#!/bin/sh\n/usr/bin/rsync "$@" || exit $?\n'
        f'sed -i "s#{DATABASE_SOURCE_ROOT} ro -#{DATABASE_SOURCE_ROOT} rw -#" {mountinfo_path}\n'
    )
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
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError, match="read-only"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=tmp_path / "placement" / "result.json",
            failure_path=failure_path,
        )

    assert load_database_replica_cold_failure(failure_path).classification == "source-mount-invalid"
    replicas = cache_root / "replicas"
    assert not (replicas / fixture.runspec.payload.database.source_manifest_sha256).exists()
    assert not tuple(replicas.glob(".population-*"))


def test_copy_interruption_removes_owned_temp_without_publishing_evidence(
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

    def interrupt_copy(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupt_copy)
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(KeyboardInterrupt):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    replicas = cache_root / "replicas"
    assert not result_path.exists()
    assert not failure_path.exists()
    assert not (replicas / fixture.runspec.payload.database.source_manifest_sha256).exists()
    assert not tuple(replicas.glob(".population-*"))


def test_atomic_publication_failure_is_classified_and_removes_owned_temp(
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

    def unavailable_renameat2(*_args: object) -> int:
        ctypes.set_errno(errno.ENOSYS)
        return -1

    _replace_libc_renameat2(monkeypatch, unavailable_renameat2)
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError, match="ENOSYS"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert load_database_replica_cold_failure(failure_path).classification == "publication-failed"
    replicas = cache_root / "replicas"
    assert not result_path.exists()
    assert not (replicas / fixture.runspec.payload.database.source_manifest_sha256).exists()
    assert not tuple(replicas.glob(".population-*"))


def test_direct_cold_coordinator_repairs_an_invalid_existing_final(
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
    final = replicas / fixture.runspec.payload.database.source_manifest_sha256
    final.mkdir()
    marker = final / "existing"
    marker.write_text("preserve")
    failure_path = tmp_path / "placement" / "failure.json"
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )

    result = _place_cold_stage_required_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "placement" / "result.json",
        failure_path=failure_path,
        wait=LockWait.for_timeout(1),
        preprobe_contended=False,
        paths=fixture.database_placement_paths,
        services=_production_cold_replica_services(),
    )

    assert isinstance(result, DatabaseReplicaColdResult)
    assert not marker.exists()
    assert final.is_dir()
    assert not failure_path.exists()


def test_result_publication_failure_retains_replica_and_publishes_distinct_failure(
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
    result_path.parent.mkdir(parents=True)
    manifest = _manifest()
    foreign = DatabaseReplicaColdResult(
        phase_run_id="phase-run-11111111111111111111111111111111",
        attempt_id="attempt-0001",
        phase_runspec_digest="b" * 64,
        action_id="preprocessing-chunk-000000",
        database_set=manifest.database_set,
        requested_policy=DatabaseAccessPolicy.STAGE_REQUIRED,
        source_manifest_sha256=manifest.source_manifest_sha256,
        branch_kind="staged",
        outcome=DatabasePlacementOutcomeKind.REPLICA_COLD,
        selected_container_root=SELECTED_DATABASE_ROOT,
        replica_container_root=f"{DATABASE_CACHE_ROOT}/replicas/{manifest.source_manifest_sha256}",
        verification="metadata-verified",
        source_mount=_source_mount(),
        cache_mount=_cache_mount(),
        capacity_gate=_capacity(),
        replica_manifest_sha256=database_replica_manifest_digest(manifest),
        copy_evidence=manifest.copy_evidence,
    )
    result_path.write_bytes(canonical_database_replica_cold_result_bytes(foreign))
    result_path.chmod(0o444)
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError, match="cold Result"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    final = cache_root / "replicas" / fixture.runspec.payload.database.source_manifest_sha256
    assert final.is_dir()
    assert result_path.read_bytes() == canonical_database_replica_cold_result_bytes(foreign)
    assert load_database_replica_cold_failure(failure_path).classification == "result-publication-failed"
