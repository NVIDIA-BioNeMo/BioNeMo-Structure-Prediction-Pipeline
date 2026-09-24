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

"""Public seams for stage-preferred Database Placement."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from bspp.orchestration.contract.database_capacity_fallback_result import (
    DatabaseCapacityFallbackResult,
    canonical_database_capacity_fallback_result_bytes,
    database_capacity_fallback_result_digest,
    database_capacity_fallback_result_from_mapping,
)
from bspp.orchestration.contract.database_direct_result import (
    canonical_database_direct_result_bytes,
    database_direct_result_digest,
    database_direct_result_from_mapping,
)
from bspp.orchestration.contract.database_placement import (
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
    DatabaseSetSelection,
    build_preprocessing_database_binding,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabasePostScienceObservation,
    DatabaseSourceObservation,
)
from bspp.orchestration.contract.database_replica import (
    DatabaseCapacityGate,
    DatabaseReplicaColdFailureEvidence,
    DatabaseReplicaColdResult,
    DatabaseReplicaCopyEvidence,
    DatabaseReplicaManifest,
    DatabaseReplicaMember,
    DatabaseReplicaWarmResult,
)
from bspp.orchestration.contract.database_set_provisioning import DatabaseSetIdentity
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingDatabasePlacementCommandFailureEvidence,
    PreprocessingDatabasePlacementEvidence,
    preprocessing_chunk_action_evidence_from_mapping,
)
from bspp.orchestration.control.phase_rendering import render_phase_submission_intent
from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.preprocessing import _database_placement_evidence_io as placement_evidence_io
from bspp.orchestration.runtime.preprocessing._database_action_placement import (
    DatabaseActionPlacementError,
    resolve_database_science_branch,
)
from bspp.orchestration.runtime.preprocessing._database_placement_errors import DatabasePlacementError
from bspp.orchestration.runtime.preprocessing._database_replica_evidence_io import (
    load_database_replica_cold_failure,
    publish_database_replica_cold_failure,
)
from bspp.orchestration.runtime.preprocessing.database_placement import (
    _production_staged_database_placement_services,
)
from bspp.orchestration.runtime.preprocessing.execution import PreprocessingExecutionError
from tests.support.database_cold_replica import (
    _cache_mount,
    _source_mount,
    _source_observation,
    _stage_required_fixture,
)
from tests.support.database_cold_replica import cold_cache_root as cold_cache_root
from tests.support.preprocessing_execution import (
    LocalExecutionFixture,
    configure_preprocessing_fakes,
)


def _fallback_result() -> DatabaseCapacityFallbackResult:
    return DatabaseCapacityFallbackResult(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_runspec_digest="b" * 64,
        action_id="preprocessing-chunk-000000",
        database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"),
        requested_policy=DatabaseAccessPolicy.STAGE_PREFERRED,
        source_manifest_sha256="a" * 64,
        branch_kind="direct-capacity-fallback",
        outcome=DatabasePlacementOutcomeKind.DIRECT_CAPACITY_FALLBACK,
        selected_container_root=SELECTED_DATABASE_ROOT,
        verification="metadata-verified",
        source_mount=_source_mount(),
        cache_mount=_cache_mount(),
        capacity_gate=DatabaseCapacityGate(
            available_user_bytes=127,
            allocated_replica_bytes=64,
            reserved_bytes=64,
            required_bytes=128,
            decision="insufficient",
        ),
        pre_science_observation=_source_observation(),
    )


def test_capacity_fallback_result_is_a_strict_distinct_direct_family_document() -> None:
    result = _fallback_result()
    expected = (json.dumps(result.to_mapping(), indent=2, sort_keys=True) + "\n").encode()

    assert database_capacity_fallback_result_from_mapping(result.to_mapping()) == result
    assert canonical_database_capacity_fallback_result_bytes(result) == expected
    assert database_capacity_fallback_result_digest(result) == hashlib.sha256(expected).hexdigest()
    assert database_direct_result_from_mapping(result.to_mapping()) == result
    assert canonical_database_direct_result_bytes(result) == expected
    assert database_direct_result_digest(result) == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_policy", "direct"),
        ("branch_kind", "direct-requested"),
        ("outcome", "direct-requested"),
        ("verification", "unverified"),
    ],
)
def test_capacity_fallback_result_rejects_crossed_authority(field: str, value: str) -> None:
    mapping = deepcopy(_fallback_result().to_mapping())
    inner = mapping["database_capacity_fallback_result"]
    assert isinstance(inner, dict)
    inner[field] = value

    with pytest.raises(ValueError):
        database_capacity_fallback_result_from_mapping(mapping)


@pytest.mark.parametrize("mutation", ["sufficient", "inconsistent", "missing", "extra", "same-device"])
def test_capacity_fallback_result_requires_complete_positive_insufficient_gate(mutation: str) -> None:
    mapping = deepcopy(_fallback_result().to_mapping())
    inner = mapping["database_capacity_fallback_result"]
    assert isinstance(inner, dict)
    gate = inner["capacity_gate"]
    assert isinstance(gate, dict)
    if mutation == "sufficient":
        gate["available_user_bytes"] = 128
        gate["decision"] = "sufficient"
    elif mutation == "inconsistent":
        gate["required_bytes"] = 129
    elif mutation == "missing":
        del gate["reserved_bytes"]
    elif mutation == "extra":
        gate["free_bytes"] = 127
    else:
        source = inner["source_mount"]
        cache = inner["cache_mount"]
        assert isinstance(source, dict) and isinstance(cache, dict)
        cache["device_major"] = source["device_major"]
        cache["device_minor"] = source["device_minor"]

    with pytest.raises(ValueError):
        database_capacity_fallback_result_from_mapping(mapping)


def test_capacity_fallback_result_binds_allocated_bytes_to_unique_observed_payloads() -> None:
    mapping = deepcopy(_fallback_result().to_mapping())
    inner = mapping["database_capacity_fallback_result"]
    assert isinstance(inner, dict)
    gate = inner["capacity_gate"]
    assert isinstance(gate, dict)
    gate["allocated_replica_bytes"] = 65
    gate["required_bytes"] = 129

    with pytest.raises(ValueError, match="unique observed payload bytes"):
        database_capacity_fallback_result_from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "value", "available_bytes", "allocated_bytes", "required_bytes"),
    [
        ("size_bytes", 33, 96, 33, 97),
        ("mtime_ns", 124, 95, 32, 96),
    ],
    ids=("size", "mtime"),
)
def test_capacity_fallback_result_rejects_shared_source_metadata_disagreement(
    field: str,
    value: int,
    available_bytes: int,
    allocated_bytes: int,
    required_bytes: int,
) -> None:
    mapping = deepcopy(_fallback_result().to_mapping())
    inner = mapping["database_capacity_fallback_result"]
    assert isinstance(inner, dict)
    observation = inner["pre_science_observation"]
    assert isinstance(observation, dict)
    members = observation["members"]
    assert isinstance(members, list)
    shared_path = "shared-database-member"
    for member in members:
        assert isinstance(member, dict)
        member["source_kind"] = "symlink"
        member["resolved_path"] = shared_path
        member["alias_topology"] = [{"path": member["source_path"], "target": shared_path}]
    members[1][field] = value
    gate = inner["capacity_gate"]
    assert isinstance(gate, dict)
    gate["available_user_bytes"] = available_bytes
    gate["allocated_replica_bytes"] = allocated_bytes
    gate["required_bytes"] = required_bytes

    with pytest.raises(ValueError, match="aliases disagree about resolved source metadata"):
        database_capacity_fallback_result_from_mapping(mapping)


def test_capacity_fallback_result_requires_source_mount_to_contain_protected_root() -> None:
    mapping = deepcopy(_fallback_result().to_mapping())
    inner = mapping["database_capacity_fallback_result"]
    assert isinstance(inner, dict)
    source_mount = inner["source_mount"]
    assert isinstance(source_mount, dict)
    source_mount["mount_point"] = "/run/bspp/unrelated"

    with pytest.raises(ValueError, match="contain the protected source root"):
        database_capacity_fallback_result_from_mapping(mapping)


def test_stage_preferred_insufficient_capacity_cannot_be_failure_evidence() -> None:
    fallback = _fallback_result()

    with pytest.raises(ValueError, match="successful fallback Result"):
        DatabaseReplicaColdFailureEvidence(
            phase_run_id=fallback.phase_run_id,
            attempt_id=fallback.attempt_id,
            phase_runspec_digest=fallback.phase_runspec_digest,
            action_id=fallback.action_id,
            database_set=fallback.database_set,
            requested_policy=DatabaseAccessPolicy.STAGE_PREFERRED,
            source_manifest_sha256=fallback.source_manifest_sha256,
            source_mount=fallback.source_mount,
            cache_mount=fallback.cache_mount,
            capacity_gate=fallback.capacity_gate,
            science_started=False,
            classification="insufficient-capacity",
            error="must not be representable",
        )


def _stage_preferred_fixture(
    root: Path,
    cache_profile_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LocalExecutionFixture, Path, Path]:
    fixture, manifest_path, cache_root = _stage_required_fixture(root, cache_profile_root, monkeypatch)
    original = fixture.runspec.payload.database
    action = fixture.action.payload
    preferred = build_preprocessing_database_binding(
        selection=DatabaseSetSelection(
            database_set=original.database_set,
            requested_policy=DatabaseAccessPolicy.STAGE_PREFERRED,
        ),
        source_manifest=original.source_manifest,
        source_manifest_projection=original.source_manifest_projection,
        staging=original.staging,
        gpuserver_argv=action.gpuserver_argv,
        search_argv=action.search_argv,
    )
    runspec = replace(fixture.runspec, payload=replace(fixture.runspec.payload, database=preferred))
    fixture.runspec_path.write_text(json.dumps(runspec.to_mapping(), indent=2, sort_keys=True) + "\n")
    return replace(fixture, runspec=runspec), manifest_path, cache_root


def test_stage_preferred_sufficient_cold_and_warm_reuse_preserve_requested_policy(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    cold = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "placement" / "cold.json",
        failure_path=tmp_path / "placement" / "cold-failure.json",
    )
    assert isinstance(cold, DatabaseReplicaColdResult)
    assert cold.requested_policy == DatabaseAccessPolicy.STAGE_PREFERRED

    def forbidden_capacity(
        _cache_descriptor: int,
        *,
        allocated_bytes: int,
        reserved_bytes: int,
    ) -> DatabaseCapacityGate:
        del allocated_bytes, reserved_bytes
        raise AssertionError("warm placement observed capacity or attempted population")

    def forbidden_population(
        *,
        source_descriptor: int,
        temporary_descriptor: int,
        observation: DatabaseSourceObservation,
        allocated_cpus: int,
        rsync_executable: Path,
    ) -> tuple[DatabaseReplicaCopyEvidence, tuple[DatabaseReplicaMember, ...]]:
        del source_descriptor, temporary_descriptor, observation, allocated_cpus, rsync_executable
        raise AssertionError("warm placement observed capacity or attempted population")

    services = _production_staged_database_placement_services()
    services = replace(
        services,
        cold=replace(
            services.cold,
            capacity=replace(services.cold.capacity, observe=forbidden_capacity),
            population=replace(services.cold.population, populate=forbidden_population),
        ),
    )
    warm = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "placement" / "warm.json",
        failure_path=tmp_path / "placement" / "warm-failure.json",
        staged_services=services,
    )
    assert isinstance(warm, DatabaseReplicaWarmResult)
    assert warm.requested_policy == DatabaseAccessPolicy.STAGE_PREFERRED


def test_stage_preferred_insufficient_capacity_publishes_fallback_without_copy(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=1, f_frsize=1),
    )

    def forbidden_population(
        *,
        source_descriptor: int,
        temporary_descriptor: int,
        observation: DatabaseSourceObservation,
        allocated_cpus: int,
        rsync_executable: Path,
    ) -> tuple[DatabaseReplicaCopyEvidence, tuple[DatabaseReplicaMember, ...]]:
        del source_descriptor, temporary_descriptor, observation, allocated_cpus, rsync_executable
        raise AssertionError("insufficient capacity attempted replica population")

    services = _production_staged_database_placement_services()
    services = replace(
        services,
        cold=replace(
            services.cold,
            population=replace(services.cold.population, populate=forbidden_population),
        ),
    )
    result_path = tmp_path / "placement" / "fallback.json"
    failure_path = tmp_path / "placement" / "failure.json"

    result = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        staged_services=services,
    )

    assert isinstance(result, DatabaseCapacityFallbackResult)
    assert result.capacity_gate.decision == "insufficient"
    assert result.pre_science_observation.members == fixture.runspec.payload.database.source_manifest.members
    assert database_direct_result_from_mapping(json.loads(result_path.read_text())) == result
    assert not failure_path.exists()


def test_stage_preferred_post_gate_copy_failure_never_converts_to_fallback(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )

    def fail_copy(
        *,
        source_descriptor: int,
        temporary_descriptor: int,
        observation: DatabaseSourceObservation,
        allocated_cpus: int,
        rsync_executable: Path,
    ) -> tuple[DatabaseReplicaCopyEvidence, tuple[DatabaseReplicaMember, ...]]:
        del source_descriptor, temporary_descriptor, observation, allocated_cpus, rsync_executable
        raise DatabasePlacementError("copy denied")

    services = _production_staged_database_placement_services()
    services = replace(
        services,
        cold=replace(
            services.cold,
            population=replace(services.cold.population, populate=fail_copy),
        ),
    )
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError, match="copy denied"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
            staged_services=services,
        )

    assert not result_path.exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.requested_policy == DatabaseAccessPolicy.STAGE_PREFERRED
    assert failure.classification == "copy-failed"


@pytest.mark.parametrize(
    ("fault", "classification"),
    [
        ("source-mount", "source-mount-invalid"),
        ("cache-mount", "cache-authority-invalid"),
        ("lock", "lock-unavailable"),
        ("permission", "replica-validation-failed"),
        ("integrity", "publication-failed"),
        ("replica-publication", "publication-failed"),
        ("result-publication", "result-publication-failed"),
    ],
)
def test_stage_preferred_post_gate_faults_never_convert_to_fallback(
    fault: str,
    classification: str,
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )

    services = _production_staged_database_placement_services()
    original_capacity = services.cold.capacity.observe
    original_freeze = services.cold.population.freeze

    def observe_capacity_then_inject(
        cache_descriptor: int,
        *,
        allocated_bytes: int,
        reserved_bytes: int,
    ) -> DatabaseCapacityGate:
        capacity = original_capacity(
            cache_descriptor,
            allocated_bytes=allocated_bytes,
            reserved_bytes=reserved_bytes,
        )
        if fault in {"source-mount", "cache-mount"}:
            mountinfo = fixture.database_placement_paths.mountinfo
            contents = mountinfo.read_text()
            if fault == "source-mount":
                contents = contents.replace("overlay overlay ro", "ext4 source ro")
            else:
                contents = contents.replace("tmpfs shm rw", "ext4 cache rw")
            mountinfo.write_text(contents)
        elif fault == "lock":
            cache_lock = fixture.database_placement_paths.cache_root / ".locks" / "cache.lock"
            cache_lock.rename(cache_lock.with_name("cache.lock.displaced"))
            cache_lock.write_bytes(b"")
            cache_lock.chmod(0o600)
        return capacity

    def freeze_then_inject(
        temporary_descriptor: int,
        *,
        manifest: DatabaseReplicaManifest,
        members: tuple[DatabaseReplicaMember, ...],
    ) -> None:
        if fault == "permission":
            raise DatabasePlacementError("injected permission fault")
        original_freeze(temporary_descriptor, manifest=manifest, members=members)
        if fault == "integrity":
            os.chmod(members[0].replica_path, 0o644, dir_fd=temporary_descriptor, follow_symlinks=False)

    services = replace(
        services,
        cold=replace(
            services.cold,
            capacity=replace(services.cold.capacity, observe=observe_capacity_then_inject),
            population=replace(services.cold.population, freeze=freeze_then_inject),
        ),
    )
    if fault == "replica-publication":
        real_fsync = os.fsync
        identity = fixture.runspec.payload.database.source_manifest_sha256

        def fail_replica_publication_fsync(descriptor: int) -> None:
            try:
                target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            except OSError:
                target = None
            if target == fixture.database_placement_paths.cache_root / "replicas" and (target / identity).exists():
                raise OSError("injected replica-publication fault")
            real_fsync(descriptor)

        monkeypatch.setattr(os, "fsync", fail_replica_publication_fsync)
    elif fault == "result-publication":
        real_link = os.link

        def fail_result_link(
            source: str | bytes,
            destination: str | bytes,
            *,
            src_dir_fd: int | None = None,
            dst_dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> None:
            if os.fsdecode(destination) == "result.json":
                raise PermissionError("injected result-publication fault")
            real_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )

        monkeypatch.setattr(os, "link", fail_result_link)

    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    with pytest.raises(DatabasePlacementError):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
            staged_services=services,
        )

    assert not result_path.exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.requested_policy == DatabaseAccessPolicy.STAGE_PREFERRED
    assert failure.classification == classification


def test_stage_preferred_capacity_observation_error_never_converts_to_fallback(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)

    def fail_capacity(_descriptor: int) -> object:
        raise OSError("capacity unavailable")

    monkeypatch.setattr(os, "fstatvfs", fail_capacity)
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"

    with pytest.raises(DatabasePlacementError, match="cannot observe"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert not result_path.exists()
    assert load_database_replica_cold_failure(failure_path).classification == "capacity-observation-failed"


@pytest.mark.parametrize(
    "fault",
    [
        "source-inventory",
        "source-binding",
        "source-mount",
        "cache-mount",
        "cache-binding",
        "lock",
        "result-publication",
    ],
)
def test_stage_preferred_insufficient_gate_revalidation_faults_publish_no_fallback(
    fault: str,
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=1, f_frsize=1),
    )

    services = _production_staged_database_placement_services()
    original_capacity = services.cold.capacity.observe

    def observe_capacity_then_inject(
        cache_descriptor: int,
        *,
        allocated_bytes: int,
        reserved_bytes: int,
    ) -> DatabaseCapacityGate:
        capacity = original_capacity(
            cache_descriptor,
            allocated_bytes=allocated_bytes,
            reserved_bytes=reserved_bytes,
        )
        paths = fixture.database_placement_paths
        if fault == "source-inventory":
            member = fixture.runspec.payload.database.source_manifest.members[0]
            member_path = paths.source_root / member.source_path
            member_path.chmod(0o644)
            member_path.write_bytes(b"changed")
            member_path.chmod(0o444)
        elif fault == "source-binding":
            displaced = paths.source_root.with_name(f"{paths.source_root.name}.displaced")
            paths.source_root.rename(displaced)
            paths.source_root.mkdir(mode=0o700)
        elif fault in {"source-mount", "cache-mount"}:
            contents = paths.mountinfo.read_text()
            if fault == "source-mount":
                contents = contents.replace("overlay overlay ro", "ext4 source ro")
            else:
                contents = contents.replace("tmpfs shm rw", "ext4 cache rw")
            paths.mountinfo.write_text(contents)
        elif fault == "cache-binding":
            displaced = paths.cache_root.with_name(f"{paths.cache_root.name}.displaced")
            paths.cache_root.rename(displaced)
            paths.cache_root.mkdir(mode=0o700)
        elif fault == "lock":
            cache_lock = paths.cache_root / ".locks" / "cache.lock"
            cache_lock.rename(cache_lock.with_name("cache.lock.displaced"))
            cache_lock.write_bytes(b"")
            cache_lock.chmod(0o600)
        return capacity

    services = replace(
        services,
        cold=replace(
            services.cold,
            capacity=replace(services.cold.capacity, observe=observe_capacity_then_inject),
        ),
    )
    if fault == "result-publication":
        real_link = os.link

        def fail_result_link(
            source: str | bytes,
            destination: str | bytes,
            *,
            src_dir_fd: int | None = None,
            dst_dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> None:
            if os.fsdecode(destination) == "result.json":
                raise PermissionError("injected fallback result-publication fault")
            real_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )

        monkeypatch.setattr(os, "link", fail_result_link)

    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    with pytest.raises(DatabasePlacementError):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
            staged_services=services,
        )

    assert not result_path.exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.requested_policy == DatabaseAccessPolicy.STAGE_PREFERRED
    assert failure.capacity_gate is not None
    assert failure.capacity_gate.decision == "insufficient"
    assert failure.classification != "insufficient-capacity"


@pytest.mark.parametrize(
    "fault",
    [
        "temporary-unlink",
        "parent-fsync",
        "strict-reload",
        "parent-binding",
        "post-publication-lock",
        "post-publication-source-binding",
        "post-publication-cache-binding",
    ],
)
def test_stage_preferred_nonzero_placement_with_visible_result_never_reaches_science(
    fault: str,
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=1, f_frsize=1),
    )
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    injected = 0

    def fail_once_after_visibility[**P, T](
        operation: Callable[P, T],
        *,
        error: Exception,
    ) -> Callable[P, T]:
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            nonlocal injected
            if result_path.exists() and injected == 0:
                injected += 1
                raise error
            return operation(*args, **kwargs)

        return wrapped

    if fault == "temporary-unlink":
        monkeypatch.setattr(
            os,
            "unlink",
            fail_once_after_visibility(
                os.unlink,
                error=PermissionError("injected post-link temporary unlink fault"),
            ),
        )
    elif fault == "parent-fsync":
        monkeypatch.setattr(
            os,
            "fsync",
            fail_once_after_visibility(
                os.fsync,
                error=OSError("injected post-link parent fsync fault"),
            ),
        )
    elif fault == "strict-reload":
        real_open = os.open

        def fail_destination_reload(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            nonlocal injected
            if result_path.exists() and injected == 0 and os.fsdecode(path) == result_path.name:
                injected += 1
                raise PermissionError("injected first strict destination reload fault")
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(
            os,
            "open",
            fail_destination_reload,
        )
    elif fault in {
        "parent-binding",
        "post-publication-source-binding",
        "post-publication-cache-binding",
    }:
        target_path = {
            "parent-binding": result_path.parent,
            "post-publication-source-binding": fixture.database_placement_paths.source_root,
            "post-publication-cache-binding": fixture.database_placement_paths.cache_root,
        }[fault]
        real_lstat = Path.lstat

        def fail_binding(path: Path) -> os.stat_result:
            nonlocal injected
            if result_path.exists() and injected == 0 and path == target_path:
                injected += 1
                raise PermissionError(f"injected {fault} fault")
            return real_lstat(path)

        monkeypatch.setattr(
            Path,
            "lstat",
            fail_binding,
        )
    else:
        real_stat = os.stat

        def fail_lock_rebind(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            *,
            dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> os.stat_result:
            nonlocal injected
            try:
                real_stat(result_path)
                result_visible = True
            except FileNotFoundError:
                result_visible = False
            if result_visible and injected == 0 and dir_fd is not None and os.fsdecode(path) == "cache.lock":
                injected += 1
                raise PermissionError("injected post-publication-lock fault")
            return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(os, "stat", fail_lock_rebind)

    with pytest.raises(DatabasePlacementError):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert injected == 1
    visible = placement_evidence_io.load_database_direct_result(result_path)
    assert isinstance(visible, DatabaseCapacityFallbackResult)
    assert result_path.read_bytes() == canonical_database_capacity_fallback_result_bytes(visible)
    assert not failure_path.exists()
    replica_path = Path(fixture.runspec.payload.database.branches[0].scientific_mounts[0].source)
    assert not replica_path.exists()

    rendered = (
        render_phase_submission_intent(
            fixture.runspec,
            phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
            phase_runspec_document_sha256="a" * 64,
        )
        .actions[0]
        .script_body
    )
    rendered_lines = rendered.splitlines()
    status_index = rendered_lines.index("_placement_status=$?")
    guard_index = rendered_lines.index(
        'if [[ "$_placement_status" -ne 0 '
        '|| ! -f "$DATABASE_PLACEMENT_RESULT" '
        '|| -f "$DATABASE_PLACEMENT_FAILURE" ]]; then',
        status_index,
    )
    else_index = rendered_lines.index("else", guard_index)
    failure_branch = "\n".join(rendered_lines[guard_index:else_index])
    science_branch = "\n".join(rendered_lines[else_index:])

    assert failure_branch.count("execute-chunk") == 1
    assert '"$_placement_status"' in failure_branch
    assert "select-database-science-branch" not in failure_branch
    assert "select-database-science-branch" in science_branch


def test_stage_preferred_selector_strictly_reconciles_result_failure_and_contradiction(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=1, f_frsize=1),
    )
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=result_path,
        failure_path=failure_path,
    )

    assert (
        resolve_database_science_branch(
            fixture.runspec,
            action_id=fixture.action.action_id,
            result_path=result_path,
            failure_path=failure_path,
        )
        == "direct-capacity-fallback"
    )
    runner = CliRunner()
    invocation = runner.invoke(
        cli,
        [
            "preprocessing",
            "select-database-science-branch",
            "--phase-runspec",
            str(fixture.runspec_path),
            "--action-id",
            fixture.action.action_id,
            "--database-placement-result",
            str(result_path),
            "--database-placement-failure",
            str(failure_path),
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    assert invocation.output == "direct-capacity-fallback\n"

    failure_path.write_text("contradiction")
    with pytest.raises(DatabaseActionPlacementError, match="contradictory"):
        resolve_database_science_branch(
            fixture.runspec,
            action_id=fixture.action.action_id,
            result_path=result_path,
            failure_path=failure_path,
        )


def test_stage_preferred_selector_returns_placement_failure_only_for_strict_evidence(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    binding = fixture.runspec.payload.database
    failure_path = tmp_path / "placement" / "failure.json"
    publish_database_replica_cold_failure(
        DatabaseReplicaColdFailureEvidence(
            phase_run_id=fixture.runspec.phase_run_id,
            attempt_id=fixture.runspec.attempt_id,
            phase_runspec_digest=fixture.runspec.digest,
            action_id=fixture.action.action_id,
            database_set=binding.database_set,
            requested_policy=DatabaseAccessPolicy.STAGE_PREFERRED,
            source_manifest_sha256=binding.source_manifest_sha256,
            source_mount=None,
            cache_mount=None,
            capacity_gate=None,
            science_started=False,
            classification="capacity-observation-failed",
            error="capacity probe unavailable",
        ),
        failure_path,
    )

    assert (
        resolve_database_science_branch(
            fixture.runspec,
            action_id=fixture.action.action_id,
            result_path=tmp_path / "placement" / "result.json",
            failure_path=failure_path,
        )
        == "placement-failure"
    )


@pytest.mark.parametrize("condition", ["absent", "unknown-envelope", "forged-policy", "forged-manifest"])
def test_stage_preferred_selector_rejects_missing_or_forged_authority(
    condition: str,
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    if condition == "unknown-envelope":
        result_path.parent.mkdir(parents=True)
        result_path.write_text('{"unknown": {}}\n')
    elif condition != "absent":
        mapping = _fallback_for_fixture(fixture).to_mapping()
        inner = mapping["database_capacity_fallback_result"]
        assert isinstance(inner, dict)
        if condition == "forged-policy":
            inner["requested_policy"] = "direct"
        else:
            inner["source_manifest_sha256"] = "c" * 64
            observation = inner["pre_science_observation"]
            assert isinstance(observation, dict)
            observation["source_manifest_sha256"] = "c" * 64
        result_path.parent.mkdir(parents=True)
        result_path.write_text(json.dumps(mapping) + "\n")

    with pytest.raises(DatabaseActionPlacementError):
        resolve_database_science_branch(
            fixture.runspec,
            action_id=fixture.action.action_id,
            result_path=result_path,
            failure_path=failure_path,
        )


def test_stage_preferred_selector_rejects_forged_reserved_capacity(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    mapping = _fallback_for_fixture(fixture).to_mapping()
    inner = mapping["database_capacity_fallback_result"]
    assert isinstance(inner, dict)
    gate = inner["capacity_gate"]
    assert isinstance(gate, dict)
    gate["reserved_bytes"] = 1
    gate["required_bytes"] = 3
    result_path = tmp_path / "placement" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    result_path.chmod(0o444)

    with pytest.raises(DatabasePlacementError, match="does not match RunSpec authority"):
        resolve_database_science_branch(
            fixture.runspec,
            action_id=fixture.action.action_id,
            result_path=result_path,
            failure_path=tmp_path / "placement" / "failure.json",
        )


def test_stage_preferred_selector_rejects_forged_cache_filesystem(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    mapping = _fallback_for_fixture(fixture).to_mapping()
    inner = mapping["database_capacity_fallback_result"]
    assert isinstance(inner, dict)
    cache_mount = inner["cache_mount"]
    assert isinstance(cache_mount, dict)
    cache_mount["filesystem_type"] = "xfs"
    result_path = tmp_path / "placement" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    result_path.chmod(0o444)

    with pytest.raises(DatabasePlacementError, match="does not match RunSpec authority"):
        resolve_database_science_branch(
            fixture.runspec,
            action_id=fixture.action.action_id,
            result_path=result_path,
            failure_path=tmp_path / "placement" / "failure.json",
        )


@pytest.mark.parametrize("forgery", ["allocation", "source-mount"])
def test_stage_preferred_selector_rejects_contract_invalid_fallback_authority(
    forgery: str,
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    mapping = _fallback_for_fixture(fixture).to_mapping()
    inner = mapping["database_capacity_fallback_result"]
    assert isinstance(inner, dict)
    if forgery == "allocation":
        gate = inner["capacity_gate"]
        assert isinstance(gate, dict)
        gate["allocated_replica_bytes"] = 3
        gate["required_bytes"] = 3
    else:
        source_mount = inner["source_mount"]
        assert isinstance(source_mount, dict)
        source_mount["mount_point"] = "/run/bspp/unrelated"
    result_path = tmp_path / "placement" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    result_path.chmod(0o444)

    with pytest.raises((DatabaseActionPlacementError, DatabasePlacementError)):
        resolve_database_science_branch(
            fixture.runspec,
            action_id=fixture.action.action_id,
            result_path=result_path,
            failure_path=tmp_path / "placement" / "failure.json",
        )


def test_stage_preferred_renderer_predeclares_disjoint_science_mount_closures(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    intent = render_phase_submission_intent(
        fixture.runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="a" * 64,
    )
    script = intent.actions[0].script_body
    database = fixture.runspec.payload.database
    staging = database.staging
    assert staging is not None
    staged, fallback = database.branches
    mount_lines = tuple(
        line.strip().removesuffix(" \\")
        for line in script.splitlines()
        if line.strip().startswith("--container-mounts=")
    )

    assert len(mount_lines) == 10
    assert f"{staging.cache_root}:/run/bspp-acceptance-cache-root" in mount_lines[0]
    assert f"{staging.user_cache_root}:/run/bspp/database/cache" in mount_lines[2]
    assert "/run/bspp/database/source:ro" in mount_lines[2]
    assert all(
        "/run/bspp/database/" not in line
        for line in (mount_lines[1], mount_lines[3], mount_lines[4], mount_lines[5], mount_lines[8], mount_lines[9])
    )
    assert staged.scientific_mounts[0].source in mount_lines[6]
    assert "/run/bspp/database/replica-lease.lock:ro" in mount_lines[6]
    fallback_mounts = tuple(f"{mount.source}:{mount.target}:ro" for mount in fallback.scientific_mounts)
    assert all(fallback_mount in mount_lines[7] for fallback_mount in fallback_mounts)
    assert staged.scientific_mounts[0].source not in mount_lines[7]
    assert "/run/bspp/database/replica-lease.lock" not in mount_lines[7]
    # stage-preferred renders the bootstrap best-effort so a failed bootstrap
    # does not hard-stop the action before placement can select the direct
    # capacity fallback.
    assert f"mkdir -p --mode=0700 -- {staging.cache_root} || true" in script
    assert f"/run/bspp-acceptance-cache-root/users/{staging.unix_user} || true" in script
    assert "select-database-science-branch" in script
    assert 'case "$_database_science_branch" in' in script
    assert "staged)" in script
    assert "direct-capacity-fallback)" in script
    assert "placement-failure)" in script
    assert "unknown Database science branch" in script
    assert "_selector_status" in script
    placement_capture = script.index("_placement_status=$?")
    placement_restore = script.index("set -e", placement_capture)
    failure_guard = script.index('if [[ "$_placement_status" -ne 0 || ! -f "$DATABASE_PLACEMENT_RESULT"')
    selector = script.index("select-database-science-branch")
    selector_failure = script.index('if [[ "$_selector_status" -ne 0 ]]')
    selector_failure_end = script.index("fi", selector_failure)
    branch_case = script.index('case "$_database_science_branch" in')
    assert placement_capture < placement_restore < failure_guard < selector < branch_case
    selector_failure_block = script[selector_failure:selector_failure_end]
    assert "execute-chunk" in selector_failure_block
    status_option = selector_failure_block.index("--placement-process-status")
    assert selector_failure_block[status_option:].splitlines()[1].strip() == '"$_selector_status"'
    assert "--placement-process-status" in script
    assert '"$_placement_status"' in script
    subprocess.run(("bash", "-n"), input=script, text=True, check=True)


def test_rendered_selector_failure_status_publishes_failed_action_without_science_or_finalization(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    image = tmp_path / "qualified.sqsh"
    source_bundle = tmp_path / "source" / "qualified.tar.zst"
    source_bundle.parent.mkdir()
    image.write_bytes(b"qualified image\n")
    source_bundle.write_bytes(b"qualified source\n")
    qualification = fixture.runspec.cluster.preprocessing_runtime.qualification_tuple
    qualified = replace(
        qualification,
        cluster_image_path=str(image),
        cluster_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        source_bundle_path=str(source_bundle),
        source_bundle_sha256=hashlib.sha256(source_bundle.read_bytes()).hexdigest(),
    )
    selection = replace(
        fixture.runspec.cluster.preprocessing_runtime,
        qualification_tuple=qualified,
    )
    action = replace(
        fixture.action,
        payload=replace(
            fixture.action.payload,
            site=fixture.action.payload.site.model_copy(update={"container_image": str(image)}),
        ),
    )
    runspec = replace(
        fixture.runspec,
        cluster=replace(
            fixture.runspec.cluster,
            runtime_image=str(image),
            preprocessing_runtime=selection,
        ),
        payload=replace(fixture.runspec.payload, actions=(action,)),
    )
    fixture = replace(fixture, runspec=runspec)
    runspec_document = (json.dumps(runspec.to_mapping(), indent=2, sort_keys=True) + "\n").encode()
    cluster_runspec_path = (
        Path(runspec.cluster.staging_root)
        / "bspp-phase-runs"
        / runspec.phase_run_id
        / runspec.attempt_id
        / "phase-runspec.json"
    )
    cluster_runspec_path.parent.mkdir(parents=True)
    cluster_runspec_path.write_bytes(runspec_document)
    intent = render_phase_submission_intent(
        runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256=hashlib.sha256(runspec_document).hexdigest(),
    )
    plan = intent.actions[0]
    evidence_path = Path(plan.action_evidence_path)
    evidence_path.parent.mkdir(parents=True)
    result_path = evidence_path.with_name("database-placement-result.json")
    result_path.write_bytes(canonical_database_capacity_fallback_result_bytes(_fallback_for_fixture(fixture)))
    result_path.chmod(0o444)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
while [[ $# -gt 0 && "$1" == --* ]]; do shift; done
[[ "${1:-}" == "/usr/local/bin/entrypoint.sh" ]] || exit 125
shift
if [[ "${1:-}" == "mkdir" ]]; then
  exit 0
fi
[[ "${1:-}" == "bspp-orchestration-runtime" ]] || exit 126
shift
[[ "${1:-}" == "preprocessing" ]] || exit 126
case "${2:-}" in
  place-database)
    exit 0
    ;;
  stage-input)
    exit 0
    ;;
  select-database-science-branch)
    exit 37
    ;;
  execute-chunk)
    arguments=("$@")
    status=""
    for ((index=0; index < ${#arguments[@]}; index++)); do
      if [[ "${arguments[$index]}" == "--placement-process-status" ]]; then
        status="${arguments[$((index + 1))]}"
      fi
    done
    if [[ "$status" == "0" ]]; then
      : > "$BSPP_SCIENCE_MARKER"
      exit 99
    fi
    exec "$BSPP_REAL_RUNTIME" "$@"
    ;;
  finalize-chunk)
    : > "$BSPP_FINALIZATION_MARKER"
    exit 0
    ;;
esac
exit 127
"""
    )
    fake_srun.chmod(0o755)
    repo_root = Path(__file__).resolve().parents[1]
    package_roots = os.pathsep.join(str(path) for path in sorted((repo_root / "packages").glob("*/src")))
    science_marker = tmp_path / "science-started"
    finalization_marker = tmp_path / "finalization-started"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": package_roots,
        "BSPP_REAL_RUNTIME": str(repo_root / ".venv" / "bin" / "bspp-orchestration-runtime"),
        "BSPP_SCIENCE_MARKER": str(science_marker),
        "BSPP_FINALIZATION_MARKER": str(finalization_marker),
    }

    completed = subprocess.run(
        ("bash",),
        input=plan.script_body,
        text=True,
        env=environment,
        cwd=repo_root,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not science_marker.exists()
    assert not finalization_marker.exists()
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(evidence_path.read_text()))
    assert evidence.placement_process_status == 37
    placement = evidence.database_placement
    assert isinstance(placement, PreprocessingDatabasePlacementCommandFailureEvidence)
    assert placement.placement_process_status == 37
    assert placement.classification == "nonzero-after-result"
    assert evidence.command_outcomes == ()


def test_rendered_selector_result_to_failure_race_publishes_one_payload_free_failed_action(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    image = tmp_path / "qualified.sqsh"
    source_bundle = tmp_path / "source" / "qualified.tar.zst"
    source_bundle.parent.mkdir()
    image.write_bytes(b"qualified image\n")
    source_bundle.write_bytes(b"qualified source\n")
    qualification = fixture.runspec.cluster.preprocessing_runtime.qualification_tuple
    qualified = replace(
        qualification,
        cluster_image_path=str(image),
        cluster_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        source_bundle_path=str(source_bundle),
        source_bundle_sha256=hashlib.sha256(source_bundle.read_bytes()).hexdigest(),
    )
    action = replace(
        fixture.action,
        payload=replace(
            fixture.action.payload,
            site=fixture.action.payload.site.model_copy(update={"container_image": str(image)}),
        ),
    )
    runspec = replace(
        fixture.runspec,
        cluster=replace(
            fixture.runspec.cluster,
            runtime_image=str(image),
            preprocessing_runtime=replace(
                fixture.runspec.cluster.preprocessing_runtime,
                qualification_tuple=qualified,
            ),
        ),
        payload=replace(fixture.runspec.payload, actions=(action,)),
    )
    fixture = replace(fixture, runspec=runspec)
    runspec_document = (json.dumps(runspec.to_mapping(), indent=2, sort_keys=True) + "\n").encode()
    cluster_runspec_path = (
        Path(runspec.cluster.staging_root)
        / "bspp-phase-runs"
        / runspec.phase_run_id
        / runspec.attempt_id
        / "phase-runspec.json"
    )
    cluster_runspec_path.parent.mkdir(parents=True)
    cluster_runspec_path.write_bytes(runspec_document)
    plan = render_phase_submission_intent(
        runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256=hashlib.sha256(runspec_document).hexdigest(),
    ).actions[0]
    evidence_path = Path(plan.action_evidence_path)
    evidence_path.parent.mkdir(parents=True)
    result_path = evidence_path.with_name("database-placement-result.json")
    failure_path = evidence_path.with_name("database-placement-failure.json")
    result_path.write_bytes(canonical_database_capacity_fallback_result_bytes(_fallback_for_fixture(fixture)))
    result_path.chmod(0o444)
    binding = runspec.payload.database
    race_failure_path = tmp_path / "strict-race-failure.json"
    publish_database_replica_cold_failure(
        DatabaseReplicaColdFailureEvidence(
            phase_run_id=runspec.phase_run_id,
            attempt_id=runspec.attempt_id,
            phase_runspec_digest=runspec.digest,
            action_id=action.action_id,
            database_set=binding.database_set,
            requested_policy=DatabaseAccessPolicy.STAGE_PREFERRED,
            source_manifest_sha256=binding.source_manifest_sha256,
            source_mount=None,
            cache_mount=None,
            capacity_gate=None,
            science_started=False,
            classification="capacity-observation-failed",
            error="capacity probe became unavailable",
        ),
        race_failure_path,
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
all_arguments=("$@")
mounts=""
for argument in "${all_arguments[@]}"; do
  if [[ "$argument" == --container-mounts=* ]]; then
    mounts="${argument#--container-mounts=}"
  fi
done
while [[ $# -gt 0 && "$1" == --* ]]; do shift; done
[[ "${1:-}" == "/usr/local/bin/entrypoint.sh" ]] || exit 125
shift
if [[ "${1:-}" == "mkdir" ]]; then
  exit 0
fi
[[ "${1:-}" == "bspp-orchestration-runtime" ]] || exit 126
shift
[[ "${1:-}" == "preprocessing" ]] || exit 126
case "${2:-}" in
  place-database)
    exit 0
    ;;
  stage-input)
    exit 0
    ;;
  select-database-science-branch)
    result=""
    failure=""
    arguments=("$@")
    for ((index=0; index < ${#arguments[@]}; index++)); do
      if [[ "${arguments[$index]}" == "--database-placement-result" ]]; then
        result="${arguments[$((index + 1))]}"
      elif [[ "${arguments[$index]}" == "--database-placement-failure" ]]; then
        failure="${arguments[$((index + 1))]}"
      fi
    done
    rm -- "$result"
    mv -- "$BSPP_RACE_FAILURE" "$failure"
    printf '%s\n' 'placement-failure'
    exit 0
    ;;
  execute-chunk)
    printf '%s\n' "$mounts" >> "$BSPP_EXECUTE_MOUNTS"
    printf 'execute\n' >> "$BSPP_EXECUTE_CALLS"
    exec "$BSPP_REAL_RUNTIME" "$@"
    ;;
  finalize-chunk)
    : > "$BSPP_FINALIZATION_MARKER"
    exit 0
    ;;
esac
exit 127
"""
    )
    fake_srun.chmod(0o755)
    repo_root = Path(__file__).resolve().parents[1]
    package_roots = os.pathsep.join(str(path) for path in sorted((repo_root / "packages").glob("*/src")))
    execute_mounts = tmp_path / "execute-mounts"
    execute_calls = tmp_path / "execute-calls"
    finalization_marker = tmp_path / "finalization-started"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": package_roots,
        "BSPP_REAL_RUNTIME": str(repo_root / ".venv" / "bin" / "bspp-orchestration-runtime"),
        "BSPP_RACE_FAILURE": str(race_failure_path),
        "BSPP_EXECUTE_MOUNTS": str(execute_mounts),
        "BSPP_EXECUTE_CALLS": str(execute_calls),
        "BSPP_FINALIZATION_MARKER": str(finalization_marker),
    }

    completed = subprocess.run(
        ("bash",),
        input=plan.script_body,
        text=True,
        env=environment,
        cwd=repo_root,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert execute_calls.read_text().splitlines() == ["execute"]
    mounts = execute_mounts.read_text().strip()
    assert "/run/bspp/database/" not in mounts
    assert not finalization_marker.exists()
    assert not result_path.exists()
    assert failure_path.is_file()
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(evidence_path.read_text()))
    assert evidence.outcome == "failed"
    assert evidence.placement_process_status == 0
    assert evidence.database_placement.failure is not None
    assert evidence.database_placement.science_started is False
    assert evidence.command_outcomes == ()
    assert evidence.output_hashes == ()
    assert evidence.raw_search_evidence is None
    assert evidence.paired_evidence.record_lines is None
    assert evidence.paired_evidence.log_lines is None
    assert evidence.archive_evidence.tar_size_bytes is None
    assert evidence.archive_evidence.lz4_size_bytes is None
    assert evidence.archive_evidence.tar_members is None
    assert evidence.carry_forward_adoption is None


def _fallback_for_fixture(fixture: LocalExecutionFixture) -> DatabaseCapacityFallbackResult:
    binding = fixture.runspec.payload.database
    return DatabaseCapacityFallbackResult(
        phase_run_id=fixture.runspec.phase_run_id,
        attempt_id=fixture.runspec.attempt_id,
        phase_runspec_digest=fixture.runspec.digest,
        action_id=fixture.action.action_id,
        database_set=binding.database_set,
        requested_policy=DatabaseAccessPolicy.STAGE_PREFERRED,
        source_manifest_sha256=binding.source_manifest_sha256,
        branch_kind="direct-capacity-fallback",
        outcome=DatabasePlacementOutcomeKind.DIRECT_CAPACITY_FALLBACK,
        selected_container_root=SELECTED_DATABASE_ROOT,
        verification="metadata-verified",
        source_mount=_source_mount(),
        cache_mount=_cache_mount(),
        capacity_gate=DatabaseCapacityGate(
            available_user_bytes=1,
            allocated_replica_bytes=2,
            reserved_bytes=0,
            required_bytes=2,
            decision="insufficient",
        ),
        pre_science_observation=replace(
            _source_observation(),
            source_manifest_sha256=binding.source_manifest_sha256,
            members=binding.source_manifest.members,
        ),
    )


def _configure_fallback_execution(
    fixture: LocalExecutionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> DatabaseCapacityFallbackResult:
    configure_preprocessing_fakes(fixture, monkeypatch)
    result = _fallback_for_fixture(fixture)
    fixture.database_placement_result_path.chmod(0o600)
    fixture.database_placement_result_path.write_bytes(canonical_database_capacity_fallback_result_bytes(result))
    fixture.database_placement_result_path.chmod(0o444)
    return result


def test_stage_preferred_fallback_execution_requires_complete_unchanged_source_observation(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    result = _configure_fallback_execution(fixture, monkeypatch)

    executed = fixture.execute_preprocessing_chunk_action()

    assert executed.outcome == "succeeded"
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    placement = evidence.database_placement
    assert isinstance(placement, PreprocessingDatabasePlacementEvidence)
    assert placement.result == result
    assert isinstance(placement.post_science_observation, DatabasePostScienceObservation)
    assert placement.post_science_observation.matches(result.pre_science_observation)


def test_stage_preferred_fallback_execution_rejects_source_drift_before_output_acceptance(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    _configure_fallback_execution(fixture, monkeypatch)
    member = fixture.runspec.payload.database.source_manifest.members[0]
    member_path = fixture.database_source_root / member.source_path
    member_path.chmod(0o644)
    member_path.write_bytes(b"drifted")
    member_path.chmod(0o444)

    with pytest.raises(PreprocessingExecutionError, match="drifted"):
        fixture.execute_preprocessing_chunk_action()

    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert evidence.outcome == "failed"
    assert not evidence.output_hashes


def test_stage_preferred_fallback_execution_rejects_incomplete_source_observation(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _manifest_path, _cache_root = _stage_preferred_fixture(tmp_path / "work", cold_cache_root, monkeypatch)
    _configure_fallback_execution(fixture, monkeypatch)
    member = fixture.runspec.payload.database.source_manifest.members[0]
    (fixture.database_source_root / member.source_path).unlink()

    with pytest.raises(PreprocessingExecutionError):
        fixture.execute_preprocessing_chunk_action()

    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert evidence.outcome == "failed"
    assert not evidence.output_hashes
    placement = evidence.database_placement
    assert isinstance(placement, PreprocessingDatabasePlacementEvidence)
    assert isinstance(placement.post_science_observation, DatabasePostScienceObservation)
    assert placement.post_science_observation.errors
    assert "missing source member" in placement.post_science_observation.errors[0]
