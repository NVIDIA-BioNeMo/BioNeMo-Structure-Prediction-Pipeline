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

"""Public seams for direct-requested preprocessing Database Placement."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from click.testing import CliRunner

from bspp.orchestration.contract.database_placement import (
    DATABASE_SOURCE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
    DatabaseSetSelection,
    build_preprocessing_database_binding,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabasePlacementResult,
    DatabaseSourceMountFacts,
    DatabaseSourceObservation,
    canonical_database_placement_result_bytes,
    database_placement_result_digest,
    database_placement_result_from_mapping,
)
from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseAlias,
    DatabaseSetIdentity,
    DatabaseSourceManifest,
    DatabaseSourceMember,
    canonical_database_source_manifest_bytes,
)
from bspp.orchestration.contract.phase import phase_runspec_from_mapping
from bspp.orchestration.contract.preprocessing_action import (
    preprocessing_chunk_action_evidence_from_mapping,
)
from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.preprocessing import database_placement as placement_runtime
from bspp.orchestration.runtime.preprocessing._database_placement_paths import (
    PRODUCTION_DATABASE_PLACEMENT_PATHS,
    DatabasePlacementPaths,
)
from bspp.orchestration.runtime.preprocessing.database_placement import (
    DatabasePlacementError,
    _observe_direct_database_source,
    _place_direct_requested_database,
    load_database_placement_failure,
    load_database_placement_result,
    observe_direct_database_source,
    place_direct_requested_database,
)
from tests.support.preprocessing_execution import LocalExecutionFixture, preprocessing_execution_fixture


def test_security_sensitive_filesystem_authority_helpers_are_shared() -> None:
    from bspp.orchestration.runtime.preprocessing import (
        _database_placement_evidence_io as evidence_io,
    )
    from bspp.orchestration.runtime.preprocessing import (
        _database_source_observation as source_observation,
    )
    from bspp.orchestration.runtime.preprocessing import _filesystem_authority

    assert evidence_io._real_directory_stat is _filesystem_authority._real_directory_stat
    assert source_observation._real_directory_stat is _filesystem_authority._real_directory_stat
    assert evidence_io._stat_identity is _filesystem_authority._stat_identity
    assert source_observation._stat_identity is _filesystem_authority._stat_identity


def _regular_member(
    *,
    name: str = "primary",
    role: Literal["primary", "metagenomic"] = "primary",
    database_name: str | None = None,
    size: int = 7,
    mtime_ns: int = 123,
) -> DatabaseSourceMember:
    return DatabaseSourceMember(
        role=role,
        database_name=database_name or name,
        logical_name=name,
        source_path=name,
        source_kind="regular",
        resolved_path=name,
        resolved_kind="regular",
        size_bytes=size,
        mtime_ns=mtime_ns,
        alias_topology=(),
        preexisting_checksum=None,
    )


def _mount_facts() -> DatabaseSourceMountFacts:
    return DatabaseSourceMountFacts(
        mount_id=42,
        parent_mount_id=31,
        device_major=8,
        device_minor=2,
        mount_root="/db root",
        mount_point=DATABASE_SOURCE_ROOT,
        filesystem_type="lustre",
        mount_source="server:/db root",
        mount_options=("nodev", "ro"),
        super_options=("flock", "ro"),
        read_only=True,
    )


def _result() -> DatabasePlacementResult:
    digest = "a" * 64
    observation = DatabaseSourceObservation(
        source_container_root=DATABASE_SOURCE_ROOT,
        source_manifest_sha256=digest,
        members=(
            _regular_member(),
            _regular_member(name="metagenomic", role="metagenomic"),
        ),
        verification="metadata-verified",
    )
    return DatabasePlacementResult(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_runspec_digest="b" * 64,
        action_id="preprocessing-chunk-000000",
        database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"),
        requested_policy=DatabaseAccessPolicy.DIRECT,
        source_manifest_sha256=digest,
        branch_kind="direct-requested",
        outcome=DatabasePlacementOutcomeKind.DIRECT_REQUESTED,
        selected_container_root=SELECTED_DATABASE_ROOT,
        verification="metadata-verified",
        source_mount=_mount_facts(),
        pre_science_observation=observation,
    )


def _direct_fixture(root: Path) -> LocalExecutionFixture:
    fixture = preprocessing_execution_fixture(root)
    original = fixture.runspec.payload.database
    action = fixture.action.payload
    direct = build_preprocessing_database_binding(
        selection=DatabaseSetSelection(
            database_set=original.database_set,
            requested_policy=DatabaseAccessPolicy.DIRECT,
        ),
        source_manifest=original.source_manifest,
        source_manifest_projection=original.source_manifest_projection,
        staging=None,
        gpuserver_argv=action.gpuserver_argv,
        search_argv=action.search_argv,
    )
    runspec = replace(fixture.runspec, payload=replace(fixture.runspec.payload, database=direct))
    runspec_path = root / "phase-runspec.json"
    runspec_path.write_text(json.dumps(runspec.to_mapping(), indent=2, sort_keys=True) + "\n")
    return replace(fixture, runspec=runspec, runspec_path=runspec_path)


def _replace_direct_manifest(
    fixture: LocalExecutionFixture,
    manifest: DatabaseSourceManifest,
) -> LocalExecutionFixture:
    original = fixture.runspec.payload.database
    action = fixture.action.payload
    direct = build_preprocessing_database_binding(
        selection=DatabaseSetSelection(
            database_set=manifest.database_set,
            requested_policy=DatabaseAccessPolicy.DIRECT,
        ),
        source_manifest=manifest,
        source_manifest_projection=original.source_manifest_projection,
        staging=None,
        gpuserver_argv=action.gpuserver_argv,
        search_argv=action.search_argv,
    )
    runspec = replace(fixture.runspec, payload=replace(fixture.runspec.payload, database=direct))
    fixture.runspec_path.write_text(json.dumps(runspec.to_mapping(), indent=2, sort_keys=True) + "\n")
    return replace(fixture, runspec=runspec)


def _write_mountinfo(root: Path, mountinfo_path: Path) -> None:
    device = root.stat().st_dev
    mountinfo_path.write_text(
        f"42 31 {os.major(device)}:{os.minor(device)} / {DATABASE_SOURCE_ROOT} ro,nodev - ext4 /dev/test ro\n"
    )


def _stage_direct_source(
    fixture: LocalExecutionFixture,
    tmp_path: Path,
) -> tuple[Path, Path]:
    source_root = tmp_path / "protected-source"
    source_root.mkdir()
    for member in fixture.runspec.payload.database.source_manifest.members:
        member_path = source_root / member.source_path
        member_path.parent.mkdir(parents=True, exist_ok=True)
        member_path.write_bytes(b"x" * member.size_bytes)
        os.utime(member_path, ns=(member.mtime_ns, member.mtime_ns), follow_symlinks=False)
        member_path.chmod(0)
    manifest_path = tmp_path / "database-source-manifest.json"
    manifest_path.write_bytes(
        canonical_database_source_manifest_bytes(fixture.runspec.payload.database.source_manifest)
    )
    mountinfo_path = tmp_path / "mountinfo"
    _write_mountinfo(source_root, mountinfo_path)
    return manifest_path, source_root


def _stage_shared_alias_source(
    tmp_path: Path,
) -> tuple[LocalExecutionFixture, Path, Path, tuple[DatabaseSourceMember, ...]]:
    fixture = _direct_fixture(tmp_path / "work")
    original = fixture.runspec.payload.database.source_manifest
    members = tuple(
        replace(
            member,
            source_path=f"{member.logical_name}.alias",
            source_kind="symlink",
            resolved_path="shared-member",
            alias_topology=(DatabaseAlias(path=f"{member.logical_name}.alias", target="shared-member"),),
        )
        for member in original.members
    )
    manifest = replace(original, members=members)
    fixture = _replace_direct_manifest(fixture, manifest)
    source_root = tmp_path / "protected source"
    source_root.mkdir()
    shared = source_root / "shared-member"
    shared.write_bytes(b"x")
    os.utime(shared, ns=(1, 1))
    shared.chmod(0)
    for member in members:
        (source_root / member.source_path).symlink_to("shared-member")
    manifest_path = tmp_path / "database-source-manifest.json"
    manifest_path.write_bytes(canonical_database_source_manifest_bytes(manifest))
    mountinfo_path = tmp_path / "mountinfo"
    _write_mountinfo(source_root, mountinfo_path)
    return fixture, manifest_path, source_root, members


def _test_paths(source_root: Path, mountinfo_path: Path) -> DatabasePlacementPaths:
    return replace(
        PRODUCTION_DATABASE_PLACEMENT_PATHS,
        source_root=source_root,
        selected_root=source_root,
        mountinfo=mountinfo_path,
    )


def _place_test_direct(
    fixture: LocalExecutionFixture,
    manifest_path: Path,
    source_root: Path,
    result_path: Path,
    *,
    failure_path: Path | None = None,
) -> DatabasePlacementResult:
    return _place_direct_requested_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        paths=_test_paths(source_root, manifest_path.parent / "mountinfo"),
    )


def test_database_placement_result_has_strict_canonical_round_trip_and_digest() -> None:
    result = _result()
    expected = (json.dumps(result.to_mapping(), indent=2, sort_keys=True) + "\n").encode()

    assert database_placement_result_from_mapping(result.to_mapping()) == result
    assert canonical_database_placement_result_bytes(result) == expected
    assert database_placement_result_digest(result) == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    ["missing", "missing-schema", "unknown", "alternate-policy", "alternate-outcome"],
)
def test_database_placement_result_rejects_noncanonical_shapes(mutation: str) -> None:
    mapping = deepcopy(_result().to_mapping())
    inner = mapping["database_placement_result"]
    assert isinstance(inner, dict)
    if mutation == "missing":
        del inner["phase_runspec_digest"]
    elif mutation == "missing-schema":
        del inner["schema_version"]
    elif mutation == "unknown":
        inner["cache_root"] = "/cache"
    elif mutation == "alternate-policy":
        inner["requested_policy"] = "stage-preferred"
    else:
        inner["outcome"] = "direct-capacity-fallback"

    with pytest.raises(ValueError):
        database_placement_result_from_mapping(mapping)


@pytest.mark.parametrize(
    "mutation",
    ["missing-role", "missing-database-name-target", "duplicate-logical-name"],
)
def test_database_placement_result_rejects_incomplete_observation_topology(mutation: str) -> None:
    mapping = deepcopy(_result().to_mapping())
    inner = mapping["database_placement_result"]
    assert isinstance(inner, dict)
    observation = inner["pre_science_observation"]
    assert isinstance(observation, dict)
    members = observation["members"]
    assert isinstance(members, list)
    if mutation == "missing-role":
        members.pop()
    elif mutation == "missing-database-name-target":
        metagenomic = members[1]
        assert isinstance(metagenomic, dict)
        metagenomic["database_name"] = "missing-target"
    else:
        primary = members[0]
        metagenomic = members[1]
        assert isinstance(primary, dict)
        assert isinstance(metagenomic, dict)
        metagenomic["logical_name"] = primary["logical_name"]

    with pytest.raises(ValueError):
        database_placement_result_from_mapping(mapping)


@pytest.mark.parametrize("mount_point", ["/unrelated", f"{DATABASE_SOURCE_ROOT}-prefix"])
def test_database_placement_result_rejects_mounts_not_containing_protected_source(mount_point: str) -> None:
    mapping = deepcopy(_result().to_mapping())
    inner = mapping["database_placement_result"]
    assert isinstance(inner, dict)
    source_mount = inner["source_mount"]
    assert isinstance(source_mount, dict)
    source_mount["mount_point"] = mount_point

    with pytest.raises(ValueError):
        database_placement_result_from_mapping(mapping)


def test_direct_placement_rejects_wrong_action_before_manifest_or_publication(tmp_path: Path) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    result_path = tmp_path / "result.json"

    with pytest.raises(DatabasePlacementError, match="does not match the sole declared action"):
        place_direct_requested_database(
            fixture.runspec,
            action_id="preprocessing-chunk-999999",
            source_manifest_path=tmp_path / "missing-manifest.json",
            result_path=result_path,
        )

    assert not result_path.exists()


def test_direct_placement_rejects_noncanonical_staged_manifest_before_observation(
    tmp_path: Path,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path = tmp_path / "database-source-manifest.json"
    manifest_path.write_bytes(
        canonical_database_source_manifest_bytes(fixture.runspec.payload.database.source_manifest) + b"\n"
    )
    result_path = tmp_path / "result.json"

    with pytest.raises(DatabasePlacementError, match="canonical bytes"):
        place_direct_requested_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
        )

    assert not result_path.exists()


def test_direct_placement_rejects_non_direct_policy_before_manifest_or_publication(tmp_path: Path) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work", direct_policy=False)
    result_path = tmp_path / "result.json"

    with pytest.raises(DatabasePlacementError, match="requested policy 'direct'"):
        place_direct_requested_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=tmp_path / "missing-manifest.json",
            result_path=result_path,
        )

    assert not result_path.exists()


@pytest.mark.parametrize("mutation", ["source-root", "database-set"])
def test_direct_placement_rejects_canonical_but_mismatched_staged_manifest_before_observation(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    expected = fixture.runspec.payload.database.source_manifest
    if mutation == "source-root":
        staged = replace(expected, source_root="/different/source")
    else:
        staged = replace(
            expected,
            database_set=DatabaseSetIdentity(identifier=expected.database_set.identifier, version="different"),
        )
    manifest_path = tmp_path / "database-source-manifest.json"
    manifest_path.write_bytes(canonical_database_source_manifest_bytes(staged))
    result_path = tmp_path / "result.json"

    with pytest.raises(DatabasePlacementError, match="does not match RunSpec authority"):
        place_direct_requested_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
        )

    assert not result_path.exists()


@pytest.mark.parametrize("mutation", ["wrong-digest", "extra-branch"])
def test_direct_placement_runspec_loader_rejects_wrong_manifest_digest_and_extra_branch(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    mapping = deepcopy(fixture.runspec.to_mapping())
    database = mapping["payload"]["database"]
    assert isinstance(database, dict)
    if mutation == "wrong-digest":
        database["source_manifest_sha256"] = "f" * 64
    else:
        database["branches"].append(deepcopy(database["branches"][0]))

    with pytest.raises(ValueError):
        phase_runspec_from_mapping(mapping)


def test_direct_placement_observes_regular_members_without_reading_payload_and_publishes_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    result_path = tmp_path / "result.json"

    result = _place_test_direct(fixture, manifest_path, source_root, result_path)

    assert result.phase_runspec_digest == fixture.runspec.digest
    assert result.pre_science_observation.members == fixture.runspec.payload.database.source_manifest.members
    assert result.source_mount.mount_point == DATABASE_SOURCE_ROOT
    assert stat.S_IMODE(result_path.stat().st_mode) == 0o444
    assert load_database_placement_result(result_path) == result


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "extra-directory", "unrecorded-symlink", "special", "size", "mtime", "hardlink"],
)
def test_direct_placement_rejects_incomplete_or_changed_source_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    members = fixture.runspec.payload.database.source_manifest.members
    first = source_root / members[0].source_path
    second = source_root / members[1].source_path
    if mutation == "missing":
        first.unlink()
    elif mutation == "extra":
        (source_root / "unexpected").write_bytes(b"x")
    elif mutation == "extra-directory":
        unexpected = source_root / "unexpected"
        unexpected.mkdir()
        (unexpected / "leaf").write_bytes(b"x")
    elif mutation == "unrecorded-symlink":
        first.unlink()
        first.symlink_to(members[1].source_path)
    elif mutation == "special":
        first.unlink()
        os.mkfifo(first)
    elif mutation == "size":
        first.chmod(0o600)
        first.write_bytes(b"xx")
        os.utime(first, ns=(members[0].mtime_ns, members[0].mtime_ns))
    elif mutation == "mtime":
        os.utime(first, ns=(2, 2))
    else:
        second.unlink()
        os.link(first, second)

    result_path = tmp_path / "result.json"
    with pytest.raises(DatabasePlacementError):
        _place_test_direct(fixture, manifest_path, source_root, result_path)
    assert not result_path.exists()


def test_direct_placement_allows_explicit_aliases_to_one_shared_resolved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, source_root, members = _stage_shared_alias_source(tmp_path)

    result = _place_test_direct(
        fixture,
        manifest_path,
        source_root,
        tmp_path / "result.json",
    )

    assert result.pre_science_observation.members == members


@pytest.mark.parametrize("mutation", ["target-mismatch", "escape", "cycle"])
def test_direct_placement_rejects_alias_topology_mismatch_escape_and_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture, manifest_path, source_root, members = _stage_shared_alias_source(tmp_path)
    first = source_root / members[0].source_path
    first.unlink()
    if mutation == "target-mismatch":
        other = source_root / "other-member"
        other.write_bytes(b"x")
        os.utime(other, ns=(1, 1))
        first.symlink_to("other-member")
    elif mutation == "escape":
        first.symlink_to("../outside")
    else:
        second = source_root / members[1].source_path
        second.unlink()
        first.symlink_to(members[1].source_path)
        second.symlink_to(members[0].source_path)

    result_path = tmp_path / "result.json"
    with pytest.raises(DatabasePlacementError):
        _place_test_direct(fixture, manifest_path, source_root, result_path)
    assert not result_path.exists()


def test_direct_placement_decodes_mountinfo_and_selects_longest_component_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    mountinfo_path = tmp_path / "mountinfo"
    device = source_root.stat().st_dev
    mountinfo_path.write_text(
        f"1 0 {os.major(device)}:{os.minor(device)} / / rw - ext4 /dev/root rw\n"
        f"42 31 {os.major(device)}:{os.minor(device)} / {DATABASE_SOURCE_ROOT} ro,nodev "
        r"- lustre server:\040share\134path flock,ro"
        "\n"
    )

    result = _place_test_direct(
        fixture,
        manifest_path,
        source_root,
        tmp_path / "result.json",
    )

    assert result.source_mount.mount_id == 42
    assert result.source_mount.mount_point == DATABASE_SOURCE_ROOT
    assert result.source_mount.mount_source == "server: share\\path"


@pytest.mark.parametrize(
    "mutation",
    ["rw", "device", "malformed-escape", "noncontaining", "ambiguity", "malformed", "missing-procfs"],
)
def test_direct_placement_rejects_invalid_or_ambiguous_mount_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    mountinfo_path = tmp_path / "mountinfo"
    line = mountinfo_path.read_text()
    if mutation == "rw":
        mountinfo_path.write_text(line.replace(" ro,nodev ", " rw,nodev "))
    elif mutation == "device":
        device = source_root.stat().st_dev
        mountinfo_path.write_text(
            line.replace(f"{os.major(device)}:{os.minor(device)}", f"{os.major(device) + 1}:{os.minor(device)}")
        )
    elif mutation == "malformed-escape":
        mountinfo_path.write_text(line.replace(DATABASE_SOURCE_ROOT, f"{DATABASE_SOURCE_ROOT}\\999"))
    elif mutation == "noncontaining":
        mountinfo_path.write_text(line.replace(DATABASE_SOURCE_ROOT, "/not-containing"))
    elif mutation == "ambiguity":
        mountinfo_path.write_text(line + line.replace("42 31", "43 31"))
    elif mutation == "malformed":
        mountinfo_path.write_text("42 31 8:1 /\n")
    else:
        mountinfo_path.unlink()

    result_path = tmp_path / "result.json"
    with pytest.raises(DatabasePlacementError):
        _place_test_direct(fixture, manifest_path, source_root, result_path)
    assert not result_path.exists()


def test_direct_placement_rejects_mount_authority_change_during_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    mountinfo_path = tmp_path / "mountinfo"
    real_open = io.open
    mount_reads = 0

    def changing_mountinfo(
        path: str | bytes | int | os.PathLike[str] | os.PathLike[bytes],
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: object | None = None,
    ) -> object:
        nonlocal mount_reads
        if not isinstance(path, int) and Path(path) == mountinfo_path and "r" in mode:
            mount_reads += 1
            if mount_reads > 1:
                with real_open(path, mode, buffering, encoding, errors, newline, closefd, opener) as handle:
                    return io.StringIO(handle.read().replace("42 31", "43 31"))
        return real_open(path, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(io, "open", changing_mountinfo)
    result_path = tmp_path / "result.json"
    failure_path = tmp_path / "failure.json"

    with pytest.raises(DatabasePlacementError, match="mount"):
        _place_test_direct(
            fixture,
            manifest_path,
            source_root,
            result_path,
            failure_path=failure_path,
        )
    assert not result_path.exists()
    assert load_database_placement_failure(failure_path).classification == "source-mount-invalid"


def test_post_science_complete_tree_rejects_member_identity_race_after_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    result = _place_test_direct(
        fixture,
        manifest_path,
        source_root,
        tmp_path / "result.json",
    )
    first_member = fixture.runspec.payload.database.source_manifest.members[0]
    first_path = source_root / first_member.source_path
    real_listdir = os.listdir
    mutated = False

    def mutate_before_closure_walk(path: int | str | os.PathLike[str]) -> list[str]:
        nonlocal mutated
        if isinstance(path, int) and not mutated:
            mutated = True
            first_path.chmod(0o600)
            first_path.write_bytes(b"identity-race")
            os.utime(first_path, ns=(first_member.mtime_ns, first_member.mtime_ns))
        return real_listdir(path)

    monkeypatch.setattr(placement_runtime.os, "listdir", mutate_before_closure_walk)

    post = _observe_direct_database_source(
        fixture.runspec,
        result,
        paths=_test_paths(source_root, tmp_path / "mountinfo"),
    )

    assert mutated
    assert any("changed during observation" in error for error in post.errors)
    assert not post.matches(result.pre_science_observation)


def test_post_science_observation_does_not_convert_result_authority_failure(
    tmp_path: Path,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    forged_result = replace(
        load_database_placement_result(fixture.database_placement_result_path),
        phase_runspec_digest="0" * 64,
    )

    with pytest.raises(DatabasePlacementError, match="does not match RunSpec authority"):
        observe_direct_database_source(fixture.runspec, forged_result)


def test_direct_placement_rejects_source_root_substitution_during_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    mountinfo_path = tmp_path / "mountinfo"
    displaced_root = tmp_path / "displaced-source"
    real_open = io.open
    substituted = False

    def substituting_mountinfo_read(
        path: str | bytes | int | os.PathLike[str] | os.PathLike[bytes],
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: object | None = None,
    ) -> object:
        nonlocal substituted
        if not isinstance(path, int) and Path(path) == mountinfo_path and "r" in mode and not substituted:
            substituted = True
            source_root.rename(displaced_root)
            source_root.mkdir()
            for member in fixture.runspec.payload.database.source_manifest.members:
                member_path = source_root / member.source_path
                member_path.parent.mkdir(parents=True, exist_ok=True)
                member_path.write_bytes(b"x" * member.size_bytes)
                os.utime(member_path, ns=(member.mtime_ns, member.mtime_ns))
                member_path.chmod(0)
        return real_open(path, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(io, "open", substituting_mountinfo_read)
    result_path = tmp_path / "result.json"

    with pytest.raises(DatabasePlacementError, match="root"):
        _place_test_direct(fixture, manifest_path, source_root, result_path)
    assert not result_path.exists()


def test_direct_placement_rejects_symlinked_protected_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    linked_root = tmp_path / "linked-source"
    linked_root.symlink_to(source_root, target_is_directory=True)
    mountinfo_path = tmp_path / "linked-mountinfo"
    _write_mountinfo(source_root, mountinfo_path)
    result_path = tmp_path / "result.json"

    with pytest.raises(DatabasePlacementError, match="real non-symlink directory"):
        _place_direct_requested_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            paths=_test_paths(linked_root, mountinfo_path),
        )
    assert not result_path.exists()


def test_direct_placement_rejects_nested_directory_mutation_during_complete_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    original = fixture.runspec.payload.database.source_manifest
    manifest = replace(
        original,
        members=tuple(
            replace(member, source_path=f"nested/{member.source_path}", resolved_path=f"nested/{member.resolved_path}")
            for member in original.members
        ),
    )
    fixture = _replace_direct_manifest(fixture, manifest)
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    nested = source_root / "nested"
    nested_inode = nested.stat().st_ino
    real_listdir = placement_runtime.os.listdir
    mutated = False

    def mutating_listdir(path: int | os.PathLike[str] | str) -> list[str]:
        nonlocal mutated
        entries = real_listdir(path)
        if isinstance(path, int) and os.fstat(path).st_ino == nested_inode and not mutated:
            mutated = True
            (nested / "late-entry").write_bytes(b"late")
        return entries

    monkeypatch.setattr(placement_runtime.os, "listdir", mutating_listdir)
    result_path = tmp_path / "result.json"

    with pytest.raises(DatabasePlacementError, match="changed during observation"):
        _place_test_direct(fixture, manifest_path, source_root, result_path)
    assert not result_path.exists()


def test_direct_placement_rejects_leaf_substitution_during_component_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    first_member = fixture.runspec.payload.database.source_manifest.members[0]
    first_path = source_root / first_member.source_path
    real_stat = placement_runtime.os.stat
    substituted = False

    def substituting_stat(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal substituted
        info = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if path == first_member.source_path and dir_fd is not None and not follow_symlinks and not substituted:
            substituted = True
            first_path.unlink()
            first_path.write_bytes(b"replacement")
        return info

    monkeypatch.setattr(placement_runtime.os, "stat", substituting_stat)
    result_path = tmp_path / "result.json"

    with pytest.raises(DatabasePlacementError, match="changed during observation"):
        _place_test_direct(fixture, manifest_path, source_root, result_path)
    assert not result_path.exists()


@pytest.mark.parametrize("failure", ["preexisting", "link", "file-fsync"])
def test_direct_placement_publication_failures_leave_no_claimed_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    result_path = tmp_path / "result.json"
    if failure == "preexisting":
        result_path.write_bytes(b"existing")
    elif failure == "link":

        def fail_link(*args: object, **kwargs: object) -> None:
            raise OSError("injected link failure")

        monkeypatch.setattr(placement_runtime.os, "link", fail_link)
    else:

        def fail_fsync(descriptor: int) -> None:
            raise OSError("injected file fsync failure")

        monkeypatch.setattr(placement_runtime.os, "fsync", fail_fsync)

    with pytest.raises(DatabasePlacementError):
        _place_test_direct(fixture, manifest_path, source_root, result_path)

    if failure == "preexisting":
        assert result_path.read_bytes() == b"existing"
    else:
        assert not result_path.exists()


def test_direct_placement_parent_fsync_failure_leaves_visible_result_blocking_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    result_path = tmp_path / "result.json"
    failure_path = tmp_path / "failure.json"
    real_fsync = placement_runtime.os.fsync
    calls = 0

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(placement_runtime.os, "fsync", fail_parent_fsync)

    with pytest.raises(DatabasePlacementError, match="parent fsync failure"):
        _place_test_direct(
            fixture,
            manifest_path,
            source_root,
            result_path,
            failure_path=failure_path,
        )

    assert result_path.exists()
    assert not failure_path.exists()
    assert stat.S_IMODE(result_path.stat().st_mode) == 0o444
    with pytest.raises(DatabasePlacementError):
        _place_test_direct(fixture, manifest_path, source_root, result_path)


def test_direct_placement_rejects_symlinked_publication_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    real_parent = tmp_path / "real-evidence"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-evidence"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    result_path = linked_parent / "result.json"

    with pytest.raises(DatabasePlacementError, match="parent"):
        _place_test_direct(fixture, manifest_path, source_root, result_path)
    assert not result_path.exists()


def test_direct_placement_rejects_publication_parent_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    result_parent = tmp_path / "evidence"
    result_parent.mkdir()
    displaced_parent = tmp_path / "displaced-evidence"
    result_path = result_parent / "result.json"
    real_link = placement_runtime.os.link
    substituted = False

    def substituting_link(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal substituted
        if not substituted:
            substituted = True
            result_parent.rename(displaced_parent)
            result_parent.mkdir()
            if kwargs.get("src_dir_fd") is None:
                displaced_temporary = displaced_parent / Path(source).name
                replacement_temporary = result_parent / Path(source).name
                replacement_temporary.write_bytes(displaced_temporary.read_bytes())
                replacement_temporary.chmod(0o444)
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(placement_runtime.os, "link", substituting_link)

    with pytest.raises(DatabasePlacementError, match="parent"):
        _place_test_direct(fixture, manifest_path, source_root, result_path)
    assert not result_path.exists()


def test_place_database_cli_has_exact_bounded_flags() -> None:
    runner = CliRunner()

    help_result = runner.invoke(cli, ["preprocessing", "place-database", "--help"])
    assert help_result.exit_code == 0
    for option in (
        "--phase-runspec",
        "--action-id",
        "--source-manifest",
        "--write-result",
        "--write-failure-evidence",
    ):
        assert option in help_result.output
    for forbidden in ("--policy", "--branch", "--source-root", "--mountinfo", "--cache", "--replica"):
        assert forbidden not in help_result.output


def test_place_database_cli_classifies_missing_manifest_and_consumer_publishes_failed_action(
    tmp_path: Path,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    missing_manifest_path = tmp_path / "missing-database-source-manifest.json"
    result_path = tmp_path / "result.json"
    failure_path = tmp_path / "failure.json"
    evidence_path = tmp_path / "action-evidence.json"
    runner = CliRunner()

    placement = runner.invoke(
        cli,
        [
            "preprocessing",
            "place-database",
            "--phase-runspec",
            str(fixture.runspec_path),
            "--action-id",
            fixture.action.action_id,
            "--source-manifest",
            str(missing_manifest_path),
            "--write-result",
            str(result_path),
            "--write-failure-evidence",
            str(failure_path),
        ],
    )

    assert placement.exit_code != 0
    assert not result_path.exists()
    failure = load_database_placement_failure(failure_path)
    assert failure.classification == "source-manifest-invalid"
    assert failure.science_started is False

    execution = runner.invoke(
        cli,
        [
            "preprocessing",
            "execute-chunk",
            "--phase-runspec",
            str(fixture.runspec_path),
            "--action-id",
            fixture.action.action_id,
            "--write-evidence",
            str(evidence_path),
            "--database-placement-result",
            str(result_path),
            "--database-placement-failure",
            str(failure_path),
            "--placement-process-status",
            str(placement.exit_code),
        ],
    )

    assert execution.exit_code != 0
    assert f"source-manifest-invalid (status={placement.exit_code})" in execution.output
    action_evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(evidence_path.read_text()))
    assert action_evidence.outcome == "failed"
    assert action_evidence.placement_process_status == placement.exit_code
    assert action_evidence.database_placement.science_started is False
    assert action_evidence.database_placement.failure == failure
    assert action_evidence.error is not None
    assert f"source-manifest-invalid (status={placement.exit_code})" in action_evidence.error
    assert action_evidence.command_outcomes == ()
    assert action_evidence.output_hashes == ()
    with pytest.raises(ValueError, match="payload-free"):
        replace(
            action_evidence,
            paired_evidence=replace(action_evidence.paired_evidence, record_lines=("forged",)),
        )
    with pytest.raises(ValueError, match="payload-free"):
        replace(
            action_evidence,
            archive_evidence=replace(action_evidence.archive_evidence, lz4_size_bytes=1),
        )


def test_direct_placement_publishes_classified_manifest_failure_exclusively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    manifest_path.write_bytes(b"{}\n")
    result_path = tmp_path / "result.json"
    failure_path = tmp_path / "failure.json"

    with pytest.raises(DatabasePlacementError):
        _place_test_direct(
            fixture,
            manifest_path,
            source_root,
            result_path,
            failure_path=failure_path,
        )

    assert not result_path.exists()
    failure = load_database_placement_failure(failure_path)
    assert failure.classification == "source-manifest-invalid"
    assert failure.science_started is False
    assert stat.S_IMODE(failure_path.stat().st_mode) == 0o444


@pytest.mark.parametrize(
    "classification",
    [
        "authority-invalid",
        "source-manifest-invalid",
        "source-mount-invalid",
        "source-inventory-invalid",
        "result-publication-failed",
    ],
)
def test_direct_placement_publishes_each_runtime_failure_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    classification: str,
) -> None:
    fixture = _direct_fixture(tmp_path / "work")
    manifest_path, source_root = _stage_direct_source(fixture, tmp_path)
    result_path = tmp_path / "result.json"
    failure_path = tmp_path / "failure.json"
    action_id = fixture.action.action_id
    if classification == "authority-invalid":
        action_id = "preprocessing-chunk-999999"
    elif classification == "source-manifest-invalid":
        manifest_path.write_bytes(b"{}\n")
    elif classification == "source-mount-invalid":
        mountinfo_path = tmp_path / "mountinfo"
        mountinfo_path.write_text(mountinfo_path.read_text().replace(" ro,nodev ", " rw,nodev "))
    elif classification == "source-inventory-invalid":
        first = fixture.runspec.payload.database.source_manifest.members[0]
        (source_root / first.source_path).unlink()
    else:
        real_link = placement_runtime.os.link
        calls = 0

        def fail_result_link_once(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected Result link failure")
            real_link(*args, **kwargs)

        monkeypatch.setattr(placement_runtime.os, "link", fail_result_link_once)

    with pytest.raises(DatabasePlacementError):
        _place_direct_requested_database(
            fixture.runspec,
            action_id=action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
            paths=_test_paths(source_root, tmp_path / "mountinfo"),
        )

    assert not result_path.exists()
    failure = load_database_placement_failure(failure_path)
    assert failure.classification == classification
    assert failure.action_id == action_id
    assert failure.science_started is False
    assert stat.S_IMODE(failure_path.stat().st_mode) == 0o444
