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

"""Public Database Set Provisioning contract, runtime, and CLI behavior."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from bspp.orchestration.contract.database_set_provisioning import (
    canonical_database_source_manifest_bytes,
    database_set_declaration_from_mapping,
    database_set_provisioning_evidence_from_mapping,
    database_source_manifest_digest,
    database_source_manifest_from_mapping,
    load_database_set_declaration,
)
from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.preprocessing.database_provisioning import (
    DatabaseProvisioningError,
    provision_database_set,
)


def _declaration_mapping(source_root: Path) -> dict[str, object]:
    return {
        "database_set_declaration": {
            "schema_version": 1,
            "database_set": {"identifier": "afdb-search", "version": "2023-02"},
            "source_root": str(source_root),
            "roles": [
                {
                    "role": "primary",
                    "database_name": "uniref30_2302_db",
                    "members": [
                        {
                            "logical_name": "uniref30_2302_db",
                            "source_path": "uniref30_2302_db_pad",
                            "preexisting_checksum": None,
                        },
                        {
                            "logical_name": "uniref30_2302_db.dbtype",
                            "source_path": "uniref30_2302_db_pad.dbtype",
                            "preexisting_checksum": {
                                "algorithm": "sha256",
                                "value": "a" * 64,
                            },
                        },
                    ],
                },
                {
                    "role": "metagenomic",
                    "database_name": "colabfold_envdb_202108_db",
                    "members": [
                        {
                            "logical_name": "colabfold_envdb_202108_db",
                            "source_path": "colabfold_envdb_202108_db",
                            "preexisting_checksum": None,
                        }
                    ],
                },
            ],
        }
    }


def test_database_set_declaration_has_a_strict_canonical_public_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    mapping = _declaration_mapping(source)
    declaration = database_set_declaration_from_mapping(mapping)
    declaration_path = tmp_path / "database-set.json"
    declaration_path.write_text(json.dumps(mapping))

    assert declaration.to_mapping() == mapping
    assert load_database_set_declaration(declaration_path) == declaration


def test_non_sha_checksum_round_trips_into_the_published_manifest_without_verification(tmp_path: Path) -> None:
    _, mapping, _ = _source_fixture(tmp_path)
    inner = mapping["database_set_declaration"]
    assert isinstance(inner, dict)
    roles = inner["roles"]
    assert isinstance(roles, list)
    checksum = {"algorithm": "md5", "value": "d41d8cd98f00b204e9800998ecf8427e"}
    roles[0]["members"][1]["preexisting_checksum"] = checksum
    declaration = database_set_declaration_from_mapping(mapping)

    evidence = provision_database_set(
        declaration,
        manifest_root=tmp_path / "manifests",
        evidence_path=tmp_path / "evidence.json",
    )

    manifest = database_source_manifest_from_mapping(json.loads(Path(evidence.manifest_path).read_text()))
    member = next(item for item in manifest.members if item.logical_name == "uniref30_2302_db.dbtype")
    assert declaration.to_mapping() == mapping
    assert member.preexisting_checksum is not None
    assert member.preexisting_checksum.to_mapping() == checksum


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("algorithm", "", "algorithm must be a safe lowercase token"),
        ("algorithm", "MD5", "algorithm must be a safe lowercase token"),
        ("algorithm", "../md5", "algorithm must be a safe lowercase token"),
        ("algorithm", "a" * 33, "algorithm must be a safe lowercase token"),
        ("value", "", "value must be a safe non-empty token"),
        ("value", "digest with spaces", "value must be a safe non-empty token"),
        ("value", "digest\\escape", "value must be a safe non-empty token"),
        ("value", "a" * 257, "value must be a safe non-empty token"),
    ],
)
def test_preexisting_checksum_rejects_malformed_or_unbounded_tokens(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    mapping = _declaration_mapping(tmp_path / "source")
    inner = mapping["database_set_declaration"]
    assert isinstance(inner, dict)
    roles = inner["roles"]
    assert isinstance(roles, list)
    checksum = roles[0]["members"][1]["preexisting_checksum"]
    assert isinstance(checksum, dict)
    checksum[field] = value

    with pytest.raises(ValueError, match=message):
        database_set_declaration_from_mapping(mapping)


@pytest.mark.parametrize("duplicate_scope", ["within-role", "across-roles"])
def test_database_set_declaration_rejects_duplicate_logical_member_authority(
    tmp_path: Path,
    duplicate_scope: str,
) -> None:
    mapping = _declaration_mapping(tmp_path / "source")
    inner = mapping["database_set_declaration"]
    assert isinstance(inner, dict)
    roles = inner["roles"]
    assert isinstance(roles, list)
    if duplicate_scope == "within-role":
        roles[0]["members"][1]["logical_name"] = roles[0]["members"][0]["logical_name"]
    else:
        roles[1]["members"].append(copy.deepcopy(roles[0]["members"][0]))

    with pytest.raises(ValueError, match="duplicate logical member authority"):
        database_set_declaration_from_mapping(mapping)


def test_reversed_authored_roles_publish_the_same_canonical_manifest_identity(tmp_path: Path) -> None:
    _, mapping, _ = _source_fixture(tmp_path)
    reversed_mapping = copy.deepcopy(mapping)
    reversed_inner = reversed_mapping["database_set_declaration"]
    assert isinstance(reversed_inner, dict)
    reversed_roles = reversed_inner["roles"]
    assert isinstance(reversed_roles, list)
    reversed_roles.reverse()

    canonical_declaration = database_set_declaration_from_mapping(mapping)
    reversed_declaration = database_set_declaration_from_mapping(reversed_mapping)
    canonical_evidence = provision_database_set(
        canonical_declaration,
        manifest_root=tmp_path / "canonical-manifests",
        evidence_path=tmp_path / "canonical-evidence.json",
    )
    reversed_evidence = provision_database_set(
        reversed_declaration,
        manifest_root=tmp_path / "reversed-manifests",
        evidence_path=tmp_path / "reversed-evidence.json",
    )

    assert [role.role for role in reversed_declaration.roles] == ["primary", "metagenomic"]
    assert reversed_evidence.manifest_sha256 == canonical_evidence.manifest_sha256
    assert Path(reversed_evidence.manifest_path).read_bytes() == Path(canonical_evidence.manifest_path).read_bytes()


def test_manifest_and_provisioning_evidence_have_canonical_public_round_trips() -> None:
    source = Path("/database/source")
    manifest_path = Path("/manifests/afdb-search/2023-02/database-source-manifest.json")
    manifest_mapping = {
        "database_source_manifest": {
            "schema_version": 1,
            "database_set": {"identifier": "afdb-search", "version": "2023-02"},
            "verification": "metadata-verified",
            "source_root": str(source),
            "members": [
                {
                    "role": "primary",
                    "database_name": "uniref30_2302_db",
                    "logical_name": "uniref30_2302_db",
                    "source_path": "alias/uniref30_2302_db",
                    "source_kind": "symlink",
                    "resolved_path": "payload/uniref30_2302_db_pad",
                    "resolved_kind": "regular",
                    "size_bytes": 7,
                    "mtime_ns": 123456789,
                    "alias_topology": [{"path": "alias/uniref30_2302_db", "target": "../payload/uniref30_2302_db_pad"}],
                    "preexisting_checksum": None,
                },
                {
                    "role": "metagenomic",
                    "database_name": "colabfold_envdb_202108_db",
                    "logical_name": "colabfold_envdb_202108_db",
                    "source_path": "colabfold_envdb_202108_db",
                    "source_kind": "regular",
                    "resolved_path": "colabfold_envdb_202108_db",
                    "resolved_kind": "regular",
                    "size_bytes": 11,
                    "mtime_ns": 987654321,
                    "alias_topology": [],
                    "preexisting_checksum": {"algorithm": "sha256", "value": "b" * 64},
                },
            ],
        }
    }
    manifest = database_source_manifest_from_mapping(manifest_mapping)
    canonical = (json.dumps(manifest_mapping, indent=2, sort_keys=True) + "\n").encode()

    assert manifest.to_mapping() == manifest_mapping
    assert canonical_database_source_manifest_bytes(manifest) == canonical
    assert database_source_manifest_digest(manifest) == (
        "2aa204ef44a7c15be9d53e608d1ea841bf3dbea70ae4e279957f8cf6adb870ab"
    )

    evidence_mapping = {
        "database_set_provisioning": {
            "schema_version": 1,
            "database_set": {"identifier": "afdb-search", "version": "2023-02"},
            "disposition": "published",
            "verification": "metadata-verified",
            "manifest_path": str(manifest_path),
            "manifest_sha256": "2aa204ef44a7c15be9d53e608d1ea841bf3dbea70ae4e279957f8cf6adb870ab",
            "member_count": 2,
            "payload_bytes_copied": False,
            "payload_bytes_hashed": False,
        }
    }
    evidence = database_set_provisioning_evidence_from_mapping(evidence_mapping)
    assert evidence.to_mapping() == evidence_mapping


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown-field", "unknown"),
        ("duplicate-role", "primary and metagenomic"),
        ("missing-role-target", "missing its database_name target"),
        ("escaping-member", "confined relative path"),
        ("unsafe-checksum", "safe non-empty token"),
        ("incomplete-inventory", "complete non-empty member inventory"),
        ("unnormalized-root", "normalized absolute path"),
    ],
)
def test_database_set_declaration_fails_closed_on_invalid_authority(
    tmp_path: Path, mutation: str, message: str
) -> None:
    mapping = copy.deepcopy(_declaration_mapping(tmp_path / "source"))
    inner = mapping["database_set_declaration"]
    assert isinstance(inner, dict)
    roles = inner["roles"]
    assert isinstance(roles, list)
    primary = roles[0]
    assert isinstance(primary, dict)
    primary_members = primary["members"]
    assert isinstance(primary_members, list)

    if mutation == "unknown-field":
        inner["surprise"] = True
    elif mutation == "duplicate-role":
        roles[1]["role"] = "primary"
    elif mutation == "missing-role-target":
        primary["database_name"] = "missing_db"
    elif mutation == "escaping-member":
        primary_members[0]["source_path"] = "../escape"
    elif mutation == "unsafe-checksum":
        primary_members[1]["preexisting_checksum"]["value"] = "digest with spaces"
    elif mutation == "incomplete-inventory":
        primary["members"] = []
    else:
        inner["source_root"] = "/database/../escape"

    with pytest.raises(ValueError, match=message):
        database_set_declaration_from_mapping(mapping)


def _source_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, int]]:
    source = tmp_path / "source"
    payload = source / "payload"
    aliases = source / "aliases"
    payload.mkdir(parents=True)
    aliases.mkdir()
    primary = payload / "uniref30_2302_db_pad"
    primary_type = payload / "uniref30_2302_db_pad.dbtype"
    metagenomic = source / "colabfold_envdb_202108_db"
    primary.write_bytes(b"primary-payload")
    primary_type.write_bytes(b"dbtype-payload")
    metagenomic.write_bytes(b"metagenomic-payload")
    (aliases / "uniref30_2302_db").symlink_to("../payload/uniref30_2302_db_pad")
    (aliases / "uniref30_2302_db.dbtype").symlink_to("../payload/uniref30_2302_db_pad.dbtype")
    old_atime_ns = 1_600_000_000_000_000_000
    for path in (primary, primary_type, metagenomic):
        path.touch()
        path.chmod(0o000)
        path_stat = path.stat()
        path_atimes = (old_atime_ns, path_stat.st_mtime_ns)
        path.chmod(0o000)
        os.utime(path, ns=path_atimes)

    mapping = _declaration_mapping(source)
    inner = mapping["database_set_declaration"]
    assert isinstance(inner, dict)
    roles = inner["roles"]
    assert isinstance(roles, list)
    roles[0]["members"][0]["source_path"] = "aliases/uniref30_2302_db"
    roles[0]["members"][1]["source_path"] = "aliases/uniref30_2302_db.dbtype"
    return source, mapping, {str(path): old_atime_ns for path in (primary, primary_type, metagenomic)}


def test_provision_database_set_publishes_complete_metadata_without_reading_payloads(tmp_path: Path) -> None:
    source, mapping, original_atimes = _source_fixture(tmp_path)
    declaration = database_set_declaration_from_mapping(mapping)
    manifest_root = tmp_path / "manifests"
    evidence_path = tmp_path / "evidence" / "provisioning.json"

    evidence = provision_database_set(
        declaration,
        manifest_root=manifest_root,
        evidence_path=evidence_path,
    )

    manifest_path = manifest_root / "afdb-search" / "2023-02" / "database-source-manifest.json"
    manifest_mapping = json.loads(manifest_path.read_text())
    manifest = database_source_manifest_from_mapping(manifest_mapping)
    assert evidence.disposition == "published"
    assert evidence.manifest_path == str(manifest_path.resolve())
    assert evidence.manifest_sha256 == database_source_manifest_digest(manifest)
    assert evidence.member_count == 3
    assert evidence.payload_bytes_copied is False
    assert evidence.payload_bytes_hashed is False
    assert evidence_path.read_text() == json.dumps(evidence.to_mapping(), indent=2, sort_keys=True) + "\n"
    assert manifest_path.read_bytes() == canonical_database_source_manifest_bytes(manifest)
    assert {member.logical_name for member in manifest.members} == {
        "uniref30_2302_db",
        "uniref30_2302_db.dbtype",
        "colabfold_envdb_202108_db",
    }
    primary = next(member for member in manifest.members if member.logical_name == "uniref30_2302_db")
    assert primary.source_kind == "symlink"
    assert primary.resolved_path == "payload/uniref30_2302_db_pad"
    assert [alias.to_mapping() for alias in primary.alias_topology] == [
        {"path": "aliases/uniref30_2302_db", "target": "../payload/uniref30_2302_db_pad"}
    ]
    assert all(Path(path).stat().st_atime_ns == atime for path, atime in original_atimes.items())
    assert not any(path.name.endswith("payload") for path in manifest_root.rglob("*"))
    assert source.exists()


@pytest.mark.parametrize("failure", ["missing", "unsupported", "escaping-parent", "cyclic"])
def test_provision_database_set_rejects_invalid_source_topology(tmp_path: Path, failure: str) -> None:
    source, mapping, _ = _source_fixture(tmp_path)
    metagenomic = source / "colabfold_envdb_202108_db"
    if failure == "missing":
        metagenomic.unlink()
    elif failure == "unsupported":
        metagenomic.unlink()
        metagenomic.mkdir()
    elif failure == "escaping-parent":
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "uniref30_2302_db").write_bytes(b"escaped")
        (outside / "uniref30_2302_db.dbtype").write_bytes(b"escaped-type")
        aliases = source / "aliases"
        for child in aliases.iterdir():
            child.unlink()
        aliases.rmdir()
        aliases.symlink_to(outside, target_is_directory=True)
    else:
        first = source / "aliases" / "uniref30_2302_db"
        first.unlink()
        first.symlink_to("cycle")
        (first.parent / "cycle").symlink_to("uniref30_2302_db")

    declaration = database_set_declaration_from_mapping(mapping)
    with pytest.raises((ValueError, RuntimeError)):
        provision_database_set(
            declaration,
            manifest_root=tmp_path / "manifests",
            evidence_path=tmp_path / "evidence.json",
        )
    assert not (tmp_path / "manifests" / "afdb-search" / "2023-02" / "database-source-manifest.json").exists()


def test_provision_database_set_rejects_pre_post_metadata_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, mapping, _ = _source_fixture(tmp_path)
    declaration = database_set_declaration_from_mapping(mapping)
    target = source / "colabfold_envdb_202108_db"
    original_lstat = Path.lstat
    observations = 0

    def drifting_lstat(path: Path) -> os.stat_result:
        nonlocal observations
        if path == target:
            observations += 1
            if observations == 3:
                before = original_lstat(path)
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", drifting_lstat)
    with pytest.raises(RuntimeError, match="unstable Database Set source observation"):
        provision_database_set(
            declaration,
            manifest_root=tmp_path / "manifests",
            evidence_path=tmp_path / "evidence.json",
        )


def test_provision_database_set_rejects_pre_post_alias_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, mapping, _ = _source_fixture(tmp_path)
    declaration = database_set_declaration_from_mapping(mapping)
    alias = source / "aliases" / "uniref30_2302_db"
    original_readlink = os.readlink
    observations = 0

    def drifting_readlink(path: os.PathLike[str] | str) -> str:
        nonlocal observations
        result = original_readlink(path)
        if Path(path) == alias:
            observations += 1
            if observations == 2:
                return "../payload/uniref30_2302_db_pad.dbtype"
        return result

    monkeypatch.setattr(os, "readlink", drifting_readlink)
    with pytest.raises(RuntimeError, match="unstable Database Set source observation"):
        provision_database_set(
            declaration,
            manifest_root=tmp_path / "manifests",
            evidence_path=tmp_path / "evidence.json",
        )


def test_database_set_provisioning_cli_publishes_and_reports_stable_json(tmp_path: Path) -> None:
    _, mapping, _ = _source_fixture(tmp_path)
    declaration_path = tmp_path / "database-set.json"
    declaration_path.write_text(json.dumps(mapping))
    manifest_root = tmp_path / "manifests"
    evidence_path = tmp_path / "evidence" / "provisioning.json"

    result = CliRunner().invoke(
        cli,
        [
            "preprocessing",
            "database-set",
            "provision",
            "--declaration",
            str(declaration_path),
            "--manifest-root",
            str(manifest_root),
            "--write-evidence",
            str(evidence_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == json.loads(evidence_path.read_text())
    assert json.loads(result.output)["database_set_provisioning"]["disposition"] == "published"


def test_publication_uses_same_filesystem_atomic_rename_and_durable_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, mapping, _ = _source_fixture(tmp_path)
    declaration = database_set_declaration_from_mapping(mapping)
    events: list[tuple[str, Path | None, Path | None]] = []
    original_fchmod = os.fchmod
    original_fsync = os.fsync
    original_replace = os.replace

    def recording_fchmod(fd: int, mode: int) -> None:
        events.append(("fchmod", None, None))
        original_fchmod(fd, mode)

    def recording_fsync(fd: int) -> None:
        events.append(("fsync", None, None))
        original_fsync(fd)

    def recording_replace(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> None:
        source_path, destination_path = Path(source), Path(destination)
        events.append(("replace", source_path, destination_path))
        original_replace(source, destination)

    monkeypatch.setattr(os, "fchmod", recording_fchmod)
    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)

    evidence = provision_database_set(
        declaration,
        manifest_root=tmp_path / "manifests",
        evidence_path=tmp_path / "evidence" / "provisioning.json",
    )

    assert [event[0] for event in events[:4]] == ["fchmod", "fsync", "replace", "fsync"]
    replacements = [event for event in events if event[0] == "replace"]
    assert len(replacements) == 2
    assert all(source.parent == destination.parent for _, source, destination in replacements)
    assert not list(tmp_path.rglob("*.tmp"))
    assert (Path(evidence.manifest_path).stat().st_mode & 0o777) == 0o444
    assert ((tmp_path / "evidence" / "provisioning.json").stat().st_mode & 0o777) == 0o444
    assert (tmp_path / "manifests" / ".locks" / "afdb-search" / "2023-02.lock").is_file()


def test_existing_database_set_version_reuses_only_exact_immutable_document(tmp_path: Path) -> None:
    source, mapping, _ = _source_fixture(tmp_path)
    declaration = database_set_declaration_from_mapping(mapping)
    manifest_root = tmp_path / "manifests"
    first = provision_database_set(
        declaration,
        manifest_root=manifest_root,
        evidence_path=tmp_path / "first-evidence.json",
    )
    manifest_path = Path(first.manifest_path)
    original_bytes = manifest_path.read_bytes()

    second = provision_database_set(
        declaration,
        manifest_root=manifest_root,
        evidence_path=tmp_path / "second-evidence.json",
    )
    assert second.disposition == "reused"
    assert manifest_path.read_bytes() == original_bytes

    changed = source / "colabfold_envdb_202108_db"
    before = changed.stat()
    os.utime(changed, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
    with pytest.raises(RuntimeError, match="identity does not match"):
        provision_database_set(
            declaration,
            manifest_root=manifest_root,
            evidence_path=tmp_path / "mismatch-evidence.json",
        )
    assert manifest_path.read_bytes() == original_bytes
    assert not (tmp_path / "mismatch-evidence.json").exists()

    os.utime(changed, ns=(before.st_atime_ns, before.st_mtime_ns))
    manifest_path.chmod(0o644)
    with pytest.raises(RuntimeError, match="not immutable"):
        provision_database_set(
            declaration,
            manifest_root=manifest_root,
            evidence_path=tmp_path / "mutable-evidence.json",
        )


def test_atomic_publication_fault_leaves_no_visible_manifest_or_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, mapping, _ = _source_fixture(tmp_path)
    declaration = database_set_declaration_from_mapping(mapping)

    def fail_replace(_source: os.PathLike[str] | str, _destination: os.PathLike[str] | str) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected rename failure"):
        provision_database_set(
            declaration,
            manifest_root=tmp_path / "manifests",
            evidence_path=tmp_path / "evidence.json",
        )

    assert not (tmp_path / "manifests" / "afdb-search" / "2023-02" / "database-source-manifest.json").exists()
    assert not (tmp_path / "evidence.json").exists()
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("managed_target", ["manifest", "lock"])
def test_provision_database_set_rejects_evidence_collision_with_managed_paths_before_publication(
    tmp_path: Path,
    managed_target: str,
) -> None:
    _, mapping, _ = _source_fixture(tmp_path)
    declaration = database_set_declaration_from_mapping(mapping)
    manifest_root = tmp_path / "manifests"
    manifest_path = manifest_root / "afdb-search" / "2023-02" / "database-source-manifest.json"
    lock_path = manifest_root / ".locks" / "afdb-search" / "2023-02.lock"
    evidence_path = manifest_path if managed_target == "manifest" else lock_path

    with pytest.raises(DatabaseProvisioningError, match="evidence path collides with managed"):
        provision_database_set(
            declaration,
            manifest_root=manifest_root,
            evidence_path=evidence_path,
        )

    assert not manifest_path.exists()
    assert not lock_path.exists()
    assert not evidence_path.exists()
    assert not list(tmp_path.rglob("*.tmp"))
