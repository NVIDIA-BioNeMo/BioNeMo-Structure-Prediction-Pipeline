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

"""Tests for ``DatabaseSetDeclaration`` s3 source and ``download_database_set_from_s3``.

Covers:
* ``DatabaseSetDeclaration`` s3 round-trip + validation.
* ``download_database_set_from_s3`` (fake transfer): staging + SHA-256
  (missing/unsupported/non-64-hex fail closed), URI construction
  ``{source_uri.rstrip("/")}/{source_path}``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseSetDeclaration,
    DatabaseSetIdentity,
    DeclaredDatabaseMember,
    DeclaredDatabaseRole,
    PreexistingChecksum,
    database_set_declaration_from_mapping,
)
from bspp.orchestration.runtime.data_movement.common import PlannedTransfer, TransferResult
from bspp.orchestration.runtime.preprocessing.database_intake import (
    DatabaseIntakeError,
    download_database_set_from_s3,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SOURCE_URI = "s3://example-bucket-bucket/databases/bspp-search-2026-08"


def _ok_transfer(src: str, dst: str) -> TransferResult:
    return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=0, elapsed_s=0.0)


def _fail_transfer(src: str, dst: str) -> TransferResult:
    return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=1, elapsed_s=0.0, stderr_tail="boom")


def _planned_transfer(src: str, dst: str) -> PlannedTransfer:
    return PlannedTransfer(tool="s5cmd", argv=("s5cmd", "cp", src, dst), note="dry-run")


_UNSET = object()


def _make_declaration(
    *,
    source_kind: str = "s3",
    source_uri: str | None = _SOURCE_URI,
    primary_sha: str = _SHA_A,
    meta_sha: str = _SHA_B,
    primary_checksum: object = _UNSET,
    meta_checksum: object = _UNSET,
    source_root: str = "/databases/bspp-search",
) -> DatabaseSetDeclaration:
    if primary_checksum is _UNSET:
        primary_checksum = PreexistingChecksum(algorithm="sha256", value=primary_sha)
    if meta_checksum is _UNSET:
        meta_checksum = PreexistingChecksum(algorithm="sha256", value=meta_sha)
    return DatabaseSetDeclaration(
        database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"),
        source_root=source_root,
        roles=(
            DeclaredDatabaseRole(
                role="primary",
                database_name="afdb",
                members=(
                    DeclaredDatabaseMember(
                        logical_name="afdb",
                        source_path="afdb/afdb.db",
                        preexisting_checksum=primary_checksum,
                    ),
                ),
            ),
            DeclaredDatabaseRole(
                role="metagenomic",
                database_name="metagenomic",
                members=(
                    DeclaredDatabaseMember(
                        logical_name="metagenomic",
                        source_path="meta/meta.db",
                        preexisting_checksum=meta_checksum,
                    ),
                ),
            ),
        ),
        source_kind=source_kind,  # type: ignore[arg-type]
        source_uri=source_uri,
    )


# ---------------------------------------------------------------------------
# DatabaseSetDeclaration s3 round-trip + validation
# ---------------------------------------------------------------------------


def test_s3_declaration_round_trip() -> None:
    """An s3 DatabaseSetDeclaration serializes, deserializes, and round-trips."""
    decl = _make_declaration()
    mapping = decl.to_mapping()
    inner = mapping["database_set_declaration"]
    assert inner["source_kind"] == "s3"
    assert inner["source_uri"] == _SOURCE_URI
    reloaded = database_set_declaration_from_mapping(mapping)
    assert reloaded == decl
    assert reloaded.source_kind == "s3"
    assert reloaded.source_uri == _SOURCE_URI


def test_local_declaration_omits_source_fields() -> None:
    """A local DatabaseSetDeclaration omits source_kind and source_uri (digest preservation)."""
    decl = _make_declaration(source_kind="local", source_uri=None)
    mapping = decl.to_mapping()
    inner = mapping["database_set_declaration"]
    assert "source_kind" not in inner
    assert "source_uri" not in inner
    reloaded = database_set_declaration_from_mapping(mapping)
    assert reloaded == decl
    assert reloaded.source_kind == "local"
    assert reloaded.source_uri is None


def test_s3_requires_source_uri() -> None:
    """s3 source_kind without a source_uri raises."""
    with pytest.raises(ValueError, match="s3 source_kind requires a non-empty s3:// source_uri"):
        _make_declaration(source_uri=None)


def test_local_forbids_source_uri() -> None:
    """local source_kind with a source_uri raises."""
    with pytest.raises(ValueError, match="local source_kind requires source_uri to be None"):
        _make_declaration(source_kind="local", source_uri=_SOURCE_URI)


def test_s3_rejects_non_s3_uri() -> None:
    """s3 source_kind with a non-s3:// URI raises."""
    with pytest.raises(ValueError, match="s3 source_kind requires a non-empty s3:// source_uri"):
        _make_declaration(source_uri="https://example.com/db")


# ---------------------------------------------------------------------------
# download_database_set_from_s3
# ---------------------------------------------------------------------------


def test_download_database_set_no_op_for_local(tmp_path: Path) -> None:
    """download_database_set_from_s3 is a no-op for source_kind='local'."""
    decl = _make_declaration(source_kind="local", source_uri=None)
    calls: list[tuple[str, str]] = []

    def fake_transfer(src: str, dst: str) -> TransferResult:
        calls.append((src, dst))
        return _ok_transfer(src, dst)

    download_database_set_from_s3(decl, transfer=fake_transfer)
    assert calls == []


def test_download_database_set_s3_success(tmp_path: Path) -> None:
    """S3 download stages members and verifies SHA-256."""
    primary_bytes = b"primary-db-content"
    meta_bytes = b"meta-db-content"
    primary_sha = hashlib.sha256(primary_bytes).hexdigest()
    meta_sha = hashlib.sha256(meta_bytes).hexdigest()
    decl = _make_declaration(primary_sha=primary_sha, meta_sha=meta_sha, source_root=str(tmp_path / "dbs"))
    calls: list[tuple[str, str]] = []

    def fake_transfer(src: str, dst: str) -> TransferResult:
        calls.append((src, dst))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        if "afdb" in src:
            Path(dst).write_bytes(primary_bytes)
        else:
            Path(dst).write_bytes(meta_bytes)
        return _ok_transfer(src, dst)

    download_database_set_from_s3(decl, transfer=fake_transfer)
    assert len(calls) == 2
    # URI construction: {source_uri.rstrip("/")}/{source_path}
    assert calls[0][0] == f"{_SOURCE_URI}/afdb/afdb.db"
    assert calls[1][0] == f"{_SOURCE_URI}/meta/meta.db"
    assert (tmp_path / "dbs" / "afdb" / "afdb.db").read_bytes() == primary_bytes
    assert (tmp_path / "dbs" / "meta" / "meta.db").read_bytes() == meta_bytes


def test_download_database_set_missing_checksum_fails_closed(tmp_path: Path) -> None:
    """A member without a preexisting_checksum fails closed."""
    decl = _make_declaration(
        primary_checksum=None,
        source_root=str(tmp_path / "dbs"),
    )
    with pytest.raises(DatabaseIntakeError, match="no preexisting_checksum"):
        download_database_set_from_s3(decl, transfer=_ok_transfer)


def test_download_database_set_unsupported_algorithm_fails_closed(tmp_path: Path) -> None:
    """A checksum with an unsupported algorithm fails closed."""
    decl = _make_declaration(
        primary_checksum=PreexistingChecksum(algorithm="md5", value="abc123"),
    )
    with pytest.raises(DatabaseIntakeError, match="checksum algorithm must be 'sha256'"):
        download_database_set_from_s3(decl, transfer=_ok_transfer)


def test_download_database_set_non_64_hex_checksum_fails_closed(tmp_path: Path) -> None:
    """A checksum with a non-64-hex value fails closed."""
    decl = _make_declaration(
        primary_checksum=PreexistingChecksum(algorithm="sha256", value="short"),
    )
    with pytest.raises(DatabaseIntakeError, match="checksum value must be 64-hex sha256"):
        download_database_set_from_s3(decl, transfer=_ok_transfer)


def test_download_database_set_sha256_drift_fails_closed(tmp_path: Path) -> None:
    """SHA-256 drift after download fails closed."""
    primary_bytes = b"primary-db-content"
    primary_sha = hashlib.sha256(primary_bytes).hexdigest()
    meta_bytes = b"meta-db-content"
    meta_sha = hashlib.sha256(meta_bytes).hexdigest()
    decl = _make_declaration(primary_sha=primary_sha, meta_sha=meta_sha, source_root=str(tmp_path / "dbs"))

    def fake_transfer(src: str, dst: str) -> TransferResult:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        # Write wrong bytes
        Path(dst).write_bytes(b"wrong-content")
        return _ok_transfer(src, dst)

    with pytest.raises(DatabaseIntakeError, match=r"sha256.*does not match"):
        download_database_set_from_s3(decl, transfer=fake_transfer)


def test_download_database_set_rejects_planned_transfer(tmp_path: Path) -> None:
    """A PlannedTransfer (dry-run) is rejected."""
    decl = _make_declaration(source_root=str(tmp_path / "dbs"))
    with pytest.raises(DatabaseIntakeError, match="executed transfer, not a dry-run plan"):
        download_database_set_from_s3(decl, transfer=_planned_transfer)


def test_download_database_set_non_ok_fail_closed(tmp_path: Path) -> None:
    """A non-OK TransferResult fails closed."""
    decl = _make_declaration(source_root=str(tmp_path / "dbs"))
    with pytest.raises(DatabaseIntakeError, match="download failed with returncode 1"):
        download_database_set_from_s3(decl, transfer=_fail_transfer)
