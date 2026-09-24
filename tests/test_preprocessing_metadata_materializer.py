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

"""Safety and publication tests for the preprocessing metadata materializer."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import tarfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER = ROOT / "containers" / "scripts" / "preprocessing_metadata_materializer.py"
PREFLIGHT_SHA256 = "a" * 64


def _load_materializer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bspp_test_preprocessing_metadata_materializer", MATERIALIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


materializer = _load_materializer()
MAPPING = materializer.LOGICAL_MAPPING
TAXONOMY = materializer.LOGICAL_TAXONOMY


def _restore_tmp_permissions(path: Path) -> None:
    """Make test artifacts removable without traversing symlink targets."""
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(path_stat.st_mode):
        return
    if not stat.S_ISDIR(path_stat.st_mode):
        os.chmod(path, 0o600, follow_symlinks=False)
        return

    os.chmod(path, 0o700, follow_symlinks=False)
    with os.scandir(path) as entries:
        for entry in entries:
            _restore_tmp_permissions(Path(entry.path))


@pytest.fixture(autouse=True)
def _make_published_artifacts_removable(tmp_path: Path) -> Iterator[None]:
    yield
    _restore_tmp_permissions(tmp_path)


def _regular(name: str, payload: bytes, *, pax_headers: dict[str, str] | None = None) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.pax_headers = pax_headers or {}
    return info, payload


def _link(name: str, target: str, *, symbolic: bool = True) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE if symbolic else tarfile.LNKTYPE
    info.linkname = target
    return info, None


def _sparse(name: str) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.GNUTYPE_SPARSE
    info.size = 0
    return info, b""


def _write_archive(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> Path:
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for info, payload in members:
            archive.addfile(info, None if payload is None else io.BytesIO(payload))
    return path


def _valid_members(
    *,
    mapping_name: str = "uniref30_2302_mapping",
    taxonomy_name: str = "uniref30_2302_taxonomy",
    mapping: bytes = b"0\t10\n1\t11\n",
    taxonomy: bytes = b"10\tA\n11\tB\n",
) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    return [_regular(mapping_name, mapping), _regular(taxonomy_name, taxonomy)]


def _run(
    archive: Path,
    root: Path,
    *,
    expected_mapping: int | None = None,
    expected_taxonomy: int | None = None,
) -> dict[str, Any]:
    mapping_payload_size = 10 if expected_mapping is None else expected_mapping
    taxonomy_payload_size = 10 if expected_taxonomy is None else expected_taxonomy
    return cast(
        dict[str, Any],
        materializer.materialize_archive(
            archive,
            root,
            preflight_sha256=PREFLIGHT_SHA256,
            expected_archive_size_bytes=archive.stat().st_size,
            expected_output_sizes={MAPPING: mapping_payload_size, TAXONOMY: taxonomy_payload_size},
            local_test_context=True,
        ),
    )


def _manifest(result: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(result["manifest_path"]).read_text()))


def test_tmp_permission_cleanup_restores_artifacts_without_following_symlinks(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    immutable_file = artifact / "manifest.json"
    immutable_file.write_text("{}")
    immutable_file.chmod(0o400)
    artifact.chmod(0o500)

    external_file = tmp_path.parent / f"{tmp_path.name}-external"
    external_file.write_text("external")
    external_file.chmod(0o400)
    external_mode = stat.S_IMODE(external_file.lstat().st_mode)
    (tmp_path / "external-link").symlink_to(external_file)

    _restore_tmp_permissions(tmp_path)

    assert stat.S_IMODE(artifact.lstat().st_mode) == 0o700
    assert stat.S_IMODE(immutable_file.lstat().st_mode) == 0o600
    assert stat.S_IMODE(external_file.lstat().st_mode) == external_mode
    assert (tmp_path / "external-link").is_symlink()


def test_manifest_publication_rejects_zero_progress_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def zero_write(_descriptor: int, _payload: bytes | memoryview) -> int:
        nonlocal calls
        calls += 1
        return 0

    monkeypatch.setattr(materializer.os, "write", zero_write)

    with pytest.raises(OSError, match="write made no progress"):
        materializer._write_new_file(tmp_path / "manifest.json", b"{}", 0o400)

    assert calls == 1


@pytest.mark.parametrize(
    ("mapping_name", "taxonomy_name"),
    [
        ("uniref30_2302_mapping", "uniref30_2302_taxonomy"),
        (MAPPING, TAXONOMY),
        (f"metadata/{MAPPING}", f"metadata/{TAXONOMY}"),
    ],
)
def test_materializes_each_supported_candidate_spelling_to_fixed_names(
    tmp_path: Path,
    mapping_name: str,
    taxonomy_name: str,
) -> None:
    mapping = b"mapping-bytes"
    taxonomy = b"taxonomy-bytes"
    archive = _write_archive(
        tmp_path / "database.tar.gz",
        _valid_members(
            mapping_name=mapping_name,
            taxonomy_name=taxonomy_name,
            mapping=mapping,
            taxonomy=taxonomy,
        ),
    )

    result = _run(
        archive,
        tmp_path / "published",
        expected_mapping=len(mapping),
        expected_taxonomy=len(taxonomy),
    )
    manifest = _manifest(result)
    artifact = Path(result["artifact_root"])

    assert result["outcome"] == "success"
    assert (artifact / MAPPING).read_bytes() == mapping
    assert (artifact / TAXONOMY).read_bytes() == taxonomy
    assert manifest["outputs"][MAPPING]["source_member"] == mapping_name
    assert manifest["outputs"][TAXONOMY]["source_member"] == taxonomy_name
    assert manifest["scan"]["inventory_member_count"] == 2


def test_missing_candidate_publishes_complete_failure_inventory_and_preserves_partial_pair(tmp_path: Path) -> None:
    mapping = b"mapping"
    archive = _write_archive(
        tmp_path / "missing.tar.gz",
        [_regular("before", b"x"), _regular("uniref30_2302_mapping", mapping), _regular("after", b"y")],
    )

    result = _run(archive, tmp_path / "published", expected_mapping=len(mapping))
    manifest = _manifest(result)
    artifact = Path(result["artifact_root"])

    assert result["outcome"] == "failure"
    assert [entry["path"] for entry in manifest["inventory"]] == ["before", "uniref30_2302_mapping", "after"]
    assert manifest["scan"]["tar_iteration_complete"] is True
    assert {error["code"] for error in manifest["errors"]} == {"candidate_missing"}
    assert (artifact / MAPPING).read_bytes() == mapping
    assert not (artifact / TAXONOMY).exists()


def test_duplicate_candidates_fail_only_after_inventory_reaches_the_end(tmp_path: Path) -> None:
    members: list[tuple[tarfile.TarInfo, bytes | None]] = [
        _regular("uniref30_2302_mapping", b"legacy"),
        _regular(MAPPING, b"db-spelling"),
        _regular("uniref30_2302_taxonomy", b"taxonomy"),
        _regular("last-member", b"still inventoried"),
    ]
    archive = _write_archive(tmp_path / "duplicate.tar.gz", members)

    result = _run(archive, tmp_path / "published", expected_mapping=6, expected_taxonomy=8)
    manifest = _manifest(result)

    assert result["outcome"] == "failure"
    assert manifest["inventory"][-1]["path"] == "last-member"
    assert manifest["inventory"][1]["disposition"] == "duplicate_candidate_not_extracted"
    assert [error["code"] for error in manifest["errors"]] == ["candidate_count_mismatch"]


def test_local_candidate_write_failure_drains_member_and_inventories_later_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = b"mapping-payload"
    taxonomy = b"taxonomy-payload"
    archive = _write_archive(
        tmp_path / "write-failure.tar.gz",
        [
            _regular("uniref30_2302_mapping", mapping),
            _regular("between", b"still-seen"),
            _regular("uniref30_2302_taxonomy", taxonomy),
            _regular("after", b"also-seen"),
        ],
    )
    original_write: Callable[[int, bytes], None] = materializer._write_output_chunk
    calls = 0

    def fail_first_write(descriptor: int, chunk: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected output failure")
        original_write(descriptor, chunk)

    monkeypatch.setattr(materializer, "_write_output_chunk", fail_first_write)
    result = _run(
        archive,
        tmp_path / "published",
        expected_mapping=len(mapping),
        expected_taxonomy=len(taxonomy),
    )
    manifest = _manifest(result)
    artifact = Path(result["artifact_root"])

    assert result["outcome"] == "failure"
    assert manifest["scan"]["tar_iteration_complete"] is True
    assert [entry["path"] for entry in manifest["inventory"]] == [
        "uniref30_2302_mapping",
        "between",
        "uniref30_2302_taxonomy",
        "after",
    ]
    assert manifest["inventory"][0]["disposition"] == "partial_extraction"
    assert manifest["inventory"][2]["disposition"] == "extracted"
    assert {error["code"] for error in manifest["errors"]} >= {
        "local_output_error",
        "candidate_incomplete",
        "expected_size_mismatch",
    }
    assert manifest["outputs"][MAPPING]["status"] == "partial"
    assert (artifact / MAPPING).read_bytes() == b""
    assert (artifact / TAXONOMY).read_bytes() == taxonomy


@pytest.mark.parametrize(
    ("unsafe_member", "expected_reason"),
    [
        (_regular("../uniref30_2302_mapping", b"mapping"), "unsafe_path"),
        (_link("uniref30_2302_mapping", "somewhere"), "not_regular"),
        (_link("uniref30_2302_mapping", "somewhere", symbolic=False), "not_regular"),
        (_sparse("uniref30_2302_mapping"), "sparse_payload"),
    ],
)
def test_unsafe_link_traversal_and_sparse_candidates_are_never_extracted(
    tmp_path: Path,
    unsafe_member: tuple[tarfile.TarInfo, bytes | None],
    expected_reason: str,
) -> None:
    archive = _write_archive(
        tmp_path / "unsafe.tar.gz",
        [unsafe_member, _regular("uniref30_2302_taxonomy", b"taxonomy")],
    )

    result = _run(archive, tmp_path / "published", expected_taxonomy=8)
    manifest = _manifest(result)
    mapping_entry = next(entry for entry in manifest["inventory"] if entry["candidate_logical_name"] == MAPPING)

    assert result["outcome"] == "failure"
    assert expected_reason in mapping_entry["unsafe_reasons"]
    assert mapping_entry["disposition"] == "candidate_rejected"
    assert not (Path(result["artifact_root"]) / MAPPING).exists()
    assert "unsafe_candidate" in {error["code"] for error in manifest["errors"]}


def test_pax_path_override_is_recorded_and_rejected(tmp_path: Path) -> None:
    overridden = _regular(
        "placeholder",
        b"mapping",
        pax_headers={"path": "../uniref30_2302_mapping", "comment": "preserved"},
    )
    archive = _write_archive(
        tmp_path / "pax.tar.gz",
        [overridden, _regular("uniref30_2302_taxonomy", b"taxonomy")],
    )

    result = _run(archive, tmp_path / "published", expected_taxonomy=8)
    entry = _manifest(result)["inventory"][0]

    assert result["outcome"] == "failure"
    assert entry["pax_headers"] == {"comment": "preserved", "path": "../uniref30_2302_mapping"}
    assert entry["unsafe_reasons"] == ["unsafe_path", "pax_override:path"]


def test_benign_pax_metadata_is_inventoried_without_rejecting_a_regular_candidate(tmp_path: Path) -> None:
    mapping = _regular("uniref30_2302_mapping", b"mapping", pax_headers={"comment": "provenance"})
    archive = _write_archive(
        tmp_path / "benign-pax.tar.gz",
        [mapping, _regular("uniref30_2302_taxonomy", b"taxonomy")],
    )

    result = _run(archive, tmp_path / "published", expected_mapping=7, expected_taxonomy=8)
    entry = _manifest(result)["inventory"][0]

    assert result["outcome"] == "success"
    assert entry["pax_headers"] == {"comment": "provenance"}
    assert entry["unsafe_reasons"] == []


def test_size_differences_are_findings_not_early_scan_failures(tmp_path: Path) -> None:
    mapping = b"short"
    taxonomy = b"also-short"
    archive = _write_archive(
        tmp_path / "sizes.tar.gz",
        [*_valid_members(mapping=mapping, taxonomy=taxonomy), _regular("after", b"inventoried")],
    )

    result = _run(archive, tmp_path / "published", expected_mapping=999, expected_taxonomy=888)
    manifest = _manifest(result)

    assert result["outcome"] == "failure"
    assert manifest["inventory"][-1]["path"] == "after"
    assert manifest["scan"]["tar_iteration_complete"] is True
    assert [error["code"] for error in manifest["errors"]] == [
        "expected_size_mismatch",
        "expected_size_mismatch",
    ]
    assert manifest["size_findings"] == {
        MAPPING: {"expected_size_bytes": 999, "matches_expected": False, "observed_size_bytes": len(mapping)},
        TAXONOMY: {"expected_size_bytes": 888, "matches_expected": False, "observed_size_bytes": len(taxonomy)},
    }
    reused = _run(archive, tmp_path / "published", expected_mapping=999, expected_taxonomy=888)
    assert reused["outcome"] == "failure"
    assert reused["reused"] is True


def test_crc_corruption_and_non_gzip_trailing_data_publish_failures(tmp_path: Path) -> None:
    valid = _write_archive(tmp_path / "valid.tar.gz", _valid_members())
    valid_bytes = valid.read_bytes()
    corrupted_bytes = bytearray(valid_bytes)
    corrupted_bytes[-8] ^= 0x01
    corrupted = tmp_path / "corrupt.tar.gz"
    corrupted.write_bytes(corrupted_bytes)
    trailing = tmp_path / "trailing.tar.gz"
    trailing.write_bytes(valid_bytes + b"not-another-gzip-member" + b"x" * (2 * 1024 * 1024))
    truncated = tmp_path / "truncated.tar.gz"
    truncated.write_bytes(valid_bytes[:-4])

    corrupt_result = _run(corrupted, tmp_path / "corrupt-publication")
    trailing_result = _run(trailing, tmp_path / "trailing-publication")
    truncated_result = _run(truncated, tmp_path / "truncated-publication")
    corrupt_codes = {error["code"] for error in _manifest(corrupt_result)["errors"]}
    trailing_codes = {error["code"] for error in _manifest(trailing_result)["errors"]}
    truncated_codes = {error["code"] for error in _manifest(truncated_result)["errors"]}

    assert corrupt_result["outcome"] == "failure"
    assert trailing_result["outcome"] == "failure"
    assert truncated_result["outcome"] == "failure"
    assert "gzip_validation_error" in corrupt_codes or "archive_read_error" in corrupt_codes
    assert "gzip_validation_error" in trailing_codes or "archive_read_error" in trailing_codes
    assert "gzip_validation_error" in truncated_codes or "archive_read_error" in truncated_codes
    for path, result in ((corrupted, corrupt_result), (trailing, trailing_result), (truncated, truncated_result)):
        manifest = _manifest(result)
        expected_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest["archive_sha256"] == expected_digest
        assert manifest["scan"]["archive_sha256"] == expected_digest
        assert manifest["scan"]["archive_sha256_complete"] is True
        assert manifest["scan"]["compressed_bytes_consumed"] == path.stat().st_size
    assert _manifest(trailing_result)["scan"]["raw_drain_after_gzip_bytes"] > 0


def test_single_zero_tar_terminator_is_rejected_even_with_valid_gzip_eof(tmp_path: Path) -> None:
    uncompressed = io.BytesIO()
    with tarfile.open(fileobj=uncompressed, mode="w") as archive:
        for info, payload in _valid_members():
            assert payload is not None
            archive.addfile(info, io.BytesIO(payload))
    raw_tar = uncompressed.getvalue()
    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as archive:
        final = archive.getmembers()[-1]
    payload_end = final.offset_data + ((final.size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE
    archive_path = tmp_path / "one-terminator.tar.gz"
    archive_path.write_bytes(gzip.compress(raw_tar[:payload_end] + bytes(tarfile.BLOCKSIZE)))

    result = _run(archive_path, tmp_path / "published")
    manifest = _manifest(result)

    assert result["outcome"] == "failure"
    assert manifest["scan"]["gzip_eof_validated"] is True
    assert manifest["scan"]["exact_compressed_eof"] is True
    assert manifest["scan"]["tar_zero_termination_validated"] is False
    assert "tar_terminator_incomplete" in {error["code"] for error in manifest["errors"]}


def test_success_records_exact_eof_and_unchanged_archive_stat(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "database.tar.gz", _valid_members())

    result = _run(archive, tmp_path / "published")
    manifest = _manifest(result)

    assert (
        manifest["scan"]
        | {
            "compressed_bytes_consumed": archive.stat().st_size,
            "exact_compressed_eof": True,
            "gzip_eof_validated": True,
            "archive_descriptor_unchanged": True,
        }
        == manifest["scan"]
    )
    assert manifest["archive_stat_unchanged"] is True
    assert manifest["archive_before"] == manifest["archive_after"]
    assert manifest["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert manifest["scan"]["archive_sha256_complete"] is True


def test_manifest_records_deliberate_local_execution_context(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "database.tar.gz", _valid_members())

    result = _run(archive, tmp_path / "published")
    manifest = _manifest(result)
    context = manifest["execution_context"]

    assert context == {
        "context_kind": "local-test",
        "materializer_sha256": manifest["intent"]["materializer_sha256"],
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "slurm_job_id": None,
        "slurmd_nodename": None,
    }


def test_path_stat_drift_after_scan_converts_complete_payloads_to_immutable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_archive(tmp_path / "database.tar.gz", _valid_members())
    original_scan: Callable[..., object] = materializer._scan_archive

    def mutate_after_scan(*args: object, **kwargs: object) -> object:
        scanned = original_scan(*args, **kwargs)
        current = archive.stat()
        os.utime(archive, ns=(current.st_atime_ns, current.st_mtime_ns + 1))
        return scanned

    monkeypatch.setattr(materializer, "_scan_archive", mutate_after_scan)
    result = _run(archive, tmp_path / "published")
    manifest = _manifest(result)

    assert result["outcome"] == "failure"
    assert manifest["archive_stat_unchanged"] is False
    assert "archive_path_changed" in {error["code"] for error in manifest["errors"]}
    assert set(manifest["outputs"]) == {MAPPING, TAXONOMY}


def test_publication_and_pointer_are_atomic_immutable_and_reused_without_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_archive(tmp_path / "database.tar.gz", _valid_members())
    publication = tmp_path / "published"

    first = _run(archive, publication)
    artifact = Path(first["artifact_root"])
    pointer = Path(first["pointer_path"])
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o500
    assert stat.S_IMODE((artifact / "manifest.json").stat().st_mode) == 0o400
    assert stat.S_IMODE((artifact / MAPPING).stat().st_mode) == 0o400
    assert stat.S_IMODE((artifact / TAXONOMY).stat().st_mode) == 0o400
    assert stat.S_IMODE(pointer.parent.stat().st_mode) == 0o500
    assert stat.S_IMODE(pointer.stat().st_mode) == 0o400
    assert list((publication / ".staging").iterdir()) == []

    def forbidden_scan(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("reuse must not rescan the archive")

    monkeypatch.setattr(materializer, "_scan_archive", forbidden_scan)
    monkeypatch.setattr(
        materializer._HashingReader,
        "read",
        lambda *_args, **_kwargs: pytest.fail("reuse must not reread compressed archive bytes"),
    )
    second = _run(archive, publication)

    assert first["artifact_root"] == second["artifact_root"]
    assert second["reused"] is True


def test_slurm_job_identity_is_evidence_but_not_part_of_reusable_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_archive(tmp_path / "database.tar.gz", _valid_members())
    publication = tmp_path / "published"
    arguments = {
        "preflight_sha256": PREFLIGHT_SHA256,
        "expected_archive_size_bytes": archive.stat().st_size,
        "expected_output_sizes": {MAPPING: 10, TAXONOMY: 10},
    }
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURMD_NODENAME", "node-a")
    first = cast(dict[str, Any], materializer.materialize_archive(archive, publication, **arguments))
    first_manifest = _manifest(first)

    monkeypatch.setenv("SLURM_JOB_ID", "67890")
    monkeypatch.setenv("SLURMD_NODENAME", "node-b")
    monkeypatch.setattr(
        materializer,
        "_scan_archive",
        lambda *_args, **_kwargs: pytest.fail("a later Slurm job must reuse the completed intent"),
    )
    second = cast(dict[str, Any], materializer.materialize_archive(archive, publication, **arguments))

    assert second["reused"] is True
    assert second["intent_key"] == first["intent_key"]
    assert first_manifest["execution_context"]["slurm_job_id"] == "12345"
    assert first_manifest["execution_context"]["slurmd_nodename"] == "node-a"


def test_reuse_exactly_validates_non_job_execution_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_archive(tmp_path / "database.tar.gz", _valid_members())
    publication = tmp_path / "published"
    _run(archive, publication)
    original_builder: Callable[..., dict[str, Any]] = materializer._build_execution_context

    def changed_python_context(*args: object, **kwargs: object) -> dict[str, Any]:
        context = original_builder(*args, **kwargs)
        context["python_executable"] = "/different/pinned/python3.12"
        return context

    monkeypatch.setattr(materializer, "_build_execution_context", changed_python_context)
    monkeypatch.setattr(
        materializer,
        "_scan_archive",
        lambda *_args, **_kwargs: pytest.fail("execution-context drift must fail closed without rescanning"),
    )

    with pytest.raises(materializer.ReuseValidationError, match="python_executable"):
        _run(archive, publication)


def test_non_test_invocation_requires_slurm_job_and_node_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_archive(tmp_path / "database.tar.gz", _valid_members())
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)

    with pytest.raises(materializer.MaterializationError, match="SLURM_JOB_ID and SLURMD_NODENAME"):
        materializer.materialize_archive(
            archive,
            tmp_path / "published",
            preflight_sha256=PREFLIGHT_SHA256,
            expected_archive_size_bytes=archive.stat().st_size,
            expected_output_sizes={MAPPING: 10, TAXONOMY: 10},
        )


def test_reuse_fails_closed_when_output_or_manifest_modes_or_hashes_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_archive(tmp_path / "database.tar.gz", _valid_members())
    publication = tmp_path / "published"
    first = _run(archive, publication)
    output = Path(first["artifact_root"]) / MAPPING
    output.chmod(0o600)
    output.write_bytes(b"tampered")
    output.chmod(0o400)

    monkeypatch.setattr(
        materializer,
        "_scan_archive",
        lambda *_args, **_kwargs: pytest.fail("invalid reuse must fail closed rather than rescan"),
    )
    with pytest.raises(materializer.ReuseValidationError, match="hash or size mismatch"):
        _run(archive, publication)


def test_intent_lock_allows_only_one_scan_for_concurrent_callers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_archive(tmp_path / "database.tar.gz", _valid_members())
    publication = tmp_path / "published"
    original_scan: Callable[..., object] = materializer._scan_archive
    entered = threading.Event()
    release = threading.Event()
    scan_count = 0
    count_lock = threading.Lock()

    def slow_scan(*args: object, **kwargs: object) -> object:
        nonlocal scan_count
        with count_lock:
            scan_count += 1
        entered.set()
        assert release.wait(timeout=5)
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(materializer, "_scan_archive", slow_scan)
    results: list[dict[str, Any]] = []
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            results.append(_run(archive, publication))
        except BaseException as exc:  # pragma: no cover - diagnostic collection for the assertion below
            failures.append(exc)

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    time.sleep(0.1)
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert failures == []
    assert scan_count == 1
    assert sorted(result["reused"] for result in results) == [False, True]


def test_failure_pointer_reuses_immutable_partial_recovery_without_rescan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_archive(tmp_path / "missing.tar.gz", [_regular("uniref30_2302_mapping", b"mapping")])
    publication = tmp_path / "published"
    first = _run(archive, publication, expected_mapping=7)
    partial = Path(first["artifact_root"]) / MAPPING

    assert first["outcome"] == "failure"
    assert partial.read_bytes() == b"mapping"
    assert stat.S_IMODE(partial.stat().st_mode) == 0o400

    monkeypatch.setattr(
        materializer,
        "_scan_archive",
        lambda *_args, **_kwargs: pytest.fail("completed failures must be reused without rescanning"),
    )
    second = _run(archive, publication, expected_mapping=7)

    assert second["outcome"] == "failure"
    assert second["reused"] is True
    assert second["artifact_root"] == first["artifact_root"]


def test_archive_symlink_and_wrong_bound_size_are_rejected_before_publication(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "database.tar.gz", _valid_members())
    symlink = tmp_path / "database-link.tar.gz"
    symlink.symlink_to(archive.name)

    with pytest.raises(materializer.ArchiveValidationError, match="must not be a symlink"):
        materializer.materialize_archive(
            symlink,
            tmp_path / "symlink-publication",
            preflight_sha256=PREFLIGHT_SHA256,
            expected_archive_size_bytes=archive.stat().st_size,
            expected_output_sizes={MAPPING: 10, TAXONOMY: 10},
        )
    with pytest.raises(materializer.ArchiveValidationError, match="archive size"):
        materializer.materialize_archive(
            archive,
            tmp_path / "size-publication",
            preflight_sha256=PREFLIGHT_SHA256,
            expected_archive_size_bytes=archive.stat().st_size + 1,
            expected_output_sizes={MAPPING: 10, TAXONOMY: 10},
        )

    assert not (tmp_path / "symlink-publication").exists()
    assert not (tmp_path / "size-publication").exists()
