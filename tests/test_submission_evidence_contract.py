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

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from bspp.orchestration.contract.submission_evidence import (
    EvidenceIndex,
    EvidenceIndexEntry,
    SubmissionExpectation,
    SubmissionResult,
    SubmissionToken,
    build_evidence_index,
    evidence_index_from_bytes,
    evidence_index_from_mapping,
    submission_expectation_from_bytes,
    submission_expectation_from_mapping,
    submission_result_from_bytes,
    submission_result_from_mapping,
    submission_token_from_mapping,
    validate_submission_result,
    verify_evidence_index,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _token() -> SubmissionToken:
    return SubmissionToken.create(
        run_id="run-1",
        runspec_sha256=SHA_A,
        step_index=3,
        step_name="slurm",
        slice_id="0-9",
        attempt=1,
        script_sha256=SHA_B,
        bootstrap_sha256=SHA_D,
        control_state_sha256=SHA_C,
        runtime_qualification_sha256="e" * 64,
    )


def test_submission_token_is_deterministic_strict_and_content_bound() -> None:
    token = _token()
    assert token.token == hashlib.sha256(token.canonical_identity_bytes()).hexdigest()
    assert submission_token_from_mapping(token.to_mapping()) == token
    changed = dict(token.to_mapping(), attempt=2)
    with pytest.raises(ValueError, match="token does not match canonical identity"):
        submission_token_from_mapping(changed)
    with pytest.raises(ValueError, match=r"extra=\['surprise'\]"):
        submission_token_from_mapping(dict(token.to_mapping(), surprise=True))


def test_expectation_and_result_v1_bind_all_governed_inputs_and_observations() -> None:
    expectation = SubmissionExpectation(
        token=_token(),
        rendered_script_sha256=SHA_B,
        control_state_sha256=SHA_C,
        bootstrap_sha256=SHA_D,
        runtime_qualification_sha256="e" * 64,
        runtime_ipsae_source_revision="f" * 40,
        runtime_ipsae_binary_sha256="f" * 64,
    )
    result = SubmissionResult(
        token=_token(),
        rendered_script_sha256=SHA_B,
        control_state_sha256=SHA_C,
        bootstrap_sha256=SHA_D,
        runtime_qualification_sha256="e" * 64,
        runtime_ipsae_source_revision="f" * 40,
        runtime_ipsae_binary_sha256="f" * 64,
        job_id="12345",
        scheduler_status="PENDING",
        runtime_observations=(("cluster", "eos"), ("transport", "ssh")),
    )
    assert submission_expectation_from_mapping(expectation.to_mapping()) == expectation
    assert submission_result_from_mapping(result.to_mapping()) == result
    with pytest.raises(ValueError, match="runtime_observations must be canonical sorted unique"):
        submission_result_from_mapping(
            dict(result.to_mapping(), runtime_observations=[{"name": "z", "value": "1"}, {"name": "a", "value": "2"}])
        )


def test_expectation_serializes_its_own_format_version() -> None:
    expectation = object.__new__(SubmissionExpectation)
    object.__setattr__(expectation, "token", _token())
    object.__setattr__(expectation, "rendered_script_sha256", SHA_B)
    object.__setattr__(expectation, "control_state_sha256", SHA_C)
    object.__setattr__(expectation, "bootstrap_sha256", SHA_D)
    object.__setattr__(expectation, "runtime_qualification_sha256", "e" * 64)
    object.__setattr__(expectation, "runtime_ipsae_source_revision", "f" * 40)
    object.__setattr__(expectation, "runtime_ipsae_binary_sha256", "f" * 64)
    object.__setattr__(expectation, "format_version", 2)

    assert expectation.to_mapping()["format_version"] == 2


def test_result_rejects_duplicate_runtime_observation_names() -> None:
    with pytest.raises(ValueError, match="runtime observation names must be unique"):
        SubmissionResult(
            _token(),
            SHA_B,
            SHA_C,
            SHA_D,
            "e" * 64,
            "f" * 40,
            "f" * 64,
            job_id="12345",
            scheduler_status="COMPLETED",
            runtime_observations=(("image", "sha256:a"), ("image", "sha256:b")),
        )


def test_evidence_index_is_sorted_relative_and_reconciles_exact_tree(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "z.json").write_text("z")
    (tmp_path / "nested" / "a.json").write_text("a")
    index = build_evidence_index(tmp_path)
    assert [entry.path for entry in index.entries] == ["nested/a.json", "z.json"]
    assert verify_evidence_index(tmp_path, index).ok

    (tmp_path / "extra.json").write_text("extra")
    validation = verify_evidence_index(tmp_path, index)
    assert validation.issues == ("unexpected evidence path: extra.json",)


@pytest.mark.parametrize("mutation", ["missing", "replaced", "duplicate"])
def test_evidence_reconciliation_rejects_missing_replaced_and_duplicate(tmp_path: Path, mutation: str) -> None:
    evidence = tmp_path / "result.json"
    evidence.write_text("original")
    index = build_evidence_index(tmp_path)
    if mutation == "missing":
        evidence.unlink()
        assert verify_evidence_index(tmp_path, index).issues == ("missing evidence path: result.json",)
    elif mutation == "replaced":
        evidence.write_text("replacement")
        assert verify_evidence_index(tmp_path, index).issues == ("evidence digest mismatch: result.json",)
    else:
        with pytest.raises(ValueError, match="canonical sorted unique"):
            EvidenceIndex(format_version=1, entries=(index.entries[0], index.entries[0]))


def test_index_parser_rejects_noncanonical_paths_and_order() -> None:
    canonical = EvidenceIndex(format_version=1, entries=(EvidenceIndexEntry("a", SHA_A),))
    assert evidence_index_from_mapping(canonical.to_mapping()) == canonical
    with pytest.raises(ValueError, match=r"extra=\['extra'\]"):
        evidence_index_from_mapping(dict(canonical.to_mapping(), extra=True))
    entries = (
        EvidenceIndexEntry("z", SHA_A),
        EvidenceIndexEntry("a", SHA_B),
    )
    with pytest.raises(ValueError, match="canonical sorted unique"):
        EvidenceIndex(format_version=1, entries=entries)
    with pytest.raises(ValueError, match="relative canonical POSIX"):
        EvidenceIndexEntry("../escape", SHA_A)
    with pytest.raises(ValueError, match="relative canonical POSIX"):
        EvidenceIndexEntry("/absolute", SHA_A)
    with pytest.raises(ValueError, match="relative canonical POSIX"):
        EvidenceIndexEntry(".", SHA_A)


def test_wire_records_require_bounded_canonical_json_bytes() -> None:
    expectation = SubmissionExpectation(_token(), SHA_B, SHA_C, SHA_D, "e" * 64, "f" * 40, "f" * 64)
    result = SubmissionResult(
        _token(), SHA_B, SHA_C, SHA_D, "e" * 64, "f" * 40, "f" * 64, job_id="1", scheduler_status="PENDING"
    )
    expectation_bytes = expectation.canonical_bytes()
    result_bytes = result.canonical_bytes()
    assert submission_expectation_from_bytes(expectation_bytes) == expectation
    assert submission_result_from_bytes(result_bytes) == result
    with pytest.raises(ValueError, match="canonical compact JSON"):
        submission_expectation_from_bytes(b" " + expectation_bytes)
    with pytest.raises(ValueError, match="canonical compact JSON"):
        submission_result_from_bytes(result_bytes.replace(b'"job_id":"1"', b'"job_id":"2"') + b" ")
    with pytest.raises(ValueError, match="exceeds size bound"):
        submission_result_from_bytes(b"{" + b" " * (1024 * 1024) + b"}")


@pytest.mark.parametrize(
    "field",
    [
        "token",
        "rendered_script_sha256",
        "control_state_sha256",
        "bootstrap_sha256",
        "runtime_qualification_sha256",
        "runtime_ipsae_source_revision",
        "runtime_ipsae_binary_sha256",
    ],
)
def test_result_must_match_immutable_expectation(field: str) -> None:
    expectation = SubmissionExpectation(_token(), SHA_B, SHA_C, SHA_D, "e" * 64, "f" * 40, "f" * 64)
    values: dict[str, object] = {
        "token": _token(),
        "rendered_script_sha256": SHA_B,
        "control_state_sha256": SHA_C,
        "bootstrap_sha256": SHA_D,
        "runtime_qualification_sha256": "e" * 64,
        "runtime_ipsae_source_revision": "f" * 40,
        "runtime_ipsae_binary_sha256": "f" * 64,
    }
    if field == "token":
        values[field] = SubmissionToken.create(
            run_id="run-1",
            runspec_sha256=SHA_A,
            step_index=3,
            step_name="slurm",
            slice_id="0-9",
            attempt=2,
            script_sha256=SHA_B,
            bootstrap_sha256=SHA_D,
            control_state_sha256=SHA_C,
            runtime_qualification_sha256="e" * 64,
        )
    elif field == "rendered_script_sha256":
        values["token"] = SubmissionToken.create(
            run_id="run-1",
            runspec_sha256=SHA_A,
            step_index=3,
            step_name="slurm",
            slice_id="0-9",
            attempt=1,
            script_sha256="e" * 64,
            bootstrap_sha256=SHA_D,
            control_state_sha256=SHA_C,
            runtime_qualification_sha256="e" * 64,
        )
        values[field] = "e" * 64
    else:
        changed_digest = "f" * 64 if field == "runtime_qualification_sha256" else "e" * 64
        values[field] = changed_digest
        values["token"] = SubmissionToken.create(
            run_id="run-1",
            runspec_sha256=SHA_A,
            step_index=3,
            step_name="slurm",
            slice_id="0-9",
            attempt=1,
            script_sha256=SHA_B,
            bootstrap_sha256=changed_digest if field == "bootstrap_sha256" else SHA_D,
            control_state_sha256=changed_digest if field == "control_state_sha256" else SHA_C,
            runtime_qualification_sha256=(
                changed_digest if field == "runtime_qualification_sha256" else _token().runtime_qualification_sha256
            ),
        )
    result = SubmissionResult(**values, job_id="1", scheduler_status="PENDING")
    with pytest.raises(ValueError, match=f"SubmissionResult {field} does not match expectation"):
        validate_submission_result(expectation, result)


def test_index_wire_parser_rejects_oversized_and_too_many_entries() -> None:
    with pytest.raises(ValueError, match="EvidenceIndex exceeds size bound"):
        evidence_index_from_bytes(b"{" + b" " * (1024 * 1024) + b"}")
    payload = {"format_version": 1, "entries": [{"path": f"{i:06d}", "sha256": SHA_A} for i in range(1001)]}
    with pytest.raises(ValueError, match="entry count exceeds bound"):
        evidence_index_from_mapping(payload)
    with pytest.raises(ValueError, match="entry count exceeds bound"):
        EvidenceIndex(1, tuple(EvidenceIndexEntry(f"{i:06d}", SHA_A) for i in range(1001)))


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_index_entry_parser_rejects_missing_and_extra_fields(mutation: str) -> None:
    entry: dict[str, object] = {"path": "a", "sha256": SHA_A}
    if mutation == "missing":
        del entry["sha256"]
    else:
        entry["size"] = 1
    with pytest.raises(ValueError, match=f"{mutation}="):
        evidence_index_from_mapping({"format_version": 1, "entries": [entry]})


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_evidence_index_rejects_nonexclusive_or_nonregular_files(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    source.write_text("x")
    destination = tmp_path / "destination"
    if kind == "symlink":
        destination.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, destination)
    else:
        os.mkfifo(destination)
    with pytest.raises(ValueError, match=r"regular files|hard-linked"):
        build_evidence_index(tmp_path)


def test_evidence_index_rejects_symlinked_root(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "result.json").write_text("result")
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ValueError, match="evidence root must be a real directory"):
        build_evidence_index(alias)


def test_evidence_index_rejects_file_replaced_during_stable_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "result.json"
    evidence.write_text("original")
    replacement = tmp_path / "replacement"
    replacement.write_text("replacement")
    real_open = os.open
    replaced = False

    def replacing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == "result.json" and dir_fd is not None:
            replaced = True
            os.replace(replacement, evidence)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replacing_open)
    with pytest.raises(ValueError, match="changed during open"):
        build_evidence_index(tmp_path)


def test_evidence_index_rejects_hardlink_created_during_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = tmp_path / "result.json"
    evidence.write_text("result")
    alias = tmp_path.parent / f"{tmp_path.name}-alias"
    real_open = os.open
    linked = False

    def linking_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal linked
        if not linked and path == "result.json" and dir_fd is not None:
            linked = True
            os.link(evidence, alias)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", linking_open)
    with pytest.raises(ValueError, match="changed during open"):
        build_evidence_index(tmp_path)


def test_evidence_index_rejects_intermediate_directory_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "result.json").write_text("result")
    moved = tmp_path / "moved"
    real_stat = os.stat
    nested_stats = 0

    def replacing_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal nested_stats
        if path == "nested" and kwargs.get("dir_fd") is not None:
            nested_stats += 1
            if nested_stats == 2:
                os.rename(nested, moved)
                nested.symlink_to(moved, target_is_directory=True)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", replacing_stat)
    with pytest.raises(ValueError, match=r"(directory|root) changed during traversal"):
        build_evidence_index(tmp_path)


def test_evidence_index_rejects_root_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "result.json").write_text("result")
    moved = tmp_path.parent / f"{tmp_path.name}-moved"
    original_lstat = Path.lstat
    root_stats = 0

    def replacing_lstat(path: Path) -> os.stat_result:
        nonlocal root_stats
        result = original_lstat(path)
        if path == tmp_path:
            root_stats += 1
            if root_stats == 2:
                os.rename(tmp_path, moved)
                tmp_path.mkdir()
                (tmp_path / "result.json").write_text("alternate")
                return original_lstat(path)
        return result

    monkeypatch.setattr(Path, "lstat", replacing_lstat)
    with pytest.raises(ValueError, match=r"root changed during (open|traversal)"):
        build_evidence_index(tmp_path)
