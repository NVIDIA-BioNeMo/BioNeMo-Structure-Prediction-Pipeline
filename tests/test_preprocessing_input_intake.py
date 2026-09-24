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

"""Tests for ``download_remote_fasta`` (preprocessing remote FASTA intake).

Uses an injected fake transfer to exercise the full verification path without
network I/O.  Covers: success, drift fail-closed, ``PlannedTransfer`` rejection,
non-OK fail-closed, non-symlink check, zero-size rejection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bspp.orchestration.contract.phase import VerifiedRemoteInputLocation
from bspp.orchestration.runtime.data_movement.common import PlannedTransfer, TransferResult
from bspp.orchestration.runtime.preprocessing.input_intake import (
    InputIntakeError,
    download_remote_fasta,
    normalize_fasta_to_identity_headers,
)

_SHA = "a" * 64
_URI = "s3://example-bucket-bucket/inputs/remote.fa"
_CONTENT = b">seq description\nAAAA:TT\n"


def _ok_transfer(src: str, dst: str) -> TransferResult:
    return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=0, elapsed_s=0.0)


def _fail_transfer(src: str, dst: str) -> TransferResult:
    return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=1, elapsed_s=0.0, stderr_tail="boom")


def _planned_transfer(src: str, dst: str) -> PlannedTransfer:
    return PlannedTransfer(tool="s5cmd", argv=("s5cmd", "cp", src, dst), note="dry-run")


def _make_location(size: int = len(_CONTENT), sha: str | None = None) -> VerifiedRemoteInputLocation:
    actual_sha = sha or hashlib.sha256(_CONTENT).hexdigest()
    return VerifiedRemoteInputLocation(
        source_uri=_URI,
        sha256=actual_sha,
        size_bytes=size,
        path="inputs/remote.fa",
    )


def test_download_remote_fasta_success(tmp_path: Path) -> None:
    """A matching download returns the verified path."""
    location = _make_location()
    dest = tmp_path / "remote.fa"

    def fake_transfer(src: str, dst: str) -> TransferResult:
        Path(dst).write_bytes(_CONTENT)
        return _ok_transfer(src, dst)

    result = download_remote_fasta(location, dest, transfer=fake_transfer)
    assert result == dest
    assert dest.read_bytes() == _CONTENT


def test_download_remote_fasta_drift_fail_closed(tmp_path: Path) -> None:
    """Size/SHA-256 drift fails closed."""
    location = _make_location(sha="b" * 64)
    dest = tmp_path / "remote.fa"

    def fake_transfer(src: str, dst: str) -> TransferResult:
        Path(dst).write_bytes(_CONTENT)
        return _ok_transfer(src, dst)

    with pytest.raises(InputIntakeError, match=r"sha256.*does not match"):
        download_remote_fasta(location, dest, transfer=fake_transfer)


def test_download_remote_fasta_size_drift_fail_closed(tmp_path: Path) -> None:
    """Size drift fails closed."""
    location = _make_location(size=len(_CONTENT) + 10)
    dest = tmp_path / "remote.fa"

    def fake_transfer(src: str, dst: str) -> TransferResult:
        Path(dst).write_bytes(_CONTENT)
        return _ok_transfer(src, dst)

    with pytest.raises(InputIntakeError, match=r"size.*does not match"):
        download_remote_fasta(location, dest, transfer=fake_transfer)


def test_download_remote_fasta_rejects_planned_transfer(tmp_path: Path) -> None:
    """A PlannedTransfer (dry-run) is rejected before .ok is checked."""
    location = _make_location()
    dest = tmp_path / "remote.fa"
    with pytest.raises(InputIntakeError, match="executed transfer, not a dry-run plan"):
        download_remote_fasta(location, dest, transfer=_planned_transfer)


def test_download_remote_fasta_non_ok_fail_closed(tmp_path: Path) -> None:
    """A non-OK TransferResult fails closed."""
    location = _make_location()
    dest = tmp_path / "remote.fa"
    with pytest.raises(InputIntakeError, match="download failed with returncode 1"):
        download_remote_fasta(location, dest, transfer=_fail_transfer)


def test_download_remote_fasta_non_symlink_check(tmp_path: Path) -> None:
    """A symlink destination is rejected."""
    location = _make_location()
    dest = tmp_path / "remote.fa"
    real_file = tmp_path / "real.fa"
    real_file.write_bytes(_CONTENT)

    def fake_transfer(src: str, dst: str) -> TransferResult:
        Path(dst).symlink_to(real_file)
        return _ok_transfer(src, dst)

    with pytest.raises(InputIntakeError, match="regular non-symlink file"):
        download_remote_fasta(location, dest, transfer=fake_transfer)


def test_download_remote_fasta_zero_size_rejection(tmp_path: Path) -> None:
    """A declared zero size_bytes is rejected (must be positive).

    VerifiedRemoteInputLocation itself rejects zero size at construction, so we
    use a mock to test the defense-in-depth check in download_remote_fasta.
    """
    from unittest.mock import MagicMock

    location = MagicMock()
    location.source_uri = _URI
    location.sha256 = hashlib.sha256(_CONTENT).hexdigest()
    location.size_bytes = 0
    dest = tmp_path / "remote.fa"

    def fake_transfer(src: str, dst: str) -> TransferResult:
        Path(dst).write_bytes(_CONTENT)
        return _ok_transfer(src, dst)

    with pytest.raises(InputIntakeError, match="declared size_bytes must be positive"):
        download_remote_fasta(location, dest, transfer=fake_transfer)  # type: ignore[arg-type]


def test_fetch_input_command_downloads_remote_fasta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime CLI fetch-input command dispatches to remote FASTA intake."""
    from click.testing import CliRunner

    from bspp.orchestration.runtime.cli import cli as runtime_cli
    from bspp.orchestration.runtime.preprocessing import execution, input_intake

    location = _make_location()

    class _FakeRunspec:
        input_location = location

    monkeypatch.setattr(execution, "load_preprocessing_phase_runspec", lambda _p: _FakeRunspec())

    captured: dict[str, object] = {}

    def _fake_download(loc: VerifiedRemoteInputLocation, dest: Path, *, transfer=None) -> Path:
        captured["loc"] = loc
        captured["dest"] = dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_CONTENT)
        return dest

    monkeypatch.setattr(input_intake, "download_remote_fasta", _fake_download)

    runspec_path = tmp_path / "runspec.json"
    runspec_path.write_text("{}")
    workspace = tmp_path / "workspace"
    runner = CliRunner()
    result = runner.invoke(
        runtime_cli,
        [
            "preprocessing",
            "fetch-input",
            "--phase-runspec",
            str(runspec_path),
            "--workspace-root",
            str(workspace),
        ],
    )
    assert result.exit_code == 0, f"stdout={result.stdout}, exception={result.exception}"
    assert captured["loc"] is location
    assert captured["dest"] == workspace / location.path
    assert (workspace / location.path).read_bytes() == b">seq\nAAAA:TT\n"


def test_fetch_input_command_rejects_local_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch-input refuses a local input location (only remote inputs are downloaded)."""
    from click.testing import CliRunner

    from bspp.orchestration.contract.phase import VerifiedLocalInputLocation
    from bspp.orchestration.runtime.cli import cli as runtime_cli
    from bspp.orchestration.runtime.preprocessing import execution

    local = VerifiedLocalInputLocation(
        path="inputs/local.fa",
        sha256="a" * 64,
        size_bytes=1,
    )

    class _FakeRunspec:
        input_location = local

    monkeypatch.setattr(execution, "load_preprocessing_phase_runspec", lambda _p: _FakeRunspec())

    runspec_path = tmp_path / "runspec.json"
    runspec_path.write_text("{}")
    runner = CliRunner()
    result = runner.invoke(
        runtime_cli,
        [
            "preprocessing",
            "fetch-input",
            "--phase-runspec",
            str(runspec_path),
            "--workspace-root",
            str(tmp_path / "workspace"),
        ],
    )
    assert result.exit_code != 0
    assert "verified-remote-file" in result.output


def test_normalize_fasta_to_identity_headers_strips_descriptions(tmp_path: Path) -> None:
    """normalize_fasta_to_identity_headers rewrites >id description to >id."""
    path = tmp_path / "input.fa"
    path.write_bytes(b">pdb_5snm_assembly_1 RCSB PDB 5snm biological assembly 1\nAAA:AAA\n")
    result = normalize_fasta_to_identity_headers(path)
    assert result is path
    assert path.read_bytes() == b">pdb_5snm_assembly_1\nAAA:AAA\n"


def test_normalize_fasta_to_identity_headers_preserves_identity_only(tmp_path: Path) -> None:
    """Headers without descriptions are preserved unchanged."""
    path = tmp_path / "input.fa"
    original = b">AFDB_AF-1234567890123456\nAAAA\n>AFDB_AF-2345678901234567\nTTTT\n"
    path.write_bytes(original)
    normalize_fasta_to_identity_headers(path)
    assert path.read_bytes() == original


def test_normalize_fasta_rejects_bare_gt_header_with_input_intake_error(tmp_path: Path) -> None:
    """A bare '>' header line raises InputIntakeError, not IndexError (N8)."""
    path = tmp_path / "input.fa"
    path.write_bytes(b">\nAAA\n")
    with pytest.raises(InputIntakeError, match="no identity token"):
        normalize_fasta_to_identity_headers(path)
    # The temp file must be cleaned up on failure.
    assert not (tmp_path / ".input.fa.normalize.tmp").exists()


def test_normalize_and_carry_forward_produce_identical_identity_only_bytes(tmp_path: Path) -> None:
    """fasta_bytes (control-plane) and _fasta_bytes (runtime) produce identical identity-only bytes."""
    from bspp.orchestration.control.phase_carry_forward import fasta_bytes as control_fasta_bytes
    from bspp.orchestration.runtime.preprocessing.carry_forward import _fasta_bytes as runtime_fasta_bytes

    class _Record:
        def __init__(self, identity: str, sequence: str) -> None:
            self.identity = identity
            self.sequence = sequence

    records = (
        _Record("pdb_5snm_assembly_1", "AAA:TT"),
        _Record("AFDB_AF-1234567890123456", "GGG"),
    )
    expected = b">pdb_5snm_assembly_1\nAAA:TT\n>AFDB_AF-1234567890123456\nGGG\n"
    assert control_fasta_bytes(records) == expected
    assert runtime_fasta_bytes(records) == expected
