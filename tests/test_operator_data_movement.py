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

"""Tests for the operator data-movement surface (contract + runtime + control).

All offline: no network, no s5cmd/gcloud binary required. External boundaries
are injected as fake callables or monkeypatched subprocess.run.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest
import yaml
from click.testing import CliRunner

from bspp.orchestration.contract.operator_data_movement import (
    OperatorTransferEvidence,
    OperatorTransferItem,
    OperatorTransferPlan,
    SeamTransferDecision,
    build_manual_transfer_plan,
    build_plan_referenced_transfer_plan,
    operator_transfer_evidence_from_json,
    operator_transfer_evidence_from_mapping,
    operator_transfer_evidence_id,
    operator_transfer_plan_from_json,
    operator_transfer_plan_from_mapping,
    resolve_execution_tool,
    resolve_seam_transfer_decision,
    resolve_transfer_destination,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    VerifiedLocalBundledArtifactLocation,
)
from bspp.orchestration.runtime.data_movement.common import TransferResult
from bspp.orchestration.runtime.data_movement.operator_transfer import (
    OperatorTransferError,
    _acquire_destination_lock,
    _default_verify_fn,
    _release_destination_lock,
    _remote_object_exists,
    execute_operator_transfer,
    verify_operator_transfer,
)
from bspp.orchestration.runtime.data_movement.s3.client import S3Credentials


def _make_verify_fn(size: int, sha: str):
    """Create a fake verify function returning (size, sha)."""

    def verify_fn(uri: str, path: Path, **kwargs: object) -> tuple[int, str]:
        return (size, sha)

    return verify_fn


def _make_require_tool_fn():
    """Create a fake require_tool that always returns 's5cmd'."""

    def require_tool(tool: str, hint: str = "") -> str:
        return "s5cmd"

    return require_tool


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "docs" / "examples" / "folding-benchmark"
LOCAL_PLAN = FIXTURE_DIR / "folding-phase-plan.yaml"
REMOTE_PLAN = FIXTURE_DIR / "run-plan.yaml"

_SHA256_HEX = "a" * 64
_LZ4_SHA256 = "b" * 64
_S3_PREFIX = "s3://example-bucket/test"
_OVERRIDE_PREFIX = "s3://other-bucket/recovery/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_transfer(src: str, dst: str, **kwargs: object) -> TransferResult:
    return TransferResult(
        tool="s5cmd",
        argv=("s5cmd", "cp", src, dst),
        returncode=0,
        elapsed_s=0.01,
    )


def _make_s3_creds() -> S3Credentials:
    return S3Credentials(
        access_key_id="test-key",
        secret_access_key="test-secret",
        endpoint_url="https://swiftstack.example.com",
    )


def _make_fake_ls_fn(returncode: int, stdout: str = "", stderr: str = ""):
    def fake_ls(destination: str, *, credentials: S3Credentials | None = None) -> TransferResult:
        return TransferResult(
            tool="s5cmd",
            argv=("s5cmd", "ls", destination),
            returncode=returncode,
            elapsed_s=0.0,
            stdout_tail=stdout,
            stderr_tail=stderr,
        )

    return fake_ls


def _make_manual_plan(
    *,
    dry_run: bool = False,
    source: str = "/tmp/test.tar.lz4",
    destination: str = "s3://bucket/prefix/test.tar.lz4",
    size_bytes: int = 64,
    sha256: str = _SHA256_HEX,
) -> OperatorTransferPlan:
    return build_manual_transfer_plan(
        source=source,
        destination=destination,
        size_bytes=size_bytes,
        sha256=sha256,
        dry_run=dry_run,
    )


def _write_local_file(path: Path, content: bytes = b"x" * 64) -> str:
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return digest


# ---------------------------------------------------------------------------
# Tests 1-3: resolve_seam_transfer_decision
# ---------------------------------------------------------------------------


def test_resolve_seam_transfer_decision_local() -> None:
    decision = resolve_seam_transfer_decision(policy="local")
    assert decision.object_storage_leg is False
    assert decision.tool is None
    assert decision.note == "local pass-through (no object-storage leg)"


def test_resolve_seam_transfer_decision_publish() -> None:
    decision = resolve_seam_transfer_decision(policy="publish-to-s3")
    assert decision.object_storage_leg is True
    assert decision.tool == "s5cmd"
    assert decision.note == "s5cmd cp against the S3 endpoint"


def test_resolve_seam_transfer_decision_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        resolve_seam_transfer_decision(policy="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests 4-6: plan-referenced transfer plans
# ---------------------------------------------------------------------------


def test_plan_referenced_from_folding_phase_plan_local() -> None:
    plan = build_plan_referenced_transfer_plan(
        phase_plan_path=LOCAL_PLAN,
        s3_prefix=_S3_PREFIX,
    )
    assert plan.mode == "plan-referenced"
    assert plan.operator_initiated is True
    assert plan.dry_run is True
    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.source_kind == "verified-local-bundled"
    assert item.source != ""
    expected_dst = f"{_S3_PREFIX}/{item.sha256}.tar.lz4"
    assert item.destination == expected_dst
    # Decision is informational; tool is None for local transport
    assert plan.decision.tool is None
    # But execution tool is s5cmd (destination is s3://)
    assert resolve_execution_tool(item.destination) == "s5cmd"
    # artifact_set_id and artifact_location_id populated
    assert item.artifact_set_id is not None
    assert item.artifact_location_id is not None


def test_plan_referenced_from_folding_phase_plan_remote() -> None:
    plan = build_plan_referenced_transfer_plan(
        phase_plan_path=REMOTE_PLAN,
        s3_prefix=_S3_PREFIX,
    )
    assert plan.mode == "plan-referenced"
    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.source_kind == "verified-remote-bundled"
    assert item.source == "s3://example-bucket/users/example-user/rerun-inputs/folding-benchmark/msa-set.lz4"
    expected_dst = f"{_S3_PREFIX}/{item.sha256}.tar.lz4"
    assert item.destination == expected_dst
    assert item.source != item.destination
    assert item.artifact_set_id is not None
    assert item.artifact_location_id is not None


def test_plan_referenced_with_override_prefix() -> None:
    plan = build_plan_referenced_transfer_plan(
        phase_plan_path=LOCAL_PLAN,
        s3_prefix=None,
        override_prefix=_OVERRIDE_PREFIX,
    )
    assert plan.override_destination_prefix == _OVERRIDE_PREFIX
    item = plan.items[0]
    expected_dst = f"{_OVERRIDE_PREFIX.rstrip('/')}/{item.sha256}.tar.lz4"
    assert item.destination == expected_dst
    assert plan.authority_reference == str(LOCAL_PLAN)


# ---------------------------------------------------------------------------
# Tests 7-8: manual mode and dry-run default
# ---------------------------------------------------------------------------


def test_manual_mode_basic() -> None:
    plan = build_manual_transfer_plan(
        source="/tmp/test.tar.lz4",
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=_SHA256_HEX,
    )
    assert plan.mode == "manual"
    assert plan.items[0].source_kind == "explicit-path"
    assert plan.operator_initiated is True
    assert plan.authority_reference is None


def test_dry_run_default() -> None:
    plan = build_plan_referenced_transfer_plan(
        phase_plan_path=LOCAL_PLAN,
        s3_prefix=_S3_PREFIX,
    )
    assert plan.dry_run is True


# ---------------------------------------------------------------------------
# Test 9: round-trip serialization
# ---------------------------------------------------------------------------


def test_plan_to_mapping_round_trip() -> None:
    # Plan-referenced
    plan_ref = build_plan_referenced_transfer_plan(
        phase_plan_path=LOCAL_PLAN,
        s3_prefix=_S3_PREFIX,
        dry_run=False,
    )
    mapping = plan_ref.to_mapping()
    # null for None-valued optionals
    assert "authority_reference" in mapping
    assert "override_destination_prefix" in mapping
    assert mapping["override_destination_prefix"] is None
    round_trip = operator_transfer_plan_from_mapping(mapping)
    assert round_trip == plan_ref

    # Manual
    plan_manual = _make_manual_plan()
    mapping_m = plan_manual.to_mapping()
    assert mapping_m["authority_reference"] is None
    assert mapping_m["override_destination_prefix"] is None
    round_trip_m = operator_transfer_plan_from_mapping(mapping_m)
    assert round_trip_m == plan_manual

    # from_json
    json_text = plan_ref.to_json()
    assert operator_transfer_plan_from_json(json_text) == plan_ref

    # Missing keys treated as None for optionals (manual plan, where authority_reference is None)
    mapping_m2 = dict(mapping_m)
    del mapping_m2["override_destination_prefix"]
    del mapping_m2["authority_reference"]
    rt_missing = operator_transfer_plan_from_mapping(mapping_m2)
    assert rt_missing.override_destination_prefix is None
    assert rt_missing.authority_reference is None


# ---------------------------------------------------------------------------
# Tests 10-11: evidence record content
# ---------------------------------------------------------------------------


def test_evidence_record_content_plan_referenced() -> None:
    plan = build_plan_referenced_transfer_plan(
        phase_plan_path=LOCAL_PLAN,
        s3_prefix=_S3_PREFIX,
        dry_run=False,
    )
    item = plan.items[0]
    nonce = "abcdef0123456789"
    evidence_id = operator_transfer_evidence_id(
        {
            "source": item.source,
            "destination": item.destination,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "nonce": nonce,
        }
    )
    evidence = OperatorTransferEvidence(
        evidence_id=evidence_id,
        mode="plan-referenced",
        operator_initiated=True,
        authority_reference=str(LOCAL_PLAN),
        authority_digest=plan.authority_digest,
        override_destination_prefix=None,
        item=item,
        transfer_tool="s5cmd",
        transfer_argv=("s5cmd", "cp", item.source, item.destination),
        transfer_returncode=0,
        transfer_elapsed_s=0.01,
        verified_size_bytes=item.size_bytes,
        verified_sha256=item.sha256,
        transferred_at="2026-01-01T00:00:00.000000Z",
        original_evidence_preserved=True,
        evidence_nonce=nonce,
    )
    assert evidence.mode == "plan-referenced"
    assert evidence.operator_initiated is True
    assert evidence.authority_reference == str(LOCAL_PLAN)
    assert evidence.original_evidence_preserved is True
    # evidence_id matches
    assert evidence.evidence_id == operator_transfer_evidence_id(evidence.to_mapping_for_id())

    # Mismatched evidence_id rejected
    with pytest.raises(ValueError, match="evidence_id does not match"):
        OperatorTransferEvidence(
            evidence_id="wrong-id",
            mode="plan-referenced",
            operator_initiated=True,
            authority_reference=str(LOCAL_PLAN),
            authority_digest=plan.authority_digest,
            override_destination_prefix=None,
            item=item,
            transfer_tool="s5cmd",
            transfer_argv=("s5cmd", "cp"),
            transfer_returncode=0,
            transfer_elapsed_s=0.01,
            verified_size_bytes=item.size_bytes,
            verified_sha256=item.sha256,
            transferred_at="2026-01-01T00:00:00.000000Z",
            original_evidence_preserved=False,
            evidence_nonce=nonce,
        )


def test_evidence_record_content_manual() -> None:
    plan = _make_manual_plan()
    item = plan.items[0]
    nonce = "1234567890abcdef"
    evidence_id = operator_transfer_evidence_id(
        {
            "source": item.source,
            "destination": item.destination,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "nonce": nonce,
        }
    )
    evidence = OperatorTransferEvidence(
        evidence_id=evidence_id,
        mode="manual",
        operator_initiated=True,
        authority_reference=None,
        authority_digest=None,
        override_destination_prefix=None,
        item=item,
        transfer_tool="s5cmd",
        transfer_argv=("s5cmd", "cp", item.source, item.destination),
        transfer_returncode=0,
        transfer_elapsed_s=0.01,
        verified_size_bytes=item.size_bytes,
        verified_sha256=item.sha256,
        transferred_at="2026-01-01T00:00:00.000000Z",
        original_evidence_preserved=False,
        evidence_nonce=nonce,
    )
    assert evidence.mode == "manual"
    assert evidence.authority_reference is None

    # Round-trip
    rt = operator_transfer_evidence_from_mapping(evidence.to_mapping())
    assert rt == evidence
    assert operator_transfer_evidence_from_json(evidence.to_json()) == evidence


# ---------------------------------------------------------------------------
# Test 12: overwrite refusal and --force preserves original evidence
# ---------------------------------------------------------------------------


def test_overwrite_refusal_on_existing_object(tmp_path: Path) -> None:
    # First transfer succeeds
    local_file = tmp_path / "test.tar.lz4"
    sha = _write_local_file(local_file)
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=sha,
    )
    evidence_dir = tmp_path / "evidence"

    # Fake ls that reports existing object
    existing_ls = _make_fake_ls_fn(0, stdout="2026/06/09 01:42:00         64  s3://bucket/prefix/test.tar.lz4\n")
    absent_ls = _make_fake_ls_fn(1, stderr='ERROR "ls s3://bucket/prefix/test.tar.lz4": no object found\n')

    # First transfer: destination absent, succeeds
    fake_verify = _make_verify_fn(64, sha)
    evidence1 = execute_operator_transfer(
        plan,
        evidence_dir=evidence_dir,
        transfer_fn_map={"s5cmd": _ok_transfer},
        verify_fn=fake_verify,
        ls_fn=absent_ls,
    )
    assert len(evidence1) == 1

    # Second transfer without --force: destination exists, raises
    with pytest.raises(OperatorTransferError, match="destination already exists"):
        execute_operator_transfer(
            plan,
            evidence_dir=evidence_dir,
            transfer_fn_map={"s5cmd": _ok_transfer},
            verify_fn=fake_verify,
            ls_fn=existing_ls,
        )

    # Second transfer with --force: succeeds, original evidence preserved
    evidence2 = execute_operator_transfer(
        plan,
        force=True,
        evidence_dir=evidence_dir,
        transfer_fn_map={"s5cmd": _ok_transfer},
        verify_fn=fake_verify,
        ls_fn=existing_ls,
    )
    assert len(evidence2) == 1
    assert evidence2[0].original_evidence_preserved is True
    # evidence_id differs between the two records
    assert evidence1[0].evidence_id != evidence2[0].evidence_id
    # Original evidence file is preserved byte-identically
    original_path = evidence_dir / f"{evidence1[0].evidence_id}.json"
    assert original_path.exists()
    original_bytes = original_path.read_bytes()
    new_path = evidence_dir / f"{evidence2[0].evidence_id}.json"
    assert new_path.exists()
    assert original_path.read_bytes() == original_bytes  # unchanged


# ---------------------------------------------------------------------------
# Test 13: checksum mismatch fail-closed
# ---------------------------------------------------------------------------


def test_checksum_mismatch_fail_closed(tmp_path: Path) -> None:
    local_file = tmp_path / "test.tar.lz4"
    sha = _write_local_file(local_file)
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=sha,
    )
    evidence_dir = tmp_path / "evidence"
    absent_ls = _make_fake_ls_fn(1, stderr='ERROR "ls": no object found\n')
    bad_verify = _make_verify_fn(64, "0" * 64)

    with pytest.raises(OperatorTransferError, match="sha256 mismatch"):
        execute_operator_transfer(
            plan,
            evidence_dir=evidence_dir,
            transfer_fn_map={"s5cmd": _ok_transfer},
            verify_fn=bad_verify,
            ls_fn=absent_ls,
        )
    # No evidence file written
    if evidence_dir.exists():
        assert len(list(evidence_dir.iterdir())) == 0


# ---------------------------------------------------------------------------
# Tests 14-15: seam transport delegation
# ---------------------------------------------------------------------------


def test_seam_transport_delegates_to_contract() -> None:
    from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer
    from bspp.orchestration.runtime.folding.seam_transport import resolve_seam_transfer

    resolved = resolve_seam_transfer(policy="publish-to-s3")
    decision = resolve_seam_transfer_decision(policy="publish-to-s3")
    assert resolved.tool == decision.tool
    assert resolved.object_storage_leg == decision.object_storage_leg
    assert resolved.note == decision.note
    assert resolved.transfer is s3_transfer.cp


def test_seam_transport_wraps_value_error_as_seam_transport_error() -> None:
    from bspp.orchestration.runtime.folding.seam_transport import SeamTransportError, resolve_seam_transfer

    with pytest.raises(SeamTransportError):
        resolve_seam_transfer(policy="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 16: control plane boundary
# ---------------------------------------------------------------------------


def test_control_plane_boundary_no_runtime_import() -> None:
    import ast

    module_path = (
        Path(__file__).resolve().parents[1]
        / "packages"
        / "orchestration-control"
        / "src"
        / "bspp"
        / "orchestration"
        / "control"
        / "data_movement.py"
    )
    tree = ast.parse(module_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom | ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("bspp.orchestration.runtime"), (
                    f"control.data_movement must not import runtime: {alias.name}"
                )


# ---------------------------------------------------------------------------
# Test 17: runtime CLI plan JSON deserialization
# ---------------------------------------------------------------------------


def test_runtime_cli_plan_json_deserialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _make_manual_plan(dry_run=False)
    plan_json = tmp_path / "plan.json"
    plan_json.write_text(plan.to_json())

    from bspp.orchestration.runtime import cli as runtime_cli

    fake_called: list[OperatorTransferPlan] = []

    def fake_execute(plan_arg: OperatorTransferPlan, **kwargs: object) -> tuple[OperatorTransferEvidence, ...]:
        fake_called.append(plan_arg)
        return ()

    monkeypatch.setattr(
        "bspp.orchestration.runtime.data_movement.operator_transfer.execute_operator_transfer",
        fake_execute,
    )

    runner = CliRunner()
    result = runner.invoke(
        runtime_cli.cli,
        ["data-movement", "operator", "execute", "--plan-json", str(plan_json)],
    )
    assert result.exit_code == 0, result.output
    assert len(fake_called) == 1
    assert fake_called[0] == plan


# ---------------------------------------------------------------------------
# Test 18: remote no-override destination not self-copy
# ---------------------------------------------------------------------------


def test_remote_no_override_destination_not_self_copy() -> None:
    plan = build_plan_referenced_transfer_plan(
        phase_plan_path=REMOTE_PLAN,
        s3_prefix=_S3_PREFIX,
    )
    item = plan.items[0]
    assert item.source != item.destination
    assert item.destination == f"{_S3_PREFIX}/{item.sha256}.tar.lz4"


# ---------------------------------------------------------------------------
# Test 19: execute rejects dry-run plan
# ---------------------------------------------------------------------------


def test_execute_rejects_dry_run_plan(tmp_path: Path) -> None:
    plan = _make_manual_plan(dry_run=True)
    fake_transfer = MagicMock()
    with pytest.raises(OperatorTransferError, match="dry-run"):
        execute_operator_transfer(plan, transfer_fn_map={"s5cmd": fake_transfer})
    fake_transfer.assert_not_called()


# ---------------------------------------------------------------------------
# Tests 20-21: pre-transfer source verification
# ---------------------------------------------------------------------------


def test_pre_transfer_source_verification_rejects_stale_bytes(tmp_path: Path) -> None:
    local_file = tmp_path / "test.tar.lz4"
    content = b"x" * 64
    local_file.write_bytes(content)
    # Plan declares a different sha256
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256="0" * 64,
    )
    fake_transfer = MagicMock()
    with pytest.raises(OperatorTransferError, match="refusing to transfer stale bytes"):
        execute_operator_transfer(plan, transfer_fn_map={"s5cmd": fake_transfer})
    fake_transfer.assert_not_called()


def test_pre_transfer_source_verification_passes_on_valid_bytes(tmp_path: Path) -> None:
    local_file = tmp_path / "test.tar.lz4"
    content = b"x" * 64
    local_file.write_bytes(content)
    real_sha = hashlib.sha256(content).hexdigest()
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=real_sha,
    )
    absent_ls = _make_fake_ls_fn(1, stderr='ERROR "ls": no object found\n')
    fake_verify = _make_verify_fn(64, real_sha)
    evidence_dir = tmp_path / "evidence"
    evidence = execute_operator_transfer(
        plan,
        evidence_dir=evidence_dir,
        transfer_fn_map={"s5cmd": _ok_transfer},
        verify_fn=fake_verify,
        ls_fn=absent_ls,
    )
    assert len(evidence) == 1
    assert evidence[0].verified_sha256 == real_sha
    assert evidence[0].verified_size_bytes == 64
    # Evidence file written
    ev_path = evidence_dir / f"{evidence[0].evidence_id}.json"
    assert ev_path.exists()


# ---------------------------------------------------------------------------
# Tests 22-24: resolve_execution_tool
# ---------------------------------------------------------------------------


def test_resolve_execution_tool_s3() -> None:
    assert resolve_execution_tool("s3://bucket/key") == "s5cmd"


def test_resolve_execution_tool_rejects_gs() -> None:
    with pytest.raises(ValueError, match="GCS destinations are not supported"):
        resolve_execution_tool("gs://bucket/key")


def test_resolve_execution_tool_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="unsupported destination scheme"):
        resolve_execution_tool("file:///local/path")


# ---------------------------------------------------------------------------
# Test 25: per-item tool dispatch
# ---------------------------------------------------------------------------


def test_per_item_tool_dispatch(tmp_path: Path) -> None:
    local_file = tmp_path / "test.tar.lz4"
    sha = _write_local_file(local_file)
    item1 = OperatorTransferItem(
        source=str(local_file),
        destination="s3://bucket/prefix/item1.tar.lz4",
        size_bytes=64,
        sha256=sha,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="explicit-path",
        description="item1",
    )
    item2 = OperatorTransferItem(
        source=str(local_file),
        destination="s3://bucket/prefix/item2.tar.lz4",
        size_bytes=64,
        sha256=sha,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="explicit-path",
        description="item2",
    )
    decision = resolve_seam_transfer_decision(policy="publish-to-s3")
    plan = OperatorTransferPlan(
        mode="manual",
        items=(item1, item2),
        decision=decision,
        operator_initiated=True,
        authority_reference=None,
        authority_digest=None,
        override_destination_prefix=None,
        dry_run=False,
    )
    calls: list[tuple[str, str]] = []

    def recording_transfer(src: str, dst: str, **kwargs: object) -> TransferResult:
        calls.append((src, dst))
        return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=0, elapsed_s=0.0)

    absent_ls = _make_fake_ls_fn(1, stderr='ERROR "ls": no object found\n')
    fake_verify = _make_verify_fn(64, sha)
    evidence = execute_operator_transfer(
        plan,
        transfer_fn_map={"s5cmd": recording_transfer},
        verify_fn=fake_verify,
        ls_fn=absent_ls,
    )
    assert len(evidence) == 2
    assert len(calls) == 2
    assert calls[0][1] == "s3://bucket/prefix/item1.tar.lz4"
    assert calls[1][1] == "s3://bucket/prefix/item2.tar.lz4"


# ---------------------------------------------------------------------------
# Test 26: resolve_transfer_destination rejects empty bucket
# ---------------------------------------------------------------------------


def test_resolve_transfer_destination_rejects_empty_bucket(tmp_path: Path) -> None:
    # We need a location object for resolve_transfer_destination
    # Use a simple mock-like approach: the function only uses lz4_sha256
    location = MagicMock(spec=VerifiedLocalBundledArtifactLocation)
    location.lz4_sha256 = _LZ4_SHA256

    for bad_prefix in ["s3://", "s3:///", "s3:", "not-a-uri"]:
        with pytest.raises(ValueError):
            resolve_transfer_destination(
                location=location,
                s3_prefix=bad_prefix,
                override_prefix=None,
            )

    # Valid prefixes succeed
    dst1 = resolve_transfer_destination(location=location, s3_prefix="s3://bucket", override_prefix=None)
    assert dst1 == f"s3://bucket/{_LZ4_SHA256}.tar.lz4"

    dst2 = resolve_transfer_destination(location=location, s3_prefix="s3://bucket/prefix/", override_prefix=None)
    assert dst2 == f"s3://bucket/prefix/{_LZ4_SHA256}.tar.lz4"


# ---------------------------------------------------------------------------
# Tests 27-29: OperatorTransferItem validation
# ---------------------------------------------------------------------------


def test_operator_transfer_item_rejects_gs_destination() -> None:
    with pytest.raises(ValueError, match="destination must start with s3://"):
        OperatorTransferItem(
            source="local",
            destination="gs://bucket/key",
            size_bytes=64,
            sha256=_SHA256_HEX,
            artifact_set_id=None,
            artifact_location_id=None,
            source_kind="explicit-path",
            description="test",
        )


def test_operator_transfer_item_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="non-negative int"):
        OperatorTransferItem(
            source="/tmp/a",
            destination="s3://bucket/key",
            size_bytes=-1,
            sha256=_SHA256_HEX,
            artifact_set_id=None,
            artifact_location_id=None,
            source_kind="explicit-path",
            description="test",
        )


def test_operator_transfer_item_rejects_bad_sha256() -> None:
    with pytest.raises(ValueError, match="sha256 must be 64 lowercase hex"):
        OperatorTransferItem(
            source="/tmp/a",
            destination="s3://bucket/key",
            size_bytes=64,
            sha256="not-hex",
            artifact_set_id=None,
            artifact_location_id=None,
            source_kind="explicit-path",
            description="test",
        )


# ---------------------------------------------------------------------------
# Test 30: evidence atomicity multi-item
# ---------------------------------------------------------------------------


def test_evidence_atomicity_multi_item(tmp_path: Path) -> None:
    local_file = tmp_path / "test.tar.lz4"
    sha = _write_local_file(local_file)
    item1 = OperatorTransferItem(
        source=str(local_file),
        destination="s3://bucket/prefix/item1.tar.lz4",
        size_bytes=64,
        sha256=sha,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="explicit-path",
        description="item1",
    )
    item2 = OperatorTransferItem(
        source=str(local_file),
        destination="s3://bucket/prefix/item2.tar.lz4",
        size_bytes=64,
        sha256=sha,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="explicit-path",
        description="item2",
    )
    decision = resolve_seam_transfer_decision(policy="publish-to-s3")
    plan = OperatorTransferPlan(
        mode="manual",
        items=(item1, item2),
        decision=decision,
        operator_initiated=True,
        authority_reference=None,
        authority_digest=None,
        override_destination_prefix=None,
        dry_run=False,
    )
    evidence_dir = tmp_path / "evidence"
    absent_ls = _make_fake_ls_fn(1, stderr='ERROR "ls": no object found\n')

    call_count = [0]

    def failing_transfer(src: str, dst: str, **kwargs: object) -> TransferResult:
        call_count[0] += 1
        if call_count[0] == 2:
            return TransferResult(
                tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=1, elapsed_s=0.0, stderr_tail="boom"
            )
        return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=0, elapsed_s=0.0)

    sha = hashlib.sha256(b"x" * 64).hexdigest()
    fake_verify = _make_verify_fn(64, sha)
    with pytest.raises(OperatorTransferError):
        execute_operator_transfer(
            plan,
            evidence_dir=evidence_dir,
            transfer_fn_map={"s5cmd": failing_transfer},
            verify_fn=fake_verify,
            ls_fn=absent_ls,
        )
    # No evidence files written
    if evidence_dir.exists():
        assert len(list(evidence_dir.iterdir())) == 0


# ---------------------------------------------------------------------------
# Test 31: resolve_execution_tool wrapped as OperatorTransferError
# ---------------------------------------------------------------------------


def test_resolve_execution_tool_wrapped_as_operator_transfer_error() -> None:
    # Construct an item with gs:// destination bypassing __post_init__ is not possible
    # since __post_init__ rejects it. Instead, test via a plan with a gs:// item
    # constructed via from_mapping (which also validates). So we test the execute
    # path directly: the item must already exist with s3://, and we monkeypatch
    # resolve_execution_tool to raise ValueError.
    import bspp.orchestration.runtime.data_movement.operator_transfer as ot

    plan = _make_manual_plan(dry_run=False)

    def bad_resolve(destination: str) -> str:
        raise ValueError("unsupported destination scheme for operator transfer: gs://bucket/key")

    original_fn = ot.resolve_execution_tool
    ot.resolve_execution_tool = bad_resolve
    try:
        with pytest.raises(OperatorTransferError, match="unsupported destination scheme"):
            execute_operator_transfer(plan, transfer_fn_map={"s5cmd": _ok_transfer})
    finally:
        ot.resolve_execution_tool = original_fn


# ---------------------------------------------------------------------------
# Test 32: remote source size pre-flight
# ---------------------------------------------------------------------------


def test_remote_source_size_preflight(tmp_path: Path) -> None:
    item = OperatorTransferItem(
        source="s3://source-bucket/key.tar.lz4",
        destination="s3://dest-bucket/key.tar.lz4",
        size_bytes=64,
        sha256=_SHA256_HEX,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="verified-remote-bundled",
        description="remote source",
    )
    decision = resolve_seam_transfer_decision(policy="publish-to-s3")
    plan = OperatorTransferPlan(
        mode="manual",
        items=(item,),
        decision=decision,
        operator_initiated=True,
        authority_reference=None,
        authority_digest=None,
        override_destination_prefix=None,
        dry_run=False,
    )
    # Source ls reports size mismatch
    source_ls = _make_fake_ls_fn(0, stdout="2026/06/09 01:42:00         128  s3://source-bucket/key.tar.lz4\n")
    # Destination ls reports absent
    dest_ls = _make_fake_ls_fn(1, stderr='ERROR "ls": no object found\n')

    call_count = [0]

    def dual_ls(destination: str, *, credentials: S3Credentials | None = None) -> TransferResult:
        call_count[0] += 1
        if call_count[0] == 1:
            return source_ls(destination, credentials=credentials)
        return dest_ls(destination, credentials=credentials)

    fake_transfer = MagicMock()
    with pytest.raises(OperatorTransferError, match="remote source size mismatch"):
        execute_operator_transfer(
            plan,
            transfer_fn_map={"s5cmd": fake_transfer},
            ls_fn=dual_ls,
        )
    fake_transfer.assert_not_called()


# ---------------------------------------------------------------------------
# Test 33: manual mode rejects override_prefix
# ---------------------------------------------------------------------------


def test_manual_mode_rejects_override_prefix() -> None:
    with pytest.raises(ValueError, match="does not accept override_prefix"):
        build_manual_transfer_plan(
            source="/tmp/test.tar.lz4",
            destination="s3://bucket/prefix/test.tar.lz4",
            size_bytes=64,
            sha256=_SHA256_HEX,
            override_prefix="s3://bucket/fix/",
        )


# ---------------------------------------------------------------------------
# Tests 34-38: real s5cmd argv/exit-code/output contract via subprocess.run monkeypatch
# ---------------------------------------------------------------------------


def _monkeypatch_subprocess_run(monkeypatch: pytest.MonkeyPatch, responses: list[subprocess.CompletedProcess[str]]):
    call_idx = [0]

    def fake_run(argv, **kwargs):
        idx = min(call_idx[0], len(responses) - 1)
        call_idx[0] += 1
        return responses[idx]

    monkeypatch.setattr("bspp.orchestration.runtime.data_movement.operator_transfer.subprocess.run", fake_run)
    return call_idx


def test_remote_object_exists_treats_no_object_found_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    _monkeypatch_subprocess_run(
        monkeypatch,
        [
            subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr='ERROR "ls s3://bucket/key": no object found\n'
            )
        ],
    )
    monkeypatch.setattr(
        "bspp.orchestration.runtime.data_movement.operator_transfer.require_tool",
        _make_require_tool_fn(),
    )
    exists, size = _remote_object_exists("s3://bucket/key", credentials=_make_s3_creds(), ls_fn=None)
    assert exists is False
    assert size is None


def test_remote_object_exists_returns_true_on_existing_object(monkeypatch: pytest.MonkeyPatch) -> None:
    _monkeypatch_subprocess_run(
        monkeypatch,
        [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="2026/06/09 01:42:00         101939200  s3://bucket/key\n",
                stderr="",
            )
        ],
    )
    monkeypatch.setattr(
        "bspp.orchestration.runtime.data_movement.operator_transfer.require_tool",
        _make_require_tool_fn(),
    )
    exists, size = _remote_object_exists("s3://bucket/key", credentials=_make_s3_creds(), ls_fn=None)
    assert exists is True
    assert size == 101939200


def test_remote_object_exists_normalizes_relative_names(monkeypatch: pytest.MonkeyPatch) -> None:
    _monkeypatch_subprocess_run(
        monkeypatch,
        [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="2026/06/09 01:42:00         101939200  key\n",
                stderr="",
            )
        ],
    )
    monkeypatch.setattr(
        "bspp.orchestration.runtime.data_movement.operator_transfer.require_tool",
        _make_require_tool_fn(),
    )
    exists, size = _remote_object_exists("s3://bucket/key", credentials=_make_s3_creds(), ls_fn=None)
    assert exists is True
    assert size == 101939200


def test_remote_object_exists_swiftstack_bucket_relative_key() -> None:
    # s5cmd 2.3.0 `ls <exact key>` on SwiftStack prints the full bucket-relative
    # key WITHOUT the s3://bucket prefix (recorded in the data-movement exercise
    # evidence). This is the shape the old parser doubled into a wrong path.
    destination = "s3://example-bucket/users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4"
    ls_fn = _make_fake_ls_fn(
        0,
        stdout=(
            "2026/09/17 12:15:46               280  "
            "users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4\n"
        ),
    )
    exists, size = _remote_object_exists(destination, credentials=_make_s3_creds(), ls_fn=ls_fn)
    assert exists is True
    assert size == 280


def test_remote_object_exists_swiftstack_prefix_relative_basename() -> None:
    # s5cmd 2.3.0 `ls <prefix>/` prints the name relative to the listed prefix
    # (recorded in the data-movement exercise evidence). In the exact-key ls
    # path this is the legacy "basename under parent" shape.
    destination = "s3://example-bucket/users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4"
    ls_fn = _make_fake_ls_fn(0, stdout="2026/09/17 12:15:46               280  msa-set.tar.lz4\n")
    exists, size = _remote_object_exists(destination, credentials=_make_s3_creds(), ls_fn=ls_fn)
    assert exists is True
    assert size == 280


def test_remote_object_exists_whitespace_key_bucket_relative() -> None:
    # Keys containing whitespace must not be truncated by the row parser, and a
    # key containing the literal " DIR " token must not be dropped as if it were
    # a directory row.
    destination = "s3://example-bucket/users/example-user/my DIR file.tar.lz4"
    ls_fn = _make_fake_ls_fn(
        0,
        stdout="2026/09/17 12:15:46               280  users/example-user/my DIR file.tar.lz4\n",
    )
    exists, size = _remote_object_exists(destination, credentials=_make_s3_creds(), ls_fn=ls_fn)
    assert exists is True
    assert size == 280


def test_remote_object_exists_whitespace_key_prefix_relative() -> None:
    destination = "s3://example-bucket/users/example-user/data-movement-exercise/up/s5cmd/my file.tar.lz4"
    ls_fn = _make_fake_ls_fn(0, stdout="2026/09/17 12:15:46               280  my file.tar.lz4\n")
    exists, size = _remote_object_exists(destination, credentials=_make_s3_creds(), ls_fn=ls_fn)
    assert exists is True
    assert size == 280


def test_remote_object_exists_skips_dir_rows_and_matches_object() -> None:
    destination = "s3://example-bucket/users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4"
    ls_fn = _make_fake_ls_fn(
        0,
        stdout=(
            "                           DIR  users/example-user/data-movement-exercise/up/s5cmd/\n"
            "2026/09/17 12:15:46               280  "
            "users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4\n"
        ),
    )
    exists, size = _remote_object_exists(destination, credentials=_make_s3_creds(), ls_fn=ls_fn)
    assert exists is True
    assert size == 280


def test_remote_object_exists_empty_listing_is_absent() -> None:
    ls_fn = _make_fake_ls_fn(0, stdout="")
    exists, size = _remote_object_exists("s3://bucket/key", credentials=_make_s3_creds(), ls_fn=ls_fn)
    assert exists is False
    assert size is None


def test_remote_object_exists_fail_closed_on_nonmatching_listing() -> None:
    # rc=0 with a non-empty listing that does NOT contain the destination must
    # never be reported as "absent" — the overwrite-refusal path fails closed
    # when the listing shape is not understood.
    destination = "s3://example-bucket/users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4"
    ls_fn = _make_fake_ls_fn(0, stdout="2026/09/17 12:15:46               280  users/someone-else/other.tar.lz4\n")
    with pytest.raises(OperatorTransferError, match="refusing to treat as absent"):
        _remote_object_exists(destination, credentials=_make_s3_creds(), ls_fn=ls_fn)


def test_execute_operator_transfer_overwrite_refusal_swiftstack_bucket_relative(tmp_path: Path) -> None:
    # The data-safety property: a pre-existing destination listed with the
    # SwiftStack bucket-relative shape must refuse without --force.
    local_file = tmp_path / "test.tar.lz4"
    sha = _write_local_file(local_file)
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://example-bucket/users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4",
        size_bytes=64,
        sha256=sha,
    )
    existing_ls = _make_fake_ls_fn(
        0,
        stdout=(
            "2026/09/17 12:15:46               280  "
            "users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4\n"
        ),
    )
    with pytest.raises(OperatorTransferError, match="destination already exists"):
        execute_operator_transfer(
            plan,
            transfer_fn_map={"s5cmd": _ok_transfer},
            ls_fn=existing_ls,
            credentials=_make_s3_creds(),
        )


def test_remote_source_swiftstack_bucket_relative_proceeds_to_transfer() -> None:
    # The C6 repair: a remote s3:// source listed with the SwiftStack
    # bucket-relative shape must pass the size pre-flight and reach the
    # transfer callable (previously it was reported "not found").
    item = OperatorTransferItem(
        source="s3://example-bucket/users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4",
        destination=("s3://example-bucket/users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4.copy"),
        size_bytes=280,
        sha256=_SHA256_HEX,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="verified-remote-bundled",
        description="remote source",
    )
    decision = resolve_seam_transfer_decision(policy="publish-to-s3")
    plan = OperatorTransferPlan(
        mode="manual",
        items=(item,),
        decision=decision,
        operator_initiated=True,
        authority_reference=None,
        authority_digest=None,
        override_destination_prefix=None,
        dry_run=False,
    )
    source_ls = _make_fake_ls_fn(
        0,
        stdout=(
            "2026/09/17 12:15:46               280  "
            "users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4\n"
        ),
    )
    dest_ls = _make_fake_ls_fn(1, stderr='ERROR "ls": no object found\n')

    call_count = [0]

    def dual_ls(destination: str, *, credentials: S3Credentials | None = None) -> TransferResult:
        call_count[0] += 1
        if call_count[0] == 1:
            return source_ls(destination, credentials=credentials)
        return dest_ls(destination, credentials=credentials)

    calls: list[tuple[str, str]] = []

    def recording_transfer(src: str, dst: str, **kwargs: object) -> TransferResult:
        calls.append((src, dst))
        return _ok_transfer(src, dst)

    evidence = execute_operator_transfer(
        plan,
        transfer_fn_map={"s5cmd": recording_transfer},
        verify_fn=_make_verify_fn(280, _SHA256_HEX),
        ls_fn=dual_ls,
        credentials=_make_s3_creds(),
    )
    assert len(evidence) == 1
    assert calls == [
        (
            "s3://example-bucket/users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4",
            "s3://example-bucket/users/example-user/data-movement-exercise/up/s5cmd/msa-set.tar.lz4.copy",
        )
    ]


def test_remote_object_exists_fail_closed_on_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _monkeypatch_subprocess_run(
        monkeypatch,
        [subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="ERROR access denied\n")],
    )
    monkeypatch.setattr(
        "bspp.orchestration.runtime.data_movement.operator_transfer.require_tool",
        _make_require_tool_fn(),
    )
    with pytest.raises(OperatorTransferError, match="failed with rc=1"):
        _remote_object_exists("s3://bucket/key", credentials=_make_s3_creds(), ls_fn=None)


def test_remote_object_exists_argv_is_direct_ls_not_glob(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def capturing_run(argv, **kwargs):
        captured.append(list(argv))
        return subprocess.CompletedProcess(args=argv, returncode=1, stdout="", stderr='ERROR "ls": no object found\n')

    monkeypatch.setattr("bspp.orchestration.runtime.data_movement.operator_transfer.subprocess.run", capturing_run)
    monkeypatch.setattr(
        "bspp.orchestration.runtime.data_movement.operator_transfer.require_tool",
        _make_require_tool_fn(),
    )
    _remote_object_exists("s3://bucket/key", credentials=_make_s3_creds(), ls_fn=None)
    assert captured[0] == ["s5cmd", "--endpoint-url", "https://swiftstack.example.com", "ls", "s3://bucket/key"]
    # NOT ["s5cmd", ..., "ls", "s3://bucket/key/*"]


# ---------------------------------------------------------------------------
# Tests 39-41: end-to-end execution with real s5cmd contract
# ---------------------------------------------------------------------------


def test_execute_operator_transfer_overwrite_check_uses_real_s5cmd_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_file = tmp_path / "test.tar.lz4"
    sha = _write_local_file(local_file)
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://dest/key.tar.lz4",
        size_bytes=64,
        sha256=sha,
    )
    # First subprocess call: destination ls → no object found (rc=1)
    # Second subprocess call: transfer cp → success (rc=0) — but we inject transfer_fn_map
    # So only the ls subprocess call happens
    _monkeypatch_subprocess_run(
        monkeypatch,
        [
            subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr='ERROR "ls s3://dest/key.tar.lz4": no object found\n'
            )
        ],
    )
    monkeypatch.setattr(
        "bspp.orchestration.runtime.data_movement.operator_transfer.require_tool",
        _make_require_tool_fn(),
    )
    fake_verify = _make_verify_fn(64, sha)
    evidence = execute_operator_transfer(
        plan,
        transfer_fn_map={"s5cmd": _ok_transfer},
        verify_fn=fake_verify,
        ls_fn=None,
        credentials=_make_s3_creds(),
    )
    assert len(evidence) == 1
    assert evidence[0].verified_sha256 == sha


def test_execute_operator_transfer_overwrite_refusal_with_real_s5cmd_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_file = tmp_path / "test.tar.lz4"
    sha = _write_local_file(local_file)
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://dest/key.tar.lz4",
        size_bytes=64,
        sha256=sha,
    )
    _monkeypatch_subprocess_run(
        monkeypatch,
        [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="2026/06/09 01:42:00         64  s3://dest/key.tar.lz4\n",
                stderr="",
            )
        ],
    )
    monkeypatch.setattr(
        "bspp.orchestration.runtime.data_movement.operator_transfer.require_tool",
        _make_require_tool_fn(),
    )
    with pytest.raises(OperatorTransferError, match="destination already exists"):
        execute_operator_transfer(
            plan,
            transfer_fn_map={"s5cmd": _ok_transfer},
            ls_fn=None,
            credentials=_make_s3_creds(),
        )

    # With force, it proceeds (the ls still returns existing, but force bypasses)
    evidence = execute_operator_transfer(
        plan,
        force=True,
        transfer_fn_map={"s5cmd": _ok_transfer},
        verify_fn=_make_verify_fn(64, sha),
        ls_fn=None,
        credentials=_make_s3_creds(),
    )
    assert len(evidence) == 1


def test_remote_source_preflight_uses_real_s5cmd_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    item = OperatorTransferItem(
        source="s3://source-bucket/key.tar.lz4",
        destination="s3://dest-bucket/key.tar.lz4",
        size_bytes=64,
        sha256=_SHA256_HEX,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="verified-remote-bundled",
        description="remote source",
    )
    decision = resolve_seam_transfer_decision(policy="publish-to-s3")
    plan = OperatorTransferPlan(
        mode="manual",
        items=(item,),
        decision=decision,
        operator_initiated=True,
        authority_reference=None,
        authority_digest=None,
        override_destination_prefix=None,
        dry_run=False,
    )
    _monkeypatch_subprocess_run(
        monkeypatch,
        [
            subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr='ERROR "ls s3://source-bucket/key.tar.lz4": no object found\n'
            )
        ],
    )
    monkeypatch.setattr(
        "bspp.orchestration.runtime.data_movement.operator_transfer.require_tool",
        _make_require_tool_fn(),
    )
    with pytest.raises(OperatorTransferError, match="remote source size mismatch or missing"):
        execute_operator_transfer(
            plan,
            transfer_fn_map={"s5cmd": _ok_transfer},
            ls_fn=None,
            credentials=_make_s3_creds(),
        )


# ---------------------------------------------------------------------------
# Tests 42-43: bool rejection for size_bytes
# ---------------------------------------------------------------------------


def test_operator_transfer_item_rejects_bool_size() -> None:
    with pytest.raises(ValueError, match="non-negative int"):
        OperatorTransferItem(
            source="/tmp/a",
            destination="s3://bucket/key",
            size_bytes=True,  # type: ignore[arg-type]
            sha256=_SHA256_HEX,
            artifact_set_id=None,
            artifact_location_id=None,
            source_kind="explicit-path",
            description="test",
        )


def test_operator_transfer_evidence_rejects_bool_verified_size() -> None:
    item = OperatorTransferItem(
        source="/tmp/a",
        destination="s3://bucket/key",
        size_bytes=64,
        sha256=_SHA256_HEX,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="explicit-path",
        description="test",
    )
    nonce = "abcdef0123456789"
    evidence_id = operator_transfer_evidence_id(
        {
            "source": item.source,
            "destination": item.destination,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "nonce": nonce,
        }
    )
    with pytest.raises(ValueError, match="non-negative int"):
        OperatorTransferEvidence(
            evidence_id=evidence_id,
            mode="manual",
            operator_initiated=True,
            authority_reference=None,
            authority_digest=None,
            override_destination_prefix=None,
            item=item,
            transfer_tool="s5cmd",
            transfer_argv=("s5cmd", "cp"),
            transfer_returncode=0,
            transfer_elapsed_s=0.01,
            verified_size_bytes=True,  # type: ignore[arg-type]
            verified_sha256=_SHA256_HEX,
            transferred_at="2026-01-01T00:00:00.000000Z",
            original_evidence_preserved=False,
            evidence_nonce=nonce,
        )


# ---------------------------------------------------------------------------
# Test 44: schema_version required explicit
# ---------------------------------------------------------------------------


def test_schema_version_required_explicit() -> None:
    # OperatorTransferItem with schema_version=None
    with pytest.raises(ValueError, match="schema_version must be declared explicitly"):
        OperatorTransferItem(
            source="/tmp/a",
            destination="s3://bucket/key",
            size_bytes=64,
            sha256=_SHA256_HEX,
            artifact_set_id=None,
            artifact_location_id=None,
            source_kind="explicit-path",
            description="test",
            schema_version=None,  # type: ignore[arg-type]
        )

    # OperatorTransferPlan with schema_version=None
    item = OperatorTransferItem(
        source="/tmp/a",
        destination="s3://bucket/key",
        size_bytes=64,
        sha256=_SHA256_HEX,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="explicit-path",
        description="test",
    )
    decision = SeamTransferDecision(
        policy="publish-to-s3",
        object_storage_leg=True,
        tool="s5cmd",
        note="s5cmd cp against the S3 endpoint",
    )
    with pytest.raises(ValueError, match="schema_version must be declared explicitly"):
        OperatorTransferPlan(
            mode="manual",
            items=(item,),
            decision=decision,
            operator_initiated=True,
            authority_reference=None,
            authority_digest=None,
            override_destination_prefix=None,
            dry_run=False,
            schema_version=None,  # type: ignore[arg-type]
        )

    # OperatorTransferEvidence with schema_version=None
    nonce = "abcdef0123456789"
    evidence_id = operator_transfer_evidence_id(
        {
            "source": item.source,
            "destination": item.destination,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "nonce": nonce,
        }
    )
    with pytest.raises(ValueError, match="schema_version must be declared explicitly"):
        OperatorTransferEvidence(
            evidence_id=evidence_id,
            mode="manual",
            operator_initiated=True,
            authority_reference=None,
            authority_digest=None,
            override_destination_prefix=None,
            item=item,
            transfer_tool="s5cmd",
            transfer_argv=("s5cmd", "cp"),
            transfer_returncode=0,
            transfer_elapsed_s=0.01,
            verified_size_bytes=64,
            verified_sha256=_SHA256_HEX,
            transferred_at="2026-01-01T00:00:00.000000Z",
            original_evidence_preserved=False,
            evidence_nonce=nonce,
            schema_version=None,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# B-1: original_evidence_preserved semantics
# ---------------------------------------------------------------------------


def test_b1_prior_evidence_exists_but_remote_object_absent(tmp_path: Path) -> None:
    """B-1(a): prior evidence exists but remote object absent -> original_evidence_preserved=True."""
    local_file = tmp_path / "test.tar.lz4"
    sha = _write_local_file(local_file)
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=sha,
    )
    evidence_dir = tmp_path / "evidence"
    absent_ls = _make_fake_ls_fn(1, stderr='ERROR "ls": no object found\n')
    fake_verify = _make_verify_fn(64, sha)

    # First transfer: succeeds, writes evidence
    evidence1 = execute_operator_transfer(
        plan,
        evidence_dir=evidence_dir,
        transfer_fn_map={"s5cmd": _ok_transfer},
        verify_fn=fake_verify,
        ls_fn=absent_ls,
    )
    assert len(evidence1) == 1
    assert evidence1[0].original_evidence_preserved is False

    # Second transfer with --force: remote object is absent (ls reports absent),
    # but a prior evidence file exists. original_evidence_preserved must be True.
    evidence2 = execute_operator_transfer(
        plan,
        force=True,
        evidence_dir=evidence_dir,
        transfer_fn_map={"s5cmd": _ok_transfer},
        verify_fn=fake_verify,
        ls_fn=absent_ls,
    )
    assert len(evidence2) == 1
    assert evidence2[0].original_evidence_preserved is True


def test_b1_remote_object_present_but_no_prior_evidence(tmp_path: Path) -> None:
    """B-1(b): remote object present but no prior evidence -> original_evidence_preserved=False."""
    local_file = tmp_path / "test.tar.lz4"
    sha = _write_local_file(local_file)
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=sha,
    )
    evidence_dir = tmp_path / "evidence"
    # Destination already has an object with matching size
    existing_ls = _make_fake_ls_fn(0, stdout="2026/06/09 01:42:00         64  s3://bucket/prefix/test.tar.lz4\n")
    fake_verify = _make_verify_fn(64, sha)

    # Transfer with --force: remote object exists, but no prior evidence file.
    # original_evidence_preserved must be False.
    evidence = execute_operator_transfer(
        plan,
        force=True,
        evidence_dir=evidence_dir,
        transfer_fn_map={"s5cmd": _ok_transfer},
        verify_fn=fake_verify,
        ls_fn=existing_ls,
    )
    assert len(evidence) == 1
    assert evidence[0].original_evidence_preserved is False


# ---------------------------------------------------------------------------
# B-2: manual --destination full object key validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_dest",
    [
        "s3://",
        "s3:///",
        "s3://bucket",
        "s3://bucket/",
        "s3://bucket/prefix/",
        "s3://bucket/prefix//",
    ],
)
def test_b2_manual_destination_rejects_invalid_shapes(bad_dest: str) -> None:
    with pytest.raises(ValueError):
        OperatorTransferItem(
            source="/tmp/a",
            destination=bad_dest,
            size_bytes=64,
            sha256=_SHA256_HEX,
            artifact_set_id=None,
            artifact_location_id=None,
            source_kind="explicit-path",
            description="test",
        )


def test_b2_manual_destination_accepts_full_object_key() -> None:
    item = OperatorTransferItem(
        source="/tmp/a",
        destination="s3://bucket/prefix/file.tar.lz4",
        size_bytes=64,
        sha256=_SHA256_HEX,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="explicit-path",
        description="test",
    )
    assert item.destination == "s3://bucket/prefix/file.tar.lz4"


# ---------------------------------------------------------------------------
# NB-2: verify_operator_transfer and _default_verify_fn tests
# ---------------------------------------------------------------------------


def test_nb2_verify_operator_transfer_with_injected_verify_fn() -> None:
    """verify_operator_transfer returns (size, sha) from injected verify_fn."""
    item = OperatorTransferItem(
        source="/tmp/a",
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=_SHA256_HEX,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="explicit-path",
        description="test",
    )
    fake_verify = _make_verify_fn(64, _SHA256_HEX)
    result = verify_operator_transfer(item, verify_fn=fake_verify)
    assert result == (64, _SHA256_HEX)


def test_nb2_verify_operator_transfer_mismatch_raises() -> None:
    """verify_operator_transfer raises on size mismatch."""
    item = OperatorTransferItem(
        source="/tmp/a",
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=_SHA256_HEX,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="explicit-path",
        description="test",
    )
    bad_verify = _make_verify_fn(128, _SHA256_HEX)
    with pytest.raises(OperatorTransferError, match="verify size mismatch"):
        verify_operator_transfer(item, verify_fn=bad_verify)


def test_nb2_default_verify_fn_calls_s3_transfer_cp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """_default_verify_fn downloads via s3_transfer.cp and hashes locally."""
    import bspp.orchestration.runtime.data_movement.s3.transfer as s3_transfer_mod

    # Create a temp file that will be the "downloaded" object
    content = b"hello world" * 10
    expected_sha = hashlib.sha256(content).hexdigest()

    captured: list[str] = []

    def fake_cp(src: str, dst: str, **kwargs: object) -> TransferResult:
        captured.append(src)
        Path(dst).write_bytes(content)
        return TransferResult(
            tool="s5cmd",
            argv=("s5cmd", "cp", src, dst),
            returncode=0,
            elapsed_s=0.01,
        )

    monkeypatch.setattr(s3_transfer_mod, "cp", fake_cp)

    temp_path = tmp_path / "verify.tmp"
    size, sha = _default_verify_fn("s3://bucket/key", temp_path, credentials=_make_s3_creds())
    assert size == len(content)
    assert sha == expected_sha
    assert captured == ["s3://bucket/key"]


def test_nb2_default_verify_fn_raises_on_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """_default_verify_fn raises on dry-run (PlannedTransfer) result."""
    import bspp.orchestration.runtime.data_movement.s3.transfer as s3_transfer_mod
    from bspp.orchestration.runtime.data_movement.common import PlannedTransfer

    def fake_cp(src: str, dst: str, **kwargs: object) -> PlannedTransfer:
        return PlannedTransfer(
            tool="s5cmd",
            argv=("s5cmd", "cp", src, dst),
            note="dry-run",
        )

    monkeypatch.setattr(s3_transfer_mod, "cp", fake_cp)

    with pytest.raises(OperatorTransferError, match="dry-run plan"):
        _default_verify_fn("s3://bucket/key", tmp_path / "verify.tmp", credentials=_make_s3_creds())


def test_nb2_runtime_operator_verify_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime 'operator verify' CLI command re-verifies a remote object."""
    from bspp.orchestration.runtime import cli as runtime_cli

    # Build an evidence file
    plan = _make_manual_plan(dry_run=False)
    item = plan.items[0]
    nonce = "abcdef0123456789"
    evidence_id = operator_transfer_evidence_id(
        {
            "source": item.source,
            "destination": item.destination,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "nonce": nonce,
        }
    )
    evidence = OperatorTransferEvidence(
        evidence_id=evidence_id,
        mode="manual",
        operator_initiated=True,
        authority_reference=None,
        authority_digest=None,
        override_destination_prefix=None,
        item=item,
        transfer_tool="s5cmd",
        transfer_argv=("s5cmd", "cp", item.source, item.destination),
        transfer_returncode=0,
        transfer_elapsed_s=0.01,
        verified_size_bytes=item.size_bytes,
        verified_sha256=item.sha256,
        transferred_at="2026-01-01T00:00:00.000000Z",
        original_evidence_preserved=False,
        evidence_nonce=nonce,
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(evidence.to_json())

    # Monkeypatch verify_operator_transfer to avoid real download
    fake_verify_fn = _make_verify_fn(item.size_bytes, item.sha256)

    def fake_verify_operator_transfer(
        ev_item: OperatorTransferItem,
        *,
        credentials: S3Credentials | None = None,
        verify_fn: object = None,
    ) -> tuple[int, str]:
        return fake_verify_fn(ev_item.destination, Path(tempfile.mkdtemp()) / "verify.tmp")

    monkeypatch.setattr(
        "bspp.orchestration.runtime.data_movement.operator_transfer.verify_operator_transfer",
        fake_verify_operator_transfer,
    )

    runner = CliRunner()
    result = runner.invoke(
        runtime_cli.cli,
        ["data-movement", "operator", "verify", "--evidence", str(evidence_path)],
    )
    assert result.exit_code == 0, result.output
    import json as _json

    payload = _json.loads(result.output)
    assert payload["evidence_id"] == evidence_id
    assert payload["verified_size_bytes"] == item.size_bytes
    assert payload["verified_sha256"] == item.sha256
    assert payload["ok"] is True


# ---------------------------------------------------------------------------
# NB-3: _validate_timestamp rejects impossible dates
# ---------------------------------------------------------------------------


def test_nb3_validate_timestamp_rejects_impossible_dates() -> None:
    """OperatorTransferEvidence rejects impossible dates like 2026-13-99T99:99:99Z."""
    item = OperatorTransferItem(
        source="/tmp/a",
        destination="s3://bucket/key",
        size_bytes=64,
        sha256=_SHA256_HEX,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="explicit-path",
        description="test",
    )
    nonce = "abcdef0123456789"
    evidence_id = operator_transfer_evidence_id(
        {
            "source": item.source,
            "destination": item.destination,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "nonce": nonce,
        }
    )
    with pytest.raises(ValueError, match="must be a valid UTC timestamp"):
        OperatorTransferEvidence(
            evidence_id=evidence_id,
            mode="manual",
            operator_initiated=True,
            authority_reference=None,
            authority_digest=None,
            override_destination_prefix=None,
            item=item,
            transfer_tool="s5cmd",
            transfer_argv=("s5cmd", "cp"),
            transfer_returncode=0,
            transfer_elapsed_s=0.01,
            verified_size_bytes=64,
            verified_sha256=_SHA256_HEX,
            transferred_at="2026-13-99T99:99:99Z",
            original_evidence_preserved=False,
            evidence_nonce=nonce,
        )


# ---------------------------------------------------------------------------
# NB-7: _remote_object_exists wraps ValueError as OperatorTransferError
# ---------------------------------------------------------------------------


def test_nb7_remote_object_exists_wraps_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """_remote_object_exists wraps ValueError from _parse_object_ls_rows as OperatorTransferError."""
    # Monkeypatch subprocess.run to return rc=0 with garbage output
    _monkeypatch_subprocess_run(
        monkeypatch,
        [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="garbage line that cannot be parsed at all because it has too few fields",
                stderr="",
            )
        ],
    )
    monkeypatch.setattr(
        "bspp.orchestration.runtime.data_movement.operator_transfer.require_tool",
        _make_require_tool_fn(),
    )
    with pytest.raises(OperatorTransferError, match="unparseable s5cmd ls"):
        _remote_object_exists("s3://bucket/key", credentials=_make_s3_creds(), ls_fn=None)


# ---------------------------------------------------------------------------
# NB-7: OperatorTransferEvidence requires operator_initiated is True
# ---------------------------------------------------------------------------


def test_nb7_evidence_requires_operator_initiated_true() -> None:
    """OperatorTransferEvidence.__post_init__ must require operator_initiated is True."""
    item = OperatorTransferItem(
        source="/tmp/a",
        destination="s3://bucket/key",
        size_bytes=64,
        sha256=_SHA256_HEX,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="explicit-path",
        description="test",
    )
    nonce = "abcdef0123456789"
    evidence_id = operator_transfer_evidence_id(
        {
            "source": item.source,
            "destination": item.destination,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "nonce": nonce,
        }
    )
    with pytest.raises(ValueError, match="operator_initiated must be True"):
        OperatorTransferEvidence(
            evidence_id=evidence_id,
            mode="manual",
            operator_initiated=False,
            authority_reference=None,
            authority_digest=None,
            override_destination_prefix=None,
            item=item,
            transfer_tool="s5cmd",
            transfer_argv=("s5cmd", "cp"),
            transfer_returncode=0,
            transfer_elapsed_s=0.01,
            verified_size_bytes=64,
            verified_sha256=_SHA256_HEX,
            transferred_at="2026-01-01T00:00:00.000000Z",
            original_evidence_preserved=False,
            evidence_nonce=nonce,
        )


# ---------------------------------------------------------------------------
# NB-5: --s3-prefix accepted in manual mode
# ---------------------------------------------------------------------------


def test_nb5_s3_prefix_accepted_in_manual_mode(tmp_path: Path) -> None:
    """Control CLI 'data plan' in manual mode accepts --s3-prefix without --phase-plan."""
    from bspp.orchestration.control.cli import cli as control_cli

    runner = CliRunner()
    result = runner.invoke(
        control_cli,
        [
            "data",
            "plan",
            "--source",
            str(tmp_path / "test.tar.lz4"),
            "--destination",
            "s3://bucket/prefix/test.tar.lz4",
            "--size-bytes",
            "64",
            "--sha256",
            _SHA256_HEX,
            "--s3-prefix",
            "s3://some/prefix/",
            "--no-dry-run",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# NB-6: runtime CLI uses *_from_json helpers (non-mapping top-level JSON)
# ---------------------------------------------------------------------------


def test_nb6_runtime_execute_rejects_non_mapping_json(tmp_path: Path) -> None:
    """Runtime 'operator execute' surfaces a clean error for a non-mapping JSON top-level."""
    from bspp.orchestration.runtime import cli as runtime_cli

    plan_json = tmp_path / "plan.json"
    plan_json.write_text("[1, 2, 3]")

    runner = CliRunner()
    result = runner.invoke(
        runtime_cli.cli,
        ["data-movement", "operator", "execute", "--plan-json", str(plan_json)],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, click.ClickException) or result.exit_code != 0


def test_nb6_runtime_verify_rejects_non_mapping_json(tmp_path: Path) -> None:
    """Runtime 'operator verify' surfaces a clean error for a non-mapping JSON top-level."""
    from bspp.orchestration.runtime import cli as runtime_cli

    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text("[1, 2, 3]")

    runner = CliRunner()
    result = runner.invoke(
        runtime_cli.cli,
        ["data-movement", "operator", "verify", "--evidence", str(evidence_path)],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# NB-4: control CLI catches yaml.YAMLError
# ---------------------------------------------------------------------------


def test_nb4_control_cli_catches_yaml_error(tmp_path: Path) -> None:
    """Control 'data plan' converts yaml.YAMLError to ClickException (no traceback)."""
    from bspp.orchestration.control.cli import cli as control_cli

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("\tinvalid: yaml: with: tabs\n  - [unclosed")

    runner = CliRunner()
    result = runner.invoke(
        control_cli,
        [
            "data",
            "plan",
            "--phase-plan",
            str(bad_yaml),
            "--s3-prefix",
            "s3://test/",
        ],
    )
    assert result.exit_code != 0
    # Should be a ClickException (handled as SystemExit by Click), not a raw yaml.YAMLError traceback
    import yaml as _yaml

    assert not isinstance(result.exception, _yaml.YAMLError)


# ---------------------------------------------------------------------------
# Greptile review fixes
# ---------------------------------------------------------------------------


def test_greptile_evidence_contradiction_rejected() -> None:
    """An evidence record whose verified content contradicts its item is rejected."""
    plan = _make_manual_plan(
        source="/tmp/a.tar.lz4",
        destination="s3://bucket/prefix/a.tar.lz4",
        size_bytes=64,
        sha256=_SHA256_HEX,
    )
    item = plan.items[0]
    nonce = "feedfacefeedface"
    evidence_id = operator_transfer_evidence_id(
        {
            "source": item.source,
            "destination": item.destination,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "nonce": nonce,
        }
    )
    with pytest.raises(ValueError, match=r"verified_size_bytes must match item\.size_bytes"):
        OperatorTransferEvidence(
            evidence_id=evidence_id,
            mode="manual",
            operator_initiated=True,
            authority_reference=None,
            authority_digest=None,
            override_destination_prefix=None,
            item=item,
            transfer_tool="s5cmd",
            transfer_argv=("s5cmd", "cp"),
            transfer_returncode=0,
            transfer_elapsed_s=0.01,
            verified_size_bytes=item.size_bytes + 1,
            verified_sha256=item.sha256,
            transferred_at="2026-01-01T00:00:00.000000Z",
            original_evidence_preserved=False,
            evidence_nonce=nonce,
        )
    other_sha = "f" * 64
    with pytest.raises(ValueError, match=r"verified_sha256 must match item\.sha256"):
        OperatorTransferEvidence(
            evidence_id=evidence_id,
            mode="manual",
            operator_initiated=True,
            authority_reference=None,
            authority_digest=None,
            override_destination_prefix=None,
            item=item,
            transfer_tool="s5cmd",
            transfer_argv=("s5cmd", "cp"),
            transfer_returncode=0,
            transfer_elapsed_s=0.01,
            verified_size_bytes=item.size_bytes,
            verified_sha256=other_sha,
            transferred_at="2026-01-01T00:00:00.000000Z",
            original_evidence_preserved=False,
            evidence_nonce=nonce,
        )


def test_greptile_authority_digest_bound() -> None:
    """Plan-referenced plans bind the phase-plan content digest; manual plans do not."""
    referenced = build_plan_referenced_transfer_plan(
        phase_plan_path=LOCAL_PLAN,
        s3_prefix=_S3_PREFIX,
        dry_run=False,
    )
    from bspp.orchestration.contract.phase import folding_phase_plan_from_mapping

    data = yaml.safe_load(LOCAL_PLAN.read_bytes())
    phase_plan = folding_phase_plan_from_mapping(data)
    assert referenced.authority_digest == phase_plan.digest
    assert referenced.authority_digest is not None

    manual = _make_manual_plan()
    assert manual.authority_digest is None

    # Round-trip preserves the digest.
    rt = operator_transfer_plan_from_mapping(referenced.to_mapping())
    assert rt.authority_digest == phase_plan.digest


# ---------------------------------------------------------------------------
# Greptile round-2 fixes: source snapshot + destination lock
# ---------------------------------------------------------------------------


def test_local_source_snapshot_closes_toctou(tmp_path: Path) -> None:
    """A concurrent writer mutating the source after the snapshot cannot affect the upload."""
    local_file = tmp_path / "source.tar.lz4"
    content = b"x" * 64
    local_file.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=sha,
    )
    captured: dict[str, object] = {}

    def mutating_transfer(src: str, dst: str, **kwargs: object) -> TransferResult:
        # Simulate a concurrent writer mutating the ORIGINAL source after the
        # snapshot was taken; the transfer must still use the immutable snapshot.
        local_file.write_bytes(b"y" * 64)
        captured["src"] = src
        captured["snapshot_content"] = Path(src).read_bytes()
        return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=0, elapsed_s=0.0)

    evidence = execute_operator_transfer(
        plan,
        transfer_fn_map={"s5cmd": mutating_transfer},
        verify_fn=_make_verify_fn(64, sha),
        ls_fn=_make_fake_ls_fn(1, stderr='ERROR "ls": no object found\n'),
        lock_root=tmp_path / "locks",
    )
    assert len(evidence) == 1
    # The transfer used a snapshot (temp path), not the original source.
    assert captured["src"] != str(local_file)
    # The snapshot content is the ORIGINAL content, not the mutated original.
    assert captured["snapshot_content"] == content


def test_destination_lock_contention(tmp_path: Path) -> None:
    """A second concurrent execution targeting the same destination fails closed."""
    local_file = tmp_path / "source.tar.lz4"
    sha = _write_local_file(local_file)
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=sha,
    )
    lock_root = tmp_path / "locks"

    # Simulate a concurrent writer already holding the destination lock.
    fd, _ = _acquire_destination_lock("s3://bucket/prefix/test.tar.lz4", lock_root)
    try:
        with pytest.raises(OperatorTransferError, match="concurrent transfer in progress"):
            execute_operator_transfer(
                plan,
                transfer_fn_map={"s5cmd": _ok_transfer},
                verify_fn=_make_verify_fn(64, sha),
                ls_fn=_make_fake_ls_fn(1, stderr='ERROR "ls": no object found\n'),
                lock_root=lock_root,
            )
    finally:
        _release_destination_lock(fd)


def test_destination_lock_released_after_success(tmp_path: Path) -> None:
    """After a successful transfer the destination lock is released."""
    local_file = tmp_path / "source.tar.lz4"
    sha = _write_local_file(local_file)
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=sha,
    )
    lock_root = tmp_path / "locks"

    evidence = execute_operator_transfer(
        plan,
        transfer_fn_map={"s5cmd": _ok_transfer},
        verify_fn=_make_verify_fn(64, sha),
        ls_fn=_make_fake_ls_fn(1, stderr='ERROR "ls": no object found\n'),
        lock_root=lock_root,
    )
    assert len(evidence) == 1
    # The lock is released: a fresh acquisition on the same destination succeeds.
    fd, _ = _acquire_destination_lock("s3://bucket/prefix/test.tar.lz4", lock_root)
    _release_destination_lock(fd)


def test_snapshot_root_honored(tmp_path: Path) -> None:
    """The source snapshot + verify download are staged under --snapshot-root."""
    local_file = tmp_path / "source.tar.lz4"
    sha = _write_local_file(local_file)
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=sha,
    )
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    captured: dict[str, object] = {}

    def recording_transfer(src: str, dst: str, **kwargs: object) -> TransferResult:
        captured["src"] = src
        return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=0, elapsed_s=0.0)

    execute_operator_transfer(
        plan,
        transfer_fn_map={"s5cmd": recording_transfer},
        verify_fn=_make_verify_fn(64, sha),
        ls_fn=_make_fake_ls_fn(1, stderr='ERROR "ls": no object found\n'),
        lock_root=tmp_path / "locks",
        snapshot_root=snapshot_root,
    )
    assert str(snapshot_root) in str(captured["src"])


def test_snapshot_root_defaults_to_slurm_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --snapshot-root, $SLURM_TMPDIR is used as the staging root."""
    slurm_tmp = tmp_path / "slurm-tmp"
    slurm_tmp.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(slurm_tmp))
    local_file = tmp_path / "source.tar.lz4"
    sha = _write_local_file(local_file)
    plan = _make_manual_plan(
        source=str(local_file),
        destination="s3://bucket/prefix/test.tar.lz4",
        size_bytes=64,
        sha256=sha,
    )
    captured: dict[str, object] = {}

    def recording_transfer(src: str, dst: str, **kwargs: object) -> TransferResult:
        captured["src"] = src
        return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=0, elapsed_s=0.0)

    execute_operator_transfer(
        plan,
        transfer_fn_map={"s5cmd": recording_transfer},
        verify_fn=_make_verify_fn(64, sha),
        ls_fn=_make_fake_ls_fn(1, stderr='ERROR "ls": no object found\n'),
        lock_root=tmp_path / "locks",
    )
    assert str(slurm_tmp) in str(captured["src"])
