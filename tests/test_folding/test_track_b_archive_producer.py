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

"""Archive-producer tests for the folding companion-engine track (e05s04)."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_archive import (
    ArchivePlanOptions,
)
from bspp.orchestration.contract.folding_index import FoldingIndexRecord, make_folding_index
from bspp.orchestration.contract.prediction_bundle import prediction_archive_bundle_from_mapping
from bspp.orchestration.runtime.folding.archive import plan_folding_archives
from bspp.orchestration.runtime.folding.execution.archive_producer import (
    ArchiveCommandResult,
    compute_sha256,
    derive_archive_run_tag,
    execute_archive_batch,
    produce_archives,
)
from bspp.orchestration.runtime.folding.results import scan_folding_results


def _record(ordinal: int, protein_id: str, *, length: int = 2) -> FoldingIndexRecord:
    return FoldingIndexRecord(
        source_ordinal=ordinal,
        protein_id=protein_id,
        msa_path=f"/msa/{protein_id}.a3m",
        query_sequence="A" * length,
        sequence_length=length,
        chain_lengths=(length,),
        chain_count=1,
        chain_cardinalities=(1,),
        msa_depth=1,
        total_length=length,
    )


def _write(path: Path, text: str = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _complete_inventory(tmp_path: Path, count: int = 5):
    index = make_folding_index(tuple(_record(ordinal, f"model-{ordinal}") for ordinal in range(count)))
    root = tmp_path / "results"
    for ordinal in range(count):
        _write(root / f"model-{ordinal}_unrelaxed_rank_001_model_1.pdb")
        _write(root / f"model-{ordinal}_scores_rank_001_model_1.json", "{}")
    return scan_folding_results(index, root)


class _FakeRunner:
    def __init__(
        self,
        payload: bytes = b"fake-lz4-bytes",
        *,
        returncode: int = 0,
        stderr: str = "",
    ) -> None:
        self.payload = payload
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...], Path]] = []

    def __call__(
        self,
        tar_argv: Sequence[str],
        lz4_argv: Sequence[str],
        *,
        stdout_path: Path,
    ) -> ArchiveCommandResult:
        self.calls.append((tuple(tar_argv), tuple(lz4_argv), stdout_path))
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_bytes(self.payload)
        return ArchiveCommandResult(returncode=self.returncode, stderr=self.stderr)


def test_derive_archive_run_tag_format_and_utc_determinism() -> None:
    utc = datetime(2026, 9, 9, 14, 30, tzinfo=UTC)
    assert derive_archive_run_tag(utc, "a") == "bspp_260909_1430_a"
    assert derive_archive_run_tag(utc, "a") == derive_archive_run_tag(utc, "a")
    assert derive_archive_run_tag(utc, "b") != derive_archive_run_tag(utc, "a")
    assert derive_archive_run_tag(datetime(2026, 9, 9, 14, 31, tzinfo=UTC), "a") != derive_archive_run_tag(utc, "a")
    # Naive datetimes are accepted and interpreted as UTC.
    assert derive_archive_run_tag(datetime(2026, 9, 9, 14, 30), "a") == "bspp_260909_1430_a"

    for invalid in ("A", "ab", "", "1"):
        with pytest.raises(ValueError, match="lowercase ASCII letter"):
            derive_archive_run_tag(utc, invalid)
    with pytest.raises(ValueError, match="must be UTC"):
        derive_archive_run_tag(datetime(2026, 9, 9, 14, 30, tzinfo=timezone(timedelta(hours=1))), "a")


def test_compute_sha256_correctness(tmp_path: Path) -> None:
    abc = tmp_path / "abc"
    abc.write_bytes(b"abc")
    assert compute_sha256(abc) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert compute_sha256(empty) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_execute_archive_batch_stages_and_verifies(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path, count=1)
    options = ArchivePlanOptions(
        run_tag="bspp_260909_1430_a",
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=False,
    )
    batch = plan_folding_archives(inventory, options).batches[0]

    payload = b"fake-lz4-bytes"
    runner = _FakeRunner(payload=payload)
    bundle = execute_archive_batch(batch, runner=runner)

    stage_dir = Path(batch.stage_dir)
    for member in batch.members:
        assert (stage_dir / member.stage_name).read_bytes() == Path(member.source_path).read_bytes()

    assert len(runner.calls) == 1
    recorded_tar, recorded_lz4, recorded_stdout = runner.calls[0]
    assert recorded_tar == batch.tar_argv
    assert recorded_lz4 == batch.lz4_argv
    # The pipeline writes to a fresh same-directory temporary path that is
    # atomically renamed onto the planner output after verification.
    assert recorded_stdout != Path(batch.stdout_path)
    assert recorded_stdout.parent == Path(batch.stdout_path).parent
    archive_path = Path(batch.stdout_path)
    assert archive_path.read_bytes() == payload

    canonical_path = archive_path.parent / bundle.bundle_name
    assert canonical_path.exists()
    assert canonical_path.read_bytes() == payload
    assert archive_path.exists()

    assert bundle.bundle_name == "bspp_260909_1430_a00000.tar.lz4"
    assert bundle.sha256 == compute_sha256(canonical_path)
    assert bundle.sha256 == compute_sha256(archive_path)
    assert bundle.size_bytes == len(payload) > 0
    assert bundle.member_ids == tuple(member.stage_name for member in batch.members)
    assert bundle.member_count == len(bundle.member_ids)
    assert bundle.created_at is None


def test_execute_archive_batch_accepts_exact_existing_alias(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path, count=1)
    options = ArchivePlanOptions(
        run_tag="bspp_260909_1430_a",
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=False,
    )
    batch = plan_folding_archives(inventory, options).batches[0]

    payload = b"fake-lz4-bytes"
    first = execute_archive_batch(batch, runner=_FakeRunner(payload=payload))
    second = execute_archive_batch(batch, runner=_FakeRunner(payload=payload))

    assert second == first
    assert second.sha256 == first.sha256
    assert second.size_bytes == first.size_bytes


def test_execute_archive_batch_rejects_conflicting_alias(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path, count=1)
    options = ArchivePlanOptions(
        run_tag="bspp_260909_1430_a",
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=False,
    )
    batch = plan_folding_archives(inventory, options).batches[0]

    archive_root = Path(batch.stdout_path).parent
    archive_root.mkdir(parents=True, exist_ok=True)
    canonical_path = archive_root / "bspp_260909_1430_a00000.tar.lz4"
    canonical_path.write_bytes(b"conflicting-bytes")

    runner = _FakeRunner(payload=b"fake-lz4-bytes")
    with pytest.raises(ValueError, match="not hard-linked"):
        execute_archive_batch(batch, runner=runner)

    # The foreign alias is failed closed before the pipeline runs and its bytes
    # are untouched.
    assert runner.calls == []
    assert canonical_path.read_bytes() == b"conflicting-bytes"


def test_same_attempt_restart_reuses_verified_bundle_unchanged(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path, count=1)
    options = ArchivePlanOptions(
        run_tag="bspp_260909_1430_a",
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=False,
    )
    batch = plan_folding_archives(inventory, options).batches[0]

    payload = b"fake-lz4-bytes"
    first = execute_archive_batch(batch, runner=_FakeRunner(payload=payload))

    archive_path = Path(batch.stdout_path)
    canonical_path = archive_path.parent / first.bundle_name
    assert os.path.samefile(archive_path, canonical_path)
    verified_bytes = canonical_path.read_bytes()

    # Same-attempt restart with a runner that would produce different bytes:
    # the verified bundle is reloaded and reused unchanged, and the archive
    # pipeline is never run into the verified evidence.
    second_runner = _FakeRunner(payload=b"different-bytes")
    second = execute_archive_batch(batch, runner=second_runner)

    assert second == first
    assert second_runner.calls == []
    assert canonical_path.read_bytes() == verified_bytes
    assert archive_path.read_bytes() == verified_bytes
    assert os.path.samefile(archive_path, canonical_path)


def test_restart_with_unlinked_planner_output_fails_closed(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path, count=1)
    options = ArchivePlanOptions(
        run_tag="bspp_260909_1430_a",
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=False,
    )
    batch = plan_folding_archives(inventory, options).batches[0]

    payload = b"fake-lz4-bytes"
    bundle = execute_archive_batch(batch, runner=_FakeRunner(payload=payload))

    # Tamper: replace the planner output with an unlinked different file, so
    # the verified alias loses its ownership link to the planner output.
    archive_path = Path(batch.stdout_path)
    canonical_path = archive_path.parent / bundle.bundle_name
    archive_path.unlink()
    archive_path.write_bytes(b"tampered-bytes")

    runner = _FakeRunner(payload=payload)
    with pytest.raises(ValueError, match="not hard-linked"):
        execute_archive_batch(batch, runner=runner)

    assert runner.calls == []
    assert canonical_path.read_bytes() == payload
    assert archive_path.read_bytes() == b"tampered-bytes"


def test_restart_after_crash_before_alias_reproduces_atomically(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path, count=1)
    options = ArchivePlanOptions(
        run_tag="bspp_260909_1430_a",
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=False,
    )
    batch = plan_folding_archives(inventory, options).batches[0]

    # Simulate a crash between archive publication and alias creation: the
    # planner output exists but is not linked to any canonical alias.
    archive_path = Path(batch.stdout_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(b"stale-unverified-bytes")

    payload = b"fake-lz4-bytes"
    runner = _FakeRunner(payload=payload)
    bundle = execute_archive_batch(batch, runner=runner)

    assert len(runner.calls) == 1
    # The runner wrote to a fresh temporary path, not the stale planner output.
    assert runner.calls[0][2] != archive_path
    assert runner.calls[0][2].parent == archive_path.parent
    canonical_path = archive_path.parent / bundle.bundle_name
    assert archive_path.read_bytes() == payload
    assert canonical_path.read_bytes() == payload
    assert os.path.samefile(archive_path, canonical_path)
    # No temporary files remain in the archive directory.
    assert [entry.name for entry in archive_path.parent.iterdir() if entry.name.startswith("tmp")] == []


def test_prediction_archive_bundle_round_trip(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path, count=1)
    options = ArchivePlanOptions(
        run_tag="bspp_260909_1430_a",
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=False,
    )
    batch = plan_folding_archives(inventory, options).batches[0]
    bundle = execute_archive_batch(batch, runner=_FakeRunner())

    assert prediction_archive_bundle_from_mapping(bundle.to_mapping()) == bundle
    assert re.fullmatch(r"bspp_[0-9]{6}_[0-9]{4}_[a-z][0-9]{5}\.tar\.lz4", bundle.bundle_name)
    assert bundle.member_count == len(bundle.member_ids)
    assert re.fullmatch(r"[0-9a-f]{64}", bundle.sha256)
    assert bundle.size_bytes > 0


def test_produce_archives_end_to_end_attempt_owned_name(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path, count=5)
    options = ArchivePlanOptions(
        run_tag="placeholder",
        proteins_per_archive=2,
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=False,
    )
    runner = _FakeRunner()
    bundles = produce_archives(
        inventory,
        options,
        runner=runner,
        attempt_timestamp=datetime(2026, 9, 9, 14, 30, tzinfo=UTC),
        letter="a",
    )

    assert len(bundles) == 3
    assert all(bundle.bundle_name.startswith("bspp_260909_1430_a") for bundle in bundles)
    assert all(not bundle.bundle_name.startswith("placeholder") for bundle in bundles)

    plan = plan_folding_archives(inventory, replace(options, run_tag="bspp_260909_1430_a"))
    assert [call[0] for call in runner.calls] == [batch.tar_argv for batch in plan.batches]
    assert [call[1] for call in runner.calls] == [batch.lz4_argv for batch in plan.batches]


def test_execute_archive_batch_error_paths(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path, count=1)
    options = ArchivePlanOptions(
        run_tag="bspp_260909_1430_a",
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=False,
    )
    batch = plan_folding_archives(inventory, options).batches[0]

    with pytest.raises(RuntimeError, match="returncode 1"):
        execute_archive_batch(batch, runner=_FakeRunner(returncode=1, stderr="boom"))

    with pytest.raises(ValueError, match="empty archive"):
        execute_archive_batch(batch, runner=_FakeRunner(payload=b""))
