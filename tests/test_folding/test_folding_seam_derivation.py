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

"""Folding→postprocessing seam derivation tests.

Covers the deterministic mapping from a completed folding run's canonical-pair
index + canonical-pair action evidence to the postprocessing master parquet and
tracking parquet. The runtime derivation reuses ``build_master_parquet_row``,
``write_master_parquet``, and ``create_tracking_parquet`` unchanged; these tests
exercise the seam end to end (extending the intended chain proven by
``test_track_b_integration.py::test_archive_to_master_parquet_to_staging_consumer``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bspp.orchestration.contract.master_parquet_projection import MASTER_PARQUET_PROJECTION
from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.runtime.folding.seam_derivation import (
    SeamDerivationError,
    _destination_lock,
    derive_seam_parquets,
)

_COLABFOLD_TOOL = VALID_TOOL_USED[0]
_BIOIR_TOOL = "OpenFold2 (BioNeMo IR) / AlphaFold-Multimer"

_SEQ_SHA = "a" * 64


def _scores(*, plddt: list[int], ptm: float | None, iptm: float | None) -> dict[str, object]:
    n = len(plddt)
    return {
        "schema_version": 1,
        "plddt": plddt,
        "pae": [[0.0] * n for _ in range(n)],
        "max_pae": 3.0,
        "ptm": ptm,
        "iptm": iptm,
    }


def _pair(*, model_entity_id: str, tool_used: str, ptm: float | None, iptm: float | None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_entity_id": model_entity_id,
        "tool_used": tool_used,
        "structure_path": f"/data/{model_entity_id}-model_v1.pdb",
        "scores_path": f"/data/{model_entity_id}-meta_v1.json",
        "scores": _scores(plddt=[90, 85, 80], ptm=ptm, iptm=iptm),
    }


def _evidence_entry(
    *,
    target_id: str,
    model_entity_id: str,
    tool_used: str,
    ptm: float | None,
    iptm: float | None,
) -> dict[str, object]:
    return {
        "target_id": target_id,
        "sequence_sha256": _SEQ_SHA,
        "pair": _pair(model_entity_id=model_entity_id, tool_used=tool_used, ptm=ptm, iptm=iptm),
    }


def _index_entry(
    *,
    target_id: str,
    model_entity_id: str,
    tool_used: str,
) -> dict[str, object]:
    return {
        "target_id": target_id,
        "sequence_sha256": _SEQ_SHA,
        "model_entity_id": model_entity_id,
        "tool_used": tool_used,
        "structure_path": f"/data/{model_entity_id}-model_v1.pdb",
        "scores_path": f"/data/{model_entity_id}-meta_v1.json",
    }


def _write_index(tmp_path: Path, entries: list[dict[str, object]], run_id: str = "phase-run-abc123") -> Path:
    path = tmp_path / "canonical-pair-index.json"
    path.write_text(json.dumps({"schema_version": 1, "run_id": run_id, "entries": entries}), encoding="utf-8")
    return path


def _write_evidence(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    path = tmp_path / "canonical-pair-evidence.json"
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return path


def _derive(
    tmp_path: Path,
    *,
    index_entries: list[dict[str, object]],
    evidence_entries: list[dict[str, object]],
    **overrides: object,
) -> object:
    index_path = _write_index(tmp_path, index_entries)
    evidence_path = _write_evidence(tmp_path, evidence_entries)
    master_output = tmp_path / "master.parquet"
    tracking_output = tmp_path / "tracking.parquet"
    kwargs: dict[str, object] = {
        "index_path": index_path,
        "evidence_path": evidence_path,
        "master_output": master_output,
        "tracking_output": tracking_output,
        "s3_output_prefix": "s3://public-bucket/users/test-user/postproc/",
        "source_run": "n0010-postproc",
        "archive_name": "bspp_260917_1234_a00001.tar.lz4",
    }
    kwargs.update(overrides)
    return derive_seam_parquets(**kwargs)  # type: ignore[arg-type]


def test_happy_path_colabfold(tmp_path: Path) -> None:
    result = _derive(
        tmp_path,
        index_entries=[
            _index_entry(
                target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
            )
        ],
        evidence_entries=[
            _evidence_entry(
                target_id="AF-0000000000000001",
                model_entity_id="AF-0000000000000001",
                tool_used=_COLABFOLD_TOOL,
                ptm=0.9,
                iptm=0.8,
            )
        ],
    )

    master = pq.read_table(result.master_path)
    assert master.num_rows == 1
    assert master.column_names == [column.name for column in MASTER_PARQUET_PROJECTION]
    row = master.to_pylist()[0]
    assert row["msa_path"] == "AF-0000000000000001"
    assert row["source_run"] == "n0010-postproc"
    assert row["pdb_residue_count"] == 3
    assert row["mean_plddt"] == pytest.approx(85.0)
    assert row["plddt_above_70"] == pytest.approx(1.0)
    assert row["ptm"] == pytest.approx(0.9)
    assert row["iptm"] == pytest.approx(0.8)
    assert row["swiftstack_archive"] == "bspp_260917_1234_a00001.tar.lz4"
    assert row["uploaded_to_gcp"] == "no"

    tracking = pq.read_table(result.tracking_path)
    assert tracking.num_rows == 1
    tracking_row = tracking.to_pylist()[0]
    assert tracking_row["dataset_name"] == "n0010-postproc"
    assert tracking_row["postprocess_status"] == "pending"
    assert tracking_row["s3_destination"] == "s3://public-bucket/users/test-user/postproc/"


def test_bioir_null_ptm_iptm_are_resolved_nulls(tmp_path: Path) -> None:
    result = _derive(
        tmp_path,
        index_entries=[
            _index_entry(target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_BIOIR_TOOL)
        ],
        evidence_entries=[
            _evidence_entry(
                target_id="AF-0000000000000001",
                model_entity_id="AF-0000000000000001",
                tool_used=_BIOIR_TOOL,
                ptm=None,
                iptm=None,
            )
        ],
    )

    master = pq.read_table(result.master_path)
    row = master.to_pylist()[0]
    assert row["ptm"] is None
    assert row["iptm"] is None
    assert row["mean_plddt"] == pytest.approx(85.0)
    assert row["plddt_above_70"] == pytest.approx(1.0)


def test_multiple_entries_ordered_by_index(tmp_path: Path) -> None:
    first = _index_entry(
        target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
    )
    second = _index_entry(
        target_id="AF-0000000000000002", model_entity_id="AF-0000000000000002", tool_used=_COLABFOLD_TOOL
    )
    result = _derive(
        tmp_path,
        index_entries=[first, second],
        evidence_entries=[
            _evidence_entry(
                target_id="AF-0000000000000001",
                model_entity_id="AF-0000000000000001",
                tool_used=_COLABFOLD_TOOL,
                ptm=0.9,
                iptm=0.8,
            ),
            _evidence_entry(
                target_id="AF-0000000000000002",
                model_entity_id="AF-0000000000000002",
                tool_used=_COLABFOLD_TOOL,
                ptm=0.9,
                iptm=0.8,
            ),
        ],
    )

    master = pq.read_table(result.master_path)
    assert master.num_rows == 2
    assert [row["msa_path"] for row in master.to_pylist()] == [
        "AF-0000000000000001",
        "AF-0000000000000002",
    ]


def test_index_evidence_target_mismatch_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SeamDerivationError, match="target_id sets do not match"):
        _derive(
            tmp_path,
            index_entries=[
                _index_entry(
                    target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
                )
            ],
            evidence_entries=[
                _evidence_entry(
                    target_id="AF-0000000000000002",
                    model_entity_id="AF-0000000000000002",
                    tool_used=_COLABFOLD_TOOL,
                    ptm=0.9,
                    iptm=0.8,
                )
            ],
        )


def test_empty_index_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SeamDerivationError, match="no entries"):
        _derive(
            tmp_path,
            index_entries=[],
            evidence_entries=[
                _evidence_entry(
                    target_id="AF-0000000000000001",
                    model_entity_id="AF-0000000000000001",
                    tool_used=_COLABFOLD_TOOL,
                    ptm=0.9,
                    iptm=0.8,
                )
            ],
        )


def test_missing_index_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SeamDerivationError, match="cannot load canonical-pair index"):
        derive_seam_parquets(
            index_path=tmp_path / "missing-index.json",
            evidence_path=_write_evidence(
                tmp_path,
                [
                    _evidence_entry(
                        target_id="AF-0000000000000001",
                        model_entity_id="AF-0000000000000001",
                        tool_used=_COLABFOLD_TOOL,
                        ptm=0.9,
                        iptm=0.8,
                    )
                ],
            ),
            master_output=tmp_path / "master.parquet",
            tracking_output=tmp_path / "tracking.parquet",
            s3_output_prefix="s3://public-bucket/users/test-user/postproc/",
            source_run="n0010-postproc",
            archive_name="bspp_260917_1234_a00001.tar.lz4",
        )


def test_existing_output_without_force_fails_closed(tmp_path: Path) -> None:
    master_output = tmp_path / "master.parquet"
    master_output.write_text("existing", encoding="utf-8")
    with pytest.raises(SeamDerivationError, match="already exists"):
        _derive(
            tmp_path,
            index_entries=[
                _index_entry(
                    target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
                )
            ],
            evidence_entries=[
                _evidence_entry(
                    target_id="AF-0000000000000001",
                    model_entity_id="AF-0000000000000001",
                    tool_used=_COLABFOLD_TOOL,
                    ptm=0.9,
                    iptm=0.8,
                )
            ],
            master_output=master_output,
        )


def test_provenance_binding_run_id_mismatch_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SeamDerivationError, match="does not match the validated phase_run_id"):
        _derive(
            tmp_path,
            index_entries=[
                _index_entry(
                    target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
                )
            ],
            evidence_entries=[
                _evidence_entry(
                    target_id="AF-0000000000000001",
                    model_entity_id="AF-0000000000000001",
                    tool_used=_COLABFOLD_TOOL,
                    ptm=0.9,
                    iptm=0.8,
                )
            ],
            phase_run_id="phase-run-xyz789",
        )


def test_provenance_binding_matching_run_id_passes(tmp_path: Path) -> None:
    result = _derive(
        tmp_path,
        index_entries=[
            _index_entry(
                target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
            )
        ],
        evidence_entries=[
            _evidence_entry(
                target_id="AF-0000000000000001",
                model_entity_id="AF-0000000000000001",
                tool_used=_COLABFOLD_TOOL,
                ptm=0.9,
                iptm=0.8,
            )
        ],
        phase_run_id="phase-run-abc123",
    )
    assert result.row_count == 1


@pytest.mark.parametrize("field", ["tool_used", "structure_path", "scores_path"])
def test_pair_reconciliation_contradiction_fails_closed(tmp_path: Path, field: str) -> None:
    index_entry = _index_entry(
        target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
    )
    if field == "tool_used":
        index_entry["tool_used"] = _BIOIR_TOOL
    elif field == "structure_path":
        index_entry["structure_path"] = "/data/AF-0000000000000002-model_v1.pdb"
    else:
        index_entry["scores_path"] = "/data/AF-0000000000000002-meta_v1.json"
    with pytest.raises(SeamDerivationError, match=f"{field} differs"):
        _derive(
            tmp_path,
            index_entries=[index_entry],
            evidence_entries=[
                _evidence_entry(
                    target_id="AF-0000000000000001",
                    model_entity_id="AF-0000000000000001",
                    tool_used=_COLABFOLD_TOOL,
                    ptm=0.9,
                    iptm=0.8,
                )
            ],
        )


def test_invalid_s3_prefix_prevalidated_before_any_write(tmp_path: Path) -> None:
    master_output = tmp_path / "master.parquet"
    tracking_output = tmp_path / "tracking.parquet"
    with pytest.raises(SeamDerivationError, match="s3_output_prefix"):
        _derive(
            tmp_path,
            index_entries=[
                _index_entry(
                    target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
                )
            ],
            evidence_entries=[
                _evidence_entry(
                    target_id="AF-0000000000000001",
                    model_entity_id="AF-0000000000000001",
                    tool_used=_COLABFOLD_TOOL,
                    ptm=0.9,
                    iptm=0.8,
                )
            ],
            s3_output_prefix="not-a-uri",
            master_output=master_output,
            tracking_output=tracking_output,
        )
    assert not master_output.exists()
    assert not tracking_output.exists()


def test_partial_publish_rolls_back_on_commit_rename_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    master_output = tmp_path / "master.parquet"
    tracking_output = tmp_path / "tracking.parquet"
    real_replace = os.replace

    def flaky_replace(src: object, dst: object) -> None:
        if Path(dst) == master_output:
            raise OSError("simulated master rename failure")
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(SeamDerivationError, match="seam parquet derivation failed"):
        _derive(
            tmp_path,
            index_entries=[
                _index_entry(
                    target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
                )
            ],
            evidence_entries=[
                _evidence_entry(
                    target_id="AF-0000000000000001",
                    model_entity_id="AF-0000000000000001",
                    tool_used=_COLABFOLD_TOOL,
                    ptm=0.9,
                    iptm=0.8,
                )
            ],
            master_output=master_output,
            tracking_output=tracking_output,
        )
    assert not master_output.exists()
    assert not tracking_output.exists()


def test_forced_replacement_restores_old_outputs_on_commit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    master_output = tmp_path / "master.parquet"
    tracking_output = tmp_path / "tracking.parquet"
    master_output.write_bytes(b"old-master")
    tracking_output.write_bytes(b"old-tracking")
    real_replace = os.replace
    flaky: dict[str, bool] = {"failed": False}

    def flaky_replace(src: object, dst: object) -> None:
        if Path(dst) == master_output and not flaky["failed"]:
            flaky["failed"] = True
            raise OSError("simulated master rename failure")
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(SeamDerivationError, match="seam parquet derivation failed"):
        _derive(
            tmp_path,
            index_entries=[
                _index_entry(
                    target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
                )
            ],
            evidence_entries=[
                _evidence_entry(
                    target_id="AF-0000000000000001",
                    model_entity_id="AF-0000000000000001",
                    tool_used=_COLABFOLD_TOOL,
                    ptm=0.9,
                    iptm=0.8,
                )
            ],
            master_output=master_output,
            tracking_output=tracking_output,
            force=True,
        )
    assert master_output.read_bytes() == b"old-master"
    assert tracking_output.read_bytes() == b"old-tracking"


def test_derivation_never_touches_predictable_bak_siblings(tmp_path: Path) -> None:
    master_output = tmp_path / "master.parquet"
    tracking_output = tmp_path / "tracking.parquet"
    predictable_master_bak = tmp_path / "master.parquet.bak"
    predictable_tracking_bak = tmp_path / "tracking.parquet.bak"
    predictable_master_bak.write_bytes(b"legit-master-bak")
    predictable_tracking_bak.write_bytes(b"legit-tracking-bak")

    _derive(
        tmp_path,
        index_entries=[
            _index_entry(
                target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
            )
        ],
        evidence_entries=[
            _evidence_entry(
                target_id="AF-0000000000000001",
                model_entity_id="AF-0000000000000001",
                tool_used=_COLABFOLD_TOOL,
                ptm=0.9,
                iptm=0.8,
            )
        ],
        master_output=master_output,
        tracking_output=tracking_output,
    )
    assert predictable_master_bak.read_bytes() == b"legit-master-bak"
    assert predictable_tracking_bak.read_bytes() == b"legit-tracking-bak"


def test_concurrent_derivations_are_serialized(tmp_path: Path) -> None:
    import fcntl
    import os
    import threading
    import time

    master_output = tmp_path / "master.parquet"
    tracking_output = tmp_path / "tracking.parquet"
    lock_path = master_output.with_name(master_output.name + ".lock")
    gate = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(gate, fcntl.LOCK_EX)

    started = threading.Event()
    outcome: dict[str, object] = {}

    def run() -> None:
        started.set()
        try:
            outcome["result"] = _derive(
                tmp_path,
                index_entries=[
                    _index_entry(
                        target_id="AF-0000000000000001",
                        model_entity_id="AF-0000000000000001",
                        tool_used=_COLABFOLD_TOOL,
                    )
                ],
                evidence_entries=[
                    _evidence_entry(
                        target_id="AF-0000000000000001",
                        model_entity_id="AF-0000000000000001",
                        tool_used=_COLABFOLD_TOOL,
                        ptm=0.9,
                        iptm=0.8,
                    )
                ],
                master_output=master_output,
                tracking_output=tracking_output,
            )
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(timeout=5)
    time.sleep(0.5)
    assert not master_output.exists()  # blocked on the destination lock
    assert not tracking_output.exists()

    fcntl.flock(gate, fcntl.LOCK_UN)
    os.close(gate)
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert "result" in outcome
    assert master_output.exists()
    assert tracking_output.exists()

    # A second force=False derivation now fails closed instead of overwriting.
    with pytest.raises(SeamDerivationError, match="already exists"):
        _derive(
            tmp_path,
            index_entries=[
                _index_entry(
                    target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
                )
            ],
            evidence_entries=[
                _evidence_entry(
                    target_id="AF-0000000000000001",
                    model_entity_id="AF-0000000000000001",
                    tool_used=_COLABFOLD_TOOL,
                    ptm=0.9,
                    iptm=0.8,
                )
            ],
            master_output=master_output,
            tracking_output=tracking_output,
        )


def test_lock_covers_tracking_destination(tmp_path: Path) -> None:
    import fcntl
    import os
    import threading
    import time

    master_output = tmp_path / "master-a.parquet"
    tracking_output = tmp_path / "tracking.parquet"
    lock_path = tracking_output.resolve().with_name(tracking_output.name + ".lock")
    gate = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(gate, fcntl.LOCK_EX)

    started = threading.Event()
    outcome: dict[str, object] = {}

    def run() -> None:
        started.set()
        try:
            outcome["result"] = _derive(
                tmp_path,
                index_entries=[
                    _index_entry(
                        target_id="AF-0000000000000001",
                        model_entity_id="AF-0000000000000001",
                        tool_used=_COLABFOLD_TOOL,
                    )
                ],
                evidence_entries=[
                    _evidence_entry(
                        target_id="AF-0000000000000001",
                        model_entity_id="AF-0000000000000001",
                        tool_used=_COLABFOLD_TOOL,
                        ptm=0.9,
                        iptm=0.8,
                    )
                ],
                master_output=master_output,
                tracking_output=tracking_output,
            )
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(timeout=5)
    time.sleep(0.5)
    assert not master_output.exists()  # blocked on the tracking destination lock
    assert not tracking_output.exists()

    fcntl.flock(gate, fcntl.LOCK_UN)
    os.close(gate)
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert master_output.exists()
    assert tracking_output.exists()


def test_lock_creates_missing_parent_directory(tmp_path: Path) -> None:
    master_output = tmp_path / "nested" / "master.parquet"
    tracking_output = tmp_path / "nested" / "tracking.parquet"
    result = _derive(
        tmp_path,
        index_entries=[
            _index_entry(
                target_id="AF-0000000000000001", model_entity_id="AF-0000000000000001", tool_used=_COLABFOLD_TOOL
            )
        ],
        evidence_entries=[
            _evidence_entry(
                target_id="AF-0000000000000001",
                model_entity_id="AF-0000000000000001",
                tool_used=_COLABFOLD_TOOL,
                ptm=0.9,
                iptm=0.8,
            )
        ],
        master_output=master_output,
        tracking_output=tracking_output,
    )
    assert master_output.exists()
    assert tracking_output.exists()
    assert result.row_count == 1


def test_destination_lock_closes_fd_when_flock_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl
    import os

    real_open = os.open
    real_close = os.close
    real_flock = fcntl.flock
    opened: list[int] = []
    closed: list[int] = []

    def spy_open(path: object, flags: int, mode: int = 0o777) -> int:
        fd = real_open(path, flags, mode)  # type: ignore[arg-type]
        opened.append(fd)
        return fd

    def spy_close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    def flaky_flock(fd: int, op: int) -> None:
        if op == fcntl.LOCK_EX:
            raise OSError("simulated flock failure")
        real_flock(fd, op)

    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "close", spy_close)
    monkeypatch.setattr(fcntl, "flock", flaky_flock)

    with (
        pytest.raises(OSError, match="simulated flock failure"),
        _destination_lock(tmp_path / "master.parquet", tmp_path / "tracking.parquet"),
    ):
        pass

    assert opened
    assert set(opened) <= set(closed)
