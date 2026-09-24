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

"""Tests for local-tar decompressed payload parity."""

from __future__ import annotations

import tarfile
from io import BytesIO
from pathlib import Path

import pytest
import zstandard
from click.testing import CliRunner

from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.validation.tar_payload_parity import compare_tar_payload_parity


def _write_tar(path: Path, files: dict[str, bytes | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.stem}-staging"
    staging.mkdir()
    for relative_path, content in files.items():
        source = staging / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        payload = content.encode() if isinstance(content, str) else content
        source.write_bytes(payload)
    with tarfile.open(path, "w") as archive:
        for source in sorted(p for p in staging.rglob("*") if p.is_file()):
            archive.add(source, arcname=str(source.relative_to(staging)))


def _write_tar_entries(path: Path, entries: tuple[tuple[str, bytes | str], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, content in entries:
            payload = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))


def _zstd(payload: bytes | str) -> bytes:
    raw = payload.encode() if isinstance(payload, str) else payload
    return zstandard.ZstdCompressor().compress(raw)


def test_compare_tar_payload_parity_accepts_matching_raw_members(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_tar(baseline / "local_tars" / "shard_1" / "batch_0.tar", {"AF-1.pdb": "MODEL\n"})
    _write_tar(candidate / "local_tars" / "shard_1" / "batch_0.tar", {"AF-1.pdb": "MODEL\n"})

    report = compare_tar_payload_parity(baseline, candidate, workers=2, sample_limit=7)

    assert report.ok
    assert report.compared_members == 1
    assert report.files[0].compressed_size_mismatch_count == 0
    assert report.to_redacted_dict()["sample_limit"] == 7


def test_compare_tar_payload_parity_normalizes_run_name_in_member_names(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_tar(
        baseline / "local_tars" / "metadata" / "shard_1_metadata.tar",
        {"metadata/search/AF-metadata-1-of-2-upstream_run.json": "same\n"},
    )
    _write_tar(
        candidate / "local_tars" / "metadata" / "shard_1_metadata.tar",
        {"metadata/search/AF-metadata-1-of-2-orchestration_run.json": "same\n"},
    )

    report = compare_tar_payload_parity(
        baseline,
        candidate,
        baseline_run_name="upstream_run",
        candidate_run_name="orchestration_run",
    )

    assert report.ok
    assert report.files[0].compared_members == 1


def test_compare_tar_payload_parity_detects_payload_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_tar(baseline / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL A\n"})
    _write_tar(candidate / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL B\n"})

    report = compare_tar_payload_parity(baseline, candidate)

    assert not report.ok
    assert report.payload_mismatch_count == 1
    assert report.files[0].payload_mismatch_sample == ("AF-1.pdb",)


def test_compare_tar_payload_parity_detects_tar_inventory_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_tar(baseline / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL\n"})
    _write_tar(candidate / "local_tars" / "batch_1.tar", {"AF-1.pdb": "MODEL\n"})

    report = compare_tar_payload_parity(baseline, candidate)

    assert not report.ok
    assert report.baseline_only_tars == ("batch_0.tar",)
    assert report.candidate_only_tars == ("batch_1.tar",)
    payload = report.to_redacted_dict()
    assert payload["baseline_tar_count"] == 1
    assert payload["candidate_tar_count"] == 1


def test_compare_tar_payload_parity_exclude_skips_extra_candidate_metadata_tars(tmp_path: Path) -> None:
    """Candidate metadata tars absent from baseline are excluded via --exclude."""
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_tar(baseline / "local_tars" / "shard_1" / "batch_0.tar", {"AF-1.pdb": "MODEL\n"})
    _write_tar(candidate / "local_tars" / "shard_1" / "batch_0.tar", {"AF-1.pdb": "MODEL\n"})
    _write_tar(candidate / "local_tars" / "metadata" / "shard_1706_metadata.tar", {"meta.json": "{}\n"})

    report = compare_tar_payload_parity(baseline, candidate, exclude=("metadata/",))

    assert report.tar_file_list_ok
    assert report.baseline_only_tars == ()
    assert report.candidate_only_tars == ()
    assert report.ok


def test_compare_tar_payload_parity_fails_empty_inventory(tmp_path: Path) -> None:
    report = compare_tar_payload_parity(tmp_path / "baseline", tmp_path / "candidate")

    assert not report.ok
    assert report.inventory_errors == (
        "no paired tar files found under relative_dir='local_tars' (baseline=0, candidate=0)",
    )
    payload = report.to_redacted_dict()
    assert payload["compared_tar_count"] == 0
    assert payload["error_count"] == 1


def test_compare_tar_payload_parity_reports_duplicate_normalized_members(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_tar_entries(
        baseline / "local_tars" / "batch_0.tar",
        (("AF-1.pdb", "MODEL\n"), ("AF-1.pdb", "MODEL\n")),
    )
    _write_tar(candidate / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL\n"})

    report = compare_tar_payload_parity(baseline, candidate)

    assert not report.ok
    assert report.files[0].duplicate_baseline_normalized_names == ("AF-1.pdb",)
    assert report.to_redacted_dict()["duplicate_normalized_member_count"] == 1


def test_compare_tar_payload_parity_by_tar_inventory_only_skips_payload_hashes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_tar(baseline / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL A\n"})
    _write_tar(candidate / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL B\n"})

    report = compare_tar_payload_parity(baseline, candidate, payload_sample_count=0)

    assert report.ok
    assert report.compared_members == 0
    assert report.inventory_only is True
    assert report.to_redacted_dict()["payload_hash_scope"] == "inventory-only"
    assert report.files[0].member_names_ok


def test_compare_tar_payload_parity_by_tar_sample_detects_sampled_payload_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_tar(baseline / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL A\n"})
    _write_tar(candidate / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL B\n"})

    report = compare_tar_payload_parity(baseline, candidate, payload_sample_count=10)

    assert not report.ok
    assert report.compared_members == 1
    assert report.payload_mismatch_count == 1


def test_compare_tar_payload_parity_reports_modelcif_bcif_rounding_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    model_id = "AF-0000000210947144"
    baseline_cif = """\
data_AF-0000000210947144
loop_
_ma_qa_metric_global.ordinal_id
_ma_qa_metric_global.model_id
_ma_qa_metric_global.metric_id
_ma_qa_metric_global.metric_value
1 1 1 81.58
2 1 3 0.2
"""
    candidate_cif = baseline_cif.replace("81.58", "81.59")

    _write_tar(
        baseline / "local_tars" / "shard_1710" / "shard_1710_batch_1.tar",
        {
            f"{model_id}-model_v1.cif.zst": _zstd(baseline_cif),
            f"{model_id}-model_v1.bcif.zst": _zstd(b"BCIF\x00_ma_qa_metric_global.metric_value=81.58"),
            f"{model_id}-model_v1.pdb.zst": _zstd("MODEL\n"),
        },
    )
    _write_tar(
        candidate / "local_tars" / "shard_1710" / "shard_1710_batch_1.tar",
        {
            f"{model_id}-model_v1.cif.zst": _zstd(candidate_cif),
            f"{model_id}-model_v1.bcif.zst": _zstd(b"BCIF\x00_ma_qa_metric_global.metric_value=81.59"),
            f"{model_id}-model_v1.pdb.zst": _zstd("MODEL\n"),
        },
    )

    report = compare_tar_payload_parity(baseline, candidate)

    assert not report.ok
    assert report.compared_members == 3
    assert report.payload_mismatch_count == 2
    assert report.files[0].relative_path == "shard_1710/shard_1710_batch_1.tar"
    assert report.files[0].payload_mismatch_sample == (
        f"{model_id}-model_v1.bcif.zst",
        f"{model_id}-model_v1.cif.zst",
    )
    redacted = report.to_redacted_dict()
    assert redacted["ok"] is False
    assert redacted["payload_mismatch_count"] == 2

    inventory_report = compare_tar_payload_parity(baseline, candidate, payload_sample_count=0)
    assert inventory_report.ok
    assert inventory_report.compared_members == 0
    assert inventory_report.payload_mismatch_count == 0

    sampled_report = compare_tar_payload_parity(baseline, candidate, payload_sample_count=10)
    assert not sampled_report.ok
    assert sampled_report.compared_members == 3
    assert sampled_report.payload_mismatch_count == 2


def test_compare_tar_payload_parity_aggregate_mode_ignores_tar_placement(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_tar(baseline / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL 1\n"})
    _write_tar(baseline / "local_tars" / "batch_1.tar", {"AF-2.pdb": "MODEL 2\n"})
    _write_tar(candidate / "local_tars" / "batch_0.tar", {"AF-2.pdb": "MODEL 2\n"})
    _write_tar(candidate / "local_tars" / "batch_1.tar", {"AF-1.pdb": "MODEL 1\n"})

    by_tar = compare_tar_payload_parity(baseline, candidate)
    aggregate = compare_tar_payload_parity(baseline, candidate, match_mode="aggregate")

    assert not by_tar.ok
    assert aggregate.ok
    assert aggregate.compared_members == 2
    assert aggregate.files[0].relative_path == "<aggregate>"


def test_compare_tar_payload_parity_aggregate_inventory_only_skips_payload_hashes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_tar(baseline / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL A\n"})
    _write_tar(candidate / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL B\n"})

    report = compare_tar_payload_parity(
        baseline,
        candidate,
        match_mode="aggregate",
        payload_sample_count=0,
    )

    assert report.ok
    assert report.compared_members == 0
    assert report.to_redacted_dict()["payload_hash_scope"] == "inventory-only"
    assert report.files[0].baseline_member_count == 1
    assert report.files[0].candidate_member_count == 1


def test_compare_tar_payload_parity_aggregate_sample_detects_sampled_payload_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_tar(baseline / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL A\n"})
    _write_tar(candidate / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL B\n"})

    report = compare_tar_payload_parity(
        baseline,
        candidate,
        match_mode="aggregate",
        payload_sample_count=10,
    )

    assert not report.ok
    assert report.compared_members == 1
    assert report.payload_mismatch_count == 1


def test_compare_tar_payload_parity_hashes_zstd_members_without_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    compressor = zstandard.ZstdCompressor()
    payload = compressor.compress(b"MODEL\n")
    _write_tar(baseline / "local_tars" / "batch_0.tar", {"AF-1.pdb.zst": payload})
    _write_tar(candidate / "local_tars" / "batch_0.tar", {"AF-1.pdb.zst": payload})

    def fail_cli(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("zstd CLI should not be used when zstandard is importable")

    monkeypatch.setattr("bspp.orchestration.runtime.validation.tar_payload_parity.subprocess.run", fail_cli)

    report = compare_tar_payload_parity(baseline, candidate)

    assert report.ok
    assert report.compared_members == 1


def test_validate_tar_payload_parity_cli_writes_report_and_fails_strict_on_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    report_dir = tmp_path / "report"
    _write_tar(baseline / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL A\n"})
    _write_tar(candidate / "local_tars" / "batch_0.tar", {"AF-1.pdb": "MODEL B\n"})

    result = CliRunner().invoke(
        cli,
        [
            "validate",
            "tar-payload-parity",
            "--baseline-dir",
            str(baseline),
            "--candidate-dir",
            str(candidate),
            "--write-report",
            str(report_dir),
            "--strict",
        ],
    )

    assert result.exit_code == 1
    assert '"ok": false' in result.output
    assert (report_dir / "tar_payload_parity_report.json").exists()
    assert (report_dir / "tar_payload_parity_report.txt").exists()


def test_validate_tar_payload_parity_cli_writes_report_and_fails_strict_on_empty_inventory(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    (tmp_path / "baseline").mkdir()
    (tmp_path / "candidate").mkdir()

    result = CliRunner().invoke(
        cli,
        [
            "validate",
            "tar-payload-parity",
            "--baseline-dir",
            str(tmp_path / "baseline"),
            "--candidate-dir",
            str(tmp_path / "candidate"),
            "--write-report",
            str(report_dir),
            "--strict",
        ],
    )

    assert result.exit_code == 1
    assert "no paired tar files found" in result.output
    assert (report_dir / "tar_payload_parity_report.json").exists()
    assert (report_dir / "tar_payload_parity_report.txt").exists()
