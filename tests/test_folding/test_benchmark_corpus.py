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

"""Tests for benchmark corpus fetch and fail-closed integrity verification."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest

from bspp.orchestration.control.folding_benchmark.curator import _dataset_fingerprint
from bspp.orchestration.runtime.data_movement.common import TransferResult
from bspp.orchestration.runtime.data_movement.s3.client import S3Credentials
from bspp.orchestration.runtime.folding.benchmark.corpus import (
    _recompute_dataset_fingerprint,
    _verify_dataset_fingerprint,
    fetch_pinned_corpus,
)

_S3_LOCATION = "s3://benchmarks/pdb-temporal-2022-2025-v1/"


def _spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": "pdb-temporal-2022-2025-v1",
        "description": "test",
        "selection_seed": "test-seed",
        "source": {"provider": "RCSB PDB"},
        "filters": {"minimum_chain_length": 1},
        "strata": [
            {
                "name": "monomer_short",
                "chain_count": 1,
                "minimum_total_residues": 1,
                "maximum_total_residues": 10,
                "count": 2,
            }
        ],
        "throughput_subset_sizes": [10],
    }


def _record(
    target_id: str,
    *,
    sequence_sha256: str,
    stratum: str,
    reference_sha256: str,
    source_gzip_sha256: str = "",
    source_mmcif_sha256: str = "",
) -> dict[str, object]:
    record: dict[str, object] = {
        "target_id": target_id,
        "sequence_sha256": sequence_sha256,
        "stratum": stratum,
        "reference_sha256": reference_sha256,
        "description": "desc",
        "sequence": "AG",
    }
    if source_gzip_sha256:
        record["source_gzip_sha256"] = source_gzip_sha256
    if source_mmcif_sha256:
        record["source_mmcif_sha256"] = source_mmcif_sha256
    return record


def _records() -> list[dict[str, object]]:
    return [
        _record(
            "t2",
            sequence_sha256="b" * 64,
            stratum="monomer_short",
            reference_sha256="d" * 64,
            source_gzip_sha256="f" * 64,
        ),
        _record(
            "t1",
            sequence_sha256="a" * 64,
            stratum="monomer_short",
            reference_sha256="c" * 64,
            source_gzip_sha256="e" * 64,
        ),
    ]


def _mmcif_records() -> list[dict[str, object]]:
    return [
        _record(
            "t2",
            sequence_sha256="b" * 64,
            stratum="monomer_short",
            reference_sha256="d" * 64,
            source_mmcif_sha256="f" * 64,
        ),
        _record(
            "t1",
            sequence_sha256="a" * 64,
            stratum="monomer_short",
            reference_sha256="c" * 64,
            source_mmcif_sha256="e" * 64,
        ),
    ]


def _write_checksums(root: Path) -> None:
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}" for p in paths]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_corpus(root: Path) -> str:
    """Materialize a curator-shaped corpus in *root* and return its fingerprint."""
    root.mkdir(parents=True, exist_ok=True)
    spec = _spec()
    records = sorted(_records(), key=lambda item: str(item["target_id"]))
    references = root / "references"
    references.mkdir()
    for record in records:
        (references / f"{record['target_id']}.pdb").write_bytes(f"CA {record['target_id']}".encode())
    fingerprint = _dataset_fingerprint(
        spec,
        [
            (
                str(record["target_id"]),
                str(record["sequence_sha256"]),
                str(record["stratum"]),
                str(record["reference_sha256"]),
                str(record["source_gzip_sha256"]),
            )
            for record in records
        ],
    )
    dataset = {
        "schema_version": 1,
        "dataset_id": "pdb-temporal-2022-2025-v1",
        "dataset_fingerprint": fingerprint,
        "specification": spec,
    }
    (root / "dataset.json").write_text(json.dumps(dataset, sort_keys=True), encoding="utf-8")
    (root / "targets.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_checksums(root)
    return fingerprint


def _install_fake_transfer(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    *,
    ok: bool = True,
    returncode: int = 0,
) -> dict[str, object]:
    calls: dict[str, object] = {"src": None, "credentials": None}

    def _cp(
        src: str | Path,
        dst: str | Path,
        *,
        credentials: S3Credentials | None = None,
        numworkers: int | None = None,
        extra_args: tuple[str, ...] = (),
        dry_run: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> TransferResult:
        calls["src"] = str(src)
        calls["credentials"] = credentials
        if ok:
            shutil.copytree(source, Path(dst), dirs_exist_ok=True)
        return TransferResult(tool="s5cmd", argv=("cp", str(src), str(dst)), returncode=returncode, elapsed_s=0.0)

    monkeypatch.setattr("bspp.orchestration.runtime.data_movement.s3.transfer.cp", _cp)
    return calls


def test_valid_corpus_returns_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    fingerprint = _build_corpus(source)
    destination = tmp_path / "corpus"
    _install_fake_transfer(monkeypatch, source)

    result = fetch_pinned_corpus(_S3_LOCATION, destination, expected_fingerprint=fingerprint)

    assert result == destination
    assert (destination / "dataset.json").is_file()


def test_wrong_expected_fingerprint_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    _build_corpus(source)
    _install_fake_transfer(monkeypatch, source)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        fetch_pinned_corpus(_S3_LOCATION, tmp_path / "corpus", expected_fingerprint="0" * 64)


def test_wrong_stored_fingerprint_raises(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    fingerprint = _build_corpus(root)
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["dataset_fingerprint"] = "1" * 64
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _verify_dataset_fingerprint(root, fingerprint)


def test_tampered_payload_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    fingerprint = _build_corpus(source)
    reference = next((source / "references").iterdir())
    reference.write_bytes(b"tampered")
    _install_fake_transfer(monkeypatch, source)

    with pytest.raises(ValueError, match="digest mismatch"):
        fetch_pinned_corpus(_S3_LOCATION, tmp_path / "corpus", expected_fingerprint=fingerprint)


def test_checksums_lists_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    fingerprint = _build_corpus(source)
    sums_path = source / "SHA256SUMS"
    sums_path.write_text(
        sums_path.read_text(encoding="utf-8") + f"{'9' * 64}  references/missing.pdb\n",
        encoding="utf-8",
    )
    _install_fake_transfer(monkeypatch, source)

    with pytest.raises(ValueError, match="missing file"):
        fetch_pinned_corpus(_S3_LOCATION, tmp_path / "corpus", expected_fingerprint=fingerprint)


def test_missing_checksums_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    fingerprint = _build_corpus(source)
    (source / "SHA256SUMS").unlink()
    _install_fake_transfer(monkeypatch, source)

    with pytest.raises(ValueError, match="SHA256SUMS"):
        fetch_pinned_corpus(_S3_LOCATION, tmp_path / "corpus", expected_fingerprint=fingerprint)


def test_missing_dataset_json_raises(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _build_corpus(root)
    (root / "dataset.json").unlink()

    with pytest.raises(ValueError, match=r"dataset\.json"):
        _verify_dataset_fingerprint(root, "0" * 64)


def test_failed_transfer_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    fingerprint = _build_corpus(source)
    _install_fake_transfer(monkeypatch, source, ok=False, returncode=1)

    with pytest.raises(ValueError, match="transfer failed"):
        fetch_pinned_corpus(_S3_LOCATION, tmp_path / "corpus", expected_fingerprint=fingerprint)


def test_credentials_and_location_forwarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    fingerprint = _build_corpus(source)
    calls = _install_fake_transfer(monkeypatch, source)
    credentials = S3Credentials(
        access_key_id="access",
        secret_access_key="secret",
        endpoint_url="https://swiftstack.example",
    )

    fetch_pinned_corpus(
        _S3_LOCATION,
        tmp_path / "corpus",
        credentials=credentials,
        expected_fingerprint=fingerprint,
    )

    assert calls["src"] == "s3://benchmarks/pdb-temporal-2022-2025-v1/*"
    assert calls["credentials"] == credentials


@pytest.mark.parametrize(
    ("s3_location", "expected"),
    [
        ("s3://benchmarks/pdb-temporal-2022-2025-v1/", "s3://benchmarks/pdb-temporal-2022-2025-v1/*"),
        ("s3://benchmarks/pdb-temporal-2022-2025-v1", "s3://benchmarks/pdb-temporal-2022-2025-v1/*"),
        ("s3://benchmarks/pdb-temporal-2022-2025-v1/*", "s3://benchmarks/pdb-temporal-2022-2025-v1/*"),
    ],
)
def test_corpus_prefix_normalized_to_recursive_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    s3_location: str,
    expected: str,
) -> None:
    source = tmp_path / "source"
    fingerprint = _build_corpus(source)
    calls = _install_fake_transfer(monkeypatch, source)

    fetch_pinned_corpus(s3_location, tmp_path / "corpus", expected_fingerprint=fingerprint)

    assert calls["src"] == expected


def test_recompute_matches_curator_fingerprint() -> None:
    spec = _spec()
    records = sorted(_records(), key=lambda item: str(item["target_id"]))
    expected = _dataset_fingerprint(
        spec,
        [
            (
                str(record["target_id"]),
                str(record["sequence_sha256"]),
                str(record["stratum"]),
                str(record["reference_sha256"]),
                str(record["source_gzip_sha256"]),
            )
            for record in records
        ],
    )

    assert _recompute_dataset_fingerprint({"specification": spec}, records) == expected


def test_recompute_mmcif_fingerprint_matches_curator() -> None:
    """Build corpus with source_mmcif_sha256, verify _recompute selects mmCIF profile."""
    spec = _spec()
    records = sorted(_mmcif_records(), key=lambda item: str(item["target_id"]))
    expected = _dataset_fingerprint(
        spec,
        [
            (
                str(record["target_id"]),
                str(record["sequence_sha256"]),
                str(record["stratum"]),
                str(record["reference_sha256"]),
                str(record["source_mmcif_sha256"]),
            )
            for record in records
        ],
        source_key="source_mmcif_sha256",
        fingerprint_prefix=b"afdb-pdb-benchmark-mmcif-v1\0",
    )
    assert _recompute_dataset_fingerprint({"specification": spec}, records) == expected


def test_recompute_gzip_fingerprint_still_works() -> None:
    """Backward compatibility: gzip-profile records still work."""
    spec = _spec()
    records = sorted(_records(), key=lambda item: str(item["target_id"]))
    result = _recompute_dataset_fingerprint({"specification": spec}, records)
    assert len(result) == 64


def test_recompute_selects_profile_from_first_record() -> None:
    """Profile detection: first record determines which profile is used."""
    spec = _spec()
    mmcif_records = sorted(_mmcif_records(), key=lambda item: str(item["target_id"]))
    gzip_records = sorted(_records(), key=lambda item: str(item["target_id"]))
    mmcif_fp = _recompute_dataset_fingerprint({"specification": spec}, mmcif_records)
    gzip_fp = _recompute_dataset_fingerprint({"specification": spec}, gzip_records)
    assert mmcif_fp != gzip_fp


def test_recompute_generator_input_not_dropped() -> None:
    """Pass a generator to _recompute_dataset_fingerprint and verify no records are lost."""
    spec = _spec()
    records = sorted(_records(), key=lambda item: str(item["target_id"]))

    def _gen():
        yield from records

    result = _recompute_dataset_fingerprint({"specification": spec}, _gen())
    expected = _dataset_fingerprint(
        spec,
        [
            (
                str(record["target_id"]),
                str(record["sequence_sha256"]),
                str(record["stratum"]),
                str(record["reference_sha256"]),
                str(record["source_gzip_sha256"]),
            )
            for record in records
        ],
    )
    assert result == expected
