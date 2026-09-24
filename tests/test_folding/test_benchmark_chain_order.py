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

"""Explicit synthetic unit fixtures for verified chain grouping, never live evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from tests.test_folding.test_benchmark_corpus import _write_checksums
from tests.test_folding.test_benchmark_structure import _atom_line

from bspp.orchestration.contract.folding_execution import folding_target_sequence_sha256
from bspp.orchestration.runtime.folding.benchmark.chain_order import ColabFoldChainOrderResolver
from bspp.orchestration.runtime.folding.benchmark.corpus import _recompute_dataset_fingerprint
from bspp.orchestration.runtime.folding.benchmark.index import load_canonical_pair_index
from bspp.orchestration.runtime.folding.benchmark.suite import load_validation_suite
from bspp.orchestration.runtime.folding.benchmark.validation import validate_run
from bspp.orchestration.runtime.folding.execution.bioir_session import BIOIR_TOOL_USED


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n")


def _fixture(
    root: Path,
    *,
    chains: tuple[str, ...] = ("AA", "GGG", "AA", "GGG"),
    reference_ids: tuple[str, ...] = ("A", "B", "C", "D"),
) -> tuple[Path, Path, Path, Path, str]:
    run = root / "run"
    corpus = root / "corpus"
    run.mkdir()
    corpus.mkdir()
    permutation = tuple(i for sequence in dict.fromkeys(chains) for i, item in enumerate(chains) if item == sequence)
    grouped = tuple(chains[i] for i in permutation)
    # Every chain has distinct, asymmetric coordinates. Only the proven
    # permutation, rather than merely any bijection, yields RMSD zero.
    coordinates = [(0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, 5.0, 2.0), (4.0, 3.0, 8.0)]
    reference = "".join(_atom_line(i + 1, 1, *coordinates[i], chain=chain) for i, chain in enumerate(reference_ids))
    predicted = "".join(
        _atom_line(i + 1, 1, *coordinates[original], chain=chr(ord("A") + i)) for i, original in enumerate(permutation)
    )
    (corpus / "reference.pdb").write_text(reference)
    (run / "pdb_test_assembly_1-model_v1.pdb").write_text(predicted)
    _write(
        run / "pdb_test_assembly_1-meta_v1.json",
        {
            "schema_version": 1,
            "plddt": [90.0] * len(chains),
            "pae": [[0.1] * len(chains) for _ in chains],
            "max_pae": 0.1,
            "ptm": 0.5,
            "iptm": 0.4,
        },
    )
    original_hash = folding_target_sequence_sha256(chains)
    record = {
        "target_id": "pdb_test_assembly_1",
        "sequence_sha256": original_hash,
        "sequence": ":".join(chains),
        "chains": chains,
        "chain_ids": reference_ids,
        "stratum": "test",
        "source_mmcif_sha256": "f" * 64,
        "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
        "reference_path": "reference.pdb",
    }
    dataset: dict[str, object] = {"specification": {"name": "explicit-unit-test-fixture"}}
    fingerprint = _recompute_dataset_fingerprint(dataset, [record])
    dataset["dataset_fingerprint"] = fingerprint
    _write(corpus / "dataset.json", dataset)
    _write(corpus / "targets.jsonl", record)
    _write_checksums(corpus)
    suite = root / "suite.json"
    _write(
        suite,
        {
            "schema_version": 1,
            "dataset_id": "unit-test",
            "fingerprint": fingerprint,
            "cases": [
                {
                    "target_id": record["target_id"],
                    "sequence_sha256": original_hash,
                    "reference_structure": "reference.pdb",
                    "reference_sha256": record["reference_sha256"],
                    "expected_pair_mode": "unpaired_paired",
                    "thresholds": {"ca_coverage": 0.7},
                    "chain_map": {chr(ord("A") + i): chain for i, chain in reversed(list(enumerate(reference_ids)))},
                }
            ],
        },
    )
    index = root / "index.json"
    _write(
        index,
        {
            "schema_version": 1,
            "run_id": "unit-test",
            "entries": [
                {
                    "target_id": record["target_id"],
                    "sequence_sha256": folding_target_sequence_sha256(grouped),
                    "model_entity_id": record["target_id"],
                    "tool_used": BIOIR_TOOL_USED,
                    "structure_path": "pdb_test_assembly_1-model_v1.pdb",
                    "scores_path": "pdb_test_assembly_1-meta_v1.json",
                }
            ],
        },
    )
    return run, suite, index, corpus, fingerprint


@pytest.mark.parametrize(
    "chains,reference_ids",
    [
        (("AA", "GGG", "AA", "GGG"), ("A", "B", "C", "D")),
        (("AA", "GGG", "AA", "GGG"), ("W", "X", "Y", "Z")),
        (("AA", "GGG", "AA"), ("A", "B", "C")),
    ],
)
def test_verified_grouping_maps_actual_coordinates_and_preserves_inputs(
    tmp_path: Path, chains: tuple[str, ...], reference_ids: tuple[str, ...]
) -> None:
    run, suite, index, corpus, fingerprint = _fixture(tmp_path, chains=chains, reference_ids=reference_ids)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = validate_run(run, suite, index, corpus, expected_fingerprint=fingerprint)
    case = result["cases"][0]
    assert result["passed_count"] == 1
    assert case["identity_valid"] and case["quality_valid"] and case["structure_valid"]
    assert case["ca_coverage"] == 1.0
    assert case["ca_rmsd"] == pytest.approx(0.0, abs=1e-10)
    expected_permutation = [0, 2, 1, 3] if len(chains) == 4 else [0, 2, 1]
    order = case["chain_order_identity"]
    assert order["match_mode"] == "verified-colabfold-grouping"
    assert order["grouped_to_original"] == expected_permutation
    assert order["effective_chain_map"] == {
        chr(ord("A") + i): reference_ids[original] for i, original in enumerate(expected_permutation)
    }
    assert order["original_sequence_sha256"] != order["observed_sequence_sha256"]
    assert pq.read_schema(run / "validation/validation.parquet").names == [
        "target_id",
        "identity_valid",
        "quality_valid",
        "structure_valid",
        "ca_coverage",
        "ca_rmsd",
        "output_has_nan",
        "passed",
    ]
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize(
    "mutation", ["residue", "cardinality", "permutation", "target", "tool", "pair-mode", "missing-pin"]
)
def test_unknown_identity_changes_are_not_grouping_equivalence(tmp_path: Path, mutation: str) -> None:
    run, suite, index, corpus, fingerprint = _fixture(tmp_path)
    payload = json.loads(index.read_text())
    entry = payload["entries"][0]
    if mutation == "residue":
        entry["sequence_sha256"] = folding_target_sequence_sha256(("AA", "AA", "GGG", "GGA"))
    elif mutation == "cardinality":
        entry["sequence_sha256"] = folding_target_sequence_sha256(("AA", "AA", "AA", "GGG"))
    elif mutation == "permutation":
        entry["sequence_sha256"] = folding_target_sequence_sha256(("GGG", "GGG", "AA", "AA"))
    elif mutation == "target":
        entry["target_id"] = "pdb_other_assembly_1"
    elif mutation == "tool":
        entry["tool_used"] = "OpenFold / AlphaFold-Multimer"
    elif mutation == "pair-mode":
        cases = json.loads(suite.read_text())
        cases["cases"][0]["expected_pair_mode"] = "paired"
        _write(suite, cases)
    _write(index, payload)
    result = validate_run(
        run, suite, index, corpus, expected_fingerprint=None if mutation == "missing-pin" else fingerprint
    )
    assert result["passed_count"] == 0
    assert result["cases"][0]["identity_valid"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-targets-checksum",
        "missing-dataset-checksum",
        "duplicate-checksum",
        "conflicting-checksum",
        "checksum-drift",
        "fingerprint-drift",
        "duplicate-record",
        "tampered-chains",
        "empty-chain",
        "reference-path",
        "reference-hash",
        "missing-map",
        "duplicate-map",
        "wrong-map",
        "duplicate-chain-id",
    ],
)
def test_unverified_or_inconsistent_corpus_cannot_enable_fallback(tmp_path: Path, mutation: str) -> None:
    run, suite, index, corpus, fingerprint = _fixture(tmp_path)
    record = json.loads((corpus / "targets.jsonl").read_text())
    if mutation in {"missing-targets-checksum", "missing-dataset-checksum"}:
        name = "targets.jsonl" if mutation == "missing-targets-checksum" else "dataset.json"
        sums = (corpus / "SHA256SUMS").read_text().splitlines()
        (corpus / "SHA256SUMS").write_text("\n".join(line for line in sums if not line.endswith(name)) + "\n")
    elif mutation in {"duplicate-checksum", "conflicting-checksum"}:
        path = corpus / "SHA256SUMS"
        line = next(line for line in path.read_text().splitlines() if line.endswith("targets.jsonl"))
        if mutation == "conflicting-checksum":
            line = "0" * 64 + line[64:]
        path.write_text(path.read_text() + line + "\n")
    elif mutation == "checksum-drift":
        (corpus / "targets.jsonl").write_text((corpus / "targets.jsonl").read_text() + " ")
    elif mutation == "fingerprint-drift":
        record["source_mmcif_sha256"] = "0" * 64
        _write(corpus / "targets.jsonl", record)
        _write_checksums(corpus)
    elif mutation == "duplicate-record":
        path = corpus / "targets.jsonl"
        path.write_text(path.read_text() * 2)
        _write_checksums(corpus)
    elif mutation in {"tampered-chains", "empty-chain", "reference-path", "reference-hash", "duplicate-chain-id"}:
        if mutation == "tampered-chains":
            record["chains"][0] = "GG"
        elif mutation == "empty-chain":
            record["chains"][0] = ""
        elif mutation == "reference-path":
            record["reference_path"] = "another.pdb"
        elif mutation == "reference-hash":
            record["reference_sha256"] = "0" * 64
        else:
            record["chain_ids"][1] = record["chain_ids"][0]
        _write(corpus / "targets.jsonl", record)
        _write_checksums(corpus)
    else:
        payload = json.loads(suite.read_text())
        chain_map = payload["cases"][0]["chain_map"]
        if mutation == "missing-map":
            del chain_map["B"]
        elif mutation == "duplicate-map":
            chain_map["B"] = chain_map["A"]
        else:
            chain_map["B"], chain_map["C"] = chain_map["C"], chain_map["B"]
        _write(suite, payload)
    raises = mutation in {
        "missing-targets-checksum",
        "missing-dataset-checksum",
        "duplicate-checksum",
        "conflicting-checksum",
        "checksum-drift",
        "fingerprint-drift",
        "duplicate-record",
        "empty-chain",
        "reference-hash",
    }
    if raises:
        with pytest.raises(ValueError):
            validate_run(run, suite, index, corpus, expected_fingerprint=fingerprint)
    else:
        result = validate_run(run, suite, index, corpus, expected_fingerprint=fingerprint)
        assert result["passed_count"] == 0
        assert result["cases"][0]["identity_valid"] is False


def test_exact_match_remains_valid_without_corpus_metadata(tmp_path: Path) -> None:
    run, suite, index, corpus, fingerprint = _fixture(tmp_path, chains=("AA", "AA", "GGG", "GGG"))
    for name in ("SHA256SUMS", "dataset.json", "targets.jsonl"):
        (corpus / name).unlink()
    result = validate_run(run, suite, index, corpus, expected_fingerprint=fingerprint)
    assert result["passed_count"] == 1
    assert "chain_order_identity" not in result["cases"][0]


def test_verified_snapshots_do_not_reread_mutated_metadata(tmp_path: Path) -> None:
    _run, suite_path, index_path, corpus, fingerprint = _fixture(tmp_path)
    case = load_validation_suite(suite_path).cases[0]
    entry = load_canonical_pair_index(index_path).entries[0]
    resolver = ColabFoldChainOrderResolver(corpus, fingerprint)
    first = resolver.resolve(case, entry)
    (corpus / "targets.jsonl").write_text("untrusted later replacement")
    assert resolver.resolve(case, entry) == first
    assert first is not None
    with pytest.raises(ValueError, match="checksum mismatch"):
        ColabFoldChainOrderResolver(corpus, fingerprint).resolve(case, entry)


@pytest.mark.parametrize("name", ["SHA256SUMS", "dataset.json", "targets.jsonl"])
@pytest.mark.parametrize("kind", ["fifo", "symlink"])
def test_nonregular_metadata_cannot_block_or_enable_fallback(tmp_path: Path, name: str, kind: str) -> None:
    _run, suite_path, index_path, corpus, fingerprint = _fixture(tmp_path)
    path = corpus / name
    original = path.read_bytes()
    path.unlink()
    if kind == "fifo":
        os.mkfifo(path)
    else:
        target = corpus / "replacement"
        target.write_bytes(original)
        path.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        ColabFoldChainOrderResolver(corpus, fingerprint).resolve(
            load_validation_suite(suite_path).cases[0], load_canonical_pair_index(index_path).entries[0]
        )
