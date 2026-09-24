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

"""Tests for the checksum-pinned validation-suite model and loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from bspp.orchestration.runtime.folding.benchmark.suite import (
    VALIDATION_SUITE_SCHEMA_VERSION,
    ValidationCase,
    ValidationSuite,
    load_validation_suite,
)

_GOLDEN = Path("folding") / "benchmark" / "validation-suite.json"


def _case(
    *,
    target_id: str = "pdb-temporal-2022-2025-v1/0001",
    sequence_sha256: str = "a" * 64,
    thresholds: dict[str, float] | None = None,
    require_no_nan: bool = True,
    expected_pair_mode: str = "single",
    reference_structure: str = "refs/0001.pdb",
    reference_sha256: str = "b" * 64,
    chain_map: dict[str, str] | None = None,
    metadata: dict[str, object] | None = None,
) -> ValidationCase:
    return ValidationCase(
        target_id=target_id,
        sequence_sha256=sequence_sha256,
        thresholds=thresholds if thresholds is not None else {"ca_coverage": 0.7},
        require_no_nan=require_no_nan,
        expected_pair_mode=expected_pair_mode,
        reference_structure=reference_structure,
        reference_sha256=reference_sha256,
        chain_map=chain_map if chain_map is not None else {"A": "A"},
        metadata=metadata if metadata is not None else {},
    )


class TestValidationCaseValidation:
    def test_empty_target_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_id"):
            _case(target_id="")

    def test_short_sequence_sha256_rejected(self) -> None:
        with pytest.raises(ValueError, match="64-character"):
            _case(sequence_sha256="a" * 63)

    def test_non_hex_sequence_sha256_rejected(self) -> None:
        with pytest.raises(ValueError, match="hexadecimal"):
            _case(sequence_sha256="g" * 64)

    def test_uppercase_sequence_sha256_is_normalized(self) -> None:
        case = _case(sequence_sha256="A" * 64)
        assert case.sequence_sha256 == "a" * 64

    def test_empty_reference_structure_rejected(self) -> None:
        with pytest.raises(ValueError, match="reference_structure"):
            _case(reference_structure="")

    def test_empty_reference_sha256_rejected(self) -> None:
        with pytest.raises(ValueError, match="reference_sha256"):
            _case(reference_sha256="")

    def test_empty_expected_pair_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="expected_pair_mode"):
            _case(expected_pair_mode="")

    def test_non_bool_require_no_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="require_no_nan"):
            ValidationCase(
                target_id="t",
                sequence_sha256="a" * 64,
                thresholds={"ca_coverage": 0.7},
                require_no_nan="yes",  # type: ignore[arg-type]
                expected_pair_mode="single",
                reference_structure="refs/0001.pdb",
                reference_sha256="b" * 64,
                chain_map={"A": "A"},
                metadata={},
            )

    def test_non_mapping_chain_map_rejected(self) -> None:
        with pytest.raises(ValueError, match="chain_map"):
            ValidationCase(
                target_id="t",
                sequence_sha256="a" * 64,
                thresholds={"ca_coverage": 0.7},
                require_no_nan=True,
                expected_pair_mode="single",
                reference_structure="refs/0001.pdb",
                reference_sha256="b" * 64,
                chain_map=["A"],  # type: ignore[arg-type]
                metadata={},
            )

    def test_non_mapping_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError, match="thresholds"):
            ValidationCase(
                target_id="t",
                sequence_sha256="a" * 64,
                thresholds=["ca_coverage"],  # type: ignore[arg-type]
                require_no_nan=True,
                expected_pair_mode="single",
                reference_structure="refs/0001.pdb",
                reference_sha256="b" * 64,
                chain_map={"A": "A"},
                metadata={},
            )


class TestValidationSuiteValidation:
    def test_bad_schema_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            ValidationSuite(schema_version=2, dataset_id="d", fingerprint="f", cases=(_case(),))

    def test_bool_schema_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            ValidationSuite(
                schema_version=True,  # type: ignore[arg-type]
                dataset_id="d",
                fingerprint="f",
                cases=(_case(),),
            )

    def test_empty_dataset_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="dataset_id"):
            ValidationSuite(schema_version=1, dataset_id="", fingerprint="f", cases=(_case(),))

    def test_empty_fingerprint_rejected(self) -> None:
        with pytest.raises(ValueError, match="fingerprint"):
            ValidationSuite(schema_version=1, dataset_id="d", fingerprint="", cases=(_case(),))

    def test_empty_cases_rejected(self) -> None:
        with pytest.raises(ValueError, match="cases"):
            ValidationSuite(schema_version=1, dataset_id="d", fingerprint="f", cases=())


class TestLoadValidationSuite:
    def test_absent_file_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Cannot read"):
            load_validation_suite(tmp_path / "missing.json")

    def test_malformed_json_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_bytes(b"{not json")
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_validation_suite(path)

    def test_non_object_top_level_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1,2,3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            load_validation_suite(path)

    def test_unknown_schema_version_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "suite.json"
        payload = {"schema_version": 2, "dataset_id": "d", "fingerprint": "f", "cases": []}
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version"):
            load_validation_suite(path)

    def test_empty_cases_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "suite.json"
        payload = {"schema_version": 1, "dataset_id": "d", "fingerprint": "f", "cases": []}
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="at least one case"):
            load_validation_suite(path)

    def test_invalid_case_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "suite.json"
        payload = {
            "schema_version": 1,
            "dataset_id": "d",
            "fingerprint": "f",
            "cases": [
                {
                    "target_id": "t",
                    "sequence_sha256": "a" * 63,
                    "thresholds": {"ca_coverage": 0.7},
                    "require_no_nan": True,
                    "expected_pair_mode": "single",
                    "reference_structure": "refs/0001.pdb",
                    "reference_sha256": "b" * 64,
                    "chain_map": {"A": "A"},
                    "metadata": {},
                }
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="64-character"):
            load_validation_suite(path)


class TestGoldenFixture:
    def test_load_golden_suite(self, fixtures_dir: Path) -> None:
        suite = load_validation_suite(fixtures_dir / _GOLDEN)
        assert suite.schema_version == VALIDATION_SUITE_SCHEMA_VERSION
        assert suite.dataset_id == "pdb-temporal-2022-2025-v1"
        assert len(suite.cases) == 2
        assert [case.target_id for case in suite.cases] == [
            "pdb-temporal-2022-2025-v1/0001",
            "pdb-temporal-2022-2025-v1/0002",
        ]

    def test_checksum_pins_raw_bytes(self, fixtures_dir: Path) -> None:
        path = fixtures_dir / _GOLDEN
        suite = load_validation_suite(path)
        assert suite.checksum == hashlib.sha256(path.read_bytes()).hexdigest()


class TestFrozenMappings:
    def test_loaded_case_mappings_are_proxy(self, fixtures_dir: Path) -> None:
        case = load_validation_suite(fixtures_dir / _GOLDEN).cases[0]
        assert isinstance(case.thresholds, MappingProxyType)
        assert isinstance(case.chain_map, MappingProxyType)
        assert isinstance(case.metadata, MappingProxyType)

    def test_thresholds_mutation_raises_type_error(self, fixtures_dir: Path) -> None:
        case = load_validation_suite(fixtures_dir / _GOLDEN).cases[0]
        with pytest.raises(TypeError):
            case.thresholds["ca_coverage"] = 0.9  # type: ignore[index]

    def test_chain_map_mutation_raises_type_error(self, fixtures_dir: Path) -> None:
        case = load_validation_suite(fixtures_dir / _GOLDEN).cases[0]
        with pytest.raises(TypeError):
            case.chain_map["A"] = "B"  # type: ignore[index]

    def test_metadata_mutation_raises_type_error(self, fixtures_dir: Path) -> None:
        case = load_validation_suite(fixtures_dir / _GOLDEN).cases[0]
        with pytest.raises(TypeError):
            case.metadata["class"] = "dimer"  # type: ignore[index]
