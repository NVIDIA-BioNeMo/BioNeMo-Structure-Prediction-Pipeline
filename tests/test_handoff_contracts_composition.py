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

"""Cross-story composition tests for the phase-0 handoff contracts.

These tests execute the continuous seam that the runtime lane previously
exercised inline: MSA Artifact Set -> folding input -> prediction pair ->
prediction bundle -> master-parquet projection.  They exist to catch drift
between the independently shipped contract modules.
"""

from __future__ import annotations

from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.master_parquet_projection import MASTER_PARQUET_PROJECTION
from bspp.orchestration.contract.model_identity import normalize_model_entity_id
from bspp.orchestration.contract.prediction_bundle import (
    PredictionArchiveBundle,
    prediction_archive_bundle_from_mapping,
)
from bspp.orchestration.contract.prediction_pair import prediction_pair_from_mapping
from bspp.orchestration.contract.preprocessing_handoff import (
    MsaArtifactSetManifest,
    MsaChunkManifestReference,
    msa_artifact_set_id,
)
from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION

_MODEL_ID = "AF-0000000000000001"


def _scores_mapping(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "plddt": [90.0, 85.0, 80.0],
        "pae": [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
        "max_pae": 3.0,
        "ptm": 0.9,
        "iptm": 0.8,
    }
    payload.update(overrides)
    return payload


def _pair_mapping(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "model_entity_id": _MODEL_ID,
        "tool_used": VALID_TOOL_USED[0],
        "structure_path": f"/data/{_MODEL_ID}-model_v1.pdb",
        "scores_path": f"/data/{_MODEL_ID}-meta_v1.json",
        "scores": _scores_mapping(),
    }
    payload.update(overrides)
    return payload


def _make_reference(member_count: int) -> MsaChunkManifestReference:
    chunk_name = "sample_tranche00_00001.fa"
    return MsaChunkManifestReference(
        chunk_name=chunk_name,
        logical_path=f"chunks/{chunk_name.removesuffix('.fa')}.json",
        sha256="a" * 64,
        member_count=member_count,
        logical_bytes=member_count * 100,
    )


def _make_manifest(member_count: int) -> MsaArtifactSetManifest:
    reference = _make_reference(member_count)
    logical_bytes = member_count * 100
    return MsaArtifactSetManifest(
        artifact_set_id=msa_artifact_set_id((reference,), member_count, logical_bytes),
        chunks=(reference,),
        member_count=member_count,
        logical_bytes=logical_bytes,
    )


def test_member_stem_normalizes_into_prediction_pair() -> None:
    normalized = normalize_model_entity_id("AFDB_AF-0000000000000001")

    assert normalized == _MODEL_ID
    pair = prediction_pair_from_mapping(_pair_mapping(model_entity_id=normalized))
    assert pair.model_entity_id == normalized


def test_bundle_round_trip_members_are_prediction_pair_identities() -> None:
    member_ids = tuple(
        normalize_model_entity_id(stem) for stem in ("AFDB_AF-0000000000000001", "AFDB_AF-0000000000000002")
    )
    bundle = PredictionArchiveBundle(
        bundle_name="bspp_260903_1234_a00001.tar.lz4",
        member_ids=member_ids,
        member_count=len(member_ids),
        sha256="a" * 64,
        size_bytes=100,
        created_at=None,
    )

    parsed = prediction_archive_bundle_from_mapping(bundle.to_mapping())

    assert parsed == bundle
    for member_id in parsed.member_ids:
        pair = prediction_pair_from_mapping(
            _pair_mapping(
                model_entity_id=member_id,
                structure_path=f"/data/{member_id}-model_v1.pdb",
                scores_path=f"/data/{member_id}-meta_v1.json",
            )
        )
        assert pair.model_entity_id == member_id


_ARCHIVE_COLUMN_DTYPES = {
    "archive_file": "string",
    "archive_sha256": "string",
    "archive_member_count": "Int64",
    "archive_size_bytes": "Int64",
    "archive_created_at": "string",
}


def test_bundle_fields_map_to_master_parquet_archive_columns() -> None:
    columns = {column.name: column for column in MASTER_PARQUET_PROJECTION}

    for column_name, dtype in _ARCHIVE_COLUMN_DTYPES.items():
        assert column_name in columns, f"missing master parquet archive column {column_name}"
        assert columns[column_name].dtype == dtype


def test_msa_set_to_folding_input_to_prediction_pair() -> None:
    manifest = _make_manifest(member_count=1)
    consumption = MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=("a3ms/AFDB_AF-0000000000000001.a3m",),
        requires_paired_query_header=True,
    )

    consumption.validate_against_manifest(manifest)

    member_stem = consumption.member_a3m_paths[0].removeprefix("a3ms/").removesuffix(".a3m")
    model_entity_id = normalize_model_entity_id(member_stem)
    assert model_entity_id == _MODEL_ID
    pair = prediction_pair_from_mapping(_pair_mapping(model_entity_id=model_entity_id))
    assert pair.model_entity_id == model_entity_id
