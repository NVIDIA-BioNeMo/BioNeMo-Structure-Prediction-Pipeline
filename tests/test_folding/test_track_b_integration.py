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

"""Cross-story integration tests for the Track-B folding companion engines.

These tests never run a real kernel or subprocess. They exercise the full
production seam: parse a chain manifest, run both emitters with fakes, verify
the canonical-pair contract and homogeneous output modes, then carry a produced
archive bundle through master-parquet publication and the existing archive
staging consumer.
"""

from __future__ import annotations

import json
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bspp.orchestration.contract.folding_archive import (
    ArchivePlanOptions,
    FoldingResultAssociation,
    FoldingResultInventory,
)
from bspp.orchestration.contract.prediction_pair import prediction_pair_from_mapping
from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.runtime.folding.archive import plan_folding_archives
from bspp.orchestration.runtime.folding.execution.archive_producer import (
    ArchiveCommandResult,
    execute_archive_batch,
)
from bspp.orchestration.runtime.folding.execution.backend import FoldingBackend
from bspp.orchestration.runtime.folding.execution.chain_manifest import (
    ambiguous_chain_manifest_metadata,
    parse_chain_manifest,
)
from bspp.orchestration.runtime.folding.execution.colabfold_backend import (
    COLABFOLD_TOOL_USED,
    ColabFoldBackend,
    ColabFoldConfig,
)
from bspp.orchestration.runtime.folding.execution.master_parquet_writer import (
    build_master_parquet_row,
    write_master_parquet,
)
from bspp.orchestration.runtime.folding.execution.models import (
    PreparedInput,
    ProteinTarget,
    StructurePrediction,
)
from bspp.orchestration.runtime.folding.execution.openfold_trt_backend import (
    OPENFOLD_TRT_TOOL_USED,
    OpenFoldTrtBackend,
)
from bspp.orchestration.runtime.inputs.archives import archives_for_dataset, plan_archive_staging

_GOOD_TARGET = "AF_0000000000000001_AF_0000000000000002"
_BAD_TARGET = "AF_0000000000000003_AF_0000000000000004"
_MODEL_TYPE = "alphafold2_multimer_v3"

_MANIFEST_BODY = (
    "model_entity_id,entity_id,chain_id,uniprot_ac\n"
    f"{_GOOD_TARGET},e1,A,P1\n"
    f"{_GOOD_TARGET},e2,B,P2\n"
    f"{_BAD_TARGET},e3,A,P1\n"
    f"{_BAD_TARGET},e4,A,P2\n"
)


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text(_MANIFEST_BODY, encoding="utf-8")
    return path


def _prepared(tmp_path: Path, backend: str) -> PreparedInput:
    return PreparedInput(
        backend,
        fasta_dir=tmp_path / "fasta",
        alignment_dir=tmp_path / "alignments",
        template_dir=tmp_path / "templates",
    )


def _write_raw_rank_one_pair(
    output_dir: Path,
    target_id: str,
    *,
    model_n: int = 3,
    seed: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb = output_dir / f"{target_id}_unrelaxed_rank_001_{_MODEL_TYPE}_model_{model_n}_seed_{seed:03d}.pdb"
    scores = output_dir / f"{target_id}_scores_rank_001_{_MODEL_TYPE}_model_{model_n}_seed_{seed:03d}.json"
    pdb.write_text(f"HEADER    {target_id}\nEND\n", encoding="utf-8")
    scores.write_text(
        json.dumps(
            {
                "plddt": [1.0, 2.0],
                "pae": [[1.0, 2.0], [2.0, 1.0]],
                "max_pae": 2.0,
                "ptm": 0.8765,
                "iptm": 0.8125,
            }
        ),
        encoding="utf-8",
    )


class _ColabFoldFakeRunner:
    """Materializes the harvested raw rank-1 pair and never spawns a subprocess."""

    def __init__(self, *, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def __call__(self, argv: Sequence[str], *, env: Mapping[str, str]) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((tuple(argv), dict(env)))
        output_dir = Path(argv[-1])
        target_id = Path(argv[-2]).stem
        _write_raw_rank_one_pair(output_dir, target_id)
        return subprocess.CompletedProcess(argv, self.returncode)


class _OpenFoldRecordingModelFn:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def __call__(self, prepared: PreparedInput) -> dict[str, object]:
        self.calls.append(prepared)
        return {
            "plddt": [81.585, 90.123],
            "predicted_aligned_error": [[0.0, 1.234], [1.234, 0.0]],
            "ptm": 0.8765,
            "iptm": 0.8125,
            "pdb_string": _openfold_pdb(),
        }


def _openfold_pdb() -> str:
    return (
        "ATOM      1  CA  GLY A   1      28.000  38.000  38.000  1.00 20.00           C\n"
        "ATOM      2  CA  GLY A   2      29.000  38.000  38.000  1.00 20.00           C\n"
        "END\n"
    )


def _colabfold_backend(tmp_path: Path, manifest_path: Path) -> ColabFoldBackend:
    config = ColabFoldConfig(
        weights_dir=tmp_path / "weights",
        msa_cache_dir=tmp_path / "msa_cache",
        structures_dir=tmp_path / "structures",
    )
    return ColabFoldBackend(config, parse_chain_manifest(manifest_path), runner=_ColabFoldFakeRunner())


def _openfold_backend(manifest_path: Path) -> OpenFoldTrtBackend:
    return OpenFoldTrtBackend(parse_chain_manifest(manifest_path), model_fn=_OpenFoldRecordingModelFn())


def _assert_canonical_pair_round_trips(prediction: StructurePrediction, tool_used: str, target_id: str) -> None:
    structure_path = prediction.structure_path
    scores_path = prediction.scores_path
    scores = json.loads(scores_path.read_bytes())
    pair = prediction_pair_from_mapping(
        {
            "schema_version": 1,
            "model_entity_id": target_id,
            "tool_used": tool_used,
            "structure_path": str(structure_path),
            "scores_path": str(scores_path),
            "scores": scores,
        }
    )
    assert pair.tool_used in VALID_TOOL_USED
    assert stat.S_IMODE(structure_path.stat().st_mode) == 0o644
    assert stat.S_IMODE(scores_path.stat().st_mode) == 0o644


def test_parsed_manifest_drives_both_backends_target_scoped(tmp_path: Path) -> None:
    manifest = parse_chain_manifest(_write_manifest(tmp_path))
    cf_runner = _ColabFoldFakeRunner()
    of_model_fn = _OpenFoldRecordingModelFn()
    cf_config = ColabFoldConfig(
        weights_dir=tmp_path / "weights",
        msa_cache_dir=tmp_path / "msa_cache",
        structures_dir=tmp_path / "structures",
    )
    colabfold = ColabFoldBackend(cf_config, manifest, runner=cf_runner)
    openfold = OpenFoldTrtBackend(manifest, model_fn=of_model_fn)

    good = ProteinTarget(_GOOD_TARGET, "desc", ("A", "B"))
    bad = ProteinTarget(_BAD_TARGET, "desc", ("A", "B"))

    good_colabfold = colabfold.run(good, _prepared(tmp_path, "colabfold"), tmp_path / "cf-good")
    bad_colabfold = colabfold.run(bad, _prepared(tmp_path, "colabfold"), tmp_path / "cf-bad")

    good_openfold = openfold.run(good, _prepared(tmp_path, "openfold-trt"), tmp_path / "of-good")
    bad_openfold = openfold.run(bad, _prepared(tmp_path, "openfold-trt"), tmp_path / "of-bad")

    assert len(good_colabfold.predictions) == 1
    assert len(good_openfold.predictions) == 1

    assert bad_colabfold.predictions == ()
    assert bad_openfold.predictions == ()
    assert bad_colabfold.metadata == ambiguous_chain_manifest_metadata()
    assert bad_openfold.metadata == ambiguous_chain_manifest_metadata()

    # The bad target is resolved from the same parsed manifest and must not
    # consume either backend's executable collaborator.
    assert len(cf_runner.calls) == 1
    assert len(of_model_fn.calls) == 1


def test_backend_output_homogeneity(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    colabfold = _colabfold_backend(tmp_path, manifest_path)
    openfold = _openfold_backend(manifest_path)
    target = ProteinTarget(_GOOD_TARGET, "desc", ("A", "B"))

    cf_result = colabfold.run(target, _prepared(tmp_path, "colabfold"), tmp_path / "cf")
    of_result = openfold.run(target, _prepared(tmp_path, "openfold-trt"), tmp_path / "of")

    # Both backends satisfy the structural protocol.
    cf_protocol: FoldingBackend = colabfold
    of_protocol: FoldingBackend = openfold
    assert cf_protocol.name == "colabfold"
    assert of_protocol.name == "openfold-trt"

    assert cf_result.metadata["tool_used"] == COLABFOLD_TOOL_USED
    assert of_result.metadata["tool_used"] == OPENFOLD_TRT_TOOL_USED

    _assert_canonical_pair_round_trips(cf_result.predictions[0], COLABFOLD_TOOL_USED, _GOOD_TARGET)
    _assert_canonical_pair_round_trips(of_result.predictions[0], OPENFOLD_TRT_TOOL_USED, _GOOD_TARGET)


class _FakeArchiveRunner:
    def __init__(self, payload: bytes = b"fake-lz4-bytes") -> None:
        self.payload = payload
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
        return ArchiveCommandResult(returncode=0)


def _complete_inventory(tmp_path: Path) -> FoldingResultInventory:
    root = tmp_path / "results"
    root.mkdir(parents=True)
    pdb = root / "model-0_unrelaxed_rank_001_model_1.pdb"
    scores = root / "model-0_scores_rank_001_model_1.json"
    pdb.write_text("pdb", encoding="utf-8")
    scores.write_text("{}", encoding="utf-8")
    return FoldingResultInventory(
        associations=(
            FoldingResultAssociation(
                source_ordinal=0,
                protein_id="model-0",
                normalized_protein_id="model-0",
                status="complete",
                pdb_paths=(str(pdb),),
                json_paths=(str(scores),),
            ),
        ),
        unmatched=(),
    )


def _complete_kwargs() -> dict[str, object]:
    return {
        "msa_path": "AFDB_AF-0000000000000001",
        "source_run": "r1",
        "pdb_path": "p.pdb",
        "json_path": "p.json",
        "pdb_residue_count": 100,
        "mean_plddt": 90.0,
        "plddt_above_70": 80.0,
        "ptm": 0.9,
        "iptm": 0.8,
        "max_pae": 5.0,
        "output_has_nan": "no",
        "pdb_json_match": "yes",
        "swiftstack_archive": "bspp_260909_1430_a00001.tar.lz4",
        "uploaded_to_gcp": "no",
    }


def test_archive_to_master_parquet_to_staging_consumer(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path)
    options = ArchivePlanOptions(
        run_tag="bspp_260909_1430_a",
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=False,
    )
    batch = plan_folding_archives(inventory, options).batches[0]

    payload = b"fake-lz4-bytes"
    bundle = execute_archive_batch(batch, runner=_FakeArchiveRunner(payload=payload))

    archive_root = Path(options.archive_root)
    planner_output = Path(batch.stdout_path)
    canonical = archive_root / bundle.bundle_name
    assert canonical.read_bytes() == payload
    assert planner_output.read_bytes() == payload
    assert planner_output.exists()

    kwargs = _complete_kwargs()
    kwargs["swiftstack_archive"] = bundle.bundle_name
    row = build_master_parquet_row(**kwargs)
    assert row is not None

    tracking = tmp_path / "tracking.parquet"
    pq.write_table(
        pa.table(
            {
                "source_run": ["r1"],
                "swiftstack_archive": [bundle.bundle_name],
            }
        ),
        tracking,
    )
    coverage = archives_for_dataset(tracking, "r1")
    assert coverage.archives == (bundle.bundle_name,)

    staging = plan_archive_staging(coverage, archive_root)
    assert staging.items[0].archive == bundle.bundle_name
    assert staging.items[0].present is True
    assert staging.items[0].action == "skip"


def test_publication_failure_is_closed_and_creates_no_artifact(tmp_path: Path) -> None:
    row = build_master_parquet_row(**_complete_kwargs())
    assert row is not None
    bad_row = dict(row)
    bad_row["swiftstack_archive"] = None

    path = tmp_path / "nested" / "master.parquet"
    with pytest.raises(ValueError, match="pending required field"):
        write_master_parquet([bad_row], path)

    assert not path.exists()
    assert not path.parent.exists()
