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

"""Phase-level tests for the non-executing folding lifecycle seam."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import ValidationError

from bspp.orchestration import contract as contract_surface
from bspp.orchestration.contract import folding_archive as archive_contract
from bspp.orchestration.contract import folding_checkpoint as checkpoint_contract
from bspp.orchestration.contract import folding_index as index_contract
from bspp.orchestration.contract import folding_lifecycle as lifecycle_contract
from bspp.orchestration.contract import folding_queue as queue_contract
from bspp.orchestration.contract.folding_archive import (
    FoldingResultAssociation,
    FoldingResultInventory,
    ResultStatus,
)
from bspp.orchestration.contract.folding_checkpoint import FoldingCheckpointState, FoldingFailureRecord
from bspp.orchestration.contract.folding_index import FoldingIndex, FoldingIndexRecord, make_folding_index
from bspp.orchestration.contract.folding_lifecycle import (
    FoldingArchiveLifecycleDecision,
    FoldingArchiveLifecycleRequest,
    FoldingCheckpointInputs,
    FoldingIndexLifecycleRequest,
    FoldingLifecycleConfig,
    FoldingManifestEvidence,
    FoldingMergeLifecycleRequest,
    FoldingPreflightLifecycleRequest,
    FoldingResumeLifecycleRequest,
    FoldingStatusLifecycleRequest,
    FoldingSubmitLifecycleRequest,
    folding_archive_lifecycle_decision_from_mapping,
    folding_archive_lifecycle_request_from_mapping,
    folding_index_lifecycle_decision_from_mapping,
    folding_index_lifecycle_request_from_mapping,
    folding_lifecycle_config_from_mapping,
    folding_merge_lifecycle_decision_from_mapping,
    folding_merge_lifecycle_request_from_mapping,
    folding_preflight_lifecycle_decision_from_mapping,
    folding_preflight_lifecycle_request_from_mapping,
    folding_resume_lifecycle_decision_from_mapping,
    folding_resume_lifecycle_request_from_mapping,
    folding_status_lifecycle_decision_from_mapping,
    folding_status_lifecycle_request_from_mapping,
    folding_submit_lifecycle_decision_from_mapping,
    folding_submit_lifecycle_request_from_mapping,
)
from bspp.orchestration.contract.versioning import UnsupportedSchemaVersionError
from bspp.orchestration.runtime import folding as folding_runtime_surface
from bspp.orchestration.runtime.folding import a3m as a3m_runtime
from bspp.orchestration.runtime.folding import archive as archive_runtime
from bspp.orchestration.runtime.folding import batch_info as batch_info_runtime
from bspp.orchestration.runtime.folding import checkpoint as checkpoint_runtime
from bspp.orchestration.runtime.folding import indexing as indexing_runtime
from bspp.orchestration.runtime.folding import lifecycle as lifecycle_runtime
from bspp.orchestration.runtime.folding import queue as queue_runtime
from bspp.orchestration.runtime.folding import queue_planning as queue_planning_runtime
from bspp.orchestration.runtime.folding import queue_render as queue_render_runtime
from bspp.orchestration.runtime.folding import results as results_runtime
from bspp.orchestration.runtime.folding import resume as resume_runtime
from bspp.orchestration.runtime.folding.lifecycle import (
    interpret_folding_status_lifecycle,
    plan_folding_archive_lifecycle,
    plan_folding_index_lifecycle,
    plan_folding_merge_lifecycle,
    plan_folding_resume_lifecycle,
    plan_folding_submit_lifecycle,
    preflight_folding_lifecycle,
)


def _config_mapping(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_name": "fold-smoke",
        "declared_a3m_inputs": ["/input/a.a3m", "/input/b.a3m"],
        "batch_info_path": "/input/batch_info.parquet",
        "output_dir": "/output/fold-smoke",
        "work_dir": "/output/fold-smoke/work",
        "predictions_root": "/output/fold-smoke/predictions",
        "checkpoint_dir": "/output/fold-smoke/checkpoints",
        "container_image": "/containers/openfold.sqsh",
        "trt_bionemo_dir": "/opt/trt-bionemo",
        "trt_engines_dir": "/models/trt-engines",
        "trt_checkpoint_path": "/models/params.pt",
        "model_preset": "model_1_multimer_v3",
        "max_recycling_iters": 4,
        "trt_fallback_threshold": 1536,
        "preprocessing_enabled": True,
        "sort_by_length": True,
        "index_worker_count": 16,
        "queue_strategy": "runtime_balanced",
        "queue_layout": "per_node",
        "node_count": 3,
        "gpus_per_node": 4,
        "worker_count": 3,
        "cpus_per_task": 16,
        "max_proteins_per_batch": 50,
        "job_name": "fold-smoke",
        "account": "project-account",
        "partition": "batch",
        "walltime": "04:00:00",
        "archive_enabled": True,
        "archive_run_tag": "fold-smoke",
        "archive_stage_root": "/scratch/archive",
        "archive_root": "/output/fold-smoke/archives",
        "proteins_per_archive": 5000,
        "archive_start_index": 0,
        "archive_max_archives": None,
        "archive_shuffle": True,
        "archive_shuffle_seed": 42,
        "lz4_executable": "lz4",
    }
    payload.update(overrides)
    return payload


def _folding_index(*, msa_root: Path | None = None) -> FoldingIndex:
    if msa_root is not None:
        msa_root.mkdir(parents=True)
        (msa_root / "model-a.a3m").write_text(">query\nAAAAA\n")
        (msa_root / "model-b.a3m").write_text(">query\nBBB\n")
    records = (
        FoldingIndexRecord(
            source_ordinal=0,
            protein_id="model-a",
            msa_path=str(msa_root / "model-a.a3m") if msa_root is not None else "/input/model-a.a3m",
            query_sequence="AAAAA",
            sequence_length=5,
            chain_lengths=(5,),
            chain_count=1,
            chain_cardinalities=(1,),
            msa_depth=1,
            total_length=5,
        ),
        FoldingIndexRecord(
            source_ordinal=1,
            protein_id="model-b",
            msa_path=str(msa_root / "model-b.a3m") if msa_root is not None else "/input/model-b.a3m",
            query_sequence="BBB",
            sequence_length=3,
            chain_lengths=(3,),
            chain_count=1,
            chain_cardinalities=(1,),
            msa_depth=1,
            total_length=3,
        ),
    )
    return make_folding_index(records)


def _ready_execution_config(tmp_path: Path, **overrides: object) -> FoldingLifecycleConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    batch_info = tmp_path / "batch_info.parquet"
    batch_info.write_bytes(b"fixture")
    container_image = tmp_path / "openfold.sqsh"
    container_image.write_bytes(b"fixture")
    trt_bionemo_dir = tmp_path / "trt-bionemo"
    trt_bionemo_dir.mkdir(exist_ok=True)
    trt_engines_dir = tmp_path / "trt-engines"
    trt_engines_dir.mkdir(exist_ok=True)
    trt_checkpoint = tmp_path / "params.pt"
    trt_checkpoint.write_bytes(b"fixture")
    ready_overrides: dict[str, object] = {
        "batch_info_path": str(batch_info),
        "container_image": str(container_image),
        "trt_bionemo_dir": str(trt_bionemo_dir),
        "trt_engines_dir": str(trt_engines_dir),
        "trt_checkpoint_path": str(trt_checkpoint),
    }
    ready_overrides.update(overrides)
    return folding_lifecycle_config_from_mapping(_config_mapping(**ready_overrides))


def _result_inventory(
    *,
    model_a: ResultStatus,
    model_b: ResultStatus,
    result_root: Path | None = None,
) -> FoldingResultInventory:
    associations = []
    for ordinal, (protein_id, status) in enumerate((("model-a", model_a), ("model-b", model_b))):
        if result_root is not None:
            result_root.mkdir(parents=True, exist_ok=True)
        pdb_path = str(result_root / f"{protein_id}.pdb") if result_root is not None else f"/results/{protein_id}.pdb"
        json_path = (
            str(result_root / f"{protein_id}.json") if result_root is not None else f"/results/{protein_id}.json"
        )
        if result_root is not None and status in {"complete", "pdb-only"}:
            Path(pdb_path).write_text("PDB")
        if result_root is not None and status in {"complete", "json-only"}:
            Path(json_path).write_text("{}")
        associations.append(
            FoldingResultAssociation(
                source_ordinal=ordinal,
                protein_id=protein_id,
                normalized_protein_id=protein_id,
                status=status,
                pdb_paths=(pdb_path,) if status in {"complete", "pdb-only"} else (),
                json_paths=(json_path,) if status in {"complete", "json-only"} else (),
            )
        )
    return FoldingResultInventory(associations=tuple(associations), unmatched=())


def test_folding_package_exports_cover_every_owned_public_symbol_deterministically() -> None:
    contract_modules = (
        archive_contract,
        checkpoint_contract,
        index_contract,
        lifecycle_contract,
        queue_contract,
    )
    contract_names = {name for module in contract_modules for name in module.__all__}
    assert contract_names <= set(contract_surface.__all__)
    assert len(contract_surface.__all__) == len(set(contract_surface.__all__))
    for name in contract_names:
        assert getattr(contract_surface, name) is not None

    runtime_modules = (
        a3m_runtime,
        archive_runtime,
        batch_info_runtime,
        checkpoint_runtime,
        indexing_runtime,
        lifecycle_runtime,
        queue_runtime,
        queue_planning_runtime,
        queue_render_runtime,
        results_runtime,
        resume_runtime,
    )
    runtime_names = {name for module in runtime_modules for name in module.__all__}
    assert set(folding_runtime_surface.__all__) == runtime_names
    assert len(folding_runtime_surface.__all__) == len(set(folding_runtime_surface.__all__))
    for name in runtime_names:
        assert getattr(folding_runtime_surface, name) is not None


def test_folding_lifecycle_config_is_frozen_fail_closed_and_resource_coherent() -> None:
    config = folding_lifecycle_config_from_mapping(_config_mapping())

    assert isinstance(config, FoldingLifecycleConfig)
    assert config.node_count == 3
    assert config.gpus_per_node == 4
    assert config.worker_count == 3
    assert config.model_dump(mode="json") == _config_mapping()

    with pytest.raises(ValidationError, match="frozen"):
        config.node_count = 2
    with pytest.raises(ValidationError, match="extra"):
        folding_lifecycle_config_from_mapping(_config_mapping(unexpected="no"))
    with pytest.raises(UnsupportedSchemaVersionError):
        folding_lifecycle_config_from_mapping(_config_mapping(schema_version=2))
    with pytest.raises(ValidationError, match="worker_count must equal node_count"):
        folding_lifecycle_config_from_mapping(_config_mapping(worker_count=12))

    legacy = folding_lifecycle_config_from_mapping(_config_mapping(queue_layout="legacy_per_gpu", worker_count=12))
    assert legacy.worker_count == legacy.node_count * legacy.gpus_per_node


def test_index_lifecycle_delegates_to_indexer_and_returns_typed_decision(tmp_path: Path) -> None:
    long_a3m = tmp_path / "long.a3m"
    long_a3m.write_text("#5 1\n>query\nABCDE\n>hit\nABCDE\n")
    short_a3m = tmp_path / "short.a3m"
    short_a3m.write_text("#3 1\n>query\nABC\n>hit\nABC\n")
    config = folding_lifecycle_config_from_mapping(_config_mapping(declared_a3m_inputs=[str(long_a3m), str(short_a3m)]))
    request = FoldingIndexLifecycleRequest(config=config)

    decision = plan_folding_index_lifecycle(request)

    assert tuple(record.protein_id for record in decision.index.records) == ("short", "long")
    assert decision.index_worker_count == 16
    assert decision.sort_by_length is True
    assert folding_index_lifecycle_request_from_mapping(request.to_mapping()) == request
    assert folding_index_lifecycle_decision_from_mapping(decision.to_mapping()) == decision
    with pytest.raises(FrozenInstanceError):
        decision.index_worker_count = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="unknown fields"):
        folding_index_lifecycle_request_from_mapping({**request.to_mapping(), "extra": "no"})

    directory_config = folding_lifecycle_config_from_mapping(_config_mapping(declared_a3m_inputs=[str(tmp_path)]))
    directory_decision = plan_folding_index_lifecycle(FoldingIndexLifecycleRequest(config=directory_config))
    assert tuple(record.protein_id for record in directory_decision.index.records) == ("short", "long")


def test_submit_lifecycle_plans_queue_views_without_authorizing_execution(tmp_path: Path) -> None:
    config = _ready_execution_config(tmp_path)
    request = FoldingSubmitLifecycleRequest(config=config, index=_folding_index(msa_root=tmp_path / "a3m"))

    decision = plan_folding_submit_lifecycle(request)

    assert tuple(assignment.protein_id for assignment in decision.queue_plan.assignments) == (
        "model-b",
        "model-a",
    )
    assert tuple(declaration.path for declaration in decision.queue_writes) == (
        "/output/fold-smoke/work/node0_batches.csv",
        "/output/fold-smoke/work/node1_batches.csv",
    )
    assert decision.submission_planned is True
    assert decision.execution_authorized is False
    assert folding_submit_lifecycle_request_from_mapping(request.to_mapping()) == request
    assert folding_submit_lifecycle_decision_from_mapping(decision.to_mapping()) == decision


def test_merge_lifecycle_keeps_checkpoint_layouts_explicit_and_global_views_highest_precedence(
    tmp_path: Path,
) -> None:
    node_completed = tmp_path / "completed_node_0.csv"
    node_completed.write_text(
        "protein_id,runtime_seconds,timestamp,node_id,gpu_id\n"
        "model-d,1,2026-01-01T00:00:00,node-0,0\n"
        "model-a,1,2026-01-01T00:00:00,node-0,0\n"
        "model-b,1,2026-01-01T00:00:00,node-0,0\n"
    )
    node_failed = tmp_path / "failed_node_0.csv"
    node_failed.write_text(
        "protein_id,error_message,timestamp,node_id,gpu_id\nmodel-c,TIMEOUT,2026-01-01T00:00:00,node-0,0\n"
    )
    current_gpu_completed = tmp_path / "completed_gpu_current_0.csv"
    current_gpu_completed.write_text(
        "protein_id,runtime_seconds,timestamp,node_id,gpu_id\nmodel-a,2,2026-01-01T00:01:00,node-0,0\n"
    )
    historical_gpu_completed = tmp_path / "completed_gpu_legacy_0.csv"
    historical_gpu_completed.write_text("protein_id,runtime_seconds,timestamp\nmodel-a,3,2026-01-01T00:02:00\n")
    merged_completed = tmp_path / "completed_all.csv"
    merged_completed.write_text(
        "protein_id,runtime_seconds,timestamp,node_id,gpu_id\n"
        "model-a,4,2026-01-01T00:03:00,node-global,0\n"
        "model-c,4,2026-01-01T00:03:00,node-global,0\n"
    )
    merged_failed = tmp_path / "failed_all.csv"
    merged_failed.write_text(
        "protein_id,error_message,timestamp,node_id,gpu_id\nmodel-b,EXIT_CODE,2026-01-01T00:03:00,node-global,0\n"
    )
    sources = FoldingCheckpointInputs(
        node_completion_shards=(str(node_completed),),
        node_failure_shards=(str(node_failed),),
        current_gpu_completion_shards=(str(current_gpu_completed),),
        current_gpu_failure_shards=(),
        historical_gpu_completion_shards=(str(historical_gpu_completed),),
        historical_gpu_failure_shards=(),
        merged_completion_view=str(merged_completed),
        merged_failure_view=str(merged_failed),
    )
    config = folding_lifecycle_config_from_mapping(_config_mapping(checkpoint_dir=str(tmp_path / "new-checkpoints")))
    request = FoldingMergeLifecycleRequest(config=config, checkpoint_inputs=sources)

    decision = plan_folding_merge_lifecycle(request)

    assert tuple(record.protein_id for record in decision.state.completions) == ("model-a", "model-c", "model-d")
    assert decision.state.completions[0].runtime_seconds == 4
    assert decision.state.completions[2].source_path == str(node_completed)
    assert tuple(record.protein_id for record in decision.state.failures) == ("model-b",)
    assert decision.completed_view_path == str(tmp_path / "new-checkpoints" / "completed_all.csv")
    assert decision.failed_view_path == str(tmp_path / "new-checkpoints" / "failed_all.csv")
    assert decision.write_planned is True
    assert not Path(decision.completed_view_path).exists()
    assert folding_merge_lifecycle_request_from_mapping(request.to_mapping()) == request
    assert folding_merge_lifecycle_decision_from_mapping(decision.to_mapping()) == decision

    duplicate = FoldingCheckpointInputs(
        node_completion_shards=(str(node_completed),),
        node_failure_shards=(),
        current_gpu_completion_shards=(str(node_completed),),
        current_gpu_failure_shards=(),
        historical_gpu_completion_shards=(),
        historical_gpu_failure_shards=(),
        merged_completion_view=None,
        merged_failure_view=None,
    )
    with pytest.raises(ValueError, match="more than one lifecycle checkpoint slot"):
        plan_folding_merge_lifecycle(FoldingMergeLifecycleRequest(config=config, checkpoint_inputs=duplicate))


def test_resume_lifecycle_combines_checkpoint_and_only_complete_result_recovery(tmp_path: Path) -> None:
    config = _ready_execution_config(tmp_path)
    index = _folding_index(msa_root=tmp_path / "a3m")
    checkpoint = FoldingCheckpointState(
        completions=(),
        failures=(
            FoldingFailureRecord(
                protein_id="model-a",
                error_message="CUDA_OUT_OF_MEMORY",
                timestamp="2026-01-01T00:00:00",
                node_id="node-0",
                gpu_id=0,
                source_layout="current-per-node",
                source_path="/checkpoints/failed_node_0.csv",
                source_ordinal=0,
                row_number=2,
            ),
            FoldingFailureRecord(
                protein_id="model-b",
                error_message="TIMEOUT",
                timestamp="2026-01-01T00:01:00",
                node_id="node-0",
                gpu_id=1,
                source_layout="current-per-node",
                source_path="/checkpoints/failed_node_0.csv",
                source_ordinal=0,
                row_number=3,
            ),
        ),
    )
    partial_request = FoldingResumeLifecycleRequest(
        config=config,
        index=index,
        checkpoint_state=checkpoint,
        result_inventory=_result_inventory(
            model_a="complete",
            model_b="pdb-only",
            result_root=tmp_path / "results",
        ),
        retry_failed=False,
    )

    partial = plan_folding_resume_lifecycle(partial_request)

    assert partial.recovered_result_ids == ("model-a",)
    assert partial.remaining_work.completed_model_ids == ("model-a",)
    assert partial.remaining_work.transient_failure_ids == ("model-b",)
    assert partial.remaining_work.permanent_failure_ids == ()
    assert partial.remaining_work.remaining_model_ids == ("model-b",)
    assert tuple(item.protein_id for item in partial.submit_decision.queue_plan.assignments) == ("model-b",)
    assert partial.submit_decision.submission_planned is True
    assert folding_resume_lifecycle_request_from_mapping(partial_request.to_mapping()) == partial_request
    assert folding_resume_lifecycle_decision_from_mapping(partial.to_mapping()) == partial

    complete = plan_folding_resume_lifecycle(
        FoldingResumeLifecycleRequest(
            config=config,
            index=index,
            checkpoint_state=checkpoint,
            result_inventory=_result_inventory(
                model_a="complete",
                model_b="complete",
                result_root=tmp_path / "results",
            ),
            retry_failed=False,
        )
    )
    assert complete.remaining_work.remaining_model_ids == ()
    assert complete.submit_decision.queue_plan.assignments == ()
    assert complete.submit_decision.queue_writes == ()
    assert complete.submit_decision.submission_planned is False


def test_archive_lifecycle_explicitly_scans_results_and_returns_only_nonexecuting_plans(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for protein_id in ("model-a", "model-b"):
        (predictions / f"{protein_id}_unrelaxed_rank_001_model.pdb").write_text("PDB")
        (predictions / f"{protein_id}_scores_rank_001_model.json").write_text("{}")
    (predictions / "config.json").write_text("{}")
    archive_root = tmp_path / "archives"
    stage_root = tmp_path / "stage"
    config = folding_lifecycle_config_from_mapping(
        _config_mapping(
            predictions_root=str(predictions),
            archive_stage_root=str(stage_root),
            archive_root=str(archive_root),
            proteins_per_archive=1,
            archive_shuffle=False,
        )
    )
    request = FoldingArchiveLifecycleRequest(
        config=config,
        index=_folding_index(),
        result_inventory=None,
        scan_predictions_root=str(predictions),
        selected_protein_ids=None,
        prior_manifest=(),
        force=False,
    )

    decision = plan_folding_archive_lifecycle(request)

    assert tuple(item.status for item in decision.result_inventory.associations) == ("complete", "complete")
    assert tuple(item.reason for item in decision.result_inventory.unmatched) == ("unplanned-identity",)
    assert tuple(batch.protein_ids for batch in decision.archive_plan.batches) == (("model-a",), ("model-b",))
    assert decision.manifest_jsonl.count("\n") == 2
    assert decision.execution_authorized is False
    assert not archive_root.exists()
    assert not stage_root.exists()
    archive_preflight = preflight_folding_lifecycle(
        FoldingPreflightLifecycleRequest(
            action="archive",
            config=config,
            index=request.index,
            checkpoint_inputs=None,
            checkpoint_state=None,
            result_inventory=decision.result_inventory,
            scan_predictions_root=None,
        )
    )
    result_check = next(check for check in archive_preflight.checks if check.name == "result")
    assert (result_check.status, result_check.detail) == (
        "warning",
        "2 valid complete result associations; 1 unplanned or malformed result files remain informational",
    )
    assert archive_preflight.can_plan is True
    assert folding_archive_lifecycle_request_from_mapping(request.to_mapping()) == request
    assert folding_archive_lifecycle_decision_from_mapping(decision.to_mapping()) == decision

    tampered_manifest = decision.to_mapping()
    tampered_manifest["manifest_jsonl"] = "not-json\n" * len(decision.archive_plan.batches)
    with pytest.raises(ValueError, match="not valid JSON"):
        folding_archive_lifecycle_decision_from_mapping(tampered_manifest)
    mismatched_manifest = decision.to_mapping()
    mismatched_manifest["manifest_jsonl"] = decision.manifest_jsonl.replace("fold-smoke", "other-run", 1)
    with pytest.raises(ValueError, match="does not match its archive batch"):
        folding_archive_lifecycle_decision_from_mapping(mismatched_manifest)
    with pytest.raises(ValueError, match="only complete result associations"):
        FoldingArchiveLifecycleDecision(
            result_inventory=_result_inventory(model_a="missing", model_b="complete"),
            archive_plan=decision.archive_plan,
            manifest_jsonl=decision.manifest_jsonl,
        )


def test_preflight_surfaces_phase_prerequisites_before_planning(tmp_path: Path) -> None:
    config = folding_lifecycle_config_from_mapping(_config_mapping())
    missing = FoldingPreflightLifecycleRequest(
        action="resume",
        config=config,
        index=None,
        checkpoint_inputs=None,
        checkpoint_state=None,
        result_inventory=None,
        scan_predictions_root=None,
    )

    blocked = preflight_folding_lifecycle(missing)

    assert tuple((check.name, check.status) for check in blocked.checks) == (
        ("config", "fail"),
        ("index", "fail"),
        ("checkpoint", "fail"),
        ("result", "fail"),
        ("archive", "not-required"),
    )
    assert blocked.can_plan is False
    assert folding_preflight_lifecycle_request_from_mapping(missing.to_mapping()) == missing
    assert folding_preflight_lifecycle_decision_from_mapping(blocked.to_mapping()) == blocked
    with pytest.raises(ValueError, match="folding submit preflight failed"):
        plan_folding_submit_lifecycle(FoldingSubmitLifecycleRequest(config=config, index=_folding_index()))

    config = _ready_execution_config(tmp_path)
    index = _folding_index(msa_root=tmp_path / "a3m")
    ready = preflight_folding_lifecycle(
        FoldingPreflightLifecycleRequest(
            action="submit",
            config=config,
            index=index,
            checkpoint_inputs=None,
            checkpoint_state=None,
            result_inventory=None,
            scan_predictions_root=None,
        )
    )
    assert ready.can_plan is True
    assert tuple(check.status for check in ready.checks) == (
        "warning",
        "pass",
        "not-required",
        "not-required",
        "not-required",
    )
    assert plan_folding_submit_lifecycle(FoldingSubmitLifecycleRequest(config=config, index=index)).submission_planned

    stale_batch_config = _ready_execution_config(
        tmp_path / "stale-batch-config",
        batch_info_path=str(tmp_path / "does-not-exist.parquet"),
    )
    assert plan_folding_submit_lifecycle(
        FoldingSubmitLifecycleRequest(config=stale_batch_config, index=index)
    ).submission_planned

    missing_result_files = preflight_folding_lifecycle(
        FoldingPreflightLifecycleRequest(
            action="archive",
            config=config,
            index=index,
            checkpoint_inputs=None,
            checkpoint_state=None,
            result_inventory=_result_inventory(model_a="complete", model_b="complete"),
            scan_predictions_root=None,
        )
    )
    assert next(check for check in missing_result_files.checks if check.name == "result").status == "fail"
    assert missing_result_files.can_plan is False

    archive_disabled = folding_lifecycle_config_from_mapping(
        _config_mapping(
            archive_enabled=False,
            archive_run_tag=None,
            archive_stage_root=None,
            archive_root=None,
        )
    )
    archive_blocked = preflight_folding_lifecycle(
        FoldingPreflightLifecycleRequest(
            action="archive",
            config=archive_disabled,
            index=index,
            checkpoint_inputs=None,
            checkpoint_state=None,
            result_inventory=_result_inventory(
                model_a="complete",
                model_b="complete",
                result_root=tmp_path / "preflight-results",
            ),
            scan_predictions_root=None,
        )
    )
    assert tuple((check.name, check.status) for check in archive_blocked.checks)[-2:] == (
        ("result", "pass"),
        ("archive", "fail"),
    )
    assert archive_blocked.can_plan is False


def test_status_lifecycle_interprets_only_caller_supplied_local_evidence() -> None:
    config = folding_lifecycle_config_from_mapping(_config_mapping())
    index = _folding_index()
    initial_request = FoldingStatusLifecycleRequest(
        config=config,
        index=index,
        manifest=None,
        checkpoint_state=None,
        result_inventory=None,
        archive_plan=None,
    )

    initial = interpret_folding_status_lifecycle(initial_request)

    assert initial.state == "initial"
    assert initial.planned_model_count == 2
    assert initial.completed_model_count == 0
    assert folding_status_lifecycle_request_from_mapping(initial_request.to_mapping()) == initial_request
    assert folding_status_lifecycle_decision_from_mapping(initial.to_mapping()) == initial

    impossible_initial = initial.to_mapping()
    impossible_initial.update(
        evidence_kinds=["manifest", "checkpoint", "archive"],
        manifest_model_count=2,
        checkpoint_failed_count=1,
        planned_archive_count=1,
    )
    with pytest.raises(ValueError, match="initial state cannot contain"):
        folding_status_lifecycle_decision_from_mapping(impossible_initial)

    partial_request = FoldingStatusLifecycleRequest(
        config=config,
        index=index,
        manifest=FoldingManifestEvidence(total_proteins=2, queue_artifact_count=2, queue_layout="per_node"),
        checkpoint_state=FoldingCheckpointState(completions=(), failures=()),
        result_inventory=_result_inventory(model_a="complete", model_b="missing"),
        archive_plan=None,
    )
    partial = interpret_folding_status_lifecycle(partial_request)
    assert partial.state == "partial"
    assert partial.recovered_result_count == 1
    assert partial.completed_model_count == 1
    assert partial.incomplete_result_count == 1

    complete = interpret_folding_status_lifecycle(
        FoldingStatusLifecycleRequest(
            config=config,
            index=index,
            manifest=partial_request.manifest,
            checkpoint_state=partial_request.checkpoint_state,
            result_inventory=_result_inventory(model_a="complete", model_b="complete"),
            archive_plan=None,
        )
    )
    assert complete.state == "complete"
    assert complete.completed_model_count == 2

    invalid = interpret_folding_status_lifecycle(
        FoldingStatusLifecycleRequest(
            config=config,
            index=index,
            manifest=FoldingManifestEvidence(total_proteins=3, queue_artifact_count=2, queue_layout="per_node"),
            checkpoint_state=None,
            result_inventory=None,
            archive_plan=None,
        )
    )
    assert invalid.state == "invalid"
    assert invalid.invalid_evidence == ("manifest total does not match folding index",)

    colliding_records = (
        FoldingIndexRecord(
            source_ordinal=0,
            protein_id="AF_1x",
            msa_path="/input/AF_1x.a3m",
            query_sequence="A",
            sequence_length=1,
            chain_lengths=(1,),
            chain_count=1,
            chain_cardinalities=(1,),
            msa_depth=1,
            total_length=1,
        ),
        FoldingIndexRecord(
            source_ordinal=1,
            protein_id="AF-1x",
            msa_path="/input/AF-1x.a3m",
            query_sequence="B",
            sequence_length=1,
            chain_lengths=(1,),
            chain_count=1,
            chain_cardinalities=(1,),
            msa_depth=1,
            total_length=1,
        ),
    )
    colliding_index = make_folding_index(colliding_records)
    colliding_inventory = FoldingResultInventory(
        associations=tuple(
            FoldingResultAssociation(
                source_ordinal=record.source_ordinal,
                protein_id=record.protein_id,
                normalized_protein_id="AF-1x",
                status="complete",
                pdb_paths=(f"/results/{record.protein_id}.pdb",),
                json_paths=(f"/results/{record.protein_id}.json",),
            )
            for record in colliding_records
        ),
        unmatched=(),
    )
    colliding = interpret_folding_status_lifecycle(
        FoldingStatusLifecycleRequest(
            config=config,
            index=colliding_index,
            manifest=None,
            checkpoint_state=None,
            result_inventory=colliding_inventory,
            archive_plan=None,
        )
    )
    assert colliding.state == "invalid"
    assert colliding.invalid_evidence == ("folding index identities collide after checkpoint normalization",)
