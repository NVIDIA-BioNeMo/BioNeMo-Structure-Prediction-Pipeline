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

"""Per-restart postprocessing classification evidence tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bspp.orchestration.contract.phase import PhaseMountSnapshot, PhaseSlurmResources
from bspp.orchestration.contract.postprocessing_acceptance_reference import (
    PostprocessingAcceptanceSnapshotReference,
)
from bspp.orchestration.contract.postprocessing_action_contract import (
    PostprocessingActionSemanticsV2,
    PostprocessingActionSemanticV2,
    PostprocessingRuntimeAction,
    normalize_slurm_array,
    postprocessing_action_graph_digest,
)
from bspp.orchestration.contract.postprocessing_execution import (
    PostprocessingAttemptPaths,
    PostprocessingClusterSnapshot,
    PostprocessingExecutionProjection,
    PostprocessingPhaseExecutionIdentity,
    QualifiedPostprocessingRuntimeSelection,
)
from bspp.orchestration.contract.postprocessing_failure_classification import PostprocessingFailureClassification
from bspp.orchestration.contract.postprocessing_logical_identity import (
    PhysicalInputLocator,
    PostprocessingLogicalInputIdentityManifestV2,
    PostprocessingLogicalInputIdentityV2,
    PostprocessingScientificIdentityV2,
    PostprocessingSemanticField,
)
from bspp.orchestration.contract.postprocessing_phase_ids import (
    POSTPROCESSING_ACTION_IDS,
    postprocessing_action_id,
)
from bspp.orchestration.contract.postprocessing_restart_classification import (
    PostprocessingRestartClassificationEvidence,
    postprocessing_restart_classification_evidence_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runspec_v2 import (
    PostprocessingPhaseRunSpec,
    PostprocessingPhaseRunSpecPayload,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from bspp.orchestration.runtime.postprocessing.restart_classification import (
    record_restart_classification,
)

_STEPS = (
    "preflight",
    "recipe",
    "preprocess",
    "slurm",
    "analysis-finalize",
    "acceptance-tar-payload-parity",
    "acceptance-semantic",
    "acceptance-verify-evidence",
    "acceptance-adjudication",
)
_DEPENDENCIES = {
    "preflight": (),
    "recipe": ("preflight",),
    "preprocess": ("recipe",),
    "slurm": ("preprocess",),
    "analysis-finalize": ("slurm",),
    "acceptance-tar-payload-parity": ("analysis-finalize",),
    "acceptance-semantic": ("analysis-finalize",),
    "acceptance-verify-evidence": ("acceptance-tar-payload-parity", "acceptance-semantic"),
    "acceptance-adjudication": (
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    ),
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _classification(
    group: str = "A",
    failure_kind: str = "timeout",
    reason: str = "audited timeout",
) -> PostprocessingFailureClassification:
    return PostprocessingFailureClassification(group=group, failure_kind=failure_kind, reason=reason)


def _evidence(**overrides: object) -> PostprocessingRestartClassificationEvidence:
    fields: dict[str, object] = {
        "phase_run_id": "phase-run-" + _sha("phase-run")[:32],
        "attempt_id": "attempt-0001",
        "phase_runspec_digest": _sha("runspec"),
        "action_graph_digest": _sha("graph"),
        "action_id": POSTPROCESSING_ACTION_IDS["preflight"],
        "runtime_action_digest": _sha("action"),
        "task_index": None,
        "restart_ordinal": 0,
        "classification": _classification(),
        "classified_at": "2026-09-11T00:00:00Z",
    }
    fields.update(overrides)
    return PostprocessingRestartClassificationEvidence(**fields)


def _write_runspec(tmp_path: Path, *, max_batch_requeue: int | None) -> Path:
    phase_run_id = "phase-run-" + _sha("phase-run")[:32]
    attempt_id = "attempt-0001"
    phase_plan_digest = _sha("phase-plan")
    semantic_digest = _sha("acceptance-semantic")

    actions = tuple(
        PostprocessingRuntimeAction(
            action_id=postprocessing_action_id(step),
            step_name=step,
            dependencies=tuple(postprocessing_action_id(dep) for dep in _DEPENDENCIES[step]),
            resources=PhaseSlurmResources(
                partition="x",
                cpus_per_task=1,
                memory="1G",
                time="1:00:00",
                array="1" if step == "slurm" else None,
            ),
            normalized_array=normalize_slurm_array("1" if step == "slurm" else None)[0],
            expected_task_indexes=normalize_slurm_array("1" if step == "slurm" else None)[1],
        )
        for step in _STEPS
    )
    action_graph_digest = postprocessing_action_graph_digest(actions)

    logical_inputs = PostprocessingLogicalInputIdentityManifestV2(
        entries=(
            PostprocessingLogicalInputIdentityV2(
                name="input",
                member_identity="input-member",
                expected_content_sha256=_sha("logical-input-content"),
                expected_size_bytes=1,
            ),
        )
    )
    scientific_identity = PostprocessingScientificIdentityV2(
        dataset_scope_digest=_sha("dataset-scope"),
        scientific_parameters_digest=_sha("scientific-parameters"),
        logical_input_identity_digest=logical_inputs.digest,
    )
    acceptance_policy = PostprocessingAcceptanceSnapshotReference(
        location="attempts/attempt-0001/acceptance-policy.json",
        sha256=_sha("acceptance-policy-sha"),
        size_bytes=10,
        semantic_digest=semantic_digest,
        policy_id=f"postprocessing-acceptance-policy-{semantic_digest}",
    )
    semantic_field = PostprocessingSemanticField(path="mode", value="archive")
    action_semantics = PostprocessingActionSemanticsV2(
        scientific_identity_digest=scientific_identity.digest,
        logical_input_identity_digest=logical_inputs.digest,
        acceptance_semantic_digest=semantic_digest,
        semantic_fields=(semantic_field,),
        actions=tuple(
            PostprocessingActionSemanticV2(
                action_id=action.action_id,
                step_name=action.step_name,
                mode=None,
                dependencies=action.dependencies,
                normalized_task_scope=action.expected_task_indexes,
                command_contract=f"postprocessing-{action.step_name}-v1",
                normalized_command_sha256=_sha(f"command-{action.step_name}"),
                normalized_arguments=(semantic_field,),
            )
            for action in actions
        ),
    )
    qualified_runtime = QualifiedPostprocessingRuntimeSelection(
        tuple_id=_sha("tuple"),
        qualification_location="/qualification.json",
        qualification_sha256=_sha("qualification"),
        qualification_size_bytes=10,
        qualified_at="2026-09-11T00:00:00Z",
        expires_at="2026-09-12T00:00:00Z",
        image_path="/image.sqsh",
        image_sha256=_sha("image"),
        image_size_bytes=10,
        image_policy="digest-checked",
        source_kind="baked",
        source_revision="0" * 40,
        source_package_path="/source.tar",
        toolkit_package_path=None,
        runtime_ipsae_binary_path="/ipsae",
        runtime_ipsae_binary_sha256=_sha("ipsae"),
        runtime_ipsae_binary_size_bytes=10,
        source_identity_digest=_sha("source-identity"),
        source_package_identity_digest=_sha("source-package"),
        toolkit_identity_digest=_sha("toolkit"),
        bootstrap_sha256=_sha("bootstrap"),
        runtime_component_identity_digest=_sha("runtime-component"),
        requeue_exit=85 if max_batch_requeue is not None else None,
        max_batch_requeue=max_batch_requeue,
    )
    attempt_paths = PostprocessingAttemptPaths(
        legacy_run_id="legacy-run",
        output_dir=str(tmp_path / "output"),
        evidence_dir=str(tmp_path / "evidence"),
        staging_dir=str(tmp_path / "staging"),
        object_prefix="s3://bucket/prefix",
    )
    cluster = PostprocessingClusterSnapshot(
        profile_name="profile",
        owner="owner",
        transport="ssh",
        ssh_target="cluster",
        account="account",
        project_root="/project",
        staging_root="/staging",
        orchestration_repo="/repo",
        runtime_image=qualified_runtime.image_path,
        extra_mounts=(PhaseMountSnapshot(source="/src", target="/tgt"),),
    )
    phase_identity = PostprocessingPhaseExecutionIdentity(
        phase_plan_digest=phase_plan_digest,
        logical_input_manifest_digest=logical_inputs.digest,
        scientific_identity_digest=scientific_identity.digest,
        action_semantics_digest=action_semantics.digest,
        acceptance_semantic_digest=semantic_digest,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        qualified_runtime_digest=qualified_runtime.digest,
        output_namespace="namespace",
        substitutions=attempt_paths,
    )
    projection = PostprocessingExecutionProjection(
        document_location="attempts/attempt-0001/legacy-runspec.yaml",
        document_sha256=_sha("legacy-runspec-document"),
        document_size_bytes=10,
        legacy_schema_version=CURRENT_CONTRACT_SCHEMA_VERSION,
        phase_identity=phase_identity,
        phase_identity_digest=phase_identity.digest,
    )
    payload = PostprocessingPhaseRunSpecPayload(
        actions=actions,
        action_graph_digest=action_graph_digest,
        action_semantics_digest=action_semantics.digest,
        action_semantics=action_semantics,
        scientific_identity=scientific_identity,
        logical_inputs=logical_inputs,
        physical_inputs=(PhysicalInputLocator(name="input", locator="/loc", purpose="input"),),
        execution_projection=projection,
        acceptance_policy=acceptance_policy,
        qualified_runtime=qualified_runtime,
        attempt_paths=attempt_paths,
    )
    runspec = PostprocessingPhaseRunSpec(
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        phase_plan_digest=phase_plan_digest,
        materialized_at="2026-09-11T00:00:00Z",
        cluster=cluster,
        payload=payload,
    )
    path = tmp_path / "phase-runspec.json"
    path.write_bytes((json.dumps(runspec.to_mapping(), indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode())
    return path


def test_contract_rejects_group_b_classification() -> None:
    with pytest.raises(ValueError, match="group-A"):
        _evidence(classification=_classification(group="B"))


def test_contract_rejects_negative_restart_ordinal() -> None:
    with pytest.raises(ValueError, match="restart ordinal"):
        _evidence(restart_ordinal=-1)


def test_contract_rejects_bool_restart_ordinal() -> None:
    with pytest.raises(ValueError, match="restart ordinal"):
        _evidence(restart_ordinal=True)


def test_contract_rejects_acceptance_adjudication_action_id() -> None:
    with pytest.raises(ValueError, match="Actions 01--08"):
        _evidence(action_id=POSTPROCESSING_ACTION_IDS["acceptance-adjudication"])


def test_contract_from_mapping_rejects_unknown_field() -> None:
    mapping = _evidence().to_mapping()
    mapping["extra"] = "x"
    with pytest.raises(ValueError, match="unknown"):
        postprocessing_restart_classification_evidence_from_mapping(mapping)


def test_contract_rejects_unsupported_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        _evidence(schema_version=2)


def test_contract_round_trips() -> None:
    evidence = _evidence()
    assert postprocessing_restart_classification_evidence_from_mapping(evidence.to_mapping()) == evidence


def test_contract_rejects_bool_task_index() -> None:
    with pytest.raises(ValueError, match="task index"):
        _evidence(task_index=True)


def test_contract_rejects_negative_task_index() -> None:
    with pytest.raises(ValueError, match="task index"):
        _evidence(task_index=-1)


def test_contract_rejects_empty_classified_at() -> None:
    with pytest.raises(ValueError, match="classified_at"):
        _evidence(classified_at="")


def test_contract_is_frozen() -> None:
    evidence = _evidence()
    with pytest.raises(FrozenInstanceError):
        evidence.restart_ordinal = 1  # type: ignore[misc]


def test_writer_writes_one_record_per_action_task_ordinal(tmp_path: Path) -> None:
    runspec_path = _write_runspec(tmp_path, max_batch_requeue=2)
    command_digest = _sha("command")
    classification = _classification()

    preflight = record_restart_classification(
        phase_runspec_path=runspec_path,
        action_id=POSTPROCESSING_ACTION_IDS["preflight"],
        command_digest=command_digest,
        classification=classification,
    )
    assert preflight.restart_ordinal == 0
    assert preflight.task_index is None

    slurm = record_restart_classification(
        phase_runspec_path=runspec_path,
        action_id=POSTPROCESSING_ACTION_IDS["slurm"],
        command_digest=command_digest,
        classification=classification,
        task_index=1,
        restart_count_env="1",
    )
    assert slurm.restart_ordinal == 1
    assert slurm.task_index == 1

    evidence_root = tmp_path / "evidence" / "phase-actions/restarts"
    assert (evidence_root / POSTPROCESSING_ACTION_IDS["preflight"] / "restart-0000000000-classification.json").is_file()
    assert (
        evidence_root / POSTPROCESSING_ACTION_IDS["slurm"] / "restart-0000000001-0000000001-classification.json"
    ).is_file()


def test_writer_derives_zero_when_restart_count_absent(tmp_path: Path) -> None:
    runspec_path = _write_runspec(tmp_path, max_batch_requeue=3)
    evidence = record_restart_classification(
        phase_runspec_path=runspec_path,
        action_id=POSTPROCESSING_ACTION_IDS["preflight"],
        command_digest=_sha("command"),
        classification=_classification(),
    )
    assert evidence.restart_ordinal == 0
    assert (
        tmp_path
        / "evidence"
        / "phase-actions/restarts"
        / POSTPROCESSING_ACTION_IDS["preflight"]
        / "restart-0000000000-classification.json"
    ).is_file()


@pytest.mark.parametrize("value", ["abc", "-1", "1.5", ""])
def test_writer_rejects_non_digit_restart_count(tmp_path: Path, value: str) -> None:
    runspec_path = _write_runspec(tmp_path, max_batch_requeue=3)
    with pytest.raises(ValueError, match="non-negative integer"):
        record_restart_classification(
            phase_runspec_path=runspec_path,
            action_id=POSTPROCESSING_ACTION_IDS["preflight"],
            command_digest=_sha("command"),
            classification=_classification(),
            restart_count_env=value,
        )


def test_writer_bounds_ordinal_by_cap(tmp_path: Path) -> None:
    runspec_path = _write_runspec(tmp_path, max_batch_requeue=2)
    ok = record_restart_classification(
        phase_runspec_path=runspec_path,
        action_id=POSTPROCESSING_ACTION_IDS["preflight"],
        command_digest=_sha("command"),
        classification=_classification(),
        restart_count_env="2",
    )
    assert ok.restart_ordinal == 2
    with pytest.raises(ValueError, match="exceeds"):
        record_restart_classification(
            phase_runspec_path=runspec_path,
            action_id=POSTPROCESSING_ACTION_IDS["preflight"],
            command_digest=_sha("command"),
            classification=_classification(),
            restart_count_env="3",
        )


def test_writer_reconciles_identical_replay(tmp_path: Path) -> None:
    runspec_path = _write_runspec(tmp_path, max_batch_requeue=2)
    kwargs = {
        "phase_runspec_path": runspec_path,
        "action_id": POSTPROCESSING_ACTION_IDS["preflight"],
        "command_digest": _sha("command"),
        "classification": _classification(),
        "restart_count_env": "1",
    }
    first = record_restart_classification(**kwargs)
    second = record_restart_classification(**kwargs)
    assert second == first
    assert second.classified_at == first.classified_at
    directory = tmp_path / "evidence" / "phase-actions/restarts" / POSTPROCESSING_ACTION_IDS["preflight"]
    assert sorted(path.name for path in directory.iterdir()) == ["restart-0000000001-classification.json"]


def test_writer_rejects_collision(tmp_path: Path) -> None:
    runspec_path = _write_runspec(tmp_path, max_batch_requeue=2)
    action_id = POSTPROCESSING_ACTION_IDS["preflight"]
    record_restart_classification(
        phase_runspec_path=runspec_path,
        action_id=action_id,
        command_digest=_sha("command"),
        classification=_classification(reason="first"),
        restart_count_env="1",
    )
    with pytest.raises(ValueError, match="differs"):
        record_restart_classification(
            phase_runspec_path=runspec_path,
            action_id=action_id,
            command_digest=_sha("command"),
            classification=_classification(reason="second"),
            restart_count_env="1",
        )


def test_writer_rejects_group_b_classification(tmp_path: Path) -> None:
    runspec_path = _write_runspec(tmp_path, max_batch_requeue=2)
    with pytest.raises(ValueError, match="group-A"):
        record_restart_classification(
            phase_runspec_path=runspec_path,
            action_id=POSTPROCESSING_ACTION_IDS["preflight"],
            command_digest=_sha("command"),
            classification=_classification(group="B"),
        )


def test_writer_rejects_acceptance_adjudication_action_id(tmp_path: Path) -> None:
    runspec_path = _write_runspec(tmp_path, max_batch_requeue=2)
    with pytest.raises(ValueError, match="Action 09"):
        record_restart_classification(
            phase_runspec_path=runspec_path,
            action_id=POSTPROCESSING_ACTION_IDS["acceptance-adjudication"],
            command_digest=_sha("command"),
            classification=_classification(),
        )


def test_writer_rejects_missing_cap(tmp_path: Path) -> None:
    runspec_path = _write_runspec(tmp_path, max_batch_requeue=None)
    with pytest.raises(ValueError, match="max_batch_requeue"):
        record_restart_classification(
            phase_runspec_path=runspec_path,
            action_id=POSTPROCESSING_ACTION_IDS["preflight"],
            command_digest=_sha("command"),
            classification=_classification(),
        )
