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

"""Runtime transport-failure adapter and exit-85 helper tests."""

from __future__ import annotations

import errno
import hashlib
import json
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
from bspp.orchestration.contract.postprocessing_failure_classification import (
    PostprocessingTransportFailureObservation,
)
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
from bspp.orchestration.contract.postprocessing_runspec_v2 import (
    PostprocessingPhaseRunSpec,
    PostprocessingPhaseRunSpecPayload,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from bspp.orchestration.runtime.data_movement.common import TransferResult
from bspp.orchestration.runtime.postprocessing.failure_adapter import (
    TransportFailure,
    classify_and_exit_for_autorequeue,
    classify_audited_transport_failure,
    raise_transport_failure_if_audited,
    transport_failure_from_http_status,
    transport_failure_from_oserror,
    transport_failure_from_result,
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

_AUDITED_KINDS = (
    "timeout",
    "connection-reset",
    "temporary-dns",
    "http-500",
    "http-502",
    "http-503",
    "http-504",
)
_NON_AUDITED_KINDS = ("oom", "signal", "credential-error", "unknown")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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


@pytest.mark.parametrize("kind", _AUDITED_KINDS)
def test_classify_maps_audited_kinds(kind: str) -> None:
    failure = TransportFailure(kind=kind, detail="audited failure")  # type: ignore[arg-type]
    observation = classify_audited_transport_failure(failure)
    assert observation is not None
    assert isinstance(observation, PostprocessingTransportFailureObservation)
    assert observation.failure_kind == kind
    assert observation.failure_detail == "audited failure"


@pytest.mark.parametrize("kind", _NON_AUDITED_KINDS)
def test_classify_returns_none_for_non_audited_kinds(kind: str) -> None:
    failure = TransportFailure(kind=kind, detail="non-audited failure")  # type: ignore[arg-type]
    assert classify_audited_transport_failure(failure) is None


@pytest.mark.parametrize("detail", ["", "   ", "multi\nline", "trailing \r"])
def test_transport_failure_rejects_bad_detail(detail: str) -> None:
    with pytest.raises(ValueError, match="detail"):
        TransportFailure(kind="timeout", detail=detail)


def test_transport_failure_carries_detail_as_str() -> None:
    failure = TransportFailure(kind="timeout", detail="connection timed out")
    assert str(failure) == "connection timed out"
    assert failure.kind == "timeout"
    assert failure.detail == "connection timed out"


def test_exit_85_writes_one_record_and_exits_for_group_a(tmp_path: Path) -> None:
    runspec_path = _write_runspec(tmp_path, max_batch_requeue=2)
    action_id = POSTPROCESSING_ACTION_IDS["preflight"]
    failure = TransportFailure(kind="timeout", detail="connection timed out")

    with pytest.raises(SystemExit) as exc_info:
        classify_and_exit_for_autorequeue(
            failure,
            phase_runspec_path=runspec_path,
            action_id=action_id,
            command_digest=_sha("command"),
        )

    assert exc_info.value.code == 85
    evidence_root = tmp_path / "evidence" / "phase-actions/restarts" / action_id
    assert (evidence_root / "restart-0000000000-classification.json").is_file()
    assert sorted(path.name for path in evidence_root.iterdir()) == ["restart-0000000000-classification.json"]


def test_exit_85_never_writes_success_evidence(tmp_path: Path) -> None:
    runspec_path = _write_runspec(tmp_path, max_batch_requeue=2)
    action_id = POSTPROCESSING_ACTION_IDS["preflight"]
    failure = TransportFailure(kind="http-503", detail="HTTP 503 from s5cmd")

    with pytest.raises(SystemExit) as exc_info:
        classify_and_exit_for_autorequeue(
            failure,
            phase_runspec_path=runspec_path,
            action_id=action_id,
            command_digest=_sha("command"),
        )

    assert exc_info.value.code == 85
    evidence_root = tmp_path / "evidence" / "phase-actions/restarts" / action_id
    assert not (evidence_root / "task.json").exists()
    assert (evidence_root / "restart-0000000000-classification.json").is_file()


@pytest.mark.parametrize("kind", _NON_AUDITED_KINDS)
def test_group_b_returns_without_exit_or_record(tmp_path: Path, kind: str) -> None:
    runspec_path = _write_runspec(tmp_path, max_batch_requeue=2)
    action_id = POSTPROCESSING_ACTION_IDS["preflight"]
    failure = TransportFailure(kind=kind, detail="non-audited failure")  # type: ignore[arg-type]

    result = classify_and_exit_for_autorequeue(
        failure,
        phase_runspec_path=runspec_path,
        action_id=action_id,
        command_digest=_sha("command"),
    )
    assert result is None
    evidence_root = tmp_path / "evidence" / "phase-actions/restarts" / action_id
    assert not evidence_root.exists()


@pytest.mark.parametrize(
    ("errno_value", "expected_kind"),
    [
        (errno.ECONNRESET, "connection-reset"),
        (errno.ETIMEDOUT, "timeout"),
        (errno.EHOSTUNREACH, "temporary-dns"),
        (errno.ENETUNREACH, "temporary-dns"),
    ],
)
def test_transport_failure_from_oserror_maps_audited_errnos(errno_value: int, expected_kind: str) -> None:
    failure = transport_failure_from_oserror(OSError(errno_value, "failure"))
    assert failure is not None
    assert failure.kind == expected_kind


@pytest.mark.parametrize("errno_value", [errno.EIO, errno.ENOSPC, errno.EACCES])
def test_transport_failure_from_oserror_returns_none_for_non_audited(errno_value: int) -> None:
    assert transport_failure_from_oserror(OSError(errno_value, "failure")) is None


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_transport_failure_from_http_status_maps_audited(status: int) -> None:
    failure = transport_failure_from_http_status(status, detail=f"HTTP {status} from s5cmd")
    assert failure is not None
    assert failure.kind == f"http-{status}"


@pytest.mark.parametrize("status", [403, 404, 0])
def test_transport_failure_from_http_status_returns_none_for_non_audited(status: int) -> None:
    assert transport_failure_from_http_status(status, detail="not audited") is None


def test_transport_failure_from_result_ok_is_none() -> None:
    result = TransferResult(tool="s5cmd", argv=("cp",), returncode=0, elapsed_s=0.0)
    assert transport_failure_from_result(result) is None


def test_transport_failure_from_result_generic_nonzero_is_none() -> None:
    result = TransferResult(tool="s5cmd", argv=("cp",), returncode=1, elapsed_s=0.0)
    assert transport_failure_from_result(result) is None


def test_transport_failure_from_result_maps_http_from_tail() -> None:
    result = TransferResult(
        tool="s5cmd",
        argv=("cp",),
        returncode=1,
        elapsed_s=0.0,
        stderr_tail="ERROR cp: RequestError: 503 Service Unavailable",
    )
    failure = transport_failure_from_result(result)
    assert failure is not None
    assert failure.kind == "http-503"
    assert failure.detail == "HTTP 503 from s5cmd"


def test_transport_failure_from_result_http_word_boundary() -> None:
    result = TransferResult(
        tool="s5cmd",
        argv=("cp",),
        returncode=1,
        elapsed_s=0.0,
        stderr_tail="error code 1503 is not an HTTP status",
    )
    assert transport_failure_from_result(result) is None


def test_raise_transport_failure_if_audited_raises() -> None:
    result = TransferResult(
        tool="s5cmd",
        argv=("cp",),
        returncode=1,
        elapsed_s=0.0,
        stderr_tail="ERROR cp: 503 Service Unavailable",
    )
    with pytest.raises(TransportFailure) as exc_info:
        raise_transport_failure_if_audited(result)
    assert exc_info.value.kind == "http-503"


def test_raise_transport_failure_if_audited_noop_for_ok() -> None:
    result = TransferResult(tool="s5cmd", argv=("cp",), returncode=0, elapsed_s=0.0)
    assert raise_transport_failure_if_audited(result) is None
