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

"""Scheduler-free postprocessing Phase materialization and replay tests."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner

from bspp.orchestration.contract.phase import (
    canonical_mapping_digest,
    phase_plan_from_mapping,
    phase_runspec_from_mapping,
)
from bspp.orchestration.contract.phase_postprocessing import (
    HistoricalPostprocessingPhaseRunSpecV1,
    PostprocessingRuntimeAction,
    postprocessing_phase_plan_from_mapping,
    postprocessing_phase_runspec_from_mapping,
    read_postprocessing_phase_runspec_from_mapping,
)
from bspp.orchestration.contract.postprocessing_autorequeue_policy import (
    PostprocessingAutorequeuePolicy,
)
from bspp.orchestration.contract.postprocessing_execution import (
    PostprocessingAttemptPaths,
    PostprocessingCredentialMountSnapshot,
)
from bspp.orchestration.contract.postprocessing_execution_parsing import (
    postprocessing_attempt_paths_from_mapping,
    postprocessing_phase_execution_identity_from_mapping,
)
from bspp.orchestration.contract.postprocessing_lifecycle import (
    PostprocessingActionTerminalObservedPayload,
    PostprocessingAttemptRetriedPayload,
    PostprocessingCancelledPayload,
    PostprocessingFinalizedPayload,
    PostprocessingJobCancellationRequestIntendedPayload,
    PostprocessingJobCancellationRequestResultPayload,
)
from bspp.orchestration.contract.postprocessing_receipt import (
    PostprocessingRuntimeInputAttestation,
    PostprocessingRuntimeInputAttestationSet,
    PostprocessingRuntimeQualificationAttestation,
)
from bspp.orchestration.contract.postprocessing_runspec_v2 import (
    PostprocessingPhaseRunSpec,
    PostprocessingPhaseRunSpecPayload,
)
from bspp.orchestration.contract.postprocessing_scheduler_evidence import (
    postprocessing_scheduler_evidence_from_mapping,
)
from bspp.orchestration.contract.postprocessing_submission_events import (
    POSTPROCESSING_RENDERER_CONTRACT_CURRENT,
    POSTPROCESSING_RENDERER_CONTRACT_SUPPORTED,
    POSTPROCESSING_V3_RENDERER_CONTRACT_SUPPORTED,
    PostprocessingSubmissionActionPlan,
    PostprocessingSubmissionIntendedPayload,
)
from bspp.orchestration.contract.runplan import load_runplan
from bspp.orchestration.contract.runspec import MountSpec
from bspp.orchestration.contract.secrets import SecretRef
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.monitoring import (
    SlurmCommandSnapshot,
    SlurmJobRecord,
    SlurmJobState,
    SlurmObservation,
)
from bspp.orchestration.control.phase_materialization import materialize_phase
from bspp.orchestration.control.phase_retry import retry_phase
from bspp.orchestration.control.phase_status import status_phase
from bspp.orchestration.control.postprocessing_action_commands import resolve_postprocessing_action_command
from bspp.orchestration.control.postprocessing_attempt_projection import (
    retarget_v3_attempt_runspec_mapping,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    append_event,
    canonical_json_bytes,
    mapping_digest,
)
from bspp.orchestration.control.postprocessing_identity import (
    build_action_semantics,
    build_scientific_identity,
)
from bspp.orchestration.control.postprocessing_phase_adapter import (
    PostprocessingAuthority,
    PostprocessingRenderInput,
    cancel_postprocessing_phase,
    load_postprocessing_phase_plan,
    postprocessing_action_command_digest,
    render_postprocessing_action_script,
    resume_postprocessing_phase,
    status_postprocessing_phase,
    submit_postprocessing_phase,
    validate_postprocessing_authority,
)
from bspp.orchestration.control.postprocessing_phase_finalization import (
    PostprocessingPhaseFinalizationResult,
    finalize_postprocessing_phase,
)
from bspp.orchestration.control.postprocessing_phase_lifecycle import (
    _array_parent_cancelled_before_task_instantiation,
    _postprocessing_submission_id,
    _rendered_action_scripts,
    _renderer_contract_for_authority,
    parse_postprocessing_timestamp,
)
from bspp.orchestration.control.postprocessing_phase_materialization import (
    materialize_postprocessing_phase,
)
from bspp.orchestration.control.postprocessing_phase_rendering import postprocessing_v3_renderer
from bspp.orchestration.control.postprocessing_phase_rendering_v3_contract3 import (
    _render_postprocessing_action_script_with_credential_mounts,
)
from bspp.orchestration.control.postprocessing_phase_retry import (
    PostprocessingRetryProjectionStore,
    retry_postprocessing_phase,
)
from bspp.orchestration.control.postprocessing_renderer3_compatibility import (
    render_authenticated_renderer3_scripts,
)
from bspp.orchestration.control.postprocessing_runtime_qualification import (
    replay_postprocessing_runtime,
)
from bspp.orchestration.control.postprocessing_scheduler_evidence import (
    export_postprocessing_scheduler_evidence,
)
from bspp.orchestration.control.postprocessing_scheduler_identity import (
    postprocessing_cluster_action_script,
    postprocessing_scheduler_correlation_token,
)
from bspp.orchestration.control.postprocessing_status_projection import project_postprocessing_authority_status
from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.transport import (
    CommandResult,
    RemoteSlurmTransport,
    SlurmAction,
    SlurmSubmission,
)
from bspp.orchestration.runtime.inputs.archives import archives_for_runspec_staging
from bspp.orchestration.runtime.postprocessing.finalization_bundle import (
    generate_scientific_output_root,
    publish_action09_finalization_bundle,
    record_successful_action_task,
)
from bspp.orchestration.runtime.postprocessing.phase_acceptance import (
    adjudicate_acceptance,
    capture_acceptance,
)
from bspp.orchestration.runtime.postprocessing.runspec_artifacts import build_recipe_config
from tests.runtime_ipsae_fixtures import runtime_ipsae_evidence

RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
SHA = "1" * 64
V2_RENDERED_ACTION_IDENTITIES = {
    "postprocessing-01-preflight": "4bd1a5ce9e574f7fd9743df7bd666234a28c5475b52ad2e14b4984bd98356876",
    "postprocessing-02-recipe": "016ac36a7297aa3806b37ed7abb199b300bb0173694f7b273e1bb0e7168c0a97",
    "postprocessing-03-preprocess": "15351a2cfc9a8ab5774d420ed39890c0c21fb31be95524f4c0604f111d79f302",
    "postprocessing-04-slurm": "3fe21121431eb4f74452ad949fe05237eec9102a1e6bf67b2b69345e25e28a78",
    "postprocessing-05-analysis-finalize": "58bddf97ecf19f77228cef381e9cb89f16a53e1a4abddb2aaae6bf57ef9000dc",
    "postprocessing-06-acceptance-tar-payload-parity": (
        "020a2d005a415e739ac89bacddd699b55dbef39a5db80ae597e86edc4eb2f1c5"
    ),
    "postprocessing-07-acceptance-semantic": "89f92501ecd0e64c7dcdcb08dbd1bdeb7d5a26c7ea4f01d038ed51a5343f5831",
    "postprocessing-08-acceptance-verify-evidence": (
        "623e5f7aaa826edb6548dd3ade1595f1c70e3d2bbe333fa4435243b95c70c932"
    ),
    "postprocessing-09-acceptance-adjudication": ("4839f147174dd92259d2aa780a16c4d3d54d39db41c8fdffda45914f21e4ace0"),
}
V3_RENDERED_ACTION_IDENTITIES = {
    "postprocessing-01-preflight": "096ebf58329cc4de6fbfde8b944ba18caf9973e8b1572c7ef638a6f80e7662f4",
    "postprocessing-02-recipe": "7a808c18188a400b9363e2fc62eeccf6046dcd090d6f7b829a5147ede90b6d2e",
    "postprocessing-03-preprocess": "b155affe3f55f34b61cc4deb24ef32c9b3d4ca2c959ff001859aa7a11a891f24",
    "postprocessing-04-slurm": "0b2e9650457caf24b5118f4175f860e590bcacbd0e72fa8c52887848dc9d876c",
    "postprocessing-05-analysis-finalize": "e16b34d3d7cd2b02ae5a958036c76bd99a80beb25acb8b60f981d70d9ad2ad8d",
    "postprocessing-06-acceptance-tar-payload-parity": (
        "1d1f0e37a98628c5701c2ac4ebd9b58c4f8f5396210917062b8fc32e173dde77"
    ),
    "postprocessing-07-acceptance-semantic": "1a36b660c0faf7fc78eb205f261b673f501695a23d713f7656b536335c940ae1",
    "postprocessing-08-acceptance-verify-evidence": (
        "156297e934310f9be0a89af19e6d64425d2b29a148db2df190a36dd01e9f9d3a"
    ),
    "postprocessing-09-acceptance-adjudication": ("bdbee353236dff8af15f2a6179f339bd72da2137948aea0e87e4ae624a1c7dd1"),
}


@dataclass(frozen=True)
class _FinalizationFixture:
    authority_root: Path
    scientific_path: Path
    scientific_tar: Path
    evidence_root: Path
    handoff: Path
    scheduler_evidence: Path

    @property
    def aggregate(self) -> Path:
        return self.handoff / "aggregate-action-evidence.json"

    @property
    def adjudication(self) -> Path:
        return self.handoff / "acceptance/adjudication.json"


def test_task853_phase_example_is_explicitly_non_runnable_without_authentic_qualification(
    tmp_path: Path,
) -> None:
    example = Path(__file__).parents[1] / "tests" / "fixtures" / "postprocessing_phase" / "neutral_v3"
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match="authentic promoted v1 record"):
        materialize_phase(
            example / "postprocessing-phase-plan.yaml",
            authority_root=authority_root,
            config_path=example / "profiles.yaml",
            source_repo=tmp_path,
            clock=lambda: NOW,
            phase_run_id_factory=lambda: RUN_ID,
        )


def test_materialization_never_replaces_an_existing_empty_authority_directory(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    existing = authority_root / RUN_ID
    existing.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="authority already exists"):
        materialize_phase(
            plan_path,
            authority_root=authority_root,
            config_path=profile_path,
            source_repo=source_repo,
            clock=lambda: NOW,
            phase_run_id_factory=lambda: RUN_ID,
        )

    assert list(existing.iterdir()) == []
    assert not tuple(authority_root.glob(f".{RUN_ID}.staging-*"))


@pytest.mark.parametrize("changed_input", ("phase-plan", "cluster-profile"))
def test_materialization_rejects_paths_changed_from_captured_operator_inputs(
    tmp_path: Path,
    changed_input: str,
) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    phase_plan_document = plan_path.read_bytes()
    config_document = profile_path.read_bytes()
    changed_path = plan_path if changed_input == "phase-plan" else profile_path
    changed_path.write_bytes(changed_path.read_bytes() + b"# changed after operator capture\n")
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match="differs from the captured operator input"):
        materialize_postprocessing_phase(
            plan_path,
            authority_root=authority_root,
            config_path=profile_path,
            source_repo=source_repo,
            clock=lambda: NOW,
            phase_run_id_factory=lambda: RUN_ID,
            phase_plan_document=phase_plan_document,
            config_document=config_document,
        )

    assert not authority_root.exists()


def test_materializes_once_rendered_legacy_projection_and_strict_authority(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    authority_root = tmp_path / "authority"

    result = materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )

    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    attempt_root = authority_root / RUN_ID / "attempts" / "attempt-0001"
    assert result.phase_runspec_digest == authority.runspec.digest
    assert authority.runspec.phase_kind == "postprocessing"
    assert (
        authority.runspec.payload.execution_projection.document_sha256
        == hashlib.sha256((attempt_root / "legacy-runspec.yaml").read_bytes()).hexdigest()
    )
    assert authority.runspec.payload.execution_projection.phase_identity.to_mapping()["phase_run_id"] == RUN_ID
    assert authority.runspec.payload.actions[-1].action_id == "postprocessing-09-acceptance-adjudication"
    assert authority.runspec.payload.actions[3].expected_task_indexes == (853,)
    assert postprocessing_phase_runspec_from_mapping(authority.runspec.to_mapping()) == authority.runspec
    with pytest.raises(ValueError, match="Unknown PhaseRunSpec field"):
        phase_runspec_from_mapping(authority.runspec.to_mapping())
    assert authority.legacy_runspec.dataset.run_id.endswith(f"{RUN_ID}-attempt-0001")
    assert (attempt_root / "runtime-qualification.json").read_bytes() == (
        tmp_path / "runtime-qualification.json"
    ).read_bytes()
    for action in authority.runspec.payload.actions:
        script = render_postprocessing_action_script(
            _render_input(authority),
            action,
            renderer_contract_version=3,
        )
        normalized_script = _normalized_characterization_script(
            script,
            authority=authority,
            action=action,
            fixture_root=tmp_path,
        )
        assert hashlib.sha256(normalized_script.encode()).hexdigest() == V3_RENDERED_ACTION_IDENTITIES[action.action_id]
        assert "\n+  " not in script
        syntax = subprocess.run(
            ("bash", "-n"),
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr
        if action.step_name.startswith("acceptance-") and action.step_name != "acceptance-adjudication":
            assert "--raw-stdout" in script
            assert "--raw-stderr" in script
        if action.step_name == "acceptance-adjudication":
            assert "--expected-phase-runspec-digest" in script
            assert "publish-action09" in script
            assert "record-action" not in script
        else:
            assert "record-action" in script
        if action.step_name == "analysis-finalize":
            assert "scientific-root" in script
        if action.step_name == "acceptance-verify-evidence":
            command = resolve_postprocessing_action_command(authority.legacy_runspec, action)
            assert f"evidence_root = Path({str(authority.legacy_runspec.submission.evidence_dir)!r})" in command
            assert "os.chdir(evidence_root)" in command
            assert "Path('acceptance/tar_payload_parity/tar_payload_parity_report.json')" in command
            assert "Path('acceptance/semantic_acceptance/semantic_acceptance_summary.json')" in command
            assert (
                str(authority.legacy_runspec.submission.evidence_dir)
                + "/acceptance/tar_payload_parity/tar_payload_parity_report.json"
                not in command
            )
            assert (
                str(authority.legacy_runspec.submission.evidence_dir)
                + "/acceptance/semantic_acceptance/semantic_acceptance_summary.json"
                not in command
            )


@pytest.mark.parametrize("renderer_contract_version", (3, 4, 5))
def test_requeue_directive_emitted_only_for_enabled_allowlisted_actions(
    tmp_path: Path, renderer_contract_version: int
) -> None:
    authority = validate_postprocessing_authority(_materialized_authority(tmp_path), RUN_ID)
    listed_id = "postprocessing-01-preflight"
    enabled = PostprocessingAutorequeuePolicy(mode="enabled", action_ids=(listed_id,))
    enabled_payload = replace(authority.runspec.payload, autorequeue_policy=enabled)
    enabled_runspec = replace(authority.runspec, payload=enabled_payload)
    render_input = PostprocessingRenderInput(runspec=enabled_runspec, legacy_runspec=authority.legacy_runspec)

    listed = authority.runspec.payload.actions[0]  # preflight == listed_id
    unlisted = authority.runspec.payload.actions[1]  # recipe, not allowlisted
    listed_script = render_postprocessing_action_script(
        render_input, listed, renderer_contract_version=renderer_contract_version
    )
    unlisted_script = render_postprocessing_action_script(
        render_input, unlisted, renderer_contract_version=renderer_contract_version
    )

    assert "#SBATCH --requeue" in listed_script
    assert "#SBATCH --requeue" not in unlisted_script
    assert "scontrol" not in listed_script  # task 3 regression
    for script in (listed_script, unlisted_script):
        syntax = subprocess.run(("bash", "-n"), input=script, text=True, capture_output=True, check=False)
        assert syntax.returncode == 0, syntax.stderr


def test_requeue_directive_absent_when_policy_disabled(tmp_path: Path) -> None:
    authority = validate_postprocessing_authority(_materialized_authority(tmp_path), RUN_ID)
    assert authority.runspec.payload.autorequeue_policy.mode == "disabled"
    assert authority.runspec.payload.autorequeue_policy.action_ids == ()
    for action in authority.runspec.payload.actions:
        script = render_postprocessing_action_script(_render_input(authority), action, renderer_contract_version=3)
        assert "#SBATCH --requeue" not in script
        syntax = subprocess.run(("bash", "-n"), input=script, text=True, capture_output=True, check=False)
        assert syntax.returncode == 0, syntax.stderr


def test_requeue_directive_absent_when_enabled_allowlist_empty(tmp_path: Path) -> None:
    authority = validate_postprocessing_authority(_materialized_authority(tmp_path), RUN_ID)
    empty_enabled = PostprocessingAutorequeuePolicy(mode="enabled", action_ids=())
    enabled_payload = replace(authority.runspec.payload, autorequeue_policy=empty_enabled)
    enabled_runspec = replace(authority.runspec, payload=enabled_payload)
    render_input = PostprocessingRenderInput(runspec=enabled_runspec, legacy_runspec=authority.legacy_runspec)
    for action in authority.runspec.payload.actions:
        script = render_postprocessing_action_script(render_input, action, renderer_contract_version=3)
        assert "#SBATCH --requeue" not in script


def test_task853_shaped_materialization_unifies_every_runtime_output_authority(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    authored = load_runplan(tmp_path / "run-plan.yaml")
    authority_root = tmp_path / "authority"

    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )

    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    spec = authority.legacy_runspec
    output_root = Path(authority.runspec.payload.attempt_paths.output_dir)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    assert authority.status == "materialized"
    assert authority.submission_state is None
    assert [event.event_type for event in authority.events] == ["phase-materialized"]
    assert authority.runspec.cluster_output_root == profile.output_root
    assert authority.runspec.object_output_base_prefix == authored.storage.s3_output_prefix
    assert spec.dataset.name == authored.dataset.name
    assert spec.references == authored.references
    assert spec.storage.s3_archive_prefix == authored.storage.s3_archive_prefix
    assert spec.acceptance is not None and authored.acceptance is not None
    assert spec.acceptance.baseline_output_dir == authored.acceptance.baseline_output_dir
    assert spec.storage.local_tar_dir == output_root / "local_tars"
    assert spec.storage.local_tar_manifest_csv == output_root / "local_tars.csv"
    assert spec.analysis_metadata.csv_path == output_root / "analysis_metadata.csv"
    assert spec.analysis_metadata.parquet_path == output_root / "analysis_metadata.parquet"
    assert spec.analysis_metadata.selected_ids_path == output_root / "high_quality_model_ids.txt"

    recipe = build_recipe_config(spec, ["task853.tar.zst"])
    assert recipe["paths"]["output_dir"] == str(output_root)
    assert recipe["upload"]["local_tar_dir"] == str(output_root / "local_tars")
    assert recipe["upload"]["local_tar_manifest_csv"] == str(output_root / "local_tars.csv")
    assert recipe["analysis_metadata"]["csv_path"] == str(output_root / "analysis_metadata.csv")
    action_commands = {
        action.step_name: resolve_postprocessing_action_command(spec, action)
        for action in authority.runspec.payload.actions
        if action.step_name != "acceptance-adjudication"
    }
    for step in ("analysis-finalize", "acceptance-tar-payload-parity", "acceptance-semantic"):
        assert str(output_root) in action_commands[step]
    assert str(authored.storage.local_tar_dir) not in action_commands["analysis-finalize"]


def test_materialization_rejects_an_output_outside_the_authored_root_before_authority(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    run_plan_path = tmp_path / "run-plan.yaml"
    authored = yaml.safe_load(run_plan_path.read_text())
    authored["storage"]["local_tar_dir"] = str(tmp_path / "split-output" / "local_tars")
    run_plan_path.write_text(yaml.safe_dump(authored, sort_keys=False))
    phase_plan = yaml.safe_load(plan_path.read_text())
    phase_plan["legacy_run_plan"] = _document("legacy-run-plan", run_plan_path)
    plan_path.write_text(yaml.safe_dump(phase_plan, sort_keys=False))
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match=r"storage\.local_tar_dir.*fresh V3 Phase Run"):
        materialize_phase(
            plan_path,
            authority_root=authority_root,
            config_path=profile_path,
            source_repo=source_repo,
            clock=lambda: NOW,
            phase_run_id_factory=lambda: RUN_ID,
        )

    assert not (authority_root / RUN_ID).exists()


@pytest.mark.parametrize(
    "s3_output_prefix",
    (
        "s3://example-output/task853",
        "s3://example-output/task853//",
        "s3://example-output/task853/?query=yes",
        f"s3://example-output/{RUN_ID}-attempt-0001/interior/",
    ),
)
def test_materialization_rejects_malformed_or_already_attempt_scoped_s3_base(
    tmp_path: Path,
    s3_output_prefix: str,
) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    run_plan_path = tmp_path / "run-plan.yaml"
    authored = yaml.safe_load(run_plan_path.read_text())
    authored["storage"]["s3_output_prefix"] = s3_output_prefix
    run_plan_path.write_text(yaml.safe_dump(authored, sort_keys=False))
    phase_plan = yaml.safe_load(plan_path.read_text())
    phase_plan["legacy_run_plan"] = _document("legacy-run-plan", run_plan_path)
    plan_path.write_text(yaml.safe_dump(phase_plan, sort_keys=False))
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match="Attempt output projection is unsafe"):
        materialize_phase(
            plan_path,
            authority_root=authority_root,
            config_path=profile_path,
            source_repo=source_repo,
            clock=lambda: NOW,
            phase_run_id_factory=lambda: RUN_ID,
        )

    assert not (authority_root / RUN_ID).exists()


def test_split_v3_authority_stays_readable_but_fails_closed_before_mutation_or_transport(
    tmp_path: Path,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    _rewrite_as_consistent_split_v3(authority_root)
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    before_events = tuple((authority.authority_path / "events").iterdir())
    observed = status_postprocessing_phase(RUN_ID, authority_root=authority_root)
    assert observed.status == "materialized"
    assert observed.details["contract_family"] == "postprocessing-runspec-v3"

    def forbidden_runner(_argv: tuple[str, ...]) -> CommandResult:
        raise AssertionError("split V3 authority reached transport")

    with pytest.raises(ValueError, match="Attempt output projection is unsafe"):
        submit_postprocessing_phase(RUN_ID, authority_root=authority_root, runner=forbidden_runner)
    with pytest.raises(ValueError, match="Attempt output projection is unsafe"):
        retry_postprocessing_phase(
            RUN_ID,
            authority_root=authority_root,
            config_path=tmp_path / "must-not-be-read.yaml",
            runner=forbidden_runner,
        )
    with pytest.raises(ValueError, match="Attempt output projection is unsafe"):
        finalize_postprocessing_phase(
            RUN_ID,
            authority_root=authority_root,
            scheduler_evidence_path=tmp_path / "must-not-be-read-scheduler.json",
            aggregate_action_evidence_path=tmp_path / "must-not-be-read-aggregate.json",
            handoff_path=tmp_path / "must-not-be-read-handoff",
            acceptance_adjudication_path=tmp_path / "must-not-be-read-adjudication.json",
        )
    assert tuple((authority.authority_path / "events").iterdir()) == before_events


@pytest.mark.parametrize("mismatch", ("attempt-id", "output-namespace", "staging-root"))
def test_envelope_path_mismatch_fails_before_submit_retry_or_finalize_side_effects(
    tmp_path: Path,
    mismatch: str,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    _rewrite_as_consistent_envelope_path_mismatch(authority_root, mismatch=mismatch)
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    before_files = {
        path.relative_to(authority.authority_path): path.read_bytes()
        for path in authority.authority_path.rglob("*")
        if path.is_file()
    }
    observed = status_postprocessing_phase(RUN_ID, authority_root=authority_root)
    assert observed.status == "materialized"

    def forbidden_runner(_argv: tuple[str, ...]) -> CommandResult:
        raise AssertionError("envelope/path mismatch reached transport")

    with pytest.raises(ValueError, match="envelope-derived"):
        submit_postprocessing_phase(RUN_ID, authority_root=authority_root, runner=forbidden_runner)
    with pytest.raises(ValueError, match="envelope-derived"):
        retry_postprocessing_phase(
            RUN_ID,
            authority_root=authority_root,
            config_path=tmp_path / "must-not-be-read.yaml",
            runner=forbidden_runner,
        )
    with pytest.raises(ValueError, match="envelope-derived"):
        finalize_postprocessing_phase(
            RUN_ID,
            authority_root=authority_root,
            scheduler_evidence_path=tmp_path / "must-not-be-read-scheduler.json",
            aggregate_action_evidence_path=tmp_path / "must-not-be-read-aggregate.json",
            handoff_path=tmp_path / "must-not-be-read-handoff",
            acceptance_adjudication_path=tmp_path / "must-not-be-read-adjudication.json",
        )
    after_files = {
        path.relative_to(authority.authority_path): path.read_bytes()
        for path in authority.authority_path.rglob("*")
        if path.is_file()
    }
    assert after_files == before_files


def test_materialized_projection_preserves_tracking_selector_for_high_array_index(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    run_plan_path = tmp_path / "run-plan.yaml"
    tracking_path = tmp_path / "tracking.parquet"
    authored = yaml.safe_load(run_plan_path.read_text())
    dataset_name = authored["dataset"]["name"]
    pq.write_table(
        pa.table(
            {
                "dataset_name": [dataset_name] * 854,
                "swiftstack_archive": [f"archive-{index:04d}.tar.lz4" for index in range(854)],
            }
        ),
        tracking_path,
    )
    authored["references"]["tracking_parquet"]["path"] = str(tracking_path)
    run_plan_path.write_text(yaml.safe_dump(authored, sort_keys=False))
    phase_plan = yaml.safe_load(plan_path.read_text())
    phase_plan["legacy_run_plan"] = _document("legacy-run-plan", run_plan_path)
    plan_path.write_text(yaml.safe_dump(phase_plan, sort_keys=False))

    authority_root = tmp_path / "authority"
    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )
    authority = validate_postprocessing_authority(authority_root, RUN_ID)

    assert authority.legacy_runspec.dataset.name == dataset_name
    assert authority.legacy_runspec.dataset.run_id != dataset_name
    coverage = archives_for_runspec_staging(authority.legacy_runspec)
    assert coverage.archives == ("archive-0853.tar.lz4",)


def test_v2_renderer_contract_two_mounts_aws_profile_without_secret_values(tmp_path: Path) -> None:
    authority_root = _materialized_authority(tmp_path)
    authority = _v2_authority(validate_postprocessing_authority(authority_root, RUN_ID), tmp_path)
    action = authority.runspec.payload.actions[0]
    render_input = _render_input(authority)

    v1 = render_postprocessing_action_script(render_input, action, renderer_contract_version=1)
    v2 = render_postprocessing_action_script(render_input, action, renderer_contract_version=2)

    assert f"{Path.home() / '.aws' / 'credentials'}:/workspace/bspp-aws/credentials:ro" not in v1
    assert f"{Path.home() / '.aws' / 'config'}:/workspace/bspp-aws/config:ro" not in v1
    assert f"{Path.home() / '.aws' / 'credentials'}:/workspace/bspp-aws/credentials:ro" in v2
    assert f"{Path.home() / '.aws' / 'config'}:/workspace/bspp-aws/config:ro" in v2
    assert f"{Path.home() / '.aws'}:/workspace/bspp-aws:ro" not in v2
    assert "AWS_SHARED_CREDENTIALS_FILE=/workspace/bspp-aws/credentials" in v2
    assert "AWS_CONFIG_FILE=/workspace/bspp-aws/config" in v2
    assert "AWS_PROFILE=example-account" in v2
    assert "AWS_ACCESS_KEY_ID" not in v2
    assert "AWS_SECRET_ACCESS_KEY" not in v2

    for target in ("/workspace/bspp-aws", "/workspace", "/workspace/bspp-aws/child"):
        mount = MountSpec(source=tmp_path, target=Path(target), read_only=True)
        changed_container = authority.legacy_runspec.container.model_copy(
            update={"mounts": (*authority.legacy_runspec.container.mounts, mount)}
        )
        changed_runspec = authority.legacy_runspec.model_copy(update={"container": changed_container})
        with pytest.raises(ValueError, match="overlaps a governed container target"):
            render_postprocessing_action_script(
                PostprocessingRenderInput(runspec=authority.runspec, legacy_runspec=changed_runspec),
                action,
                renderer_contract_version=2,
            )

    for target, error in (
        ("/workspace/x/../afcdb-aws", "must not contain traversal aliases"),
        ("workspace/bspp-aws", "must be absolute"),
        ("//workspace/bspp-aws", "must use a single-root absolute path"),
    ):
        mount = MountSpec(source=tmp_path, target=Path(target), read_only=True)
        changed_container = authority.legacy_runspec.container.model_copy(
            update={"mounts": (*authority.legacy_runspec.container.mounts, mount)}
        )
        changed_runspec = authority.legacy_runspec.model_copy(update={"container": changed_container})
        with pytest.raises(ValueError, match=error):
            render_postprocessing_action_script(
                PostprocessingRenderInput(runspec=authority.runspec, legacy_runspec=changed_runspec),
                action,
                renderer_contract_version=2,
            )


def test_v2_renderer_contract_one_characterization_remains_frozen(tmp_path: Path) -> None:
    authority_root = _materialized_authority(tmp_path)
    authority = _v2_authority(validate_postprocessing_authority(authority_root, RUN_ID), tmp_path)

    for action in authority.runspec.payload.actions:
        script = render_postprocessing_action_script(
            _render_input(authority),
            action,
            renderer_contract_version=1,
        )
        normalized = _normalized_characterization_script(
            script,
            authority=authority,
            action=action,
            fixture_root=tmp_path,
        )
        assert hashlib.sha256(normalized.encode()).hexdigest() == V2_RENDERED_ACTION_IDENTITIES[action.action_id]


def test_public_v2_renderer_default_remains_contract_one(tmp_path: Path) -> None:
    authority_root = _materialized_authority(tmp_path)
    authority = _v2_authority(validate_postprocessing_authority(authority_root, RUN_ID), tmp_path)
    action = authority.runspec.payload.actions[0]

    rendered_default = render_postprocessing_action_script(_render_input(authority), action)
    rendered_contract_one = render_postprocessing_action_script(
        _render_input(authority),
        action,
        renderer_contract_version=1,
    )

    assert rendered_default == rendered_contract_one


def test_v2_renderer_contract_auto_profile_omits_aws_profile(tmp_path: Path) -> None:
    authority_root = _materialized_authority(tmp_path)
    authority = _v2_authority(validate_postprocessing_authority(authority_root, RUN_ID), tmp_path)
    action = authority.runspec.payload.actions[0]
    auto_secrets = authority.legacy_runspec.secrets.model_copy(
        update={"s3_credentials_ref": SecretRef.parse("aws:auto")}
    )
    auto_runspec = authority.legacy_runspec.model_copy(update={"secrets": auto_secrets})

    rendered = render_postprocessing_action_script(
        PostprocessingRenderInput(runspec=authority.runspec, legacy_runspec=auto_runspec),
        action,
        renderer_contract_version=2,
    )

    assert f"{Path.home() / '.aws' / 'credentials'}:/workspace/bspp-aws/credentials:ro" in rendered
    assert f"{Path.home() / '.aws' / 'config'}:/workspace/bspp-aws/config:ro" in rendered
    assert "AWS_SHARED_CREDENTIALS_FILE=/workspace/bspp-aws/credentials" in rendered
    assert "AWS_CONFIG_FILE=/workspace/bspp-aws/config" in rendered
    assert "AWS_PROFILE=" not in rendered


def test_phase_acceptance_verification_reports_paths_relative_to_attempt_evidence_from_unrelated_cwd(
    tmp_path: Path,
) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    evidence_root = authority.legacy_runspec.submission.evidence_dir
    parity = evidence_root / "acceptance/tar_payload_parity/tar_payload_parity_report.json"
    semantic = evidence_root / "acceptance/semantic_acceptance/semantic_acceptance_summary.json"
    parity.parent.mkdir(parents=True)
    semantic.parent.mkdir(parents=True)
    parity.write_text(
        json.dumps(
            {
                "ok": False,
                "payload_mismatch_count": 1,
                "error_count": 0,
                "compared_tar_count": 1,
                "compared_members": 1,
                "payload_sample_count": 1,
            }
        )
    )
    semantic.write_text(json.dumps({"ok": False, "errors": ["known residual"]}))
    action = next(item for item in authority.runspec.payload.actions if item.step_name == "acceptance-verify-evidence")
    without_submission = authority.legacy_runspec.model_copy(update={"submission": None})
    with pytest.raises(ValueError, match=r"requires submission\.evidence_dir"):
        resolve_postprocessing_action_command(without_submission, action)
    command = resolve_postprocessing_action_command(authority.legacy_runspec, action)
    environment = dict(os.environ)
    environment["PYTHON_BIN"] = sys.executable
    environment["PYTHONPATH"] = ":".join(
        (
            str(Path(__file__).parents[1] / "packages/orchestration-contract/src"),
            str(Path(__file__).parents[1] / "packages/orchestration-runtime/src"),
        )
    )
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    result = subprocess.run(
        ("bash", "-c", command),
        cwd=unrelated_cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    report = json.loads((evidence_root / "acceptance/verify_evidence/acceptance_evidence_report.json").read_text())
    parity_path = "acceptance/tar_payload_parity/tar_payload_parity_report.json"
    semantic_path = "acceptance/semantic_acceptance/semantic_acceptance_summary.json"
    assert report["parity_report_path"] == parity_path
    assert report["semantic_report_path"] == semantic_path
    assert report["issues"] == [
        {
            "check": "tar-payload-parity",
            "message": "ok is not true",
            "report_path": parity_path,
        },
        {
            "check": "tar-payload-parity",
            "message": "payload_mismatch_count is not 0: 1",
            "report_path": parity_path,
        },
        {
            "check": "semantic-acceptance",
            "message": "ok is not true",
            "report_path": semantic_path,
        },
        {
            "check": "semantic-acceptance",
            "message": "errors is not empty: ['known residual']",
            "report_path": semantic_path,
        },
    ]


def test_historical_v1_runspec_is_readable_but_never_executable(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )
    current = validate_postprocessing_authority(authority_root, RUN_ID).runspec.to_mapping()
    historical = json.loads(json.dumps(current))
    del historical["runspec_kind"]
    del historical["cluster_output_root"]
    del historical["object_output_base_prefix"]
    del historical["credential_mounts"]
    payload = historical["payload"]
    logical = {
        "schema_version": 1,
        "manifest_kind": "postprocessing-logical-inputs-v1",
        "entries": [
            {
                "schema_version": 1,
                "name": item["name"],
                "verification_kind": "authority-declared-content-v1",
                "authority": f"historical-authority:{item['name']}",
                "member_identity": item["member_identity"],
                "expected_content_sha256": item["expected_content_sha256"],
                "expected_size_bytes": item["expected_size_bytes"],
            }
            for item in payload["logical_inputs"]["entries"]
        ],
    }
    payload["logical_inputs"] = logical
    payload["scientific_identity"] = {
        "schema_version": 1,
        "identity_kind": "postprocessing-scientific-identity-v1",
        "dataset_scope_digest": current["payload"]["scientific_identity"]["dataset_scope_digest"],
        "scientific_parameters_digest": current["payload"]["scientific_identity"]["scientific_parameters_digest"],
        "logical_input_identity_digest": canonical_mapping_digest(logical),
        "acceptance_semantic_digest": payload["acceptance_policy"]["semantic_digest"],
    }
    del payload["action_semantics"]
    payload.pop("autorequeue_policy", None)
    phase_identity = payload["execution_projection"]["phase_identity"]
    phase_identity["identity_kind"] = "postprocessing-execution-projection-v1"
    phase_identity["logical_input_manifest_digest"] = canonical_mapping_digest(logical)
    phase_identity["scientific_identity_digest"] = canonical_mapping_digest(payload["scientific_identity"])
    del phase_identity["action_semantics_digest"]
    del phase_identity["acceptance_semantic_digest"]
    payload["execution_projection"]["phase_identity_digest"] = canonical_mapping_digest(phase_identity)

    inspected = read_postprocessing_phase_runspec_from_mapping(historical)
    assert isinstance(inspected, HistoricalPostprocessingPhaseRunSpecV1)
    assert inspected.phase_run_id == RUN_ID
    assert inspected.to_mapping() == historical
    with pytest.raises(ValueError, match="historical postprocessing V1 authority is read-only; rematerialize"):
        postprocessing_phase_runspec_from_mapping(historical)


def test_locator_changes_do_not_change_v2_science_or_action_semantics(tmp_path: Path) -> None:
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first_plan, first_profile, first_source = _fixture(tmp_path / "first")
    second_plan, second_profile, second_source = _fixture(tmp_path / "second")
    second_run_plan = second_plan.parent / "run-plan.yaml"
    changed = yaml.safe_load(second_run_plan.read_text())
    changed["references"]["master_parquet"]["path"] = "/relocated/input/master.parquet"
    second_run_plan.write_text(yaml.safe_dump(changed, sort_keys=False))
    second_phase_plan = yaml.safe_load(second_plan.read_text())
    second_phase_plan["legacy_run_plan"] = _document("legacy-run-plan", second_run_plan)
    second_plan.write_text(yaml.safe_dump(second_phase_plan, sort_keys=False))

    first_root = tmp_path / "first-authority"
    second_root = tmp_path / "second-authority"
    materialize_phase(
        first_plan,
        authority_root=first_root,
        config_path=first_profile,
        source_repo=first_source,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )
    materialize_phase(
        second_plan,
        authority_root=second_root,
        config_path=second_profile,
        source_repo=second_source,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )
    first = validate_postprocessing_authority(first_root, RUN_ID).runspec
    second = validate_postprocessing_authority(second_root, RUN_ID).runspec

    assert first.payload.logical_inputs == second.payload.logical_inputs
    assert first.payload.scientific_identity == second.payload.scientific_identity
    assert first.payload.action_semantics == second.payload.action_semantics
    assert first.payload.physical_inputs != second.payload.physical_inputs
    assert first.payload.execution_projection.phase_identity_digest != (
        second.payload.execution_projection.phase_identity_digest
    )
    semantic_document = json.dumps(
        {
            "logical_inputs": first.payload.logical_inputs.to_mapping(),
            "scientific_identity": first.payload.scientific_identity.to_mapping(),
            "action_semantics": first.payload.action_semantics.to_mapping(),
        },
        sort_keys=True,
    )
    assert "/relocated/input/master.parquet" not in semantic_document
    assert str(tmp_path) not in semantic_document


def test_v2_semantics_change_for_science_encoding_acceptance_and_workflow_mode(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )
    authority = _v2_authority(validate_postprocessing_authority(authority_root, RUN_ID), tmp_path)
    baseline_plan = load_runplan(tmp_path / "run-plan.yaml")

    def identities(plan: object, legacy_runspec: object = authority.legacy_runspec) -> tuple[str, str]:
        scientific = build_scientific_identity(plan, logical_inputs=authority.runspec.payload.logical_inputs)
        actions = build_action_semantics(
            authority.runspec.payload.actions,
            legacy_plan=plan,
            legacy_runspec=legacy_runspec,
            scientific_identity=scientific,
            logical_inputs=authority.runspec.payload.logical_inputs,
            acceptance_policy=authority.acceptance_policy,
        )
        return scientific.digest, actions.digest

    baseline = identities(baseline_plan)
    science_plan = baseline_plan.model_copy(
        update={
            "worker": baseline_plan.worker.model_copy(
                update={"dssp_algorithm": f"{baseline_plan.worker.dssp_algorithm}-changed"}
            )
        }
    )
    encoding_plan = baseline_plan.model_copy(
        update={"worker": baseline_plan.worker.model_copy(update={"batch_size": baseline_plan.worker.batch_size + 1})}
    )
    validation_plan = baseline_plan.model_copy(
        update={
            "validation": baseline_plan.validation.model_copy(
                update={"expected_tar_count": (baseline_plan.validation.expected_tar_count or 0) + 1}
            )
        }
    )
    acceptance_plan = baseline_plan.model_copy(
        update={
            "acceptance": baseline_plan.acceptance.model_copy(
                update={"payload_sample_count": (baseline_plan.acceptance.payload_sample_count or 0) + 1}
            )
        }
    )
    assert identities(science_plan)[0] != baseline[0]
    for changed_plan in (science_plan, encoding_plan, validation_plan, acceptance_plan):
        assert identities(changed_plan)[1] != baseline[1]
    for changed_plan in (encoding_plan, validation_plan, acceptance_plan):
        assert identities(changed_plan)[0] == baseline[0]

    assert authority.legacy_runspec.workflow is not None
    steps = list(authority.legacy_runspec.workflow.steps)
    slurm_index = next(index for index, step in enumerate(steps) if step.name == "slurm")
    steps[slurm_index] = steps[slurm_index].model_copy(update={"mode": "monitor-existing", "job_id": "12345"})
    changed_workflow = authority.legacy_runspec.workflow.model_copy(update={"steps": tuple(steps)})
    changed_runspec = authority.legacy_runspec.model_copy(update={"workflow": changed_workflow})
    assert identities(baseline_plan, changed_runspec)[1] != baseline[1]
    with pytest.raises(ValueError, match="stored action semantics differ from rendered command"):
        render_postprocessing_action_script(
            PostprocessingRenderInput(runspec=authority.runspec, legacy_runspec=changed_runspec),
            authority.runspec.payload.actions[slurm_index],
            renderer_contract_version=2,
        )


def _v2_authority(authority: PostprocessingAuthority, fixture_root: Path) -> PostprocessingAuthority:
    """Reconstruct the exact legacy V2 rendering inputs from a V3 fixture."""
    paths = authority.runspec.payload.attempt_paths
    legacy_dataset = authority.legacy_runspec.dataset.model_copy(update={"name": paths.legacy_run_id})
    legacy_plan = load_runplan(fixture_root / "run-plan.yaml")
    frozen_example = load_runplan(
        Path(__file__).parents[1] / "tests" / "fixtures" / "postprocessing_phase" / "neutral_v3" / "run-plan.yaml"
    )
    legacy_storage = frozen_example.storage.model_copy(update={"s3_output_prefix": paths.object_prefix})
    legacy_runspec = authority.legacy_runspec.model_copy(
        update={
            "dataset": legacy_dataset,
            "storage": legacy_storage,
            "analysis_metadata": frozen_example.analysis_metadata,
        }
    )
    scientific = build_scientific_identity(
        legacy_plan,
        logical_inputs=authority.runspec.payload.logical_inputs,
    )
    semantics = build_action_semantics(
        authority.runspec.payload.actions,
        legacy_plan=legacy_plan,
        legacy_runspec=legacy_runspec,
        scientific_identity=scientific,
        logical_inputs=authority.runspec.payload.logical_inputs,
        acceptance_policy=authority.acceptance_policy,
    )
    phase_identity = replace(
        authority.runspec.payload.execution_projection.phase_identity,
        scientific_identity_digest=scientific.digest,
        action_semantics_digest=semantics.digest,
    )
    projection = replace(
        authority.runspec.payload.execution_projection,
        phase_identity=phase_identity,
        phase_identity_digest=phase_identity.digest,
    )
    current = authority.runspec.payload
    payload = PostprocessingPhaseRunSpecPayload(
        actions=current.actions,
        action_graph_digest=current.action_graph_digest,
        action_semantics_digest=semantics.digest,
        action_semantics=semantics,
        scientific_identity=scientific,
        logical_inputs=current.logical_inputs,
        physical_inputs=current.physical_inputs,
        execution_projection=projection,
        acceptance_policy=current.acceptance_policy,
        qualified_runtime=current.qualified_runtime,
        attempt_paths=current.attempt_paths,
    )
    runspec = PostprocessingPhaseRunSpec(
        phase_run_id=authority.phase_run_id,
        attempt_id=authority.attempt_id,
        phase_plan_digest=authority.phase_plan.digest,
        materialized_at=authority.runspec.materialized_at,
        cluster=authority.runspec.cluster,
        payload=payload,
    )
    return replace(authority, runspec=runspec, legacy_runspec=legacy_runspec)


def test_status_projection_distinguishes_executable_v2_and_v3_families(tmp_path: Path) -> None:
    authority = validate_postprocessing_authority(_materialized_authority(tmp_path), RUN_ID)
    v2 = _v2_authority(authority, tmp_path)

    assert project_postprocessing_authority_status(v2).contract_family == "postprocessing-runspec-v2"
    assert project_postprocessing_authority_status(authority).contract_family == "postprocessing-runspec-v3"


def _render_input(authority: PostprocessingAuthority) -> PostprocessingRenderInput:
    return PostprocessingRenderInput(runspec=authority.runspec, legacy_runspec=authority.legacy_runspec)


def _normalized_characterization_script(
    script: str,
    *,
    authority: PostprocessingAuthority,
    action: PostprocessingRuntimeAction,
    fixture_root: Path,
) -> str:
    qualified = authority.runspec.payload.qualified_runtime
    replacements = (
        (str(fixture_root), "<FIXTURE_ROOT>"),
        # The governed credential seam mounts the invoking user's AWS shared
        # credential files, so rendered scripts embed the operator's HOME
        # twice (credentials and config). Mask it so the pinned identities are
        # user-independent.
        (str(Path.home()), "<HOME>"),
        (Path(qualified.source_package_path).name, "<SOURCE_PACKAGE>"),
        (Path(qualified.toolkit_package_path or "").name, "<TOOLKIT_PACKAGE>"),
        (authority.runspec.digest, "<PHASE_RUNSPEC_DIGEST>"),
        (qualified.qualification_sha256, "<QUALIFICATION_SHA256>"),
        (qualified.source_revision, "<SOURCE_REVISION>"),
        (qualified.tuple_id, "<QUALIFICATION_TUPLE_ID>"),
        (postprocessing_action_command_digest(_render_input(authority), action), "<COMMAND_DIGEST>"),
    )
    for old, new in replacements:
        script = script.replace(old, new)
    return re.sub(r"bspp-pp-[0-9a-f]{48}", "bspp-pp-<CORRELATION>", script)


def test_postprocessing_phase_plan_roundtrip_and_cross_family_boundaries(tmp_path: Path) -> None:
    plan_path, _profile_path, _source_repo = _fixture(tmp_path)
    plan = load_postprocessing_phase_plan(plan_path)
    mapping = plan.to_mapping()

    assert postprocessing_phase_plan_from_mapping(mapping) == plan
    assert postprocessing_phase_plan_from_mapping(mapping).digest == plan.digest
    with pytest.raises(ValueError, match="Unknown PhasePlan field"):
        phase_plan_from_mapping(mapping)

    wrong_kind = json.loads(json.dumps(mapping))
    wrong_kind["phase_kind"] = "preprocessing"
    with pytest.raises(ValueError, match="phase_kind 'postprocessing'"):
        postprocessing_phase_plan_from_mapping(wrong_kind)

    wrong_nested_version = json.loads(json.dumps(mapping))
    wrong_nested_version["legacy_run_plan"]["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version 2"):
        postprocessing_phase_plan_from_mapping(wrong_nested_version)

    interpolated = json.loads(json.dumps(mapping))
    interpolated["output_namespace"] = "${BSPP_OUTPUT_NAMESPACE}"
    with pytest.raises(ValueError, match="environment interpolation"):
        postprocessing_phase_plan_from_mapping(interpolated)

    unknown = json.loads(json.dumps(mapping))
    unknown["invented"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        postprocessing_phase_plan_from_mapping(unknown)


def test_phase_cli_materializes_and_reads_postprocessing_without_changing_legacy_help(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    runner = CliRunner()
    legacy_help = runner.invoke(cli, ["run", "--help"]).output

    materialized = runner.invoke(
        cli,
        [
            "--config",
            str(profile_path),
            "phase",
            "materialize",
            str(plan_path),
            "--authority-root",
            str(authority_root),
            "--source-repo",
            str(source_repo),
        ],
    )
    assert materialized.exit_code == 0, materialized.output
    payload = json.loads(materialized.output)
    phase_run_id = payload["phase_run_id"]

    status = runner.invoke(
        cli,
        ["phase", "status", phase_run_id, "--authority-root", str(authority_root), "--format", "json"],
    )
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["phase_kind"] == "postprocessing"
    assert runner.invoke(cli, ["run", "--help"]).output == legacy_help


def test_authority_replay_rejects_changed_legacy_projection_bytes(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )
    projection = authority_root / RUN_ID / "attempts" / "attempt-0001" / "legacy-runspec.yaml"
    projection.write_bytes(projection.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="legacy RunSpec projection bytes differ"):
        validate_postprocessing_authority(authority_root, RUN_ID)


@pytest.mark.parametrize("missing_field", ("created_at", "status", "sealed"))
def test_authority_replay_requires_exact_phase_run_snapshot(tmp_path: Path, missing_field: str) -> None:
    authority_root = _materialized_authority(tmp_path)
    phase_run_path = authority_root / RUN_ID / "phase-run.json"
    phase_run = json.loads(phase_run_path.read_bytes())
    del phase_run[missing_field]
    phase_run_path.write_text(json.dumps(phase_run, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="invalid strict envelope"):
        validate_postprocessing_authority(authority_root, RUN_ID)


def test_authority_replay_rejects_symlinked_canonical_json(tmp_path: Path) -> None:
    authority_root = _materialized_authority(tmp_path)
    phase_run_path = authority_root / RUN_ID / "phase-run.json"
    replacement = tmp_path / "phase-run-copy.json"
    replacement.write_bytes(phase_run_path.read_bytes())
    phase_run_path.unlink()
    phase_run_path.symlink_to(replacement)

    with pytest.raises(ValueError, match="regular non-symlink file"):
        validate_postprocessing_authority(authority_root, RUN_ID)


def test_submission_stages_all_attempt_projections_before_first_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    events: list[str] = []
    stage_calls: list[tuple[Path, str, str, str]] = []
    staged: dict[str, str] = {}
    submitted: list[SlurmAction] = []

    def stage_immutable_artifact(
        _transport: RemoteSlurmTransport,
        local_path: Path,
        target_path: str,
        *,
        expected_sha256: str,
        staging_token: str,
    ) -> None:
        prior = staged.setdefault(target_path, expected_sha256)
        assert prior == expected_sha256
        events.append("stage")
        stage_calls.append((local_path, target_path, expected_sha256, staging_token))

    def command(_transport: RemoteSlurmTransport, argv: tuple[str, ...]) -> CommandResult:
        events.append("command")
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")

    def submit_action(_transport: RemoteSlurmTransport, action: SlurmAction) -> SlurmSubmission:
        events.append("submit")
        submitted.append(action)
        job_id = str(9001 + len(submitted))
        result = CommandResult(
            argv=("sbatch", "--parsable", str(action.script_path)),
            returncode=0,
            stdout=job_id,
            stderr="",
        )
        return SlurmSubmission(job_id=job_id, command=result.argv, result=result)

    monkeypatch.setattr(RemoteSlurmTransport, "stage_immutable_artifact", stage_immutable_artifact)
    monkeypatch.setattr(RemoteSlurmTransport, "command", command)
    monkeypatch.setattr(RemoteSlurmTransport, "query_submissions_by_correlation", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(RemoteSlurmTransport, "submit_action", submit_action)

    result = submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    repeated = submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)

    attempt_root = authority.authority_path / "attempts" / authority.attempt_id
    cluster_root = (
        Path(authority.runspec.cluster.staging_root) / "bspp-phase-runs" / authority.phase_run_id / authority.attempt_id
    )
    expected = (
        (attempt_root / "phase-runspec.json", str(cluster_root / "phase-runspec.json")),
        (attempt_root / "legacy-runspec.yaml", str(cluster_root / "legacy-runspec.yaml")),
        (attempt_root / "acceptance-policy.json", str(cluster_root / "acceptance-policy.json")),
        (attempt_root / "runtime-qualification.json", str(cluster_root / "runtime-qualification.json")),
    )
    assert [(local, remote) for local, remote, _digest, _token in stage_calls[:4]] == list(expected)
    assert [digest for _local, _remote, digest, _token in stage_calls[:4]] == [
        hashlib.sha256(local.read_bytes()).hexdigest() for local, _remote in expected
    ]
    assert events.index("submit") > 3
    assert events[:4] == ["stage"] * 4
    assert len(submitted) == len(authority.runspec.payload.actions)
    assert repeated == result
    assert len(stage_calls) == 4 + len(authority.runspec.payload.actions)
    recorded = validate_postprocessing_authority(authority_root, RUN_ID)
    assert recorded.submission_state is not None
    assert {plan.renderer_contract_version for plan in recorded.submission_state.actions} == {5}


def test_v3_authority_cannot_render_or_resume_with_a_legacy_renderer(tmp_path: Path) -> None:
    authority_root = _materialized_authority(tmp_path)
    authority = validate_postprocessing_authority(authority_root, RUN_ID)

    with pytest.raises(ValueError, match="V3 supports only renderer contracts 3, 4, and 5"):
        _rendered_action_scripts(authority, renderer_contract_version=1)
    assert validate_postprocessing_authority(authority_root, RUN_ID).status == "materialized"


def test_v3_renderer_contract_registry_selects_five_and_rejects_six() -> None:
    assert POSTPROCESSING_RENDERER_CONTRACT_CURRENT == 5
    assert frozenset({1, 2, 3, 4, 5}) == POSTPROCESSING_RENDERER_CONTRACT_SUPPORTED
    assert frozenset({3, 4, 5}) == POSTPROCESSING_V3_RENDERER_CONTRACT_SUPPORTED
    with pytest.raises(ValueError, match="unsupported postprocessing V3 renderer contract version: 6"):
        postprocessing_v3_renderer(6)


def test_renderer_contract_four_is_home_independent_and_uses_frozen_mount_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = validate_postprocessing_authority(_materialized_authority(tmp_path), RUN_ID)
    action = authority.runspec.payload.actions[0]
    assert authority.runspec.credential_mounts is not None

    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: Path("/first-home")))
    first = render_postprocessing_action_script(_render_input(authority), action, renderer_contract_version=4)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: Path("/second-home")))
    second = render_postprocessing_action_script(_render_input(authority), action, renderer_contract_version=4)

    assert first == second
    assert "/home/example/.aws/credentials:/workspace/bspp-aws/credentials:ro" in first
    assert "/home/example/.aws/config:/workspace/bspp-aws/config:ro" in first
    assert "/first-home" not in first and "/second-home" not in first


def test_renderer_contract_five_forwards_array_parent_and_preserves_scalar_renderer_four_bytes(tmp_path: Path) -> None:
    authority = validate_postprocessing_authority(_materialized_authority(tmp_path), RUN_ID)
    array_action = authority.runspec.payload.actions[3]
    scalar_action = authority.runspec.payload.actions[0]
    array_v5 = render_postprocessing_action_script(_render_input(authority), array_action, renderer_contract_version=5)
    array_v4 = render_postprocessing_action_script(_render_input(authority), array_action, renderer_contract_version=4)
    scalar_v5 = render_postprocessing_action_script(
        _render_input(authority), scalar_action, renderer_contract_version=5
    )
    scalar_v4 = render_postprocessing_action_script(
        _render_input(authority), scalar_action, renderer_contract_version=4
    )

    assert "SLURM_JOB_ID=${SLURM_ARRAY_JOB_ID:?}" in array_v5
    assert "${SLURM_JOB_ID:?}_${SLURM_ARRAY_TASK_ID:?}" in array_v5
    assert "${SLURM_ARRAY_JOB_ID:?}_${SLURM_ARRAY_TASK_ID:?}" in array_v4
    srun_argv = shlex.split(next(line for line in array_v5.splitlines() if line.startswith("srun ")))
    exec_argv = json.loads(srun_argv[srun_argv.index("--exec-argv-json") + 1])
    assert exec_argv[:2] == ["/usr/bin/bash", "-c"]
    success_argv = shlex.split(
        next(line for line in exec_argv[2].splitlines() if " record-action --phase-runspec " in line)
    )
    assert success_argv[1:4] == [
        "-m",
        "bspp.orchestration.runtime.postprocessing.finalization_bundle",
        "record-action",
    ]
    assert success_argv[success_argv.index("--scheduler-job-id") + 1] == "${SLURM_JOB_ID:?}_${SLURM_ARRAY_TASK_ID:?}"
    assert success_argv[success_argv.index("--task-index") + 1] == "${SLURM_ARRAY_TASK_ID:?}"
    assert scalar_v5 == scalar_v4
    assert hashlib.sha256(scalar_v5.encode()).hexdigest() == hashlib.sha256(scalar_v4.encode()).hexdigest()
    scripts_v4 = _rendered_action_scripts(authority, renderer_contract_version=4)
    assert _postprocessing_submission_id(
        authority, scripts_v4, renderer_contract_version=4
    ) != _postprocessing_submission_id(authority, scripts_v4, renderer_contract_version=5)


def test_materialization_rejects_missing_aws_mount_locator_profile_before_authority_write(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    profile = yaml.safe_load(profile_path.read_text())
    del profile["clusters"]["example-cluster"]["postprocessing_credential_mounts"]
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False))
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match="postprocessing_credential_mounts"):
        materialize_phase(
            plan_path,
            authority_root=authority_root,
            config_path=profile_path,
            source_repo=source_repo,
            clock=lambda: NOW,
            phase_run_id_factory=lambda: RUN_ID,
        )

    assert not authority_root.exists()


def test_unsubmitted_historical_v3_fails_closed_before_transport(
    tmp_path: Path,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    _rewrite_without_credential_mount_snapshot(authority_root)

    def unexpected(_argv: tuple[str, ...]) -> CommandResult:
        raise AssertionError("transport must not be used by an unsubmitted historical V3 authority")

    with pytest.raises(ValueError, match=r"pre-renderer-4.*cannot create a new submission; rematerialize"):
        submit_postprocessing_phase(
            RUN_ID,
            authority_root=authority_root,
            clock=lambda: NOW,
            runner=unexpected,
        )

    assert validate_postprocessing_authority(authority_root, RUN_ID).status == "materialized"


def test_historical_renderer_three_requires_complete_hash_proof_and_resumes_cross_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    intended = _append_historical_renderer_three_submission_intent(authority_root)
    submission = intended.submission_state
    assert submission is not None
    payload = PostprocessingSubmissionIntendedPayload(
        submission_id=submission.submission_id,
        phase_runspec_digest=intended.runspec.digest,
        actions=submission.actions,
    )

    authenticated = render_authenticated_renderer3_scripts(
        intended,
        payload,
        candidate_homes=(Path("/home/example"),),
    )
    assert tuple(authenticated) == tuple(action.action_id for action in intended.runspec.payload.actions)
    altered_actions = list(payload.actions)
    altered_actions[-1] = replace(altered_actions[-1], script_sha256="f" * 64)
    with pytest.raises(ValueError, match="no complete script hash proof"):
        render_authenticated_renderer3_scripts(
            intended,
            replace(payload, actions=tuple(altered_actions)),
            candidate_homes=(Path("/home/example"),),
        )
    with pytest.raises(ValueError, match="no complete script hash proof"):
        render_authenticated_renderer3_scripts(
            intended,
            replace(payload, submission_id="postprocessing-submission-" + "e" * 64),
            candidate_homes=(Path("/home/example"),),
        )
    with pytest.raises(ValueError, match="no complete script hash proof"):
        render_authenticated_renderer3_scripts(
            intended,
            payload,
            candidate_homes=(Path("/wrong-1"), Path("/wrong-2"), Path("/wrong-3"), Path("/home/example")),
        )

    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: Path("/different-operator-home")))
    assert validate_postprocessing_authority(authority_root, RUN_ID).submission_state == submission
    submitted: list[SlurmAction] = []

    def submit_action(_transport: RemoteSlurmTransport, action: SlurmAction) -> SlurmSubmission:
        submitted.append(action)
        job_id = str(3900 + len(submitted))
        result = CommandResult(argv=("sbatch",), returncode=0, stdout=f"{job_id}\n", stderr="")
        return SlurmSubmission(job_id=job_id, command=result.argv, result=result)

    _patch_transport(monkeypatch, submit_action=submit_action)
    result = submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)

    assert result.status == "submitted"
    assert len(submitted) == len(intended.runspec.payload.actions)
    replayed = validate_postprocessing_authority(authority_root, RUN_ID)
    assert sum(event.event_type == "phase-submission-intended" for event in replayed.events) == 1
    assert {action.renderer_contract_version for action in replayed.submission_state.actions} == {3}


def test_submit_preserves_frozen_action_graph_and_resume_requires_exact_task_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    submitted: list[SlurmAction] = []

    def submit_action(_transport: RemoteSlurmTransport, action: SlurmAction) -> SlurmSubmission:
        submitted.append(action)
        job_id = str(1000 + len(submitted))
        result = CommandResult(argv=("sbatch",), returncode=0, stdout=f"{job_id}\n", stderr="")
        return SlurmSubmission(job_id=job_id, command=result.argv, result=result)

    _patch_transport(monkeypatch, submit_action=submit_action)
    result = submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)

    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    action_ids = tuple(action.action_id for action in authority.runspec.payload.actions)
    job_by_action = dict(zip(action_ids, (str(1001 + index) for index in range(len(action_ids))), strict=True))
    assert result.status == "submitted"
    assert tuple(action.action_id for action in submitted) == action_ids
    assert tuple(action.dependency_job_ids for action in submitted) == tuple(
        tuple(job_by_action[dependency] for dependency in action.dependencies)
        for action in authority.runspec.payload.actions
    )

    complete = _observation(authority, job_by_action=job_by_action)
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "query_observation_best_effort",
        lambda _transport, _job_ids, **_kwargs: complete,
    )
    resumed = resume_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)

    assert resumed.status == "submitted"
    assert resumed.details["newly_terminal_action_ids"] == list(action_ids)
    replayed = validate_postprocessing_authority(authority_root, RUN_ID)
    terminal_events = tuple(
        event.payload
        for event in replayed.events
        if isinstance(event.payload, PostprocessingActionTerminalObservedPayload)
    )
    assert len(terminal_events) == len(action_ids)
    array_event = terminal_events[3]
    assert array_event.expected_task_indexes == (853,)
    assert array_event.tasks[0].scheduler_job_id == f"{job_by_action[action_ids[3]]}_853"

    resumed_again = resume_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    assert resumed_again.details["newly_terminal_action_ids"] == []
    replayed_again = validate_postprocessing_authority(authority_root, RUN_ID)
    assert len(replayed_again.terminal_payloads) == len(action_ids)

    scheduler_path = tmp_path / "scheduler-evidence.json"
    exported = export_postprocessing_scheduler_evidence(
        RUN_ID,
        authority_root=authority_root,
        output=scheduler_path,
    )
    scheduler = postprocessing_scheduler_evidence_from_mapping(json.loads(scheduler_path.read_bytes()))
    assert exported.scheduler_evidence_id == scheduler.scheduler_evidence_id
    assert tuple(item.action_id for item in scheduler.actions) == action_ids
    assert scheduler.actions[3].tasks[0].scheduler_job_id == f"{job_by_action[action_ids[3]]}_853"
    assert (
        export_postprocessing_scheduler_evidence(
            RUN_ID,
            authority_root=authority_root,
            output=scheduler_path,
        )
        == exported
    )
    scheduler_path.write_text("{}\n")
    with pytest.raises(ValueError, match="differs from durable authority"):
        export_postprocessing_scheduler_evidence(
            RUN_ID,
            authority_root=authority_root,
            output=scheduler_path,
        )


def test_parallel_terminal_actions_replay_when_action07_arrives_before_action06(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root, authority, job_by_action = _submitted_fixture(tmp_path, monkeypatch, first_job_id=1501)
    action_ids = tuple(action.action_id for action in authority.runspec.payload.actions)
    action06 = action_ids[5]
    action07 = action_ids[6]
    first_completed = {*action_ids[:5], action07}
    first_observation = _observation(
        authority,
        job_by_action=job_by_action,
        completed_action_ids=first_completed,
    )
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "query_observation_best_effort",
        lambda _transport, _job_ids, **_kwargs: first_observation,
    )

    first = resume_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    assert set(first.details["newly_terminal_action_ids"]) == first_completed

    complete = _observation(authority, job_by_action=job_by_action)
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "query_observation_best_effort",
        lambda _transport, _job_ids, **_kwargs: complete,
    )
    second = resume_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    assert second.details["newly_terminal_action_ids"] == [action06, *action_ids[7:]]

    replayed = validate_postprocessing_authority(authority_root, RUN_ID)
    terminal_order = [
        event.payload.action_id
        for event in replayed.events
        if isinstance(event.payload, PostprocessingActionTerminalObservedPayload)
    ]
    assert terminal_order.index(action07) < terminal_order.index(action06)


def test_missing_array_child_stays_unresolved_for_status_resume_and_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    submitted: list[SlurmAction] = []

    def submit_action(_transport: RemoteSlurmTransport, action: SlurmAction) -> SlurmSubmission:
        submitted.append(action)
        job_id = str(2000 + len(submitted))
        result = CommandResult(argv=("sbatch",), returncode=0, stdout=f"{job_id}\n", stderr="")
        return SlurmSubmission(job_id=job_id, command=result.argv, result=result)

    _patch_transport(monkeypatch, submit_action=submit_action)
    submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    action_ids = tuple(action.action_id for action in authority.runspec.payload.actions)
    job_by_action = dict(zip(action_ids, (str(2001 + index) for index in range(len(action_ids))), strict=True))
    array_action = authority.runspec.payload.actions[3]
    incomplete = _observation(
        authority,
        job_by_action=job_by_action,
        omit_task_for=array_action.action_id,
    )
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "query_observation_best_effort",
        lambda _transport, _job_ids, **_kwargs: incomplete,
    )

    resumed = resume_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    assert array_action.action_id not in resumed.details["newly_terminal_action_ids"]
    status = status_postprocessing_phase(RUN_ID, authority_root=authority_root)
    array_status = status.details["actions"][3]
    assert array_status["terminal"] is None
    assert array_status["tasks"] == [
        {
            "task_index": 853,
            "scheduler_job_id": f"{job_by_action[array_action.action_id]}_853",
            "observation_status": "missing",
            "state": None,
            "exit_code": None,
            "source": None,
            "restarts": None,
        }
    ]
    routed = status_phase(RUN_ID, authority_root=authority_root)
    assert routed.to_mapping() == status.to_mapping()
    expected_action = (
        f"  action_id={array_action.action_id} durable_status=submitted job_id={job_by_action[array_action.action_id]} "
        "scheduler_status=missing scheduler_state=COMPLETED scheduler_source=sacct scheduler_exit_code=0:0"
    )
    expected_task = (
        f"    task_index=853 scheduler_job_id={job_by_action[array_action.action_id]}_853 observation_status=missing "
        "state=null source=null exit_code=null"
    )
    assert expected_action in routed.render_table()
    assert expected_task in routed.render_table()
    cancelled = cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    assert cancelled.status == "cancelling"
    assert cancelled.details["pending_parent_job_ids"] == [job_by_action[array_action.action_id]]
    replayed = validate_postprocessing_authority(authority_root, RUN_ID)
    assert replayed.events[-1].event_type != "phase-cancelled"

    complete = _observation(authority, job_by_action=job_by_action)
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "query_observation_best_effort",
        lambda _transport, _job_ids, **_kwargs: complete,
    )
    cancelled = cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    assert cancelled.status == "cancelled"
    replayed = validate_postprocessing_authority(authority_root, RUN_ID)
    cancelled_payload = replayed.events[-1].payload
    assert isinstance(cancelled_payload, PostprocessingCancelledPayload)
    assert cancelled_payload.terminal_parent_job_ids == tuple(sorted(job_by_action.values(), key=int))
    assert cancelled_payload.terminal_action_ids == action_ids
    assert cancelled_payload.cancelled_parent_job_ids == ()


def test_parent_only_cancelled_array_closes_explicit_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root, authority, job_by_action = _submitted_fixture(tmp_path, monkeypatch, first_job_id=2301)
    array_action = authority.runspec.payload.actions[3]
    observation = _observation(
        authority,
        job_by_action=job_by_action,
        parent_only_cancelled_for=array_action.action_id,
    )
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "query_observation_best_effort",
        lambda _transport, _job_ids, **_kwargs: observation,
    )
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "request_job_cancellation",
        lambda _transport, job_id: CommandResult(argv=("scancel", job_id), returncode=0, stdout="", stderr=""),
    )

    resumed = resume_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    assert resumed.status == "submitted"
    replayed = validate_postprocessing_authority(authority_root, RUN_ID)
    assert array_action.action_id not in {payload.action_id for payload in replayed.terminal_payloads}

    result = cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)

    assert result.status == "cancelled"
    replayed = validate_postprocessing_authority(authority_root, RUN_ID)
    assert "phase-array-parent-cancelled-observed" in tuple(event.event_type for event in replayed.events)
    status = status_postprocessing_phase(RUN_ID, authority_root=authority_root)
    task = status.details["actions"][3]["tasks"][0]
    assert task["observation_status"] == "not-instantiated"


@pytest.mark.parametrize(
    "associated_records",
    [
        (
            SlurmJobRecord(
                job_id="2304",
                source="sacct",
                requested_job_id="2304",
                state="CANCELLED",
                exit_code="0:0",
            ),
            SlurmJobRecord(
                job_id="2304_853",
                source="sacct",
                requested_job_id="2304",
                state="CANCELLED",
                exit_code="0:0",
            ),
        ),
        (
            SlurmJobRecord(
                job_id="2304",
                source="sacct",
                requested_job_id="2304",
                state="CANCELLED",
                exit_code="0:0",
            ),
            SlurmJobRecord(
                job_id="2304.batch",
                source="sacct",
                requested_job_id="2304",
                state="CANCELLED",
                exit_code="0:0",
            ),
        ),
        (
            SlurmJobRecord(
                job_id="2304_[853-900]",
                source="sacct",
                requested_job_id="2304",
                state="CANCELLED",
                exit_code="0:0",
            ),
        ),
        (
            SlurmJobRecord(
                job_id="2304",
                source="sacct",
                requested_job_id="2304",
                state="CANCELLED",
                exit_code=None,
            ),
        ),
    ],
    ids=("child", "step", "compressed-placeholder", "missing-exit"),
)
def test_array_parent_cancellation_requires_one_exact_parent_and_no_associated_records(
    tmp_path: Path,
    associated_records: tuple[SlurmJobRecord, ...],
) -> None:
    authority_root = _materialized_authority(tmp_path)
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    array_action = authority.runspec.payload.actions[3]

    assert (
        _array_parent_cancelled_before_task_instantiation(
            array_action,
            "2304",
            associated_records,
        )
        is None
    )


def test_array_parent_cancellation_ignores_slurm_step_records() -> None:
    action = type("ArrayAction", (), {"expected_task_indexes": (853,)})()
    records = (
        SlurmJobRecord(
            job_id="2304",
            source="sacct",
            requested_job_id="2304",
            state="CANCELLED",
            exit_code="0:0",
        ),
        SlurmJobRecord(
            job_id="2304.batch",
            source="sacct",
            requested_job_id=None,
            state="COMPLETED",
            exit_code="0:0",
        ),
        SlurmJobRecord(
            job_id="2304.extern",
            source="sacct",
            requested_job_id=None,
            state="COMPLETED",
            exit_code="0:0",
        ),
        SlurmJobRecord(
            job_id="2304.0",
            source="sacct",
            requested_job_id=None,
            state="COMPLETED",
            exit_code="0:0",
        ),
    )
    evidence = _array_parent_cancelled_before_task_instantiation(action, "2304", records)
    assert evidence is not None
    assert evidence.scheduler_job_id == "2304"
    assert evidence.state == "CANCELLED"
    assert evidence.exit_code == "0:0"


def test_submission_rejects_expired_qualification_before_transport_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    effects: list[str] = []

    def unexpected(*_args: object, **_kwargs: object) -> object:
        effects.append("transport")
        raise AssertionError("transport must not be used after qualification expiry")

    monkeypatch.setattr(RemoteSlurmTransport, "stage_immutable_artifact", unexpected)
    monkeypatch.setattr(RemoteSlurmTransport, "command", unexpected)
    monkeypatch.setattr(RemoteSlurmTransport, "submit_action", unexpected)

    expires_at = parse_postprocessing_timestamp(
        validate_postprocessing_authority(authority_root, RUN_ID).runspec.payload.qualified_runtime.expires_at
    )
    with pytest.raises(ValueError, match="Qualification is not current"):
        submit_postprocessing_phase(
            RUN_ID,
            authority_root=authority_root,
            clock=lambda: expires_at,
        )

    assert effects == []
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    assert tuple(event.event_type for event in authority.events) == ("phase-materialized",)


def test_authority_replay_rejects_dispatch_that_no_longer_binds_submission_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    submitted: list[SlurmAction] = []

    def submit_action(_transport: RemoteSlurmTransport, action: SlurmAction) -> SlurmSubmission:
        submitted.append(action)
        job_id = str(2500 + len(submitted))
        result = CommandResult(argv=("sbatch",), returncode=0, stdout=f"{job_id}\n", stderr="")
        return SlurmSubmission(job_id=job_id, command=result.argv, result=result)

    _patch_transport(monkeypatch, submit_action=submit_action)
    submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    event_path = next((authority_root / RUN_ID / "events").glob("*-phase-action-dispatch-intended.json"))
    event = json.loads(event_path.read_bytes())
    event["payload"]["scheduler_correlation_token"] = "bspp-pp-" + "f" * 48
    _write_json(event_path, event)

    with pytest.raises(ValueError, match="dispatch differs from submission"):
        validate_postprocessing_authority(authority_root, RUN_ID)


def test_durable_submission_intent_can_resume_after_qualification_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)

    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated transport interruption")

    monkeypatch.setattr(RemoteSlurmTransport, "stage_immutable_artifact", interrupted)
    with pytest.raises(OSError, match="transport interruption"):
        submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    intended = validate_postprocessing_authority(authority_root, RUN_ID)
    assert tuple(event.event_type for event in intended.events[-1:]) == ("phase-submission-intended",)

    submitted: list[SlurmAction] = []

    def submit_action(_transport: RemoteSlurmTransport, action: SlurmAction) -> SlurmSubmission:
        submitted.append(action)
        job_id = str(3000 + len(submitted))
        result = CommandResult(argv=("sbatch",), returncode=0, stdout=f"{job_id}\n", stderr="")
        return SlurmSubmission(job_id=job_id, command=result.argv, result=result)

    _patch_transport(monkeypatch, submit_action=submit_action)
    expires_at = parse_postprocessing_timestamp(intended.runspec.payload.qualified_runtime.expires_at)
    result = submit_postprocessing_phase(
        RUN_ID,
        authority_root=authority_root,
        clock=lambda: expires_at,
    )

    assert result.status == "submitted"
    assert len(submitted) == 9
    replayed = validate_postprocessing_authority(authority_root, RUN_ID)
    assert sum(event.event_type == "phase-submission-intended" for event in replayed.events) == 1


def test_cancel_before_submission_is_durable_and_scheduler_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("scheduler transport must not be used before submission")

    monkeypatch.setattr(RemoteSlurmTransport, "query_observation_best_effort", unexpected)
    monkeypatch.setattr(RemoteSlurmTransport, "request_job_cancellation", unexpected)
    result = cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)

    assert result.status == "cancelled"
    assert result.details["cancellation_requested_job_ids"] == []
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    assert tuple(event.event_type for event in authority.events[-2:]) == (
        "phase-cancellation-intended",
        "phase-cancelled",
    )


def test_cancel_resumes_incomplete_durable_request_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root, authority, job_by_action = _submitted_fixture(tmp_path, monkeypatch, first_job_id=5001)
    pending = _observation(authority, job_by_action=job_by_action, completed_action_ids=set())
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "query_observation_best_effort",
        lambda _transport, _job_ids, **_kwargs: pending,
    )
    calls: list[str] = []

    def interrupted(_transport: RemoteSlurmTransport, job_id: str) -> CommandResult:
        calls.append(job_id)
        raise OSError("simulated crash after durable cancellation intent")

    monkeypatch.setattr(RemoteSlurmTransport, "request_job_cancellation", interrupted)
    with pytest.raises(OSError, match="after durable cancellation intent"):
        cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)

    first_action = authority.runspec.payload.actions[0].action_id
    interrupted_authority = validate_postprocessing_authority(authority_root, RUN_ID)
    first_action_events = [
        event
        for event in interrupted_authority.events
        if isinstance(event.payload, PostprocessingJobCancellationRequestIntendedPayload)
        and event.payload.action_id == first_action
    ]
    assert [event.event_type for event in first_action_events] == ["phase-job-cancellation-request-intended"]

    monkeypatch.setattr(
        RemoteSlurmTransport,
        "request_job_cancellation",
        lambda _transport, job_id: CommandResult(argv=("scancel", job_id), returncode=0, stdout="", stderr=""),
    )
    resumed = cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    assert resumed.status == "cancelling"
    replayed = validate_postprocessing_authority(authority_root, RUN_ID)
    first_action_events = [
        event
        for event in replayed.events
        if isinstance(
            event.payload,
            (PostprocessingJobCancellationRequestIntendedPayload, PostprocessingJobCancellationRequestResultPayload),
        )
        and event.payload.action_id == first_action
    ]
    assert [event.event_type for event in first_action_events] == [
        "phase-job-cancellation-request-intended",
        "phase-job-cancellation-request-result",
    ]
    assert {event.payload.request_ordinal for event in first_action_events} == {1}
    assert calls == [job_by_action[first_action]]


def test_cancel_nonzero_result_is_durable_and_retried_with_a_new_ordinal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root, authority, job_by_action = _submitted_fixture(tmp_path, monkeypatch, first_job_id=6001)
    pending = _observation(authority, job_by_action=job_by_action, completed_action_ids=set())
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "query_observation_best_effort",
        lambda _transport, _job_ids, **_kwargs: pending,
    )
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "request_job_cancellation",
        lambda _transport, job_id: CommandResult(
            argv=("scancel", job_id), returncode=1, stdout="", stderr=f"failed {job_id}"
        ),
    )
    with pytest.raises(ValueError, match="failed"):
        cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)

    monkeypatch.setattr(
        RemoteSlurmTransport,
        "request_job_cancellation",
        lambda _transport, job_id: CommandResult(argv=("scancel", job_id), returncode=0, stdout="", stderr=""),
    )
    retried = cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    assert retried.status == "cancelling"

    first_action = authority.runspec.payload.actions[0].action_id
    replayed = validate_postprocessing_authority(authority_root, RUN_ID)
    first_action_events = [
        event
        for event in replayed.events
        if isinstance(
            event.payload,
            (PostprocessingJobCancellationRequestIntendedPayload, PostprocessingJobCancellationRequestResultPayload),
        )
        and event.payload.action_id == first_action
    ]
    assert [event.payload.request_ordinal for event in first_action_events] == [1, 1, 2, 2]
    assert [
        event.payload.return_code
        for event in first_action_events
        if isinstance(event.payload, PostprocessingJobCancellationRequestResultPayload)
    ] == [1, 0]


def test_retry_materializes_clean_immutable_successor_without_carry_forward(tmp_path: Path) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    predecessor_projection_path = authority_root / RUN_ID / "attempts" / predecessor.attempt_id / "legacy-runspec.yaml"
    predecessor_projection_bytes = predecessor_projection_path.read_bytes()
    runtime_resolver = _retry_resolution(tmp_path=tmp_path, authority=predecessor)

    result = retry_postprocessing_phase(
        RUN_ID,
        authority_root=authority_root,
        config_path=tmp_path / "profiles.yaml",
        source_repo=tmp_path / "source-repo",
        clock=lambda: NOW,
        runtime_resolver=runtime_resolver,
    )

    successor = validate_postprocessing_authority(authority_root, RUN_ID)
    assert result.successor_attempt_id == "attempt-0002"
    assert successor.attempt_id == "attempt-0002"
    assert successor.status == "materialized"
    assert successor.current_attempt_projection_complete is True
    assert successor.runspec.payload.action_graph_digest == predecessor.runspec.payload.action_graph_digest
    assert successor.runspec.payload.logical_inputs == predecessor.runspec.payload.logical_inputs
    assert successor.runspec.payload.physical_inputs == predecessor.runspec.payload.physical_inputs
    assert successor.runspec.payload.scientific_identity == predecessor.runspec.payload.scientific_identity
    assert successor.runspec.payload.action_semantics == predecessor.runspec.payload.action_semantics
    assert successor.runspec.payload.attempt_paths != predecessor.runspec.payload.attempt_paths
    assert successor.runspec.cluster_output_root == predecessor.runspec.cluster_output_root
    assert successor.runspec.object_output_base_prefix == predecessor.runspec.object_output_base_prefix
    assert successor.runspec.payload.qualified_runtime != predecessor.runspec.payload.qualified_runtime
    predecessor_runtime = predecessor.runspec.payload.qualified_runtime
    successor_runtime = successor.runspec.payload.qualified_runtime
    assert successor_runtime.image_path != predecessor_runtime.image_path
    assert successor_runtime.image_sha256 != predecessor_runtime.image_sha256
    assert successor_runtime.source_package_path != predecessor_runtime.source_package_path
    assert successor_runtime.source_package_identity_digest != predecessor_runtime.source_package_identity_digest
    assert successor_runtime.source_revision != predecessor_runtime.source_revision
    assert successor_runtime.toolkit_package_path != predecessor_runtime.toolkit_package_path
    assert successor_runtime.toolkit_identity_digest != predecessor_runtime.toolkit_identity_digest
    assert successor_runtime.runtime_ipsae_binary_sha256 != predecessor_runtime.runtime_ipsae_binary_sha256
    assert successor_runtime.runtime_component_identity_digest != predecessor_runtime.runtime_component_identity_digest
    predecessor_legacy = yaml.safe_load(predecessor.legacy_runspec_bytes)
    successor_legacy = yaml.safe_load(successor.legacy_runspec_bytes)
    successor_paths = successor.runspec.payload.attempt_paths
    successor_output = Path(successor_paths.output_dir)
    assert predecessor_projection_path.read_bytes() == predecessor_projection_bytes
    assert successor_legacy["dataset"]["name"] == predecessor_legacy["dataset"]["name"]
    assert successor_legacy["dataset"]["run_id"] == successor_paths.legacy_run_id
    assert successor_legacy["paths"] == {
        **predecessor_legacy["paths"],
        "staging_dir": successor_paths.staging_dir,
        "output_dir": successor_paths.output_dir,
        "log_dir": str(successor_output / "logs"),
        "recipe_dir": str(successor_output / "rendered_recipe"),
    }
    assert successor_legacy["storage"]["s3_output_prefix"] == successor_paths.object_prefix
    assert successor_legacy["storage"]["local_tar_dir"] == str(successor_output / "local_tars")
    assert successor_legacy["storage"]["local_tar_manifest_csv"] == str(successor_output / "local_tars.csv")
    assert successor_legacy["analysis_metadata"]["csv_path"] == str(successor_output / "analysis_metadata.csv")
    assert successor_legacy["analysis_metadata"]["parquet_path"] == str(successor_output / "analysis_metadata.parquet")
    assert successor_legacy["analysis_metadata"]["selected_ids_path"] == str(
        successor_output / "high_quality_model_ids.txt"
    )
    assert predecessor.runspec.payload.attempt_paths.output_dir.encode() not in successor.legacy_runspec_bytes
    assert successor_legacy["resources"]["gpu_worker"] != predecessor_legacy["resources"]["gpu_worker"]
    assert successor.runspec.cluster.profile_name == predecessor.runspec.cluster.profile_name
    assert successor.runspec.cluster.runtime_image != predecessor.runspec.cluster.runtime_image
    assert successor.runspec.payload.execution_projection.phase_identity_digest != (
        predecessor.runspec.payload.execution_projection.phase_identity_digest
    )
    assert successor.acceptance_policy_bytes == predecessor.acceptance_policy_bytes
    assert successor.runtime_qualification_bytes != predecessor.runtime_qualification_bytes
    attempt_root = authority_root / RUN_ID / "attempts" / "attempt-0002"
    assert sorted(path.name for path in attempt_root.iterdir()) == [
        "acceptance-policy.json",
        "legacy-runspec.yaml",
        "phase-runspec.json",
        "runtime-qualification.json",
    ]
    retry_events = tuple(
        event.payload for event in successor.events if isinstance(event.payload, PostprocessingAttemptRetriedPayload)
    )
    assert len(retry_events) == 1
    assert retry_events[0].carry_forward is None


def test_retry_of_unsubmitted_historical_v3_snapshots_mounts_for_renderer_four(tmp_path: Path) -> None:
    authority_root = _materialized_authority(tmp_path)
    _rewrite_without_credential_mount_snapshot(authority_root)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    assert predecessor.runspec.credential_mounts is None

    retry_postprocessing_phase(
        RUN_ID,
        authority_root=authority_root,
        config_path=tmp_path / "profiles.yaml",
        source_repo=tmp_path / "source-repo",
        clock=lambda: NOW,
        runtime_resolver=_retry_resolution(tmp_path=tmp_path, authority=predecessor),
    )

    successor = validate_postprocessing_authority(authority_root, RUN_ID)
    assert successor.runspec.credential_mounts == PostprocessingCredentialMountSnapshot(
        aws_shared_credentials_file="/home/example/.aws/credentials",
        aws_config_file="/home/example/.aws/config",
    )
    assert _renderer_contract_for_authority(successor) == 5


def test_failed_submitted_renderer_three_retry_preserves_predecessor_and_selects_renderer_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    _append_historical_renderer_three_submission_intent(authority_root)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: Path("/different-operator-home")))
    next_job_id = 9100

    def submit_action(_transport: RemoteSlurmTransport, action: SlurmAction) -> SlurmSubmission:
        nonlocal next_job_id
        job_id = str(next_job_id)
        next_job_id += 1
        result = CommandResult(argv=("sbatch",), returncode=0, stdout=f"{job_id}\n", stderr="")
        return SlurmSubmission(job_id=job_id, command=result.argv, result=result)

    _patch_transport(monkeypatch, submit_action=submit_action)
    submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    submitted = validate_postprocessing_authority(authority_root, RUN_ID)
    assert submitted.submission_state is not None
    assert {item.renderer_contract_version for item in submitted.submission_state.actions} == {3}
    job_by_action = submitted.submission_state.job_ids_by_action()
    first_action_id = submitted.runspec.payload.actions[0].action_id
    records: list[SlurmJobRecord] = []
    selected: list[SlurmJobState] = []
    for action in submitted.runspec.payload.actions:
        parent_job_id = job_by_action[action.action_id]
        state, exit_code = ("FAILED", "1:0") if action.action_id == first_action_id else ("CANCELLED", "0:0")
        selected.append(SlurmJobState(parent_job_id, state, "sacct", exit_code))
        records.extend(
            SlurmJobRecord(
                job_id=parent_job_id if task_index is None else f"{parent_job_id}_{task_index}",
                source="sacct",
                requested_job_id=parent_job_id,
                state=state,
                exit_code=exit_code,
            )
            for task_index in (action.expected_task_indexes or (None,))
        )
    observation = SlurmObservation(
        requested_job_ids=tuple(job_by_action[action.action_id] for action in submitted.runspec.payload.actions),
        squeue=SlurmCommandSnapshot(kind="squeue", argv=(), returncode=0, parser="fixture"),
        sacct=SlurmCommandSnapshot(kind="sacct", argv=(), returncode=0, parser="fixture"),
        squeue_jobs=(),
        sacct_jobs=tuple(records),
        selected_states=tuple(selected),
    )
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "query_observation_best_effort",
        lambda *_args, **_kwargs: observation,
    )
    failed = resume_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    assert failed.status == "failed"
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    authority_path = authority_root / RUN_ID
    predecessor_files = {
        path.relative_to(authority_path): path.read_bytes()
        for path in authority_path.rglob("*")
        if path.is_file() and not path.is_symlink()
    }

    retry_postprocessing_phase(
        RUN_ID,
        authority_root=authority_root,
        config_path=tmp_path / "profiles.yaml",
        source_repo=tmp_path / "source-repo",
        clock=lambda: NOW,
        runtime_resolver=_retry_resolution(tmp_path=tmp_path, authority=predecessor),
    )

    successor = validate_postprocessing_authority(authority_root, RUN_ID)
    assert successor.attempt_id == "attempt-0002"
    assert successor.runspec.credential_mounts == PostprocessingCredentialMountSnapshot(
        aws_shared_credentials_file="/home/example/.aws/credentials",
        aws_config_file="/home/example/.aws/config",
    )
    assert _renderer_contract_for_authority(successor) == 5
    submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    resubmitted = validate_postprocessing_authority(authority_root, RUN_ID)
    assert resubmitted.submission_state is not None
    assert {item.renderer_contract_version for item in resubmitted.submission_state.actions} == {5}
    assert {relative: (authority_path / relative).read_bytes() for relative in predecessor_files} == predecessor_files


def test_failed_renderer_four_attempt_two_is_immutable_and_retry_attempt_three_submits_renderer_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    first_attempt = validate_postprocessing_authority(authority_root, RUN_ID)
    retry_postprocessing_phase(
        RUN_ID,
        authority_root=authority_root,
        config_path=tmp_path / "profiles.yaml",
        source_repo=tmp_path / "source-repo",
        clock=lambda: NOW,
        runtime_resolver=_retry_resolution(tmp_path=tmp_path, authority=first_attempt),
    )
    attempt_two = validate_postprocessing_authority(authority_root, RUN_ID)
    assert attempt_two.attempt_id == "attempt-0002"
    scripts = _rendered_action_scripts(attempt_two, renderer_contract_version=4)
    submission_id = _postprocessing_submission_id(attempt_two, scripts, renderer_contract_version=4)
    append_event(
        attempt_two,
        event_type="phase-submission-intended",
        occurred_at="2026-09-03T12:00:00Z",
        payload=PostprocessingSubmissionIntendedPayload(
            submission_id=submission_id,
            phase_runspec_digest=attempt_two.runspec.digest,
            actions=tuple(
                PostprocessingSubmissionActionPlan(
                    action_id=action.action_id,
                    runtime_action_digest=mapping_digest(action.to_mapping()),
                    dependencies=action.dependencies,
                    cluster_script_path=str(postprocessing_cluster_action_script(attempt_two.runspec, action)),
                    script_sha256=hashlib.sha256(scripts[action.action_id].encode()).hexdigest(),
                    scheduler_correlation_token=postprocessing_scheduler_correlation_token(attempt_two.runspec, action),
                    renderer_contract_version=4,
                )
                for action in attempt_two.runspec.payload.actions
            ),
        ),
    )
    next_job_id = 9200

    def submit_action(_transport: RemoteSlurmTransport, action: SlurmAction) -> SlurmSubmission:
        nonlocal next_job_id
        job_id = str(next_job_id)
        next_job_id += 1
        result = CommandResult(argv=("sbatch",), returncode=0, stdout=f"{job_id}\n", stderr="")
        return SlurmSubmission(job_id=job_id, command=result.argv, result=result)

    _patch_transport(monkeypatch, submit_action=submit_action)
    submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    submitted = validate_postprocessing_authority(authority_root, RUN_ID)
    assert submitted.submission_state is not None
    assert {item.renderer_contract_version for item in submitted.submission_state.actions} == {4}
    job_by_action = submitted.submission_state.job_ids_by_action()
    failed_action_id = submitted.runspec.payload.actions[0].action_id
    selected: list[SlurmJobState] = []
    records: list[SlurmJobRecord] = []
    for action in submitted.runspec.payload.actions:
        parent_job_id = job_by_action[action.action_id]
        state, exit_code = ("FAILED", "1:0") if action.action_id == failed_action_id else ("CANCELLED", "0:0")
        selected.append(SlurmJobState(parent_job_id, state, "sacct", exit_code))
        records.extend(
            SlurmJobRecord(
                job_id=parent_job_id if task_index is None else f"{parent_job_id}_{task_index}",
                source="sacct",
                requested_job_id=parent_job_id,
                state=state,
                exit_code=exit_code,
            )
            for task_index in (action.expected_task_indexes or (None,))
        )
    observation = SlurmObservation(
        requested_job_ids=tuple(job_by_action[action.action_id] for action in submitted.runspec.payload.actions),
        squeue=SlurmCommandSnapshot(kind="squeue", argv=(), returncode=0, parser="fixture"),
        sacct=SlurmCommandSnapshot(kind="sacct", argv=(), returncode=0, parser="fixture"),
        squeue_jobs=(),
        sacct_jobs=tuple(records),
        selected_states=tuple(selected),
    )
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "query_observation_best_effort",
        lambda *_args, **_kwargs: observation,
    )
    assert resume_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW).status == "failed"
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    authority_path = authority_root / RUN_ID
    predecessor_files = {
        path.relative_to(authority_path): path.read_bytes()
        for path in authority_path.rglob("*")
        if path.is_file() and not path.is_symlink()
    }

    retry_postprocessing_phase(
        RUN_ID,
        authority_root=authority_root,
        config_path=tmp_path / "profiles.yaml",
        source_repo=tmp_path / "source-repo",
        clock=lambda: NOW,
        runtime_resolver=_retry_resolution(tmp_path=tmp_path, authority=predecessor),
    )

    successor = validate_postprocessing_authority(authority_root, RUN_ID)
    assert successor.attempt_id == "attempt-0003"
    submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    resubmitted = validate_postprocessing_authority(authority_root, RUN_ID)
    assert resubmitted.submission_state is not None
    assert {item.renderer_contract_version for item in resubmitted.submission_state.actions} == {5}
    assert {relative: (authority_path / relative).read_bytes() for relative in predecessor_files} == predecessor_files


def test_retry_recovers_projection_after_durable_event_interruption(tmp_path: Path) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    runtime_resolver = _retry_resolution(tmp_path=tmp_path, authority=predecessor)

    def interrupted(_path: Path, _payload: bytes) -> None:
        raise OSError("simulated Retry projection interruption")

    with pytest.raises(OSError, match="projection interruption"):
        retry_postprocessing_phase(
            RUN_ID,
            authority_root=authority_root,
            config_path=tmp_path / "profiles.yaml",
            clock=lambda: NOW,
            runtime_resolver=runtime_resolver,
            projection_store=PostprocessingRetryProjectionStore(publish_exact=interrupted),
        )
    incomplete = validate_postprocessing_authority(authority_root, RUN_ID)
    assert incomplete.attempt_id == "attempt-0002"
    assert incomplete.current_attempt_projection_complete is False

    recovered = retry_postprocessing_phase(
        RUN_ID,
        authority_root=authority_root,
        config_path=tmp_path / "profiles.yaml",
        clock=lambda: NOW,
    )
    assert recovered.successor_attempt_id == "attempt-0002"
    assert validate_postprocessing_authority(authority_root, RUN_ID).current_attempt_projection_complete is True


def test_retry_rejects_carry_forward_without_reading_request(tmp_path: Path) -> None:
    authority_root = _materialized_authority(tmp_path)
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    unreadable_or_missing = tmp_path / "must-not-be-read.json"

    with pytest.raises(ValueError, match="does not support --carry-forward"):
        retry_phase(
            RUN_ID,
            authority_root=authority_root,
            config_path=tmp_path / "unused-profiles.yaml",
            carry_forward_path=unreadable_or_missing,
            clock=lambda: NOW,
        )

    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    assert authority.attempt_id == "attempt-0001"


def test_finalization_seals_exact_output_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _finalizable_authority(tmp_path, monkeypatch)

    first = _finalize_fixture(fixture)
    second = _finalize_fixture(fixture)

    assert second == first
    authority = validate_postprocessing_authority(fixture.authority_root, RUN_ID)
    assert authority.status == "accepted"
    assert authority.sealed is True
    finalized = tuple(
        event.payload for event in authority.events if isinstance(event.payload, PostprocessingFinalizedPayload)
    )
    assert len(finalized) == 1
    payload = finalized[0]
    assert payload.receipt.baseline_id == "task853-fixed-fork"
    assert payload.receipt.acceptance_policy_size_bytes == len(authority.acceptance_policy_bytes)
    assert [item.root_name for item in payload.output_handoff.artifact_set.children] == [
        "acceptance-evidence",
        "scientific-output",
    ]

    fixture.scientific_path.write_bytes(b"changed remote parquet bytes")
    assert _finalize_fixture(fixture) == first
    assert len(tuple((fixture.authority_root / RUN_ID / "events").glob("*.json"))) == len(authority.events)


def test_finalization_rejects_changed_fetched_member_without_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _finalizable_authority(tmp_path, monkeypatch)
    capture = fixture.handoff / "acceptance/captures/acceptance-semantic.json"
    capture.write_bytes(capture.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="differs from its index"):
        _finalize_fixture(fixture)

    replayed = validate_postprocessing_authority(fixture.authority_root, RUN_ID)
    assert replayed.status == "submitted"
    assert replayed.sealed is False
    assert all(not isinstance(event.payload, PostprocessingFinalizedPayload) for event in replayed.events)


def test_finalization_rejects_changed_fetched_runtime_input_attestations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _finalizable_authority(tmp_path, monkeypatch)
    attestations = fixture.handoff / "inputs/runtime-input-attestations.json"
    attestations.write_bytes(attestations.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="differs from its index"):
        _finalize_fixture(fixture)


def test_finalization_uses_runtime_tar_manifest_without_opening_scientific_tar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _finalizable_authority(tmp_path, monkeypatch)
    fixture.scientific_tar.unlink()

    result = _finalize_fixture(fixture)
    assert result.status == "accepted"
    authority = validate_postprocessing_authority(fixture.authority_root, RUN_ID)
    finalized = next(
        event.payload for event in authority.events if isinstance(event.payload, PostprocessingFinalizedPayload)
    )
    members = finalized.scientific_output_inventory.members
    tar_member = next(item for item in members if item.path == "local_tars/shard_1/batch_0.tar")
    metadata_member = next(item for item in members if item.path == "local_tars/metadata/shard_1_metadata.tar")
    assert tar_member.verification_kind == "inventory-metadata-v1"
    assert metadata_member.verification_kind == "content-sha256-v1"


def test_finalization_rejects_changed_fetched_tar_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _finalizable_authority(tmp_path, monkeypatch)
    manifest = next((fixture.handoff / "outputs/tar-manifests").iterdir())
    manifest.write_bytes(manifest.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="differs from its index"):
        _finalize_fixture(fixture)


def test_finalization_rejects_identical_aggregate_copy_outside_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _finalizable_authority(tmp_path, monkeypatch)
    copied_aggregate = tmp_path / "aggregate-action-evidence-copy.json"
    copied_aggregate.write_bytes(fixture.aggregate.read_bytes())

    with pytest.raises(ValueError, match="must be the exact indexed member beneath --handoff"):
        finalize_postprocessing_phase(
            RUN_ID,
            authority_root=fixture.authority_root,
            scheduler_evidence_path=fixture.scheduler_evidence,
            aggregate_action_evidence_path=copied_aggregate,
            handoff_path=fixture.handoff,
            acceptance_adjudication_path=fixture.adjudication,
            clock=lambda: NOW,
        )


def _finalizable_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    restarts: int | None = 0,
) -> _FinalizationFixture:
    authority_root = _materialized_authority(tmp_path)
    submitted: list[SlurmAction] = []

    def submit_action(_transport: RemoteSlurmTransport, action: SlurmAction) -> SlurmSubmission:
        submitted.append(action)
        job_id = str(4000 + len(submitted))
        result = CommandResult(argv=("sbatch",), returncode=0, stdout=f"{job_id}\n", stderr="")
        return SlurmSubmission(job_id=job_id, command=result.argv, result=result)

    _patch_transport(monkeypatch, submit_action=submit_action)
    submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    action_ids = tuple(action.action_id for action in authority.runspec.payload.actions)
    job_by_action = dict(zip(action_ids, (str(4001 + index) for index in range(len(action_ids))), strict=True))
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "query_observation_best_effort",
        lambda _transport, _job_ids, **_kwargs: _observation(authority, job_by_action=job_by_action, restarts=restarts),
    )
    resume_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    output_root = Path(authority.runspec.payload.attempt_paths.output_dir)
    evidence_root = Path(authority.runspec.payload.attempt_paths.evidence_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    scientific_path = output_root / "analysis" / "analysis_metadata.parquet"
    scientific_path.parent.mkdir(parents=True, exist_ok=True)
    scientific_path.write_bytes(b"parquet fixture bytes")
    scientific_tar = output_root / "local_tars/shard_1/batch_0.tar"
    scientific_tar.parent.mkdir(parents=True)
    with tarfile.open(scientific_tar, "w") as archive:
        member = tarfile.TarInfo("payload/result.json")
        payload = b'{"ok": true}\n'
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    metadata_tar = output_root / "local_tars/metadata/shard_1_metadata.tar"
    metadata_tar.parent.mkdir(parents=True)
    metadata_tar.write_bytes(b"metadata tar identity fixture")
    (output_root / "local_tars.csv").write_text(
        "tar_type,tar_name,shard_id,size_bytes\n"
        f"batch,batch_0.tar,1,{scientific_tar.stat().st_size}\n"
        f"metadata,shard_1_metadata.tar,1,{metadata_tar.stat().st_size}\n"
    )
    _write_final_acceptance(authority, evidence_root, tar_relative_paths=("shard_1/batch_0.tar",))
    _write_input_attestations(authority, evidence_root)
    phase_runspec = authority.authority_path / "attempts" / authority.attempt_id / "phase-runspec.json"
    generate_scientific_output_root(phase_runspec_path=phase_runspec, workers=2)
    for action in authority.runspec.payload.actions[:-1]:
        for task_index in action.expected_task_indexes or (None,):
            scheduler_job_id = job_by_action[action.action_id]
            if task_index is not None:
                scheduler_job_id = f"{scheduler_job_id}_{task_index}"
            record_successful_action_task(
                phase_runspec_path=phase_runspec,
                action_id=action.action_id,
                command_digest=postprocessing_action_command_digest(_render_input(authority), action),
                scheduler_job_id=scheduler_job_id,
                task_index=task_index,
                completed_at="2026-09-03T12:00:00Z",
            )
    action09 = authority.runspec.payload.actions[-1]
    projection_path = authority.authority_path / authority.runspec.payload.execution_projection.document_location
    policy_path = authority.authority_path / authority.runspec.payload.acceptance_policy.location
    published = publish_action09_finalization_bundle(
        phase_runspec_path=phase_runspec,
        execution_projection_path=projection_path,
        acceptance_policy_path=policy_path,
        command_digest=postprocessing_action_command_digest(_render_input(authority), action09),
        workers=2,
        assembled_at="2026-09-03T12:00:00Z",
    )
    scheduler_evidence = tmp_path / "scheduler-evidence.json"
    export_postprocessing_scheduler_evidence(
        RUN_ID,
        authority_root=authority_root,
        output=scheduler_evidence,
    )
    return _FinalizationFixture(
        authority_root=authority_root,
        scientific_path=scientific_path,
        scientific_tar=scientific_tar,
        evidence_root=evidence_root,
        handoff=published.destination,
        scheduler_evidence=scheduler_evidence,
    )


def _finalize_fixture(fixture: _FinalizationFixture) -> PostprocessingPhaseFinalizationResult:
    return finalize_postprocessing_phase(
        RUN_ID,
        authority_root=fixture.authority_root,
        scheduler_evidence_path=fixture.scheduler_evidence,
        aggregate_action_evidence_path=fixture.aggregate,
        handoff_path=fixture.handoff,
        acceptance_adjudication_path=fixture.adjudication,
        clock=lambda: NOW,
    )


def _write_final_acceptance(
    authority: object,
    evidence_root: Path,
    *,
    tar_relative_paths: tuple[str, ...] = (),
) -> None:
    parity_path = "acceptance/tar_payload_parity/tar_payload_parity_report.json"
    semantic_path = "acceptance/semantic_acceptance/semantic_acceptance_summary.json"
    verify_path = "acceptance/verify_evidence/acceptance_evidence_report.json"
    output_root = authority.runspec.payload.attempt_paths.output_dir
    baseline_root = next(
        item.locator for item in authority.runspec.payload.physical_inputs if item.name == "baseline-output"
    )
    _write_json(
        evidence_root / parity_path,
        {
            "baseline_dir": baseline_root,
            "candidate_dir": output_root,
            "relative_dir": "local_tars",
            "candidate_tar_count": len(tar_relative_paths),
            "ok": True,
            "inventory_errors": [],
            "baseline_only_tars": [],
            "candidate_only_tars": [],
            "payload_mismatch_count": 0,
            "error_count": 0,
            "files": [{"relative_path": item} for item in tar_relative_paths],
        },
    )
    _write_json(
        evidence_root / semantic_path,
        {
            "baseline_dir": baseline_root,
            "candidate_dir": output_root,
            "ok": False,
            "errors": ["known-task853-residual"],
        },
    )
    _write_json(
        evidence_root / verify_path,
        {
            "schema_version": 1,
            "ok": False,
            "parity_report_path": parity_path,
            "semantic_report_path": semantic_path,
            "issues": [{"check": "semantic-acceptance", "message": "ok is not true", "report_path": semantic_path}],
        },
    )
    raw_root = evidence_root / "phase-acceptance/raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    policy_path = authority.authority_path / authority.runspec.payload.acceptance_policy.location
    policy_sha = authority.runspec.payload.acceptance_policy.sha256
    captures = []
    for step in (
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    ):
        stdout = raw_root / f"{step}.stdout"
        stderr = raw_root / f"{step}.stderr"
        stdout.write_text(f"{step} stdout\n")
        stderr.write_text(f"{step} stderr\n")
        captures.append(
            capture_acceptance(
                policy_path=policy_path,
                expected_policy_sha256=policy_sha,
                evidence_root=evidence_root,
                phase_run_id=authority.phase_run_id,
                attempt_id=authority.attempt_id,
                action_id=next(item.action_id for item in authority.runspec.payload.actions if item.step_name == step),
                step_name=step,
                raw_exit_code=0 if step == "acceptance-tar-payload-parity" else 1,
                raw_stdout_path=stdout,
                raw_stderr_path=stderr,
                output_path=evidence_root / "phase-acceptance" / f"{step}-capture.json",
                completed_at="2026-09-03T12:00:00Z",
            )
        )
    adjudication = adjudicate_acceptance(
        policy_path=policy_path,
        expected_policy_sha256=policy_sha,
        evidence_root=evidence_root,
        phase_run_id=authority.phase_run_id,
        attempt_id=authority.attempt_id,
        output_path=evidence_root / "phase-acceptance/adjudication.json",
        adjudicated_at="2026-09-03T12:00:00Z",
        expected_baseline_locator=baseline_root,
    )
    assert adjudication.result == "passed"
    assert adjudication.capture_digests == tuple(item.digest for item in captures)


def _write_input_attestations(authority: object, evidence_root: Path) -> None:
    physical = {item.name: item for item in authority.runspec.payload.physical_inputs}
    proof_root = evidence_root / "phase-inputs/proofs"
    proof_root.mkdir(parents=True, exist_ok=True)
    items = []
    for entry in authority.runspec.payload.logical_inputs.entries:
        physical_name = entry.name
        locator = physical[physical_name]
        relative = f"phase-inputs/proofs/{physical_name}.json"
        proof = {"schema_version": 1, "accessible": True, "locator": locator.locator}
        _write_json(evidence_root / relative, proof)
        document = (evidence_root / relative).read_bytes()
        items.append(
            PostprocessingRuntimeInputAttestation(
                logical_input_name=entry.name,
                verification_kind="authority-declared-content-v1",
                authority=entry.member_identity,
                content_sha256=entry.expected_content_sha256,
                size_bytes=entry.expected_size_bytes,
                verification_source="runtime-preflight",
                observed_at="2026-09-03T12:00:00Z",
                physical_input_name=physical_name,
                physical_locator=locator.locator,
                member_identity=entry.member_identity,
                accessibility_evidence_path=relative,
                accessibility_evidence_sha256=hashlib.sha256(document).hexdigest(),
                accessibility_evidence_size_bytes=len(document),
            )
        )
    attestation_set = PostprocessingRuntimeInputAttestationSet(
        phase_run_id=authority.phase_run_id,
        attempt_id=authority.attempt_id,
        phase_runspec_digest=authority.runspec.digest,
        runtime_qualification=PostprocessingRuntimeQualificationAttestation(
            qualified_runtime_digest=authority.runspec.payload.qualified_runtime.digest,
            record_location=authority.runspec.payload.qualified_runtime.qualification_location,
            record_sha256=authority.runspec.payload.qualified_runtime.qualification_sha256,
            record_size_bytes=authority.runspec.payload.qualified_runtime.qualification_size_bytes,
            tuple_id=authority.runspec.payload.qualified_runtime.tuple_id,
            source_identity_digest=authority.runspec.payload.qualified_runtime.source_identity_digest,
            source_package_identity_digest=(authority.runspec.payload.qualified_runtime.source_package_identity_digest),
            toolkit_identity_digest=authority.runspec.payload.qualified_runtime.toolkit_identity_digest,
            runtime_component_identity_digest=(
                authority.runspec.payload.qualified_runtime.runtime_component_identity_digest
            ),
            observed_at="2026-09-03T12:00:00Z",
        ),
        attestations=tuple(sorted(items, key=lambda item: item.logical_input_name)),
    )
    _write_json(evidence_root / "phase-inputs/runtime-input-attestations.json", attestation_set.to_mapping())


def _materialized_authority(tmp_path: Path) -> Path:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )
    return authority_root


def _retry_resolution(
    *,
    tmp_path: Path,
    authority: PostprocessingAuthority,
) -> object:
    original_profile = resolve_cluster_profile("example-cluster", config_path=tmp_path / "profiles.yaml")
    qualification = json.loads(authority.runtime_qualification_bytes)
    qualification_tuple = qualification["tuple"]
    smoke = qualification["smoke_evidence"]
    smoke_job = qualification["smoke_job"]

    image_identity = dict(qualification_tuple["image_identity"])
    new_image_path = str(Path(image_identity["path"]).with_name("bspp-orchestration-retry.sqsh"))
    image_identity.update(path=new_image_path, size_bytes=2048, sha256="8" * 64)

    source_identity = dict(qualification_tuple["source_package_identity"])
    source_commit = "d" * 40
    source_path = str(Path(source_identity["package_path"]).with_name(f"bspp-orchestration-{source_commit}.tar"))
    source_identity.update(
        package_path=source_path,
        package_size_bytes=3072,
        package_sha256="9" * 64,
        manifest_sha256="a" * 64,
        commit=source_commit,
        tree="e" * 40,
    )

    toolkit_identity = dict(qualification_tuple["toolkit_package_identity"])
    toolkit_commit = "f" * 40
    toolkit_path = str(Path(toolkit_identity["package_path"]).with_name(f"afdb-toolkit-{toolkit_commit}.tar"))
    toolkit_identity.update(
        package_path=toolkit_path,
        package_size_bytes=5120,
        package_sha256="b" * 64,
        manifest_sha256="c" * 64,
        commit=toolkit_commit,
        tree="1" * 40,
    )
    selected_source = {
        **qualification_tuple["selected_source"],
        "revision": toolkit_commit,
        "toolkit_package_identity": toolkit_identity,
    }
    runtime_ipsae = runtime_ipsae_evidence(source_revision=toolkit_commit)
    runtime_ipsae["binary"] = {
        "path": "runtime-ipsae/ipsae_cpp",
        "sha256": "2" * 64,
        "size_bytes": 8192,
    }
    runtime_ipsae["version"] = {
        "scheme": "source-revision+binary-sha256-v1",
        "source_revision": toolkit_commit,
        "binary_sha256": "2" * 64,
        "output": f"ipsae_cpp source={toolkit_commit} sha256={'2' * 64}",
    }

    qualification_tuple.update(
        execution_runtime_image=new_image_path,
        runtime_image_size_bytes=image_identity["size_bytes"],
        runtime_image_sha256=image_identity["sha256"],
        image_identity=image_identity,
        source_bundle_id=f"bspp-orchestration-{source_commit}",
        source_bundle_path=source_path,
        source_package_identity=source_identity,
        toolkit_package_identity=toolkit_identity,
        toolkit_package_path=toolkit_path,
        selected_source=selected_source,
    )
    tuple_id = canonical_mapping_digest(qualification_tuple)
    old_tuple_id = qualification["tuple_id"]
    old_attempt_token = smoke_job["attempt_token"]
    new_attempt_token = "e" * 32
    qualification["tuple_id"] = tuple_id
    qualification["submitted_at"] = "2026-09-03T10:00:00Z"
    qualification["qualified_at"] = "2026-09-03T11:00:00Z"
    qualification["expires_at"] = "2026-09-11T00:00:00Z"
    smoke.update(
        tuple_id=tuple_id,
        job_id="8802",
        attempt_token=new_attempt_token,
        bootstrap_sha256="3" * 64,
        source_package_identity=source_identity,
        toolkit_package_identity=toolkit_identity,
        image_identity=image_identity,
        selected_source_identity=selected_source,
        runtime_ipsae=runtime_ipsae,
    )
    smoke_job.update(job_id="8802", attempt_token=new_attempt_token)
    for key, value in smoke_job.items():
        if isinstance(value, str):
            smoke_job[key] = value.replace(old_tuple_id, tuple_id).replace(old_attempt_token, new_attempt_token)
    qualification_document = (json.dumps(qualification, indent=2, sort_keys=True) + "\n").encode()

    profile_resources = dict(original_profile.resources)
    profile_resources["gpu_worker"] = profile_resources["gpu_worker"].model_copy(
        update={"cpus_per_task": 32, "memory": "192G", "time": "05:00:00"}
    )
    profile = replace(original_profile, image=new_image_path, resources=profile_resources)

    class FixtureRuntimeResolver:
        def resolve(self, **values: object) -> tuple[object, object, bytes]:
            attempt_id = values["attempt_id"]
            assert isinstance(attempt_id, str)
            profile_name, selection = replay_postprocessing_runtime(qualification_document, attempt_id=attempt_id)
            assert profile_name == profile.name
            return profile, selection, qualification_document

    return FixtureRuntimeResolver()


def _patch_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    submit_action: object,
) -> None:
    monkeypatch.setattr(RemoteSlurmTransport, "stage_immutable_artifact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        RemoteSlurmTransport,
        "command",
        lambda _transport, argv: CommandResult(argv=argv, returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(RemoteSlurmTransport, "query_submissions_by_correlation", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(RemoteSlurmTransport, "submit_action", submit_action)


def _submitted_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    first_job_id: int,
) -> tuple[Path, object, dict[str, str]]:
    authority_root = _materialized_authority(tmp_path)
    next_job_id = first_job_id

    def submit_action(_transport: RemoteSlurmTransport, action: SlurmAction) -> SlurmSubmission:
        nonlocal next_job_id
        job_id = str(next_job_id)
        next_job_id += 1
        result = CommandResult(argv=("sbatch",), returncode=0, stdout=f"{job_id}\n", stderr="")
        return SlurmSubmission(job_id=job_id, command=result.argv, result=result)

    _patch_transport(monkeypatch, submit_action=submit_action)
    submit_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    action_ids = tuple(action.action_id for action in authority.runspec.payload.actions)
    job_by_action = dict(zip(action_ids, (str(first_job_id + index) for index in range(len(action_ids))), strict=True))
    return authority_root, authority, job_by_action


def _observation(
    authority: object,
    *,
    job_by_action: dict[str, str],
    omit_task_for: str | None = None,
    parent_only_cancelled_for: str | None = None,
    completed_action_ids: set[str] | None = None,
    restarts: int | None = 0,
) -> SlurmObservation:
    actions = authority.runspec.payload.actions
    records: list[SlurmJobRecord] = []
    pending_records: list[SlurmJobRecord] = []
    selected: list[SlurmJobState] = []
    completed = {action.action_id for action in actions} if completed_action_ids is None else completed_action_ids
    for action in actions:
        parent_job_id = job_by_action[action.action_id]
        if action.action_id == parent_only_cancelled_for:
            selected.append(SlurmJobState(job_id=parent_job_id, state="CANCELLED", source="sacct", exit_code="0:0"))
            records.append(
                SlurmJobRecord(
                    job_id=parent_job_id,
                    source="sacct",
                    requested_job_id=parent_job_id,
                    state="CANCELLED",
                    exit_code="0:0",
                )
            )
            continue
        if action.action_id not in completed:
            selected.append(SlurmJobState(job_id=parent_job_id, state="PENDING", source="squeue"))
            pending_records.append(
                SlurmJobRecord(
                    job_id=parent_job_id,
                    source="squeue",
                    requested_job_id=parent_job_id,
                    state="PENDING",
                )
            )
            continue
        selected.append(SlurmJobState(job_id=parent_job_id, state="COMPLETED", source="sacct", exit_code="0:0"))
        if action.action_id == omit_task_for:
            records.append(
                SlurmJobRecord(
                    job_id=parent_job_id,
                    source="sacct",
                    requested_job_id=parent_job_id,
                    state="COMPLETED",
                    exit_code="0:0",
                    restarts=restarts,
                )
            )
            continue
        task_indexes: tuple[int | None, ...] = action.expected_task_indexes or (None,)
        records.extend(
            SlurmJobRecord(
                job_id=parent_job_id if task_index is None else f"{parent_job_id}_{task_index}",
                source="sacct",
                requested_job_id=parent_job_id,
                state="COMPLETED",
                exit_code="0:0",
                restarts=restarts,
            )
            for task_index in task_indexes
        )
    requested = tuple(job_by_action[action.action_id] for action in actions)
    return SlurmObservation(
        requested_job_ids=requested,
        squeue=SlurmCommandSnapshot(kind="squeue", argv=(), returncode=0, parser="fixture"),
        sacct=SlurmCommandSnapshot(kind="sacct", argv=(), returncode=0, parser="fixture"),
        squeue_jobs=tuple(pending_records),
        sacct_jobs=tuple(records),
        selected_states=tuple(selected),
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    example = Path(__file__).parents[1] / "tests" / "fixtures" / "postprocessing_phase" / "neutral_v3"
    run_plan_path = tmp_path / "run-plan.yaml"
    workflow_path = tmp_path / "workflow.yaml"
    profile_path = tmp_path / "profiles.yaml"
    run_plan_path.write_bytes((example / "run-plan.yaml").read_bytes())
    workflow_path.write_bytes((example / "workflow.yaml").read_bytes())
    profile_path.write_bytes((example / "profiles.yaml").read_bytes())
    profile = yaml.safe_load(profile_path.read_text())
    cluster = profile["clusters"]["example-cluster"]
    run_plan = yaml.safe_load(run_plan_path.read_text())
    authored_run_id = run_plan["dataset"]["run_id"]
    old_authored_root = Path(cluster["paths"]["output_root"]) / authored_run_id
    cluster["paths"]["output_root"] = str(tmp_path / "cluster-output")
    cluster["paths"]["staging_root"] = str(tmp_path / "cluster-staging")
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False))
    new_authored_root = Path(cluster["paths"]["output_root"]) / authored_run_id
    for section, field in (
        ("storage", "local_tar_dir"),
        ("storage", "local_tar_manifest_csv"),
        ("analysis_metadata", "csv_path"),
        ("analysis_metadata", "parquet_path"),
        ("analysis_metadata", "selected_ids_path"),
    ):
        value = Path(run_plan[section][field])
        run_plan[section][field] = str(new_authored_root / value.relative_to(old_authored_root))
    run_plan_path.write_text(yaml.safe_dump(run_plan, sort_keys=False))
    image = str(Path(cluster["paths"]["runtime_image_cache_root"]) / Path(cluster["paths"]["image"]).name)

    source_repo = tmp_path / "source-repo"
    source_repo.mkdir()
    subprocess.run(("git", "init", "-q", str(source_repo)), check=True)
    subprocess.run(("git", "-C", str(source_repo), "config", "user.email", "fixture@example.invalid"), check=True)
    subprocess.run(("git", "-C", str(source_repo), "config", "user.name", "Fixture"), check=True)
    (source_repo / "README.md").write_text("governed fixture\n")
    subprocess.run(("git", "-C", str(source_repo), "add", "README.md"), check=True)
    subprocess.run(("git", "-C", str(source_repo), "commit", "-qm", "fixture"), check=True)
    source_commit = subprocess.check_output(("git", "-C", str(source_repo), "rev-parse", "HEAD"), text=True).strip()
    source_tree = subprocess.check_output(
        ("git", "-C", str(source_repo), "rev-parse", "HEAD^{tree}"), text=True
    ).strip()
    source_package_path = str(tmp_path / "cluster-packages" / f"bspp-orchestration-{source_commit}.tar")
    toolkit_commit = "a" * 40
    toolkit_package_path = str(tmp_path / "cluster-packages" / f"afdb-toolkit-{toolkit_commit}.tar")
    image_identity = {
        "format_version": 1,
        "policy": cluster.get("runtime_image_policy", "digest-checked"),
        "path": image,
        "size_bytes": 1024,
        "sha256": SHA,
    }
    source_identity = {
        "format_version": 1,
        "format": "bspp-tar-v1",
        "verifier": "safe-tar-v1",
        "package_path": source_package_path,
        "package_size_bytes": 2048,
        "package_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "commit": source_commit,
        "tree": source_tree,
        "package_role": "orchestration",
        "policy_version": 1,
    }
    toolkit_identity = {
        "format_version": 1,
        "format": "bspp-tar-v1",
        "verifier": "safe-tar-v1",
        "package_path": toolkit_package_path,
        "package_size_bytes": 4096,
        "package_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "commit": toolkit_commit,
        "tree": "b" * 40,
        "package_role": "toolkit",
        "policy_version": 1,
    }
    selected_source = {
        "source_kind": "override",
        "root": "/workspace/AFDB-Integration-Kit",
        "revision": toolkit_commit,
        "toolkit_package_identity": toolkit_identity,
    }
    qualification_tuple = {
        "cluster_profile": "example-cluster",
        "scheduling_class": "gpu_worker",
        "execution_runtime_image": image,
        "runtime_image_policy": image_identity["policy"],
        "runtime_image_size_bytes": image_identity["size_bytes"],
        "runtime_image_sha256": image_identity["sha256"],
        "image_identity": image_identity,
        "runtime_facts": {"gpu_worker_gres": cluster["resources"]["gpu_worker"]["gres"]},
        "source_bundle_id": f"bspp-orchestration-{source_commit}",
        "source_bundle_path": source_package_path,
        "source_package_identity": source_identity,
        "toolkit_source": cluster["paths"]["afdb_toolkit_repo"],
        "toolkit_package_identity": toolkit_identity,
        "toolkit_package_path": toolkit_package_path,
        "selected_source": selected_source,
    }
    tuple_id = hashlib.sha256(
        json.dumps(qualification_tuple, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    job_id = "8801"
    attempt_token = "6" * 32
    smoke_root = tmp_path / "qualification" / "example-cluster" / "attempts" / tuple_id / attempt_token

    qualification = {
        "schema_version": 1,
        "profile": "example-cluster",
        "status": "qualified",
        "evidence_status": "qualified",
        "submitted_at": "2026-09-02T00:00:00Z",
        "tuple_id": tuple_id,
        "qualified_at": "2026-09-02T00:00:00Z",
        # Wall-clock-relative: unclocked CLI paths must observe a current qualification.
        "expires_at": (datetime.now(UTC) + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tuple": qualification_tuple,
        "attempt_history": [],
        "smoke_job": {
            "job_id": job_id,
            "script_path": str(smoke_root / "smoke.sbatch"),
            "submit_command": ["sbatch", str(smoke_root / "smoke.sbatch")],
            "record_path": str(tmp_path / "qualification" / "example-cluster" / f"{tuple_id}.json"),
            "attempt_token": attempt_token,
            "input_path": str(smoke_root / "input.json"),
            "result_path": str(smoke_root / "result.json"),
            "remote_script_path": str(smoke_root / "smoke.sbatch"),
            "remote_input_path": str(smoke_root / "input.json"),
            "remote_result_path": str(smoke_root / "result.json"),
        },
        "smoke_evidence": {
            "tuple_id": tuple_id,
            "job_id": job_id,
            "status": "succeeded",
            "attempt_token": attempt_token,
            "python": "Python 3.12.11",
            "gpu": "NVIDIA H100",
            "bootstrap_sha256": "7" * 64,
            "source_package_identity": source_identity,
            "toolkit_package_identity": toolkit_identity,
            "image_identity": image_identity,
            "selected_source_identity": selected_source,
            "runtime_ipsae": runtime_ipsae_evidence(source_revision=toolkit_commit),
        },
    }
    qualification_path = tmp_path / "runtime-qualification.json"
    qualification_path.write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n")

    inventory = {
        "schema_version": 1,
        "inventory_kind": "postprocessing-logical-input-inventory-v1",
        "members": [
            {
                "schema_version": 1,
                "name": name,
                "member_identity": f"task853:{name}",
                "expected_content_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "expected_size_bytes": index + 1,
            }
            for index, name in enumerate(
                (
                    "baseline-output",
                    "reference-heterodimer-id-manifest",
                    "reference-manifest-csv",
                    "reference-master-parquet",
                    "reference-tracking-parquet",
                    "reference-uniprot-duckdb",
                    "s3-archive-prefix",
                )
            )
        ],
    }
    inventory_path = tmp_path / "logical-input-inventory.yaml"
    inventory_path.write_text(yaml.safe_dump(inventory, sort_keys=False))

    policy = {
        "schema_version": 1,
        "policy_kind": "postprocessing-sealable-v1",
        "policy_schema": "bspp-postprocessing-acceptance",
        "policy_version": "1",
        "baseline_id": "task853-fixed-fork",
        "baseline_version": "c8f824d",
        "residual_allowances": [
            {
                "schema_version": 1,
                "report": "acceptance/semantic_acceptance/semantic_acceptance_summary.json",
                "json_pointer": "/errors",
                "match_kind": "json-pointer-count",
                "expected_value": "known-task853-residual",
                "cardinality_kind": "exact",
                "required_count": 1,
                "permitted_min": 1,
                "permitted_max": 1,
            },
            {
                "schema_version": 1,
                "report": "acceptance/verify_evidence/acceptance_evidence_report.json",
                "json_pointer": "/issues",
                "match_kind": "json-pointer-count",
                "expected_value": {
                    "check": "semantic-acceptance",
                    "message": "ok is not true",
                    "report_path": "acceptance/semantic_acceptance/semantic_acceptance_summary.json",
                },
                "cardinality_kind": "exact",
                "required_count": 1,
                "permitted_min": 1,
                "permitted_max": 1,
            },
        ],
        "completion_exit_contracts": [
            {
                "schema_version": 1,
                "step_name": step,
                "allowed_raw_exit_codes": [0, 1],
                "report_paths": [report],
                "report_schema": schema,
                "report_schema_version": "1",
                "outcome_report_path": report,
                "outcome_json_pointer": "/ok",
                "raw_exit_report_outcomes": [
                    {"raw_exit_code": 0, "report_ok": True},
                    {"raw_exit_code": 1, "report_ok": False},
                ],
            }
            for step, report, schema in (
                (
                    "acceptance-tar-payload-parity",
                    "acceptance/tar_payload_parity/tar_payload_parity_report.json",
                    "tar-payload-parity-report",
                ),
                (
                    "acceptance-semantic",
                    "acceptance/semantic_acceptance/semantic_acceptance_summary.json",
                    "semantic-acceptance-summary",
                ),
                (
                    "acceptance-verify-evidence",
                    "acceptance/verify_evidence/acceptance_evidence_report.json",
                    "acceptance-evidence-report",
                ),
            )
        ],
        "baseline_report_bindings": [
            {
                "schema_version": 1,
                "report": "acceptance/tar_payload_parity/tar_payload_parity_report.json",
                "baseline_locator_json_pointer": "/baseline_dir",
            },
            {
                "schema_version": 1,
                "report": "acceptance/semantic_acceptance/semantic_acceptance_summary.json",
                "baseline_locator_json_pointer": "/baseline_dir",
            },
        ],
        "cross_report_reconciliations": [
            {
                "schema_version": 1,
                "left_report": "acceptance/tar_payload_parity/tar_payload_parity_report.json",
                "left_json_pointer": "/candidate_dir",
                "right_report": "acceptance/semantic_acceptance/semantic_acceptance_summary.json",
                "right_json_pointer": "/candidate_dir",
                "comparison": "equal",
            }
        ],
    }
    policy_path = tmp_path / "acceptance-policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy, sort_keys=False))

    phase_plan = {
        "schema_version": 1,
        "phase_kind": "postprocessing",
        "target_cluster": "example-cluster",
        "output_namespace": "task853-phase",
        "legacy_run_plan": _document("legacy-run-plan", run_plan_path),
        "acceptance_policy": _document("acceptance-policy", policy_path),
        "logical_input_inventory": _document("logical-input-inventory", inventory_path),
        "runtime_qualification": _document("runtime-qualification", qualification_path),
    }
    phase_plan_path = tmp_path / "postprocessing-phase-plan.yaml"
    phase_plan_path.write_text(yaml.safe_dump(phase_plan, sort_keys=False))
    return phase_plan_path, profile_path, source_repo


def _document(kind: str, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "schema_version": 1,
        "document_kind": kind,
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _rewrite_as_consistent_split_v3(authority_root: Path) -> None:
    """Reproduce the pre-fix V3 split while keeping every stored digest consistent."""
    authority_path = authority_root / RUN_ID
    attempt_path = authority_path / "attempts" / "attempt-0001"
    legacy_path = attempt_path / "legacy-runspec.yaml"
    legacy = yaml.safe_load(legacy_path.read_text())
    legacy["storage"]["local_tar_dir"] = "/pre-fix/authored-output/local_tars"
    phase_runspec_path = attempt_path / "phase-runspec.json"
    phase_runspec = json.loads(phase_runspec_path.read_bytes())
    phase_runspec.pop("cluster_output_root")
    phase_runspec.pop("object_output_base_prefix")
    _write_consistently_rehashed_v3(authority_path, legacy=legacy, phase_runspec=phase_runspec)


def _rewrite_without_credential_mount_snapshot(authority_root: Path) -> None:
    authority_path = authority_root / RUN_ID
    attempt_path = authority_path / "attempts" / "attempt-0001"
    legacy = yaml.safe_load((attempt_path / "legacy-runspec.yaml").read_text())
    phase_runspec = json.loads((attempt_path / "phase-runspec.json").read_bytes())
    del phase_runspec["credential_mounts"]
    _write_consistently_rehashed_v3(authority_path, legacy=legacy, phase_runspec=phase_runspec)


def _append_historical_renderer_three_submission_intent(authority_root: Path) -> PostprocessingAuthority:
    _rewrite_without_credential_mount_snapshot(authority_root)
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    render_input = _render_input(authority)
    credential_mounts = PostprocessingCredentialMountSnapshot(
        aws_shared_credentials_file="/home/example/.aws/credentials",
        aws_config_file="/home/example/.aws/config",
    )
    scripts = {
        action.action_id: _render_postprocessing_action_script_with_credential_mounts(
            render_input,
            action,
            credential_mounts,
        )
        for action in authority.runspec.payload.actions
    }
    submission_id = _postprocessing_submission_id(
        authority,
        scripts,
        renderer_contract_version=3,
    )
    return append_event(
        authority,
        event_type="phase-submission-intended",
        occurred_at="2026-09-03T12:00:00Z",
        payload=PostprocessingSubmissionIntendedPayload(
            submission_id=submission_id,
            phase_runspec_digest=authority.runspec.digest,
            actions=tuple(
                PostprocessingSubmissionActionPlan(
                    action_id=action.action_id,
                    runtime_action_digest=mapping_digest(action.to_mapping()),
                    dependencies=action.dependencies,
                    cluster_script_path=str(postprocessing_cluster_action_script(authority.runspec, action)),
                    script_sha256=hashlib.sha256(scripts[action.action_id].encode()).hexdigest(),
                    scheduler_correlation_token=postprocessing_scheduler_correlation_token(
                        authority.runspec,
                        action,
                    ),
                    renderer_contract_version=3,
                )
                for action in authority.runspec.payload.actions
            ),
        ),
    )


def _rewrite_as_consistent_envelope_path_mismatch(authority_root: Path, *, mismatch: str) -> None:
    authority_path = authority_root / RUN_ID
    attempt_path = authority_path / "attempts" / "attempt-0001"
    legacy = yaml.safe_load((attempt_path / "legacy-runspec.yaml").read_text())
    phase_runspec = json.loads((attempt_path / "phase-runspec.json").read_bytes())
    payload = phase_runspec["payload"]
    projection = payload["execution_projection"]
    old_paths = postprocessing_attempt_paths_from_mapping(payload["attempt_paths"])
    if mismatch == "attempt-id":
        old_suffix = f"{RUN_ID}-attempt-0001"
        new_suffix = f"{RUN_ID}-attempt-0002"

        def successor(value: str) -> str:
            assert old_suffix in value
            return value.replace(old_suffix, new_suffix)

        new_paths = PostprocessingAttemptPaths(
            legacy_run_id=successor(old_paths.legacy_run_id),
            output_dir=successor(old_paths.output_dir),
            evidence_dir=successor(old_paths.evidence_dir),
            staging_dir=successor(old_paths.staging_dir),
            object_prefix=successor(old_paths.object_prefix),
        )
        legacy = retarget_v3_attempt_runspec_mapping(
            legacy,
            old_paths=old_paths,
            new_paths=new_paths,
        )
    elif mismatch == "output-namespace":
        tampered_namespace = "rehashed-output"
        tampered_run_id = old_paths.legacy_run_id.replace("task853-", f"{tampered_namespace}-", 1)

        def renamed(value: str) -> str:
            assert old_paths.legacy_run_id in value
            return value.replace(old_paths.legacy_run_id, tampered_run_id)

        new_paths = PostprocessingAttemptPaths(
            legacy_run_id=tampered_run_id,
            output_dir=renamed(old_paths.output_dir),
            evidence_dir=renamed(old_paths.evidence_dir),
            staging_dir=renamed(old_paths.staging_dir),
            object_prefix=old_paths.object_prefix,
        )
        legacy = retarget_v3_attempt_runspec_mapping(
            legacy,
            old_paths=old_paths,
            new_paths=new_paths,
        )
        projection["phase_identity"]["output_namespace"] = tampered_namespace
    elif mismatch == "staging-root":
        new_paths = replace(
            old_paths,
            staging_dir=f"/outside-cluster-staging/{old_paths.legacy_run_id}/staging",
        )
        legacy["paths"]["staging_dir"] = new_paths.staging_dir
    else:
        raise AssertionError(f"unknown envelope/path mismatch fixture: {mismatch}")
    payload["attempt_paths"] = new_paths.to_mapping()
    projection["phase_identity"]["substitutions"] = new_paths.to_mapping()
    _write_consistently_rehashed_v3(authority_path, legacy=legacy, phase_runspec=phase_runspec)


def _write_consistently_rehashed_v3(
    authority_path: Path,
    *,
    legacy: dict[str, object],
    phase_runspec: dict[str, object],
) -> None:
    attempt_path = authority_path / "attempts" / "attempt-0001"
    legacy_path = attempt_path / "legacy-runspec.yaml"
    legacy_bytes = yaml.safe_dump(legacy, sort_keys=False).encode()
    legacy_path.write_bytes(legacy_bytes)
    phase_runspec_path = attempt_path / "phase-runspec.json"
    projection = phase_runspec["payload"]["execution_projection"]
    projection["document_sha256"] = hashlib.sha256(legacy_bytes).hexdigest()
    projection["document_size_bytes"] = len(legacy_bytes)
    identity = postprocessing_phase_execution_identity_from_mapping(projection["phase_identity"])
    projection["phase_identity_digest"] = identity.digest
    parsed = postprocessing_phase_runspec_from_mapping(phase_runspec)
    phase_runspec_path.write_bytes(canonical_json_bytes(phase_runspec))

    phase_run_path = authority_path / "phase-run.json"
    phase_run = json.loads(phase_run_path.read_bytes())
    phase_run["attempts"][0]["phase_runspec_digest"] = parsed.digest
    phase_run_path.write_bytes(canonical_json_bytes(phase_run))

    event_path = authority_path / "events" / "000001-phase-materialized.json"
    event = json.loads(event_path.read_bytes())
    event["payload"]["phase_runspec_digest"] = parsed.digest
    event_path.write_bytes(canonical_json_bytes(event))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
