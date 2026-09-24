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

"""Local-fixture tests for the folding branches of the generic lifecycle services.

No cluster, no subprocess Slurm, and no network. Submission/status/resume/cancel
use a command runner that executes the local staging commands for real and only
synthesizes the ``sbatch`` assignment. Finalization uses the control-side
folding evidence validator to build the canonical-pair index and the attempt-bound
folding receipt through the generalized ``PhaseAuthorityStore``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.phase import (
    FoldingPhasePlan,
    FoldingPhasePlanPayload,
    FoldingPhaseRunSpec,
    phase_runspec_family_from_mapping,
)
from bspp.orchestration.contract.phase_receipt import ProvidedSuccessfulSchedulerEvidence
from bspp.orchestration.contract.phase_reconciliation import (
    PhaseActionTerminalObservedEvent,
    PhaseActionTerminalObservedPayload,
)
from bspp.orchestration.contract.phase_state import PhaseMaterializedEvent, PhaseRun
from bspp.orchestration.contract.preprocessing_handoff import (
    BundledMemberVerification,
    MsaArtifactSetManifest,
    MsaChunkManifestReference,
    VerifiedLocalBundledArtifactLocation,
    msa_artifact_set_id,
    verified_local_bundled_artifact_location_id,
)
from bspp.orchestration.control.folding_phase_adapter import validate_folding_plan_runspec_binding
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.control.phase_cancellation import cancel_phase
from bspp.orchestration.control.phase_finalization import finalize_phase
from bspp.orchestration.control.phase_materialization import materialize_phase
from bspp.orchestration.control.phase_resume import resume_phase
from bspp.orchestration.control.phase_retry import retry_phase
from bspp.orchestration.control.phase_status import status_phase
from bspp.orchestration.control.phase_submission import submit_phase
from bspp.orchestration.control.transport import CommandResult, default_command_runner

FIXED_TIME = datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC)
FIXED_RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
ARTIFACT_SET_ID = "sha256:" + "a" * 64
MEMBER_NAME = "AFDB_AF-0000000000000001.a3m"
MEMBER_PATH = f"a3ms/{MEMBER_NAME}"
RUNTIME_IMAGE = "registry/bspp-runtime:latest"

ACTION_IDS = (
    "msa-flatten-000001",
    "split-000001",
    "preprocess-000001",
    "fold-000001",
    "canonical-pair-000001",
)


def _bundled_location(tmp_path: Path, artifact_set_id: str = ARTIFACT_SET_ID) -> VerifiedLocalBundledArtifactLocation:
    bundle_path = tmp_path / "msa-set" / "msa-set.tar.lz4"
    tar_path = tmp_path / "msa-set" / "msa-set.tar"
    bundle_path.parent.mkdir(parents=True)
    bundle_bytes = b"bspp-fixture-lz4-bundle\n"
    tar_bytes = b"bspp-fixture-tar\n"
    bundle_path.write_bytes(bundle_bytes)
    tar_path.write_bytes(tar_bytes)
    lz4_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    tar_sha256 = hashlib.sha256(tar_bytes).hexdigest()
    members = (
        BundledMemberVerification(
            logical_path=MEMBER_PATH,
            member_name=MEMBER_NAME,
            raw_member_name=MEMBER_NAME,
            size_bytes=1,
            sha256="b" * 64,
        ),
    )
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=Path(bundle_path).as_uri(),
        tar_size_bytes=len(tar_bytes),
        tar_sha256=tar_sha256,
        lz4_size_bytes=len(bundle_bytes),
        lz4_sha256=lz4_sha256,
        raw_tar_members=(MEMBER_NAME,),
        members=members,
    )
    return VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=Path(bundle_path).as_uri(),
        tar_size_bytes=len(tar_bytes),
        tar_sha256=tar_sha256,
        lz4_size_bytes=len(bundle_bytes),
        lz4_sha256=lz4_sha256,
        raw_tar_members=(MEMBER_NAME,),
        members=members,
        verified_at="2026-01-01T00:00:00Z",
    )


def _manifest() -> MsaArtifactSetManifest:
    chunk = MsaChunkManifestReference(
        chunk_name="foo_tranche00_00001.fa",
        logical_path="chunks/foo_tranche00_00001.json",
        sha256="f" * 64,
        member_count=1,
        logical_bytes=1,
    )
    return MsaArtifactSetManifest(
        artifact_set_id=msa_artifact_set_id((chunk,), 1, 1, member_lengths=(1,)),
        chunks=(chunk,),
        member_count=1,
        logical_bytes=1,
        member_lengths=(1,),
    )


def _phase_plan(tmp_path: Path, *, with_manifest: bool = True) -> FoldingPhasePlan:
    manifest = _manifest() if with_manifest else None
    artifact_set_id = manifest.artifact_set_id if manifest is not None else ARTIFACT_SET_ID
    msa_set = MsaSetConsumption(
        artifact_set_id=artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=(MEMBER_PATH,),
        requires_paired_query_header=True,
    )
    return FoldingPhasePlan(
        target_cluster="example-cluster",
        input_location=_bundled_location(tmp_path, artifact_set_id),
        payload=FoldingPhasePlanPayload(msa_set=msa_set, backend="openfold-cli", msa_set_manifest=manifest),
    )


def _write_profile(tmp_path: Path) -> Path:
    profile_path = tmp_path / "profiles.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "clusters": {
                    "example-cluster": {
                        "owner": "tester",
                        "transport": "local-slurm",
                        "project_root": str(tmp_path / "project"),
                        "output_root": str(tmp_path / "output"),
                        "staging_root": str(tmp_path / "staging"),
                        "orchestration_repo": str(tmp_path / "orchestration"),
                        "image": RUNTIME_IMAGE,
                        "folding_backend_images": [
                            {"backend": "openfold-cli", "image": "registry/bspp-folding-openfold-cli:latest"},
                        ],
                        "folding_backend_assets": [
                            {
                                "backend": "openfold-cli",
                                "chain_manifest_csv": "/assets/chains.csv",
                                "openfold_model_dir": "/assets/models",
                            }
                        ],
                        "extra_mounts": [
                            {"source": "/assets/chains.csv", "target": "/assets/chains.csv", "read_only": True},
                            {"source": "/assets/models", "target": "/assets/models", "read_only": True},
                        ],
                    }
                }
            },
            sort_keys=True,
        )
    )
    return profile_path


def _materialize_folding(tmp_path: Path) -> tuple[Path, str]:
    profile_path = _write_profile(tmp_path)
    plan = _phase_plan(tmp_path)
    plan_path = tmp_path / "folding-phase-plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan.to_mapping(), sort_keys=True))
    authority_root = tmp_path / "authority"
    result = materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    return authority_root, result.phase_run_id


def _publish_historical_folding_authority(tmp_path: Path) -> tuple[Path, str]:
    """Publish a historically loadable manifest-less folding authority.

    A complete authority is first materialized through the executable guards,
    then rewritten to drop the root manifest and backend-assets snapshot so the
    records remain loadable and binding-valid but are no longer executable.
    """
    complete_root, phase_run_id = _materialize_folding(tmp_path)
    store = PhaseAuthorityStore(complete_root)
    authority = store.validate(phase_run_id)
    plan = authority.phase_plan
    assert isinstance(plan, FoldingPhasePlan)
    runspec = authority.phase_runspec
    assert isinstance(runspec, FoldingPhaseRunSpec)

    hist_plan = replace(plan, payload=replace(plan.payload, msa_set_manifest=None))
    hist_runspec = replace(
        runspec,
        phase_plan_digest=hist_plan.digest,
        payload=replace(runspec.payload, msa_set_manifest=None, fold_shard_projection=None),
        cluster=replace(runspec.cluster, backend_assets=None),
    )
    attempt = replace(authority.phase_run.attempts[0], phase_runspec_digest=hist_runspec.digest)
    phase_run = replace(
        authority.phase_run,
        phase_plan_digest=hist_plan.digest,
        attempts=(attempt,),
    )
    event = replace(
        authority.materialized_event,
        payload=replace(
            authority.materialized_event.payload,
            phase_run=phase_run,
            phase_runspec=hist_runspec,
        ),
    )
    historical_root = tmp_path / "historical-authority"
    PhaseAuthorityStore(historical_root).publish(
        phase_plan=hist_plan,
        phase_run=phase_run,
        phase_runspec=hist_runspec,
        materialized_event=event,
    )
    return historical_root, phase_run_id


class FoldingSubmissionRunner:
    """Run local staging commands and synthesize only the Slurm assignments."""

    def __init__(self) -> None:
        self._next_job_id = 1001
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "sbatch":
            job_id = str(self._next_job_id)
            self._next_job_id += 1
            return CommandResult(argv=argv, returncode=0, stdout=f"{job_id}\n", stderr="")
        return default_command_runner(argv)


class FoldingCancellationRunner:
    """Synthesize RUNNING squeue, acknowledged scancel, and terminal sacct rows."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._observation_index = 0

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "scancel":
            return CommandResult(argv, 0, "", "")
        if argv[0] == "squeue":
            job_ids = argv[argv.index("-j") + 1].split(",")
            jobs = [{"job_id": int(job_id), "job_state": "RUNNING"} for job_id in job_ids]
            return CommandResult(argv, 0, json.dumps({"jobs": jobs}), "")
        if argv[0] == "sacct":
            job_ids = argv[argv.index("-j") + 1].split(",")
            state, exit_code = ("RUNNING", None) if self._observation_index == 0 else ("CANCELLED", "0:15")
            self._observation_index += 1
            jobs = [{"job_id_raw": job_id, "state": state, "exit_code": exit_code} for job_id in job_ids]
            return CommandResult(argv, 0, json.dumps({"jobs": jobs}), "")
        raise AssertionError(f"unexpected command: {argv}")


def _record_folding_terminal_failure(authority_root: Path, phase_run_id: str) -> None:
    store = PhaseAuthorityStore(authority_root)
    authority = store.validate(phase_run_id)
    assert authority.submission is not None
    action = next(item for item in authority.submission.actions if item.action_id == "msa-flatten-000001")
    assert action.job_id is not None
    payload = PhaseActionTerminalObservedPayload(
        submission_id=authority.submission.submission_id,
        phase_runspec_digest=authority.phase_runspec.digest,
        action_id=action.action_id,
        runtime_action_digest=action.plan.runtime_action_digest,
        scheduler_correlation_token=action.scheduler_correlation_token,
        job_id=action.job_id,
        state="FAILED",
        exit_code="1:0",
        outcome="failed",
    )
    store.append_event(
        phase_run_id,
        lambda sequence: PhaseActionTerminalObservedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at="2026-09-11T12:03:00.000000Z",
            payload=payload,
        ),
    )


def _prediction_pair_mapping(model_entity_id: str = "AF-0000000000000001") -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_entity_id": model_entity_id,
        "tool_used": "OpenFold / AlphaFold-Multimer",
        "structure_path": f"{model_entity_id}-model_v1.pdb",
        "scores_path": f"{model_entity_id}-meta_v1.json",
        "scores": {
            "schema_version": 1,
            "plddt": [0.5, 0.6],
            "pae": [[0.0, 0.1], [0.1, 0.0]],
            "max_pae": 0.1,
            "ptm": 0.9,
            "iptm": 0.8,
        },
    }


def _full_evidence() -> dict[str, dict[str, object]]:
    pair = _prediction_pair_mapping()
    return {
        "msa-flatten-000001": {"a3m_paths": [MEMBER_PATH]},
        "split-000001": {"chain_files": ["chain_1.a3m", "chain_2.a3m"]},
        "preprocess-000001": {"fasta_dir": "fasta", "alignment_dir": "alignments", "layout": "openfold"},
        "fold-000001": {"pairs": [pair]},
        "canonical-pair-000001": {
            "entries": [
                {"target_id": "target-1", "sequence_sha256": "f" * 64, "pair": pair},
            ]
        },
    }


def test_materialize_folding_phase_resolves_family_and_replays_authority(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)

    validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(validation.phase_plan, FoldingPhasePlan)
    assert validation.phase_run.phase_kind == "folding"
    assert [action.action_id for action in validation.phase_runspec.payload.actions] == list(ACTION_IDS)
    run_root = authority_root / phase_run_id
    assert (run_root / "phase-plan.json").is_file()
    assert (run_root / "phase-run.json").is_file()
    assert (run_root / "attempts" / "attempt-0001" / "phase-runspec.json").is_file()
    # Folding authority never writes the preprocessing database source-manifest projection.
    assert not (run_root / "attempts" / "attempt-0001" / "database-source-manifest.json").exists()


def test_materialize_rejects_missing_manifest_without_publishing(tmp_path: Path) -> None:
    """Public materialization is side-effect free when the root manifest is missing."""
    profile_path = _write_profile(tmp_path)
    plan = _phase_plan(tmp_path, with_manifest=False)
    plan_path = tmp_path / "folding-phase-plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan.to_mapping(), sort_keys=True))
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match="root MSA set manifest"):
        materialize_phase(
            plan_path,
            authority_root=authority_root,
            config_path=profile_path,
            clock=lambda: FIXED_TIME,
            phase_run_id_factory=lambda: FIXED_RUN_ID,
        )

    assert not authority_root.exists()


def test_submit_rejects_historical_manifestless_authority_before_scheduler(tmp_path: Path) -> None:
    """Historical manifest-less authority is loadable but never reaches sbatch."""
    authority_root, phase_run_id = _publish_historical_folding_authority(tmp_path)
    runner = FoldingSubmissionRunner()

    with pytest.raises(ValueError, match="root MSA set manifest"):
        submit_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: FIXED_TIME,
            runner=runner,
        )

    assert runner.calls == []


def test_submit_status_resume_folding_lifecycle(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    runner = FoldingSubmissionRunner()

    submitted = submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: FIXED_TIME,
        runner=runner,
    )

    assert submitted.status == "submitted"
    assert [(item.action_id, item.job_id) for item in submitted.actions] == [
        ("msa-flatten-000001", "1001"),
        ("split-000001", "1002"),
        ("preprocess-000001", "1003"),
        ("fold-000001", "1004"),
        ("canonical-pair-000001", "1005"),
    ]
    assert [call[0] for call in runner.calls].count("sbatch") == 5

    report = status_phase(phase_run_id, authority_root=authority_root, runner=FoldingSubmissionRunner())
    assert report.status == "submitted"
    assert [action.action_id for action in report.actions] == list(ACTION_IDS)
    assert all(action.durable_status == "submitted" for action in report.actions)

    resumed = resume_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    assert resumed.outcome == "no-op"
    assert [action.action_id for action in resumed.actions] == list(ACTION_IDS)


def test_resume_folding_scalar_actions_use_exact_accounting_fallback(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    submit_phase(
        phase_run_id, authority_root=authority_root, clock=lambda: FIXED_TIME, runner=FoldingSubmissionRunner()
    )
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        if argv[0] == "squeue":
            return CommandResult(argv, 0, json.dumps({"jobs": []}), "")
        assert argv[0] == "sacct"
        job_id = argv[argv.index("-j") + 1]
        if "--parsable2" in argv:
            return CommandResult(argv, 0, f"{job_id}|{job_id}|COMPLETED|0:0|0|\n", "")
        job = {
            "job_id": int(job_id),
            "state": {"current": ["COMPLETED"]},
            "exit_code": {
                "return_code": {"set": True, "infinite": False, "number": 0},
                "signal": {"id": {"set": False, "infinite": False, "number": 0}, "name": ""},
            },
        }
        return CommandResult(argv, 0, json.dumps({"jobs": [job]}), "")

    result = resume_phase(phase_run_id, authority_root=authority_root, clock=lambda: FIXED_TIME, runner=runner)

    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert result.terminal_action_ids == ACTION_IDS
    assert len(replay.terminal_observations) == len(ACTION_IDS)
    assert all(item.exit_code == "0:0" and item.outcome == "succeeded" for item in replay.terminal_observations)
    assert len([call for call in calls if "--parsable2" in call]) == len(ACTION_IDS)
    assert not any(call[0] == "sbatch" for call in calls)
    assert not replay.lifecycle.sealed and replay.receipt is None


def test_cancel_folding_drives_generic_append_only_transition(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    runner = FoldingCancellationRunner()

    result = cancel_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: FIXED_TIME,
        runner=runner,
    )

    assert result.status == "cancelled"
    assert [call for call in runner.calls if call[0] == "scancel"] == [
        ("scancel", "1001"),
        ("scancel", "1002"),
        ("scancel", "1003"),
        ("scancel", "1004"),
        ("scancel", "1005"),
    ]
    replay = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    terminal_events = [event for event in replay.events if isinstance(event, PhaseActionTerminalObservedEvent)]
    assert len(terminal_events) == 5
    assert {event.payload.action_id for event in terminal_events} == set(ACTION_IDS)
    assert replay.lifecycle.attempt_status == replay.lifecycle.run_status == "cancelled"


def test_retry_folding_ignores_carry_forward_and_materializes_successor(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    carry_request = tmp_path / "carry-request.json"
    carry_request.write_text("{}\n")

    # Folding carry-forward is automatic: the
    # caller-authored --carry-forward subset is not applicable and is ignored
    # rather than rejected. A non-retryable Attempt still fails on lifecycle,
    # never on the flag or on a preprocessing carry-request parse.
    with pytest.raises(ValueError, match="cannot cross"):
        retry_phase(
            phase_run_id,
            authority_root=authority_root,
            config_path=_write_profile(tmp_path),
            carry_forward_path=carry_request,
        )

    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    _record_folding_terminal_failure(authority_root, phase_run_id)

    result = retry_phase(
        phase_run_id,
        authority_root=authority_root,
        config_path=_write_profile(tmp_path),
        clock=lambda: FIXED_TIME,
    )

    assert result.successor_attempt_id == "attempt-0002"
    validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(validation.phase_runspec, FoldingPhaseRunSpec)


@pytest.mark.parametrize("evidence_format", ["json", "yaml"])
def test_finalize_folding_phase_produces_folding_receipt(tmp_path: Path, evidence_format: str) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    canonical_action = next(
        item for item in authority.phase_runspec.payload.actions if item.action_kind == "canonical-pair"
    )
    submitted_action = next(
        item for item in authority.submission.actions if item.action_id == canonical_action.action_id
    )

    scheduler_path = tmp_path / "scheduler-evidence.json"
    scheduler_path.write_text(
        json.dumps(
            ProvidedSuccessfulSchedulerEvidence(
                phase_run_id=phase_run_id,
                attempt_id=authority.phase_runspec.attempt_id,
                phase_runspec_digest=authority.phase_runspec.digest,
                action_id=canonical_action.action_id,
                job_id=submitted_action.job_id,
                observed_at="2026-09-11T11:30:00Z",
            ).to_mapping(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    evidence_path = tmp_path / "folding-action-evidence.json"
    evidence = _full_evidence()
    evidence_text = (
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        if evidence_format == "json"
        else yaml.safe_dump(evidence, sort_keys=True)
    )
    evidence_path.write_text(evidence_text)

    result = finalize_phase(
        phase_run_id,
        authority_root=authority_root,
        scheduler_evidence_path=scheduler_path,
        action_evidence_path=evidence_path,
        clock=lambda: FIXED_TIME,
    )

    assert result.status == "accepted"
    assert result.artifact_set_id is None
    assert result.artifact_location_id is None
    sealed = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert sealed.receipt is not None
    assert sealed.receipt.action_id == "canonical-pair-000001"
    assert sealed.receipt.canonical_pair_index_digest is not None
    assert len(sealed.receipt.folding_action_evidence_digests) == 5


def test_folding_finalization_is_idempotent(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    submit_phase(
        phase_run_id,
        authority_root=authority_root,
        clock=lambda: FIXED_TIME,
        runner=FoldingSubmissionRunner(),
    )
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    canonical_action = next(
        item for item in authority.phase_runspec.payload.actions if item.action_kind == "canonical-pair"
    )
    submitted_action = next(
        item for item in authority.submission.actions if item.action_id == canonical_action.action_id
    )
    scheduler_path = tmp_path / "scheduler-evidence.json"
    scheduler_path.write_text(
        json.dumps(
            ProvidedSuccessfulSchedulerEvidence(
                phase_run_id=phase_run_id,
                attempt_id=authority.phase_runspec.attempt_id,
                phase_runspec_digest=authority.phase_runspec.digest,
                action_id=canonical_action.action_id,
                job_id=submitted_action.job_id,
                observed_at="2026-09-11T11:30:00Z",
            ).to_mapping(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    evidence_path = tmp_path / "folding-action-evidence.json"
    evidence_path.write_text(json.dumps(_full_evidence(), indent=2, sort_keys=True) + "\n")

    first = finalize_phase(
        phase_run_id,
        authority_root=authority_root,
        scheduler_evidence_path=scheduler_path,
        action_evidence_path=evidence_path,
        clock=lambda: FIXED_TIME,
    )
    second = finalize_phase(
        phase_run_id,
        authority_root=authority_root,
        scheduler_evidence_path=scheduler_path,
        action_evidence_path=evidence_path,
        clock=lambda: FIXED_TIME,
    )

    assert first == second


def test_unknown_phase_kind_still_raises(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    plan = _phase_plan(tmp_path)
    mapping = plan.to_mapping()
    mapping["phase_kind"] = "bogus"
    plan_path = tmp_path / "bogus-phase-plan.yaml"
    plan_path.write_text(yaml.safe_dump(mapping, sort_keys=True))

    with pytest.raises(ValueError, match="phase_kind"):
        materialize_phase(
            plan_path,
            authority_root=tmp_path / "authority",
            config_path=profile_path,
        )


def _replacement_publish_records(
    authority_root: Path,
    phase_run_id: str,
    tamper: Callable[[FoldingPhaseRunSpec], FoldingPhaseRunSpec],
) -> tuple[FoldingPhasePlan, PhaseRun, FoldingPhaseRunSpec, PhaseMaterializedEvent]:
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    plan = authority.phase_plan
    assert isinstance(plan, FoldingPhasePlan)
    runspec = authority.phase_runspec
    assert isinstance(runspec, FoldingPhaseRunSpec)
    tampered = tamper(runspec)
    attempt = replace(authority.phase_run.attempts[0], phase_runspec_digest=tampered.digest)
    phase_run = replace(authority.phase_run, attempts=(attempt,))
    event = replace(
        authority.materialized_event,
        payload=replace(
            authority.materialized_event.payload,
            phase_run=phase_run,
            phase_runspec=tampered,
        ),
    )
    return plan, phase_run, tampered, event


def _drift_backend(runspec: FoldingPhaseRunSpec) -> FoldingPhaseRunSpec:
    return replace(runspec, payload=replace(runspec.payload, backend="colabfold"))


def _drift_msa_set(runspec: FoldingPhaseRunSpec) -> FoldingPhaseRunSpec:
    drifted = replace(runspec.payload.msa_set, member_a3m_paths=("a3ms/AFDB_AF-0000000000000002.a3m",))
    return replace(runspec, payload=replace(runspec.payload, msa_set=drifted))


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (_drift_backend, "backend does not bind"),
        (_drift_msa_set, "msa_set does not bind"),
    ],
)
def test_publish_rejects_plan_runspec_scientific_binding_drift(
    tmp_path: Path,
    tamper: Callable[[FoldingPhaseRunSpec], FoldingPhaseRunSpec],
    message: str,
) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    plan, phase_run, runspec, event = _replacement_publish_records(authority_root, phase_run_id, tamper)
    replacement_root = tmp_path / "replacement-authority"

    with pytest.raises(ValueError, match=message):
        PhaseAuthorityStore(replacement_root).publish(
            phase_plan=plan,
            phase_run=phase_run,
            phase_runspec=runspec,
            materialized_event=event,
        )

    assert not replacement_root.exists()


def _rewrite_folding_authority(
    run_root: Path,
    *,
    mutate_runspec: Callable[[dict[str, object]], None],
    mutate_run: Callable[[dict[str, object]], None] | None = None,
) -> None:
    runspec_path = run_root / "attempts/attempt-0001/phase-runspec.json"
    runspec_mapping = json.loads(runspec_path.read_text())
    mutate_runspec(runspec_mapping)
    runspec = phase_runspec_family_from_mapping(runspec_mapping)
    assert isinstance(runspec, FoldingPhaseRunSpec)

    run_path = run_root / "phase-run.json"
    run_mapping = json.loads(run_path.read_text())
    if mutate_run is not None:
        mutate_run(run_mapping)
    run_mapping["attempts"][0]["phase_runspec_digest"] = runspec.digest

    event_path = run_root / "events/000001-phase-materialized.json"
    event_mapping = json.loads(event_path.read_text())
    event_mapping["payload"]["phase_run"] = run_mapping
    event_mapping["payload"]["phase_runspec"] = runspec.to_mapping()

    runspec_path.write_text(_canonical_json(runspec.to_mapping()))
    run_path.write_text(_canonical_json(run_mapping))
    event_path.write_text(_canonical_json(event_mapping))


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


@pytest.mark.parametrize(
    ("mutate_runspec", "message"),
    [
        (lambda mapping: mapping["payload"].update({"backend": "colabfold"}), "backend does not bind"),
        (
            lambda mapping: mapping["payload"]["msa_set"].update(
                {"member_a3m_paths": ["a3ms/AFDB_AF-0000000000000002.a3m"]}
            ),
            "msa_set does not bind",
        ),
        (lambda mapping: mapping["payload"]["actions"].pop(), "action graph is not the canonical"),
    ],
    ids=["backend-drift", "msa-set-drift", "canonical-pair-removed"],
)
def test_replay_and_submit_reject_internally_consistent_tampered_authority(
    tmp_path: Path,
    mutate_runspec: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    run_root = authority_root / phase_run_id
    _rewrite_folding_authority(run_root, mutate_runspec=mutate_runspec)

    with pytest.raises(ValueError, match=message):
        PhaseAuthorityStore(authority_root).validate(phase_run_id)

    runner = FoldingSubmissionRunner()
    with pytest.raises(ValueError, match=message):
        submit_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=lambda: FIXED_TIME,
            runner=runner,
        )

    assert runner.calls == []


def test_replay_rejects_folding_phase_run_kind_mismatch(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    run_root = authority_root / phase_run_id
    _rewrite_folding_authority(
        run_root,
        mutate_runspec=lambda mapping: None,
        mutate_run=lambda mapping: mapping.update({"phase_kind": "preprocessing"}),
    )

    with pytest.raises(ValueError, match="phase_kind does not match"):
        PhaseAuthorityStore(authority_root).validate(phase_run_id)


def test_untampered_folding_authority_satisfies_plan_runspec_binding(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialize_folding(tmp_path)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(authority.phase_plan, FoldingPhasePlan)
    assert isinstance(authority.phase_runspec, FoldingPhaseRunSpec)

    validate_folding_plan_runspec_binding(authority.phase_plan, authority.phase_runspec)
    assert [action.action_id for action in authority.phase_runspec.payload.actions] == list(ACTION_IDS)
