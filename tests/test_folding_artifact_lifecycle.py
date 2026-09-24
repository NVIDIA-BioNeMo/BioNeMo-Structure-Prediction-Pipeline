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

"""Public local-authority lifecycle with actual tiny files and scheduler test doubles."""

import json
import shutil
from dataclasses import replace
from datetime import timedelta

import yaml

from bspp.orchestration.contract.phase import FoldingPhasePlan, FoldingPhasePlanPayload
from bspp.orchestration.contract.phase_receipt import ProvidedSuccessfulSchedulerEvidence
from bspp.orchestration.contract.phase_reconciliation import (
    PhaseActionTerminalObservedEvent,
    PhaseActionTerminalObservedPayload,
)
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.control.phase_finalization import finalize_phase
from bspp.orchestration.control.phase_materialization import materialize_phase
from bspp.orchestration.control.phase_submission import submit_phase
from bspp.orchestration.runtime.folding.executor import run_execute_action
from tests.support.folding_artifact_fixture import action_root, fixture
from tests.test_phase_folding_lifecycle import FIXED_TIME, FoldingSubmissionRunner, _write_profile


def test_public_materialize_execute_fetch_finalize_and_exact_receipt_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "baked")
    monkeypatch.setenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", "f" * 40)
    initial, deps = fixture(tmp_path)
    profile_path = _write_profile(tmp_path)
    profiles = yaml.safe_load(profile_path.read_text())
    profile = profiles["clusters"]["example-cluster"]
    assets = initial.cluster.backend_assets
    profile["folding_backend_images"] = [{"backend": "bioir", "image": "registry/bioir:fixed"}]
    profile["folding_backend_assets"] = [assets.to_mapping()]
    profile["folding_backend_assets"][0].pop("schema_version", None)
    profile["extra_mounts"] = [
        {"source": path, "target": path, "read_only": True}
        for path in (assets.bioir_checkpoint, assets.bioir_monomer_checkpoint)
    ]
    profile["resources"] = {
        "gpu_worker": {
            "partition": "gpu",
            "cpus_per_task": 1,
            "memory": "2G",
            "time": "00:05:00",
            "nodes": 2,
            "tasks_per_node": 1,
            "gpus_per_task": 1,
            "gres": None,
        }
    }
    profile_path.write_text(yaml.safe_dump(profiles))
    plan = FoldingPhasePlan(
        target_cluster="example-cluster",
        input_location=initial.input_location,
        payload=FoldingPhasePlanPayload(
            msa_set=initial.payload.msa_set,
            backend="bioir",
            msa_set_manifest=initial.payload.msa_set_manifest,
            bioir_model_policy=initial.payload.bioir_model_policy,
            evidence_profile="artifact-backed-v2",
        ),
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan.to_mapping()))
    authority_root = tmp_path / "authority"
    result = materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: initial.phase_run_id,
    )
    submit_phase(
        result.phase_run_id, authority_root=authority_root, clock=lambda: FIXED_TIME, runner=FoldingSubmissionRunner()
    )
    store = PhaseAuthorityStore(authority_root)
    authority = store.validate(result.phase_run_id)
    runspec = authority.phase_runspec
    deps = replace(deps, load_runspec=lambda _: runspec)
    runspec_path = authority_root / runspec.phase_run_id / authority.current_attempt.phase_runspec_location
    for action in runspec.payload.actions:
        root = action_root(runspec, action.action_id)
        root.mkdir(parents=True, exist_ok=True)
        for rank in range(action.resources.workers if action.action_kind == "fold" else 1):
            run_execute_action(
                phase_runspec_path=runspec_path,
                action_id=action.action_id,
                action_evidence_path=root / "action-evidence.json",
                handoff_path=root / "handoff.json",
                deps=deps,
                rank=rank,
            )
    for assigned in authority.submission.actions:
        payload = PhaseActionTerminalObservedPayload(
            submission_id=authority.submission.submission_id,
            phase_runspec_digest=runspec.digest,
            action_id=assigned.action_id,
            runtime_action_digest=assigned.plan.runtime_action_digest,
            scheduler_correlation_token=assigned.scheduler_correlation_token,
            job_id=assigned.job_id,
            state="COMPLETED",
            exit_code="0:0",
            outcome="succeeded",
        )
        store.append_event(
            runspec.phase_run_id,
            lambda seq, payload=payload: PhaseActionTerminalObservedEvent(
                sequence=seq,
                phase_run_id=runspec.phase_run_id,
                attempt_id=runspec.attempt_id,
                occurred_at="2026-09-11T12:03:00Z",
                payload=payload,
            ),
        )
    from bspp.orchestration.control.folding_artifact_evidence import fetch_folding_finalization_evidence

    class Source:
        def fetch_stable_artifact(self, path, *, remote_root, maximum_bytes):
            data = path.read_bytes()
            assert len(data) <= maximum_bytes
            return data

    bundle = tmp_path / "fetched"
    fetch_folding_finalization_evidence(
        runspec.phase_run_id, authority_root=authority_root, destination=bundle, source=Source()
    )
    assigned = authority.submission.actions[-1]
    scheduler = tmp_path / "scheduler.json"
    scheduler.write_text(
        json.dumps(
            ProvidedSuccessfulSchedulerEvidence(
                phase_run_id=runspec.phase_run_id,
                attempt_id=runspec.attempt_id,
                phase_runspec_digest=runspec.digest,
                action_id=assigned.action_id,
                job_id=assigned.job_id,
                observed_at="2026-09-11T12:03:00Z",
            ).to_mapping()
        )
    )
    # Control accepts authenticated metadata after prediction files are unavailable locally.
    shutil.rmtree(action_root(runspec, next(a.action_id for a in runspec.payload.actions if a.action_kind == "fold")))
    kwargs = dict(
        authority_root=authority_root,
        scheduler_evidence_path=scheduler,
        action_evidence_path=bundle / assigned.action_id / "action-evidence.json",
        handoff_path=bundle,
        clock=lambda: FIXED_TIME + timedelta(minutes=5),
    )
    first = finalize_phase(runspec.phase_run_id, **kwargs)
    second = finalize_phase(runspec.phase_run_id, **kwargs)
    assert first == second
    accepted = store.validate(runspec.phase_run_id)
    assert accepted.receipt.phase_receipt_id == first.phase_receipt_id
    assert len(accepted.receipt.folding_action_evidence_digests) == 5
