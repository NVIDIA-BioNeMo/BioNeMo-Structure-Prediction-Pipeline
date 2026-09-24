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

"""Contracts, durable authority, service, and CLI tests for Phase Materialization."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from bspp.orchestration.contract.database_placement import (
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabaseSetSelection,
)
from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseSetIdentity,
    canonical_database_source_manifest_bytes,
    database_source_manifest_from_mapping,
)
from bspp.orchestration.contract.phase import (
    PhasePlan,
    PreprocessingPhasePlanPayload,
    VerifiedLocalInputLocation,
    phase_plan_from_mapping,
    phase_runspec_from_mapping,
)
from bspp.orchestration.contract.phase_state import (
    phase_materialized_event_from_mapping,
    phase_run_from_mapping,
)
from bspp.orchestration.contract.preprocessing import PreprocessingPlanOptions
from bspp.orchestration.contract.preprocessing_execution import (
    PreprocessingRuntimeCoordinates,
    PreprocessingScientificConfig,
    PreprocessingSiteConfig,
    preprocessing_chunk_execution_intent_from_plan,
)
from bspp.orchestration.contract.preprocessing_runtime import (
    PREPROCESSING_ADAPTER_VERSION,
    PREPROCESSING_RUNTIME_COMMAND,
    PREPROCESSING_RUNTIME_CONTRACT_ID,
    PreprocessingRuntimeGpuEvidence,
    PreprocessingRuntimeImageEvidence,
    PreprocessingRuntimeQualificationRecord,
    PreprocessingRuntimeSmokeEvidence,
    PreprocessingRuntimeSourceEvidence,
    PreprocessingRuntimeToolEvidence,
    preprocessing_runtime_tuple_id,
)
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.phase_authority import (
    PhaseAuthorityCollisionError,
    PhaseAuthorityStore,
)
from bspp.orchestration.control.phase_materialization import materialize_phase
from bspp.orchestration.control.phase_rendering import render_phase_submission_intent
from bspp.orchestration.control.preprocessing_runtime_qualification import (
    preprocessing_runtime_qualification_path,
    preprocessing_runtime_qualification_tuple,
)
from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.runtime.preprocessing.commands import plan_preprocessing_chunk_execution
from bspp.orchestration.runtime.preprocessing.planning import plan_preprocessing_fasta, plan_preprocessing_records

FIXED_RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
FIXED_TIME = datetime(2026, 8, 19, 12, 34, 56, 123456, tzinfo=UTC)


@dataclass(frozen=True)
class MaterializationFixture:
    plan: PhasePlan
    plan_path: Path
    profile_path: Path
    source_repo: Path


def _append_unique_action(data: dict[str, object]) -> None:
    actions = data["payload"]["actions"]
    action = deepcopy(actions[0])
    action["action_id"] = "preprocessing-chunk-000001"
    actions.append(action)


def _make_action_cycle(data: dict[str, object]) -> None:
    actions = data["payload"]["actions"]
    second = deepcopy(actions[0])
    second["action_id"] = "preprocessing-chunk-000001"
    second["dependencies"] = [actions[0]["action_id"]]
    actions[0]["dependencies"] = [second["action_id"]]
    actions.append(second)


def _make_known_nonempty_dependency(data: dict[str, object]) -> None:
    actions = data["payload"]["actions"]
    second = deepcopy(actions[0])
    second["action_id"] = "preprocessing-chunk-000001"
    actions[0]["dependencies"] = [second["action_id"]]
    actions.append(second)


def test_phase_plan_round_trip_has_stable_digest_and_exact_port_records(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    loaded = phase_plan_from_mapping(fixture.plan.to_mapping())

    assert loaded == fixture.plan
    assert loaded.digest == fixture.plan.digest
    assert re.fullmatch(r"[0-9a-f]{64}", loaded.digest)
    assert len(loaded.payload.work_plan.tranches) == 1
    assert len(loaded.payload.work_plan.chunks) == 1
    assert loaded.payload.chunk_execution_intent.chunk_name == loaded.payload.work_plan.chunks[0].name


@pytest.mark.parametrize("scientific_version", [2, 3])
def test_scientific_plan_runspec_and_event_pass_recursive_version_validation(
    tmp_path: Path, scientific_version: int
) -> None:
    fixture = _fixture(tmp_path)
    mapping = fixture.plan.to_mapping()
    mapping["payload"]["chunk_execution_intent"]["scientific"]["schema_version"] = scientific_version
    fixture = replace(fixture, plan=phase_plan_from_mapping(mapping))
    fixture.plan_path.write_text(yaml.safe_dump(mapping))
    assert fixture.plan.payload.chunk_execution_intent.scientific.schema_version == scientific_version

    # phase_plan_from_mapping recurses over the payload and must accept the scoped scientific version.
    assert phase_plan_from_mapping(fixture.plan.to_mapping()) == fixture.plan

    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    runspec_mapping = json.loads(
        (authority_root / FIXED_RUN_ID / "attempts" / "attempt-0001" / "phase-runspec.json").read_text()
    )
    loaded_runspec = phase_runspec_from_mapping(runspec_mapping)
    assert loaded_runspec.payload.actions[0].payload.scientific.schema_version == scientific_version

    # PhaseAuthorityStore.validate loads the materialized event through the recursive
    # phase_state validator, so it pins the event path explicitly.
    authority = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert authority.phase_runspec.payload.actions[0].payload.scientific.schema_version == scientific_version


def test_phase_plan_rejects_scientific_schema_version_4(tmp_path: Path) -> None:
    mapping = deepcopy(_fixture(tmp_path).plan.to_mapping())
    mapping["payload"]["chunk_execution_intent"]["scientific"]["schema_version"] = 4

    with pytest.raises(ValueError, match="scientific schema_version 4"):
        phase_plan_from_mapping(mapping)


def test_phase_plan_rejects_non_scientific_schema_version_2(tmp_path: Path) -> None:
    mapping = deepcopy(_fixture(tmp_path).plan.to_mapping())
    mapping["payload"]["chunk_execution_intent"]["runtime"]["schema_version"] = 2

    with pytest.raises(ValueError, match="runtime schema_version 2"):
        phase_plan_from_mapping(mapping)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update({"unexpected": True}), "Unknown PhasePlan"),
        (lambda data: data.update({"phase_kind": "folding"}), "unsupported PhasePlan phase_kind"),
        (
            lambda data: data["payload"].update({"phase_kind": "folding"}),
            "unsupported PreprocessingPhasePlanPayload phase_kind",
        ),
        (lambda data: data["payload"].pop("schema_version"), "missing explicit schema_version"),
        (
            lambda data: data["payload"]["work_plan"]["input"]["records"][0].update({"schema_version": 99}),
            "Unsupported",
        ),
        (lambda data: data.update({"target_cluster": "${BSPP_CLUSTER}"}), "environment interpolation"),
    ],
)
def test_phase_plan_loader_fails_closed_recursively(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    mapping = deepcopy(_fixture(tmp_path).plan.to_mapping())
    assert callable(mutation)
    mutation(mapping)

    with pytest.raises(ValueError, match=message):
        phase_plan_from_mapping(mapping)


def test_phase_plan_rejects_zero_work_and_mismatched_expected_source_identity(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    empty_work = plan_preprocessing_records(
        source_path=str(tmp_path / "empty.fa"),
        records=(),
        options=PreprocessingPlanOptions(),
    )

    with pytest.raises(ValueError, match="exactly one non-empty"):
        PreprocessingPhasePlanPayload(
            work_plan=empty_work,
            chunk_execution_intent=fixture.plan.payload.chunk_execution_intent,
            database=fixture.plan.payload.database,
        )

    mapping = deepcopy(fixture.plan.to_mapping())
    mapping["payload"]["chunk_execution_intent"]["expected_a3ms"][0]["record_identity"] = "different"
    with pytest.raises(ValueError, match="record identity must match"):
        phase_plan_from_mapping(mapping)


@pytest.mark.parametrize("sequence", ["AAAA", "AAAA:AAAA", "AAAA:TT:GG", "AAAA:TT:GG:CC"])
def test_phase_plan_accepts_n_ary_records(
    tmp_path: Path,
    sequence: str,
) -> None:
    """Monomer, homodimer, trimer, and tetramer records are now admissible."""
    mapping = deepcopy(_fixture(tmp_path).plan.to_mapping())
    mapping["payload"]["work_plan"]["input"]["records"][0]["sequence"] = sequence

    phase_plan_from_mapping(mapping)


@pytest.mark.parametrize("sequence", ["AAAA:", ":TT"])
def test_phase_plan_rejects_empty_chain_records(
    tmp_path: Path,
    sequence: str,
) -> None:
    mapping = deepcopy(_fixture(tmp_path).plan.to_mapping())
    mapping["payload"]["work_plan"]["input"]["records"][0]["sequence"] = sequence

    with pytest.raises(ValueError, match="each searched record must have at least one non-empty chain"):
        phase_plan_from_mapping(mapping)


def test_materialization_publishes_exact_replayable_scheduler_free_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bspp.orchestration.control import transport

    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    before_runtime_modules = {name for name in sys.modules if name.startswith("bspp.orchestration.runtime")}

    def reject_runner(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Phase Materialization must not invoke a command runner")

    monkeypatch.setattr(transport, "default_command_runner", reject_runner)

    result = materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )

    run_root = authority_root / FIXED_RUN_ID
    assert result.to_mapping() == {
        "phase_run_id": FIXED_RUN_ID,
        "attempt_id": "attempt-0001",
        "phase_runspec_digest": result.phase_runspec_digest,
        "authority_root": str(authority_root),
    }
    assert _relative_entries(run_root) == {
        "attempts",
        "attempts/attempt-0001",
        "attempts/attempt-0001/database-source-manifest.json",
        "attempts/attempt-0001/phase-runspec.json",
        "events",
        "events/000001-phase-materialized.json",
        "phase-plan.json",
        "phase-run.json",
    }
    validation = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert validation.phase_plan == fixture.plan
    assert validation.phase_run.phase_plan_digest == fixture.plan.digest
    assert validation.phase_runspec.digest == result.phase_runspec_digest
    assert validation.materialized_event.payload.phase_run == validation.phase_run
    assert validation.materialized_event.payload.phase_runspec == validation.phase_runspec
    assert validation.phase_runspec.cluster.profile_name == "example-cluster"
    assert validation.phase_runspec.cluster.runtime_image == "/images/preprocessing.sqsh"
    assert validation.phase_runspec.cluster.preprocessing_runtime.qualification_tuple.source_bundle_sha256 == "1" * 64
    assert validation.phase_runspec.cluster.extra_mounts[0].target == "/mounted"
    assert validation.phase_runspec.payload.actions[0].dependencies == ()
    assert validation.phase_runspec.payload.actions[0].resources.partition == "example-gpu"
    database = validation.phase_runspec.payload.database
    action_payload = validation.phase_runspec.payload.actions[0].payload
    assert database.staging is not None
    assert database.staging.unix_user == "tester"
    assert database.selected_container_root == action_payload.site.database_root == SELECTED_DATABASE_ROOT
    assert database.branches[0].gpuserver_argv == action_payload.gpuserver_argv
    assert database.branches[0].search_argv == action_payload.search_argv
    assert (run_root / database.source_manifest_projection).read_bytes() == canonical_database_source_manifest_bytes(
        database.source_manifest
    )
    assert (run_root / database.source_manifest_projection).read_bytes() == (
        tmp_path / "database-source-manifest.json"
    ).read_bytes()
    after_runtime_modules = {name for name in sys.modules if name.startswith("bspp.orchestration.runtime")}
    assert after_runtime_modules == before_runtime_modules


def test_initial_database_manifest_projection_is_published_read_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    manifest_path = authority_root / FIXED_RUN_ID / "attempts/attempt-0001/database-source-manifest.json"

    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o444


def test_current_main_profile_cannot_materialize_preprocessing_authority(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    profile_mapping = yaml.safe_load(fixture.profile_path.read_text())
    cluster = profile_mapping["clusters"]["example-cluster"]
    for key in (
        "database_sets",
        "database_access_policies",
        "database_cache_root",
        "database_cache_unix_user",
        "database_cache_filesystem_type",
        "database_cache_reserve_bytes",
        "database_lock_wait_seconds",
    ):
        cluster.pop(key)
    fixture.profile_path.write_text(yaml.safe_dump(profile_mapping, sort_keys=True))
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match="requires complete database staging authority"):
        materialize_phase(
            fixture.plan_path,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
        )

    assert not authority_root.exists()


def test_authority_validation_rejects_writable_database_manifest_projection(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    manifest_path = authority_root / FIXED_RUN_ID / "attempts/attempt-0001/database-source-manifest.json"
    manifest_path.chmod(0o644)

    with pytest.raises(ValueError, match="read-only database source-manifest projection"):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)


@pytest.mark.parametrize(
    ("policy", "branch_kinds"),
    [
        (DatabaseAccessPolicy.STAGE_REQUIRED, ("staged",)),
        (
            DatabaseAccessPolicy.STAGE_PREFERRED,
            ("staged", "direct-capacity-fallback"),
        ),
        (DatabaseAccessPolicy.DIRECT, ("direct-requested",)),
    ],
)
def test_materialization_closes_only_the_profile_allowed_database_policy(
    tmp_path: Path,
    policy: DatabaseAccessPolicy,
    branch_kinds: tuple[str, ...],
) -> None:
    fixture = _fixture(tmp_path, policy=policy)
    materialize_phase(
        fixture.plan_path,
        authority_root=tmp_path / "authority",
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )

    binding = PhaseAuthorityStore(tmp_path / "authority").validate(FIXED_RUN_ID).phase_runspec.payload.database
    assert binding.requested_policy == policy
    assert tuple(branch.branch_kind for branch in binding.branches) == branch_kinds
    assert all(not branch.finalization_mounts for branch in binding.branches)
    assert all(
        mount.target != "/run/bspp/database/cache"
        for branch in binding.branches
        if branch.branch_kind != "staged"
        for mount in branch.placement_mounts
    )


def test_initial_materialization_preserves_array_resource_compatibility(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    profile_mapping = yaml.safe_load(fixture.profile_path.read_text())
    profile_mapping["clusters"]["example-cluster"]["resources"] = {"gpu_worker": {"array": "0-3%2"}}
    fixture.profile_path.write_text(yaml.safe_dump(profile_mapping, sort_keys=True))
    authority_root = tmp_path / "authority"

    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )

    authority = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    assert authority.phase_runspec.payload.actions[0].resources.array == "0-3%2"


def test_packed_topology_materialization_preserves_typed_resources_and_renders_derived_gres(tmp_path: Path) -> None:
    """A packed-topology profile materializes actions that retain nodes/gpus_per_task
    and render the derived --gres line in the submission script."""
    fixture = _fixture(tmp_path)
    profile_mapping = yaml.safe_load(fixture.profile_path.read_text())
    cluster = profile_mapping["clusters"]["example-cluster"]
    cluster["resources"] = {
        "gpu_worker": {
            "gres": None,
            "nodes": 2,
            "tasks_per_node": 5,
            "gpus_per_task": 1,
        }
    }
    fixture.profile_path.write_text(yaml.safe_dump(profile_mapping, sort_keys=True))
    authority_root = tmp_path / "authority"

    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )

    authority = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    action = authority.phase_runspec.payload.actions[0]
    assert action.resources.gres is None
    assert action.resources.nodes == 2
    assert action.resources.gpus_per_task == 1
    qualification = authority.phase_runspec.cluster.preprocessing_runtime.qualification_tuple
    assert qualification.gpu_worker_gres == "gpu:1"

    runspec_path = authority.authority_path / authority.phase_run.attempts[0].phase_runspec_location
    document_hash = hashlib.sha256(runspec_path.read_bytes()).hexdigest()
    intent = render_phase_submission_intent(
        authority.phase_runspec,
        phase_runspec_location=authority.phase_run.attempts[0].phase_runspec_location,
        phase_runspec_document_sha256=document_hash,
    )
    script = intent.actions[0].script_body
    assert "#SBATCH --gres=gpu:1" in script


def test_materialization_freezes_cold_acceptance_node_into_the_slurm_action(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, nodelist="gpu-node-017")
    authority_root = tmp_path / "authority"

    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )

    authority = PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
    action = authority.phase_runspec.payload.actions[0]
    runspec_path = authority.authority_path / authority.phase_run.attempts[0].phase_runspec_location
    document_sha256 = hashlib.sha256(runspec_path.read_bytes()).hexdigest()
    intent = render_phase_submission_intent(
        authority.phase_runspec,
        phase_runspec_location=authority.phase_run.attempts[0].phase_runspec_location,
        phase_runspec_document_sha256=document_sha256,
    )

    assert action.resources.nodelist == "gpu-node-017"
    assert action.resources.to_mapping()["nodelist"] == "gpu-node-017"
    assert intent.actions[0].script_body.count("#SBATCH --nodelist=gpu-node-017") == 1


def test_initial_materialization_resolves_profile_before_invoking_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bspp.orchestration.control import phase_materialization

    fixture = _fixture(tmp_path)

    def reject_profile(*_args: object, **_kwargs: object) -> object:
        raise ValueError("profile resolution failed first")

    monkeypatch.setattr(phase_materialization, "resolve_cluster_profile", reject_profile)
    with pytest.raises(ValueError, match="profile resolution failed first"):
        materialize_phase(
            fixture.plan_path,
            authority_root=tmp_path / "authority",
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: (_ for _ in ()).throw(AssertionError("clock must not run before profile resolution")),
        )


def test_authority_bytes_are_identical_across_roots_for_same_materialization(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    roots = (tmp_path / "authority-a", tmp_path / "authority-b")
    for root in roots:
        materialize_phase(
            fixture.plan_path,
            authority_root=root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: FIXED_TIME,
            phase_run_id_factory=lambda: FIXED_RUN_ID,
        )

    first = _file_bytes(roots[0] / FIXED_RUN_ID)
    second = _file_bytes(roots[1] / FIXED_RUN_ID)
    assert first == second


def test_materialization_rejects_input_hash_and_size_mismatch_before_publication(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.plan_path.parent.joinpath("input.fa").write_text(">alpha\nCHANGED\n")
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match=r"size mismatch|SHA-256 mismatch"):
        materialize_phase(
            fixture.plan_path,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
        )

    assert not authority_root.exists()


def test_materialization_rejects_noncanonical_database_source_manifest_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = tmp_path / "database-source-manifest.json"
    manifest_path.write_text(json.dumps(json.loads(manifest_path.read_bytes()), sort_keys=True))
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match="exact canonical encoding"):
        materialize_phase(
            fixture.plan_path,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: FIXED_TIME,
            phase_run_id_factory=lambda: FIXED_RUN_ID,
        )

    assert not authority_root.exists()


def test_materialization_rejects_database_source_manifest_identity_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = tmp_path / "database-source-manifest.json"
    mapping = json.loads(manifest_path.read_bytes())
    mapping["database_source_manifest"]["database_set"]["version"] = "different"
    mismatched = database_source_manifest_from_mapping(mapping)
    manifest_path.write_bytes(canonical_database_source_manifest_bytes(mismatched))
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match="identity does not match"):
        materialize_phase(
            fixture.plan_path,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: FIXED_TIME,
            phase_run_id_factory=lambda: FIXED_RUN_ID,
        )

    assert not authority_root.exists()


def test_materialization_rejects_symlinked_database_source_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = tmp_path / "database-source-manifest.json"
    external_copy = tmp_path / "external-database-source-manifest.json"
    external_copy.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(external_copy)
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match="regular non-symlink file"):
        materialize_phase(
            fixture.plan_path,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: FIXED_TIME,
            phase_run_id_factory=lambda: FIXED_RUN_ID,
        )

    assert not authority_root.exists()


@pytest.mark.parametrize(
    ("site_field", "replacement", "message"),
    [
        ("container_image", "/images/unqualified.sqsh", "container_image does not match"),
        ("mmseqs_executable", "/opt/mmseqs/bin/mmseqs", "executable paths do not match"),
        (
            "colabfold_search_executable",
            "/opt/colabfold/bin/colabfold_search",
            "executable paths do not match",
        ),
        ("tar_executable", "/opt/archive/bin/tar", "executable paths do not match"),
        ("lz4_executable", "/opt/archive/bin/lz4", "executable paths do not match"),
    ],
    ids=["image", "mmseqs", "colabfold", "tar", "lz4"],
)
def test_materialization_rejects_unqualified_plan_runtime_before_authority_publication(
    tmp_path: Path,
    site_field: str,
    replacement: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    original = fixture.plan.payload.chunk_execution_intent
    changed_site = original.site.model_copy(update={site_field: replacement})
    changed_package = original.package
    if site_field == "tar_executable":
        changed_package = replace(
            changed_package,
            tar_argv=(replacement, *changed_package.tar_argv[1:]),
        )
    elif site_field == "lz4_executable":
        changed_package = replace(
            changed_package,
            lz4_argv=(replacement, *changed_package.lz4_argv[1:]),
        )
    changed_execution = replace(original, site=changed_site, package=changed_package)
    changed_plan = replace(
        fixture.plan,
        payload=replace(fixture.plan.payload, chunk_execution_intent=changed_execution),
    )
    fixture.plan_path.write_text(yaml.safe_dump(changed_plan.to_mapping(), sort_keys=True))
    authority_root = tmp_path / "rejected-authority"

    with pytest.raises(ValueError, match=message):
        materialize_phase(
            fixture.plan_path,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: FIXED_TIME,
            phase_run_id_factory=lambda: FIXED_RUN_ID,
        )

    assert not authority_root.exists()


def test_materialization_refuses_use_env_true_before_authority_publication(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, use_env=True)
    authority_root = tmp_path / "rejected-authority"

    with pytest.raises(ValueError, match=r"use_env=true.*adapter-v3.*characterization"):
        materialize_phase(
            fixture.plan_path,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: FIXED_TIME,
            phase_run_id_factory=lambda: FIXED_RUN_ID,
        )

    assert not authority_root.exists()


def test_materialization_accepts_gate_on_conforming_stem_and_pins_use_env_off(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        expected_a3m_members=("AFDB_AF-1234567890123456.a3m",),
        require_afdb_model_id_stem=True,
        use_env=False,
        source_bytes=b">AFDB_AF-1234567890123456\nAAAA:TT\n",
    )
    authority_root = tmp_path / "authority"

    result = materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )

    assert result.phase_run_id == FIXED_RUN_ID
    runspec_path = authority_root / FIXED_RUN_ID / "attempts" / "attempt-0001" / "phase-runspec.json"
    runspec = phase_runspec_from_mapping(json.loads(runspec_path.read_text()))
    action_payload = runspec.payload.actions[0].payload
    assert action_payload.scientific.use_env is False
    assert action_payload.scientific.require_afdb_model_id_stem is True
    assert action_payload.search_argv[action_payload.search_argv.index("--use-env") + 1] == "0"


def test_existing_run_and_existing_empty_directory_are_classified_as_collisions(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    existing_bytes = _file_bytes(authority_root / FIXED_RUN_ID)

    with pytest.raises(PhaseAuthorityCollisionError, match=FIXED_RUN_ID):
        materialize_phase(
            fixture.plan_path,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: FIXED_TIME,
            phase_run_id_factory=lambda: FIXED_RUN_ID,
        )
    assert _file_bytes(authority_root / FIXED_RUN_ID) == existing_bytes

    empty_run_id = "phase-run-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    empty_destination = authority_root / empty_run_id
    empty_destination.mkdir()
    with pytest.raises(PhaseAuthorityCollisionError, match=empty_run_id):
        materialize_phase(
            fixture.plan_path,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: FIXED_TIME,
            phase_run_id_factory=lambda: empty_run_id,
        )
    assert list(empty_destination.iterdir()) == []


def test_prepublication_fault_leaves_no_visible_partial_run(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"

    def fail(_staging_path: Path) -> None:
        raise RuntimeError("injected publication fault")

    with pytest.raises(RuntimeError, match="injected publication fault"):
        materialize_phase(
            fixture.plan_path,
            authority_root=authority_root,
            config_path=fixture.profile_path,
            source_repo=fixture.source_repo,
            clock=lambda: FIXED_TIME,
            phase_run_id_factory=lambda: FIXED_RUN_ID,
            authority_store=PhaseAuthorityStore(authority_root, before_publish=fail),
        )

    assert [path.name for path in authority_root.iterdir() if not path.name.startswith(".")] == []
    assert not any(path.name.startswith(f".{FIXED_RUN_ID}.staging-") for path in authority_root.iterdir())


def test_authority_validation_rejects_symlinked_phase_run_directory(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    real_authority_root = tmp_path / "real-authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=real_authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    alias_authority_root = tmp_path / "alias-authority"
    alias_authority_root.mkdir()
    (alias_authority_root / FIXED_RUN_ID).symlink_to(
        real_authority_root / FIXED_RUN_ID,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="missing Phase Run authority directory"):
        PhaseAuthorityStore(alias_authority_root).validate(FIXED_RUN_ID)


def test_authority_validation_rejects_symlink_inside_exact_layout(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    phase_plan_path = authority_root / FIXED_RUN_ID / "phase-plan.json"
    external_copy = tmp_path / "external-phase-plan.json"
    external_copy.write_bytes(phase_plan_path.read_bytes())
    phase_plan_path.unlink()
    phase_plan_path.symlink_to(external_copy)

    with pytest.raises(ValueError, match=r"must not contain symlinks: phase-plan\.json"):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)


def test_authority_validation_rejects_database_source_manifest_projection_tampering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    manifest_path = authority_root / FIXED_RUN_ID / "attempts/attempt-0001/database-source-manifest.json"
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    manifest_path.chmod(0o444)

    with pytest.raises(ValueError, match="differs from embedded RunSpec authority"):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)


@pytest.mark.parametrize("damage", ["missing", "extra", "tamper", "gap"])
def test_authority_validation_fails_closed_for_incomplete_or_tampered_state(tmp_path: Path, damage: str) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    run_root = authority_root / FIXED_RUN_ID
    event_path = run_root / "events/000001-phase-materialized.json"
    if damage == "missing":
        event_path.unlink()
    elif damage == "extra":
        (run_root / "extra.json").write_text("{}\n")
    elif damage == "tamper":
        payload = json.loads((run_root / "phase-run.json").read_text())
        payload["phase_plan_digest"] = "f" * 64
        (run_root / "phase-run.json").write_text(json.dumps(payload))
    else:
        event_path.rename(run_root / "events/000002-phase-materialized.json")

    with pytest.raises(ValueError):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)


def test_phase_runspec_loader_requires_explicit_embedded_database_source_manifest_version(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = materialize_phase(
        fixture.plan_path,
        authority_root=tmp_path / "authority",
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    mapping = json.loads(
        (result.authority_root / result.phase_run_id / "attempts/attempt-0001/phase-runspec.json").read_text()
    )
    mapping["payload"]["database"]["source_manifest"]["database_source_manifest"].pop("schema_version")
    standalone_manifest = database_source_manifest_from_mapping(mapping["payload"]["database"]["source_manifest"])

    assert standalone_manifest.schema_version == 1

    with pytest.raises(
        ValueError,
        match=(
            r"missing explicit schema_version at "
            r"phase_runspec\.payload\.database\.source_manifest\.database_source_manifest"
        ),
    ):
        phase_runspec_from_mapping(mapping)


def test_phase_runspec_loader_rejects_unsupported_embedded_database_source_manifest_version(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = materialize_phase(
        fixture.plan_path,
        authority_root=tmp_path / "authority",
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    mapping = json.loads(
        (result.authority_root / result.phase_run_id / "attempts/attempt-0001/phase-runspec.json").read_text()
    )
    mapping["payload"]["database"]["source_manifest"]["database_source_manifest"]["schema_version"] = 2

    with pytest.raises(
        ValueError,
        match=(
            r"Unsupported phase_runspec\.payload\.database\.source_manifest\.database_source_manifest "
            r"schema_version 2"
        ),
    ):
        phase_runspec_from_mapping(mapping)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["payload"].update({"actions": []}), "exactly one Runtime Action"),
        (
            lambda data: data["payload"]["actions"].append(deepcopy(data["payload"]["actions"][0])),
            "ids must be unique",
        ),
        (_append_unique_action, "exactly one Runtime Action"),
        (
            lambda data: data["payload"]["actions"][0].update(
                {"dependencies": [data["payload"]["actions"][0]["action_id"]]}
            ),
            "cannot depend on itself",
        ),
        (
            lambda data: data["payload"]["actions"][0].update({"dependencies": ["preprocessing-chunk-999999"]}),
            "dangling dependencies",
        ),
        (_make_action_cycle, "contains a cycle"),
        (_make_known_nonempty_dependency, "exactly one Runtime Action"),
    ],
)
def test_phase_runspec_loader_rejects_invalid_action_graphs(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    result = materialize_phase(
        fixture.plan_path,
        authority_root=tmp_path / "authority",
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    mapping = json.loads(
        (result.authority_root / result.phase_run_id / "attempts/attempt-0001/phase-runspec.json").read_text()
    )
    assert callable(mutation)
    mutation(mapping)

    with pytest.raises(ValueError, match=message):
        phase_runspec_from_mapping(mapping)


def test_state_loaders_reject_event_sequence_and_cross_record_tampering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    run_mapping = json.loads((authority_root / FIXED_RUN_ID / "phase-run.json").read_text())
    event_mapping = json.loads((authority_root / FIXED_RUN_ID / "events/000001-phase-materialized.json").read_text())
    run_mapping["attempts"][0].pop("schema_version")
    with pytest.raises(ValueError, match="missing explicit schema_version"):
        phase_run_from_mapping(run_mapping)

    event_mapping["sequence"] = 2
    with pytest.raises(ValueError, match="sequence 1"):
        phase_materialized_event_from_mapping(event_mapping)


def test_phase_materialize_cli_emits_json_and_leaves_legacy_run_help_unchanged(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    runner = CliRunner()
    run_help_before = runner.invoke(cli, ["run", "--help"]).output

    result = runner.invoke(
        cli,
        [
            "--config",
            str(fixture.profile_path),
            "phase",
            "materialize",
            str(fixture.plan_path),
            "--authority-root",
            str(authority_root),
            "--source-repo",
            str(fixture.source_repo),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    payload = json.loads(result.output)
    assert re.fullmatch(r"phase-run-[0-9a-f]{32}", payload["phase_run_id"])
    assert payload["attempt_id"] == "attempt-0001"
    assert payload["authority_root"] == str(authority_root)
    assert (
        PhaseAuthorityStore(authority_root).validate(payload["phase_run_id"]).phase_runspec.digest
        == payload["phase_runspec_digest"]
    )
    assert runner.invoke(cli, ["run", "--help"]).output == run_help_before


def test_cli_reports_schema_and_collision_errors_without_traceback(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    invalid = yaml.safe_load(fixture.plan_path.read_text())
    invalid["unexpected"] = True
    fixture.plan_path.write_text(yaml.safe_dump(invalid, sort_keys=True))

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(fixture.profile_path),
            "phase",
            "materialize",
            str(fixture.plan_path),
            "--authority-root",
            str(tmp_path / "authority"),
        ],
    )

    assert result.exit_code != 0
    assert "Unknown PhasePlan field" in result.output
    assert "Traceback" not in result.output


def _fixture(
    tmp_path: Path,
    *,
    policy: DatabaseAccessPolicy = DatabaseAccessPolicy.STAGE_REQUIRED,
    nodelist: str | None = None,
    use_env: bool = False,
    expected_a3m_members: tuple[str, ...] = ("alpha.a3m",),
    require_afdb_model_id_stem: bool = False,
    source_bytes: bytes = b">alpha description\nAAAA:TT\n",
) -> MaterializationFixture:
    source_repo = tmp_path / "source-repo"
    source_repo.mkdir(parents=True)
    subprocess.run(("git", "init", "-q", str(source_repo)), check=True)
    subprocess.run(("git", "-C", str(source_repo), "config", "user.email", "test@example.invalid"), check=True)
    subprocess.run(("git", "-C", str(source_repo), "config", "user.name", "Test"), check=True)
    (source_repo / "source.txt").write_text("qualified source\n")
    subprocess.run(("git", "-C", str(source_repo), "add", "source.txt"), check=True)
    subprocess.run(("git", "-C", str(source_repo), "commit", "-qm", "fixture"), check=True)
    source_commit = subprocess.run(
        ("git", "-C", str(source_repo), "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    source = tmp_path / "input.fa"
    source.write_bytes(source_bytes)
    work_plan = plan_preprocessing_fasta(
        source,
        PreprocessingPlanOptions(
            requested_tranches=1,
            records_per_chunk=300,
            nodes=1,
            gpus_per_node=1,
        ),
    )
    execution_plan = plan_preprocessing_chunk_execution(
        chunk=work_plan.chunks[0],
        records=work_plan.input.records,
        expected_a3m_members=expected_a3m_members,
        scientific=PreprocessingScientificConfig(
            require_afdb_model_id_stem=require_afdb_model_id_stem,
            use_env=use_env,
        ),
        site=PreprocessingSiteConfig(
            mmseqs_executable="/usr/local/bin/mmseqs",
            colabfold_search_executable="/usr/local/bin/colabfold_search",
            tar_executable="/usr/bin/tar",
            lz4_executable="/usr/bin/lz4",
            database_root="/databases",
            input_root="/phase/input",
            scratch_output_root="/phase/scratch",
            project_logs_root="/phase/logs",
            finished_msa_root="/phase/finished-msa",
            split_input_root="/phase/split-input",
            finished_input_root="/phase/finished-input",
            container_image="/images/preprocessing.sqsh",
            container_mounts=(),
            max_concurrency=1,
            gpu_delay_seconds=0,
        ),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    phase_plan = PhasePlan(
        target_cluster="example-cluster",
        input_location=VerifiedLocalInputLocation(
            path=str(source),
            sha256=hashlib.sha256(source_bytes).hexdigest(),
            size_bytes=len(source_bytes),
        ),
        payload=PreprocessingPhasePlanPayload(
            work_plan=work_plan,
            chunk_execution_intent=preprocessing_chunk_execution_intent_from_plan(execution_plan),
            database=DatabaseSetSelection(
                database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"),
                requested_policy=policy,
            ),
        ),
    )
    plan_path = tmp_path / "phase-plan.yaml"
    plan_path.write_text(yaml.safe_dump(phase_plan.to_mapping(), sort_keys=True))
    profile_path = tmp_path / "profiles.yaml"
    manifest = database_source_manifest_from_mapping(
        {
            "database_source_manifest": {
                "schema_version": 1,
                "database_set": {"identifier": "bspp-search", "version": "2026-08"},
                "verification": "metadata-verified",
                "source_root": "/databases",
                "members": [
                    {
                        "role": "primary",
                        "database_name": "uniref30_2302_db",
                        "logical_name": "uniref30_2302_db",
                        "source_path": "uniref30_2302_db",
                        "source_kind": "regular",
                        "resolved_path": "uniref30_2302_db",
                        "resolved_kind": "regular",
                        "size_bytes": 1,
                        "mtime_ns": 1,
                        "alias_topology": [],
                        "preexisting_checksum": None,
                    },
                    {
                        "role": "metagenomic",
                        "database_name": "colabfold_envdb_202108_db",
                        "logical_name": "colabfold_envdb_202108_db",
                        "source_path": "colabfold_envdb_202108_db",
                        "source_kind": "regular",
                        "resolved_path": "colabfold_envdb_202108_db",
                        "resolved_kind": "regular",
                        "size_bytes": 1,
                        "mtime_ns": 1,
                        "alias_topology": [],
                        "preexisting_checksum": None,
                    },
                ],
            }
        }
    )
    manifest_path = tmp_path / "database-source-manifest.json"
    manifest_path.write_bytes(canonical_database_source_manifest_bytes(manifest))
    profile_path.write_text(
        yaml.safe_dump(
            {
                "clusters": {
                    "example-cluster": {
                        "owner": "tester",
                        "transport": "local-slurm",
                        "paths": {
                            "project_root": "/project",
                            "output_root": "/output",
                            "staging_root": "/staging",
                            "afdb_toolkit_repo": "/unused-toolkit",
                            "orchestration_repo": str(source_repo),
                            "image": "/images/postprocessing.sqsh",
                            "source_bundle_root": str(tmp_path / "source-bundles"),
                            "runtime_image_cache_root": "/image-cache",
                            "runtime_qualification_root": str(tmp_path / "qualifications"),
                        },
                        "preprocessing_runtime": {
                            "cluster_image_path": "/images/preprocessing.sqsh",
                            "cluster_image_sha256": "a" * 64,
                            "oci_digest": "sha256:" + "b" * 64,
                            "image_lock_sha256": "c" * 64,
                            "contract_wheel_sha256": "d" * 64,
                            "runtime_wheel_sha256": "e" * 64,
                            "control_wheel_sha256": "f" * 64,
                            "source_commit": source_commit,
                            "source_bundle_sha256": "1" * 64,
                            "colabfold_version": "1.6.2",
                            "mmseqs_version": "18-8cc5c",
                            "rsync_version": "3.4.4",
                            "cuda_version": "12.6.3",
                        },
                        "database_sets": [
                            {
                                "identifier": "bspp-search",
                                "version": "2026-08",
                                "manifest_path": str(manifest_path),
                            }
                        ],
                        "database_access_policies": [policy.value],
                        "database_cache_root": "/node-cache/bspp",
                        "database_cache_unix_user": "tester",
                        "database_cache_filesystem_type": "ext4",
                        "database_cache_reserve_bytes": 0,
                        "database_lock_wait_seconds": 60,
                        "resources": {"gpu_worker": {"nodelist": nodelist}} if nodelist is not None else {},
                        "extra_mounts": [{"source": "/host-mounted", "target": "/mounted"}],
                    }
                }
            },
            sort_keys=True,
        )
    )
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)
    record_path = preprocessing_runtime_qualification_path(profile, tuple_id=tuple_id)
    record_path.parent.mkdir(parents=True)
    record = PreprocessingRuntimeQualificationRecord(
        status="qualified",
        tuple_id=tuple_id,
        qualification_tuple=qualification_tuple,
        submitted_at="2026-08-19T10:00:00.000000Z",
        qualified_at="2026-08-19T11:00:00.000000Z",
        expires_at="2030-08-26T11:00:00.000000Z",
        job_id="12345",
        smoke_evidence=PreprocessingRuntimeSmokeEvidence(
            runtime_command=PREPROCESSING_RUNTIME_COMMAND,
            runtime_contract_id=PREPROCESSING_RUNTIME_CONTRACT_ID,
            adapter_version=PREPROCESSING_ADAPTER_VERSION,
            command_order=("gpuserver", "search", "record-ls", "tar", "lz4"),
            action_evidence_sha256="f" * 64,
            tools=PreprocessingRuntimeToolEvidence(
                python_version="Python 3.12.0",
                contract_version="0.1.0",
                runtime_version="0.1.0",
                control_version="0.1.0",
                mmseqs_version="18-8cc5c",
                rsync_version="3.4.4",
                colabfold_version="1.6.2",
                tar_version="tar 1.35",
                lz4_version="lz4 1.10.0",
                flock_version="flock 2.42",
            ),
            image=PreprocessingRuntimeImageEvidence(
                manifest_path="/opt/bspp/preprocessing-runtime-image.json",
                manifest_sha256="0" * 64,
                cluster_image_sha256="a" * 64,
                oci_digest="sha256:" + "b" * 64,
            ),
            source=PreprocessingRuntimeSourceEvidence(
                bundle_id=qualification_tuple.source_bundle_id,
                bundle_path=qualification_tuple.source_bundle_path,
                bundle_sha256="1" * 64,
            ),
            gpu=PreprocessingRuntimeGpuEvidence(nvidia_smi="Fixture GPU, driver 999"),
        ),
    )
    record_path.write_text(json.dumps(record.to_mapping(), indent=2, sort_keys=True) + "\n")
    return MaterializationFixture(
        plan=phase_plan, plan_path=plan_path, profile_path=profile_path, source_repo=source_repo
    )


def _relative_entries(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*")}


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_authority_validation_rejects_preprocessing_phase_run_kind_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    run_root = authority_root / FIXED_RUN_ID

    run_path = run_root / "phase-run.json"
    run_mapping = json.loads(run_path.read_text())
    run_mapping["phase_kind"] = "folding"
    run_path.write_text(json.dumps(run_mapping, indent=2, sort_keys=True) + "\n")

    event_path = run_root / "events/000001-phase-materialized.json"
    event_mapping = json.loads(event_path.read_text())
    event_mapping["payload"]["phase_run"] = run_mapping
    event_path.write_text(json.dumps(event_mapping, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="phase_kind does not match"):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)


def test_paired_filter_policy_changes_scientific_identity_but_not_input_identity(tmp_path: Path) -> None:
    from bspp.orchestration.contract.phase_retry import (
        phase_input_set_identity_digest,
        phase_scientific_identity_digest,
    )

    current = _fixture(tmp_path).plan
    mapping = current.to_mapping()
    mapping["payload"]["chunk_execution_intent"]["scientific"]["schema_version"] = 2
    historical = phase_plan_from_mapping(mapping)
    assert current.payload.chunk_execution_intent.scientific.schema_version == 3
    assert phase_input_set_identity_digest(current) == phase_input_set_identity_digest(historical)
    assert phase_scientific_identity_digest(current) != phase_scientific_identity_digest(historical)
