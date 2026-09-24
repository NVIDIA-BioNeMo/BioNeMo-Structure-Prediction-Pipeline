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

"""Phase-local mount descriptor and no-carry rendering compatibility tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from bspp.orchestration.contract.database_placement import (
    DATABASE_CACHE_ROOT,
    DATABASE_REPLICA_LEASE_TARGET,
    DATABASE_SOURCE_ROOT,
    SELECTED_DATABASE_ROOT,
)
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardRequest,
    AttemptCarryForwardSelection,
)
from bspp.orchestration.contract.phase_submission import PhaseContainerMountDescriptor
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.control.phase_carry_forward import derive_attempt_carry_forward
from bspp.orchestration.control.phase_rendering import (
    _carry_bootstrap_lines,
    _container_mounts,
    render_phase_submission_intent,
)
from bspp.orchestration.runtime.preprocessing.commands import plan_preprocessing_chunk_execution
from tests.support.preprocessing_execution import preprocessing_execution_fixture
from tests.test_phase_carry_forward import _carried_retry_payload, _failed_attested_authority
from tests.test_phase_resume import _materialized_authority


def test_no_carry_renderer_is_deterministic_and_omits_every_carry_field(tmp_path: Path) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    path = authority.authority_path / authority.current_attempt.phase_runspec_location
    document_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    first = render_phase_submission_intent(
        authority.phase_runspec,
        phase_runspec_location=authority.current_attempt.phase_runspec_location,
        phase_runspec_document_sha256=document_sha256,
    )
    second = render_phase_submission_intent(
        authority.phase_runspec,
        phase_runspec_location=authority.current_attempt.phase_runspec_location,
        phase_runspec_document_sha256=document_sha256,
    )
    assert first.to_mapping() == second.to_mapping()
    assert first.submission_id == second.submission_id
    assert first.actions[0].script_body == second.actions[0].script_body
    assert "carry_forward" not in str(first.to_mapping())
    assert "--carry-forward-record" not in first.actions[0].script_body
    assert "--phase-submission-id" not in first.actions[0].script_body


def test_direct_action_renders_ordered_same_allocation_mount_closures_and_status_guard(
    tmp_path: Path,
) -> None:
    authority_root, phase_run_id = _materialized_authority(tmp_path)
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    path = authority.authority_path / authority.current_attempt.phase_runspec_location
    intent = render_phase_submission_intent(
        authority.phase_runspec,
        phase_runspec_location=authority.current_attempt.phase_runspec_location,
        phase_runspec_document_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    script = intent.actions[0].script_body
    database = authority.phase_runspec.payload.database
    branch = database.branches[0]
    source = branch.placement_mounts[0].source
    placement_mount = f"{source}:{DATABASE_SOURCE_ROOT}:ro"
    science_mounts = tuple(f"{mount.source}:{mount.target}:ro" for mount in branch.scientific_mounts)
    mount_lines = tuple(
        line.strip().removesuffix(" \\")
        for line in script.splitlines()
        if line.strip().startswith("--container-mounts=")
    )

    assert script.index("place-database") < script.index("execute-chunk") < script.index("finalize-chunk")
    assert len(mount_lines) == 5
    assert "/run/bspp-acceptance-cache-root" not in script
    assert "mkdir -p --mode=0700" not in script
    assert "database-source-manifest.json" in mount_lines[1]
    assert "database-source-manifest.json" not in mount_lines[0]
    assert "database-source-manifest.json" not in mount_lines[2]
    assert "database-source-manifest.json" not in mount_lines[3]
    assert "database-source-manifest.json" not in mount_lines[4]
    assert placement_mount in mount_lines[1]
    assert all(science_mount not in mount_lines[0] for science_mount in science_mounts)
    assert all(science_mount not in mount_lines[1] for science_mount in science_mounts)
    assert all(science_mount in mount_lines[3] for science_mount in science_mounts)
    assert placement_mount not in mount_lines[0]
    assert placement_mount not in mount_lines[3]
    assert all(source not in line for line in (mount_lines[0], mount_lines[2], mount_lines[4]))
    assert all(DATABASE_SOURCE_ROOT not in line for line in (mount_lines[0], mount_lines[2], mount_lines[4]))
    assert all(SELECTED_DATABASE_ROOT not in line for line in (mount_lines[0], mount_lines[2], mount_lines[4]))
    assert script.index("set +e") < script.index("place-database")
    assert script.index("_placement_status=$?") < script.index("set -e", script.index("set +e") + 1)
    assert "--placement-process-status" in script
    assert '"$_placement_status"' in script
    guard = script.index('if [[ "$_placement_status" -ne 0 || ! -f "$DATABASE_PLACEMENT_RESULT"')
    assert guard < script.index("execute-chunk")
    subprocess.run(("bash", "-n"), input=script, text=True, check=True)


def test_stage_required_renderer_gives_science_only_replica_and_read_only_lease_file(
    tmp_path: Path,
) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work", direct_policy=False)
    database = fixture.runspec.payload.database
    branch = database.branches[0]
    staging = database.staging
    assert staging is not None
    intent = render_phase_submission_intent(
        fixture.runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="a" * 64,
    )
    script = intent.actions[0].script_body
    mount_lines = tuple(
        line.strip().removesuffix(" \\")
        for line in script.splitlines()
        if line.strip().startswith("--container-mounts=")
    )
    source = branch.placement_mounts[0].source
    cache = branch.placement_mounts[1].source
    replica = branch.scientific_mounts[0].source
    lease = branch.scientific_mounts[1].source

    assert len(mount_lines) == 6
    # Bootstrap step creates the effective-user cache namespace in-job.
    assert f"{staging.cache_root}:/run/bspp-acceptance-cache-root" in mount_lines[0]
    assert all(
        value not in mount_lines[0]
        for value in (
            DATABASE_SOURCE_ROOT,
            DATABASE_CACHE_ROOT,
            SELECTED_DATABASE_ROOT,
            DATABASE_REPLICA_LEASE_TARGET,
        )
    )
    assert f"{source}:{DATABASE_SOURCE_ROOT}:ro" in mount_lines[2]
    assert f"{cache}:{DATABASE_CACHE_ROOT}" in mount_lines[2]
    assert all(value not in mount_lines[3] for value in (source, cache, replica, lease))
    assert f"{replica}:{SELECTED_DATABASE_ROOT}:ro" in mount_lines[4]
    assert f"{lease}:{DATABASE_REPLICA_LEASE_TARGET}:ro" in mount_lines[4]
    assert mount_lines[4].count(f"{replica}:{SELECTED_DATABASE_ROOT}:ro") == 1
    assert mount_lines[4].count(f"{lease}:{DATABASE_REPLICA_LEASE_TARGET}:ro") == 1
    assert source not in mount_lines[4]
    assert f"{cache}:{DATABASE_CACHE_ROOT}" not in mount_lines[4]
    assert all(value not in mount_lines[5] for value in (source, cache, replica, lease))
    assert all(
        value not in mount_lines[1] and value not in mount_lines[5]
        for value in (
            DATABASE_SOURCE_ROOT,
            DATABASE_CACHE_ROOT,
            SELECTED_DATABASE_ROOT,
            DATABASE_REPLICA_LEASE_TARGET,
        )
    )
    assert "--placement-process-status" in script
    assert '"$_placement_status"' in script
    subprocess.run(("bash", "-n"), input=script, text=True, check=True)


def test_stage_required_renderer_creates_acceptance_cache_namespace_in_job(
    tmp_path: Path,
) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work", direct_policy=False)
    staging = fixture.runspec.payload.database.staging
    assert staging is not None
    intent = render_phase_submission_intent(
        fixture.runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="a" * 64,
    )
    script = intent.actions[0].script_body

    # Host-side anchor creates the cache root so the bootstrap mount source exists.
    assert f"mkdir -p --mode=0700 -- {staging.cache_root}" in script
    # The container bootstrap creates the effective-user namespace (0700) before placement.
    assert f"/run/bspp-acceptance-cache-root/users/{staging.unix_user}" in script
    assert "--mode=0700" in script
    assert script.index("mkdir -p --mode=0700") < script.index("place-database")
    assert script.index("/run/bspp-acceptance-cache-root/users") < script.index("place-database")
    # The bootstrap mounts the cache root read-write at a target outside the
    # protected /run/bspp/database namespace.
    assert f"{staging.cache_root}:/run/bspp-acceptance-cache-root" in script
    assert "/run/bspp-acceptance-cache-root:ro" not in script
    # stage-required keeps the bootstrap fatal: no fallback exists, so the
    # rendered bootstrap must not be best-effort (|| true).
    assert f"mkdir -p --mode=0700 -- {staging.cache_root} || true" not in script
    assert f"/run/bspp-acceptance-cache-root/users/{staging.unix_user} || true" not in script
    subprocess.run(("bash", "-n"), input=script, text=True, check=True)


def test_descriptor_pipeline_preserves_legacy_text_and_renders_exact_read_only_file() -> None:
    carry = PhaseContainerMountDescriptor(
        source="/host/source/member.a3m",
        target="/run/bspp-carry/source/member.a3m",
        source_kind="file",
        read_only=True,
        origin="carry-source",
    )
    rendered = _container_mounts(
        declared=("/authored:/logical",),
        extra=(("/extra", "/extra"),),
        required=(("/runspec.json", "/runspec.json"), ("/action", "/action"), ("/source", "/source")),
        carry=(carry,),
    )
    assert rendered == (
        "/authored:/logical,/extra:/extra,/runspec.json:/runspec.json,"
        "/action:/action,/source:/source,"
        "/host/source/member.a3m:/run/bspp-carry/source/member.a3m:ro"
    )


@pytest.mark.parametrize(
    "declared",
    [
        ("/host:/run/bspp-carry/private",),
        ("/host:/target:ro",),
        ("relative:/target",),
    ],
)
def test_descriptor_pipeline_rejects_private_shadow_mode_suffix_and_relative_paths(
    declared: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        _container_mounts(
            declared=declared,
            extra=(),
            required=(("/runspec.json", "/runspec.json"), ("/action", "/action"), ("/source", "/source")),
        )


@pytest.mark.parametrize(
    ("declared", "protected"),
    [
        (("/attacker:/usr/local/bin",), ("/usr/local/bin",)),
        (("/attacker:/databases",), ("/databases",)),
        (("/attacker:/usr/bin/tar",), ("/usr/bin/tar",)),
    ],
)
def test_descriptor_pipeline_rejects_mounts_shadowing_protected_runtime_targets(
    declared: tuple[str, ...],
    protected: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="shadows protected runtime target"):
        _container_mounts(
            declared=declared,
            extra=(),
            required=(("/runspec.json", "/runspec.json"), ("/action", "/action"), ("/source", "/source")),
            protected=protected,
        )


def test_descriptor_pipeline_allows_identity_mount_for_protected_database() -> None:
    rendered = _container_mounts(
        declared=("/databases:/databases",),
        extra=(),
        required=(("/runspec.json", "/runspec.json"), ("/action", "/action"), ("/source", "/source")),
        protected=("/databases",),
    )

    assert rendered.startswith("/databases:/databases,")


@pytest.mark.parametrize(
    "database_namespace",
    [
        "/database/source",
        "/cache/users/alice",
        "/cache/users/alice/replicas/manifest-digest",
        "/run/bspp/database/selected",
    ],
    ids=["source", "user-cache", "replica", "selected"],
)
@pytest.mark.parametrize("relation", ["exact", "ancestor", "nested"])
@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize("origin", ["authored", "cluster-extra"])
def test_database_namespaces_reject_authored_and_profile_mount_source_or_destination_overlap(
    database_namespace: str,
    relation: str,
    side: str,
    origin: str,
) -> None:
    protected = Path(database_namespace)
    overlap = {
        "exact": protected,
        "ancestor": protected.parent,
        "nested": protected / "nested",
    }[relation]
    source = str(overlap) if side == "source" else "/safe/database-mount-source"
    target = str(overlap) if side == "destination" else "/safe/database-mount-target"
    declared = (f"{source}:{target}",) if origin == "authored" else ()
    extra = ((source, target),) if origin == "cluster-extra" else ()

    with pytest.raises(ValueError, match="overlaps protected database namespace"):
        _container_mounts(
            declared=declared,
            extra=extra,
            required=(("/runspec.json", "/runspec.json"), ("/action", "/action"), ("/source", "/source")),
            identity_protected=(database_namespace,),
        )


@pytest.mark.parametrize("relation", ["exact", "ancestor", "nested"])
@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize("origin", ["carry-source", "required"])
def test_database_namespaces_reject_carry_and_required_mount_endpoint_overlap(
    relation: str,
    side: str,
    origin: str,
) -> None:
    database_namespace = Path("/database/source")
    overlap = {
        "exact": database_namespace,
        "ancestor": database_namespace.parent,
        "nested": database_namespace / "member",
    }[relation]
    source = str(overlap) if side == "source" else "/safe/mount-source"
    target = str(overlap) if side == "destination" else "/safe/mount-target"
    carry = (
        (
            PhaseContainerMountDescriptor(
                source=source,
                target=target,
                source_kind="file",
                read_only=True,
                origin="carry-source",
            ),
        )
        if origin == "carry-source"
        else ()
    )
    required = (
        (source, target) if origin == "required" else ("/runspec.json", "/runspec.json"),
        ("/action", "/action"),
        ("/source", "/source"),
    )

    with pytest.raises(ValueError, match="overlaps protected database namespace"):
        _container_mounts(
            declared=(),
            extra=(),
            required=required,
            carry=carry,
            identity_protected=(str(database_namespace),),
        )


@pytest.mark.parametrize(
    "descriptor",
    [
        PhaseContainerMountDescriptor(
            source="/attempt/work",
            target="/usr/local/bin",
            source_kind="directory",
            read_only=False,
            origin="carry-workspace",
        ),
        PhaseContainerMountDescriptor(
            source="/attempt/work",
            target="/databases",
            source_kind="directory",
            read_only=False,
            origin="carry-workspace",
        ),
    ],
)
def test_descriptor_pipeline_rejects_generated_workspace_over_protected_runtime_targets(
    descriptor: PhaseContainerMountDescriptor,
) -> None:
    expected = (
        "overlaps protected database namespace"
        if descriptor.target == "/databases"
        else "shadows protected runtime target"
    )
    with pytest.raises(ValueError, match=expected):
        _container_mounts(
            declared=(),
            extra=(),
            required=(("/runspec.json", "/runspec.json"), ("/action", "/action"), ("/source", "/source")),
            carry=(descriptor,),
            protected=("/usr/local/bin",),
            identity_protected=("/databases",),
            protected_owners=((descriptor.target, descriptor),),
        )


def test_descriptor_pipeline_rejects_generated_source_bundle_over_executables() -> None:
    with pytest.raises(ValueError, match="shadows protected runtime target"):
        _container_mounts(
            declared=(),
            extra=(),
            required=(
                ("/runspec.json", "/runspec.json"),
                ("/action", "/action"),
                ("/usr/local/bin", "/usr/local/bin"),
            ),
            protected=("/usr/local/bin",),
        )


@pytest.mark.parametrize(
    ("logical_root_kind", "expected"),
    [
        ("entrypoint", "shadows protected runtime target"),
        ("database", "container mount overlaps protected database namespace"),
    ],
)
def test_carried_renderer_rejects_profile_workspace_roots_over_runtime_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    logical_root_kind: str,
    expected: str,
) -> None:
    store, phase_run_id = _failed_attested_authority(tmp_path, monkeypatch)
    baseline = _carried_retry_payload(store, phase_run_id)
    baseline_record = baseline.carry_forward_record
    assert baseline_record is not None
    successor = baseline.successor_phase_runspec
    action = successor.payload.actions[0]
    logical_root = "/usr/local/bin" if logical_root_kind == "entrypoint" else action.payload.site.database_root
    malicious_site = action.payload.site.model_copy(update={"project_logs_root": logical_root})
    work_plan = store.validate(phase_run_id).phase_plan.payload.work_plan
    chunk = next(item for item in work_plan.chunks if item.name == action.payload.chunk_name)
    records_by_ordinal = {item.source_ordinal: item for item in work_plan.input.records}
    malicious_execution = plan_preprocessing_chunk_execution(
        chunk=chunk,
        records=tuple(records_by_ordinal[ordinal] for ordinal in chunk.record_ordinals),
        expected_a3m_members=tuple(item.member_name for item in action.payload.expected_a3ms),
        scientific=action.payload.scientific,
        site=malicious_site,
        runtime=action.payload.runtime,
    )
    malicious_action = replace(action, payload=malicious_execution)
    malicious_successor = replace(
        successor,
        carry_forward=None,
        payload=replace(successor.payload, actions=(malicious_action,)),
    )
    request = AttemptCarryForwardRequest(
        source_attempt_id=baseline_record.source_attempt_id,
        content=tuple(
            AttemptCarryForwardSelection(
                member_name=item.member_name,
                size_bytes=item.size_bytes,
                sha256=item.sha256,
            )
            for item in baseline_record.content
        ),
    )
    record, carried_successor = derive_attempt_carry_forward(
        authority=store.validate(phase_run_id),
        successor_runspec=malicious_successor,
        target_attempt_ordinal=baseline.successor_attempt.ordinal,
        request=request,
        declared_at=malicious_successor.materialized_at,
    )

    with pytest.raises(ValueError, match=expected):
        render_phase_submission_intent(
            carried_successor,
            phase_runspec_location=baseline.successor_attempt.phase_runspec_location,
            phase_runspec_document_sha256="a" * 64,
            carry_forward_record=record,
            carry_forward_document_sha256="b" * 64,
        )


def test_descriptor_pipeline_rejects_ambiguous_directory_overlap_and_source_reuse() -> None:
    required = (("/runspec.json", "/runspec.json"), ("/action", "/action"), ("/source", "/source"))
    with pytest.raises(ValueError, match="ambiguous nested directory"):
        _container_mounts(
            declared=("/host-a:/logical", "/host-b:/logical/child"),
            extra=(),
            required=required,
        )
    with pytest.raises(ValueError, match="source reuse"):
        _container_mounts(
            declared=("/same:/logical-a", "/same:/logical-b"),
            extra=(),
            required=required,
        )


def test_carry_bootstrap_creates_exact_sentinel_once_and_rejects_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, phase_run_id = _failed_attested_authority(tmp_path, monkeypatch)
    record = _carried_retry_payload(store, phase_run_id).carry_forward_record
    assert record is not None
    action_root = Path(record.workspace.workspace_root).parent
    action_root.mkdir(parents=True)
    submission_id = "phase-submission-" + "8" * 64
    script = "set -euo pipefail\n" + "\n".join(_carry_bootstrap_lines(record, submission_id, str(action_root)))

    completed = subprocess.run(("bash", "-c", script), check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    sentinel = Path(record.workspace.identity_sentinel_path)
    assert json.loads(sentinel.read_text()) == {
        "schema_version": 1,
        "phase_run_id": record.phase_run_id,
        "attempt_id": record.target_attempt_id,
        "action_id": record.workspace.target_action_id,
        "attempt_carry_forward_id": record.attempt_carry_forward_id,
        "attempt_carry_forward_digest": record.digest,
        "phase_submission_id": submission_id,
        "workspace_mapping_digest": record.workspace.digest,
    }
    assert all(Path(item.physical_root).is_dir() for item in record.workspace.roots)

    repeated = subprocess.run(("bash", "-c", script), check=False, capture_output=True, text=True)
    assert repeated.returncode == 127
    assert "carry workspace already exists" in repeated.stderr


def test_carry_bootstrap_rejects_symlink_action_root_before_workspace_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, phase_run_id = _failed_attested_authority(tmp_path, monkeypatch)
    record = _carried_retry_payload(store, phase_run_id).carry_forward_record
    assert record is not None
    action_root = Path(record.workspace.workspace_root).parent
    action_root.parent.mkdir(parents=True)
    redirected = tmp_path / "redirected-action"
    redirected.mkdir()
    action_root.symlink_to(redirected, target_is_directory=True)
    script = "set -euo pipefail\n" + "\n".join(
        _carry_bootstrap_lines(record, "phase-submission-" + "8" * 64, str(action_root))
    )

    completed = subprocess.run(("bash", "-c", script), check=False, capture_output=True, text=True)

    assert completed.returncode == 127
    assert "action root escapes through a symlinked component" in completed.stderr
    assert not Path(record.workspace.workspace_root).exists()


def test_carry_bootstrap_rejects_symlinked_action_root_ancestor_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, phase_run_id = _failed_attested_authority(tmp_path, monkeypatch)
    record = _carried_retry_payload(store, phase_run_id).carry_forward_record
    assert record is not None
    action_root = Path(record.workspace.workspace_root).parent
    output_root = next(parent for parent in action_root.parents if parent.name == "output")
    redirected = tmp_path / "redirected-output"
    output_root.rename(redirected)
    output_root.symlink_to(redirected, target_is_directory=True)
    script = "set -euo pipefail\n" + "\n".join(
        _carry_bootstrap_lines(record, "phase-submission-" + "8" * 64, str(action_root))
    )

    completed = subprocess.run(("bash", "-c", script), check=False, capture_output=True, text=True)

    assert completed.returncode == 127
    assert "action root escapes through a symlinked component" in completed.stderr
    assert not Path(record.workspace.workspace_root).exists()
    assert not Path(record.workspace.identity_sentinel_path).exists()
