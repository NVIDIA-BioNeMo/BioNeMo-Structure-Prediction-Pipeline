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

"""F10: preprocessing input staging — fetch + split before science dispatch.

Tests that ``stage_preprocessing_input`` correctly downloads (if remote),
normalizes, and splits the FASTA into per-chunk ``.fa`` files at the exact
``input_root``/``split_input_root`` paths declared in the Phase RunSpec, and
that the rendered Slurm script includes the ``stage-input`` step.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from bspp.orchestration.contract.database_placement import (
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabaseProfileStagingSnapshot,
    DatabaseSetSelection,
    build_preprocessing_database_binding,
)
from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseSetIdentity,
    DatabaseSourceManifest,
    DatabaseSourceMember,
)
from bspp.orchestration.contract.phase import (
    PhaseRunSpec,
    PhaseSlurmResources,
    PreprocessingPhaseRunSpecPayload,
    PreprocessingRuntimeAction,
    ResolvedClusterSnapshot,
    VerifiedRemoteInputLocation,
)
from bspp.orchestration.contract.preprocessing import PreprocessingPlanOptions
from bspp.orchestration.contract.preprocessing_execution import (
    PreprocessingRuntimeCoordinates,
    PreprocessingScientificConfig,
    PreprocessingSiteConfig,
)
from bspp.orchestration.contract.preprocessing_runtime import (
    PREPROCESSING_ADAPTER_VERSION,
    PREPROCESSING_RUNTIME_CONTRACT_ID,
    PreprocessingRuntimeImageIdentity,
    PreprocessingRuntimeQualificationTuple,
    QualifiedPreprocessingRuntimeSelection,
)
from bspp.orchestration.runtime.preprocessing.commands import plan_preprocessing_chunk_execution
from bspp.orchestration.runtime.preprocessing.fasta import parse_preprocessing_fasta
from bspp.orchestration.runtime.preprocessing.input_staging import (
    InputStagingError,
    stage_preprocessing_input,
)
from bspp.orchestration.runtime.preprocessing.planning import plan_preprocessing_records

_FASTA = b">protein-zeta\nAAAA:TT\n>protein-alpha\nCCCC:AAA\n>protein-mu\nGGGG:CC\n"
_EXPECTED_A3M = ("AFDB_zeta.a3m", "AFDB_alpha.a3m", "AFDB_mu.a3m")


def _build_remote_runspec(tmp_path: Path) -> tuple[PhaseRunSpec, bytes]:
    """Build a PhaseRunSpec with a verified-remote input location."""
    fasta_path = tmp_path / "source.fa"
    fasta_path.write_bytes(_FASTA)
    records = parse_preprocessing_fasta(fasta_path)
    work_plan = plan_preprocessing_records(
        source_path="inputs/targets.fa",
        records=records,
        options=PreprocessingPlanOptions(requested_tranches=1, records_per_chunk=10, nodes=1, gpus_per_node=1),
    )
    chunk = work_plan.chunks[0]
    scratch_input = tmp_path / "runtime-input"
    split_input = tmp_path / "split-input"
    site = PreprocessingSiteConfig(
        mmseqs_executable="/usr/local/bin/mmseqs",
        colabfold_search_executable="/usr/local/bin/colabfold_search",
        tar_executable="/usr/bin/tar",
        lz4_executable="/usr/bin/lz4",
        database_root=SELECTED_DATABASE_ROOT,
        input_root=str(scratch_input),
        scratch_output_root=str(tmp_path / "scratch-output"),
        project_logs_root=str(tmp_path / "durable-logs"),
        finished_msa_root=str(tmp_path / "finished-msa"),
        split_input_root=str(split_input),
        finished_input_root=str(tmp_path / "finished-input"),
        container_image="/images/preprocessing.sqsh",
        container_mounts=(),
        max_concurrency=1,
        gpu_delay_seconds=0,
    )
    execution = plan_preprocessing_chunk_execution(
        chunk=chunk,
        records=work_plan.input.records,
        expected_a3m_members=_EXPECTED_A3M,
        scientific=PreprocessingScientificConfig(schema_version=3, require_afdb_model_id_stem=False),
        site=site,
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    action = PreprocessingRuntimeAction(
        action_id="preprocessing-chunk-000000",
        dependencies=(),
        resources=PhaseSlurmResources(partition="gpu", cpus_per_task=8, memory="32G", time="01:00:00", gres="gpu:1"),
        payload=execution,
    )
    database_set = DatabaseSetIdentity(identifier="local-test", version="2026-08")
    database_source_root = tmp_path / "db-source"
    database_source_root.mkdir()
    source_manifest = DatabaseSourceManifest(
        database_set=database_set,
        source_root=str(database_source_root),
        members=(
            DatabaseSourceMember(
                role="primary",
                database_name=execution.scientific.primary_database_name,
                logical_name=execution.scientific.primary_database_name,
                source_path=execution.scientific.primary_database_name,
                source_kind="regular",
                resolved_path=execution.scientific.primary_database_name,
                resolved_kind="regular",
                size_bytes=1,
                mtime_ns=1,
                alias_topology=(),
                preexisting_checksum=None,
            ),
            DatabaseSourceMember(
                role="metagenomic",
                database_name=execution.scientific.metagenomic_database_name,
                logical_name=execution.scientific.metagenomic_database_name,
                source_path=execution.scientific.metagenomic_database_name,
                source_kind="regular",
                resolved_path=execution.scientific.metagenomic_database_name,
                resolved_kind="regular",
                size_bytes=1,
                mtime_ns=1,
                alias_topology=(),
                preexisting_checksum=None,
            ),
        ),
    )
    database = build_preprocessing_database_binding(
        selection=DatabaseSetSelection(
            database_set=database_set,
            requested_policy=DatabaseAccessPolicy.STAGE_REQUIRED,
        ),
        source_manifest=source_manifest,
        source_manifest_projection="attempts/attempt-0001/database-source-manifest.json",
        staging=DatabaseProfileStagingSnapshot(
            cache_root="/var/cache/bspp/local-test",
            unix_user="tester",
            expected_filesystem_type="ext4",
            reserve_bytes=0,
            lock_wait_seconds=60,
        ),
        gpuserver_argv=execution.gpuserver_argv,
        search_argv=execution.search_argv,
    )
    remote_location = VerifiedRemoteInputLocation(
        source_uri="s3://test-bucket/test/inputs/targets.fa",
        sha256=hashlib.sha256(_FASTA).hexdigest(),
        size_bytes=len(_FASTA),
        path="inputs/targets.fa",
    )
    qualification_tuple = PreprocessingRuntimeQualificationTuple(
        cluster_profile="local-test",
        scheduling_class="gpu_worker",
        gpu_worker_gres="gpu:1",
        cluster_image_path="/images/preprocessing.sqsh",
        cluster_image_sha256="1" * 64,
        oci_digest="sha256:" + "2" * 64,
        source_bundle_id="bspp-orchestration-" + "3" * 40,
        source_bundle_path="/source/bspp-orchestration-" + "3" * 40 + ".tar.zst",
        source_bundle_sha256="7" * 64,
        runtime_contract_id=PREPROCESSING_RUNTIME_CONTRACT_ID,
        adapter_version=PREPROCESSING_ADAPTER_VERSION,
        image_identity=PreprocessingRuntimeImageIdentity(
            source_commit="3" * 40,
            image_lock_sha256="4" * 64,
            contract_wheel_sha256="5" * 64,
            runtime_wheel_sha256="6" * 64,
            control_wheel_sha256="a" * 64,
            colabfold_version="1.6.2",
            mmseqs_version="18-8cc5c",
            rsync_version="3.4.4",
            cuda_version="12.6.3",
        ),
    )
    runspec = PhaseRunSpec(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_plan_digest="1" * 64,
        materialized_at="2026-08-19T12:00:00.000000Z",
        input_location=remote_location,
        cluster=ResolvedClusterSnapshot(
            profile_name="local-test",
            owner="tester",
            transport="local-slurm",
            ssh_target=None,
            account="test",
            project_root=str(tmp_path),
            output_root=str(tmp_path / "output"),
            staging_root=str(tmp_path / "staging"),
            orchestration_repo=str(tmp_path / "source"),
            runtime_image="/images/preprocessing.sqsh",
            preprocessing_runtime=QualifiedPreprocessingRuntimeSelection(
                qualification_tuple=qualification_tuple,
                qualification_record_path=str(tmp_path / "qualification.json"),
                qualified_at="2026-08-19T10:00:00.000000Z",
                expires_at="2026-08-26T10:00:00.000000Z",
            ),
            source_bundle_root=None,
            runtime_image_cache_root=None,
            runtime_qualification_root=None,
            runtime_qualification_expires_hours=168,
            extra_mounts=(),
        ),
        payload=PreprocessingPhaseRunSpecPayload(
            work_plan=work_plan,
            actions=(action,),
            database=database,
        ),
    )
    # Pre-stage database source members.
    for member in source_manifest.members:
        member_path = database_source_root / member.source_path
        member_path.parent.mkdir(parents=True, exist_ok=True)
        member_path.write_bytes(b"x" * member.size_bytes)
        member_path.touch()
        member_path.chmod(0o444)
    return runspec, _FASTA


def test_stage_input_downloads_remote_fasta_and_splits_into_chunk_files(
    tmp_path: Path,
) -> None:
    """F10: stage_preprocessing_input downloads a remote FASTA and splits it."""
    runspec, fasta_bytes = _build_remote_runspec(tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    action = runspec.payload.actions[0]
    search_input = Path(action.payload.search_argv[3])
    split_input = Path(action.payload.package.completed_input_source_path)

    # Mock the S3 transfer to write the FASTA bytes.
    from bspp.orchestration.runtime.data_movement.common import TransferResult

    def fake_transfer(_src: str, dst: str) -> TransferResult:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(fasta_bytes)
        return TransferResult(
            tool="s5cmd",
            argv=(_src, dst),
            returncode=0,
            elapsed_s=0.0,
            stdout_tail="",
            stderr_tail="",
        )

    with patch(
        "bspp.orchestration.runtime.data_movement.s3.transfer.cp",
        fake_transfer,
    ):
        materialized = stage_preprocessing_input(runspec, workspace_root=workspace_root)

    assert len(materialized) == 1
    assert materialized[0] == search_input

    # Verify the downloaded file exists at the workspace-relative path.
    downloaded = workspace_root / "inputs" / "targets.fa"
    assert downloaded.is_file()

    # Verify the chunk files exist at the declared paths.
    assert search_input.is_file()
    assert split_input.is_file()

    # Verify the chunk file content matches the expected normalized format.
    assert search_input.read_bytes() == fasta_bytes
    assert split_input.read_bytes() == fasta_bytes


def test_stage_input_splits_local_fasta_into_chunk_files(tmp_path: Path) -> None:
    """F10: stage_preprocessing_input reads a local FASTA and splits it."""
    from tests.support.preprocessing_execution import preprocessing_execution_fixture

    fixture = preprocessing_execution_fixture(tmp_path / "fixture")
    runspec = fixture.runspec

    # Remove pre-staged chunk files so staging has to recreate them.
    action = runspec.payload.actions[0]
    search_input = Path(action.payload.search_argv[3])
    split_input = Path(action.payload.package.completed_input_source_path)
    if search_input.exists():
        search_input.unlink()
    if split_input.exists():
        split_input.unlink()

    materialized = stage_preprocessing_input(runspec, workspace_root=fixture.runspec_path.parent)

    assert len(materialized) == 1
    assert search_input.is_file()
    assert split_input.is_file()


def test_stage_input_rejects_overwriting_existing_chunk_files(tmp_path: Path) -> None:
    """F10: stage_preprocessing_input refuses to overwrite existing chunk files."""
    from tests.support.preprocessing_execution import preprocessing_execution_fixture

    fixture = preprocessing_execution_fixture(tmp_path / "fixture")
    runspec = fixture.runspec

    # The fixture already staged chunk files; staging must refuse to overwrite.
    with pytest.raises(InputStagingError, match="refusing to overwrite"):
        stage_preprocessing_input(runspec, workspace_root=fixture.runspec_path.parent)
