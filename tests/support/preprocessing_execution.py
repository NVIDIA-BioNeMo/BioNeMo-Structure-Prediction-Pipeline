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

"""Reusable fake-kernel fixture for preprocessing execution/finalization tests."""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from bspp.orchestration.contract.database_placement import (
    DATABASE_SOURCE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
    DatabaseProfileStagingSnapshot,
    DatabaseSetSelection,
    build_preprocessing_database_binding,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabasePlacementResult,
    DatabaseSourceMountFacts,
    DatabaseSourceObservation,
    canonical_database_placement_result_bytes,
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
    VerifiedLocalInputLocation,
)
from bspp.orchestration.contract.phase_carry_forward import AttemptCarryForwardRecord
from bspp.orchestration.contract.preprocessing import PreprocessingPlanOptions
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingChunkActionEvidence,
    preprocessing_chunk_action_evidence_from_mapping,
)
from bspp.orchestration.contract.preprocessing_execution import (
    PreprocessingRuntimeCoordinates,
    PreprocessingScientificConfig,
    PreprocessingSiteConfig,
    preprocessing_chunk_execution_plan_from_mapping,
)
from bspp.orchestration.contract.preprocessing_runtime import (
    PREPROCESSING_ADAPTER_VERSION,
    PREPROCESSING_RUNTIME_CONTRACT_ID,
    PreprocessingRuntimeImageIdentity,
    PreprocessingRuntimeQualificationTuple,
    QualifiedPreprocessingRuntimeSelection,
)
from bspp.orchestration.runtime.preprocessing._database_placement_paths import (
    PRODUCTION_DATABASE_PLACEMENT_PATHS,
    DatabasePlacementPaths,
)
from bspp.orchestration.runtime.preprocessing.commands import plan_preprocessing_chunk_execution
from bspp.orchestration.runtime.preprocessing.planning import plan_preprocessing_fasta

if TYPE_CHECKING:
    from bspp.orchestration.runtime.preprocessing._database_replica_services import (
        StagedDatabasePlacementServices,
    )
    from bspp.orchestration.runtime.preprocessing.database_placement import DatabasePlacementCommandResult
    from bspp.orchestration.runtime.preprocessing.execution import ScientificKernelLauncher


@dataclass(frozen=True)
class ExecutionResult:
    """Small CLI-shaped result for direct private-coordinator tests."""

    exit_code: int
    output: str
    exception: BaseException | None = None


@dataclass(frozen=True)
class LocalExecutionFixture:
    runspec: PhaseRunSpec
    runspec_path: Path
    evidence_path: Path
    invocation_path: Path
    database_placement_result_path: Path
    database_placement_failure_path: Path
    database_source_root: Path
    database_placement_paths: DatabasePlacementPaths = PRODUCTION_DATABASE_PLACEMENT_PATHS

    @property
    def action(self) -> PreprocessingRuntimeAction:
        return self.runspec.payload.actions[0]

    def place_database(
        self,
        runspec: PhaseRunSpec,
        *,
        action_id: str,
        source_manifest_path: Path,
        result_path: Path,
        failure_path: Path | None,
        staged_services: StagedDatabasePlacementServices | None = None,
    ) -> DatabasePlacementCommandResult:
        """Invoke the private placement coordinator with fixture path authority."""
        from bspp.orchestration.runtime.preprocessing.database_placement import (
            _place_database,
            _production_staged_database_placement_services,
        )

        return _place_database(
            runspec,
            action_id=action_id,
            source_manifest_path=source_manifest_path,
            result_path=result_path,
            failure_path=failure_path,
            paths=self.database_placement_paths,
            staged_services=(
                staged_services if staged_services is not None else _production_staged_database_placement_services()
            ),
        )

    def execute_preprocessing_chunk_action(
        self,
        *,
        action_id: str | None = None,
        placement_process_status: int = 0,
        carry_forward_record: AttemptCarryForwardRecord | None = None,
        phase_submission_id: str | None = None,
        scientific_kernel_launcher: ScientificKernelLauncher | None = None,
    ) -> PreprocessingChunkActionEvidence:
        """Invoke the private execution coordinator with fixture path authority."""
        from bspp.orchestration.runtime.preprocessing.execution import (
            PRODUCTION_PREPROCESSING_ACTION_EVIDENCE_STORE,
            PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER,
            _execute_preprocessing_chunk_action,
        )

        observed_at = datetime.fromisoformat(self.runspec.materialized_at.removesuffix("Z") + "+00:00")
        return _execute_preprocessing_chunk_action(
            self.runspec,
            action_id=action_id or self.action.action_id,
            evidence_path=self.evidence_path,
            database_placement_result_path=self.database_placement_result_path,
            database_placement_failure_path=self.database_placement_failure_path,
            placement_process_status=placement_process_status,
            carry_forward_record=carry_forward_record,
            phase_submission_id=phase_submission_id,
            clock=lambda: observed_at,
            sleeper=lambda _seconds: time.sleep(0.05),
            database_placement_paths=self.database_placement_paths,
            scientific_kernel_launcher=(
                scientific_kernel_launcher
                if scientific_kernel_launcher is not None
                else PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER
            ),
            evidence_store=PRODUCTION_PREPROCESSING_ACTION_EVIDENCE_STORE,
        )


def preprocessing_execution_fixture(
    root: Path,
    *,
    failed_tool: str | None = None,
    direct_policy: bool = True,
    expected_a3m_members: tuple[str, ...] = ("AFDB_zeta.a3m", "AFDB_alpha.a3m", "AFDB_mu.a3m"),
    scientific: PreprocessingScientificConfig | None = None,
    fasta_bytes: bytes | None = None,
    legacy_paired: bool = False,
) -> LocalExecutionFixture:
    root.mkdir(parents=True, exist_ok=True)
    bin_dir = root / "bin"
    bin_dir.mkdir()
    mmseqs = bin_dir / "mmseqs"
    search = bin_dir / "colabfold_search"
    failing = bin_dir / "fail-tool"
    _write_executable(mmseqs, _fake_gpuserver_source())
    _write_executable(search, _fake_search_source())
    _write_executable(failing, "#!/bin/sh\nexit 19\n")

    source = root / "input.fa"
    source_bytes = preprocessing_chunk_bytes() if fasta_bytes is None else fasta_bytes
    source.write_bytes(source_bytes)
    work_plan = plan_preprocessing_fasta(
        source,
        PreprocessingPlanOptions(
            requested_tranches=1, records_per_chunk=max(3, len(expected_a3m_members)), nodes=1, gpus_per_node=1
        ),
    )
    scratch_input = root / "runtime-input"
    split_input = root / "split-input"
    chunk = work_plan.chunks[0]
    staged_input = scratch_input / "n0g0" / chunk.name
    split_path = split_input / chunk.name
    staged_input.parent.mkdir(parents=True)
    split_path.parent.mkdir(parents=True)
    staged_input.write_bytes(source_bytes)
    split_path.write_bytes(source_bytes)

    site = PreprocessingSiteConfig(
        mmseqs_executable=str(mmseqs),
        colabfold_search_executable=str(search),
        tar_executable=str(failing) if failed_tool == "tar" else "/usr/bin/tar",
        lz4_executable=str(failing) if failed_tool == "lz4" else "/usr/bin/lz4",
        database_root=SELECTED_DATABASE_ROOT,
        input_root=str(scratch_input),
        scratch_output_root=str(root / "scratch-output"),
        project_logs_root=str(root / "durable-logs"),
        finished_msa_root=str(root / "finished-msa"),
        split_input_root=str(split_input),
        finished_input_root=str(root / "finished-input"),
        container_image="/images/preprocessing.sqsh",
        container_mounts=(),
        max_concurrency=1,
        gpu_delay_seconds=0,
    )
    execution = plan_preprocessing_chunk_execution(
        chunk=chunk,
        records=work_plan.input.records,
        expected_a3m_members=expected_a3m_members,
        scientific=scientific
        or PreprocessingScientificConfig(schema_version=2 if legacy_paired else 3, require_afdb_model_id_stem=False),
        site=site,
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    if legacy_paired:
        # Exercise the public replay seam; fresh construction remains strict.
        sealed = execution.to_mapping()
        argv = list(execution.search_argv)
        argv[argv.index("--pair-mode") + 1] = "paired"
        sealed["search_argv"] = argv
        execution = preprocessing_chunk_execution_plan_from_mapping(sealed)
    action = PreprocessingRuntimeAction(
        action_id="preprocessing-chunk-000000",
        dependencies=(),
        resources=PhaseSlurmResources(
            partition="gpu",
            cpus_per_task=8,
            memory="32G",
            time="01:00:00",
            gres="gpu:1",
        ),
        payload=execution,
    )
    database_set = DatabaseSetIdentity(identifier="local-test", version="2026-08")
    database_source_root = root.parent / f"{root.name}-database-source"
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
            requested_policy=(DatabaseAccessPolicy.DIRECT if direct_policy else DatabaseAccessPolicy.STAGE_REQUIRED),
        ),
        source_manifest=source_manifest,
        source_manifest_projection="attempts/attempt-0001/database-source-manifest.json",
        staging=(
            None
            if direct_policy
            else DatabaseProfileStagingSnapshot(
                cache_root="/var/cache/bspp/local-test",
                unix_user="tester",
                expected_filesystem_type="ext4",
                reserve_bytes=0,
                lock_wait_seconds=60,
            )
        ),
        gpuserver_argv=execution.gpuserver_argv,
        search_argv=execution.search_argv,
    )
    source_bytes = source.read_bytes()
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
        input_location=VerifiedLocalInputLocation(
            path=str(source),
            sha256=hashlib.sha256(source_bytes).hexdigest(),
            size_bytes=len(source_bytes),
        ),
        cluster=ResolvedClusterSnapshot(
            profile_name="local-test",
            owner="tester",
            transport="local-slurm",
            ssh_target=None,
            account="test",
            project_root=str(root),
            output_root=str(root / "output"),
            staging_root=str(root / "staging"),
            orchestration_repo=str(root / "source"),
            runtime_image="/images/preprocessing.sqsh",
            preprocessing_runtime=QualifiedPreprocessingRuntimeSelection(
                qualification_tuple=qualification_tuple,
                qualification_record_path=str(root / "qualification.json"),
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
    runspec_path = root / "phase-runspec.json"
    runspec_path.write_text(json.dumps(runspec.to_mapping(), indent=2, sort_keys=True) + "\n")
    for member in source_manifest.members:
        member_path = database_source_root / member.source_path
        member_path.parent.mkdir(parents=True, exist_ok=True)
        member_path.write_bytes(b"x" * member.size_bytes)
        member_path.touch()
        member_path.chmod(0o444)
        member_path_time = member.mtime_ns
        os.utime(member_path, ns=(member_path_time, member_path_time))
    device = database_source_root.stat().st_dev
    placement_result = DatabasePlacementResult(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_id=action.action_id,
        database_set=database_set,
        requested_policy=DatabaseAccessPolicy.DIRECT,
        source_manifest_sha256=database.source_manifest_sha256,
        branch_kind="direct-requested",
        outcome=DatabasePlacementOutcomeKind.DIRECT_REQUESTED,
        selected_container_root=SELECTED_DATABASE_ROOT,
        verification="metadata-verified",
        source_mount=DatabaseSourceMountFacts(
            mount_id=1,
            parent_mount_id=0,
            device_major=os.major(device),
            device_minor=os.minor(device),
            mount_root="/",
            mount_point=DATABASE_SOURCE_ROOT,
            filesystem_type="local-test",
            mount_source=str(database_source_root),
            mount_options=("ro",),
            super_options=("ro",),
            read_only=True,
        ),
        pre_science_observation=DatabaseSourceObservation(
            source_container_root=DATABASE_SOURCE_ROOT,
            source_manifest_sha256=database.source_manifest_sha256,
            members=source_manifest.members,
        ),
    )
    placement_result_path = root / "placement" / "result.json"
    placement_result_path.parent.mkdir()
    placement_result_path.write_bytes(canonical_database_placement_result_bytes(placement_result))
    placement_result_path.chmod(0o444)
    return LocalExecutionFixture(
        runspec=runspec,
        runspec_path=runspec_path,
        evidence_path=root / "evidence" / "action.json",
        invocation_path=root / "fake-invocations.jsonl",
        database_placement_result_path=placement_result_path,
        database_placement_failure_path=root / "placement" / "failure.json",
        database_source_root=database_source_root,
        database_placement_paths=replace(
            PRODUCTION_DATABASE_PLACEMENT_PATHS,
            source_root=database_source_root,
            selected_root=database_source_root,
        ),
    )


def configure_preprocessing_fakes(
    fixture: LocalExecutionFixture,
    monkeypatch: pytest.MonkeyPatch,
    *,
    search_mode: str = "success",
    gpuserver_exit: bool = False,
) -> None:
    _write_fixture_placement_result(fixture)
    monkeypatch.setenv("BSPP_FAKE_INVOCATIONS", str(fixture.invocation_path))
    monkeypatch.setenv(
        "BSPP_FAKE_A3MS",
        "|".join(item.member_name for item in fixture.action.payload.expected_a3ms),
    )
    monkeypatch.setenv("BSPP_FAKE_SEARCH_MODE", search_mode)
    monkeypatch.setenv("BSPP_FAKE_GPUSERVER_EXIT", "1" if gpuserver_exit else "0")
    monkeypatch.setenv("BSPP_FAKE_SYMLINK_TARGET", str(fixture.runspec_path.parent / "symlink-target.a3m"))


def _write_fixture_placement_result(fixture: LocalExecutionFixture) -> None:
    binding = fixture.runspec.payload.database
    device = fixture.database_source_root.stat().st_dev
    result = DatabasePlacementResult(
        phase_run_id=fixture.runspec.phase_run_id,
        attempt_id=fixture.runspec.attempt_id,
        phase_runspec_digest=fixture.runspec.digest,
        action_id=fixture.action.action_id,
        database_set=binding.database_set,
        requested_policy=DatabaseAccessPolicy.DIRECT,
        source_manifest_sha256=binding.source_manifest_sha256,
        branch_kind="direct-requested",
        outcome=DatabasePlacementOutcomeKind.DIRECT_REQUESTED,
        selected_container_root=SELECTED_DATABASE_ROOT,
        verification="metadata-verified",
        source_mount=DatabaseSourceMountFacts(
            mount_id=1,
            parent_mount_id=0,
            device_major=os.major(device),
            device_minor=os.minor(device),
            mount_root="/",
            mount_point=DATABASE_SOURCE_ROOT,
            filesystem_type="local-test",
            mount_source=str(fixture.database_source_root),
            mount_options=("ro",),
            super_options=("ro",),
            read_only=True,
        ),
        pre_science_observation=DatabaseSourceObservation(
            source_container_root=DATABASE_SOURCE_ROOT,
            source_manifest_sha256=binding.source_manifest_sha256,
            members=binding.source_manifest.members,
        ),
    )
    fixture.database_placement_result_path.parent.mkdir(parents=True, exist_ok=True)
    if fixture.database_placement_result_path.exists():
        fixture.database_placement_result_path.chmod(0o600)
    fixture.database_placement_result_path.write_bytes(canonical_database_placement_result_bytes(result))
    fixture.database_placement_result_path.chmod(0o444)


def skip_preprocessing_server_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compatibility no-op; direct fixture execution composes a bounded sleeper."""
    del monkeypatch


def invoke_preprocessing_execution(
    fixture: LocalExecutionFixture,
    *,
    action_id: str | None = None,
    placement_process_status: int = 0,
    extra_args: tuple[str, ...] = (),
    scientific_kernel_launcher: ScientificKernelLauncher | None = None,
) -> ExecutionResult:
    """Invoke fixture-owned execution without crossing the Click boundary."""
    from bspp.orchestration.runtime.preprocessing.carry_forward import (
        load_attempt_carry_forward_record,
    )
    from bspp.orchestration.runtime.preprocessing.execution import PreprocessingExecutionError

    carry_forward_record: AttemptCarryForwardRecord | None = None
    phase_submission_id: str | None = None
    if extra_args:
        if (
            len(extra_args) != 4
            or extra_args[0] != "--carry-forward-record"
            or extra_args[2] != "--phase-submission-id"
        ):
            raise ValueError(f"unsupported direct execution arguments: {extra_args!r}")
        carry_forward_record = load_attempt_carry_forward_record(Path(extra_args[1]))
        phase_submission_id = extra_args[3]
    try:
        evidence = fixture.execute_preprocessing_chunk_action(
            action_id=action_id,
            placement_process_status=placement_process_status,
            carry_forward_record=carry_forward_record,
            phase_submission_id=phase_submission_id,
            scientific_kernel_launcher=scientific_kernel_launcher,
        )
    except (OSError, TypeError, ValueError, PreprocessingExecutionError) as exc:
        return ExecutionResult(exit_code=1, output=f"Error: {exc}\n", exception=exc)
    return ExecutionResult(
        exit_code=0,
        output=json.dumps(evidence.to_mapping(), indent=2, sort_keys=True) + "\n",
    )


def load_failed_preprocessing_evidence(path: Path) -> PreprocessingChunkActionEvidence:
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(path.read_text()))
    assert evidence.outcome == "failed"
    assert evidence.error
    return evidence


def preprocessing_chunk_bytes() -> bytes:
    return b">protein-zeta\nAAAA:TT\n>protein-alpha\nCCCC:AAA\n>protein-mu\nGGGG:CC\n"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def _fake_gpuserver_source() -> str:
    return textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json
        import os
        import signal
        import sys
        import time
        from pathlib import Path

        evidence = Path(os.environ["BSPP_FAKE_INVOCATIONS"])
        def write(kind):
            with evidence.open("a") as handle:
                payload = {"kind": kind, "argv": sys.argv[1:], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}
                handle.write(json.dumps(payload) + "\\n")
        write("gpuserver")
        if os.environ.get("BSPP_FAKE_GPUSERVER_EXIT") == "1":
            raise SystemExit(9)
        def stop(_signum, _frame):
            write("gpuserver-stopped")
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while True:
            time.sleep(0.05)
        """
    )


def _fake_search_source() -> str:
    return textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json
        import os
        import sys
        from pathlib import Path

        evidence = Path(os.environ["BSPP_FAKE_INVOCATIONS"])
        with evidence.open("a") as handle:
            payload = {"kind": "search", "argv": sys.argv[1:], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}
            handle.write(json.dumps(payload) + "\\n")
        mode = os.environ.get("BSPP_FAKE_SEARCH_MODE", "success")
        if mode == "fail":
            print("search failure stdout", flush=True)
            print("search failure stderr", file=sys.stderr, flush=True)
            raise SystemExit(7)
        output = Path(sys.argv[5])
        output.mkdir(parents=True, exist_ok=True)
        members = os.environ["BSPP_FAKE_A3MS"].split("|")
        fasta_lines = Path(sys.argv[3]).read_text().splitlines()
        sequences = [line for line in fasta_lines if line and not line.startswith(">")]
        unique_chains = [dict.fromkeys(sequence.split(":")) for sequence in sequences]
        chains = [chain for unique in unique_chains for chain in unique]
        cardinalities = [
            sequence.split(":").count(chain)
            for sequence, unique in zip(sequences, unique_chains) for chain in unique
        ]
        selected = members[:-1] if mode == "missing" else list(members)
        for index, member in enumerate(selected):
            path = output / member
            if mode == "nonfile" and index == 0:
                path.mkdir()
            elif mode == "special" and index == 0:
                os.mkfifo(path)
            elif mode == "symlink" and index == 0:
                target = Path(os.environ["BSPP_FAKE_SYMLINK_TARGET"])
                target.write_text(">linked\\nAAAA\\n")
                path.symlink_to(target)
            elif mode == "empty" and index == 0:
                path.write_bytes(b"")
            elif mode == "malformed" and index == 0:
                path.write_text("not-an-a3m\\n")
            else:
                unique = unique_chains[index]
                lengths = ",".join(str(len(chain)) for chain in unique)
                counts = ",".join(str(sequences[index].split(":").count(chain)) for chain in unique)
                path.write_text(f"#{lengths}\\t{counts}\\n>{Path(member).stem}\\n{''.join(unique)}\\n")
        if mode == "extra":
            (output / "AFDB_unexpected.a3m").write_text("#4,2\\t1,1\\n>unexpected\\nAAAATT\\n")
        pair_mode = sys.argv[sys.argv.index("--pair-mode") + 1]
        if mode == "unexpected-numeric":
            (output / f"{len(members)}.a3m").write_text("#4\\t1\\n")
        raw_ids = range(len(members), len(chains)) if pair_mode == "paired" else ()
        for raw_id in raw_ids:
            path = output / f"{raw_id}.a3m"
            modeled = chains[raw_id]
            cardinality = cardinalities[raw_id]
            if mode == "placeholder-missing" and raw_id == len(members):
                continue
            if mode == "placeholder-empty" and raw_id == len(members):
                path.write_bytes(b"")
            elif mode == "placeholder-symlink" and raw_id == len(members):
                target = Path(os.environ["BSPP_FAKE_SYMLINK_TARGET"])
                target.write_text(f"#{len(modeled)}\\t{cardinality}\\n")
                path.symlink_to(target)
            elif mode == "placeholder-special" and raw_id == len(members):
                os.mkfifo(path)
            elif mode == "placeholder-malformed" and raw_id == len(members):
                path.write_text("not-a-placeholder\\n")
            elif mode == "placeholder-length-drift" and raw_id == len(members):
                path.write_text(f"#{len(modeled) + 1}\\t{cardinality}\\n")
            elif mode == "placeholder-name-drift" and raw_id == len(members):
                (output / f"drift-{raw_id}.a3m").write_text(f"#{len(modeled)}\\t{cardinality}\\n")
            else:
                path.write_text(f"#{len(modeled)}\\t{cardinality}\\n")
        print("search complete")
        """
    )


def preprocessing_invocations(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


__all__ = [
    "ExecutionResult",
    "LocalExecutionFixture",
    "configure_preprocessing_fakes",
    "invoke_preprocessing_execution",
    "load_failed_preprocessing_evidence",
    "preprocessing_chunk_bytes",
    "preprocessing_execution_fixture",
    "preprocessing_invocations",
    "skip_preprocessing_server_warmup",
]
