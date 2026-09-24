#!/opt/bspp/environment/bin/python
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

"""Self-contained local and scheduled smoke for the baked preprocessing image."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pwd
import subprocess
import tempfile
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
from bspp.orchestration.contract.preprocessing import PreprocessingPlanOptions
from bspp.orchestration.contract.preprocessing_action import (
    PREPROCESSING_COMMAND_ORDER,
    preprocessing_chunk_action_evidence_from_mapping,
)
from bspp.orchestration.contract.preprocessing_execution import (
    PreprocessingRuntimeCoordinates,
    PreprocessingScientificConfig,
    PreprocessingSiteConfig,
)
from bspp.orchestration.contract.preprocessing_runtime import (
    PREPROCESSING_ADAPTER_VERSION,
    PREPROCESSING_RUNTIME_COMMAND,
    PREPROCESSING_RUNTIME_CONTRACT_ID,
    PreprocessingRuntimeGpuEvidence,
    PreprocessingRuntimeImageEvidence,
    PreprocessingRuntimeImageIdentity,
    PreprocessingRuntimeQualificationRecord,
    PreprocessingRuntimeQualificationTuple,
    PreprocessingRuntimeSmokeEvidence,
    PreprocessingRuntimeSourceEvidence,
    PreprocessingRuntimeToolEvidence,
    QualifiedPreprocessingRuntimeSelection,
    normalize_rsync_version,
    preprocessing_runtime_image_identity_from_mapping,
    preprocessing_runtime_qualification_record_from_mapping,
    preprocessing_runtime_qualification_tuple_from_mapping,
    preprocessing_runtime_tool_evidence_from_mapping,
    preprocessing_runtime_tuple_id,
    validate_preprocessing_runtime_tool_evidence,
)
from bspp.orchestration.runtime.preprocessing.commands import plan_preprocessing_chunk_execution
from bspp.orchestration.runtime.preprocessing.execution import reconcile_preprocessing_chunk_action_evidence
from bspp.orchestration.runtime.preprocessing.planning import plan_preprocessing_fasta

_MANIFEST = Path("/opt/bspp/preprocessing-runtime-image.json")
_CUDA_COMPAT_DIR = Path("/usr/local/cuda-12.6/compat")
_CUDA_COMPAT_LIBRARY = _CUDA_COMPAT_DIR / "libcuda.so.1"
_LOCKED_CHARACTERIZATION_EXECUTABLES = (
    Path("/opt/bspp/environment/bin/python"),
    Path("/opt/bspp/environment/bin/timeout"),
    Path("/opt/bspp/environment/bin/sleep"),
    Path("/opt/bspp/environment/bin/tail"),
    Path("/usr/local/bin/mmseqs"),
    Path("/usr/local/bin/colabfold_search"),
    Path("/usr/bin/rsync"),
    Path("/usr/bin/s5cmd"),
    Path("/opt/bspp/bin/bspp-preprocessing-carry-characterization"),
    Path("/opt/bspp/bin/bspp-preprocessing-cuda-driver-probe"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-record", type=Path)
    args = parser.parse_args()
    _verify_characterization_composition()
    manifest_bytes = _MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    image_identity = preprocessing_runtime_image_identity_from_mapping(manifest)
    tools = _tool_observations(image_identity)
    with tempfile.TemporaryDirectory(prefix="bspp-preprocessing-smoke-") as temporary:
        root = Path(temporary)
        runspec, runspec_path, evidence_path = _fixture(root, image_identity)
        placement_result_path = _write_direct_placement_result(runspec, root)
        environment = dict(os.environ)
        environment.update(
            {
                "BSPP_FAKE_INVOCATIONS": str(root / "invocations.jsonl"),
                "BSPP_FAKE_A3MS": "AFDB_zeta.a3m|AFDB_alpha.a3m|AFDB_mu.a3m",
                "BSPP_FAKE_SEARCH_MODE": "success",
                "BSPP_FAKE_GPUSERVER_EXIT": "0",
            }
        )
        completed = subprocess.run(
            (
                *PREPROCESSING_RUNTIME_COMMAND,
                "--phase-runspec",
                str(runspec_path),
                "--action-id",
                runspec.payload.actions[0].action_id,
                "--write-evidence",
                str(evidence_path),
                "--database-placement-result",
                str(placement_result_path),
                "--placement-process-status",
                "0",
            ),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)
        evidence_bytes = evidence_path.read_bytes()
        evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(evidence_bytes))
        reconcile_preprocessing_chunk_action_evidence(runspec, evidence)
        if evidence.adapter_version != PREPROCESSING_ADAPTER_VERSION:
            raise RuntimeError("adapter version mismatch")
        if tuple(item.command_kind for item in evidence.command_outcomes) != PREPROCESSING_COMMAND_ORDER:
            raise RuntimeError("command order mismatch")
        if evidence.phase_runspec_digest != runspec.digest or evidence.outcome != "succeeded":
            raise RuntimeError("focused Runtime smoke evidence mismatch")
        durable_lz4 = Path(runspec.payload.actions[0].payload.package.durable_lz4_path)
        if evidence_path.stat().st_size == 0 or durable_lz4.stat().st_size == 0:
            raise RuntimeError("focused Runtime smoke produced an empty artifact")
        action_sha = hashlib.sha256(evidence_bytes).hexdigest()
        paired_row_sha = _verify_paired_row_identity(root, runspec.payload.actions[0].payload.search_argv)

    _reject_source_overlays()
    report = {
        "schema_version": 1,
        "runtime_command": list(PREPROCESSING_RUNTIME_COMMAND),
        "runtime_contract_id": PREPROCESSING_RUNTIME_CONTRACT_ID,
        "adapter_version": PREPROCESSING_ADAPTER_VERSION,
        "command_order": list(PREPROCESSING_COMMAND_ORDER),
        "action_evidence_sha256": action_sha,
        "synthetic_paired_row_identity_sha256": paired_row_sha,
        "tools": tools.to_mapping(),
        "image_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    if args.qualification_record is not None:
        _publish_qualification(args.qualification_record, report, image_identity)
    print(json.dumps(report, indent=2, sort_keys=True))


def _verify_characterization_composition() -> None:
    """Check the pinned loader policy and baked real-kernel helper composition."""
    try:
        compat_directory = _CUDA_COMPAT_DIR.resolve(strict=True)
        compat_target = _CUDA_COMPAT_LIBRARY.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("pinned CUDA compatibility composition is missing") from exc
    if not compat_directory.is_dir() or not os.access(compat_directory, os.R_OK | os.X_OK):
        raise RuntimeError("pinned CUDA compatibility directory is not readable")
    if not _CUDA_COMPAT_LIBRARY.is_symlink():
        raise RuntimeError("pinned CUDA compatibility soname is not a symlink")
    if not compat_target.is_file() or not os.access(compat_target, os.R_OK):
        raise RuntimeError("pinned CUDA compatibility target is not a readable regular file")
    if not compat_target.is_relative_to(compat_directory):
        raise RuntimeError("pinned CUDA compatibility target escapes its directory")
    loader_entries = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if not loader_entries or loader_entries[0] != str(_CUDA_COMPAT_DIR):
        raise RuntimeError("pinned CUDA compatibility directory is not first in LD_LIBRARY_PATH")
    for executable in _LOCKED_CHARACTERIZATION_EXECUTABLES:
        if not executable.is_file() or not os.access(executable, os.R_OK | os.X_OK):
            raise RuntimeError(f"locked characterization executable is unavailable: {executable}")


def _tool_observations(
    image_identity: PreprocessingRuntimeImageIdentity,
) -> PreprocessingRuntimeToolEvidence:
    values = {
        "python_version": _run(("python", "--version")),
        "contract_version": importlib.metadata.version("bspp-orchestration-contract"),
        "runtime_version": importlib.metadata.version("bspp-orchestration-runtime"),
        "control_version": importlib.metadata.version("bspp-orchestration-control"),
        "mmseqs_version": _run(("/usr/local/bin/mmseqs", "version")),
        "colabfold_version": importlib.metadata.version("colabfold"),
        "rsync_version": normalize_rsync_version(_run(("/usr/bin/rsync", "--version"), first_line=True)),
        "tar_version": _run(("/usr/bin/tar", "--version"), first_line=True),
        "lz4_version": _run(("/usr/bin/lz4", "--version"), first_line=True),
        "flock_version": _run(("/usr/bin/flock", "--version"), first_line=True),
    }
    if not values["python_version"].startswith("Python 3.12"):
        raise RuntimeError("image Python is not 3.12")
    observed = PreprocessingRuntimeToolEvidence(
        python_version=values["python_version"],
        contract_version=values["contract_version"],
        runtime_version=values["runtime_version"],
        control_version=values["control_version"],
        mmseqs_version=values["mmseqs_version"],
        colabfold_version=values["colabfold_version"],
        rsync_version=values["rsync_version"],
        tar_version=values["tar_version"],
        lz4_version=values["lz4_version"],
        flock_version=values["flock_version"],
    )
    try:
        validate_preprocessing_runtime_tool_evidence(image_identity, observed)
    except ValueError as exc:
        raise RuntimeError("qualified tool versions do not match the image manifest") from exc
    return observed


def _publish_qualification(record_path: Path, report: dict[str, object], image_identity: object) -> None:
    tuple_payload = json.loads(os.environ["BSPP_PREPROCESSING_QUALIFICATION_TUPLE"])
    qualification_tuple = preprocessing_runtime_qualification_tuple_from_mapping(tuple_payload)
    if qualification_tuple.image_identity != image_identity:
        raise RuntimeError("embedded image identity does not match qualification tuple")
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)
    if tuple_id != os.environ["BSPP_PREPROCESSING_QUALIFICATION_TUPLE_ID"]:
        raise RuntimeError("qualification tuple id mismatch")
    submitted = preprocessing_runtime_qualification_record_from_mapping(json.loads(record_path.read_text()))
    if submitted.status != "submitted" or submitted.qualification_tuple != qualification_tuple:
        raise RuntimeError("qualification record is not the matching submitted intent")
    source_bundle = Path(os.environ["BSPP_PREPROCESSING_SOURCE_BUNDLE"])
    if str(source_bundle) != qualification_tuple.source_bundle_path or not source_bundle.is_file():
        raise RuntimeError("qualified source bundle is missing or mismatched")
    with source_bundle.open("rb") as source_handle:
        source_sha = hashlib.file_digest(source_handle, "sha256").hexdigest()
    if source_sha != qualification_tuple.source_bundle_sha256:
        raise RuntimeError("observed source bundle hash does not match qualification tuple")
    image_sha = os.environ["BSPP_PREPROCESSING_IMAGE_SHA256"]
    if image_sha != qualification_tuple.cluster_image_sha256:
        raise RuntimeError("observed cluster image hash does not match qualification tuple")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("scheduled qualification requires SLURM_JOB_ID")
    gpu = _scheduled_gpu_evidence()
    now = datetime.now(UTC)
    expires = now + timedelta(hours=int(os.environ["BSPP_PREPROCESSING_QUALIFICATION_EXPIRES_HOURS"]))
    manifest_sha = str(report["image_manifest_sha256"])
    smoke = PreprocessingRuntimeSmokeEvidence(
        runtime_command=PREPROCESSING_RUNTIME_COMMAND,
        runtime_contract_id=PREPROCESSING_RUNTIME_CONTRACT_ID,
        adapter_version=PREPROCESSING_ADAPTER_VERSION,
        command_order=PREPROCESSING_COMMAND_ORDER,
        action_evidence_sha256=str(report["action_evidence_sha256"]),
        tools=preprocessing_runtime_tool_evidence_from_mapping(report["tools"]),  # type: ignore[arg-type]
        image=PreprocessingRuntimeImageEvidence(
            manifest_path=str(_MANIFEST),
            manifest_sha256=manifest_sha,
            cluster_image_sha256=image_sha,
            oci_digest=qualification_tuple.oci_digest,
        ),
        source=PreprocessingRuntimeSourceEvidence(
            bundle_id=qualification_tuple.source_bundle_id,
            bundle_path=str(source_bundle),
            bundle_sha256=source_sha,
        ),
        gpu=PreprocessingRuntimeGpuEvidence(nvidia_smi=gpu),
    )
    qualified = PreprocessingRuntimeQualificationRecord(
        status="qualified",
        tuple_id=tuple_id,
        qualification_tuple=qualification_tuple,
        submitted_at=submitted.submitted_at,
        qualified_at=_timestamp(now),
        expires_at=_timestamp(expires),
        job_id=job_id,
        smoke_evidence=smoke,
    )
    _write_record_atomic(record_path, qualified)


def _write_record_atomic(path: Path, record: PreprocessingRuntimeQualificationRecord) -> None:
    payload = json.dumps(record.to_mapping(), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _fixture(root: Path, image_identity: object) -> tuple[PhaseRunSpec, Path, Path]:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    mmseqs = bin_dir / "mmseqs"
    search = bin_dir / "colabfold_search"
    _write_executable(mmseqs, _fake_gpuserver())
    _write_executable(search, _fake_search())
    source = root / "input.fa"
    source.write_bytes(_chunk_bytes())
    work_plan = plan_preprocessing_fasta(
        source, PreprocessingPlanOptions(requested_tranches=1, records_per_chunk=3, nodes=1, gpus_per_node=1)
    )
    chunk = work_plan.chunks[0]
    scratch_input = root / "runtime-input"
    split_input = root / "split-input"
    staged = scratch_input / "n0g0" / chunk.name
    split = split_input / chunk.name
    staged.parent.mkdir(parents=True)
    split.parent.mkdir(parents=True)
    staged.write_bytes(_chunk_bytes())
    split.write_bytes(_chunk_bytes())
    site = PreprocessingSiteConfig(
        mmseqs_executable=str(mmseqs),
        colabfold_search_executable=str(search),
        tar_executable="/usr/bin/tar",
        lz4_executable="/usr/bin/lz4",
        database_root=SELECTED_DATABASE_ROOT,
        input_root=str(scratch_input),
        scratch_output_root=str(root / "scratch-output"),
        project_logs_root=str(root / "logs"),
        finished_msa_root=str(root / "finished-msa"),
        split_input_root=str(split_input),
        finished_input_root=str(root / "finished-input"),
        container_image="/images/preprocessing.sqsh",
        container_mounts=("/tmp:/tmp",),
        max_concurrency=1,
        gpu_delay_seconds=0,
    )
    execution = plan_preprocessing_chunk_execution(
        chunk=chunk,
        records=work_plan.input.records,
        expected_a3m_members=("AFDB_zeta.a3m", "AFDB_alpha.a3m", "AFDB_mu.a3m"),
        # Synthetic fake-kernel stems are not real AFDB model IDs; opt out by stem policy.
        scientific=PreprocessingScientificConfig(require_afdb_model_id_stem=False),
        site=site,
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    action = PreprocessingRuntimeAction(
        action_id="preprocessing-chunk-000000",
        dependencies=(),
        resources=PhaseSlurmResources(partition="gpu", cpus_per_task=8, memory="32G", time="01:00:00", gres="gpu:1"),
        payload=execution,
    )
    database_set = DatabaseSetIdentity(identifier="image-smoke", version="2026-08")
    source_manifest = DatabaseSourceManifest(
        database_set=database_set,
        source_root="/srv/bspp/databases/image-smoke/2026-08",
        members=tuple(
            DatabaseSourceMember(
                role=role,
                database_name=name,
                logical_name=name,
                source_path=name,
                source_kind="regular",
                resolved_path=name,
                resolved_kind="regular",
                size_bytes=1,
                mtime_ns=1,
                alias_topology=(),
                preexisting_checksum=None,
            )
            for role, name in (
                ("primary", execution.scientific.primary_database_name),
                ("metagenomic", execution.scientific.metagenomic_database_name),
            )
        ),
    )
    database = build_preprocessing_database_binding(
        selection=DatabaseSetSelection(
            database_set=database_set,
            requested_policy=DatabaseAccessPolicy.DIRECT,
        ),
        source_manifest=source_manifest,
        source_manifest_projection="attempts/attempt-0001/database-source-manifest.json",
        staging=DatabaseProfileStagingSnapshot(
            cache_root="/var/cache/bspp/image-smoke",
            unix_user=pwd.getpwuid(os.geteuid()).pw_name,
            expected_filesystem_type="ext4",
            reserve_bytes=0,
            lock_wait_seconds=60,
        ),
        gpuserver_argv=execution.gpuserver_argv,
        search_argv=execution.search_argv,
    )
    qualification_tuple = PreprocessingRuntimeQualificationTuple(
        cluster_profile="image-smoke",
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
        image_identity=image_identity,  # type: ignore[arg-type]
    )
    selection = QualifiedPreprocessingRuntimeSelection(
        qualification_tuple=qualification_tuple,
        qualification_record_path="/qualification/image-smoke.json",
        qualified_at="2026-08-20T00:00:00.000000Z",
        expires_at="2027-08-20T00:00:00.000000Z",
    )
    source_bytes = source.read_bytes()
    runspec = PhaseRunSpec(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_plan_digest="4" * 64,
        materialized_at="2026-08-20T00:00:00.000000Z",
        input_location=VerifiedLocalInputLocation(
            path=str(source), sha256=hashlib.sha256(source_bytes).hexdigest(), size_bytes=len(source_bytes)
        ),
        cluster=ResolvedClusterSnapshot(
            profile_name="image-smoke",
            owner="image-smoke",
            transport="local-slurm",
            ssh_target=None,
            account="image-smoke",
            project_root=str(root),
            output_root=str(root / "output"),
            staging_root=str(root / "staging"),
            orchestration_repo="/opt/bspp",
            runtime_image="/images/preprocessing.sqsh",
            preprocessing_runtime=selection,
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
    return runspec, runspec_path, root / "evidence.json"


def _write_direct_placement_result(
    runspec: PhaseRunSpec,
    root: Path,
    *,
    selected_root: Path = Path(SELECTED_DATABASE_ROOT),
) -> Path:
    """Create the isolated direct-placement fixture required by execute-chunk."""
    selected_root.mkdir(parents=True)
    manifest = runspec.payload.database.source_manifest
    for member in manifest.members:
        member_path = selected_root / member.source_path
        member_path.parent.mkdir(parents=True, exist_ok=True)
        member_path.write_bytes(b"x" * member.size_bytes)
        os.utime(member_path, ns=(member.mtime_ns, member.mtime_ns))
        member_path.chmod(0o444)
    device = selected_root.stat().st_dev
    result = DatabasePlacementResult(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_id=runspec.payload.actions[0].action_id,
        database_set=manifest.database_set,
        requested_policy=DatabaseAccessPolicy.DIRECT,
        source_manifest_sha256=runspec.payload.database.source_manifest_sha256,
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
            filesystem_type="image-smoke",
            mount_source=str(selected_root),
            mount_options=("ro",),
            super_options=("ro",),
            read_only=True,
        ),
        pre_science_observation=DatabaseSourceObservation(
            source_container_root=DATABASE_SOURCE_ROOT,
            source_manifest_sha256=runspec.payload.database.source_manifest_sha256,
            members=manifest.members,
        ),
    )
    result_path = root / "database-placement-result.json"
    result_path.write_bytes(canonical_database_placement_result_bytes(result))
    result_path.chmod(0o444)
    return result_path


def _verify_paired_row_identity(root: Path, search_argv: tuple[str, ...]) -> str:
    """Exercise the real pinned tools with tiny synthetic rows, never a search."""
    from colabfold.input import msa_to_str

    filter_mode = search_argv[search_argv.index("--filter") + 1]
    if filter_mode != "1":
        raise RuntimeError("fresh preprocessing smoke must preserve paired rows with --filter 1")
    # Two generic query sequences; all aligned hits below are artificial.
    sequences = ("ACDEFGHIKLMNPQRSTVWY" * 2, "YVWTSRQPNMLKIHGFEDCA" * 2)
    digest = hashlib.sha256()
    for case_index, fractions in enumerate((((0.1,), (0.5,)), ((0.1, 0.5), (0.5, 0.1)))):
        case_root = root / f"synthetic-paired-rows-{case_index}"
        case_root.mkdir()
        query_file = case_root / "query.fa"
        target_file = case_root / "artificial-hits.fa"
        alignment_file = case_root / "artificial-alignments.tsv"
        query_file.write_text("".join(f">{101 + i}\n{seq}\n" for i, seq in enumerate(sequences)))
        targets: list[str] = []
        alignments: list[str] = []
        hits: list[list[str]] = []
        target_id = 0
        for chain_index, (sequence, identities) in enumerate(zip(sequences, fractions, strict=True)):
            chain_hits: list[str] = []
            for row_index, fraction in enumerate(identities):
                length = len(sequence)
                retained = round(length * fraction)
                hit = sequence[:retained] + "".join("A" if c != "A" else "C" for c in sequence[retained:])
                chain_hits.append(hit)
                targets.append(f">synthetic_row{row_index}_chain{chain_index}\n{hit}\n")
                identity = sum(left == right for left, right in zip(sequence, hit, strict=True)) / length
                alignments.append(
                    f"{chain_index}\t{target_id}\t200\t{identity}\t1e-10\t0\t{length - 1}\t{length}"
                    f"\t0\t{length - 1}\t{length}\t{length}M\n"
                )
                target_id += 1
            hits.append(chain_hits)
        target_file.write_text("".join(targets))
        alignment_file.write_text("".join(alignments))
        qdb, tdb, result, msa = (case_root / name for name in ("qdb", "tdb", "result", "msa"))
        unpack = case_root / "unpack"
        unpack.mkdir()
        commands = (
            ("createdb", str(query_file), str(qdb), "--shuffle", "0", "--dbtype", "1"),
            ("createdb", str(target_file), str(tdb), "--shuffle", "0", "--dbtype", "1"),
            ("tsv2db", str(alignment_file), str(result), "--output-dbtype", "5"),
            (
                "result2msa",
                str(qdb),
                str(tdb),
                str(result),
                str(msa),
                "--db-load-mode",
                "2",
                "--msa-format-mode",
                "5",
                "--threads",
                "1",
                "--filter-msa",
                str(int(filter_mode == "2")),
                "--filter-min-enable",
                "1000",
                "--diff",
                "3000",
                "--qid",
                "0.2,0.4,0.6,0.8,1.0",
                "--qsc",
                "0",
                "--max-seq-id",
                "0.95",
            ),
            ("unpackdb", str(msa), str(unpack), "--unpack-name-mode", "0", "--unpack-suffix", ".paired.a3m"),
        )
        for command in commands:
            _run(("/usr/local/bin/mmseqs", *command))
        paired = [(unpack / f"{index}.paired.a3m").read_text() for index in range(len(sequences))]
        assembled = msa_to_str(None, paired, list(sequences), [1, 1])
        expected_lines = [f"#{len(sequences[0])},{len(sequences[1])}\t1,1", ">101\t102", "".join(sequences)]
        for row_index in range(len(hits[0])):
            expected_lines.extend(
                (
                    f">synthetic_row{row_index}_chain0\tsynthetic_row{row_index}_chain1",
                    hits[0][row_index] + hits[1][row_index],
                )
            )
        if assembled.splitlines() != expected_lines:
            raise RuntimeError("synthetic paired row identity or sequence content changed")
        digest.update(assembled.encode())
    return digest.hexdigest()


def _run(argv: tuple[str, ...], *, first_line: bool = False) -> str:
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    value = (result.stdout or result.stderr).strip()
    return value.splitlines()[0] if first_line else value


def _scheduled_gpu_evidence() -> str:
    """Return exact nonblank GPU evidence captured by the scheduled host shell."""
    evidence = os.environ.get("BSPP_PREPROCESSING_GPU_EVIDENCE")
    if evidence is None or not evidence.strip():
        raise RuntimeError("scheduled preprocessing Runtime Qualification requires host nvidia-smi GPU evidence")
    return evidence


def _reject_source_overlays() -> None:
    forbidden = {".git", "MMSA", "bspp-orchestration-control"}
    for path in Path("/opt/bspp").rglob("*"):
        if path.name in forbidden:
            raise RuntimeError(f"forbidden source/control overlay in image: {path}")


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _chunk_bytes() -> bytes:
    return b">protein-zeta\nAAAA:TT\n>protein-alpha\nCCCC:AAA\n>protein-mu\nGGGG:CC\n"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def _fake_gpuserver() -> str:
    return textwrap.dedent("""\
        #!/usr/bin/env python3
        import json, os, signal, sys, time
        from pathlib import Path
        output = Path(os.environ["BSPP_FAKE_INVOCATIONS"])
        def write(kind):
            with output.open("a") as handle:
                handle.write(json.dumps({"kind": kind, "argv": sys.argv[1:]}) + "\\n")
        write("gpuserver")
        def stop(_signum, _frame):
            write("gpuserver-stopped"); raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        while True: time.sleep(0.05)
    """)


def _fake_search() -> str:
    return textwrap.dedent("""\
        #!/usr/bin/env python3
        import json, os, sys
        from pathlib import Path
        with Path(os.environ["BSPP_FAKE_INVOCATIONS"]).open("a") as handle:
            handle.write(json.dumps({"kind": "search", "argv": sys.argv[1:]}) + "\\n")
        output = Path(sys.argv[5])
        members = os.environ["BSPP_FAKE_A3MS"].split("|")
        sequences = [
            line for line in Path(sys.argv[3]).read_text().splitlines()
            if line and not line.startswith(">")
        ]
        chains = [chain for sequence in sequences for chain in sequence.split(":")]
        for index, member in enumerate(members):
            left, right = sequences[index].split(":")
            (output / member).write_text(
                f"#{len(left)},{len(right)}\\t1,1\\n>{Path(member).stem}\\n{left}{right}\\n"
            )
        pair_mode = sys.argv[sys.argv.index("--pair-mode") + 1]
        raw_ids = range(len(members), len(members) * 2) if pair_mode == "paired" else ()
        for raw_id in raw_ids:
            (output / f"{raw_id}.a3m").write_text(f"#{len(chains[raw_id])}\\t1\\n")
        print("search complete")
    """)


if __name__ == "__main__":
    main()
