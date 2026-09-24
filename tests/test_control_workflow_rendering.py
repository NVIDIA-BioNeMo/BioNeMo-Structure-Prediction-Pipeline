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

"""Tests for Control Plane workflow Slurm rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from bspp.orchestration.contract.runspec import runspec_from_mapping
from bspp.orchestration.contract.source_package import SourcePackageIdentity
from bspp.orchestration.control.execution_bootstrap import ImageIdentity
from bspp.orchestration.control.transport import CommandResult, RemoteSlurmTransport
from bspp.orchestration.control.workflow_rendering import (
    WorkflowRenderContext,
    render_workflow_step_script,
)

ROOT = Path(__file__).resolve().parents[1]


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        return CommandResult(argv=argv, returncode=0, stdout="1234\n", stderr="")


def test_render_workflow_step_scripts_cover_runtime_worker_finalizer_acceptance_and_data_placement(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "materialized.runspec.yaml"
    spec = runspec_from_mapping(_runspec_data(tmp_path), source_path=spec_path, source_hash="abc123")
    assert spec.workflow is not None
    context = WorkflowRenderContext(
        materialized_runspec=spec_path,
        source_bundle_path=tmp_path / "bundles" / "bspp-orchestration-abc123.tar.zst",
        runtime_qualification_path=tmp_path / "qualifications" / "tuple.json",
    )

    rendered = {
        step.name: render_workflow_step_script(spec, step, context=context).script_path.read_text()
        for step in spec.workflow.steps
    }

    preflight = rendered["preflight"]
    assert 'bspp-orchestration-runtime runspec preflight "$BSPP_RUNSPEC" --write-report --strict' in preflight
    assert f"BSPP_RUNTIME_QUALIFICATION={tmp_path}/qualifications/tuple.json" in preflight
    assert "#SBATCH --partition=cpu" in preflight
    assert "#SBATCH --gres" not in preflight

    recipe = rendered["recipe"]
    assert 'bspp-orchestration-runtime inputs coverage --runspec "$BSPP_RUNSPEC" --write-report' in recipe
    assert 'bspp-orchestration-runtime runspec render-recipe "$BSPP_RUNSPEC" --execute' in recipe
    assert "#SBATCH --partition=cpu" in recipe
    assert "#SBATCH --gres" not in recipe

    preprocess = rendered["preprocess"]
    assert 'bspp-orchestration-runtime inputs prepare --runspec "$BSPP_RUNSPEC"' in preprocess
    assert 'bspp-orchestration-runtime runspec render-preprocess "$BSPP_RUNSPEC" --execute --no-allowlists' in (
        preprocess
    )
    assert "#SBATCH --partition=cpu" in preprocess
    assert "#SBATCH --gres" not in preprocess

    worker = rendered["slurm"]
    assert "#SBATCH --partition=gpu" in worker
    assert "#SBATCH --array=0-3" in worker
    assert "#SBATCH --gres=gpu:1" in worker
    assert "srun \\" in worker
    assert f"--container-image={tmp_path}/image-cache/bspp.sqsh" in worker
    assert f"{tmp_path}/orchestration:/workspace/bspp-orchestration" in worker
    assert f"{spec_path}:/workspace/bspp-runspec/runspec.yaml" in worker
    assert f"BSPP_SOURCE_BUNDLE={tmp_path}/bundles/bspp-orchestration-abc123.tar.zst" in worker
    assert 'tar -xaf "$BSPP_SOURCE_BUNDLE"' in worker
    assert 'bspp-orchestration-runtime worker archive-task --runspec "$BSPP_RUNSPEC"' in worker

    finalizer = rendered["analysis-finalize"]
    assert "#SBATCH --partition=cpu" in finalizer
    assert "-m bspp.orchestration.runtime.postprocessing.analysis_finalizer" in finalizer
    assert f"--parquet {tmp_path}/output/run1/analysis_metadata.parquet" in finalizer

    parity = rendered["acceptance-tar-payload-parity"]
    assert "#SBATCH --time=00:30:00" in parity
    assert '"${ORCHESTRATION_ROOT}/containers/scripts/slurm-tar-payload-parity.sh"' in parity
    assert '--workers "$SLURM_CPUS_PER_TASK"' in parity
    assert "--payload-sample-count 20" in parity
    assert "--exclude metadata/" in parity

    semantic = rendered["acceptance-semantic"]
    assert '"${ORCHESTRATION_ROOT}/containers/scripts/slurm-semantic-acceptance.sh"' in semantic
    assert "--expected-selected-ids 2" in semantic

    verify = rendered["acceptance-verify-evidence"]
    assert "#SBATCH --partition=cpu" in verify
    assert "#SBATCH --gres" not in verify
    assert f"root = Path({str(tmp_path / 'output/run1/evidence/acceptance')!r})" in verify
    assert "parity_report_path=root / 'tar_payload_parity' / 'tar_payload_parity_report.json'" in verify
    assert "semantic_report_path=root / 'semantic_acceptance' / 'semantic_acceptance_summary.json'" in verify

    upload = rendered["upload-s3"]
    assert "bspp-orchestration-runtime upload-s3" in upload
    assert "--tool dm" in upload
    assert f"--dataset-output-dir {tmp_path}/output/run1" in upload
    assert f"--data-dir {tmp_path}/output/run1/evidence/data-placement/s3" in upload
    assert f"--write-evidence {tmp_path}/output/run1/evidence/data-placement/s3/data-placement.json" in upload
    assert "--s3-destination-prefix s3://example-bucket/users/test/postprocessed/ds1/" in upload


def test_render_upload_s3_uses_configured_non_dm_mover(tmp_path: Path) -> None:
    spec_path = tmp_path / "materialized.runspec.yaml"
    spec = runspec_from_mapping(_runspec_data(tmp_path, s3_tool="s5cmd"), source_path=spec_path, source_hash="abc123")
    assert spec.workflow is not None
    upload_step = next(step for step in spec.workflow.steps if step.name == "upload-s3")

    script = render_workflow_step_script(
        spec,
        upload_step,
        context=WorkflowRenderContext(materialized_runspec=spec_path, source_bundle_path=None),
    ).script_path.read_text()

    assert "--tool s5cmd" in script
    assert "dm job copy" not in script


def test_render_workflow_step_without_toolkit_mount_omits_toolkit_in_srun_mounts(tmp_path: Path) -> None:
    """When RunSpec has no afdb_toolkit_repo, srun --container-mounts omits toolkit."""
    spec_path = tmp_path / "materialized.runspec.yaml"
    spec = runspec_from_mapping(_baked_runspec_data(tmp_path), source_path=spec_path, source_hash="abc123")
    assert spec.workflow is not None
    slurm_step = next(step for step in spec.workflow.steps if step.name == "slurm")

    script = render_workflow_step_script(
        spec,
        slurm_step,
        context=WorkflowRenderContext(materialized_runspec=spec_path, source_bundle_path=None),
    ).script_path.read_text()

    # The srun --container-mounts line should not include the toolkit mount
    mount_line = next(line for line in script.splitlines() if "--container-mounts=" in line)
    assert "/workspace/AFDB-Integration-Kit" not in mount_line
    assert "/workspace/bspp-orchestration" in mount_line


def test_render_workflow_step_with_explicit_toolkit_includes_toolkit_in_srun_mounts(tmp_path: Path) -> None:
    """When RunSpec has afdb_toolkit_repo, srun --container-mounts includes toolkit."""
    spec_path = tmp_path / "materialized.runspec.yaml"
    spec = runspec_from_mapping(_runspec_data(tmp_path), source_path=spec_path, source_hash="abc123")
    assert spec.workflow is not None
    slurm_step = next(step for step in spec.workflow.steps if step.name == "slurm")

    script = render_workflow_step_script(
        spec,
        slurm_step,
        context=WorkflowRenderContext(materialized_runspec=spec_path, source_bundle_path=None),
    ).script_path.read_text()

    assert f"{tmp_path}/toolkit:/workspace/AFDB-Integration-Kit" in script


def _baked_runspec_data(tmp_path: Path) -> dict[str, object]:
    """Return runspec data with no afdb_toolkit_repo (baked mode)."""
    data = _runspec_data(tmp_path)
    paths = data["paths"]
    assert isinstance(paths, dict)
    paths.pop("afdb_toolkit_repo")
    # Remove toolkit mount from container
    container = data["container"]
    assert isinstance(container, dict)
    mounts = container["mounts"]
    assert isinstance(mounts, list)
    container["mounts"] = [
        m for m in mounts if isinstance(m, dict) and m.get("target") != "/workspace/AFDB-Integration-Kit"
    ]
    return data


def test_governed_workflow_rejects_missing_identities(tmp_path: Path) -> None:
    data = _runspec_data(tmp_path)
    data["run_kind"] = "canary"
    data["workflow"] = {"steps": [{"name": "preflight", "run": True}]}
    spec_path = tmp_path / "materialized.runspec.yaml"
    spec = runspec_from_mapping(data, source_path=spec_path, source_hash="abc123")
    assert spec.workflow is not None
    with pytest.raises(ValueError, match="governed workflow requires source package, toolkit, and image identities"):
        render_workflow_step_script(
            spec,
            spec.workflow.steps[0],
            context=WorkflowRenderContext(materialized_runspec=spec_path, source_bundle_path=None),
        )


def test_governed_runtime_argv_uses_bootstrap_unique_scratch_placeholders() -> None:
    from bspp.orchestration.control.workflow_rendering import _governed_runtime_argv

    data = _runspec_data(Path("/synthetic"))
    data["run_kind"] = "canary"
    data["workflow"] = {"steps": [{"name": "preflight", "run": True}]}
    spec = runspec_from_mapping(data)
    assert spec.workflow is not None
    argv = _governed_runtime_argv(spec, spec.workflow.steps[0])
    assert "{BSPP_SOURCE_ROOT}" in argv
    assert "{BSPP_TOOLKIT_ROOT}" in argv
    assert not any("/tmp/bspp" in value for value in argv)


def test_governed_workflow_mounts_only_exact_read_only_identity_files(tmp_path: Path) -> None:
    data = _runspec_data(tmp_path)
    data["run_kind"] = "canary"
    data["workflow"] = {"steps": [{"name": "preflight", "run": True}]}
    data["container"]["mounts"] = [  # type: ignore[index]
        {"source": str(tmp_path / "input"), "target": "/data/input"},
        {"source": str(tmp_path / "output"), "target": "/data/output"},
        {"source": str(tmp_path / "orchestration"), "target": "/ambient/orchestration"},
        {"source": str(tmp_path / "toolkit"), "target": "/ambient/toolkit"},
    ]
    spec_path = tmp_path / "runspec.yaml"
    spec = runspec_from_mapping(data, source_path=spec_path)
    assert spec.workflow is not None

    def package(path: Path, marker: str) -> SourcePackageIdentity:
        return SourcePackageIdentity(
            format="bspp-tar-v1",
            verifier="safe-tar-v1",
            package_path=path.absolute(),
            package_size_bytes=1,
            package_sha256=marker * 64,
            manifest_sha256="c" * 64,
            commit="d" * 40,
            tree="e" * 40,
        )

    script = render_workflow_step_script(
        spec,
        spec.workflow.steps[0],
        context=WorkflowRenderContext(
            materialized_runspec=spec_path,
            source_bundle_path=None,
            runtime_qualification_path=tmp_path / "qualification.json",
            source_package_identity=package(tmp_path / "source.tar", "a"),
            toolkit_package_identity=package(tmp_path / "toolkit.tar", "b"),
            image_identity=ImageIdentity(1, "digest-checked", (tmp_path / "image.sqsh").absolute(), 1, "f" * 64),
            runtime_ipsae_binary_path=(tmp_path / "runtime-ipsae" / "ipsae_cpp").absolute(),
            runtime_ipsae_binary_sha256="9" * 64,
            runtime_qualification_sha256="8" * 64,
            runtime_bootstrap_sha256="7" * 64,
        ),
    ).script_path.read_text()
    mount_arg = next(part for part in script.split() if part.startswith("--container-mounts="))
    assert mount_arg.count(":ro") == 6
    assert f"{tmp_path}/source.tar:/run/bspp/source-package.tar:ro" in mount_arg
    assert f"{tmp_path}/toolkit.tar:/run/bspp/toolkit-package.tar:ro" in mount_arg
    assert f"{tmp_path}/runtime-ipsae/ipsae_cpp:/run/bspp/runtime-ipsae/ipsae_cpp:ro" in mount_arg
    assert "--expected-runtime-ipsae-sha256" in script
    assert str(tmp_path / "orchestration") not in mount_arg
    assert str(tmp_path / "toolkit") + ":/workspace" not in mount_arg
    assert f"{tmp_path}/input:/data/input:ro" in mount_arg
    assert f"{tmp_path}/output:/data/output" in mount_arg


def test_governed_workflow_rejects_writable_alias_over_protected_inputs(tmp_path: Path) -> None:
    data = _runspec_data(tmp_path)
    data["run_kind"] = "canary"
    data["workflow"] = {"steps": [{"name": "preflight", "run": True}]}
    data["container"]["mounts"] = [  # type: ignore[index]
        {"source": str(tmp_path), "target": "/run"}
    ]
    spec_path = tmp_path / "runspec.yaml"
    spec = runspec_from_mapping(data, source_path=spec_path)
    assert spec.workflow is not None
    identity = SourcePackageIdentity(
        format="bspp-tar-v1",
        verifier="safe-tar-v1",
        package_path=(tmp_path / "source.tar").absolute(),
        package_size_bytes=1,
        package_sha256="a" * 64,
        manifest_sha256="b" * 64,
        commit="c" * 40,
        tree="d" * 40,
        package_role="orchestration",
    )
    toolkit = SourcePackageIdentity(
        **{**identity.__dict__, "package_path": (tmp_path / "toolkit.tar").absolute(), "package_role": "toolkit"}
    )
    with pytest.raises(ValueError, match="overlaps protected"):
        render_workflow_step_script(
            spec,
            spec.workflow.steps[0],
            context=WorkflowRenderContext(
                materialized_runspec=spec_path,
                source_bundle_path=None,
                runtime_qualification_path=tmp_path / "qualification.json",
                source_package_identity=identity,
                toolkit_package_identity=toolkit,
                image_identity=ImageIdentity(1, "digest-checked", (tmp_path / "image").absolute(), 1, "e" * 64),
                runtime_ipsae_binary_path=(tmp_path / "runtime-ipsae" / "ipsae_cpp").absolute(),
                runtime_ipsae_binary_sha256="9" * 64,
                runtime_qualification_sha256="8" * 64,
                runtime_bootstrap_sha256="7" * 64,
            ),
        )


def test_render_workflow_mounts_aws_profile_for_aws_secret_ref(tmp_path: Path) -> None:
    data = _runspec_data(tmp_path)
    data["secrets"] = {"s3_credentials_ref": "aws:example-account"}
    spec_path = tmp_path / "materialized.runspec.yaml"
    spec = runspec_from_mapping(data, source_path=spec_path, source_hash="abc123")
    assert spec.workflow is not None
    upload_step = next(step for step in spec.workflow.steps if step.name == "upload-s3")

    script = render_workflow_step_script(
        spec,
        upload_step,
        context=WorkflowRenderContext(materialized_runspec=spec_path, source_bundle_path=None),
    ).script_path.read_text()

    assert f"{Path.home() / '.aws'}:/workspace/bspp-aws" in script
    assert "AWS_SHARED_CREDENTIALS_FILE=/workspace/bspp-aws/credentials" in script
    assert "AWS_CONFIG_FILE=/workspace/bspp-aws/config" in script
    assert "AWS_PROFILE=example-account" in script
    assert "export AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE AWS_PROFILE" in script
    assert "AWS_ACCESS_KEY_ID" not in script
    assert "AWS_SECRET_ACCESS_KEY" not in script


def test_runspec_rejects_data_placement_step_without_configured_mover(tmp_path: Path) -> None:
    data = _runspec_data(tmp_path)
    data.pop("data_placement")

    with pytest.raises(ValueError, match=r"workflow step upload-s3 requires data_placement\.s3"):
        runspec_from_mapping(data)


def test_acceptance_wrapper_scripts_delegate_to_reusable_validators() -> None:
    parity = ROOT / "containers" / "scripts" / "slurm-tar-payload-parity.sh"
    semantic = ROOT / "containers" / "scripts" / "slurm-semantic-acceptance.sh"
    submitter = ROOT / "containers" / "scripts" / "submit-acceptance-checks.sh"

    assert parity.stat().st_mode & 0o111
    assert semantic.stat().st_mode & 0o111
    assert submitter.stat().st_mode & 0o111
    assert "validate tar-payload-parity" in parity.read_text()
    assert "validate semantic-acceptance" in semantic.read_text()
    assert 'sbatch "${submit_args[@]}" "$PARITY_SCRIPT"' in submitter.read_text()
    assert "afterok:${parity_job}:${semantic_job}" in submitter.read_text()


def test_remote_slurm_transport_submits_script_with_afterok_dependencies() -> None:
    runner = RecordingRunner()
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None, runner=runner)

    submission = transport.submit_script(Path("/runs/evidence/slurm/preprocess.sbatch"), dependencies=("1001",))

    assert submission.job_id == "1234"
    assert runner.calls == [
        ("sbatch", "--parsable", "--dependency=afterok:1001", "/runs/evidence/slurm/preprocess.sbatch")
    ]


def _runspec_data(tmp_path: Path, *, s3_tool: str = "dm") -> dict[str, object]:
    output_dir = tmp_path / "output" / "run1"
    return {
        "run_kind": "dev",
        "dataset": {
            "name": "ds1",
            "run_id": "run1",
            "mode": "archive",
            "array": "0-3",
            "archive_source": "tracking",
        },
        "cluster": {"name": "example-cluster", "account": "user-account", "owner": "tester"},
        "paths": {
            "project_root": str(tmp_path / "project"),
            "staging_dir": str(tmp_path / "staging" / "ds1"),
            "output_dir": str(output_dir),
            "log_dir": str(output_dir / "logs"),
            "afdb_toolkit_repo": str(tmp_path / "toolkit"),
            "orchestration_repo": str(tmp_path / "orchestration"),
            "recipe_dir": str(output_dir / "rendered_recipe"),
        },
        "references": {
            "master_parquet": str(tmp_path / "refs" / "master.parquet"),
            "tracking_parquet": str(tmp_path / "refs" / "tracking.parquet"),
            "manifest_csv": str(tmp_path / "refs" / "manifest.csv"),
            "uniprot_duckdb": str(tmp_path / "refs" / "uniprot.duckdb"),
        },
        "container": {
            "image": str(tmp_path / "image-cache" / "bspp.sqsh"),
            "mounts": [
                {"source": str(tmp_path / "orchestration"), "target": "/workspace/bspp-orchestration"},
                {"source": str(tmp_path / "toolkit"), "target": "/workspace/AFDB-Integration-Kit"},
                {"source": str(tmp_path / "project"), "target": str(tmp_path / "project")},
            ],
        },
        "resources": {
            "gpu_worker": {
                "partition": "gpu",
                "cpus_per_task": 30,
                "memory": "128G",
                "time": "04:00:00",
                "gres": "gpu:1",
                "array": "0-3",
            },
            "control_cpu": {"partition": "cpu", "cpus_per_task": 4, "memory": "16G", "time": "00:20:00"},
            "analysis_finalize": {"partition": "cpu", "cpus_per_task": 4, "memory": "16G", "time": "00:20:00"},
            "acceptance_tar_payload_parity": {
                "partition": "cpu",
                "cpus_per_task": 8,
                "memory": "32G",
                "time": "00:30:00",
            },
            "acceptance_semantic": {"partition": "cpu", "cpus_per_task": 4, "memory": "16G", "time": "00:20:00"},
        },
        "worker": {
            "stages": "metadata_export",
            "workers": 24,
            "batch_size": 500,
            "shards_per_archive": 2,
            "self_upload": False,
            "local_scratch": True,
            "scratch_dir": "/dev/shm",
            "s5cmd_path": "s5cmd",
            "upload_slots": 4,
        },
        "storage": {
            "s3_archive_prefix": "s3://example-bucket/structures/",
            "s3_output_prefix": "s3://example-bucket/users/test/postprocessed/ds1/",
            "allow_production_prefixes": True,
            "upload_mode": "tar",
            "local_tar_dir": str(output_dir / "local_tars"),
            "local_tar_manifest_csv": str(output_dir / "local_tars.csv"),
        },
        "data_placement": {"s3": {"tool": s3_tool}},
        "analysis_metadata": {
            "enabled": True,
            "csv_path": str(output_dir / "analysis_metadata.csv"),
            "parquet_path": str(output_dir / "analysis_metadata.parquet"),
            "selected_ids_path": str(output_dir / "high_quality_model_ids.txt"),
            "finalize_after_gpu": True,
        },
        "validation": {"expected_archives": 4, "expected_selected_ids": 2},
        "acceptance": {
            "baseline_output_dir": str(tmp_path / "baseline"),
            "baseline_run_name": "baseline",
            "candidate_run_name": "candidate",
            "payload_sample_count": 20,
        },
        "workflow": {
            "steps": [
                {"name": "preflight", "run": True},
                {"name": "recipe", "run": True},
                {"name": "preprocess", "run": True},
                {"name": "slurm", "run": True, "mode": "submit-and-monitor"},
                {"name": "analysis-finalize", "run": True, "mode": "submit-and-monitor"},
                {"name": "acceptance-tar-payload-parity", "run": True, "mode": "submit-and-monitor"},
                {"name": "acceptance-semantic", "run": True, "mode": "submit-and-monitor"},
                {"name": "acceptance-verify-evidence", "run": True},
                {"name": "upload-s3", "run": True},
            ]
        },
        "submission": {
            "evidence_dir": str(output_dir / "evidence"),
            "report_path": str(output_dir / "RUN_REPORT.md"),
        },
        "secrets": {"s3_credentials_ref": "env:SWIFTSTACK"},
    }
