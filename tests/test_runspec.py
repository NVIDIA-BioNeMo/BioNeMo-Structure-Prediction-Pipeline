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

"""Tests for canonical RunSpec loading and guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from bspp.orchestration.contract.runspec import (
    ReferenceArtifact,
    load_runspec,
    render_artifact_header,
    render_dry_run,
    resolve_hq_chunk_publication_target,
)
from bspp.orchestration.contract.secrets import _DEFAULT_ENV_VARS, MissingSecretError
from bspp.orchestration.runtime.cli import cli


def _base_runspec() -> dict[str, object]:
    return {
        "dataset": {
            "name": "20251128_full_run_v2",
            "run_id": "test-run",
            "mode": "archive",
            "array": "0-0",
        },
        "cluster": {"name": "example-cluster", "account": "example-account"},
        "paths": {
            "project_root": "/proj",
            "staging_dir": "/proj/input/staging",
            "output_dir": "/proj/output",
            "log_dir": "/proj/logs",
            "legacy_repo": "/repo/AFDB-Integration-Kit",
            "orchestration_repo": "/repo/bspp-orchestration",
        },
        "references": {
            "master_parquet": "/refs/master.parquet",
            "tracking_parquet": "/refs/tracking.parquet",
            "manifest_csv": "/refs/manifest.csv",
            "uniprot_duckdb": "/refs/uniprot.duckdb",
        },
        "container": {
            "image": "bspp-orchestration:current",
            "workdir": "/workspace/bspp-orchestration",
            "mounts": [
                {
                    "source": "/repo/AFDB-Integration-Kit",
                    "target": "/workspace/AFDB-Integration-Kit",
                    "read_only": True,
                }
            ],
        },
        "resources": {
            "gpu_worker": {
                "partition": "batch_singlenode",
                "cpus_per_task": 30,
                "memory": "128G",
                "time": "04:00:00",
                "gres": "gpu:1",
                "array": "0-0",
            },
            "aggregate": {
                "partition": "cpu_short",
                "cpus_per_task": 1,
                "memory": "8G",
                "time": "00:20:00",
            },
        },
        "worker": {
            "stages": "ipsae dssp validation metadata_export modelcif_export",
            "workers": 24,
            "batch_size": 500,
            "shards_per_archive": 2,
            "self_upload": True,
            "local_scratch": True,
            "scratch_dir": "/dev/shm",
            "s5cmd_path": "/bin/s5cmd-pdx",
            "upload_slots": 4,
        },
        "storage": {
            "s3_archive_prefix": "s3://example-bucket/structures/",
            "s3_output_prefix": "s3://example-bucket/users/example-user/postprocessed-test/20251128_full_run_v2/",
            "gcs_destination_prefix": None,
            "allow_production_prefixes": False,
        },
        "validation": {
            "expected_archives": 38,
            "expected_allowed_ids": 370679,
            "expected_one_archive_models": 3365,
            "expected_one_archive_objects": 23505,
            "expected_one_archive_aggregate_rows": 6730,
            "gcs_dry_run_only": True,
        },
        "secrets": {"s3_credentials_ref": "env:bspp/s3", "gcs_credentials_ref": None},
    }


def _active_runspec() -> dict[str, object]:
    data = _base_runspec()
    paths = data["paths"]
    container = data["container"]
    validation = data["validation"]
    assert isinstance(paths, dict)
    assert isinstance(container, dict)
    assert isinstance(validation, dict)

    paths.pop("legacy_repo")
    paths["afdb_toolkit_repo"] = "/repo/AFDB-Integration-Kit"
    container.pop("workdir")
    mounts = container["mounts"]
    assert isinstance(mounts, list)
    for mount in mounts:
        assert isinstance(mount, dict)
        mount.pop("read_only", None)
    data["workflow"] = {
        "steps": [
            {"name": "preflight", "run": False},
            {
                "name": "slurm",
                "run": True,
                "mode": "submit-and-monitor",
                "job_id": "12345",
                "array_range": "0-0",
                "rendered_script": "/proj/output/evidence/wp5/run_archive.sbatch",
            },
            {"name": "acceptance-verify-evidence", "run": True},
        ]
    }
    data["submission"] = {
        "evidence_dir": "/proj/output/evidence/test-run",
        "report_path": "/proj/output/RUN_REPORT.md",
    }
    data["acceptance"] = {
        "baseline_output_dir": "/proj/output/baseline",
        "baseline_run_name": "baseline",
        "candidate_run_name": "candidate",
        "payload_sample_count": 20,
    }
    validation.update(
        {
            "expected_tar_count": 2,
            "expected_local_tars_rows": 2,
            "expected_failed_rows": 0,
            "expected_analysis_rows": 3365,
            "expected_selected_ids": 100,
        }
    )
    return data


def _write_runspec(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "runspec.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_load_runspec_parses_documented_archive_scope(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path, _base_runspec()))

    assert spec.run_kind is None
    assert spec.dataset.name == "20251128_full_run_v2"
    assert spec.dataset.array == "0-0"
    assert spec.dataset.archive_source == "tracking"
    assert spec.worker.shards_per_archive == 2
    assert spec.worker.s5cmd_numworkers == 32
    assert spec.worker.provider_copyrights == ("Copyright BSPP Orchestration contributors. All rights reserved.",)
    assert spec.validation.expected_archives == 38
    assert spec.references.master_parquet == Path("/refs/master.parquet")
    assert spec.storage.upload_mode == "files"
    assert spec.storage.tar_compression == "none"
    assert spec.analysis_metadata.enabled is False


def test_load_runspec_parses_valid_run_kind(tmp_path: Path) -> None:
    data = _base_runspec()
    data["run_kind"] = "canary"
    data["worker"]["self_upload"] = False  # type: ignore[index]

    spec = load_runspec(_write_runspec(tmp_path, data))

    assert spec.run_kind == "canary"


def test_load_runspec_rejects_invalid_run_kind(tmp_path: Path) -> None:
    data = _base_runspec()
    data["run_kind"] = "staging"

    with pytest.raises(ValidationError, match="run_kind"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_parses_active_workflow_schema(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path, _active_runspec()))

    assert spec.workflow is not None
    assert [step.name for step in spec.workflow.steps] == [
        "preflight",
        "slurm",
        "acceptance-verify-evidence",
    ]
    assert spec.workflow.steps[0].run is False
    assert spec.workflow.steps[1].mode == "submit-and-monitor"
    assert spec.workflow.steps[1].job_id == "12345"
    assert spec.workflow.steps[1].array_range == "0-0"
    assert spec.workflow.steps[1].rendered_script == Path("/proj/output/evidence/wp5/run_archive.sbatch")
    assert spec.submission is not None
    assert spec.submission.evidence_dir == Path("/proj/output/evidence/test-run")
    assert spec.submission.report_path == Path("/proj/output/RUN_REPORT.md")
    assert spec.submission.controller_runtime == "enroot"
    assert spec.acceptance is not None
    assert spec.acceptance.baseline_output_dir == Path("/proj/output/baseline")
    assert spec.acceptance.tar_payload_match_mode == "by-tar"
    assert spec.acceptance.payload_sample_count == 20
    assert spec.acceptance.candidate_parquet_required is True
    assert spec.paths.afdb_toolkit_repo == Path("/repo/AFDB-Integration-Kit")
    assert spec.paths.legacy_repo == Path("/repo/AFDB-Integration-Kit")
    assert spec.container.workdir == Path("/workspace/bspp-orchestration")
    assert spec.container.mounts[0].read_only is False
    assert spec.validation.expected_tar_count == 2
    assert spec.validation.expected_local_tars_rows == 2
    assert spec.validation.expected_failed_rows == 0
    assert spec.validation.expected_analysis_rows == 3365
    assert spec.validation.expected_selected_ids == 100


def test_load_runspec_rejects_workflow_step_missing_run(tmp_path: Path) -> None:
    data = _active_runspec()
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = [{"name": "preflight"}]

    with pytest.raises(ValidationError, match="run"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_workflow_step_run_string(tmp_path: Path) -> None:
    data = _active_runspec()
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = [{"name": "preflight", "run": "true"}]

    with pytest.raises(ValidationError, match="run"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_empty_workflow_steps(tmp_path: Path) -> None:
    data = _active_runspec()
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = []

    with pytest.raises(ValidationError, match="steps"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_unknown_workflow_step(tmp_path: Path) -> None:
    data = _active_runspec()
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = [{"name": "unknown-step", "run": True}]

    with pytest.raises(ValidationError, match="Unknown workflow step name"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_duplicate_workflow_steps(tmp_path: Path) -> None:
    data = _active_runspec()
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = [
        {"name": "preflight", "run": True},
        {"name": "preflight", "run": False},
    ]

    with pytest.raises(ValidationError, match="Duplicate workflow step name"):
        load_runspec(_write_runspec(tmp_path, data))


@pytest.mark.parametrize(
    "step_name",
    [
        "slurm",
        "analysis-finalize",
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
    ],
)
def test_load_runspec_requires_mode_for_mode_capable_steps(tmp_path: Path, step_name: str) -> None:
    data = _active_runspec()
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = [{"name": step_name, "run": True}]

    with pytest.raises(ValidationError, match="requires mode"):
        load_runspec(_write_runspec(tmp_path, data))


@pytest.mark.parametrize(
    "step_name",
    [
        "slurm",
        "analysis-finalize",
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
    ],
)
def test_load_runspec_monitor_existing_requires_job_id(tmp_path: Path, step_name: str) -> None:
    data = _active_runspec()
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = [{"name": step_name, "run": True, "mode": "monitor-existing"}]

    with pytest.raises(ValidationError, match="monitor-existing mode requires job_id"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_slurm_monitor_existing_requires_array_range(tmp_path: Path) -> None:
    data = _active_runspec()
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = [{"name": "slurm", "run": True, "mode": "monitor-existing", "job_id": "123"}]

    with pytest.raises(ValidationError, match="monitor-existing mode requires array_range"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_mode_for_preflight_step(tmp_path: Path) -> None:
    data = _active_runspec()
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = [{"name": "preflight", "run": True, "mode": "submit-and-monitor"}]

    with pytest.raises(ValidationError, match="does not accept mode"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_invalid_workflow_step_mode(tmp_path: Path) -> None:
    data = _active_runspec()
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = [{"name": "slurm", "run": True, "mode": "submit-only"}]

    with pytest.raises(ValidationError, match="mode"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_accepts_reserved_workflow_step(tmp_path: Path) -> None:
    data = _active_runspec()
    workflow = data["workflow"]
    assert isinstance(workflow, dict)
    workflow["steps"] = [{"name": "download-manifest", "run": False}]

    spec = load_runspec(_write_runspec(tmp_path, data))

    assert spec.workflow is not None
    assert spec.workflow.steps[0].name == "download-manifest"


def test_load_runspec_rejects_active_legacy_repo(tmp_path: Path) -> None:
    data = _active_runspec()
    paths = data["paths"]
    assert isinstance(paths, dict)
    paths["legacy_repo"] = "/repo/legacy"

    with pytest.raises(ValidationError, match=r"paths\.legacy_repo is removed"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_accepts_active_missing_afdb_toolkit_repo_baked_mode(tmp_path: Path) -> None:
    data = _active_runspec()
    paths = data["paths"]
    assert isinstance(paths, dict)
    paths.pop("afdb_toolkit_repo")

    spec = load_runspec(_write_runspec(tmp_path, data))

    assert spec.paths.afdb_toolkit_repo is None
    assert spec.paths.legacy_repo == Path("/opt/afdb-toolkit")


def test_load_runspec_baked_mode_backfills_legacy_repo_to_baked_path(tmp_path: Path) -> None:
    data = _active_runspec()
    paths = data["paths"]
    assert isinstance(paths, dict)
    paths.pop("afdb_toolkit_repo")

    spec = load_runspec(_write_runspec(tmp_path, data))

    from bspp.orchestration.contract.runspec import BAKED_TOOLKIT_PATH

    assert spec.paths.legacy_repo == BAKED_TOOLKIT_PATH


def test_active_runspec_explicit_toolkit_path_still_validated(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path, _active_runspec()))

    assert spec.paths.afdb_toolkit_repo == Path("/repo/AFDB-Integration-Kit")
    assert spec.paths.legacy_repo == Path("/repo/AFDB-Integration-Kit")


def test_legacy_runspec_backfills_legacy_repo_from_afdb_toolkit_repo(tmp_path: Path) -> None:
    data = _base_runspec()
    paths = data["paths"]
    assert isinstance(paths, dict)
    paths.pop("legacy_repo")
    paths["afdb_toolkit_repo"] = "/repo/AFDB-Integration-Kit"

    spec = load_runspec(_write_runspec(tmp_path, data))

    assert spec.paths.afdb_toolkit_repo == Path("/repo/AFDB-Integration-Kit")
    assert spec.paths.legacy_repo == Path("/repo/AFDB-Integration-Kit")


def test_load_runspec_rejects_active_container_workdir(tmp_path: Path) -> None:
    data = _active_runspec()
    container = data["container"]
    assert isinstance(container, dict)
    container["workdir"] = "/workspace/custom"

    with pytest.raises(ValidationError, match=r"container\.workdir is removed"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_active_mount_read_only(tmp_path: Path) -> None:
    data = _active_runspec()
    container = data["container"]
    assert isinstance(container, dict)
    mounts = container["mounts"]
    assert isinstance(mounts, list)
    mount = mounts[0]
    assert isinstance(mount, dict)
    mount["read_only"] = True

    with pytest.raises(ValidationError, match=r"container\.mounts\[\]\.read_only is removed"):
        load_runspec(_write_runspec(tmp_path, data))


@pytest.mark.parametrize(
    ("removed_key", "message"),
    [
        ("workflow", "require top-level workflow"),
        ("submission", "require top-level submission"),
    ],
)
def test_load_runspec_rejects_missing_required_active_sections(
    tmp_path: Path,
    removed_key: str,
    message: str,
) -> None:
    data = _active_runspec()
    data.pop(removed_key)

    with pytest.raises(ValidationError, match=message):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_submission_only_active_contract(tmp_path: Path) -> None:
    data = _base_runspec()
    data["submission"] = {"evidence_dir": "/proj/output/evidence/test-run"}

    with pytest.raises(ValidationError, match="require top-level workflow"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_acceptance_only_active_contract(tmp_path: Path) -> None:
    data = _base_runspec()
    data["acceptance"] = {"baseline_output_dir": "/proj/output/baseline"}

    with pytest.raises(ValidationError, match="require top-level workflow"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_validation_expected_count_string(tmp_path: Path) -> None:
    data = _active_runspec()
    validation = data["validation"]
    assert isinstance(validation, dict)
    validation["expected_tar_count"] = "2"

    with pytest.raises(ValidationError, match="expected_tar_count"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_invalid_controller_runtime(tmp_path: Path) -> None:
    data = _active_runspec()
    submission = data["submission"]
    assert isinstance(submission, dict)
    submission["controller_runtime"] = "docker"

    with pytest.raises(ValidationError, match="controller_runtime"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_negative_payload_sample_count(tmp_path: Path) -> None:
    data = _active_runspec()
    acceptance = data["acceptance"]
    assert isinstance(acceptance, dict)
    acceptance["payload_sample_count"] = -1

    with pytest.raises(ValidationError, match=r"payload_sample_count must be non-negative"):
        load_runspec(_write_runspec(tmp_path, data))


@pytest.mark.parametrize(
    ("submission_update", "message"),
    [
        ({"evidence_dir": "/proj/evidence/test-run"}, r"submission\.evidence_dir must be under paths\.output_dir"),
        ({"report_path": "/proj/RUN_REPORT.md"}, r"submission\.report_path must be under paths\.output_dir"),
    ],
)
def test_load_runspec_requires_submission_paths_under_output_dir(
    tmp_path: Path,
    submission_update: dict[str, str],
    message: str,
) -> None:
    data = _active_runspec()
    submission = data["submission"]
    assert isinstance(submission, dict)
    submission.update(submission_update)

    with pytest.raises(ValidationError, match=message):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_parses_phase2_tar_delivery_fields(tmp_path: Path) -> None:
    data = _base_runspec()
    dataset = data["dataset"]
    references = data["references"]
    worker = data["worker"]
    storage = data["storage"]
    assert isinstance(dataset, dict)
    assert isinstance(references, dict)
    assert isinstance(worker, dict)
    assert isinstance(storage, dict)
    dataset["archive_source"] = "staging_dir"
    references["heterodimer_id_manifest"] = "/refs/heterodimer_ids.csv"
    worker.update(
        {
            "heterodimers": True,
            "clash_device": "cuda",
            "clash_batch_size": 128,
            "dssp_algorithm": "pydssp",
            "parallel_stages": False,
            "retry_failed_only": True,
            "retry_metadata_delta_tag": "retry_delta",
        }
    )
    storage.update(
        {
            "upload_mode": "tar",
            "s3_output_prefix": "s3://example-bucket/users/example-user/postprocessed-test/phase2/",
            "s3_tar_prefix": None,
            "s3_tar_manifest_csv": None,
            "local_tar_dir": "/proj/output/local_tars",
            "local_tar_manifest_csv": "/proj/output/local_tars.csv",
            "tar_compression": "zstd-members",
        }
    )
    data["analysis_metadata"] = {
        "enabled": True,
        "csv_path": "/proj/output/analysis_metadata.csv",
        "ipsae_threshold": 0.6,
        "pdockq2_threshold": 0.23,
        "finalize_after_gpu": True,
        "parquet_path": "/proj/output/analysis_metadata.parquet",
        "selected_ids_path": "/proj/output/high_quality_model_ids.txt",
        "chunk_size": 100000,
        "finalize_partition": "cpu_long",
        "finalize_cpus_per_task": 8,
        "finalize_memory": "128G",
        "finalize_time": "24:00:00",
        "high_quality_from_tars": {
            "enabled": True,
            "s3_prefix": "s3://example-bucket/users/example-user/postprocessed-test/phase2_hq/",
            "work_dir": "/tmp/bspp_phase2_hq",
        },
    }

    spec = load_runspec(_write_runspec(tmp_path, data))

    assert spec.dataset.archive_source == "staging_dir"
    assert spec.references.heterodimer_id_manifest == Path("/refs/heterodimer_ids.csv")
    assert spec.worker.heterodimers is True
    assert spec.worker.retry_failed_only is True
    assert spec.worker.retry_metadata_delta_tag == "retry_delta"
    assert spec.storage.upload_mode == "tar"
    assert spec.storage.local_tar_dir == Path("/proj/output/local_tars")
    assert spec.storage.local_tar_manifest_csv == Path("/proj/output/local_tars.csv")
    assert spec.storage.tar_compression == "zstd-members"
    assert spec.analysis_metadata.enabled is True
    assert spec.analysis_metadata.csv_path == Path("/proj/output/analysis_metadata.csv")
    assert spec.analysis_metadata.parquet_path == Path("/proj/output/analysis_metadata.parquet")
    assert spec.analysis_metadata.high_quality_from_tars.enabled is True
    assert spec.analysis_metadata.high_quality_from_tars.work_dir == Path("/tmp/bspp_phase2_hq")


def test_tar_delivery_requires_destination(tmp_path: Path) -> None:
    data = _base_runspec()
    storage = data["storage"]
    assert isinstance(storage, dict)
    storage.update({"upload_mode": "tar", "tar_compression": "zstd-members"})

    with pytest.raises(ValueError, match="tar upload mode requires"):
        load_runspec(_write_runspec(tmp_path, data))


def test_analysis_metadata_requires_csv_path(tmp_path: Path) -> None:
    data = _base_runspec()
    data["analysis_metadata"] = {"enabled": True}

    with pytest.raises(ValueError, match=r"analysis_metadata\.enabled requires"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_accepts_reference_artifact_mapping(tmp_path: Path) -> None:
    data = _base_runspec()
    references = data["references"]
    assert isinstance(references, dict)
    references["master_parquet"] = {
        "path": "/refs/master.parquet",
        "source_uri": "s3://example-bucket/reference/master.parquet",
        "sha256": "a" * 64,
        "min_size_bytes": 1024,
        "version": "2026-02",
    }

    spec = load_runspec(_write_runspec(tmp_path, data))

    assert spec.references.master_parquet == Path("/refs/master.parquet")
    master = spec.references.artifact("master_parquet")
    assert master.path == Path("/refs/master.parquet")
    assert master.source_uri == "s3://example-bucket/reference/master.parquet"
    assert master.sha256 == "a" * 64
    assert master.min_size_bytes == 1024
    assert master.version == "2026-02"


def test_load_runspec_keeps_reference_shorthand_artifacts(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path, _base_runspec()))

    assert spec.references.artifact("master_parquet") == ReferenceArtifact(path=Path("/refs/master.parquet"))
    assert spec.references.tracking_parquet == Path("/refs/tracking.parquet")


def test_load_runspec_rejects_unknown_nested_fields(tmp_path: Path) -> None:
    data = _base_runspec()
    dataset = data["dataset"]
    assert isinstance(dataset, dict)
    dataset["unexpected"] = "ignored-before-pydantic"

    with pytest.raises(ValidationError, match="unexpected"):
        load_runspec(_write_runspec(tmp_path, data))


def test_load_runspec_rejects_wrong_primitive_types(tmp_path: Path) -> None:
    data = _base_runspec()
    worker = data["worker"]
    assert isinstance(worker, dict)
    worker["workers"] = "24"

    with pytest.raises(ValidationError, match="workers"):
        load_runspec(_write_runspec(tmp_path, data))


def test_render_dry_run_redacts_secret_values(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path, _base_runspec()))
    statuses = spec.resolve_secrets(
        environ={
            "AWS_ACCESS_KEY_ID": "id-value",
            "AWS_SECRET_ACCESS_KEY": "secret-value",
            "S3_ENDPOINT_URL": "https://s3.example.test",
        }
    )

    rendered = render_dry_run(spec, statuses)

    assert "env:<redacted>" in rendered
    assert "secret-value" not in rendered
    assert "0-0" in rendered
    assert "s3://example-bucket/users/example-user/postprocessed-test/20251128_full_run_v2/" in rendered


def test_validate_required_secrets_fails_strictly_without_changing_load(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path, _base_runspec()))

    with pytest.raises(MissingSecretError, match="Missing required secrets"):
        spec.validate_required_secrets(environ={})


def test_secret_execution_env_exposes_keys_without_repr_values(tmp_path: Path) -> None:
    secret_file = tmp_path / "gcs.json"
    secret_file.write_text("file-secret")
    secret_file.chmod(0o600)
    data = _base_runspec()
    secrets = data["secrets"]
    assert isinstance(secrets, dict)
    secrets["gcs_credentials_ref"] = f"file:{secret_file}"
    spec = load_runspec(_write_runspec(tmp_path, data))

    overlay = spec.secret_execution_env(
        environ={
            "AWS_ACCESS_KEY_ID": "id-value",
            "AWS_SECRET_ACCESS_KEY": "secret-value",
            "S3_ENDPOINT_URL": "https://s3.example.test",
        }
    )

    assert overlay.keys() == (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "S3_ENDPOINT_URL",
    )
    assert overlay.as_mapping()["AWS_SECRET_ACCESS_KEY"] == "secret-value"
    assert overlay.as_mapping()["GOOGLE_APPLICATION_CREDENTIALS"] == str(secret_file.resolve())
    assert "secret-value" not in repr(overlay)
    assert "file-secret" not in repr(overlay)


def test_legacy_swiftstack_storage_keys_are_rejected(tmp_path: Path) -> None:
    """Legacy SwiftStack-shaped storage keys fail closed (no alias acceptance)."""
    data = _base_runspec()
    storage = data["storage"]
    assert isinstance(storage, dict)
    storage["swiftstack_archive_prefix"] = "s3://example-bucket/structures/"
    storage["swiftstack_output_prefix"] = "s3://example-bucket/users/example-user/postprocessed-test/legacy/"

    with pytest.raises(ValidationError):
        load_runspec(_write_runspec(tmp_path, data))


def test_s3_storage_keys_load_and_expose_renamed_fields(tmp_path: Path) -> None:
    data = _base_runspec()
    spec = load_runspec(_write_runspec(tmp_path, data))

    assert spec.storage.s3_archive_prefix == "s3://example-bucket/structures/"
    assert spec.storage.s3_output_prefix.startswith("s3://example-bucket/users/example-user/postprocessed-test/")


def test_legacy_swiftstack_secret_key_is_not_a_default_env_target() -> None:
    """Legacy ``bspp/swiftstack`` / ``swiftstack`` secret keys are no longer default env targets."""
    assert "bspp/swiftstack" not in _DEFAULT_ENV_VARS
    assert "swiftstack" not in _DEFAULT_ENV_VARS
    assert "bspp/s3" in _DEFAULT_ENV_VARS
    assert "s3" in _DEFAULT_ENV_VARS


def test_legacy_swiftstack_secret_target_is_rejected(tmp_path: Path) -> None:
    """Legacy SwiftStack secret targets fail closed even though they are valid SecretRefs."""
    for legacy_target in ("bspp/swiftstack", "swiftstack"):
        data = _base_runspec()
        secrets = data["secrets"]
        assert isinstance(secrets, dict)
        secrets["s3_credentials_ref"] = f"env:{legacy_target}"

        with pytest.raises(ValidationError, match="Legacy secret target"):
            load_runspec(_write_runspec(tmp_path, data))


def test_s3_secret_targets_still_load(tmp_path: Path) -> None:
    for target in ("bspp/s3", "s3"):
        data = _base_runspec()
        secrets = data["secrets"]
        assert isinstance(secrets, dict)
        secrets["s3_credentials_ref"] = f"env:{target}"

        spec = load_runspec(_write_runspec(tmp_path, data))
        assert spec.secrets.s3_credentials_ref.target == target


def test_hq_chunk_publication_resolves_explicit_nonproduction_prefix(tmp_path: Path) -> None:
    data = _base_runspec()
    data["analysis_metadata"] = {
        "high_quality_from_tars": {
            "enabled": True,
            "s3_prefix": "s3://example-bucket/postprocessed/legacy-flat-contract/",
            "publication": {
                "enabled": True,
                "target_prefix": "s3://example-bucket/users/example-user/hq-chunks-canary",
                "overwrite": True,
            },
        }
    }

    spec = load_runspec(_write_runspec(tmp_path, data))

    assert resolve_hq_chunk_publication_target(spec) == "s3://example-bucket/users/example-user/hq-chunks-canary/"
    assert spec.analysis_metadata.high_quality_from_tars.publication.overwrite is True


def test_hq_chunk_publication_requires_explicit_target_prefix(tmp_path: Path) -> None:
    data = _base_runspec()
    data["analysis_metadata"] = {"high_quality_from_tars": {"publication": {"enabled": True}}}

    with pytest.raises(ValueError, match="requires target_prefix"):
        load_runspec(_write_runspec(tmp_path, data))


def test_hq_chunk_publication_does_not_fall_back_to_flat_s3_prefix(tmp_path: Path) -> None:
    data = _base_runspec()
    data["analysis_metadata"] = {
        "high_quality_from_tars": {
            "s3_prefix": "s3://example-bucket/users/example-user/flat-hq/",
            "publication": {"enabled": True, "default_target_prefix_from_recipe": False},
        }
    }

    with pytest.raises(ValueError, match="requires target_prefix"):
        load_runspec(_write_runspec(tmp_path, data))


def test_gcs_destination_requires_secret_ref(tmp_path: Path) -> None:
    data = _base_runspec()
    storage = data["storage"]
    assert isinstance(storage, dict)
    storage["gcs_destination_prefix"] = "gs://isolated-test/postprocessed/20251128_full_run_v2/"

    with pytest.raises(ValueError, match="requires a GCS credentials reference"):
        load_runspec(_write_runspec(tmp_path, data))


def test_runspec_validate_cli_prints_redacted_dry_run(tmp_path: Path) -> None:
    path = _write_runspec(tmp_path, _base_runspec())
    runner = CliRunner()

    result = runner.invoke(cli, ["runspec", "validate", str(path)])

    assert result.exit_code == 0
    assert "RunSpec dry run" in result.output
    assert "env:<redacted>" in result.output
    assert "AWS_SECRET_ACCESS_KEY" in result.output


def test_render_dry_run_includes_enriched_run_fields(tmp_path: Path) -> None:
    data = _base_runspec()
    references = data["references"]
    assert isinstance(references, dict)
    references["manifest_csv"] = {
        "path": "/refs/manifest.csv",
        "source_uri": "s3://example-bucket/reference/manifest.csv",
        "min_size_bytes": 512,
    }
    spec = load_runspec(_write_runspec(tmp_path, data))

    rendered = render_dry_run(spec, ())

    assert "references:" in rendered
    assert "manifest_csv: path=/refs/manifest.csv, source=s3://example-bucket/reference/manifest.csv" in rendered
    assert "container_mounts:" in rendered
    assert "/repo/AFDB-Integration-Kit -> /workspace/AFDB-Integration-Kit (ro)" in rendered
    assert "worker:" in rendered
    assert "shards_per_archive: 2" in rendered
    assert "s5cmd_numworkers: 32" in rendered
    assert "duckdb_memory_limit: 1GB" in rendered
    assert "provider_copyrights: ['Copyright BSPP Orchestration contributors. All rights reserved.']" in rendered
    assert "s3_archive_prefix: s3://example-bucket/structures/" in rendered
    assert "validation_counts:" in rendered
    assert "expected_allowed_ids: 370679" in rendered


def test_render_artifact_header_marks_generated_files(tmp_path: Path) -> None:
    path = _write_runspec(tmp_path, _base_runspec())
    spec = load_runspec(path)

    header = render_artifact_header(spec, "recipe/config.yaml")

    assert "Generated artifact: recipe/config.yaml" in header
    assert "DO NOT EDIT" in header
    assert str(path) in header
    assert spec.source_hash is not None
    assert spec.source_hash in header
