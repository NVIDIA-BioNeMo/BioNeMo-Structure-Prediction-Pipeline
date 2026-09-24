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

"""Shared helpers for active workflow RunSpec tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from bspp.orchestration.contract.runspec import RunSpec, load_runspec


def workflow_runspec_data(tmp_path: Path) -> dict[str, object]:
    """Return a concrete active workflow RunSpec mapping with no path probing assumptions."""
    project_root = tmp_path / "project"
    output_dir = project_root / "output"
    afdb_toolkit_repo = tmp_path / "AFDB-Integration-Kit"
    orchestration_repo = tmp_path / "bspp-orchestration"
    return {
        "dataset": {
            "name": "ds1",
            "run_id": "run1",
            "mode": "archive",
            "array": "0-0",
            "archive_source": "tracking",
        },
        "cluster": {"name": "example-cluster", "account": "acct", "owner": "tester"},
        "paths": {
            "project_root": str(project_root),
            "staging_dir": str(project_root / "input" / "staging"),
            "output_dir": str(output_dir),
            "log_dir": str(output_dir / "logs"),
            "afdb_toolkit_repo": str(afdb_toolkit_repo),
            "orchestration_repo": str(orchestration_repo),
        },
        "references": {
            "master_parquet": str(tmp_path / "refs" / "master.parquet"),
            "tracking_parquet": str(tmp_path / "refs" / "tracking.parquet"),
            "manifest_csv": str(tmp_path / "refs" / "manifest.csv"),
            "uniprot_duckdb": str(tmp_path / "refs" / "uniprot.duckdb"),
        },
        "container": {
            "image": str(tmp_path / "containers" / "bspp-orchestration.sqsh"),
            "mounts": [
                {
                    "source": str(orchestration_repo),
                    "target": "/workspace/bspp-orchestration",
                },
                {
                    "source": str(afdb_toolkit_repo),
                    "target": "/workspace/AFDB-Integration-Kit",
                },
                {
                    "source": str(project_root),
                    "target": str(project_root),
                },
            ],
        },
        "resources": {
            "gpu_worker": {
                "partition": "gpu",
                "cpus_per_task": 30,
                "memory": "128G",
                "time": "04:00:00",
                "gres": "gpu:1",
                "array": "0-0",
            },
            "analysis_finalize": {
                "partition": "cpu",
                "cpus_per_task": 4,
                "memory": "32G",
                "time": "00:30:00",
            },
            "acceptance_tar_payload_parity": {
                "partition": "cpu",
                "cpus_per_task": 8,
                "memory": "32G",
                "time": "00:30:00",
            },
            "acceptance_semantic": {
                "partition": "cpu",
                "cpus_per_task": 4,
                "memory": "16G",
                "time": "00:20:00",
            },
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
            "gcs_destination_prefix": None,
            "allow_production_prefixes": False,
        },
        "validation": {
            "expected_archives": 1,
            "expected_tar_count": 1,
            "expected_local_tars_rows": 1,
            "expected_failed_rows": 0,
            "expected_analysis_rows": 10,
            "expected_selected_ids": 2,
            "gcs_dry_run_only": True,
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
            ]
        },
        "submission": {
            "evidence_dir": str(output_dir / "evidence" / "run1"),
            "report_path": str(output_dir / "RUN_REPORT.md"),
            "controller_runtime": "enroot",
        },
        "acceptance": {
            "baseline_output_dir": str(output_dir / "baseline"),
            "baseline_run_name": "baseline",
            "candidate_run_name": "candidate",
            "payload_sample_count": 20,
        },
        "secrets": {"s3_credentials_ref": "env:bspp/s3", "gcs_credentials_ref": None},
    }


def write_workflow_runspec(tmp_path: Path, data: dict[str, object]) -> Path:
    """Write a workflow RunSpec mapping to a temporary YAML file."""
    path = tmp_path / "runspec.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def load_workflow_runspec(tmp_path: Path, data: dict[str, object] | None = None) -> RunSpec:
    """Load a workflow RunSpec from a temporary YAML file."""
    return load_runspec(write_workflow_runspec(tmp_path, data or workflow_runspec_data(tmp_path)))


def workflow_runspec_data_baked(tmp_path: Path) -> dict[str, object]:
    """Return a concrete active workflow RunSpec mapping with no toolkit path (baked mode)."""
    data = workflow_runspec_data(tmp_path)
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


def load_workflow_runspec_baked(tmp_path: Path) -> RunSpec:
    """Load a baked-mode workflow RunSpec from a temporary YAML file."""
    return load_runspec(write_workflow_runspec(tmp_path, workflow_runspec_data_baked(tmp_path)))
