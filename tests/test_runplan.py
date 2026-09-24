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

"""Tests for user-authored Run Plan loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bspp.orchestration.contract.runplan import load_runplan


def test_load_runplan_parses_single_cluster_plan_and_env_secret_ref(tmp_path: Path) -> None:
    plan = load_runplan(_write_runplan(tmp_path, _runplan_data()))

    assert plan.run_kind == "dev"
    assert plan.target_cluster == "example-cluster"
    assert plan.workflow_template == "workflow.yaml"
    assert plan.dataset.run_id == "run1"
    assert plan.references.master_parquet == Path("/refs/master.parquet")
    assert plan.secrets.s3_credentials_ref.scheme == "env"
    assert plan.secrets.s3_credentials_ref.target == "bspp/s3"


def test_load_runplan_parses_data_placement_mover_selections(tmp_path: Path) -> None:
    data = _runplan_data()
    data["data_placement"] = {
        "inputs": {"tool": "dm", "source": "s3://example-bucket/structures/", "destination": "/staging/ds1"},
        "s3": {"tool": "dm"},
        "gcs": {"tool": "gcloud"},
        "publication": {"tool": "s5cmd"},
    }

    plan = load_runplan(_write_runplan(tmp_path, data))

    assert plan.data_placement is not None
    assert plan.data_placement.inputs is not None
    assert plan.data_placement.inputs.tool == "dm"
    assert plan.data_placement.s3 is not None
    assert plan.data_placement.s3.tool == "dm"
    assert plan.data_placement.gcs is not None
    assert plan.data_placement.gcs.tool == "gcloud"


def test_load_runplan_rejects_data_placement_tool_for_wrong_stage(tmp_path: Path) -> None:
    data = _runplan_data()
    data["data_placement"] = {"gcs": {"tool": "s5cmd"}}

    with pytest.raises(ValidationError, match="Tool 's5cmd' is not supported for stage 'gcs'"):
        load_runplan(_write_runplan(tmp_path, data))


def test_load_runplan_rejects_unknown_run_kind(tmp_path: Path) -> None:
    data = _runplan_data()
    data["run_kind"] = "staging"

    with pytest.raises(ValidationError, match="run_kind"):
        load_runplan(_write_runplan(tmp_path, data))


def test_load_runplan_rejects_missing_run_kind(tmp_path: Path) -> None:
    data = _runplan_data()
    data.pop("run_kind")

    with pytest.raises(ValidationError, match="run_kind"):
        load_runplan(_write_runplan(tmp_path, data))


@pytest.mark.parametrize(
    "update",
    [
        {"targets": ["example-cluster", "example-cluster"]},
        {"target_clusters": ["example-cluster", "example-cluster"]},
        {"target_cluster": ["example-cluster", "example-cluster"]},
    ],
)
def test_load_runplan_rejects_multi_cluster_shapes(tmp_path: Path, update: dict[str, object]) -> None:
    data = _runplan_data()
    data.update(update)

    with pytest.raises(ValidationError, match="single target_cluster"):
        load_runplan(_write_runplan(tmp_path, data))


def test_load_runplan_rejects_environment_interpolation_but_not_env_secret_refs(tmp_path: Path) -> None:
    data = _runplan_data()
    dataset = data["dataset"]
    assert isinstance(dataset, dict)
    dataset["name"] = "$DATASET"

    with pytest.raises(ValueError, match=r"environment interpolation.*dataset\.name"):
        load_runplan(_write_runplan(tmp_path, data))


def _write_runplan(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "run-plan.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _runplan_data() -> dict[str, object]:
    return {
        "run_kind": "dev",
        "target_cluster": "example-cluster",
        "workflow_template": "workflow.yaml",
        "dataset": {
            "name": "ds1",
            "run_id": "run1",
            "mode": "archive",
            "array": "0-3",
            "archive_source": "tracking",
        },
        "references": {
            "master_parquet": "/refs/master.parquet",
            "tracking_parquet": "/refs/tracking.parquet",
            "manifest_csv": "/refs/manifest.csv",
            "uniprot_duckdb": "/refs/uniprot.duckdb",
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
            "gcs_dry_run_only": True,
        },
        "acceptance": {
            "baseline_output_dir": "/baseline/output",
            "baseline_run_name": "baseline",
            "candidate_run_name": "candidate",
            "payload_sample_count": 20,
        },
        "secrets": {"s3_credentials_ref": "env:bspp/s3", "gcs_credentials_ref": None},
    }
