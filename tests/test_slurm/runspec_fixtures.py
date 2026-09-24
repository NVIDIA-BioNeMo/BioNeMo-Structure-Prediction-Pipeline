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

"""Shared RunSpec fixtures for SLURM rendering tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def _write_runspec(
    tmp_path: Path,
    *,
    array: str = "0-1",
    resource_array: str | None = None,
    workers: int = 24,
    s5cmd_path: str = "/bin/s5cmd",
    upload_slots: int = 4,
    s5cmd_numworkers: int | None = None,
    duckdb_memory_limit: str | None = None,
    phase2_tar: bool = False,
    self_upload: bool = True,
    s3_tar_prefix: bool = True,
) -> Path:
    tracking = tmp_path / "tracking.parquet"
    manifest = tmp_path / "manifest.csv"
    uniprot = tmp_path / "uniprot.duckdb"
    manifest.write_text("model_entity_id,chain_id,uniprot_ac\n")
    uniprot.write_text("duckdb")
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "dataset_name": ["ds1", "ds1"],
                "swiftstack_archive": ["archive_a", "archive_b"],
            }
        ),
        tracking,
    )
    references: dict[str, object] = {
        "master_parquet": str(tracking),
        "tracking_parquet": str(tracking),
        "manifest_csv": str(manifest),
        "uniprot_duckdb": str(uniprot),
    }
    worker: dict[str, object] = {
        "stages": "metadata_export",
        "workers": workers,
        "batch_size": 500,
        "shards_per_archive": 2,
        "self_upload": self_upload,
        "local_scratch": True,
        "scratch_dir": "/dev/shm",
        "s5cmd_path": s5cmd_path,
        "upload_slots": upload_slots,
        **({"s5cmd_numworkers": s5cmd_numworkers} if s5cmd_numworkers is not None else {}),
        **({"duckdb_memory_limit": duckdb_memory_limit} if duckdb_memory_limit is not None else {}),
    }
    storage: dict[str, object] = {
        "s3_archive_prefix": "s3://example-bucket/structures/",
        "s3_output_prefix": "s3://example-bucket/users/test/postprocessed/ds1/",
        "gcs_destination_prefix": None,
        "allow_production_prefixes": False,
    }
    analysis_metadata: dict[str, object] | None = None
    if phase2_tar:
        (tmp_path / "heterodimer_ids.csv").write_text("model_entity_id,entity_id,chain_id,uniprot_ac\n")
        references["heterodimer_id_manifest"] = str(tmp_path / "heterodimer_ids.csv")
        worker.update(
            {
                "heterodimers": True,
                "retry_failed_only": True,
                "retry_metadata_delta_tag": "retry_delta",
            }
        )
        storage.update(
            {
                "upload_mode": "tar",
                **(
                    {"s3_tar_prefix": "s3://example-bucket/users/test/postprocessed/ds1/tars/"} if s3_tar_prefix else {}
                ),
                "s3_tar_manifest_csv": str(tmp_path / "uploaded_tars.csv"),
                "local_tar_dir": str(tmp_path / "local-tars"),
                "local_tar_manifest_csv": str(tmp_path / "local-tars" / "manifest.csv"),
                "tar_compression": "zstd-members",
            }
        )
        analysis_metadata = {
            "enabled": True,
            "csv_path": str(tmp_path / "analysis_metadata.csv"),
            "ipsae_threshold": 0.6,
            "pdockq2_threshold": 0.23,
            "finalize_after_gpu": True,
            "parquet_path": str(tmp_path / "analysis_metadata.parquet"),
            "selected_ids_path": str(tmp_path / "high_quality_model_ids.txt"),
        }
    data: dict[str, object] = {
        "dataset": {"name": "ds1", "run_id": "run", "mode": "archive", "array": array},
        "cluster": {"name": "example-cluster", "account": "acct", "owner": "tester"},
        "paths": {
            "project_root": str(tmp_path),
            "staging_dir": str(tmp_path / "input" / "staging"),
            "output_dir": str(tmp_path / "output"),
            "log_dir": str(tmp_path / "output" / "logs"),
            "legacy_repo": str(tmp_path / "AFDB-Integration-Kit"),
            "afdb_toolkit_repo": str(tmp_path / "AFDB-Integration-Kit"),
            "orchestration_repo": str(tmp_path / "bspp-orchestration"),
        },
        "references": references,
        "container": {
            "image": "image.sqsh",
            "workdir": "/workspace/bspp-orchestration",
            "mounts": [
                {
                    "source": str(tmp_path / "AFDB-Integration-Kit"),
                    "target": "/workspace/afdb-toolkit",
                    "read_only": True,
                },
                {
                    "source": str(tmp_path / "bspp-orchestration"),
                    "target": "/workspace/bspp-orchestration",
                    "read_only": False,
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
                "array": resource_array if resource_array is not None else array,
            }
        },
        "worker": worker,
        "storage": storage,
        "validation": {"expected_archives": 2},
        "secrets": {"s3_credentials_ref": "env:bspp/s3", "gcs_credentials_ref": None},
    }
    if analysis_metadata is not None:
        data["analysis_metadata"] = analysis_metadata
    path = tmp_path / "runspec.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _write_finalizer_runspec(tmp_path: Path, *, high_quality: bool = False) -> Path:
    legacy = tmp_path / "AFDB-Integration-Kit"
    orchestration = tmp_path / "bspp-orchestration"
    legacy.mkdir()
    orchestration.mkdir()
    storage: dict[str, object] = {
        "s3_archive_prefix": "s3://example-bucket/structures/",
        "s3_output_prefix": "s3://example-bucket/users/test/postprocessed/ds1/",
        "gcs_destination_prefix": None,
        "upload_mode": "tar",
        "s3_tar_prefix": "s3://example-bucket/users/test/postprocessed/ds1/tars/",
        "s3_tar_manifest_csv": str(tmp_path / "output" / "uploaded_tars.csv"),
        "tar_compression": "zstd-members",
        "allow_production_prefixes": False,
    }
    high_quality_from_tars = {
        "enabled": high_quality,
        "s3_prefix": "s3://example-bucket/users/test/postprocessed/ds1_hq/" if high_quality else None,
        "work_dir": str(tmp_path / "hq-work") if high_quality else None,
    }
    data: dict[str, object] = {
        "dataset": {"name": "ds1", "run_id": "run", "mode": "archive", "array": "0-0"},
        "cluster": {"name": "example-cluster", "account": "acct", "owner": "tester"},
        "paths": {
            "project_root": str(tmp_path),
            "staging_dir": str(tmp_path / "input" / "staging"),
            "output_dir": str(tmp_path / "output"),
            "log_dir": str(tmp_path / "output" / "logs"),
            "legacy_repo": str(legacy),
            "orchestration_repo": str(orchestration),
        },
        "references": {
            "master_parquet": str(tmp_path / "master.parquet"),
            "tracking_parquet": str(tmp_path / "tracking.parquet"),
            "manifest_csv": str(tmp_path / "manifest.csv"),
            "uniprot_duckdb": str(tmp_path / "uniprot.duckdb"),
        },
        "container": {
            "image": "image.sqsh",
            "workdir": "/workspace/bspp-orchestration",
            "mounts": [
                {"source": str(legacy), "target": "/workspace/afdb-toolkit", "read_only": True},
                {"source": str(orchestration), "target": "/workspace/bspp-orchestration", "read_only": False},
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
                "partition": "cpu_long",
                "cpus_per_task": 8,
                "memory": "128G",
                "time": "24:00:00",
            },
        },
        "worker": {
            "stages": "metadata_export",
            "workers": 24,
            "batch_size": 500,
            "shards_per_archive": 2,
            "self_upload": True,
            "local_scratch": True,
            "scratch_dir": "/dev/shm",
            "s5cmd_path": "/bin/s5cmd",
            "upload_slots": 4,
        },
        "storage": storage,
        "analysis_metadata": {
            "enabled": True,
            "csv_path": str(tmp_path / "output" / "analysis_metadata.csv"),
            "ipsae_threshold": 0.6,
            "pdockq2_threshold": 0.23,
            "finalize_after_gpu": True,
            "parquet_path": str(tmp_path / "output" / "analysis_metadata.parquet"),
            "selected_ids_path": str(tmp_path / "output" / "high_quality_model_ids.txt"),
            "chunk_size": 100000,
            "high_quality_from_tars": high_quality_from_tars,
        },
        "validation": {"expected_archives": 2},
        "secrets": {"s3_credentials_ref": "env:bspp/s3", "gcs_credentials_ref": None},
    }
    path = tmp_path / "runspec.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
