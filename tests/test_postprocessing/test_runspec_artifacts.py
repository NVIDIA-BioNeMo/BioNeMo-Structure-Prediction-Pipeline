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

"""Tests for RunSpec-driven archive preprocess artifact rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from pytest import MonkeyPatch

from bspp.orchestration.contract.runspec import load_runspec
from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.inputs.archives import archives_for_dataset
from bspp.orchestration.runtime.postprocessing import runspec_artifacts as runspec_artifacts_module
from bspp.orchestration.runtime.postprocessing.runspec_artifacts import (
    build_recipe_config,
    normalized_archive_names,
    render_archive_preprocess_artifacts,
)


class _Overlay:
    def __init__(self, base: object, **overrides: object) -> None:
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> object:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def test_normalized_archive_names_adds_suffix_and_sorts() -> None:
    assert normalized_archive_names(["b", "a.tar.lz4", "b"]) == ("a.tar.lz4", "b.tar.lz4")


def test_build_recipe_config_uses_runspec_and_full_archive_array(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    coverage = archives_for_dataset(spec.references.tracking_parquet, spec.dataset.name)

    config = build_recipe_config(spec, coverage)

    assert config["cluster"] == "example-cluster"
    assert config["slurm"]["array_range"] == "0-1"
    assert config["paths"]["output_dir"] == str(spec.paths.output_dir)
    assert config["paths"]["manifest_csv"] == str(spec.references.manifest_csv)
    assert config["upload"]["s3_prefix"] == spec.storage.s3_output_prefix
    assert config["worker"]["batch_size"] == spec.worker.batch_size


def test_build_recipe_config_emits_tar_delivery_and_analysis_metadata(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    spec_with_tar_fields = _Overlay(
        spec,
        references=_Overlay(
            spec.references,
            heterodimer_id_manifest=tmp_path / "heterodimer_ids.csv",
        ),
        storage=_Overlay(
            spec.storage,
            upload_mode="tar",
            s3_tar_prefix="s3://example-bucket/users/test/postprocessed/ds1/tars/",
            s3_tar_manifest_csv=tmp_path / "uploaded_tars.csv",
            local_tar_dir=tmp_path / "tar-output",
            local_tar_manifest_csv=tmp_path / "tar-output" / "manifest.csv",
            tar_compression="zstd-members",
        ),
        analysis_metadata={
            "high_quality_from_tars": True,
            "quality_source": "tar-manifest",
        },
    )
    coverage = archives_for_dataset(spec.references.tracking_parquet, spec.dataset.name)

    config = build_recipe_config(spec_with_tar_fields, coverage)  # type: ignore[arg-type]

    assert config["upload"]["mode"] == "tar"
    assert config["upload"]["tar_prefix"] == "s3://example-bucket/users/test/postprocessed/ds1/tars/"
    assert config["upload"]["tar_manifest_csv"] == str(tmp_path / "uploaded_tars.csv")
    assert config["upload"]["local_tar_dir"] == str(tmp_path / "tar-output")
    assert config["upload"]["local_tar_manifest_csv"] == str(tmp_path / "tar-output" / "manifest.csv")
    assert config["upload"]["tar_compression"] == "zstd-members"
    assert config["analysis_metadata"] == {
        "high_quality_from_tars": True,
        "quality_source": "tar-manifest",
    }
    assert config["paths"]["heterodimer_id_manifest"] == str(tmp_path / "heterodimer_ids.csv")


def test_render_archive_preprocess_artifacts_dry_run_writes_nothing(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    coverage = archives_for_dataset(spec.references.tracking_parquet, spec.dataset.name)

    plan = render_archive_preprocess_artifacts(spec, coverage, dry_run=True)

    assert plan.dry_run is True
    assert plan.archives == ("archive_a.tar.lz4", "archive_b.tar.lz4")
    assert plan.array_range == "0-1"
    assert not (spec.paths.output_dir / "archive_list.json").exists()
    assert plan.allowlists is not None
    assert plan.allowlists.total_ids == 3


def test_load_allowlist_ids_prunes_columns_and_filters_dataset(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    tracking = tmp_path / "tracking.parquet"
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1", "ds1", "other"],
                "swiftstack_archive": ["archive_a", "archive_b", "archive_z"],
                "model_entity_id": [
                    "AF-0000000000000001",
                    "AF-0000000000000002",
                    "AF-0000000000000099",
                ],
                "large_unused_payload": ["x" * 100, "y" * 100, "z" * 100],
            }
        ),
        tracking,
    )
    calls: list[dict[str, object]] = []
    original_read_table = runspec_artifacts_module.pq.read_table

    def spy_read_table(*args: object, **kwargs: object) -> pa.Table:
        calls.append(kwargs)
        return original_read_table(*args, **kwargs)

    monkeypatch.setattr(runspec_artifacts_module.pq, "read_table", spy_read_table)

    ids = runspec_artifacts_module._load_allowlist_ids(tracking, "ds1")

    assert ids == {
        "archive_a": {"AF-0000000000000001"},
        "archive_b": {"AF-0000000000000002"},
    }
    assert calls == [
        {
            "columns": ["dataset_name", "swiftstack_archive", "model_entity_id"],
            "filters": [("dataset_name", "=", "ds1")],
        }
    ]


def test_render_archive_preprocess_artifacts_execute_writes_compatible_files(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    coverage = archives_for_dataset(spec.references.tracking_parquet, spec.dataset.name)

    plan = render_archive_preprocess_artifacts(spec, coverage, dry_run=False)

    assert plan.dry_run is False
    archive_list = json.loads((spec.paths.output_dir / "archive_list.json").read_text())
    assert archive_list == ["archive_a.tar.lz4", "archive_b.tar.lz4"]
    shard_config = json.loads((spec.paths.output_dir / "shard_config.json").read_text())
    assert shard_config["array_range"] == "0-1"
    assert shard_config["has_allowlists"] is True
    assert json.loads((spec.paths.output_dir / "dataset_config.json").read_text())["providerId"] == "NVDA"
    provider = json.loads((spec.paths.output_dir / "provider.json").read_text())
    assert provider["providerName"] == "NVIDIA"
    assert provider["copyrights"] == ["Copyright 2024 NVIDIA. All rights reserved."]
    assert (spec.paths.output_dir / "allowlists" / "archive_a.tar.lz4.txt").read_text().splitlines() == [
        "AF-0000000000000001",
        "AF-0000000000000002",
    ]
    recipe_text = (spec.paths.recipe_dir / "config.yaml").read_text()
    assert "DO NOT EDIT" in recipe_text
    recipe = yaml.safe_load(recipe_text)
    assert recipe["run_name"] == "ds1"


def test_runspec_render_preprocess_cli_dry_run(tmp_path: Path) -> None:
    spec_path = _write_runspec(tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["runspec", "render-preprocess", str(spec_path)])

    assert result.exit_code == 0
    rendered = json.loads(result.output)
    assert rendered["plan"]["archive_count"] == 2
    assert rendered["plan"]["array_range"] == "0-1"
    assert rendered["plan"]["allowlists"]["total_ids"] == 3


def test_runspec_render_preprocess_cli_uses_staging_archive_source(tmp_path: Path) -> None:
    spec_path = _write_runspec(tmp_path, archive_source="staging_dir")
    staging = tmp_path / "input" / "staging"
    staging.mkdir(parents=True)
    (staging / "staged_b.tar.lz4").write_text("b")
    (staging / "staged_a.tar.lz4").write_text("a")
    runner = CliRunner()

    result = runner.invoke(cli, ["runspec", "render-preprocess", str(spec_path), "--no-allowlists"])

    assert result.exit_code == 0
    rendered = json.loads(result.output)
    assert rendered["plan"]["archives"] == ["staged_a.tar.lz4", "staged_b.tar.lz4"]
    assert rendered["plan"]["archive_count"] == 2
    assert rendered["plan"]["array_range"] == "0-1"


def test_runspec_render_preprocess_fails_on_all_empty_allowlists(tmp_path: Path) -> None:
    spec_path = _write_runspec(tmp_path, archive_source="staging_dir")
    staging = tmp_path / "input" / "staging"
    staging.mkdir(parents=True)
    (staging / "staged_a.tar.lz4").write_text("a")

    result = CliRunner().invoke(cli, ["runspec", "render-preprocess", str(spec_path)])

    assert result.exit_code != 0
    assert "rendered allowlists contain zero IDs for every archive" in result.output


def test_render_archive_preprocess_artifacts_does_not_write_all_empty_allowlists(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path, archive_source="staging_dir"))
    staging = tmp_path / "input" / "staging"
    staging.mkdir(parents=True)
    (staging / "staged_a.tar.lz4").write_text("a")

    with pytest.raises(ValueError, match="rendered allowlists contain zero IDs"):
        render_archive_preprocess_artifacts(spec, ["staged_a.tar.lz4"], dry_run=False)

    assert not (spec.paths.output_dir / "allowlists").exists()


def _write_runspec(tmp_path: Path, *, archive_source: str = "tracking") -> Path:
    tracking = tmp_path / "tracking.parquet"
    master = tmp_path / "master.parquet"
    manifest = tmp_path / "manifest.csv"
    uniprot = tmp_path / "uniprot.duckdb"
    manifest.write_text("model_entity_id,chain_id,uniprot_ac\n")
    uniprot.write_text("duckdb")
    table = pa.table(
        {
            "dataset_name": ["ds1", "ds1", "ds1", "other"],
            "swiftstack_archive": ["archive_b", "archive_a", "archive_a", "archive_z"],
            "model_entity_id": [
                "AF-0000000000000003",
                "AF-0000000000000001",
                "AF-0000000000000002",
                "AF-0000000000000099",
            ],
        }
    )
    pq.write_table(table, tracking)
    pq.write_table(table, master)
    data = {
        "dataset": {
            "name": "ds1",
            "run_id": "run",
            "mode": "archive",
            "array": "0-0",
            "archive_source": archive_source,
        },
        "cluster": {"name": "example-cluster", "account": "acct", "owner": "tester"},
        "paths": {
            "project_root": str(tmp_path),
            "staging_dir": str(tmp_path / "input" / "staging"),
            "output_dir": str(tmp_path / "output"),
            "log_dir": str(tmp_path / "output" / "logs"),
            "legacy_repo": str(tmp_path / "AFDB-Integration-Kit"),
            "orchestration_repo": str(tmp_path / "bspp-orchestration"),
            "recipe_dir": str(tmp_path / "recipe"),
        },
        "references": {
            "master_parquet": str(master),
            "tracking_parquet": str(tracking),
            "manifest_csv": str(manifest),
            "uniprot_duckdb": str(uniprot),
        },
        "container": {"image": "image", "workdir": "/work", "mounts": []},
        "resources": {
            "gpu_worker": {
                "partition": "gpu",
                "cpus_per_task": 30,
                "memory": "128G",
                "time": "04:00:00",
                "gres": "gpu:1",
                "array": "0-0",
            }
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
            "tool_used": "ColabFold v1.6.0 / AlphaFold-Multimer",
            "provider_id": "NVDA",
            "provider_name": "NVIDIA",
            "provider_url": "https://alphafold.ebi.ac.uk",
            "provider_copyrights": ["Copyright 2024 NVIDIA. All rights reserved."],
        },
        "storage": {
            "s3_archive_prefix": "s3://example-bucket/structures/",
            "s3_output_prefix": "s3://example-bucket/users/test/postprocessed/ds1/",
            "gcs_destination_prefix": None,
            "allow_production_prefixes": False,
        },
        "validation": {"expected_archives": 2, "expected_allowed_ids": 3},
        "secrets": {"s3_credentials_ref": "env:bspp/s3", "gcs_credentials_ref": None},
    }
    path = tmp_path / "runspec.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


# --- pdb-assembly allowlist extraction tests (additive) ---


def test_load_allowlist_ids_extracts_pdb_assembly_from_msa_path_bare(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking.parquet"
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1", "ds1"],
                "swiftstack_archive": ["archive_a", "archive_a"],
                "msa_path": ["pdb_5snm_assembly_1", "pdb_6snm_assembly_2"],
            }
        ),
        tracking,
    )
    ids = runspec_artifacts_module._load_allowlist_ids(tracking, "ds1")
    assert ids == {"archive_a": {"pdb_5snm_assembly_1", "pdb_6snm_assembly_2"}}


def test_load_allowlist_ids_extracts_pdb_assembly_from_full_path_msa_path(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking.parquet"
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1"],
                "swiftstack_archive": ["archive_a"],
                "msa_path": ["/data/corpus/inputs/pdb_5snm_assembly_1.a3m"],
            }
        ),
        tracking,
    )
    ids = runspec_artifacts_module._load_allowlist_ids(tracking, "ds1")
    assert ids == {"archive_a": {"pdb_5snm_assembly_1"}}


def test_load_allowlist_ids_extracts_pdb_assembly_from_prefixed_msa_path(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking.parquet"
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1"],
                "swiftstack_archive": ["archive_a"],
                "msa_path": ["CORPUS_pdb_5snm_assembly_1"],
            }
        ),
        tracking,
    )
    ids = runspec_artifacts_module._load_allowlist_ids(tracking, "ds1")
    assert ids == {"archive_a": {"pdb_5snm_assembly_1"}}


def test_load_allowlist_ids_passes_through_pdb_assembly_from_model_entity_id(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking.parquet"
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1"],
                "swiftstack_archive": ["archive_a"],
                "model_entity_id": ["pdb_5snm_assembly_1"],
            }
        ),
        tracking,
    )
    ids = runspec_artifacts_module._load_allowlist_ids(tracking, "ds1")
    assert ids == {"archive_a": {"pdb_5snm_assembly_1"}}


@pytest.mark.parametrize(
    "msa_path",
    [
        "PDB_5snm_assembly_1",
        "pdb_5SNM_assembly_1",
        "pdb_5snm_assembly_",
        "pdb_5snm_assembly",
        "pdb_assembly_1",
        "pdb_5snm_assembly_1_extra",
    ],
)
def test_malformed_pdb_assembly_msa_path_produces_no_id(tmp_path: Path, msa_path: str) -> None:
    tracking = tmp_path / "tracking.parquet"
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1"],
                "swiftstack_archive": ["archive_a"],
                "msa_path": [msa_path],
            }
        ),
        tracking,
    )
    ids = runspec_artifacts_module._load_allowlist_ids(tracking, "ds1")
    assert ids == {}


def test_legacy_af_msa_path_allowlist_extraction_unchanged(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking.parquet"
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1"],
                "swiftstack_archive": ["archive_a"],
                "msa_path": ["AFDB_AF-0000000000000001"],
            }
        ),
        tracking,
    )
    ids = runspec_artifacts_module._load_allowlist_ids(tracking, "ds1")
    assert ids == {"archive_a": {"AF-0000000000000001"}}


def test_mixed_af_and_pdb_assembly_allowlist_extraction(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking.parquet"
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1", "ds1"],
                "swiftstack_archive": ["archive_a", "archive_a"],
                "msa_path": ["AFDB_AF-0000000000000001", "/data/corpus/pdb_5snm_assembly_1.a3m"],
            }
        ),
        tracking,
    )
    ids = runspec_artifacts_module._load_allowlist_ids(tracking, "ds1")
    assert ids == {"archive_a": {"AF-0000000000000001", "pdb_5snm_assembly_1"}}
