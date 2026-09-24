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

"""End-to-end integration tests for the bspp-orchestration processing flow.

Fast tier: exercises the full orchestration flow using extracted fixtures
and mock pipeline output (no production deps required).

Full tier (e2e_full): requires afdb-toolkit[production] and torch.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bspp.orchestration.runtime.discovery import build_file_index, discover
from bspp.orchestration.runtime.extraction.archives import extract_archive
from bspp.orchestration.runtime.postprocessing.aggregate import (
    aggregate_dataset,
    discover_shard_parquets,
    load_failed_model_ids,
)
from bspp.orchestration.runtime.postprocessing.cleanup import cleanup_shard_outputs
from bspp.orchestration.runtime.postprocessing.manifest import (
    filter_manifest_for_models,
    write_file_index,
    write_model_ids,
)
from bspp.orchestration.runtime.postprocessing.runner import prefilter_batch
from bspp.orchestration.runtime.postprocessing.shard_config import (
    compute_shard_config,
    read_shard_config,
    write_shard_config,
)
from bspp.orchestration.runtime.postprocessing.sharding import (
    compute_shard_slice,
    create_symlink_shard,
    filter_manifest_for_shard,
    persist_shard_manifest,
)
from bspp.orchestration.runtime.postprocessing.tracking import (
    create_tracking_parquet,
    query_status,
    update_status,
)

SAMPLE_ARCHIVE = Path(__file__).parent / "fixtures" / "sample_archive.tar.lz4"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_shard_parquet(path: Path, model_ids: list[str], shard_id: int) -> None:
    """Create a minimal shard_manifest.parquet with required columns."""
    table = pa.table(
        {
            "model_entity_id": model_ids,
            "shard_id": [shard_id] * len(model_ids),
            "pdb_path": [f"shard_{shard_id}/{mid}-model_v1.pdb" for mid in model_ids],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Fast tier: extract -> preprocess
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_ARCHIVE.exists(), reason="sample_archive.tar.lz4 fixture missing")
def test_e2e_extract_to_preprocess(
    tmp_path: Path,
    sample_archive: Path,
    manifest_csv: Path,
) -> None:
    """Extract archive, discover models, build file index, compute shard config,
    filter manifest -- the complete pre-processing phase."""
    # 1. Extract archive
    archive_copy = tmp_path / "sample_archive.tar.lz4"
    shutil.copy2(sample_archive, archive_copy)
    extracted_dir = tmp_path / "input"
    file_count = extract_archive(archive_copy, extracted_dir, keep_archive=True)
    assert file_count == 10  # 5 models x 2 files

    # 2. Discover model IDs
    model_ids = discover(extracted_dir)
    assert len(model_ids) == 5
    assert model_ids[0].startswith("AF-")

    # 3. Build file index
    file_index = build_file_index(extracted_dir)
    assert len(file_index) == 5
    for mid in model_ids:
        assert mid in file_index
        assert len(file_index[mid]) == 2  # pdb + meta json

    # 4. Write model IDs
    model_ids_path = tmp_path / "model_ids.txt"
    write_model_ids(model_ids, model_ids_path)
    assert model_ids_path.exists()
    written_ids = model_ids_path.read_text().strip().splitlines()
    assert len(written_ids) == 5

    # 5. Compute shard config
    config = compute_shard_config(len(model_ids))
    assert config["total_models"] == 5
    assert config["required_shards"] == 1
    assert config["array_range"] == "0-0"

    # 6. Write and read back shard config
    shard_config_path = tmp_path / "shard_config.json"
    write_shard_config(config, shard_config_path)
    loaded_config = read_shard_config(shard_config_path)
    assert loaded_config == config

    # 7. Filter manifest for discovered models
    filtered_manifest = tmp_path / "filtered_manifest.csv"
    rows_written = filter_manifest_for_models(
        manifest_csv,
        set(model_ids),
        filtered_manifest,
    )
    assert rows_written == 5

    # 8. Write file index
    file_index_path = tmp_path / "file_index.json"
    write_file_index(file_index, file_index_path)
    assert file_index_path.exists()


# ---------------------------------------------------------------------------
# Fast tier: shard + prefilter
# ---------------------------------------------------------------------------


def test_e2e_shard_and_prefilter(
    tmp_path: Path,
    input_dir: Path,
    manifest_csv: Path,
) -> None:
    """Discover models, build file index, compute shard slice, create symlink
    shard, filter manifest for shard, and prefilter the batch."""
    # 1. Discover model IDs
    model_ids = discover(input_dir)
    assert len(model_ids) == 5

    # 2. Build file index
    file_index = build_file_index(input_dir)
    assert len(file_index) == 5

    # 3. Compute shard slice (1 shard with all 5 models)
    start, end = compute_shard_slice(shard_id=0, total_items=len(model_ids), num_shards=1)
    assert start == 0
    assert end == 5
    shard_model_ids = model_ids[start:end]
    assert len(shard_model_ids) == 5

    # 4. Create symlink shard
    shard_dir = tmp_path / "shard_0"
    linked = create_symlink_shard(shard_model_ids, file_index, input_dir, shard_dir)
    assert linked == 10  # 5 models x 2 files
    assert shard_dir.exists()
    # Verify symlinks are valid
    for mid in shard_model_ids:
        pdb_link = shard_dir / f"{mid}-model_v1.pdb"
        assert pdb_link.is_symlink()
        assert pdb_link.resolve().exists()

    # 5. Filter manifest for shard
    shard_manifest_csv = tmp_path / "shard_manifest.csv"
    shard_rows = filter_manifest_for_shard(
        manifest_csv,
        set(shard_model_ids),
        shard_manifest_csv,
    )
    assert shard_rows == 5

    # 6. Prefilter batch (check meta JSONs) -- all should pass
    good_ids, bad_ids = prefilter_batch(shard_model_ids, shard_dir)
    assert len(good_ids) == 5
    assert len(bad_ids) == 0


# ---------------------------------------------------------------------------
# Fast tier: aggregate + track
# ---------------------------------------------------------------------------


def test_e2e_aggregate_and_track(
    tmp_path: Path,
    master_parquet: Path,
) -> None:
    """Create shard output structure, aggregate, and exercise the tracking lifecycle."""
    dataset = "test_dataset"
    output_base = tmp_path / "output"
    output_dir = output_base / dataset

    # 1. Create fake shard parquets
    shard_0_ids = ["AF-0000000000000001", "AF-0000000000000002", "AF-0000000000000003"]
    shard_1_ids = ["AF-0000000000000004", "AF-0000000000000005"]
    _write_shard_parquet(output_dir / "shard_0" / "shard_manifest.parquet", shard_0_ids, 0)
    _write_shard_parquet(output_dir / "shard_1" / "shard_manifest.parquet", shard_1_ids, 1)

    # 2. Create failed_models.tsv with 1 failed model
    (output_dir / "failed_models.tsv").write_text("AF-0000000000000003\textract\tnull meta json\n")

    # 3. Verify discover + load helpers
    parquets = discover_shard_parquets(output_dir)
    assert len(parquets) == 2

    failed_ids = load_failed_model_ids(output_dir)
    assert failed_ids == {"AF-0000000000000003"}

    # 4. Aggregate
    aggregated_path = aggregate_dataset(dataset, output_base)
    assert aggregated_path.exists()
    table = pq.read_table(aggregated_path)
    assert table.num_rows == 5
    assert "status" in table.schema.names

    statuses = dict(
        zip(
            table.column("model_entity_id").to_pylist(),
            table.column("status").to_pylist(),
            strict=True,
        )
    )
    assert statuses["AF-0000000000000003"] == "failed"
    assert statuses["AF-0000000000000001"] == "success"
    assert statuses["AF-0000000000000005"] == "success"

    # 5. Create tracking parquet from master
    tracking_path = tmp_path / "tracking.parquet"
    n_rows = create_tracking_parquet(
        master_parquet, tracking_path, s3_output_prefix="s3://example-bucket/postprocessed/test_dataset/"
    )
    assert n_rows == 5

    # 6. Update status
    updated = update_status(
        tracking_path,
        match_column="source_run",
        match_substring="test_dataset",
        new_status="processed",
    )
    assert updated == 5

    # 7. Query status
    counts = query_status(tracking_path, dataset="test_dataset")
    assert counts == {"processed": 5}


# ---------------------------------------------------------------------------
# Fast tier: cleanup safety
# ---------------------------------------------------------------------------


def test_e2e_cleanup_safety(tmp_path: Path) -> None:
    """Verify cleanup refuses without force/tracking, and force mode removes
    success_outputs while preserving shard metadata."""
    output_dir = tmp_path / "dataset"
    for i in range(2):
        shard = output_dir / f"shard_{i}"
        success = shard / "success_outputs"
        success.mkdir(parents=True)
        (success / f"model_{i}.pdb").write_text("pdb data")
        (success / f"model_{i}.cif").write_text("cif data")
        # Metadata that should survive cleanup
        (shard / "failed_models.tsv").write_text("")
        (shard / "shard_manifest.parquet").write_text("placeholder")

    # Without force and no tracking -> should raise ValueError
    with pytest.raises(ValueError, match="tracking_path is required"):
        cleanup_shard_outputs(output_dir, force=False)

    # success_outputs should still be intact
    assert (output_dir / "shard_0" / "success_outputs").exists()
    assert (output_dir / "shard_1" / "success_outputs").exists()

    # With force -> should remove success_outputs
    cleaned = cleanup_shard_outputs(output_dir, force=True)
    assert cleaned == 2

    # Verify success_outputs are gone
    assert not (output_dir / "shard_0" / "success_outputs").exists()
    assert not (output_dir / "shard_1" / "success_outputs").exists()

    # Verify shard metadata is preserved
    assert (output_dir / "shard_0" / "failed_models.tsv").exists()
    assert (output_dir / "shard_0" / "shard_manifest.parquet").exists()
    assert (output_dir / "shard_1" / "failed_models.tsv").exists()
    assert (output_dir / "shard_1" / "shard_manifest.parquet").exists()


# ---------------------------------------------------------------------------
# Fast tier: full orchestration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_ARCHIVE.exists(), reason="sample_archive.tar.lz4 fixture missing")
def test_e2e_full_orchestration(
    tmp_path: Path,
    sample_archive: Path,
    manifest_csv: Path,
    master_parquet: Path,
    expected_outputs_dir: Path,
) -> None:
    """Complete fast-tier pipeline: extract -> discover -> preprocess -> shard
    -> prefilter -> mock process -> persist -> aggregate -> track."""
    dataset = "test_dataset"

    # --- EXTRACT ---
    archive_copy = tmp_path / "sample_archive.tar.lz4"
    shutil.copy2(sample_archive, archive_copy)
    extracted_dir = tmp_path / "extracted"
    file_count = extract_archive(archive_copy, extracted_dir, keep_archive=True)
    assert file_count == 10

    # --- DISCOVER ---
    model_ids = discover(extracted_dir)
    assert len(model_ids) == 5
    file_index = build_file_index(extracted_dir)
    assert len(file_index) == 5

    # --- PREPROCESS ---
    preprocess_dir = tmp_path / "preprocess"
    preprocess_dir.mkdir()

    model_ids_path = preprocess_dir / "model_ids.txt"
    write_model_ids(model_ids, model_ids_path)
    assert model_ids_path.exists()

    config = compute_shard_config(len(model_ids))
    shard_config_path = preprocess_dir / "shard_config.json"
    write_shard_config(config, shard_config_path)
    loaded_config = read_shard_config(shard_config_path)
    assert loaded_config["required_shards"] == 1

    file_index_path = preprocess_dir / "file_index.json"
    write_file_index(file_index, file_index_path)

    filtered_manifest_path = preprocess_dir / "filtered_manifest.csv"
    rows = filter_manifest_for_models(manifest_csv, set(model_ids), filtered_manifest_path)
    assert rows == 5

    # --- SHARD ---
    output_base = tmp_path / "output"
    shard_output = output_base / dataset
    num_shards = loaded_config["required_shards"]

    start, end = compute_shard_slice(0, len(model_ids), num_shards)
    shard_model_ids = model_ids[start:end]
    assert len(shard_model_ids) == 5

    shard_dir = shard_output / "shard_0"
    linked = create_symlink_shard(shard_model_ids, file_index, extracted_dir, shard_dir)
    assert linked == 10

    shard_manifest_csv = shard_dir / "shard_manifest.csv"
    shard_rows = filter_manifest_for_shard(
        filtered_manifest_path,
        set(shard_model_ids),
        shard_manifest_csv,
    )
    assert shard_rows == 5

    # --- PREFILTER ---
    good_ids, bad_ids = prefilter_batch(shard_model_ids, shard_dir)
    assert len(good_ids) == 5
    assert len(bad_ids) == 0

    # --- MOCK PROCESS ---
    # Simulate pipeline output by copying expected_outputs into success_outputs
    success_dir = shard_dir / "success_outputs"
    success_dir.mkdir()
    for output_file in expected_outputs_dir.iterdir():
        if output_file.is_file():
            shutil.copy2(output_file, success_dir / output_file.name)

    # Verify mock outputs exist
    output_files = list(success_dir.iterdir())
    assert len(output_files) > 0
    # Expected: confidence, PAE, PDB for each of 5 models = 15 files
    assert len(output_files) == 15

    # --- PERSIST ---
    shard_manifest_parquet = shard_dir / "shard_manifest.parquet"
    persist_shard_manifest(
        shard_manifest_csv,
        shard_id=0,
        dataset_tag=dataset,
        output_path=shard_manifest_parquet,
    )
    assert shard_manifest_parquet.exists()
    persisted = pq.read_table(shard_manifest_parquet)
    assert persisted.num_rows == 5
    assert "shard_id" in persisted.schema.names
    assert "dataset_tag" in persisted.schema.names

    # --- AGGREGATE ---
    aggregated_path = aggregate_dataset(dataset, output_base)
    assert aggregated_path.exists()
    agg_table = pq.read_table(aggregated_path)
    assert agg_table.num_rows == 5
    # No failures in this run
    statuses = set(agg_table.column("status").to_pylist())
    assert statuses == {"success"}

    # --- TRACK ---
    tracking_path = tmp_path / "tracking.parquet"
    n = create_tracking_parquet(
        master_parquet, tracking_path, s3_output_prefix="s3://example-bucket/postprocessed/test_dataset/"
    )
    assert n == 5

    # Initial state: all pending
    counts = query_status(tracking_path, dataset=dataset)
    assert counts == {"pending": 5}

    # Update to processed
    updated = update_status(
        tracking_path,
        match_column="source_run",
        match_substring=dataset,
        new_status="processed",
    )
    assert updated == 5

    counts = query_status(tracking_path, dataset=dataset)
    assert counts == {"processed": 5}

    # Update to done
    updated = update_status(
        tracking_path,
        match_column="source_run",
        match_substring=dataset,
        new_status="done",
    )
    assert updated == 5

    counts = query_status(tracking_path, dataset=dataset)
    assert counts == {"done": 5}


# ---------------------------------------------------------------------------
# Full tier: requires production deps
# ---------------------------------------------------------------------------


@pytest.mark.e2e_full
@pytest.mark.skipif(not SAMPLE_ARCHIVE.exists(), reason="sample_archive.tar.lz4 fixture missing")
def test_e2e_full_pipeline(
    tmp_path: Path,
    sample_archive: Path,
    manifest_csv: Path,
    uniprot_db: Path,
) -> None:
    """Full E2E: extract, discover, shard, run production_pipeline, verify output.

    Requires torch and afdb-toolkit[production] to be installed.
    """
    pytest.importorskip("torch", reason="Full E2E requires torch")

    from bspp.orchestration.runtime.postprocessing.runner import run_pipeline

    # Extract
    archive_copy = tmp_path / "sample_archive.tar.lz4"
    shutil.copy2(sample_archive, archive_copy)
    extracted_dir = tmp_path / "extracted"
    extract_archive(archive_copy, extracted_dir, keep_archive=True)

    # Discover + shard
    model_ids = discover(extracted_dir)
    file_index = build_file_index(extracted_dir)

    shard_dir = tmp_path / "shard_0"
    create_symlink_shard(model_ids, file_index, extracted_dir, shard_dir)

    shard_manifest_csv = tmp_path / "shard_manifest.csv"
    filter_manifest_for_shard(manifest_csv, set(model_ids), shard_manifest_csv)

    # Run the actual pipeline
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    exit_code = run_pipeline(
        shard_dir,
        output_dir,
        shard_manifest_csv,
        uniprot_db=uniprot_db,
        workers=1,
    )

    if exit_code == 127:
        pytest.skip("production_pipeline.py not found -- afdb-toolkit not fully installed")

    assert exit_code == 0, f"Pipeline exited with code {exit_code}"

    # Verify output files exist for each model
    success_dir = output_dir / "success_outputs"
    if success_dir.exists():
        output_files = list(success_dir.iterdir())
        assert len(output_files) > 0

        # Check for expected output types
        file_names = {f.name for f in output_files}
        # At minimum we expect some JSON and PDB outputs
        has_json = any(f.name.endswith(".json") for f in output_files)
        has_pdb = any(f.name.endswith(".pdb") for f in output_files)
        assert has_json or has_pdb, f"Expected JSON or PDB outputs, got: {file_names}"
