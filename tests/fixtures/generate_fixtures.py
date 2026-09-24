#!/usr/bin/env python3
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

"""
Generate synthetic test fixtures for bspp-orchestration.

Self-contained, deterministic (seed=42) fixture generator.
Produces all files needed by pipeline integration tests:
  - ColabFold input files (PDB + meta JSON)
  - AFDB output files (confidence JSON, PAE JSON, output PDB)
  - Tracking and master parquet files
  - DuckDB with UniProt entries
  - Config files (manifest, dataset_config, provider, shard, recipe)
  - tar.lz4 archive of input files
  - Validation fixtures (good_dataset / bad_dataset)

Usage:
    python tests/fixtures/generate_fixtures.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
NUM_MODELS = 5
# Keep residue count small so PAE matrices stay well under 1.5 MB.
# 50 residues per chain x 2 chains (homodimer) = 100 total residues.
# PAE matrix: 100x100 floats ~ 80 KB in JSON.
RESIDUES_PER_CHAIN = 50
TOTAL_RESIDUES = RESIDUES_PER_CHAIN * 2

FIXTURES_DIR = Path(__file__).resolve().parent

# Deterministic epoch timestamp for tar members so the generated archive has
# no personal mtime metadata (e11s07).
FIXED_MTIME = 0

MODEL_IDS = [f"AF-{str(i).zfill(16)}" for i in range(1, NUM_MODELS + 1)]

UNIPROT_ENTRIES: list[dict[str, Any]] = [
    {
        "primary_ac": "P12345",
        "entry_name": "TEST1_HUMAN",
        "organism": "Homo sapiens",
        "taxid": 9606,
        "sequence": "M" + "A" * (TOTAL_RESIDUES - 1),
        "protein_full_names": ["Test protein 1"],
        "protein_short_names": ["TP1"],
        "gene_names": ["TEST1"],
        "gene_synonyms": ["TP1", "TESTP1"],
        "organism_common_names": ["Human"],
        "organism_synonyms": [],
        "reviewed": True,
        "is_uniprot_reference_proteome": True,
        "sequence_version_date": "2024-01-01",
    },
    {
        "primary_ac": "P67890",
        "entry_name": "TEST2_MOUSE",
        "organism": "Mus musculus",
        "taxid": 10090,
        "sequence": "M" + "G" * (TOTAL_RESIDUES - 1),
        "protein_full_names": ["Test protein 2"],
        "protein_short_names": ["TP2"],
        "gene_names": ["Test2"],
        "gene_synonyms": [],
        "organism_common_names": ["Mouse"],
        "organism_synonyms": ["House mouse"],
        "reviewed": True,
        "is_uniprot_reference_proteome": True,
        "sequence_version_date": "2024-02-01",
    },
    {
        "primary_ac": "Q11111",
        "entry_name": "TEST3_YEAST",
        "organism": "Saccharomyces cerevisiae",
        "taxid": 559292,
        "sequence": "M" + "S" * (TOTAL_RESIDUES - 1),
        "protein_full_names": ["Test protein 3"],
        "protein_short_names": [],
        "gene_names": ["TST3"],
        "gene_synonyms": ["YAL001C"],
        "organism_common_names": ["Baker's yeast"],
        "organism_synonyms": [],
        "reviewed": False,
        "is_uniprot_reference_proteome": False,
        "sequence_version_date": "2024-03-01",
    },
]


# ---------------------------------------------------------------------------
# Data generators (adapted from legacy setup_mock_data.py)
# ---------------------------------------------------------------------------


def generate_plddt_scores(seq_length: int) -> list[float]:
    """Generate realistic pLDDT scores in [40, 95]."""
    return [round(random.uniform(40.0, 95.0), 2) for _ in range(seq_length)]


def generate_pae_matrix(seq_length: int) -> list[list[float]]:
    """Generate a realistic PAE matrix.

    Diagonal ~ 0, off-diagonal grows with distance (capped at ~31.75).
    """
    matrix: list[list[float]] = []
    for i in range(seq_length):
        row: list[float] = []
        for j in range(seq_length):
            dist = abs(i - j)
            base = min(dist * 0.5, 25.0)
            noise = random.uniform(-2.0, 2.0)
            value = max(0.25, min(31.75, base + noise))
            row.append(round(value, 2))
        matrix.append(row)
    return matrix


def generate_pdb_file(model_id: str, seq_length: int) -> str:
    """Generate a minimal valid PDB file with CA atoms along a helix."""
    lines = [
        f"HEADER    PREDICTION                              {datetime.now().strftime('%d-%b-%y').upper()}   {model_id}",
        f"TITLE     ALPHAFOLD PREDICTION FOR {model_id}",
    ]
    atom_num = 1
    for res_num in range(1, seq_length + 1):
        theta = res_num * 0.3
        x = math.cos(theta) * 5.0 + res_num * 0.1
        y = math.sin(theta) * 5.0
        z = res_num * 1.5
        bfactor = random.uniform(40.0, 95.0)
        line = f"ATOM  {atom_num:5d}  CA  ALA A{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}           C"
        lines.append(line)
        atom_num += 1
    lines.append("END")
    return "\n".join(lines)


def generate_meta_json(model_id: str, seq_length: int) -> dict[str, Any]:
    """Generate ColabFold-style meta JSON (input format).

    Keys: plddt, pae, max_pae, ptm, iptm.
    """
    plddt = generate_plddt_scores(seq_length)
    pae = generate_pae_matrix(seq_length)
    max_pae = max(max(row) for row in pae)
    return {
        "plddt": plddt,
        "pae": pae,
        "max_pae": round(max_pae, 2),
        "ptm": round(random.uniform(0.5, 0.95), 3),
        "iptm": round(random.uniform(0.4, 0.9), 3),
    }


def generate_confidence_json(seq_length: int) -> dict[str, Any]:
    """Generate AFDB confidence JSON (output format).

    Categories: V (>90), H (70-90), M (50-70), L (30-50), D (<30).
    """
    plddt = generate_plddt_scores(seq_length)

    def _cat(s: float) -> str:
        if s >= 90:
            return "V"
        if s >= 70:
            return "H"
        if s >= 50:
            return "M"
        if s >= 30:
            return "L"
        return "D"

    return {
        "residueNumber": list(range(1, seq_length + 1)),
        "confidenceScore": plddt,
        "confidenceCategory": [_cat(s) for s in plddt],
    }


def generate_pae_json(seq_length: int) -> list[dict[str, Any]]:
    """Generate AFDB PAE JSON (output format)."""
    pae = generate_pae_matrix(seq_length)
    return [
        {
            "predicted_aligned_error": pae,
            "max_predicted_aligned_error": round(max(max(row) for row in pae), 2),
        }
    ]


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------


def create_duckdb(db_path: Path) -> None:
    """Create a mock DuckDB with UniProt entries."""
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE entry (
            primary_ac VARCHAR PRIMARY KEY,
            entry_name VARCHAR,
            organism VARCHAR,
            taxid INTEGER,
            sequence VARCHAR,
            protein_full_names VARCHAR[],
            protein_short_names VARCHAR[],
            gene_names VARCHAR[],
            gene_synonyms VARCHAR[],
            gene_ordered_locus_names VARCHAR[],
            gene_orf_names VARCHAR[],
            organism_common_names VARCHAR[],
            organism_synonyms VARCHAR[],
            reviewed BOOLEAN,
            is_uniprot_reference_proteome BOOLEAN,
            sequence_version_date DATE
        )
    """)
    for entry in UNIPROT_ENTRIES:
        con.execute(
            "INSERT INTO entry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                entry["primary_ac"],
                entry["entry_name"],
                entry["organism"],
                entry["taxid"],
                entry["sequence"],
                entry.get("protein_full_names", []),
                entry.get("protein_short_names", []),
                entry.get("gene_names", []),
                entry.get("gene_synonyms", []),
                [],  # gene_ordered_locus_names
                [],  # gene_orf_names
                entry.get("organism_common_names", []),
                entry.get("organism_synonyms", []),
                entry.get("reviewed", False),
                entry.get("is_uniprot_reference_proteome", False),
                entry.get("sequence_version_date"),
            ],
        )
    con.close()
    print(f"  DuckDB: {db_path} ({len(UNIPROT_ENTRIES)} entries)")


# ---------------------------------------------------------------------------
# Input / expected-output generators
# ---------------------------------------------------------------------------


def generate_input_files(input_dir: Path) -> None:
    """Generate flat ColabFold input files (PDB + meta JSON) per model."""
    input_dir.mkdir(parents=True, exist_ok=True)
    for i, model_id in enumerate(MODEL_IDS):
        entry = UNIPROT_ENTRIES[i % len(UNIPROT_ENTRIES)]
        seq_length = len(entry["sequence"])

        # PDB
        pdb_path = input_dir / f"{model_id}-model_v1.pdb"
        pdb_path.write_text(generate_pdb_file(model_id, seq_length))

        # Meta JSON (ColabFold format)
        meta_path = input_dir / f"{model_id}-meta_v1.json"
        meta_path.write_text(json.dumps(generate_meta_json(model_id, seq_length), indent=2))

    print(f"  Input files: {input_dir} ({len(MODEL_IDS)} models x 2 files)")


def generate_expected_outputs(output_dir: Path) -> None:
    """Generate expected pipeline output files per model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, model_id in enumerate(MODEL_IDS):
        entry = UNIPROT_ENTRIES[i % len(UNIPROT_ENTRIES)]
        seq_length = len(entry["sequence"])

        # Confidence JSON
        conf_path = output_dir / f"{model_id}-confidence_v1.json"
        conf_path.write_text(json.dumps(generate_confidence_json(seq_length), indent=2))

        # PAE JSON
        pae_path = output_dir / f"{model_id}-predicted_aligned_error_v1.json"
        pae_path.write_text(json.dumps(generate_pae_json(seq_length), indent=2))

        # Output PDB (same as input for mock)
        pdb_path = output_dir / f"{model_id}-model_v1.pdb"
        pdb_path.write_text(generate_pdb_file(model_id, seq_length))

    print(f"  Expected outputs: {output_dir} ({len(MODEL_IDS)} models x 3 files)")


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------


def generate_config_files(config_dir: Path) -> None:
    """Generate pipeline config files."""
    config_dir.mkdir(parents=True, exist_ok=True)

    # --- manifest.csv (ColabFold manifest) ---
    manifest_path = config_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_entity_id", "uniprot_ac", "chain_id"])
        writer.writeheader()
        for i, model_id in enumerate(MODEL_IDS):
            entry = UNIPROT_ENTRIES[i % len(UNIPROT_ENTRIES)]
            writer.writerow(
                {
                    "model_entity_id": model_id,
                    "uniprot_ac": entry["primary_ac"],
                    "chain_id": "A",
                }
            )

    # --- dataset_config.json ---
    dataset_config = {
        "providerId": "test-provider",
        "toolUsed": "AlphaFold",
        "latestVersion": 1,
        "allVersions": [1],
        "entityType": "protein",
        "modelCreatedDate": "2024-01-01T00:00:00Z",
        "uniqueIdTemplate": "{model_entity_id}",
        "versionTag": "v1",
    }
    (config_dir / "dataset_config.json").write_text(json.dumps(dataset_config, indent=2))

    # --- provider.json ---
    provider = {
        "providerId": "test-provider",
        "providerName": "Test Provider",
        "providerUrl": "https://example.com",
        "copyrights": ["Copyright 2024 Test Provider. All rights reserved."],
    }
    (config_dir / "provider.json").write_text(json.dumps(provider, indent=2))

    # --- af_mapping.tsv (single column, no header) ---
    af_mapping_path = config_dir / "af_mapping.tsv"
    af_mapping_path.write_text("\n".join(MODEL_IDS) + "\n")

    # --- shard_config.json ---
    shard_config = {
        "total_models": NUM_MODELS,
        "max_per_shard": 5000,
        "required_shards": 1,
        "array_range": "0-0",
    }
    (config_dir / "shard_config.json").write_text(json.dumps(shard_config, indent=2))

    # --- recipe/config.yaml (minimal local recipe, no SLURM) ---
    recipe_dir = config_dir / "recipe"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    recipe = {
        "dataset": "test_dataset",
        "version": "v1",
        "provider_id": "test-provider",
        "input_source": "local",
        "input_path": "input/",
        "output_path": "output/",
        "num_models": NUM_MODELS,
        "batch_size": NUM_MODELS,
        "upload": {"s3": False, "gcs": False},
    }
    (recipe_dir / "config.yaml").write_text(yaml.dump(recipe, default_flow_style=False, sort_keys=False))

    print(f"  Config files: {config_dir} (manifest, dataset_config, provider, af_mapping, shard_config, recipe)")


# ---------------------------------------------------------------------------
# Parquet files
# ---------------------------------------------------------------------------


def generate_master_parquet(parquet_path: Path) -> None:
    """Generate master.parquet with pipeline tracking columns."""
    pdb_paths = [f"input/{mid}-model_v1.pdb" for mid in MODEL_IDS]
    table = pa.table(
        {
            "pdb_path": pa.array(pdb_paths, type=pa.string()),
            "model_entity_id": pa.array(MODEL_IDS, type=pa.string()),
            "source_run": pa.array(["test_dataset"] * NUM_MODELS, type=pa.string()),
            "swiftstack_archive": pa.array(["test_archive.tar.lz4"] * NUM_MODELS, type=pa.string()),
            "predictions_path": pa.array(
                [f"expected_outputs/{mid}-confidence_v1.json" for mid in MODEL_IDS],
                type=pa.string(),
            ),
        }
    )
    pq.write_table(table, str(parquet_path))
    print(f"  Master parquet: {parquet_path} ({NUM_MODELS} rows)")


def generate_tracking_parquet(parquet_path: Path) -> None:
    """Generate tracking.parquet extending master with status columns."""
    pdb_paths = [f"input/{mid}-model_v1.pdb" for mid in MODEL_IDS]
    table = pa.table(
        {
            "pdb_path": pa.array(pdb_paths, type=pa.string()),
            "model_entity_id": pa.array(MODEL_IDS, type=pa.string()),
            "source_run": pa.array(["test_dataset"] * NUM_MODELS, type=pa.string()),
            "swiftstack_archive": pa.array(["test_archive.tar.lz4"] * NUM_MODELS, type=pa.string()),
            "predictions_path": pa.array(
                [f"expected_outputs/{mid}-confidence_v1.json" for mid in MODEL_IDS],
                type=pa.string(),
            ),
            "dataset_name": pa.array(["test_dataset"] * NUM_MODELS, type=pa.string()),
            "postprocess_status": pa.array(["pending"] * NUM_MODELS, type=pa.string()),
            "postprocess_started_at": pa.array([None] * NUM_MODELS, type=pa.timestamp("us")),
            "postprocess_completed_at": pa.array([None] * NUM_MODELS, type=pa.timestamp("us")),
            "s3_destination": pa.array([None] * NUM_MODELS, type=pa.string()),
            "gcs_destination": pa.array([None] * NUM_MODELS, type=pa.string()),
            "needs_archive_resolution": pa.array([False] * NUM_MODELS, type=pa.bool_()),
        }
    )
    pq.write_table(table, str(parquet_path))
    print(f"  Tracking parquet: {parquet_path} ({NUM_MODELS} rows)")


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


def _neutralize_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Normalize tar member metadata to neutral, deterministic values."""
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = FIXED_MTIME
    return info


def generate_tar_lz4(archive_path: Path, input_dir: Path) -> None:
    """Create a .tar.lz4 archive from input files.

    Uses the system lz4 CLI tool via subprocess.
    """
    # First create a .tar in a temp location
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tar_path = Path(tmp.name)

    try:
        with tarfile.open(str(tar_path), "w") as tar:
            for f in sorted(input_dir.iterdir()):
                if f.is_file():
                    tar.add(str(f), arcname=f.name, filter=_neutralize_tarinfo)

        # Compress with lz4
        subprocess.run(
            ["lz4", "-f", str(tar_path), str(archive_path)],
            check=True,
            capture_output=True,
        )
        print(f"  Archive: {archive_path} ({archive_path.stat().st_size} bytes)")
    finally:
        tar_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Validation fixtures (copied from legacy)
# ---------------------------------------------------------------------------


def copy_validation_fixtures(fixtures_dir: Path) -> None:
    """Copy validation fixtures from the legacy repo (read-only)."""
    legacy_base = Path("/srv/example/legacy/AFDB-Integration-Kit/tests/fixtures/validation")
    dest_base = fixtures_dir / "validation"

    for subset in ("good_dataset", "bad_dataset"):
        src = legacy_base / subset
        dst = dest_base / subset
        if not src.exists():
            print(f"  WARNING: Legacy validation dir not found: {src}")
            continue
        # Remove existing and copy fresh
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(str(src), str(dst))
        file_count = sum(1 for _ in dst.iterdir())
        print(f"  Validation/{subset}: {dst} ({file_count} files)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed(SEED)

    print(f"Generating fixtures in {FIXTURES_DIR}")
    print(f"  Models: {NUM_MODELS}, Residues: {TOTAL_RESIDUES}")
    print()

    # 1. Input files (flat layout)
    generate_input_files(FIXTURES_DIR / "input")

    # 2. Expected output files
    # Reset seed so outputs are independently reproducible
    random.seed(SEED + 1)
    generate_expected_outputs(FIXTURES_DIR / "expected_outputs")

    # 3. Config files
    generate_config_files(FIXTURES_DIR / "config")

    # 4. DuckDB
    create_duckdb(FIXTURES_DIR / "uniprot_test.duckdb")

    # 5. Master parquet
    generate_master_parquet(FIXTURES_DIR / "master.parquet")

    # 6. Tracking parquet
    generate_tracking_parquet(FIXTURES_DIR / "tracking.parquet")

    # 7. tar.lz4 archive
    generate_tar_lz4(FIXTURES_DIR / "sample_archive.tar.lz4", FIXTURES_DIR / "input")

    # 8. Validation fixtures from legacy
    copy_validation_fixtures(FIXTURES_DIR)

    print()
    print("Done. All fixtures generated.")


if __name__ == "__main__":
    main()
