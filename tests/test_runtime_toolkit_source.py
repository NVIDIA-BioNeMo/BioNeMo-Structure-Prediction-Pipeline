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

"""Unit tests for the central toolkit resolver — override-first, baked-fallback, fail-closed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.runtime.toolkit import (
    EXPECTED_TOOLKIT_COMMIT,
    PROVENANCE_FILENAME,
    BakedToolkitLayoutError,
    BakedToolkitNotFoundError,
    BakedToolkitProvenanceError,
    ToolkitOverrideLayoutError,
    ToolkitOverrideNotFoundError,
    resolve_toolkit,
    resolve_toolkit_for_container,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _runspec(
    tmp_path: Path,
    *,
    afdb_toolkit_repo: str | None = None,
    mounts: list[dict[str, str]] | None = None,
) -> RunSpec:
    """Build a minimal RunSpec for toolkit resolver tests."""
    data: dict[str, object] = {
        "dataset": {"name": "test_ds", "run_id": "test-run", "mode": "archive", "array": "0-0"},
        "cluster": {"name": "example-cluster", "account": "test-account"},
        "paths": {
            "project_root": str(tmp_path),
            "staging_dir": str(tmp_path / "staging"),
            "output_dir": str(tmp_path / "output"),
            "log_dir": str(tmp_path / "logs"),
            "legacy_repo": str(tmp_path / "AFDB-Integration-Kit"),
            "orchestration_repo": str(tmp_path / "bspp-orchestration"),
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
            "mounts": mounts or [],
        },
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
            "self_upload": False,
            "local_scratch": True,
            "scratch_dir": "/dev/shm",
            "s5cmd_path": "s5cmd",
            "upload_slots": 4,
        },
        "storage": {
            "s3_archive_prefix": "s3://test/structures/",
            "s3_output_prefix": "s3://test/output/",
            "gcs_destination_prefix": None,
            "allow_production_prefixes": False,
        },
        "secrets": {
            "s3_credentials_ref": "env:BSPP_SWIFTSTACK",
            "gcs_credentials_ref": None,
        },
    }
    if afdb_toolkit_repo is not None:
        data["paths"]["afdb_toolkit_repo"] = afdb_toolkit_repo
        data["paths"]["legacy_repo"] = afdb_toolkit_repo
    return RunSpec.from_mapping(data)


def _write_provenance(root: Path, commit: str = EXPECTED_TOOLKIT_COMMIT) -> Path:
    """Write a valid provenance.json at *root* matching the Dockerfile contract."""
    prov = root / PROVENANCE_FILENAME
    prov.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository": "https://example.com/toolkit",
                "branch": "main",
                "commit": commit,
                "checked_out_at": "2025-01-01T00:00:00Z",
                "toolkit_path": str(root),
            }
        ),
        encoding="utf-8",
    )
    return prov


def _baked_tree(
    root: Path,
    *,
    with_provenance: bool = True,
    commit: str = EXPECTED_TOOLKIT_COMMIT,
    executable_ipsae: bool = True,
) -> None:
    """Create a minimal baked toolkit tree at *root* matching the Dockerfile layout."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "production_pipeline.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='afdb-toolkit'\n", encoding="utf-8")
    ipsae_bin = root / "ipsae_cpp"
    ipsae_bin.write_text("binary", encoding="utf-8")
    if executable_ipsae:
        ipsae_bin.chmod(0o755)
    if with_provenance:
        _write_provenance(root, commit=commit)


def _override_tree(
    root: Path,
    *,
    include_production_pipeline: bool = True,
    include_pyproject: bool = True,
) -> None:
    """Create a minimal override toolkit tree at *root*."""
    root.mkdir(parents=True, exist_ok=True)
    if include_production_pipeline:
        (root / "scripts").mkdir(parents=True, exist_ok=True)
        (root / "scripts" / "production_pipeline.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    if include_pyproject:
        (root / "pyproject.toml").write_text("[project]\nname='afdb-toolkit'\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# resolve_toolkit — baked mode (runtime, filesystem checks)
# ---------------------------------------------------------------------------


def test_baked_happy_path(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Baked mode with valid /opt/afdb-toolkit returns baked source."""
    baked = tmp_path / "opt" / "afdb-toolkit"
    _baked_tree(baked)
    monkeypatch.setattr("bspp.orchestration.runtime.toolkit.BAKED_TOOLKIT_ROOT", baked)

    spec = _runspec(tmp_path)
    result = resolve_toolkit(spec)
    assert result.source == "baked"
    assert result.root == baked
    assert result.provenance_commit == EXPECTED_TOOLKIT_COMMIT
    assert result.production_pipeline == baked / "scripts" / "production_pipeline.py"
    assert result.ipsae_path == baked / "ipsae_cpp"


def test_baked_missing_root_raises(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Baked mode with absent /opt/afdb-toolkit raises BakedToolkitNotFoundError."""
    baked = tmp_path / "opt" / "afdb-toolkit"
    monkeypatch.setattr("bspp.orchestration.runtime.toolkit.BAKED_TOOLKIT_ROOT", baked)

    spec = _runspec(tmp_path)
    with pytest.raises(BakedToolkitNotFoundError, match="missing from the container image"):
        resolve_toolkit(spec)


def test_baked_missing_provenance_raises(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Baked mode with root but no provenance.json raises."""
    baked = tmp_path / "opt" / "afdb-toolkit"
    _baked_tree(baked, with_provenance=False)
    monkeypatch.setattr("bspp.orchestration.runtime.toolkit.BAKED_TOOLKIT_ROOT", baked)

    spec = _runspec(tmp_path)
    with pytest.raises(BakedToolkitProvenanceError, match="provenance file"):
        resolve_toolkit(spec)


def test_baked_malformed_provenance_raises(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Baked mode with unparseable JSON provenance raises."""
    baked = tmp_path / "opt" / "afdb-toolkit"
    _baked_tree(baked)
    (baked / PROVENANCE_FILENAME).write_text("{broken: [\n", encoding="utf-8")
    monkeypatch.setattr("bspp.orchestration.runtime.toolkit.BAKED_TOOLKIT_ROOT", baked)

    spec = _runspec(tmp_path)
    with pytest.raises(BakedToolkitProvenanceError, match="unparseable"):
        resolve_toolkit(spec)


def test_baked_provenance_not_object_raises(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Baked mode with valid JSON that is not a mapping raises."""
    baked = tmp_path / "opt" / "afdb-toolkit"
    _baked_tree(baked)
    (baked / PROVENANCE_FILENAME).write_text("[1, 2, 3]\n", encoding="utf-8")
    monkeypatch.setattr("bspp.orchestration.runtime.toolkit.BAKED_TOOLKIT_ROOT", baked)

    spec = _runspec(tmp_path)
    with pytest.raises(BakedToolkitProvenanceError, match="not a JSON object"):
        resolve_toolkit(spec)


def test_baked_wrong_commit_raises(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Baked mode with a non-40-hex provenance commit raises."""
    baked = tmp_path / "opt" / "afdb-toolkit"
    _baked_tree(baked, commit="deadbeefdeadbeef")
    monkeypatch.setattr("bspp.orchestration.runtime.toolkit.BAKED_TOOLKIT_ROOT", baked)

    spec = _runspec(tmp_path)
    with pytest.raises(BakedToolkitProvenanceError, match="not a 40-hex sha"):
        resolve_toolkit(spec)


def test_baked_missing_production_pipeline_raises(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Baked mode missing scripts/production_pipeline.py raises."""
    baked = tmp_path / "opt" / "afdb-toolkit"
    _baked_tree(baked)
    (baked / "scripts" / "production_pipeline.py").unlink()
    monkeypatch.setattr("bspp.orchestration.runtime.toolkit.BAKED_TOOLKIT_ROOT", baked)

    spec = _runspec(tmp_path)
    with pytest.raises(BakedToolkitLayoutError, match="missing required files"):
        resolve_toolkit(spec)


def test_baked_ipsae_not_executable_raises(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Baked mode with non-executable iPSAE raises."""
    baked = tmp_path / "opt" / "afdb-toolkit"
    _baked_tree(baked, executable_ipsae=False)
    monkeypatch.setattr("bspp.orchestration.runtime.toolkit.BAKED_TOOLKIT_ROOT", baked)

    spec = _runspec(tmp_path)
    with pytest.raises(BakedToolkitLayoutError, match="not executable"):
        resolve_toolkit(spec)


# ---------------------------------------------------------------------------
# resolve_toolkit — override mode (runtime, filesystem checks)
# ---------------------------------------------------------------------------


def test_override_happy_path(tmp_path: Path) -> None:
    """Override mode with mapped and valid container path succeeds."""
    host_toolkit = tmp_path / "host-toolkit"
    container_toolkit = tmp_path / "container-toolkit"
    _override_tree(container_toolkit)
    spec = _runspec(
        tmp_path,
        afdb_toolkit_repo=str(host_toolkit),
        mounts=[{"source": str(host_toolkit), "target": str(container_toolkit)}],
    )
    result = resolve_toolkit(spec)
    assert result.source == "override"
    assert result.root == container_toolkit
    assert result.provenance_commit is None
    assert result.production_pipeline == container_toolkit / "scripts" / "production_pipeline.py"


def test_override_no_mount_mapping_raises(tmp_path: Path) -> None:
    """Override declared but host path not covered by any mount raises."""
    host_toolkit = tmp_path / "host-toolkit"
    spec = _runspec(
        tmp_path,
        afdb_toolkit_repo=str(host_toolkit),
        mounts=[],
    )
    with pytest.raises(ToolkitOverrideNotFoundError, match="not mapped by any container mount"):
        resolve_toolkit(spec)


def test_override_mapped_path_missing_raises(tmp_path: Path) -> None:
    """Override mapped but the container target path does not exist raises."""
    host_toolkit = tmp_path / "host-toolkit"
    container_toolkit = tmp_path / "container-toolkit"
    # Don't create container_toolkit
    spec = _runspec(
        tmp_path,
        afdb_toolkit_repo=str(host_toolkit),
        mounts=[{"source": str(host_toolkit), "target": str(container_toolkit)}],
    )
    with pytest.raises(ToolkitOverrideNotFoundError, match="does not exist"):
        resolve_toolkit(spec)


def test_override_missing_production_pipeline_raises(tmp_path: Path) -> None:
    """Override with missing scripts/production_pipeline.py raises."""
    host_toolkit = tmp_path / "host-toolkit"
    container_toolkit = tmp_path / "container-toolkit"
    _override_tree(container_toolkit, include_production_pipeline=False)
    spec = _runspec(
        tmp_path,
        afdb_toolkit_repo=str(host_toolkit),
        mounts=[{"source": str(host_toolkit), "target": str(container_toolkit)}],
    )
    with pytest.raises(ToolkitOverrideLayoutError, match="missing required files"):
        resolve_toolkit(spec)


def test_override_missing_pyproject_raises(tmp_path: Path) -> None:
    """Override with missing pyproject.toml raises."""
    host_toolkit = tmp_path / "host-toolkit"
    container_toolkit = tmp_path / "container-toolkit"
    _override_tree(container_toolkit, include_pyproject=False)
    spec = _runspec(
        tmp_path,
        afdb_toolkit_repo=str(host_toolkit),
        mounts=[{"source": str(host_toolkit), "target": str(container_toolkit)}],
    )
    with pytest.raises(ToolkitOverrideLayoutError, match="missing required files"):
        resolve_toolkit(spec)


def test_override_wins_over_baked(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """When override is valid, baked is never consulted."""
    baked = tmp_path / "opt" / "afdb-toolkit"
    _baked_tree(baked)
    monkeypatch.setattr("bspp.orchestration.runtime.toolkit.BAKED_TOOLKIT_ROOT", baked)

    host_toolkit = tmp_path / "host-toolkit"
    container_toolkit = tmp_path / "container-toolkit"
    _override_tree(container_toolkit)
    spec = _runspec(
        tmp_path,
        afdb_toolkit_repo=str(host_toolkit),
        mounts=[{"source": str(host_toolkit), "target": str(container_toolkit)}],
    )
    result = resolve_toolkit(spec)
    assert result.source == "override"
    assert result.root == container_toolkit
    assert result.provenance_commit is None


def test_invalid_override_does_not_fallback_to_baked(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Invalid override raises error even when baked is valid."""
    baked = tmp_path / "opt" / "afdb-toolkit"
    _baked_tree(baked)
    monkeypatch.setattr("bspp.orchestration.runtime.toolkit.BAKED_TOOLKIT_ROOT", baked)

    host_toolkit = tmp_path / "host-toolkit"
    spec = _runspec(
        tmp_path,
        afdb_toolkit_repo=str(host_toolkit),
        mounts=[],  # no mount — unmapped
    )
    with pytest.raises(ToolkitOverrideNotFoundError):
        resolve_toolkit(spec)


# ---------------------------------------------------------------------------
# resolve_toolkit_for_container — planning variant (no filesystem checks)
# ---------------------------------------------------------------------------


def test_planning_baked_mode_returns_opt_afdb_toolkit(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Planning resolver returns baked /opt/afdb-toolkit without fs checks."""
    baked = tmp_path / "opt" / "afdb-toolkit"
    monkeypatch.setattr("bspp.orchestration.runtime.toolkit.BAKED_TOOLKIT_ROOT", baked)

    spec = _runspec(tmp_path)
    result = resolve_toolkit_for_container(spec)
    assert result.source == "baked"
    assert result.root == baked
    assert result.provenance_commit is None
    assert result.production_pipeline == baked / "scripts" / "production_pipeline.py"


def test_planning_override_mode_returns_mapped_path(tmp_path: Path) -> None:
    """Planning resolver translates override through mounts without fs checks."""
    host_toolkit = tmp_path / "host-toolkit"
    container_toolkit = tmp_path / "container-toolkit"
    spec = _runspec(
        tmp_path,
        afdb_toolkit_repo=str(host_toolkit),
        mounts=[{"source": str(host_toolkit), "target": str(container_toolkit)}],
    )
    result = resolve_toolkit_for_container(spec)
    assert result.source == "override"
    assert result.root == container_toolkit
    assert result.provenance_commit is None
    assert result.production_pipeline == container_toolkit / "scripts" / "production_pipeline.py"


def test_planning_override_no_mount_raises(tmp_path: Path) -> None:
    """Planning resolver raises when override has no mount mapping."""
    host_toolkit = tmp_path / "host-toolkit"
    spec = _runspec(
        tmp_path,
        afdb_toolkit_repo=str(host_toolkit),
        mounts=[],
    )
    with pytest.raises(ToolkitOverrideNotFoundError, match="not mapped by any container mount"):
        resolve_toolkit_for_container(spec)


def test_planning_override_does_not_check_filesystem(tmp_path: Path) -> None:
    """Planning resolver succeeds even when the container path doesn't exist."""
    host_toolkit = tmp_path / "host-toolkit"
    container_toolkit = tmp_path / "nonexistent-container"
    spec = _runspec(
        tmp_path,
        afdb_toolkit_repo=str(host_toolkit),
        mounts=[{"source": str(host_toolkit), "target": str(container_toolkit)}],
    )
    result = resolve_toolkit_for_container(spec)
    assert result.source == "override"
    assert result.root == container_toolkit
