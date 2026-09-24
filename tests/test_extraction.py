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

"""Tests for bspp.orchestration.runtime.extraction.archives."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.extraction.archives import (
    _create_marker,
    _is_extracted,
    _marker_path,
    extract_archive,
    extract_parallel,
    extract_sequential,
    find_archives,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_ARCHIVE = FIXTURES / "sample_archive.tar.lz4"


# ---------------------------------------------------------------------------
# find_archives
# ---------------------------------------------------------------------------


def test_find_archives(tmp_path: Path) -> None:
    """Flat directory with .tar.lz4 files should be discovered."""
    (tmp_path / "a.tar.lz4").touch()
    (tmp_path / "b.tar.lz4").touch()
    (tmp_path / "not_an_archive.txt").touch()

    result = find_archives(tmp_path)

    assert len(result) == 2
    assert all(p.suffix == ".lz4" for p in result)
    assert [p.name for p in result] == ["a.tar.lz4", "b.tar.lz4"]


def test_find_archives_nested(tmp_path: Path) -> None:
    """Archives inside nested subdirectories (dm layout) should be found."""
    nested = tmp_path / "structures" / "batch_001"
    nested.mkdir(parents=True)
    (nested / "chunk.tar.lz4").touch()
    (tmp_path / "top.tar.lz4").touch()

    result = find_archives(tmp_path)

    assert len(result) == 2
    names = {p.name for p in result}
    assert names == {"chunk.tar.lz4", "top.tar.lz4"}


def test_find_archives_empty(tmp_path: Path) -> None:
    """Empty directory returns an empty list."""
    assert find_archives(tmp_path) == []


def test_find_archives_nonexistent(tmp_path: Path) -> None:
    """Non-existent directory returns an empty list (no error)."""
    assert find_archives(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------


def test_marker_path(tmp_path: Path) -> None:
    archive = Path("/data/staging/chunk_00.tar.lz4")
    result = _marker_path(tmp_path, archive)
    assert result == tmp_path / "chunk_00.tar.lz4"


def test_is_extracted_false(tmp_path: Path) -> None:
    archive = Path("/data/staging/chunk_00.tar.lz4")
    assert _is_extracted(tmp_path, archive) is False


def test_create_marker_and_is_extracted(tmp_path: Path) -> None:
    archive = Path("/data/staging/chunk_00.tar.lz4")
    marker_dir = tmp_path / "markers"

    assert not marker_dir.exists()
    _create_marker(marker_dir, archive)

    assert marker_dir.exists()
    assert _is_extracted(marker_dir, archive) is True
    assert _marker_path(marker_dir, archive).is_file()


# ---------------------------------------------------------------------------
# extract_archive (requires lz4 + tar + the sample fixture)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_ARCHIVE.exists(), reason="sample_archive.tar.lz4 fixture missing")
def test_extract_archive(tmp_path: Path) -> None:
    """Extract the sample archive and verify the expected files appear."""
    import shutil

    archive_copy = tmp_path / "sample_archive.tar.lz4"
    shutil.copy2(SAMPLE_ARCHIVE, archive_copy)

    out = tmp_path / "output"
    count = extract_archive(archive_copy, out, keep_archive=True)

    assert count == 10  # 5 models x 2 files (pdb + json)
    assert (out / "AF-0000000000000001-model_v1.pdb").exists()
    assert (out / "AF-0000000000000005-meta_v1.json").exists()
    # keep_archive=True, so archive should still exist
    assert archive_copy.exists()


@pytest.mark.skipif(not SAMPLE_ARCHIVE.exists(), reason="sample_archive.tar.lz4 fixture missing")
def test_extract_archive_deletes_when_not_kept(tmp_path: Path) -> None:
    """With keep_archive=False the archive file is deleted after extraction."""
    import shutil

    archive_copy = tmp_path / "sample_archive.tar.lz4"
    shutil.copy2(SAMPLE_ARCHIVE, archive_copy)

    out = tmp_path / "output"
    extract_archive(archive_copy, out, keep_archive=False)

    assert not archive_copy.exists()


# ---------------------------------------------------------------------------
# extract_sequential
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_ARCHIVE.exists(), reason="sample_archive.tar.lz4 fixture missing")
def test_extract_sequential_with_markers(tmp_path: Path) -> None:
    """Sequential extraction creates markers; re-running skips extracted archives."""
    import shutil

    staging = tmp_path / "staging"
    staging.mkdir()
    shutil.copy2(SAMPLE_ARCHIVE, staging / "batch_01.tar.lz4")
    shutil.copy2(SAMPLE_ARCHIVE, staging / "batch_02.tar.lz4")

    out = tmp_path / "output"
    marker_dir = tmp_path / "markers"
    archives = find_archives(staging)

    total = extract_sequential(
        archives,
        out,
        keep_archives=True,
        marker_dir=marker_dir,
    )

    assert total == 20  # 10 files x 2 archives
    assert _is_extracted(marker_dir, archives[0])
    assert _is_extracted(marker_dir, archives[1])

    # Second run should skip everything
    total_again = extract_sequential(
        archives,
        out,
        keep_archives=True,
        marker_dir=marker_dir,
    )
    assert total_again == 0


# ---------------------------------------------------------------------------
# extract_parallel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_ARCHIVE.exists(), reason="sample_archive.tar.lz4 fixture missing")
def test_extract_parallel(tmp_path: Path) -> None:
    """Parallel extraction with workers=2 produces the same result as sequential."""
    import shutil

    staging = tmp_path / "staging"
    staging.mkdir()
    shutil.copy2(SAMPLE_ARCHIVE, staging / "batch_01.tar.lz4")
    shutil.copy2(SAMPLE_ARCHIVE, staging / "batch_02.tar.lz4")

    out = tmp_path / "output"
    marker_dir = tmp_path / "markers"
    archives = find_archives(staging)

    total = extract_parallel(
        archives,
        out,
        keep_archives=True,
        marker_dir=marker_dir,
        workers=2,
    )

    assert total == 20
    assert _is_extracted(marker_dir, archives[0])
    assert _is_extracted(marker_dir, archives[1])


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_extract_cli_help() -> None:
    """``bspp-orchestration extract --help`` should succeed and show options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["extract", "--help"])
    assert result.exit_code == 0
    assert "--staging-dir" in result.output
    assert "--output-dir" in result.output
    assert "--parallel" in result.output
    assert "--keep-archives" in result.output
    assert "--dry-run" in result.output
