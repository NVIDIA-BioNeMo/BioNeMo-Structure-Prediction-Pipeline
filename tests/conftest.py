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

"""Shared fixtures and marker registration for the bspp-orchestration test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e_full: marks tests requiring full afdb-toolkit[production] deps")


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def input_dir(fixtures_dir: Path) -> Path:
    """Path to fixture input models."""
    return fixtures_dir / "input"


@pytest.fixture
def sample_archive(fixtures_dir: Path) -> Path:
    """Path to the sample .tar.lz4 archive."""
    return fixtures_dir / "sample_archive.tar.lz4"


@pytest.fixture
def manifest_csv(fixtures_dir: Path) -> Path:
    """Path to fixture manifest CSV."""
    return fixtures_dir / "config" / "manifest.csv"


@pytest.fixture
def master_parquet(fixtures_dir: Path) -> Path:
    """Path to fixture master parquet."""
    return fixtures_dir / "master.parquet"


@pytest.fixture
def tracking_parquet(fixtures_dir: Path) -> Path:
    """Path to fixture tracking parquet."""
    return fixtures_dir / "tracking.parquet"


@pytest.fixture
def uniprot_db(fixtures_dir: Path) -> Path:
    """Path to fixture UniProt DuckDB."""
    return fixtures_dir / "uniprot_test.duckdb"


@pytest.fixture
def expected_outputs_dir(fixtures_dir: Path) -> Path:
    """Path to fixture expected pipeline outputs."""
    return fixtures_dir / "expected_outputs"
