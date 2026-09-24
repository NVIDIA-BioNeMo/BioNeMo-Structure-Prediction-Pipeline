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

"""Tests for bspp.orchestration.runtime.postprocessing.shard_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from bspp.orchestration.runtime.postprocessing.shard_config import (
    ShardConfig,
    compute_archive_shard_config,
    compute_archive_shard_config_model,
    compute_shard_config,
    compute_shard_config_model,
    load_shard_config,
    read_shard_config,
    write_shard_config,
)


def test_compute_shard_config_single_shard() -> None:
    """Few models fit into a single shard."""
    cfg = compute_shard_config(100, max_per_shard=5000)
    assert cfg["total_models"] == 100
    assert cfg["max_per_shard"] == 5000
    assert cfg["required_shards"] == 1
    assert cfg["array_range"] == "0-0"


def test_compute_shard_config_multiple_shards() -> None:
    """12000 models at 5000/shard requires 3 shards."""
    cfg = compute_shard_config(12000, max_per_shard=5000)
    assert cfg["required_shards"] == 3
    assert cfg["array_range"] == "0-2"


def test_compute_shard_config_exact_boundary() -> None:
    """Exactly on the boundary: 10000 models / 5000 per shard = 2 shards."""
    cfg = compute_shard_config(10000, max_per_shard=5000)
    assert cfg["required_shards"] == 2
    assert cfg["array_range"] == "0-1"


def test_compute_shard_config_zero_models() -> None:
    """Zero models still produces at least one shard."""
    cfg = compute_shard_config(0, max_per_shard=5000)
    assert cfg["required_shards"] == 1
    assert cfg["array_range"] == "0-0"


def test_compute_shard_config_one_over_boundary() -> None:
    """One model over a boundary bumps the shard count."""
    cfg = compute_shard_config(5001, max_per_shard=5000)
    assert cfg["required_shards"] == 2
    assert cfg["array_range"] == "0-1"


def test_compute_shard_config_invalid_max_per_shard() -> None:
    with pytest.raises(ValueError, match="max_per_shard must be positive"):
        compute_shard_config(100, max_per_shard=0)


def test_compute_shard_config_model() -> None:
    cfg = compute_shard_config_model(12000, max_per_shard=5000)
    assert isinstance(cfg, ShardConfig)
    assert cfg.required_shards == 3
    assert cfg.to_compat_dict() == compute_shard_config(12000, max_per_shard=5000)


def test_compute_archive_shard_config() -> None:
    archives = ["batch_01.tar.lz4", "batch_02.tar.lz4", "batch_03.tar.lz4"]
    cfg = compute_archive_shard_config(archives, shards_per_archive=2)
    assert cfg["archive_mode"] is True
    assert cfg["total_archives"] == 3
    assert cfg["shards_per_archive"] == 2
    assert cfg["required_shards"] == 3
    assert cfg["array_range"] == "0-2"


def test_compute_archive_shard_config_empty() -> None:
    cfg = compute_archive_shard_config([], shards_per_archive=2)
    assert cfg["total_archives"] == 0
    assert cfg["required_shards"] == 0
    assert cfg["array_range"] == "0-0"


def test_compute_archive_shard_config_model() -> None:
    cfg = compute_archive_shard_config_model(["a.tar.lz4"], shards_per_archive=4)
    assert cfg.archive_mode is True
    assert cfg.total_archives == 1
    assert cfg.to_compat_dict() == compute_archive_shard_config(["a.tar.lz4"], shards_per_archive=4)


def test_write_read_shard_config_roundtrip(tmp_path: Path) -> None:
    original = compute_shard_config(7500, max_per_shard=5000)
    path = tmp_path / "shard_config.json"
    write_shard_config(original, path)

    loaded = read_shard_config(path)
    assert loaded == original


def test_load_shard_config_model_roundtrip(tmp_path: Path) -> None:
    original = compute_shard_config_model(7500, max_per_shard=5000)
    path = tmp_path / "shard_config.json"
    write_shard_config(original, path)

    loaded = load_shard_config(path)
    assert loaded.required_shards == original.required_shards
    assert loaded.to_compat_dict() == original.to_compat_dict()


def test_load_shard_config_model_rejects_inconsistent_array_range(tmp_path: Path) -> None:
    path = tmp_path / "shard_config.json"
    write_shard_config({"total_models": 10, "max_per_shard": 5, "required_shards": 2, "array_range": "0-2"}, path)

    with pytest.raises(ValueError, match="array_range"):
        load_shard_config(path)


def test_load_shard_config_model_rejects_inconsistent_archive_count(tmp_path: Path) -> None:
    path = tmp_path / "shard_config.json"
    write_shard_config(
        {
            "archive_mode": True,
            "total_archives": 3,
            "shards_per_archive": 2,
            "required_shards": 2,
            "array_range": "0-1",
        },
        path,
    )

    with pytest.raises(ValueError, match="total_archives"):
        load_shard_config(path)


def test_write_shard_config_creates_parents(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "shard_config.json"
    write_shard_config({"test": True}, nested)
    assert nested.exists()


def test_write_shard_config_is_atomic(tmp_path: Path) -> None:
    """Regression: writes go through tempfile + os.replace.

    A reader that lands mid-rewrite must observe either the previous
    complete file or the new complete file — never a truncated/empty
    file. We can't easily race the test reliably, but we can verify the
    visible contract: no leftover .tmp file, and the destination is
    always parseable JSON after the call returns.
    """
    target = tmp_path / "shard_config.json"
    target.write_text('{"required_shards": 1, "array_range": "0-0"}\n')

    # Now overwrite with a larger config; previous content must remain
    # parseable until the moment of replacement.
    write_shard_config(compute_shard_config(12000, max_per_shard=5000), target)

    assert target.exists()
    # No .tmp turd left over.
    assert not (target.parent / "shard_config.json.tmp").exists()
    # Final file must be parseable JSON with the new content.
    loaded = read_shard_config(target)
    assert loaded["required_shards"] == 3


def test_write_shard_config_does_not_corrupt_destination_on_dump_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If serialization fails mid-write, the destination is unchanged.

    Atomic semantics require that a failed write cannot trash the
    previous file. Monkeypatch json.dumps to raise after we've staged a
    "previous" file, then assert the destination is still readable.
    """
    import json as _json

    target = tmp_path / "shard_config.json"
    write_shard_config(compute_shard_config(100, max_per_shard=5000), target)
    previous_bytes = target.read_bytes()

    def _boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("serialization exploded")

    monkeypatch.setattr(_json, "dumps", _boom)

    with pytest.raises(RuntimeError, match="serialization exploded"):
        write_shard_config({"required_shards": 99}, target)

    # Destination unchanged; no half-written or empty file.
    assert target.read_bytes() == previous_bytes
