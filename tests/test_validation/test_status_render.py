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

"""Tests for enriched status rendering."""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.runtime.status import format_status_report, render_full_status
from bspp.orchestration.runtime.validation.count_outputs import CountReport, ShardCount


def test_format_status_report_plain() -> None:
    out = format_status_report({"pending": 10, "done": 90}, dataset="ds1")

    assert "dataset: ds1" in out
    assert "pending" in out and "done" in out
    assert "100" in out  # total


def test_render_full_status_appends_count_report() -> None:
    shards = (
        ShardCount(shard_id=0, expected=5, actual=5, source="success_outputs"),
        ShardCount(shard_id=1, expected=5, actual=3, source="success_outputs"),
    )
    count_report = CountReport(
        output_dir=Path("/tmp/example"),
        total_models=10,
        required_shards=2,
        shards=shards,
    )

    rendered = render_full_status(
        {"done": 5, "processing": 5},
        dataset="dsA",
        count_report=count_report,
        use_rich=False,
    )

    assert "Mismatched shards: 1" in rendered
    assert "shard_1" in rendered


def test_render_full_status_rich_falls_back_gracefully(monkeypatch) -> None:
    # Force the rich import to fail, confirm we get plain output.
    import builtins

    real_import = builtins.__import__

    def _raise(name: str, *args, **kwargs):
        if name == "rich.progress_bar" or name == "rich.table" or name == "rich.console":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise)

    out = render_full_status({"done": 1}, dataset="dsA", use_rich=True)

    assert "dsA" in out
