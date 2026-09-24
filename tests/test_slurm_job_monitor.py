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

"""Tests for the sourceable, phase-agnostic Slurm job-monitor library."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "containers" / "scripts" / "slurm-job-monitor.sh"

_FAKE_SQUEUE = """#!/usr/bin/env bash
state_dir="$SLURM_MONITOR_TEST_DIR"
calls_file="$state_dir/squeue.calls"
n=$(cat "$calls_file" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$calls_file"
limit=$(cat "$state_dir/squeue.limit" 2>/dev/null || echo 1)
format=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --format=*) format="${1#--format=}" ;;
    --format) format="$2"; shift ;;
  esac
  shift
done
if [[ $n -le $limit ]]; then
  if [[ "$format" == "%i" ]]; then
    printf '12345\\n'
  else
    printf 'job=12345 state=RUNNING elapsed=00:00:30 node=node1\\n'
  fi
fi
"""

_FAKE_SACCT = """#!/usr/bin/env bash
state_dir="$SLURM_MONITOR_TEST_DIR"
calls_file="$state_dir/sacct.calls"
m=$(cat "$calls_file" 2>/dev/null || echo 0); m=$((m+1)); echo "$m" > "$calls_file"
empty=$(cat "$state_dir/sacct.empty" 2>/dev/null || echo 0)
row=$(cat "$state_dir/sacct.row" 2>/dev/null || echo '12345|COMPLETED|0:0|00:01:00|node1')
if [[ $m -le $empty ]]; then :; else printf '%s\\n' "$row"; fi
"""

_FAKE_SLEEP = """#!/usr/bin/env bash
exit 0
"""


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def _monitor_env(bin_dir: Path, state_dir: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SLURM_MONITOR_TEST_DIR": str(state_dir),
    }


def _run_monitor(
    bin_dir: Path,
    state_dir: Path,
    job_id: str = "12345",
    poll_seconds: str = "1",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; bspp_monitor_slurm_job "$2" "$3"',
            "test-shell",
            str(LIBRARY),
            job_id,
            poll_seconds,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_monitor_env(bin_dir, state_dir),
    )


def _make_fake_bin(bin_dir: Path, *, include_sacct: bool = True) -> None:
    _write_executable(bin_dir / "squeue", _FAKE_SQUEUE)
    if include_sacct:
        _write_executable(bin_dir / "sacct", _FAKE_SACCT)
    _write_executable(bin_dir / "sleep", _FAKE_SLEEP)


def test_library_is_source_only_and_has_bash_syntax() -> None:
    syntax = subprocess.run(["bash", "-n", str(LIBRARY)], capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr

    # Double-sourcing is a no-op and performs no work.
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; source "$1"; echo sourced-ok', "test-shell", str(LIBRARY)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "sourced-ok"


def test_monitor_returns_zero_when_job_completes(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "state"
    bin_dir.mkdir()
    state_dir.mkdir()
    _make_fake_bin(bin_dir)
    (state_dir / "squeue.limit").write_text("2")

    result = _run_monitor(bin_dir, state_dir)

    assert result.returncode == 0, result.stderr
    assert "12345|COMPLETED|0:0|00:01:00|node1" in result.stdout


def test_monitor_fails_on_terminal_failure_state(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "state"
    bin_dir.mkdir()
    state_dir.mkdir()
    _make_fake_bin(bin_dir)
    (state_dir / "squeue.limit").write_text("1")
    (state_dir / "sacct.row").write_text("12345|FAILED|1:0|00:00:10|node1")

    result = _run_monitor(bin_dir, state_dir)

    assert result.returncode != 0
    assert "reached terminal state FAILED" in result.stderr


def test_monitor_fails_on_cancelled_state(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "state"
    bin_dir.mkdir()
    state_dir.mkdir()
    _make_fake_bin(bin_dir)
    (state_dir / "squeue.limit").write_text("1")
    (state_dir / "sacct.row").write_text("12345|CANCELLED|0:15|00:00:00|")

    result = _run_monitor(bin_dir, state_dir)

    assert result.returncode != 0
    assert "reached terminal state CANCELLED" in result.stderr


def test_monitor_retries_accounting_race_and_then_completes(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "state"
    bin_dir.mkdir()
    state_dir.mkdir()
    _make_fake_bin(bin_dir)
    (state_dir / "squeue.limit").write_text("1")
    (state_dir / "sacct.empty").write_text("2")

    result = _run_monitor(bin_dir, state_dir)

    assert result.returncode == 0, result.stderr
    assert "12345|COMPLETED|0:0|00:01:00|node1" in result.stdout


def test_monitor_fails_when_accounting_never_publishes(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "state"
    bin_dir.mkdir()
    state_dir.mkdir()
    _make_fake_bin(bin_dir)
    (state_dir / "squeue.limit").write_text("1")
    (state_dir / "sacct.empty").write_text("999")

    result = _run_monitor(bin_dir, state_dir)

    assert result.returncode != 0
    assert "did not publish conclusive terminal evidence" in result.stderr


def test_monitor_rejects_invalid_job_id(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "state"
    bin_dir.mkdir()
    state_dir.mkdir()
    _make_fake_bin(bin_dir)

    result = _run_monitor(bin_dir, state_dir, job_id="not-a-job")

    assert result.returncode != 0
    assert "invalid job id" in result.stderr


def test_monitor_requires_squeue_and_sacct(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "state"
    bin_dir.mkdir()
    state_dir.mkdir()
    _make_fake_bin(bin_dir, include_sacct=False)

    result = _run_monitor(bin_dir, state_dir)

    assert result.returncode != 0
    assert "sacct is required" in result.stderr
