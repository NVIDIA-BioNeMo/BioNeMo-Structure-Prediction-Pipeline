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

"""Direct Slurm submission and monitoring helpers."""

from __future__ import annotations

import glob
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)

FAILURE_STATES = TERMINAL_STATES - {"COMPLETED"}

_JOB_ID_RE = re.compile(r"^(?P<job_id>\d+)(?:;[^\s]+)?$")
_NUMERIC_TASK_RE = re.compile(r"_(?P<task_id>\d+)$")


@dataclass(frozen=True)
class CommandResult:
    """Completed command output captured for Slurm evidence."""

    returncode: int
    stdout: str
    stderr: str

    def to_redacted_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready command data."""
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


CommandRunner = Callable[[Sequence[str]], CommandResult]
SleepFn = Callable[[float], None]


@dataclass(frozen=True)
class SlurmSubmissionEvidence:
    """Evidence from one sbatch submission attempt."""

    argv: tuple[str, ...]
    result: CommandResult
    job_id: str | None

    @property
    def success(self) -> bool:
        """Return true when sbatch succeeded and yielded a job id."""
        return self.result.returncode == 0 and self.job_id is not None

    def to_redacted_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready submission evidence."""
        return {
            "argv": list(self.argv),
            "returncode": self.result.returncode,
            "stdout": self.result.stdout,
            "stderr": self.result.stderr,
            "job_id": self.job_id,
            "success": self.success,
        }


@dataclass(frozen=True)
class SqueueSnapshot:
    """One squeue poll captured while monitoring a job."""

    poll_index: int
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    in_queue: bool

    def to_redacted_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready squeue evidence."""
        return {
            "poll_index": self.poll_index,
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "in_queue": self.in_queue,
        }


@dataclass(frozen=True)
class SacctRow:
    """One allocation-level sacct row."""

    job_id: str
    job_id_raw: str
    state: str
    exit_code: str
    elapsed: str
    node_list: str

    @property
    def task_id(self) -> int | None:
        """Return the numeric array task id when this is an expanded task row."""
        match = _NUMERIC_TASK_RE.search(self.job_id)
        return int(match.group("task_id")) if match is not None else None

    @property
    def normalized_state(self) -> str:
        """Return the state token used for terminal/success decisions."""
        return self.state.strip().upper().split()[0] if self.state.strip() else ""

    @property
    def is_terminal(self) -> bool:
        """Return true when Slurm accounting reports a terminal state."""
        return self.normalized_state in TERMINAL_STATES

    @property
    def is_success(self) -> bool:
        """Return true for completed rows with zero Slurm exit code."""
        return self.normalized_state == "COMPLETED" and self.exit_code == "0:0"

    def to_redacted_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready accounting evidence."""
        return {
            "job_id": self.job_id,
            "job_id_raw": self.job_id_raw,
            "state": self.state,
            "exit_code": self.exit_code,
            "elapsed": self.elapsed,
            "node_list": self.node_list,
            "task_id": self.task_id,
            "is_terminal": self.is_terminal,
            "is_success": self.is_success,
        }


@dataclass(frozen=True)
class MonitorResult:
    """Terminal Slurm monitoring evidence for a single job id."""

    job_id: str
    is_array: bool
    success: bool
    accounting_complete: bool
    expected_task_ids: tuple[int, ...]
    completed_task_ids: tuple[int, ...]
    rows: tuple[SacctRow, ...]
    poll_snapshots: tuple[SqueueSnapshot, ...]
    log_tails: Mapping[str, str]
    failure_reason: str | None = None

    @property
    def expected_task_count(self) -> int:
        """Return expected task count for array evidence."""
        return len(self.expected_task_ids)

    @property
    def final_states(self) -> tuple[str, ...]:
        """Return compact final accounting states."""
        return tuple(f"{row.job_id}|{row.state}|{row.exit_code}" for row in self.rows)

    def to_redacted_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready monitoring evidence."""
        return {
            "job_id": self.job_id,
            "is_array": self.is_array,
            "success": self.success,
            "accounting_complete": self.accounting_complete,
            "expected_task_ids": list(self.expected_task_ids),
            "expected_task_count": self.expected_task_count,
            "completed_task_ids": list(self.completed_task_ids),
            "final_states": list(self.final_states),
            "rows": [row.to_redacted_dict() for row in self.rows],
            "poll_snapshots": [snapshot.to_redacted_dict() for snapshot in self.poll_snapshots],
            "log_tails": dict(sorted(self.log_tails.items())),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class _AccountingEvaluation:
    complete: bool
    success: bool
    completed_task_ids: tuple[int, ...]
    failure_reason: str | None


def default_command_runner(argv: Sequence[str]) -> CommandResult:
    """Run a command and return captured stdout/stderr."""
    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def parse_job_id(stdout: str) -> str:
    """Parse Slurm ``sbatch --parsable`` output into a bare numeric job id."""
    text = stdout.strip().splitlines()[0] if stdout.strip() else ""
    match = _JOB_ID_RE.fullmatch(text)
    if match is None:
        msg = f"Could not parse sbatch job id from output: {stdout!r}"
        raise ValueError(msg)
    return match.group("job_id")


def submit_sbatch(
    script: Path,
    *,
    runner: CommandRunner = default_command_runner,
) -> SlurmSubmissionEvidence:
    """Submit an sbatch script and capture parsable job-id evidence."""
    argv = ("sbatch", "--parsable", str(script))
    result = runner(argv)
    job_id = None
    if result.returncode == 0:
        try:
            job_id = parse_job_id(result.stdout)
        except ValueError:
            job_id = None
    return SlurmSubmissionEvidence(argv=argv, result=result, job_id=job_id)


def query_sacct(
    job_id: str,
    *,
    runner: CommandRunner = default_command_runner,
) -> tuple[SacctRow, ...]:
    """Query allocation-level Slurm accounting rows for a job id."""
    argv = (
        "sacct",
        "-X",
        "-j",
        job_id,
        "--format=JobID,JobIDRaw,State,ExitCode,Elapsed,NodeList",
        "-n",
        "-P",
    )
    result = runner(argv)
    if result.returncode != 0:
        return ()
    return _parse_sacct_rows(result.stdout)


def monitor_job(
    job_id: str,
    *,
    is_array: bool,
    expected_task_ids: Sequence[int] = (),
    runner: CommandRunner = default_command_runner,
    sleep: SleepFn = time.sleep,
    poll_interval: float = 30.0,
    max_polls: int = 720,
    sacct_retries: int = 5,
    log_patterns: Sequence[str] = (),
) -> MonitorResult:
    """Poll squeue/sacct until a job reaches terminal accounting or fails incomplete."""
    snapshots: list[SqueueSnapshot] = []
    expected = tuple(expected_task_ids)
    rows: tuple[SacctRow, ...] = ()
    evaluation = _evaluate_accounting(rows, is_array=is_array, expected_task_ids=expected)
    if max_polls < 1:
        max_polls = 1
    for poll_index in range(max_polls):
        snapshot = _poll_squeue(job_id, poll_index=poll_index, runner=runner)
        snapshots.append(snapshot)
        rows = query_sacct(job_id, runner=runner)
        evaluation = _evaluate_accounting(rows, is_array=is_array, expected_task_ids=expected)
        if evaluation.complete:
            return _monitor_result(
                job_id,
                is_array=is_array,
                expected_task_ids=expected,
                rows=rows,
                snapshots=tuple(snapshots),
                evaluation=evaluation,
                log_patterns=log_patterns,
            )
        if not snapshot.in_queue:
            rows, evaluation = _retry_sacct_until_complete(
                job_id,
                is_array=is_array,
                expected_task_ids=expected,
                runner=runner,
                sleep=sleep,
                poll_interval=poll_interval,
                sacct_retries=sacct_retries,
            )
            return _monitor_result(
                job_id,
                is_array=is_array,
                expected_task_ids=expected,
                rows=rows,
                snapshots=tuple(snapshots),
                evaluation=evaluation,
                log_patterns=log_patterns,
            )
        sleep(poll_interval)

    return _monitor_result(
        job_id,
        is_array=is_array,
        expected_task_ids=expected,
        rows=rows,
        snapshots=tuple(snapshots),
        evaluation=_AccountingEvaluation(
            complete=False,
            success=False,
            completed_task_ids=evaluation.completed_task_ids,
            failure_reason="max_polls_exhausted",
        ),
        log_patterns=log_patterns,
    )


def collect_log_tails(
    patterns: Sequence[str],
    *,
    job_id: str,
    is_array: bool,
    tail_lines: int = 40,
) -> dict[str, str]:
    """Collect best-effort log tails from Slurm output/error patterns."""
    result: dict[str, str] = {}
    for pattern in patterns:
        rendered = _render_log_pattern(pattern, job_id=job_id, is_array=is_array)
        for match in sorted(glob.glob(rendered)):
            try:
                result[match] = _tail_text(Path(match), tail_lines=tail_lines)
            except OSError as exc:
                result[match] = f"<unable to read log tail: {exc}>"
    return result


def _poll_squeue(
    job_id: str,
    *,
    poll_index: int,
    runner: CommandRunner,
) -> SqueueSnapshot:
    argv = ("squeue", "-h", "-j", job_id)
    result = runner(argv)
    return SqueueSnapshot(
        poll_index=poll_index,
        argv=argv,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        in_queue=result.returncode != 0 or bool(result.stdout.strip()),
    )


def _retry_sacct_until_complete(
    job_id: str,
    *,
    is_array: bool,
    expected_task_ids: tuple[int, ...],
    runner: CommandRunner,
    sleep: SleepFn,
    poll_interval: float,
    sacct_retries: int,
) -> tuple[tuple[SacctRow, ...], _AccountingEvaluation]:
    rows: tuple[SacctRow, ...] = ()
    evaluation = _evaluate_accounting(rows, is_array=is_array, expected_task_ids=expected_task_ids)
    retry_count = max(1, sacct_retries)
    for retry_index in range(retry_count):
        if retry_index > 0:
            sleep(poll_interval)
        rows = query_sacct(job_id, runner=runner)
        evaluation = _evaluate_accounting(rows, is_array=is_array, expected_task_ids=expected_task_ids)
        if evaluation.complete:
            return rows, evaluation
    return rows, _AccountingEvaluation(
        complete=False,
        success=False,
        completed_task_ids=evaluation.completed_task_ids,
        failure_reason=evaluation.failure_reason or "accounting_incomplete",
    )


def _evaluate_accounting(
    rows: tuple[SacctRow, ...],
    *,
    is_array: bool,
    expected_task_ids: tuple[int, ...],
) -> _AccountingEvaluation:
    if is_array:
        return _evaluate_array_accounting(rows, expected_task_ids=expected_task_ids)
    return _evaluate_single_accounting(rows)


def _evaluate_array_accounting(
    rows: tuple[SacctRow, ...],
    *,
    expected_task_ids: tuple[int, ...],
) -> _AccountingEvaluation:
    expected = set(expected_task_ids)
    expanded_rows = tuple(row for row in rows if row.task_id is not None)
    terminal_by_task: dict[int, list[SacctRow]] = {}
    for row in expanded_rows:
        if row.is_terminal:
            assert row.task_id is not None
            terminal_by_task.setdefault(row.task_id, []).append(row)

    completed = tuple(
        sorted(task_id for task_id, task_rows in terminal_by_task.items() if any(row.is_success for row in task_rows))
    )
    missing = sorted(expected - set(terminal_by_task))
    if missing:
        reason = "accounting_incomplete"
        if any(_is_unexpanded_array_row(row) for row in rows):
            reason = "accounting_unexpanded_array_row"
        return _AccountingEvaluation(
            complete=False,
            success=False,
            completed_task_ids=completed,
            failure_reason=reason,
        )

    expected_terminal_rows = [row for task_id in sorted(expected) for row in terminal_by_task.get(task_id, ())]
    if not expected_terminal_rows and expected:
        return _AccountingEvaluation(
            complete=False,
            success=False,
            completed_task_ids=(),
            failure_reason="accounting_incomplete",
        )
    failed = [row for row in expected_terminal_rows if not row.is_success]
    if failed:
        return _AccountingEvaluation(
            complete=True,
            success=False,
            completed_task_ids=completed,
            failure_reason="terminal_failure",
        )
    return _AccountingEvaluation(
        complete=True,
        success=True,
        completed_task_ids=completed,
        failure_reason=None,
    )


def _evaluate_single_accounting(rows: tuple[SacctRow, ...]) -> _AccountingEvaluation:
    terminal_rows = tuple(row for row in rows if row.task_id is None and row.is_terminal)
    if not terminal_rows:
        return _AccountingEvaluation(
            complete=False,
            success=False,
            completed_task_ids=(),
            failure_reason="accounting_incomplete",
        )
    failed = tuple(row for row in terminal_rows if not row.is_success)
    return _AccountingEvaluation(
        complete=True,
        success=not failed and any(row.is_success for row in terminal_rows),
        completed_task_ids=(),
        failure_reason="terminal_failure" if failed else None,
    )


def _monitor_result(
    job_id: str,
    *,
    is_array: bool,
    expected_task_ids: tuple[int, ...],
    rows: tuple[SacctRow, ...],
    snapshots: tuple[SqueueSnapshot, ...],
    evaluation: _AccountingEvaluation,
    log_patterns: Sequence[str],
) -> MonitorResult:
    return MonitorResult(
        job_id=job_id,
        is_array=is_array,
        success=evaluation.complete and evaluation.success,
        accounting_complete=evaluation.complete,
        expected_task_ids=expected_task_ids,
        completed_task_ids=evaluation.completed_task_ids,
        rows=rows,
        poll_snapshots=snapshots,
        log_tails=collect_log_tails(log_patterns, job_id=job_id, is_array=is_array),
        failure_reason=evaluation.failure_reason,
    )


def _parse_sacct_rows(stdout: str) -> tuple[SacctRow, ...]:
    rows: list[SacctRow] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split("|")
        if len(fields) < 6:
            continue
        rows.append(
            SacctRow(
                job_id=fields[0],
                job_id_raw=fields[1],
                state=fields[2],
                exit_code=fields[3],
                elapsed=fields[4],
                node_list=fields[5],
            )
        )
    return tuple(rows)


def _is_unexpanded_array_row(row: SacctRow) -> bool:
    return "_[" in row.job_id


def _render_log_pattern(pattern: str, *, job_id: str, is_array: bool) -> str:
    rendered = pattern.replace("%A", job_id).replace("%j", job_id)
    return rendered.replace("%a", "*" if is_array else "0")


def _tail_text(path: Path, *, tail_lines: int) -> str:
    lines = path.read_text(errors="replace").splitlines()
    tail = lines[-tail_lines:] if tail_lines > 0 else lines
    return "\n".join(tail) + ("\n" if tail else "")


__all__ = [
    "CommandResult",
    "CommandRunner",
    "MonitorResult",
    "SacctRow",
    "SlurmSubmissionEvidence",
    "collect_log_tails",
    "default_command_runner",
    "monitor_job",
    "parse_job_id",
    "query_sacct",
    "submit_sbatch",
]
