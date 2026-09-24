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

"""Structured Slurm monitoring observations for Control Plane reports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from tabulate import tabulate

_NO_VAL = {"", "none", "null", "n/a", "unknown", "4294967294", "4294967295"}
_ARRAY_SUFFIX_RE = re.compile(r"^(?P<base>\d+)(?:_\[?(?P<task>[^\]]+)\]?)?$")
_SACCT_JOB_ID_RE = re.compile(r"^\d+(?:[_.\[].+)?$")


@dataclass(frozen=True)
class SlurmCommandSnapshot:
    """One scheduler command result captured for structured monitoring."""

    kind: str
    argv: tuple[str, ...]
    returncode: int
    parser: str
    stderr: str = ""
    raw_json: Any | None = None
    raw_text: str | None = None
    warning: str | None = None

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        mapping: dict[str, object] = {
            "kind": self.kind,
            "argv": list(self.argv),
            "returncode": self.returncode,
            "parser": self.parser,
        }
        if self.stderr:
            mapping["stderr"] = self.stderr
        if self.raw_json is not None:
            mapping["raw_json"] = self.raw_json
        if self.raw_text is not None:
            mapping["raw_text"] = self.raw_text
        if self.warning is not None:
            mapping["warning"] = self.warning
        return mapping


@dataclass(frozen=True)
class SlurmJobRecord:
    """Normalized scheduler record from either squeue or sacct."""

    job_id: str
    source: str
    requested_job_id: str | None
    state: str | None = None
    name: str | None = None
    user: str | None = None
    partition: str | None = None
    exit_code: str | None = None
    elapsed: str | None = None
    time_limit: str | None = None
    max_rss: str | None = None
    req_mem: str | None = None
    nodes: str | None = None
    nodelist: str | None = None
    reason: str | None = None
    comment: str | None = None
    submitted_at: str | None = None
    raw: Any | None = None
    restarts: int | None = None

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        mapping: dict[str, object] = {
            "job_id": self.job_id,
            "source": self.source,
            "requested_job_id": self.requested_job_id,
        }
        for key in (
            "state",
            "name",
            "user",
            "partition",
            "exit_code",
            "elapsed",
            "time_limit",
            "max_rss",
            "req_mem",
            "nodes",
            "nodelist",
            "reason",
            "comment",
            "submitted_at",
            "restarts",
        ):
            value = getattr(self, key)
            if value is not None:
                mapping[key] = value
        if self.raw is not None:
            mapping["raw"] = self.raw
        return mapping


@dataclass(frozen=True)
class SlurmJobState:
    """Selected scheduler state for one requested Slurm job id."""

    job_id: str
    state: str
    source: str
    exit_code: str | None = None

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        mapping: dict[str, object] = {
            "job_id": self.job_id,
            "state": self.state,
            "source": self.source,
        }
        if self.exit_code is not None:
            mapping["exit_code"] = self.exit_code
        return mapping


@dataclass(frozen=True)
class SlurmObservation:
    """Structured observation from squeue and sacct for a set of job ids."""

    requested_job_ids: tuple[str, ...]
    squeue: SlurmCommandSnapshot
    sacct: SlurmCommandSnapshot
    squeue_jobs: tuple[SlurmJobRecord, ...]
    sacct_jobs: tuple[SlurmJobRecord, ...]
    selected_states: tuple[SlurmJobState, ...]
    warnings: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "requested_job_ids": list(self.requested_job_ids),
            "commands": {
                "squeue": self.squeue.to_mapping(),
                "sacct": self.sacct.to_mapping(),
            },
            "squeue_jobs": [job.to_mapping() for job in self.squeue_jobs],
            "sacct_jobs": [job.to_mapping() for job in self.sacct_jobs],
            "selected_states": [state.to_mapping() for state in self.selected_states],
            "warnings": list(self.warnings),
        }

    def selected_state_by_job_id(self) -> dict[str, SlurmJobState]:
        """Return selected states keyed by requested job id."""
        return {state.job_id: state for state in self.selected_states}

    def render_jobs_table(self) -> str:
        """Render a compact scheduler table from normalized records."""
        if not self.requested_job_ids:
            return "none\n"
        squeue_by_id = _records_by_requested_job_id(self.squeue_jobs)
        sacct_by_id = _records_by_requested_job_id(self.sacct_jobs)
        selected_by_id = self.selected_state_by_job_id()
        rows = []
        for job_id in self.requested_job_ids:
            squeue = squeue_by_id.get(job_id)
            sacct = sacct_by_id.get(job_id)
            selected = selected_by_id.get(job_id)
            rows.append(
                (
                    job_id,
                    _record_field(squeue, "state"),
                    _record_field(sacct, "state"),
                    selected.state if selected is not None else "",
                    selected.source if selected is not None else "",
                    _record_field(sacct, "exit_code"),
                    _record_field(sacct, "elapsed"),
                    _record_field(sacct, "req_mem"),
                )
            )
        return (
            tabulate(
                rows,
                headers=("job_id", "squeue", "sacct", "selected", "source", "exit_code", "elapsed", "req_mem"),
                tablefmt="github",
                stralign="left",
                disable_numparse=True,
            )
            + "\n"
        )

    def render_json(self) -> str:
        """Render the observation as stable JSON."""
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


def parse_squeue_json(stdout: str, *, requested: tuple[str, ...]) -> tuple[SlurmJobRecord, ...]:
    """Parse ``squeue --json`` output into normalized records."""
    payload = _load_json_payload(stdout, command="squeue")
    jobs = _payload_jobs(payload, command="squeue")
    requested_set = set(requested)
    records: list[SlurmJobRecord] = []
    for raw_job in jobs:
        if not isinstance(raw_job, dict):
            continue
        job_id = _squeue_job_id(raw_job)
        if job_id is None:
            continue
        records.append(
            SlurmJobRecord(
                job_id=job_id,
                source="squeue",
                requested_job_id=requested_job_id(job_id, requested_set=requested_set),
                state=normalize_state(_first_value(raw_job, "job_state", "state", "state_current")),
                name=_string_value(_first_value(raw_job, "name", "job_name")),
                user=_string_value(_first_value(raw_job, "user_name", "user")),
                partition=_string_value(_first_value(raw_job, "partition")),
                elapsed=_duration_value(_first_value(raw_job, "time_used", "elapsed", "time")),
                time_limit=_duration_value(_first_value(raw_job, "time_limit", "time_limit_raw")),
                nodes=_string_value(_first_value(raw_job, "nodes", "node_count", "minimum_nodes")),
                nodelist=_string_value(_first_value(raw_job, "nodes", "node_list", "nodelist")),
                reason=_string_value(_first_value(raw_job, "state_reason", "reason")),
                comment=_string_value(_first_value(raw_job, "comment")),
                submitted_at=_string_value(_first_value(raw_job, "submit_time", "submit", "time_submit")),
                raw=raw_job,
            )
        )
    return tuple(records)


def parse_sacct_json(
    stdout: str,
    *,
    requested: tuple[str, ...],
    strict_rows: bool = False,
) -> tuple[SlurmJobRecord, ...]:
    """Parse ``sacct --json`` output into normalized records."""
    payload = _load_json_payload(stdout, command="sacct")
    jobs = _payload_jobs(payload, command="sacct")
    requested_set = set(requested)
    records: list[SlurmJobRecord] = []
    allocation_bindings: dict[str, str] = {}
    logical_bindings: dict[str, str] = {}
    for row_number, raw_job in enumerate(jobs, start=1):
        if not isinstance(raw_job, dict):
            if strict_rows:
                raise ValueError(f"malformed sacct JSON row at index {row_number}")
            continue
        job_id = _sacct_job_id(raw_job)
        state = normalize_state(_first_value(raw_job, "state", "job_state"))
        if job_id is None or (strict_rows and (_SACCT_JOB_ID_RE.fullmatch(job_id) is None or not state)):
            if strict_rows:
                raise ValueError(f"malformed sacct JSON row at index {row_number}")
            continue
        raw_id = _string_value(_first_value(raw_job, "job_id_raw", "job_id", "jobid", "id"))
        if raw_id is not None:
            _bind_sacct_array_allocation(raw_id, job_id, allocation_bindings, logical_bindings)
        records.append(
            SlurmJobRecord(
                job_id=job_id,
                source="sacct",
                requested_job_id=requested_job_id(job_id, requested_set=requested_set),
                state=state,
                name=_string_value(_first_value(raw_job, "job_name", "name")),
                user=_string_value(_first_value(raw_job, "user_name", "user")),
                partition=_string_value(_first_value(raw_job, "partition")),
                exit_code=_exit_code_value(_first_value(raw_job, "exit_code", "exitcode")),
                elapsed=_duration_value(_first_value(raw_job, "elapsed", "time_used")),
                max_rss=_string_value(_first_value(raw_job, "max_rss", "maxrss")),
                req_mem=_memory_value(_first_value(raw_job, "req_mem", "required_memory", "required")),
                nodes=_string_value(_first_value(raw_job, "alloc_nodes", "nodes")),
                nodelist=_string_value(_first_value(raw_job, "node_list", "nodelist")),
                reason=_string_value(_first_value(raw_job, "state_reason", "reason")),
                comment=_string_value(_first_value(raw_job, "comment")),
                submitted_at=_string_value(_first_value(raw_job, "submit_time", "submit", "time_submit")),
                raw=raw_job,
                restarts=_restarts_value(_first_value(raw_job, "restarts", "Restarts", "restart_cnt")),
            )
        )
    return tuple(records)


def parse_parsable_state_rows(
    stdout: str,
    *,
    source: str,
    requested: tuple[str, ...],
) -> tuple[SlurmJobRecord, ...]:
    """Parse fallback Slurm ``|`` state rows into normalized records."""
    requested_set = set(requested)
    records: list[SlurmJobRecord] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id = parts[0].strip()
        requested_id = requested_job_id(job_id, requested_set=requested_set)
        if requested_id is None:
            continue
        state = normalize_state(parts[1])
        if not state:
            continue
        records.append(
            SlurmJobRecord(
                job_id=job_id,
                source=source,
                requested_job_id=requested_id,
                state=state,
                raw=line,
            )
        )
    return tuple(records)


def parse_sacct_parsable_rows(
    stdout: str,
    *,
    requested: tuple[str, ...],
) -> tuple[SlurmJobRecord, ...]:
    """Strictly parse JobIDRaw/State/ExitCode and optional final Restarts rows.

    This decoder is deliberately separate from the two-column squeue fallback:
    terminal accounting must retain an exact Slurm exit code before Resume can
    make durable lifecycle assertions.
    """
    requested_set = set(requested)
    records: list[SlurmJobRecord] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("|")
        if fields and fields[-1] == "":
            fields.pop()
        if len(fields) not in {3, 4}:
            raise ValueError(f"malformed sacct parsable row at line {line_number}")
        job_id, raw_state, exit_code = (field.strip() for field in fields[:3])
        restarts = fields[3].strip() if len(fields) == 4 else None
        state = normalize_state(raw_state)
        if _SACCT_JOB_ID_RE.fullmatch(job_id) is None or not state or re.fullmatch(r"\d+:\d+", exit_code) is None:
            raise ValueError(f"malformed sacct parsable row at line {line_number}")
        records.append(
            SlurmJobRecord(
                job_id=job_id,
                source="sacct",
                requested_job_id=requested_job_id(job_id, requested_set=requested_set),
                state=state,
                exit_code=exit_code,
                raw=line,
                restarts=_restarts_value(restarts),
            )
        )
    return tuple(records)


def parse_sacct_identity_parsable_rows(
    stdout: str,
    *,
    requested: tuple[str, ...],
) -> tuple[SlurmJobRecord, ...]:
    """Strictly parse identity-cross-checked terminal accounting rows.

    Some Slurm deployments expose an array allocation's numeric scheduler id in
    ``JobIDRaw`` while ``JobID`` retains the logical ``parent_task`` identity.
    Both columns are therefore required: a distinct numeric allocation id is
    bound by its paired logical id under a requested parent. Explicit raw task
    identities and step suffixes must agree. Slurm builds without Restarts omit
    that final column; its absence remains unknown rather than interpreted as zero.
    """
    requested_set = set(requested)
    records: list[SlurmJobRecord] = []
    allocation_bindings: dict[str, str] = {}
    logical_bindings: dict[str, str] = {}
    raw_re = re.compile(r"(?P<base>\d+)(?:_(?P<task>\d+))?(?P<step>\.[A-Za-z0-9_-]+)?")
    display_re = re.compile(r"(?P<base>\d+)(?:_(?P<task>\d+))?(?P<step>\.[A-Za-z0-9_-]+)?")
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("|")
        if fields and fields[-1] == "":
            fields.pop()
        if len(fields) not in {4, 5}:
            raise ValueError(f"malformed sacct identity parsable row at line {line_number}")
        raw_id, display_id, raw_state, exit_code = (field.strip() for field in fields[:4])
        restarts = fields[4].strip() if len(fields) == 5 else None
        raw_match = raw_re.fullmatch(raw_id)
        display_match = display_re.fullmatch(display_id)
        state = normalize_state(raw_state)
        if (
            raw_match is None
            or display_match is None
            or (
                raw_match.group("base") != display_match.group("base")
                and not (
                    raw_match.group("task") is None
                    and display_match.group("task") is not None
                    and display_match.group("base") in requested_set
                    and raw_match.group("base") not in requested_set
                )
            )
            or (raw_match.group("task") is not None and raw_match.group("task") != display_match.group("task"))
            or raw_match.group("step") != display_match.group("step")
            or state
            not in {
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
            or re.fullmatch(r"\d+:\d+", exit_code) is None
        ):
            raise ValueError(f"malformed sacct identity parsable row at line {line_number}")
        _bind_sacct_array_allocation(raw_id, display_id, allocation_bindings, logical_bindings)
        records.append(
            SlurmJobRecord(
                job_id=display_id,
                source="sacct",
                requested_job_id=(
                    display_match.group("base") if display_match.group("base") in requested_set else None
                ),
                state=state,
                exit_code=exit_code,
                raw=line,
                restarts=_restarts_value(restarts),
            )
        )
    return tuple(records)


def _bind_sacct_array_allocation(
    raw_id: str,
    logical_id: str,
    allocation_bindings: dict[str, str],
    logical_bindings: dict[str, str],
) -> None:
    """Reject conflicting physical allocation aliases, preserving logical raw IDs."""
    raw_allocation = raw_id.split(".", 1)[0]
    logical_allocation = logical_id.split(".", 1)[0]
    if not raw_allocation.isdecimal() or "_" not in logical_allocation:
        return
    if (
        allocation_bindings.get(raw_allocation, logical_allocation) != logical_allocation
        or logical_bindings.get(logical_allocation, raw_allocation) != raw_allocation
    ):
        raise ValueError("conflicting sacct array identity")
    allocation_bindings[raw_allocation] = logical_allocation
    logical_bindings[logical_allocation] = raw_allocation


def selected_records_by_job_id(records: tuple[SlurmJobRecord, ...]) -> dict[str, SlurmJobRecord]:
    """Return one record per requested job id, preferring richer top-level records."""
    selected: dict[str, SlurmJobRecord] = {}
    for record in records:
        if record.requested_job_id is None or not record.state:
            continue
        existing = selected.get(record.requested_job_id)
        if existing is None or _record_score(record) > _record_score(existing):
            selected[record.requested_job_id] = record
    return selected


def requested_job_id(job_id_raw: str, *, requested_set: set[str]) -> str | None:
    """Map a raw Slurm id or array id back to a requested base job id."""
    if "." in job_id_raw:
        return None
    if job_id_raw in requested_set:
        return job_id_raw
    match = _ARRAY_SUFFIX_RE.match(job_id_raw.strip())
    if match is not None and match.group("base") in requested_set:
        return match.group("base")
    return None


def normalize_state(value: object) -> str:
    """Return a stable upper-case Slurm state token from JSON or text values."""
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("current", "state", "value", "name"):
            state = normalize_state(value.get(key))
            if state:
                return state
        return ""
    if isinstance(value, list):
        for item in value:
            state = normalize_state(item)
            if state:
                return state
        return ""
    text = str(value).strip()
    return text.upper().split(maxsplit=1)[0] if text else ""


def _records_by_requested_job_id(records: tuple[SlurmJobRecord, ...]) -> dict[str, SlurmJobRecord]:
    return selected_records_by_job_id(records)


def _record_field(record: SlurmJobRecord | None, field: str) -> str:
    if record is None:
        return ""
    value = getattr(record, field)
    return value or ""


def _record_score(record: SlurmJobRecord) -> tuple[int, int]:
    top_level = 0 if "." in record.job_id else 1
    richness = sum(1 for field in (record.exit_code, record.elapsed, record.req_mem, record.name) if field)
    return (top_level, richness)


def _load_json_payload(stdout: str, *, command: str) -> object:
    try:
        return json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        msg = f"{command} --json returned invalid JSON: {exc.msg}"
        raise ValueError(msg) from exc


def _payload_jobs(payload: object, *, command: str) -> list[object]:
    if isinstance(payload, dict):
        jobs = payload.get("jobs")
        if isinstance(jobs, list):
            return jobs
        msg = f"{command} --json output is missing a jobs list"
        raise ValueError(msg)
    if isinstance(payload, list):
        return payload
    msg = f"{command} --json output has unexpected top-level type {type(payload).__name__}"
    raise ValueError(msg)


def _squeue_job_id(raw_job: dict[str, object]) -> str | None:
    raw_id = _slurm_json_id_value(_first_value(raw_job, "job_id", "jobid", "id"))
    if raw_id is None:
        return None
    array_job_id = _slurm_json_id_value(_first_value(raw_job, "array_job_id"))
    array_task_id = _slurm_json_array_task_value(_first_value(raw_job, "array_task_id", "array_task_string"))
    if array_job_id is not None and array_job_id != "0" and array_job_id.lower() not in _NO_VAL:
        task = _clean_array_task(array_task_id)
        if task is not None:
            return f"{array_job_id}_{task}"
        return array_job_id
    return raw_id


def _sacct_job_id(raw_job: dict[str, object]) -> str | None:
    raw_id = _string_value(_first_value(raw_job, "job_id_raw", "job_id", "jobid", "id"))
    array = raw_job.get("array")
    if array is None:
        return raw_id
    if not isinstance(array, dict):
        return None
    parent = _concrete_array_number(array.get("job_id"))
    if parent == "0":
        return raw_id
    if parent is None or raw_id is None:
        return None
    raw_match = re.fullmatch(r"(?P<base>\d+)(?:_(?P<task>\d+))?(?P<step>\.[A-Za-z0-9_-]+)?", raw_id)
    if raw_match is None:
        return None
    task_value = array.get("task_id")
    if isinstance(task_value, dict) and task_value.get("set") is False:
        # A parent-only row cannot prove an instantiated child. Keep an explicit
        # logical raw identity, but never promote a different allocation id.
        return raw_id if raw_match.group("base") == parent else None
    task = _concrete_array_number(task_value)
    if task is None:
        return None
    if raw_match.group("task") is not None and (raw_match.group("base") != parent or raw_match.group("task") != task):
        return None
    return f"{parent}_{task}{raw_match.group('step') or ''}"


def _concrete_array_number(value: object) -> str | None:
    """Read a finite concrete identity, including task zero but never an unset wrapper."""
    if isinstance(value, dict):
        if value.get("set") is not True or value.get("infinite") is not False:
            return None
        value = value.get("number")
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    rendered = str(value)
    if re.fullmatch(r"0|[1-9][0-9]*", rendered) is None or rendered in _NO_VAL:
        return None
    return rendered


def _clean_array_task(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned.lower() in _NO_VAL:
        return None
    return cleaned.strip("[]")


def _slurm_json_id_value(value: object) -> str | None:
    """Decode Slurm's numeric JSON wrappers without changing generic fields."""
    if isinstance(value, dict):
        if value.get("set") is False:
            return None
        for key in ("number", "value", "id"):
            if key in value:
                return _slurm_json_id_value(value[key])
        return None
    if isinstance(value, bool) or value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _slurm_json_array_task_value(value: object) -> str | None:
    """Decode a target-cluster array task wrapper, respecting its explicit ``set`` flag."""
    if isinstance(value, dict) and value.get("set") is False:
        return None
    return _slurm_json_id_value(value)


def _first_value(mapping: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _string_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("set", "number", "seconds", "value", "name", "current"):
            if key in value:
                nested = _string_value(value[key])
                if nested is not None:
                    return nested
        return None
    if isinstance(value, list):
        for item in value:
            nested = _string_value(item)
            if nested is not None:
                return nested
        return None
    text = str(value).strip()
    return text or None


def _duration_value(value: object) -> str | None:
    if isinstance(value, dict):
        for key in ("string", "set", "elapsed", "value", "number", "seconds"):
            if key in value:
                return _duration_value(value[key])
        return None
    if isinstance(value, (int, float)):
        return f"{int(value)}s"
    return _string_value(value)


def _memory_value(value: object) -> str | None:
    if isinstance(value, dict):
        for key in ("set", "memory", "value", "number"):
            if key in value:
                return _memory_value(value[key])
        return None
    return _string_value(value)


def _exit_code_value(value: object) -> str | None:
    if isinstance(value, dict):
        return_code = _slurm_json_id_value(value.get("return_code"))
        signal = _slurm_json_id_value(value.get("signal"))
        if return_code is not None and signal is not None:
            return f"{return_code}:{signal}"
        return None
    return _string_value(value)


def _restarts_value(value: object) -> int | None:
    """Normalize a sacct Restarts value to a non-negative int or None.

    Malformed or unavailable restarts accounting stays unresolved (None) rather
    than being synthesized, so lifecycle finalization can require it only on
    successful task records.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, dict):
        if value.get("set") is False:
            return None
        for key in ("number", "value", "id"):
            if key in value:
                return _restarts_value(value[key])
        return None
    if isinstance(value, list):
        for item in value:
            nested = _restarts_value(item)
            if nested is not None:
                return nested
        return None
    text = str(value).strip()
    if not text or text.lower() in _NO_VAL:
        return None
    try:
        parsed = int(text)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


__all__ = [
    "SlurmCommandSnapshot",
    "SlurmJobRecord",
    "SlurmJobState",
    "SlurmObservation",
    "normalize_state",
    "parse_parsable_state_rows",
    "parse_sacct_identity_parsable_rows",
    "parse_sacct_json",
    "parse_sacct_parsable_rows",
    "parse_squeue_json",
    "requested_job_id",
    "selected_records_by_job_id",
]
