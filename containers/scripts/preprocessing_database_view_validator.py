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

"""Build and validate the immutable schema-2 two-root database view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

NAMES = (
    "colabfold_envdb_202108_db",
    "colabfold_envdb_202108_db.dbtype",
    "colabfold_envdb_202108_db.index",
    "colabfold_envdb_202108_db_aln",
    "colabfold_envdb_202108_db_aln.dbtype",
    "colabfold_envdb_202108_db_aln.index",
    "colabfold_envdb_202108_db_h",
    "colabfold_envdb_202108_db_h.dbtype",
    "colabfold_envdb_202108_db_h.index",
    "colabfold_envdb_202108_db_seq",
    "colabfold_envdb_202108_db_seq.dbtype",
    "colabfold_envdb_202108_db_seq.index",
    "colabfold_envdb_202108_db_seq_h",
    "colabfold_envdb_202108_db_seq_h.dbtype",
    "colabfold_envdb_202108_db_seq_h.index",
    "uniref30_2302_db",
    "uniref30_2302_db.dbtype",
    "uniref30_2302_db.index",
    "uniref30_2302_db.lookup",
    "uniref30_2302_db_aln",
    "uniref30_2302_db_aln.dbtype",
    "uniref30_2302_db_aln.index",
    "uniref30_2302_db_h",
    "uniref30_2302_db_h.dbtype",
    "uniref30_2302_db_h.index",
    "uniref30_2302_db_mapping",
    "uniref30_2302_db_pad",
    "uniref30_2302_db_pad.dbtype",
    "uniref30_2302_db_pad.index",
    "uniref30_2302_db_pad.lookup",
    "uniref30_2302_db_pad_h",
    "uniref30_2302_db_pad_h.dbtype",
    "uniref30_2302_db_pad_h.index",
    "uniref30_2302_db_seq",
    "uniref30_2302_db_seq.dbtype",
    "uniref30_2302_db_seq.index",
    "uniref30_2302_db_seq_h",
    "uniref30_2302_db_seq_h.dbtype",
    "uniref30_2302_db_seq_h.index",
    "uniref30_2302_db_taxonomy",
)
RECOVERED = frozenset({"uniref30_2302_db_mapping", "uniref30_2302_db_taxonomy"})
TOP_KEYS = frozenset(
    {
        "schema_version",
        "logical_primary_name",
        "physical_primary_name",
        "database_source",
        "recovered_root",
        "database_view",
        "entries",
        "materialization_manifest_sha256",
        "materialization_outputs",
        "database_content_identity",
        "execution_context",
    }
)
ENTRY_KEYS = frozenset(
    {
        "logical_name",
        "root_kind",
        "physical_name",
        "physical_source",
        "resolved_source",
        "source_kind",
        "resolved_kind",
        "resolved_mode",
        "resolved_size_bytes",
        "resolved_mtime_ns",
        "readable",
        "hash_policy",
        "sha256",
    }
)
MATERIALIZATION_TOP_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "outcome",
        "intent_key",
        "intent",
        "execution_context",
        "archive_before",
        "archive_after",
        "archive_stat_unchanged",
        "archive_sha256",
        "scan",
        "inventory",
        "outputs",
        "size_findings",
        "errors",
    }
)
MATERIALIZATION_OUTPUT_KEYS = frozenset(
    {
        "logical_name",
        "file_name",
        "source_member",
        "source_basename",
        "declared_size_bytes",
        "size_bytes",
        "sha256",
        "status",
        "expected_size_bytes",
        "size_matches_expected",
    }
)
VIEW_MATERIALIZATION_OUTPUT_KEYS = frozenset({"logical_name", "size_bytes", "sha256"})
EXECUTION_CONTEXT_KEYS = frozenset(
    {"context_kind", "slurm_job_id", "slurmd_nodename", "tool_sha256", "python_executable", "python_version"}
)


def execution_context() -> dict[str, str | None]:
    if sys.version_info[:2] != (3, 12):
        raise ValueError(f"database view tool requires pinned Python 3.12, observed {platform.python_version()}")
    try:
        executable = str(Path(sys.executable).resolve(strict=True))
    except OSError as exc:
        raise ValueError(f"Python executable cannot be resolved: {sys.executable}: {exc}") from exc
    job, node = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURMD_NODENAME")
    if bool(job) != bool(node) or (job is not None and any(part.isspace() for part in (job, node or ""))):
        raise ValueError("Slurm execution context must provide non-whitespace job and node together")
    with Path(__file__).open("rb") as handle:
        tool_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "context_kind": "slurm" if job else "local-test",
        "slurm_job_id": job,
        "slurmd_nodename": node,
        "tool_sha256": tool_sha256,
        "python_executable": executable,
        "python_version": platform.python_version(),
    }


def _validate_execution_context(value: object) -> None:
    if not isinstance(value, dict) or set(value) != EXECUTION_CONTEXT_KEYS:
        raise ValueError("execution_context has an unexpected shape")
    context_kind, job, node = value["context_kind"], value["slurm_job_id"], value["slurmd_nodename"]
    if context_kind == "slurm":
        if (
            not isinstance(job, str)
            or not job
            or not isinstance(node, str)
            or not node
            or any(part.isspace() for part in (job, node))
        ):
            raise ValueError("Slurm execution_context is incoherent")
    elif context_kind == "local-test":
        if job is not None or node is not None:
            raise ValueError("local-test execution_context must not claim a Slurm job or node")
    else:
        raise ValueError("execution_context.context_kind is invalid")
    current = execution_context()
    for key in ("tool_sha256", "python_executable", "python_version"):
        if value[key] != current[key]:
            raise ValueError(f"execution_context.{key} does not match the current pinned runtime")


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _physical(name: str) -> str:
    if name.startswith("uniref30_2302_db_h"):
        return name.replace("uniref30_2302_db_h", "uniref30_2302_db_pad_h", 1)
    if name.startswith("uniref30_2302_db") and name not in RECOVERED:
        suffix = name.removeprefix("uniref30_2302_db")
        if suffix in {"", ".dbtype", ".index", ".lookup"}:
            return "uniref30_2302_db_pad" + suffix
    return name


def _regular_readable(path: Path, label: str) -> os.stat_result:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not stat.S_IMODE(info.st_mode) & 0o444:
        raise ValueError(f"{label} is not a readable regular file: {path}")
    return info


def _contained(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} escapes its declared root: {path}") from exc
    return resolved


def _materialization(manifest: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(manifest.read_text())
    if (
        not isinstance(payload, dict)
        or set(payload) != MATERIALIZATION_TOP_KEYS
        or payload["schema_version"] != 1
        or payload["kind"] != "preprocessing-metadata-materialization"
        or payload["outcome"] != "success"
        or payload["archive_stat_unchanged"] is not True
        or payload["errors"] != []
    ):
        raise ValueError("materialization manifest has an unexpected schema")
    outputs = payload["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != RECOVERED:
        raise ValueError("materialization manifest must inventory recovered outputs")
    normalized: list[dict[str, Any]] = []
    for logical_name, item in outputs.items():
        if (
            not isinstance(item, dict)
            or set(item) != MATERIALIZATION_OUTPUT_KEYS
            or logical_name not in RECOVERED
            or item["logical_name"] != logical_name
            or item["file_name"] != logical_name
            or item["status"] != "complete"
            or item["size_matches_expected"] is not True
        ):
            raise ValueError("materialization output has an unexpected schema")
        if (
            not isinstance(item["size_bytes"], int)
            or item["size_bytes"] <= 0
            or not isinstance(item["declared_size_bytes"], int)
            or not isinstance(item["expected_size_bytes"], int)
            or not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        ):
            raise ValueError("materialization output values are invalid")
        if item["declared_size_bytes"] != item["size_bytes"] or item["expected_size_bytes"] != item["size_bytes"]:
            raise ValueError("materialization output size attestations disagree")
        normalized.append({"logical_name": logical_name, "size_bytes": item["size_bytes"], "sha256": item["sha256"]})
    return _digest(manifest), sorted(normalized, key=lambda item: item["logical_name"])


def _entry(logical: str, root: Path, root_kind: str, expected_recovered: dict[str, dict[str, Any]]) -> dict[str, Any]:
    physical_name = _physical(logical)
    physical = root / physical_name
    resolved = _contained(physical, root, root_kind)
    info = _regular_readable(resolved, root_kind)
    source_kind = "symlink" if physical.is_symlink() else "regular"
    sha = _digest(resolved) if root_kind == "recovered" or logical.endswith(".dbtype") else None
    if root_kind == "recovered":
        expected = expected_recovered[logical]
        if info.st_size != expected["size_bytes"] or sha != expected["sha256"]:
            raise ValueError(f"recovered materialization drift: {logical}")
    return {
        "logical_name": logical,
        "root_kind": root_kind,
        "physical_name": physical_name,
        "physical_source": str(physical),
        "resolved_source": str(resolved),
        "source_kind": source_kind,
        "resolved_kind": "regular",
        "resolved_mode": oct(stat.S_IMODE(info.st_mode)),
        "resolved_size_bytes": info.st_size,
        "resolved_mtime_ns": info.st_mtime_ns,
        "readable": True,
        "hash_policy": "sha256" if sha is not None else "stat",
        "sha256": sha,
    }


def canonical_identity(entries: list[dict[str, Any]]) -> str:
    content = [
        {
            key: entry[key]
            for key in (
                "logical_name",
                "root_kind",
                "physical_name",
                "resolved_mode",
                "resolved_size_bytes",
                "hash_policy",
                "sha256",
            )
        }
        for entry in sorted(entries, key=lambda item: item["logical_name"])
    ]
    for item, entry in zip(content, sorted(entries, key=lambda item: item["logical_name"]), strict=True):
        if entry["hash_policy"] == "stat":
            item["resolved_mtime_ns"] = entry["resolved_mtime_ns"]
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_manifest(
    source_root: Path, recovered_root: Path, materialization_manifest: Path, view: Path | None = None
) -> dict[str, Any]:
    context = execution_context()
    source_root, recovered_root = source_root.resolve(strict=True), recovered_root.resolve(strict=True)
    materialization_sha, outputs = _materialization(materialization_manifest)
    recovered = {item["logical_name"]: item for item in outputs}
    entries = [
        _entry(
            name,
            recovered_root if name in RECOVERED else source_root,
            "recovered" if name in RECOVERED else "source",
            recovered,
        )
        for name in NAMES
    ]
    if view is not None:
        view.mkdir(mode=0o700)
        for entry in entries:
            destination = view / entry["logical_name"]
            temporary = view / f".{entry['logical_name']}.new"
            os.symlink(entry["physical_source"], temporary)
            os.replace(temporary, destination)
        descriptor = os.open(view, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {
        "schema_version": 2,
        "logical_primary_name": "uniref30_2302_db",
        "physical_primary_name": "uniref30_2302_db_pad",
        "database_source": str(source_root),
        "recovered_root": str(recovered_root),
        "database_view": str(view) if view else None,
        "entries": entries,
        "materialization_manifest_sha256": materialization_sha,
        "materialization_outputs": outputs,
        "database_content_identity": canonical_identity(entries),
        "execution_context": context,
    }


def validate_manifest(
    payload: object,
    view: Path | None = None,
    expected_identity: str | None = None,
    materialization_manifest: Path | None = None,
) -> None:
    if not isinstance(payload, dict) or set(payload) != TOP_KEYS or payload.get("schema_version") != 2:
        raise ValueError("database view schema is invalid")
    _validate_execution_context(payload["execution_context"])
    if (
        payload["logical_primary_name"] != "uniref30_2302_db"
        or payload["physical_primary_name"] != "uniref30_2302_db_pad"
    ):
        raise ValueError("primary physical padding mapping is invalid")
    entries = payload["entries"]
    if (
        not isinstance(entries, list)
        or len(entries) != len(NAMES)
        or {item.get("logical_name") for item in entries if isinstance(item, dict)} != set(NAMES)
    ):
        raise ValueError("database view must contain the exact 40 logical names")
    if any(name.endswith(".idx") or ".idx." in name for name in NAMES):
        raise ValueError("idx-style logical names are forbidden")
    if (
        not isinstance(payload["database_source"], str)
        or not isinstance(payload["recovered_root"], str)
        or not isinstance(payload["database_view"], str)
    ):
        raise ValueError("database root/view paths must be strings")
    source_root, recovered_root = Path(payload["database_source"]), Path(payload["recovered_root"])
    if (
        not source_root.is_absolute()
        or not recovered_root.is_absolute()
        or not source_root.is_dir()
        or not recovered_root.is_dir()
    ):
        raise ValueError("database root paths must be readable absolute directories")
    if view is not None and (not view.is_dir() or Path(payload["database_view"]).resolve() != view.resolve()):
        raise ValueError("database_view manifest binding does not match --view")
    outputs = payload["materialization_outputs"]
    if (
        not isinstance(payload["materialization_manifest_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", payload["materialization_manifest_sha256"]) is None
        or not isinstance(outputs, list)
        or len(outputs) != len(RECOVERED)
        or {item.get("logical_name") for item in outputs if isinstance(item, dict)} != RECOVERED
        or any(not isinstance(item, dict) or set(item) != VIEW_MATERIALIZATION_OUTPUT_KEYS for item in outputs)
        or any(
            not isinstance(item["size_bytes"], int)
            or item["size_bytes"] <= 0
            or not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            for item in outputs
        )
    ):
        raise ValueError("recovered materialization outputs are invalid")
    if materialization_manifest is not None:
        manifest_sha, manifest_outputs = _materialization(materialization_manifest)
        if manifest_sha != payload["materialization_manifest_sha256"] or manifest_outputs != outputs:
            raise ValueError("materialization manifest hash or output inventory mismatch")
    recovered = {item["logical_name"]: item for item in outputs}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
            raise ValueError("database view entry schema is invalid")
        logical = entry["logical_name"]
        root = recovered_root if logical in RECOVERED else source_root
        root_kind = "recovered" if logical in RECOVERED else "source"
        if (
            entry["root_kind"] != root_kind
            or entry["physical_name"] != _physical(logical)
            or Path(entry["physical_source"]) != root / _physical(logical)
        ):
            raise ValueError(f"physical mapping drift: {logical}")
        resolved = _contained(Path(entry["physical_source"]), root, root_kind)
        info = _regular_readable(resolved, logical)
        expected_source_kind = "symlink" if Path(entry["physical_source"]).is_symlink() else "regular"
        expected_hash_policy = "sha256" if logical in RECOVERED or logical.endswith(".dbtype") else "stat"
        if (
            str(resolved) != entry["resolved_source"]
            or entry["resolved_kind"] != "regular"
            or entry["source_kind"] != expected_source_kind
            or entry["readable"] is not True
            or entry["resolved_mode"] != oct(stat.S_IMODE(info.st_mode))
            or entry["resolved_size_bytes"] != info.st_size
            or entry["resolved_mtime_ns"] != info.st_mtime_ns
        ):
            raise ValueError(f"database source drift: {logical}")
        sha = _digest(resolved) if expected_hash_policy == "sha256" else None
        if entry["hash_policy"] != expected_hash_policy or entry["sha256"] != sha:
            raise ValueError(f"database digest policy drift: {logical}")
        if logical in RECOVERED and (
            entry["hash_policy"] != "sha256"
            or info.st_size != recovered[logical]["size_bytes"]
            or sha != recovered[logical]["sha256"]
        ):
            raise ValueError(f"recovered materialization drift: {logical}")
        if view is not None:
            link = view / logical
            if not link.is_symlink() or os.readlink(link) != entry["physical_source"]:
                raise ValueError(f"database view link target drift: {logical}")
    identity = canonical_identity(entries)
    if identity != payload["database_content_identity"] or (
        expected_identity is not None and identity != expected_identity
    ):
        raise ValueError("database content identity mismatch")


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source-root", type=Path, required=True)
    b.add_argument("--recovered-root", type=Path, required=True)
    b.add_argument("--materialization-manifest", type=Path, required=True)
    b.add_argument("--view", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("validate")
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--view", type=Path)
    v.add_argument("--expected-identity")
    v.add_argument("--materialization-manifest", type=Path)
    a = p.parse_args()
    if a.command == "build":
        _atomic_json_write(
            a.output, build_manifest(a.source_root, a.recovered_root, a.materialization_manifest, a.view)
        )
    else:
        validate_manifest(json.loads(a.manifest.read_text()), a.view, a.expected_identity, a.materialization_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
