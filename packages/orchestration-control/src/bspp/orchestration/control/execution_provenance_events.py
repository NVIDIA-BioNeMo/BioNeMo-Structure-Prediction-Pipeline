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

"""Create-once expectation/result persistence for one governed attempt."""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.contract.submission_evidence import SubmissionExpectation, SubmissionResult
from bspp.orchestration.control.governed_submission import (
    FinalizedEvidenceIndex,
    finalize_evidence_index,
    stable_read_descendant,
    write_immutable_descendant,
)


class ImmutableExecutionProvenanceStore:
    """Persist one immutable expectation and result per tokenized attempt."""

    def __init__(self, evidence_dir: Path) -> None:
        self.evidence_dir = evidence_dir

    def write_expectation(self, expectation: SubmissionExpectation) -> Path:
        return self._write(expectation, "expectation.json", expectation.canonical_bytes())

    def write_result(self, result: SubmissionResult) -> Path:
        return self._write(result, "result.json", result.canonical_bytes())

    def finalize(self) -> FinalizedEvidenceIndex:
        return finalize_evidence_index(self.evidence_dir)

    def _write(self, record: SubmissionExpectation, name: str, data: bytes) -> Path:
        token = record.token
        relative = Path("submissions") / f"{token.step_index:04d}-{token.step_name}-attempt-{token.attempt:04d}" / name
        path = self.evidence_dir / relative
        try:
            return write_immutable_descendant(self.evidence_dir, relative, data)
        except FileExistsError:
            if stable_read_descendant(self.evidence_dir, relative) != data:
                raise ValueError(f"immutable governed evidence changed: {path}") from None
            return path


def render_runtime_result_epilogue(
    expectation_path: Path,
    result_path: Path,
    *,
    runtime_qualification_path: Path,
    runtime_ipsae_binary_path: Path,
    evidence_root: Path | None = None,
) -> str:
    """Render an isolated create-once runtime result writer for a governed payload."""
    root = evidence_root if evidence_root is not None else expectation_path.parent
    try:
        expectation_relative = expectation_path.relative_to(root)
        result_relative = result_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("runtime provenance paths must be descendants of the evidence root") from exc
    code = "\n".join(
        (
            "import hashlib, json, os, stat, time",
            f"evidence_root = {str(root)!r}",
            f"# expectation: {expectation_path}",
            f"# result: {result_path}",
            f"expectation_parts = {expectation_relative.parts!r}",
            f"result_parts = {result_relative.parts!r}",
            f"qualification_path = {str(runtime_qualification_path)!r}",
            f"binary_path = {str(runtime_ipsae_binary_path)!r}",
            "def open_parent(parts):",
            "    before = os.lstat(evidence_root)",
            "    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):",
            "        raise SystemExit('governed evidence root is not a real directory')",
            "    fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_NOFOLLOW', 0))",
            "    try:",
            "        opened = os.fstat(fd)",
            "        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):",
            "            raise SystemExit('governed evidence root changed during open')",
            "        for part in parts[:-1]:",
            "            expected = os.stat(part, dir_fd=fd, follow_symlinks=False)",
            "            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_NOFOLLOW', 0), dir_fd=fd)",
            "            actual = os.fstat(child)",
            "            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):",
            "                os.close(child); raise SystemExit('governed evidence directory changed during open')",
            "            os.close(fd); fd = child",
            "        return fd, parts[-1]",
            "    except BaseException:",
            "        os.close(fd); raise",
            "def verify_parent(parts, held_fd):",
            "    current, _name = open_parent(parts)",
            "    try:",
            "        held = os.fstat(held_fd); reached = os.fstat(current)",
            "        if (held.st_dev, held.st_ino) != (reached.st_dev, reached.st_ino):",
            "            raise SystemExit('governed evidence directory was replaced during operation')",
            "    finally: os.close(current)",
            "def stable_bytes_at(parent_fd, name, limit):",
            "    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)",
            "    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:",
            "        raise SystemExit('invalid governed evidence artifact')",
            "    fd = os.open(name, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0), dir_fd=parent_fd)",
            "    try:",
            "        opened = os.fstat(fd); data = b''",
            "        while len(data) <= limit:",
            "            chunk = os.read(fd, min(1048576, limit + 1 - len(data)))",
            "            if not chunk: break",
            "            data += chunk",
            "        finished = os.fstat(fd)",
            "    finally: os.close(fd)",
            "    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)",
            "    key = lambda value: (value.st_dev,value.st_ino,value.st_nlink,value.st_size,value.st_mtime_ns)",
            "    if len(data) > limit or key(before) != key(opened) or key(opened) != key(finished):",
            "        raise SystemExit('governed evidence artifact changed during read')",
            "    if key(finished) != key(current):",
            "        raise SystemExit('governed evidence artifact changed during read')",
            "    return data",
            "def stable_bytes(path, limit):",
            "    before = os.lstat(path)",
            "    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:",
            "        raise SystemExit('invalid governed runtime observation artifact')",
            "    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))",
            "    try:",
            "        opened = os.fstat(fd); data = b''",
            "        while len(data) <= limit:",
            "            chunk = os.read(fd, min(1048576, limit + 1 - len(data)))",
            "            if not chunk: break",
            "            data += chunk",
            "        finished = os.fstat(fd)",
            "    finally: os.close(fd)",
            "    current = os.lstat(path)",
            "    def key(value):",
            "        return (value.st_dev,value.st_ino,value.st_nlink,value.st_size,",
            "            value.st_mtime_ns,value.st_ctime_ns)",
            "    if len(data) > limit or key(before) != key(opened) or key(opened) != key(finished):",
            "        raise SystemExit('governed runtime observation changed during read')",
            "    if key(finished) != key(current):",
            "        raise SystemExit('governed runtime observation changed during read')",
            "    return data",
            "expectation_parent, expectation_name = open_parent(expectation_parts)",
            "try: data = stable_bytes_at(expectation_parent, expectation_name, 16777216)",
            "finally: os.close(expectation_parent)",
            "if hashlib.sha256(data).hexdigest() != os.environ['BSPP_EXPECTATION_SHA256']:",
            "    raise SystemExit('governed expectation digest mismatch')",
            "expectation = json.loads(data)",
            "if expectation['token']['token'] != os.environ['BSPP_SUBMISSION_TOKEN']:",
            "    raise SystemExit('governed submission token mismatch')",
            "status = int(os.environ['BSPP_PAYLOAD_STATUS'])",
            "qualification = json.loads(stable_bytes(qualification_path, 16777216))",
            "observed_revision = qualification['smoke_evidence']['runtime_ipsae']['source_revision']",
            "observed_binary_sha256 = hashlib.sha256(stable_bytes(binary_path, 1073741824)).hexdigest()",
            "result = dict(expectation)",
            "result.update(job_id=os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ['SLURM_JOB_ID'],",
            "    scheduler_status='COMPLETED' if status == 0 else 'FAILED',",
            "    runtime_observations=[",
            "        {'name':'runtime_ipsae_binary_sha256','value':observed_binary_sha256},",
            "        {'name':'runtime_ipsae_source_revision','value':observed_revision}])",
            "encoded = (json.dumps(result, sort_keys=True, separators=(',', ':')) + '\\n').encode()",
            "result_parent, result_name = open_parent(result_parts)",
            "lock_name = result_name + '.create-lock'",
            "for _ in range(6000):",
            "    try:",
            "        os.mkdir(lock_name, 0o700, dir_fd=result_parent)",
            "        break",
            "    except FileExistsError:",
            "        time.sleep(0.01)",
            "else: raise SystemExit('timed out waiting for provenance result publication')",
            "try:",
            "    try: current = os.stat(result_name, dir_fd=result_parent, follow_symlinks=False)",
            "    except FileNotFoundError: current = None",
            "    if current is not None:",
            "        fd = os.open(result_name, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0), dir_fd=result_parent)",
            "        try:",
            "            metadata = os.fstat(fd)",
            "            existing = os.read(fd, len(encoded) + 1)",
            "        finally: os.close(fd)",
            "        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or existing != encoded:",
            "            raise SystemExit('conflicting provenance result already exists')",
            "    else:",
            "        fd = os.open(result_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0),",
            "            0o600, dir_fd=result_parent)",
            "        try:",
            "            view = memoryview(encoded)",
            "            while view:",
            "                written = os.write(fd, view)",
            "                if written <= 0: raise OSError('stalled provenance result write')",
            "                view = view[written:]",
            "            os.fsync(fd)",
            "        finally: os.close(fd)",
            "finally:",
            "    os.rmdir(lock_name, dir_fd=result_parent)",
            "os.fsync(result_parent)",
            "verify_parent(result_parts, result_parent)",
            "os.close(result_parent)",
        )
    )
    return (
        "export BSPP_PAYLOAD_STATUS\n"
        "/usr/bin/python3 -I -S <<'BSPP_WRITE_EXECUTION_RESULT'\n" + code + "\nBSPP_WRITE_EXECUTION_RESULT"
    )


__all__ = ["ImmutableExecutionProvenanceStore", "render_runtime_result_epilogue"]
