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

"""Filesystem qualification for postprocessing directory publication."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import multiprocessing
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

from bspp.orchestration.runtime.postprocessing.finalization_io import _rename_directory_no_replace

_MAX_RESULT_BYTES = 4096
_PAYLOAD = b'{"publication":"compatibility-v1"}\n'


def _forced_einval(*_args: object) -> None:
    raise OSError(errno.EINVAL, os.strerror(errno.EINVAL))


def _worker(
    source: str,
    destination: str,
    barrier: Any,
    results: Any,
) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    try:
        barrier.wait(timeout=30)
        try:
            _rename_directory_no_replace(source_path, destination_path, renameat2=_forced_einval)
        except FileExistsError:
            observed = _read_exact_payload(destination_path / "payload.json")
            if observed != _PAYLOAD:
                raise ValueError("published compatibility winner differs") from None
            shutil.rmtree(source_path)
            results.put("idempotent-loser")
        else:
            results.put("winner")
    except BaseException as exc:
        results.put(f"error:{type(exc).__name__}:{exc}")


def _read_exact_payload(path: Path) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("compatibility payload is unsafe")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        document = b""
        while chunk := os.read(descriptor, 4096):
            document += chunk
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    if signature(before) != signature(opened) or signature(opened) != signature(after):
        raise ValueError("compatibility payload changed while reading")
    return document


def run_compatibility(*, root: Path, output: Path) -> dict[str, object]:
    """Exercise fallback publication with two independent publishers."""
    if not root.is_absolute() or not output.is_absolute():
        raise ValueError("compatibility paths must be absolute")
    if root.parent.resolve(strict=True) != root.parent or output.parent != root.parent:
        raise ValueError("compatibility paths must have one canonical parent")
    if root != root.parent / root.name or output != root.parent / f"{root.name}.json":
        raise ValueError("compatibility output must be the fixed sibling of its root")
    if root.exists() or os.path.lexists(root) or output.exists() or os.path.lexists(output):
        raise FileExistsError("compatibility attempt paths must be new")
    root.mkdir(mode=0o700)
    destination = root / "published"
    sources = (root / "stage-0", root / "stage-1")
    for source in sources:
        source.mkdir(mode=0o700)
        (source / "payload.json").write_bytes(_PAYLOAD)

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = tuple(
        context.Process(target=_worker, args=(str(source), str(destination), barrier, results)) for source in sources
    )
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        if process.exitcode != 0:
            raise RuntimeError(f"compatibility publisher exited {process.exitcode}")
    outcomes = sorted(results.get(timeout=5) for _ in processes)
    if outcomes != ["idempotent-loser", "winner"]:
        raise RuntimeError(f"compatibility publication outcomes differ: {outcomes!r}")
    if any(source.exists() or os.path.lexists(source) for source in sources):
        raise RuntimeError("compatibility staging residue remains")
    document = _read_exact_payload(destination / "payload.json")
    if document != _PAYLOAD:
        raise RuntimeError("compatibility published payload differs")
    evidence: dict[str, object] = {
        "schema_version": 1,
        "check": "postprocessing-directory-publication-v1",
        "status": "passed",
        "process_count": 2,
        "published_directory_count": 1,
        "fallback_errno": errno.EINVAL,
        "output_identity": {
            "path": "published/payload.json",
            "sha256": hashlib.sha256(document).hexdigest(),
            "size_bytes": len(document),
        },
    }
    encoded = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    if len(encoded) > _MAX_RESULT_BYTES:
        raise RuntimeError("compatibility result exceeds its fixed bound")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.rename(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return evidence


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run_compatibility(root=arguments.root, output=arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
