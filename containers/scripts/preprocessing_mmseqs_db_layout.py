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

"""Inspect, extract, and copy confined canonical MMseqs database layouts.

This tool models the shard discovery used by pinned MMseqs
8cc5ce367b5638c4306c2d7cfc652dd099a4643f: ``name.0``, ``name.1``, ...
are preferred over ``name``.  It deliberately rejects ambiguous directories
instead of silently accepting components MMseqs would ignore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

BUFFER_BYTES = 1024 * 1024
MAX_RESULT_ROWS = 1_000_000
MAX_INDEX_ENTRIES = 1_000_000
MAX_DATA_SHARDS = 10_000
MAX_DIRECTORY_ENTRIES = 10_000
MAX_DIRECTORY_NAME_BYTES = 255
MAX_LINK_TARGET_BYTES = 4096
MAX_INDEX_LINE_BYTES = BUFFER_BYTES
UINT64_MAX = (1 << 64) - 1
ALIGNMENT_RES = 5
PREFILTER_RES = 7
# MMseqs DBTYPE flags in the pinned source's Parameters.h and DBReader.h/.cpp.
# After shifting, DBReader's 0x7FFE mask excludes extended bit 0 (legacy
# compressed compatibility) and bit 15 (the overall bit-31 compression marker):
# https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/DBReader.h#L370-L376
COMPRESSED_FLAG = 0x80000000
DBTYPE_BASE_MASK = 0xFFFF
PADDED_EXTENDED_FLAG = 8
REQUIRED_COPY_DATABASES = ("qdb", "qdb_h", "prof_res", "prof_res_h", "res_exp_realign")
_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\Z")


class LayoutError(ValueError):
    """Raised when a supplied MMseqs layout is unsafe or ambiguous."""


@dataclass(frozen=True)
class Component:
    basename: str
    path: Path
    resolved: Path
    link_target: str | None
    size_bytes: int
    mode: int
    device: int
    inode: int
    mtime_ns: int
    sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "basename": self.basename,
            "path": str(self.path),
            "link_target": self.link_target,
            "resolved_path": str(self.resolved),
            "size_bytes": self.size_bytes,
            "mode": oct(self.mode),
            "device": self.device,
            "inode": self.inode,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
        }

    def identity(self) -> dict[str, Any]:
        return {"basename": self.basename, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class IndexEntry:
    key: int
    offset: int
    length: int


@dataclass(frozen=True)
class Layout:
    root: Path
    basename: str
    data: tuple[Component, ...]
    companions: tuple[Component, ...]
    dbtype_value: int
    dbtype_base: int
    compressed: bool
    extended_flags: int
    index: tuple[IndexEntry, ...]
    index_summary: dict[str, int | str]
    data_absent: bool

    def components(self) -> tuple[Component, ...]:
        return (*self.data, *self.companions)

    def identity(self) -> dict[str, Any]:
        return {
            "basename": self.basename,
            "components": [item.identity() for item in self.components()],
            "dbtype": {
                "value": self.dbtype_value,
                "base": self.dbtype_base,
                "compressed": self.compressed,
                "extended_flags": self.extended_flags,
                "padded": bool(self.extended_flags & PADDED_EXTENDED_FLAG),
            },
            "index": self.index_summary,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "basename": self.basename,
            "data_absent": self.data_absent,
            "components": [item.manifest() for item in self.components()],
            "dbtype": {
                "value": self.dbtype_value,
                "raw_hex": self.dbtype_value.to_bytes(4, "little").hex(),
                "base": self.dbtype_base,
                "compressed": self.compressed,
                "extended_flags": self.extended_flags,
                "padded": bool(self.extended_flags & PADDED_EXTENDED_FLAG),
            },
            "index_summary": self.index_summary,
            "missing_optional_companions": [
                suffix
                for suffix in (".lookup", ".source")
                if not any(item.basename == f"{self.basename}{suffix}" for item in self.companions)
            ],
            "logical_identity": self.identity(),
        }


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _validate_basename(value: str) -> str:
    if not _BASENAME.fullmatch(value) or value in {".", ".."} or "/" in value:
        raise LayoutError(f"invalid logical database basename: {value!r}")
    return value


def _root(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise LayoutError(f"database root is not a directory: {path}")
    return resolved


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_no_symlink_ancestors(path: Path) -> None:
    """Reject a publication path whose existing parents can redirect writes."""
    for candidate in (*reversed(path.parents), path):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise LayoutError(f"output parent is a symlink: {candidate}")
        if not stat.S_ISDIR(info.st_mode):
            raise LayoutError(f"output parent is not a directory: {candidate}")


def _prepare_output_path(path: Path, source_root: Path) -> Path:
    """Return a regular-file publication target outside the read-only source."""
    absolute = _absolute(path)
    _assert_no_symlink_ancestors(absolute.parent)
    try:
        info = absolute.lstat()
    except FileNotFoundError:
        info = None
    if info is not None and (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)):
        raise LayoutError(f"output is not a regular non-symlink file: {absolute}")
    resolved = absolute.resolve(strict=False)
    if _is_within(resolved, source_root):
        raise LayoutError(f"output must be outside source root: {absolute}")
    absolute.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _assert_no_symlink_ancestors(absolute.parent)
    return absolute


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_destination_root(path: Path, source_root: Path) -> Path:
    absolute = _absolute(path)
    _assert_no_symlink_ancestors(absolute.parent)
    try:
        info = absolute.lstat()
    except FileNotFoundError:
        info = None
    if info is not None and stat.S_ISLNK(info.st_mode):
        raise LayoutError(f"destination root is a symlink: {absolute}")
    if info is not None and not stat.S_ISDIR(info.st_mode):
        raise LayoutError(f"destination root is not a directory: {absolute}")
    prospective = absolute.resolve(strict=False)
    if _is_within(prospective, source_root) or _is_within(source_root, prospective):
        raise LayoutError("destination root must be distinct and non-overlapping with source root")
    absolute.mkdir(mode=0o700, parents=True, exist_ok=True)
    _assert_no_symlink_ancestors(absolute)
    if not stat.S_ISDIR(absolute.lstat().st_mode):
        raise LayoutError(f"destination root is not a directory: {absolute}")
    destination = absolute.resolve(strict=True)
    if _is_within(destination, source_root) or _is_within(source_root, destination):
        raise LayoutError("destination root must be distinct and non-overlapping with source root")
    return destination


def _within(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise LayoutError(f"component symlink escapes source root: {path}") from exc


def _hash_regular(path: Path) -> tuple[os.stat_result, str]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LayoutError(f"component is not a regular file: {path}")
        digest = hashlib.sha256()
        while block := os.read(descriptor, BUFFER_BYTES):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise LayoutError(f"component drifted while being hashed: {path}")
    return before, digest.hexdigest()


def _component_matches(component: Component, info: os.stat_result) -> bool:
    """Return whether an open descriptor still names the inspected component."""
    return stat.S_ISREG(info.st_mode) and (
        info.st_size,
        info.st_dev,
        info.st_ino,
        info.st_mtime_ns,
    ) == (
        component.size_bytes,
        component.device,
        component.inode,
        component.mtime_ns,
    )


def _revalidate_component_path(component: Component, *, hash_bytes: bool = False) -> None:
    """Ensure logical link provenance and target stat still agree; hash only when requested."""
    try:
        link_info = component.path.lstat()
    except FileNotFoundError as exc:
        raise LayoutError(f"component disappeared: {component.path}") from exc
    if component.link_target is None:
        if stat.S_ISLNK(link_info.st_mode) or component.path != component.resolved:
            raise LayoutError(f"component logical path changed: {component.path}")
        resolved = component.path
    else:
        if not stat.S_ISLNK(link_info.st_mode) or os.readlink(component.path) != component.link_target:
            raise LayoutError(f"component symlink changed: {component.path}")
        try:
            resolved = component.path.resolve(strict=True)
        except OSError as exc:
            raise LayoutError(f"component symlink became unreadable: {component.path}") from exc
        _within(resolved, component.path.parent)
        if stat.S_ISLNK(resolved.lstat().st_mode):
            raise LayoutError(f"component resolves through a symlink: {component.path}")
    if resolved != component.resolved:
        raise LayoutError(f"component target changed: {component.path}")
    info = resolved.lstat()
    if not _component_matches(component, info):
        raise LayoutError(f"component stat drifted: {component.path}")
    if hash_bytes:
        hashed_info, digest = _hash_regular(resolved)
        if not _component_matches(component, hashed_info) or digest != component.sha256:
            raise LayoutError(f"component bytes drifted: {component.path}")


def _open_inspected_component(component: Component) -> int:
    """Open an inspected component without following a final symlink."""
    _revalidate_component_path(component)
    descriptor = os.open(component.resolved, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        if not _component_matches(component, os.fstat(descriptor)):
            raise LayoutError(f"component drifted before use: {component.path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _finish_inspected_component(descriptor: int, component: Component, digest: Any) -> None:
    """Check an open component's identity and exact bytes after consumption."""
    try:
        if not _component_matches(component, os.fstat(descriptor)) or digest.hexdigest() != component.sha256:
            raise LayoutError(f"component drifted while being used: {component.path}")
        _revalidate_component_path(component)
    finally:
        os.close(descriptor)


def _component(root: Path, name: str, *, required: bool) -> Component | None:
    path = root / name
    try:
        link_info = path.lstat()
    except FileNotFoundError:
        if required:
            raise LayoutError(f"missing required component: {path}") from None
        return None
    link_target: str | None = None
    if stat.S_ISLNK(link_info.st_mode):
        link_target = os.readlink(path)
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise LayoutError(f"broken component symlink: {path}") from exc
        _within(resolved, root)
        if stat.S_ISLNK(resolved.lstat().st_mode):
            raise LayoutError(f"component resolves through a symlink: {path}")
    else:
        resolved = path
    info, digest = _hash_regular(resolved)
    return Component(
        basename=name,
        path=path,
        resolved=resolved,
        link_target=link_target,
        size_bytes=info.st_size,
        mode=stat.S_IMODE(info.st_mode),
        device=info.st_dev,
        inode=info.st_ino,
        mtime_ns=info.st_mtime_ns,
        sha256=digest,
    )


def _directory_inventory(root: Path) -> list[dict[str, Any]]:
    """Return a bounded lstat inventory without following directory entries."""
    values: list[dict[str, Any]] = []
    for path in root.iterdir():
        if len(values) >= MAX_DIRECTORY_ENTRIES:
            raise LayoutError("bounded directory inventory entry cap exceeded")
        if len(path.name.encode()) > MAX_DIRECTORY_NAME_BYTES:
            raise LayoutError(f"directory entry name exceeds bounded cap: {path.name!r}")
        info = path.lstat()
        kind = "symlink" if stat.S_ISLNK(info.st_mode) else "regular" if stat.S_ISREG(info.st_mode) else "other"
        link_target = os.readlink(path) if kind == "symlink" else None
        if link_target is not None and len(os.fsencode(link_target)) > MAX_LINK_TARGET_BYTES:
            raise LayoutError(f"directory entry link target exceeds bounded cap: {path.name!r}")
        values.append(
            {
                "name": path.name,
                "type": kind,
                "size_bytes": info.st_size,
                "mode": oct(stat.S_IMODE(info.st_mode)),
                "mtime_ns": info.st_mtime_ns,
                "device": info.st_dev,
                "inode": info.st_ino,
                "link_target": link_target,
            }
        )
    return sorted(values, key=lambda item: str(item["name"]))


def _shard_names(root: Path, basename: str) -> list[str]:
    numbers: list[int] = []
    for path in root.iterdir():
        prefix = f"{basename}."
        if not path.name.startswith(prefix):
            continue
        suffix = path.name.removeprefix(prefix)
        if suffix.isdigit():
            if not _DECIMAL.fullmatch(suffix):
                raise LayoutError(f"leading-zero numeric shard alias: {path.name}")
            number = int(suffix)
            if number >= MAX_DATA_SHARDS:
                raise LayoutError(f"numeric shard exceeds cap {MAX_DATA_SHARDS}: {path.name}")
            numbers.append(number)
    if not numbers:
        return []
    numbers.sort()
    if numbers[0] != 0 or any(number != previous + 1 for previous, number in pairwise(numbers)):
        raise LayoutError(f"non-contiguous numeric shards for {basename}")
    return [f"{basename}.{number}" for number in numbers]


def _parse_index(
    component: Component, *, stream_size: int | None
) -> tuple[tuple[IndexEntry, ...], dict[str, int | str]]:
    entries: list[IndexEntry] = []
    keys: set[int] = set()
    indexed_span = 0
    duplicate_count = 0
    descriptor = _open_inspected_component(component)
    digest = hashlib.sha256()
    canonical_digest = hashlib.sha256()
    carry = b""
    line_number = 0
    try:
        while block := os.read(descriptor, BUFFER_BYTES):
            digest.update(block)
            parts = (carry + block).split(b"\n")
            rows = parts[:-1]
            carry = parts[-1]
            if any(len(row) > MAX_INDEX_LINE_BYTES for row in (*rows, carry)):
                raise LayoutError(f"index row is not bounded text: {component.path}:{line_number + 1}")
            for raw in rows:
                line_number += 1
                # MMseqs writes whitespace-delimited index columns.  Accept the
                # tabs emitted by upstream as well as spaces used by fixtures.
                fields = raw.split()
                if len(fields) != 3 or any(
                    _DECIMAL.fullmatch(field.decode("ascii", "ignore")) is None for field in fields
                ):
                    raise LayoutError(f"malformed index row: {component.path}:{line_number}")
                try:
                    key, offset, length = (int(field) for field in fields)
                except ValueError as exc:
                    raise LayoutError(f"malformed index row: {component.path}:{line_number}") from exc
                if any(value > UINT64_MAX for value in (key, offset, length)):
                    raise LayoutError(f"index value exceeds uint64: {component.path}:{line_number}")
                if len(entries) >= MAX_INDEX_ENTRIES:
                    raise LayoutError(f"index entry cap {MAX_INDEX_ENTRIES} exceeded: {component.path}")
                if key in keys:
                    if stream_size is None:
                        duplicate_count += 1
                    else:
                        raise LayoutError(f"duplicate index key: {key}")
                keys.add(key)
                end = offset + length
                if end > UINT64_MAX:
                    raise LayoutError(f"index offset overflow: {component.path}:{line_number}")
                if stream_size is not None and end > stream_size:
                    raise LayoutError(f"index extent outside logical data stream: {component.path}:{line_number}")
                indexed_span = max(indexed_span, end)
                entries.append(IndexEntry(key, offset, length))
                canonical_digest.update(f"{key} {offset} {length}\n".encode())
        if carry:
            raise LayoutError(f"index row is not bounded newline-terminated text: {component.path}:{line_number + 1}")
    finally:
        _finish_inspected_component(descriptor, component, digest)
    if stream_size is not None:
        previous_end = 0
        sparse_gap_count = sparse_gap_bytes = 0
        for entry in sorted(entries, key=lambda item: (item.offset, item.length, item.key)):
            if entry.offset < previous_end:
                raise LayoutError(f"overlapping index extents: {component.path}")
            if entry.offset > previous_end:
                sparse_gap_count += 1
                sparse_gap_bytes += entry.offset - previous_end
            previous_end = entry.offset + entry.length
        if previous_end < stream_size:
            sparse_gap_count += 1
            sparse_gap_bytes += stream_size - previous_end
    else:
        sparse_gap_count = sparse_gap_bytes = 0
    return tuple(entries), {
        "row_count": len(entries),
        "key_count": len(keys),
        "duplicate_key_count": duplicate_count,
        "indexed_span_bytes": indexed_span,
        "sparse_gap_count": sparse_gap_count,
        "sparse_gap_bytes": sparse_gap_bytes,
        "source_sha256": digest.hexdigest(),
        "canonical_sha256": canonical_digest.hexdigest(),
    }


def inspect_layout(root_path: Path, basename: str) -> Layout:
    root, basename = _root(root_path), _validate_basename(basename)
    dbtype = _component(root, f"{basename}.dbtype", required=True)
    index_component = _component(root, f"{basename}.index", required=True)
    if dbtype is None or index_component is None:
        raise LayoutError(f"required dbtype/index component disappeared: {basename}")
    if dbtype.size_bytes != 4:
        raise LayoutError(f"dbtype must have exactly four bytes: {dbtype.path}")
    dbtype_descriptor = _open_inspected_component(dbtype)
    dbtype_digest = hashlib.sha256()
    try:
        dbtype_bytes = os.read(dbtype_descriptor, 5)
        dbtype_digest.update(dbtype_bytes)
        if len(dbtype_bytes) != 4:
            raise LayoutError(f"dbtype must have exactly four bytes: {dbtype.path}")
    finally:
        _finish_inspected_component(dbtype_descriptor, dbtype, dbtype_digest)
    dbtype_value = int.from_bytes(dbtype_bytes, "little")
    shards = _shard_names(root, basename)
    merged = _component(root, basename, required=False)
    if merged is not None and shards:
        raise LayoutError(f"ambiguous merged and numeric-sharded data: {basename}")
    data = (
        tuple(_component(root, name, required=True) for name in shards)
        if shards
        else (() if merged is None else (merged,))
    )
    if any(item is None for item in data):
        raise LayoutError(f"required data component disappeared: {basename}")
    data_components = tuple(item for item in data if item is not None)
    stream_size = sum(item.size_bytes for item in data_components) if data_components else None
    if stream_size is not None and stream_size > UINT64_MAX:
        raise LayoutError(f"logical data stream exceeds uint64: {basename}")
    entries, summary = _parse_index(index_component, stream_size=stream_size)
    if stream_size is not None:
        boundaries: set[int] = set()
        position = 0
        for item in data_components[:-1]:
            position += item.size_bytes
            boundaries.add(position)
        for entry in entries:
            if entry.length and any(entry.offset < boundary < entry.offset + entry.length for boundary in boundaries):
                raise LayoutError(f"index extent crosses a shard boundary: {entry.key}")
    optional = tuple(
        item
        for item in (
            _component(root, f"{basename}.lookup", required=False),
            _component(root, f"{basename}.source", required=False),
        )
        if item is not None
    )
    return Layout(
        root=root,
        basename=basename,
        data=data_components,
        companions=(dbtype, index_component, *optional),
        dbtype_value=dbtype_value,
        dbtype_base=dbtype_value & DBTYPE_BASE_MASK,
        compressed=bool(dbtype_value & COMPRESSED_FLAG),
        extended_flags=(dbtype_value >> 16) & 0x7FFE,
        index=entries,
        index_summary=summary,
        data_absent=not data_components,
    )


def _extent_chunks(layout: Layout, offset: int, length: int) -> Iterator[bytes]:
    if length == 0:
        return
    position = 0
    for component in layout.data:
        if position <= offset and offset + length <= position + component.size_bytes:
            descriptor = _open_inspected_component(component)
            try:
                os.lseek(descriptor, offset - position, os.SEEK_SET)
                remaining = length
                while remaining:
                    block = os.read(descriptor, min(BUFFER_BYTES, remaining))
                    if not block:
                        raise LayoutError("short read from validated index extent")
                    remaining -= len(block)
                    yield block
                return
            finally:
                if not _component_matches(component, os.fstat(descriptor)):
                    os.close(descriptor)
                    raise LayoutError(f"component drifted while streaming extent: {component.path}")
                os.close(descriptor)
        position += component.size_bytes
    raise LayoutError("validated index extent cannot be located")


def _verify_components(components: Sequence[Component]) -> None:
    """Revalidate logical names and exact bytes before immutable publication."""
    for component in components:
        _revalidate_component_path(component, hash_bytes=True)


def _immutable_write(path: Path, data: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o400)
    except FileExistsError:
        info, observed = _hash_regular(path)
        if (
            stat.S_IMODE(info.st_mode) != 0o400
            or observed != hashlib.sha256(data).hexdigest()
            or info.st_size != len(data)
        ):
            raise LayoutError(f"immutable output differs or has wrong mode: {path}") from None
        return
    published = False
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view[:BUFFER_BYTES])
            if written <= 0:
                raise LayoutError(f"zero-byte write while publishing immutable output: {path}")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        published = True
    finally:
        os.close(descriptor)
        if not published:
            path.unlink(missing_ok=True)
            _fsync_directory(path.parent)
    _fsync_directory(path.parent)


def _open_immutable_output(path: Path) -> int | None:
    try:
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o400)
    except FileExistsError:
        return None


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view[:BUFFER_BYTES])
        if written <= 0:
            raise LayoutError("zero-byte write while publishing result target keys")
        view = view[written:]


def _result_key(entry: IndexEntry, line: bytes) -> bytes | None:
    if not line.strip():
        return None
    field = line.split(None, 1)[0]
    try:
        rendered = field.decode("ascii")
    except UnicodeDecodeError as exc:
        raise LayoutError(f"result key is not canonical decimal: {entry.key}") from exc
    if _DECIMAL.fullmatch(rendered) is None:
        raise LayoutError(f"result key is not canonical decimal: {entry.key}")
    return field


def _write_result_key(descriptor: int | None, digest: Any, key: bytes) -> None:
    if descriptor is not None:
        _write_all(descriptor, key + b"\n")
    digest.update(key)
    digest.update(b"\n")


def _extract_result_record(
    layout: Layout, entry: IndexEntry, descriptor: int | None, digest: Any, remaining_rows: int
) -> tuple[int, int]:
    if entry.length == 0:
        return 0, 0
    carry = b""
    consumed = rows = keys = 0
    terminal = False
    for block in _extent_chunks(layout, entry.offset, entry.length):
        consumed += len(block)
        nul_count = block.count(b"\0")
        if nul_count:
            if terminal or nul_count != 1 or consumed != entry.length or not block.endswith(b"\0"):
                raise LayoutError(f"result record has invalid NUL framing: {entry.key}")
            terminal = True
            block = block[:-1]
        elif terminal:
            raise LayoutError(f"result record has invalid NUL framing: {entry.key}")
        parts = (carry + block).split(b"\n")
        lines = parts[:-1]
        carry = parts[-1]
        if any(len(line) > BUFFER_BYTES for line in (*lines, carry)):
            raise LayoutError(f"result record line exceeds bounded buffer: {entry.key}")
        for line in lines:
            field = _result_key(entry, line)
            if field is None:
                continue
            if rows >= remaining_rows:
                raise LayoutError(f"result rows exceed bounded cap {MAX_RESULT_ROWS}")
            rows += 1
            keys += 1
            _write_result_key(descriptor, digest, field)
    if not terminal:
        raise LayoutError(f"result record has invalid NUL framing: {entry.key}")
    if carry.strip():
        field = _result_key(entry, carry)
        if field is None:
            raise LayoutError(f"result record has no key: {entry.key}")
        if rows >= remaining_rows:
            raise LayoutError(f"result rows exceed bounded cap {MAX_RESULT_ROWS}")
        rows += 1
        keys += 1
        _write_result_key(descriptor, digest, field)
    return rows, keys


def directory_inventory(root: Path, manifest_path: Path) -> dict[str, Any]:
    source_root = _root(root)
    manifest_path = _prepare_output_path(manifest_path, source_root)
    entries = _directory_inventory(source_root)
    manifest = {
        "schema_version": 1,
        "kind": "mmseqs-directory-inventory",
        "outcome": "success",
        "source_root": str(source_root),
        "entry_count": len(entries),
        "entries": entries,
    }
    _immutable_write(manifest_path, _canonical_json(manifest))
    return manifest


def result_target_keys(root: Path, basename: str, manifest_path: Path, keys_path: Path) -> dict[str, Any]:
    layout = inspect_layout(root, basename)
    manifest_path = _prepare_output_path(manifest_path, layout.root)
    keys_path = _prepare_output_path(keys_path, layout.root)
    if manifest_path == keys_path:
        raise LayoutError("manifest and result-target-key outputs must be distinct")
    if (
        layout.dbtype_base not in {ALIGNMENT_RES, PREFILTER_RES}
        or layout.compressed
        or layout.extended_flags & PADDED_EXTENDED_FLAG
    ):
        raise LayoutError(
            "result-target-keys requires an uncompressed, unpadded ALIGNMENT_RES or PREFILTER_RES database"
        )
    if layout.data_absent:
        if keys_path.exists():
            raise LayoutError(f"data_absent output refuses stale result-target-key file: {keys_path}")
        _verify_components(layout.components())
        manifest = {
            "schema_version": 2,
            "kind": "mmseqs-result-target-keys",
            "outcome": "data_absent",
            "source_basename": layout.basename,
            "source": layout.manifest(),
            "target_keys": {
                "semantics": (
                    "first whitespace-delimited field of each nonempty result row; "
                    "a target key in the producer's target namespace, not the result database primary key"
                ),
                "result_record_count": len(layout.index),
                "target_key_occurrence_count": 0,
                "sha256": None,
            },
        }
        _immutable_write(manifest_path, _canonical_json(manifest))
        return manifest
    rows = keys = 0
    digest = hashlib.sha256()
    descriptor = _open_immutable_output(keys_path)
    keys_published = False
    try:
        for entry in sorted(layout.index, key=lambda item: item.offset):
            record_rows, record_keys = _extract_result_record(layout, entry, descriptor, digest, MAX_RESULT_ROWS - rows)
            rows += record_rows
            keys += record_keys
        _verify_components(layout.components())
        if descriptor is not None:
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            keys_published = True
    finally:
        if descriptor is not None:
            os.close(descriptor)
            if not keys_published:
                keys_path.unlink(missing_ok=True)
                _fsync_directory(keys_path.parent)
    if descriptor is not None:
        _fsync_directory(keys_path.parent)
    output_sha256 = digest.hexdigest()
    if descriptor is None:
        info, observed = _hash_regular(keys_path)
        if stat.S_IMODE(info.st_mode) != 0o400 or observed != output_sha256:
            raise LayoutError(f"immutable output differs or has wrong mode: {keys_path}")
    manifest = {
        "schema_version": 2,
        "kind": "mmseqs-result-target-keys",
        "outcome": "success",
        "source_basename": layout.basename,
        "source": layout.manifest(),
        "target_keys": {
            "semantics": (
                "first whitespace-delimited field of each nonempty result row; "
                "a target key in the producer's target namespace, not the result database primary key"
            ),
            "result_record_count": len(layout.index),
            "target_key_occurrence_count": keys,
            "sha256": output_sha256,
            "path": str(keys_path),
        },
    }
    _immutable_write(manifest_path, _canonical_json(manifest))
    return manifest


def _copy_component(component: Component, destination: Path) -> None:
    try:
        output = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        info, observed_sha256 = _hash_regular(destination)
        if (
            info.st_size != component.size_bytes
            or observed_sha256 != component.sha256
            or stat.S_IMODE(info.st_mode) != 0o400
        ):
            raise LayoutError(f"immutable copied component differs: {destination}") from None
        return
    source: int | None = None
    copied = False
    try:
        source = _open_inspected_component(component)
        before = os.fstat(source)
        if not stat.S_ISREG(before.st_mode) or (before.st_size, before.st_dev, before.st_ino, before.st_mtime_ns) != (
            component.size_bytes,
            component.device,
            component.inode,
            component.mtime_ns,
        ):
            raise LayoutError(f"source component drifted before copy: {component.path}")
        copy_digest = hashlib.sha256()
        while block := os.read(source, BUFFER_BYTES):
            copy_digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(output, view)
                if written <= 0:
                    raise LayoutError(f"zero-byte write while copying component: {destination}")
                view = view[written:]
        after = os.fstat(source)
        if (after.st_size, after.st_dev, after.st_ino, after.st_mtime_ns) != (
            component.size_bytes,
            component.device,
            component.inode,
            component.mtime_ns,
        ) or copy_digest.hexdigest() != component.sha256:
            raise LayoutError(f"source component drifted during copy: {component.path}")
        _revalidate_component_path(component)
        os.fchmod(output, 0o400)
        os.fsync(output)
        copied = True
    finally:
        if source is not None:
            os.close(source)
        os.close(output)
        if not copied:
            destination.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
    info, observed_sha256 = _hash_regular(destination)
    if info.st_size != component.size_bytes or observed_sha256 != component.sha256:
        raise LayoutError(f"copied component drifted: {destination}")
    _fsync_directory(destination.parent)


def copy_databases(
    source_root: Path, destination_root: Path, basenames: Sequence[str], manifest_path: Path
) -> dict[str, Any]:
    if len(basenames) != len(REQUIRED_COPY_DATABASES) or set(basenames) != set(REQUIRED_COPY_DATABASES):
        raise LayoutError(f"copy requires exactly: {', '.join(REQUIRED_COPY_DATABASES)}")
    source_directory = _root(source_root)
    destination = _prepare_destination_root(destination_root, source_directory)
    manifest_path = _prepare_output_path(manifest_path, source_directory)
    if _is_within(manifest_path.resolve(strict=False), destination):
        raise LayoutError("copy manifest must be outside destination root")
    source_layouts = [inspect_layout(source_directory, value) for value in REQUIRED_COPY_DATABASES]
    expected_names = {component.basename for layout in source_layouts for component in layout.components()}
    try:
        destination_names = {item.name for item in destination.iterdir()}
    except OSError as exc:
        raise LayoutError(f"cannot inspect destination root: {destination}") from exc
    manifest_exists = manifest_path.exists()
    if destination_names and not manifest_exists:
        raise LayoutError("destination is not fresh and has no immutable completion manifest")
    if manifest_exists and not destination_names:
        raise LayoutError("completion manifest exists but destination is empty")
    if destination_names and destination_names != expected_names:
        raise LayoutError("destination does not contain the exact replay component set")
    for name in destination_names:
        if not stat.S_ISREG((destination / name).lstat().st_mode):
            raise LayoutError(f"destination component is not a regular file: {destination / name}")
    for layout in source_layouts:
        if layout.data_absent:
            raise LayoutError(f"copy source has no data representation: {layout.basename}")
        if layout.basename == "qdb" and not any(item.basename == "qdb.lookup" for item in layout.companions):
            raise LayoutError("qdb.lookup is required for replay copy")
        for component in layout.components():
            _copy_component(component, destination / component.basename)
    _verify_components([component for layout in source_layouts for component in layout.components()])
    destination_layouts = [inspect_layout(destination, value) for value in REQUIRED_COPY_DATABASES]
    for source_layout, copied in zip(source_layouts, destination_layouts, strict=True):
        if source_layout.identity() != copied.identity():
            raise LayoutError(f"source/destination logical identity differs: {source_layout.basename}")
    manifest = {
        "schema_version": 1,
        "kind": "mmseqs-logical-copy",
        "outcome": "success",
        "source_root": str(source_directory),
        "destination_root": str(destination),
        "databases": [
            {
                "source": source_layout.manifest(),
                "destination": copied.manifest(),
                "logical_identity": source_layout.identity(),
            }
            for source_layout, copied in zip(source_layouts, destination_layouts, strict=True)
        ],
    }
    _immutable_write(manifest_path, _canonical_json(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    targets = commands.add_parser("result-target-keys")
    targets.add_argument("--source-root", type=Path, required=True)
    targets.add_argument("--basename", required=True)
    targets.add_argument("--manifest", type=Path, required=True)
    targets.add_argument("--keys-output", type=Path, required=True)
    inventory = commands.add_parser("directory-inventory")
    inventory.add_argument("--source-root", type=Path, required=True)
    inventory.add_argument("--manifest", type=Path, required=True)
    copy = commands.add_parser("copy")
    copy.add_argument("--source-root", type=Path, required=True)
    copy.add_argument("--destination-root", type=Path, required=True)
    copy.add_argument("--basename", action="append", required=True)
    copy.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "result-target-keys":
            result_target_keys(args.source_root, args.basename, args.manifest, args.keys_output)
        elif args.command == "directory-inventory":
            directory_inventory(args.source_root, args.manifest)
        else:
            copy_databases(args.source_root, args.destination_root, args.basename, args.manifest)
    except (OSError, LayoutError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
