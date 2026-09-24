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

"""Bounded streaming preflight for padded MMseqs databases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TAX = {
    "TaxID": re.compile(r"TaxID=(\d+)"),
    "UniRefTaxID": re.compile(r"\bn=\S+\s+Tax=(\d+)(?:\s|$)"),
    "OX": re.compile(r"\bOX=(\d+)"),
    "UniRefTaxName": re.compile(r"\bn=\S+\s+Tax=([^\r\n]+)"),
}
MAX_BITMAP_BYTES = 128 * 1024 * 1024
MAX_ACTIVE_BITMAP_BYTES = 256 * 1024 * 1024
MAX_RANGE_TO_RECORD_RATIO = 32
MAX_RESULT_ROWS = 1_000_000
MAX_WITNESSES = 16
TARGET_DATABASE_SIZE = 36_293_491
EXPECTED_MISSING_ALIGNMENT_COUNT = 33
CAPTURED_MISSING_ALIGNMENT_OCCURRENCE_COUNT = 66
_CANONICAL_UNSIGNED = re.compile(r"(?:0|[1-9][0-9]*)\Z")
EXECUTION_CONTEXT_KEYS = frozenset(
    {"context_kind", "slurm_job_id", "slurmd_nodename", "tool_sha256", "python_executable", "python_version"}
)


@dataclass(frozen=True)
class SourceIdentity:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


def _identity(info: os.stat_result) -> SourceIdentity:
    return SourceIdentity(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_source(path: Path, expected: SourceIdentity | None = None) -> tuple[int, SourceIdentity]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise ValueError(f"unreadable database file: {path}") from exc
    try:
        info = os.fstat(descriptor)
        observed = _identity(info)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"unreadable database file: {path}")
        if expected is not None and observed != expected:
            raise ValueError(f"source identity drift: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, observed


def _capture_identity(path: Path) -> SourceIdentity:
    descriptor, identity = _open_source(path)
    os.close(descriptor)
    return identity


def _revalidate_source(path: Path, descriptor: int, expected: SourceIdentity) -> None:
    if _identity(os.fstat(descriptor)) != expected:
        raise ValueError(f"source identity drift: {path}")
    verifying_descriptor, _ = _open_source(path, expected)
    os.close(verifying_descriptor)


def _lines(path: Path, expected: SourceIdentity | None = None) -> Iterator[tuple[int, str]]:
    descriptor, identity = _open_source(path, expected)
    try:
        handle = os.fdopen(descriptor, "r", encoding="utf-8", newline="")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        yield from enumerate(handle, 1)
    finally:
        try:
            _revalidate_source(path, handle.fileno(), identity)
        finally:
            handle.close()


def execution_context() -> dict[str, str | None]:
    if sys.version_info[:2] != (3, 12):
        raise ValueError(f"preflight requires pinned Python 3.12, observed {platform.python_version()}")
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


def keys(path: Path, identity: SourceIdentity | None = None) -> Iterator[int]:
    for number, line in _lines(path, identity):
        fields = line.split()
        if not fields:
            continue
        if _CANONICAL_UNSIGNED.fullmatch(fields[0]) is None:
            raise ValueError(f"noncanonical key {path}:{number}")
        yield int(fields[0])


def lookup_records(path: Path, identity: SourceIdentity | None = None) -> Iterator[tuple[int, str, int]]:
    """Parse exact makepaddedseqdb id/name/original-key TAB records."""
    for number, line in _lines(path, identity):
        row = line.removesuffix("\n").removesuffix("\r")
        if not row:
            continue
        fields = row.split("\t")
        if len(fields) != 3 or not fields[1]:
            raise ValueError(f"malformed lookup mapping {path}:{number}")
        if _CANONICAL_UNSIGNED.fullmatch(fields[0]) is None:
            raise ValueError(f"invalid lookup column 1 key {path}:{number}")
        if _CANONICAL_UNSIGNED.fullmatch(fields[2]) is None:
            raise ValueError(f"invalid lookup column 3 key {path}:{number}")
        yield int(fields[0]), fields[1], int(fields[2])


def stats(path: Path, identity: SourceIdentity) -> tuple[int, int, int]:
    it = keys(path, identity)
    try:
        first = next(it)
    except StopIteration as e:
        raise ValueError(f"empty index: {path}") from e
    lo = hi = first
    count = 1
    for k in it:
        lo = min(lo, k)
        hi = max(hi, k)
        count += 1
    return count, lo, hi


def lookup_stats(path: Path, identity: SourceIdentity) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    iterator = lookup_records(path, identity)
    try:
        first_column_1, _, first_column_3 = next(iterator)
    except StopIteration as exc:
        raise ValueError(f"empty lookup: {path}") from exc
    count = 1
    minimum_1 = maximum_1 = first_column_1
    minimum_3 = maximum_3 = first_column_3
    for column_1, _, column_3 in iterator:
        count += 1
        minimum_1, maximum_1 = min(minimum_1, column_1), max(maximum_1, column_1)
        minimum_3, maximum_3 = min(minimum_3, column_3), max(maximum_3, column_3)
    return (count, minimum_1, maximum_1), (count, minimum_3, maximum_3)


def guard_bitmap(summary: tuple[int, int, int], label: str) -> None:
    count, minimum, maximum = summary
    span = maximum - minimum + 1
    if (maximum + 8) // 8 > MAX_BITMAP_BYTES or span > count * MAX_RANGE_TO_RECORD_RATIO:
        raise ValueError(f"{label} range/cardinality density is unsafe for bounded bitmap allocation")


@dataclass
class BitmapBudget:
    active_bytes: int = 0
    peak_bytes: int = 0

    def allocate(self, maximum: int, label: str) -> bytearray:
        size = (maximum + 8) // 8
        if size > MAX_BITMAP_BYTES:
            raise ValueError(f"{label} exceeds per-namespace bitmap budget")
        if self.active_bytes + size > MAX_ACTIVE_BITMAP_BYTES:
            raise ValueError(f"aggregate live bitmap budget exceeded while allocating {label}")
        self.active_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.active_bytes)
        return bytearray(size)

    def release(self, bits: bytearray) -> None:
        size = len(bits)
        self.active_bytes -= size
        bits.clear()


@dataclass
class KeyBitmap:
    label: str
    summary: tuple[int, int, int]
    bits: bytearray
    count: int


def _set_unique(bits: bytearray, key: int, label: str) -> None:
    byte, bit = divmod(key, 8)
    flag = 1 << bit
    if bits[byte] & flag:
        raise ValueError(f"duplicate {label} key {key}")
    bits[byte] |= flag


def bitmap(
    path: Path,
    summary: tuple[int, int, int],
    label: str,
    budget: BitmapBudget,
    identity: SourceIdentity,
) -> KeyBitmap:
    maximum = summary[2]
    bits = budget.allocate(maximum, label)
    count = 0
    for k in keys(path, identity):
        if k > maximum:
            raise ValueError(f"{label} key changed after summary: {k}")
        _set_unique(bits, k, label)
        count += 1
    if count != summary[0]:
        raise ValueError(f"{label} cardinality changed after summary")
    return KeyBitmap(label, summary, bits, count)


def lookup_bitmaps(
    path: Path,
    column_1_summary: tuple[int, int, int],
    column_3_summary: tuple[int, int, int],
    budget: BitmapBudget,
    identity: SourceIdentity,
) -> tuple[KeyBitmap, KeyBitmap]:
    column_1_bits = budget.allocate(column_1_summary[2], "lookup column 1")
    column_3_bits = budget.allocate(column_3_summary[2], "lookup column 3")
    count = 0
    for column_1, _, column_3 in lookup_records(path, identity):
        if column_1 > column_1_summary[2]:
            raise ValueError(f"lookup column 1 key changed after summary: {column_1}")
        if column_3 > column_3_summary[2]:
            raise ValueError(f"lookup column 3 key changed after summary: {column_3}")
        _set_unique(column_1_bits, column_1, "lookup column 1")
        _set_unique(column_3_bits, column_3, "lookup column 3")
        count += 1
    if count != column_1_summary[0] or count != column_3_summary[0]:
        raise ValueError("lookup cardinality changed after summary")
    return (
        KeyBitmap("lookup_column_1", column_1_summary, column_1_bits, count),
        KeyBitmap("lookup_column_3", column_3_summary, column_3_bits, count),
    )


def _contains(bitmap: KeyBitmap, key: int) -> bool:
    byte, bit = divmod(key, 8)
    return byte < len(bitmap.bits) and bool(bitmap.bits[byte] & (1 << bit))


def _only_witnesses(source: KeyBitmap, other: KeyBitmap) -> list[int]:
    witnesses: list[int] = []
    for byte_number, source_byte in enumerate(source.bits):
        other_byte = other.bits[byte_number] if byte_number < len(other.bits) else 0
        missing = source_byte & ~other_byte
        while missing and len(witnesses) < MAX_WITNESSES:
            bit = (missing & -missing).bit_length() - 1
            witnesses.append(byte_number * 8 + bit)
            missing &= missing - 1
        if len(witnesses) >= MAX_WITNESSES:
            break
    return witnesses


def compare_bitmaps(left: KeyBitmap, right: KeyBitmap) -> dict[str, Any]:
    overlap = sum(
        (left_byte & (right.bits[number] if number < len(right.bits) else 0)).bit_count()
        for number, left_byte in enumerate(left.bits)
    )
    left_only, right_only = left.count - overlap, right.count - overlap
    return {
        "status": "exact" if left_only == 0 and right_only == 0 else "different",
        "left_count": left.count,
        "right_count": right.count,
        "overlap_count": overlap,
        "left_only_count": left_only,
        "right_only_count": right_only,
        "left_only_witnesses": _only_witnesses(left, right),
        "right_only_witnesses": _only_witnesses(right, left),
    }


def _unobserved(reason: str) -> dict[str, Any]:
    return {
        "outcome": "not_observed",
        "reason": reason,
        "total_target_count": None,
        "distinct_target_count": None,
        "coverage": None,
    }


def result_observation(path: Path | None, references: dict[str, KeyBitmap]) -> dict[str, Any]:
    if path is None:
        return _unobserved("input omitted")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return _unobserved("input unavailable")
    if not stat.S_ISREG(info.st_mode):
        return _unobserved("input is not a regular non-symlink file")
    seen: set[int] = set()
    total = 0
    coverage: dict[str, dict[str, Any]] = {
        name: {
            "occurrence_absent_count": 0,
            "distinct_absent_count": 0,
            "absent_witnesses": [],
        }
        for name in references
    }
    for key in keys(path):
        total += 1
        if total > MAX_RESULT_ROWS:
            raise ValueError(f"result target input exceeds bounded row cap {MAX_RESULT_ROWS}: {path}")
        fresh = key not in seen
        if fresh:
            seen.add(key)
        for name, reference in references.items():
            if _contains(reference, key):
                continue
            item: dict[str, Any] = coverage[name]
            item["occurrence_absent_count"] += 1
            if fresh:
                item["distinct_absent_count"] += 1
                witnesses: list[int] = item["absent_witnesses"]
                if len(witnesses) < MAX_WITNESSES:
                    witnesses.append(key)
    for item in coverage.values():
        occurrence_absent = int(item["occurrence_absent_count"])
        distinct_absent = int(item["distinct_absent_count"])
        item.update(
            {
                "occurrence_present_count": total - occurrence_absent,
                "distinct_present_count": len(seen) - distinct_absent,
                "occurrence_coverage": None if total == 0 else (total - occurrence_absent) / total,
                "distinct_coverage": None if not seen else (len(seen) - distinct_absent) / len(seen),
                "verdict": "confirmed" if distinct_absent == 0 else "refuted",
            }
        )
    return {
        "outcome": "observed",
        "reason": None,
        "total_target_count": total,
        "distinct_target_count": len(seen),
        "coverage": coverage,
    }


def encodings(index: Path, headers: Path, limit: int = 10000) -> dict[str, int]:
    result = {k: 0 for k in TAX}
    size = headers.stat().st_size
    with index.open() as idx, headers.open("rb") as data:
        for _, line in zip(range(limit), idx, strict=False):
            values = line.split()
            if len(values) < 3:
                raise ValueError("malformed _seq_h.index")
            _, offset, length = map(int, values[:3])
            if min(offset, length) < 0 or offset + length > size:
                raise ValueError("_seq_h index out of bounds")
            data.seek(offset)
            text = data.read(length).decode("utf-8", "strict")
            hits = {name: pattern.search(text) is not None for name, pattern in TAX.items()}
            numeric = any(hits[name] for name in ("TaxID", "UniRefTaxID", "OX"))
            for name in TAX:
                result[name] += int(hits[name] and (name != "UniRefTaxName" or not numeric))
    return result


def preflight(
    root: Path,
    *,
    search_target_keys: Path | None = None,
    expanded_target_keys: Path | None = None,
    expected_target_database_size: int = TARGET_DATABASE_SIZE,
    expected_missing_alignment_count: int = EXPECTED_MISSING_ALIGNMENT_COUNT,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    report: dict[str, Any] = {
        "schema_version": 2,
        "outcome": "failed",
        "failures": failures,
        "database_root": str(root),
        "indexes": {},
        "lookup": None,
        "pad_lookup_column_1_composition": None,
        "base_lookup_column_3_composition": None,
        "aln_primary_observations": None,
        "search_result_targets": _unobserved("validation did not reach search result targets"),
        "expanded_result_targets": _unobserved("validation did not reach expanded result targets"),
        "target_database_size": expected_target_database_size,
        "target_database_size_confirmed": None,
        "expected_missing_alignment_count": expected_missing_alignment_count,
        "captured_missing_alignment_occurrence_count": CAPTURED_MISSING_ALIGNMENT_OCCURRENCE_COUNT,
        "bitmap_budget": {
            "per_namespace_bytes": MAX_BITMAP_BYTES,
            "aggregate_live_bytes": MAX_ACTIVE_BITMAP_BYTES,
            "peak_live_bytes": 0,
        },
    }
    budget = BitmapBudget()
    try:
        report["execution_context"] = execution_context()
        if expected_target_database_size <= 0 or expected_missing_alignment_count < 0:
            raise ValueError("expected baseline counts must be non-negative and target size must be positive")
        root = root.resolve()
        report["database_root"] = str(root)
        paths = {name: root / f"uniref30_2302_db_{name}.index" for name in ("pad", "aln", "seq")}
        paths["lookup"] = root / "uniref30_2302_db_pad.lookup"
        paths["base"] = root / "uniref30_2302_db.index"
        identities = {name: _capture_identity(path) for name, path in paths.items()}

        summaries: dict[str, tuple[int, int, int]] = {}
        for name in ("pad", "base", "aln", "seq"):
            summaries[name] = stats(paths[name], identities[name])
            guard_bitmap(summaries[name], name)
            report["indexes"][name] = {
                "count": summaries[name][0],
                "minimum": summaries[name][1],
                "maximum": summaries[name][2],
            }
        lookup_column_1_summary, lookup_column_3_summary = lookup_stats(paths["lookup"], identities["lookup"])
        guard_bitmap(lookup_column_1_summary, "lookup column 1")
        guard_bitmap(lookup_column_3_summary, "lookup column 3")
        report["lookup"] = {
            "row_count": lookup_column_1_summary[0],
            "column_count": 3,
            "column_1": {
                "count": lookup_column_1_summary[0],
                "minimum": lookup_column_1_summary[1],
                "maximum": lookup_column_1_summary[2],
            },
            "column_2_nonempty_name_count": lookup_column_1_summary[0],
            "column_3": {
                "count": lookup_column_3_summary[0],
                "minimum": lookup_column_3_summary[1],
                "maximum": lookup_column_3_summary[2],
            },
        }

        pad = bitmap(paths["pad"], summaries["pad"], "pad", budget, identities["pad"])
        base = bitmap(paths["base"], summaries["base"], "base", budget, identities["base"])
        lookup_column_1, lookup_column_3 = lookup_bitmaps(
            paths["lookup"],
            lookup_column_1_summary,
            lookup_column_3_summary,
            budget,
            identities["lookup"],
        )
        pad_lookup = compare_bitmaps(pad, lookup_column_1)
        base_lookup = compare_bitmaps(base, lookup_column_3)
        report["pad_lookup_column_1_composition"] = pad_lookup
        report["base_lookup_column_3_composition"] = base_lookup
        if pad_lookup["status"] != "exact":
            failures.append(
                {"code": "pad_lookup_column_1_mismatch", "message": "pad index differs from lookup column 1"}
            )
        if base_lookup["status"] != "exact":
            failures.append(
                {"code": "base_lookup_column_3_mismatch", "message": "base index differs from lookup column 3"}
            )
        budget.release(lookup_column_1.bits)
        budget.release(lookup_column_3.bits)

        aln = bitmap(paths["aln"], summaries["aln"], "aln", budget, identities["aln"])
        report["aln_primary_observations"] = {
            "against_pad": compare_bitmaps(aln, pad),
            "against_base": compare_bitmaps(aln, base),
            "structural_gate": False,
        }
        search = result_observation(search_target_keys, {"pad": pad, "base": base, "aln": aln})
        if search["outcome"] == "observed":
            aln_coverage = search["coverage"]["aln"]
            search.update(
                {
                    "missing_alignment_occurrence_count": aln_coverage["occurrence_absent_count"],
                    "missing_alignment_distinct_count": aln_coverage["distinct_absent_count"],
                    "expected_missing_alignment_distinct_count": expected_missing_alignment_count,
                    "missing_alignment_count_confirmed": (
                        aln_coverage["distinct_absent_count"] == expected_missing_alignment_count
                    ),
                }
            )
        else:
            search.update(
                {
                    "missing_alignment_occurrence_count": None,
                    "missing_alignment_distinct_count": None,
                    "expected_missing_alignment_distinct_count": expected_missing_alignment_count,
                    "missing_alignment_count_confirmed": None,
                }
            )
        report["search_result_targets"] = search
        budget.release(aln.bits)
        budget.release(pad.bits)
        budget.release(base.bits)

        seq = bitmap(paths["seq"], summaries["seq"], "seq", budget, identities["seq"])
        report["expanded_result_targets"] = result_observation(expanded_target_keys, {"seq": seq})
        budget.release(seq.bits)

        header_data = root / "uniref30_2302_db_seq_h"
        header_index = root / "uniref30_2302_db_seq_h.index"
        headers = (
            encodings(header_index, header_data)
            if header_data.is_file() and header_index.is_file()
            else {name: 0 for name in TAX}
        )
        numeric_header_count = sum(headers[name] for name in ("TaxID", "UniRefTaxID", "OX"))
        report.update(
            {
                "target_database_size_confirmed": pad.count == expected_target_database_size,
                "header_encodings": headers,
                "header_numeric_encoding_count": numeric_header_count,
                "header_name_only_encoding_count": headers["UniRefTaxName"],
                "taxid_crosscheck": (
                    "required"
                    if numeric_header_count
                    else "not_applicable_name_only_organism_encoding"
                    if headers["UniRefTaxName"]
                    else "not_applicable"
                ),
            }
        )
    except (OSError, UnicodeError, ValueError) as exc:
        failures.append({"code": "structural_validation_error", "message": str(exc)})
    report["bitmap_budget"]["peak_live_bytes"] = budget.peak_bytes
    report["outcome"] = "success" if not failures else "failed"
    return report


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_report(path: Path, report: dict[str, Any]) -> None:
    data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    try:
        info = path.lstat()
    except FileNotFoundError:
        info = None
    if info is not None:
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o400 or path.read_bytes() != data:
            raise ValueError(f"stale immutable preflight report differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    published = False
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ValueError(f"zero-byte write while publishing preflight report: {path}")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            existing = path.lstat()
            if (
                not stat.S_ISREG(existing.st_mode)
                or stat.S_IMODE(existing.st_mode) != 0o400
                or path.read_bytes() != data
            ):
                raise ValueError(f"stale immutable preflight report differs: {path}") from None
        _fsync_directory(path.parent)
        published = True
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        if not published and path.exists():
            existing = path.lstat()
            if (
                not stat.S_ISREG(existing.st_mode)
                or stat.S_IMODE(existing.st_mode) != 0o400
                or path.read_bytes() != data
            ):
                raise ValueError(f"stale immutable preflight report differs: {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--database-root", type=Path, required=True)
    p.add_argument("--search-target-keys", type=Path)
    p.add_argument("--expanded-target-keys", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = preflight(
        a.database_root,
        search_target_keys=a.search_target_keys,
        expanded_target_keys=a.expanded_target_keys,
    )
    try:
        _publish_report(a.output, report)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
