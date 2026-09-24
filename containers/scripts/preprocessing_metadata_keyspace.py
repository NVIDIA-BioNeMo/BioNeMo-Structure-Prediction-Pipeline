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

"""Bounded keyspace and sampled organism audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

NUMERIC_HEADER_ENCODINGS = {
    "TaxID": re.compile(r"\bTaxID=(\d+)"),
    "UniRefTaxID": re.compile(r"\bn=\S+\s+Tax=(\d+)(?:\s|$)"),
    "OX": re.compile(r"\bOX=(\d+)"),
}
UNIREF_TAX_NAME = re.compile(r"\bn=\S+\s+Tax=([^\r\n]+)")
MAX_BITMAP_BYTES = 128 * 1024 * 1024
MAX_RANGE_TO_RECORD_RATIO = 32
EXECUTION_CONTEXT_KEYS = frozenset(
    {"context_kind", "slurm_job_id", "slurmd_nodename", "tool_sha256", "python_executable", "python_version"}
)


def execution_context() -> dict[str, str | None]:
    if sys.version_info[:2] != (3, 12):
        raise ValueError(f"keyspace audit requires pinned Python 3.12, observed {platform.python_version()}")
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


def records(path: Path) -> Iterator[tuple[int, int | None]]:
    with path.open() as h:
        for n, line in enumerate(h, 1):
            f = line.split()
            if not f:
                continue
            try:
                k = int(f[0])
            except ValueError as e:
                raise ValueError(f"invalid key {path}:{n}") from e
            if k < 0:
                raise ValueError("negative key")
            if len(f) > 1 and not f[1].isdigit():
                raise ValueError(f"non-numeric mapping taxid {path}:{n}")
            tax = int(f[1]) if len(f) > 1 else None
            yield k, tax


def index(path: Path) -> Iterator[tuple[int, int, int]]:
    with path.open() as h:
        for n, line in enumerate(h, 1):
            f = line.split()
            if len(f) < 3:
                raise ValueError(f"malformed index {path}:{n}")
            try:
                k, o, length = map(int, f[:3])
            except ValueError as e:
                raise ValueError(f"malformed index {path}:{n}") from e
            if min(k, o, length) < 0:
                raise ValueError("negative index value")
            yield k, o, length


def audit(
    mapping: Path,
    seq_index: Path,
    *,
    seq_h_index: Path | None = None,
    seq_h: Path | None = None,
    minimum_coverage: float = 0.99,
    sample_size: int = 10000,
    sample_seed: int = 0,
) -> dict[str, Any]:
    context = execution_context()
    for path in (
        mapping,
        seq_index,
        *(() if seq_h_index is None else (seq_h_index,)),
        *(() if seq_h is None else (seq_h,)),
    ):
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"unreadable audit input: {path}")
    if sample_size <= 0 or not 0 < minimum_coverage <= 1:
        raise ValueError("sample_size must be positive and minimum_coverage must be in (0, 1]")
    maximum = -1
    minimum: int | None = None
    seq_record_count = 0
    bits = bytearray()
    for k, _, _ in index(seq_index):
        required_bytes = (k + 8) // 8
        if required_bytes > MAX_BITMAP_BYTES:
            raise ValueError("_seq range/cardinality density is unsafe for bounded bitmap allocation")
        if required_bytes > len(bits):
            bits.extend(b"\0" * (required_bytes - len(bits)))
        byte, bit = divmod(k, 8)
        flag = 1 << bit
        if bits[byte] & flag:
            raise ValueError("duplicate _seq key")
        bits[byte] |= flag
        maximum = max(maximum, k)
        minimum = k if minimum is None else min(minimum, k)
        seq_record_count += 1
    if maximum < 0 or minimum is None:
        raise ValueError("empty _seq index")
    span = maximum - minimum + 1
    if (maximum + 8) // 8 > MAX_BITMAP_BYTES or span > seq_record_count * MAX_RANGE_TO_RECORD_RATIO:
        raise ValueError("_seq range/cardinality density is unsafe for bounded bitmap allocation")
    mapping_record_count = 0
    mapping_minimum: int | None = None
    mapping_maximum = -1
    seen = bytearray(len(bits))
    covered = 0
    mapping_outside_seq = False
    sample_tax: dict[int, int | None] = {}
    sample: dict[int, tuple[int, int]] = {}
    if seq_h_index is not None:
        generator = random.Random(sample_seed)
        header_seen = bytearray(len(bits))
        reservoir: list[tuple[int, int, int]] = []
        for position, (k, offset, length) in enumerate(index(seq_h_index)):
            if k > maximum:
                raise ValueError("_seq_h key outside _seq range")
            byte, bit = divmod(k, 8)
            if not bits[byte] & (1 << bit):
                raise ValueError("_seq_h key is absent from _seq")
            if header_seen[byte] & (1 << bit):
                raise ValueError("duplicate _seq_h key")
            header_seen[byte] |= 1 << bit
            if position < sample_size:
                reservoir.append((k, offset, length))
                continue
            replacement = generator.randrange(position + 1)
            if replacement < sample_size:
                reservoir[replacement] = (k, offset, length)
        sample = {key: (offset, length) for key, offset, length in reservoir}
    for key, tax in records(mapping):
        mapping_record_count += 1
        mapping_minimum = key if mapping_minimum is None else min(mapping_minimum, key)
        mapping_maximum = max(mapping_maximum, key)
        if key > maximum:
            mapping_outside_seq = True
            continue
        byte, bit = divmod(key, 8)
        flag = 1 << bit
        if seen[byte] & flag:
            raise ValueError("duplicate mapping key")
        seen[byte] |= flag
        covered += int(bool(bits[byte] & flag))
        if key in sample:
            sample_tax[key] = tax
    if not mapping_record_count or mapping_minimum is None:
        raise ValueError("empty mapping")
    mapping_span = mapping_maximum - mapping_minimum + 1
    if mapping_span > mapping_record_count * MAX_RANGE_TO_RECORD_RATIO:
        raise ValueError("mapping range/cardinality density is unsafe")
    if mapping_outside_seq:
        raise ValueError("mapping key outside _seq range")
    if covered / mapping_record_count < minimum_coverage:
        raise ValueError("mapping:_seq coverage below threshold")
    result = {
        "execution_context": context,
        "mapping_key_count": mapping_record_count,
        "mapping_record_count": mapping_record_count,
        "seq_key_count": seq_record_count,
        "mapping_key_range": [mapping_minimum, mapping_maximum],
        "mapping_key_span": mapping_span,
        "mapping_range_to_record_ratio": mapping_span / mapping_record_count,
        "seq_key_range": [minimum, maximum],
        "seq_key_span": span,
        "seq_range_to_record_ratio": span / seq_record_count,
        "covered_mapping_key_count": covered,
        "mapping_seq_coverage": covered / mapping_record_count,
        "mapping_to_seq_cardinality_ratio": mapping_record_count / seq_record_count,
        "minimum_coverage": minimum_coverage,
        "taxid_crosscheck": "not_applicable",
    }
    if seq_h_index is not None or seq_h is not None:
        if seq_h_index is None or seq_h is None:
            raise ValueError("both _seq_h inputs required")
        size = seq_h.stat().st_size
        encoded = numeric_encoded = name_only_encoded = comparable = matches = 0
        encoding_counts = {name: 0 for name in (*NUMERIC_HEADER_ENCODINGS, "UniRefTaxName")}
        context_counts = {
            "sampled": 0,
            "encoded": 0,
            "numeric_encoded": 0,
            "name_only_encoded": 0,
            "comparable": 0,
            "matching": 0,
        }
        with seq_h.open("rb") as h:
            for k, (o, length) in sample.items():
                if o + length > size:
                    raise ValueError("_seq_h index out of bounds")
                h.seek(o)
                context_counts["sampled"] += 1
                text = h.read(length).decode("utf-8", "strict")
                hits = [(name, pattern.search(text)) for name, pattern in NUMERIC_HEADER_ENCODINGS.items()]
                for name, hit in hits:
                    encoding_counts[name] += int(hit is not None)
                hit = next((value for _, value in hits if value is not None), None)
                if hit is not None:
                    encoded += 1
                    numeric_encoded += 1
                    context_counts["encoded"] += 1
                    context_counts["numeric_encoded"] += 1
                    if sample_tax.get(k) is not None:
                        comparable += 1
                        matches += int(int(hit.group(1)) == sample_tax[k])
                        context_counts["comparable"] += 1
                        context_counts["matching"] += int(int(hit.group(1)) == sample_tax[k])
                elif UNIREF_TAX_NAME.search(text):
                    encoded += 1
                    name_only_encoded += 1
                    encoding_counts["UniRefTaxName"] += 1
                    context_counts["encoded"] += 1
                    context_counts["name_only_encoded"] += 1
        result.update(
            {
                "header_sample_count": len(sample),
                "header_encoded_count": encoded,
                "header_numeric_encoded_count": numeric_encoded,
                "header_name_only_encoded_count": name_only_encoded,
                "header_comparable_count": comparable,
                "header_match_count": matches,
                "header_encoding_type_counts": encoding_counts,
                "header_encoding_context_counts": context_counts,
                "header_sample_seed": sample_seed,
            }
        )
        if comparable and matches != comparable:
            raise ValueError("mapping/header organism agreement failed")
        if comparable:
            result["taxid_crosscheck"] = "required"
        elif numeric_encoded:
            result["taxid_crosscheck"] = "not_applicable_no_numeric_mapping_taxid"
        elif name_only_encoded:
            result["taxid_crosscheck"] = "not_applicable_name_only_organism_encoding"
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--seq-index", type=Path, required=True)
    p.add_argument("--seq-h-index", type=Path)
    p.add_argument("--seq-h", type=Path)
    p.add_argument("--minimum-coverage", type=float, default=0.99)
    p.add_argument("--sample-size", type=int, default=10000)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(
        json.dumps(
            audit(
                a.mapping,
                a.seq_index,
                seq_h_index=a.seq_h_index,
                seq_h=a.seq_h,
                minimum_coverage=a.minimum_coverage,
                sample_size=a.sample_size,
                sample_seed=a.sample_seed,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
