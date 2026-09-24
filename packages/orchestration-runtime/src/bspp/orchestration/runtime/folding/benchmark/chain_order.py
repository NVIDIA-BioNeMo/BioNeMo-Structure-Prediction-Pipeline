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

"""Verify one ColabFold grouping against pinned original corpus chain identity.

Multiset matching is permitted while the actual ordered
MSA-derived target hash is retained. This consumer never rewrites either identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from bspp.orchestration.contract.folding_execution import folding_target_sequence_sha256
from bspp.orchestration.runtime.folding.execution.bioir_session import BIOIR_TOOL_USED

from .corpus import _recompute_dataset_fingerprint
from .index import CanonicalPairIndexEntry
from .suite import ValidationCase

_MAX_METADATA_BYTES = 64 * 1024 * 1024


def _snapshot(root: Path, name: str) -> bytes:
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"chain-order metadata escapes the corpus root: {name}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError(f"chain-order metadata must be a regular non-symlink file: {name}")
        data = handle.read(_MAX_METADATA_BYTES + 1)
    if len(data) > _MAX_METADATA_BYTES:
        raise ValueError(f"chain-order metadata exceeds the byte limit: {name}")
    return data


def _verified_records(root: Path, fingerprint: str) -> dict[str, Mapping[str, object]]:
    """Parse exactly the byte snapshots verified by the checksum inventory."""
    checksums: dict[str, str] = {}
    for line in _snapshot(root, "SHA256SUMS").decode("utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise ValueError("malformed chain-order checksum inventory")
        path = PurePosixPath(parts[1])
        if path.is_absolute() or ".." in path.parts or "\\" in parts[1]:
            raise ValueError("chain-order checksum path is not confined")
        name = str(path)
        if name in checksums:
            raise ValueError(f"duplicate chain-order checksum declaration: {name}")
        checksums[name] = parts[0]
    snapshots: dict[str, bytes] = {}
    for name in ("dataset.json", "targets.jsonl"):
        if name not in checksums:
            raise ValueError(f"chain-order metadata missing checksum: {name}")
        data = _snapshot(root, name)
        if hashlib.sha256(data).hexdigest() != checksums[name]:
            raise ValueError(f"chain-order metadata checksum mismatch: {name}")
        snapshots[name] = data
    dataset = json.loads(snapshots["dataset.json"])
    if not isinstance(dataset, dict):
        raise ValueError("chain-order dataset must be an object")
    records: dict[str, Mapping[str, object]] = {}
    for line in snapshots["targets.jsonl"].decode("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("target_id"), str) or not record["target_id"]:
            raise ValueError("chain-order target record has no target identity")
        identity = record["target_id"]
        if identity in records:
            raise ValueError(f"duplicate chain-order target record: {identity}")
        records[identity] = record
    recomputed = _recompute_dataset_fingerprint(dataset, [records[key] for key in sorted(records)])
    if dataset.get("dataset_fingerprint") != recomputed or recomputed != fingerprint:
        raise ValueError("chain-order corpus fingerprint mismatch")
    return records


def _strings(record: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = record.get(key)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"chain-order metadata requires nonempty {key}")
    return tuple(cast("list[str]", value))


@dataclass(frozen=True)
class VerifiedChainOrder:
    original_sha256: str
    observed_sha256: str
    grouped_to_original: tuple[int, ...]
    chain_map: tuple[tuple[str, str], ...]

    def to_summary(self) -> dict[str, object]:
        return {
            "match_mode": "verified-colabfold-grouping",
            "original_sequence_sha256": self.original_sha256,
            "observed_sequence_sha256": self.observed_sha256,
            "grouped_to_original": list(self.grouped_to_original),
            "effective_chain_map": dict(self.chain_map),
        }


class ColabFoldChainOrderResolver:
    """Lazy per-run verifier; exact identity matches need no corpus metadata."""

    def __init__(self, corpus_dir: Path, expected_fingerprint: str | None) -> None:
        self._root = corpus_dir
        self._fingerprint = expected_fingerprint
        self._records: dict[str, Mapping[str, object]] | None = None

    def resolve(self, case: ValidationCase, candidate: CanonicalPairIndexEntry) -> VerifiedChainOrder | None:
        if (
            self._fingerprint is None
            or candidate.target_id != case.target_id
            or candidate.sequence_sha256 == case.sequence_sha256
            or candidate.tool_used != BIOIR_TOOL_USED
            or case.expected_pair_mode != "unpaired_paired"
        ):
            return None
        if self._records is None:
            self._records = _verified_records(self._root, self._fingerprint)
        record = self._records.get(case.target_id)
        if record is None:
            return None
        chains = _strings(record, "chains")
        chain_ids = _strings(record, "chain_ids")
        if (
            len(chains) != len(chain_ids)
            or len(chains) > 26
            or len(set(chain_ids)) != len(chain_ids)
            or any(":" in chain or any(char.isspace() for char in chain) for chain in chains)
            or record.get("sequence") != ":".join(chains)
            or record.get("sequence_sha256") != case.sequence_sha256
            or folding_target_sequence_sha256(chains) != case.sequence_sha256
            or record.get("reference_path") != case.reference_structure
            or record.get("reference_sha256") != case.reference_sha256
        ):
            return None
        labels = tuple(chr(ord("A") + index) for index in range(len(chains)))
        if set(case.chain_map) != set(labels) or tuple(case.chain_map[label] for label in labels) != chain_ids:
            return None
        permutation = tuple(
            index for chain in dict.fromkeys(chains) for index, item in enumerate(chains) if item == chain
        )
        grouped = tuple(chains[index] for index in permutation)
        if grouped == chains or folding_target_sequence_sha256(grouped) != candidate.sequence_sha256:
            return None
        return VerifiedChainOrder(
            original_sha256=case.sequence_sha256,
            observed_sha256=candidate.sequence_sha256,
            grouped_to_original=permutation,
            chain_map=tuple(
                (label, case.chain_map[labels[index]]) for label, index in zip(labels, permutation, strict=True)
            ),
        )
