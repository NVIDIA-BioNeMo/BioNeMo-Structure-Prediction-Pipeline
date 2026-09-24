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

"""Deterministic local A3M enumeration and folding indexing.

Port Baseline:
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/preprocessing/generate_batch_info.py:78-187``.
The port makes the baseline's unordered directory and worker-result traversal
deterministic, follows linked directories like its ``os.walk`` source, tracks
resolved directory identities to stop cycles, and fails instead of silently
dropping scan or parse failures.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from bspp.orchestration.contract.folding_index import FoldingIndex, FoldingIndexRecord, make_folding_index
from bspp.orchestration.runtime.folding.a3m import ParsedA3m, parse_a3m


def index_folding_a3ms(declared_inputs: Iterable[str | Path], *, sort_by_length: bool = False) -> FoldingIndex:
    """Index declared A3M files/directories in deterministic source order."""
    paths = _enumerate_declared_a3ms(declared_inputs)
    records = tuple(
        _record_from_parsed_a3m(parse_a3m(path), source_ordinal=source_ordinal)
        for source_ordinal, path in enumerate(paths)
    )
    return make_folding_index(records, sort_by_length=sort_by_length)


def _enumerate_declared_a3ms(declared_inputs: Iterable[str | Path]) -> tuple[Path, ...]:
    declarations = tuple(Path(value) for value in declared_inputs)
    if not declarations:
        msg = "At least one A3M file or directory must be declared"
        raise ValueError(msg)

    paths: list[Path] = []
    resolved_paths: set[Path] = set()
    for declaration in declarations:
        try:
            exists = declaration.exists()
            is_file = declaration.is_file()
            is_dir = declaration.is_dir()
        except OSError as exc:
            msg = f"Cannot inspect declared A3M input {declaration}: {exc}"
            raise ValueError(msg) from exc
        if not exists:
            msg = f"Declared A3M input does not exist: {declaration}"
            raise ValueError(msg)
        discovered: tuple[Path, ...]
        if is_file:
            if declaration.suffix != ".a3m":
                msg = f"Declared input is not an .a3m file: {declaration}"
                raise ValueError(msg)
            discovered = (declaration,)
        elif is_dir:
            discovered = _walk_a3m_files(declaration)
            if not discovered:
                msg = f"Declared directory contains no .a3m files: {declaration}"
                raise ValueError(msg)
        else:
            msg = f"Declared A3M input is not a readable file or directory: {declaration}"
            raise ValueError(msg)

        for path in discovered:
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                msg = f"Cannot resolve declared A3M input {path}: {exc}"
                raise ValueError(msg) from exc
            if resolved in resolved_paths:
                msg = f"A3M input was declared more than once: {path}"
                raise ValueError(msg)
            resolved_paths.add(resolved)
            paths.append(path)

    return tuple(paths)


def _walk_a3m_files(root: Path) -> tuple[Path, ...]:
    """Follow nested directory links once while preserving sorted walk order."""
    discovered: list[Path] = []
    pending = [root]
    visited_directories: set[Path] = set()
    while pending:
        directory = pending.pop()
        try:
            identity = directory.resolve(strict=True)
        except OSError as exc:
            msg = f"Cannot resolve declared A3M directory {directory}: {exc}"
            raise ValueError(msg) from exc
        if identity in visited_directories:
            continue
        visited_directories.add(identity)

        entries = _scandir_entries(directory)
        child_directories: list[Path] = []
        for entry in entries:
            try:
                is_directory = entry.is_dir(follow_symlinks=True)
                is_file = entry.is_file(follow_symlinks=True)
            except OSError as exc:
                msg = f"Cannot inspect entry {entry.path} while scanning declared A3M directory: {exc}"
                raise ValueError(msg) from exc
            path = Path(entry.path)
            if entry.name.endswith(".a3m"):
                if not is_file:
                    msg = f"Declared A3M entry {path} is not a readable file"
                    raise ValueError(msg)
                discovered.append(path)
            elif is_directory:
                child_directories.append(path)

        # The stack is LIFO, so reverse the sorted child list to visit it in
        # lexical order. Files at each level precede files in child folders,
        # matching os.walk's top-down shape while removing filesystem order.
        pending.extend(reversed(child_directories))
    return tuple(discovered)


def _scandir_entries(directory: Path) -> tuple[os.DirEntry[str], ...]:
    try:
        with os.scandir(directory) as iterator:
            return tuple(sorted(iterator, key=lambda entry: entry.name))
    except OSError as exc:
        msg = f"Cannot scan declared A3M directory {directory}: {exc}"
        raise ValueError(msg) from exc


def _record_from_parsed_a3m(parsed: ParsedA3m, *, source_ordinal: int) -> FoldingIndexRecord:
    return FoldingIndexRecord(
        source_ordinal=source_ordinal,
        protein_id=parsed.path.stem,
        msa_path=str(parsed.path),
        query_sequence=parsed.query_sequence,
        sequence_length=parsed.sequence_length,
        chain_lengths=parsed.chain_lengths,
        chain_count=parsed.chain_count,
        chain_cardinalities=parsed.chain_cardinalities,
        msa_depth=1,
        total_length=parsed.total_length,
    )


__all__ = ["index_folding_a3ms"]
