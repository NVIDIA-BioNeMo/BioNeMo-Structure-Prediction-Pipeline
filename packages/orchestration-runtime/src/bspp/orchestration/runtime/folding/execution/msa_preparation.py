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

"""Track-A composition seam from verified MSA projection to per-chain inputs.

This adapter is the only place that turns a verified ``project_msa_members`` /
``project_from_remote`` projection into the ``MSAResult`` consumed by the
OpenFold and BioIR input preprocessors.  It performs no submission, cluster
selection, or retry; it only binds the projected logical-path inventory to the
consumer declaration, selects the single member matching the target, splits the
merged A3M, and records the split in a 1-based ``ChainAlignment`` bundle
(per the documented contract).

The split-action boundary is real: ``msa_result_from_split_paths``
is the public constructor that validates already-split chain paths and builds
``MSAResult``, while ``prepare_projected_msa`` remains a behavior-compatible
wrapper that selects the merged member, calls ``split_merged_a3m`` exactly once,
then delegates to the constructor.  Only the split action calls
``split_merged_a3m``; a later preprocess stage reconstructs ``MSAResult`` from
its predecessor handoff through the constructor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.model_identity import normalize_model_entity_id

from .a3m_split import _read_a3m, split_merged_a3m
from .errors import FoldingBackendError
from .models import ProteinTarget, require_valid_target_identity
from .msa_models import ChainAlignment, MSAResult

__all__ = ["msa_result_from_split_paths", "prepare_projected_msa"]

_BACKEND_IDENTITY = "track-a-projected-msa"


def _member_stem_matches_target(logical_path: str, normalized_target: str) -> bool:
    """Return whether ``logical_path``'s stem normalizes to ``normalized_target``.

    A stem outside the contract model-ID grammar is a fail-closed
    :class:`FoldingBackendError` rather than a silent non-match, preserving the
    wrapper's historical error text for malformed member identities.
    """
    member_stem = Path(logical_path).stem
    try:
        return normalize_model_entity_id(member_stem) == normalized_target
    except ValueError as exc:
        raise FoldingBackendError(
            f"projected MSA member has an invalid model entity identity: {logical_path!r}"
        ) from exc


def msa_result_from_split_paths(
    split_paths: Sequence[Path],
    target: ProteinTarget,
    *,
    artifact_set_id: str,
    selected_logical_path: str,
    merged_source_path: Path,
) -> MSAResult:
    """Build ``MSAResult`` from already-split per-chain A3M paths.

    This is the split-action boundary: it validates the split output
    produced by ``split_merged_a3m`` — inventory, target-match, chain-count, and
    A3M row-count — plus the merged-source metadata, and binds one 1-based
    ``ChainAlignment`` per target chain.  ``artifact_set_id``,
    ``selected_logical_path``, and ``merged_source_path`` are the exact three
    keys carried in ``MSAResult.metadata``, so both the wrapper and a later
    executor reconstructing from a ``FoldingSplitHandoff`` can pass them
    through.
    """
    normalized_target = require_valid_target_identity(target)

    if not artifact_set_id:
        raise FoldingBackendError("MSA artifact set id must be a non-empty string")
    if not selected_logical_path:
        raise FoldingBackendError("selected MSA logical path must be a non-empty string")
    if not merged_source_path.is_file():
        raise FoldingBackendError(f"merged MSA source is not a regular file: {merged_source_path}")
    if not _member_stem_matches_target(selected_logical_path, normalized_target):
        raise FoldingBackendError(f"selected MSA member does not match target {target.target_id!r}")
    if len(split_paths) != len(target.chains):
        raise FoldingBackendError(
            f"merged A3M split produced {len(split_paths)} chains for {len(target.chains)} target chains"
        )

    chains: list[ChainAlignment] = []
    for index, (sequence, split_path) in enumerate(zip(target.chains, split_paths, strict=True), start=1):
        if not split_path.is_file():
            raise FoldingBackendError(f"split chain A3M is not a regular file: {split_path}")
        records = _read_a3m(split_path)
        if not records:
            raise FoldingBackendError(f"split chain A3M contains no records: {split_path}")
        chains.append(
            ChainAlignment(
                chain_index=index,
                query_sequence=sequence,
                alignments={"colabfold": split_path},
                sequence_counts={"colabfold": len(records)},
            )
        )

    return MSAResult(
        backend=_BACKEND_IDENTITY,
        chains=tuple(chains),
        metadata={
            "artifact_set_id": artifact_set_id,
            "selected_logical_path": selected_logical_path,
            "merged_source_path": str(merged_source_path),
        },
    )


def prepare_projected_msa(
    consumption: MsaSetConsumption,
    projected: Mapping[str, Path],
    target: ProteinTarget,
    output_dir: Path,
) -> MSAResult:
    """Compose a verified projection into the ``MSAResult`` the preprocessors consume.

    ``projected`` is the ``dict[str, Path]`` returned by ``project_msa_members``
    or ``project_from_remote``, keyed by logical path.  The adapter requires
    those keys to be exactly the consumer's declared member paths, selects the
    single member whose normalized stem equals the normalized target identity,
    splits the merged A3M exactly once, and delegates to
    ``msa_result_from_split_paths`` to bind the split into one 1-based
    ``ChainAlignment`` per target chain.
    """
    if set(projected) != set(consumption.member_a3m_paths):
        raise FoldingBackendError("projected MSA logical-path keys do not match the consumption member paths")

    normalized_target = require_valid_target_identity(target)

    matches: list[tuple[str, Path]] = []
    for logical_path in consumption.member_a3m_paths:
        if _member_stem_matches_target(logical_path, normalized_target):
            matches.append((logical_path, projected[logical_path]))

    if not matches:
        raise FoldingBackendError(
            f"no projected MSA member matches target {target.target_id!r} among {consumption.member_a3m_paths!r}"
        )
    if len(matches) > 1:
        raise FoldingBackendError(f"multiple projected MSA members match target {target.target_id!r}")

    selected_logical_path, selected_path = matches[0]
    if not selected_path.is_file():
        raise FoldingBackendError(f"selected MSA projection is not a regular file: {selected_path}")

    split_paths = split_merged_a3m(selected_path, target, output_dir)
    return msa_result_from_split_paths(
        split_paths,
        target,
        artifact_set_id=consumption.artifact_set_id,
        selected_logical_path=selected_logical_path,
        merged_source_path=selected_path,
    )
