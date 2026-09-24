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

"""BioIR input preprocessor for Track-A folding.

Port Baseline: frozen reference pipeline src/afdb_pipeline/bioir_backend.py
(``BioIRInputPreprocessor`` and its A3M helpers).  The preprocessor materializes
``fasta/``, ``alignments/polymer_NN/{unpaired,paired}.a3m``, ``bioir-request.json``,
and ``layout.json`` in the BioIR directory shape; it performs no BioIR/torch import.

``bioir-request.json`` and ``layout.json`` are strict contract artifacts built
from the frozen ``BioIRRequestManifest`` / ``FoldingInputLayout`` records.
Non-contract row-count diagnostics stay in ``PreparedInput.metadata`` only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from bspp.orchestration.contract.folding_input import (
    BioIRPairingMode,
    BioIRPolymer,
    BioIRRequestManifest,
    FoldingInputLayout,
)

from .a3m_split import _a3m_layout, _read_a3m, _split_a3m_sequence, _ungapped_a3m
from .errors import FoldingBackendError
from .models import PreparedInput, ProteinTarget, require_valid_target_identity
from .msa_models import MSAResult

__all__ = ["BioIRInputPreprocessor"]

_MAX_POLYMERS = 100


def _chain_id(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    # BioIR permits one to four alphanumeric characters.  The fallback keeps
    # the adapter general for complexes beyond the usual A-Z chain range.
    return f"A{index - 25}"


def _has_aligned_residue(sequence: str) -> bool:
    return any(character not in "-." and not character.islower() for character in sequence)


def _write_a3m(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def _copy_capped_a3m(
    source: Path,
    destination: Path,
    max_non_query_rows: int,
) -> tuple[int, int]:
    """Copy an A3M while retaining its query and a bounded number of hits."""

    lines = source.read_text(encoding="utf-8").splitlines()
    prefix: list[str] = []
    records: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith(">"):
            if current is not None:
                records.append(current)
            current = [line]
        elif current is None:
            prefix.append(line)
        else:
            current.append(line)
    if current is not None:
        records.append(current)
    if not records:
        raise FoldingBackendError(f"A3M contains no records: {source}")

    retained = records[: max_non_query_rows + 1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join([*prefix, *(line for record in retained for line in record)]) + "\n",
        encoding="utf-8",
    )
    return len(records), len(retained)


def _paired_records_from_merged_a3m(
    source: Path,
    unique_sequences: list[str],
) -> list[list[tuple[str, str]]] | None:
    """Split ColabFold paired rows while preserving cross-chain row indices.

    The regular per-chain split intentionally removes all-gap rows and is ideal
    for unpaired MSAs, but it destroys row synchronization.  BioIR pairs by row
    number, so paired records must be derived from the original merged A3M.
    """

    layout = _a3m_layout(source)
    records = _read_a3m(source)
    if layout is None or not records:
        return None
    lengths, _ = layout
    query_pieces = _split_a3m_sequence(records[0][1], lengths)
    query_sequences = [_ungapped_a3m(piece) for piece in query_pieces]
    try:
        source_indices = [query_sequences.index(sequence) for sequence in unique_sequences]
    except ValueError as exc:
        raise FoldingBackendError(f"merged A3M query layout does not match requested polymers: {source}") from exc

    paired: list[list[tuple[str, str]]] = [[] for _ in unique_sequences]
    for row_index, (header, sequence) in enumerate(records):
        pieces = _split_a3m_sequence(sequence, lengths)
        selected = [pieces[index] for index in source_indices]
        # Row zero is the query.  Subsequent rows are paired only when every
        # unique polymer has at least one aligned residue in that row.
        if row_index and not all(_has_aligned_residue(piece) for piece in selected):
            continue
        paired_header = "query" if row_index == 0 else f"pair_{row_index:06d} {header}"
        for polymer_index, piece in enumerate(selected):
            paired[polymer_index].append((paired_header, piece))
    return paired


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class BioIRInputPreprocessor:
    """Materialize BioIR request metadata plus synchronized A3M inputs."""

    name = "bioir-inputs"

    def __init__(
        self,
        *,
        use_paired_msa: bool = False,
        max_non_query_msa_rows: int = 5_000,
    ) -> None:
        if max_non_query_msa_rows < 0:
            raise ValueError("max_non_query_msa_rows must be non-negative")
        self.use_paired_msa = use_paired_msa
        self.max_non_query_msa_rows = max_non_query_msa_rows

    def run(self, target: ProteinTarget, msa: MSAResult, output_dir: Path) -> PreparedInput:
        # The target identity names the FASTA file, so it must satisfy the
        # model-ID grammar before any filesystem operation (including the
        # destructive re-creation of ``output_dir`` below).
        require_valid_target_identity(target)
        if len(msa.chains) != len(target.chains):
            raise FoldingBackendError("MSA result does not contain one alignment bundle per chain")
        if not target.chains:
            raise FoldingBackendError("BioIR input preprocessor requires a non-empty target")
        if not msa.chains:
            raise FoldingBackendError("BioIR input preprocessor requires a non-empty MSA result")

        chain_ids = [_chain_id(index) for index in range(len(target.chains))]
        unique_sequences: list[str] = []
        polymer_chain_ids: list[list[str]] = []
        first_chain_indices: list[int] = []
        for chain_index, (chain_id, sequence) in enumerate(zip(chain_ids, target.chains, strict=True)):
            if sequence in unique_sequences:
                polymer_chain_ids[unique_sequences.index(sequence)].append(chain_id)
            else:
                unique_sequences.append(sequence)
                polymer_chain_ids.append([chain_id])
                first_chain_indices.append(chain_index)

        if len(unique_sequences) > _MAX_POLYMERS:
            raise FoldingBackendError(
                f"BioIR input preprocessor supports at most {_MAX_POLYMERS} distinct polymers; "
                f"got {len(unique_sequences)}"
            )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        fasta_dir = output_dir / "fasta"
        alignment_dir = output_dir / "alignments"
        template_dir = output_dir / "templates"
        for path in (fasta_dir, alignment_dir, template_dir):
            path.mkdir(parents=True, exist_ok=True)

        polymers: list[BioIRPolymer] = []
        original_unpaired_rows = 0
        retained_unpaired_rows = 0
        for polymer_index, (sequence, ids, chain_index) in enumerate(
            zip(unique_sequences, polymer_chain_ids, first_chain_indices, strict=True)
        ):
            chain_msa = msa.chains[chain_index]
            source_alignment = next(iter(chain_msa.alignments.values()))
            polymer_dir = alignment_dir / f"polymer_{polymer_index:02d}"
            unpaired = polymer_dir / "unpaired.a3m"
            polymer_dir.mkdir(parents=True, exist_ok=True)
            original_rows, retained_rows = _copy_capped_a3m(
                source_alignment,
                unpaired,
                self.max_non_query_msa_rows,
            )
            original_unpaired_rows += original_rows
            retained_unpaired_rows += retained_rows
            try:
                polymer = BioIRPolymer(
                    chain_ids=tuple(ids),
                    sequence=sequence,
                    unpaired_msa=str(unpaired.relative_to(output_dir)),
                    paired_msa=None,
                )
            except ValueError as exc:
                raise FoldingBackendError(f"BioIR polymer construction failed: {exc}") from exc
            polymers.append(polymer)

        pairing_mode: BioIRPairingMode = "bioir-homomer-dummy"
        paired_row_count = 0
        if len(polymers) > 1 and not self.use_paired_msa:
            pairing_mode = "method-c-unpaired"
        elif len(polymers) > 1:
            merged_source_value = msa.metadata.get("merged_source_path")
            merged_source = (
                Path(merged_source_value) if isinstance(merged_source_value, str) and merged_source_value else None
            )
            paired_records = (
                _paired_records_from_merged_a3m(merged_source, unique_sequences)
                if merged_source is not None and merged_source.is_file()
                else None
            )
            if paired_records is None:
                pairing_mode = "unpaired-only-no-merged-source"
            else:
                counts = {len(records) for records in paired_records}
                if len(counts) != 1:
                    raise FoldingBackendError("BioIR paired A3Ms do not have identical row counts")
                paired_row_count = counts.pop()
                pairing_mode = "colabfold-merged-row-index"
                for polymer_index, records in enumerate(paired_records):
                    paired = alignment_dir / f"polymer_{polymer_index:02d}" / "paired.a3m"
                    _write_a3m(paired, records)
                    polymer = polymers[polymer_index]
                    try:
                        polymers[polymer_index] = BioIRPolymer(
                            chain_ids=polymer.chain_ids,
                            sequence=polymer.sequence,
                            unpaired_msa=polymer.unpaired_msa,
                            paired_msa=str(paired.relative_to(output_dir)),
                        )
                    except ValueError as exc:
                        raise FoldingBackendError(f"BioIR polymer construction failed: {exc}") from exc

        fasta_path = fasta_dir / f"{target.target_id}.fasta"
        fasta_path.write_text(
            "".join(f">{chain_id}\n{sequence}\n" for chain_id, sequence in zip(chain_ids, target.chains, strict=True)),
            encoding="utf-8",
        )

        try:
            manifest = BioIRRequestManifest(input_id=target.target_id, polymers=tuple(polymers))
        except ValueError as exc:
            raise FoldingBackendError(f"BioIR request manifest construction failed: {exc}") from exc
        manifest_path = output_dir / "bioir-request.json"
        _write_json(manifest_path, manifest.to_mapping())

        try:
            layout = FoldingInputLayout(
                layout="bioir",
                chain_ids=tuple(chain_ids),
                template_mode="none",
                pairing=pairing_mode,
                polymer_count=len(polymers),
                paired_row_count=paired_row_count,
                use_paired_msa=self.use_paired_msa,
                max_non_query_msa_rows=self.max_non_query_msa_rows,
                bioir_request="bioir-request.json",
            )
            layout.validate_against_request(manifest)
        except ValueError as exc:
            raise FoldingBackendError(f"BioIR folding-input layout construction failed: {exc}") from exc
        _write_json(output_dir / "layout.json", layout.to_mapping())

        metadata = {
            "chain_ids": chain_ids,
            "polymer_count": len(polymers),
            "template_mode": "none",
            "pairing": pairing_mode,
            "paired_row_count": paired_row_count,
            "use_paired_msa": self.use_paired_msa,
            "max_non_query_msa_rows": self.max_non_query_msa_rows,
            "original_unpaired_rows": original_unpaired_rows,
            "retained_unpaired_rows": retained_unpaired_rows,
            "bioir_request": str(manifest_path.relative_to(output_dir)),
        }
        return PreparedInput(self.name, fasta_dir, alignment_dir, template_dir, metadata)
