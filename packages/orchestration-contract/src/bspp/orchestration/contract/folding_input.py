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

"""Pure Phase-0 contracts for the MSA-to-folding seam.

This module is a schema-only surface: it performs no filesystem access, no
network access, and no A3M row splitting.  The folding phase consumes the
logical MSA Artifact Set ``bspp.msa-set/v1`` and later splits the merged A3M
bytes into per-chain A3Ms inside the folding runtime; this module only types
and validates the records that describe that seam.

Member stem grammar: a published A3M
member stem is either the monomer form ``AFDB_AF[-_]<16 digits>``, the
compound heterodimer form ``AFDB_AF[-_]<16 digits>_AF[-_]<16 digits>``, or a
PDB assembly form ``pdb_[a-z0-9]+_assembly_\\d+``.  That grammar is enforced
by the producer in ``preprocessing_execution._AFDB_MODEL_ID_STEM_RE`` and
``preprocessing_execution._PDB_ASSEMBLY_STEM_RE`` and is deliberately
documented and mirrored here rather than imported, preserving the contract
dependency boundary.

Consumption constraint: ``requires_paired_query_header`` is now
informational — the folding consumer accepts both ``True`` and ``False``,
supporting monomer/homomer/n-ary targets that do not require the paired-query
header ``#<L0>,<L1>\t1,1``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.model_identity import is_human_string_model_entity_id, normalize_model_entity_id
from bspp.orchestration.contract.preprocessing_handoff import MsaArtifactSetManifest
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

LayoutKind = Literal["openfold", "bioir"]
TemplateMode = Literal["none"]
OpenFoldPairingMode = Literal["species-from-colabfold-headers"]
BioIRPairingMode = Literal[
    "bioir-homomer-dummy",
    "method-c-unpaired",
    "unpaired-only-no-merged-source",
    "colabfold-merged-row-index",
]

_ARTIFACT_SET_ID = re.compile(r"sha256:[0-9a-f]{64}")
_UNPAIRED_A3M = re.compile(r"alignments/polymer_(\d{2})/unpaired\.a3m")
_PAIRED_A3M = re.compile(r"alignments/polymer_(\d{2})/paired\.a3m")
_MEMBER_STEM = re.compile(r"(?:AFDB_AF[-_]\d{16}(?:_AF[-_]\d{16})?|pdb_[a-z0-9]+_assembly_\d+)")

_OPENFOLD_LAYOUT_FIELDS = {
    "schema_version",
    "layout",
    "fasta_dir",
    "alignment_dir",
    "template_dir",
    "chain_ids",
    "template_mode",
    "pairing",
}
_BIOIR_LAYOUT_FIELDS = _OPENFOLD_LAYOUT_FIELDS | {
    "polymer_count",
    "paired_row_count",
    "use_paired_msa",
    "max_non_query_msa_rows",
    "bioir_request",
}

_A3M_SPLIT_HEADER_LINE = "first-non-blank"
_A3M_SPLIT_HEADER_PREFIX = "#"
_A3M_SPLIT_TUPLE_SEPARATOR = "\t"
_A3M_SPLIT_MINIMUM_ARITY = 1
_A3M_SPLIT_LENGTHS_CARDINALITIES_RULE = "positive-integers-equal-arity"
_A3M_SPLIT_LOWERCASE_RULE = "insertion-never-advances-aligned-position"
_A3M_SPLIT_GAP_CHARACTERS = (".", "-")
_A3M_SPLIT_QUERY_RECORD = "first-record"
_A3M_SPLIT_HOMOMER_MAPPING_RULE = "multiset-match-of-ungapped-query-pieces"
_A3M_SPLIT_CHAIN_INDEX_BASE = 1
_A3M_SPLIT_OUTPUT_PATTERN = "chain_{index}.a3m"


@dataclass(frozen=True)
class BioIRPolymer:
    """One BioIR polymer entry inside ``bioir-request.json``."""

    chain_ids: tuple[str, ...]
    sequence: str
    unpaired_msa: str
    paired_msa: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "BioIRPolymer")
        _validate_str_tuple(self.chain_ids, "BioIRPolymer chain_ids")
        if len(set(self.chain_ids)) != len(self.chain_ids):
            raise ValueError("BioIRPolymer chain_ids must be unique")
        _validate_non_empty_str(self.sequence, "BioIRPolymer sequence")
        unpaired_match = _UNPAIRED_A3M.fullmatch(self.unpaired_msa)
        if unpaired_match is None:
            raise ValueError("BioIRPolymer unpaired_msa must be alignments/polymer_NN/unpaired.a3m")
        if self.paired_msa is not None:
            paired_match = _PAIRED_A3M.fullmatch(self.paired_msa)
            if paired_match is None:
                raise ValueError("BioIRPolymer paired_msa must be alignments/polymer_NN/paired.a3m")
            if paired_match.group(1) != unpaired_match.group(1):
                raise ValueError("BioIRPolymer paired_msa must be the sibling of unpaired_msa")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "chain_ids": list(self.chain_ids),
            "sequence": self.sequence,
            "unpaired_msa": self.unpaired_msa,
            "paired_msa": self.paired_msa,
        }


@dataclass(frozen=True)
class BioIRRequestManifest:
    """The ``bioir-request.json`` manifest consumed by the BioIR folding backend."""

    input_id: str
    polymers: tuple[BioIRPolymer, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "BioIRRequestManifest")
        _validate_non_empty_str(self.input_id, "BioIRRequestManifest input_id")
        if not isinstance(self.polymers, tuple) or not self.polymers:
            raise ValueError("BioIRRequestManifest polymers must be a non-empty immutable tuple")
        sequences: list[str] = []
        chain_ids: list[str] = []
        for position, polymer in enumerate(self.polymers):
            if not isinstance(polymer, BioIRPolymer):
                raise ValueError("BioIRRequestManifest polymers must contain BioIRPolymer records")
            index = _polymer_position(polymer)
            if index != position:
                raise ValueError(
                    f"BioIRRequestManifest polymer path index {index:02d} must match tuple position {position}"
                )
            sequences.append(polymer.sequence)
            chain_ids.extend(polymer.chain_ids)
        if len(set(sequences)) != len(sequences):
            raise ValueError("BioIRRequestManifest polymer sequences must be unique")
        if len(set(chain_ids)) != len(chain_ids):
            raise ValueError("BioIRRequestManifest chain ids must be globally unique")

    @property
    def chain_ids(self) -> tuple[str, ...]:
        """Flattened chain ids in declared polymer order."""
        return tuple(chain_id for polymer in self.polymers for chain_id in polymer.chain_ids)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "input_id": self.input_id,
            "polymers": [polymer.to_mapping() for polymer in self.polymers],
        }


@dataclass(frozen=True)
class FoldingInputLayout:
    """One prepared folding-input layout discriminated by ``layout``.

    BioIR-only metadata is optional at the type level but required by
    discriminator validation, mirroring the harvested ``PreparedInput``
    separation between ``layout.json`` and ``bioir-request.json``.
    """

    layout: LayoutKind
    chain_ids: tuple[str, ...]
    template_mode: TemplateMode
    pairing: OpenFoldPairingMode | BioIRPairingMode
    fasta_dir: str = "fasta"
    alignment_dir: str = "alignments"
    template_dir: str = "templates"
    polymer_count: int | None = None
    paired_row_count: int | None = None
    use_paired_msa: bool | None = None
    max_non_query_msa_rows: int | None = None
    bioir_request: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "FoldingInputLayout")
        if self.layout not in {"openfold", "bioir"}:
            raise ValueError(f"unsupported folding-input layout: {self.layout!r}")
        if self.fasta_dir != "fasta" or self.alignment_dir != "alignments" or self.template_dir != "templates":
            raise ValueError("folding-input layout directories must be exactly fasta, alignments, templates")
        _validate_str_tuple(self.chain_ids, "FoldingInputLayout chain_ids")
        if len(set(self.chain_ids)) != len(self.chain_ids):
            raise ValueError("FoldingInputLayout chain_ids must be unique")
        if self.template_mode != "none":
            raise ValueError("folding-input layout template_mode must be 'none'")

        if self.layout == "openfold":
            if self.pairing != "species-from-colabfold-headers":
                raise ValueError("openfold folding-input layout pairing must be species-from-colabfold-headers")
            if any(
                value is not None
                for value in (
                    self.polymer_count,
                    self.paired_row_count,
                    self.use_paired_msa,
                    self.max_non_query_msa_rows,
                    self.bioir_request,
                )
            ):
                raise ValueError("openfold folding-input layout must not carry BioIR-only metadata")
            return

        if self.pairing not in {
            "bioir-homomer-dummy",
            "method-c-unpaired",
            "unpaired-only-no-merged-source",
            "colabfold-merged-row-index",
        }:
            raise ValueError(f"unsupported BioIR folding-input pairing: {self.pairing!r}")
        _validate_positive_int(self.polymer_count, "BioIR folding-input polymer_count")
        _validate_non_negative_int(self.paired_row_count, "BioIR folding-input paired_row_count")
        _validate_bool(self.use_paired_msa, "BioIR folding-input use_paired_msa")
        _validate_non_negative_int(self.max_non_query_msa_rows, "BioIR folding-input max_non_query_msa_rows")
        if self.bioir_request != "bioir-request.json":
            raise ValueError("BioIR folding-input bioir_request must be bioir-request.json")

    def validate_against_request(self, manifest: BioIRRequestManifest) -> None:
        """Cross-validate this BioIR layout against its request manifest."""
        if self.layout != "bioir":
            raise ValueError("validate_against_request is only legal for a BioIR folding-input layout")
        if self.polymer_count != len(manifest.polymers):
            raise ValueError("BioIR folding-input polymer_count must equal the request polymer count")
        if self.chain_ids != manifest.chain_ids:
            raise ValueError("BioIR folding-input chain_ids must equal the request flattened chain ids")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "layout": self.layout,
            "fasta_dir": self.fasta_dir,
            "alignment_dir": self.alignment_dir,
            "template_dir": self.template_dir,
            "chain_ids": list(self.chain_ids),
            "template_mode": self.template_mode,
            "pairing": self.pairing,
        }
        if self.layout == "bioir":
            result["polymer_count"] = self.polymer_count
            result["paired_row_count"] = self.paired_row_count
            result["use_paired_msa"] = self.use_paired_msa
            result["max_non_query_msa_rows"] = self.max_non_query_msa_rows
            result["bioir_request"] = self.bioir_request
        return result


@dataclass(frozen=True)
class MsaSetConsumption:
    """What the folding phase reads from one ``bspp.msa-set/v1`` Artifact Set."""

    artifact_set_id: str
    expected_chunk_count: int
    member_a3m_paths: tuple[str, ...]
    requires_paired_query_header: bool
    artifact_type: str = "bspp.msa-set/v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "MsaSetConsumption")
        if self.artifact_type != "bspp.msa-set/v1":
            raise ValueError("unsupported MSA set consumption artifact_type")
        if _ARTIFACT_SET_ID.fullmatch(self.artifact_set_id) is None:
            raise ValueError("MSA set consumption artifact_set_id must be sha256:<64 lowercase hex>")
        if not isinstance(self.expected_chunk_count, int) or isinstance(self.expected_chunk_count, bool):
            raise ValueError("MSA set consumption expected_chunk_count must be an integer")
        if self.expected_chunk_count != 1:
            raise ValueError("MSA set consumption expected_chunk_count must be exactly 1")
        if not isinstance(self.member_a3m_paths, tuple) or not self.member_a3m_paths:
            raise ValueError("MSA set consumption member_a3m_paths must be a non-empty immutable tuple")
        for path in self.member_a3m_paths:
            _validate_member_a3m_path(path, "MSA set consumption member_a3m_paths")
        if len(set(self.member_a3m_paths)) != len(self.member_a3m_paths):
            raise ValueError("MSA set consumption member_a3m_paths must be unique")
        _validate_bool(self.requires_paired_query_header, "MSA set consumption requires_paired_query_header")
        # requires_paired_query_header is now informational; the folding consumer
        # accepts both True and False.

    def validate_against_manifest(self, manifest: MsaArtifactSetManifest) -> None:
        """Return normally when this consumption matches the root manifest."""
        if manifest.artifact_type != "bspp.msa-set/v1":
            raise ValueError("MSA Artifact Set manifest artifact_type must be bspp.msa-set/v1")
        if manifest.artifact_set_id != self.artifact_set_id:
            raise ValueError("MSA set consumption artifact_set_id must match the manifest")
        if self.expected_chunk_count != len(manifest.chunks):
            raise ValueError("MSA set consumption expected_chunk_count must match the manifest chunk count")
        if len(self.member_a3m_paths) != manifest.member_count:
            raise ValueError("MSA set consumption member path count must match the manifest member_count")
        if sum(chunk.member_count for chunk in manifest.chunks) != len(self.member_a3m_paths):
            raise ValueError("manifest chunk member counts must sum to the consumer member path count")
        if manifest.member_lengths is not None and len(manifest.member_lengths) != len(self.member_a3m_paths):
            raise ValueError("MSA set consumption member path count must match the manifest member_lengths count")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "artifact_set_id": self.artifact_set_id,
            "expected_chunk_count": self.expected_chunk_count,
            "member_a3m_paths": list(self.member_a3m_paths),
            "requires_paired_query_header": self.requires_paired_query_header,
        }


@dataclass(frozen=True)
class A3mSplitRules:
    """Declarative record pinning the harvested general N-ary A3M split rule.

    Cardinality greater than one (homomers) is formalized from harvested
    behavior but is not yet live-proven by the current producer, which only
    emits the paired-query case ``#<L0>,<L1>\\t1,1``.
    """

    header_line: str = _A3M_SPLIT_HEADER_LINE
    header_prefix: str = _A3M_SPLIT_HEADER_PREFIX
    tuple_separator: str = _A3M_SPLIT_TUPLE_SEPARATOR
    minimum_arity: int = _A3M_SPLIT_MINIMUM_ARITY
    lengths_cardinalities_rule: str = _A3M_SPLIT_LENGTHS_CARDINALITIES_RULE
    lowercase_rule: str = _A3M_SPLIT_LOWERCASE_RULE
    gap_characters: tuple[str, ...] = _A3M_SPLIT_GAP_CHARACTERS
    query_record: str = _A3M_SPLIT_QUERY_RECORD
    homomer_mapping_rule: str = _A3M_SPLIT_HOMOMER_MAPPING_RULE
    chain_index_base: int = _A3M_SPLIT_CHAIN_INDEX_BASE
    output_pattern: str = _A3M_SPLIT_OUTPUT_PATTERN
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "A3mSplitRules")
        if self.header_line != _A3M_SPLIT_HEADER_LINE:
            raise ValueError("A3M split header_line must be 'first-non-blank'")
        if self.header_prefix != _A3M_SPLIT_HEADER_PREFIX:
            raise ValueError("A3M split header_prefix must be '#'")
        if self.tuple_separator != _A3M_SPLIT_TUPLE_SEPARATOR:
            raise ValueError("A3M split tuple_separator must be one tab")
        _validate_positive_int(self.minimum_arity, "A3M split minimum_arity")
        if self.minimum_arity != _A3M_SPLIT_MINIMUM_ARITY:
            raise ValueError("A3M split minimum_arity must be 1")
        if self.lengths_cardinalities_rule != _A3M_SPLIT_LENGTHS_CARDINALITIES_RULE:
            raise ValueError("A3M split lengths/cardinalities must be positive integers with equal arity")
        if self.lowercase_rule != _A3M_SPLIT_LOWERCASE_RULE:
            raise ValueError("A3M split lowercase rule must be insertion-never-advances-aligned-position")
        if not isinstance(self.gap_characters, tuple) or self.gap_characters != _A3M_SPLIT_GAP_CHARACTERS:
            raise ValueError("A3M split gap characters must be ('.', '-')")
        if self.query_record != _A3M_SPLIT_QUERY_RECORD:
            raise ValueError("A3M split query_record must be 'first-record'")
        if self.homomer_mapping_rule != _A3M_SPLIT_HOMOMER_MAPPING_RULE:
            raise ValueError("A3M split homomer mapping must be multiset-match-of-ungapped-query-pieces")
        _validate_positive_int(self.chain_index_base, "A3M split chain_index_base")
        if self.chain_index_base != _A3M_SPLIT_CHAIN_INDEX_BASE:
            raise ValueError("A3M split chain_index_base must be 1")
        if self.output_pattern != _A3M_SPLIT_OUTPUT_PATTERN:
            raise ValueError("A3M split output_pattern must be 'chain_{index}.a3m'")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "header_line": self.header_line,
            "header_prefix": self.header_prefix,
            "tuple_separator": self.tuple_separator,
            "minimum_arity": self.minimum_arity,
            "lengths_cardinalities_rule": self.lengths_cardinalities_rule,
            "lowercase_rule": self.lowercase_rule,
            "gap_characters": list(self.gap_characters),
            "query_record": self.query_record,
            "homomer_mapping_rule": self.homomer_mapping_rule,
            "chain_index_base": self.chain_index_base,
            "output_pattern": self.output_pattern,
        }


def parse_merged_a3m_header(line: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Parse one already-selected merged-A3M layout header line.

    Returns ``(lengths, cardinalities)`` or raises ``ValueError``.  This
    function performs no file scanning; the caller supplies the first
    non-blank line, whose selection rule is documented on :class:`A3mSplitRules`.
    """
    if not isinstance(line, str):
        raise ValueError("merged A3M layout header must be a string")
    text = line.strip()
    if not text:
        raise ValueError("merged A3M layout header must not be blank")
    if not text.startswith("#"):
        raise ValueError("merged A3M layout header must start with '#'")
    fields = text[1:].split("\t")
    if len(fields) != 2:
        raise ValueError("merged A3M layout header must contain exactly two tab-separated fields")
    # _parse_header_tuple raises on empty or malformed tuples, so both results are
    # guaranteed non-empty here; arity >= 1 needs no separate guard.
    lengths = _parse_header_tuple(fields[0], "lengths")
    cardinalities = _parse_header_tuple(fields[1], "cardinalities")
    if len(lengths) != len(cardinalities):
        raise ValueError("merged A3M layout header lengths and cardinalities must have equal arity")
    if any(value <= 0 for value in (*lengths, *cardinalities)):
        raise ValueError("merged A3M layout header values must be strictly positive")
    return lengths, cardinalities


def split_boundaries(lengths: tuple[int, ...]) -> tuple[int, ...]:
    """Return cumulative endpoints for one N-ary merged-A3M split."""
    if not isinstance(lengths, tuple) or not lengths:
        raise ValueError("split lengths must be a non-empty tuple")
    boundaries: list[int] = []
    total = 0
    for value in lengths:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("split lengths must be strictly positive integers")
        total += value
        boundaries.append(total)
    return tuple(boundaries)


def folding_input_layout_from_mapping(payload: Mapping[str, object]) -> FoldingInputLayout:
    layout = _str(payload, "layout")
    if layout not in {"openfold", "bioir"}:
        raise ValueError(f"unsupported folding-input layout: {layout!r}")
    if layout == "openfold":
        _strict(payload, _OPENFOLD_LAYOUT_FIELDS, _OPENFOLD_LAYOUT_FIELDS, "FoldingInputLayout")
        return FoldingInputLayout(
            schema_version=_schema(payload, "FoldingInputLayout"),
            layout=cast("LayoutKind", layout),
            fasta_dir=_str(payload, "fasta_dir"),
            alignment_dir=_str(payload, "alignment_dir"),
            template_dir=_str(payload, "template_dir"),
            chain_ids=_str_tuple(payload, "chain_ids"),
            template_mode=cast("TemplateMode", _str(payload, "template_mode")),
            pairing=cast("OpenFoldPairingMode", _str(payload, "pairing")),
        )
    _strict(payload, _BIOIR_LAYOUT_FIELDS, _BIOIR_LAYOUT_FIELDS, "FoldingInputLayout")
    return FoldingInputLayout(
        schema_version=_schema(payload, "FoldingInputLayout"),
        layout=cast("LayoutKind", layout),
        fasta_dir=_str(payload, "fasta_dir"),
        alignment_dir=_str(payload, "alignment_dir"),
        template_dir=_str(payload, "template_dir"),
        chain_ids=_str_tuple(payload, "chain_ids"),
        template_mode=cast("TemplateMode", _str(payload, "template_mode")),
        pairing=cast("BioIRPairingMode", _str(payload, "pairing")),
        polymer_count=_int(payload, "polymer_count"),
        paired_row_count=_int(payload, "paired_row_count"),
        use_paired_msa=_bool(payload, "use_paired_msa"),
        max_non_query_msa_rows=_int(payload, "max_non_query_msa_rows"),
        bioir_request=_str(payload, "bioir_request"),
    )


def bioir_polymer_from_mapping(payload: Mapping[str, object]) -> BioIRPolymer:
    _strict(
        payload,
        {"schema_version", "chain_ids", "sequence", "unpaired_msa", "paired_msa"},
        {"schema_version", "chain_ids", "sequence", "unpaired_msa"},
        "BioIRPolymer",
    )
    return BioIRPolymer(
        schema_version=_schema(payload, "BioIRPolymer"),
        chain_ids=_str_tuple(payload, "chain_ids"),
        sequence=_str(payload, "sequence"),
        unpaired_msa=_str(payload, "unpaired_msa"),
        paired_msa=_optional_str(payload, "paired_msa"),
    )


def bioir_request_manifest_from_mapping(payload: Mapping[str, object]) -> BioIRRequestManifest:
    _strict(
        payload,
        {"schema_version", "input_id", "polymers"},
        {"schema_version", "input_id", "polymers"},
        "BioIRRequestManifest",
    )
    return BioIRRequestManifest(
        schema_version=_schema(payload, "BioIRRequestManifest"),
        input_id=_str(payload, "input_id"),
        polymers=tuple(bioir_polymer_from_mapping(item) for item in _mappings(payload, "polymers")),
    )


def msa_set_consumption_from_mapping(payload: Mapping[str, object]) -> MsaSetConsumption:
    _strict(
        payload,
        {
            "schema_version",
            "artifact_type",
            "artifact_set_id",
            "expected_chunk_count",
            "member_a3m_paths",
            "requires_paired_query_header",
        },
        {
            "schema_version",
            "artifact_type",
            "artifact_set_id",
            "expected_chunk_count",
            "member_a3m_paths",
            "requires_paired_query_header",
        },
        "MsaSetConsumption",
    )
    return MsaSetConsumption(
        schema_version=_schema(payload, "MsaSetConsumption"),
        artifact_type=_str(payload, "artifact_type"),
        artifact_set_id=_str(payload, "artifact_set_id"),
        expected_chunk_count=_int(payload, "expected_chunk_count"),
        member_a3m_paths=_str_tuple(payload, "member_a3m_paths"),
        requires_paired_query_header=_bool(payload, "requires_paired_query_header"),
    )


def a3m_split_rules_from_mapping(payload: Mapping[str, object]) -> A3mSplitRules:
    _strict(
        payload,
        {
            "schema_version",
            "header_line",
            "header_prefix",
            "tuple_separator",
            "minimum_arity",
            "lengths_cardinalities_rule",
            "lowercase_rule",
            "gap_characters",
            "query_record",
            "homomer_mapping_rule",
            "chain_index_base",
            "output_pattern",
        },
        {
            "schema_version",
            "header_line",
            "header_prefix",
            "tuple_separator",
            "minimum_arity",
            "lengths_cardinalities_rule",
            "lowercase_rule",
            "gap_characters",
            "query_record",
            "homomer_mapping_rule",
            "chain_index_base",
            "output_pattern",
        },
        "A3mSplitRules",
    )
    return A3mSplitRules(
        schema_version=_schema(payload, "A3mSplitRules"),
        header_line=_str(payload, "header_line"),
        header_prefix=_str(payload, "header_prefix"),
        tuple_separator=_str(payload, "tuple_separator"),
        minimum_arity=_int(payload, "minimum_arity"),
        lengths_cardinalities_rule=_str(payload, "lengths_cardinalities_rule"),
        lowercase_rule=_str(payload, "lowercase_rule"),
        gap_characters=_str_tuple(payload, "gap_characters"),
        query_record=_str(payload, "query_record"),
        homomer_mapping_rule=_str(payload, "homomer_mapping_rule"),
        chain_index_base=_int(payload, "chain_index_base"),
        output_pattern=_str(payload, "output_pattern"),
    )


def _polymer_position(polymer: BioIRPolymer) -> int:
    match = _UNPAIRED_A3M.fullmatch(polymer.unpaired_msa)
    if match is None:
        raise ValueError("BioIRPolymer unpaired_msa must be alignments/polymer_NN/unpaired.a3m")
    return int(match.group(1))


def _parse_header_tuple(text: str, name: str) -> tuple[int, ...]:
    if not text:
        raise ValueError(f"merged A3M layout header {name} must not be empty")
    values: list[int] = []
    for part in text.split(","):
        if not part or any(not ("0" <= char <= "9") for char in part):
            raise ValueError(f"merged A3M layout header {name} must contain only decimal integers")
        values.append(int(part))
    return tuple(values)


def _validate_member_a3m_path(value: str, name: str) -> None:
    _validate_confined_relative_path(value, name)
    if not value.startswith("a3ms/") or not value.endswith(".a3m") or value.count("/") != 1:
        raise ValueError(f"{name} must be a stable a3ms/<member>.a3m path")
    member_name = value.removeprefix("a3ms/")
    if member_name in {".a3m", "..a3m"} or ".." in value.split("/"):
        raise ValueError(f"{name} must be a stable a3ms/<member>.a3m path")
    stem = member_name.removesuffix(".a3m")
    human_string_stem = is_human_string_model_entity_id(stem) and normalize_model_entity_id(stem) == stem
    if _MEMBER_STEM.fullmatch(stem) is None and not human_string_stem:
        raise ValueError(f"{name} member stem does not match the accepted AFDB_AF, PDB assembly or HumanSTRING grammar")


def _validate_confined_relative_path(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if value.startswith("/") or "\\" in value:
        raise ValueError(f"{name} must be a confined relative path")
    if any(component in {"", ".", ".."} for component in value.split("/")):
        raise ValueError(f"{name} must be a confined relative path")


def _validate_non_empty_str(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _validate_str_tuple(value: object, name: str) -> None:
    if not isinstance(value, tuple) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be an immutable tuple of non-empty strings")
    if not value:
        raise ValueError(f"{name} must not be empty")


def _validate_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _strict(payload: Mapping[str, object], allowed: set[str], required: set[str], name: str) -> None:
    if "schema_version" not in payload:
        raise ValueError(f"missing explicit schema_version at {name}")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing {name} field(s): {', '.join(missing)}")


def _schema(payload: Mapping[str, object], name: str) -> int:
    value = payload.get("schema_version")
    if value is None:
        raise ValueError(f"missing explicit schema_version at {name}")
    return validate_schema_version(value, record_name=name)


def _str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return cast("tuple[str, ...]", tuple(value))


def _mappings(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        raise ValueError(f"{key} must be a list")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must contain mappings")
    return cast("tuple[Mapping[str, object], ...]", tuple(value))


def _validate_schema(value: int, name: str) -> None:
    if validate_schema_version(value, record_name=name) != value:
        raise ValueError(f"{name} schema_version must be explicit")


__all__ = [
    "A3mSplitRules",
    "BioIRPairingMode",
    "BioIRPolymer",
    "BioIRRequestManifest",
    "FoldingInputLayout",
    "LayoutKind",
    "MsaSetConsumption",
    "OpenFoldPairingMode",
    "TemplateMode",
    "a3m_split_rules_from_mapping",
    "bioir_polymer_from_mapping",
    "bioir_request_manifest_from_mapping",
    "folding_input_layout_from_mapping",
    "msa_set_consumption_from_mapping",
    "parse_merged_a3m_header",
    "split_boundaries",
]
