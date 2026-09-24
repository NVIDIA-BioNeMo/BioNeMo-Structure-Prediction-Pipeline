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

"""Immutable preprocessing execution-plan contracts.

Port Baseline:
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/msa_on_GPU.sh:48-61,
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/msa_batch.sh:12-52, and
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/massivemsa.env.sh:12-84.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pydantic import StrictBool, StrictInt, StrictStr, field_validator, model_validator

from bspp.orchestration.contract.config_models import FrozenConfigModel
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

# Mirrors the postprocessing discovery grammar; deliberately not imported —
# the contract package must not depend on the runtime package.
_AFDB_MODEL_ID_STEM_RE = re.compile(r"AFDB_AF[_-]\d{16}(?:_AF[_-]\d{16})?")
# PDB assembly model-ID stem (lowercase-only, matching the folding curator).
# Grounded in mmsa: extends the owned stem gate to accept PDB assembly
# identities, a new identity family absent from the mmsa baseline
# (scripts/msa_batch.sh:23-29 never validates member-name stems).
_PDB_ASSEMBLY_STEM_RE = re.compile(r"pdb_[a-z0-9]+_assembly_\d+")

# Scoped scientific schema version for the two scientific models only. The global
# CURRENT_CONTRACT_SCHEMA_VERSION stays 1 because it is embedded in ~11 unrelated
# digest preimages (submission-id, scheduler-token, runtime-contract-id); a global
# bump would break sealed evidence. v1 = old records, gates ABSENT
# (use_env=False / require_afdb_model_id_stem=False); v2 introduced authoring
# defaults False/True with the stem gate ON. v3 preserves paired rows by
# disabling independent per-chain paired filtering; v1/v2 retain filter=2.
CURRENT_PREPROCESSING_SCIENTIFIC_SCHEMA_VERSION: Literal[3] = 3
SUPPORTED_PREPROCESSING_SCIENTIFIC_SCHEMA_VERSIONS = frozenset({1, 2, 3})


def _validate_scientific_schema_version(value: object, *, record_name: str) -> int:
    """Return a supported scoped scientific schema version or fail closed."""
    if value is None:
        return CURRENT_PREPROCESSING_SCIENTIFIC_SCHEMA_VERSION
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value not in SUPPORTED_PREPROCESSING_SCIENTIFIC_SCHEMA_VERSIONS
    ):
        supported = ", ".join(str(version) for version in sorted(SUPPORTED_PREPROCESSING_SCIENTIFIC_SCHEMA_VERSIONS))
        msg = f"Unsupported {record_name} schema_version {value!r}; supported versions: {supported}"
        raise ValueError(msg)
    return value


def validate_scientific_schema_version(value: object, *, record_name: str) -> int:
    """Validate a serialized scientific sub-mapping's scoped schema version.

    The recursive fail-closed walkers (phase.py, phase_state.py, phase_retry.py)
    use this helper for the ``scientific`` sub-mapping so its scoped {1,2,3} version
    space is checked with the correct ruler; every other nested record keeps the
    global {1} rule.
    """
    return _validate_scientific_schema_version(value, record_name=record_name)


def _colabfold_filter_mode(scientific_schema_version: int) -> str:
    """Preserve historical vectors; v3 keeps paired row identities aligned."""
    version = _validate_scientific_schema_version(
        scientific_schema_version, record_name="PreprocessingScientificConfig"
    )
    return "1" if version == 3 else "2"


def afdb_model_id_member_name_conforms(member_name: str) -> bool:
    """Return whether an .a3m member name carries a discoverable AFDB model ID stem."""
    stem = member_name.removesuffix(".a3m")
    return _AFDB_MODEL_ID_STEM_RE.fullmatch(stem) is not None


def member_name_conforms(member_name: str) -> bool:
    """Return whether an .a3m member name carries a discoverable AFDB or PDB assembly model ID stem.

    Extends the owned stem gate to accept both AFDB model-ID
    stems and PDB assembly stems (``pdb_[a-z0-9]+_assembly_\\d+``).  Grounded in
    mmsa: mmsa (``scripts/msa_batch.sh:23-29``) never validates member-name
    stems; the Phase's stem gate was an owned hardening, and this extends it
    to a new identity family absent from the mmsa baseline.
    """
    stem = member_name.removesuffix(".a3m")
    return _AFDB_MODEL_ID_STEM_RE.fullmatch(stem) is not None or _PDB_ASSEMBLY_STEM_RE.fullmatch(stem) is not None


def expected_member_name(record_identity: str) -> str:
    """Return the adapter-derived member name for a record identity (first FASTA token).

        Replaces ``safe_filename(source_header.strip()[1:]) + '.a3m'`` with
        ``safe_filename(record_identity) + '.a3m'`` (first FASTA token only, no
        description).  Grounded in mmsa: mmsa's ``reshape.py`` emits only the first
        whitespace-delimited header token (ledger ``P46-INPUT-002``); the Phase's
        full-header derivation was an owned seam, and this refines it
        to align with mmsa's first-token-only behavior for the search input path
    .
    """
    return safe_filename(record_identity) + ".a3m"


def is_pdb_assembly_member_name(member_name: str) -> bool:
    """Return whether an .a3m member name carries a PDB assembly model ID stem."""
    stem = member_name.removesuffix(".a3m")
    return _PDB_ASSEMBLY_STEM_RE.fullmatch(stem) is not None


def safe_filename(file: str) -> str:
    """Replicate ColabFold v1.6.2 ``safe_filename`` byte-for-byte.

    ColabFold ``colabfold/input.py`` defines
    ``safe_filename(file) = "".join(c if c.isalnum() or c in ["_", ".", "-"] else "_" for c in file)``
    and ``colabfold/mmseqs/search.py`` renames the produced A3M to
    ``f"{safe_filename(raw_jobname)}.a3m"``. Unicode-aware ``str.isalnum()`` is
    preserved verbatim (no regex "improvement"). This is the compile-time
    derivation of the qualified adapter named by
    ``PREPROCESSING_ADAPTER_VERSION`` in ``contract/preprocessing_identity.py``
    (``preprocessing-scientific-backend-v3``); a second engine supplies its own
    derivation and this function is the documented extension point.
    """
    return "".join(c if c.isalnum() or c in ["_", ".", "-"] else "_" for c in file)


class PreprocessingScientificConfig(FrozenConfigModel):
    """User-authored scientific database names and optional gpuserver limit."""

    schema_version: Literal[1, 2, 3] = CURRENT_PREPROCESSING_SCIENTIFIC_SCHEMA_VERSION
    primary_database_name: StrictStr = "uniref30_2302_db"
    metagenomic_database_name: StrictStr = "colabfold_envdb_202108_db"
    max_sequences: StrictInt | None = 10_000
    use_env: StrictBool = False
    require_afdb_model_id_stem: StrictBool = True

    @model_validator(mode="before")
    @classmethod
    def _validate_schema(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        schema_version = data.get("schema_version")
        if schema_version is None:
            # Absent: the field default (v3) applies for fresh authoring.
            return data
        schema_version = _validate_scientific_schema_version(
            schema_version, record_name="PreprocessingScientificConfig"
        )
        if schema_version == 1:
            use_env = data.get("use_env")
            require_afdb_model_id_stem = data.get("require_afdb_model_id_stem")
            if use_env is None and require_afdb_model_id_stem is None:
                # v1 gate-absent semantics: inject the two knobs as False/False.
                data["use_env"] = False
                data["require_afdb_model_id_stem"] = False
            elif use_env is False and require_afdb_model_id_stem is False:
                # The contract's own intent->config propagation passes values explicitly.
                pass
            else:
                raise ValueError("v2 scientific fields in a schema_version 1 record")
        data["schema_version"] = schema_version
        return data

    @field_validator("primary_database_name", "metagenomic_database_name")
    @classmethod
    def _validate_database_name(cls, value: str) -> str:
        if not value or "/" in value:
            msg = "database names must be non-empty path components"
            raise ValueError(msg)
        return value

    @field_validator("max_sequences")
    @classmethod
    def _validate_max_sequences(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            msg = "max_sequences must be positive or null"
            raise ValueError(msg)
        return value


class PreprocessingSiteConfig(FrozenConfigModel):
    """Explicit site paths, executables, container, and scheduling choices."""

    schema_version: Literal[1] = CURRENT_CONTRACT_SCHEMA_VERSION
    mmseqs_executable: StrictStr
    colabfold_search_executable: StrictStr
    tar_executable: StrictStr
    lz4_executable: StrictStr
    database_root: StrictStr
    input_root: StrictStr
    scratch_output_root: StrictStr
    project_logs_root: StrictStr
    finished_msa_root: StrictStr
    split_input_root: StrictStr
    finished_input_root: StrictStr
    container_image: StrictStr
    container_mounts: tuple[StrictStr, ...]
    max_concurrency: StrictInt
    gpu_delay_seconds: StrictInt

    @model_validator(mode="before")
    @classmethod
    def _validate_schema(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        data["schema_version"] = validate_schema_version(
            data.get("schema_version"), record_name="PreprocessingSiteConfig"
        )
        return data

    @field_validator(
        "mmseqs_executable",
        "colabfold_search_executable",
        "tar_executable",
        "lz4_executable",
        "database_root",
        "input_root",
        "scratch_output_root",
        "project_logs_root",
        "finished_msa_root",
        "split_input_root",
        "finished_input_root",
        "container_image",
    )
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        if not value:
            msg = "site paths and executables must be non-empty"
            raise ValueError(msg)
        return value

    @field_validator("container_mounts")
    @classmethod
    def _validate_mounts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not mount for mount in value):
            msg = "container mounts must be non-empty strings"
            raise ValueError(msg)
        return value

    @field_validator("max_concurrency")
    @classmethod
    def _validate_concurrency(cls, value: int) -> int:
        if value <= 0:
            msg = "max_concurrency must be positive"
            raise ValueError(msg)
        return value

    @field_validator("gpu_delay_seconds")
    @classmethod
    def _validate_gpu_delay(cls, value: int) -> int:
        if value < 0:
            msg = "gpu_delay_seconds must be non-negative"
            raise ValueError(msg)
        return value


class PreprocessingScientificIntent(FrozenConfigModel):
    """Authored scientific intent with Database Set names deliberately absent."""

    schema_version: Literal[1, 2, 3] = CURRENT_PREPROCESSING_SCIENTIFIC_SCHEMA_VERSION
    max_sequences: StrictInt | None = 10_000
    use_env: StrictBool = False
    require_afdb_model_id_stem: StrictBool = True

    @model_validator(mode="before")
    @classmethod
    def _validate_schema(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        schema_version = data.get("schema_version")
        if schema_version is None:
            # Absent: the field default (v3) applies for fresh authoring.
            return data
        schema_version = _validate_scientific_schema_version(
            schema_version, record_name="PreprocessingScientificIntent"
        )
        if schema_version == 1:
            use_env = data.get("use_env")
            require_afdb_model_id_stem = data.get("require_afdb_model_id_stem")
            if use_env is None and require_afdb_model_id_stem is None:
                # v1 gate-absent semantics: inject the two knobs as False/False.
                data["use_env"] = False
                data["require_afdb_model_id_stem"] = False
            elif use_env is False and require_afdb_model_id_stem is False:
                # The contract's own intent->config propagation passes values explicitly.
                pass
            else:
                raise ValueError("v2 scientific fields in a schema_version 1 record")
        data["schema_version"] = schema_version
        return data

    @field_validator("max_sequences")
    @classmethod
    def _validate_max_sequences(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_sequences must be positive or null")
        return value


def scientific_canonical_mapping(
    scientific: PreprocessingScientificConfig | PreprocessingScientificIntent,
) -> dict[str, object]:
    """Return the canonical digest/file-write mapping for one scientific record.

    v1 records carry gate-absent semantics: the two identity-bearing knobs
    (use_env, require_afdb_model_id_stem) are excluded so retained sealed
    evidence stays byte-stable. v2/v3 records include both knobs and their
    policy-bearing scoped version. Fail closed if a v1 record reaches this helper
    with non-False/False knob values (reachable only via ``model_copy`` bypass,
    since the model validator rejects them).
    """
    mapping = scientific.model_dump(mode="json")
    if scientific.schema_version == 1:
        if scientific.use_env is not False or scientific.require_afdb_model_id_stem is not False:
            raise ValueError(
                "schema_version 1 scientific record must carry use_env=False and require_afdb_model_id_stem=False"
            )
        mapping.pop("use_env")
        mapping.pop("require_afdb_model_id_stem")
    return mapping


class PreprocessingSiteIntent(FrozenConfigModel):
    """Authored non-database paths, executables, image, and scheduling choices."""

    schema_version: Literal[1] = CURRENT_CONTRACT_SCHEMA_VERSION
    mmseqs_executable: StrictStr
    colabfold_search_executable: StrictStr
    tar_executable: StrictStr
    lz4_executable: StrictStr
    input_root: StrictStr
    scratch_output_root: StrictStr
    project_logs_root: StrictStr
    finished_msa_root: StrictStr
    split_input_root: StrictStr
    finished_input_root: StrictStr
    container_image: StrictStr
    container_mounts: tuple[StrictStr, ...]
    max_concurrency: StrictInt
    gpu_delay_seconds: StrictInt

    @model_validator(mode="before")
    @classmethod
    def _validate_schema(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        data["schema_version"] = validate_schema_version(
            data.get("schema_version"), record_name="PreprocessingSiteIntent"
        )
        return data

    @field_validator(
        "mmseqs_executable",
        "colabfold_search_executable",
        "tar_executable",
        "lz4_executable",
        "input_root",
        "scratch_output_root",
        "project_logs_root",
        "finished_msa_root",
        "split_input_root",
        "finished_input_root",
        "container_image",
    )
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        if not value:
            raise ValueError("site paths and executables must be non-empty")
        return value

    @field_validator("container_mounts")
    @classmethod
    def _validate_mounts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not mount for mount in value):
            raise ValueError("container mounts must be non-empty strings")
        return value

    @field_validator("max_concurrency")
    @classmethod
    def _validate_concurrency(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_concurrency must be positive")
        return value

    @field_validator("gpu_delay_seconds")
    @classmethod
    def _validate_gpu_delay(cls, value: int) -> int:
        if value < 0:
            raise ValueError("gpu_delay_seconds must be non-negative")
        return value


@dataclass(frozen=True)
class PreprocessingRuntimeCoordinates:
    """Ephemeral runtime coordinates used only for execution scratch paths."""

    slurm_node_id: int
    gpu_id: int
    submission_counter: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingRuntimeCoordinates")
        for field_name, value in (
            ("slurm_node_id", self.slurm_node_id),
            ("gpu_id", self.gpu_id),
            ("submission_counter", self.submission_counter),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                msg = f"{field_name} must be a non-negative integer"
                raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready runtime coordinates."""
        return {
            "schema_version": self.schema_version,
            "slurm_node_id": self.slurm_node_id,
            "gpu_id": self.gpu_id,
            "submission_counter": self.submission_counter,
        }


@dataclass(frozen=True)
class ExpectedA3M:
    """One declared A3M member associated with its source FASTA record."""

    chunk_name: str
    record_identity: str
    source_ordinal: int
    source_header: str
    member_name: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "ExpectedA3M")
        if not re.fullmatch(r"[^/]+_tranche\d{2}_\d{5}\.fa", self.chunk_name):
            msg = "chunk_name must be a canonical five-digit preprocessing chunk name"
            raise ValueError(msg)
        if not self.source_header.startswith(">") or len(self.source_header) == 1 or self.source_header[1].isspace():
            msg = "source_header must contain a canonical FASTA identity immediately after '>'"
            raise ValueError(msg)
        header_identity = self.source_header[1:].split(maxsplit=1)[0]
        if not self.record_identity or header_identity != self.record_identity:
            msg = "record identity must match the source FASTA header"
            raise ValueError(msg)
        if not isinstance(self.source_ordinal, int) or isinstance(self.source_ordinal, bool) or self.source_ordinal < 0:
            msg = "source_ordinal must be a non-negative integer"
            raise ValueError(msg)
        if not self.member_name.endswith(".a3m") or "/" in self.member_name or self.member_name in {".a3m", "..a3m"}:
            msg = "A3M member names must be top-level .a3m basenames"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready expected-output data."""
        return {
            "schema_version": self.schema_version,
            "chunk_name": self.chunk_name,
            "record_identity": self.record_identity,
            "source_ordinal": self.source_ordinal,
            "source_header": self.source_header,
            "member_name": self.member_name,
        }


@dataclass(frozen=True)
class PreprocessingEvidencePlan:
    """Scratch and durable evidence locations for one chunk."""

    chunk_name: str
    raw_search_output_directory: str
    scratch_output_directory: str
    scratch_log_directory: str
    scratch_log_path: str
    scratch_record_path: str
    a3m_record_glob: str
    durable_log_path: str
    durable_record_path: str
    missing_glob_writes_no_such_file: bool = True
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingEvidencePlan")
        values = (
            self.chunk_name,
            self.raw_search_output_directory,
            self.scratch_output_directory,
            self.scratch_log_directory,
            self.scratch_log_path,
            self.scratch_record_path,
            self.a3m_record_glob,
            self.durable_log_path,
            self.durable_record_path,
        )
        if any(not value for value in values):
            msg = "preprocessing evidence paths must be non-empty"
            raise ValueError(msg)
        if not self.missing_glob_writes_no_such_file:
            msg = "baseline record evidence retains the missing-glob No such file signal"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready evidence-path data."""
        return {
            "schema_version": self.schema_version,
            "chunk_name": self.chunk_name,
            "raw_search_output_directory": self.raw_search_output_directory,
            "scratch_output_directory": self.scratch_output_directory,
            "scratch_log_directory": self.scratch_log_directory,
            "scratch_log_path": self.scratch_log_path,
            "scratch_record_path": self.scratch_record_path,
            "a3m_record_glob": self.a3m_record_glob,
            "durable_log_path": self.durable_log_path,
            "durable_record_path": self.durable_record_path,
            "missing_glob_writes_no_such_file": self.missing_glob_writes_no_such_file,
        }


@dataclass(frozen=True)
class PreprocessingPackagePlan:
    """Declared staging inventory and literal baseline tar/lz4 plan."""

    chunk_name: str
    staging_directory: str
    declared_stage_members: tuple[str, ...]
    tar_member_scope: Literal["."]
    scratch_tar_path: str
    scratch_lz4_path: str
    durable_tar_path: str
    durable_lz4_path: str
    completed_input_source_path: str
    completed_input_path: str
    tar_argv: tuple[str, ...]
    lz4_argv: tuple[str, ...]
    archive_inside_staging_directory: bool = True
    declared_members_exhaustive: bool = False
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingPackagePlan")
        if not self.declared_stage_members or len(set(self.declared_stage_members)) != len(self.declared_stage_members):
            msg = "package staging members must be non-empty and unique"
            raise ValueError(msg)
        if self.tar_member_scope != ".":
            msg = "baseline tar member scope must remain '.'"
            raise ValueError(msg)
        if not self.archive_inside_staging_directory or self.declared_members_exhaustive:
            msg = "package plan must preserve self-archive scope without claiming exhaustive tar members"
            raise ValueError(msg)
        if not self.tar_argv or not self.lz4_argv:
            msg = "package argv must be non-empty immutable vectors"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready package data."""
        return {
            "schema_version": self.schema_version,
            "chunk_name": self.chunk_name,
            "staging_directory": self.staging_directory,
            "declared_stage_members": list(self.declared_stage_members),
            "tar_member_scope": self.tar_member_scope,
            "scratch_tar_path": self.scratch_tar_path,
            "scratch_lz4_path": self.scratch_lz4_path,
            "durable_tar_path": self.durable_tar_path,
            "durable_lz4_path": self.durable_lz4_path,
            "completed_input_source_path": self.completed_input_source_path,
            "completed_input_path": self.completed_input_path,
            "tar_argv": list(self.tar_argv),
            "lz4_argv": list(self.lz4_argv),
            "archive_inside_staging_directory": self.archive_inside_staging_directory,
            "declared_members_exhaustive": self.declared_members_exhaustive,
        }


@dataclass(frozen=True)
class PreprocessingChunkExecutionIntent:
    """Authored chunk intent before Database Set names and commands are materialized."""

    chunk_name: str
    scientific: PreprocessingScientificIntent
    site: PreprocessingSiteIntent
    runtime: PreprocessingRuntimeCoordinates
    expected_a3ms: tuple[ExpectedA3M, ...]
    evidence: PreprocessingEvidencePlan
    package: PreprocessingPackagePlan
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingChunkExecutionIntent")
        if not isinstance(self.scientific, PreprocessingScientificIntent):
            raise ValueError("execution intent scientific field must use PreprocessingScientificIntent")
        if not isinstance(self.site, PreprocessingSiteIntent):
            raise ValueError("execution intent site field must use PreprocessingSiteIntent")
        # Reuse the mature materialized contract to validate all non-database
        # evidence and package relationships without duplicating those rules.
        materialize_preprocessing_chunk_execution_plan(
            self,
            selected_database_root="/run/bspp/database/intent-validation",
            primary_database_name="intent_primary",
            metagenomic_database_name="intent_metagenomic",
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "chunk_name": self.chunk_name,
            "scientific": scientific_canonical_mapping(self.scientific),
            "site": self.site.model_dump(mode="json"),
            "runtime": self.runtime.to_mapping(),
            "expected_a3ms": [expected.to_mapping() for expected in self.expected_a3ms],
            "evidence": self.evidence.to_mapping(),
            "package": self.package.to_mapping(),
        }


@dataclass(frozen=True)
class PreprocessingChunkExecutionPlan:
    """One immutable and non-executing Scientific Kernel command plan."""

    chunk_name: str
    scientific: PreprocessingScientificConfig
    site: PreprocessingSiteConfig
    runtime: PreprocessingRuntimeCoordinates
    expected_a3ms: tuple[ExpectedA3M, ...]
    evidence: PreprocessingEvidencePlan
    package: PreprocessingPackagePlan
    gpuserver_argv: tuple[str, ...]
    gpuserver_environment: tuple[tuple[str, str], ...]
    search_argv: tuple[str, ...]
    search_environment: tuple[tuple[str, str], ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION
    # Deserialization-only compatibility seam: when True, the
    # ``--pair-mode`` position also accepts the legacy ``paired`` argv of
    # sealed legacy evidence.  Set only by
    # ``preprocessing_chunk_execution_plan_from_mapping``; never serialized,
    # excluded from equality; fresh construction defaults to strict.
    allow_legacy_pair_mode: bool = field(default=False, kw_only=True, compare=False, repr=False)

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingChunkExecutionPlan")
        _validate_execution_plan(self, allow_legacy_pair_mode=self.allow_legacy_pair_mode)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready execution-plan data."""
        return {
            "schema_version": self.schema_version,
            "chunk_name": self.chunk_name,
            "scientific": scientific_canonical_mapping(self.scientific),
            "site": self.site.model_dump(mode="json"),
            "runtime": self.runtime.to_mapping(),
            "expected_a3ms": [expected.to_mapping() for expected in self.expected_a3ms],
            "evidence": self.evidence.to_mapping(),
            "package": self.package.to_mapping(),
            "gpuserver_argv": list(self.gpuserver_argv),
            "gpuserver_environment": [list(entry) for entry in self.gpuserver_environment],
            "search_argv": list(self.search_argv),
            "search_environment": [list(entry) for entry in self.search_environment],
        }


def preprocessing_chunk_execution_intent_from_plan(
    plan: PreprocessingChunkExecutionPlan,
) -> PreprocessingChunkExecutionIntent:
    """Drop materialized database names, root, and argv from an existing plan."""
    site = plan.site
    return PreprocessingChunkExecutionIntent(
        chunk_name=plan.chunk_name,
        scientific=PreprocessingScientificIntent(
            schema_version=plan.scientific.schema_version,
            max_sequences=plan.scientific.max_sequences,
            use_env=plan.scientific.use_env,
            require_afdb_model_id_stem=plan.scientific.require_afdb_model_id_stem,
        ),
        site=PreprocessingSiteIntent(
            mmseqs_executable=site.mmseqs_executable,
            colabfold_search_executable=site.colabfold_search_executable,
            tar_executable=site.tar_executable,
            lz4_executable=site.lz4_executable,
            input_root=site.input_root,
            scratch_output_root=site.scratch_output_root,
            project_logs_root=site.project_logs_root,
            finished_msa_root=site.finished_msa_root,
            split_input_root=site.split_input_root,
            finished_input_root=site.finished_input_root,
            container_image=site.container_image,
            container_mounts=site.container_mounts,
            max_concurrency=site.max_concurrency,
            gpu_delay_seconds=site.gpu_delay_seconds,
        ),
        runtime=plan.runtime,
        expected_a3ms=plan.expected_a3ms,
        evidence=plan.evidence,
        package=plan.package,
    )


def materialize_preprocessing_chunk_execution_plan(
    intent: PreprocessingChunkExecutionIntent,
    *,
    selected_database_root: str,
    primary_database_name: str,
    metagenomic_database_name: str,
) -> PreprocessingChunkExecutionPlan:
    """Purely bind one authored intent to exact database names, root, and argv."""
    scientific = PreprocessingScientificConfig(
        schema_version=intent.scientific.schema_version,
        primary_database_name=primary_database_name,
        metagenomic_database_name=metagenomic_database_name,
        max_sequences=intent.scientific.max_sequences,
        use_env=intent.scientific.use_env,
        require_afdb_model_id_stem=intent.scientific.require_afdb_model_id_stem,
    )
    authored_site = intent.site
    site = PreprocessingSiteConfig(
        mmseqs_executable=authored_site.mmseqs_executable,
        colabfold_search_executable=authored_site.colabfold_search_executable,
        tar_executable=authored_site.tar_executable,
        lz4_executable=authored_site.lz4_executable,
        database_root=selected_database_root,
        input_root=authored_site.input_root,
        scratch_output_root=authored_site.scratch_output_root,
        project_logs_root=authored_site.project_logs_root,
        finished_msa_root=authored_site.finished_msa_root,
        split_input_root=authored_site.split_input_root,
        finished_input_root=authored_site.finished_input_root,
        container_image=authored_site.container_image,
        container_mounts=authored_site.container_mounts,
        max_concurrency=authored_site.max_concurrency,
        gpu_delay_seconds=authored_site.gpu_delay_seconds,
    )
    gpu_environment = (("CUDA_VISIBLE_DEVICES", str(intent.runtime.gpu_id)),)
    gpuserver_argv: tuple[str, ...] = (
        site.mmseqs_executable,
        "gpuserver",
        _join(selected_database_root, primary_database_name),
        "--db-load-mode",
        "0",
        "--prefilter-mode",
        "1",
    )
    if scientific.max_sequences is not None:
        gpuserver_argv = (*gpuserver_argv, "--max-seqs", str(scientific.max_sequences))
    runtime_folder = f"n{intent.runtime.slurm_node_id}g{intent.runtime.gpu_id}"
    search_argv = (
        site.colabfold_search_executable,
        "--mmseqs",
        site.mmseqs_executable,
        _join(site.input_root, runtime_folder, intent.chunk_name),
        selected_database_root,
        intent.evidence.raw_search_output_directory,
        "--use-env",
        "1" if scientific.use_env else "0",
        "--pairing_strategy",
        "1",
        "--pair-mode",
        "unpaired_paired",
        "--filter",
        _colabfold_filter_mode(scientific.schema_version),
        "--db-load-mode",
        "2",
        "--gpu",
        "1",
        "--gpu-server",
        "1",
        "--threads",
        "64",
        "--db1",
        primary_database_name,
        "--db3",
        metagenomic_database_name,
    )
    return PreprocessingChunkExecutionPlan(
        chunk_name=intent.chunk_name,
        scientific=scientific,
        site=site,
        runtime=intent.runtime,
        expected_a3ms=intent.expected_a3ms,
        evidence=intent.evidence,
        package=intent.package,
        gpuserver_argv=gpuserver_argv,
        gpuserver_environment=gpu_environment,
        search_argv=search_argv,
        search_environment=gpu_environment,
    )


def _validate_execution_plan(
    plan: PreprocessingChunkExecutionPlan,
    *,
    allow_legacy_pair_mode: bool = False,
) -> None:
    if not plan.expected_a3ms or any(expected.chunk_name != plan.chunk_name for expected in plan.expected_a3ms):
        msg = "expected A3Ms must be non-empty and reference the execution chunk"
        raise ValueError(msg)
    expected_members = tuple(expected.member_name for expected in plan.expected_a3ms)
    if len(set(expected_members)) != len(expected_members) or plan.package.declared_stage_members != expected_members:
        msg = "package members must exactly match unique expected A3M members"
        raise ValueError(msg)
    source_ordinals = tuple(expected.source_ordinal for expected in plan.expected_a3ms)
    record_identities = tuple(expected.record_identity for expected in plan.expected_a3ms)
    if len(set(source_ordinals)) != len(source_ordinals) or len(set(record_identities)) != len(record_identities):
        msg = "expected A3Ms must reference unique source ordinals and record identities"
        raise ValueError(msg)
    if plan.evidence.chunk_name != plan.chunk_name or plan.package.chunk_name != plan.chunk_name:
        msg = "evidence and package plans must reference the execution chunk"
        raise ValueError(msg)
    if plan.scientific.require_afdb_model_id_stem:
        for expected in plan.expected_a3ms:
            if not member_name_conforms(expected.member_name):
                msg = (
                    "A3M member stem must carry a discoverable AFDB or PDB assembly model ID "
                    "(AFDB_AF[-_]<16 digits> or AFDB_AF[-_]<16 digits>_AF[-_]<16 digits> "
                    r"or pdb_[a-z0-9]+_assembly_\d+): "
                    f"{expected.member_name}"
                )
                raise ValueError(msg)
            derived = expected_member_name(expected.record_identity)
            if expected.member_name != derived:
                msg = (
                    "A3M member name must equal the adapter-derived record-identity name "
                    f"(declared {expected.member_name!r}, derived {derived!r})"
                )
                raise ValueError(msg)

    runtime_folder = f"n{plan.runtime.slurm_node_id}g{plan.runtime.gpu_id}"
    scratch_suffix = f"{runtime_folder}_{plan.runtime.submission_counter}"
    chunk_stem = plan.chunk_name.removesuffix(".fa")
    scratch_output_directory = _join(plan.site.scratch_output_root, scratch_suffix)
    raw_search_output_directory = _join(plan.site.scratch_output_root, "raw-search", scratch_suffix)
    scratch_log_directory = _join(plan.site.scratch_output_root, "logs", scratch_suffix)
    expected_evidence = (
        raw_search_output_directory,
        scratch_output_directory,
        scratch_log_directory,
        _join(scratch_log_directory, f"{chunk_stem}.log"),
        _join(scratch_log_directory, f"{chunk_stem}.record"),
        _join(scratch_output_directory, "*.a3m"),
        _join(plan.site.project_logs_root, f"{chunk_stem}.log"),
        _join(plan.site.project_logs_root, f"{chunk_stem}.record"),
    )
    observed_evidence = (
        plan.evidence.raw_search_output_directory,
        plan.evidence.scratch_output_directory,
        plan.evidence.scratch_log_directory,
        plan.evidence.scratch_log_path,
        plan.evidence.scratch_record_path,
        plan.evidence.a3m_record_glob,
        plan.evidence.durable_log_path,
        plan.evidence.durable_record_path,
    )
    if observed_evidence != expected_evidence:
        msg = "evidence paths must preserve exact scratch coordinates and flat durable identity"
        raise ValueError(msg)

    gpu_environment = (("CUDA_VISIBLE_DEVICES", str(plan.runtime.gpu_id)),)
    expected_gpuserver: tuple[str, ...] = (
        plan.site.mmseqs_executable,
        "gpuserver",
        _join(plan.site.database_root, plan.scientific.primary_database_name),
        "--db-load-mode",
        "0",
        "--prefilter-mode",
        "1",
    )
    if plan.scientific.max_sequences is not None:
        expected_gpuserver = (*expected_gpuserver, "--max-seqs", str(plan.scientific.max_sequences))
    expected_search = (
        plan.site.colabfold_search_executable,
        "--mmseqs",
        plan.site.mmseqs_executable,
        _join(plan.site.input_root, runtime_folder, plan.chunk_name),
        plan.site.database_root,
        raw_search_output_directory,
        "--use-env",
        "1" if plan.scientific.use_env else "0",
        "--pairing_strategy",
        "1",
        "--pair-mode",
        "unpaired_paired",
        "--filter",
        _colabfold_filter_mode(plan.scientific.schema_version),
        "--db-load-mode",
        "2",
        "--gpu",
        "1",
        "--gpu-server",
        "1",
        "--threads",
        "64",
        "--db1",
        plan.scientific.primary_database_name,
        "--db3",
        plan.scientific.metagenomic_database_name,
    )
    # Accept "paired" (old sealed evidence) at the --pair-mode position only
    # when the plan itself carries the deserialization-only compatibility seam
    # (``allow_legacy_pair_mode=True``, set solely by
    # ``preprocessing_chunk_execution_plan_from_mapping``).  Fresh construction
    # (materializer, dataclasses.replace of a current plan, hand-authored
    # RunSpecs) keeps the default False, so a plan pinned to "paired" is
    # rejected at authoring time. Scientific v3 also rejects the
    # historical paired mode at deserialization; no v3 evidence used that mode.
    pm_idx = expected_search.index("--pair-mode")
    paired_search = (
        *expected_search[: pm_idx + 1],
        "paired",
        *expected_search[pm_idx + 2 :],
    )
    if (
        plan.gpuserver_argv != expected_gpuserver
        or (
            plan.search_argv != expected_search
            and not (
                allow_legacy_pair_mode and plan.scientific.schema_version < 3 and plan.search_argv == paired_search
            )
        )
        or plan.gpuserver_environment != gpu_environment
        or plan.search_environment != gpu_environment
    ):
        msg = "Scientific Kernel argv and environments must match the pinned command vectors"
        raise ValueError(msg)

    scratch_tar_path = _join(scratch_output_directory, f"{chunk_stem}.tar")
    expected_package = (
        scratch_output_directory,
        scratch_tar_path,
        f"{scratch_tar_path}.lz4",
        _join(plan.site.finished_msa_root, f"{chunk_stem}.tar"),
        _join(plan.site.finished_msa_root, f"{chunk_stem}.tar.lz4"),
        _join(plan.site.split_input_root, plan.chunk_name),
        _join(plan.site.finished_input_root, plan.chunk_name),
        (plan.site.tar_executable, "cf", scratch_tar_path, "-C", scratch_output_directory, "."),
        (plan.site.lz4_executable, "-v", "-3", scratch_tar_path, f"{scratch_tar_path}.lz4"),
    )
    observed_package = (
        plan.package.staging_directory,
        plan.package.scratch_tar_path,
        plan.package.scratch_lz4_path,
        plan.package.durable_tar_path,
        plan.package.durable_lz4_path,
        plan.package.completed_input_source_path,
        plan.package.completed_input_path,
        plan.package.tar_argv,
        plan.package.lz4_argv,
    )
    if observed_package != expected_package:
        msg = "package paths and argv must match the literal baseline tar/lz4 plan"
        raise ValueError(msg)


def _join(root: str, *parts: str) -> str:
    return "/".join((root.rstrip("/"), *parts))


def _validate_direct_schema_version(schema_version: int, record_name: str) -> None:
    validated = validate_schema_version(schema_version, record_name=record_name)
    if validated != schema_version:
        msg = f"{record_name} schema_version must be declared explicitly"
        raise ValueError(msg)


def preprocessing_runtime_coordinates_from_mapping(payload: Mapping[str, object]) -> PreprocessingRuntimeCoordinates:
    """Load ephemeral runtime coordinates from a fail-closed mapping."""
    _reject_unknown_fields(
        payload,
        {"schema_version", "slurm_node_id", "gpu_id", "submission_counter"},
        "PreprocessingRuntimeCoordinates",
    )
    return PreprocessingRuntimeCoordinates(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingRuntimeCoordinates"
        ),
        slurm_node_id=_required_int(payload, "slurm_node_id"),
        gpu_id=_required_int(payload, "gpu_id"),
        submission_counter=_required_int(payload, "submission_counter"),
    )


def expected_a3m_from_mapping(payload: Mapping[str, object]) -> ExpectedA3M:
    """Load one expected A3M association from a fail-closed mapping."""
    _reject_unknown_fields(
        payload,
        {"schema_version", "chunk_name", "record_identity", "source_ordinal", "source_header", "member_name"},
        "ExpectedA3M",
    )
    return ExpectedA3M(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="ExpectedA3M"),
        chunk_name=_required_str(payload, "chunk_name"),
        record_identity=_required_str(payload, "record_identity"),
        source_ordinal=_required_int(payload, "source_ordinal"),
        source_header=_required_str(payload, "source_header"),
        member_name=_required_str(payload, "member_name"),
    )


def preprocessing_evidence_plan_from_mapping(payload: Mapping[str, object]) -> PreprocessingEvidencePlan:
    """Load chunk evidence paths from a fail-closed mapping."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "chunk_name",
            "raw_search_output_directory",
            "scratch_output_directory",
            "scratch_log_directory",
            "scratch_log_path",
            "scratch_record_path",
            "a3m_record_glob",
            "durable_log_path",
            "durable_record_path",
            "missing_glob_writes_no_such_file",
        },
        "PreprocessingEvidencePlan",
    )
    return PreprocessingEvidencePlan(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingEvidencePlan"),
        chunk_name=_required_str(payload, "chunk_name"),
        raw_search_output_directory=_required_str(payload, "raw_search_output_directory"),
        scratch_output_directory=_required_str(payload, "scratch_output_directory"),
        scratch_log_directory=_required_str(payload, "scratch_log_directory"),
        scratch_log_path=_required_str(payload, "scratch_log_path"),
        scratch_record_path=_required_str(payload, "scratch_record_path"),
        a3m_record_glob=_required_str(payload, "a3m_record_glob"),
        durable_log_path=_required_str(payload, "durable_log_path"),
        durable_record_path=_required_str(payload, "durable_record_path"),
        missing_glob_writes_no_such_file=_required_bool(payload, "missing_glob_writes_no_such_file"),
    )


def preprocessing_package_plan_from_mapping(payload: Mapping[str, object]) -> PreprocessingPackagePlan:
    """Load a package plan from a fail-closed mapping."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "chunk_name",
            "staging_directory",
            "declared_stage_members",
            "tar_member_scope",
            "scratch_tar_path",
            "scratch_lz4_path",
            "durable_tar_path",
            "durable_lz4_path",
            "completed_input_source_path",
            "completed_input_path",
            "tar_argv",
            "lz4_argv",
            "archive_inside_staging_directory",
            "declared_members_exhaustive",
        },
        "PreprocessingPackagePlan",
    )
    tar_member_scope = _required_str(payload, "tar_member_scope")
    if tar_member_scope != ".":
        msg = "tar_member_scope must be '.'"
        raise ValueError(msg)
    return PreprocessingPackagePlan(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PreprocessingPackagePlan"),
        chunk_name=_required_str(payload, "chunk_name"),
        staging_directory=_required_str(payload, "staging_directory"),
        declared_stage_members=_required_str_tuple(payload, "declared_stage_members"),
        tar_member_scope=cast("Literal['.']", tar_member_scope),
        scratch_tar_path=_required_str(payload, "scratch_tar_path"),
        scratch_lz4_path=_required_str(payload, "scratch_lz4_path"),
        durable_tar_path=_required_str(payload, "durable_tar_path"),
        durable_lz4_path=_required_str(payload, "durable_lz4_path"),
        completed_input_source_path=_required_str(payload, "completed_input_source_path"),
        completed_input_path=_required_str(payload, "completed_input_path"),
        tar_argv=_required_str_tuple(payload, "tar_argv"),
        lz4_argv=_required_str_tuple(payload, "lz4_argv"),
        archive_inside_staging_directory=_required_bool(payload, "archive_inside_staging_directory"),
        declared_members_exhaustive=_required_bool(payload, "declared_members_exhaustive"),
    )


def preprocessing_chunk_execution_plan_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingChunkExecutionPlan:
    """Load a complete execution plan and validate every nested reference.

    Deserialization carries the compatibility seam locally on the constructed
    plan (``allow_legacy_pair_mode=True``) so sealed legacy evidence with
    ``--pair-mode paired`` still loads; no module-global state is involved.
    """
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "chunk_name",
            "scientific",
            "site",
            "runtime",
            "expected_a3ms",
            "evidence",
            "package",
            "gpuserver_argv",
            "gpuserver_environment",
            "search_argv",
            "search_environment",
        },
        "PreprocessingChunkExecutionPlan",
    )
    return PreprocessingChunkExecutionPlan(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingChunkExecutionPlan"
        ),
        chunk_name=_required_str(payload, "chunk_name"),
        scientific=PreprocessingScientificConfig.model_validate(_required_mapping(payload, "scientific")),
        site=PreprocessingSiteConfig.model_validate(_required_mapping(payload, "site")),
        runtime=preprocessing_runtime_coordinates_from_mapping(_required_mapping(payload, "runtime")),
        expected_a3ms=tuple(
            expected_a3m_from_mapping(item) for item in _required_mapping_sequence(payload, "expected_a3ms")
        ),
        evidence=preprocessing_evidence_plan_from_mapping(_required_mapping(payload, "evidence")),
        package=preprocessing_package_plan_from_mapping(_required_mapping(payload, "package")),
        gpuserver_argv=_required_str_tuple(payload, "gpuserver_argv"),
        gpuserver_environment=_required_environment(payload, "gpuserver_environment"),
        search_argv=_required_str_tuple(payload, "search_argv"),
        search_environment=_required_environment(payload, "search_environment"),
        allow_legacy_pair_mode=True,
    )


def preprocessing_chunk_execution_intent_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingChunkExecutionIntent:
    """Strict-load authored chunk intent with database authority absent."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "chunk_name",
            "scientific",
            "site",
            "runtime",
            "expected_a3ms",
            "evidence",
            "package",
        },
        "PreprocessingChunkExecutionIntent",
    )
    return PreprocessingChunkExecutionIntent(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingChunkExecutionIntent"
        ),
        chunk_name=_required_str(payload, "chunk_name"),
        scientific=PreprocessingScientificIntent.model_validate(_required_mapping(payload, "scientific")),
        site=PreprocessingSiteIntent.model_validate(_required_mapping(payload, "site")),
        runtime=preprocessing_runtime_coordinates_from_mapping(_required_mapping(payload, "runtime")),
        expected_a3ms=tuple(
            expected_a3m_from_mapping(item) for item in _required_mapping_sequence(payload, "expected_a3ms")
        ),
        evidence=preprocessing_evidence_plan_from_mapping(_required_mapping(payload, "evidence")),
        package=preprocessing_package_plan_from_mapping(_required_mapping(payload, "package")),
    )


def _reject_unknown_fields(payload: Mapping[str, object], allowed: set[str], record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = f"Unknown {record_name} field(s): {', '.join(unknown)}"
        raise ValueError(msg)


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be a mapping"
        raise ValueError(msg)
    return value


def _required_mapping_sequence(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"{key}[{index}] must be a mapping"
            raise ValueError(msg)
        result.append(item)
    return tuple(result)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{key} must be a non-empty string"
        raise ValueError(msg)
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean"
        raise ValueError(msg)
    return value


def _required_str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple) or not all(isinstance(item, str) for item in value):
        msg = f"{key} must be a list of strings"
        raise ValueError(msg)
    return tuple(value)


def _required_environment(payload: Mapping[str, object], key: str) -> tuple[tuple[str, str], ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list of key/value pairs"
        raise ValueError(msg)
    result: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list | tuple) or len(item) != 2 or not all(isinstance(part, str) for part in item):
            msg = f"{key}[{index}] must be a string key/value pair"
            raise ValueError(msg)
        result.append((item[0], item[1]))
    return tuple(result)


__all__ = [
    "ExpectedA3M",
    "PreprocessingChunkExecutionIntent",
    "PreprocessingChunkExecutionPlan",
    "PreprocessingEvidencePlan",
    "PreprocessingPackagePlan",
    "PreprocessingRuntimeCoordinates",
    "PreprocessingScientificConfig",
    "PreprocessingScientificIntent",
    "PreprocessingSiteConfig",
    "PreprocessingSiteIntent",
    "afdb_model_id_member_name_conforms",
    "expected_a3m_from_mapping",
    "expected_member_name",
    "is_pdb_assembly_member_name",
    "materialize_preprocessing_chunk_execution_plan",
    "member_name_conforms",
    "preprocessing_chunk_execution_intent_from_mapping",
    "preprocessing_chunk_execution_intent_from_plan",
    "preprocessing_chunk_execution_plan_from_mapping",
    "preprocessing_evidence_plan_from_mapping",
    "preprocessing_package_plan_from_mapping",
    "preprocessing_runtime_coordinates_from_mapping",
    "safe_filename",
    "scientific_canonical_mapping",
    "validate_scientific_schema_version",
]
