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

"""Closed contracts used by executable folding Runtime Actions.

This module owns the versioned, closed records that the folding Runtime
executor publishes and reloads across the immutable five-action graph
``msa-flatten -> split -> preprocess -> fold -> canonical-pair``:

- ``FoldingBackendAssetsSnapshot`` — the container-visible (never secret)
  external weights/checkpoint/chain-manifest paths selected for one backend.
- ``FoldingTargetIdentity`` — the deterministic identity of one derived
  folding target (normalized member stem, description, ordered chains, and the
  colon-joined sequence SHA-256 matching benchmark corpus semantics).
- One closed, versioned handoff record per action kind. Each handoff carries
  the phase/attempt/action identity and the predecessor handoff digest so a
  successor action can prove it consumed the exact durable predecessor bytes.

The module is deliberately self-contained: it imports only the shared
``versioning`` contract (and lazily imports ``prediction_pair`` and
``preprocessing_handoff`` loaders inside functions) so ``phase.py`` can import
these records without a module-level circular dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, cast

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

if TYPE_CHECKING:
    from bspp.orchestration.contract.prediction_pair import PredictionPair

_BACKENDS = {"openfold-cli", "colabfold", "bioir", "openfold-trt"}
_FOLDING_ACTION_KINDS = ("msa-flatten", "split", "preprocess", "fold", "canonical-pair")
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_ACTION_ID = re.compile(r"(msa-flatten|split|preprocess|fold|canonical-pair)-[0-9]{6}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_IDENTITY_FIELDS = ("schema_version", "phase_run_id", "attempt_id", "action_id", "predecessor_digest")

# openfold-trt stays fail-closed until a documented image-provided
# ``model_fn`` factory contract lands. No speculative factory field is added to
# the asset contract, so Control and Runtime share this one stable error.
OPENFOLD_TRT_DEFERRED_MODEL_FN_ERROR = (
    "openfold-trt folding is deferred: no documented image-provided model_fn factory is available"
)


def folding_target_sequence_sha256(chains: tuple[str, ...]) -> str:
    """Return the deterministic colon-joined sequence SHA-256 for ordered chains."""
    return hashlib.sha256(":".join(chains).encode("utf-8")).hexdigest()


def folding_action_kind_for_action_id(action_id: str) -> str:
    """Return the folding action kind encoded in one action id."""
    match = _ACTION_ID.fullmatch(action_id)
    if match is None:
        raise ValueError(f"folding action id must be <kind>-NNNNNN: {action_id!r}")
    return match.group(1)


def _folding_handoff_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_absolute_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if not PurePosixPath(value).is_absolute():
        raise ValueError(f"{name} must be an absolute container path")
    if "\\" in value:
        raise ValueError(f"{name} must be a POSIX path")
    return value


def _validate_target_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("folding target_id must be a non-empty string")
    if value.startswith("/") or "\\" in value or any(component in {"", ".", ".."} for component in value.split("/")):
        raise ValueError("folding target_id must be a confined non-traversal identity")
    return value


def _validate_handoff_identity(
    phase_run_id: str,
    attempt_id: str,
    action_id: str,
    action_kind: str,
    predecessor_digest: str | None,
) -> None:
    if _PHASE_RUN_ID.fullmatch(phase_run_id) is None:
        raise ValueError(f"folding handoff phase_run_id must be phase-run-<32 hex>: {phase_run_id!r}")
    if _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise ValueError(f"folding handoff attempt_id must be attempt-NNNN: {attempt_id!r}")
    if folding_action_kind_for_action_id(action_id) != action_kind:
        raise ValueError(f"folding handoff action_id {action_id!r} must belong to kind {action_kind!r}")
    if predecessor_digest is None:
        if action_kind != "msa-flatten":
            raise ValueError(f"{action_kind} folding handoff requires a predecessor digest")
    elif _SHA256.fullmatch(predecessor_digest) is None:
        raise ValueError("folding handoff predecessor_digest must be 64 lowercase hex characters")


@dataclass(frozen=True)
class FoldingBackendAssetsSnapshot:
    """Container-visible, non-secret assets selected for one folding backend."""

    backend: str
    chain_manifest_csv: str | None = None
    openfold_model_dir: str | None = None
    colabfold_weights_dir: str | None = None
    bioir_checkpoint: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION
    bioir_monomer_checkpoint: str | None = None

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="FoldingBackendAssetsSnapshot")
        if self.backend not in _BACKENDS:
            raise ValueError(f"unsupported folding backend assets backend: {self.backend!r}")
        for name in (
            "chain_manifest_csv",
            "openfold_model_dir",
            "colabfold_weights_dir",
            "bioir_checkpoint",
            "bioir_monomer_checkpoint",
        ):
            value = getattr(self, name)
            if value is not None and (not value or not PurePosixPath(value).is_absolute()):
                raise ValueError(f"folding backend asset {name} must be an absolute non-empty path")
        if self.backend in {"openfold-cli", "colabfold"} and self.chain_manifest_csv is None:
            raise ValueError(f"{self.backend} requires chain_manifest_csv")
        if self.backend == "openfold-cli" and self.openfold_model_dir is None:
            raise ValueError("openfold-cli requires openfold_model_dir")
        if self.backend == "colabfold" and self.colabfold_weights_dir is None:
            raise ValueError("colabfold requires colabfold_weights_dir")
        if self.backend == "bioir":
            if self.bioir_checkpoint is None:
                raise ValueError("bioir requires bioir_checkpoint")
            if not self.bioir_checkpoint.endswith(".pt"):
                raise ValueError("bioir_checkpoint must end in .pt")
            if self.bioir_monomer_checkpoint is not None and not self.bioir_monomer_checkpoint.endswith(".pt"):
                raise ValueError("bioir_monomer_checkpoint must end in .pt")
        allowed = {
            "openfold-cli": {"chain_manifest_csv", "openfold_model_dir"},
            "colabfold": {"chain_manifest_csv", "colabfold_weights_dir"},
            "bioir": {"chain_manifest_csv", "bioir_checkpoint", "bioir_monomer_checkpoint"},
            "openfold-trt": {"chain_manifest_csv"},
        }[self.backend]
        for name in {
            "chain_manifest_csv",
            "openfold_model_dir",
            "colabfold_weights_dir",
            "bioir_checkpoint",
            "bioir_monomer_checkpoint",
        } - allowed:
            if getattr(self, name) is not None:
                raise ValueError(f"{self.backend} does not accept folding backend asset {name}")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {"schema_version": self.schema_version, "backend": self.backend}
        for name in (
            "chain_manifest_csv",
            "openfold_model_dir",
            "colabfold_weights_dir",
            "bioir_checkpoint",
            "bioir_monomer_checkpoint",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


def folding_backend_assets_snapshot_from_mapping(payload: Mapping[str, object]) -> FoldingBackendAssetsSnapshot:
    allowed = {
        "schema_version",
        "backend",
        "chain_manifest_csv",
        "openfold_model_dir",
        "colabfold_weights_dir",
        "bioir_checkpoint",
        "bioir_monomer_checkpoint",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown FoldingBackendAssetsSnapshot field(s): {', '.join(unknown)}")
    backend = payload.get("backend")
    if not isinstance(backend, str):
        raise ValueError("folding backend assets backend must be a string")
    values: dict[str, str | None] = {}
    for name in (
        "chain_manifest_csv",
        "openfold_model_dir",
        "colabfold_weights_dir",
        "bioir_checkpoint",
        "bioir_monomer_checkpoint",
    ):
        value = payload.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"folding backend asset {name} must be a string")
        values[name] = value
    return FoldingBackendAssetsSnapshot(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="FoldingBackendAssetsSnapshot"
        ),
        backend=backend,
        **values,
    )


@dataclass(frozen=True)
class FoldingTargetIdentity:
    """The deterministic identity of one derived folding target."""

    target_id: str
    description: str
    chains: tuple[str, ...]
    sequence_sha256: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="FoldingTargetIdentity")
        _validate_target_id(self.target_id)
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("folding target description must be a non-empty string")
        if (
            not isinstance(self.chains, tuple)
            or not self.chains
            or any(not isinstance(chain, str) or not chain for chain in self.chains)
        ):
            raise ValueError("folding target chains must be a non-empty immutable tuple of non-empty strings")
        if _SHA256.fullmatch(self.sequence_sha256) is None:
            raise ValueError("folding target sequence_sha256 must be 64 lowercase hex characters")
        if folding_target_sequence_sha256(self.chains) != self.sequence_sha256:
            raise ValueError("folding target sequence_sha256 does not match its ordered chains")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "description": self.description,
            "chains": list(self.chains),
            "sequence_sha256": self.sequence_sha256,
        }


def folding_target_identity_from_mapping(payload: Mapping[str, object]) -> FoldingTargetIdentity:
    _reject_unknown(
        payload,
        {"schema_version", "target_id", "description", "chains", "sequence_sha256"},
        "FoldingTargetIdentity",
    )
    chains = payload.get("chains")
    if not isinstance(chains, list | tuple) or any(not isinstance(item, str) or not item for item in chains):
        raise ValueError("folding target chains must be a list of non-empty strings")
    return FoldingTargetIdentity(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="FoldingTargetIdentity"),
        target_id=_str(payload, "target_id"),
        description=_str(payload, "description"),
        chains=tuple(chains),
        sequence_sha256=_str(payload, "sequence_sha256"),
    )


@dataclass(frozen=True)
class FoldingMsaFlattenHandoff:
    """msa-flatten action handoff: ordered projection inventory + local record."""

    phase_run_id: str
    attempt_id: str
    action_id: str
    predecessor_digest: str | None
    projected_members: tuple[tuple[str, str], ...]
    local_location: Mapping[str, object]
    install_mode: str | None = None
    orchestration_source_commit: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="FoldingMsaFlattenHandoff")
        _validate_handoff_identity(
            self.phase_run_id, self.attempt_id, self.action_id, "msa-flatten", self.predecessor_digest
        )
        if self.predecessor_digest is not None:
            raise ValueError("msa-flatten folding handoff must not declare a predecessor")
        if not isinstance(self.projected_members, tuple) or not self.projected_members:
            raise ValueError("msa-flatten projected_members must be a non-empty immutable tuple")
        for logical_path, projected_path in self.projected_members:
            if not isinstance(logical_path, str) or not logical_path:
                raise ValueError("msa-flatten projected member logical path must be non-empty")
            _validate_absolute_path(projected_path, "msa-flatten projected member path")
        if not isinstance(self.local_location, Mapping) or not self.local_location:
            raise ValueError("msa-flatten local_location must be a non-empty mapping")
        _validate_install_mode(self.install_mode)
        _validate_source_commit(self.orchestration_source_commit)

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "action_id": self.action_id,
            "predecessor_digest": self.predecessor_digest,
            "projected_members": [[logical, projected] for logical, projected in self.projected_members],
            "local_location": dict(self.local_location),
        }
        if self.install_mode is not None:
            result["install_mode"] = self.install_mode
        if self.orchestration_source_commit is not None:
            result["orchestration_source_commit"] = self.orchestration_source_commit
        return result


@dataclass(frozen=True)
class FoldingSplitTarget:
    """One split action target: derived identity plus merged source and split files."""

    target: FoldingTargetIdentity
    merged_source: str
    chain_files: tuple[str, ...]
    chain_count: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="FoldingSplitTarget")
        if not isinstance(self.target, FoldingTargetIdentity):
            raise ValueError("split target must be a FoldingTargetIdentity")
        _validate_absolute_path(self.merged_source, "split target merged_source")
        if not isinstance(self.chain_files, tuple) or any(
            not isinstance(path, str) or not path.endswith(".a3m") for path in self.chain_files
        ):
            raise ValueError("split target chain_files must be an immutable tuple of chain_*.a3m paths")
        if len(self.chain_files) != len(self.target.chains):
            raise ValueError("split target chain_files count must equal the target chain count")
        if self.chain_count != len(self.chain_files):
            raise ValueError("split target chain_count must equal the chain_files count")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target": self.target.to_mapping(),
            "merged_source": self.merged_source,
            "chain_files": list(self.chain_files),
            "chain_count": self.chain_count,
        }


@dataclass(frozen=True)
class FoldingSplitHandoff:
    """split action handoff: ordered derived targets with their split outputs."""

    phase_run_id: str
    attempt_id: str
    action_id: str
    predecessor_digest: str
    targets: tuple[FoldingSplitTarget, ...]
    install_mode: str | None = None
    orchestration_source_commit: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="FoldingSplitHandoff")
        _validate_handoff_identity(self.phase_run_id, self.attempt_id, self.action_id, "split", self.predecessor_digest)
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError("split handoff targets must be a non-empty immutable tuple")
        if any(not isinstance(target, FoldingSplitTarget) for target in self.targets):
            raise ValueError("split handoff targets must be FoldingSplitTarget records")
        _reject_duplicate_target_ids(tuple(target.target.target_id for target in self.targets))
        _validate_install_mode(self.install_mode)
        _validate_source_commit(self.orchestration_source_commit)

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "action_id": self.action_id,
            "predecessor_digest": self.predecessor_digest,
            "targets": [target.to_mapping() for target in self.targets],
        }
        if self.install_mode is not None:
            result["install_mode"] = self.install_mode
        if self.orchestration_source_commit is not None:
            result["orchestration_source_commit"] = self.orchestration_source_commit
        return result


@dataclass(frozen=True)
class FoldingPreprocessTarget:
    """One preprocess action target: identity plus backend layout and dirs."""

    target: FoldingTargetIdentity
    layout: str
    fasta_dir: str
    alignment_dir: str
    template_dir: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="FoldingPreprocessTarget")
        if not isinstance(self.target, FoldingTargetIdentity):
            raise ValueError("preprocess target must be a FoldingTargetIdentity")
        if self.layout not in {"openfold", "bioir", "colabfold"}:
            raise ValueError(f"unsupported preprocess target layout: {self.layout!r}")
        _validate_absolute_path(self.fasta_dir, "preprocess target fasta_dir")
        _validate_absolute_path(self.alignment_dir, "preprocess target alignment_dir")
        _validate_absolute_path(self.template_dir, "preprocess target template_dir")
        if PurePosixPath(self.fasta_dir).name != "fasta" or PurePosixPath(self.alignment_dir).name != "alignments":
            raise ValueError("preprocess target fasta_dir/alignment_dir must end in fasta/alignments")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target": self.target.to_mapping(),
            "layout": self.layout,
            "fasta_dir": self.fasta_dir,
            "alignment_dir": self.alignment_dir,
            "template_dir": self.template_dir,
        }


@dataclass(frozen=True)
class FoldingPreprocessHandoff:
    """preprocess action handoff: targets plus per-target prepared layout."""

    phase_run_id: str
    attempt_id: str
    action_id: str
    predecessor_digest: str
    targets: tuple[FoldingPreprocessTarget, ...]
    install_mode: str | None = None
    orchestration_source_commit: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="FoldingPreprocessHandoff")
        _validate_handoff_identity(
            self.phase_run_id, self.attempt_id, self.action_id, "preprocess", self.predecessor_digest
        )
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError("preprocess handoff targets must be a non-empty immutable tuple")
        if any(not isinstance(target, FoldingPreprocessTarget) for target in self.targets):
            raise ValueError("preprocess handoff targets must be FoldingPreprocessTarget records")
        _reject_duplicate_target_ids(tuple(target.target.target_id for target in self.targets))
        _validate_install_mode(self.install_mode)
        _validate_source_commit(self.orchestration_source_commit)

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "action_id": self.action_id,
            "predecessor_digest": self.predecessor_digest,
            "targets": [target.to_mapping() for target in self.targets],
        }
        if self.install_mode is not None:
            result["install_mode"] = self.install_mode
        if self.orchestration_source_commit is not None:
            result["orchestration_source_commit"] = self.orchestration_source_commit
        return result


@dataclass(frozen=True)
class FoldingFoldTarget:
    """One fold action target: identity plus its prediction pair and model metadata."""

    target: FoldingTargetIdentity
    pair: PredictionPair
    model_metadata: Mapping[str, object]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        from bspp.orchestration.contract.prediction_pair import PredictionPair

        validate_schema_version(self.schema_version, record_name="FoldingFoldTarget")
        if not isinstance(self.target, FoldingTargetIdentity):
            raise ValueError("fold target must be a FoldingTargetIdentity")
        if not isinstance(self.pair, PredictionPair):
            raise ValueError("fold target pair must be a PredictionPair")
        if not isinstance(self.model_metadata, Mapping):
            raise ValueError("fold target model_metadata must be a mapping")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target": self.target.to_mapping(),
            "pair": self.pair.to_mapping(),
            "model_metadata": dict(self.model_metadata),
        }


@dataclass(frozen=True)
class FoldingFoldHandoff:
    """fold action handoff: targets plus strict prediction pairs and model metadata."""

    phase_run_id: str
    attempt_id: str
    action_id: str
    predecessor_digest: str
    backend: str
    targets: tuple[FoldingFoldTarget, ...]
    install_mode: str | None = None
    orchestration_source_commit: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="FoldingFoldHandoff")
        _validate_handoff_identity(self.phase_run_id, self.attempt_id, self.action_id, "fold", self.predecessor_digest)
        if self.backend not in _BACKENDS:
            raise ValueError(f"unsupported fold handoff backend: {self.backend!r}")
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError("fold handoff targets must be a non-empty immutable tuple")
        if any(not isinstance(target, FoldingFoldTarget) for target in self.targets):
            raise ValueError("fold handoff targets must be FoldingFoldTarget records")
        _reject_duplicate_target_ids(tuple(target.target.target_id for target in self.targets))
        _validate_install_mode(self.install_mode)
        _validate_source_commit(self.orchestration_source_commit)

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "action_id": self.action_id,
            "predecessor_digest": self.predecessor_digest,
            "backend": self.backend,
            "targets": [target.to_mapping() for target in self.targets],
        }
        if self.install_mode is not None:
            result["install_mode"] = self.install_mode
        if self.orchestration_source_commit is not None:
            result["orchestration_source_commit"] = self.orchestration_source_commit
        return result


@dataclass(frozen=True)
class FoldingCanonicalPairEntry:
    """One canonical index entry: target identity bound to a prediction pair."""

    target_id: str
    sequence_sha256: str
    model_entity_id: str
    tool_used: str
    structure_path: str
    scores_path: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="FoldingCanonicalPairEntry")
        _validate_target_id(self.target_id)
        if _SHA256.fullmatch(self.sequence_sha256) is None:
            raise ValueError("canonical-pair entry sequence_sha256 must be 64 lowercase hex characters")
        if not isinstance(self.model_entity_id, str) or not self.model_entity_id:
            raise ValueError("canonical-pair entry model_entity_id must be non-empty")
        if not isinstance(self.tool_used, str) or not self.tool_used:
            raise ValueError("canonical-pair entry tool_used must be non-empty")
        _validate_absolute_path(self.structure_path, "canonical-pair entry structure_path")
        _validate_absolute_path(self.scores_path, "canonical-pair entry scores_path")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "sequence_sha256": self.sequence_sha256,
            "model_entity_id": self.model_entity_id,
            "tool_used": self.tool_used,
            "structure_path": self.structure_path,
            "scores_path": self.scores_path,
        }


@dataclass(frozen=True)
class FoldingCanonicalPairHandoff:
    """canonical-pair action handoff: the canonical index path, entries, and digest."""

    phase_run_id: str
    attempt_id: str
    action_id: str
    predecessor_digest: str
    index_path: str
    index_digest: str
    entries: tuple[FoldingCanonicalPairEntry, ...]
    install_mode: str | None = None
    orchestration_source_commit: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="FoldingCanonicalPairHandoff")
        _validate_handoff_identity(
            self.phase_run_id, self.attempt_id, self.action_id, "canonical-pair", self.predecessor_digest
        )
        _validate_absolute_path(self.index_path, "canonical-pair handoff index_path")
        if _SHA256.fullmatch(self.index_digest) is None:
            raise ValueError("canonical-pair handoff index_digest must be 64 lowercase hex characters")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("canonical-pair handoff entries must be a non-empty immutable tuple")
        if any(not isinstance(entry, FoldingCanonicalPairEntry) for entry in self.entries):
            raise ValueError("canonical-pair handoff entries must be FoldingCanonicalPairEntry records")
        _reject_duplicate_target_ids(tuple(entry.target_id for entry in self.entries))
        _validate_install_mode(self.install_mode)
        _validate_source_commit(self.orchestration_source_commit)

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "action_id": self.action_id,
            "predecessor_digest": self.predecessor_digest,
            "index_path": self.index_path,
            "index_digest": self.index_digest,
            "entries": [entry.to_mapping() for entry in self.entries],
        }
        if self.install_mode is not None:
            result["install_mode"] = self.install_mode
        if self.orchestration_source_commit is not None:
            result["orchestration_source_commit"] = self.orchestration_source_commit
        return result


def _reject_duplicate_target_ids(target_ids: tuple[str, ...]) -> None:
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("folding handoff must not contain duplicate target ids")


def _validate_install_mode(value: str | None) -> None:
    if value is not None and value not in {"override", "baked"}:
        raise ValueError(f"install_mode must be 'override' or 'baked' when set, got {value!r}")


def _validate_source_commit(value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError("orchestration_source_commit must be null or a non-empty string")


def _reject_unknown(payload: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")


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


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _mappings(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return cast("tuple[Mapping[str, object], ...]", tuple(value))


def _require_schema(payload: Mapping[str, object], name: str) -> int:
    if "schema_version" not in payload:
        raise ValueError(f"missing explicit schema_version at {name}")
    return validate_schema_version(payload.get("schema_version"), record_name=name)


def folding_msa_flatten_handoff_from_mapping(payload: Mapping[str, object]) -> FoldingMsaFlattenHandoff:
    _reject_unknown(
        payload,
        {
            "schema_version",
            "phase_run_id",
            "attempt_id",
            "action_id",
            "predecessor_digest",
            "projected_members",
            "local_location",
            "install_mode",
            "orchestration_source_commit",
        },
        "FoldingMsaFlattenHandoff",
    )
    raw_members = payload.get("projected_members")
    if not isinstance(raw_members, list | tuple) or any(
        not isinstance(item, list | tuple)
        or len(item) != 2
        or any(not isinstance(part, str) or not part for part in item)
        for item in raw_members
    ):
        raise ValueError("projected_members must be a list of two-item string pairs")
    local_location = _mapping(payload, "local_location")
    from bspp.orchestration.contract.preprocessing_handoff import verified_local_bundled_artifact_location_from_mapping

    verified_local_bundled_artifact_location_from_mapping(local_location)
    return FoldingMsaFlattenHandoff(
        schema_version=_require_schema(payload, "FoldingMsaFlattenHandoff"),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        action_id=_str(payload, "action_id"),
        predecessor_digest=_optional_str(payload, "predecessor_digest"),
        projected_members=tuple((item[0], item[1]) for item in raw_members),
        local_location=local_location,
        install_mode=_optional_str(payload, "install_mode"),
        orchestration_source_commit=_optional_str(payload, "orchestration_source_commit"),
    )


def _split_target_from_mapping(payload: Mapping[str, object]) -> FoldingSplitTarget:
    _reject_unknown(
        payload,
        {"schema_version", "target", "merged_source", "chain_files", "chain_count"},
        "FoldingSplitTarget",
    )
    chain_files = payload.get("chain_files")
    if not isinstance(chain_files, list | tuple):
        raise ValueError("chain_files must be a list")
    return FoldingSplitTarget(
        schema_version=_require_schema(payload, "FoldingSplitTarget"),
        target=folding_target_identity_from_mapping(_mapping(payload, "target")),
        merged_source=_str(payload, "merged_source"),
        chain_files=tuple(str(item) for item in chain_files),
        chain_count=_int(payload, "chain_count"),
    )


def folding_split_handoff_from_mapping(payload: Mapping[str, object]) -> FoldingSplitHandoff:
    _reject_unknown(
        payload,
        {*_IDENTITY_FIELDS, "targets", "install_mode", "orchestration_source_commit"},
        "FoldingSplitHandoff",
    )
    return FoldingSplitHandoff(
        schema_version=_require_schema(payload, "FoldingSplitHandoff"),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        action_id=_str(payload, "action_id"),
        predecessor_digest=_str(payload, "predecessor_digest"),
        targets=tuple(_split_target_from_mapping(item) for item in _mappings(payload, "targets")),
        install_mode=_optional_str(payload, "install_mode"),
        orchestration_source_commit=_optional_str(payload, "orchestration_source_commit"),
    )


def _preprocess_target_from_mapping(payload: Mapping[str, object]) -> FoldingPreprocessTarget:
    _reject_unknown(
        payload,
        {"schema_version", "target", "layout", "fasta_dir", "alignment_dir", "template_dir"},
        "FoldingPreprocessTarget",
    )
    return FoldingPreprocessTarget(
        schema_version=_require_schema(payload, "FoldingPreprocessTarget"),
        target=folding_target_identity_from_mapping(_mapping(payload, "target")),
        layout=_str(payload, "layout"),
        fasta_dir=_str(payload, "fasta_dir"),
        alignment_dir=_str(payload, "alignment_dir"),
        template_dir=_str(payload, "template_dir"),
    )


def folding_preprocess_handoff_from_mapping(payload: Mapping[str, object]) -> FoldingPreprocessHandoff:
    _reject_unknown(
        payload,
        {*_IDENTITY_FIELDS, "targets", "install_mode", "orchestration_source_commit"},
        "FoldingPreprocessHandoff",
    )
    return FoldingPreprocessHandoff(
        schema_version=_require_schema(payload, "FoldingPreprocessHandoff"),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        action_id=_str(payload, "action_id"),
        predecessor_digest=_str(payload, "predecessor_digest"),
        targets=tuple(_preprocess_target_from_mapping(item) for item in _mappings(payload, "targets")),
        install_mode=_optional_str(payload, "install_mode"),
        orchestration_source_commit=_optional_str(payload, "orchestration_source_commit"),
    )


def _fold_target_from_mapping(payload: Mapping[str, object]) -> FoldingFoldTarget:
    from bspp.orchestration.contract.prediction_pair import prediction_pair_from_mapping

    _reject_unknown(payload, {"schema_version", "target", "pair", "model_metadata"}, "FoldingFoldTarget")
    return FoldingFoldTarget(
        schema_version=_require_schema(payload, "FoldingFoldTarget"),
        target=folding_target_identity_from_mapping(_mapping(payload, "target")),
        pair=prediction_pair_from_mapping(_mapping(payload, "pair")),
        model_metadata=_mapping(payload, "model_metadata"),
    )


def folding_fold_handoff_from_mapping(payload: Mapping[str, object]) -> FoldingFoldHandoff:
    _reject_unknown(
        payload,
        {*_IDENTITY_FIELDS, "backend", "targets", "install_mode", "orchestration_source_commit"},
        "FoldingFoldHandoff",
    )
    return FoldingFoldHandoff(
        schema_version=_require_schema(payload, "FoldingFoldHandoff"),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        action_id=_str(payload, "action_id"),
        predecessor_digest=_str(payload, "predecessor_digest"),
        backend=_str(payload, "backend"),
        targets=tuple(_fold_target_from_mapping(item) for item in _mappings(payload, "targets")),
        install_mode=_optional_str(payload, "install_mode"),
        orchestration_source_commit=_optional_str(payload, "orchestration_source_commit"),
    )


def _canonical_pair_entry_from_mapping(payload: Mapping[str, object]) -> FoldingCanonicalPairEntry:
    _reject_unknown(
        payload,
        {
            "schema_version",
            "target_id",
            "sequence_sha256",
            "model_entity_id",
            "tool_used",
            "structure_path",
            "scores_path",
        },
        "FoldingCanonicalPairEntry",
    )
    return FoldingCanonicalPairEntry(
        schema_version=_require_schema(payload, "FoldingCanonicalPairEntry"),
        target_id=_str(payload, "target_id"),
        sequence_sha256=_str(payload, "sequence_sha256"),
        model_entity_id=_str(payload, "model_entity_id"),
        tool_used=_str(payload, "tool_used"),
        structure_path=_str(payload, "structure_path"),
        scores_path=_str(payload, "scores_path"),
    )


def folding_canonical_pair_handoff_from_mapping(payload: Mapping[str, object]) -> FoldingCanonicalPairHandoff:
    _reject_unknown(
        payload,
        {*_IDENTITY_FIELDS, "index_path", "index_digest", "entries", "install_mode", "orchestration_source_commit"},
        "FoldingCanonicalPairHandoff",
    )
    return FoldingCanonicalPairHandoff(
        schema_version=_require_schema(payload, "FoldingCanonicalPairHandoff"),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        action_id=_str(payload, "action_id"),
        predecessor_digest=_str(payload, "predecessor_digest"),
        index_path=_str(payload, "index_path"),
        index_digest=_str(payload, "index_digest"),
        entries=tuple(_canonical_pair_entry_from_mapping(item) for item in _mappings(payload, "entries")),
        install_mode=_optional_str(payload, "install_mode"),
        orchestration_source_commit=_optional_str(payload, "orchestration_source_commit"),
    )


def folding_handoff_from_mapping(
    payload: Mapping[str, object],
) -> (
    FoldingMsaFlattenHandoff
    | FoldingSplitHandoff
    | FoldingPreprocessHandoff
    | FoldingFoldHandoff
    | FoldingCanonicalPairHandoff
):
    """Dispatch one folding handoff through its action id kind discriminator."""
    action_id = payload.get("action_id")
    if not isinstance(action_id, str) or not action_id:
        raise ValueError("folding handoff action_id must be a non-empty string")
    kind = folding_action_kind_for_action_id(action_id)
    if kind == "msa-flatten":
        return folding_msa_flatten_handoff_from_mapping(payload)
    if kind == "split":
        return folding_split_handoff_from_mapping(payload)
    if kind == "preprocess":
        return folding_preprocess_handoff_from_mapping(payload)
    if kind == "fold":
        return folding_fold_handoff_from_mapping(payload)
    return folding_canonical_pair_handoff_from_mapping(payload)


def _int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


__all__ = [
    "OPENFOLD_TRT_DEFERRED_MODEL_FN_ERROR",
    "FoldingBackendAssetsSnapshot",
    "FoldingCanonicalPairEntry",
    "FoldingCanonicalPairHandoff",
    "FoldingFoldHandoff",
    "FoldingFoldTarget",
    "FoldingMsaFlattenHandoff",
    "FoldingPreprocessHandoff",
    "FoldingPreprocessTarget",
    "FoldingSplitHandoff",
    "FoldingSplitTarget",
    "FoldingTargetIdentity",
    "folding_action_kind_for_action_id",
    "folding_backend_assets_snapshot_from_mapping",
    "folding_canonical_pair_handoff_from_mapping",
    "folding_fold_handoff_from_mapping",
    "folding_handoff_from_mapping",
    "folding_msa_flatten_handoff_from_mapping",
    "folding_preprocess_handoff_from_mapping",
    "folding_split_handoff_from_mapping",
    "folding_target_identity_from_mapping",
    "folding_target_sequence_sha256",
]
