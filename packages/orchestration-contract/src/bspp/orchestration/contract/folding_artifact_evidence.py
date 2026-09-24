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

"""Bounded artifact-backed folding evidence; legacy score-bearing types stay frozen.

These records describe authenticated original prediction files. They never
contain PAE values and perform no filesystem or scheduler operations.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from .folding_bioir import BIOIR_MONOMER_TOOL_USED, BIOIR_MULTIMER_TOOL_USED, BioIRModelPolicy
from .folding_execution import FoldingTargetIdentity, folding_target_identity_from_mapping
from .model_identity import normalize_model_entity_id

if TYPE_CHECKING:
    from .phase import FoldingPhaseRunSpec

ARTIFACT_EVIDENCE_PROFILE = "artifact-backed-v2"
MAX_ARTIFACT_TARGETS = 16_384
MAX_SCORE_BYTES = 1_073_741_824
MAX_METADATA_BYTES = 134_217_728
MAX_BUNDLE_BYTES = 268_435_456
MAX_FINALIZATION_INDEX_BYTES = 1_048_576
MAX_ARTIFACT_PATH_BYTES = 1_024
_SHA = re.compile(r"[0-9a-f]{64}")


def _keys(value: Mapping[str, object], keys: set[str], optional: set[str] | None = None) -> None:
    if keys - set(value) or set(value) - keys - (optional or set()):
        raise ValueError("artifact evidence fields differ from the closed schema")


def _text(value: object, name: str, limit: int = MAX_ARTIFACT_PATH_BYTES) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) > limit or any(ord(c) < 32 for c in value):
        raise ValueError(f"invalid artifact evidence {name}")
    return value


def _sha(value: object) -> str:
    result = _text(value, "SHA-256", 64)
    if _SHA.fullmatch(result) is None:
        raise ValueError("artifact evidence requires a lowercase SHA-256")
    return result


def _positive(value: object, name: str, maximum: int = MAX_SCORE_BYTES) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"invalid artifact evidence {name}")
    return value


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("artifact evidence requires a string-keyed object")
    return value


def artifact_path(value: object) -> str:
    path = _text(value, "path")
    if not path.startswith("/") or "\\" in path or str(PurePosixPath(path)) != path or ".." in path.split("/"):
        raise ValueError("artifact evidence path must be normalized and absolute")
    return path


def model_metadata(policy: BioIRModelPolicy, chain_count: int) -> dict[str, object]:
    """The same sealed model identity for Runtime emission and Control validation."""
    source = policy.model_source_for_chain_count(chain_count)
    monomer = source == policy.monomer_model_source
    return {
        "tool_used": BIOIR_MONOMER_TOOL_USED if monomer else BIOIR_MULTIMER_TOOL_USED,
        "model_source": source,
        "checkpoint_sha256": policy.monomer_checkpoint_sha256 if monomer else policy.multimer_checkpoint_sha256,
        "checkpoint_size_bytes": policy.monomer_checkpoint_size_bytes
        if monomer
        else policy.multimer_checkpoint_size_bytes,
        "bioir_model_policy_digest": policy.digest,
    }


@dataclass(frozen=True)
class PredictionArtifactIdentity:
    role: str
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"structure", "scores"}:
            raise ValueError("unknown prediction artifact role")
        artifact_path(self.path)
        _positive(self.size_bytes, "artifact size")
        _sha(self.sha256)

    def to_mapping(self) -> dict[str, object]:
        return {"role": self.role, "path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PredictionArtifactIdentity:
        _keys(value, {"role", "path", "size_bytes", "sha256"})
        return cls(
            _text(value["role"], "role"),
            artifact_path(value["path"]),
            _positive(value["size_bytes"], "artifact size"),
            _sha(value["sha256"]),
        )


@dataclass(frozen=True)
class ArtifactFoldTarget:
    target: FoldingTargetIdentity
    structure: PredictionArtifactIdentity
    scores: PredictionArtifactIdentity
    tool_used: str
    model_source: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    model_policy_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.target, FoldingTargetIdentity):
            raise ValueError("artifact fold target requires a typed identity")
        _text(self.target.target_id, "target ID")
        _text(self.target.description, "target description")
        if (
            self.structure.role != "structure"
            or self.scores.role != "scores"
            or self.structure.path == self.scores.path
        ):
            raise ValueError("artifact pair must have distinct structure and score identities")
        if (PurePosixPath(self.structure.path).name, PurePosixPath(self.scores.path).name) != (
            f"{self.model_entity_id}-model_v1.pdb",
            f"{self.model_entity_id}-meta_v1.json",
        ):
            raise ValueError("artifact prediction basenames do not match their canonical target identity")
        if self.model_source not in {"openfold2_ptm_1", "alphafold2_multimer_1"}:
            raise ValueError("unsupported artifact model source")
        expected = BIOIR_MONOMER_TOOL_USED if self.model_source == "openfold2_ptm_1" else BIOIR_MULTIMER_TOOL_USED
        if self.tool_used != expected:
            raise ValueError("artifact model source and tool disagree")
        _sha(self.checkpoint_sha256)
        _sha(self.model_policy_digest)
        _positive(self.checkpoint_size_bytes, "checkpoint size")

    @property
    def model_entity_id(self) -> str:
        return normalize_model_entity_id(self.target.target_id)

    @property
    def residue_count(self) -> int:
        return sum(map(len, self.target.chains))

    @property
    def model_metadata(self) -> dict[str, object]:
        return {
            "tool_used": self.tool_used,
            "model_source": self.model_source,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "bioir_model_policy_digest": self.model_policy_digest,
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "target": self.target.to_mapping(),
            "pair_reference": {
                "model_entity_id": self.model_entity_id,
                "tool_used": self.tool_used,
                "artifacts": [self.structure.to_mapping(), self.scores.to_mapping()],
                "model_metadata": self.model_metadata,
                "content_validation": {
                    "profile": "prediction-scores-full-pae-v1",
                    "outcome": "passed",
                    "residue_count": self.residue_count,
                    "plddt_count": self.residue_count,
                    "pae_rows": self.residue_count,
                    "pae_columns": self.residue_count,
                },
            },
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ArtifactFoldTarget:
        _keys(value, {"target", "pair_reference"})
        target = folding_target_identity_from_mapping(_mapping(value["target"]))
        pair = _mapping(value["pair_reference"])
        _keys(pair, {"model_entity_id", "tool_used", "artifacts", "model_metadata", "content_validation"})
        witness = _mapping(pair["content_validation"])
        _keys(witness, {"profile", "outcome", "residue_count", "plddt_count", "pae_rows", "pae_columns"})
        for name in ("residue_count", "plddt_count", "pae_rows", "pae_columns"):
            _positive(witness[name], name)
        artifacts = pair["artifacts"]
        if not isinstance(artifacts, list) or len(artifacts) != 2:
            raise ValueError("artifact pair must contain exactly two artifacts")
        meta = _mapping(pair["model_metadata"])
        _keys(
            meta,
            {"tool_used", "model_source", "checkpoint_sha256", "checkpoint_size_bytes", "bioir_model_policy_digest"},
        )
        result = cls(
            target,
            PredictionArtifactIdentity.from_mapping(_mapping(artifacts[0])),
            PredictionArtifactIdentity.from_mapping(_mapping(artifacts[1])),
            _text(meta["tool_used"], "tool"),
            _text(meta["model_source"], "model source"),
            _sha(meta["checkpoint_sha256"]),
            _positive(meta["checkpoint_size_bytes"], "checkpoint size"),
            _sha(meta["bioir_model_policy_digest"]),
        )
        # Equality includes every dimension and fixed validation discriminator;
        # no extra arrays/fields can be hidden in this bounded metadata record.
        if result.to_mapping() != dict(value):
            raise ValueError("artifact pair identity or content-validation dimensions disagree")
        return result


@dataclass(frozen=True)
class ArtifactFoldEvidence:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_id: str
    action_digest: str
    predecessor_digest: str
    shard_projection_sha256: str
    entries: tuple[ArtifactFoldTarget, ...]
    install_mode: str | None = None
    orchestration_source_commit: str | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"phase-run-[0-9a-f]{32}", self.phase_run_id) is None:
            raise ValueError("invalid artifact evidence Phase")
        if re.fullmatch(r"attempt-[0-9]{4}", self.attempt_id) is None:
            raise ValueError("invalid artifact evidence Attempt")
        if re.fullmatch(r"(?:fold|canonical-pair)-[0-9]{6}", self.action_id) is None:
            raise ValueError("invalid artifact evidence action")
        for digest in (
            self.phase_runspec_digest,
            self.action_digest,
            self.predecessor_digest,
            self.shard_projection_sha256,
        ):
            _sha(digest)
        if not isinstance(self.entries, tuple) or not 0 < len(self.entries) <= MAX_ARTIFACT_TARGETS:
            raise ValueError("artifact evidence target count exceeds fixed bounds")
        if any(not isinstance(entry, ArtifactFoldTarget) for entry in self.entries):
            raise ValueError("artifact evidence entries must be typed")
        ids = [entry.target.target_id for entry in self.entries]
        models = [entry.model_entity_id for entry in self.entries]
        paths = [artifact.path for entry in self.entries for artifact in (entry.structure, entry.scores)]
        if len(set(ids)) != len(ids) or len(set(models)) != len(models) or len(set(paths)) != len(paths):
            raise ValueError("artifact evidence contains duplicate target, model or artifact paths")
        if self.install_mode is not None and self.install_mode not in {"baked", "override"}:
            raise ValueError("invalid artifact evidence install mode")
        if (
            self.orchestration_source_commit is not None
            and re.fullmatch(r"[0-9a-f]{40}", self.orchestration_source_commit) is None
        ):
            raise ValueError("artifact evidence source identity must be a full commit hash")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "evidence_profile": ARTIFACT_EVIDENCE_PROFILE,
            "backend": "bioir",
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_id": self.action_id,
            "action_digest": self.action_digest,
            "predecessor_digest": self.predecessor_digest,
            "shard_projection_sha256": self.shard_projection_sha256,
            "entries": [entry.to_mapping() for entry in self.entries],
        }
        if self.install_mode is not None:
            result["install_mode"] = self.install_mode
        if self.orchestration_source_commit is not None:
            result["orchestration_source_commit"] = self.orchestration_source_commit
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ArtifactFoldEvidence:
        _keys(
            value,
            {
                "evidence_profile",
                "backend",
                "phase_run_id",
                "attempt_id",
                "phase_runspec_digest",
                "action_id",
                "action_digest",
                "predecessor_digest",
                "shard_projection_sha256",
                "entries",
            },
            {"install_mode", "orchestration_source_commit"},
        )
        if value["evidence_profile"] != ARTIFACT_EVIDENCE_PROFILE or value["backend"] != "bioir":
            raise ValueError("unsupported artifact evidence profile or backend")
        entries = value["entries"]
        if not isinstance(entries, list) or not 0 < len(entries) <= MAX_ARTIFACT_TARGETS:
            raise ValueError("artifact evidence target count exceeds fixed bounds")
        return cls(
            phase_run_id=_text(value["phase_run_id"], "Phase"),
            attempt_id=_text(value["attempt_id"], "Attempt"),
            phase_runspec_digest=_sha(value["phase_runspec_digest"]),
            action_id=_text(value["action_id"], "action ID"),
            action_digest=_sha(value["action_digest"]),
            predecessor_digest=_sha(value["predecessor_digest"]),
            shard_projection_sha256=_sha(value["shard_projection_sha256"]),
            entries=tuple(ArtifactFoldTarget.from_mapping(_mapping(entry)) for entry in entries),
            install_mode=_text(value["install_mode"], "install mode") if "install_mode" in value else None,
            orchestration_source_commit=_text(value["orchestration_source_commit"], "source commit", 64)
            if "orchestration_source_commit" in value
            else None,
        )

    def validate_binding(self, runspec: FoldingPhaseRunSpec, *, actions_root: PurePosixPath) -> None:
        from .phase import canonical_mapping_digest

        policy = runspec.payload.bioir_model_policy
        projection = runspec.payload.fold_shard_projection
        if runspec.payload.evidence_profile != ARTIFACT_EVIDENCE_PROFILE or policy is None or projection is None:
            raise ValueError("artifact evidence requires its sealed profile, policy and shard projection")
        actions = {action.action_id: action for action in runspec.payload.actions}
        action = actions.get(self.action_id)
        if action is None or self.phase_run_id != runspec.phase_run_id or self.attempt_id != runspec.attempt_id:
            raise ValueError("artifact evidence authority mismatch")
        if self.phase_runspec_digest != runspec.digest or self.action_digest != canonical_mapping_digest(
            action.to_mapping()
        ):
            raise ValueError("artifact evidence RunSpec/action digest mismatch")
        if self.shard_projection_sha256 != projection.sha256:
            raise ValueError("artifact evidence shard projection mismatch")
        fold_action = next(item for item in actions.values() if item.action_kind == "fold")
        fold_root = actions_root / fold_action.action_id
        members = runspec.payload.msa_set.member_a3m_paths
        expected = tuple(normalize_model_entity_id(PurePosixPath(member).stem) for member in members)
        if tuple(entry.target.target_id for entry in self.entries) != expected:
            raise ValueError("artifact evidence does not cover exact ordered MSA targets")
        manifest = runspec.payload.msa_set_manifest
        if manifest is None or manifest.member_lengths is None:
            raise ValueError("artifact evidence requires attested member lengths")
        if tuple(entry.residue_count for entry in self.entries) != manifest.member_lengths:
            raise ValueError("artifact evidence lengths differ from attested MSA members")
        for entry, member in zip(self.entries, members, strict=True):
            if entry.target.description != member or entry.model_metadata != model_metadata(
                policy, len(entry.target.chains)
            ):
                raise ValueError("artifact target description or model policy mismatch")
            for artifact in (entry.structure, entry.scores):
                if not PurePosixPath(artifact.path).is_relative_to(fold_root):
                    raise ValueError("artifact path escapes current Attempt fold root")


def finalization_member_paths(fold_action_id: str, canonical_action_id: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                f"{fold_action_id}/handoff.json",
                f"{canonical_action_id}/handoff.json",
                f"{canonical_action_id}/canonical-pair-index.json",
                f"{canonical_action_id}/action-evidence.json",
            )
        )
    )


@dataclass(frozen=True)
class FoldingMetadataMember:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _text(self.path, "metadata member path")
        if (
            re.fullmatch(
                r"(?:fold|canonical-pair)-[0-9]{6}/(?:handoff|canonical-pair-index|action-evidence)\.json", self.path
            )
            is None
        ):
            raise ValueError("invalid indexed folding metadata path")
        _positive(self.size_bytes, "metadata member size", MAX_METADATA_BYTES)
        _sha(self.sha256)

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FoldingMetadataMember:
        _keys(value, {"path", "size_bytes", "sha256"})
        return cls(
            _text(value["path"], "metadata member path"),
            _positive(value["size_bytes"], "metadata member size", MAX_METADATA_BYTES),
            _sha(value["sha256"]),
        )


@dataclass(frozen=True)
class FoldingFinalizationIndex:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    fold_action_id: str
    fold_action_digest: str
    canonical_action_id: str
    canonical_action_digest: str
    orchestration_source_commit: str
    members: tuple[FoldingMetadataMember, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"phase-run-[0-9a-f]{32}", self.phase_run_id) is None:
            raise ValueError("invalid folding finalization Phase")
        if re.fullmatch(r"attempt-[0-9]{4}", self.attempt_id) is None:
            raise ValueError("invalid folding finalization Attempt")
        if (
            re.fullmatch(r"fold-[0-9]{6}", self.fold_action_id) is None
            or re.fullmatch(r"canonical-pair-[0-9]{6}", self.canonical_action_id) is None
        ):
            raise ValueError("invalid folding finalization action IDs")
        for digest in (self.phase_runspec_digest, self.fold_action_digest, self.canonical_action_digest):
            _sha(digest)
        if re.fullmatch(r"[0-9a-f]{40}", self.orchestration_source_commit) is None:
            raise ValueError("folding finalization requires the exact source commit")
        if (
            not isinstance(self.members, tuple)
            or len(self.members) != 4
            or any(not isinstance(member, FoldingMetadataMember) for member in self.members)
        ):
            raise ValueError("folding finalization index requires exactly four typed members")
        if tuple(member.path for member in self.members) != finalization_member_paths(
            self.fold_action_id, self.canonical_action_id
        ):
            raise ValueError("folding finalization index differs from its exact member set")
        if sum(member.size_bytes for member in self.members) + MAX_FINALIZATION_INDEX_BYTES > MAX_BUNDLE_BYTES:
            raise ValueError("folding finalization bundle exceeds fixed aggregate bound")

    def to_mapping(self) -> dict[str, object]:
        return {
            "evidence_profile": ARTIFACT_EVIDENCE_PROFILE,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "fold_action_id": self.fold_action_id,
            "fold_action_digest": self.fold_action_digest,
            "canonical_action_id": self.canonical_action_id,
            "canonical_action_digest": self.canonical_action_digest,
            "orchestration_source_commit": self.orchestration_source_commit,
            "members": [member.to_mapping() for member in self.members],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FoldingFinalizationIndex:
        _keys(
            value,
            {
                "evidence_profile",
                "phase_run_id",
                "attempt_id",
                "phase_runspec_digest",
                "fold_action_id",
                "fold_action_digest",
                "canonical_action_id",
                "canonical_action_digest",
                "orchestration_source_commit",
                "members",
            },
        )
        if value["evidence_profile"] != ARTIFACT_EVIDENCE_PROFILE:
            raise ValueError("unsupported folding finalization index profile")
        members = value["members"]
        if not isinstance(members, list) or len(members) != 4:
            raise ValueError("folding finalization index requires exactly four members")
        return cls(
            phase_run_id=_text(value["phase_run_id"], "Phase"),
            attempt_id=_text(value["attempt_id"], "Attempt"),
            phase_runspec_digest=_sha(value["phase_runspec_digest"]),
            fold_action_id=_text(value["fold_action_id"], "fold action"),
            fold_action_digest=_sha(value["fold_action_digest"]),
            canonical_action_id=_text(value["canonical_action_id"], "canonical action"),
            canonical_action_digest=_sha(value["canonical_action_digest"]),
            orchestration_source_commit=_text(value["orchestration_source_commit"], "source commit"),
            members=tuple(FoldingMetadataMember.from_mapping(_mapping(member)) for member in members),
        )

    def validate_binding(self, runspec: FoldingPhaseRunSpec) -> None:
        from .phase import canonical_mapping_digest

        if runspec.payload.evidence_profile != ARTIFACT_EVIDENCE_PROFILE:
            raise ValueError("folding finalization index requires sealed artifact evidence profile")
        if (self.phase_run_id, self.attempt_id, self.phase_runspec_digest) != (
            runspec.phase_run_id,
            runspec.attempt_id,
            runspec.digest,
        ):
            raise ValueError("folding finalization index authority mismatch")
        for kind, action_id, digest in (
            ("fold", self.fold_action_id, self.fold_action_digest),
            ("canonical-pair", self.canonical_action_id, self.canonical_action_digest),
        ):
            action = next(item for item in runspec.payload.actions if item.action_kind == kind)
            if action.action_id != action_id or canonical_mapping_digest(action.to_mapping()) != digest:
                raise ValueError("folding finalization index action binding mismatch")
