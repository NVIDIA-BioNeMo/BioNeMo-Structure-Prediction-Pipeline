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

"""Runtime producers for postprocessing input and scientific-output authority."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_artifacts import (
    PostprocessingEvidenceArtifact,
    PostprocessingScientificOutputInventory,
    postprocessing_scientific_output_inventory_from_mapping,
)
from bspp.orchestration.contract.postprocessing_attestations import (
    PostprocessingRuntimeInputAttestation,
    PostprocessingRuntimeInputAttestationSet,
    PostprocessingRuntimeQualificationAttestation,
    postprocessing_runtime_input_attestation_set_from_mapping,
)
from bspp.orchestration.contract.postprocessing_logical_identity import (
    PhysicalInputLocator,
)
from bspp.orchestration.contract.postprocessing_runspec import (
    ExecutablePostprocessingPhaseRunSpec,
    postprocessing_phase_runspec_from_mapping,
)
from bspp.orchestration.contract.postprocessing_tar_inventory import resolve_tar_manifest_member
from bspp.orchestration.runtime.data_movement.s3.client import MissingS3CredentialsError, load_credentials_from_env
from bspp.orchestration.runtime.data_movement.s3.inventory import build_list_prefix_argv

ProbeRunner = Callable[[str], Mapping[str, object]]


def attest_runtime_inputs(
    *,
    phase_runspec_path: Path,
    qualification_path: Path,
    evidence_root: Path,
    output_path: Path,
    observed_at: str | None = None,
    probe_runner: ProbeRunner | None = None,
) -> PostprocessingRuntimeInputAttestationSet:
    """Create once exact scientific-input and attempt-Runtime attestations."""
    runspec = _load_runspec(phase_runspec_path)
    qualification_document = _stable_regular_bytes(qualification_path, label="Runtime Qualification")
    qualified = runspec.payload.qualified_runtime
    if (
        hashlib.sha256(qualification_document).hexdigest() != qualified.qualification_sha256
        or len(qualification_document) != qualified.qualification_size_bytes
    ):
        raise ValueError("Runtime Qualification bytes differ from the exact qualified Runtime selection")
    physical = {item.name: item for item in runspec.payload.physical_inputs}
    if output_path.exists():
        existing = postprocessing_runtime_input_attestation_set_from_mapping(
            _json_mapping(output_path.read_bytes(), label="runtime input attestation set")
        )
        if (
            existing.phase_run_id,
            existing.attempt_id,
            existing.phase_runspec_digest,
        ) != (runspec.phase_run_id, runspec.attempt_id, runspec.digest):
            raise ValueError("existing runtime input attestations differ from the current RunSpec")
        _reconcile_existing_runtime_input_attestations(
            existing,
            runspec=runspec,
            physical=physical,
            evidence_root=evidence_root,
        )
        return existing
    moment = observed_at or _timestamp()
    qualification_attestation = PostprocessingRuntimeQualificationAttestation(
        qualified_runtime_digest=qualified.digest,
        record_location=qualified.qualification_location,
        record_sha256=qualified.qualification_sha256,
        record_size_bytes=qualified.qualification_size_bytes,
        tuple_id=qualified.tuple_id,
        source_identity_digest=qualified.source_identity_digest,
        source_package_identity_digest=qualified.source_package_identity_digest,
        toolkit_identity_digest=qualified.toolkit_identity_digest,
        runtime_component_identity_digest=qualified.runtime_component_identity_digest,
        observed_at=moment,
    )
    attestations: list[PostprocessingRuntimeInputAttestation] = []
    for entry in runspec.payload.logical_inputs.entries:
        locator = physical.get(entry.name)
        if locator is None:
            raise ValueError(f"logical input has no exact physical locator: {entry.name!r}")
        proof = dict((probe_runner or _probe_locator)(locator.locator))
        proof.update(
            {
                "schema_version": 1,
                "proof_kind": "postprocessing-runtime-input-accessibility-v1",
                "logical_input_name": entry.name,
                "member_identity": entry.member_identity,
                "physical_input_name": entry.name,
                "physical_locator": locator.locator,
                "observed_at": moment,
                "accessible": True,
            }
        )
        proof_relative = f"phase-inputs/proofs/{entry.name}.json"
        proof_bytes = _canonical_bytes(proof)
        _write_create_once(evidence_root / proof_relative, proof_bytes)
        attestations.append(
            PostprocessingRuntimeInputAttestation(
                logical_input_name=entry.name,
                verification_kind="authority-declared-content-v1",
                authority=entry.member_identity,
                content_sha256=entry.expected_content_sha256,
                size_bytes=entry.expected_size_bytes,
                verification_source="runtime-preflight",
                observed_at=moment,
                physical_input_name=entry.name,
                physical_locator=locator.locator,
                member_identity=entry.member_identity,
                accessibility_evidence_path=proof_relative,
                accessibility_evidence_sha256=hashlib.sha256(proof_bytes).hexdigest(),
                accessibility_evidence_size_bytes=len(proof_bytes),
            )
        )
    result = PostprocessingRuntimeInputAttestationSet(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        runtime_qualification=qualification_attestation,
        attestations=tuple(sorted(attestations, key=lambda item: item.logical_input_name)),
    )
    _write_create_once(output_path, _canonical_bytes(result.to_mapping()))
    return result


def _reconcile_existing_runtime_input_attestations(
    existing: PostprocessingRuntimeInputAttestationSet,
    *,
    runspec: ExecutablePostprocessingPhaseRunSpec,
    physical: Mapping[str, PhysicalInputLocator],
    evidence_root: Path,
) -> None:
    qualified = runspec.payload.qualified_runtime
    attested = existing.runtime_qualification
    if (
        attested.qualified_runtime_digest,
        attested.record_location,
        attested.record_sha256,
        attested.record_size_bytes,
        attested.tuple_id,
        attested.source_identity_digest,
        attested.source_package_identity_digest,
        attested.toolkit_identity_digest,
        attested.runtime_component_identity_digest,
    ) != (
        qualified.digest,
        qualified.qualification_location,
        qualified.qualification_sha256,
        qualified.qualification_size_bytes,
        qualified.tuple_id,
        qualified.source_identity_digest,
        qualified.source_package_identity_digest,
        qualified.toolkit_identity_digest,
        qualified.runtime_component_identity_digest,
    ):
        raise ValueError("existing Runtime Qualification attestation differs from the current RunSpec")
    expected_logical = {item.name: item for item in runspec.payload.logical_inputs.entries}
    observed = {item.logical_input_name: item for item in existing.attestations}
    if set(observed) != set(expected_logical):
        raise ValueError("existing runtime input attestations do not cover the current logical inputs")
    for name, logical in expected_logical.items():
        locator = physical.get(name)
        if locator is None:
            raise ValueError(f"logical input has no exact physical locator: {name!r}")
        item = observed[name]
        if (
            item.verification_kind,
            item.authority,
            item.content_sha256,
            item.size_bytes,
            item.verification_source,
            item.physical_input_name,
            item.physical_locator,
            item.member_identity,
        ) != (
            "authority-declared-content-v1",
            logical.member_identity,
            logical.expected_content_sha256,
            logical.expected_size_bytes,
            "runtime-preflight",
            name,
            locator.locator,
            logical.member_identity,
        ):
            raise ValueError(f"existing runtime input attestation differs for {name!r}")
        if (
            item.accessibility_evidence_path is None
            or item.accessibility_evidence_sha256 is None
            or item.accessibility_evidence_size_bytes is None
        ):
            raise ValueError(f"existing runtime input attestation proof is incomplete for {name!r}")
        proof = _stable_regular_bytes(
            evidence_root / item.accessibility_evidence_path,
            label=f"runtime input attestation proof {name!r}",
        )
        if (
            hashlib.sha256(proof).hexdigest() != item.accessibility_evidence_sha256
            or len(proof) != item.accessibility_evidence_size_bytes
        ):
            raise ValueError(f"existing runtime input attestation proof differs for {name!r}")


def inventory_scientific_outputs(
    *,
    phase_runspec_path: Path,
    output_path: Path,
) -> PostprocessingScientificOutputInventory:
    """Create a full output inventory without rehashing tar payloads."""
    runspec = _load_runspec(phase_runspec_path)
    output_root = Path(runspec.payload.attempt_paths.output_dir)
    evidence_root = Path(runspec.payload.attempt_paths.evidence_dir)
    if output_path.exists():
        existing = postprocessing_scientific_output_inventory_from_mapping(
            _json_mapping(output_path.read_bytes(), label="scientific output inventory")
        )
        if (
            existing.phase_run_id,
            existing.attempt_id,
            existing.phase_runspec_digest,
            Path(existing.output_root),
        ) != (runspec.phase_run_id, runspec.attempt_id, runspec.digest, output_root):
            raise ValueError("existing scientific output inventory differs from the current RunSpec")
        return existing
    if not output_root.is_absolute() or output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("scientific output root must be an existing absolute non-symlink directory")
    tar_sources = _tar_source_rows(output_root)
    members: list[PostprocessingEvidenceArtifact] = []
    for path in _regular_output_files(output_root, evidence_root):
        relative = path.relative_to(output_root).as_posix()
        source = tar_sources.get(relative)
        if source is None:
            document = path.read_bytes()
            members.append(
                PostprocessingEvidenceArtifact(
                    path=relative,
                    sha256=hashlib.sha256(document).hexdigest(),
                    size_bytes=len(document),
                )
            )
            continue
        source_path, source_record = source
        members.append(
            PostprocessingEvidenceArtifact(
                path=relative,
                sha256=_metadata_record_digest(relative, source_path, source_record),
                size_bytes=path.stat().st_size,
                verification_kind="inventory-metadata-v1",
                verification_source_path=source_path,
            )
        )
    result = PostprocessingScientificOutputInventory(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        output_root=str(output_root),
        members=tuple(sorted(members, key=lambda item: item.path)),
    )
    _write_create_once(output_path, _canonical_bytes(result.to_mapping()))
    return result


def _tar_source_rows(output_root: Path) -> dict[str, tuple[str, dict[str, str]]]:
    result: dict[str, tuple[str, dict[str, str]]] = {}
    for source in sorted(output_root.rglob("*.csv")):
        if source.is_symlink() or not source.is_file():
            continue
        source_relative = source.relative_to(output_root).as_posix()
        try:
            with source.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    continue
                rows = tuple(reader)
        except UnicodeDecodeError:
            continue
        for raw_row in rows:
            row = {key: value or "" for key, value in raw_row.items() if key is not None}
            locator = row.get("tar_path") or row.get("tar_name") or row.get("path") or row.get("relative_path")
            if not locator:
                continue
            try:
                resolved, member = resolve_tar_manifest_member(
                    row,
                    manifest_path=source,
                    local_tar_dir=output_root / "local_tars",
                )
                resolved.resolve().relative_to(output_root.resolve())
            except (FileNotFoundError, ValueError):
                continue
            if not member.endswith(
                (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tbz2", ".tar.xz", ".txz", ".tar.zst", ".tzst")
            ):
                continue
            if member in result:
                raise ValueError(f"multiple scientific inventory rows identify tar member {member!r}")
            result[member] = (source_relative, row)
    return result


def _regular_output_files(output_root: Path, evidence_root: Path) -> tuple[Path, ...]:
    resolved_output = output_root.resolve(strict=True)
    resolved_evidence = evidence_root.resolve(strict=True) if evidence_root.exists() else evidence_root.resolve()
    result: list[Path] = []
    for directory, names, filenames in os.walk(resolved_output, followlinks=False):
        root = Path(directory)
        if root == resolved_evidence or resolved_evidence in root.parents:
            names[:] = []
            continue
        for name in tuple(names):
            child = root / name
            if child.is_symlink():
                raise ValueError(f"scientific output tree contains a symlink: {child}")
            if child.resolve() == resolved_evidence:
                names.remove(name)
        for name in filenames:
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"scientific output member is not a regular file: {path}")
            result.append(path)
    if not result:
        raise ValueError("scientific output inventory requires at least one output member")
    return tuple(sorted(result))


def _probe_locator(locator: str) -> Mapping[str, object]:
    if "://" not in locator or locator.startswith("file://"):
        path = Path(locator.removeprefix("file://"))
        if path.is_symlink() or not path.exists():
            raise ValueError(f"runtime input locator is inaccessible: {locator}")
        stat = path.stat()
        return {
            "access_kind": "local-filesystem-stat-v1",
            "file_kind": "directory" if path.is_dir() else "regular-file" if path.is_file() else "other",
            "observed_size_bytes": stat.st_size,
        }
    try:
        credentials = load_credentials_from_env()
    except MissingS3CredentialsError as exc:
        raise ValueError("runtime object input credentials are unavailable") from exc
    try:
        completed = subprocess.run(
            build_list_prefix_argv(locator, credentials=credentials),
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
            env={**os.environ, **credentials.as_env()},
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"runtime object input locator probe timed out: {locator}") from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ValueError(f"runtime object input locator is inaccessible: {locator}")
    return {
        "access_kind": "s5cmd-listing-v1",
        "listing_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "listing_line_count": len(completed.stdout.splitlines()),
    }


def _metadata_record_digest(member_path: str, source_path: str, source_record: Mapping[str, str]) -> str:
    return canonical_mapping_digest(
        {
            "schema_version": 1,
            "identity_kind": "inventory-metadata-v1",
            "member_path": member_path,
            "verification_source_path": source_path,
            "source_record": dict(sorted(source_record.items())),
        }
    )


def _load_runspec(path: Path) -> ExecutablePostprocessingPhaseRunSpec:
    return postprocessing_phase_runspec_from_mapping(
        _json_mapping(_stable_regular_bytes(path, label="postprocessing RunSpec"), label="postprocessing RunSpec")
    )


def _stable_regular_bytes(path: Path, *, label: str) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_stat = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or _stat_signature(before) != _stat_signature(after)
        or _stat_signature(before) != _stat_signature(path_stat)
    ):
        raise ValueError(f"{label} changed while being read")
    document = b"".join(chunks)
    if len(document) != before.st_size:
        raise ValueError(f"{label} size changed while being read")
    return document


def _stat_signature(item: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _json_mapping(document: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(document)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must be a JSON mapping")
    return payload


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def _write_create_once(path: Path, document: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != document:
                raise ValueError(f"existing postprocessing runtime authority differs: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inputs = subparsers.add_parser("attest-inputs")
    inputs.add_argument("--phase-runspec", type=Path, required=True)
    inputs.add_argument("--runtime-qualification", type=Path, required=True)
    inputs.add_argument("--evidence-root", type=Path, required=True)
    inputs.add_argument("--output", type=Path, required=True)
    outputs = subparsers.add_parser("inventory-outputs")
    outputs.add_argument("--phase-runspec", type=Path, required=True)
    outputs.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "attest-inputs":
        attest_runtime_inputs(
            phase_runspec_path=args.phase_runspec,
            qualification_path=args.runtime_qualification,
            evidence_root=args.evidence_root,
            output_path=args.output,
        )
    else:
        inventory_scientific_outputs(phase_runspec_path=args.phase_runspec, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["attest_runtime_inputs", "inventory_scientific_outputs", "main"]
