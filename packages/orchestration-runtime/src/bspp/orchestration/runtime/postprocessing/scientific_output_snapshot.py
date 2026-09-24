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

"""Runtime-only scientific roots and atomic postprocessing finalization bundles.

Large scientific tar payloads never enter the bounded handoff.  Runtime streams
their regular-file members into content manifests, hashes ordinary small output
files, verifies all prior Runtime and acceptance evidence, and atomically
publishes the JSON-only Action 09 handoff.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import stat
import tarfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, cast

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_bundle_manifest import (
    PostprocessingScientificOutputRoot,
    PostprocessingSmallOutputIdentity,
    PostprocessingTarManifest,
    PostprocessingTarManifestReference,
    PostprocessingTarMemberIdentity,
    postprocessing_scientific_output_root_from_mapping,
    postprocessing_tar_manifest_from_mapping,
)
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_ACTION_IDS
from bspp.orchestration.contract.postprocessing_runspec import (
    ExecutablePostprocessingPhaseRunSpec,
)
from bspp.orchestration.contract.postprocessing_tar_inventory import resolve_tar_manifest_member
from bspp.orchestration.contract.postprocessing_transfer_limits import (
    POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from bspp.orchestration.runtime.postprocessing.finalization_io import (
    _absolute_directory,
    _assert_stable_stat,
    _canonical_bytes,
    _canonical_mapping,
    _load_runspec,
    _regular_nonsymlink_file,
    _stable_file_bytes,
    _stable_identity,
    _StableFile,
    _verify_source_files,
    _write_create_once,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TAR_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".tar.zst",
    ".tzst",
)
_ACCEPTANCE_STEPS = (
    "acceptance-tar-payload-parity",
    "acceptance-semantic",
    "acceptance-verify-evidence",
)
_CAPTURE_DESTINATIONS = {step: f"acceptance/captures/{step}.json" for step in _ACCEPTANCE_STEPS}
_REPORT_DESTINATIONS = {
    "acceptance-tar-payload-parity": "acceptance/reports/tar-payload-parity-report.json",
    "acceptance-semantic": "acceptance/reports/semantic-acceptance-summary.json",
    "acceptance-verify-evidence": "acceptance/reports/acceptance-evidence-report.json",
}
_ACTION09_ID = POSTPROCESSING_ACTION_IDS["acceptance-adjudication"]


class ScientificSnapshotIO(Protocol):
    """Typed observation seam for stable-file and tar streaming operations."""

    def fstat(self, descriptor: int) -> os.stat_result: ...

    def open_tar(self, *, fileobj: BinaryIO, mode: Literal["r|*"]) -> tarfile.TarFile: ...


@dataclass(frozen=True)
class LocalScientificSnapshotIO:
    def fstat(self, descriptor: int) -> os.stat_result:
        return os.fstat(descriptor)

    def open_tar(self, *, fileobj: BinaryIO, mode: Literal["r|*"]) -> tarfile.TarFile:
        return tarfile.open(fileobj=fileobj, mode=mode)


LOCAL_SCIENTIFIC_SNAPSHOT_IO = LocalScientificSnapshotIO()


@dataclass(frozen=True)
class _ScientificDocuments:
    root: PostprocessingScientificOutputRoot
    documents: Mapping[str, bytes]
    manifests: tuple[PostprocessingTarManifest, ...]
    source_files: tuple[_StableFile, ...]


def generate_scientific_output_root(
    *,
    phase_runspec_path: Path,
    workers: int = 1,
    snapshot_io: ScientificSnapshotIO = LOCAL_SCIENTIFIC_SNAPSHOT_IO,
) -> PostprocessingScientificOutputRoot:
    """Create once the Runtime scientific root and its content-addressed tar manifests."""
    runspec = _load_runspec(phase_runspec_path)
    scientific = _build_scientific_documents(runspec, workers=workers, snapshot_io=snapshot_io)
    evidence_root = _absolute_directory(Path(runspec.payload.attempt_paths.evidence_dir), "evidence root", create=True)
    phase_output = evidence_root / "phase-output"
    for relative, document in sorted(scientific.documents.items()):
        source_relative = relative.removeprefix("outputs/")
        _write_create_once(phase_output / source_relative, document)
    _verify_source_files(scientific.source_files)
    _write_create_once(
        phase_output / "source-fingerprints.json",
        _scientific_source_fingerprints(runspec, scientific.source_files),
    )
    return scientific.root


def _build_scientific_documents(
    runspec: ExecutablePostprocessingPhaseRunSpec,
    *,
    workers: int,
    snapshot_io: ScientificSnapshotIO,
) -> _ScientificDocuments:
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1 or workers > 256:
        raise ValueError("postprocessing tar manifest workers must be between 1 and 256")
    output_root = _absolute_directory(Path(runspec.payload.attempt_paths.output_dir), "output root", create=False)
    evidence_root = Path(runspec.payload.attempt_paths.evidence_dir)
    tar_paths = _inventory_tar_paths(output_root)
    ordered_tars = tuple(sorted(tar_paths.items()))
    with ThreadPoolExecutor(max_workers=min(workers, len(ordered_tars))) as executor:
        manifests = tuple(
            executor.map(
                lambda item: _manifest_tar(item[1], item[0], snapshot_io=snapshot_io),
                ordered_tars,
            )
        )
    total_members = sum(len(manifest.members) for manifest in manifests)
    if total_members > POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.max_total_tar_members:
        raise ValueError("postprocessing tar manifests exceed the total member limit")

    tar_relative_paths = set(tar_paths)
    small_outputs: list[PostprocessingSmallOutputIdentity] = []
    stable_files: list[_StableFile] = []
    for path in _regular_output_files(output_root, evidence_root):
        relative = path.relative_to(output_root).as_posix()
        if relative in tar_relative_paths:
            continue
        if relative.endswith(_TAR_SUFFIXES) and not relative.startswith("local_tars/metadata/"):
            raise ValueError(f"scientific tar output is absent from local_tars.csv: {relative!r}")
        digest, stable = _stream_file_sha256(path, snapshot_io=snapshot_io)
        small_outputs.append(
            PostprocessingSmallOutputIdentity(path=relative, sha256=digest, size_bytes=stable.size_bytes)
        )
        stable_files.append(stable)
    references = tuple(
        sorted(
            (
                PostprocessingTarManifestReference(
                    path=f"outputs/tar-manifests/{manifest.manifest_id}.json",
                    manifest_id=manifest.manifest_id,
                    member_count=len(manifest.members),
                )
                for manifest in manifests
            ),
            key=lambda item: item.path,
        )
    )
    root = PostprocessingScientificOutputRoot(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        output_root=str(output_root),
        small_outputs=tuple(sorted(small_outputs, key=lambda item: item.path)),
        tar_manifests=references,
    )
    documents = {
        "outputs/scientific-output-root.json": _canonical_bytes(root.to_mapping()),
        **{
            f"outputs/tar-manifests/{manifest.manifest_id}.json": _canonical_bytes(manifest.to_mapping())
            for manifest in manifests
        },
    }
    tar_stable = tuple(
        _StableFile(
            path=path,
            device=manifest.stat_device,
            inode=manifest.stat_inode,
            mode=path.stat(follow_symlinks=False).st_mode,
            size_bytes=manifest.tar_size_bytes,
            mtime_ns=manifest.stat_mtime_ns,
            ctime_ns=path.stat(follow_symlinks=False).st_ctime_ns,
        )
        for (relative, path), manifest in zip(ordered_tars, manifests, strict=True)
    )
    return _ScientificDocuments(
        root=root,
        documents=documents,
        manifests=manifests,
        source_files=tuple(sorted((*stable_files, *tar_stable), key=lambda item: str(item.path))),
    )


def _scientific_source_fingerprints(
    runspec: ExecutablePostprocessingPhaseRunSpec,
    files: tuple[_StableFile, ...],
) -> bytes:
    output_root = Path(runspec.payload.attempt_paths.output_dir)
    return _canonical_bytes(
        {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "fingerprint_kind": "postprocessing-scientific-source-fingerprints-v2",
            "phase_run_id": runspec.phase_run_id,
            "attempt_id": runspec.attempt_id,
            "phase_runspec_digest": runspec.digest,
            "output_root": str(output_root),
            "files": [
                {
                    "path": item.path.relative_to(output_root).as_posix(),
                    "device": item.device,
                    "inode": item.inode,
                    "mode": item.mode,
                    "size_bytes": item.size_bytes,
                    "mtime_ns": item.mtime_ns,
                    "ctime_ns": item.ctime_ns,
                }
                for item in files
            ],
        }
    )


def _load_staged_scientific_documents(
    runspec: ExecutablePostprocessingPhaseRunSpec,
    evidence_root: Path,
) -> _ScientificDocuments:
    phase_output = evidence_root / "phase-output"
    root_document = _stable_file_bytes(phase_output / "scientific-output-root.json")
    root = postprocessing_scientific_output_root_from_mapping(
        _canonical_mapping(root_document, label="staged scientific output root")
    )
    if (root.phase_run_id, root.attempt_id, root.phase_runspec_digest, root.output_root) != (
        runspec.phase_run_id,
        runspec.attempt_id,
        runspec.digest,
        runspec.payload.attempt_paths.output_dir,
    ):
        raise ValueError("staged scientific output root differs from the frozen Phase RunSpec")
    documents: dict[str, bytes] = {"outputs/scientific-output-root.json": root_document}
    manifests: list[PostprocessingTarManifest] = []
    for reference in root.tar_manifests:
        relative = reference.path.removeprefix("outputs/")
        document = _stable_file_bytes(phase_output / Path(*relative.split("/")))
        manifest = postprocessing_tar_manifest_from_mapping(
            _canonical_mapping(document, label=f"staged tar manifest {reference.manifest_id}")
        )
        if manifest.manifest_id != reference.manifest_id or len(manifest.members) != reference.member_count:
            raise ValueError("staged tar manifest differs from its scientific root reference")
        documents[reference.path] = document
        manifests.append(manifest)
    fingerprint_document = _stable_file_bytes(phase_output / "source-fingerprints.json")
    fingerprint = _canonical_mapping(fingerprint_document, label="staged scientific source fingerprints")
    expected_fields = {
        "schema_version",
        "fingerprint_kind",
        "phase_run_id",
        "attempt_id",
        "phase_runspec_digest",
        "output_root",
        "files",
    }
    if set(fingerprint) != expected_fields or (
        fingerprint.get("schema_version"),
        fingerprint.get("fingerprint_kind"),
        fingerprint.get("phase_run_id"),
        fingerprint.get("attempt_id"),
        fingerprint.get("phase_runspec_digest"),
        fingerprint.get("output_root"),
    ) != (
        CURRENT_CONTRACT_SCHEMA_VERSION,
        "postprocessing-scientific-source-fingerprints-v2",
        runspec.phase_run_id,
        runspec.attempt_id,
        runspec.digest,
        runspec.payload.attempt_paths.output_dir,
    ):
        raise ValueError("staged scientific source fingerprints differ from the frozen Phase RunSpec")
    raw_files = fingerprint.get("files")
    if not isinstance(raw_files, list) or any(not isinstance(item, Mapping) for item in raw_files):
        raise ValueError("staged scientific source fingerprints files must be mappings")
    output_root = Path(runspec.payload.attempt_paths.output_dir)
    files = tuple(_stable_file_from_fingerprint(output_root, cast("Mapping[str, object]", item)) for item in raw_files)
    paths = tuple(item.path.relative_to(output_root).as_posix() for item in files)
    expected_paths = tuple(
        sorted((*[item.path for item in root.small_outputs], *[item.tar_path for item in manifests]))
    )
    if paths != expected_paths or len(set(paths)) != len(paths):
        raise ValueError("staged scientific source fingerprints do not cover the exact logical outputs")
    _verify_source_files_descriptor(output_root, files)
    return _ScientificDocuments(
        root=root,
        documents=documents,
        manifests=tuple(manifests),
        source_files=files,
    )


def _stable_file_from_fingerprint(output_root: Path, payload: Mapping[str, object]) -> _StableFile:
    expected = {"path", "device", "inode", "mode", "size_bytes", "mtime_ns", "ctime_ns"}
    path_value = payload.get("path")
    if set(payload) != expected or not isinstance(path_value, str):
        raise ValueError("staged scientific source fingerprint has missing or extra fields")
    relative = Path(path_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or relative.as_posix() != path_value:
        raise ValueError("staged scientific source fingerprint path is unsafe")

    def integer(name: str) -> int:
        value = payload.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("staged scientific source fingerprint stat is invalid")
        return value

    return _StableFile(
        path=output_root / relative,
        device=integer("device"),
        inode=integer("inode"),
        mode=integer("mode"),
        size_bytes=integer("size_bytes"),
        mtime_ns=integer("mtime_ns"),
        ctime_ns=integer("ctime_ns"),
    )


def _verify_source_files_descriptor(
    output_root: Path,
    files: tuple[_StableFile, ...],
    *,
    snapshot_io: ScientificSnapshotIO = LOCAL_SCIENTIFIC_SNAPSHOT_IO,
) -> None:
    root_descriptor = os.open(
        output_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for expected in files:
            relative = expected.path.relative_to(output_root)
            current = os.dup(root_descriptor)
            try:
                for component in relative.parts[:-1]:
                    child = os.open(
                        component,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=current,
                    )
                    os.close(current)
                    current = child
                descriptor = os.open(
                    relative.parts[-1],
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current,
                )
                try:
                    observed = snapshot_io.fstat(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                os.close(current)
            if not stat.S_ISREG(observed.st_mode) or (
                observed.st_dev,
                observed.st_ino,
                observed.st_mode,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            ) != (
                expected.device,
                expected.inode,
                expected.mode,
                expected.size_bytes,
                expected.mtime_ns,
                expected.ctime_ns,
            ):
                raise ValueError(f"scientific output changed before finalization publication: {expected.path}")
    finally:
        os.close(root_descriptor)


def _inventory_tar_paths(output_root: Path) -> dict[str, Path]:
    manifest_path = output_root / "local_tars.csv"
    document = _stable_file_bytes(manifest_path)
    try:
        text = document.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("local_tars.csv must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        raise ValueError("local_tars.csv must have a header")
    result: dict[str, Path] = {}
    seen: set[str] = set()
    for row_number, raw in enumerate(reader, start=2):
        row = {key: value or "" for key, value in raw.items() if key is not None}
        try:
            resolved, declared_relative = resolve_tar_manifest_member(
                row,
                manifest_path=manifest_path,
                local_tar_dir=output_root / "local_tars",
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"local_tars.csv row {row_number} does not resolve exactly") from exc
        candidate = resolved if resolved.is_absolute() else manifest_path.parent / resolved
        canonical = candidate.resolve(strict=True)
        if candidate != canonical or output_root not in canonical.parents:
            raise ValueError(f"local_tars.csv row {row_number} resolves outside or through a link")
        _regular_nonsymlink_file(canonical, label=f"local_tars.csv row {row_number}")
        relative = canonical.relative_to(output_root).as_posix()
        if declared_relative != relative:
            raise ValueError(f"local_tars.csv row {row_number} has a noncanonical resolved member path")
        if not relative.endswith(_TAR_SUFFIXES):
            raise ValueError(f"local_tars.csv row {row_number} does not identify a supported tar")
        if relative in seen:
            raise ValueError(f"local_tars.csv contains a duplicate tar: {relative!r}")
        seen.add(relative)
        is_metadata = row.get("tar_type", "").strip() == "metadata"
        if is_metadata and not relative.startswith("local_tars/metadata/"):
            raise ValueError(f"local_tars.csv metadata row {row_number} resolves outside the metadata namespace")
        declared_size = row.get("size_bytes", "").strip()
        if declared_size and (not declared_size.isdigit() or int(declared_size) != canonical.stat().st_size):
            raise ValueError(f"local_tars.csv row {row_number} size differs from the tar")
        if is_metadata:
            continue
        result[relative] = canonical
        if len(result) > POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.max_tar_manifests:
            raise ValueError("local_tars.csv exceeds the tar manifest limit")
    if not result:
        raise ValueError("local_tars.csv must identify at least one scientific tar")
    return result


def _manifest_tar(
    path: Path,
    relative: str,
    *,
    snapshot_io: ScientificSnapshotIO,
) -> PostprocessingTarManifest:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            before = snapshot_io.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"scientific tar is not a regular file: {relative!r}")
            members: list[PostprocessingTarMemberIdentity] = []
            seen: set[str] = set()
            try:
                with snapshot_io.open_tar(fileobj=handle, mode="r|*") as archive:
                    for member in archive:
                        if not member.isfile() or member.issym() or member.islnk():
                            raise ValueError(
                                f"scientific tar contains a non-regular member: {relative!r}:{member.name!r}"
                            )
                        identity = PostprocessingTarMemberIdentity(
                            path=member.name,
                            sha256="0" * 64,
                            size_bytes=member.size,
                        )
                        if identity.path in seen:
                            raise ValueError(
                                f"scientific tar contains a duplicate member: {relative!r}:{identity.path!r}"
                            )
                        seen.add(identity.path)
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise ValueError(f"scientific tar regular member cannot be streamed: {member.name!r}")
                        digest = hashlib.sha256()
                        observed_size = 0
                        with extracted:
                            while chunk := extracted.read(1024 * 1024):
                                digest.update(chunk)
                                observed_size += len(chunk)
                        if observed_size != member.size:
                            raise ValueError(f"scientific tar member size changed while streaming: {member.name!r}")
                        members.append(
                            PostprocessingTarMemberIdentity(
                                path=identity.path,
                                sha256=digest.hexdigest(),
                                size_bytes=observed_size,
                            )
                        )
                        if len(members) > POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.max_members_per_tar_manifest:
                            raise ValueError(f"scientific tar exceeds the member limit: {relative!r}")
            except tarfile.TarError as exc:
                raise ValueError(f"scientific tar cannot be read: {relative!r}") from exc
            after = snapshot_io.fstat(descriptor)
    finally:
        os.close(descriptor)
    _assert_stable_stat(path, before, after)
    sorted_members = tuple(sorted(members, key=lambda item: item.path))
    identity_mapping: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "manifest_kind": "postprocessing-tar-manifest-v1",
        "tar_path": relative,
        "members": [item.to_mapping() for item in sorted_members],
    }
    return PostprocessingTarManifest(
        tar_path=relative,
        tar_size_bytes=before.st_size,
        stat_device=before.st_dev,
        stat_inode=before.st_ino,
        stat_mtime_ns=before.st_mtime_ns,
        members=sorted_members,
        manifest_id=canonical_mapping_digest(identity_mapping),
    )


def _regular_output_files(output_root: Path, evidence_root: Path) -> tuple[Path, ...]:
    resolved_evidence = evidence_root.resolve(strict=False)
    result: list[Path] = []
    for directory, names, filenames in os.walk(output_root, followlinks=False):
        root = Path(directory)
        for name in tuple(names):
            child = root / name
            if child.is_symlink():
                raise ValueError(f"scientific output tree contains a symlink: {child}")
            if child.resolve(strict=False) == resolved_evidence:
                names.remove(name)
        for name in filenames:
            path = root / name
            _regular_nonsymlink_file(path, label="scientific output member")
            result.append(path)
    if not result:
        raise ValueError("scientific output root contains no regular files")
    return tuple(sorted(result))


def _stream_file_sha256(
    path: Path,
    *,
    snapshot_io: ScientificSnapshotIO,
) -> tuple[str, _StableFile]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            before = snapshot_io.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"file is not regular: {path}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = snapshot_io.fstat(descriptor)
    finally:
        os.close(descriptor)
    _assert_stable_stat(path, before, after)
    return digest.hexdigest(), _stable_identity(path, before)
