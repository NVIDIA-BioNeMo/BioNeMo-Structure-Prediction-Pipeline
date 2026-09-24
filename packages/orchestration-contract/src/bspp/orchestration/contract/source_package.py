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

"""Versioned, deterministic and fail-closed governed source packages."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, BinaryIO, Final

SOURCE_PACKAGE_FORMAT_VERSION: Final = 1
MANIFEST_MEMBER: Final = ".bspp-source-manifest.json"
MAX_MEMBERS: Final = 100_000
MAX_MEMBER_BYTES: Final = 1 << 30
MAX_PACKAGE_BYTES: Final = 8 << 30
MAX_MANIFEST_BYTES: Final = 64 << 20
_PRODUCERS: Final = frozenset({"bspp-tar-v1"})
_GOVERNED_RUNTIME_PREFIXES: Final = (
    "packages/orchestration-contract/src/",
    "packages/orchestration-runtime/src/",
)
_GOVERNED_RUNTIME_FILES: Final = frozenset(
    {
        "containers/scripts/slurm-semantic-acceptance.sh",
        "containers/scripts/slurm-tar-payload-parity.sh",
    }
)
_REGISTERED_PAIRS: Final = frozenset({("bspp-tar-v1", "safe-tar-v1"), ("safe-tar-v1", "safe-tar-v1")})


@dataclass(frozen=True)
class SourcePackageManifestEntry:
    """Canonical identity of one payload file."""

    path: str
    size_bytes: int
    sha256: str
    mode: str

    def to_mapping(self) -> dict[str, object]:
        return {"mode": self.mode, "path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class SourcePackageIdentity:
    """Strict identity binding for a final source-package archive."""

    format: str
    verifier: str
    package_path: Path
    package_size_bytes: int
    package_sha256: str
    manifest_sha256: str
    commit: str
    tree: str
    package_role: str = "generic"
    policy_version: int = 1
    format_version: int = SOURCE_PACKAGE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != SOURCE_PACKAGE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported SourcePackageIdentity format_version {self.format_version}; supported versions: 1"
            )
        if not self.package_path.is_absolute():
            raise ValueError("SourcePackageIdentity package_path must be absolute")
        if self.package_size_bytes < 0:
            raise ValueError("SourcePackageIdentity package_size_bytes must be non-negative")
        if self.package_role not in {"generic", "orchestration", "toolkit"} or self.policy_version != 1:
            raise ValueError("Unsupported SourcePackageIdentity package role or policy version")
        for name, value, length in (
            ("package_sha256", self.package_sha256, 64),
            ("manifest_sha256", self.manifest_sha256, 64),
            ("commit", self.commit, 40),
            ("tree", self.tree, 40),
        ):
            if len(value) != length or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"SourcePackageIdentity {name} must be {length} lowercase hexadecimal characters")

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "format": self.format,
            "verifier": self.verifier,
            "package_path": str(self.package_path),
            "package_size_bytes": self.package_size_bytes,
            "package_sha256": self.package_sha256,
            "manifest_sha256": self.manifest_sha256,
            "commit": self.commit,
            "tree": self.tree,
            "package_role": self.package_role,
            "policy_version": self.policy_version,
        }

    @classmethod
    def for_existing_archive(
        cls,
        path: Path,
        *,
        format: str,
        verifier: str,
        manifest_sha256: str,
        commit: str,
        tree: str,
    ) -> SourcePackageIdentity:
        absolute = path.absolute()
        size, digest = _stable_file_identity(absolute)
        return cls(format, verifier, absolute, size, digest, manifest_sha256, commit, tree)


def source_package_identity_from_mapping(payload: Mapping[str, object]) -> SourcePackageIdentity:
    """Parse a strict v1 identity, rejecting missing, extra and future fields."""
    expected = {
        "format_version",
        "format",
        "verifier",
        "package_path",
        "package_size_bytes",
        "package_sha256",
        "manifest_sha256",
        "commit",
        "tree",
        "package_role",
        "policy_version",
    }
    extra = set(payload) - expected
    missing = expected - set(payload)
    if extra or missing:
        raise ValueError(f"Invalid SourcePackageIdentity fields; missing={sorted(missing)}, extra={sorted(extra)}")
    version = payload["format_version"]
    if type(version) is not int or version != SOURCE_PACKAGE_FORMAT_VERSION:
        raise ValueError(f"Unsupported SourcePackageIdentity format_version {version}; supported versions: 1")
    if type(payload["package_size_bytes"]) is not int:
        raise ValueError("SourcePackageIdentity package_size_bytes must be an integer")
    if type(payload["policy_version"]) is not int:
        raise ValueError("SourcePackageIdentity policy_version must be an integer")
    string_fields = (
        "format",
        "verifier",
        "package_path",
        "package_sha256",
        "manifest_sha256",
        "commit",
        "tree",
        "package_role",
    )
    if any(not isinstance(payload[name], str) for name in string_fields):
        raise ValueError("SourcePackageIdentity string fields must be strings")
    return SourcePackageIdentity(
        format=str(payload["format"]),
        verifier=str(payload["verifier"]),
        package_path=Path(str(payload["package_path"])),
        package_size_bytes=int(payload["package_size_bytes"]),
        package_sha256=str(payload["package_sha256"]),
        manifest_sha256=str(payload["manifest_sha256"]),
        commit=str(payload["commit"]),
        tree=str(payload["tree"]),
        package_role=str(payload["package_role"]),
        policy_version=int(payload["policy_version"]),
        format_version=version,
    )


def build_source_package(
    source_root: Path,
    package_path: Path,
    *,
    commit: str,
    tree: str,
    format: str = "bspp-tar-v1",
    tracked_git: bool = False,
    git_subtree: str | None = None,
    governed_runtime_only: bool = True,
    allow_untracked: bool = False,
    package_role: str | None = None,
) -> SourcePackageIdentity:
    """Build a deterministic archive from regular files under ``source_root``."""
    if format not in _PRODUCERS:
        raise ValueError(f"Unknown source package producer format: {format}")
    root = source_root.resolve(strict=True)
    output = package_path.absolute()
    if output.exists():
        raise ValueError(f"Source package output already exists: {output}")
    entries: list[tuple[SourcePackageManifestEntry, Path]] = []
    if tracked_git:
        tree_ref = "HEAD^{tree}" if git_subtree is None else f"HEAD:{git_subtree}"
        if _git(root, "rev-parse", "HEAD") != commit or _git(root, "rev-parse", tree_ref) != tree:
            raise ValueError("Governed source package commit/tree do not match repository HEAD")
        scope = git_subtree or "."
        untracked_mode = "no" if allow_untracked else "all"
        status = _git(root, "status", "--porcelain=v1", f"--untracked-files={untracked_mode}", "--", scope)
        if status:
            raise ValueError("Governed source package requires a clean tracked repository")
        raw = subprocess.run(
            ("git", "-C", str(root), "ls-files", "-z", "--stage", "--", scope), check=True, capture_output=True
        ).stdout
        candidates: list[tuple[str, Path, str]] = []
        for record in raw.split(b"\0"):
            if not record:
                continue
            index_metadata, encoded_path = record.split(b"\t", 1)
            git_mode, _object_id, stage = index_metadata.decode("ascii").split()
            relative = encoded_path.decode("utf-8", errors="strict")
            if stage != "0" or git_mode not in {"100644", "100755"}:
                raise ValueError(f"Governed source tracked member is not a regular stage-0 file: {relative}")
            if git_subtree is not None:
                prefix = git_subtree.rstrip("/") + "/"
                if not relative.startswith(prefix):
                    raise ValueError("Governed source member escaped configured Git subtree")
                relative = relative.removeprefix(prefix)
            if (
                governed_runtime_only
                and not relative.startswith(_GOVERNED_RUNTIME_PREFIXES)
                and relative not in _GOVERNED_RUNTIME_FILES
            ):
                continue
            repository_path = encoded_path.decode("utf-8", errors="strict")
            candidates.append((relative, root / repository_path, "0o755" if git_mode == "100755" else "0o644"))
    else:
        candidates = []
        for path in root.rglob("*"):
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                continue
            relative = path.relative_to(root).as_posix()
            candidates.append((relative, path, "0o755" if metadata.st_mode & 0o111 else "0o644"))
    for relative, path, mode in sorted(candidates, key=lambda item: item[0].encode("utf-8")):
        _validate_member_name(relative)
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Source package payload must contain only regular files: {relative}")
        size, digest = _stable_file_identity(path)
        if size > MAX_MEMBER_BYTES:
            raise ValueError(f"Source package member exceeds size bound: {relative}")
        entries.append((SourcePackageManifestEntry(relative, size, digest, mode), path))
    if len(entries) + 1 > MAX_MEMBERS or sum(entry.size_bytes for entry, _path in entries) > MAX_PACKAGE_BYTES:
        raise ValueError("Source package payload exceeds bounds")
    manifest = _canonical_manifest(tuple(entry for entry, _path in entries), commit=commit, tree=tree)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve existing USTAR bytes whenever possible. GNU long-name records
    # cover valid tracked paths that USTAR cannot encode without introducing
    # the PAX metadata that the verifier deliberately rejects.
    archive_format = tarfile.USTAR_FORMAT
    for entry, _path in entries:
        try:
            _tar_info(entry.path, entry.size_bytes, int(entry.mode, 8)).tobuf(format=tarfile.USTAR_FORMAT)
        except ValueError:
            archive_format = tarfile.GNU_FORMAT
            break
    with tarfile.open(output, "x", format=archive_format) as archive:
        manifest_info = _tar_info(MANIFEST_MEMBER, len(manifest), 0o600)
        archive.addfile(manifest_info, _BytesReader(manifest))
        for entry, path in entries:
            info = _tar_info(entry.path, entry.size_bytes, int(entry.mode, 8))
            with open(path, "rb", opener=_no_follow_opener) as stream:
                reader = _DigestingReader(stream)
                archive.addfile(info, reader)
                if reader.size != entry.size_bytes or reader.digest.hexdigest() != entry.sha256:
                    raise ValueError(f"Source package member changed while archiving: {entry.path}")
    size, package_digest = _stable_file_identity(output)
    return SourcePackageIdentity(
        format=format,
        verifier="safe-tar-v1",
        package_path=output,
        package_size_bytes=size,
        package_sha256=package_digest,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        commit=commit,
        tree=tree,
        package_role=package_role or ("orchestration" if tracked_git and governed_runtime_only else "generic"),
    )


def verify_source_package(
    identity: SourcePackageIdentity, *, destination: Path | None = None, expected_role: str | None = None
) -> None:
    """Verify identity and all members, then optionally extract without tar traversal."""
    if (identity.format, identity.verifier) not in _REGISTERED_PAIRS:
        raise ValueError(f"Unknown source package format/verifier pair: {identity.format}/{identity.verifier}")
    if expected_role is not None and identity.package_role != expected_role:
        raise ValueError(f"Source package role mismatch: expected {expected_role}")
    descriptor = os.open(identity.package_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Source package is not a regular file")
        if before.st_size > MAX_PACKAGE_BYTES:
            raise ValueError("Source package exceeds size bound")
        if before.st_size != identity.package_size_bytes:
            raise ValueError("Source package size mismatch")
        package_digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            package_digest.update(chunk)
        if package_digest.hexdigest() != identity.package_sha256:
            raise ValueError("Source package SHA256 mismatch")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with (
            os.fdopen(os.dup(descriptor), "rb") as package_stream,
            tarfile.open(fileobj=package_stream, mode="r:") as archive,
        ):
            _verify_archive_stream(archive, identity, destination)
        after = os.fstat(descriptor)
        if _stat_signature(before) != _stat_signature(after):
            raise ValueError("Source package changed while verifying")
    finally:
        os.close(descriptor)


def _verify_archive_stream(archive: tarfile.TarFile, identity: SourcePackageIdentity, destination: Path | None) -> None:
    first = archive.next()
    if first is None:
        raise ValueError("Source package manifest must be the first member")
    _validate_tar_member(first)
    if first.name != MANIFEST_MEMBER:
        raise ValueError("Source package manifest must be the first member")
    if first.size > MAX_MANIFEST_BYTES:
        raise ValueError("Source package manifest exceeds size bound")
    manifest_stream = archive.extractfile(first)
    if manifest_stream is None:
        raise ValueError("Source package manifest is truncated")
    manifest_bytes = manifest_stream.read(first.size + 1)
    if len(manifest_bytes) != first.size:
        raise ValueError("Source package manifest is truncated")
    if hashlib.sha256(manifest_bytes).hexdigest() != identity.manifest_sha256:
        raise ValueError("Source package manifest SHA256 mismatch")
    entries = _parse_manifest(manifest_bytes, identity)
    if identity.package_role == "orchestration" and any(
        not entry.path.startswith(_GOVERNED_RUNTIME_PREFIXES) and entry.path not in _GOVERNED_RUNTIME_FILES
        for entry in entries
    ):
        raise ValueError("Orchestration package contains a member outside governed runtime policy")
    if len(entries) + 1 > MAX_MEMBERS:
        raise ValueError("Source package exceeds member bound")
    if destination is not None:
        if destination.exists() or destination.is_symlink():
            raise ValueError("Source package extraction destination must not already exist")
        destination.mkdir(mode=0o700, parents=False)
    total_size = 0
    for entry in entries:
        member = archive.next()
        if member is None or member.name != entry.path:
            raise ValueError("Source package has extra, missing, or reordered payload members")
        _validate_tar_member(member)
        total_size += member.size
        if member.size > MAX_MEMBER_BYTES or total_size > MAX_PACKAGE_BYTES:
            raise ValueError(f"Source package member exceeds size bound: {member.name}")
        if member.size != entry.size_bytes or (member.mode & 0o777) != int(entry.mode, 8):
            raise ValueError(f"Source package payload size or mode mismatch: {entry.path}")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"Truncated source package member: {entry.path}")
        target = destination.joinpath(*PurePosixPath(entry.path).parts) if destination is not None else None
        _stream_verified_member(stream, entry, target)
    if archive.next() is not None:
        raise ValueError("Source package has extra payload members")


def _validate_tar_member(member: tarfile.TarInfo) -> None:
    _validate_member_name(member.name)
    if not member.isfile() or member.sparse is not None or member.pax_headers:
        raise ValueError(f"Source package members must be regular files without PAX/sparse metadata: {member.name}")


def _stream_verified_member(stream: IO[bytes], entry: SourcePackageManifestEntry, target: Path | None) -> None:
    descriptor: int | None = None
    if target is not None:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), int(entry.mode, 8)
        )
    digest = hashlib.sha256()
    remaining = entry.size_bytes
    try:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"Truncated source package member: {entry.path}")
            remaining -= len(chunk)
            digest.update(chunk)
            if descriptor is not None:
                offset = 0
                while offset < len(chunk):
                    offset += os.write(descriptor, chunk[offset:])
        if digest.hexdigest() != entry.sha256:
            raise ValueError(f"Source package payload digest mismatch: {entry.path}")
        if descriptor is not None:
            os.fchmod(descriptor, int(entry.mode, 8))
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _canonical_manifest(entries: tuple[SourcePackageManifestEntry, ...], *, commit: str, tree: str) -> bytes:
    value = {"commit": commit, "entries": [entry.to_mapping() for entry in entries], "tree": tree}
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _parse_manifest(data: bytes, identity: SourcePackageIdentity) -> tuple[SourcePackageManifestEntry, ...]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Source package manifest is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != {"commit", "entries", "tree"}:
        raise ValueError("Source package manifest schema mismatch")
    if value["commit"] != identity.commit or value["tree"] != identity.tree:
        raise ValueError("Source package commit/tree mismatch")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("Source package manifest entries must be a list")
    entries: list[SourcePackageManifestEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, dict) or set(raw) != {"mode", "path", "sha256", "size_bytes"}:
            raise ValueError("Source package manifest entry schema mismatch")
        entry = SourcePackageManifestEntry(
            str(raw["path"]), int(raw["size_bytes"]), str(raw["sha256"]), str(raw["mode"])
        )
        _validate_member_name(entry.path)
        if entry.path == MANIFEST_MEMBER or entry.mode not in {"0o644", "0o755"}:
            raise ValueError("Source package manifest contains invalid control path or mode")
        entries.append(entry)
    if [entry.path for entry in entries] != sorted(
        (entry.path for entry in entries), key=lambda path: path.encode("utf-8")
    ):
        raise ValueError("Source package manifest entries are not canonical lexicographic order")
    if len({entry.path for entry in entries}) != len(entries):
        raise ValueError("Source package manifest contains duplicate paths")
    if _canonical_manifest(tuple(entries), commit=identity.commit, tree=identity.tree) != data:
        raise ValueError("Source package manifest is not canonical compact JSON")
    return tuple(entries)


def _validate_member_name(name: str) -> None:
    try:
        name.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"Source package member has unsafe non-UTF-8 name: {name!r}") from exc
    path = PurePosixPath(name)
    if not name or name.startswith("/") or "\\" in name or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Source package member has unsafe path: {name!r}")


def _stable_file_identity(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Source package path is unsafe or unavailable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Source package path is not a regular file: {path}")
        if before.st_size > MAX_PACKAGE_BYTES:
            raise ValueError("Source package exceeds size bound")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Source package path changed while hashing: {path}")
        return before.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _stable_file_read(path: Path) -> tuple[bytes, int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Source package path is unsafe or unavailable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Source package path is not a regular file: {path}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Source package path changed while hashing: {path}")
        return b"".join(chunks), before.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", "-C", str(repo), *args), check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _tar_info(name: str, size: int, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


class _BytesReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._data) - self._offset
        result = self._data[self._offset : self._offset + size]
        self._offset += len(result)
        return result


class _DigestingReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(size)
        self.digest.update(chunk)
        self.size += len(chunk)
        return chunk


def _no_follow_opener(path: str, flags: int) -> int:
    return os.open(path, flags | getattr(os, "O_NOFOLLOW", 0))


__all__ = [
    "MANIFEST_MEMBER",
    "SOURCE_PACKAGE_FORMAT_VERSION",
    "SourcePackageIdentity",
    "SourcePackageManifestEntry",
    "build_source_package",
    "source_package_identity_from_mapping",
    "verify_source_package",
]
