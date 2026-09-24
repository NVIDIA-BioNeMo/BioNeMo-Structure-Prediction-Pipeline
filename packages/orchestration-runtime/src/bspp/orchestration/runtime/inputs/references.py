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

"""Reference artifact checks and side-effect-free download planning."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard, cast
from urllib.parse import ParseResult, urlparse, urlunparse

REFERENCE_FIELDS: tuple[str, ...] = ("master_parquet", "tracking_parquet", "manifest_csv", "uniprot_duckdb")
DOWNLOADABLE_SCHEMES: tuple[str, ...] = ("file", "s3", "gs", "http", "https")


@dataclass(frozen=True)
class ReferenceArtifact:
    """Local reference path with optional source URI and integrity metadata."""

    name: str
    path: Path
    uri: str | None = None
    expected_size: int | None = None
    expected_sha256: str | None = None

    @property
    def source(self) -> str:
        """Return the artifact URI if present, otherwise the local path."""
        return self.uri or str(self.path)

    def to_redacted_dict(self) -> dict[str, Any]:
        """Return JSON-serializable artifact metadata without credential material."""
        return {
            "name": self.name,
            "path": str(self.path),
            "uri": redact_uri(self.uri) if self.uri else None,
            "expected_size": self.expected_size,
            "expected_sha256": self.expected_sha256,
        }


@dataclass(frozen=True)
class ReferenceStatus:
    """Filesystem and integrity status for one reference artifact."""

    artifact: ReferenceArtifact
    present: bool
    size: int | None = None
    sha256: str | None = None
    size_ok: bool | None = None
    sha256_ok: bool | None = None

    @property
    def ok(self) -> bool:
        """Return ``True`` when the file exists and provided integrity checks pass."""
        return self.present and self.size_ok is not False and self.sha256_ok is not False

    def to_redacted_dict(self) -> dict[str, Any]:
        """Return JSON-serializable status data without secrets."""
        return {
            "artifact": self.artifact.to_redacted_dict(),
            "present": self.present,
            "ok": self.ok,
            "size": self.size,
            "sha256": self.sha256,
            "size_ok": self.size_ok,
            "sha256_ok": self.sha256_ok,
        }


@dataclass(frozen=True)
class DownloadPlan:
    """A planned reference download or copy."""

    name: str
    source: str
    destination: Path
    scheme: str
    action: str

    def to_redacted_dict(self) -> dict[str, Any]:
        """Return JSON-serializable plan data without credential material."""
        return {
            "name": self.name,
            "source": redact_uri(self.source),
            "destination": str(self.destination),
            "scheme": self.scheme,
            "action": self.action,
        }


def references_from_spec(spec_or_references: object) -> tuple[ReferenceArtifact, ...]:
    """Build artifacts from a RunSpec or ReferenceSpec.

    Future artifact objects may expose metadata such as ``uri``, ``size`` or
    ``sha256``. Current RunSpec fields are plain paths, so this function falls
    back to the path field value.
    """
    refs = getattr(spec_or_references, "references", spec_or_references)
    artifacts: list[ReferenceArtifact] = []
    for name in REFERENCE_FIELDS:
        if hasattr(refs, "artifact"):
            artifacts.append(_artifact_from_value(name, cast(Any, refs).artifact(name)))
            continue
        if hasattr(refs, name):
            artifacts.append(_artifact_from_value(name, getattr(refs, name)))
    return tuple(artifacts)


def check_references(references: object) -> tuple[ReferenceStatus, ...]:
    """Check reference presence, size and hash where metadata is available."""
    artifacts: tuple[ReferenceArtifact, ...] = (
        tuple(references) if _is_artifact_sequence(references) else references_from_spec(references)
    )
    return tuple(check_reference(artifact) for artifact in artifacts)


def check_reference(artifact: ReferenceArtifact) -> ReferenceStatus:
    """Check one reference artifact."""
    if not artifact.path.exists():
        return ReferenceStatus(artifact=artifact, present=False)

    size = artifact.path.stat().st_size
    size_ok = size >= artifact.expected_size if artifact.expected_size is not None else None
    sha256 = None
    sha256_ok = None
    if artifact.expected_sha256 is not None:
        sha256 = _sha256_file(artifact.path)
        sha256_ok = sha256.lower() == artifact.expected_sha256.lower()
    return ReferenceStatus(
        artifact=artifact,
        present=True,
        size=size,
        sha256=sha256,
        size_ok=size_ok,
        sha256_ok=sha256_ok,
    )


def plan_reference_downloads(references: object) -> tuple[DownloadPlan, ...]:
    """Plan downloads for missing references with supported URI schemes."""
    plans: list[DownloadPlan] = []
    for status in check_references(references):
        if status.present:
            continue
        source = status.artifact.source
        parsed = urlparse(source)
        if status.artifact.uri is None and not parsed.scheme:
            plans.append(
                DownloadPlan(
                    name=status.artifact.name,
                    source=source,
                    destination=status.artifact.path,
                    scheme="local",
                    action="missing",
                )
            )
            continue
        scheme = parsed.scheme or "file"
        if scheme not in DOWNLOADABLE_SCHEMES:
            raise ValueError(f"Unsupported reference URI scheme {scheme!r} for {status.artifact.name}")
        action = "copy" if scheme == "file" else "download"
        plans.append(
            DownloadPlan(
                name=status.artifact.name,
                source=source,
                destination=status.artifact.path,
                scheme=scheme,
                action=action,
            )
        )
    return tuple(plans)


def ensure_references(references: object, *, dry_run: bool = True) -> tuple[DownloadPlan, ...]:
    """Ensure references are present, returning planned work.

    In dry-run mode this function performs no filesystem writes. Non-dry-run
    execution supports local ``file://`` copies only; remote schemes are planned
    explicitly for an external transfer layer.
    """
    plans = plan_reference_downloads(references)
    if dry_run:
        return plans

    for plan in plans:
        if plan.action == "missing":
            continue
        if plan.scheme != "file":
            raise NotImplementedError(f"Reference download for {plan.scheme!r} is not implemented")
        source_path = _file_uri_to_path(plan.source)
        plan.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, plan.destination)
    return plans


def redact_uri(uri: str) -> str:
    """Redact URI userinfo, query strings and fragments."""
    parsed = urlparse(uri)
    if not parsed.scheme:
        return uri
    netloc = parsed.hostname or ""
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username or parsed.password:
        netloc = f"<redacted>@{netloc}"
    return urlunparse(ParseResult(parsed.scheme, netloc, parsed.path, parsed.params, "", ""))


def _artifact_from_value(name: str, value: object) -> ReferenceArtifact:
    path_value = _first_attr(value, ("path", "local_path", "destination", "destination_path"), default=value)
    uri_value = _first_attr(value, ("uri", "source_uri", "source", "url"), default=None)
    expected_size = _first_attr(value, ("expected_size", "min_size_bytes", "size", "bytes"), default=None)
    expected_sha256 = _first_attr(value, ("expected_sha256", "sha256", "checksum_sha256"), default=None)

    path = _path_from_value(path_value)
    uri = str(uri_value) if uri_value is not None else _uri_from_path_value(path_value)
    return ReferenceArtifact(
        name=name,
        path=path,
        uri=uri,
        expected_size=_optional_int(expected_size),
        expected_sha256=str(expected_sha256) if expected_sha256 is not None else None,
    )


def _first_attr(value: object, names: tuple[str, ...], *, default: object) -> object:
    for name in names:
        if hasattr(value, name):
            attr = getattr(value, name)
            if attr is not None:
                return attr
    return default


def _path_from_value(value: object) -> Path:
    if isinstance(value, Path):
        return value
    text = str(value)
    parsed = urlparse(text)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if parsed.scheme:
        name = Path(parsed.path).name
        return Path(name or parsed.netloc)
    return Path(text)


def _uri_from_path_value(value: object) -> str | None:
    text = str(value)
    parsed = urlparse(text)
    return text if parsed.scheme else None


def _file_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if parsed.scheme:
        raise ValueError(f"Expected file URI or path, got {uri!r}")
    return Path(uri)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    msg = f"Expected integer-like size metadata, got {type(value).__name__}"
    raise TypeError(msg)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_artifact_sequence(value: object) -> TypeGuard[tuple[ReferenceArtifact, ...] | list[ReferenceArtifact]]:
    if not isinstance(value, tuple | list):
        return False
    return all(isinstance(item, ReferenceArtifact) for item in value)


__all__ = [
    "DOWNLOADABLE_SCHEMES",
    "REFERENCE_FIELDS",
    "DownloadPlan",
    "ReferenceArtifact",
    "ReferenceStatus",
    "check_reference",
    "check_references",
    "ensure_references",
    "plan_reference_downloads",
    "redact_uri",
    "references_from_spec",
]
