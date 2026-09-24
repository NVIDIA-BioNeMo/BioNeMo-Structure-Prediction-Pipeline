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

"""Preprocessing Source Bundle build and staging for Runtime Qualification.

Builds the preprocessing Source Bundle (``bspp-orchestration-<commit>.tar.zst``)
from a clean source checkout and stages its exact bytes to the Cluster Profile
``source_bundle_root`` over the profile transport. The printed record carries
the ``source_commit`` and ``source_bundle_sha256`` pins that the Cluster Profile
``preprocessing_runtime`` block requires before
``bsppctl runtime preprocessing qualify``.

The bundle is provenance-only: the qualification smoke and every rendered
preprocessing action verify its path and SHA-256 but never extract it — the
runtime image's baked entry points are the executable source. Despite the
historical ``.tar.zst`` suffix (load-bearing in the qualification tuple path
derivation, so it stays) the payload is an uncompressed deterministic tar, as
the retired producer emitted it. Bytes come from
``contract.source_package.build_source_package`` over the whole tracked tree
with git-index-normalized modes, so a clean commit rebuilds identical bytes
independent of checkout umask.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from bspp.orchestration.contract.source_package import build_source_package, verify_source_package
from bspp.orchestration.control.profiles import ResolvedClusterProfile
from bspp.orchestration.control.transport import CommandRunner, RemoteSlurmTransport, default_command_runner


class PreprocessingSourceBundleError(ValueError):
    """The preprocessing Source Bundle cannot be built or staged."""


@dataclass(frozen=True)
class PreprocessingSourceBundleRecord:
    """Evidence for one built (and optionally staged) preprocessing Source Bundle."""

    source_bundle_id: str
    source_bundle_path: str
    source_bundle_sha256: str
    source_commit: str
    source_tree: str
    source_bundle_root: str
    archive_path: Path
    archive_size_bytes: int
    manifest_sha256: str
    staged: bool

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic YAML/JSON-ready data."""
        return {
            "preprocessing_source_bundle": {
                "source_bundle_id": self.source_bundle_id,
                "source_bundle_path": self.source_bundle_path,
                "source_bundle_sha256": self.source_bundle_sha256,
                "payload_format": "uncompressed-tar",
                "source_commit": self.source_commit,
                "source_tree": self.source_tree,
                "source_bundle_root": self.source_bundle_root,
                "archive_path": str(self.archive_path),
                "archive_size_bytes": self.archive_size_bytes,
                "manifest_sha256": self.manifest_sha256,
                "staged": self.staged,
            }
        }


def build_and_stage_preprocessing_source_bundle(
    source_repo: Path,
    *,
    build_dir: Path,
    profile: ResolvedClusterProfile,
    runner: CommandRunner = default_command_runner,
    dry_run: bool = False,
) -> PreprocessingSourceBundleRecord:
    """Build the deterministic bundle from a clean checkout and stage its exact bytes."""
    if profile.source_bundle_root is None:
        raise PreprocessingSourceBundleError(f"Cluster Profile {profile.name!r} does not define source_bundle_root")
    repo = source_repo.resolve(strict=True)
    if Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise PreprocessingSourceBundleError(
            f"preprocessing Source Bundle source repo must be the checkout root, not a subdirectory: {repo}"
        )
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise PreprocessingSourceBundleError("preprocessing Source Bundle requires clean source")
    bundle_id = f"bspp-orchestration-{commit}"
    target_path = str(Path(profile.source_bundle_root) / f"{bundle_id}.tar.zst")
    build_dir.mkdir(parents=True, exist_ok=True)
    temporary = build_dir / f"{bundle_id}.{uuid.uuid4().hex}.tar.zst"
    try:
        identity = build_source_package(
            repo,
            temporary,
            commit=commit,
            tree=tree,
            tracked_git=True,
            governed_runtime_only=False,
            allow_untracked=False,
        )
        verify_source_package(identity)
        archive_path = build_dir / f"{bundle_id}.tar.zst"
        if archive_path.exists():
            if _file_sha256(archive_path) != identity.package_sha256:
                raise PreprocessingSourceBundleError(
                    f"preprocessing Source Bundle rebuild is not byte-deterministic: {archive_path}"
                )
            temporary.unlink()
        else:
            os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    staged = False
    if not dry_run:
        pinned = profile.preprocessing_runtime
        if (
            pinned is not None
            and pinned.source_commit == commit
            and pinned.source_bundle_sha256 != "0" * 64
            and pinned.source_bundle_sha256 != identity.package_sha256
        ):
            raise PreprocessingSourceBundleError(
                "Cluster Profile preprocessing_runtime already pins a different source_bundle_sha256 "
                f"for commit {commit}: {pinned.source_bundle_sha256} != {identity.package_sha256}"
            )
        transport = RemoteSlurmTransport(kind=profile.transport, ssh_target=profile.ssh_target, runner=runner)
        transport.stage_immutable_artifact(
            archive_path,
            target_path,
            expected_sha256=identity.package_sha256,
            staging_token=f"preprocessing-source-bundle-{commit[:12]}",
        )
        staged = True
    record = PreprocessingSourceBundleRecord(
        source_bundle_id=bundle_id,
        source_bundle_path=target_path,
        source_bundle_sha256=identity.package_sha256,
        source_commit=commit,
        source_tree=tree,
        source_bundle_root=profile.source_bundle_root,
        archive_path=archive_path,
        archive_size_bytes=identity.package_size_bytes,
        manifest_sha256=identity.manifest_sha256,
        staged=staged,
    )
    evidence_path = build_dir / f"{bundle_id}.source-bundle-identity.json"
    evidence_path.write_text(json.dumps(record.to_mapping(), sort_keys=True, indent=2) + "\n")
    return record


def render_preprocessing_source_bundle_yaml(record: PreprocessingSourceBundleRecord) -> str:
    """Render Source Bundle build/stage evidence as stable YAML."""
    return yaml.safe_dump(record.to_mapping(), sort_keys=False)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", "-C", str(repo), *args), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise PreprocessingSourceBundleError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


__all__ = [
    "PreprocessingSourceBundleError",
    "PreprocessingSourceBundleRecord",
    "build_and_stage_preprocessing_source_bundle",
    "render_preprocessing_source_bundle_yaml",
]
