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

"""Governed Source Package staging for Runtime Qualification.

Builds the governed orchestration source package locally, stages its exact
bytes to the cluster governed package root over the profile transport, and
persists a rebound identity record. This module is the §7.2 migration target
for the retired ``bsppctl provision source-bundle stage`` command and is kept
after the legacy provisioning planners are removed.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

from bspp.orchestration.contract.source_package import SourcePackageIdentity, build_source_package
from bspp.orchestration.control.profiles import ResolvedClusterProfile
from bspp.orchestration.control.transport import (
    CommandResult,
    CommandRunner,
    artifact_copy_argv,
    command_argv,
    default_command_runner,
)


def build_and_stage_governed_source_package(
    source_repo: Path,
    *,
    build_dir: Path,
    target_path: Path,
    profile: ResolvedClusterProfile,
    runner: CommandRunner = default_command_runner,
    verify_remote_digest: bool = True,
) -> SourcePackageIdentity:
    """Build locally, stage exact bytes through transport, and persist the rebound identity."""
    repo = source_repo.resolve(strict=True)
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Governed source package requires a clean source repository")
    build_dir.mkdir(parents=True, exist_ok=True)
    local_path = build_dir / f"source-package.{uuid.uuid4().hex}.tar"
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    local_identity = build_source_package(repo, local_path, commit=commit, tree=tree, tracked_git=True)
    target = (
        Path(profile.governed_package_root) / "orchestration" / commit / f"{local_identity.package_sha256}.tar"
        if profile.governed_package_root is not None
        else target_path.absolute()
    )
    temporary = f"{target}.tmp.{uuid.uuid4().hex}"
    commands = (
        command_argv(("mkdir", "-p", str(target.parent)), transport=profile.transport, ssh_target=profile.ssh_target),
        artifact_copy_argv(local_path, temporary, transport=profile.transport, ssh_target=profile.ssh_target),
        command_argv(
            (
                "/usr/bin/python3",
                "-I",
                "-S",
                "-c",
                _NO_CLOBBER_PUBLISHER,
                temporary,
                str(target),
                str(local_identity.package_size_bytes),
                local_identity.package_sha256,
            ),
            transport=profile.transport,
            ssh_target=profile.ssh_target,
        ),
        command_argv(("sha256sum", str(target)), transport=profile.transport, ssh_target=profile.ssh_target),
    )
    try:
        results = tuple(_run_required(command, runner=runner) for command in commands)
    finally:
        runner(command_argv(("rm", "-f", temporary), transport=profile.transport, ssh_target=profile.ssh_target))
        local_path.unlink(missing_ok=True)
    if verify_remote_digest:
        remote_digest = _sha256sum_stdout(results[-1].stdout)
        if remote_digest != local_identity.package_sha256:
            raise ValueError(
                f"staged governed source package checksum mismatch: expected {local_identity.package_sha256}, "
                f"got {remote_digest}"
            )
    staged = SourcePackageIdentity(
        format=local_identity.format,
        verifier=local_identity.verifier,
        package_path=target,
        package_size_bytes=local_identity.package_size_bytes,
        package_sha256=local_identity.package_sha256,
        manifest_sha256=local_identity.manifest_sha256,
        commit=local_identity.commit,
        tree=local_identity.tree,
        package_role=local_identity.package_role,
        policy_version=local_identity.policy_version,
    )
    (build_dir / "source-package-identity.json").write_text(
        json.dumps(staged.to_mapping(), sort_keys=True, indent=2) + "\n"
    )
    return staged


_NO_CLOBBER_PUBLISHER = r"""import hashlib,os,stat,sys
temporary,target,expected_size,expected_sha=sys.argv[1:]; expected_size=int(expected_size)
def identity(path):
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd)
  if not stat.S_ISREG(before.st_mode): raise RuntimeError('package is not regular')
  h=hashlib.sha256()
  while True:
   chunk=os.read(fd,1048576)
   if not chunk: break
   h.update(chunk)
  after=os.fstat(fd)
  before_sig=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
  after_sig=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
  if before_sig!=after_sig: raise RuntimeError('package mutated')
  return before.st_size,h.hexdigest()
 finally: os.close(fd)
try:
 if identity(temporary)!=(expected_size,expected_sha): raise RuntimeError('staged package mismatch')
 fd=os.open(temporary,os.O_RDONLY); os.fsync(fd); os.close(fd)
 try: os.link(temporary,target)
 except FileExistsError:
  if identity(target)!=(expected_size,expected_sha): raise RuntimeError('existing package diverges')
 directory=os.open(os.path.dirname(target),os.O_RDONLY)
 try: os.fsync(directory)
 except OSError: pass
 finally: os.close(directory)
finally:
 try: os.unlink(temporary)
 except FileNotFoundError: pass
"""


def _run_required(command: tuple[str, ...], *, runner: CommandRunner) -> CommandResult:
    result = runner(command)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        msg = f"Source Bundle staging command failed ({' '.join(command)}): {detail}"
        raise ValueError(msg)
    return result


def _sha256sum_stdout(stdout: str) -> str:
    digest = stdout.strip().split(maxsplit=1)[0] if stdout.strip() else ""
    if len(digest) != 64:
        msg = "Source Bundle staging verification did not return a sha256 checksum"
        raise ValueError(msg)
    return digest


def _git(repo: Path, *args: str) -> str:
    return _git_bytes(repo, *args).decode().strip()


def _git_bytes(repo: Path, *args: str) -> bytes:
    result = subprocess.run(("git", "-C", str(repo), *args), capture_output=True, check=False)
    if result.returncode != 0:
        msg = result.stderr.decode().strip() or f"git {' '.join(args)} failed"
        raise ValueError(msg)
    return result.stdout


__all__ = [
    "build_and_stage_governed_source_package",
]
