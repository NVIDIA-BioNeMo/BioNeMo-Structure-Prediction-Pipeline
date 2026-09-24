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

"""Tests for governed Source Package staging (kept after legacy provisioning removal)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.source_package_staging import (
    _NO_CLOBBER_PUBLISHER,
    build_and_stage_governed_source_package,
)
from bspp.orchestration.control.transport import CommandResult
from tests.support.transport_argv import maybe_unwrap_remote_command


class FakeStageRunner:
    def __init__(self, sha256: str) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.sha256 = sha256

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        stdout = f"{self.sha256}  {argv[-1]}\n" if any("sha256sum" in part for part in argv) else ""
        return CommandResult(argv=argv, returncode=0, stdout=stdout, stderr="")


def test_governed_source_package_stages_exact_identity_without_local_remote_target(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    remote_target = tmp_path / "remote-not-local" / "source.tar"
    stage_runner = FakeStageRunner("placeholder")
    first = build_and_stage_governed_source_package(
        source_repo,
        build_dir=tmp_path / "build",
        target_path=remote_target,
        profile=profile,
        runner=stage_runner,
        verify_remote_digest=False,
    )
    assert first.package_path == remote_target.absolute()
    assert not remote_target.exists()
    assert (tmp_path / "build" / "source-package-identity.json").exists()
    rendered_calls = "\n".join(" ".join(call) for call in stage_runner.calls)
    assert ".tmp." in rendered_calls
    assert " mv " not in f" {rendered_calls} "


def test_no_clobber_publisher_is_concurrent_idempotent_and_rejects_divergence_or_symlink(tmp_path: Path) -> None:
    payload = b"exact package"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    target = tmp_path / "published.tar"

    def command(temporary: Path) -> tuple[str, ...]:
        temporary.write_bytes(payload)
        return (
            sys.executable,
            "-I",
            "-S",
            "-c",
            _NO_CLOBBER_PUBLISHER,
            str(temporary),
            str(target),
            str(len(payload)),
            digest,
        )

    first = subprocess.Popen(command(tmp_path / "one.tmp"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    second = subprocess.Popen(command(tmp_path / "two.tmp"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert first.wait() == 0, first.stderr.read().decode() if first.stderr else ""
    assert second.wait() == 0, second.stderr.read().decode() if second.stderr else ""
    assert target.read_bytes() == payload

    target.write_bytes(b"divergent!!!")
    assert subprocess.run(command(tmp_path / "three.tmp"), capture_output=True).returncode != 0
    target.unlink()
    target.symlink_to(tmp_path / "missing")
    assert subprocess.run(command(tmp_path / "four.tmp"), capture_output=True).returncode != 0


def test_governed_stage_cleans_unique_remote_temp_when_copy_fails(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    profile = resolve_cluster_profile(
        "example-cluster", config_path=_write_profiles(tmp_path, transport="ssh", ssh_target="login")
    )

    class CopyFailureRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def __call__(self, argv: tuple[str, ...]) -> CommandResult:
            self.calls.append(argv)
            return CommandResult(argv, 1 if argv[0] == "scp" else 0, "", "copy failed" if argv[0] == "scp" else "")

    runner = CopyFailureRunner()
    with pytest.raises(ValueError, match="copy failed"):
        build_and_stage_governed_source_package(
            source_repo,
            build_dir=tmp_path / "build",
            target_path=Path("/remote/source.tar"),
            profile=profile,
            runner=runner,
        )
    assert any(
        call[0:2] == ("ssh", "login")
        and "rm -f" in maybe_unwrap_remote_command(call[-1])
        and ".tmp." in maybe_unwrap_remote_command(call[-1])
        for call in runner.calls
    )


def _init_git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "config", "user.email", "tester@example.com")
    _git(path, "config", "user.name", "Tester")
    (path / "README.md").write_text("hello\n")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    return path


def _write_profiles(
    tmp_path: Path,
    *,
    transport: str = "local-slurm",
    ssh_target: str | None = None,
    source_bundle_root: str | None = "/bundles",
    runtime_image_cache_root: str | None = "/image-cache",
) -> Path:
    cluster = {
        "owner": "tester",
        "project_root": "/project",
        "output_root": "/output",
        "staging_root": "/staging",
        "afdb_toolkit_repo": "/toolkit",
        "orchestration_repo": "/orchestration",
        "image": "/images/bspp.sqsh",
        "transport": transport,
        "account": "user-account",
        "resources": {
            "gpu_worker": {
                "partition": "gpu",
                "cpus_per_task": 30,
                "memory": "128G",
                "time": "04:00:00",
                "gres": "gpu:1",
            }
        },
    }
    if ssh_target is not None:
        cluster["ssh_target"] = ssh_target
    if source_bundle_root is not None:
        cluster["source_bundle_root"] = source_bundle_root
    if runtime_image_cache_root is not None:
        cluster["runtime_image_cache_root"] = runtime_image_cache_root
    return _write_yaml(
        tmp_path / "profiles.yaml",
        {"clusters": {"example-cluster": cluster}},
    )


def _write_yaml(path: Path, data: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", "-C", str(repo), *args), capture_output=True, text=True, check=True)
    return result.stdout.strip()
