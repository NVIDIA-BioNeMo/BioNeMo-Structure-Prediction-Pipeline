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

"""Tests for the preprocessing Source Bundle producer (provenance-only qualification artifact)."""

from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.preprocessing_source_bundle import (
    PreprocessingSourceBundleError,
    build_and_stage_preprocessing_source_bundle,
    render_preprocessing_source_bundle_yaml,
)
from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.transport import CommandResult
from tests.support.transport_argv import maybe_unwrap_remote_command


def test_source_bundle_build_is_deterministic_and_rebuild_reuses_evidence(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    profile = resolve_cluster_profile("example-cluster", config_path=_write_profiles(tmp_path))

    first = build_and_stage_preprocessing_source_bundle(
        source_repo, build_dir=tmp_path / "build-a", profile=profile, dry_run=True
    )
    second = build_and_stage_preprocessing_source_bundle(
        source_repo, build_dir=tmp_path / "build-b", profile=profile, dry_run=True
    )
    assert first.source_bundle_sha256 == second.source_bundle_sha256
    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()

    rebuilt = build_and_stage_preprocessing_source_bundle(
        source_repo, build_dir=tmp_path / "build-a", profile=profile, dry_run=True
    )
    assert rebuilt.source_bundle_sha256 == first.source_bundle_sha256
    assert rebuilt.archive_path == first.archive_path


def test_source_bundle_modes_come_from_git_index_not_disk(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    (source_repo / "notes.txt").write_text("disk mode must not leak\n")
    (source_repo / "run.sh").write_text("#!/usr/bin/env bash\n")
    _git(source_repo, "add", "notes.txt", "run.sh")
    _git(source_repo, "update-index", "--chmod=+x", "run.sh")
    (source_repo / "notes.txt").chmod(0o660)
    (source_repo / "run.sh").chmod(0o770)
    _git(source_repo, "commit", "-m", "mode fixture")
    profile = resolve_cluster_profile("example-cluster", config_path=_write_profiles(tmp_path))

    record = build_and_stage_preprocessing_source_bundle(
        source_repo, build_dir=tmp_path / "build", profile=profile, dry_run=True
    )

    with tarfile.open(record.archive_path, "r:") as archive:
        modes = {member.name: member.mode for member in archive.getmembers() if member.isfile()}
    assert modes["notes.txt"] == 0o644
    assert modes["run.sh"] == 0o755


def test_source_bundle_requires_clean_source(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    profile = resolve_cluster_profile("example-cluster", config_path=_write_profiles(tmp_path))

    (source_repo / "untracked.txt").write_text("not committed\n")
    with pytest.raises(PreprocessingSourceBundleError, match="requires clean source"):
        build_and_stage_preprocessing_source_bundle(
            source_repo, build_dir=tmp_path / "build", profile=profile, dry_run=True
        )
    (source_repo / "untracked.txt").unlink()

    (source_repo / "README.md").write_text("modified\n")
    with pytest.raises(PreprocessingSourceBundleError, match="requires clean source"):
        build_and_stage_preprocessing_source_bundle(
            source_repo, build_dir=tmp_path / "build", profile=profile, dry_run=True
        )
    assert not (tmp_path / "build").exists() or list((tmp_path / "build").iterdir()) == []


def test_source_bundle_layout_is_uncompressed_manifest_first_whole_tracked_tree(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    (source_repo / "pkg").mkdir()
    (source_repo / "pkg" / "module.py").write_text("X = 1\n")
    _git(source_repo, "add", "pkg/module.py")
    _git(source_repo, "commit", "-m", "add package module")
    (source_repo / "ignored.pyc").write_bytes(b"not tracked")
    (source_repo / ".gitignore").write_text("*.pyc\n")
    _git(source_repo, "add", ".gitignore")
    _git(source_repo, "commit", "-m", "ignore pyc")
    profile = resolve_cluster_profile("example-cluster", config_path=_write_profiles(tmp_path))

    record = build_and_stage_preprocessing_source_bundle(
        source_repo, build_dir=tmp_path / "build", profile=profile, dry_run=True
    )

    # "r:" (not "r:*") proves the payload is an uncompressed tar despite the .tar.zst suffix.
    with tarfile.open(record.archive_path, "r:") as archive:
        members = archive.getmembers()
        manifest_stream = archive.extractfile(members[0])
        assert manifest_stream is not None
        manifest = json.loads(manifest_stream.read())
    names = [member.name for member in members]
    assert names[0] == ".bspp-source-manifest.json"
    payload = names[1:]
    assert payload == sorted(payload)
    assert set(payload) == {".gitignore", "README.md", "pkg/module.py"}
    assert "ignored.pyc" not in names
    for member in members:
        assert member.uid == 0 and member.gid == 0
        assert member.uname == "" and member.gname == ""
        assert member.mtime == 0
    assert manifest["commit"] == record.source_commit
    assert manifest["tree"] == record.source_tree


def test_source_bundle_stages_over_local_transport_and_restages_idempotently(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    bundle_root = tmp_path / "cluster-bundles"
    profile = resolve_cluster_profile(
        "example-cluster", config_path=_write_profiles(tmp_path, source_bundle_root=str(bundle_root))
    )

    record = build_and_stage_preprocessing_source_bundle(source_repo, build_dir=tmp_path / "build", profile=profile)

    commit = _git(source_repo, "rev-parse", "HEAD")
    expected_target = bundle_root / f"bspp-orchestration-{commit}.tar.zst"
    assert record.staged is True
    assert record.source_bundle_id == f"bspp-orchestration-{commit}"
    assert record.source_bundle_path == str(expected_target)
    assert expected_target.is_file()
    assert _sha256(expected_target) == record.source_bundle_sha256
    identity_path = tmp_path / "build" / f"bspp-orchestration-{commit}.source-bundle-identity.json"
    evidence = json.loads(identity_path.read_text())
    assert evidence["preprocessing_source_bundle"]["source_bundle_sha256"] == record.source_bundle_sha256
    assert evidence["preprocessing_source_bundle"]["payload_format"] == "uncompressed-tar"
    assert evidence["preprocessing_source_bundle"]["staged"] is True

    restaged = build_and_stage_preprocessing_source_bundle(source_repo, build_dir=tmp_path / "build-2", profile=profile)
    assert restaged.source_bundle_sha256 == record.source_bundle_sha256
    assert _sha256(expected_target) == record.source_bundle_sha256


def test_source_bundle_stage_refuses_divergent_remote_content(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    bundle_root = tmp_path / "cluster-bundles"
    profile = resolve_cluster_profile(
        "example-cluster", config_path=_write_profiles(tmp_path, source_bundle_root=str(bundle_root))
    )
    commit = _git(source_repo, "rev-parse", "HEAD")
    target = bundle_root / f"bspp-orchestration-{commit}.tar.zst"
    bundle_root.mkdir(parents=True)
    target.write_bytes(b"divergent pre-existing content")

    with pytest.raises(ValueError, match=r"publication|collision|failed"):
        build_and_stage_preprocessing_source_bundle(source_repo, build_dir=tmp_path / "build", profile=profile)
    assert target.read_bytes() == b"divergent pre-existing content"


def test_source_bundle_dry_run_stages_nothing(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    bundle_root = tmp_path / "cluster-bundles"
    profile = resolve_cluster_profile(
        "example-cluster", config_path=_write_profiles(tmp_path, source_bundle_root=str(bundle_root))
    )

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def __call__(self, argv: tuple[str, ...]) -> CommandResult:
            self.calls.append(argv)
            return CommandResult(argv, 0, "", "")

    runner = RecordingRunner()
    record = build_and_stage_preprocessing_source_bundle(
        source_repo, build_dir=tmp_path / "build", profile=profile, runner=runner, dry_run=True
    )

    assert runner.calls == []
    assert record.staged is False
    assert record.archive_path.is_file()
    assert not bundle_root.exists()
    assert (tmp_path / "build" / f"{record.source_bundle_id}.source-bundle-identity.json").is_file()
    rendered = yaml.safe_load(render_preprocessing_source_bundle_yaml(record))
    assert rendered["preprocessing_source_bundle"]["source_bundle_sha256"] == record.source_bundle_sha256


def test_source_bundle_refuses_subdirectory_of_checkout(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    (source_repo / "packages").mkdir()
    (source_repo / "packages" / "placeholder.txt").write_text("subdirectory content\n")
    _git(source_repo, "add", "packages/placeholder.txt")
    _git(source_repo, "commit", "-m", "add packages directory")
    profile = resolve_cluster_profile("example-cluster", config_path=_write_profiles(tmp_path))

    with pytest.raises(PreprocessingSourceBundleError, match="checkout root"):
        build_and_stage_preprocessing_source_bundle(
            source_repo / "packages", build_dir=tmp_path / "build", profile=profile, dry_run=True
        )


def test_source_bundle_stage_refuses_real_conflicting_profile_pin(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    commit = _git(source_repo, "rev-parse", "HEAD")
    bundle_root = tmp_path / "cluster-bundles"
    profile = resolve_cluster_profile(
        "example-cluster",
        config_path=_write_profiles(
            tmp_path,
            source_bundle_root=str(bundle_root),
            preprocessing_runtime={"source_commit": commit, "source_bundle_sha256": "a" * 64},
        ),
    )

    with pytest.raises(PreprocessingSourceBundleError, match="already pins a different source_bundle_sha256"):
        build_and_stage_preprocessing_source_bundle(source_repo, build_dir=tmp_path / "build", profile=profile)
    assert not bundle_root.exists()


def test_source_bundle_stage_allows_placeholder_pin_bootstrap(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    commit = _git(source_repo, "rev-parse", "HEAD")
    bundle_root = tmp_path / "cluster-bundles"
    profile = resolve_cluster_profile(
        "example-cluster",
        config_path=_write_profiles(
            tmp_path,
            source_bundle_root=str(bundle_root),
            preprocessing_runtime={"source_commit": commit, "source_bundle_sha256": "0" * 64},
        ),
    )

    record = build_and_stage_preprocessing_source_bundle(source_repo, build_dir=tmp_path / "build", profile=profile)

    assert record.staged is True
    target = bundle_root / f"bspp-orchestration-{commit}.tar.zst"
    assert target.is_file()
    assert _sha256(target) == record.source_bundle_sha256


def test_source_bundle_requires_source_bundle_root(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    profile = resolve_cluster_profile("example-cluster", config_path=_write_profiles(tmp_path, source_bundle_root=None))

    with pytest.raises(PreprocessingSourceBundleError, match="source_bundle_root"):
        build_and_stage_preprocessing_source_bundle(
            source_repo, build_dir=tmp_path / "build", profile=profile, dry_run=True
        )


def test_source_bundle_stage_over_ssh_wraps_every_step_in_transport(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    profile = resolve_cluster_profile(
        "example-cluster",
        config_path=_write_profiles(tmp_path, transport="ssh", ssh_target="login", source_bundle_root="/bundles"),
    )
    commit = _git(source_repo, "rev-parse", "HEAD")
    expected_sha: str | None = None

    class FakeSshRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def __call__(self, argv: tuple[str, ...]) -> CommandResult:
            self.calls.append(argv)
            stdout = (
                f"{expected_sha}  staged\n"
                if any("sha256sum" in maybe_unwrap_remote_command(part) for part in argv)
                else ""
            )
            return CommandResult(argv, 0, stdout, "")

    runner = FakeSshRunner()
    # Build once locally to learn the digest the fake remote must report.
    preview = build_and_stage_preprocessing_source_bundle(
        source_repo, build_dir=tmp_path / "build", profile=profile, dry_run=True
    )
    expected_sha = preview.source_bundle_sha256

    record = build_and_stage_preprocessing_source_bundle(
        source_repo, build_dir=tmp_path / "build", profile=profile, runner=runner
    )

    assert record.staged is True
    assert record.source_bundle_path == f"/bundles/bspp-orchestration-{commit}.tar.zst"
    assert any(call[:2] == ("scp", str(record.archive_path)) for call in runner.calls)
    assert all(call[0] in {"ssh", "scp"} for call in runner.calls)
    published = [call for call in runner.calls if call[0] == "ssh" and "ln --" in maybe_unwrap_remote_command(call[-1])]
    assert published, runner.calls
    published_command = maybe_unwrap_remote_command(published[0][-1])
    assert f".bspp-stage-preprocessing-source-bundle-{commit[:12]}.tmp" in published_command


def test_source_bundle_rebuild_conflict_with_divergent_local_archive_fails_closed(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    profile = resolve_cluster_profile("example-cluster", config_path=_write_profiles(tmp_path))
    first = build_and_stage_preprocessing_source_bundle(
        source_repo, build_dir=tmp_path / "build", profile=profile, dry_run=True
    )
    first.archive_path.write_bytes(b"corrupted local evidence")

    with pytest.raises(PreprocessingSourceBundleError, match="not byte-deterministic"):
        build_and_stage_preprocessing_source_bundle(
            source_repo, build_dir=tmp_path / "build", profile=profile, dry_run=True
        )


def test_bsppctl_stage_source_bundle_dry_run_prints_profile_pins(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config_path),
            "runtime",
            "preprocessing",
            "stage-source-bundle",
            "--profile",
            "example-cluster",
            "--source-repo",
            str(source_repo),
            "--build-dir",
            str(tmp_path / "build"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    document = yaml.safe_load(result.output)["preprocessing_source_bundle"]
    commit = _git(source_repo, "rev-parse", "HEAD")
    assert document["source_commit"] == commit
    assert document["source_bundle_id"] == f"bspp-orchestration-{commit}"
    assert document["source_bundle_path"].endswith(f"/bundles/bspp-orchestration-{commit}.tar.zst")
    assert len(document["source_bundle_sha256"]) == 64
    assert document["payload_format"] == "uncompressed-tar"
    assert document["staged"] is False


def test_bsppctl_stage_source_bundle_rejects_dirty_source(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    (source_repo / "dirty.txt").write_text("untracked\n")
    config_path = _write_profiles(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config_path),
            "runtime",
            "preprocessing",
            "stage-source-bundle",
            "--profile",
            "example-cluster",
            "--source-repo",
            str(source_repo),
            "--build-dir",
            str(tmp_path / "build"),
        ],
    )

    assert result.exit_code != 0
    assert "requires clean source" in result.output


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
    preprocessing_runtime: dict[str, str] | None = None,
) -> Path:
    cluster: dict[str, object] = {
        "owner": "tester",
        "project_root": "/project",
        "output_root": "/output",
        "staging_root": "/staging",
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
    if preprocessing_runtime is not None:
        block = {
            "cluster_image_path": "/images/bspp-orchestration-preprocessing.sqsh",
            "cluster_image_sha256": "b" * 64,
            "oci_digest": f"sha256:{'c' * 64}",
            "image_lock_sha256": "d" * 64,
            "contract_wheel_sha256": "e" * 64,
            "runtime_wheel_sha256": "f" * 64,
            "control_wheel_sha256": "a" * 64,
            "source_commit": "1" * 40,
            "source_bundle_sha256": "2" * 64,
            "colabfold_version": "1.6.2",
            "mmseqs_version": "15.6f452",
            "rsync_version": "3.4.4",
            "cuda_version": "12.6",
        }
        block.update(preprocessing_runtime)
        cluster["preprocessing_runtime"] = block
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump({"clusters": {"example-cluster": cluster}}, sort_keys=False))
    return path


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", "-C", str(repo), *args), capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    import hashlib

    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def test_bsppctl_source_bundle_dry_run_preserves_tracked_long_names(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    relative = "evidence/operator-transfer-evidence-" + "a" * 64 + ".verify.log"
    payload = source_repo / relative
    payload.parent.mkdir()
    payload.write_bytes(b"tracked acceptance evidence\n")
    _git(source_repo, "add", relative)
    _git(source_repo, "commit", "-m", "retain long-named evidence")
    config_path = _write_profiles(tmp_path)
    records = []
    for build_name in ("build-a", "build-b"):
        result = CliRunner().invoke(
            cli,
            [
                "--config",
                str(config_path),
                "runtime",
                "preprocessing",
                "stage-source-bundle",
                "--profile",
                "example-cluster",
                "--source-repo",
                str(source_repo),
                "--build-dir",
                str(tmp_path / build_name),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        records.append(yaml.safe_load(result.output)["preprocessing_source_bundle"])
    assert records[0]["source_bundle_sha256"] == records[1]["source_bundle_sha256"]
    assert all(record["staged"] is False for record in records)
    with tarfile.open(records[0]["archive_path"], "r:") as archive:
        member = archive.extractfile(relative)
        assert member is not None
        assert member.read() == payload.read_bytes()
