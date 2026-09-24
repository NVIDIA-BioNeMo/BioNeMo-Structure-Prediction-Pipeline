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

"""Governed source-package identity and archive safety tests."""

from __future__ import annotations

import io
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from bspp.orchestration.contract.source_package import (
    MAX_MANIFEST_BYTES,
    MAX_PACKAGE_BYTES,
    SourcePackageIdentity,
    build_source_package,
    source_package_identity_from_mapping,
    verify_source_package,
)


def test_source_package_identity_round_trips_and_rejects_future_versions(tmp_path: Path) -> None:
    archive = (tmp_path / "source.tar").resolve()
    identity = SourcePackageIdentity(
        format_version=1,
        format="bspp-tar-v1",
        verifier="safe-tar-v1",
        package_path=archive,
        package_size_bytes=12,
        package_sha256="a" * 64,
        manifest_sha256="b" * 64,
        commit="c" * 40,
        tree="d" * 40,
    )
    assert source_package_identity_from_mapping(identity.to_mapping()) == identity
    with pytest.raises(ValueError, match="Unsupported SourcePackageIdentity format_version 2"):
        source_package_identity_from_mapping({**identity.to_mapping(), "format_version": 2})
    with pytest.raises(ValueError, match="absolute"):
        SourcePackageIdentity(**{**identity.__dict__, "package_path": Path("relative.tar")})


def test_bspp_source_package_is_deterministic_and_extracts_verified_payload(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "bin").mkdir()
    (source / "README.md").write_text("hello\n")
    executable = source / "bin" / "run"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)

    first = build_source_package(source, tmp_path / "first.tar", commit="a" * 40, tree="b" * 40)
    second = build_source_package(source, tmp_path / "second.tar", commit="a" * 40, tree="b" * 40)

    assert first.package_sha256 == second.package_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    destination = tmp_path / "extract"
    verify_source_package(first, destination=destination)
    assert (destination / "README.md").read_text() == "hello\n"
    assert (destination / "bin" / "run").stat().st_mode & 0o111


def test_source_package_verifier_fails_closed_for_unknown_format_and_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_text("payload")
    identity = build_source_package(source, tmp_path / "source.tar", commit="a" * 40, tree="b" * 40)
    with pytest.raises(ValueError, match="Unknown source package format"):
        verify_source_package(SourcePackageIdentity(**{**identity.__dict__, "format": "future-v2"}))
    identity.package_path.write_bytes(identity.package_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match=r"package (size|SHA256)"):
        verify_source_package(identity)


@pytest.mark.parametrize("member_name", ["/absolute", "../traversal", "back\\slash"])
def test_source_package_rejects_ambiguous_or_escaping_member_names(tmp_path: Path, member_name: str) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
        info = tarfile.TarInfo(member_name)
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    identity = SourcePackageIdentity.for_existing_archive(
        archive,
        format="safe-tar-v1",
        verifier="safe-tar-v1",
        manifest_sha256="0" * 64,
        commit="a" * 40,
        tree="b" * 40,
    )
    with pytest.raises(ValueError, match=r"unsafe|manifest"):
        verify_source_package(identity)


def test_source_package_rejects_links_and_preexisting_destination(tmp_path: Path) -> None:
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "target"
        tar.addfile(info)
    identity = SourcePackageIdentity.for_existing_archive(
        archive,
        format="safe-tar-v1",
        verifier="safe-tar-v1",
        manifest_sha256="0" * 64,
        commit="a" * 40,
        tree="b" * 40,
    )
    with pytest.raises(ValueError, match="regular files"):
        verify_source_package(identity)

    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_text("payload")
    valid = build_source_package(source, tmp_path / "valid.tar", commit="a" * 40, tree="b" * 40)
    destination = tmp_path / "exists"
    destination.mkdir()
    with pytest.raises(ValueError, match="must not already exist"):
        verify_source_package(valid, destination=destination)


def test_tracked_git_package_excludes_git_and_ignored_untracked_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(("git", "-C", str(repo), "init", "-q"), check=True)
    subprocess.run(("git", "-C", str(repo), "config", "user.email", "test@example.invalid"), check=True)
    subprocess.run(("git", "-C", str(repo), "config", "user.name", "Test"), check=True)
    (repo / ".gitignore").write_text("ignored.bin\nbuild/\n")
    tracked = repo / "packages/orchestration-runtime/src/tracked.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked\n")
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(("git", "-C", str(repo), "commit", "-qm", "fixture"), check=True)
    (repo / "ignored.bin").write_bytes(b"ignored secret")
    commit = subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD"), text=True).strip()
    tree = subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD^{tree}"), text=True).strip()

    first = build_source_package(repo, repo / "build" / "first.tar", commit=commit, tree=tree, tracked_git=True)
    second = build_source_package(repo, repo / "build" / "second.tar", commit=commit, tree=tree, tracked_git=True)

    assert first.package_sha256 == second.package_sha256
    with tarfile.open(first.package_path) as archive:
        names = archive.getnames()
    assert "packages/orchestration-runtime/src/tracked.txt" in names
    assert "ignored.bin" not in names
    assert not any(name == ".git" or name.startswith(".git/") for name in names)


def test_tracked_git_package_includes_only_governed_runtime_payload(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(("git", "-C", str(repo), "init", "-q"), check=True)
    subprocess.run(("git", "-C", str(repo), "config", "user.email", "test@example.invalid"), check=True)
    subprocess.run(("git", "-C", str(repo), "config", "user.name", "Test"), check=True)
    members = {
        "packages/orchestration-contract/src/contract.py": b"contract",
        "packages/orchestration-runtime/src/runtime.py": b"runtime",
        "containers/scripts/slurm-tar-payload-parity.sh": b"#!/bin/sh\n",
        "containers/scripts/slurm-semantic-acceptance.sh": b"#!/bin/sh\n",
        "devdocs/evidence/history.json": b"history",
        "skills/bspp-homodimer/SKILL.md": b"skill",
        "examples/sample.txt": b"sample",
        "docs/superpowers/plan.md": b"plan",
        "profiles/named-cluster.yaml": b"cluster",
    }
    for name, data in members.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(("git", "-C", str(repo), "commit", "-qm", "fixture"), check=True)
    commit = subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD"), text=True).strip()
    tree = subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD^{tree}"), text=True).strip()

    identity = build_source_package(repo, tmp_path / "runtime.tar", commit=commit, tree=tree, tracked_git=True)

    with tarfile.open(identity.package_path) as archive:
        packaged = set(archive.getnames())
    assert packaged == {
        ".bspp-source-manifest.json",
        "packages/orchestration-contract/src/contract.py",
        "packages/orchestration-runtime/src/runtime.py",
        "containers/scripts/slurm-tar-payload-parity.sh",
        "containers/scripts/slurm-semantic-acceptance.sh",
    }


def test_source_package_rejects_oversized_sparse_archive_before_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = tmp_path / "oversized.tar"
    with archive.open("wb") as stream:
        stream.truncate(MAX_PACKAGE_BYTES + 1)
    identity = SourcePackageIdentity(
        format="safe-tar-v1",
        verifier="safe-tar-v1",
        package_path=archive,
        package_size_bytes=MAX_PACKAGE_BYTES + 1,
        package_sha256="0" * 64,
        manifest_sha256="0" * 64,
        commit="a" * 40,
        tree="b" * 40,
    )
    monkeypatch.setattr(os, "read", lambda *_args: pytest.fail("oversized package was read"))
    with pytest.raises(ValueError, match="size bound"):
        verify_source_package(identity)


def test_source_package_rejects_oversized_declared_manifest_before_allocation(tmp_path: Path) -> None:
    archive = tmp_path / "huge-manifest.tar"
    info = tarfile.TarInfo(".bspp-source-manifest.json")
    info.size = MAX_MANIFEST_BYTES + 1
    archive.write_bytes(info.tobuf(format=tarfile.USTAR_FORMAT) + b"\0" * 1024)
    identity = SourcePackageIdentity.for_existing_archive(
        archive,
        format="safe-tar-v1",
        verifier="safe-tar-v1",
        manifest_sha256="0" * 64,
        commit="a" * 40,
        tree="b" * 40,
    )
    with pytest.raises(ValueError, match="manifest exceeds size bound"):
        verify_source_package(identity)


def test_tracked_subtree_package_uses_relative_members_and_same_canonical_algorithm(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(("git", "-C", str(repo), "init", "-q"), check=True)
    subprocess.run(("git", "-C", str(repo), "config", "user.email", "test@example.invalid"), check=True)
    subprocess.run(("git", "-C", str(repo), "config", "user.name", "Test"), check=True)
    toolkit = repo / "toolkit"
    toolkit.mkdir()
    (toolkit / "run.py").write_text("print('ok')\n")
    (repo / "outside.txt").write_text("outside\n")
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(("git", "-C", str(repo), "commit", "-qm", "fixture"), check=True)
    commit = subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD"), text=True).strip()
    tree = subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD:toolkit"), text=True).strip()

    first = build_source_package(
        repo,
        tmp_path / "first.tar",
        commit=commit,
        tree=tree,
        tracked_git=True,
        git_subtree="toolkit",
        governed_runtime_only=False,
    )
    second = build_source_package(
        repo,
        tmp_path / "second.tar",
        commit=commit,
        tree=tree,
        tracked_git=True,
        git_subtree="toolkit",
        governed_runtime_only=False,
    )
    assert first.package_sha256 == second.package_sha256
    with tarfile.open(first.package_path) as archive:
        assert set(archive.getnames()) == {".bspp-source-manifest.json", "run.py"}


@pytest.mark.parametrize("change", ["modified", "staged-addition", "deletion"])
def test_toolkit_tracked_package_ignores_untracked_files_but_rejects_tracked_changes(
    tmp_path: Path, change: str
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(("git", "-C", str(repo), "init", "-q"), check=True)
    subprocess.run(("git", "-C", str(repo), "config", "user.email", "test@example.invalid"), check=True)
    subprocess.run(("git", "-C", str(repo), "config", "user.name", "Test"), check=True)
    (repo / ".gitignore").write_text("*.ignored\n")
    (repo / "tracked.py").write_text("VALUE = 1\n")
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(("git", "-C", str(repo), "commit", "-qm", "fixture"), check=True)
    (repo / "ordinary.tmp").write_text("ordinary\n")
    (repo / "cache.ignored").write_text("ignored\n")
    commit = subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD"), text=True).strip()
    tree = subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD^{tree}"), text=True).strip()

    identity = build_source_package(
        repo,
        tmp_path / "toolkit.tar",
        commit=commit,
        tree=tree,
        tracked_git=True,
        governed_runtime_only=False,
        allow_untracked=True,
    )
    with tarfile.open(identity.package_path) as archive:
        assert "ordinary.tmp" not in archive.getnames()
        assert "cache.ignored" not in archive.getnames()

    if change == "modified":
        (repo / "tracked.py").write_text("VALUE = 2\n")
    elif change == "staged-addition":
        (repo / "added.py").write_text("added\n")
        subprocess.run(("git", "-C", str(repo), "add", "added.py"), check=True)
    else:
        (repo / "tracked.py").unlink()
    with pytest.raises(ValueError, match="clean tracked repository"):
        build_source_package(
            repo,
            tmp_path / "changed.tar",
            commit=commit,
            tree=tree,
            tracked_git=True,
            governed_runtime_only=False,
            allow_untracked=True,
        )


def test_orchestration_role_rejects_self_consistent_disallowed_payload(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "devdocs").mkdir()
    (source / "devdocs/evidence.json").write_text("history\n")
    generic = build_source_package(source, tmp_path / "forged.tar", commit="a" * 40, tree="b" * 40)
    forged = SourcePackageIdentity(**{**generic.__dict__, "package_role": "orchestration"})
    with pytest.raises(ValueError, match="outside governed runtime policy"):
        verify_source_package(forged, expected_role="orchestration")


@pytest.mark.parametrize(
    "relative",
    [
        "evidence/operator-transfer-evidence-" + "a" * 64 + ".verify.log",
        "/".join(("a" * 80, "b" * 80, "c" * 80, "payload.txt")),
    ],
)
def test_source_package_long_paths_are_deterministic_and_verified(tmp_path: Path, relative: str) -> None:
    source = tmp_path / "source"
    payload = source / relative
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"long-path payload\n")
    payload.chmod(0o644)

    first = build_source_package(source, tmp_path / "first.tar", commit="a" * 40, tree="b" * 40)
    second = build_source_package(source, tmp_path / "second.tar", commit="a" * 40, tree="b" * 40)

    assert first.package_sha256 == second.package_sha256
    assert first.package_path.read_bytes() == second.package_path.read_bytes()
    destination = tmp_path / "verified"
    verify_source_package(first, destination=destination)
    assert (destination / relative).read_bytes() == payload.read_bytes()
    with tarfile.open(first.package_path, "r:") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == [".bspp-source-manifest.json", relative]
    assert all(member.isfile() and not member.pax_headers for member in members)


def test_source_package_short_paths_preserve_historical_ustar_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_bytes(b"hello\n")
    (source / "README.md").chmod(0o644)
    (source / "bin").mkdir()
    executable = source / "bin" / "run"
    executable.write_bytes(b"#!/bin/sh\n")
    executable.chmod(0o755)

    identity = build_source_package(source, tmp_path / "source.tar", commit="a" * 40, tree="b" * 40)

    # Captured from the original USTAR producer with these canonical inputs.
    assert identity.package_sha256 == "3b6e1c12f3aea4fa9914efa2a4e487eda7987fc5518b04c6a2277522abba02fb"
    verify_source_package(identity)


def test_source_package_verifier_still_rejects_pax_metadata(tmp_path: Path) -> None:
    path = tmp_path / "pax.tar"
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive:
        member = tarfile.TarInfo(".bspp-source-manifest.json")
        member.pax_headers = {"comment": "forbidden metadata"}
        archive.addfile(member, io.BytesIO(b""))
    identity = SourcePackageIdentity.for_existing_archive(
        path,
        format="safe-tar-v1",
        verifier="safe-tar-v1",
        manifest_sha256="0" * 64,
        commit="a" * 40,
        tree="b" * 40,
    )

    with pytest.raises(ValueError, match="without PAX/sparse metadata"):
        verify_source_package(identity)


def test_source_package_verifier_rejects_gnu_long_name_traversal(tmp_path: Path) -> None:
    path = tmp_path / "gnu-unsafe.tar"
    with tarfile.open(path, "w", format=tarfile.GNU_FORMAT) as archive:
        member = tarfile.TarInfo("../" + "a" * 102)
        archive.addfile(member, io.BytesIO(b""))
    identity = SourcePackageIdentity.for_existing_archive(
        path,
        format="safe-tar-v1",
        verifier="safe-tar-v1",
        manifest_sha256="0" * 64,
        commit="a" * 40,
        tree="b" * 40,
    )

    with pytest.raises(ValueError, match="unsafe path"):
        verify_source_package(identity)
