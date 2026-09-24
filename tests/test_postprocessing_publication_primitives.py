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

"""Shared behavioral contract for private Runtime and Control publishers."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from types import ModuleType

import pytest

import bspp.orchestration.control.postprocessing_evidence_transfer as control_publication
import bspp.orchestration.runtime.postprocessing.finalization_io as runtime_publication

FALLBACK_ERRNOS = tuple(
    sorted({errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)})
)


@pytest.fixture(params=(runtime_publication, control_publication), ids=("runtime", "control"))
def publication_module(request: pytest.FixtureRequest) -> ModuleType:
    return request.param


@pytest.mark.parametrize("collision_kind", ("directory", "file", "dangling-symlink"))
def test_fallback_detects_every_existing_destination(
    tmp_path: Path, publication_module: ModuleType, collision_kind: str
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    if collision_kind == "directory":
        destination.mkdir()
    elif collision_kind == "file":
        destination.write_text("occupied")
    else:
        destination.symlink_to(tmp_path / "missing")

    with pytest.raises(FileExistsError):
        publication_module._rename_directory_no_replace(source, destination, renameat2=_unsupported)
    assert source.is_dir()


@pytest.mark.parametrize("fallback_errno", FALLBACK_ERRNOS)
def test_defined_capability_errors_activate_fallback(
    tmp_path: Path, publication_module: ModuleType, fallback_errno: int
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()

    def unsupported(*_args: object) -> None:
        raise OSError(fallback_errno, os.strerror(fallback_errno))

    publication_module._rename_directory_no_replace(source, destination, renameat2=unsupported)
    assert destination.is_dir()
    assert not source.exists()


def test_noncapability_renameat2_error_never_falls_back(
    tmp_path: Path, publication_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()

    def cross_device(*_args: object) -> None:
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV))

    monkeypatch.setattr(publication_module.fcntl, "flock", lambda *_args: pytest.fail("fallback lock used"))
    with pytest.raises(OSError) as error:
        publication_module._rename_directory_no_replace(source, destination, renameat2=cross_device)
    assert error.value.errno == errno.EXDEV
    assert source.is_dir() and not destination.exists()


def test_fallback_lock_failure_is_fail_closed(
    tmp_path: Path, publication_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()

    def lock_failure(*_args: object) -> None:
        raise OSError(errno.EIO, "lock failed")

    monkeypatch.setattr(publication_module.fcntl, "flock", lock_failure)
    with pytest.raises(OSError, match="lock failed"):
        publication_module._rename_directory_no_replace(source, destination, renameat2=_unsupported)
    assert source.is_dir() and not destination.exists()


def test_fallback_rename_failure_is_fail_closed(
    tmp_path: Path, publication_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()

    def rename_failure(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EIO, "rename failed")

    monkeypatch.setattr(publication_module.os, "rename", rename_failure)
    with pytest.raises(OSError, match="rename failed"):
        publication_module._rename_directory_no_replace(source, destination, renameat2=_unsupported)
    assert source.is_dir() and not destination.exists()


def test_fallback_recheck_failure_is_fail_closed(
    tmp_path: Path, publication_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    real_stat = os.stat
    source_checks = 0

    def failing_recheck(path: object, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal source_checks
        if path == "source" and kwargs.get("dir_fd") is not None:
            source_checks += 1
            if source_checks == 2:
                raise OSError(errno.EIO, "recheck failed")
        return real_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(publication_module.os, "stat", failing_recheck)
    with pytest.raises(OSError, match="recheck failed"):
        publication_module._rename_directory_no_replace(source, destination, renameat2=_unsupported)
    assert source.is_dir() and not destination.exists()


def test_fallback_parent_fsync_failure_surfaces_after_atomic_publication(
    tmp_path: Path, publication_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    monkeypatch.setattr(publication_module.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError(errno.EIO, "fsync")))
    with pytest.raises(OSError, match="fsync"):
        publication_module._rename_directory_no_replace(source, destination, renameat2=_unsupported)
    assert destination.is_dir() and not source.exists()


def test_fallback_unlock_failure_surfaces_after_atomic_publication(
    tmp_path: Path, publication_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    real_flock = publication_module.fcntl.flock

    def failing_unlock(descriptor: int, operation: int) -> None:
        if operation == publication_module.fcntl.LOCK_UN:
            raise OSError(errno.EIO, "unlock failed")
        real_flock(descriptor, operation)

    monkeypatch.setattr(publication_module.fcntl, "flock", failing_unlock)
    with pytest.raises(OSError, match="unlock failed"):
        publication_module._rename_directory_no_replace(source, destination, renameat2=_unsupported)
    assert destination.is_dir() and not source.exists()


def _unsupported(*_args: object) -> None:
    raise OSError(errno.EINVAL, os.strerror(errno.EINVAL))
