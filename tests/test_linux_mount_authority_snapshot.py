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

"""Security regressions for one-pass Linux mount-authority snapshots."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from bspp.orchestration.runtime.preprocessing import _linux_mount_authority as authority


def test_unrelated_numeric_mount_row_with_malformed_paths_is_ignored(tmp_path: Path) -> None:
    retained = tmp_path / "retained"
    retained.mkdir()
    descriptor = os.open(retained, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        mount_id = authority._statx_identity(descriptor).mount_id
        mountinfo = tmp_path / "mountinfo"
        mountinfo.write_text(
            _mountinfo_row(mount_id, descriptor) + "999999 1 0:1 relative ../relative rw - tmpfs tmpfs rw\n"
        )

        frozen = authority.freeze_directory_mount_authority(
            role="retained",
            path=retained,
            descriptor=descriptor,
            mountinfo_path=mountinfo,
        )

        assert frozen.statx.mount_id == mount_id
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("fault", ["malformed-required", "missing-required", "duplicate-required"])
def test_required_mount_row_must_be_unique_complete_and_well_formed(tmp_path: Path, fault: str) -> None:
    retained = tmp_path / "retained"
    retained.mkdir()
    descriptor = os.open(retained, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        mount_id = authority._statx_identity(descriptor).mount_id
        valid = _mountinfo_row(mount_id, descriptor)
        if fault == "malformed-required":
            payload = f"{mount_id} 1 0:1 relative ../relative rw - tmpfs tmpfs rw\n"
        elif fault == "missing-required":
            payload = _mountinfo_row(mount_id + 1, descriptor)
        else:
            payload = valid + valid
        mountinfo = tmp_path / "mountinfo"
        mountinfo.write_text(payload)

        with pytest.raises(authority.LinuxMountAuthorityError):
            authority.freeze_directory_mount_authority(
                role="retained",
                path=retained,
                descriptor=descriptor,
                mountinfo_path=mountinfo,
            )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("mount_layout", ["shared", "multiple"])
def test_evidence_snapshot_supports_shared_and_multiple_required_mount_ids(
    tmp_path: Path,
    mount_layout: str,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    managed_parent = tmp_path
    temporary_managed_parent: Path | None = None
    if mount_layout == "multiple":
        try:
            temporary_managed_parent = Path(tempfile.mkdtemp(prefix="bspp-mount-authority-", dir="/dev/shm"))
        except OSError as exc:
            pytest.skip(f"a second unprivileged filesystem is unavailable: {exc}")
        managed_parent = temporary_managed_parent
    managed = managed_parent / "managed"
    managed.mkdir()
    evidence_descriptor = os.open(evidence, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    managed_descriptor = os.open(managed, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    mountinfo = _CountingPath("/proc/self/mountinfo")
    try:
        frozen = authority.freeze_evidence_mount_authority(
            evidence_parent_path=evidence,
            evidence_parent_descriptor=evidence_descriptor,
            managed_directories=(("managed", managed, managed_descriptor),),
            mountinfo_path=mountinfo,
        )

        if mount_layout == "shared":
            assert frozen.evidence_parent.statx.mount_id == frozen.managed[0].statx.mount_id
        else:
            assert frozen.evidence_parent.statx.mount_id != frozen.managed[0].statx.mount_id
        assert mountinfo.read_count == 1
    finally:
        os.close(managed_descriptor)
        os.close(evidence_descriptor)
        if temporary_managed_parent is not None:
            shutil.rmtree(temporary_managed_parent)


def test_retained_identity_change_during_mountinfo_window_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = tmp_path / "retained"
    retained.mkdir()
    descriptor = os.open(retained, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    baseline = authority._statx_identity(descriptor)
    real_fstat = os.fstat
    retained_observations = 0

    def changing_fstat(observed_descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal retained_observations
        observed = real_fstat(observed_descriptor)
        if observed_descriptor != descriptor:
            return observed
        retained_observations += 1
        if retained_observations == 1:
            return observed
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino + 1,
            st_uid=observed.st_uid,
            st_mode=observed.st_mode,
        )

    monkeypatch.setattr(os, "fstat", changing_fstat)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(_mountinfo_row(baseline.mount_id, descriptor))
    try:
        with pytest.raises(authority.LinuxMountAuthorityError, match=r"changed|disagrees"):
            authority.freeze_directory_mount_authority(
                role="retained",
                path=retained,
                descriptor=descriptor,
                mountinfo_path=mountinfo,
            )
        assert retained_observations == 2
    finally:
        os.close(descriptor)


def test_visible_path_rebind_during_mountinfo_read_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = tmp_path / "retained"
    retained.mkdir()
    descriptor = os.open(retained, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    retained_before = os.fstat(descriptor)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(_mountinfo_row(authority._statx_identity(descriptor).mount_id, descriptor))
    displaced = tmp_path / "retained-displaced"
    rebound = False

    class RebindingMountinfoPath(type(Path())):
        def read_text(self, *args: object, **kwargs: object) -> str:
            nonlocal rebound
            payload = super().read_text(*args, **kwargs)
            if not rebound:
                retained.rename(displaced)
                retained.mkdir()
                rebound = True
            return payload

    rebinding_mountinfo = RebindingMountinfoPath(mountinfo)
    try:
        with pytest.raises(authority.LinuxMountAuthorityError, match=r"visible.*changed"):
            authority.freeze_directory_mount_authority(
                role="retained",
                path=retained,
                descriptor=descriptor,
                mountinfo_path=rebinding_mountinfo,
            )

        retained_after = os.fstat(descriptor)
        assert rebound
        assert (retained_after.st_dev, retained_after.st_ino) == (
            retained_before.st_dev,
            retained_before.st_ino,
        )
        assert (retained.stat().st_dev, retained.stat().st_ino) != (
            retained_before.st_dev,
            retained_before.st_ino,
        )
    finally:
        os.close(descriptor)


class _CountingPath(type(Path())):
    read_count = 0

    def read_text(self, *args: object, **kwargs: object) -> str:
        self.read_count += 1
        return super().read_text(*args, **kwargs)


def _mountinfo_row(mount_id: int, descriptor: int, *, root: str = "/") -> str:
    info = os.fstat(descriptor)
    major = os.major(info.st_dev)
    minor = os.minor(info.st_dev)
    return f"{mount_id} 1 {major}:{minor} {root} / rw - tmpfs tmpfs rw\n"
