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

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from bspp.orchestration.contract.submission_evidence import SubmissionToken
from bspp.orchestration.control.submission_coordinator import SubmissionCoordinator


def _token(*, script_sha256: str = "b" * 64) -> SubmissionToken:
    return SubmissionToken.create(
        run_id="run-1",
        runspec_sha256="a" * 64,
        step_index=3,
        step_name="slurm",
        slice_id="0-9",
        attempt=1,
        script_sha256=script_sha256,
        bootstrap_sha256="c" * 64,
        control_state_sha256="d" * 64,
        runtime_qualification_sha256="e" * 64,
    )


def test_directory_fsync_uses_nofollow_directory_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bspp.orchestration.control.submission_coordinator as module

    original_open = os.open
    observed_flags: list[int] = []

    def recording_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        observed_flags.append(flags)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", recording_open)
    module._fsync_directory(tmp_path)

    assert observed_flags == [os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)]


def test_first_claim_is_atomic_and_restart_returns_prior_token(tmp_path: Path) -> None:
    first = SubmissionCoordinator(tmp_path).claim(_token())
    second = SubmissionCoordinator(tmp_path).claim(_token())
    assert first.created is True
    assert second.created is False
    assert second.record == first.record
    assert len(tuple((tmp_path / "submission-coordinator").glob("*.json"))) == 1


def test_uncertain_submit_fails_closed_and_never_authorizes_second_submit(tmp_path: Path) -> None:
    calls = 0

    def submit() -> str:
        nonlocal calls
        calls += 1
        raise TimeoutError("scheduler response lost")

    coordinator = SubmissionCoordinator(tmp_path)
    claim = coordinator.claim(_token())
    assert claim.created
    with pytest.raises(TimeoutError):
        submit()
    restarted = SubmissionCoordinator(tmp_path).claim(_token())
    assert restarted.created is False
    assert restarted.record.status == "prepared"
    assert restarted.record.job_id is None
    assert calls == 1


def test_submitted_job_is_bound_once_and_reused_after_restart(tmp_path: Path) -> None:
    coordinator = SubmissionCoordinator(tmp_path)
    coordinator.claim(_token())
    submitted = coordinator.bind_job(_token(), job_id="12345", scheduler_status="PENDING")
    assert submitted.job_id == "12345"
    restarted = SubmissionCoordinator(tmp_path).claim(_token())
    assert restarted.record.job_id == "12345"
    assert restarted.record.status == "submitted"
    assert coordinator.bind_job(_token(), job_id="12345", scheduler_status="PENDING") == submitted
    with pytest.raises(ValueError, match="already bound to scheduler job 12345"):
        coordinator.bind_job(_token(), job_id="99999", scheduler_status="PENDING")
    with pytest.raises(ValueError, match="already bound to scheduler job 12345"):
        coordinator.bind_job(_token(), job_id="12345", scheduler_status="RUNNING")


@pytest.mark.parametrize(("job_id", "scheduler_status"), [("", "PENDING"), ("123", "")])
def test_bind_job_rejects_empty_scheduler_identity_without_corrupting_claim(
    tmp_path: Path, job_id: str, scheduler_status: str
) -> None:
    coordinator = SubmissionCoordinator(tmp_path)
    prepared = coordinator.claim(_token()).record

    with pytest.raises(ValueError, match="must be a non-empty string"):
        coordinator.bind_job(_token(), job_id=job_id, scheduler_status=scheduler_status)

    assert coordinator.claim(_token()).record == prepared


def test_same_slot_with_changed_script_fails_closed(tmp_path: Path) -> None:
    coordinator = SubmissionCoordinator(tmp_path)
    coordinator.claim(_token())
    with pytest.raises(ValueError, match="submission slot already claimed with a different token"):
        coordinator.claim(_token(script_sha256="c" * 64))


def test_concurrent_claims_create_exactly_one_authorization(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = tuple(executor.map(lambda _: SubmissionCoordinator(tmp_path).claim(_token()), range(32)))
    assert sum(claim.created for claim in claims) == 1
    assert len({claim.record.token for claim in claims}) == 1


@pytest.mark.parametrize("changed", ["runspec", "script"])
def test_same_slot_rejects_changed_governed_identity(tmp_path: Path, changed: str) -> None:
    coordinator = SubmissionCoordinator(tmp_path)
    coordinator.claim(_token())
    token = SubmissionToken.create(
        run_id="run-1",
        runspec_sha256=("c" if changed == "runspec" else "a") * 64,
        step_index=3,
        step_name="slurm",
        slice_id="0-9",
        attempt=1,
        script_sha256=("c" if changed == "script" else "b") * 64,
        bootstrap_sha256="c" * 64,
        control_state_sha256="d" * 64,
        runtime_qualification_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match="different token"):
        coordinator.claim(token)


def test_coordinator_rejects_replaced_token_bytes(tmp_path: Path) -> None:
    coordinator = SubmissionCoordinator(tmp_path)
    coordinator.claim(_token())
    record = next(coordinator.root.glob("*.json"))
    payload = json.loads(record.read_text())
    payload["token"]["run_id"] = "other-run"
    record.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"token does not match canonical identity|canonical coordinator record"):
        coordinator.claim(_token())


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_coordinator_rejects_unsafe_existing_record_destination(tmp_path: Path, kind: str) -> None:
    coordinator = SubmissionCoordinator(tmp_path)
    coordinator.root.mkdir()
    path = coordinator._path(_token())
    source = tmp_path / "source"
    source.write_text("{}")
    if kind == "symlink":
        path.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, path)
    else:
        os.mkfifo(path)
    with pytest.raises(ValueError, match=r"regular file|hard-linked"):
        coordinator.claim(_token())


def test_interrupted_bind_temp_publication_fails_closed(tmp_path: Path) -> None:
    coordinator = SubmissionCoordinator(tmp_path)
    coordinator.claim(_token())
    coordinator._path(_token()).with_suffix(".tmp").write_text("partial")
    with pytest.raises(ValueError, match="interrupted coordinator publication"):
        coordinator.bind_job(_token(), job_id="1", scheduler_status="PENDING")
    assert coordinator.claim(_token()).record.status == "prepared"


def test_coordinator_rejects_oversized_record_before_json_decode(tmp_path: Path) -> None:
    coordinator = SubmissionCoordinator(tmp_path)
    coordinator.root.mkdir()
    coordinator._path(_token()).write_bytes(b"{" + b" " * (64 * 1024) + b"}")
    with pytest.raises(ValueError, match="exceeds size bound"):
        coordinator.claim(_token())


def test_coordinator_rejects_symlinked_record_root(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    (tmp_path / "submission-coordinator").symlink_to(actual, target_is_directory=True)

    with pytest.raises(ValueError, match="coordinator root must be a real directory"):
        SubmissionCoordinator(tmp_path).claim(_token())


@pytest.mark.parametrize(
    ("status", "job_id", "scheduler_status"),
    [
        ("prepared", "123", "PENDING"),
        ("prepared", None, "PENDING"),
        ("submitted", None, None),
        ("submitted", "", "PENDING"),
        ("submitted", "123", ""),
    ],
)
def test_coordinator_rejects_inconsistent_lifecycle_record(
    tmp_path: Path,
    status: str,
    job_id: str | None,
    scheduler_status: str | None,
) -> None:
    coordinator = SubmissionCoordinator(tmp_path)
    coordinator.claim(_token())
    record = coordinator._path(_token())
    payload = json.loads(record.read_text())
    payload.update(status=status, job_id=job_id, scheduler_status=scheduler_status)
    record.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(ValueError, match="invalid submission coordinator record"):
        coordinator.claim(_token())
