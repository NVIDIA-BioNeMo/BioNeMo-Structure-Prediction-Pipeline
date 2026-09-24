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

"""Bounded no-follow evidence reads used by durable Phase attestations."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from bspp.orchestration.control.transport import CommandResult, RemoteSlurmTransport
from tests.support.transport_argv import maybe_unwrap_remote_command


def test_local_attestation_read_is_bounded_utf8_regular_and_no_follow(tmp_path: Path) -> None:
    transport = RemoteSlurmTransport(kind="local-slurm", ssh_target=None)
    evidence = tmp_path / "action-evidence.json"
    evidence.write_bytes(b'{"schema_version":1}\n')

    assert transport.read_immutable_text_artifact_no_follow(str(evidence)) == evidence.read_bytes()

    with pytest.raises(ValueError, match="bounded read"):
        transport.read_immutable_text_artifact_no_follow(str(evidence), max_bytes=3)
    symlink = tmp_path / "evidence-link.json"
    symlink.symlink_to(evidence)
    with pytest.raises(ValueError, match="symbolic links"):
        transport.read_immutable_text_artifact_no_follow(str(symlink))
    with pytest.raises(ValueError, match="not a regular file"):
        transport.read_immutable_text_artifact_no_follow(str(tmp_path))
    evidence.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="must be UTF-8"):
        transport.read_immutable_text_artifact_no_follow(str(evidence))


def test_ssh_attestation_read_uses_the_same_encoded_no_follow_program() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        return CommandResult(
            argv=argv,
            returncode=0,
            stdout=base64.b64encode(b"canonical evidence\n").decode(),
            stderr="",
        )

    transport = RemoteSlurmTransport(kind="ssh", ssh_target="example-cluster-login", runner=runner)

    assert transport.read_immutable_text_artifact_no_follow("/evidence/action.json") == b"canonical evidence\n"
    assert len(calls) == 1
    assert calls[0][:2] == ("ssh", "example-cluster-login")
    rendered = maybe_unwrap_remote_command(calls[0][-1])
    assert "O_NOFOLLOW" in rendered
    assert "/evidence/action.json" in rendered
    assert str(16 * 1024 * 1024) in rendered
