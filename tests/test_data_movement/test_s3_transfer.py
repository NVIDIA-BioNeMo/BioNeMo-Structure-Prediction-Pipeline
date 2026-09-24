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

"""Unit tests for the s5cmd subprocess wrapper."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from bspp.orchestration.runtime.data_movement.common import PlannedTransfer
from bspp.orchestration.runtime.data_movement.s3 import client as s3_client
from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer
from bspp.orchestration.runtime.data_movement.s3.client import (
    MissingS3CredentialsError,
    S3Credentials,
)

_CREDS = S3Credentials(
    access_key_id="AKIA-test",
    secret_access_key="secret",
    endpoint_url="https://swiftstack.test",
)


@pytest.fixture
def fake_s5cmd(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/s5cmd" if name == "s5cmd" else None)
    return "/usr/local/bin/s5cmd"


def test_load_credentials_from_env_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://e")

    creds = s3_client.load_credentials_from_env()

    assert creds == S3Credentials(access_key_id="k", secret_access_key="s", endpoint_url="https://e")
    assert creds.as_env() == {
        "AWS_ACCESS_KEY_ID": "k",
        "AWS_SECRET_ACCESS_KEY": "s",
        "S3_ENDPOINT_URL": "https://e",
    }


def test_load_credentials_missing_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    # Isolate the AWS shared credentials/config file discovery: hosts with a
    # real ~/.aws/credentials would otherwise supply a fallback and the
    # missing-credentials error would never be raised.
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-config"))

    with pytest.raises(MissingS3CredentialsError) as excinfo:
        s3_client.load_credentials_from_env()

    msg = str(excinfo.value)
    assert "AWS_ACCESS_KEY_ID" in msg
    assert "S3_ENDPOINT_URL" in msg


def test_load_credentials_from_shared_aws_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    credentials_file = tmp_path / "credentials"
    config_file = tmp_path / "config"
    credentials_file.write_text(
        "\n".join(
            [
                "[example-account]",
                "aws_access_key_id = k",
                "aws_secret_access_key = s-%",
                "",
            ]
        )
    )
    config_file.write_text(
        "\n".join(
            [
                "[profile example-account]",
                "services = swiftstack-s3",
                "",
                "[services swiftstack-s3]",
                "s3 =",
                "  endpoint_url = https://swiftstack.test",
                "",
            ]
        )
    )
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "example-account")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_file))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))

    creds = s3_client.load_credentials_from_env()

    assert creds == S3Credentials(access_key_id="k", secret_access_key="s-%", endpoint_url="https://swiftstack.test")


def test_load_credentials_auto_selects_single_non_default_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_file = tmp_path / "credentials"
    config_file = tmp_path / "config"
    credentials_file.write_text(
        "\n".join(
            [
                "[swiftstack]",
                "aws_access_key_id = k",
                "aws_secret_access_key = s",
                "",
            ]
        )
    )
    config_file.write_text(
        "\n".join(
            [
                "[profile swiftstack]",
                "endpoint_url = https://swiftstack.test",
                "",
            ]
        )
    )
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_file))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))

    creds = s3_client.load_credentials_from_env()

    assert creds == S3Credentials(access_key_id="k", secret_access_key="s", endpoint_url="https://swiftstack.test")


def test_load_credentials_uses_standard_s3_endpoint_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_file = tmp_path / "credentials"
    credentials_file.write_text(
        "\n".join(
            [
                "[example-account]",
                "aws_access_key_id = k",
                "aws_secret_access_key = s",
                "",
            ]
        )
    )
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "example-account")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://swiftstack.test")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_file))

    creds = s3_client.load_credentials_from_env()

    assert creds == S3Credentials(access_key_id="k", secret_access_key="s", endpoint_url="https://swiftstack.test")


def test_cp_dry_run_includes_endpoint_and_workers(fake_s5cmd: str) -> None:
    planned = s3_transfer.cp(
        "src/file",
        "s3://bucket/key",
        credentials=_CREDS,
        numworkers=32,
        dry_run=True,
    )

    assert isinstance(planned, PlannedTransfer)
    assert planned.argv == (
        fake_s5cmd,
        "--endpoint-url",
        _CREDS.endpoint_url,
        "--numworkers",
        "32",
        "cp",
        "src/file",
        "s3://bucket/key",
    )


def test_cp_merges_credentials_into_env(monkeypatch: pytest.MonkeyPatch, fake_s5cmd: str) -> None:
    captured_env: dict[str, str] = {}

    def fake_run(argv, capture_output, text, env, check):  # type: ignore[no-untyped-def]
        captured_env.update(env)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    s3_transfer.cp("s3://a", "s3://b", credentials=_CREDS)

    assert captured_env["AWS_ACCESS_KEY_ID"] == _CREDS.access_key_id
    assert captured_env["AWS_SECRET_ACCESS_KEY"] == _CREDS.secret_access_key
    assert captured_env["S3_ENDPOINT_URL"] == _CREDS.endpoint_url


def test_write_cp_command_file_formats_lines(tmp_path: Path) -> None:
    out = tmp_path / "cmds.txt"
    pairs = [("/a/b", "s3://bucket/prefix/"), ("/c/d", "s3://bucket/prefix/")]

    written = s3_transfer.write_cp_command_file(pairs, out)

    assert written == out
    contents = out.read_text().splitlines()
    assert contents == ["cp /a/b s3://bucket/prefix/", "cp /c/d s3://bucket/prefix/"]


# ---------------------------------------------------------------------------
# #102 regression: dry-run renders without the tool binary or credentials
# ---------------------------------------------------------------------------


def test_cp_dry_run_without_s5cmd_or_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Dry-run must render the plan even when s5cmd is absent and creds are missing."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-config"))

    planned = s3_transfer.cp("src/file", "s3://bucket/key", dry_run=True)

    assert isinstance(planned, PlannedTransfer)
    assert planned.tool == "s5cmd"
    assert planned.argv[0] == "s5cmd"  # bare tool name, not a resolved path
    assert "--endpoint-url" in planned.argv
    assert "<credentials not available>" in planned.argv
    assert "cp" in planned.argv
    assert "src/file" in planned.argv
    assert "s3://bucket/key" in planned.argv


def test_sync_dry_run_without_s5cmd_or_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Dry-run sync must also render without s5cmd or credentials."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-config"))

    planned = s3_transfer.sync("src/", "s3://bucket/prefix/", dry_run=True)

    assert isinstance(planned, PlannedTransfer)
    assert planned.argv[0] == "s5cmd"
    assert "sync" in planned.argv


def test_run_command_file_dry_run_without_s5cmd_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dry-run run-command-file must also render without s5cmd or credentials."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-config"))

    cmd_file = tmp_path / "cmds.txt"
    cmd_file.write_text("cp /a/b s3://bucket/prefix/\n")
    planned = s3_transfer.run_command_file(cmd_file, dry_run=True)

    assert isinstance(planned, PlannedTransfer)
    assert planned.argv[0] == "s5cmd"
    assert "run" in planned.argv


def test_cp_non_dry_run_raises_without_s5cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-dry-run must still raise ToolMissingError when s5cmd is absent."""
    from bspp.orchestration.runtime.data_movement.common import ToolMissingError

    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(ToolMissingError):
        s3_transfer.cp("s3://a", "s3://b", credentials=_CREDS)


def test_cp_non_dry_run_raises_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
    fake_s5cmd: str,
    tmp_path: Path,
) -> None:
    """Non-dry-run must still raise MissingS3CredentialsError when creds are missing."""
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-config"))

    with pytest.raises(MissingS3CredentialsError):
        s3_transfer.cp("s3://a", "s3://b")
