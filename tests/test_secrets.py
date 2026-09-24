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

"""Tests for secret references and redacted resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from bspp.orchestration.contract.secrets import (
    MissingSecretError,
    SecretRef,
    build_secret_execution_env,
    reject_literal_secret_values,
    resolve_secret,
    validate_required_secrets,
)


def test_secret_ref_requires_supported_scheme() -> None:
    with pytest.raises(ValueError, match="Unsupported secret reference scheme"):
        SecretRef.parse("literal:not-supported")


def test_secret_ref_validates_string_and_mapping_forms() -> None:
    env_ref = SecretRef.model_validate("env:bspp/s3")
    file_ref = SecretRef.model_validate({"scheme": "file", "target": "/secure/key.json"})
    aws_ref = SecretRef.model_validate("aws:example-account")

    assert env_ref.scheme == "env"
    assert env_ref.target == "bspp/s3"
    assert file_ref.scheme == "file"
    assert file_ref.target == "/secure/key.json"
    assert aws_ref.scheme == "aws"
    assert aws_ref.target == "example-account"


def test_secret_ref_rejects_unknown_mapping_fields() -> None:
    with pytest.raises(ValidationError, match="literal_value"):
        SecretRef.model_validate({"scheme": "env", "target": "TOKEN", "literal_value": "secret"})


def test_env_secret_resolution_reports_names_without_values() -> None:
    ref = SecretRef.parse("env:bspp/s3")
    status = resolve_secret(
        ref,
        environ={
            "AWS_ACCESS_KEY_ID": "id-value",
            "AWS_SECRET_ACCESS_KEY": "secret-value",
            "S3_ENDPOINT_URL": "https://s3.example.test",
        },
    )

    rendered = status.as_redacted_mapping()
    assert status.ok is True
    assert rendered["ref"] == "env:<redacted>"
    assert "secret-value" not in str(rendered)
    assert rendered["env_vars"] == ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT_URL"]


def test_file_secret_resolution_checks_file_without_reading_contents(tmp_path: Path) -> None:
    secret_file = tmp_path / "service-account.json"
    secret_file.write_text("private contents")
    secret_file.chmod(0o600)

    status = resolve_secret(SecretRef.parse(f"file:{secret_file}"))

    rendered = status.as_redacted_mapping()
    assert status.ok is True
    assert rendered["ref"] == "file:<redacted>"
    assert "private contents" not in str(rendered)


def test_aws_secret_resolution_uses_shared_credentials_without_exposing_values(tmp_path: Path) -> None:
    credentials_file = tmp_path / "credentials"
    config_file = tmp_path / "config"
    credentials_file.write_text(
        "\n".join(
            [
                "[example-account]",
                "aws_access_key_id = id-value",
                "aws_secret_access_key = secret-%-value",
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

    status = resolve_secret(
        SecretRef.parse("aws:example-account"),
        environ={
            "AWS_SHARED_CREDENTIALS_FILE": str(credentials_file),
            "AWS_CONFIG_FILE": str(config_file),
        },
    )

    rendered = status.as_redacted_mapping()
    assert status.ok is True
    assert rendered["ref"] == "aws:<redacted>"
    assert rendered["file"] == str(credentials_file)
    assert "id-value" not in str(rendered)
    assert "secret-%-value" not in str(rendered)


def test_validate_required_secrets_raises_redacted_error() -> None:
    ref = SecretRef.parse("env:bspp/s3")

    with pytest.raises(MissingSecretError) as error:
        validate_required_secrets([ref], environ={})

    assert "env:<redacted>" in str(error.value)
    assert "bspp/s3" not in str(error.value)


def test_secret_execution_env_hides_values_from_reporting(tmp_path: Path) -> None:
    gcs_file = tmp_path / "gcs.json"
    gcs_file.write_text("private contents")
    gcs_file.chmod(0o600)

    overlay = build_secret_execution_env(
        {
            "s3_credentials": SecretRef.parse("env:bspp/s3"),
            "gcs_credentials": SecretRef.parse(f"file:{gcs_file}"),
        },
        environ={
            "AWS_ACCESS_KEY_ID": "id-value",
            "AWS_SECRET_ACCESS_KEY": "secret-value",
            "S3_ENDPOINT_URL": "https://s3.example.test",
        },
    )

    env = overlay.as_mapping()
    assert env["AWS_ACCESS_KEY_ID"] == "id-value"
    assert env["GOOGLE_APPLICATION_CREDENTIALS"] == str(gcs_file.resolve())
    assert "secret-value" not in repr(overlay)
    assert "private contents" not in str(overlay.as_redacted_mapping())


def test_aws_secret_execution_env_points_to_shared_files_without_key_values(tmp_path: Path) -> None:
    credentials_file = tmp_path / "credentials"
    config_file = tmp_path / "config"
    credentials_file.write_text(
        "\n".join(
            [
                "[example-account]",
                "aws_access_key_id = id-value",
                "aws_secret_access_key = secret-%-value",
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

    overlay = build_secret_execution_env(
        {"s3_credentials": SecretRef.parse("aws:example-account")},
        environ={
            "AWS_SHARED_CREDENTIALS_FILE": str(credentials_file),
            "AWS_CONFIG_FILE": str(config_file),
        },
    )

    env = overlay.as_mapping()
    assert env == {
        "AWS_CONFIG_FILE": str(config_file),
        "AWS_PROFILE": "example-account",
        "AWS_SHARED_CREDENTIALS_FILE": str(credentials_file),
        "S3_ENDPOINT_URL": "https://swiftstack.test",
    }
    assert "AWS_ACCESS_KEY_ID" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "id-value" not in repr(overlay)
    assert "secret-%-value" not in str(overlay.as_redacted_mapping())


def test_literal_secret_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="literal secret material"):
        reject_literal_secret_values({"storage": {"aws_secret_access_key": "abc123"}})


def test_secret_reference_keys_are_allowed() -> None:
    reject_literal_secret_values({"secrets": {"s3_credentials_ref": "env:bspp/s3"}})
