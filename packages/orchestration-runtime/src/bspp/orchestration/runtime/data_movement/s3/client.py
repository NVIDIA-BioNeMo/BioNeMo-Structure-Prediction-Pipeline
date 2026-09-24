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

"""S3 credential discovery for the S3 endpoint.

We do not talk to S3 from Python directly; ``s5cmd`` handles transfers.
This module owns the credential discovery contract for that subprocess:

- explicit environment variables, or
- AWS shared credentials/config files such as ``~/.aws/credentials``.

Credentials are never written by this package. The loader validates presence and
returns a typed record so callers can pass a minimal env overlay to ``s5cmd``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

S3_ENDPOINT_ENV = "S3_ENDPOINT_URL"
ACCESS_KEY_ENV = "AWS_ACCESS_KEY_ID"
SECRET_KEY_ENV = "AWS_SECRET_ACCESS_KEY"
AWS_PROFILE_ENV = "AWS_PROFILE"
AWS_SHARED_CREDENTIALS_ENV = "AWS_SHARED_CREDENTIALS_FILE"
AWS_CONFIG_ENV = "AWS_CONFIG_FILE"
AWS_ENDPOINT_ENV = "AWS_ENDPOINT_URL"
AWS_S3_ENDPOINT_ENV = "AWS_ENDPOINT_URL_S3"
_AWS_ACCESS_KEY_KEYS = ("aws_access_key_id", "access_key_id")
_AWS_SECRET_KEY_KEYS = ("aws_secret_access_key", "secret_access_key")
_AWS_ENDPOINT_KEYS = ("endpoint_url", "s3_endpoint_url")


class MissingS3CredentialsError(RuntimeError):
    """Raised when required S3 credentials cannot be discovered."""


@dataclass(frozen=True)
class S3Credentials:
    """Credentials + endpoint needed to drive s5cmd against S3."""

    access_key_id: str
    secret_access_key: str
    endpoint_url: str

    def as_env(self) -> dict[str, str]:
        """Return the minimal env-var dict to layer onto ``os.environ``."""
        return {
            ACCESS_KEY_ENV: self.access_key_id,
            SECRET_KEY_ENV: self.secret_access_key,
            S3_ENDPOINT_ENV: self.endpoint_url,
        }


def load_credentials_from_env() -> S3Credentials:
    """Read S3 credentials from env or AWS shared credentials/config files.

    The historical function name is kept for API compatibility. Discovery now
    prefers explicit env vars and then falls back to AWS shared files using
    ``AWS_PROFILE``/``AWS_SHARED_CREDENTIALS_FILE``/``AWS_CONFIG_FILE`` when
    provided, otherwise the standard ``~/.aws`` locations.
    """
    access = os.environ.get(ACCESS_KEY_ENV)
    secret = os.environ.get(SECRET_KEY_ENV)
    endpoint = os.environ.get(S3_ENDPOINT_ENV)
    if access and secret and endpoint:
        return S3Credentials(
            access_key_id=access,
            secret_access_key=secret,
            endpoint_url=endpoint,
        )

    shared = _load_credentials_from_shared_files()
    if shared is not None:
        return shared

    missing = _missing_env_names(access=access, secret=secret, endpoint=endpoint)
    profile = os.environ.get(AWS_PROFILE_ENV) or "<auto>"
    credentials_file = _aws_credentials_file()
    config_file = _aws_config_file()
    msg = (
        "Missing S3 credentials. Provide "
        + ", ".join(missing)
        + " in the environment or configure AWS shared credentials profile "
        f"{profile!r} in {credentials_file} with endpoint_url in {credentials_file} or {config_file}. "
        "When AWS_PROFILE is unset, discovery selects a complete default profile or the only complete profile."
    )
    raise MissingS3CredentialsError(msg)


def _load_credentials_from_shared_files() -> S3Credentials | None:
    credentials_file = _aws_credentials_file()
    if not credentials_file.exists():
        return None
    credentials = _read_ini(credentials_file)
    config_file = _aws_config_file()
    config = _read_ini(config_file) if config_file.exists() else ConfigParser(interpolation=None)
    profile = _select_aws_profile(credentials=credentials, config=config)
    if profile is None:
        return None
    credential_section = _aws_credentials_section(credentials, profile)
    if credential_section is None:
        return None
    access = _first_present(credential_section, _AWS_ACCESS_KEY_KEYS)
    secret = _first_present(credential_section, _AWS_SECRET_KEY_KEYS)
    endpoint = _aws_endpoint(profile=profile, credentials=credentials, config=config)
    if not access or not secret or not endpoint:
        return None
    return S3Credentials(access_key_id=access, secret_access_key=secret, endpoint_url=endpoint)


def _missing_env_names(*, access: str | None, secret: str | None, endpoint: str | None) -> list[str]:
    return [
        name
        for name, value in (
            (ACCESS_KEY_ENV, access),
            (SECRET_KEY_ENV, secret),
            (S3_ENDPOINT_ENV, endpoint),
        )
        if not value
    ]


def _aws_credentials_file() -> Path:
    value = os.environ.get(AWS_SHARED_CREDENTIALS_ENV)
    if value:
        return Path(value).expanduser()
    return Path.home() / ".aws" / "credentials"


def _aws_config_file() -> Path:
    value = os.environ.get(AWS_CONFIG_ENV)
    if value:
        return Path(value).expanduser()
    return Path.home() / ".aws" / "config"


def _read_ini(path: Path) -> ConfigParser:
    parser = ConfigParser(interpolation=None)
    parser.read(path)
    return parser


def _select_aws_profile(*, credentials: ConfigParser, config: ConfigParser) -> str | None:
    if profile := os.environ.get(AWS_PROFILE_ENV):
        return profile
    candidates = _complete_aws_profiles(credentials=credentials, config=config)
    if "default" in candidates:
        return "default"
    if len(candidates) == 1:
        return candidates[0]
    return None


def _complete_aws_profiles(*, credentials: ConfigParser, config: ConfigParser) -> tuple[str, ...]:
    candidates: list[str] = []
    for profile in credentials.sections():
        credential_section = _aws_credentials_section(credentials, profile)
        if not _first_present(credential_section, _AWS_ACCESS_KEY_KEYS):
            continue
        if not _first_present(credential_section, _AWS_SECRET_KEY_KEYS):
            continue
        if _aws_endpoint(profile=profile, credentials=credentials, config=config) is None:
            continue
        candidates.append(profile)
    return tuple(sorted(candidates))


def _aws_credentials_section(parser: ConfigParser, profile: str) -> Mapping[str, str] | None:
    if parser.has_section(profile):
        return parser[profile]
    return None


def _aws_config_section(parser: ConfigParser, profile: str) -> Mapping[str, str] | None:
    names = ("default",) if profile == "default" else (f"profile {profile}", profile)
    for name in names:
        if parser.has_section(name):
            return parser[name]
    return None


def _aws_services_section(parser: ConfigParser, profile: str) -> Mapping[str, str] | None:
    profile_section = _aws_config_section(parser, profile)
    if profile_section is None:
        return None
    services_name = profile_section.get("services")
    if not services_name:
        return None
    section_name = f"services {services_name}"
    if parser.has_section(section_name):
        return parser[section_name]
    return None


def _first_present(section: Mapping[str, str] | None, keys: tuple[str, ...]) -> str | None:
    if section is None:
        return None
    for key in keys:
        value = section.get(key)
        if value:
            return value
    return None


def _aws_service_endpoint(config: ConfigParser, profile: str) -> str | None:
    services_section = _aws_services_section(config, profile)
    if services_section is None:
        return None
    s3_config = services_section.get("s3")
    if not s3_config:
        return None
    return _endpoint_from_service_value(s3_config)


def _endpoint_from_service_value(value: str) -> str | None:
    for line in value.splitlines():
        key, separator, endpoint = line.partition("=")
        if separator and key.strip() == "endpoint_url" and endpoint.strip():
            return endpoint.strip()
    return None


def _aws_endpoint(*, profile: str, credentials: ConfigParser, config: ConfigParser) -> str | None:
    if endpoint := os.environ.get(S3_ENDPOINT_ENV):
        return endpoint
    if endpoint := os.environ.get(AWS_S3_ENDPOINT_ENV):
        return endpoint
    if endpoint := os.environ.get(AWS_ENDPOINT_ENV):
        return endpoint
    if endpoint := _first_present(_aws_credentials_section(credentials, profile), _AWS_ENDPOINT_KEYS):
        return endpoint
    if endpoint := _aws_service_endpoint(config, profile):
        return endpoint
    return _first_present(_aws_config_section(config, profile), _AWS_ENDPOINT_KEYS)


__all__ = [
    "ACCESS_KEY_ENV",
    "AWS_CONFIG_ENV",
    "AWS_ENDPOINT_ENV",
    "AWS_PROFILE_ENV",
    "AWS_S3_ENDPOINT_ENV",
    "AWS_SHARED_CREDENTIALS_ENV",
    "S3_ENDPOINT_ENV",
    "SECRET_KEY_ENV",
    "MissingS3CredentialsError",
    "S3Credentials",
    "load_credentials_from_env",
]
