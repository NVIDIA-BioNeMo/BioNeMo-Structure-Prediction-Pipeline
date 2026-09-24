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

"""Secret references and redacted resolution helpers."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable, Mapping
from configparser import ConfigParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from pydantic import ConfigDict, field_validator, model_validator

from bspp.orchestration.contract.config_models import FrozenConfigModel

SUPPORTED_SECRET_SCHEMES = frozenset({"env", "file", "aws"})

_DEFAULT_ENV_VARS: dict[str, tuple[str, ...]] = {
    "bspp/s3": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT_URL"),
    "s3": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT_URL"),
    "bspp/gcs": ("GOOGLE_APPLICATION_CREDENTIALS",),
    "gcs": ("GOOGLE_APPLICATION_CREDENTIALS",),
}
_AWS_ACCESS_KEY_KEYS = ("aws_access_key_id", "access_key_id")
_AWS_SECRET_KEY_KEYS = ("aws_secret_access_key", "secret_access_key")
_AWS_ENDPOINT_KEYS = ("endpoint_url", "s3_endpoint_url")
_AWS_PROFILE_ENV = "AWS_PROFILE"
_AWS_SHARED_CREDENTIALS_ENV = "AWS_SHARED_CREDENTIALS_FILE"
_AWS_CONFIG_ENV = "AWS_CONFIG_FILE"
_S3_ENDPOINT_ENV = "S3_ENDPOINT_URL"
_AWS_ENDPOINT_ENV = "AWS_ENDPOINT_URL"
_AWS_S3_ENDPOINT_ENV = "AWS_ENDPOINT_URL_S3"
_AWS_AUTO_PROFILE_TARGETS = frozenset({"auto", "profile:auto"})

_SENSITIVE_KEY_FRAGMENTS = (
    "access_key",
    "api_key",
    "client_secret",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)


class SecretRef(FrozenConfigModel):
    """A reference to secret material, never the secret value itself."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scheme: str
    target: str

    @model_validator(mode="before")
    @classmethod
    def _parse_string_ref(cls, value: Any) -> Any:
        if isinstance(value, str):
            scheme, separator, target = value.partition(":")
            if not separator or not scheme or not target:
                msg = f"Secret references must use '<scheme>:<target>', got {value!r}"
                raise ValueError(msg)
            return {"scheme": scheme, "target": target}
        return value

    @field_validator("scheme")
    @classmethod
    def _validate_scheme(cls, value: str) -> str:
        if not value:
            msg = "Secret reference scheme must be non-empty"
            raise ValueError(msg)
        if value not in SUPPORTED_SECRET_SCHEMES:
            msg = f"Unsupported secret reference scheme {value!r}; expected one of {sorted(SUPPORTED_SECRET_SCHEMES)}"
            raise ValueError(msg)
        return value

    @field_validator("target")
    @classmethod
    def _validate_target(cls, value: str) -> str:
        if not value:
            msg = "Secret reference target must be non-empty"
            raise ValueError(msg)
        return value

    @classmethod
    def parse(cls, raw: str) -> Self:
        """Parse ``scheme:target`` secret references."""
        return cls.model_validate(raw)

    def redacted(self) -> str:
        """Return a log-safe representation of the reference."""
        return f"{self.scheme}:<redacted>"


@dataclass(frozen=True)
class ResolvedSecret:
    """Resolution status with secret values intentionally omitted."""

    ref: SecretRef
    ok: bool
    message: str
    env_vars: tuple[str, ...] = ()
    file_path: Path | None = None

    def as_redacted_mapping(self) -> dict[str, object]:
        """Return a status mapping that is safe for dry runs and logs."""
        data: dict[str, object] = {
            "ref": self.ref.redacted(),
            "ok": self.ok,
            "message": self.message,
        }
        if self.env_vars:
            data["env_vars"] = list(self.env_vars)
        if self.file_path is not None:
            data["file"] = str(self.file_path)
        return data


class MissingSecretError(RuntimeError):
    """Raised when strict secret validation finds missing required secrets."""


@dataclass(frozen=True, repr=False)
class SecretExecutionEnv:
    """Subprocess environment overlay with values hidden from repr/logging."""

    _env: Mapping[str, str]

    def keys(self) -> tuple[str, ...]:
        """Return environment variable names supplied by this overlay."""
        return tuple(sorted(self._env))

    def as_mapping(self) -> dict[str, str]:
        """Return a copy suitable for subprocess ``env`` construction."""
        return dict(self._env)

    def apply_to(self, base_env: Mapping[str, str]) -> dict[str, str]:
        """Return ``base_env`` with the secret overlay applied."""
        env = dict(base_env)
        env.update(self._env)
        return env

    def as_redacted_mapping(self) -> dict[str, object]:
        """Return a log-safe summary of injected variable names."""
        return {"env_vars": list(self.keys()), "values": "<redacted>"}

    def __repr__(self) -> str:
        keys = ", ".join(self.keys())
        return f"SecretExecutionEnv(keys=({keys}), values=<redacted>)"


@dataclass(frozen=True)
class EnvSecretResolver:
    """Validate environment-backed secrets without exposing values."""

    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    required_vars_by_target: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: _DEFAULT_ENV_VARS)

    def resolve(self, ref: SecretRef) -> ResolvedSecret:
        """Resolve an ``env:`` reference to required environment variable names."""
        if ref.scheme != "env":
            msg = f"EnvSecretResolver cannot resolve {ref.scheme!r} references"
            raise ValueError(msg)
        required = self.required_vars_by_target.get(ref.target, (ref.target,))
        missing = tuple(name for name in required if not self.environ.get(name))
        if missing:
            return ResolvedSecret(
                ref=ref,
                ok=False,
                message=f"missing environment variables: {', '.join(missing)}",
                env_vars=required,
            )
        return ResolvedSecret(ref=ref, ok=True, message="resolved from environment", env_vars=required)


@dataclass(frozen=True)
class FileSecretResolver:
    """Validate file-backed secrets without reading their contents."""

    def resolve(self, ref: SecretRef) -> ResolvedSecret:
        """Resolve a ``file:`` reference to an existing, non-world-writable file."""
        if ref.scheme != "file":
            msg = f"FileSecretResolver cannot resolve {ref.scheme!r} references"
            raise ValueError(msg)
        path = Path(ref.target).expanduser().resolve()
        if not path.exists():
            return ResolvedSecret(ref=ref, ok=False, message="file does not exist", file_path=path)
        if not path.is_file():
            return ResolvedSecret(ref=ref, ok=False, message="path is not a file", file_path=path)
        mode = path.stat().st_mode
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            return ResolvedSecret(ref=ref, ok=False, message="file is group/world writable", file_path=path)
        return ResolvedSecret(ref=ref, ok=True, message="file exists with restricted write permissions", file_path=path)


@dataclass(frozen=True)
class AwsSharedCredentialsResolver:
    """Validate AWS shared credentials/config files without exposing values."""

    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)

    def resolve(self, ref: SecretRef) -> ResolvedSecret:
        """Resolve an ``aws:`` reference to shared AWS credential/config files."""
        if ref.scheme != "aws":
            msg = f"AwsSharedCredentialsResolver cannot resolve {ref.scheme!r} references"
            raise ValueError(msg)
        credentials_file = _aws_credentials_file(self.environ)
        config_file = _aws_config_file(self.environ)
        if not credentials_file.exists():
            return ResolvedSecret(ref=ref, ok=False, message="AWS shared credentials file does not exist")
        credentials = _read_ini(credentials_file)
        config = _read_ini(config_file) if config_file.exists() else ConfigParser(interpolation=None)
        profile_selection = _select_aws_profile(self.environ, ref.target, credentials=credentials, config=config)
        if profile_selection.profile is None:
            return ResolvedSecret(
                ref=ref,
                ok=False,
                message=profile_selection.message,
                file_path=credentials_file,
            )
        profile = profile_selection.profile
        credential_section = _aws_credentials_section(credentials, profile)
        if credential_section is None:
            return ResolvedSecret(
                ref=ref,
                ok=False,
                message=f"AWS profile {profile!r} is missing from shared credentials",
                file_path=credentials_file,
            )
        missing = []
        if not _first_present(credential_section, _AWS_ACCESS_KEY_KEYS):
            missing.append("aws_access_key_id")
        if not _first_present(credential_section, _AWS_SECRET_KEY_KEYS):
            missing.append("aws_secret_access_key")
        endpoint = _aws_endpoint(self.environ, profile=profile, credentials=credentials, config=config)
        if endpoint is None:
            missing.append("endpoint_url")
        if missing:
            return ResolvedSecret(
                ref=ref,
                ok=False,
                message=f"AWS profile {profile!r} is missing: {', '.join(missing)}",
                file_path=credentials_file,
            )
        return ResolvedSecret(
            ref=ref,
            ok=True,
            message=f"resolved from AWS shared credentials profile {profile!r}",
            file_path=credentials_file,
        )


def resolve_secret(ref: SecretRef, *, environ: Mapping[str, str] | None = None) -> ResolvedSecret:
    """Resolve a supported secret reference and return redacted status."""
    if ref.scheme == "env":
        return EnvSecretResolver(environ=os.environ if environ is None else environ).resolve(ref)
    if ref.scheme == "file":
        return FileSecretResolver().resolve(ref)
    if ref.scheme == "aws":
        return AwsSharedCredentialsResolver(environ=os.environ if environ is None else environ).resolve(ref)
    msg = f"Unsupported secret reference scheme {ref.scheme!r}"
    raise ValueError(msg)


def validate_required_secrets(
    refs: Iterable[SecretRef],
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[ResolvedSecret, ...]:
    """Resolve refs and raise if any required secret is unavailable."""
    statuses = tuple(resolve_secret(ref, environ=environ) for ref in refs)
    missing = tuple(status for status in statuses if not status.ok)
    if missing:
        details = "; ".join(f"{status.ref.redacted()}: {status.message}" for status in missing)
        msg = f"Missing required secrets: {details}"
        raise MissingSecretError(msg)
    return statuses


def build_secret_execution_env(
    refs: Mapping[str, SecretRef],
    *,
    environ: Mapping[str, str] | None = None,
) -> SecretExecutionEnv:
    """Build a subprocess env overlay for resolved secrets without logging values.

    ``env:`` refs copy only the configured required variables into the overlay.
    ``file:`` refs expose credential file paths for logical secrets that are
    file-backed by convention, currently GCS service-account credentials.
    ``aws:`` refs expose standard AWS shared-credential file paths, profile, and
    endpoint metadata without copying access-key values into the environment.
    """
    source_env = os.environ if environ is None else environ
    overlay: dict[str, str] = {}
    for logical_name, ref in refs.items():
        status = resolve_secret(ref, environ=source_env)
        if not status.ok:
            continue
        if ref.scheme == "env":
            for name in status.env_vars:
                value = source_env.get(name)
                if value:
                    overlay[name] = value
        elif ref.scheme == "file":
            file_env_var = _file_env_var_for(logical_name)
            if file_env_var and status.file_path is not None:
                overlay[file_env_var] = str(status.file_path)
        elif ref.scheme == "aws":
            credentials_file = _aws_credentials_file(source_env)
            config_file = _aws_config_file(source_env)
            credentials = _read_ini(credentials_file) if credentials_file.exists() else ConfigParser(interpolation=None)
            config = _read_ini(config_file) if config_file.exists() else ConfigParser(interpolation=None)
            profile_selection = _select_aws_profile(source_env, ref.target, credentials=credentials, config=config)
            if profile_selection.profile is None:
                continue
            profile = profile_selection.profile
            overlay[_AWS_PROFILE_ENV] = profile
            overlay[_AWS_SHARED_CREDENTIALS_ENV] = str(credentials_file)
            if config_file.exists():
                overlay[_AWS_CONFIG_ENV] = str(config_file)
            endpoint = _aws_endpoint(source_env, profile=profile, credentials=credentials, config=config)
            if endpoint:
                overlay[_S3_ENDPOINT_ENV] = endpoint
    return SecretExecutionEnv(overlay)


def _file_env_var_for(logical_name: str) -> str | None:
    normalized = logical_name.lower()
    if normalized in {"gcs", "gcs_credentials", "gcs_credentials_ref"}:
        return "GOOGLE_APPLICATION_CREDENTIALS"
    return None


@dataclass(frozen=True)
class _AwsProfileSelection:
    profile: str | None
    message: str


def _aws_profile_name(target: str) -> str:
    return target.removeprefix("profile:") or "default"


def _select_aws_profile(
    environ: Mapping[str, str],
    target: str,
    *,
    credentials: ConfigParser,
    config: ConfigParser,
) -> _AwsProfileSelection:
    if target not in _AWS_AUTO_PROFILE_TARGETS:
        profile = _aws_profile_name(target)
        return _AwsProfileSelection(profile=profile, message=f"using AWS profile {profile!r}")

    env_profile = environ.get(_AWS_PROFILE_ENV)
    if env_profile:
        return _AwsProfileSelection(profile=env_profile, message=f"using AWS_PROFILE {env_profile!r}")

    candidates = _complete_aws_profiles(environ, credentials=credentials, config=config)
    if "default" in candidates:
        return _AwsProfileSelection(profile="default", message="selected complete AWS profile 'default'")
    if len(candidates) == 1:
        profile = candidates[0]
        return _AwsProfileSelection(profile=profile, message=f"selected complete AWS profile {profile!r}")
    if not candidates:
        return _AwsProfileSelection(
            profile=None,
            message="no AWS shared credentials profile has access key, secret key, and endpoint_url",
        )
    return _AwsProfileSelection(
        profile=None,
        message=(
            "multiple AWS shared credentials profiles are complete; set AWS_PROFILE or use aws:<profile>: "
            + ", ".join(candidates)
        ),
    )


def _complete_aws_profiles(
    environ: Mapping[str, str],
    *,
    credentials: ConfigParser,
    config: ConfigParser,
) -> tuple[str, ...]:
    candidates: list[str] = []
    for profile in credentials.sections():
        credential_section = _aws_credentials_section(credentials, profile)
        if not _first_present(credential_section, _AWS_ACCESS_KEY_KEYS):
            continue
        if not _first_present(credential_section, _AWS_SECRET_KEY_KEYS):
            continue
        if _aws_endpoint(environ, profile=profile, credentials=credentials, config=config) is None:
            continue
        candidates.append(profile)
    return tuple(sorted(candidates))


def _aws_credentials_file(environ: Mapping[str, str]) -> Path:
    value = environ.get(_AWS_SHARED_CREDENTIALS_ENV)
    if value:
        return Path(value).expanduser()
    return Path.home() / ".aws" / "credentials"


def _aws_config_file(environ: Mapping[str, str]) -> Path:
    value = environ.get(_AWS_CONFIG_ENV)
    if value:
        return Path(value).expanduser()
    return Path.home() / ".aws" / "config"


def _read_ini(path: Path) -> ConfigParser:
    parser = ConfigParser(interpolation=None)
    parser.read(path)
    return parser


def _aws_credentials_section(parser: ConfigParser, profile: str) -> Mapping[str, str] | None:
    if parser.has_section(profile):
        return parser[profile]
    if profile == "default" and parser.has_section("default"):
        return parser["default"]
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


def _aws_endpoint(
    environ: Mapping[str, str],
    *,
    profile: str,
    credentials: ConfigParser,
    config: ConfigParser,
) -> str | None:
    if endpoint := environ.get(_S3_ENDPOINT_ENV):
        return endpoint
    if endpoint := environ.get(_AWS_S3_ENDPOINT_ENV):
        return endpoint
    if endpoint := environ.get(_AWS_ENDPOINT_ENV):
        return endpoint
    credential_section = _aws_credentials_section(credentials, profile)
    if endpoint := _first_present(credential_section, _AWS_ENDPOINT_KEYS):
        return endpoint
    if endpoint := _aws_service_endpoint(config, profile):
        return endpoint
    config_section = _aws_config_section(config, profile)
    return _first_present(config_section, _AWS_ENDPOINT_KEYS)


def reject_literal_secret_values(value: object, *, path: str = "runspec") -> None:
    """Reject likely literal secret values in user-authored mappings.

    Keys ending in ``_ref`` are allowed because they should hold ``SecretRef``
    strings such as ``env:bspp/s3`` or ``file:/secure/key.json``.
    """
    if isinstance(value, Mapping):
        for key_obj, child in value.items():
            key = str(key_obj)
            child_path = f"{path}.{key}"
            normalized = key.lower()
            child_is_container = isinstance(child, (Mapping, list, tuple))
            if (
                not normalized.endswith("_ref")
                and any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS)
                and not child_is_container
                and child not in (None, "")
            ):
                msg = f"{child_path} looks like literal secret material; store a *_ref value instead"
                raise ValueError(msg)
            reject_literal_secret_values(child, path=child_path)
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            reject_literal_secret_values(child, path=f"{path}[{index}]")


__all__ = [
    "EnvSecretResolver",
    "FileSecretResolver",
    "MissingSecretError",
    "ResolvedSecret",
    "SecretExecutionEnv",
    "SecretRef",
    "build_secret_execution_env",
    "reject_literal_secret_values",
    "resolve_secret",
    "validate_required_secrets",
]
