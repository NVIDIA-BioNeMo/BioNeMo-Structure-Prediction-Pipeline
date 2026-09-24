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

"""Shared Runtime Qualification contract records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from bspp.orchestration.contract.versioning import validate_schema_version


@dataclass(frozen=True)
class SelectedSourceIdentity:
    """Attested identity of the toolkit source selected for qualification."""

    source_kind: str
    root: str
    revision: str
    image_identity: Mapping[str, object] | None = None
    toolkit_package_identity: Mapping[str, object] | None = None

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "source_kind": self.source_kind,
            "root": self.root,
            "revision": self.revision,
        }
        if self.image_identity is not None:
            result["image_identity"] = dict(self.image_identity)
        if self.toolkit_package_identity is not None:
            result["toolkit_package_identity"] = dict(self.toolkit_package_identity)
        return result


def selected_source_identity_from_mapping(value: Mapping[str, object]) -> SelectedSourceIdentity:
    """Strictly parse a selected-source identity mapping."""
    if not isinstance(value, Mapping):
        raise ValueError("SelectedSourceIdentity must be a mapping")
    kind = value.get("source_kind")
    root = value.get("root")
    revision = value.get("revision")
    if kind not in ("baked", "override"):
        raise ValueError("SelectedSourceIdentity source_kind must be 'baked' or 'override'")
    if not isinstance(root, str) or not root or not root.startswith("/"):
        raise ValueError("SelectedSourceIdentity root must be a non-empty absolute path")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("SelectedSourceIdentity revision must be a 40-char hex commit")
    image_identity = value.get("image_identity")
    toolkit_package_identity = value.get("toolkit_package_identity")
    if kind == "baked":
        if not isinstance(image_identity, Mapping):
            raise ValueError("SelectedSourceIdentity baked mode requires image_identity")
        if toolkit_package_identity is not None:
            raise ValueError("SelectedSourceIdentity baked mode must not set toolkit_package_identity")
    else:
        if not isinstance(toolkit_package_identity, Mapping):
            raise ValueError("SelectedSourceIdentity override mode requires toolkit_package_identity")
        if image_identity is not None:
            raise ValueError("SelectedSourceIdentity override mode must not set image_identity")
    allowed = {"source_kind", "root", "revision"}
    if kind == "baked":
        allowed.add("image_identity")
    else:
        allowed.add("toolkit_package_identity")
    if set(value) != allowed:
        raise ValueError("SelectedSourceIdentity has extra fields")
    return SelectedSourceIdentity(
        source_kind=str(kind),
        root=str(root),
        revision=str(revision),
        image_identity=dict(image_identity) if isinstance(image_identity, Mapping) else None,
        toolkit_package_identity=dict(toolkit_package_identity)
        if isinstance(toolkit_package_identity, Mapping)
        else None,
    )


@dataclass(frozen=True)
class RuntimeQualificationRecord:
    """Persisted qualification evidence for one runtime tuple."""

    tuple_id: str
    path: Path
    status: str | None = None
    job_id: str | None = None
    script_path: Path | None = None


@dataclass(frozen=True)
class RuntimeQualificationSnapshot:
    """One immutable, self-identifying Runtime Qualification document."""

    document: bytes
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.size_bytes != len(self.document) or self.sha256 != sha256(self.document).hexdigest():
            raise ValueError("Runtime Qualification snapshot identity differs from its document")


@dataclass(frozen=True)
class RuntimeQualificationCheck:
    """Current qualification status for the runtime tuple expected now."""

    current: bool
    reason: str
    tuple_id: str | None
    record_path: Path
    snapshot: RuntimeQualificationSnapshot | None = None

    def __post_init__(self) -> None:
        if self.current != (self.snapshot is not None):
            raise ValueError("Runtime Qualification current status and snapshot presence differ")


_MAX_COMMAND_OUTPUT_BYTES = 64 * 1024
_MAX_BUILD_LOG_BYTES = 1024 * 1024
_MAX_IPSAE_BINARY_BYTES = 1024 * 1024 * 1024
RUNTIME_IPSAE_FIXTURE_MODEL_ID = "BSPP-RQ-PAIR"
LEGACY_RUNTIME_IPSAE_FIXTURE_MODEL_ID = "AFCDB-RQ-PAIR"
RUNTIME_IPSAE_FIXTURE_PDB = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C
ATOM      2  CA  ALA A   2       0.000   2.000   0.000  1.00 90.00           C
ATOM      3  CA  ALA B   1       0.000   0.000   4.000  1.00 90.00           C
ATOM      4  CA  ALA B   2       0.000   2.000   4.000  1.00 90.00           C
END
"""
RUNTIME_IPSAE_FIXTURE_PAE = (
    '{"max_pae":10.0,"pae":[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]}\n'
)
RUNTIME_IPSAE_FIXTURE_SHA256 = sha256(
    RUNTIME_IPSAE_FIXTURE_PDB.encode() + b"\0" + RUNTIME_IPSAE_FIXTURE_PAE.encode()
).hexdigest()
RUNTIME_IPSAE_EXPECTED_SCORE_AB = "1.000000"
RUNTIME_IPSAE_EXPECTED_SCORE_BA = "1.000000"
RUNTIME_IPSAE_EXPECTED_RESULT_SHA256 = sha256(
    b'{"ipsae_AB":"1.000000","ipsae_BA":"1.000000","model_id":"BSPP-RQ-PAIR"}\n'
).hexdigest()
LEGACY_RUNTIME_IPSAE_EXPECTED_RESULT_SHA256 = sha256(
    b'{"ipsae_AB":"1.000000","ipsae_BA":"1.000000","model_id":"AFCDB-RQ-PAIR"}\n'
).hexdigest()


@dataclass(frozen=True)
class RuntimeCommandEvidence:
    """Bounded result from one command executed during runtime qualification."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool

    def to_mapping(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
        }


@dataclass(frozen=True)
class RuntimeArtifactEvidence:
    """Identity of one retained runtime-built artifact."""

    path: str
    sha256: str
    size_bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class RuntimeFunctionalTestEvidence:
    """Successful deterministic functional-test evidence."""

    command: RuntimeCommandEvidence
    fixture_sha256: str
    result_sha256: str
    check: str
    expected_output: str
    status: str
    model_id: str
    ipsae_ab: str
    ipsae_ba: str

    def to_mapping(self) -> dict[str, object]:
        return {
            **self.command.to_mapping(),
            "fixture_sha256": self.fixture_sha256,
            "result_sha256": self.result_sha256,
            "check": self.check,
            "expected_output": self.expected_output,
            "status": self.status,
            "model_id": self.model_id,
            "ipsae_ab": self.ipsae_ab,
            "ipsae_ba": self.ipsae_ba,
        }


@dataclass(frozen=True)
class RuntimeVersionIdentity:
    """Human-readable version identity for a binary without a version CLI."""

    scheme: str
    source_revision: str
    binary_sha256: str
    output: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "source_revision": self.source_revision,
            "binary_sha256": self.binary_sha256,
            "output": self.output,
        }


@dataclass(frozen=True)
class RuntimeIpsaeEvidence:
    """Evidence for iPSAE built and tested inside the selected runtime image."""

    format_version: int
    source_revision: str
    source_path: str
    build_command: tuple[str, ...]
    toolchain: Mapping[str, RuntimeCommandEvidence]
    build_result: RuntimeCommandEvidence
    build_log: RuntimeArtifactEvidence
    binary: RuntimeArtifactEvidence
    version: RuntimeVersionIdentity
    functional_test: RuntimeFunctionalTestEvidence

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "source_revision": self.source_revision,
            "source_path": self.source_path,
            "build_command": list(self.build_command),
            "toolchain": {key: value.to_mapping() for key, value in self.toolchain.items()},
            "build_result": self.build_result.to_mapping(),
            "build_log": self.build_log.to_mapping(),
            "binary": self.binary.to_mapping(),
            "version": self.version.to_mapping(),
            "functional_test": self.functional_test.to_mapping(),
        }


def _command_evidence_from_mapping(value: object, *, require_success: bool) -> RuntimeCommandEvidence:
    if not isinstance(value, Mapping) or set(value) != {
        "argv",
        "returncode",
        "stdout",
        "stderr",
        "stdout_truncated",
        "stderr_truncated",
    }:
        raise ValueError("Runtime command evidence has missing or extra fields")
    argv = value.get("argv")
    returncode = value.get("returncode")
    stdout = value.get("stdout")
    stderr = value.get("stderr")
    stdout_truncated = value.get("stdout_truncated")
    stderr_truncated = value.get("stderr_truncated")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
        or type(returncode) is not int
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or type(stdout_truncated) is not bool
        or type(stderr_truncated) is not bool
    ):
        raise ValueError("Runtime command evidence has invalid field types")
    if len(stdout.encode()) > _MAX_COMMAND_OUTPUT_BYTES or len(stderr.encode()) > _MAX_COMMAND_OUTPUT_BYTES:
        raise ValueError("Runtime command evidence output is not bounded")
    if require_success and returncode != 0:
        raise ValueError("Runtime command evidence did not succeed")
    return RuntimeCommandEvidence(tuple(argv), returncode, stdout, stderr, stdout_truncated, stderr_truncated)


def _artifact_evidence_from_mapping(
    value: object, *, label: str, max_size: int, require_nonempty: bool
) -> RuntimeArtifactEvidence:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{label} evidence has missing or extra fields")
    path_value = value.get("path")
    digest = value.get("sha256")
    size = value.get("size_bytes")
    if not isinstance(path_value, str) or not path_value or Path(path_value).is_absolute():
        raise ValueError(f"{label} evidence path must be relative")
    normalized = Path(path_value)
    if normalized != Path(*normalized.parts) or ".." in normalized.parts:
        raise ValueError(f"{label} evidence path is unsafe")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{label} evidence SHA-256 is invalid")
    if type(size) is not int or size < int(require_nonempty) or size > max_size:
        raise ValueError(f"{label} evidence size is invalid")
    return RuntimeArtifactEvidence(path_value, digest, size)


def runtime_ipsae_evidence_from_mapping(payload: Mapping[str, object]) -> RuntimeIpsaeEvidence:
    """Strictly validate independently versioned runtime-built iPSAE evidence."""
    required = {
        "format_version",
        "source_revision",
        "source_path",
        "build_command",
        "toolchain",
        "build_result",
        "build_log",
        "binary",
        "version",
        "functional_test",
    }
    if set(payload) != required or payload.get("format_version") != 1:
        raise ValueError("RuntimeIpsaeEvidence has missing, extra, or unsupported fields")
    revision = payload.get("source_revision")
    source_path = payload.get("source_path")
    build_command = payload.get("build_command")
    toolchain_value = payload.get("toolchain")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None:
        raise ValueError("RuntimeIpsaeEvidence source revision is invalid")
    if (
        not isinstance(source_path, str)
        or not source_path
        or Path(source_path).is_absolute()
        or ".." in Path(source_path).parts
    ):
        raise ValueError("RuntimeIpsaeEvidence source path is unsafe")
    if source_path != "afdb_integration_kit/ipsae":
        raise ValueError("RuntimeIpsaeEvidence source path is unsupported")
    if (
        not isinstance(build_command, list)
        or not build_command
        or any(not isinstance(item, str) or not item for item in build_command)
    ):
        raise ValueError("RuntimeIpsaeEvidence build command is invalid")
    assert isinstance(build_command, list)
    if (
        len(build_command) != 5
        or build_command[:3] != ["make", "-B", "-C"]
        or not build_command[3].endswith("/" + source_path)
        or build_command[4] != "CXX=g++"
    ):
        raise ValueError("RuntimeIpsaeEvidence build command does not force the supported fresh build")
    if not isinstance(toolchain_value, Mapping) or set(toolchain_value) != {"make", "cxx"}:
        raise ValueError("RuntimeIpsaeEvidence toolchain mapping is invalid")
    toolchain: dict[str, RuntimeCommandEvidence] = {}
    for name, value in toolchain_value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("RuntimeIpsaeEvidence toolchain name is invalid")
        toolchain[name] = _command_evidence_from_mapping(value, require_success=True)
    if toolchain["make"].argv != ("make", "--version") or toolchain["cxx"].argv != ("g++", "--version"):
        raise ValueError("RuntimeIpsaeEvidence toolchain command is unsupported")
    if any(
        not command.stdout.strip() or command.stdout_truncated or command.stderr_truncated
        for command in toolchain.values()
    ):
        raise ValueError("RuntimeIpsaeEvidence toolchain version output is incomplete")
    build_result = _command_evidence_from_mapping(payload.get("build_result"), require_success=True)
    if build_result.argv != tuple(build_command):
        raise ValueError("RuntimeIpsaeEvidence build result does not bind the exact build command")
    binary = _artifact_evidence_from_mapping(
        payload.get("binary"), label="iPSAE binary", max_size=_MAX_IPSAE_BINARY_BYTES, require_nonempty=True
    )
    if binary.path != "runtime-ipsae/ipsae_cpp":
        raise ValueError("RuntimeIpsaeEvidence binary path is unsupported")
    version_value = payload.get("version")
    if not isinstance(version_value, Mapping) or set(version_value) != {
        "scheme",
        "source_revision",
        "binary_sha256",
        "output",
    }:
        raise ValueError("RuntimeIpsaeEvidence version identity is invalid")
    version_output = f"ipsae_cpp source={revision} sha256={binary.sha256}"
    if (
        version_value.get("scheme") != "source-revision+binary-sha256-v1"
        or version_value.get("source_revision") != revision
        or version_value.get("binary_sha256") != binary.sha256
        or version_value.get("output") != version_output
    ):
        raise ValueError("RuntimeIpsaeEvidence version identity contradicts source or binary identity")
    version = RuntimeVersionIdentity(
        scheme="source-revision+binary-sha256-v1",
        source_revision=revision,
        binary_sha256=binary.sha256,
        output=version_output,
    )
    functional_value = payload.get("functional_test")
    if not isinstance(functional_value, Mapping):
        raise ValueError("RuntimeIpsaeEvidence functional test is invalid")
    functional_command = _command_evidence_from_mapping(
        {
            key: value
            for key, value in functional_value.items()
            if key
            not in {
                "fixture_sha256",
                "result_sha256",
                "check",
                "expected_output",
                "status",
                "model_id",
                "ipsae_ab",
                "ipsae_ba",
            }
        },
        require_success=True,
    )
    fixture_sha = functional_value.get("fixture_sha256")
    result_sha = functional_value.get("result_sha256")
    if fixture_sha != RUNTIME_IPSAE_FIXTURE_SHA256:
        raise ValueError("RuntimeIpsaeEvidence fixture SHA-256 is invalid")
    if result_sha not in (RUNTIME_IPSAE_EXPECTED_RESULT_SHA256, LEGACY_RUNTIME_IPSAE_EXPECTED_RESULT_SHA256):
        raise ValueError("RuntimeIpsaeEvidence result SHA-256 is invalid")
    if (
        functional_value.get("check") != "paired-pdb-pae-semantic-v1"
        or functional_value.get("expected_output")
        not in (
            "one-row:BSPP-RQ-PAIR:ipsae_AB=1.000000:ipsae_BA=1.000000",
            "one-row:AFCDB-RQ-PAIR:ipsae_AB=1.000000:ipsae_BA=1.000000",
        )
        or functional_value.get("status") != "passed"
        or functional_value.get("model_id")
        not in (RUNTIME_IPSAE_FIXTURE_MODEL_ID, LEGACY_RUNTIME_IPSAE_FIXTURE_MODEL_ID)
        or functional_value.get("ipsae_ab") != RUNTIME_IPSAE_EXPECTED_SCORE_AB
        or functional_value.get("ipsae_ba") != RUNTIME_IPSAE_EXPECTED_SCORE_BA
    ):
        raise ValueError("RuntimeIpsaeEvidence functional test semantics or status is unsupported")
    functional_argv = functional_command.argv
    if (
        len(functional_argv) != 10
        or not functional_argv[0].endswith("/" + binary.path)
        or functional_argv[1] != "--batch"
        or not functional_argv[2].endswith("/runtime-ipsae/functional-fixture")
        or functional_argv[3:6] != ("10.0", "8.0", "--summary")
        or not functional_argv[6].endswith("/runtime-ipsae/functional-summary.csv")
        or functional_argv[7:] != ("--workers", "1", "--quiet")
    ):
        raise ValueError("RuntimeIpsaeEvidence functional test command is unsupported")
    return RuntimeIpsaeEvidence(
        format_version=1,
        source_revision=revision,
        source_path=source_path,
        build_command=tuple(build_command),
        toolchain=toolchain,
        build_result=build_result,
        build_log=_artifact_evidence_from_mapping(
            payload.get("build_log"), label="iPSAE build log", max_size=_MAX_BUILD_LOG_BYTES, require_nonempty=False
        ),
        binary=binary,
        version=version,
        functional_test=RuntimeFunctionalTestEvidence(
            functional_command,
            fixture_sha,
            result_sha,
            "paired-pdb-pae-semantic-v1",
            cast(str, functional_value["expected_output"]),
            "passed",
            cast(str, functional_value["model_id"]),
            RUNTIME_IPSAE_EXPECTED_SCORE_AB,
            RUNTIME_IPSAE_EXPECTED_SCORE_BA,
        ),
    )


def runtime_qualification_payload_from_mapping(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate a Runtime Qualification evidence mapping."""
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="RuntimeQualification")
    data = dict(payload)
    data["schema_version"] = schema_version
    return data


__all__ = [
    "RuntimeArtifactEvidence",
    "RuntimeCommandEvidence",
    "RuntimeFunctionalTestEvidence",
    "RuntimeIpsaeEvidence",
    "RuntimeQualificationCheck",
    "RuntimeQualificationRecord",
    "RuntimeQualificationSnapshot",
    "RuntimeVersionIdentity",
    "SelectedSourceIdentity",
    "runtime_ipsae_evidence_from_mapping",
    "runtime_qualification_payload_from_mapping",
    "selected_source_identity_from_mapping",
]
