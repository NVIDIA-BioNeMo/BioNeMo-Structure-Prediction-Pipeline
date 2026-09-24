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

"""Small generic runtime-iPSAE evidence fixtures for contract/control tests."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from bspp.orchestration.contract.runtime_qualification import (
    RUNTIME_IPSAE_EXPECTED_RESULT_SHA256,
    RUNTIME_IPSAE_EXPECTED_SCORE_AB,
    RUNTIME_IPSAE_EXPECTED_SCORE_BA,
    RUNTIME_IPSAE_FIXTURE_MODEL_ID,
    RUNTIME_IPSAE_FIXTURE_SHA256,
)
from bspp.orchestration.control.runtime_qualification import BAKED_TOOLKIT_COMMIT


def runtime_ipsae_evidence(*, source_revision: str = "a" * 40) -> dict[str, object]:
    source_root = "/private/runtime/toolkit/afdb_integration_kit/ipsae"
    binary_sha256 = "c" * 64
    command_result = {
        "argv": ["make", "-B", "-C", source_root, "CXX=g++"],
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
    }
    return {
        "format_version": 1,
        "source_revision": source_revision,
        "source_path": "afdb_integration_kit/ipsae",
        "build_command": ["make", "-B", "-C", source_root, "CXX=g++"],
        "toolchain": {
            "make": {**command_result, "argv": ["make", "--version"], "stdout": "GNU Make 4.4"},
            "cxx": {**command_result, "argv": ["g++", "--version"], "stdout": "g++ 14.2.0"},
        },
        "build_result": command_result,
        "build_log": {"path": "runtime-ipsae/build.log", "sha256": "b" * 64, "size_bytes": 128},
        "binary": {"path": "runtime-ipsae/ipsae_cpp", "sha256": binary_sha256, "size_bytes": 4096},
        "version": {
            "scheme": "source-revision+binary-sha256-v1",
            "source_revision": source_revision,
            "binary_sha256": binary_sha256,
            "output": f"ipsae_cpp source={source_revision} sha256={binary_sha256}",
        },
        "functional_test": {
            **command_result,
            "argv": [
                "/private/runtime/result/runtime-ipsae/ipsae_cpp",
                "--batch",
                "/private/runtime/result/runtime-ipsae/functional-fixture",
                "10.0",
                "8.0",
                "--summary",
                "/private/runtime/result/runtime-ipsae/functional-summary.csv",
                "--workers",
                "1",
                "--quiet",
            ],
            "fixture_sha256": RUNTIME_IPSAE_FIXTURE_SHA256,
            "result_sha256": RUNTIME_IPSAE_EXPECTED_RESULT_SHA256,
            "check": "paired-pdb-pae-semantic-v1",
            "expected_output": "one-row:BSPP-RQ-PAIR:ipsae_AB=1.000000:ipsae_BA=1.000000",
            "status": "passed",
            "model_id": RUNTIME_IPSAE_FIXTURE_MODEL_ID,
            "ipsae_ab": RUNTIME_IPSAE_EXPECTED_SCORE_AB,
            "ipsae_ba": RUNTIME_IPSAE_EXPECTED_SCORE_BA,
        },
    }


def write_runtime_ipsae_artifacts(result_path: Path, *, source_revision: str) -> dict[str, object]:
    root = result_path.parent / "runtime-ipsae"
    root.mkdir(parents=True, exist_ok=True)
    log = b"synthetic runtime build log\n"
    binary = b"#!/bin/sh\nexit 0\n"
    (root / "build.log").write_bytes(log)
    (root / "ipsae_cpp").write_bytes(binary)
    evidence = runtime_ipsae_evidence(source_revision=source_revision)
    evidence["build_log"] = {
        "path": "runtime-ipsae/build.log",
        "sha256": sha256(log).hexdigest(),
        "size_bytes": len(log),
    }
    evidence["binary"] = {
        "path": "runtime-ipsae/ipsae_cpp",
        "sha256": sha256(binary).hexdigest(),
        "size_bytes": len(binary),
    }
    evidence["version"] = {
        "scheme": "source-revision+binary-sha256-v1",
        "source_revision": source_revision,
        "binary_sha256": sha256(binary).hexdigest(),
        "output": f"ipsae_cpp source={source_revision} sha256={sha256(binary).hexdigest()}",
    }
    return evidence


def write_publication_compatibility_artifact(result_path: Path) -> dict[str, object]:
    evidence: dict[str, object] = {
        "schema_version": 1,
        "check": "postprocessing-directory-publication-v1",
        "status": "passed",
        "process_count": 2,
        "published_directory_count": 1,
        "fallback_errno": 22,
        "output_identity": {
            "path": "published/payload.json",
            "sha256": "bf04841124813309efa145d98b02da3eae3e3dd0ec4009a709c25d55af684c05",
            "size_bytes": 35,
        },
    }
    document = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    artifact_path = result_path.parent / "publication-compatibility.json"
    artifact_path.write_bytes(document)
    return {
        **evidence,
        "artifact": {
            "path": "publication-compatibility.json",
            "sha256": sha256(document).hexdigest(),
            "size_bytes": len(document),
        },
    }


def selected_source_identity(*, source_kind: str = "override", revision: str = "a" * 40) -> dict[str, object]:
    """Build a minimal selected-source identity mapping for tests."""
    result: dict[str, object] = {
        "source_kind": source_kind,
        "root": "/opt/afdb-toolkit" if source_kind == "baked" else "/workspace/AFDB-Integration-Kit",
        "revision": revision,
    }
    if source_kind == "baked":
        result["image_identity"] = {
            "format_version": 1,
            "policy": "digest-checked",
            "path": "/images/runtime.sqsh",
            "size_bytes": 1024,
            "sha256": "e" * 64,
        }
    else:
        result["toolkit_package_identity"] = {
            "format_version": 1,
            "format": "bspp-tar-v1",
            "verifier": "safe-tar-v1",
            "package_path": "/packages/toolkit.tar",
            "package_size_bytes": 10240,
            "package_sha256": "f" * 64,
            "manifest_sha256": "a" * 64,
            "commit": revision,
            "tree": "b" * 40,
            "package_role": "toolkit",
            "policy_version": 1,
        }
    return result


def write_baked_provenance(
    path: Path, *, commit: str = BAKED_TOOLKIT_COMMIT, filename: str = "provenance.json"
) -> Path:
    """Write a baked-provenance JSON file for in-container smoke testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_path = path / filename
    file_path.write_text(json.dumps({"commit": commit, "source": "afdb-toolkit", "format_version": 1}) + "\n")
    return file_path
