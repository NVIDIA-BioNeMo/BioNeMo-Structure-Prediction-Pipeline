#!/usr/bin/env python3
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

"""Self-contained local smoke for the openfold-cli kernel image composition."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path

_MANIFEST = Path("/opt/bspp/folding-runtime-image.json")
_CUDA_COMPAT_DIR = Path(os.environ["BSPP_CUDA_COMPAT_DIR"])
_LOCKED_EXECUTABLES = (Path("/usr/local/bin/run_pretrained_openfold.py"),)


def main() -> None:
    _verify_composition()
    manifest_bytes = _MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("folding openfold-cli image manifest is not schema version 1")
    image_lock_sha256 = manifest.get("image_lock_sha256")
    if not isinstance(image_lock_sha256, str) or len(image_lock_sha256) != 64:
        raise RuntimeError("folding openfold-cli image manifest has a malformed image_lock_sha256")
    int(image_lock_sha256, 16)
    _verify_distributions()
    python_version = _run(("python", "--version"))
    if not python_version.startswith("Python 3.12"):
        raise RuntimeError(f"image Python is not 3.12: {python_version}")
    _verify_executables()
    _verify_tools()
    _verify_help()
    report = {
        "schema_version": 1,
        "image_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "image_lock_sha256": image_lock_sha256,
        "python_version": python_version,
        "contract_version": importlib.metadata.version("bspp-orchestration-contract"),
        "runtime_version": importlib.metadata.version("bspp-orchestration-runtime"),
        "control_version": importlib.metadata.version("bspp-orchestration-control"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _verify_composition() -> None:
    """Check the CUDA compat loader policy (graceful if absent)."""
    if not _CUDA_COMPAT_DIR.exists():
        return
    try:
        compat_directory = _CUDA_COMPAT_DIR.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("CUDA compatibility directory cannot be resolved") from exc
    if not compat_directory.is_dir() or not os.access(compat_directory, os.R_OK | os.X_OK):
        raise RuntimeError("CUDA compatibility directory is not readable")
    compat_library = _CUDA_COMPAT_DIR / "libcuda.so.1"
    if not compat_library.is_symlink():
        raise RuntimeError("CUDA compatibility soname is not a symlink")
    compat_target = compat_library.resolve(strict=True)
    if not compat_target.is_file() or not os.access(compat_target, os.R_OK):
        raise RuntimeError("CUDA compatibility target is not a readable regular file")
    if not compat_target.is_relative_to(compat_directory):
        raise RuntimeError("CUDA compatibility target escapes its directory")


def _verify_distributions() -> None:
    importlib.metadata.version("bspp-orchestration-control")
    importlib.metadata.version("bspp-orchestration-contract")
    importlib.metadata.version("bspp-orchestration-runtime")
    import click  # noqa: F401
    import numpy  # noqa: F401
    import openfold  # noqa: F401
    import pydantic  # noqa: F401
    import yaml  # noqa: F401


def _verify_executables() -> None:
    for executable in _LOCKED_EXECUTABLES:
        if not executable.is_file() or not os.access(executable, os.R_OK | os.X_OK):
            raise RuntimeError(f"locked executable is unavailable: {executable}")


def _verify_tools() -> None:
    for tool in ("hhsearch", "hmmscan", "kalign", "s5cmd"):
        if not shutil.which(tool):
            raise RuntimeError(f"tool is not on PATH: {tool}")


def _verify_help() -> None:
    subprocess.run(
        ["bspp-orchestration-runtime", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )


def _run(argv: tuple[str, ...], *, first_line: bool = False) -> str:
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    value = (result.stdout or result.stderr).strip()
    return value.splitlines()[0] if first_line else value


if __name__ == "__main__":
    main()
