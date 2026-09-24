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

"""Exercise the BioIR image's real Python C-extension prerequisite smoke."""

from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def smoke(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("BSPP_CUDA_COMPAT_DIR", "/unused-cuda-compat")
    path = Path(__file__).resolve().parents[1] / "containers/folding/bioir/image-smoke.py"
    spec = importlib.util.spec_from_file_location("bioir_image_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_compiler() -> None:
    compiler = shlex.split(sysconfig.get_config_var("CC") or "cc")[0]
    if shutil.which(compiler) is None:
        pytest.skip(f"C compiler {compiler} is unavailable")


def test_python_development_compiles_and_imports_real_extension(smoke: ModuleType) -> None:
    _require_compiler()
    report = smoke._verify_python_development()
    assert report["header_version_hex"] == sys.hexversion
    assert report["extension_suffix"] == sysconfig.get_config_var("EXT_SUFFIX")


def test_python_development_rejects_missing_headers(
    smoke: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(smoke.sysconfig, "get_path", lambda key: str(tmp_path))
    with pytest.raises(RuntimeError, match=r"Python\.h is absent"):
        smoke._verify_python_development()


def test_python_development_preserves_compiler_failure(smoke: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    failure = subprocess.CalledProcessError(2, ["cc", "fixture.c"])

    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(smoke.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        smoke._verify_python_development()
    assert caught.value is failure


def test_python_development_rejects_header_interpreter_version_mismatch(
    smoke: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_compiler()
    monkeypatch.setattr(smoke.sys, "hexversion", sys.hexversion + 1)
    with pytest.raises(RuntimeError, match="do not match the running interpreter"):
        smoke._verify_python_development()
