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

"""Behavioral checks for the scalar image's actual folding entrypoint smoke."""

from __future__ import annotations

import importlib.util
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def smoke() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "containers/folding/runtime/image-smoke.py"
    spec = importlib.util.spec_from_file_location("folding_runtime_image_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_folding_modules_and_legacy_command_reject_empty_handoff(smoke: ModuleType) -> None:
    report = smoke._verify_folding_entrypoints()
    assert report["numpy_version"].startswith("2.")
    assert report["legacy_import_missing_handoff_rejected"] is True


def test_folding_entrypoints_preserve_missing_dependency(smoke: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    failure = ModuleNotFoundError("No module named 'numpy'", name="numpy")

    def fail(name: str) -> None:
        raise failure

    monkeypatch.setattr(smoke.importlib, "import_module", fail)
    with pytest.raises(ModuleNotFoundError) as caught:
        smoke._verify_folding_entrypoints()
    assert caught.value is failure


@pytest.mark.parametrize("publish_output", [False, True])
def test_folding_entrypoints_reject_unexpected_legacy_result(
    smoke: ModuleType, monkeypatch: pytest.MonkeyPatch, publish_output: bool
) -> None:
    original_run = subprocess.run

    def unexpected(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "legacy-msa-import" not in argv:
            return original_run(argv, **kwargs)  # type: ignore[call-overload,no-any-return]
        if publish_output:
            Path(argv[argv.index("--output-dir") + 1]).mkdir()
            return subprocess.CompletedProcess(
                argv, 1, "", "legacy MSA handoff record must be a regular file: artifact-set.json"
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(smoke.subprocess, "run", unexpected)
    message = "published output" if publish_output else "did not reject missing handoff records"
    with pytest.raises(RuntimeError, match=message):
        smoke._verify_folding_entrypoints()
