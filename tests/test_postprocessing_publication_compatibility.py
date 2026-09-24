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

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_publication_compatibility_module_proves_forced_fallback(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "publication-compatibility"
    output = tmp_path.resolve() / "publication-compatibility.json"
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "bspp.orchestration.runtime.postprocessing.publication_compatibility",
            "--root",
            str(root),
            "--output",
            str(output),
        ),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "packages/orchestration-contract/src:packages/orchestration-runtime/src"},
    )
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(output.read_text())
    assert evidence == {
        "check": "postprocessing-directory-publication-v1",
        "fallback_errno": 22,
        "output_identity": {
            "path": "published/payload.json",
            "sha256": "bf04841124813309efa145d98b02da3eae3e3dd0ec4009a709c25d55af684c05",
            "size_bytes": 35,
        },
        "process_count": 2,
        "published_directory_count": 1,
        "schema_version": 1,
        "status": "passed",
    }
    assert output.stat().st_size <= 4096
    assert not tuple(root.glob("stage-*"))


def test_publication_compatibility_rejects_nonfixed_output(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "publication-compatibility"
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "bspp.orchestration.runtime.postprocessing.publication_compatibility",
            "--root",
            str(root),
            "--output",
            str(tmp_path.resolve() / "wrong.json"),
        ),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "packages/orchestration-contract/src:packages/orchestration-runtime/src"},
    )
    assert completed.returncode != 0
    assert not (tmp_path / "wrong.json").exists()
