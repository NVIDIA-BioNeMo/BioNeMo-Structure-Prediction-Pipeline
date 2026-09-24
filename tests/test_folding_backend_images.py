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

"""Offline tests for per-backend folding image lock files and smoke scripts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDING = ROOT / "containers" / "folding"
KERNEL_IMAGES = ("colabfold", "openfold-cli", "bioir")
PLACEHOLDER_DIGEST = "sha256:" + "0" * 64
RUNTIME_REAL_DIGEST = "sha256:6b3201183858bad08441837f5a5efc2c75290135cc25fcc87d9ff763190cfd09"

EXPECTED_BASE_REFS = {
    "runtime": "docker.io/nvidia/cuda:12.6.3-base-ubuntu24.04",
    "colabfold": "ghcr.io/sokrypton/colabfold:1.6.2-cuda12",
    "openfold-cli": "nvidia/cuda:12.1.1-devel-ubuntu22.04",
    "bioir": "nvidia/cuda:13.0.3-devel-ubuntu24.04",
}

KERNEL_REAL_DIGESTS = {
    "colabfold": "sha256:22f1243a8feca427b6942889d4dc3b549a3b75b23df9cf85c5f512a85e4f7f77",
    "openfold-cli": "sha256:7012e535a47883527d402da998384c30b936140c05e2537158c80b8143ee7425",
    "bioir": "sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266",
}


def test_kernel_locks_have_real_digest() -> None:
    for image in KERNEL_IMAGES:
        lock = json.loads((FOLDING / image / "image-lock.json").read_text())
        assert lock["base_image"]["linux_amd64_digest"] == KERNEL_REAL_DIGESTS[image], image


def test_runtime_lock_has_real_digest() -> None:
    lock = json.loads((FOLDING / "runtime" / "image-lock.json").read_text())
    assert lock["base_image"]["linux_amd64_digest"] == RUNTIME_REAL_DIGEST


def test_base_image_references_match_expected() -> None:
    for image, ref in EXPECTED_BASE_REFS.items():
        lock = json.loads((FOLDING / image / "image-lock.json").read_text())
        assert lock["base_image"]["reference"] == ref, image


def test_openfold_cli_lock_has_source_commit() -> None:
    lock = json.loads((FOLDING / "openfold-cli" / "image-lock.json").read_text())
    assert lock["openfold_source_commit"] == "be2ec1841f16c966c65ae0e7599ebbadc725757d"


def test_bioir_lock_has_version() -> None:
    lock = json.loads((FOLDING / "bioir" / "image-lock.json").read_text())
    assert lock["bioir_version"] == "0.1.0"


def test_colabfold_lock_has_version() -> None:
    lock = json.loads((FOLDING / "colabfold" / "image-lock.json").read_text())
    assert lock["colabfold_version"] == "1.6.2"


def test_all_locks_have_schema_version_1() -> None:
    for image in ("runtime", *KERNEL_IMAGES):
        lock = json.loads((FOLDING / image / "image-lock.json").read_text())
        assert lock["schema_version"] == 1, image


def test_no_openfold_trt_in_any_lock() -> None:
    for image in ("runtime", *KERNEL_IMAGES):
        lock = json.loads((FOLDING / image / "image-lock.json").read_text())
        assert "openfold_trt" not in lock, image


def test_runtime_lock_has_pixi_and_rsync() -> None:
    lock = json.loads((FOLDING / "runtime" / "image-lock.json").read_text())
    assert "pixi" in lock
    assert "rsync" in lock


def test_runtime_pixi_toml_is_trimmed() -> None:
    toml = (FOLDING / "runtime" / "pixi.toml").read_text()
    assert "[pypi-dependencies]" not in toml
    assert "colabfold" not in toml


def test_runtime_pixi_lock_absent() -> None:
    # The runtime image resolves its conda environment non-locked (no committed
    # pixi.lock); pixi is installed inside the Dockerfile.
    assert not (FOLDING / "runtime" / "pixi.lock").exists()


def test_runtime_smoke_no_colabfold_refs() -> None:
    smoke = (FOLDING / "runtime" / "image-smoke.py").read_text()
    assert "colabfold_version" not in smoke
    assert 'importlib.metadata.version("colabfold")' not in smoke
    assert "import colabfold" not in smoke
    assert "colabfold_batch" not in smoke
    assert "run_pretrained_openfold" not in smoke


def test_kernel_smokes_use_env_python3() -> None:
    for image in KERNEL_IMAGES:
        first_line = (FOLDING / image / "image-smoke.py").read_text().splitlines()[0]
        assert first_line == "#!/usr/bin/env python3", image


def test_kernel_smokes_read_compat_dir() -> None:
    for image in KERNEL_IMAGES:
        smoke = (FOLDING / image / "image-smoke.py").read_text()
        assert "BSPP_CUDA_COMPAT_DIR" in smoke, image


def test_kernel_smokes_assert_help() -> None:
    for image in KERNEL_IMAGES:
        smoke = (FOLDING / image / "image-smoke.py").read_text()
        assert "bspp-orchestration-runtime" in smoke, image


def test_kernel_locks_have_runtime_deps() -> None:
    for image in KERNEL_IMAGES:
        lock = json.loads((FOLDING / image / "image-lock.json").read_text())
        deps = lock.get("runtime_deps", {})
        assert "click" in deps, image
        assert "pydantic" in deps, image
        assert "pyyaml" in deps, image


def test_openfold_cli_lock_has_numpy_dep() -> None:
    lock = json.loads((FOLDING / "openfold-cli" / "image-lock.json").read_text())
    assert "numpy" in lock["runtime_deps"]


def test_openfold_cli_lock_has_torch_pin() -> None:
    lock = json.loads((FOLDING / "openfold-cli" / "image-lock.json").read_text())
    assert "torch" in lock
    assert lock["torch"]["spec"] == "torch==2.5.1+cu121"
    assert lock["torch"]["index_url"] == "https://download.pytorch.org/whl/cu121"


def test_bioir_lock_has_torch_pin() -> None:
    lock = json.loads((FOLDING / "bioir" / "image-lock.json").read_text())
    assert "torch" in lock
    assert lock["torch"]["spec"] == "torch==2.11.0+cu130"
    assert lock["torch"]["index_url"] == "https://download.pytorch.org/whl/cu130"


def test_openfold_cli_dockerfile_provisions_python312() -> None:
    """The openfold-cli base image is Ubuntu 22.04 (Python 3.10); the Dockerfile
    must provision Python 3.12 so the Contract/Runtime wheels install."""
    dockerfile = (FOLDING / "openfold-cli" / "Dockerfile").read_text()
    assert "python3.12" in dockerfile, "openfold-cli Dockerfile must provision Python 3.12"
    assert "deadsnakes" in dockerfile, "openfold-cli Dockerfile must use deadsnakes PPA"
    assert "update-alternatives" in dockerfile, "openfold-cli Dockerfile must set python/python3 to 3.12"
    # All pip installs must use python3.12 -m pip, not bare pip.
    assert "python3.12 -m pip" in dockerfile


def test_openfold_cli_build_sh_no_base_python_gate() -> None:
    """The build.sh must not gate on the base image's Python version since the
    Dockerfile provisions Python 3.12 itself."""
    build_sh = (FOLDING / "openfold-cli" / "build.sh").read_text()
    assert "BASE_PYTHON_VERSION" not in build_sh, "build.sh must not gate on base image Python"
    assert "Python 3.12" not in build_sh or "provisioned" in build_sh.lower()


def test_colabfold_dockerfile_no_self_symlink() -> None:
    """The colabfold Dockerfile must not create a dangling self-symlink for
    colabfold_batch."""
    dockerfile = (FOLDING / "colabfold" / "Dockerfile").read_text()
    # The broken pattern: ln -sf /usr/local/bin/colabfold_batch /usr/local/bin/colabfold_batch
    assert "ln -sf /usr/local/bin/colabfold_batch /usr/local/bin/colabfold_batch" not in dockerfile
    # The correct pattern must use command -v to discover the source path.
    assert "command -v colabfold_batch" in dockerfile


def test_cuda_versions() -> None:
    expected_cuda = {
        "runtime": "12.6.3",
        "colabfold": "12.x",
        "openfold-cli": "12.1",
        "bioir": "13.0",
    }
    for image, cuda in expected_cuda.items():
        lock = json.loads((FOLDING / image / "image-lock.json").read_text())
        assert lock["base_image"]["cuda_version"] == cuda, image


def test_folding_context_dirs_are_gitignored() -> None:
    """Each per-image build.sh creates a context/ directory that must be
    gitignored so the clean-checkout gate does not fail on the second image
    when building 'folding all' (review B1)."""
    gitignore = (ROOT / ".gitignore").read_text()
    assert "containers/folding/*/context/" in gitignore, (
        ".gitignore must include containers/folding/*/context/ to prevent "
        "untracked context/ dirs from breaking the clean-checkout gate"
    )


_S5CMD_SHA256 = "81d02a17a13797dc5949adb99734ad4217d005638a7827f36d435945527b2e69"


def test_kernel_images_bake_pinned_s5cmd() -> None:
    """All three folding kernel images bake s5cmd via the pinned-deb pattern."""
    for image in KERNEL_IMAGES:
        lock = json.loads((FOLDING / image / "image-lock.json").read_text())
        assert lock["s5cmd"]["version"] == "2.3.0", image
        assert lock["s5cmd"]["artifact_sha256"] == _S5CMD_SHA256, image
        assert lock["s5cmd"]["artifact_url"] == (
            "https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_linux_amd64.deb"
        ), image

        dockerfile = (FOLDING / image / "Dockerfile").read_text()
        assert "ARG S5CMD_VERSION=2.3.0" in dockerfile, image
        assert "ARG S5CMD_SHA256" in dockerfile, image
        assert "sha256sum --check --status" in dockerfile, image
        assert "dpkg -i /tmp/s5cmd.deb" in dockerfile, image

        build = (FOLDING / image / "build.sh").read_text()
        assert 'S5CMD_SHA="$(jq -r \'.s5cmd.artifact_sha256\' "$HERE/image-lock.json")"' in build, image
        assert 'S5CMD_VERSION="$(jq -r \'.s5cmd.version\' "$HERE/image-lock.json")"' in build, image
        assert '--build-arg "S5CMD_VERSION=$S5CMD_VERSION"' in build, image
        assert '--build-arg "S5CMD_SHA256=$S5CMD_SHA"' in build, image


def test_kernel_smokes_assert_s5cmd_where_they_assert_tools() -> None:
    """Colabfold and openfold-cli smokes already assert tool inventories;
    extend them for s5cmd. Bioir does not assert a tool inventory."""
    colabfold_smoke = (FOLDING / "colabfold" / "image-smoke.py").read_text()
    assert 'shutil.which("s5cmd")' in colabfold_smoke

    openfold_smoke = (FOLDING / "openfold-cli" / "image-smoke.py").read_text()
    assert '"s5cmd"' in openfold_smoke
