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

"""Offline behavioral contracts for the per-backend folding container images."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDING = ROOT / "containers" / "folding"
IMAGES = ("runtime", "colabfold", "openfold-cli", "bioir")
KERNEL_IMAGES = ("colabfold", "openfold-cli", "bioir")
PLACEHOLDER_DIGEST = "sha256:" + "0" * 64
RUNTIME_REAL_DIGEST = "sha256:6b3201183858bad08441837f5a5efc2c75290135cc25fcc87d9ff763190cfd09"
KERNEL_REAL_DIGESTS = {
    "colabfold": "sha256:22f1243a8feca427b6942889d4dc3b549a3b75b23df9cf85c5f512a85e4f7f77",
    "openfold-cli": "sha256:7012e535a47883527d402da998384c30b936140c05e2537158c80b8143ee7425",
    "bioir": "sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266",
}


def test_folding_image_manifests_present() -> None:
    for image in IMAGES:
        img_dir = FOLDING / image
        for name in ("build.sh", "Dockerfile", "image-lock.json", "entrypoint.sh", "smoke-local.sh"):
            assert (img_dir / name).is_file(), f"{image}/{name}"
        for name in ("image-smoke.py", ".dockerignore", "README.md"):
            assert (img_dir / name).is_file(), f"{image}/{name}"

    # Runtime image-lock.json has a real base digest (not placeholder).
    runtime_lock = json.loads((FOLDING / "runtime" / "image-lock.json").read_text())
    assert runtime_lock["schema_version"] == 1
    assert runtime_lock["base_image"]["linux_amd64_digest"] == RUNTIME_REAL_DIGEST

    # Kernel image-lock.json files have real pinned base digests.
    for image in KERNEL_IMAGES:
        lock = json.loads((FOLDING / image / "image-lock.json").read_text())
        assert lock["schema_version"] == 1
        assert lock["base_image"]["linux_amd64_digest"] == KERNEL_REAL_DIGESTS[image]


def test_folding_image_scripts_are_executable() -> None:
    for image in IMAGES:
        img_dir = FOLDING / image
        for name in ("build.sh", "entrypoint.sh", "smoke-local.sh", "image-smoke.py"):
            path = img_dir / name
            assert path.is_file(), f"{image}/{name}"
            assert os.access(path, os.X_OK), f"{image}/{name} is not executable"


def test_runtime_image_has_pixi_files() -> None:
    assert (FOLDING / "runtime" / "pixi.toml").is_file()
    assert not (FOLDING / "runtime" / "pixi.lock").exists()
    toml = (FOLDING / "runtime" / "pixi.toml").read_text()
    assert "[pypi-dependencies]" not in toml
    assert "colabfold" not in toml


def test_runtime_image_smoke_no_colabfold_refs() -> None:
    smoke = (FOLDING / "runtime" / "image-smoke.py").read_text()
    assert "colabfold_batch" not in smoke
    assert "run_pretrained_openfold" not in smoke
    assert "colabfold_version" not in smoke
    assert 'importlib.metadata.version("colabfold")' not in smoke
    assert "import colabfold" not in smoke


def test_common_build_routes_folding_images(tmp_path: Path) -> None:
    """build.sh folding <image> delegates to containers/folding/<image>/build.sh."""
    repo = tmp_path / "repo"
    scripts = repo / "containers" / "scripts"
    folding = repo / "containers" / "folding"
    bin_dir = tmp_path / "bin"
    scripts.mkdir(parents=True)
    folding.mkdir(parents=True)
    bin_dir.mkdir()
    shutil.copy2(ROOT / "containers" / "scripts" / "build.sh", scripts / "build.sh")

    for image in IMAGES:
        (folding / image).mkdir(parents=True)

    builder_logs: dict[str, Path] = {}
    for image in IMAGES:
        builder_log = tmp_path / f"builder-{image}.json"
        builder_logs[image] = builder_log
        _write_executable(
            folding / image / "build.sh",
            f"""#!/usr/bin/env python3
import json
import os
import sys

with open(os.environ["BUILDER_LOG_{image}"], "w", encoding="utf-8") as stream:
    json.dump(sys.argv[1:], stream)
""",
        )

    enroot_log = tmp_path / "enroot.json"
    _write_executable(
        bin_dir / "enroot",
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

with open(os.environ["ENROOT_LOG"], "w", encoding="utf-8") as stream:
    json.dump(sys.argv[1:], stream)
output = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
output.write_bytes(b"sqsh")
""",
    )

    for image in IMAGES:
        output = tmp_path / f"folding-{image}.sqsh"
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            f"BUILDER_LOG_{image}": str(builder_logs[image]),
            "ENROOT_LOG": str(enroot_log),
        }
        result = subprocess.run(
            [
                str(scripts / "build.sh"),
                "folding",
                image,
                "--squashfs",
                str(output),
                "--no-cache",
                "--progress=plain",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{image}: {result.stderr}"
        expected_local = f"bspp-orchestration:folding-{image}"
        assert json.loads(builder_logs[image].read_text()) == [expected_local, "--no-cache", "--progress=plain"]
        assert json.loads(enroot_log.read_text()) == ["import", "-o", str(output), f"dockerd://{expected_local}"]
        assert output.read_bytes() == b"sqsh"


def test_push_sh_folding_image_branch_present() -> None:
    push = ROOT / "containers" / "scripts" / "push.sh"
    result = subprocess.run(["bash", "-n", str(push)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    content = push.read_text()
    assert "folding" in content
    assert "build_and_push_folding_image" in content
    # Verify the function is actually defined (not just referenced as a string).
    assert "build_and_push_folding_image()" in content
    # The old monolithic function must be gone.
    assert "build_and_push_folding()" not in content
    # The old monolithic variables must be gone.
    assert "FOLDING_LOCAL_IMAGE=" not in content
    assert "FOLDING_REGISTRY_IMAGE=" not in content
    assert "FOLDING_BUILD_RECORD=" not in content
    # The matrix discovers dedicated builders (preprocessing/postprocessing/
    # folding), not generic variant .env files.
    assert "containers/folding" in content


def test_build_push_pull_folding_syntax() -> None:
    scripts = ROOT / "containers" / "scripts"
    targets = [scripts / "build.sh", scripts / "push.sh", scripts / "pull-sqsh.sh"]
    for image in IMAGES:
        targets.append(FOLDING / image / "build.sh")
        targets.append(FOLDING / image / "smoke-local.sh")
        targets.append(FOLDING / image / "entrypoint.sh")
    for target in targets:
        result = subprocess.run(["bash", "-n", str(target)], capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"{target}: {result.stderr}"


def test_folding_no_monolithic_files() -> None:
    for name in (
        "Dockerfile",
        "build.sh",
        "pixi.toml",
        "pixi.lock",
        "image-lock.json",
        "entrypoint.sh",
        "image-smoke.py",
        "smoke-local.sh",
        ".dockerignore",
    ):
        assert not (FOLDING / name).exists(), f"monolithic file should be deleted: {name}"


def test_openfold_trt_has_no_image() -> None:
    assert not (FOLDING / "openfold-trt").is_dir()


def test_kernel_image_locks_have_runtime_deps() -> None:
    for image in KERNEL_IMAGES:
        lock = json.loads((FOLDING / image / "image-lock.json").read_text())
        deps = lock["runtime_deps"]
        assert "click" in deps
        assert "pydantic" in deps
        assert "pyyaml" in deps
    # openfold-cli also has numpy
    openfold_lock = json.loads((FOLDING / "openfold-cli" / "image-lock.json").read_text())
    assert "numpy" in openfold_lock["runtime_deps"]


def test_kernel_image_locks_have_schema_version() -> None:
    for image in KERNEL_IMAGES:
        lock = json.loads((FOLDING / image / "image-lock.json").read_text())
        assert lock["schema_version"] == 1


def test_kernel_image_smoke_uses_env_python3_and_compat_dir() -> None:
    for image in KERNEL_IMAGES:
        smoke = (FOLDING / image / "image-smoke.py").read_text()
        first_line = (FOLDING / image / "image-smoke.py").read_text().splitlines()[0]
        assert first_line == "#!/usr/bin/env python3", f"{image} smoke shebang"
        assert "BSPP_CUDA_COMPAT_DIR" in smoke, f"{image} smoke must read BSPP_CUDA_COMPAT_DIR"
        assert "bspp-orchestration-runtime" in smoke, f"{image} smoke must assert --help works"


def test_base_image_references() -> None:
    expected = {
        "runtime": "docker.io/nvidia/cuda:12.6.3-base-ubuntu24.04",
        "colabfold": "ghcr.io/sokrypton/colabfold:1.6.2-cuda12",
        "openfold-cli": "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        "bioir": "nvidia/cuda:13.0.3-devel-ubuntu24.04",
    }
    for image, ref in expected.items():
        lock = json.loads((FOLDING / image / "image-lock.json").read_text())
        assert lock["base_image"]["reference"] == ref, f"{image} base_image.reference"


def test_openfold_cli_lock_has_source_commit() -> None:
    lock = json.loads((FOLDING / "openfold-cli" / "image-lock.json").read_text())
    assert lock["openfold_source_commit"] == "be2ec1841f16c966c65ae0e7599ebbadc725757d"
    assert "pdbfixer_ref" in lock
    assert "torch" in lock
    assert "spec" in lock["torch"]
    assert "index_url" in lock["torch"]


def test_bioir_lock_has_version_and_torch() -> None:
    lock = json.loads((FOLDING / "bioir" / "image-lock.json").read_text())
    assert lock["bioir_version"] == "0.1.0"
    assert "torch" in lock
    assert "spec" in lock["torch"]
    assert "index_url" in lock["torch"]


def test_colabfold_lock_has_version() -> None:
    lock = json.loads((FOLDING / "colabfold" / "image-lock.json").read_text())
    assert lock["colabfold_version"] == "1.6.2"


def test_no_openfold_trt_in_any_lock() -> None:
    for image in IMAGES:
        lock = json.loads((FOLDING / image / "image-lock.json").read_text())
        assert "openfold_trt" not in lock


def test_runtime_lock_has_pixi_and_rsync() -> None:
    lock = json.loads((FOLDING / "runtime" / "image-lock.json").read_text())
    assert "pixi" in lock
    assert "rsync" in lock


def test_runtime_dockerfile_pins_and_verifies_pixi_binary() -> None:
    """The pixi binary is pinned and sha256-verified, never curl|bash."""
    dockerfile = (FOLDING / "runtime" / "Dockerfile").read_text()
    assert "https://pixi.sh/install.sh" not in dockerfile
    assert "PIXI_URL" in dockerfile
    assert "PIXI_SHA256" in dockerfile
    assert "sha256sum --check --status" in dockerfile


def test_openfold_cli_has_patch_setup() -> None:
    assert (FOLDING / "openfold-cli" / "patch_setup.py").is_file()


def test_openfold_cli_dockerfile_makes_run_pretrained_openfold_executable() -> None:
    """The executor invokes run_pretrained_openfold.py directly, so the image must
    prepend a shebang and set the executable bit before symlinking it onto PATH."""
    dockerfile = (FOLDING / "openfold-cli" / "Dockerfile").read_text()
    assert "sed -i '1i #!/usr/bin/env python3' /opt/openfold/run_pretrained_openfold.py" in dockerfile
    assert "chmod +x /opt/openfold/run_pretrained_openfold.py" in dockerfile
    assert "ln -sf /opt/openfold/run_pretrained_openfold.py /usr/local/bin/run_pretrained_openfold.py" in dockerfile


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
