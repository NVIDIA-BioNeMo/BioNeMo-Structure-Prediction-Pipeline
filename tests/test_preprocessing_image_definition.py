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

"""Static reproducibility assertions for the dedicated preprocessing image."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "containers" / "preprocessing"
CHARACTERIZATION = ROOT / "containers" / "scripts" / "slurm-preprocessing-carry-characterization.sh"


def test_image_lock_cites_immutable_primary_scientific_provenance() -> None:
    lock = json.loads((IMAGE / "image-lock.json").read_text())
    assert lock["colabfold"]["source_commit"] == "c7d1772352cc9619df25c6d36cb0f218c0c6610e"
    assert lock["mmseqs"]["source_commit"] == "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
    assert lock["mmseqs"]["artifact_sha256"] == "83969dd5c7d4c32858c2fc9a4d1024c15e8fe5da768ce76e787ab0195ffd64e7"
    assert lock["base_image"]["linux_amd64_digest"].startswith("sha256:")
    dockerfile = (IMAGE / "Dockerfile").read_text()
    assert lock["base_image"]["linux_amd64_digest"] in dockerfile
    cuda_major_minor = ".".join(lock["base_image"]["cuda_version"].split(".")[:2])
    pinned_compatibility_directory = f"/usr/local/cuda-{cuda_major_minor}/compat"
    assert pinned_compatibility_directory in (IMAGE / "entrypoint.sh").read_text()
    assert pinned_compatibility_directory in (IMAGE / "image-smoke.py").read_text()
    assert pinned_compatibility_directory in (IMAGE / "cuda-driver-probe.py").read_text()


def test_rsync_is_exactly_pinned_and_cross_bound_to_the_single_linux_lock_entry() -> None:
    manifest = tomllib.loads((IMAGE / "pixi.toml").read_text())
    assert manifest["dependencies"]["rsync"] == {"version": "==3.4.4", "build": "h5440a77_0"}

    pixi_lock = yaml.safe_load((IMAGE / "pixi.lock").read_text())
    rsync_entries = tuple(
        item for item in pixi_lock["packages"] if "conda" in item and Path(item["conda"]).name.startswith("rsync-")
    )
    assert len(rsync_entries) == 1
    locked = rsync_entries[0]
    assert Path(locked["conda"]).name == "rsync-3.4.4-h5440a77_0.conda"
    assert re.fullmatch(r"[0-9a-f]{64}", locked["sha256"])

    image_lock = json.loads((IMAGE / "image-lock.json").read_text())
    assert image_lock["rsync"] == {
        "version": "3.4.4",
        "build": "h5440a77_0",
        "artifact_url": locked["conda"],
        "artifact_sha256": locked["sha256"],
    }


def test_image_exposes_locked_rsync_and_records_documentation_only_version() -> None:
    dockerfile = (IMAGE / "Dockerfile").read_text()
    build = (IMAGE / "build.sh").read_text()
    smoke = (IMAGE / "image-smoke.py").read_text()

    assert "ln -sf /opt/bspp/environment/bin/rsync /usr/bin/rsync" in dockerfile
    # No host-side pixi: the rsync pin is documentation-only authority read from
    # image-lock.json, and the pixi binary is downloaded inside the Dockerfile.
    assert "command -v pixi" not in build
    assert "pixi list --locked" not in build
    assert "curl -fsSL" not in build
    assert '--build-arg "PIXI_URL=$PIXI_URL"' in build
    assert '--build-arg "MMSEQS_URL=$MMSEQS_URL"' in build
    image_manifest_build, distribution_build_record = build.split(
        '> "$CONTEXT/preprocessing-runtime-image.json"', maxsplit=1
    )
    assert "rsync_version: $rsync_version" in image_manifest_build
    assert "rsync_version" not in distribution_build_record
    assert '--arg image_id "$IMAGE_ID"' in distribution_build_record
    assert "oci_digest" not in distribution_build_record
    assert "RepoDigests" not in build
    assert 'Path("/usr/bin/rsync")' in smoke
    assert '_run(("/usr/bin/rsync", "--version"), first_line=True)' in smoke
    assert "normalize_rsync_version" in smoke


def test_s5cmd_is_pinned_and_baked_into_the_preprocessing_image() -> None:
    """The preprocessing image bakes s5cmd via the same pinned-deb pattern as
    folding/runtime, because in-job preprocessing paths (stage-input FASTA
    download and s3 database intake) reach require_tool('s5cmd')."""
    lock = json.loads((IMAGE / "image-lock.json").read_text())
    assert lock["s5cmd"]["version"] == "2.3.0"
    assert lock["s5cmd"]["artifact_sha256"] == "81d02a17a13797dc5949adb99734ad4217d005638a7827f36d435945527b2e69"
    assert lock["s5cmd"]["artifact_url"] == (
        "https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_linux_amd64.deb"
    )

    dockerfile = (IMAGE / "Dockerfile").read_text()
    assert "ARG S5CMD_VERSION=2.3.0" in dockerfile
    assert "ARG S5CMD_SHA256" in dockerfile
    assert (
        "github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_linux_amd64.deb" in dockerfile
    )
    assert "sha256sum --check --status" in dockerfile
    assert "dpkg -i /tmp/s5cmd.deb" in dockerfile
    assert "rm /tmp/s5cmd.deb" in dockerfile

    build = (IMAGE / "build.sh").read_text()
    assert 'S5CMD_SHA="$(jq -r \'.s5cmd.artifact_sha256\' "$HERE/image-lock.json")"' in build
    assert 'S5CMD_VERSION="$(jq -r \'.s5cmd.version\' "$HERE/image-lock.json")"' in build
    assert '--build-arg "S5CMD_VERSION=$S5CMD_VERSION"' in build
    assert '--build-arg "S5CMD_SHA256=$S5CMD_SHA"' in build

    smoke = (IMAGE / "image-smoke.py").read_text()
    assert 'Path("/usr/bin/s5cmd")' in smoke


def test_common_container_tooling_delegates_preprocessing_to_the_dedicated_builder() -> None:
    common_build = (ROOT / "containers" / "scripts" / "build.sh").read_text()
    common_push = (ROOT / "containers" / "scripts" / "push.sh").read_text()
    common_pull = (ROOT / "containers" / "scripts" / "pull-sqsh.sh").read_text()

    assert '[[ "$VARIANT" == "preprocessing" ]]' in common_build
    assert '"${REPO_ROOT}/containers/preprocessing/build.sh"' in common_build
    assert 'IMAGE_NAME="bspp-orchestration:preprocessing"' in common_build
    assert 'PREPROCESSING_LOCAL_IMAGE="bspp-orchestration:preprocessing"' in common_push
    assert 'docker tag "$record_image_id" "$PREPROCESSING_REGISTRY_IMAGE"' in common_push
    assert "containers/scripts/pull-sqsh.sh preprocessing" in common_pull


def test_locked_image_bakes_contract_control_runtime_and_no_source_overlay() -> None:
    lock = yaml.safe_load((IMAGE / "pixi.lock").read_text())
    urls = json.dumps(lock)
    for package in ("python-3.12", "bash-5.2", "coreutils-9", "tar-1.35", "lz4-c-1.10", "util-linux-2"):
        assert package in urls
    assert "name: colabfold\n  version: 1.6.2" in (IMAGE / "pixi.lock").read_text()
    dockerfile = (IMAGE / "Dockerfile").read_text()
    assert "pip install --no-deps" in dockerfile
    assert "bspp-orchestration-control" in dockerfile
    assert 'importlib.metadata.version("bspp-orchestration-control")' in dockerfile
    assert "context/wheels/*.whl" in dockerfile
    assert "COPY packages" not in dockerfile and "MMSA" not in dockerfile
    assert "dist/" in (IMAGE / ".dockerignore").read_text()
    smoke = (IMAGE / "smoke-local.sh").read_text()
    assert "--network none" in smoke and "source=" not in smoke and "run --rm" in smoke


def test_carry_characterization_selects_real_paired_gpu_server_and_monitors_accounting() -> None:
    script = CHARACTERIZATION.read_text()
    helper = (IMAGE / "carry-characterization.sh").read_text()
    assert ">AFDB_alpha\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL:" in script
    assert ">AFDB_mu\nMNKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE:" in script
    assert "--pair-mode paired" in helper
    assert "--gpu-server 1" in helper
    assert '"$fixture/search-output"' in helper
    assert '"$COLABFOLD_SEARCH" --mmseqs "$MMSEQS"' in helper
    assert "--threads 64" in helper
    assert 'TIMEOUT="/opt/bspp/environment/bin/timeout"' in helper
    assert helper.count('"$TIMEOUT" --signal=TERM --kill-after=10s 1800s') == 1
    assert '"$TIMEOUT" --signal=TERM --kill-after=10s 900s' not in helper
    assert script.count("${BSPP_CARRY_TIME:-00:45:00}") == 1
    assert "${BSPP_CARRY_TIME:-00:30:00}" not in script
    assert script.count('"search_timeout_seconds": 1800') == 1
    assert '"search_timeout_seconds": 900' not in script
    assert "/bin/bash /opt/bspp/bin/bspp-preprocessing-carry-characterization" in script
    assert "bash -lc" not in script
    assert 'expected_search_inventory = tuple(sorted((*expected, "2.a3m", "3.a3m")))' in script
    assert 'placeholder_bytes = {"2.a3m": b"#49\\t1\\n", "3.a3m": b"#36\\t1\\n"}' in script
    assert "if sequence_count < 2:" not in script
    assert "sequence_count = validate_a3m(data, label=str(path))" in script
    assert '"sequence_count": sequence_count' in script
    assert '"raw_search_inventory": search_inventory' in script
    assert '[[ "$fixture_sha" == "$BSPP_CARRY_INPUT_SHA256" ]]' in script
    assert 'job_id="${job_id%%;*}"' in script
    assert "for _ in $(seq 1 12)" in script
    assert "malformed characterized multimer metadata" in script
    assert "${BSPP_CARRY_CPUS:-30}" in script
    assert "${BSPP_CARRY_MEMORY:-128G}" in script
    assert "host-slurm-job-id.txt" in script
    assert "host-slurmd-nodename.txt" in script
    assert "host-cuda-visible-devices.txt" in script
    assert "host-nvidia-smi.csv" in script
    assert '"$host_job_id" "$host_node" "$host_visible_devices" "$host_gpu_rows" "$wrapper_sha"' in script


def test_carry_characterization_assembles_evidence_with_locked_image_python() -> None:
    script = CHARACTERIZATION.read_text()
    smoke = (IMAGE / "image-smoke.py").read_text()
    pinned_python = "/opt/bspp/environment/bin/python"
    locked_executables = smoke.split("_LOCKED_CHARACTERIZATION_EXECUTABLES = (\n", maxsplit=1)[1].split(
        ")\n\n", maxsplit=1
    )[0]
    container_boundary = """\
srun \\
  --container-image="$BSPP_CARRY_IMAGE" \\
  --container-mounts="$mounts" \\
  --no-container-mount-home \\
  /usr/local/bin/entrypoint.sh \\
"""

    assert f'Path("{pinned_python}")' in locked_executables
    assert script.count("\nsrun \\\n") == 2
    assert script.count(container_boundary) == 2
    assert f"  {pinned_python} - \\\n" in script
    assert "\npython3 - \\\n" not in script
    assert '  "$fixture" "$destination" \\\n' in script
    assert '  "$host_job_id" "$host_node" "$host_visible_devices" "$host_gpu_rows" "$wrapper_sha" <<\'PY\'' in script


def test_preprocessing_image_installs_locked_characterization_helpers() -> None:
    dockerfile = (IMAGE / "Dockerfile").read_text()
    smoke = (IMAGE / "image-smoke.py").read_text()
    for installed_path in (
        "/opt/bspp/bin/bspp-preprocessing-carry-characterization",
        "/opt/bspp/bin/bspp-preprocessing-cuda-driver-probe",
    ):
        assert installed_path in dockerfile
        assert installed_path in smoke


def test_characterization_helper_absolute_executables_are_bound_to_image_smoke() -> None:
    helper = (IMAGE / "carry-characterization.sh").read_text()
    smoke = (IMAGE / "image-smoke.py").read_text()
    assignments = dict(re.findall(r'^([A-Z_]+)="(/[^"]+)"$', helper, flags=re.MULTILINE))

    for name in ("PYTHON", "TIMEOUT", "SLEEP", "TAIL", "MMSEQS", "COLABFOLD_SEARCH", "CUDA_DRIVER_PROBE"):
        assert name in assignments
        assert f'Path("{assignments[name]}")' in smoke


def test_preprocessing_image_neither_installs_driver_utilities_nor_probes_them_inside() -> None:
    dockerfile = (IMAGE / "Dockerfile").read_text().lower()
    image_smoke = (IMAGE / "image-smoke.py").read_text()

    assert "nvidia-utils" not in dockerfile
    assert "nvidia-driver" not in dockerfile
    assert "nvidia-smi" not in dockerfile
    # curl is installed only to download the pinned pixi/mmseqs/s5cmd binaries
    # and is purged again inside the same layer.
    assert "apt-get install -y --no-install-recommends curl" in dockerfile
    assert "apt-get purge -y --auto-remove curl" in dockerfile
    assert '_run(("nvidia-smi' not in image_smoke
    assert "BSPP_PREPROCESSING_GPU_EVIDENCE" in image_smoke


def test_image_smoke_opts_out_of_the_afdb_stem_gate_for_its_synthetic_members() -> None:
    smoke = (IMAGE / "image-smoke.py").read_text()

    assert "PreprocessingScientificConfig(require_afdb_model_id_stem=False)" in smoke
    assert "PreprocessingScientificConfig()" not in smoke
