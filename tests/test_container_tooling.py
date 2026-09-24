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

"""Behavioral contracts for the shared container build, push, and import tools."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCAL_IMAGE = "bspp-orchestration:preprocessing"
BSPP_REGISTRY = "registry.example.com"
BSPP_IMAGE_REPOSITORY = "bspp-orchestration"
REGISTRY_REPOSITORY = f"{BSPP_REGISTRY}/{BSPP_IMAGE_REPOSITORY}"
REGISTRY_IMAGE = f"{REGISTRY_REPOSITORY}:preprocessing"
TOOLKIT_REPO = "https://example.invalid/org/AFDB-Integration-Kit.git"
IMAGE_ID = "sha256:" + "1" * 64
OCI_DIGEST = "sha256:" + "2" * 64


def test_dedicated_backend_consumes_image_and_forwards_build_arguments(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    image_root = repo / "containers" / "preprocessing"
    bin_dir = tmp_path / "bin"
    image_root.mkdir(parents=True)
    bin_dir.mkdir()
    shutil.copy2(ROOT / "containers" / "preprocessing" / "build.sh", image_root / "build.sh")

    rsync_sha = "3" * 64
    pixi_sha = "6" * 64
    mmseqs_sha = "5" * 64
    image_lock = {
        "base_image": {"cuda_version": "12.6.3"},
        "colabfold": {"version": "1.6.2"},
        "mmseqs": {
            "source_commit": "4" * 40,
            "artifact_url": "https://fixtures.invalid/mmseqs",
            "artifact_sha256": mmseqs_sha,
        },
        "pixi": {
            "artifact_url": "https://fixtures.invalid/pixi",
            "artifact_sha256": pixi_sha,
        },
        "rsync": {
            "version": "3.4.4",
            "build": "h5440a77_0",
            "artifact_url": "https://fixtures.invalid/rsync.conda",
            "artifact_sha256": rsync_sha,
        },
    }
    (image_root / "image-lock.json").write_text(json.dumps(image_lock))
    (image_root / "pixi.toml").write_text('[workspace]\nplatforms = ["linux-64"]\n')
    (repo / ".gitignore").write_text("containers/preprocessing/context/\ncontainers/preprocessing/dist/\n")

    _write_executable(
        bin_dir / "uv",
        """#!/usr/bin/env bash
set -euo pipefail
package=""
out_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --package) package="$2"; shift ;;
    --out-dir) out_dir="$2"; shift ;;
  esac
  shift
done
mkdir -p "$out_dir"
case "$package" in
  bspp-orchestration-contract) printf contract >"$out_dir/bspp_orchestration_contract-0.0.0.whl" ;;
  bspp-orchestration-control) printf control >"$out_dir/bspp_orchestration_control-0.0.0.whl" ;;
  bspp-orchestration-runtime) printf runtime >"$out_dir/bspp_orchestration_runtime-0.0.0.whl" ;;
  *) exit 2 ;;
esac
""",
    )
    engine_log = tmp_path / "engine.jsonl"
    _write_executable(
        bin_dir / "fake-engine",
        f"""#!/usr/bin/env python3
import json
import os
import sys

with open(os.environ["ENGINE_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:4] == ["image", "inspect", "--format"]:
    print({IMAGE_ID!r})
""",
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Container Tooling Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")

    build_arguments = ["--no-cache", "--progress=plain", "--build-arg", "EXAMPLE=value"]
    result = subprocess.run(
        [str(image_root / "build.sh"), "chosen-preprocessing:image", *build_arguments],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CONTAINER_ENGINE": "fake-engine",
            "ENGINE_LOG": str(engine_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    engine_calls = [json.loads(line) for line in engine_log.read_text().splitlines()]
    # Caller build arguments are forwarded, but the lock-derived build arguments
    # and the tag come last so they take precedence over any caller override.
    assert all(argument in engine_calls[0] for argument in build_arguments)
    assert engine_calls[0][-3:] == ["--tag", "chosen-preprocessing:image", str(image_root)]
    assert "PIXI_URL=https://fixtures.invalid/pixi" in engine_calls[0]
    assert f"PIXI_SHA256={pixi_sha}" in engine_calls[0]
    assert "MMSEQS_URL=https://fixtures.invalid/mmseqs" in engine_calls[0]
    assert f"MMSEQS_SHA256={mmseqs_sha}" in engine_calls[0]
    assert engine_calls[1] == ["image", "inspect", "--format", "{{.Id}}", "chosen-preprocessing:image"]
    assert all("RepoDigests" not in argument for call in engine_calls for argument in call)

    record = json.loads((image_root / "dist" / "build-record.json").read_text())
    assert record["image"] == "chosen-preprocessing:image"
    assert record["image_id"] == IMAGE_ID
    assert "oci_digest" not in record
    assert "registry_image" not in record


def test_common_build_routes_preprocessing_and_reuses_squashfs_conversion(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    scripts = repo / "containers" / "scripts"
    preprocessing = repo / "containers" / "preprocessing"
    bin_dir = tmp_path / "bin"
    scripts.mkdir(parents=True)
    preprocessing.mkdir(parents=True)
    bin_dir.mkdir()
    shutil.copy2(ROOT / "containers" / "scripts" / "build.sh", scripts / "build.sh")

    builder_log = tmp_path / "builder.json"
    _write_executable(
        preprocessing / "build.sh",
        """#!/usr/bin/env python3
import json
import os
import sys

with open(os.environ["BUILDER_LOG"], "w", encoding="utf-8") as stream:
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
    output = tmp_path / "preprocessing.sqsh"
    result = subprocess.run(
        [
            str(scripts / "build.sh"),
            "preprocessing",
            "--squashfs",
            str(output),
            "--no-cache",
            "--progress=plain",
        ],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "BUILDER_LOG": str(builder_log),
            "ENROOT_LOG": str(enroot_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(builder_log.read_text()) == [LOCAL_IMAGE, "--no-cache", "--progress=plain"]
    assert json.loads(enroot_log.read_text()) == ["import", "-o", str(output), f"dockerd://{LOCAL_IMAGE}"]
    assert output.read_bytes() == b"sqsh"


def test_common_build_routes_postprocessing_to_dedicated_backend(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    scripts = repo / "containers" / "scripts"
    postprocessing = repo / "containers" / "postprocessing"
    bin_dir = tmp_path / "bin"
    scripts.mkdir(parents=True)
    postprocessing.mkdir(parents=True)
    bin_dir.mkdir()
    shutil.copy2(ROOT / "containers" / "scripts" / "build.sh", scripts / "build.sh")

    builder_log = tmp_path / "builder.json"
    _write_executable(
        postprocessing / "build.sh",
        """#!/usr/bin/env python3
import json
import os
import sys

with open(os.environ["BUILDER_LOG"], "w", encoding="utf-8") as stream:
    json.dump(sys.argv[1:], stream)
""",
    )
    build_arguments = ["--no-cache", "--progress=plain", "--build-arg", "EXAMPLE=value"]
    result = subprocess.run(
        [str(scripts / "build.sh"), "postprocessing", *build_arguments],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "BUILDER_LOG": str(builder_log)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(builder_log.read_text()) == ["bspp-orchestration:postprocessing", *build_arguments]


def test_common_build_unknown_target_lists_all_explicit_targets() -> None:
    result = subprocess.run(
        [str(ROOT / "containers" / "scripts" / "build.sh"), "unknown"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "postprocessing preprocessing" in result.stderr


def test_preprocessing_push_tags_fresh_image_and_atomically_supplements_record(tmp_path: Path) -> None:
    digests = [
        "example.invalid/unrelated@sha256:" + "9" * 64,
        f"{REGISTRY_REPOSITORY}@{OCI_DIGEST}",
        f"{REGISTRY_REPOSITORY}@{OCI_DIGEST}",
    ]
    result, repo, docker_calls, initial_record, _ = _run_preprocessing_push(tmp_path, digests=digests)

    assert result.returncode == 0, result.stderr
    assert ["tag", IMAGE_ID, REGISTRY_IMAGE] in docker_calls
    assert ["push", REGISTRY_IMAGE] in docker_calls
    builder_args = json.loads((tmp_path / "builder.json").read_text())
    assert builder_args == [LOCAL_IMAGE, "--no-cache", "--progress=plain"]

    record = json.loads((repo / "containers" / "preprocessing" / "dist" / "build-record.json").read_text())
    assert {key: record[key] for key in initial_record} == initial_record
    assert record["registry_image"] == REGISTRY_IMAGE
    assert record["oci_digest"] == OCI_DIGEST


@pytest.mark.parametrize(
    "digests",
    [
        None,
        [],
        ["example.invalid/unrelated@sha256:" + "4" * 64],
        [f"{REGISTRY_REPOSITORY}@sha256:short"],
        [f"{REGISTRY_REPOSITORY}@sha256:" + "A" * 64],
        [f"{REGISTRY_REPOSITORY}@{OCI_DIGEST}", f"{REGISTRY_REPOSITORY}@sha256:" + "5" * 64],
    ],
)
def test_preprocessing_push_rejects_missing_malformed_or_ambiguous_digest(
    tmp_path: Path, digests: list[str] | None
) -> None:
    result, repo, _, _, initial_record_bytes = _run_preprocessing_push(tmp_path, digests=digests)

    assert result.returncode != 0
    record_path = repo / "containers" / "preprocessing" / "dist" / "build-record.json"
    assert record_path.read_bytes() == initial_record_bytes
    assert not tuple(record_path.parent.glob("build-record.json.tmp.*"))


def test_preprocessing_push_rejects_valid_digest_prefix_followed_by_invalid_json(
    tmp_path: Path,
) -> None:
    valid_prefix = json.dumps([{"RepoDigests": [f"{REGISTRY_REPOSITORY}@{OCI_DIGEST}"]}])
    result, repo, docker_calls, _, initial_record_bytes = _run_preprocessing_push(
        tmp_path,
        digests=None,
        inspect_stdout=f"{valid_prefix}\ntruncated{{",
    )

    assert result.returncode != 0
    assert ["push", REGISTRY_IMAGE] in docker_calls
    record_path = repo / "containers" / "preprocessing" / "dist" / "build-record.json"
    assert record_path.read_bytes() == initial_record_bytes
    assert not tuple(record_path.parent.glob("build-record.json.tmp.*"))


def test_preprocessing_push_rejects_registry_tag_identity_change_before_push(tmp_path: Path) -> None:
    result, repo, docker_calls, _, initial_record_bytes = _run_preprocessing_push(
        tmp_path,
        digests=[f"{REGISTRY_REPOSITORY}@{OCI_DIGEST}"],
        registry_image_id="sha256:" + "6" * 64,
    )

    assert result.returncode != 0
    assert ["push", REGISTRY_IMAGE] not in docker_calls
    record_path = repo / "containers" / "preprocessing" / "dist" / "build-record.json"
    assert record_path.read_bytes() == initial_record_bytes


def test_preprocessing_push_rejects_local_tag_identity_change_before_tagging(tmp_path: Path) -> None:
    result, repo, docker_calls, _, initial_record_bytes = _run_preprocessing_push(
        tmp_path,
        digests=[f"{REGISTRY_REPOSITORY}@{OCI_DIGEST}"],
        local_image_id="sha256:" + "8" * 64,
    )

    assert result.returncode != 0
    assert ["tag", IMAGE_ID, REGISTRY_IMAGE] not in docker_calls
    assert ["push", REGISTRY_IMAGE] not in docker_calls
    record_path = repo / "containers" / "preprocessing" / "dist" / "build-record.json"
    assert record_path.read_bytes() == initial_record_bytes


@pytest.mark.parametrize(
    "build_record",
    [
        {
            "image": LOCAL_IMAGE,
            "source_commit": "7" * 40,
            "nested_provenance": {"preserved": True},
        },
        {
            "image": LOCAL_IMAGE,
            "image_id": "sha256:short",
            "source_commit": "7" * 40,
            "nested_provenance": {"preserved": True},
        },
        {
            "image": LOCAL_IMAGE,
            "image_id": "sha256:" + "3" * 64,
            "source_commit": "7" * 40,
            "nested_provenance": {"preserved": True},
        },
    ],
    ids=["missing", "malformed", "mismatched"],
)
def test_preprocessing_push_rejects_invalid_build_record_image_id_without_mutating_record(
    tmp_path: Path, build_record: dict[str, object]
) -> None:
    result, repo, docker_calls, _, initial_record_bytes = _run_preprocessing_push(
        tmp_path,
        digests=[f"{REGISTRY_REPOSITORY}@{OCI_DIGEST}"],
        build_record=build_record,
    )

    assert result.returncode != 0
    assert not any(call[0] in {"tag", "push"} for call in docker_calls)
    record_path = repo / "containers" / "preprocessing" / "dist" / "build-record.json"
    assert record_path.read_bytes() == initial_record_bytes


def test_push_all_matrix_builds_postprocessing_as_dedicated_target(tmp_path: Path) -> None:
    # `push.sh all` treats postprocessing as a dedicated target (like
    # preprocessing/folding): it builds + pushes with a build-record, not a
    # bare generic digest.txt.
    fixture = _matrix_fixture(tmp_path)
    build_arguments = ["--no-cache", "--progress=plain"]

    result = _run_matrix(fixture, *build_arguments, evidence_dir=tmp_path / "evidence")

    assert result.returncode == 0, result.stderr
    calls = _docker_calls(fixture)
    pushes = [call for call in calls if call and call[0] == "push"]
    # Push order is the matrix order: preprocessing, postprocessing, folding.
    assert pushes == [
        ["push", f"{REGISTRY_REPOSITORY}:preprocessing"],
        ["push", f"{REGISTRY_REPOSITORY}:postprocessing"],
        ["push", f"{REGISTRY_REPOSITORY}:folding-runtime"],
        ["push", f"{REGISTRY_REPOSITORY}:folding-bioir"],
    ]
    # The summary reports per-image status; postprocessing has a composition
    # smoke like the other dedicated targets.
    assert "postprocessing " in result.stdout and "smoke=pass" in result.stdout
    # Postprocessing records a build-record.json (dedicated target) enriched
    # with the pushed registry identity, plus a local-smoke.json.
    postprocessing_evidence = tmp_path / "evidence" / "postprocessing"
    record = json.loads((postprocessing_evidence / "build-record.json").read_text())
    assert record["oci_digest"] == MATRIX_DIGEST
    assert json.loads((postprocessing_evidence / "local-smoke.json").read_text()) == {"ok": True}
    digest_txt = (postprocessing_evidence / "digest.txt").read_text()
    assert f"oci_digest: {MATRIX_DIGEST}" in digest_txt
    assert f"registry_image: {REGISTRY_REPOSITORY}:postprocessing" in digest_txt


def test_push_all_default_evidence_dir_is_gitignored() -> None:
    """The default evidence dir is gitignored so per-image evidence does not
    dirty the tree and break the next dedicated build's clean-tree check."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "devdocs/evidence/sample-run-plans/container-rebuild-*/evidence/" in gitignore


def test_pull_sqsh_imports_preprocessing_and_reports_hash(tmp_path: Path) -> None:
    container_dir = tmp_path / "containers"
    result, enroot_call = _run_pull_sqsh(tmp_path, container_dir=container_dir)

    assert result.returncode == 0, result.stderr
    output = container_dir / "bspp-orchestration-preprocessing.sqsh"
    expected_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    assert enroot_call == [
        "import",
        "-o",
        str(output),
        f"docker://{BSPP_REGISTRY}/{BSPP_IMAGE_REPOSITORY}:preprocessing",
    ]
    assert f"SHA-256: {expected_hash}" in result.stdout


def test_pull_sqsh_import_reference_uses_import_registry_override(tmp_path: Path) -> None:
    # enroot 3.4.1 rejects registry:port/repo:tag references at parse time;
    # the override swaps only the enroot-side registry host.
    container_dir = tmp_path / "containers"
    import_registry = "registry-import.example.com"
    result, enroot_call = _run_pull_sqsh(tmp_path, container_dir=container_dir, import_registry=import_registry)

    assert result.returncode == 0, result.stderr
    assert enroot_call == [
        "import",
        "-o",
        str(container_dir / "bspp-orchestration-preprocessing.sqsh"),
        f"docker://{import_registry}/{BSPP_IMAGE_REPOSITORY}:preprocessing",
    ]


def test_pull_sqsh_import_reference_defaults_to_push_registry(tmp_path: Path) -> None:
    container_dir = tmp_path / "containers"
    result, enroot_call = _run_pull_sqsh(tmp_path, container_dir=container_dir)

    assert result.returncode == 0, result.stderr
    assert enroot_call[-1] == f"docker://{BSPP_REGISTRY}/{BSPP_IMAGE_REPOSITORY}:preprocessing"


def test_pull_sqsh_honors_container_image_output_override(tmp_path: Path) -> None:
    output = tmp_path / "custom" / "preprocessing-runtime.sqsh"
    result, enroot_call = _run_pull_sqsh(tmp_path, output=output)

    assert result.returncode == 0, result.stderr
    assert enroot_call == [
        "import",
        "-o",
        str(output),
        f"docker://{BSPP_REGISTRY}/{BSPP_IMAGE_REPOSITORY}:preprocessing",
    ]
    assert output.read_bytes() == b"imported-preprocessing-sqsh"
    assert f"SHA-256: {hashlib.sha256(output.read_bytes()).hexdigest()}" in result.stdout


def test_preprocessing_smoke_keeps_tmp_executable_for_fake_scientific_kernels(tmp_path: Path) -> None:
    image_root = tmp_path / "containers" / "preprocessing"
    bin_dir = tmp_path / "bin"
    image_root.mkdir(parents=True)
    (image_root / "dist").mkdir()
    bin_dir.mkdir()
    shutil.copy2(ROOT / "containers" / "preprocessing" / "smoke-local.sh", image_root / "smoke-local.sh")
    _write_executable(
        bin_dir / "fake-engine",
        """#!/usr/bin/env python3
import sys

arguments = sys.argv[1:]
if arguments[:2] == ["image", "inspect"]:
    raise SystemExit(0)
if arguments[:1] != ["run"]:
    raise SystemExit(2)
if "--mount" in arguments:
    mount = arguments[arguments.index("--mount") + 1]
    if mount.startswith("type=tmpfs,destination=/tmp") and "exec" not in mount.split(","):
        print("Permission denied: '/tmp/bspp-preprocessing-smoke/bin/mmseqs'", file=sys.stderr)
        raise SystemExit(126)
if "--tmpfs" in arguments:
    mount = arguments[arguments.index("--tmpfs") + 1]
    if mount.split(":", 1)[0] == "/tmp" and "exec" not in mount.split(":", 1)[-1].split(","):
        print("Permission denied: '/tmp/bspp-preprocessing-smoke/bin/mmseqs'", file=sys.stderr)
        raise SystemExit(126)
print("{}")
""",
    )

    result = subprocess.run(
        [str(image_root / "smoke-local.sh"), LOCAL_IMAGE],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "CONTAINER_ENGINE": "fake-engine"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _run_preprocessing_push(
    tmp_path: Path,
    *,
    digests: list[str] | None,
    local_image_id: str = IMAGE_ID,
    registry_image_id: str = IMAGE_ID,
    build_record: dict[str, object] | None = None,
    inspect_stdout: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, list[list[str]], dict[str, object], bytes]:
    repo = tmp_path / "repo"
    scripts = repo / "containers" / "scripts"
    preprocessing = repo / "containers" / "preprocessing"
    dist = preprocessing / "dist"
    bin_dir = tmp_path / "bin"
    scripts.mkdir(parents=True)
    dist.mkdir(parents=True)
    bin_dir.mkdir()
    shutil.copy2(ROOT / "containers" / "scripts" / "push.sh", scripts / "push.sh")

    initial_record = build_record or {
        "image": LOCAL_IMAGE,
        "image_id": IMAGE_ID,
        "source_commit": "7" * 40,
        "nested_provenance": {"preserved": True, "values": [1, 2, 3]},
    }
    initial_record_bytes = json.dumps(initial_record, separators=(",", ":")).encode() + b"\n"
    builder_log = tmp_path / "builder.json"
    _write_executable(
        preprocessing / "build.sh",
        f"""#!/usr/bin/env python3
import json
import os
import pathlib
import sys

with open(os.environ["BUILDER_LOG"], "w", encoding="utf-8") as stream:
    json.dump(sys.argv[1:], stream)
path = pathlib.Path(__file__).parent / "dist" / "build-record.json"
path.write_bytes({initial_record_bytes!r})
""",
    )

    docker_log = tmp_path / "docker.jsonl"
    inspect_payload = [{}] if digests is None else [{"RepoDigests": digests}]
    inspect_output = inspect_stdout if inspect_stdout is not None else json.dumps(inspect_payload)
    _write_executable(
        bin_dir / "docker",
        f"""#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["DOCKER_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[:4] == ["image", "inspect", "--format", "{{{{.Id}}}}"]:
    print({registry_image_id!r} if args[4] == {REGISTRY_IMAGE!r} else {local_image_id!r})
elif args == ["image", "inspect", {REGISTRY_IMAGE!r}]:
    sys.stdout.write({inspect_output!r})
""",
    )
    docker_config = tmp_path / "home" / ".docker" / "config.json"
    docker_config.parent.mkdir(parents=True)
    docker_config.write_text(json.dumps({"auths": {BSPP_REGISTRY: {}}}))

    result = subprocess.run(
        [str(scripts / "push.sh"), "preprocessing", "--no-cache", "--progress=plain"],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "HOME": str(tmp_path / "home"),
            "BUILDER_LOG": str(builder_log),
            "DOCKER_LOG": str(docker_log),
            "BSPP_REGISTRY": BSPP_REGISTRY,
            "BSPP_IMAGE_REPOSITORY": BSPP_IMAGE_REPOSITORY,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    docker_calls = [json.loads(line) for line in docker_log.read_text().splitlines()] if docker_log.exists() else []
    return result, repo, docker_calls, initial_record, initial_record_bytes


def _generic_tooling_fixture(tmp_path: Path, script_name: str) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    containers = repo / "containers"
    scripts = containers / "scripts"
    variants = containers / "variants"
    manifests = containers / "pyprojects"
    preprocessing = containers / "preprocessing"
    bin_dir = tmp_path / "bin"
    for directory in (scripts, variants, manifests, preprocessing, bin_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ROOT / "containers" / "scripts" / script_name, scripts / script_name)
    shutil.copy2(ROOT / "containers" / "scripts" / "ensure-ssh-agent.sh", scripts / "ensure-ssh-agent.sh")
    (containers / "Dockerfile").write_text("FROM scratch\n")
    for variant, base_image, cuda_version in (("postprocessing", "fixture/postprocessing:base", "13.0"),):
        manifest = f"containers/pyprojects/{variant}.toml"
        (manifests / f"{variant}.toml").write_text("[workspace]\n")
        (variants / f"{variant}.env").write_text(
            "\n".join(
                [
                    f"VARIANT_TAG={variant}",
                    f"BASE_IMAGE={base_image}",
                    f"PIXI_MANIFEST={manifest}",
                    f"CUDA_VERSION={cuda_version}",
                    "PLATFORMS=linux/amd64",
                    "CUDA_COMPAT_PACKAGE=",
                    "CUDA_COMPAT_DIR=",
                    "",
                ]
            )
        )

    preprocessing_marker = tmp_path / "preprocessing-invoked"
    _write_executable(
        preprocessing / "build.sh",
        f"""#!/usr/bin/env python3
import pathlib

pathlib.Path({str(preprocessing_marker)!r}).write_text("invoked")
raise SystemExit(97)
""",
    )
    docker_log = tmp_path / "docker.jsonl"
    _write_executable(
        bin_dir / "docker",
        """#!/usr/bin/env python3
import json
import os
import sys

with open(os.environ["DOCKER_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
""",
    )
    _write_executable(
        bin_dir / "dpkg",
        """#!/usr/bin/env bash
printf '%s\\n' amd64
""",
    )
    _write_executable(
        bin_dir / "ssh-add",
        """#!/usr/bin/env bash
exit 0
""",
    )
    return repo, bin_dir, docker_log, preprocessing_marker


def _run_pull_sqsh(
    tmp_path: Path,
    *,
    container_dir: Path | None = None,
    output: Path | None = None,
    import_registry: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
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
output.write_bytes(b"imported-preprocessing-sqsh")
""",
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "BSPP_ORCH": str(ROOT),
        "CONTAINER_DIR": str(container_dir or tmp_path / "containers"),
        "SLURM_TMPDIR": str(tmp_path / "slurm"),
        "ENROOT_LOG": str(enroot_log),
        "BSPP_REGISTRY": BSPP_REGISTRY,
        "BSPP_IMAGE_REPOSITORY": BSPP_IMAGE_REPOSITORY,
    }
    if output is not None:
        env["CONTAINER_IMAGE"] = str(output)
    if import_registry is not None:
        env["BSPP_REGISTRY_IMPORT"] = import_registry
    result = subprocess.run(
        [str(ROOT / "containers" / "scripts" / "pull-sqsh.sh"), "preprocessing"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    # A fail-closed rejection before enroot runs leaves no log behind.
    calls = json.loads(enroot_log.read_text()) if enroot_log.exists() else []
    return result, calls


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# push.sh all — the full-matrix canonical command
# ---------------------------------------------------------------------------

MATRIX_IMAGE_ID = "sha256:" + "1" * 64
MATRIX_DIGEST = "sha256:" + "2" * 64


def _matrix_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A fake repo running the REAL push.sh matrix mode with stubbed tools.

    The repo is a real git repository with a fetchable local `origin` (the
    commit-sync preflight is fail-closed and must be able to fetch). Docker,
    the dedicated builders, and the smokes are fakes; jq and git are real.
    """
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    scripts = repo / "containers" / "scripts"
    variants = repo / "containers" / "variants"
    manifests = repo / "containers" / "pyprojects"
    preprocessing = repo / "containers" / "preprocessing"
    postprocessing = repo / "containers" / "postprocessing"
    bin_dir = tmp_path / "bin"
    for directory in (scripts, variants, manifests, preprocessing, postprocessing, bin_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "containers" / "scripts" / "push.sh", scripts / "push.sh")
    shutil.copy2(ROOT / "containers" / "scripts" / "ensure-ssh-agent.sh", scripts / "ensure-ssh-agent.sh")
    (repo / "containers" / "Dockerfile").write_text("FROM scratch\n")

    for variant in ("postprocessing",):
        (manifests / f"{variant}.toml").write_text("[workspace]\n")
        (variants / f"{variant}.env").write_text(
            "\n".join(
                [
                    f"VARIANT_TAG={variant}",
                    f"BASE_IMAGE=fixture/{variant}:base",
                    f"PIXI_MANIFEST=containers/pyprojects/{variant}.toml",
                    "CUDA_VERSION=12.9",
                    "PLATFORMS=linux/amd64",
                    "CUDA_COMPAT_PACKAGE=",
                    "CUDA_COMPAT_DIR=",
                    "",
                ]
            )
        )

    builder = """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

with open(os.environ["BUILDER_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
here = pathlib.Path(__file__).parent
if here.name == "preprocessing":
    record_image = "bspp-orchestration:preprocessing"
elif here.name == "postprocessing":
    record_image = "bspp-orchestration:postprocessing"
else:
    record_image = f"bspp-orchestration:folding-{here.name}"
registry_root = os.environ["BSPP_REGISTRY"] + "/" + os.environ["BSPP_IMAGE_REPOSITORY"]
registry_image = record_image.replace("bspp-orchestration:", registry_root + ":")
if here.name == "preprocessing":
    target = "preprocessing"
elif here.name == "postprocessing":
    target = "postprocessing"
else:
    target = f"folding:{here.name}"
if os.environ.get("FAIL_ON") == target:
    raise SystemExit(1)
dist = here / "dist"
dist.mkdir(exist_ok=True)
(dist / "build-record.json").write_text(json.dumps({
    "image": record_image,
    "image_id": "__IMAGE_ID__",
    "source_commit": "__SOURCE_COMMIT__",
}))
""".replace("__IMAGE_ID__", MATRIX_IMAGE_ID).replace("__SOURCE_COMMIT__", "7" * 40)
    smoke = """#!/usr/bin/env bash
set -euo pipefail
echo "$1" >> "${SMOKE_LOG}"
mkdir -p "$(dirname "${BASH_SOURCE[0]}")/dist"
printf '{"ok": true}\\n' > "$(dirname "${BASH_SOURCE[0]}")/dist/local-smoke.json"
"""
    _write_executable(preprocessing / "build.sh", builder)
    _write_executable(preprocessing / "smoke-local.sh", smoke)
    _write_executable(postprocessing / "build.sh", builder)
    _write_executable(postprocessing / "smoke-local.sh", smoke)
    for image in ("runtime", "bioir"):
        image_dir = repo / "containers" / "folding" / image
        image_dir.mkdir(parents=True)
        _write_executable(image_dir / "build.sh", builder)
        _write_executable(image_dir / "smoke-local.sh", smoke)

    _write_executable(
        bin_dir / "docker",
        """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["DOCKER_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\\n")
if not args:
    raise SystemExit(0)
if args[0] in ("info", "login"):
    raise SystemExit(0)
if args[:4] == ["image", "inspect", "--format", "{{.Id}}"]:
    print("__IMAGE_ID__")
elif args[:2] == ["image", "inspect"]:
    repo = os.environ["BSPP_REGISTRY"] + "/" + os.environ["BSPP_IMAGE_REPOSITORY"]
    override = os.environ.get("MOCK_INSPECT_REPODIGESTS")
    if override and args[-1] == repo + ":postprocessing":
        sys.stdout.write(override)
    else:
        sys.stdout.write(json.dumps([{"RepoDigests": [f"{repo}@" + "__DIGEST__"]}]))
raise SystemExit(0)
""".replace("__IMAGE_ID__", MATRIX_IMAGE_ID).replace("__DIGEST__", MATRIX_DIGEST),
    )
    _write_executable(bin_dir / "curl", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(bin_dir / "uv", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(bin_dir / "dpkg", "#!/usr/bin/env bash\nprintf '%s\\n' amd64\n")
    _write_executable(bin_dir / "ssh-add", "#!/usr/bin/env bash\nexit 0\n")

    _git(repo, "init", "-b", "wip")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "fixture")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "fixture")
    _git(repo, "init", "--bare", str(remote))
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "origin", "HEAD:main")

    docker_config = tmp_path / "home" / ".docker" / "config.json"
    docker_config.parent.mkdir(parents=True)
    docker_config.write_text(json.dumps({"auths": {BSPP_REGISTRY: {}}}))

    return repo, bin_dir, tmp_path / "docker.jsonl", tmp_path / "smoke.log"


def _run_matrix(
    fixture: tuple[Path, Path, Path, Path],
    *arguments: str,
    fail_on: str = "",
    evidence_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    repo, bin_dir, docker_log, smoke_log = fixture
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(repo.parent / "home"),
        "DOCKER_LOG": str(docker_log),
        "BUILDER_LOG": str(repo.parent / "builder.jsonl"),
        "SMOKE_LOG": str(smoke_log),
        "FAIL_ON": fail_on,
        "BSPP_REGISTRY": BSPP_REGISTRY,
        "BSPP_IMAGE_REPOSITORY": BSPP_IMAGE_REPOSITORY,
        "TOOLKIT_REPO": TOOLKIT_REPO,
    }
    argv = [str(repo / "containers" / "scripts" / "push.sh"), "all", *arguments]
    if evidence_dir is not None:
        argv += ["--evidence-dir", str(evidence_dir)]
    return subprocess.run(argv, env=env, capture_output=True, text=True, check=False)


def _docker_calls(fixture: tuple[Path, Path, Path, Path]) -> list[list[str]]:
    docker_log = fixture[2]
    if not docker_log.exists():
        return []
    return [json.loads(line) for line in docker_log.read_text().splitlines()]


def test_push_all_matrix_dry_run_covers_dedicated_images_and_executes_nothing(tmp_path: Path) -> None:
    fixture = _matrix_fixture(tmp_path)
    result = _run_matrix(fixture, "--dry-run")

    assert result.returncode == 0, result.stderr
    plan = result.stdout
    assert "- preprocessing: push.sh preprocessing" in plan
    assert "- postprocessing: push.sh postprocessing" in plan
    assert plan.index("folding runtime") < plan.index("folding bioir")
    assert "smoke: containers/preprocessing/smoke-local.sh" in plan
    assert "smoke: containers/folding/bioir/smoke-local.sh" in plan
    assert "generic postprocessing" not in plan
    assert _docker_calls(fixture) == []


def test_push_all_matrix_preflight_fails_closed_when_fetch_fails(tmp_path: Path) -> None:
    fixture = _matrix_fixture(tmp_path)
    repo = fixture[0]
    _git(repo, "remote", "remove", "origin")

    result = _run_matrix(fixture)

    assert result.returncode == 1
    assert "could not fetch origin main" in result.stderr
    assert _docker_calls(fixture) == []


def test_push_all_matrix_preflight_fails_closed_when_diverged_from_origin_main(tmp_path: Path) -> None:
    fixture = _matrix_fixture(tmp_path)
    repo = fixture[0]
    # Move remote main to a commit the checkout does not contain.
    _git(repo, "checkout", "--orphan", "remote-moved")
    _git(repo, "commit", "--allow-empty", "-m", "remote moved on")
    _git(repo, "push", "--force", "origin", "HEAD:main")
    _git(repo, "checkout", "wip")

    result = _run_matrix(fixture)

    assert result.returncode == 1
    assert "not an ancestor" in result.stderr
    assert _docker_calls(fixture) == []


def test_push_all_matrix_preflight_fails_closed_on_dirty_tree_before_any_build(tmp_path: Path) -> None:
    fixture = _matrix_fixture(tmp_path)
    repo = fixture[0]
    (repo / "untracked.txt").write_text("dirty")

    result = _run_matrix(fixture)

    assert result.returncode == 1
    assert "not clean" in result.stderr
    assert _docker_calls(fixture) == []


def test_push_all_matrix_source_preflight_runs_before_registry_auth(tmp_path: Path) -> None:
    """A failing source-state check must prevent any registry-auth invocation.

    The fixture's docker config carries the registry in `auths`; rewrite it
    empty so the login block WOULD authenticate if reached, then fail the
    clean-tree check and assert no login call was recorded.
    """
    fixture = _matrix_fixture(tmp_path)
    repo = fixture[0]
    docker_config = repo.parent / "home" / ".docker" / "config.json"
    docker_config.write_text(json.dumps({"auths": {}}))
    (repo / "untracked.txt").write_text("dirty")

    result = _run_matrix(fixture)

    assert result.returncode == 1
    assert "not clean" in result.stderr
    assert _docker_calls(fixture) == []


def test_push_all_matrix_registry_auth_follows_successful_preflight(tmp_path: Path) -> None:
    """On a clean, in-sync tree the login runs after the preflight, before pushes."""
    fixture = _matrix_fixture(tmp_path)
    repo = fixture[0]
    docker_config = repo.parent / "home" / ".docker" / "config.json"
    docker_config.write_text(json.dumps({"auths": {}}))

    result = _run_matrix(fixture)

    assert result.returncode == 0, result.stderr
    calls = _docker_calls(fixture)
    login_calls = [call for call in calls if call and call[0] == "login"]
    assert login_calls == [["login", BSPP_REGISTRY]]
    push_index = next(i for i, call in enumerate(calls) if call and call[0] == "push")
    login_index = calls.index(login_calls[0])
    assert login_index < push_index
    # The preflight ran first: its output precedes the login line.
    assert result.stdout.index("== preflight ==") < result.stdout.index("Logging in")


def test_push_all_matrix_stops_on_first_failure_and_names_the_step(tmp_path: Path) -> None:
    fixture = _matrix_fixture(tmp_path)
    result = _run_matrix(fixture, fail_on="folding:bioir")

    assert result.returncode == 1
    assert "FAILED at: folding:bioir (push)" in result.stderr
    calls = _docker_calls(fixture)
    pushes = [call for call in calls if call and call[0] == "push"]
    assert pushes == [
        ["push", f"{REGISTRY_REPOSITORY}:preprocessing"],
        ["push", f"{REGISTRY_REPOSITORY}:postprocessing"],
        ["push", f"{REGISTRY_REPOSITORY}:folding-runtime"],
    ]
    assert "status=pushed" in result.stderr or "status=pushed" in result.stdout
    assert "folding:bioir" in result.stderr


def test_push_all_matrix_full_run_captures_evidence_and_accurate_summary(tmp_path: Path) -> None:
    fixture = _matrix_fixture(tmp_path)
    evidence_dir = tmp_path / "evidence"
    result = _run_matrix(fixture, evidence_dir=evidence_dir)

    assert result.returncode == 0, result.stderr
    for target, tag in (
        ("preprocessing", "preprocessing"),
        ("folding:runtime", "folding-runtime"),
        ("folding:bioir", "folding-bioir"),
    ):
        record = json.loads((evidence_dir / target / "build-record.json").read_text())
        assert record["oci_digest"] == MATRIX_DIGEST
        assert json.loads((evidence_dir / target / "local-smoke.json").read_text()) == {"ok": True}
        digest_txt = (evidence_dir / target / "digest.txt").read_text()
        assert f"oci_digest: {MATRIX_DIGEST}" in digest_txt
        assert f"registry_image: {REGISTRY_REPOSITORY}:{tag}" in digest_txt
        assert f"{target} " in result.stdout
        assert "status=pushed" in result.stdout
    # Postprocessing is a dedicated target with a build-record and smoke.
    postprocessing_record = json.loads((evidence_dir / "postprocessing" / "build-record.json").read_text())
    assert postprocessing_record["oci_digest"] == MATRIX_DIGEST
    assert json.loads((evidence_dir / "postprocessing" / "local-smoke.json").read_text()) == {"ok": True}
    postprocessing_digest_txt = (evidence_dir / "postprocessing" / "digest.txt").read_text()
    assert f"oci_digest: {MATRIX_DIGEST}" in postprocessing_digest_txt
    assert f"registry_image: {REGISTRY_REPOSITORY}:postprocessing" in postprocessing_digest_txt
    assert "digest=sha256:" in result.stdout
    assert "smoke=pass" in result.stdout
    assert "All built images pushed; per-image status above is the record." in result.stdout


# ---------------------------------------------------------------------------
# push.sh single-variant --include-postprocessing-internal
# ---------------------------------------------------------------------------


def _postprocessing_single_fixture(tmp_path: Path, *, with_internal: bool = True) -> tuple[Path, Path, Path, Path]:
    """A fake repo for single-variant `push.sh postprocessing`.

    The dedicated builder, smoke, and docker are stubs.  When *with_internal*
    is True a fake ``containers/nvidia/build-internal.sh`` is installed so
    the internal build path can be exercised.
    """
    repo = tmp_path / "repo"
    scripts = repo / "containers" / "scripts"
    postprocessing = repo / "containers" / "postprocessing"
    nvidia_dir = repo / "containers" / "nvidia"
    bin_dir = tmp_path / "bin"
    for directory in (scripts, postprocessing, nvidia_dir, bin_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "containers" / "scripts" / "push.sh", scripts / "push.sh")

    internal_marker = tmp_path / "internal-invoked"
    builder = """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

log = os.environ.get("BUILDER_LOG")
if log:
    with open(log, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(sys.argv[1:]) + "\\n")
here = pathlib.Path(__file__).parent
dist = here / "dist"
dist.mkdir(exist_ok=True)
(dist / "build-record.json").write_text(json.dumps({
    "image": "bspp-orchestration:postprocessing",
    "image_id": "__IMAGE_ID__",
    "source_commit": "__SOURCE_COMMIT__",
}))
""".replace("__IMAGE_ID__", MATRIX_IMAGE_ID).replace("__SOURCE_COMMIT__", "7" * 40)
    _write_executable(postprocessing / "build.sh", builder)

    smoke = """#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$(dirname "${BASH_SOURCE[0]}")/dist"
printf '{"ok": true}\\n' > "$(dirname "${BASH_SOURCE[0]}")/dist/local-smoke.json"
"""
    _write_executable(postprocessing / "smoke-local.sh", smoke)

    if with_internal:
        _write_executable(
            nvidia_dir / "build-internal.sh",
            f"""#!/usr/bin/env bash
set -euo pipefail
echo "internal build marker"
printf 'invoked' > "{internal_marker!s}"
""",
        )

    _write_executable(
        bin_dir / "docker",
        """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["DOCKER_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\\n")
if not args:
    raise SystemExit(0)
if args[0] in ("info", "login"):
    raise SystemExit(0)
if args[:4] == ["image", "inspect", "--format", "{{.Id}}"]:
    print("__IMAGE_ID__")
elif args[:2] == ["image", "inspect"]:
    repo = os.environ["BSPP_REGISTRY"] + "/" + os.environ["BSPP_IMAGE_REPOSITORY"]
    sys.stdout.write(json.dumps([{"RepoDigests": [f"{repo}@__DIGEST__"]}]))
raise SystemExit(0)
""".replace("__IMAGE_ID__", MATRIX_IMAGE_ID).replace("__DIGEST__", MATRIX_DIGEST),
    )

    docker_config = tmp_path / "home" / ".docker" / "config.json"
    docker_config.parent.mkdir(parents=True)
    docker_config.write_text(json.dumps({"auths": {BSPP_REGISTRY: {}}}))

    return repo, bin_dir, tmp_path / "docker.jsonl", internal_marker


def _run_single_postprocessing(
    fixture: tuple[Path, Path, Path, Path], *arguments: str
) -> subprocess.CompletedProcess[str]:
    repo, bin_dir, docker_log, _ = fixture
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(repo.parent / "home"),
        "DOCKER_LOG": str(docker_log),
        "BSPP_REGISTRY": BSPP_REGISTRY,
        "BSPP_IMAGE_REPOSITORY": BSPP_IMAGE_REPOSITORY,
    }
    argv = [str(repo / "containers" / "scripts" / "push.sh"), "postprocessing", *arguments]
    return subprocess.run(argv, env=env, capture_output=True, text=True, check=False)


def test_single_postprocessing_with_internal_builds_both(tmp_path: Path) -> None:
    """`push.sh postprocessing --include-postprocessing-internal` pushes the
    public image and then invokes build-internal.sh (build + tag only, never
    pushed)."""
    fixture = _postprocessing_single_fixture(tmp_path, with_internal=True)
    _, _, docker_log, internal_marker = fixture

    result = _run_single_postprocessing(fixture, "--include-postprocessing-internal")

    assert result.returncode == 0, result.stderr
    # The public postprocessing image was pushed.
    calls = [json.loads(line) for line in docker_log.read_text().splitlines()]
    push_calls = [c for c in calls if c and c[0] == "push"]
    assert push_calls == [["push", f"{REGISTRY_REPOSITORY}:postprocessing"]]
    # The internal build was invoked.
    assert internal_marker.exists()
    assert internal_marker.read_text() == "invoked"
    assert "built (not pushed by design)" in result.stdout


def test_single_postprocessing_without_internal_flag_skips_internal_build(tmp_path: Path) -> None:
    """Without the flag the internal build is never invoked."""
    fixture = _postprocessing_single_fixture(tmp_path, with_internal=True)
    _, _, _, internal_marker = fixture

    result = _run_single_postprocessing(fixture)

    assert result.returncode == 0, result.stderr
    assert not internal_marker.exists()
    assert "built (not pushed by design)" not in result.stdout


def test_single_postprocessing_internal_fails_when_assets_absent(tmp_path: Path) -> None:
    """`--include-postprocessing-internal` without containers/nvidia/ fails fast."""
    fixture = _postprocessing_single_fixture(tmp_path, with_internal=False)

    result = _run_single_postprocessing(fixture, "--include-postprocessing-internal")

    assert result.returncode != 0
    assert "containers/nvidia/" in result.stderr
    assert "absent from this tree" in result.stderr


def test_single_preprocessing_with_internal_flag_fails_fast(tmp_path: Path) -> None:
    """`push.sh preprocessing --include-postprocessing-internal` must fail fast."""
    repo = tmp_path / "repo"
    scripts = repo / "containers" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "containers" / "scripts" / "push.sh", scripts / "push.sh")
    docker_config = tmp_path / "home" / ".docker" / "config.json"
    docker_config.parent.mkdir(parents=True)
    docker_config.write_text(json.dumps({"auths": {BSPP_REGISTRY: {}}}))

    result = subprocess.run(
        [str(scripts / "push.sh"), "preprocessing", "--include-postprocessing-internal"],
        env={
            **os.environ,
            "HOME": str(tmp_path / "home"),
            "BSPP_REGISTRY": BSPP_REGISTRY,
            "BSPP_IMAGE_REPOSITORY": BSPP_IMAGE_REPOSITORY,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--include-postprocessing-internal" in result.stderr
    assert "not 'preprocessing'" in result.stderr


def test_single_folding_with_internal_flag_fails_fast(tmp_path: Path) -> None:
    """`push.sh folding <image> --include-postprocessing-internal` must fail fast."""
    repo = tmp_path / "repo"
    scripts = repo / "containers" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "containers" / "scripts" / "push.sh", scripts / "push.sh")
    docker_config = tmp_path / "home" / ".docker" / "config.json"
    docker_config.parent.mkdir(parents=True)
    docker_config.write_text(json.dumps({"auths": {BSPP_REGISTRY: {}}}))

    result = subprocess.run(
        [str(scripts / "push.sh"), "folding", "runtime", "--include-postprocessing-internal"],
        env={
            **os.environ,
            "HOME": str(tmp_path / "home"),
            "BSPP_REGISTRY": BSPP_REGISTRY,
            "BSPP_IMAGE_REPOSITORY": BSPP_IMAGE_REPOSITORY,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--include-postprocessing-internal" in result.stderr
    assert "not 'folding'" in result.stderr


def test_single_postprocessing_pass_through_args(tmp_path: Path) -> None:
    """Genuine docker-build args still pass through to the build backend in
    single-variant mode (pass-through contract preserved)."""
    fixture = _postprocessing_single_fixture(tmp_path, with_internal=False)
    repo, bin_dir, docker_log, _ = fixture
    builder_log = tmp_path / "builder.jsonl"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(repo.parent / "home"),
        "DOCKER_LOG": str(docker_log),
        "BUILDER_LOG": str(builder_log),
        "BSPP_REGISTRY": BSPP_REGISTRY,
        "BSPP_IMAGE_REPOSITORY": BSPP_IMAGE_REPOSITORY,
    }
    argv = [
        str(repo / "containers" / "scripts" / "push.sh"),
        "postprocessing",
        "--no-cache",
        "--progress=plain",
    ]
    result = subprocess.run(argv, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    builder_calls = json.loads(builder_log.read_text())
    # The builder receives the local image name first, then the pass-through args.
    assert builder_calls[0] == "bspp-orchestration:postprocessing"
    assert "--no-cache" in builder_calls
    assert "--progress=plain" in builder_calls
