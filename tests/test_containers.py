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

"""Container variant metadata checks."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_PYPROJECT = ROOT.parent / "bspp" / "AFDB-Integration-Kit" / "pyproject.toml"
MEMBER_PYPROJECTS = (
    ROOT / "packages" / "orchestration-contract" / "pyproject.toml",
    ROOT / "packages" / "orchestration-control" / "pyproject.toml",
    ROOT / "packages" / "orchestration-runtime" / "pyproject.toml",
)

PLATFORM_TO_PIXI = {
    "linux/amd64": "linux-64",
    "linux/arm64": "linux-aarch64",
}


def test_container_python_variants_satisfy_mounted_source_requirements() -> None:
    toolkit_pyproject = _require_toolkit_pyproject()
    required = max(_minimum_python_version(path) for path in (toolkit_pyproject, *MEMBER_PYPROJECTS))

    for env_path in sorted((ROOT / "containers" / "variants").glob("*.env")):
        env = _read_env(env_path)
        manifest = _read_manifest(env)
        assert _version_tuple(manifest["dependencies"]["python"]) >= required, env_path.name


def test_container_variants_have_matching_pixi_manifests_and_platforms() -> None:
    for env_path in sorted((ROOT / "containers" / "variants").glob("*.env")):
        env = _read_env(env_path)
        manifest_path = ROOT / env["PIXI_MANIFEST"]
        assert manifest_path.exists(), manifest_path

        manifest = _read_manifest(env)
        pixi_platforms = set(manifest["workspace"]["platforms"])
        for platform in env["PLATFORMS"].split(","):
            assert PLATFORM_TO_PIXI[platform] in pixi_platforms


def test_torch_cluster_is_prebuilt_and_abi_matched() -> None:
    for env_path in sorted((ROOT / "containers" / "variants").glob("*.env")):
        env = _read_env(env_path)
        manifest = _read_manifest(env)
        torch_cluster = manifest["pypi-dependencies"]["torch-cluster"]
        torch = manifest["pypi-dependencies"]["torch"]
        find_links = {entry["url"] for entry in manifest["pypi-options"]["find-links"]}

        assert "git" not in torch_cluster.lower(), env_path.name
        assert torch_cluster == env["TORCH_CLUSTER_VERSION"]
        assert env["PYG_WHEEL_INDEX"] in find_links
        assert env["TORCH_CUDA_FLAVOR"] in torch
        assert f"+pt{_torch_abi(env['TORCH_VERSION'])}" in torch_cluster


def test_container_manifests_include_nvcomp_for_cuda_variant() -> None:
    post = _read_manifest(_read_env(ROOT / "containers" / "variants" / "postprocessing.env"))
    assert post["pypi-dependencies"]["nvidia-nvcomp-cu13"] == "==5.2.0.13"


def test_postprocessing_exposes_cuda_compatibility_metadata() -> None:
    post = _read_env(ROOT / "containers" / "variants" / "postprocessing.env")
    assert post["CUDA_COMPAT_PACKAGE"] == "cuda-compat-13-0"
    assert post["CUDA_COMPAT_DIR"] == "/usr/local/cuda-13.0/compat"

    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    assert "CUDA_COMPAT_PACKAGE" in dockerfile
    assert "BSPP_CUDA_COMPAT_DIR" in dockerfile
    assert "ENV BSPP_CUDA_COMPAT=off" in dockerfile


def test_postprocessing_respects_torch_setuptools_constraint() -> None:
    post = _read_manifest(_read_env(ROOT / "containers" / "variants" / "postprocessing.env"))
    assert post["pypi-dependencies"]["torch"] == "==2.11.0+cu130"
    assert post["dependencies"]["setuptools"] == "<82"


def test_postprocessing_includes_toolkit_jsonschema_format_dependencies() -> None:
    """The toolkit imports format validators that plain jsonschema omits."""
    post = _read_manifest(_read_env(ROOT / "containers" / "variants" / "postprocessing.env"))
    assert post["pypi-dependencies"]["jsonschema"] == {
        "version": ">=4.24.0",
        "extras": ["format-nongpl"],
    }


def test_variants_declare_target_and_validated_sms() -> None:
    post = _read_env(ROOT / "containers" / "variants" / "postprocessing.env")
    assert set(post["TARGET_SMS"].split(",")) >= {"sm_80", "sm_90", "sm_100", "sm_120"}
    assert "VALIDATED_SMS" in post


def test_container_build_path_uses_pixi_not_uv_locks() -> None:
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    assert "pixi install" in dockerfile
    assert "pixi install --locked || pixi install" not in dockerfile
    assert "uv pip install" not in dockerfile
    assert "containers/locks" not in dockerfile


def test_dockerfile_pins_and_verifies_pixi_binary() -> None:
    """The pixi binary is pinned and sha256-verified, never curl|bash."""
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    assert "https://pixi.sh/install.sh" not in dockerfile
    assert "PIXI_VERSION" in dockerfile
    assert "PIXI_SHA256" in dockerfile
    assert "pixi-x86_64-unknown-linux-musl" in dockerfile
    assert "sha256sum --check --status" in dockerfile


def test_container_system_packages_include_compression_tools() -> None:
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    system_packages = dockerfile.split("apt-get install -y --no-install-recommends", 1)[1].split("&&", 1)[0]

    assert re.search(r"\blz4\b", system_packages)
    assert re.search(r"\bzstd\b", system_packages)


def test_container_installs_s5cmd() -> None:
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()

    assert "S5CMD_VERSION" in dockerfile
    assert "S5CMD_SHA256" in dockerfile
    assert "s5cmd.deb" in dockerfile
    assert "sha256sum --check --status" in dockerfile
    assert "dpkg -i /tmp/s5cmd.deb" in dockerfile


def test_gpu_smoke_records_sm_support_and_json_artifact() -> None:
    smoke = (ROOT / "containers" / "scripts" / "smoke-gpu.sh").read_text()
    assert "torch.cuda.get_arch_list()" in smoke
    assert "device_sm_supported_by_torch" in smoke
    assert "device_ptx_supported_by_torch" in smoke
    assert "radius_graph_cuda_ok" in smoke
    assert "nvcomp_zstd_raw_encode_ok" in smoke
    assert "nvcomp_zstd_cli_decode_ok" in smoke
    assert "json.dumps(record" in smoke

    acceptance = (ROOT / "containers" / "SM_ACCEPTANCE.md").read_text()
    assert "device_sm_supported_by_torch True" in acceptance
    assert "Image Digest / sqsh Hash" in acceptance
    assert "Toolkit Commit" in acceptance
    assert "sm_100" in acceptance
    assert "sm_120" in acceptance
    assert "Stage 13 Timing" in acceptance


def test_sm_acceptance_documents_nvcomp_signals() -> None:
    acceptance = (ROOT / "containers" / "SM_ACCEPTANCE.md").read_text()
    readme = (ROOT / "containers" / "README.md").read_text()

    assert "nvcomp_zstd_raw_encode_ok" in acceptance
    assert "nvcomp_zstd_cli_decode_ok" in acceptance
    assert "tar_compression: zstd-members" in readme


def test_slurm_smoke_wrapper_runs_container_smoke_with_json_output() -> None:
    wrapper = (ROOT / "containers" / "scripts" / "slurm-smoke-gpu.sh").read_text()
    assert "#SBATCH --gres=gpu:1" in wrapper
    assert "resolve-repo-root.sh" in wrapper
    assert "${ORCH_DIR:-}/containers/scripts/resolve-repo-root.sh" in wrapper
    assert "BASH_SOURCE[0]" not in wrapper
    assert "/usr/local/bin/entrypoint.sh bspp-container-smoke-gpu" in wrapper
    assert "bspp-container-smoke-gpu" in wrapper
    assert "SMOKE_JSON" in wrapper


def test_legacy_bridge_shell_entrypoints_are_absent_and_acceptance_wrappers_are_present() -> None:
    script_dir = ROOT / "containers" / "scripts"

    assert (script_dir / "submit-acceptance-checks.sh").is_file()
    assert (script_dir / "slurm-tar-payload-parity.sh").is_file()
    assert (script_dir / "slurm-semantic-acceptance.sh").is_file()
    for script_name in (
        "submit-native-local-tar-run.sh",
        "render-acceptance-runbook.sh",
        "verify-acceptance-evidence.sh",
    ):
        assert not (script_dir / script_name).exists()


@pytest.mark.parametrize(
    "script_path",
    [
        "containers/scripts/build.sh",
        "containers/scripts/push.sh",
        "containers/scripts/pull-sqsh.sh",
        "containers/scripts/submit-acceptance-checks.sh",
        "containers/scripts/slurm-tar-payload-parity.sh",
        "containers/scripts/slurm-semantic-acceptance.sh",
        "containers/scripts/slurm-preprocessing-carry-characterization.sh",
        "containers/scripts/preprocessing-next-command-lib.sh",
        "containers/scripts/survey-runtime-postprocessing-ceiling.sh",
        "containers/preprocessing/build.sh",
        "containers/preprocessing/entrypoint.sh",
        "containers/preprocessing/carry-characterization.sh",
    ],
)
def test_current_shell_entrypoints_have_bash_syntax(script_path: str) -> None:
    submitter = ROOT / script_path

    result = subprocess.run(["bash", "-n", str(submitter)], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def test_entrypoint_override_first_then_baked_fallback() -> None:
    """Override check (_check_toolkit_override) runs before baked fallback (_check_baked_toolkit)."""
    entrypoint = (ROOT / "containers" / "scripts" / "entrypoint.sh").read_text()
    assert "_check_toolkit_override" in entrypoint
    assert "_check_baked_toolkit" in entrypoint
    # Override runs first; baked only if override returns non-zero.
    assert "if ! _check_toolkit_override; then" in entrypoint
    assert "    _check_baked_toolkit" in entrypoint


def test_entrypoint_exports_toolkit_source_env() -> None:
    """Override and baked paths export BSPP_TOOLKIT_ROOT, BSPP_TOOLKIT_SOURCE, and BSPP_TOOLKIT_PROVENANCE_COMMIT."""
    entrypoint = (ROOT / "containers" / "scripts" / "entrypoint.sh").read_text()
    assert "export BSPP_TOOLKIT_ROOT=" in entrypoint
    assert "export BSPP_TOOLKIT_SOURCE=" in entrypoint
    assert "export BSPP_TOOLKIT_PROVENANCE_COMMIT=" in entrypoint
    # Override path
    assert 'BSPP_TOOLKIT_SOURCE="override"' in entrypoint
    # Baked path
    assert 'BSPP_TOOLKIT_SOURCE="baked"' in entrypoint


def test_entrypoint_no_longer_builds_ipsae_from_workspace() -> None:
    """The iPSAE make-from-workspace block must be removed."""
    entrypoint = (ROOT / "containers" / "scripts" / "entrypoint.sh").read_text()
    assert "make -s" not in entrypoint
    assert "Makefile" not in entrypoint


def test_entrypoint_fails_when_neither_valid() -> None:
    """When override is absent and baked is missing, entrypoint must exit 1 with an actionable message."""
    entrypoint = (ROOT / "containers" / "scripts" / "entrypoint.sh").read_text()
    assert "no AFDB toolkit found" in entrypoint
    assert "exit 1" in entrypoint


def test_entrypoint_baked_checks_provenance_and_ipsae() -> None:
    """Baked mode validates provenance.json, production_pipeline.py, and iPSAE executable."""
    entrypoint = (ROOT / "containers" / "scripts" / "entrypoint.sh").read_text()
    assert "provenance.json" in entrypoint
    assert "baked toolkit provenance missing" in entrypoint
    assert "baked toolkit provenance invalid" in entrypoint
    assert "baked toolkit missing required file" in entrypoint
    assert "baked toolkit iPSAE not executable" in entrypoint
    assert "ipsae_cpp" in entrypoint


def test_entrypoint_baked_validates_commit() -> None:
    """Baked mode parses provenance.json and validates the commit SHA."""
    entrypoint = (ROOT / "containers" / "scripts" / "entrypoint.sh").read_text()
    assert "expected commit" in entrypoint
    assert "e2fa757aa0cb2cec8e4a8382627fcbbca7599556" in entrypoint
    assert "actual_commit" in entrypoint


def test_entrypoint_override_mount_without_pyproject_fails_closed() -> None:
    """A mount at a recognized path without pyproject.toml must fail, not fall through to baked."""
    entrypoint = (ROOT / "containers" / "scripts" / "entrypoint.sh").read_text()
    assert "mount override present but missing pyproject.toml" in entrypoint


def test_entrypoint_override_validates_required_files() -> None:
    """Override mode validates production_pipeline.py and pyproject.toml."""
    entrypoint = (ROOT / "containers" / "scripts" / "entrypoint.sh").read_text()
    assert "mount override missing production_pipeline.py" in entrypoint
    assert "mount override missing pyproject.toml" in entrypoint


def test_entrypoint_uses_pythonpath_for_read_only_toolkit_mount() -> None:
    """Both override and baked paths set PYTHONPATH without requiring editable install."""
    entrypoint = (ROOT / "containers" / "scripts" / "entrypoint.sh").read_text()
    # PYTHONPATH is exported in both override and baked paths
    assert 'export PYTHONPATH="${toolkit_root}:' in entrypoint or 'export PYTHONPATH="${baked_root}:' in entrypoint
    assert "export PYTHONPATH=" in entrypoint
    # Editable install only if BSPP_INSTALL_TOOLKIT_EDITABLE=on
    assert "BSPP_INSTALL_TOOLKIT_EDITABLE" in entrypoint
    assert 'pip install "${INSTALL_FLAGS[@]}" -e "${toolkit_root}"' in entrypoint


def test_entrypoint_sources_orchestration_source_helper_not_inline_install() -> None:
    entrypoint = (ROOT / "containers" / "scripts" / "entrypoint.sh").read_text()

    # The baked-first orchestration-source resolution lives in the shared helper.
    assert "source /usr/local/bin/install-orchestration-source.sh" in entrypoint
    # The old inline editable-install of workspace members is gone.
    assert "-e /workspace/bspp-orchestration/packages/orchestration-contract" not in entrypoint
    assert "-e /workspace/bspp-orchestration/packages/orchestration-runtime" not in entrypoint
    assert "-e /workspace/bspp-orchestration >/dev/null" not in entrypoint


def test_container_import_helper_documents_external_partition_account() -> None:
    helper = (ROOT / "containers" / "scripts" / "pull-sqsh.sh").read_text()
    assert "#SBATCH --partition=" not in helper
    assert "#SBATCH --account=" not in helper
    assert "--partition" in helper
    assert "--account" in helper
    assert "resolve-repo-root.sh" in helper
    assert "BASH_SOURCE[0]" not in helper
    assert "#SBATCH --gres=gpu" not in helper


def test_container_manifests_cover_no_deps_editable_runtime_dependencies() -> None:
    toolkit_pyproject = _require_toolkit_pyproject()
    required = set(_project_dependency_names(ROOT / "packages" / "orchestration-runtime" / "pyproject.toml"))
    required.update(_project_dependency_names(toolkit_pyproject))
    required.update(_project_dependency_names(toolkit_pyproject, extra="production"))
    required.difference_update({"bspp-orchestration-contract", "afdb-toolkit"})

    aliases = {
        "google-cloud-storage": {"google-cloud-storage"},
        "pyyaml": {"pyyaml"},
        "torch": {"torch"},
    }

    for env_path in sorted((ROOT / "containers" / "variants").glob("*.env")):
        manifest = _read_manifest(_read_env(env_path))
        pypi_deps = set(manifest["pypi-dependencies"])
        missing = sorted(dep for dep in required if not (aliases.get(dep, {dep}) & pypi_deps))
        assert missing == [], f"{env_path.name} missing runtime deps: {missing}"


# ---------------------------------------------------------------------------
# Baked toolkit provenance — static Dockerfile tests
# ---------------------------------------------------------------------------


def test_dockerfile_clones_toolkit_into_opt_afdb_toolkit() -> None:
    """Dockerfile git-clones the explicit TOOLKIT_REPO to /opt/afdb-toolkit."""
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    assert "git clone" in dockerfile
    assert "/opt/afdb-toolkit" in dockerfile
    assert "${TOOLKIT_REPO}" in dockerfile


def test_dockerfile_clones_toolkit_once_after_pixi_install() -> None:
    """The cache-efficient toolkit layer is unique and follows dependency installation."""
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    assert dockerfile.count("git clone") == 1
    assert dockerfile.index("RUN pixi install") < dockerfile.index("git clone")


def test_dockerfile_pins_exact_toolkit_commit() -> None:
    """The exact public commit is baked as the TOOLKIT_REF ARG."""
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    assert "ARG TOOLKIT_REF=e2fa757aa0cb2cec8e4a8382627fcbbca7599556" in dockerfile


def test_dockerfile_removes_git_history_after_checkout() -> None:
    """.git directory and remote URLs are removed in the clone RUN layer."""
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    assert "rm -rf .git" in dockerfile
    assert ("git remote remove" in dockerfile) or ("remote remove" in dockerfile)
    # No git remote URL should survive in later layers
    if "rm -rf .git" in dockerfile:
        post_cleanup = dockerfile.split("rm -rf .git", 1)[1]
        for line in post_cleanup.splitlines():
            if line.strip().startswith("RUN ") and "oauth2" in line:
                raise AssertionError(f"oauth2 token URL survives after git cleanup: {line}")


def test_dockerfile_has_no_token_arg_or_env() -> None:
    """No credential appears in an ARG, ENV, or executable Dockerfile line."""
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    lines = dockerfile.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith(("ARG ", "ENV ")) and (
            "TOKEN" in stripped.upper() or "PASSWORD" in stripped.upper()
        ):
            raise AssertionError(f"Credential-bearing ARG/ENV: {stripped}")
        if ("BSPP_GITLAB_TOKEN" in stripped or "oauth2:" in stripped) and not stripped.startswith("#"):
            raise AssertionError(f"Credential appears outside a comment: {stripped}")


def test_dockerfile_clones_publicly_without_ssh_agent() -> None:
    """The canonical clone needs no SSH agent or private host key."""
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    assert "--mount=type=ssh" not in dockerfile
    assert "ssh-keyscan" not in dockerfile
    assert "ssh://" not in dockerfile


def test_dockerfile_writes_provenance_record() -> None:
    """provenance.json is written at /opt/afdb-toolkit/provenance.json."""
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    assert "/opt/afdb-toolkit/provenance.json" in dockerfile


def test_qualify_baked_toolkit_script_has_valid_bash_syntax() -> None:
    """qualify-baked-toolkit.sh passes bash -n."""
    script = ROOT / "containers" / "scripts" / "qualify-baked-toolkit.sh"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_dockerfile_invokes_qualify_baked_toolkit_script() -> None:
    """Dockerfile runs /usr/local/bin/qualify-baked-toolkit.sh."""
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    assert "qualify-baked-toolkit.sh" in dockerfile


def test_postprocessing_backend_requires_public_toolkit_repo() -> None:
    """The postprocessing backend requires TOOLKIT_REPO (public default); no SSH agent is forwarded."""
    build_script = (ROOT / "containers" / "scripts" / "build.sh").read_text()
    postprocessing_backend = (ROOT / "containers" / "postprocessing" / "build.sh").read_text()
    assert "ensure-ssh-agent.sh" not in build_script
    assert "--ssh default" not in build_script
    assert '--build-arg "TOOLKIT_REPO=' in postprocessing_backend
    assert "containers/preprocessing/build.sh" in build_script
    assert "containers/postprocessing/build.sh" in build_script


def test_preprocessing_build_resolves_uv_workspace_from_repo_root(tmp_path: Path) -> None:
    """The dedicated backend finds its uv workspace independently of caller cwd."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_cwd_marker = tmp_path / "uv-cwd"
    real_git = shutil.which("git")
    assert real_git is not None

    fake_commands = {
        "git": """#!/usr/bin/env bash
if [[ " $* " == *" status "* ]]; then
  exit 0
fi
exec "$BSPP_REAL_GIT" "$@"
""",
        "uv": """#!/usr/bin/env bash
if [[ "${1:-}" == "--directory" ]]; then
  cd "$2"
fi
printf '%s\t%s\n' "$PWD" "$*" >> "$BSPP_UV_CWD_MARKER"
if [[ " $* " == *" --package bspp-orchestration-runtime "* ]]; then
  exit 41
fi
exit 0
""",
        "pixi": "#!/usr/bin/env bash\nexit 0\n",
        "docker": "#!/usr/bin/env bash\nexit 0\n",
        "rm": "#!/usr/bin/env bash\nexit 0\n",
        "mkdir": "#!/usr/bin/env bash\nexit 0\n",
    }
    for name, body in fake_commands.items():
        command = bin_dir / name
        command.write_text(body)
        command.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "containers" / "preprocessing" / "build.sh"),
            "bspp-preprocessing-runtime:test",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "BSPP_REAL_GIT": real_git,
            "BSPP_UV_CWD_MARKER": str(uv_cwd_marker),
            "CONTAINER_ENGINE": "docker",
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 41, result.stderr
    uv_calls = [line.split("\t", maxsplit=1) for line in uv_cwd_marker.read_text().splitlines()]
    assert [cwd for cwd, _argv in uv_calls] == [str(ROOT), str(ROOT)]
    assert "--package bspp-orchestration-contract" in uv_calls[0][1]
    assert "--package bspp-orchestration-runtime" in uv_calls[1][1]


def test_dockerfile_builds_ipsae_from_baked_source() -> None:
    """Dockerfile compiles iPSAE from /opt/afdb-toolkit/afdb_integration_kit/ipsae/."""
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    assert "/opt/afdb-toolkit/afdb_integration_kit/ipsae" in dockerfile
    assert "make -B -C" in dockerfile
    assert "ipsae_cpp" in dockerfile


def test_push_script_requires_public_toolkit_repo() -> None:
    """push.sh requires TOOLKIT_REPO (via the dedicated postprocessing backend)
    and forwards no SSH agent."""
    push_script = (ROOT / "containers" / "scripts" / "push.sh").read_text()
    postprocessing_backend = (ROOT / "containers" / "postprocessing" / "build.sh").read_text()
    assert "ensure-ssh-agent.sh" not in push_script
    assert "--ssh default" not in push_script
    assert '--build-arg "TOOLKIT_REPO=' in postprocessing_backend


def test_push_all_matrix_defaults_public_toolkit_repo() -> None:
    """push.sh all defaults TOOLKIT_REPO to the public upstream (via the dedicated
    postprocessing backend) and forwards no SSH agent."""
    script = (ROOT / "containers" / "scripts" / "push.sh").read_text()
    postprocessing_backend = (ROOT / "containers" / "postprocessing" / "build.sh").read_text()
    assert "--ssh default" not in script
    public_default = "TOOLKIT_REPO=${TOOLKIT_REPO:-https://github.com/PDBeurope/AFDB-Integration-Kit.git}"
    assert public_default in postprocessing_backend
    # The full-matrix mode lives in the canonical push.sh and fails closed on
    # an unverifiable source checkout before any build or publish.
    assert "could not fetch origin main" in script
    assert "matrix_preflight" in script


def test_postprocessing_ceiling_survey_scans_past_ten_newer_pages(tmp_path: Path) -> None:
    result = _run_ceiling_survey_with_fake_index(tmp_path, control_arch="x86_64")

    assert result.returncode == 0, result.stderr
    assert "BUMP AVAILABLE" in result.stdout
    assert "TORCH_VERSION=2.12.0+cu130" in result.stdout
    assert "TORCH_CLUSTER_VERSION===1.6.4+pt212cu130" in result.stdout


def test_postprocessing_ceiling_survey_control_requires_x86_64(tmp_path: Path) -> None:
    result = _run_ceiling_survey_with_fake_index(tmp_path, control_arch="aarch64")

    assert result.returncode == 1
    assert "pinned wheel not found on its own index page" in result.stdout


def test_postprocessing_ceiling_survey_rejects_empty_decisive_page(tmp_path: Path) -> None:
    result = _run_ceiling_survey_with_fake_index(
        tmp_path,
        control_arch="x86_64",
        empty_newest_page=True,
    )

    assert result.returncode == 1
    assert "unreachable or empty — survey INVALID" in result.stdout


def test_env_example_documents_public_default_and_preprocessing() -> None:
    """The environment guide documents the public toolkit default and the tag set."""
    env_example = (ROOT / "containers" / ".env.example").read_text()
    assert "postprocessing" in env_example
    assert "preprocessing" in env_example
    assert "public upstream" in env_example
    assert "BSPP_GITLAB_TOKEN" not in env_example
    # Internal source coordinates must not leak into the public env template.
    assert "gitlab-master" + ".nvidia.com" not in env_example
    assert "nvidia-" + "postproc" not in env_example


def _require_toolkit_pyproject() -> Path:
    """Return the optional sibling toolkit pyproject, skipping when it is absent.

    The two container compatibility checks below read the sibling
    ``../bspp/AFDB-Integration-Kit/pyproject.toml`` as an explicit optional test
    input. It is present in the internal development layout, while a standalone
    public checkout self-tests without fabricating or fetching that sibling.
    """
    if not TOOLKIT_PYPROJECT.is_file():
        pytest.skip(
            "optional sibling AFDB-Integration-Kit source checkout is absent "
            f"({TOOLKIT_PYPROJECT}); this compatibility check requires the internal development layout"
        )
    return TOOLKIT_PYPROJECT


def _minimum_python_version(pyproject: Path) -> tuple[int, int]:
    data = tomllib.loads(pyproject.read_text())
    requires_python = data["project"]["requires-python"]
    match = re.search(r">=\s*(\d+)\.(\d+)", requires_python)
    if match is None:
        msg = f"Unsupported requires-python constraint: {requires_python}"
        raise AssertionError(msg)
    return int(match.group(1)), int(match.group(2))


def _run_ceiling_survey_with_fake_index(
    tmp_path: Path,
    *,
    control_arch: str,
    empty_newest_page: bool = False,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pages = ["torch-2.11.0%2Bcu130.html", "torch-2.12.0%2Bcu130.html"]
    pages.extend(f"torch-2.{minor}.0%2Bcu130.html" for minor in range(13, 23))
    curl = bin_dir / "curl"
    newest_response = ":" if empty_newest_page else "printf '%s\\n' 'torch_cluster-1.6.5-cp311-cp311-linux_x86_64.whl'"
    curl.write_text(
        """#!/usr/bin/env bash
url="${!#}"
case "$url" in
  https://data.pyg.org/whl/)
    printf '%s\\n' """
        + " ".join(pages)
        + """
    ;;
  *torch-2.22.0%2Bcu130.html)
    """
        + newest_response
        + """
    ;;
  *torch-2.12.0%2Bcu130.html)
    printf '%s\\n' 'torch_cluster-1.6.4%2Bpt212cu130-cp312-cp312-linux_x86_64.whl'
    ;;
  *torch-2.11.0%2Bcu130.html)
    printf '%s\\n' 'torch_cluster-1.6.3%2Bpt211cu130-cp312-cp312-linux_"""
        + control_arch
        + """.whl'
    ;;
  https://data.pyg.org/whl/*)
    printf '%s\\n' 'torch_cluster-1.6.5-cp311-cp311-linux_x86_64.whl'
    ;;
esac
""",
    )
    curl.chmod(0o755)
    survey = ROOT / "containers" / "scripts" / "survey-runtime-postprocessing-ceiling.sh"
    return subprocess.run(
        ["bash", str(survey)],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )


def _read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        result[key] = value
    return result


def _read_manifest(env: dict[str, str]) -> dict[str, Any]:
    return tomllib.loads((ROOT / env["PIXI_MANIFEST"]).read_text())


def _project_dependency_names(pyproject: Path, *, extra: str | None = None) -> list[str]:
    data = tomllib.loads(pyproject.read_text())
    raw_deps = (
        data["project"].get("optional-dependencies", {}).get(extra, [])
        if extra
        else data["project"].get("dependencies", [])
    )
    names = []
    for dep in raw_deps:
        name = re.split(r"[\s<>=!~;\[]", dep, maxsplit=1)[0].lower().replace("_", "-")
        names.append(name)
    return names


def _version_tuple(value: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\.(\d+)", value)
    if match is None:
        msg = f"Unsupported version value: {value}"
        raise AssertionError(msg)
    return int(match.group(1)), int(match.group(2))


def _torch_abi(value: str) -> str:
    match = re.match(r"(\d+)\.(\d+)", value)
    if match is None:
        msg = f"Unsupported torch version value: {value}"
        raise AssertionError(msg)
    major, minor = match.groups()
    return f"{major}{minor}"
