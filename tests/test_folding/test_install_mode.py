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

"""Tests for the shared install-mode helper and entrypoint sourcing order.

These tests source ``containers/scripts/install-orchestration-source.sh`` in a
subprocess with a controlled environment (fake python, no git, minimal PATH)
to verify the mode-selection, provenance, fail-closed, dev-mount, and opt-out
logic without real pip side effects.

The helper is baked-first: the baked wheels are the default; a bind-mount is a
dev override that must be explicitly requested with
``BSPP_ORCHESTRATION_DEV_MOUNT=1``. ``BSPP_INSTALL_MODE_SKIP=1`` remains the
benchmark-determinism opt-out that forces baked mode.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

HELPER = Path("containers/scripts/install-orchestration-source.sh")
_COMMIT_SHA = "a" * 40
_BAKED_COMMIT_SHA = "b" * 40


def _make_fake_python(tmp_path: Path, image_json_path: Path | None = None) -> Path:
    """Create a fake python shim that handles -m pip (no-op) and -c (JSON read)."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    fake = bin_dir / "python"
    script_lines = [
        "#!/usr/bin/env bash",
        'echo "FAKE_PYTHON_CALLED: $@" >> "$BSPP_FAKE_PYTHON_LOG"',
    ]
    if image_json_path is not None:
        script_lines.append(
            textwrap.dedent(
                f"""\
                if [[ "$1" == "-c" ]]; then
                    if [[ "$*" == *"source_commit"* ]]; then
                        python3 -c "
import json, sys
try:
    v = json.load(open('{image_json_path}')).get('source_commit', '')
    sys.stdout.write(v if isinstance(v, str) else '')
except Exception:
    pass
" 2>/dev/null
                    fi
                fi
                """
            )
        )
    script_lines.append("exit 0")
    fake.write_text("\n".join(script_lines))
    fake.chmod(0o755)
    return bin_dir


def _source_helper(
    tmp_path: Path,
    orch_root: Path | str | None,
    *,
    image_json: Path | None = None,
    fake_bin: Path | None = None,
    skip: bool = False,
    dev_mount: bool = False,
    allow_mismatch: bool = False,
) -> tuple[int, dict[str, str], str]:
    """Source the helper in a subprocess and return (returncode, env, log)."""
    log_file = tmp_path / "python-calls.log"
    env: dict[str, str] = {
        "BSPP_FAKE_PYTHON_LOG": str(log_file),
        "HOME": str(tmp_path),
    }
    if orch_root is not None:
        env["BSPP_INSTALL_ORCH_ROOT"] = str(orch_root)
    if skip:
        env["BSPP_INSTALL_MODE_SKIP"] = "1"
    if dev_mount:
        env["BSPP_ORCHESTRATION_DEV_MOUNT"] = "1"
    if allow_mismatch:
        env["BSPP_ORCHESTRATION_DEV_MOUNT_ALLOW_MISMATCH"] = "1"
    if image_json is not None:
        env["BSPP_IMAGE_JSON"] = str(image_json)
    # Build a minimal PATH that has cat/grep/cut but NOT git.
    path_parts: list[str] = []
    if fake_bin is not None:
        path_parts.append(str(fake_bin))
    path_parts.append("/bin")
    env["PATH"] = ":".join(path_parts)

    # Source the helper, then print the relevant env vars.
    cmd = f"source {HELPER} 2>&1; echo '---ENV---'; env | grep -E '^BSPP_ORCHESTRATION_' || true"
    proc = subprocess.run(
        ["bash", "-c", cmd],
        env=env,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout
    # Parse env vars from the output
    env_vars: dict[str, str] = {}
    in_env = False
    for line in stdout.splitlines():
        if line == "---ENV---":
            in_env = True
            continue
        if in_env and "=" in line:
            key, _, val = line.partition("=")
            env_vars[key] = val
    log_text = log_file.read_text() if log_file.exists() else ""
    return proc.returncode, env_vars, log_text


def _make_valid_orch_root(tmp_path: Path) -> Path:
    """Create a fake orchestration source with all three pyproject.toml files."""
    orch_root = tmp_path / "orchestration"
    orch_root.mkdir()
    for member in ("orchestration-contract", "orchestration-control", "orchestration-runtime"):
        member_dir = orch_root / "packages" / member
        member_dir.mkdir(parents=True)
        (member_dir / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    return orch_root


def _make_image_json(tmp_path: Path, commit: str = "abc123") -> Path:
    import json

    p = tmp_path / "image.json"
    p.write_text(json.dumps({"source_commit": commit}))
    return p


# --- Baked-first default ---


def test_baked_default_no_mount(tmp_path: Path) -> None:
    """No mount and no dev flag: baked mode."""
    fake_bin = _make_fake_python(tmp_path)
    rc, env, _log = _source_helper(tmp_path, None, fake_bin=fake_bin)
    assert rc == 0
    assert env.get("BSPP_ORCHESTRATION_SOURCE") == "baked"
    assert "BSPP_ORCHESTRATION_PROVENANCE_COMMIT" not in env


def test_baked_default_with_image_json(tmp_path: Path) -> None:
    """Baked mode reads source_commit from image JSON."""
    image_json = _make_image_json(tmp_path, "abc123")
    fake_bin = _make_fake_python(tmp_path, image_json)
    rc, env, _log = _source_helper(tmp_path, None, image_json=image_json, fake_bin=fake_bin)
    assert rc == 0
    assert env.get("BSPP_ORCHESTRATION_SOURCE") == "baked"
    assert env.get("BSPP_ORCHESTRATION_PROVENANCE_COMMIT") == "abc123"


def test_baked_default_present_mount_without_dev_flag_fails_closed(tmp_path: Path) -> None:
    """A present mount without the dev flag fails closed (no silent override)."""
    orch_root = _make_valid_orch_root(tmp_path)
    fake_bin = _make_fake_python(tmp_path)
    rc, _env, _log = _source_helper(tmp_path, orch_root, fake_bin=fake_bin)
    assert rc == 1


def test_baked_default_non_directory_mount_fails_closed(tmp_path: Path) -> None:
    """A regular file at the mount path without the dev flag fails closed."""
    orch_root = tmp_path / "orchestration"
    orch_root.write_text("not a directory")
    fake_bin = _make_fake_python(tmp_path)
    rc, _env, _log = _source_helper(tmp_path, orch_root, fake_bin=fake_bin)
    assert rc == 1


# --- Dev-mount override ---


def test_dev_mount_git_free_detached_sha(tmp_path: Path) -> None:
    """Dev-mount with detached HEAD SHA in .git/HEAD, no git binary."""
    orch_root = _make_valid_orch_root(tmp_path)
    git_dir = orch_root / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(_COMMIT_SHA + "\n")
    image_json = _make_image_json(tmp_path, _COMMIT_SHA)
    fake_bin = _make_fake_python(tmp_path, image_json)
    rc, env, log = _source_helper(tmp_path, orch_root, image_json=image_json, fake_bin=fake_bin, dev_mount=True)
    assert rc == 0
    assert env.get("BSPP_ORCHESTRATION_SOURCE") == "override"
    assert env.get("BSPP_ORCHESTRATION_PROVENANCE_COMMIT") == _COMMIT_SHA
    assert "--no-build-isolation" in log


def test_dev_mount_git_free_branch_ref(tmp_path: Path) -> None:
    """Dev-mount with branch ref in .git/HEAD, no git binary."""
    orch_root = _make_valid_orch_root(tmp_path)
    git_dir = orch_root / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    refs_dir = git_dir / "refs" / "heads"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text(_COMMIT_SHA + "\n")
    image_json = _make_image_json(tmp_path, _COMMIT_SHA)
    fake_bin = _make_fake_python(tmp_path, image_json)
    rc, env, _log = _source_helper(tmp_path, orch_root, image_json=image_json, fake_bin=fake_bin, dev_mount=True)
    assert rc == 0
    assert env.get("BSPP_ORCHESTRATION_SOURCE") == "override"
    assert env.get("BSPP_ORCHESTRATION_PROVENANCE_COMMIT") == _COMMIT_SHA


def test_dev_mount_git_free_packed_refs(tmp_path: Path) -> None:
    """Dev-mount with packed-refs fallback, no git binary."""
    orch_root = _make_valid_orch_root(tmp_path)
    git_dir = orch_root / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "packed-refs").write_text(f"{_COMMIT_SHA} refs/heads/main\n")
    image_json = _make_image_json(tmp_path, _COMMIT_SHA)
    fake_bin = _make_fake_python(tmp_path, image_json)
    rc, env, _log = _source_helper(tmp_path, orch_root, image_json=image_json, fake_bin=fake_bin, dev_mount=True)
    assert rc == 0
    assert env.get("BSPP_ORCHESTRATION_SOURCE") == "override"
    assert env.get("BSPP_ORCHESTRATION_PROVENANCE_COMMIT") == _COMMIT_SHA


def test_dev_mount_no_git_no_dotgit_fails_closed(tmp_path: Path) -> None:
    """Dev-mount with no .git and no baked commit fails closed: the agreement
    check requires a verifiable baked commit before substituting a mount."""
    orch_root = _make_valid_orch_root(tmp_path)
    fake_bin = _make_fake_python(tmp_path)
    rc, _env, _log = _source_helper(tmp_path, orch_root, fake_bin=fake_bin, dev_mount=True)
    assert rc == 1


def test_dev_mount_no_git_no_dotgit_allowed_by_override(tmp_path: Path) -> None:
    """The dev-only mismatch override permits a mount with no verifiable commit."""
    orch_root = _make_valid_orch_root(tmp_path)
    fake_bin = _make_fake_python(tmp_path)
    rc, env, _log = _source_helper(tmp_path, orch_root, fake_bin=fake_bin, dev_mount=True, allow_mismatch=True)
    assert rc == 0
    assert env.get("BSPP_ORCHESTRATION_SOURCE") == "override"
    assert "BSPP_ORCHESTRATION_PROVENANCE_COMMIT" not in env


def test_dev_mount_undeterminable_commit_fails_closed_when_baked_known(tmp_path: Path) -> None:
    """A dev-mount whose commit cannot be determined fails closed when the
    baked commit is known (no silent substitution of unknown source)."""
    orch_root = _make_valid_orch_root(tmp_path)  # no .git
    image_json = _make_image_json(tmp_path, _BAKED_COMMIT_SHA)
    fake_bin = _make_fake_python(tmp_path, image_json)
    rc, _env, _log = _source_helper(tmp_path, orch_root, image_json=image_json, fake_bin=fake_bin, dev_mount=True)
    assert rc == 1


def test_dev_mount_undeterminable_commit_allowed_by_override(tmp_path: Path) -> None:
    """The dev-only mismatch override permits an undeterminable mount commit."""
    orch_root = _make_valid_orch_root(tmp_path)
    image_json = _make_image_json(tmp_path, _BAKED_COMMIT_SHA)
    fake_bin = _make_fake_python(tmp_path, image_json)
    rc, env, _log = _source_helper(
        tmp_path, orch_root, image_json=image_json, fake_bin=fake_bin, dev_mount=True, allow_mismatch=True
    )
    assert rc == 0
    assert env.get("BSPP_ORCHESTRATION_SOURCE") == "override"


def test_helper_direct_execution_exits_cleanly(tmp_path: Path) -> None:
    """Direct execution (not sourced) exits 0 rather than falling through into
    the fail-closed branches."""
    fake_bin = _make_fake_python(tmp_path)
    env = {
        "BSPP_FAKE_PYTHON_LOG": str(tmp_path / "direct.log"),
        "HOME": str(tmp_path),
        "PATH": f"{fake_bin}:/bin",
    }
    proc = subprocess.run(["bash", str(HELPER)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0


def test_dev_mount_without_mount_fails_closed(tmp_path: Path) -> None:
    """Dev-mount requested but no mount present: fail closed."""
    fake_bin = _make_fake_python(tmp_path)
    rc, _env, _log = _source_helper(tmp_path, None, fake_bin=fake_bin, dev_mount=True)
    assert rc == 1


def test_dev_mount_missing_pyproject_fails_closed(tmp_path: Path) -> None:
    """Dev-mount present but missing a member pyproject.toml: fail closed."""
    orch_root = tmp_path / "orchestration"
    orch_root.mkdir()
    (orch_root / "packages").mkdir()
    fake_bin = _make_fake_python(tmp_path)
    rc, _env, _log = _source_helper(tmp_path, orch_root, fake_bin=fake_bin, dev_mount=True)
    assert rc == 1


def test_dev_mount_commit_mismatch_fails_closed(tmp_path: Path) -> None:
    """Dev-mount whose commit disagrees with the baked commit fails closed."""
    orch_root = _make_valid_orch_root(tmp_path)
    git_dir = orch_root / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(_COMMIT_SHA + "\n")
    image_json = _make_image_json(tmp_path, _BAKED_COMMIT_SHA)
    fake_bin = _make_fake_python(tmp_path, image_json)
    rc, _env, _log = _source_helper(tmp_path, orch_root, image_json=image_json, fake_bin=fake_bin, dev_mount=True)
    assert rc == 1


def test_dev_mount_commit_mismatch_allowed_by_override(tmp_path: Path) -> None:
    """The dev-only mismatch override permits a disagreeing mount commit."""
    orch_root = _make_valid_orch_root(tmp_path)
    git_dir = orch_root / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(_COMMIT_SHA + "\n")
    image_json = _make_image_json(tmp_path, _BAKED_COMMIT_SHA)
    fake_bin = _make_fake_python(tmp_path, image_json)
    rc, env, _log = _source_helper(
        tmp_path, orch_root, image_json=image_json, fake_bin=fake_bin, dev_mount=True, allow_mismatch=True
    )
    assert rc == 0
    assert env.get("BSPP_ORCHESTRATION_SOURCE") == "override"
    assert env.get("BSPP_ORCHESTRATION_PROVENANCE_COMMIT") == _COMMIT_SHA


def test_dev_mount_matching_commit_is_override(tmp_path: Path) -> None:
    """A dev-mount commit equal to the baked commit is a clean override."""
    orch_root = _make_valid_orch_root(tmp_path)
    git_dir = orch_root / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(_COMMIT_SHA + "\n")
    image_json = _make_image_json(tmp_path, _COMMIT_SHA)
    fake_bin = _make_fake_python(tmp_path, image_json)
    rc, env, _log = _source_helper(tmp_path, orch_root, image_json=image_json, fake_bin=fake_bin, dev_mount=True)
    assert rc == 0
    assert env.get("BSPP_ORCHESTRATION_SOURCE") == "override"
    assert env.get("BSPP_ORCHESTRATION_PROVENANCE_COMMIT") == _COMMIT_SHA


# --- Opt-out (benchmark determinism) ---


def test_opt_out_skip_with_valid_mount(tmp_path: Path) -> None:
    """BSPP_INSTALL_MODE_SKIP=1 with a valid mount: baked mode, no pip install."""
    orch_root = _make_valid_orch_root(tmp_path)
    image_json = _make_image_json(tmp_path, "baked_commit")
    fake_bin = _make_fake_python(tmp_path, image_json)
    rc, env, log = _source_helper(tmp_path, orch_root, image_json=image_json, fake_bin=fake_bin, skip=True)
    assert rc == 0
    assert env.get("BSPP_ORCHESTRATION_SOURCE") == "baked"
    assert env.get("BSPP_ORCHESTRATION_PROVENANCE_COMMIT") == "baked_commit"
    # No pip install -e calls should appear in the log
    assert "pip install" not in log


def test_opt_out_skip_no_image_json(tmp_path: Path) -> None:
    """BSPP_INSTALL_MODE_SKIP=1 with no image JSON: baked, provenance unset."""
    orch_root = _make_valid_orch_root(tmp_path)
    fake_bin = _make_fake_python(tmp_path)
    rc, env, _log = _source_helper(tmp_path, orch_root, fake_bin=fake_bin, skip=True)
    assert rc == 0
    assert env.get("BSPP_ORCHESTRATION_SOURCE") == "baked"
    assert "BSPP_ORCHESTRATION_PROVENANCE_COMMIT" not in env


# --- Entrypoint sourcing order ---


def test_colabfold_sources_helper_before_first_exec() -> None:
    """The colabfold entrypoint must source the helper before the first exec."""
    content = Path("containers/folding/colabfold/entrypoint.sh").read_text()
    lines = content.splitlines()
    source_line = None
    exec_line = None
    for i, line in enumerate(lines):
        if "install-orchestration-source.sh" in line and "source" in line:
            source_line = i
        if 'exec "$@"' in line and exec_line is None:
            exec_line = i
    assert source_line is not None, "helper not sourced"
    assert exec_line is not None, "no exec found"
    assert source_line < exec_line, "helper must be sourced before first exec"


def test_all_entrypoints_source_helper() -> None:
    """All 4 folding entrypoints and the postprocessing entrypoint source the helper."""
    for image in ("runtime", "colabfold", "openfold-cli", "bioir"):
        content = Path(f"containers/folding/{image}/entrypoint.sh").read_text()
        assert "install-orchestration-source.sh" in content, f"{image} entrypoint missing helper source"
    postprocessing = Path("containers/scripts/entrypoint.sh").read_text()
    assert "install-orchestration-source.sh" in postprocessing, "postprocessing entrypoint missing helper source"


def test_benchmark_optout_in_srun_env_not_heredoc(tmp_path: Path) -> None:
    """The opt-out guard must be in the srun command, not the heredoc."""
    from bspp.orchestration.contract.folding_release import FoldingReleasePreset
    from bspp.orchestration.contract.phase import PhaseSlurmResources
    from bspp.orchestration.control.folding_benchmark_submit import (
        render_benchmark_validate_submission,
    )
    from bspp.orchestration.control.profiles import (
        PostprocessingCredentialMountProfile,
        ResolvedClusterProfile,
    )

    profile = ResolvedClusterProfile(
        name="test",
        owner="bspp",
        project_root="/lustre/bspp",
        output_root="/lustre/bspp/output",
        staging_root="/lustre/bspp/staging",
        afdb_toolkit_repo=None,
        orchestration_repo="/lustre/bspp/orchestration",
        image="registry/bspp-runtime:latest",
        transport="local-slurm",
        ssh_target=None,
        probe_root=None,
        source_bundle_root=None,
        runtime_image_cache_root=None,
        runtime_image_policy="digest-checked",
        runtime_qualification_root=None,
        runtime_qualification_control_root=None,
        governed_package_root=None,
        runtime_qualification_expires_hours=168,
        preprocessing_runtime=None,
        postprocessing_credential_mounts=PostprocessingCredentialMountProfile(
            aws_shared_credentials_file="/home/user/.aws/credentials",
            aws_config_file="/home/user/.aws/config",
        ),
        database_sets=(),
        database_access_policies=(),
        database_staging=None,
        database_acceptance_cache=None,
        extra_mounts=(),
        account="bspp",
        resources={"control_cpu": PhaseSlurmResources(partition="cpu", cpus_per_task=4, memory="16G", time="01:00:00")},
        folding_release_preset=FoldingReleasePreset.PUBLIC,
        folding_backend_images={},
        folding_backend_assets={},
        mount_orchestration_source=False,
    )
    script_path = tmp_path / "validate-run.sbatch"
    render_benchmark_validate_submission(
        cluster_profile=profile,
        run_dir=Path("/lustre/bspp/run"),
        suite=Path("/lustre/bspp/suite"),
        index=Path("/lustre/bspp/index"),
        corpus="test-corpus",
        fingerprint="test-fingerprint",
        output_dir=Path("/lustre/bspp/output"),
        script_path=script_path,
    )
    script = script_path.read_text()
    assert "BSPP_INSTALL_MODE_SKIP" in script
    heredoc_start = script.index("<<'BSPP_BENCHMARK_VALIDATE'")
    srun_section = script[:heredoc_start]
    assert "BSPP_INSTALL_MODE_SKIP" in srun_section, "opt-out must be in the srun command, not the heredoc"
    heredoc_end = script.index("BSPP_BENCHMARK_VALIDATE", heredoc_start + 1)
    heredoc_body = script[heredoc_start:heredoc_end]
    assert "BSPP_INSTALL_MODE_SKIP" not in heredoc_body, "opt-out must not be in the heredoc body"
