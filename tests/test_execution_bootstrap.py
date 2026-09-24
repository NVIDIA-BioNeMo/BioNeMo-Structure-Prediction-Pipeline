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

"""Governed execution identity and bootstrap tests."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

import bspp.orchestration.control.execution_bootstrap as execution_bootstrap
from bspp.orchestration.contract.source_package import build_source_package
from bspp.orchestration.control.execution_bootstrap import (
    ImageIdentity,
    _check_baked_runtime_ipsae_revision,
    _install_runtime_ipsae_binary,
    execute_governed,
    identify_runtime_image,
    image_identity_from_mapping,
    main,
    qualification_smoke,
    render_governed_srun,
    run_governed_bootstrap,
    verify_runtime_image,
)


def test_governed_bootstrap_rejects_replaced_identity_record_before_parsing(tmp_path: Path) -> None:
    identity = tmp_path / "runtime-qualification.json"
    identity.write_text("{}\n")

    with pytest.raises(ValueError, match="identity record SHA-256"):
        run_governed_bootstrap(
            identity_record=identity,
            source_destination=None,
            toolkit_destination=None,
            exec_argv_json='["/usr/bin/true"]',
            expected_identity_record_sha256="0" * 64,
        )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(("git", "-C", str(repo), *args), check=True, capture_output=True)


def _toolkit_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    toolkit = repo / "toolkit"
    toolkit.mkdir()
    (toolkit / "library.py").write_text("VALUE = 1\n")
    executable = toolkit / "run"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    _git(repo, "add", "toolkit")
    _git(repo, "commit", "-qm", "fixture")
    return toolkit


def _toolkit_package(tmp_path: Path):
    toolkit = _toolkit_repo(tmp_path)
    repo = toolkit.parent
    commit = subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD"), text=True).strip()
    tree = subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD:toolkit"), text=True).strip()
    return build_source_package(
        repo,
        tmp_path / "toolkit.tar",
        commit=commit,
        tree=tree,
        tracked_git=True,
        git_subtree="toolkit",
        governed_runtime_only=False,
        package_role="toolkit",
    )


def test_runtime_image_identity_defaults_to_original_path_and_rejects_symlinks(tmp_path: Path) -> None:
    image = tmp_path / "runtime.sqsh"
    image.write_bytes(b"image")
    identity = identify_runtime_image(image)
    assert identity.policy == "digest-checked"
    assert identity.path == image.resolve()
    verify_runtime_image(identity)
    link = tmp_path / "link.sqsh"
    link.symlink_to(image)
    with pytest.raises(ValueError, match=r"unsafe|symlink"):
        identify_runtime_image(link)


def test_runtime_image_identity_canonicalizes_relative_and_dotdot_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image = images / "runtime.sqsh"
    image.write_bytes(b"image")
    monkeypatch.chdir(tmp_path)

    relative = identify_runtime_image(Path("images/runtime.sqsh"))
    dotdot = identify_runtime_image(tmp_path / "images" / ".." / "images" / "runtime.sqsh")

    assert relative.path == image.resolve(strict=True)
    assert dotdot.path == image.resolve(strict=True)
    assert image_identity_from_mapping(relative.to_mapping()) == relative
    assert image_identity_from_mapping(dotdot.to_mapping()) == dotdot


def test_runtime_image_identity_canonicalizes_ancestor_symlink_but_rejects_final_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    image = real / "runtime.sqsh"
    image.write_bytes(b"image")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    assert identify_runtime_image(alias / "runtime.sqsh").path == image.resolve(strict=True)
    final_alias = tmp_path / "final.sqsh"
    final_alias.symlink_to(image)
    with pytest.raises(ValueError, match=r"unsafe|symlink"):
        identify_runtime_image(final_alias)


def test_runtime_image_trusted_cache_uses_same_canonical_path_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "runtime.sqsh"
    cache = tmp_path / "cache.sqsh"
    image.write_bytes(b"image")
    cache.write_bytes(b"image")
    monkeypatch.chdir(tmp_path)

    identity = identify_runtime_image(Path("runtime.sqsh"), policy="trusted-cache", trusted_cache=Path("cache.sqsh"))

    assert identity.path == cache.resolve(strict=True)
    assert image_identity_from_mapping(identity.to_mapping()) == identity


def test_runtime_image_verification_rejects_replacement(tmp_path: Path) -> None:
    image = tmp_path / "runtime.sqsh"
    image.write_bytes(b"image")
    identity = identify_runtime_image(image)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"other")
    replacement.replace(image)
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_runtime_image(identity)


def test_governed_srun_isolated_direct_python_and_prechecks_image(tmp_path: Path) -> None:
    image = ImageIdentity(
        format_version=1,
        policy="digest-checked",
        path=Path("/images/runtime.sqsh"),
        size_bytes=5,
        sha256="a" * 64,
    )
    text = render_governed_srun(
        image,
        mounts="/stage:/stage",
        bootstrap_args=("run", "--step", "preflight"),
    )
    lines = text.splitlines()
    assert len(lines) == 2
    assert "/usr/bin/python3 -I -S -c" in lines[0]
    assert lines[1].startswith("srun ")
    assert all(command not in text for command in ("sha256sum", "wc -c", "test ! -L"))
    assert "env -i" in text
    assert "/opt/bspp-orchestration-env/.pixi/envs/default/bin/python -I /opt/bspp/execution_bootstrap.py" in text
    assert "entrypoint.sh" not in text
    for forbidden in ("PYTHONPATH", "PYTHONHOME", "CONDA", "VIRTUAL_ENV", "PIXI", "LD_PRELOAD", "BASH_ENV"):
        assert forbidden not in text


def test_renderer_five_slurm_environment_contract_forwards_array_parent_through_bootstrap_scrub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = ImageIdentity(1, "digest-checked", Path("/images/runtime.sqsh"), 5, "a" * 64)
    fake_srun = tmp_path / "srun"
    fake_srun.write_text(
        "#!/bin/sh\n"
        'test "$1" = "--container-image=/images/runtime.sqsh"\n'
        'test "$2" = "--container-mounts=/stage:/stage"\n'
        'test "$3" = "--no-container-mount-home"\n'
        'printf invoked > "$FAKE_SRUN_MARKER"\n'
        "shift 3\n"
        'PATH=/usr/bin:/bin exec "$@"\n'
    )
    fake_srun.chmod(0o755)
    fake_python = tmp_path / "fake-python"
    fake_python.write_text('#!/bin/sh\nprintf \'%s_%s\\n\' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID"\n')
    fake_python.chmod(0o755)
    monkeypatch.setattr(execution_bootstrap, "PIXI_PYTHON_PATH", fake_python)
    rendered = render_governed_srun(
        image,
        mounts="/stage:/stage",
        bootstrap_args=("run",),
        slurm_environment_contract_version=2,
    )
    srun_line = rendered.splitlines()[1]
    invocation_marker = tmp_path / "fake-srun-invoked"
    completed = subprocess.run(
        ("/bin/bash", "-c", srun_line),
        env={
            **os.environ,
            "PATH": str(tmp_path),
            "SLURM_ARRAY_JOB_ID": "4000",
            "SLURM_JOB_ID": "4007",
            "SLURM_ARRAY_TASK_ID": "853",
            "FAKE_SRUN_MARKER": str(invocation_marker),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "4000_853\n"
    assert "SLURM_JOB_ID=${SLURM_ARRAY_JOB_ID:?}" in srun_line
    assert invocation_marker.read_text() == "invoked"

    monkeypatch.setenv("SLURM_JOB_ID", "4000")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "853")
    scrubbed = execution_bootstrap._scrubbed_environment(tmp_path)
    assert scrubbed["SLURM_JOB_ID"] == "4000"
    assert scrubbed["SLURM_ARRAY_TASK_ID"] == "853"

    invocation_marker.unlink()

    missing_parent = subprocess.run(
        ("/bin/bash", "-c", srun_line),
        env={
            **os.environ,
            "PATH": str(tmp_path),
            "SLURM_JOB_ID": "4007",
            "SLURM_ARRAY_TASK_ID": "853",
            "FAKE_SRUN_MARKER": str(invocation_marker),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_parent.returncode != 0
    assert not invocation_marker.exists()


def test_rendered_host_image_verifier_rejects_replacement_and_final_symlink(tmp_path: Path) -> None:
    image_path = tmp_path / "runtime.sqsh"
    image_path.write_bytes(b"image")
    identity = identify_runtime_image(image_path)
    verifier = render_governed_srun(identity, mounts="/stage:/stage", bootstrap_args=("qualify",)).splitlines()[0]
    assert subprocess.run(verifier, shell=True, capture_output=True).returncode == 0

    replacement = tmp_path / "replacement.sqsh"
    replacement.write_bytes(b"other")
    replacement.replace(image_path)
    assert subprocess.run(verifier, shell=True, capture_output=True).returncode != 0

    image_path.unlink()
    other = tmp_path / "other.sqsh"
    other.write_bytes(b"image")
    image_path.symlink_to(other)
    assert subprocess.run(verifier, shell=True, capture_output=True).returncode != 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: {**value, "extra": True},
        lambda value: {key: item for key, item in value.items() if key != "path"},
        lambda value: {**value, "path": "relative.sqsh"},
        lambda value: {**value, "size_bytes": -1},
        lambda value: {**value, "size_bytes": (64 << 30) + 1},
        lambda value: {**value, "sha256": "A" * 64},
    ],
)
def test_image_identity_mapping_rejects_noncanonical_or_forged_fields(tmp_path: Path, mutation: object) -> None:
    image = tmp_path / "runtime.sqsh"
    image.write_bytes(b"image")
    mapping = identify_runtime_image(image).to_mapping()
    with pytest.raises(ValueError):
        image_identity_from_mapping(mutation(mapping))  # type: ignore[operator]


def test_container_bakes_fixed_governed_bootstrap() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "containers" / "Dockerfile").read_text()
    source = "packages/orchestration-control/src/bspp/orchestration/control/execution_bootstrap.py"
    assert f"COPY {source} /opt/bspp/execution_bootstrap.py" in dockerfile


def test_execute_governed_verifies_sources_before_runtime_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        "bspp.orchestration.control.execution_bootstrap.verify_runtime_image", lambda _identity: events.append("image")
    )
    monkeypatch.setattr(
        "bspp.orchestration.control.execution_bootstrap.prepare_governed_sources",
        lambda *_args, **_kwargs: events.append("sources"),
    )
    execute_governed(
        source=object(),  # type: ignore[arg-type]
        toolkit=object(),  # type: ignore[arg-type]
        image=object(),  # type: ignore[arg-type]
        source_destination=tmp_path / "source",
        toolkit_destination=tmp_path / "toolkit",
        runtime_entrypoint=lambda: events.append("import"),
    )
    assert events == ["image", "sources", "import"]


def test_bootstrap_installs_only_the_exact_qualified_runtime_ipsae_binary(tmp_path: Path) -> None:
    source = tmp_path / "qualified-ipsae"
    payload = b"#!/bin/sh\nexit 0\n"
    source.write_bytes(payload)
    destination = tmp_path / "private-toolkit" / "afdb_integration_kit" / "ipsae" / "ipsae_cpp"

    _install_runtime_ipsae_binary(
        source,
        destination,
        expected_sha256=sha256(payload).hexdigest(),
        expected_size=len(payload),
    )

    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o777 == 0o500
    with pytest.raises(ValueError, match="identity mismatch"):
        _install_runtime_ipsae_binary(
            source,
            destination,
            expected_sha256="0" * 64,
            expected_size=len(payload),
        )


def test_runtime_ipsae_install_handles_forced_short_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "qualified-ipsae"
    payload = b"0123456789abcdef"
    source.write_bytes(payload)
    destination = tmp_path / "private-toolkit" / "ipsae_cpp"
    real_write = os.write
    writes: list[int] = []

    def short_write(descriptor: int, data: bytes | bytearray | memoryview) -> int:
        chunk = bytes(data[:3])
        writes.append(len(chunk))
        return real_write(descriptor, chunk)

    monkeypatch.setattr(os, "write", short_write)
    _install_runtime_ipsae_binary(
        source,
        destination,
        expected_sha256=sha256(payload).hexdigest(),
        expected_size=len(payload),
    )

    assert len(writes) > 1
    assert destination.read_bytes() == payload


def test_runtime_ipsae_install_fails_closed_when_write_stalls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "qualified-ipsae"
    payload = b"payload"
    source.write_bytes(payload)
    destination = tmp_path / "private-toolkit" / "ipsae_cpp"
    monkeypatch.setattr(os, "write", lambda _descriptor, _data: 0)

    with pytest.raises(ValueError, match="write stalled"):
        _install_runtime_ipsae_binary(
            source,
            destination,
            expected_sha256=sha256(payload).hexdigest(),
            expected_size=len(payload),
        )

    assert not destination.exists()
    assert not destination.with_name(".ipsae_cpp.qualified").exists()


def test_runtime_ipsae_install_independently_rejects_truncated_temporary_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "qualified-ipsae"
    payload = b"payload"
    source.write_bytes(payload)
    destination = tmp_path / "private-toolkit" / "ipsae_cpp"
    real_write = os.write

    def dishonest_write(descriptor: int, data: bytes | bytearray | memoryview) -> int:
        real_write(descriptor, bytes(data[:1]))
        return len(data)

    monkeypatch.setattr(os, "write", dishonest_write)
    with pytest.raises(ValueError, match="temporary identity mismatch"):
        _install_runtime_ipsae_binary(
            source,
            destination,
            expected_sha256=sha256(payload).hexdigest(),
            expected_size=len(payload),
        )

    assert not destination.exists()


def test_qualification_smoke_records_success_after_gpu_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record = tmp_path / "record.json"
    record.write_text('{"status":"submitted","evidence_status":"pending"}')
    monkeypatch.setattr(
        "bspp.orchestration.control.execution_bootstrap.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(("nvidia-smi",), 0),
    )
    qualification_smoke(record, tuple_id="a" * 64)
    payload = __import__("json").loads(record.read_text())
    assert payload["status"] == "succeeded"
    assert payload["evidence_status"] == "complete"
    assert payload["tuple_id"] == "a" * 64


def test_qualification_smoke_replaces_record_atomically(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record = tmp_path / "qualification.json"
    record.write_text("{}\n")
    monkeypatch.setattr(execution_bootstrap.subprocess, "run", lambda *args, **kwargs: None)
    replaced: list[tuple[Path, Path]] = []
    original_replace = os.replace

    def record_replace(source: Path, destination: Path) -> None:
        replaced.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(execution_bootstrap.os, "replace", record_replace)

    qualification_smoke(record, tuple_id="b" * 64)

    assert len(replaced) == 1
    assert replaced[0][1] == record


def test_render_governed_srun_does_not_expose_unused_bootstrap_module_parameter() -> None:
    assert "bootstrap_module" not in inspect.signature(render_governed_srun).parameters


def test_bootstrap_main_dispatches_qualification(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        "bspp.orchestration.control.execution_bootstrap.qualification_smoke",
        lambda path, *, tuple_id: calls.append((path, tuple_id)),
    )
    main(("qualify", "--tuple-id", "a" * 64, "--record", str(tmp_path / "record.json")))
    assert calls == [(tmp_path / "record.json", "a" * 64)]


def test_bootstrap_run_subprocess_verifies_then_executes_exact_argv(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload_file = source / "packages/orchestration-runtime/src/payload.py"
    payload_file.parent.mkdir(parents=True)
    payload_file.write_text("VALUE = 1\n")
    package = build_source_package(
        source, tmp_path / "source.tar", commit="a" * 40, tree="b" * 40, package_role="orchestration"
    )
    toolkit = _toolkit_package(tmp_path / "toolkit-case")
    image_path = tmp_path / "image.sqsh"
    image_path.write_bytes(b"image")
    image = identify_runtime_image(image_path)
    record = tmp_path / "identities.json"
    record.write_text(
        json.dumps(
            {
                "source_package_identity": package.to_mapping(),
                "toolkit_package_identity": toolkit.to_mapping(),
                "image_identity": image.to_mapping(),
            }
        )
    )
    marker = tmp_path / "marker.json"
    code = "import json,os,sys;open(sys.argv[1],'w').write(json.dumps({'argv':sys.argv[2:],'env':dict(os.environ)}))"
    intended = [sys.executable, "-I", "-S", "-c", code, str(marker), "exact", "argv"]
    process_environment = {**os.environ, "SLURM_CPUS_PER_TASK": "7"}
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "bspp.orchestration.control.execution_bootstrap",
            "run",
            "--identity-record",
            str(record),
            "--source-destination",
            str(tmp_path / "extracted"),
            "--toolkit-destination",
            str(tmp_path / "toolkit-copy"),
            "--exec-argv-json",
            json.dumps(intended),
        ),
        capture_output=True,
        text=True,
        env=process_environment,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(marker.read_text())
    assert observed["argv"] == ["exact", "argv"]
    assert observed["env"]["SLURM_CPUS_PER_TASK"] == "7"
    observed_names = set(observed["env"]) - {"__CF_USER_TEXT_ENCODING"}
    assert observed_names <= {
        "BSPP_RUNSPEC",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_ID",
        "SLURM_TMPDIR",
    }


def test_bootstrap_run_does_not_exec_after_package_tamper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload_file = source / "packages/orchestration-runtime/src/payload"
    payload_file.parent.mkdir(parents=True)
    payload_file.write_text("payload")
    package = build_source_package(
        source, tmp_path / "source.tar", commit="a" * 40, tree="b" * 40, package_role="orchestration"
    )
    package.package_path.write_bytes(package.package_path.read_bytes() + b"tamper")
    toolkit = _toolkit_package(tmp_path / "toolkit-case")
    image_path = tmp_path / "image.sqsh"
    image_path.write_bytes(b"image")
    record = tmp_path / "identities.json"
    record.write_text(
        json.dumps(
            {
                "source_package_identity": package.to_mapping(),
                "toolkit_package_identity": toolkit.to_mapping(),
                "image_identity": identify_runtime_image(image_path).to_mapping(),
            }
        )
    )
    monkeypatch.setattr(os, "execve", lambda *_args: pytest.fail("execve called after failed verification"))
    with pytest.raises(ValueError, match="package size mismatch"):
        main(
            (
                "run",
                "--identity-record",
                str(record),
                "--source-destination",
                str(tmp_path / "out"),
                "--toolkit-destination",
                str(tmp_path / "toolkit-out"),
                "--exec-argv-json",
                json.dumps([sys.executable, "-c", "pass"]),
            )
        )


def test_bootstrap_run_does_not_exec_after_toolkit_package_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload_file = source / "packages/orchestration-runtime/src/payload"
    payload_file.parent.mkdir(parents=True)
    payload_file.write_text("payload")
    package = build_source_package(
        source, tmp_path / "source.tar", commit="a" * 40, tree="b" * 40, package_role="orchestration"
    )
    toolkit = _toolkit_package(tmp_path / "toolkit-case")
    toolkit.package_path.write_bytes(toolkit.package_path.read_bytes() + b"tamper")
    image_path = tmp_path / "image.sqsh"
    image_path.write_bytes(b"image")
    record = tmp_path / "identities.json"
    record.write_text(
        json.dumps(
            {
                "source_package_identity": package.to_mapping(),
                "toolkit_package_identity": toolkit.to_mapping(),
                "image_identity": identify_runtime_image(image_path).to_mapping(),
            }
        )
    )
    monkeypatch.setattr(os, "execve", lambda *_args: pytest.fail("execve called after failed verification"))
    with pytest.raises(ValueError, match="package size mismatch"):
        main(
            (
                "run",
                "--identity-record",
                str(record),
                "--source-destination",
                str(tmp_path / "out"),
                "--toolkit-destination",
                str(tmp_path / "toolkit-out"),
                "--exec-argv-json",
                json.dumps([sys.executable, "-c", "pass"]),
            )
        )


# ---------------------------------------------------------------------------
# Baked runtime-iPSAE revision check (Option A: file-authoritative provenance)
# ---------------------------------------------------------------------------


def _write_baked_provenance(root: Path, commit: str = "a" * 40) -> Path:
    prov = root / "provenance.json"
    prov.write_text(json.dumps({"schema_version": 1, "commit": commit}), encoding="utf-8")
    return prov


def test_baked_runtime_ipsae_revision_matches_provenance_passes(tmp_path: Path) -> None:
    _write_baked_provenance(tmp_path, commit="a" * 40)
    _check_baked_runtime_ipsae_revision(tmp_path, "a" * 40)  # no raise


def test_baked_runtime_ipsae_revision_mismatch_fails_closed(tmp_path: Path) -> None:
    _write_baked_provenance(tmp_path, commit="a" * 40)
    with pytest.raises(ValueError, match="does not match baked toolkit provenance"):
        _check_baked_runtime_ipsae_revision(tmp_path, "b" * 40)


def test_baked_runtime_ipsae_revision_missing_provenance_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provenance missing or unsafe"):
        _check_baked_runtime_ipsae_revision(tmp_path, "a" * 40)


def test_baked_runtime_ipsae_revision_unreadable_provenance_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "provenance.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance unreadable"):
        _check_baked_runtime_ipsae_revision(tmp_path, "a" * 40)


def test_baked_runtime_ipsae_revision_non_hex_commit_fails_closed(tmp_path: Path) -> None:
    _write_baked_provenance(tmp_path, commit="not-a-sha")
    with pytest.raises(ValueError, match="not a 40-hex sha"):
        _check_baked_runtime_ipsae_revision(tmp_path, "a" * 40)
