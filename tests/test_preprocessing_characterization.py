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

"""Behavioral coverage for the preprocessing CUDA loader and characterization."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "containers" / "preprocessing"
ENTRYPOINT = IMAGE / "entrypoint.sh"
HELPER = IMAGE / "carry-characterization.sh"
PROBE = IMAGE / "cuda-driver-probe.py"
WRAPPER = ROOT / "containers" / "scripts" / "slurm-preprocessing-carry-characterization.sh"
COMPAT_DIR = "/usr/local/cuda-12.6/compat"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source))
    path.chmod(0o755)


def _copy_entrypoint(tmp_path: Path, compat: Path) -> Path:
    script = tmp_path / "entrypoint.sh"
    script.write_text(ENTRYPOINT.read_text().replace(f'CUDA_COMPAT_DIR="{COMPAT_DIR}"', f'CUDA_COMPAT_DIR="{compat}"'))
    script.chmod(0o755)
    return script


def _valid_compatibility_directory(tmp_path: Path) -> tuple[Path, Path]:
    compat = tmp_path / "compat"
    compat.mkdir()
    target = compat / "libcuda.so.560.35.05"
    target.write_bytes(b"fake CUDA compatibility library")
    (compat / "libcuda.so.1").symlink_to(target.name)
    return compat, target


def test_entrypoint_prepends_loader_path_and_preserves_argv_and_environment(tmp_path: Path) -> None:
    compat, target = _valid_compatibility_directory(tmp_path)
    script = _copy_entrypoint(tmp_path, compat)
    record = tmp_path / "record.json"
    executable = tmp_path / "record-exec"
    _write_executable(
        executable,
        """\
        #!/usr/bin/env python3
        import json, os, sys
        from pathlib import Path
        Path(os.environ["RECORD"]).write_text(json.dumps({
            "argv": sys.argv[1:], "ld": os.environ["LD_LIBRARY_PATH"], "path": os.environ["PATH"]
        }))
        """,
    )
    inherited_path = os.environ["PATH"]
    argv = ["space separated", "literal*glob", "quotes'\"", "--flag=value"]
    result = subprocess.run(
        [str(script), str(executable), *argv],
        env={**os.environ, "RECORD": str(record), "LD_LIBRARY_PATH": "/inherited/one:/inherited/two"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads(record.read_text())
    assert observed["argv"] == argv
    assert observed["ld"] == f"{compat}:/inherited/one:/inherited/two"
    assert observed["path"] == f"/opt/bspp/environment/bin:{inherited_path}"
    assert (compat / "libcuda.so.1").resolve() == target


@pytest.mark.parametrize(
    "failure", ["missing-library", "regular-soname", "dangling-library", "escaped-target", "nonregular-target"]
)
def test_entrypoint_rejects_invalid_compatibility_library(tmp_path: Path, failure: str) -> None:
    compat = tmp_path / "compat"
    compat.mkdir()
    if failure == "regular-soname":
        (compat / "libcuda.so.1").write_bytes(b"not a pinned symlink")
    elif failure == "dangling-library":
        (compat / "libcuda.so.1").symlink_to("missing-target")
    elif failure == "escaped-target":
        outside = tmp_path / "outside-libcuda.so"
        outside.write_bytes(b"outside")
        (compat / "libcuda.so.1").symlink_to(outside)
    elif failure == "nonregular-target":
        target = compat / "directory-target"
        target.mkdir()
        (compat / "libcuda.so.1").symlink_to(target.name)
    script = _copy_entrypoint(tmp_path, compat)

    result = subprocess.run([str(script), "/bin/true"], capture_output=True, text=True, check=False)

    assert result.returncode == 126
    assert "CUDA compatibility" in result.stderr


def test_entrypoint_rejects_missing_compatibility_directory(tmp_path: Path) -> None:
    script = _copy_entrypoint(tmp_path, tmp_path / "missing")

    result = subprocess.run([str(script), "/bin/true"], capture_output=True, text=True, check=False)

    assert result.returncode == 126
    assert "directory is unavailable" in result.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="uid 0 bypasses ordinary read-permission checks")
@pytest.mark.parametrize("unreadable", ["directory", "target"])
def test_entrypoint_rejects_unreadable_compatibility_composition(tmp_path: Path, unreadable: str) -> None:
    compat, target = _valid_compatibility_directory(tmp_path)
    changed = compat if unreadable == "directory" else target
    changed.chmod(0)
    script = _copy_entrypoint(tmp_path, compat)

    try:
        result = subprocess.run([str(script), "/bin/true"], capture_output=True, text=True, check=False)
    finally:
        changed.chmod(0o755 if changed.is_dir() else 0o644)

    assert result.returncode == 126


def _load_probe(path: Path = PROBE) -> ModuleType:
    spec = importlib.util.spec_from_file_location("bspp_test_cuda_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mapped_system_libz() -> Path | None:
    try:
        ctypes.CDLL("libz.so.1")
    except OSError:
        return None
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and Path(fields[5]).name.startswith("libz.so"):
            return Path(fields[5]).resolve()
    return None


def _probe_copy(tmp_path: Path, *, soname: str, required_directory: Path) -> Path:
    path = tmp_path / "cuda-driver-probe.py"
    source = PROBE.read_text()
    source = source.replace('CUDA_SONAME = "libcuda.so.1"', f"CUDA_SONAME = {json.dumps(soname)}")
    source = source.replace(
        'CUDA_COMPAT_DIR = Path("/usr/local/cuda-12.6/compat")',
        f"CUDA_COMPAT_DIR = Path({json.dumps(str(required_directory))})",
    )
    path.write_text(source)
    return path


def _run_loader_probe(probe: Path, *, library_path: Path) -> subprocess.CompletedProcess[str]:
    code = textwrap.dedent(
        f"""\
        import importlib.util, json, sys
        spec = importlib.util.spec_from_file_location("isolated_cuda_probe", {str(probe)!r})
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _library, observed = module.observe_loader()
        print(json.dumps(observed.__dict__, sort_keys=True))
        """
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "LD_LIBRARY_PATH": str(library_path)},
        capture_output=True,
        text=True,
        check=False,
    )


def test_probe_real_loader_stage_accepts_a_contained_library(tmp_path: Path) -> None:
    system_libz = _mapped_system_libz()
    if system_libz is None:
        pytest.skip("libz.so.1 is unavailable")
    compat = tmp_path / "compat"
    compat.mkdir()
    copied = compat / system_libz.name
    shutil.copy2(system_libz, copied)
    (compat / "libz.so.1").symlink_to(copied.name)
    probe = _probe_copy(tmp_path, soname="libz.so.1", required_directory=compat)

    result = _run_loader_probe(probe, library_path=compat)

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["compat_library_resolved_target"] == str(copied)
    assert observed["mapped_library_resolved_target"] == str(copied)
    assert str(copied) in observed["maps_line"]


@pytest.mark.parametrize("escape", ["symlink", "prefix-sibling"])
def test_probe_real_loader_stage_rejects_library_escape(tmp_path: Path, escape: str) -> None:
    system_libz = _mapped_system_libz()
    if system_libz is None:
        pytest.skip("libz.so.1 is unavailable")
    compat = tmp_path / "compat"
    compat.mkdir()
    outside = tmp_path / ("compat-sibling" if escape == "prefix-sibling" else "outside")
    outside.mkdir()
    copied = outside / system_libz.name
    shutil.copy2(system_libz, copied)
    if escape == "symlink":
        (compat / "libz.so.1").symlink_to(copied)
        library_path = compat
    else:
        (outside / "libz.so.1").symlink_to(copied.name)
        (compat / "libz.so.1").symlink_to(system_libz.name)
        shutil.copy2(system_libz, compat / system_libz.name)
        library_path = outside
    probe = _probe_copy(tmp_path, soname="libz.so.1", required_directory=compat)

    result = _run_loader_probe(probe, library_path=library_path)

    assert result.returncode != 0
    assert "pinned compatibility policy" in result.stderr or "escapes" in result.stderr


def test_probe_containment_predicate_rejects_nonregular_mapped_target(tmp_path: Path) -> None:
    module = _load_probe()
    compat = tmp_path / "compat"
    compat.mkdir()
    target = compat / "libtest.so.1.0"
    target.write_bytes(b"library")
    (compat / "libtest.so.1").symlink_to(target.name)
    nonregular = compat / "mapped-directory"
    nonregular.mkdir()

    with pytest.raises(RuntimeError, match="not a readable regular file"):
        module.validate_mapped_library(mapped_path=nonregular, soname="libtest.so.1", required_directory=compat)


def test_probe_visible_device_cardinality_rejects_blank_entries() -> None:
    module = _load_probe()

    assert module._visible_device_count("0,GPU-abcd") == 2
    with pytest.raises(RuntimeError, match="nonblank comma-separated"):
        module._visible_device_count("0,")


def _copy_helper(tmp_path: Path, *, probe: Path, mmseqs: Path, search: Path) -> Path:
    fast_sleep = tmp_path / "sleep"
    _write_executable(fast_sleep, "#!/bin/bash\n/bin/sleep 0.005\n")
    substitutions = {
        'PYTHON="/opt/bspp/environment/bin/python"': f'PYTHON="{sys.executable}"',
        'TIMEOUT="/opt/bspp/environment/bin/timeout"': 'TIMEOUT="/usr/bin/timeout"',
        'SLEEP="/opt/bspp/environment/bin/sleep"': f'SLEEP="{fast_sleep}"',
        'TAIL="/opt/bspp/environment/bin/tail"': 'TAIL="/usr/bin/tail"',
        'MMSEQS="/usr/local/bin/mmseqs"': f'MMSEQS="{mmseqs}"',
        'COLABFOLD_SEARCH="/usr/local/bin/colabfold_search"': f'COLABFOLD_SEARCH="{search}"',
        'CUDA_DRIVER_PROBE="/opt/bspp/bin/bspp-preprocessing-cuda-driver-probe"': (f'CUDA_DRIVER_PROBE="{probe}"'),
    }
    source = HELPER.read_text()
    for old, new in substitutions.items():
        assert old in source
        source = source.replace(old, new)
    helper = tmp_path / "carry-characterization.sh"
    helper.write_text(source)
    helper.chmod(0o755)
    return helper


def _fake_probe(path: Path) -> None:
    _write_executable(
        path,
        """\
        #!/usr/bin/env python3
        import json, sys
        from pathlib import Path
        Path(sys.argv[1]).write_text(json.dumps({"probe": "passed"}) + "\\n")
        """,
    )


def _fake_server(path: Path) -> None:
    _write_executable(
        path,
        """\
        #!/usr/bin/env python3
        import os, signal, sys, time
        from pathlib import Path
        print("fake gpuserver started", flush=True)
        mode = os.environ.get("FAKE_SERVER_MODE", "success")
        if mode == "warmup-death":
            raise SystemExit(42)
        if mode == "post-warmup-death":
            marker = Path(os.environ["FAKE_SEARCH_STARTED_MARKER"])
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not marker.exists():
                raise SystemExit(44)
            raise SystemExit(43)
        signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
        while True:
            time.sleep(0.02)
        """,
    )


def _fake_search(path: Path) -> None:
    _write_executable(
        path,
        """\
        #!/usr/bin/env python3
        import os, subprocess, sys, time
        from pathlib import Path
        mode = os.environ.get("FAKE_SEARCH_MODE", "success")
        marker = os.environ.get("FAKE_GRANDCHILD_MARKER")
        Path(os.environ["FAKE_SEARCH_STARTED_MARKER"]).write_text("started")
        if mode == "long":
            child = f"import time; from pathlib import Path; time.sleep(1); Path({marker!r}).write_text('escaped')"
            subprocess.Popen([sys.executable, "-c", child])
            while True:
                time.sleep(0.05)
        output = Path(sys.argv[5])
        output.joinpath("AFDB_alpha.a3m").write_text(">query\\nAAAA\\n>hit\\nCCCC\\n")
        output.joinpath("AFDB_mu.a3m").write_text(">query\\nGGGG\\n>hit\\nTTTT\\n")
        output.joinpath("2.a3m").write_bytes(b"#49\\t1\\n")
        output.joinpath("3.a3m").write_bytes(b"#36\\t1\\n")
        """,
    )


def _helper_fixture(tmp_path: Path) -> tuple[Path, Path]:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    fixture.joinpath("remaining.fa").write_text(">AFDB_alpha\nAAAA\n>AFDB_mu\nCCCC\n")
    fixture.joinpath("search-output").mkdir()
    fixture.joinpath("output").mkdir()
    probe = tmp_path / "probe"
    mmseqs = tmp_path / "mmseqs"
    search = tmp_path / "search"
    _fake_probe(probe)
    _fake_server(mmseqs)
    _fake_search(search)
    return _copy_helper(tmp_path, probe=probe, mmseqs=mmseqs, search=search), fixture


def _run_helper(helper: Path, fixture: Path, *, server_mode: str, search_mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(helper), str(fixture)],
        env={
            **os.environ,
            "BSPP_CARRY_DATABASE_ROOT": "/database",
            "CUDA_VISIBLE_DEVICES": "0",
            "LD_LIBRARY_PATH": f"{COMPAT_DIR}:/inherited",
            "SLURM_JOB_ID": "12345",
            "SLURMD_NODENAME": "node-a",
            "FAKE_SERVER_MODE": server_mode,
            "FAKE_SEARCH_MODE": search_mode,
            "FAKE_GRANDCHILD_MARKER": str(fixture / "grandchild-escaped"),
            "FAKE_SEARCH_STARTED_MARKER": str(fixture / "search-started"),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_helper_fails_immediately_when_server_dies_during_warmup(tmp_path: Path) -> None:
    helper, fixture = _helper_fixture(tmp_path)

    result = _run_helper(helper, fixture, server_mode="warmup-death", search_mode="success")

    assert result.returncode == 42
    assert "during the 60-second warmup" in result.stderr
    assert not (tmp_path / "evidence.json").exists()


def test_helper_kills_complete_search_process_group_when_server_dies(tmp_path: Path) -> None:
    helper, fixture = _helper_fixture(tmp_path)

    result = _run_helper(helper, fixture, server_mode="post-warmup-death", search_mode="long")
    subprocess.run(["/bin/sleep", "1.2"], check=True)

    assert result.returncode == 43
    assert "while the bounded search was running" in result.stderr
    assert fixture.joinpath("search-started").read_text() == "started"
    assert not (fixture / "grandchild-escaped").exists()
    assert not (tmp_path / "evidence.json").exists()


def test_helper_completes_a_bounded_search_and_uses_pinned_inventory(tmp_path: Path) -> None:
    helper, fixture = _helper_fixture(tmp_path)

    result = _run_helper(helper, fixture, server_mode="success", search_mode="success")

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in fixture.joinpath("search-output").iterdir()) == [
        "2.a3m",
        "3.a3m",
        "AFDB_alpha.a3m",
        "AFDB_mu.a3m",
    ]


def _validator_source() -> str:
    wrapper = WRAPPER.read_text()
    marker = "<<'PY'\n"
    assert wrapper.count(marker) == 1
    body = wrapper.split(marker, 1)[1]
    assert body.endswith("\nPY\n")
    return body[: -len("\nPY\n")]


def test_wrapper_emits_a2_supervision_and_requests_45_minutes_by_default() -> None:
    wrapper = WRAPPER.read_text()

    assert wrapper.count("${BSPP_CARRY_TIME:-00:45:00}") == 1
    assert wrapper.count('"gpuserver_warmup_seconds": 60') == 1
    assert wrapper.count('"search_timeout_seconds": 1800') == 1
    assert wrapper.count('"search_kill_after_seconds": 10') == 1
    assert wrapper.count('"search_threads": 64') == 1


def _builder_fake_probe(path: Path, real_probe: Path, helper: Path) -> None:
    _write_executable(
        path,
        f"""\
        #!/usr/bin/env python3
        import hashlib, importlib.util, json, os, sys
        from pathlib import Path
        spec = importlib.util.spec_from_file_location("builder_probe", {str(real_probe)!r})
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        target = "{COMPAT_DIR}/libcuda.so.560.35.05"
        loader = module.LoaderObservation(
            soname="libcuda.so.1", compat_directory="{COMPAT_DIR}",
            compat_soname_path="{COMPAT_DIR}/libcuda.so.1",
            compat_library_resolved_target=target, mapped_library_path=target,
            mapped_library_resolved_target=target, maps_line="0000-1111 r-xp 0000 00:00 0 " + target,
        )
        cuda = module.CudaObservation(
            driver_api_version=12060, device_count=1,
            cu_driver_get_version_return_code=0, cu_init_return_code=0,
            cu_device_get_count_return_code=0,
        )
        digest = lambda value: hashlib.sha256(Path(value).read_bytes()).hexdigest()
        payload = module.build_evidence_payload(
            loader=loader, cuda=cuda, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
            ld_library_path=os.environ["LD_LIBRARY_PATH"], slurm_job_id=os.environ["SLURM_JOB_ID"],
            slurmd_nodename=os.environ["SLURMD_NODENAME"], probe_sha256=digest(sys.argv[0]),
            helper_sha256=digest({str(helper)!r}),
        )
        Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
        """,
    )


def _chained_fixture(tmp_path: Path) -> tuple[Path, dict[str, str], list[Path]]:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    fixture.joinpath("remaining.fa").write_text(">AFDB_alpha\nAAAA\n>AFDB_mu\nCCCC\n")
    fixture.joinpath("search-output").mkdir()
    fixture.joinpath("output").mkdir()
    mmseqs = tmp_path / "mmseqs"
    search = tmp_path / "search"
    probe = tmp_path / "probe"
    _fake_server(mmseqs)
    _fake_search(search)
    placeholder_probe = tmp_path / "placeholder-probe"
    _fake_probe(placeholder_probe)
    helper = _copy_helper(tmp_path, probe=placeholder_probe, mmseqs=mmseqs, search=search)
    _builder_fake_probe(probe, PROBE, helper)
    helper.write_text(
        helper.read_text().replace(f'CUDA_DRIVER_PROBE="{placeholder_probe}"', f'CUDA_DRIVER_PROBE="{probe}"')
    )

    host_paths = [
        fixture / "host-slurm-job-id.txt",
        fixture / "host-slurmd-nodename.txt",
        fixture / "host-cuda-visible-devices.txt",
        fixture / "host-nvidia-smi.csv",
        fixture / "slurm-wrapper-sha256.txt",
    ]
    values = ["12345\n", "node-a\n", "0\n", "NVIDIA A100, GPU-uuid, 535.104.12\n", "f" * 64 + "\n"]
    for path, value in zip(host_paths, values, strict=True):
        path.write_text(value)
    environment = {
        **os.environ,
        "BSPP_CARRY_DATABASE_ROOT": "/database",
        "BSPP_CARRY_CLUSTER_PROFILE": "example-cluster-oci-iad",
        "BSPP_CARRY_RUNTIME_CONTRACT_ID": "0" * 64,
        "BSPP_CARRY_ADAPTER_VERSION": "preprocessing-scientific-backend-v3",
        "BSPP_CARRY_IMAGE": "/image/preprocessing.sqsh",
        "BSPP_CARRY_IMAGE_SHA256": "1" * 64,
        "BSPP_CARRY_OCI_DIGEST": "sha256:" + "2" * 64,
        "BSPP_CARRY_SOURCE_BUNDLE_ID": "source-bundle",
        "BSPP_CARRY_SOURCE_BUNDLE": "/source/bundle.tar.zst",
        "BSPP_CARRY_SOURCE_BUNDLE_SHA256": "3" * 64,
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": f"{COMPAT_DIR}:/inherited",
        "SLURM_JOB_ID": "12345",
        "SLURMD_NODENAME": "node-a",
        "FAKE_SERVER_MODE": "success",
        "FAKE_SEARCH_MODE": "success",
        "FAKE_GRANDCHILD_MARKER": str(fixture / "grandchild-escaped"),
        "FAKE_SEARCH_STARTED_MARKER": str(fixture / "search-started"),
    }
    environment["BSPP_CARRY_INPUT_SHA256"] = hashlib.sha256(fixture.joinpath("remaining.fa").read_bytes()).hexdigest()
    result = subprocess.run(
        [str(helper), str(fixture)], env=environment, capture_output=True, text=True, check=False, timeout=10
    )
    assert result.returncode == 0, result.stderr
    return fixture, environment, host_paths


def _run_validator(
    tmp_path: Path,
    fixture: Path,
    environment: dict[str, str],
    host_paths: list[Path],
    *,
    preexisting_destination: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    validator = tmp_path / "validator.py"
    validator.write_text(_validator_source())
    destination = tmp_path / "evidence.json"
    if preexisting_destination is not None:
        destination.write_text(preexisting_destination)
    result = subprocess.run(
        [sys.executable, str(validator), str(fixture), str(destination), *(str(path) for path in host_paths)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, destination


def test_wrapper_exports_default_contract_metadata_to_child_processes(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "exported-defaults.txt"
    _write_executable(
        fake_bin / "sbatch",
        """\
        #!/bin/bash
        printf '%s\\n%s\\n' "$BSPP_CARRY_RUNTIME_CONTRACT_ID" "$BSPP_CARRY_ADAPTER_VERSION" > "$CAPTURE"
        printf '{}\\n' > "$BSPP_CARRY_WORK_ROOT/evidence.json"
        printf '12345\\n'
        """,
    )
    _write_executable(fake_bin / "squeue", "#!/bin/bash\nexit 0\n")
    _write_executable(
        fake_bin / "sacct",
        """\
        #!/bin/bash
        printf '12345|COMPLETED|0:0|00:00:01|node-a\\n'
        """,
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "BSPP_CARRY_CLUSTER_PROFILE": "example-cluster-oci-iad",
        "BSPP_CARRY_IMAGE": "/image/preprocessing.sqsh",
        "BSPP_CARRY_IMAGE_SHA256": "1" * 64,
        "BSPP_CARRY_OCI_DIGEST": "sha256:" + "2" * 64,
        "BSPP_CARRY_SOURCE_BUNDLE": "/source/bundle.tar.zst",
        "BSPP_CARRY_SOURCE_BUNDLE_ID": "source-bundle",
        "BSPP_CARRY_SOURCE_BUNDLE_SHA256": "3" * 64,
        "BSPP_CARRY_DATABASE_ROOT": "/database",
        "BSPP_CARRY_WORK_ROOT": str(tmp_path / "work"),
        "BSPP_CARRY_INPUT_SHA256": "4" * 64,
        "BSPP_CARRY_ACCOUNT": "account",
    }
    environment.pop("BSPP_CARRY_RUNTIME_CONTRACT_ID", None)
    environment.pop("BSPP_CARRY_ADAPTER_VERSION", None)

    result = subprocess.run([str(WRAPPER)], env=environment, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == [
        "6826d13f71a176d5ac483f42d85a9967485f8c171a96e7d20c7d6350efbf3796",
        "preprocessing-scientific-backend-v3",
    ]


def test_real_helper_builder_and_exact_validator_publish_schema_v2_evidence(tmp_path: Path) -> None:
    fixture, environment, host_paths = _chained_fixture(tmp_path)

    result, destination = _run_validator(tmp_path, fixture, environment, host_paths)

    assert result.returncode == 0, result.stderr
    evidence = json.loads(destination.read_text())
    assert evidence["schema_version"] == 2
    assert evidence["characterization"] == "preprocessing-carry-complement-search-v2"
    assert evidence["slurm_job_id"] == "12345"
    assert evidence["slurmd_nodename"] == "node-a"
    assert evidence["cuda_driver_evidence"]["cuda_device_count"] == 1
    assert evidence["source_hashes"]["slurm_wrapper_sha256"] == "f" * 64
    assert [output["sequence_count"] for output in evidence["outputs"]] == [2, 2]
    for member_name in ("AFDB_alpha.a3m", "AFDB_mu.a3m"):
        assert (
            fixture.joinpath("search-output", member_name).read_bytes()
            == fixture.joinpath("output", member_name).read_bytes()
        )
    for relative in (
        "cuda-driver-evidence.json",
        "search-output",
        "output",
        "remaining.fa",
        "gpuserver.log",
        "host-slurm-job-id.txt",
        "host-slurmd-nodename.txt",
        "host-cuda-visible-devices.txt",
        "host-nvidia-smi.csv",
        "slurm-wrapper-sha256.txt",
    ):
        assert fixture.joinpath(relative).exists()
    assert "set(probe) != probe_keys" in _validator_source()
    assert "hashlib.file_digest" in _validator_source()


def _mutate_query_only_named_a3ms(fixture: Path) -> None:
    query_only = {
        "AFDB_alpha.a3m": b"#4,4\t1,1\n>101\t102\nAAAACCCC\n",
        "AFDB_mu.a3m": b"#4,4\t1,1\n>101\t102\nCCCCAAAA\n",
    }
    for member_name, data in query_only.items():
        fixture.joinpath("search-output", member_name).write_bytes(data)


def test_exact_validator_accepts_structurally_valid_query_only_named_a3ms(tmp_path: Path) -> None:
    fixture, environment, host_paths = _chained_fixture(tmp_path)
    remaining_bytes = fixture.joinpath("remaining.fa").read_bytes()
    _mutate_query_only_named_a3ms(fixture)

    result, destination = _run_validator(tmp_path, fixture, environment, host_paths)

    assert result.returncode == 0, result.stderr
    evidence = json.loads(destination.read_text())
    assert [output["sequence_count"] for output in evidence["outputs"]] == [1, 1]
    for member_name in ("AFDB_alpha.a3m", "AFDB_mu.a3m"):
        assert (
            fixture.joinpath("search-output", member_name).read_bytes()
            == fixture.joinpath("output", member_name).read_bytes()
        )
    assert fixture.joinpath("remaining.fa").read_bytes() == remaining_bytes
    assert hashlib.sha256(remaining_bytes).hexdigest() == environment["BSPP_CARRY_INPUT_SHA256"]


@pytest.mark.parametrize(
    ("invalid_data", "expected_error"),
    [
        (b"#4,4\t1\n>101\t102\nAAAACCCC\n", "malformed characterized multimer metadata"),
        (b"#4,4\t1,1\n>101\t102\n", "characterized final A3M record is empty"),
    ],
    ids=["malformed-metadata", "empty-final-record"],
)
def test_exact_validator_rejects_invalid_query_only_multimer_a3m(
    tmp_path: Path, invalid_data: bytes, expected_error: str
) -> None:
    fixture, environment, host_paths = _chained_fixture(tmp_path)
    _mutate_query_only_named_a3ms(fixture)
    fixture.joinpath("search-output", "AFDB_alpha.a3m").write_bytes(invalid_data)

    result, destination = _run_validator(tmp_path, fixture, environment, host_paths)

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not destination.exists()


def test_exact_validator_removes_temporary_file_when_link_publication_fails(tmp_path: Path) -> None:
    fixture, environment, host_paths = _chained_fixture(tmp_path)

    result, destination = _run_validator(
        tmp_path, fixture, environment, host_paths, preexisting_destination="preexisting evidence\n"
    )

    assert result.returncode != 0
    assert destination.read_text() == "preexisting evidence\n"
    assert not destination.with_suffix(".tmp").exists()


def _mutate_job_id(fixture: Path, _environment: dict[str, str]) -> None:
    fixture.joinpath("host-slurm-job-id.txt").write_text("different-job\n")


def _mutate_node(fixture: Path, _environment: dict[str, str]) -> None:
    fixture.joinpath("host-slurmd-nodename.txt").write_text("different-node\n")


def _mutate_blank_row(fixture: Path, _environment: dict[str, str]) -> None:
    fixture.joinpath("host-nvidia-smi.csv").write_text("\n")


def _mutate_row_cardinality(fixture: Path, _environment: dict[str, str]) -> None:
    fixture.joinpath("host-nvidia-smi.csv").write_text(
        "NVIDIA A100, GPU-one, 535.104.12\nNVIDIA A100, GPU-two, 535.104.12\n"
    )


def _mutate_device_count(fixture: Path, _environment: dict[str, str]) -> None:
    path = fixture / "cuda-driver-evidence.json"
    payload = json.loads(path.read_text())
    payload["cuda_device_count"] = 2
    path.write_text(json.dumps(payload))


def _mutate_compat_target(fixture: Path, _environment: dict[str, str]) -> None:
    path = fixture / "cuda-driver-evidence.json"
    payload = json.loads(path.read_text())
    payload["mapped_library_resolved_target"] = "/usr/local/cuda-12.6/compat-sibling/libcuda.so.1"
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    "mutation",
    [
        _mutate_job_id,
        _mutate_node,
        _mutate_blank_row,
        _mutate_row_cardinality,
        _mutate_device_count,
        _mutate_compat_target,
    ],
    ids=["job-id", "node", "nonblank-row", "row-cardinality", "device-count", "compat-target"],
)
def test_exact_validator_rejects_each_binding_without_publishing(
    tmp_path: Path, mutation: Callable[[Path, dict[str, str]], None]
) -> None:
    fixture, environment, host_paths = _chained_fixture(tmp_path)
    mutation(fixture, environment)

    result, destination = _run_validator(tmp_path, fixture, environment, host_paths)

    assert result.returncode != 0
    assert not destination.exists()
