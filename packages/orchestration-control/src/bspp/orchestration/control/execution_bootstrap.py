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

"""Small governed-execution identity and bootstrap boundary."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

_FIXED_BOOTSTRAP_LIB = "/opt/bspp/lib"
if _FIXED_BOOTSTRAP_LIB not in sys.path:
    sys.path.insert(0, _FIXED_BOOTSTRAP_LIB)

from bspp.orchestration.contract.runtime_qualification import runtime_ipsae_evidence_from_mapping  # noqa: E402
from bspp.orchestration.contract.source_package import (  # noqa: E402
    SourcePackageIdentity,
    source_package_identity_from_mapping,
    verify_source_package,
)

IdentityPolicy = Literal["digest-checked", "trusted-cache"]


@dataclass(frozen=True)
class _SlurmEnvironmentContract:
    """Closed renderer-to-bootstrap Slurm environment compatibility contract."""

    version: Literal[1, 2]
    job_id_assignment: str
    array_task_id_assignment: str
    array_success_job_id: str


def slurm_environment_contract(version: int = 1) -> _SlurmEnvironmentContract:
    """Return one closed Slurm environment contract; arbitrary expressions are forbidden."""
    if version == 1:
        return _SlurmEnvironmentContract(
            version=1,
            job_id_assignment="SLURM_JOB_ID=${SLURM_JOB_ID:-}",
            array_task_id_assignment="SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}",
            array_success_job_id='"${SLURM_ARRAY_JOB_ID:?}_${SLURM_ARRAY_TASK_ID:?}"',
        )
    if version == 2:
        return _SlurmEnvironmentContract(
            version=2,
            job_id_assignment="SLURM_JOB_ID=${SLURM_ARRAY_JOB_ID:?}",
            array_task_id_assignment="SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:?}",
            array_success_job_id='"${SLURM_JOB_ID:?}_${SLURM_ARRAY_TASK_ID:?}"',
        )
    raise ValueError(f"unsupported Slurm environment contract version: {version}")


# In-container Python interpreter. The image bakes the pixi environment at
# /opt/bspp-orchestration-env and has no system /usr/bin/python3, so governed
# srun commands must target the pixi interpreter.
PIXI_PYTHON_PATH = Path("/opt/bspp-orchestration-env/.pixi/envs/default/bin/python")

_HOST_IMAGE_VERIFIER = """import hashlib,os,stat,sys
p=sys.argv[1]; expected_size=int(sys.argv[2]); expected_sha=sys.argv[3]
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
try:
 b=os.fstat(fd)
 if not stat.S_ISREG(b.st_mode): raise RuntimeError('runtime image is not regular')
 h=hashlib.sha256(); n=0
 while True:
  chunk=os.read(fd,1048576)
  if not chunk: break
  n+=len(chunk); h.update(chunk)
 a=os.fstat(fd); q=os.stat(p,follow_symlinks=False)
 sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 changed=sig(b)!=sig(a) or not stat.S_ISREG(q.st_mode) or (q.st_dev,q.st_ino)!=(b.st_dev,b.st_ino)
 if changed: raise RuntimeError('runtime image changed while checking')
 if n!=expected_size or h.hexdigest()!=expected_sha: raise RuntimeError('runtime image identity mismatch')
finally:
 os.close(fd)
"""


@dataclass(frozen=True)
class ImageIdentity:
    format_version: int
    policy: IdentityPolicy
    path: Path
    size_bytes: int
    sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "policy": self.policy,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def image_identity_from_mapping(value: Mapping[str, object]) -> ImageIdentity:
    if set(value) != {"format_version", "policy", "path", "size_bytes", "sha256"}:
        raise ValueError("ImageIdentity has missing or extra fields")
    policy = value.get("policy")
    size = value.get("size_bytes")
    path_value = value.get("path")
    digest = value.get("sha256")
    if (
        type(value.get("format_version")) is not int
        or value.get("format_version") != 1
        or policy not in {"digest-checked", "trusted-cache"}
        or type(size) is not int
        or not isinstance(path_value, str)
        or not isinstance(digest, str)
    ):
        raise ValueError("Unsupported ImageIdentity format_version or policy")
    assert isinstance(size, int)
    # Sanity bound only: reject absurd/corrupted sizes. Real BSPP runtime images
    # (CUDA + pixi + baked toolkit) are routinely 10-30 GiB, so the ceiling must
    # comfortably exceed that.
    if size < 0 or size > 64 << 30:
        raise ValueError("ImageIdentity size is out of bounds")
    path = Path(path_value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError("ImageIdentity path must be canonical absolute")
    _require_lower_hex(digest, 64, "ImageIdentity sha256")
    return ImageIdentity(1, cast(IdentityPolicy, policy), path, size, digest)


def identify_runtime_image(
    path: Path, *, policy: IdentityPolicy = "digest-checked", trusted_cache: Path | None = None
) -> ImageIdentity:
    """Bind the configured image itself; cache use is explicit compatibility only."""
    selected, size, digest = _bind_runtime_image_path(path)
    if policy == "trusted-cache" and trusted_cache is not None:
        cache_lexical = trusted_cache.absolute()
        if cache_lexical.exists() or cache_lexical.is_symlink():
            cache, cache_size, cache_digest = _bind_runtime_image_path(trusted_cache)
            if (cache_size, cache_digest) != (size, digest):
                raise ValueError("Trusted runtime image cache collision")
            selected = cache
    return ImageIdentity(1, policy, selected, size, digest)


def _bind_runtime_image_path(path: Path) -> tuple[Path, int, str]:
    """Canonicalize only after rejecting a final symlink, then bind both names."""
    lexical = path.absolute()
    try:
        before = lexical.lstat()
    except OSError as exc:
        raise ValueError(f"Identity path is unsafe or unavailable: {lexical}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Identity path is unsafe, a symlink, or not regular: {lexical}")
    canonical = lexical.resolve(strict=True)
    size, digest, _mode = _stable_regular_identity(canonical)
    try:
        after = lexical.lstat()
        canonical_metadata = canonical.lstat()
    except OSError as exc:
        raise ValueError(f"Identity path changed while binding: {lexical}") from exc
    original_identity = (before.st_dev, before.st_ino)
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (after.st_dev, after.st_ino) != original_identity
        or (canonical_metadata.st_dev, canonical_metadata.st_ino) != original_identity
    ):
        raise ValueError(f"Identity path changed while binding: {lexical}")
    return canonical, size, digest


def verify_runtime_image(identity: ImageIdentity) -> None:
    """Rehash an image immediately before governed use."""
    if identity.format_version != 1 or identity.policy not in {"digest-checked", "trusted-cache"}:
        raise ValueError("Unsupported ImageIdentity")
    size, digest, _mode = _stable_regular_identity(identity.path)
    if (size, digest) != (identity.size_bytes, identity.sha256):
        raise ValueError("Runtime image identity mismatch")


def prepare_governed_sources(
    source: SourcePackageIdentity,
    toolkit: SourcePackageIdentity,
    *,
    source_destination: Path,
    toolkit_destination: Path,
) -> None:
    """Verify/extract package and privately copy toolkit before imports are enabled."""
    verify_source_package(source, destination=source_destination, expected_role="orchestration")
    verify_source_package(toolkit, destination=toolkit_destination, expected_role="toolkit")


def execute_governed(
    *,
    source: SourcePackageIdentity,
    toolkit: SourcePackageIdentity,
    image: ImageIdentity,
    source_destination: Path,
    toolkit_destination: Path,
    runtime_entrypoint: Callable[[], None],
) -> None:
    """Verify every identity before transferring control to runtime imports."""
    verify_runtime_image(image)
    prepare_governed_sources(
        source,
        toolkit,
        source_destination=source_destination,
        toolkit_destination=toolkit_destination,
    )
    runtime_entrypoint()


def qualification_smoke(record_path: Path, *, tuple_id: str) -> None:
    """Run the minimal GPU smoke and complete its pre-created qualification record."""
    subprocess.run(("nvidia-smi",), check=True, env={"PATH": "/usr/local/bin:/usr/bin:/bin"})
    payload = json.loads(record_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Runtime Qualification record must be a JSON object")
    payload.update(
        {
            "tuple_id": tuple_id,
            "status": "succeeded",
            "evidence_status": "complete",
            "qualified_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{record_path.name}.", suffix=".tmp", dir=record_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, record_path)
        directory = os.open(record_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_governed_bootstrap(
    *,
    identity_record: Path,
    source_destination: Path | None,
    toolkit_destination: Path | None,
    exec_argv_json: str,
    expected_image_sha256: str | None = None,
    expected_identity_record_sha256: str | None = None,
    source_package: Path | None = None,
    toolkit_package: Path | None = None,
    runtime_ipsae_binary: Path | None = None,
    expected_runtime_ipsae_sha256: str | None = None,
) -> None:
    """Verify the bound release inputs and replace this process with exact argv."""
    identity_bytes, _size, identity_sha256, _mode = _stable_regular_read(identity_record)
    if expected_identity_record_sha256 is not None and identity_sha256 != expected_identity_record_sha256:
        raise ValueError("governed identity record SHA-256 does not match submission expectation")
    payload = json.loads(identity_bytes)
    if not isinstance(payload, dict):
        raise ValueError("Governed identity record must be a JSON object")
    tuple_payload = payload.get("tuple", payload)
    if not isinstance(tuple_payload, dict):
        raise ValueError("Governed identity tuple must be a JSON object")
    source_value = tuple_payload.get("source_package_identity")
    toolkit_value = tuple_payload.get("toolkit_package_identity")
    image_value = tuple_payload.get("image_identity")
    selected_source = tuple_payload.get("selected_source")
    is_baked = isinstance(selected_source, dict) and selected_source.get("source_kind") == "baked"
    if not isinstance(source_value, dict) or not isinstance(image_value, dict):
        raise ValueError("Governed identity record is missing source or image identity")
    if not is_baked and not isinstance(toolkit_value, dict):
        raise ValueError("Governed identity record is missing toolkit identity")
    source = source_package_identity_from_mapping(source_value)
    toolkit: SourcePackageIdentity | None = None
    if toolkit_value is not None and isinstance(toolkit_value, dict):
        toolkit = source_package_identity_from_mapping(toolkit_value)
    if source_package is not None:
        source = SourcePackageIdentity(**{**source.__dict__, "package_path": source_package})
    if toolkit_package is not None and toolkit is not None:
        toolkit = SourcePackageIdentity(**{**toolkit.__dict__, "package_path": toolkit_package})
    image = image_identity_from_mapping(image_value)
    if expected_image_sha256 is not None and image.sha256 != expected_image_sha256:
        raise ValueError("Runtime image expectation does not match governed identity record")
    raw_argv = json.loads(exec_argv_json)
    if not isinstance(raw_argv, list) or not raw_argv or any(not isinstance(value, str) for value in raw_argv):
        raise ValueError("Governed exec argv must be a non-empty JSON string list")
    scratch_root: Path | None = None
    if is_baked and toolkit_destination is None:
        toolkit_destination = Path("/opt/afdb-toolkit")
    if source_destination is None or toolkit_destination is None:
        scratch_parent = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
        scratch_root = Path(tempfile.mkdtemp(prefix="bspp-governed-", dir=scratch_parent))
        scratch_root.chmod(0o700)
        if source_destination is None:
            source_destination = scratch_root / "source"
        if toolkit_destination is None:
            toolkit_destination = scratch_root / "toolkit"
    exec_argv = tuple(
        value.replace("{BSPP_SOURCE_ROOT}", str(source_destination)).replace(
            "{BSPP_TOOLKIT_ROOT}", str(toolkit_destination)
        )
        for value in raw_argv
    )
    executable = Path(exec_argv[0])
    if not executable.is_absolute():
        raise ValueError("Governed executable must be an absolute fixed image path")

    def transfer_control() -> None:
        os.execve(str(executable), exec_argv, _scrubbed_environment(scratch_root or source_destination.parent))

    try:
        if is_baked:
            verify_source_package(source, destination=source_destination, expected_role="orchestration")
        else:
            assert toolkit is not None
            prepare_governed_sources(
                source,
                toolkit,
                source_destination=source_destination,
                toolkit_destination=toolkit_destination,
            )
        if runtime_ipsae_binary is not None or expected_runtime_ipsae_sha256 is not None:
            if runtime_ipsae_binary is None or expected_runtime_ipsae_sha256 is None:
                raise ValueError("runtime iPSAE binary and SHA-256 must be supplied together")
            smoke = payload.get("smoke_evidence")
            runtime_ipsae_value = smoke.get("runtime_ipsae") if isinstance(smoke, dict) else None
            if not isinstance(runtime_ipsae_value, dict):
                raise ValueError("qualification record lacks runtime-built iPSAE evidence")
            runtime_ipsae = runtime_ipsae_evidence_from_mapping(runtime_ipsae_value)
            if is_baked:
                # Baked mode: the baked commit is file-authoritative — read it
                # from provenance.json (the image's own baked provenance) and
                # fail closed on mismatch, exactly as before; only the source of
                # truth moved (provenance.json, not the tuple placeholder).
                _check_baked_runtime_ipsae_revision(toolkit_destination, runtime_ipsae.source_revision)
            elif toolkit is not None and runtime_ipsae.source_revision != toolkit.commit:
                raise ValueError("runtime-built iPSAE source revision does not match toolkit package")
            if runtime_ipsae.binary.sha256 != expected_runtime_ipsae_sha256:
                raise ValueError("runtime-built iPSAE expectation does not match qualification evidence")
            _install_runtime_ipsae_binary(
                runtime_ipsae_binary,
                toolkit_destination / runtime_ipsae.source_path / "ipsae_cpp",
                expected_sha256=runtime_ipsae.binary.sha256,
                expected_size=runtime_ipsae.binary.size_bytes,
            )
    except Exception:
        if scratch_root is not None:
            shutil.rmtree(scratch_root, ignore_errors=True)
        raise
    transfer_control()


def _check_baked_runtime_ipsae_revision(toolkit_root: Path, source_revision: str) -> None:
    """Fail closed when the runtime-built iPSAE revision does not match the
    file-authoritative baked provenance commit.

    The baked commit's source of truth is ``provenance.json`` (written by the
    Dockerfile from ``TOOLKIT_REF`` and bound by the image sha256), not the
    qualification tuple's public placeholder. The invariant — the runtime iPSAE
    must be built from the same revision the image baked — is unchanged.
    """
    provenance_path = toolkit_root / "provenance.json"
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise ValueError("baked toolkit provenance missing or unsafe for iPSAE revision check")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"baked toolkit provenance unreadable for iPSAE revision check: {exc}") from exc
    baked_revision = provenance.get("commit") if isinstance(provenance, dict) else None
    if (
        not isinstance(baked_revision, str)
        or len(baked_revision) != 40
        or any(c not in "0123456789abcdef" for c in baked_revision)
    ):
        raise ValueError("baked toolkit provenance commit is not a 40-hex sha for iPSAE revision check")
    if source_revision != baked_revision:
        raise ValueError("runtime-built iPSAE source revision does not match baked toolkit provenance")


def _install_runtime_ipsae_binary(source: Path, destination: Path, *, expected_sha256: str, expected_size: int) -> None:
    """Verify and install the exact qualified binary into the private toolkit copy."""
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    temporary = destination.with_name(f".{destination.name}.qualified")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise ValueError("runtime-built iPSAE binary size or type mismatch")
        digest = hashlib.sha256()
        destination.parent.mkdir(parents=True, exist_ok=True)
        output = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o500)
        try:
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
                _write_all(output, chunk)
            os.fsync(output)
        finally:
            os.close(output)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or digest.hexdigest() != expected_sha256:
            raise ValueError("runtime-built iPSAE binary identity mismatch")
        temporary_identity = _verify_runtime_ipsae_temporary(
            temporary,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
        os.replace(temporary, destination)
        installed = destination.lstat()
        if (
            not stat.S_ISREG(installed.st_mode)
            or stat.S_IMODE(installed.st_mode) != 0o500
            or (installed.st_dev, installed.st_ino, installed.st_size) != temporary_identity[:3]
        ):
            raise ValueError("installed runtime-built iPSAE binary identity mismatch")
        parent_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise ValueError("runtime-built iPSAE binary write stalled")
        remaining = remaining[written:]


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _verify_runtime_ipsae_temporary(
    path: Path, *, expected_sha256: str, expected_size: int
) -> tuple[int, int, int, int, int]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o500 or before.st_size != expected_size:
            raise ValueError("runtime-built iPSAE temporary identity mismatch")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        current = path.lstat()
        if (
            _stat_signature(before) != _stat_signature(after)
            or _stat_signature(after) != _stat_signature(current)
            or size != expected_size
            or digest.hexdigest() != expected_sha256
        ):
            raise ValueError("runtime-built iPSAE temporary identity mismatch")
        return _stat_signature(after)
    finally:
        os.close(descriptor)


def _scrubbed_environment(scratch_root: Path) -> dict[str, str]:
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(scratch_root / "home"),
    }
    for name in (
        "BSPP_RUNSPEC",
        "CUDA_VISIBLE_DEVICES",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_ID",
        "SLURM_TMPDIR",
    ):
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    return environment


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch the fixed bootstrap operations available inside the image."""
    parser = argparse.ArgumentParser(prog="bspp-governed-bootstrap")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--tuple-id", required=True)
    qualify.add_argument("--record", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--identity-record", type=Path, required=True)
    run.add_argument("--source-destination", type=Path)
    run.add_argument("--toolkit-destination", type=Path)
    run.add_argument("--exec-argv-json", required=True)
    run.add_argument("--expected-image-sha256")
    run.add_argument("--expected-identity-record-sha256")
    run.add_argument("--source-package", type=Path)
    run.add_argument("--toolkit-package", type=Path)
    run.add_argument("--runtime-ipsae-binary", type=Path)
    run.add_argument("--expected-runtime-ipsae-sha256")
    arguments = parser.parse_args(argv)
    if arguments.operation == "qualify":
        qualification_smoke(arguments.record, tuple_id=arguments.tuple_id)
    elif arguments.operation == "run":
        run_governed_bootstrap(
            identity_record=arguments.identity_record,
            source_destination=arguments.source_destination,
            toolkit_destination=arguments.toolkit_destination,
            exec_argv_json=arguments.exec_argv_json,
            expected_image_sha256=arguments.expected_image_sha256,
            expected_identity_record_sha256=arguments.expected_identity_record_sha256,
            source_package=arguments.source_package,
            toolkit_package=arguments.toolkit_package,
            runtime_ipsae_binary=arguments.runtime_ipsae_binary,
            expected_runtime_ipsae_sha256=arguments.expected_runtime_ipsae_sha256,
        )


def render_governed_srun(
    image: ImageIdentity,
    *,
    mounts: str,
    bootstrap_args: tuple[str, ...],
    slurm_environment_contract_version: int = 1,
    legacy: bool = False,
) -> str:
    """Render a host precheck followed by isolated, direct image Python."""
    # The bootstrap imports the contract package, which transitively imports
    # yaml from the pixi site-packages, so -S (no site-packages) must NOT be
    # passed here. -I still ignores ambient PYTHON* env and user site.
    pixi_python = "/opt/afcdb-orchestration-env/.pixi/envs/default/bin/python" if legacy else str(PIXI_PYTHON_PATH)
    bootstrap_py = "/opt/afcdb/execution_bootstrap.py" if legacy else "/opt/bspp/execution_bootstrap.py"
    home_suffix = "afcdb-home" if legacy else "bspp-home"
    python_argv = (
        pixi_python,
        "-I",
        bootstrap_py,
        *bootstrap_args,
    )
    slurm_contract = slurm_environment_contract(slurm_environment_contract_version)
    allowlist = (
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "LANG=C.UTF-8",
        "LC_ALL=C.UTF-8",
        f"HOME=${{SLURM_TMPDIR:-/tmp}}/{home_suffix}",
        "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}",
        "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}",
        slurm_contract.job_id_assignment,
        slurm_contract.array_task_id_assignment,
        "SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-}",
        "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}",
        "SLURM_TMPDIR=${SLURM_TMPDIR:-}",
    )
    srun_prefix = " ".join(
        shlex.quote(part)
        for part in (
            "srun",
            f"--container-image={image.path}",
            f"--container-mounts={mounts}",
            "--no-container-mount-home",
        )
    )

    # Keep the complete closed allowlist raw, exactly as historic renderers did:
    # Slurm and CUDA assignments must expand in the outer batch shell.  Contract
    # 2 changes only the two Slurm identity strings selected above.
    isolated_command = " ".join(("env -i", *allowlist, *(shlex.quote(part) for part in python_argv)))
    verifier_command = " ".join(
        shlex.quote(part)
        for part in (
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            f"import base64;exec(base64.b64decode({base64.b64encode(_HOST_IMAGE_VERIFIER.encode()).decode()!r}))",
            str(image.path),
            str(image.size_bytes),
            image.sha256,
        )
    )
    return "\n".join(
        (
            verifier_command,
            f"{srun_prefix} {isolated_command}",
        )
    )


def _require_lower_hex(value: str, length: int, name: str) -> None:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")


def _stable_regular_identity(path: Path) -> tuple[int, str, str]:
    _data, size, digest, mode = _stable_regular_read(path)
    return size, digest, mode


def _stable_regular_read(path: Path) -> tuple[bytes, int, str, str]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError(f"Identity path is unsafe, unavailable, or a symlink: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Identity path is not a regular file: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)

        def signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

        if signature(before) != signature(after):
            raise ValueError(f"Identity path mutated while reading: {path}")
        mode = "0o755" if before.st_mode & 0o111 else "0o644"
        return b"".join(chunks), before.st_size, digest.hexdigest(), mode
    finally:
        os.close(descriptor)


__all__ = [
    "ImageIdentity",
    "execute_governed",
    "identify_runtime_image",
    "image_identity_from_mapping",
    "main",
    "prepare_governed_sources",
    "qualification_smoke",
    "render_governed_srun",
    "run_governed_bootstrap",
    "verify_runtime_image",
]


if __name__ == "__main__":
    main()
