#!/opt/bspp/environment/bin/python
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

"""Validate and record the CUDA user-mode driver selected in the container."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CUDA_SONAME = "libcuda.so.1"
CUDA_COMPAT_DIR = Path("/usr/local/cuda-12.6/compat")
PROBE_SOURCE = Path("/opt/bspp/bin/bspp-preprocessing-cuda-driver-probe")
HELPER_SOURCE = Path("/opt/bspp/bin/bspp-preprocessing-carry-characterization")


@dataclass(frozen=True)
class LoaderObservation:
    soname: str
    compat_directory: str
    compat_soname_path: str
    compat_library_resolved_target: str
    mapped_library_path: str
    mapped_library_resolved_target: str
    maps_line: str


@dataclass(frozen=True)
class CudaObservation:
    driver_api_version: int
    device_count: int
    cu_driver_get_version_return_code: int
    cu_init_return_code: int
    cu_device_get_count_return_code: int


def validate_mapped_library(
    *, mapped_path: Path, soname: str = CUDA_SONAME, required_directory: Path = CUDA_COMPAT_DIR
) -> LoaderObservation:
    """Validate a mapped library against the pinned soname and containment policy."""
    try:
        directory_target = required_directory.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"CUDA compatibility directory cannot be resolved: {required_directory}") from exc
    if not directory_target.is_dir() or not os.access(directory_target, os.R_OK | os.X_OK):
        raise RuntimeError(f"CUDA compatibility directory is not readable: {required_directory}")

    soname_path = required_directory / soname
    if not soname_path.is_symlink():
        raise RuntimeError(f"CUDA compatibility soname is not a symlink: {soname_path}")
    try:
        compat_target = soname_path.resolve(strict=True)
        mapped_target = mapped_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("CUDA compatibility library target cannot be resolved") from exc
    for label, target in (("compatibility", compat_target), ("mapped", mapped_target)):
        if not target.is_file() or not os.access(target, os.R_OK):
            raise RuntimeError(f"{label} CUDA library target is not a readable regular file: {target}")
        if not target.is_relative_to(directory_target):
            raise RuntimeError(f"{label} CUDA library target escapes {directory_target}: {target}")
    if mapped_target != compat_target:
        raise RuntimeError(f"mapped CUDA library does not match the pinned soname target: {mapped_target}")
    return LoaderObservation(
        soname=soname,
        compat_directory=str(required_directory),
        compat_soname_path=str(soname_path),
        compat_library_resolved_target=str(compat_target),
        mapped_library_path=str(mapped_path),
        mapped_library_resolved_target=str(mapped_target),
        maps_line="",
    )


def observe_loader(
    *, soname: str = CUDA_SONAME, required_directory: Path = CUDA_COMPAT_DIR
) -> tuple[ctypes.CDLL, LoaderObservation]:
    """Dlopen the soname and prove which compatibility-library file was mapped."""
    library = ctypes.CDLL(soname)
    library_prefix = soname.split(".so", 1)[0] + ".so"
    candidates: list[tuple[Path, str]] = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        mapped_path = Path(fields[5])
        if mapped_path.name.startswith(library_prefix):
            candidates.append((mapped_path, line))
    if not candidates:
        raise RuntimeError(f"dlopen succeeded but {soname} was not found in /proc/self/maps")

    failures: list[str] = []
    for mapped_path, maps_line in candidates:
        try:
            observation = validate_mapped_library(
                mapped_path=mapped_path, soname=soname, required_directory=required_directory
            )
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        return library, LoaderObservation(**{**observation.__dict__, "maps_line": maps_line})
    raise RuntimeError(f"no mapped {soname} satisfies the pinned compatibility policy: {'; '.join(failures)}")


def observe_cuda(library: ctypes.CDLL) -> CudaObservation:
    """Call the minimum CUDA Driver API needed for real-device characterization."""
    library.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.cuDriverGetVersion.restype = ctypes.c_int
    library.cuInit.argtypes = [ctypes.c_uint]
    library.cuInit.restype = ctypes.c_int
    library.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.cuDeviceGetCount.restype = ctypes.c_int

    driver_api_version = ctypes.c_int()
    get_version_rc = int(library.cuDriverGetVersion(ctypes.byref(driver_api_version)))
    if get_version_rc != 0:
        raise RuntimeError(f"cuDriverGetVersion failed with return code {get_version_rc}")
    init_rc = int(library.cuInit(0))
    if init_rc != 0:
        raise RuntimeError(f"cuInit failed with return code {init_rc}")
    device_count = ctypes.c_int()
    get_count_rc = int(library.cuDeviceGetCount(ctypes.byref(device_count)))
    if get_count_rc != 0:
        raise RuntimeError(f"cuDeviceGetCount failed with return code {get_count_rc}")
    if driver_api_version.value <= 0 or device_count.value <= 0:
        raise RuntimeError("CUDA Driver API returned an invalid version or no visible device")
    return CudaObservation(
        driver_api_version=driver_api_version.value,
        device_count=device_count.value,
        cu_driver_get_version_return_code=get_version_rc,
        cu_init_return_code=init_rc,
        cu_device_get_count_return_code=get_count_rc,
    )


def build_evidence_payload(
    *,
    loader: LoaderObservation,
    cuda: CudaObservation,
    cuda_visible_devices: str,
    ld_library_path: str,
    slurm_job_id: str,
    slurmd_nodename: str,
    probe_sha256: str,
    helper_sha256: str,
) -> dict[str, Any]:
    """Build the persisted probe payload from already observed facts."""
    return {
        "schema_version": 1,
        "loader_soname": loader.soname,
        "compat_directory": loader.compat_directory,
        "compat_soname_path": loader.compat_soname_path,
        "compat_library_resolved_target": loader.compat_library_resolved_target,
        "mapped_library_path": loader.mapped_library_path,
        "mapped_library_resolved_target": loader.mapped_library_resolved_target,
        "maps_line": loader.maps_line,
        "driver_api_version": cuda.driver_api_version,
        "cuda_device_count": cuda.device_count,
        "cu_driver_get_version_return_code": cuda.cu_driver_get_version_return_code,
        "cu_init_return_code": cuda.cu_init_return_code,
        "cu_device_get_count_return_code": cuda.cu_device_get_count_return_code,
        "cuda_visible_devices": cuda_visible_devices,
        "effective_ld_library_path": ld_library_path,
        "slurm_job_id": slurm_job_id,
        "slurmd_nodename": slurmd_nodename,
        "probe_sha256": probe_sha256,
        "helper_sha256": helper_sha256,
    }


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"missing nonblank environment value: {name}")
    return value


def _visible_device_count(value: str) -> int:
    devices = tuple(part.strip() for part in value.split(","))
    if not devices or any(not part for part in devices):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be a nonblank comma-separated list")
    return len(devices)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    library, loader = observe_loader()
    cuda = observe_cuda(library)
    cuda_visible_devices = _required_environment("CUDA_VISIBLE_DEVICES")
    if cuda.device_count != _visible_device_count(cuda_visible_devices):
        raise RuntimeError("CUDA Driver API device count does not match CUDA_VISIBLE_DEVICES cardinality")
    payload = build_evidence_payload(
        loader=loader,
        cuda=cuda,
        cuda_visible_devices=cuda_visible_devices,
        ld_library_path=_required_environment("LD_LIBRARY_PATH"),
        slurm_job_id=_required_environment("SLURM_JOB_ID"),
        slurmd_nodename=_required_environment("SLURMD_NODENAME"),
        probe_sha256=_sha256(PROBE_SOURCE),
        helper_sha256=_sha256(HELPER_SOURCE),
    )
    _write_atomic(args.destination, payload)


if __name__ == "__main__":
    main()
