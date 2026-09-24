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

"""Shared native worker container runtime contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.runspec import MountSpec, RunSpec
from bspp.orchestration.runtime.toolkit import (
    resolve_toolkit_for_container,
)

RUNSPEC_CONTAINER_PATH = Path("/workspace/bspp-runspec/runspec.yaml")
PIXI_PYTHON_PATH = Path("/opt/bspp-orchestration-env/.pixi/envs/default/bin/python")
CONTROL_CONTRACT_SOURCE_RELATIVE = Path("packages/orchestration-contract/src")
RUNTIME_SOURCE_RELATIVE = Path("packages/orchestration-runtime/src")


@dataclass(frozen=True)
class NativeRuntimeContract:
    """Container paths and tools required by the native archive worker."""

    runspec_container_path: Path
    toolkit_container_path: Path
    orchestration_container_path: Path
    container_workdir: Path
    pixi_python_path: Path
    pythonpath_entries: tuple[Path, ...]
    required_tools: tuple[str, ...]
    production_pipeline_container_path: Path
    s5cmd_required: bool
    zstd_required: bool
    toolkit_source_mode: str | None = None
    toolkit_provenance_commit: str | None = None

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable runtime contract data."""
        result: dict[str, object] = {
            "runspec_container_path": str(self.runspec_container_path),
            "toolkit_container_path": str(self.toolkit_container_path),
            "orchestration_container_path": str(self.orchestration_container_path),
            "container_workdir": str(self.container_workdir),
            "pixi_python_path": str(self.pixi_python_path),
            "pythonpath_entries": [str(path) for path in self.pythonpath_entries],
            "required_tools": list(self.required_tools),
            "production_pipeline_container_path": str(self.production_pipeline_container_path),
            "s5cmd_required": self.s5cmd_required,
            "zstd_required": self.zstd_required,
        }
        if self.toolkit_source_mode is not None:
            result["toolkit_source_mode"] = self.toolkit_source_mode
        if self.toolkit_provenance_commit is not None:
            result["toolkit_provenance_commit"] = self.toolkit_provenance_commit
        return result


def build_native_runtime_contract(spec: RunSpec) -> NativeRuntimeContract:
    """Build the native worker's expected in-container runtime contract."""
    runspec_path = runspec_container_path(spec)
    toolkit_source = resolve_toolkit_for_container(spec)
    toolkit_root = toolkit_source.root
    orchestration_root = container_path_for(spec, spec.paths.orchestration_repo) or spec.container.workdir
    s5cmd_required = requires_s5cmd(spec)
    zstd_required = requires_zstd_members(spec)
    required_tools = ["tar", "lz4"]
    if zstd_required:
        required_tools.append("zstd")
    if s5cmd_required:
        required_tools.append(str(spec.worker.s5cmd_path))
    return NativeRuntimeContract(
        runspec_container_path=runspec_path,
        toolkit_container_path=toolkit_root,
        orchestration_container_path=orchestration_root,
        container_workdir=spec.container.workdir,
        pixi_python_path=PIXI_PYTHON_PATH,
        pythonpath_entries=(
            toolkit_root,
            orchestration_root / CONTROL_CONTRACT_SOURCE_RELATIVE,
            orchestration_root / RUNTIME_SOURCE_RELATIVE,
        ),
        required_tools=tuple(required_tools),
        production_pipeline_container_path=toolkit_source.production_pipeline,
        s5cmd_required=s5cmd_required,
        zstd_required=zstd_required,
        toolkit_source_mode=toolkit_source.source,
        toolkit_provenance_commit=toolkit_source.provenance_commit,
    )


def requires_s5cmd(spec: RunSpec) -> bool:
    """Return whether native runtime must have the configured ``s5cmd``."""
    return spec.worker.self_upload


def requires_zstd_members(spec: RunSpec) -> bool:
    """Return whether native runtime must have the ``zstd`` CLI."""
    return _optional_value(spec.storage, "tar_compression", "s3_tar_compression") == "zstd-members"


def all_container_mounts(spec: RunSpec, runspec_path: Path) -> tuple[MountSpec, ...]:
    """Return native worker mounts, including implicit Lustre and RunSpec mounts."""
    mounts: list[MountSpec] = [MountSpec(source=Path("/lustre"), target=Path("/lustre"), read_only=False)]
    mounts.extend(spec.container.mounts)
    if spec.source_path is not None and container_path_for_mounts(mounts, spec.source_path) is None:
        mounts.append(MountSpec(source=spec.source_path, target=runspec_path, read_only=True))
    return tuple(mounts)


def runspec_container_path(spec: RunSpec) -> Path:
    """Return where the RunSpec will be visible inside the container."""
    if spec.source_path is None:
        return RUNSPEC_CONTAINER_PATH
    mounts = _translation_mounts(spec)
    return container_path_for_mounts(mounts, spec.source_path) or RUNSPEC_CONTAINER_PATH


def container_path_for(spec: RunSpec, host_path: Path) -> Path | None:
    """Translate a host path through native worker container mounts."""
    return container_path_for_mounts(_translation_mounts(spec), host_path)


def _translation_mounts(spec: RunSpec) -> list[MountSpec]:
    """Return mounts ordered for host-to-container path translation.

    Explicit source mounts must win over the broad implicit Lustre mount, or
    source paths under /lustre resolve to their host path instead of the
    intended /workspace editable mount.
    """
    mounts = list(spec.container.mounts)
    mounts.append(MountSpec(source=Path("/lustre"), target=Path("/lustre"), read_only=False))
    return mounts


def container_path_for_mounts(mounts: list[MountSpec], host_path: Path) -> Path | None:
    """Translate a host path through an explicit mount table."""
    for mount in mounts:
        try:
            relative = host_path.relative_to(mount.source)
        except ValueError:
            continue
        return mount.target / relative
    return None


def _optional_value(value: object, *names: str) -> object | None:
    for name in names:
        if isinstance(value, dict) and name in value:
            result: object = value[name]
            return result
        raw: object | None = getattr(value, name, None)
        if raw is not None:
            return raw
    return None


__all__ = [
    "CONTROL_CONTRACT_SOURCE_RELATIVE",
    "PIXI_PYTHON_PATH",
    "RUNSPEC_CONTAINER_PATH",
    "RUNTIME_SOURCE_RELATIVE",
    "NativeRuntimeContract",
    "all_container_mounts",
    "build_native_runtime_contract",
    "container_path_for",
    "container_path_for_mounts",
    "requires_s5cmd",
    "requires_zstd_members",
    "runspec_container_path",
]
