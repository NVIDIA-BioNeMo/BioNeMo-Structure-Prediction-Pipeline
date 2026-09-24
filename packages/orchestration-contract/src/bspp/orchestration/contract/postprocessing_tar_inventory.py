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

"""Shared resolution semantics for postprocessing local-tar manifest rows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlparse


def resolve_tar_manifest_member(
    row: Mapping[str, str],
    *,
    manifest_path: Path,
    local_tar_dir: Path | None = None,
) -> tuple[Path, str]:
    """Resolve maintained tar_path/file-URI/tar_name manifest field shapes."""
    tar_path_text = row.get("tar_path", "").strip()
    manifest_dir = manifest_path.parent
    if tar_path_text:
        tar_path = Path(tar_path_text)
        resolved = tar_path if tar_path.is_absolute() else manifest_dir / tar_path
        return resolved, _relative_tar_path(resolved, manifest_dir)

    file_uri_path = _path_from_file_uri(row.get("s3_uri", "").strip())
    if file_uri_path is not None:
        return file_uri_path, _relative_tar_path(file_uri_path, manifest_dir)

    tar_name = row.get("tar_name", "").strip()
    if not tar_name:
        raise ValueError(f"local tar manifest row has neither tar_path nor tar_name in {manifest_path}")
    tar_root = local_tar_dir or manifest_dir / "local_tars"
    candidates = tar_name_candidates(tar_root, tar_name, row)
    for candidate in candidates:
        if candidate.exists():
            return candidate, _relative_tar_path(candidate, manifest_dir)
    raise FileNotFoundError(candidates[0])


def tar_name_candidates(tar_root: Path, tar_name: str, row: Mapping[str, str]) -> tuple[Path, ...]:
    tar_type = row.get("tar_type", "").strip()
    shard_id = row.get("shard_id", "").strip()
    shard_path = tar_root / f"shard_{shard_id}" / tar_name if shard_id else None
    metadata_path = tar_root / "metadata" / tar_name
    flat_path = tar_root / tar_name
    candidates: list[Path] = []
    if tar_type == "metadata":
        candidates.append(metadata_path)
    if shard_path is not None:
        candidates.append(shard_path)
    candidates.append(flat_path)
    if tar_type != "metadata":
        candidates.append(metadata_path)
    return tuple(dict.fromkeys(candidates))


def _relative_tar_path(tar_path: Path, root: Path) -> str:
    try:
        return tar_path.relative_to(root).as_posix()
    except ValueError:
        return str(tar_path)


def _path_from_file_uri(value: str) -> Path | None:
    if not value.startswith("file://"):
        return None
    parsed = urlparse(value)
    if parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"unsupported file URI host in local tar manifest: {value}")
    return Path(unquote(parsed.path))


__all__ = ["resolve_tar_manifest_member", "tar_name_candidates"]
