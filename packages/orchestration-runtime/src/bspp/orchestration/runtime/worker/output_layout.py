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

"""Pure output suffixes, destination paths, and transfer-pair planning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TransferPair:
    """One local source and destination object key/URI for file-mode uploads."""

    source: Path
    destination: str

    def s5cmd_cp_args(self) -> tuple[str, str, str]:
        """Return argv-style arguments for an s5cmd copy operation."""
        return ("cp", str(self.source), self.destination)


def _require_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{name} must be an integer, got {value!r}"
        raise ValueError(msg)


def _require_non_negative_int(name: str, value: int) -> None:
    _require_int(name, value)
    if value < 0:
        msg = f"{name} must be non-negative, got {value}"
        raise ValueError(msg)


def _require_positive_int(name: str, value: int) -> None:
    _require_int(name, value)
    if value <= 0:
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


def shard_dir(output_root: Path, logical_shard_id: int) -> Path:
    """Return the local output directory for a zero-based logical shard ID."""
    _require_non_negative_int("logical_shard_id", logical_shard_id)
    return output_root / f"shard_{logical_shard_id}"


def success_outputs_dir(shard_path: Path) -> Path:
    """Return the production pipeline success output directory for a shard."""
    return shard_path / "success_outputs"


def uploaded_marker_path(shard_path: Path) -> Path:
    """Return the shard self-upload completion marker path."""
    return shard_path / ".uploaded"


def batch_done_marker_path(shard_path: Path, batch_id: int) -> Path:
    """Return the successful batch marker path for a shard-local batch."""
    _require_non_negative_int("batch_id", batch_id)
    return shard_path / f".batch_{batch_id}_done"


def metadata_search_filename(logical_shard_id: int, total_shards: int, dataset_tag: str | None = None) -> str:
    """Return the WP8a-compatible search metadata filename for a shard."""
    return _metadata_filename("AF-metadata", logical_shard_id, total_shards, dataset_tag)


def metadata_collection_filename(logical_shard_id: int, total_shards: int, dataset_tag: str | None = None) -> str:
    """Return the WP8a-compatible chain collection metadata filename for a shard."""
    return _metadata_filename("AF-chain-metadata", logical_shard_id, total_shards, dataset_tag)


def metadata_search_relative_path(logical_shard_id: int, total_shards: int, dataset_tag: str | None = None) -> Path:
    """Return the success_outputs-relative path for search metadata."""
    return Path("metadata") / "search" / metadata_search_filename(logical_shard_id, total_shards, dataset_tag)


def metadata_collection_relative_path(logical_shard_id: int, total_shards: int, dataset_tag: str | None = None) -> Path:
    """Return the success_outputs-relative path for chain collection metadata."""
    return (
        Path("metadata")
        / "collection"
        / metadata_collection_filename(
            logical_shard_id,
            total_shards,
            dataset_tag,
        )
    )


def _metadata_filename(prefix: str, logical_shard_id: int, total_shards: int, dataset_tag: str | None) -> str:
    _require_non_negative_int("logical_shard_id", logical_shard_id)
    _require_positive_int("total_shards", total_shards)
    if logical_shard_id >= total_shards:
        msg = f"logical_shard_id {logical_shard_id} is outside shard range 0..{total_shards - 1}"
        raise IndexError(msg)
    tag_suffix = f"-{dataset_tag}" if dataset_tag else ""
    return f"{prefix}-{logical_shard_id + 1}-of-{total_shards}{tag_suffix}.json"


def plan_file_upload_transfers(
    success_outputs_path: Path,
    relative_output_files: Iterable[str | Path],
    *,
    prefix: str,
) -> tuple[TransferPair, ...]:
    """Return source-to-destination pairs for flat file-mode uploads.

    ``prefix`` is intentionally required by the caller; this helper has no
    production bucket or dataset default.
    """
    return tuple(
        TransferPair(
            source=success_outputs_path / relative_path,
            destination=join_destination_prefix(prefix, relative_path.as_posix()),
        )
        for relative_path in (_validate_relative_output_file(path) for path in relative_output_files)
    )


def join_destination_prefix(prefix: str, relative_key: str) -> str:
    """Join an explicit destination prefix and relative object key."""
    if not isinstance(prefix, str):
        msg = f"prefix must be a string, got {prefix!r}"
        raise ValueError(msg)
    if "\\" in relative_key or relative_key.startswith("/") or relative_key == "" or ".." in Path(relative_key).parts:
        msg = (
            f"relative_key must be a non-empty relative path without '..' or backslash separators, got {relative_key!r}"
        )
        raise ValueError(msg)
    return f"{prefix.rstrip('/')}/{relative_key.lstrip('/')}" if prefix else relative_key.lstrip("/")


def batch_tar_name(logical_shard_id: int, batch_id: int, compression: str | None = None) -> str:
    """Return the local tar name for one shard batch."""
    _require_non_negative_int("logical_shard_id", logical_shard_id)
    _require_non_negative_int("batch_id", batch_id)
    return f"shard_{logical_shard_id}_batch_{batch_id}{tar_suffix(compression)}"


def metadata_tar_name(logical_shard_id: int, compression: str | None = None) -> str:
    """Return the local tar name for one shard's metadata bundle."""
    _require_non_negative_int("logical_shard_id", logical_shard_id)
    return f"shard_{logical_shard_id}_metadata{tar_suffix(compression)}"


def tar_suffix(compression: str | None = None) -> str:
    """Return the archive filename suffix implied by tar-level compression."""
    if compression in (None, "", "none", "zstd-members"):
        return ".tar"
    if compression in ("gz", "gzip", "tar.gz"):
        return ".tar.gz"
    msg = f"unsupported tar compression: {compression!r}"
    raise ValueError(msg)


def _validate_relative_output_file(path: str | Path) -> Path:
    if "\\" in str(path):
        msg = f"output file must use '/' separators and cannot contain backslashes, got {path!r}"
        raise ValueError(msg)
    relative_path = Path(path)
    if relative_path.is_absolute() or relative_path == Path(".") or ".." in relative_path.parts:
        msg = f"output file must be relative to success_outputs without '..', got {path!r}"
        raise ValueError(msg)
    return relative_path


__all__ = [
    "TransferPair",
    "batch_done_marker_path",
    "batch_tar_name",
    "join_destination_prefix",
    "metadata_collection_filename",
    "metadata_collection_relative_path",
    "metadata_search_filename",
    "metadata_search_relative_path",
    "metadata_tar_name",
    "plan_file_upload_transfers",
    "shard_dir",
    "success_outputs_dir",
    "tar_suffix",
    "uploaded_marker_path",
]
