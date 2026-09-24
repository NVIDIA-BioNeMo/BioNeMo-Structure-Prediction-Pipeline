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

"""Native analysis metadata finalization helpers.

This module intentionally has no dependency on AFDB-Integration-Kit.  It covers
the local finalization artifacts that the orchestration layer can produce from
native worker outputs:

* ``analysis_metadata.parquet`` from ``analysis_metadata.csv``
* ``high_quality_model_ids.txt`` from the CSV quality flag
* optional ``model_tar_index.csv`` from local tar manifests and tar members
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

MODEL_ID_COLUMN = "model_id"
QUALITY_COLUMN = "passes_quality_threshold"
REQUIRED_ANALYSIS_COLUMNS = (
    "model_id",
    "original_id",
    "upload_status",
    "passes_quality_threshold",
    "failure_reason",
)
NON_OUTPUT_UPLOAD_STATUSES = frozenset({"model_failed", "pipeline_failed", "input_failed", "input_validation_failed"})
DEFAULT_PARQUET_BLOCK_SIZE_BYTES = 1 << 20

__all__ = [
    "AnalysisFinalizerResult",
    "build_model_tar_index",
    "extract_high_quality_from_tars",
    "finalize_analysis_metadata",
    "main",
    "read_analysis_model_ids",
    "read_analysis_model_index_ids",
    "upload_files_with_s5cmd",
    "write_analysis_metadata_parquet",
    "write_high_quality_model_ids",
]


@dataclass(frozen=True, slots=True)
class AnalysisFinalizerResult:
    """Paths and row counts produced by native analysis finalization."""

    csv_path: Path
    parquet_path: Path
    selected_ids_path: Path
    model_tar_index_path: Path | None
    analysis_row_count: int
    selected_model_count: int
    indexed_model_count: int | None
    extracted_file_count: int | None
    uploaded_file_count: int | None

    def to_redacted_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary."""
        return {
            "csv_path": str(self.csv_path),
            "parquet_path": str(self.parquet_path),
            "selected_ids_path": str(self.selected_ids_path),
            "model_tar_index_path": str(self.model_tar_index_path) if self.model_tar_index_path else None,
            "analysis_row_count": self.analysis_row_count,
            "selected_model_count": self.selected_model_count,
            "indexed_model_count": self.indexed_model_count,
            "extracted_file_count": self.extracted_file_count,
            "uploaded_file_count": self.uploaded_file_count,
        }


def finalize_analysis_metadata(
    *,
    csv_path: Path,
    parquet_path: Path,
    selected_ids_path: Path,
    local_tars_csv: Path | None = None,
    local_tar_dir: Path | None = None,
    model_tar_index_path: Path | None = None,
    high_quality_work_dir: Path | None = None,
    high_quality_s3_prefix: str | None = None,
    s5cmd_path: str = "s5cmd",
    s5cmd_numworkers: int | None = None,
    parquet_block_size_bytes: int = DEFAULT_PARQUET_BLOCK_SIZE_BYTES,
) -> AnalysisFinalizerResult:
    """Finalize native analysis metadata outputs.

    Args:
        csv_path: Source ``analysis_metadata.csv``.
        parquet_path: Destination ``analysis_metadata.parquet``.
        selected_ids_path: Destination ``high_quality_model_ids.txt``.
        local_tars_csv: Optional source ``local_tars.csv``.
        local_tar_dir: Optional root containing local tar files. Defaults to
            ``<local_tars_csv parent>/local_tars`` when tar paths must be
            resolved from ``tar_name`` manifest rows.
        model_tar_index_path: Optional destination ``model_tar_index.csv``.
        high_quality_work_dir: Optional scratch directory for selected
            high-quality tar members.
        high_quality_s3_prefix: Optional object-store prefix to upload extracted
            high-quality files with ``s5cmd``.
        s5cmd_path: s5cmd executable used when uploading high-quality files.
        s5cmd_numworkers: Optional s5cmd worker count.
        parquet_block_size_bytes: PyArrow CSV streaming block size.

    Returns:
        Produced artifact paths and counts.
    """
    analysis_model_ids, non_output_model_ids = read_analysis_model_index_ids(csv_path)
    analysis_row_count = write_analysis_metadata_parquet(
        csv_path,
        parquet_path,
        block_size_bytes=parquet_block_size_bytes,
    )
    selected_model_ids = write_high_quality_model_ids(csv_path, selected_ids_path)

    indexed_model_count: int | None = None
    extracted_file_count: int | None = None
    uploaded_file_count: int | None = None
    if model_tar_index_path is not None:
        if local_tars_csv is None:
            msg = "local_tars_csv is required when model_tar_index_path is provided"
            raise ValueError(msg)
        indexed_model_count = build_model_tar_index(
            local_tars_csv=local_tars_csv,
            model_ids=analysis_model_ids,
            output_path=model_tar_index_path,
            local_tar_dir=local_tar_dir,
            allowed_missing_model_ids=non_output_model_ids,
        )
    if high_quality_work_dir is not None or high_quality_s3_prefix is not None:
        if local_tars_csv is None:
            msg = "local_tars_csv is required when extracting high-quality files"
            raise ValueError(msg)
        if high_quality_work_dir is None:
            msg = "high_quality_work_dir is required when high_quality_s3_prefix is provided"
            raise ValueError(msg)
        extracted_files = extract_high_quality_from_tars(
            local_tars_csv=local_tars_csv,
            selected_model_ids=selected_model_ids,
            work_dir=high_quality_work_dir,
            local_tar_dir=local_tar_dir,
        )
        extracted_file_count = len(extracted_files)
        if high_quality_s3_prefix is not None:
            uploaded_file_count = upload_files_with_s5cmd(
                files=extracted_files,
                destination_prefix=high_quality_s3_prefix,
                s5cmd_path=s5cmd_path,
                numworkers=s5cmd_numworkers,
                command_dir=high_quality_work_dir,
            )

    return AnalysisFinalizerResult(
        csv_path=csv_path,
        parquet_path=parquet_path,
        selected_ids_path=selected_ids_path,
        model_tar_index_path=model_tar_index_path,
        analysis_row_count=analysis_row_count,
        selected_model_count=len(selected_model_ids),
        indexed_model_count=indexed_model_count,
        extracted_file_count=extracted_file_count,
        uploaded_file_count=uploaded_file_count,
    )


def write_analysis_metadata_parquet(
    csv_path: Path,
    parquet_path: Path,
    *,
    block_size_bytes: int = DEFAULT_PARQUET_BLOCK_SIZE_BYTES,
) -> int:
    """Stream ``analysis_metadata.csv`` to parquet and return row count."""
    if block_size_bytes < 1:
        msg = "block_size_bytes must be positive"
        raise ValueError(msg)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    header = _read_csv_header(csv_path)
    convert_options = pacsv.ConvertOptions(column_types={column: pa.string() for column in header})
    reader = pacsv.open_csv(
        csv_path,
        read_options=pacsv.ReadOptions(block_size=block_size_bytes),
        convert_options=convert_options,
    )
    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, batch.schema, compression="snappy")
            writer.write_table(pa.Table.from_batches([batch]))
            row_count += batch.num_rows
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pq.write_table(_empty_string_table(header), parquet_path, compression="snappy")

    return row_count


def write_high_quality_model_ids(
    csv_path: Path,
    output_path: Path,
    *,
    model_id_column: str = MODEL_ID_COLUMN,
    quality_column: str = QUALITY_COLUMN,
) -> tuple[str, ...]:
    """Write selected model IDs from analysis metadata and return them.

    The selected ID file preserves first-seen CSV order while deduplicating
    repeated model IDs.
    """
    selected: list[str] = []
    seen: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, csv_path, REQUIRED_ANALYSIS_COLUMNS)
        for row in reader:
            model_id = row.get(model_id_column, "").strip()
            if not model_id or model_id in seen:
                continue
            if _selected_quality_value(row.get(quality_column, "")):
                selected.append(model_id)
                seen.add(model_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    return tuple(selected)


def read_analysis_model_ids(
    csv_path: Path,
    *,
    model_id_column: str = MODEL_ID_COLUMN,
) -> tuple[str, ...]:
    """Return first-seen model IDs from analysis metadata after schema validation."""
    model_ids, _ = read_analysis_model_index_ids(csv_path, model_id_column=model_id_column)
    return model_ids


def read_analysis_model_index_ids(
    csv_path: Path,
    *,
    model_id_column: str = MODEL_ID_COLUMN,
) -> tuple[tuple[str, ...], frozenset[str]]:
    """Return first-seen model IDs and IDs not expected to have tar payloads."""
    model_ids: list[str] = []
    seen: set[str] = set()
    non_output_model_ids: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, csv_path, REQUIRED_ANALYSIS_COLUMNS)
        for row_number, row in enumerate(reader, start=2):
            model_id = row.get(model_id_column, "").strip()
            if not model_id:
                msg = f"{csv_path} row {row_number} has empty {model_id_column}"
                raise ValueError(msg)
            if model_id not in seen:
                model_ids.append(model_id)
                seen.add(model_id)
            if _non_output_analysis_row(row):
                non_output_model_ids.add(model_id)
    return tuple(model_ids), frozenset(non_output_model_ids)


def build_model_tar_index(
    *,
    local_tars_csv: Path,
    model_ids: Sequence[str],
    output_path: Path,
    local_tar_dir: Path | None = None,
    allowed_missing_model_ids: Sequence[str] | set[str] | frozenset[str] = (),
) -> int:
    """Build ``model_tar_index.csv`` for analysis model IDs by scanning local tars.

    The output schema is intentionally small and compatible with the validation
    checks already used in this repository: ``model_id,tar_path``.  Each model
    is indexed at most once, using the first tar-manifest row whose members
    contain an artifact for that model.
    """
    if not local_tars_csv.exists():
        raise FileNotFoundError(local_tars_csv)

    all_model_ids = tuple(dict.fromkeys(model_id for model_id in model_ids if model_id))
    allowed_missing = set(allowed_missing_model_ids)
    remaining = set(all_model_ids)
    rows: list[dict[str, str]] = []

    with local_tars_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = f"CSV has no header: {local_tars_csv}"
            raise ValueError(msg)
        if "tar_path" not in reader.fieldnames and "tar_name" not in reader.fieldnames:
            msg = f"{local_tars_csv} requires either tar_path or tar_name"
            raise ValueError(msg)

        for manifest_row in reader:
            if not remaining:
                break
            tar_path, index_tar_path = _resolve_tar_path(
                manifest_row,
                manifest_path=local_tars_csv,
                local_tar_dir=local_tar_dir,
            )
            ids_in_tar = _model_ids_in_tar(tar_path, remaining)
            for model_id in all_model_ids:
                if model_id in remaining and model_id in ids_in_tar:
                    rows.append({"model_id": model_id, "tar_path": index_tar_path})
                    remaining.remove(model_id)
    unexpected_missing = remaining - allowed_missing
    if unexpected_missing:
        sample = ", ".join(sorted(unexpected_missing)[:10])
        suffix = "" if len(unexpected_missing) <= 10 else f", ... ({len(unexpected_missing)} missing)"
        msg = f"{local_tars_csv} did not index all analysis models: {sample}{suffix}"
        raise ValueError(msg)
    for model_id in all_model_ids:
        if model_id in remaining:
            rows.append({"model_id": model_id, "tar_path": ""})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model_id", "tar_path"))
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def extract_high_quality_from_tars(
    *,
    local_tars_csv: Path,
    selected_model_ids: Sequence[str],
    work_dir: Path,
    local_tar_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Extract selected model artifacts and metadata members from local tars."""
    if not local_tars_csv.exists():
        raise FileNotFoundError(local_tars_csv)

    selected_ids = set(dict.fromkeys(model_id for model_id in selected_model_ids if model_id))
    extracted_dir = work_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    written_names: set[str] = set()

    with local_tars_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = f"CSV has no header: {local_tars_csv}"
            raise ValueError(msg)
        if (
            "tar_path" not in reader.fieldnames
            and "tar_name" not in reader.fieldnames
            and "s3_uri" not in reader.fieldnames
        ):
            msg = f"{local_tars_csv} requires tar_path, tar_name, or file:// s3_uri"
            raise ValueError(msg)

        for manifest_row in reader:
            tar_path, _ = _resolve_tar_path(
                manifest_row,
                manifest_path=local_tars_csv,
                local_tar_dir=local_tar_dir,
            )
            tar_type = manifest_row.get("tar_type", "").strip()
            extracted.extend(
                _extract_matching_members(
                    tar_path=tar_path,
                    output_dir=extracted_dir,
                    selected_model_ids=selected_ids,
                    include_all_members=tar_type == "metadata",
                    written_names=written_names,
                )
            )

    return tuple(extracted)


def upload_files_with_s5cmd(
    *,
    files: Sequence[Path],
    destination_prefix: str,
    s5cmd_path: str,
    command_dir: Path,
    numworkers: int | None = None,
) -> int:
    """Upload files to an object-store prefix using an ``s5cmd run`` file."""
    if not files:
        return 0
    command_dir.mkdir(parents=True, exist_ok=True)
    command_file = command_dir / "s5cmd_high_quality_upload.txt"
    prefix = destination_prefix.rstrip("/")
    with command_file.open("w", encoding="utf-8") as handle:
        for path in files:
            destination = f"{prefix}/{path.name}"
            handle.write(f"cp {shlex.quote(str(path))} {shlex.quote(destination)}\n")
    argv = [s5cmd_path]
    if numworkers is not None and numworkers > 0:
        argv.extend(["--numworkers", str(numworkers)])
    argv.extend(["run", str(command_file)])
    subprocess.run(argv, check=True)
    return len(files)


def main(argv: Sequence[str] | None = None) -> int:
    """Run native analysis finalization as a small module CLI."""
    parser = _arg_parser()
    args = parser.parse_args(argv)
    result = finalize_analysis_metadata(
        csv_path=args.csv,
        parquet_path=args.parquet,
        selected_ids_path=args.selected_ids,
        local_tars_csv=args.local_tars_csv,
        local_tar_dir=args.local_tar_dir,
        model_tar_index_path=args.model_tar_index,
        high_quality_work_dir=args.high_quality_work_dir,
        high_quality_s3_prefix=args.high_quality_s3_prefix,
        s5cmd_path=args.s5cmd_path,
        s5cmd_numworkers=args.s5cmd_numworkers,
        parquet_block_size_bytes=args.parquet_block_size_bytes,
    )
    print(json.dumps(result.to_redacted_dict(), sort_keys=True))
    return 0


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize native BSPP analysis metadata artifacts.")
    parser.add_argument("--csv", required=True, type=Path, help="Input analysis_metadata.csv")
    parser.add_argument("--parquet", required=True, type=Path, help="Output analysis_metadata.parquet")
    parser.add_argument("--selected-ids", required=True, type=Path, help="Output high_quality_model_ids.txt")
    parser.add_argument("--local-tars-csv", type=Path, help="Optional input local_tars.csv")
    parser.add_argument("--local-tar-dir", type=Path, help="Optional local_tars directory")
    parser.add_argument("--model-tar-index", type=Path, help="Optional output model_tar_index.csv")
    parser.add_argument("--high-quality-work-dir", type=Path, help="Optional high-quality extraction work dir")
    parser.add_argument("--high-quality-s3-prefix", help="Optional destination prefix for extracted high-quality files")
    parser.add_argument("--s5cmd-path", default="s5cmd", help="s5cmd executable for high-quality uploads")
    parser.add_argument("--s5cmd-numworkers", type=int, help="Optional s5cmd worker count")
    parser.add_argument(
        "--parquet-block-size-bytes",
        type=int,
        default=DEFAULT_PARQUET_BLOCK_SIZE_BYTES,
        help="PyArrow CSV streaming block size",
    )
    return parser


def _read_csv_header(csv_path: Path) -> tuple[str, ...]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            msg = f"CSV has no header: {csv_path}"
            raise ValueError(msg) from exc
    if not header:
        msg = f"CSV has no header: {csv_path}"
        raise ValueError(msg)
    return header


def _empty_string_table(columns: Sequence[str]) -> pa.Table:
    return pa.table({column: pa.array([], type=pa.string()) for column in columns})


def _require_columns(fieldnames: Sequence[str] | None, path: Path, required: Sequence[str]) -> None:
    if fieldnames is None:
        msg = f"CSV has no header: {path}"
        raise ValueError(msg)
    missing = [column for column in required if column not in fieldnames]
    if missing:
        msg = f"{path} is missing required columns: {', '.join(missing)}"
        raise ValueError(msg)


def _selected_quality_value(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _non_output_analysis_row(row: dict[str, str]) -> bool:
    status = (row.get("upload_status") or "").strip().lower()
    failure_reason = (row.get("failure_reason") or "").strip()
    if status in NON_OUTPUT_UPLOAD_STATUSES:
        return True
    return bool(failure_reason) and status not in {"uploaded", "local_tarred"}


def _resolve_tar_path(
    row: dict[str, str],
    *,
    manifest_path: Path,
    local_tar_dir: Path | None,
) -> tuple[Path, str]:
    tar_path_text = row.get("tar_path", "").strip()
    manifest_dir = manifest_path.parent
    if tar_path_text:
        tar_path = Path(tar_path_text)
        resolved = tar_path if tar_path.is_absolute() else manifest_dir / tar_path
        return resolved, tar_path_text

    file_uri_path = _path_from_file_uri(row.get("s3_uri", "").strip())
    if file_uri_path is not None:
        return file_uri_path, _relative_tar_path(file_uri_path, manifest_dir)

    tar_name = row.get("tar_name", "").strip()
    if not tar_name:
        msg = f"local tar manifest row has neither tar_path nor tar_name in {manifest_path}"
        raise ValueError(msg)

    tar_root = local_tar_dir or manifest_dir / "local_tars"
    candidates = _tar_name_candidates(tar_root, tar_name, row)
    for candidate in candidates:
        if candidate.exists():
            return candidate, _relative_tar_path(candidate, manifest_dir)

    raise FileNotFoundError(candidates[0])


def _tar_name_candidates(tar_root: Path, tar_name: str, row: dict[str, str]) -> tuple[Path, ...]:
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
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme != "file":
        return None
    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
        msg = f"unsupported file URI host in local tar manifest: {value}"
        raise ValueError(msg)
    return Path(unquote(parsed.path))


def _model_ids_in_tar(tar_path: Path, candidate_model_ids: set[str]) -> set[str]:
    if not candidate_model_ids:
        return set()

    found: set[str] = set()
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            basename = PurePosixPath(member.name).name
            for model_id in candidate_model_ids - found:
                if _member_matches_model_id(basename, model_id):
                    found.add(model_id)
                    break
            if found == candidate_model_ids:
                break
    return found


def _extract_matching_members(
    *,
    tar_path: Path,
    output_dir: Path,
    selected_model_ids: set[str],
    include_all_members: bool,
    written_names: set[str],
) -> tuple[Path, ...]:
    extracted: list[Path] = []
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            basename = PurePosixPath(member.name).name
            if not basename:
                continue
            if not include_all_members and not any(
                _member_matches_model_id(basename, model_id) for model_id in selected_model_ids
            ):
                continue
            if basename in written_names:
                msg = f"duplicate extracted high-quality member name: {basename}"
                raise ValueError(msg)
            source = archive.extractfile(member)
            if source is None:
                continue
            output_path = output_dir / basename
            with source, output_path.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            written_names.add(basename)
            extracted.append(output_path)
    return tuple(extracted)


def _member_matches_model_id(basename: str, model_id: str) -> bool:
    return basename == model_id or basename.startswith(f"{model_id}-") or basename.startswith(f"{model_id}.")


if __name__ == "__main__":
    raise SystemExit(main())
