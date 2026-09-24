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

"""Pipeline runner: pre-filtering and subprocess invocation.

Provides input validation (null-check on meta JSONs) and a wrapper to
invoke the production pipeline as a subprocess.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

META_JSON_SUFFIX = "-meta_v1.json"

__all__ = [
    "PipelineResult",
    "check_meta_json_for_null",
    "prefilter_batch",
    "run_pipeline",
]


def check_meta_json_for_null(meta_path: Path) -> bool:
    """Return ``True`` if the meta JSON has null values in critical fields.

    Performs a fast string scan for null tokens (``[null]``, ``,null,``,
    ``,null]``, ``[null,``).  Falls back to full JSON parse if the file
    cannot be read as text.

    This catches models whose PAE/pLDDT/max_pae arrays contain null
    entries, which cause ipSAE to crash.

    Args:
        meta_path: Path to a ``*-meta_v1.json`` file.

    Returns:
        ``True`` if any null values are detected, ``False`` otherwise.
        Returns ``False`` if the file does not exist.
    """
    if not meta_path.exists():
        return False

    raw = meta_path.read_text()

    # Fast string scan for null tokens in JSON arrays.
    # Covers both compact (orjson: ",null,") and spaced (stdlib: ", null,") formats.
    return (
        ",null," in raw
        or "[null]" in raw
        or ",null]" in raw
        or "[null," in raw
        or ", null," in raw
        or ", null]" in raw
        or "[null, " in raw
    )


def prefilter_batch(
    model_ids: list[str],
    input_dir: Path,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Pre-validate input meta JSONs for null values.

    For each model ID, locates its ``<model_id>-meta_v1.json`` file in
    *input_dir* and checks for null values in critical fields.

    Args:
        model_ids: Model IDs to check.
        input_dir: Directory containing model input files.

    Returns:
        A tuple of ``(good_ids, failed_ids_with_reason)`` where
        ``failed_ids_with_reason`` is a list of ``(model_id, reason)``
        tuples.
    """
    good_ids: list[str] = []
    bad_list: list[tuple[str, str]] = []

    for mid in model_ids:
        meta_path = input_dir / f"{mid}{META_JSON_SUFFIX}"
        if check_meta_json_for_null(meta_path):
            bad_list.append((mid, "null in input json"))
        else:
            good_ids.append(mid)

    return good_ids, bad_list


@dataclass
class PipelineResult:
    """Result of a pipeline subprocess execution."""

    exit_code: int
    models_processed: int
    models_failed: int
    duration: float


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    manifest_csv: Path,
    *,
    workers: int,
    stages: str = "ipsae dssp validation metadata_export modelcif_export",
    resume: bool = True,
    uniprot_db: Path | None = None,
    **kwargs: Any,
) -> int:
    """Invoke the production pipeline as a subprocess.

    Builds a command line for ``production_pipeline.py`` and runs it via
    :func:`subprocess.run`.

    Args:
        input_dir: Directory with model input files.
        output_dir: Destination for pipeline outputs.
        manifest_csv: Path to the shard's filtered manifest CSV.
        stages: Space-separated list of pipeline stages to run.
        workers: Number of parallel workers.
        resume: If ``True``, enable caching/resume mode.
        uniprot_db: Optional path to UniProt DuckDB database.
        **kwargs: Additional keyword arguments passed as ``--key value``
            CLI flags.

    Returns:
        Exit code of the subprocess.
    """
    # Locate production_pipeline.py relative to afdb-toolkit
    try:
        toolkit_module = import_module("afdb_integration_kit")
        toolkit_file = getattr(toolkit_module, "__file__", None)
        toolkit_dir = Path(toolkit_file).resolve().parent.parent if toolkit_file is not None else None
    except (ImportError, AttributeError):
        toolkit_dir = None

    pipeline_script: Path | None = None
    if toolkit_dir is not None:
        candidate = toolkit_dir / "scripts" / "production_pipeline.py"
        if candidate.exists():
            pipeline_script = candidate

    if pipeline_script is None:
        logger.error("production_pipeline.py not found")
        return 127

    cmd: list[str] = [
        sys.executable,
        str(pipeline_script),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--chain-mapping",
        str(manifest_csv),
        "--workers",
        str(workers),
        "--python-cmd",
        sys.executable,
        "--clash-device",
        str(kwargs.pop("clash_device", "auto")),
    ]

    if uniprot_db is not None:
        cmd.append("--heterodimers")

    if resume:
        cmd.append("--resume")

    if uniprot_db is not None:
        cmd.extend(["--uniprot-db", str(uniprot_db)])

    for key, value in kwargs.items():
        cli_key = key.replace("_", "-")
        cmd.extend([f"--{cli_key}", str(value)])

    logger.info("Running pipeline: %s", " ".join(cmd))

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        logger.error(
            "Pipeline failed (exit %d, %.1fs):\nstdout: %s\nstderr: %s",
            result.returncode,
            elapsed,
            result.stdout[-500:] if result.stdout else "",
            result.stderr[-500:] if result.stderr else "",
        )
    else:
        logger.info("Pipeline completed successfully in %.1fs", elapsed)

    return result.returncode
