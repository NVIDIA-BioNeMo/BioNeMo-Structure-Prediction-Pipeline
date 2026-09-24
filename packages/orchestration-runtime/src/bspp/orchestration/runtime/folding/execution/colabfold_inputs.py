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

"""ColabFold input preparation: stage one verified merged A3M for direct intake.

ColabFold's ``colabfold_batch`` accepts a single merged A3M directly, so this
adapter does not re-split, re-parse, or fold anything.  It stages the already
verified merged A3M at ``alignments/<target_id>.a3m`` (the exact path
``ColabFoldBackend.run`` later resolves) plus deterministic empty ``fasta/`` and
``templates/`` directory stubs, and returns a ``PreparedInput`` describing that
layout.  It never imports or calls ``OpenFoldInputPreprocessor``; ColabFold's
direct-A3M intake is deliberately not routed through the OpenFold FASTA/Stockholm
path.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .errors import FoldingBackendError
from .models import PreparedInput, ProteinTarget, require_valid_target_identity
from .msa_models import MSAResult

__all__ = ["prepare"]

_BACKEND_IDENTITY = "colabfold-inputs"


def prepare(target: ProteinTarget, msa: MSAResult, output_dir: Path) -> PreparedInput:
    """Stage one verified merged A3M at ``alignments/<target_id>.a3m``.

    The target identity names the staged A3M file, so it must satisfy the
    model-ID grammar before any filesystem operation.  The merged source is
    resolved from ``msa.metadata["merged_source_path"]`` and must be a regular
    file; no re-split or re-parse is performed, so the staged bytes are
    identical to the already-verified merged A3M.
    """
    require_valid_target_identity(target)
    if len(msa.chains) != len(target.chains):
        raise FoldingBackendError("MSA result does not contain one alignment bundle per chain")

    # Fail closed on a non-fresh output directory: silently reusing one would let
    # stale FASTA/template/alignment files survive into the PreparedInput, and a
    # wholesale rmtree could wipe a partially completed rerun's evidence (plan
    # review finding 10). The executor's nonempty-action-root rerun guard makes a
    # fresh directory the normal case; anything else is an operator-visible error.
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FoldingBackendError(f"colabfold output path is not a directory: {output_dir}")
        pre_existing = sorted(path for path in output_dir.rglob("*") if path.is_file() or path.is_symlink())
        if pre_existing:
            raise FoldingBackendError(
                "colabfold input preparation requires a fresh output directory; "
                f"found pre-existing content: {pre_existing[0]}"
            )

    merged_source_value = msa.metadata.get("merged_source_path")
    if not isinstance(merged_source_value, str) or not merged_source_value:
        raise FoldingBackendError("MSA result metadata is missing a non-empty merged_source_path")
    merged_source = Path(merged_source_value)
    if not merged_source.is_file():
        raise FoldingBackendError(f"merged MSA source is not a regular file: {merged_source}")

    fasta_dir = output_dir / "fasta"
    alignment_dir = output_dir / "alignments"
    template_dir = output_dir / "templates"
    for path in (fasta_dir, alignment_dir, template_dir):
        path.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(merged_source, alignment_dir / f"{target.target_id}.a3m")

    return PreparedInput(
        backend=_BACKEND_IDENTITY,
        fasta_dir=fasta_dir,
        alignment_dir=alignment_dir,
        template_dir=template_dir,
        metadata={
            "layout": "colabfold",
            "merged_source_path": str(merged_source),
            "target_id": target.target_id,
            "chain_count": len(target.chains),
        },
    )
