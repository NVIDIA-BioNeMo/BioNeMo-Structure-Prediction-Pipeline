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

"""OpenFold-TRT folding backend: in-process worker with ColabFold naming and PDB normalization.

This backend ports the harvested OpenFold-TRT in-process worker (no
subprocess, no nested container) as a single ``model_fn(batch) -> dict[str, object]``
collaborator. It applies the harvested ``_normalize_pdb`` PDB rewrite and emits the
canonical ``-model_v1.pdb`` / ``-meta_v1.json`` pair with full raw scores through
deterministic Step-0 half-even rounding.

Port Baseline: origin/openfold_rev_pipeline_site @ 3864d0e
``folding/openfold-pipeline/implementations/trt-bionemo/run_benchmark_multiworker.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from bspp.orchestration.contract.runspec import VALID_TOOL_USED

from .chain_manifest import ChainManifest, ambiguous_chain_manifest_metadata, resolve_tool_used
from .emitter_support import atomic_write_text, canonical_pair_names, serialize_scores_json
from .models import FoldingResult, PreparedInput, ProteinTarget, StructurePrediction

OPENFOLD_TRT_TOOL_USED: str = VALID_TOOL_USED[1]

ModelFn = Callable[[PreparedInput], dict[str, object]]


def normalize_pdb(pdb_string: str) -> str:
    lines: list[str] = pdb_string.splitlines()
    out_lines: list[str] = []
    seen_remark: bool = False
    prev_chain: str | None = None
    atom_serial: int = 0
    chain_resid_offset: dict[str, int] = {}

    for line in lines:
        record = line[:6].strip()

        if record == "PARENT":
            continue
        if record == "REMARK":
            if seen_remark:
                continue
            seen_remark = True

        if record == "ATOM":
            atom_serial += 1
            chain = line[21]
            raw_resid = int(line[22:26])

            if chain not in chain_resid_offset:
                chain_resid_offset[chain] = raw_resid - 1

            new_resid = raw_resid - chain_resid_offset[chain]

            if prev_chain is not None and chain != prev_chain:
                ter_serial = atom_serial
                atom_serial += 1
                ter_line = f"TER   {ter_serial:>5d}      {out_lines[-1][17:20]} {prev_chain}{out_lines[-1][22:26]}"
                out_lines.append(ter_line)

            new_line = f"{line[:6]}{atom_serial:>5d}{line[11:21]}{chain}{new_resid:>4d}{line[26:]}"
            out_lines.append(new_line)
            prev_chain = chain
            continue

        if record == "TER":
            continue

        if record == "ENDMDL":
            if prev_chain is not None:
                atom_serial += 1
                ter_line = f"TER   {atom_serial:>5d}      {out_lines[-1][17:20]} {prev_chain}{out_lines[-1][22:26]}"
                out_lines.append(ter_line)
            out_lines.append(line)
            continue

        out_lines.append(line)

    return "\n".join(out_lines) + "\n"


def build_scores_payload(
    *,
    plddt: Sequence[float | int],
    pae: Sequence[Sequence[float | int]],
    max_pae: float | int,
    ptm: float | int | None = None,
    iptm: float | int | None = None,
) -> str:
    """Serialize the OpenFold-TRT harvested score payload at its fixed precision.

    This story compatibility function delegates to the backend-neutral
    serializer with 2-decimal ``plddt``/``pae`` and 4-decimal scalar scores,
    preserving the harvested byte-for-byte output.
    """

    return serialize_scores_json(
        plddt=plddt,
        pae=pae,
        max_pae=max_pae,
        ptm=ptm,
        iptm=iptm,
        decimals=2,
        scalar_decimals=4,
    )


def _optional_score(output: dict[str, object], primary: str, fallback: str) -> float | None:
    """Read an optional scalar score with the harvest's fallback key.

    A missing primary key falls back to the alternate key, then to NaN; NaN is
    normalized to ``None`` so the emitted payload serializes an explicit null.
    """

    raw = output.get(primary, output.get(fallback, np.nan))
    value = float(np.asarray(cast("Any", raw)).item())
    if math.isnan(value):
        return None
    return value


class OpenFoldTrtBackend:
    """Run one target through the in-process OpenFold-TRT worker and emit the canonical pair."""

    name = "openfold-trt"

    def __init__(
        self,
        chain_manifest: ChainManifest | None = None,
        *,
        model_fn: ModelFn | None = None,
        use_colabfold_naming: bool = True,
    ) -> None:
        self._chain_manifest = chain_manifest
        self._model_fn = model_fn
        self._use_colabfold_naming = use_colabfold_naming

    def run(self, target: ProteinTarget, prepared: PreparedInput, output_dir: Path) -> FoldingResult:
        if self._chain_manifest is None or self._model_fn is None:
            raise ValueError("OpenFoldTrtBackend requires a ChainManifest and a model_fn")

        tool_used = resolve_tool_used(self._chain_manifest, target.target_id, tool_used=OPENFOLD_TRT_TOOL_USED)
        if tool_used is None:
            return FoldingResult(backend=self.name, predictions=(), metadata=ambiguous_chain_manifest_metadata())

        output = self._model_fn(prepared)

        plddt = [float(x) for x in np.asarray(cast("Any", output["plddt"])).flatten()]
        pae_np = np.asarray(cast("Any", output["predicted_aligned_error"]))
        pae = [[float(x) for x in row] for row in pae_np]
        max_pae = float(pae_np.max())
        ptm = _optional_score(output, "ptm", "ptm_score")
        iptm = _optional_score(output, "iptm", "iptm_score")

        pdb_string = output.get("pdb_string")
        if not isinstance(pdb_string, str):
            raise RuntimeError("model_fn output must include a 'pdb_string' str value")
        pdb_content = normalize_pdb(pdb_string)

        output_dir.mkdir(parents=True, exist_ok=True)
        structure_name, scores_name = canonical_pair_names(target.target_id)
        structure_path = output_dir / structure_name
        scores_path = output_dir / scores_name
        atomic_write_text(structure_path, pdb_content)
        atomic_write_text(
            scores_path,
            build_scores_payload(plddt=plddt, pae=pae, max_pae=max_pae, ptm=ptm, iptm=iptm),
        )

        return FoldingResult(
            backend=self.name,
            predictions=(
                StructurePrediction(
                    rank=1,
                    structure_path=structure_path,
                    scores_path=scores_path,
                ),
            ),
            metadata={"tool_used": tool_used},
        )


__all__ = ["OPENFOLD_TRT_TOOL_USED", "OpenFoldTrtBackend", "build_scores_payload", "normalize_pdb"]
