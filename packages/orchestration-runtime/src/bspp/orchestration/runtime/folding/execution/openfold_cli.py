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

"""Job-local OpenFold CLI backend with earliest-complete score capture.

Port Baseline: frozen reference pipeline
  - ``OpenFoldContainerBackend`` (src/afdb_pipeline/backends/local.py)
  - ``_quality_from_openfold_output`` (src/afdb_pipeline/cluster_worker.py)

Runs the harvested ``run_pretrained_openfold.py`` CLI as a checked subprocess
(no ContainerRuntime, no Slurm, no nested container) and emits the
canonical ``-model_v1.pdb`` / ``-meta_v1.json`` pair through the Step-0 emitter
helpers. Full ``plddt`` / ``pae`` / ``max_pae`` (+ optional ``ptm``/``iptm``)
are captured at the earliest complete boundary, before the output_dict pickle
is unlinked, and are never reconstructed from summaries.

The harvested invocation passes the FASTA *directory* (not one file) as the
first positional argument and always adds ``--save_outputs`` so the full output
dictionary is emitted.  Output artifacts are discovered recursively under
``raw/`` using the harvested ``"-".join(chain_ids)`` output tag.
"""

from __future__ import annotations

import json
import math
import pickle
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import SupportsFloat, cast

import numpy as np

from bspp.orchestration.contract.runspec import VALID_TOOL_USED

from .emitter_support import canonical_pair_names, select_tool_used, serialize_scores_json
from .errors import FoldingBackendError
from .models import FoldingResult, PreparedInput, ProteinTarget, StructurePrediction

# The plain OpenFold engine has one tool vocabulary entry (VALID_TOOL_USED[2]);
# both the homodimer and heterodimer selections use the same frozen string.
_OPENFOLD_TOOL_USED = VALID_TOOL_USED[2]

# Raw output directory leaf written by the harvested CLI's ``--output_dir``.
_RAW_OUTPUT_DIR_NAME = "raw"


class OpenFoldCliBackend:
    """Job-local OpenFold backend running the harvested CLI as a subprocess."""

    name = "openfold-cli"

    def __init__(
        self,
        model_dir: Path,
        model_preset: str = "model_1_multimer_v3",
        parameter_file: str = "params_model_1_multimer_v3.npz",
        seed: int = 0,
    ) -> None:
        self.model_dir = model_dir
        self.model_preset = model_preset
        self.parameter_file = parameter_file
        self.seed = seed

    def run(
        self,
        target: ProteinTarget,
        prepared: PreparedInput,
        output_dir: Path,
        *,
        leaked_homodimer: bool = False,
    ) -> FoldingResult:
        # ``leaked_homodimer`` is keyword-only with a default only so this
        # method stays structurally assignable to the frozen ``FoldingBackend``
        # protocol. The real caller (Joint Final envelope) resolves the
        # classification from the external chain manifest and passes it
        # explicitly.
        parameter_path = self.model_dir / self.parameter_file
        if not parameter_path.is_file():
            msg = f"OpenFold parameter file not found: {parameter_path}"
            raise FoldingBackendError(msg)

        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = output_dir / _RAW_OUTPUT_DIR_NAME
        argv = build_openfold_cli_argv(
            prepared,
            output_dir,
            model_dir=self.model_dir,
            model_preset=self.model_preset,
            parameter_file=self.parameter_file,
            seed=self.seed,
        )
        # Checked subprocess. A nonzero exit surfaces as
        # subprocess.CalledProcessError; the task scope does not wrap it.
        subprocess.run(argv, check=True)

        output_tag = _derive_output_tag(prepared)
        output, pickle_path = _read_openfold_output(raw_dir, output_tag, self.model_preset)
        plddt, pae, max_pae, ptm, iptm = capture_openfold_scores(output)

        structure_name, scores_name = canonical_pair_names(target.target_id)
        structure_path = output_dir / structure_name
        scores_path = output_dir / scores_name

        structure_source = _find_unrelaxed_pdb(raw_dir, output_tag, self.model_preset)

        # Serialize the canonical scores JSON strictly before unlinking the
        # pickle, so the earliest complete boundary is observable.
        scores_path.write_text(
            serialize_scores_json(plddt=plddt, pae=pae, max_pae=max_pae, ptm=ptm, iptm=iptm),
            encoding="utf-8",
        )
        shutil.copyfile(structure_source, structure_path)
        if pickle_path is not None:
            pickle_path.unlink(missing_ok=True)

        tool_used = select_tool_used(
            target.target_id,
            leaked_homodimer=leaked_homodimer,
            homodimer_tool_used=_OPENFOLD_TOOL_USED,
            heterodimer_tool_used=_OPENFOLD_TOOL_USED,
        )
        return FoldingResult(
            backend=self.name,
            predictions=(StructurePrediction(rank=1, structure_path=structure_path, scores_path=scores_path),),
            metadata={
                "tool_used": tool_used,
                "model_preset": self.model_preset,
                "parameter_file": self.parameter_file,
                "seed": self.seed,
            },
        )


def build_openfold_cli_argv(
    prepared: PreparedInput,
    output_dir: Path,
    *,
    model_dir: Path,
    model_preset: str,
    parameter_file: str,
    seed: int,
) -> list[str]:
    """Build the harvested host-local OpenFold CLI argv list.

    The harvested CLI takes the FASTA directory as its first positional
    argument, not a single FASTA file, and always receives ``--save_outputs``
    so the full output dictionary is emitted for canonical score capture.
    """

    raw_dir = output_dir / _RAW_OUTPUT_DIR_NAME
    return [
        "run_pretrained_openfold.py",
        str(prepared.fasta_dir),
        str(prepared.template_dir),
        "--use_precomputed_alignments",
        str(prepared.alignment_dir),
        "--output_dir",
        str(raw_dir),
        "--model_device",
        "cuda:0",
        "--config_preset",
        model_preset,
        "--jax_param_path",
        str(model_dir / parameter_file),
        "--data_random_seed",
        str(seed),
        "--skip_relaxation",
        "--save_outputs",
    ]


def _derive_output_tag(prepared: PreparedInput) -> str:
    """Derive the harvested output tag as ``"-".join(chain_ids)``."""

    chain_ids_value = prepared.metadata.get("chain_ids")
    if not isinstance(chain_ids_value, (list, tuple)) or not chain_ids_value:
        raise FoldingBackendError("OpenFold prepared input is missing non-empty 'chain_ids' metadata")
    chain_ids: list[str] = []
    for item in chain_ids_value:
        if not isinstance(item, str) or not item:
            raise FoldingBackendError("OpenFold 'chain_ids' metadata must contain only non-empty strings")
        chain_ids.append(item)
    return "-".join(chain_ids)


def capture_openfold_scores(
    output: Mapping[str, object],
) -> tuple[list[float], list[list[float]], float, float | None, float | None]:
    """Extract full scores from an OpenFold output dict.

    Returns ``(plddt, pae, max_pae, ptm, iptm)``. Missing/empty/non-finite
    ``plddt`` or ``pae`` raise :class:`FoldingBackendError`; scores are never
    reconstructed from summaries.  ``pae`` must be square before canonical
    serialization: its row count and every row width must equal the ``plddt``
    residue count, so a ragged or non-square matrix can never become a
    canonical score artifact.
    """

    if "plddt" not in output:
        raise FoldingBackendError("OpenFold output is missing 'plddt'")
    plddt = _to_float_list(output["plddt"], "plddt")

    if "predicted_aligned_error" in output:
        pae = _to_float_matrix(output["predicted_aligned_error"], "predicted_aligned_error")
    elif "pae" in output:
        pae = _to_float_matrix(output["pae"], "pae")
    else:
        raise FoldingBackendError("OpenFold output is missing both 'predicted_aligned_error' and 'pae'")

    if len(pae) != len(plddt):
        raise FoldingBackendError(f"OpenFold 'pae' has {len(pae)} rows but 'plddt' has {len(plddt)} residues")
    for row_index, row in enumerate(pae):
        if len(row) != len(plddt):
            raise FoldingBackendError(
                f"OpenFold 'pae' row {row_index} has {len(row)} columns but 'plddt' has {len(plddt)} residues"
            )

    ptm = _optional_float(output["ptm"] if "ptm" in output else output.get("ptm_score"), "ptm")
    iptm = _optional_float(output["iptm"] if "iptm" in output else output.get("iptm_score"), "iptm")

    max_pae = max(item for row in pae for item in row)
    return plddt, pae, max_pae, ptm, iptm


def _to_float_list(value: object, name: str) -> list[float]:
    try:
        arr = np.asarray(value)
    except ValueError as exc:
        raise FoldingBackendError(f"OpenFold '{name}' must be a 1-D array") from exc
    if arr.ndim != 1:
        raise FoldingBackendError(f"OpenFold '{name}' must be a 1-D array")
    result = [float(item) for item in arr.tolist()]
    if not result:
        raise FoldingBackendError(f"OpenFold '{name}' is empty")
    if not all(math.isfinite(item) for item in result):
        raise FoldingBackendError(f"OpenFold '{name}' contains non-finite values")
    return result


def _to_float_matrix(value: object, name: str) -> list[list[float]]:
    try:
        arr = np.asarray(value)
    except ValueError as exc:
        raise FoldingBackendError(f"OpenFold '{name}' must be a 2-D array") from exc
    if arr.ndim != 2:
        raise FoldingBackendError(f"OpenFold '{name}' must be a 2-D array")
    result = [[float(item) for item in row] for row in arr.tolist()]
    if not result or any(not row for row in result):
        raise FoldingBackendError(f"OpenFold '{name}' is empty")
    if not all(math.isfinite(item) for row in result for item in row):
        raise FoldingBackendError(f"OpenFold '{name}' contains non-finite values")
    return result


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise FoldingBackendError(f"OpenFold '{name}' must be a finite number")
    try:
        result = float(cast("str | SupportsFloat", value))
    except (TypeError, ValueError) as exc:
        raise FoldingBackendError(f"OpenFold '{name}' must be a finite number") from exc
    if not math.isfinite(result):
        raise FoldingBackendError(f"OpenFold '{name}' must be a finite number")
    return result


def _read_openfold_output(
    raw_dir: Path,
    output_tag: str,
    model_preset: str,
) -> tuple[Mapping[str, object], Path | None]:
    """Read the full OpenFold output dict: tagged pickle, fallback scores JSON.

    Returns ``(mapping, pickle_path)`` where ``pickle_path`` is the exact tagged
    pickle that was read, or ``None`` when the structure-derived scores JSON was
    used instead.  The caller unlinks only that exact pickle after canonical
    serialization.
    """

    pickle_candidates = sorted(raw_dir.rglob(f"{output_tag}_{model_preset}_output_dict.pkl"))
    if pickle_candidates:
        pickle_path = pickle_candidates[0]
        with pickle_path.open("rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, Mapping):
            raise FoldingBackendError(f"OpenFold output pickle at {pickle_path} is not a mapping")
        return cast("Mapping[str, object]", payload), pickle_path

    structure = _find_unrelaxed_pdb(raw_dir, output_tag, model_preset)
    scores_path = _derive_scores_path(structure)
    if scores_path.is_file():
        with scores_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, Mapping):
            raise FoldingBackendError(f"OpenFold scores JSON at {scores_path} is not a mapping")
        return cast("Mapping[str, object]", payload), None

    raise FoldingBackendError(
        f"OpenFold output not found under {raw_dir}: expected a tagged output_dict pickle "
        "or a structure-derived scores JSON"
    )


def _find_unrelaxed_pdb(raw_dir: Path, output_tag: str, model_preset: str) -> Path:
    """Locate the harvested unrelaxed PDB under ``raw/``.

    Discovery order matches the harvest: the tagged preset name first, then the
    ranked NVIDIA-fork names.  Multiple ranked candidates resolve
    deterministically by ascending rank (lexicographic on the zero-padded rank).
    """

    primary = sorted(raw_dir.rglob(f"{output_tag}_{model_preset}_unrelaxed.pdb"))
    if primary:
        return primary[0]
    ranked = sorted(raw_dir.rglob(f"{output_tag}_unrelaxed_rank_*.pdb"))
    if not ranked:
        raise FoldingBackendError(f"no unrelaxed PDB found under {raw_dir} for output tag {output_tag!r}")
    return ranked[0]


def _derive_scores_path(structure: Path) -> Path:
    """Derive the sibling scores JSON name from a selected structure name.

    ``*_unrelaxed.pdb`` maps to ``*_scores.json`` and ``*_unrelaxed_*.pdb``
    maps to ``*_scores_*.json``.
    """

    name = structure.name
    if not name.endswith(".pdb"):
        raise FoldingBackendError(f"cannot derive scores filename from non-PDB structure: {structure}")
    stem = name[: -len(".pdb")]
    if stem.endswith("_unrelaxed"):
        return structure.with_name(f"{stem[: -len('_unrelaxed')]}_scores.json")
    if "_unrelaxed_" in stem:
        return structure.with_name(stem.replace("_unrelaxed_", "_scores_", 1) + ".json")
    raise FoldingBackendError(f"cannot derive scores filename from structure: {structure}")
