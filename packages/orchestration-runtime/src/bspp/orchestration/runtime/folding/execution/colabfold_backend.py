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

"""ColabFold folding backend: checked ``colabfold_batch`` subprocess with direct A3M intake.

This backend runs the harvested ``colabfold_batch`` invocation as a checked
subprocess, takes direct A3M input through the prepared-input
boundary, parameterizes every container-internal path, and emits
the canonical ``-model_v1.pdb`` / ``-meta_v1.json`` pair with full raw scores
through the backend-neutral Step-0 emitter helpers.

Port Baseline: origin/site-branch @ 0a3b80a
``folding/colabfold-slurm/python/colabfold_runner.py`` (``run_colabfold_batch``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bspp.orchestration.contract.runspec import VALID_TOOL_USED

from .chain_manifest import ChainManifest, ambiguous_chain_manifest_metadata, resolve_tool_used
from .emitter_support import atomic_write_bytes, atomic_write_text, canonical_pair_names, serialize_scores_json
from .models import FoldingResult, PreparedInput, ProteinTarget, StructurePrediction

COLABFOLD_TOOL_USED: str = VALID_TOOL_USED[0]


@dataclass(frozen=True)
class ColabFoldConfig:
    """Container-internal paths and parameters for one ``colabfold_batch`` run."""

    weights_dir: Path
    msa_cache_dir: Path
    structures_dir: Path
    model_type: str = "alphafold2_multimer_v3"
    num_recycle: int = 3
    num_models: int = 5
    num_seeds: int = 1
    gpu_id: int = 0


@dataclass(frozen=True)
class CommandSpec:
    """A fully-rendered ``colabfold_batch`` argv plus its environment overrides.

    ``env`` holds only the backend-specific overrides; the execution seam
    overlays them on a copy of the inherited process environment.
    """

    argv: tuple[str, ...]
    env: dict[str, str]


class Runner(Protocol):
    """Callable collaborator that executes one argv with a complete child environment.

    The mapping passed as ``env`` is the inherited process environment with the
    backend-specific overrides already applied, suitable for ``subprocess.run``.
    """

    def __call__(self, argv: Sequence[str], *, env: Mapping[str, str]) -> subprocess.CompletedProcess[bytes]: ...


def build_colabfold_command(config: ColabFoldConfig, a3m_path: Path, output_dir: Path) -> CommandSpec:
    """Assemble the deterministic ``colabfold_batch`` argv and environment overrides.

    The function is pure and config-only: every path is parameterized and
    rendered verbatim, with no hardcoded container-internal literals. The
    returned ``env`` holds only the backend-specific overrides;
    ``ColabFoldBackend.run`` overlays them on the inherited environment.
    """

    argv = (
        "colabfold_batch",
        f"--model-type={config.model_type}",
        "--data",
        str(config.weights_dir),
        f"--num-recycle={config.num_recycle}",
        f"--num-models={config.num_models}",
        f"--num-seeds={config.num_seeds}",
        "--skip-output",
        "msa,plots,pae_json",
        str(a3m_path),
        str(output_dir),
    )
    env = {
        "COLABFOLD_CONFIG": str(config.weights_dir),
        "CUDA_VISIBLE_DEVICES": str(config.gpu_id),
        "JAX_COMPILATION_CACHE_DIR": str(config.structures_dir / "jax_cache"),
    }
    return CommandSpec(argv=argv, env=env)


def stage_canonical_pair(
    model_entity_id: str,
    raw_pdb_path: Path,
    raw_json_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Publish the harvested raw pair to canonical names and re-serialize scores.

    The raw PDB bytes are preserved exactly and atomically written to the
    canonical structure filename at mode ``0644``. The raw scores JSON is
    re-serialized through ``serialize_scores_json`` so the emitted bytes are
    deterministic; ``plddt``/``pae``/``max_pae`` are required, ``ptm``/``iptm``
    are optional, and unknown keys are preserved in extras. The raw PDB is
    removed only after both canonical members have been published successfully.
    """

    structure_name, scores_name = canonical_pair_names(model_entity_id)
    structure_path = output_dir / structure_name
    scores_path = output_dir / scores_name

    raw_pdb_bytes = raw_pdb_path.read_bytes()

    raw = json.loads(raw_json_path.read_text(encoding="utf-8"))
    named = {"plddt", "pae", "max_pae", "ptm", "iptm"}
    extras = {key: value for key, value in raw.items() if key not in named and key != "schema_version"}
    payload = serialize_scores_json(
        plddt=raw["plddt"],
        pae=raw["pae"],
        max_pae=raw["max_pae"],
        ptm=raw.get("ptm"),
        iptm=raw.get("iptm"),
        extras=extras,
    )

    atomic_write_bytes(structure_path, raw_pdb_bytes)
    atomic_write_text(scores_path, payload)
    raw_pdb_path.unlink()
    return structure_path, scores_path


def _resolve_rank_one_pair(target_id: str, model_type: str, output_dir: Path) -> tuple[Path, Path]:
    """Resolve the raw rank-1 PDB/JSON pair for one ColabFold target.

    The harvested filenames carry an actual ``model_<n>_seed_<m>`` suffix that
    is not always ``model_1_seed_000``, so this resolver discovers the exact
    suffix rather than assuming it. It requires exactly one PDB candidate and
    its exactly matching scores JSON; zero, multiple, or unmatched candidates
    fail closed with a ``ValueError``. The first glob result is never picked
    silently.
    """

    pdb_re = re.compile(
        rf"^{re.escape(target_id)}_unrelaxed_rank_001_{re.escape(model_type)}_model_(\d+)_seed_(\d+)\.pdb$"
    )
    json_re = re.compile(
        rf"^{re.escape(target_id)}_scores_rank_001_{re.escape(model_type)}_model_(\d+)_seed_(\d+)\.json$"
    )

    pdb_by_suffix: dict[tuple[str, str], Path] = {}
    json_by_suffix: dict[tuple[str, str], Path] = {}
    try:
        entries = sorted(output_dir.iterdir(), key=lambda entry: entry.name)
    except OSError as exc:
        msg = f"cannot inspect ColabFold output directory {output_dir}: {exc}"
        raise ValueError(msg) from exc

    for entry in entries:
        try:
            is_file = entry.is_file()
        except OSError as exc:
            msg = f"cannot inspect ColabFold output candidate {entry}: {exc}"
            raise ValueError(msg) from exc
        if not is_file:
            continue
        match = pdb_re.match(entry.name)
        if match is not None:
            pdb_by_suffix[(match.group(1), match.group(2))] = entry
            continue
        match = json_re.match(entry.name)
        if match is not None:
            json_by_suffix[(match.group(1), match.group(2))] = entry

    if not pdb_by_suffix and not json_by_suffix:
        raise ValueError(f"no rank-1 raw pair found for target {target_id!r} in {output_dir}")
    if len(pdb_by_suffix) != 1 or len(json_by_suffix) != 1:
        raise ValueError(f"multiple rank-1 raw candidates found for target {target_id!r} in {output_dir}")

    pdb_key = next(iter(pdb_by_suffix))
    json_key = next(iter(json_by_suffix))
    if pdb_key != json_key:
        raise ValueError(f"unmatched rank-1 raw pair for target {target_id!r} in {output_dir}")

    return pdb_by_suffix[pdb_key], json_by_suffix[json_key]


class ColabFoldBackend:
    """Run one target through ``colabfold_batch`` and emit the canonical pair."""

    name = "colabfold"

    def __init__(
        self,
        config: ColabFoldConfig,
        chain_manifest: ChainManifest | None = None,
        *,
        runner: Runner | None = None,
    ) -> None:
        self.config = config
        self._chain_manifest = chain_manifest
        self._runner = runner or subprocess.run

    def run(self, target: ProteinTarget, prepared: PreparedInput, output_dir: Path) -> FoldingResult:
        if self._chain_manifest is None:
            raise ValueError("ColabFoldBackend requires a ChainManifest")

        tool_used = resolve_tool_used(self._chain_manifest, target.target_id, tool_used=COLABFOLD_TOOL_USED)
        if tool_used is None:
            return FoldingResult(backend=self.name, predictions=(), metadata=ambiguous_chain_manifest_metadata())

        a3m_path = prepared.alignment_dir / f"{target.target_id}.a3m"
        output_dir.mkdir(parents=True, exist_ok=True)
        spec = build_colabfold_command(self.config, a3m_path, output_dir)

        # Overlay the backend overrides on a copy of the inherited environment:
        # ``subprocess.run(env=...)`` replaces the child environment wholesale,
        # which would discard an activated Pixi/Conda PATH and inherited CUDA/JAX
        # settings needed to resolve and run the unqualified ``colabfold_batch``
        # executable. No inherited variable is deliberately removed.
        env = {**os.environ, **spec.env}
        result = self._runner(spec.argv, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"colabfold_batch exited with returncode {result.returncode}")

        raw_pdb, raw_json = _resolve_rank_one_pair(target.target_id, self.config.model_type, output_dir)
        structure_path, scores_path = stage_canonical_pair(target.target_id, raw_pdb, raw_json, output_dir)

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


__all__ = [
    "COLABFOLD_TOOL_USED",
    "ColabFoldBackend",
    "ColabFoldConfig",
    "CommandSpec",
    "Runner",
    "build_colabfold_command",
    "stage_canonical_pair",
]
