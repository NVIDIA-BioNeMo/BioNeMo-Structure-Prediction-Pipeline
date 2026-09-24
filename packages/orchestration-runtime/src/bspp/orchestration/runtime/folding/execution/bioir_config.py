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

"""BioIR OpenFold model/settings shapes and checkpoint-source helpers.

Port Baseline: frozen reference pipeline src/afdb_pipeline/cluster_config.py
(``OpenFoldModelSettings`` / ``OpenFoldSettings``, restricted to the BioIR-relevant fields)
and src/afdb_pipeline/bioir_backend.py (``bioir_model_source`` / ``bioir_checkpoint_env``).
Only the frozen field shapes and the two source-derivation helpers are ported; no
``ClusterProfile`` machinery, loader, or BioIR/torch import lives in this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import FoldingBackendError

__all__ = ["OpenFoldModelSettings", "OpenFoldSettings", "bioir_checkpoint_env", "bioir_model_source"]

_MULTIMER_PRESET = re.compile(r"^model_([1-5])_multimer_v3$")
_MULTIMER_SOURCE = re.compile(r"^alphafold2_multimer_([1-5])$")


@dataclass(frozen=True)
class OpenFoldModelSettings:
    """One independently executable OpenFold model/parameter/seed combination."""

    model_id: str
    model_preset: str
    parameter_file: str
    seed: int = 0
    # BioIR names the five AlphaFold2 multimer checkpoints
    # ``alphafold2_multimer_1`` ... ``_5``.  The field is optional so legacy
    # OpenFold profiles remain valid and BioIR can derive the name from the
    # existing ``model_<N>_multimer_v3`` preset.
    model_source: str | None = None


@dataclass(frozen=True)
class OpenFoldSettings:
    # The singular fields remain part of the frozen profile for compatibility
    # with profiles created before multi-model execution was introduced. New
    # profiles should use ``models``; these fields mirror its first entry.
    model_preset: str = "model_1_multimer_v3"
    parameter_file: str = "params_model_1_multimer_v3.pt"
    seed: int = 0
    models: tuple[OpenFoldModelSettings, ...] = ()
    compact_scores: bool = True
    continue_on_error: bool = True
    profile_inference: bool = False
    # BioIR counts the initial model execution as a cycle, so five matches
    # OpenFold/ColabFold ``max_recycling_iters=4``.
    recycling_steps: int | None = 5
    backend: str = "bioir"
    # Database-scale BioIR defaults validated by the controlled benchmark.
    # Paired inputs and deeper MSAs remain opt-in for experiments that accept
    # their additional runtime and potentially different interface scores.
    use_paired_msa: bool = False
    max_non_query_msa_rows: int = 5_000


def bioir_model_source(model: OpenFoldModelSettings) -> str:
    """Return the BioIR registry key for a configured OpenFold model."""

    if model.model_source:
        return model.model_source
    match = _MULTIMER_PRESET.fullmatch(model.model_preset)
    if match is None:
        raise FoldingBackendError("BioIR model_source is required when model_preset is not model_<1-5>_multimer_v3")
    return f"alphafold2_multimer_{match.group(1)}"


def bioir_checkpoint_env(model_source: str) -> str:
    """Return the checkpoint environment variable name for a BioIR model source."""

    if model_source == "openfold2_ptm_1":
        return "OPENFOLD2_PTM_1_CKPT"
    match = _MULTIMER_SOURCE.fullmatch(model_source)
    if match is None:
        raise FoldingBackendError(f"unsupported BioIR OpenFold2 model source for this pipeline: {model_source}")
    return f"ALPHAFOLD2_MULTIMER_{match.group(1)}_CKPT"
