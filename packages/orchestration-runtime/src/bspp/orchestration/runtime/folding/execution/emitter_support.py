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

"""Backend-neutral canonical-pair emitter helpers.

These helpers live on the runtime side of the fold-to-postprocessing seam and
are deliberately backend-neutral: every folding backend emits the
same canonical ``-model_v1.pdb`` / ``-meta_v1.json`` pair, selects the same
leaked-homodimer ``tool_used``, and serializes the same full-scores payload
with deterministic half-even rounding. No backend-specific raw-score capture
points belong here.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from bspp.orchestration.contract.model_identity import (
    CANONICAL_META_SUFFIX,
    CANONICAL_MODEL_SUFFIX,
    is_homodimer_model_entity_id,
    is_pdb_assembly_model_entity_id,
    normalize_model_entity_id,
    parse_compound_model_entity_id,
    parse_human_string_model_entity_id,
)
from bspp.orchestration.contract.prediction_pair import PredictionScoresPayload
from bspp.orchestration.contract.runspec import VALID_TOOL_USED

from .rounding import round_float

__all__ = [
    "atomic_write_bytes",
    "atomic_write_text",
    "canonical_pair_names",
    "select_tool_used",
    "serialize_scores_json",
]


def canonical_pair_names(model_entity_id: str) -> tuple[str, str]:
    """Return the canonical structure/scores filenames for a model entity ID.

    The ID is normalized through the shared contract identity surface, so
    ``AF_``, ``AFDB_``, compound, and harvested ``_model`` forms all collapse
    to the same canonical pair.
    """

    normalized = normalize_model_entity_id(model_entity_id)
    return (
        f"{normalized}{CANONICAL_MODEL_SUFFIX}",
        f"{normalized}{CANONICAL_META_SUFFIX}",
    )


def select_tool_used(
    model_entity_id: str,
    *,
    leaked_homodimer: bool,
    homodimer_tool_used: str,
    heterodimer_tool_used: str,
) -> str:
    """Select the ``tool_used`` provenance string for a prediction target.

    A homodimer form takes ``homodimer_tool_used``. A compound takes
    ``homodimer_tool_used`` when ``leaked_homodimer`` is true, otherwise
    ``heterodimer_tool_used``. Both tool strings are validated against the
    frozen ``VALID_TOOL_USED`` set before any identity work, giving
    deterministic error precedence.

    HumanSTRING homo roots select the homodimer tool; hetero roots use the
    same explicit leaked flag as legacy compounds. Their IDs remain unchanged.

    Leaked-homodimer classification is deliberately NOT derived from the
    model-ID components: the handoff contract defines it by UniProt-accession
    equality through the external chain manifest (handoff plan section 3.4),
    which this seam does not consume. The caller owns the manifest and passes
    its resolved classification explicitly, so no ID-minting invariant is
    relied upon.
    """

    if homodimer_tool_used not in VALID_TOOL_USED:
        msg = f"homodimer_tool_used must be one of {VALID_TOOL_USED!r}; got {homodimer_tool_used!r}"
        raise ValueError(msg)
    if heterodimer_tool_used not in VALID_TOOL_USED:
        msg = f"heterodimer_tool_used must be one of {VALID_TOOL_USED!r}; got {heterodimer_tool_used!r}"
        raise ValueError(msg)
    if not isinstance(leaked_homodimer, bool):
        msg = f"leaked_homodimer must be a bool; got {leaked_homodimer!r}"
        raise ValueError(msg)
    if is_homodimer_model_entity_id(model_entity_id):
        return homodimer_tool_used
    human_accessions = parse_human_string_model_entity_id(model_entity_id)
    if human_accessions is not None:
        if len(human_accessions) == 1:
            return homodimer_tool_used
        return homodimer_tool_used if leaked_homodimer else heterodimer_tool_used
    if is_pdb_assembly_model_entity_id(model_entity_id):
        return homodimer_tool_used if leaked_homodimer else heterodimer_tool_used
    parse_compound_model_entity_id(model_entity_id)
    return homodimer_tool_used if leaked_homodimer else heterodimer_tool_used


def serialize_scores_json(
    *,
    plddt: Sequence[float | int],
    pae: Sequence[Sequence[float | int]],
    max_pae: float | int,
    ptm: float | int | None = None,
    iptm: float | int | None = None,
    extras: Mapping[str, object] | None = None,
    decimals: int = 2,
    scalar_decimals: int | None = None,
) -> str:
    """Serialize a full-scores payload with deterministic half-even rounding.

    ``plddt`` and ``pae`` always use ``decimals``. The scalar fields
    (``max_pae``, ``ptm``, ``iptm``) use ``scalar_decimals`` when supplied,
    otherwise ``decimals``. Unknown keys in ``extras`` are preserved verbatim by
    the payload's open-extras mapping. The returned JSON preserves
    the payload's insertion order, so bytes are deterministic for a given input.
    """

    scalar_decimals = decimals if scalar_decimals is None else scalar_decimals
    payload = PredictionScoresPayload(
        plddt=tuple(round_float(value, decimals) for value in plddt),
        pae=tuple(tuple(round_float(value, decimals) for value in row) for row in pae),
        max_pae=round_float(max_pae, scalar_decimals),
        ptm=None if ptm is None else round_float(ptm, scalar_decimals),
        iptm=None if iptm is None else round_float(iptm, scalar_decimals),
        extras={} if extras is None else extras,
    )
    return json.dumps(payload.to_mapping())


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    """Atomically replace ``path`` with ``payload`` at an explicit file mode.

    The bytes are written to a same-directory temporary file whose descriptor
    mode is set explicitly before publication, flushed and closed, then
    ``os.replace``d into place. The temporary file is removed on any failure.
    This gives every backend the same canonical-file mode instead of
    ``mkstemp``'s backend-specific ``0600`` result.
    """

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def atomic_write_text(path: Path, text: str, *, mode: int = 0o644) -> None:
    """Atomically replace ``path`` with UTF-8 ``text`` at an explicit file mode."""

    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)
