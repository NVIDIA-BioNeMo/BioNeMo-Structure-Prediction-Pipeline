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

"""BioIR score normalization and earliest-complete full-score capture.

Port Baseline: frozen reference pipeline
  - ``_numeric_values`` / ``quality_from_bioir_scores`` (src/afdb_pipeline/bioir_backend.py)

``quality_from_bioir_scores`` is a verbatim port of the harvested registry
normalization and is intentionally tolerant: it flattens whatever ``plddt`` /
``pae`` values are present and never raises.  Strictness lives only in
:func:`capture_bioir_full_scores`, which reads the unmodified full ``get_scores``
result and fails closed on missing/empty/non-finite arrays so the
canonical scores JSON always carries the complete ``plddt`` / ``pae`` / ``max_pae``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, SupportsFloat, cast

from .errors import FoldingBackendError

__all__ = ["capture_bioir_full_scores", "quality_from_bioir_scores"]


def _numeric_values(value: Any) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    stack = list(value)
    result: list[float] = []
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result.append(float(item))
    return result


def quality_from_bioir_scores(scores: Mapping[str, Any]) -> dict[str, Any]:
    plddt = _numeric_values(scores.get("plddt"))
    pae = _numeric_values(scores.get("pae"))
    finite_plddt = [value for value in plddt if math.isfinite(value)]
    finite_pae = [value for value in pae if math.isfinite(value)]

    def scalar(name: str) -> float | None:
        value = scores.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            return value if math.isfinite(value) else None
        return None

    ptm = scalar("ptm")
    iptm = scalar("iptm")
    emitted_max_pae = scalar("max_pae")
    mean_plddt = sum(finite_plddt) / len(finite_plddt) if finite_plddt else None
    ranking_confidence: float | None
    if ptm is not None and iptm is not None:
        ranking_confidence = 0.8 * iptm + 0.2 * ptm
        ranking_source = "0.8_iptm_plus_0.2_ptm"
    else:
        ranking_confidence = mean_plddt / 100.0 if mean_plddt is not None else None
        ranking_source = "mean_plddt_fraction"
    scalar_values = [value for value in (ptm, iptm, emitted_max_pae) if value is not None]
    output_has_nan = any(not math.isfinite(value) for value in (*plddt, *pae, *scalar_values))
    return {
        "ranking_confidence": ranking_confidence,
        "ranking_confidence_source": ranking_source,
        "mean_plddt": mean_plddt,
        "min_plddt": min(finite_plddt) if finite_plddt else None,
        "max_plddt": max(finite_plddt) if finite_plddt else None,
        "plddt_above_70": (sum(value >= 70.0 for value in finite_plddt) / len(finite_plddt) if finite_plddt else None),
        "ptm": ptm,
        "iptm": iptm,
        "max_pae": max(finite_pae) if finite_pae else emitted_max_pae,
        "output_has_nan": output_has_nan,
        "residue_count": len(plddt),
        "source": "bioir_scores_json",
    }


def capture_bioir_full_scores(
    scores: Mapping[str, Any],
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...], float, float | None, float | None]:
    """Capture the unmodified full ``get_scores`` result.

    Returns ``(plddt, pae, max_pae, ptm, iptm)``.  ``plddt`` and ``pae`` are the
    complete arrays (``pae`` stays 2-D), ``max_pae`` is derived only from the
    ``pae`` matrix, and ``ptm``/``iptm`` are optional scalars.  Missing, empty,
    or non-finite arrays raise :class:`FoldingBackendError`; scores are never
    reconstructed from summaries.  ``pae`` must be square: its row count and
    every row width must equal the ``plddt`` residue count, so a ragged or
    non-square matrix can never become a canonical score artifact.
    """

    plddt_raw = scores.get("plddt")
    if plddt_raw is None:
        raise FoldingBackendError("BioIR scores are missing 'plddt'")
    plddt = _to_plddt_tuple(plddt_raw)

    pae_raw = scores.get("pae")
    if pae_raw is None:
        raise FoldingBackendError("BioIR scores are missing 'pae'")
    pae = _to_pae_tuple(pae_raw)

    if len(pae) != len(plddt):
        raise FoldingBackendError(f"BioIR 'pae' has {len(pae)} rows but 'plddt' has {len(plddt)} residues")
    for row_index, row in enumerate(pae):
        if len(row) != len(plddt):
            raise FoldingBackendError(
                f"BioIR 'pae' row {row_index} has {len(row)} columns but 'plddt' has {len(plddt)} residues"
            )

    ptm = _optional_finite_scalar(scores.get("ptm"), "ptm")
    iptm = _optional_finite_scalar(scores.get("iptm"), "iptm")
    max_pae = max(value for row in pae for value in row)
    return plddt, pae, max_pae, ptm, iptm


def _iter_items(value: object, name: str) -> Iterable[object]:
    if isinstance(value, (str, bytes, Mapping)):
        raise FoldingBackendError(f"BioIR '{name}' must be a numeric array")
    try:
        return iter(cast("Iterable[object]", value))
    except TypeError as exc:
        raise FoldingBackendError(f"BioIR '{name}' must be a numeric array") from exc


def _finite_float(item: object, name: str) -> float:
    if isinstance(item, (bool, str, bytes)):
        raise FoldingBackendError(f"BioIR '{name}' must contain only finite numbers")
    try:
        value = float(cast("str | SupportsFloat", item))
    except (TypeError, ValueError) as exc:
        raise FoldingBackendError(f"BioIR '{name}' must contain only finite numbers") from exc
    if not math.isfinite(value):
        raise FoldingBackendError(f"BioIR '{name}' must contain only finite numbers")
    return value


def _to_plddt_tuple(value: object) -> tuple[float, ...]:
    result = tuple(_finite_float(item, "plddt") for item in _iter_items(value, "plddt"))
    if not result:
        raise FoldingBackendError("BioIR 'plddt' is empty")
    return result


def _to_pae_tuple(value: object) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for row in _iter_items(value, "pae"):
        row_tuple = tuple(_finite_float(item, "pae") for item in _iter_items(row, "pae"))
        if not row_tuple:
            raise FoldingBackendError("BioIR 'pae' rows must be non-empty")
        rows.append(row_tuple)
    if not rows:
        raise FoldingBackendError("BioIR 'pae' is empty")
    return tuple(rows)


def _optional_finite_scalar(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise FoldingBackendError(f"BioIR '{name}' must be a finite number")
    try:
        result = float(cast("str | SupportsFloat", value))
    except (TypeError, ValueError) as exc:
        raise FoldingBackendError(f"BioIR '{name}' must be a finite number") from exc
    if not math.isfinite(result):
        raise FoldingBackendError(f"BioIR '{name}' must be a finite number")
    return result
