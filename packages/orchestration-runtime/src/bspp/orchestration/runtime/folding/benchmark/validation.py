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

"""validate_run core: identity, quality, C-alpha structure, and evidence output.

Track-C port of the run-side half of ``frozen reference pipeline``
(``src/afdb_pipeline/validation.py``). This module implements the tracer-bullet
vertical: a completed run directory plus a checksum-pinned validation suite
(e05s02) and a canonical-pair index (e05s01) yields a per-case validity verdict
and writes bounded validation evidence (``validation.parquet`` + ``summary.json``).

Track-C deviations from the reference (authoritative):
- The reference pipeline's registry/MSA/preprocess discovery is replaced by the canonical-pair
  index identity check.
- The reference pipeline's post-processing check is dropped entirely: no post-processing
  snapshot class and no post-processing file/attribute reads exist in this
  module.
- The raw-scores quality gate enforces finite output and ``require_no_nan``
  only; it never invents RMSD/confidence thresholds. The shipped
  suite's ``thresholds`` carry only the structure-domain ``ca_coverage`` key,
  which is enforced by the structure check, so the quality gate applies no
  numeric pass threshold.
- The C-alpha structure check is a simplified Kabsch SVD alignment over
  order-matched, chain-mapped C-alpha coordinates; coverage is enforced against
  ``ca_coverage`` (default 0.7) but ``ca_rmsd`` is reported and never gated.
- Evidence output is bounded: ``validation.parquet`` stays
  cluster-resident while ``summary.json``-sized data is returned to the
  workstation.
- When ``expected_fingerprint`` is supplied, the suite's declared fingerprint
  must equal it (i.e. the verified corpus fingerprint) or loading fails closed;
  an independently supplied suite can never validate the wrong corpus.
- Index ``structure_path``/``scores_path`` may be relative or absolute, but
  must resolve beneath ``run_dir``. Suite ``reference_structure`` must remain
  relative and resolve beneath ``corpus_dir``. Escaping paths, including
  symlink and parent-traversal escapes, fail closed before artifact reads.
- An explicit ``chain_map`` must cover every predicted chain and every
  reference chain exactly once (one-to-one); omissions, extras, and duplicate
  reference mappings fail closed so a wrong complex composition can never pass.
- A BioIR ``unpaired_paired`` target with a different ordered hash can match
  only the deterministic ColabFold first-occurrence grouping of verified
  original corpus chains. The original and observed hashes remain intact;
  the proven permutation composes the suite's explicit chain map for actual
  coordinate comparison and is recorded in summary-only diagnostic metadata.
- Fail-closed: an absent/invalid index or suite raises at the loader; a missing
  target, unproved ``sequence_sha256`` mismatch, malformed scores JSON, missing
  reference structure, or sha256 mismatch yields a per-case failure and never a
  synthesized pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, TypeGuard, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bspp.orchestration.contract import (
    CURRENT_CONTRACT_SCHEMA_VERSION,
    PredictionPair,
    prediction_pair_from_mapping,
)
from bspp.orchestration.runtime.folding.benchmark.chain_order import (
    ColabFoldChainOrderResolver,
    VerifiedChainOrder,
)
from bspp.orchestration.runtime.folding.benchmark.index import (
    CanonicalPairIndex,
    CanonicalPairIndexEntry,
    load_canonical_pair_index,
)
from bspp.orchestration.runtime.folding.benchmark.suite import (
    ValidationCase,
    ValidationSuite,
    load_validation_suite,
)


def _is_finite(value: object) -> TypeGuard[float | int]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _is_number(value: object) -> TypeGuard[float | int]:
    """A finite-or-NaN JSON number (bool excluded)."""
    return _is_finite(value) or _is_nan(value)


def _extract_number_list(value: object) -> list[float] | None:
    """Extract a non-empty JSON list of finite-or-NaN numbers, else ``None``."""
    if not isinstance(value, list) or not value:
        return None
    numbers: list[float] = []
    for item in value:
        if not _is_number(item):
            return None
        numbers.append(float(item))
    return numbers


def _extract_number_matrix(value: object) -> list[list[float]] | None:
    """Extract a JSON matrix of finite-or-NaN numbers, else ``None``."""
    if not isinstance(value, list):
        return None
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list):
            return None
        row_numbers: list[float] = []
        for item in row:
            if not _is_number(item):
                return None
            row_numbers.append(float(item))
        rows.append(row_numbers)
    return rows


def _contained_artifact_path(root: Path, raw_path: str, *, allow_absolute: bool = False) -> Path:
    """Resolve an artifact path beneath ``root``, failing closed on escape.

    Corpus references remain relative-only. Canonical prediction indices may
    carry absolute paths from the executor, accepted only when the resolved
    path remains beneath the resolved run root. Resolution catches symlink
    and parent-traversal escapes before any artifact contents are read.
    """
    if Path(raw_path).is_absolute() and not allow_absolute:
        raise ValueError(f"artifact path must be relative: {raw_path!r}")
    resolved = (root / raw_path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes its root: {raw_path!r}")
    return resolved


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 hex digest of a file via chunked reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_pdb_ca(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    """Parse the first-model C-alpha coordinates of a PDB file, keyed by chain id.

    Pure-stdlib port of the reference pipeline's ``_ca_chains`` with the return type simplified
    from ``dict[str, dict[tuple[str, str], np.ndarray]]`` to an order-preserving
    ``dict[str, list[tuple[float, float, float]]]``. Only the first ``MODEL``
    block is read; if the file has no ``MODEL`` record every ``ATOM`` is read.
    Only ``ATOM`` records whose atom name is ``CA`` (altloc blank or ``A``) are
    kept, residues are deduplicated by ``(chain, resseq, icode)``, and records
    with malformed coordinates are skipped.
    """
    chains: dict[str, list[tuple[float, float, float]]] = {}
    seen: set[tuple[str, str, str]] = set()
    in_first_model = True
    saw_model = False
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                if saw_model:
                    in_first_model = False
                saw_model = True
                continue
            if line.startswith("ENDMDL") and saw_model:
                break
            if not in_first_model or not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA" or line[16:17] not in {" ", "A"}:
                continue
            chain = line[21:22].strip() or "_"
            residue = (chain, line[22:26].strip(), line[26:27])
            if residue in seen:
                continue
            seen.add(residue)
            try:
                coordinate = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            chains.setdefault(chain, []).append(coordinate)
    return chains


class CaMetrics(TypedDict):
    """The six C-alpha alignment metrics produced by :func:`ca_metrics`."""

    predicted_ca_atoms: int
    reference_ca_atoms: int
    matched_ca_atoms: int
    ca_match_mode: str
    ca_coverage: float
    ca_rmsd: float


def ca_metrics(predicted: np.ndarray, reference: np.ndarray) -> CaMetrics:
    """Align two C-alpha coordinate arrays with a Kabsch SVD and return metrics.

    ``predicted`` and ``reference`` are ``(N, 3)`` / ``(M, 3)`` coordinate
    arrays; chain mapping and concatenation happen in the caller. Atoms are
    matched by order (first ``min(N, M)`` rows), so ``ca_coverage`` is
    ``matched / max(N, M)``. Identical inputs yield ``ca_rmsd == 0.0`` and
    ``ca_coverage == 1.0``; a zero-atom input raises ``ValueError``. There is
    deliberately no ``< 3`` atom guard: Kabsch SVD is numerically valid down to
    a single point. Sub-``1e-12`` Å residuals are clamped to ``0.0`` to strip
    SVD round-off from identical/rigid-transform inputs.
    """
    predicted_count = predicted.shape[0]
    reference_count = reference.shape[0]
    if predicted_count == 0 or reference_count == 0:
        raise ValueError("predicted or reference structure contains no C-alpha atoms")
    matched_count = min(predicted_count, reference_count)
    predicted_array = predicted[:matched_count].astype(np.float64)
    reference_array = reference[:matched_count].astype(np.float64)
    predicted_centered = predicted_array - predicted_array.mean(axis=0)
    reference_centered = reference_array - reference_array.mean(axis=0)
    left, _, right = np.linalg.svd(predicted_centered.T @ reference_centered)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        right[-1, :] *= -1
        rotation = left @ right
    aligned = predicted_centered @ rotation
    ca_rmsd = float(np.sqrt(np.mean(np.sum((aligned - reference_centered) ** 2, axis=1))))
    if ca_rmsd < 1e-12:
        ca_rmsd = 0.0
    ca_coverage = matched_count / max(predicted_count, reference_count)
    ca_match_mode = "full" if predicted_count == reference_count else "truncated"
    return CaMetrics(
        predicted_ca_atoms=predicted_count,
        reference_ca_atoms=reference_count,
        matched_ca_atoms=matched_count,
        ca_match_mode=ca_match_mode,
        ca_coverage=ca_coverage,
        ca_rmsd=ca_rmsd,
    )


_VALIDATION_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("target_id", pa.string(), nullable=False),
        pa.field("identity_valid", pa.bool_(), nullable=False),
        pa.field("quality_valid", pa.bool_(), nullable=False),
        pa.field("structure_valid", pa.bool_(), nullable=False),
        pa.field("ca_coverage", pa.float64(), nullable=True),
        pa.field("ca_rmsd", pa.float64(), nullable=True),
        pa.field("output_has_nan", pa.bool_(), nullable=False),
        pa.field("passed", pa.bool_(), nullable=False),
    ]
)

_PARQUET_COLUMNS = (
    "target_id",
    "identity_valid",
    "quality_valid",
    "structure_valid",
    "ca_coverage",
    "ca_rmsd",
    "output_has_nan",
    "passed",
)


def _project_parquet_row(row: Mapping[str, object]) -> dict[str, object]:
    """Project a per-case verdict dict onto the fixed eight parquet columns."""
    return {column: row.get(column) for column in _PARQUET_COLUMNS}


def _write_validation_evidence(
    output_dir: Path,
    rows: list[dict[str, object]],
    suite: ValidationSuite,
) -> dict[str, object]:
    """Write ``validation.parquet`` + ``summary.json`` and return the summary.

    Both files are written atomically (temp file in the same directory followed
    by ``os.replace``). The parquet is the fixed eight-column schema (one row
    per case); the summary is a superset carrying every per-case verdict plus
    the suite identity and pass counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    passed_count = sum(1 for row in rows if row.get("passed") is True)
    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": suite.dataset_id,
        "fingerprint": suite.fingerprint,
        "case_count": len(rows),
        "passed_count": passed_count,
        "cases": rows,
    }

    parquet_path = output_dir / "validation.parquet"
    parquet_fd, parquet_temporary = tempfile.mkstemp(dir=str(output_dir), suffix=".parquet.tmp")
    os.close(parquet_fd)
    try:
        projected = [_project_parquet_row(row) for row in rows]
        with pq.ParquetWriter(
            parquet_temporary,
            _VALIDATION_PARQUET_SCHEMA,
            compression="zstd",
            version="2.6",
        ) as writer:
            writer.write_table(pa.Table.from_pylist(projected, schema=_VALIDATION_PARQUET_SCHEMA))
        os.replace(parquet_temporary, parquet_path)
    finally:
        Path(parquet_temporary).unlink(missing_ok=True)

    summary_path = output_dir / "summary.json"
    summary_fd, summary_temporary = tempfile.mkstemp(dir=str(output_dir), suffix=".json.tmp")
    os.close(summary_fd)
    try:
        with open(summary_temporary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(summary_temporary, summary_path)
    finally:
        Path(summary_temporary).unlink(missing_ok=True)

    return summary


def validate_run(
    run_dir: Path,
    suite_path: Path,
    index_path: Path,
    corpus_dir: Path,
    *,
    output_dir: Path | None = None,
    expected_fingerprint: str | None = None,
) -> dict[str, object]:
    """Validate a completed folding run against a suite and canonical-pair index.

    ``corpus_dir`` is threaded through to each case validator for the
    reference-structure check. ``output_dir`` defaults to ``run_dir/validation``
    and receives the bounded evidence pair (``validation.parquet`` +
    ``summary.json``). The returned dict is exactly the summary written to
    ``summary.json``. An absent/invalid suite or index raises at the loader and
    never reaches the per-case checks. When ``expected_fingerprint`` is given
    (the verified corpus fingerprint from the pinned fetch), a suite whose
    declared fingerprint differs raises at the loader, so an independently
    supplied stale or unrelated suite can never false-fail or false-pass.
    """
    suite = load_validation_suite(suite_path)
    if expected_fingerprint is not None and suite.fingerprint != expected_fingerprint:
        raise ValueError(
            "validation suite fingerprint does not match the verified corpus fingerprint: "
            f"suite={suite.fingerprint!r}, corpus={expected_fingerprint!r}"
        )
    index = load_canonical_pair_index(index_path)
    chain_order = ColabFoldChainOrderResolver(corpus_dir, expected_fingerprint)
    cases: list[dict[str, object]] = [
        _CaseValidator(case, index, run_dir, corpus_dir, chain_order=chain_order).run() for case in suite.cases
    ]
    resolved_output_dir = output_dir or (run_dir / "validation")
    return _write_validation_evidence(resolved_output_dir, cases, suite)


class _CaseValidator:
    """Per-case identity, raw-scores quality, and C-alpha structure validation."""

    def __init__(
        self,
        case: ValidationCase,
        index: CanonicalPairIndex,
        run_dir: Path,
        corpus_dir: Path,
        *,
        chain_order: ColabFoldChainOrderResolver | None = None,
    ) -> None:
        self.case = case
        self.index = index
        self.run_dir = run_dir
        self.corpus_dir = corpus_dir
        self._chain_order = chain_order
        self._verified_order: VerifiedChainOrder | None = None
        self._entry: CanonicalPairIndexEntry | None = None
        self._pair: PredictionPair | None = None
        self._raw_scores: Mapping[str, object] | None = None
        self._prediction_count = 0
        self._output_has_nan = False
        self._mean_plddt: float | None = None
        self._min_plddt: float | None = None
        self._max_plddt: float | None = None
        self._plddt_above_70: float | None = None
        self._max_pae: float | None = None
        self._ca_metrics: CaMetrics | None = None

    def run(self) -> dict[str, object]:
        identity_valid = self.identity_valid()
        quality_valid = self.quality_valid()
        structure_valid = self.structure_valid()
        result: dict[str, object] = {
            "target_id": self.case.target_id,
            "identity_valid": identity_valid,
            "quality_valid": quality_valid,
            "structure_valid": structure_valid,
            "output_has_nan": self._output_has_nan,
            "prediction_count": self._prediction_count,
            "mean_plddt": self._mean_plddt,
            "min_plddt": self._min_plddt,
            "max_plddt": self._max_plddt,
            "plddt_above_70": self._plddt_above_70,
            "max_pae": self._max_pae,
            "predicted_ca_atoms": None,
            "reference_ca_atoms": None,
            "matched_ca_atoms": None,
            "ca_match_mode": None,
            "ca_coverage": None,
            "ca_rmsd": None,
            "passed": identity_valid and quality_valid and structure_valid,
        }
        if self._ca_metrics is not None:
            metrics = self._ca_metrics
            result["predicted_ca_atoms"] = metrics["predicted_ca_atoms"]
            result["reference_ca_atoms"] = metrics["reference_ca_atoms"]
            result["matched_ca_atoms"] = metrics["matched_ca_atoms"]
            result["ca_match_mode"] = metrics["ca_match_mode"]
            result["ca_coverage"] = metrics["ca_coverage"]
            result["ca_rmsd"] = metrics["ca_rmsd"]
        if self._verified_order is not None:
            result["chain_order_identity"] = self._verified_order.to_summary()
        return result

    def identity_valid(self) -> bool:
        return self._load_pair()

    def quality_valid(self) -> bool:
        if self._raw_scores is None:
            return False
        return self._quality_from_raw_scores()

    def structure_valid(self) -> bool:
        """Verify the pinned reference and align C-alpha coordinates via Kabsch SVD.

        Fails closed (``False``, no metrics) on a missing/unreadable reference,
        a reference sha256 mismatch, an unresolved index entry, a malformed/
        missing PDB, an unresolvable chain map, or a Kabsch failure.
        ``ca_coverage`` is enforced against ``case.thresholds["ca_coverage"]``
        (default 0.7); ``ca_rmsd`` is reported but never gated.
        """
        entry = self._resolve_entry()
        if entry is None:
            return False
        try:
            reference_path = _contained_artifact_path(self.corpus_dir, self.case.reference_structure)
            digest = _sha256_file(reference_path)
        except (OSError, ValueError):
            return False
        if digest != self.case.reference_sha256:
            return False
        try:
            predicted_path = _contained_artifact_path(self.run_dir, entry.structure_path, allow_absolute=True)
            predicted = parse_pdb_ca(predicted_path)
            reference = parse_pdb_ca(reference_path)
            chain_map = self._resolve_chain_map(predicted, reference)
            predicted_array, reference_array = self._matched_arrays(predicted, reference, chain_map)
            metrics = ca_metrics(predicted_array, reference_array)
        except (OSError, ValueError, np.linalg.LinAlgError):
            return False
        self._ca_metrics = metrics
        coverage_threshold = self.case.thresholds.get("ca_coverage", 0.7)
        return metrics["ca_coverage"] >= coverage_threshold

    def _resolve_entry(self) -> CanonicalPairIndexEntry | None:
        """Resolve the index entry matching this case's target and sequence."""
        for candidate in self.index.entries:
            if candidate.target_id == self.case.target_id:
                if candidate.sequence_sha256 == self.case.sequence_sha256:
                    return candidate
                if self._chain_order is not None:
                    self._verified_order = self._chain_order.resolve(self.case, candidate)
                    if self._verified_order is not None:
                        return candidate
        return None

    def _resolve_chain_map(
        self,
        predicted: Mapping[str, list[tuple[float, float, float]]],
        reference: Mapping[str, list[tuple[float, float, float]]],
    ) -> dict[str, str]:
        """Resolve the predicted→reference chain map (explicit or by fallback)."""
        if self._verified_order is not None:
            chain_map = dict(self._verified_order.chain_map)
        elif self.case.chain_map:
            chain_map = dict(self.case.chain_map)
        elif set(predicted) == set(reference):
            chain_map = {chain: chain for chain in predicted}
        elif len(predicted) == len(reference) == 1:
            chain_map = {next(iter(predicted)): next(iter(reference))}
        else:
            raise ValueError("multichain structures require an explicit unambiguous chain_map")
        mapped_predicted = set(chain_map)
        mapped_reference = set(chain_map.values())
        if mapped_predicted != set(predicted) or mapped_reference != set(reference):
            raise ValueError(
                "chain_map must cover predicted and reference chains exactly, and refer to existing chains: "
                f"unmapped_predicted={sorted(set(predicted) - mapped_predicted)}, "
                f"unmapped_reference={sorted(set(reference) - mapped_reference)}, "
                f"unknown_predicted={sorted(mapped_predicted - set(predicted))}, "
                f"unknown_reference={sorted(mapped_reference - set(reference))}"
            )
        if len(mapped_reference) != len(chain_map):
            raise ValueError("chain_map must map reference chains one-to-one; a duplicate mapping was supplied")
        return chain_map

    @staticmethod
    def _matched_arrays(
        predicted: Mapping[str, list[tuple[float, float, float]]],
        reference: Mapping[str, list[tuple[float, float, float]]],
        chain_map: Mapping[str, str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate chain-mapped coordinate lists into ``(N, 3)`` arrays."""
        predicted_coordinates: list[tuple[float, float, float]] = []
        reference_coordinates: list[tuple[float, float, float]] = []
        for predicted_chain, reference_chain in chain_map.items():
            predicted_coordinates.extend(predicted[predicted_chain])
            reference_coordinates.extend(reference[reference_chain])
        return (
            np.asarray(predicted_coordinates, dtype=np.float64),
            np.asarray(reference_coordinates, dtype=np.float64),
        )

    def _load_pair(self) -> bool:
        """Resolve the canonical pair for this case and load it via the contract.

        Fails closed (``False``) on a missing target, a ``sequence_sha256``
        mismatch, an unreadable/malformed scores JSON, or a pair that the frozen
        ``prediction_pair_from_mapping`` loader rejects. The raw parsed scores
        mapping is retained even when the contract load fails so the quality
        gate can still observe NaN payloads.
        """
        entry = self._resolve_entry()
        if entry is None:
            return False
        try:
            scores_path = _contained_artifact_path(self.run_dir, entry.scores_path, allow_absolute=True)
            raw = json.loads(scores_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            return False
        if not isinstance(raw, dict):
            return False
        self._raw_scores = cast("Mapping[str, object]", raw)
        try:
            pair = prediction_pair_from_mapping(
                {
                    "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
                    "model_entity_id": entry.model_entity_id,
                    "tool_used": entry.tool_used,
                    "structure_path": entry.structure_path,
                    "scores_path": entry.scores_path,
                    "scores": self._raw_scores,
                }
            )
        except (ValueError, TypeError):
            return False
        self._entry = entry
        self._pair = pair
        self._prediction_count = 1
        return True

    def _quality_from_raw_scores(self) -> bool:
        """Derive raw-scores metrics and enforce the finite/no-NaN quality gate.

        Operates on the raw parsed JSON (not the contract-validated payload) so
        NaN is observable. No numeric pass threshold is applied: the shipped
        suite's ``thresholds`` carry only the structure-domain ``ca_coverage``
        key (enforced by the structure check), so the gate is finite output plus
        ``require_no_nan``.
        """
        raw = self._raw_scores
        if raw is None:
            return False
        plddt = _extract_number_list(raw.get("plddt"))
        if plddt is None:
            return False
        pae = _extract_number_matrix(raw.get("pae"))
        if pae is None:
            return False
        max_pae_value = raw.get("max_pae")
        if not _is_number(max_pae_value):
            return False
        max_pae = float(max_pae_value)

        output_has_nan = any(_is_nan(value) for value in plddt) or any(_is_nan(value) for row in pae for value in row)
        mean_plddt = sum(plddt) / len(plddt)
        min_plddt = min(plddt)
        max_plddt = max(plddt)
        plddt_above_70 = sum(1 for value in plddt if value >= 70.0) / len(plddt)

        self._output_has_nan = output_has_nan
        self._mean_plddt = mean_plddt
        self._min_plddt = min_plddt
        self._max_plddt = max_plddt
        self._plddt_above_70 = plddt_above_70
        self._max_pae = max_pae

        output_finite = all(
            _is_finite(metric) for metric in (mean_plddt, min_plddt, max_plddt, plddt_above_70, max_pae)
        )
        return output_finite and not (self.case.require_no_nan and output_has_nan)


__all__ = ["ca_metrics", "parse_pdb_ca", "validate_run"]
