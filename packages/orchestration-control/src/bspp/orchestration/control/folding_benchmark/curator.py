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

"""Reproducibly curate a reference-backed PDB benchmark without running inference.

Pure-stdlib port of ``prepare_benchmark_dataset`` from the frozen reference pipeline
harvest source (``src/afdb_pipeline/benchmark_dataset.py``), adapted to the
parsed ``BenchmarkSpec``/``BenchmarkStratum`` model and the mapping-based
``ParsedAssembly`` from this package (e05s05). The only external boundary is
:func:`_request`; PyArrow is imported lazily through ``importlib`` so the control
plane keeps no heavy dependency.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from bspp.orchestration.control.folding_benchmark.mmcif_assembly import MmcifCaData, parse_mmcif_assembly
from bspp.orchestration.control.folding_benchmark.pdb_assembly import (
    PDB_RESIDUES,
    CaResidue,
    ParsedAssembly,
    normalize_ca_residue_numbers,
    parse_pdb_assembly,
)
from bspp.orchestration.control.folding_benchmark.spec import BenchmarkSpec, BenchmarkStratum

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_API = "https://data.rcsb.org/rest/v1/core"
FILES_URL = "https://files.rcsb.org/download"
USER_AGENT = "afdb-folding-pipeline-benchmark-curator/0.1"
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 100 * 1024 * 1024
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
PREDICTED_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PYARROW_SETUP_COMMAND = "uv run --isolated --with pyarrow bsppctl prepare-benchmark"

ONE_TO_THREE = {one_letter: three_letter for three_letter, one_letter in PDB_RESIDUES.items() if three_letter != "MSE"}


class CandidateRejected(ValueError):  # noqa: N818 - accepted domain event terminology
    """A normal scientific-filter rejection, retained only as a count."""


class BenchmarkCurationFailed(ValueError):  # noqa: N818 - accepted domain event terminology
    """Expected curation exhaustion/selection failure, not a programming defect."""


class PyArrowUnavailable(RuntimeError):  # noqa: N818 - accepted domain event terminology
    """PyArrow could not be imported for the optional parquet writer."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _source_methods(spec: BenchmarkSpec) -> tuple[str, ...]:
    value = spec.source.get("required_any_experimental_methods", ())
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str) and item)
    return ()


def _source_release_min(spec: BenchmarkSpec) -> str:
    value = spec.source.get("release_date_min", "")
    return value if isinstance(value, str) else ""


def _source_release_max(spec: BenchmarkSpec) -> str:
    value = spec.source.get("release_date_max", "")
    return value if isinstance(value, str) else ""


def _source_max_resolution(spec: BenchmarkSpec) -> float:
    value = spec.source.get("max_resolution_angstrom", 3.0)
    if isinstance(value, bool):
        return 3.0
    if isinstance(value, (int, float)):
        return float(value)
    return 3.0


def _source_assembly_id(spec: BenchmarkSpec) -> str:
    value = spec.source.get("assembly_id", "1")
    return value if isinstance(value, str) else "1"


def _filter_bool(spec: BenchmarkSpec, key: str, default: bool) -> bool:
    value = spec.filters.get(key, default)
    return value if isinstance(value, bool) else default


def _filter_int(spec: BenchmarkSpec, key: str, default: int) -> int:
    value = spec.filters.get(key, default)
    if isinstance(value, bool):
        return default
    return value if isinstance(value, int) else default


def _filter_float(spec: BenchmarkSpec, key: str, default: float) -> float:
    value = spec.filters.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _spec_description(spec: BenchmarkSpec) -> str:
    return spec.description


def _spec_mapping(spec: BenchmarkSpec) -> dict[str, object]:
    """Return the verbatim parsed specification mapping used for fingerprinting."""
    return dict(spec.raw_specification)


def _request(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: int = 60,
    maximum_bytes: int | None = None,
) -> bytes:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": USER_AGENT}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if maximum_bytes is not None:
                    declared = response.headers.get("Content-Length")
                    if declared is not None and int(declared) > maximum_bytes:
                        raise CandidateRejected("source coordinate file exceeds size limit")
                    value = cast(bytes, response.read(maximum_bytes + 1))
                    if len(value) > maximum_bytes:
                        raise CandidateRejected("source coordinate file exceeds size limit")
                    return value
                return cast(bytes, response.read())
        except CandidateRejected:
            raise
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            status = getattr(exc, "code", None)
            if status not in {429, 500, 502, 503, 504} and attempt == 0:
                break
            if attempt < 4:
                time.sleep(min(2**attempt, 8))
    raise CandidateRejected(f"source request failed: {last_error}")


def _request_json(url: str, *, payload: Mapping[str, Any] | None = None) -> Any:
    try:
        return json.loads(_request(url, payload=payload).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateRejected(f"source returned invalid JSON: {exc}") from exc


def _terminal(attribute: str, operator: str, value: Any) -> dict[str, Any]:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def _search_candidates(spec: BenchmarkSpec, stratum: BenchmarkStratum) -> list[str]:
    methods = [_terminal("exptl.method", "exact_match", method) for method in _source_methods(spec)]
    nodes: list[dict[str, Any]] = [
        _terminal(
            "rcsb_accession_info.initial_release_date",
            "greater_or_equal",
            _source_release_min(spec),
        ),
        _terminal(
            "rcsb_accession_info.initial_release_date",
            "less_or_equal",
            _source_release_max(spec),
        ),
        _terminal(
            "rcsb_entry_info.resolution_combined",
            "less_or_equal",
            _source_max_resolution(spec),
        ),
        _terminal(
            "rcsb_assembly_info.selected_polymer_entity_types",
            "exact_match",
            "Protein (only)",
        ),
        _terminal(
            "rcsb_assembly_info.polymer_entity_instance_count",
            "equals",
            stratum.chain_count,
        ),
        _terminal(
            "rcsb_assembly_info.polymer_monomer_count",
            "greater_or_equal",
            stratum.minimum_total_residues,
        ),
        _terminal(
            "rcsb_assembly_info.polymer_monomer_count",
            "less_or_equal",
            stratum.maximum_total_residues,
        ),
        {"type": "group", "logical_operator": "or", "nodes": methods},
    ]
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "request_options": {"return_all_hits": True, "results_verbosity": "compact"},
        "return_type": "assembly",
    }
    response = _request_json(SEARCH_URL, payload=query)
    if not isinstance(response, dict):
        raise BenchmarkCurationFailed(f"RCSB search returned no candidates for {stratum.name}")
    result_set = response.get("result_set")
    if not isinstance(result_set, list):
        raise BenchmarkCurationFailed(f"RCSB search returned no candidates for {stratum.name}")
    candidates: list[str] = []
    for value in result_set:
        if not isinstance(value, str) or "-" not in value:
            continue
        pdb_id, assembly_id = value.rsplit("-", 1)
        if len(pdb_id) == 4 and assembly_id == _source_assembly_id(spec):
            candidates.append(f"{pdb_id.upper()}-{assembly_id}")
    candidates.sort(
        key=lambda item: hashlib.sha256(f"{spec.selection_seed}\0{stratum.name}\0{item}".encode()).hexdigest()
    )
    return candidates


def _canonical_entity_sequence(entity: Mapping[str, Any]) -> str | None:
    entity_poly = entity.get("entity_poly")
    if not isinstance(entity_poly, dict):
        return None
    value = entity_poly.get("pdbx_seq_one_letter_code_can")
    if not isinstance(value, str):
        return None
    return "".join(value.split()).upper()


def _entity_cluster(entity: Mapping[str, Any], identity: int) -> str | None:
    memberships = entity.get("rcsb_cluster_membership")
    if not isinstance(memberships, list):
        return None
    for membership in memberships:
        if (
            isinstance(membership, dict)
            and membership.get("identity") == identity
            and membership.get("cluster_id") is not None
        ):
            return str(membership["cluster_id"])
    return None


def _resolution(entry: Mapping[str, Any]) -> float:
    info = entry.get("rcsb_entry_info")
    values = info.get("resolution_combined") if isinstance(info, dict) else None
    if not isinstance(values, list) or not values:
        raise CandidateRejected("entry has no experimental resolution")
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        raise CandidateRejected("entry has no numeric experimental resolution")
    return min(numeric)


def _seqres_lines(chain_ids: Iterable[str], sequences: Iterable[str]) -> list[str]:
    lines: list[str] = []
    for chain_id, sequence in zip(chain_ids, sequences, strict=True):
        residues = [ONE_TO_THREE[residue] for residue in sequence]
        for serial, offset in enumerate(range(0, len(residues), 13), start=1):
            values = " ".join(residues[offset : offset + 13])
            lines.append(f"SEQRES {serial:3d} {chain_id} {len(residues):4d}  {values}")
    return lines


def _gunzip_bounded(compressed: bytes, maximum_bytes: int = MAX_DECOMPRESSED_BYTES) -> bytes:
    """Stream-decompress a gzip payload with an explicit expanded-byte cap.

    The compressed-byte download cap alone cannot bound curator memory: a
    highly compressible response expands arbitrarily. Reads stop at
    ``maximum_bytes + 1`` and anything beyond the cap fails with
    :class:`CandidateRejected`.
    """
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        value = stream.read(maximum_bytes + 1)
    if len(value) > maximum_bytes:
        raise CandidateRejected("decompressed coordinate file exceeds size limit")
    return value


def _materialize_candidate(
    identifier: str,
    spec: BenchmarkSpec,
    stratum: BenchmarkStratum,
) -> dict[str, Any]:
    pdb_id, assembly_id = identifier.split("-", 1)
    assembly = _request_json(f"{DATA_API}/assembly/{pdb_id}/{assembly_id}")
    entry = _request_json(f"{DATA_API}/entry/{pdb_id}")
    if not isinstance(assembly, dict) or not isinstance(entry, dict):
        raise CandidateRejected("entry or assembly metadata is not an object")
    assembly_details = assembly.get("pdbx_struct_assembly")
    if not isinstance(assembly_details, dict):
        raise CandidateRejected("assembly metadata is missing")
    if (
        _filter_bool(spec, "require_candidate_biological_assembly", True)
        and assembly_details.get("rcsb_candidate_assembly") != "Y"
    ):
        raise CandidateRejected("assembly is not marked as an RCSB candidate assembly")
    accession = entry.get("rcsb_accession_info")
    release_value = accession.get("initial_release_date") if isinstance(accession, dict) else None
    if not isinstance(release_value, str):
        raise CandidateRejected("entry release date is missing")
    release_date = release_value[:10]
    if not _source_release_min(spec) <= release_date <= _source_release_max(spec):
        raise CandidateRejected("entry release date is outside the frozen window")
    methods_value = entry.get("exptl")
    methods = (
        tuple(
            item["method"] for item in methods_value if isinstance(item, dict) and isinstance(item.get("method"), str)
        )
        if isinstance(methods_value, list)
        else ()
    )
    if not methods or not set(methods) & set(_source_methods(spec)):
        raise CandidateRejected("entry experimental method is outside the specification")
    resolution = _resolution(entry)
    if resolution > _source_max_resolution(spec):
        raise CandidateRejected("entry resolution is outside the specification")

    source_url = f"{FILES_URL}/{pdb_id}.pdb{assembly_id}.gz"
    compressed = _request(source_url, maximum_bytes=MAX_DOWNLOAD_BYTES)
    try:
        coordinate_bytes = _gunzip_bounded(compressed)
        coordinate_text = coordinate_bytes.decode("utf-8", errors="replace")
    except (OSError, EOFError) as exc:
        raise CandidateRejected(f"invalid compressed assembly file: {exc}") from exc
    try:
        parsed = parse_pdb_assembly(coordinate_text)
    except ValueError as exc:
        raise CandidateRejected(str(exc)) from exc

    chain_ids = tuple(parsed.chains.keys())
    sequences = tuple("".join(PDB_RESIDUES[item] for item in parsed.chains[chain_id]) for chain_id in chain_ids)
    if len(sequences) != stratum.chain_count:
        raise CandidateRejected("reference chain count does not match the stratum")
    lengths = tuple(map(len, sequences))
    total_length = sum(lengths)
    if not stratum.minimum_total_residues <= total_length <= stratum.maximum_total_residues:
        raise CandidateRejected("reference length does not match the stratum")
    if min(lengths) < _filter_int(spec, "minimum_chain_length", 40):
        raise CandidateRejected("reference contains a chain below minimum length")
    if _filter_bool(spec, "canonical_amino_acids_only", True) and any(
        set(sequence) - STANDARD_AMINO_ACIDS for sequence in sequences
    ):
        raise CandidateRejected("reference contains non-canonical sequence characters")
    ca_counts = tuple(len(parsed.ca_residues[chain_id]) for chain_id in chain_ids)
    coverages = tuple(count / length for count, length in zip(ca_counts, lengths, strict=True))
    if min(coverages) < _filter_float(spec, "minimum_coordinate_coverage", 0.7):
        raise CandidateRejected("reference coordinate coverage is below minimum")

    identifiers = entry.get("rcsb_entry_container_identifiers")
    entity_ids = identifiers.get("polymer_entity_ids") if isinstance(identifiers, dict) else None
    if not isinstance(entity_ids, list) or not entity_ids:
        raise CandidateRejected("entry polymer entity identifiers are missing")
    entities = [_request_json(f"{DATA_API}/polymer_entity/{pdb_id}/{entity_id}") for entity_id in entity_ids]
    cluster_identity = _filter_int(spec, "monomer_sequence_cluster_identity", 30)
    clusters_by_sequence: dict[str, str] = {}
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        sequence = _canonical_entity_sequence(entity)
        cluster = _entity_cluster(entity, cluster_identity)
        if sequence is not None and cluster is not None:
            clusters_by_sequence[sequence] = cluster
    try:
        cluster_ids = tuple(clusters_by_sequence[sequence] for sequence in sequences)
    except KeyError as exc:
        raise CandidateRejected("cannot map an assembly chain to its RCSB sequence cluster") from exc

    target_id = f"pdb_{pdb_id.lower()}_assembly_{assembly_id}"
    sequence_value = ":".join(sequences)
    sequence_sha256 = hashlib.sha256(sequence_value.encode()).hexdigest()
    try:
        normalized_ca_lines = normalize_ca_residue_numbers(parsed)
    except ValueError as exc:
        raise CandidateRejected(str(exc)) from exc
    reference_text = "\n".join([*_seqres_lines(chain_ids, sequences), *normalized_ca_lines, "END"]) + "\n"
    reference_bytes = reference_text.encode("utf-8")
    info = assembly.get("rcsb_assembly_info")
    composition = info.get("polymer_composition") if isinstance(info, dict) else None
    return {
        "target_id": target_id,
        "description": (
            f"RCSB PDB {pdb_id} biological assembly {assembly_id}; stratum={stratum.name}; release={release_date}"
        ),
        "pdb_id": pdb_id,
        "assembly_id": assembly_id,
        "stratum": stratum.name,
        "chain_count": len(sequences),
        "chain_ids": list(chain_ids),
        "chain_lengths": list(lengths),
        "chains": list(sequences),
        "sequence": sequence_value,
        "sequence_sha256": sequence_sha256,
        "total_residues": total_length,
        "ca_counts": list(ca_counts),
        "coordinate_coverage": list(coverages),
        "minimum_coordinate_coverage": min(coverages),
        "cluster_identity": cluster_identity,
        "cluster_ids": list(cluster_ids),
        "cluster_signature": ":".join(sorted(cluster_ids)),
        "release_date": release_date,
        "experimental_methods": list(methods),
        "resolution_angstrom": resolution,
        "assembly_composition": composition,
        "assembly_details": assembly_details.get("details"),
        "assembly_oligomeric_details": assembly_details.get("oligomeric_details"),
        "source_url": source_url,
        "source_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
        "source_pdb_sha256": hashlib.sha256(coordinate_bytes).hexdigest(),
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "reference_bytes": reference_bytes,
    }


def _candidate_outcome(
    identifier: str,
    spec: BenchmarkSpec,
    stratum: BenchmarkStratum,
) -> tuple[str, dict[str, Any] | None, str | None]:
    try:
        return identifier, _materialize_candidate(identifier, spec, stratum), None
    except CandidateRejected as exc:
        message = str(exc)
        categories = {
            "reference contains unsupported residue": "unsupported_reference_residue",
            "source request failed": "source_request_failed",
            "invalid compressed assembly file": "invalid_compressed_assembly",
        }
        reason = next(
            (category for prefix, category in categories.items() if message.startswith(prefix)),
            message.split(":", 1)[0].replace(" ", "_").lower(),
        )
        return identifier, None, reason


def _ranked_throughput_order(
    records: Iterable[dict[str, Any]],
    strata: Iterable[BenchmarkStratum],
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record["stratum"]].append(record)
    definitions = {item.name: item for item in strata}
    for name, values in buckets.items():
        values.sort(key=lambda item: item["selection_rank"])
        if len(values) != definitions[name].count:
            raise BenchmarkCurationFailed(f"selected benchmark stratum {name} has the wrong size")
    consumed: Counter[str] = Counter()
    ordered: list[dict[str, Any]] = []
    total = sum(item.count for item in definitions.values())
    for _ in range(total):
        available = [item for item in definitions.values() if consumed[item.name] < item.count]
        selected = min(available, key=lambda item: (consumed[item.name] / item.count, item.name))
        ordered.append(buckets[selected.name][consumed[selected.name]])
        consumed[selected.name] += 1
    return ordered


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "reference_bytes"}


def _write_fasta(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(f">{record['target_id']} {record['description']}\n")
                handle.write(f"{record['sequence']}\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pyarrow_modules() -> tuple[Any, Any]:
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise PyArrowUnavailable(
            f"pyarrow is required to write the benchmark parquet output; install it with: {PYARROW_SETUP_COMMAND}"
        ) from exc
    return pa, pq


def _write_parquet(
    path: Path,
    records: list[dict[str, Any]],
    *,
    source_key: str = "source_gzip_sha256",
) -> None:
    pa, pq = _pyarrow_modules()
    rows = []
    for record in records:
        rows.append(
            {
                "target_id": record["target_id"],
                "pdb_id": record["pdb_id"],
                "assembly_id": record["assembly_id"],
                "stratum": record["stratum"],
                "chain_count": record["chain_count"],
                "chain_lengths_json": json.dumps(record["chain_lengths"]),
                "total_residues": record["total_residues"],
                "sequence_sha256": record["sequence_sha256"],
                "release_date": record["release_date"],
                "experimental_methods_json": json.dumps(record["experimental_methods"]),
                "resolution_angstrom": record["resolution_angstrom"],
                "minimum_coordinate_coverage": record["minimum_coordinate_coverage"],
                "cluster_identity": record["cluster_identity"],
                "cluster_ids_json": json.dumps(record["cluster_ids"]),
                "reference_path": record["reference_path"],
                "reference_sha256": record["reference_sha256"],
                "source_url": record["source_url"],
                source_key: record[source_key],
            }
        )
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd", version="2.6")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS" and ".tmp" not in path.name
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_fingerprint(
    spec: Mapping[str, object],
    records: Iterable[tuple[str, str, str, str, str]],
    *,
    source_key: str = "source_gzip_sha256",
    fingerprint_prefix: bytes = b"afdb-pdb-benchmark-v1\0",
) -> str:
    digest = hashlib.sha256()
    digest.update(fingerprint_prefix)
    digest.update(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode())
    for target_id, sequence_sha256, stratum, reference_sha256, source_hash in records:
        value = {
            "target_id": target_id,
            "sequence_sha256": sequence_sha256,
            "stratum": stratum,
            "reference_sha256": reference_sha256,
            source_key: source_hash,
        }
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _materialize_dataset(
    staging: Path,
    spec: BenchmarkSpec,
    selected: list[dict[str, Any]],
    rejection_counts: Mapping[str, int],
    *,
    source_key: str = "source_gzip_sha256",
    fingerprint_prefix: bytes = b"afdb-pdb-benchmark-v1\0",
    selection_description: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    references = staging / "references"
    references.mkdir(parents=True)
    for record in selected:
        relative = Path("references") / f"{record['target_id']}.pdb"
        (staging / relative).write_bytes(record["reference_bytes"])
        record["reference_path"] = str(relative)
    selected.sort(key=lambda item: item["target_id"])
    public_records = [_public_record(record) for record in selected]
    _write_fasta(staging / "targets.fasta", public_records)
    _write_jsonl(staging / "targets.jsonl", public_records)
    _write_parquet(staging / "targets.parquet", public_records, source_key=source_key)

    fingerprint = _dataset_fingerprint(
        _spec_mapping(spec),
        (
            (
                record["target_id"],
                record["sequence_sha256"],
                record["stratum"],
                record["reference_sha256"],
                record[source_key],
            )
            for record in public_records
        ),
        source_key=source_key,
        fingerprint_prefix=fingerprint_prefix,
    )
    minimum_coordinate_coverage = _filter_float(spec, "minimum_coordinate_coverage", 0.7)
    validation_cases = []
    for record in public_records:
        validation_cases.append(
            {
                "target_id": record["target_id"],
                "sequence_sha256": record["sequence_sha256"],
                "thresholds": {"ca_coverage": minimum_coordinate_coverage},
                "require_no_nan": True,
                "expected_pair_mode": "unpaired_paired",
                "reference_structure": record["reference_path"],
                "reference_sha256": record["reference_sha256"],
                "chain_map": {
                    PREDICTED_CHAIN_IDS[index]: chain_id for index, chain_id in enumerate(record["chain_ids"])
                },
                "metadata": {
                    "pdb_id": record["pdb_id"],
                    "assembly_id": record["assembly_id"],
                    "stratum": record["stratum"],
                    "release_date": record["release_date"],
                    "resolution_angstrom": record["resolution_angstrom"],
                },
            }
        )
    _write_json(
        staging / "validation-suite.json",
        {
            "schema_version": 1,
            "dataset_id": spec.dataset_id,
            "fingerprint": fingerprint,
            "cases": validation_cases,
        },
    )

    throughput_order = _ranked_throughput_order(public_records, spec.strata)
    validation_by_target = {item["target_id"]: item for item in validation_cases}
    subset_summaries = []
    for size in spec.throughput_subset_sizes:
        subset = throughput_order[:size]
        subset_dir = staging / "subsets" / f"n{size:04d}"
        _write_fasta(subset_dir / "targets.fasta", subset)
        _write_jsonl(
            subset_dir / "target-ids.jsonl",
            ({"target_id": record["target_id"]} for record in subset),
        )
        validation_name = f"validation-suite-n{size:04d}.json"
        _write_json(
            staging / validation_name,
            {
                "schema_version": 1,
                "dataset_id": f"{spec.dataset_id}-n{size:04d}",
                "fingerprint": fingerprint,
                "cases": [validation_by_target[record["target_id"]] for record in subset],
            },
        )
        counts = Counter(record["stratum"] for record in subset)
        subset_summaries.append(
            {
                "size": size,
                "path": str(Path("subsets") / f"n{size:04d}" / "targets.fasta"),
                "validation_suite": validation_name,
                "total_residues": sum(record["total_residues"] for record in subset),
                "strata": dict(sorted(counts.items())),
            }
        )
    _write_json(
        staging / "throughput-subsets.json",
        {"schema_version": 1, "nested": True, "subsets": subset_summaries},
    )

    strata_counts = Counter(record["stratum"] for record in public_records)
    method_counts = Counter(method for record in public_records for method in record["experimental_methods"])
    dataset = {
        "schema_version": 1,
        "dataset_id": spec.dataset_id,
        "dataset_fingerprint": fingerprint,
        "description": _spec_description(spec),
        "generated_at": _utc_now(),
        "specification": _spec_mapping(spec),
        "targets": len(public_records),
        "chains": sum(record["chain_count"] for record in public_records),
        "total_residues": sum(record["total_residues"] for record in public_records),
        "strata": dict(sorted(strata_counts.items())),
        "experimental_methods": dict(sorted(method_counts.items())),
        "release_date_range": [
            min(record["release_date"] for record in public_records),
            max(record["release_date"] for record in public_records),
        ],
        "resolution_angstrom": {
            "minimum": min(record["resolution_angstrom"] for record in public_records),
            "maximum": max(record["resolution_angstrom"] for record in public_records),
        },
        "selection": (
            dict(selection_description)
            if selection_description is not None
            else {
                "monomers": "unique RCSB sequence cluster at configured identity",
                "complexes": "unique sorted multiset of RCSB chain-cluster IDs",
                "candidate_order": "SHA-256(seed, stratum, assembly identifier)",
                "rejections": dict(sorted(rejection_counts.items())),
            }
        ),
        "files": {
            "fasta": "targets.fasta",
            "targets_jsonl": "targets.jsonl",
            "targets_parquet": "targets.parquet",
            "validation_suite": "validation-suite.json",
            "throughput_subsets": "throughput-subsets.json",
            "checksums": "SHA256SUMS",
        },
        "data_license": "CC0 1.0 Universal; see DATASET-LICENSE.md",
    }
    _write_json(staging / "dataset.json", dataset)
    (staging / "DATASET-LICENSE.md").write_text(
        "# Dataset provenance and terms\n\n"
        "Coordinate and metadata records are derived from the RCSB Protein Data Bank. "
        "The wwPDB archive and RCSB programmatic API data are provided under the "
        "CC0 1.0 Universal Public Domain Dedication. RCSB encourages attribution "
        "of the original structure authors where possible.\n\n"
        "- RCSB usage policy: https://www.rcsb.org/pages/usage-policy\n"
        "- Search API: https://search.rcsb.org/\n"
        "- Data API: https://data.rcsb.org/\n\n"
        "This generated benchmark contains normalized sequences, C-alpha-only "
        "reference PDB files, source identifiers, release dates, experimental "
        "methods, resolutions, and checksums. It does not redistribute MSA databases "
        "or model parameters.\n",
        encoding="utf-8",
    )
    _write_checksums(staging)
    return dataset


def prepare_benchmark_dataset(
    output: Path,
    specification: BenchmarkSpec,
    *,
    workers: int = 12,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Discover, filter, pin, and materialize a reference-backed benchmark."""
    spec = specification
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be an integer >= 1")
    output_path = output.resolve()
    if output_path.exists():
        dataset_path = output_path / "dataset.json"
        if dataset_path.is_file():
            existing = json.loads(dataset_path.read_text(encoding="utf-8"))
            if (
                isinstance(existing, dict)
                and existing.get("dataset_id") == spec.dataset_id
                and existing.get("specification") == _spec_mapping(spec)
            ):
                return existing
        raise ValueError(f"benchmark output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", suffix=".staging", dir=output_path.parent))
    selected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    used_entries: set[str] = set()
    used_sequences: set[str] = set()
    used_monomer_clusters: set[str] = set()
    used_complex_signatures: set[str] = set()
    try:
        for stratum in spec.strata:
            progress(f"Discovering {stratum.name} candidates...")
            candidates = _search_candidates(spec, stratum)
            progress(f"{stratum.name}: {len(candidates)} candidate assembly records; selecting {stratum.count}")
            accepted = 0
            examined = 0
            batch_size = max(workers * 2, 16)
            for offset in range(0, len(candidates), batch_size):
                batch = [
                    item
                    for item in candidates[offset : offset + batch_size]
                    if item.split("-", 1)[0] not in used_entries
                ]
                if not batch:
                    continue
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    outcomes = list(
                        executor.map(
                            lambda identifier, stratum=stratum: _candidate_outcome(identifier, spec, stratum),
                            batch,
                        )
                    )
                for identifier, record, rejection in outcomes:
                    examined += 1
                    if rejection is not None:
                        rejection_counts[rejection] += 1
                        continue
                    assert record is not None
                    entry = record["pdb_id"]
                    if entry in used_entries:
                        rejection_counts["entry_already_selected"] += 1
                        continue
                    if record["sequence_sha256"] in used_sequences:
                        rejection_counts["duplicate_target_sequence"] += 1
                        continue
                    if record["chain_count"] == 1:
                        cluster = record["cluster_ids"][0]
                        if cluster in used_monomer_clusters:
                            rejection_counts["duplicate_monomer_cluster"] += 1
                            continue
                    else:
                        signature = record["cluster_signature"]
                        if signature in used_complex_signatures:
                            rejection_counts["duplicate_complex_cluster_signature"] += 1
                            continue
                    record["selection_rank"] = offset + batch.index(identifier)
                    selected.append(record)
                    used_entries.add(entry)
                    used_sequences.add(record["sequence_sha256"])
                    if record["chain_count"] == 1:
                        used_monomer_clusters.add(record["cluster_ids"][0])
                    else:
                        used_complex_signatures.add(record["cluster_signature"])
                    accepted += 1
                    if accepted % 25 == 0 or accepted == stratum.count:
                        progress(
                            f"{stratum.name}: accepted {accepted}/{stratum.count} "
                            f"after {examined} materialized candidates"
                        )
                    if accepted == stratum.count:
                        break
                if accepted == stratum.count:
                    break
            if accepted != stratum.count:
                raise BenchmarkCurationFailed(
                    f"could not fill {stratum.name}: selected {accepted}/{stratum.count} "
                    f"from {examined} materialized candidates; staging={staging}"
                )
        dataset = _materialize_dataset(
            staging,
            spec,
            selected,
            rejection_counts,
        )
        os.replace(staging, output_path)
        progress(f"Prepared {dataset['targets']} targets and {dataset['total_residues']} residues at {output_path}")
        dataset["output"] = str(output_path)
        return dataset
    except Exception:
        progress(f"Benchmark preparation did not complete; staging retained at {staging}")
        raise


def _is_subsequence(candidate: tuple[str, ...], full: tuple[str, ...]) -> bool:
    """Return True if *candidate* is a subsequence of *full* (both 3-letter residue tuples)."""
    offset = 0
    for residue in candidate:
        try:
            offset = full.index(residue, offset) + 1
        except ValueError:
            return False
    return True


def _match_label_asym_ids_to_pinned_chains(
    mmcif_ca: MmcifCaData,
    pinned_chains: list[str],
) -> list[str]:
    """Find a bijection between label_asym_ids and pinned chain positions.

    Uses Kuhn's augmenting-path maximum bipartite matching where an edge exists
    iff the observed CA subsequence of the label_asym_id is a subsequence of the
    pinned chain. Returns label_asym_id values in pinned-chain order.
    Raises CandidateRejected if no perfect matching exists.

    Deterministic: pinned-index outer loop (0, 1, 2, ...), sorted label_asym_id
    inner loop. The exact homomer tie-break is an implementation detail of
    Kuhn's algorithm but is deterministic, so the fingerprint is stable.
    """
    asym_ids = sorted(mmcif_ca.ca_residues.keys())
    n = len(pinned_chains)
    if len(asym_ids) != n:
        raise CandidateRejected("reconstruction chain count does not match the number of mmCIF chains")
    pinned_3letter: list[tuple[str, ...]] = [
        tuple(ONE_TO_THREE[residue] for residue in chain) for chain in pinned_chains
    ]
    adj: list[list[str]] = []
    for pinned_index in range(n):
        candidates = []
        for asym_id in asym_ids:
            ca_residues = mmcif_ca.ca_residues[asym_id]
            ca_3letter = tuple(residue[1] for residue in ca_residues)
            if _is_subsequence(ca_3letter, pinned_3letter[pinned_index]):
                candidates.append(asym_id)
        adj.append(candidates)

    match_asym: dict[str, int | None] = {asym_id: None for asym_id in asym_ids}

    def try_augment(pinned_index: int, visited: set[str]) -> bool:
        for asym_id in adj[pinned_index]:
            if asym_id in visited:
                continue
            visited.add(asym_id)
            matched = match_asym[asym_id]
            if matched is None or try_augment(matched, visited):
                match_asym[asym_id] = pinned_index
                return True
        return False

    for pinned_index in range(n):
        if not try_augment(pinned_index, set()):
            raise CandidateRejected("no perfect bipartite matching between mmCIF chains and pinned chains")

    result: list[str] = [""] * n
    for asym_id, matched_index in match_asym.items():
        if matched_index is not None:
            result[matched_index] = asym_id
    if any(not slot for slot in result):
        raise CandidateRejected("no perfect bipartite matching between mmCIF chains and pinned chains")
    return result


_TARGET_ID_RE = re.compile(r"^pdb_([a-z0-9]+)_assembly_([0-9]+)$")


def _parse_target_record(record: Mapping[str, object]) -> dict[str, Any]:
    """Parse a pinned JSONL target record into a reconstruction working dict."""
    target_id = record.get("target_id")
    if not isinstance(target_id, str):
        raise ValueError("pinned target record is missing target_id")
    match = _TARGET_ID_RE.match(target_id)
    if match is None:
        raise ValueError(f"pinned target_id does not match expected pattern: {target_id}")
    pdb_id = match.group(1).upper()
    assembly_id = match.group(2)
    description = record.get("description")
    if not isinstance(description, str):
        raise ValueError(f"pinned target {target_id} is missing description")
    # Strip "<target_id> " prefix (Decision E).
    if description.startswith(f"{target_id} "):
        description = description[len(target_id) + 1 :]
    # Parse stratum and release_date from description.
    # Description format: "RCSB PDB <pdb_id> biological assembly <n>; stratum=<name>; release=<date>"
    stratum: str | None = None
    release_date: str | None = None
    for part in description.split(";"):
        part = part.strip()
        if part.startswith("stratum="):
            stratum = part.split("=", 1)[1].strip()
        elif part.startswith("release="):
            release_date = part.split("=", 1)[1].strip()
    if stratum is None:
        raise ValueError(f"pinned target {target_id} description has no stratum")
    if release_date is None:
        raise ValueError(f"pinned target {target_id} description has no release_date")
    chains = record.get("chains")
    if not isinstance(chains, list) or not all(isinstance(c, str) for c in chains):
        raise ValueError(f"pinned target {target_id} has invalid chains")
    chain_lengths = record.get("chain_lengths")
    if not isinstance(chain_lengths, list) or not all(isinstance(length, int) for length in chain_lengths):
        raise ValueError(f"pinned target {target_id} has invalid chain_lengths")
    total_length = record.get("total_length")
    if not isinstance(total_length, int):
        raise ValueError(f"pinned target {target_id} has invalid total_length")
    sequence_sha256 = record.get("sequence_sha256")
    if not isinstance(sequence_sha256, str):
        raise ValueError(f"pinned target {target_id} has invalid sequence_sha256")
    return {
        "target_id": target_id,
        "description": description,
        "pdb_id": pdb_id,
        "assembly_id": assembly_id,
        "stratum": stratum,
        "release_date": release_date,
        "chains": chains,
        "chain_lengths": chain_lengths,
        "total_length": total_length,
        "sequence_sha256": sequence_sha256,
    }


def _materialize_reconstruction_target(
    record: Mapping[str, object],
    spec: BenchmarkSpec,
    stratum: BenchmarkStratum,
) -> dict[str, Any]:
    """Reconstruct one target from pinned data + mmCIF + Data API."""
    parsed = _parse_target_record(record)
    pdb_id = parsed["pdb_id"]
    assembly_id = parsed["assembly_id"]
    pinned_chains = parsed["chains"]
    # N8: chain_count cross-check.
    if len(pinned_chains) != stratum.chain_count:
        raise CandidateRejected("reconstruction chain count does not match the stratum")
    # Verify chain_lengths and total_length.
    if list(parsed["chain_lengths"]) != [len(chain) for chain in pinned_chains]:
        raise CandidateRejected("reconstruction chain_lengths do not match pinned chains")
    if sum(len(chain) for chain in pinned_chains) != parsed["total_length"]:
        raise CandidateRejected("reconstruction total_length does not match pinned chains")
    # Step 5a: Download mmCIF assembly (N5: bounded by MAX_DOWNLOAD_BYTES).
    mmcif_url = f"{FILES_URL}/{pdb_id.lower()}-assembly{assembly_id}.cif.gz"
    compressed = _request(mmcif_url, maximum_bytes=MAX_DOWNLOAD_BYTES)
    decompressed = _gunzip_bounded(compressed)
    # N10: source_mmcif_sha256 = SHA-256 of raw decompressed bytes.
    source_mmcif_sha256 = hashlib.sha256(decompressed).hexdigest()
    mmcif_text = decompressed.decode("utf-8", errors="replace")
    # Step 5c: Parse mmCIF.
    mmcif_ca = parse_mmcif_assembly(mmcif_text)
    # Step 5e: Match label_asym_ids to pinned chains (N9 deterministic).
    matched_asym_ids = _match_label_asym_ids_to_pinned_chains(mmcif_ca, pinned_chains)
    # Step 5f: Map to synthetic single-character IDs (B1).
    synthetic_chain_ids = [PREDICTED_CHAIN_IDS[i] for i in range(len(matched_asym_ids))]
    # Build ParsedAssembly with synthetic IDs.
    chains_3letter: dict[str, tuple[str, ...]] = {}
    ca_residues: dict[str, tuple[CaResidue, ...]] = {}
    for i, (syn_id, asym_id) in enumerate(zip(synthetic_chain_ids, matched_asym_ids, strict=True)):
        chains_3letter[syn_id] = tuple(ONE_TO_THREE[residue] for residue in pinned_chains[i])
        ca_residues[syn_id] = mmcif_ca.ca_residues[asym_id]
    parsed_assembly = ParsedAssembly(chains=chains_3letter, ca_residues=ca_residues)
    # Step 5h: Verify sequence_sha256.
    sequence_value = ":".join(pinned_chains)
    recomputed_sha256 = hashlib.sha256(sequence_value.encode()).hexdigest()
    if recomputed_sha256 != parsed["sequence_sha256"]:
        raise CandidateRejected("reconstruction sequence_sha256 does not match pinned value")
    # Step 5i: Compute ca_counts and coordinate_coverage.
    chain_ids_tuple = tuple(synthetic_chain_ids)
    lengths = tuple(len(chain) for chain in pinned_chains)
    ca_counts = tuple(len(ca_residues[syn_id]) for syn_id in chain_ids_tuple)
    coverages = tuple(count / length for count, length in zip(ca_counts, lengths, strict=True))
    minimum_coverage = _filter_float(spec, "minimum_coordinate_coverage", 0.7)
    if min(coverages) < minimum_coverage:
        raise CandidateRejected("reconstruction coordinate coverage is below minimum")
    # Step 5j: Normalize CA residue numbers.
    normalized_ca_lines = normalize_ca_residue_numbers(parsed_assembly)
    # Step 5k: Build reference PDB text.
    reference_text = "\n".join([*_seqres_lines(chain_ids_tuple, pinned_chains), *normalized_ca_lines, "END"]) + "\n"
    reference_bytes = reference_text.encode("utf-8")
    reference_sha256 = hashlib.sha256(reference_bytes).hexdigest()
    # Step 5m: Fetch experimental_methods and resolution from Data API (Decision H: uppercase).
    entry = _request_json(f"{DATA_API}/entry/{pdb_id.upper()}")
    if not isinstance(entry, dict):
        raise CandidateRejected("entry metadata is not an object")
    methods_value = entry.get("exptl")
    methods = (
        tuple(
            item["method"] for item in methods_value if isinstance(item, dict) and isinstance(item.get("method"), str)
        )
        if isinstance(methods_value, list)
        else ()
    )
    resolution = _resolution(entry)
    # Step 5n: Build record dict.
    return {
        "target_id": parsed["target_id"],
        "description": parsed["description"],
        "pdb_id": pdb_id,
        "assembly_id": assembly_id,
        "stratum": parsed["stratum"],
        "chain_count": len(pinned_chains),
        "chain_ids": list(synthetic_chain_ids),
        "chain_lengths": list(lengths),
        "chains": list(pinned_chains),
        "sequence": sequence_value,
        "sequence_sha256": parsed["sequence_sha256"],
        "total_residues": parsed["total_length"],
        "ca_counts": list(ca_counts),
        "coordinate_coverage": list(coverages),
        "minimum_coordinate_coverage": min(coverages),
        "cluster_identity": None,
        "cluster_ids": [],
        "cluster_signature": "",
        "release_date": parsed["release_date"],
        "experimental_methods": list(methods),
        "resolution_angstrom": resolution,
        "source_url": mmcif_url,
        "source_mmcif_sha256": source_mmcif_sha256,
        "reference_sha256": reference_sha256,
        "reference_bytes": reference_bytes,
        "selection_rank": 0,  # assigned later
    }


def reconstruct_benchmark_dataset(
    output: Path,
    specification: BenchmarkSpec,
    target_list: Path,
    *,
    workers: int = 12,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Reconstruct a benchmark corpus from a pinned target list + mmCIF assemblies."""
    spec = specification
    # Step 1: Validate workers.
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be an integer >= 1")
    # Step 2: Output-exists guard (always raises, no idempotent short-circuit).
    output_path = output.resolve()
    if output_path.exists():
        raise ValueError(f"reconstruction output already exists: {output_path}")
    # Step 3: Create output.parent (N6).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Step 4: Atomic staging.
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", suffix=".staging", dir=output_path.parent))
    try:
        # Step 5: Read pinned JSONL target list.
        raw_records: list[dict[str, Any]] = []
        for line in target_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("pinned target list contains a non-object record")
            raw_records.append(payload)
        # Map stratum name -> BenchmarkStratum.
        stratum_by_name = {s.name: s for s in spec.strata}
        # Parse and group by stratum; reject duplicate target IDs (P1).
        targets_by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen_target_ids: set[str] = set()
        for raw in raw_records:
            parsed = _parse_target_record(raw)
            target_id = parsed["target_id"]
            if target_id in seen_target_ids:
                raise ValueError(f"pinned target list contains duplicate target_id: {target_id}")
            seen_target_ids.add(target_id)
            if parsed["stratum"] not in stratum_by_name:
                raise ValueError(f"pinned target {target_id} has unknown stratum: {parsed['stratum']}")
            targets_by_stratum[parsed["stratum"]].append(raw)
        # Step 6: Reconstruct each target in parallel, wrapped in per-target try/except (N3).
        selected: list[dict[str, Any]] = []
        rejection_counts: Counter[str] = Counter()
        all_targets: list[tuple[dict[str, Any], BenchmarkStratum]] = []
        for stratum_name, records in targets_by_stratum.items():
            stratum = stratum_by_name[stratum_name]
            for record in records:
                all_targets.append((record, stratum))

        def _reconstruct_one(
            record: Mapping[str, Any],
            stratum: BenchmarkStratum,
            progress_fn: Callable[[str], None],
        ) -> tuple[Mapping[str, Any], dict[str, Any] | None, str | None]:
            try:
                return record, _materialize_reconstruction_target(record, spec, stratum), None
            except CandidateRejected as exc:
                reason = str(exc)
                progress_fn(f"rejected {record.get('target_id', '?')}: {reason}")
                return record, None, reason
            except Exception as exc:
                progress_fn(f"failed {record.get('target_id', '?')}: {exc}")
                return record, None, str(exc)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = list(
                executor.map(
                    lambda record, stratum: _reconstruct_one(record, stratum, progress),
                    [r for r, s in all_targets],
                    [s for r, s in all_targets],
                )
            )
        for _record, result, reason in futures:
            if reason is not None:
                rejection_counts[reason] += 1
                continue
            assert result is not None
            selected.append(result)
        # Step 7: Sort selected records by target_id. Assign selection_rank.
        selected.sort(key=lambda item: item["target_id"])
        for index, record in enumerate(selected):
            record["selection_rank"] = index
        # Step 8: Per-stratum count validation.
        strata_counts = Counter(record["stratum"] for record in selected)
        for stratum in spec.strata:
            if strata_counts.get(stratum.name, 0) != stratum.count:
                raise BenchmarkCurationFailed(
                    f"reconstruction stratum {stratum.name} has {strata_counts.get(stratum.name, 0)} "
                    f"targets, expected {stratum.count}"
                )
        # Step 9: Materialize dataset.
        reconstruction_selection = {
            "monomers": "pinned target list (reconstruction)",
            "complexes": "pinned target list (reconstruction)",
            "candidate_order": "pinned target_id sort order",
            "rejections": {},
        }
        dataset = _materialize_dataset(
            staging,
            spec,
            selected,
            rejection_counts,
            source_key="source_mmcif_sha256",
            fingerprint_prefix=b"afdb-pdb-benchmark-mmcif-v1\0",
            selection_description=reconstruction_selection,
        )
    except Exception:
        progress(f"Benchmark reconstruction did not complete; staging retained at {staging}")
        raise
    # Step 10: Publish.
    os.replace(staging, output_path)
    # Step 11: Set dataset["output"] (N7).
    dataset["output"] = str(output_path)
    progress(f"Reconstructed {dataset['targets']} targets at {output_path}")
    return dataset


def remove_incomplete_staging(path: Path) -> None:
    """Remove a partially materialized staging directory (idempotent)."""
    if path.is_dir():
        shutil.rmtree(path)


__all__ = [
    "BenchmarkCurationFailed",
    "CandidateRejected",
    "PyArrowUnavailable",
    "prepare_benchmark_dataset",
    "reconstruct_benchmark_dataset",
    "remove_incomplete_staging",
]
