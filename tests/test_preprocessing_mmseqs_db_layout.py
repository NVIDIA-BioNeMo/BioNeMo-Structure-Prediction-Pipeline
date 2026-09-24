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

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "preprocessing_mmseqs_db_layout", ROOT / "containers" / "scripts" / "preprocessing_mmseqs_db_layout.py"
)
assert SPEC and SPEC.loader
LAYOUT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAYOUT
SPEC.loader.exec_module(LAYOUT)


def make_db(
    root: Path,
    name: str,
    records: list[bytes] | None = None,
    *,
    dbtype: int = LAYOUT.ALIGNMENT_RES,
    index_text: bytes | None = None,
    shards: list[bytes] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.dbtype").write_bytes(dbtype.to_bytes(4, "little"))
    if records is not None:
        data = b"".join(records)
        if shards is None:
            (root / name).write_bytes(data)
        else:
            assert b"".join(shards) == data
            for number, shard in enumerate(shards):
                (root / f"{name}.{number}").write_bytes(shard)
        if index_text is None:
            offset = 0
            rows: list[bytes] = []
            for key, record in enumerate(records, 1):
                rows.append(f"{key} {offset} {len(record)}\n".encode())
                offset += len(record)
            index_text = b"".join(rows)
    (root / f"{name}.index").write_bytes(index_text if index_text is not None else b"")


def targets(root: Path, *, name: str = "res") -> tuple[Path, Path, dict[str, object]]:
    output = root.parent / "outputs" / root.name
    manifest, keys = output / "manifest.json", output / "keys.txt"
    report = LAYOUT.result_target_keys(root, name, manifest, keys)
    return manifest, keys, report


def test_result_target_keys_handles_merged_sharded_duplicates_and_zero_length_shards(tmp_path: Path) -> None:
    make_db(
        tmp_path,
        "res",
        [b"10 x\n10 duplicate\n\0", b"11 x\n\0"],
        shards=[b"10 x\n10 duplicate\n\0", b"", b"11 x\n\0"],
    )
    manifest, keys, report = targets(tmp_path)
    assert keys.read_bytes() == b"10\n10\n11\n"
    assert report["schema_version"] == 2
    assert report["kind"] == "mmseqs-result-target-keys"
    assert report["source_basename"] == "res"
    assert report["target_keys"]["target_key_occurrence_count"] == 3
    assert manifest.stat().st_mode & 0o777 == 0o400


def test_result_target_keys_labels_expanded_result_namespace(tmp_path: Path) -> None:
    make_db(tmp_path, "res_exp", [b"100 x\n\0"])
    _, keys, report = targets(tmp_path, name="res_exp")
    assert keys.read_bytes() == b"100\n"
    assert report["source_basename"] == "res_exp"
    assert "target namespace" in report["target_keys"]["semantics"]


@pytest.mark.parametrize(
    ("suffixes", "message"),
    [(["0", "2"], "non-contiguous"), (["00"], "leading-zero"), (["0"], "ambiguous")],
)
def test_layout_rejects_ambiguous_gap_and_alias_shards(tmp_path: Path, suffixes: list[str], message: str) -> None:
    make_db(tmp_path, "res", None, index_text=b"")
    if suffixes == ["0"]:
        (tmp_path / "res").write_bytes(b"x")
    for suffix in suffixes:
        (tmp_path / f"res.{suffix}").write_bytes(b"x")
    with pytest.raises(LAYOUT.LayoutError, match=message):
        LAYOUT.inspect_layout(tmp_path, "res")


def test_layout_rejects_huge_numeric_shard_before_range_allocation(tmp_path: Path) -> None:
    make_db(tmp_path, "res", None, index_text=b"")
    (tmp_path / f"res.{LAYOUT.MAX_DATA_SHARDS}").write_bytes(b"x")
    with pytest.raises(LAYOUT.LayoutError, match="exceeds cap"):
        LAYOUT.inspect_layout(tmp_path, "res")


@pytest.mark.parametrize(
    "index_text",
    [b"1 0\n", b"1 x 1\n", b"1 0 1", b"1 0 1\n1 1 1\n", b"1 0 2\n2 1 1\n", b"1 9 1\n"],
)
def test_layout_rejects_malformed_duplicate_overlap_and_out_of_range_index(tmp_path: Path, index_text: bytes) -> None:
    make_db(tmp_path, "res", [b"x\0", b"y\0"], index_text=index_text)
    with pytest.raises(LAYOUT.LayoutError):
        LAYOUT.inspect_layout(tmp_path, "res")


def test_layout_records_sparse_index_gaps_but_rejects_cross_shard_extent(tmp_path: Path) -> None:
    make_db(tmp_path, "res", [b"a\0", b"b\0"], index_text=b"1 0 1\n2 2 1\n")
    summary = LAYOUT.inspect_layout(tmp_path, "res").index_summary
    assert summary["indexed_span_bytes"] == 3
    assert summary["sparse_gap_count"] == 2
    assert summary["sparse_gap_bytes"] == 2
    assert summary["source_sha256"] == summary["canonical_sha256"]
    make_db(tmp_path, "cross", [b"a\0", b"b\0"], shards=[b"a", b"\0b\0"], index_text=b"1 0 2\n")
    with pytest.raises(LAYOUT.LayoutError, match="crosses a shard"):
        LAYOUT.inspect_layout(tmp_path, "cross")


def test_layout_accepts_upstream_whitespace_delimited_index_rows(tmp_path: Path) -> None:
    make_db(tmp_path, "res", [b"a\0", b"b\0"], index_text=b"1\t0\t2\n2  2  2\n")
    layout = LAYOUT.inspect_layout(tmp_path, "res")
    assert [(entry.key, entry.offset, entry.length) for entry in layout.index] == [(1, 0, 2), (2, 2, 2)]
    assert layout.index_summary["source_sha256"] != layout.index_summary["canonical_sha256"]


def test_confined_lndb_symlinks_are_copied_to_regular_files(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    backing = source / "backing"
    make_db(backing, "payload", [b"binary\0profile\0"], dbtype=0x40000001)
    for suffix in ("", ".dbtype", ".index"):
        (source / f"prof_res_h{suffix}").symlink_to(backing / f"payload{suffix}")
    for name in ("qdb", "qdb_h", "prof_res", "res_exp_realign"):
        make_db(source, name, [b"opaque\0"], dbtype=1)
    (source / "qdb.lookup").write_bytes(b"lookup\0")
    manifest = tmp_path / "copy.json"
    report = LAYOUT.copy_databases(
        source, destination, ["qdb", "qdb_h", "prof_res", "prof_res_h", "res_exp_realign"], manifest
    )
    assert report["outcome"] == "success"
    assert not (destination / "prof_res_h").is_symlink()
    assert (destination / "prof_res_h").read_bytes() == b"binary\0profile\0"


def test_linked_source_root_is_resolved_before_confined_component_links(tmp_path: Path) -> None:
    backing, linked_root = tmp_path / "backing", tmp_path / "linked-root"
    logical = backing / "logical"
    make_db(logical, "payload", [b"1 x\n\0"])
    for suffix in ("", ".dbtype", ".index"):
        (logical / f"res{suffix}").symlink_to(logical / f"payload{suffix}")
    linked_root.symlink_to(logical, target_is_directory=True)
    manifest, keys, report = targets(linked_root)
    assert report["outcome"] == "success"
    assert keys.read_bytes() == b"1\n"
    assert manifest.exists()


def test_layout_rejects_escaping_symlink_and_nonregular_component(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"
    make_db(outside, "payload", [b"x\0"])
    tmp_path.mkdir(exist_ok=True)
    for suffix in ("", ".dbtype", ".index"):
        (tmp_path / f"res{suffix}").symlink_to(outside / f"payload{suffix}")
    with pytest.raises(LAYOUT.LayoutError, match="escapes"):
        LAYOUT.inspect_layout(tmp_path, "res")
    for suffix in ("", ".dbtype", ".index"):
        (tmp_path / f"res{suffix}").unlink()
    make_db(tmp_path, "res", [b"x\0"])
    (tmp_path / "res.lookup").mkdir()
    with pytest.raises(LAYOUT.LayoutError, match="regular"):
        LAYOUT.inspect_layout(tmp_path, "res")


def test_layout_rejects_missing_required_companion_and_logical_link_retarget(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_db(source, "res", [b"1 x\n\0"])
    (source / "res.dbtype").unlink()
    with pytest.raises(LAYOUT.LayoutError, match="missing required component"):
        LAYOUT.inspect_layout(source, "res")

    backing_a, backing_b = source / "backing-a", source / "backing-b"
    make_db(backing_a, "payload", [b"1 x\n\0"])
    make_db(backing_b, "payload", [b"1 x\n\0"])
    for suffix in ("", ".dbtype", ".index"):
        (source / f"linked{suffix}").symlink_to(backing_a / f"payload{suffix}")
    layout = LAYOUT.inspect_layout(source, "linked")
    (source / "linked").unlink()
    (source / "linked").symlink_to(backing_b / "payload")
    with pytest.raises(LAYOUT.LayoutError, match="symlink changed"):
        LAYOUT._verify_components(layout.components())


def test_result_targets_reject_bad_result_framing_key_and_flags(tmp_path: Path) -> None:
    make_db(tmp_path, "res", [b"10\0bad\0"])
    with pytest.raises(LAYOUT.LayoutError, match="NUL framing"):
        targets(tmp_path)
    make_db(tmp_path, "res", [b"01 x\n\0"])
    with pytest.raises(LAYOUT.LayoutError, match="canonical decimal"):
        targets(tmp_path)
    make_db(tmp_path, "res", [b"1 x\n\0"], dbtype=0x80000005)
    with pytest.raises(LAYOUT.LayoutError, match="uncompressed"):
        targets(tmp_path)
    make_db(tmp_path, "res", [b"1 x\n\0"], dbtype=0x00080005)
    with pytest.raises(LAYOUT.LayoutError, match="unpadded"):
        targets(tmp_path)
    make_db(tmp_path, "res", [b"1 x\n\0"], dbtype=99)
    with pytest.raises(LAYOUT.LayoutError, match="ALIGNMENT_RES"):
        targets(tmp_path)


def test_result_targets_accept_both_result_types_empty_output_and_data_absent(tmp_path: Path) -> None:
    make_db(tmp_path, "res", [b"\0"], dbtype=LAYOUT.PREFILTER_RES)
    _, keys, report = targets(tmp_path)
    assert keys.read_bytes() == b""
    assert report["target_keys"]["target_key_occurrence_count"] == 0
    absent = tmp_path / "absent"
    make_db(absent, "res", None, index_text=b"1 0 7\n")
    manifest, absent_keys, absent_report = targets(absent)
    assert absent_report["outcome"] == "data_absent"
    assert not absent_keys.exists()
    assert absent_report["source"]["index_summary"]["row_count"] == 1
    assert absent_report["target_keys"]["result_record_count"] == 1
    assert manifest.exists()


def test_data_absent_validates_literal_dbtype_and_refuses_stale_keys(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    make_db(source, "res", None, dbtype=0x00000005, index_text=b"1 0 7\n")
    stale_keys = output / "keys.txt"
    stale_keys.parent.mkdir()
    stale_keys.write_bytes(b"stale\n")
    with pytest.raises(LAYOUT.LayoutError, match="stale"):
        LAYOUT.result_target_keys(source, "res", output / "manifest.json", stale_keys)
    stale_keys.unlink()
    report = LAYOUT.result_target_keys(source, "res", output / "manifest.json", stale_keys)
    assert report["source"]["dbtype"]["raw_hex"] == "05000000"
    assert report["source"]["missing_optional_companions"] == [".lookup", ".source"]
    assert report["source_basename"] == "res"
    make_db(source, "bad", None, dbtype=0x80000005, index_text=b"")
    with pytest.raises(LAYOUT.LayoutError, match="uncompressed"):
        LAYOUT.result_target_keys(source, "bad", output / "bad.json", output / "bad.txt")


def test_whole_directory_inventory_is_bounded_lstat_evidence_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "unrelated").write_bytes(b"x")
    (source / "nested").mkdir()
    (source / "linked").symlink_to(source / "unrelated")
    manifest = output / "inventory.json"

    report = LAYOUT.directory_inventory(source, manifest)
    assert report["kind"] == "mmseqs-directory-inventory"
    assert report["entry_count"] == 3
    assert {item["name"]: item["type"] for item in report["entries"]} == {
        "linked": "symlink",
        "nested": "other",
        "unrelated": "regular",
    }
    assert next(item for item in report["entries"] if item["name"] == "linked")["link_target"] == str(
        source / "unrelated"
    )
    LAYOUT.directory_inventory(source, manifest)
    (source / "later").write_bytes(b"y")
    with pytest.raises(LAYOUT.LayoutError, match="immutable output"):
        LAYOUT.directory_inventory(source, manifest)

    fresh = output / "bounded.json"
    monkeypatch.setattr(LAYOUT, "MAX_DIRECTORY_ENTRIES", 1)
    with pytest.raises(LAYOUT.LayoutError, match="entry cap"):
        LAYOUT.directory_inventory(source, fresh)
    monkeypatch.setattr(LAYOUT, "MAX_DIRECTORY_ENTRIES", 10)
    monkeypatch.setattr(LAYOUT, "MAX_DIRECTORY_NAME_BYTES", 1)
    with pytest.raises(LAYOUT.LayoutError, match="name exceeds"):
        LAYOUT.directory_inventory(source, fresh)


def test_result_targets_accept_final_row_without_newline_before_nul(tmp_path: Path) -> None:
    make_db(tmp_path, "res", [b"42 annotation\0"])
    _, keys, report = targets(tmp_path)
    assert keys.read_bytes() == b"42\n"
    assert report["target_keys"]["target_key_occurrence_count"] == 1


def test_index_bounds_uint64_and_sparse_gap_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_db(tmp_path, "res", [b"a\0", b"b\0"], index_text=b"1 0 1\n2 2 1\n")
    summary = LAYOUT.inspect_layout(tmp_path, "res").index_summary
    assert summary["sparse_gap_count"] == 2
    assert summary["sparse_gap_bytes"] == 2
    make_db(tmp_path, "too_large", [b"a\0"], index_text=b"18446744073709551616 0 1\n")
    with pytest.raises(LAYOUT.LayoutError, match="uint64"):
        LAYOUT.inspect_layout(tmp_path, "too_large")
    monkeypatch.setattr(LAYOUT, "MAX_INDEX_ENTRIES", 1)
    with pytest.raises(LAYOUT.LayoutError, match="entry cap"):
        LAYOUT.inspect_layout(tmp_path, "res")


def test_single_zero_shard_and_optional_source_are_preserved_in_exact_copy(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    for name in LAYOUT.REQUIRED_COPY_DATABASES:
        make_db(source, name, [b"opaque\0"], dbtype=1)
    (source / "qdb.lookup").write_bytes(b"lookup\0")
    (source / "prof_res.source").write_bytes(b"source\0")
    (source / "res_exp_realign").unlink()
    (source / "res_exp_realign.0").write_bytes(b"opaque\0")
    report = LAYOUT.copy_databases(
        source, destination, list(reversed(LAYOUT.REQUIRED_COPY_DATABASES)), tmp_path / "copy.json"
    )
    assert report["outcome"] == "success"
    assert (destination / "res_exp_realign.0").is_file()
    assert (destination / "prof_res.source").read_bytes() == b"source\0"
    assert {item.name for item in destination.iterdir()} == {
        component["basename"] for database in report["databases"] for component in database["destination"]["components"]
    }


def test_copy_requires_exact_five_and_rejects_unsafe_destination_state(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    for name in LAYOUT.REQUIRED_COPY_DATABASES:
        make_db(source, name, [b"opaque\0"], dbtype=1)
    (source / "qdb.lookup").write_bytes(b"lookup\0")
    with pytest.raises(LAYOUT.LayoutError, match="exactly"):
        LAYOUT.copy_databases(source, destination, ["qdb"], tmp_path / "copy.json")
    destination.mkdir()
    (destination / "unexpected").write_bytes(b"x")
    with pytest.raises(LAYOUT.LayoutError, match="not fresh"):
        LAYOUT.copy_databases(source, destination, list(LAYOUT.REQUIRED_COPY_DATABASES), tmp_path / "copy.json")


def test_outputs_and_destinations_reject_source_overlap_and_symlinks(tmp_path: Path) -> None:
    make_db(tmp_path, "res", [b"1 x\n\0"])
    nested_output = tmp_path / "not-created" / "manifest.json"
    with pytest.raises(LAYOUT.LayoutError, match="outside source"):
        LAYOUT.result_target_keys(tmp_path, "res", nested_output, tmp_path.parent / "keys.txt")
    assert not nested_output.parent.exists()
    output = tmp_path.parent / "linked-output"
    output.symlink_to(tmp_path.parent / "real-output")
    with pytest.raises(LAYOUT.LayoutError, match="symlink"):
        LAYOUT.result_target_keys(tmp_path, "res", output / "manifest.json", tmp_path.parent / "keys.txt")
    source = tmp_path / "copy-source"
    for name in LAYOUT.REQUIRED_COPY_DATABASES:
        make_db(source, name, [b"opaque\0"], dbtype=1)
    (source / "qdb.lookup").write_bytes(b"lookup\0")
    with pytest.raises(LAYOUT.LayoutError, match="non-overlapping"):
        LAYOUT.copy_databases(
            source, source / "destination", list(LAYOUT.REQUIRED_COPY_DATABASES), tmp_path.parent / "copy.json"
        )
    assert not (source / "destination").exists()


def test_expansion_rehashes_each_data_component_once_after_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [b"1 first\n\0", b"2 second\n\0", b"3 third\n\0"]
    make_db(tmp_path, "res", records)
    original_hash = LAYOUT._hash_regular
    data_hashes = 0

    def counted_hash(path: Path) -> tuple[object, str]:
        nonlocal data_hashes
        if path.name == "res":
            data_hashes += 1
        return original_hash(path)

    monkeypatch.setattr(LAYOUT, "_hash_regular", counted_hash)
    targets(tmp_path)
    assert data_hashes == 2


def test_cli_reports_data_absent_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    make_db(source, "res", None, index_text=b"")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "layout",
            "result-target-keys",
            "--source-root",
            str(source),
            "--basename",
            "res",
            "--manifest",
            str(output / "manifest.json"),
            "--keys-output",
            str(output / "keys.txt"),
        ],
    )
    assert LAYOUT.main() == 0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "layout",
            "result-target-keys",
            "--source-root",
            str(source),
            "--basename",
            "missing",
            "--manifest",
            str(output / "missing.json"),
            "--keys-output",
            str(output / "missing.txt"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        LAYOUT.main()
    assert error.value.code == 2


def test_immutable_outputs_reuse_equal_and_refuse_changed_source(tmp_path: Path) -> None:
    make_db(tmp_path, "res", [b"1 x\n\0"])
    manifest, keys, _ = targets(tmp_path)
    LAYOUT.result_target_keys(tmp_path, "res", manifest, keys)
    (tmp_path / "res").write_bytes(b"2 x\n\0")
    with pytest.raises(LAYOUT.LayoutError, match="immutable output"):
        LAYOUT.result_target_keys(tmp_path, "res", manifest, keys)


def test_copy_requires_qdb_lookup_and_root_independent_identity(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    for name in ("qdb", "qdb_h", "prof_res", "prof_res_h", "res_exp_realign"):
        make_db(source, name, [b"binary\0\xff\0"], dbtype=1)
    with pytest.raises(LAYOUT.LayoutError, match=r"qdb\.lookup"):
        LAYOUT.copy_databases(
            source, destination, ["qdb", "qdb_h", "prof_res", "prof_res_h", "res_exp_realign"], tmp_path / "x"
        )
    (source / "qdb.lookup").write_bytes(b"lookup\0")
    report = LAYOUT.copy_databases(
        source, destination, ["qdb", "qdb_h", "prof_res", "prof_res_h", "res_exp_realign"], tmp_path / "copy.json"
    )
    assert all(
        item["source"]["logical_identity"] == item["destination"]["logical_identity"] for item in report["databases"]
    )
    assert (destination / "prof_res").read_bytes() == b"binary\0\xff\0"


def test_copy_immutable_reuse_and_refusal(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    for name in ("qdb", "qdb_h", "prof_res", "prof_res_h", "res_exp_realign"):
        make_db(source, name, [b"opaque\0"], dbtype=1)
    (source / "qdb.lookup").write_bytes(b"lookup\0")
    manifest = tmp_path / "copy.json"
    basenames = ["qdb", "qdb_h", "prof_res", "prof_res_h", "res_exp_realign"]
    LAYOUT.copy_databases(source, destination, basenames, manifest)
    LAYOUT.copy_databases(source, destination, basenames, manifest)
    (source / "prof_res").write_bytes(b"changed\0")
    with pytest.raises(LAYOUT.LayoutError, match="immutable copied component"):
        LAYOUT.copy_databases(source, destination, basenames, manifest)


def _force_stat_drift(path: Path) -> None:
    """Make a just-rewritten file's stat deterministically differ from any
    earlier snapshot, on any host.

    Kernel inode timestamps use a coarse clock, so a rewrite that happens
    within one tick of the original snapshot keeps the same ``mtime_ns`` on
    fast machines and the stat-drift defense would not be exercised. Bumping
    the timestamp explicitly pins the drift deterministically.
    """
    current = path.stat()
    os.utime(path, ns=(current.st_atime_ns + 1_000_000_000, current.st_mtime_ns + 1_000_000_000))


def test_parsing_and_extent_streaming_reject_component_drift(tmp_path: Path) -> None:
    make_db(tmp_path, "res", [b"1 x\n\0"])
    index_component = LAYOUT._component(tmp_path, "res.index", required=True)
    assert index_component is not None
    (tmp_path / "res.index").write_bytes(b"1 0 7\n")
    _force_stat_drift(tmp_path / "res.index")
    with pytest.raises(LAYOUT.LayoutError, match="stat drifted"):
        LAYOUT._parse_index(index_component, stream_size=7)
    (tmp_path / "res.index").write_bytes(b"1 0 5\n")
    layout = LAYOUT.inspect_layout(tmp_path, "res")
    (tmp_path / "res").write_bytes(b"2 x\n\0")
    _force_stat_drift(tmp_path / "res")
    with pytest.raises(LAYOUT.LayoutError, match="stat drifted"):
        list(LAYOUT._extent_chunks(layout, 0, 5))
