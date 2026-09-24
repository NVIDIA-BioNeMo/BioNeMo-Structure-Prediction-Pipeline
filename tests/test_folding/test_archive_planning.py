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

"""Result association and non-executing archive-plan tests for issue #52."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_archive import (
    ArchivePlanOptions,
    archive_plan_from_mapping,
    archive_plan_options_from_mapping,
    folding_result_inventory_from_mapping,
)
from bspp.orchestration.contract.folding_index import FoldingIndexRecord, make_folding_index
from bspp.orchestration.runtime.folding.archive import (
    parse_archive_manifest_jsonl,
    plan_folding_archives,
    render_archive_manifest_jsonl,
)
from bspp.orchestration.runtime.folding.results import (
    normalize_result_protein_id,
    scan_folding_results,
)


def _record(ordinal: int, protein_id: str, *, length: int = 2) -> FoldingIndexRecord:
    return FoldingIndexRecord(
        source_ordinal=ordinal,
        protein_id=protein_id,
        msa_path=f"/msa/{protein_id}.a3m",
        query_sequence="A" * length,
        sequence_length=length,
        chain_lengths=(length,),
        chain_count=1,
        chain_cardinalities=(1,),
        msa_depth=1,
        total_length=length,
    )


def _write(path: Path, text: str = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _complete_inventory(tmp_path: Path, count: int = 5):
    index = make_folding_index(tuple(_record(ordinal, f"model-{ordinal}") for ordinal in range(count)))
    root = tmp_path / "results"
    for ordinal in range(count):
        _write(root / f"model-{ordinal}_unrelaxed_rank_001_model_1.pdb")
        _write(root / f"model-{ordinal}_scores_rank_001_model_1.json", "{}")
    return scan_folding_results(index, root)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("AF_1_unrelaxed_rank_001_model_1.pdb", "AF-1"),
        ("AF_2_relaxed_rank_007_model_3.pdb", "AF-2"),
        ("AF_3_scores_rank_001_model_1.json", "AF-3"),
        ("AF_4_model_1_multimer_v3_scores.json", "AF-4"),
        ("AF_5_scores.json", "AF-5"),
        ("plain.pdb", "plain"),
        ("plain.json", "plain"),
    ],
)
def test_result_filename_normalization_matches_supported_baseline_forms(filename: str, expected: str) -> None:
    assert normalize_result_protein_id(Path(filename)) == expected


def test_scan_classifies_complete_one_sided_duplicate_collision_unplanned_and_malformed(tmp_path: Path) -> None:
    protein_ids = ("alpha", "beta", "gamma", "delta", "AF_1", "AF-1", "missing")
    index = make_folding_index(tuple(_record(ordinal, protein_id) for ordinal, protein_id in enumerate(protein_ids)))
    root = tmp_path / "run"
    _write(root / "alpha_unrelaxed_rank_001_x.pdb")
    _write(root / "predictions" / "alpha_scores_rank_001_x.json", "{}")
    _write(root / "predictions" / "beta_relaxed_rank_001_x.pdb")
    _write(root / "predictions" / "gamma_model_1_multimer_v3_scores.json", "{}")
    _write(root / "gpu0" / "delta_unrelaxed_rank_001_x.pdb")
    _write(root / "gpu0" / "delta_relaxed_rank_002_x.pdb")
    _write(root / "gpu0" / "delta_scores_rank_001_x.json", "{}")
    _write(root / "AF_1_unrelaxed_rank_001_x.pdb")
    _write(root / "AF_1_scores_rank_001_x.json", "{}")
    _write(root / "orphan.pdb")
    _write(root / "_rank_001_x.pdb")
    _write(root / "ignore.txt")

    inventory = scan_folding_results(index, root)

    assert tuple((item.protein_id, item.status) for item in inventory.associations) == (
        ("alpha", "complete"),
        ("beta", "pdb-only"),
        ("gamma", "json-only"),
        ("delta", "duplicate"),
        ("AF_1", "identity-collision"),
        ("AF-1", "identity-collision"),
        ("missing", "missing"),
    )
    assert tuple((item.path.rsplit("/", 1)[-1], item.reason) for item in inventory.unmatched) == (
        ("_rank_001_x.pdb", "malformed-name"),
        ("orphan.pdb", "unplanned-identity"),
    )
    assert folding_result_inventory_from_mapping(inventory.to_mapping()) == inventory

    plan = plan_folding_archives(
        inventory,
        ArchivePlanOptions(
            run_tag="classified",
            stage_root=str(tmp_path / "stage"),
            archive_root=str(tmp_path / "archives"),
            shuffle=False,
        ),
    )
    assert tuple(protein_id for batch in plan.batches for protein_id in batch.protein_ids) == ("alpha",)


def test_scan_restores_source_order_when_primary_index_is_length_sorted(tmp_path: Path) -> None:
    index = make_folding_index(
        (_record(0, "long", length=5), _record(1, "short", length=2)),
        sort_by_length=True,
    )
    root = tmp_path / "empty-results"
    root.mkdir()

    inventory = scan_folding_results(index, root)

    assert tuple(item.protein_id for item in inventory.associations) == ("long", "short")
    assert tuple(item.source_ordinal for item in inventory.associations) == (0, 1)


def test_scan_deduplicates_layout_directories_without_resolving_each_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = make_folding_index((_record(0, "alpha"),))
    root = tmp_path / "results"
    _write(root / "alpha_unrelaxed_rank_001_x.pdb")
    _write(root / "alpha_scores_rank_001_x.json", "{}")
    (root / "predictions").symlink_to(root, target_is_directory=True)
    resolved: list[Path] = []
    original_resolve = Path.resolve

    def recording_resolve(path: Path, *, strict: bool = False) -> Path:
        resolved.append(path)
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", recording_resolve)

    inventory = scan_folding_results(index, root)

    assert inventory.associations[0].status == "complete"
    assert inventory.associations[0].pdb_paths == (str(root / "alpha_unrelaxed_rank_001_x.pdb"),)
    assert inventory.associations[0].json_paths == (str(root / "alpha_scores_rank_001_x.json"),)
    assert resolved == [root, root / "predictions"]


def test_archive_planning_batches_only_complete_pairs_and_freezes_safe_argv(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path)
    options = ArchivePlanOptions(
        run_tag="run-A",
        proteins_per_archive=2,
        stage_root=str(tmp_path / "stage root"),
        archive_root=str(tmp_path / "archive root"),
        lz4_executable=str(tmp_path / "tool dir" / "lz4"),
        start_index=7,
        shuffle=False,
    )

    first = plan_folding_archives(inventory, options)
    second = plan_folding_archives(inventory, options)

    assert first == second
    assert tuple(batch.archive_name for batch in first.batches) == (
        "run-A_00007.tar.lz4",
        "run-A_00008.tar.lz4",
        "run-A_00009.tar.lz4",
    )
    assert first.batches[0].protein_ids == ("model-0", "model-1")
    assert first.batches[0].tar_argv == (
        "tar",
        "-cf",
        "-",
        "-C",
        str(tmp_path / "stage root" / "run-A_00007"),
        ".",
    )
    assert first.batches[0].lz4_argv == (str(tmp_path / "tool dir" / "lz4"), "-1", "-")
    assert first.batches[0].stdout_path == str(tmp_path / "archive root" / "run-A_00007.tar.lz4")
    assert tuple((member.protein_id, member.kind) for member in first.batches[0].members) == (
        ("model-0", "pdb"),
        ("model-0", "json"),
        ("model-1", "pdb"),
        ("model-1", "json"),
    )
    assert all(batch.manifest_record.archive_status == "planned" for batch in first.batches)


def test_seeded_shuffle_limits_batches_and_is_deterministic(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path, count=8)
    options = ArchivePlanOptions(
        run_tag="seeded",
        proteins_per_archive=3,
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=True,
        shuffle_seed=42,
        max_archives=2,
    )

    first = plan_folding_archives(inventory, options)
    second = plan_folding_archives(inventory, options)

    assert first == second
    assert tuple(pid for batch in first.batches for pid in batch.protein_ids) == (
        "model-1",
        "model-5",
        "model-0",
        "model-7",
        "model-2",
        "model-4",
    )

    no_batches = plan_folding_archives(inventory, replace(options, max_archives=0))
    assert no_batches.batches == ()


def test_manifest_replay_is_idempotent_and_force_uses_fresh_archive_indices(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path, count=3)
    options = ArchivePlanOptions(
        run_tag="resume",
        proteins_per_archive=2,
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=False,
    )
    original = plan_folding_archives(inventory, options)
    manifest = render_archive_manifest_jsonl(original)
    replayed = parse_archive_manifest_jsonl(manifest)

    assert replayed == tuple(batch.manifest_record for batch in original.batches)
    assert render_archive_manifest_jsonl(original) == manifest
    assert plan_folding_archives(inventory, options, prior_manifest=replayed).batches == ()

    forced = plan_folding_archives(inventory, options, prior_manifest=replayed, force=True)
    assert forced.batches[0].archive_index == 2
    assert forced.batches[0].protein_ids == ("model-0", "model-1")

    appended = parse_archive_manifest_jsonl(manifest + render_archive_manifest_jsonl(forced))
    assert plan_folding_archives(inventory, options, prior_manifest=appended).batches == ()
    forced_again = plan_folding_archives(inventory, options, prior_manifest=appended, force=True)
    assert tuple(batch.archive_index for batch in forced_again.batches) == (4, 5)


def test_archive_plan_rejects_disappeared_member_and_never_claims_archive_success(tmp_path: Path) -> None:
    inventory = _complete_inventory(tmp_path, count=1)
    Path(inventory.associations[0].json_paths[0]).unlink()
    options = ArchivePlanOptions(
        run_tag="missing",
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
    )

    with pytest.raises(ValueError, match="not a readable file"):
        plan_folding_archives(inventory, options)
    with pytest.raises(ValueError, match="archive_status"):
        parse_archive_manifest_jsonl(
            '{"schema_version":1,"archive_status":"archived","archive_batch_id":"x_00000",'
            '"archive_file":"x_00000.tar.lz4","archive_index":0,"run_tag":"x",'
            '"protein_ids":["model-0"],"member_names":["a.pdb","a.json"]}\n'
        )


def test_archive_contract_round_trip_is_frozen_versioned_and_fail_closed(tmp_path: Path) -> None:
    options = ArchivePlanOptions(
        run_tag="roundtrip",
        stage_root=str(tmp_path / "stage"),
        archive_root=str(tmp_path / "archives"),
        shuffle=False,
    )
    plan = plan_folding_archives(
        _complete_inventory(tmp_path, count=2),
        options,
    )

    loaded = archive_plan_from_mapping(plan.to_mapping())

    assert loaded == plan
    assert archive_plan_options_from_mapping(options.to_mapping()) == options
    with pytest.raises(FrozenInstanceError):
        loaded.batches = ()  # type: ignore[misc]
    unknown = plan.to_mapping()
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        archive_plan_from_mapping(unknown)
    unsupported = plan.to_mapping()
    unsupported["schema_version"] = 2
    with pytest.raises(ValueError, match="Unsupported FoldingArchivePlan schema_version 2"):
        archive_plan_from_mapping(unsupported)

    tampered_command = plan.to_mapping()
    batches = tampered_command["batches"]
    assert isinstance(batches, list)
    batch = batches[0]
    assert isinstance(batch, dict)
    batch["tar_argv"] = ["tar", "-cf", "/tmp/unbounded.tar", "."]
    with pytest.raises(ValueError, match="tar_argv"):
        archive_plan_from_mapping(tampered_command)

    with pytest.raises(ValueError, match="schema_version must be declared explicitly"):
        replace(options, schema_version=None)  # type: ignore[arg-type]


def test_archive_runtime_modules_are_non_executing_and_contract_is_dependency_light() -> None:
    import bspp.orchestration.contract.folding_archive as contract_module
    import bspp.orchestration.runtime.folding.archive as archive_module
    import bspp.orchestration.runtime.folding.results as results_module

    contract_tree = ast.parse(Path(contract_module.__file__).read_text())
    contract_imports = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(contract_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    contract_imports |= {
        (node.module or "").split(".", maxsplit=1)[0]
        for node in ast.walk(contract_tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not contract_imports & {"numpy", "pandas", "pyarrow", "subprocess", "bspp.orchestration.runtime"}

    for module in (archive_module, results_module):
        tree = ast.parse(Path(module.__file__).read_text())
        assert not any(
            isinstance(node, ast.Import) and any(alias.name == "subprocess" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert not any(isinstance(node, ast.ImportFrom) and node.module == "subprocess" for node in ast.walk(tree))
