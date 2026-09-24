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

"""Public-seam tests for preprocessing command and package planning."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bspp.orchestration.contract.preprocessing import PreprocessingChunk, PreprocessingFastaRecord
from bspp.orchestration.contract.preprocessing_execution import (
    PreprocessingRuntimeCoordinates,
    PreprocessingScientificConfig,
    PreprocessingScientificIntent,
    PreprocessingSiteConfig,
    afdb_model_id_member_name_conforms,
    expected_member_name,
    materialize_preprocessing_chunk_execution_plan,
    member_name_conforms,
    preprocessing_chunk_execution_intent_from_plan,
    preprocessing_chunk_execution_plan_from_mapping,
    safe_filename,
    scientific_canonical_mapping,
)
from bspp.orchestration.runtime.preprocessing.commands import plan_preprocessing_chunk_execution


def test_chunk_plan_renders_exact_pinned_gpuserver_and_search_argv() -> None:
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("alpha.a3m", "beta.a3m"),
        scientific=_scientific(),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=7, gpu_id=2, submission_counter=3),
    )

    assert plan.gpuserver_argv == (
        "/opt/mm seqs/mmseqs",
        "gpuserver",
        "/db root/uniref30_2302_db",
        "--db-load-mode",
        "0",
        "--prefilter-mode",
        "1",
        "--max-seqs",
        "10000",
    )
    assert plan.gpuserver_environment == (("CUDA_VISIBLE_DEVICES", "2"),)
    assert plan.search_argv == (
        "/opt/colab fold/colabfold_search",
        "--mmseqs",
        "/opt/mm seqs/mmseqs",
        "/scratch input/n7g2/proteins_tranche00_00000.fa",
        "/db root",
        "/scratch output/raw-search/n7g2_3",
        "--use-env",
        "0",
        "--pairing_strategy",
        "1",
        "--pair-mode",
        "unpaired_paired",
        "--filter",
        "1",
        "--db-load-mode",
        "2",
        "--gpu",
        "1",
        "--gpu-server",
        "1",
        "--threads",
        "64",
        "--db1",
        "uniref30_2302_db",
        "--db3",
        "colabfold_envdb_202108_db",
    )
    assert plan.search_environment == (("CUDA_VISIBLE_DEVICES", "2"),)


def test_use_env_renders_search_argv_flag_and_round_trips() -> None:
    default_plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_conforming_records(),
        expected_a3m_members=("AFDB_AF-1234567890123456.a3m", "AFDB_AF-2345678901234567.a3m"),
        scientific=PreprocessingScientificConfig(),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    assert default_plan.scientific.use_env is False
    assert default_plan.search_argv[default_plan.search_argv.index("--use-env") + 1] == "0"

    opt_in_plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_conforming_records(),
        expected_a3m_members=("AFDB_AF-1234567890123456.a3m", "AFDB_AF-2345678901234567.a3m"),
        scientific=PreprocessingScientificConfig(use_env=True),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    assert opt_in_plan.scientific.use_env is True
    assert opt_in_plan.search_argv[opt_in_plan.search_argv.index("--use-env") + 1] == "1"
    # round-trip through the strict loader + _validate_execution_plan
    assert preprocessing_chunk_execution_plan_from_mapping(opt_in_plan.to_mapping()) == opt_in_plan


def test_use_env_opt_in_plan_rejects_tampered_search_argv_token() -> None:
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_conforming_records(),
        expected_a3m_members=("AFDB_AF-1234567890123456.a3m", "AFDB_AF-2345678901234567.a3m"),
        scientific=PreprocessingScientificConfig(use_env=True),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )

    tampered = deepcopy(plan.to_mapping())
    assert isinstance(tampered["search_argv"], list)
    use_env_index = tampered["search_argv"].index("--use-env") + 1
    tampered["search_argv"][use_env_index] = "0"
    with pytest.raises(ValueError, match="pinned command vectors"):
        preprocessing_chunk_execution_plan_from_mapping(tampered)


def test_freshly_authored_paired_plan_is_rejected() -> None:
    """A freshly constructed plan with --pair-mode paired must be rejected (B2)."""
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_conforming_records(),
        expected_a3m_members=("AFDB_AF-1234567890123456.a3m", "AFDB_AF-2345678901234567.a3m"),
        scientific=PreprocessingScientificConfig(use_env=True),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    paired_argv = list(plan.search_argv)
    pm_idx = paired_argv.index("--pair-mode")
    paired_argv[pm_idx + 1] = "paired"
    with pytest.raises(ValueError, match="pinned command vectors"):
        replace(plan, search_argv=tuple(paired_argv))


def test_sealed_paired_evidence_replays_through_from_mapping() -> None:
    """Old sealed evidence with --pair-mode paired still loads via from_mapping (B2)."""
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_conforming_records(),
        expected_a3m_members=("AFDB_AF-1234567890123456.a3m", "AFDB_AF-2345678901234567.a3m"),
        scientific=PreprocessingScientificConfig(schema_version=2, use_env=True),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    legacy_mapping = deepcopy(plan.to_mapping())
    assert isinstance(legacy_mapping["search_argv"], list)
    pm_idx = legacy_mapping["search_argv"].index("--pair-mode")
    legacy_mapping["search_argv"][pm_idx + 1] = "paired"
    replayed = preprocessing_chunk_execution_plan_from_mapping(legacy_mapping)
    assert replayed.search_argv[replayed.search_argv.index("--pair-mode") + 1] == "paired"


def test_legacy_load_does_not_widen_fresh_construction() -> None:
    """Legacy deserialization carries the pair-mode seam locally; fresh construction stays strict."""
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_conforming_records(),
        expected_a3m_members=("AFDB_AF-1234567890123456.a3m", "AFDB_AF-2345678901234567.a3m"),
        scientific=PreprocessingScientificConfig(schema_version=2, use_env=True),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    legacy_mapping = deepcopy(plan.to_mapping())
    assert isinstance(legacy_mapping["search_argv"], list)
    pm_idx = legacy_mapping["search_argv"].index("--pair-mode")
    legacy_mapping["search_argv"][pm_idx + 1] = "paired"
    preprocessing_chunk_execution_plan_from_mapping(legacy_mapping)
    paired_argv = list(plan.search_argv)
    paired_argv[plan.search_argv.index("--pair-mode") + 1] = "paired"
    with pytest.raises(ValueError, match="pinned command vectors"):
        replace(plan, search_argv=tuple(paired_argv))


def test_expected_a3ms_preserve_chunk_and_source_record_association() -> None:
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("alpha result.a3m", "beta.a3m"),
        scientific=_scientific(),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )

    assert tuple(
        (
            expected.chunk_name,
            expected.record_identity,
            expected.source_ordinal,
            expected.source_header,
            expected.member_name,
        )
        for expected in plan.expected_a3ms
    ) == (
        ("proteins_tranche00_00000.fa", "alpha", 0, ">alpha description", "alpha result.a3m"),
        ("proteins_tranche00_00000.fa", "beta", 1, ">beta", "beta.a3m"),
    )


def test_scratch_coordinates_are_ephemeral_while_durable_evidence_is_flat() -> None:
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("alpha.a3m", "beta.a3m"),
        scientific=_scientific(),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=7, gpu_id=2, submission_counter=3),
    )

    assert plan.evidence.scratch_output_directory == "/scratch output/n7g2_3"
    assert plan.evidence.raw_search_output_directory == "/scratch output/raw-search/n7g2_3"
    assert plan.evidence.scratch_log_directory == "/scratch output/logs/n7g2_3"
    assert plan.evidence.scratch_log_path == "/scratch output/logs/n7g2_3/proteins_tranche00_00000.log"
    assert plan.evidence.scratch_record_path == "/scratch output/logs/n7g2_3/proteins_tranche00_00000.record"
    assert plan.evidence.a3m_record_glob == "/scratch output/n7g2_3/*.a3m"
    assert plan.evidence.durable_log_path == "/project logs/proteins_tranche00_00000.log"
    assert plan.evidence.durable_record_path == "/project logs/proteins_tranche00_00000.record"


def test_package_plan_preserves_literal_tar_lz4_scope_and_declared_a3m_inventory() -> None:
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("alpha result.a3m", "beta.a3m"),
        scientific=_scientific(),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=7, gpu_id=2, submission_counter=3),
    )

    package = plan.package
    assert package.staging_directory == "/scratch output/n7g2_3"
    assert package.declared_stage_members == ("alpha result.a3m", "beta.a3m")
    assert package.tar_member_scope == "."
    assert package.scratch_tar_path == "/scratch output/n7g2_3/proteins_tranche00_00000.tar"
    assert package.scratch_lz4_path == "/scratch output/n7g2_3/proteins_tranche00_00000.tar.lz4"
    assert package.tar_argv == (
        "/usr/bin/tar",
        "cf",
        "/scratch output/n7g2_3/proteins_tranche00_00000.tar",
        "-C",
        "/scratch output/n7g2_3",
        ".",
    )
    assert package.lz4_argv == (
        "/opt/lz 4/lz4",
        "-v",
        "-3",
        "/scratch output/n7g2_3/proteins_tranche00_00000.tar",
        "/scratch output/n7g2_3/proteins_tranche00_00000.tar.lz4",
    )
    assert package.durable_tar_path == "/finished msa/proteins_tranche00_00000.tar"
    assert package.durable_lz4_path == "/finished msa/proteins_tranche00_00000.tar.lz4"
    assert package.completed_input_source_path == "/split input/proteins_tranche00_00000.fa"
    assert package.completed_input_path == "/finished input/proteins_tranche00_00000.fa"


def test_optional_max_sequences_is_omitted_and_metagenomic_server_stays_inactive() -> None:
    scientific = _scientific(
        primary_database_name="primary_db",
        metagenomic_database_name="environment_db",
        max_sequences=None,
    )

    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("alpha.a3m", "beta.a3m"),
        scientific=scientific,
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )

    assert "--max-seqs" not in plan.gpuserver_argv
    assert "/db root/primary_db" in plan.gpuserver_argv
    assert all("environment_db" not in argument for argument in plan.gpuserver_argv)
    assert plan.search_argv[-4:] == ("--db1", "primary_db", "--db3", "environment_db")


@pytest.mark.parametrize(
    ("record_order", "members", "message"),
    [
        ("reversed", ("beta.a3m", "alpha.a3m"), "exactly match"),
        ("source", ("alpha.a3m",), "exactly one"),
        ("source", ("same.a3m", "same.a3m"), "unique"),
        ("source", ("../alpha.a3m", "beta.a3m"), "top-level"),
    ],
)
def test_chunk_plan_rejects_incoherent_record_and_member_declarations(
    record_order: str,
    members: tuple[str, ...],
    message: str,
) -> None:
    records = _records()[::-1] if record_order == "reversed" else _records()
    with pytest.raises(ValueError, match=message):
        plan_preprocessing_chunk_execution(
            chunk=_chunk(),
            records=records,
            expected_a3m_members=members,
            scientific=_scientific(),
            site=_site_config(),
            runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
        )


def test_configuration_is_frozen_and_rejects_unknown_fields_and_versions() -> None:
    site = _site_config()
    payload = site.model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        PreprocessingSiteConfig.model_validate(payload)
    with pytest.raises(ValueError, match="Unsupported PreprocessingScientificConfig schema_version 4"):
        PreprocessingScientificConfig.model_validate({"schema_version": 4})
    with pytest.raises(ValidationError, match="frozen"):
        site.max_concurrency = 4  # type: ignore[misc]


def test_direct_execution_records_reject_missing_schema_versions() -> None:
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("alpha.a3m", "beta.a3m"),
        scientific=_scientific(),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )

    for record in (
        plan.runtime,
        plan.expected_a3ms[0],
        plan.evidence,
        plan.package,
        plan,
    ):
        with pytest.raises(ValueError, match="schema_version must be declared explicitly"):
            replace(record, schema_version=None)


def test_planning_does_not_execute_or_materialize_any_paths(tmp_path: Path) -> None:
    site = _site_config().model_copy(
        update={
            "scratch_output_root": str(tmp_path / "scratch"),
            "project_logs_root": str(tmp_path / "logs"),
            "finished_msa_root": str(tmp_path / "finished-msas"),
            "finished_input_root": str(tmp_path / "finished-input"),
        }
    )

    first = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("alpha.a3m", "beta.a3m"),
        scientific=_scientific(),
        site=site,
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    replay = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("alpha.a3m", "beta.a3m"),
        scientific=_scientific(),
        site=site,
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )

    assert replay == first
    assert tuple(tmp_path.iterdir()) == ()


def test_execution_plan_mapping_round_trip_is_versioned_and_fail_closed() -> None:
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("alpha.a3m", "beta.a3m"),
        scientific=_scientific(),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )

    mapping = plan.to_mapping()

    assert preprocessing_chunk_execution_plan_from_mapping(mapping) == plan
    assert mapping["schema_version"] == 1
    mapping["unexpected"] = True
    with pytest.raises(ValueError, match=r"Unknown PreprocessingChunkExecutionPlan field\(s\): unexpected"):
        preprocessing_chunk_execution_plan_from_mapping(mapping)


def test_execution_plan_loader_rejects_nested_versions_unknown_fields_and_tampering() -> None:
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("alpha.a3m", "beta.a3m"),
        scientific=_scientific(),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )

    unsupported_runtime = deepcopy(plan.to_mapping())
    assert isinstance(unsupported_runtime["runtime"], dict)
    unsupported_runtime["runtime"]["schema_version"] = 2
    with pytest.raises(ValueError, match="Unsupported PreprocessingRuntimeCoordinates schema_version 2"):
        preprocessing_chunk_execution_plan_from_mapping(unsupported_runtime)

    unknown_package = deepcopy(plan.to_mapping())
    assert isinstance(unknown_package["package"], dict)
    unknown_package["package"]["undeclared"] = True
    with pytest.raises(ValueError, match=r"Unknown PreprocessingPackagePlan field\(s\): undeclared"):
        preprocessing_chunk_execution_plan_from_mapping(unknown_package)

    tampered_search = deepcopy(plan.to_mapping())
    assert isinstance(tampered_search["search_argv"], list)
    threads_index = tampered_search["search_argv"].index("--threads") + 1
    tampered_search["search_argv"][threads_index] = "32"
    with pytest.raises(ValueError, match="pinned command vectors"):
        preprocessing_chunk_execution_plan_from_mapping(tampered_search)

    missing_gpu_server = deepcopy(plan.to_mapping())
    assert isinstance(missing_gpu_server["search_argv"], list)
    gpu_server_index = missing_gpu_server["search_argv"].index("--gpu-server")
    del missing_gpu_server["search_argv"][gpu_server_index : gpu_server_index + 2]
    with pytest.raises(ValueError, match="pinned command vectors"):
        preprocessing_chunk_execution_plan_from_mapping(missing_gpu_server)

    tampered_evidence = deepcopy(plan.to_mapping())
    assert isinstance(tampered_evidence["evidence"], dict)
    tampered_evidence["evidence"]["durable_log_path"] = "/wrong/nested/path.log"
    with pytest.raises(ValueError, match="flat durable identity"):
        preprocessing_chunk_execution_plan_from_mapping(tampered_evidence)

    missing_raw_directory = deepcopy(plan.to_mapping())
    assert isinstance(missing_raw_directory["evidence"], dict)
    del missing_raw_directory["evidence"]["raw_search_output_directory"]
    with pytest.raises(ValueError, match="raw_search_output_directory must be a non-empty string"):
        preprocessing_chunk_execution_plan_from_mapping(missing_raw_directory)

    aliased_raw_directory = deepcopy(plan.to_mapping())
    assert isinstance(aliased_raw_directory["evidence"], dict)
    aliased_raw_directory["evidence"]["raw_search_output_directory"] = plan.evidence.scratch_output_directory
    assert isinstance(aliased_raw_directory["search_argv"], list)
    aliased_raw_directory["search_argv"][5] = plan.evidence.scratch_output_directory
    with pytest.raises(ValueError, match="exact scratch coordinates"):
        preprocessing_chunk_execution_plan_from_mapping(aliased_raw_directory)

    malformed_header = deepcopy(plan.to_mapping())
    assert isinstance(malformed_header["expected_a3ms"], list)
    malformed_header["expected_a3ms"][0]["source_header"] = ">"
    with pytest.raises(ValueError, match="canonical FASTA identity"):
        preprocessing_chunk_execution_plan_from_mapping(malformed_header)

    duplicate_source = deepcopy(plan.to_mapping())
    assert isinstance(duplicate_source["expected_a3ms"], list)
    duplicate_source["expected_a3ms"][1]["source_ordinal"] = 0
    with pytest.raises(ValueError, match="unique source ordinals"):
        preprocessing_chunk_execution_plan_from_mapping(duplicate_source)

    escaping_chunk = deepcopy(plan.to_mapping())
    assert isinstance(escaping_chunk["expected_a3ms"], list)
    escaping_chunk["expected_a3ms"][0]["chunk_name"] = "../../escape.fa"
    with pytest.raises(ValueError, match="canonical five-digit"):
        preprocessing_chunk_execution_plan_from_mapping(escaping_chunk)


def test_scientific_intent_v1_load_fills_gate_absent_defaults() -> None:
    intent = PreprocessingScientificIntent.model_validate({"schema_version": 1, "max_sequences": 10000})

    assert intent.model_dump(mode="json") == {
        "schema_version": 1,
        "max_sequences": 10000,
        "use_env": False,
        "require_afdb_model_id_stem": False,
    }


def test_scientific_intent_v2_load_fills_gate_on_defaults() -> None:
    intent = PreprocessingScientificIntent.model_validate({"schema_version": 2, "max_sequences": 10000})

    assert intent.model_dump(mode="json") == {
        "schema_version": 2,
        "max_sequences": 10000,
        "use_env": False,
        "require_afdb_model_id_stem": True,
    }


def test_scientific_config_v1_load_fills_gate_absent_defaults() -> None:
    config = PreprocessingScientificConfig.model_validate(
        {
            "schema_version": 1,
            "primary_database_name": "p",
            "metagenomic_database_name": "m",
            "max_sequences": 10000,
        }
    )

    assert config.model_dump(mode="json") == {
        "schema_version": 1,
        "primary_database_name": "p",
        "metagenomic_database_name": "m",
        "max_sequences": 10000,
        "use_env": False,
        "require_afdb_model_id_stem": False,
    }


def test_scientific_config_v2_load_fills_gate_on_defaults() -> None:
    config = PreprocessingScientificConfig.model_validate(
        {
            "schema_version": 2,
            "primary_database_name": "p",
            "metagenomic_database_name": "m",
            "max_sequences": 10000,
        }
    )

    assert config.model_dump(mode="json") == {
        "schema_version": 2,
        "primary_database_name": "p",
        "metagenomic_database_name": "m",
        "max_sequences": 10000,
        "use_env": False,
        "require_afdb_model_id_stem": True,
    }


def test_scientific_fresh_construction_stamps_v3_gate_on_defaults() -> None:
    config = PreprocessingScientificConfig()
    intent = PreprocessingScientificIntent()

    assert config.schema_version == 3
    assert (config.use_env, config.require_afdb_model_id_stem) == (False, True)
    assert intent.schema_version == 3
    assert (intent.use_env, intent.require_afdb_model_id_stem) == (False, True)


def test_scientific_v1_accepts_explicit_false_false_propagation() -> None:
    config = PreprocessingScientificConfig.model_validate(
        {"schema_version": 1, "use_env": False, "require_afdb_model_id_stem": False}
    )
    intent = PreprocessingScientificIntent.model_validate(
        {"schema_version": 1, "use_env": False, "require_afdb_model_id_stem": False}
    )

    assert config.schema_version == 1
    assert (config.use_env, config.require_afdb_model_id_stem) == (False, False)
    assert intent.schema_version == 1
    assert (intent.use_env, intent.require_afdb_model_id_stem) == (False, False)
    assert scientific_canonical_mapping(config) == {
        "schema_version": 1,
        "primary_database_name": "uniref30_2302_db",
        "metagenomic_database_name": "colabfold_envdb_202108_db",
        "max_sequences": 10000,
    }


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (PreprocessingScientificConfig, {"schema_version": 1, "use_env": True}),
        (PreprocessingScientificConfig, {"schema_version": 1, "require_afdb_model_id_stem": True}),
        (PreprocessingScientificIntent, {"schema_version": 1, "use_env": True}),
        (PreprocessingScientificIntent, {"schema_version": 1, "require_afdb_model_id_stem": True}),
    ],
)
def test_scientific_v1_rejects_non_default_v2_knob_values(
    model: type[PreprocessingScientificConfig] | type[PreprocessingScientificIntent],
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="v2 scientific fields in a schema_version 1 record"):
        model.model_validate(payload)


def test_scientific_v1_canonical_mapping_excludes_gate_knobs() -> None:
    intent = PreprocessingScientificIntent.model_validate({"schema_version": 1, "max_sequences": 10000})

    assert intent.use_env is False
    assert intent.require_afdb_model_id_stem is False
    assert scientific_canonical_mapping(intent) == {"schema_version": 1, "max_sequences": 10000}


def test_scientific_canonical_mapping_fails_closed_on_v1_gate_on_bypass() -> None:
    config = PreprocessingScientificConfig(require_afdb_model_id_stem=False)
    bypassed = config.model_copy(update={"schema_version": 1, "require_afdb_model_id_stem": True})

    with pytest.raises(ValueError, match="schema_version 1 scientific record"):
        scientific_canonical_mapping(bypassed)


def test_safe_filename_matches_colabfold_semantics() -> None:
    assert safe_filename("alpha description") == "alpha_description"
    assert safe_filename("AFDB_AF-1234567890123456") == "AFDB_AF-1234567890123456"
    assert safe_filename("A0A9W3HR45_A0A9W3HR47") == "A0A9W3HR45_A0A9W3HR47"


def test_gate_on_rejects_unrelated_conforming_member_at_plan_creation() -> None:
    # Reviewers' attack: a description-bearing header paired with an unrelated but
    # grammar-conforming member must be rejected by derivation equality.
    with pytest.raises(ValueError, match="adapter-derived record-identity name"):
        plan_preprocessing_chunk_execution(
            chunk=_chunk(),
            records=_records(),
            expected_a3m_members=("AFDB_AF-1234567890123456.a3m", "AFDB_AF-2345678901234567.a3m"),
            scientific=PreprocessingScientificConfig(),
            site=_site_config(),
            runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
        )


def test_gate_on_rejects_unrelated_conforming_member_at_materialization() -> None:
    # Build a gate-OFF plan with the same mismatched member, then flip the gate ON
    # via model_copy (bypassing __post_init__) so materialization is the seam
    # under test.
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("AFDB_AF-1234567890123456.a3m", "AFDB_AF-2345678901234567.a3m"),
        scientific=_scientific(),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    intent = preprocessing_chunk_execution_intent_from_plan(plan)
    flipped = copy(intent)
    object.__setattr__(
        flipped,
        "scientific",
        intent.scientific.model_copy(update={"require_afdb_model_id_stem": True}),
    )

    with pytest.raises(ValueError, match="adapter-derived record-identity name"):
        materialize_preprocessing_chunk_execution_plan(
            flipped,
            selected_database_root="/db root",
            primary_database_name="uniref30_2302_db",
            metagenomic_database_name="colabfold_envdb_202108_db",
        )


@pytest.mark.parametrize("model", [PreprocessingScientificConfig, PreprocessingScientificIntent])
@pytest.mark.parametrize("field", ["use_env", "require_afdb_model_id_stem"])
@pytest.mark.parametrize("value", [0, 1, "yes", "no", "true"])
def test_scientific_boolean_fields_reject_non_boolean_values(
    model: type[PreprocessingScientificConfig] | type[PreprocessingScientificIntent],
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate({field: value})


def test_scientific_boolean_fields_accept_real_booleans_and_serialize_unchanged() -> None:
    config = PreprocessingScientificConfig.model_validate({"use_env": True, "require_afdb_model_id_stem": False})
    intent = PreprocessingScientificIntent.model_validate({"use_env": True, "require_afdb_model_id_stem": False})

    assert config.use_env is True
    assert config.require_afdb_model_id_stem is False
    assert intent.use_env is True
    assert intent.require_afdb_model_id_stem is False
    assert config.model_dump(mode="json")["use_env"] is True
    assert config.model_dump(mode="json")["require_afdb_model_id_stem"] is False
    assert intent.model_dump(mode="json")["use_env"] is True
    assert intent.model_dump(mode="json")["require_afdb_model_id_stem"] is False


@pytest.mark.parametrize(
    "member_name",
    [
        "AFDB_AF-0123456789012345.a3m",
        "AFDB_AF_0123456789012345.a3m",
        "AFDB_AF-0123456789012345_AF-0123456789012345.a3m",
        "AFDB_AF_0123456789012345_AF_0123456789012345.a3m",
        "AFDB_AF-0123456789012345_AF_0123456789012345.a3m",
    ],
)
def test_afdb_model_id_member_name_predicate_accepts_conforming_stems(member_name: str) -> None:
    assert afdb_model_id_member_name_conforms(member_name)


@pytest.mark.parametrize(
    "member_name",
    ["AFDB_alpha.a3m", "A0A9W3HR45_A0A9W3HR47.a3m", "alpha result.a3m", "AFDB_zeta.a3m"],
)
def test_afdb_model_id_member_name_predicate_rejects_non_conforming_stems(member_name: str) -> None:
    assert not afdb_model_id_member_name_conforms(member_name)


@pytest.mark.parametrize(
    "member_name",
    ["AFDB_alpha.a3m", "A0A9W3HR45_A0A9W3HR47.a3m", "alpha result.a3m"],
)
def test_chunk_plan_rejects_non_conforming_stem_when_gate_on(member_name: str) -> None:
    with pytest.raises(ValueError, match="discoverable AFDB or PDB assembly model ID"):
        plan_preprocessing_chunk_execution(
            chunk=_chunk(),
            records=_records(),
            expected_a3m_members=(member_name, "AFDB_AF-0123456789012345.a3m"),
            scientific=PreprocessingScientificConfig(),
            site=_site_config(),
            runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
        )


@pytest.mark.parametrize(
    "member_name",
    ["AFDB_alpha.a3m", "A0A9W3HR45_A0A9W3HR47.a3m", "alpha result.a3m"],
)
def test_chunk_plan_accepts_non_conforming_stem_when_gate_off(member_name: str) -> None:
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=(member_name, "AFDB_AF-0123456789012345.a3m"),
        scientific=_scientific(),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )

    assert plan.expected_a3ms[0].member_name == member_name


def test_expected_member_name_derives_from_record_identity() -> None:
    assert expected_member_name("pdb_5snm_assembly_1") == "pdb_5snm_assembly_1.a3m"
    assert expected_member_name("AFDB_AF-1234567890123456") == "AFDB_AF-1234567890123456.a3m"


@pytest.mark.parametrize(
    "member_name",
    [
        "AFDB_AF-0123456789012345.a3m",
        "AFDB_AF_0123456789012345.a3m",
        "pdb_5snm_assembly_1.a3m",
        "pdb_1abc_assembly_3.a3m",
    ],
)
def test_member_name_conforms_accepts_afdb_and_pdb_stems(member_name: str) -> None:
    assert member_name_conforms(member_name)


@pytest.mark.parametrize(
    "member_name",
    ["AFDB_alpha.a3m", "PDB_5SNM_ASSEMBLY_1.a3m", "alpha.a3m", "pdb_5snm_Assembly_1.a3m"],
)
def test_member_name_conforms_rejects_non_conforming_stems(member_name: str) -> None:
    assert not member_name_conforms(member_name)


def test_chunk_plan_accepts_pdb_assembly_record_with_gate_on() -> None:
    """A pdb_*_assembly_* record passes the stem gate and derives the correct member name."""
    records = (
        PreprocessingFastaRecord(
            header=">pdb_5snm_assembly_1 RCSB PDB 5snm biological assembly 1",
            sequence="AAA",
            identity="pdb_5snm_assembly_1",
            source_ordinal=0,
        ),
        PreprocessingFastaRecord(
            header=">AFDB_AF-2345678901234567",
            sequence="TTT",
            identity="AFDB_AF-2345678901234567",
            source_ordinal=1,
        ),
    )
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=records,
        expected_a3m_members=("pdb_5snm_assembly_1.a3m", "AFDB_AF-2345678901234567.a3m"),
        scientific=PreprocessingScientificConfig(),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    assert plan.expected_a3ms[0].member_name == "pdb_5snm_assembly_1.a3m"


def test_search_argv_contains_unpaired_paired() -> None:
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_conforming_records(),
        expected_a3m_members=("AFDB_AF-1234567890123456.a3m", "AFDB_AF-2345678901234567.a3m"),
        scientific=_scientific(),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=0, gpu_id=0, submission_counter=0),
    )
    assert "--pair-mode" in plan.search_argv
    idx = plan.search_argv.index("--pair-mode")
    assert plan.search_argv[idx + 1] == "unpaired_paired"


def _chunk() -> PreprocessingChunk:
    return PreprocessingChunk(
        name="proteins_tranche00_00000.fa",
        tranche_name="tranche00",
        ordinal=0,
        tranche_chunk_ordinal=0,
        record_ordinals=(0, 1),
    )


def _records() -> tuple[PreprocessingFastaRecord, ...]:
    return (
        PreprocessingFastaRecord(header=">alpha description", sequence="AAA", identity="alpha", source_ordinal=0),
        PreprocessingFastaRecord(header=">beta", sequence="TTT", identity="beta", source_ordinal=1),
    )


def _conforming_records() -> tuple[PreprocessingFastaRecord, ...]:
    """Pure-ID headers whose adapter-derived names match the conforming members."""
    return (
        PreprocessingFastaRecord(
            header=">AFDB_AF-1234567890123456",
            sequence="AAA",
            identity="AFDB_AF-1234567890123456",
            source_ordinal=0,
        ),
        PreprocessingFastaRecord(
            header=">AFDB_AF-2345678901234567",
            sequence="TTT",
            identity="AFDB_AF-2345678901234567",
            source_ordinal=1,
        ),
    )


def _site_config() -> PreprocessingSiteConfig:
    return PreprocessingSiteConfig(
        mmseqs_executable="/opt/mm seqs/mmseqs",
        colabfold_search_executable="/opt/colab fold/colabfold_search",
        tar_executable="/usr/bin/tar",
        lz4_executable="/opt/lz 4/lz4",
        database_root="/db root",
        input_root="/scratch input",
        scratch_output_root="/scratch output",
        project_logs_root="/project logs",
        finished_msa_root="/finished msa",
        split_input_root="/split input",
        finished_input_root="/finished input",
        container_image="/images/mm seqs.sqsh",
        container_mounts=("/host db:/db root",),
        max_concurrency=3,
        gpu_delay_seconds=0,
    )


def _scientific(**kwargs: Any) -> PreprocessingScientificConfig:
    return PreprocessingScientificConfig(require_afdb_model_id_stem=False, **kwargs)


@pytest.mark.parametrize(
    ("version", "mode", "expected_digest"),
    [
        (1, "unpaired_paired", "94d3b0e0b1197cb6ecd35104fe536b00e65e2e2d134d7baba3b9bf7f6284768e"),
        (1, "paired", "388469db57419862a23864b50f0ad5982f1f5d89400cf14478d043fc7407ae3e"),
        (2, "unpaired_paired", "bde96c0e3b692b660d7b0140cfa397242a6e228752c07e47090a5de9efb8d70c"),
        (2, "paired", "88ac0d0adc3a7b6d2ee05af15b2af1cd225560ee345750f68c5d77096285928c"),
    ],
)
def test_historical_scientific_versions_preserve_complete_execution_digests(
    version: int, mode: str, expected_digest: str
) -> None:
    from bspp.orchestration.contract.phase import canonical_mapping_digest

    # Digests captured from the unchanged pre-v3 implementation, not regenerated.
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("alpha.a3m", "beta.a3m"),
        scientific=_scientific(schema_version=version, use_env=False),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=7, gpu_id=2, submission_counter=3),
    )
    mapping = plan.to_mapping()
    mapping["search_argv"][mapping["search_argv"].index("--pair-mode") + 1] = mode
    assert canonical_mapping_digest(mapping) == expected_digest
    replayed = preprocessing_chunk_execution_plan_from_mapping(mapping)
    assert replayed.to_mapping() == mapping
    assert canonical_mapping_digest(replayed.to_mapping()) == expected_digest
    assert replayed.search_argv[replayed.search_argv.index("--filter") + 1] == "2"


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("tamper", ["filter", "pair-mode", "pairing_strategy", "duplicate-filter"])
def test_scientific_filter_policy_rejects_mismatched_vectors(version: int, tamper: str) -> None:
    plan = plan_preprocessing_chunk_execution(
        chunk=_chunk(),
        records=_records(),
        expected_a3m_members=("alpha.a3m", "beta.a3m"),
        scientific=_scientific(schema_version=version, use_env=False),
        site=_site_config(),
        runtime=PreprocessingRuntimeCoordinates(slurm_node_id=7, gpu_id=2, submission_counter=3),
    )
    mapping = plan.to_mapping()
    argv = mapping["search_argv"]
    if tamper == "duplicate-filter":
        argv.extend(["--filter", "1"])
    else:
        argv[argv.index(f"--{tamper}") + 1] = {
            "filter": "2" if version == 3 else "1",
            "pair-mode": "paired" if version == 3 else "unpaired",
            "pairing_strategy": "0",
        }[tamper]
    with pytest.raises(ValueError, match="pinned command vectors"):
        preprocessing_chunk_execution_plan_from_mapping(mapping)


def test_scientific_v3_changes_only_version_and_filter_and_preserves_intent_roundtrip() -> None:
    plans = [
        plan_preprocessing_chunk_execution(
            chunk=_chunk(),
            records=_records(),
            expected_a3m_members=("alpha.a3m", "beta.a3m"),
            scientific=_scientific(schema_version=version, use_env=False),
            site=_site_config(),
            runtime=PreprocessingRuntimeCoordinates(slurm_node_id=7, gpu_id=2, submission_counter=3),
        )
        for version in (2, 3)
    ]
    old, new = plans
    expected = old.to_mapping()
    expected["scientific"]["schema_version"] = 3
    expected["search_argv"][expected["search_argv"].index("--filter") + 1] = "1"
    assert new.to_mapping() == expected
    assert preprocessing_chunk_execution_plan_from_mapping(expected) == new
    intent = preprocessing_chunk_execution_intent_from_plan(new)
    rebound = materialize_preprocessing_chunk_execution_plan(
        intent,
        selected_database_root=new.site.database_root,
        primary_database_name=new.scientific.primary_database_name,
        metagenomic_database_name=new.scientific.metagenomic_database_name,
    )
    assert rebound == new


@pytest.mark.parametrize("model", [PreprocessingScientificConfig, PreprocessingScientificIntent])
@pytest.mark.parametrize("version", [True, "3", 0, 4])
def test_scientific_version_space_stays_strict(model: type, version: object) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        model.model_validate({"schema_version": version})
