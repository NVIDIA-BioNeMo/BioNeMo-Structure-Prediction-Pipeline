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

"""Characterization matrix for shared database contract validation policy."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest

from bspp.orchestration.contract._database_validation import (
    required_int,
    required_list,
    required_nonempty_str,
    required_sequence,
    required_str,
    validate_placement_absolute_path,
    validate_provisioning_absolute_path,
    validate_replica_absolute_path,
)
from bspp.orchestration.contract.database_placement import (
    DATABASE_CACHE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
    DatabaseSetSelection,
    build_preprocessing_database_binding,
    database_mount_descriptor_from_mapping,
    database_placement_branch_from_mapping,
    database_profile_staging_snapshot_from_mapping,
    database_set_selection_from_mapping,
    preprocessing_database_binding_from_mapping,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabasePlacementResult,
    database_placement_failure_evidence_from_mapping,
    database_placement_result_from_mapping,
)
from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaColdResult,
    DatabaseReplicaWarmResult,
    DatabaseWarmCacheMountFacts,
    database_replica_cold_result_from_mapping,
    database_replica_manifest_digest,
    database_replica_manifest_from_mapping,
    database_replica_warm_result_from_mapping,
    database_warm_cache_mount_facts_from_mapping,
)
from bspp.orchestration.contract.database_replica_facts import (
    _cache_mount_from_mapping,
    _capacity_from_mapping,
    _copy_evidence_from_mapping,
    _rsync_outcome_from_mapping,
    _source_mount_from_mapping,
    _source_observation_from_mapping,
)
from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseSetProvisioningEvidence,
    DatabaseSourceManifest,
    database_set_declaration_from_mapping,
    database_set_provisioning_evidence_from_mapping,
    database_source_manifest_from_mapping,
)
from tests.support.database_cold_replica import _cache_mount, _capacity, _manifest, _source_mount


@pytest.mark.parametrize(
    ("validator", "value", "error"),
    [
        (validate_placement_absolute_path, "/cache\x00root", None),
        (
            validate_replica_absolute_path,
            "/cache\x00root",
            "path must be a normalized absolute path",
        ),
        (
            validate_provisioning_absolute_path,
            "/cache\x00root",
            "path must be a normalized absolute path",
        ),
        (
            validate_placement_absolute_path,
            "/cache/../root",
            "path must be a normalized absolute path",
        ),
        (
            validate_replica_absolute_path,
            "cache/root",
            "path must be a normalized absolute path",
        ),
        (
            validate_provisioning_absolute_path,
            "/cache//root",
            "path must be a normalized absolute path",
        ),
    ],
)
def test_database_absolute_path_policy_matrix(
    validator: Callable[[str, str], None],
    value: str,
    error: str | None,
) -> None:
    if error is None:
        validator(value, "path")
        return

    with pytest.raises(ValueError, match=f"^{error}$"):
        validator(value, "path")


@pytest.mark.parametrize(
    ("extractor", "value", "expected", "error"),
    [
        (required_str, "", "", None),
        (required_nonempty_str, "", None, "field must be a non-empty string"),
        (required_list, [], [], None),
        (required_list, (), None, "field must be a list"),
        (required_sequence, [], [], None),
        (required_sequence, (), (), None),
        (required_int, 1, 1, None),
        (required_int, True, None, "field must be an integer"),
    ],
)
def test_database_scalar_and_collection_policy_matrix(
    extractor: Callable[[dict[str, object], str], object],
    value: object,
    expected: object,
    error: str | None,
) -> None:
    if error is None:
        assert extractor({"field": value}, "field") == expected
        return

    with pytest.raises(ValueError, match=f"^{error}$"):
        extractor({"field": value}, "field")


def _staging_mapping() -> dict[str, object]:
    return {
        "cache_root": "/run/bspp/database/cache",
        "unix_user": "runner",
        "expected_filesystem_type": "tmpfs",
        "reserve_bytes": 0,
        "lock_wait_seconds": 1,
    }


def _declaration_mapping() -> dict[str, object]:
    return {
        "database_set_declaration": {
            "database_set": {"identifier": "bspp-search", "version": "2026-08"},
            "source_root": "/database/source",
            "roles": [
                {
                    "role": "primary",
                    "database_name": "primary",
                    "members": [
                        {
                            "logical_name": "primary",
                            "source_path": "primary",
                            "preexisting_checksum": None,
                        }
                    ],
                },
                {
                    "role": "metagenomic",
                    "database_name": "metagenomic",
                    "members": [
                        {
                            "logical_name": "metagenomic",
                            "source_path": "metagenomic",
                            "preexisting_checksum": None,
                        }
                    ],
                },
            ],
        }
    }


def _cold_result() -> DatabaseReplicaColdResult:
    manifest = _manifest()
    return DatabaseReplicaColdResult(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_runspec_digest="b" * 64,
        action_id="preprocessing-chunk-000000",
        database_set=manifest.database_set,
        requested_policy=DatabaseAccessPolicy.STAGE_REQUIRED,
        source_manifest_sha256=manifest.source_manifest_sha256,
        branch_kind="staged",
        outcome=DatabasePlacementOutcomeKind.REPLICA_COLD,
        selected_container_root=SELECTED_DATABASE_ROOT,
        replica_container_root=f"{DATABASE_CACHE_ROOT}/replicas/{'a' * 64}",
        verification="metadata-verified",
        source_mount=_source_mount(),
        cache_mount=_cache_mount(),
        capacity_gate=_capacity(),
        replica_manifest_sha256=database_replica_manifest_digest(manifest),
        copy_evidence=manifest.copy_evidence,
    )


def _warm_result() -> DatabaseReplicaWarmResult:
    cold_mount = _cache_mount()
    manifest = _manifest()
    return DatabaseReplicaWarmResult(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_runspec_digest="b" * 64,
        action_id="preprocessing-chunk-000000",
        database_set=manifest.database_set,
        requested_policy=DatabaseAccessPolicy.STAGE_REQUIRED,
        source_manifest_sha256=manifest.source_manifest_sha256,
        branch_kind="staged",
        outcome=DatabasePlacementOutcomeKind.REPLICA_WARM,
        selected_container_root=SELECTED_DATABASE_ROOT,
        replica_container_root=f"{DATABASE_CACHE_ROOT}/replicas/{'a' * 64}",
        verification="metadata-verified",
        cache_mount=DatabaseWarmCacheMountFacts(
            mount_id=cold_mount.mount_id,
            parent_mount_id=cold_mount.parent_mount_id,
            device_major=cold_mount.device_major,
            device_minor=cold_mount.device_minor,
            mount_root=cold_mount.mount_root,
            mount_point=cold_mount.mount_point,
            filesystem_type=cold_mount.filesystem_type,
            mount_source=cold_mount.mount_source,
            mount_options=cold_mount.mount_options,
            super_options=cold_mount.super_options,
            read_write=True,
        ),
        replica_manifest_sha256=database_replica_manifest_digest(manifest),
    )


def _direct_result() -> DatabasePlacementResult:
    manifest = _manifest()
    return DatabasePlacementResult(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_runspec_digest="b" * 64,
        action_id="preprocessing-chunk-000000",
        database_set=manifest.database_set,
        requested_policy=DatabaseAccessPolicy.DIRECT,
        source_manifest_sha256=manifest.source_manifest_sha256,
        branch_kind="direct-requested",
        outcome=DatabasePlacementOutcomeKind.DIRECT_REQUESTED,
        selected_container_root=SELECTED_DATABASE_ROOT,
        verification="metadata-verified",
        source_mount=_source_mount(),
        pre_science_observation=manifest.pre_copy_source_observation,
    )


def _source_manifest() -> DatabaseSourceManifest:
    manifest = _manifest()
    return DatabaseSourceManifest(
        database_set=manifest.database_set,
        source_root="/database/source",
        members=manifest.pre_copy_source_observation.members,
    )


def _binding_mapping() -> dict[str, object]:
    source_manifest = _source_manifest()
    return build_preprocessing_database_binding(
        selection=DatabaseSetSelection(
            database_set=source_manifest.database_set,
            requested_policy=DatabaseAccessPolicy.DIRECT,
        ),
        source_manifest=source_manifest,
        source_manifest_projection="attempts/attempt-0001/database-source-manifest.json",
        staging=None,
        gpuserver_argv=("gpuserver",),
        search_argv=("search",),
    ).to_mapping()


def _selection_mapping() -> dict[str, object]:
    source_manifest = _source_manifest()
    return DatabaseSetSelection(
        database_set=source_manifest.database_set,
        requested_policy=DatabaseAccessPolicy.DIRECT,
    ).to_mapping()


def _branch_mapping() -> dict[str, object]:
    branches = _binding_mapping()["branches"]
    assert isinstance(branches, list)
    branch = branches[0]
    assert isinstance(branch, dict)
    return branch


def _mount_mapping() -> dict[str, object]:
    mounts = _branch_mapping()["placement_mounts"]
    assert isinstance(mounts, list)
    mount = mounts[0]
    assert isinstance(mount, dict)
    return mount


def _source_manifest_mapping() -> dict[str, object]:
    return _source_manifest().to_mapping()


def _provisioning_mapping() -> dict[str, object]:
    return DatabaseSetProvisioningEvidence(
        database_set=_source_manifest().database_set,
        disposition="published",
        manifest_path="/manifests/database-source-manifest.json",
        manifest_sha256="c" * 64,
        member_count=2,
    ).to_mapping()


_OPTIONAL_SCHEMA_READER_CASES = [
    (
        "selection",
        database_set_selection_from_mapping,
        _selection_mapping,
        None,
        "database_set",
        "Database Set selection",
    ),
    (
        "staging",
        database_profile_staging_snapshot_from_mapping,
        lambda: {"schema_version": 1, **_staging_mapping()},
        None,
        "cache_root",
        "database profile staging snapshot",
    ),
    (
        "mount",
        database_mount_descriptor_from_mapping,
        _mount_mapping,
        None,
        "source",
        "database mount descriptor",
    ),
    (
        "branch",
        database_placement_branch_from_mapping,
        _branch_mapping,
        None,
        "branch_kind",
        "database placement branch",
    ),
    (
        "binding",
        preprocessing_database_binding_from_mapping,
        _binding_mapping,
        None,
        "database_set",
        "preprocessing database binding",
    ),
    (
        "declaration",
        database_set_declaration_from_mapping,
        _declaration_mapping,
        "database_set_declaration",
        "database_set",
        "database_set_declaration",
    ),
    (
        "source-manifest",
        database_source_manifest_from_mapping,
        _source_manifest_mapping,
        "database_source_manifest",
        "database_set",
        "database_source_manifest",
    ),
    (
        "provisioning-evidence",
        database_set_provisioning_evidence_from_mapping,
        _provisioning_mapping,
        "database_set_provisioning",
        "database_set",
        "database_set_provisioning",
    ),
]


def _inner_mapping(mapping: dict[str, object], wrapper: str | None) -> dict[str, object]:
    if wrapper is None:
        return mapping
    inner = mapping[wrapper]
    assert isinstance(inner, dict)
    return inner


@pytest.mark.parametrize(
    ("loader", "mapping_factory", "wrapper"),
    [(case[1], case[2], case[3]) for case in _OPTIONAL_SCHEMA_READER_CASES],
    ids=[case[0] for case in _OPTIONAL_SCHEMA_READER_CASES],
)
def test_public_placement_and_provisioning_readers_preserve_optional_schema_policy(
    loader: Callable[[dict[str, object]], object],
    mapping_factory: Callable[[], dict[str, object]],
    wrapper: str | None,
) -> None:
    explicit = mapping_factory()
    assert loader(explicit).schema_version == 1  # type: ignore[attr-defined]

    omitted = mapping_factory()
    _inner_mapping(omitted, wrapper).pop("schema_version", None)
    assert loader(omitted).schema_version == 1  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("loader", "mapping_factory", "wrapper", "required_field", "record_name"),
    [(case[1], case[2], case[3], case[4], case[5]) for case in _OPTIONAL_SCHEMA_READER_CASES],
    ids=[case[0] for case in _OPTIONAL_SCHEMA_READER_CASES],
)
def test_public_placement_and_provisioning_readers_reject_missing_required_fields_exactly(
    loader: Callable[[dict[str, object]], object],
    mapping_factory: Callable[[], dict[str, object]],
    wrapper: str | None,
    required_field: str,
    record_name: str,
) -> None:
    mapping = mapping_factory()
    inner = _inner_mapping(mapping, wrapper)
    inner.pop("schema_version", None)
    del inner[required_field]

    with pytest.raises(ValueError) as exc_info:
        loader(mapping)

    assert str(exc_info.value) == (f"{record_name} fields are invalid; missing=['{required_field}']; unknown=[]")


@pytest.mark.parametrize(
    ("loader", "mapping_factory", "wrapper", "record_name"),
    [(case[1], case[2], case[3], case[5]) for case in _OPTIONAL_SCHEMA_READER_CASES],
    ids=[case[0] for case in _OPTIONAL_SCHEMA_READER_CASES],
)
def test_public_placement_and_provisioning_readers_reject_unknown_fields_exactly(
    loader: Callable[[dict[str, object]], object],
    mapping_factory: Callable[[], dict[str, object]],
    wrapper: str | None,
    record_name: str,
) -> None:
    mapping = mapping_factory()
    inner = _inner_mapping(mapping, wrapper)
    inner.pop("schema_version", None)
    inner["unexpected"] = True

    with pytest.raises(ValueError) as exc_info:
        loader(mapping)

    assert str(exc_info.value) == (f"{record_name} fields are invalid; missing=[]; unknown=['unexpected']")


def test_database_set_selection_preserves_omitted_requested_policy_default() -> None:
    mapping = _selection_mapping()
    del mapping["requested_policy"]
    selection = database_set_selection_from_mapping(mapping)

    assert selection.requested_policy is DatabaseAccessPolicy.STAGE_REQUIRED


def test_public_path_readers_preserve_nul_policy_asymmetry() -> None:
    staging_mapping = _staging_mapping()
    staging_mapping["cache_root"] = "/cache\x00root"
    assert database_profile_staging_snapshot_from_mapping(staging_mapping).cache_root == "/cache\x00root"

    direct_mapping = _direct_result().to_mapping()
    direct_inner = direct_mapping["database_placement_result"]
    assert isinstance(direct_inner, dict)
    direct_mount = direct_inner["source_mount"]
    assert isinstance(direct_mount, dict)
    direct_mount["mount_root"] = "/source\x00root"
    parsed_direct = database_placement_result_from_mapping(direct_mapping)
    assert parsed_direct.source_mount.mount_root == "/source\x00root"

    declaration_mapping = _declaration_mapping()
    inner = declaration_mapping["database_set_declaration"]
    assert isinstance(inner, dict)
    inner["source_root"] = "/source\x00root"
    with pytest.raises(ValueError, match=r"^source_root must be a normalized absolute path$"):
        database_set_declaration_from_mapping(declaration_mapping)

    cold_mapping = _cold_result().to_mapping()
    cold_inner = cold_mapping["database_replica_cold_result"]
    assert isinstance(cold_inner, dict)
    cache_mount = cold_inner["cache_mount"]
    assert isinstance(cache_mount, dict)
    cache_mount["mount_root"] = "/cache\x00root"
    with pytest.raises(
        ValueError,
        match=r"^database cache mount_root must be a normalized absolute path$",
    ):
        database_replica_cold_result_from_mapping(cold_mapping)

    warm_mapping = _warm_result().to_mapping()
    warm_inner = warm_mapping["database_replica_warm_result"]
    assert isinstance(warm_inner, dict)
    warm_cache_mount = warm_inner["cache_mount"]
    assert isinstance(warm_cache_mount, dict)
    warm_cache_mount["mount_root"] = "/cache\x00root"
    with pytest.raises(
        ValueError,
        match=r"^warm database cache mount_root must be a normalized absolute path$",
    ):
        database_replica_warm_result_from_mapping(warm_mapping)


@pytest.mark.parametrize(
    ("loader", "mapping_factory", "wrapper", "record_name"),
    [
        (
            database_placement_result_from_mapping,
            lambda: _direct_result().to_mapping(),
            "database_placement_result",
            "Database Placement Result",
        ),
        (
            database_placement_failure_evidence_from_mapping,
            lambda: {
                "database_placement_failure": {
                    "schema_version": 1,
                    "phase_run_id": "phase-run-0123456789abcdef0123456789abcdef",
                    "attempt_id": "attempt-0001",
                    "phase_runspec_digest": "b" * 64,
                    "action_id": "preprocessing-chunk-000000",
                    "database_set": {},
                    "requested_policy": "direct",
                    "source_manifest_sha256": "a" * 64,
                    "source_mount": None,
                    "science_started": False,
                    "classification": "authority-invalid",
                    "error": "failure",
                }
            },
            "database_placement_failure",
            "Database Placement failure",
        ),
        (
            database_replica_cold_result_from_mapping,
            lambda: _cold_result().to_mapping(),
            "database_replica_cold_result",
            "Database Replica cold Result",
        ),
        (
            database_replica_warm_result_from_mapping,
            lambda: _warm_result().to_mapping(),
            "database_replica_warm_result",
            "Database Replica warm Result",
        ),
        (
            database_replica_manifest_from_mapping,
            lambda: _manifest().to_mapping(),
            "database_replica_manifest",
            "Database Replica Manifest",
        ),
    ],
)
def test_result_documents_reject_omitted_schema_before_nested_parsing(
    loader: Callable[[dict[str, object]], object],
    mapping_factory: Callable[[], dict[str, object]],
    wrapper: str,
    record_name: str,
) -> None:
    mapping = mapping_factory()
    inner = mapping[wrapper]
    assert isinstance(inner, dict)
    del inner["schema_version"]

    with pytest.raises(ValueError) as exc_info:
        loader(mapping)

    assert str(exc_info.value) == (f"{record_name} fields are invalid; missing=['schema_version']; unknown=[]")


@pytest.mark.parametrize(
    ("loader", "mapping_factory", "wrapper", "record_name"),
    [
        (
            database_placement_result_from_mapping,
            lambda: _direct_result().to_mapping(),
            "database_placement_result",
            "Database Placement Result",
        ),
        (
            database_replica_cold_result_from_mapping,
            lambda: _cold_result().to_mapping(),
            "database_replica_cold_result",
            "Database Replica cold Result",
        ),
        (
            database_replica_warm_result_from_mapping,
            lambda: _warm_result().to_mapping(),
            "database_replica_warm_result",
            "Database Replica warm Result",
        ),
        (
            database_replica_manifest_from_mapping,
            lambda: _manifest().to_mapping(),
            "database_replica_manifest",
            "Database Replica Manifest",
        ),
    ],
)
def test_result_documents_reject_unknown_fields_exactly(
    loader: Callable[[dict[str, object]], object],
    mapping_factory: Callable[[], dict[str, object]],
    wrapper: str,
    record_name: str,
) -> None:
    mapping = mapping_factory()
    inner = mapping[wrapper]
    assert isinstance(inner, dict)
    inner["unexpected"] = True

    with pytest.raises(ValueError) as exc_info:
        loader(mapping)

    assert str(exc_info.value) == (f"{record_name} fields are invalid; missing=[]; unknown=['unexpected']")


@pytest.mark.parametrize(
    ("nested_path", "record_name"),
    [
        (("cache_mount",), "database cache mount facts"),
        (("capacity_gate",), "Database Capacity Gate"),
        (("copy_evidence",), "Database Replica copy evidence"),
        (("copy_evidence", "outcomes", 0), "Database rsync outcome"),
        (("source_mount",), "database source mount facts"),
    ],
)
def test_cold_result_nested_facts_require_explicit_schema(
    nested_path: tuple[str | int, ...],
    record_name: str,
) -> None:
    mapping = deepcopy(_cold_result().to_mapping())
    node: object = mapping["database_replica_cold_result"]
    for component in nested_path:
        assert isinstance(node, dict | list)
        node = node[component]  # type: ignore[index]
    assert isinstance(node, dict)
    del node["schema_version"]

    with pytest.raises(ValueError) as exc_info:
        database_replica_cold_result_from_mapping(mapping)

    assert str(exc_info.value) == (f"{record_name} fields are invalid; missing=['schema_version']; unknown=[]")


def test_warm_result_nested_cache_facts_require_explicit_schema() -> None:
    mapping = deepcopy(_warm_result().to_mapping())
    inner = mapping["database_replica_warm_result"]
    assert isinstance(inner, dict)
    cache_mount = inner["cache_mount"]
    assert isinstance(cache_mount, dict)
    del cache_mount["schema_version"]

    with pytest.raises(ValueError) as exc_info:
        database_replica_warm_result_from_mapping(mapping)

    assert str(exc_info.value) == (
        "warm database cache mount facts fields are invalid; missing=['schema_version']; unknown=[]"
    )


@pytest.mark.parametrize(
    "observation_name",
    ["pre_copy_source_observation", "post_copy_source_observation"],
)
def test_replica_manifest_source_observations_require_explicit_schema(observation_name: str) -> None:
    mapping = deepcopy(_manifest().to_mapping())
    inner = mapping["database_replica_manifest"]
    assert isinstance(inner, dict)
    observation = inner[observation_name]
    assert isinstance(observation, dict)
    del observation["schema_version"]

    with pytest.raises(ValueError) as exc_info:
        database_replica_manifest_from_mapping(mapping)

    assert str(exc_info.value) == (
        "Database Source observation fields are invalid; missing=['schema_version']; unknown=[]"
    )


@pytest.mark.parametrize(
    ("loader", "mapping_factory", "record_name"),
    [
        (_cache_mount_from_mapping, lambda: _cache_mount().to_mapping(), "database cache mount facts"),
        (
            database_warm_cache_mount_facts_from_mapping,
            lambda: _warm_result().cache_mount.to_mapping(),
            "warm database cache mount facts",
        ),
        (_capacity_from_mapping, lambda: _capacity().to_mapping(), "Database Capacity Gate"),
        (
            _copy_evidence_from_mapping,
            lambda: _manifest().copy_evidence.to_mapping(),
            "Database Replica copy evidence",
        ),
        (
            _rsync_outcome_from_mapping,
            lambda: _manifest().copy_evidence.outcomes[0].to_mapping(),
            "Database rsync outcome",
        ),
        (_source_mount_from_mapping, lambda: _source_mount().to_mapping(), "database source mount facts"),
        (
            _source_observation_from_mapping,
            lambda: _manifest().pre_copy_source_observation.to_mapping(),
            "Database Source observation",
        ),
    ],
)
def test_direct_nested_readers_require_explicit_schema_with_exact_errors(
    loader: Callable[[dict[str, object]], object],
    mapping_factory: Callable[[], dict[str, object]],
    record_name: str,
) -> None:
    mapping = mapping_factory()
    del mapping["schema_version"]

    with pytest.raises(ValueError) as exc_info:
        loader(mapping)

    assert str(exc_info.value) == (f"{record_name} fields are invalid; missing=['schema_version']; unknown=[]")
