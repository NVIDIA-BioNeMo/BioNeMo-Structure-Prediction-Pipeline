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

"""Strict canonical cold Database Replica contract seams."""

from __future__ import annotations

import hashlib
import json

import pytest

from bspp.orchestration.contract.database_placement import (
    DATABASE_CACHE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
)
from bspp.orchestration.contract.database_placement_result import database_placement_result_from_mapping
from bspp.orchestration.contract.database_replica import (
    DatabaseCapacityGate,
    DatabaseReplicaColdFailureEvidence,
    DatabaseReplicaColdResult,
    canonical_database_replica_cold_failure_evidence_bytes,
    canonical_database_replica_cold_result_bytes,
    canonical_database_replica_manifest_bytes,
    database_replica_cold_failure_evidence_from_mapping,
    database_replica_cold_result_from_mapping,
    database_replica_manifest_digest,
    database_replica_manifest_from_mapping,
)
from bspp.orchestration.contract.database_set_provisioning import DatabaseSetIdentity
from tests.support.database_cold_replica import _cache_mount, _capacity, _manifest, _source_mount


def test_cold_contract_round_trips_strict_canonical_documents_without_widening_direct_parser() -> None:
    manifest = _manifest()
    result = DatabaseReplicaColdResult(
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

    assert database_replica_manifest_from_mapping(manifest.to_mapping()) == manifest
    assert database_replica_cold_result_from_mapping(result.to_mapping()) == result
    assert (
        canonical_database_replica_manifest_bytes(manifest)
        == (json.dumps(manifest.to_mapping(), indent=2, sort_keys=True) + "\n").encode()
    )
    assert (
        database_replica_manifest_digest(manifest)
        == hashlib.sha256(canonical_database_replica_manifest_bytes(manifest)).hexdigest()
    )
    assert canonical_database_replica_cold_result_bytes(result).endswith(b"\n")

    with pytest.raises(ValueError):
        database_placement_result_from_mapping(result.to_mapping())
    with pytest.raises(ValueError):
        database_replica_cold_result_from_mapping({"database_placement_result": {}})


def test_capacity_gate_treats_exact_equality_as_sufficient_and_rejects_inconsistent_decisions() -> None:
    assert _capacity().decision == "sufficient"
    with pytest.raises(ValueError, match="decision"):
        DatabaseCapacityGate(
            available_user_bytes=127,
            allocated_replica_bytes=64,
            reserved_bytes=64,
            required_bytes=128,
            decision="sufficient",
        )


def test_cold_failure_contract_is_separate_strict_and_bounded() -> None:
    failure = DatabaseReplicaColdFailureEvidence(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        attempt_id="attempt-0001",
        phase_runspec_digest="b" * 64,
        action_id="preprocessing-chunk-000000",
        database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"),
        requested_policy=DatabaseAccessPolicy.STAGE_REQUIRED,
        source_manifest_sha256="a" * 64,
        source_mount=_source_mount(),
        cache_mount=_cache_mount(),
        capacity_gate=DatabaseCapacityGate(
            available_user_bytes=127,
            allocated_replica_bytes=64,
            reserved_bytes=64,
            required_bytes=128,
            decision="insufficient",
        ),
        science_started=False,
        classification="insufficient-capacity",
        error="insufficient user-available capacity",
    )

    assert database_replica_cold_failure_evidence_from_mapping(failure.to_mapping()) == failure
    assert canonical_database_replica_cold_failure_evidence_bytes(failure).endswith(b"\n")
    with pytest.raises(ValueError):
        database_replica_cold_result_from_mapping(failure.to_mapping())


@pytest.mark.parametrize("mutation", ["argv", "understated-size"])
def test_replica_manifest_loader_rejects_forged_copy_authority(mutation: str) -> None:
    mapping = _manifest().to_mapping()
    inner = mapping["database_replica_manifest"]
    assert isinstance(inner, dict)
    copy = inner["copy_evidence"]
    assert isinstance(copy, dict)
    outcomes = copy["outcomes"]
    assert isinstance(outcomes, list)
    first = outcomes[0]
    assert isinstance(first, dict)
    if mutation == "argv":
        argv = first["argv"]
        assert isinstance(argv, list)
        argv[1] = "--delete"
    else:
        first["size_bytes"] = 31
        copy["total_copied_bytes"] = 63
        copy["aggregate_bytes_per_second"] = 3_937_500_000

    with pytest.raises(ValueError):
        database_replica_manifest_from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "contradictory_value"),
    [("size_bytes", 31), ("mtime_ns", 124)],
)
def test_replica_manifest_loader_rejects_contradictory_alias_group_metadata(
    field: str,
    contradictory_value: int,
) -> None:
    mapping = _manifest().to_mapping()
    inner = mapping["database_replica_manifest"]
    assert isinstance(inner, dict)
    members = inner["members"]
    assert isinstance(members, list)
    primary = next(item for item in members if isinstance(item, dict) and item.get("logical_name") == "primary")
    assert isinstance(primary, dict)
    primary["resolved_source_path"] = "metagenomic"
    primary["hardlink_group"] = "metagenomic"
    primary[field] = contradictory_value

    for observation_name in ("pre_copy_source_observation", "post_copy_source_observation"):
        observation = inner[observation_name]
        assert isinstance(observation, dict)
        observed_members = observation["members"]
        assert isinstance(observed_members, list)
        observed_primary = next(
            item for item in observed_members if isinstance(item, dict) and item.get("logical_name") == "primary"
        )
        assert isinstance(observed_primary, dict)
        observed_primary["resolved_path"] = "metagenomic"
        observed_primary[field] = contradictory_value

    copy = inner["copy_evidence"]
    assert isinstance(copy, dict)
    outcomes = copy["outcomes"]
    assert isinstance(outcomes, list)
    first = outcomes[0]
    assert isinstance(first, dict)
    first["logical_names"] = ["metagenomic", "primary"]
    copy["source_count"] = 1
    copy["selected_workers"] = 1
    copy["total_copied_bytes"] = 32
    copy["aggregate_bytes_per_second"] = 2_000_000_000
    copy["outcomes"] = [first]

    with pytest.raises(ValueError, match="metadata identity"):
        database_replica_manifest_from_mapping(mapping)
