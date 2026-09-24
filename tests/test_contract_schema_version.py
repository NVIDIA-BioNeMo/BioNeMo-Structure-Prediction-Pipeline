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

"""Contract schema-version behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bspp.orchestration.contract.data_placement import (
    DataPlacementRecord,
    data_placement_record_from_mapping,
    validate_data_placement_tool_for_stage,
)
from bspp.orchestration.contract.database_set_provisioning import database_set_declaration_from_mapping
from bspp.orchestration.contract.provisioning import (
    ProvisioningPlan,
    RuntimeImageCacheCheckRecord,
    RuntimeImagePlan,
    RuntimeImageProvisionRecord,
    SourceBundleBuildRecord,
    SourceBundleManifestEntry,
    SourceBundlePlan,
    SourceBundleStageRecord,
    SourcePackageIdentity,
    provisioning_plan_from_mapping,
    runtime_image_plan_from_mapping,
    source_bundle_manifest_digest,
    source_bundle_plan_from_mapping,
    source_package_identity_from_source_bundle,
)
from bspp.orchestration.contract.runplan import load_runplan
from bspp.orchestration.contract.runspec import load_runspec
from bspp.orchestration.contract.runtime_qualification import runtime_qualification_payload_from_mapping
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from tests.test_runplan import _runplan_data
from tests.test_runspec import _base_runspec


def test_runspec_defaults_to_current_schema_version(tmp_path: Path) -> None:
    spec = load_runspec(_write_yaml(tmp_path / "runspec.yaml", _base_runspec()))

    assert spec.schema_version == 1


def test_runplan_defaults_to_current_schema_version(tmp_path: Path) -> None:
    plan = load_runplan(_write_yaml(tmp_path / "run-plan.yaml", _runplan_data()))

    assert plan.schema_version == 1


def test_runspec_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    data = _base_runspec()
    data["schema_version"] = 2

    with pytest.raises(ValueError, match="Unsupported RunSpec schema_version 2; supported versions: 1"):
        load_runspec(_write_yaml(tmp_path / "runspec.yaml", data))


def test_runplan_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    data = _runplan_data()
    data["schema_version"] = 2

    with pytest.raises(ValueError, match="Unsupported RunPlan schema_version 2; supported versions: 1"):
        load_runplan(_write_yaml(tmp_path / "run-plan.yaml", data))


def test_provisioning_contract_records_default_to_current_schema_version() -> None:
    source_bundle = SourceBundlePlan(
        source_repo="/repo",
        source_state="committed",
        commit="a" * 40,
        tree="d" * 40,
        dirty_evidence_hash=None,
        bundle_id="bspp-orchestration-abc123",
        target_root="/bundles",
        target_path="/bundles/bspp-orchestration-abc123.tar.zst",
    )
    runtime_image = RuntimeImagePlan(
        reference="/images/bspp.sqsh",
        cache_root="/cache",
        cache_path="/cache/bspp.sqsh",
        cache_status="present",
    )
    cache_check = RuntimeImageCacheCheckRecord(
        runtime_image=runtime_image,
        status="present",
        checked=True,
        probe_command=("test", "-f", "/cache/bspp.sqsh"),
        probe_returncode=0,
        evidence_path=Path("/evidence/cache.yaml"),
    )
    provision_record = RuntimeImageProvisionRecord(
        runtime_image=runtime_image,
        cache_check=cache_check,
        status="present",
        evidence_path=Path("/evidence/provision.yaml"),
    )
    plan = ProvisioningPlan(
        cluster="example-cluster",
        run_kind="dev",
        source_bundle=source_bundle,
        runtime_image=runtime_image,
        payload_bytes_moved=False,
    )
    entries = (SourceBundleManifestEntry(path="README.md", size_bytes=6, sha256="a" * 64, mode="0o100644"),)
    manifest_sha256 = source_bundle_manifest_digest(entries)
    build_record = SourceBundleBuildRecord(
        source_bundle=source_bundle,
        archive_path=Path("/local/bundle.tar.zst"),
        archive_size_bytes=123,
        archive_sha256="b" * 64,
        manifest_entries=entries,
        manifest_sha256=manifest_sha256,
        dirty_source_policy="dev-clean",
        evidence_path=Path("/local/build.yaml"),
    )
    stage_record = SourceBundleStageRecord(
        source_bundle=source_bundle,
        archive_path=Path("/local/bundle.tar.zst"),
        target_path="/bundles/bspp-orchestration-abc123.tar.zst",
        archive_sha256="b" * 64,
        verified_sha256="b" * 64,
        commands=(("mkdir", "-p", "/bundles"),),
        evidence_path=Path("/local/stage.yaml"),
    )

    mapping = plan.to_mapping()["provisioning_plan"]

    assert mapping["schema_version"] == 1
    assert source_bundle.to_mapping()["schema_version"] == 1
    assert runtime_image.to_mapping()["schema_version"] == 1
    assert cache_check.to_mapping()["runtime_image_cache_check"]["schema_version"] == 1
    assert provision_record.to_mapping()["runtime_image_provisioning"]["schema_version"] == 1
    assert build_record.to_mapping()["source_bundle_build"]["schema_version"] == 1
    assert stage_record.to_mapping()["source_bundle_stage"]["schema_version"] == 1
    identity = source_package_identity_from_source_bundle(build_record)
    assert isinstance(identity, SourcePackageIdentity)
    assert identity.commit == source_bundle.commit
    assert identity.tree == source_bundle.tree
    assert identity.package_sha256 == build_record.archive_sha256


def test_source_bundle_contract_rejects_unsupported_schema_version() -> None:
    with pytest.raises(ValueError, match="Unsupported SourceBundle schema_version 2; supported versions: 1"):
        source_bundle_plan_from_mapping(
            {
                "schema_version": 2,
                "source_repo": "/repo",
                "source_state": "committed",
                "commit": "abc123",
                "tree": "def456",
                "dirty_evidence_hash": None,
                "bundle_id": "bspp-orchestration-abc123",
                "target_root": "/bundles",
                "target_path": "/bundles/bspp-orchestration-abc123.tar.zst",
            }
        )


def test_runtime_image_contract_rejects_unsupported_schema_version() -> None:
    with pytest.raises(ValueError, match="Unsupported RuntimeImage schema_version 2; supported versions: 1"):
        runtime_image_plan_from_mapping(
            {
                "schema_version": 2,
                "reference": "/images/bspp.sqsh",
                "cache_root": "/cache",
                "cache_path": "/cache/bspp.sqsh",
                "cache_status": "present",
            }
        )


def test_provisioning_plan_contract_rejects_unsupported_schema_version() -> None:
    with pytest.raises(ValueError, match="Unsupported ProvisioningPlan schema_version 2; supported versions: 1"):
        provisioning_plan_from_mapping(
            {
                "provisioning_plan": {
                    "schema_version": 2,
                    "cluster": "example-cluster",
                    "run_kind": "dev",
                    "source_bundle": {
                        "source_repo": "/repo",
                        "source_state": "committed",
                        "commit": "abc123",
                        "tree": "def456",
                        "dirty_evidence_hash": None,
                        "bundle_id": "bspp-orchestration-abc123",
                        "target_root": "/bundles",
                        "target_path": "/bundles/bspp-orchestration-abc123.tar.zst",
                    },
                    "runtime_image": {
                        "reference": "/images/bspp.sqsh",
                        "cache_root": "/cache",
                        "cache_path": "/cache/bspp.sqsh",
                        "cache_status": "present",
                    },
                    "payload_bytes_moved": False,
                }
            }
        )


def test_runtime_qualification_payload_defaults_to_current_schema_version() -> None:
    payload = runtime_qualification_payload_from_mapping({"status": "qualified"})

    assert payload["schema_version"] == 1


def test_data_placement_contract_defaults_and_validates_stage_tool() -> None:
    record = DataPlacementRecord(
        stage="gcs",
        tool="dm",
        dataset="ds1",
        source="s3://example-bucket/postprocessed/ds1/",
        destination="gs://example-gcs-bucket/postprocessed/ds1/",
        payload_bytes_moved=False,
        evidence_path="/evidence/data-placement.json",
    )

    mapping = record.to_mapping()
    parsed = data_placement_record_from_mapping(mapping)

    assert mapping["schema_version"] == 1
    assert parsed == record
    with pytest.raises(ValueError, match="Tool 's5cmd' is not supported for stage 'gcs'"):
        validate_data_placement_tool_for_stage("s5cmd", "gcs")


def test_data_placement_contract_rejects_unsupported_schema_version() -> None:
    with pytest.raises(ValueError, match="Unsupported DataPlacement schema_version 2; supported versions: 1"):
        data_placement_record_from_mapping(
            {
                "schema_version": 2,
                "stage": "gcs",
                "tool": "dm",
                "dataset": "ds1",
                "source": "s3://example-bucket/postprocessed/ds1/",
                "destination": "gs://example-gcs-bucket/postprocessed/ds1/",
                "payload_bytes_moved": False,
                "evidence_path": None,
            }
        )


def test_database_set_declaration_defaults_and_rejects_unsupported_schema_version() -> None:
    payload = {
        "database_set_declaration": {
            "database_set": {"identifier": "afdb-search", "version": "2023-02"},
            "source_root": "/database/source",
            "roles": [
                {
                    "role": "primary",
                    "database_name": "primary_db",
                    "members": [
                        {
                            "logical_name": "primary_db",
                            "source_path": "primary_db",
                            "preexisting_checksum": None,
                        }
                    ],
                },
                {
                    "role": "metagenomic",
                    "database_name": "metagenomic_db",
                    "members": [
                        {
                            "logical_name": "metagenomic_db",
                            "source_path": "metagenomic_db",
                            "preexisting_checksum": None,
                        }
                    ],
                },
            ],
        }
    }
    declaration = database_set_declaration_from_mapping(payload)
    assert declaration.schema_version == CURRENT_CONTRACT_SCHEMA_VERSION

    payload["database_set_declaration"]["schema_version"] = 2
    with pytest.raises(ValueError, match="Unsupported DatabaseSetDeclaration schema_version 2"):
        database_set_declaration_from_mapping(payload)


def _write_yaml(path: Path, data: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
