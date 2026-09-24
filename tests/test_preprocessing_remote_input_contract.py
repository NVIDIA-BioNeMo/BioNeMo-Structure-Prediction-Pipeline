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

"""Tests for the preprocessing remote FASTA input contract and payload transport fields.

Covers:
* ``VerifiedRemoteInputLocation`` round-trip + validation.
* Preprocessing Plan/RunSpec payload ``transport`` / ``s3_publish_prefix``
  conditional serialization and digest preservation.
* ``_preprocessing_seam_transport_policy`` defaults to ``"local"``.
* Materialization copies transport/prefix Plan→RunSpec; binding guards reject drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bspp.orchestration.contract.phase import (
    VerifiedLocalInputLocation,
    VerifiedRemoteInputLocation,
    phase_runspec_from_mapping,
    preprocessing_input_location_from_mapping,
    verified_remote_input_location_from_mapping,
)
from bspp.orchestration.control.phase_materialization import materialize_phase

# Reuse the existing materialization fixture from test_phase_materialization
from tests.test_phase_materialization import FIXED_RUN_ID, FIXED_TIME, _fixture

_SHA = "a" * 64
_URI = "s3://example-bucket-bucket/inputs/remote.fa"


# ---------------------------------------------------------------------------
# VerifiedRemoteInputLocation
# ---------------------------------------------------------------------------


def test_verified_remote_input_location_round_trip() -> None:
    """VerifiedRemoteInputLocation serializes, deserializes, and round-trips exactly."""
    loc = VerifiedRemoteInputLocation(
        source_uri=_URI,
        sha256=_SHA,
        size_bytes=100,
        path="inputs/remote.fa",
    )
    mapping = loc.to_mapping()
    assert mapping["kind"] == "verified-remote-file"
    assert mapping["source_uri"] == _URI
    assert mapping["sha256"] == _SHA
    assert mapping["size_bytes"] == 100
    assert mapping["path"] == "inputs/remote.fa"
    reloaded = verified_remote_input_location_from_mapping(mapping)
    assert reloaded == loc


def test_verified_remote_input_location_rejects_bare_s3() -> None:
    """A bare 's3://' (no object key) is rejected."""
    with pytest.raises(ValueError, match="non-empty s3:// object key"):
        VerifiedRemoteInputLocation(source_uri="s3://", sha256=_SHA, size_bytes=1, path="a.fa")


def test_verified_remote_input_location_rejects_non_s3_uri() -> None:
    """A non-s3:// URI is rejected."""
    with pytest.raises(ValueError, match="non-empty s3:// object key"):
        VerifiedRemoteInputLocation(source_uri="https://example.com/x.fa", sha256=_SHA, size_bytes=1, path="a.fa")


def test_verified_remote_input_location_rejects_bad_sha256() -> None:
    """A non-64-hex sha256 is rejected."""
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        VerifiedRemoteInputLocation(source_uri=_URI, sha256="short", size_bytes=1, path="a.fa")


def test_verified_remote_input_location_rejects_zero_size() -> None:
    """A zero size_bytes is rejected."""
    with pytest.raises(ValueError, match="positive integer"):
        VerifiedRemoteInputLocation(source_uri=_URI, sha256=_SHA, size_bytes=0, path="a.fa")


def test_verified_remote_input_location_rejects_absolute_path() -> None:
    """An absolute path is rejected (must be relative under the attempt workspace)."""
    with pytest.raises(ValueError, match=r"relative .fa path"):
        VerifiedRemoteInputLocation(source_uri=_URI, sha256=_SHA, size_bytes=1, path="/abs/path.fa")


def test_verified_remote_input_location_rejects_traversal_path() -> None:
    """A path with '..' components is rejected (must not escape the attempt workspace)."""
    with pytest.raises(ValueError, match=r"'..' path components"):
        VerifiedRemoteInputLocation(source_uri=_URI, sha256=_SHA, size_bytes=1, path="../outside.fa")


def test_preprocessing_input_location_from_mapping_dispatches_local() -> None:
    """The dispatch helper routes 'verified-local-file' to the local loader."""
    loc = VerifiedLocalInputLocation(path="a.fa", sha256=_SHA, size_bytes=10)
    result = preprocessing_input_location_from_mapping(loc.to_mapping())
    assert isinstance(result, VerifiedLocalInputLocation)
    assert result == loc


def test_preprocessing_input_location_from_mapping_dispatches_remote() -> None:
    """The dispatch helper routes 'verified-remote-file' to the remote loader."""
    loc = VerifiedRemoteInputLocation(source_uri=_URI, sha256=_SHA, size_bytes=10, path="a.fa")
    result = preprocessing_input_location_from_mapping(loc.to_mapping())
    assert isinstance(result, VerifiedRemoteInputLocation)
    assert result == loc


def test_preprocessing_input_location_from_mapping_rejects_unknown_kind() -> None:
    """An unknown kind is rejected by the dispatch helper."""
    with pytest.raises(ValueError, match="unsupported input location kind"):
        preprocessing_input_location_from_mapping({"kind": "unknown", "schema_version": 1})


# ---------------------------------------------------------------------------
# Preprocessing payload transport fields
# ---------------------------------------------------------------------------


def _materialize(tmp_path: Path) -> dict[str, object]:
    """Materialize a preprocessing phase and return the RunSpec mapping."""
    fixture = _fixture(tmp_path)
    authority_root = tmp_path / "authority"
    result = materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    return json.loads(
        (authority_root / result.phase_run_id / "attempts" / "attempt-0001" / "phase-runspec.json").read_text()
    )


def test_default_transport_is_local_and_omitted_from_payload(tmp_path: Path) -> None:
    """Default transport serializes without the transport field in the payload (digest preservation)."""
    mapping = _materialize(tmp_path)
    runspec = phase_runspec_from_mapping(mapping)
    assert runspec.payload.transport == "local"
    assert runspec.payload.s3_publish_prefix is None
    assert "transport" not in mapping["payload"]
    assert "s3_publish_prefix" not in mapping["payload"]
    # Top-level PhaseRunSpec must NOT have transport fields
    assert "transport" not in mapping
    assert "s3_publish_prefix" not in mapping
    serialized = runspec.to_mapping()
    assert "transport" not in serialized["payload"]
    assert "s3_publish_prefix" not in serialized["payload"]
    assert phase_runspec_from_mapping(serialized) == runspec


def test_publish_to_s3_transport_round_trips_in_payload(tmp_path: Path) -> None:
    """A payload with transport='publish-to-s3' serializes and round-trips both fields."""
    mapping = _materialize(tmp_path)
    mapping["payload"]["transport"] = "publish-to-s3"
    mapping["payload"]["s3_publish_prefix"] = "s3://bucket/msa"
    runspec = phase_runspec_from_mapping(mapping)
    assert runspec.payload.transport == "publish-to-s3"
    assert runspec.payload.s3_publish_prefix == "s3://bucket/msa"
    serialized = runspec.to_mapping()
    assert serialized["payload"]["transport"] == "publish-to-s3"
    assert serialized["payload"]["s3_publish_prefix"] == "s3://bucket/msa"
    assert phase_runspec_from_mapping(serialized) == runspec


def test_publish_requires_prefix_in_payload(tmp_path: Path) -> None:
    """transport='publish-to-s3' without a prefix raises."""
    mapping = _materialize(tmp_path)
    mapping["payload"]["transport"] = "publish-to-s3"
    with pytest.raises(ValueError, match="requires a non-None s3_publish_prefix"):
        phase_runspec_from_mapping(mapping)


def test_prefix_rejected_when_transport_is_local_in_payload(tmp_path: Path) -> None:
    """s3_publish_prefix must be None when transport is 'local'."""
    mapping = _materialize(tmp_path)
    mapping["payload"]["s3_publish_prefix"] = "s3://bucket/msa"
    with pytest.raises(ValueError, match="s3_publish_prefix must be None"):
        phase_runspec_from_mapping(mapping)


def test_invalid_prefix_rejected_in_payload(tmp_path: Path) -> None:
    """An invalid s3:// prefix is rejected."""
    mapping = _materialize(tmp_path)
    mapping["payload"]["transport"] = "publish-to-s3"
    mapping["payload"]["s3_publish_prefix"] = "not-an-s3-prefix"
    with pytest.raises(ValueError, match="s3_publish_prefix must be a non-empty s3://"):
        phase_runspec_from_mapping(mapping)


def test_digest_preserved_when_defaults_in_payload(tmp_path: Path) -> None:
    """Adding transport fields with defaults does not change the digest."""
    mapping = _materialize(tmp_path)
    runspec = phase_runspec_from_mapping(mapping)
    original_digest = runspec.digest
    serialized = runspec.to_mapping()
    assert "transport" not in serialized["payload"]
    assert "s3_publish_prefix" not in serialized["payload"]
    assert runspec.digest == original_digest


def test_preprocessing_seam_transport_policy_defaults_to_local() -> None:
    """_preprocessing_seam_transport_policy defaults to 'local' when the key is absent."""
    from bspp.orchestration.contract.phase import _preprocessing_seam_transport_policy

    assert _preprocessing_seam_transport_policy({}, "transport") == "local"


def test_materialization_copies_transport_prefix_plan_to_runspec(tmp_path: Path) -> None:
    """Materialization copies the Plan payload's transport/prefix into the RunSpec payload."""
    import yaml

    fixture = _fixture(tmp_path)
    # Patch the plan to add transport fields
    plan_mapping = yaml.safe_load(fixture.plan_path.read_text())
    plan_mapping["payload"]["transport"] = "publish-to-s3"
    plan_mapping["payload"]["s3_publish_prefix"] = "s3://bucket/msa"
    fixture.plan_path.write_text(yaml.safe_dump(plan_mapping, sort_keys=True))
    authority_root = tmp_path / "authority"
    result = materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    runspec_path = authority_root / result.phase_run_id / "attempts" / "attempt-0001" / "phase-runspec.json"
    runspec = phase_runspec_from_mapping(json.loads(runspec_path.read_text()))
    assert runspec.payload.transport == "publish-to-s3"
    assert runspec.payload.s3_publish_prefix == "s3://bucket/msa"


def test_binding_guard_rejects_transport_drift(tmp_path: Path) -> None:
    """The authority validation rejects a RunSpec whose payload transport differs from the Plan's."""
    import yaml

    fixture = _fixture(tmp_path)
    plan_mapping = yaml.safe_load(fixture.plan_path.read_text())
    plan_mapping["payload"]["transport"] = "publish-to-s3"
    plan_mapping["payload"]["s3_publish_prefix"] = "s3://bucket/msa"
    fixture.plan_path.write_text(yaml.safe_dump(plan_mapping, sort_keys=True))
    authority_root = tmp_path / "authority"
    materialize_phase(
        fixture.plan_path,
        authority_root=authority_root,
        config_path=fixture.profile_path,
        source_repo=fixture.source_repo,
        clock=lambda: FIXED_TIME,
        phase_run_id_factory=lambda: FIXED_RUN_ID,
    )
    # Corrupt the runspec payload transport to differ from the plan
    runspec_path = authority_root / FIXED_RUN_ID / "attempts" / "attempt-0001" / "phase-runspec.json"
    runspec_mapping = json.loads(runspec_path.read_text())
    runspec_mapping["payload"]["transport"] = "local"
    runspec_mapping["payload"].pop("s3_publish_prefix", None)
    runspec_path.write_text(json.dumps(runspec_mapping, indent=2, sort_keys=True) + "\n")
    # Recompute the runspec digest and update the materialized event + phase-run
    from bspp.orchestration.contract.phase import phase_runspec_from_mapping

    runspec = phase_runspec_from_mapping(runspec_mapping)
    new_digest = runspec.digest
    phase_run_path = authority_root / FIXED_RUN_ID / "phase-run.json"
    phase_run_mapping = json.loads(phase_run_path.read_text())
    phase_run_mapping["attempts"][0]["phase_runspec_digest"] = new_digest
    phase_run_path.write_text(json.dumps(phase_run_mapping, indent=2, sort_keys=True) + "\n")
    event_path = authority_root / FIXED_RUN_ID / "events" / "000001-phase-materialized.json"
    event_mapping = json.loads(event_path.read_text())
    event_mapping["payload"]["phase_run"] = phase_run_mapping
    event_mapping["payload"]["phase_runspec"] = runspec_mapping
    event_path.write_text(json.dumps(event_mapping, indent=2, sort_keys=True) + "\n")

    from bspp.orchestration.control.phase_authority import PhaseAuthorityStore

    with pytest.raises(ValueError, match="transport does not bind"):
        PhaseAuthorityStore(authority_root).validate(FIXED_RUN_ID)
