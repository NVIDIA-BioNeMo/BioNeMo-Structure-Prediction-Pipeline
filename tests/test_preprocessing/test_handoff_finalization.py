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

"""Runtime finalization tests for one exact preprocessing bundle."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner
from tests.support.preprocessing_execution import (
    LocalExecutionFixture,
    configure_preprocessing_fakes,
    invoke_preprocessing_execution,
    preprocessing_execution_fixture,
    skip_preprocessing_server_warmup,
)

from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingChunkActionEvidence,
    preprocessing_chunk_action_evidence_from_mapping,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    msa_artifact_set_manifest_from_mapping,
    msa_chunk_manifest_from_mapping,
    preprocessing_content_validation_evidence_from_mapping,
    verified_local_bundled_artifact_location_from_mapping,
    verified_local_bundled_artifact_location_id,
)
from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.preprocessing import finalization as finalization_module
from bspp.orchestration.runtime.preprocessing.execution import (
    reconcile_preprocessing_chunk_action_evidence,
    reconcile_preprocessing_chunk_action_evidence_for_finalization,
)
from bspp.orchestration.runtime.preprocessing.finalization import (
    PreprocessingFinalizationError,
    finalize_preprocessing_chunk,
)


def _successful_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    legacy_paired: bool = False,
) -> tuple[LocalExecutionFixture, PreprocessingChunkActionEvidence]:
    fixture = preprocessing_execution_fixture(tmp_path, legacy_paired=legacy_paired)
    configure_preprocessing_fakes(fixture, monkeypatch)
    skip_preprocessing_server_warmup(monkeypatch)
    result = invoke_preprocessing_execution(fixture)
    assert result.exit_code == 0, result.output
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    return fixture, evidence


def _replace_first_durable_tar_member(
    fixture: LocalExecutionFixture,
    evidence: PreprocessingChunkActionEvidence,
    *,
    replacement_name: str | None = None,
    replacement_payload: bytes | None = None,
    replacement_type: bytes | None = None,
    duplicate_first_and_omit_second: bool = False,
) -> PreprocessingChunkActionEvidence:
    tar_path = Path(fixture.action.payload.package.durable_tar_path)
    with tarfile.open(tar_path, mode="r:") as archive:
        original_headers = tuple(archive.getmembers())
        original_payloads = {
            header.name: stream.read()
            for header in original_headers
            if header.isfile() and (stream := archive.extractfile(header)) is not None
        }
    regular_headers = tuple(header for header in original_headers if header.isfile())
    assert len(regular_headers) >= 2
    rewritten_headers = list(regular_headers)
    if duplicate_first_and_omit_second:
        rewritten_headers[1] = regular_headers[0]

    with tarfile.open(tar_path, mode="w:") as archive:
        for original in original_headers:
            if original.isdir():
                header = tarfile.TarInfo(original.name)
                header.type = tarfile.DIRTYPE
                archive.addfile(header)
        for index, original in enumerate(rewritten_headers):
            name = replacement_name if index == 0 and replacement_name is not None else original.name
            header = tarfile.TarInfo(name)
            if index == 0 and replacement_type is not None:
                header.type = replacement_type
                header.linkname = regular_headers[1].name
                archive.addfile(header)
                continue
            payload = (
                replacement_payload
                if index == 0 and replacement_payload is not None
                else original_payloads[original.name]
            )
            header.size = len(payload)
            archive.addfile(header, io.BytesIO(payload))

    mapping = deepcopy(evidence.to_mapping())
    tar_bytes = tar_path.read_bytes()
    archive_mapping = cast("dict[str, object]", mapping["archive_evidence"])
    archive_mapping["tar_size_bytes"] = len(tar_bytes)
    for output in cast("list[dict[str, object]]", mapping["output_hashes"]):
        if output["role"] == "tar":
            output["size_bytes"] = len(tar_bytes)
            output["sha256"] = hashlib.sha256(tar_bytes).hexdigest()
    return preprocessing_chunk_action_evidence_from_mapping(mapping)


def _assert_failed_content_validation_only(handoff: Path) -> None:
    assert {path.relative_to(handoff) for path in handoff.rglob("*")} == {Path("content-validation.json")}
    failed = preprocessing_content_validation_evidence_from_mapping(
        json.loads((handoff / "content-validation.json").read_text())
    )
    assert failed.outcome == "failed"
    assert failed.artifact_set_id is None
    assert failed.artifact_location_id is None


def test_finalizer_rechecks_afdb_model_id_stem_when_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch)
    action = fixture.action
    # The plan-time gate rejects a non-conforming member with the flag ON, so flip
    # the flag on a shallow copy without re-running ``__post_init__`` validation:
    # the re-check is defence-in-depth for drift, not a second plan-time gate.
    new_payload = copy(action.payload)
    object.__setattr__(
        new_payload,
        "scientific",
        action.payload.scientific.model_copy(update={"require_afdb_model_id_stem": True}),
    )
    new_action = copy(action)
    object.__setattr__(new_action, "payload", new_payload)
    new_runspec_payload = copy(fixture.runspec.payload)
    object.__setattr__(new_runspec_payload, "actions", (new_action,))
    new_runspec = copy(fixture.runspec)
    object.__setattr__(new_runspec, "payload", new_runspec_payload)
    monkeypatch.setattr(
        finalization_module,
        "reconcile_preprocessing_chunk_action_evidence_for_finalization",
        lambda runspec, evidence: None,
    )
    handoff = tmp_path / "stem-recheck-handoff"

    with pytest.raises(
        PreprocessingFinalizationError,
        match="does not carry a discoverable AFDB or PDB assembly model ID",
    ):
        finalize_preprocessing_chunk(new_runspec, evidence, handoff_path=handoff)

    _assert_failed_content_validation_only(handoff)


def test_finalize_chunk_publishes_deterministic_logical_and_separate_location_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch)
    handoff = tmp_path / "handoff"

    result = CliRunner().invoke(
        cli,
        [
            "preprocessing",
            "finalize-chunk",
            "--phase-runspec",
            str(fixture.runspec_path),
            "--action-evidence",
            str(fixture.evidence_path),
            "--write-handoff",
            str(handoff),
        ],
    )

    assert result.exit_code == 0, result.output
    chunk_path = handoff / f"chunks/{fixture.action.payload.chunk_name.removesuffix('.fa')}.json"
    assert {path.relative_to(handoff) for path in handoff.rglob("*")} == {
        Path("chunks"),
        chunk_path.relative_to(handoff),
        Path("artifact-set.json"),
        Path("artifact-location.json"),
        Path("content-validation.json"),
    }
    chunk = msa_chunk_manifest_from_mapping(json.loads(chunk_path.read_text()))
    artifact_set = msa_artifact_set_manifest_from_mapping(json.loads((handoff / "artifact-set.json").read_text()))
    location = verified_local_bundled_artifact_location_from_mapping(
        json.loads((handoff / "artifact-location.json").read_text())
    )
    validation = preprocessing_content_validation_evidence_from_mapping(
        json.loads((handoff / "content-validation.json").read_text())
    )
    assert tuple(member.source_ordinal for member in chunk.members) == tuple(
        sorted(member.source_ordinal for member in chunk.members)
    )
    assert artifact_set.chunks[0].sha256 == chunk.digest
    assert artifact_set.artifact_set_id == validation.artifact_set_id
    assert location.artifact_location_id == validation.artifact_location_id
    assert location.artifact_set_id == artifact_set.artifact_set_id
    assert (
        validation.action_evidence_digest
        == hashlib.sha256(json.dumps(evidence.to_mapping(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )


@pytest.mark.parametrize("legacy_paired", [False, True], ids=["unpaired_paired", "legacy-paired"])
def test_finalization_scoped_reconciliation_never_reads_scratch_a3ms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_paired: bool,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch, legacy_paired=legacy_paired)
    for output in evidence.output_hashes:
        if output.role == "a3m":
            Path(output.path).unlink()
    assert evidence.raw_search_evidence is not None
    shutil.rmtree(evidence.raw_search_evidence.raw_search_output_directory)

    with pytest.raises(OSError):
        reconcile_preprocessing_chunk_action_evidence(fixture.runspec, evidence)

    reconcile_preprocessing_chunk_action_evidence_for_finalization(fixture.runspec, evidence)
    result = CliRunner().invoke(
        cli,
        [
            "preprocessing",
            "finalize-chunk",
            "--phase-runspec",
            str(fixture.runspec_path),
            "--action-evidence",
            str(fixture.evidence_path),
            "--write-handoff",
            str(tmp_path / "handoff"),
        ],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("mutation", ["named-hash", "placeholder-model", "unexpected-numeric"])
@pytest.mark.parametrize("verify_raw_files", [False, True])
def test_finalization_rejects_raw_attestation_tampering_without_opening_raw_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    verify_raw_files: bool,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch, legacy_paired=mutation == "placeholder-model")
    assert evidence.raw_search_evidence is not None
    if not verify_raw_files:
        shutil.rmtree(evidence.raw_search_evidence.raw_search_output_directory)
    mapping = deepcopy(evidence.to_mapping())
    raw = cast("dict[str, object]", mapping["raw_search_evidence"])
    artifacts = cast("list[dict[str, object]]", raw["artifacts"])
    if mutation == "named-hash":
        artifacts[0]["sha256"] = "f" * 64
    elif mutation == "placeholder-model":
        artifacts[-1]["modeled_chain_length"] = cast("int", artifacts[-1]["modeled_chain_length"]) + 1
    else:
        data = b"#3\t1\n"
        artifacts.append(
            {
                "schema_version": 1,
                "role": "numeric-placeholder",
                "member_name": "3.a3m",
                "path": f"{raw['raw_search_output_directory']}/3.a3m",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "source_ordinal": None,
                "declared_member": None,
                "raw_query_id": 3,
                "modeled_chain_length": 3,
                "modeled_cardinality": 1,
            }
        )
    tampered = preprocessing_chunk_action_evidence_from_mapping(mapping)

    with pytest.raises(ValueError, match=r"raw (named|numeric|artifact count)"):
        if verify_raw_files:
            reconcile_preprocessing_chunk_action_evidence(fixture.runspec, tampered)
        else:
            reconcile_preprocessing_chunk_action_evidence_for_finalization(fixture.runspec, tampered)


def test_finalizer_rejects_member_drift_even_when_durable_tar_attestation_is_updated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch)
    tar_path = Path(fixture.action.payload.package.durable_tar_path)
    with tarfile.open(tar_path, mode="w:") as archive:
        root = tarfile.TarInfo("./")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for index, expected in enumerate(fixture.action.payload.expected_a3ms):
            payload = b">drifted\nCHANGED\n" if index == 0 else f">{expected.record_identity}\nAAAA\n".encode()
            name = f"./{expected.member_name}"
            header = tarfile.TarInfo(name)
            header.size = len(payload)
            archive.addfile(header, io.BytesIO(payload))
    with tarfile.open(tar_path, mode="r:") as archive:
        raw_names = archive.getnames()
    mapping = deepcopy(evidence.to_mapping())
    tar_bytes = tar_path.read_bytes()
    archive_mapping = cast("dict[str, object]", mapping["archive_evidence"])
    archive_mapping["tar_size_bytes"] = len(tar_bytes)
    archive_mapping["tar_members"] = raw_names
    for output in cast("list[dict[str, object]]", mapping["output_hashes"]):
        if output["role"] == "tar":
            output["size_bytes"] = len(tar_bytes)
            output["sha256"] = hashlib.sha256(tar_bytes).hexdigest()
    forged_path = tmp_path / "forged-action.json"
    forged_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    forged_path.chmod(0o444)

    result = CliRunner().invoke(
        cli,
        [
            "preprocessing",
            "finalize-chunk",
            "--phase-runspec",
            str(fixture.runspec_path),
            "--action-evidence",
            str(forged_path),
            "--write-handoff",
            str(tmp_path / "failed-handoff"),
        ],
    )

    assert result.exit_code != 0
    assert "declared A3M must start with a ColabFold metadata header" in result.output
    assert {path.name for path in (tmp_path / "failed-handoff").iterdir()} == {"content-validation.json"}
    failed = preprocessing_content_validation_evidence_from_mapping(
        json.loads((tmp_path / "failed-handoff/content-validation.json").read_text())
    )
    assert failed.outcome == "failed"
    assert failed.artifact_set_id is None


def test_finalizer_rejects_duplicate_member_with_missing_declared_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch)
    forged = _replace_first_durable_tar_member(
        fixture,
        evidence,
        duplicate_first_and_omit_second=True,
    )
    handoff = tmp_path / "duplicate-member-handoff"

    with pytest.raises(PreprocessingFinalizationError, match="inventory does not match"):
        finalize_preprocessing_chunk(fixture.runspec, forged, handoff_path=handoff)

    _assert_failed_content_validation_only(handoff)


@pytest.mark.parametrize("unsafe_prefix", ["/", "../"], ids=["absolute", "parent"])
def test_finalizer_rejects_unsafe_durable_tar_member_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_prefix: str,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch)
    member_name = fixture.action.payload.expected_a3ms[0].member_name
    forged = _replace_first_durable_tar_member(
        fixture,
        evidence,
        replacement_name=f"{unsafe_prefix}{member_name}",
    )
    handoff = tmp_path / f"unsafe-{unsafe_prefix.replace('/', 'slash').replace('.', 'dot')}-handoff"

    with pytest.raises(PreprocessingFinalizationError, match="unsafe preprocessing tar member"):
        finalize_preprocessing_chunk(fixture.runspec, forged, handoff_path=handoff)

    _assert_failed_content_validation_only(handoff)


def test_finalizer_rejects_nested_durable_tar_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch)
    member_name = fixture.action.payload.expected_a3ms[0].member_name
    forged = _replace_first_durable_tar_member(
        fixture,
        evidence,
        replacement_name=f"nested/{member_name}",
    )
    handoff = tmp_path / "nested-member-handoff"

    with pytest.raises(PreprocessingFinalizationError, match="undeclared preprocessing tar member type or path"):
        finalize_preprocessing_chunk(fixture.runspec, forged, handoff_path=handoff)

    _assert_failed_content_validation_only(handoff)


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE], ids=["symlink", "hardlink"])
def test_finalizer_rejects_link_replacing_declared_regular_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_type: bytes,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch)
    forged = _replace_first_durable_tar_member(
        fixture,
        evidence,
        replacement_type=link_type,
    )
    handoff = tmp_path / f"link-{link_type.decode()}-handoff"

    with pytest.raises(PreprocessingFinalizationError, match="undeclared preprocessing tar member type or path"):
        finalize_preprocessing_chunk(fixture.runspec, forged, handoff_path=handoff)

    _assert_failed_content_validation_only(handoff)


def test_finalizer_rejects_invalid_utf8_durable_member_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch)
    forged = _replace_first_durable_tar_member(
        fixture,
        evidence,
        replacement_payload=b"\xff\xfe\x80",
    )
    handoff = tmp_path / "invalid-utf8-handoff"

    with pytest.raises(PreprocessingFinalizationError, match="not UTF-8 text"):
        finalize_preprocessing_chunk(fixture.runspec, forged, handoff_path=handoff)

    _assert_failed_content_validation_only(handoff)


def test_finalizer_rejects_wrong_paired_metadata_without_raw_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch)
    assert evidence.raw_search_evidence is not None
    shutil.rmtree(evidence.raw_search_evidence.raw_search_output_directory)
    for output in evidence.output_hashes:
        if output.role == "a3m":
            Path(output.path).unlink()
    forged = _replace_first_durable_tar_member(
        fixture,
        evidence,
        replacement_payload=b"#1,1\t1,1\n>query\nAA:TT\n",
    )
    handoff = tmp_path / "wrong-paired-metadata-handoff"

    with pytest.raises(
        PreprocessingFinalizationError,
        match="declared A3M metadata does not match expected chain lengths",
    ):
        finalize_preprocessing_chunk(fixture.runspec, forged, handoff_path=handoff)

    _assert_failed_content_validation_only(handoff)


def test_finalization_rejects_inexact_a3m_attestation_metadata_without_scratch_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch)
    mapping = deepcopy(evidence.to_mapping())
    outputs = cast("list[dict[str, object]]", mapping["output_hashes"])
    a3m = next(output for output in outputs if output["role"] == "a3m")
    a3m["path"] = str(tmp_path / "undeclared" / cast("str", a3m["member_name"]))
    forged = preprocessing_chunk_action_evidence_from_mapping(mapping)

    with pytest.raises(ValueError, match="exact declared outputs"):
        reconcile_preprocessing_chunk_action_evidence_for_finalization(fixture.runspec, forged)


def test_finalizer_rejects_corrupt_lz4_even_when_its_file_attestation_is_updated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch)
    lz4_path = Path(fixture.action.payload.package.durable_lz4_path)
    lz4_path.write_bytes(b"not-an-lz4-stream")
    mapping = deepcopy(evidence.to_mapping())
    payload = lz4_path.read_bytes()
    archive_mapping = cast("dict[str, object]", mapping["archive_evidence"])
    archive_mapping["lz4_size_bytes"] = len(payload)
    for output in cast("list[dict[str, object]]", mapping["output_hashes"]):
        if output["role"] == "tar-lz4":
            output["size_bytes"] = len(payload)
            output["sha256"] = hashlib.sha256(payload).hexdigest()
    forged_path = tmp_path / "forged-lz4-action.json"
    forged_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    forged_path.chmod(0o444)

    result = CliRunner().invoke(
        cli,
        [
            "preprocessing",
            "finalize-chunk",
            "--phase-runspec",
            str(fixture.runspec_path),
            "--action-evidence",
            str(forged_path),
            "--write-handoff",
            str(tmp_path / "failed-lz4-handoff"),
        ],
    )

    assert result.exit_code != 0
    assert "LZ4 verification failed" in result.output
    assert {path.name for path in (tmp_path / "failed-lz4-handoff").iterdir()} == {"content-validation.json"}


def test_finalization_refuses_to_replace_an_existing_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _evidence = _successful_execution(tmp_path, monkeypatch)
    handoff = tmp_path / "owned"
    handoff.mkdir()
    marker = handoff / "owner.txt"
    marker.write_text("preserve")

    result = CliRunner().invoke(
        cli,
        [
            "preprocessing",
            "finalize-chunk",
            "--phase-runspec",
            str(fixture.runspec_path),
            "--action-evidence",
            str(fixture.evidence_path),
            "--write-handoff",
            str(handoff),
        ],
    )

    assert result.exit_code != 0
    assert marker.read_text() == "preserve"
    assert list(handoff.iterdir()) == [marker]


def test_artifact_set_identity_is_location_independent_and_loaders_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, evidence = _successful_execution(tmp_path, monkeypatch)
    handoff_path = tmp_path / "handoff"
    result = CliRunner().invoke(
        cli,
        [
            "preprocessing",
            "finalize-chunk",
            "--phase-runspec",
            str(fixture.runspec_path),
            "--action-evidence",
            str(fixture.evidence_path),
            "--write-handoff",
            str(handoff_path),
        ],
    )
    assert result.exit_code == 0, result.output
    artifact_set_mapping = json.loads((handoff_path / "artifact-set.json").read_text())
    location = verified_local_bundled_artifact_location_from_mapping(
        json.loads((handoff_path / "artifact-location.json").read_text())
    )
    relocated_bundle = (tmp_path / "relocated/bundle.tar.lz4").absolute()
    relocated_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=location.artifact_set_id,
        tar_path=str((tmp_path / "relocated/bundle.tar").absolute()),
        bundle_path=str(relocated_bundle),
        bundle_uri=relocated_bundle.as_uri(),
        tar_size_bytes=location.tar_size_bytes,
        tar_sha256=location.tar_sha256,
        lz4_size_bytes=location.lz4_size_bytes,
        lz4_sha256=location.lz4_sha256,
        raw_tar_members=location.raw_tar_members,
        members=location.members,
    )
    relocated = replace(
        location,
        artifact_location_id=relocated_id,
        tar_path=str((tmp_path / "relocated/bundle.tar").absolute()),
        bundle_path=str(relocated_bundle),
        bundle_uri=relocated_bundle.as_uri(),
    )
    assert relocated.artifact_set_id == location.artifact_set_id
    assert relocated.artifact_location_id != location.artifact_location_id
    assert msa_artifact_set_manifest_from_mapping(artifact_set_mapping).artifact_set_id == location.artifact_set_id

    artifact_set_mapping["unexpected"] = evidence.action_id
    with pytest.raises(ValueError, match="Unknown MsaArtifactSetManifest"):
        msa_artifact_set_manifest_from_mapping(artifact_set_mapping)
