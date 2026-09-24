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

"""Cross-story integration tests for use_env and the AFDB model-ID stem gate.

These tests walk the whole preprocessing chain — config -> plan -> materialize ->
execute -> finalize — with both new scientific knobs live, which no single
existing unit test exercises. Every per-story seam is unit-tested, but the
runtime lane flagged the missing whole-chain coverage for the production shape
(gate ON + conforming members + use_env=True) and for a plan legitimately built
gate-ON that later drifts.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
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
from bspp.orchestration.contract.preprocessing_execution import (
    PreprocessingScientificConfig,
    preprocessing_chunk_execution_intent_from_plan,
    preprocessing_chunk_execution_plan_from_mapping,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    preprocessing_content_validation_evidence_from_mapping,
)
from bspp.orchestration.runtime.preprocessing import finalization as finalization_module
from bspp.orchestration.runtime.preprocessing.content_validation import normalize_preprocessing_tar_member
from bspp.orchestration.runtime.preprocessing.finalization import (
    PreprocessingFinalizationError,
    finalize_preprocessing_chunk,
)

CONFORMING_MEMBERS = (
    "AFDB_AF-0000000000000001.a3m",
    "AFDB_AF-0000000000000002.a3m",
    "AFDB_AF-0000000000000003.a3m",
)
# Headers must be pure AFDB model IDs so the adapter-derived member names equal
# CONFORMING_MEMBERS under the gate-ON derivation-equality check.
CONFORMING_FASTA = (
    b">AFDB_AF-0000000000000001\nAAAA:TT\n>AFDB_AF-0000000000000002\nCCCC:AAA\n>AFDB_AF-0000000000000003\nGGGG:CC\n"
)


def _gate_on_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_env: bool,
) -> LocalExecutionFixture:
    fixture = preprocessing_execution_fixture(
        tmp_path,
        expected_a3m_members=CONFORMING_MEMBERS,
        scientific=PreprocessingScientificConfig(require_afdb_model_id_stem=True, use_env=use_env),
        fasta_bytes=CONFORMING_FASTA,
    )
    configure_preprocessing_fakes(fixture, monkeypatch)
    skip_preprocessing_server_warmup(monkeypatch)
    return fixture


def _rename_durable_tar_member(
    fixture: LocalExecutionFixture,
    evidence: PreprocessingChunkActionEvidence,
    *,
    original_name: str,
    replacement_name: str,
) -> PreprocessingChunkActionEvidence:
    """Rename one durable tar member by its declared name and re-attest the tar."""
    tar_path = Path(fixture.action.payload.package.durable_tar_path)
    with tarfile.open(tar_path, mode="r:") as archive:
        members = archive.getmembers()
        payloads = {
            header.name: stream.read()
            for header in members
            if header.isfile() and (stream := archive.extractfile(header)) is not None
        }
    with tarfile.open(tar_path, mode="w:") as archive:
        for header in members:
            if header.isdir():
                directory = tarfile.TarInfo(header.name)
                directory.type = tarfile.DIRTYPE
                archive.addfile(directory)
        for header in members:
            if not header.isfile():
                continue
            name = replacement_name if normalize_preprocessing_tar_member(header.name) == original_name else header.name
            rewritten = tarfile.TarInfo(name)
            payload = payloads[header.name]
            rewritten.size = len(payload)
            archive.addfile(rewritten, io.BytesIO(payload))

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


def test_use_env_and_stem_gate_walk_the_full_chain_with_conforming_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _gate_on_fixture(tmp_path, monkeypatch, use_env=True)
    plan = fixture.action.payload

    # e01s03 renderer emits --use-env 1 when use_env is true.
    assert plan.search_argv[plan.search_argv.index("--use-env") + 1] == "1"
    # e01s03 validator agrees with the renderer.
    assert preprocessing_chunk_execution_plan_from_mapping(plan.to_mapping()) == plan
    # e01s01 config -> intent carries both fields.
    intent = preprocessing_chunk_execution_intent_from_plan(plan)
    assert intent.scientific.use_env is True
    assert intent.scientific.require_afdb_model_id_stem is True

    # Runtime accepts the gate-ON plan and the fake kernel publishes conforming members.
    result = invoke_preprocessing_execution(fixture)
    assert result.exit_code == 0, result.output

    # e01s02 re-check passes on conforming published members.
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    handoff = tmp_path / "handoff"
    finalize_preprocessing_chunk(fixture.runspec, evidence, handoff_path=handoff)
    assert (handoff / "artifact-set.json").exists()
    validation = preprocessing_content_validation_evidence_from_mapping(
        json.loads((handoff / "content-validation.json").read_text())
    )
    assert validation.outcome == "passed"
    assert validation.artifact_set_id is not None


def test_finalizer_rechecks_stem_after_a_gate_on_plan_drifts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _gate_on_fixture(tmp_path, monkeypatch, use_env=False)
    result = invoke_preprocessing_execution(fixture)
    assert result.exit_code == 0, result.output
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))

    # Introduce drift after a legitimate gate-ON execution: rename the first
    # declared durable tar member AND the matching ExpectedA3M.member_name so the
    # tar inventory check still passes and the stem re-check is the thing that fires.
    action = fixture.action
    original_member = action.payload.expected_a3ms[0].member_name
    drifted_expected = replace(action.payload.expected_a3ms[0], member_name="AFDB_drifted.a3m")
    new_payload = copy(action.payload)
    object.__setattr__(new_payload, "expected_a3ms", (drifted_expected, *action.payload.expected_a3ms[1:]))
    new_action = copy(action)
    object.__setattr__(new_action, "payload", new_payload)
    new_runspec_payload = copy(fixture.runspec.payload)
    object.__setattr__(new_runspec_payload, "actions", (new_action,))
    new_runspec = copy(fixture.runspec)
    object.__setattr__(new_runspec, "payload", new_runspec_payload)

    forged = _rename_durable_tar_member(
        fixture,
        evidence,
        original_name=original_member,
        replacement_name="./AFDB_drifted.a3m",
    )
    # Mutating the RunSpec changes its digest; the reconciliation under test is
    # the finalization re-check, not evidence identity, so scope it out as the
    # existing drift test does.
    monkeypatch.setattr(
        finalization_module,
        "reconcile_preprocessing_chunk_action_evidence_for_finalization",
        lambda _runspec, _evidence: None,
    )
    handoff = tmp_path / "drift-handoff"

    with pytest.raises(
        PreprocessingFinalizationError,
        match="does not carry a discoverable AFDB or PDB assembly model ID",
    ):
        finalize_preprocessing_chunk(new_runspec, forged, handoff_path=handoff)

    _assert_failed_content_validation_only(handoff)
