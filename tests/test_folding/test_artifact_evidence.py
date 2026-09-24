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

"""Actual-file v2 reduction and bounded transfer; generic fixtures only."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.support.folding_artifact_fixture import _ACTION_IDS, action_root, fixture, run_action, run_to_fold

from bspp.orchestration.contract.folding_artifact_evidence import ArtifactFoldEvidence, ArtifactFoldTarget
from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.control import folding_artifact_evidence as control
from bspp.orchestration.runtime.folding import artifact_evidence as runtime
from bspp.orchestration.runtime.folding.executor import FoldingExecutorError


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "baked")
    monkeypatch.setenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", "f" * 40)
    runspec, deps = fixture(tmp_path)
    fold = run_to_fold(tmp_path, runspec, deps)
    outputs_before = {path: path.read_bytes() for path in fold.rglob("*") if path.is_file()}
    canonical = run_action(tmp_path, runspec, deps, "canonical-pair")
    assert all(path.read_bytes() == content for path, content in outputs_before.items())
    bundle = tmp_path / "bundle"
    index = json.loads((canonical / "finalization-index.json").read_bytes())
    for relative in [f"{canonical.name}/finalization-index.json", *(item["path"] for item in index["members"])]:
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(canonical.parent / relative, target)
    return runspec, canonical, bundle


def test_actual_original_files_reduce_to_bounded_validated_bundle(completed):
    runspec, canonical, bundle = completed
    evidence = control.validate_local_folding_bundle(bundle, runspec=runspec)
    fold = ArtifactFoldEvidence.from_mapping(evidence[_ACTION_IDS["fold"]])
    terminal = ArtifactFoldEvidence.from_mapping(evidence[_ACTION_IDS["canonical-pair"]])
    assert fold.entries == terminal.entries
    assert terminal.predecessor_digest == canonical_mapping_digest(fold.to_mapping())
    record = fold.entries[0]
    score_bytes = Path(record.scores.path).read_bytes()
    assert record.scores.sha256 == hashlib.sha256(score_bytes).hexdigest()
    assert json.loads(score_bytes)["preserved_extra"] == {"unicode": "\u03b1", "signed_zero": -0.0}
    assert record.model_metadata["model_source"] == "alphafold2_multimer_1"
    assert '"pae":' not in (canonical / "action-evidence.json").read_text()
    assert '"scores": {' not in (canonical / "action-evidence.json").read_text()
    assert (
        json.loads((canonical / "canonical-pair-index.json").read_bytes())["entries"][0]["target_id"]
        == record.target.target_id
    )


@pytest.mark.parametrize("workers", [16, 32])
def test_real_rank_journal_closure_for_requested_topologies(tmp_path, monkeypatch, workers):
    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "baked")
    monkeypatch.setenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", "f" * 40)
    runspec, deps = fixture(tmp_path, workers=workers)
    fold = run_to_fold(tmp_path, runspec, deps)
    assert len(list(fold.glob("ranks/*/journal.jsonl"))) == workers
    canonical = run_action(tmp_path, runspec, deps, "canonical-pair")
    assert (canonical / "finalization-index.json").is_file()


@pytest.mark.parametrize("witness", ["residue_count", "plddt_count", "pae_rows", "pae_columns"])
@pytest.mark.parametrize("invalid", [True, 1.0, "1", 0, -1])
def test_content_witness_integer_types_are_strict_even_for_length_one(completed, witness, invalid):
    _, canonical, _ = completed
    payload = json.loads((canonical / "action-evidence.json").read_bytes())[_ACTION_IDS["fold"]]["entries"][0]
    payload["target"]["chains"] = ["A"]
    payload["target"]["sequence_sha256"] = hashlib.sha256(b"A").hexdigest()
    for field in ("residue_count", "plddt_count", "pae_rows", "pae_columns"):
        payload["pair_reference"]["content_validation"][field] = 1
    payload["pair_reference"]["content_validation"][witness] = invalid
    with pytest.raises(ValueError):
        ArtifactFoldTarget.from_mapping(payload)


@pytest.mark.parametrize("field", ["structure", "scores"])
@pytest.mark.parametrize("wrong", ["foreign-model_v1.pdb", "foreign-meta_v1.json", "raw.pdb", "raw.json"])
def test_prediction_descriptor_names_cannot_name_another_target(completed, field, wrong):
    _, canonical, _ = completed
    payload = json.loads((canonical / "action-evidence.json").read_bytes())[_ACTION_IDS["fold"]]["entries"][0]
    artifact = payload["pair_reference"]["artifacts"][field == "scores"]
    artifact["path"] = str(Path(artifact["path"]).with_name(wrong))
    with pytest.raises(ValueError, match="basenames"):
        ArtifactFoldTarget.from_mapping(payload)


@pytest.mark.parametrize("mutation", ["missing_model", "wrong_model", "ragged", "nonfinite", "wrong_length"])
def test_new_profile_rejects_invalid_original_score_bytes(tmp_path, mutation):
    path = tmp_path / "scores.json"
    data = {
        "schema_version": 1,
        "plddt": [80.0],
        "pae": [[0.1]],
        "max_pae": 0.1,
        "ptm": 0.8,
        "iptm": None,
        "bioir_model_source": "openfold2_ptm_1",
    }
    if mutation == "missing_model":
        del data["bioir_model_source"]
    elif mutation == "wrong_model":
        data["bioir_model_source"] = "alphafold2_multimer_1"
    elif mutation == "ragged":
        data["pae"] = [[0.1, 0.2]]
    elif mutation == "nonfinite":
        data["pae"] = [[float("nan")]]
    else:
        data["plddt"] = [80.0, 80.0]
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        runtime.read_scores(path, root=tmp_path, length=1, model_source="openfold2_ptm_1")
    assert path.read_bytes() == before


@pytest.mark.parametrize("ancestor", [False, True])
def test_snapshot_rejects_leaf_and_ancestor_symlinks(tmp_path, ancestor):
    actual = tmp_path / "actual"
    actual.mkdir()
    file = actual / "scores.json"
    file.write_bytes(b"{}")
    link = tmp_path / "link"
    link.symlink_to(actual if ancestor else file)
    with pytest.raises((ValueError, OSError)):
        runtime.snapshot_file(link / file.name if ancestor else link, root=tmp_path, maximum_bytes=100)


def test_snapshot_enforces_size_before_read(tmp_path):
    path = tmp_path / "large"
    path.write_bytes(b"12345")
    with pytest.raises(ValueError, match="size bound"):
        runtime.snapshot_file(path, root=tmp_path, maximum_bytes=4)


def test_metadata_publication_failure_preserves_destination_and_cleans_partial(tmp_path, monkeypatch):
    path = tmp_path / "metadata.json"
    path.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        runtime.write_metadata(path, {"value": 1})
    assert path.read_bytes() == b"preserved"
    path.unlink()
    monkeypatch.setattr(runtime, "MAX_METADATA_BYTES", 10)
    with pytest.raises(ValueError, match="size bound"):
        runtime.write_metadata(path, {"large": "x" * 100})
    assert list(tmp_path.iterdir()) == []


def test_bundle_tampering_extra_files_and_wrong_action_path_fail_closed(completed):
    runspec, canonical, bundle = completed
    with pytest.raises(ValueError, match="exact indexed"):
        control.validate_local_folding_bundle(
            bundle, runspec=runspec, action_evidence_path=canonical / "action-evidence.json"
        )
    extra = bundle / "extra.json"
    extra.write_bytes(b"{}")
    with pytest.raises(ValueError, match="bundle"):
        control.validate_local_folding_bundle(bundle, runspec=runspec)
    extra.unlink()
    member = bundle / canonical.name / "canonical-pair-index.json"
    member.write_bytes(member.read_bytes().replace(b"AlphaFold-Multimer", b"AlphaFold-Multimo"))
    with pytest.raises(ValueError, match="published index"):
        control.validate_local_folding_bundle(bundle, runspec=runspec)


def _authority(runspec):
    canonical = _ACTION_IDS["canonical-pair"]
    return SimpleNamespace(
        current_runspec_projection_complete=True,
        phase_runspec=runspec,
        submission=SimpleNamespace(
            status="submitted", actions=[SimpleNamespace(action_id=canonical, status="submitted", job_id="42")]
        ),
        terminal_observations=[
            SimpleNamespace(
                action_id=canonical,
                job_id="42",
                state="COMPLETED",
                exit_code="0:0",
                source="sacct",
                outcome="succeeded",
            )
        ],
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "43"),
        ("state", "RUNNING"),
        ("exit_code", "0"),
        ("exit_code", "1:0"),
        ("source", "squeue"),
        ("outcome", "failed"),
    ],
)
def test_fetch_requires_exact_durable_canonical_success(completed, field, value):
    runspec, _, _ = completed
    authority = _authority(runspec)
    setattr(authority.terminal_observations[0], field, value)
    with pytest.raises(ValueError, match="assigned canonical"):
        control._require_terminal(authority)


def test_public_transfer_reads_only_indexed_metadata_and_is_create_once(completed, tmp_path, monkeypatch):
    runspec, canonical, _ = completed
    monkeypatch.setattr(control.PhaseAuthorityStore, "validate", lambda *_: _authority(runspec))
    fetched = []

    class Source:
        def fetch_stable_artifact(self, path, *, remote_root, maximum_bytes):
            assert path.is_relative_to(canonical.parent)
            assert path.suffix == ".json"
            assert "/ranks/" not in str(path)
            fetched.append(path)
            data = path.read_bytes()
            assert len(data) <= maximum_bytes
            return data

    destination = tmp_path / "fetched"
    result = control.fetch_folding_finalization_evidence(
        runspec.phase_run_id, authority_root=tmp_path, destination=destination, source=Source()
    )
    assert result["indexed_file_count"] == 4
    assert len(fetched) == 6  # index, four members, unchanged index again
    assert control.validate_local_folding_bundle(destination, runspec=runspec)
    before = {p: p.read_bytes() for p in destination.rglob("*.json")}
    control.fetch_folding_finalization_evidence(
        runspec.phase_run_id, authority_root=tmp_path, destination=destination, source=Source()
    )
    assert before == {p: p.read_bytes() for p in destination.rglob("*.json")}


def test_fetch_accepts_relative_destination(completed, tmp_path, monkeypatch):
    runspec, _, _ = completed
    monkeypatch.setattr(control.PhaseAuthorityStore, "validate", lambda *_: _authority(runspec))

    class Source:
        def fetch_stable_artifact(self, path, *, remote_root, maximum_bytes):
            return path.read_bytes()

    destination = tmp_path / "fetched"
    relative = Path("fetched")
    monkeypatch.chdir(tmp_path)
    result = control.fetch_folding_finalization_evidence(
        runspec.phase_run_id, authority_root=tmp_path, destination=relative, source=Source()
    )
    assert result["status"] == "fetched"
    assert destination.exists()
    assert control.validate_local_folding_bundle(destination, runspec=runspec)


@pytest.mark.parametrize("skip_workers", [False, True])
def test_carried_original_files_reduce_under_current_attempt(tmp_path, monkeypatch, skip_workers):
    from tests.support.folding_artifact_fixture import successor_with_carry

    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "baked")
    monkeypatch.setenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", "f" * 40)
    runspec, deps = fixture(tmp_path)
    fold = run_to_fold(tmp_path, runspec, deps)
    original = {p: p.read_bytes() for p in fold.rglob("*") if p.is_file()}
    successor, successor_deps, carry = successor_with_carry(tmp_path, runspec, deps, fold)
    for kind in ("msa-flatten", "split", "preprocess"):
        run_action(tmp_path, successor, successor_deps, kind)
    if not skip_workers:
        for rank in range(2):
            run_action(tmp_path, successor, successor_deps, "fold", rank=rank, carry_record_path=carry)
    canonical = run_action(tmp_path, successor, successor_deps, "canonical-pair", carry_record_path=carry)
    mapping = json.loads((canonical / "action-evidence.json").read_bytes())
    entry = ArtifactFoldEvidence.from_mapping(mapping[_ACTION_IDS["fold"]]).entries[0]
    assert "/attempt-0002/" in entry.scores.path
    assert "/attempt-0002/" in entry.structure.path
    assert all(path.read_bytes() == data for path, data in original.items())
    assert len((canonical.parent / _ACTION_IDS["fold"] / "ranks/0/adopted.jsonl").read_text().splitlines()) == 1


def test_packed_completed_score_objects_are_released_before_session_close(tmp_path, monkeypatch):
    import weakref

    references = []
    original = runtime.read_scores

    def observe(*args, **kwargs):
        scores, identity = original(*args, **kwargs)
        references.append(weakref.ref(scores))
        return scores, identity

    monkeypatch.setattr(runtime, "read_scores", observe)
    runspec, deps = fixture(tmp_path)
    session_factory = deps.bioir_session_factory

    class Session(session_factory):
        def close(self):
            assert references and all(ref() is None for ref in references)
            super().close()

    deps = replace(deps, bioir_session_factory=Session)
    run_to_fold(tmp_path, runspec, deps)
    assert len(references) == 1
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize("point", ["score", "aggregate", "index"])
def test_failed_reduction_never_publishes_accepted_index(tmp_path, monkeypatch, point):
    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "baked")
    monkeypatch.setenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", "f" * 40)
    runspec, deps = fixture(tmp_path)
    fold = run_to_fold(tmp_path, runspec, deps)
    canonical = action_root(runspec, _ACTION_IDS["canonical-pair"])
    original_write = runtime.write_metadata

    def fail_selected(path, payload):
        if (point == "aggregate" and path == canonical / "action-evidence.json") or (
            point == "index" and path.name == "finalization-index.json"
        ):
            raise OSError("diagnostic publication failure")
        original_write(path, payload)

    if point == "score":
        score = next(fold.rglob("*-meta_v1.json"))
        score.write_bytes(score.read_bytes() + b" ")
    else:
        monkeypatch.setattr(runtime, "write_metadata", fail_selected)
    with pytest.raises((ValueError, OSError, FoldingExecutorError)):
        run_action(tmp_path, runspec, deps, "canonical-pair")
    assert not (canonical / "finalization-index.json").exists()
    assert all(path.name != "finalization-index.json" for path in canonical.glob("*"))


def test_fsync_failure_cleans_unpublished_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.os, "fsync", lambda *_: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(OSError, match="fsync failed"):
        runtime.write_metadata(tmp_path / "evidence.json", {"value": 1})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("data", [b'{"x":1,"x":2}', b'{"x":NaN}', b"[]", b"{", b'{"x":Infinity}'])
def test_strict_metadata_json_rejects_ambiguous_documents(data):
    for loader in (runtime.json_mapping, control._mapping):
        with pytest.raises(ValueError):
            loader(data)


@pytest.mark.parametrize("layer", ["runtime", "control"])
def test_replaced_path_during_snapshot_is_rejected(tmp_path, monkeypatch, layer):
    path = tmp_path / "file.json"
    path.write_bytes(b'{"value":1}')
    module = runtime if layer == "runtime" else control
    symbol = "_open_regular" if layer == "runtime" else "_open"
    original = getattr(module, symbol)
    calls = 0

    def opener(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b'{"value":2}')
            replacement.replace(path)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, symbol, opener)
    with pytest.raises(ValueError, match="changed"):
        if layer == "runtime":
            runtime.snapshot_file(path, root=tmp_path, maximum_bytes=100)
        else:
            control._snapshot(path, 100)


def _reindex(bundle, canonical, relative, mapping):
    data = json.dumps(mapping, sort_keys=True, indent=2).encode() + b"\n"
    (bundle / relative).write_bytes(data)
    index_path = bundle / canonical.name / "finalization-index.json"
    index = json.loads(index_path.read_bytes())
    for member in index["members"]:
        if member["path"] == relative:
            member.update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    index_path.write_text(json.dumps(index))


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase_run_id", "phase-run-" + "b" * 32),
        ("attempt_id", "attempt-0002"),
        ("phase_runspec_digest", "a" * 64),
        ("action_digest", "a" * 64),
        ("shard_projection_sha256", "a" * 64),
        ("orchestration_source_commit", "b" * 40),
    ],
)
def test_rehashed_foreign_evidence_cannot_escape_cross_bindings(completed, field, value):
    runspec, canonical, bundle = completed
    relative = canonical.name + "/action-evidence.json"
    aggregate = json.loads((bundle / relative).read_bytes())
    aggregate[_ACTION_IDS["canonical-pair"]][field] = value
    _reindex(bundle, canonical, relative, aggregate)
    with pytest.raises(ValueError):
        control.validate_local_folding_bundle(bundle, runspec=runspec)


@pytest.mark.parametrize("ancestor", [False, True])
def test_bundle_rejects_symlink_ancestors_and_leaf(completed, tmp_path, ancestor):
    runspec, canonical, bundle = completed
    if ancestor:
        link = tmp_path / "link"
        link.symlink_to(bundle)
        bundle = link
    else:
        target = bundle / canonical.name / "finalization-index.json"
        original = tmp_path / "original-index"
        target.replace(original)
        target.symlink_to(original)
    with pytest.raises((ValueError, OSError)):
        control.validate_local_folding_bundle(bundle, runspec=runspec)


def test_transfer_rechecks_published_index_before_creating_destination(completed, tmp_path, monkeypatch):
    runspec, _, _ = completed
    monkeypatch.setattr(control.PhaseAuthorityStore, "validate", lambda *_: _authority(runspec))
    calls = []

    class Source:
        def fetch_stable_artifact(self, path, *, remote_root, maximum_bytes):
            calls.append(path)
            data = path.read_bytes()
            return data + b" " if len(calls) == 6 else data

    target = tmp_path / "must-not-publish"
    with pytest.raises(ValueError, match="index changed"):
        control.fetch_folding_finalization_evidence(
            runspec.phase_run_id, authority_root=tmp_path, destination=target, source=Source()
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".must-not-publish*"))


@pytest.mark.parametrize(
    "fault", ["missing_source", "stale_reference", "wrong_source_attempt", "duplicate_adoption", "partial_journals"]
)
def test_carry_failure_preserves_sources_and_never_publishes_index(tmp_path, monkeypatch, fault):
    from tests.support.folding_artifact_fixture import successor_with_carry

    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "baked")
    monkeypatch.setenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", "f" * 40)
    runspec, deps = fixture(tmp_path)
    fold = run_to_fold(tmp_path, runspec, deps)
    successor, successor_deps, carry = successor_with_carry(tmp_path, runspec, deps, fold)
    canonical = action_root(successor, _ACTION_IDS["canonical-pair"])
    if fault == "stale_reference":
        successor = replace(successor, carry_forward=replace(successor.carry_forward, digest="a" * 64))
        successor_deps = replace(successor_deps, load_runspec=lambda _: successor)
        with pytest.raises(FoldingExecutorError, match="sealed RunSpec"):
            run_action(tmp_path, successor, successor_deps, "fold", carry_record_path=carry)
    else:
        for kind in ("msa-flatten", "split", "preprocess"):
            run_action(tmp_path, successor, successor_deps, kind)
        if fault == "missing_source":
            next(fold.rglob("*-meta_v1.json")).unlink()
        elif fault == "partial_journals":
            # One zero-target worker ran, so canonical may not synthesize the missing rank.
            run_action(tmp_path, successor, successor_deps, "fold", rank=1, carry_record_path=carry)
        else:
            for rank in range(2):
                run_action(tmp_path, successor, successor_deps, "fold", rank=rank, carry_record_path=carry)
            adopted = action_root(successor, _ACTION_IDS["fold"]) / "ranks/0/adopted.jsonl"
            original = adopted.read_bytes()
            if fault == "duplicate_adoption":
                adopted.write_bytes(original + original)
            else:
                mapping = json.loads(original)
                mapping["source_attempt_id"] = "attempt-0000"
                adopted.write_text(json.dumps(mapping) + "\n")
        with pytest.raises((ValueError, OSError, FoldingExecutorError)):
            run_action(tmp_path, successor, successor_deps, "canonical-pair", carry_record_path=carry)
    assert not (canonical / "finalization-index.json").exists()


def test_successful_canonical_rerun_is_rejected_without_publication_changes(completed, tmp_path):
    _, canonical, _ = completed
    from bspp.orchestration.runtime.folding.executor import run_execute_action

    snapshot = {p: p.read_bytes() for p in canonical.rglob("*") if p.is_file()}
    with pytest.raises(FoldingExecutorError, match="already completed"):
        run_execute_action(
            phase_runspec_path=tmp_path / "unused.json",
            action_id=canonical.name,
            action_evidence_path=canonical / "action-evidence.json",
            handoff_path=canonical / "handoff.json",
        )
    assert snapshot == {p: p.read_bytes() for p in canonical.rglob("*") if p.is_file()}


def test_canonical_index_byte_digest_cannot_be_replaced_by_semantic_equality(completed):
    runspec, canonical, bundle = completed
    relative = canonical.name + "/canonical-pair-index.json"
    data = (bundle / relative).read_bytes() + b" "
    (bundle / relative).write_bytes(data)
    path = bundle / canonical.name / "finalization-index.json"
    index = json.loads(path.read_bytes())
    for member in index["members"]:
        if member["path"] == relative:
            member.update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index digest"):
        control.validate_local_folding_bundle(bundle, runspec=runspec)


def test_partial_carry_combines_one_native_and_one_adopted_target_on_same_rank(tmp_path, monkeypatch):
    from tests.support.folding_artifact_fixture import add_second_target, successor_with_carry

    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "baked")
    monkeypatch.setenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", "f" * 40)
    runspec, deps = fixture(tmp_path)
    runspec, deps = add_second_target(tmp_path, runspec, deps)
    fold = run_to_fold(tmp_path, runspec, deps)
    successor, successor_deps, carry = successor_with_carry(tmp_path, runspec, deps, fold)
    original = {p: p.read_bytes() for p in fold.rglob("*") if p.is_file()}
    predictions = []
    factory = successor_deps.bioir_session_factory

    class Session(factory):
        def run(self, target, prepared, output_dir):
            predictions.append(target.target_id)
            return super().run(target, prepared, output_dir)

    successor_deps = replace(successor_deps, bioir_session_factory=Session)
    for kind in ("msa-flatten", "split", "preprocess"):
        run_action(tmp_path, successor, successor_deps, kind)
    for rank in range(2):
        run_action(tmp_path, successor, successor_deps, "fold", rank=rank, carry_record_path=carry)
    current_fold = action_root(successor, _ACTION_IDS["fold"])
    before = {p: p.read_bytes() for p in current_fold.rglob("*") if p.is_file()}
    canonical = run_action(tmp_path, successor, successor_deps, "canonical-pair", carry_record_path=carry)
    assert len(predictions) == 1
    native = [json.loads(line) for line in (current_fold / "ranks/0/journal.jsonl").read_bytes().splitlines()]
    adopted = [json.loads(line) for line in (current_fold / "ranks/0/adopted.jsonl").read_bytes().splitlines()]
    assert [row["target_id"] for row in native] == predictions
    assert len(adopted) == 1 and adopted[0]["target_id"] != predictions[0]
    assert (current_fold / "ranks/1/journal.jsonl").read_bytes() == b""
    assert all(p.read_bytes() == data for p, data in before.items())
    assert all(p.read_bytes() == data for p, data in original.items())
    evidence = json.loads((canonical / "action-evidence.json").read_bytes())
    entries = ArtifactFoldEvidence.from_mapping(evidence[_ACTION_IDS["fold"]]).entries
    assert {entry.target.target_id for entry in entries} == {predictions[0], adopted[0]["target_id"]}
    assert all("/attempt-0002/" in entry.scores.path for entry in entries)


def test_metadata_serializer_preserves_exact_existing_json_bytes(tmp_path):
    mapping = {"nested": [{"float": -0.0, "exponent": 1e-20, "unicode": "\u03b1", "escapes": '\n"\\'}], "integer": 1}
    expected = (json.dumps(mapping, indent=2, sort_keys=True) + "\n").encode()
    assert b"".join(runtime.iter_metadata_bytes(mapping)) == expected
    target = tmp_path / "metadata.json"
    runtime.write_metadata(target, mapping)
    assert target.read_bytes() == expected
    assert target.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize(
    "fault", ["duplicate_member", "extra_member", "missing_member", "oversize_file", "oversize_bundle", "wrong_attempt"]
)
def test_remote_index_is_rejected_before_fetching_members(completed, tmp_path, monkeypatch, fault):
    runspec, canonical, _ = completed
    monkeypatch.setattr(control.PhaseAuthorityStore, "validate", lambda *_: _authority(runspec))
    mapping = json.loads((canonical / "finalization-index.json").read_bytes())
    if fault == "duplicate_member":
        mapping["members"][1] = mapping["members"][0]
    elif fault == "extra_member":
        mapping["members"].append(mapping["members"][0])
    elif fault == "missing_member":
        mapping["members"].pop()
    elif fault == "oversize_file":
        mapping["members"][0]["size_bytes"] = 134217729
    elif fault == "oversize_bundle":
        for member in mapping["members"]:
            member["size_bytes"] = 134217728
    else:
        mapping["attempt_id"] = "attempt-0002"
    calls = []

    class Source:
        def fetch_stable_artifact(self, path, *, remote_root, maximum_bytes):
            calls.append(path)
            assert len(calls) == 1
            return json.dumps(mapping).encode()

    destination = tmp_path / "not-published"
    with pytest.raises(ValueError):
        control.fetch_folding_finalization_evidence(
            runspec.phase_run_id, authority_root=tmp_path, destination=destination, source=Source()
        )
    assert len(calls) == 1
    assert not destination.exists()


def test_retry_prerequisites_match_renderer_without_unmounted_carry_record(tmp_path, monkeypatch):
    from tests.support.folding_artifact_fixture import successor_with_carry

    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "baked")
    monkeypatch.setenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", "f" * 40)
    runspec, deps = fixture(tmp_path)
    fold = run_to_fold(tmp_path, runspec, deps)
    successor, successor_deps, carry = successor_with_carry(tmp_path, runspec, deps, fold)
    for kind in ("msa-flatten", "split", "preprocess"):
        # Public renderer intentionally neither stages nor mounts carry here.
        run_action(tmp_path, successor, successor_deps, kind)
    for kind in ("fold", "canonical-pair"):
        with pytest.raises(FoldingExecutorError, match="requires --carry-record"):
            run_action(tmp_path, successor, successor_deps, kind)
    canonical = run_action(tmp_path, successor, successor_deps, "canonical-pair", carry_record_path=carry)
    assert (canonical / "finalization-index.json").is_file()
