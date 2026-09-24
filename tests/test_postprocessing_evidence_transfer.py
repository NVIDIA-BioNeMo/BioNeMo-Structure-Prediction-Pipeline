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

"""Bounded, index-confined postprocessing finalization evidence transfer."""

from __future__ import annotations

import errno
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import bspp.orchestration.control.postprocessing_evidence_transfer as evidence_transfer
from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_finalization_bundle import (
    POSTPROCESSING_FINALIZATION_FIXED_PATHS,
    PostprocessingBundleMemberIdentity,
    PostprocessingFinalizationHandoffIndex,
    PostprocessingTarManifest,
    PostprocessingTarMemberIdentity,
)
from bspp.orchestration.control.postprocessing_evidence_transfer import (
    PostprocessingEvidenceFetchContext,
    PostprocessingEvidencePublicationStore,
    PostprocessingHandoffAuthorityBinding,
    fetch_postprocessing_finalization_evidence,
    validate_local_postprocessing_handoff,
)

RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
ATTEMPT_ID = "attempt-0001"
SHA = "1" * 64


def test_control_publication_falls_back_on_einval_and_rejects_dangling_collision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()

    def unsupported(*_args: object) -> None:
        raise OSError(errno.EINVAL, "forced unsupported capability")

    destination.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        evidence_transfer._rename_directory_no_replace(source, destination, renameat2=unsupported)
    assert source.is_dir()


def test_local_handoff_requires_exact_indexed_nonsymlink_layout(tmp_path: Path) -> None:
    handoff, _ = _bundle(tmp_path)
    index = validate_local_postprocessing_handoff(handoff)
    assert index.declared_tar_manifest_count == 1

    extra = handoff / "extra.json"
    extra.write_text("{}\n")
    with pytest.raises(ValueError, match="missing or extra"):
        validate_local_postprocessing_handoff(handoff)
    extra.unlink()

    target = handoff / "acceptance/adjudication.json"
    target.unlink()
    target.symlink_to(handoff / "acceptance/bundle.json")
    with pytest.raises(ValueError, match=r"unsafe|layout"):
        validate_local_postprocessing_handoff(handoff)


def test_fetch_validates_index_before_descendants_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    source, documents = _bundle(tmp_path / "source")
    destination = tmp_path / "fetched"
    authority = _authority()
    calls: list[str] = []

    class FakeTransport:
        def fetch_stable_artifact(self, remote_path: Path, *, remote_root: Path, maximum_bytes: int) -> bytes:
            relative = remote_path.relative_to(remote_root).as_posix()
            calls.append(relative)
            assert maximum_bytes > 0
            return documents[relative]

    result = fetch_postprocessing_finalization_evidence(
        RUN_ID,
        authority_root=tmp_path / "authority",
        destination=destination,
        context=PostprocessingEvidenceFetchContext(authority=authority, source=FakeTransport()),
    )

    assert result.destination == destination
    assert calls[0] == "handoff-index.json"
    assert calls[1:] == [item.path for item in validate_local_postprocessing_handoff(source).members]
    assert not tuple(tmp_path.glob(".fetched.fetch-*"))
    validate_local_postprocessing_handoff(destination, authority=authority)


def test_fetch_rejects_malicious_index_before_any_descendant_fetch(
    tmp_path: Path,
) -> None:
    _, documents = _bundle(tmp_path / "source")
    payload = json.loads(documents["handoff-index.json"])
    payload["members"][0]["size_bytes"] = 16_777_217
    documents["handoff-index.json"] = _canonical(payload)
    calls: list[str] = []

    class FakeTransport:
        def fetch_stable_artifact(self, remote_path: Path, *, remote_root: Path, maximum_bytes: int) -> bytes:
            relative = remote_path.relative_to(remote_root).as_posix()
            calls.append(relative)
            return documents[relative]

    authority = _authority()
    with pytest.raises(ValueError, match="size"):
        fetch_postprocessing_finalization_evidence(
            RUN_ID,
            authority_root=tmp_path / "authority",
            destination=tmp_path / "fetched",
            context=PostprocessingEvidenceFetchContext(authority=authority, source=FakeTransport()),
        )
    assert calls == ["handoff-index.json"]


def test_fetch_existing_exact_destination_is_idempotent_and_never_uses_transport(
    tmp_path: Path,
) -> None:
    destination, _ = _bundle(tmp_path)
    authority = _authority()

    class ForbiddenTransport:
        def fetch_stable_artifact(self, *_args: object, **_kwargs: object) -> bytes:
            raise AssertionError("transport must not run")

    result = fetch_postprocessing_finalization_evidence(
        RUN_ID,
        authority_root=tmp_path / "authority",
        destination=destination,
        context=PostprocessingEvidenceFetchContext(authority=authority, source=ForbiddenTransport()),
    )
    assert result.indexed_file_count == 15


def test_concurrent_fetch_accepts_only_the_identical_winner(
    tmp_path: Path,
) -> None:
    _, documents = _bundle(tmp_path / "source")
    destination = tmp_path / "fetched"
    authority = _authority()

    class FakeTransport:
        def fetch_stable_artifact(self, remote_path: Path, *, remote_root: Path, maximum_bytes: int) -> bytes:
            assert maximum_bytes > 0
            return documents[remote_path.relative_to(remote_root).as_posix()]

    from bspp.orchestration.control.postprocessing_evidence_transfer import (
        CURRENT_POSTPROCESSING_EVIDENCE_PUBLICATION_STORE,
    )

    real_publish = CURRENT_POSTPROCESSING_EVIDENCE_PUBLICATION_STORE.publish_directory_no_replace
    barrier = threading.Barrier(2)

    def synchronized(source: Path, target: Path) -> None:
        barrier.wait(timeout=5)
        real_publish(source, target)

    publication_store = PostprocessingEvidencePublicationStore(publish_directory_no_replace=synchronized)

    def fetch(_index: int) -> object:
        return fetch_postprocessing_finalization_evidence(
            RUN_ID,
            authority_root=tmp_path / "authority",
            destination=destination,
            context=PostprocessingEvidenceFetchContext(authority=authority, source=FakeTransport()),
            publication_store=publication_store,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(fetch, range(2)))

    assert results[0] == results[1]
    validate_local_postprocessing_handoff(destination, authority=authority)
    assert not tuple(tmp_path.glob(".fetched.fetch-*"))


def _bundle(root: Path) -> tuple[Path, dict[str, bytes]]:
    root.mkdir(parents=True, exist_ok=True)
    tar_member = PostprocessingTarMemberIdentity(path="member.json.zst", sha256="2" * 64, size_bytes=4)
    tar_identity = {
        "schema_version": 1,
        "manifest_kind": "postprocessing-tar-manifest-v1",
        "tar_path": "local_tars/shard_1/batch_0.tar",
        "members": [tar_member.to_mapping()],
    }
    manifest = PostprocessingTarManifest(
        tar_path="local_tars/shard_1/batch_0.tar",
        tar_size_bytes=1024,
        stat_device=1,
        stat_inode=2,
        stat_mtime_ns=3,
        members=(tar_member,),
        manifest_id=canonical_mapping_digest(tar_identity),
    )
    dynamic = f"outputs/tar-manifests/{manifest.manifest_id}.json"
    documents = {
        path: _canonical({"schema_version": 1, "fixture_path": path})
        for path in POSTPROCESSING_FINALIZATION_FIXED_PATHS
    }
    documents[dynamic] = _canonical(manifest.to_mapping())
    identities = tuple(
        PostprocessingBundleMemberIdentity(
            path=path,
            sha256=hashlib.sha256(document).hexdigest(),
            size_bytes=len(document),
        )
        for path, document in sorted(documents.items())
    )
    index = PostprocessingFinalizationHandoffIndex(
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_runspec_digest=SHA,
        action_graph_digest=SHA,
        execution_projection_sha256=SHA,
        acceptance_policy_sha256=SHA,
        members=identities,
        tar_manifest_member_counts=((dynamic, 1),),
        declared_file_count=len(identities),
        declared_aggregate_bytes=sum(len(document) for document in documents.values()),
        declared_tar_manifest_count=1,
        declared_total_tar_members=1,
    )
    documents["handoff-index.json"] = _canonical(index.to_mapping())
    for relative, document in documents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(document)
    return root, documents


def _authority() -> PostprocessingHandoffAuthorityBinding:
    return PostprocessingHandoffAuthorityBinding(
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_runspec_digest=SHA,
        action_graph_digest=SHA,
        execution_projection_sha256=SHA,
        acceptance_policy_sha256=SHA,
        evidence_dir="/lustre/remote/evidence",
    )


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
