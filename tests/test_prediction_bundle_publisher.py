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

"""Tests for the prediction-bundle publisher (acceptance criterion #24).

Verifies ``publish_prediction_bundles_to_s3`` and
``PredictionBundleUploadEvidence`` including the ``operator_attested`` field.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bspp.orchestration.contract.prediction_bundle import PredictionArchiveBundle
from bspp.orchestration.runtime.data_movement.common import TransferResult
from bspp.orchestration.runtime.folding.seam_transport import (
    PredictionBundleUploadEvidence,
    SeamTransportError,
    publish_prediction_bundles_to_s3,
)

_BUNDLE_NAME = "bspp_260901_1200_a00001.tar.lz4"
_SHA256 = "a" * 64
_SIZE = 64


def _make_bundle() -> PredictionArchiveBundle:
    return PredictionArchiveBundle(
        bundle_name=_BUNDLE_NAME,
        member_ids=("AF-0000000000000001",),
        member_count=1,
        sha256=_SHA256,
        size_bytes=_SIZE,
        created_at="2026-09-01T12:00:00.000000Z",
    )


def _ok_transfer(src: str, dst: str) -> TransferResult:
    return TransferResult(tool="s5cmd", argv=("s5cmd", "cp", src, dst), returncode=0, elapsed_s=0.0)


def _fail_transfer(src: str, dst: str) -> TransferResult:
    return TransferResult(
        tool="s5cmd",
        argv=("s5cmd", "cp", src, dst),
        returncode=1,
        elapsed_s=0.0,
        stderr_tail="boom",
    )


def test_prediction_bundle_upload_evidence_has_operator_attested() -> None:
    evidence = PredictionBundleUploadEvidence(
        bundle_name=_BUNDLE_NAME,
        object_key="s3://bucket/predictions/" + _SHA256 + "/" + _BUNDLE_NAME,
        size_bytes=_SIZE,
        sha256=_SHA256,
        member_count=1,
        operator_attested=True,
        transfer_result=_ok_transfer("src", "dst"),
    )
    mapping = evidence.to_mapping()
    assert mapping["operator_attested"] is True
    assert mapping["bundle_name"] == _BUNDLE_NAME
    assert mapping["object_key"] == "s3://bucket/predictions/" + _SHA256 + "/" + _BUNDLE_NAME
    assert mapping["sha256"] == _SHA256
    assert mapping["size_bytes"] == _SIZE
    assert mapping["member_count"] == 1
    assert mapping["returncode"] == 0


def test_publish_prediction_bundles_with_mocked_boundary(tmp_path: Path) -> None:
    bundle = _make_bundle()
    bundle_path = tmp_path / _BUNDLE_NAME
    bundle_path.write_bytes(b"x" * _SIZE)
    # Fix the sha256 to match the actual content
    actual_sha = hashlib.sha256(b"x" * _SIZE).hexdigest()
    bundle = PredictionArchiveBundle(
        bundle_name=_BUNDLE_NAME,
        member_ids=("AF-0000000000000001",),
        member_count=1,
        sha256=actual_sha,
        size_bytes=_SIZE,
        created_at="2026-09-01T12:00:00.000000Z",
    )
    calls: list[tuple[str, str]] = []

    def recording_transfer(src: str, dst: str) -> TransferResult:
        calls.append((src, dst))
        return _ok_transfer(src, dst)

    all_evidence = publish_prediction_bundles_to_s3(
        bundles=(bundle,),
        local_paths=(str(bundle_path),),
        s3_prefix="s3://bucket/predictions",
        transfer=recording_transfer,
    )
    assert len(all_evidence) == 1
    evidence = all_evidence[0]
    assert evidence.operator_attested is True
    assert evidence.bundle_name == _BUNDLE_NAME
    assert evidence.object_key == f"s3://bucket/predictions/{actual_sha}/{_BUNDLE_NAME}"
    assert evidence.sha256 == actual_sha
    assert evidence.size_bytes == _SIZE
    assert evidence.member_count == 1
    assert len(calls) == 1
    assert calls[0] == (str(bundle_path), evidence.object_key)


def test_publish_prediction_bundles_fails_closed_on_transfer_error(tmp_path: Path) -> None:
    bundle = _make_bundle()
    bundle_path = tmp_path / _BUNDLE_NAME
    bundle_path.write_bytes(b"x" * _SIZE)
    actual_sha = hashlib.sha256(b"x" * _SIZE).hexdigest()
    bundle = PredictionArchiveBundle(
        bundle_name=_BUNDLE_NAME,
        member_ids=("AF-0000000000000001",),
        member_count=1,
        sha256=actual_sha,
        size_bytes=_SIZE,
        created_at="2026-09-01T12:00:00.000000Z",
    )
    with pytest.raises(SeamTransportError):
        publish_prediction_bundles_to_s3(
            bundles=(bundle,),
            local_paths=(str(bundle_path),),
            s3_prefix="s3://bucket/predictions",
            transfer=_fail_transfer,
        )


def test_publish_prediction_bundles_fails_closed_on_bad_prefix(tmp_path: Path) -> None:
    bundle = _make_bundle()
    bundle_path = tmp_path / _BUNDLE_NAME
    bundle_path.write_bytes(b"x" * _SIZE)
    with pytest.raises(SeamTransportError, match="s3_prefix must be"):
        publish_prediction_bundles_to_s3(
            bundles=(bundle,),
            local_paths=(str(bundle_path),),
            s3_prefix="not-an-s3-prefix",
            transfer=_ok_transfer,
        )


def test_publish_prediction_bundles_fails_closed_on_missing_bundle(tmp_path: Path) -> None:
    bundle = _make_bundle()
    bundle_path = tmp_path / _BUNDLE_NAME
    # Don't create the file
    with pytest.raises(SeamTransportError, match="missing or not a regular file"):
        publish_prediction_bundles_to_s3(
            bundles=(bundle,),
            local_paths=(str(bundle_path),),
            s3_prefix="s3://bucket/predictions",
            transfer=_ok_transfer,
        )


def test_publish_prediction_bundles_fails_closed_on_size_mismatch(tmp_path: Path) -> None:
    bundle = _make_bundle()
    bundle_path = tmp_path / _BUNDLE_NAME
    bundle_path.write_bytes(b"z" * (_SIZE + 1))
    with pytest.raises(SeamTransportError, match=r"size.*does not match"):
        publish_prediction_bundles_to_s3(
            bundles=(bundle,),
            local_paths=(str(bundle_path),),
            s3_prefix="s3://bucket/predictions",
            transfer=_ok_transfer,
        )


def test_publish_prediction_bundles_fails_closed_on_content_mismatch(tmp_path: Path) -> None:
    bundle = _make_bundle()
    bundle_path = tmp_path / _BUNDLE_NAME
    bundle_path.write_bytes(b"z" * _SIZE)
    with pytest.raises(SeamTransportError, match="refusing to publish stale bytes"):
        publish_prediction_bundles_to_s3(
            bundles=(bundle,),
            local_paths=(str(bundle_path),),
            s3_prefix="s3://bucket/predictions",
            transfer=_ok_transfer,
        )


def test_publish_prediction_bundles_rejects_empty() -> None:
    with pytest.raises(SeamTransportError, match="at least one prediction bundle"):
        publish_prediction_bundles_to_s3(
            bundles=(),
            local_paths=(),
            s3_prefix="s3://bucket/predictions",
            transfer=_ok_transfer,
        )


def test_publish_prediction_bundles_rejects_length_mismatch(tmp_path: Path) -> None:
    bundle = _make_bundle()
    bundle_path = tmp_path / _BUNDLE_NAME
    bundle_path.write_bytes(b"x" * _SIZE)
    with pytest.raises(SeamTransportError, match="same length"):
        publish_prediction_bundles_to_s3(
            bundles=(bundle,),
            local_paths=(str(bundle_path), str(bundle_path)),
            s3_prefix="s3://bucket/predictions",
            transfer=_ok_transfer,
        )


def test_publish_prediction_bundles_multi_bundle_flat_return(tmp_path: Path) -> None:
    """Multiple bundles produce one evidence per bundle in a flat tuple."""
    bundle1_bytes = b"bundle1-content"
    bundle1_path = tmp_path / "bspp_260901_1200_a00001.tar.lz4"
    bundle1_path.write_bytes(bundle1_bytes)
    sha1 = hashlib.sha256(bundle1_bytes).hexdigest()
    bundle1 = PredictionArchiveBundle(
        bundle_name="bspp_260901_1200_a00001.tar.lz4",
        member_ids=("AF-0000000000000001",),
        member_count=1,
        sha256=sha1,
        size_bytes=len(bundle1_bytes),
        created_at="2026-09-01T12:00:00.000000Z",
    )
    bundle2_bytes = b"bundle2-content"
    bundle2_path = tmp_path / "bspp_260901_1200_a00002.tar.lz4"
    bundle2_path.write_bytes(bundle2_bytes)
    sha2 = hashlib.sha256(bundle2_bytes).hexdigest()
    bundle2 = PredictionArchiveBundle(
        bundle_name="bspp_260901_1200_a00002.tar.lz4",
        member_ids=("AF-0000000000000002",),
        member_count=1,
        sha256=sha2,
        size_bytes=len(bundle2_bytes),
        created_at="2026-09-01T12:00:00.000000Z",
    )
    calls: list[tuple[str, str]] = []

    def recording_transfer(src: str, dst: str) -> TransferResult:
        calls.append((src, dst))
        return _ok_transfer(src, dst)

    all_evidence = publish_prediction_bundles_to_s3(
        bundles=(bundle1, bundle2),
        local_paths=(str(bundle1_path), str(bundle2_path)),
        s3_prefix="s3://bucket/predictions",
        transfer=recording_transfer,
    )
    # Flat return type: one evidence per bundle
    assert isinstance(all_evidence, tuple)
    assert len(all_evidence) == 2
    assert all(ev.operator_attested is True for ev in all_evidence)
    assert all_evidence[0].bundle_name == "bspp_260901_1200_a00001.tar.lz4"
    assert all_evidence[1].bundle_name == "bspp_260901_1200_a00002.tar.lz4"
    assert all_evidence[0].object_key == f"s3://bucket/predictions/{sha1}/bspp_260901_1200_a00001.tar.lz4"
    assert all_evidence[1].object_key == f"s3://bucket/predictions/{sha2}/bspp_260901_1200_a00002.tar.lz4"
    assert len(calls) == 2
