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

"""Small fixture evidence exercises JSON memory path and YAML compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bspp.orchestration.contract.folding_evidence import CanonicalPairActionEvidence, FoldActionEvidence
from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.control.phase_finalization import _load_folding_action_evidence
from tests.test_phase_folding_lifecycle import _full_evidence


def test_generated_json_bypasses_yaml_and_preserves_action_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _full_evidence()
    path = tmp_path / "action-evidence.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

    def reject_yaml(_data: object) -> object:
        raise AssertionError("Generated JSON must not allocate a YAML node tree")

    monkeypatch.setattr(yaml, "safe_load", reject_yaml)
    loaded = _load_folding_action_evidence(path)
    assert loaded == evidence
    for action_id, payload in loaded.items():
        assert canonical_mapping_digest(payload) == canonical_mapping_digest(evidence[action_id])
    FoldActionEvidence.from_mapping(loaded["fold-000001"])
    CanonicalPairActionEvidence.from_mapping(loaded["canonical-pair-000001"])


def test_yaml_compatibility_keeps_full_scores_and_digests(tmp_path: Path) -> None:
    evidence = _full_evidence()
    path = tmp_path / "action-evidence.json"  # Historically YAML was also accepted at this extension.
    path.write_text(yaml.safe_dump(evidence, sort_keys=True))
    loaded = _load_folding_action_evidence(path)
    assert loaded == evidence
    assert canonical_mapping_digest(loaded) == canonical_mapping_digest(evidence)


def test_yaml_fallback_uses_the_same_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "action-evidence.yaml"
    original = b"fold-000001:\n  pairs: []\n"
    path.write_bytes(original)

    def reject_json_after_replacement(data: bytes) -> object:
        assert data == original
        path.write_bytes(b"replaced: {}\n")
        raise json.JSONDecodeError("unit fixture YAML", data.decode(), 0)

    monkeypatch.setattr(json, "loads", reject_json_after_replacement)
    assert _load_folding_action_evidence(path) == {"fold-000001": {"pairs": []}}


@pytest.mark.parametrize("data", [b"[]", b"null", b"{}", b'{"fold": []}', b"fold: [", b"\xff"])
def test_malformed_or_nonmapping_evidence_still_rejects(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "action-evidence.json"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="folding action evidence"):
        _load_folding_action_evidence(path)


@pytest.mark.parametrize("mutation", ["missing-scores", "nonfinite-scores"])
def test_native_json_keeps_strict_score_contract_checks(tmp_path: Path, mutation: str) -> None:
    evidence = _full_evidence()
    pair = evidence["fold-000001"]["pairs"][0]
    if mutation == "missing-scores":
        del pair["scores"]
    else:
        pair["scores"]["pae"][0][0] = float("nan")
    path = tmp_path / "action-evidence.json"
    path.write_text(json.dumps(evidence))
    loaded = _load_folding_action_evidence(path)
    with pytest.raises(ValueError):
        FoldActionEvidence.from_mapping(loaded["fold-000001"])
