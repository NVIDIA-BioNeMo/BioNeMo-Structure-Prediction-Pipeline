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

"""Exact-format and publication regressions using small synthetic mappings."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from bspp.orchestration.runtime.folding.executor import _atomic_write_json


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"z": [None, True, False, 17], "a": {"text": '\u03b1\n"\\\t', "tuple": (1, 2)}},
        {"floats": [-0.0, 0.0, 1e-25, 1e25, 1.2345678901234567, float("inf"), float("nan")]},
        {"scores": {"pae": [[0.01, 27.15], [3.0, 0.0]], "plddt": [84.52, 90.0], "max_pae": 27.15}},
    ],
)
def test_streamed_writer_preserves_historical_bytes_and_digest(tmp_path: Path, payload: dict[str, object]) -> None:
    expected = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = tmp_path / "nested" / "evidence.json"
    _atomic_write_json(destination, payload)
    actual = destination.read_bytes()
    assert actual == expected
    assert hashlib.sha256(actual).hexdigest() == hashlib.sha256(expected).hexdigest()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("existing", [False, True])
def test_partial_encoding_failure_never_publishes_or_leaks_temporary(tmp_path: Path, existing: bool) -> None:
    destination = tmp_path / "evidence.json"
    original = b"previous accepted bytes\n"
    if existing:
        destination.write_bytes(original)
        destination.chmod(0o600)
    # Sorted key order writes more than a buffer before reaching the bad value.
    with pytest.raises(TypeError, match="not JSON serializable"):
        _atomic_write_json(destination, {"a": list(range(10000)), "z": object()})
    if existing:
        assert destination.read_bytes() == original
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    else:
        assert not destination.exists()
    assert list(tmp_path.iterdir()) == ([destination] if existing else [])


def test_sync_failure_preserves_previous_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "evidence.json"
    destination.write_bytes(b"previous bytes\n")

    def fail_sync(_descriptor: int) -> None:
        raise OSError("diagnostic fsync failure")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="diagnostic fsync failure"):
        _atomic_write_json(destination, {"replacement": [1, 2, 3]})
    assert destination.read_bytes() == b"previous bytes\n"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize(
    "name",
    ["expected-msa-flatten-evidence.json", "expected-split-evidence.json", "expected-preprocess-evidence.json"],
)
def test_existing_evidence_fixture_values_keep_the_original_wire_format(tmp_path: Path, name: str) -> None:
    fixture = Path(__file__).parent.parent / "fixtures" / "folding" / "executable" / name
    payload = json.loads(fixture.read_bytes())
    expected = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = tmp_path / name
    _atomic_write_json(destination, payload)
    assert destination.read_bytes() == expected
