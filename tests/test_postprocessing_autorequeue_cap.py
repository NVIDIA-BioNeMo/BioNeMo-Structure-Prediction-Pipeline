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

"""Qualified fixed autorequeue cap tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bspp.orchestration.contract.postprocessing_autorequeue_policy import (
    PostprocessingAutorequeuePolicy,
)
from bspp.orchestration.contract.postprocessing_execution import (
    QualifiedPostprocessingRuntimeSelection,
)
from bspp.orchestration.contract.postprocessing_execution_parsing import (
    qualified_postprocessing_runtime_from_mapping,
)
from bspp.orchestration.control.phase_materialization import materialize_phase
from bspp.orchestration.control.postprocessing_authority_v2 import validate_postprocessing_authority
from bspp.orchestration.control.runtime_qualification_validation import (
    RuntimeQualificationAutorequeueCap,
    validate_promoted_runtime_qualification,
)
from tests.test_postprocessing_phase_materialization import (
    NOW,
    RUN_ID,
    _document,
    _fixture,
)


def _selection(
    requeue_exit: int | None,
    max_batch_requeue: int | None,
) -> QualifiedPostprocessingRuntimeSelection:
    sha = "1" * 64
    revision = "a" * 40
    return QualifiedPostprocessingRuntimeSelection(
        tuple_id=sha,
        qualification_location="/qualification.json",
        qualification_sha256=sha,
        qualification_size_bytes=1,
        qualified_at="2026-09-02T00:00:00Z",
        expires_at="2026-09-11T00:00:00Z",
        image_path="/image.sqsh",
        image_sha256=sha,
        image_size_bytes=1,
        image_policy="digest-checked",
        source_kind="baked",
        source_revision=revision,
        source_package_path="/source.tar",
        toolkit_package_path=None,
        runtime_ipsae_binary_path="/ipsae",
        runtime_ipsae_binary_sha256=sha,
        runtime_ipsae_binary_size_bytes=1,
        source_identity_digest=sha,
        source_package_identity_digest=sha,
        toolkit_identity_digest=sha,
        bootstrap_sha256=sha,
        runtime_component_identity_digest=sha,
        requeue_exit=requeue_exit,
        max_batch_requeue=max_batch_requeue,
    )


@pytest.mark.parametrize("requeue_exit", [86, True])
def test_selection_rejects_requeue_exit_not_85(requeue_exit: object) -> None:
    with pytest.raises(ValueError, match="requeue_exit"):
        _selection(requeue_exit, 5)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_batch_requeue", [-1, True])
def test_selection_rejects_negative_or_bool_max_batch_requeue(max_batch_requeue: object) -> None:
    with pytest.raises(ValueError, match="max_batch_requeue"):
        _selection(85, max_batch_requeue)  # type: ignore[arg-type]


@pytest.mark.parametrize(("requeue_exit", "max_batch_requeue"), [(85, None), (None, 5)])
def test_selection_rejects_half_present_cap(
    requeue_exit: int | None,
    max_batch_requeue: int | None,
) -> None:
    with pytest.raises(ValueError, match="both present or both null"):
        _selection(requeue_exit, max_batch_requeue)


@pytest.mark.parametrize(("requeue_exit", "max_batch_requeue"), [(None, None), (85, 5)])
def test_selection_roundtrips_cap_and_absence(
    requeue_exit: int | None,
    max_batch_requeue: int | None,
) -> None:
    selection = _selection(requeue_exit, max_batch_requeue)
    assert qualified_postprocessing_runtime_from_mapping(selection.to_mapping()) == selection


def test_selection_mapping_omits_cap_when_absent() -> None:
    mapping = _selection(None, None).to_mapping()
    assert "requeue_exit" not in mapping
    assert "max_batch_requeue" not in mapping


def test_selection_mapping_includes_cap_when_present() -> None:
    mapping = _selection(85, 5).to_mapping()
    assert mapping["requeue_exit"] == 85
    assert mapping["max_batch_requeue"] == 5


def _qualification_bytes(tmp_path: Path) -> bytes:
    return (tmp_path / "runtime-qualification.json").read_bytes()


def _write_cap(tmp_path: Path, cap: object) -> None:
    path = tmp_path / "runtime-qualification.json"
    payload = json.loads(path.read_bytes())
    payload["autorequeue_cap"] = cap
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_qualification_validator_accepts_absent_cap(tmp_path: Path) -> None:
    _fixture(tmp_path)
    validated = validate_promoted_runtime_qualification(
        _qualification_bytes(tmp_path),
        profile=None,
        source_repo=None,
        observed_at=None,
    )
    assert validated.autorequeue_cap is None


@pytest.mark.parametrize("max_batch_requeue", [0, 5])
def test_qualification_validator_accepts_valid_cap(tmp_path: Path, max_batch_requeue: int) -> None:
    _fixture(tmp_path)
    _write_cap(tmp_path, {"requeue_exit": 85, "max_batch_requeue": max_batch_requeue})
    validated = validate_promoted_runtime_qualification(
        _qualification_bytes(tmp_path),
        profile=None,
        source_repo=None,
        observed_at=None,
    )
    assert validated.autorequeue_cap == RuntimeQualificationAutorequeueCap(85, max_batch_requeue)


@pytest.mark.parametrize(
    "cap",
    [
        {"requeue_exit": 86, "max_batch_requeue": 5},
        {"requeue_exit": 85.0, "max_batch_requeue": 5},
        {"requeue_exit": 85, "max_batch_requeue": -1},
        {"requeue_exit": 85, "max_batch_requeue": True},
        {"requeue_exit": 85},
        {"requeue_exit": 85, "max_batch_requeue": 5, "extra": 1},
        "not-a-mapping",
    ],
)
def test_qualification_validator_rejects_malformed_cap(tmp_path: Path, cap: object) -> None:
    _fixture(tmp_path)
    _write_cap(tmp_path, cap)
    with pytest.raises(ValueError):
        validate_promoted_runtime_qualification(
            _qualification_bytes(tmp_path),
            profile=None,
            source_repo=None,
            observed_at=None,
        )


def _write_phase_plan_with_policy_and_cap(
    tmp_path: Path,
    policy: dict[str, object] | None,
    cap: object | None,
) -> tuple[Path, Path, Path]:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    if cap is not None:
        _write_cap(tmp_path, cap)
    plan = yaml.safe_load(plan_path.read_text())
    if cap is not None:
        plan["runtime_qualification"] = _document("runtime-qualification", tmp_path / "runtime-qualification.json")
    if policy is not None:
        plan["autorequeue_policy"] = policy
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False))
    return plan_path, profile_path, source_repo


def _enabled_policy() -> dict[str, object]:
    return PostprocessingAutorequeuePolicy(
        mode="enabled",
        action_ids=("postprocessing-01-preflight",),
    ).to_mapping()


def test_materialization_enabled_policy_without_cap_fails_closed(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _write_phase_plan_with_policy_and_cap(
        tmp_path,
        policy=_enabled_policy(),
        cap=None,
    )
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match="autorequeue"):
        materialize_phase(
            plan_path,
            authority_root=authority_root,
            config_path=profile_path,
            source_repo=source_repo,
            clock=lambda: NOW,
            phase_run_id_factory=lambda: RUN_ID,
        )

    assert not authority_root.exists()


def test_materialization_enabled_policy_with_mismatched_cap_fails_closed(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _write_phase_plan_with_policy_and_cap(
        tmp_path,
        policy=_enabled_policy(),
        cap={"requeue_exit": 86, "max_batch_requeue": 5},
    )
    authority_root = tmp_path / "authority"

    with pytest.raises(ValueError, match="requeue_exit"):
        materialize_phase(
            plan_path,
            authority_root=authority_root,
            config_path=profile_path,
            source_repo=source_repo,
            clock=lambda: NOW,
            phase_run_id_factory=lambda: RUN_ID,
        )

    assert not authority_root.exists()


def test_materialization_enabled_policy_with_valid_cap_succeeds(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _write_phase_plan_with_policy_and_cap(
        tmp_path,
        policy=_enabled_policy(),
        cap={"requeue_exit": 85, "max_batch_requeue": 5},
    )
    authority_root = tmp_path / "authority"

    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )

    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    assert authority.runspec.payload.autorequeue_policy.mode == "enabled"
    assert authority.runspec.payload.qualified_runtime.requeue_exit == 85
    assert authority.runspec.payload.qualified_runtime.max_batch_requeue == 5


def test_materialization_disabled_policy_without_cap_succeeds(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    authority_root = tmp_path / "authority"

    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )

    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    assert authority.runspec.payload.autorequeue_policy.mode == "disabled"
    assert authority.runspec.payload.qualified_runtime.requeue_exit is None
    assert authority.runspec.payload.qualified_runtime.max_batch_requeue is None
