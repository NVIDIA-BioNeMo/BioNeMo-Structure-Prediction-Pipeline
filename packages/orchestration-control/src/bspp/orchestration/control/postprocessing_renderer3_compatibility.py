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

"""Hash-proven replay for historical renderer-contract-3 submissions."""

from __future__ import annotations

import hashlib
import pwd
import re
from collections.abc import Iterable
from contextlib import suppress
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any, cast

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_execution import PostprocessingCredentialMountSnapshot
from bspp.orchestration.contract.postprocessing_submission_events import PostprocessingSubmissionIntendedPayload
from bspp.orchestration.control.postprocessing_phase_rendering import PostprocessingRenderInput
from bspp.orchestration.control.postprocessing_phase_rendering_v3_contract3 import (
    _aws_profile_name,
    _render_postprocessing_action_script_with_credential_mounts,
)
from bspp.orchestration.control.postprocessing_phase_types import PostprocessingAuthority

MAX_RENDERER3_HOME_CANDIDATES = 3
_SAFE_OWNER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")


def render_authenticated_renderer3_scripts(
    authority: PostprocessingAuthority,
    submission: PostprocessingSubmissionIntendedPayload,
    *,
    candidate_homes: Iterable[Path] | None = None,
) -> dict[str, str]:
    """Return contract-3 scripts only after the complete stored hash proof matches."""
    render_input = PostprocessingRenderInput(runspec=authority.runspec, legacy_runspec=authority.legacy_runspec)
    if _aws_profile_name(cast(Any, render_input)) is None:
        scripts = _render_scripts(
            authority,
            PostprocessingCredentialMountSnapshot(
                aws_shared_credentials_file=None,
                aws_config_file=None,
            ),
        )
        if not _submission_matches(authority, submission, scripts):
            raise ValueError("postprocessing renderer-contract-3 submission failed its complete script hash proof")
        return scripts

    homes = _bounded_candidate_homes(authority.runspec.cluster.owner, candidate_homes=candidate_homes)
    matched: list[dict[str, str]] = []
    for home in homes:
        scripts = _render_scripts(
            authority,
            PostprocessingCredentialMountSnapshot(
                aws_shared_credentials_file=str(home / ".aws" / "credentials"),
                aws_config_file=str(home / ".aws" / "config"),
            ),
        )
        if _submission_matches(authority, submission, scripts):
            matched.append(scripts)
    if len(matched) != 1:
        qualifier = "no" if not matched else "multiple"
        raise ValueError(
            f"postprocessing renderer-contract-3 compatibility found {qualifier} complete script hash proof"
        )
    return matched[0]


def _render_scripts(
    authority: PostprocessingAuthority,
    credential_mounts: PostprocessingCredentialMountSnapshot,
) -> dict[str, str]:
    render_input = PostprocessingRenderInput(runspec=authority.runspec, legacy_runspec=authority.legacy_runspec)
    return {
        action.action_id: _render_postprocessing_action_script_with_credential_mounts(
            cast(Any, render_input),
            action,
            credential_mounts,
        )
        for action in authority.runspec.payload.actions
    }


def _submission_matches(
    authority: PostprocessingAuthority,
    submission: PostprocessingSubmissionIntendedPayload,
    scripts: dict[str, str],
) -> bool:
    plans = {plan.action_id: plan for plan in submission.actions}
    if tuple(plans) != tuple(action.action_id for action in authority.runspec.payload.actions):
        return False
    if any(
        plans[action.action_id].renderer_contract_version != 3
        or plans[action.action_id].script_sha256 != hashlib.sha256(scripts[action.action_id].encode()).hexdigest()
        for action in authority.runspec.payload.actions
    ):
        return False
    expected_submission_id = "postprocessing-submission-" + canonical_mapping_digest(
        {
            "schema_version": 1,
            "phase_run_id": authority.phase_run_id,
            "attempt_id": authority.attempt_id,
            "phase_runspec_digest": authority.runspec.digest,
            "actions": [
                {
                    "action_id": action.action_id,
                    "script_sha256": hashlib.sha256(scripts[action.action_id].encode()).hexdigest(),
                    "renderer_contract_version": 3,
                }
                for action in authority.runspec.payload.actions
            ],
        }
    )
    return submission.submission_id == expected_submission_id


def _bounded_candidate_homes(owner: str, *, candidate_homes: Iterable[Path] | None) -> tuple[Path, ...]:
    raw: Iterable[Path]
    if candidate_homes is not None:
        raw = candidate_homes
    else:
        defaults = [Path.home()]
        with suppress(KeyError, OSError):
            defaults.append(Path(pwd.getpwnam(owner).pw_dir))
        if _SAFE_OWNER.fullmatch(owner) is not None:
            defaults.append(Path("/home") / owner)
        raw = defaults
    result: list[Path] = []
    for home in islice(raw, MAX_RENDERER3_HOME_CANDIDATES):
        lexical = PurePosixPath(str(home))
        if (
            not lexical.is_absolute()
            or lexical.anchor != "/"
            or ".." in lexical.parts
            or str(lexical) != str(home)
            or any(character in str(home) for character in ("\x00", "\n", "\r", ",", ":"))
        ):
            continue
        canonical = Path(str(lexical))
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise ValueError("postprocessing renderer-contract-3 compatibility has no safe home candidates")
    return tuple(result)


__all__ = ["MAX_RENDERER3_HOME_CANDIDATES", "render_authenticated_renderer3_scripts"]
