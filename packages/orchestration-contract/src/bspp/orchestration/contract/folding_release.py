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

"""BioIR release preset contract: public package source resolution.

This module defines the named release preset that controls WHERE BioIR comes
from — not how it folds.  The preset is carried through the folding cluster
snapshot into the qualification tuple identity so that presets are never
conflated in qualification or replay.

The preset resolves to the public PyPI ``bionemo-ir==0.1.0`` package from the
public Docker Hub base ``nvidia/cuda:13.0.3-devel-ubuntu24.04``.  The enum
remains for qualification identity compatibility.

The pure resolver (``resolve_folding_release_preset``) is a documentation-only
record of the build-time facts associated with each preset.  It is NOT wired
into any production build path; the ADR records these values as operator
guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import StrictStr, field_validator

from bspp.orchestration.contract.config_models import FrozenConfigModel
from bspp.orchestration.contract.phase import FOLDING_BACKENDS


class FoldingReleasePreset(StrEnum):
    """Named BioIR release preset.

    ``PUBLIC`` resolves to the public PyPI ``bionemo-ir`` package
    (per-backend kernel images).
    """

    PUBLIC = "public"


@dataclass(frozen=True)
class FoldingReleaseResolution:
    """Pure resolution of one BioIR release preset — no GPU, no Docker.

    This dataclass is a documentation-only record of the build-time facts
    associated with each preset. It is NOT wired into any production build path;
    the ADR records these values as operator guidance.
    """

    preset: FoldingReleasePreset
    package_source: str
    package_spec: str
    base_image: str
    checkpoint_source: str
    checkpoint_fetch_command: str
    public_repo_url: str
    docs_url: str


def resolve_folding_release_preset(preset: FoldingReleasePreset) -> FoldingReleaseResolution:
    """Pure function resolving a preset enum to its documented values.

    The input is a :class:`FoldingReleasePreset` enum, which already rejects
    unknown values at construction time. This function never raises for a valid
    enum member.
    """
    # FoldingReleasePreset.PUBLIC is the sole enum member
    return FoldingReleaseResolution(
        preset=preset,
        package_source="pypi",
        package_spec="bionemo-ir==0.1.0",
        base_image="nvidia/cuda:13.0.3-devel-ubuntu24.04",
        checkpoint_source="fetch-weights-script",
        # NOTE: This is a documentation template, NOT an executable command.
        # The <name> placeholder is parameterized by the model source at build time.
        checkpoint_fetch_command="scripts/fetch_weights.sh --model <name>",
        public_repo_url="https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime",
        docs_url="https://docs.nvidia.com/bionemo/inference-runtime/overview/",
    )


class FoldingBackendImageOverride(FrozenConfigModel):
    """One per-backend kernel image override entry in a Cluster Profile."""

    backend: StrictStr
    image: StrictStr

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        if value not in FOLDING_BACKENDS:
            raise ValueError(f"backend must be one of {FOLDING_BACKENDS!r}")
        return value

    @field_validator("image")
    @classmethod
    def _validate_image_nonempty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("image must be a non-empty string")
        return value


__all__ = [
    "FoldingBackendImageOverride",
    "FoldingReleasePreset",
    "FoldingReleaseResolution",
    "resolve_folding_release_preset",
]
