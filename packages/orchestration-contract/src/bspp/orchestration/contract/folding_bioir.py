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

"""Explicit BioIR model routing and checkpoint content identity.

This opt-in policy is scientific intent. Checkpoint paths remain operational
assets; their authenticated contents and the supported model choices live here.
Legacy folding plans omit the policy and retain their multimer-only behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

BIOIR_MONOMER_TOOL_USED = "OpenFold2 (BioNeMo IR) / OpenFold-pTM"
BIOIR_MULTIMER_TOOL_USED = "OpenFold2 (BioNeMo IR) / AlphaFold-Multimer"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "monomer_model_source",
        "multimer_model_source",
        "monomer_checkpoint_sha256",
        "monomer_checkpoint_size_bytes",
        "multimer_checkpoint_sha256",
        "multimer_checkpoint_size_bytes",
    }
)


@dataclass(frozen=True)
class BioIRModelPolicy:
    """Version-one routing by actual expanded chain count, with fixed presets."""

    monomer_checkpoint_sha256: str
    monomer_checkpoint_size_bytes: int
    multimer_checkpoint_sha256: str
    multimer_checkpoint_size_bytes: int
    policy: str = "expanded-chain-count-v1"
    monomer_model_source: str = "openfold2_ptm_1"
    multimer_model_source: str = "alphafold2_multimer_1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("BioIRModelPolicy schema_version must be 1")
        if self.policy != "expanded-chain-count-v1":
            raise ValueError("unsupported BioIRModelPolicy policy")
        if self.monomer_model_source != "openfold2_ptm_1":
            raise ValueError("BioIRModelPolicy requires monomer model source openfold2_ptm_1")
        if self.multimer_model_source != "alphafold2_multimer_1":
            raise ValueError("BioIRModelPolicy requires multimer model source alphafold2_multimer_1")
        for name in ("monomer_checkpoint_sha256", "multimer_checkpoint_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"BioIRModelPolicy {name} must be a lowercase SHA-256")
        for name in ("monomer_checkpoint_size_bytes", "multimer_checkpoint_size_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"BioIRModelPolicy {name} must be a positive integer")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "monomer_model_source": self.monomer_model_source,
            "multimer_model_source": self.multimer_model_source,
            "monomer_checkpoint_sha256": self.monomer_checkpoint_sha256,
            "monomer_checkpoint_size_bytes": self.monomer_checkpoint_size_bytes,
            "multimer_checkpoint_sha256": self.multimer_checkpoint_sha256,
            "multimer_checkpoint_size_bytes": self.multimer_checkpoint_size_bytes,
        }

    @property
    def digest(self) -> str:
        raw = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def model_source_for_chain_count(self, expanded_count: int) -> str:
        """Select by expanded chains, so one repeated polymer is a multimer."""
        if type(expanded_count) is not int or expanded_count <= 0:
            raise ValueError("BioIR expanded chain count must be a positive integer")
        return self.monomer_model_source if expanded_count == 1 else self.multimer_model_source


def bioir_model_policy_from_mapping(payload: Mapping[str, object]) -> BioIRModelPolicy:
    """Load all versioned fields explicitly, rejecting unknown or missing keys."""
    if set(payload) != _FIELDS:
        missing = sorted(_FIELDS - set(payload))
        unknown = sorted(set(payload) - _FIELDS)
        raise ValueError(f"BioIRModelPolicy fields differ: missing={missing}, unknown={unknown}")
    strings: dict[str, str] = {}
    for name in (
        "policy",
        "monomer_model_source",
        "multimer_model_source",
        "monomer_checkpoint_sha256",
        "multimer_checkpoint_sha256",
    ):
        value = payload[name]
        if not isinstance(value, str):
            raise ValueError(f"BioIRModelPolicy {name} must be a string")
        strings[name] = value
    integers: dict[str, int] = {}
    for name in ("schema_version", "monomer_checkpoint_size_bytes", "multimer_checkpoint_size_bytes"):
        value = payload[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"BioIRModelPolicy {name} must be an integer")
        integers[name] = value
    return BioIRModelPolicy(
        schema_version=integers["schema_version"],
        policy=strings["policy"],
        monomer_model_source=strings["monomer_model_source"],
        multimer_model_source=strings["multimer_model_source"],
        monomer_checkpoint_sha256=strings["monomer_checkpoint_sha256"],
        monomer_checkpoint_size_bytes=integers["monomer_checkpoint_size_bytes"],
        multimer_checkpoint_sha256=strings["multimer_checkpoint_sha256"],
        multimer_checkpoint_size_bytes=integers["multimer_checkpoint_size_bytes"],
    )


__all__ = [
    "BIOIR_MONOMER_TOOL_USED",
    "BIOIR_MULTIMER_TOOL_USED",
    "BioIRModelPolicy",
    "bioir_model_policy_from_mapping",
]
