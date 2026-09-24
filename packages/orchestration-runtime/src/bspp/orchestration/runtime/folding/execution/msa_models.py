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

"""Track-A MSA value objects (harvested from the reference pipeline's models.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ChainAlignment", "MSAResult"]


@dataclass(frozen=True)
class ChainAlignment:
    chain_index: int
    query_sequence: str
    alignments: dict[str, Path]
    sequence_counts: dict[str, int]


@dataclass(frozen=True)
class MSAResult:
    backend: str
    chains: tuple[ChainAlignment, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
