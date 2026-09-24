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

"""Small shared validation rules for scheduler-facing contract values."""

from __future__ import annotations

import re

_EXACT_NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def validate_exact_slurm_node(value: str | None, *, field_name: str) -> str | None:
    """Accept one exact node token, never a Slurm expression or shell text."""
    if value is not None and _EXACT_NODE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must select one safe exact node name")
    return value


__all__ = ["validate_exact_slurm_node"]
