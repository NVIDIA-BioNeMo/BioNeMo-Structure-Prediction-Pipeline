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

"""Deterministic fixed-decimal rounding helpers.

Port Baseline: afdb_integration_kit/utils/rounding.py (companion AFDB-Integration-Kit
fork). Behavior is identical to the deployed postprocessing rounding: every numeric
value is rounded from its exact Decimal value with ROUND_HALF_EVEN, not from the
binary float, so exact half-cent boundaries are deterministic and order-independent.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

Number = int | float | str | Decimal


def quantize_half_even(value: Number, decimals: int) -> Decimal:
    """Round ``value`` to ``decimals`` places via exact Decimal half-to-even."""
    quantum = Decimal(1).scaleb(-decimals)
    return Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_EVEN)


def round_float(value: Number, decimals: int) -> float:
    """Deterministic drop-in for ``round(float, n)`` returning a float."""
    return float(quantize_half_even(value, decimals))
