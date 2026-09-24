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

"""Focused tests for deterministic Decimal rounding helpers."""

from __future__ import annotations

from decimal import Decimal

from bspp.orchestration.runtime.folding.execution.rounding import (
    quantize_half_even,
    round_float,
)


def test_quantize_half_even_boundaries() -> None:
    assert quantize_half_even("1.005", 2) == Decimal("1.00")
    assert quantize_half_even("1.015", 2) == Decimal("1.02")
    assert quantize_half_even("81.585", 2) == Decimal("81.58")
    assert quantize_half_even("81.575", 2) == Decimal("81.58")


def test_round_float_boundaries() -> None:
    assert round_float("81.585", 2) == 81.58
    assert round_float("81.575", 2) == 81.58


def test_round_float_returns_float() -> None:
    assert isinstance(round_float("81.585", 2), float)
    assert isinstance(round_float(Decimal("1.005"), 2), float)


def test_input_type_equivalence() -> None:
    assert quantize_half_even("1.005", 2) == Decimal("1.00")
    assert quantize_half_even(1.005, 2) == Decimal("1.00")
    assert quantize_half_even(Decimal("1.005"), 2) == Decimal("1.00")
    assert round_float("1.015", 2) == round_float(1.015, 2) == round_float(Decimal("1.015"), 2)


def test_integer_input() -> None:
    assert quantize_half_even(2, 2) == Decimal("2.00")
    assert round_float(2, 2) == 2.0
