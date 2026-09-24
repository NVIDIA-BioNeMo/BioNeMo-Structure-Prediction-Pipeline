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

"""Tests for shared SLURM array parsing."""

from __future__ import annotations

import pytest

from bspp.orchestration.runtime.slurm.arrays import array_task_count, parse_array_spec, render_zero_based_array


def test_parse_array_spec_accepts_throttled_range() -> None:
    spec = parse_array_spec("0-37%25")

    assert spec.task_count == 38
    assert spec.parallelism == 25
    assert spec.render() == "0-37%25"


def test_parse_array_spec_accepts_comma_ranges() -> None:
    assert parse_array_spec("0,2-4,8").task_count == 5
    assert array_task_count("0-3,8-9%2") == 6
    assert parse_array_spec("0-3,8-9%2").render() == "0-3,8-9%2"


def test_render_zero_based_array_accepts_throttle() -> None:
    assert render_zero_based_array(38, throttle=25) == "0-37%25"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "37-0",
        "0-3%0",
        "0-3%%2",
        "0,,2",
        "-1-3",
        "abc",
        "0-3:2",
    ],
)
def test_parse_array_spec_rejects_invalid_ranges(value: str) -> None:
    with pytest.raises(ValueError):
        parse_array_spec(value)
