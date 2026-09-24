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

"""Shared fixtures for the slurm test package.

submitit's SlurmExecutor raises at construction when ``srun`` isn't on
PATH — a safety check that correctly prevents silent LocalExecutor
fallback in production, but blocks every test that needs to construct
a SlurmExecutor on a host without SLURM installed. We neutralize it
via the classmethod ``affinity()`` which returns 0 when ``srun`` is
missing; forcing it to a positive int satisfies the guard without
actually using srun.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fake_slurm_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    from submitit.slurm.slurm import SlurmExecutor

    monkeypatch.setattr(SlurmExecutor, "affinity", classmethod(lambda cls: 2))
