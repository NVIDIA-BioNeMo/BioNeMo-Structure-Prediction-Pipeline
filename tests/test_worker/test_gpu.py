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

from __future__ import annotations

from bspp.orchestration.runtime.worker import GpuKeepalivePolicy, gpu_keepalive


class FakeProcess:
    def __init__(self) -> None:
        self.started = False
        self.terminated = False
        self.join_timeout: float | None = None

    def start(self) -> None:
        self.started = True

    def terminate(self) -> None:
        self.terminated = True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout


def test_gpu_keepalive_is_disabled_by_default() -> None:
    created = False

    def factory() -> FakeProcess:
        nonlocal created
        created = True
        return FakeProcess()

    with gpu_keepalive(process_factory=factory) as process:
        assert process is None

    assert created is False


def test_gpu_keepalive_uses_injected_process_when_enabled() -> None:
    fake = FakeProcess()

    with gpu_keepalive(GpuKeepalivePolicy(enabled=True, join_timeout_seconds=1.5), process_factory=lambda: fake):
        assert fake.started is True
        assert fake.terminated is False

    assert fake.terminated is True
    assert fake.join_timeout == 1.5
