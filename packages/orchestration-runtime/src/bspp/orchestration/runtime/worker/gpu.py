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

"""Optional GPU keepalive policies for the native worker."""

from __future__ import annotations

import importlib
import multiprocessing
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, cast


class KeepaliveProcess(Protocol):
    """Small process protocol used by the keepalive context."""

    def start(self) -> None: ...
    def terminate(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...


@dataclass(frozen=True, slots=True)
class GpuKeepalivePolicy:
    """Configuration for upload-time GPU keepalive.

    The native side-effect layer keeps this disabled by default so unit tests
    and CPU-only runs never import torch or allocate a GPU.
    """

    enabled: bool = False
    join_timeout_seconds: float = 5.0


@contextmanager
def gpu_keepalive(
    policy: GpuKeepalivePolicy | None = None,
    *,
    process_factory: Callable[[], KeepaliveProcess] | None = None,
) -> Iterator[KeepaliveProcess | None]:
    """Optionally hold a GPU allocation active for a non-GPU section."""

    active_policy = policy or GpuKeepalivePolicy()
    if not active_policy.enabled:
        yield None
        return

    process = (process_factory or _default_process_factory)()
    process.start()
    try:
        yield process
    finally:
        process.terminate()
        process.join(timeout=active_policy.join_timeout_seconds)


def _default_process_factory() -> KeepaliveProcess:
    return multiprocessing.Process(target=_gpu_hold_loop, daemon=True)


def _gpu_hold_loop() -> None:
    try:
        torch = cast(Any, importlib.import_module("torch"))
    except ImportError:
        return
    try:
        device = torch.device("cuda:0")
        total_mem = torch.cuda.get_device_properties(device).total_mem
        buffer = torch.empty(int(total_mem * 0.8) // 4, dtype=torch.float32, device=device)
        tick = torch.ones(1, device=device)
    except Exception:
        return
    while True:
        tick.add_(1)
        torch.cuda.synchronize(device)
        time.sleep(5)
        # Keep the allocation referenced for the lifetime of the loop.
        _ = buffer


__all__ = [
    "GpuKeepalivePolicy",
    "KeepaliveProcess",
    "gpu_keepalive",
]
