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

"""Public data-movement backend discovery.

Data-movement backends are discovered through a named distribution entry-point
group (:data:`BACKEND_ENTRY_POINT_GROUP`). The two built-in public backends
(``s5cmd`` and ``gcloud``) resolve directly to their transfer modules; any other
named backend (notably the internal ``dm`` mover) is resolved
lazily through the installed distribution entry points.

When no provider is registered for a requested backend — or when the provider's
entry point fails to load — :class:`BackendUnavailableError` is raised. Callers
must never observe a raw :class:`ImportError` from backend discovery.
"""

from __future__ import annotations

import importlib.metadata
from types import ModuleType
from typing import cast

from bspp.orchestration.runtime.data_movement.gcs import transfer as _gcs_transfer
from bspp.orchestration.runtime.data_movement.s3 import transfer as _s3_transfer

BACKEND_ENTRY_POINT_GROUP = "bspp.orchestration.data_movement.backend"
"""Distribution entry-point group under which data-movement backends register.

A backend entry point's :attr:`~importlib.metadata.EntryPoint.name` is the
backend name accepted by :func:`get_backend`, and its ``load()`` target must
resolve to the backend's transfer module.
"""


class BackendUnavailableError(RuntimeError):
    """Raised when a requested data-movement backend cannot be resolved."""

    def __init__(self, name: str, *, hint: str | None = None) -> None:
        self.name = name
        self.hint = hint
        message = f"data-movement backend {name!r} is not available"
        if hint:
            message = f"{message}. {hint}"
        super().__init__(message)


_BUILTIN_BACKENDS: dict[str, ModuleType] = {
    "s5cmd": _s3_transfer,
    "gcloud": _gcs_transfer,
}


def get_backend(name: str) -> ModuleType:
    """Return the transfer module for the named data-movement backend.

    ``s5cmd`` and ``gcloud`` resolve to the built-in public transfer modules.
    Any other name is resolved through the :data:`BACKEND_ENTRY_POINT_GROUP`
    distribution entry points. Raises :class:`BackendUnavailableError` when no
    provider is registered or when the provider's entry point fails to load.
    """
    builtin = _BUILTIN_BACKENDS.get(name)
    if builtin is not None:
        return builtin

    entry_points = importlib.metadata.entry_points(group=BACKEND_ENTRY_POINT_GROUP)
    matches = entry_points.select(name=name)
    if not matches:
        raise BackendUnavailableError(name)

    entry_point = next(iter(matches))
    try:
        return cast(ModuleType, entry_point.load())
    except ImportError as exc:
        raise BackendUnavailableError(name) from exc
    except Exception as exc:
        raise BackendUnavailableError(name) from exc


__all__ = ["BACKEND_ENTRY_POINT_GROUP", "BackendUnavailableError", "get_backend"]
