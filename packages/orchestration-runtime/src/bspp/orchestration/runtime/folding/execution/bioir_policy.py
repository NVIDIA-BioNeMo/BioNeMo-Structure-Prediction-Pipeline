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

"""Execute an explicitly sealed BioIR chain-count model policy."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from bspp.orchestration.contract.folding_bioir import (
    BIOIR_MONOMER_TOOL_USED,
    BIOIR_MULTIMER_TOOL_USED,
    BioIRModelPolicy,
)
from bspp.orchestration.contract.folding_execution import FoldingBackendAssetsSnapshot

from .errors import FoldingBackendError
from .models import FoldingResult, PreparedInput, ProteinTarget

if TYPE_CHECKING:
    from .bioir_session import BioIRFoldSession


def bioir_model_metadata(policy: BioIRModelPolicy, chain_count: int) -> dict[str, object]:
    """Derive provenance from sealed science and the actual expanded chains."""
    source = policy.model_source_for_chain_count(chain_count)
    monomer = source == policy.monomer_model_source
    return {
        "tool_used": BIOIR_MONOMER_TOOL_USED if monomer else BIOIR_MULTIMER_TOOL_USED,
        "model_source": source,
        "checkpoint_sha256": policy.monomer_checkpoint_sha256 if monomer else policy.multimer_checkpoint_sha256,
        "checkpoint_size_bytes": policy.monomer_checkpoint_size_bytes
        if monomer
        else policy.multimer_checkpoint_size_bytes,
        "bioir_model_policy_digest": policy.digest,
    }


def _verify_checkpoint(path: Path, *, sha256: str, size: int) -> None:
    """Verify the selected regular checkpoint before its first session load."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise FoldingBackendError(f"BioIR checkpoint type/size differs from model policy: {path}")
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after = os.fstat(handle.fileno())

    def signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    if digest != sha256 or signature(before) != signature(after) or signature(after) != signature(path.lstat()):
        raise FoldingBackendError(f"BioIR checkpoint bytes differ from model policy: {path}")


class BioIRPolicySessions:
    """At most two lazy persistent sessions, each using its verified checkpoint."""

    def __init__(
        self,
        policy: BioIRModelPolicy,
        assets: FoldingBackendAssetsSnapshot,
        output_dir: Path,
        factory: Callable[..., BioIRFoldSession],
    ) -> None:
        self.policy = policy
        self.assets = assets
        self.output_dir = output_dir
        self._factory = factory
        self._sessions: dict[str, BioIRFoldSession] = {}
        self._cleanup = ExitStack()

    def run(self, target: ProteinTarget, prepared: PreparedInput) -> FoldingResult:
        metadata = bioir_model_metadata(self.policy, len(target.chains))
        source = self.policy.model_source_for_chain_count(len(target.chains))
        session = self._sessions.get(source)
        if session is None:
            monomer = source == self.policy.monomer_model_source
            checkpoint = self.assets.bioir_monomer_checkpoint if monomer else self.assets.bioir_checkpoint
            if checkpoint is None:
                raise FoldingBackendError(f"BioIR model policy has no checkpoint path for {source}")
            _verify_checkpoint(
                Path(checkpoint),
                sha256=self.policy.monomer_checkpoint_sha256 if monomer else self.policy.multimer_checkpoint_sha256,
                size=self.policy.monomer_checkpoint_size_bytes
                if monomer
                else self.policy.multimer_checkpoint_size_bytes,
            )
            session = self._factory(Path(checkpoint), self.output_dir, model_source=source)
            self._cleanup.callback(session.close)
            self._sessions[source] = session
        result = session.run(target, prepared, session.output_dir)
        for key in ("model_source", "tool_used"):
            if result.metadata.get(key) != metadata[key]:
                raise FoldingBackendError(f"BioIR result {key} differs from selected model policy")
        return replace(result, metadata={**result.metadata, **metadata})

    def close(self) -> None:
        self._cleanup.close()
        self._sessions.clear()
