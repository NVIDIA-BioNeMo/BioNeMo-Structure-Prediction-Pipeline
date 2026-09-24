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

"""Checkpoint binding at eager construction and deferred processor execution.

The optional installed-BioIR case exercises its real lazy SerialProcessor with
a diagnostic UDF that stops before any model, features, or prediction executes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bspp.orchestration.runtime.folding.execution.bioir_config import OpenFoldModelSettings, OpenFoldSettings
from bspp.orchestration.runtime.folding.execution.bioir_session import BioIRFoldSession
from bspp.orchestration.runtime.folding.execution.errors import FoldingBackendError
from bspp.orchestration.runtime.folding.execution.models import ProteinTarget

_ENV = "ALPHAFOLD2_MULTIMER_1_CKPT"
_MODEL = OpenFoldModelSettings("model_1_multimer_v3", "model_1_multimer_v3", "params_model_1_multimer_v3.pt")
_TARGET = ProteinTarget("diagnostic", "diagnostic", ("A",))


class _StopBeforeInferenceError(RuntimeError):
    pass


def _bind_prior(monkeypatch: pytest.MonkeyPatch, prior: str | None) -> None:
    if prior is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, prior)


def _session_with_processor(
    checkpoint: Path,
    output: Path,
    processor: Any,
) -> BioIRFoldSession:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"diagnostic placeholder: never deserialized")

    class DiagnosticSession(BioIRFoldSession):
        def _build_processor(self, *args: Any) -> Any:
            assert os.environ[_ENV] == str(checkpoint.resolve())
            return processor

        def _request(self, prepared_dir: Path) -> None:
            return None

    return DiagnosticSession(_MODEL, OpenFoldSettings(), checkpoint, output)


@pytest.fixture
def cpu_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))


@pytest.mark.parametrize("prior", [None, "/prior/model.pt"])
@pytest.mark.parametrize("raises", [False, True])
def test_processor_call_binds_checkpoint_and_restores_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_torch: None, prior: str | None, raises: bool
) -> None:
    _bind_prior(monkeypatch, prior)
    observed = []

    def processor(rows: Any) -> list[Any]:
        observed.append(os.environ.get(_ENV))
        if raises:
            raise _StopBeforeInferenceError
        return []

    checkpoint = tmp_path / "selected.pt"
    session = _session_with_processor(checkpoint, tmp_path / "output", processor)
    assert os.environ.get(_ENV) == prior
    error = _StopBeforeInferenceError if raises else FoldingBackendError
    with pytest.raises(error):
        session.predict(_TARGET, tmp_path)
    assert observed == [str(checkpoint.resolve())]
    assert os.environ.get(_ENV) == prior


def test_sequential_sessions_keep_distinct_checkpoints_and_reuse_processors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_torch: None
) -> None:
    monkeypatch.setenv(_ENV, "/ambient/model.pt")
    observed = []

    def processor(rows: Any) -> list[Any]:
        observed.append(os.environ.get(_ENV))
        return []

    first = _session_with_processor(tmp_path / "first.pt", tmp_path / "first", processor)
    second = _session_with_processor(tmp_path / "second.pt", tmp_path / "second", processor)
    for session in (first, second, first):
        with pytest.raises(FoldingBackendError, match="returned 0 rows"):
            session.predict(_TARGET, tmp_path)
        assert os.environ[_ENV] == "/ambient/model.pt"
        assert session.processor is processor
    assert observed == [str(tmp_path / name) for name in ("first.pt", "second.pt", "first.pt")]


def test_stored_checkpoint_resolution_survives_changed_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_torch: None
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(_ENV, raising=False)
    observed = []

    def processor(rows: Any) -> list[Any]:
        observed.append(os.environ.get(_ENV))
        return []

    session = _session_with_processor(Path("selected.pt"), tmp_path / "output", processor)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(FoldingBackendError, match="returned 0 rows"):
        session.predict(_TARGET, tmp_path)
    assert observed == [str(tmp_path / "selected.pt")]
    assert _ENV not in os.environ


@pytest.mark.parametrize("prior", [None, "/prior/model.pt"])
def test_installed_serial_processor_lazy_constructor_sees_selected_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior: str | None
) -> None:
    """Run explicitly in the pinned BioIR image; no model or target inference."""
    processor_module = pytest.importorskip("bionemo_ir.pipeline.processor.base")
    stage_module = pytest.importorskip("bionemo_ir.pipeline.stages.base")
    _bind_prior(monkeypatch, prior)
    observed = []

    class DiagnosticUDF(stage_module.StatefulStageUDF):
        def __init__(self, **kwargs: Any) -> None:
            observed.append(os.environ.get(_ENV))
            raise _StopBeforeInferenceError("diagnostic constructor: no scientific work")

    processor = processor_module.SerialProcessor(
        processor_module.ProcessorConfig(model_source="alphafold2_multimer_1", executor_backend=None),
        [stage_module.StatefulStage(fn=DiagnosticUDF)],
    )
    checkpoint = tmp_path / "selected.pt"
    session = _session_with_processor(checkpoint, tmp_path / "output", processor)
    assert observed == []
    assert processor._udf_instances == {}
    assert os.environ.get(_ENV) == prior
    with pytest.raises(_StopBeforeInferenceError, match="no scientific work"):
        session.predict(_TARGET, tmp_path)
    assert observed == [str(checkpoint.resolve())]
    assert os.environ.get(_ENV) == prior
