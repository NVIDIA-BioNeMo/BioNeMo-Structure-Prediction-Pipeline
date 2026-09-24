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

"""Model routing and content verification, with only scientific sessions replaced."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_bioir import BioIRModelPolicy
from bspp.orchestration.contract.folding_execution import FoldingBackendAssetsSnapshot
from bspp.orchestration.runtime.folding.execution.bioir_policy import BioIRPolicySessions, bioir_model_metadata
from bspp.orchestration.runtime.folding.execution.bioir_session import bioir_checkpoint_env_overlay
from bspp.orchestration.runtime.folding.execution.errors import FoldingBackendError
from bspp.orchestration.runtime.folding.execution.models import FoldingResult, PreparedInput, ProteinTarget


@dataclass
class Session:
    output_dir: Path
    source: str
    calls: int = 0
    closed: bool = False
    fail: bool = False

    def run(self, target: ProteinTarget, prepared: PreparedInput, output_dir: Path) -> FoldingResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("scientific session failed")
        metadata = {
            "model_source": self.source,
            "tool_used": "OpenFold2 (BioNeMo IR) / "
            + ("OpenFold-pTM" if self.source == "openfold2_ptm_1" else "AlphaFold-Multimer"),
        }
        return FoldingResult(backend="bioir", predictions=(), metadata=metadata)

    def close(self) -> None:
        self.closed = True


def setup_router(tmp_path: Path):
    paths = [tmp_path / "mono.pt", tmp_path / "multi.pt"]
    contents = [b"diagnostic monomer weights", b"diagnostic multimer weights"]
    for path, content in zip(paths, contents, strict=True):
        path.write_bytes(content)
    policy = BioIRModelPolicy(
        monomer_checkpoint_sha256=hashlib.sha256(contents[0]).hexdigest(),
        monomer_checkpoint_size_bytes=len(contents[0]),
        multimer_checkpoint_sha256=hashlib.sha256(contents[1]).hexdigest(),
        multimer_checkpoint_size_bytes=len(contents[1]),
    )
    assets = FoldingBackendAssetsSnapshot(
        backend="bioir", bioir_checkpoint=str(paths[1]), bioir_monomer_checkpoint=str(paths[0])
    )
    sessions: list[Session] = []
    checkpoints: list[Path] = []

    def factory(checkpoint: Path, output_dir: Path, *, model_source: str):
        checkpoints.append(checkpoint)
        session = Session(output_dir, model_source)
        sessions.append(session)
        return session

    router = BioIRPolicySessions(policy, assets, tmp_path / "outputs", factory)
    prepared = PreparedInput("bioir", tmp_path, tmp_path, tmp_path)
    return router, prepared, policy, paths, sessions, checkpoints


def test_expanded_chains_select_and_reuse_two_verified_sessions(tmp_path: Path) -> None:
    router, prepared, policy, paths, sessions, checkpoints = setup_router(tmp_path)
    try:
        for index, chains in enumerate((("AA",), ("AA", "AA"), ("AA", "CC"), ("DD",))):
            target = ProteinTarget(f"pdb_test_{index}", "diagnostic", chains)
            result = router.run(target, prepared)
            assert result.metadata == bioir_model_metadata(policy, len(chains))
        assert checkpoints == paths
        assert [session.calls for session in sessions] == [2, 2]
        assert [session.source for session in sessions] == ["openfold2_ptm_1", "alphafold2_multimer_1"]
    finally:
        router.close()
    assert all(session.closed for session in sessions)


@pytest.mark.parametrize("mutation", ["bytes", "size", "symlink", "fifo"])
def test_selected_checkpoint_rejected_before_session_creation(tmp_path: Path, mutation: str) -> None:
    router, prepared, _, paths, sessions, _ = setup_router(tmp_path)
    selected = paths[0]
    if mutation == "bytes":
        selected.write_bytes(b"x" * selected.stat().st_size)
    elif mutation == "size":
        selected.write_bytes(b"short")
    elif mutation == "symlink":
        target = tmp_path / "original.pt"
        selected.rename(target)
        selected.symlink_to(target)
    else:
        selected.unlink()
        os.mkfifo(selected)
    with pytest.raises((FoldingBackendError, OSError)):
        router.run(ProteinTarget("pdb_test_1", "", ("AA",)), prepared)
    assert sessions == []
    router.close()


def test_prediction_failure_closes_all_created_sessions(tmp_path: Path) -> None:
    router, prepared, _, _, sessions, _ = setup_router(tmp_path)
    mono = ProteinTarget("pdb_test_1", "", ("AA",))
    multi = ProteinTarget("pdb_test_2", "", ("AA", "AA"))
    router.run(mono, prepared)
    router.run(multi, prepared)
    sessions[0].fail = True
    try:
        with pytest.raises(RuntimeError, match="scientific session failed"):
            router.run(mono, prepared)
    finally:
        router.close()
    assert all(session.closed for session in sessions)


def test_result_provenance_cannot_disagree_with_selected_model(tmp_path: Path) -> None:
    router, prepared, _, _, sessions, _ = setup_router(tmp_path)
    target = ProteinTarget("pdb_test_1", "", ("AA",))
    router.run(target, prepared)
    sessions[0].source = "alphafold2_multimer_1"
    try:
        with pytest.raises(FoldingBackendError, match="model_source differs"):
            router.run(target, prepared)
    finally:
        router.close()


def test_native_monomer_checkpoint_overlay_uses_supported_local_hub(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENFOLD2_PTM_1_CKPT", "prior")
    monkeypatch.setenv("ALPHAFOLD2_MULTIMER_1_CKPT", "other-model")
    with pytest.raises(RuntimeError), bioir_checkpoint_env_overlay(tmp_path / "native.pt", "openfold2_ptm_1"):
        assert os.environ["OPENFOLD2_PTM_1_CKPT"] == str(tmp_path / "native.pt")
        assert os.environ["ALPHAFOLD2_MULTIMER_1_CKPT"] == "other-model"
        raise RuntimeError("load failure")
    assert os.environ["OPENFOLD2_PTM_1_CKPT"] == "prior"
    assert os.environ["ALPHAFOLD2_MULTIMER_1_CKPT"] == "other-model"


@pytest.mark.parametrize("source", ["openfold2_ptm_1", "alphafold2_multimer_1"])
def test_production_factory_selects_registered_preset_without_changing_inference_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    from bspp.orchestration.runtime.folding import executor
    from bspp.orchestration.runtime.folding.execution import bioir_session
    from bspp.orchestration.runtime.folding.execution.bioir_config import OpenFoldSettings

    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(bioir_session, "BioIRFoldSession", capture)
    checkpoint = tmp_path / "selected.pt"
    executor._production_bioir_session_factory(checkpoint, tmp_path, model_source=source)
    assert captured["checkpoint"] == checkpoint
    assert captured["settings"] == OpenFoldSettings()
    model = captured["model"]
    assert model.seed == 0
    if source == "openfold2_ptm_1":
        assert model.model_source == source
        assert model.model_preset == "openfold2_ptm_1"
        assert model.parameter_file == "finetuning_ptm_1.pt"
    else:
        assert model.model_source is None
        assert model.model_preset == "model_1_multimer_v3"
        assert model.parameter_file == "params_model_1_multimer_v3.pt"


@pytest.mark.parametrize(
    "source,target_id,chains",
    [
        ("openfold2_ptm_1", "pdb_test_assembly_1", ("AC",)),
        ("alphafold2_multimer_1", "homo_P12345", ("A", "A")),
        ("alphafold2_multimer_1", "hetero_P12345_Q12345", ("A", "C")),
    ],
)
def test_session_emits_selected_model_provenance_and_unmodified_full_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, target_id: str, chains: tuple[str, ...]
) -> None:
    import json
    import sys
    from types import SimpleNamespace

    from bspp.orchestration.runtime.folding.artifact_evidence import read_scores
    from bspp.orchestration.runtime.folding.execution.bioir_config import OpenFoldModelSettings, OpenFoldSettings
    from bspp.orchestration.runtime.folding.execution.bioir_session import BioIRFoldSession
    from bspp.orchestration.runtime.folding.execution.emitter_support import serialize_scores_json

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    checkpoint = tmp_path / "diagnostic.pt"
    checkpoint.write_bytes(b"test boundary: never loaded")
    structure = tmp_path / "raw.pdb"
    structure.write_text("TEST STRUCTURE\n")
    monomer = source == "openfold2_ptm_1"
    raw_scores = {"plddt": [80.0, 90.0], "pae": [[0.0, 1.0], [2.0, 0.0]], "ptm": 0.8, "iptm": None if monomer else 0.9}

    class DiagnosticSession(BioIRFoldSession):
        def _build_processor(self, *args):
            return lambda rows: [{"output_path": str(structure), "scores": raw_scores}]

        def _request(self, prepared_dir):
            return None

    session = DiagnosticSession(
        OpenFoldModelSettings(source, source, "diagnostic.pt", model_source=source),
        OpenFoldSettings(),
        checkpoint,
        tmp_path / "outputs",
    )
    result = session.predict(ProteinTarget(target_id, "", chains), tmp_path)
    assert result.metadata["model_source"] == source
    assert result.metadata["tool_used"] == "OpenFold2 (BioNeMo IR) / " + (
        "OpenFold-pTM" if monomer else "AlphaFold-Multimer"
    )
    score_path = result.predictions[0].scores_path
    payload, identity = read_scores(score_path, root=tmp_path, length=2, model_source=source)
    assert payload.extras == {"bioir_model_source": source}
    assert identity.sha256 == hashlib.sha256(score_path.read_bytes()).hexdigest()
    written = json.loads(score_path.read_bytes())
    for key, value in raw_scores.items():
        assert written[key] == value
    assert result.predictions[0].structure_path.read_bytes() == structure.read_bytes()
    assert written.pop("bioir_model_source") == source
    assert json.dumps(written) == serialize_scores_json(
        plddt=[80.0, 90.0], pae=[[0.0, 1.0], [2.0, 0.0]], max_pae=2.0, ptm=0.8, iptm=None if monomer else 0.9
    )
    for provenance in ({}, {"bioir_model_source": "wrong-model"}):
        refused = tmp_path / "invalid-provenance.json"
        refused.write_text(json.dumps({**written, **provenance}))
        with pytest.raises(ValueError, match="model source differs"):
            read_scores(refused, root=tmp_path, length=2, model_source=source)


@pytest.mark.parametrize("image", ["runtime", "bioir"])
def test_image_smoke_exercises_real_producer_and_restores_torch(monkeypatch: pytest.MonkeyPatch, image: str) -> None:
    import importlib.util
    import sys
    from types import ModuleType

    monkeypatch.setenv("BSPP_CUDA_COMPAT_DIR", "/unused-cuda-compat")
    prior = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", prior)
    path = Path(__file__).resolve().parents[2] / "containers" / "folding" / image / "image-smoke.py"
    spec = importlib.util.spec_from_file_location(f"{image}_producer_smoke", path)
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    smoke._verify_bioir_score_producer()
    assert sys.modules["torch"] is prior
