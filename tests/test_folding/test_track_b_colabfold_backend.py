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

"""Focused tests for the ColabFold backend emitter."""

from __future__ import annotations

import json
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from bspp.orchestration.contract.prediction_pair import prediction_pair_from_mapping
from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.runtime.folding.execution.chain_manifest import (
    ChainManifest,
    ChainManifestRow,
    ambiguous_chain_manifest_metadata,
)
from bspp.orchestration.runtime.folding.execution.colabfold_backend import (
    COLABFOLD_TOOL_USED,
    ColabFoldBackend,
    ColabFoldConfig,
    build_colabfold_command,
    stage_canonical_pair,
)
from bspp.orchestration.runtime.folding.execution.models import PreparedInput, ProteinTarget

_MODEL_TYPE = "alphafold2_multimer_v3"


class _RecordingRunner:
    """Test runner collaborator: records argv/env and never spawns a subprocess."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def __call__(self, argv: Sequence[str], *, env: Mapping[str, str]) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((tuple(argv), dict(env)))
        return subprocess.CompletedProcess(argv, self.returncode)


def _config(tmp_path: Path) -> ColabFoldConfig:
    return ColabFoldConfig(
        weights_dir=tmp_path / "weights",
        msa_cache_dir=tmp_path / "msa_cache",
        structures_dir=tmp_path / "structures",
    )


def _manifest(target_id: str) -> ChainManifest:
    return ChainManifest((ChainManifestRow(target_id, "e1", "A", "P1"),))


def _prepared(tmp_path: Path) -> PreparedInput:
    return PreparedInput(
        "colabfold",
        fasta_dir=tmp_path / "fasta",
        alignment_dir=tmp_path / "alignments",
        template_dir=tmp_path / "templates",
    )


def _write_raw_pair(
    target_id: str,
    output_dir: Path,
    scores: dict[str, object],
    *,
    model_n: int = 1,
    seed: int = 0,
) -> tuple[Path, Path]:
    raw_pdb = output_dir / f"{target_id}_unrelaxed_rank_001_{_MODEL_TYPE}_model_{model_n}_seed_{seed:03d}.pdb"
    raw_json = output_dir / f"{target_id}_scores_rank_001_{_MODEL_TYPE}_model_{model_n}_seed_{seed:03d}.json"
    raw_pdb.write_text("HEADER    AF-0000000000000001\nEND\n", encoding="utf-8")
    raw_json.write_text(json.dumps(scores), encoding="utf-8")
    return raw_pdb, raw_json


def test_build_command_parameterizes_all_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)
    a3m_path = tmp_path / "msa_cache" / "AF-0000000000000001.a3m"
    output_dir = tmp_path / "structures" / "out"

    spec = build_colabfold_command(config, a3m_path, output_dir)

    assert spec.argv[0] == "colabfold_batch"
    assert spec.argv[1] == f"--model-type={_MODEL_TYPE}"
    assert "--data" in spec.argv
    assert spec.argv[spec.argv.index("--data") + 1] == str(config.weights_dir)
    assert f"--num-recycle={config.num_recycle}" in spec.argv
    assert f"--num-models={config.num_models}" in spec.argv
    assert f"--num-seeds={config.num_seeds}" in spec.argv
    assert "--skip-output" in spec.argv
    assert spec.argv[spec.argv.index("--skip-output") + 1] == "msa,plots,pae_json"
    assert spec.argv[-2] == str(a3m_path)
    assert spec.argv[-1] == str(output_dir)

    assert spec.env["COLABFOLD_CONFIG"] == str(config.weights_dir)
    assert spec.env["CUDA_VISIBLE_DEVICES"] == str(config.gpu_id)
    assert spec.env["JAX_COMPILATION_CACHE_DIR"] == str(config.structures_dir / "jax_cache")

    # No hardcoded container-internal literal paths anywhere in argv or env.
    assert "/weights" not in spec.argv
    assert "/global_msa_cache" not in spec.argv
    assert "/output" not in spec.argv
    assert "/tmp/jax_cache" not in spec.env.values()


def test_build_command_env_is_parameterized(tmp_path: Path) -> None:
    config = ColabFoldConfig(
        weights_dir=tmp_path / "w",
        msa_cache_dir=tmp_path / "m",
        structures_dir=tmp_path / "s",
        gpu_id=3,
    )
    spec = build_colabfold_command(config, tmp_path / "m" / "x.a3m", tmp_path / "s" / "out")
    assert spec.env["COLABFOLD_CONFIG"] == str(tmp_path / "w")
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "3"
    assert spec.env["JAX_COMPILATION_CACHE_DIR"] == str(tmp_path / "s" / "jax_cache")


def test_run_overlays_overrides_on_inherited_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An inherited variable survives the call: the child environment is the
    # inherited environment with the backend overrides applied, not a bare
    # three-key replacement.
    monkeypatch.setenv("BSPP_INHERITED_SENTINEL", "sentinel-value")
    # An inherited collision loses to the config-derived override.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")

    runner = _RecordingRunner(0)
    config = _config(tmp_path)
    backend = ColabFoldBackend(config, _manifest("AF-0000000000000001"), runner=runner)
    target = ProteinTarget("AF-0000000000000001", "desc", ("A",))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_raw_pair(target.target_id, output_dir, {"plddt": [1.0], "pae": [[1.0]], "max_pae": 1.0})

    result = backend.run(target, _prepared(tmp_path), output_dir)

    assert len(result.predictions) == 1
    assert len(runner.calls) == 1
    _argv, env = runner.calls[0]
    assert env["BSPP_INHERITED_SENTINEL"] == "sentinel-value"
    assert env["CUDA_VISIBLE_DEVICES"] == str(config.gpu_id)
    assert env["COLABFOLD_CONFIG"] == str(config.weights_dir)
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(config.structures_dir / "jax_cache")


def test_nonzero_returncode_raises(tmp_path: Path) -> None:
    backend = ColabFoldBackend(_config(tmp_path), _manifest("AF-0000000000000001"), runner=_RecordingRunner(7))
    target = ProteinTarget("AF-0000000000000001", "desc", ("A",))
    with pytest.raises(RuntimeError, match="returncode 7"):
        backend.run(target, _prepared(tmp_path), tmp_path / "out")


def test_direct_a3m_intake_from_alignment_dir(tmp_path: Path) -> None:
    runner = _RecordingRunner(0)
    backend = ColabFoldBackend(_config(tmp_path), _manifest("AF-0000000000000001"), runner=runner)
    target = ProteinTarget("AF-0000000000000001", "desc", ("A",))
    prepared = _prepared(tmp_path)
    prepared.alignment_dir.mkdir(parents=True)
    a3m_path = prepared.alignment_dir / f"{target.target_id}.a3m"
    a3m_path.write_text(">AF-0000000000000001\nAAAA\n", encoding="utf-8")

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    _write_raw_pair(target.target_id, output_dir, {"plddt": [1.0], "pae": [[1.0]], "max_pae": 1.0})

    backend.run(target, prepared, output_dir)

    assert len(runner.calls) == 1
    argv, _env = runner.calls[0]
    assert argv[-2] == str(a3m_path)
    assert argv[-1] == str(output_dir)


def test_stage_canonical_pair_renames_and_reserializes(tmp_path: Path) -> None:
    scores = {
        "plddt": [81.585, 90.123],
        "pae": [[0.0, 1.234], [1.234, 0.0]],
        "max_pae": 3.456,
        "ptm": 0.8765,
        "iptm": 0.8125,
        "ranking_confidence": 0.99,
    }
    raw_pdb, raw_json = _write_raw_pair("AFDB_AF-0000000000000001", tmp_path, scores)

    structure_path, scores_path = stage_canonical_pair("AFDB_AF-0000000000000001", raw_pdb, raw_json, tmp_path)

    assert structure_path.name == "AF-0000000000000001-model_v1.pdb"
    assert scores_path.name == "AF-0000000000000001-meta_v1.json"
    assert structure_path.read_text(encoding="utf-8") == "HEADER    AF-0000000000000001\nEND\n"
    assert not raw_pdb.exists()

    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    assert payload["plddt"] == [81.58, 90.12]
    assert payload["pae"] == [[0.0, 1.23], [1.23, 0.0]]
    assert payload["max_pae"] == 3.46
    assert payload["ptm"] == 0.88
    assert payload["iptm"] == 0.81
    assert payload["ranking_confidence"] == 0.99


def test_stage_canonical_pair_missing_optional_scores(tmp_path: Path) -> None:
    raw_pdb, raw_json = _write_raw_pair(
        "AF-0000000000000001",
        tmp_path,
        {"plddt": [1.0], "pae": [[1.0]], "max_pae": 1.0},
    )
    _structure_path, scores_path = stage_canonical_pair("AF-0000000000000001", raw_pdb, raw_json, tmp_path)
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    assert payload["ptm"] is None
    assert payload["iptm"] is None


def test_stage_canonical_pair_serialization_is_deterministic(tmp_path: Path) -> None:
    scores = {"plddt": [1.0, 2.0], "pae": [[1.0, 2.0], [2.0, 1.0]], "max_pae": 2.0}
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    raw_pdb_1, raw_json_1 = _write_raw_pair("AF-0000000000000001", first_dir, scores)
    raw_pdb_2, raw_json_2 = _write_raw_pair("AF-0000000000000001", second_dir, scores)

    _structure_1, scores_1 = stage_canonical_pair("AF-0000000000000001", raw_pdb_1, raw_json_1, first_dir)
    _structure_2, scores_2 = stage_canonical_pair("AF-0000000000000001", raw_pdb_2, raw_json_2, second_dir)

    assert scores_1.read_bytes() == scores_2.read_bytes()


def test_ambiguous_manifest_fails_closed_without_running(tmp_path: Path) -> None:
    runner = _RecordingRunner(0)
    # No rows for the requested target: classify_target resolves "ambiguous".
    backend = ColabFoldBackend(_config(tmp_path), _manifest("AF-0000000000000002"), runner=runner)
    target = ProteinTarget("AF-0000000000000001", "desc", ("A",))

    result = backend.run(target, _prepared(tmp_path), tmp_path / "out")

    assert result.backend == "colabfold"
    assert result.predictions == ()
    assert result.metadata == ambiguous_chain_manifest_metadata()
    assert runner.calls == []


def test_default_num_models_five_resolves_actual_rank_one_suffix(tmp_path: Path) -> None:
    # The shipped config default keeps num_models=5; rank 1 is not assumed to be
    # model_1_seed_000. The resolver must discover the real harvested suffix.
    config = _config(tmp_path)
    assert config.num_models == 5

    runner = _RecordingRunner(0)
    backend = ColabFoldBackend(config, _manifest("AF-0000000000000001"), runner=runner)
    target = ProteinTarget("AF-0000000000000001", "desc", ("A",))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_raw_pair(
        target.target_id,
        output_dir,
        {"plddt": [1.0], "pae": [[1.0]], "max_pae": 1.0},
        model_n=3,
        seed=0,
    )

    result = backend.run(target, _prepared(tmp_path), output_dir)

    assert len(result.predictions) == 1
    assert result.predictions[0].structure_path.name == "AF-0000000000000001-model_v1.pdb"


@pytest.mark.parametrize(
    "setup",
    [
        "zero",
        "multiple_pdb",
        "multiple_json",
        "unpaired",
    ],
)
def test_rank_one_resolution_fails_closed(setup: str, tmp_path: Path) -> None:
    runner = _RecordingRunner(0)
    backend = ColabFoldBackend(_config(tmp_path), _manifest("AF-0000000000000001"), runner=runner)
    target = ProteinTarget("AF-0000000000000001", "desc", ("A",))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    scores = {"plddt": [1.0], "pae": [[1.0]], "max_pae": 1.0}

    if setup == "multiple_pdb":
        _write_raw_pair(target.target_id, output_dir, scores, model_n=1)
        _write_raw_pair(target.target_id, output_dir, scores, model_n=2)
        match = "multiple rank-1"
    elif setup == "multiple_json":
        _write_raw_pair(target.target_id, output_dir, scores, model_n=1)
        # A second JSON with a different suffix creates an unmatched/multiple set.
        extra_json = output_dir / f"{target.target_id}_scores_rank_001_{_MODEL_TYPE}_model_2_seed_000.json"
        extra_json.write_text(json.dumps(scores), encoding="utf-8")
        match = "multiple rank-1"
    elif setup == "unpaired":
        raw_pdb = output_dir / f"{target.target_id}_unrelaxed_rank_001_{_MODEL_TYPE}_model_1_seed_000.pdb"
        raw_pdb.write_text("HEADER\nEND\n", encoding="utf-8")
        raw_json = output_dir / f"{target.target_id}_scores_rank_001_{_MODEL_TYPE}_model_2_seed_000.json"
        raw_json.write_text(json.dumps(scores), encoding="utf-8")
        match = "unmatched rank-1"
    else:  # zero
        match = "no rank-1"

    with pytest.raises(ValueError, match=match):
        backend.run(target, _prepared(tmp_path), output_dir)

    # The subprocess was invoked once; only the rank-1 resolution failed closed.
    assert len(runner.calls) == 1


def test_canonical_pair_members_have_mode_0644_and_no_temp_files(tmp_path: Path) -> None:
    runner = _RecordingRunner(0)
    backend = ColabFoldBackend(_config(tmp_path), _manifest("AF-0000000000000001"), runner=runner)
    target = ProteinTarget("AF-0000000000000001", "desc", ("A",))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_raw_pair(target.target_id, output_dir, {"plddt": [1.0], "pae": [[1.0]], "max_pae": 1.0})

    result = backend.run(target, _prepared(tmp_path), output_dir)

    prediction = result.predictions[0]
    assert stat.S_IMODE(prediction.structure_path.stat().st_mode) == 0o644
    assert stat.S_IMODE(prediction.scores_path.stat().st_mode) == 0o644
    # No mkstemp temporary files remain from the atomic writes.
    assert [entry.name for entry in output_dir.iterdir() if entry.name.startswith("tmp")] == []


def test_run_requires_chain_manifest(tmp_path: Path) -> None:
    backend = ColabFoldBackend(_config(tmp_path))
    with pytest.raises(ValueError, match="ChainManifest"):
        backend.run(ProteinTarget("AF-0000000000000001", "desc", ("A",)), _prepared(tmp_path), tmp_path / "out")


def test_emitted_pair_round_trips_through_contract(tmp_path: Path) -> None:
    runner = _RecordingRunner(0)
    backend = ColabFoldBackend(_config(tmp_path), _manifest("AF-0000000000000001"), runner=runner)
    target = ProteinTarget("AF-0000000000000001", "desc", ("A",))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_raw_pair(
        target.target_id,
        output_dir,
        {"plddt": [1.0, 2.0], "pae": [[1.0, 2.0], [2.0, 1.0]], "max_pae": 2.0, "ptm": 0.8765, "iptm": 0.8125},
    )

    result = backend.run(target, _prepared(tmp_path), output_dir)

    assert result.backend == "colabfold"
    assert len(result.predictions) == 1
    prediction = result.predictions[0]
    assert prediction.rank == 1
    assert result.metadata["tool_used"] == COLABFOLD_TOOL_USED

    scores = json.loads(prediction.scores_path.read_bytes())
    pair = prediction_pair_from_mapping(
        {
            "schema_version": 1,
            "model_entity_id": target.target_id,
            "tool_used": result.metadata["tool_used"],
            "structure_path": str(prediction.structure_path),
            "scores_path": str(prediction.scores_path),
            "scores": scores,
        }
    )

    assert pair.model_entity_id == "AF-0000000000000001"
    assert pair.tool_used in VALID_TOOL_USED
    assert pair.scores.plddt == (1.0, 2.0)
    assert pair.scores.pae == ((1.0, 2.0), (2.0, 1.0))
    assert pair.scores.max_pae == 2.0
    assert pair.scores.ptm == 0.88
    assert pair.scores.iptm == 0.81


def test_colabfold_tool_used_constant_is_valid() -> None:
    assert VALID_TOOL_USED[0] == COLABFOLD_TOOL_USED
    assert COLABFOLD_TOOL_USED in VALID_TOOL_USED
