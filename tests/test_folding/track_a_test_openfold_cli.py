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

"""Focused tests for the job-local OpenFold CLI backend.

Adapted from the reference pipeline's ``test_openfold2_nim_compat.py`` (fake executable
entrypoint) and ``test_local_backends.py`` (backend behavior), landing on the
Step-0 canonical-pair emitter seam. The fake ``run_pretrained_openfold.py``
entrypoint is built at runtime under ``tmp_path``; no checked-in fixture is
required.  The fake consumes a FASTA directory, derives the harvested output
tag from FASTA headers, and emits artifacts recursively using harvested names.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from bspp.orchestration.contract.prediction_pair import prediction_pair_from_mapping
from bspp.orchestration.runtime.folding.execution.backend import FoldingBackend
from bspp.orchestration.runtime.folding.execution.errors import FoldingBackendError
from bspp.orchestration.runtime.folding.execution.models import PreparedInput, ProteinTarget
from bspp.orchestration.runtime.folding.execution.openfold_cli import (
    OpenFoldCliBackend,
    build_openfold_cli_argv,
    capture_openfold_scores,
)

_TARGET_ID = "AF-0000000000000001"
_TOOL_USED = "OpenFold / AlphaFold-Multimer"
_CHAIN_ID = f"{_TARGET_ID}_A"
_OUTPUT_TAG = _CHAIN_ID
_MODEL_PRESET = "model_1_multimer_v3"


def _write_fake_entrypoint(bin_dir: Path) -> Path:
    """Write an executable ``run_pretrained_openfold.py`` that replays a spec pickle."""

    entrypoint = bin_dir / "run_pretrained_openfold.py"
    entrypoint.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python
            import argparse
            import json
            import os
            import pickle
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("fasta_dir")
            parser.add_argument("templates")
            parser.add_argument("--output_dir", required=True)
            parser.add_argument("--config_preset", required=True)
            parser.add_argument("--jax_param_path", required=True)
            parser.add_argument("--data_random_seed", required=True)
            parser.add_argument("--use_precomputed_alignments", required=True)
            parser.add_argument("--model_device", required=True)
            parser.add_argument("--skip_relaxation", action="store_true")
            parser.add_argument("--save_outputs", action="store_true")
            args = parser.parse_args()

            marker = os.environ.get("OPENFOLD_FAKE_MARKER")
            if marker:
                Path(marker).write_text("ran", encoding="utf-8")

            fasta_dir = Path(args.fasta_dir)
            headers = []
            for path in sorted(fasta_dir.glob("*.fasta")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.startswith(">"):
                        headers.append(line[1:].strip())
            output_tag = "-".join(headers)

            output_dir = Path(args.output_dir)
            predictions = output_dir / "predictions"
            predictions.mkdir(parents=True, exist_ok=True)

            mode = os.environ.get("OPENFOLD_FAKE_MODE", "upstream")
            with open(os.environ["OPENFOLD_FAKE_SPEC"], "rb") as fh:
                spec = pickle.load(fh)

            if mode == "upstream":
                structure = predictions / f"{output_tag}_{args.config_preset}_unrelaxed.pdb"
                structure.write_text("HEADER    OPENFOLD\\nEND\\n", encoding="utf-8")
                with open(predictions / f"{output_tag}_{args.config_preset}_output_dict.pkl", "wb") as fh:
                    pickle.dump(spec, fh)
            elif mode == "ranked":
                structure = predictions / f"{output_tag}_unrelaxed_rank_001.pdb"
                structure.write_text("HEADER    OPENFOLD\\nEND\\n", encoding="utf-8")
                with open(predictions / f"{output_tag}_{args.config_preset}_output_dict.pkl", "wb") as fh:
                    pickle.dump(spec, fh)
            elif mode == "json-fallback":
                structure = predictions / f"{output_tag}_unrelaxed_rank_001.pdb"
                structure.write_text("HEADER    OPENFOLD\\nEND\\n", encoding="utf-8")
                with open(predictions / f"{output_tag}_scores_rank_001.json", "w", encoding="utf-8") as fh:
                    json.dump(spec, fh)
            else:
                raise SystemExit(f"unknown OPENFOLD_FAKE_MODE: {mode}")
            """
        ),
        encoding="utf-8",
    )
    entrypoint.chmod(0o755)
    return entrypoint


def _write_spec(spec_path: Path, spec: dict[str, object]) -> None:
    with spec_path.open("wb") as fh:
        pickle.dump(spec, fh)


def _prep_fake_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec: dict[str, object]) -> Path:
    """Build the fake entrypoint, model dir, spec, and PATH; return the model dir."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_entrypoint(bin_dir)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "params_model_1_multimer_v3.npz").write_bytes(b"params")
    spec_path = tmp_path / "spec.pkl"
    _write_spec(spec_path, spec)
    monkeypatch.setenv("OPENFOLD_FAKE_SPEC", str(spec_path))
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:{os.environ['PATH']}")
    return model_dir


def _prepared(tmp_path: Path) -> PreparedInput:
    fasta_dir = tmp_path / "fasta"
    fasta_dir.mkdir(exist_ok=True)
    (fasta_dir / f"{_TARGET_ID}.fasta").write_text(f">{_CHAIN_ID}\nA\n", encoding="utf-8")
    return PreparedInput(
        "openfold-cli",
        fasta_dir,
        tmp_path / "align",
        tmp_path / "tmpl",
        {"chain_ids": [_CHAIN_ID]},
    )


# ---------------------------------------------------------------------------
# build_openfold_cli_argv
# ---------------------------------------------------------------------------


def test_build_openfold_cli_argv_exact() -> None:
    prepared = PreparedInput("openfold-cli", Path("/fasta"), Path("/align"), Path("/tmpl"), {})
    argv = build_openfold_cli_argv(
        prepared,
        Path("/out"),
        model_dir=Path("/models"),
        model_preset=_MODEL_PRESET,
        parameter_file="params_model_1_multimer_v3.npz",
        seed=7,
    )
    assert argv == [
        "run_pretrained_openfold.py",
        "/fasta",
        "/tmpl",
        "--use_precomputed_alignments",
        "/align",
        "--output_dir",
        "/out/raw",
        "--model_device",
        "cuda:0",
        "--config_preset",
        _MODEL_PRESET,
        "--jax_param_path",
        "/models/params_model_1_multimer_v3.npz",
        "--data_random_seed",
        "7",
        "--skip_relaxation",
        "--save_outputs",
    ]


# ---------------------------------------------------------------------------
# capture_openfold_scores
# ---------------------------------------------------------------------------

_PLDDT = np.array([81.585, 90.123, 75.0])
_PAE = np.array([[0.0, 1.234, 2.345], [1.234, 0.0, 3.456], [2.345, 3.456, 0.0]])


def test_capture_openfold_scores_full_extraction() -> None:
    plddt, pae, max_pae, ptm, iptm = capture_openfold_scores(
        {"plddt": _PLDDT, "predicted_aligned_error": _PAE, "ptm": 0.8765, "iptm": 0.8125}
    )
    assert plddt == [81.585, 90.123, 75.0]
    assert pae == [[0.0, 1.234, 2.345], [1.234, 0.0, 3.456], [2.345, 3.456, 0.0]]
    assert max_pae == 3.456
    assert ptm == 0.8765
    assert iptm == 0.8125


def test_capture_openfold_scores_pae_key_fallback() -> None:
    _plddt, pae, max_pae, _ptm, _iptm = capture_openfold_scores({"plddt": [1.0, 2.0], "pae": [[0.0, 1.0], [1.0, 0.0]]})
    assert pae == [[0.0, 1.0], [1.0, 0.0]]
    assert max_pae == 1.0


def test_capture_openfold_scores_ptm_iptm_score_fallback() -> None:
    _plddt, _pae, _max_pae, ptm, iptm = capture_openfold_scores(
        {"plddt": [1.0], "pae": [[0.0]], "ptm_score": 0.5, "iptm_score": 0.6}
    )
    assert ptm == 0.5
    assert iptm == 0.6


def test_capture_openfold_scores_optional_scores_absent() -> None:
    _plddt, _pae, _max_pae, ptm, iptm = capture_openfold_scores({"plddt": [1.0], "pae": [[0.0]]})
    assert ptm is None
    assert iptm is None


def test_capture_openfold_scores_missing_plddt() -> None:
    with pytest.raises(FoldingBackendError):
        capture_openfold_scores({"predicted_aligned_error": [[0.0]]})


def test_capture_openfold_scores_missing_pae() -> None:
    with pytest.raises(FoldingBackendError):
        capture_openfold_scores({"plddt": [1.0]})


def test_capture_openfold_scores_empty_plddt() -> None:
    with pytest.raises(FoldingBackendError):
        capture_openfold_scores({"plddt": [], "pae": [[0.0]]})


def test_capture_openfold_scores_empty_pae() -> None:
    with pytest.raises(FoldingBackendError):
        capture_openfold_scores({"plddt": [1.0], "pae": []})
    with pytest.raises(FoldingBackendError):
        capture_openfold_scores({"plddt": [1.0, 2.0], "pae": [[], []]})


def test_capture_openfold_scores_nan_plddt() -> None:
    with pytest.raises(FoldingBackendError):
        capture_openfold_scores({"plddt": [1.0, np.nan], "pae": [[0.0, 1.0], [1.0, 0.0]]})


def test_capture_openfold_scores_nan_pae() -> None:
    with pytest.raises(FoldingBackendError):
        capture_openfold_scores({"plddt": [1.0], "pae": [[np.nan]]})


def test_capture_openfold_scores_pae_plddt_cardinality_mismatch() -> None:
    with pytest.raises(FoldingBackendError, match="3 rows but 'plddt' has 2"):
        capture_openfold_scores({"plddt": [1.0, 2.0], "pae": [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]})


def test_capture_openfold_scores_rejects_non_square_pae() -> None:
    """A rectangular PAE wider than the residue count fails capture."""
    with pytest.raises(FoldingBackendError, match="row 0 has 3 columns but 'plddt' has 2 residues"):
        capture_openfold_scores({"plddt": [1.0, 2.0], "pae": [[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]]})


def test_capture_openfold_scores_rejects_ragged_pae() -> None:
    """A ragged nested PAE cannot even form a 2-D array and fails capture."""
    with pytest.raises(FoldingBackendError, match="must be a 2-D array"):
        capture_openfold_scores({"plddt": [1.0, 2.0], "pae": [[0.0, 1.0], [1.0]]})


def test_capture_openfold_scores_malformed_optional_scalar() -> None:
    with pytest.raises(FoldingBackendError, match="'ptm' must be a finite number"):
        capture_openfold_scores({"plddt": [1.0], "pae": [[0.0]], "ptm": True})


# ---------------------------------------------------------------------------
# OpenFoldCliBackend
# ---------------------------------------------------------------------------


def test_openfold_cli_backend_name_and_run_callable() -> None:
    backend = OpenFoldCliBackend(model_dir=Path("/models"))
    assert backend.name == "openfold-cli"
    assert callable(backend.run)


def test_openfold_cli_backend_satisfies_protocol() -> None:
    backend: FoldingBackend = OpenFoldCliBackend(model_dir=Path("/models"))
    assert backend.name == "openfold-cli"
    assert callable(backend.run)


def test_openfold_cli_backend_upstream_structure_and_pickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = _prep_fake_environment(
        monkeypatch,
        tmp_path,
        {"plddt": _PLDDT, "predicted_aligned_error": _PAE, "ptm": 0.8765, "iptm": 0.8125},
    )
    backend = OpenFoldCliBackend(model_dir=model_dir)
    result = backend.run(
        ProteinTarget(_TARGET_ID, "desc", ("A",)),
        _prepared(tmp_path),
        tmp_path / "out",
    )

    assert result.backend == "openfold-cli"
    assert len(result.predictions) == 1
    prediction = result.predictions[0]
    assert prediction.rank == 1
    assert prediction.structure_path.name.endswith("-model_v1.pdb")
    assert prediction.scores_path.name.endswith("-meta_v1.json")
    assert prediction.structure_path.exists()
    assert prediction.scores_path.exists()

    raw_dir = tmp_path / "out" / "raw"
    predictions_dir = raw_dir / "predictions"
    assert (predictions_dir / f"{_OUTPUT_TAG}_{_MODEL_PRESET}_unrelaxed.pdb").exists()
    assert not (predictions_dir / f"{_OUTPUT_TAG}_{_MODEL_PRESET}_output_dict.pkl").exists()

    scores = json.loads(prediction.scores_path.read_bytes())
    assert scores["plddt"] == [81.58, 90.12, 75.0]
    assert scores["pae"] == [[0.0, 1.23, 2.34], [1.23, 0.0, 3.46], [2.34, 3.46, 0.0]]
    assert scores["max_pae"] == 3.46
    assert scores["ptm"] == 0.88
    assert scores["iptm"] == 0.81

    pair = prediction_pair_from_mapping(
        {
            "schema_version": 1,
            "model_entity_id": _TARGET_ID,
            "tool_used": result.metadata["tool_used"],
            "structure_path": str(prediction.structure_path),
            "scores_path": str(prediction.scores_path),
            "scores": scores,
        }
    )
    assert pair.tool_used == _TOOL_USED
    assert pair.scores.plddt == (81.58, 90.12, 75.0)
    assert pair.scores.pae == ((0.0, 1.23, 2.34), (1.23, 0.0, 3.46), (2.34, 3.46, 0.0))
    assert pair.scores.max_pae == 3.46

    assert result.metadata["tool_used"] == _TOOL_USED
    assert result.metadata["model_preset"] == _MODEL_PRESET
    assert result.metadata["parameter_file"] == "params_model_1_multimer_v3.npz"
    assert result.metadata["seed"] == 0


def test_openfold_cli_backend_ranked_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENFOLD_FAKE_MODE", "ranked")
    model_dir = _prep_fake_environment(
        monkeypatch,
        tmp_path,
        {"plddt": _PLDDT, "predicted_aligned_error": _PAE, "ptm": 0.8765, "iptm": 0.8125},
    )
    backend = OpenFoldCliBackend(model_dir=model_dir)
    result = backend.run(
        ProteinTarget(_TARGET_ID, "desc", ("A",)),
        _prepared(tmp_path),
        tmp_path / "out",
    )

    raw_dir = tmp_path / "out" / "raw"
    predictions_dir = raw_dir / "predictions"
    assert (predictions_dir / f"{_OUTPUT_TAG}_unrelaxed_rank_001.pdb").exists()
    assert not (predictions_dir / f"{_OUTPUT_TAG}_{_MODEL_PRESET}_output_dict.pkl").exists()
    assert result.predictions[0].structure_path.exists()
    assert result.predictions[0].scores_path.exists()


def test_openfold_cli_backend_structure_derived_json_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENFOLD_FAKE_MODE", "json-fallback")
    spec = {
        "plddt": [81.585, 90.123, 75.0],
        "predicted_aligned_error": [[0.0, 1.234, 2.345], [1.234, 0.0, 3.456], [2.345, 3.456, 0.0]],
        "ptm": 0.8765,
        "iptm": 0.8125,
    }
    model_dir = _prep_fake_environment(monkeypatch, tmp_path, spec)
    backend = OpenFoldCliBackend(model_dir=model_dir)
    result = backend.run(
        ProteinTarget(_TARGET_ID, "desc", ("A",)),
        _prepared(tmp_path),
        tmp_path / "out",
    )

    raw_dir = tmp_path / "out" / "raw"
    predictions_dir = raw_dir / "predictions"
    assert (predictions_dir / f"{_OUTPUT_TAG}_unrelaxed_rank_001.pdb").exists()
    assert (predictions_dir / f"{_OUTPUT_TAG}_scores_rank_001.json").exists()
    assert result.predictions[0].structure_path.exists()
    scores = json.loads(result.predictions[0].scores_path.read_bytes())
    assert scores["plddt"] == [81.58, 90.12, 75.0]


def test_openfold_cli_backend_cleans_selected_nested_pickle_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = _prep_fake_environment(
        monkeypatch,
        tmp_path,
        {"plddt": _PLDDT, "predicted_aligned_error": _PAE, "ptm": 0.8765, "iptm": 0.8125},
    )
    backend = OpenFoldCliBackend(model_dir=model_dir)
    backend.run(
        ProteinTarget(_TARGET_ID, "desc", ("A",)),
        _prepared(tmp_path),
        tmp_path / "out",
    )

    raw_dir = tmp_path / "out" / "raw"
    predictions_dir = raw_dir / "predictions"
    tagged_pickle = predictions_dir / f"{_OUTPUT_TAG}_{_MODEL_PRESET}_output_dict.pkl"
    assert not tagged_pickle.exists()
    # The invented root-level name must never have been created or removed.
    assert not (raw_dir / "output_dict.pkl").exists()
    assert (predictions_dir / f"{_OUTPUT_TAG}_{_MODEL_PRESET}_unrelaxed.pdb").exists()


def test_openfold_cli_backend_missing_parameter_file_raises_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_entrypoint(bin_dir)
    monkeypatch.setenv("OPENFOLD_FAKE_SPEC", str(tmp_path / "unused.pkl"))
    monkeypatch.setenv("OPENFOLD_FAKE_MARKER", str(tmp_path / "entrypoint_ran"))
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:{os.environ['PATH']}")

    backend = OpenFoldCliBackend(model_dir=tmp_path / "does-not-exist")
    with pytest.raises(FoldingBackendError):
        backend.run(
            ProteinTarget(_TARGET_ID, "desc", ("A",)),
            _prepared(tmp_path),
            tmp_path / "out",
        )
    assert not (tmp_path / "entrypoint_ran").exists()
    assert not (tmp_path / "out").exists()


def test_openfold_cli_backend_missing_pae_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = _prep_fake_environment(monkeypatch, tmp_path, {"plddt": _PLDDT})
    backend = OpenFoldCliBackend(model_dir=model_dir)
    with pytest.raises(FoldingBackendError):
        backend.run(
            ProteinTarget(_TARGET_ID, "desc", ("A",)),
            _prepared(tmp_path),
            tmp_path / "out",
        )


def test_openfold_cli_backend_missing_chain_ids_metadata_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = _prep_fake_environment(
        monkeypatch,
        tmp_path,
        {"plddt": _PLDDT, "predicted_aligned_error": _PAE, "ptm": 0.8765, "iptm": 0.8125},
    )
    backend = OpenFoldCliBackend(model_dir=model_dir)
    prepared = PreparedInput("openfold-cli", tmp_path / "fasta", tmp_path / "align", tmp_path / "tmpl", {})
    with pytest.raises(FoldingBackendError, match="chain_ids"):
        backend.run(ProteinTarget(_TARGET_ID, "desc", ("A",)), prepared, tmp_path / "out")
