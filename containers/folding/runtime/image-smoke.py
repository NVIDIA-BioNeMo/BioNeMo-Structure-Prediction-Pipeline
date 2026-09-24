#!/opt/bspp/environment/bin/python
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

"""Self-contained local smoke for the folding runtime image composition.

This is a composition smoke only: it proves the pinned CUDA compatibility
loader, the locked toolchain executables, the baked Contract/Control/Runtime wheels,
the presence of the Control distribution, and the embedded image manifest. It
deliberately performs no CUDA init, no GPU work, no nested container, and no
scientific folding inference — the folding adapter owns that gate.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_MANIFEST = Path("/opt/bspp/folding-runtime-image.json")
_CUDA_COMPAT_DIR = Path("/usr/local/cuda-12.6/compat")
_CUDA_COMPAT_LIBRARY = _CUDA_COMPAT_DIR / "libcuda.so.1"
_LOCKED_EXECUTABLES = (
    Path("/opt/bspp/environment/bin/python"),
    Path("/usr/bin/rsync"),
    Path("/usr/bin/tar"),
    Path("/usr/bin/lz4"),
    Path("/usr/bin/flock"),
    Path("/usr/bin/s5cmd"),
)


def _verify_artifact_evidence() -> None:
    """Exercise installed v2 score parsing on one synthetic residue, without inference."""
    from bspp.orchestration.contract.folding_artifact_evidence import ARTIFACT_EVIDENCE_PROFILE
    from bspp.orchestration.runtime.folding.artifact_evidence import read_scores

    if ARTIFACT_EVIDENCE_PROFILE != "artifact-backed-v2":
        raise RuntimeError("artifact evidence profile is not installed")
    with tempfile.TemporaryDirectory(prefix="bspp-artifact-smoke-") as directory:
        root = Path(directory)
        path = root / "synthetic-score.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "plddt": [80.0],
                    "pae": [[0.1]],
                    "max_pae": 0.1,
                    "ptm": 0.8,
                    "iptm": None,
                    "bioir_model_source": "openfold2_ptm_1",
                }
            )
        )
        scores, identity = read_scores(path, root=root, length=1, model_source="openfold2_ptm_1")
        if scores.pae != ((0.1,),) or identity.sha256 != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError("artifact evidence content/byte smoke failed")


def _verify_bioir_score_producer() -> None:
    """Run the installed producer/reader boundary with a CPU-only fake processor."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from bspp.orchestration.runtime.folding.artifact_evidence import read_scores
    from bspp.orchestration.runtime.folding.execution.bioir_config import OpenFoldModelSettings, OpenFoldSettings
    from bspp.orchestration.runtime.folding.execution.bioir_session import BioIRFoldSession
    from bspp.orchestration.runtime.folding.execution.models import ProteinTarget

    with tempfile.TemporaryDirectory(prefix="bspp-bioir-producer-smoke-") as directory:
        root = Path(directory)
        checkpoint = root / "synthetic.pt"
        checkpoint.write_bytes(b"CPU diagnostic: not model weights and never loaded")
        structure = root / "synthetic.pdb"
        structure.write_text("CPU DIAGNOSTIC STRUCTURE\n")
        for source in ("openfold2_ptm_1", "alphafold2_multimer_1"):
            monomer = source == "openfold2_ptm_1"
            raw_scores = {
                "plddt": [80.0, 90.0],
                "pae": [[0.0, 1.0], [2.0, 0.0]],
                "ptm": 0.8,
                "iptm": None if monomer else 0.9,
            }

            class DiagnosticSession(BioIRFoldSession):
                def _build_processor(self, *args, scores=raw_scores):
                    return lambda rows: [{"output_path": str(structure), "scores": scores}]

                def _request(self, prepared_dir):
                    return None

            with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))}):
                session = DiagnosticSession(
                    OpenFoldModelSettings(source, source, "synthetic.pt", model_source=source),
                    OpenFoldSettings(),
                    checkpoint,
                    root / source,
                )
                result = session.predict(
                    ProteinTarget("pdb_smoke_assembly_1", "CPU diagnostic", ("AC",) if monomer else ("A", "C")), root
                )
            prediction = result.predictions[0]
            scores, identity = read_scores(prediction.scores_path, root=root, length=2, model_source=source)
            emitted = json.loads(prediction.scores_path.read_bytes())
            if (
                scores.extras.get("bioir_model_source") != source
                or any(emitted[key] != value for key, value in raw_scores.items())
                or prediction.structure_path.read_bytes() != structure.read_bytes()
                or identity.sha256 != hashlib.sha256(prediction.scores_path.read_bytes()).hexdigest()
            ):
                raise RuntimeError("installed BioIR producer/reader provenance smoke failed")


def main() -> None:
    _verify_composition()
    manifest_bytes = _MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("folding runtime image manifest is not schema version 1")
    image_lock_sha256 = manifest.get("image_lock_sha256")
    if not isinstance(image_lock_sha256, str) or len(image_lock_sha256) != 64:
        raise RuntimeError("folding runtime image manifest has a malformed image_lock_sha256")
    int(image_lock_sha256, 16)
    _verify_distributions()
    _verify_artifact_evidence()
    _verify_bioir_score_producer()
    python_version = _run(("/opt/bspp/environment/bin/python", "--version"))
    if not python_version.startswith("Python 3.12"):
        raise RuntimeError("image Python is not 3.12")
    folding_entrypoints = _verify_folding_entrypoints()
    report = {
        "schema_version": 1,
        "image_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "image_lock_sha256": image_lock_sha256,
        "python_version": python_version,
        "folding_entrypoints": folding_entrypoints,
        "contract_version": importlib.metadata.version("bspp-orchestration-contract"),
        "runtime_version": importlib.metadata.version("bspp-orchestration-runtime"),
        "control_version": importlib.metadata.version("bspp-orchestration-control"),
        "bioir_producer_score_provenance": "passed-both-model-families-cpu-only",
        "rsync_version": _run(("/usr/bin/rsync", "--version"), first_line=True),
        "tar_version": _run(("/usr/bin/tar", "--version"), first_line=True),
        "lz4_version": _run(("/usr/bin/lz4", "--version"), first_line=True),
        "flock_version": _run(("/usr/bin/flock", "--version"), first_line=True),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _verify_folding_entrypoints() -> dict[str, str | list[str] | bool]:
    """Reach real scalar imports and command handlers without scientific input."""
    modules = (
        "bspp.orchestration.runtime.folding.executor",
        "bspp.orchestration.runtime.folding.legacy_msa_import",
        "bspp.orchestration.runtime.folding.benchmark.index",
    )
    for name in modules:
        importlib.import_module(name)
    _run((sys.executable, "-m", modules[0], "--help"))
    with tempfile.TemporaryDirectory(prefix="bspp-folding-entrypoints-") as directory:
        handoff_root = Path(directory) / "empty-handoff"
        handoff_root.mkdir()
        output_dir = Path(directory) / "unpublished-output"
        result = subprocess.run(
            [
                str(Path(sys.executable).with_name("bspp-orchestration-runtime")),
                "folding",
                "legacy-msa-import",
                "--handoff-root",
                str(handoff_root),
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if (
            result.returncode != 1
            or "legacy MSA handoff record must be a regular file:" not in result.stderr
            or "artifact-set.json" not in result.stderr
        ):
            raise RuntimeError(f"legacy MSA import did not reject missing handoff records: {result.stderr}")
        if output_dir.exists() or output_dir.is_symlink():
            raise RuntimeError("legacy MSA import published output for an invalid handoff")
    return {
        "numpy_version": importlib.metadata.version("numpy"),
        "modules": list(modules),
        "legacy_import_missing_handoff_rejected": True,
    }


def _verify_composition() -> None:
    """Check the pinned loader policy and baked toolchain composition."""
    try:
        compat_directory = _CUDA_COMPAT_DIR.resolve(strict=True)
        compat_target = _CUDA_COMPAT_LIBRARY.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("pinned CUDA compatibility composition is missing") from exc
    if not compat_directory.is_dir() or not os.access(compat_directory, os.R_OK | os.X_OK):
        raise RuntimeError("pinned CUDA compatibility directory is not readable")
    if not _CUDA_COMPAT_LIBRARY.is_symlink():
        raise RuntimeError("pinned CUDA compatibility soname is not a symlink")
    if not compat_target.is_file() or not os.access(compat_target, os.R_OK):
        raise RuntimeError("pinned CUDA compatibility target is not a readable regular file")
    if not compat_target.is_relative_to(compat_directory):
        raise RuntimeError("pinned CUDA compatibility target escapes its directory")
    loader_entries = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if not loader_entries or loader_entries[0] != str(_CUDA_COMPAT_DIR):
        raise RuntimeError("pinned CUDA compatibility directory is not first in LD_LIBRARY_PATH")
    for executable in _LOCKED_EXECUTABLES:
        if not executable.is_file() or not os.access(executable, os.R_OK | os.X_OK):
            raise RuntimeError(f"locked executable is unavailable: {executable}")


def _verify_distributions() -> None:
    importlib.metadata.version("bspp-orchestration-control")
    importlib.metadata.version("bspp-orchestration-contract")
    importlib.metadata.version("bspp-orchestration-runtime")


def _run(argv: tuple[str, ...], *, first_line: bool = False) -> str:
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    value = (result.stdout or result.stderr).strip()
    return value.splitlines()[0] if first_line else value


if __name__ == "__main__":
    main()
