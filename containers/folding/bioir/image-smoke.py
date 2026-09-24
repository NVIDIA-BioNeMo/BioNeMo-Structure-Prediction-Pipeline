#!/usr/bin/env python3
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

"""Self-contained local smoke for the bioir kernel image composition."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

_MANIFEST = Path("/opt/bspp/folding-runtime-image.json")
_CUDA_COMPAT_DIR = Path(os.environ["BSPP_CUDA_COMPAT_DIR"])


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
        raise RuntimeError("folding bioir image manifest is not schema version 1")
    image_lock_sha256 = manifest.get("image_lock_sha256")
    if not isinstance(image_lock_sha256, str) or len(image_lock_sha256) != 64:
        raise RuntimeError("folding bioir image manifest has a malformed image_lock_sha256")
    int(image_lock_sha256, 16)
    _verify_distributions()
    _verify_artifact_evidence()
    _verify_bioir_score_producer()
    python_version = _run(("python", "--version"))
    if not python_version.startswith("Python 3.12"):
        raise RuntimeError(f"image Python is not 3.12: {python_version}")
    python_development = _verify_python_development()
    _verify_help()
    report = {
        "schema_version": 1,
        "image_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "image_lock_sha256": image_lock_sha256,
        "python_version": python_version,
        "python_development": python_development,
        "contract_version": importlib.metadata.version("bspp-orchestration-contract"),
        "runtime_version": importlib.metadata.version("bspp-orchestration-runtime"),
        "control_version": importlib.metadata.version("bspp-orchestration-control"),
        "bioir_producer_score_provenance": "passed-both-model-families-cpu-only",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _verify_python_development() -> dict[str, str | int | list[str]]:
    """Compile and import a fixed C extension without initializing CUDA."""
    include_paths = list(dict.fromkeys(sysconfig.get_path(key) for key in ("include", "platinclude")))
    if not any((Path(directory) / "Python.h").is_file() for directory in include_paths):
        raise RuntimeError(f"Python development header Python.h is absent from {include_paths}")
    compiler = shlex.split(sysconfig.get_config_var("CC") or "cc")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not compiler or not isinstance(suffix, str) or not suffix:
        raise RuntimeError("Python extension compiler/suffix configuration is missing")
    name = "_bspp_python_development_smoke"
    source = """#include <Python.h>
static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, "_bspp_python_development_smoke", NULL, -1, NULL
};
PyMODINIT_FUNC PyInit__bspp_python_development_smoke(void) {
    PyObject *module = PyModule_Create(&definition);
    if (module == NULL) return NULL;
    if (PyModule_AddIntConstant(module, "header_version_hex", PY_VERSION_HEX) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
"""
    with tempfile.TemporaryDirectory(prefix="bspp-python-development-") as directory:
        c_path = Path(directory) / f"{name}.c"
        extension = Path(directory) / f"{name}{suffix}"
        c_path.write_text(source, encoding="utf-8")
        subprocess.run(
            [
                *compiler,
                "-shared",
                "-fPIC",
                *(f"-I{path}" for path in include_paths),
                str(c_path),
                "-o",
                str(extension),
            ],
            check=True,
            timeout=60,
        )
        spec = importlib.util.spec_from_file_location(name, extension)
        if spec is None or spec.loader is None:
            raise RuntimeError("Compiled Python extension cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        header_version_hex = int(module.header_version_hex)
        if header_version_hex != sys.hexversion:
            raise RuntimeError("Python development headers do not match the running interpreter")
    return {"include_paths": include_paths, "extension_suffix": suffix, "header_version_hex": header_version_hex}


def _verify_composition() -> None:
    """Check the CUDA compat loader policy (graceful if absent)."""
    if not _CUDA_COMPAT_DIR.exists():
        return
    try:
        compat_directory = _CUDA_COMPAT_DIR.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("CUDA compatibility directory cannot be resolved") from exc
    if not compat_directory.is_dir() or not os.access(compat_directory, os.R_OK | os.X_OK):
        raise RuntimeError("CUDA compatibility directory is not readable")
    compat_library = _CUDA_COMPAT_DIR / "libcuda.so.1"
    if not compat_library.is_symlink():
        raise RuntimeError("CUDA compatibility soname is not a symlink")
    compat_target = compat_library.resolve(strict=True)
    if not compat_target.is_file() or not os.access(compat_target, os.R_OK):
        raise RuntimeError("CUDA compatibility target is not a readable regular file")
    if not compat_target.is_relative_to(compat_directory):
        raise RuntimeError("CUDA compatibility target escapes its directory")


def _verify_distributions() -> None:
    importlib.metadata.version("bspp-orchestration-control")
    importlib.metadata.version("bspp-orchestration-contract")
    importlib.metadata.version("bspp-orchestration-runtime")
    import bionemo_ir  # noqa: F401
    import click  # noqa: F401
    import pydantic  # noqa: F401
    import yaml  # noqa: F401


def _verify_help() -> None:
    subprocess.run(
        ["bspp-orchestration-runtime", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )


def _run(argv: tuple[str, ...], *, first_line: bool = False) -> str:
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    value = (result.stdout or result.stderr).strip()
    return value.splitlines()[0] if first_line else value


if __name__ == "__main__":
    main()
