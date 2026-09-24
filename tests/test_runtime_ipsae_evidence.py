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

"""MR2 contract tests for runtime-built iPSAE evidence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import bspp.orchestration.contract.runtime_qualification as qualification_contract
from bspp.orchestration.contract.runtime_qualification import (
    RUNTIME_IPSAE_EXPECTED_SCORE_AB,
    RUNTIME_IPSAE_EXPECTED_SCORE_BA,
    RUNTIME_IPSAE_FIXTURE_MODEL_ID,
    RUNTIME_IPSAE_FIXTURE_PAE,
    RUNTIME_IPSAE_FIXTURE_PDB,
)
from bspp.orchestration.control.runtime_qualification import _runtime_ipsae_smoke_python, qualify_runtime
from tests.runtime_ipsae_fixtures import runtime_ipsae_evidence, write_publication_compatibility_artifact
from tests.test_control_runtime_qualification import FakeSubmitRunner, _init_git_repo, _write_profiles

_RUNTIME_TOOL_PATH = "/usr/local/bin:/usr/bin:/bin"
_requires_runtime_gxx = pytest.mark.skipif(
    shutil.which("g++", path=_RUNTIME_TOOL_PATH) is None,
    reason=f"g++ unavailable on generated runtime PATH {_RUNTIME_TOOL_PATH}",
)


def test_runtime_ipsae_evidence_has_a_strict_independently_versioned_contract() -> None:
    assert hasattr(qualification_contract, "runtime_ipsae_evidence_from_mapping"), (
        "MR2 requires a strict RuntimeIpsaeEvidence parser"
    )
    parser = qualification_contract.runtime_ipsae_evidence_from_mapping
    payload = runtime_ipsae_evidence()

    parsed = parser(payload)

    assert parsed.to_mapping() == payload
    for missing in (
        "source_revision",
        "source_path",
        "build_command",
        "toolchain",
        "build_log",
        "binary",
        "version",
        "functional_test",
    ):
        invalid = dict(payload)
        invalid.pop(missing)
        try:
            parser(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"RuntimeIpsaeEvidence accepted missing {missing}")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["build_result"].update(returncode=1),
        lambda value: value["build_result"].update(argv=["make"]),
        lambda value: value["functional_test"].update(returncode=1),
        lambda value: value["functional_test"].update(argv=["true"]),
        lambda value: value["functional_test"].update(fixture_sha256="d" * 64),
        lambda value: value["functional_test"].update(status="unknown"),
        lambda value: value["functional_test"].update(ipsae_ab="0.500000"),
        lambda value: value["functional_test"].update(ipsae_ba="0.500000"),
        lambda value: value["binary"].update(path="/tmp/ipsae_cpp"),
        lambda value: value["version"].update(binary_sha256="f" * 64),
        lambda value: value["version"].update(output="arbitrary"),
        lambda value: value["toolchain"].update(extra=value["toolchain"]["make"]),
        lambda value: value["toolchain"]["cxx"].update(argv=["c++", "--version"]),
        lambda value: value["toolchain"]["make"].update(stdout=""),
        lambda value: value["build_log"].update(sha256="A" * 64),
        lambda value: value.update(format_version=2),
        lambda value: value.update(extra="not allowed"),
    ],
)
def test_runtime_ipsae_evidence_fails_closed_on_unsuccessful_or_ambiguous_evidence(mutation: object) -> None:
    payload = deepcopy(runtime_ipsae_evidence())
    mutation(payload)  # type: ignore[operator]
    with pytest.raises(ValueError):
        qualification_contract.runtime_ipsae_evidence_from_mapping(payload)


def test_runtime_qualification_builds_and_tests_ipsae_inside_the_selected_image(tmp_path: Path) -> None:
    source_repo = _init_git_repo(tmp_path / "source")
    config_path = _write_profiles(tmp_path)

    record = qualify_runtime(
        profile_name="example-cluster",
        config_path=config_path,
        source_repo=source_repo,
        runner=FakeSubmitRunner(["8801"]),
    )

    payload = json.loads(record.path.read_text())
    assert "toolkit_ipsae_sha256" not in json.dumps(payload, sort_keys=True)
    assert set(payload["tuple"]) >= {
        "source_package_identity",
        "toolkit_package_identity",
        "image_identity",
    }
    assert "runtime_ipsae" not in payload["tuple"]
    script = record.script_path.read_text()
    assert "toolkit_ipsae_sha256" not in script
    assert "/run/bspp/source-package.tar:ro" in script
    assert "/run/bspp/toolkit-package.tar:ro" in script
    assert "runtime-ipsae/build.log" in script
    assert "functional_test" in script
    assert "runtime_ipsae" in script
    qualification_program = _runtime_ipsae_smoke_python()
    assert "build_command=['make','-B','-C',str(source),'CXX=g++']" in qualification_program
    assert "built.unlink()" in qualification_program
    assert "version=run([str(binary),'--version'])" not in qualification_program
    assert "/usr/bin/g++" not in script
    assert "/usr/bin/c++" not in script


def _run_representative_runtime_ipsae_smoke(
    tmp_path: Path,
    *,
    emitted_ab: str = RUNTIME_IPSAE_EXPECTED_SCORE_AB,
    emitted_ba: str = RUNTIME_IPSAE_EXPECTED_SCORE_BA,
    metadata_suffix: str = "-meta_v1.json",
    pae_text: str = RUNTIME_IPSAE_FIXTURE_PAE,
    directional_columns: bool = True,
) -> subprocess.CompletedProcess[str]:
    toolkit = tmp_path / "private-toolkit"
    ipsae = toolkit / "afdb_integration_kit" / "ipsae"
    ipsae.mkdir(parents=True)
    (ipsae / "ipsae_cpp.cpp").write_text(
        "#include <filesystem>\n#include <fstream>\n#include <iostream>\n#include <string>\n"
        "int main(int argc,char** argv){\n"
        " std::string input,summary;\n"
        " for(int i=1;i<argc;i++){std::string a=argv[i];"
        'if(a=="--batch"&&i+1<argc)input=argv[++i];'
        'else if(a=="--summary"&&i+1<argc)summary=argv[++i];}\n'
        " std::string pdb,pae,meta_name; for(auto const& e:std::filesystem::directory_iterator(input)){"
        "std::ifstream f(e.path());std::string s((std::istreambuf_iterator<char>(f)),{});"
        'if(e.path().extension()==".pdb")pdb=s;else if(e.path().extension()==".json"){'
        "std::string fn=e.path().filename().string();"
        'if(fn.find("' + metadata_suffix + '")!=std::string::npos){meta_name=fn;if(pae.empty())pae=s;}else pae=s;}}\n'
        ' if(pdb.find("ATOM")==std::string::npos||pdb.find(" A ")==std::string::npos||'
        'pdb.find(" B ")==std::string::npos||meta_name.find("-meta_v1.json")==std::string::npos||'
        'pae.empty()||pae[0]!=\'{\'||pae.find("\\"pae\\"")==std::string::npos)return 3;\n'
        ' std::ofstream out(summary);out<<"'
        + (
            "pdb_path,ipsae_AB,ipsae_BA\\nBSPP-RQ-PAIR-model_v1.pdb," + emitted_ab + "," + emitted_ba
            if directional_columns
            else "pdb_path,ipsae\\nBSPP-RQ-PAIR-model_v1.pdb," + emitted_ab
        )
        + '\\n";\n'
        " return 0;}\n"
    )
    prebuilt = ipsae / "ipsae_cpp"
    prebuilt.write_bytes(b"prebuilt-must-not-be-reused\n")
    future = 2_000_000_000
    os.utime(prebuilt, (future, future))
    (ipsae / "Makefile").write_text(
        "ipsae_cpp: ipsae_cpp.cpp\n\t$(CXX) -std=c++17 -O0 -o ipsae_cpp ipsae_cpp.cpp\n\t@echo fresh-compile-marker\n"
    )
    result_path = tmp_path / "result" / "smoke.json"
    result_path.parent.mkdir()
    gpu = tmp_path / "nvidia-smi"
    gpu.write_text("#!/bin/sh\necho synthetic-gpu\n")
    gpu.chmod(0o755)
    bootstrap = tmp_path / "execution_bootstrap.py"
    bootstrap.write_text("print('synthetic-bootstrap')\n")
    write_publication_compatibility_artifact(result_path)
    compatibility_file = result_path.parent / "publication-compatibility.json"
    revision = "a" * 40
    bound = {
        "source_package_identity": {"marker": "source"},
        "toolkit_package_identity": {"commit": revision},
        "image_identity": {"marker": "image"},
    }

    return subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-c",
            _runtime_ipsae_smoke_python(),
            str(result_path),
            "b" * 64,
            json.dumps(bound),
            "c" * 32,
            str(toolkit),
            "override",
            revision,
            str(gpu),
            RUNTIME_IPSAE_FIXTURE_MODEL_ID,
            RUNTIME_IPSAE_FIXTURE_PDB,
            pae_text,
            metadata_suffix,
            RUNTIME_IPSAE_EXPECTED_SCORE_AB,
            RUNTIME_IPSAE_EXPECTED_SCORE_BA,
            str(bootstrap),
            str(compatibility_file),
        ),
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": _RUNTIME_TOOL_PATH},
    )


@_requires_runtime_gxx
def test_runtime_ipsae_smoke_forces_rebuild_and_checks_real_paired_inputs(tmp_path: Path) -> None:
    completed = _run_representative_runtime_ipsae_smoke(tmp_path)

    assert completed.returncode == 0, completed.stderr
    result_path = tmp_path / "result" / "smoke.json"
    payload = json.loads(result_path.read_text())
    evidence = qualification_contract.runtime_ipsae_evidence_from_mapping(payload["runtime_ipsae"])
    assert "fresh-compile-marker" in evidence.build_result.stdout
    assert evidence.binary.sha256 != __import__("hashlib").sha256(b"prebuilt-must-not-be-reused\n").hexdigest()
    assert evidence.functional_test.model_id == RUNTIME_IPSAE_FIXTURE_MODEL_ID
    assert evidence.functional_test.ipsae_ab == RUNTIME_IPSAE_EXPECTED_SCORE_AB
    assert evidence.functional_test.ipsae_ba == RUNTIME_IPSAE_EXPECTED_SCORE_BA


@_requires_runtime_gxx
def test_runtime_ipsae_smoke_rejects_wrong_semantic_calculation(tmp_path: Path) -> None:
    completed = _run_representative_runtime_ipsae_smoke(tmp_path, emitted_ab="0.500000")

    assert completed.returncode != 0
    assert "functional semantic result mismatch" in completed.stderr
    assert not (tmp_path / "result" / "smoke.json").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metadata_suffix": "-predicted_aligned_error_v1.json"},
        {"pae_text": '[{"pae":[[0.0]]}]\n'},
        {"directional_columns": False},
        {"emitted_ba": "0.500000"},
    ],
)
@_requires_runtime_gxx
def test_runtime_ipsae_smoke_rejects_non_toolkit_fixture_or_directional_output(
    tmp_path: Path, kwargs: dict[str, object]
) -> None:
    completed = _run_representative_runtime_ipsae_smoke(tmp_path, **kwargs)  # type: ignore[arg-type]

    assert completed.returncode != 0
    assert not (tmp_path / "result" / "smoke.json").exists()
