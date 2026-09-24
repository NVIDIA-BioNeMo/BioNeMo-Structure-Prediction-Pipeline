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

"""Local sandbox tests for build-time baked toolkit qualification."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUALIFY_SCRIPT = ROOT / "containers" / "scripts" / "qualify-baked-toolkit.sh"


def _make_simulated_toolkit(tmp_path: Path, ipsae_source_code: str) -> Path:
    """Create a minimal toolkit tree with a compilable iPSAE suitable for local testing."""
    toolkit = tmp_path / "opt" / "afdb-toolkit"
    ipsae_dir = toolkit / "afdb_integration_kit" / "ipsae"
    ipsae_dir.mkdir(parents=True)

    (ipsae_dir / "ipsae_cpp.cpp").write_text(ipsae_source_code)
    (ipsae_dir / "Makefile").write_text(
        "ipsae_cpp: ipsae_cpp.cpp\n\t$(CXX) -std=c++17 -O0 -o ipsae_cpp ipsae_cpp.cpp\n"
    )
    # Pre-populate provenance.json
    prov = toolkit / "provenance.json"
    prov.parent.mkdir(parents=True, exist_ok=True)
    prov.write_text('{"schema_version":1,"commit":"c8f824d9fc3502d60258168358342bc7d91af886"}\n')
    return toolkit


_GOOD_IPSAE_CPP = r"""
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
int main(int argc,char** argv){
 std::string input,summary;
 for(int i=1;i<argc;i++){std::string a=argv[i];
   if(a=="--batch"&&i+1<argc)input=argv[++i];
   else if(a=="--summary"&&i+1<argc)summary=argv[++i];}
 std::string pdb,pae,meta_name;
 for(auto const& e:std::filesystem::directory_iterator(input)){
   std::ifstream f(e.path());std::string s((std::istreambuf_iterator<char>(f)),{});
   if(e.path().extension()==".pdb")pdb=s;
   else if(e.path().extension()==".json"){pae=s;meta_name=e.path().filename().string();}}
 if(pdb.find("ATOM")==std::string::npos||pdb.find(" A ")==std::string::npos||
    pdb.find(" B ")==std::string::npos||meta_name.find("-meta_v1.json")==std::string::npos||
    pae.empty()||pae[0]!='{'||pae.find("\"pae\"")==std::string::npos)return 3;
 // Real iPSAE writes a WIDE summary CSV (pdb_path,pae_cutoff,dist_cutoff,iptm_af,
 // ipsae_AB,ipsae_BA,...) and the ABSOLUTE pdb path. A position-based parser
 // would misread pae_cutoff (10.0) as ipsae_AB; the script must resolve by name.
 std::ofstream out(summary);
 out<<"pdb_path,pae_cutoff,dist_cutoff,iptm_af,ipsae_AB,ipsae_BA\n"
    <<input<<"/BSPP-RQ-PAIR-model_v1.pdb,10.0,8.0,-1.0,1.000000,1.000000\n";
 return 0;}
"""

_BAD_SCORE_IPSAE_CPP = r"""
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
int main(int argc,char** argv){
 std::string input,summary;
 for(int i=1;i<argc;i++){std::string a=argv[i];
   if(a=="--batch"&&i+1<argc)input=argv[++i];
   else if(a=="--summary"&&i+1<argc)summary=argv[++i];}
 std::string pdb,pae,meta_name;
 for(auto const& e:std::filesystem::directory_iterator(input)){
   std::ifstream f(e.path());std::string s((std::istreambuf_iterator<char>(f)),{});
   if(e.path().extension()==".pdb")pdb=s;
   else if(e.path().extension()==".json"){pae=s;meta_name=e.path().filename().string();}}
 if(pdb.find("ATOM")==std::string::npos||pdb.find(" A ")==std::string::npos||
    pdb.find(" B ")==std::string::npos||meta_name.find("-meta_v1.json")==std::string::npos||
    pae.empty()||pae[0]!='{'||pae.find("\"pae\"")==std::string::npos)return 3;
 std::ofstream out(summary);
 out<<"pdb_path,pae_cutoff,dist_cutoff,iptm_af,ipsae_AB,ipsae_BA\n"
    <<"BSPP-RQ-PAIR-model_v1.pdb,10.0,8.0,-1.0,0.999999,0.999999\n";
 return 0;}
"""

_SINGLE_COLUMN_IPSAE_CPP = r"""
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
int main(int argc,char** argv){
 std::string input,summary;
 for(int i=1;i<argc;i++){std::string a=argv[i];
   if(a=="--batch"&&i+1<argc)input=argv[++i];
   else if(a=="--summary"&&i+1<argc)summary=argv[++i];}
 std::string pdb,pae,meta_name;
 for(auto const& e:std::filesystem::directory_iterator(input)){
   std::ifstream f(e.path());std::string s((std::istreambuf_iterator<char>(f)),{});
   if(e.path().extension()==".pdb")pdb=s;
   else if(e.path().extension()==".json"){pae=s;meta_name=e.path().filename().string();}}
 if(pdb.find("ATOM")==std::string::npos||pdb.find(" A ")==std::string::npos||
    pdb.find(" B ")==std::string::npos||meta_name.find("-meta_v1.json")==std::string::npos||
    pae.empty()||pae[0]!='{'||pae.find("\"pae\"")==std::string::npos)return 3;
 std::ofstream out(summary);out<<"pdb_path,ipsae\nBSPP-RQ-PAIR-model_v1.pdb,1.000000\n";
 return 0;}
"""


@pytest.fixture(autouse=True)
def _require_gxx() -> None:
    """Skip all qualification tests if g++ is not available."""
    if subprocess.run(["which", "g++"], capture_output=True, check=False).returncode != 0:
        pytest.skip("g++ not available on PATH")


def test_qualification_accepts_correct_paired_ipsae_scores(tmp_path: Path) -> None:
    """Build-time qualification passes when iPSAE emits correct paired scores."""
    toolkit = _make_simulated_toolkit(tmp_path, _GOOD_IPSAE_CPP)

    result = subprocess.run(
        ["bash", str(QUALIFY_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "BSPP_BAKED_TOOLKIT": str(toolkit)},
        cwd=tmp_path,
    )
    assert result.returncode == 0, f"qualification failed: {result.stderr}\n{result.stdout}"


def test_qualification_rejects_wrong_ipsae_scores(tmp_path: Path) -> None:
    """Build-time qualification fails when iPSAE emits wrong decimal scores."""
    toolkit = _make_simulated_toolkit(tmp_path, _BAD_SCORE_IPSAE_CPP)

    result = subprocess.run(
        ["bash", str(QUALIFY_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "BSPP_BAKED_TOOLKIT": str(toolkit)},
        cwd=tmp_path,
    )
    assert result.returncode != 0, "qualification should have failed on wrong scores"
    combined = (result.stderr + result.stdout).lower()
    assert "mismatch" in combined or "failed" in combined


def test_qualification_rejects_single_column_output(tmp_path: Path) -> None:
    """Build-time qualification fails when iPSAE emits non-directional (single-column) output."""
    toolkit = _make_simulated_toolkit(tmp_path, _SINGLE_COLUMN_IPSAE_CPP)

    result = subprocess.run(
        ["bash", str(QUALIFY_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "BSPP_BAKED_TOOLKIT": str(toolkit)},
        cwd=tmp_path,
    )
    assert result.returncode != 0, "qualification should have rejected single-column ipsae output"


def test_qualification_rejects_missing_makefile(tmp_path: Path) -> None:
    """Build-time qualification fails when iPSAE source Makefile is absent."""
    toolkit = tmp_path / "opt" / "afdb-toolkit"
    ipsae_dir = toolkit / "afdb_integration_kit" / "ipsae"
    ipsae_dir.mkdir(parents=True)
    # No Makefile created

    result = subprocess.run(
        ["bash", str(QUALIFY_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "BSPP_BAKED_TOOLKIT": str(toolkit)},
        cwd=tmp_path,
    )
    assert result.returncode != 0, "qualification should have failed when Makefile is missing"


def test_qualification_produces_result_json_on_success(tmp_path: Path) -> None:
    """A successful qualification writes qualification-result.json."""
    import json

    toolkit = _make_simulated_toolkit(tmp_path, _GOOD_IPSAE_CPP)

    result = subprocess.run(
        ["bash", str(QUALIFY_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "BSPP_BAKED_TOOLKIT": str(toolkit)},
        cwd=tmp_path,
    )
    assert result.returncode == 0

    result_json = toolkit / "qualification-result.json"
    assert result_json.is_file()
    payload = json.loads(result_json.read_text())
    assert payload["status"] == "passed"
    assert "ipsae_sha256" in payload


def test_qualify_script_defines_expected_fixture_constants() -> None:
    """The qualification script bakes the expected fixture constants inline."""
    script_text = QUALIFY_SCRIPT.read_text()
    assert "BSPP-RQ-PAIR" in script_text
    assert "1.000000" in script_text  # expected ipsae_AB and ipsae_BA


def test_qualify_script_is_bash_without_python_dependency() -> None:
    """Qualification runs in pure bash (no python, no pip, no pixi)."""
    script_text = QUALIFY_SCRIPT.read_text()
    assert "/usr/bin/env python" not in script_text
    assert "pip install" not in script_text
    assert "pixi" not in script_text
    # Check that the shebang line is the only place 'import' could appear as a Python import
    lines = script_text.splitlines()
    for line in lines:
        if line.startswith("#!"):
            continue
        if line.strip().startswith("import "):
            raise AssertionError(f"Python import in qualification script: {line}")
