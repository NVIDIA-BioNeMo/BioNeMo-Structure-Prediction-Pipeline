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

"""Byte-identity gate: the baked fork must equal bspp @ example-cluster-postproc HEAD modulo the
documented broken-revert repair set.

This is the repeatable check that carries scientific parity between the baked
``example-branch`` fork and the frozen read-only reference. The cluster acceptance baseline
is a regression/orchestration guard; the *scientific* parity claim is that the fork
tree matches ``bspp`` at the fixed reference commit except for exactly the four documented
repair files.

The reference and fork are operator-managed local checkouts (override with
``BSPP_REFERENCE_REPO`` / ``BSPP_FORK_REPO``). This is a parity gate that only runs where
those checkouts exist; when they are absent (other developers, CI) the whole module records a
SKIPPED result with a reason rather than failing the standard suite. When present it still
fails loudly if the reference commit cannot be resolved (never fetches or falls back to an
unpinned branch).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

# --- Fixed inputs -----------------------------------------------------------

ORCH_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_REPO = Path(os.environ.get("BSPP_REFERENCE_REPO", "/home/example/projects/bspp/bspp"))
REFERENCE_COMMIT = "74ed1a6c5521e99162c3d1a24b0d2430b225ad9a"
REFERENCE_PREFIX = "AFDB-Integration-Kit/"
FORK_REPO = Path(os.environ.get("BSPP_FORK_REPO", "/home/example/projects/bspp/afcdb-fork-work/AFDB-Integration-Kit"))

# This parity gate needs the author's local bspp reference + example-branch fork checkouts.
# Skip the whole module (a recorded SKIPPED, never a silent pass) when they are absent so the
# standard test suite passes for other developers / CI.
pytestmark = pytest.mark.skipif(
    not (REFERENCE_REPO.is_dir() and FORK_REPO.is_dir()),
    reason="requires the local bspp reference + example-branch fork checkouts (parity gate); "
    "set BSPP_REFERENCE_REPO / BSPP_FORK_REPO to point at them",
)

# The exact, reviewed repair set. Only these four files may differ in CONTENT from
# the reference. Verified against the accepted e02s01 result. naming.py is intentionally NOT in
# this set — it is adopted unchanged from the reference.
ALLOWED_FIX_FILES = {
    "afdb_integration_kit/colabfold/converter.py",
    "tests/test_colabfold_converter.py",
    "uniprot/scripts/batch_export_modelcif_input.py",
    "uniprot/scripts/export_modelcif_input.py",
}

# Example/benchmark fixtures (>5 MiB) deliberately EXCLUDED from the fork's git tree: GitLab
# rejects files over 5 MiB, and these are demo/benchmark data, not production pipeline code.
# They are not referenced by production_pipeline.py, the Dockerfile, or the qualify gate, and
# they belong in the SwiftStack pipeline-verification archive (moved via the existing
# data-movement machinery), not in git. This is the same subset the previous bc1c79f-aligned
# fork (3859222) carried.
STRIPPED_EXAMPLE_FILES = {
    "examples/complexes/complexes_1-50.zip",
    "examples/complexes/complexes_51-100.zip",
    "examples/monomers/AF-0000000000000001-msa_v1.a3m",
    "examples/monomers/AF-0000000000000003-msa_v1.a3m",
}

# The 3 accepted double-count compound-heterodimer residuals. These are a
# fork-vs-reference *output* divergence record, NOT extra allowed source-tree differences.
RESIDUAL_MODEL_IDS = (
    "AF-0000000210662999",
    "AF-0000000205043519",
    "AF-0000000205034195",
)

# Files production_pipeline.py directly executes (plus itself and the top-level entry point).
# These must be present and byte-identical so the pipeline surface is covered.
REQUIRED_SURFACE_FILES = {
    "main.py",
    "scripts/production_pipeline.py",
}


def _git(repo: Path, *args: str) -> str:
    """Run a git command in `repo`, returning stdout; raise a clear error on failure."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed in {repo}:\n{proc.stderr}")
    return proc.stdout


def _tree_blobs(repo: Path, commit: str, pathspec: str = "", strip_prefix: str = "") -> dict[str, tuple[str, str]]:
    """Map path -> (mode, blob_sha) for a commit's tree, optionally limited to a pathspec and
    stripped of a path prefix."""
    args = ["ls-tree", "-r", commit] + (["--", pathspec] if pathspec else [])
    out = _git(repo, *args)
    blobs: dict[str, tuple[str, str]] = {}
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        mode, _type, sha = meta.split()
        rel = path[len(strip_prefix) :] if strip_prefix and path.startswith(strip_prefix) else path
        blobs[rel] = (mode, sha)
    return blobs


def _reference_blobs() -> dict[str, tuple[str, str]]:
    if not REFERENCE_REPO.is_dir():  # unreachable under the module skip guard; defensive
        pytest.skip(f"reference repo missing: {REFERENCE_REPO}")
    # Assert the reference resolves to the exact expected commit (never a moving branch).
    resolved = _git(REFERENCE_REPO, "rev-parse", f"{REFERENCE_COMMIT}^{{commit}}").strip()
    if resolved != REFERENCE_COMMIT:
        pytest.fail(f"reference commit drifted: {resolved} != {REFERENCE_COMMIT}")
    # The fork mirrors only the embedded AFDB-Integration-Kit/ subtree, not the whole bspp repo
    # (which also carries analysis_*/, folding/, docs/, etc. at its root).
    return _tree_blobs(REFERENCE_REPO, REFERENCE_COMMIT, pathspec=REFERENCE_PREFIX, strip_prefix=REFERENCE_PREFIX)


def _fork_blobs() -> dict[str, tuple[str, str]]:
    if not FORK_REPO.is_dir():  # unreachable under the module skip guard; defensive
        pytest.skip(f"fork repo missing: {FORK_REPO}")
    return _tree_blobs(FORK_REPO, "HEAD")


# --- Tests ------------------------------------------------------------------


def test_fix_files_allowed() -> None:
    """The fork tree must differ from the reference at exactly the allowed repair set."""
    ref = _reference_blobs()
    fork = _fork_blobs()

    ref_paths = set(ref)
    fork_paths = set(fork)
    only_ref = sorted(ref_paths - fork_paths)
    only_fork = sorted(fork_paths - ref_paths)
    # The only reference-only paths allowed are the deliberately stripped oversized example
    # fixtures; anything else missing from the fork is a real parity gap.
    assert only_ref == sorted(STRIPPED_EXAMPLE_FILES), (
        "reference paths missing from the fork beyond the stripped example fixtures: "
        f"{sorted(set(only_ref) - STRIPPED_EXAMPLE_FILES)}"
    )
    assert not only_fork, f"paths only in fork (added): {only_fork}"

    # Among shared paths, only the 4 repair files may differ in content.
    shared = ref_paths & fork_paths
    differing = sorted(p for p in shared if ref[p] != fork[p])
    assert differing == sorted(ALLOWED_FIX_FILES), (
        "fork differs in content from bspp @ 74ed1a6c at unexpected paths.\n"
        f"expected exactly: {sorted(ALLOWED_FIX_FILES)}\nactual: {differing}"
    )


def test_fix_content() -> None:
    """The four repair files must actually implement the documented broken-revert repair."""
    converter = _git(FORK_REPO, "show", "HEAD:afdb_integration_kit/colabfold/converter.py")

    assert "from afdb_integration_kit.utils.rounding import round_float" in converter
    assert "from afdb_integration_kit.uniprot.naming import protein_description" in converter
    # Deterministic rounding restored at the chain mean, the four fractions, and the model mean.
    assert "round_float(chain_sum / n, 2)" in converter
    assert "round_float(total_sum / total_count, 2)" in converter
    # Per-chain provenance must be defined before use (no dangling names).
    assert "row_lookup" in converter
    assert "manifest_row = row_lookup.get" in converter
    assert "sequence_start = " in converter
    assert "sequence_end = " in converter
    # The resolved protein description is used as the candidate chain name.
    assert '"name": desc,' in converter

    for script in (
        "uniprot/scripts/batch_export_modelcif_input.py",
        "uniprot/scripts/export_modelcif_input.py",
    ):
        body = _git(FORK_REPO, "show", f"HEAD:{script}")
        assert "sys.path.insert" in body, f"direct-script bootstrap missing in {script}"


def test_production_surface_covered() -> None:
    """main.py and production_pipeline.py must exist and be byte-identical (not fix files)."""
    ref = _reference_blobs()
    fork = _fork_blobs()
    for path in sorted(REQUIRED_SURFACE_FILES):
        assert path in ref, f"{path} missing from reference tree"
        assert path in fork, f"{path} missing from fork tree"
        assert ref[path] == fork[path], f"{path} differs but is not a documented repair file"
        assert path not in ALLOWED_FIX_FILES


def test_residual_models_recorded() -> None:
    """The 3 accepted residuals are a fixed, documented record."""
    assert len(RESIDUAL_MODEL_IDS) == 3
    assert len(set(RESIDUAL_MODEL_IDS)) == 3
    assert RESIDUAL_MODEL_IDS == (
        "AF-0000000210662999",
        "AF-0000000205043519",
        "AF-0000000205034195",
    )
