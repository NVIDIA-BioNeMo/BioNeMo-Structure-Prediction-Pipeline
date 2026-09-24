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

"""Central toolkit resolver — single source of truth for override-first, baked-fallback selection.

Every runtime toolkit consumer routes through this module.  The resolver
validates the selected source and fails with an actionable error when it is
absent or inconsistent.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bspp.orchestration.contract.runspec import BAKED_TOOLKIT_COMMIT, RunSpec

# ---------------------------------------------------------------------------
# Baked-image constants (e01s01 / Container image contract)
# ---------------------------------------------------------------------------

BAKED_TOOLKIT_ROOT: Path = Path("/opt/afdb-toolkit")
PROVENANCE_FILENAME: str = "provenance.json"
# Canonical source is contract.runspec.BAKED_TOOLKIT_COMMIT. An internal image
# may override this at build time through ``BSPP_EXPECTED_TOOLKIT_COMMIT``;
# when the variable is absent the public pin is the fail-closed default.
EXPECTED_TOOLKIT_COMMIT: str = os.environ.get("BSPP_EXPECTED_TOOLKIT_COMMIT") or BAKED_TOOLKIT_COMMIT

REQUIRED_TOOLKIT_FILES: tuple[str, ...] = (
    "scripts/production_pipeline.py",
    "pyproject.toml",
    "ipsae_cpp",
)

OVERRIDE_MOUNT_TARGETS: tuple[Path, ...] = (
    Path("/workspace/AFDB-Integration-Kit"),
    Path("/workspace/afdb-toolkit"),
)


# ---------------------------------------------------------------------------
# Error hierarchy — every failure mode carries an actionable message.
# ---------------------------------------------------------------------------


class ToolkitResolutionError(Exception):
    """Base for all toolkit resolution failures."""


class BakedToolkitNotFoundError(ToolkitResolutionError):
    """``/opt/afdb-toolkit`` is absent from the container image."""


class BakedToolkitProvenanceError(ToolkitResolutionError):
    """Provenance file is missing, malformed, or the commit does not match."""


class BakedToolkitLayoutError(ToolkitResolutionError):
    """One or more required files are missing from the baked toolkit root."""


class ToolkitOverrideNotFoundError(ToolkitResolutionError):
    """The declared override host path is not mapped by any container mount, or the mapped path is absent."""


class ToolkitOverrideLayoutError(ToolkitResolutionError):
    """The override toolkit root is missing one or more required files."""


# ---------------------------------------------------------------------------
# ToolkitSource — validated selection result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolkitSource:
    """Resolved, validated toolkit location.

    Attributes:
        root: Absolute path to the validated toolkit directory.
        source: ``"baked"`` (immutable image layer) or ``"override"`` (operator mount).
        provenance_commit: Exact commit SHA baked into the image (always set for baked, ``None`` for override).
        production_pipeline: Resolved path to ``scripts/production_pipeline.py``.
        ipsae_path: Path to the iPSAE binary (``None`` when not required / not available).
    """

    root: Path
    source: Literal["baked", "override"]
    provenance_commit: str | None
    production_pipeline: Path
    ipsae_path: Path | None


# ---------------------------------------------------------------------------
# Runtime resolver — filesystem checks (runs inside the container)
# ---------------------------------------------------------------------------


def resolve_toolkit(spec: RunSpec) -> ToolkitSource:
    """Resolve the toolkit at runtime with full filesystem validation.

    This variant **must** run inside the container because it calls
    ``container_path_for`` and performs ``exists()`` / ``is_dir()`` /
    ``is_file()`` checks against container-scoped paths.

    Raises:
        ToolkitOverrideNotFoundError: Override declared but unmapped or absent.
        ToolkitOverrideLayoutError: Override missing a required file.
        BakedToolkitNotFoundError: Baked root ``/opt/afdb-toolkit`` absent.
        BakedToolkitProvenanceError: Provenance missing, unparseable, or wrong commit.
        BakedToolkitLayoutError: Baked root missing a required file / iPSAE not executable.
    """
    from bspp.orchestration.runtime.slurm.native_contract import (
        container_path_for,  # deferred to break circular import
    )

    override_host = spec.paths.afdb_toolkit_repo

    # ── override mode ──────────────────────────────────────────────
    if override_host is not None:
        container_path = container_path_for(spec, override_host)
        if container_path is None:
            raise ToolkitOverrideNotFoundError(
                f"Declared toolkit override {override_host} is not mapped by any container mount; "
                f"add a mount covering this host path."
            )
        if not container_path.exists() or not container_path.is_dir():
            raise ToolkitOverrideNotFoundError(
                f"Declared toolkit override maps to {container_path} but that path does not exist "
                f"or is not a directory inside the container."
            )
        missing = _missing_required_files(container_path, _override_required_files())
        if missing:
            raise ToolkitOverrideLayoutError(
                f"Toolkit override at {container_path} is missing required files: {', '.join(missing)}"
            )
        production_pipeline = container_path / "scripts" / "production_pipeline.py"
        ipsae_path = container_path / "ipsae_cpp"
        return ToolkitSource(
            root=container_path,
            source="override",
            provenance_commit=None,
            production_pipeline=production_pipeline,
            ipsae_path=ipsae_path if ipsae_path.is_file() else None,
        )

    # ── baked mode ─────────────────────────────────────────────────
    if not BAKED_TOOLKIT_ROOT.exists() or not BAKED_TOOLKIT_ROOT.is_dir():
        raise BakedToolkitNotFoundError(
            f"Baked toolkit root {BAKED_TOOLKIT_ROOT} is missing from the container image; "
            f"the image must include /opt/afdb-toolkit."
        )

    provenance_path = BAKED_TOOLKIT_ROOT / PROVENANCE_FILENAME
    if not provenance_path.is_file():
        raise BakedToolkitProvenanceError(
            f"Baked toolkit provenance file {provenance_path} is missing; "
            f"the image must write {PROVENANCE_FILENAME} at {BAKED_TOOLKIT_ROOT}."
        )

    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BakedToolkitProvenanceError(
            f"Baked toolkit provenance at {provenance_path} is unparseable: {exc}"
        ) from exc

    if not isinstance(provenance, dict):
        raise BakedToolkitProvenanceError(f"Baked toolkit provenance at {provenance_path} is not a JSON object.")

    commit = provenance.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise BakedToolkitProvenanceError(
            f"Baked toolkit provenance commit is missing or not a 40-hex sha: {commit!r} in {provenance_path}"
        )
    # Defense-in-depth: when the image declares its baked pin via the
    # un-scrubbed entrypoint env, cross-check it. Otherwise provenance.json is
    # authoritative — the image is sha256-verified, so the baked file is bound.
    env_commit = os.environ.get("BSPP_EXPECTED_TOOLKIT_COMMIT")
    if env_commit is not None and commit != env_commit:
        raise BakedToolkitProvenanceError(
            f"Baked toolkit provenance commit {commit!r} does not match "
            f"BSPP_EXPECTED_TOOLKIT_COMMIT {env_commit!r} in {provenance_path}"
        )

    missing = _missing_required_files(BAKED_TOOLKIT_ROOT, REQUIRED_TOOLKIT_FILES)
    if missing:
        raise BakedToolkitLayoutError(
            f"Baked toolkit at {BAKED_TOOLKIT_ROOT} is missing required files: {', '.join(missing)}"
        )

    ipsae_path = BAKED_TOOLKIT_ROOT / "ipsae_cpp"
    if not ipsae_path.is_file() or not os.access(str(ipsae_path), os.X_OK):
        raise BakedToolkitLayoutError(f"Baked toolkit iPSAE at {ipsae_path} is missing or not executable.")

    production_pipeline = BAKED_TOOLKIT_ROOT / "scripts" / "production_pipeline.py"
    return ToolkitSource(
        root=BAKED_TOOLKIT_ROOT,
        source="baked",
        provenance_commit=commit,
        production_pipeline=production_pipeline,
        ipsae_path=ipsae_path,
    )


# ---------------------------------------------------------------------------
# Planning resolver — no filesystem checks (runs on control plane)
# ---------------------------------------------------------------------------


def resolve_toolkit_for_container(spec: RunSpec) -> ToolkitSource:
    """Resolve the toolkit mode and container path **without** filesystem checks.

    This variant is safe to call on the control plane during SLURM script
    rendering, recipe generation, and contract building.  It still validates
    that an explicit override has a mount mapping (configuration error), but
    skips ``exists()`` / ``is_dir()`` / ``is_file()`` calls.

    Raises:
        ToolkitOverrideNotFoundError: Override declared but not covered by any container mount.
    """
    from bspp.orchestration.runtime.slurm.native_contract import (
        container_path_for,  # deferred to break circular import
    )

    override_host = spec.paths.afdb_toolkit_repo

    # ── override mode ──────────────────────────────────────────────
    if override_host is not None:
        container_path = container_path_for(spec, override_host)
        if container_path is None:
            raise ToolkitOverrideNotFoundError(
                f"Declared toolkit override {override_host} is not mapped by any container mount; "
                f"add a mount covering this host path."
            )
        production_pipeline = container_path / "scripts" / "production_pipeline.py"
        ipsae_path = container_path / "ipsae_cpp"
        return ToolkitSource(
            root=container_path,
            source="override",
            provenance_commit=None,
            production_pipeline=production_pipeline,
            ipsae_path=ipsae_path,
        )

    # ── baked mode (planning — the image's baked commit is not knowable on the
    # control plane; provenance_commit is None and only serialized, never compared) ──
    production_pipeline = BAKED_TOOLKIT_ROOT / "scripts" / "production_pipeline.py"
    ipsae_path = BAKED_TOOLKIT_ROOT / "ipsae_cpp"
    return ToolkitSource(
        root=BAKED_TOOLKIT_ROOT,
        source="baked",
        provenance_commit=None,
        production_pipeline=production_pipeline,
        ipsae_path=ipsae_path,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _missing_required_files(root: Path, required: tuple[str, ...]) -> tuple[str, ...]:
    """Return relative paths from *required* that do not exist under *root*."""
    missing: list[str] = []
    for rel in required:
        if not (root / rel).is_file():
            missing.append(rel)
    return tuple(missing)


def _override_required_files() -> tuple[str, ...]:
    """Return the subset of required files checked in override mode.

    Override mode does not require a pre-built iPSAE binary because the
    operator may provide it separately or the worker may not need it.
    """
    return ("scripts/production_pipeline.py", "pyproject.toml")


__all__ = [
    "BAKED_TOOLKIT_ROOT",
    "EXPECTED_TOOLKIT_COMMIT",
    "OVERRIDE_MOUNT_TARGETS",
    "PROVENANCE_FILENAME",
    "REQUIRED_TOOLKIT_FILES",
    "BakedToolkitLayoutError",
    "BakedToolkitNotFoundError",
    "BakedToolkitProvenanceError",
    "ToolkitOverrideLayoutError",
    "ToolkitOverrideNotFoundError",
    "ToolkitResolutionError",
    "ToolkitSource",
    "resolve_toolkit",
    "resolve_toolkit_for_container",
]
