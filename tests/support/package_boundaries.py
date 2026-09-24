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

"""Declared distribution boundaries for BSPP orchestration tests."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PackageSurface:
    """One physical install/import surface in the orchestration workspace."""

    name: str
    distribution_name: str
    import_roots: tuple[str, ...]
    forbidden_imports: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundaryViolation:
    """Static import boundary violation."""

    surface: str
    path: Path
    imported: str
    forbidden: str


RUNTIME_HEAVY_IMPORTS = (
    "afdb_integration_kit",
    "boto3",
    "botocore",
    "duckdb",
    "google",
    "numpy",
    "orjson",
    "pyarrow",
    "rich",
    "submitit",
    "torch",
    "zstandard",
)
RUNTIME_PACKAGE_IMPORTS = ("bspp.orchestration.runtime",)
CONTRACT_IMPORT_ROOTS = (
    "bspp.orchestration.contract",
    "bspp.orchestration.contract.config_models",
    "bspp.orchestration.contract.control_state",
    "bspp.orchestration.contract.data_placement",
    "bspp.orchestration.contract.database_set_provisioning",
    "bspp.orchestration.contract.phase",
    "bspp.orchestration.contract.phase_state",
    "bspp.orchestration.contract.provisioning",
    "bspp.orchestration.contract.runplan",
    "bspp.orchestration.contract.runspec",
    "bspp.orchestration.contract.runspec_policies",
    "bspp.orchestration.contract.runspec_validation",
    "bspp.orchestration.contract.runtime_qualification",
    "bspp.orchestration.contract.secrets",
    "bspp.orchestration.contract.versioning",
)
CONTROL_PACKAGE_IMPORTS = ("bspp.orchestration.control",)
ROOT_CONTRACT_COMPATIBILITY_IMPORTS: tuple[str, ...] = ()

PACKAGE_SURFACES = {
    "contract": PackageSurface(
        name="contract",
        distribution_name="bspp-orchestration-contract",
        import_roots=CONTRACT_IMPORT_ROOTS,
        forbidden_imports=(*CONTROL_PACKAGE_IMPORTS, *RUNTIME_PACKAGE_IMPORTS, *RUNTIME_HEAVY_IMPORTS),
    ),
    "control": PackageSurface(
        name="control",
        distribution_name="bspp-orchestration-control",
        import_roots=CONTROL_PACKAGE_IMPORTS,
        forbidden_imports=(*RUNTIME_PACKAGE_IMPORTS, *RUNTIME_HEAVY_IMPORTS),
    ),
    "runtime": PackageSurface(
        name="runtime",
        distribution_name="bspp-orchestration-runtime",
        import_roots=("bspp.orchestration.runtime",),
        forbidden_imports=CONTROL_PACKAGE_IMPORTS,
    ),
}


def check_boundary_imports(source_roots: Mapping[str, Path]) -> list[BoundaryViolation]:
    """Return static import violations for surfaces with declared forbidden imports."""
    violations: list[BoundaryViolation] = []
    for surface in PACKAGE_SURFACES.values():
        if not surface.forbidden_imports:
            continue
        source_root = source_roots[surface.name]
        for path in _surface_python_files(source_root, surface.import_roots):
            for imported in _imported_modules(path):
                forbidden = _matching_forbidden_import(imported, surface.forbidden_imports)
                if forbidden is not None:
                    violations.append(
                        BoundaryViolation(
                            surface=surface.name,
                            path=path,
                            imported=imported,
                            forbidden=forbidden,
                        )
                    )
    return violations


def _surface_python_files(source_root: Path, import_roots: Iterable[str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for import_root in import_roots:
        root_path = _module_path(source_root, import_root)
        if root_path.is_dir():
            paths.extend(sorted(path for path in root_path.rglob("*.py") if "__pycache__" not in path.parts))
        elif root_path.is_file():
            paths.append(root_path)
    return tuple(dict.fromkeys(paths))


def _module_path(source_root: Path, module: str) -> Path:
    relative = Path(*module.split("."))
    package_dir = source_root / relative
    if package_dir.is_dir():
        return package_dir
    return source_root / relative.with_suffix(".py")


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            imports.append(node.module)
    return tuple(imports)


def _matching_forbidden_import(imported: str, forbidden_imports: Iterable[str]) -> str | None:
    for forbidden in forbidden_imports:
        if imported == forbidden or imported.startswith(f"{forbidden}."):
            return forbidden
    return None


__all__ = [
    "CONTROL_PACKAGE_IMPORTS",
    "PACKAGE_SURFACES",
    "ROOT_CONTRACT_COMPATIBILITY_IMPORTS",
    "RUNTIME_HEAVY_IMPORTS",
    "RUNTIME_PACKAGE_IMPORTS",
    "BoundaryViolation",
    "PackageSurface",
    "check_boundary_imports",
]
