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

"""Keep Database Placement tests at external or explicit composition seams."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_SCOPED_TESTS = (
    *_REPOSITORY_ROOT.joinpath("tests").glob("test_preprocessing_database_*.py"),
    _REPOSITORY_ROOT / "tests/test_linux_mount_authority_snapshot.py",
    _REPOSITORY_ROOT / "tests/test_phase_finalization.py",
    _REPOSITORY_ROOT / "tests/test_preprocessing/test_cli_execution.py",
    _REPOSITORY_ROOT / "tests/test_preprocessing/test_local_execution.py",
    _REPOSITORY_ROOT / "tests/support/database_cold_replica.py",
    _REPOSITORY_ROOT / "tests/support/preprocessing_execution.py",
)

_EXPECTED_EXTERNAL_MUTATIONS = Counter(
    {
        "Path.lstat": 4,
        "ctypes.CDLL": 2,
        "ctypes.sizeof": 1,
        "fcntl.flock": 1,
        "io.open": 3,
        "os.fchmod": 3,
        "os.fstat": 4,
        "os.fstatvfs": 53,
        "os.fsync": 12,
        "os.link": 14,
        "os.listdir": 4,
        "os.open": 4,
        "os.readlink": 1,
        "os.replace": 2,
        "os.rmdir": 1,
        "os.stat": 9,
        "os.unlink": 2,
        "os.write": 2,
        "subprocess.run": 4,
    }
)

_PRODUCT_PREFIX = "bspp.orchestration"
_EXTERNAL_MODULE_NAMES = frozenset({"ctypes", "fcntl", "io", "os", "subprocess"})


class _MutationAudit(NamedTuple):
    external: Counter[str]
    forbidden: list[str]


def _resolve_expression(node: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        owner = _resolve_expression(node.value, aliases)
        return f"{owner}.{node.attr}" if owner is not None else None
    if not isinstance(node, ast.Call):
        return None

    function = _resolve_expression(node.func, aliases)
    if function in {"importlib.import_module", "import_module"} and node.args:
        module_name = node.args[0]
        if isinstance(module_name, ast.Constant) and isinstance(module_name.value, str):
            return module_name.value
    if function in {"builtins.getattr", "getattr"} and node.args:
        owner = _resolve_expression(node.args[0], aliases)
        if owner is None:
            return None
        if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str):
            return f"{owner}.*"
        return f"{owner}.{node.args[1].value}"
    if function is not None and function.startswith(_PRODUCT_PREFIX):
        return f"{function}()"
    return None


def _collect_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {
        "delattr": "builtins.delattr",
        "getattr": "builtins.getattr",
        "setattr": "builtins.setattr",
    }
    assignments: list[tuple[str, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                aliases[imported.asname or imported.name.split(".")[0]] = imported.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for imported in node.names:
                if imported.name != "*":
                    aliases[imported.asname or imported.name] = f"{node.module}.{imported.name}"
        elif isinstance(node, ast.Assign):
            assignments.extend((target.id, node.value) for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            assignments.append((node.target.id, node.value))

    for _ in range(len(assignments) + 1):
        changed = False
        for name, value in assignments:
            resolved = _resolve_expression(value, aliases)
            if resolved is not None and aliases.get(name) != resolved:
                aliases[name] = resolved
                changed = True
        if not changed:
            break
    return aliases


def _external_target(target: str | None, attribute: str) -> str | None:
    if target is None:
        return None
    parts = target.removesuffix("()").split(".")
    if parts[-1] == "Path" and "pathlib" in parts:
        return f"Path.{attribute}"
    for module_name in _EXTERNAL_MODULE_NAMES:
        if parts[-1] == module_name:
            return f"{module_name}.{attribute}"
    return None


def _is_product_target(target: str | None) -> bool:
    return target is not None and target.startswith(_PRODUCT_PREFIX)


def _diagnostic(filename: str, node: ast.AST, mutation: str) -> str:
    return f"{filename}:{getattr(node, 'lineno', 0)}: {mutation}"


def _direct_mutation_target(node: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Attribute):
        return _resolve_expression(node.value, aliases)
    if isinstance(node, ast.Subscript):
        value = _resolve_expression(node.value, aliases)
        if value is not None and value.endswith(".__dict__"):
            return value.removesuffix(".__dict__")
        if isinstance(node.value, ast.Call) and _resolve_expression(node.value.func, aliases) in {
            "builtins.vars",
            "vars",
        }:
            return _resolve_expression(node.value.args[0], aliases) if node.value.args else None
    return None


def _audit_source(source: str, *, filename: str) -> _MutationAudit:
    tree = ast.parse(source, filename=filename)
    aliases = _collect_aliases(tree)
    external: Counter[str] = Counter()
    forbidden: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = _resolve_expression(node.func, aliases)
            if function == "object.__setattr__":
                forbidden.append(_diagnostic(filename, node, function))
                continue
            if function in {"builtins.setattr", "builtins.delattr"}:
                forbidden.append(_diagnostic(filename, node, function))
                continue
            if function is not None and (
                function.endswith(".patch")
                or function.endswith(".patch.object")
                or function in {"patch", "patch.object"}
            ):
                forbidden.append(_diagnostic(filename, node, function))
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            owner = _resolve_expression(node.func.value, aliases)
            if owner not in {"monkeypatch", "mocker"}:
                continue
            method = node.func.attr
            if method in {"delattr", "setitem", "delitem"}:
                forbidden.append(_diagnostic(filename, node, f"{owner}.{method}"))
                continue
            if method != "setattr":
                continue
            if (
                len(node.args) < 2
                or not isinstance(node.args[1], ast.Constant)
                or not isinstance(node.args[1].value, str)
            ):
                forbidden.append(_diagnostic(filename, node, f"{owner}.setattr(dynamic)"))
                continue
            target = _resolve_expression(node.args[0], aliases)
            attribute = node.args[1].value
            external_name = _external_target(target, attribute)
            if external_name is not None:
                external[external_name] += 1
                continue
            rendered = f"{target or ast.unparse(node.args[0])}.{attribute}"
            forbidden.append(_diagnostic(filename, node, rendered))
            continue

        mutation_targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            mutation_targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            mutation_targets.append(node.target)
        elif isinstance(node, ast.Delete):
            mutation_targets.extend(node.targets)
        for target_node in mutation_targets:
            target = _direct_mutation_target(target_node, aliases)
            if _is_product_target(target):
                forbidden.append(_diagnostic(filename, node, ast.unparse(target_node)))

    return _MutationAudit(external=external, forbidden=forbidden)


def test_database_placement_tests_mutate_only_inventoried_external_boundaries() -> None:
    """Reject private product mutation and unreviewed external fault seams."""
    actual: Counter[str] = Counter()
    forbidden: list[str] = []
    for path in sorted(set(_SCOPED_TESTS)):
        audit = _audit_source(path.read_text(), filename=str(path.relative_to(_REPOSITORY_ROOT)))
        actual.update(audit.external)
        forbidden.extend(audit.forbidden)

    assert forbidden == []
    assert actual == _EXPECTED_EXTERNAL_MUTATIONS


@pytest.mark.parametrize(
    "source",
    [
        """
from bspp.orchestration.runtime.preprocessing import execution as product
attribute = "_utc_now"
monkeypatch.setattr(product, attribute, replacement)
""",
        """
from bspp.orchestration.runtime.preprocessing import execution as product
monkeypatch.delattr(product, "_utc_now")
""",
        """
from bspp.orchestration.runtime.preprocessing import execution as product
monkeypatch.setitem(product.__dict__, "_utc_now", replacement)
""",
        """
from bspp.orchestration.runtime.preprocessing import execution as product
monkeypatch.delitem(product.__dict__, "_utc_now")
""",
        """
from bspp.orchestration.runtime.preprocessing import execution as product
setattr(product, "_utc_now", replacement)
""",
        """
from bspp.orchestration.runtime.preprocessing import execution as product
delattr(product, "_utc_now")
""",
        """
from bspp.orchestration.runtime.preprocessing import execution as product
product._utc_now = replacement
""",
        """
from bspp.orchestration.runtime.preprocessing import execution as product
del product._utc_now
""",
        """
from unittest.mock import patch
patch("bspp.orchestration.runtime.preprocessing.execution._utc_now", replacement)
""",
        """
import unittest.mock as mock
mock.patch.object(target, "_utc_now", replacement)
""",
        """
mocker.patch("bspp.orchestration.runtime.preprocessing.execution._utc_now", replacement)
""",
        """
from importlib import import_module
product = import_module("bspp.orchestration.runtime.preprocessing.execution")
alias = product
alias.__dict__["_utc_now"] = replacement
""",
        """
from bspp.orchestration.contract.database_placement_result import DatabasePlacementResult
record = DatabasePlacementResult()
record.schema_version = 2
""",
    ],
)
def test_mutation_policy_rejects_product_mutation_bypasses(source: str) -> None:
    audit = _audit_source(source, filename="malicious_test.py")

    assert audit.external == Counter()
    assert audit.forbidden


def test_mutation_policy_counts_only_literal_inventoried_external_boundaries() -> None:
    audit = _audit_source(
        """
import os
monkeypatch.setattr(os, "open", replacement)
""",
        filename="external_test.py",
    )

    assert audit == _MutationAudit(external=Counter({"os.open": 1}), forbidden=[])
