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

"""SPDX header sweep: header constant, scope matcher, verifier, and applier.

Single source of truth for the bspp-orchestration SPDX header policy. The
enforcement test (``tests/test_spdx_headers.py``) and the one-time codemod both
import from this module so the exclusion map can never drift.

Run as a CLI::

    python tests/support/spdx_headers.py --mode dry-run   # inventory + missing list
    python tests/support/spdx_headers.py --mode apply     # write headers in place
    python tests/support/spdx_headers.py --mode verify    # exit 1 if any missing
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

# Byte-exact header, verbatim from the operator. Every in-scope file must begin
# with this block (after an optional shebang) followed by one blank line.
HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
    "#\n"
    '# Licensed under the Apache License, Version 2.0 (the "License");\n'
    "# you may not use this file except in compliance with the License.\n"
    "# You may obtain a copy of the License at\n"
    "#\n"
    "# http://www.apache.org/licenses/LICENSE-2.0\n"
    "#\n"
    "# Unless required by applicable law or agreed to in writing, software\n"
    '# distributed under the License is distributed on an "AS IS" BASIS,\n'
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    "# See the License for the specific language governing permissions and\n"
    "# limitations under the License.\n"
)

_DOCKERFILE = "Dockerfile"

# In-scope: authored source/build/config only. Documentation, planning YAML,
# fixtures/data, lockfiles, and dotfiles are out of scope.
#
#   packages/   .py .toml   (source + pyproject.toml)
#   tests/      .py         (test source only; fixtures/data are non-.py)
#   containers/ .py .sh .sbatch .toml Dockerfile
#   scripts/    .py .sh .sbatch
#   lint/       .py
#   repo root   .toml       (pyproject.toml, nvidia.toml, pipelines.example.toml)
_DIR_EXTENSIONS: dict[str, frozenset[str]] = {
    "packages/": frozenset({".py", ".toml"}),
    "tests/": frozenset({".py"}),
    "containers/": frozenset({".py", ".sh", ".sbatch", ".toml"}),
    "scripts/": frozenset({".py", ".sh", ".sbatch"}),
    "lint/": frozenset({".py"}),
}

# Explicit never-touch list (defense in depth; most are already outside the
# positive scope above). LICENSE/SECURITY.md are frozen release-critical files
# byte-pinned by tests/test_release_critical_files.py.
EXCLUDED_PATHS: frozenset[str] = frozenset(
    {
        "LICENSE",
        "SECURITY.md",
        "docs/SECURITY.md",
        "uv.lock",
    }
)

EXCLUDED_PREFIXES: tuple[str, ...] = (
    "devdocs/evidence/",
    "docs/benchmarks/",
)


def _extension(name: str) -> str | None:
    if "." not in name:
        return None
    return "." + name.rsplit(".", 1)[-1]


def in_scope(rel_path: str) -> bool:
    """Return True if ``rel_path`` (git-style, forward slashes) gets a header."""
    if rel_path in EXCLUDED_PATHS:
        return False
    if rel_path.startswith(EXCLUDED_PREFIXES):
        return False
    if any(part.startswith(".") for part in rel_path.split("/")):
        return False  # dotfiles and hidden directories are out of scope

    name = rel_path.rsplit("/", 1)[-1]
    if name == _DOCKERFILE:
        return rel_path.startswith("containers/")

    ext = _extension(name)
    if ext is None:
        return False

    if "/" not in rel_path:
        return ext == ".toml"  # repo-root authored config

    for prefix, allowed in _DIR_EXTENSIONS.items():
        if rel_path.startswith(prefix):
            return ext in allowed
    return False


def tracked_files(repo_root: Path) -> list[str]:
    """Git-tracked file paths (relative, forward slashes)."""
    out = subprocess.check_output(["git", "ls-files"], cwd=repo_root, text=True)
    return out.splitlines()


def in_scope_files(repo_root: Path) -> list[Path]:
    return [repo_root / p for p in tracked_files(repo_root) if in_scope(p)]


def has_header(path: Path) -> bool:
    """Return True if ``path`` begins with HEADER (after an optional shebang), then a blank line or EOF."""
    text = path.read_text(encoding="utf-8")
    body = text
    if body.startswith("#!"):
        newline = body.find("\n")
        if newline == -1:
            return False
        body = body[newline + 1 :]
    if not body.startswith(HEADER):
        return False
    rest = body[len(HEADER) :]
    return rest == "" or rest.startswith("\n")


def missing_headers(repo_root: Path) -> list[Path]:
    return [p for p in in_scope_files(repo_root) if not has_header(p)]


def apply_header(path: Path) -> bool:
    """Insert HEADER after an optional shebang; return True if the file changed."""
    if has_header(path):
        return False
    text = path.read_text(encoding="utf-8")
    if "\x00" in text:
        return False  # defensive: never touch binary
    lines = text.split("\n")
    shebang = ""
    if lines and lines[0].startswith("#!"):
        shebang = lines[0] + "\n"
        lines = lines[1:]
    while lines and lines[0].strip() == "":
        lines.pop(0)
    body = "\n".join(lines)
    path.write_text(shebang + HEADER + "\n" + body, encoding="utf-8")
    return True


def _area(rel_path: str) -> str:
    parts = rel_path.split("/")
    if parts[0] in ("packages", "containers") and len(parts) > 1:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def _report(repo_root: Path) -> None:
    files = in_scope_files(repo_root)
    missing = missing_headers(repo_root)
    by_ext = Counter(_extension(p.name) or p.name for p in files)
    by_area = Counter(_area(p.relative_to(repo_root).as_posix()) for p in files)
    print(f"in-scope files: {len(files)}")
    print(f"missing header: {len(missing)}")
    print("by extension:", dict(sorted(by_ext.items())))
    print("by area:", dict(sorted(by_area.items())))


def _print_missing(repo_root: Path, missing: Iterable[Path], limit: int = 20) -> None:
    for p in list(missing)[:limit]:
        print(f"  {p.relative_to(repo_root)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SPDX header sweep tool")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "apply", "verify"),
        default="dry-run",
        help="dry-run: inventory + missing list (default); apply: write headers; verify: gate",
    )
    args = parser.parse_args(argv)

    if args.mode == "apply":
        changed = sum(apply_header(p) for p in in_scope_files(args.repo_root))
        print(f"headers written to {changed} files")
        return 0

    if args.mode == "verify":
        missing = missing_headers(args.repo_root)
        if missing:
            print(f"{len(missing)} in-scope files lack the SPDX header:")
            _print_missing(args.repo_root, missing, limit=50)
            return 1
        print("all in-scope files carry the SPDX header")
        return 0

    _report(args.repo_root)
    missing = missing_headers(args.repo_root)
    if missing:
        print("missing (first 20):")
        _print_missing(args.repo_root, missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
