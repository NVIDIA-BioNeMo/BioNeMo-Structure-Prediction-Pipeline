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

"""Enforce the SPDX header policy on every in-scope source file.

This is the durable gate that makes "newly-created files must carry the header"
mechanically true forever: any in-scope file without the header fails the suite.
The scope and header byte-exactness live in ``tests.support.spdx_headers`` so the
verifier and this test can never drift from the codemod's exclusion map.
"""

from __future__ import annotations

from pathlib import Path

from tests.support.spdx_headers import HEADER, in_scope_files, missing_headers

REPO_ROOT = Path(__file__).resolve().parent.parent

# Independent byte-exact pin of the operator's template. This literal must match
# HEADER in tests.support.spdx_headers exactly; any drift fails the gate.
EXPECTED_HEADER = (
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


def test_header_constant_is_byte_exact() -> None:
    """The header constant must match the operator's template byte-for-byte."""
    assert HEADER == EXPECTED_HEADER


def test_every_in_scope_file_carries_header() -> None:
    """Every in-scope file must begin with the SPDX header (after any shebang)."""
    missing = missing_headers(REPO_ROOT)
    assert not missing, (
        f"{len(missing)} in-scope files lack the SPDX header: {[str(p.relative_to(REPO_ROOT)) for p in missing[:20]]}"
    )


def test_scope_is_non_empty_and_excludes_frozen_files() -> None:
    """The scope must be non-empty and must never include frozen release files."""
    files = in_scope_files(REPO_ROOT)
    assert files
    rel = {p.relative_to(REPO_ROOT).as_posix() for p in files}
    assert "LICENSE" not in rel
    assert "SECURITY.md" not in rel
