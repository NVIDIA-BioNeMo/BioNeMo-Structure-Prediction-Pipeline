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

"""Enable Blackwell code generation when OpenFold is built with CUDA 12.8."""

from __future__ import annotations

from pathlib import Path

path = Path("/opt/openfold/setup.py")
text = path.read_text(encoding="utf-8")
assignment_old = "_, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)\n"
assignment_new = (
    "_, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)\n"
)
guard_old = "if int(bare_metal_major) >= 13:\n"
guard_new = (
    "if int(bare_metal_major) >= 13 or "
    "(int(bare_metal_major) == 12 and int(bare_metal_minor) >= 8):\n"
)
if assignment_old not in text or guard_old not in text:
    raise SystemExit("OpenFold setup.py architecture guard changed upstream")
path.write_text(
    text.replace(assignment_old, assignment_new, 1).replace(guard_old, guard_new, 1),
    encoding="utf-8",
)
