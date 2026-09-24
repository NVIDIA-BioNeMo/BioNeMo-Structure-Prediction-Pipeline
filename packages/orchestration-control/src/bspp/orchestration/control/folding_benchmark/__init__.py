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

"""Folding benchmark tooling package anchor.

This package owns the benchmark corpus specification model and the pure PDB
assembly parser used to curate the BSPP folding benchmark. The anchor imports
nothing so the control-plane import boundary stays clean: no numpy/pyarrow and
no runtime dependency is reachable through this package.
"""

from __future__ import annotations
