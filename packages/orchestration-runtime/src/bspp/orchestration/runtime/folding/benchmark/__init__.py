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

"""Track-C folding benchmark/validation namespace.

Holds the canonical-pair index schema, strict loader, and pure builder API
(canonical-pair index schema, strict loader, and pure builder API),
plus the validation-suite model and validate_run in later
stories. This anchor is non-executing: it imports nothing and re-exports
nothing; sibling modules are imported directly by callers.
"""

from __future__ import annotations
