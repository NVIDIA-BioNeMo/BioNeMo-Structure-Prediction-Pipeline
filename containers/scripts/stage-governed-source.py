#!/usr/bin/env python3
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

"""Stage the governed orchestration Source Package for Runtime Qualification.

Builds a source package from the current clean HEAD, stages the exact bytes to
the cluster governed package root over the profile transport, and writes
``source-package-identity.json`` into ``--build-dir``. Pass that record to
``bsppctl runtime qualify --source-package-identity``.

The source repository must be clean (no uncommitted changes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.source_package_staging import build_and_stage_governed_source_package
from bspp.orchestration.control.transport import default_command_runner


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Cluster Profile config path")
    parser.add_argument("--profile", required=True, help="Cluster Profile id")
    parser.add_argument("--source-repo", required=True, type=Path, help="Local orchestration checkout")
    parser.add_argument(
        "--build-dir",
        required=True,
        type=Path,
        help="Local build dir for the source package and identity record",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    profile = resolve_cluster_profile(args.profile, config_path=args.config)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    build_and_stage_governed_source_package(
        args.source_repo,
        build_dir=args.build_dir,
        target_path=args.build_dir / "source-package.tar",
        profile=profile,
        runner=default_command_runner,
    )
    identity_path = args.build_dir / "source-package-identity.json"
    print(identity_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
