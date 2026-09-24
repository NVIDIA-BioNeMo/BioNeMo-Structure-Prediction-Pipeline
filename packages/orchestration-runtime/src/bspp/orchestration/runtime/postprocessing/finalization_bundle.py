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

"""Stable runtime API and CLI for postprocessing finalization evidence."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from bspp.orchestration.runtime.postprocessing.finalization_assembly import (
    PostprocessingFinalizationBundleResult,
    publish_action09_finalization_bundle,
)
from bspp.orchestration.runtime.postprocessing.runtime_action_evidence import (
    record_successful_action_task,
    validate_postprocessing_runtime_action_evidence_aggregate,
)
from bspp.orchestration.runtime.postprocessing.scientific_output_snapshot import (
    generate_scientific_output_root,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scientific = commands.add_parser("scientific-root")
    scientific.add_argument("--phase-runspec", type=Path, required=True)
    scientific.add_argument("--workers", type=int, default=1)
    record = commands.add_parser("record-action")
    record.add_argument("--phase-runspec", type=Path, required=True)
    record.add_argument("--action-id", required=True)
    record.add_argument("--command-digest", required=True)
    record.add_argument("--scheduler-job-id", required=True)
    record.add_argument("--task-index", type=int)
    publish = commands.add_parser("publish-action09")
    publish.add_argument("--phase-runspec", type=Path, required=True)
    publish.add_argument("--execution-projection", type=Path, required=True)
    publish.add_argument("--acceptance-policy", type=Path, required=True)
    publish.add_argument("--command-digest", required=True)
    publish.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "scientific-root":
        generate_scientific_output_root(phase_runspec_path=args.phase_runspec, workers=args.workers)
    elif args.command == "record-action":
        record_successful_action_task(
            phase_runspec_path=args.phase_runspec,
            action_id=args.action_id,
            command_digest=args.command_digest,
            scheduler_job_id=args.scheduler_job_id,
            task_index=args.task_index,
        )
    else:
        publish_action09_finalization_bundle(
            phase_runspec_path=args.phase_runspec,
            execution_projection_path=args.execution_projection,
            acceptance_policy_path=args.acceptance_policy,
            command_digest=args.command_digest,
            workers=args.workers,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PostprocessingFinalizationBundleResult",
    "generate_scientific_output_root",
    "main",
    "publish_action09_finalization_bundle",
    "record_successful_action_task",
    "validate_postprocessing_runtime_action_evidence_aggregate",
]
