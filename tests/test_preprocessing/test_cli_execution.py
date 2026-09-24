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

"""Click compatibility for privately composed preprocessing coordinators."""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from bspp.orchestration.runtime.cli import (
    PRODUCTION_EXECUTION_COORDINATOR as PRODUCTION_CLI_EXECUTION_COORDINATOR,
)
from bspp.orchestration.runtime.cli import (
    ExecutionCoordinator as CliExecutionCoordinator,
)
from bspp.orchestration.runtime.cli import (
    _build_cli,
    _production_execute_chunk,
    _production_load_carry_forward_record,
    _production_load_phase_runspec,
    _production_place_database,
    cli,
)
from bspp.orchestration.runtime.preprocessing._database_placement_paths import (
    PRODUCTION_DATABASE_PLACEMENT_PATHS,
)
from bspp.orchestration.runtime.preprocessing.execution import (
    PRODUCTION_EXECUTION_COORDINATOR,
    PRODUCTION_PREPROCESSING_ACTION_EVIDENCE_STORE,
    PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER,
    PreprocessingExecutionError,
)

_EXECUTE_HELP = """Usage: cli preprocessing execute-chunk [OPTIONS]

  Execute one exact chunk without lifecycle or scheduler ownership.

Options:
  --phase-runspec FILE            Immutable preprocessing Phase RunSpec
                                  JSON/YAML.  [required]
  --action-id TEXT                Exact sole Runtime Action id.  [required]
  --write-evidence FILE           Exclusive destination for structured action
                                  evidence.  [required]
  --database-placement-result FILE
                                  Exclusive Database Placement Result authority
                                  path.  [required]
  --database-placement-failure FILE
                                  Classified Database Placement failure path
                                  when no Result exists.
  --placement-process-status INTEGER RANGE
                                  Exact exit status captured from the Database
                                  Placement command.  [0<=x<=255; required]
  --carry-forward-record FILE     Canonical staged Attempt carry-forward record.
  --phase-submission-id TEXT      Exact carried Phase Submission identity bound
                                  by the workspace sentinel.
  --help                          Show this message and exit.
"""

_PLACE_HELP = """Usage: cli preprocessing place-database [OPTIONS]

  Validate or populate the policy-selected database before science starts.

Options:
  --phase-runspec FILE           Immutable preprocessing Phase RunSpec
                                 JSON/YAML.  [required]
  --action-id TEXT               Exact sole Runtime Action id.  [required]
  --source-manifest FILE         Canonical staged Database Source Manifest.
                                 [required]
  --write-result FILE            Exclusive immutable Database Placement Result
                                 destination.  [required]
  --write-failure-evidence FILE  Exclusive immutable classified Database
                                 Placement failure destination.  [required]
  --help                         Show this message and exit.
"""


@dataclass(frozen=True)
class _MappedEvidence:
    payload: dict[str, object]

    def to_mapping(self) -> dict[str, object]:
        return self.payload


@dataclass
class _Recorder:
    runspec: object = field(default_factory=object)
    carry_record: object = field(default_factory=object)
    calls: list[tuple[str, object, dict[str, object]]] = field(default_factory=list)
    execute_result: _MappedEvidence = field(
        default_factory=lambda: _MappedEvidence({"recorded_action_evidence": {"status": "ok"}})
    )
    placement_result: _MappedEvidence = field(
        default_factory=lambda: _MappedEvidence({"recorded_database_placement": {"status": "ok"}})
    )
    execution_error: Exception | None = None

    def load_phase_runspec(self, path: Path) -> Any:
        self.calls.append(("load-runspec", path, {}))
        return self.runspec

    def load_carry_forward_record(self, path: Path) -> Any:
        self.calls.append(("load-carry", path, {}))
        return self.carry_record

    def execute_chunk(self, runspec: object, **kwargs: object) -> Any:
        self.calls.append(("execute", runspec, kwargs))
        if self.execution_error is not None:
            raise self.execution_error
        return self.execute_result

    def place_database(self, runspec: object, **kwargs: object) -> Any:
        self.calls.append(("place", runspec, kwargs))
        return self.placement_result

    def coordinator(self) -> CliExecutionCoordinator:
        return CliExecutionCoordinator(
            load_phase_runspec=self.load_phase_runspec,
            load_carry_forward_record=self.load_carry_forward_record,
            execute_chunk=self.execute_chunk,
            place_database=self.place_database,
        )


def _execute_args(tmp_path: Path) -> tuple[list[str], dict[str, Path]]:
    paths = {
        "runspec": tmp_path / "phase-runspec.json",
        "evidence": tmp_path / "action-evidence.json",
        "result": tmp_path / "database-placement-result.json",
        "failure": tmp_path / "database-placement-failure.json",
        "carry": tmp_path / "attempt-carry-forward.json",
    }
    paths["runspec"].write_text("{}\n")
    paths["carry"].write_text("{}\n")
    return (
        [
            "preprocessing",
            "execute-chunk",
            "--phase-runspec",
            str(paths["runspec"]),
            "--action-id",
            "preprocessing-chunk-000123",
            "--write-evidence",
            str(paths["evidence"]),
            "--database-placement-result",
            str(paths["result"]),
            "--database-placement-failure",
            str(paths["failure"]),
            "--placement-process-status",
            "17",
            "--carry-forward-record",
            str(paths["carry"]),
            "--phase-submission-id",
            "phase-submission-" + "1" * 64,
        ],
        paths,
    )


def test_recording_execute_chunk_forwards_every_argument_and_preserves_success_bytes(tmp_path: Path) -> None:
    recorder = _Recorder()
    args, paths = _execute_args(tmp_path)

    result = CliRunner().invoke(_build_cli(recorder.coordinator()), args)

    assert result.exit_code == 0, result.output
    assert result.stdout == json.dumps(recorder.execute_result.to_mapping(), indent=2, sort_keys=True) + "\n"
    assert result.stderr == ""
    assert recorder.calls == [
        ("load-runspec", paths["runspec"], {}),
        ("load-carry", paths["carry"], {}),
        (
            "execute",
            recorder.runspec,
            {
                "action_id": "preprocessing-chunk-000123",
                "evidence_path": paths["evidence"],
                "database_placement_result_path": paths["result"],
                "database_placement_failure_path": paths["failure"],
                "placement_process_status": 17,
                "carry_forward_record": recorder.carry_record,
                "phase_submission_id": "phase-submission-" + "1" * 64,
            },
        ),
    ]


def test_recording_place_database_forwards_every_argument_and_preserves_success_bytes(tmp_path: Path) -> None:
    recorder = _Recorder()
    runspec = tmp_path / "phase-runspec.json"
    runspec.write_text("{}\n")
    paths = {
        "manifest": tmp_path / "database-source-manifest.json",
        "result": tmp_path / "database-placement-result.json",
        "failure": tmp_path / "database-placement-failure.json",
    }

    result = CliRunner().invoke(
        _build_cli(recorder.coordinator()),
        [
            "preprocessing",
            "place-database",
            "--phase-runspec",
            str(runspec),
            "--action-id",
            "preprocessing-chunk-000123",
            "--source-manifest",
            str(paths["manifest"]),
            "--write-result",
            str(paths["result"]),
            "--write-failure-evidence",
            str(paths["failure"]),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == json.dumps(recorder.placement_result.to_mapping(), indent=2, sort_keys=True) + "\n"
    assert result.stderr == ""
    assert recorder.calls == [
        ("load-runspec", runspec, {}),
        (
            "place",
            recorder.runspec,
            {
                "action_id": "preprocessing-chunk-000123",
                "source_manifest_path": paths["manifest"],
                "result_path": paths["result"],
                "failure_path": paths["failure"],
            },
        ),
    ]


def test_execute_chunk_preserves_click_exception_exit_message_and_streams(tmp_path: Path) -> None:
    recorder = _Recorder(execution_error=PreprocessingExecutionError("recorded execution failure"))
    args, _ = _execute_args(tmp_path)

    result = CliRunner().invoke(_build_cli(recorder.coordinator()), args)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: recorded execution failure\n"


def test_preprocessing_cli_help_is_byte_compatible_and_exposes_no_composition_override() -> None:
    recorder_app = _build_cli(_Recorder().coordinator())
    runner = CliRunner()

    for application in (cli, recorder_app):
        assert runner.invoke(application, ["preprocessing", "execute-chunk", "--help"]).output == _EXECUTE_HELP
        assert runner.invoke(application, ["preprocessing", "place-database", "--help"]).output == _PLACE_HELP


def _bound_coordinator(application: Any, command_name: str) -> object:
    preprocessing = application.commands["preprocessing"]
    callback = preprocessing.commands[command_name].callback
    assert callback is not None
    return inspect.getclosurevars(callback).nonlocals["coordinator"]


def test_exported_cli_and_runtime_execution_use_exact_production_composition() -> None:
    expected_cli_coordinator = CliExecutionCoordinator(
        load_phase_runspec=_production_load_phase_runspec,
        load_carry_forward_record=_production_load_carry_forward_record,
        execute_chunk=_production_execute_chunk,
        place_database=_production_place_database,
    )
    assert expected_cli_coordinator == PRODUCTION_CLI_EXECUTION_COORDINATOR
    assert _bound_coordinator(cli, "execute-chunk") is PRODUCTION_CLI_EXECUTION_COORDINATOR
    assert _bound_coordinator(cli, "place-database") is PRODUCTION_CLI_EXECUTION_COORDINATOR
    assert PRODUCTION_EXECUTION_COORDINATOR.database_placement_paths is PRODUCTION_DATABASE_PLACEMENT_PATHS
    assert PRODUCTION_EXECUTION_COORDINATOR.clock.__name__ == "_utc_now"
    assert PRODUCTION_EXECUTION_COORDINATOR.sleeper is time.sleep
    assert PRODUCTION_EXECUTION_COORDINATOR.scientific_kernel_launcher is PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER
    assert PRODUCTION_EXECUTION_COORDINATOR.evidence_store is PRODUCTION_PREPROCESSING_ACTION_EVIDENCE_STORE
