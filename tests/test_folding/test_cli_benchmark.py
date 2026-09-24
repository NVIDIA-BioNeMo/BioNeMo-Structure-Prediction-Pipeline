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

"""Tests for the cluster-side ``benchmark validate-run`` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.data_movement.common import ToolMissingError
from bspp.orchestration.runtime.data_movement.s3.client import MissingS3CredentialsError


def test_validate_run_help() -> None:
    result = CliRunner().invoke(cli, ["benchmark", "validate-run", "--help"])
    assert result.exit_code == 0
    assert "--run-dir" in result.output
    assert "--corpus" in result.output
    assert "recursively" in result.output


def test_validate_run_requires_arguments() -> None:
    result = CliRunner().invoke(cli, ["benchmark", "validate-run"])
    assert result.exit_code != 0


def _fake_summary(
    run_dir: Path,
    suite_path: Path,
    index_path: Path,
    corpus_dir: Path,
    *,
    output_dir: Path | None = None,
    expected_fingerprint: str | None = None,
) -> dict[str, object]:
    return {
        "case_count": 1,
        "passed_count": 1,
        "cases": [{"target_id": "t1", "passed": True}],
    }


def test_validate_run_success_bounded_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    suite = tmp_path / "suite.json"
    suite.write_text("{}", encoding="utf-8")
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "bspp.orchestration.runtime.folding.benchmark.corpus.fetch_pinned_corpus",
        lambda s3_location, destination, *, credentials=None, expected_fingerprint: destination,
    )
    monkeypatch.setattr(
        "bspp.orchestration.runtime.folding.benchmark.validation.validate_run",
        _fake_summary,
    )

    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "validate-run",
            "--run-dir",
            str(run_dir),
            "--suite",
            str(suite),
            "--index",
            str(index),
            "--corpus",
            "s3://benchmarks/pdb-temporal-2022-2025-v1/",
            "--fingerprint",
            "a" * 64,
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["case_count"] == 1
    assert payload["passed_count"] == 1
    assert payload["cases"] == [{"target_id": "t1", "passed": True}]


def test_validate_run_forwards_verified_corpus_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, suite, index = _invoke_validate_run(tmp_path)
    captured: dict[str, object] = {}

    def fake_validate(
        run_dir: Path,
        suite_path: Path,
        index_path: Path,
        corpus_dir: Path,
        *,
        output_dir: Path | None = None,
        expected_fingerprint: str | None = None,
    ) -> dict[str, object]:
        captured["expected_fingerprint"] = expected_fingerprint
        return _fake_summary(run_dir, suite_path, index_path, corpus_dir, output_dir=output_dir)

    monkeypatch.setattr(
        "bspp.orchestration.runtime.folding.benchmark.corpus.fetch_pinned_corpus",
        lambda s3_location, destination, *, credentials=None, expected_fingerprint: destination,
    )
    monkeypatch.setattr(
        "bspp.orchestration.runtime.folding.benchmark.validation.validate_run",
        fake_validate,
    )

    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "validate-run",
            "--run-dir",
            str(run_dir),
            "--suite",
            str(suite),
            "--index",
            str(index),
            "--corpus",
            "s3://benchmarks/pdb-temporal-2022-2025-v1/",
            "--fingerprint",
            "b" * 64,
        ],
    )

    assert result.exit_code == 0
    assert captured["expected_fingerprint"] == "b" * 64


def test_validate_run_removes_temporary_corpus_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, suite, index = _invoke_validate_run(tmp_path)
    captured: dict[str, Path] = {}

    def fake_validate(
        run_dir: Path,
        suite_path: Path,
        index_path: Path,
        corpus_dir: Path,
        *,
        output_dir: Path | None = None,
        expected_fingerprint: str | None = None,
    ) -> dict[str, object]:
        captured["corpus_dir"] = corpus_dir
        (corpus_dir / "dataset.json").write_text("{}", encoding="utf-8")
        return _fake_summary(run_dir, suite_path, index_path, corpus_dir, output_dir=output_dir)

    monkeypatch.setattr(
        "bspp.orchestration.runtime.folding.benchmark.corpus.fetch_pinned_corpus",
        lambda s3_location, destination, *, credentials=None, expected_fingerprint: destination,
    )
    monkeypatch.setattr(
        "bspp.orchestration.runtime.folding.benchmark.validation.validate_run",
        fake_validate,
    )

    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "validate-run",
            "--run-dir",
            str(run_dir),
            "--suite",
            str(suite),
            "--index",
            str(index),
            "--corpus",
            "s3://benchmarks/pdb-temporal-2022-2025-v1/",
            "--fingerprint",
            "a" * 64,
        ],
    )

    assert result.exit_code == 0
    assert not captured["corpus_dir"].exists()


def test_validate_run_removes_temporary_corpus_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, suite, index = _invoke_validate_run(tmp_path)
    captured: dict[str, Path] = {}

    def fail_fetch(
        s3_location: str,
        destination: Path,
        *,
        credentials: object = None,
        expected_fingerprint: str,
    ) -> Path:
        captured["corpus_dir"] = destination
        (destination / "dataset.json").write_text("{}", encoding="utf-8")
        raise ValueError("fingerprint mismatch")

    monkeypatch.setattr(
        "bspp.orchestration.runtime.folding.benchmark.corpus.fetch_pinned_corpus",
        fail_fetch,
    )

    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "validate-run",
            "--run-dir",
            str(run_dir),
            "--suite",
            str(suite),
            "--index",
            str(index),
            "--corpus",
            "s3://benchmarks/pdb-temporal-2022-2025-v1/",
            "--fingerprint",
            "a" * 64,
        ],
    )

    assert result.exit_code != 0
    assert "fingerprint mismatch" in result.output
    assert not captured["corpus_dir"].exists()


def _invoke_validate_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    suite = tmp_path / "suite.json"
    suite.write_text("{}", encoding="utf-8")
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")
    return run_dir, suite, index


def test_validate_run_missing_s3_credentials_is_click_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, suite, index = _invoke_validate_run(tmp_path)

    def raise_missing_credentials(*args: object, **kwargs: object) -> Path:
        raise MissingS3CredentialsError("Missing S3 credentials.")

    monkeypatch.setattr(
        "bspp.orchestration.runtime.folding.benchmark.corpus.fetch_pinned_corpus",
        raise_missing_credentials,
    )

    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "validate-run",
            "--run-dir",
            str(run_dir),
            "--suite",
            str(suite),
            "--index",
            str(index),
            "--corpus",
            "s3://benchmarks/pdb-temporal-2022-2025-v1/",
            "--fingerprint",
            "a" * 64,
        ],
    )

    assert result.exit_code != 0
    assert "Error: Missing S3 credentials." in result.output
    assert "Traceback" not in result.output


def test_validate_run_missing_tool_is_click_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, suite, index = _invoke_validate_run(tmp_path)

    def raise_missing_tool(*args: object, **kwargs: object) -> Path:
        raise ToolMissingError(tool="s5cmd", hint="")

    monkeypatch.setattr(
        "bspp.orchestration.runtime.folding.benchmark.corpus.fetch_pinned_corpus",
        raise_missing_tool,
    )

    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "validate-run",
            "--run-dir",
            str(run_dir),
            "--suite",
            str(suite),
            "--index",
            str(index),
            "--corpus",
            "s3://benchmarks/pdb-temporal-2022-2025-v1/",
            "--fingerprint",
            "a" * 64,
        ],
    )

    assert result.exit_code != 0
    assert "Error: Required CLI 's5cmd' not found on PATH." in result.output
    assert "Traceback" not in result.output
