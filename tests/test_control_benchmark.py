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

"""CliRunner tests for the ``bsppctl prepare-benchmark`` and ``validate-run`` commands."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.folding_benchmark_submit import (
    BenchmarkValidateResult,
    _summary_accepted,
    render_benchmark_validate_submission,
    submit_benchmark_validation,
)
from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.transport import CommandResult


def _write_spec(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "test-benchmark",
                "selection_seed": "x",
                "source": {},
                "filters": {},
                "strata": [{"name": "monomer_short", "chain_count": 1, "total_residues": 100, "count": 1}],
                "throughput_subset_sizes": [10],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_prepare_benchmark_help() -> None:
    result = CliRunner().invoke(cli, ["prepare-benchmark", "--help"])

    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--spec" in result.output
    assert "--workers" in result.output


def test_prepare_benchmark_requires_output(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path / "spec.json")

    result = CliRunner().invoke(cli, ["prepare-benchmark", "--spec", str(spec_path)])

    assert result.exit_code != 0
    assert "Missing option '--output'" in result.output


def test_prepare_benchmark_reconstruct_option_in_help() -> None:
    result = CliRunner().invoke(cli, ["prepare-benchmark", "--help"])
    assert result.exit_code == 0
    assert "--reconstruct" in result.output


def test_prepare_benchmark_reconstruct_mocked_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec_path = _write_spec(tmp_path / "spec.json")
    target_list = tmp_path / "targets.jsonl"
    target_list.write_text("", encoding="utf-8")
    summary = {
        "dataset_id": "test-benchmark",
        "fingerprint": "f" * 64,
        "target_count": 1,
        "files": {"fasta": "targets.fasta"},
    }

    def fake_reconstruct(
        output: Path, specification: Any, target_list: Any, *, workers: int = 12, progress: Any = None
    ) -> dict[str, Any]:
        return summary

    monkeypatch.setattr(
        "bspp.orchestration.control.folding_benchmark.curator.reconstruct_benchmark_dataset",
        fake_reconstruct,
    )

    result = CliRunner().invoke(
        cli,
        [
            "prepare-benchmark",
            "--spec",
            str(spec_path),
            "--output",
            str(tmp_path / "out"),
            "--reconstruct",
            str(target_list),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == summary


def test_prepare_benchmark_mocked_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec_path = _write_spec(tmp_path / "spec.json")
    summary = {
        "dataset_id": "test-benchmark",
        "fingerprint": "f" * 64,
        "target_count": 1,
        "files": {"fasta": "targets.fasta"},
    }

    def fake(output: Path, specification: Any, *, workers: int = 12, progress: Any = None) -> dict[str, Any]:
        return summary

    monkeypatch.setattr(
        "bspp.orchestration.control.folding_benchmark.curator.prepare_benchmark_dataset",
        fake,
    )

    result = CliRunner().invoke(
        cli,
        ["prepare-benchmark", "--spec", str(spec_path), "--output", str(tmp_path / "out")],
    )

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert json.loads(result.output) == summary
    assert result.output == json.dumps(summary, sort_keys=True)


def test_prepare_benchmark_pyarrow_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bspp.orchestration.control.folding_benchmark.curator import PyArrowUnavailable

    spec_path = _write_spec(tmp_path / "spec.json")
    output_path = tmp_path / "out"

    def raise_pyarrow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise PyArrowUnavailable("pyarrow not installed")

    monkeypatch.setattr(
        "bspp.orchestration.control.folding_benchmark.curator.prepare_benchmark_dataset",
        raise_pyarrow,
    )

    result = CliRunner().invoke(
        cli,
        ["prepare-benchmark", "--spec", str(spec_path), "--output", str(output_path)],
    )

    assert result.exit_code != 0
    assert "uv run --isolated --with pyarrow bsppctl prepare-benchmark --spec" in result.output
    assert str(spec_path) in result.output
    assert str(output_path) in result.output


def test_prepare_benchmark_curation_failure_is_click_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bspp.orchestration.control.folding_benchmark.curator import BenchmarkCurationFailed

    spec_path = _write_spec(tmp_path / "spec.json")
    output_path = tmp_path / "out"

    def raise_curation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise BenchmarkCurationFailed(
            "could not fill monomer_short: selected 0/1 from 0 materialized candidates; "
            f"staging={tmp_path / '.out.staging'}"
        )

    monkeypatch.setattr(
        "bspp.orchestration.control.folding_benchmark.curator.prepare_benchmark_dataset",
        raise_curation,
    )

    result = CliRunner().invoke(
        cli,
        ["prepare-benchmark", "--spec", str(spec_path), "--output", str(output_path)],
    )

    assert result.exit_code != 0
    assert "Error: could not fill monomer_short" in result.output
    assert "staging=" in result.output
    assert "Traceback" not in result.output


def _write_benchmark_profiles(tmp_path: Path) -> Path:
    cluster = {
        "owner": "tester",
        "project_root": str(tmp_path / "project"),
        "output_root": str(tmp_path / "output"),
        "staging_root": str(tmp_path / "staging"),
        "orchestration_repo": str(tmp_path / "orchestration"),
        "image": str(tmp_path / "images" / "bspp.sqsh"),
        "transport": "local-slurm",
        "account": "user-account",
        "postprocessing_credential_mounts": {
            "aws_shared_credentials_file": str(tmp_path / "aws" / "credentials"),
            "aws_config_file": str(tmp_path / "aws" / "config"),
        },
    }
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump({"clusters": {"example-cluster": cluster}}, sort_keys=False))
    return path


def _write_benchmark_profiles_without_credentials(tmp_path: Path) -> Path:
    cluster = {
        "owner": "tester",
        "project_root": str(tmp_path / "project"),
        "output_root": str(tmp_path / "output"),
        "staging_root": str(tmp_path / "staging"),
        "orchestration_repo": str(tmp_path / "orchestration"),
        "image": str(tmp_path / "images" / "bspp.sqsh"),
        "transport": "local-slurm",
        "account": "user-account",
    }
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump({"clusters": {"example-cluster": cluster}}, sort_keys=False))
    return path


class FetchCopyingRunner:
    """RecordingRunner that really copies the bounded summary during fetch."""

    def __init__(self, responses: list[CommandResult]) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.responses = responses

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(argv)
        if argv and argv[0] == "cp":
            shutil.copy(argv[1], argv[2])
            return CommandResult(argv=argv, returncode=0, stdout="", stderr="")
        response = self.responses.pop(0)
        return CommandResult(argv=argv, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr)


def _validate_run_args(tmp_path: Path, *, output_dir: Path) -> tuple[Any, ...]:
    return (
        tmp_path / "run",
        tmp_path / "suite.json",
        tmp_path / "index.json",
        "s3://example-bucket/benchmark-corpus",
        "f" * 64,
        output_dir,
    )


def test_validate_run_help() -> None:
    result = CliRunner().invoke(cli, ["validate-run", "--help"])

    assert result.exit_code == 0
    for option in (
        "--run-dir",
        "--suite",
        "--index",
        "--corpus",
        "--fingerprint",
        "--profile",
        "--output-dir",
        "--aws-profile",
    ):
        assert option in result.output


def test_validate_run_requires_arguments() -> None:
    result = CliRunner().invoke(cli, ["validate-run"])

    assert result.exit_code != 0


def test_summary_accepted_uses_passed_count() -> None:
    real_shape = {
        "schema_version": 1,
        "dataset_id": "bench",
        "fingerprint": "f" * 64,
        "case_count": 2,
        "passed_count": 2,
        "cases": [{"passed": True}, {"passed": True}],
    }
    assert _summary_accepted(real_shape) is True

    failing = {**real_shape, "passed_count": 1}
    assert _summary_accepted(failing) is False


def test_summary_accepted_falls_back_to_passed_key() -> None:
    legacy_shape = {"case_count": 2, "passed": 2}
    assert _summary_accepted(legacy_shape) is True

    failing = {"case_count": 2, "passed": 1}
    assert _summary_accepted(failing) is False


def test_render_benchmark_validate_submission(tmp_path: Path) -> None:
    config_path = _write_benchmark_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    script_path = tmp_path / "sbatch" / "validate-run.sbatch"
    run_dir, suite, index, corpus, fingerprint, output_dir = _validate_run_args(
        tmp_path, output_dir=tmp_path / "evidence"
    )

    rendered = render_benchmark_validate_submission(
        cluster_profile=profile,
        run_dir=run_dir,
        suite=suite,
        index=index,
        corpus=corpus,
        fingerprint=fingerprint,
        output_dir=output_dir,
        script_path=script_path,
    )

    assert "bspp-orchestration-runtime benchmark validate-run" in rendered.action_command
    assert "--run-dir" in rendered.action_command
    assert "--output-dir" in rendered.action_command
    assert rendered.job_name.startswith("bspp_validate_run_")
    assert rendered.summary_path == output_dir / "summary.json"
    script_text = script_path.read_text(encoding="utf-8")
    assert "#SBATCH --partition=" in script_text
    assert "--account=" in script_text
    assert "#SBATCH --job-name=" + rendered.job_name in script_text


def test_render_benchmark_validate_submission_mounts_and_aws_exports(tmp_path: Path) -> None:
    config_path = _write_benchmark_profiles(tmp_path)
    credentials_path = tmp_path / "aws" / "credentials"
    config_aws_path = tmp_path / "aws" / "config"
    credentials_path.parent.mkdir(parents=True)
    credentials_path.write_text("[prod]\naws_access_key_id = SECRET_ACCESS_KEY\n", encoding="utf-8")
    config_aws_path.write_text("[profile prod]\nendpoint_url = SECRET_ENDPOINT\n", encoding="utf-8")
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    script_path = tmp_path / "sbatch" / "validate-run.sbatch"
    run_dir, suite, index, corpus, fingerprint, output_dir = _validate_run_args(
        tmp_path, output_dir=tmp_path / "evidence"
    )

    render_benchmark_validate_submission(
        cluster_profile=profile,
        run_dir=run_dir,
        suite=suite,
        index=index,
        corpus=corpus,
        fingerprint=fingerprint,
        output_dir=output_dir,
        script_path=script_path,
        aws_profile="my profile",
    )

    script_text = script_path.read_text(encoding="utf-8")
    assert f"{tmp_path / 'orchestration'}:/workspace/bspp-orchestration:ro" in script_text
    assert f"{credentials_path}:/workspace/bspp-aws/credentials:ro" in script_text
    assert f"{config_aws_path}:/workspace/bspp-aws/config:ro" in script_text
    assert "AWS_SHARED_CREDENTIALS_FILE=/workspace/bspp-aws/credentials" in script_text
    assert "AWS_CONFIG_FILE=/workspace/bspp-aws/config" in script_text
    assert "export AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE" in script_text
    assert "AWS_PROFILE='my profile'" in script_text
    assert "export AWS_PROFILE" in script_text
    assert "SECRET_ACCESS_KEY" not in script_text
    assert "SECRET_ENDPOINT" not in script_text


def test_render_benchmark_validate_submission_requires_credential_mounts(tmp_path: Path) -> None:
    config_path = _write_benchmark_profiles_without_credentials(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    script_path = tmp_path / "sbatch" / "validate-run.sbatch"
    run_dir, suite, index, corpus, fingerprint, output_dir = _validate_run_args(
        tmp_path, output_dir=tmp_path / "evidence"
    )

    with pytest.raises(ValueError, match="postprocessing_credential_mounts"):
        render_benchmark_validate_submission(
            cluster_profile=profile,
            run_dir=run_dir,
            suite=suite,
            index=index,
            corpus=corpus,
            fingerprint=fingerprint,
            output_dir=output_dir,
            script_path=script_path,
        )


def test_submit_benchmark_validation_success(tmp_path: Path) -> None:
    config_path = _write_benchmark_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    output_dir = tmp_path / "evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "dataset_id": "bench",
        "fingerprint": "f" * 64,
        "case_count": 2,
        "passed_count": 2,
        "cases": [{"verdict": "ok"}, {"verdict": "ok"}],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    runner = FetchCopyingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="4242\n", stderr=""),
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id_raw": "4242",
                                "state": {"current": "COMPLETED"},
                                "exit_code": {"return_code": 0, "signal": 0},
                            }
                        ]
                    }
                ),
                stderr="",
            ),
        ]
    )
    run_dir, suite, index, corpus, fingerprint, _ = _validate_run_args(tmp_path, output_dir=output_dir)

    result = submit_benchmark_validation(
        cluster_profile=profile,
        run_dir=run_dir,
        suite=suite,
        index=index,
        corpus=corpus,
        fingerprint=fingerprint,
        output_dir=output_dir,
        runner=runner,
    )

    assert result.job_id == "4242"
    assert result.accepted is True
    assert json.loads(result.render_json()) == summary


def test_submit_benchmark_validation_creates_local_slurm_log_dir(tmp_path: Path) -> None:
    config_path = _write_benchmark_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    assert profile.transport == "local-slurm"
    output_dir = tmp_path / "evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "dataset_id": "bench",
        "fingerprint": "f" * 64,
        "case_count": 1,
        "passed_count": 1,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert not (output_dir / "slurm-logs").exists()
    runner = FetchCopyingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="4242\n", stderr=""),
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id_raw": "4242",
                                "state": {"current": "COMPLETED"},
                                "exit_code": {"return_code": 0, "signal": 0},
                            }
                        ]
                    }
                ),
                stderr="",
            ),
        ]
    )
    run_dir, suite, index, corpus, fingerprint, _ = _validate_run_args(tmp_path, output_dir=output_dir)

    result = submit_benchmark_validation(
        cluster_profile=profile,
        run_dir=run_dir,
        suite=suite,
        index=index,
        corpus=corpus,
        fingerprint=fingerprint,
        output_dir=output_dir,
        runner=runner,
    )

    assert result.job_id == "4242"
    assert (output_dir / "slurm-logs").is_dir()


def test_submit_benchmark_validation_failed_job_raises(tmp_path: Path) -> None:
    config_path = _write_benchmark_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    output_dir = tmp_path / "evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = FetchCopyingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="4242\n", stderr=""),
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id_raw": "4242",
                                "state": {"current": "FAILED"},
                                "exit_code": {"return_code": 1, "signal": 0},
                            }
                        ]
                    }
                ),
                stderr="",
            ),
        ]
    )
    run_dir, suite, index, corpus, fingerprint, _ = _validate_run_args(tmp_path, output_dir=output_dir)

    with pytest.raises(ValueError, match="FAILED"):
        submit_benchmark_validation(
            cluster_profile=profile,
            run_dir=run_dir,
            suite=suite,
            index=index,
            corpus=corpus,
            fingerprint=fingerprint,
            output_dir=output_dir,
            runner=runner,
        )


def test_submit_benchmark_validation_not_accepted(tmp_path: Path) -> None:
    config_path = _write_benchmark_profiles(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config_path)
    output_dir = tmp_path / "evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "dataset_id": "bench", "fingerprint": "f" * 64, "case_count": 2, "passed_count": 1}
    (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    runner = FetchCopyingRunner(
        [
            CommandResult(argv=(), returncode=0, stdout="4242\n", stderr=""),
            CommandResult(argv=(), returncode=0, stdout=json.dumps({"jobs": []}), stderr=""),
            CommandResult(
                argv=(),
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id_raw": "4242",
                                "state": {"current": "COMPLETED"},
                                "exit_code": {"return_code": 0, "signal": 0},
                            }
                        ]
                    }
                ),
                stderr="",
            ),
        ]
    )
    run_dir, suite, index, corpus, fingerprint, _ = _validate_run_args(tmp_path, output_dir=output_dir)

    result = submit_benchmark_validation(
        cluster_profile=profile,
        run_dir=run_dir,
        suite=suite,
        index=index,
        corpus=corpus,
        fingerprint=fingerprint,
        output_dir=output_dir,
        runner=runner,
    )

    assert result.accepted is False
    assert json.loads(result.render_json()) == summary


def test_validate_run_cli_echoes_summary_and_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_benchmark_profiles(tmp_path)
    summary = {"schema_version": 1, "dataset_id": "bench", "fingerprint": "f" * 64, "case_count": 1, "passed_count": 1}

    def fake_submit(**kwargs: Any) -> BenchmarkValidateResult:
        return BenchmarkValidateResult(job_id="4242", summary=summary, accepted=True)

    monkeypatch.setattr(
        "bspp.orchestration.control.folding_benchmark_submit.submit_benchmark_validation",
        fake_submit,
    )

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config_path),
            "validate-run",
            "--run-dir",
            str(tmp_path / "run"),
            "--suite",
            str(tmp_path / "suite.json"),
            "--index",
            str(tmp_path / "index.json"),
            "--corpus",
            "s3://example-bucket/benchmark-corpus",
            "--fingerprint",
            "f" * 64,
            "--profile",
            "example-cluster",
            "--output-dir",
            str(tmp_path / "evidence"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == summary


def test_validate_run_cli_exits_nonzero_when_not_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_benchmark_profiles(tmp_path)
    summary = {"schema_version": 1, "dataset_id": "bench", "fingerprint": "f" * 64, "case_count": 2, "passed_count": 1}

    def fake_submit(**kwargs: Any) -> BenchmarkValidateResult:
        return BenchmarkValidateResult(job_id="4242", summary=summary, accepted=False)

    monkeypatch.setattr(
        "bspp.orchestration.control.folding_benchmark_submit.submit_benchmark_validation",
        fake_submit,
    )

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config_path),
            "validate-run",
            "--run-dir",
            str(tmp_path / "run"),
            "--suite",
            str(tmp_path / "suite.json"),
            "--index",
            str(tmp_path / "index.json"),
            "--corpus",
            "s3://example-bucket/benchmark-corpus",
            "--fingerprint",
            "f" * 64,
            "--profile",
            "example-cluster",
            "--output-dir",
            str(tmp_path / "evidence"),
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.output) == summary


def test_validate_run_cli_forwards_aws_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_benchmark_profiles(tmp_path)
    summary = {"schema_version": 1, "dataset_id": "bench", "fingerprint": "f" * 64, "case_count": 1, "passed_count": 1}
    captured: dict[str, Any] = {}

    def fake_submit(**kwargs: Any) -> BenchmarkValidateResult:
        captured.update(kwargs)
        return BenchmarkValidateResult(job_id="4242", summary=summary, accepted=True)

    monkeypatch.setattr(
        "bspp.orchestration.control.folding_benchmark_submit.submit_benchmark_validation",
        fake_submit,
    )

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config_path),
            "validate-run",
            "--run-dir",
            str(tmp_path / "run"),
            "--suite",
            str(tmp_path / "suite.json"),
            "--index",
            str(tmp_path / "index.json"),
            "--corpus",
            "s3://example-bucket/benchmark-corpus",
            "--fingerprint",
            "f" * 64,
            "--profile",
            "example-cluster",
            "--output-dir",
            str(tmp_path / "evidence"),
            "--aws-profile",
            "prod",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["aws_profile"] == "prod"
