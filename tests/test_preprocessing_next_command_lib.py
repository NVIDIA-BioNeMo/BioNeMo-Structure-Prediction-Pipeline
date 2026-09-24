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

"""Tests for the sourceable, reentrant preprocessing handoff library."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "containers" / "scripts" / "preprocessing-next-command-lib.sh"


def _run(script: str, *arguments: Path | str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script, "test-shell", *(str(argument) for argument in arguments)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def test_library_is_source_only_has_bash_syntax_and_self_verifies() -> None:
    expected = hashlib.sha256(LIBRARY.read_bytes()).hexdigest()
    syntax = subprocess.run(["bash", "-n", str(LIBRARY)], capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr

    result = _run(
        'source "$1"; source "$1"; bspp_verify_library_self "$2" "$1"',
        LIBRARY,
        expected,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""

    mismatch = _run('source "$1"; bspp_verify_library_self "$2" "$1"', LIBRARY, "0" * 64)
    assert mismatch.returncode != 0
    assert "SHA-256 mismatch" in mismatch.stderr


def test_sha_verified_staging_is_immutable_and_reentrant(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "staged" / "source"
    source.write_bytes(b"verified payload\n")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    first = _run('source "$1"; bspp_stage_verified "$2" "$3" "$4"', LIBRARY, source, expected, destination)
    second = _run('source "$1"; bspp_stage_verified "$2" "$3" "$4"', LIBRARY, source, expected, destination)

    assert first.returncode == second.returncode == 0
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o400

    destination.chmod(0o600)
    wrong_mode = _run('source "$1"; bspp_stage_verified "$2" "$3" "$4"', LIBRARY, source, expected, destination)
    assert wrong_mode.returncode != 0
    assert "mode mismatch" in wrong_mode.stderr
    destination.chmod(0o400)

    source.write_bytes(b"drifted payload\n")
    changed_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    drift = _run('source "$1"; bspp_stage_verified "$2" "$3" "$4"', LIBRARY, source, changed_sha, destination)
    assert drift.returncode != 0
    assert "SHA-256 mismatch" in drift.stderr
    assert destination.read_bytes() == b"verified payload\n"


def test_bounded_copy_rejects_oversize_before_staging(tmp_path: Path) -> None:
    source = tmp_path / "large"
    destination = tmp_path / "copy"
    source.write_bytes(b"0123456789")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    result = _run('source "$1"; bspp_bounded_copy "$2" "$3" 5 "$4"', LIBRARY, source, destination, expected)

    assert result.returncode != 0
    assert "exceeds 5 bytes" in result.stderr
    assert not destination.exists()


def test_write_once_rejects_matching_content_with_wrong_mode(tmp_path: Path) -> None:
    destination = tmp_path / "immutable.record"
    destination.write_text("same\n")
    destination.chmod(0o600)

    result = _run('source "$1"; bspp_write_once "$2" 0400 same', LIBRARY, destination)

    assert result.returncode != 0
    assert "mode mismatch" in result.stderr


def test_verbatim_submission_publishes_intent_job_and_result_then_reuses_job(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    _write_executable(
        fake_bin / "sbatch",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >>\"$FAKE_SBATCH_LOG\"\nprintf '4242;cluster\\n'\n",
    )
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
    script = tmp_path / "rendered.sbatch"
    script.write_text("#!/usr/bin/env bash\n#SBATCH --job-name=depth-batch-0\nprintf 'verbatim script\\n'\n")
    script.chmod(0o500)
    expected = hashlib.sha256(script.read_bytes()).hexdigest()
    state = tmp_path / "state"
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "FAKE_SBATCH_LOG": str(sbatch_log)}
    command = 'source "$1"; bspp_submit_rendered_sbatch "$2" "$3" "$4" depth-batch-0'

    first = _run(command, LIBRARY, script, expected, state, env=environment)
    second = _run(command, LIBRARY, script, expected, state, env=environment)

    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert first.stdout.strip() == second.stdout.strip() == "4242"
    assert sbatch_log.read_text().splitlines() == [f"--parsable -- {script}"]
    assert (state / "slurm-job-id").read_text() == "4242\n"
    assert (state / "submission.intent").read_text() == (f"script_sha256={expected} job_name=depth-batch-0\n")
    assert (state / "submission.result").read_text() == (f"submitted job_id=4242 script_sha256={expected}\n")
    assert script.read_text() == (
        "#!/usr/bin/env bash\n#SBATCH --job-name=depth-batch-0\nprintf 'verbatim script\\n'\n"
    )

    mismatch = _run(
        'source "$1"; bspp_submit_rendered_sbatch "$2" "$3" "$4" different-name',
        LIBRARY,
        script,
        expected,
        tmp_path / "other-state",
        env=environment,
    )
    assert mismatch.returncode != 0
    assert "differs from rendered sbatch" in mismatch.stderr


def test_reentrant_job_discovery_uses_unique_live_squeue_match(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nprintf '5252\\n'\n")
    state = tmp_path / "state"
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = _run('source "$1"; bspp_discover_job "$2" depth-batch-1', LIBRARY, state, env=environment)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "5252\n"
    assert (state / "slurm-job-id").read_text() == "5252\n"


def test_job_discovery_fails_closed_on_malformed_persisted_job_id(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "sbatch", f"#!/usr/bin/env bash\nprintf 'called\\n' >>{sbatch_log}\n")
    state = tmp_path / "state"
    state.mkdir()
    (state / "slurm-job-id").write_text("not-a-job-id\n")
    (state / "slurm-job-id").chmod(0o400)
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = _run('source "$1"; bspp_discover_job "$2" depth-invalid', LIBRARY, state, env=environment)

    assert result.returncode == 2
    assert "invalid persisted Slurm job id" in result.stderr
    assert not sbatch_log.exists()


def test_job_discovery_recovers_completed_exact_name_from_sacct_history(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sacct_log = tmp_path / "sacct.log"
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >\"$FAKE_SACCT_LOG\"\nprintf '5353|depth-history|COMPLETED\\n'\n",
    )
    state = tmp_path / "state"
    state.mkdir()
    (state / "submission.intent").write_text(f"script_sha256={'a' * 64} job_name=depth-history\n")
    (state / "submission.intent").chmod(0o400)
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "FAKE_SACCT_LOG": str(sacct_log)}

    result = _run('source "$1"; bspp_discover_job "$2" depth-history', LIBRARY, state, env=environment)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "5353\n"
    assert (state / "slurm-job-id").read_text() == "5353\n"
    assert "--name depth-history" in sacct_log.read_text()
    assert "--format=JobIDRaw,JobName,State" in sacct_log.read_text()
    assert "state=%T" not in sacct_log.read_text()


def test_historical_job_discovery_fails_closed_on_ambiguous_exact_name(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\nprintf '5454|depth-ambiguous|COMPLETED\\n5455|depth-ambiguous|FAILED\\n'\n",
    )
    state = tmp_path / "state"
    state.mkdir()
    (state / "submission.intent").write_text(f"script_sha256={'b' * 64} job_name=depth-ambiguous\n")
    (state / "submission.intent").chmod(0o400)
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = _run('source "$1"; bspp_discover_job "$2" depth-ambiguous', LIBRARY, state, env=environment)

    assert result.returncode == 2
    assert "multiple historical jobs" in result.stderr
    assert not (state / "slurm-job-id").exists()


def test_historical_job_discovery_fails_closed_when_submission_intent_has_no_match(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "sacct", "#!/usr/bin/env bash\nexit 0\n")
    state = tmp_path / "state"
    state.mkdir()
    (state / "submission.intent").write_text(f"script_sha256={'c' * 64} job_name=depth-missing\n")
    (state / "submission.intent").chmod(0o400)
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = _run('source "$1"; bspp_discover_job "$2" depth-missing', LIBRARY, state, env=environment)

    assert result.returncode == 2
    assert "no historical job matches" in result.stderr
    assert not (state / "slurm-job-id").exists()


def test_submit_reuses_historical_job_after_post_sbatch_crash_without_duplicate(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\nprintf '5555|depth-crash|COMPLETED\\n'\n",
    )
    _write_executable(
        fake_bin / "sbatch",
        "#!/usr/bin/env bash\nprintf 'called\\n' >>\"$FAKE_SBATCH_LOG\"\nprintf '9999\\n'\n",
    )
    script = tmp_path / "rendered.sbatch"
    script.write_text("#!/usr/bin/env bash\n#SBATCH --job-name=depth-crash\nexit 0\n")
    script.chmod(0o500)
    expected = hashlib.sha256(script.read_bytes()).hexdigest()
    state = tmp_path / "state"
    state.mkdir()
    (state / "submission.intent").write_text(f"script_sha256={expected} job_name=depth-crash\n")
    (state / "submission.intent").chmod(0o400)
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "FAKE_SBATCH_LOG": str(sbatch_log)}

    result = _run(
        'source "$1"; bspp_submit_rendered_sbatch "$2" "$3" "$4" depth-crash',
        LIBRARY,
        script,
        expected,
        state,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "5555\n"
    assert not sbatch_log.exists()
    assert (state / "slurm-job-id").read_text() == "5555\n"


def test_monitor_records_terminal_accounting_with_path_shims(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\nprintf '6161|COMPLETED|0:0|00:10:00|node-1\\n'\n",
    )
    state = tmp_path / "state"
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = _run('source "$1"; bspp_monitor_job 6161 "$2" 1', LIBRARY, state, env=environment)

    assert result.returncode == 0, result.stderr
    assert "6161|COMPLETED|0:0|00:10:00|node-1" in result.stdout
    assert (state / "monitor.result").read_text() == ("state=COMPLETED\nexit_code=0:0\nelapsed=00:10:00\nnode=node-1\n")


def test_monitor_treats_failed_squeue_as_presence_unknown_and_exhausts_nonterminal_accounting(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    counter = tmp_path / "sacct-count"
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 2\n")
    _write_executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\n"
        "count=0\n"
        '[[ ! -f "$FAKE_SACCT_COUNT" ]] || count=$(<"$FAKE_SACCT_COUNT")\n'
        "count=$((count + 1))\n"
        'printf \'%s\\n\' "$count" >"$FAKE_SACCT_COUNT"\n'
        "printf '6464|RUNNING|0:0|00:01:00|node-4\\n'\n",
    )
    state = tmp_path / "state"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_SACCT_COUNT": str(counter),
    }

    result = _run('source "$1"; bspp_monitor_job 6464 "$2" 1', LIBRARY, state, env=environment)

    assert result.returncode != 0
    assert counter.read_text() == "12\n"
    assert "presence unknown" in result.stderr
    assert "RUNNING" in result.stderr
    assert not (state / "monitor.result").exists()


def test_monitor_treats_failed_squeue_as_presence_unknown_and_accepts_terminal_accounting(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 2\n")
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\nprintf '6474|COMPLETED|0:0|00:01:04|node-7\\n'\n",
    )
    state = tmp_path / "state"
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = _run('source "$1"; bspp_monitor_job 6474 "$2" 1', LIBRARY, state, env=environment)

    assert result.returncode == 0, result.stderr
    assert "presence unknown" in result.stderr
    assert "6474|COMPLETED|0:0|00:01:04|node-7" in result.stdout
    assert (state / "monitor.result").read_text() == ("state=COMPLETED\nexit_code=0:0\nelapsed=00:01:04\nnode=node-7\n")


def test_monitor_retries_nonterminal_accounting_before_publishing_terminal_record(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    counter = tmp_path / "sacct-count"
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\n"
        "count=0\n"
        '[[ ! -f "$FAKE_SACCT_COUNT" ]] || count=$(<"$FAKE_SACCT_COUNT")\n'
        "count=$((count + 1))\n"
        'printf \'%s\\n\' "$count" >"$FAKE_SACCT_COUNT"\n'
        'if [[ "$count" == 1 ]]; then\n'
        "  printf '6565|RUNNING|0:0|00:00:03|node-5\\n'\n"
        "else\n"
        "  printf '6565|COMPLETED|0:0|00:00:05|node-5\\n'\n"
        "fi\n",
    )
    state = tmp_path / "state"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_SACCT_COUNT": str(counter),
    }

    result = _run('source "$1"; bspp_monitor_job 6565 "$2" 1', LIBRARY, state, env=environment)

    assert result.returncode == 0, result.stderr
    assert counter.read_text() == "2\n"
    assert (state / "monitor.result").read_text() == ("state=COMPLETED\nexit_code=0:0\nelapsed=00:00:05\nnode=node-5\n")


def test_monitor_exhausts_nonterminal_accounting_without_publishing_record(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    counter = tmp_path / "sacct-count"
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\n"
        "count=0\n"
        '[[ ! -f "$FAKE_SACCT_COUNT" ]] || count=$(<"$FAKE_SACCT_COUNT")\n'
        "count=$((count + 1))\n"
        'printf \'%s\\n\' "$count" >"$FAKE_SACCT_COUNT"\n'
        "printf '6666|COMPLETING|0:0|00:00:05|node-6\\n'\n",
    )
    state = tmp_path / "state"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_SACCT_COUNT": str(counter),
    }

    result = _run('source "$1"; bspp_monitor_job 6666 "$2" 1', LIBRARY, state, env=environment)

    assert result.returncode != 0
    assert counter.read_text() == "12\n"
    assert "COMPLETING" in result.stderr
    assert not (state / "monitor.result").exists()


def test_monitor_prints_live_detail_before_querying_terminal_accounting(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    queue_counter = tmp_path / "squeue-presence-count"
    event_log = tmp_path / "events"
    sacct_counter = tmp_path / "sacct-count"
    _write_executable(
        fake_bin / "squeue",
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"--format=%i"* ]]; then\n'
        "  count=0\n"
        '  [[ ! -f "$FAKE_SQUEUE_COUNT" ]] || count=$(<"$FAKE_SQUEUE_COUNT")\n'
        "  count=$((count + 1))\n"
        '  printf \'%s\\n\' "$count" >"$FAKE_SQUEUE_COUNT"\n'
        '  if [[ "$count" == 1 ]]; then\n'
        "    printf 'live-presence\\n' >>\"$FAKE_EVENT_LOG\"\n"
        "    printf '6767\\n'\n"
        "  else\n"
        "    printf 'empty-presence\\n' >>\"$FAKE_EVENT_LOG\"\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        '[[ ! -e "$FAKE_SACCT_COUNT" ]] || exit 91\n'
        "printf 'live-detail\\n' >>\"$FAKE_EVENT_LOG\"\n"
        "printf 'job=6767 state=RUNNING elapsed=00:00:03 node=node-6\\n'\n",
    )
    _write_executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\n"
        "printf '1\\n' >\"$FAKE_SACCT_COUNT\"\n"
        "printf 'accounting\\n' >>\"$FAKE_EVENT_LOG\"\n"
        "printf '6767|COMPLETED|0:0|00:00:08|node-6\\n'\n",
    )
    state = tmp_path / "state"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_SQUEUE_COUNT": str(queue_counter),
        "FAKE_SACCT_COUNT": str(sacct_counter),
        "FAKE_EVENT_LOG": str(event_log),
    }

    result = _run('source "$1"; bspp_monitor_job 6767 "$2" 1', LIBRARY, state, env=environment)

    assert result.returncode == 0, result.stderr
    assert "job=6767 state=RUNNING elapsed=00:00:03 node=node-6" in result.stdout
    assert event_log.read_text().splitlines() == [
        "live-presence",
        "live-detail",
        "empty-presence",
        "accounting",
    ]
    assert queue_counter.read_text() == "2\n"
    assert sacct_counter.read_text() == "1\n"
    assert (state / "monitor.result").read_text() == ("state=COMPLETED\nexit_code=0:0\nelapsed=00:00:08\nnode=node-6\n")


def test_monitor_fails_closed_when_detailed_live_query_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sacct_sentinel = tmp_path / "sacct-called"
    _write_executable(
        fake_bin / "squeue",
        '#!/usr/bin/env bash\nif [[ "$*" == *"--format=%i"* ]]; then\n  printf \'6868\\n\'\n  exit 0\nfi\nexit 3\n',
    )
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\nprintf 'called\\n' >\"$FAKE_SACCT_SENTINEL\"\n",
    )
    state = tmp_path / "state"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_SACCT_SENTINEL": str(sacct_sentinel),
    }

    result = _run('source "$1"; bspp_monitor_job 6868 "$2" 1', LIBRARY, state, env=environment)

    assert result.returncode != 0
    assert not sacct_sentinel.exists()
    assert not (state / "monitor.result").exists()


def test_monitor_resets_accounting_fields_across_an_empty_top_level_row(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    counter = tmp_path / "sacct-count"
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\n"
        "count=0\n"
        '[[ ! -f "$FAKE_SACCT_COUNT" ]] || count=$(<"$FAKE_SACCT_COUNT")\n'
        "count=$((count + 1))\n"
        'printf \'%s\\n\' "$count" >"$FAKE_SACCT_COUNT"\n'
        'case "$count" in\n'
        "  1) printf '6969|COMPLETED|0:0|00:00:01|\\n' ;;\n"
        "  2) printf '6969.batch|COMPLETED|0:0|00:00:04|node-step\\n' ;;\n"
        "  3) printf '6969|COMPLETED|0:0|00:00:09|node-9\\n' ;;\n"
        "esac\n",
    )
    state = tmp_path / "state"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_SACCT_COUNT": str(counter),
    }

    result = _run('source "$1"; bspp_monitor_job 6969 "$2" 1', LIBRARY, state, env=environment)

    assert result.returncode == 0, result.stderr
    assert counter.read_text() == "3\n"
    assert "6969|COMPLETED|0:0|00:00:09|node-9" in result.stdout
    assert (state / "monitor.result").read_text() == ("state=COMPLETED\nexit_code=0:0\nelapsed=00:00:09\nnode=node-9\n")


def test_monitor_persists_killed_top_level_exit_code_without_helper_conflation(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\nprintf '6262|NODE_FAIL|0:9|00:04:12|node-9\\n'\n",
    )
    state = tmp_path / "state"
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = _run('source "$1"; bspp_monitor_job 6262 "$2" 1', LIBRARY, state, env=environment)

    assert result.returncode != 0
    assert (state / "monitor.result").read_text() == ("state=NODE_FAIL\nexit_code=0:9\nelapsed=00:04:12\nnode=node-9\n")
    assert "helper" not in (state / "monitor.result").read_text()


def test_monitor_treats_evidence_job_completed_independently_of_recorded_helper_timeout(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "sacct",
        "#!/usr/bin/env bash\nprintf '6363|COMPLETED|0:0|00:16:01|node-3\\n'\n",
    )
    state = tmp_path / "state"
    state.mkdir()
    (state / "helper-exit-code.txt").write_text("124\n")
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = _run('source "$1"; bspp_monitor_job 6363 "$2" 1', LIBRARY, state, env=environment)

    assert result.returncode == 0, result.stderr
    assert (state / "helper-exit-code.txt").read_text() == "124\n"
    assert (state / "monitor.result").read_text() == ("state=COMPLETED\nexit_code=0:0\nelapsed=00:16:01\nnode=node-3\n")


def test_captured_command_must_equal_rendered_intent_bytes(tmp_path: Path) -> None:
    rendered = tmp_path / "rendered.json"
    captured = tmp_path / "captured.json"
    rendered.write_text('["srun","helper"]\n')
    captured.write_bytes(rendered.read_bytes())

    equal = _run('source "$1"; bspp_compare_captured_command "$2" "$3"', LIBRARY, rendered, captured)
    assert equal.returncode == 0, equal.stderr

    captured.write_text('["srun","other"]\n')
    changed = _run('source "$1"; bspp_compare_captured_command "$2" "$3"', LIBRARY, rendered, captured)
    assert changed.returncode != 0
    assert "differs from rendered intent" in changed.stderr


def test_phase_records_are_write_once_and_keep_intent_result_job_id_separate(tmp_path: Path) -> None:
    state = tmp_path / "state"
    command = (
        'source "$1"; '
        'bspp_publish_phase_record "$2" discovery intent intent-sha; '
        'bspp_publish_phase_record "$2" discovery job-id 7171; '
        'bspp_publish_phase_record "$2" discovery result completed'
    )

    result = _run(command, LIBRARY, state)

    assert result.returncode == 0, result.stderr
    assert (state / "discovery.intent").read_text() == "intent-sha\n"
    assert (state / "discovery.job-id").read_text() == "7171\n"
    assert (state / "discovery.result").read_text() == "completed\n"
