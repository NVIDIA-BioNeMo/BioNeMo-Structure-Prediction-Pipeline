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

"""Strict preprocessing image identity and scheduled qualification tests."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import importlib.util
import json
import os
import pwd
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from bspp.orchestration.contract.database_placement import DatabaseAccessPolicy
from bspp.orchestration.contract.database_placement_result import database_placement_result_from_mapping
from bspp.orchestration.contract.preprocessing_action import PREPROCESSING_COMMAND_ORDER
from bspp.orchestration.contract.preprocessing_runtime import (
    PREPROCESSING_ADAPTER_VERSION,
    PREPROCESSING_RUNTIME_COMMAND,
    PREPROCESSING_RUNTIME_CONTRACT_ID,
    PreprocessingRuntimeGpuEvidence,
    PreprocessingRuntimeImageEvidence,
    PreprocessingRuntimeImageIdentity,
    PreprocessingRuntimeQualificationRecord,
    PreprocessingRuntimeSmokeEvidence,
    PreprocessingRuntimeSourceEvidence,
    PreprocessingRuntimeToolEvidence,
    normalize_rsync_version,
    preprocessing_runtime_qualification_record_from_mapping,
    preprocessing_runtime_tool_evidence_from_mapping,
    preprocessing_runtime_tuple_id,
    validate_preprocessing_runtime_tool_evidence,
)
from bspp.orchestration.control.preprocessing_runtime_qualification import (
    PreprocessingRuntimeQualificationError,
    check_preprocessing_runtime_qualification,
    derive_smoke_gres,
    preprocessing_runtime_qualification_path,
    preprocessing_runtime_qualification_tuple,
    qualify_preprocessing_runtime,
    render_preprocessing_runtime_qualification_script,
    resolve_preprocessing_runtime_qualification,
)
from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.transport import CommandResult
from tests.support.transport_argv import maybe_unwrap_remote_command

NOW = datetime(2026, 8, 20, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3.4.4", "3.4.4"),
        ("rsync  version 3.4.4  protocol version 32", "3.4.4"),
    ],
)
def test_rsync_version_parser_accepts_only_normalized_values_or_an_exact_first_banner_line(
    value: str,
    expected: str,
) -> None:
    assert normalize_rsync_version(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "rsync version 3.4 protocol version 32",
        "rsync version 3.4.4",
        "prefix rsync version 3.4.4 protocol version 32",
        "rsync version 3.4.4 protocol version 32\nCapabilities:",
        "3.4.4 ",
    ],
)
def test_rsync_version_parser_rejects_malformed_or_non_first_line_values(value: str) -> None:
    with pytest.raises(ValueError, match="rsync_version"):
        normalize_rsync_version(value)


def test_contract_ids_use_canonical_json_and_records_are_strict(tmp_path: Path) -> None:
    profile_path, source_repo = _profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    canonical = json.dumps(qualification_tuple.to_mapping(), sort_keys=True, separators=(",", ":")).encode()
    assert preprocessing_runtime_tuple_id(qualification_tuple) == hashlib.sha256(canonical).hexdigest()
    assert "nodelist" not in qualification_tuple.to_mapping()
    assert PREPROCESSING_ADAPTER_VERSION == "preprocessing-scientific-backend-v3"
    assert PREPROCESSING_RUNTIME_CONTRACT_ID == "6826d13f71a176d5ac483f42d85a9967485f8c171a96e7d20c7d6350efbf3796"
    record = _qualified_record(qualification_tuple)
    assert preprocessing_runtime_qualification_record_from_mapping(record.to_mapping()) == record

    unknown = deepcopy(record.to_mapping())
    unknown["smoke_evidence"]["tools"]["invented"] = "accepted"  # type: ignore[index]
    with pytest.raises(ValueError, match="Unknown PreprocessingRuntimeToolEvidence"):
        preprocessing_runtime_qualification_record_from_mapping(unknown)

    missing_version = deepcopy(record.to_mapping())
    missing_smoke = missing_version["smoke_evidence"]
    assert isinstance(missing_smoke, dict)
    missing_smoke.pop("schema_version")
    with pytest.raises(ValueError, match="schema_version"):
        preprocessing_runtime_qualification_record_from_mapping(missing_version)

    reversed_timestamps = deepcopy(record.to_mapping())
    reversed_timestamps["qualified_at"] = "2026-08-20T08:00:00.000000Z"
    with pytest.raises(ValueError, match="must not precede submitted_at"):
        preprocessing_runtime_qualification_record_from_mapping(reversed_timestamps)

    missing_rsync = deepcopy(record.to_mapping())
    del missing_rsync["qualification_tuple"]["image_identity"]["rsync_version"]  # type: ignore[index]
    with pytest.raises(ValueError, match="rsync_version"):
        preprocessing_runtime_qualification_record_from_mapping(missing_rsync)

    mismatched_rsync = deepcopy(record.to_mapping())
    mismatched_rsync["smoke_evidence"]["tools"]["rsync_version"] = "3.4.5"  # type: ignore[index]
    with pytest.raises(ValueError, match="observed tool identity"):
        preprocessing_runtime_qualification_record_from_mapping(mismatched_rsync)


def test_tool_evidence_legacy_control_absent_alias_is_validated() -> None:
    """The legacy control_absent key maps to control_version=None only when it is
    exactly "true"; ambiguity (both keys) and invalid values are rejected."""
    base = {
        "schema_version": 1,
        "python_version": "Python 3.12.0",
        "contract_version": "0.1.0",
        "runtime_version": "0.1.0",
        "mmseqs_version": "18-8cc5c",
        "colabfold_version": "1.6.2",
        "rsync_version": "3.4.4",
        "tar_version": "tar 1.35",
        "lz4_version": "lz4 1.10.0",
        "flock_version": "flock 2.42",
    }
    legacy = {**base, "control_absent": "true"}
    assert preprocessing_runtime_tool_evidence_from_mapping(legacy).control_version is None

    with pytest.raises(ValueError, match='control_absent must be "true"'):
        preprocessing_runtime_tool_evidence_from_mapping({**base, "control_absent": "false"})

    with pytest.raises(ValueError, match="both control_version and control_absent"):
        preprocessing_runtime_tool_evidence_from_mapping({**base, "control_version": "0.1.0", "control_absent": "true"})

    # Current-schema evidence must attest a non-empty Control version.
    with pytest.raises(ValueError, match="control_version"):
        preprocessing_runtime_tool_evidence_from_mapping(base)


def test_tool_evidence_requires_control_version_when_image_bakes_control() -> None:
    """validate_preprocessing_runtime_tool_evidence rejects evidence with no
    Control version when the baked image identity declares a control wheel."""
    image_identity = PreprocessingRuntimeImageIdentity(
        source_commit="3" * 40,
        image_lock_sha256="4" * 64,
        contract_wheel_sha256="5" * 64,
        runtime_wheel_sha256="6" * 64,
        control_wheel_sha256="a" * 64,
        colabfold_version="1.6.2",
        mmseqs_version="18-8cc5c",
        rsync_version="3.4.4",
        cuda_version="12.6.3",
    )
    observed = PreprocessingRuntimeToolEvidence(
        python_version="Python 3.12.0",
        contract_version="0.1.0",
        runtime_version="0.1.0",
        control_version=None,
        mmseqs_version="18-8cc5c",
        colabfold_version="1.6.2",
        rsync_version="3.4.4",
        tar_version="tar 1.35",
        lz4_version="lz4 1.10.0",
        flock_version="flock 2.42",
    )
    with pytest.raises(ValueError, match="Control version"):
        validate_preprocessing_runtime_tool_evidence(image_identity, observed)


def test_preprocessing_runtime_profile_requires_normalized_rsync_identity(tmp_path: Path) -> None:
    profile_path, _ = _profile(tmp_path)
    payload = yaml.safe_load(profile_path.read_text())
    runtime = payload["clusters"]["example-cluster"]["preprocessing_runtime"]
    del runtime["rsync_version"]
    profile_path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="rsync_version"):
        resolve_cluster_profile("example-cluster", config_path=profile_path)

    runtime["rsync_version"] = "rsync version 3.4.4 protocol version 32"
    profile_path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="normalized"):
        resolve_cluster_profile("example-cluster", config_path=profile_path)


def test_qualification_accepts_the_baked_mmseqs_commit_inside_the_observed_version_banner(tmp_path: Path) -> None:
    profile_path, source_repo = _profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    base_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    base_record = _qualified_record(base_tuple)
    commit = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
    image_identity = replace(base_tuple.image_identity, mmseqs_version=commit)
    qualification_tuple = replace(base_tuple, image_identity=image_identity)
    assert base_record.smoke_evidence is not None
    tools = replace(
        base_record.smoke_evidence.tools,
        mmseqs_version=f"MMseqs Version: {commit}",
    )
    smoke = replace(base_record.smoke_evidence, tools=tools)
    record = replace(
        base_record,
        qualification_tuple=qualification_tuple,
        tuple_id=preprocessing_runtime_tuple_id(qualification_tuple),
        smoke_evidence=smoke,
    )

    assert preprocessing_runtime_qualification_record_from_mapping(record.to_mapping()) == record


def test_qualification_script_has_exact_gpu_image_and_smoke_argv(tmp_path: Path) -> None:
    profile_path, source_repo = _profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)
    record_path = preprocessing_runtime_qualification_path(profile, tuple_id=tuple_id)
    script = render_preprocessing_runtime_qualification_script(
        profile=profile, qualification_tuple=qualification_tuple, tuple_id=tuple_id, record_path=record_path
    )
    assert "#SBATCH --partition=example-gpu" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "--container-image=/images/preprocessing.sqsh" in script
    expected_flock = "/usr/bin/flock \\" + "\n  -x"
    assert expected_flock in script
    assert "/opt/bspp/bin/bspp-preprocessing-image-smoke" in script
    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in script
    assert "unset BSPP_PREPROCESSING_GPU_EVIDENCE" in script
    assert "command -v nvidia-smi >/dev/null" in script
    assert 'BSPP_PREPROCESSING_GPU_EVIDENCE="$(nvidia-smi ' in script
    assert "export BSPP_PREPROCESSING_GPU_EVIDENCE" in script
    assert 'export BSPP_PREPROCESSING_GPU_EVIDENCE="$(' not in script
    assert script.index("BSPP_PREPROCESSING_GPU_EVIDENCE=") < script.index("srun \\")
    assert 'SMOKE_TMP_PARENT="${SLURM_TMPDIR:-/tmp}"' in script
    assert 'SMOKE_DATABASE_HOST_ROOT="$(mktemp -d ' in script
    assert "SMOKE_DATABASE_TARGET=/run/bspp/database" in script
    assert '--container-mounts="$CONTAINER_MOUNTS"' in script
    assert "nvidia-smi:/" not in script
    assert "\n+  " not in script
    assert qualification_tuple.source_bundle_path in script
    subprocess.run(("bash", "-n"), input=script, check=True, text=True)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("/scratch", "/run"),
        ("/scratch", "/run/bspp/database/selected"),
        ("/run/bspp/database", "/scratch"),
    ],
)
def test_qualification_rejects_mounts_overlapping_the_isolated_smoke_database(
    tmp_path: Path,
    source: str,
    target: str,
) -> None:
    profile_path, source_repo = _profile(tmp_path)
    profile_mapping = yaml.safe_load(profile_path.read_text())
    profile_mapping["clusters"]["example-cluster"]["extra_mounts"] = [{"source": source, "target": target}]
    profile_path.write_text(yaml.safe_dump(profile_mapping, sort_keys=True))
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)

    with pytest.raises(PreprocessingRuntimeQualificationError, match="protected smoke database namespace"):
        render_preprocessing_runtime_qualification_script(
            profile=profile,
            qualification_tuple=qualification_tuple,
            tuple_id=tuple_id,
            record_path=preprocessing_runtime_qualification_path(profile, tuple_id=tuple_id),
        )


def test_qualification_tuple_and_script_bind_the_selected_gpu_node(tmp_path: Path) -> None:
    profile_path, source_repo = _profile(tmp_path)
    profile_mapping = yaml.safe_load(profile_path.read_text())
    profile_mapping["clusters"]["example-cluster"]["resources"] = {"gpu_worker": {"nodelist": "gpu-node-017"}}
    profile_path.write_text(yaml.safe_dump(profile_mapping, sort_keys=True))
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)

    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)
    script = render_preprocessing_runtime_qualification_script(
        profile=profile,
        qualification_tuple=qualification_tuple,
        tuple_id=tuple_id,
        record_path=preprocessing_runtime_qualification_path(profile, tuple_id=tuple_id),
    )

    assert qualification_tuple.nodelist == "gpu-node-017"
    assert qualification_tuple.to_mapping()["nodelist"] == "gpu-node-017"
    assert script.count("#SBATCH --nodelist=gpu-node-017") == 1

    mismatched_tuple = replace(qualification_tuple, nodelist="gpu-node-018")
    with pytest.raises(PreprocessingRuntimeQualificationError, match="nodelist does not match profile"):
        render_preprocessing_runtime_qualification_script(
            profile=profile,
            qualification_tuple=mismatched_tuple,
            tuple_id=preprocessing_runtime_tuple_id(mismatched_tuple),
            record_path=preprocessing_runtime_qualification_path(
                profile,
                tuple_id=preprocessing_runtime_tuple_id(mismatched_tuple),
            ),
        )


def test_qualification_script_relays_host_gpu_evidence_and_fails_before_srun(tmp_path: Path) -> None:
    cases = {
        "success": ("#!/bin/sh\nprintf '%s' 'GPU A, 555.1\nGPU B, 555.1'\n", 0),
        "missing": (None, 127),
        "nonzero": ("#!/bin/sh\nexit 1\n", 127),
        "blank": ("#!/bin/sh\nprintf ' \t\n'\n", 127),
    }
    expected = "GPU A, 555.1\nGPU B, 555.1"
    for name, (nvidia_smi, expected_returncode) in cases.items():
        case_root = tmp_path / name
        case_root.mkdir()
        script = _render_qualification_script_with_real_image(case_root)
        stub_dir = case_root / "bin"
        stub_dir.mkdir()
        _write_shell_stub(stub_dir / "sha256sum", 'exec /usr/bin/sha256sum "$@"')
        _write_shell_stub(stub_dir / "awk", 'exec /usr/bin/awk "$@"')
        _write_shell_stub(stub_dir / "mktemp", 'exec /usr/bin/mktemp "$@"')
        _write_shell_stub(stub_dir / "find", 'exec /usr/bin/find "$@"')
        _write_shell_stub(stub_dir / "rmdir", 'exec /usr/bin/rmdir "$@"')
        slurm_tmpdir = case_root / "slurm-tmp"
        slurm_tmpdir.mkdir()
        marker = case_root / "srun-called"
        observed_evidence = case_root / "srun-evidence"
        observed_database_root = case_root / "srun-database-root"
        _write_shell_stub(
            stub_dir / "srun",
            'mounts=""\n'
            'for argument in "$@"; do\n'
            '  case "$argument" in --container-mounts=*) mounts="${argument#*=}" ;; esac\n'
            "done\n"
            'smoke_mapping="${mounts##*,}"\n'
            'smoke_database_root="${smoke_mapping%:/run/bspp/database}"\n'
            '[ "$smoke_database_root" != "$smoke_mapping" ] || exit 64\n'
            '[ -d "$smoke_database_root" ] && [ -w "$smoke_database_root" ] || exit 65\n'
            'case "$smoke_database_root" in '
            '"$SLURM_TMPDIR"/bspp-preprocessing-qualification.*) ;; *) exit 66 ;; esac\n'
            'printf "%s" "$smoke_database_root" > "$BSPP_TEST_DATABASE_ROOT"\n'
            'printf "%s" "${BSPP_PREPROCESSING_GPU_EVIDENCE-}" > "$BSPP_TEST_GPU_EVIDENCE"\n'
            ': > "$BSPP_TEST_SRUN_MARKER"',
        )
        if nvidia_smi is not None:
            _write_shell_stub(stub_dir / "nvidia-smi", nvidia_smi.removeprefix("#!/bin/sh\n"))
        environment = {
            "PATH": str(stub_dir),
            "SLURM_JOB_ID": "12345",
            "SLURM_TMPDIR": str(slurm_tmpdir),
            "BSPP_PREPROCESSING_GPU_EVIDENCE": "inherited evidence must be discarded",
            "BSPP_TEST_DATABASE_ROOT": str(observed_database_root),
            "BSPP_TEST_GPU_EVIDENCE": str(observed_evidence),
            "BSPP_TEST_SRUN_MARKER": str(marker),
        }
        completed = subprocess.run(
            ("/bin/bash", str(script)), env=environment, check=False, capture_output=True, text=True
        )
        assert completed.returncode == expected_returncode, completed.stderr
        if name == "success":
            assert marker.is_file()
            assert observed_evidence.read_text() == expected
            database_root = Path(observed_database_root.read_text())
            assert database_root.parent == slurm_tmpdir
            assert database_root.name.startswith("bspp-preprocessing-qualification.12345.")
            assert not database_root.exists()
        else:
            assert not marker.exists()
        assert list(slurm_tmpdir.iterdir()) == []


def test_scheduled_gpu_evidence_returns_exact_multiline_value(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = "GPU A, 555.1\nGPU B, 555.1"
    monkeypatch.setenv("BSPP_PREPROCESSING_GPU_EVIDENCE", expected)
    assert _image_smoke_module()._scheduled_gpu_evidence() == expected  # type: ignore[attr-defined]


def test_image_smoke_fixture_binds_staging_to_the_effective_unix_user(tmp_path: Path) -> None:
    profile_path, source_repo = _profile(tmp_path / "profile")
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    image_identity = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo).image_identity
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()

    runspec, _, _ = _image_smoke_module()._fixture(fixture_root, image_identity)  # type: ignore[attr-defined]

    staging = runspec.payload.database.staging
    assert staging is not None
    assert staging.unix_user == pwd.getpwuid(os.geteuid()).pw_name
    assert runspec.payload.database.requested_policy is DatabaseAccessPolicy.DIRECT

    result_path = _image_smoke_module()._write_direct_placement_result(  # type: ignore[attr-defined]
        runspec,
        fixture_root,
        selected_root=tmp_path / "selected-database",
    )
    result = database_placement_result_from_mapping(json.loads(result_path.read_text()))
    assert result.phase_runspec_digest == runspec.digest
    assert result.pre_science_observation.members == runspec.payload.database.source_manifest.members


@pytest.mark.parametrize("value", [None, "", " \t\n"])
def test_scheduled_gpu_evidence_rejects_missing_or_blank_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv("BSPP_PREPROCESSING_GPU_EVIDENCE", raising=False)
    else:
        monkeypatch.setenv("BSPP_PREPROCESSING_GPU_EVIDENCE", value)
    with pytest.raises(RuntimeError, match="host nvidia-smi GPU evidence"):
        _image_smoke_module()._scheduled_gpu_evidence()  # type: ignore[attr-defined]


def test_submitted_intent_precedes_sbatch_and_never_satisfies_gate(tmp_path: Path) -> None:
    profile_path, source_repo = _profile(tmp_path)
    observed: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> CommandResult:
        observed.append(argv)
        profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
        qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
        path = preprocessing_runtime_qualification_path(
            profile, tuple_id=preprocessing_runtime_tuple_id(qualification_tuple)
        )
        persisted = preprocessing_runtime_qualification_record_from_mapping(json.loads(path.read_text()))
        assert persisted.status == "submitted" and persisted.job_id is None
        with (
            Path(f"{path}.lock").open("a+") as contender,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return CommandResult(argv=argv, returncode=0, stdout="12345\n", stderr="")

    record = qualify_preprocessing_runtime(
        profile_name="example-cluster", config_path=profile_path, source_repo=source_repo, now=NOW, runner=runner
    )
    assert record.status == "submitted" and record.job_id == "12345"
    assert observed and observed[0][:2] == ("sbatch", "--parsable")
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    path = preprocessing_runtime_qualification_path(
        profile, tuple_id=preprocessing_runtime_tuple_id(qualification_tuple)
    )
    with Path(f"{path}.lock").open("a+") as contender:
        fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(PreprocessingRuntimeQualificationError, match="submitted"):
        check_preprocessing_runtime_qualification(profile=profile, source_repo=source_repo, now=NOW)


def test_failed_submission_leaves_durable_tuple_correlated_intent(tmp_path: Path) -> None:
    profile_path, source_repo = _profile(tmp_path)

    def runner(argv: tuple[str, ...]) -> CommandResult:
        script_path = Path(argv[-1])
        profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
        qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
        tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)
        assert script_path.name == f"{tuple_id}.smoke.sbatch"
        assert f"#SBATCH --job-name=bspp_preprocess_qual_{tuple_id[:12]}" in script_path.read_text()
        return CommandResult(argv=argv, returncode=1, stdout="", stderr="scheduler unavailable")

    with pytest.raises(ValueError, match="scheduler unavailable"):
        qualify_preprocessing_runtime(
            profile_name="example-cluster", config_path=profile_path, source_repo=source_repo, now=NOW, runner=runner
        )

    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)
    record_path = preprocessing_runtime_qualification_path(profile, tuple_id=tuple_id)
    record = preprocessing_runtime_qualification_record_from_mapping(json.loads(record_path.read_text()))
    assert record.status == "submitted" and record.job_id is None


def test_ssh_qualification_stages_small_authority_and_submits_the_cluster_script(tmp_path: Path) -> None:
    profile_path, source_repo = _ssh_profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)
    remote_record = f"/cluster/qualifications/preprocessing/example-cluster/{tuple_id}.json"
    calls: list[tuple[str, ...]] = []
    copied_sources: list[Path] = []
    immutable_authority_exists = False

    def rendered(call: tuple[str, ...]) -> str:
        return " ".join(maybe_unwrap_remote_command(part) for part in call)

    def runner(argv: tuple[str, ...]) -> CommandResult:
        nonlocal immutable_authority_exists
        calls.append(argv)
        if argv[0] == "scp":
            source = Path(argv[1])
            copied_sources.append(source)
            return CommandResult(argv=argv, returncode=0, stdout="", stderr="")
        rendered = " ".join(maybe_unwrap_remote_command(part) for part in argv)
        if "immutable artifact collision" in rendered and f"target={remote_record}\n" in rendered:
            if immutable_authority_exists:
                return CommandResult(argv=argv, returncode=73, stdout="", stderr="immutable artifact collision")
            immutable_authority_exists = True
        if "sha256sum" in rendered:
            return CommandResult(
                argv=argv,
                returncode=0,
                stdout=f"{hashlib.sha256(copied_sources[-1].read_bytes()).hexdigest()}  staged\n",
                stderr="",
            )
        if "sbatch --parsable" in rendered:
            return CommandResult(argv=argv, returncode=0, stdout="12345;example-cluster\n", stderr="")
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")

    record = qualify_preprocessing_runtime(
        profile_name="example-cluster",
        config_path=profile_path,
        source_repo=source_repo,
        now=NOW,
        runner=runner,
    )
    refreshed = qualify_preprocessing_runtime(
        profile_name="example-cluster",
        config_path=profile_path,
        source_repo=source_repo,
        now=NOW + timedelta(hours=1),
        runner=runner,
    )

    local_record = preprocessing_runtime_qualification_path(profile, tuple_id=record.tuple_id)
    remote_root = f"/cluster/qualifications/preprocessing/example-cluster/{record.tuple_id}"
    assert local_record == (
        tmp_path / "workstation-qualifications" / "preprocessing" / "example-cluster" / f"{record.tuple_id}.json"
    )
    assert record.job_id == "12345"
    assert refreshed.job_id == "12345"
    assert {path.suffix for path in copied_sources} == {".json", ".sbatch"}
    assert any(call[0] == "scp" and call[-1].startswith(f"example-cluster-login:{remote_root}") for call in calls)
    submit = next(call for call in calls if "sbatch --parsable" in rendered(call))
    assert f"{remote_root}.smoke.sbatch" in rendered(submit)
    assert str(tmp_path / "workstation-qualifications") not in rendered(submit)
    replacements = [call for call in calls if "flock -x 9" in rendered(call)]
    assert len(replacements) == 2
    assert all('mv -f -- "$temporary" "$target"' in rendered(call) for call in replacements)
    assert all(rendered(call).count("os.fsync") == 2 for call in replacements)
    assert any(f"mkdir -p {remote_root.rsplit('/', 1)[0]}/slurm-logs" in rendered(call) for call in calls)


def test_ssh_qualification_resolve_mirrors_only_matching_qualified_authority(tmp_path: Path) -> None:
    profile_path, source_repo = _ssh_profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    remote_record = _qualified_record(qualification_tuple)
    remote_bytes = (json.dumps(remote_record.to_mapping(), indent=2, sort_keys=True) + "\n").encode()
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> CommandResult:
        calls.append(argv)
        return CommandResult(
            argv=argv,
            returncode=0,
            stdout=base64.b64encode(remote_bytes).decode(),
            stderr="",
        )

    resolved = resolve_preprocessing_runtime_qualification(
        profile_name="example-cluster",
        config_path=profile_path,
        source_repo=source_repo,
        now=NOW,
        runner=runner,
    )

    local_path = preprocessing_runtime_qualification_path(profile, tuple_id=remote_record.tuple_id)
    assert resolved == remote_record
    assert local_path.read_bytes() == remote_bytes
    assert len(calls) == 1
    assert calls[0][:2] == ("ssh", "example-cluster-login")
    assert (
        f"/cluster/qualifications/preprocessing/example-cluster/{remote_record.tuple_id}.json"
        in maybe_unwrap_remote_command(calls[0][-1])
    )


def test_current_strict_record_selects_complete_identity(tmp_path: Path) -> None:
    profile_path, source_repo = _profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    record = _qualified_record(qualification_tuple)
    path = preprocessing_runtime_qualification_path(profile, tuple_id=record.tuple_id)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record.to_mapping()))
    selection = check_preprocessing_runtime_qualification(profile=profile, source_repo=source_repo, now=NOW)
    assert selection.qualification_tuple == qualification_tuple
    assert selection.qualification_record_path == str(path)


@pytest.mark.parametrize(
    ("identity_path", "replacement", "matching_smoke_paths"),
    [
        (("cluster_profile",), "other-cluster", ()),
        (("gpu_worker_gres",), "gpu:2", ()),
        (("cluster_image_path",), "/images/other.sqsh", ()),
        (
            ("cluster_image_sha256",),
            "2" * 64,
            (("smoke_evidence", "image", "cluster_image_sha256"),),
        ),
        (("oci_digest",), "sha256:" + "3" * 64, (("smoke_evidence", "image", "oci_digest"),)),
        (
            ("source_bundle_id",),
            "bspp-orchestration-other",
            (("smoke_evidence", "source", "bundle_id"),),
        ),
        (
            ("source_bundle_path",),
            "/bundles/other.tar.zst",
            (("smoke_evidence", "source", "bundle_path"),),
        ),
        (
            ("source_bundle_sha256",),
            "4" * 64,
            (("smoke_evidence", "source", "bundle_sha256"),),
        ),
        (("image_identity", "source_commit"), "9" * 40, ()),
        (("image_identity", "image_lock_sha256"), "5" * 64, ()),
        (("image_identity", "contract_wheel_sha256"), "6" * 64, ()),
        (("image_identity", "runtime_wheel_sha256"), "7" * 64, ()),
        (
            ("image_identity", "colabfold_version"),
            "1.6.3",
            (("smoke_evidence", "tools", "colabfold_version"),),
        ),
        (
            ("image_identity", "mmseqs_version"),
            "other-mmseqs",
            (("smoke_evidence", "tools", "mmseqs_version"),),
        ),
        (
            ("image_identity", "rsync_version"),
            "3.4.5",
            (("smoke_evidence", "tools", "rsync_version"),),
        ),
        (("image_identity", "cuda_version"), "12.7.0", ()),
        (("nodelist",), "gpu-node-018", ()),
    ],
    ids=[
        "profile",
        "gres",
        "image-path",
        "image-sha",
        "oci-digest",
        "source-bundle-id",
        "source-bundle-path",
        "source-bundle-sha",
        "source-commit",
        "image-lock-sha",
        "contract-wheel-sha",
        "runtime-wheel-sha",
        "colabfold-version",
        "mmseqs-version",
        "rsync-version",
        "cuda-version",
        "nodelist",
    ],
)
def test_check_rejects_every_mismatched_qualified_tuple_identity(
    tmp_path: Path,
    identity_path: tuple[str, ...],
    replacement: str,
    matching_smoke_paths: tuple[tuple[str, ...], ...],
) -> None:
    profile_path, source_repo = _profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    expected = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    expected_path = preprocessing_runtime_qualification_path(profile, tuple_id=preprocessing_runtime_tuple_id(expected))
    mapping = deepcopy(_qualified_record(expected).to_mapping())
    qualification_tuple = _mapping_at(mapping, ("qualification_tuple",))
    _set_mapping_value(qualification_tuple, identity_path, replacement)
    for smoke_path in matching_smoke_paths:
        _set_mapping_value(mapping, smoke_path, replacement)
    mapping["tuple_id"] = _canonical_mapping_sha256(qualification_tuple)
    expected_path.parent.mkdir(parents=True)
    expected_path.write_text(json.dumps(mapping))

    with pytest.raises(PreprocessingRuntimeQualificationError, match=r"does not match|missing or invalid"):
        check_preprocessing_runtime_qualification(profile=profile, source_repo=source_repo, now=NOW)


def test_check_rejects_missing_and_expired_qualification_records(tmp_path: Path) -> None:
    missing_profile_path, missing_source_repo = _profile(tmp_path / "missing")
    missing_profile = resolve_cluster_profile("example-cluster", config_path=missing_profile_path)
    with pytest.raises(PreprocessingRuntimeQualificationError, match="missing or invalid"):
        check_preprocessing_runtime_qualification(
            profile=missing_profile,
            source_repo=missing_source_repo,
            now=NOW,
        )

    expired_profile_path, expired_source_repo = _profile(tmp_path / "expired")
    expired_profile = resolve_cluster_profile("example-cluster", config_path=expired_profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(
        expired_profile,
        source_repo=expired_source_repo,
    )
    mapping = _qualified_record(qualification_tuple).to_mapping()
    mapping["expires_at"] = "2026-08-20T11:00:00.000000Z"
    path = preprocessing_runtime_qualification_path(
        expired_profile,
        tuple_id=preprocessing_runtime_tuple_id(qualification_tuple),
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(mapping))
    with pytest.raises(PreprocessingRuntimeQualificationError, match="expired"):
        check_preprocessing_runtime_qualification(
            profile=expired_profile,
            source_repo=expired_source_repo,
            now=NOW,
        )


@pytest.mark.parametrize(
    ("record_path", "replacement"),
    [
        (("qualification_tuple", "runtime_contract_id"), "0" * 64),
        (("qualification_tuple", "adapter_version"), "unsupported-adapter"),
        (("smoke_evidence",), None),
    ],
    ids=["wrong-runtime-contract", "wrong-adapter", "incomplete-smoke"],
)
def test_check_rejects_unsupported_or_incomplete_qualified_evidence(
    tmp_path: Path,
    record_path: tuple[str, ...],
    replacement: object,
) -> None:
    profile_path, source_repo = _profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    expected = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    mapping = deepcopy(_qualified_record(expected).to_mapping())
    _set_mapping_value(mapping, record_path, replacement)
    qualification_tuple = _mapping_at(mapping, ("qualification_tuple",))
    mapping["tuple_id"] = _canonical_mapping_sha256(qualification_tuple)
    path = preprocessing_runtime_qualification_path(profile, tuple_id=preprocessing_runtime_tuple_id(expected))
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(mapping))

    with pytest.raises(PreprocessingRuntimeQualificationError, match="missing or invalid"):
        check_preprocessing_runtime_qualification(profile=profile, source_repo=source_repo, now=NOW)


def _profile(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_repo = tmp_path / "source"
    source_repo.mkdir()
    subprocess.run(("git", "init", "-q", str(source_repo)), check=True)
    subprocess.run(("git", "-C", str(source_repo), "config", "user.email", "test@example.invalid"), check=True)
    subprocess.run(("git", "-C", str(source_repo), "config", "user.name", "Test"), check=True)
    (source_repo / "tracked").write_text("source\n")
    subprocess.run(("git", "-C", str(source_repo), "add", "tracked"), check=True)
    subprocess.run(("git", "-C", str(source_repo), "commit", "-qm", "source"), check=True)
    commit = subprocess.run(
        ("git", "-C", str(source_repo), "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    profile_path = tmp_path / "profiles.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "clusters": {
                    "example-cluster": {
                        "owner": "tester",
                        "database_access_policies": ["direct"],
                        "transport": "local-slurm",
                        "paths": {
                            "project_root": "/project",
                            "output_root": "/output",
                            "staging_root": "/staging",
                            "afdb_toolkit_repo": "/toolkit",
                            "orchestration_repo": str(source_repo),
                            "image": "/images/postprocessing.sqsh",
                            "source_bundle_root": str(tmp_path / "bundles"),
                            "runtime_qualification_root": str(tmp_path / "qualification"),
                        },
                        "preprocessing_runtime": {
                            "cluster_image_path": "/images/preprocessing.sqsh",
                            "cluster_image_sha256": "a" * 64,
                            "oci_digest": "sha256:" + "b" * 64,
                            "image_lock_sha256": "c" * 64,
                            "contract_wheel_sha256": "d" * 64,
                            "runtime_wheel_sha256": "e" * 64,
                            "control_wheel_sha256": "f" * 64,
                            "source_commit": commit,
                            "source_bundle_sha256": "1" * 64,
                            "colabfold_version": "1.6.2",
                            "mmseqs_version": "18-8cc5c",
                            "rsync_version": "3.4.4",
                            "cuda_version": "12.6.3",
                        },
                    }
                }
            }
        )
    )
    return profile_path, source_repo


def _ssh_profile(tmp_path: Path) -> tuple[Path, Path]:
    profile_path, source_repo = _profile(tmp_path)
    payload = yaml.safe_load(profile_path.read_text())
    profile = payload["clusters"]["example-cluster"]
    profile["transport"] = "ssh"
    profile["ssh_target"] = "example-cluster-login"
    profile["paths"]["runtime_qualification_root"] = "/cluster/qualifications"
    profile["paths"]["runtime_qualification_control_root"] = str(tmp_path / "workstation-qualifications")
    profile_path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return profile_path, source_repo


def _render_qualification_script_with_real_image(root: Path) -> str:
    profile_path, source_repo = _profile(root)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    image = root / "preprocessing.sqsh"
    image.write_bytes(b"immutable preprocessing image\n")
    qualification_tuple = replace(
        preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo),
        cluster_image_path=str(image),
        cluster_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
    )
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)
    record_path = preprocessing_runtime_qualification_path(profile, tuple_id=tuple_id)
    script = render_preprocessing_runtime_qualification_script(
        profile=profile,
        qualification_tuple=qualification_tuple,
        tuple_id=tuple_id,
        record_path=record_path,
    )
    script_path = root / "qualification.sbatch"
    script_path.write_text(script)
    return str(script_path)


def _write_shell_stub(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


def _image_smoke_module() -> ModuleType:
    script_path = Path(__file__).parents[1] / "containers" / "preprocessing" / "image-smoke.py"
    module_name = "bspp_preprocessing_image_smoke_test"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _qualified_record(qualification_tuple: object) -> PreprocessingRuntimeQualificationRecord:
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)  # type: ignore[arg-type]
    smoke = PreprocessingRuntimeSmokeEvidence(
        runtime_command=PREPROCESSING_RUNTIME_COMMAND,
        runtime_contract_id=PREPROCESSING_RUNTIME_CONTRACT_ID,
        adapter_version=PREPROCESSING_ADAPTER_VERSION,
        command_order=PREPROCESSING_COMMAND_ORDER,
        action_evidence_sha256="f" * 64,
        tools=PreprocessingRuntimeToolEvidence(
            python_version="Python 3.12.0",
            contract_version="0.1.0",
            runtime_version="0.1.0",
            control_version="0.1.0",
            mmseqs_version="18-8cc5c",
            rsync_version="3.4.4",
            colabfold_version="1.6.2",
            tar_version="tar 1.35",
            lz4_version="lz4 1.10",
            flock_version="flock 2.42",
        ),
        image=PreprocessingRuntimeImageEvidence(
            manifest_path="/opt/bspp/preprocessing-runtime-image.json",
            manifest_sha256="0" * 64,
            cluster_image_sha256="a" * 64,
            oci_digest="sha256:" + "b" * 64,
        ),
        source=PreprocessingRuntimeSourceEvidence(
            bundle_id=qualification_tuple.source_bundle_id,  # type: ignore[attr-defined]
            bundle_path=qualification_tuple.source_bundle_path,  # type: ignore[attr-defined]
            bundle_sha256="1" * 64,
        ),
        gpu=PreprocessingRuntimeGpuEvidence(nvidia_smi="Fixture GPU"),
    )
    return PreprocessingRuntimeQualificationRecord(
        status="qualified",
        tuple_id=tuple_id,
        qualification_tuple=qualification_tuple,  # type: ignore[arg-type]
        submitted_at="2026-08-20T09:00:00.000000Z",
        qualified_at="2026-08-20T10:00:00.000000Z",
        expires_at="2026-08-27T10:00:00.000000Z",
        job_id="12345",
        smoke_evidence=smoke,
    )


def _mapping_at(mapping: dict[str, object], path: tuple[str, ...]) -> dict[str, object]:
    current = mapping
    for field_name in path:
        value = current[field_name]
        assert isinstance(value, dict)
        current = value
    return current


def _set_mapping_value(mapping: dict[str, object], path: tuple[str, ...], value: object) -> None:
    parent = _mapping_at(mapping, path[:-1])
    parent[path[-1]] = value


def _canonical_mapping_sha256(mapping: dict[str, object]) -> str:
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _packed_profile(tmp_path: Path) -> tuple[Path, Path]:
    """Cluster Profile with typed topology (nodes+gpus_per_task) and no legacy gres."""
    profile_path, source_repo = _profile(tmp_path)
    payload = yaml.safe_load(profile_path.read_text())
    cluster = payload["clusters"]["example-cluster"]
    cluster["resources"] = {
        "gpu_worker": {
            "partition": "example-gpu",
            "cpus_per_task": 30,
            "memory": "128G",
            "time": "04:00:00",
            "gres": None,
            "nodes": 2,
            "tasks_per_node": 5,
            "gpus_per_task": 1,
        }
    }
    profile_path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return profile_path, source_repo


def test_packed_profile_derives_gres_from_typed_topology(tmp_path: Path) -> None:
    """A packed-topology profile (nodes+gpus_per_task, no gres) derives gpu:<gpus_per_task>."""
    profile_path, source_repo = _packed_profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    assert qualification_tuple.gpu_worker_gres == "gpu:1"


def test_packed_profile_renders_derived_gres_in_smoke_script(tmp_path: Path) -> None:
    """The qualification smoke script renders the derived --gres line for a packed profile."""
    profile_path, source_repo = _packed_profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)
    record_path = preprocessing_runtime_qualification_path(profile, tuple_id=tuple_id)
    script = render_preprocessing_runtime_qualification_script(
        profile=profile, qualification_tuple=qualification_tuple, tuple_id=tuple_id, record_path=record_path
    )
    assert "#SBATCH --gres=gpu:1" in script
    subprocess.run(("bash", "-n"), input=script, check=True, text=True)


def test_legacy_profile_gres_is_byte_identical_regression(tmp_path: Path) -> None:
    """Legacy profiles with an explicit gres string produce the same tuple and script as before."""
    profile_path, source_repo = _profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    resource = profile.resources["gpu_worker"]
    assert resource.gres == "gpu:1"
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    assert qualification_tuple.gpu_worker_gres == "gpu:1"
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)
    record_path = preprocessing_runtime_qualification_path(profile, tuple_id=tuple_id)
    script = render_preprocessing_runtime_qualification_script(
        profile=profile, qualification_tuple=qualification_tuple, tuple_id=tuple_id, record_path=record_path
    )
    assert "#SBATCH --gres=gpu:1" in script


def test_derive_smoke_gres_returns_legacy_gres_unchanged() -> None:
    """derive_smoke_gres returns the legacy gres string byte-identical."""
    assert derive_smoke_gres(gres="gpu:4", nodes=None, gpus_per_task=None, profile_name="p") == "gpu:4"


def test_derive_smoke_gres_derives_from_typed_topology() -> None:
    """derive_smoke_gres produces gpu:<gpus_per_task> from typed topology when gres is absent."""
    assert derive_smoke_gres(gres=None, nodes=2, gpus_per_task=1, profile_name="p") == "gpu:1"
    assert derive_smoke_gres(gres=None, nodes=4, gpus_per_task=3, profile_name="p") == "gpu:3"


def test_derive_smoke_gres_fails_closed_without_gres_or_topology() -> None:
    """A profile with neither gres nor typed topology must not qualify silently."""
    with pytest.raises(PreprocessingRuntimeQualificationError, match="gres or typed topology"):
        derive_smoke_gres(gres=None, nodes=None, gpus_per_task=None, profile_name="no-gpu")


def test_qualification_tuple_fails_closed_for_profile_without_gres_or_topology(tmp_path: Path) -> None:
    """End-to-end: a profile whose gpu_worker has no gres and no typed topology fails closed."""
    profile_path, source_repo = _profile(tmp_path)
    payload = yaml.safe_load(profile_path.read_text())
    cluster = payload["clusters"]["example-cluster"]
    cluster["resources"] = {
        "gpu_worker": {
            "partition": "example-gpu",
            "cpus_per_task": 30,
            "memory": "128G",
            "time": "04:00:00",
            "gres": None,
        }
    }
    profile_path.write_text(yaml.safe_dump(payload, sort_keys=True))
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    with pytest.raises(PreprocessingRuntimeQualificationError, match="gres or typed topology"):
        preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)


def test_packed_profile_qualification_tuple_contract_round_trip(tmp_path: Path) -> None:
    """A derived qualification tuple serialises and deserialises through the contract."""
    profile_path, source_repo = _packed_profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    assert qualification_tuple.gpu_worker_gres == "gpu:1"
    mapping = qualification_tuple.to_mapping()
    assert mapping["gpu_worker_gres"] == "gpu:1"
    from bspp.orchestration.contract.preprocessing_runtime import (
        preprocessing_runtime_qualification_tuple_from_mapping,
    )

    restored = preprocessing_runtime_qualification_tuple_from_mapping(mapping)
    assert restored == qualification_tuple
    assert restored.gpu_worker_gres == "gpu:1"


def test_packed_profile_qualified_record_contract_round_trip(tmp_path: Path) -> None:
    """A qualified record with a derived gres value round-trips through the contract."""
    profile_path, source_repo = _packed_profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    record = _qualified_record(qualification_tuple)
    assert record.qualification_tuple.gpu_worker_gres == "gpu:1"
    mapping = record.to_mapping()
    assert mapping["qualification_tuple"]["gpu_worker_gres"] == "gpu:1"
    restored = preprocessing_runtime_qualification_record_from_mapping(mapping)
    assert restored == record
    assert restored.qualification_tuple.gpu_worker_gres == "gpu:1"


def test_packed_profile_check_passes(tmp_path: Path) -> None:
    """A packed profile qualification passes the check gate end-to-end."""
    profile_path, source_repo = _packed_profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=profile_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    record = _qualified_record(qualification_tuple)
    path = preprocessing_runtime_qualification_path(profile, tuple_id=record.tuple_id)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record.to_mapping()))
    selection = check_preprocessing_runtime_qualification(profile=profile, source_repo=source_repo, now=NOW)
    assert selection.qualification_tuple == qualification_tuple
    assert selection.qualification_tuple.gpu_worker_gres == "gpu:1"
