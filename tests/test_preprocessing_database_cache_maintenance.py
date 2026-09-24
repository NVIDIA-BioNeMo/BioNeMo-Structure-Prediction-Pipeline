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

"""Public Runtime tests for acceptance Database Replica cache maintenance."""

from __future__ import annotations

import ctypes
import fcntl
import functools
import hashlib
import json
import multiprocessing
import os
import pwd
import shutil
import stat
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from bspp.orchestration.contract.database_cache_maintenance import (
    DatabaseCacheIdentityLockObservation,
    load_database_cache_maintenance_evidence,
)
from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.preprocessing import _database_cache_authority as maintenance_authority
from bspp.orchestration.runtime.preprocessing import _database_cache_evidence as maintenance_evidence
from bspp.orchestration.runtime.preprocessing import _database_cache_scope as maintenance_scope
from bspp.orchestration.runtime.preprocessing import _linux_mount_authority as mount_authority
from bspp.orchestration.runtime.preprocessing import database_cache_maintenance as maintenance_runtime
from bspp.orchestration.runtime.preprocessing._database_cache_maintenance_types import (
    DatabaseCacheEvidenceStore,
    DatabaseCacheMaintenancePaths,
    OwnedTreeRemover,
)
from bspp.orchestration.runtime.preprocessing._database_replica_errors import ClassifiedDatabaseReplicaError
from bspp.orchestration.runtime.preprocessing._owned_tree import remove_owned_tree

_FINAL = "a" * 64
_OTHER_FINAL = "b" * 64
_POPULATION = f".population-{_FINAL}-{'c' * 32}"
_BIND_NAMESPACE_CHILD = "BSPP_DATABASE_CACHE_BIND_NAMESPACE_CHILD"
_BIND_NAMESPACE_ROOT = "BSPP_DATABASE_CACHE_BIND_NAMESPACE_ROOT"
_AUTHORITY_NAMESPACE_CHILD = "BSPP_DATABASE_CACHE_AUTHORITY_NAMESPACE_CHILD"
_AUTHORITY_NAMESPACE_ROOT = "BSPP_DATABASE_CACHE_AUTHORITY_NAMESPACE_ROOT"
_AUTHORITY_NAMESPACE_NODE = "BSPP_DATABASE_CACHE_AUTHORITY_NAMESPACE_NODE"


def test_database_cache_clear_has_exact_profile_only_cli() -> None:
    result = CliRunner().invoke(cli, ["preprocessing", "database-cache", "clear", "--help"])

    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--profile" in result.output
    assert "--write-evidence" in result.output
    assert "cache-root" not in result.output
    assert "unix-user" not in result.output
    assert "manifest" not in result.output


def test_database_cache_clear_exact_scope_and_bounded_evidence(tmp_path: Path) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    sibling_root, sibling_user = _cache(tmp_path / "sibling" / "acceptance" / "cache")
    production_root, production_user = _cache(tmp_path / "production" / "database" / "cache")
    final = _entry(user_root, _FINAL, payload=b"do-not-inventory-this-final-payload")
    population = _entry(user_root, _POPULATION, payload=b"do-not-inventory-this-population-payload")
    selected_locks_inode = (user_root / ".locks").stat().st_ino
    sibling_marker = _entry(sibling_user, _OTHER_FINAL, payload=b"sibling")
    production_marker = _entry(production_user, _OTHER_FINAL, payload=b"production")
    other_user = cache_root / "users" / "somebody-else"
    other_user.mkdir(mode=0o700)
    other_marker = other_user / "keep"
    other_marker.write_bytes(b"other-user")
    evidence_path = tmp_path / "evidence" / "clear.json"
    config_path = _config(
        tmp_path,
        selected=cache_root,
        sibling=sibling_root,
        production=production_root,
    )

    result = CliRunner().invoke(
        cli,
        [
            "preprocessing",
            "database-cache",
            "clear",
            "--config",
            str(config_path),
            "--profile",
            "selected",
            "--write-evidence",
            str(evidence_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert not final.exists()
    assert not population.exists()
    assert (user_root / ".locks").stat().st_ino == selected_locks_inode
    assert (sibling_marker / "payload").read_bytes() == b"sibling"
    assert (production_marker / "payload").read_bytes() == b"production"
    assert other_marker.read_bytes() == b"other-user"
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "cleared"
    assert evidence.removed_count == 2
    assert tuple(item.basename for item in evidence.removed_entry_samples) == (_POPULATION, _FINAL)
    assert evidence.removed_entries_sha256 != hashlib.sha256(f"{_POPULATION}\n{_FINAL}\n".encode()).hexdigest()
    assert evidence.omitted_count == 0
    assert evidence.cache_lock_acquired is True
    assert evidence.identity_lock_observations[0].source_manifest_sha256 == _FINAL
    raw = evidence_path.read_bytes()
    assert b"do-not-inventory" not in raw
    assert stat_mode(evidence_path) == 0o444


def test_database_cache_clear_empty_is_idempotent_without_manufacturing_lock_state(tmp_path: Path) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache", locks=False, replicas=False)
    config_path = _config(tmp_path, selected=cache_root)

    for ordinal in range(2):
        evidence_path = tmp_path / "evidence" / f"empty-{ordinal}.json"
        result = _invoke(config_path, evidence_path)
        assert result.exit_code == 0, result.output
        evidence = load_database_cache_maintenance_evidence(evidence_path)
        assert evidence.terminal_result == "already-empty"
        assert evidence.removed_count == 0

    assert not (user_root / ".locks").exists()
    assert not (user_root / "replicas").exists()


@pytest.mark.parametrize("unknown", ["unknown", ".population-not-an-identity", "A" * 64])
def test_database_cache_clear_refuses_unknown_top_level_entry_without_deleting(
    tmp_path: Path,
    unknown: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    known = _entry(user_root, _FINAL, payload=b"keep")
    malformed = _entry(user_root, unknown, payload=b"keep-unknown")
    evidence_path = tmp_path / "evidence" / "refused.json"

    result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)

    assert result.exit_code != 0
    assert known.exists()
    assert malformed.exists()
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "refused"
    assert evidence.removed_count == 0


@pytest.mark.parametrize("reader_kind", ["scientific-lease", "warm-reader"])
def test_database_cache_clear_active_shared_reader_deletes_nothing(
    tmp_path: Path,
    reader_kind: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    first = _entry(user_root, _FINAL, payload=b"first")
    second = _entry(user_root, _OTHER_FINAL, payload=b"second")
    held = os.open(user_root / ".locks" / f"{_OTHER_FINAL}.lock", os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(held, fcntl.LOCK_SH | fcntl.LOCK_NB)
    evidence_path = tmp_path / "evidence" / "contended.json"
    try:
        result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)
    finally:
        os.close(held)

    assert result.exit_code != 0
    assert first.exists() and second.exists()
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "refused"
    assert evidence.removed_count == 0
    assert [item.observation for item in evidence.identity_lock_observations] == ["acquired", "contended"]
    assert reader_kind in {"scientific-lease", "warm-reader"}


@pytest.mark.parametrize("population_kind", ["same-identity", "different-identity"])
def test_database_cache_clear_active_population_deletes_nothing(
    tmp_path: Path,
    population_kind: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    identity = _FINAL if population_kind == "same-identity" else _OTHER_FINAL
    population_name = f".population-{identity}-{'d' * 32}"
    population = _entry(user_root, population_name, payload=b"active-population")
    sibling = _entry(user_root, _FINAL, payload=b"sibling-final") if identity == _OTHER_FINAL else None
    held = os.open(user_root / ".locks" / f"{identity}.lock", os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    evidence_path = tmp_path / "evidence" / f"{population_kind}.json"
    try:
        result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)
    finally:
        os.close(held)

    assert result.exit_code != 0
    assert population.exists()
    assert sibling is None or sibling.exists()
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "refused"
    assert evidence.removed_count == 0
    assert evidence.identity_lock_observations[-1].observation == "contended"


def test_database_cache_clear_waits_boundedly_for_cache_then_clears(tmp_path: Path) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"clear-after-cache-release")
    context = multiprocessing.get_context("fork")
    ready = context.Event()

    def hold_cache_temporarily() -> None:
        descriptor = os.open(user_root / ".locks" / "cache.lock", os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            ready.set()
            time.sleep(0.15)
        finally:
            os.close(descriptor)

    holder = context.Process(target=hold_cache_temporarily)
    holder.start()
    try:
        assert ready.wait(timeout=5)
        evidence_path = tmp_path / "evidence" / "cache-contended.json"
        result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)
        holder.join(timeout=5)
    finally:
        if holder.is_alive():
            holder.terminate()
        holder.join(timeout=5)

    assert holder.exitcode == 0
    assert result.exit_code == 0, result.output
    assert not entry.exists()
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "cleared"
    assert evidence.cache_lock_contended is True


def test_identity_holder_waiting_for_cache_cannot_deadlock_maintenance(
    tmp_path: Path,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"must-survive-contention")
    context = multiprocessing.get_context("fork")
    identity_ready = context.Event()
    maintenance_has_cache = context.Event()
    population_got_cache = context.Event()

    def identity_then_cache_population() -> None:
        identity_fd = os.open(user_root / ".locks" / f"{_FINAL}.lock", os.O_RDWR | os.O_NOFOLLOW)
        cache_fd = os.open(user_root / ".locks" / "cache.lock", os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(identity_fd, fcntl.LOCK_EX)
            identity_ready.set()
            deadline = time.monotonic() + 5
            while True:
                try:
                    fcntl.flock(cache_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    maintenance_has_cache.set()
                    break
                fcntl.flock(cache_fd, fcntl.LOCK_UN)
                if time.monotonic() >= deadline:
                    os._exit(21)
                time.sleep(0.001)
            fcntl.flock(cache_fd, fcntl.LOCK_EX)
            population_got_cache.set()
        finally:
            os.close(cache_fd)
            os.close(identity_fd)

    population = context.Process(target=identity_then_cache_population)
    population.start()
    try:
        assert identity_ready.wait(timeout=5)
        evidence_path = tmp_path / "evidence" / "identity-then-cache.json"
        result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)
        assert population_got_cache.wait(timeout=5)
        population.join(timeout=5)
    finally:
        maintenance_has_cache.set()
        if population.is_alive():
            population.terminate()
        population.join(timeout=5)

    assert population.exitcode == 0
    assert result.exit_code != 0
    assert entry.exists()
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "refused"
    assert evidence.removed_count == 0
    assert evidence.identity_lock_observations == (DatabaseCacheIdentityLockObservation(_FINAL, "contended"),)


def test_database_cache_lock_evidence_is_bounded_without_limiting_cache_size(tmp_path: Path) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    identities = tuple(f"{ordinal:064x}" for ordinal in range(40))
    for identity in identities:
        _entry(user_root, identity, payload=b"bounded-lock-evidence")
        lock_path = user_root / ".locks" / f"{identity}.lock"
        lock_path.touch(mode=0o600)
        lock_path.chmod(0o600)
    evidence_path = tmp_path / "evidence" / "bounded-locks.json"

    result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)

    assert result.exit_code == 0, result.output
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    summary = evidence.identity_locks
    assert summary.total_count == 40
    assert summary.acquired_count == 40
    assert summary.contended_count == 0
    assert summary.observations_sha256 == "15dfac37b8c8f5aa8984782eeaeb2361b4a01881e0915c893259b7ee66738157"
    assert len(summary.samples) == 32
    assert summary.omitted_count == 8
    assert identities[-1].encode() not in evidence_path.read_bytes()
    assert not tuple((user_root / "replicas").iterdir())


def test_database_cache_clear_refuses_cross_user_and_evidence_inside_target(tmp_path: Path) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"keep")
    config_path = _config(tmp_path, selected=cache_root)
    raw = yaml.safe_load(config_path.read_text())
    raw["clusters"]["selected"]["database_cache_unix_user"] = "somebody-else"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=True))

    mismatch = _invoke(config_path, tmp_path / "evidence" / "cross-user.json")
    assert mismatch.exit_code != 0
    assert entry.exists()
    assert (
        load_database_cache_maintenance_evidence(tmp_path / "evidence" / "cross-user.json").terminal_result == "refused"
    )

    raw["clusters"]["selected"]["database_cache_unix_user"] = _user()
    config_path.write_text(yaml.safe_dump(raw, sort_keys=True))
    inside = _invoke(config_path, user_root / "clear.json")
    assert inside.exit_code != 0
    assert entry.exists()
    assert not (user_root / "clear.json").exists()


def test_database_cache_clear_refuses_nonacceptance_or_aliased_profile_before_mutation(tmp_path: Path) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"keep")
    config_path = _config(tmp_path, selected=cache_root, production=cache_root)

    result = _invoke(config_path, tmp_path / "evidence" / "alias.json")

    assert result.exit_code != 0
    assert entry.exists()
    assert not (tmp_path / "evidence" / "alias.json").exists()


def test_database_cache_clear_refuses_symlinked_production_root_alias_without_deletion(tmp_path: Path) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"must-survive-physical-alias")
    production_root = tmp_path / "production" / "database" / "cache"
    production_root.parent.mkdir(parents=True)
    production_root.symlink_to(cache_root, target_is_directory=True)
    evidence_path = tmp_path / "evidence" / "symlink-alias.json"

    result = _invoke(_config(tmp_path, selected=cache_root, production=production_root), evidence_path)

    assert result.exit_code != 0
    assert entry.exists()
    assert load_database_cache_maintenance_evidence(evidence_path).terminal_result == "refused"


def test_database_cache_clear_refuses_same_device_inode_production_alias_boundary(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return
    cache_root, user_root = _cache(case_root / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"must-survive-same-file-alias")
    production_root = case_root / "production" / "database" / "cache"
    production_root.mkdir(parents=True)
    evidence_path = case_root / "evidence" / "same-file-alias.json"
    subprocess.run(("mount", "--bind", str(cache_root), str(production_root)), check=True)
    try:
        result = _invoke(
            _config(case_root, selected=cache_root, production=production_root),
            evidence_path,
        )
    finally:
        subprocess.run(("umount", str(production_root)), check=True)

    assert result.exit_code != 0
    assert entry.exists()
    assert load_database_cache_maintenance_evidence(evidence_path).terminal_result == "refused"


def test_database_cache_clear_rechecks_sibling_root_authority_before_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"must-survive-alias-rebind")
    production_root, production_user = _cache(tmp_path / "production" / "database" / "cache")
    production_marker = _entry(production_user, _OTHER_FINAL, payload=b"production-survives")
    parked_production = tmp_path / "production-root-before-rebind"
    evidence_store = _filesystem_evidence_store()
    real_verify = evidence_store.verify
    rebound = False

    def rebind_sibling_before_removal(*args: object, **kwargs: object) -> None:
        nonlocal rebound
        real_verify(*args, **kwargs)
        if not rebound:
            rebound = True
            production_root.rename(parked_production)
            production_root.symlink_to(cache_root, target_is_directory=True)

    evidence_store = replace(evidence_store, verify=rebind_sibling_before_removal)
    evidence_path = tmp_path / "evidence" / "alias-rebind.json"

    result = _invoke_with_evidence_store(
        _config(tmp_path, selected=cache_root, production=production_root),
        evidence_path,
        evidence_store,
    )

    assert result.exit_code != 0
    assert entry.exists()
    assert (parked_production / "users" / _user() / "replicas" / production_marker.name).exists()
    assert load_database_cache_maintenance_evidence(evidence_path).terminal_result == "refused"


def test_database_cache_clear_refuses_production_root_inside_removable_replica(tmp_path: Path) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    production_root = _entry(user_root, _FINAL, payload=b"production-root-inside-removable-replica")
    production_info = production_root.stat()
    evidence_path = tmp_path / "evidence" / "production-inside-replica.json"

    result = _invoke(_config(tmp_path, selected=cache_root, production=production_root), evidence_path)

    assert production_root.exists()
    assert (production_root.stat().st_dev, production_root.stat().st_ino) == (
        production_info.st_dev,
        production_info.st_ino,
    )
    assert (production_root / "payload").read_bytes() == b"production-root-inside-removable-replica"
    assert result.exit_code != 0
    assert not evidence_path.exists()


def test_database_cache_clear_refuses_selected_root_inside_production_root(tmp_path: Path) -> None:
    production_root = tmp_path / "production" / "database" / "cache"
    production_root.mkdir(parents=True, mode=0o700)
    production_marker = production_root / "production-marker"
    production_marker.write_bytes(b"preserve-production-root")
    cache_root, user_root = _cache(production_root / "nested" / "selected" / "acceptance" / "cache")
    selected_entry = _entry(user_root, _FINAL, payload=b"selected-entry-inside-production-root")
    selected_info = selected_entry.stat()
    evidence_path = tmp_path / "evidence" / "selected-inside-production.json"

    result = _invoke(_config(tmp_path, selected=cache_root, production=production_root), evidence_path)

    assert selected_entry.exists()
    assert (selected_entry.stat().st_dev, selected_entry.stat().st_ino) == (
        selected_info.st_dev,
        selected_info.st_ino,
    )
    assert (selected_entry / "payload").read_bytes() == b"selected-entry-inside-production-root"
    assert production_marker.read_bytes() == b"preserve-production-root"
    assert result.exit_code != 0
    assert not evidence_path.exists()


@pytest.mark.parametrize("relationship", ["production-inside-selected", "selected-inside-production"])
def test_database_cache_clear_refuses_physically_nested_configured_cache_roots(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    relationship: str,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return

    if relationship == "production-inside-selected":
        cache_root, user_root = _cache(case_root / "selected-backing" / "acceptance" / "cache")
        selected_entry = _entry(user_root, _FINAL, payload=b"production-backing-inside-selected")
        production_root = case_root / "configured-production"
        production_root.mkdir(mode=0o700)
        bind_source = selected_entry
        bind_target = production_root
    else:
        production_root = case_root / "production-backing"
        production_root.mkdir(parents=True, mode=0o700)
        production_marker = production_root / "production-marker"
        production_marker.write_bytes(b"preserve-production-backing")
        selected_backing, user_root = _cache(production_root / "nested" / "selected" / "acceptance" / "cache")
        selected_entry = _entry(user_root, _FINAL, payload=b"selected-backing-inside-production")
        cache_root = case_root / "configured-selected"
        cache_root.mkdir(mode=0o700)
        bind_source = selected_backing
        bind_target = cache_root
    selected_info = selected_entry.stat()
    evidence_path = case_root / "safe-evidence" / f"{relationship}.json"
    subprocess.run(("mount", "--bind", str(bind_source), str(bind_target)), check=True)
    try:
        result = _invoke(_config(case_root, selected=cache_root, production=production_root), evidence_path)
    finally:
        subprocess.run(("umount", str(bind_target)), check=True)

    assert selected_entry.exists()
    assert (selected_entry.stat().st_dev, selected_entry.stat().st_ino) == (
        selected_info.st_dev,
        selected_info.st_ino,
    )
    expected_payload = (
        b"production-backing-inside-selected"
        if relationship == "production-inside-selected"
        else b"selected-backing-inside-production"
    )
    assert (selected_entry / "payload").read_bytes() == expected_payload
    if relationship == "selected-inside-production":
        assert production_marker.read_bytes() == b"preserve-production-backing"
    assert result.exit_code != 0
    assert evidence_path.exists()
    assert stat_mode(evidence_path) == 0o444
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "refused"
    assert evidence.removed_count == 0
    assert not tuple(evidence_path.parent.glob(f".{evidence_path.name}.*"))


@pytest.mark.parametrize("backing_kind", ["configured-production", "unconfigured-external"])
def test_database_cache_clear_refuses_replicas_backing_escape_without_deletion(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    backing_kind: str,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return

    cache_root, user_root = _cache(case_root / "selected" / "acceptance" / "cache")
    backing_root = case_root / backing_kind / "database" / "cache"
    backing_root.mkdir(parents=True, mode=0o700)
    backing_entry = backing_root / _FINAL
    backing_entry.mkdir(mode=0o555)
    backing_payload = backing_entry / "payload"
    backing_payload.write_bytes(f"preserve-{backing_kind}-backing".encode())
    backing_root_info = backing_root.stat()
    backing_entry_info = backing_entry.stat()
    config_path = _config(
        case_root,
        selected=cache_root,
        production=backing_root if backing_kind == "configured-production" else None,
    )
    evidence_path = case_root / "safe-evidence" / f"{backing_kind}-replicas-escape.json"
    replicas_path = user_root / "replicas"

    subprocess.run(("mount", "--bind", str(backing_root), str(replicas_path)), check=True)
    try:
        result = _invoke(config_path, evidence_path)
    finally:
        subprocess.run(("umount", str(replicas_path)), check=True)

    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert (
        result.exit_code != 0,
        evidence.terminal_result,
        evidence.removed_count,
        backing_entry.exists(),
    ) == (True, "refused", 0, True)
    assert (backing_root.stat().st_dev, backing_root.stat().st_ino) == (
        backing_root_info.st_dev,
        backing_root_info.st_ino,
    )
    assert (backing_entry.stat().st_dev, backing_entry.stat().st_ino) == (
        backing_entry_info.st_dev,
        backing_entry_info.st_ino,
    )
    assert backing_payload.read_bytes() == f"preserve-{backing_kind}-backing".encode()
    assert stat_mode(evidence_path) == 0o444
    assert not tuple(evidence_path.parent.glob(f".{evidence_path.name}.*"))


def test_database_cache_clear_preserves_production_root_bound_directly_on_accepted_entry(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return

    cache_root, user_root = _cache(case_root / "selected" / "acceptance" / "cache")
    selected_entry = _entry(user_root, _FINAL, payload=b"preserve-selected-entry-under-bind")
    selected_entry_info = selected_entry.stat()
    production_root = case_root / "production" / "database" / "cache"
    production_root.mkdir(parents=True, mode=0o555)
    production_payload = production_root / "payload"
    production_payload.write_bytes(b"preserve-production-entry-mount")
    production_info = production_root.stat()
    evidence_path = case_root / "safe-evidence" / "production-on-accepted-entry.json"

    subprocess.run(("mount", "--bind", str(production_root), str(selected_entry)), check=True)
    try:
        result = _invoke(
            _config(case_root, selected=cache_root, production=production_root),
            evidence_path,
        )
    finally:
        subprocess.run(("umount", str(selected_entry)), check=True)

    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert (result.exit_code != 0, evidence.terminal_result, evidence.removed_count) == (True, "failed", 0)
    assert (production_root.stat().st_dev, production_root.stat().st_ino) == (
        production_info.st_dev,
        production_info.st_ino,
    )
    assert production_payload.read_bytes() == b"preserve-production-entry-mount"
    assert (selected_entry.stat().st_dev, selected_entry.stat().st_ino) == (
        selected_entry_info.st_dev,
        selected_entry_info.st_ino,
    )
    assert (selected_entry / "payload").read_bytes() == b"preserve-selected-entry-under-bind"
    assert stat_mode(evidence_path) == 0o444
    assert not tuple(evidence_path.parent.glob(f".{evidence_path.name}.*"))


def test_database_cache_clear_refuses_bound_locks_backing_before_empty_evidence(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return

    cache_root, user_root = _cache(
        case_root / "selected" / "acceptance" / "cache",
        replicas=False,
    )
    original_locks = user_root / ".locks"
    original_locks_info = original_locks.stat()
    external_locks = case_root / "external-lock-backing"
    external_locks.mkdir(mode=0o700)
    evidence_path = external_locks / "empty.json"

    subprocess.run(("mount", "--bind", str(external_locks), str(original_locks)), check=True)
    try:
        result = _invoke(_config(case_root, selected=cache_root), evidence_path)
        alias_evidence_exists = (original_locks / evidence_path.name).exists()
        backing_evidence_exists = evidence_path.exists()
    finally:
        subprocess.run(("umount", str(original_locks)), check=True)

    assert result.exit_code != 0
    assert (original_locks.stat().st_dev, original_locks.stat().st_ino) == (
        original_locks_info.st_dev,
        original_locks_info.st_ino,
    )
    assert not alias_evidence_exists
    assert not backing_evidence_exists
    assert not (user_root / "replicas").exists()


def test_database_cache_clear_refuses_locks_directory_substitution_with_original_shared_lease(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return

    cache_root, user_root = _cache(case_root / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"preserve-under-original-shared-lease")
    original_locks = user_root / ".locks"
    original_identity = original_locks / f"{_FINAL}.lock"
    original_identity_info = original_identity.stat()
    held = os.open(original_identity, os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(held, fcntl.LOCK_SH | fcntl.LOCK_NB)
    alternate_locks = case_root / "alternate-lock-authority"
    alternate_locks.mkdir(mode=0o700)
    _lock_file(alternate_locks / "cache.lock")
    _lock_file(alternate_locks / f"{_FINAL}.lock")
    evidence_path = case_root / "evidence" / "locks-directory-substitution.json"

    subprocess.run(("mount", "--bind", str(alternate_locks), str(original_locks)), check=True)
    try:
        result = _invoke(_config(case_root, selected=cache_root), evidence_path)
    finally:
        subprocess.run(("umount", str(original_locks)), check=True)
    try:
        evidence = load_database_cache_maintenance_evidence(evidence_path)
        assert (
            result.exit_code != 0,
            evidence.terminal_result,
            evidence.removed_count,
            entry.exists(),
            _exclusive_lock_is_contended(original_identity),
        ) == (True, "refused", 0, True, True)
        assert (original_identity.stat().st_dev, original_identity.stat().st_ino) == (
            original_identity_info.st_dev,
            original_identity_info.st_ino,
        )
        assert (entry / "payload").read_bytes() == b"preserve-under-original-shared-lease"
    finally:
        os.close(held)


def test_database_cache_clear_refuses_identity_lock_file_substitution_with_original_shared_lease(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return

    cache_root, user_root = _cache(case_root / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"preserve-under-original-identity-file-lease")
    identity_path = user_root / ".locks" / f"{_FINAL}.lock"
    identity_info = identity_path.stat()
    held = os.open(identity_path, os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(held, fcntl.LOCK_SH | fcntl.LOCK_NB)
    alternate_identity = _lock_file(case_root / "alternate-identity.lock")
    evidence_path = case_root / "evidence" / "identity-file-substitution.json"

    subprocess.run(("mount", "--bind", str(alternate_identity), str(identity_path)), check=True)
    try:
        result = _invoke(_config(case_root, selected=cache_root), evidence_path)
    finally:
        subprocess.run(("umount", str(identity_path)), check=True)
    try:
        evidence = load_database_cache_maintenance_evidence(evidence_path)
        assert (
            result.exit_code != 0,
            evidence.terminal_result,
            evidence.removed_count,
            entry.exists(),
            _exclusive_lock_is_contended(identity_path),
        ) == (True, "refused", 0, True, True)
        assert (identity_path.stat().st_dev, identity_path.stat().st_ino) == (
            identity_info.st_dev,
            identity_info.st_ino,
        )
        assert (entry / "payload").read_bytes() == b"preserve-under-original-identity-file-lease"
    finally:
        os.close(held)


def test_database_cache_clear_refuses_individual_lock_file_substitution_for_active_population(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return

    cache_root, user_root = _cache(case_root / "selected" / "acceptance" / "cache")
    population = _entry(user_root, _POPULATION, payload=b"preserve-active-population")
    locks = user_root / ".locks"
    identity_path = locks / f"{_FINAL}.lock"
    identity_info = identity_path.stat()
    held_identity = os.open(identity_path, os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(held_identity, fcntl.LOCK_EX | fcntl.LOCK_NB)
    alternate_identity = _lock_file(case_root / "active-population-identity.lock")
    evidence_path = case_root / "evidence" / "active-population-lock-substitution.json"

    subprocess.run(("mount", "--bind", str(alternate_identity), str(identity_path)), check=True)
    try:
        result = _invoke(_config(case_root, selected=cache_root), evidence_path)
    finally:
        subprocess.run(("umount", str(identity_path)), check=True)
    try:
        evidence = load_database_cache_maintenance_evidence(evidence_path)
        assert (
            result.exit_code != 0,
            evidence.terminal_result,
            evidence.removed_count,
            population.exists(),
            _exclusive_lock_is_contended(identity_path),
        ) == (True, "refused", 0, True, True)
        assert (identity_path.stat().st_dev, identity_path.stat().st_ino) == (
            identity_info.st_dev,
            identity_info.st_ino,
        )
        assert (population / "payload").read_bytes() == b"preserve-active-population"
    finally:
        os.close(held_identity)


def test_database_cache_clear_refuses_cache_lock_file_substitution_with_original_cache_owner(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return

    cache_root, user_root = _cache(case_root / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"preserve-under-original-cache-owner")
    locks = user_root / ".locks"
    other_identity_path = locks / f"{_OTHER_FINAL}.lock"
    cache_path = locks / "cache.lock"
    cache_info = cache_path.stat()
    held_other_identity = os.open(other_identity_path, os.O_RDWR | os.O_NOFOLLOW)
    held_cache = os.open(cache_path, os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(held_other_identity, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(held_cache, fcntl.LOCK_EX | fcntl.LOCK_NB)
    alternate_cache = _lock_file(case_root / "alternate-cache.lock")
    evidence_path = case_root / "evidence" / "cache-file-substitution.json"

    subprocess.run(("mount", "--bind", str(alternate_cache), str(cache_path)), check=True)
    try:
        result = _invoke(_config(case_root, selected=cache_root), evidence_path)
    finally:
        subprocess.run(("umount", str(cache_path)), check=True)
    try:
        evidence = load_database_cache_maintenance_evidence(evidence_path)
        assert (
            result.exit_code != 0,
            evidence.terminal_result,
            evidence.removed_count,
            entry.exists(),
            _exclusive_lock_is_contended(cache_path),
        ) == (True, "refused", 0, True, True)
        assert (cache_path.stat().st_dev, cache_path.stat().st_ino) == (
            cache_info.st_dev,
            cache_info.st_ino,
        )
        assert (entry / "payload").read_bytes() == b"preserve-under-original-cache-owner"
    finally:
        os.close(held_cache)
        os.close(held_other_identity)


@pytest.mark.parametrize("symlink_kind", ["terminal", "intermediate"])
def test_database_cache_clear_refuses_configured_sibling_symlink_to_accepted_entry(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    symlink_kind: str,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return

    cache_root, user_root = _cache(case_root / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"preserve-sibling-symlink-target")
    entry_info = entry.stat()
    if symlink_kind == "terminal":
        production_root = case_root / "production" / "database" / "cache"
        production_root.parent.mkdir(parents=True)
        production_root.symlink_to(entry, target_is_directory=True)
    else:
        entry.chmod(0o700)
        nested_production_root = entry / "database" / "cache"
        nested_production_root.mkdir(parents=True, mode=0o700)
        nested_marker = nested_production_root / "production-marker"
        nested_marker.write_bytes(b"preserve-nested-production")
        entry.chmod(0o555)
        production_alias = case_root / "production-alias"
        production_alias.symlink_to(entry, target_is_directory=True)
        production_root = production_alias / "database" / "cache"
    evidence_path = case_root / "evidence" / f"sibling-{symlink_kind}.json"

    result = _invoke(
        _config(case_root, selected=cache_root, production=production_root),
        evidence_path,
    )

    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert (
        result.exit_code != 0,
        evidence.terminal_result,
        evidence.removed_count,
        entry.exists(),
    ) == (True, "refused", 0, True)
    assert (entry.stat().st_dev, entry.stat().st_ino) == (entry_info.st_dev, entry_info.st_ino)
    assert (entry / "payload").read_bytes() == b"preserve-sibling-symlink-target"
    if symlink_kind == "intermediate":
        assert nested_marker.read_bytes() == b"preserve-nested-production"


def test_database_cache_clear_preserves_external_backing_bound_on_population_claim(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return

    cache_root, user_root = _cache(case_root / "selected" / "acceptance" / "cache")
    population = _entry(user_root, _POPULATION, payload=b"preserve-selected-population")
    population_info = population.stat()
    external = case_root / "external-population-backing"
    external.mkdir(mode=0o700)
    external_payload = external / "payload"
    external_payload.write_bytes(b"preserve-external-population")
    external_info = external.stat()
    evidence_path = case_root / "evidence" / "population-claim.json"

    subprocess.run(("mount", "--bind", str(external), str(population)), check=True)
    try:
        result = _invoke(_config(case_root, selected=cache_root), evidence_path)
        evidence = load_database_cache_maintenance_evidence(evidence_path)
        external_payload_during = external_payload.read_bytes()
    finally:
        subprocess.run(("umount", str(population)), check=True)

    assert (result.exit_code != 0, evidence.terminal_result, evidence.removed_count) == (True, "failed", 0)
    assert (external.stat().st_dev, external.stat().st_ino) == (external_info.st_dev, external_info.st_ino)
    assert external_payload_during == b"preserve-external-population"
    assert (population.stat().st_dev, population.stat().st_ino) == (
        population_info.st_dev,
        population_info.st_ino,
    )
    assert (population / "payload").read_bytes() == b"preserve-selected-population"


def test_database_cache_maintenance_evidence_collision_is_exact_only(tmp_path: Path) -> None:
    cache_root, _ = _cache(tmp_path / "selected" / "acceptance" / "cache", replicas=False)
    config_path = _config(tmp_path, selected=cache_root)
    evidence_path = tmp_path / "evidence" / "empty.json"

    assert _invoke(config_path, evidence_path).exit_code == 0
    assert _invoke(config_path, evidence_path).exit_code == 0
    evidence_path.chmod(0o644)
    evidence_path.write_text("{}\n")
    evidence_path.chmod(0o444)
    third = _invoke(config_path, evidence_path)
    assert third.exit_code != 0


def test_evidence_final_name_is_never_observer_writable(
    tmp_path: Path,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"clear-after-immutable-reservation")
    config_path = _config(tmp_path, selected=cache_root)
    evidence_path = tmp_path / "evidence" / "immutable-at-creation.json"
    reservation_visible = threading.Event()
    observer_finished = threading.Event()
    evidence_store = _filesystem_evidence_store()
    real_authorize = evidence_store.authorize

    def pause_after_reservation(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        result = real_authorize(*args, **kwargs)  # type: ignore[arg-type]
        reservation_visible.set()
        assert observer_finished.wait(timeout=5)
        return result

    paused_store = replace(evidence_store, authorize=pause_after_reservation)
    completed: list[object] = []
    worker = threading.Thread(
        target=lambda: completed.append(_invoke_with_evidence_store(config_path, evidence_path, paused_store))
    )
    worker.start()
    observer_descriptor: int | None = None
    observer_open_failed = False
    try:
        assert reservation_visible.wait(timeout=5)
        assert stat_mode(evidence_path) == 0o444
        try:
            observer_descriptor = os.open(evidence_path, os.O_WRONLY | os.O_NOFOLLOW)
        except PermissionError:
            observer_open_failed = True
        observer_finished.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        if observer_descriptor is not None:
            os.lseek(observer_descriptor, 0, os.SEEK_SET)
            os.write(observer_descriptor, b"observer-corruption\n")
    finally:
        observer_finished.set()
        worker.join(timeout=5)
        if observer_descriptor is not None:
            os.close(observer_descriptor)

    assert observer_open_failed
    assert len(completed) == 1
    assert completed[0].exit_code == 0  # type: ignore[union-attr]
    assert not entry.exists()
    assert load_database_cache_maintenance_evidence(evidence_path).terminal_result == "cleared"


def test_evidence_parent_construction_rejects_intermediate_symlink_without_escaped_residue(tmp_path: Path) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"preserve-on-intermediate-evidence-symlink")
    entry_info = entry.stat()
    escaped_root = tmp_path / "escaped-evidence-root"
    escaped_root.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-evidence-parent"
    linked_parent.symlink_to(escaped_root, target_is_directory=True)
    evidence_path = linked_parent / "missing-a" / "missing-b" / "result.json"

    result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)

    assert result.exit_code != 0
    assert entry.exists()
    assert (entry.stat().st_dev, entry.stat().st_ino) == (entry_info.st_dev, entry_info.st_ino)
    assert (entry / "payload").read_bytes() == b"preserve-on-intermediate-evidence-symlink"
    assert not (escaped_root / "missing-a").exists()
    assert not evidence_path.exists()
    assert not tuple(escaped_root.rglob(f".{evidence_path.name}.*"))


def test_evidence_parent_construction_cleans_only_invocation_created_empty_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"preserve-on-evidence-preflight-fault")
    entry_info = entry.stat()
    preexisting_parent = tmp_path / "preexisting-evidence-parent"
    preexisting_parent.mkdir(mode=0o700)
    sentinel = preexisting_parent / "sentinel"
    sentinel.write_bytes(b"preexisting")
    created_root = preexisting_parent / "created-a"
    evidence_path = created_root / "created-b" / "result.json"

    def fail_preflight_write(*_args: object) -> int:
        raise OSError("injected evidence preflight write fault after parent construction")

    monkeypatch.setattr(maintenance_runtime.os, "write", fail_preflight_write)

    result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)

    assert result.exit_code != 0
    assert entry.exists()
    assert (entry.stat().st_dev, entry.stat().st_ino) == (entry_info.st_dev, entry_info.st_ino)
    assert (entry / "payload").read_bytes() == b"preserve-on-evidence-preflight-fault"
    assert preexisting_parent.exists()
    assert sentinel.read_bytes() == b"preexisting"
    assert not created_root.exists()
    assert not evidence_path.exists()
    assert not tuple(preexisting_parent.rglob(f".{evidence_path.name}.*"))


@pytest.mark.parametrize("fault", ["write", "file-fsync", "parent-fsync", "parent-rebind"])
def test_evidence_authority_preflight_faults_before_any_cache_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"keep-on-preflight-failure")
    evidence_path = tmp_path / "evidence" / f"{fault}.json"
    config_path = _config(tmp_path, selected=cache_root)
    if fault == "write":
        monkeypatch.setattr(maintenance_runtime.os, "write", lambda *_args: (_ for _ in ()).throw(OSError("write")))
    else:
        real_fsync = maintenance_runtime.os.fsync
        injected = False
        parked_parent = tmp_path / "evidence-before-rebind"

        def fail_selected_fsync(descriptor: int) -> None:
            nonlocal injected
            if not injected:
                target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
                is_parent = target == evidence_path.parent
                if fault == "parent-rebind" and is_parent:
                    evidence_path.parent.rename(parked_parent)
                    evidence_path.parent.mkdir()
                    injected = True
                    real_fsync(descriptor)
                    return
                if (fault == "parent-fsync" and is_parent) or (fault == "file-fsync" and not is_parent):
                    injected = True
                    raise OSError(fault)
            real_fsync(descriptor)

        monkeypatch.setattr(maintenance_runtime.os, "fsync", fail_selected_fsync)

    result = _invoke(config_path, evidence_path)

    assert result.exit_code != 0
    assert entry.exists()
    assert not evidence_path.exists()
    assert not tuple(evidence_path.parent.glob(f".{evidence_path.name}.*"))


@pytest.mark.parametrize("unsafe_parent", ["symlink", "world-writable", "divergent-destination"])
def test_unsafe_or_preexisting_evidence_destination_blocks_before_deletion(
    tmp_path: Path,
    unsafe_parent: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"keep-on-unsafe-evidence")
    evidence_path = tmp_path / "evidence" / "terminal.json"
    if unsafe_parent == "symlink":
        real_parent = tmp_path / "real-evidence"
        real_parent.mkdir()
        evidence_path.parent.symlink_to(real_parent, target_is_directory=True)
    else:
        evidence_path.parent.mkdir()
        if unsafe_parent == "world-writable":
            evidence_path.parent.chmod(0o777)
        else:
            evidence_path.write_text("{}\n")
            evidence_path.chmod(0o444)

    result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)

    assert result.exit_code != 0
    assert entry.exists()


@pytest.mark.parametrize("alias_scope", ["user-namespace", "descendant"])
def test_database_cache_clear_rejects_physical_evidence_parent_containment_before_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_scope: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"keep-on-physical-evidence-alias")
    evidence_path = tmp_path / "external-evidence" / "physical-alias.json"
    alias_parent = user_root
    if alias_scope == "descendant":
        alias_parent = user_root / "nested-evidence"
        alias_parent.mkdir(mode=0o700)
    real_open = maintenance_runtime.os.open
    real_lstat = Path.lstat

    def open_with_bind_alias(  # type: ignore[no-untyped-def]
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd is None and Path(path) == evidence_path.parent:  # type: ignore[arg-type]
            return real_open(alias_parent, flags, mode)
        if dir_fd is None:
            return real_open(path, flags, mode)  # type: ignore[arg-type]
        return real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    def lstat_with_bind_alias(path: Path) -> os.stat_result:
        return real_lstat(alias_parent if path == evidence_path.parent else path)

    monkeypatch.setattr(maintenance_runtime.os, "open", open_with_bind_alias)
    monkeypatch.setattr(Path, "lstat", lstat_with_bind_alias)

    result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)

    assert result.exit_code != 0
    assert entry.exists()
    assert not evidence_path.exists()
    assert not (alias_parent / evidence_path.name).exists()
    assert not tuple(alias_parent.glob(f".{evidence_path.name}.*"))


def test_database_cache_clear_rejects_real_descendant_bind_mount_before_deletion(
    tmp_path: Path,
) -> None:
    is_child = os.environ.get(_BIND_NAMESPACE_CHILD) == "1"
    case_root = Path(os.environ[_BIND_NAMESPACE_ROOT]) if is_child else tmp_path / "namespace-case"
    if not is_child:
        if not _user_namespaces_available():
            pytest.skip("unprivileged user/mount namespaces are unavailable on this host")
        child_basetemp = tmp_path / "child-pytest"
        environment = {
            **os.environ,
            _BIND_NAMESPACE_CHILD: "1",
            _BIND_NAMESPACE_ROOT: str(case_root),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            child = subprocess.run(
                (
                    "unshare",
                    "--user",
                    "--map-root-user",
                    "--mount",
                    "--fork",
                    "bash",
                    "-c",
                    'mount --make-rprivate / && exec "$@"',
                    "database-cache-bind-test",
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    f"--basetemp={child_basetemp}",
                    f"{Path(__file__).resolve()}::{test_database_cache_clear_rejects_real_descendant_bind_mount_before_deletion.__name__}",
                ),
                check=False,
                capture_output=True,
                env=environment,
                text=True,
            )
        finally:
            if child_basetemp.exists():
                shutil.rmtree(child_basetemp)
        assert not child_basetemp.exists()
        assert child.returncode == 0, f"{child.stdout}\n{child.stderr}"
        expected_entry = case_root / "selected" / "acceptance" / "cache" / "users" / "root" / "replicas" / _FINAL
        expected_identity = json.loads((case_root / "expected-entry-identity.json").read_text())
        entry_info = expected_entry.stat()
        assert (entry_info.st_dev, entry_info.st_ino) == tuple(expected_identity)
        assert (expected_entry / "payload").read_bytes() == b"keep-on-real-bind-alias"
        evidence_name = "physical-alias.json"
        external_parent = case_root / "external-evidence"
        alias_parent = expected_entry.parents[1] / "nested-evidence"
        assert not (external_parent / evidence_name).exists()
        assert not (alias_parent / evidence_name).exists()
        assert not tuple(external_parent.glob(f".{evidence_name}.*"))
        assert not tuple(alias_parent.glob(f".{evidence_name}.*"))
        (case_root / "expected-entry-identity.json").unlink()
        return

    cache_root, user_root = _cache(case_root / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"keep-on-real-bind-alias")
    entry_info = entry.stat()
    (case_root / "expected-entry-identity.json").write_text(json.dumps((entry_info.st_dev, entry_info.st_ino)))
    alias_parent = user_root / "nested-evidence"
    alias_parent.mkdir(mode=0o700)
    external_parent = case_root / "external-evidence"
    external_parent.mkdir(mode=0o700)
    evidence_path = external_parent / "physical-alias.json"
    subprocess.run(("mount", "--bind", str(alias_parent), str(external_parent)), check=True)
    try:
        result = _invoke(_config(case_root, selected=cache_root), evidence_path)

        assert result.exit_code != 0
        assert entry.exists()
        assert (entry.stat().st_dev, entry.stat().st_ino) == (entry_info.st_dev, entry_info.st_ino)
        assert (entry / "payload").read_bytes() == b"keep-on-real-bind-alias"
        assert not evidence_path.exists()
        assert not (alias_parent / evidence_path.name).exists()
        assert not tuple(alias_parent.glob(f".{evidence_path.name}.*"))
    finally:
        subprocess.run(("umount", str(external_parent)), check=True)


@pytest.mark.parametrize(
    "refusal_stage",
    ["profile-user-mismatch", "scope-invalid", "sibling-alias", "replicas-invalid"],
)
def test_database_cache_clear_never_publishes_before_physical_evidence_authority_is_proven(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    refusal_stage: str,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return

    cache_root, user_root = _cache(case_root / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"preserve-before-publication-proof")
    entry_info = entry.stat()
    evidence_parent = case_root / "external-evidence"
    evidence_parent.mkdir(mode=0o700)
    evidence_path = evidence_parent / f"{refusal_stage}.json"
    production_root: Path | None = None
    config_path = _config(case_root, selected=cache_root)
    if refusal_stage == "profile-user-mismatch":
        raw = yaml.safe_load(config_path.read_text())
        raw["clusters"]["selected"]["database_cache_unix_user"] = "somebody-else"
        config_path.write_text(yaml.safe_dump(raw, sort_keys=True))
    elif refusal_stage == "scope-invalid":
        user_root.chmod(0o755)
    elif refusal_stage == "sibling-alias":
        production_root = case_root / "production" / "database" / "cache"
        production_root.parent.mkdir(parents=True)
        production_root.symlink_to(cache_root, target_is_directory=True)
        config_path = _config(case_root, selected=cache_root, production=production_root)
    else:
        (user_root / "replicas").chmod(0o755)

    subprocess.run(("mount", "--bind", str(user_root), str(evidence_parent)), check=True)
    try:
        result = _invoke(config_path, evidence_path)
        entry_after = entry.stat()
        entry_payload = (entry / "payload").read_bytes()
        alias_evidence_exists = evidence_path.exists()
        backing_evidence_exists = (user_root / evidence_path.name).exists()
        alias_temps = tuple(evidence_parent.glob(f".{evidence_path.name}.*"))
        backing_temps = tuple(user_root.glob(f".{evidence_path.name}.*"))
    finally:
        subprocess.run(("umount", str(evidence_parent)), check=True)

    assert result.exit_code != 0
    assert (entry_after.st_dev, entry_after.st_ino) == (entry_info.st_dev, entry_info.st_ino)
    assert entry_payload == b"preserve-before-publication-proof"
    assert not alias_evidence_exists
    assert not backing_evidence_exists
    assert not alias_temps
    assert not backing_temps
    assert not tuple(evidence_parent.iterdir())


@pytest.mark.parametrize(
    "fault",
    ["statx-unavailable", "mountinfo-unavailable", "mountinfo-malformed", "mountinfo-ambiguous"],
)
def test_database_cache_clear_fails_closed_when_mount_authority_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"keep-without-mount-authority")
    evidence_path = tmp_path / "evidence" / "mount-authority.json"
    if fault == "statx-unavailable":
        real_sizeof = ctypes.sizeof

        def incompatible_statx_size(value: object) -> int:
            if value is mount_authority._Statx:
                return 0
            return real_sizeof(value)

        monkeypatch.setattr(ctypes, "sizeof", incompatible_statx_size)
    else:
        mountinfo_path = tmp_path / "mountinfo"
        if fault == "mountinfo-malformed":
            mountinfo_path.write_text("malformed mount authority\n")
        elif fault == "mountinfo-ambiguous":
            # Duplicate the mountinfo entry that COVERS the target path: the
            # authority is target-scoped (it only flags duplicate mount IDs
            # among the target's retained ancestry), so duplicating an
            # arbitrary line - e.g. the first/root entry - only trips the
            # ambiguity defense when the target shares that mount, which
            # depends on the host's mount topology. The covering entry's
            # mount ID is the target's own statx mount ID, so its duplicate
            # is guaranteed to collide in the required set on any host.
            resolved_target = tmp_path.resolve()
            lines = Path("/proc/self/mountinfo").read_text().splitlines()
            covering_line: str | None = None
            covering_length = -1
            for candidate in lines:
                fields = candidate.split()
                if len(fields) < 5 or fields.count("-") != 1:
                    continue
                mount_point = mount_authority._decode_mountinfo_path(fields[4]).as_posix().rstrip("/")
                covers = mount_point == "" or str(resolved_target).startswith(mount_point + "/")
                if covers and len(mount_point) > covering_length:
                    covering_line = candidate
                    covering_length = len(mount_point)
            assert covering_line is not None, "no mountinfo entry covers the target path"
            mountinfo_path.write_text("\n".join((*lines, covering_line)) + "\n")
    result = _invoke_with_paths(
        _config(tmp_path, selected=cache_root),
        evidence_path,
        DatabaseCacheMaintenancePaths(
            mountinfo=mountinfo_path if fault != "statx-unavailable" else Path("/proc/self/mountinfo")
        ),
    )

    assert result.exit_code != 0
    assert "maintenance evidence mount authority is unavailable" in result.output
    assert entry.exists()
    assert not evidence_path.exists()
    assert not tuple(evidence_path.parent.glob(f".{evidence_path.name}.*"))


@pytest.mark.parametrize("authority_change", ["target-mountinfo-record", "visible-cache-root-rebind"])
def test_database_cache_clear_rechecks_complete_mount_authority_before_every_removal(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    authority_change: str,
) -> None:
    case_root = _private_mount_namespace_case(tmp_path, request)
    if case_root is None:
        return

    backing_cache_root, backing_user_root = _cache(case_root / "backing" / "acceptance" / "cache")
    first = _entry(backing_user_root, _FINAL, payload=b"first-removal-before-target-change")
    second = _entry(backing_user_root, _OTHER_FINAL, payload=b"preserve-after-target-change")
    first_info = first.stat()
    second_info = second.stat()
    cache_root = case_root / "selected" / "acceptance" / "cache"
    cache_root.mkdir(parents=True, mode=0o700)
    subprocess.run(("mount", "--bind", str(backing_cache_root), str(cache_root)), check=True)
    mount_layers = 1
    mountinfo_path = case_root / "mountinfo"
    mountinfo_path.write_bytes(Path("/proc/self/mountinfo").read_bytes())
    real_remove = remove_owned_tree
    alternate_cache_root, _ = _cache(case_root / "alternate" / "acceptance" / "cache")
    evidence_path = case_root / "evidence" / "changed-target-authority.json"
    removals = 0
    target_user_descriptor = os.open(cache_root / "users" / _user(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        target_mount_id = mount_authority._statx_identity(target_user_descriptor).mount_id
    finally:
        os.close(target_user_descriptor)

    def remove_then_change_mount_authority(*args: object, **kwargs: object) -> object:
        nonlocal mount_layers, removals
        removed_authority = real_remove(*args, **kwargs)  # type: ignore[arg-type]
        removals += 1
        if removals == 1 and authority_change == "target-mountinfo-record":
            changed: list[str] = []
            for line in mountinfo_path.read_text().splitlines():
                mount_id, parent_id, remainder = line.split(" ", 2)
                if int(mount_id) == target_mount_id:
                    parent_id = str(int(parent_id) + 1_000_000)
                changed.append(f"{mount_id} {parent_id} {remainder}")
            mountinfo_path.write_text("\n".join(changed) + "\n")
        elif removals == 1:
            subprocess.run(("mount", "--bind", str(alternate_cache_root), str(cache_root)), check=True)
            mount_layers += 1
        return removed_authority

    try:
        result = _invoke_with_owned_tree_remover(
            _config(case_root, selected=cache_root),
            evidence_path,
            OwnedTreeRemover(remove_then_change_mount_authority),
            paths=DatabaseCacheMaintenancePaths(mountinfo=mountinfo_path),
        )
    finally:
        while mount_layers:
            subprocess.run(("umount", str(cache_root)), check=True)
            mount_layers -= 1

    assert removals == 1
    assert not first.exists()
    assert second.exists()
    assert (second.stat().st_dev, second.stat().st_ino) == (second_info.st_dev, second_info.st_ino)
    assert (second / "payload").read_bytes() == b"preserve-after-target-change"
    assert result.exit_code != 0
    assert "authority" in result.output
    assert evidence_path.exists()
    assert stat_mode(evidence_path) == 0o444
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "failed"
    assert evidence.removed_count == 1
    assert len(evidence.removed_entry_samples) == 1
    removed = evidence.removed_entry_samples[0]
    assert (removed.kind, removed.source_manifest_sha256, removed.basename) == ("replica", _FINAL, _FINAL)
    assert (removed.device, removed.inode) == (first_info.st_dev, first_info.st_ino)
    expected_digest_payload = {
        "removed_entries": [
            {
                "basename": _FINAL,
                "device": first_info.st_dev,
                "inode": first_info.st_ino,
                "kind": "replica",
                "source_manifest_sha256": _FINAL,
            }
        ]
    }
    expected_digest = hashlib.sha256(
        (json.dumps(expected_digest_payload, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    assert evidence.removed_entries_sha256 == expected_digest
    assert not tuple(evidence_path.parent.glob(f".{evidence_path.name}.*"))


@pytest.mark.parametrize("fault", ["post-write-parent-fsync", "strict-reload"])
def test_post_mutation_evidence_fault_publishes_exact_failed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"removed-before-final-evidence-check")
    entry_info = entry.stat()
    evidence_path = tmp_path / "evidence" / f"{fault}.json"
    injected = False
    if fault == "post-write-parent-fsync":
        real_fsync = maintenance_runtime.os.fsync

        def fail_after_publication(descriptor: int) -> None:
            nonlocal injected
            if (
                not injected
                and evidence_path.exists()
                and stat_mode(evidence_path) == 0o444
                and evidence_path.stat().st_size > 0
                and Path(os.readlink(f"/proc/self/fd/{descriptor}")) == evidence_path.parent
            ):
                injected = True
                raise OSError("post-write parent fsync")
            real_fsync(descriptor)

        monkeypatch.setattr(maintenance_runtime.os, "fsync", fail_after_publication)
    else:
        durability = _filesystem_evidence_durability()
        real_loader = durability.load

        def fail_strict_reload(
            path: Path,
            *,
            dir_fd: int | None = None,
        ):  # type: ignore[no-untyped-def]
            nonlocal injected
            if not injected and path.name == evidence_path.name and dir_fd is not None:
                injected = True
                raise ValueError("injected strict reload failure")
            return real_loader(path, dir_fd=dir_fd)

        evidence_store = _evidence_store_with_durability(replace(durability, load=fail_strict_reload))

    result = (
        _invoke(_config(tmp_path, selected=cache_root), evidence_path)
        if fault == "post-write-parent-fsync"
        else _invoke_with_evidence_store(
            _config(tmp_path, selected=cache_root),
            evidence_path,
            evidence_store,
        )
    )

    assert injected
    assert result.exit_code != 0
    assert not entry.exists()
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "failed"
    assert evidence.removed_count == 1
    assert len(evidence.removed_entry_samples) == 1
    removed = evidence.removed_entry_samples[0]
    assert (removed.kind, removed.source_manifest_sha256, removed.basename) == ("replica", _FINAL, _FINAL)
    assert (removed.device, removed.inode) == (entry_info.st_dev, entry_info.st_ino)


@pytest.mark.parametrize("fault", ["second-write", "nonempty-file-fsync"])
def test_post_mutation_final_file_fault_publishes_exact_failed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"removed-before-transient-finalization-fault")
    entry_info = entry.stat()
    evidence_path = tmp_path / "evidence" / f"{fault}.json"
    injected = False
    if fault == "second-write":
        durability = _filesystem_evidence_durability()
        real_write_all = durability.write_all
        calls = 0

        def fail_second_write(descriptor: int, content: bytes) -> None:
            nonlocal calls, injected
            calls += 1
            if calls == 1:
                injected = True
                raise OSError("transient final evidence write fault")
            real_write_all(descriptor, content)

        evidence_store = _evidence_store_with_durability(replace(durability, write_all=fail_second_write))
    else:
        real_fsync = maintenance_runtime.os.fsync

        def fail_nonempty_final_file_fsync(descriptor: int) -> None:
            nonlocal injected
            if (
                not injected
                and Path(os.readlink(f"/proc/self/fd/{descriptor}")) == evidence_path
                and os.fstat(descriptor).st_size > 0
            ):
                injected = True
                raise OSError("transient nonempty final evidence fsync fault")
            real_fsync(descriptor)

        monkeypatch.setattr(maintenance_runtime.os, "fsync", fail_nonempty_final_file_fsync)

    result = (
        _invoke_with_evidence_store(
            _config(tmp_path, selected=cache_root),
            evidence_path,
            evidence_store,
        )
        if fault == "second-write"
        else _invoke(_config(tmp_path, selected=cache_root), evidence_path)
    )

    assert injected
    assert result.exit_code != 0
    assert not entry.exists()
    assert evidence_path.exists()
    assert stat_mode(evidence_path) == 0o444
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "failed"
    assert evidence.removed_count == 1
    assert len(evidence.removed_entry_samples) == 1
    removed = evidence.removed_entry_samples[0]
    assert (removed.kind, removed.source_manifest_sha256, removed.basename) == ("replica", _FINAL, _FINAL)
    assert (removed.device, removed.inode) == (entry_info.st_dev, entry_info.st_ino)
    raw = evidence_path.read_bytes()
    assert raw == (json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n").encode()


@pytest.mark.parametrize(
    "forgery",
    [
        "missing-locks",
        "missing-descriptor",
        "wrong-digest",
        "reordered-samples",
        "wrong-lock-count",
        "wrong-lock-digest",
        "reordered-lock-samples",
        "descriptor-uid-drift",
        "descriptor-device-drift",
        "removed-device-drift",
    ],
)
def test_strict_loader_rejects_internally_inconsistent_terminal_evidence(
    tmp_path: Path,
    forgery: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    _entry(user_root, _FINAL, payload=b"first")
    _entry(user_root, _POPULATION, payload=b"second")
    _entry(user_root, _OTHER_FINAL, payload=b"third")
    evidence_path = tmp_path / "evidence" / "valid.json"
    assert _invoke(_config(tmp_path, selected=cache_root), evidence_path).exit_code == 0
    payload = json.loads(evidence_path.read_bytes())
    body = payload["database_cache_maintenance"]
    if forgery == "missing-locks":
        body["identity_locks"] = {
            "acquired_count": 0,
            "contended_count": 0,
            "observations_sha256": "ba33de21739378ff7e909c7f1507046ff6c187e67d4d4f6ed25e44d944a0b922",
            "omitted_count": 0,
            "samples": [],
            "total_count": 0,
        }
    elif forgery == "missing-descriptor":
        body["descriptors"] = body["descriptors"][:-1]
    elif forgery == "wrong-digest":
        body["removed"]["entries_sha256"] = "0" * 64
    elif forgery == "reordered-samples":
        body["removed"]["samples"] = list(reversed(body["removed"]["samples"]))
    elif forgery == "wrong-lock-count":
        body["identity_locks"]["total_count"] += 1
    elif forgery == "wrong-lock-digest":
        body["identity_locks"]["observations_sha256"] = "0" * 64
    elif forgery == "reordered-lock-samples":
        body["identity_locks"]["samples"] = list(reversed(body["identity_locks"]["samples"]))
    elif forgery == "descriptor-uid-drift":
        body["descriptors"][0]["uid"] += 1
    elif forgery == "descriptor-device-drift":
        body["descriptors"][1]["device"] += 1
    else:
        body["removed"]["samples"][0]["device"] += 1
        body["removed"]["count"] += 1
        body["removed"]["omitted_count"] += 1
    forged = tmp_path / "evidence" / f"forged-{forgery}.json"
    forged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    forged.chmod(0o444)

    with pytest.raises(ValueError):
        load_database_cache_maintenance_evidence(forged)


@pytest.mark.parametrize("forgery", ["incomplete-sequence", "wrong-order"])
def test_strict_loader_rejects_impossible_unlocked_refused_descriptor_roles(
    tmp_path: Path,
    forgery: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    _entry(user_root, "unknown", payload=b"keep")
    evidence_path = tmp_path / "evidence" / "valid-refusal.json"
    assert _invoke(_config(tmp_path, selected=cache_root), evidence_path).exit_code != 0
    assert load_database_cache_maintenance_evidence(evidence_path).terminal_result == "refused"
    payload = json.loads(evidence_path.read_bytes())
    descriptors = payload["database_cache_maintenance"]["descriptors"]
    if forgery == "incomplete-sequence":
        payload["database_cache_maintenance"]["descriptors"] = descriptors[1:2]
    else:
        payload["database_cache_maintenance"]["descriptors"] = [
            descriptors[1],
            descriptors[0],
            *descriptors[2:],
        ]
    forged = tmp_path / "evidence" / f"forged-refusal-{forgery}.json"
    forged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    forged.chmod(0o444)

    with pytest.raises(ValueError):
        load_database_cache_maintenance_evidence(forged)


def test_database_cache_clear_stable_rescan_change_deletes_nothing(
    tmp_path: Path,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    original = _entry(user_root, _FINAL, payload=b"keep")
    context = multiprocessing.get_context("fork")
    changed = context.Event()

    def mutate_after_cache_lock() -> None:
        descriptor = os.open(user_root / ".locks" / "cache.lock", os.O_RDWR | os.O_NOFOLLOW)
        try:
            deadline = time.monotonic() + 5
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    _entry(user_root, _POPULATION.replace(_FINAL, _OTHER_FINAL), payload=b"new")
                    changed.set()
                    return
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                if time.monotonic() >= deadline:
                    os._exit(22)
                time.sleep(0.001)
        finally:
            os.close(descriptor)

    contender = context.Process(target=mutate_after_cache_lock)
    contender.start()
    evidence_path = tmp_path / "evidence" / "changed.json"

    try:
        result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)
        contender.join(timeout=5)
    finally:
        if contender.is_alive():
            contender.terminate()
        contender.join(timeout=5)

    assert contender.exitcode == 0
    assert changed.is_set()
    assert result.exit_code != 0
    assert original.exists()
    assert load_database_cache_maintenance_evidence(evidence_path).terminal_result == "refused"


@pytest.mark.parametrize("interruption", ["keyboard-interrupt", "system-exit"])
def test_database_cache_clear_publishes_failed_evidence_before_post_removal_interruption(
    tmp_path: Path,
    interruption: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"remove-before-interruption")
    original = entry.stat()
    config_path = _config(tmp_path, selected=cache_root)
    profile = maintenance_runtime.load_database_acceptance_cache_profile(config_path, "selected")
    evidence_path = tmp_path / "evidence" / f"post-removal-{interruption}.json"
    real_remove = remove_owned_tree

    def remove_then_interrupt(*args: object, **kwargs: object) -> None:
        real_remove(*args, **kwargs)  # type: ignore[arg-type]
        if interruption == "keyboard-interrupt":
            raise KeyboardInterrupt
        raise SystemExit(37)

    expected = KeyboardInterrupt if interruption == "keyboard-interrupt" else SystemExit

    with pytest.raises(expected) as caught:
        maintenance_runtime._clear_database_acceptance_cache(
            profile,
            evidence_path=evidence_path,
            owned_tree_remover=OwnedTreeRemover(remove_then_interrupt),
            paths=DatabaseCacheMaintenancePaths(mountinfo=Path("/proc/self/mountinfo")),
            evidence_store=_filesystem_evidence_store(),
        )

    if isinstance(caught.value, SystemExit):
        assert caught.value.code == 37
    assert not entry.exists()
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "failed"
    assert evidence.removed_count == 1
    assert len(evidence.removed_entry_samples) == 1
    assert evidence.removed_entry_samples[0].basename == _FINAL
    assert (evidence.removed_entry_samples[0].device, evidence.removed_entry_samples[0].inode) == (
        original.st_dev,
        original.st_ino,
    )
    assert stat_mode(evidence_path) == 0o444
    assert _exclusive_locks_are_reacquirable(
        (user_root / ".locks" / "cache.lock", user_root / ".locks" / f"{_FINAL}.lock")
    )


@pytest.mark.parametrize("fault", ["keyboard-interrupt", "oserror"])
def test_database_cache_clear_post_claim_fault_does_not_report_surviving_inode_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"survive-under-accepted-claim")
    original = entry.stat()
    original_identity = (original.st_dev, original.st_ino)
    config_path = _config(tmp_path, selected=cache_root)
    profile = maintenance_runtime.load_database_acceptance_cache_profile(config_path, "selected")
    evidence_path = tmp_path / "evidence" / f"post-claim-{fault}.json"
    real_fchmod = os.fchmod
    injected = False

    def interrupt_first_root_fchmod(descriptor: int, mode: int) -> None:
        nonlocal injected
        opened = os.fstat(descriptor)
        if not injected and mode == 0o700 and (opened.st_dev, opened.st_ino) == original_identity:
            injected = True
            if fault == "keyboard-interrupt":
                raise KeyboardInterrupt
            raise OSError("injected first post-claim fault")
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(os, "fchmod", interrupt_first_root_fchmod)
    if fault == "keyboard-interrupt":
        with pytest.raises(KeyboardInterrupt):
            maintenance_runtime.clear_database_acceptance_cache(profile, evidence_path=evidence_path)
    else:
        with pytest.raises(
            maintenance_runtime.DatabaseCacheMaintenanceError,
            match="database cache entry removal failed",
        ):
            maintenance_runtime.clear_database_acceptance_cache(profile, evidence_path=evidence_path)

    assert injected
    assert not entry.exists()
    surviving_claims = tuple(
        candidate
        for candidate in (user_root / "replicas").iterdir()
        if (candidate.stat().st_dev, candidate.stat().st_ino) == original_identity
    )
    assert len(surviving_claims) == 1
    claim = surviving_claims[0]
    population_match = maintenance_scope._POPULATION.fullmatch(claim.name)
    assert population_match is not None
    assert population_match.group(1) == _FINAL
    assert (claim / "payload").read_bytes() == b"survive-under-accepted-claim"
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "failed"
    assert evidence.removed_count == 0
    assert evidence.removed_entry_samples == ()
    assert stat_mode(evidence_path) == 0o444
    assert _exclusive_locks_are_reacquirable(
        (user_root / ".locks" / "cache.lock", user_root / ".locks" / f"{_FINAL}.lock")
    )


@pytest.mark.parametrize(
    "exception_type",
    [
        pytest.param(KeyboardInterrupt, id="keyboard-interrupt"),
        pytest.param(SystemExit, id="system-exit"),
    ],
)
def test_database_cache_clear_redacts_post_removal_interruption_value(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    exception_type: type[BaseException],
) -> None:
    secret = "private-payload-member-ultra-secret.ffdata"
    interruption = exception_type(secret)
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"remove-before-secret-interruption")
    original = entry.stat()
    config_path = _config(tmp_path, selected=cache_root)
    profile = maintenance_runtime.load_database_acceptance_cache_profile(config_path, "selected")
    evidence_path = tmp_path / "evidence" / f"post-removal-{exception_type.__name__}.json"
    real_remove = remove_owned_tree

    def remove_then_interrupt(*args: object, **kwargs: object) -> None:
        real_remove(*args, **kwargs)  # type: ignore[arg-type]
        raise interruption

    with pytest.raises(exception_type) as caught:
        maintenance_runtime._clear_database_acceptance_cache(
            profile,
            evidence_path=evidence_path,
            owned_tree_remover=OwnedTreeRemover(remove_then_interrupt),
            paths=DatabaseCacheMaintenancePaths(mountinfo=Path("/proc/self/mountinfo")),
            evidence_store=_filesystem_evidence_store(),
        )

    assert type(caught.value) is exception_type
    assert caught.value is interruption
    assert caught.value.args == (secret,)
    if isinstance(caught.value, SystemExit):
        assert caught.value.code == secret
    assert not entry.exists()
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "failed"
    assert evidence.removed_count == 1
    assert len(evidence.removed_entry_samples) == 1
    assert evidence.removed_entry_samples[0].basename == _FINAL
    assert (evidence.removed_entry_samples[0].device, evidence.removed_entry_samples[0].inode) == (
        original.st_dev,
        original.st_ino,
    )
    assert stat_mode(evidence_path) == 0o444
    assert _exclusive_locks_are_reacquirable(
        (user_root / ".locks" / "cache.lock", user_root / ".locks" / f"{_FINAL}.lock")
    )
    captured = capsys.readouterr()
    assert (
        evidence.diagnostic,
        secret.encode() in evidence_path.read_bytes(),
        secret in captured.out,
        secret in captured.err,
    ) == (exception_type.__name__, False, False, False)


@pytest.mark.parametrize(
    "exception_type",
    [
        pytest.param(KeyboardInterrupt, id="keyboard-interrupt"),
        pytest.param(SystemExit, id="system-exit"),
    ],
)
def test_database_cache_clear_redacts_post_claim_interruption_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exception_type: type[BaseException],
) -> None:
    secret = "private-payload-member-ultra-secret.ffdata"
    interruption = exception_type(secret)
    payload = b"survive-under-accepted-secret-claim"
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=payload)
    original = entry.stat()
    original_identity = (original.st_dev, original.st_ino)
    config_path = _config(tmp_path, selected=cache_root)
    profile = maintenance_runtime.load_database_acceptance_cache_profile(config_path, "selected")
    evidence_path = tmp_path / "evidence" / f"post-claim-{exception_type.__name__}.json"
    real_fchmod = os.fchmod
    injected = False

    def interrupt_first_root_fchmod(descriptor: int, mode: int) -> None:
        nonlocal injected
        opened = os.fstat(descriptor)
        if not injected and mode == 0o700 and (opened.st_dev, opened.st_ino) == original_identity:
            injected = True
            raise interruption
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(os, "fchmod", interrupt_first_root_fchmod)

    with pytest.raises(exception_type) as caught:
        maintenance_runtime.clear_database_acceptance_cache(profile, evidence_path=evidence_path)

    assert type(caught.value) is exception_type
    assert caught.value is interruption
    assert caught.value.args == (secret,)
    if isinstance(caught.value, SystemExit):
        assert caught.value.code == secret
    assert injected
    assert not entry.exists()
    surviving_claims = tuple(
        candidate
        for candidate in (user_root / "replicas").iterdir()
        if (candidate.stat().st_dev, candidate.stat().st_ino) == original_identity
    )
    assert len(surviving_claims) == 1
    claim = surviving_claims[0]
    population_match = maintenance_scope._POPULATION.fullmatch(claim.name)
    assert population_match is not None
    assert population_match.group(1) == _FINAL
    assert claim.stat().st_dev == original.st_dev
    assert claim.stat().st_ino == original.st_ino
    assert (claim / "payload").read_bytes() == payload
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "failed"
    assert evidence.removed_count == 0
    assert evidence.removed_entry_samples == ()
    assert stat_mode(evidence_path) == 0o444
    assert _exclusive_locks_are_reacquirable(
        (user_root / ".locks" / "cache.lock", user_root / ".locks" / f"{_FINAL}.lock")
    )
    captured = capsys.readouterr()
    assert (
        evidence.diagnostic,
        secret.encode() in evidence_path.read_bytes(),
        secret in captured.out,
        secret in captured.err,
    ) == (exception_type.__name__, False, False, False)


@pytest.mark.parametrize(
    "exception_type",
    [
        pytest.param(KeyboardInterrupt, id="keyboard-interrupt"),
        pytest.param(SystemExit, id="system-exit"),
    ],
)
@pytest.mark.parametrize(
    "finalization_point",
    ["write", "nonempty-fsync", "parent-fsync", "strict-reload", "final-verify"],
)
def test_database_cache_clear_recovers_terminal_evidence_after_finalization_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exception_type: type[BaseException],
    finalization_point: str,
) -> None:
    secret = "private-finalization-payload-member-ultra-secret.ffdata"
    interruption = exception_type(secret)
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"remove-before-finalization-interruption")
    original = entry.stat()
    config_path = _config(tmp_path, selected=cache_root)
    profile = maintenance_runtime.load_database_acceptance_cache_profile(config_path, "selected")
    evidence_path = tmp_path / "evidence" / f"finalization-{finalization_point}-{exception_type.__name__}.json"
    durability = _filesystem_evidence_durability()
    real_write_all = durability.write_all
    real_fsync = durability.sync
    real_load = durability.load
    real_verify = durability.verify
    cleared_payload_written = False
    cleared_file_fsync_complete = False
    cleared_write_seen = False
    cleared_reload_complete = False
    injected = False

    def interrupt_finalization_write(descriptor: int, content: bytes) -> None:
        nonlocal cleared_payload_written, cleared_write_seen, injected
        is_cleared_evidence = b'"terminal_result": "cleared"' in content
        if finalization_point == "write" and is_cleared_evidence and not injected:
            injected = True
            raise interruption
        real_write_all(descriptor, content)
        if is_cleared_evidence:
            cleared_payload_written = True
            cleared_write_seen = True

    def interrupt_nonempty_evidence_fsync(descriptor: int) -> None:
        nonlocal cleared_file_fsync_complete, cleared_payload_written, injected
        if cleared_payload_written:
            cleared_payload_written = False
            if finalization_point == "nonempty-fsync" and not injected:
                injected = True
                assert os.fstat(descriptor).st_size > 0
                raise interruption
            real_fsync(descriptor)
            cleared_file_fsync_complete = True
            return
        if cleared_file_fsync_complete:
            cleared_file_fsync_complete = False
            if finalization_point == "parent-fsync" and not injected:
                injected = True
                raise interruption
        real_fsync(descriptor)

    def interrupt_strict_reload(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal cleared_reload_complete, cleared_write_seen, injected
        if finalization_point == "strict-reload" and cleared_write_seen and not injected:
            cleared_write_seen = False
            injected = True
            raise interruption
        loaded = real_load(*args, **kwargs)
        if finalization_point == "final-verify" and cleared_write_seen:
            cleared_write_seen = False
            cleared_reload_complete = True
        return loaded

    def interrupt_final_verify(*args: object, **kwargs: object) -> None:
        nonlocal cleared_reload_complete, injected
        if cleared_reload_complete and not injected:
            cleared_reload_complete = False
            injected = True
            raise interruption
        real_verify(*args, **kwargs)

    evidence_store = _evidence_store_with_durability(
        replace(
            durability,
            write_all=interrupt_finalization_write,
            sync=interrupt_nonempty_evidence_fsync,
            load=interrupt_strict_reload,
            verify=interrupt_final_verify,
        )
    )

    with pytest.raises(exception_type) as caught:
        maintenance_runtime._clear_database_acceptance_cache(
            profile,
            evidence_path=evidence_path,
            owned_tree_remover=OwnedTreeRemover(remove_owned_tree),
            paths=DatabaseCacheMaintenancePaths(mountinfo=Path("/proc/self/mountinfo")),
            evidence_store=evidence_store,
        )

    assert injected
    assert type(caught.value) is exception_type
    assert caught.value is interruption
    assert caught.value.args == (secret,)
    if isinstance(caught.value, SystemExit):
        assert caught.value.code == secret
    assert not entry.exists()
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "failed"
    assert evidence.removed_count == 1
    assert len(evidence.removed_entry_samples) == 1
    assert evidence.removed_entry_samples[0].basename == _FINAL
    assert (evidence.removed_entry_samples[0].device, evidence.removed_entry_samples[0].inode) == (
        original.st_dev,
        original.st_ino,
    )
    assert stat_mode(evidence_path) == 0o444
    assert _exclusive_locks_are_reacquirable(
        (user_root / ".locks" / "cache.lock", user_root / ".locks" / f"{_FINAL}.lock")
    )
    captured = capsys.readouterr()
    assert (
        evidence.diagnostic,
        secret.encode() in evidence_path.read_bytes(),
        secret in captured.out,
        secret in captured.err,
    ) == (exception_type.__name__, False, False, False)


def test_database_cache_clear_records_exact_partial_removal_on_fault(
    tmp_path: Path,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    first = _entry(user_root, _FINAL, payload=b"first")
    second = _entry(user_root, _OTHER_FINAL, payload=b"second")
    real_remove = remove_owned_tree
    calls = 0

    def fail_second(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected removal fault")
        return real_remove(*args, **kwargs)  # type: ignore[arg-type]

    evidence_path = tmp_path / "evidence" / "partial.json"

    result = _invoke_with_owned_tree_remover(
        _config(tmp_path, selected=cache_root),
        evidence_path,
        OwnedTreeRemover(fail_second),
    )

    assert result.exit_code != 0
    assert not first.exists()
    assert second.exists()
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "failed"
    assert evidence.removed_count == 1
    assert evidence.removed_entry_samples[0].basename == _FINAL


def test_database_cache_clear_sanitizes_nested_removal_failure_evidence(
    tmp_path: Path,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"keep-after-nested-removal-fault")
    secret_member = "private-payload-member-ultra-secret.ffdata"

    def fail_with_nested_member(*_args: object, **_kwargs: object) -> None:
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed",
            f"nested cache child changed: {secret_member}",
        )

    evidence_path = tmp_path / "evidence" / "sanitized-failure.json"

    result = _invoke_with_owned_tree_remover(
        _config(tmp_path, selected=cache_root),
        evidence_path,
        OwnedTreeRemover(fail_with_nested_member),
    )

    assert result.exit_code != 0
    assert entry.exists()
    evidence = load_database_cache_maintenance_evidence(evidence_path)
    assert evidence.terminal_result == "failed"
    assert evidence.diagnostic == "database cache entry removal failed"
    assert secret_member not in result.output
    assert secret_member.encode() not in evidence_path.read_bytes()


@pytest.mark.parametrize(
    ("target", "fault"),
    [
        ("cache.lock", "device"),
        (f"{_FINAL}.lock", "device"),
        ("cache.lock", "owner"),
        (f"{_FINAL}.lock", "type"),
    ],
)
def test_database_cache_clear_refuses_invalid_lock_file_authority_without_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    fault: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    entry = _entry(user_root, _FINAL, payload=b"keep-on-invalid-lock-authority")
    real_stat = os.fstat

    def invalid_lock_stat(descriptor: int) -> os.stat_result:
        info = real_stat(descriptor)
        if Path(os.readlink(f"/proc/self/fd/{descriptor}")).name != target:
            return info
        values = list(info)
        if fault == "device":
            values[stat.ST_DEV] += 1
        elif fault == "owner":
            values[stat.ST_UID] += 1
        else:
            values[stat.ST_MODE] = stat.S_IFDIR | 0o600
        return os.stat_result(values)

    monkeypatch.setattr(os, "fstat", invalid_lock_stat)
    evidence_path = tmp_path / "evidence" / f"invalid-{target}-{fault}.json"

    result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)

    assert result.exit_code != 0
    assert entry.exists()
    assert load_database_cache_maintenance_evidence(evidence_path).terminal_result == "refused"


def test_database_cache_clear_refuses_cache_entry_basename_rebind_without_deletion(
    tmp_path: Path,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache")
    original = _entry(user_root, _FINAL, payload=b"original-survives-rebind")
    parked = user_root / "replicas" / _POPULATION
    real_remove = remove_owned_tree
    swapped = False

    def rebind_before_descriptor_open(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal swapped
        if not swapped:
            swapped = True
            original.rename(parked)
            _entry(user_root, _FINAL, payload=b"replacement-survives-rebind")
        return real_remove(*args, **kwargs)  # type: ignore[arg-type]

    evidence_path = tmp_path / "evidence" / "entry-rebind.json"

    result = _invoke_with_owned_tree_remover(
        _config(tmp_path, selected=cache_root),
        evidence_path,
        OwnedTreeRemover(rebind_before_descriptor_open),
    )

    assert result.exit_code != 0
    assert (parked / "payload").read_bytes() == b"original-survives-rebind"
    assert (original / "payload").read_bytes() == b"replacement-survives-rebind"
    assert load_database_cache_maintenance_evidence(evidence_path).terminal_result == "failed"


@pytest.mark.parametrize("authority", ["cache-symlink", "replicas-symlink", "wrong-user-mode"])
def test_database_cache_clear_refuses_unsafe_path_or_descriptor_authority(
    tmp_path: Path,
    authority: str,
) -> None:
    cache_root, user_root = _cache(tmp_path / "selected" / "acceptance" / "cache", replicas=False)
    if authority == "cache-symlink":
        real_root = tmp_path / "real" / "acceptance" / "cache"
        real_root.parent.mkdir(parents=True)
        cache_root.rename(real_root)
        cache_root.parent.mkdir(parents=True, exist_ok=True)
        cache_root.symlink_to(real_root, target_is_directory=True)
    elif authority == "replicas-symlink":
        external = tmp_path / "external" / "replicas"
        external.mkdir(parents=True)
        (user_root / "replicas").symlink_to(external, target_is_directory=True)
    else:
        user_root.chmod(0o755)
    evidence_path = tmp_path / "evidence" / f"{authority}.json"

    result = _invoke(_config(tmp_path, selected=cache_root), evidence_path)

    assert result.exit_code != 0
    assert not evidence_path.exists()


@functools.lru_cache(maxsize=1)
def _user_namespaces_available() -> bool:
    """Return whether unprivileged user/mount namespaces can be created here.

    The authority attack cases below re-enter pytest inside a disposable
    private user/mount namespace (``unshare --user --map-root-user --mount``).
    Locked-down kernels deny unprivileged user namespaces, so on those hosts
    the namespace-requiring cases skip rather than fail: the namespace-free
    authority coverage still runs everywhere.
    """
    try:
        child = subprocess.run(
            ("unshare", "--user", "--map-root-user", "--mount", "--fork", "true"),
            check=False,
            capture_output=True,
        )
    except OSError:
        return False
    return child.returncode == 0


def _private_mount_namespace_case(tmp_path: Path, request: pytest.FixtureRequest) -> Path | None:
    """Re-enter one exact test node in a disposable private user/mount namespace."""
    if os.environ.get(_AUTHORITY_NAMESPACE_CHILD) == "1":
        assert os.environ[_AUTHORITY_NAMESPACE_NODE] == request.node.name
        return Path(os.environ[_AUTHORITY_NAMESPACE_ROOT])

    if not _user_namespaces_available():
        pytest.skip("unprivileged user/mount namespaces are unavailable on this host")
    case_root = tmp_path / "authority-namespace-case"
    child_basetemp = tmp_path / "authority-child-pytest"
    environment = {
        **os.environ,
        _AUTHORITY_NAMESPACE_CHILD: "1",
        _AUTHORITY_NAMESPACE_ROOT: str(case_root),
        _AUTHORITY_NAMESPACE_NODE: request.node.name,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        child = subprocess.run(
            (
                "unshare",
                "--user",
                "--map-root-user",
                "--mount",
                "--fork",
                "bash",
                "-c",
                'mount --make-rprivate / && exec "$@"',
                "database-cache-authority-test",
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--basetemp={child_basetemp}",
                f"{Path(__file__).resolve()}::{request.node.name}",
            ),
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )
    finally:
        if child_basetemp.exists():
            shutil.rmtree(child_basetemp)
    assert not child_basetemp.exists()
    assert child.returncode == 0, f"{child.stdout}\n{child.stderr}"
    return None


def _invoke(config_path: Path, evidence_path: Path):  # type: ignore[no-untyped-def]
    return CliRunner().invoke(
        cli,
        [
            "preprocessing",
            "database-cache",
            "clear",
            "--config",
            str(config_path),
            "--profile",
            "selected",
            "--write-evidence",
            str(evidence_path),
        ],
    )


def _invoke_with_owned_tree_remover(
    config_path: Path,
    evidence_path: Path,
    owned_tree_remover: OwnedTreeRemover,
    *,
    paths: DatabaseCacheMaintenancePaths | None = None,
    evidence_store: DatabaseCacheEvidenceStore | None = None,
) -> SimpleNamespace:
    try:
        authority = maintenance_runtime.load_database_acceptance_cache_authority(config_path, "selected")
        evidence = maintenance_runtime._clear_database_acceptance_cache(
            authority.profile,
            evidence_path=evidence_path,
            configured_sibling_cache_roots=authority.configured_sibling_cache_roots,
            owned_tree_remover=owned_tree_remover,
            paths=paths or DatabaseCacheMaintenancePaths(mountinfo=Path("/proc/self/mountinfo")),
            evidence_store=evidence_store or _filesystem_evidence_store(),
        )
    except (OSError, TypeError, ValueError, maintenance_runtime.DatabaseCacheMaintenanceError) as exc:
        return SimpleNamespace(exit_code=1, output=f"Error: {exc}\n")
    return SimpleNamespace(exit_code=0, output=json.dumps(evidence.to_mapping(), indent=2, sort_keys=True))


def _invoke_with_paths(
    config_path: Path,
    evidence_path: Path,
    paths: DatabaseCacheMaintenancePaths,
) -> SimpleNamespace:
    return _invoke_with_owned_tree_remover(
        config_path,
        evidence_path,
        OwnedTreeRemover(remove_owned_tree),
        paths=paths,
    )


def _invoke_with_evidence_store(
    config_path: Path,
    evidence_path: Path,
    evidence_store: DatabaseCacheEvidenceStore,
) -> SimpleNamespace:
    return _invoke_with_owned_tree_remover(
        config_path,
        evidence_path,
        OwnedTreeRemover(remove_owned_tree),
        evidence_store=evidence_store,
    )


def _filesystem_evidence_store() -> DatabaseCacheEvidenceStore:
    return DatabaseCacheEvidenceStore(
        reserve=maintenance_evidence._reserve_evidence_destination,
        authorize=maintenance_authority._authorize_evidence_publication,
        verify=maintenance_evidence._verify_evidence_destination,
        finish=maintenance_evidence._finish_evidence_destination,
        close=maintenance_evidence._close_evidence_destination,
    )


def _filesystem_evidence_durability() -> maintenance_evidence.DatabaseCacheEvidenceDurability:
    return maintenance_evidence.DatabaseCacheEvidenceDurability(
        write_all=maintenance_evidence._write_all,
        sync=os.fsync,
        load=maintenance_evidence.load_database_cache_maintenance_evidence,
        verify=maintenance_evidence._verify_evidence_destination,
    )


def _evidence_store_with_durability(
    durability: maintenance_evidence.DatabaseCacheEvidenceDurability,
) -> DatabaseCacheEvidenceStore:
    store = _filesystem_evidence_store()

    def finish(destination: object, evidence: object) -> None:
        maintenance_evidence._finish_evidence_destination_with_durability(
            destination,  # type: ignore[arg-type]
            evidence,  # type: ignore[arg-type]
            durability=durability,
        )

    return replace(store, finish=finish)


def _cache(root: Path, *, locks: bool = True, replicas: bool = True) -> tuple[Path, Path]:
    root.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    users = root / "users"
    users.mkdir(mode=0o700)
    user_root = users / _user()
    user_root.mkdir(mode=0o700)
    if locks:
        lock_root = user_root / ".locks"
        lock_root.mkdir(mode=0o700)
        for name in ("cache.lock", f"{_FINAL}.lock", f"{_OTHER_FINAL}.lock"):
            path = lock_root / name
            path.touch(mode=0o600)
            path.chmod(0o600)
    if replicas:
        (user_root / "replicas").mkdir(mode=0o700)
    return root, user_root


def _entry(user_root: Path, name: str, *, payload: bytes) -> Path:
    entry = user_root / "replicas" / name
    entry.mkdir(mode=0o700)
    (entry / "payload").write_bytes(payload)
    if len(name) == 64 and all(character in "0123456789abcdef" for character in name):
        entry.chmod(0o555)
    return entry


def _lock_file(path: Path) -> Path:
    path.touch(mode=0o600)
    path.chmod(0o600)
    return path


def _exclusive_lock_is_contended(path: Path) -> bool:
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _exclusive_locks_are_reacquirable(paths: tuple[Path, ...]) -> bool:
    descriptors: list[int] = []
    try:
        for path in paths:
            descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            descriptors.append(descriptor)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        return True
    finally:
        for descriptor in reversed(descriptors):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _config(
    tmp_path: Path,
    *,
    selected: Path,
    sibling: Path | None = None,
    production: Path | None = None,
) -> Path:
    profiles: dict[str, dict[str, object]] = {"selected": _profile(selected, marker="acceptance")}
    if sibling is not None:
        profiles["sibling"] = _profile(sibling, marker="acceptance")
    if production is not None:
        profiles["production"] = _profile(production, marker=None)
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump({"clusters": profiles}, sort_keys=True))
    return path


def _profile(root: Path, *, marker: str | None) -> dict[str, object]:
    return {
        "database_cache_namespace": marker,
        "database_cache_root": str(root),
        "database_cache_unix_user": _user(),
        "database_cache_filesystem_type": _filesystem_type(root),
        "database_cache_reserve_bytes": 0,
        "database_lock_wait_seconds": 1,
    }


def _filesystem_type(path: Path) -> str:
    device = path.stat().st_dev
    selected: tuple[int, str] | None = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields = before.split()
        major, minor = (int(item) for item in fields[2].split(":"))
        if os.makedev(major, minor) != device:
            continue
        mountpoint = fields[4].replace("\\040", " ")
        try:
            path.relative_to(mountpoint)
        except ValueError:
            continue
        candidate = (len(Path(mountpoint).parts), after.split()[0])
        if selected is None or candidate[0] > selected[0]:
            selected = candidate
    assert selected is not None
    return selected[1]


def _user() -> str:
    return pwd.getpwuid(os.geteuid()).pw_name


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o7777


def test_namespace_cases_skip_when_user_namespaces_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The authority attack cases skip (not fail) on userns-denying hosts.

    Locked-down kernels deny unprivileged user/mount namespaces; the
    namespace-requiring cases must report a clean skip there.
    """

    class _Node:
        name = "test_database_cache_clear_refuses_same_device_inode_production_alias_boundary"

    class _DeniedResult:
        returncode = 1
        stdout = ""
        stderr = "unshare: write failed /proc/self/uid_map: Operation not permitted"

    def _denied_unshare(*args: object, **kwargs: object) -> object:
        return _DeniedResult()

    # Fault the inventoried external subprocess boundary (not the product
    # helper, per the mutation policy) and drive the real capability
    # preflight, cache-cleared so the recomputation observes the fault.
    monkeypatch.setattr(subprocess, "run", _denied_unshare)
    _user_namespaces_available.cache_clear()
    try:
        with pytest.raises(pytest.skip.Exception, match="unprivileged user/mount namespaces"):
            _private_mount_namespace_case(tmp_path, _Node())  # type: ignore[arg-type]
    finally:
        _user_namespaces_available.cache_clear()
