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

"""Runtime-side execution and verification of operator transfer plans.

This module owns the **execution and verification** layer for
operator-initiated data movement. It receives an
:class:`~bspp.orchestration.contract.operator_data_movement.OperatorTransferPlan`
and executes each transfer via s5cmd (dispatched by destination URI scheme),
then performs full post-transfer verification (download + sha256, no sampling).

All external boundaries (s5cmd subprocess, credential loading, download+hash)
are injectable for testing. The production path uses ``s5cmd`` via
:mod:`bspp.orchestration.runtime.data_movement.s3.transfer`.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from bspp.orchestration.contract.operator_data_movement import (
    OperatorTransferEvidence,
    OperatorTransferItem,
    OperatorTransferPlan,
    operator_transfer_evidence_id,
    resolve_execution_tool,
)
from bspp.orchestration.runtime.data_movement.common import (
    PlannedTransfer,
    TransferResult,
    require_tool,
)
from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer
from bspp.orchestration.runtime.data_movement.s3.client import S3Credentials, load_credentials_from_env

TransferCallable = Callable[..., TransferResult | PlannedTransfer]
VerifyCallable = Callable[..., tuple[int, str]]
LsCallable = Callable[..., TransferResult]


class OperatorTransferError(RuntimeError):
    """Fail-closed error surface for operator transfer operations."""


# ---------------------------------------------------------------------------
# s5cmd ls helpers (matching hq_publication.py contract)
# ---------------------------------------------------------------------------


def _is_s5cmd_no_object_found(stderr: str) -> bool:
    return "no object found" in stderr.lower()


def _build_object_ls_argv(
    destination: str,
    *,
    credentials: S3Credentials,
    s5cmd_path: str = "s5cmd",
) -> tuple[str, ...]:
    """Build a direct ``s5cmd ls <destination>`` argv (no ``/*`` glob)."""
    argv = [s5cmd_path, "--endpoint-url", credentials.endpoint_url]
    return tuple([*argv, "ls", destination])


def _s3_bucket(uri: str) -> str:
    """Return the bucket component of an ``s3://bucket/key`` URI."""
    return uri[len("s3://") :].split("/", 1)[0]


def _parse_object_ls_rows(output: str) -> tuple[tuple[str, int], ...]:
    """Parse ``s5cmd ls`` output into ``(name, size_bytes)`` rows.

    s5cmd prints different name shapes depending on endpoint and version: a
    full ``s3://`` URI, a bucket-relative key (exact-key ls), or a name
    relative to the listed prefix. On some S3-compatible endpoints (observed
    with s5cmd 2.3.0) the name is printed without the ``s3://`` prefix. The
    name is the remainder after the date, time, and size columns, so it is
    extracted with ``maxsplit=3`` to preserve any embedded whitespace.
    Directory rows (``DIR`` in the size column, with or without a timestamp)
    are skipped. Any line that cannot be parsed raises ``ValueError``;
    callers must treat that as an error, never as "absent".
    """
    rows: list[tuple[str, int]] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=3)
        if len(parts) == 2 and parts[0] == "DIR":
            # Directory row without a timestamp: "DIR  name/"
            continue
        if len(parts) != 4:
            msg = f"unparseable s5cmd ls line {line_number}: {line!r}"
            raise ValueError(msg) from None
        if parts[2] == "DIR":
            # Directory row with a timestamp: "DATE TIME DIR name/"
            continue
        try:
            size_bytes = int(parts[2])
        except ValueError:
            msg = f"unparseable s5cmd ls line {line_number}: {line!r}"
            raise ValueError(msg) from None
        rows.append((parts[3], size_bytes))
    return tuple(rows)


def _name_matches_destination(name: str, *, destination: str, bucket: str, prefix: str) -> bool:
    """Return whether a listed ``s5cmd ls`` name denotes *destination*.

    s5cmd prints different name shapes depending on endpoint and version:

    - a full ``s3://`` URI (compared directly);
    - a bucket-relative key (exact-key ls on some S3-compatible endpoints) —
      prefixed with the bucket;
    - a name relative to the listed prefix (legacy/prefix ls) — joined onto ``prefix``.

    Only an exact reconstruction is accepted; anything else is not a match.
    """
    if name == destination:
        return True
    if name.startswith("s3://"):
        return False
    candidates = (
        f"s3://{bucket}/{name.lstrip('/')}",
        f"{prefix.rstrip('/')}/{name.lstrip('/')}",
    )
    return destination in candidates


def _run_s5cmd_ls(destination: str, *, credentials: S3Credentials) -> TransferResult:
    """Run ``s5cmd ls <destination>`` and return a TransferResult."""
    s5cmd = require_tool("s5cmd", hint="Install s5cmd or use the container image.")
    argv = _build_object_ls_argv(destination, credentials=credentials, s5cmd_path=s5cmd)
    started = time.monotonic()
    env = dict(os.environ)
    env.update(credentials.as_env())
    completed = subprocess.run(list(argv), capture_output=True, text=True, env=env, check=False)
    elapsed = time.monotonic() - started
    return TransferResult(
        tool="s5cmd",
        argv=argv,
        returncode=completed.returncode,
        elapsed_s=elapsed,
        stdout_tail=completed.stdout,
        stderr_tail=completed.stderr[-4096:],
    )


def _remote_object_exists(
    destination: str,
    *,
    credentials: S3Credentials | None,
    ls_fn: LsCallable | None = None,
) -> tuple[bool, int | None]:
    """Check whether a remote object exists and return its size.

    s5cmd exit-code contract (matching ``hq_publication._run_s5cmd`` /
    ``hq_publication._is_s5cmd_no_object_found``):

    - Exit code 1 + ``"no object found"`` on stderr → ``(False, None)`` (normal absent case).
    - Exit code 0 → parse the listing and look for the exact destination URI.
    - Any other non-zero exit → fail-closed ``OperatorTransferError``.

    For exit code 0, a non-empty listing that does not contain the
    destination is an error, never "absent": the overwrite-refusal path must
    fail closed when the listing shape is not understood.
    """
    if ls_fn is not None:
        result = ls_fn(destination, credentials=credentials)
    else:
        creds = credentials or load_credentials_from_env()
        result = _run_s5cmd_ls(destination, credentials=creds)
    if result.returncode != 0 and _is_s5cmd_no_object_found(result.stderr_tail):
        return (False, None)
    if result.returncode != 0:
        raise OperatorTransferError(
            f"s5cmd ls {destination} failed with rc={result.returncode}: {result.stderr_tail.strip()}"
        )
    bucket = _s3_bucket(destination)
    # Parent directory of the destination: the prefix used to normalize the
    # legacy "basename under parent" listing shape.
    parent_prefix = destination.rsplit("/", 1)[0] if "/" in destination[5:] else destination
    try:
        rows = _parse_object_ls_rows(result.stdout_tail)
    except ValueError as exc:
        raise OperatorTransferError(f"s5cmd ls {destination} produced unparseable output: {exc}") from exc
    if len(rows) == 0:
        return (False, None)
    for name, size_bytes in rows:
        if _name_matches_destination(name, destination=destination, bucket=bucket, prefix=parent_prefix):
            return (True, size_bytes)
    listed = ", ".join(repr(name) for name, _ in rows)
    raise OperatorTransferError(
        f"s5cmd ls {destination} returned objects that do not match the destination "
        f"(listed: {listed}); refusing to treat as absent"
    )


# ---------------------------------------------------------------------------
# Local file hashing
# ---------------------------------------------------------------------------


def _hash_bundle(path: Path) -> str:
    """Stream the bundle's sha256 (8 MiB chunks)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Post-transfer verification
# ---------------------------------------------------------------------------


def _default_verify_fn(uri: str, temp_path: Path, *, credentials: S3Credentials | None = None) -> tuple[int, str]:
    """Download a remote object and stream sha256 (production path)."""
    result = s3_transfer.cp(uri, str(temp_path), credentials=credentials)
    if isinstance(result, PlannedTransfer):
        raise OperatorTransferError(f"verify download returned a dry-run plan for {uri}")
    if not result.ok:
        raise OperatorTransferError(
            f"verify download failed for {uri} with rc={result.returncode}: {result.stderr_tail.strip()}"
        )
    size = temp_path.stat().st_size
    sha = _hash_bundle(temp_path)
    return (size, sha)


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------


def _has_prior_evidence(evidence_dir: Path, source: str, destination: str) -> bool:
    """Check if a prior evidence file exists for the same source/destination pair."""
    if not evidence_dir.exists():
        return False
    for path in evidence_dir.iterdir():
        if not path.name.endswith(".json"):
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        item = data.get("item", {})
        if not isinstance(item, dict):
            continue
        if item.get("source") == source and item.get("destination") == destination:
            return True
    return False


def _write_evidence_files(evidence_dir: Path, records: tuple[OperatorTransferEvidence, ...]) -> None:
    """Stage all evidence files, then commit them; roll back on failure.

    The evidence set is written in two phases so a mid-set write failure does
    not leave a misleading partial evidence set: every record is first staged
    to a temp file, then all temp files are renamed to their final names. On
    any failure the already-committed files and remaining temp files are
    removed (best-effort) and the error propagates.
    """
    evidence_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for evidence in records:
            final_path = evidence_dir / f"{evidence.evidence_id}.json"
            tmp_path = evidence_dir / f".{evidence.evidence_id}-{uuid.uuid4().hex}.tmp"
            tmp_path.write_text(evidence.to_json() + "\n")
            staged.append((tmp_path, final_path))
        for tmp_path, final_path in staged:
            os.replace(tmp_path, final_path)
            committed.append(final_path)
    except OSError:
        for final_path in committed:
            final_path.unlink(missing_ok=True)
        for tmp_path, _ in staged:
            tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


def execute_operator_transfer(
    plan: OperatorTransferPlan,
    *,
    force: bool = False,
    evidence_dir: Path | None = None,
    transfer_fn_map: Mapping[str, TransferCallable] | None = None,
    verify_fn: VerifyCallable | None = None,
    ls_fn: LsCallable | None = None,
    credentials: S3Credentials | None = None,
    lock_root: Path | None = None,
    snapshot_root: Path | None = None,
) -> tuple[OperatorTransferEvidence, ...]:
    """Execute an :class:`OperatorTransferPlan` and return evidence records.

    Rejects dry-run plans immediately. For each item:

    1. Snapshots local sources to an immutable temp file and verifies the
       snapshot (size + sha256), or size-preflights remote s3:// sources, so the
       bytes that are hashed are exactly the bytes that are uploaded (no TOCTOU
       between pre-transfer re-hash and the transfer subprocess reopen).
    2. Acquires a destination-scoped advisory lock, serializing the
       existence-check + copy + verify critical section against concurrent
       executions targeting the same key on the same node.
    3. Checks destination existence (overwrite-refusal without ``--force``).
    4. Dispatches the transfer via the per-item resolved tool.
    5. Post-verifies (full download + sha256, no sampling).

    All evidence records are buffered in memory and written only after every
    item succeeds. Partial uploads of earlier items are not rolled back on
    failure; only evidence writing is atomic (staged then committed).

    The destination-scoped lock is a same-node advisory lock (``flock`` on a
    lock file keyed by the destination hash under ``lock_root``). Plan-referenced
    destinations are content-addressed, so concurrent identical transfers are
    idempotent; cross-node concurrency on a manual-mode explicit destination
    remains an operator-coordination concern.
    """
    if plan.dry_run:
        raise OperatorTransferError("cannot execute a dry-run plan; re-plan with dry_run=False")

    resolved_lock_root = (
        lock_root if lock_root is not None else Path(tempfile.gettempdir()) / "bspp-operator-transfer-locks"
    )
    # The snapshot + verification download together need roughly twice the
    # bundle size; honor SLURM_TMPDIR (node-local NVMe) when present and allow
    # the caller to select a larger staging location explicitly.
    resolved_snapshot_root = (
        snapshot_root if snapshot_root is not None else Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
    )

    evidence_records: list[OperatorTransferEvidence] = []
    creds = credentials

    for item in plan.items:
        evidence_records.append(
            _execute_one_item(
                item=item,
                plan=plan,
                force=force,
                evidence_dir=evidence_dir,
                transfer_fn_map=transfer_fn_map,
                verify_fn=verify_fn,
                ls_fn=ls_fn,
                credentials=creds,
                lock_root=resolved_lock_root,
                snapshot_root=resolved_snapshot_root,
            )
        )

    if evidence_dir is not None:
        _write_evidence_files(evidence_dir, tuple(evidence_records))

    return tuple(evidence_records)


def _execute_one_item(
    *,
    item: OperatorTransferItem,
    plan: OperatorTransferPlan,
    force: bool,
    evidence_dir: Path | None,
    transfer_fn_map: Mapping[str, TransferCallable] | None,
    verify_fn: VerifyCallable | None,
    ls_fn: LsCallable | None,
    credentials: S3Credentials | None,
    lock_root: Path,
    snapshot_root: Path,
) -> OperatorTransferEvidence:
    """Execute one transfer item and return its evidence record."""
    # Resolve the execution tool from the destination scheme.
    try:
        tool = resolve_execution_tool(item.destination)
    except ValueError as exc:
        raise OperatorTransferError(str(exc)) from exc

    with tempfile.TemporaryDirectory(dir=str(snapshot_root)) as tmpdir:
        tmp = Path(tmpdir)
        # Snapshot local sources (or size-preflight remote sources) so the bytes
        # that are hashed are exactly the bytes that are later uploaded.
        transfer_source = _prepare_source(item, credentials=credentials, ls_fn=ls_fn, tmpdir=tmp)

        # Serialize the destination existence-check + copy + verify critical
        # section against concurrent executions targeting the same key.
        fd, _lock_path = _acquire_destination_lock(item.destination, lock_root)
        try:
            # Overwrite-refusal without --force
            exists, existing_size = _remote_object_exists(item.destination, credentials=credentials, ls_fn=ls_fn)
            if exists and not force:
                raise OperatorTransferError(
                    f"destination already exists with size {existing_size}: {item.destination}; use --force to clobber"
                )

            # Dispatch the transfer
            if transfer_fn_map is not None:
                transfer_fn = transfer_fn_map.get(tool)
                if transfer_fn is None:
                    raise OperatorTransferError(f"no transfer callable registered for tool {tool!r}")
            else:
                if tool == "s5cmd":
                    transfer_fn = s3_transfer.cp
                else:
                    raise OperatorTransferError(f"no production transfer callable for tool {tool!r}")

            result = transfer_fn(transfer_source, item.destination, credentials=credentials)
            if isinstance(result, PlannedTransfer):
                raise OperatorTransferError(f"transfer returned a dry-run plan for {item.destination}")
            if not result.ok:
                raise OperatorTransferError(
                    f"transfer failed for {item.destination} with rc={result.returncode}: {result.stderr_tail.strip()}"
                )

            # Post-transfer verification (full download + sha256, no sampling)
            verify_path = tmp / "verify.tmp"
            if verify_fn is not None:
                verified_size, verified_sha = verify_fn(item.destination, verify_path, credentials=credentials)
            else:
                verified_size, verified_sha = _default_verify_fn(item.destination, verify_path, credentials=credentials)

            if verified_size != item.size_bytes:
                raise OperatorTransferError(
                    f"post-transfer size mismatch for {item.destination}: {verified_size} != declared {item.size_bytes}"
                )
            if verified_sha != item.sha256:
                raise OperatorTransferError(
                    f"post-transfer sha256 mismatch for {item.destination}: refusing to attest stale bytes"
                )

            # Build evidence record
            nonce = uuid.uuid4().hex
            evidence_id = operator_transfer_evidence_id(
                {
                    "source": item.source,
                    "destination": item.destination,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "nonce": nonce,
                }
            )

            # original_evidence_preserved is True ONLY when force=True AND a prior
            # evidence file for the same (item.source, item.destination) pair already
            # exists in evidence_dir; False otherwise.
            original_preserved = False
            if force and evidence_dir is not None:
                original_preserved = _has_prior_evidence(evidence_dir, item.source, item.destination)

            return OperatorTransferEvidence(
                evidence_id=evidence_id,
                mode=plan.mode,
                operator_initiated=plan.operator_initiated,
                authority_reference=plan.authority_reference,
                authority_digest=plan.authority_digest,
                override_destination_prefix=plan.override_destination_prefix,
                item=item,
                transfer_tool=result.tool,
                transfer_argv=result.argv,
                transfer_returncode=result.returncode,
                transfer_elapsed_s=result.elapsed_s,
                verified_size_bytes=verified_size,
                verified_sha256=verified_sha,
                transferred_at=_timestamp(),
                original_evidence_preserved=original_preserved,
                evidence_nonce=nonce,
            )
        finally:
            _release_destination_lock(fd)


def _prepare_source(
    item: OperatorTransferItem,
    *,
    credentials: S3Credentials | None,
    ls_fn: LsCallable | None,
    tmpdir: Path,
) -> str:
    """Prepare the transfer source, returning the path/URI to transfer from.

    For local sources this snapshots the file into ``tmpdir`` and verifies the
    SNAPSHOT (size + streamed sha256), so the bytes that are hashed are exactly
    the bytes that are later uploaded — closing the TOCTOU window between the
    pre-transfer re-hash and the transfer subprocess reopening the source path.
    For remote s3:// sources this performs a size pre-flight and returns the URI.
    """
    if item.source.startswith("s3://"):
        exists, remote_size = _remote_object_exists(item.source, credentials=credentials, ls_fn=ls_fn)
        if not exists:
            raise OperatorTransferError(f"remote source size mismatch or missing: {item.source} not found")
        if remote_size != item.size_bytes:
            raise OperatorTransferError(
                f"remote source size mismatch or missing: {item.source} "
                f"has size {remote_size}, expected {item.size_bytes}"
            )
        return item.source

    source_path = Path(item.source)
    if not source_path.exists() or source_path.is_symlink() or not source_path.is_file():
        raise OperatorTransferError(
            f"refusing to transfer stale bytes: source is missing or not a regular file: {source_path}"
        )
    snapshot = tmpdir / "source-snapshot"
    shutil.copyfile(source_path, snapshot)
    actual_size = snapshot.stat().st_size
    if actual_size != item.size_bytes:
        raise OperatorTransferError(
            f"refusing to transfer stale bytes: source size {actual_size} != "
            f"declared {item.size_bytes} for {source_path}"
        )
    actual_sha = _hash_bundle(snapshot)
    if actual_sha != item.sha256:
        raise OperatorTransferError(
            f"refusing to transfer stale bytes: source sha256 mismatch for {source_path} "
            f"(recomputed {actual_sha}, declared {item.sha256})"
        )
    return str(snapshot)


def _destination_lock_path(destination: str, lock_root: Path) -> Path:
    key = hashlib.sha256(destination.encode()).hexdigest()
    return lock_root / f"{key}.lock"


def _acquire_destination_lock(destination: str, lock_root: Path) -> tuple[int, Path]:
    """Acquire an exclusive advisory lock scoped to one destination object key.

    Serializes the destination-existence check + copy + verify critical section
    against concurrent operator executions targeting the same key on the same
    node. Returns ``(fd, lock_path)``; release with
    :func:`_release_destination_lock`.
    """
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = _destination_lock_path(destination, lock_root)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise OperatorTransferError(f"concurrent transfer in progress for destination {destination}") from exc
    return fd, lock_path


def _release_destination_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def verify_operator_transfer(
    item: OperatorTransferItem,
    *,
    credentials: S3Credentials | None = None,
    verify_fn: VerifyCallable | None = None,
) -> tuple[int, str]:
    """Download and verify a remote object's size+sha256 (full hash, no sampling).

    Returns ``(verified_size_bytes, verified_sha256)``. Raises
    :class:`OperatorTransferError` on any mismatch.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        verify_path = Path(tmpdir) / "verify.tmp"
        if verify_fn is not None:
            verified_size, verified_sha = verify_fn(item.destination, verify_path, credentials=credentials)
        else:
            verified_size, verified_sha = _default_verify_fn(item.destination, verify_path, credentials=credentials)

    if verified_size != item.size_bytes:
        raise OperatorTransferError(
            f"verify size mismatch for {item.destination}: {verified_size} != declared {item.size_bytes}"
        )
    if verified_sha != item.sha256:
        raise OperatorTransferError(
            f"verify sha256 mismatch for {item.destination}: {verified_sha} != declared {item.sha256}"
        )
    return (verified_size, verified_sha)


__all__ = [
    "LsCallable",
    "OperatorTransferError",
    "TransferCallable",
    "VerifyCallable",
    "execute_operator_transfer",
    "verify_operator_transfer",
]
