# Preprocessing Database Placement architecture

Database Placement keeps its public contract and runtime entry points fixed while
composing physical paths and nondeterministic capabilities behind private
coordinators. Production composition continues to bind `/run/bspp`,
`/proc/self/mountinfo`, and `/usr/bin/rsync`; tests supply disposable paths and
typed collaborators directly to private coordinators. No physical-authority or
test-only option is exposed through the CLI.

## Authority records

Filesystem authority is represented by frozen named records instead of
positional primitive tuples:

- `FilesystemAuthority` records kind, device, inode, owner, and permissions.
- `FilesystemObservation` records kind, device, inode, size, and modification
  time.
- `PathObservation` pairs an observation with its symlink target.
- `FilesystemBinding`, `DirectoryAuthority`, and
  `FilesystemObjectIdentity` retain the exact field set required at their
  respective evidence, cache-directory, and lease/repair boundaries.

Callers compare complete records unless they deliberately project a named
device/inode identity. Wire mappings, canonical bytes, digests, modes, and
lock order are unchanged.

## Module seams

`_database_replica_lock.py` is a compatibility facade over dependency-free
types, descriptor-relative authority checks, and lock coordination. Ordinary
placement retains identity-to-cache ordering; maintenance retains the opposite
cache-to-identities ordering.

`database_cache_maintenance.py` remains the public facade and state-machine
coordinator. Scope discovery, authority freezing, evidence durability, and
shared types live in focused private modules. Deterministic tests compose an
`OwnedTreeRemover` or evidence store only at those private boundaries.

Contract readers share strict primitives from private
`contract/_database_validation.py`. Consumer modules still own their deliberate
schema, collection, empty-string, NUL, normalization, and error-message
asymmetries.

Cold placement and execution use immutable `DatabasePlacementPaths` plus typed
capacity, selection, population, repair, evidence, wait, and scientific-kernel
services. Successful unprivileged execution tests call the private coordinator;
small CLI tests retain argument, help, and error-conversion coverage.

## Failed execution diagnostics

After exceptional kernel cleanup and source re-observation, Runtime retains its
own exclusively created search log at the action's declared durable log path.
It requires a terminal kernel (or a failed start), a regular non-symlink log,
and no rejected direct-source observation. Preflight failures cannot publish a
pre-existing scratch log. Publication uses the same exclusive, fsynced copy as
successful execution; a collision or copy failure preserves the original error
and adds the retention error without replacing the destination.

Retained logs remain diagnostic evidence of a failed action. Direct failures
use the existing exact log hash and paired log lines. Failed staged actions may
retain paired log lines but still have no accepted output hashes or raw-search
evidence. Database drift rejection, lease lifetime, successful publication,
and success-only finalization are unchanged. A nonterminal kernel is reported
explicitly and its still-changing log is not published as a final diagnostic.

## Test mutation inventory

The Database Placement, replica, lease, cache-maintenance, staged-execution,
phase-finalization, local-execution, Linux mount-authority, and shared-support
surface contains zero private product monkeypatches and zero
`object.__setattr__` calls. The 125
remaining mutations are external-boundary faults:

| Boundary | Count |
| --- | ---: |
| `os.*` syscalls and filesystem observations | 111 |
| `Path.lstat` | 4 |
| `ctypes.CDLL` / `ctypes.sizeof` | 3 |
| `io.open` | 3 |
| `subprocess.run` | 3 |
| `fcntl.flock` | 1 |
| Total | 125 |

`tests/test_preprocessing_database_mutation_policy.py` parses the complete
surface with `ast`, rejects private product mutation, and pins the exact narrow
external inventory. Product-module aliases such as `replica_runtime.os` are
normalized to the underlying standard-library `os` boundary.

Private product monkeypatches that remain elsewhere in the repository are
recorded baseline debt, outside Database Placement: folding A3M prefix reading
(1), preprocessing carry-forward file I/O (2), preprocessing metadata
materialization/scanning (10), MMseqs layout hashing (1), and worker packaging
compression (5). This refactor does not weaken or silently absorb those
unrelated subsystem tests.
