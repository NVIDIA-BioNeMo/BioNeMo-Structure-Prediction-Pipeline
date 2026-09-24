# Governed release execution

Development runs remain intentionally lightweight: they may render and run from repository paths and do not need
release-provenance evidence. Canary and production runs are governed. Their inputs must be immutable identities for a
source package, a clean tracked toolkit subtree, and the execution image; absent, legacy, future-version, or unknown
identity formats fail closed only in those governed run kinds.

Governance is an execution boundary, not publication authority. Governed processing may produce candidate outputs and
validation results, but it cannot publish them. Publication requires a separate, acceptance-bound approval referring to
the accepted result identity. This prevents a successful compute job from implicitly approving its own outputs.

## Source package format and verification

Source packages use a dispatch registry. `bspp-tar-v1` is the initial deterministic producer and verifier format;
`safe-tar-v1` is verifier-only. Unknown producers, formats, verifiers, or identity versions are rejected. The embedded
canonical manifest is the sole non-payload control member. It records relative POSIX UTF-8 paths in bytewise
lexicographic order with size, SHA-256, and normalized mode. Compact sorted JSON plus one newline is hashed exactly.
The package SHA-256 covers the final archive bytes; the manifest SHA-256 covers the exact embedded manifest bytes.
Commit and tree identities are bound by both the external identity and embedded manifest.

The verifier treats archives as hostile. It bounds package/member sizes and counts and rejects absolute or traversing
paths, backslash ambiguity, duplicates, links, devices, FIFOs, non-regular and sparse members, PAX path overrides,
truncation, extra or missing members, and size, mode, or digest mismatches. Extraction is allowed only into a new private
destination and never delegates path handling to `tar`.

## Toolkit and image identity

The toolkit identity is derived from `git ls-files --stage` for the configured subtree. Governed execution rejects any
staged or unstaged change, deletion, type change, or mode change that can influence the tracked snapshot. Untracked files,
whether ordinary or ignored, cannot influence that snapshot and are allowed but never packaged. Tracked symlinks,
gitlinks, and non-regular files are rejected. Only tracked regular files are copied into a mode-0700 scratch tree, with
executable bits preserved, using no-follow stable reads and a canonical manifest digest.
For governed execution that tracked snapshot is encoded with the same `bspp-tar-v1` / `safe-tar-v1`
`SourcePackageIdentity` used by the orchestration payload. The qualification tuple names the two instances separately as
`source_package_identity` and `toolkit_package_identity`; no checkout identity is persisted or mounted. Packages are
published by digest below `governed_package_root` (defaulting to `<runtime_qualification_root>/packages`) with unique
temporary names and no-clobber publication. An existing regular, non-symlink target is accepted only when its exact size
and SHA-256 match.
Each identity also binds its package role and policy version. An `orchestration` package is accepted only when every
manifest member belongs to the generic runtime-source/script allowlist; relabeling a self-consistent full-repository
archive therefore fails. A `toolkit` package is produced solely from stage-0 tracked regular files.

Runtime Qualification has two intentionally separate roots. `runtime_qualification_root` is cluster-visible; SSH
attempt input, script, immutable result, and scheduler logs live in a unique attempt directory beneath it.
`runtime_qualification_control_root` is an explicit absolute controller-local root (required for SSH and defaulting to
the qualification root for local Slurm). Canonical records, attempt metadata, and bounded fetched results live there.
The controller renders locally, stages exact input and script files, submits the remote script pathname, and later
fetches and verifies the immutable result before atomically promoting that exact local submitted attempt. Restart uses
the persisted local/remote path tuple; remote pathnames are never interpreted through controller-local filesystem APIs.
The durable lifecycle is `prepared` (no scheduler side effect), `submitting` (immediately before `sbatch`), `submitted`
(a parsed job id is bound), then `qualified`. Every attempt has a cryptographic token in its scheduler name, input, and
result. A restart reconciles a `submitting` token by exact profile and owner and rejects ambiguous matches; failed, lost,
or expired attempts receive a new immutable token while their prior metadata remains in the canonical history.

After a smoke job completes, `bsppctl runtime resolve` is the public one-shot authority boundary. It promotes eligible
submitted evidence, authenticates the complete promoted-v1 document, and reports its tuple, pathname, byte size, and
SHA-256. The checker captures those canonical bytes with bounded stable reads. A settled valid same-tuple refresh may be
selected, but malformed evidence, tuple/path/source drift, or a pathname that does not settle fails closed. Currentness
and an immutable snapshot are returned together; legacy submission and postprocessing Retry consume only that snapshot
and never reopen the mutable authority pathname after authorization.

The runtime image policy defaults to `digest-checked`: use the canonical absolute original image path and bind its size
and SHA-256 through one stable, no-follow descriptor. A `trusted-cache` policy remains an explicit compatibility option;
it must reject cache collisions and does not require a permanent cache. The image is rehashed immediately before every
governed `srun`, including qualification smoke and workflow jobs. That precheck is one fixed isolated host-Python
invocation: it opens the original configured absolute pathname with `O_NOFOLLOW`, compares descriptor metadata before
and after streaming the expected size and SHA-256, and confirms the final pathname still names the same device/inode.

## Trust and TOCTOU limits

These controls remove ambient checkout, interpreter, archive, and inherited-environment trust. Governed jobs invoke the
fixed image Python directly with isolated flags and a small environment allowlist, verify identities before extraction
or import, and use unique mode-0700 private scratch copies. Governed containers receive read-only file mounts for only
the two packages, materialized RunSpec, and qualification record; ambient checkout and parent-directory mounts are not
used. They cannot protect against a malicious kernel, scheduler, container runtime,
storage administrator, or mutation after the final precheck by a principal able to replace an already-open execution
object. The descriptor cannot be handed through Pyxis in this account-neutral design. The check therefore closes drift
between qualification and the immediately preceding pre-`srun` gate, but an unavoidable pathname-open micro-gap remains
between that check and the container runtime opening the image. Stable descriptors and immediate rechecks narrow rather
than mathematically eliminate TOCTOU exposure.

Runtime Qualification mounts its submitted identity record read-only and writes smoke results into a separate designated
attempt result directory. The result binds the tuple, scheduler job, source package, toolkit package, image, Python, and
GPU probe. A control-side check atomically promotes only that exact submitted attempt to `qualified`; stale or mismatched
results cannot qualify a different attempt.

## Runtime-built iPSAE evidence

Qualification builds iPSAE from the private tracked toolkit package inside the selected runtime image and its scrubbed
environment. It does not require a prebuilt `ipsae_cpp` in the checkout, perform a host build, or compare host and
runtime binary hashes. Any packaged binary is removed from the private copy and `make -B` forces compilation before the
result is retained. Because `ipsae_cpp` has no version command, its compatible version identity is the exact toolkit
source revision plus the newly built binary SHA-256; compiler and Make version output are recorded separately. A failed
build, toolchain probe, functional test, or GPU check prevents a success result
from being written and therefore prevents governed submission.

Postprocessing renderer contracts 2 and 3 may mount only the AWS shared-profile `credentials` and `config` files read-only at
`/workspace/bspp-aws/credentials` and `/workspace/bspp-aws/config`. The post-bootstrap action body sets only
credential-file and selected-profile variable names; it never renders credential values. This is a source-bundle
renderer change, not a bootstrap/image overlay.

The successful smoke result contains an independently versioned `RuntimeIpsaeEvidence` record. It binds the toolkit
source revision and relative source path, exact build argv, bounded stdout/stderr and retained build-log identity,
compiler and build-tool version output, the retained binary path/size/SHA-256, and a deterministic single-worker batch
functional check. The build binds `CXX=g++` and records that same compiler's version. The functional check uses a tiny
generic two-chain PDB and paired object-form `*-meta_v1.json` PAE input and requires the exact model identity,
directional `ipsae_AB`/`ipsae_BA` scores, and canonical semantic result digest; a wrong filename, array-form metadata,
missing directional columns, header-only result, or numerically wrong result fails qualification. Runtime iPSAE evidence
is smoke evidence,
not part of the qualification tuple: the tuple identifies
the immutable inputs, while the evidence records what those inputs produced in the selected runtime. Canary and
production retain that successful record; development remains unaffected.

## Create-once submission and execution evidence

Each governed workflow step is claimed before `sbatch` through a default single-controller coordinator under the run's
evidence directory. Its content-derived token binds the canonical materialized RunSpec, rendered payload script,
bootstrap, current Control State, Runtime Qualification record, logical step and slice, and attempt. Slurm receives that
exact token as both a job-name selector and a protected runtime binding. A restart reuses the bound job, reconciles the
token against live and completed scheduler accounting, or fails closed; an uncertain claim never authorizes a second
submission. Deployments that genuinely have multiple controllers may supply a lease/fence adapter, but do not need that
machinery in the default topology.

The controller writes a create-once expectation before scheduler contact. The job-side wrapper authenticates that exact
expectation and token, runs the governed payload, and durably creates the matching result with its scheduler job ID,
outcome, and qualified runtime iPSAE source/binary observations. Scheduler completion alone cannot promote a governed
step: a missing result remains pending for reconciliation, while an altered or wrongly bound result fails closed.
Completed evidence is reconciled through a canonical byte-sorted relative-path/SHA-256 index; missing, extra, replaced,
duplicate, linked, and non-regular entries are rejected. These controls do not change the easy development path.

## Independent acceptance and publication approval

Governed canary and production RunSpecs cannot self-upload and cannot contain upload workflow steps. Development retains
its existing direct behavior. Release acceptance is computed independently from candidate inventory, local tar member
names and payload digests, semantic observations, scheduler terminal states plus retained results, zero external uploads,
and the exact governed-release verifier processing-provenance index. Expected counts and scientific thresholds are inputs to the release
record; the library contains no campaign-specific cohort or threshold.

The processing index is bound by canonical relative path, SHA-256, processing step indices, and materialized RunSpec
SHA-256. Acceptance reopens the bounded index without following links, parses the governed-release verifier contract, and delegates exact-tree
verification to the governed-release verifier's public processing-boundary verifier. Worker `ok` flags cannot override a missing result, altered
archive member, semantic mismatch, scheduler failure, unexpected upload, wrong RunSpec, or changed evidence tree.

`bsppctl release approve-publication` reopens the canonical acceptance record, reruns every acceptance check, and then
creates one durable approval bound to the acceptance SHA-256 and exact destination. The approval is separate from
processing and is consumable exactly once at the external publication boundary. Absent, stale, altered,
destination-mismatched, linked, or consumed approval records fail closed. Creating an approval does not itself upload
anything.
