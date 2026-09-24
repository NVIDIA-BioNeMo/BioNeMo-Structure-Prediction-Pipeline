# Preprocessing Runtime Image

This directory defines the dedicated, baked-wheel preprocessing Execution
Runtime Image. `image-lock.json` records immutable primary provenance:
ColabFold v1.6.2 commit `c7d1772352cc9619df25c6d36cb0f218c0c6610e`
selects MMseqs commit `8cc5ce367b5638c4306c2d7cfc652dd099a4643f` in
its immutable Dockerfile; release `18-8cc5c` supplies the digest-verified GPU
artifact and its `gpuserver` command. The linux/amd64 CUDA base is also selected
by manifest digest. `pixi.lock` fixes Python, ColabFold, archive tools, shell,
coreutils, `flock`, and exact `rsync` 3.4.4 build `h5440a77_0`. The image
exposes that locked binary at the fixed `/usr/bin/rsync` path and qualification
cross-binds its normalized version to the embedded image identity.

The image entrypoint unconditionally selects the CUDA 12.6 user-mode
compatibility library baked by the pinned CUDA 12.6.3 base. It resolves
`/usr/local/cuda-12.6/compat/libcuda.so.1`, rejects a missing, unreadable,
non-regular, or escaping target, and prepends the compatibility directory to
`LD_LIBRARY_PATH`. There is no runtime override. This composition rule is not
by itself a claim that the library supports every host driver: changing the
cluster driver requires a fresh real-kernel characterization on that Cluster
Profile.

The repository-wide container commands are the supported operator interface.
For a local development build and composition smoke, start from a clean
committed checkout with network access and Docker (no host-side Pixi is
required — the pinned Pixi and MMseqs binaries are downloaded and verified
inside the Dockerfile):

```bash
containers/scripts/build.sh preprocessing
containers/preprocessing/smoke-local.sh bspp-orchestration:preprocessing
```

For a published qualification image, `push.sh preprocessing` performs a fresh
build. A smoke run before that command therefore does not attest the pushed
image. Publish first, confirm the local
`bspp-orchestration:preprocessing` image ID matches `image_id` in
`dist/build-record.json`, and then run:

```bash
containers/scripts/push.sh preprocessing
containers/preprocessing/smoke-local.sh bspp-orchestration:preprocessing
```

On the cluster, import that same registry target through the established CPU
Slurm helper:

```bash
sbatch containers/scripts/pull-sqsh.sh preprocessing
```

The build downloads and sha256-verifies the pinned Pixi and MMseqs artifacts
inside the Dockerfile, builds and hashes the Contract and Runtime wheels, and
writes the untracked `dist/build-record.json`.
The local build records only the Docker `image_id`. The push strictly tags
that exact fresh image ID, publishes the canonical registry reference, and
atomically supplements the record with `registry_image` and the canonical
registry `oci_digest`. It never treats a local image ID as an OCI registry
digest.
The local smoke runs with `--network none`, no source mount, real image archive
tools, and fake scientific kernels through the exact `preprocessing
execute-chunk` process boundary. It also checks the pinned compatibility
symlink, effective loader ordering, and the absolute baked characterization
helper/probe paths. It proves image composition, not live cluster or
scientific-data acceptance, and deliberately requires neither a GPU nor an
image-native `nvidia-smi` binary. In particular, it does not initialize CUDA.

The smoke also runs the actual pinned MMseqs `result2msa` and ColabFold
assembler on tiny generic synthetic alignments. Its fresh scientific-v3
plan selects `--filter 1`, retaining unpaired filtering while preserving
paired row identities. Two cases check exact row order and sequence content,
including an input where independent chain filtering would silently cross-pair
equal row counts. The resulting `synthetic_paired_row_identity_sha256` is
composition evidence only. No real search or database is involved. Historical
scientific-v1/v2 and carry-characterization records remain unchanged at
`--filter 2`; the new policy requires a fresh Phase and matching qualified image.

`containers/preprocessing/build.sh` remains the dedicated implementation
backend for the common target and may still be invoked directly for backend
development. It is not a separate live build/publish/import workflow. New
pipeline images must be wired into `containers/scripts/build.sh`, `push.sh`,
and `pull-sqsh.sh`; do not create issue-specific Docker, registry, Enroot, or
SquashFS scripts.

Before direct-requested science, the image exposes the strict four-argument
placement boundary:

```bash
bspp-orchestration-runtime preprocessing place-database \
  --phase-runspec /path/to/phase-runspec.json \
  --action-id preprocessing-chunk-000000 \
  --source-manifest /path/to/database-source-manifest.json \
  --write-result /path/to/database-placement-result.json
```

The command validates the protected read-only source mount, performs one
complete metadata-only pre-science inventory without reading payload bodies,
and exclusively publishes the immutable Result. The boundary intentionally
requires Linux `/proc/self/mountinfo`, `O_DIRECTORY|O_NOFOLLOW`, and
descriptor-relative filesystem operations; it has no path-reopening portability
fallback. It has no source-root,
mountinfo, policy, branch, cache, capacity, mover, or replica override. Direct
scientific execution and post-science evidence remain the boundary check.

## Acceptance cache clearing

Before a deliberate cold live-acceptance run, use only a dedicated profile
whose cache root is isolated from production and which includes:

```yaml
database_cache_namespace: acceptance
database_cache_root: /absolute/dedicated/acceptance/cache
database_cache_unix_user: exact-effective-user
database_cache_filesystem_type: lustre
database_cache_reserve_bytes: 0
database_lock_wait_seconds: 300
```

Then run the cluster-side operator command outside any Phase Action:

```bash
bspp-orchestration-runtime preprocessing database-cache clear \
  --config /path/to/profiles.yaml \
  --profile example-cluster-acceptance \
  --write-evidence /path/outside-the-cache/database-cache-clear.json
```

The command derives the effective-user namespace; it accepts no cache or user
override. It is idempotent for a missing/empty `replicas` directory, refuses
production, lexically or physically aliased, broad, cross-user, escaping, or
active namespaces with no deletion, and preserves `.locks`, siblings, and the
namespace roots. Configured selected and sibling cache roots must not be equal
or lexically or physically contain one another in either direction. Runtime
binds the selected cache/user/replica scope and every available sibling before
it reserves evidence. It creates a missing evidence parent component by
component without following links and cleans up only unchanged empty
directories created by that invocation.

The retained selected cache root, `users`, and effective-user namespace form
one exact physical chain. Optional existing `.locks` and `replicas` must each
be exact direct children of that namespace: every child remains on the
parent's mount and at the parent's backing coordinate plus its exact basename.
When `.locks` or `replicas` is absent at freeze, Runtime rechecks that exact
direct-child absence and does not manufacture either; a nonempty `replicas`
inventory requires `.locks` to exist.
Runtime checks every configured sibling root against every retained level for
equality or backing containment in either direction. It re-observes the
complete hierarchy and full sibling-by-selected matrix before each removal, so
a sibling or arbitrary external directory bind-mounted onto `users`, the user
namespace, `.locks`, or `replicas` refuses without deletion. Any configured
sibling root whose terminal or intermediate component is a symlink is
categorically refused before mutation. Runtime uses the separately opened,
descriptor-resolved sibling only to exclude unsafe evidence publication. A
configured root mounted directly on an accepted top-level entry still fails
closed at the atomic claim boundary and its backing is preserved.

After acquisition, `cache.lock` and every sorted relevant identity lock are
frozen as exact direct regular-file children of `.locks`. Runtime rechecks
their visible and retained mount authority, together with the directory and
sibling graph, immediately before each removal and at the terminal held-lock
boundary.

Publication remains disabled until the retained evidence parent is proved
physically outside every bound managed or sibling backing through descriptor
ancestry and Linux statx mount facts joined to strict
`/proc/self/mountinfo` root/mountpoint coordinates. Unavailable, malformed, or
ambiguous mount authority before that proof leaves no terminal evidence. An
evidence parent whose backing is equal to, above, or below any managed backing
also prevents publication proof and leaves no terminal evidence. By contrast,
fully bound selected/sibling physical overlap is safely reported in strict
refused evidence through a separately authorized external parent. Once proved,
evidence authority remains independent of the deletion-target authority
rechecked immediately before every removal. Target drift after a confirmed
removal stops further deletion and publishes strict `failed` evidence for the
confirmed removals through the retained parent descriptor. Final verification
and reload are descriptor-relative. A hostile same-UID unlink or replacement
of the evidence basename fails closed; Runtime never relinks or overwrites the
uncertain name and cannot guarantee the requested visible pathname in that
adversarial case. The normal result is one strict immutable-at-creation `0444`
evidence file containing
digest-bound complete removal/lock summaries with bounded top-level samples
but no payload inventory. A transient final write or file-fsync fault after a
removal is reported by a strict `failed` record containing the exact confirmed
removal facts.

The prepared one-command-at-a-time procedure for the single adapter-v3 cold
Example cluster lifecycle is documented in a dedicated acceptance runbook.
Its checked-in preparation directory is explicitly not live evidence;
only a later checksummed sample-run directory may close that acceptance claim.

Scheduled Runtime Qualification runs a guarded `nvidia-smi` query on the
allocated host before the sealed container step, then relays its exact output
to the image as qualification evidence. The query retains one line for every
host-visible GPU rather than selecting one silently, and records GPU name plus
driver version. This proves host GPU/driver evidence in the allocated job
context; it does not prove that the image received `/dev/nvidia*` devices or
driver libraries, nor that a scientific CUDA kernel initializes. The separate
real-kernel characterization remains the live gate for those claims. That gate
binds one immutable qualification tuple and Cluster Profile to same-job host
GPU evidence, in-container compatibility-loader evidence, successful CUDA
Driver API initialization, and a bounded real MMseqs/ColabFold search. Its CPU
and memory request—not a partition name—match the production Cluster Profile;
node-identity equality relies on the pinned single-node allocation.

The scheduled composition smoke mounts a fresh job-local directory at
`/run/bspp/database`, so it remains writable when Pyxis runs the image as the
cluster user and cannot write through an operator-supplied database mount.
Qualification rejects every profile mount whose source or target overlaps that
protected namespace and removes the isolated directory when the job exits.

The characterization Bash helper and CUDA probe are baked into the image so a
remote checkout cannot overlay or hotfix them. Any helper/probe change therefore
changes the image and requires a new image, tuple, scheduled composition
qualification, and real-kernel characterization. After the scientific helper
returns, schema-2 evidence assembly runs as a second sequential step through
the same image, `/cluster` mount, no-home policy, entrypoint, and the qualified
absolute Python at `/opt/bspp/environment/bin/python`; host Python is not part
of this execution contract. Although the Slurm wrapper is not baked into the
image, a wrapper-only commit still changes the source commit pinned in the
embedded image manifest and therefore requires a fresh image, Source Bundle,
qualification tuple, scheduled qualification, and characterization.

## Product raw-search closure

Adapter v3 sends the pinned search to a dedicated raw-search directory,
separate from named package staging. Current `unpaired_paired` execution
accepts exactly `M` declared named A3Ms for `M` source-ordered records: ColabFold
consumes and removes its per-chain unpaired files during assembly. Historical
sealed `paired` plans additionally require numeric placeholders `M..U-1`,
where `U` counts per-record unique chains; their exact bytes derive from chain
lengths and cardinalities. Validation and reconciliation select the inventory
from the sealed argv, reject any missing or extra artifact, and retain all
named A3M content and hash checks. Runtime publishes only the validated
named files into staging. Raw artifacts are retained in typed action evidence;
they are not package members or output hashes.

This product contract is distinct from the diagnostic
`raw_search_inventory` below. Diagnostic tools preserve bounded observations
for discovery and qualification; they do not load product RunSpecs or confer
adapter-v3 execution authority. The
one-record live-slice input and provenance record select the exact
one-record live-slice input. Their prior observed depth of 1/1 is not a depth
adequacy or Product B claim.

## Diagnostic metadata recovery and fixture discovery

The recovery path is deliberately separate from product acceptance.
Committed tools under `containers/scripts/` preflight the padded database,
stream the readable metadata archive once into an immutable application-owned
pair, audit mapping/key compatibility, build and validate a two-root database
view, and measure a fixed discovery-only candidate corpus. They run through
the pinned container Python from a read-only source mount. The local Python
pass only renders and hash-binds an intent; the sourceable Bash library stages
that intent verbatim, submits it, monitors Slurm, and retrieves bounded
evidence. Example cluster host Python is not part of the execution contract.

The monitor treats a failed `squeue` query as unknown live presence rather
than queue absence and then defers to bounded `sacct` polling. It publishes
immutable `monitor.result` bytes only for a complete, recognized top-level
terminal accounting row. A failed accounting query, accounting lag, or
nonterminal exhaustion returns failure without publishing a record, leaving
the handoff safely resumable.

Candidate-depth evaluation has a narrow, failure-only scratch exception. Its
schema-1 `raw_search_inventory` requires `search-output`, when present, to be
a real non-symlink directory and hashes every regular direct child. Helper
exit 0 rejects every direct-child directory. A nonzero helper exit may retain
only one real direct-child directory named exactly `tmp`; the evaluator treats
that scratch as opaque, never opens or lists it, and omits it from both the
inventory and result. Any other directory, symlink, FIFO, or special object is
rejected for every exit. This live discovery inventory is separate from the
preserved-layout `directory-inventory` described below. A regular file named
`tmp` is an ordinary inventory member; for exit 0 the later exact N-named plus
N-numeric closure rejects it as an extra artifact.

The sharded-layout helper models pinned MMseqs commit
`8cc5ce367b5638c4306c2d7cfc652dd099a4643f`: `.0`, `.1`, ... data shards are
canonical only when contiguous from zero; an unsuffixed data file is accepted
only when no shard exists. This follows
[`FileUtil::findDatafiles` lines 330--345](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/FileUtil.cpp#L330-L345);
result access is deliberately narrower than
[`DBReader::open` lines 108--132](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/DBReader.cpp#L108-L132)
and
[`DBReader::getDataByOffset` lines 630--642](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/DBReader.cpp#L630-L642).
It treats copy payloads as opaque, permits only
source-root-confined component links, and emits regular destination files plus
immutable manifests. Before any result layout is opened, the bounded
`directory-inventory` operation publishes immutable `lstat` evidence for the
entire preserved search-output directory, including type, mode, size, mtime,
device, inode, and link target. It caps entries and name length and never
follows links.

The generic `result-target-keys` operation accepts only uncompressed, unpadded
alignment/prefilter result databases with NUL-framed text records. Its
schema-2 manifest records the source basename so `res` and `res_exp` cannot be
confused. It identifies the first field of every result row as a target-member
key, not a database primary key; duplicates across query records are valid and
remain in the one-million-row-capped stream. Missing data alongside valid
`.dbtype`/`.index` is recorded as `data_absent`, not fabricated as an empty
result and not accompanied by a key file.
The raw little-endian dbtype evidence and decoding follow
[`Parameters.h` lines 68--94](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/Parameters.h#L68-L94),
[`DBReader.h` lines 370--376](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/DBReader.h#L370-L376),
and [the bit-31 compression check](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/DBReader.cpp#L1044-L1045).

The padded-database preflight also emits schema 2. Each nonblank
`uniref30_2302_db_pad.lookup` row must contain exactly three TAB-separated
fields after only the line ending is removed. Columns 1 and 3 are canonical
bounded unsigned-decimal identifiers; column 2 is nonempty and may contain
spaces. Lookup column 1 must equal the padded-primary `pad.index` key set and
column 3 must equal the base-primary `db.index` key set exactly; duplicate
identifiers in either lookup column are invalid. The pinned provenance for
this correspondence is the index and lookup construction in
[`makepaddedseqdb.cpp` lines 90--128](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/util/makepaddedseqdb.cpp#L90-L128)
and the exact `id<TAB>entryName<TAB>fileNumber` writer in
[`DBReader.cpp` lines 741--748](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/DBReader.cpp#L741-L748).
No direct equality is asserted between pad and base primary keys.

Every other cross-namespace comparison is explicitly observational:
`_aln.index` overlap/only counts are measured against pad and base; `res`
target-key coverage is measured against pad, base, and `_aln`; and `res_exp`
target-key coverage is measured against `_seq.index`. These measurements
include total and distinct absent counts, bounded witnesses, and
confirmation/refutation verdicts where an expectation exists, but never form
a structural gate. The captured 33 missing alignments is evaluated against
distinct `res` targets absent from `_aln`, while the corresponding repeated-row
occurrence count (captured as 66) remains separate. `makepaddedseqdb` does not
renumber the stock `_aln`, and neither key magnitude nor overlap implies
identity across namespaces.

The preflight constructs pad, base, alignment, and sequence key sets
independently. Each bitmap is sized from its own namespace maximum and guarded
by both an absolute byte cap and a range/cardinality-density bound. Phase-local
release plus an aggregate live-bitmap budget limits peak memory even when each
individual allocation is legal. A candidate from another namespace that is
above the tested set's maximum is absent evidence, not a pathological-key
failure. Schema-2 reports publish a top-level `outcome` and enumerated
`failures` atomically before a structural-error exit; omitted or unavailable
result inputs remain explicit `not_observed` evidence.

Replay copy accepts exactly `qdb`, `qdb_h`, `prof_res`, `prof_res_h`, and
`res_exp_realign`, with `qdb.lookup` required. Outputs must be outside the
source root and have non-symlinked parents; an existing copy destination is
accepted only as a matching completed immutable retry.
An uncatchable hard kill can still leave a partial mode-`0400`
result-target-key file before the helper reaches its cleanup path. Treat that
as a failed attempt: use a fresh output path rather than trying to repair or
reuse the file.

Recovered mapping and taxonomy files are proven to come from the readable
archive by full compressed-stream accounting, archive identity, member
inventory, exact sizes, and output hashes. The inaccessible mode-0600 database
files cannot be compared byte-for-byte. Consequently, a passing keyspace audit
establishes interface compatibility; it establishes taxonomy correctness only
when database headers expose an organism identifier and the deterministic
sample agrees completely. A schema-2 database-view identity is a declared
stat/hash-policy drift detector, not a full hash of the other large live
database entries. It carries mtime in the identity only for stat-policy files
(although validation checks mtime for every entry), allowing byte-identical
hash-policy recovered files to move between attempt roots.

The committed 24-record FASTA is a public-sequence diagnostic asset generated
only from the pinned Git object in a canonical reference repository
recorded in its provenance manifest. The manifest's copy command
requires an environment variable to identify a checkout of that canonical origin;
a similarly named file in this repository or another mutable worktree is not
authority. Its 48 chains contain 10,320 public sequence bytes (10,816 FASTA
bytes including headers, separators, and newlines). The three required curated
records come first; the remaining 21 are the exact source-order,
distinct-chain, 50--500-aa fill recorded in the manifest. It is not the product
carry-forward fixture and does not pin product scientific constants. Phase-A
runs use the existing e804043 image strictly as a content-verified diagnostic
runtime. Only a depth-qualified winner can license a later product commit,
whose new image, Source Bundle, qualification tuple, scheduled qualification,
and real-kernel run must independently pass. Paired depth is reported as both
wrapper-compatible A3M records and complete-pair records; the latter excludes
dummy or half-dummy segment rows.

Schema-1 candidate-depth evidence keeps `res_exp_cardinality` as an optional
reported diagnostic. A `null` value means the pinned MMseqs `expandaln` did
not report that auxiliary metric; it means neither zero nor absent scientific
output, and the evaluator does not infer it from missing-alignment messages,
A3M depths, or a later count. A successful helper still closes through the
exact N named plus N numeric output inventory, parsing every named A3M, both
required `pairaln` durations, and an observed post-expansion
`res_exp_realign` cardinality. That required cardinality may validly be zero.

Selector upgrades never re-evaluate an old intent. Roots without a valid
result are rendered and executed afresh under the current selector. An
immutable systematic parent intent/result pair may instead license a split:
`render-split` validates the parent relationship, binds the child to the
current selector in both selector and allocated-source hashes, and preserves
the exact parent intent/result hashes in `parent_retry`. Selection consumes
terminal results executed under the current selector.

The accepted Phase-A evidence activated one bounded A2 pass. Its canonical
selection file is pinned by SHA-256
`24001644c106b03ab502f7cb0f11599338f5a29f6a8b8993719d05b997685e10`
and embedded evidence identity
`b1bcabca16d5ec2e18ce8ac49840f739eef3610064d32906921bf35bee6dcae3`;
only `A0A9W3HR45_A0A9W3HR47`, `A8T7C4_Q6RXH2`, and
`Q64845_A0A9W3HR60` may be retried. `render-a2` requires the exact canonical
base timeout result for its candidate and requires a fresh image path and hash,
changed helper bytes, the parsed search timeout from 900 to 1800 seconds, and
the job wall time from
30 to 45 minutes. Warmup remains 60 seconds, kill grace 10 seconds, search
threads 64, and CPU, memory, GRES, account, partition, Cluster Profile,
scientific argv, corpus, and identical schema-2 database view remain fixed.
The wall-time calculation is `60 + 1800 + 10 + 600 = 2470` seconds, rounded up
to the next five minutes (`2700`, or `00:45:00`). Ordinary `render` and
`render-split` intentionally retain the 30-minute default and therefore reject
an A2 helper at wall-time validation; only `render-a2` supplies the fixed
45-minute override.

Increasing Coreutils `timeout` changes only when SIGTERM is delivered. A helper
that exited 0 within 900 seconds was not truncated, so its measured depth is
invariant; only the three exit-124 singleton depths are unmeasured. The pinned
base depths are therefore immutable decision inputs to `resolve-a2`, while old
bindings and allocations are never mixed into ordinary homogeneous selection.
The resolver reconstructs all 24 candidates and calls the unchanged ranking
gate. The sixteen completed base candidates at depth `1/1` make a reduced
winner impossible. A repeat exit 124 stays truthfully classified
`unmeasured_timeout` but receives terminal policy disposition
`a2_timeout_disqualified`; another systematic singleton failure is likewise
terminal. Only a current-selector A2 candidate meeting the unchanged high gate
can win, and every conclusion is self-hashed.

No canonical batch-context baseline is reachable for the three A2 candidates:
every six-candidate batch and three-candidate half containing one also contains
a previously systematic candidate. The winning singleton timing is therefore
a measured lower bound. Product B uses the supervision-derived `00:45:00`
bound; its separate forward/reverse gate retains the recorded symmetry
assumption and fails closed. Any winner still requires a clean commit, newly
built image and Source Bundle, a fresh qualification tuple and scheduled
qualification, identical-view revalidation, and a separate real-kernel
characterization. A terminal A2 resolution ends discovery without product B.
