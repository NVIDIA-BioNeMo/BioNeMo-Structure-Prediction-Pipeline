# BSPP Orchestration

BSPP Orchestration describes how BSPP processing runs are specified, controlled, executed, and accepted across local workstations and HPC clusters.

## Language

**BSPP Processing Phase**:
A durable domain phase of BSPP production work, such as preprocessing, folding, or postprocessing. The current orchestration implementation covers all three of those phases behind one phase lifecycle, and the Control Plane and installation surfaces must remain able to expand across further BSPP Processing Phases.
This is distinct from historical "Phase 1" or "Phase 2" implementation milestone language in older docs and tests.
_Avoid_: package boundary, implementation milestone

**Phase Progression**:
The human-mediated transition from an accepted predecessor Phase Run and its validated Artifact Sets to an eligible successor phase. Eligibility never dispatches the successor; an operator must explicitly perform its Phase Submission.
_Avoid_: automatic phase chaining, downstream auto-launch, scheduler dependency

**Phase Receipt**:
The immutable successful-completion authority for one Phase Run, binding exactly one successful Phase Attempt and its accepted inputs, output Artifact Sets, acceptance results, and retained evidence. Earlier attempt outcomes remain in the Phase Run history; only a validated Phase Receipt makes successor Phase Progression eligible.
_Avoid_: Control State, Slurm success, operator approval, mutable completion marker

**Phase Finalization**:
The acceptance boundary that produces complete validated output Artifact Sets and verified local Artifact Locations before a successful Phase Receipt is published. Scientific Kernel completion and durable shared-store publication are neither substitutes nor universal prerequisites.
_Avoid_: kernel success, scheduler success, mandatory publication

**Postprocessing Finalization Handoff**:
The bounded, indexed, JSON-only directory atomically published by postprocessing Runtime after all scientific traversal and acceptance adjudication. It contains the Runtime action aggregate, input attestations, acceptance captures and reports, scientific tar manifests, and output Artifact Set/Location records that Control may validate locally. Its exact indexed members are authority; copied arguments, remote scientific locators, and unindexed descendants are not.
_Avoid_: output directory, arbitrary evidence copy, Control-side remote scan

**Postprocessing Acceptance Residual Allowance**:
A baseline-bound literal exception evaluated only after acceptance comparator
reports are proven coherent. Under `postprocessing-sealable-v2`, a tar
allowance names one exact difference kind and member with optional cardinality
`0..1`; duplicate occurrences, malformed values, globs, regexes, and derived
aggregate or verifier issues cannot match. The pinned reference includes an exact ModelCIF variant selected for
identity comparison and only the observed extra direction for its interface
metadata member. Historical V1 allowance
identity and evaluation remain unchanged.
_Avoid_: mismatch pattern, required known failure, verifier-issue allowance

**Acceptance Comparator Coherence**:
The pure Contract check that recomputes parity file flags, report aggregates,
semantic outcome, and the ordered verifier issue projection from primary
comparator facts. A disagreement is a synthetic unallowlisted occurrence and
can never be converted into an accepted residual.
_Avoid_: trusting report aggregates, allowlisting a forged verifier issue

**Action 09 Prepublication Witness**:
The in-memory postprocessing record that binds action 09's command and complete prospective handoff index before atomic publication. It is not successful Runtime action evidence. Control combines this witness with independently exported scheduler evidence and durable assignment events to prove action 09 succeeded without creating a circular prepublication claim.
_Avoid_: action-09 success evidence, publication receipt, scheduler observation

**Postprocessing Dataset Selector**:
The user-authored `dataset.name` that selects and labels the scientific tracking scope in a legacy postprocessing Run Plan. Postprocessing V3 preserves it across materialization and retry; an attempt-scoped legacy `dataset.run_id` is operational identity and must not replace it.
_Avoid_: Phase Run ID, Attempt ID, generated legacy run ID

**Postprocessing Attempt Output Authority**:
The complete set of active local and remote postprocessing output locators deterministically projected beneath or suffixed by one immutable V3 Phase Attempt. The Phase envelope's independent Cluster-output and object-base snapshots, Cluster staging root, envelope IDs, and execution-identity namespace derive the Attempt paths; payload paths cannot redefine that authority. Authored local leaves preserve their strict relative suffix beneath the Attempt output root; remote leaves follow explicit field-specific prefix rules. The projected legacy document digest binds the leaves in execution identity, and retry replaces every leaf without mutating the predecessor. Input locators, scratch, and inactive Data Placement destinations are outside this projection.
_Avoid_: top-level output-directory substitution, global locator rewrite, shared retry output

**Postprocessing Credential Mount Snapshot**:
The immutable V3 Attempt record of the canonical absolute POSIX source locators for the AWS shared-profile `credentials` and `config` files. Both locators are present together for an `aws:` reference and both are null otherwise. It is operational mount authority consumed by renderer 4 and 5 and contains neither credential values nor file contents.
_Avoid_: credential copy, secret snapshot, ambient HOME lookup

**Postprocessing Slurm Environment Contract**:
The closed internal renderer-to-bootstrap compatibility version controlling the outer governed `srun` Slurm assignments and the array success identity together. Renderer 5 arrays normalize `SLURM_JOB_ID` from `SLURM_ARRAY_JOB_ID` before the unchanged bootstrap scrub, then record `<parent>_<task>`; renderer 5 scalars deliberately use the frozen renderer-4-compatible version.
_Avoid_: caller-provided shell expressions, bootstrap image rebuild, changing historical renderer bytes

**Postprocessing Autorequeue**:
The optional postprocessing Attempt policy that requests Slurm to requeue a failed action incarnation and, for the audited group-A transport failures only, records one create-once per-restart classification and exits with the reserved code 85. It is disabled by default, requires a qualified Runtime cap pinning `RequeueExit 85` and a non-negative `MaxBatchRequeue`, and may only allowlist Actions 01--08. The restart ordinal is carried from the outer batch shell through a rendered restart-count file because the closed `env -i` allowlist does not forward `SLURM_RESTART_COUNT`.
_Avoid_: automatic retry of every failure, unconditional requeue, Action 09 allowlisting

**Phase Operation Intent**:
The create-once local coordinator record that binds one restartable Phase operation to its immutable inputs and expected authority transition. Initial-run intent binds the Phase Plan digest, pre-generated Phase Run ID, authority and execution roots, source and profile/config identities, and polling options. Explicit Retry intent instead embeds the exact authenticated predecessor-to-successor Retry event and Runtime Qualification snapshot with those identities and options. Durable Phase authority remains lifecycle truth; the operation intent prevents a changed invocation or a different Attempt from taking over the same execution root.
_Avoid_: Phase RunSpec, mutable checkpoint, lifecycle authority

**Phase Diagnostic Summary**:
A bounded, non-authoritative troubleshooting artifact derived from durable scheduler assignments, durable-first or fresh exact-exit causal failure selection, and, where a renderer contract defines exact paths, selected failed-action log tails. Dependency-cancelled actions are not treated as causal failures. It is stored outside Phase authority and Runtime handoff, is warning-only, and cannot affect Phase outcome or acceptance.
For a selected renderer-3/4/5 action09 failure it may additionally read the exact attempt-bound acceptance adjudication and, only for a failed verdict with positive non-allowlisted errors, its fixed capture/report support closure. Those reads are bounded, no-follow, and explanatory only.
_Avoid_: action evidence, scheduler authority, acceptance report

**Compatibility Port**:
A repository-owned representation of behavior from a pinned upstream BSPP Processing Phase implementation. It preserves the upstream scientific behavior and defaults while fitting the accepted BSPP Orchestration product boundaries. A Compatibility Port is complete when its in-scope behavior and minimum stable phase boundaries are owned and testable; it does not imply that scientific payload execution or end-to-end phase orchestration is available.
_Avoid_: scientific-kernel rewrite, end-to-end implementation

**Port Baseline**:
The immutable upstream source revision against which a Compatibility Port's behavior and provenance are evaluated. Later upstream drift does not change an active Port Baseline; adopting a different revision is a separately reviewed decision.
_Avoid_: live branch head, mutable upstream

**Scientific Kernel**:
An external scientific tool or engine that performs domain computation for an BSPP Processing Phase, such as MMseqs, ColabFold, OpenFold, or TRT-BioNeMo. BSPP Orchestration may validate inputs and construct invocations for a Scientific Kernel without owning or reimplementing its scientific algorithm.
_Avoid_: orchestration logic, phase controller

**Scientific Backend Adapter**:
The repository-owned boundary that translates phase work into a Scientific Kernel invocation and interprets its resulting artifacts within one Runtime Action. It does not own the phase action graph or job submission, retry, top-up, status, cancellation, or resume.
_Avoid_: Scientific Kernel, Workflow Backend, Phase Execution Coordinator

**Control Plane**:
The trusted local side of BSPP orchestration that owns run intent, cluster targeting, submission, monitoring, and evidence. It does not perform BSPP payload processing.
The local Control Plane command is `bsppctl`; it is the normal local CLI name for split Control Plane workflows.
_Avoid_: controller container, login-node runtime

**Control State**:
The durable workflow state written by the Control Plane into a run's evidence directory. It supports status and resume before the future Run Index exists.
_Avoid_: local registry, hidden laptop state

**Phase Execution Coordinator**:
The Control Plane role that coordinates the ordered work and lifecycle of one exact BSPP Processing Phase Attempt. It is the phase-orchestration boundary: it delegates scheduled-job transport to the Remote Slurm Transport and scientific computation to Scientific Kernels, without a separate workflow-engine backend. Coordinators for initial materialization and explicit Retry share per-Phase ownership but keep distinct create-once intents and results; neither may follow durable authority to an Attempt it did not bind.
_Avoid_: Workflow Backend, Phase Scheduler, Phase Workflow Controller, within-phase orchestrator

**Phase Lifecycle**:
The Control Plane lifecycle of one BSPP Processing Phase, comprising phase-scoped materialization, submission, status, resume, cancellation, finalization, and explicit retry. Legacy run-scoped control is a temporary compatibility surface, not a parallel domain model.
_Avoid_: Run Lifecycle, workflow-engine lifecycle

**Phase Materialization**:
The Phase Lifecycle transition that durably creates a Phase Run, its first Phase Attempt, and that attempt's immutable Phase RunSpec without dispatching Runtime Actions. It is an authoritative pre-submission boundary, not a disposable preview.
_Avoid_: plan preview, submission, implicit submit-time expansion

**Phase Submission**:
The Phase Lifecycle transition that durably records a complete direct-Slurm submission intent, then dispatches all initially declared Runtime Actions for an already materialized Phase Attempt using its exact stored Phase RunSpec bytes. It records each scheduler assignment, never rematerializes the attempt, and returns after dispatch rather than waiting for the phase to become terminal; an uncertain dispatch is correlated instead of blindly repeated.
_Avoid_: synchronous phase execution, plan-only submission, first-action submission

**Phase Run**:
One durable requested execution of an BSPP Processing Phase, identified across its Phase Lifecycle by an opaque, stable Phase Run ID. Its Phase Plan and input Artifact Set identities remain fixed across its immutable Phase Attempts, and a validated Phase Receipt permanently seals it; evidence storage and any future top-level run linkage are attributes, not identity.
_Avoid_: evidence-directory identity, composite run-and-phase identity

**Phase Attempt**:
One immutable materialized execution try beneath a Phase Run, binding exactly one Phase RunSpec and preserving its own outcome and evidence. A failed or cancelled Phase Attempt is never rewritten as its successor.
_Avoid_: Phase Run, resume cycle, mutable retry

**Attempt Carry-Forward**:
The immutable provenance record by which a successor Phase Attempt declares exact verified partial artifacts or checkpoints adopted from an earlier attempt in the same Phase Run. The current implementation is an opt-in Phase Retry feature for a nonempty proper subset of preprocessing A3Ms: Control attests source evidence, Runtime revalidates and copies exact bytes into a fresh target workspace, and the successful receipt references the ordered provenance closure. It never changes the Phase Run's scientific inputs or makes a shared work directory authoritative.
_Avoid_: implicit checkpoint discovery, shared-directory reuse, merged attempt evidence

**Phase Plan**:
The versioned user-authored intent for one BSPP Processing Phase, using a common lifecycle envelope and a phase-kind-discriminated configuration. It is distinct from the legacy postprocessing-shaped Run Plan and never relies on optional fields for other phase kinds.
_Avoid_: legacy Run Plan, cross-phase optional-field schema, unversioned phase config

**Phase RunSpec**:
The versioned concrete execution contract materialized from a Phase Plan for one Phase Attempt beneath a Phase Run. It combines common lifecycle authority with exactly one phase-kind-specific execution payload; legacy conversion is explicit rather than implicit loader behavior.
_Avoid_: legacy RunSpec, generic configuration bag, implicit schema migration

**Phase Status**:
A read-only observation of a Phase Run derived from durable authority and current scheduler accounting. It never writes lifecycle state or dispatches Runtime Actions.
_Avoid_: reconciliation command, progress command, implicit resume

**Phase Resume**:
An idempotent Phase Lifecycle transition that reconciles the current Phase Attempt and dispatches only newly eligible, bounded recovery or top-up Runtime Actions from its frozen declared action set. It does not discover or dynamically grow the graph, cross a failed or cancelled outcome, resubmit durably submitted work, or create a new Phase Attempt.
_Avoid_: status, automatic restart, unbounded retry loop

**Exact Scheduler Endpoint**:
The scalar parent or array `parent_task` identity derived from a frozen Runtime Action and its durable numeric Slurm parent assignment. Slurm's explicit nested JSON array parent/task metadata identifies each child even when its physical numeric allocation ID differs. Exact terminal fallback binds that numeric `JobIDRaw` to its paired logical `JobID` under the requested parent, preserving matching step suffixes and rejecting conflicting bindings. It never derives a child from a parent-only record, and an unset JSON signal still requires exact accounting fallback rather than an assumed zero.
_Avoid_: inferred array child, accounting range, step identity, scheduler-discovered task

**Phase Retry**:
The explicit Phase Lifecycle transition that reopens a failed or cancelled Phase Run by operationally rematerializing its next immutable Phase Attempt without changing the Phase Plan or input Artifact Set identities. It preserves every earlier attempt, is never implied by Phase Resume, and is unavailable after a validated Phase Receipt seals the Phase Run.
_Avoid_: resume flag, in-place restart, new Phase Run

**Phase Cancellation**:
An idempotent Phase Lifecycle transition that durably records cancellation intent, prevents further dispatch, and requests cancellation of every non-terminal scheduled job. The Phase Run becomes cancelled only after scheduler accounting confirms terminal jobs; outputs and evidence are preserved.
_Avoid_: immediate logical cancellation, cleanup operation, evidence deletion

**Control Preflight**:
The lightweight validation performed by the Control Plane before submission. It checks run shape, cluster access, and scheduling readiness without BSPP runtime dependencies or payload inspection.
_Avoid_: runtime validation, acceptance check

**Artifact Set**:
A content-identified phase input or output whose root manifest references deterministic per-chunk manifests. Each chunk manifest inventories and hashes its exact logical members; physical locations are recorded separately and never affect Artifact Set identity.
_Avoid_: monolithic member manifest, compressed-file-only identity, location-derived identity

**Artifact Location**:
A separately governed physical replica of an Artifact Set, including its scope, availability, and verification state. A verified location accessible from the target cluster is required for Phase Submission, but it does not determine Phase Receipt validity or logical progression eligibility.
_Avoid_: Artifact Set identity, receipt-embedded URI, storage listing discovery

**Data Placement**:
The declared location, layout, and movement state of BSPP inputs, baselines, and outputs for a target run or cluster. It may materialize a verified target layout without changing content identity; the Control Plane may plan, trigger, or verify Data Placement but never proxies large payload bytes through the local workstation.
_Avoid_: laptop transfer, implicit data availability

**Database Access Policy**:
The preprocessing choice among requiring a node-local database replica (`stage-required`), preferring one with an allowed direct-access fallback (`stage-preferred`), or using the source database in place (`direct`). The Phase Plan requests the policy, the Cluster Profile constrains the policies a site permits, and `stage-required` is both the Phase Plan default and the only policy permitted by an omitted profile allowlist; fallback is an explicit policy choice rather than an implicit recovery behavior.
_Avoid_: staging boolean, no-stage, automatic fallback

**Database Capacity Gate**:
The recorded pre-copy decision that determines whether a cold Database Replica may be allocated by comparing node-local user-available capacity with the destination-allocated replica requirement plus an explicit Cluster Profile-owned reserved-byte floor. Every staging-capable profile must declare that floor as an absolute byte value; `0` is valid when deliberate, while omission fails profile validation before submission. A valid warm replica requires no capacity decision. The Phase Plan cannot weaken the reserve; only an insufficient cold result may activate `stage-preferred` direct access, while errors after staging begins remain failures.
_Avoid_: disk-full recovery, staging-error fallback

**Database Source Manifest**:
The immutable document and complete logical-member inventory describing the MMseqs database version used by a preprocessing Phase Run. Every Database Access Policy references the same manifest so that placement choices retain the same declared input; its lightweight member evidence does not claim cryptographic payload integrity.
_Avoid_: database directory scan, unversioned database path, staging file list, payload content hash

**Database Set**:
The stable logical identifier and version selected by a preprocessing Phase Plan for its related primary and metagenomic MMseqs inputs. It is the sole authority for those database roles and logical names; the Cluster Profile resolves it to a site-specific Database Source Manifest, while the immutable Phase RunSpec pins the verified manifest document and names.
_Avoid_: Lustre database path, mutable profile mapping, independently selected database directory, duplicate database-name fields

**Database Set Provisioning**:
The explicit cluster-side operation that resolves a declared Database Set source into an atomically published Database Source Manifest without reading or hashing the large payload bodies. Source changes require provisioning a new Database Set version; Phase Materialization and Runtime never refresh a manifest implicitly.
_Avoid_: hand-authored file list, runtime inventory discovery, mutable manifest refresh

**Database Replica**:
A self-contained node-local realization of one Database Source Manifest. It stores each unique resolved source once, supplies every required logical MMseqs member as a regular file using replica-internal hard-link aliases, and has no dependency on the source location after publication.
_Avoid_: copied symlink farm, partial database copy, Lustre-backed local view

**Database Replica Manifest**:
The immutable publication record binding a complete Database Replica to its Database Source Manifest, exact logical topology, source-stability observations, copy outcome, and metadata-verified member inventory. It supports fast cache reuse but deliberately does not prove whole-payload cryptographic integrity.
_Avoid_: payload checksum proof, cache directory listing, completion marker alone

**Database Replica Population**:
The identity-then-cache exclusive-lock operation that copies a Database Set into a same-filesystem temporary directory, validates the complete replica, and publishes it by atomic rename. A caught failure or cancellation removes only its own temporary population. A process-death orphan remains hidden and is never reusable. A later genuinely fresh owner may remove only exact same-identity, exact-grammar orphan names under both exclusive locks before starting over; a failed waited contender and a valid warm reader never clean. The contract does not resume partial copies. Cross-identity operator maintenance is only the explicit acceptance-cache operation and is never an ordinary Runtime Action.
_Avoid_: in-place cache population, visible partial replica, cross-job partial-copy resume

**Database Replica Cache**:
The Cluster Profile-located, node-local, manifest-addressed collection of complete Database Replicas available for reuse by the same Unix user across jobs. The profile pins its expected filesystem type and a positive replica-lock wait timeout that fits within the Runtime Action's total Slurm wall time. A cold attempt consumes that value as one absolute monotonic deadline across identity-exclusive and then cache-exclusive acquisition. Same-identity contenders revalidate and reuse a valid winner as a fresh action-bound warm Result; a contender whose owner releases without a valid final fails hard and performs no cleanup or population. Different identities may hold their identity locks concurrently but serialize repair, scoped cleanup, fresh capacity measurement, copy, publication, and cold Result evidence under the cache lock. Warm hits remain nonblocking shared-lease reads. After all shared probe descriptors close and readers drain, a visible invalid replica is revalidated, atomically retired, cleaned only within its exact identity, and repopulated under continuous identity-then-cache exclusive ownership. Timeout is a hard `lock-unavailable` failure rather than capacity fallback.
_Avoid_: job-private database copy, mutable shared database directory, assumed warm cache

**Database Acceptance Cache Namespace**:
A cache root isolated through a dedicated Cluster Profile marked exactly `database_cache_namespace: acceptance` and used only for repeatable live validation of cold Database Placement. The explicit cluster-side maintenance command derives the sole target from that profile plus the effective passwd user, physically binds and rechecks every configured sibling root, takes cache-exclusive ownership, tries every relevant identity-exclusive lock without waiting, stable-rescans, and removes only exact top-level replica or population names. It publishes bounded immutable terminal evidence without payload inventory: complete removal and lock-observation sequences are digest-bound while samples remain capped. Phase Plans cannot select, request, or override maintenance; ordinary Runtime never evicts valid replicas, and sibling-user, sibling-profile, and production cache entries are unaffected.
_Avoid_: production-cache eviction for testing, Phase Plan cache override, assumed cold node

**Database Replica Lease**:
The persistent per-manifest protocol file and advisory shared lock used for a Database Replica. Warm placement opens the existing file read-only and holds `LOCK_SH` through complete validation and strict warm Result publication. After placement exits, scientific execution acquires a fresh shared lease, revalidates the selected replica, and holds it until every started Scientific Kernel process is terminal. Warm readers never create, truncate, replace, remove, or upgrade the protocol file. An invalid warm probe closes all shared and candidate descriptors before the existing exclusive coordinator waits for every reader to drain and revalidates the exact identity for repair.
_Avoid_: placement-only lock, cache ownership marker, permanent reservation

**Database Placement**:
The preprocessing Runtime Action work that validates or materializes the selected database location on its allocated node before invoking the Scientific Kernel. It shares the scientific action's Slurm allocation because a node-local Database Replica cannot be handed off through an ordinary dependency job, and it exposes exactly one selected read-only database location to the Scientific Kernel while retaining cache write access only within placement.
_Avoid_: database staging job, login-node copy, cross-job node-local handoff

**Database Placement Result**:
The immutable Runtime evidence selecting the exact staged or direct database branch predeclared by a preprocessing Phase RunSpec. Its outcome is one of `replica-cold`, `replica-warm`, `direct-requested`, or `direct-capacity-fallback`. Cold and warm replica Results use distinct strict envelopes and bind different placement facts while sharing one closed staged Result union; their subsequent lease evidence must use the matching digest family. A direct result under `stage-preferred` is valid only when its Database Capacity Gate records insufficient capacity, and every direct result binds matching pre- and post-kernel source-metadata observations. A policy-authorized direct result is ordinarily receipt-eligible rather than a degraded lifecycle state. The Phase Receipt binds the requested policy and source-manifest digest, summarizes each action's outcome, and references the detailed placement evidence rather than duplicating it.
_Avoid_: rewritten RunSpec, dynamic database path, implicit fallback

**Database Placement Failure Evidence**:
The bounded, durable policy-specific Runtime evidence published before any Result when Database Placement cannot select a valid branch. It records the attempted policy and source-manifest identity, relevant node/cache and capacity facts, and a classified failure without creating a successful Database Placement Result. If an exact Result is already visible and the placement command later exits nonzero, Runtime does not publish contradictory policy Failure evidence; action-level Database Placement Command Failure Evidence retains the exact Result instead. Action execution records either pre-science failure as a payload-free failed action with `science_started: false` before the Scientific Kernel can be invoked.
_Avoid_: partial placement result, missing pre-science failure record, inferred failure from logs alone

**Database Placement Command Failure Evidence**:
The bounded, durable action-level state produced by strict reconciliation of the exact captured Database Placement process status with its immutable Result/Failure files. Every action record requires that non-boolean status as top-level `placement_process_status`; a failed selector supplies its own exact nonzero status to the no-payload reconciliation invocation. `nonzero-after-result` retains one exact, digest-bound Result while prohibiting science; `evidence-reconciliation-failed` trusts only the RunSpec/action/database identity and carries no unauthorized Result when files are absent, malformed, contradictory, wrong-family, or mismatched. Both classifications require payload-free failed action evidence, exclusive read-only publication, strict reload, and no carry-forward adoption, payload access, lease acquisition, Scientific Kernel, output, finalization, or handoff.
_Avoid_: contradictory placement failure, selecting science after nonzero placement, log-only command failure

**A3M Materialization**:
The first Run Data Placement Runtime Action in a Folding Phase Attempt, reconstructing and verifying loose A3M files from a logical MSA Artifact Set without changing its content identity. Folding scientific Runtime Actions depend on its verified Artifact Location; neither phase's Scientific Backend Adapter owns it.
_Avoid_: preprocessing finalization, folding input discovery, archive identity change

**Provisioning Data Placement**:
Data Placement work needed to prepare orchestration infrastructure, small run artifacts, source bundles, or runtime images before workflow execution.
_Avoid_: payload transfer, publication

**Run Data Placement**:
Data Placement work that moves or materializes BSPP inputs, baselines, outputs, or publication artifacts and therefore belongs in the run workflow with evidence. Run Plans/RunSpecs declare selected movers such as `dm`, `s5cmd`, or `gcloud`; scheduled workflow jobs write per-stage evidence for the source, destination, mover, command, job id, and terminal result.
_Avoid_: provisioning, hidden transfer

**Data Mover**:
A configured mechanism for moving BSPP data between object storage and cluster storage. `dm` can be selected as a Data Mover where available, but it is not assumed by the domain model.
_Avoid_: hardcoded dm dependency, laptop transfer

**Cluster Profile**:
The Control Plane configuration for one target cluster, including how to connect to it and how to resolve cluster-specific paths and scheduling conventions. It is used to materialize concrete RunSpecs, not to replace them.
It uses a simple stable id and avoids metadata unless that metadata has a specific operational use. Cluster evidence should use the profile identity and transport, not login-node hostname matching.
_Avoid_: RunSpec, bundled clusters.yaml

**Cluster Profile Template**:
The repo-provided shared cluster convention layer used to resolve a user-local Cluster Profile. It excludes user-specific roots, credentials, and secret values.
_Avoid_: user profile, RunSpec

**Execution Runtime**:
The cluster-side runtime context for BSPP payload work and acceptance gates. It is entered only through scheduled cluster jobs.
_Avoid_: host runtime, controller runtime

**Runtime Action**:
A bounded unit of phase work performed inside the Execution Runtime by one scheduled job. It produces declared outputs and evidence but never schedules another Runtime Action.
_Avoid_: workflow, phase, controller job

**Runtime Qualification**:
Reusable evidence for a particular cluster setup and Execution Runtime tuple. It applies to a concrete tuple of cluster profile, scheduling class, runtime image, planned orchestration Source Bundle identity/path, toolkit source, and relevant GPU/runtime facts; it is reused across production runs until that tuple changes or the qualification expires.
It is created or refreshed by a separate runtime qualification command that submits a scheduled smoke job inside the Execution Runtime; only a smoke-written qualified record satisfies canary or production submission policy when relevant.
_Avoid_: per-run preflight, acceptance

**Execution Runtime Image**:
The container image selected for an BSPP run's Execution Runtime. The concrete RunSpec records the cluster-local image artifact used by scheduled jobs.
_Avoid_: controller image, local development environment

**Preprocessing Execution Runtime Image**:
The BSPP Orchestration-owned reproducible image definition for one preprocessing Scientific Backend Adapter. It pins the orchestration Runtime and Scientific Kernel dependencies, excludes the reference MSA pipeline source, and is resolved to a content-bound, scheduled-smoke-qualified cluster artifact through the Cluster Profile. Phase Materialization embeds that exact qualified selection and fails closed instead of falling back to the generic image.
_Avoid_: its image, site-owned scientific image, mounted scientific environment

**Cluster Image Cache**:
The target cluster storage area where Execution Runtime Images are made available in the format required by the cluster runtime.
_Avoid_: Docker build cache, registry

**Login Node Probe**:
A small non-payload operation that the Control Plane may perform on a cluster login node to prepare, submit, or observe scheduled work. It must not require BSPP runtime dependencies or inspect payload data.
_Avoid_: login-node execution, host Python run

**Remote Slurm Transport**:
The Control Plane boundary used to submit jobs, monitor accounting, and move small orchestration artifacts on a target Slurm cluster without running payload work locally.
It submits and observes each workflow job directly instead of starting a cluster-side orchestrator job.
_Avoid_: submitit over SSH, Slurm REST transport, login-node execution runtime

**Run Plan**:
The user-authored description of run intent before cluster-specific resolution. It targets one cluster and combines with one Cluster Profile to materialize one concrete RunSpec.
Expansion prints the concrete RunSpec for inspection without provisioning or contacting the cluster.
_Avoid_: RunSpec, recipe

**Run Kind**:
The Run Plan policy class for a run: `dev`, `canary`, or `production`. It controls lightweight safety policy such as dirty source, production prefixes, runtime qualification, and confirmation.
_Avoid_: free-form environment, deployment stage metadata

**Run Index**:
A compact query surface for locating run provenance, status, acceptance, publication, and evidence records. It indexes RunSpecs and evidence; it does not own run configuration.
_Avoid_: run registry, source of truth

**RunSpec**:
The concrete contract for one BSPP run, including dataset, target cluster, paths, workflow, runtime, acceptance, and evidence expectations.
_Avoid_: recipe, ad hoc config

**Source Bundle**:
The versioned orchestration source staged by the Control Plane for use by cluster jobs. It is distinct from a developer checkout and is referenced by the concrete RunSpec.
_Avoid_: cluster checkout, cloned repository

**Toolkit Source**:
The AFDB toolkit code selected for an BSPP run while the toolkit remains outside the Execution Runtime image. It is mounted into cluster jobs and recorded in the concrete RunSpec.
_Avoid_: legacy repo, implicit sibling checkout

**Recipe**:
A legacy execution artifact rendered from a RunSpec for compatibility with older pipeline components.
_Avoid_: source of truth, RunSpec

## Example Dialogue

Developer: "Can the Control Plane validate this RunSpec from my laptop?"

Domain expert: "Yes, as long as payload processing stays in the Execution Runtime and the RunSpec remains the source of truth."

Developer: "Can I change clusters by changing only the Cluster Profile?"

Domain expert: "You can materialize a new concrete RunSpec from a different Cluster Profile, but the submitted RunSpec must still contain the resolved cluster paths and runtime choices."

Developer: "Can one Run Plan target multiple clusters?"

Domain expert: "No. Use one Run Plan per cluster; cross-cluster comparison is a separate workflow, not a common production run shape."

Developer: "Should I put the Slurm partition in the Run Plan?"

Domain expert: "Only if this run needs to override the Cluster Profile; otherwise the Run Plan should describe intent and let materialization resolve cluster-specific defaults."

Developer: "Where do I query which runs passed acceptance?"

Domain expert: "Use the Run Index when that catalog surface exists; until then, query the concrete RunSpec evidence directories and reports."

Developer: "Can I resume a run if my laptop disconnects?"

Domain expert: "Yes, if the Control Plane wrote Control State and job evidence into the run evidence directory before disconnecting."

Developer: "Does every cluster need a cloned orchestration repository?"

Domain expert: "No. The Control Plane should stage a Source Bundle for the run, and the concrete RunSpec should point jobs at that staged source."

Developer: "Can the Toolkit Source stay mounted while orchestration source is bundled?"

Domain expert: "Yes. The Toolkit Source is still selected and recorded explicitly until it becomes stable enough to bake into the Execution Runtime."

Developer: "Who makes sure the cluster has the right Execution Runtime Image?"

Domain expert: "The Control Plane checks or plans the Cluster Image Cache; image provisioning is Provisioning Data Placement and must complete before scheduled runtime work depends on that image."

Developer: "Does the Remote Slurm Transport process model files?"

Domain expert: "No. It submits and observes scheduled work; processing happens in the Execution Runtime."

Developer: "Can the Control Plane run a quick command on a login node?"

Domain expert: "Yes, when it is a Login Node Probe such as checking a path or submitting a job, not when it executes BSPP runtime code or reads payload data."

Developer: "Do we need a runtime smoke test before every production run?"

Domain expert: "No. Runtime Qualification is required for a new or changed setup, then reused until that setup changes or the qualification expires."

Developer: "Is Runtime Qualification only part of run submission?"

Domain expert: "No. It has a separate qualification command, and run submission checks whether current qualification exists when the Run Kind requires it."

Developer: "Can the Control Plane download all archives to my laptop before submitting?"

Domain expert: "No. It should coordinate Data Placement using cluster-side or object-store movement mechanisms, not proxy large BSPP payloads locally."

Developer: "Is image staging the same kind of data movement as publishing outputs?"

Domain expert: "No. Image staging is Provisioning Data Placement; moving BSPP inputs or publication artifacts is Run Data Placement and should have workflow evidence."

Developer: "Is `dm` always required for Data Placement?"

Domain expert: "No. `dm` can be configured as the Data Mover for a cluster or run, but other cluster-side movement mechanisms remain valid."

Developer: "Should we edit the Recipe to change the archive scope?"

Domain expert: "No. Change the RunSpec, then render a new Recipe if legacy compatibility still needs one."


### Artifact-backed folding evidence

Packed BioIR can explicitly select `artifact-backed-v2` with the sealed model
policy. Runtime validates original full score files one target at a time and
publishes four bounded metadata members followed by their finalization index.
Control's existing fetch/finalize lifecycle validates those commitments and
retains the existing receipt schema. The profile changes operational evidence
identity only; legacy omission, scientific settings and original PAE files stay
unchanged. Independent original-file audits remain distinct from Control's
acceptance of Runtime attestations. V2 Retry consumes native/adopted journals
without duplicate adoption, and only a fully carried skipped worker action may
be adopted by canonicalization itself.

### HumanSTRING Folding Identity

The exact `homo_<accession>` and `hetero_<accession>_<accession>` roots identify
original HumanSTRING dimers. Folding and legacy-MSA import preserve
their names, accession order and sequence identity. HumanSTRING preprocessing
uses the explicit `require_afdb_model_id_stem: false` setting. The legacy AF
homodimer/compound predicates retain their original meaning; dedicated
HumanSTRING helpers classify the new roots. Equal sequences from distinct
accessions do not rename a hetero target. The frozen postprocessing discovery
boundary rejects HumanSTRING explicitly instead of silently dropping a mixed
archive subset. Accepted folding plus file-integrity evidence is not a
reference-structure accuracy validation.
