# Postprocessing Reference Example

This directory is a schema-complete operator example.
The Phase Plan is the preferred entry point;
the legacy Run Plan is pinned as historical input only.

| File | Purpose |
|---|---|
| `profiles.yaml` | User-local Cluster Profile example for Example cluster. |
| `workflow.yaml` | Workflow template with preflight, rendering, worker, finalizer, and acceptance steps. |
| `run-plan.yaml` | One-archive task-853 Run Plan shape example. |
| `postprocessing-phase-plan.yaml` | Phase Plan that pins the exact bytes of the four authority inputs. |
| `acceptance-policy.yaml` | Bounded acceptance residual and report-reconciliation policy shape. |
| `logical-input-inventory.yaml` | Expected identities, SHA-256 values, and sizes for remote scientific inputs. |
| `runtime-qualification.json` | Non-runnable qualification shape illustration; strict materialization deliberately rejects it. |

The paths and object-store prefixes are real task-853 example values. The
qualification identities, expiry window, and logical-input hashes/sizes are
illustrative fixtures, not production claims. The acceptance policy is the
maintained fixed-baseline decision described in the acceptance runbook.
This directory is nevertheless intentionally non-runnable: the checked-in
qualification is not an authentic promoted producer record, and strict Phase
materialization rejects it. Before materializing a new run, copy the directory,
replace `runtime-qualification.json` with the current canonical record emitted
by Runtime Qualification, and replace the other illustrative values with
independently established evidence. Update the exact size and SHA-256 references
in `postprocessing-phase-plan.yaml` after changing any referenced document.

Never widen `acceptance-policy.yaml` merely to make the current outputs pass.
Its residuals and cardinalities must be established against the approved
baseline before submission. The Phase Lifecycle preserves the legacy Run Plan
as a pinned projection input; the `run`/`plan`/`provision` commands are removed
and this example is phase-only.

After replacing the illustrative records, prefer the V3 restart-safe Phase
operator. It preserves the authored
`run-plan.yaml` `dataset.name` as the tracking selector and derives a
fresh attempt-scoped `dataset.run_id` and all operational paths. Give it a
dedicated authority root and a separate dedicated execution root outside the
clean source checkout. The execution root will contain only the create-once
operation intent, atomic current checkpoint, lock, derived handoff and
scheduler evidence, plus warning-only diagnostics.

The checked-in local tar and analysis destinations are deliberately beneath
`profiles.yaml` `paths.output_root` plus the authored `dataset.run_id` from
`run-plan.yaml`. V3 keeps their relative names (`local_tars`, `local_tars.csv`,
and the analysis files) but places them beneath the immutable Attempt output
root. The rendered recipe, finalizer, parity check, and semantic acceptance
therefore all consume the same Attempt-owned tree. When adapting this example,
keep every local output under that authored base; split roots, relative paths,
and traversal are rejected before authority is published.

The example Cluster Profile also declares both
`postprocessing_credential_mounts` source paths required by its `aws:`
SwiftStack credential reference. Materialization records only these locator
strings and renderer 5 mounts the two files read-only; the example does not
contain or authorize copying credential bytes. Current renderer 5 retains those
frozen mounts and, for the array, forwards the Slurm array parent job ID
through the unchanged governed bootstrap so success evidence uses
`<parent>_853`; it does not require a bootstrap or runtime-image rebuild.

The example does not authorize repairing historical authority. A V2 Phase Run
can pass first submission or explicit retry only when Control proves its stored
`dataset.name` matches the exact pinned Run Plan bytes. A broken or
unverifiable V2 projection must be materialized as a new V3 Phase Run.
