# Cluster configuration

A Cluster Profile supplies deployment paths, transport, images, and scheduler
resources. A Phase Plan supplies the requested computation. Start with the
[commented profile template](../../skills/examples/run-plans/cluster-profile.yaml)
and the matching [preprocessing](../../skills/examples/run-plans/preprocessing-phase-plan.yaml),
[folding](../../skills/examples/run-plans/folding-phase-plan.yaml), or
[postprocessing](../../skills/examples/run-plans/postprocessing-phase-plan.yaml)
Plan. These templates contain illustrative values and are intentionally not
ready to submit.

## Select the profile and transport

Profiles live under `clusters:`, keyed by the identifier used by the Plan's
`target_cluster`. Select the file explicitly:

```bash
bsppctl --config /path/to/profiles.yaml phase materialize \
  /path/to/phase-plan.yaml --authority-root /path/to/authority \
  --source-repo /path/to/clean-checkout
```

Materialization creates durable authority but does not submit jobs. Configuration
lookup uses `--config`, then `BSPPCTL_CONFIG`, then
`~/.config/bsppctl/profiles.yaml`. Author complete site values: the loader does
not expand environment variables inside YAML or inherit missing fields from
another profile. Use nested `paths` or their flat equivalents, not both.

| Transport | Where Control runs | Profile requirements |
| --- | --- | --- |
| `ssh` | Operator machine | Set `ssh_target`; it must reach the cluster's Slurm tools without an interactive prompt during commands. |
| `local-slurm` | Site host with Slurm commands | Omit `ssh_target`; use a clean source checkout and a native Control environment. |

Native Control can reduce submission latency for dependent jobs. It retains the
same public lifecycle and authority model. Choose the transport before
materialization; changing a YAML file does not rewrite an existing immutable
RunSpec. See [dependency recovery](troubleshooting.md#dependent-submission-and-slow-transport)
for partial submissions.

## Place data and images

`project_root`, `output_root`, `staging_root`, `orchestration_repo`, and image
paths refer to the cluster filesystem. They must be visible to the relevant
allocated nodes. With SSH, `runtime_qualification_root` is cluster-side while
`runtime_qualification_control_root` is a separate operator-side mirror. With
`local-slurm`, the latter defaults to the former.

For folding, `paths.image` runs non-fold actions such as MSA preparation and
canonicalization. `folding_backend_images` selects the scientific image for
each backend. An omitted override falls back to `paths.image`; do not assume the
small folding Runtime image contains BioIR, ColabFold, or OpenFold. Configure
the matching external assets in `folding_backend_assets`. Every asset path must
be covered by an explicitly read-only `extra_mounts` target.

Use shared storage for inputs, accepted outputs, and evidence needed by later
jobs. Size storage for compressed archives, expanded MSAs, scratch files, model
weights, images, and outputs together. A quota is different from filesystem
free space. Node-local caches and import temporary files can reduce shared
storage traffic, but must satisfy the site's ownership, lifetime, and placement
rules.

Each `extra_mounts` source must already exist on the execution node. The renderer
does not create arbitrary source directories. In particular, a directory made
under `/dev/shm` by one job may disappear before another job starts. A
node-local mount is usable only on nodes where that exact path exists; test its
lifetime and the actual container mount before relying on it across jobs.

For preprocessing, `database_sets` maps logical database identities to real
manifests. `database_access_policies: [direct]` accesses the selected database
without cache staging. `stage-required` and `stage-preferred` require the full
five-field cache configuration in the template. These policies still verify
manifest, filesystem, capacity, and ownership constraints; a staging preference
is not permission to use an arbitrary unverified database copy.

## Describe the GPU allocation

The shipped execution path uses Slurm GPU allocation with Pyxis/Enroot. Confirm
the site's partition, account, GPU request syntax, container support, and node
architecture. The supplied images target Linux x86-64. The profile does not
provide an exclusive-node substitute for a site without GPU resource accounting.

A scalar GPU worker uses a request such as `gres: gpu:1`. Packed folding uses
typed topology instead. For example, replace the scalar `gpu_worker` block with
site-appropriate values following this shape:

```yaml
gpu_worker:
  partition: gpu-short
  cpus_per_task: 8
  memory: 128G
  time: "04:00:00"
  nodes: 2
  tasks_per_node: 8
  gpus_per_task: 1
  max_parallel: 2
```

Omit `gres` and legacy `array` when using these typed fields. Packed folding
requires one GPU per task. This example requests 16 workers as an array of two
elements, each with one node, eight tasks, and eight GPUs. CPU counts are per
task (64 CPUs per node here); memory is per node.

`nodes` is the number of array elements, not an atomic multi-node allocation.
`max_parallel` limits concurrent elements; it does not guarantee simultaneous
starts or exclusive ownership of a node. Measure actual overlap from scheduler
records when reporting a parallel benchmark. Choose a homogeneous GPU pool when
comparing hardware, and retain the actual node and GPU identities.

## Bind deployment evidence

Use real image hashes, source identities, asset hashes, and database manifests.
The template's repeated-character digests and version strings are placeholders,
not valid deployment evidence. Preprocessing's `preprocessing_runtime` block
must be authored as a complete set of thirteen fields from the actual build,
import, and source-bundle records. Its public qualification commands are shown
in [Containers](containers.md#qualification-and-scientific-readiness).

Keep registry and object-store secrets outside profiles and Plans. Credential
mount fields identify external files; they do not contain credential bytes.
Preserve the authority and fetched evidence for every attempt, including failed
or cancelled attempts, when changing operational configuration.
