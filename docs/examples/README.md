# Example catalog

Use one canonical set of commented YAML templates when authoring Plans and
Profiles. The templates below are checked by repository schema tests. They
contain illustrative values and do not submit runnable work until an operator
supplies genuine paths, identities, assets and qualification records.

| Document | Purpose |
| --- | --- |
| [Cluster Profile](../../skills/examples/run-plans/cluster-profile.yaml) | Shared site configuration with commented field and enum alternatives |
| [Preprocessing Phase Plan](../../skills/examples/run-plans/preprocessing-phase-plan.yaml) | Input identity, chunking, database placement and preprocessing execution |
| [Folding Phase Plan](../../skills/examples/run-plans/folding-phase-plan.yaml) | Verified MSA input, backend policy, packed workers and output transport |
| [Postprocessing Phase Plan](../../skills/examples/run-plans/postprocessing-phase-plan.yaml) | Qualified worker recipe, acceptance policy and predecessor inventory |
| [Postprocessing companion files](../../skills/examples/run-plans/postprocessing/) | Linked illustrative recipe, qualification, inventory and acceptance policy |
| [Folding transport examples](folding-benchmark/README.md) | Separate local and remote MSA-location shapes with illustrative identities |

For the command sequence, read the [phase lifecycle](../guides/phase-lifecycle.md).
For BioIR benchmark preparation and execution, read the
[benchmark walkthrough](../benchmarks/bioir-workflow.md). The
[configuration reference](../reference/configuration.md) explains how authored
values become an immutable RunSpec.
