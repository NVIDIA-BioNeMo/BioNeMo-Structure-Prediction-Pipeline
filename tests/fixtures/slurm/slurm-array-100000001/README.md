# Synthetic failed array accounting fixture

Fully synthetic fixture modeling a genuine failed Slurm array capture, using
neutral example labels and obviously synthetic job identities and timestamps
(parent array `100000001`; `JobIDRaw` 100000002 maps to the `100000001_0`
element, while physical job 100000001 is the logical `100000001_1` element,
not an additional parent row).

Both elements failed; this fixture is not successful benchmark or scientific
evidence. Field structure mirrors `sacct --json` / `sacct --parsable2` output:
JSON keys, ordering, states, exits, and resource fields are internally
consistent.

Commands the fixture models (with `TZ=UTC`):

```text
sacct --json -j 100000001 --format=JobIDRaw,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem
sacct -j 100000001 --format=JobIDRaw,JobID,State,ExitCode --noheader --parsable2
```

The same commands with `,Restarts` first return status 1 and the preserved
stderr. JSON leaves signal unset; do not infer signal 0 or reconstruct a
parsable exit from JSON return 143. Exact text accounting reports `15:0`.

Tests that modify these records explicitly construct synthetic unit cases;
they do not claim those variants are genuine successful results.
