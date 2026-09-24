# Next Container Validation

This checkpoint follows the successful Example cluster A100 smoke tests for both
container variants:

- `latest`: `containers/smoke-results/latest-9584436-0.json`
- `current`: `containers/smoke-results/current-9584878-0.json`

Both smoke artifacts show native `sm_80` support, CUDA `radius_graph` success,
and no AFDB fallback.

## Planned Tests

1. Render isolated one-archive run specs for `current` and `latest` with
   distinct output directories and SwiftStack prefixes.
2. Run `latest` first on archive `0-0` with `batch_size: 1000`.
3. Capture total runtime, Stage 13 runtime, upload count, and final status in
   `containers/POSTPROCESSING_PIPELINE_CONTAINER_VALIDATION.md`.
4. Rerender `latest` into a fresh output prefix with the baseline provider
   copyright text and conservative `s5cmd_numworkers`, then repeat strict byte
   parity against the historical baseline.
5. Run the matching `current` one-archive baseline if timing/parity needs a
   direct local comparison rather than the existing historical run.
6. Compare output object counts and deterministic metadata/model artifacts
   between `current` and `latest`.
7. Summarize timing and parity results in `SM_ACCEPTANCE.md` after the detailed
   pilot record is complete.
8. Only after parity and timing are acceptable, plan the full archive array
   with an explicit concurrency throttle.

## Notes

- Do not treat the RTX 5090 Docker builder as production validation.
- Do not rely on Docker `ENTRYPOINT` behavior under Pyxis; wrappers call the
  container entrypoint explicitly.
- Submit wrapper scripts from a context where `BSPP_ORCH` is set, or from the
  repo root, so `SLURM_SUBMIT_DIR` is useful if the environment is not loaded.
- Strict byte parity currently depends on regenerated RunSpec artifacts, because
  the provider copyright text is rendered before the legacy toolkit runs.
- The historical baseline is
  `s3://example-bucket/users/example-user/postprocessed-test/20251128_full_run_v2/`.
- Keep `s5cmd_numworkers` below the legacy default of 256 while upload remains
  on the GPU worker critical path.
