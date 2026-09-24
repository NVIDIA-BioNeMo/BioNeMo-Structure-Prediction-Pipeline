# Developing bspp-orchestration

This document describes how to set up and develop `bspp-orchestration`.

## Project identity

`bspp-orchestration` is a control surface for orchestrating BSPP
high-throughput pipelines across preprocessing, folding, and postprocessing
phases. It is not an official BSPP production interface. This project is
currently not accepting external contributions.

Internal operator handoffs, decision records, and historical evidence are not
part of the public contribution surface. Public-facing changes should target
the README, the public `docs/` tree, and the package source; do not add
internal host, cluster, credential, or storage identities to public files.

## Setup

The repository is a uv workspace with three installable distributions:
`bspp-orchestration-contract`, `bspp-orchestration-control`, and
`bspp-orchestration-runtime`. Python 3.12 is required.

```bash
uv sync --all-packages --group dev
```

## Build and test

```bash
pytest
pytest --cov=bspp.orchestration tests
ruff check packages tests containers
ruff format --check packages tests
mypy packages/orchestration-contract/src packages/orchestration-control/src packages/orchestration-runtime/src
```

## Code style

- Python 3.12.
- `from __future__ import annotations` in every module.
- Dataclasses for value objects.
- Full type annotations (mypy strict).
- Modules should be focused and small.
- Ruff line-length 120, target py312, rules `E`, `F`, `I`, `N`, `W`, `UP`,
  `B`, `SIM`, `RUF`.
- No hardcoded secrets or credentials.
- Test our logic, not third-party behavior. Mock only at external boundaries
  (object-storage clients, network); prefer real objects when cheap.

## Conventions

- Keep the public documentation surface (`README.md`, `docs/`) free of internal
  identities, hosts, clusters, and storage labels.
- Do not reference internal operator or decision material from public files.
- Run the quality gates before opening a change.
