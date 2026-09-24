# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Control Plane CLI entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import click
import yaml
from pydantic import ValidationError


@click.group(name="bsppctl", help="BSPP Control Plane.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None) -> None:
    """BSPP Control Plane."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@cli.group("phase")
def phase_group() -> None:
    """Phase Lifecycle operations."""


@phase_group.group("evidence")
def phase_evidence_group() -> None:
    """Bounded Phase evidence transfer and export operations."""


@phase_evidence_group.command("fetch")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing the durable Phase Run authority.",
)
@click.option(
    "--destination",
    required=True,
    type=click.Path(path_type=Path),
    help="New local directory for the bounded Runtime finalization bundle.",
)
def phase_evidence_fetch_cmd(phase_run_id: str, *, authority_root: Path, destination: Path) -> None:
    """Fetch indexed postprocessing or artifact-backed folding finalization metadata."""
    from bspp.orchestration.control.postprocessing_evidence_transfer import (
        fetch_postprocessing_finalization_evidence,
    )

    try:
        from bspp.orchestration.control.phase_adapters import phase_authority_family

        if phase_authority_family(authority_root, phase_run_id) == "folding":
            from bspp.orchestration.control.folding_artifact_evidence import fetch_folding_finalization_evidence

            payload = fetch_folding_finalization_evidence(
                phase_run_id, authority_root=authority_root, destination=destination
            )
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
            return
        result = fetch_postprocessing_finalization_evidence(
            phase_run_id,
            authority_root=authority_root,
            destination=destination,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)


@phase_evidence_group.command("export-scheduler")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing the durable Phase Run authority.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Create-once local scheduler evidence JSON path.",
)
def phase_evidence_export_scheduler_cmd(phase_run_id: str, *, authority_root: Path, output: Path) -> None:
    """Export successful scheduler evidence from durable postprocessing events."""
    from bspp.orchestration.control.postprocessing_scheduler_evidence import (
        export_postprocessing_scheduler_evidence,
    )

    try:
        result = export_postprocessing_scheduler_evidence(
            phase_run_id,
            authority_root=authority_root,
            output=output,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)


@phase_group.command("materialize")
@click.argument("phase_plan", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Caller-selected root for durable Phase Run authority.",
)
@click.option(
    "--source-repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Clean committed orchestration source represented by qualification.",
)
@click.pass_context
def phase_materialize_cmd(
    ctx: click.Context,
    phase_plan: Path,
    *,
    authority_root: Path,
    source_repo: Path | None,
) -> None:
    """Materialize one scheduler-free Phase Run from a typed Phase Plan."""
    from bspp.orchestration.control.phase_materialization import materialize_phase

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    try:
        result = materialize_phase(
            phase_plan,
            authority_root=authority_root,
            config_path=config_path,
            source_repo=source_repo or Path.cwd(),
        )
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)


@phase_group.command("run-postprocessing")
@click.argument("phase_plan", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Root for durable postprocessing Phase authority.",
)
@click.option(
    "--execution-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Dedicated coordinator root; all local evidence paths are derived below it.",
)
@click.option(
    "--source-repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Clean committed orchestration source represented by pinned qualification.",
)
@click.option("--poll-interval", type=click.FloatRange(min=0.001), default=30.0, show_default=True)
@click.option("--timeout", type=click.FloatRange(min=0.001), default=86400.0, show_default=True)
@click.pass_context
def phase_run_postprocessing_cmd(
    ctx: click.Context,
    phase_plan: Path,
    *,
    authority_root: Path,
    execution_root: Path,
    source_repo: Path | None,
    poll_interval: float,
    timeout: float,
) -> None:
    """Run or restart one postprocessing Phase through accepted finalization."""
    from bspp.orchestration.control.postprocessing_phase_operator import (
        DEFAULT_POSTPROCESSING_PHASE_OPERATOR_DEPENDENCIES,
        run_postprocessing_phase,
    )

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    dependencies = replace(
        DEFAULT_POSTPROCESSING_PHASE_OPERATOR_DEPENDENCIES,
        emit=lambda event: click.echo(json.dumps(event, sort_keys=True, separators=(",", ":"))),
    )
    try:
        result = run_postprocessing_phase(
            phase_plan,
            authority_root=authority_root,
            execution_root=execution_root,
            source_repo=source_repo or Path.cwd(),
            config_path=config_path,
            poll_interval_seconds=poll_interval,
            timeout_seconds=timeout,
            dependencies=dependencies,
        )
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)
    if result.status != "accepted":
        ctx.exit(1)


@phase_group.command("retry-postprocessing")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing durable postprocessing Phase authority.",
)
@click.option(
    "--execution-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Dedicated create-once coordinator root for this explicit Retry.",
)
@click.option(
    "--source-repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Clean committed orchestration source represented by current qualification.",
)
@click.option("--poll-interval", type=click.FloatRange(min=0.001), default=30.0, show_default=True)
@click.option("--timeout", type=click.FloatRange(min=0.001), default=86400.0, show_default=True)
@click.pass_context
def phase_retry_postprocessing_cmd(
    ctx: click.Context,
    phase_run_id: str,
    *,
    authority_root: Path,
    execution_root: Path,
    source_repo: Path | None,
    poll_interval: float,
    timeout: float,
) -> None:
    """Explicitly Retry and drive one postprocessing Phase to finalization."""
    from bspp.orchestration.control.postprocessing_phase_retry_operator import (
        DEFAULT_POSTPROCESSING_PHASE_RETRY_OPERATOR_DEPENDENCIES,
        retry_postprocessing_phase_to_completion,
    )

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    dependencies = replace(
        DEFAULT_POSTPROCESSING_PHASE_RETRY_OPERATOR_DEPENDENCIES,
        lifecycle=replace(
            DEFAULT_POSTPROCESSING_PHASE_RETRY_OPERATOR_DEPENDENCIES.lifecycle,
            emit=lambda event: click.echo(json.dumps(event, sort_keys=True, separators=(",", ":"))),
        ),
    )
    try:
        result = retry_postprocessing_phase_to_completion(
            phase_run_id,
            authority_root=authority_root,
            execution_root=execution_root,
            source_repo=source_repo or Path.cwd(),
            config_path=config_path,
            poll_interval_seconds=poll_interval,
            timeout_seconds=timeout,
            dependencies=dependencies,
        )
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)
    if result.status != "accepted":
        ctx.exit(1)


@phase_group.command("retry")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing the durable Phase Run authority.",
)
@click.option("--profile", "profile_name", default=None, help="Explicit successor Cluster Profile id.")
@click.option(
    "--source-repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Clean committed orchestration source represented by qualification.",
)
@click.option(
    "--carry-forward",
    "carry_forward_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Preprocessing-only strict JSON selection of verified predecessor A3Ms to adopt.",
)
@click.pass_context
def phase_retry_cmd(
    ctx: click.Context,
    phase_run_id: str,
    *,
    authority_root: Path,
    profile_name: str | None,
    source_repo: Path | None,
    carry_forward_path: Path | None,
) -> None:
    """Materialize one clean immutable successor Attempt without submission."""
    from bspp.orchestration.control.phase_retry import retry_phase

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    try:
        result = retry_phase(
            phase_run_id,
            authority_root=authority_root,
            config_path=config_path,
            profile_name=profile_name,
            source_repo=source_repo or Path.cwd(),
            carry_forward_path=carry_forward_path,
        )
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)


@phase_group.command("submit")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing the durable materialized Phase Run authority.",
)
def phase_submit_cmd(phase_run_id: str, *, authority_root: Path) -> None:
    """Dispatch every declared Runtime Action and return after assignment."""
    from bspp.orchestration.control.phase_submission import submit_phase

    try:
        result = submit_phase(phase_run_id, authority_root=authority_root)
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)


@phase_group.command("status")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing the durable Phase Run authority.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(("table", "text", "json")),
    default="table",
    show_default=True,
    help="Render format for stdout.",
)
def phase_status_cmd(phase_run_id: str, *, authority_root: Path, output_format: str) -> None:
    """Observe durable Phase state and Slurm accounting without mutation."""
    from bspp.orchestration.control.phase_status import status_phase

    try:
        report = status_phase(phase_run_id, authority_root=authority_root)
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    rendered = report.render_json() if output_format == "json" else report.render_table()
    click.echo(rendered, nl=False)


@phase_group.command("diagnostics")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing the durable Phase Run authority.",
)
@click.option(
    "--diagnostics-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="External directory for the replaceable warning-only diagnostic summary.",
)
def phase_diagnostics_cmd(phase_run_id: str, *, authority_root: Path, diagnostics_root: Path) -> None:
    """Recapture bounded scheduler, failed-job, and action09 acceptance diagnostics."""
    from bspp.orchestration.control.postprocessing_phase_diagnostics import (
        capture_postprocessing_phase_diagnostics,
    )

    try:
        result = capture_postprocessing_phase_diagnostics(
            phase_run_id,
            authority_root=authority_root,
            diagnostics_root=diagnostics_root,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)


@phase_group.command("resume")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing the durable Phase Run authority.",
)
def phase_resume_cmd(phase_run_id: str, *, authority_root: Path) -> None:
    """Reconcile the current active Phase Attempt exactly once."""
    from bspp.orchestration.control.phase_resume import resume_phase

    try:
        result = resume_phase(phase_run_id, authority_root=authority_root)
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)


@phase_group.command("cancel")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing the durable Phase Run authority.",
)
def phase_cancel_cmd(phase_run_id: str, *, authority_root: Path) -> None:
    """Cancel the current Phase Attempt and confirm completion through accounting."""
    from bspp.orchestration.control.phase_cancellation import cancel_phase

    try:
        result = cancel_phase(phase_run_id, authority_root=authority_root)
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)


@phase_group.command("finalize")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing the durable Phase Run authority.",
)
@click.option(
    "--scheduler-evidence",
    "scheduler_evidence_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Family-specific successful terminal scheduler evidence.",
)
@click.option(
    "--action-evidence",
    "action_evidence_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Family-specific successful action evidence or Runtime aggregate.",
)
@click.option(
    "--handoff",
    "handoff_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Family-specific validated Runtime handoff directory.",
)
@click.option(
    "--acceptance-adjudication",
    "acceptance_adjudication_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Postprocessing-only adjudication inside the exact fetched handoff.",
)
def phase_finalize_cmd(
    phase_run_id: str,
    *,
    authority_root: Path,
    scheduler_evidence_path: Path | None,
    action_evidence_path: Path | None,
    handoff_path: Path | None,
    acceptance_adjudication_path: Path | None,
) -> None:
    """Issue an attempt-bound receipt and permanently seal one Phase Run."""
    from bspp.orchestration.control.phase_finalization import finalize_phase

    try:
        result = finalize_phase(
            phase_run_id,
            authority_root=authority_root,
            scheduler_evidence_path=scheduler_evidence_path,
            action_evidence_path=action_evidence_path,
            handoff_path=handoff_path,
            acceptance_adjudication_path=acceptance_adjudication_path,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)


@phase_group.command("publish-preprocessing")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing the durable Phase Run authority.",
)
@click.option(
    "--handoff-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing the preprocessing handoff (artifact-location.json).",
)
@click.option(
    "--evidence-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to receive artifact-location-remote.json and msa-set-upload-evidence.json.",
)
@click.option("--profile", "profile_name", required=True, help="Cluster Profile id for Slurm submission.")
@click.pass_context
def phase_publish_preprocessing_cmd(
    ctx: click.Context,
    phase_run_id: str,
    *,
    authority_root: Path,
    handoff_path: Path,
    evidence_dir: Path,
    profile_name: str,
) -> None:
    """Publish a verified local MSA-set bundle to S3 cluster-side via a Slurm job.

    The runtime publish command executes inside the imported runtime container
    on the cluster (never the local workstation interpreter), reading the
    bundle from Lustre and uploading to S3.  See ADR-0075 § "Boundary
    classification for the publish family".
    """
    from bspp.orchestration.control.phase_publish import publish_preprocessing_phase
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    try:
        profile = resolve_cluster_profile(profile_name, config_path=config_path)
        result = publish_preprocessing_phase(
            phase_run_id,
            authority_root=authority_root,
            handoff_path=handoff_path,
            evidence_dir=evidence_dir,
            cluster_profile=profile,
        )
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)


@phase_group.command("publish-folding")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing the durable Phase Run authority.",
)
@click.option(
    "--bundles",
    "bundles_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON file containing operator-attested prediction bundle records.",
)
@click.option(
    "--local-paths",
    "local_paths_json",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON file containing the list of local bundle file paths.",
)
@click.option(
    "--evidence-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to receive prediction-bundle-upload-evidence.json.",
)
@click.option("--profile", "profile_name", required=True, help="Cluster Profile id for Slurm submission.")
@click.pass_context
def phase_publish_folding_cmd(
    ctx: click.Context,
    phase_run_id: str,
    *,
    authority_root: Path,
    bundles_path: Path,
    local_paths_json: Path,
    evidence_dir: Path,
    profile_name: str,
) -> None:
    """Publish operator-attested prediction bundles to S3 cluster-side via a Slurm job.

    The runtime publish command executes inside the imported runtime container
    on the cluster (never the local workstation interpreter), reading the
    bundles from Lustre and uploading to S3.  See ADR-0075 § "Boundary
    classification for the publish family".
    """
    from bspp.orchestration.control.phase_publish import publish_folding_phase
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    try:
        profile = resolve_cluster_profile(profile_name, config_path=config_path)
        result = publish_folding_phase(
            phase_run_id,
            authority_root=authority_root,
            bundles_path=bundles_path,
            local_paths_json=local_paths_json,
            evidence_dir=evidence_dir,
            cluster_profile=profile,
        )
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)


@phase_group.command("derive-seam-parquets")
@click.argument("phase_run_id")
@click.option(
    "--authority-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing the durable Phase Run authority.",
)
@click.option(
    "--index",
    "index_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Completed folding run's canonical-pair index JSON.",
)
@click.option(
    "--evidence",
    "evidence_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Completed folding run's canonical-pair action evidence JSON.",
)
@click.option(
    "--master-output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination for the derived master parquet.",
)
@click.option(
    "--tracking-output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination for the derived tracking parquet.",
)
@click.option("--s3-output-prefix", required=True, help="Postprocessing s3:// output prefix.")
@click.option("--source-run", required=True, help="Authored postprocessing dataset.name selector.")
@click.option("--archive-name", required=True, help="Prediction bundle name for swiftstack_archive.")
@click.option("--gcs-destination-prefix", default=None, help="Optional gs:// destination prefix.")
@click.option(
    "--evidence-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to receive seam-derivation-result.json.",
)
@click.option("--force", is_flag=True, help="Overwrite existing output files.")
def phase_derive_seam_parquets_cmd(
    phase_run_id: str,
    *,
    authority_root: Path,
    index_path: Path,
    evidence_path: Path,
    master_output: Path,
    tracking_output: Path,
    s3_output_prefix: str,
    source_run: str,
    archive_name: str,
    gcs_destination_prefix: str | None,
    evidence_dir: Path,
    force: bool,
) -> None:
    """Derive master + tracking parquets from a completed folding run.

    Pure local transform driven from the Control Plane, mirroring the
    publish-folding transport discipline: validate the Phase Run, write the
    input JSON, and delegate to the runtime CLI's phase command. The supplied
    canonical-pair index is provenance-bound to the validated phase_run_id.
    """
    from bspp.orchestration.contract.phase import FoldingPhaseRunSpec
    from bspp.orchestration.control.phase_authority import PhaseAuthorityStore

    try:
        store = PhaseAuthorityStore(authority_root)
        authority = store.validate(phase_run_id)
        runspec = authority.phase_runspec
        if not isinstance(runspec, FoldingPhaseRunSpec):
            raise click.ClickException("authority is not a folding phase run")
        for output in (master_output, tracking_output):
            resolved = output.resolve()
            if resolved == authority_root.resolve() or str(resolved).startswith(str(authority_root.resolve()) + "/"):
                raise click.ClickException(
                    "--master-output/--tracking-output must not resolve inside the authority root"
                )
        evidence_resolved = evidence_dir.resolve()
        if evidence_resolved == authority_root.resolve() or str(evidence_resolved).startswith(
            str(authority_root.resolve()) + "/"
        ):
            raise click.ClickException("--evidence-dir must not resolve inside the authority root")
        evidence_dir.mkdir(parents=True, exist_ok=True)
        input_payload = {
            "index_path": str(index_path),
            "evidence_path": str(evidence_path),
            "master_output": str(master_output),
            "tracking_output": str(tracking_output),
            "s3_output_prefix": s3_output_prefix,
            "source_run": source_run,
            "archive_name": archive_name,
            "phase_run_id": phase_run_id,
            "gcs_destination_prefix": gcs_destination_prefix,
            "force": force,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=evidence_dir) as f:
            json.dump(input_payload, f, indent=2, sort_keys=True)
            input_json_path = Path(f.name)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bspp.orchestration.runtime.cli",
                    "phase",
                    "derive-seam-parquets",
                    "--input-json",
                    str(input_json_path),
                    "--evidence-dir",
                    str(evidence_dir),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        finally:
            input_json_path.unlink(missing_ok=True)
    except FileNotFoundError as exc:
        raise click.ClickException("runtime CLI not available — install bspp-orchestration-runtime") from exc
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"runtime CLI failed: {exc.stderr.strip() or exc.stdout.strip()}") from exc
    except click.ClickException:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.stdout, nl=False)


@cli.group("runtime")
def runtime_group() -> None:
    """Runtime qualification operations."""


@runtime_group.group("preprocessing")
def runtime_preprocessing_group() -> None:
    """Preprocessing Runtime Qualification operations."""


@runtime_preprocessing_group.command("qualify")
@click.option("--profile", "profile_name", required=True, help="Cluster Profile id to qualify.")
@click.option(
    "--source-repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Clean committed orchestration source represented by the image.",
)
@click.pass_context
def runtime_preprocessing_qualify_cmd(
    ctx: click.Context,
    *,
    profile_name: str,
    source_repo: Path | None,
) -> None:
    """Submit the scheduled smoke for one exact preprocessing runtime."""
    from bspp.orchestration.control.preprocessing_runtime_qualification import (
        preprocessing_runtime_qualification_path,
        qualify_preprocessing_runtime,
        render_preprocessing_runtime_qualification_yaml,
    )
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    try:
        profile = resolve_cluster_profile(profile_name, config_path=config_path)
        record = qualify_preprocessing_runtime(
            profile_name=profile_name,
            config_path=config_path,
            source_repo=source_repo or Path.cwd(),
        )
        record_path = preprocessing_runtime_qualification_path(profile, tuple_id=record.tuple_id)
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(render_preprocessing_runtime_qualification_yaml(record, record_path=record_path), nl=False)


@runtime_preprocessing_group.command("resolve")
@click.option("--profile", "profile_name", required=True, help="Cluster Profile id to resolve.")
@click.option(
    "--source-repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Clean committed orchestration source represented by the image.",
)
@click.pass_context
def runtime_preprocessing_resolve_cmd(
    ctx: click.Context,
    *,
    profile_name: str,
    source_repo: Path | None,
) -> None:
    """Mirror qualified cluster authority into workstation Control state."""
    from bspp.orchestration.control.preprocessing_runtime_qualification import (
        preprocessing_runtime_qualification_path,
        render_preprocessing_runtime_qualification_yaml,
        resolve_preprocessing_runtime_qualification,
    )
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    try:
        profile = resolve_cluster_profile(profile_name, config_path=config_path)
        record = resolve_preprocessing_runtime_qualification(
            profile_name=profile_name,
            config_path=config_path,
            source_repo=source_repo or Path.cwd(),
        )
        record_path = preprocessing_runtime_qualification_path(profile, tuple_id=record.tuple_id)
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(render_preprocessing_runtime_qualification_yaml(record, record_path=record_path), nl=False)


@runtime_preprocessing_group.command("stage-source-bundle")
@click.option(
    "--profile",
    "profile_name",
    required=True,
    help="Cluster Profile id whose source_bundle_root receives the bundle.",
)
@click.option(
    "--source-repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Clean committed orchestration source to bundle.",
)
@click.option(
    "--build-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Local directory for the bundle archive and identity evidence (default: <source-repo>/.bspp-source-bundles).",
)
@click.option("--dry-run", is_flag=True, help="Build and verify the bundle without staging it to the cluster.")
@click.pass_context
def runtime_preprocessing_stage_source_bundle_cmd(
    ctx: click.Context,
    *,
    profile_name: str,
    source_repo: Path | None,
    build_dir: Path | None,
    dry_run: bool,
) -> None:
    """Build and stage the provenance-only preprocessing Source Bundle."""
    from bspp.orchestration.control.preprocessing_source_bundle import (
        build_and_stage_preprocessing_source_bundle,
        render_preprocessing_source_bundle_yaml,
    )
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    local_source_repo = source_repo or Path.cwd()
    local_build_dir = build_dir or local_source_repo / ".bspp-source-bundles"
    try:
        profile = resolve_cluster_profile(profile_name, config_path=config_path)
        record = build_and_stage_preprocessing_source_bundle(
            local_source_repo,
            build_dir=local_build_dir,
            profile=profile,
            dry_run=dry_run,
        )
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(render_preprocessing_source_bundle_yaml(record), nl=False)


@cli.group("release")
def release_group() -> None:
    """Independent release acceptance and publication approval."""


@release_group.command("approve-publication")
@click.argument("acceptance", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--evidence-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--expected-runspec-sha256", required=True)
@click.option("--destination", required=True)
@click.option("--approval", "approval_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
def release_approve_publication_cmd(
    acceptance: Path,
    *,
    evidence_root: Path,
    expected_runspec_sha256: str,
    destination: str,
    approval_path: Path,
) -> None:
    """Create one approval after independently revalidating release evidence."""
    from bspp.orchestration.control.release_approval import approve_publication

    try:
        approval = approve_publication(
            acceptance,
            evidence_root=evidence_root,
            expected_runspec_sha256=expected_runspec_sha256,
            destination=destination,
            approval_path=approval_path,
        )
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(yaml.safe_dump({"publication_approval": approval.to_mapping()}, sort_keys=False), nl=False)


@runtime_group.command("qualify")
@click.option("--profile", "profile_name", required=True, help="Cluster Profile id to qualify.")
@click.option(
    "--source-repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Local source repository to identify for Runtime Qualification.",
)
@click.option(
    "--source-package-identity",
    "source_package_identity_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Staged governed SourcePackageIdentity record (required for SSH profiles).",
)
@click.pass_context
def runtime_qualify_cmd(
    ctx: click.Context,
    *,
    profile_name: str,
    source_repo: Path | None,
    source_package_identity_path: Path | None,
) -> None:
    """Create or refresh Runtime Qualification evidence."""
    from bspp.orchestration.control.runtime_qualification import (
        qualify_runtime,
        render_runtime_qualification_yaml,
    )

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    try:
        record = qualify_runtime(
            profile_name=profile_name,
            config_path=config_path,
            source_repo=source_repo or Path.cwd(),
            source_package_identity_path=source_package_identity_path,
        )
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(render_runtime_qualification_yaml(record), nl=False)


@runtime_group.command("resolve")
@click.option("--profile", "profile_name", required=True, help="Cluster Profile id to resolve.")
@click.option(
    "--source-repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Clean committed orchestration source represented by the qualification.",
)
@click.pass_context
def runtime_resolve_cmd(
    ctx: click.Context,
    *,
    profile_name: str,
    source_repo: Path | None,
) -> None:
    """Resolve and authenticate an exact completed Runtime Qualification."""
    from bspp.orchestration.control.runtime_qualification import (
        check_runtime_qualification,
        render_runtime_qualification_check_yaml,
    )

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    try:
        check = check_runtime_qualification(
            profile_name=profile_name,
            config_path=config_path,
            source_repo=source_repo or Path.cwd(),
        )
        output = render_runtime_qualification_check_yaml(check)
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(output, nl=False)
    if not check.current:
        ctx.exit(1)


def _config_path(config_path: Path | None) -> Path:
    if config_path is not None:
        return config_path
    env_config = os.environ.get("BSPPCTL_CONFIG")
    if env_config:
        return Path(env_config)
    return Path.home() / ".config" / "bsppctl" / "profiles.yaml"


@cli.command("prepare-benchmark")
@click.option(
    "--spec",
    "spec_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("configs/benchmark.pdb-temporal-v1.json"),
    show_default=True,
    help="Benchmark corpus spec JSON (schema_version 1).",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="New output directory for the materialized corpus.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=12,
    show_default=True,
    help="Parallel candidate materialization workers.",
)
@click.option(
    "--reconstruct",
    "reconstruct_target_list",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Reconstruct from a pinned JSONL target list instead of discovering from RCSB.",
)
def prepare_benchmark_cmd(
    *,
    spec_path: Path,
    output: Path,
    workers: int,
    reconstruct_target_list: Path | None,
) -> None:
    """Curate the pinned folding benchmark corpus into a new output directory."""
    from bspp.orchestration.control.folding_benchmark.curator import (
        PYARROW_SETUP_COMMAND,
        PyArrowUnavailable,
        prepare_benchmark_dataset,
        reconstruct_benchmark_dataset,
    )
    from bspp.orchestration.control.folding_benchmark.spec import load_benchmark_spec

    try:
        specification = load_benchmark_spec(spec_path)
        if reconstruct_target_list is not None:
            summary = reconstruct_benchmark_dataset(
                output,
                specification,
                reconstruct_target_list,
                workers=workers,
                progress=lambda message: click.echo(message, err=True),
            )
        else:
            summary = prepare_benchmark_dataset(
                output,
                specification,
                workers=workers,
                progress=lambda message: click.echo(message, err=True),
            )
    except PyArrowUnavailable as exc:
        raise click.ClickException(f"{PYARROW_SETUP_COMMAND} --spec {spec_path} --output {output}") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary, sort_keys=True), nl=False)


@cli.command("validate-run")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Cluster-resident completed-run directory.",
)
@click.option(
    "--suite",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Cluster-resident validation-suite JSON.",
)
@click.option(
    "--index",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Cluster-resident canonical-pair index JSON.",
)
@click.option("--corpus", required=True, help="S3 prefix whose contents are fetched recursively.")
@click.option("--fingerprint", required=True, help="Expected pinned corpus fingerprint.")
@click.option("--profile", "profile_name", required=True, help="Cluster Profile id.")
@click.option(
    "--aws-profile",
    "aws_profile",
    default=None,
    help="AWS shared-profile name for the corpus fetch (defaults to runtime discovery).",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Cluster-resident evidence directory (receives summary.json + validation.parquet).",
)
@click.option("--poll-interval", type=click.FloatRange(min=0.001), default=30.0, show_default=True)
@click.option("--timeout", type=click.FloatRange(min=0.001), default=86400.0, show_default=True)
@click.pass_context
def validate_run_cmd(
    ctx: click.Context,
    run_dir: Path,
    suite: Path,
    index: Path,
    corpus: str,
    fingerprint: str,
    profile_name: str,
    output_dir: Path,
    poll_interval: float,
    timeout: float,
    aws_profile: str | None,
) -> None:
    """Submit and monitor a cluster-side folding benchmark validation."""
    from bspp.orchestration.control.folding_benchmark_submit import submit_benchmark_validation
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    try:
        profile = resolve_cluster_profile(profile_name, config_path=config_path)
        result = submit_benchmark_validation(
            cluster_profile=profile,
            run_dir=run_dir,
            suite=suite,
            index=index,
            corpus=corpus,
            fingerprint=fingerprint,
            output_dir=output_dir,
            poll_interval_seconds=poll_interval,
            timeout_seconds=timeout,
            aws_profile=aws_profile,
        )
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)
    if not result.accepted:
        ctx.exit(1)


@cli.command("legacy-msa-import")
@click.option(
    "--handoff-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Cluster-resident legacy preprocessing MSA handoff directory.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Cluster-resident enriched handoff directory (empty for a new submission).",
)
@click.option("--profile", "profile_name", required=True, help="Cluster Profile id.")
@click.option("--lz4", "lz4_executable", default="lz4", show_default=True, help="lz4 executable name.")
@click.option("--poll-interval", type=click.FloatRange(min=0.001), default=30.0, show_default=True)
@click.option("--timeout", type=click.FloatRange(min=0.001), default=86400.0, show_default=True)
@click.option("--resume-job-id", default=None, help="Verify and monitor this existing scalar job; never submit.")
@click.pass_context
def legacy_msa_import_cmd(
    ctx: click.Context,
    handoff_root: Path,
    output_dir: Path,
    profile_name: str,
    lz4_executable: str,
    poll_interval: float,
    timeout: float,
    resume_job_id: str | None,
) -> None:
    """Submit or resume a cluster-side legacy-MSA import for length enrichment."""
    from bspp.orchestration.control.legacy_msa_import import resume_legacy_msa_import, submit_legacy_msa_import
    from bspp.orchestration.control.profiles import resolve_cluster_profile

    config_path = _config_path(ctx.obj.get("config_path") if ctx.obj else None)
    try:
        profile = resolve_cluster_profile(profile_name, config_path=config_path)
        if resume_job_id is None:
            result = submit_legacy_msa_import(
                cluster_profile=profile,
                handoff_root=handoff_root,
                output_dir=output_dir,
                poll_interval_seconds=poll_interval,
                timeout_seconds=timeout,
                lz4_executable=lz4_executable,
            )
        else:
            result = resume_legacy_msa_import(
                job_id=resume_job_id,
                cluster_profile=profile,
                handoff_root=handoff_root,
                output_dir=output_dir,
                poll_interval_seconds=poll_interval,
                timeout_seconds=timeout,
                lz4_executable=lz4_executable,
            )
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.render_json(), nl=False)


@cli.group("data")
def data_group() -> None:
    """Operator data-movement planning."""


@data_group.command("plan")
@click.option(
    "--phase-plan",
    "phase_plan_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Plan-referenced mode: path to a FoldingPhasePlan YAML.",
)
@click.option("--source", default=None, help="Manual mode: local source path or s3:// URI.")
@click.option(
    "--destination",
    default=None,
    help="Manual mode: full S3 object key (e.g. s3://bucket/prefix/file.tar.lz4), not a directory prefix.",
)
@click.option("--size-bytes", type=int, default=None, help="Manual mode: verified size in bytes.")
@click.option("--sha256", default=None, help="Manual mode: verified sha256 (64 lowercase hex).")
@click.option(
    "--s3-prefix",
    default=None,
    help="REQUIRED in plan-referenced mode without --override-prefix; optional in manual mode.",
)
@click.option(
    "--override-prefix",
    default=None,
    help="Recovery mode: different destination prefix (s3://...).",
)
@click.option("--dry-run/--no-dry-run", default=True, help="Plan preview only (default: dry-run).")
@click.option("--format", type=click.Choice(["json", "yaml"]), default="yaml", help="Output format.")
def data_plan_cmd(
    phase_plan_path: Path | None,
    source: str | None,
    destination: str | None,
    size_bytes: int | None,
    sha256: str | None,
    s3_prefix: str | None,
    override_prefix: str | None,
    dry_run: bool,
    format: str,
) -> None:
    """Plan an operator data-movement transfer (dry-run by default)."""
    from bspp.orchestration.control.data_movement import (
        plan_data_movement,
        render_operator_transfer_plan_json,
        render_operator_transfer_plan_yaml,
    )

    if phase_plan_path is not None and s3_prefix is None and override_prefix is None:
        raise click.ClickException("--phase-plan requires --s3-prefix (or --override-prefix for recovery mode)")
    try:
        plan = plan_data_movement(
            phase_plan_path=phase_plan_path,
            source=source,
            destination=destination,
            size_bytes=size_bytes,
            sha256=sha256,
            s3_prefix=s3_prefix,
            override_prefix=override_prefix,
            dry_run=dry_run,
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc
    if format == "json":
        click.echo(render_operator_transfer_plan_json(plan))
    else:
        click.echo(render_operator_transfer_plan_yaml(plan), nl=False)
