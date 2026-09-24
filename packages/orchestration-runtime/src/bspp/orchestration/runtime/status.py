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

"""Human-readable status reporting for tracking parquets and shard outputs.

Two rendering paths:

- :func:`format_status_report` — plain text, used by the basic CLI
  ``status`` subcommand. Preserved for back-compat with existing tests
  that assert on string output.
- :func:`render_status_rich` — optional rich-powered rendering with a
  progress bar + table, used when the caller opts in with ``--rich``.
  Falls back to plain text if ``rich`` is not installed so the package
  keeps working in minimal environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bspp.orchestration.runtime.validation.count_outputs import CountReport

__all__ = [
    "format_status_report",
    "render_full_status",
    "render_status_rich",
]


def format_status_report(
    status_counts: dict[str, int],
    dataset: str | None = None,
) -> str:
    """Format a human-readable status table from status counts.

    Args:
        status_counts: Mapping of status name to row count.
        dataset: Optional dataset name to include in the header.

    Returns:
        Multi-line string suitable for terminal display.
    """
    total = sum(status_counts.values())

    lines: list[str] = []
    header = f"Status report for dataset: {dataset}" if dataset else "Status report (all datasets)"
    lines.append(header)
    lines.append("-" * len(header))

    if total == 0:
        lines.append("No rows found.")
        return "\n".join(lines)

    lines.append(f"Total rows: {total:,}")
    lines.append("")

    from bspp.orchestration.runtime.postprocessing.tracking import LIFECYCLE_STATUSES

    ordered = [s for s in LIFECYCLE_STATUSES if s in status_counts]
    extra = sorted(set(status_counts) - set(LIFECYCLE_STATUSES))
    ordered.extend(extra)

    for status in ordered:
        count = status_counts[status]
        pct = 100.0 * count / total if total else 0.0
        lines.append(f"  {status:<15s} {count:>10,}  ({pct:5.1f}%)")

    return "\n".join(lines)


def render_full_status(
    status_counts: dict[str, int],
    *,
    dataset: str | None,
    count_report: CountReport | None = None,
    use_rich: bool = False,
) -> str:
    """Render status + optional per-shard count details.

    ``use_rich=True`` produces a rich-formatted table with a progress
    bar when available; otherwise falls back to the plain format.
    """
    if use_rich:
        try:
            return render_status_rich(status_counts, dataset=dataset, count_report=count_report)
        except ImportError:
            # rich not installed — fall back to plain text
            pass

    text = format_status_report(status_counts, dataset=dataset)
    if count_report is not None:
        from bspp.orchestration.runtime.validation.count_outputs import render_count_report

        text = text + "\n\n" + render_count_report(count_report)
    return text


def render_status_rich(
    status_counts: dict[str, int],
    *,
    dataset: str | None,
    count_report: CountReport | None = None,
) -> str:
    """Render the status + optional count report using rich.

    Raises :class:`ImportError` if ``rich`` is not available; callers
    that want a graceful fallback should use :func:`render_full_status`.
    """
    from io import StringIO

    from rich.console import Console
    from rich.progress_bar import ProgressBar
    from rich.table import Table

    from bspp.orchestration.runtime.postprocessing.tracking import LIFECYCLE_STATUSES

    buf = StringIO()
    console = Console(file=buf, width=100, record=True, force_terminal=False)

    total = sum(status_counts.values())
    header = f"Status: {dataset}" if dataset else "Status (all datasets)"
    console.rule(header)

    if total == 0:
        console.print("[yellow]No rows found.[/yellow]")
        return buf.getvalue()

    # Lifecycle progress: "done" count / total
    done_count = status_counts.get("done", 0)
    bar = ProgressBar(total=total, completed=done_count, width=60)
    console.print(bar)
    console.print(f"done: {done_count:,}/{total:,} ({100 * done_count / total:5.1f}%)")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    table.add_column("Share", justify="right")
    ordered = [s for s in LIFECYCLE_STATUSES if s in status_counts]
    extra = sorted(set(status_counts) - set(LIFECYCLE_STATUSES))
    ordered.extend(extra)
    for status in ordered:
        count = status_counts[status]
        pct = 100.0 * count / total
        table.add_row(status, f"{count:,}", f"{pct:5.1f}%")
    console.print(table)

    if count_report is not None:
        console.print()
        shard_table = Table(title="Per-shard output counts", show_header=True, header_style="bold")
        shard_table.add_column("Shard")
        shard_table.add_column("Expected", justify="right")
        shard_table.add_column("Actual", justify="right")
        shard_table.add_column("Diff", justify="right")
        shard_table.add_column("Source")
        for s in count_report.shards:
            diff_style = "green" if s.ok else "red"
            shard_table.add_row(
                f"shard_{s.shard_id}",
                f"{s.expected:,}",
                f"{s.actual:,}",
                f"[{diff_style}]{s.diff:+,}[/{diff_style}]",
                s.source,
            )
        console.print(shard_table)

    return buf.getvalue()
