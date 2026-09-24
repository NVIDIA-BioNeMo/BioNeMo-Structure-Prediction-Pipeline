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

"""Job-local persistent BioIR/OpenFold2 fold session.

Port Baseline: frozen reference pipeline src/afdb_pipeline/bioir_backend.py
(``BioIRFoldSession`` and its ``_request`` / ``predict`` / ``close`` methods).

BioIR stays an in-process persistent session (``predict``/``close``), NOT a
``run(target, prepared, output_dir)`` subprocess: it does not submit Slurm and
does not launch a nested container.  All BioIR/torch imports are lazy inside
``__init__`` / ``_request`` / ``predict`` / ``close`` so control-plane commands
and unit tests never require the GPU runtime.  The explicit ``.pt`` checkpoint
is validated before any BioIR import and is bound during session construction
and processor calls through the selected model's scoped checkpoint overlay.
This covers eager and lazy model loading while restoring the prior value or
absence in ``finally``.  The full
unmodified ``get_scores`` result is captured before any compaction.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from bspp.orchestration.contract.folding_bioir import BIOIR_MONOMER_TOOL_USED
from bspp.orchestration.contract.folding_input import bioir_request_manifest_from_mapping

from .bioir_config import OpenFoldModelSettings, OpenFoldSettings, bioir_checkpoint_env, bioir_model_source
from .bioir_scores import capture_bioir_full_scores, quality_from_bioir_scores
from .emitter_support import canonical_pair_names, select_tool_used, serialize_scores_json
from .errors import FoldingBackendError
from .models import FoldingResult, PreparedInput, ProteinTarget, StructurePrediction

__all__ = ["BIOIR_TOOL_USED", "BioIRFoldSession", "bioir_checkpoint_env_overlay"]

# Frozen VALID_TOOL_USED[3] literal.  Both the homodimer and heterodimer
# selections use the same string, so the leaked-homodimer flag is inert for
# BioIR (see ``predict``).
BIOIR_TOOL_USED = "OpenFold2 (BioNeMo IR) / AlphaFold-Multimer"


@contextmanager
def bioir_checkpoint_env_overlay(checkpoint: Path, model_source: str) -> Iterator[None]:
    """Bind ``bioir_checkpoint_env(model_source)`` to ``checkpoint`` for a scope.

    The prior value (or absence) is restored in ``finally``.  No
    checkpoint existence validation happens here; that belongs to
    :class:`BioIRFoldSession.__init__`.  ``checkpoint.resolve()`` is non-strict
    so a fake path works in tests.
    """

    env_var = bioir_checkpoint_env(model_source)
    resolved = str(checkpoint.resolve())
    prior = os.environ.get(env_var)
    os.environ[env_var] = resolved
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = prior


class BioIRFoldSession:
    """One persistent BioIR/OpenFold2 model bound to one GPU worker."""

    name = "bioir"

    def __init__(
        self,
        model: OpenFoldModelSettings,
        settings: OpenFoldSettings,
        checkpoint: Path,
        output_dir: Path,
    ) -> None:
        model_source = bioir_model_source(model)
        if checkpoint.suffix != ".pt":
            raise FoldingBackendError(f"BioIR checkpoint must be a .pt file: {checkpoint}")
        if not checkpoint.is_file():
            raise FoldingBackendError(f"BioIR checkpoint is missing: {checkpoint}")
        # ``settings.compact_scores`` is accepted but deliberately
        # ignored.  The harvested ``_enable_compact_bioir_scores`` PAE-pop is
        # dropped so the full quadratic PAE matrix always survives into the
        # canonical scores JSON.
        self.model = model
        self.settings = settings
        self.model_source = model_source
        self.checkpoint = checkpoint.resolve()
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        with bioir_checkpoint_env_overlay(self.checkpoint, model_source):
            self.processor = self._build_processor(settings, model_source, model, output_dir)

    def _build_processor(
        self,
        settings: OpenFoldSettings,
        model_source: str,
        model: OpenFoldModelSettings,
        output_dir: Path,
    ) -> Any:
        from bionemo_ir.pipeline.processor.engine_proc import (  # type: ignore[import-not-found]
            EngineProcessorConfig,
            build_processor,
        )
        from bionemo_ir.pipeline.stages.configs import (  # type: ignore[import-not-found]
            FeatureGeneratorStageConfig,
            WriterStageConfig,
        )

        runtime_args: dict[str, Any] = {}
        if settings.recycling_steps is not None:
            runtime_args["recycling_steps"] = settings.recycling_steps
        config = EngineProcessorConfig(
            model_source=model_source,
            executor_backend=None,
            batch_size=1,
            should_continue_on_error=settings.continue_on_error,
            runtime_args=runtime_args,
            engine_kwargs={"profile_inference": settings.profile_inference},
            feature_generator_stage=FeatureGeneratorStageConfig(init_context={"random_seed": model.seed}),
            writer_stage=WriterStageConfig(
                output_path=str(output_dir.resolve()),
                format="pdb",
            ),
        )
        return build_processor(config)

    def _request(self, prepared_dir: Path) -> Any:
        from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer  # type: ignore[import-not-found]

        try:
            payload = json.loads((prepared_dir / "bioir-request.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FoldingBackendError(f"BioIR prepared input is unreadable: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise FoldingBackendError("BioIR prepared input must be a JSON object")
        try:
            manifest = bioir_request_manifest_from_mapping(payload)
        except ValueError as exc:
            raise FoldingBackendError(f"BioIR prepared input is malformed: {exc}") from exc

        polymers = []
        for polymer in manifest.polymers:
            unpaired = prepared_dir / polymer.unpaired_msa
            paired = prepared_dir / polymer.paired_msa if polymer.paired_msa else None
            polymers.append(
                Polymer(
                    polymer_type="protein",
                    chain_id=list(polymer.chain_ids),
                    sequence=polymer.sequence,
                    msas=[MSARecord(path=str(unpaired.resolve()), format="a3m")],
                    paired_msas=([MSARecord(path=str(paired.resolve()), format="a3m")] if paired is not None else None),
                    templates=None,
                )
            )
        return InputRequest(input_id=manifest.input_id, polymers=polymers)

    def run(self, target: ProteinTarget, prepared: PreparedInput, output_dir: Path) -> FoldingResult:
        """Protocol-shaped façade over the persistent ``predict`` seam.

        The prepared directories must share one prepared-input root, and
        ``output_dir`` must be the session's configured output root.  The
        persistent processor is reused and never closed here; ``close()``
        remains explicit and caller-owned.
        """

        prepared_root = self._prepared_input_root(prepared)
        if output_dir != self.output_dir:
            raise FoldingBackendError(
                f"BioIR run output_dir {output_dir!s} does not match the session output root {self.output_dir!s}"
            )
        return self.predict(target, prepared_root)

    def _prepared_input_root(self, prepared: PreparedInput) -> Path:
        roots = {prepared.fasta_dir.parent, prepared.alignment_dir.parent, prepared.template_dir.parent}
        if len(roots) != 1:
            raise FoldingBackendError("BioIR prepared directories must share one prepared-input root")
        return roots.pop()

    def predict(self, target: ProteinTarget, prepared_dir: Path) -> FoldingResult:
        import torch

        request = self._request(prepared_dir)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with bioir_checkpoint_env_overlay(self.checkpoint, self.model_source):
            rows = self.processor([{"record": request, "__record_id": target.target_id}])
        wall_seconds = time.perf_counter() - started
        if len(rows) != 1:
            raise FoldingBackendError(f"BioIR returned {len(rows)} rows for {target.target_id}; expected one")
        row = rows[0]
        error = row.get("__inference_error__")
        if isinstance(error, Mapping) and error.get("error_msg"):
            raise FoldingBackendError(f"BioIR inference failed for {target.target_id}: {error['error_msg']}")
        output_path = row.get("output_path")
        if not isinstance(output_path, str) or not Path(output_path).is_file():
            raise FoldingBackendError(f"BioIR produced no PDB structure for {target.target_id}")
        scores_value = row.get("scores")
        try:
            scores = json.loads(scores_value) if isinstance(scores_value, str) else scores_value
        except json.JSONDecodeError as exc:
            raise FoldingBackendError(f"BioIR returned invalid scores for {target.target_id}") from exc
        if not isinstance(scores, Mapping):
            raise FoldingBackendError(f"BioIR returned no scores for {target.target_id}")

        # capture the full unmodified get_scores result BEFORE any
        # compacting step.  Missing/empty/non-finite arrays raise here.
        plddt, pae, max_pae, ptm, iptm = capture_bioir_full_scores(scores)
        quality = quality_from_bioir_scores(scores)
        quality.update(
            {
                "bioir_model_source": self.model_source,
                "fold_wall_seconds": wall_seconds,
                "model_inference_seconds": row.get("model_inference_time"),
                "engine_time_seconds": row.get("time_taken"),
                "cuda_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
                ),
                "cuda_peak_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
                ),
            }
        )

        structure_name, scores_name = canonical_pair_names(target.target_id)
        structure_path = self.output_dir / structure_name
        scores_path = self.output_dir / scores_name
        shutil.copyfile(Path(output_path), structure_path)
        scores_path.write_text(
            serialize_scores_json(
                plddt=plddt,
                pae=pae,
                max_pae=max_pae,
                ptm=ptm,
                iptm=iptm,
                extras={"bioir_model_source": self.model_source},
            ),
            encoding="utf-8",
        )

        # Provenance identifies the actual selected model family. Homodimer
        # classification cannot change that selection.
        selected_tool = BIOIR_MONOMER_TOOL_USED if self.model_source == "openfold2_ptm_1" else BIOIR_TOOL_USED
        tool_used = select_tool_used(
            target.target_id,
            leaked_homodimer=False,
            homodimer_tool_used=selected_tool,
            heterodimer_tool_used=selected_tool,
        )
        return FoldingResult(
            backend=self.name,
            predictions=(StructurePrediction(rank=1, structure_path=structure_path, scores_path=scores_path),),
            metadata={
                "tool_used": tool_used,
                "model_source": self.model_source,
                "quality": quality,
            },
        )

    def close(self) -> None:
        self.processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
