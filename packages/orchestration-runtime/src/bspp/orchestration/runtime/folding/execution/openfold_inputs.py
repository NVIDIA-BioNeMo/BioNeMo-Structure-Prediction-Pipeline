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

"""OpenFold input preprocessor for Track-A folding.

Port Baseline: frozen reference pipeline src/afdb_pipeline/backends/local.py
(``OpenFoldInputPreprocessor``) with ``NO_TEMPLATE_CIF`` harvested from
src/afdb_pipeline/cluster_worker.py.  The template sentinel is written
unconditionally here (rather than conditionally in the cluster worker), and the
numeric ``chain_index`` remains 1-based while the chain_id letters stay on the
harvested 0-based ``A``/``B``/``C`` offset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from bspp.orchestration.contract.folding_input import FoldingInputLayout

from .a3m_split import a3m_to_stockholm
from .errors import FoldingBackendError
from .models import PreparedInput, ProteinTarget, require_valid_target_identity
from .msa_models import MSAResult

__all__ = ["NO_TEMPLATE_CIF", "OpenFoldInputPreprocessor"]

NO_TEMPLATE_CIF = """data_openfold_no_template
_entry.id openfold_no_template
#
"""


class OpenFoldInputPreprocessor:
    name = "openfold-inputs"

    def run(self, target: ProteinTarget, msa: MSAResult, output_dir: Path) -> PreparedInput:
        # The target identity names chain directories and the FASTA file, so it
        # must satisfy the model-ID grammar before any filesystem operation.
        require_valid_target_identity(target)
        if len(msa.chains) != len(target.chains):
            raise FoldingBackendError("MSA result does not contain one alignment bundle per chain")
        fasta_dir = output_dir / "fasta"
        alignment_dir = output_dir / "alignments"
        template_dir = output_dir / "templates"
        for path in (fasta_dir, alignment_dir, template_dir):
            path.mkdir(parents=True, exist_ok=True)

        fasta_records: list[str] = []
        chain_ids: list[str] = []
        for index, (sequence, chain_msa) in enumerate(zip(target.chains, msa.chains, strict=True)):
            chain_id = f"{target.target_id}_{chr(ord('A') + index)}"
            chain_ids.append(chain_id)
            fasta_records.append(f">{chain_id}\n{sequence}\n")
            chain_dir = alignment_dir / chain_id
            chain_dir.mkdir(exist_ok=True)
            primary_alignment = next(iter(chain_msa.alignments.values()))
            shutil.copyfile(primary_alignment, chain_dir / "alignment.a3m")
            (chain_dir / "uniprot_hits.sto").write_text(a3m_to_stockholm(chain_dir / "alignment.a3m"), encoding="utf-8")

        fasta_path = fasta_dir / f"{target.target_id}.fasta"
        fasta_path.write_text("".join(fasta_records), encoding="utf-8")
        metadata = {
            "chain_ids": chain_ids,
            "template_mode": "none",
            "pairing": "species-from-colabfold-headers",
        }
        layout = FoldingInputLayout(
            layout="openfold",
            chain_ids=tuple(chain_ids),
            template_mode="none",
            pairing="species-from-colabfold-headers",
        )
        (output_dir / "layout.json").write_text(
            json.dumps(layout.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (template_dir / "openfold_no_template.cif").write_text(NO_TEMPLATE_CIF, encoding="utf-8")
        return PreparedInput(self.name, fasta_dir, alignment_dir, template_dir, metadata)
