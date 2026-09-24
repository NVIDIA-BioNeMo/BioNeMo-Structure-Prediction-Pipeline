#!/usr/bin/env bash
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

# Verify that the baked Torch/PyG stack uses the CUDA torch-cluster fast path.
#
# Usage:
#   bspp-container-smoke-gpu
#   bspp-container-smoke-gpu /path/to/smoke.json
#   BSPP_SMOKE_JSON=/path/to/smoke.json bspp-container-smoke-gpu
set -euo pipefail

JSON_OUT="${1:-${BSPP_SMOKE_JSON:-}}"

python - "${JSON_OUT}" <<'PY'
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

record = {
    "status": "failed",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.version.replace("\n", " "),
}
json_out = sys.argv[1] or None


def log(key: str, value: object) -> None:
    record[key] = value
    print(key, value)


def write_record() -> None:
    if json_out:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print("smoke_json", str(path))


try:
    import torch
    from torch_cluster import radius_graph

    log("python", record["python"])
    log("torch", torch.__version__)
    log("torch_cuda", torch.version.cuda)
    log("cuda_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    device_sm = f"sm_{capability[0]}{capability[1]}"
    arch_list = list(torch.cuda.get_arch_list())
    arch_set = set(arch_list)
    ptx_arch = f"compute_{capability[0]}{capability[1]}"
    exact_sm_supported = device_sm in arch_set
    ptx_supported = ptx_arch in arch_set

    log("gpu", gpu_name)
    log("capability", ".".join(map(str, capability)))
    log("device_sm", device_sm)
    log("torch_arch_list", arch_list)
    log("device_sm_supported_by_torch", exact_sm_supported)
    log("device_ptx_supported_by_torch", ptx_supported)

    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name,compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip()
        log("nvidia_smi", smi)
    except Exception as exc:
        log("nvidia_smi_unavailable", repr(exc))

    if not exact_sm_supported:
        raise RuntimeError(
            f"{device_sm} is not in torch.cuda.get_arch_list(); "
            f"compiled arches are {arch_list}"
        )

    x = torch.rand((1000, 3), device="cuda")
    batch = torch.zeros(1000, dtype=torch.long, device="cuda")
    edge_index = radius_graph(x, r=0.1, batch=batch, max_num_neighbors=128)
    if edge_index.device.type != "cuda":
        raise RuntimeError(f"radius_graph returned non-CUDA tensor: {edge_index.device}")
    log("radius_graph_cuda_ok", {"shape": tuple(edge_index.shape), "device": str(edge_index.device)})

    import numpy as np
    from nvidia import nvcomp

    payload = b"bspp-container-nvcomp-zstd-smoke" * 1024
    codec = nvcomp.Codec(algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW)
    encoded = codec.encode([nvcomp.as_array(np.frombuffer(payload, dtype=np.uint8))])
    compressed = bytes(encoded[0].cpu())
    log("nvcomp_zstd_raw_encode_ok", {"bytes": len(compressed)})
    decoded = subprocess.check_output(["zstd", "-q", "-d", "-c"], input=compressed)
    if decoded != payload:
        raise RuntimeError("nvCOMP Zstd RAW payload did not round-trip through zstd")
    log("nvcomp_zstd_cli_decode_ok", True)

    try:
        from afdb_integration_kit.gpu import clashes
    except Exception as exc:
        log("afdb_fallback_check_skipped", repr(exc))
    else:
        fallback = getattr(clashes, "_RADIUS_GRAPH_USE_FALLBACK", None)
        log("afdb_radius_graph_fallback", fallback)
        if fallback:
            raise RuntimeError("AFDB radius_graph fallback is active")

    record["status"] = "passed"
except Exception as exc:
    record["error"] = repr(exc)
    record["traceback"] = traceback.format_exc()
    write_record()
    raise
else:
    write_record()
PY
