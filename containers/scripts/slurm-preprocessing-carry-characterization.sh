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

set -euo pipefail

# Self-submitting and self-monitoring real-kernel characterization for #80.
# All paths and digests are explicit so the evidence binds one qualified tuple.

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "missing required environment: ${name}" >&2; exit 2; }
}

for name in \
  BSPP_CARRY_CLUSTER_PROFILE BSPP_CARRY_IMAGE BSPP_CARRY_IMAGE_SHA256 \
  BSPP_CARRY_OCI_DIGEST BSPP_CARRY_SOURCE_BUNDLE BSPP_CARRY_SOURCE_BUNDLE_ID \
  BSPP_CARRY_SOURCE_BUNDLE_SHA256 BSPP_CARRY_DATABASE_ROOT BSPP_CARRY_WORK_ROOT \
  BSPP_CARRY_INPUT_SHA256; do
  require_env "$name"
done

BSPP_CARRY_RUNTIME_CONTRACT_ID="${BSPP_CARRY_RUNTIME_CONTRACT_ID:-6826d13f71a176d5ac483f42d85a9967485f8c171a96e7d20c7d6350efbf3796}"
BSPP_CARRY_ADAPTER_VERSION="${BSPP_CARRY_ADAPTER_VERSION:-preprocessing-scientific-backend-v3}"
export BSPP_CARRY_RUNTIME_CONTRACT_ID BSPP_CARRY_ADAPTER_VERSION
[[ "$BSPP_CARRY_RUNTIME_CONTRACT_ID" == "6826d13f71a176d5ac483f42d85a9967485f8c171a96e7d20c7d6350efbf3796" ]] || {
  echo "unsupported runtime contract id" >&2; exit 2;
}
[[ "$BSPP_CARRY_ADAPTER_VERSION" == "preprocessing-scientific-backend-v3" ]] || {
  echo "unsupported preprocessing adapter version" >&2; exit 2;
}

mkdir -p -- "$BSPP_CARRY_WORK_ROOT"

if [[ "${BSPP_CARRY_CHARACTERIZATION_MODE:-submit}" != "worker" ]]; then
  job_id="$(sbatch \
    --parsable \
    --job-name=bspp-carry-characterization \
    --partition="${BSPP_CARRY_PARTITION:-gpu}" \
    --account="${BSPP_CARRY_ACCOUNT:?set BSPP_CARRY_ACCOUNT}" \
    --nodes=1 --ntasks=1 \
    --cpus-per-task="${BSPP_CARRY_CPUS:-30}" \
    --mem="${BSPP_CARRY_MEMORY:-128G}" \
    --time="${BSPP_CARRY_TIME:-00:45:00}" \
    --gres="${BSPP_CARRY_GRES:-gpu:1}" \
    --output="$BSPP_CARRY_WORK_ROOT/slurm-%j.out" \
    --error="$BSPP_CARRY_WORK_ROOT/slurm-%j.err" \
    --export=ALL,BSPP_CARRY_CHARACTERIZATION_MODE=worker \
    "$0")"
  job_id="${job_id%%;*}"
  echo "preprocessing_carry_characterization_job=${job_id}"
  while squeue -h -j "$job_id" | grep -q .; do
    squeue -h -j "$job_id" -o 'job=%i state=%T elapsed=%M node=%N'
    sleep 15
  done
  state=""
  for _ in $(seq 1 12); do
    state="$(sacct -X -j "$job_id" --noheader --parsable2 --format=JobIDRaw,State | \
      awk -F'|' -v id="$job_id" '$1 == id {sub(/ .*/, "", $2); print $2; exit}')"
    [[ -n "$state" ]] && break
    sleep 5
  done
  sacct -X -j "$job_id" --noheader --parsable2 --format=JobIDRaw,State,ExitCode,Elapsed,NodeList
  tail -n 120 "$BSPP_CARRY_WORK_ROOT/slurm-${job_id}.out" 2>/dev/null || true
  tail -n 120 "$BSPP_CARRY_WORK_ROOT/slurm-${job_id}.err" 2>/dev/null || true
  [[ "$state" == "COMPLETED" ]] || exit 1
  [[ -s "$BSPP_CARRY_WORK_ROOT/evidence.json" ]] || exit 1
  exit 0
fi

require_env SLURM_JOB_ID
require_env SLURMD_NODENAME
require_env CUDA_VISIBLE_DEVICES

observed_image_sha="$(sha256sum "$BSPP_CARRY_IMAGE" | awk '{print $1}')"
observed_source_sha="$(sha256sum "$BSPP_CARRY_SOURCE_BUNDLE" | awk '{print $1}')"
[[ "$observed_image_sha" == "$BSPP_CARRY_IMAGE_SHA256" ]] || { echo "image SHA-256 mismatch" >&2; exit 3; }
[[ "$observed_source_sha" == "$BSPP_CARRY_SOURCE_BUNDLE_SHA256" ]] || {
  echo "Source Bundle SHA-256 mismatch" >&2; exit 3;
}

fixture="$BSPP_CARRY_WORK_ROOT/fixture"
destination="$BSPP_CARRY_WORK_ROOT/evidence.json"
[[ ! -e "$fixture" ]] || { echo "characterization fixture already exists" >&2; exit 4; }
[[ ! -e "$destination" ]] || { echo "characterization evidence already exists" >&2; exit 4; }
mkdir -- "$fixture"
cat > "$fixture/remaining.fa" <<'EOF'
>AFDB_alpha
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL:MNKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE
>AFDB_mu
MNKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE:MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL
EOF
fixture_sha="$(sha256sum "$fixture/remaining.fa" | awk '{print $1}')"
[[ "$fixture_sha" == "$BSPP_CARRY_INPUT_SHA256" ]] || {
  echo "characterization fixture SHA-256 mismatch" >&2; exit 4;
}
mkdir -- "$fixture/search-output" "$fixture/output"

host_job_id="$fixture/host-slurm-job-id.txt"
host_node="$fixture/host-slurmd-nodename.txt"
host_visible_devices="$fixture/host-cuda-visible-devices.txt"
host_gpu_rows="$fixture/host-nvidia-smi.csv"
wrapper_sha="$fixture/slurm-wrapper-sha256.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$host_job_id"
printf '%s\n' "$SLURMD_NODENAME" > "$host_node"
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$host_visible_devices"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable in the allocation" >&2; exit 5; }
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader > "$host_gpu_rows" || {
  echo "host nvidia-smi query failed" >&2; exit 5;
}
[[ -s "$host_gpu_rows" ]] || { echo "host nvidia-smi query returned blank output" >&2; exit 5; }
awk 'NF == 0 {exit 1} END {if (NR == 0) exit 1}' "$host_gpu_rows" || {
  echo "host nvidia-smi query contains blank rows" >&2; exit 5;
}
IFS=',' read -r -a host_devices <<< "$CUDA_VISIBLE_DEVICES"
for device in "${host_devices[@]}"; do
  [[ "$device" =~ [^[:space:]] ]] || { echo "host CUDA_VISIBLE_DEVICES contains a blank entry" >&2; exit 5; }
done
host_gpu_count="$(awk 'NF != 0 {count++} END {print count + 0}' "$host_gpu_rows")"
[[ "${#host_devices[@]}" -eq "$host_gpu_count" ]] || {
  echo "host GPU query-row count does not match CUDA_VISIBLE_DEVICES cardinality" >&2; exit 5;
}
sha256sum "$0" | awk '{print $1}' > "$wrapper_sha"
chmod 0444 "$host_job_id" "$host_node" "$host_visible_devices" "$host_gpu_rows" "$wrapper_sha"

mounts="${CARRY_MOUNTS:-/lustre:/lustre}"
srun \
  --container-image="$BSPP_CARRY_IMAGE" \
  --container-mounts="$mounts" \
  --no-container-mount-home \
  /usr/local/bin/entrypoint.sh \
  /bin/bash /opt/bspp/bin/bspp-preprocessing-carry-characterization "$fixture"

srun \
  --container-image="$BSPP_CARRY_IMAGE" \
  --container-mounts="$mounts" \
  --no-container-mount-home \
  /usr/local/bin/entrypoint.sh \
  /opt/bspp/environment/bin/python - \
  "$fixture" "$destination" \
  "$host_job_id" "$host_node" "$host_visible_devices" "$host_gpu_rows" "$wrapper_sha" <<'PY'
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath

fixture = Path(sys.argv[1])
destination = Path(sys.argv[2])
host_job_id_path = Path(sys.argv[3])
host_node_path = Path(sys.argv[4])
host_visible_devices_path = Path(sys.argv[5])
host_gpu_rows_path = Path(sys.argv[6])
wrapper_sha_path = Path(sys.argv[7])

expected_host_paths = (
    (host_job_id_path, fixture / "host-slurm-job-id.txt"),
    (host_node_path, fixture / "host-slurmd-nodename.txt"),
    (host_visible_devices_path, fixture / "host-cuda-visible-devices.txt"),
    (host_gpu_rows_path, fixture / "host-nvidia-smi.csv"),
    (wrapper_sha_path, fixture / "slurm-wrapper-sha256.txt"),
)
if any(observed != expected for observed, expected in expected_host_paths):
    raise SystemExit("host evidence paths are not bound to the characterization fixture")

def read_single_line(path: Path, label: str) -> str:
    lines = path.read_text().splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise SystemExit(f"{label} must contain exactly one nonblank line")
    return lines[0]

def visible_devices(value: str, label: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in value.split(","))
    if not devices or any(not part for part in devices):
        raise SystemExit(f"{label} is not a nonblank CUDA_VISIBLE_DEVICES list")
    return devices

def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

host_job_id = read_single_line(host_job_id_path, "host Slurm job id")
host_node = read_single_line(host_node_path, "host Slurm node")
host_visible = read_single_line(host_visible_devices_path, "host CUDA_VISIBLE_DEVICES")
host_rows_text = host_gpu_rows_path.read_text()
host_rows = host_rows_text.splitlines()
if not host_rows or any(not row.strip() for row in host_rows):
    raise SystemExit("host GPU evidence must contain only nonblank rows")
parsed_host_rows = []
for row in host_rows:
    fields = tuple(field.strip() for field in row.split(",", 2))
    if len(fields) != 3 or any(not field for field in fields):
        raise SystemExit("host GPU evidence row must contain name, UUID, and driver version")
    parsed_host_rows.append({"name": fields[0], "uuid": fields[1], "driver_version": fields[2], "raw": row})
if host_job_id != os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("captured host job id does not match SLURM_JOB_ID")
if host_node != os.environ.get("SLURMD_NODENAME"):
    raise SystemExit("captured host node does not match SLURMD_NODENAME")
host_devices = visible_devices(host_visible, "host CUDA_VISIBLE_DEVICES")
if len(host_rows) != len(host_devices):
    raise SystemExit("host GPU query-row count does not match visible-device cardinality")

probe_path = fixture / "cuda-driver-evidence.json"
probe = json.loads(probe_path.read_text())
probe_keys = {
    "schema_version", "loader_soname", "compat_directory", "compat_soname_path",
    "compat_library_resolved_target", "mapped_library_path", "mapped_library_resolved_target",
    "maps_line", "driver_api_version", "cuda_device_count", "cu_driver_get_version_return_code",
    "cu_init_return_code", "cu_device_get_count_return_code", "cuda_visible_devices",
    "effective_ld_library_path", "slurm_job_id", "slurmd_nodename", "probe_sha256", "helper_sha256",
}
if set(probe) != probe_keys or probe["schema_version"] != 1:
    raise SystemExit("CUDA driver evidence schema mismatch")
if probe["slurm_job_id"] != host_job_id or probe["slurmd_nodename"] != host_node:
    raise SystemExit("container CUDA evidence is not bound to the host job and node")
container_devices = visible_devices(probe["cuda_visible_devices"], "container CUDA_VISIBLE_DEVICES")
if probe["cuda_device_count"] != len(container_devices):
    raise SystemExit("CUDA Driver API device count does not match container-visible cardinality")
for key in ("cu_driver_get_version_return_code", "cu_init_return_code", "cu_device_get_count_return_code"):
    if probe[key] != 0:
        raise SystemExit(f"CUDA Driver API call failed: {key}")
if not isinstance(probe["driver_api_version"], int) or probe["driver_api_version"] <= 0:
    raise SystemExit("CUDA user-mode driver API version is invalid")
compat_directory = PurePosixPath("/usr/local/cuda-12.6/compat")
compat_soname = compat_directory / "libcuda.so.1"
if probe["loader_soname"] != "libcuda.so.1":
    raise SystemExit("CUDA loader soname mismatch")
if PurePosixPath(probe["compat_directory"]) != compat_directory:
    raise SystemExit("CUDA compatibility directory mismatch")
if PurePosixPath(probe["compat_soname_path"]) != compat_soname:
    raise SystemExit("CUDA compatibility soname path mismatch")
compat_target = PurePosixPath(probe["compat_library_resolved_target"])
mapped_path = PurePosixPath(probe["mapped_library_path"])
mapped_target = PurePosixPath(probe["mapped_library_resolved_target"])
if (
    not compat_target.is_absolute()
    or ".." in compat_target.parts
    or not compat_target.is_relative_to(compat_directory)
):
    raise SystemExit("persisted CUDA compatibility target escapes its directory")
if mapped_target != compat_target:
    raise SystemExit("mapped CUDA library is not the resolved compatibility-library target")
if mapped_path != mapped_target:
    raise SystemExit("mapped CUDA library path is not its resolved compatibility-library target")
if probe["mapped_library_path"] not in probe["maps_line"]:
    raise SystemExit("persisted CUDA maps line does not contain the mapped path")
loader_entries = probe["effective_ld_library_path"].split(":")
if not loader_entries or PurePosixPath(loader_entries[0]) != compat_directory:
    raise SystemExit("CUDA compatibility directory was not first in effective LD_LIBRARY_PATH")
sha_pattern = re.compile(r"[0-9a-f]{64}")
for key in ("probe_sha256", "helper_sha256"):
    if not isinstance(probe[key], str) or sha_pattern.fullmatch(probe[key]) is None:
        raise SystemExit(f"invalid baked source hash: {key}")
wrapper_sha = read_single_line(wrapper_sha_path, "Slurm wrapper SHA-256")
if sha_pattern.fullmatch(wrapper_sha) is None:
    raise SystemExit("invalid Slurm wrapper SHA-256")

expected = ("AFDB_alpha.a3m", "AFDB_mu.a3m")
search_output = fixture / "search-output"
published_output = fixture / "output"
search_entries = tuple(sorted(search_output.iterdir(), key=lambda item: item.name))
expected_search_inventory = tuple(sorted((*expected, "2.a3m", "3.a3m")))
if tuple(item.name for item in search_entries) != expected_search_inventory:
    raise SystemExit("real kernel did not produce the pinned raw search inventory")
placeholder_bytes = {"2.a3m": b"#49\t1\n", "3.a3m": b"#36\t1\n"}
outputs = []
search_inventory = []

def validate_a3m(data: bytes, *, label: str) -> int:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SystemExit(f"characterized A3M is not UTF-8: {label}") from exc
    if lines and lines[0].startswith("#"):
        match = re.fullmatch(r"#([1-9]\d*(?:,[1-9]\d*)*)\t([1-9]\d*(?:,[1-9]\d*)*)", lines[0])
        if match is None or len(match.group(1).split(",")) != len(match.group(2).split(",")):
            raise SystemExit(f"malformed characterized multimer metadata: {label}")
        lines = lines[1:]
    if not lines or not lines[0].startswith(">"):
        raise SystemExit(f"malformed characterized A3M: {label}")
    has_header = False
    current_has_sequence = False
    record_count = 0
    for line in lines:
        if line.startswith(">"):
            if not line[1:].strip() or (has_header and not current_has_sequence):
                raise SystemExit(f"empty characterized A3M record: {label}")
            has_header = True
            current_has_sequence = False
            record_count += 1
        elif line.strip():
            if not has_header:
                raise SystemExit(f"characterized sequence precedes header: {label}")
            current_has_sequence = True
    if not current_has_sequence:
        raise SystemExit(f"characterized final A3M record is empty: {label}")
    return record_count

for path in search_entries:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size <= 0:
        raise SystemExit(f"invalid raw search artifact: {path}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    search_inventory.append({"member_name": path.name, "size_bytes": len(data), "sha256": digest})
    if path.name in placeholder_bytes:
        if data != placeholder_bytes[path.name]:
            raise SystemExit(f"unexpected pinned ColabFold placeholder content: {path}")
        continue
    sequence_count = validate_a3m(data, label=str(path))
    published = published_output / path.name
    with published.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    outputs.append({
        "member_name": path.name,
        "size_bytes": len(data),
        "sha256": digest,
        "sequence_count": sequence_count,
    })
if tuple(entry["member_name"] for entry in outputs) != expected:
    raise SystemExit("real kernel did not publish the exact complement inventory")
remaining_input_sha256 = sha256_file(fixture / "remaining.fa")
if remaining_input_sha256 != os.environ["BSPP_CARRY_INPUT_SHA256"]:
    raise SystemExit("remaining input no longer matches the pinned fixture")
gpuserver_log = fixture / "gpuserver.log"
gpuserver_info = gpuserver_log.lstat()
if not stat.S_ISREG(gpuserver_info.st_mode) or gpuserver_log.is_symlink() or gpuserver_info.st_size <= 0:
    raise SystemExit("gpuserver log is missing, empty, or not a regular file")

payload = {
    "schema_version": 2,
    "characterization": "preprocessing-carry-complement-search-v2",
    "outcome": "passed",
    "cluster_profile": os.environ["BSPP_CARRY_CLUSTER_PROFILE"],
    "runtime_contract_id": os.environ["BSPP_CARRY_RUNTIME_CONTRACT_ID"],
    "adapter_version": os.environ["BSPP_CARRY_ADAPTER_VERSION"],
    "cluster_image_path": os.environ["BSPP_CARRY_IMAGE"],
    "cluster_image_sha256": os.environ["BSPP_CARRY_IMAGE_SHA256"],
    "oci_digest": os.environ["BSPP_CARRY_OCI_DIGEST"],
    "source_bundle_id": os.environ["BSPP_CARRY_SOURCE_BUNDLE_ID"],
    "source_bundle_path": os.environ["BSPP_CARRY_SOURCE_BUNDLE"],
    "source_bundle_sha256": os.environ["BSPP_CARRY_SOURCE_BUNDLE_SHA256"],
    "database_root": os.environ["BSPP_CARRY_DATABASE_ROOT"],
    "remaining_input_sha256": remaining_input_sha256,
    "raw_search_inventory": search_inventory,
    "outputs": outputs,
    "slurm_job_id": host_job_id,
    "slurmd_nodename": host_node,
    "host_gpu_evidence": {
        "cuda_visible_devices": host_visible,
        "query_rows": parsed_host_rows,
    },
    "cuda_driver_evidence": probe,
    "effective_ld_library_path": probe["effective_ld_library_path"],
    "gpuserver_log_sha256": sha256_file(gpuserver_log),
    "source_hashes": {
        "slurm_wrapper_sha256": wrapper_sha,
        "baked_helper_sha256": probe["helper_sha256"],
        "baked_probe_sha256": probe["probe_sha256"],
    },
    "search_supervision": {
        "gpuserver_warmup_seconds": 60,
        "search_timeout_seconds": 1800,
        "search_kill_after_seconds": 10,
        "search_threads": 64,
    },
}
temporary = destination.with_suffix(".tmp")
with temporary.open("x") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
try:
    os.link(temporary, destination)
except OSError:
    temporary.unlink(missing_ok=True)
    raise
else:
    temporary.unlink()
PY
