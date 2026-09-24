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

# =============================================================================
# Build and push a container variant to a configured OCI registry (single-arch).
#
# Usage:
#   ./containers/scripts/push.sh postprocessing
#   ./containers/scripts/push.sh preprocessing
#   ./containers/scripts/push.sh folding <runtime|colabfold|openfold-cli|bioir|all>
#   ./containers/scripts/push.sh all
#   ./containers/scripts/push.sh postprocessing --no-cache
#   ./containers/scripts/push.sh postprocessing --include-postprocessing-internal
#
# Single-variant modes (postprocessing, preprocessing, folding):
#   Unrecognized arguments pass through to docker build (existing contract).
#   --include-postprocessing-internal is honored ONLY for the `postprocessing`
#   variant: after the public image is built and pushed, the internal image is
#   built and tagged locally via containers/nvidia/build-internal.sh (build + tag
#   only, NEVER pushed by design). Passing it with any other single variant is
#   a fail-fast error.
#
# One canonical command for the FULL image matrix (preflight checks, dynamic
# discovery, per-image smoke verification, evidence capture, fail-fast):
#   ./containers/scripts/push.sh all [--dry-run]
#                                    [--include-postprocessing-internal] [--evidence-dir DIR]
#
#   `all` covers the dedicated images: preprocessing, postprocessing, and every
#   containers/folding/*/ image. --include-postprocessing-internal additionally
#   builds the postprocessing-internal image via containers/nvidia/build-internal.sh
#   when those assets exist in the tree — build + tag only, never pushed by design.
#
# Environment:
#   BSPP_REGISTRY          (required) OCI registry host, e.g.
#                           registry.example.com:5005
#   BSPP_IMAGE_REPOSITORY  (optional) image repository name;
#                           default: bspp-orchestration
#   TOOLKIT_REPO            (optional) public Git URL of the toolkit repository;
#                           defaults to the public upstream toolkit
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BSPP_REGISTRY="${BSPP_REGISTRY:?BSPP_REGISTRY is required; set it to your OCI registry host (e.g. registry.example.com:5005)}"
BSPP_IMAGE_REPOSITORY="${BSPP_IMAGE_REPOSITORY:-bspp-orchestration}"
REPO="${BSPP_REGISTRY}/${BSPP_IMAGE_REPOSITORY}"
PREPROCESSING_LOCAL_IMAGE="bspp-orchestration:preprocessing"
PREPROCESSING_REGISTRY_IMAGE="${REPO}:preprocessing"
PREPROCESSING_BUILD_RECORD="${REPO_ROOT}/containers/preprocessing/dist/build-record.json"
POSTPROCESSING_LOCAL_IMAGE="bspp-orchestration:postprocessing"
POSTPROCESSING_REGISTRY_IMAGE="${REPO}:postprocessing"
POSTPROCESSING_BUILD_RECORD="${REPO_ROOT}/containers/postprocessing/dist/build-record.json"

build_and_push_preprocessing() {
    CONTAINER_ENGINE=docker \
        "${REPO_ROOT}/containers/preprocessing/build.sh" "$PREPROCESSING_LOCAL_IMAGE" "$@"

    [[ -f "$PREPROCESSING_BUILD_RECORD" ]] || {
        echo "ERROR: preprocessing build did not write ${PREPROCESSING_BUILD_RECORD}" >&2
        return 1
    }

    local record_image record_image_id local_image_id registry_image_id
    record_image="$(jq -er '.image | select(type == "string")' "$PREPROCESSING_BUILD_RECORD")"
    record_image_id="$(jq -er '.image_id | select(type == "string")' "$PREPROCESSING_BUILD_RECORD")"
    [[ "$record_image" == "$PREPROCESSING_LOCAL_IMAGE" ]] || {
        echo "ERROR: preprocessing build record names an unexpected local image" >&2
        return 1
    }
    [[ "$record_image_id" =~ ^sha256:[0-9a-f]{64}$ ]] || {
        echo "ERROR: preprocessing build record has a malformed local image ID" >&2
        return 1
    }
    if jq -e 'has("registry_image") or has("oci_digest")' "$PREPROCESSING_BUILD_RECORD" >/dev/null; then
        echo "ERROR: preprocessing build record contains registry identity before push" >&2
        return 1
    fi

    local_image_id="$(docker image inspect --format '{{.Id}}' "$PREPROCESSING_LOCAL_IMAGE")"
    [[ "$local_image_id" == "$record_image_id" ]] || {
        echo "ERROR: preprocessing local image ID does not match its fresh build record" >&2
        return 1
    }

    docker tag "$record_image_id" "$PREPROCESSING_REGISTRY_IMAGE"
    registry_image_id="$(docker image inspect --format '{{.Id}}' "$PREPROCESSING_REGISTRY_IMAGE")"
    [[ "$registry_image_id" == "$record_image_id" ]] || {
        echo "ERROR: preprocessing registry tag does not point to the freshly built image ID" >&2
        return 1
    }

    echo "Pushing ${PREPROCESSING_REGISTRY_IMAGE} ..."
    docker push "$PREPROCESSING_REGISTRY_IMAGE"

    local inspect_json digest_candidates candidate digest_suffix oci_digest record_before record_after_without_transport
    local -a canonical_digests=()
    inspect_json="$(docker image inspect "$PREPROCESSING_REGISTRY_IMAGE")"
    if ! digest_candidates="$(
        jq -r 'if type == "array" and length == 1 and (.[0].RepoDigests | type) == "array"
                  then .[0].RepoDigests[] | select(type == "string")
                  else empty end' <<<"$inspect_json" | sort -u
    )"; then
        echo "ERROR: preprocessing push returned invalid Docker inspect JSON" >&2
        return 1
    fi
    while IFS= read -r candidate; do
        [[ -n "$candidate" ]] || continue
        if [[ "$candidate" == "${REPO}@"* ]]; then
            digest_suffix="${candidate#"${REPO}@"}"
            [[ "$digest_suffix" =~ ^sha256:[0-9a-f]{64}$ ]] || {
                echo "ERROR: preprocessing push returned a malformed canonical RepoDigest" >&2
                return 1
            }
            canonical_digests+=("$candidate")
        fi
    done <<<"$digest_candidates"

    [[ ${#canonical_digests[@]} -eq 1 ]] || {
        echo "ERROR: preprocessing push requires exactly one canonical RepoDigest; found ${#canonical_digests[@]}" >&2
        return 1
    }
    oci_digest="${canonical_digests[0]#"${REPO}@"}"

    record_before="$(jq -cS . "$PREPROCESSING_BUILD_RECORD")"
    local record_tmp
    record_tmp="$(mktemp "${PREPROCESSING_BUILD_RECORD}.tmp.XXXXXX")"
    if ! jq \
        --arg registry_image "$PREPROCESSING_REGISTRY_IMAGE" \
        --arg oci_digest "$oci_digest" \
        '. + {registry_image: $registry_image, oci_digest: $oci_digest}' \
        "$PREPROCESSING_BUILD_RECORD" >"$record_tmp"; then
        rm -f -- "$record_tmp"
        return 1
    fi
    record_after_without_transport="$(jq -cS 'del(.registry_image, .oci_digest)' "$record_tmp")"
    if [[ "$record_after_without_transport" != "$record_before" ]] \
        || ! jq -e \
            --arg registry_image "$PREPROCESSING_REGISTRY_IMAGE" \
            --arg oci_digest "$oci_digest" \
            '.registry_image == $registry_image and .oci_digest == $oci_digest' \
            "$record_tmp" >/dev/null; then
        rm -f -- "$record_tmp"
        echo "ERROR: preprocessing registry identity did not preserve the build record" >&2
        return 1
    fi
    if ! mv -f -- "$record_tmp" "$PREPROCESSING_BUILD_RECORD"; then
        rm -f -- "$record_tmp"
        echo "ERROR: could not atomically update the preprocessing build record" >&2
        return 1
    fi
    echo "$PREPROCESSING_BUILD_RECORD"
}

build_and_push_postprocessing() {
    CONTAINER_ENGINE=docker \
        "${REPO_ROOT}/containers/postprocessing/build.sh" "$POSTPROCESSING_LOCAL_IMAGE" "$@"

    [[ -f "$POSTPROCESSING_BUILD_RECORD" ]] || {
        echo "ERROR: postprocessing build did not write ${POSTPROCESSING_BUILD_RECORD}" >&2
        return 1
    }

    local record_image record_image_id local_image_id registry_image_id
    record_image="$(jq -er '.image | select(type == "string")' "$POSTPROCESSING_BUILD_RECORD")"
    record_image_id="$(jq -er '.image_id | select(type == "string")' "$POSTPROCESSING_BUILD_RECORD")"
    [[ "$record_image" == "$POSTPROCESSING_LOCAL_IMAGE" ]] || {
        echo "ERROR: postprocessing build record names an unexpected local image" >&2
        return 1
    }
    [[ "$record_image_id" =~ ^sha256:[0-9a-f]{64}$ ]] || {
        echo "ERROR: postprocessing build record has a malformed local image ID" >&2
        return 1
    }
    if jq -e 'has("registry_image") or has("oci_digest")' "$POSTPROCESSING_BUILD_RECORD" >/dev/null; then
        echo "ERROR: postprocessing build record contains registry identity before push" >&2
        return 1
    fi

    local_image_id="$(docker image inspect --format '{{.Id}}' "$POSTPROCESSING_LOCAL_IMAGE")"
    [[ "$local_image_id" == "$record_image_id" ]] || {
        echo "ERROR: postprocessing local image ID does not match its fresh build record" >&2
        return 1
    }

    # Gate publication on the composition smoke (baked orchestration source):
    # a failing image must never reach the registry.
    "${REPO_ROOT}/containers/postprocessing/smoke-local.sh" "$POSTPROCESSING_LOCAL_IMAGE" || {
        echo "ERROR: postprocessing composition smoke failed; refusing to push" >&2
        return 1
    }

    docker tag "$record_image_id" "$POSTPROCESSING_REGISTRY_IMAGE"
    registry_image_id="$(docker image inspect --format '{{.Id}}' "$POSTPROCESSING_REGISTRY_IMAGE")"
    [[ "$registry_image_id" == "$record_image_id" ]] || {
        echo "ERROR: postprocessing registry tag does not point to the freshly built image ID" >&2
        return 1
    }

    echo "Pushing ${POSTPROCESSING_REGISTRY_IMAGE} ..."
    docker push "$POSTPROCESSING_REGISTRY_IMAGE"

    local inspect_json digest_candidates candidate digest_suffix oci_digest record_before record_after_without_transport
    local -a canonical_digests=()
    inspect_json="$(docker image inspect "$POSTPROCESSING_REGISTRY_IMAGE")"
    if ! digest_candidates="$(
        jq -r 'if type == "array" and length == 1 and (.[0].RepoDigests | type) == "array"
                  then .[0].RepoDigests[] | select(type == "string")
                  else empty end' <<<"$inspect_json" | sort -u
    )"; then
        echo "ERROR: postprocessing push returned invalid Docker inspect JSON" >&2
        return 1
    fi
    while IFS= read -r candidate; do
        [[ -n "$candidate" ]] || continue
        if [[ "$candidate" == "${REPO}@"* ]]; then
            digest_suffix="${candidate#"${REPO}@"}"
            [[ "$digest_suffix" =~ ^sha256:[0-9a-f]{64}$ ]] || {
                echo "ERROR: postprocessing push returned a malformed canonical RepoDigest" >&2
                return 1
            }
            canonical_digests+=("$candidate")
        fi
    done <<<"$digest_candidates"

    [[ ${#canonical_digests[@]} -eq 1 ]] || {
        echo "ERROR: postprocessing push requires exactly one canonical RepoDigest; found ${#canonical_digests[@]}" >&2
        return 1
    }
    oci_digest="${canonical_digests[0]#"${REPO}@"}"

    record_before="$(jq -cS . "$POSTPROCESSING_BUILD_RECORD")"
    local record_tmp
    record_tmp="$(mktemp "${POSTPROCESSING_BUILD_RECORD}.tmp.XXXXXX")"
    if ! jq \
        --arg registry_image "$POSTPROCESSING_REGISTRY_IMAGE" \
        --arg oci_digest "$oci_digest" \
        '. + {registry_image: $registry_image, oci_digest: $oci_digest}' \
        "$POSTPROCESSING_BUILD_RECORD" >"$record_tmp"; then
        rm -f -- "$record_tmp"
        return 1
    fi
    record_after_without_transport="$(jq -cS 'del(.registry_image, .oci_digest)' "$record_tmp")"
    if [[ "$record_after_without_transport" != "$record_before" ]] \
        || ! jq -e \
            --arg registry_image "$POSTPROCESSING_REGISTRY_IMAGE" \
            --arg oci_digest "$oci_digest" \
            '.registry_image == $registry_image and .oci_digest == $oci_digest' \
            "$record_tmp" >/dev/null; then
        rm -f -- "$record_tmp"
        echo "ERROR: postprocessing registry identity did not preserve the build record" >&2
        return 1
    fi
    if ! mv -f -- "$record_tmp" "$POSTPROCESSING_BUILD_RECORD"; then
        rm -f -- "$record_tmp"
        echo "ERROR: could not atomically update the postprocessing build record" >&2
        return 1
    fi
    echo "$POSTPROCESSING_BUILD_RECORD"
}

build_and_push_folding_image() {
    local image="$1"; shift
    local local_image="bspp-orchestration:folding-${image}"
    local registry_image="${REPO}:folding-${image}"
    local build_record="${REPO_ROOT}/containers/folding/${image}/dist/build-record.json"

    CONTAINER_ENGINE=docker \
        "${REPO_ROOT}/containers/folding/${image}/build.sh" "$local_image" "$@"

    [[ -f "$build_record" ]] || {
        echo "ERROR: folding ${image} build did not write ${build_record}" >&2
        return 1
    }

    local record_image record_image_id local_image_id registry_image_id
    record_image="$(jq -er '.image | select(type == "string")' "$build_record")"
    record_image_id="$(jq -er '.image_id | select(type == "string")' "$build_record")"
    [[ "$record_image" == "$local_image" ]] || {
        echo "ERROR: folding ${image} build record names an unexpected local image" >&2
        return 1
    }
    [[ "$record_image_id" =~ ^sha256:[0-9a-f]{64}$ ]] || {
        echo "ERROR: folding ${image} build record has a malformed local image ID" >&2
        return 1
    }
    if jq -e 'has("registry_image") or has("oci_digest")' "$build_record" >/dev/null; then
        echo "ERROR: folding ${image} build record contains registry identity before push" >&2
        return 1
    fi

    local_image_id="$(docker image inspect --format '{{.Id}}' "$local_image")"
    [[ "$local_image_id" == "$record_image_id" ]] || {
        echo "ERROR: folding ${image} local image ID does not match its fresh build record" >&2
        return 1
    }

    docker tag "$record_image_id" "$registry_image"
    registry_image_id="$(docker image inspect --format '{{.Id}}' "$registry_image")"
    [[ "$registry_image_id" == "$record_image_id" ]] || {
        echo "ERROR: folding ${image} registry tag does not point to the freshly built image ID" >&2
        return 1
    }

    echo "Pushing ${registry_image} ..."
    docker push "$registry_image"

    local inspect_json digest_candidates candidate digest_suffix oci_digest record_before record_after_without_transport
    local -a canonical_digests=()
    inspect_json="$(docker image inspect "$registry_image")"
    if ! digest_candidates="$(
        jq -r 'if type == "array" and length == 1 and (.[0].RepoDigests | type) == "array"
                  then .[0].RepoDigests[] | select(type == "string")
                  else empty end' <<<"$inspect_json" | sort -u
    )"; then
        echo "ERROR: folding ${image} push returned invalid Docker inspect JSON" >&2
        return 1
    fi
    while IFS= read -r candidate; do
        [[ -n "$candidate" ]] || continue
        if [[ "$candidate" == "${REPO}@"* ]]; then
            digest_suffix="${candidate#"${REPO}@"}"
            [[ "$digest_suffix" =~ ^sha256:[0-9a-f]{64}$ ]] || {
                echo "ERROR: folding ${image} push returned a malformed canonical RepoDigest" >&2
                return 1
            }
            canonical_digests+=("$candidate")
        fi
    done <<<"$digest_candidates"

    [[ ${#canonical_digests[@]} -eq 1 ]] || {
        echo "ERROR: folding ${image} push requires exactly one canonical RepoDigest; found ${#canonical_digests[@]}" >&2
        return 1
    }
    oci_digest="${canonical_digests[0]#"${REPO}@"}"

    record_before="$(jq -cS . "$build_record")"
    local record_tmp
    record_tmp="$(mktemp "${build_record}.tmp.XXXXXX")"
    if ! jq \
        --arg registry_image "$registry_image" \
        --arg oci_digest "$oci_digest" \
        '. + {registry_image: $registry_image, oci_digest: $oci_digest}' \
        "$build_record" >"$record_tmp"; then
        rm -f -- "$record_tmp"
        return 1
    fi
    record_after_without_transport="$(jq -cS 'del(.registry_image, .oci_digest)' "$record_tmp")"
    if [[ "$record_after_without_transport" != "$record_before" ]] \
        || ! jq -e \
            --arg registry_image "$registry_image" \
            --arg oci_digest "$oci_digest" \
            '.registry_image == $registry_image and .oci_digest == $oci_digest' \
            "$record_tmp" >/dev/null; then
        rm -f -- "$record_tmp"
        echo "ERROR: folding ${image} registry identity did not preserve the build record" >&2
        return 1
    fi
    if ! mv -f -- "$record_tmp" "$build_record"; then
        rm -f -- "$record_tmp"
        echo "ERROR: could not atomically update the folding ${image} build record" >&2
        return 1
    fi
    echo "$build_record"
}

# ---------------------------------------------------------------------------
# Full-matrix mode (`push.sh all`): preflight, dynamic discovery, per-image
# smoke verification, evidence capture, fail-fast with the exact step named.
# ---------------------------------------------------------------------------

MATRIX_TARGETS=()
declare -A MATRIX_DIGEST=()
declare -A MATRIX_SMOKE=()
declare -A MATRIX_STATUS=()

matrix_discover() {
    MATRIX_TARGETS=()
    if [[ -f "${REPO_ROOT}/containers/preprocessing/build.sh" ]]; then
        MATRIX_TARGETS+=("preprocessing")
    fi
    if [[ -f "${REPO_ROOT}/containers/postprocessing/build.sh" ]]; then
        MATRIX_TARGETS+=("postprocessing")
    fi

    local folding_images=() d img
    for d in "${REPO_ROOT}"/containers/folding/*/; do
        [[ -f "${d}build.sh" ]] || continue
        folding_images+=("$(basename "$d")")
    done
    if [[ ${#folding_images[@]} -gt 0 ]]; then
        # Proven order: the runtime image first, then kernels by name.
        local ordered=()
        for img in ${folding_images[@]+"${folding_images[@]}"}; do
            [[ "$img" == "runtime" ]] && ordered+=("$img")
        done
        while IFS= read -r img; do ordered+=("$img"); done < <(printf '%s\n' ${folding_images[@]+"${folding_images[@]}"} | sort | grep -v '^runtime$' || true)
        for img in ${ordered[@]+"${ordered[@]}"}; do
            MATRIX_TARGETS+=("folding:${img}")
        done
    fi

    if [[ ${INCLUDE_POSTPROCESSING_INTERNAL} -eq 1 ]]; then
        if [[ -f "${REPO_ROOT}/containers/nvidia/build-internal.sh" ]]; then
            MATRIX_TARGETS+=("postprocessing-internal")
        else
            echo "ERROR: --include-postprocessing-internal needs the internal build assets (containers/nvidia/), absent from this tree" >&2
            exit 1
        fi
    fi
}

matrix_plan() {
    echo "push.sh all — plan (registry: ${BSPP_REGISTRY}/${BSPP_IMAGE_REPOSITORY})"
    echo "  evidence: ${EVIDENCE_DIR}"
    local target img
    for target in ${MATRIX_TARGETS[@]+"${MATRIX_TARGETS[@]}"}; do
        case "$target" in
            preprocessing)
                echo "  - preprocessing: push.sh preprocessing (fresh build + push)"
                echo "      smoke: containers/preprocessing/smoke-local.sh bspp-orchestration:preprocessing"
                ;;
            postprocessing)
                echo "  - postprocessing: push.sh postprocessing (fresh build + push)"
                echo "      smoke: containers/postprocessing/smoke-local.sh bspp-orchestration:postprocessing"
                ;;
            folding:*)
                img="${target#folding:}"
                echo "  - folding ${img}: push.sh folding ${img} (fresh build + push)"
                echo "      smoke: containers/folding/${img}/smoke-local.sh bspp-orchestration:folding-${img}"
                ;;
            postprocessing-internal)
                echo "  - postprocessing-internal: containers/nvidia/build-internal.sh (build + tag only; never pushed)"
                ;;
        esac
    done
}

matrix_preflight() {
    echo "== preflight =="
    local tool
    for tool in docker git jq curl uv; do
        command -v "$tool" >/dev/null || { echo "ERROR: required tool not on PATH: $tool" >&2; exit 1; }
    done
    echo "tools: docker git jq curl uv present"

    if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ]]; then
        echo "ERROR: source repository is not clean; commit or stash first" >&2
        exit 1
    fi
    echo "clean-tree: ok"

    # Commit-sync is fail-closed: publishing from a stale or unverifiable
    # checkout is never acceptable.
    if ! git -C "${REPO_ROOT}" fetch origin main 2>/dev/null; then
        echo "ERROR: could not fetch origin main; cannot prove the checkout is current" >&2
        exit 1
    fi
    if ! git -C "${REPO_ROOT}" rev-parse --verify --quiet origin/main >/dev/null; then
        echo "ERROR: origin/main is not resolvable; cannot prove the checkout is current" >&2
        exit 1
    fi
    if ! git -C "${REPO_ROOT}" merge-base --is-ancestor origin/main HEAD; then
        echo "ERROR: origin/main is not an ancestor of HEAD (behind or diverged); sync first:" >&2
        git -C "${REPO_ROOT}" log --oneline --left-right --graph HEAD...origin/main 2>/dev/null | head -10 >&2 || true
        exit 1
    fi
    echo "commit-sync: ok ($(git -C "${REPO_ROOT}" rev-parse --short HEAD) contains origin/main)"

    docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not reachable" >&2; exit 1; }
    echo "docker daemon: ok"

    # Eigen reachability probe (informational): the iPSAE build fetches Eigen
    # from gitlab.com when deps/eigen-3.4.0 is not pre-seeded in the toolkit.
    echo "Eigen reachability probe (informational):"
    if curl -sI --max-time 10 https://gitlab.com >/dev/null 2>&1; then
        echo "  probe: gitlab.com reachable"
    else
        echo "  probe: gitlab.com NOT reachable — if an image build fails fetching Eigen 3.4.0," >&2
        echo "  pre-seed the toolkit's deps/eigen-3.4.0 or set EIGEN_DIR to a local Eigen tree" >&2
    fi
}

matrix_fail() {
    echo "" >&2
    echo "FAILED at: $1 (stopped before the remaining images)" >&2
    matrix_summary
    exit 1
}

matrix_smoke() {
    local target="$1" script="$2" image="$3"
    if [[ -f "$script" ]]; then
        echo "===== smoke: ${target} ====="
        "$script" "$image" || matrix_fail "${target} (smoke)"
        MATRIX_SMOKE[$target]="pass"
    else
        MATRIX_SMOKE[$target]="n/a"
    fi
}

matrix_evidence() {
    local target="$1" record_src="$2" smoke_src="${3:-}"
    local dest="${EVIDENCE_DIR}/${target}"
    mkdir -p "$dest"
    if [[ -f "$record_src" ]]; then
        cp "$record_src" "$dest/build-record.json"
        jq -r '"registry_image: " + .registry_image, "oci_digest: " + .oci_digest' "$record_src" > "$dest/digest.txt" 2>/dev/null || true
        MATRIX_DIGEST[$target]="$(jq -r '.oci_digest // "?"' "$record_src" 2>/dev/null || echo '?')"
    fi
    if [[ -n "$smoke_src" && -f "$smoke_src" ]]; then
        cp "$smoke_src" "$dest/local-smoke.json"
    fi
}

matrix_summary() {
    echo ""
    echo "================ push.sh all — summary ================"
    local target
    for target in ${MATRIX_TARGETS[@]+"${MATRIX_TARGETS[@]}"}; do
        printf '  %-24s status=%-28s digest=%s smoke=%s\n' \
            "$target" "${MATRIX_STATUS[$target]:-skipped}" \
            "${MATRIX_DIGEST[$target]:-—}" "${MATRIX_SMOKE[$target]:-—}"
    done
}

matrix_run() {
    if [[ ${#MATRIX_TARGETS[@]} -eq 0 ]]; then
        echo "ERROR: no images discovered under containers/ (nothing to build)" >&2
        exit 1
    fi
    local target img
    for target in ${MATRIX_TARGETS[@]+"${MATRIX_TARGETS[@]}"}; do
        echo ""
        echo "===== build + push: ${target} ====="
        case "$target" in
            preprocessing)
                build_and_push_preprocessing "$@" || matrix_fail "preprocessing (push)"
                MATRIX_STATUS[$target]="pushed"
                matrix_smoke "$target" "${REPO_ROOT}/containers/preprocessing/smoke-local.sh" "bspp-orchestration:preprocessing"
                matrix_evidence "$target" "${REPO_ROOT}/containers/preprocessing/dist/build-record.json" "${REPO_ROOT}/containers/preprocessing/dist/local-smoke.json"
                ;;
            postprocessing)
                # build_and_push_postprocessing runs the composition smoke before
                # the push, so the smoke gates publication.
                build_and_push_postprocessing "$@" || matrix_fail "postprocessing (push)"
                MATRIX_STATUS[$target]="pushed"
                MATRIX_SMOKE[$target]="pass"
                matrix_evidence "$target" "${REPO_ROOT}/containers/postprocessing/dist/build-record.json" "${REPO_ROOT}/containers/postprocessing/dist/local-smoke.json"
                ;;
            folding:*)
                img="${target#folding:}"
                build_and_push_folding_image "$img" "$@" || matrix_fail "${target} (push)"
                MATRIX_STATUS[$target]="pushed"
                matrix_smoke "$target" "${REPO_ROOT}/containers/folding/${img}/smoke-local.sh" "bspp-orchestration:folding-${img}"
                matrix_evidence "$target" "${REPO_ROOT}/containers/folding/${img}/dist/build-record.json" "${REPO_ROOT}/containers/folding/${img}/dist/local-smoke.json"
                ;;
            postprocessing-internal)
                "${REPO_ROOT}/containers/nvidia/build-internal.sh" || matrix_fail "postprocessing-internal (build)"
                MATRIX_STATUS[$target]="built (not pushed by design)"
                MATRIX_SMOKE[$target]="n/a"
                ;;
        esac
    done
    matrix_summary
    if [[ ${INCLUDE_POSTPROCESSING_INTERNAL} -eq 1 ]]; then
        echo "Dedicated images pushed; the postprocessing-internal image was built and tagged locally (never pushed by design)."
    else
        echo "All built images pushed; per-image status above is the record."
    fi
}

# --- main ---
VARIANT="${1:?Usage: $0 <postprocessing|preprocessing|folding <runtime|colabfold|openfold-cli|bioir|all>|all> [docker-build-args...]}"
shift

FOLDING_IMAGE=""
if [[ "$VARIANT" == "folding" ]]; then
    FOLDING_IMAGE="${1:?Usage: $0 folding <runtime|colabfold|openfold-cli|bioir|all> [docker-build-args...]}"
    shift
fi

# Matrix-mode flags: `all` parses the full set (--dry-run,
# --include-postprocessing-internal, --evidence-dir). Single-variant modes parse
# only --include-postprocessing-internal (honored for postprocessing, fail-fast
# for other variants). Anything unrecognized keeps the existing contract and
# passes through to docker build.
DRY_RUN=0
INCLUDE_POSTPROCESSING_INTERNAL=0
EVIDENCE_DIR=""
if [[ "$VARIANT" == "all" ]]; then
    REMAINING_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) DRY_RUN=1; shift ;;
            --include-postprocessing-internal) INCLUDE_POSTPROCESSING_INTERNAL=1; shift ;;
            --evidence-dir)
                [[ $# -ge 2 ]] || { echo "ERROR: --evidence-dir needs a value" >&2; exit 2; }
                EVIDENCE_DIR="$2"; shift 2 ;;
            *) REMAINING_ARGS+=("$1"); shift ;;
        esac
    done
    set -- ${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}
    if [[ -z "${EVIDENCE_DIR}" ]]; then
        # Default to the internal evidence tree when it exists; otherwise use a
        # neutral local directory so a checkout without that tree still works.
        if [[ -d "${REPO_ROOT}/devdocs/evidence/sample-run-plans" ]]; then
            EVIDENCE_DIR="${REPO_ROOT}/devdocs/evidence/sample-run-plans/container-rebuild-$(date -u +%Y%m%d)/evidence"
        else
            EVIDENCE_DIR="${REPO_ROOT}/container-rebuild-evidence-$(date -u +%Y%m%d)/evidence"
        fi
    fi
    matrix_discover
    if [[ ${DRY_RUN} -eq 1 ]]; then
        matrix_plan
        exit 0
    fi
else
    # Single-variant mode: parse --include-postprocessing-internal (honored
    # only for postprocessing; fail-fast for any other variant). All other
    # args pass through to docker build per the existing contract.
    REMAINING_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --include-postprocessing-internal)
                if [[ "$VARIANT" == "postprocessing" ]]; then
                    INCLUDE_POSTPROCESSING_INTERNAL=1
                else
                    echo "ERROR: --include-postprocessing-internal is only valid with 'postprocessing' or 'all', not '${VARIANT}'" >&2
                    exit 2
                fi
                shift
                ;;
            *) REMAINING_ARGS+=("$1"); shift ;;
        esac
    done
    set -- ${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}
fi

# Registry login — match the registry as a quoted JSON key (exact host, not a
# substring of a longer host), with or without an https:// scheme prefix.
# Deliberately jq-free: the login check predates the matrix preflight and
# avoids a jq dependency in this path.
registry_login() {
    local docker_config="${HOME}/.docker/config.json"
    local registry_bare="${BSPP_REGISTRY#https://}"
    registry_bare="${registry_bare#http://}"
    local registry_re="${registry_bare//./\\.}"
    if [[ -f "$docker_config" ]] && \
        grep -qE "\"${registry_re}\"|\"https://${registry_re}\"" "$docker_config" 2>/dev/null; then
        echo "Already authenticated to ${BSPP_REGISTRY}"
    else
        echo "Logging in to ${BSPP_REGISTRY} ..."
        docker login "${BSPP_REGISTRY}"
    fi
}

if [[ "$VARIANT" == "all" ]]; then
    # Source-state preflight BEFORE registry authentication: a dirty, stale,
    # or diverged checkout must fail before any credential prompt or auth
    # state change (review finding on the initial matrix flow).
    matrix_preflight
    registry_login
    matrix_run "$@"
elif [[ "$VARIANT" == "preprocessing" ]]; then
    registry_login
    build_and_push_preprocessing "$@"
elif [[ "$VARIANT" == "postprocessing" ]]; then
    registry_login
    build_and_push_postprocessing "$@"
    if [[ ${INCLUDE_POSTPROCESSING_INTERNAL} -eq 1 ]]; then
        if [[ -f "${REPO_ROOT}/containers/nvidia/build-internal.sh" ]]; then
            echo "Building postprocessing-internal image (build + tag only; never pushed by design)..."
            "${REPO_ROOT}/containers/nvidia/build-internal.sh"
            echo "postprocessing-internal: built (not pushed by design)"
        else
            echo "ERROR: --include-postprocessing-internal needs the internal build assets (containers/nvidia/), absent from this tree" >&2
            exit 1
        fi
    fi
elif [[ "$VARIANT" == "folding" ]]; then
    registry_login
    FOLDING_IMAGES=(runtime colabfold openfold-cli bioir)
    if [[ "$FOLDING_IMAGE" == "all" ]]; then
        for img in "${FOLDING_IMAGES[@]}"; do
            build_and_push_folding_image "$img" "$@"
        done
    else
        build_and_push_folding_image "$FOLDING_IMAGE" "$@"
    fi
else
    echo "ERROR: Unknown variant '${VARIANT}'. Available: postprocessing preprocessing 'folding <runtime|colabfold|openfold-cli|bioir|all>' all" >&2
    exit 1
fi

echo "Done."
