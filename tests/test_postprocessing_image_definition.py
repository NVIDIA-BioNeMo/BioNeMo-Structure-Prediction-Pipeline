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

"""Static reproducibility assertions for the dedicated postprocessing image."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "containers" / "postprocessing"
DOCKERFILE = ROOT / "containers" / "Dockerfile"
VARIANT = ROOT / "containers" / "variants" / "postprocessing.env"


def test_postprocessing_dockerfile_bakes_contract_control_runtime_wheels() -> None:
    """The postprocessing image installs all three workspace wheels and asserts
    Control is present (the reverse of the old deps-only mount model)."""
    dockerfile = DOCKERFILE.read_text()
    assert 'ARG BASE_DIGEST=""' in dockerfile
    assert "FROM ${BASE_IMAGE}${BASE_DIGEST:+@${BASE_DIGEST}}" in dockerfile
    assert "COPY containers/postprocessing/context/wheels/ /opt/bspp/wheels/" in dockerfile
    assert "python -m pip install --no-deps /opt/bspp/wheels/*.whl" in dockerfile
    assert "bspp-orchestration-control" in dockerfile
    control_assert = "python -c 'import importlib.metadata; importlib.metadata.version(\"bspp-orchestration-control\")'"
    assert control_assert in dockerfile
    # The orchestration-source install-mode helper reads the baked commit here.
    assert "ENV BSPP_IMAGE_JSON=/opt/bspp/postprocessing-runtime-image.json" in dockerfile
    helper_copy = (
        "COPY containers/scripts/install-orchestration-source.sh /usr/local/bin/install-orchestration-source.sh"
    )
    assert helper_copy in dockerfile


def test_postprocessing_build_sources_variant_env_and_bakes_control_wheel() -> None:
    """The dedicated backend sources the variant env and builds all three wheels."""
    build = (IMAGE / "build.sh").read_text()
    assert 'source "$VARIANT_ENV"' in build
    assert "containers/variants/postprocessing.env" in build
    assert "--package bspp-orchestration-contract" in build
    assert "--package bspp-orchestration-control" in build
    assert "--package bspp-orchestration-runtime" in build
    assert 'CONTROL_WHEEL="$(find "$CONTEXT/wheels" -name \'bspp_orchestration_control-*.whl\' -print -quit)"' in build
    assert "control_wheel_sha256" in build
    assert '> "$CONTEXT/postprocessing-runtime-image.json"' in build
    assert '> "$DIST/build-record.json"' in build


def test_common_tooling_routes_postprocessing_to_dedicated_backend() -> None:
    """build.sh and push.sh delegate postprocessing to the dedicated backend."""
    common_build = (ROOT / "containers" / "scripts" / "build.sh").read_text()
    common_push = (ROOT / "containers" / "scripts" / "push.sh").read_text()
    assert '[[ "$VARIANT" == "postprocessing" ]]' in common_build
    assert '"${REPO_ROOT}/containers/postprocessing/build.sh"' in common_build
    assert 'IMAGE_NAME="bspp-orchestration:postprocessing"' in common_build
    assert 'POSTPROCESSING_LOCAL_IMAGE="bspp-orchestration:postprocessing"' in common_push
    assert 'docker tag "$record_image_id" "$POSTPROCESSING_REGISTRY_IMAGE"' in common_push


def test_postprocessing_manifest_includes_control_dependency() -> None:
    """tabulate (the Control Plane's only new transitive dep) is in the manifest."""
    manifest = (ROOT / "containers" / "pyprojects" / "latest.toml").read_text()
    assert 'tabulate = ">=0.9"' in manifest


def test_postprocessing_composition_smoke_is_wired() -> None:
    """The postprocessing image carries a composition smoke that proves the baked
    orchestration source, and the matrix runs it before pushing."""
    dockerfile = DOCKERFILE.read_text()
    assert "COPY containers/postprocessing/image-smoke.py /opt/bspp/bin/bspp-postprocessing-image-smoke" in dockerfile
    assert "/opt/bspp/bin/bspp-postprocessing-image-smoke" in dockerfile

    smoke_local = (IMAGE / "smoke-local.sh").read_text()
    assert "--network none" in smoke_local
    assert "/opt/bspp/bin/bspp-postprocessing-image-smoke" in smoke_local
    assert "dist/local-smoke.json" in smoke_local

    image_smoke = (IMAGE / "image-smoke.py").read_text()
    assert "BSPP_ORCHESTRATION_SOURCE" in image_smoke
    assert "bspp-orchestration-control" in image_smoke
    assert "postprocessing-runtime-image.json" in image_smoke

    push = (ROOT / "containers" / "scripts" / "push.sh").read_text()
    # The smoke gates publication: build_and_push_postprocessing runs it before
    # docker push, and the matrix records the result as pass.
    assert 'containers/postprocessing/smoke-local.sh" "$POSTPROCESSING_LOCAL_IMAGE"' in push
    assert "refusing to push" in push
    assert 'MATRIX_SMOKE[$target]="pass"' in push
    assert "containers/postprocessing/dist/local-smoke.json" in push
