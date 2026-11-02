#!/bin/bash
set -euo pipefail
unset SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE

# This explicit all-heads qualification does not use Ray's pinned 0.32 LLM lock.
: "${RAY_LATEST_WHEEL_URL:?Provide the matching native Ray wheel URL}"
: "${VLLM_LATEST_WHEEL_URL:?Provide the current vLLM owner wheel URL}"
: "${LMCACHE_LATEST_WHEEL_URL:?Provide the current native LMCache wheel URL}"
: "${DYNAMO_LATEST_WHEEL_URL:?Provide the current Dynamo wheel URL}"
: "${RAY_EXPECTED_COMMIT:?Provide the native Ray source commit}"

python_bin="${PYTHON_BIN:-python3}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
"${python_bin}" -c 'import sys; assert sys.prefix != sys.base_prefix, "Use a virtual environment"'
overrides="$(mktemp)"
trap 'rm -f "${overrides}"' EXIT
printf 'vllm[audio] @ %s\ntorch==2.14.0+cu130\n' "${VLLM_LATEST_WHEEL_URL}" > "${overrides}"

uv pip install --python "${python_bin}" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match --override "${overrides}" \
    "ray[default,serve,llm] @ ${RAY_LATEST_WHEEL_URL}" \
    "vllm[audio] @ ${VLLM_LATEST_WHEEL_URL}" \
    "lmcache @ ${LMCACHE_LATEST_WHEEL_URL}" \
    "ai-dynamo-runtime @ ${DYNAMO_LATEST_WHEEL_URL}" \
    fastokens==0.2.0 pytest==7.4.4 pytest-asyncio==0.17.2 \
    pytest-timeout==2.1.0 pytest-mock==3.14.0

# Main already preserves model EOS IDs and defers Triton checks to workers.
# Its compile-cache paths still omit the device index needed by Ray workers.
site_packages="$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
device_patch="${repo_root}/python/requirements/llm/patches/vllm-device-aware-compile-cache.patch"
(
    cd "${site_packages}"
    if ! git apply --reverse --check "${device_patch}"; then
        git apply --check "${device_patch}"
        git apply "${device_patch}"
    fi
)

"${python_bin}" -c 'import os, ray; assert ray.__commit__ == os.environ["RAY_EXPECTED_COMMIT"]'
"${python_bin}" "${repo_root}/python/ray/setup-dev.py" -y --skip dashboard
