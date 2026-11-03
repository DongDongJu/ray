#!/bin/bash
set -euo pipefail
unset SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE

# The BYOD build context contains only this script, executed from /tmp.
# Keep the reviewed wheel and Python-source inputs explicit until acceptance.
: "${RAY_LATEST_WHEEL_URL:?Provide the matching native Ray wheel URL}"
: "${VLLM_LATEST_WHEEL_URL:?Provide the reviewed vLLM idle-event wheel URL}"
: "${LMCACHE_LATEST_WHEEL_URL:?Provide the reviewed native-event LMCache wheel URL}"
: "${DYNAMO_LATEST_WHEEL_URL:?Provide the reviewed Dynamo wheel URL}"
: "${RAY_EXPECTED_COMMIT:?Provide the native Ray wheel commit}"
: "${RAY_PYTHON_SOURCE_COMMIT:?Provide the immutable Ray Python-source commit}"

python_bin="${PYTHON_BIN:-python3}"
"${python_bin}" - <<'PY'
import os
import pathlib
import platform
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit("The reviewed integration requires Python 3.12.")
libc, version = platform.libc_ver()
if libc != "glibc" or tuple(map(int, version.split("."))) < (2, 39):
    raise SystemExit(
        "The reviewed Dynamo wheel requires glibc 2.39+. Use a compatible "
        "Ubuntu 24.04 custom image; the stock Ubuntu 22.04 BYOD base is unsupported."
    )
if sys.prefix == sys.base_prefix and not pathlib.Path(sys.prefix, "conda-meta").is_dir():
    raise SystemExit("Use a virtual environment or the BYOD image's conda environment.")
if os.environ.get("PYTHONPATH"):
    raise SystemExit("Use a clean environment without PYTHONPATH overlays.")
PY

[[ "${RAY_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${RAY_PYTHON_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
if [[ -n "${RAY_PYTHON_SOURCE_DIR:-}" ]]; then
    source_dir="${RAY_PYTHON_SOURCE_DIR}"
else
    : "${RAY_PYTHON_SOURCE_REPO:?Provide the repository containing the reviewed source commit}"
    source_dir="$("${python_bin}" -c 'import sys; print(sys.prefix)')/ray-integration-source"
    [[ ! -e "${source_dir}" ]]
    git init "${source_dir}"
    git -C "${source_dir}" fetch --depth=1 "${RAY_PYTHON_SOURCE_REPO}" "${RAY_PYTHON_SOURCE_COMMIT}"
    git -C "${source_dir}" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${RAY_PYTHON_SOURCE_COMMIT}" ]]
git -C "${source_dir}" diff --quiet
git -C "${source_dir}" diff --cached --quiet

# Reinstalling Ray must not traverse links into any existing source checkout.
"${python_bin}" - <<'PY'
import importlib.util
import os
from pathlib import Path
import sysconfig

ray_home = Path(sysconfig.get_paths()["purelib"]).resolve() / "ray"
spec = importlib.util.find_spec("ray")
if spec is not None:
    origins = [spec.origin] if spec.origin else list(spec.submodule_search_locations or [])
    if any(not Path(origin).resolve().is_relative_to(ray_home) for origin in origins):
        raise SystemExit("Existing editable Ray import: use a fresh, unlinked environment.")

def check_link(path):
    if path.is_symlink():
        target = path.resolve()
        if not target.is_relative_to(ray_home):
            raise SystemExit(
                f"Existing Ray source link: {path} -> {target}. "
                "Use a fresh, unlinked environment to preserve the checkout."
            )

check_link(ray_home)
for directory, dirs, files in os.walk(ray_home, followlinks=False):
    for name in dirs + files:
        check_link(Path(directory, name))
PY

# Resolve declared dependencies while replacing Ray's baseline vLLM 0.32 pin.
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

# The reviewed vLLM already preserves EOS IDs and defers Triton validation.
site_packages="$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
device_patch="${source_dir}/python/requirements/llm/patches/vllm-device-aware-compile-cache.patch"
(
    cd "${site_packages}"
    # A venv inside a checkout must not inherit that repository's patch prefix.
    GIT_CEILING_DIRECTORIES="$(dirname "$(pwd -P)")"
    export GIT_CEILING_DIRECTORIES
    if ! git apply --reverse --check "${device_patch}"; then
        git apply --check "${device_patch}"
        git apply "${device_patch}"
    fi
)

"${python_bin}" -c 'import os, ray; assert ray.__commit__ == os.environ["RAY_EXPECTED_COMMIT"]'
"${python_bin}" "${source_dir}/python/ray/setup-dev.py" -y --skip dashboard
uv pip check --python "${python_bin}"
"${python_bin}" - "${source_dir}" <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import ray
import ray.llm
import ray.serve
import lmcache.cuda_ops
from lmcache.v1.multiprocess import custom_types
from lmcache.integration.vllm.vllm_multi_process_adapter import LMCacheMPSchedulerAdapter
from vllm.v1.core.sched.scheduler import Scheduler

source = Path(sys.argv[1]).resolve()
assert hasattr(custom_types, "KVEventProfile")
assert hasattr(LMCacheMPSchedulerAdapter, "get_kv_events")
assert hasattr(Scheduler, "publish_kv_events")
for module in (ray.llm, ray.serve):
    assert Path(module.__file__).resolve().is_relative_to(source / "python"), module
print(json.dumps({
    "ray_native_commit": ray.__commit__,
    "ray_python_source_commit": os.environ["RAY_PYTHON_SOURCE_COMMIT"],
    "ray_python_source_dir": str(source),
    "versions": {name: importlib.metadata.version(name) for name in (
        "ray", "vllm", "torch", "lmcache", "ai-dynamo-runtime"
    )},
}))
PY
