#!/bin/bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"
"${python_bin}" -c 'import sys; assert sys.prefix != sys.base_prefix, "Use a virtual environment"'
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export RAY_PYTHON_SOURCE_DIR="${repo_root}"
RAY_PYTHON_SOURCE_COMMIT="${RAY_PYTHON_SOURCE_COMMIT:-$(git -C "${repo_root}" rev-parse HEAD)}"
export RAY_PYTHON_SOURCE_COMMIT
bash "${repo_root}/release/ray_release/byod/byod_llm_lmcache_mp_test.sh"
