#!/bin/bash
set -euo pipefail

# This manual job stays gated until the native-event dependency is accepted.
: "${LMCACHE_MP_WHEEL:?Provide the reviewed native-event LMCache wheel artifact}"

uv pip install --python "$(command -v python3)" fastokens==0.2.0
uv pip install --python "$(command -v python3)" --no-deps ai-dynamo-runtime==1.6.0.dev20260918
# Retain Ray's NCCL override while resolving LMCache's declared dependencies.
uv pip install --python "$(command -v python3)" \
    --override <(printf 'nvidia-nccl-cu13==2.30.7\n') \
    "$LMCACHE_MP_WHEEL" 'torch==2.13.0+cu130' 'vllm==0.31.0' \
    'grpcio==1.78.0' 'prometheus-client==0.24.1' 'cupy-cuda13x==14.2.0'
# The two CuPy distributions share files; restore the CUDA 13 package afterward.
uv pip uninstall --python "$(command -v python3)" cupy-cuda12x
uv pip install --python "$(command -v python3)" --no-deps --reinstall cupy-cuda13x==14.2.0
python3 -c 'import torch; from lmcache.v1.multiprocess.custom_types import KVEventProfile; import lmcache.cuda_ops'
