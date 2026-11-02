#!/bin/bash
set -euo pipefail

# Both owner changes must be accepted before this manual job becomes supported.
: "${LMCACHE_MP_WHEEL:?Provide the reviewed scheduler-event LMCache wheel artifact}"
: "${VLLM_KV_EVENTS_WHEEL:?Provide the reviewed vLLM 0.31 idle-event wheel artifact}"

uv pip install --python "$(command -v python3)" fastokens==0.2.0
uv pip install --python "$(command -v python3)" --no-deps ai-dynamo-runtime==1.6.0.dev20260918
# Retain Ray's NCCL override while resolving LMCache's declared dependencies.
uv pip install --python "$(command -v python3)" \
    --reinstall-package vllm \
    --override <(printf 'nvidia-nccl-cu13==2.30.7\n') \
    "$LMCACHE_MP_WHEEL" "$VLLM_KV_EVENTS_WHEEL" \
    'torch==2.13.0+cu130' 'vllm==0.31.0' \
    'grpcio==1.78.0' 'prometheus-client==0.24.1' 'cupy-cuda13x==14.2.0'
# The two CuPy distributions share files; restore the CUDA 13 package afterward.
uv pip uninstall --python "$(command -v python3)" cupy-cuda12x
uv pip install --python "$(command -v python3)" --no-deps --reinstall cupy-cuda13x==14.2.0
python3 -c 'import torch; from lmcache.v1.multiprocess.custom_types import KVEventProfile; import lmcache.cuda_ops'
python3 -c 'from lmcache.integration.vllm.vllm_multi_process_adapter import LMCacheMPSchedulerAdapter; from vllm.v1.core.sched.scheduler import Scheduler; assert hasattr(LMCacheMPSchedulerAdapter, "get_kv_events"); assert hasattr(Scheduler, "publish_kv_events")'
