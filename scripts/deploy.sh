#!/usr/bin/env bash

set -euo pipefail

TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-64}" \
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}" \
swift deploy \
    --model "${VIDEOPRO_DEPLOY_MODEL:-/inspire/hdd/global_user/lichenglin-253208540324/VideoPro_model/hf_full_model}" \
    --infer_backend vllm \
    --torch_dtype bfloat16 \
    --port "${VIDEOPRO_PORT:-8007}" \
    --vllm_tensor_parallel_size "${VIDEOPRO_TP_SIZE:-4}" \
    --served_model_name "${VIDEOPRO_MODEL:-qwen3vl}"




TORCH_SYMM_MEM_DISABLE_MULTICAST=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
FPS_MAX_FRAMES=64 \
VIDEO_MAX_PIXELS=50176 \
swift deploy \
--adapters lora1=/checkpoint-1597 \
--infer_backend vllm \
--torch_dtype bfloat16 \
--port 8007 \
--vllm_tensor_parallel_size 8 \
--vllm_max_lora_rank 64 \
--served_model_name "qwen3vl"