#!/bin/bash
# convert_weights.sh — one-shot HF -> Megatron torch_dist weight conversion.
#
# Everything this needs ships inside the training_scope image: the patched
# Megatron-LM at /opt/Megatron-LM (editable-installed), mbridge, and the
# conversion tool from the pinned slime submodule. Run inside a container of
# registry-offlinebiz.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/mtai/training_scope_c16160d6:1.0.0
# (e.g. from a Workbench session), pointing at weights on DolphinFS.
#
# Skips the conversion when DIST_OUT_DIR already exists.
#
# Env vars:
#   HF_MODEL_DIR   HF weights dir (default: DolphinFS Qwen3-4B-Instruct-2507)
#   DIST_OUT_DIR   torch_dist output dir (default: <HF_MODEL_DIR>_torch_dist)
#   MODEL_SCRIPT   slime model args script (default: qwen3-4B-Instruct-2507.sh;
#                  Instruct uses rotary_base=5000000; do not use the 1e6
#                  default from the base qwen3-4B.sh)
#   MEGATRON_DIR   patched Megatron-LM (default: /opt/Megatron-LM from the image)
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
MEGATRON_DIR="${MEGATRON_DIR:-/opt/Megatron-LM}"
HF_MODEL_DIR="${HF_MODEL_DIR:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/jiaminghui06/hf_models/Qwen3-4B-Instruct-2507}"
DIST_OUT_DIR="${DIST_OUT_DIR:-${HF_MODEL_DIR%/*}/$(basename "${HF_MODEL_DIR}")_torch_dist}"
MODEL_SCRIPT="${MODEL_SCRIPT:-qwen3-4B-Instruct-2507.sh}"

echo "[convert_weights] HF=${HF_MODEL_DIR} -> torch_dist=${DIST_OUT_DIR}"

source /opt/venv/bin/activate

if [ -d "${DIST_OUT_DIR}" ]; then
    echo "[convert_weights] torch_dist already exists, skip conversion: ${DIST_OUT_DIR}"
    exit 0
fi

mkdir -p "${DIST_OUT_DIR}"
cd "${PROJECT_ROOT}/slime"
source "scripts/models/${MODEL_SCRIPT}"
PYTHONPATH="${MEGATRON_DIR}" python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint "${HF_MODEL_DIR}" \
    --save "${DIST_OUT_DIR}"

echo "[convert_weights] done: ${DIST_OUT_DIR}"
