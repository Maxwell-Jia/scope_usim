#!/bin/bash
# env_prepare.sh — pin the training stack in the job container (run once per job).
#
# Steps:
#   1. Clone Megatron-LM and SGLang at the commits pinned by slime into
#      PROJECT_ROOT and apply their patches (existing clones are reused).
#   2. Editable-install the pinned slime and usim packages.
#   3. Start the GPU keeper (keeps nvidia-smi utilization up during idle stages).
#   4. Convert HF weights to torch_dist with the same Megatron stack.
#
# SGLang is re-pinned because the platform image ships an older sglang than
# the camera-ready slime expects (missing --data-parallel-size in
# ServerArgs.add_cli_args). The compiled deps (sgl-kernel/TE/FA) come from
# the image; this source switch was already exercised by a full training run
# (2026-10-21, scope_install.sh used the same sglang commit).
#
# Env vars:
#   MEGATRON_REPO        Megatron-LM remote, default GitHub HTTPS
#   SGLANG_REPO          SGLang remote, default GitHub HTTPS
#   HF_MODEL_DIR         HF weights dir (default: DolphinFS Qwen3-4B-Instruct-2507)
#   DIST_OUT_DIR         torch_dist output dir (default: next to HF_MODEL_DIR)
#   PROJECT_ROOT         scope_usim dir (default: parent of this script)
#   GPU_KEEPER_ENABLED   true/false, default true
#   GPU_KEEPER_THRESHOLD utilization threshold (%), default 75
#
# Credentials are injected at runtime, never stored here.
set -euo pipefail

MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
SGLANG_REPO="${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"
# sglang v0.5.9, the stable stack pinned by slime's docker/README.md
SGLANG_COMMIT=bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2
SLIME_MEGATRON_PATCH=docker/patch/latest/megatron.patch
SLIME_SGLANG_PATCH=docker/patch/latest/sglang.patch
GPU_KEEPER_ENABLED="${GPU_KEEPER_ENABLED:-true}"
GPU_KEEPER_THRESHOLD="${GPU_KEEPER_THRESHOLD:-75}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
HF_MODEL_DIR="${HF_MODEL_DIR:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/jiaminghui06/hf_models/Qwen3-4B-Instruct-2507}"
DIST_OUT_DIR="${DIST_OUT_DIR:-${HF_MODEL_DIR%/*}/$(basename "${HF_MODEL_DIR}")_torch_dist}"

echo "[env_prepare] PROJECT_ROOT=${PROJECT_ROOT}"

source /opt/venv/bin/activate
pip install --upgrade pip

# ---------- 1. Megatron: pinned commit + slime patch ----------
if [ ! -d "${MEGATRON_DIR}" ]; then
    git clone --filter=blob:none --no-checkout "${MEGATRON_REPO}" "${MEGATRON_DIR}"
    git -C "${MEGATRON_DIR}" fetch --depth 1 origin "${MEGATRON_COMMIT}"
fi
git -C "${MEGATRON_DIR}" checkout --force --detach "${MEGATRON_COMMIT}"
git -C "${MEGATRON_DIR}" apply "${PROJECT_ROOT}/slime/${SLIME_MEGATRON_PATCH}" --3way

# ---------- 2. SGLang: pinned commit + slime patch ----------
SGLANG_DIR="${SGLANG_DIR:-${PROJECT_ROOT}/SGLang}"
if [ ! -d "${SGLANG_DIR}" ]; then
    git clone --filter=blob:none --no-checkout "${SGLANG_REPO}" "${SGLANG_DIR}"
    git -C "${SGLANG_DIR}" fetch --depth 1 origin "${SGLANG_COMMIT}"
fi
git -C "${SGLANG_DIR}" checkout --force --detach "${SGLANG_COMMIT}"
git -C "${SGLANG_DIR}" apply "${PROJECT_ROOT}/slime/${SLIME_SGLANG_PATCH}" --3way
# Replace the image's older sglang with the pinned one; compiled deps
# (sgl-kernel etc.) stay from the image.
pip install -e "${SGLANG_DIR}/python" --no-deps --no-build-isolation

# ---------- 3. Editable installs ----------
# mbridge is pinned by the slime Dockerfile and needed by convert_hf_to_torch_dist.
pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install -e "${PROJECT_ROOT}/slime" --no-deps --no-build-isolation
pip install -e "${PROJECT_ROOT}[slime,tau2]" --no-deps
pip install -e "${PROJECT_ROOT}/external/tau2-bench" --no-deps

# ---------- 4. GPU keeper ----------
if [ "${GPU_KEEPER_ENABLED}" == "true" ]; then
    pkill -f gpu_continuous_compute.py 2>/dev/null || true
    keeper_log="${PROJECT_ROOT}/gpu_keeper_$(hostname).log"
    python3 "${PROJECT_ROOT}/scripts/gpu_continuous_compute.py" \
        --gpu_number 8 --gpu_utilization_threshold "${GPU_KEEPER_THRESHOLD}" --duration 0 \
        >"${keeper_log}" 2>&1 &
    echo "[env_prepare] GPU keeper started: pid=$! threshold=${GPU_KEEPER_THRESHOLD}% log=${keeper_log}"
else
    echo "[env_prepare] GPU keeper disabled (GPU_KEEPER_ENABLED=${GPU_KEEPER_ENABLED})"
fi

# ---------- 5. HF -> torch_dist (converted with the training stack) ----------
# Skip conversion if the torch_dist output already exists.
if [ -d "${DIST_OUT_DIR}" ]; then
    echo "[env_prepare] torch_dist already exists, skip conversion: ${DIST_OUT_DIR}"
else
    mkdir -p "${DIST_OUT_DIR}"
    cd "${PROJECT_ROOT}/slime"
    # Instruct uses rotary-base 5000000; do not use the 1e6 default from qwen3-4B.sh.
    source scripts/models/qwen3-4B-Instruct-2507.sh
    PYTHONPATH="${MEGATRON_DIR}" python tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "${HF_MODEL_DIR}" \
        --save "${DIST_OUT_DIR}"
fi

echo "[env_prepare] done. For training:"
echo "[env_prepare]   PYTHONPATH=${MEGATRON_DIR}  (must be passed to the Ray runtime env)"
echo "[env_prepare]   torch_dist: ${DIST_OUT_DIR}"
