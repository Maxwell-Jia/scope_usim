#!/bin/bash

# tau2-bench retail, single-simulator collapse study: train
# Qwen3-4B-Instruct-2507 against a frozen deepseek-v4-flash-meituan user.
# Agent (trainable): Qwen3-4B-Instruct-2507 via SGLang
# User sim (fixed): deepseek-v4-flash-meituan via the Friday gateway.
#
# Node use: before launching, this script kills every SGLang and Ray
# process on the machine (pkill -9 sglang; ray stop --force; pkill -9 ray)
# and starts a fresh Ray head that claims all NUM_GPUS GPUs. Run it on a
# dedicated node, or set SKIP_PROCESS_CLEANUP=1 to skip the kill step.
#
# All tunables live in the parameter block right below this header.
# MODEL_DIR (required) must contain Qwen3-4B-Instruct-2507/ and
# Qwen3-4B-Instruct-2507_torch_dist/. See cmd/slime/README.md.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime"

# ===================== Parameters: edit here =====================

# Required: directory that holds the actor checkpoints.
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the directory that holds the model checkpoints; see cmd/slime/README.md}"

# Output directory. A YYYYMMDD_HHMM (Asia/Shanghai) suffix is appended on
# every launcher invocation, including platform retries; retries within
# the same minute share a directory.
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/jiaminghui06/outputs/scope/tau2-retail-single-deepseek}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)"

# Optional resume: a previous run's --save directory (the one containing
# latest_checkpointed_iteration.txt), e.g. the save dir of the crashed run
# psxa8p7xru4vuomj, which saved iteration 31 before dying at step 50.
# Requires megatron_to_hf_mode=raw (default): slime detects the Megatron
# checkpoint and restores optimizer, RNG and start_rollout_id from it.
RESUME_FROM="${RESUME_FROM:-}"

# Node topology.
NUM_GPUS=8

# Fixed opponent (user simulator) via the Friday gateway.
FIXED_OPPONENT_MODEL="deepseek-v4-flash-meituan"
FIXED_OPPONENT_BASE_URL="https://aigc.sankuai.com/v1/openai/native"
FIXED_OPPONENT_API_KEY_VAR="FRIDAY_API_KEY"
USIM_DOMAIN="retail"
MAX_TURNS=30

# Rollout / GRPO batch sizes.
NUM_ROLLOUT=240
ROLLOUT_BATCH_SIZE=16
N_SAMPLES_PER_PROMPT=8
ROLLOUT_MAX_RESPONSE_LEN=16384
ROLLOUT_TEMPERATURE=0.7
GLOBAL_BATCH_SIZE=128

# Optimization.
LR=1e-6
KL_LOSS_COEF=0.01
EPS_CLIP=0.2
EPS_CLIP_HIGH=0.28

# Memory/perf. MAX_TOKENS_PER_GPU was raised 8192 -> 16384 on 2026-11-03
# after long sequences crashed training; keep it at least as large as the
# longest packed sequence per GPU.
MAX_TOKENS_PER_GPU=8192
CONTEXT_PARALLEL_SIZE=2
SGLANG_MEM_FRACTION_STATIC=0.75

# Checkpoint / eval cadence.
SAVE_INTERVAL=16
EVAL_INTERVAL=16
EVAL_CONFIG_SOURCE="${PROJECT_ROOT}/eval_configs/tau2_retail_friday_3model.yaml"

# Trajectory archiving (per-rollout JSONL under trajectories/; eval
# conversations under trajectories/eval/). Full conversation text with
# reward/status metadata — token-level logprobs are NOT saved.
TRAJECTORY_OUTPUT_DIR="${OUTPUT_DIR}/trajectories"

# ===================== End parameters =====================

mkdir -p "${OUTPUT_DIR}"

# Mirror all script output — including the streamed `ray job submit` logs —
# into a file under OUTPUT_DIR, so the console log is archived with the run
# itself instead of living only on the MLP platform. stdout/stderr still go
# to the platform unchanged (tee passes them through). Requires bash
# (process substitution); platform retries within the same minute append to
# the same-minute log rather than overwrite (-a, timestamp is to the second).
RUN_LOG="${OUTPUT_DIR}/run_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "=== ${BASH_SOURCE[0]} started $(TZ=Asia/Shanghai date '+%F %T') ; log: ${RUN_LOG} ==="
trap 'echo "=== script exiting with status $? at $(TZ=Asia/Shanghai date "+%F %T") ==="' EXIT

# Stop SGLang and Ray processes left on this node (see the header).
if [ "${SKIP_PROCESS_CLEANUP:-0}" != "1" ]; then
    pkill -9 sglang 2>/dev/null || true
    sleep 3
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    sleep 3
    pkill -f gpu_continuous_compute.py 2>/dev/null || true
fi

set -e

export PYTHONUNBUFFERED=1
export WEAVE_PRINT_CALL_LINK=false

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

cd "${PROJECT_ROOT}"
source /opt/venv/bin/activate
pip install --upgrade pip
pip install --no-deps "pyarrow==25.0.1"
pip install -e ".[slime,p4g,tau2,dev]" --no-deps
pip install -e ./slime --no-deps
pip install -e ./external/tau2-bench --no-deps

# GPU keeper: hold nvidia-smi utilization above the platform's idle-kill
# threshold during rollout waits (moved here from the deleted cmd/env_prepare.sh).
# Disable with GPU_KEEPER_ENABLED=false.
if [ "${GPU_KEEPER_ENABLED:-true}" == "true" ]; then
    KEEPER_LOG="${OUTPUT_DIR}/gpu_keeper_$(hostname).log"
    python3 "${PROJECT_ROOT}/scripts/gpu_continuous_compute.py" \
        --gpu_number "${NUM_GPUS}" --gpu_utilization_threshold "${GPU_KEEPER_THRESHOLD:-85}" --duration 0 \
        >"${KEEPER_LOG}" 2>&1 &
    echo "GPU keeper started: pid=$! threshold=${GPU_KEEPER_THRESHOLD:-85}% log=${KEEPER_LOG}"
fi

# Source model configuration (Instruct, rotary_base=5000000)
source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-4B-Instruct-2507"
   --ref-load "${MODEL_DIR}/Qwen3-4B-Instruct-2507_torch_dist"
   --save "${OUTPUT_DIR}/Qwen3-4B-Instruct-2507_usim_tau2/"
   --save-interval "${SAVE_INTERVAL}"
)

if [ -n "${RESUME_FROM}" ]; then
    if [ -f "${RESUME_FROM}/latest_checkpointed_iteration.txt" ]; then
        CKPT_ARGS+=( --load "${RESUME_FROM}" )
        echo "RESUME: loading Megatron checkpoint from ${RESUME_FROM} (iteration $(cat "${RESUME_FROM}/latest_checkpointed_iteration.txt"))"
    else
        echo "ERROR: RESUME_FROM=${RESUME_FROM} has no latest_checkpointed_iteration.txt" >&2
        exit 1
    fi
fi

ROLLOUT_ARGS=(
   --data-source-path usim.slime.data_source.get_tau2_data_source
   --rollout-function-path usim.slime.rollout.usim_generate_rollout
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

# Agent (trainable) = Qwen3-4B-Instruct via SGLang
# User sim (fixed opponent) = the FIXED_OPPONENT_* model above
USIM_ARGS=(
   --trainable-role agent
   --max-turns "${MAX_TURNS}"
   --usim-domain "${USIM_DOMAIN}"
   --usim-fixed-opponent-model "${FIXED_OPPONENT_MODEL}"
   --usim-fixed-opponent-base-url "${FIXED_OPPONENT_BASE_URL}"
   --usim-fixed-opponent-api-key-var "${FIXED_OPPONENT_API_KEY_VAR}"
)

TRAJECTORY_ARGS=(
   --trajectory-output-dir "${TRAJECTORY_OUTPUT_DIR}"
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-grpo-std-normalization
   --disable-rewards-normalization
   --use-kl-loss
   --kl-loss-coef "${KL_LOSS_COEF}"
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip "${EPS_CLIP}"
   --eps-clip-high "${EPS_CLIP_HIGH}"
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT:-scope}"
   --wandb-group "qwen3-4B-Instruct-2507-tau2-deepseek-$(TZ=Asia/Shanghai date +%y%m%d)"
   --disable-wandb-random-suffix
   --wandb-dir "${OUTPUT_DIR}/wandb"
)

TB_ARGS=(
   --use-tensorboard
)
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"

# Eval config (no ${...} variables inside, so a plain copy is enough —
# the training image has no envsubst/gettext).
EVAL_CONFIG_FILE="${OUTPUT_DIR}/eval_config.yaml"
cp "${EVAL_CONFIG_SOURCE}" "${EVAL_CONFIG_FILE}"

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL}"
   --skip-eval-before-train
   --eval-config "${EVAL_CONFIG_FILE}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR:-/opt/Megatron-LM}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m train_usim_slime \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${USIM_ARGS[@]} \
   ${TRAJECTORY_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${TB_ARGS[@]}
