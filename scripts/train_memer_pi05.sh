#!/usr/bin/env bash

set -euo pipefail

# MemER pi0.5 training launcher.
#
# All commonly changed settings can be overridden as environment variables:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=64 NUM_TRAIN_STEPS=45000 \
#     EXP_NAME=memer_wa01_bs64 ./scripts/train_memer_pi05.sh
#
# The first positional argument optionally selects another OpenPI config. Any
# remaining arguments are forwarded to scripts/train.py and therefore override
# values assembled below:
#   ./scripts/train_memer_pi05.sh memer_wa01 --seed=123

OPENPI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '3,15p' "${BASH_SOURCE[0]}"
  exit 0
fi

CONFIG_NAME="${CONFIG_NAME:-memer_wa01}"
if [[ -n "${1:-}" && "${1:0:1}" != "-" ]]; then
  CONFIG_NAME="$1"
  shift
fi

# Dataset and pretrained checkpoint locations.
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data1/shared_workspace/chensihang/dataset/memer/pi05}"
DATASET_REPO_ID="${DATASET_REPO_ID:-wa01_200seeds_v061_subtasks_v2}"
DATASET_REPO_IDS="${DATASET_REPO_IDS:-${OPENPI_MULTI_DATASETS:-${MEMER_MULTI_DATASETS:-}}}"
OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR:-/data1/shared_workspace/tangzhipeng/ckpts/openpi/openpi_assets}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_DIR}/.cache/openpi}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${OPENPI_DIR}/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${OPENPI_DIR}/checkpoints}"

# Runtime and data pipeline settings. BATCH_SIZE is the global batch size over
# all visible GPUs, not the per-GPU batch size.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-48}"
REFERENCE_NUM_TRAIN_STEPS="${REFERENCE_NUM_TRAIN_STEPS:-60000}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"
NUM_WORKERS="${NUM_WORKERS:-32}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
SEED="${SEED:-42}"

# Logging and checkpoint settings.
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_bs${BATCH_SIZE}}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
RESUME="${RESUME:-false}"
OVERWRITE="${OVERWRITE:-false}"

# MemBench currently trains with two camera views. PyAV is required on this
# server because TorchCodec cannot load its FFmpeg shared libraries.
OPENPI_MEMBENCH_CAMERA_KEYS="${OPENPI_MEMBENCH_CAMERA_KEYS:-observation.images.robot0_agentview_right,observation.images.robot0_eye_in_hand}"
OPENPI_MEMBENCH_VIDEO_BACKEND="${OPENPI_MEMBENCH_VIDEO_BACKEND:-pyav}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

# Norm stats have already been generated for WA01. Set this to true after
# changing the dataset or state/action transforms.
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS:-false}"
OPENPI_NORM_STATS_BATCH_SIZE="${OPENPI_NORM_STATS_BATCH_SIZE:-512}"
NORM_STATS_MAX_FRAMES="${NORM_STATS_MAX_FRAMES:-}"

# Set DRY_RUN=true to print the resolved commands without running them.
DRY_RUN="${DRY_RUN:-false}"
RUN_TRAINING="${RUN_TRAINING:-true}"

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
}

require_boolean() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "${name} must be true or false, got: ${value}" >&2
    exit 2
  fi
}

require_positive_integer BATCH_SIZE "${BATCH_SIZE}"
require_positive_integer REFERENCE_BATCH_SIZE "${REFERENCE_BATCH_SIZE}"
require_positive_integer REFERENCE_NUM_TRAIN_STEPS "${REFERENCE_NUM_TRAIN_STEPS}"
require_positive_integer NUM_WORKERS "${NUM_WORKERS}"
require_positive_integer FSDP_DEVICES "${FSDP_DEVICES}"
require_positive_integer LOG_INTERVAL "${LOG_INTERVAL}"
require_positive_integer SAVE_INTERVAL "${SAVE_INTERVAL}"
require_positive_integer KEEP_PERIOD "${KEEP_PERIOD}"
require_boolean WANDB_ENABLED "${WANDB_ENABLED}"
require_boolean RESUME "${RESUME}"
require_boolean OVERWRITE "${OVERWRITE}"
require_boolean COMPUTE_NORM_STATS "${COMPUTE_NORM_STATS}"
require_boolean DRY_RUN "${DRY_RUN}"
require_boolean RUN_TRAINING "${RUN_TRAINING}"

if [[ "${RESUME}" == "true" && "${OVERWRITE}" == "true" ]]; then
  echo "RESUME and OVERWRITE cannot both be true." >&2
  exit 2
fi

# If NUM_TRAIN_STEPS is omitted, preserve the number of samples seen by the
# reference run. For example, 48 x 60000 / 64 = 45000 steps.
target_samples=$((REFERENCE_BATCH_SIZE * REFERENCE_NUM_TRAIN_STEPS))
if [[ -z "${NUM_TRAIN_STEPS}" ]]; then
  if (( target_samples % BATCH_SIZE != 0 )); then
    echo "Cannot exactly preserve ${target_samples} samples with BATCH_SIZE=${BATCH_SIZE}." >&2
    echo "Set NUM_TRAIN_STEPS explicitly." >&2
    exit 2
  fi
  NUM_TRAIN_STEPS=$((target_samples / BATCH_SIZE))
fi
require_positive_integer NUM_TRAIN_STEPS "${NUM_TRAIN_STEPS}"

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
gpu_count="${#visible_gpus[@]}"
if (( gpu_count == 0 )); then
  echo "CUDA_VISIBLE_DEVICES must contain at least one GPU id." >&2
  exit 2
fi
if (( BATCH_SIZE % gpu_count != 0 )); then
  echo "Global BATCH_SIZE=${BATCH_SIZE} must be divisible by ${gpu_count} visible GPUs." >&2
  exit 2
fi
if (( gpu_count % FSDP_DEVICES != 0 )); then
  echo "GPU count ${gpu_count} must be divisible by FSDP_DEVICES=${FSDP_DEVICES}." >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not available in PATH." >&2
  exit 1
fi
if [[ -n "${DATASET_REPO_IDS}" ]]; then
  IFS=',' read -r -a dataset_ids <<< "${DATASET_REPO_IDS}"
  for dataset_id in "${dataset_ids[@]}"; do
    dataset_id="${dataset_id//[[:space:]]/}"
    [[ -n "${dataset_id}" ]] || continue
    if [[ ! -f "${HF_LEROBOT_HOME}/${dataset_id}/meta/info.json" ]]; then
      echo "Dataset metadata not found: ${HF_LEROBOT_HOME}/${dataset_id}/meta/info.json" >&2
      exit 1
    fi
  done
else
  if [[ ! -f "${HF_LEROBOT_HOME}/${DATASET_REPO_ID}/meta/info.json" ]]; then
    echo "Dataset metadata not found: ${HF_LEROBOT_HOME}/${DATASET_REPO_ID}/meta/info.json" >&2
    exit 1
  fi
fi
if [[ ! -d "${OPENPI_PRETRAIN_MODEL_DIR}/pi05_base/params" ]]; then
  echo "pi0.5 base params not found: ${OPENPI_PRETRAIN_MODEL_DIR}/pi05_base/params" >&2
  exit 1
fi

mkdir -p "${OPENPI_DATA_HOME}" "${CHECKPOINT_BASE_DIR}"

export HF_LEROBOT_HOME
export DATASET_REPO_ID
export DATASET_REPO_IDS
if [[ -n "${OPENPI_MULTI_DATASETS:-}" ]]; then
  export OPENPI_MULTI_DATASETS
fi
if [[ -n "${MEMER_MULTI_DATASETS:-}" ]]; then
  export MEMER_MULTI_DATASETS
fi
export OPENPI_PRETRAIN_MODEL_DIR
export OPENPI_DATA_HOME
export OPENPI_MEMBENCH_CAMERA_KEYS
export OPENPI_MEMBENCH_VIDEO_BACKEND
export XLA_PYTHON_CLIENT_MEM_FRACTION
export CUDA_VISIBLE_DEVICES
unset LEROBOT_HOME

train_args=(
  "${CONFIG_NAME}"
  "--exp-name=${EXP_NAME}"
  "--batch-size=${BATCH_SIZE}"
  "--num-train-steps=${NUM_TRAIN_STEPS}"
  "--num-workers=${NUM_WORKERS}"
  "--fsdp-devices=${FSDP_DEVICES}"
  "--seed=${SEED}"
  "--log-interval=${LOG_INTERVAL}"
  "--save-interval=${SAVE_INTERVAL}"
  "--keep-period=${KEEP_PERIOD}"
  "--assets-base-dir=${ASSETS_BASE_DIR}"
  "--checkpoint-base-dir=${CHECKPOINT_BASE_DIR}"
)

if [[ "${WANDB_ENABLED}" == "true" ]]; then
  train_args+=("--wandb-enabled")
else
  train_args+=("--no-wandb-enabled")
fi
if [[ "${RESUME}" == "true" ]]; then
  train_args+=("--resume")
fi
if [[ "${OVERWRITE}" == "true" ]]; then
  train_args+=("--overwrite")
fi
train_args+=("$@")

echo "Resolved pi0.5 training settings:"
echo "  config / experiment : ${CONFIG_NAME} / ${EXP_NAME}"
if [[ -n "${DATASET_REPO_IDS}" ]]; then
  echo "  datasets             : ${HF_LEROBOT_HOME}/{${DATASET_REPO_IDS}}"
else
  echo "  dataset              : ${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"
fi
echo "  GPUs                 : ${CUDA_VISIBLE_DEVICES} (${gpu_count})"
echo "  global / per-GPU BS  : ${BATCH_SIZE} / $((BATCH_SIZE / gpu_count))"
echo "  train steps          : ${NUM_TRAIN_STEPS}"
echo "  samples seen         : $((BATCH_SIZE * NUM_TRAIN_STEPS))"
echo "  video / cameras      : ${OPENPI_MEMBENCH_VIDEO_BACKEND} / ${OPENPI_MEMBENCH_CAMERA_KEYS}"
echo "  checkpoint directory : ${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"

cd "${OPENPI_DIR}"

if [[ "${COMPUTE_NORM_STATS}" == "true" ]]; then
  norm_args=("--config-name=${CONFIG_NAME}" "--assets-base-dir=${ASSETS_BASE_DIR}")
  if [[ -n "${NORM_STATS_MAX_FRAMES}" ]]; then
    require_positive_integer NORM_STATS_MAX_FRAMES "${NORM_STATS_MAX_FRAMES}"
    norm_args+=("--max-frames=${NORM_STATS_MAX_FRAMES}")
  fi
  echo -n "Norm-stats command:"
  printf ' %q' env OPENPI_SKIP_IMAGE_DECODE=1 "OPENPI_NORM_STATS_BATCH_SIZE=${OPENPI_NORM_STATS_BATCH_SIZE}" uv run scripts/compute_norm_stats.py "${norm_args[@]}"
  echo
  if [[ "${DRY_RUN}" != "true" ]]; then
    OPENPI_SKIP_IMAGE_DECODE=1 OPENPI_NORM_STATS_BATCH_SIZE="${OPENPI_NORM_STATS_BATCH_SIZE}" \
      uv run scripts/compute_norm_stats.py "${norm_args[@]}"
  fi
fi

if [[ "${RUN_TRAINING}" == "true" ]]; then
  echo -n "Training command:"
  printf ' %q' uv run scripts/train.py "${train_args[@]}"
  echo
  if [[ "${DRY_RUN}" != "true" ]]; then
    uv run scripts/train.py "${train_args[@]}"
  fi
else
  echo "Training disabled (RUN_TRAINING=false)."
fi
