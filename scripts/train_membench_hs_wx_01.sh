#!/usr/bin/env bash

set -euo pipefail

OPENPI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${OPENPI_DIR}/../.." && pwd)"

CONFIG_NAME="${CONFIG_NAME:-pi05_membench_hs_wx_01_lora}"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}}"
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR:-${WORKSPACE_DIR}/lerobot_data}"
OPENPI_DATA_HOME_DIR="${OPENPI_DATA_HOME_DIR:-${WORKSPACE_DIR}/.openpi-cache}"
OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR:-${WORKSPACE_DIR}/pretrain_model}"
DATASET_REPO_ID="${DATASET_REPO_ID:-membench_hs_wx_01}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
BATCH_SIZE="${BATCH_SIZE:-32}"
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS:-true}"
RUN_TRAINING="${RUN_TRAINING:-true}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
OPENPI_NORM_STATS_BATCH_SIZE="${OPENPI_NORM_STATS_BATCH_SIZE:-512}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not available. Install uv first, then run: cd ${OPENPI_DIR} && uv sync" >&2
  exit 1
fi

if [[ ! -d "${LEROBOT_HOME_DIR}/${DATASET_REPO_ID}" ]]; then
  echo "Dataset not found: ${LEROBOT_HOME_DIR}/${DATASET_REPO_ID}" >&2
  echo "Set LEROBOT_HOME_DIR to the parent directory that contains ${DATASET_REPO_ID}." >&2
  exit 1
fi

if [[ ! -f "${LEROBOT_HOME_DIR}/${DATASET_REPO_ID}/meta/info.json" ]]; then
  echo "LeRobot metadata not found: ${LEROBOT_HOME_DIR}/${DATASET_REPO_ID}/meta/info.json" >&2
  exit 1
fi

if [[ ! -d "${OPENPI_PRETRAIN_MODEL_DIR}/pi05_base/params" ]]; then
  echo "pi05 pretrained params not found: ${OPENPI_PRETRAIN_MODEL_DIR}/pi05_base/params" >&2
  echo "Set OPENPI_PRETRAIN_MODEL_DIR to the parent directory that contains pi05_base." >&2
  exit 1
fi

export HF_LEROBOT_HOME="${LEROBOT_HOME_DIR}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME_DIR}"
export OPENPI_PRETRAIN_MODEL_DIR
export CUDA_VISIBLE_DEVICES
unset LEROBOT_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION

cd "${OPENPI_DIR}"

echo "Starting OpenPI MemBench training with:"
echo "  CONFIG_NAME=${CONFIG_NAME}"
echo "  EXP_NAME=${EXP_NAME}"
echo "  DATASET=${LEROBOT_HOME_DIR}/${DATASET_REPO_ID}"
echo "  HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "  OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"
echo "  OPENPI_PRETRAIN_MODEL_DIR=${OPENPI_PRETRAIN_MODEL_DIR}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  COMPUTE_NORM_STATS=${COMPUTE_NORM_STATS}"
echo "  RUN_TRAINING=${RUN_TRAINING}"

if [[ "${COMPUTE_NORM_STATS}" == "true" ]]; then
  OPENPI_SKIP_IMAGE_DECODE=1 OPENPI_NORM_STATS_BATCH_SIZE="${OPENPI_NORM_STATS_BATCH_SIZE}" \
    uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
fi

if [[ "${RUN_TRAINING}" == "true" ]]; then
  uv run scripts/train.py "${CONFIG_NAME}" --exp-name="${EXP_NAME}" --batch-size="${BATCH_SIZE}" "$@"
fi
