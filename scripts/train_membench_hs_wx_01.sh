#!/usr/bin/env bash

set -euo pipefail

OPENPI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${OPENPI_DIR}/../.." && pwd)"

CONFIG_NAME="${CONFIG_NAME:-pi0_membench_hs_wx_01}"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}}"
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR:-${WORKSPACE_DIR}/lerobot_data}"
OPENPI_DATA_HOME_DIR="${OPENPI_DATA_HOME_DIR:-${WORKSPACE_DIR}/.openpi-cache}"
OPENPI_PRETRAIN_MODEL_DIR="${OPENPI_PRETRAIN_MODEL_DIR:-${WORKSPACE_DIR}/pretrain_model}"
DATASET_REPO_ID="${DATASET_REPO_ID:-membench_hs_wx_01}"
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS:-true}"
RUN_TRAINING="${RUN_TRAINING:-true}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
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

export HF_LEROBOT_HOME="${LEROBOT_HOME_DIR}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME_DIR}"
export OPENPI_PRETRAIN_MODEL_DIR
unset LEROBOT_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION

cd "${OPENPI_DIR}"

if [[ "${COMPUTE_NORM_STATS}" == "true" ]]; then
  OPENPI_SKIP_IMAGE_DECODE=1 OPENPI_NORM_STATS_BATCH_SIZE="${OPENPI_NORM_STATS_BATCH_SIZE}" \
    uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
fi

if [[ "${RUN_TRAINING}" == "true" ]]; then
  uv run scripts/train.py "${CONFIG_NAME}" --exp-name="${EXP_NAME}" "$@"
fi
