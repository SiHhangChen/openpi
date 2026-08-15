#!/usr/bin/env bash

set -euo pipefail

OPENPI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_NAME="${CONFIG_NAME:-pi05_membench_hs_wx_01_lora}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OPENPI_DIR}/checkpoints/pi05_membench_hs_wx_01_lora/pi05_lora_membench_hs_wx_01_60000/59999}"
PORT="${PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

export CUDA_VISIBLE_DEVICES
export UV_CACHE_DIR
export XLA_PYTHON_CLIENT_MEM_FRACTION

cd "${OPENPI_DIR}"

exec uv run --frozen scripts/serve_policy.py \
  --port="${PORT}" \
  policy:checkpoint \
  --policy.config="${CONFIG_NAME}" \
  --policy.dir="${CHECKPOINT_DIR}"
