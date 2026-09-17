#!/usr/bin/env bash
set -euo pipefail

# WR07 three-view full fine-tune, stride 10, 30k steps on GPUs 0-3.
# Fresh run from pi05_base (replaces the aborted stride-20 run).

OPENPI_ROOT="${OPENPI_ROOT:-/data1/workspace/chensihang/membench/policy/openpi}"
PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
DATASET_HOME="${HF_LEROBOT_HOME:-/data1/shared_workspace/chensihang/dataset/membench}"
PRETRAIN_ROOT="${OPENPI_PRETRAIN_MODEL_DIR:-/data1/shared_workspace/tangzhipeng/ckpts/openpi/openpi_assets}"

CONFIG_NAME="pi05_membench_wr07_3view_full_stride10_30k"
EXP_NAME="wr07_3view_full_s10_bs48_30k_g0123"
DATASET_ROOT="${DATASET_HOME}/wr07_200seeds_v061"
NORM_STATS="${OPENPI_ROOT}/assets/pi05_membench_wr07_3view_full_stride20/wr07_200seeds_v061/norm_stats.json"
NORM_STATS_SHA256="cf987f08d0fd83b81c197a15488172952344b43fb54f575079152b6e6e744e39"
PI05_PARAMS="${PRETRAIN_ROOT}/pi05_base/params"
TRAIN_RUN="${OPENPI_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"

AUTOMATION_DIR="${OPENPI_ROOT}/logs/automation/pi05_wr07_3view_full_stride10_30k"
LOCK_FILE="${AUTOMATION_DIR}/queue.lock"
GPU_LIST=(0 1 2 3)
WAIT_SECONDS="${WAIT_SECONDS:-30}"

mkdir -p "${AUTOMATION_DIR}" "${OPENPI_ROOT}/.cache/openpi"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) another watcher already holds ${LOCK_FILE}"
  exit 1
fi

log() {
  echo "$(date --iso-8601=seconds) $*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

gpu_is_busy() {
  local gpu="$1"
  local pids
  if ! pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
    return 2
  fi
  grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$' <<<"${pids}"
}

wait_for_gpus() {
  local busy unavailable gpu rc
  while true; do
    busy=0
    unavailable=0
    for gpu in "${GPU_LIST[@]}"; do
      if gpu_is_busy "${gpu}"; then
        busy=1
      else
        rc=$?
        if [[ "${rc}" == 2 ]]; then
          unavailable=1
        fi
      fi
    done
    if [[ "${busy}" == 0 && "${unavailable}" == 0 ]]; then
      log "GPUs ${GPU_LIST[*]} are free"
      return
    fi
    if [[ "${unavailable}" == 1 ]]; then
      log "GPU status unavailable; checking again in ${WAIT_SECONDS}s"
    else
      log "waiting for GPUs ${GPU_LIST[*]}; checking again in ${WAIT_SECONDS}s"
    fi
    sleep "${WAIT_SECONDS}"
  done
}

[[ -x "${PYTHON_BIN}" ]] || fail "missing Python executable: ${PYTHON_BIN}"
[[ -f "${DATASET_ROOT}/meta/info.json" ]] || fail "missing WR07 dataset: ${DATASET_ROOT}"
[[ -f "${NORM_STATS}" ]] || fail "missing norm stats: ${NORM_STATS}"
[[ -d "${PI05_PARAMS}" ]] || fail "missing pi0.5 base params: ${PI05_PARAMS}"
[[ ! -e "${TRAIN_RUN}" ]] || fail "training directory already exists: ${TRAIN_RUN}"

cd "${OPENPI_ROOT}"
export PYTHONPATH="${OPENPI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_LEROBOT_HOME="${DATASET_HOME}"
export OPENPI_PRETRAIN_MODEL_DIR="${PRETRAIN_ROOT}"
export OPENPI_DATA_HOME="${OPENPI_ROOT}/.cache/openpi"
export OPENPI_MEMBENCH_CAMERA_KEYS="observation.images.robot0_agentview_left,observation.images.robot0_agentview_right,observation.images.robot0_eye_in_hand"
export OPENPI_MEMBENCH_VIDEO_BACKEND="torchcodec"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

log "running CPU-only WR07 stride-10 contract and data preflight"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OPENPI_SKIP_IMAGE_DECODE=1 \
  "${PYTHON_BIN}" - "${CONFIG_NAME}" "${NORM_STATS}" "${NORM_STATS_SHA256}" "${PI05_PARAMS}" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

import jax

jax.config.update("jax_platforms", "cpu")

from openpi.training.config import get_config
from openpi.training.data_loader import create_torch_dataset

config_name, stats_arg, expected_sha256, params_arg = sys.argv[1:]
config = get_config(config_name)

assert (
    config.model.pi05,
    config.model.discrete_state_input,
    config.model.action_dim,
    config.model.action_horizon,
    config.model.max_token_len,
) == (True, True, 32, 50, 200)
assert (config.model.paligemma_variant, config.model.action_expert_variant) == ("gemma_2b", "gemma_300m")
assert config.data.repo_id == "wr07_200seeds_v061"
assert config.data.state_keep_dim == 30
assert config.data.sample_stride == 10
assert config.data.base_config.prompt_from_task
assert not config.data.base_config.prompt_from_subtask
assert (config.batch_size, config.num_train_steps, config.seed) == (48, 30_000, 42)
assert (config.num_workers, config.fsdp_devices, config.ema_decay) == (32, 1, None)
assert (config.log_interval, config.save_interval, config.keep_period) == (100, 5_000, 5_000)
assert (
    config.lr_schedule.warmup_steps,
    config.lr_schedule.peak_lr,
    config.lr_schedule.decay_steps,
    config.lr_schedule.decay_lr,
) == (1_000, 2.5e-5, 30_000, 2.5e-6)
assert pathlib.Path(config.weight_loader.params_path) == pathlib.Path(params_arg)

cameras = tuple(config.data.repack_transforms.inputs[0].structure["images"])
assert cameras == ("agentview_left", "agentview_right", "eye_in_hand")
data_config = config.data.create(config.assets_dirs, config.model)
assert data_config.sample_stride == 10
assert data_config.use_quantile_norm

stats_path = pathlib.Path(stats_arg)
assert hashlib.sha256(stats_path.read_bytes()).hexdigest() == expected_sha256
stats = json.loads(stats_path.read_bytes())["norm_stats"]
assert len(stats["state"]["mean"]) == 30
assert len(stats["actions"]["mean"]) == 13
assert all(math.isfinite(value) for item in stats.values() for values in item.values() for value in values)

source = create_torch_dataset(data_config, config.model.action_horizon, config.model)
assert len(source) == 14_763

print(
    f"PREFLIGHT_OK config={config.name} stride={data_config.sample_stride} samples={len(source)} "
    f"cameras={cameras} stats_sha256={expected_sha256}"
)
PY

wait_for_gpus
[[ ! -e "${TRAIN_RUN}" ]] || fail "training directory appeared during preflight: ${TRAIN_RUN}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
log "starting ${CONFIG_NAME}/${EXP_NAME} on GPUs 0,1,2,3 at commit $(git rev-parse --short HEAD)"
log "global batch 48, 30000 updates, anchor stride 10, contiguous horizon 50, data parallel 4, FSDP 1"
exec "${PYTHON_BIN}" -u scripts/train.py "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --seed=42 \
  --batch-size=48 \
  --num-train-steps=30000 \
  --fsdp-devices=1 \
  --num-workers=32 \
  --log-interval=100 \
  --save-interval=5000 \
  --keep-period=5000 \
  --no-wandb-enabled
