#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_ROOT="${WORKSPACE}/policy/openpi"
PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
DATA_HOME="${DATA_HOME:-${WORKSPACE}/data/real_robot}"
SOURCE="${DATA_HOME}/wr07_lerobot_v3"
ANNOTATIONS="${DATA_HOME}/wr07_real_robot_v3"
DERIVED="${DATA_HOME}/wr07_lerobot_v3_subtasks"
CONFIG_NAME="pi05_real_robot_wr07_v3_subgoal"
EXP_NAME="${EXP_NAME:-wr07_real_robot_v3_subgoal_s10_bs48_30k}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
LOG_DIR="${OPENPI_ROOT}/logs/automation/${CONFIG_NAME}"
TRAIN_RUN="${OPENPI_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"

mkdir -p "${LOG_DIR}"
exec 9>"${LOG_DIR}/queue.lock"
flock -n 9 || { echo "A WR07 subgoal queue is already running" >&2; exit 1; }
exec >>"${LOG_DIR}/queue.log" 2>&1

log() { echo "$(date --iso-8601=seconds) $*"; }
fail() { log "ERROR: $*"; exit 1; }

[[ -x "${PYTHON_BIN}" ]] || fail "Missing Python environment: ${PYTHON_BIN}"
[[ -f "${SOURCE}/meta/info.json" ]] || fail "Missing source dataset: ${SOURCE}"
[[ -f "${ANNOTATIONS}/manifest.json" ]] || fail "Missing annotations: ${ANNOTATIONS}"
[[ -f "${OPENPI_ROOT}/assets/pi05_real_robot_wr07_v3/wr07_lerobot_v3/norm_stats.json" ]] || fail "Missing normalization stats"
[[ -f "${WORKSPACE}/data/pi05_base/params/_CHECKPOINT_METADATA" ]] || fail "Missing pi0.5 base checkpoint"
[[ ! -e "${TRAIN_RUN}" ]] || fail "Training run already exists: ${TRAIN_RUN}"

if [[ ! -d "${DERIVED}" ]]; then
  log "Building frame-aligned WR07 color-prompt sidecars"
  "${PYTHON_BIN}" "${OPENPI_ROOT}/scripts/build_real_robot_wr07_pi05_subtasks.py" \
    --source "${SOURCE}" --annotations "${ANNOTATIONS}" --output "${DERIVED}"
fi

export HF_LEROBOT_HOME="${DATA_HOME}"
export OPENPI_PRETRAIN_MODEL_DIR="${WORKSPACE}/data"
export OPENPI_DATA_HOME="${OPENPI_ROOT}/.cache/openpi"
export OPENPI_MEMBENCH_CAMERA_KEYS="observation.images.front,observation.images.left,observation.images.right"
export OPENPI_MEMBENCH_VIDEO_BACKEND="pyav"
export PYTHONPATH="${OPENPI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

cd "${OPENPI_ROOT}"
log "Running CPU data and configuration preflight"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OPENPI_SKIP_IMAGE_DECODE=1 \
  "${PYTHON_BIN}" - "${CONFIG_NAME}" "${DERIVED}" "${ANNOTATIONS}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

from openpi.training.config import get_config
from openpi.training.data_loader import create_torch_dataset

config_name, derived_arg, annotations_arg = sys.argv[1:]
derived, annotations = pathlib.Path(derived_arg), pathlib.Path(annotations_arg)
manifest = json.loads((derived / "subtask_manifest.json").read_text())
assert pathlib.Path(manifest["annotation_root"]) == annotations.resolve()
for episode, expected in manifest["annotation_sha256_by_episode"].items():
    path = annotations / "teacher_raw" / f"episode_{int(episode):06d}" / "annotation.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path

config = get_config(config_name)
assert config.model.pi05 and config.model.action_horizon == 50
assert config.model.action_dim == 32 and config.data.sample_stride == 10
assert config.data.base_config.prompt_from_subtask
assert not config.data.base_config.prompt_from_task
assert config.num_train_steps == 30_000 and config.batch_size == 48
assert config.data.assets.asset_id == "wr07_lerobot_v3"
assert pathlib.Path(config.weight_loader.params_path).resolve() == (
    pathlib.Path(os.environ["OPENPI_PRETRAIN_MODEL_DIR"]) / "pi05_base/params"
).resolve()
resolved = config.data.create(config.assets_dirs, config.model)
assert resolved.norm_stats["state"].q01.shape == (17,)
assert resolved.norm_stats["actions"].q01.shape == (17,)
source = create_torch_dataset(resolved, config.model.action_horizon, config.model)
assert len(source) == manifest["sample_anchors"] == 13_365
raw = source._dataset if hasattr(source, "_dataset") else source
assert set(raw.subtasks.values()) == set(manifest["prompt_by_color"].values())
assert set(map(int, np.unique(raw._subtask_indices))) == set(raw.subtasks)
for subtask_index in raw.subtasks:
    raw_index = next(i for i, value in enumerate(raw._subtask_indices[raw._sample_indices]) if value == subtask_index)
    sample = source[raw_index]
    assert sample["action"].shape == (50, 17)
    assert sample["prompt"] == raw.subtasks[subtask_index]
print(f"PREFLIGHT_OK samples={len(source)} prompts={len(raw.subtasks)} horizon=50 action_dim=17")
PY

IFS=',' read -r -a GPU_INDICES <<<"${GPU_LIST}"
[[ ${#GPU_INDICES[@]} -gt 0 ]] || fail "GPU_LIST is empty"
while true; do
  busy=0
  for gpu in "${GPU_INDICES[@]}"; do
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)" || fail "Cannot query GPU ${gpu}"
    if [[ -n "${pids//[[:space:]]/}" ]]; then busy=1; fi
  done
  if [[ ${busy} == 0 ]]; then break; fi
  log "Waiting for GPUs ${GPU_LIST}; checking again in ${WAIT_SECONDS}s"
  sleep "${WAIT_SECONDS}"
done

[[ ! -e "${TRAIN_RUN}" ]] || fail "Training run appeared while waiting: ${TRAIN_RUN}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
log "Starting ${CONFIG_NAME}/${EXP_NAME} on GPUs ${GPU_LIST}"
exec "${PYTHON_BIN}" -u scripts/train.py "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --seed=42 \
  --batch-size=48 \
  --num-train-steps=30000 \
  --fsdp-devices=1 \
  --num-workers=32 \
  --log-interval=100 \
  --save-interval=10000 \
  --keep-period=10000 \
  --no-wandb-enabled
